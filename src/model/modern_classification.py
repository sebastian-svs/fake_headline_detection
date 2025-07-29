import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoConfig,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix
from typing import List, Dict, Any, Tuple, Optional
import os
import json
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class ModernClassificationDataset(Dataset):
    """Dataset class for classification with external features"""
    
    def __init__(self, texts_a: List[str], texts_b: List[str], 
                 labels: List[int], features: List[List[float]], 
                 tokenizer, max_length: int = 512):
        self.texts_a = texts_a
        self.texts_b = texts_b
        self.labels = labels
        self.features = features
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts_a)
    
    def __getitem__(self, idx):
        text_a = str(self.texts_a[idx])
        text_b = str(self.texts_b[idx]) if self.texts_b else None
        
        # Tokenize
        if text_b:
            encoding = self.tokenizer(
                text_a, text_b,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
        else:
            encoding = self.tokenizer(
                text_a,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
            'external_features': torch.tensor(self.features[idx] if self.features[idx] else [0.0], dtype=torch.float)
        }


class DeBERTaWithExternalFeatures(nn.Module):
    """DeBERTa/RoBERTa model with external feature integration - FIXED VERSION"""
    
    def __init__(self, model_name: str, num_labels: int, feature_dim: int = 0):
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=self.config
        )
        
        self.feature_dim = feature_dim
        
        if feature_dim > 0:
            # Add feature integration layers
            hidden_size = self.config.hidden_size
            self.feature_projection = nn.Linear(feature_dim, hidden_size)
            self.feature_dropout = nn.Dropout(0.1)
            
            # Replace the classifier
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, num_labels)
            )
    
    def forward(self, input_ids, attention_mask, external_features=None, labels=None):
        # Get backbone outputs (works for DeBERTa, RoBERTa, BERT, etc.)
        if hasattr(self.backbone, 'deberta'):
            # DeBERTa
            outputs = self.backbone.deberta(input_ids=input_ids, attention_mask=attention_mask)
        elif hasattr(self.backbone, 'roberta'):
            # RoBERTa
            outputs = self.backbone.roberta(input_ids=input_ids, attention_mask=attention_mask)
        elif hasattr(self.backbone, 'bert'):
            # BERT
            outputs = self.backbone.bert(input_ids=input_ids, attention_mask=attention_mask)
        else:
            # Fallback - try direct call
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            outputs = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs.last_hidden_state
        
        if hasattr(outputs, 'last_hidden_state'):
            sequence_output = outputs.last_hidden_state
        else:
            sequence_output = outputs
        
        # Use [CLS] token representation
        cls_output = sequence_output[:, 0, :]  # [batch_size, hidden_size]
        
        if self.feature_dim > 0 and external_features is not None:
            # Project external features
            feature_output = self.feature_projection(external_features)
            feature_output = self.feature_dropout(feature_output)
            
            # Combine CLS output with features (element-wise multiplication)
            combined_output = cls_output * feature_output
        else:
            combined_output = cls_output
        
        # Apply classifier
        logits = self.backbone.classifier(combined_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
        }
    
    # ✅ FIX: Füge save_pretrained und from_pretrained Methoden hinzu
    def save_pretrained(self, save_directory):
        """Save the model to a directory"""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the backbone model
        self.backbone.save_pretrained(save_directory)
        
        # Save additional custom components
        if self.feature_dim > 0:
            custom_state = {
                'feature_projection': self.feature_projection.state_dict(),
                'feature_dropout': self.feature_dropout.state_dict(),
                'feature_dim': self.feature_dim,
                'config': self.config
            }
            torch.save(custom_state, os.path.join(save_directory, 'custom_components.pth'))
        
        # Save model config
        model_config = {
            'model_class': 'DeBERTaWithExternalFeatures',
            'feature_dim': self.feature_dim,
            'num_labels': self.config.num_labels
        }
        
        with open(os.path.join(save_directory, 'model_config.json'), 'w') as f:
            json.dump(model_config, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, model_path, feature_dim=0):
        """Load a saved model from directory"""
        
        # Load model config
        config_path = os.path.join(model_path, 'model_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                model_config = json.load(f)
                feature_dim = model_config.get('feature_dim', feature_dim)
                num_labels = model_config.get('num_labels', 4)
        else:
            num_labels = 4
        
        # Create new instance
        model = cls(model_path, num_labels, feature_dim)
        
        # Load custom components if they exist
        custom_path = os.path.join(model_path, 'custom_components.pth')
        if os.path.exists(custom_path) and feature_dim > 0:
            custom_state = torch.load(custom_path, map_location='cpu')
            model.feature_projection.load_state_dict(custom_state['feature_projection'])
            model.feature_dropout.load_state_dict(custom_state['feature_dropout'])
        
        return model


class OutClassificationModel:
    """Modern classification model compatible with the original interface"""
    
    def __init__(self, model_type: str, model_name: str, num_labels: int = None, 
                 weight=None, args=None, use_cuda=True, cuda_device=-1, **kwargs):
        
        self.model_type = model_type
        self.model_name = model_name
        self.num_labels = num_labels or 4
        
        # Extract feature dimension from args
        self.feature_dim = 0
        if args and 'value_head' in args:
            self.feature_dim = args['value_head']
        
        # Set device
        if use_cuda and torch.cuda.is_available():
            if cuda_device == -1:
                self.device = torch.device('cuda')
            else:
                self.device = torch.device(f'cuda:{cuda_device}')
        else:
            self.device = torch.device('cpu')
        
        # Default arguments
        self.args = {
            'learning_rate': 8e-6,              # ✅ Optimiert (war 1e-5)
            'num_train_epochs': 6,              # ✅ Mehr Epochen (war 3)
            'train_batch_size': 8,              # ✅ Bewährt - funktioniert
            'eval_batch_size': 16,              # ✅ Etwas größer für Speed
            'max_seq_length': 512,
            'warmup_ratio': 0.05,               # ✅ Weniger Warmup (war 0.1)
            'weight_decay': 0.01,
            'logging_steps': 25,                # ✅ Mehr Logs (war 100)
            'eval_steps': 0,                    # KEINE Zwischenevaluierung
            'save_steps': 0,                    # KEINE Steps-basierte Speicherung  
            'save_model_every_epoch': True,     # Nach jeder Epoche speichern
            'fp16': True,
            'overwrite_output_dir': True,
            'reprocess_input_data': True,
            'value_head': 0,
            'seed': 42,
        }
        if args:
            self.args.update(args)
        
        # Map model types to actual model names
        model_name_mapping = {
            'deberta-v3': 'microsoft/deberta-v3-large',
            'roberta': 'roberta-large',
            'bert': 'bert-large-uncased',
        }
        
        if model_type in model_name_mapping and not os.path.exists(model_name):
            actual_model_name = model_name_mapping[model_type]
        else:
            actual_model_name = model_name
        
        # Initialize tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(actual_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.error(f"Could not load tokenizer for {actual_model_name}: {e}")
            raise
        
        # Initialize model
        try:
            self.model = DeBERTaWithExternalFeatures(
                actual_model_name, self.num_labels, self.feature_dim
            ).to(self.device)
        except Exception as e:
            logger.error(f"Could not load model {actual_model_name}: {e}")
            raise
        
        self.results = {}
    
    def prepare_dataset(self, df: pd.DataFrame, is_training: bool = True) -> ModernClassificationDataset:
        """Prepare dataset from DataFrame"""
        
        texts_a = df['text_a'].tolist()
        texts_b = df['text_b'].tolist() if 'text_b' in df.columns else [None] * len(texts_a)
        
        if is_training and 'labels' in df.columns:
            labels = df['labels'].tolist()
        else:
            labels = [0] * len(texts_a)  # Dummy labels for prediction
        
        if 'feature' in df.columns and self.feature_dim > 0:
            features = df['feature'].tolist()
            # Handle cases where features might be scalars, arrays, or None
            processed_features = []
            for feat in features:
                if feat is None:
                    processed_features.append([0.0] * self.feature_dim)
                elif isinstance(feat, (int, float)) and feat == 0:
                    processed_features.append([0.0] * self.feature_dim)
                elif isinstance(feat, (int, float)):
                    processed_features.append([float(feat)] * self.feature_dim)
                elif isinstance(feat, (list, tuple)):
                    processed_features.append(list(feat))
                elif hasattr(feat, '__iter__'):  # numpy array or similar
                    processed_features.append(list(feat))
                else:
                    processed_features.append([0.0] * self.feature_dim)
            features = processed_features
        else:
            features = [[0.0] * max(1, self.feature_dim)] * len(texts_a)
        
        return ModernClassificationDataset(
            texts_a, texts_b, labels, features, 
            self.tokenizer, self.args['max_seq_length']
        )
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average='macro')
        mcc = matthews_corrcoef(labels, predictions)
        
        # Calculate confusion matrix elements for binary case
        if self.num_labels == 2:
            tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
            return {
                'acc': accuracy,
                'f1_macro': f1_macro,
                'mcc': mcc,
                'tp': tp,
                'tn': tn,
                'fp': fp,
                'fn': fn
            }
        else:
            return {
                'acc': accuracy,
                'f1_macro': f1_macro,
                'mcc': mcc
            }
    
    def train_model(self, train_df: pd.DataFrame, eval_df: Optional[pd.DataFrame] = None,
                output_dir: str = './outputs', **kwargs):
        """Train the model - CLEANED VERSION"""
        
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_df, is_training=True)
        eval_dataset = self.prepare_dataset(eval_df, is_training=True) if eval_df is not None else None
        
        # ✅ CLEANED TrainingArguments - KEINE Zwischenspeicherungen
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=self.args['learning_rate'],
            num_train_epochs=self.args['num_train_epochs'],
            per_device_train_batch_size=self.args['train_batch_size'],
            per_device_eval_batch_size=self.args['eval_batch_size'],
            warmup_ratio=self.args['warmup_ratio'],
            weight_decay=self.args['weight_decay'],
            logging_steps=self.args['logging_steps'],
            
            # ✅ CRITICAL: Disable ALL intermediate saving
            eval_strategy='no',                    # Keine Evaluation während Training
            save_strategy='epoch',                 # Nur am Ende jeder Epoche speichern
            save_steps=999999,                     # Extrem hohe Zahl = praktisch nie
            save_total_limit=1,                    # Nur 1 Checkpoint behalten
            load_best_model_at_end=False,          # Nicht automatisch bestes laden
            
            # ✅ Logging und Performance
            report_to=[],                          # Keine externe Reports
            overwrite_output_dir=self.args['overwrite_output_dir'],
            fp16=self.args['fp16'],
            run_name=f"deberta_training_{hash(output_dir) % 10000}",
            disable_tqdm=False,
            logging_first_step=True,
            
            # ✅ Cleanup Settings
            dataloader_pin_memory=False,           # Weniger Memory Usage
            remove_unused_columns=True,            # Cleanup
            prediction_loss_only=True,             # Weniger Output
        )
        
        # Custom data collator to handle external features
        def data_collator(features):
            batch = {}
            batch['input_ids'] = torch.stack([f['input_ids'] for f in features])
            batch['attention_mask'] = torch.stack([f['attention_mask'] for f in features])
            batch['labels'] = torch.stack([f['labels'] for f in features])
            
            if self.feature_dim > 0:
                batch['external_features'] = torch.stack([f['external_features'] for f in features])
            
            return batch
        
        # ✅ CLEAN Epoch Progress Callback (weniger Output)
        from transformers import TrainerCallback
        
        class CleanEpochProgressCallback(TrainerCallback):
            def __init__(self):
                self.epoch_start_time = None
                self.training_start_time = None
                self.current_epoch = 0
                self.total_epochs = 0
                
            def on_train_begin(self, args, state, control, **kwargs):
                import time
                from datetime import datetime
                self.training_start_time = time.time()
                self.total_epochs = args.num_train_epochs
                
                print(f"\n🚀 TRAINING START")
                print(f"📊 Total Epochs: {self.total_epochs}")
                print(f"📊 Total Steps: {state.max_steps}")
                print(f"⏰ Started at: {datetime.now().strftime('%H:%M:%S')}")
                print("=" * 60)
            
            def on_epoch_begin(self, args, state, control, **kwargs):
                import time
                from datetime import datetime
                self.current_epoch = int(state.epoch) + 1
                self.epoch_start_time = time.time()
                
                print(f"\n🔥 EPOCH {self.current_epoch}/{self.total_epochs} STARTING")
                print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
                print("-" * 40)
            
            def on_epoch_end(self, args, state, control, **kwargs):
                import time
                from datetime import timedelta
                if self.epoch_start_time:
                    epoch_duration = time.time() - self.epoch_start_time
                    
                    print(f"\n✅ EPOCH {self.current_epoch}/{self.total_epochs} COMPLETED")
                    print(f"⏰ Duration: {timedelta(seconds=int(epoch_duration))}")
                    print("=" * 60)
            
            def on_train_end(self, args, state, control, **kwargs):
                import time
                from datetime import timedelta
                if self.training_start_time:
                    total_duration = time.time() - self.training_start_time
                    print(f"\n🎉 TRAINING COMPLETED!")
                    print(f"⏰ Total Duration: {timedelta(seconds=int(total_duration))}")
                    print("=" * 60)
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics if eval_dataset else None,
            data_collator=data_collator,
            callbacks=[CleanEpochProgressCallback()],
        )
        
        # Train
        trainer.train()
        
        # ✅ ONLY save at the very end
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save additional config
        with open(os.path.join(output_dir, 'model_config.json'), 'w') as f:
            json.dump({
                'model_type': self.model_type,
                'model_name': self.model_name,
                'num_labels': self.num_labels,
                'feature_dim': self.feature_dim,
                'args': self.args
            }, f, indent=2)
        
        # Custom data collator to handle external features
        def data_collator(features):
            batch = {}
            batch['input_ids'] = torch.stack([f['input_ids'] for f in features])
            batch['attention_mask'] = torch.stack([f['attention_mask'] for f in features])
            batch['labels'] = torch.stack([f['labels'] for f in features])
            
            if self.feature_dim > 0:
                batch['external_features'] = torch.stack([f['external_features'] for f in features])
            
            return batch
        
        # Epoch Progress Callback
        from transformers import TrainerCallback
        
        class EpochProgressCallback(TrainerCallback):  # 🎯 Erbt von TrainerCallback
            def __init__(self):
                self.epoch_start_time = None
                self.training_start_time = None
                self.current_epoch = 0
                self.total_epochs = 0
                
            def on_train_begin(self, args, state, control, **kwargs):
                import time
                from datetime import datetime
                self.training_start_time = time.time()
                self.total_epochs = args.num_train_epochs
                steps_per_epoch = state.max_steps // self.total_epochs if state.max_steps > 0 else len(kwargs.get('train_dataloader', []))
                
                print(f"\n🚀 TRAINING START")
                print(f"📊 Total Epochs: {self.total_epochs}")
                print(f"📊 Steps per Epoch: ~{steps_per_epoch}")
                print(f"📊 Total Steps: {state.max_steps}")
                print(f"⏰ Started at: {datetime.now().strftime('%H:%M:%S')}")
                print("=" * 60)
            
            def on_epoch_begin(self, args, state, control, **kwargs):
                import time
                from datetime import datetime, timedelta
                self.current_epoch = int(state.epoch) + 1
                self.epoch_start_time = time.time()
                
                print(f"\n🔥 EPOCH {self.current_epoch}/{self.total_epochs} STARTING")
                print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
                
                if self.current_epoch > 1 and self.training_start_time:
                    elapsed = time.time() - self.training_start_time
                    avg_time_per_epoch = elapsed / (self.current_epoch - 1)
                    remaining_epochs = self.total_epochs - (self.current_epoch - 1)
                    estimated_remaining = avg_time_per_epoch * remaining_epochs
                    
                    print(f"⏱️  Elapsed: {timedelta(seconds=int(elapsed))}")
                    print(f"⏱️  ETA: {timedelta(seconds=int(estimated_remaining))}")
                print("-" * 40)
            
            def on_epoch_end(self, args, state, control, **kwargs):
                import time
                from datetime import timedelta
                if self.epoch_start_time:
                    epoch_duration = time.time() - self.epoch_start_time
                    
                    print(f"\n✅ EPOCH {self.current_epoch}/{self.total_epochs} COMPLETED")
                    print(f"⏰ Duration: {timedelta(seconds=int(epoch_duration))}")
                    
                    if self.current_epoch < self.total_epochs:
                        remaining_epochs = self.total_epochs - self.current_epoch
                        estimated_remaining = epoch_duration * remaining_epochs
                        print(f"🔮 Estimated remaining: {timedelta(seconds=int(estimated_remaining))}")
                    
                    print("=" * 60)
            
            def on_train_end(self, args, state, control, **kwargs):
                import time
                from datetime import timedelta
                if self.training_start_time:
                    total_duration = time.time() - self.training_start_time
                    print(f"\n🎉 TRAINING COMPLETED!")
                    print(f"⏰ Total Duration: {timedelta(seconds=int(total_duration))}")
                    print(f"📊 Average per Epoch: {timedelta(seconds=int(total_duration / self.total_epochs))}")
                    print("=" * 60)
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics if eval_dataset else None,
            data_collator=data_collator,
            callbacks=[EpochProgressCallback()],  # 🎯 EPOCH TRACKING!
        )
        
        # Train
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save additional config
        with open(os.path.join(output_dir, 'model_config.json'), 'w') as f:
            json.dump({
                'model_type': self.model_type,
                'model_name': self.model_name,
                'num_labels': self.num_labels,
                'feature_dim': self.feature_dim,
                'args': self.args
            }, f, indent=2)
    
    def predict(self, test_data: List) -> Tuple[List[int], np.ndarray]:
        """Make predictions on test data"""
        
        # Handle different input formats
        if isinstance(test_data[0], list) and len(test_data[0]) >= 3:
            # Format: [text_a, text_b, features]
            texts_a = [item[0] for item in test_data]
            texts_b = [item[1] for item in test_data]
            features = [item[2] for item in test_data]
            
            df = pd.DataFrame({
                'text_a': texts_a,
                'text_b': texts_b,
                'feature': features
            })
        elif isinstance(test_data[0], list) and len(test_data[0]) == 2:
            # Format: [text_a, text_b]
            texts_a = [item[0] for item in test_data]
            texts_b = [item[1] for item in test_data]
            
            df = pd.DataFrame({
                'text_a': texts_a,
                'text_b': texts_b
            })
        else:
            # Simple text input
            df = pd.DataFrame({'text_a': test_data})
        
        dataset = self.prepare_dataset(df, is_training=False)
        
        # Make predictions
        self.model.eval()
        predictions = []
        model_outputs = []
        
        with torch.no_grad():
            for i in range(len(dataset)):
                batch = dataset[i]
                
                # Move to device
                input_ids = batch['input_ids'].unsqueeze(0).to(self.device)
                attention_mask = batch['attention_mask'].unsqueeze(0).to(self.device)
                external_features = batch['external_features'].unsqueeze(0).to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    external_features=external_features if self.feature_dim > 0 else None
                )
                
                logits = outputs['logits']
                model_outputs.append(logits.cpu().numpy())
                predictions.append(torch.argmax(logits, dim=-1).cpu().numpy()[0])
        
        return predictions, np.vstack(model_outputs)
    
    def eval_model(self, eval_df: pd.DataFrame, **kwargs) -> Tuple[Dict, np.ndarray, List]:
        """Evaluate the model"""
        
        # Prepare test input from DataFrame
        test_input = []
        for _, row in eval_df.iterrows():
            if 'feature' in row and row['feature'] is not None:
                test_input.append([row['text_a'], row['text_b'], row['feature']])
            else:
                test_input.append([row['text_a'], row['text_b']])
        
        # Make predictions
        predictions, model_outputs = self.predict(test_input)
        
        # Calculate metrics
        if 'labels' in eval_df.columns:
            true_labels = eval_df['labels'].tolist()
            accuracy = accuracy_score(true_labels, predictions)
            f1_macro = f1_score(true_labels, predictions, average='macro')
            mcc = matthews_corrcoef(true_labels, predictions)
            
            results = {
                'acc': accuracy,
                'f1_macro': f1_macro,
                'mcc': mcc
            }
            
            # Add custom metrics from kwargs
            for metric_name, metric_func in kwargs.items():
                if callable(metric_func):
                    try:
                        results[metric_name] = metric_func(true_labels, predictions)
                    except:
                        pass
            
            # Calculate confusion matrix elements for binary case
            if self.num_labels == 2:
                try:
                    tn, fp, fn, tp = confusion_matrix(true_labels, predictions, labels=[0, 1]).ravel()
                    results.update({'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn})
                except:
                    pass
            
            # Find wrong predictions
            wrong_predictions = []
            for i, (true_label, pred_label) in enumerate(zip(true_labels, predictions)):
                if true_label != pred_label:
                    wrong_predictions.append(i)
        else:
            results = {}
            wrong_predictions = []
        
        self.results.update(results)
        return results, model_outputs, wrong_predictions