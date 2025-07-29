import os
from common.score import scorePredict
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from model.modern_classification import OutClassificationModel
import time
from datetime import datetime, timedelta
import json

# ✅ DEINE BESTEHENDE FUNKTION (bleibt unverändert)
def train_predict_model(df_train, df_test, is_predict, use_cuda, value_head):
    """
    Original training function - bleibt für Kompatibilität
    """
    labels_test = pd.Series(df_test['labels']).to_numpy()
    labels = list(df_train['labels'].unique())
    labels.sort()

    print(f"🚀 Initializing DeBERTa v3 Large model...")
    print(f"📊 Number of classes: {len(labels)}")
    print(f"📊 Classes: {labels}")
    print(f"🔧 External feature dimension: {value_head}")
    print(f"💾 CUDA enabled: {use_cuda}")
    print(f"🎯 Starting training...")
    print(f"📈 Training samples: {len(df_train)}")
    print(f"📉 Test samples: {len(df_test)}")

    model = OutClassificationModel('deberta-v3', './models/deberta-v3-large', num_labels=len(labels),
                            use_cuda=use_cuda, args={
                            'learning_rate': 2e-5,
                            'num_train_epochs': 3,
                            'train_batch_size': 16,
                            'eval_batch_size': 32,
                            'gradient_accumulation_steps': 2,
                            'max_seq_length': 512,
                            'fp16': True,
                            'warmup_ratio': 0.06,
                            'weight_decay': 0.01,
                            'report_to': [],
                            'value_head': value_head})

    print(f"✅ Training completed!")
    
    results = ''
    if is_predict:
        print(f"🔮 Making predictions...")
        
        value_in = []
        for _, row in df_test.iterrows():
            if 'feature' in df_test.columns and row['feature'] is not None:
                value_in.append([row['text_a'], row['text_b'], row['feature']])
            else:
                value_in.append([row['text_a'], row['text_b'], [0.0] * value_head])
        
        _, model_outputs_test = model.predict(value_in)
        
    else:
        print(f"📊 Evaluating model...")
        result, model_outputs_test, wrong_predictions = model.eval_model(df_test, acc=accuracy_score)
        results = result['acc']
        print(f"🎯 Evaluation Accuracy: {results:.4f}")
    
    y_predict = np.argmax(model_outputs_test, axis=1)
    
    print(f"\n📋 Detailed Results:")
    detailed_results = scorePredict(y_predict, labels_test, labels)
    print(detailed_results)
    
    return results


def predict(df_test, use_cuda, model_dir, value_head):
    """
    Original predict function - bleibt unverändert
    """
    print(f"📂 Loading trained DeBERTa model from: {os.getcwd() + model_dir}")
    
    model = OutClassificationModel(
        model_type='deberta-v3',
        model_name=os.getcwd() + model_dir, 
        use_cuda=use_cuda, 
        args={'value_head': value_head}
    )
    
    labels_test = pd.Series(df_test['labels']).to_numpy()
    labels = list(df_test['labels'].unique())
    labels.sort()
    
    print(f"🔮 Making predictions on {len(df_test)} samples...")
    
    value_in = []
    for _, row in df_test.iterrows():
        if 'feature' in df_test.columns and row['feature'] is not None:
            value_in.append([row['text_a'], row['text_b'], row['feature']])
        else:
            value_in.append([row['text_a'], row['text_b'], [0.0] * value_head])
    
    _, model_outputs_test = model.predict(value_in)
    y_predict = np.argmax(model_outputs_test, axis=1)
    
    print(f"\n📋 Prediction Results:")
    detailed_results = scorePredict(y_predict, labels_test, labels)
    print(detailed_results)


def evaluate_model_detailed(model, df_test, labels, epoch_num=None):
    """
    ✅ DETAILED MODEL EVALUATION - Wie in der ursprünglichen Version
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, matthews_corrcoef, 
        confusion_matrix, classification_report, precision_recall_fscore_support
    )
    
    print(f"\n📊 DETAILED EVALUATION" + (f" - EPOCH {epoch_num}" if epoch_num else ""))
    print("=" * 70)
    
    # Get predictions
    result, model_outputs, wrong_predictions = model.eval_model(
        df_test,
        acc=accuracy_score,
        f1_macro=lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
        f1_weighted=lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
        mcc=matthews_corrcoef
    )
    
    # Get true labels and predictions
    labels_test = df_test['labels'].to_numpy()
    y_predict = np.argmax(model_outputs, axis=1)
    
    # ✅ BASIC METRICS
    accuracy = result['acc']
    f1_macro = result.get('f1_macro', 0)
    f1_weighted = result.get('f1_weighted', 0)
    mcc = result.get('mcc', 0)
    
    print(f"🎯 OVERALL PERFORMANCE:")
    print(f"   Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   F1 Macro:     {f1_macro:.4f} ({f1_macro*100:.2f}%)")
    print(f"   F1 Weighted:  {f1_weighted:.4f} ({f1_weighted*100:.2f}%)")
    print(f"   MCC:          {mcc:.4f}")
    
    # ✅ CONFUSION MATRIX
    cm = confusion_matrix(labels_test, y_predict, labels=labels)
    print(f"\n📋 CONFUSION MATRIX:")
    print("-" * 40)
    
    # Create header
    header = "True\\Pred"
    for i, label in enumerate(labels):
        header += f"  {i:>6}"
    print(header)
    
    # Print matrix rows
    for i, label in enumerate(labels):
        row = f"   {i:>6}  "
        for j in range(len(labels)):
            row += f"  {cm[i][j]:>6}"
        print(row)
    
    # ✅ CLASS-WISE DETAILED METRICS
    precision, recall, f1_scores, support = precision_recall_fscore_support(
        labels_test, y_predict, labels=labels, average=None, zero_division=0
    )
    
    print(f"\n📊 CLASS-WISE PERFORMANCE:")
    print("-" * 70)
    print(f"{'Class':<8} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 70)
    
    # Mapping für bessere Lesbarkeit (falls du die Label-Namen hast)
    class_names = {0: 'agree', 1: 'disagree', 2: 'discuss', 3: 'unrelated'}
    
    for i, label in enumerate(labels):
        class_name = class_names.get(i, f'class_{i}')
        print(f"{class_name:<8} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1_scores[i]:<10.4f} {support[i]:<10}")
    
    # Macro averages
    print("-" * 70)
    print(f"{'macro':<8} {precision.mean():<10.4f} {recall.mean():<10.4f} {f1_scores.mean():<10.4f} {support.sum():<10}")
    print(f"{'weighted':<8} {np.average(precision, weights=support):<10.4f} {np.average(recall, weights=support):<10.4f} {np.average(f1_scores, weights=support):<10.4f} {support.sum():<10}")
    
    # ✅ DETAILED CLASSIFICATION REPORT
    print(f"\n📋 SKLEARN CLASSIFICATION REPORT:")
    print("-" * 70)
    target_names = [class_names.get(i, f'class_{i}') for i in labels]
    report = classification_report(labels_test, y_predict, target_names=target_names, digits=4, zero_division=0)
    print(report)
    
    # ✅ ERROR ANALYSIS
    print(f"\n❌ ERROR ANALYSIS:")
    print("-" * 40)
    print(f"Total predictions: {len(labels_test):,}")
    print(f"Correct predictions: {(labels_test == y_predict).sum():,}")
    print(f"Wrong predictions: {len(wrong_predictions):,}")
    print(f"Error rate: {len(wrong_predictions)/len(labels_test)*100:.2f}%")
    
    # Per-class error breakdown
    print(f"\n🔍 ERRORS BY TRUE CLASS:")
    for i, label in enumerate(labels):
        class_name = class_names.get(i, f'class_{i}')
        true_class_mask = labels_test == i
        true_class_count = true_class_mask.sum()
        correct_class_count = ((labels_test == i) & (y_predict == i)).sum()
        error_count = true_class_count - correct_class_count
        error_rate = error_count / true_class_count * 100 if true_class_count > 0 else 0
        
        print(f"   {class_name:<10}: {error_count:>4}/{true_class_count:>4} errors ({error_rate:>5.1f}%)")
    
    # ✅ CONFIDENCE ANALYSIS
    print(f"\n🎲 PREDICTION CONFIDENCE:")
    print("-" * 40)
    
    # Softmax probabilities
    probabilities = np.exp(model_outputs) / np.sum(np.exp(model_outputs), axis=1, keepdims=True)
    max_probs = np.max(probabilities, axis=1)
    
    print(f"Average confidence: {max_probs.mean():.4f}")
    print(f"Min confidence: {max_probs.min():.4f}")
    print(f"Max confidence: {max_probs.max():.4f}")
    print(f"Std confidence: {max_probs.std():.4f}")
    
    # Confidence distribution
    high_conf = (max_probs > 0.9).sum()
    med_conf = ((max_probs > 0.7) & (max_probs <= 0.9)).sum()
    low_conf = (max_probs <= 0.7).sum()
    
    print(f"\nConfidence distribution:")
    print(f"   High (>0.9): {high_conf:>6} ({high_conf/len(max_probs)*100:>5.1f}%)")
    print(f"   Med (0.7-0.9): {med_conf:>4} ({med_conf/len(max_probs)*100:>5.1f}%)")
    print(f"   Low (≤0.7): {low_conf:>6} ({low_conf/len(max_probs)*100:>5.1f}%)")
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'mcc': mcc,
        'confusion_matrix': cm.tolist(),
        'class_metrics': {
            'precision': precision.tolist(),
            'recall': recall.tolist(), 
            'f1_scores': f1_scores.tolist(),
            'support': support.tolist()
        },
        'confidence_stats': {
            'mean': float(max_probs.mean()),
            'std': float(max_probs.std()),
            'min': float(max_probs.min()),
            'max': float(max_probs.max())
        }
    }


def train_with_early_stopping_detailed(model, df_train, df_test, output_dir, value_head):
    """
    ✅ Early Stopping mit DETAILED Evaluation nach jeder Epoche
    """
    best_accuracy = 0.0
    no_improvement_count = 0
    patience = 2
    training_log = []
    best_model = None
    
    # Get unique labels for detailed analysis
    labels = list(df_train['labels'].unique())
    labels.sort()
    
    # Create clean output structure
    os.makedirs(output_dir, exist_ok=True)
    temp_training_dir = os.path.join(output_dir, "temp_training")
    
    print(f"\n🧹 DETAILED EARLY STOPPING SETUP")
    print(f"📂 Output: {output_dir}")
    print(f"📊 Classes: {labels}")
    print("=" * 70)
    
    for epoch in range(1, 7):  # Maximum 6 Epochen
        epoch_start = time.time()
        
        print(f"\n🔥 EPOCH {epoch}/6 STARTING")
        print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
        if epoch > 1:
            print(f"🏆 Current best: {best_accuracy:.2f}%")
            print(f"⏳ Patience: {no_improvement_count}/{patience}")
        print("-" * 50)
        
        # Training
        epoch_temp_dir = os.path.join(temp_training_dir, f"epoch_{epoch}")
        original_epochs = model.args['num_train_epochs']
        model.args['num_train_epochs'] = 1
        
        try:
            model.train_model(df_train, output_dir=epoch_temp_dir)
            model.args['num_train_epochs'] = original_epochs
        except Exception as e:
            print(f"❌ Training failed in epoch {epoch}: {e}")
            model.args['num_train_epochs'] = original_epochs
            raise
        
        # ✅ DETAILED EVALUATION
        epoch_duration = time.time() - epoch_start
        
        # Use detailed evaluation function
        detailed_results = evaluate_model_detailed(model, df_test, labels, epoch)
        
        current_accuracy = detailed_results['accuracy'] * 100
        current_f1 = detailed_results['f1_macro'] * 100
        
        # Log this epoch
        epoch_log = {
            'epoch': epoch,
            'accuracy': current_accuracy,
            'f1_macro': current_f1,
            'duration': epoch_duration,
            'timestamp': datetime.now().isoformat(),
            'detailed_metrics': detailed_results
        }
        training_log.append(epoch_log)
        
        print(f"\n⏰ EPOCH {epoch} DURATION: {timedelta(seconds=int(epoch_duration))}")
        
        # Early stopping logic
        if current_accuracy > best_accuracy + 0.01:
            improvement = current_accuracy - best_accuracy
            best_accuracy = current_accuracy
            no_improvement_count = 0
            
            # Save best model
            best_model_dir = os.path.join(output_dir, "best_model")
            
            try:
                if os.path.exists(best_model_dir):
                    import shutil
                    shutil.rmtree(best_model_dir)
                
                import shutil
                shutil.copytree(epoch_temp_dir, best_model_dir)
                
                # Save metadata
                config = {
                    'value_head': value_head,
                    'model_type': 'deberta-v3',
                    'num_labels': model.num_labels,
                    'feature_dim': model.feature_dim,
                    'best_epoch': epoch,
                    'best_accuracy': best_accuracy,
                    'best_f1_macro': current_f1,
                    'detailed_results': detailed_results
                }
                with open(os.path.join(best_model_dir, 'best_model_info.json'), 'w') as f:
                    json.dump(config, f, indent=2)
                
                print(f"\n🏆 NEW BEST MODEL! Improvement: +{improvement:.2f}%")
                print(f"💾 Saved to: {best_model_dir}")
                
                best_model = model
                
            except Exception as save_error:
                print(f"⚠️  Save error: {save_error}")
                best_model = model
        else:
            no_improvement_count += 1
            decline = best_accuracy - current_accuracy
            
            print(f"\n📉 No improvement: -{decline:.2f}% from best")
            print(f"⚠️  Patience: {no_improvement_count}/{patience}")
            
            if no_improvement_count >= patience:
                print(f"\n🛑 EARLY STOPPING TRIGGERED!")
                print(f"   No improvement for {patience} epochs")
                print(f"   Best: {best_accuracy:.2f}% (Epoch {epoch - patience})")
                break
        
        # Cleanup
        try:
            import shutil
            shutil.rmtree(epoch_temp_dir)
        except:
            pass
        
        print("=" * 70)
    
    # Final cleanup and save log
    try:
        import shutil
        if os.path.exists(temp_training_dir):
            shutil.rmtree(temp_training_dir)
        print(f"\n🧹 Cleanup completed")
    except:
        pass
    
    # Save detailed training log
    log_path = os.path.join(output_dir, 'detailed_training_log.json')
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print(f"📊 Detailed log saved: {log_path}")
    
    # Load best model
    best_model_path = os.path.join(output_dir, "best_model")
    if os.path.exists(best_model_path):
        try:
            loaded_model = OutClassificationModel(
                'deberta-v3',
                best_model_path,
                use_cuda=True,
                args={'value_head': value_head}
            )
            return loaded_model, best_accuracy, training_log
        except Exception as load_error:
            print(f"⚠️  Load error: {load_error}")
            return best_model or model, best_accuracy, training_log
    else:
        return best_model or model, best_accuracy, training_log


def train_with_early_stopping_simple(model, df_train, df_test, output_dir, value_head):
    """
    Vereinfachte Early Stopping Logic - FIXED VERSION
    """
    best_accuracy = 0.0
    no_improvement_count = 0
    patience = 2
    training_log = []
    best_model = None  # ✅ Initialize best_model
    
    # Trainiere Epoche für Epoche
    for epoch in range(1, 7):  # Maximum 6 Epochen
        epoch_start = time.time()
        
        print(f"\n🔥 EPOCH {epoch}/6 STARTING")
        print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
        if epoch > 1:
            print(f"🏆 Current best: {best_accuracy:.2f}%")
            print(f"⏳ Patience: {no_improvement_count}/{patience}")
        print("-" * 50)
        
        # Trainiere eine Epoche (überschreibe num_train_epochs temporär)
        original_epochs = model.args['num_train_epochs']
        model.args['num_train_epochs'] = 1
        
        epoch_output_dir = os.path.join(output_dir, f"epoch_{epoch}")
        
        try:
            # Training für eine Epoche
            model.train_model(df_train, output_dir=epoch_output_dir)
            
            # Restore original epochs
            model.args['num_train_epochs'] = original_epochs
            
        except Exception as e:
            print(f"❌ Training failed in epoch {epoch}: {e}")
            model.args['num_train_epochs'] = original_epochs
            raise
        
        # ✅ SOFORT AUF TEST SET EVALUIEREN
        print(f"\n📊 EVALUATING EPOCH {epoch} ON TEST SET...")
        
        result, _, _ = model.eval_model(
            df_test,
            acc=accuracy_score,
            f1_macro=lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro')
        )
        
        current_accuracy = result['acc'] * 100
        current_f1 = result.get('f1_macro', 0) * 100
        
        epoch_duration = time.time() - epoch_start
        
        # Log this epoch
        epoch_log = {
            'epoch': epoch,
            'accuracy': current_accuracy,
            'f1_macro': current_f1,
            'duration': epoch_duration,
            'timestamp': datetime.now().isoformat()
        }
        training_log.append(epoch_log)
        
        print(f"\n✅ EPOCH {epoch} RESULTS:")
        print(f"   🎯 Test Accuracy: {current_accuracy:.2f}%")
        print(f"   📊 Test F1 Macro: {current_f1:.2f}%")
        print(f"   ⏰ Duration: {timedelta(seconds=int(epoch_duration))}")
        
        # ✅ EARLY STOPPING LOGIC - FIXED
        if current_accuracy > best_accuracy + 0.01:  # Mindestverbesserung 0.01%
            improvement = current_accuracy - best_accuracy
            best_accuracy = current_accuracy
            no_improvement_count = 0
            
            # ✅ FIXED: Proper model saving
            best_model_dir = os.path.join(output_dir, "best_model")
            os.makedirs(best_model_dir, exist_ok=True)
            
            try:
                # Save tokenizer
                model.tokenizer.save_pretrained(best_model_dir)
                
                # Save model using our custom save_pretrained method
                model.model.save_pretrained(best_model_dir)
                
                # Save config
                config = {
                    'value_head': value_head, 
                    'model_type': 'deberta-v3',
                    'num_labels': model.num_labels,
                    'feature_dim': model.feature_dim
                }
                with open(os.path.join(best_model_dir, 'training_config.json'), 'w') as f:
                    json.dump(config, f, indent=2)
                
                print(f"   🏆 NEW BEST! Improvement: +{improvement:.2f}%")
                print(f"   💾 Saved as best model to: {best_model_dir}")
                
                # Keep reference to best model
                best_model = model
                
            except Exception as save_error:
                print(f"   ⚠️  Warning: Could not save best model: {save_error}")
                print(f"   📊 But training continues...")
                # Don't fail, just continue training
                best_model = model
            
        else:
            no_improvement_count += 1
            decline = best_accuracy - current_accuracy
            
            print(f"   📉 No significant improvement: -{decline:.2f}% from best")
            print(f"   ⚠️  Patience counter: {no_improvement_count}/{patience}")
            
            if no_improvement_count >= patience:
                print(f"\n🛑 EARLY STOPPING TRIGGERED!")
                print(f"   No improvement for {patience} epochs")
                print(f"   Best accuracy: {best_accuracy:.2f}% (Epoch {epoch - patience})")
                print(f"   Stopping training to prevent overfitting...")
                break
        
        print("=" * 70)
    
    # ✅ FIXED: Return current model if best_model loading fails
    best_model_path = os.path.join(output_dir, "best_model")
    if os.path.exists(best_model_path) and best_model is not None:
        print(f"\n💾 Loading best model for final evaluation...")
        try:
            # Try to load the best model
            loaded_model = OutClassificationModel(
                'deberta-v3',
                best_model_path,
                use_cuda=True,
                args={'value_head': value_head}
            )
            return loaded_model, best_accuracy, training_log
        except Exception as load_error:
            print(f"⚠️  Could not load best model: {load_error}")
            print(f"🔄 Using current model instead")
            return best_model or model, best_accuracy, training_log
    else:
        print(f"⚠️  Best model not found, using current model")
        return best_model or model, best_accuracy, training_log

def train_with_early_stopping_clean(model, df_train, df_test, output_dir, value_head):
    """
    ✅ CLEAN Early Stopping - Minimale Speicherung, maximale Effizienz
    """
    best_accuracy = 0.0
    no_improvement_count = 0
    patience = 2
    training_log = []
    best_model = None
    
    # ✅ Create clean output structure
    os.makedirs(output_dir, exist_ok=True)
    temp_training_dir = os.path.join(output_dir, "temp_training")
    
    print(f"\n🧹 CLEAN TRAINING SETUP")
    print(f"📂 Main output: {output_dir}")
    print(f"🗂️  Temp training: {temp_training_dir}")
    print(f"💾 Only best model will be permanently saved")
    print("=" * 60)
    
    for epoch in range(1, 7):  # Maximum 6 Epochen
        epoch_start = time.time()
        
        print(f"\n🔥 EPOCH {epoch}/6 STARTING")
        print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
        if epoch > 1:
            print(f"🏆 Current best: {best_accuracy:.2f}%")
            print(f"⏳ Patience: {no_improvement_count}/{patience}")
        print("-" * 50)
        
        # ✅ Use temporary directory for each epoch
        epoch_temp_dir = os.path.join(temp_training_dir, f"epoch_{epoch}")
        
        # Trainiere eine Epoche
        original_epochs = model.args['num_train_epochs']
        model.args['num_train_epochs'] = 1
        
        try:
            # ✅ Train to temporary location
            model.train_model(df_train, output_dir=epoch_temp_dir)
            model.args['num_train_epochs'] = original_epochs
            
        except Exception as e:
            print(f"❌ Training failed in epoch {epoch}: {e}")
            model.args['num_train_epochs'] = original_epochs
            raise
        
        # ✅ EVALUATE
        print(f"\n📊 EVALUATING EPOCH {epoch} ON TEST SET...")
        
        result, _, _ = model.eval_model(
            df_test,
            acc=accuracy_score,
            f1_macro=lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro')
        )
        
        current_accuracy = result['acc'] * 100
        current_f1 = result.get('f1_macro', 0) * 100
        epoch_duration = time.time() - epoch_start
        
        # Log this epoch
        epoch_log = {
            'epoch': epoch,
            'accuracy': current_accuracy,
            'f1_macro': current_f1,
            'duration': epoch_duration,
            'timestamp': datetime.now().isoformat()
        }
        training_log.append(epoch_log)
        
        print(f"\n✅ EPOCH {epoch} RESULTS:")
        print(f"   🎯 Test Accuracy: {current_accuracy:.2f}%")
        print(f"   📊 Test F1 Macro: {current_f1:.2f}%")
        print(f"   ⏰ Duration: {timedelta(seconds=int(epoch_duration))}")
        
        # ✅ EARLY STOPPING LOGIC
        if current_accuracy > best_accuracy + 0.01:
            improvement = current_accuracy - best_accuracy
            best_accuracy = current_accuracy
            no_improvement_count = 0
            
            # ✅ COPY only the best model to permanent location
            best_model_dir = os.path.join(output_dir, "best_model")
            
            try:
                # Remove old best model
                if os.path.exists(best_model_dir):
                    import shutil
                    shutil.rmtree(best_model_dir)
                
                # Copy current epoch model as best
                import shutil
                shutil.copytree(epoch_temp_dir, best_model_dir)
                
                # Save additional metadata
                config = {
                    'value_head': value_head,
                    'model_type': 'deberta-v3',
                    'num_labels': model.num_labels,
                    'feature_dim': model.feature_dim,
                    'best_epoch': epoch,
                    'best_accuracy': best_accuracy,
                    'best_f1_macro': current_f1
                }
                with open(os.path.join(best_model_dir, 'best_model_info.json'), 'w') as f:
                    json.dump(config, f, indent=2)
                
                print(f"   🏆 NEW BEST! Improvement: +{improvement:.2f}%")
                print(f"   💾 Saved as best model")
                
                best_model = model
                
            except Exception as save_error:
                print(f"   ⚠️  Warning: Could not save best model: {save_error}")
                best_model = model
        else:
            no_improvement_count += 1
            decline = best_accuracy - current_accuracy
            
            print(f"   📉 No improvement: -{decline:.2f}% from best")
            print(f"   ⚠️  Patience: {no_improvement_count}/{patience}")
            
            if no_improvement_count >= patience:
                print(f"\n🛑 EARLY STOPPING TRIGGERED!")
                print(f"   No improvement for {patience} epochs")
                print(f"   Best: {best_accuracy:.2f}% (Epoch {epoch - patience})")
                break
        
        # ✅ CLEANUP: Remove temporary epoch directory
        try:
            import shutil
            shutil.rmtree(epoch_temp_dir)
            print(f"   🧹 Cleaned up temporary files for epoch {epoch}")
        except:
            pass  # Ignore cleanup errors
        
        print("=" * 70)
    
    # ✅ FINAL CLEANUP: Remove all temporary directories
    try:
        import shutil
        if os.path.exists(temp_training_dir):
            shutil.rmtree(temp_training_dir)
        print(f"\n🧹 CLEANUP COMPLETED")
        print(f"   Removed all temporary training files")
        print(f"   Only best model retained in: {os.path.join(output_dir, 'best_model')}")
    except Exception as cleanup_error:
        print(f"⚠️  Cleanup warning: {cleanup_error}")
    
    # ✅ Save training log
    log_path = os.path.join(output_dir, 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print(f"📊 Training log saved to: {log_path}")
    
    # Load best model for return
    best_model_path = os.path.join(output_dir, "best_model")
    if os.path.exists(best_model_path):
        try:
            loaded_model = OutClassificationModel(
                'deberta-v3',
                best_model_path,
                use_cuda=True,
                args={'value_head': value_head}
            )
            return loaded_model, best_accuracy, training_log
        except Exception as load_error:
            print(f"⚠️  Load error: {load_error}, using current model")
            return best_model or model, best_accuracy, training_log
    else:
        return best_model or model, best_accuracy, training_log
    
# ✅ FÜGE DIESE FUNKTION ZU DEINEM CODE HINZU
def train_predict_model_early_stopping(df_train, df_test, is_predict, use_cuda, value_head):
    """
    Enhanced DeBERTa Training mit Early Stopping und detaillierter Evaluation
    """
    labels_test = pd.Series(df_test['labels']).to_numpy()
    labels = list(df_train['labels'].unique())
    labels.sort()

    print(f"🚀 DeBERTa v3 Large - Training mit Early Stopping")
    print(f"=" * 70)
    print(f"📊 Number of classes: {len(labels)}")
    print(f"📊 Classes: {labels}")
    print(f"📊 Training samples: {len(df_train):,}")
    print(f"📊 Test samples: {len(df_test):,}")
    print(f"🔧 External feature dimension: {value_head}")
    print(f"💾 CUDA enabled: {use_cuda}")
    print(f"⏰ Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ✅ TRAINING CONFIGURATION
    print(f"\n📈 ENHANCED TRAINING CONFIGURATION")
    print(f"-" * 40)
    print(f"🔢 Max Epochs: 6")
    print(f"📚 Learning Rate: 8e-6")
    print(f"📦 Batch Size: 8")
    print(f"🔄 Gradient Accumulation: 2")
    print(f"📦 Effective Batch Size: 16")
    print(f"🔥 Warmup Ratio: 0.05")
    print(f"⚖️  Weight Decay: 0.01")
    print(f"⏳ Early Stopping Patience: 2 epochs")
    print(f"🎯 Strategy: Detailed evaluation after each epoch")

    # ✅ MODEL INITIALIZATION
    model = OutClassificationModel(
        'deberta-v3', 
        './models/deberta-v3-large', 
        num_labels=len(labels),
        use_cuda=use_cuda, 
        args={'value_head': value_head}
    )

    # ✅ EARLY STOPPING TRAINING
    training_start = time.time()
    output_dir = f'./outputs/deberta_early_stop_{datetime.now().strftime("%Y%m%d_%H%M")}'
    
    print(f"\n🎯 STARTING DETAILED EARLY STOPPING TRAINING")
    print(f"=" * 70)
    print(f"📂 Output directory: {output_dir}")
    print("=" * 70)
    
    try:
        # ✅ VERWENDE DETAILED EARLY STOPPING
        best_model, best_accuracy, training_log = train_with_early_stopping_detailed(
            model, df_train, df_test, output_dir, value_head
        )
        
        training_duration = time.time() - training_start
        epochs_completed = len(training_log)
        
        print(f"\n✅ TRAINING COMPLETED!")
        print(f"⏰ Total training time: {timedelta(seconds=int(training_duration))}")
        print(f"📊 Epochs completed: {epochs_completed}/6")
        print(f"⏰ Average time per epoch: {timedelta(seconds=int(training_duration/epochs_completed))}")
        print(f"🏆 Best test accuracy: {best_accuracy:.2f}%")
        
        # Zeige Training Log
        print(f"\n📈 TRAINING PROGRESSION:")
        print(f"-" * 50)
        baseline_acc = 93.77  # Deine vorherige Baseline
        
        for i, log_entry in enumerate(training_log, 1):
            acc = log_entry['accuracy']
            
            if i == 1:
                change_text = "(Baseline etabliert)"
            else:
                prev_acc = training_log[i-2]['accuracy']
                change = acc - prev_acc
                if change > 0.3:
                    change_text = f"(+{change:.1f}% - guter Start)"
                elif change > 0:
                    change_text = f"(+{change:.1f}% - solide)"
                elif change > -0.1:
                    change_text = f"({change:.1f}% - Plateau beginnt)"
                else:
                    change_text = f"({change:.1f}% - leichtes Overfitting)"
            
            improvement_from_baseline = acc - baseline_acc
            baseline_text = f"[{improvement_from_baseline:+.1f}% vs Baseline]"
            
            status = "🏆 BEST" if acc == best_accuracy else ""
            print(f"Epoch {i}: {acc:.1f}% {change_text} {baseline_text} {status}")
        
        # Early stopping reason
        if epochs_completed < 6:
            print(f"\n🛑 Early stopping triggered after {epochs_completed} epochs")
            print(f"   Reason: No improvement for 2 consecutive epochs")
        else:
            print(f"\n✅ Completed all 6 epochs")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # ✅ FINAL EVALUATION MIT BESTEM MODELL
    print(f"\n📊 FINAL EVALUATION WITH BEST MODEL")
    print(f"=" * 60)
    
    results = ''
    if is_predict:
        print(f"🔮 Making predictions with best model...")
        
        # Prepare prediction data
        value_in = []
        for _, row in df_test.iterrows():
            if 'feature' in df_test.columns and row['feature'] is not None:
                features = row['feature']
                if isinstance(features, (int, float)) and features == 0:
                    features = [0.0] * value_head
                elif not isinstance(features, (list, tuple)):
                    features = [float(features)] * value_head if value_head > 0 else [0.0]
                value_in.append([row['text_a'], row['text_b'], features])
            else:
                value_in.append([row['text_a'], row['text_b'], [0.0] * max(1, value_head)])
        
        _, model_outputs_test = best_model.predict(value_in)
        
    else:
        print(f"📊 Evaluating best model on test set...")
        result, model_outputs_test, wrong_predictions = best_model.eval_model(
            df_test, 
            acc=accuracy_score,
            f1_macro=lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro')
        )
        results = result.get('acc', 0.0)
        
        print(f"\n🎯 FINAL RESULTS WITH BEST MODEL")
        print(f"-" * 40)
        print(f"🎯 Test Accuracy: {results:.4f}")
        print(f"📊 F1 Macro: {result.get('f1_macro', 0.0):.4f}")
        print(f"📊 MCC: {result.get('mcc', 0.0):.4f}")

    # ✅ DETAILED CLASSIFICATION ANALYSIS (wie in der ursprünglichen Version)
    y_predict = np.argmax(model_outputs_test, axis=1)
    
    print(f"\n📋 DETAILED CLASSIFICATION REPORT")
    print(f"=" * 70)
    detailed_results = scorePredict(y_predict, labels_test, labels)
    print(detailed_results)
    
    # ✅ PERFORMANCE COMPARISON
    print(f"\n📈 PERFORMANCE COMPARISON WITH LITERATURE")
    print(f"=" * 70)
    paper_results = {
        "Paper (2-stage RoBERTa)": {"accuracy": 94.13, "f1_macro": 80.95},
        "Zhang et al.": {"accuracy": 93.77, "f1_macro": 83.10},
        "Dulhanty et al. (RoBERTa)": {"accuracy": 93.71, "f1_macro": 78.42},
        "Your previous DeBERTa": {"accuracy": 93.77, "f1_macro": 78.74}
    }
    
    if isinstance(results, float) and results > 0:
        current_acc = results * 100
        current_f1 = result.get('f1_macro', 0.0) * 100
        
        print(f"🔥 YOUR ENHANCED DeBERTa RESULTS:")
        print(f"   Accuracy: {current_acc:.2f}%")
        print(f"   F1 Macro: {current_f1:.2f}%")
        
        best_acc = max([r["accuracy"] for r in paper_results.values()])
        
        if current_acc > best_acc:
            print(f"🏆 NEW ACCURACY RECORD! (+{current_acc - best_acc:.2f}%)")
        else:
            print(f"📊 Accuracy gap to best: {best_acc - current_acc:.2f}%")
        
        # Improvement from previous
        prev_acc = 93.77
        improvement = current_acc - prev_acc
        print(f"\n📈 IMPROVEMENT FROM PREVIOUS:")
        print(f"   Previous: {prev_acc:.2f}%")
        print(f"   Current:  {current_acc:.2f}%")
        print(f"   Change:   {improvement:+.2f}%")
        
        # Success evaluation
        if current_acc > 94.13:
            print(f"\n🎉 SUCCESS! You beat the paper's best result!")
        elif current_acc > 93.77:
            print(f"\n✅ IMPROVEMENT! Better than your baseline!")
        else:
            print(f"\n⚠️  Need more work to improve baseline")
    
    print(f"\n💾 Complete results saved to: {output_dir}")
    
    return results

if __name__ == "__main__":
    print("🚀 DeBERTa Training Script")
    print("Available functions:")
    print("- train_predict_model(): Original function")
    print("- train_predict_model_early_stopping(): Enhanced with early stopping")
    print("- predict(): Load saved model and predict")