import argparse
from common.loadData import load_all_data
from model.deberta_model import train_predict_model, predict, train_predict_model_early_stopping  # ✅ NEU

# Feature sets for different classification types
feature_stance = ['polarityClaim_nltk_neg', 'polarityClaim_nltk_pos', 'polarityBody_nltk_neg', 'polarityBody_nltk_pos']
feature_related = ['cosine_similarity', 'max_score_in_position', 'overlap', 'soft_cosine_similarity', 'bert_cs']
feature_all = ['cosine_similarity', 'max_score_in_position', 'overlap', 'soft_cosine_similarity', 'bert_cs',
               'polarityClaim_nltk_neg', 'polarityClaim_nltk_pos', 'polarityBody_nltk_neg', 'polarityBody_nltk_pos']


def main(parser):
    args = parser.parse_args()

    type_class = args.type_class
    use_cuda = args.use_cuda
    not_use_feature = args.not_use_feature
    training_set = args.training_set
    test_set = args.test_set
    model_dir = args.model_dir
    
    # Determine which features to use
    feature = []
    if not not_use_feature:
        if type_class == 'stance':
            feature = feature_stance
        elif type_class == 'related':
            feature = feature_related
        elif type_class == 'all':
            feature = feature_all

    print(f"🚀 Starting DeBERTa v3 Large Training")
    print(f"📊 Classification type: {type_class}")
    print(f"🔧 Features: {feature if feature else 'None (text-only)'}")
    print(f"💾 Using CUDA: {use_cuda}")
    print(f"📂 Training set: {training_set}")
    print(f"📂 Test set: {test_set}")

    if model_dir == "":
        # Training mode
        print(f"\n🎯 Mode: Enhanced Training mit Early Stopping")
        
        # Load data
        print(f"📥 Loading test data...")
        df_test = load_all_data(test_set, type_class, feature)
        print(f"📥 Loading training data...")
        df_train = load_all_data(training_set, type_class, feature)
        
        print(f"📊 Training samples: {len(df_train)}")
        print(f"📊 Test samples: {len(df_test)}")
        print(f"📊 Feature dimension: {len(feature)}")
        
        # ✅ VERWENDE NEUE EARLY STOPPING FUNKTION
        results = train_predict_model_early_stopping(df_train, df_test, True, use_cuda, len(feature))
        print(f"✅ Enhanced training mit early stopping completed!")
        
    else:
        # Prediction only mode
        print(f"\n🔮 Mode: Prediction Only")
        print(f"📂 Model directory: {model_dir}")
        
        # Load test data
        df_test = load_all_data(test_set, type_class, feature)
        print(f"📊 Test samples: {len(df_test)}")
        
        # Make predictions
        predict(df_test, use_cuda, model_dir, len(feature))
        print(f"✅ Prediction completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeBERTa v3 Large Stance Detection Training')
    
    ## Required parameters
    parser.add_argument("--type_class", 
                        choices=['related', 'stance', 'all'],
                        default='related', 
                        help="Classification type")

    parser.add_argument("--use_cuda",
                        default=False,
                        action='store_true',
                        help="Enable CUDA for GPU training")

    parser.add_argument("--not_use_feature",
                        default=False,
                        action='store_true',
                        help="Disable external features")

    parser.add_argument("--training_set",
                        default="/data/FNC_summy_textRank_train_spacy_pipeline_polarity_v2.json",
                        type=str,
                        help="Path to training dataset")

    parser.add_argument("--test_set",
                        default="/data/FNC_summy_textRank_test_spacy_pipeline_polarity_v2.json",
                        type=str,
                        help="Path to test dataset")

    parser.add_argument("--model_dir",
                        default="",
                        type=str,
                        help="Path to saved model for prediction-only mode")

    print("🤖 DeBERTa v3 Large Stance Detection mit Early Stopping")
    print("=" * 60)
    
    main(parser)