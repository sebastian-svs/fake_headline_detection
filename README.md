# Headline-Stance-Detection

**(2025/01/29) Enhanced with DeBERTa v3 Large achieving 94.55% accuracy**

### Requirements
* Python 3.8
* Pytorch 2.4.1
* Transformers 4.46.3
* Accelerate 1.0.1

### Installation
* Create a Python Environment and activate it:
```bash 
    virtualenv stance_detection --python=python3.8
    cd ./stance_detection
    source bin/activate
```
* Install the required dependencies. 
You need to have at least version 21.0.1 of pip installed. Next you may install requirements_deberta.txt.

```bash
pip install --upgrade pip
pip install -r requirements_deberta.txt
```

### Resource
Download the FNC-dataset and the generatic summaries from this link:
```bash
wget -O data.zip "https://drive.google.com/uc?export=download&id=1b_8ZAlwOPpMsBPcg-vQE2q-4PR0F4Zuk"
unzip data.zip
rm data.zip
```

### Description of the parameters
These parameters allow configuring the system to train or predict.

|Field|Description|
|---|---|
|type_class|This parameter is used to choose the type of clasificator (related, stance, all). The "related" classification detects related and unrelated classes, the "stance" classification detects agree, disagree and discuss classes, and "all" classification detects agree, disagree, discuss and unrelated classes.|
|use_cuda|This parameter should be used if cuda is present.|
|training_set|This parameter is the relative dir of training set.|
|not_use_feature|This parameter should be used if you don't want to train with the external features.|
|test_set|This parameter is the relative dir of test set.|
|model_dir|This parameter is the relative dir of model for predict.|

#### Train and predict FNC-dataset "related" classification
Execute this command to train and predict on the dataset in related classification
```bash
PYTHONPATH=src python src/scripts/train_predict_deberta.py --training_set "/data/FNC_summy_textRank_train_spacy_pipeline_polarity_v2.json" --test_set "/data/FNC_summy_textRank_test_spacy_pipeline_polarity_v2.json" --type_class "related"
```

#### Train and predict FNC-dataset "stance" classification
Execute this command to train and predict on the dataset in stance classification
```bash
PYTHONPATH=src python src/scripts/train_predict_deberta.py --training_set "/data/FNC_summy_textRank_train_spacy_pipeline_polarity_v2.json" --test_set "/data/FNC_summy_textRank_test_spacy_pipeline_polarity_v2.json" --type_class "stance"
```

#### Train and predict FNC-dataset "all" classification
Execute this command to train and predict on the dataset in all classification
```bash
PYTHONPATH=src python src/scripts/train_predict_deberta.py --training_set "/data/FNC_summy_textRank_train_spacy_pipeline_polarity_v2.json" --test_set "/data/FNC_summy_textRank_test_spacy_pipeline_polarity_v2.json" --type_class "all"
```

If you want to change the features used modify the feature_stance, feature_related, feature_all variables.

### Contacts:
If you have any questions please contact the authors.
  * Robiert Sepúlveda Torres rsepulveda911112@gmail.com  
 
### License:
  * Apache License Version 2.0
