from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer
import torch

model = DebertaV2ForSequenceClassification.from_pretrained('./models/deberta-v3-large', num_labels=4)
tokenizer = DebertaV2Tokenizer.from_pretrained('./models/deberta-v3-large')

print("✅ DeBERTa loads successfully!")