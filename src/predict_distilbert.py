import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models/distilbert_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

model.eval()

def predict_news_distilbert(text):

    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=256)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)

    fake_prob = probs

