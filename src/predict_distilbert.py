import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models/distilbert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

model.eval()

def predict_news_distilbert(text):

    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=256)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)

    fake_prob = probs[0][0].item()
    real_prob = probs[0][1].item()

    confidence = max(fake_prob,real_prob)

    if confidence < 0.60:
        label = "UNCERTAIN"
    elif fake_prob > real_prob:
        label = "FAKE"
    else:
        label = "REAL"
    
    return {"label": label, "confidence": confidence}

