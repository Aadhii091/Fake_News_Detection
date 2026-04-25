from src.preprocess import preprocess_text
from src.predict_distilbert import predict_news_distilbert
import joblib

svm_model = joblib.load(r"C:\Users\aadhi\Desktop\Projects\Fake_News_Detection\models\svm_model.pkl")
vectorizer = joblib.load(r"C:\Users\aadhi\Desktop\Projects\Fake_News_Detection\models\tfidf_vectorizer.pkl")

def predict_news(text):

    text = preprocess_text(text)

    vec = vectorizer.transform([text])

    pred = svm_model.predict(vec)[0]

    if pred == 0:
        return "FAKE"
    else:
        return "REAL"

def hybrid_predict(text):
    """
    Hybrid prediction:
    - Fast SVM for quick decision
    - DistilBERT for uncertain cases
    - Always returns a strict schema
    """

    # ---- Step 1: Fast model (SVM) ----
    svm_label = predict_news(text)

    # Normalize label (avoid mismatch issues)
    if isinstance(svm_label, str):
        svm_label = svm_label.upper().strip()

    # ---- Step 2: If confident → return early ----
    if svm_label in ["REAL", "FAKE"]:
        return {
            "model": "svm",
            "prediction": svm_label,
            "confidence": 1.0  # SVM treated as high-confidence shortcut
        }

    # ---- Step 3: Deep model fallback ----
    try:
        result = predict_news_distilbert(text)

        label = result.get("label", "UNCERTAIN")
        confidence = float(result.get("confidence", 0.0))

        # Clamp confidence (safety)
        confidence = max(0.0, min(confidence, 1.0))

        return {
            "model": "distilbert",
            "prediction": label,
            "confidence": confidence
        }

    except Exception as e:
        # ---- Step 4: Fail-safe ----
        return {
            "model": "fallback",
            "prediction": "UNCERTAIN",
            "confidence": 0.0,
        }