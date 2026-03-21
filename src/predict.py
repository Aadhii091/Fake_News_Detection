from src.preprocess import preprocess_text
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
    