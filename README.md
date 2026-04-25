# 🧠 Hybrid Fake News Detection System

An end-to-end AI system for detecting fake news using a **hybrid architecture** that combines classical machine learning and transformer-based deep learning.

---

## 🚀 Overview

This project goes beyond a simple classifier and implements a **multi-stage AI decision system**:

* ⚡ **Fast TF-IDF + SVM model** for quick filtering
* 🧠 **DistilBERT transformer** for deep semantic verification
* 🎯 **Confidence-aware decision logic** for handling uncertainty
* 🌐 **Streamlit dashboard** for interactive user experience

The system is designed to balance **speed, accuracy, and scalability**, similar to real-world AI production systems.

---

## 🧩 System Architecture

```
User Input (News Text)
        ↓
Streamlit UI
        ↓
Hybrid Prediction Engine
        ↓
 ┌───────────────┐
 │   SVM Model   │  (Fast Filter)
 └───────────────┘
        ↓
 If Low Confidence
        ↓
 ┌────────────────────┐
 │  DistilBERT Model  │ (Deep Semantic Analysis)
 └────────────────────┘
        ↓
 Final Prediction + Confidence
```

---

## ✨ Features

* ✅ Hybrid AI model (SVM + DistilBERT)
* ✅ Confidence-based prediction system
* ✅ Real-time interactive UI (Streamlit)
* ✅ Modular ML pipeline (preprocessing, training, inference)
* ✅ Production-style project structure
* ✅ Transformer fine-tuning using HuggingFace

---

## 📂 Project Structure

```
fake_news_detection/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_svm_baseline.ipynb
│   ├── 06_distilbert_training.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── predict.py
│   ├── predict_transformer.py
│   ├── predict_hybrid.py
│
├── models/
│   ├── svm_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── distilbert_model/
│
├── app/
│   ├── api.py
│
├── streamlit_app.py
├── requirements.txt
└── README.md
```

---

## 🧠 Models Used

### 1. TF-IDF + Linear SVM

* Fast and efficient baseline
* Works well on high-dimensional sparse text data
* Used as **first-stage filter**

### 2. DistilBERT (Transformer)

* Fine-tuned on fake news dataset
* Captures contextual and semantic meaning
* Used for **deep verification**

---

## 📊 Model Performance

| Model      | Accuracy | F1 Score |
| ---------- | -------- | -------- |
| SVM        | ~99.57%  | ~0.995   |
| DistilBERT | ~99.90%  | ~0.999   |

> ⚠️ Note: High accuracy is dataset-dependent. Real-world performance may vary.

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection

python -m venv venv
venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

---

## ▶️ Run Streamlit App

```bash
streamlit run streamlit_app.py
```

---

## 🌐 Run FastAPI (Optional)

```bash
uvicorn app.api:app --reload
```

---

## 🧪 Example Usage

Input:

```
"Breaking: Government confirms secret alien technology program"
```

Output:

```
Prediction: FAKE
Confidence: 0.92
Model Used: DistilBERT
```

---

## 🧠 Key Learnings

* Designing hybrid AI systems for real-world constraints
* Transformer fine-tuning with HuggingFace
* NLP preprocessing and feature engineering
* Model orchestration and decision logic
* Building ML-powered applications with Streamlit & FastAPI

---

## 🚀 Future Improvements

* 🔍 Explainability (SHAP / LIME)
* 🌐 Real-time news scraping integration
* 🤖 LLM-based reasoning layer
* ☁️ Cloud deployment (Streamlit Cloud / AWS)
* 📊 Monitoring & feedback loop (MLOps)
