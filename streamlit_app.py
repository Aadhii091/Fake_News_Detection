import time
import streamlit as st
from src.predict import hybrid_predict

st.set_page_config(
    page_title="Fake News Intelligence System",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Hybrid Fake News Detection System")
st.write("AI powered verification using DistilBERT and SVM models")

news_text = st.text_area(
    "Paste News Article / Headline",
    height=200
)

st.sidebar.title("About System")
st.sidebar.write(
"""
Hybrid AI architecture:

• Fast TF-IDF SVM filter  
• Deep DistilBERT semantic verifier  
• Confidence-aware decision  

Built for misinformation detection research.
"""
)

if st.button("Detect News Authenticity"):

    if news_text.strip() == "":
        st.warning("Please enter news text")
    else:
        with st.spinner("Analyzing news..."):
            time.sleep(1)
            result = hybrid_predict(news_text)

        st.subheader("Prediction Result")

        if result["prediction"] == "FAKE":
            st.error("🚨 This news is likely FAKE")
        elif result["prediction"] == "REAL":
            st.success("✅ This news appears REAL")
        else:
            st.warning("⚠️ News authenticity UNCERTAIN")

        if "confidence" in result:
            st.write(f"Confidence Score: {result['confidence']:.2f}")
            st.progress(result["confidence"])

        st.caption(f"Decision Model : {result['model'].upper()}")
            
