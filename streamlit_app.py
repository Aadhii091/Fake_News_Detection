import time
import streamlit as st
from src.predict_distilbert import predict_news_distilbert

st.set_page_config(
    page_title="Fake News Intelligence System",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Fake News Detection System")
st.write("AI powered verification using DistilBERT")

news_text = st.text_area(
    "Paste News Article / Headline",
    height=200
)

st.sidebar.title("About System")
st.sidebar.write(
"""AI architecture:

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
            result = predict_news_distilbert(news_text)

        st.subheader("Prediction Result")

        if result["label"] == "FAKE":
            st.error("🚨 This news is likely FAKE")
        elif result["label"] == "REAL":
            st.success("✅ This news appears REAL")
        else:
            st.warning("⚠️ News authenticity UNCERTAIN")

        if "confidence" in result:
            st.write(f"Confidence Score: {result['confidence']:.2f}")
            st.progress(result["confidence"])
            
