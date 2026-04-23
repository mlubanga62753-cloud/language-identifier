# app.py
import streamlit as st
import pickle
import numpy as np
from utils import preprocess

# Load model
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

st.title("🇰🇪 Kenyan Language Identifier (Advanced NLP)")
st.write("Supports: Swahili, English, Sheng, Luhya + Code-Mix detection")

text = st.text_area("Enter text:")

def predict_language(text):
    clean = preprocess(text)
    vec = vectorizer.transform([clean])

    probs = model.decision_function(vec)[0]
    prediction = model.predict(vec)[0]

    # CODE-MIX DETECTION LOGIC
    sorted_scores = np.sort(probs)
    margin = sorted_scores[-1] - sorted_scores[-2]

    if margin < 0.5:
        return "Code-Mixed (Swahili/English/Sheng mix)"

    return prediction

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter text")
    else:
        result = predict_language(text)
        st.success(f"Predicted Language: {result}")