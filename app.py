import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(
    page_title="SMS Phishing Detection",
    page_icon="📩",
    layout="centered"
)

st.title("📩 SMS Phishing Detection System")

st.write(
    "This prototype detects whether an SMS message is legitimate, spam, or smishing."
)

repo_id = os.environ["MODEL_REPO_ID"]
token = os.environ["HUGGINGFACE_HUB_TOKEN"]

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        token=token
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        repo_id,
        token=token
    )

    model.eval()

    return tokenizer, model

tokenizer, model = load_model()

label_map = {
    0: "Legitimate",
    1: "Spam",
    2: "Smishing"
}

risk_map = {
    "Legitimate": "Low Risk",
    "Spam": "Medium Risk",
    "Smishing": "High Risk"
}

sms_text = st.text_area(
    "Enter SMS message:",
    placeholder="Example: Your account has been suspended. Click this link to verify."
)

if st.button("Detect Message"):

    if sms_text.strip():

        inputs = tokenizer(
            sms_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)

        pred = torch.argmax(probs, dim=1).item()

        prediction = label_map[pred]

        confidence = probs[0][pred].item()

        st.success(f"Prediction: {prediction}")

        st.metric(
            "Confidence Score",
            f"{confidence * 100:.2f}%"
        )

        st.warning(
            f"Risk Level: {risk_map[prediction]}"
        )

        if prediction == "Smishing":
            st.error(
                "Advice: Do not click suspicious links or share personal information."
            )
