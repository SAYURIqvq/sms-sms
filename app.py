import os
import streamlit as st
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(
    page_title="SMS Phishing Detection",
    page_icon="📩",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    font-size: 44px;
    font-weight: 800;
}
.card {
    padding: 22px;
    border-radius: 18px;
    background-color: #1f2937;
    margin-bottom: 16px;
}
.result-high {
    padding: 18px;
    border-radius: 14px;
    background-color: #3b1117;
    color: #ffb4b4;
    font-weight: 700;
}
.result-medium {
    padding: 18px;
    border-radius: 14px;
    background-color: #3a2f12;
    color: #ffe08a;
    font-weight: 700;
}
.result-low {
    padding: 18px;
    border-radius: 14px;
    background-color: #12351f;
    color: #9ff2b2;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📩 SMS Phishing Detection System</div>', unsafe_allow_html=True)
st.write("A BERT-based prototype for detecting legitimate, spam, and smishing SMS messages.")

repo_id = os.environ["MODEL_REPO_ID"]
token = os.environ["HUGGINGFACE_HUB_TOKEN"]

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(repo_id, token=token)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id, token=token)
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

advice_map = {
    "Legitimate": "This message appears safe. No suspicious fraud pattern was detected.",
    "Spam": "This message may contain promotional or unwanted content. Be cautious before responding.",
    "Smishing": "This message may be a phishing attempt. Do not click links or share personal information."
}

col1, col2 = st.columns([1.1, 1])

with col1:
    st.subheader("Message Input")
    sms_text = st.text_area(
        "Enter SMS message:",
        height=160,
        placeholder="Example: Your bank account has been locked. Verify now using this link."
    )

    st.caption("Try one of these examples:")
    c1, c2, c3 = st.columns(3)

    if c1.button("Legitimate example"):
        sms_text = "Hi Mom, I will arrive home around 7pm. Please do not wait for dinner."
        st.rerun()

    if c2.button("Spam example"):
        sms_text = "Congratulations! You have been selected to receive a free vacation package. Reply YES now!"
        st.rerun()

    if c3.button("Smishing example"):
        sms_text = "Your bank account has been suspended. Verify immediately at http://secure-bank-login.com"
        st.rerun()

with col2:
    st.subheader("System Overview")
    st.markdown("""
    <div class="card">
    <b>Model:</b> BERT-base<br>
    <b>Task:</b> 3-class SMS classification<br>
    <b>Classes:</b> Legitimate, Spam, Smishing<br>
    <b>Deployment:</b> Streamlit Cloud + Hugging Face Hub
    </div>
    """, unsafe_allow_html=True)

if st.button("Detect Message", type="primary"):

    if not sms_text.strip():
        st.warning("Please enter an SMS message first.")
    else:
        inputs = tokenizer(
            sms_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred = torch.argmax(probs).item()

        prediction = label_map[pred]
        confidence = probs[pred].item()
        risk = risk_map[prediction]

        st.divider()
        st.subheader("Detection Result")

        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted Class", prediction)
        r2.metric("Confidence Score", f"{confidence * 100:.2f}%")
        r3.metric("Risk Level", risk)

        if prediction == "Smishing":
            st.markdown(f'<div class="result-high">⚠️ {advice_map[prediction]}</div>', unsafe_allow_html=True)
        elif prediction == "Spam":
            st.markdown(f'<div class="result-medium">⚠️ {advice_map[prediction]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-low">✅ {advice_map[prediction]}</div>', unsafe_allow_html=True)

        prob_df = pd.DataFrame({
            "Class": ["Legitimate", "Spam", "Smishing"],
            "Probability": [float(probs[0]), float(probs[1]), float(probs[2])]
        })

        st.subheader("Class Probability Distribution")
        st.bar_chart(prob_df.set_index("Class"))

        st.subheader("User Safety Recommendation")
        if prediction == "Smishing":
            st.write("Avoid clicking any link, do not provide OTP or banking details, and verify the sender through an official channel.")
        elif prediction == "Spam":
            st.write("Avoid replying to promotional messages from unknown senders and do not share personal information.")
        else:
            st.write("The message appears normal, but users should still verify unexpected links or requests.")
