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
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero-title {
    font-size: 46px;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}

.hero-subtitle {
    font-size: 18px;
    color: #b6beca;
    margin-bottom: 2rem;
}

.info-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 18px;
    padding: 24px;
    margin-top: 8px;
}

.info-card p {
    margin: 8px 0;
    font-size: 16px;
}

.small-note {
    color: #94a3b8;
    font-size: 14px;
}

.result-box {
    border-radius: 16px;
    padding: 20px;
    margin-top: 16px;
    font-size: 17px;
}

.low {
    background-color: #12351f;
    color: #9ff2b2;
}

.medium {
    background-color: #3a2f12;
    color: #ffe08a;
}

.high {
    background-color: #3b1117;
    color: #ffb4b4;
}

.footer-note {
    color: #94a3b8;
    font-size: 13px;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


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
    "Legitimate": "This message looks normal based on the model prediction. Still, users should remain careful with unexpected links or requests.",
    "Spam": "This message may be promotional or unwanted. Avoid replying if the sender is unknown.",
    "Smishing": "This message may be a phishing attempt. Do not click the link, provide OTP codes, or share banking details."
}

example_messages = {
    "Legitimate example": "Hi Mom, I will arrive home around 7pm. Please do not wait for dinner.",
    "Spam example": "Congratulations! You have been selected to receive a free vacation package. Reply YES now!",
    "Smishing example": "Your package delivery failed. Pay RM2.99 redelivery fee here: http://track-parcel-secure.com"
}

if "sms_text" not in st.session_state:
    st.session_state.sms_text = ""


def set_example(example_name):
    st.session_state.sms_text = example_messages[example_name]


st.markdown(
    '<div class="hero-title">📩 SMS Phishing Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">A BERT-based prototype for classifying SMS messages as legitimate, spam, or smishing.</div>',
    unsafe_allow_html=True
)

left_col, right_col = st.columns([1.15, 0.85], gap="large")

with left_col:
    st.subheader("Message Input")

    sms_text = st.text_area(
        "Enter SMS message:",
        key="sms_text",
        height=160,
        placeholder="Example: Your bank account has been locked. Verify now using this link."
    )

    st.caption("Quick test examples")

    ex1, ex2, ex3 = st.columns(3)

    with ex1:
        st.button(
            "Legitimate",
            use_container_width=True,
            on_click=set_example,
            args=("Legitimate example",)
        )

    with ex2:
        st.button(
            "Spam",
            use_container_width=True,
            on_click=set_example,
            args=("Spam example",)
        )

    with ex3:
        st.button(
            "Smishing",
            use_container_width=True,
            on_click=set_example,
            args=("Smishing example",)
        )

    detect_button = st.button(
        "Detect Message",
        type="primary",
        use_container_width=False
    )

with right_col:
    st.subheader("Prototype Details")

    st.markdown("""
    <div class="info-card">
        <p><b>Selected model:</b> BERT-base</p>
        <p><b>Deployment setup:</b> Streamlit Cloud + Hugging Face Hub</p>
        <p><b>Input:</b> SMS text message</p>
        <p><b>Output:</b> Message class, confidence score, risk level, and user advice</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p class="small-note">
    This prototype focuses on practical SMS fraud screening. The model output should be treated as a risk indicator, not as a final legal or security decision.
    </p>
    """, unsafe_allow_html=True)


if detect_button:
    if not st.session_state.sms_text.strip():
        st.warning("Please enter an SMS message before running detection.")
    else:
        inputs = tokenizer(
            st.session_state.sms_text,
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
        confidence = float(probs[pred])
        risk = risk_map[prediction]

        st.divider()
        st.subheader("Detection Result")

        c1, c2, c3 = st.columns(3)

        c1.metric("Predicted Class", prediction)
        c2.metric("Confidence Score", f"{confidence * 100:.2f}%")
        c3.metric("Risk Level", risk)

        if prediction == "Smishing":
            box_class = "high"
            icon = "⚠️"
        elif prediction == "Spam":
            box_class = "medium"
            icon = "⚠️"
        else:
            box_class = "low"
            icon = "✅"

        st.markdown(
            f'<div class="result-box {box_class}"><b>{icon} Recommendation:</b><br>{advice_map[prediction]}</div>',
            unsafe_allow_html=True
        )

        prob_df = pd.DataFrame({
            "Class": ["Legitimate", "Spam", "Smishing"],
            "Probability": [
                float(probs[0]),
                float(probs[1]),
                float(probs[2])
            ]
        })

        st.subheader("Class Probability Distribution")
        st.bar_chart(prob_df.set_index("Class"))

        st.subheader("Risk Interpretation")

        if prediction == "Smishing":
            st.write(
                "The message shows patterns commonly associated with phishing, such as urgent account action, suspicious links, payment requests, or identity verification prompts."
            )
        elif prediction == "Spam":
            st.write(
                "The message appears more like unwanted promotional content. It may not be directly malicious, but the user should still avoid interacting with unknown senders."
            )
        else:
            st.write(
                "The message does not show strong phishing or spam patterns. However, users should still verify unexpected messages, especially if they contain links or sensitive requests."
            )

st.markdown(
    '<div class="footer-note">Prototype developed for SMS phishing and spam detection using a fine-tuned BERT model.</div>',
    unsafe_allow_html=True
)
