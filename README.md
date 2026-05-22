# SMS-Phishing-Detection

## Project Title: 
NLP-DRIVEN SMISHING AND SPAM DETECTION FOR MOBILE FRAUD MITIGATION 

## Objective:
1.	To develop an automated NLP pipeline that can classify legitimate, spam, and smishing messages
2.	To evaluate the compute-to-performance trade-off between a heavy foundational transformer (BERT-base) and a lightweight distilled architecture (DistilBERT) for real-time mobile fraud detection.

## Overall Architecture:
![alt text](model_architecture.png)

---

## Model Deployment Guide (Streamlit Cloud)

**Goal:** To keep this GitHub repo lightweight, and we decided to store the trained model artifacts in a **private Hugging Face Hub** repo (best reliability on Streamlit Cloud).

This guide applies to **both** models in this project:
- BERT (`bert_final_model/`)
- DistilBERT (`distilbert_final_model/`)

### 1) What to upload (model artifacts)

Upload the **entire** Hugging Face `save_pretrained()` folder to **private Hugging Face Hub** repo. It should contain weights + config + tokenizer files, e.g.:
- `model.safetensors` 
- `config.json`
- `tokenizer.json` 
- `vocab.txt` 
- `tokenizer_config.json` 
- `special_tokens_map.json`

In the training notebook, this is created by:
- `model.save_pretrained(<folder>)`
- `tokenizer.save_pretrained(<folder>)`

### 2) Create a private Hugging Face model repo

1. Create an account on Hugging Face.
2. Create a **new model repository** (set it to **Private**)
	- `sms-phishing-bert`
	- `sms-phishing-distilbert`

### 3) Upload the model folder to Hugging Face Hub

### 4) Create an **access token** (read is enough for deployment; write needed only for uploading).
1. Click on the profile picture (top right) → Settings
2. In the left sidebar, go to Access Tokens
3. Click New token
4. Set: Name: streamlit-sms-bert / streamlit-sms-distillbert
5. Role/Scope: Read token = enough for Streamlit Cloud to download/load a private model

### 5) Update the config file in GitHub repo to keep track of repo id and token for communication
- Update information in huggingface_models_repo_config.json
 
### 6) Streamlit Cloud setup

1. Deploy the Streamlit app from this GitHub repo as usual.
2. In Streamlit Cloud → **App Settings → Secrets**, add:

```toml
HUGGINGFACE_HUB_TOKEN = "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"
MODEL_REPO_ID = "HuiLing990511/sms-phishing-bert"
```

Use a token that can read the private model repo.

### 7) Load the model inside Streamlit (example snippet)

In the Streamlit app code, load from the Hub using the token:

```python
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

repo_id = os.environ["MODEL_REPO_ID"]
token = os.environ.get("HUGGINGFACE_HUB_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(repo_id, token=token)
model = AutoModelForSequenceClassification.from_pretrained(repo_id, token=token)
model.eval()
```

This approach help avoids GitHub file size limits and avoids Git LFS “pointer file” issues on Streamlit Cloud.


