import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from huggingface_hub import login

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    login(token=HF_TOKEN)


MAX_LEN = 64
labels = ["Negative", "Neutral", "Positive"]


MODEL_REPOS = {
    "roberta": "subhankarmannayfy/brand-roberta",
    "distilroberta": "subhankarmannayfy/brand-distilroberta",
    "bert": "subhankarmannayfy/brand-bert",
    "albert": "subhankarmannayfy/brand-albert"
}


BASE_TOKENIZERS = {
    "roberta": "roberta-base",
    "distilroberta": "distilroberta-base",
    "bert": "bert-base-uncased",
    "albert": "albert-base-v2"
}

MODEL_CACHE = {}




def load_model(model_name):
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    print(f"🔄 Loading {model_name} from HuggingFace...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZERS[model_name])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_REPOS[model_name],
        token=HF_TOKEN
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    MODEL_CACHE[model_name] = (tokenizer, model, device)
    return tokenizer, model, device


def predict(text, model_name="roberta"):
    tokenizer, model, device = load_model(model_name)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    pred = np.argmax(probs)
    return labels[pred], probs.tolist()


def compare_all_models(text):
    results = []

    for model_name in MODEL_REPOS.keys():
        tokenizer, model, device = load_model(model_name)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LEN
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        pred = np.argmax(probs)

        results.append({
            "model": model_name,
            "prediction": labels[pred],
            "confidence": float(max(probs)),
            "negative": float(probs[0]),
            "neutral": float(probs[1]),
            "positive": float(probs[2]),
        })

    return results