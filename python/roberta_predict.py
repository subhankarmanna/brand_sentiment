# import torch
# import numpy as np
# from pathlib import Path
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
#
# # =====================================================
# # CONFIG
# # =====================================================
# MODEL_OPTIONS = {
#     "1": "roberta",
#     "2": "distilroberta",
#     "3": "bert",
#     "4": "albert"
# }
#
# MODEL_MAP = {
#     "roberta": "roberta-base",
#     "distilroberta": "distilroberta-base",
#     "bert": "bert-base-uncased",
#     "albert": "albert-base-v2"
# }
#
# MODEL_BASE = Path("models")
# MAX_LEN = 64
# labels = ["Negative", "Neutral", "Positive"]
#
# # Cache for fast switching
# MODEL_CACHE = {}
# LAST_USED = None
#
#
# # =====================================================
# # UTIL FUNCTIONS
# # =====================================================
# def get_versions(model_name):
#     folders = [
#         d for d in MODEL_BASE.iterdir()
#         if d.is_dir() and d.name.startswith(f"{model_name}_v")
#     ]
#     return sorted(folders, key=lambda x: int(x.name.split("_v")[1]))
#
#
# def load_model(model_name, version_path):
#     key = f"{model_name}_{version_path.name}"
#
#     if key in MODEL_CACHE:
#         print("⚡ Loaded from cache")
#         return MODEL_CACHE[key]
#
#     base_model = MODEL_MAP[model_name]
#
#     tokenizer = AutoTokenizer.from_pretrained(base_model)
#     model = AutoModelForSequenceClassification.from_pretrained(
#         version_path / "final_model"
#     )
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     model.eval()
#
#     MODEL_CACHE[key] = (tokenizer, model, device)
#     return tokenizer, model, device
#
#
# def predict(text, tokenizer, model, device):
#     inputs = tokenizer(
#         text,
#         return_tensors="pt",
#         truncation=True,
#         padding=True,
#         max_length=MAX_LEN
#     ).to(device)
#
#     with torch.no_grad():
#         outputs = model(**inputs)
#         probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
#
#     pred = np.argmax(probs)
#     return labels[pred], probs
#
#
# def bar(p):
#     return "█" * int(p * 30)
#
#
# # =====================================================
# # MAIN PROGRAM
# # =====================================================
# while True:
#
#     print("\n==============================")
#     print("        MODEL SELECT")
#     print("==============================")
#
#     if LAST_USED:
#         print(f"Last used → {LAST_USED}")
#
#     print("1 → Roberta")
#     print("2 → DistilRoberta")
#     print("3 → BERT")
#     print("4 → Albert")
#     print("Type 'exit' to close")
#
#     choice = input("Choice: ").strip().lower()
#
#     if choice == "exit":
#         print("Program closed.")
#         break
#
#     if choice not in MODEL_OPTIONS:
#         print("Invalid choice!")
#         continue
#
#     model_name = MODEL_OPTIONS[choice]
#
#     # Show versions
#     versions = get_versions(model_name)
#
#     if not versions:
#         print("No trained versions found!")
#         continue
#
#     print(f"\nAvailable versions for {model_name}:")
#     for i, v in enumerate(versions, 1):
#         print(f"{i} → {v.name}")
#
#     print("Type 'back' to reselect model")
#
#     v_choice = input("Select version: ").strip().lower()
#
#     if v_choice == "back":
#         continue
#
#     if not v_choice.isdigit() or int(v_choice) > len(versions):
#         print("Invalid version!")
#         continue
#
#     version_path = versions[int(v_choice) - 1]
#     LAST_USED = f"{model_name} | {version_path.name}"
#
#     print(f"\nLoading {model_name} - {version_path.name}")
#
#     tokenizer, model, device = load_model(model_name, version_path)
#
#     print("Loaded on:", device)
#
#     # ==============================
#     # PREDICTION LOOP
#     # ==============================
#     print("\nType text to predict")
#     print("exit → change model")
#     print("quit → full close")
#
#     while True:
#         text = input("\nText: ").strip()
#
#         if text.lower() == "exit":
#             break
#
#         if text.lower() == "quit":
#             print("Program closed.")
#             exit()
#
#         label, probs = predict(text, tokenizer, model, device)
#         neg, neu, pos = probs
#
#         print("\n=========== RESULT ===========")
#         print("Prediction :", label)
#         print("Confidence :", round(max(probs) * 100, 2), "%")
#
#         print("\nScores:")
#         print(f"Negative : {neg:.4f} {bar(neg)}")
#         print(f"Neutral  : {neu:.4f} {bar(neu)}")
#         print(f"Positive : {pos:.4f} {bar(pos)}")







#=======================================================================








# import warnings
# warnings.filterwarnings("ignore")
# import torch
# import numpy as np
# from pathlib import Path
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# # =====================================================
# # CONFIG
# # =====================================================
# MODEL_OPTIONS = {
#     "1": "roberta",
#     "2": "distilroberta",
#     "3": "bert",
#     "4": "albert"
# }

# MODEL_MAP = {
#     "roberta": "roberta-base",
#     "distilroberta": "distilroberta-base",
#     "bert": "bert-base-uncased",
#     "albert": "albert-base-v2"
# }

# MODEL_BASE = Path("models")
# MAX_LEN = 64
# labels = ["Negative", "Neutral", "Positive"]

# MODEL_CACHE = {}
# LAST_USED = None


# # =====================================================
# # UTIL
# # =====================================================
# def get_versions(model_name):
#     folders = [
#         d for d in MODEL_BASE.iterdir()
#         if d.is_dir() and d.name.startswith(f"{model_name}_v")
#     ]
#     return sorted(folders, key=lambda x: int(x.name.split("_v")[1]))


# def load_model(model_name, version_path):
#     key = f"{model_name}_{version_path.name}"

#     if key in MODEL_CACHE:
#         return MODEL_CACHE[key]

#     base_model = MODEL_MAP[model_name]

#     tokenizer = AutoTokenizer.from_pretrained(base_model)
#     model = AutoModelForSequenceClassification.from_pretrained(
#         version_path / "final_model"
#     )

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     model.eval()

#     MODEL_CACHE[key] = (tokenizer, model, device)
#     return tokenizer, model, device


# def predict(text, tokenizer, model, device):
#     inputs = tokenizer(
#         text,
#         return_tensors="pt",
#         truncation=True,
#         padding=True,
#         max_length=MAX_LEN
#     ).to(device)

#     with torch.no_grad():
#         outputs = model(**inputs)
#         probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

#     pred = np.argmax(probs)
#     return labels[pred], probs


# def bar(p):
#     return "█" * int(p * 25)


# # =====================================================
# # COMPARE MODE
# # =====================================================
# def compare_all_models(text):
#     print("\n====== MODEL COMPARISON ======\n")

#     for model_name in MODEL_MAP.keys():
#         versions = get_versions(model_name)
#         if not versions:
#             continue

#         latest = versions[-1]
#         tokenizer, model, device = load_model(model_name, latest)

#         label, probs = predict(text, tokenizer, model, device)
#         conf = round(max(probs) * 100, 2)

#         print(f"{model_name.upper()} ({latest.name})")
#         print(f"Prediction : {label}  |  Confidence : {conf}%")
#         print(
#             f"N:{probs[0]:.2f} {bar(probs[0])}  "
#             f"Ne:{probs[1]:.2f} {bar(probs[1])}  "
#             f"P:{probs[2]:.2f} {bar(probs[2])}"
#         )
#         print("-" * 40)


# # =====================================================
# # MAIN LOOP
# # =====================================================
# while True:

#     print("\n==============================")
#     print("        MODEL SELECT")
#     print("==============================")

#     if LAST_USED:
#         print(f"Last used → {LAST_USED}")

#     print("1 → Roberta")
#     print("2 → DistilRoberta")
#     print("3 → BERT")
#     print("4 → Albert")
#     print("Type 'exit' to close")

#     choice = input("Choice: ").strip().lower()

#     if choice == "exit":
#         print("Program closed.")
#         break

#     if choice not in MODEL_OPTIONS:
#         print("Invalid choice!")
#         continue

#     model_name = MODEL_OPTIONS[choice]
#     versions = get_versions(model_name)

#     if not versions:
#         print("No trained versions found!")
#         continue

#     print(f"\nAvailable versions for {model_name}:")
#     for i, v in enumerate(versions, 1):
#         print(f"{i} → {v.name}")

#     print("Type 'back' to reselect model")

#     v_choice = input("Select version: ").strip().lower()

#     if v_choice == "back":
#         continue

#     if not v_choice.isdigit() or int(v_choice) > len(versions):
#         print("Invalid version!")
#         continue

#     version_path = versions[int(v_choice) - 1]
#     LAST_USED = f"{model_name} | {version_path.name}"

#     tokenizer, model, device = load_model(model_name, version_path)

#     print("\nLoaded on:", device)
#     print("\nType text to predict")
#     print("exit → change model")
#     print("compare → test all models")
#     print("quit → full close")

#     # ==============================
#     # PREDICTION LOOP
#     # ==============================
#     while True:
#         text = input("\nText: ").strip()

#         if text.lower() == "exit":
#             break

#         if text.lower() == "quit":
#             print("Program closed.")
#             exit()

#         if text.lower() == "compare":
#             sample = input("Enter text to compare: ")
#             compare_all_models(sample)
#             continue

#         label, probs = predict(text, tokenizer, model, device)
#         neg, neu, pos = probs

#         print("\n=========== RESULT ===========")
#         print("Prediction :", label)
#         print("Confidence :", round(max(probs) * 100, 2), "%")

#         print("\nScores:")
#         print(f"Negative : {neg:.4f} {bar(neg)}")
#         print(f"Neutral  : {neu:.4f} {bar(neu)}")
#         print(f"Positive : {pos:.4f} {bar(pos)}")





import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =====================================================
# CONFIG
# =====================================================
MODEL_MAP = {
    "roberta": "roberta-base",
    "distilroberta": "distilroberta-base",
    "bert": "bert-base-uncased",
    "albert": "albert-base-v2"
}

MODEL_BASE = Path(__file__).parent / "models"
MAX_LEN = 64
labels = ["Negative", "Neutral", "Positive"]

MODEL_CACHE = {}

# =====================================================
# UTIL
# =====================================================
def get_versions(model_name):
    folders = [
        d for d in MODEL_BASE.iterdir()
        if d.is_dir() and d.name.startswith(f"{model_name}_v")
    ]
    return sorted(folders, key=lambda x: int(x.name.split("_v")[1]))


def load_model(model_name, version_path):
    key = f"{model_name}_{version_path.name}"

    if key in MODEL_CACHE:
        return MODEL_CACHE[key]

    base_model = MODEL_MAP[model_name]

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        version_path / "final_model"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    MODEL_CACHE[key] = (tokenizer, model, device)
    return tokenizer, model, device


# =====================================================
# SINGLE PREDICT (BACKEND FRIENDLY)
# =====================================================
def predict(text):

    # Default: Roberta latest
    model_name = "roberta"
    versions = get_versions(model_name)

    if not versions:
        raise Exception("No trained model found")

    latest = versions[-1]
    tokenizer, model, device = load_model(model_name, latest)

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
    return labels[pred], probs


# =====================================================
# COMPARE ALL MODELS
# =====================================================
def compare_all_models(text):

    results = []

    for model_name in MODEL_MAP.keys():
        versions = get_versions(model_name)
        if not versions:
            continue

        latest = versions[-1]
        tokenizer, model, device = load_model(model_name, latest)

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
            "version": latest.name,
            "prediction": labels[pred],
            "confidence": float(max(probs)),
            "negative": float(probs[0]),
            "neutral": float(probs[1]),
            "positive": float(probs[2]),
        })

    return results









