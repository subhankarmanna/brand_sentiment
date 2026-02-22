# import re
# import pandas as pd
# import numpy as np
# from pathlib import Path
# from datasets import Dataset
# from sklearn.model_selection import train_test_split
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSequenceClassification,
#     TrainingArguments,
#     Trainer,
#     set_seed
# )
#
# def main():
#     set_seed(42)
#
#     # ================== PATH AUTO ==================
#
#     PROC_BASE = Path("data_processed")
#
#     def get_latest_version_folder(base: Path):
#         versions = [d for d in base.iterdir() if d.is_dir() and d.name.startswith("v")]
#         nums = [int(re.findall(r"\d+", d.name)[0]) for d in versions]
#         latest = max(nums)
#         return base / f"v{latest}"
#
#     DATA_DIR = get_latest_version_folder(PROC_BASE)
#     CSV_PATH = list(DATA_DIR.glob("*_train.csv"))[0]
#     print("Using dataset:", CSV_PATH)
#
#     # ================== MODEL VERSION AUTO ==================
#
#     BASE_DIR = Path("models")
#     BASE_DIR.mkdir(exist_ok=True)
#
#     existing = [d for d in BASE_DIR.iterdir() if d.is_dir() and d.name.startswith("roberta_v")]
#     version = len(existing) + 1
#
#     RUN_DIR = BASE_DIR / f"roberta_v{version}"
#     CHECKPOINT_DIR = RUN_DIR / "checkpoints"
#     FINAL_DIR = RUN_DIR / "final_model"
#
#     CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
#     FINAL_DIR.mkdir(parents=True, exist_ok=True)
#
#     print(f"\n🚀 Training Version: roberta_v{version}")
#     print("Model folder:", RUN_DIR, "\n")
#
#     # ================== LOAD DATA ==================
#
#     df = pd.read_csv(CSV_PATH)
#
#     train_df, val_df = train_test_split(
#         df, test_size=0.1, stratify=df["label"], random_state=42
#     )
#
#     train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
#     val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))
#
#     # ================== TOKENIZER ==================
#
#     model_name = "roberta-base"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#
#     def tokenize(batch):
#         clean_texts = []
#         for t in batch["text"]:
#             if t is None:
#                 clean_texts.append("")
#             else:
#                 clean_texts.append(str(t))
#
#         return tokenizer(
#             clean_texts,
#             padding="max_length",
#             truncation=True,
#             max_length=64
#         )
#
#     print("Tokenizing dataset (parallel)...")
#
#     train_ds = train_ds.map(tokenize, batched=True, num_proc=4, remove_columns=["text"])
#     val_ds   = val_ds.map(tokenize, batched=True, num_proc=4, remove_columns=["text"])
#
#     train_ds.set_format("torch")
#     val_ds.set_format("torch")
#
#     # ================== MODEL ==================
#
#     model = AutoModelForSequenceClassification.from_pretrained(
#         model_name,
#         num_labels=3
#     )
#
#     # ================== METRICS ==================
#
#     def compute_metrics(p):
#         preds = np.argmax(p.predictions, axis=1)
#         return {"accuracy": (preds == p.label_ids).mean()}
#
#     # ================== TRAINING ==================
#
#     args = TrainingArguments(
#         output_dir=str(CHECKPOINT_DIR),
#         per_device_train_batch_size=8,
#         per_device_eval_batch_size=16,
#         gradient_accumulation_steps=2,
#         evaluation_strategy="epoch",
#         num_train_epochs=2,
#         fp16=True,
#         logging_steps=200,
#         save_strategy="epoch",
#         report_to="none"
#     )
#
#     trainer = Trainer(
#         model=model,
#         args=args,
#         train_dataset=train_ds,
#         eval_dataset=val_ds,
#         tokenizer=tokenizer,
#         compute_metrics=compute_metrics
#     )
#
#     trainer.train()
#
#     # ================== SAVE FINAL ==================
#
#     trainer.save_model(str(FINAL_DIR))
#     tokenizer.save_pretrained(str(FINAL_DIR))
#
#     print("\n✅ Final model saved at:", FINAL_DIR)
#
#
# # ========= WINDOWS FIX =========
# if __name__ == "__main__":
#     main()









#
# import re
# import numpy as np
# from pathlib import Path
# from datasets import load_from_disk
# from transformers import (
#     AutoModelForSequenceClassification,
#     TrainingArguments,
#     Trainer,
#     AutoTokenizer,
#     set_seed
# )
#
# def main():
#     set_seed(42)
#
#     # ========= LOAD PRE-TOKENIZED DATASET =========
#     print("Loading pre-tokenized dataset...")
#     ds = load_from_disk("tokenized_playstore_ds")
#
#     split = ds.train_test_split(test_size=0.1, seed=42)
#     train_ds = split["train"]
#     val_ds   = split["test"]
#
#     train_ds.set_format("torch")
#     val_ds.set_format("torch")
#
#     print("Train size:", len(train_ds))
#     print("Val size:", len(val_ds))
#
#     # ========= MODEL VERSION AUTO =========
#     BASE_DIR = Path("models")
#     BASE_DIR.mkdir(exist_ok=True)
#
#     existing = [d for d in BASE_DIR.iterdir() if d.is_dir() and d.name.startswith("roberta_v")]
#     version = len(existing) + 1
#
#     RUN_DIR = BASE_DIR / f"roberta_v{version}"
#     CHECKPOINT_DIR = RUN_DIR / "checkpoints"
#     FINAL_DIR = RUN_DIR / "final_model"
#
#     CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
#     FINAL_DIR.mkdir(parents=True, exist_ok=True)
#
#     print(f"\n🚀 Training Version: roberta_v{version}")
#     print("Model folder:", RUN_DIR, "\n")
#
#     # ========= MODEL =========
#     model_name = "roberta-base"
#     model = AutoModelForSequenceClassification.from_pretrained(
#         model_name,
#         num_labels=3
#     )
#
#     # ========= METRICS =========
#     def compute_metrics(p):
#         preds = np.argmax(p.predictions, axis=1)
#         return {"accuracy": (preds == p.label_ids).mean()}
#
#     # ========= TRAINING ARGS =========
#     args = TrainingArguments(
#         output_dir=str(CHECKPOINT_DIR),
#         per_device_train_batch_size=16,   # ab GPU fast chalega
#         per_device_eval_batch_size=32,
#         gradient_accumulation_steps=1,
#         evaluation_strategy="epoch",
#         save_strategy="epoch",
#         num_train_epochs=2,
#         fp16=True,
#         logging_steps=200,
#         report_to="none"
#     )
#
#     trainer = Trainer(
#         model=model,
#         args=args,
#         train_dataset=train_ds,
#         eval_dataset=val_ds,
#         compute_metrics=compute_metrics
#     )
#
#     trainer.train()
#
#     # ========= SAVE FINAL =========
#     trainer.save_model(str(FINAL_DIR))
#     print("\n✅ Final model saved at:", FINAL_DIR)
#
#
# if __name__ == "__main__":
#     main()




#
#
# import numpy as np
# from pathlib import Path
# from datasets import load_from_disk
# from transformers import (
#     AutoModelForSequenceClassification,
#     TrainingArguments,
#     Trainer,
#     set_seed
# )
#
# def main():
#     set_seed(42)
#
#     # ========= LOAD PRE-TOKENIZED DATASET =========
#     print("Loading pre-tokenized dataset...")
#     ds = load_from_disk("tokenized_playstore_ds").with_format("torch")
#
#     print("Loading dataset fully into RAM...")
#     ds = ds.map(lambda x: x, batched=True, batch_size=10000)
#     print("Done loading into RAM")
#
#     split = ds.train_test_split(test_size=0.1, seed=42)
#     train_ds = split["train"]
#     val_ds   = split["test"]
#
#     train_ds.set_format("torch")
#     val_ds.set_format("torch")
#
#     print("Train size:", len(train_ds))
#     print("Val size:", len(val_ds))
#
#     # ========= MODEL VERSION AUTO =========
#     BASE_DIR = Path("models")
#     BASE_DIR.mkdir(exist_ok=True)
#
#     existing = [d for d in BASE_DIR.iterdir() if d.is_dir() and d.name.startswith("roberta_v")]
#     version = len(existing) + 1
#
#     RUN_DIR = BASE_DIR / f"roberta_v{version}"
#     CHECKPOINT_DIR = RUN_DIR / "checkpoints"
#     FINAL_DIR = RUN_DIR / "final_model"
#
#     CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
#     FINAL_DIR.mkdir(parents=True, exist_ok=True)
#
#     print(f"\n🚀 Training Version: roberta_v{version}")
#     print("Model folder:", RUN_DIR, "\n")
#
#     # ========= MODEL =========
#     # model = AutoModelForSequenceClassification.from_pretrained(
#     #     "roberta-base",
#     #     num_labels=3
#     # )
#     #12 step pass layer
#
#     model = AutoModelForSequenceClassification.from_pretrained(
#         "distilroberta-base",
#         num_labels=3
#     )
#     #6 layer
#
#     # ========= METRICS =========
#     def compute_metrics(p):
#         preds = np.argmax(p.predictions, axis=1)
#         return {"accuracy": (preds == p.label_ids).mean()}
#
#     # ========= TRAINING ARGS (🔥 DATA PIPE FIX) =========
#     args = TrainingArguments(
#         output_dir=str(CHECKPOINT_DIR),
#
#         per_device_train_batch_size=32,
#         per_device_eval_batch_size=64,
#         gradient_accumulation_steps=1,
#
#         dataloader_num_workers=8,
#         dataloader_pin_memory=True,
#         dataloader_prefetch_factor=4,
#
#         eval_strategy="epoch",
#         save_strategy="epoch",
#         num_train_epochs=2,
#         fp16=True,
#         logging_steps=200,
#         report_to="none"
#     )
#
#     trainer = Trainer(
#         model=model,
#         args=args,
#         train_dataset=train_ds,
#         eval_dataset=val_ds,
#         compute_metrics=compute_metrics
#     )
#
#     trainer.train()
#
#     # ========= SAVE FINAL =========
#     trainer.save_model(str(FINAL_DIR))
#     print("\n✅ Final model saved at:", FINAL_DIR)
#
#
# if __name__ == "__main__":
#     main()
#








import re
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed
)

# =====================================================
# 🔥 CHANGE ONLY THIS
# =====================================================
MODEL_NAME = "distilroberta"

MODEL_MAP = {
    "roberta": "roberta-base",
    "distilroberta": "distilroberta-base",
    "bert": "bert-base-uncased",
    "albert": "albert-base-v2"
}

BASE_MODEL = MODEL_MAP[MODEL_NAME]

PROC_BASE = Path("data_processed")
MODEL_BASE = Path("models")
MAX_LEN = 64
NUM_LABELS = 3


# =====================================================
# AUTO PICK LATEST CSV
# =====================================================
def get_latest_csv():
    versions = [d for d in PROC_BASE.iterdir() if d.is_dir() and d.name.startswith("v")]
    if not versions:
        raise RuntimeError("No data_processed/vX found")

    nums = [int(re.findall(r'\d+', d.name)[0]) for d in versions]
    latest_v = max(nums)

    latest_dir = PROC_BASE / f"v{latest_v}"
    csv_files = list(latest_dir.glob("*.csv"))

    if not csv_files:
        raise RuntimeError("No CSV in latest processed folder")

    latest_csv = max(csv_files, key=lambda f: f.stat().st_mtime)
    print("Using CSV:", latest_csv)
    return latest_csv


# =====================================================
# MAIN
# =====================================================
def main():
    set_seed(42)

    # -------- LOAD CSV (RAM SAFE) --------
    csv_path = get_latest_csv()
    df = pd.read_csv(csv_path, usecols=["text", "label"])

    dataset = Dataset.from_pandas(df)

    # -------- TOKENIZER --------
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize(batch):
        texts = [str(t) if t else "" for t in batch["text"]]
        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN
        )

    print("Tokenizing...")
    dataset = dataset.map(
        tokenize,
        batched=True,
        num_proc=4,
        remove_columns=["text"]
    )

    # -------- SPLIT --------
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"].with_format("torch")
    val_ds   = split["test"].with_format("torch")

    print("Train:", len(train_ds))
    print("Val  :", len(val_ds))

    # -------- AUTO VERSION (FIXED LOGIC) --------
    MODEL_BASE.mkdir(exist_ok=True)

    existing = [
        d for d in MODEL_BASE.iterdir()
        if d.is_dir() and d.name.startswith(f"{MODEL_NAME}_v")
    ]

    if existing:
        versions = [int(re.findall(r'\d+', d.name)[0]) for d in existing]
        version = max(versions) + 1
    else:
        version = 1

    RUN_DIR = MODEL_BASE / f"{MODEL_NAME}_v{version}"
    CKPT_DIR = RUN_DIR / "checkpoints"
    FINAL_DIR = RUN_DIR / "final_model"

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 Training {MODEL_NAME} v{version}")

    # -------- MODEL --------
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_LABELS
    )

    # -------- METRICS --------
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        acc = (preds == p.label_ids).mean()
        return {"accuracy": acc}

    # -------- TRAINING ARGS --------
    args = TrainingArguments(
        output_dir=str(CKPT_DIR),

        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,

        num_train_epochs=2,

        eval_strategy="epoch",
        save_strategy="epoch",

        fp16=torch.cuda.is_available(),

        logging_steps=200,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(str(FINAL_DIR))

    print("\nSaved at:", FINAL_DIR)


if __name__ == "__main__":
    main()




