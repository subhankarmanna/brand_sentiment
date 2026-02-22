from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer


def main():
    print("Loading CSV...")
    df = pd.read_csv("data_processed/v2/playstore_zomato_train.csv")

    ds = Dataset.from_pandas(df)

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    def tok(batch):
        texts = [str(t) if t else "" for t in batch["text"]]
        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=64
        )

    print("Tokenizing (ONE TIME)...")

    ds = ds.map(
        tok,
        batched=True,
        num_proc=4,
        remove_columns=["text"]
    )

    print("Saving tokenized dataset...")
    ds.save_to_disk("tokenized_playstore_ds")

    print("Done ✅")



if __name__ == "__main__":
    main()
