import json
import pandas as pd
from pathlib import Path
import re

RAW_BASE = Path("data_raw")
PROC_BASE = Path("data_processed")
PROC_BASE.mkdir(exist_ok=True)


# ---------- Find latest version folder ----------
def get_latest_version_folder(base: Path):
    versions = [d for d in base.iterdir() if d.is_dir() and d.name.startswith("v")]
    if not versions:
        raise Exception("No version folder found in data_raw")

    nums = []
    for d in versions:
        m = re.match(r"v(\d+)", d.name)
        if m:
            nums.append(int(m.group(1)))

    latest = max(nums)
    return base / f"v{latest}", f"v{latest}"


RAW_DIR, version_name = get_latest_version_folder(RAW_BASE)
PROC_DIR = PROC_BASE / version_name
PROC_DIR.mkdir(exist_ok=True)

print(f"Reading from: {RAW_DIR}")
print(f"Saving to: {PROC_DIR}")


# ---------- Helper ----------
def rating_to_label(r):
    if r <= 2:
        return 0
    elif r == 3:
        return 1
    else:
        return 2


def minimal_clean(text: str):
    # DO NOT TOUCH EMOJI
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ---------- Process ----------
for file in RAW_DIR.glob("playstore_*.json"):
    print("Processing:", file.name)

    with open(file, encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for r in data:
        text = r.get("text")
        if not text:
            continue

        text = minimal_clean(text)

        if len(text) == 0:
            continue

        rows.append({
            "text": text,
            "label": rating_to_label(r["rating"])
        })

    df = pd.DataFrame(rows)

    out_file = PROC_DIR / file.name.replace(".json", "_train.csv")
    df.to_csv(out_file, index=False)

    print(f"Saved: {out_file} | Rows: {len(df)}")
