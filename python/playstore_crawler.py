import requests
from google_play_scraper import reviews, Sort
import json
import re
from pathlib import Path

MAX_LIMIT = 3_500_000
BASE_DIR = Path("data_raw")
BASE_DIR.mkdir(exist_ok=True)


def get_next_version_folder():
    existing = [d for d in BASE_DIR.iterdir() if d.is_dir() and d.name.startswith("v")]
    if not existing:
        return BASE_DIR / "v1"

    nums = []
    for d in existing:
        m = re.match(r"v(\d+)", d.name)
        if m:
            nums.append(int(m.group(1)))

    next_v = max(nums) + 1
    return BASE_DIR / f"v{next_v}"


def get_app_id_from_web(keyword):
    url = f"https://play.google.com/store/search?q={keyword}&c=apps"
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers).text

    match = re.search(r'href="/store/apps/details\?id=(.*?)"', html)
    if not match:
        raise Exception("App ID not found")

    return match.group(1)


def fetch_all_reviews(keyword):
    version_dir = get_next_version_folder()
    version_dir.mkdir()

    print(f"\nSaving in folder: {version_dir}")

    app_id = get_app_id_from_web(keyword)
    print(f"Found app_id: {app_id}")

    all_reviews = []
    continuation_token = None

    while True:
        result, continuation_token = reviews(
            app_id,
            lang="en",
            country="in",
            sort=Sort.NEWEST,
            count=200,
            continuation_token=continuation_token
        )

        if not result:
            break

        for r in result:
            all_reviews.append({
                "source": "playstore",
                "keyword": keyword,
                "id": r["reviewId"],
                "created_at": r["at"].strftime("%Y-%m-%d %H:%M:%S"),
                "text": r["content"],
                "rating": r["score"]
            })

            if len(all_reviews) >= MAX_LIMIT:
                break

        print(f"Fetched so far: {len(all_reviews)}")

        if continuation_token is None or len(all_reviews) >= MAX_LIMIT:
            break

    out_file = version_dir / f"playstore_{keyword}.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_reviews, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(all_reviews)} reviews → {out_file}")


if __name__ == "__main__":
    kw = input("Enter app keyword: ").strip()
    fetch_all_reviews(kw)
