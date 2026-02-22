import json
import time
from pathlib import Path
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from config import YOUTUBE_API_KEY

DATA_DIR = Path("data_raw")
DATA_DIR.mkdir(exist_ok=True)

MAX_VIDEOS = 500
COMMENTS_PER_PAGE = 100
SLEEP = 0.2


def get_video_ids(youtube, keyword):
    ids = set()
    queries = [
        keyword,
        f"{keyword} review",
        f"{keyword} experience",
        f"{keyword} problem",
    ]

    for q in queries:
        token = None

        while len(ids) < MAX_VIDEOS:
            req = youtube.search().list(
                q=q,
                part="id",
                type="video",
                maxResults=50,
                order="relevance",
                pageToken=token
            )
            res = req.execute()

            for item in res.get("items", []):
                ids.add(item["id"]["videoId"])

            token = res.get("nextPageToken")
            if not token:
                break

            time.sleep(SLEEP)

    return list(ids)[:MAX_VIDEOS]


def fetch_comments(youtube, video_id, keyword, out_path):
    next_page = None
    count = 0

    while True:
        try:
            req = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=COMMENTS_PER_PAGE,
                pageToken=next_page,
                textFormat="plainText"
            )
            res = req.execute()

        except HttpError:
            # comments disabled / API issue → skip
            return 0

        for item in res.get("items", []):
            c = item["snippet"]["topLevelComment"]["snippet"]

            record = {
                "source": "youtube",
                "keyword": keyword,
                "video_id": video_id,
                "id": item["id"],
                "created_at": c["publishedAt"],
                "text": c["textDisplay"]
            }

            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            count += 1

        next_page = res.get("nextPageToken")
        if not next_page:
            break

        time.sleep(SLEEP)

    return count


def crawl(keyword):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    out_path = DATA_DIR / f"youtube_{keyword}.jsonl"

    videos = get_video_ids(youtube, keyword)
    print("Videos found:", len(videos))

    total = 0
    for i, vid in enumerate(videos, 1):
        n = fetch_comments(youtube, vid, keyword, out_path)
        total += n
        print(f"[{i}/{len(videos)}] {vid} → {n} comments | total={total}")

    print("\nSaved to:", out_path)
    print("Total comments:", total)


if __name__ == "__main__":
    kw = input("Enter keyword: ").strip()
    crawl(kw)
