import requests
from pathlib import Path
import time
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GUARDIAN_API_KEY")
BASE_URL = "https://content.guardianapis.com/search"
# CATEGORIES = ["news", "sport", "commentisfree", "culture"]
CATEGORIES = ["commentisfree", "culture"]
PAGE_SIZE = 20
# PAGES_PER_CATEGORY = 12
SLEEP_BETWEEN_REQUESTS = 0.5

def fetch_and_save_articles(start_page=1):
    for category in CATEGORIES:
        print(f"\nCollecting articles for category: {category}")
        for page in range(start_page, start_page + 700):
            print(f"  Fetching page {page}...")
            params = {
                "api-key": API_KEY,
                "section": category,
                "page": page,
                "page-size": PAGE_SIZE,
                "order-by": "newest",
                "show-fields": "headline,byline,bodyText",
            }

            try:
                response = requests.get(BASE_URL, params=params)
                response.raise_for_status()
                results = response.json()["response"].get("results", [])

                for article in results:
                    save_article(article, category)

                time.sleep(SLEEP_BETWEEN_REQUESTS)
            except Exception as e:
                print(f"Error: {e}")

def save_article(article, category):
    fields = article.get("fields", {})
    body = (fields.get("bodyText") or "").strip()
    if not body:
        return

    section = article.get("sectionName", category.title())
    date = (article.get("webPublicationDate") or "")[:10]
    byline = fields.get("byline", "Unknown Author")
    url = article.get("webUrl", "")
    title = fields.get("headline", article.get("webTitle", "Untitled"))

    header = (
        f"The Guardian | {section} | {date}\n"
        f"By {byline}\n"
        f"{url}\n\n"
        f"{title}\n"
        f"{'-' * 60}\n"
    )

    out_dir = Path("data") / category
    out_dir.mkdir(parents=True, exist_ok=True)

    article_id = article["id"].replace("/", "_")
    out_path = out_dir / f"{article_id}.txt"

    if out_path.exists():
        print(f"    Skipping (already exists): {article_id}")
        return

    out_path.write_text(f"{header}{body}\n", encoding="utf-8")
    print(f"    Saved: {article_id}")

if __name__ == "__main__":
    fetch_and_save_articles(121) # run with 121 (109 done)