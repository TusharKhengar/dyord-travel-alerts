"""
DYORD news pipeline: fetch news for a location, resolve real article URLs,
pull article text, and classify relevance/severity for travelers using a
local LLM via Ollama (no API key, runs on-device, free).

Setup:
    pip install -r requirements.txt
    Install Ollama (https://ollama.com) and run: ollama pull llama3.2:3b

Usage:
    python dyord_pipeline.py "Mumbai"
    python dyord_pipeline.py "Mumbai" --max-articles 15 --out mumbai_alerts.csv
"""
import argparse
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from gnews import GNews
from googlenewsdecoder import gnewsdecoder

from llm_backend import chat_json

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DYORD-bot/1.0)"}
REQUEST_TIMEOUT = 10


def fetch_news(location: str, max_articles: int = 20) -> list[dict]:
    google_news = GNews(max_results=max_articles)
    return google_news.get_news(f"{location} news")


def resolve_article_url(google_news_url: str) -> str | None:
    try:
        result = gnewsdecoder(google_news_url, interval=1)
    except Exception:
        return None

    if result.get("status"):
        return result.get("decoded_url")

    return None


def extract_article_text(url: str, max_chars: int = 4000) -> str:
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException:
        return ""

    soup = BeautifulSoup(response.text, "html.parser")

    article_tag = soup.find("article")
    if article_tag:
        text = article_tag.get_text(separator=" ", strip=True)
        if text:
            return text[:max_chars]

    paragraphs = soup.find_all("p")
    if paragraphs:
        text = " ".join(p.get_text(strip=True) for p in paragraphs)
        if text:
            return text[:max_chars]

    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        return meta_desc["content"][:max_chars]

    return ""


CLASSIFY_SYSTEM_PROMPT = (
    "You are a travel-safety analyst. Given a news article's title and text, "
    "judge whether it would change what a traveler currently in or heading to "
    "the location it concerns should expect, plan for, or avoid. Relevant "
    "means: safety incidents, unrest/protests affecting movement, transport "
    "disruptions (flights/trains/roads/strikes), natural disasters or extreme "
    "weather, health/disease alerts, major closures, or scams/crimes "
    "specifically targeting tourists. NOT relevant: local politics, "
    "legal/court rulings, business/legal-industry news, celebrity or "
    "entertainment news, human-interest stories, and routine local "
    "administration — even if travelers might find them interesting, they "
    "don't change a trip. When in doubt, prefer 'none'. Respond ONLY with a "
    'JSON object: {"severity": "none"|"low"|"medium"|"high", "reason": "one '
    'sentence explanation"}.'
)


def classify_article(title: str, text: str) -> dict:
    content = f"Title: {title}\n\nArticle text: {text or '(no article text available, use title only)'}"
    try:
        result = chat_json(CLASSIFY_SYSTEM_PROMPT, content)
        severity = result.get("severity", "none")
        return {
            "relevant": severity != "none",
            "severity": severity,
            "reason": result.get("reason", ""),
        }
    except Exception as exc:
        return {"relevant": False, "severity": "none", "reason": f"classification failed: {exc}"}


def analyze_location_news(location: str, max_articles: int = 20) -> pd.DataFrame:
    news = fetch_news(location, max_articles)
    rows = []
    seen_titles = set()

    for article in news:
        title = article.get("title", "")
        if title in seen_titles:
            continue
        seen_titles.add(title)

        resolved_url = resolve_article_url(article["url"])
        article_text = extract_article_text(resolved_url) if resolved_url else ""

        classification = classify_article(title, article_text)

        rows.append(
            {
                "title": title,
                "published_date": article.get("published date", ""),
                "publisher": (article.get("publisher") or {}).get("title", ""),
                "url": resolved_url or article["url"],
                "relevant": classification["relevant"],
                "severity": classification["severity"],
                "reason": classification["reason"],
            }
        )
        time.sleep(0.3)  # be polite to both the news sites and the API rate limit

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="DYORD traveler news impact analyzer")
    parser.add_argument("location", help="City or place to check news for, e.g. 'Mumbai'")
    parser.add_argument("--max-articles", type=int, default=20)
    parser.add_argument("--out", default=None, help="Optional CSV path to save results")
    args = parser.parse_args()

    df = analyze_location_news(args.location, args.max_articles)

    relevant = df[df["relevant"]].sort_values(
        by="severity", key=lambda s: s.map({"high": 3, "medium": 2, "low": 1, "none": 0})
    , ascending=False)

    print(f"\n{len(relevant)} of {len(df)} articles flagged as relevant to travelers in {args.location}:\n")
    for _, row in relevant.iterrows():
        print(f"[{row['severity'].upper()}] {row['title']}")
        print(f"    {row['reason']}")
        print(f"    {row['url']}\n")

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"Full results saved to {args.out}")


if __name__ == "__main__":
    main()
