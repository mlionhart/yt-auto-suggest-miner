"""
Collects complaint/need-style posts and comments from Hacker News via the free Algolia HN Search API
(https://hn.algolia.com/api). No key, no cost.

For each phrase in hn_queries.txt we search comments and stories, page through the results, and append
one JSON line per unique post to results/hn_complaints.jsonl. Re-running skips posts already saved
(keyed on HN's objectID), so it is safe to resume or run again later.
"""

import argparse
import html
import json
import os
import re
import time
from datetime import datetime, timezone

import requests

SEARCH_URL = "https://hn.algolia.com/api/v1/search"
SEARCH_BY_DATE_URL = "https://hn.algolia.com/api/v1/search_by_date"
HITS_PER_PAGE = 100  # Algolia's maximum per page


def load_queries(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_seen_ids(output_path):
    seen = set()
    if not os.path.exists(output_path):
        return seen
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def clean_text(raw):
    """HN comment text is HTML. Strip tags and decode entities so the saved text is plain."""
    if not raw:
        return ""
    text = re.sub(r"<[^>]+>", " ", raw.replace("<p>", "\n"))
    return re.sub(r"[ \t]+", " ", html.unescape(text)).strip()


def search(session, query, tag, page, since=None):
    # Wrapping the query in quotes makes Algolia match the exact phrase instead of any of the words.
    params = {"query": f'"{query}"', "tags": tag, "hitsPerPage": HITS_PER_PAGE, "page": page}
    url = SEARCH_URL
    if since:
        # Date mode: only posts on/after `since`, newest first (the default ranking is by relevance).
        ts = int(datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
        params["numericFilters"] = f"created_at_i>={ts}"
        url = SEARCH_BY_DATE_URL
    response = session.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def to_record(hit, query, tag):
    is_comment = tag == "comment"
    text = clean_text(hit.get("comment_text") if is_comment else (hit.get("story_text") or hit.get("title")))
    return {
        "id": hit["objectID"],
        "type": tag,
        "query": query,
        "text": text,
        "story_title": hit.get("story_title") if is_comment else hit.get("title"),
        "author": hit.get("author"),
        "points": hit.get("points"),
        "created_at": hit.get("created_at"),
        "url": f"https://news.ycombinator.com/item?id={hit['objectID']}",
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="Collect need/complaint posts from Hacker News (Algolia search).")
    parser.add_argument("--queries", default="hn_queries.txt", help="Phrases to search, one per line.")
    parser.add_argument("--output", default="results/hn_complaints.jsonl")
    parser.add_argument("--max-pages", type=int, default=3, help="Pages of 100 results per phrase per type.")
    parser.add_argument("--tags", nargs="+", default=["comment", "story"], choices=["comment", "story"])
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between requests.")
    parser.add_argument("--since", help="Only posts on/after this date (YYYY-MM-DD), newest first.")
    args = parser.parse_args()

    queries = load_queries(args.queries)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    seen = load_seen_ids(args.output)
    session = requests.Session()

    new_total = 0
    with open(args.output, "a", encoding="utf-8") as out:
        for query in queries:
            for tag in args.tags:
                new_here = 0
                for page in range(args.max_pages):
                    try:
                        data = search(session, query, tag, page, args.since)
                    except requests.RequestException as e:
                        print(f"  ERROR '{query}' ({tag}, page {page}): {e}")
                        break
                    for hit in data.get("hits", []):
                        rec = to_record(hit, query, tag)
                        if rec["id"] in seen or not rec["text"]:
                            continue
                        seen.add(rec["id"])
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        new_here += 1
                    out.flush()
                    time.sleep(args.delay)
                    if page + 1 >= data.get("nbPages", 0):
                        break
                new_total += new_here
                print(f"'{query}' [{tag}]: {new_here} new")

    print(f"Done. {new_total} new posts saved to {args.output} (total unique: {len(seen)})")


if __name__ == "__main__":
    main()
