"""
Tags Hacker News posts (from hn_collector.py) with what the person is saying about software and money.

Per post we record: whether it is a software/tool pain point, whether they want a tool, any
willingness-to-pay signal, the product or workflow involved, a short pain summary, and the domain.
Nothing is deleted; every post gets tagged. Resumable by HN id. Reuses helpers from classify.py.
"""

import argparse
import json
import os
from datetime import datetime, timezone

import anthropic

from classify import (
    INPUT_PRICE_PER_MTOK,
    MAX_TOKENS,
    MODEL,
    OUTPUT_PRICE_PER_MTOK,
    chunked,
    estimate_tokens,
    parse_response_json,
    require_api_key,
)

BATCH_SIZE = 25
MAX_TEXT_CHARS = 700  # long comments are cut off to keep cost down; the pain is usually up front

PROMPT_TEMPLATE = """You are analyzing Hacker News posts to find evidence of software demand and willingness to pay.

For each numbered post, return:
- "pain": true if the author describes a real problem, frustration, or unmet need with software, tools, or a work/life workflow that software could address. false for opinions, news, jokes, or off-topic posts.
- "wants_tool": true if they explicitly ask for, wish for, or search for a tool/app/service.
- "pay_signal": one of "pays_for" (says they currently pay for something, ideally with the product or price), "would_pay" (says they would pay for a solution), "refuses_to_pay" (says they would not pay / expects free), or "none".
- "product": the specific product, tool, or workflow involved (under 8 words), or null.
- "pain_summary": the core complaint or need in under 15 words, or null if pain is false.
- "domain": the industry or area (e.g. "accounting", "devops", "healthcare", "personal productivity"), or null.

Judge only what the text says. Do not infer a pay_signal that is not stated.

Return ONLY a JSON array, one object per post, no other text, in this exact shape:
[{{"id": 0, "pain": true, "wants_tool": false, "pay_signal": "none", "product": "...", "pain_summary": "...", "domain": "..."}}, ...]

Include every id from 0 to {max_id} exactly once.

Posts:
{post_list}
"""


def load_posts(path, since):
    posts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if since and r["created_at"][:10] < since:
                continue
            posts.append(r)
    return posts


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def build_prompt(batch):
    items = []
    for i, p in enumerate(batch):
        text = p["text"].replace("\n", " ")[:MAX_TEXT_CHARS]
        items.append(f"{i}: {text}")
    return PROMPT_TEMPLATE.format(max_id=len(batch) - 1, post_list="\n".join(items))


def classify_batch(client, batch):
    response = client.messages.create(
        model=MODEL, max_tokens=MAX_TOKENS, messages=[{"role": "user", "content": build_prompt(batch)}]
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    results = parse_response_json(text)
    cost = (
        response.usage.input_tokens * INPUT_PRICE_PER_MTOK + response.usage.output_tokens * OUTPUT_PRICE_PER_MTOK
    ) / 1_000_000
    return results, cost


def main():
    parser = argparse.ArgumentParser(description="Tag HN posts with software pain and willingness-to-pay signals.")
    parser.add_argument("--input", default="results/hn_complaints.jsonl")
    parser.add_argument("--output", default="results/hn_classified.jsonl")
    parser.add_argument("--since", help="Only posts on/after this date (YYYY-MM-DD).")
    parser.add_argument("--limit", type=int, help="Only classify the first N remaining posts (for testing).")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    posts = load_posts(args.input, args.since)
    done = load_done(args.output)
    remaining = [p for p in posts if p["id"] not in done]
    if args.limit:
        remaining = remaining[: args.limit]

    batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
    sample = remaining[:BATCH_SIZE]
    in_tok = estimate_tokens(build_prompt(sample)) if sample else 0
    typical = batches * (in_tok * INPUT_PRICE_PER_MTOK + len(sample) * 60 * OUTPUT_PRICE_PER_MTOK) / 1_000_000
    worst = batches * (in_tok * INPUT_PRICE_PER_MTOK + MAX_TOKENS * OUTPUT_PRICE_PER_MTOK) / 1_000_000
    print(f"Posts: {len(posts)}  |  Already classified: {len(done)}  |  To classify now: {len(remaining)}")
    print(f"Batches: {batches}  |  Typical estimate: ${typical:.2f}  |  Worst-case ceiling: ${worst:.2f}")
    print("(Real cost is printed per batch. Past runs landed 40-70% above the typical estimate.)")

    if args.dry_run or not remaining:
        return
    if not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return

    require_api_key()
    client = anthropic.Anthropic()
    total_cost = 0.0
    count = 0
    with open(args.output, "a", encoding="utf-8") as out:
        for n, batch in enumerate(chunked(remaining, BATCH_SIZE), start=1):
            try:
                results, cost = classify_batch(client, batch)
            except (anthropic.APIStatusError, anthropic.APIConnectionError, ValueError, json.JSONDecodeError) as e:
                print(f"Batch {n}: ERROR ({e}) -- skipping, will retry on next run")
                continue
            total_cost += cost
            by_id = {r.get("id"): r for r in results if isinstance(r, dict) and "id" in r}
            for i, post in enumerate(batch):
                r = by_id.get(i)
                if not r:
                    continue
                rec = {
                    "id": post["id"],
                    "url": post["url"],
                    "created_at": post["created_at"],
                    "pain": r.get("pain"),
                    "wants_tool": r.get("wants_tool"),
                    "pay_signal": r.get("pay_signal"),
                    "product": r.get("product"),
                    "pain_summary": r.get("pain_summary"),
                    "domain": r.get("domain"),
                    "classified_at": datetime.now(timezone.utc).isoformat(),
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
            out.flush()
            print(f"Batch {n}/{batches}: {len(by_id)}/{len(batch)}  |  ${cost:.4f}  |  running ${total_cost:.4f}")
    print(f"Done. {count} posts classified into {args.output}  |  Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
