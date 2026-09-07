import argparse
import json
import os
import sys
from datetime import datetime, timezone

import anthropic
from dotenv import load_dotenv

MODEL = "claude-haiku-4-5"
DEFAULT_BATCH_SIZE = 75
MAX_TOKENS = 4096
INPUT_PRICE_PER_MTOK = 1.00
OUTPUT_PRICE_PER_MTOK = 5.00

PROMPT_TEMPLATE = """You are screening search-autocomplete phrases to find ideas for software products.

A "software idea" (category: software_idea) is any phrase where the underlying need could plausibly be solved by: a browser extension, a mobile app (iOS/Android), a web app or website, desktop software, a plugin/add-on, an automation script or macro, a bot, an API/integration, or a SaaS tool.

Do NOT require the phrase to literally mention "app," "extension," or "software" -- most good ideas won't use those words at all (e.g. "how to hide following list on instagram" is a strong browser-extension idea even though it never says "extension"). Judge based on the underlying need, not the literal wording.

Use category "maybe" for phrases that are plausible but ambiguous -- don't force a hard yes/no on genuinely borderline cases.
Use category "not_software" for phrases with no plausible software angle (e.g. beauty, cooking, medical, personal habits).

For phrases classified as "software_idea" or "maybe", also give your best-guess "solution_type": one of browser extension, mobile app, web app, desktop app, script/automation, bot, plugin/add-on, api/integration, saas tool, other. Use null for "not_software".

Give a short "reason" (under 12 words) for every phrase.

Return ONLY a JSON array, one object per phrase, no other text, in this exact shape:
[{{"id": 0, "category": "software_idea", "solution_type": "browser extension", "reason": "..."}}, ...]

Include every id from 0 to {max_id} exactly once.

Phrases:
{phrase_list}
"""


def estimate_tokens(text):
    """Rough offline estimate (~4 chars/token) -- no API key or network call needed."""
    return len(text) // 4


def require_api_key():
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(
            "Missing ANTHROPIC_API_KEY.\n"
            "Add it to your .env file once you've created a key at console.anthropic.com."
        )


def load_suggestions(input_files):
    seen = set()
    ordered = []
    for path in input_files:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                for s in record.get("suggestions", []):
                    if s not in seen:
                        seen.add(s)
                        ordered.append(s)
    return ordered


def load_done(output_path):
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["suggestion"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def chunked(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def build_prompt(batch):
    phrase_list = "\n".join(f"{i}: {phrase}" for i, phrase in enumerate(batch))
    return PROMPT_TEMPLATE.format(max_id=len(batch) - 1, phrase_list=phrase_list)


def parse_response_json(text):
    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("no JSON array found in response")
    return json.loads(text[start : end + 1])


def classify_batch(client, batch):
    prompt = build_prompt(batch)

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )

    text = "".join(block.text for block in response.content if block.type == "text")
    results = parse_response_json(text)

    cost = (
        response.usage.input_tokens * INPUT_PRICE_PER_MTOK
        + response.usage.output_tokens * OUTPUT_PRICE_PER_MTOK
    ) / 1_000_000

    return results, cost


def print_estimate(remaining, batch_size):
    """Offline, credential-free estimate: typical case + a worst-case ceiling."""
    num_batches = (len(remaining) + batch_size - 1) // batch_size
    sample_batch = remaining[:batch_size] or [""]
    sample_input_tokens = estimate_tokens(build_prompt(sample_batch))

    # Typical output: ~25 tokens per classified item (id + category + solution_type + short reason).
    typical_output_tokens_per_batch = len(sample_batch) * 25
    typical_cost = num_batches * (
        sample_input_tokens * INPUT_PRICE_PER_MTOK
        + typical_output_tokens_per_batch * OUTPUT_PRICE_PER_MTOK
    ) / 1_000_000

    # Worst case: model uses the full MAX_TOKENS on every batch.
    worst_cost = num_batches * (
        sample_input_tokens * INPUT_PRICE_PER_MTOK + MAX_TOKENS * OUTPUT_PRICE_PER_MTOK
    ) / 1_000_000

    print(f"Estimated batches: {num_batches} (batch size {batch_size})")
    print(f"Typical estimated cost: ${typical_cost:.2f}  |  Worst-case ceiling: ${worst_cost:.2f}")
    print("(Real cost is tracked and printed per-batch as it actually runs -- this is only a pre-flight estimate.)")


def main():
    parser = argparse.ArgumentParser(description="Classify collected autocomplete suggestions as software ideas using Claude.")
    parser.add_argument("--input", nargs="+", default=["results/results.jsonl", "results/software_ideas.jsonl"])
    parser.add_argument("--output", default="results/classified.jsonl")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    suggestions = load_suggestions(args.input)
    already_done = load_done(args.output)
    remaining = [s for s in suggestions if s not in already_done]

    print(f"Unique suggestions found: {len(suggestions)}  |  Already classified: {len(already_done)}")
    print(f"Remaining to classify: {len(remaining)}")

    if not remaining:
        print("Nothing to do.")
        return

    print_estimate(remaining, args.batch_size)

    if args.dry_run:
        return

    if not args.yes:
        confirm = input("Proceed? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return

    require_api_key()
    client = anthropic.Anthropic()

    total_cost = 0.0
    classified_count = 0
    with open(args.output, "a", encoding="utf-8") as out:
        for batch_num, batch in enumerate(chunked(remaining, args.batch_size), start=1):
            try:
                results, cost = classify_batch(client, batch)
            except (anthropic.APIStatusError, anthropic.APIConnectionError, ValueError, json.JSONDecodeError) as e:
                print(f"Batch {batch_num}: ERROR ({e}) -- skipping, will retry on next run")
                continue

            total_cost += cost
            by_id = {r.get("id"): r for r in results if isinstance(r, dict) and "id" in r}
            missing = [i for i in range(len(batch)) if i not in by_id]
            if missing:
                print(f"Batch {batch_num}: missing {len(missing)} ids from response, will retry those next run")

            for i, phrase in enumerate(batch):
                if i not in by_id:
                    continue
                r = by_id[i]
                record = {
                    "suggestion": phrase,
                    "category": r.get("category"),
                    "solution_type": r.get("solution_type"),
                    "reason": r.get("reason"),
                    "classified_at": datetime.now(timezone.utc).isoformat(),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                classified_count += 1
            out.flush()

            print(f"Batch {batch_num}: classified {len(by_id)}/{len(batch)}  |  batch cost ${cost:.4f}  |  running total ${total_cost:.4f}")

    print(f"Done. Classified {classified_count} new suggestions into {args.output}  |  Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
