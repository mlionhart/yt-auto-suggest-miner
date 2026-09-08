"""
Tags every unique suggestion collected by collector.py with a software-idea category,
using Claude (Haiku -- cheap and fast, plenty capable for a classification task like this).

Key design choices, and why:
- We TAG, we never DELETE. Every suggestion gets a category written to results/classified.jsonl,
  including "not_software" ones -- nothing is thrown away, so a better prompt later can always
  re-classify without re-collecting any data.
- We DEDUPE first. The same suggestion often shows up under multiple different queries
  (e.g. "how to remove acrylic nails" under both "how to remove" and "how to remove a"),
  so we only ever pay to classify each unique phrase once.
- We BATCH many phrases into one API call instead of one call per phrase, since that's far
  cheaper and faster than paying the per-request overhead thousands of times over.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import anthropic
from dotenv import load_dotenv

MODEL = "claude-haiku-4-5"  # small/fast model -- classification doesn't need a heavyweight one
DEFAULT_BATCH_SIZE = 75
MAX_TOKENS = 4096
INPUT_PRICE_PER_MTOK = 1.00   # Haiku 4.5 pricing: $ per 1,000,000 input tokens
OUTPUT_PRICE_PER_MTOK = 5.00  # output tokens cost more than input tokens

# This prompt is deliberately written to avoid two failure modes discussed while building
# this: (1) the model silently narrowing "software idea" to only mean "app" or "extension"
# because those are the most common words, and (2) forcing every phrase into a hard
# yes/no when some are genuinely ambiguous. The explicit list of solution types and the
# "maybe" category both exist specifically to counter those.
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
    """Rough offline estimate (~4 chars/token, a widely-used rule of thumb) -- lets us
    show a cost estimate without needing an API key or spending anything, same as
    collector.py's --dry-run."""
    return len(text) // 4


def require_api_key():
    """Only called right before we're about to actually spend money -- dry-run mode
    never needs a real key."""
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(
            "Missing ANTHROPIC_API_KEY.\n"
            "Add it to your .env file once you've created a key at console.anthropic.com."
        )


def load_suggestions(input_files):
    """Reads every collector.py output file given, flattens out the individual
    suggestions (each collected record holds a LIST of suggestions per query), and
    dedupes them -- this is the "pay to classify each phrase only once" step."""
    seen = set()
    ordered = []
    for path in input_files:
        if not os.path.exists(path):
            continue  # e.g. software_ideas.jsonl might not exist yet -- that's fine
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
    """Resume support, same pattern as collector.py: skip phrases already classified in
    a previous run so a rerun never re-bills for the same phrase twice."""
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
    """Splits a list into groups of at most `size` items -- this is what turns
    thousands of individual phrases into a manageable number of batched API calls."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


def build_prompt(batch):
    """Numbers each phrase (0, 1, 2, ...) so Claude's JSON response can reference each
    one by "id" instead of us having to trust that the response comes back in the exact
    same order we sent -- much more reliable for matching results back up."""
    phrase_list = "\n".join(f"{i}: {phrase}" for i, phrase in enumerate(batch))
    return PROMPT_TEMPLATE.format(max_id=len(batch) - 1, phrase_list=phrase_list)


def parse_response_json(text):
    """Pulls the JSON array out of Claude's text response. We search for the first '['
    and last ']' rather than assuming the whole response is pure JSON, in case the model
    adds any stray text before/after the array despite being asked not to."""
    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("no JSON array found in response")
    return json.loads(text[start : end + 1])


def classify_batch(client, batch):
    """Sends one batch of phrases to Claude and returns (parsed results, real dollar cost).
    The cost here is computed from the ACTUAL tokens used (response.usage), not a guess --
    this is the real, exact number, unlike the pre-flight estimate below."""
    prompt = build_prompt(batch)

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )

    # response.content is a list of blocks (text, thinking, etc.) -- we only want the text ones.
    text = "".join(block.text for block in response.content if block.type == "text")
    results = parse_response_json(text)

    cost = (
        response.usage.input_tokens * INPUT_PRICE_PER_MTOK
        + response.usage.output_tokens * OUTPUT_PRICE_PER_MTOK
    ) / 1_000_000

    return results, cost


def print_estimate(remaining, batch_size):
    """Offline, credential-free cost estimate. Unlike collector.py's DataForSEO pricing
    (a flat $ per request), LLM cost depends on how much text is actually generated, so
    we can't give one exact number in advance -- instead we show a realistic "typical"
    case alongside a hard "worst case" ceiling (assuming the model uses the maximum
    possible output every single time, which is very unlikely but bounds the real cost)."""
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
        return  # everything above this line needs no credentials and spends no money

    if not args.yes:
        confirm = input("Proceed? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return

    require_api_key()
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment automatically

    total_cost = 0.0
    classified_count = 0
    with open(args.output, "a", encoding="utf-8") as out:  # "a" = append, never overwrite past runs
        for batch_num, batch in enumerate(chunked(remaining, args.batch_size), start=1):
            try:
                results, cost = classify_batch(client, batch)
            except (anthropic.APIStatusError, anthropic.APIConnectionError, ValueError, json.JSONDecodeError) as e:
                # A failed batch just gets skipped -- since we never mark these phrases as
                # "done", a rerun of the script will naturally retry them.
                print(f"Batch {batch_num}: ERROR ({e}) -- skipping, will retry on next run")
                continue

            total_cost += cost
            # Look up each result by the "id" Claude gave it, rather than trusting response
            # order -- this is what build_prompt()'s numbering was for.
            by_id = {r.get("id"): r for r in results if isinstance(r, dict) and "id" in r}
            missing = [i for i in range(len(batch)) if i not in by_id]
            if missing:
                print(f"Batch {batch_num}: missing {len(missing)} ids from response, will retry those next run")

            for i, phrase in enumerate(batch):
                if i not in by_id:
                    continue  # this one will just get retried on the next run
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
            out.flush()  # write to disk immediately -- don't lose progress if interrupted

            print(f"Batch {batch_num}: classified {len(by_id)}/{len(batch)}  |  batch cost ${cost:.4f}  |  running total ${total_cost:.4f}")

    print(f"Done. Classified {classified_count} new suggestions into {args.output}  |  Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
