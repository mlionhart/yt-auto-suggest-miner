"""
Collects Google/YouTube autocomplete suggestions via DataForSEO's "standard queue" API.

Big picture of how this works (the "async" pattern we talked through):
1. We take a list of seed phrases (e.g. "how to remove") and cross each one with every
   letter of the alphabet ("how to remove a", "how to remove b", ...) to sweep as many
   real autocomplete suggestions as possible.
2. Instead of asking DataForSEO for each answer immediately (expensive "live" mode), we
   submit ALL the queries as background "tasks" first (task_post), then repeatedly check
   back later to see which ones are done (task_get) -- like a restaurant buzzer instead
   of standing at the counter. This is ~3.3x cheaper.
3. Everything is written to a JSONL file as soon as it's ready, so if the script is
   interrupted, nothing already collected is lost, and rerunning skips what's done.
"""

import argparse
import json
import os
import string
import sys
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

TASK_POST_URL = "https://api.dataforseo.com/v3/serp/google/autocomplete/task_post"
TASK_GET_URL = "https://api.dataforseo.com/v3/serp/google/autocomplete/task_get/advanced/{}"
COST_PER_REQUEST = 0.0006  # standard queue pricing (see: DataForSEO pricing page)
BATCH_SIZE = 100  # DataForSEO limit: max tasks allowed in a single task_post call
NOT_READY_CODES = {40601, 40602}  # DataForSEO's "Task Handed" / "Task In Queue" codes --
                                  # these mean "still processing," not an error


def load_credentials():
    """Reads the DataForSEO login/password out of .env. Only called right before we're
    about to actually spend money -- dry-run mode never needs this."""
    load_dotenv()
    login = os.environ.get("DATAFORSEO_LOGIN")
    password = os.environ.get("DATAFORSEO_PASSWORD")
    if not login or not password:
        sys.exit(
            "Missing DATAFORSEO_LOGIN / DATAFORSEO_PASSWORD.\n"
            "Copy .env.example to .env and fill in your DataForSEO credentials."
        )
    return login, password


def load_seeds(path):
    """Reads one seed phrase per line from a text file (e.g. seeds.txt)."""
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_queries(seeds, letters):
    """The alphabet-sweep: for each seed, query the bare seed itself, plus the seed
    followed by every letter. This is what surfaces long-tail suggestions Google/YouTube
    wouldn't show you for the bare phrase alone -- e.g. "how to hide" + "f" surfaces
    "how to hide followers on instagram" that "how to hide" alone might not."""
    queries = []
    for seed in seeds:
        queries.append(seed)
        for letter in letters:
            queries.append(f"{seed} {letter}")
    return queries


def load_done_queries(output_path):
    """Resume support: reads whatever queries are already saved in the output file, so a
    rerun only pays for and collects what's actually missing -- never re-billed for the
    same query twice."""
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                done.add(record["query"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def chunked(items, size):
    """Splits a list into groups of at most `size` items -- used to respect DataForSEO's
    100-tasks-per-call limit on task_post."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


def submit_batch(session, auth, queries, client, location_code, language_code):
    """POST a batch of queries as tasks. Returns {task_id: query} for tasks accepted."""
    payload = [
        {
            "keyword": q,
            "location_code": location_code,
            "language_code": language_code,
            "client": client,
        }
        for q in queries
    ]
    response = session.post(TASK_POST_URL, auth=auth, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()

    if data.get("status_code") != 20000:
        raise RuntimeError(f"task_post error: {data.get('status_code')} {data.get('status_message')}")

    accepted = {}
    # DataForSEO returns tasks in the same order we sent them, so zip() lets us line up
    # each response task with the query that produced it.
    for query, task in zip(queries, data.get("tasks") or []):
        if task.get("status_code") == 20100:  # 20100 = "Task Created" (accepted)
            accepted[task["id"]] = query
        else:
            print(f"  SKIP '{query}': task_post rejected it ({task.get('status_code')} {task.get('status_message')})")
    return accepted


def poll_task(session, auth, task_id):
    """Checks whether one submitted task is done yet.
    Returns (ready: bool, suggestions or None, error or None) -- three possible outcomes:
      - (False, None, None)      -> still processing, check again later
      - (True,  [...], None)     -> done, here are the suggestions
      - (True,  None, "message") -> done, but it failed for some reason
    """
    response = session.get(TASK_GET_URL.format(task_id), auth=auth, timeout=30)
    response.raise_for_status()
    data = response.json()

    tasks = data.get("tasks") or []
    if not tasks:
        return False, None, "no task in response"

    task = tasks[0]
    status = task.get("status_code")

    if status in NOT_READY_CODES:
        return False, None, None

    if status != 20000:  # some real error, not just "still working on it"
        return True, None, f"{status} {task.get('status_message')}"

    results = task.get("result") or []
    if not results:
        return True, [], None

    items = results[0].get("items") or []
    return True, [item["suggestion"] for item in items if item.get("suggestion")], None


def main():
    parser = argparse.ArgumentParser(description="Collect YouTube/Google autocomplete suggestions via DataForSEO (standard queue).")
    parser.add_argument("--seeds", default="seeds.txt", help="Path to seed phrases file (one per line).")
    parser.add_argument("--letters", default=string.ascii_lowercase, help="Letters to append to each seed.")
    parser.add_argument("--output", default="results/results.jsonl", help="Output JSONL file.")
    parser.add_argument("--client", default="youtube", choices=["youtube", "chrome", "chrome-omni", "gws-wiz-serp", "safari", "firefox"])
    parser.add_argument("--location-code", type=int, default=2840, help="DataForSEO location code (2840 = United States).")
    parser.add_argument("--language-code", default="en")
    parser.add_argument("--poll-interval", type=float, default=30, help="Seconds between polling rounds.")
    parser.add_argument("--max-wait", type=float, default=3000, help="Give up on unfinished tasks after this many seconds (default 50 min).")
    parser.add_argument("--dry-run", action="store_true", help="Show query count and estimated cost without calling the API.")
    parser.add_argument("--yes", action="store_true", help="Skip the cost confirmation prompt.")
    args = parser.parse_args()

    seeds = load_seeds(args.seeds)
    queries = build_queries(seeds, args.letters)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    already_done = load_done_queries(args.output)
    remaining = [q for q in queries if q not in already_done]

    # Cost is deterministic here (flat price per request), so this estimate is exact --
    # unlike classify.py, where LLM cost depends on how much text is generated.
    estimated_cost = len(remaining) * COST_PER_REQUEST
    print(f"Seeds: {len(seeds)}  |  Total queries: {len(queries)}  |  Already collected: {len(already_done)}")
    print(f"Remaining queries to run: {len(remaining)}  |  Estimated cost: ${estimated_cost:.4f}")

    if args.dry_run:
        return  # everything above this line needs no credentials and spends no money

    if not remaining:
        print("Nothing to do — all queries already collected in this output file.")
        return

    if not args.yes:
        confirm = input(f"Proceed and spend ~${estimated_cost:.4f}? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return

    login, password = load_credentials()
    auth = (login, password)
    session = requests.Session()  # reuses one HTTP connection across all our calls

    # --- Phase 1: submit everything as background tasks ("order all the burgers") ---
    pending = {}  # task_id -> the original query text, so we can label results later
    for batch_num, batch in enumerate(chunked(remaining, BATCH_SIZE), start=1):
        print(f"Submitting batch {batch_num} ({len(batch)} queries)...")
        try:
            accepted = submit_batch(session, auth, batch, args.client, args.location_code, args.language_code)
        except (requests.RequestException, RuntimeError) as e:
            print(f"  ERROR submitting batch {batch_num}: {e}")
            continue
        pending.update(accepted)

    print(f"{len(pending)} tasks queued. Polling every {args.poll_interval:.0f}s (timeout {args.max_wait:.0f}s)...")

    # --- Phase 2: poll until everything's ready, fails, or we give up ("check the buzzers") ---
    collected = 0
    start_time = time.monotonic()
    with open(args.output, "a", encoding="utf-8") as out:  # "a" = append, never overwrite past runs
        while pending and (time.monotonic() - start_time) < args.max_wait:
            time.sleep(args.poll_interval)
            for task_id in list(pending.keys()):  # list(...) so we can safely del while looping
                query = pending[task_id]
                try:
                    ready, suggestions, error = poll_task(session, auth, task_id)
                except requests.RequestException as e:
                    print(f"  ERROR polling '{query}': {e}")
                    continue

                if not ready:
                    continue  # still processing, check again next round

                del pending[task_id]  # resolved one way or another, stop checking it

                if error:
                    print(f"  FAILED '{query}': {error}")
                    continue

                record = {
                    "query": query,
                    "client": args.client,
                    "suggestions": suggestions,
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()  # write to disk immediately -- don't lose progress if interrupted
                collected += 1
                print(f"[{collected}] '{query}' -> {len(suggestions)} suggestions")

            elapsed = time.monotonic() - start_time
            print(f"  ...{len(pending)} still pending ({elapsed:.0f}s elapsed)")

    if pending:
        print(f"Timed out waiting on {len(pending)} tasks — rerun the script later to retry them (DataForSEO keeps results available for 30 days).")

    print(f"Done. Collected {collected} new query results into {args.output}")


if __name__ == "__main__":
    main()
