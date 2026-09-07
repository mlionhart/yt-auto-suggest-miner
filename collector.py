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
COST_PER_REQUEST = 0.0006  # standard queue pricing
BATCH_SIZE = 100  # DataForSEO limit per task_post call
NOT_READY_CODES = {40601, 40602}  # "Task Handed" / "Task In Queue"


def load_credentials():
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
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_queries(seeds, letters):
    queries = []
    for seed in seeds:
        queries.append(seed)
        for letter in letters:
            queries.append(f"{seed} {letter}")
    return queries


def load_done_queries(output_path):
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
    for query, task in zip(queries, data.get("tasks") or []):
        if task.get("status_code") == 20100:
            accepted[task["id"]] = query
        else:
            print(f"  SKIP '{query}': task_post rejected it ({task.get('status_code')} {task.get('status_message')})")
    return accepted


def poll_task(session, auth, task_id):
    """Returns (ready: bool, suggestions or None, error or None)."""
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

    if status != 20000:
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

    estimated_cost = len(remaining) * COST_PER_REQUEST
    print(f"Seeds: {len(seeds)}  |  Total queries: {len(queries)}  |  Already collected: {len(already_done)}")
    print(f"Remaining queries to run: {len(remaining)}  |  Estimated cost: ${estimated_cost:.4f}")

    if args.dry_run:
        return

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
    session = requests.Session()

    # 1. Submit all queries as tasks.
    pending = {}
    for batch_num, batch in enumerate(chunked(remaining, BATCH_SIZE), start=1):
        print(f"Submitting batch {batch_num} ({len(batch)} queries)...")
        try:
            accepted = submit_batch(session, auth, batch, args.client, args.location_code, args.language_code)
        except (requests.RequestException, RuntimeError) as e:
            print(f"  ERROR submitting batch {batch_num}: {e}")
            continue
        pending.update(accepted)

    print(f"{len(pending)} tasks queued. Polling every {args.poll_interval:.0f}s (timeout {args.max_wait:.0f}s)...")

    # 2. Poll until all tasks are ready, error out, or we time out.
    collected = 0
    start_time = time.monotonic()
    with open(args.output, "a", encoding="utf-8") as out:
        while pending and (time.monotonic() - start_time) < args.max_wait:
            time.sleep(args.poll_interval)
            for task_id in list(pending.keys()):
                query = pending[task_id]
                try:
                    ready, suggestions, error = poll_task(session, auth, task_id)
                except requests.RequestException as e:
                    print(f"  ERROR polling '{query}': {e}")
                    continue

                if not ready:
                    continue  # still processing, check again next round

                del pending[task_id]

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
                out.flush()
                collected += 1
                print(f"[{collected}] '{query}' -> {len(suggestions)} suggestions")

            elapsed = time.monotonic() - start_time
            print(f"  ...{len(pending)} still pending ({elapsed:.0f}s elapsed)")

    if pending:
        print(f"Timed out waiting on {len(pending)} tasks — rerun the script later to retry them (DataForSEO keeps results available for 30 days).")

    print(f"Done. Collected {collected} new query results into {args.output}")


if __name__ == "__main__":
    main()
