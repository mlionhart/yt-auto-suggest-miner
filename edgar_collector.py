"""
Pulls annual revenue and deferred revenue for a list of public subscription-software companies from
the SEC's free XBRL "company facts" API (https://data.sec.gov). Deferred revenue is a decent proxy for
how much customers have prepaid for subscriptions.

Rules this script follows (SEC fair-access policy): every request carries a User-Agent that identifies
you, and we stay well under 10 requests/second. Put your identification in .env as:
    SEC_USER_AGENT=Your Name your-email@example.com

Limits: paying-subscriber counts (Duolingo, Spotify, ...) are NOT standard XBRL tags; they only appear in
the filing text, so this script does not capture them.
Output: results/edgar_financials.jsonl, one line per company x metric x fiscal year. Rewritten each run.
"""

import argparse
import json
import os
import sys
import time
from datetime import date

import requests
from dotenv import load_dotenv

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# Subscription-heavy public software/app companies. Edit freely.
DEFAULT_TICKERS = [
    "DUOL", "SPOT", "DBX", "ZM", "DOCU", "BOX", "MTCH", "ADBE", "INTU", "COUR", "UDMY", "CHGG", "WIX", "ASAN",
]

# Companies use different tags for the same idea; we record whichever ones they actually report.
REVENUE_TAGS = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"]
DEFERRED_TAGS = ["DeferredRevenueCurrent", "ContractWithCustomerLiabilityCurrent"]


def get_json(session, url):
    response = session.get(url, timeout=60)
    response.raise_for_status()
    time.sleep(0.2)  # ~5 requests/second, half the SEC cap
    return response.json()


def annual_values(facts, tag, years):
    """Latest value per fiscal year from 10-K filings (full-year periods for flow metrics)."""
    units = facts.get("facts", {}).get("us-gaap", {}).get(tag, {}).get("units", {}).get("USD", [])
    by_end = {}
    for item in units:
        if item.get("form") != "10-K" or item.get("fp") != "FY":
            continue
        if "start" in item:  # flow metric: require a roughly 12-month period
            days = (date.fromisoformat(item["end"]) - date.fromisoformat(item["start"])).days
            if not 350 <= days <= 380:
                continue
        key = item["end"]
        if key not in by_end or item["filed"] > by_end[key]["filed"]:
            by_end[key] = item
    return sorted(by_end.values(), key=lambda i: i["end"])[-years:]


def main():
    parser = argparse.ArgumentParser(description="Collect annual revenue/deferred revenue from SEC EDGAR.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--years", type=int, default=4, help="Most recent fiscal years to keep per metric.")
    parser.add_argument("--output", default="results/edgar_financials.jsonl")
    args = parser.parse_args()

    load_dotenv()
    user_agent = os.environ.get("SEC_USER_AGENT")
    if not user_agent:
        sys.exit("Missing SEC_USER_AGENT. Add a line like  SEC_USER_AGENT=Your Name your-email@example.com  to .env")

    session = requests.Session()
    session.headers["User-Agent"] = user_agent

    tickers = {v["ticker"]: v for v in get_json(session, TICKERS_URL).values()}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    written = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for ticker in args.tickers:
            entry = tickers.get(ticker)
            if not entry:
                print(f"{ticker}: not found in SEC ticker list, skipping")
                continue
            try:
                facts = get_json(session, FACTS_URL.format(cik=entry["cik_str"]))
            except requests.RequestException as e:
                print(f"{ticker}: ERROR {e}")
                continue
            count = 0
            for tag in REVENUE_TAGS + DEFERRED_TAGS:
                for item in annual_values(facts, tag, args.years):
                    out.write(json.dumps({
                        "ticker": ticker,
                        "company": entry["title"],
                        "metric": tag,
                        "period_end": item["end"],
                        "fiscal_year": item.get("fy"),
                        "value_usd": item["val"],
                        "filed": item["filed"],
                        "accession": item.get("accn"),
                    }, ensure_ascii=False) + "\n")
                    count += 1
            written += count
            print(f"{ticker} ({entry['title']}): {count} data points")

    print(f"Done. {written} rows written to {args.output}")


if __name__ == "__main__":
    main()
