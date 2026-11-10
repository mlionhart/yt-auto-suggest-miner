"""
Cleans up results/classified_v2.jsonl and exports a reviewable list.

- Some rows have the model's "scope" value written into "category" (category == "minor_feature").
  Those are really software_idea rows with scope minor_feature, so we fix them.
- Writes results/classified_v2_clean.jsonl (all rows, normalized) and
  results/standalone_ideas.csv (software_idea/maybe rows tagged standalone_product).
The original classified_v2.jsonl is never modified.
"""

import csv
import json

SRC = "results/classified_v2.jsonl"
CLEAN = "results/classified_v2_clean.jsonl"
CSV_OUT = "results/standalone_ideas.csv"

rows = []
with open(SRC, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["category"] == "minor_feature":
            r["category"] = "software_idea"
            r["scope"] = "minor_feature"
        rows.append(r)

with open(CLEAN, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

standalone = [r for r in rows if r["category"] != "not_software" and r["scope"] == "standalone_product"]
standalone.sort(key=lambda r: (r["solution_type"] or "", r["suggestion"]))

# utf-8-sig so Excel opens the file with correct characters
with open(CSV_OUT, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["suggestion", "category", "solution_type", "reason"])
    for r in standalone:
        w.writerow([r["suggestion"], r["category"], r["solution_type"], r["reason"]])

print(f"Normalized {len(rows)} rows -> {CLEAN}")
print(f"Exported {len(standalone)} standalone_product ideas -> {CSV_OUT}")
