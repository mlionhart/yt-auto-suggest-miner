"""
Joins O*NET "In Demand" software (which occupations list each tool) with BLS OEWS national employment and pay.

For each in-demand tool we report: how many occupations list it, how many people work in those occupations
(sum of BLS employment over distinct occupations), and the employment-weighted average of those occupations'
median annual pay. This is occupation-level pay, not what the tool itself earns. The weighted figure is an
average of medians, a rough indicator only. Reads files downloaded into data/; see the O*NET and BLS notes
in the project memory for where they come from. Output: results/tool_demand_pay.csv.
"""

import collections
import csv

import openpyxl

ONET = "data/onet_31_0/extracted/db_31_0_excel/Software Skills.xlsx"
BLS = "data/bls_oews/extracted/oesm25nat/national_M2025_dl.xlsx"
OUT = "results/tool_demand_pay.csv"


def load_bls():
    ws = openpyxl.load_workbook(BLS, read_only=True)["national_M2025_dl"]
    rows = ws.iter_rows(values_only=True)
    hdr = next(rows)
    col = {name: i for i, name in enumerate(hdr)}
    occ = {}
    for r in rows:
        if r[col["O_GROUP"]] != "detailed":
            continue
        try:
            emp = float(r[col["TOT_EMP"]])
            pay = float(r[col["A_MEDIAN"]])
        except (TypeError, ValueError):
            continue  # suppressed values are marked with symbols like * or #
        occ[r[col["OCC_CODE"]]] = (r[col["OCC_TITLE"]], emp, pay)
    return occ


def main():
    bls = load_bls()
    ws = openpyxl.load_workbook(ONET, read_only=True).active
    tools = collections.defaultdict(set)
    for code, title, example, eid, category, hot, in_demand in list(ws.iter_rows(values_only=True))[1:]:
        if in_demand == "Y":
            tools[(example, category)].add(code[:7])  # 15-1252.00 -> 15-1252

    out_rows = []
    unmatched = collections.Counter()
    for (tool, category), socs in tools.items():
        matched = [s for s in socs if s in bls]
        for s in socs - set(matched):
            unmatched[s] += 1
        total_emp = sum(bls[s][1] for s in matched)
        if not total_emp:
            continue
        weighted_pay = sum(bls[s][1] * bls[s][2] for s in matched) / total_emp
        out_rows.append({
            "tool": tool,
            "category": category,
            "occupations": len(matched),
            "total_employment": int(total_emp),
            "weighted_median_annual_pay": round(weighted_pay),
        })

    out_rows.sort(key=lambda r: -r["total_employment"])
    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)

    print(f"{len(out_rows)} in-demand tools written to {OUT}")
    print(f"O*NET occupations with no BLS match: {len(unmatched)}")
    print(f"\n{'tool':44} {'occs':>5} {'employed':>12} {'wtd pay':>9}")
    for r in out_rows[:15]:
        print(f"{r['tool'][:44]:44} {r['occupations']:5d} {r['total_employment']:12,d} {r['weighted_median_annual_pay']:9,d}")
    print("\nHighest weighted pay (tools used by at least 50,000 workers):")
    for r in sorted([x for x in out_rows if x["total_employment"] >= 50000], key=lambda r: -r["weighted_median_annual_pay"])[:12]:
        print(f"{r['tool'][:44]:44} {r['occupations']:5d} {r['total_employment']:12,d} {r['weighted_median_annual_pay']:9,d}")


if __name__ == "__main__":
    main()
