"""Diagnostic script for Currency consistency and Financial Events recurrence analysis.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import statistics


def inspect_exchange_rates(dataset_dir: Path):
    rates_file = dataset_dir / "exchange_rates.csv"
    dates = defaultdict(list)
    with open(rates_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = row["rate_date"].strip()
            fc = row["from_currency"].strip()
            tc = row["to_currency"].strip()
            r = float(row["rate"])
            dates[d].append((fc, tc, r))

    print(f"Total dates in exchange_rates.csv: {len(dates)}")
    for d in sorted(dates.keys())[:5]:
        print(f"  Date {d}: {dates[d]}")


def inspect_financial_events(dataset_dir: Path, user_ids: list[str]):
    events_file = dataset_dir / "financial_events.csv"
    
    user_groups = defaultdict(lambda: defaultdict(list))
    status_by_cat = defaultdict(Counter)

    with open(events_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row["user_id"].strip()
            cat = row["category"].strip()
            desc = row["description"].strip()
            status = row["status"].strip().lower()
            date_str = row["event_date"].strip()
            amt_str = row["amount"].strip()
            direction = row["direction"].strip().lower()
            flexibility = row["flexibility"].strip().lower()
            linked_id = row["linked_event_id"].strip()

            status_by_cat[cat][status] += 1

            if uid in user_ids:
                d = datetime.strptime(date_str, "%Y-%m-%d").date()
                amt = float(amt_str) if amt_str else None
                user_groups[uid][(cat, desc)].append({
                    "event_id": row["event_id"].strip(),
                    "date": d,
                    "status": status,
                    "amount": amt,
                    "direction": direction,
                    "flexibility": flexibility,
                    "linked_id": linked_id,
                })

    print("\n" + "="*80)
    print("STATUS DISTRIBUTION PER CATEGORY (ALL 25,342 EVENTS):")
    print("="*80)
    for cat in sorted(status_by_cat.keys()):
        counts_str = ", ".join(f"{s}: {c}" for s, c in status_by_cat[cat].most_common())
        total = sum(status_by_cat[cat].values())
        print(f"  {cat:<25} (total {total:>5}): {counts_str}")

    print("\n" + "="*80)
    print(f"RECURRENCE ANALYSIS FOR SAMPLE USERS: {user_ids}")
    print("="*80)
    for uid in user_ids:
        print(f"\n--- USER: {uid} ---")
        groups = user_groups[uid]
        for (cat, desc), ev_list in sorted(groups.items()):
            # Sort by date
            ev_list.sort(key=lambda x: x["date"])
            dates = [e["date"] for e in ev_list]
            gaps = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
            statuses = Counter(e["status"] for e in ev_list)
            directions = Counter(e["direction"] for e in ev_list)
            amounts = [e["amount"] for e in ev_list if e["amount"] is not None]

            if len(gaps) > 0:
                med_gap = statistics.median(gaps)
                min_gap = min(gaps)
                max_gap = max(gaps)
                gap_summary = f"count={len(ev_list)}, gaps min={min_gap}, med={med_gap:.1f}, max={max_gap} (sample gaps: {gaps[:8]})"
            else:
                gap_summary = f"count={len(ev_list)} (single occurrence)"

            print(f"  [{cat}] '{desc}':")
            print(f"     {gap_summary}")
            print(f"     Statuses: {dict(statuses)}, Directions: {dict(directions)}, Amounts(first 3): {amounts[:3]}")


def analyze_recurrence_anomalies(dataset_dir: Path):
    events_file = dataset_dir / "financial_events.csv"
    user_groups = defaultdict(lambda: defaultdict(list))
    with open(events_file, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["status"].strip().lower() in ("settled", "scheduled"):
                uid = r["user_id"].strip()
                cat = r["category"].strip()
                desc = r["description"].strip()
                d = datetime.strptime(r["event_date"].strip(), "%Y-%m-%d").date()
                user_groups[uid][(cat, desc)].append(d)

    print("\n" + "="*80)
    print("RECURRENCE ANOMALY & COVERAGE ANALYSIS (ACROSS ALL 275 USERS):")
    print("="*80)

    # Check rent occurrences per user
    rent_counts = Counter()
    for uid, groups in user_groups.items():
        total_rent = sum(len(dates) for (cat, _), dates in groups.items() if cat == "rent")
        rent_counts[total_rent] += 1
    print(f"Total rent events count distribution across users: {dict(sorted(rent_counts.items()))}")

    # Check users with single rent occurrence
    single_rent_users = [uid for uid, groups in user_groups.items() if sum(len(dates) for (cat, _), dates in groups.items() if cat == "rent") == 1]
    print(f"Users with exactly 1 rent event: {single_rent_users}")

    # Check users with zero rent events (e.g. homeowners with housing/home_repair or no rent)
    zero_rent_users = [f"user_{i:02d}" for i in range(1, 276) if f"user_{i:02d}" in user_groups and sum(len(dates) for (cat, _), dates in user_groups[f"user_{i:02d}"].items() if cat == "rent") == 0]
    print(f"Users with 0 rent events (sample): count={len(zero_rent_users)}, first 10: {zero_rent_users[:10]}")

    # Check for categories with 1 occurrence vs >=2 occurrences
    cat_counts_by_group_size = defaultdict(Counter)
    cat_strictly_recurring = Counter()

    for uid, groups in user_groups.items():
        for (cat, desc), dates in groups.items():
            n = len(dates)
            if n == 1:
                cat_counts_by_group_size[cat]["single"] += 1
            else:
                cat_counts_by_group_size[cat]["multiple"] += 1
                dates.sort()
                gaps = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
                med = statistics.median(gaps)
                if all(abs(g - med) <= 3 for g in gaps):
                    cat_strictly_recurring[cat] += 1

    print("\nCategory breakdown of (user, category, desc) groups:")
    for cat in sorted(cat_counts_by_group_size.keys()):
        single = cat_counts_by_group_size[cat]["single"]
        multiple = cat_counts_by_group_size[cat]["multiple"]
        strict = cat_strictly_recurring[cat]
        print(f"  {cat:<25}: single={single:>4}, multiple={multiple:>4}, strict_recurring(med+-3)={strict:>4}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent / "dataset"
    inspect_exchange_rates(base_dir)
    inspect_financial_events(base_dir, ["user_01", "user_02", "user_03"])
    analyze_recurrence_anomalies(base_dir)

