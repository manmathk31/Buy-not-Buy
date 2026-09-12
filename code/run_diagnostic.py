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


def diagnose_parts_a_b_c():
    from datetime import timedelta
    from loaders import load_dataset
    from forecast import ForecastEngine
    from extraction import ExtractionLayer, apply_extractions_to_events

    dataset_dir = Path(__file__).resolve().parent.parent / "dataset"
    store = load_dataset(dataset_dir)
    engine = ForecastEngine(store.currency_converter)

    extractor = ExtractionLayer()
    image_results = {img.image_id: extractor.extract_image_amount(img, dataset_dir) for img in store.images}
    message_amendments = [extractor.parse_message(m) for m in store.messages]
    corrected_events = apply_extractions_to_events(store.events, image_results, message_amendments)

    events_by_user = {}
    for e in corrected_events:
        events_by_user.setdefault(e.user_id, []).append(e)

    sample_dict = {s.request_id: s for s in store.sample_requests}

    print("\n" + "="*80)
    print("PART A DIAGNOSTIC: user_08, user_05, user_25")
    print("="*80)
    for rid in ["request_08", "request_05", "request_25"]:
        s = sample_dict[rid]
        uid = s.user_id
        prof = store.profiles_by_user[uid]
        uevents = events_by_user[uid]
        req_d = engine.parse_date(s.request_date)
        print(f"\n--- {rid} ({uid}) ---")
        print(f"Req Date: {s.request_date}, Req Amt: {s.requested_amount}, True Safe: {s.amount_safe_to_pay}")
        print(f"Balance: {prof.current_available_balance}, Min Keep: {prof.minimum_balance_to_keep}, Home: {prof.home_currency}")
        print(f"Initial Surplus: {prof.current_available_balance - prof.minimum_balance_to_keep:.2f}")
        print(f"Protected Categories: {prof.protect_categories_list}")
        
        # Check messages
        user_msgs = [m for m in store.messages if m.user_id == uid]
        print(f"Messages for {uid}: {[m.message_text for m in user_msgs]}")
        
        from models import Request
        req_obj = Request(
            request_id=s.request_id,
            user_id=s.user_id,
            request_date=s.request_date,
            request_type=s.request_type,
            requested_amount=s.requested_amount,
            desired_completion_date=s.desired_completion_date,
            allows_partial_payment=s.allows_partial_payment,
            request_text=s.request_text,
        )
        res = engine.run_forecast(req_obj, prof, uevents)
        print(f"Engine Result: Safe={res.amount_safe_to_pay:.2f}, Earliest={res.earliest_date_for_full_payment}")
        if uid == "user_25":
            print(f"USER_25 DRAG CALCULATION:")
            # Let's inspect drag components
            prof = store.profiles_by_user[uid]
            # print what drag computed
            print(f"  Surplus: {prof.current_available_balance - prof.minimum_balance_to_keep}")

    print("\n" + "="*80)
    print("PART B DIAGNOSTIC: request_04, request_20, request_22, request_23")
    print("="*80)
    for rid in ["request_04", "request_20", "request_22", "request_23"]:
        s = sample_dict[rid]
        uid = s.user_id
        prof = store.profiles_by_user[uid]
        uevents = events_by_user[uid]
        req_d = engine.parse_date(s.request_date)
        req_obj = Request(
            request_id=s.request_id,
            user_id=s.user_id,
            request_date=s.request_date,
            request_type=s.request_type,
            requested_amount=s.requested_amount,
            desired_completion_date=s.desired_completion_date,
            allows_partial_payment=s.allows_partial_payment,
            request_text=s.request_text,
        )
        res = engine.run_forecast(req_obj, prof, uevents)
        print(f"\n--- {rid} ({uid}) ---")
        print(f"Req Date: {s.request_date}, Req Amt: {s.requested_amount}")
        print(f"Earliest Calc: '{res.earliest_date_for_full_payment}' vs True: '{s.earliest_date_for_full_payment}'")
        print(f"Safe Calc: {res.amount_safe_to_pay:.2f} vs True: {s.amount_safe_to_pay:.2f}")
        
        # Check cash flow items around dates
        true_d = engine.parse_date(s.earliest_date_for_full_payment) if s.earliest_date_for_full_payment else None
        calc_d = engine.parse_date(res.earliest_date_for_full_payment) if res.earliest_date_for_full_payment else None
        
        dates_to_check = [d for d in [true_d, calc_d] if d is not None]
        if dates_to_check:
            min_check = min(dates_to_check) - timedelta(days=2)
            max_check = max(dates_to_check) + timedelta(days=2)
            print(f"Margin trace between {min_check} and {max_check}:")
            for d in sorted(res.daily_balances.keys()):
                if min_check <= d <= max_check:
                    # check if full payment safe on day d:
                    # balance >= min_keep + requested_amount AND remaining >= min_keep
                    is_safe_full = (res.daily_balances[d] >= prof.minimum_balance_to_keep + s.requested_amount)
                    # check remaining min margin
                    rem_margin = min(res.daily_margins[d2] for d2 in res.daily_margins if d2 >= d)
                    print(f"  {d} | bal={res.daily_balances[d]:.2f} | margin={res.daily_margins[d]:.2f} | rem_min_margin={rem_margin:.2f} | bal>=min+req:{is_safe_full}")
        
        if uid == "user_04":
            print(f"USER_04 PROFILE:")
            print(f"  Protected: {prof.protect_categories_list}")
            print(f"  Adjustable: {prof.reduce_categories_list}")
            print(f"  Initial Surplus: {prof.current_available_balance - prof.minimum_balance_to_keep}")

    print("\n" + "="*80)
    print("PART C DEEP DIVE: request_02, request_07, request_10, request_17, request_19")
    print("="*80)
    for rid in ["request_02", "request_07", "request_10", "request_17", "request_19"]:
        s = sample_dict[rid]
        uid = s.user_id
        prof = store.profiles_by_user[uid]
        uevents = events_by_user[uid]
        req_d = engine.parse_date(s.request_date)
        req_obj = Request(
            request_id=s.request_id,
            user_id=s.user_id,
            request_date=s.request_date,
            request_type=s.request_type,
            requested_amount=s.requested_amount,
            desired_completion_date=s.desired_completion_date,
            allows_partial_payment=s.allows_partial_payment,
            request_text=s.request_text,
        )
        res = engine.run_forecast(req_obj, prof, uevents)
        print(f"\n--- {rid} ({uid}) ---")
        print(f"Req Date: {s.request_date}, Req Amt: {s.requested_amount}")
        print(f"Balance: {prof.current_available_balance}, Min Keep: {prof.minimum_balance_to_keep}, Initial Surplus: {prof.current_available_balance - prof.minimum_balance_to_keep:.2f}")
        print(f"Safe Calc: {res.amount_safe_to_pay:.2f} vs True: {s.amount_safe_to_pay:.2f} (diff: {abs(res.amount_safe_to_pay - s.amount_safe_to_pay):.2f})")
        print(f"Earliest Calc: '{res.earliest_date_for_full_payment}' vs True: '{s.earliest_date_for_full_payment}'")
        
        # Limiting date
        min_d = min(res.daily_margins.keys(), key=lambda d: res.daily_margins[d])
        print(f"Limiting Date (Calc): {min_d} | Margin: {res.daily_margins[min_d]:.2f} | Balance: {res.daily_balances[min_d]:.2f}")
        
        # Check event_1545 and event_1700
        for eid in ["event_1545", "event_1700"]:
            matching = [e for e in store.events if e.event_id == eid]
            if matching:
                ev = matching[0]
                print(f"EVENT {eid}: date={ev.event_date}, s_date={ev.settlement_date}, status={ev.status}, amt={ev.amount}, dir={ev.direction}, desc={ev.description}")
                
        if uid == "user_10":
            print(f"USER_10 SIMULATION WITHOUT GIG INCOME:")
            # Filter out credit events from candidate recurrence
            non_gig_events = [e for e in uevents if not (e.category == "salary" and any(w in e.description.lower() for w in ("driver", "delivery", "task", "app")))]
            res_nogig = engine.run_forecast(req_obj, prof, non_gig_events)
            print(f"  Safe without gig: {res_nogig.amount_safe_to_pay:.2f} (True is 12700.00)")
            min_d = min(res_nogig.daily_margins.keys(), key=lambda d: res_nogig.daily_margins[d])
            print(f"  Min Margin Date: {min_d} | Margin: {res_nogig.daily_margins[min_d]:.2f}")
            
        # Messages for this user
        u_msgs = [m for m in store.messages if m.user_id == uid]
        for m in u_msgs:
            print(f"  MESSAGE: {m.message_id} | related_event={m.related_event_id} | text={m.message_text}")


if __name__ == "__main__":
    diagnose_parts_a_b_c()

