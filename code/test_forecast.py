"""Test script comparing ForecastEngine output against sample_requests.csv.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure code/ is in sys.path
code_dir = Path(__file__).resolve().parent
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from loaders import load_dataset
from forecast import ForecastEngine
from models import Request


def run_sample_comparison():
    dataset_dir = code_dir.parent / "dataset"
    store = load_dataset(dataset_dir)
    engine = ForecastEngine(store.currency_converter)

    print(f"{'Req ID':<12} | {'Req Date':<10} | {'Req Amt':>12} | {'Safe (Calc)':>12} | {'Safe (True)':>12} | {'Diff':>10} | {'Earliest (Calc)':<15} | {'Earliest (True)':<15} | Match")
    print("-" * 115)

    safe_deltas = []
    earliest_matches = 0

    for sample in store.sample_requests:
        req = Request(
            request_id=sample.request_id,
            user_id=sample.user_id,
            request_date=sample.request_date,
            request_type=sample.request_type,
            requested_amount=sample.requested_amount,
            desired_completion_date=sample.desired_completion_date,
            allows_partial_payment=sample.allows_partial_payment,
            request_text=sample.request_text,
        )
        ctx = store.get_context_for_request(sample.request_id)
        res = engine.run_forecast(req, ctx.profile, ctx.user_events)

        diff = abs(res.amount_safe_to_pay - sample.amount_safe_to_pay)
        safe_deltas.append(diff)

        earliest_true = sample.earliest_date_for_full_payment or ""
        earliest_calc = res.earliest_date_for_full_payment
        earliest_match = (earliest_calc == earliest_true)
        if earliest_match:
            earliest_matches += 1

        match_symbol = "OK" if diff < 1.0 and earliest_match else ("~SAFE" if diff < 1.0 else ("~DATE" if earliest_match else "DIFF"))
        print(f"{sample.request_id:<12} | {sample.request_date:<10} | {sample.requested_amount:>12.2f} | {res.amount_safe_to_pay:>12.2f} | {sample.amount_safe_to_pay:>12.2f} | {diff:>10.2f} | {earliest_calc:<15} | {earliest_true:<15} | {match_symbol}")

    avg_delta = sum(safe_deltas) / len(safe_deltas)
    print("-" * 115)
    print(f"Summary across {len(store.sample_requests)} sample requests:")
    print(f"  Average safe amount delta: {avg_delta:.2f}")
    print(f"  Earliest full payment date exact matches: {earliest_matches} / {len(store.sample_requests)}")


if __name__ == "__main__":
    run_sample_comparison()
