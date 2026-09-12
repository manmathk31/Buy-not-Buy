"""Entry point for the Buy or Wait? AI financial decision agent.

Orchestrates:
  1. Loading and relational validation of all dataset CSV files.
  2. Request context assembly (profile, events, payment options, messages, images).
  3. Decision logic stub (forecasting and affordability logic to be plugged in).
  4. Output generation adhering strictly to the AGENTS.md §6.2 contract.
  5. Hard schema and financial rule validation via validator.py.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional

# Ensure local imports work whether executed from repo root or code/
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from models import (
    AffordabilityStatus,
    OutputRecord,
    RecommendedPaymentMethod,
    Request,
    FinancialEvent,
)
from loaders import DatasetStore, RequestContext, load_dataset
from validator import REQUIRED_COLUMNS, validate_output_file
from forecast import ForecastEngine
from extraction import ExtractionLayer, apply_extractions_to_events


def predict_affordability(
    ctx: RequestContext,
    engine: ForecastEngine,
    user_events: Optional[List[FinancialEvent]] = None,
) -> OutputRecord:
    """Predict affordability and recommendation for a single financial request.

    Reconstructs 90-day cash flow forecast, calculates amount_safe_to_pay,
    and determines earliest_date_for_full_payment using ForecastEngine.
    """
    req = ctx.request
    prof = ctx.profile
    events = user_events if user_events is not None else ctx.user_events

    # Run 90-day cash flow simulation
    forecast_res = engine.run_forecast(req, prof, events)
    amount_safe = forecast_res.amount_safe_to_pay
    earliest_date = forecast_res.earliest_date_for_full_payment

    # Format amounts cleanly
    safe_amt_str = f"{amount_safe:.2f}".rstrip("0").rstrip(".") if isinstance(amount_safe, float) else str(amount_safe)
    req_amt_str = f"{req.requested_amount:.2f}".rstrip("0").rstrip(".") if isinstance(req.requested_amount, float) else str(req.requested_amount)

    # Decision stub based on forecast safety check
    if amount_safe >= req.requested_amount - 1e-4:
        status = AffordabilityStatus.AFFORDABLE_NOW
        method = RecommendedPaymentMethod.FULL_PAYMENT
        plan = f"{req.request_date}:{safe_amt_str}"
        earliest = req.request_date
        spending = "none"
        explanation = (
            f"Pay {prof.home_currency} {safe_amt_str} today. "
            f"This leaves at least {prof.home_currency} {prof.minimum_balance_to_keep:.2f} available over the next 90 days."
        )
    elif earliest_date:
        status = AffordabilityStatus.AFFORDABLE_LATER
        method = RecommendedPaymentMethod.WAIT
        plan = f"{earliest_date}:{req_amt_str}"
        earliest = earliest_date
        spending = "none"
        explanation = (
            f"Wait until {earliest_date}, then pay {prof.home_currency} {req_amt_str} in full. "
            f"Paying earlier would take the balance below the {prof.home_currency} {prof.minimum_balance_to_keep:.2f} minimum."
        )
    else:
        status = AffordabilityStatus.NOT_AFFORDABLE
        method = RecommendedPaymentMethod.NOT_RECOMMENDED
        plan = "none"
        earliest = ""
        spending = "none"
        explanation = (
            f"Do not proceed with the {prof.home_currency} {req_amt_str} request. "
            f"The full amount cannot be completed safely within 90 days without spending adjustments."
        )

    return OutputRecord(
        request_id=req.request_id,
        amount_safe_to_pay=amount_safe,
        affordability_status=status,
        recommended_payment_method=method,
        payment_plan=plan,
        earliest_date_for_full_payment=earliest,
        spending_changes_needed=spending,
        decision_explanation=explanation,
    )


def write_output_csv(records: List[OutputRecord], output_path: Path) -> None:
    """Write OutputRecord objects to CSV with exact header columns in order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec.to_csv_row())


def run_pipeline(
    dataset_dir: str | Path,
    output_file: str | Path,
    validate: bool = True,
) -> List[OutputRecord]:
    """Execute end-to-end dataset loading, prediction stubbing, writing, and strict validation."""
    dataset_path = Path(dataset_dir).resolve()
    out_path = Path(output_file).resolve()

    print(f"Loading dataset from: {dataset_path}")
    store = load_dataset(dataset_path)
    print(f"Successfully loaded and indexed:")
    print(f"  - {len(store.requests)} requests")
    print(f"  - {len(store.sample_requests)} sample requests")
    print(f"  - {len(store.profiles)} user profiles")
    print(f"  - {len(store.events)} financial events")
    print(f"  - {len(store.exchange_rates)} exchange rates ({len(store.currency_converter.available_dates)} unique dates)")
    print(f"  - {len(store.payment_options)} request payment options")
    print(f"  - {len(store.messages)} messages")
    print(f"  - {len(store.images)} image links")

    # 1. Scoped extraction layer
    print("\nRunning scoped extraction layer on images and messages...")
    extractor = ExtractionLayer()
    image_results = {img.image_id: extractor.extract_image_amount(img, dataset_path) for img in store.images}
    message_amendments = [extractor.parse_message(m) for m in store.messages]
    corrected_events = apply_extractions_to_events(store.events, image_results, message_amendments)
    print(f"Extracted verified amounts for {len(image_results)} images, parsed {len(message_amendments)} messages.")

    # Group corrected events by user
    events_by_user: dict[str, list[FinancialEvent]] = {}
    for e in corrected_events:
        events_by_user.setdefault(e.user_id, []).append(e)

    engine = ForecastEngine(store.currency_converter)
    print("\nProcessing requests through 90-day forecast engine...")
    output_records: List[OutputRecord] = []
    for req in store.requests:
        ctx = store.get_context_for_request(req.request_id)
        u_events = events_by_user.get(req.user_id, ctx.user_events)
        record = predict_affordability(ctx, engine, u_events)
        output_records.append(record)

    print(f"Writing {len(output_records)} output records to: {out_path}")
    write_output_csv(output_records, out_path)

    if validate:
        print("Running strict output schema and financial rules validator...")
        validated_records = validate_output_file(
            file_path=out_path,
            requests=store.requests,
            expected_row_count=len(store.requests),
        )
        print(f"Validation PASSED! All {len(validated_records)} rows strictly conform to contract.")

    return output_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Buy or Wait? Financial Decision Agent")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(current_dir.parent / "dataset"),
        help="Path to dataset directory containing CSV files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(current_dir.parent / "dataset" / "output.csv"),
        help="Path to output CSV file",
    )
    parser.add_argument(
        "--validate-only",
        type=str,
        default=None,
        help="If set, strictly validates an existing output CSV file instead of running predictions",
    )

    args = parser.parse_args()

    if args.validate_only:
        val_path = Path(args.validate_only)
        print(f"Validating file: {val_path}")
        validate_output_file(val_path)
        print("Validation succeeded!")
        return

    run_pipeline(dataset_dir=args.dataset_dir, output_file=args.output)


if __name__ == "__main__":
    main()
