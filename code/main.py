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
)
from loaders import DatasetStore, RequestContext, load_dataset
from validator import REQUIRED_COLUMNS, validate_output_file


def predict_affordability(ctx: RequestContext) -> OutputRecord:
    """Predict affordability and recommendation for a single financial request.

    TODO: In the next turn, implement the complete decision algorithm:
      1. Reconstruct baseline cash flow position from financial_profile and settled/scheduled events.
      2. Match foreign currency cash events to dated exchange rates using CurrencyConverter.
      3. Process untrusted message/image evidence to detect salary changes, expense updates, or missing amounts.
      4. Run 90-day conservative cash flow forecasting ensuring minimum_balance_to_keep is never breached.
      5. Calculate amount_safe_to_pay on request_date.
      6. Determine earliest_date_for_full_payment.
      7. Evaluate eligible payment methods from user profile preferences and request options.
      8. If needed, identify up to 3 permitted spending changes (stop / reduce_to) on flexible categories.
      9. Rank safe plans according to problem_statement.md §Choosing Between Safe Plans.
      10. Generate concise, grounded decision_explanation.

    Returns:
      OutputRecord matching output.csv specifications.
    """
    req = ctx.request
    prof = ctx.profile

    # STUB PLACEHOLDER: conservative not_affordable default
    # Will be replaced by the exact algorithm in the next iteration.
    return OutputRecord(
        request_id=req.request_id,
        amount_safe_to_pay=0.0,
        affordability_status=AffordabilityStatus.NOT_AFFORDABLE,
        recommended_payment_method=RecommendedPaymentMethod.NOT_RECOMMENDED,
        payment_plan="none",
        earliest_date_for_full_payment="",
        spending_changes_needed="none",
        decision_explanation=(
            f"Placeholder stub: Request for {req.requested_amount:.2f} {prof.home_currency} evaluated. "
            f"Decision logic pending algorithm implementation."
        ),
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

    print("\nProcessing requests through decision stub...")
    output_records: List[OutputRecord] = []
    for req in store.requests:
        ctx = store.get_context_for_request(req.request_id)
        record = predict_affordability(ctx)
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
