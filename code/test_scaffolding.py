"""Comprehensive unit and integration test suite for the Buy or Wait? scaffolding.

Tests:
  1. Data loaders on real dataset files with strict row count assertions.
  2. Relational joins by user_id, request_id, related_event_id, linked_event_id.
  3. Currency converter: direct, inverse, multi-hop chaining (ZAR->IDR), exact date match, missing date error.
  4. Validator: acceptance of valid sample records, rejection of invalid schemas/enums/ranges/formats.
  5. End-to-end pipeline execution with stub logic.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Add code/ to path
code_dir = Path(__file__).resolve().parent
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from models import (
    AffordabilityStatus,
    OutputRecord,
    RecommendedPaymentMethod,
    Request,
)
from loaders import (
    DatasetRowCountError,
    DatasetStore,
    load_dataset,
    load_requests,
    load_sample_requests,
    load_financial_profiles,
    load_financial_events,
    load_exchange_rates,
    load_payment_options,
    load_messages,
    load_images,
)
from currency import CurrencyConverter, MissingExchangeRateError
from validator import OutputValidationError, validate_output_file, validate_spending_changes, validate_payment_plan
from main import run_pipeline


def test_currency_converter():
    print("Testing CurrencyConverter...")
    dataset_dir = code_dir.parent / "dataset"
    rates = load_exchange_rates(dataset_dir / "exchange_rates.csv")
    converter = CurrencyConverter(rates)

    # 1. Identity conversion
    assert converter.convert(100.0, "EUR", "EUR", "2024-04-15") == 100.0
    assert converter.get_rate("USD", "USD", "2024-04-15") == 1.0

    # 2. Direct lookup: EUR -> ZAR on 2024-04-15 (rate is 20 in csv)
    rate_eur_zar = converter.get_rate("EUR", "ZAR", "2024-04-15")
    assert abs(rate_eur_zar - 20.0) < 1e-6, f"Expected 20.0, got {rate_eur_zar}"

    # 3. Direct lookup: USD -> IDR on 2024-04-15 (rate is 15833.33)
    rate_usd_idr = converter.get_rate("USD", "IDR", "2024-04-15")
    assert abs(rate_usd_idr - 15833.33) < 1e-4

    # 4. Inverted rate: ZAR -> EUR on 2024-04-15 (should be 1 / 20 = 0.05)
    rate_zar_eur = converter.get_rate("ZAR", "EUR", "2024-04-15")
    assert abs(rate_zar_eur - 0.05) < 1e-6, f"Expected 0.05, got {rate_zar_eur}"

    # 5. Multi-hop chaining: ZAR -> IDR on 2024-04-15
    # ZAR -> EUR -> USD -> IDR
    # EUR -> USD is 1.09, USD -> IDR is 15833.33, ZAR -> EUR is 0.05
    # Expected: 0.05 * 1.09 * 15833.33 = 862.916485
    rate_zar_idr = converter.get_rate("ZAR", "IDR", "2024-04-15")
    expected_chain = (1.0 / 20.0) * 1.09 * 15833.33
    assert abs(rate_zar_idr - expected_chain) < 1e-3, f"Expected {expected_chain}, got {rate_zar_idr}"
    print(f"  ZAR -> IDR chained rate on 2024-04-15: {rate_zar_idr:.4f}")

    # 6. Multi-hop chaining: IDR -> ZAR on 2024-04-15
    # IDR -> USD (1/15833.33) -> EUR (0.92) -> ZAR (20)
    rate_idr_zar = converter.get_rate("IDR", "ZAR", "2024-04-15")
    expected_idr_zar = (1.0 / 15833.33) * 0.92 * 20.0
    assert abs(rate_idr_zar - expected_idr_zar) < 1e-6, f"Expected {expected_idr_zar}, got {rate_idr_zar}"
    print(f"  IDR -> ZAR chained rate on 2024-04-15: {rate_idr_zar:.8f}")

    # 7. Exact date required: Non-existent date raises MissingExchangeRateError
    try:
        converter.get_rate("EUR", "USD", "1999-01-01")
        assert False, "Should have raised MissingExchangeRateError for missing date"
    except MissingExchangeRateError as e:
        print(f"  Correctly caught missing date error: {e}")

    print("CurrencyConverter tests PASSED!\n")


def test_validator_rules():
    print("Testing Output Schema Validator Rules...")

    # 1. Spending changes validation
    validate_spending_changes("none")
    validate_spending_changes("stop:event_14")
    validate_spending_changes("stop:event_14|reduce_to:event_21:100")
    validate_spending_changes("stop:event_14|reduce_to:event_21:100.50|stop:event_30")

    # Reject commas
    try:
        validate_spending_changes("stop:event_14,reduce_to:event_21:100")
        assert False, "Should reject commas in spending changes"
    except OutputValidationError as e:
        print(f"  Correctly rejected comma in spending_changes: {e}")

    # Reject > 3 changes
    try:
        validate_spending_changes("stop:event_1|stop:event_2|stop:event_3|stop:event_4")
        assert False, "Should reject > 3 changes"
    except OutputValidationError as e:
        print(f"  Correctly rejected > 3 changes: {e}")

    # Reject mutually exclusive stop and reduce on same event
    try:
        validate_spending_changes("stop:event_14|reduce_to:event_14:50")
        assert False, "Should reject stop and reduce on same event"
    except OutputValidationError as e:
        print(f"  Correctly rejected stop + reduce on same event: {e}")

    # 2. Payment plan validation
    validate_payment_plan(
        plan_str="2024-03-03:25256",
        method=RecommendedPaymentMethod.FULL_PAYMENT,
        status=AffordabilityStatus.AFFORDABLE_NOW,
        amount_safe=25256,
    )
    validate_payment_plan(
        plan_str="2025-08-08:15952906.67|2025-09-07:15952906.67|2025-10-07:15952906.67",
        method=RecommendedPaymentMethod.INSTALLMENTS,
        status=AffordabilityStatus.AFFORDABLE_WITH_PLAN,
        amount_safe=17229139.2,
    )

    # Reject non-chronological payment plan
    try:
        validate_payment_plan(
            plan_str="2025-09-07:100|2025-08-08:100",
            method=RecommendedPaymentMethod.INSTALLMENTS,
            status=AffordabilityStatus.AFFORDABLE_WITH_PLAN,
            amount_safe=200,
        )
        assert False, "Should reject non-chronological dates"
    except OutputValidationError as e:
        print(f"  Correctly rejected non-chronological dates: {e}")

    # Reject plan when not_recommended
    try:
        validate_payment_plan(
            plan_str="2025-08-08:100",
            method=RecommendedPaymentMethod.NOT_RECOMMENDED,
            status=AffordabilityStatus.NOT_AFFORDABLE,
            amount_safe=0,
        )
        assert False, "Should reject non-none plan for not_recommended"
    except OutputValidationError as e:
        print(f"  Correctly rejected non-none plan for not_recommended: {e}")

    print("Validator rules tests PASSED!\n")


def test_sample_requests_validation():
    print("Testing Sample Requests against Validator...")
    dataset_dir = code_dir.parent / "dataset"
    samples = load_sample_requests(dataset_dir / "sample_requests.csv")
    assert len(samples) == 25

    # Convert sample requests into temporary output.csv format and validate!
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8", newline="") as tmp:
        import csv
        from validator import REQUIRED_COLUMNS
        writer = csv.DictWriter(tmp, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        for s in samples:
            safe_str = f"{s.amount_safe_to_pay:.2f}".rstrip("0").rstrip(".") if isinstance(s.amount_safe_to_pay, float) else str(s.amount_safe_to_pay)
            writer.writerow({
                "request_id": s.request_id,
                "amount_safe_to_pay": safe_str,
                "affordability_status": s.affordability_status.value,
                "recommended_payment_method": s.recommended_payment_method.value,
                "payment_plan": s.payment_plan,
                "earliest_date_for_full_payment": s.earliest_date_for_full_payment or "",
                "spending_changes_needed": s.spending_changes_needed,
                "decision_explanation": s.decision_explanation,
            })
        tmp_path = Path(tmp.name)

    try:
        sample_requests_as_requests = [
            Request(
                request_id=s.request_id,
                user_id=s.user_id,
                request_date=s.request_date,
                request_type=s.request_type,
                requested_amount=s.requested_amount,
                desired_completion_date=s.desired_completion_date,
                allows_partial_payment=s.allows_partial_payment,
                request_text=s.request_text,
            )
            for s in samples
        ]
        validated = validate_output_file(tmp_path, requests=sample_requests_as_requests, expected_row_count=25)
        assert len(validated) == 25
        print(f"  All 25 ground truth sample requests successfully PASSED strict validation!")
    finally:
        if tmp_path.exists():
            os.remove(tmp_path)

    print("Sample requests validation tests PASSED!\n")


def test_loaders_and_joins():
    print("Testing Loaders and Relational Joins on real dataset...")
    dataset_dir = code_dir.parent / "dataset"
    store = load_dataset(dataset_dir)

    assert len(store.requests) == 250
    assert len(store.sample_requests) == 25
    assert len(store.profiles) == 275
    assert len(store.events) == 25342
    assert len(store.exchange_rates) == 134
    assert len(store.payment_options) == 790
    assert len(store.messages) == 215
    assert len(store.images) == 16

    # Test relational context assembly for request_26
    ctx = store.get_context_for_request("request_26")
    assert ctx.request.user_id == "user_26"
    assert ctx.profile.user_id == "user_26"
    assert len(ctx.user_events) > 0
    print(f"  request_26 context joined: user={ctx.profile.user_id}, events={len(ctx.user_events)}, options={len(ctx.payment_options)}")

    # Test relational context for request with images (e.g. request_03 -> user_03 -> image_01)
    ctx_03 = store.get_context_for_request("request_33")
    assert ctx_03.request.user_id == "user_33"
    print(f"  request_33 context joined: user={ctx_03.profile.user_id}, events={len(ctx_03.user_events)}, images={len(ctx_03.images)}")

    print("Loaders and relational joins tests PASSED!\n")


def test_end_to_end_pipeline():
    print("Testing End-to-End Pipeline on real dataset...")
    dataset_dir = code_dir.parent / "dataset"
    output_file = dataset_dir / "output.csv"

    output_records = run_pipeline(dataset_dir, output_file, validate=True)
    assert len(output_records) == 250
    print("End-to-End Pipeline test PASSED!\n")


def main():
    test_currency_converter()
    test_validator_rules()
    test_sample_requests_validation()
    test_loaders_and_joins()
    test_end_to_end_pipeline()
    print("ALL TESTS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
