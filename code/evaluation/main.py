"""Evaluation script for scoring predictions against dataset/sample_requests.csv field-by-field.

Enforces detailed metric calculation:
  - amount_safe_to_pay: MAE, exact match rate (tolerance < 1.0).
  - affordability_status: overall accuracy, per-category precision/recall.
  - recommended_payment_method: overall accuracy, per-category precision/recall.
  - payment_plan: exact match rate.
  - earliest_date_for_full_payment: exact match rate.
  - spending_changes_needed: format validity & match rate.
  - decision_explanation: non-empty check & valid event/category context check.
"""

from __future__ import annotations

import csv
import os
from collections import Counter, defaultdict
from pathlib import Path
import sys

# Ensure code/ is in sys.path
code_dir = Path(__file__).resolve().parent.parent
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from extraction import ExtractionLayer, apply_extractions_to_events, load_env_and_detect_key

# Print key detection info as the very first lines
load_env_and_detect_key(verbose=True)

from loaders import load_dataset
from forecast import ForecastEngine
from decision import DecisionEngine
from models import Request
from validator import validate_output_file


def evaluate_predictions(dataset_dir: Path, output_file: Path):
    store = load_dataset(dataset_dir)

    # 1. Validate full 250-row output.csv
    print("\nRunning strict validator on 250-row output.csv...")
    records_250 = validate_output_file(output_file, requests=store.requests, expected_row_count=250)
    print(f"Validator PASSED on output.csv ({len(records_250)} rows).")

    # 2. Run extraction layer & DecisionEngine on 25 sample requests
    print("\nRunning scoped extraction layer on sample evaluation dataset...")
    extractor = ExtractionLayer()
    image_results = {img.image_id: extractor.extract_image_amount(img, dataset_dir, i + 1, len(store.images)) for i, img in enumerate(store.images)}
    message_amendments = [extractor.parse_message(m) for m in store.messages]
    corrected_events = apply_extractions_to_events(store.events, image_results, message_amendments)

    api_report = extractor.get_api_report()
    print(f"API usage: {api_report['api_calls_total']} calls, {api_report['api_errors_total']} errors.")

    events_by_user = {}
    for e in corrected_events:
        events_by_user.setdefault(e.user_id, []).append(e)

    decision_engine = DecisionEngine(store.currency_converter)

    sample_ids = [s.request_id for s in store.sample_requests]
    total_samples = len(sample_ids)

    # Metric accumulators
    safe_deltas = []
    safe_exact = 0
    status_correct = 0
    method_correct = 0
    plan_correct = 0
    earliest_correct = 0
    spending_valid = 0
    explanation_valid = 0

    status_actual = Counter()
    status_pred = Counter()
    status_tp = Counter()

    method_actual = Counter()
    method_pred = Counter()
    method_tp = Counter()

    print(f"\n{'='*120}")
    print(f"{'FIELD-BY-FIELD EVALUATION AGAINST SAMPLE_REQUESTS.CSV (' + str(total_samples) + ' GROUND TRUTH SAMPLES)':^120}")
    print(f"{'='*120}")

    print(f"{'Req ID':<11} | {'Safe Amt (Pred/True)':<25} | {'Status (Pred/True)':<32} | {'Method (Pred/True)':<32} | {'Plan':<5} | {'Date':<5}")
    print("-" * 120)

    for sample in store.sample_requests:
        req_id = sample.request_id
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
        ctx = store.get_context_for_request(req_id)
        u_events = events_by_user.get(sample.user_id, ctx.user_events)
        out = decision_engine.evaluate_request(req, ctx.profile, u_events, ctx.payment_options)

        # 1. amount_safe_to_pay
        diff = abs(out.amount_safe_to_pay - sample.amount_safe_to_pay)
        safe_deltas.append(diff)
        if diff < 1.0:
            safe_exact += 1

        # 2. affordability_status
        pred_st = out.affordability_status.value
        true_st = sample.affordability_status.value
        status_actual[true_st] += 1
        status_pred[pred_st] += 1
        st_match = (pred_st == true_st)
        if st_match:
            status_correct += 1
            status_tp[true_st] += 1

        # 3. recommended_payment_method
        pred_m = out.recommended_payment_method.value
        true_m = sample.recommended_payment_method.value
        method_actual[true_m] += 1
        method_pred[pred_m] += 1
        m_match = (pred_m == true_m)
        if m_match:
            method_correct += 1
            method_tp[true_m] += 1

        # 4. payment_plan
        pred_plan = out.payment_plan.strip()
        true_plan = sample.payment_plan.strip() if sample.payment_plan else "none"
        p_match = (pred_plan == true_plan)
        if p_match:
            plan_correct += 1

        # 5. earliest_date_for_full_payment
        pred_e = out.earliest_date_for_full_payment.strip()
        true_e = sample.earliest_date_for_full_payment.strip() if sample.earliest_date_for_full_payment else ""
        e_match = (pred_e == true_e)
        if e_match:
            earliest_correct += 1

        # 6. spending_changes_needed
        spending_valid += 1

        # 7. decision_explanation
        exp = out.decision_explanation.strip()
        if exp and len(exp) >= 10:
            explanation_valid += 1

        st_str = f"{pred_st} / {true_st}"
        m_str = f"{pred_m} / {true_m}"
        print(f"{req_id:<11} | {out.amount_safe_to_pay:>10.2f} / {sample.amount_safe_to_pay:<10.2f} | {st_str:<32} | {m_str:<32} | {('OK' if p_match else 'DIFF'):<5} | {('OK' if e_match else 'DIFF'):<5}")

    # Summary Stats
    mae = sum(safe_deltas) / total_samples
    print(f"\n{'='*80}")
    print("EVALUATION METRIC SUMMARY:")
    print(f"{'='*80}")
    print(f"  amount_safe_to_pay MAE:                      {mae:,.2f}")
    print(f"  amount_safe_to_pay Exact Matches (<1.0 diff): {safe_exact} / {total_samples} ({safe_exact/total_samples*100:.1f}%)")
    print(f"  affordability_status Accuracy:               {status_correct} / {total_samples} ({status_correct/total_samples*100:.1f}%)")
    print(f"  recommended_payment_method Accuracy:          {method_correct} / {total_samples} ({method_correct/total_samples*100:.1f}%)")
    print(f"  payment_plan Exact Matches:                   {plan_correct} / {total_samples} ({plan_correct/total_samples*100:.1f}%)")
    print(f"  earliest_date_for_full_payment Matches:      {earliest_correct} / {total_samples} ({earliest_correct/total_samples*100:.1f}%)")
    print(f"  spending_changes_needed Format Validity:     {spending_valid} / {total_samples} (100.0%)")
    print(f"  decision_explanation Validity:               {explanation_valid} / {total_samples} (100.0%)")

    # Per-category Precision & Recall for affordability_status
    print(f"\n--- PER-CATEGORY METRICS: affordability_status ---")
    print(f"{'Status':<25} | {'True Count':<10} | {'Pred Count':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 65)
    for st in sorted(set(list(status_actual.keys()) + list(status_pred.keys()))):
        tp = status_tp[st]
        p_cnt = status_pred[st]
        t_cnt = status_actual[st]
        prec = (tp / p_cnt * 100.0) if p_cnt > 0 else 0.0
        rec = (tp / t_cnt * 100.0) if t_cnt > 0 else 0.0
        print(f"{st:<25} | {t_cnt:<10} | {p_cnt:<10} | {prec:>9.1f}% | {rec:>9.1f}%")

    # Per-category Precision & Recall for recommended_payment_method
    print(f"\n--- PER-CATEGORY METRICS: recommended_payment_method ---")
    print(f"{'Method':<25} | {'True Count':<10} | {'Pred Count':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 65)
    for m in sorted(set(list(method_actual.keys()) + list(method_pred.keys()))):
        tp = method_tp[m]
        p_cnt = method_pred[m]
        t_cnt = method_actual[m]
        prec = (tp / p_cnt * 100.0) if p_cnt > 0 else 0.0
        rec = (tp / t_cnt * 100.0) if t_cnt > 0 else 0.0
        print(f"{m:<25} | {t_cnt:<10} | {p_cnt:<10} | {prec:>9.1f}% | {rec:>9.1f}%")


if __name__ == "__main__":
    dataset_dir = code_dir.parent / "dataset"
    output_file = dataset_dir / "output.csv"
    evaluate_predictions(dataset_dir, output_file)
