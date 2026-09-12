"""Evaluation stub for scoring output.csv against sample_requests.csv.

Provides function signatures and placeholders for per-field evaluation:
  - amount_safe_to_pay (MAE, exact match, tolerance threshold)
  - affordability_status (Overall accuracy, per-category breakdown: affordable_now, affordable_with_plan, affordable_later, not_affordable)
  - recommended_payment_method (Overall accuracy, per-category breakdown: full_payment, partial_payment, installments, wait, not_recommended)
  - payment_plan (Exact match, date accuracy, payment schedule alignment)
  - earliest_date_for_full_payment (Exact match, days-difference error)
  - spending_changes_needed (Set overlap, precision/recall on stopped/reduced events)
  - decision_explanation (Semantic grounding, key fact coverage)

TODO: Scoring algorithm will be implemented in a subsequent iteration.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent code/ directory to path if run directly
current_dir = Path(__file__).resolve().parent
parent_code_dir = current_dir.parent
if str(parent_code_dir) not in sys.path:
    sys.path.insert(0, str(parent_code_dir))

from models import AffordabilityStatus, OutputRecord, RecommendedPaymentMethod, SampleRequest
from loaders import load_sample_requests
from validator import validate_output_file


def score_amount_safe_to_pay(
    predictions: List[OutputRecord],
    ground_truth: List[SampleRequest],
) -> Dict[str, float]:
    """Score amount_safe_to_pay predictions against ground truth.
    
    TODO: Implement Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE),
    and exact match / tolerance within 1 unit.
    """
    # TODO: Implement evaluation metric
    return {
        "status": "TODO",
        "mae": 0.0,
        "exact_match_rate": 0.0,
    }


def score_affordability_status(
    predictions: List[OutputRecord],
    ground_truth: List[SampleRequest],
) -> Dict[str, Any]:
    """Score affordability_status with per-category metrics breakdown.
    
    TODO: Compute confusion matrix, overall accuracy, macro F1, and per-category
    precision / recall for:
      - affordable_now
      - affordable_with_plan
      - affordable_later
      - not_affordable
    """
    # TODO: Implement evaluation metric with per-category breakdown
    categories = [s.value for s in AffordabilityStatus]
    return {
        "status": "TODO",
        "overall_accuracy": 0.0,
        "macro_f1": 0.0,
        "per_category": {cat: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "count": 0} for cat in categories},
    }


def score_recommended_payment_method(
    predictions: List[OutputRecord],
    ground_truth: List[SampleRequest],
) -> Dict[str, Any]:
    """Score recommended_payment_method with per-category metrics breakdown.
    
    TODO: Compute confusion matrix, overall accuracy, macro F1, and per-category
    breakdown for:
      - full_payment
      - partial_payment
      - installments
      - wait
      - not_recommended
    """
    # TODO: Implement evaluation metric with per-category breakdown
    categories = [m.value for m in RecommendedPaymentMethod]
    return {
        "status": "TODO",
        "overall_accuracy": 0.0,
        "macro_f1": 0.0,
        "per_category": {cat: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "count": 0} for cat in categories},
    }


def score_payment_plan(
    predictions: List[OutputRecord],
    ground_truth: List[SampleRequest],
) -> Dict[str, float]:
    """Score payment_plan against ground truth.
    
    TODO: Parse payment installments, compare schedule dates, amounts, and payment counts.
    """
    # TODO: Implement evaluation metric
    return {
        "status": "TODO",
        "exact_match_rate": 0.0,
        "schedule_alignment_score": 0.0,
    }


def score_earliest_date_for_full_payment(
    predictions: List[OutputRecord],
    ground_truth: List[SampleRequest],
) -> Dict[str, float]:
    """Score earliest_date_for_full_payment.
    
    TODO: Compute exact match rate and mean absolute date discrepancy in days.
    """
    # TODO: Implement evaluation metric
    return {
        "status": "TODO",
        "exact_date_match": 0.0,
        "mean_absolute_days_error": 0.0,
    }


def score_spending_changes_needed(
    predictions: List[OutputRecord],
    ground_truth: List[SampleRequest],
) -> Dict[str, float]:
    """Score spending_changes_needed.
    
    TODO: Compare set of proposed actions (stop / reduce_to), calculate Jaccard index
    and precision/recall of identified events.
    """
    # TODO: Implement evaluation metric
    return {
        "status": "TODO",
        "exact_match_rate": 0.0,
        "event_jaccard_similarity": 0.0,
    }


def score_decision_explanation(
    predictions: List[OutputRecord],
    ground_truth: List[SampleRequest],
) -> Dict[str, float]:
    """Score decision_explanation for factual grounding and consistency.
    
    TODO: Measure presence of key grounded financial facts (balances, dates, minimums)
    and brevity/clarity rubric.
    """
    # TODO: Implement evaluation metric
    return {
        "status": "TODO",
        "grounding_score": 0.0,
    }


def evaluate(
    predictions: List[OutputRecord],
    ground_truth: List[SampleRequest],
) -> Dict[str, Any]:
    """Run comprehensive evaluation across all output fields.
    
    TODO: Orchestrate individual scorers and return unified metrics dictionary.
    """
    return {
        "amount_safe_to_pay": score_amount_safe_to_pay(predictions, ground_truth),
        "affordability_status": score_affordability_status(predictions, ground_truth),
        "recommended_payment_method": score_recommended_payment_method(predictions, ground_truth),
        "payment_plan": score_payment_plan(predictions, ground_truth),
        "earliest_date_for_full_payment": score_earliest_date_for_full_payment(predictions, ground_truth),
        "spending_changes_needed": score_spending_changes_needed(predictions, ground_truth),
        "decision_explanation": score_decision_explanation(predictions, ground_truth),
    }


def main() -> None:
    """CLI entry point for evaluation stub."""
    print("Evaluation module initialized. Scoring logic will be implemented in the next iteration.")


if __name__ == "__main__":
    main()
