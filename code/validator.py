"""Strict output schema and financial constraint validator for output.csv.

Enforces:
  - Exact column order and naming.
  - Strict enum validation for affordability_status and recommended_payment_method.
  - Quantitative bound: 0 <= amount_safe_to_pay <= requested_amount.
  - Chronological ISO date format for payment_plan entries.
  - Strict syntax for spending_changes_needed (stop:event_id, reduce_to:event_id:amount, max 3, comma-free).
  - Mutual exclusivity of stop and reduce on the same event.
  - Semantic consistency rules between affordability_status, payment_plan, and earliest_date_for_full_payment.

Rejects invalid output hard with detailed OutputValidationError.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from models import (
    AffordabilityStatus,
    OutputRecord,
    RecommendedPaymentMethod,
    Request,
)


class OutputValidationError(Exception):
    """Raised when output.csv violates column schema, enums, format, or financial constraints."""
    pass


REQUIRED_COLUMNS: List[str] = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]

DATE_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}$")
PAYMENT_PLAN_ENTRY_REGEX = re.compile(r"^(\d{4}-\d{2}-\d{2}):([0-9]+(?:\.[0-9]+)?)$")
STOP_REGEX = re.compile(r"^stop:([a-zA-Z0-9_]+)$")
REDUCE_REGEX = re.compile(r"^reduce_to:([a-zA-Z0-9_]+):([0-9]+(?:\.[0-9]+)?)$")


def _parse_iso_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def validate_payment_plan(
    plan_str: str,
    method: RecommendedPaymentMethod,
    status: AffordabilityStatus,
    amount_safe: float,
    request: Optional[Request] = None,
    earliest_date: Optional[str] = None,
) -> None:
    """Validate payment_plan syntax, chronology, and method-specific rules."""
    plan_str = plan_str.strip()
    if not plan_str:
        raise OutputValidationError("payment_plan cannot be empty; use 'none' when no plan applies.")

    if plan_str == "none":
        if method in (RecommendedPaymentMethod.PARTIAL_PAYMENT, RecommendedPaymentMethod.INSTALLMENTS):
            raise OutputValidationError(f"payment_plan cannot be 'none' when recommended_payment_method is '{method.value}'.")
        return

    # If method is not_recommended, plan must be none
    if method == RecommendedPaymentMethod.NOT_RECOMMENDED:
        raise OutputValidationError(f"payment_plan must be 'none' when recommended_payment_method is 'not_recommended', got '{plan_str}'.")

    # Parse pipe-separated entries
    entries = plan_str.split("|")
    parsed_entries = []
    prev_date: Optional[datetime] = None

    for idx, entry in enumerate(entries):
        match = PAYMENT_PLAN_ENTRY_REGEX.match(entry.strip())
        if not match:
            raise OutputValidationError(
                f"Invalid payment_plan entry '{entry}' at index {idx}. Must match YYYY-MM-DD:amount format."
            )
        date_str, amt_str = match.group(1), match.group(2)
        try:
            curr_date = _parse_iso_date(date_str)
        except ValueError as e:
            raise OutputValidationError(f"Invalid calendar date '{date_str}' in payment_plan: {e}") from e

        amt = float(amt_str)
        if amt <= 0:
            raise OutputValidationError(f"Payment amount must be positive, got {amt} in '{entry}'.")

        # Chronological order enforcement (date_i <= date_{i+1})
        if prev_date is not None and curr_date < prev_date:
            raise OutputValidationError(
                f"payment_plan dates must be in chronological order: '{date_str}' appears after '{prev_date.strftime('%Y-%m-%d')}'."
            )
        prev_date = curr_date
        parsed_entries.append((date_str, amt))

    # Partial payment specific contract
    if method == RecommendedPaymentMethod.PARTIAL_PAYMENT:
        if status != AffordabilityStatus.AFFORDABLE_WITH_PLAN:
            raise OutputValidationError(
                f"partial_payment requires affordability_status 'affordable_with_plan', got '{status.value}'."
            )
        if len(parsed_entries) != 2:
            raise OutputValidationError(
                f"partial_payment requires exactly 2 payments, got {len(parsed_entries)} in '{plan_str}'."
            )

        if request is not None:
            if not (0 < amount_safe < request.requested_amount):
                raise OutputValidationError(
                    f"partial_payment requires 0 < amount_safe_to_pay < requested_amount ({request.requested_amount}), got {amount_safe}."
                )

            d1, a1 = parsed_entries[0]
            d2, a2 = parsed_entries[1]

            if d1 != request.request_date:
                raise OutputValidationError(
                    f"partial_payment first payment date must match request_date '{request.request_date}', got '{d1}'."
                )
            if abs(a1 - amount_safe) > 0.05:
                raise OutputValidationError(
                    f"partial_payment first payment amount must match amount_safe_to_pay ({amount_safe}), got {a1}."
                )
            if earliest_date and d2 != earliest_date:
                raise OutputValidationError(
                    f"partial_payment second payment date must match earliest_date_for_full_payment '{earliest_date}', got '{d2}'."
                )
            if d2 > request.desired_completion_date:
                raise OutputValidationError(
                    f"partial_payment second payment date '{d2}' exceeds desired_completion_date '{request.desired_completion_date}'."
                )
            if abs((a1 + a2) - request.requested_amount) > 0.05:
                raise OutputValidationError(
                    f"partial_payment amounts ({a1} + {a2} = {a1 + a2}) must sum to requested_amount ({request.requested_amount})."
                )


def validate_spending_changes(spending_str: str) -> None:
    """Validate spending_changes_needed syntax, comma-free rule, and mutual exclusivity."""
    s = spending_str.strip()
    if not s:
        raise OutputValidationError("spending_changes_needed cannot be empty; use 'none' when no changes are needed.")

    if "," in s:
        raise OutputValidationError(
            f"spending_changes_needed must be comma-free (entries separated by '|' only), got: '{s}'"
        )

    if s == "none":
        return

    items = s.split("|")
    if len(items) > 3:
        raise OutputValidationError(
            f"spending_changes_needed allows at most 3 changes, got {len(items)} in '{s}'."
        )

    stopped_events: Set[str] = set()
    reduced_events: Set[str] = set()

    for item in items:
        item = item.strip()
        stop_match = STOP_REGEX.match(item)
        reduce_match = REDUCE_REGEX.match(item)

        if stop_match:
            ev_id = stop_match.group(1)
            if ev_id in stopped_events:
                raise OutputValidationError(f"Duplicate stop for event '{ev_id}' in spending_changes_needed.")
            stopped_events.add(ev_id)
        elif reduce_match:
            ev_id = reduce_match.group(1)
            new_amt = float(reduce_match.group(2))
            if new_amt < 0:
                raise OutputValidationError(f"reduce_to amount cannot be negative, got {new_amt} in '{item}'.")
            if ev_id in reduced_events:
                raise OutputValidationError(f"Duplicate reduce_to for event '{ev_id}' in spending_changes_needed.")
            reduced_events.add(ev_id)
        else:
            raise OutputValidationError(
                f"Invalid spending change format '{item}'. Must be 'stop:<event_id>' or 'reduce_to:<event_id>:<amount>'."
            )

    # Mutual exclusivity check
    overlap = stopped_events.intersection(reduced_events)
    if overlap:
        raise OutputValidationError(
            f"Stopping and reducing the same event is mutually exclusive: events {overlap} appear in both."
        )


def validate_output_record(
    record: OutputRecord,
    request: Optional[Request] = None,
    row_idx: int = 2,
) -> None:
    """Validate an individual OutputRecord against problem rules and request constraints."""
    # 1. amount_safe_to_pay >= 0
    if record.amount_safe_to_pay < 0:
        raise OutputValidationError(
            f"Row {row_idx} ({record.request_id}): amount_safe_to_pay must be non-negative, got {record.amount_safe_to_pay}."
        )

    # 2. amount_safe_to_pay <= requested_amount
    if request is not None:
        if record.request_id != request.request_id:
            raise OutputValidationError(
                f"Row {row_idx}: request_id mismatch: expected '{request.request_id}', got '{record.request_id}'."
            )
        if record.amount_safe_to_pay > request.requested_amount + 1e-4:
            raise OutputValidationError(
                f"Row {row_idx} ({record.request_id}): amount_safe_to_pay ({record.amount_safe_to_pay}) exceeds requested_amount ({request.requested_amount})."
            )

    # 3. earliest_date_for_full_payment
    earliest = record.earliest_date_for_full_payment.strip() if record.earliest_date_for_full_payment else ""
    if earliest:
        if not DATE_REGEX.match(earliest):
            raise OutputValidationError(
                f"Row {row_idx} ({record.request_id}): earliest_date_for_full_payment must be YYYY-MM-DD or empty, got '{earliest}'."
            )

    if record.affordability_status == AffordabilityStatus.AFFORDABLE_NOW:
        if request is not None and earliest != request.request_date:
            raise OutputValidationError(
                f"Row {row_idx} ({record.request_id}): For 'affordable_now', earliest_date_for_full_payment must equal request_date ('{request.request_date}'), got '{earliest}'."
            )

    # 4. payment_plan
    try:
        validate_payment_plan(
            plan_str=record.payment_plan,
            method=record.recommended_payment_method,
            status=record.affordability_status,
            amount_safe=record.amount_safe_to_pay,
            request=request,
            earliest_date=earliest if earliest else None,
        )
    except OutputValidationError as e:
        raise OutputValidationError(f"Row {row_idx} ({record.request_id}): {e}") from e

    # 5. spending_changes_needed
    try:
        validate_spending_changes(record.spending_changes_needed)
    except OutputValidationError as e:
        raise OutputValidationError(f"Row {row_idx} ({record.request_id}): {e}") from e

    # 6. decision_explanation
    if not record.decision_explanation.strip():
        raise OutputValidationError(
            f"Row {row_idx} ({record.request_id}): decision_explanation cannot be empty."
        )


def validate_output_file(
    file_path: str | Path,
    requests: Optional[List[Request]] = None,
    expected_row_count: Optional[int] = 250,
) -> List[OutputRecord]:
    """Strictly validate output.csv file.
    
    Verifies:
      - File existence and non-empty.
      - Exact header row matching REQUIRED_COLUMNS in exact order.
      - Row count matching expected_row_count (default 250) and len(requests).
      - Every data row adhering to typed enums, bounds, and formatting contracts.
    
    Returns:
      List[OutputRecord] if all validation checks pass.
    
    Raises:
      OutputValidationError if any validation check fails.
    """
    path = Path(file_path)
    if not path.exists():
        raise OutputValidationError(f"Output file does not exist: {path}")

    requests_by_id = {r.request_id: r for r in requests} if requests else {}

    records: List[OutputRecord] = []
    with open(path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise OutputValidationError(f"Output file {path.name} is completely empty.")

        # Header check: exact column order and names
        if header != REQUIRED_COLUMNS:
            diff_msg = f"Expected columns:\n  {REQUIRED_COLUMNS}\nGot:\n  {header}"
            raise OutputValidationError(f"Invalid header columns or column order in {path.name}.\n{diff_msg}")

        for row_idx, row in enumerate(reader, start=2):
            if len(row) != len(REQUIRED_COLUMNS):
                raise OutputValidationError(
                    f"Row {row_idx} has {len(row)} columns, expected {len(REQUIRED_COLUMNS)}: {row}"
                )

            req_id = row[0].strip()
            amt_safe_str = row[1].strip()
            status_str = row[2].strip()
            method_str = row[3].strip()
            plan_str = row[4].strip()
            earliest_str = row[5].strip()
            spending_str = row[6].strip()
            explanation_str = row[7].strip()

            # Parse float
            try:
                amt_safe = float(amt_safe_str)
            except ValueError as e:
                raise OutputValidationError(
                    f"Row {row_idx} ({req_id}): amount_safe_to_pay '{amt_safe_str}' is not a valid float: {e}"
                ) from e

            # Parse AffordabilityStatus enum
            try:
                status_enum = AffordabilityStatus(status_str)
            except ValueError as e:
                valid_statuses = [s.value for s in AffordabilityStatus]
                raise OutputValidationError(
                    f"Row {row_idx} ({req_id}): invalid affordability_status '{status_str}'. Must be one of {valid_statuses}."
                ) from e

            # Parse RecommendedPaymentMethod enum
            try:
                method_enum = RecommendedPaymentMethod(method_str)
            except ValueError as e:
                valid_methods = [m.value for m in RecommendedPaymentMethod]
                raise OutputValidationError(
                    f"Row {row_idx} ({req_id}): invalid recommended_payment_method '{method_str}'. Must be one of {valid_methods}."
                ) from e

            record = OutputRecord(
                request_id=req_id,
                amount_safe_to_pay=amt_safe,
                affordability_status=status_enum,
                recommended_payment_method=method_enum,
                payment_plan=plan_str,
                earliest_date_for_full_payment=earliest_str,
                spending_changes_needed=spending_str,
                decision_explanation=explanation_str,
            )

            # Match against request
            matching_req = None
            if requests:
                req_index = row_idx - 2
                if req_index < len(requests):
                    matching_req = requests[req_index]
                else:
                    matching_req = requests_by_id.get(req_id)

            validate_output_record(record, matching_req, row_idx=row_idx)
            records.append(record)

    # Row count verification
    if expected_row_count is not None and len(records) != expected_row_count:
        raise OutputValidationError(
            f"Row count mismatch in {path.name}: expected {expected_row_count} rows, got {len(records)}."
        )

    if requests is not None and len(records) != len(requests):
        raise OutputValidationError(
            f"Row count in {path.name} ({len(records)}) does not match requests count ({len(requests)})."
        )

    return records
