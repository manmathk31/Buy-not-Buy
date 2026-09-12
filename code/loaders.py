"""Loaders and relational joiners for the Buy or Wait? challenge dataset.

Enforces strict row count validation and fails loudly on any malformed or unparseable row.
Provides relational indexing by user_id, request_id, and related_event_id as specified in AGENTS.md §6.1.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from models import (
    AffordabilityStatus,
    EventDirection,
    EventFlexibility,
    EventStatus,
    ExchangeRate,
    FinancialEvent,
    FinancialProfile,
    ImageRecord,
    Message,
    PaymentOption,
    RecommendedPaymentMethod,
    Request,
    RequestType,
    SampleRequest,
)
from currency import CurrencyConverter


class DatasetParseError(Exception):
    """Raised when a CSV row cannot be parsed or parsed fields fail validation."""
    pass


class DatasetRowCountError(Exception):
    """Raised when a CSV file does not contain the expected number of data rows."""
    pass


EXPECTED_ROW_COUNTS = {
    "requests.csv": 250,
    "sample_requests.csv": 25,
    "financial_profiles.csv": 275,
    "financial_events.csv": 25342,
    "exchange_rates.csv": 134,
    "request_payment_options.csv": 790,
    "messages.csv": 215,
    "images.csv": 16,
}


def _check_row_count(file_name: str, actual_count: int, expected_count: Optional[int]) -> None:
    if expected_count is not None and actual_count != expected_count:
        raise DatasetRowCountError(
            f"Row count mismatch in {file_name}: expected {expected_count} rows, found {actual_count}."
        )


def load_requests(file_path: str | Path, validate_count: bool = True) -> List[Request]:
    """Load requests.csv into typed Request models."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    records: List[Request] = []
    with open(path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=2):
            try:
                records.append(
                    Request(
                        request_id=row["request_id"].strip(),
                        user_id=row["user_id"].strip(),
                        request_date=row["request_date"].strip(),
                        request_type=row["request_type"].strip(),
                        requested_amount=float(row["requested_amount"]),
                        desired_completion_date=row["desired_completion_date"].strip(),
                        allows_partial_payment=row["allows_partial_payment"].strip().lower() in ("true", "1"),
                        request_text=row["request_text"].strip(),
                    )
                )
            except Exception as e:
                raise DatasetParseError(f"Failed parsing {path.name} at line {row_idx}: {row} -> {e}") from e

    if validate_count:
        _check_row_count(path.name, len(records), EXPECTED_ROW_COUNTS.get("requests.csv"))
    return records


def load_sample_requests(file_path: str | Path, validate_count: bool = True) -> List[SampleRequest]:
    """Load sample_requests.csv into typed SampleRequest models with ground truth."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    records: List[SampleRequest] = []
    with open(path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=2):
            try:
                earliest = row.get("earliest_date_for_full_payment", "").strip()
                records.append(
                    SampleRequest(
                        request_id=row["request_id"].strip(),
                        user_id=row["user_id"].strip(),
                        request_date=row["request_date"].strip(),
                        request_type=row["request_type"].strip(),
                        requested_amount=float(row["requested_amount"]),
                        desired_completion_date=row["desired_completion_date"].strip(),
                        allows_partial_payment=row["allows_partial_payment"].strip().lower() in ("true", "1"),
                        request_text=row["request_text"].strip(),
                        amount_safe_to_pay=float(row["amount_safe_to_pay"]),
                        affordability_status=AffordabilityStatus(row["affordability_status"].strip()),
                        recommended_payment_method=RecommendedPaymentMethod(row["recommended_payment_method"].strip()),
                        payment_plan=row["payment_plan"].strip(),
                        earliest_date_for_full_payment=earliest if earliest else None,
                        spending_changes_needed=row["spending_changes_needed"].strip(),
                        decision_explanation=row["decision_explanation"].strip(),
                    )
                )
            except Exception as e:
                raise DatasetParseError(f"Failed parsing {path.name} at line {row_idx}: {row} -> {e}") from e

    if validate_count:
        _check_row_count(path.name, len(records), EXPECTED_ROW_COUNTS.get("sample_requests.csv"))
    return records


def load_financial_profiles(file_path: str | Path, validate_count: bool = True) -> List[FinancialProfile]:
    """Load financial_profiles.csv into typed FinancialProfile models."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    records: List[FinancialProfile] = []
    with open(path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=2):
            try:
                max_inst = row.get("max_installment_months", "").strip()
                records.append(
                    FinancialProfile(
                        user_id=row["user_id"].strip(),
                        home_currency=row["home_currency"].strip().upper(),
                        current_available_balance=float(row["current_available_balance"]),
                        minimum_balance_to_keep=float(row["minimum_balance_to_keep"]),
                        financial_priorities=row.get("financial_priorities", "").strip(),
                        expense_categories_to_protect=row.get("expense_categories_to_protect", "").strip(),
                        expense_categories_user_is_willing_to_reduce=row.get("expense_categories_user_is_willing_to_reduce", "").strip(),
                        expense_categories_user_is_willing_to_stop=row.get("expense_categories_user_is_willing_to_stop", "").strip(),
                        payment_methods_user_will_consider=row.get("payment_methods_user_will_consider", "").strip(),
                        max_installment_months=int(max_inst) if max_inst else None,
                    )
                )
            except Exception as e:
                raise DatasetParseError(f"Failed parsing {path.name} at line {row_idx}: {row} -> {e}") from e

    if validate_count:
        _check_row_count(path.name, len(records), EXPECTED_ROW_COUNTS.get("financial_profiles.csv"))
    return records


def load_financial_events(file_path: str | Path, validate_count: bool = True) -> List[FinancialEvent]:
    """Load financial_events.csv into typed FinancialEvent models."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    records: List[FinancialEvent] = []
    with open(path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=2):
            try:
                amt_str = row.get("amount", "").strip()
                amt = float(amt_str) if amt_str else None

                min_amt_str = row.get("minimum_allowed_amount", "").strip()
                min_amt = float(min_amt_str) if min_amt_str else None

                linked_id = row.get("linked_event_id", "").strip()

                records.append(
                    FinancialEvent(
                        event_id=row["event_id"].strip(),
                        user_id=row["user_id"].strip(),
                        event_type=row["event_type"].strip(),
                        description=row["description"].strip(),
                        category=row["category"].strip(),
                        direction=row["direction"].strip().lower(),
                        amount=amt,
                        currency=row["currency"].strip().upper(),
                        event_date=row["event_date"].strip(),
                        settlement_date=row["settlement_date"].strip(),
                        status=row["status"].strip().lower(),
                        linked_event_id=linked_id if linked_id else None,
                        flexibility=row["flexibility"].strip().lower(),
                        minimum_allowed_amount=min_amt,
                    )
                )
            except Exception as e:
                raise DatasetParseError(f"Failed parsing {path.name} at line {row_idx}: {row} -> {e}") from e

    if validate_count:
        _check_row_count(path.name, len(records), EXPECTED_ROW_COUNTS.get("financial_events.csv"))
    return records


def load_exchange_rates(file_path: str | Path, validate_count: bool = True) -> List[ExchangeRate]:
    """Load exchange_rates.csv into typed ExchangeRate models."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    records: List[ExchangeRate] = []
    with open(path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=2):
            try:
                records.append(
                    ExchangeRate(
                        rate_date=row["rate_date"].strip(),
                        from_currency=row["from_currency"].strip().upper(),
                        to_currency=row["to_currency"].strip().upper(),
                        rate=float(row["rate"]),
                    )
                )
            except Exception as e:
                raise DatasetParseError(f"Failed parsing {path.name} at line {row_idx}: {row} -> {e}") from e

    if validate_count:
        _check_row_count(path.name, len(records), EXPECTED_ROW_COUNTS.get("exchange_rates.csv"))
    return records


def load_payment_options(file_path: str | Path, validate_count: bool = True) -> List[PaymentOption]:
    """Load request_payment_options.csv into typed PaymentOption models."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    records: List[PaymentOption] = []
    with open(path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=2):
            try:
                freq_str = row.get("payment_frequency_days", "").strip()
                records.append(
                    PaymentOption(
                        payment_option_id=row["payment_option_id"].strip(),
                        request_id=row["request_id"].strip(),
                        payment_method=row["payment_method"].strip(),
                        payment_amount=float(row["payment_amount"]),
                        number_of_payments=int(row["number_of_payments"]),
                        first_payment_date=row["first_payment_date"].strip(),
                        payment_frequency_days=int(freq_str) if freq_str else None,
                        financing_fee=float(row["financing_fee"]),
                        total_payable_amount=float(row["total_payable_amount"]),
                    )
                )
            except Exception as e:
                raise DatasetParseError(f"Failed parsing {path.name} at line {row_idx}: {row} -> {e}") from e

    if validate_count:
        _check_row_count(path.name, len(records), EXPECTED_ROW_COUNTS.get("request_payment_options.csv"))
    return records


def load_messages(file_path: str | Path, validate_count: bool = True) -> List[Message]:
    """Load messages.csv into typed Message models."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    records: List[Message] = []
    with open(path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=2):
            try:
                req_id = row.get("request_id", "").strip()
                rel_event = row.get("related_event_id", "").strip()
                records.append(
                    Message(
                        message_id=row["message_id"].strip(),
                        user_id=row["user_id"].strip(),
                        request_id=req_id if req_id else None,
                        related_event_id=rel_event if rel_event else None,
                        sent_at=row["sent_at"].strip(),
                        source_type=row["source_type"].strip(),
                        message_text=row["message_text"].strip(),
                    )
                )
            except Exception as e:
                raise DatasetParseError(f"Failed parsing {path.name} at line {row_idx}: {row} -> {e}") from e

    if validate_count:
        _check_row_count(path.name, len(records), EXPECTED_ROW_COUNTS.get("messages.csv"))
    return records


def load_images(file_path: str | Path, validate_count: bool = True) -> List[ImageRecord]:
    """Load images.csv into typed ImageRecord models."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    records: List[ImageRecord] = []
    with open(path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=2):
            try:
                records.append(
                    ImageRecord(
                        image_id=row["image_id"].strip(),
                        user_id=row["user_id"].strip(),
                        request_id=row["request_id"].strip(),
                        related_event_id=row["related_event_id"].strip(),
                    )
                )
            except Exception as e:
                raise DatasetParseError(f"Failed parsing {path.name} at line {row_idx}: {row} -> {e}") from e

    if validate_count:
        _check_row_count(path.name, len(records), EXPECTED_ROW_COUNTS.get("images.csv"))
    return records


@dataclass
class RequestContext:
    """Joined context for evaluating a single request."""
    request: Request
    profile: FinancialProfile
    user_events: List[FinancialEvent]
    payment_options: List[PaymentOption]
    messages: List[Message]
    images: List[ImageRecord]
    currency_converter: CurrencyConverter
    events_by_id: Dict[str, FinancialEvent]
    messages_by_event: Dict[str, List[Message]]
    images_by_event: Dict[str, List[ImageRecord]]

    def get_event(self, event_id: str) -> Optional[FinancialEvent]:
        return self.events_by_id.get(event_id)

    def get_messages_for_event(self, event_id: str) -> List[Message]:
        return self.messages_by_event.get(event_id, [])

    def get_images_for_event(self, event_id: str) -> List[ImageRecord]:
        return self.images_by_event.get(event_id, [])


class DatasetStore:
    """In-memory indexed store of all dataset entities joined relationally."""

    def __init__(
        self,
        requests: List[Request],
        sample_requests: List[SampleRequest],
        profiles: List[FinancialProfile],
        events: List[FinancialEvent],
        exchange_rates: List[ExchangeRate],
        payment_options: List[PaymentOption],
        messages: List[Message],
        images: List[ImageRecord],
    ):
        self.requests = requests
        self.sample_requests = sample_requests
        self.profiles = profiles
        self.events = events
        self.exchange_rates = exchange_rates
        self.payment_options = payment_options
        self.messages = messages
        self.images = images

        # Converter
        self.currency_converter = CurrencyConverter(self.exchange_rates)

        # Relational Indexes
        self.requests_by_id: Dict[str, Request] = {r.request_id: r for r in self.requests}
        self.sample_requests_by_id: Dict[str, SampleRequest] = {r.request_id: r for r in self.sample_requests}
        self.profiles_by_user: Dict[str, FinancialProfile] = {p.user_id: p for p in self.profiles}

        # Events indexed by user_id and event_id
        self.events_by_user: Dict[str, List[FinancialEvent]] = {}
        self.events_by_id: Dict[str, FinancialEvent] = {}
        for ev in self.events:
            self.events_by_user.setdefault(ev.user_id, []).append(ev)
            self.events_by_id[ev.event_id] = ev

        # Payment options by request_id
        self.payment_options_by_request: Dict[str, List[PaymentOption]] = {}
        for opt in self.payment_options:
            self.payment_options_by_request.setdefault(opt.request_id, []).append(opt)

        # Messages indexed by user_id, request_id, related_event_id
        self.messages_by_user: Dict[str, List[Message]] = {}
        self.messages_by_request: Dict[str, List[Message]] = {}
        self.messages_by_event: Dict[str, List[Message]] = {}
        for msg in self.messages:
            self.messages_by_user.setdefault(msg.user_id, []).append(msg)
            if msg.request_id:
                self.messages_by_request.setdefault(msg.request_id, []).append(msg)
            if msg.related_event_id:
                self.messages_by_event.setdefault(msg.related_event_id, []).append(msg)

        # Images indexed by user_id, request_id, related_event_id
        self.images_by_user: Dict[str, List[ImageRecord]] = {}
        self.images_by_request: Dict[str, List[ImageRecord]] = {}
        self.images_by_event: Dict[str, List[ImageRecord]] = {}
        for img in self.images:
            self.images_by_user.setdefault(img.user_id, []).append(img)
            self.images_by_request.setdefault(img.request_id, []).append(img)
            self.images_by_event.setdefault(img.related_event_id, []).append(img)

    def get_context_for_request(self, request_id: str) -> RequestContext:
        """Assemble fully joined context for evaluation of request_id."""
        req = self.requests_by_id.get(request_id)
        if not req:
            sample_req = self.sample_requests_by_id.get(request_id)
            if sample_req:
                req = Request(
                    request_id=sample_req.request_id,
                    user_id=sample_req.user_id,
                    request_date=sample_req.request_date,
                    request_type=sample_req.request_type,
                    requested_amount=sample_req.requested_amount,
                    desired_completion_date=sample_req.desired_completion_date,
                    allows_partial_payment=sample_req.allows_partial_payment,
                    request_text=sample_req.request_text,
                )
            else:
                raise KeyError(f"Request '{request_id}' not found in requests.csv or sample_requests.csv")

        user_id = req.user_id
        profile = self.profiles_by_user.get(user_id)
        if not profile:
            raise KeyError(f"Profile for user '{user_id}' (from request '{request_id}') not found in financial_profiles.csv")

        user_events = self.events_by_user.get(user_id, [])
        options = self.payment_options_by_request.get(request_id, [])

        # Messages related to this request or user
        # Note: messages directly tagged with request_id, or general user messages
        req_messages = self.messages_by_request.get(request_id, [])
        user_messages = self.messages_by_user.get(user_id, [])
        combined_messages = list({m.message_id: m for m in (req_messages + user_messages)}.values())

        # Images related to this request or user
        req_images = self.images_by_request.get(request_id, [])
        user_images = self.images_by_user.get(user_id, [])
        combined_images = list({img.image_id: img for img in (req_images + user_images)}.values())

        return RequestContext(
            request=req,
            profile=profile,
            user_events=user_events,
            payment_options=options,
            messages=combined_messages,
            images=combined_images,
            currency_converter=self.currency_converter,
            events_by_id=self.events_by_id,
            messages_by_event=self.messages_by_event,
            images_by_event=self.images_by_event,
        )


def load_dataset(dataset_dir: str | Path) -> DatasetStore:
    """Load and index all CSV files from the specified dataset directory."""
    d = Path(dataset_dir)
    return DatasetStore(
        requests=load_requests(d / "requests.csv"),
        sample_requests=load_sample_requests(d / "sample_requests.csv"),
        profiles=load_financial_profiles(d / "financial_profiles.csv"),
        events=load_financial_events(d / "financial_events.csv"),
        exchange_rates=load_exchange_rates(d / "exchange_rates.csv"),
        payment_options=load_payment_options(d / "request_payment_options.csv"),
        messages=load_messages(d / "messages.csv"),
        images=load_images(d / "images.csv"),
    )
