"""Typed data models for the Buy or Wait? financial decision agent.

Represents all dataset entities with strict typing, enums, and exact column schemas.
Zero third-party dependencies (built with Python dataclasses).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any


class RequestType(str, Enum):
    PURCHASE = "purchase"
    TRAVEL = "travel"
    EDUCATION = "education"
    FAMILY_TRANSFER = "family_transfer"
    DEBT_REPAYMENT = "debt_repayment"
    INVESTMENT = "investment"
    HOUSING = "housing"
    EMERGENCY_EXPENSE = "emergency_expense"
    OTHER = "other"


class AffordabilityStatus(str, Enum):
    AFFORDABLE_NOW = "affordable_now"
    AFFORDABLE_WITH_PLAN = "affordable_with_plan"
    AFFORDABLE_LATER = "affordable_later"
    NOT_AFFORDABLE = "not_affordable"


class RecommendedPaymentMethod(str, Enum):
    FULL_PAYMENT = "full_payment"
    PARTIAL_PAYMENT = "partial_payment"
    INSTALLMENTS = "installments"
    WAIT = "wait"
    NOT_RECOMMENDED = "not_recommended"


class EventDirection(str, Enum):
    DEBIT = "debit"
    CREDIT = "credit"


class EventStatus(str, Enum):
    SETTLED = "settled"
    PENDING = "pending"
    SCHEDULED = "scheduled"
    CANCELLED = "cancelled"
    FAILED = "failed"
    UNREALIZED = "unrealized"


class EventFlexibility(str, Enum):
    FIXED = "fixed"
    STOPPABLE = "stoppable"
    REDUCIBLE = "reducible"
    FLEXIBLE = "flexible"


@dataclass(frozen=True)
class Request:
    """Represents a purchase/expense request in dataset/requests.csv."""
    request_id: str
    user_id: str
    request_date: str  # YYYY-MM-DD
    request_type: str  # RequestType value
    requested_amount: float
    desired_completion_date: str  # YYYY-MM-DD
    allows_partial_payment: bool
    request_text: str


@dataclass(frozen=True)
class SampleRequest:
    """Represents a sample request with completed ground truth fields in dataset/sample_requests.csv."""
    request_id: str
    user_id: str
    request_date: str
    request_type: str
    requested_amount: float
    desired_completion_date: str
    allows_partial_payment: bool
    request_text: str
    amount_safe_to_pay: float
    affordability_status: AffordabilityStatus
    recommended_payment_method: RecommendedPaymentMethod
    payment_plan: str
    earliest_date_for_full_payment: Optional[str]
    spending_changes_needed: str
    decision_explanation: str


@dataclass(frozen=True)
class FinancialProfile:
    """Represents user financial preferences and state in dataset/financial_profiles.csv."""
    user_id: str
    home_currency: str
    current_available_balance: float
    minimum_balance_to_keep: float
    financial_priorities: str
    expense_categories_to_protect: str
    expense_categories_user_is_willing_to_reduce: str
    expense_categories_user_is_willing_to_stop: str
    payment_methods_user_will_consider: str
    max_installment_months: Optional[int]

    @property
    def priorities_list(self) -> List[str]:
        return [p.strip() for p in self.financial_priorities.split("|") if p.strip()]

    @property
    def protect_categories_list(self) -> List[str]:
        return [c.strip() for c in self.expense_categories_to_protect.split("|") if c.strip()]

    @property
    def reduce_categories_list(self) -> List[str]:
        return [c.strip() for c in self.expense_categories_user_is_willing_to_reduce.split("|") if c.strip()]

    @property
    def stop_categories_list(self) -> List[str]:
        return [c.strip() for c in self.expense_categories_user_is_willing_to_stop.split("|") if c.strip()]

    @property
    def payment_methods_list(self) -> List[str]:
        return [m.strip() for m in self.payment_methods_user_will_consider.split("|") if m.strip()]


@dataclass(frozen=True)
class FinancialEvent:
    """Represents a historical, pending, or scheduled transaction in dataset/financial_events.csv."""
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str  # EventDirection value
    amount: Optional[float]  # None if blank (to be resolved via images.csv)
    currency: str
    event_date: str  # YYYY-MM-DD
    settlement_date: str  # YYYY-MM-DD
    status: str  # EventStatus value
    linked_event_id: Optional[str]
    flexibility: str  # EventFlexibility value
    minimum_allowed_amount: Optional[float]


@dataclass(frozen=True)
class ExchangeRate:
    """Represents a dated currency conversion rate in dataset/exchange_rates.csv."""
    rate_date: str  # YYYY-MM-DD
    from_currency: str
    to_currency: str
    rate: float


@dataclass(frozen=True)
class PaymentOption:
    """Represents an available seller/financing payment option in dataset/request_payment_options.csv."""
    payment_option_id: str
    request_id: str
    payment_method: str
    payment_amount: float
    number_of_payments: int
    first_payment_date: str  # YYYY-MM-DD
    payment_frequency_days: Optional[int]
    financing_fee: float
    total_payable_amount: float


@dataclass(frozen=True)
class Message:
    """Represents a supporting notification/message in dataset/messages.csv."""
    message_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]
    sent_at: str  # ISO-8601
    source_type: str
    message_text: str


@dataclass(frozen=True)
class ImageRecord:
    """Represents an image reference in dataset/images.csv."""
    image_id: str
    user_id: str
    request_id: str
    related_event_id: str

    @property
    def relative_file_path(self) -> str:
        return f"media/images/{self.image_id}.png"


@dataclass(frozen=True)
class OutputRecord:
    """Represents a single prediction row conforming to dataset/output.csv schema."""
    request_id: str
    amount_safe_to_pay: float
    affordability_status: AffordabilityStatus
    recommended_payment_method: RecommendedPaymentMethod
    payment_plan: str
    earliest_date_for_full_payment: str  # YYYY-MM-DD or empty string
    spending_changes_needed: str
    decision_explanation: str

    def to_csv_row(self) -> Dict[str, str]:
        # Amount formatting: format integers without decimal point if integer, or with exact decimals
        safe_amt_str = f"{self.amount_safe_to_pay:.2f}".rstrip("0").rstrip(".") if isinstance(self.amount_safe_to_pay, float) else str(self.amount_safe_to_pay)
        return {
            "request_id": self.request_id,
            "amount_safe_to_pay": safe_amt_str,
            "affordability_status": self.affordability_status.value if isinstance(self.affordability_status, AffordabilityStatus) else str(self.affordability_status),
            "recommended_payment_method": self.recommended_payment_method.value if isinstance(self.recommended_payment_method, RecommendedPaymentMethod) else str(self.recommended_payment_method),
            "payment_plan": self.payment_plan,
            "earliest_date_for_full_payment": self.earliest_date_for_full_payment or "",
            "spending_changes_needed": self.spending_changes_needed,
            "decision_explanation": self.decision_explanation,
        }
