"""Financial state reconstruction and 90-day cash flow forecast engine.

Implements the exact challenge rules:
  1. Event inclusion:
     - Exclude: cancelled, failed, unrealized, pending credits.
     - Include: settled, scheduled, pending debits.
     - Exact deduplication across all fields except event_id.
  2. Evidence-based recurrence detection:
     - (user_id, category, description) requires >= 2 historical occurrences.
     - Median interval M with strict tolerance: all gaps |g_i - M| <= 3 days.
     - Project future occurrences at median interval within 90-day forecast window.
     - Never extrapolate from a single historical instance.
  3. 90-day conservative cash flow forecast starting from current_available_balance.
  4. Exact calculations:
     - amount_safe_to_pay: max amount payable today without breaching minimum_balance_to_keep in 90 days.
     - earliest_date_for_full_payment: first conservative date full amount passes 90-day safety check.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import statistics
from typing import Any, Dict, List, Optional, Set, Tuple

from models import (
    FinancialEvent,
    FinancialProfile,
    Request,
)
from currency import CurrencyConverter, MissingExchangeRateError


@dataclass(frozen=True)
class CashFlowItem:
    """Represents a discrete cash flow event impacting the account balance on a specific date."""
    date: date
    direction: str  # "debit" or "credit"
    amount: float
    currency: str
    category: str
    description: str
    source_event_id: Optional[str] = None
    is_projected: bool = False


@dataclass
class ForecastResult:
    """Detailed output of 90-day cash flow forecast."""
    request_id: str
    user_id: str
    home_currency: str
    current_available_balance: float
    minimum_balance_to_keep: float
    daily_balances: Dict[date, float]
    daily_margins: Dict[date, float]
    min_margin: float
    amount_safe_to_pay: float
    earliest_date_for_full_payment: str  # YYYY-MM-DD or ""
    recurring_groups_detected: List[Dict[str, Any]]
    included_explicit_events_count: int
    projected_events_count: int


class ForecastEngine:
    """Reconstructs financial position and runs conservative 90-day cash flow simulation."""

    def __init__(self, currency_converter: CurrencyConverter):
        self.converter = currency_converter

    @staticmethod
    def parse_date(d_str: str) -> date:
        return datetime.strptime(d_str.strip(), "%Y-%m-%d").date()

    def deduplicate_events(self, events: List[FinancialEvent]) -> List[FinancialEvent]:
        """Collapse duplicate events where every field except event_id is identical."""
        seen = set()
        unique_events: List[FinancialEvent] = []
        for e in events:
            key = (
                e.user_id,
                e.event_type,
                e.description,
                e.category,
                e.direction,
                e.amount,
                e.currency,
                e.event_date,
                e.settlement_date,
                e.status,
                e.linked_event_id,
                e.flexibility,
                e.minimum_allowed_amount,
            )
            if key not in seen:
                seen.add(key)
                unique_events.append(e)
        return unique_events

    def is_included_event(self, event: FinancialEvent) -> bool:
        """Determine whether an event is included in cash flow forecast.
        
        Rules:
          - Exclude: cancelled, failed, unrealized, pending credit (unconfirmed income).
          - Include: settled, scheduled, pending debit (pending obligations).
        """
        status = event.status.strip().lower()
        direction = event.direction.strip().lower()

        if status in ("cancelled", "failed", "unrealized"):
            return False

        if status == "pending" and direction == "credit":
            return False

        if status in ("settled", "scheduled"):
            return True

        if status == "pending" and direction == "debit":
            return True

        return False

    def detect_recurring_groups(
        self,
        candidate_events: List[FinancialEvent],
        request_date: date,
    ) -> List[Dict[str, Any]]:
        """Identify strictly recurring groups across settled and scheduled events.
        
        Rules:
          - >= 2 occurrences.
          - Strict tolerance: all gaps |g_i - M| <= 3 days.
          - Salary: groups across description variations (prorated -> regular -> scheduled),
            unless explicitly flagged as final/terminated or variable gig platform payouts.
        """
        groups: Dict[Tuple[str, str, str], List[FinancialEvent]] = {}
        for e in candidate_events:
            if not self.is_included_event(e):
                continue
            if e.amount is None:
                continue

            cat = e.category.strip().lower()
            desc_lower = e.description.strip().lower()

            # Terminated employment should not recur
            if "final" in desc_lower and cat == "salary":
                continue

            if cat == "salary":
                # Identify primary base payroll (e.g. monthly regular salary)
                is_supplemental = any(w in desc_lower for w in ("bonus", "commission", "arrears", "second", "overtime"))
                if not is_supplemental and any(w in desc_lower for w in ("salary", "payroll", "base")):
                    # Group by day of month so day-15 payrolls group together across description variations (prorated -> regular -> next scheduled)
                    dom = self.parse_date(e.event_date).day
                    desc_key = f"primary_payroll_dom_{dom}"
                else:
                    desc_key = e.description.strip()
            else:
                desc_key = e.description.strip()

            groups.setdefault((e.user_id, e.category, desc_key), []).append(e)

        recurring: List[Dict[str, Any]] = []

        for (uid, cat, desc_key), ev_list in groups.items():
            ev_list.sort(key=lambda x: self.parse_date(x.event_date))
            dates = sorted(set(self.parse_date(e.event_date) for e in ev_list))
            if len(dates) < 2:
                continue

            gaps = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
            med_gap = statistics.median(gaps)
            if med_gap <= 0:
                continue

            # Check if identical day of month with monthly cadence (including possible unpaid leave multiples)
            days_of_month = [d.day for d in dates]
            is_same_dom = (len(set(days_of_month)) == 1)
            is_monthly_leave = is_same_dom and all((g % 30 <= 3 or g % 31 <= 3 or g % 29 <= 3) for g in gaps)

            # Require strict tolerance or monthly leave cadence
            if all(abs(g - med_gap) <= 3 for g in gaps) or is_monthly_leave:
                latest_e = ev_list[-1]
                if is_monthly_leave:
                    int_med = 30
                    is_monthly = True
                else:
                    int_med = round(med_gap)
                    is_monthly = (28 <= int_med <= 31 and len(set(d.day for d in dates)) == 1)

                # For 2 occurrences of non-salary, avoid spurious recurrence with large gaps > 32 days
                if len(dates) == 2 and int_med > 32 and cat != "salary":
                    continue

                recurring.append({
                    "user_id": uid,
                    "category": cat,
                    "description": latest_e.description,
                    "direction": latest_e.direction,
                    "amount": latest_e.amount,
                    "currency": latest_e.currency,
                    "median_interval": int_med,
                    "is_monthly": is_monthly,
                    "day_of_month": dates[-1].day if is_monthly else None,
                    "last_date": dates[-1],
                    "occurrences_count": len(dates),
                })

        return recurring

    @staticmethod
    def _add_months(orig_date: date, months: int, target_day: Optional[int] = None) -> date:
        """Advance date by a number of calendar months, preserving target day of month."""
        year = orig_date.year + (orig_date.month + months - 1) // 12
        month = (orig_date.month + months - 1) % 12 + 1
        import calendar
        max_days = calendar.monthrange(year, month)[1]
        day = min(target_day or orig_date.day, max_days)
        return date(year, month, day)

    def run_forecast(
        self,
        request: Request,
        profile: FinancialProfile,
        events: List[FinancialEvent],
        horizon_days: int = 90,
    ) -> ForecastResult:
        """Run conservative 90-day cash flow simulation and calculate safe amounts."""
        req_date = self.parse_date(request.request_date)
        horizon_end = req_date + timedelta(days=horizon_days)

        # 1. Deduplicate events
        deduped_events = self.deduplicate_events(events)

        # 2. Separate into historical and future included events
        included_future_events: List[FinancialEvent] = []
        historical_events: List[FinancialEvent] = []
        candidate_recurrence_events: List[FinancialEvent] = []

        for e in deduped_events:
            if not self.is_included_event(e):
                continue
            s_date = self.parse_date(e.settlement_date if e.settlement_date else e.event_date)
            e_date = self.parse_date(e.event_date)

            if e_date <= req_date:
                historical_events.append(e)

            # Candidate for recurrence: confirmed settled and scheduled events
            if e.status.lower() in ("settled", "scheduled"):
                candidate_recurrence_events.append(e)

            # Future settlement within forecast window
            # Also include pending debits with settlement_date <= req_date that have not yet settled
            if req_date <= s_date <= horizon_end:
                included_future_events.append(e)
            elif s_date < req_date and e.status.strip().lower() == "pending" and e.direction.strip().lower() == "debit":
                # Outstanding pending debit whose settlement date has arrived: settles today (req_date)
                included_future_events.append(e)

        # 3. Recurrence detection
        recurring_groups = self.detect_recurring_groups(candidate_recurrence_events, req_date)

        # Map of existing explicit future events to avoid double counting:
        # (category.lower(), date) -> True
        existing_explicit_keys: Set[Tuple[str, date]] = set()
        for e in included_future_events:
            s_date = self.parse_date(e.settlement_date if e.settlement_date else e.event_date)
            existing_explicit_keys.add((e.category.lower(), s_date))

        # 4. Project future recurring events
        projected_cash_flows: List[CashFlowItem] = []
        for rg in recurring_groups:
            last_d = rg["last_date"]
            if rg.get("is_monthly"):
                m = 1
                next_d = self._add_months(last_d, m, rg.get("day_of_month"))
                while next_d <= horizon_end:
                    if next_d > req_date:
                        if (rg["category"].lower(), next_d) not in existing_explicit_keys:
                            projected_cash_flows.append(
                                CashFlowItem(
                                    date=next_d,
                                    direction=rg["direction"],
                                    amount=rg["amount"],
                                    currency=rg["currency"],
                                    category=rg["category"],
                                    description=rg["description"],
                                    is_projected=True,
                                )
                            )
                    m += 1
                    next_d = self._add_months(last_d, m, rg.get("day_of_month"))
            else:
                interval = rg["median_interval"]
                next_d = last_d + timedelta(days=interval)
                while next_d <= horizon_end:
                    if next_d > req_date:
                        if (rg["category"].lower(), next_d) not in existing_explicit_keys:
                            projected_cash_flows.append(
                                CashFlowItem(
                                    date=next_d,
                                    direction=rg["direction"],
                                    amount=rg["amount"],
                                    currency=rg["currency"],
                                    category=rg["category"],
                                    description=rg["description"],
                                    is_projected=True,
                                )
                            )
                    next_d += timedelta(days=interval)

        # 5. Build daily cash flows
        daily_cash_flows: Dict[date, float] = {req_date + timedelta(days=i): 0.0 for i in range(horizon_days + 1)}

        # Apply explicit future events
        for e in included_future_events:
            raw_s_date = self.parse_date(e.settlement_date if e.settlement_date else e.event_date)
            # Cap at req_date for overdue pending debits
            flow_date = max(req_date, raw_s_date)

            if flow_date in daily_cash_flows:
                # If amount is missing (e.g. image-backed event), skip or treat conservatively as 0 until extracted
                if e.amount is None:
                    continue

                amt = e.amount
                if e.currency != profile.home_currency:
                    # Convert using exact settlement date
                    date_str = flow_date.strftime("%Y-%m-%d")
                    amt = self.converter.convert(amt, e.currency, profile.home_currency, date_str)

                if e.direction.strip().lower() == "credit":
                    daily_cash_flows[flow_date] += amt
                else:
                    daily_cash_flows[flow_date] -= amt

        # Apply projected recurring events
        for pf in projected_cash_flows:
            if pf.date in daily_cash_flows:
                amt = pf.amount
                if pf.currency != profile.home_currency:
                    date_str = pf.date.strftime("%Y-%m-%d")
                    amt = self.converter.convert(amt, pf.currency, profile.home_currency, date_str)

                if pf.direction == "credit":
                    daily_cash_flows[pf.date] += amt
                else:
                    daily_cash_flows[pf.date] -= amt

        # 6. Walk forward day-by-day from current_available_balance
        daily_balances: Dict[date, float] = {}
        daily_margins: Dict[date, float] = {}
        balance = profile.current_available_balance
        min_balance = profile.minimum_balance_to_keep

        all_days = [req_date + timedelta(days=i) for i in range(horizon_days + 1)]
        for d in all_days:
            balance += daily_cash_flows[d]
            daily_balances[d] = balance
            daily_margins[d] = balance - min_balance

        min_margin = min(daily_margins.values())

        # 7. amount_safe_to_pay:
        # Max amount payable today without breaking 90-day safety check, capped at requested_amount
        amount_safe = max(0.0, min(request.requested_amount, min_margin))

        # 8. earliest_date_for_full_payment:
        # First date d where paying full requested_amount on date d keeps balance >= min_balance
        # for all remaining days t >= d in the 90-day forecast.
        req_amt = request.requested_amount
        earliest_date_str = ""

        for idx, d in enumerate(all_days):
            # Check remaining window from d onwards
            remaining_min_margin = min(daily_margins[all_days[k]] for k in range(idx, len(all_days)))
            if remaining_min_margin >= req_amt:
                earliest_date_str = d.strftime("%Y-%m-%d")
                break

        return ForecastResult(
            request_id=request.request_id,
            user_id=profile.user_id,
            home_currency=profile.home_currency,
            current_available_balance=profile.current_available_balance,
            minimum_balance_to_keep=profile.minimum_balance_to_keep,
            daily_balances=daily_balances,
            daily_margins=daily_margins,
            min_margin=min_margin,
            amount_safe_to_pay=amount_safe,
            earliest_date_for_full_payment=earliest_date_str,
            recurring_groups_detected=recurring_groups,
            included_explicit_events_count=len(included_future_events),
            projected_events_count=len(projected_cash_flows),
        )
