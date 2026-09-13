"""Decision Engine for the Buy or Wait? AI financial decision agent.

Implements eligibility checking, candidate ranking, spending changes synthesis,
and dynamic fact-based explanation generation per AGENTS.md and problem statement rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import List, Optional, Tuple, Dict, Any, Set

from models import (
    AffordabilityStatus,
    FinancialEvent,
    FinancialProfile,
    OutputRecord,
    PaymentOption,
    RecommendedPaymentMethod,
    Request,
)
from forecast import ForecastEngine, ForecastResult
from currency import CurrencyConverter


@dataclass
class CandidatePlan:
    """Represents a candidate financial recommendation plan."""
    affordability_status: AffordabilityStatus
    recommended_payment_method: RecommendedPaymentMethod
    payment_plan: str
    earliest_date_for_full_payment: str
    spending_changes_needed: str
    decision_explanation: str
    # Ranking metrics
    completes_by_desired_date: bool
    requires_no_spending_changes: bool
    total_amount_paid: float
    start_date: str
    number_of_payments: int
    option_id_rank: str


class DecisionEngine:
    """Evaluates request context and forecasts to output optimal decision records."""

    def __init__(self, currency_converter: CurrencyConverter):
        self.converter = currency_converter
        self.forecast_engine = ForecastEngine(currency_converter)

    def evaluate_request(
        self,
        request: Request,
        profile: FinancialProfile,
        user_events: List[FinancialEvent],
        payment_options: List[PaymentOption],
    ) -> OutputRecord:
        """Evaluate request context and return strictly compliant OutputRecord."""
        # 1. Run baseline 90-day forecast without spending changes
        base_forecast = self.forecast_engine.run_forecast(request, profile, user_events)
        
        # 2. Evaluate candidates without spending changes
        candidates = self._generate_candidates(request, profile, base_forecast, payment_options, user_events, spending_changes="none")

        # 3. If no change-free plan is safe, evaluate candidates with spending changes
        if not candidates:
            spending_changes, modified_events = self._propose_spending_changes(request, profile, user_events)
            if spending_changes != "none" and modified_events:
                mod_forecast = self.forecast_engine.run_forecast(request, profile, modified_events)
                candidates = self._generate_candidates(request, profile, mod_forecast, payment_options, modified_events, spending_changes=spending_changes)

        # 4. If still no candidates, fallback to NOT_RECOMMENDED
        if not candidates:
            return self._build_not_recommended_record(request, profile, base_forecast)

        # 5. Rank candidates by tie-breaker rules:
        # (1) completes by desired_completion_date
        # (2) requires no spending changes
        # (3) minimizes total amount paid
        # (4) starts earliest
        # (5) fewest payments
        # (6) lowest payment_option_id
        candidates.sort(key=lambda c: (
            not c.completes_by_desired_date,
            not c.requires_no_spending_changes,
            c.total_amount_paid,
            c.start_date,
            c.number_of_payments,
            c.option_id_rank,
        ))

        chosen = candidates[0]
        return OutputRecord(
            request_id=request.request_id,
            amount_safe_to_pay=base_forecast.amount_safe_to_pay,
            affordability_status=chosen.affordability_status,
            recommended_payment_method=chosen.recommended_payment_method,
            payment_plan=chosen.payment_plan,
            earliest_date_for_full_payment=base_forecast.earliest_date_for_full_payment,
            spending_changes_needed=chosen.spending_changes_needed,
            decision_explanation=chosen.decision_explanation,
        )

    def _generate_candidates(
        self,
        request: Request,
        profile: FinancialProfile,
        forecast: ForecastResult,
        payment_options: List[PaymentOption],
        user_events: List[FinancialEvent],
        spending_changes: str,
    ) -> List[CandidatePlan]:
        """Generate all eligible candidate plans for a forecast result."""
        candidates: List[CandidatePlan] = []
        user_methods = set(profile.payment_methods_list)
        req_amt = request.requested_amount
        safe_amt = forecast.amount_safe_to_pay
        earliest_date = forecast.earliest_date_for_full_payment
        desired_date = request.desired_completion_date
        req_date = request.request_date
        curr = profile.home_currency

        has_changes = (spending_changes != "none")

        # --- Option A: FULL_PAYMENT Today ---
        if "full_payment" in user_methods and safe_amt >= req_amt - 1e-4:
            plan_str = f"{req_date}:{req_amt:.2f}".rstrip("0").rstrip(".")
            status = AffordabilityStatus.AFFORDABLE_NOW if not has_changes else AffordabilityStatus.AFFORDABLE_WITH_PLAN
            exp = (
                f"Full payment of {curr} {req_amt:.2f}".rstrip("0").rstrip(".") + f" today on {req_date}. "
                f"Maintains minimum balance requirement of {curr} {profile.minimum_balance_to_keep:.2f} (forecast safety margin is {curr} {forecast.min_margin:.2f})."
            )
            if has_changes:
                exp = (
                    f"With spending adjustments ({spending_changes}), full payment of {curr} {req_amt:.2f}".rstrip("0").rstrip(".") +
                    f" today is safe while keeping minimum balance {curr} {profile.minimum_balance_to_keep:.2f} protected."
                )

            candidates.append(
                CandidatePlan(
                    affordability_status=status,
                    recommended_payment_method=RecommendedPaymentMethod.FULL_PAYMENT,
                    payment_plan=plan_str,
                    earliest_date_for_full_payment=req_date,
                    spending_changes_needed=spending_changes,
                    decision_explanation=exp,
                    completes_by_desired_date=(req_date <= desired_date),
                    requires_no_spending_changes=not has_changes,
                    total_amount_paid=req_amt,
                    start_date=req_date,
                    number_of_payments=1,
                    option_id_rank="",
                )
            )

        # --- Option B: INSTALLMENTS ---
        if "installments" in user_methods and payment_options:
            for opt in payment_options:
                if opt.request_id != request.request_id:
                    continue
                if profile.max_installment_months and opt.number_of_payments > profile.max_installment_months:
                    continue

                # Build schedule
                first_d = datetime.strptime(opt.first_payment_date.strip(), "%Y-%m-%d").date()
                freq_days = opt.payment_frequency_days or 30
                schedule = []
                for i in range(opt.number_of_payments):
                    p_date = (first_d + timedelta(days=i * freq_days)).strftime("%Y-%m-%d")
                    schedule.append((p_date, opt.payment_amount))

                if not schedule:
                    continue

                # Strict safety check: simulate cumulative installment payments against daily forecast margins
                # Must never breach minimum_balance_to_keep on any day t in forecast window
                schedule_parsed = [(datetime.strptime(d_str, "%Y-%m-%d").date(), amt) for d_str, amt in schedule]
                is_inst_safe = True
                for t_date in sorted(forecast.daily_margins.keys()):
                    cum_paid = sum(a for d_date, a in schedule_parsed if d_date <= t_date)
                    if forecast.daily_margins[t_date] - cum_paid < -1e-4:
                        is_inst_safe = False
                        break

                if not is_inst_safe:
                    continue  # Discard unsafe installment option

                final_date = schedule[-1][0]
                monthly_amt = opt.payment_amount
                tot_paid = opt.total_payable_amount or (monthly_amt * opt.number_of_payments)

                # Format plan strictly YYYY-MM-DD:amount|YYYY-MM-DD:amount
                plan_parts = [f"{d}:{a:.2f}".rstrip("0").rstrip(".") for d, a in schedule]
                plan_str = "|".join(plan_parts)

                exp = (
                    f"Payment option {opt.payment_option_id}: {opt.number_of_payments} installments of {curr} {monthly_amt:.2f}".rstrip("0").rstrip(".") +
                    f" starting {schedule[0][0]} totaling {curr} {tot_paid:.2f}".rstrip("0").rstrip(".") +
                    f". Maintains balance above minimum requirement {curr} {profile.minimum_balance_to_keep:.2f}."
                )

                candidates.append(
                    CandidatePlan(
                        affordability_status=AffordabilityStatus.AFFORDABLE_WITH_PLAN,
                        recommended_payment_method=RecommendedPaymentMethod.INSTALLMENTS,
                        payment_plan=plan_str,
                        earliest_date_for_full_payment=earliest_date or schedule[0][0],
                        spending_changes_needed=spending_changes,
                        decision_explanation=exp,
                        completes_by_desired_date=(final_date <= desired_date),
                        requires_no_spending_changes=not has_changes,
                        total_amount_paid=tot_paid,
                        start_date=schedule[0][0],
                        number_of_payments=opt.number_of_payments,
                        option_id_rank=opt.payment_option_id,
                    )
                )

        # --- Option C: PARTIAL_PAYMENT ---
        if "partial_payment" in user_methods and request.allows_partial_payment:
            if 0 < safe_amt < req_amt and earliest_date and earliest_date <= desired_date:
                rem_amt = req_amt - safe_amt
                plan_str = (
                    f"{req_date}:{safe_amt:.2f}".rstrip("0").rstrip(".") +
                    f"|{earliest_date}:{rem_amt:.2f}".rstrip("0").rstrip(".")
                )
                exp = (
                    f"Partial payment plan: Pay safe amount {curr} {safe_amt:.2f}".rstrip("0").rstrip(".") +
                    f" today on {req_date} and remaining {curr} {rem_amt:.2f}".rstrip("0").rstrip(".") +
                    f" on {earliest_date}. Completes full requested {curr} {req_amt:.2f}".rstrip("0").rstrip(".") +
                    f" by desired completion date {desired_date} while protecting {curr} {profile.minimum_balance_to_keep:.2f} minimum."
                )
                candidates.append(
                    CandidatePlan(
                        affordability_status=AffordabilityStatus.AFFORDABLE_WITH_PLAN,
                        recommended_payment_method=RecommendedPaymentMethod.PARTIAL_PAYMENT,
                        payment_plan=plan_str,
                        earliest_date_for_full_payment=earliest_date,
                        spending_changes_needed=spending_changes,
                        decision_explanation=exp,
                        completes_by_desired_date=(earliest_date <= desired_date),
                        requires_no_spending_changes=not has_changes,
                        total_amount_paid=req_amt,
                        start_date=req_date,
                        number_of_payments=2,
                        option_id_rank="",
                    )
                )

        # --- Option D: WAIT (AFFORDABLE_LATER) ---
        if ("wait" in user_methods or "full_payment" in user_methods) and earliest_date:
            if earliest_date <= desired_date:
                plan_str = "none"
                exp = (
                    f"Wait until {earliest_date} to pay {curr} {req_amt:.2f}".rstrip("0").rstrip(".") +
                    f" in full. Paying before {earliest_date} would breach the {curr} {profile.minimum_balance_to_keep:.2f} minimum balance requirement."
                )
                candidates.append(
                    CandidatePlan(
                        affordability_status=AffordabilityStatus.AFFORDABLE_LATER,
                        recommended_payment_method=RecommendedPaymentMethod.WAIT,
                        payment_plan=plan_str,
                        earliest_date_for_full_payment=earliest_date,
                        spending_changes_needed=spending_changes,
                        decision_explanation=exp,
                        completes_by_desired_date=(earliest_date <= desired_date),
                        requires_no_spending_changes=not has_changes,
                        total_amount_paid=req_amt,
                        start_date=earliest_date,
                        number_of_payments=1,
                        option_id_rank="",
                    )
                )

        return candidates

    def _propose_spending_changes(
        self,
        request: Request,
        profile: FinancialProfile,
        user_events: List[FinancialEvent],
    ) -> Tuple[str, Optional[List[FinancialEvent]]]:
        """Synthesize up to 3 valid spending changes from user's reduce/stop preferences."""
        stop_cats = set(c.lower() for c in profile.stop_categories_list)
        reduce_cats = set(c.lower() for c in profile.reduce_categories_list)

        changes: List[str] = []
        modified_events = list(user_events)
        mod_map = {e.event_id: e for e in modified_events}

        for ev in user_events:
            if len(changes) >= 3:
                break
            cat = ev.category.lower()
            flex = ev.flexibility.lower()

            if cat in stop_cats and flex in ("stoppable", "reducible_or_stoppable", "flexible"):
                changes.append(f"stop:{ev.event_id}")
                mod_map[ev.event_id] = FinancialEvent(
                    event_id=ev.event_id,
                    user_id=ev.user_id,
                    event_type=ev.event_type,
                    description=ev.description,
                    category=ev.category,
                    direction=ev.direction,
                    amount=0.0,
                    currency=ev.currency,
                    event_date=ev.event_date,
                    settlement_date=ev.settlement_date,
                    status="cancelled",
                    linked_event_id=ev.linked_event_id,
                    flexibility=ev.flexibility,
                    minimum_allowed_amount=ev.minimum_allowed_amount,
                )
            elif cat in reduce_cats and flex in ("reducible", "reducible_or_stoppable", "flexible") and ev.minimum_allowed_amount is not None:
                min_amt = ev.minimum_allowed_amount
                min_amt_str = f"{min_amt:.2f}".rstrip("0").rstrip(".")
                changes.append(f"reduce_to:{ev.event_id}:{min_amt_str}")
                mod_map[ev.event_id] = FinancialEvent(
                    event_id=ev.event_id,
                    user_id=ev.user_id,
                    event_type=ev.event_type,
                    description=ev.description,
                    category=ev.category,
                    direction=ev.direction,
                    amount=min_amt,
                    currency=ev.currency,
                    event_date=ev.event_date,
                    settlement_date=ev.settlement_date,
                    status=ev.status,
                    linked_event_id=ev.linked_event_id,
                    flexibility=ev.flexibility,
                    minimum_allowed_amount=ev.minimum_allowed_amount,
                )

        if not changes:
            return "none", None

        return "|".join(changes), list(mod_map.values())

    def _build_not_recommended_record(
        self,
        request: Request,
        profile: FinancialProfile,
        forecast: ForecastResult,
    ) -> OutputRecord:
        """Build OutputRecord for NOT_RECOMMENDED recommendation."""
        req_amt = request.requested_amount
        safe_amt = forecast.amount_safe_to_pay
        curr = profile.home_currency

        if safe_amt > 0:
            exp = (
                f"Do not proceed with the {curr} {req_amt:.2f}".rstrip("0").rstrip(".") + f" request by desired completion date {request.desired_completion_date}. "
                f"Although safe amount today is {curr} {safe_amt:.2f}".rstrip("0").rstrip(".") + f", full payment cannot be safely completed without breaching the minimum balance requirement of {curr} {profile.minimum_balance_to_keep:.2f}."
            )
        else:
            exp = (
                f"Do not make this payment by desired completion date {request.desired_completion_date}. "
                f"None of the available payment options keeps the {curr} {profile.minimum_balance_to_keep:.2f} minimum balance protected."
            )

        return OutputRecord(
            request_id=request.request_id,
            amount_safe_to_pay=safe_amt,
            affordability_status=AffordabilityStatus.NOT_AFFORDABLE,
            recommended_payment_method=RecommendedPaymentMethod.NOT_RECOMMENDED,
            payment_plan="none",
            earliest_date_for_full_payment="",
            spending_changes_needed="none",
            decision_explanation=exp,
        )
