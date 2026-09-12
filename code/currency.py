"""Currency conversion utility for the Buy or Wait? financial decision agent.

Handles exact-date exchange rate lookup, rate inversion (1/rate), and multi-hop chaining
through intermediate currencies (e.g., ZAR -> EUR -> USD -> IDR).

STRICT RULE: Match rates by exact rate_date.
If no rate exists for a needed date, raise MissingExchangeRateError.
DO NOT INVENT, GUESS, OR INTERPOLATE RATES.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

from models import ExchangeRate


class MissingExchangeRateError(Exception):
    """Raised when an exchange rate cannot be found for an exact date or currency pair."""
    pass


class CurrencyConverter:
    """Graph-based currency converter operating on exact rate dates.
    
    Supported dataset base pairs in exchange_rates.csv:
      - EUR -> USD
      - EUR -> ZAR
      - USD -> EUR
      - USD -> IDR
      - USD -> INR

    Features:
      1. Direct lookup if (from_curr -> to_curr) is present for rate_date.
      2. Inversion (1 / rate) if (to_curr -> from_curr) is present for rate_date.
      3. Chaining (BFS pathfinding) when neither direct nor direct-inverse exists
         (e.g., ZAR -> EUR -> USD -> IDR).
      4. Identity: converting a currency to itself returns the original amount (rate 1.0).
      5. Strictness: if rate_date has no records, or no connected path exists,
         raises MissingExchangeRateError.
    """

    def __init__(self, rates: Optional[List[ExchangeRate]] = None):
        # graph[rate_date][from_curr][to_curr] = multiplier
        self._graph: Dict[str, Dict[str, Dict[str, float]]] = {}
        if rates:
            self.load_rates(rates)

    def load_rates(self, rates: List[ExchangeRate]) -> None:
        """Populate converter graph with dated exchange rates and their inverses."""
        for r in rates:
            if r.rate <= 0:
                raise ValueError(f"Invalid non-positive exchange rate: {r}")
            
            date_graph = self._graph.setdefault(r.rate_date, {})
            
            # Direct edge
            from_map = date_graph.setdefault(r.from_currency, {})
            from_map[r.to_currency] = r.rate

            # Inverse edge (1 / rate), only if direct edge doesn't overwrite an explicit rate
            to_map = date_graph.setdefault(r.to_currency, {})
            if r.from_currency not in to_map:
                to_map[r.from_currency] = 1.0 / r.rate

    @property
    def available_dates(self) -> List[str]:
        """List of all dates for which exchange rates are recorded."""
        return sorted(self._graph.keys())

    def get_rate(self, from_curr: str, to_curr: str, rate_date: str) -> float:
        """Find the effective conversion multiplier from from_curr to to_curr on exact rate_date.
        
        Raises:
            MissingExchangeRateError: if rate_date is unknown or no conversion path exists.
        """
        from_curr = from_curr.strip().upper()
        to_curr = to_curr.strip().upper()

        if from_curr == to_curr:
            return 1.0

        if rate_date not in self._graph:
            raise MissingExchangeRateError(
                f"No exchange rate records exist for exact date '{rate_date}'. "
                f"Available dates count: {len(self._graph)}. Rates cannot be invented."
            )

        date_graph = self._graph[rate_date]

        # 1. Check direct edge
        if from_curr in date_graph and to_curr in date_graph[from_curr]:
            return date_graph[from_curr][to_curr]

        # 2. Path search (BFS) for chaining (e.g. ZAR -> EUR -> USD -> IDR)
        queue: deque[Tuple[str, float, List[str]]] = deque([(from_curr, 1.0, [from_curr])])
        visited = {from_curr}

        while queue:
            curr, current_rate, path = queue.popleft()
            neighbors = date_graph.get(curr, {})
            
            for next_curr, step_rate in neighbors.items():
                if next_curr == to_curr:
                    final_rate = current_rate * step_rate
                    return final_rate
                
                if next_curr not in visited:
                    visited.add(next_curr)
                    queue.append((next_curr, current_rate * step_rate, path + [next_curr]))

        raise MissingExchangeRateError(
            f"No exchange rate path exists from '{from_curr}' to '{to_curr}' on date '{rate_date}'. "
            f"Known currencies for this date: {list(date_graph.keys())}. Rates cannot be invented."
        )

    def convert(self, amount: float, from_curr: str, to_curr: str, rate_date: str) -> float:
        """Convert amount from from_curr to to_curr using rate on exact rate_date."""
        rate = self.get_rate(from_curr, to_curr, rate_date)
        return amount * rate
