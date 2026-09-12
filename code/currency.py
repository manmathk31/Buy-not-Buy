"""Currency conversion utility for the Buy or Wait? financial decision agent.

Handles exact-date exchange rate lookup, canonical USD normalization, and consistent
multi-hop conversion.

STRICT GUARANTEES:
  1. For any date and reachable pair (A, B): convert(A, B, date) * convert(B, A, date) == 1.0.
  2. Triangular consistency: convert(A, B, date) * convert(B, C, date) == convert(A, C, date).
  3. Match rates by exact rate_date.
  4. If no rate exists for a needed date, raise MissingExchangeRateError.
  5. Rates are NEVER invented, guessed, or interpolated.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from models import ExchangeRate


class MissingExchangeRateError(Exception):
    """Raised when an exchange rate cannot be found for an exact date or currency pair."""
    pass


class CurrencyConverter:
    """Canonical graph-based currency converter operating on exact rate dates.
    
    To eliminate directional inconsistencies, each date's reachable currencies are
    normalized into a canonical base value (USD-equivalent value per unit).
    Every pair's rate (in either direction) is derived from this single canonical set:
      rate(A -> B) = canonical_usd_value[A] / canonical_usd_value[B]
    
    This guarantees:
      - convert(A, B) * convert(B, A) == 1.0 exactly (within floating tolerance).
      - No path divergence or accumulated rounding discrepancy.
    """

    def __init__(self, rates: Optional[List[ExchangeRate]] = None):
        # canonical_values[rate_date][currency] = USD price of 1 unit of currency
        self._canonical_values: Dict[str, Dict[str, float]] = {}
        # components[rate_date][currency] = component_id
        self._components: Dict[str, Dict[str, int]] = {}
        if rates:
            self.load_rates(rates)

    def load_rates(self, rates: List[ExchangeRate]) -> None:
        """Build canonical USD normalizations for every unique rate_date."""
        # 1. Group raw rates by date
        rates_by_date: Dict[str, List[ExchangeRate]] = {}
        for r in rates:
            if r.rate <= 0:
                raise ValueError(f"Invalid non-positive exchange rate: {r}")
            rates_by_date.setdefault(r.rate_date, []).append(r)

        # 2. For each date, construct canonical USD normalizations
        for rate_date, date_rates in rates_by_date.items():
            self._build_date_normalization(rate_date, date_rates)

    def _build_date_normalization(self, rate_date: str, date_rates: List[ExchangeRate]) -> None:
        """Compute one canonical normalization per connected component on rate_date."""
        # Graph adjacency: adj[u] = list of (v, is_direct, rate)
        # where is_direct=True means direct edge u -> v with factor `rate` (1 u = rate v)
        # and is_direct=False means direct edge v -> u with factor `rate` (1 v = rate u)
        adj: Dict[str, List[Tuple[str, bool, float]]] = {}
        currencies: Set[str] = set()

        for r in date_rates:
            fc = r.from_currency.strip().upper()
            tc = r.to_currency.strip().upper()
            currencies.add(fc)
            currencies.add(tc)
            adj.setdefault(fc, []).append((tc, True, r.rate))
            adj.setdefault(tc, []).append((fc, False, r.rate))

        canonical_map: Dict[str, float] = {}
        component_map: Dict[str, int] = {}
        visited: Set[str] = set()
        component_id = 0

        # Sort currencies so traversal order is completely deterministic
        all_currencies = sorted(currencies)

        # Process each connected component
        while len(visited) < len(all_currencies):
            # Pick anchor: prefer "USD" if in component, else first unvisited currency
            remaining = [c for c in all_currencies if c not in visited]
            # Check if USD is reachable from any remaining
            anchor = "USD" if ("USD" in remaining) else remaining[0]

            # Run BFS from anchor to assign canonical values
            # canonical_map[C] represents the value of 1 unit of C in Anchor units
            queue: deque[str] = deque([anchor])
            visited.add(anchor)
            canonical_map[anchor] = 1.0
            component_map[anchor] = component_id

            while queue:
                curr = queue.popleft()
                curr_val = canonical_map[curr]

                # Sort neighbors by name for deterministic traversal
                neighbors = sorted(adj.get(curr, []), key=lambda x: x[0])
                for next_curr, is_direct, rate in neighbors:
                    if next_curr not in visited:
                        visited.add(next_curr)
                        component_map[next_curr] = component_id

                        # If is_direct=True: curr -> next_curr with rate (1 curr = rate next_curr)
                        # So 1 next_curr = (1 / rate) curr
                        # value_of(next_curr) = curr_val / rate
                        #
                        # If is_direct=False: next_curr -> curr with rate (1 next_curr = rate curr)
                        # value_of(next_curr) = curr_val * rate
                        if is_direct:
                            canonical_map[next_curr] = curr_val / rate
                        else:
                            canonical_map[next_curr] = curr_val * rate

                        queue.append(next_curr)

            component_id += 1

        self._canonical_values[rate_date] = canonical_map
        self._components[rate_date] = component_map

    @property
    def available_dates(self) -> List[str]:
        """List of all dates for which exchange rates are recorded."""
        return sorted(self._canonical_values.keys())

    def get_rate(self, from_curr: str, to_curr: str, rate_date: str) -> float:
        """Find the exact conversion multiplier from from_curr to to_curr on exact rate_date.
        
        Guarantees:
          get_rate(A, B, d) * get_rate(B, A, d) == 1.0
        
        Raises:
            MissingExchangeRateError: if rate_date is unknown or no conversion path exists.
        """
        from_curr = from_curr.strip().upper()
        to_curr = to_curr.strip().upper()

        if from_curr == to_curr:
            return 1.0

        if rate_date not in self._canonical_values:
            raise MissingExchangeRateError(
                f"No exchange rate records exist for exact date '{rate_date}'. "
                f"Available dates count: {len(self._canonical_values)}. Rates cannot be invented."
            )

        date_values = self._canonical_values[rate_date]
        date_components = self._components[rate_date]

        if from_curr not in date_values:
            raise MissingExchangeRateError(
                f"Currency '{from_curr}' has no exchange rate records on date '{rate_date}'."
            )

        if to_curr not in date_values:
            raise MissingExchangeRateError(
                f"Currency '{to_curr}' has no exchange rate records on date '{rate_date}'."
            )

        if date_components[from_curr] != date_components[to_curr]:
            raise MissingExchangeRateError(
                f"No exchange rate path exists between '{from_curr}' and '{to_curr}' on date '{rate_date}' "
                f"(they are in disconnected components). Rates cannot be invented."
            )

        # 1 unit of from_curr = date_values[from_curr] base units
        # 1 unit of to_curr   = date_values[to_curr] base units
        # So 1 unit of from_curr = (date_values[from_curr] / date_values[to_curr]) units of to_curr
        return date_values[from_curr] / date_values[to_curr]

    def convert(self, amount: float, from_curr: str, to_curr: str, rate_date: str) -> float:
        """Convert amount from from_curr to to_curr using canonical rate on exact rate_date."""
        rate = self.get_rate(from_curr, to_curr, rate_date)
        return amount * rate
