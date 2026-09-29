"""Odds conversion, de-vigging, expected value and staking math.

All probabilities are floats in [0, 1]. Odds are American unless a name says decimal.
"""

from __future__ import annotations

from collections.abc import Sequence


def american_to_decimal(american: float) -> float:
    if american == 0:
        raise ValueError("American odds cannot be 0")
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / abs(american)


def decimal_to_american(decimal: float) -> float:
    if decimal <= 1.0:
        raise ValueError("Decimal odds must be greater than 1")
    if decimal >= 2.0:
        return round((decimal - 1.0) * 100.0)
    return round(-100.0 / (decimal - 1.0))


def implied_prob(american: float) -> float:
    """Vig-inclusive probability implied by a single American price."""
    return 1.0 / american_to_decimal(american)


def prob_to_american(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("Probability must be strictly between 0 and 1")
    return decimal_to_american(1.0 / p)


def devig_multiplicative(probs: Sequence[float]) -> list[float]:
    """Normalise implied probabilities so they sum to 1. Simple, slightly biased on longshots."""
    total = sum(probs)
    if total <= 0:
        raise ValueError("Implied probabilities must sum to a positive number")
    return [p / total for p in probs]


def devig_power(probs: Sequence[float], tol: float = 1e-10, max_iter: int = 200) -> list[float]:
    """Power method: find k such that sum(p_i ** k) == 1.

    Better than multiplicative at removing the favourite-longshot bias that books bake
    into their prices. Falls back to multiplicative if the search fails to converge.
    """
    if any(p <= 0 or p >= 1 for p in probs):
        return devig_multiplicative(probs)
    lo, hi = 0.5, 5.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        s = sum(p**mid for p in probs)
        if abs(s - 1.0) < tol:
            break
        # sum decreases as k increases (all p < 1), so a sum above 1 means k is too small
        if s > 1.0:
            lo = mid
        else:
            hi = mid
    k = (lo + hi) / 2.0
    out = [p**k for p in probs]
    total = sum(out)
    return [p / total for p in out]


def overround(probs: Sequence[float]) -> float:
    """Book margin: sum of implied probabilities minus 1 (e.g. 0.045 for a 4.5% hold)."""
    return sum(probs) - 1.0


def expected_value(fair_prob: float, american: float) -> float:
    """EV per 1 unit staked at the given price if fair_prob is the true win probability."""
    dec = american_to_decimal(american)
    return fair_prob * (dec - 1.0) - (1.0 - fair_prob)


def kelly_fraction(fair_prob: float, american: float, fraction: float = 1.0) -> float:
    """Fraction of bankroll to stake. Returns 0 when the bet has no edge."""
    dec = american_to_decimal(american)
    b = dec - 1.0
    q = 1.0 - fair_prob
    full = (b * fair_prob - q) / b
    return max(0.0, full) * fraction


def fair_prob_from_two_way(price_a: float, price_b: float, method: str = "power") -> tuple[float, float]:
    """De-vig a two-outcome market quoted in American odds. Returns (p_a, p_b)."""
    probs = [implied_prob(price_a), implied_prob(price_b)]
    if method == "multiplicative":
        out = devig_multiplicative(probs)
    else:
        out = devig_power(probs)
    return out[0], out[1]
