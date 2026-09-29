import math

import pytest

from sportsbet.pricing import (
    american_to_decimal,
    decimal_to_american,
    devig_multiplicative,
    devig_power,
    expected_value,
    fair_prob_from_two_way,
    implied_prob,
    kelly_fraction,
    overround,
)


def test_american_decimal_roundtrip():
    for a in (-110, -250, 100, 150, 400, -105):
        assert decimal_to_american(american_to_decimal(a)) == a


def test_implied_prob_standard_line():
    assert implied_prob(-110) == pytest.approx(0.5238, abs=1e-4)
    assert implied_prob(100) == pytest.approx(0.5)


def test_devig_sums_to_one():
    probs = [implied_prob(-110), implied_prob(-110)]
    assert overround(probs) == pytest.approx(0.0476, abs=1e-4)
    assert sum(devig_multiplicative(probs)) == pytest.approx(1.0)
    assert sum(devig_power(probs)) == pytest.approx(1.0)
    assert devig_power(probs)[0] == pytest.approx(0.5)


def test_power_devig_shrinks_longshot_more_than_multiplicative():
    probs = [implied_prob(-400), implied_prob(300)]
    mult = devig_multiplicative(probs)
    power = devig_power(probs)
    assert sum(power) == pytest.approx(1.0, abs=1e-8)
    assert power[1] < mult[1]  # longshot gets pushed down harder


def test_expected_value_zero_at_fair_price():
    p = 0.6
    fair_price = decimal_to_american(1 / p)
    assert expected_value(p, fair_price) == pytest.approx(0.0, abs=0.01)
    assert expected_value(0.52, -110) < 0
    assert expected_value(0.55, 100) > 0


def test_kelly_zero_without_edge_and_positive_with():
    assert kelly_fraction(0.5, -110) == 0.0
    k = kelly_fraction(0.55, 100, fraction=1.0)
    assert k == pytest.approx(0.10)
    assert kelly_fraction(0.55, 100, fraction=0.25) == pytest.approx(0.025)


def test_fair_prob_two_way():
    a, b = fair_prob_from_two_way(-290, 250)
    assert a + b == pytest.approx(1.0)
    assert 0.70 < a < 0.76
    assert not math.isnan(b)
