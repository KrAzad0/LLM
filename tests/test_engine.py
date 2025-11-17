"""Tests for the SmartCalc engine."""

from smartcalc.engine import CalculationError, SmartCalculator


def test_basic_arithmetic():
    calc = SmartCalculator()
    assert calc.evaluate("2 + 2 * 5") == 12


def test_trigonometry_in_degrees():
    calc = SmartCalculator(angle_mode="degrees")
    assert round(calc.evaluate("sin(90)"), 6) == 1
    calc.set_angle_mode("radians")
    assert round(calc.evaluate("sin(pi / 2)"), 6) == 1


def test_statistics_functions():
    calc = SmartCalculator()
    assert calc.evaluate("mean(2, 4, 6)") == 4
    assert calc.evaluate("median(2, 4, 6, 8)") == 5


def test_combinatorics():
    calc = SmartCalculator()
    assert calc.evaluate("perm(5, 2)") == 20
    assert calc.evaluate("comb(5, 2)") == 10


def test_quadratic_solver():
    calc = SmartCalculator()
    roots = calc.evaluate("solve_quadratic(1, -3, 2)")
    assert roots == (2.0, 1.0)


def test_unknown_identifier_raises():
    calc = SmartCalculator()
    try:
        calc.evaluate("foo + 2")
    except CalculationError as exc:
        assert "Unknown identifier" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected CalculationError")
