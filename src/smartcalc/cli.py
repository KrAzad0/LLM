"""Command line interface for SmartCalc."""

from __future__ import annotations

import argparse
from typing import Dict

from .engine import CalculationError, SmartCalculator


def _parse_variables(pairs: list[str]) -> Dict[str, float]:
    variables: Dict[str, float] = {}
    for item in pairs:
        if "=" not in item:
            raise CalculationError("Variables must be supplied as key=value pairs")
        key, value = item.split("=", maxsplit=1)
        try:
            variables[key] = float(value)
        except ValueError as exc:
            raise CalculationError(f"Unable to convert {value!r} to a numeric value") from exc
    return variables


def _format_result(value: float, precision: int) -> str:
    return f"{value:.{precision}g}"


def _interactive(calc: SmartCalculator, precision: int) -> None:
    print("SmartCalc interactive mode. Type 'exit' or 'quit' to leave.")
    print("Use 'mode degrees' or 'mode radians' to change angle units. 'vars' shows the scope.")
    env: Dict[str, float] = {}
    history_var = "ans"

    while True:
        try:
            raw = input("smartcalc> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue
        if raw.lower() in {"exit", "quit"}:
            break
        if raw.lower() == "vars":
            snapshot = {**env, history_var: env.get(history_var)} if env else {}
            print(snapshot)
            continue
        if raw.lower().startswith("mode "):
            _, mode = raw.split(maxsplit=1)
            try:
                calc.set_angle_mode(mode.lower())
            except ValueError as exc:
                print(f"Error: {exc}")
            else:
                print(f"Angle mode set to {calc.angle_mode}")
            continue

        try:
            result = calc.evaluate(raw, **env)
        except CalculationError as exc:
            print(f"Error: {exc}")
            continue

        env[history_var] = result
        print(_format_result(result, precision))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smart scientific calculator")
    parser.add_argument("expression", nargs="?", help="Expression to evaluate")
    parser.add_argument(
        "--degrees",
        action="store_true",
        help="Use degrees for trigonometric functions",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=12,
        help="Digits to display in results (default: 12)",
    )
    parser.add_argument(
        "--vars",
        nargs="*",
        default=[],
        metavar="NAME=VALUE",
        help="Variables available to the expression",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    calc = SmartCalculator(angle_mode="degrees" if args.degrees else "radians")

    if args.expression:
        try:
            variables = _parse_variables(args.vars)
            result = calc.evaluate(args.expression, **variables)
        except CalculationError as exc:
            parser.error(str(exc))
        else:
            print(_format_result(result, args.precision))
            return 0

    _interactive(calc, args.precision)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
