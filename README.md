# SmartCalc

SmartCalc is a batteries-included scientific calculator implemented in Python. It
offers a safe expression evaluator, rich mathematical context (statistics,
combinatorics, angle conversions, and more), and an interactive REPL-oriented
CLI.

## Highlights

* **Safe evaluation** – expressions are parsed with `ast` and executed only when
  they contain whitelisted nodes, operators, and functions.
* **Angle aware** – toggle between radians and degrees without rewriting your
  formulas.
* **Statistics helpers** – compute averages, medians, variance, and standard
  deviation straight from the REPL.
* **Combinatorics toolkit** – use `perm`, `comb`, `fact`, and `gamma` in
  expressions.
* **Session memory** – reference the `ans` variable to use the result of the last
  calculation.

## Installation

```
pip install .
```

## Command-line usage

```
smartcalc "sin(pi / 4) + mean(1, 2, 3)"
```

Start an interactive session instead:

```
smartcalc
```

Inside the REPL you can use the following meta commands:

* `mode degrees` / `mode radians` – change the active angle unit
* `vars` – inspect the variables currently in scope
* `exit` / `quit` – leave the session

## Library usage

```
from smartcalc.engine import SmartCalculator

calc = SmartCalculator(angle_mode="degrees")
print(calc.evaluate("sin(90) + comb(5, 2)"))
```
