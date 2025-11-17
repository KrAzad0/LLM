"""Core evaluation engine powering SmartCalc."""

from __future__ import annotations

from dataclasses import dataclass
import ast
import math
import statistics
from typing import Any, Callable, Dict, Iterable


class CalculationError(ValueError):
    """Raised when an expression cannot be evaluated."""


def _trig_wrapper(
    func: Callable[[float], float], *, mode_getter: Callable[[], str]
) -> Callable[[float], float]:
    def wrapped(value: float) -> float:
        if mode_getter() == "degrees":
            value = math.radians(value)
        return func(value)

    return wrapped


def _inverse_trig_wrapper(
    func: Callable[[float], float], *, mode_getter: Callable[[], str]
) -> Callable[[float], float]:
    def wrapped(value: float) -> float:
        result = func(value)
        if mode_getter() == "degrees":
            result = math.degrees(result)
        return result

    return wrapped


def _default_statistics_handler(name: str, values: Iterable[float]) -> float:
    data = list(values)
    if not data:
        raise CalculationError(f"{name} requires at least one value")
    return getattr(statistics, name)(data)


@dataclass
class EvaluationContext:
    angle_mode: str = "radians"


class SmartCalculator:
    """Safely evaluate mathematical expressions."""

    _allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Num,
        ast.Constant,
        ast.Call,
        ast.Load,
        ast.Name,
        ast.keyword,
        ast.Tuple,
        ast.List,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.FloorDiv,
        ast.USub,
        ast.UAdd,
    )

    _allowed_operators = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.FloorDiv,
    )

    _allowed_unary = (ast.UAdd, ast.USub)

    def __init__(self, *, angle_mode: str = "radians") -> None:
        if angle_mode not in {"radians", "degrees"}:
            raise ValueError("angle_mode must be either 'radians' or 'degrees'")
        self._ctx = EvaluationContext(angle_mode=angle_mode)
        self._base_env = self._build_environment()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def angle_mode(self) -> str:
        return self._ctx.angle_mode

    def set_angle_mode(self, mode: str) -> None:
        if mode not in {"radians", "degrees"}:
            raise ValueError("mode must be 'radians' or 'degrees'")
        self._ctx.angle_mode = mode

    def evaluate(self, expression: str, /, **variables: float) -> float:
        if not expression or not expression.strip():
            raise CalculationError("Empty expressions cannot be evaluated")

        env = dict(self._base_env)
        env.update({k: float(v) for k, v in variables.items()})

        try:
            parsed = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise CalculationError(str(exc)) from exc

        self._validate_ast(parsed)
        return self._eval_node(parsed.body, env)

    # ------------------------------------------------------------------
    # AST helpers
    # ------------------------------------------------------------------
    def _validate_ast(self, node: ast.AST) -> None:
        for child in ast.walk(node):
            if not isinstance(child, self._allowed_nodes):
                raise CalculationError(f"Unsupported syntax: {ast.dump(child, include_attributes=False)}")
            if isinstance(child, ast.BinOp) and not isinstance(child.op, self._allowed_operators):
                raise CalculationError(f"Operator {type(child.op).__name__} is not allowed")
            if isinstance(child, ast.UnaryOp) and not isinstance(child.op, self._allowed_unary):
                raise CalculationError(f"Unary operator {type(child.op).__name__} is not allowed")

    def _eval_node(self, node: ast.AST, env: Dict[str, Any]) -> Any:
        if isinstance(node, ast.Expression):
            return self._eval_node(node.body, env)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise CalculationError(f"Constants of type {type(node.value).__name__} are not supported")
        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, env)
            right = self._eval_node(node.right, env)
            return self._apply_operator(node.op, left, right)
        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand, env)
            return operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise CalculationError(f"Unknown identifier '{node.id}'")
            value = env[node.id]
            if callable(value):
                return value
            return value
        if isinstance(node, ast.Call):
            func = self._eval_node(node.func, env)
            if not callable(func):
                raise CalculationError("Attempted to call a non-callable value")
            args = [self._eval_node(arg, env) for arg in node.args]
            kwargs = {kw.arg: self._eval_node(kw.value, env) for kw in node.keywords}
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - surfaced as CalculationError
                raise CalculationError(str(exc)) from exc
        if isinstance(node, ast.Tuple):
            return tuple(self._eval_node(value, env) for value in node.elts)
        if isinstance(node, ast.List):
            return [self._eval_node(value, env) for value in node.elts]

        raise CalculationError(f"Unsupported expression: {ast.dump(node, include_attributes=False)}")

    def _apply_operator(self, operator: ast.AST, left: float, right: float) -> float:
        if isinstance(operator, ast.Add):
            return left + right
        if isinstance(operator, ast.Sub):
            return left - right
        if isinstance(operator, ast.Mult):
            return left * right
        if isinstance(operator, ast.Div):
            return left / right
        if isinstance(operator, ast.FloorDiv):
            return left // right
        if isinstance(operator, ast.Mod):
            return left % right
        if isinstance(operator, ast.Pow):
            return left**right
        raise CalculationError(f"Operator {type(operator).__name__} is not supported")

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    def _build_environment(self) -> Dict[str, Any]:
        def ctx_mode() -> str:
            return self._ctx.angle_mode

        env: Dict[str, Any] = {
            "pi": math.pi,
            "tau": math.tau,
            "e": math.e,
            "phi": (1 + math.sqrt(5)) / 2,
            "deg2rad": math.radians,
            "rad2deg": math.degrees,
            "abs": abs,
            "round": round,
            "sqrt": math.sqrt,
            "cbrt": lambda x: math.copysign(abs(x) ** (1 / 3), x),
            "log": lambda x, base=math.e: math.log(x, base),
            "ln": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "pow": math.pow,
            "fact": lambda x: math.factorial(int(x)),
            "gamma": math.gamma,
            "perm": math.perm,
            "comb": math.comb,
            "mean": lambda *values: _default_statistics_handler("mean", values),
            "median": lambda *values: _default_statistics_handler("median", values),
            "variance": lambda *values: _default_statistics_handler("variance", values),
            "stdev": lambda *values: _default_statistics_handler("stdev", values),
            "sum": lambda *values: sum(values),
            "max": lambda *values: max(values),
            "min": lambda *values: min(values),
            "root": lambda n, degree=2: n ** (1 / degree),
            "hypot": math.hypot,
            "deg": math.degrees,
            "rad": math.radians,
            "solve_quadratic": self._solve_quadratic,
        }

        trig_funcs = {
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "sinh": math.sinh,
            "cosh": math.cosh,
            "tanh": math.tanh,
        }
        inverse_trig = {
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
        }

        for name, func in trig_funcs.items():
            env[name] = _trig_wrapper(func, mode_getter=ctx_mode)
        for name, func in inverse_trig.items():
            env[name] = _inverse_trig_wrapper(func, mode_getter=ctx_mode)

        return env

    @staticmethod
    def _solve_quadratic(a: float, b: float, c: float) -> tuple[complex, complex]:
        if a == 0:
            raise CalculationError("Coefficient 'a' must be non-zero for a quadratic equation")
        discriminant = b**2 - 4 * a * c
        sqrt_disc = math.sqrt(discriminant) if discriminant >= 0 else math.sqrt(-discriminant) * 1j
        root1 = (-b + sqrt_disc) / (2 * a)
        root2 = (-b - sqrt_disc) / (2 * a)
        return root1, root2
