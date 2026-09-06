"""Adapt Reasoning Gym puzzles to the example's strict answer contract."""

import ast
import json
import re
from collections import Counter
from fractions import Fraction

from reasoning_gym.algorithmic.graph_color import verify_graph_coloring_solution
from reasoning_gym.utils import extract_answer as extract_gym_answer

SYSTEM_PROMPT = (
    "Solve the puzzle. Output only one <answer>...</answer> block. " "Do not include reasoning or explanation."
)


def extract_answer(response: str) -> str | None:
    if not isinstance(response, str) or len(response) > 20000:
        return None
    # Qwen can retain its terminal token in decoded text.
    response = response.strip().removesuffix("<|im_end|>").rstrip()
    if "<answer>" not in response and "</answer>" not in response:
        return response.strip()
    if response.count("<answer>") != 1 or response.count("</answer>") != 1:
        return None
    return extract_gym_answer(response)


def check_countdown(answer: str, numbers: list[int], target: int) -> bool:
    """Restrict operators and use exact arithmetic, unlike the upstream SymPy scorer."""
    if len(answer) > 256 or not re.fullmatch(r"[\d\s()+*/-]+", answer):
        return False
    used = []

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) is int and 0 < node.value <= 10000:
            used.append(node.value)
            return Fraction(node.value)
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            raise ValueError("Only integer literals and +, -, *, / are allowed")
        left, right = evaluate(node.left), evaluate(node.right)
        if isinstance(node.op, ast.Add):
            value = left + right
        elif isinstance(node.op, ast.Sub):
            value = left - right
        elif isinstance(node.op, ast.Mult):
            value = left * right
        else:
            value = left / right
        if abs(value.numerator) > 10**12 or value.denominator > 10**12:
            raise ValueError("Intermediate value too large")
        return value

    try:
        tree = ast.parse(answer, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 64:
            return False
        value = evaluate(tree.body)
        return value == target and Counter(used) == Counter(numbers)
    except (SyntaxError, ValueError, ZeroDivisionError, RecursionError):
        return False


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def check_graph_color(answer: str, puzzle: dict) -> bool:
    if len(answer) > 4096:
        return False
    try:
        colors = json.loads(answer, object_pairs_hook=_unique_object)
        if not isinstance(colors, dict) or set(colors) != {str(v) for v in puzzle["vertices"]}:
            return False
        if any(type(c) is not int for c in colors.values()):
            return False
        return verify_graph_coloring_solution(puzzle, colors)[0]
    except (ValueError, TypeError, RecursionError):
        return False


def score(response: str, label: dict | str) -> float:
    if isinstance(label, str):
        label = json.loads(label)
    answer = extract_answer(response)
    if answer is None:
        return 0.0
    if label["domain"] == "countdown":
        return float(check_countdown(answer, label["numbers"], label["target"]))
    if label["domain"] == "graph_color":
        return float(check_graph_color(answer, label["puzzle"]))
    raise ValueError(f"Unknown puzzle domain {label['domain']!r}")


async def reward_func(args, sample, **kwargs):
    return score(sample.response, sample.label)
