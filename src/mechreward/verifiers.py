"""Built-in outcome verifiers for common benchmark formats.

These are pure string-in/bool-out predicates usable with ``OutcomeReward``.
"""
from __future__ import annotations

import ast
import re
import subprocess
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Math verifiers
# ---------------------------------------------------------------------------

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_FINAL_ANSWER_RE = re.compile(r"[Ff]inal\s*[Aa]nswer\s*[:=]\s*([^\n]+)")
_ANSWER_EQUALS_RE = re.compile(r"(?:=|is|equals)\s*([\-+]?\d+(?:\.\d+)?)\s*\.?\s*$")


def _extract_numeric(text: str) -> float | None:
    """Try a sequence of extractors to find a number in ``text``."""
    for regex in (_BOXED_RE, _FINAL_ANSWER_RE, _ANSWER_EQUALS_RE):
        m = regex.search(text)
        if m:
            try:
                return float(m.group(1).strip().rstrip("."))
            except ValueError:
                continue
    # Fallback: last number in the text
    nums = re.findall(r"[\-+]?\d+(?:\.\d+)?", text)
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            return None
    return None


def math_boxed(prompt: str, completion: str, tolerance: float = 1e-4) -> bool:
    """Check if the completion's boxed answer matches the gold answer in the prompt.

    The prompt should contain ``\\boxed{<gold>}`` somewhere (for GSM8K/MATH-style).
    Falls back to numerical comparison.
    """
    gold_match = _BOXED_RE.search(prompt)
    if not gold_match:
        return False
    try:
        gold = float(gold_match.group(1).strip())
    except ValueError:
        return False

    pred = _extract_numeric(completion)
    if pred is None:
        return False
    return abs(pred - gold) < tolerance


def gsm8k_verifier(prompt: str, completion: str) -> bool:
    """GSM8K-specific verifier. The prompt has gold in "#### N" format."""
    m = re.search(r"####\s*([\-+]?\d+(?:\.\d+)?)", prompt)
    if not m:
        return False
    try:
        gold = float(m.group(1))
    except ValueError:
        return False
    pred = _extract_numeric(completion)
    if pred is None:
        return False
    return abs(pred - gold) < 1e-4


# ---------------------------------------------------------------------------
# Code verifiers
# ---------------------------------------------------------------------------


def python_syntax_ok(prompt: str, completion: str) -> bool:
    """Check that the completion is parseable Python."""
    try:
        ast.parse(completion)
        return True
    except SyntaxError:
        return False


def python_exec_ok(
    prompt: str,
    completion: str,
    timeout: int = 5,
) -> bool:
    """Run the completion as Python in a subprocess. True iff exit code 0.

    Dangerous in untrusted settings. Use only with trusted datasets.
    """
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "cand.py"
        script.write_text(completion)
        try:
            result = subprocess.run(
                ["python3", str(script)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False


def humaneval_verifier(prompt: str, completion: str, timeout: int = 10) -> bool:
    """HumanEval-style: run ``prompt + completion + test`` and check exit code."""
    import os

    tests_match = re.search(r"(?:def check|assert ).*", prompt, flags=re.DOTALL)
    if not tests_match:
        return False
    full_code = f"{prompt}\n{completion}\n{tests_match.group(0)}"
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "cand.py"
        script.write_text(full_code)
        try:
            result = subprocess.run(
                ["python3", str(script)],
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
