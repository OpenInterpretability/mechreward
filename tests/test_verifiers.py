"""Tests for outcome verifiers."""
from mechreward.verifiers import gsm8k_verifier, math_boxed, python_syntax_ok


def test_gsm8k_verifier_correct():
    prompt = "Janet has 3 apples and eats 1. How many are left?\n#### 2"
    completion = "She has 3 - 1 = 2 apples left. The answer is 2."
    assert gsm8k_verifier(prompt, completion)


def test_gsm8k_verifier_wrong():
    prompt = "Janet has 3 apples and eats 1. How many are left?\n#### 2"
    completion = "She has 5 apples."
    assert not gsm8k_verifier(prompt, completion)


def test_gsm8k_verifier_no_gold_returns_false():
    prompt = "No gold here."
    completion = "Answer: 42"
    assert not gsm8k_verifier(prompt, completion)


def test_math_boxed_correct():
    prompt = r"Compute: 3 + 4. \boxed{7}"
    completion = r"The answer is \boxed{7}."
    assert math_boxed(prompt, completion)


def test_math_boxed_wrong():
    prompt = r"\boxed{10}"
    completion = r"\boxed{11}"
    assert not math_boxed(prompt, completion)


def test_python_syntax_ok():
    assert python_syntax_ok("", "def f(x): return x + 1")
    assert not python_syntax_ok("", "def f(x: return x +")
