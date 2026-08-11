"""Tools should work without any LLM or network involved."""
from assistant.tools.builtin import calculator, current_datetime


def test_calculator_basic():
    assert calculator("2 + 2") == "4"
    assert calculator("(3 + 4) * 2") == "14"


def test_calculator_rejects_unsafe_input():
    # no names/attrs/calls allowed - only arithmetic
    result = calculator("__import__('os').system('echo hi')")
    assert result.startswith("Error")


def test_current_datetime_returns_string():
    assert isinstance(current_datetime(), str)
    assert len(current_datetime()) > 0
