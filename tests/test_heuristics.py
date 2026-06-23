import pytest
from quality_gate.heuristics import HeuristicEvaluator


@pytest.fixture
def evaluator():
    return HeuristicEvaluator()


def test_density_below_fail_threshold_returns_fail(evaluator):
    text = "a" * 50  # 50 chars / 1 page = 50 < 100
    assert evaluator.evaluate(text, 1) == "fail"


def test_density_above_pass_threshold_returns_pass(evaluator):
    text = "a" * 600  # 600 chars / 1 page = 600 > 500
    assert evaluator.evaluate(text, 1) == "pass"


def test_density_in_ambiguous_range_returns_ambiguous(evaluator):
    text = "a" * 300  # 300 chars / 1 page = 300, between 100 and 500
    assert evaluator.evaluate(text, 1) == "ambiguous"


def test_visual_pattern_overrides_high_density_to_fail(evaluator):
    text = "a" * 600 + " The image shows a document with lots of text."
    assert evaluator.evaluate(text, 1) == "fail"


def test_visual_pattern_detection_is_case_insensitive(evaluator):
    text = "a" * 600 + " THE IMAGE SHOWS a slide."
    assert evaluator.evaluate(text, 1) == "fail"


def test_appears_to_be_pattern_returns_fail(evaluator):
    text = "a" * 600 + " This appears to be a letter."
    assert evaluator.evaluate(text, 1) == "fail"


def test_multipage_density_averages_across_pages(evaluator):
    text = "a" * 200  # 200 chars / 4 pages = 50 < 100 → fail
    assert evaluator.evaluate(text, 4) == "fail"


def test_no_pattern_and_ambiguous_density_returns_ambiguous(evaluator):
    text = "a" * 300  # no patterns, 300 chars/page
    assert evaluator.evaluate(text, 1) == "ambiguous"
