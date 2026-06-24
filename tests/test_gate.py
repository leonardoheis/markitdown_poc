from unittest.mock import MagicMock
from quality_gate.gate import QualityGate
from quality_gate.heuristics import HeuristicEvaluator
from quality_gate.llm_judge import LLMJudge


def _make_gate(heuristic_verdict: str, judge_verdict: str = "pass"):
    mock_heuristic = MagicMock(spec=HeuristicEvaluator)
    mock_heuristic.evaluate.return_value = heuristic_verdict
    mock_judge = MagicMock(spec=LLMJudge)
    mock_judge.evaluate.return_value = judge_verdict
    gate = QualityGate(judge=mock_judge)
    gate._heuristic = mock_heuristic
    return gate, mock_heuristic, mock_judge


def test_gate_returns_pass_without_calling_judge_when_heuristic_passes():
    gate, _, mock_judge = _make_gate("pass")
    verdict, reason = gate.check("good text " * 100, 1)
    assert verdict == "pass"
    mock_judge.evaluate.assert_not_called()


def test_gate_returns_fail_without_calling_judge_when_heuristic_fails():
    gate, _, mock_judge = _make_gate("fail")
    verdict, reason = gate.check("bad", 1)
    assert verdict == "fail"
    mock_judge.evaluate.assert_not_called()


def test_gate_calls_judge_and_returns_pass_when_ambiguous():
    gate, _, mock_judge = _make_gate("ambiguous", "pass")
    verdict, reason = gate.check("medium text " * 20, 1)
    assert verdict == "pass"
    mock_judge.evaluate.assert_called_once_with("medium text " * 20, 1)


def test_gate_calls_judge_and_returns_fail_when_ambiguous_and_judge_fails():
    gate, _, mock_judge = _make_gate("ambiguous", "fail")
    verdict, reason = gate.check("medium text " * 20, 1)
    assert verdict == "fail"
    mock_judge.evaluate.assert_called_once()


def test_gate_always_returns_non_empty_reason_string():
    for heuristic_verdict, judge_verdict in [("pass", "pass"), ("fail", "fail"), ("ambiguous", "pass")]:
        gate, _, _ = _make_gate(heuristic_verdict, judge_verdict)
        _, reason = gate.check("text " * 50, 1)
        assert isinstance(reason, str)
        assert len(reason) > 0
