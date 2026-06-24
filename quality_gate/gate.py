from typing import Literal
from .heuristics import HeuristicEvaluator
from .llm_judge import LLMJudge


class QualityGate:
    def __init__(self, judge: LLMJudge | None = None):
        self._heuristic = HeuristicEvaluator()
        self._judge = judge if judge is not None else LLMJudge()

    def check(self, text: str, page_count: int) -> tuple[Literal["pass", "fail"], str]:
        verdict = self._heuristic.evaluate(text, page_count)

        if verdict == "pass":
            return "pass", "heuristic: density and patterns OK"
        if verdict == "fail":
            return "fail", "heuristic: density too low or visual description pattern detected"

        judge_verdict = self._judge.evaluate(text, page_count)
        return judge_verdict, f"heuristic: ambiguous density; llm judge: {judge_verdict}"
