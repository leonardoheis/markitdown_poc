import re
from typing import Literal

CHARS_PER_PAGE_FAIL = 100
CHARS_PER_PAGE_PASS = 500

VISUAL_PATTERNS = [
    r"the image (shows|displays|contains|depicts)",
    r"on the (left|right) side",
    r"appears to be",
    r"the (slide|document|page) (shows|displays|contains)",
    r"bullet points (indicating|showing|listing)",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in VISUAL_PATTERNS]


class HeuristicEvaluator:
    def evaluate(self, text: str, page_count: int) -> Literal["pass", "fail", "ambiguous"]:
        for pattern in _COMPILED:
            if pattern.search(text):
                return "fail"

        density = len(text) / max(page_count, 1)
        if density < CHARS_PER_PAGE_FAIL:
            return "fail"
        if density > CHARS_PER_PAGE_PASS:
            return "pass"
        return "ambiguous"
