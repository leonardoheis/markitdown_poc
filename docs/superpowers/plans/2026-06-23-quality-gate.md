# Quality Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `QualityGate` between MarkItDown and EasyOCR that detects sparse/empty outputs and visual-description failures, falling back to EasyOCR automatically.

**Architecture:** `HeuristicEvaluator` runs fast signal checks first (density + regex patterns); if ambiguous, `LLMJudge` calls `llama3.2` for a verdict; `QualityGate` composes both; `ConversionPipeline` orchestrates the full MarkItDown → gate → EasyOCR flow.

**Tech Stack:** Python 3.10+, markitdown, ollama, pypdfium2, easyocr, openai (Ollama-compatible), pytest

## Global Constraints

- Python >= 3.10 (uses `X | Y` union types and `match` expressions)
- All new code lives under `quality_gate/` package at the project root
- Tests live under `tests/`
- `CHARS_PER_PAGE_FAIL = 100`, `CHARS_PER_PAGE_PASS = 500` (module-level constants, never hardcoded in logic)
- `LLMJudge` defaults to `model="llama3.2"` (already available via Ollama in this project)
- `ConversionPipeline` connects to Ollama at `http://localhost:11434/v1` with `llm_model="llava"`
- `fallback_fn` is the existing `convert_pdf_to_md` callable from the notebook

---

## File Map

| File | Responsibility |
|------|---------------|
| `quality_gate/__init__.py` | Exports `ConversionPipeline` |
| `quality_gate/heuristics.py` | `HeuristicEvaluator` + constants + compiled patterns |
| `quality_gate/llm_judge.py` | `LLMJudge` + prompt template |
| `quality_gate/gate.py` | `QualityGate` composing heuristics + judge |
| `quality_gate/pipeline.py` | `ConversionPipeline` + `_get_page_count` helper |
| `tests/__init__.py` | Empty, marks tests as a package |
| `tests/test_heuristics.py` | Unit tests for `HeuristicEvaluator` |
| `tests/test_llm_judge.py` | Unit tests for `LLMJudge` (mocked ollama) |
| `tests/test_gate.py` | Unit tests for `QualityGate` (mocked heuristic + judge) |
| `tests/test_pipeline.py` | Unit tests for `ConversionPipeline` (mocked MarkItDown + gate) |

---

### Task 1: Project Setup + HeuristicEvaluator

**Files:**
- Create: `quality_gate/__init__.py`
- Create: `quality_gate/heuristics.py`
- Create: `tests/__init__.py`
- Create: `tests/test_heuristics.py`
- Modify: `pyproject.toml` (add pytest dev dependency)

**Interfaces:**
- Produces: `HeuristicEvaluator.evaluate(text: str, page_count: int) -> Literal["pass", "fail", "ambiguous"]`

- [ ] **Step 1: Add pytest to pyproject.toml**

Add after the `dependencies` block:

```toml
[tool.uv.dev-dependencies]
pytest = ">=8.0.0"
```

Run: `uv sync`
Expected: pytest available in `.venv`

- [ ] **Step 2: Create package skeleton**

Create `quality_gate/__init__.py` with empty content:
```python
```

Create `tests/__init__.py` with empty content:
```python
```

- [ ] **Step 3: Write failing tests for HeuristicEvaluator**

Create `tests/test_heuristics.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/test_heuristics.py -v`
Expected: `ModuleNotFoundError: No module named 'quality_gate'`

- [ ] **Step 5: Implement HeuristicEvaluator**

Create `quality_gate/heuristics.py`:

```python
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_heuristics.py -v`
Expected: 8 tests PASSED

- [ ] **Step 7: Commit**

```bash
git add quality_gate/__init__.py quality_gate/heuristics.py tests/__init__.py tests/test_heuristics.py pyproject.toml uv.lock
git commit -m "feat: add HeuristicEvaluator with density and pattern checks"
```

---

### Task 2: LLMJudge

**Files:**
- Create: `quality_gate/llm_judge.py`
- Create: `tests/test_llm_judge.py`

**Interfaces:**
- Consumes: `ollama.chat(model, messages)` from the `ollama` package
- Produces: `LLMJudge(model: str = "llama3.2")`, `LLMJudge.evaluate(text: str, page_count: int) -> Literal["pass", "fail"]`

- [ ] **Step 1: Write failing tests for LLMJudge**

Create `tests/test_llm_judge.py`:

```python
from unittest.mock import MagicMock, patch
from quality_gate.llm_judge import LLMJudge


def test_judge_returns_pass_when_response_contains_pass():
    mock_response = MagicMock()
    mock_response.message.content = "pass"
    with patch("quality_gate.llm_judge.ollama.chat", return_value=mock_response):
        judge = LLMJudge()
        assert judge.evaluate("Full letter content here. " * 40, 2) == "pass"


def test_judge_returns_fail_when_response_contains_fail():
    mock_response = MagicMock()
    mock_response.message.content = "fail"
    with patch("quality_gate.llm_judge.ollama.chat", return_value=mock_response):
        judge = LLMJudge()
        assert judge.evaluate("SAMPLE LETTER", 2) == "fail"


def test_judge_returns_fail_when_response_is_unparseable():
    mock_response = MagicMock()
    mock_response.message.content = "I cannot determine this from the given text."
    with patch("quality_gate.llm_judge.ollama.chat", return_value=mock_response):
        judge = LLMJudge()
        assert judge.evaluate("some text", 1) == "fail"


def test_judge_returns_fail_on_ollama_exception():
    with patch("quality_gate.llm_judge.ollama.chat", side_effect=Exception("connection refused")):
        judge = LLMJudge()
        assert judge.evaluate("some text", 1) == "fail"


def test_judge_truncates_long_text_in_prompt():
    mock_response = MagicMock()
    mock_response.message.content = "pass"
    with patch("quality_gate.llm_judge.ollama.chat", return_value=mock_response) as mock_chat:
        judge = LLMJudge()
        long_text = "a" * 5000
        judge.evaluate(long_text, 1)
        call_content = mock_chat.call_args[1]["messages"][0]["content"]
        assert "a" * 5000 not in call_content  # truncated to 2000 chars
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_llm_judge.py -v`
Expected: `ModuleNotFoundError: No module named 'quality_gate.llm_judge'`

- [ ] **Step 3: Implement LLMJudge**

Create `quality_gate/llm_judge.py`:

```python
from typing import Literal
import ollama

_PROMPT = """\
You are evaluating whether a document text extraction succeeded.

A GOOD extraction contains the actual text of the document: paragraphs, tables, headers, or data.
A BAD extraction either:
- Contains mostly empty content or just a title (e.g. only "SAMPLE LETTER")
- Describes what the document looks like instead of extracting its text

Document page count: {page_count}

Text to evaluate:
---
{text}
---

Answer with exactly one word: "pass" if the extraction is good, "fail" if it is bad.\
"""

_MAX_TEXT_CHARS = 2000


class LLMJudge:
    def __init__(self, model: str = "llama3.2"):
        self._model = model

    def evaluate(self, text: str, page_count: int) -> Literal["pass", "fail"]:
        try:
            response = ollama.chat(
                model=self._model,
                messages=[{
                    "role": "user",
                    "content": _PROMPT.format(
                        page_count=page_count,
                        text=text[:_MAX_TEXT_CHARS],
                    ),
                }],
            )
            content = response.message.content.strip().lower()
            if "pass" in content:
                return "pass"
            return "fail"
        except Exception:
            return "fail"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_llm_judge.py -v`
Expected: 5 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add quality_gate/llm_judge.py tests/test_llm_judge.py
git commit -m "feat: add LLMJudge with conservative fail-safe fallback"
```

---

### Task 3: QualityGate

**Files:**
- Create: `quality_gate/gate.py`
- Create: `tests/test_gate.py`

**Interfaces:**
- Consumes: `HeuristicEvaluator.evaluate(text, page_count) -> Literal["pass", "fail", "ambiguous"]` (from Task 1), `LLMJudge.evaluate(text, page_count) -> Literal["pass", "fail"]` (from Task 2)
- Produces: `QualityGate(judge: LLMJudge | None = None)`, `QualityGate.check(text: str, page_count: int) -> tuple[Literal["pass", "fail"], str]`

- [ ] **Step 1: Write failing tests for QualityGate**

Create `tests/test_gate.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_gate.py -v`
Expected: `ModuleNotFoundError: No module named 'quality_gate.gate'`

- [ ] **Step 3: Implement QualityGate**

Create `quality_gate/gate.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_gate.py -v`
Expected: 5 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add quality_gate/gate.py tests/test_gate.py
git commit -m "feat: add QualityGate composing heuristics and LLM judge"
```

---

### Task 4: ConversionPipeline + notebook integration

**Files:**
- Create: `quality_gate/pipeline.py`
- Create: `tests/test_pipeline.py`
- Modify: `quality_gate/__init__.py`
- Modify: `mdapp.ipynb` (add integration cell)

**Interfaces:**
- Consumes: `QualityGate.check(text, page_count) -> tuple[Literal["pass","fail"], str]` (from Task 3), `fallback_fn: Callable[[str], str]` (the existing `convert_pdf_to_md` from the notebook)
- Produces: `ConversionPipeline(fallback_fn, gate=None, md_client=None)`, `ConversionPipeline.convert(file_path: str) -> str`

- [ ] **Step 1: Write failing tests for ConversionPipeline**

Create `tests/test_pipeline.py`:

```python
from unittest.mock import MagicMock, patch
from quality_gate.pipeline import ConversionPipeline
from quality_gate.gate import QualityGate


def _make_pipeline(gate_verdict: str, md_text: str = "markitdown result", md_raises: bool = False):
    mock_md = MagicMock()
    if md_raises:
        mock_md.convert.side_effect = Exception("conversion error")
    else:
        mock_md.convert.return_value = MagicMock(text_content=md_text)

    mock_gate = MagicMock(spec=QualityGate)
    mock_gate.check.return_value = (gate_verdict, "test reason")

    mock_fallback = MagicMock(return_value="easyocr result")

    pipeline = ConversionPipeline(
        fallback_fn=mock_fallback,
        gate=mock_gate,
        md_client=mock_md,
    )
    return pipeline, mock_md, mock_gate, mock_fallback


def test_pipeline_returns_markitdown_text_when_gate_passes():
    pipeline, _, _, mock_fallback = _make_pipeline("pass", md_text="full letter content")
    with patch("quality_gate.pipeline._get_page_count", return_value=2):
        result = pipeline.convert("data/test.pdf")
    assert result == "full letter content"
    mock_fallback.assert_not_called()


def test_pipeline_calls_fallback_fn_when_gate_fails():
    pipeline, _, _, mock_fallback = _make_pipeline("fail", md_text="SAMPLE LETTER")
    with patch("quality_gate.pipeline._get_page_count", return_value=2):
        result = pipeline.convert("data/test.pdf")
    assert result == "easyocr result"
    mock_fallback.assert_called_once_with("data/test.pdf")


def test_pipeline_falls_back_without_calling_gate_when_markitdown_raises():
    pipeline, _, mock_gate, mock_fallback = _make_pipeline("pass", md_raises=True)
    with patch("quality_gate.pipeline._get_page_count", return_value=1):
        result = pipeline.convert("data/test.pdf")
    assert result == "easyocr result"
    mock_gate.check.assert_not_called()
    mock_fallback.assert_called_once_with("data/test.pdf")


def test_pipeline_passes_page_count_to_gate():
    pipeline, _, mock_gate, _ = _make_pipeline("pass")
    with patch("quality_gate.pipeline._get_page_count", return_value=3):
        pipeline.convert("data/test.pdf")
    _, call_kwargs = mock_gate.check.call_args
    args = mock_gate.check.call_args[0]
    assert args[1] == 3  # page_count is the second positional arg
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: `ModuleNotFoundError: No module named 'quality_gate.pipeline'`

- [ ] **Step 3: Implement ConversionPipeline**

Create `quality_gate/pipeline.py`:

```python
import logging
from typing import Callable, Literal
from markitdown import MarkItDown
from openai import OpenAI
import pypdfium2 as pdfium

from .gate import QualityGate

logger = logging.getLogger(__name__)

_OLLAMA_BASE_URL = "http://localhost:11434/v1"
_VISION_MODEL = "llava"


def _get_page_count(file_path: str) -> int:
    if file_path.lower().endswith(".pdf"):
        pdf = pdfium.PdfDocument(file_path)
        return len(pdf)
    return 1


class ConversionPipeline:
    def __init__(
        self,
        fallback_fn: Callable[[str], str],
        gate: QualityGate | None = None,
        md_client=None,
    ):
        if md_client is None:
            client = OpenAI(base_url=_OLLAMA_BASE_URL, api_key="ollama")
            self._md = MarkItDown(llm_client=client, llm_model=_VISION_MODEL)
        else:
            self._md = md_client
        self._gate = gate if gate is not None else QualityGate()
        self._fallback_fn = fallback_fn

    def convert(self, file_path: str) -> str:
        page_count = _get_page_count(file_path)

        try:
            result = self._md.convert(file_path)
            text = result.text_content
        except Exception as exc:
            logger.warning("MarkItDown failed for %s: %s — falling back to EasyOCR", file_path, exc)
            return self._fallback_fn(file_path)

        verdict, reason = self._gate.check(text, page_count)
        logger.info("QualityGate [%s]: %s", verdict, reason)

        if verdict == "pass":
            return text
        return self._fallback_fn(file_path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: 4 tests PASSED

- [ ] **Step 5: Update `quality_gate/__init__.py`**

```python
from .pipeline import ConversionPipeline

__all__ = ["ConversionPipeline"]
```

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: 22 tests PASSED (8 + 5 + 5 + 4)

- [ ] **Step 7: Add integration cell to mdapp.ipynb**

Add a new cell after the existing `convert_pdf_to_md` definition:

```python
from quality_gate import ConversionPipeline

pipeline = ConversionPipeline(fallback_fn=convert_pdf_to_md)

# Test: EPA letter (previously returned only "SAMPLE LETTER" with MarkItDown)
result = pipeline.convert("data/epa_sample_letter_sent_to_commissioners_dated_february_29_2015.pdf")
print(result[:500])
```

Run the cell. Expected: full EPA letter text (not just "SAMPLE LETTER").

- [ ] **Step 8: Commit**

```bash
git add quality_gate/__init__.py quality_gate/pipeline.py tests/test_pipeline.py mdapp.ipynb
git commit -m "feat: add ConversionPipeline with MarkItDown → QualityGate → EasyOCR fallback"
```
