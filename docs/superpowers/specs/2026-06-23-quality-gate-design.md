# Quality Gate: MarkItDown → EasyOCR Fallback Pipeline

**Date:** 2026-06-23
**Scope:** PDFs and images embedded in PDFs

## Problem

MarkItDown with a vision LLM (e.g., `llava`) can produce two distinct failure modes:

1. **Sparse/empty output** — the model returns far less text than the document contains (e.g., the EPA letter returned only `"SAMPLE LETTER"` for a 2-page dense letter).
2. **Visual description instead of extraction** — the model narrates what the document looks like rather than extracting its text content (e.g., `"The image shows two separate documents…"`).

The pipeline must detect both failures automatically and fall back to EasyOCR when needed.

## Architecture

```
PDF/Image
   │
   ▼
MarkItDown (with LLM vision model)
   │
   ▼
QualityGate
   ├── Stage 1: HeuristicEvaluator
   │     ├── density check: chars / page_count
   │     │     < 100 chars/page  → FAIL
   │     │     > 500 chars/page  → PASS
   │     │     100–500           → AMBIGUOUS
   │     │
   │     └── pattern check: regex for visual-description phrases
   │           any match         → FAIL (skips Stage 2)
   │           no match          → continue
   │
   └── Stage 2: LLMJudge  (only when AMBIGUOUS after Stage 1)
         → pass / fail
   │
   ▼
PASS → return MarkItDown result
FAIL → EasyOCR pipeline (convert_pdf_to_md)
```

## Components

### `HeuristicEvaluator`

```python
def evaluate(text: str, page_count: int) -> Literal["pass", "fail", "ambiguous"]
```

- Computes `len(text) / page_count` and buckets it against `CHARS_PER_PAGE_FAIL` and `CHARS_PER_PAGE_PASS`.
- Checks `text` against `VISUAL_PATTERNS` (case-insensitive regex). Any match → `"fail"` immediately.
- Never raises; returns one of three verdicts.

### `LLMJudge`

```python
def evaluate(text: str, page_count: int) -> Literal["pass", "fail"]
```

- Called only when `HeuristicEvaluator` returns `"ambiguous"`.
- Reuses the existing `ollama` client with `llama3.2`.
- Sends a short structured prompt asking whether the output is a proper text extraction.
- Parses a `pass`/`fail` keyword from the response.
- Returns `"fail"` conservatively if the response is unparseable.

### `QualityGate`

```python
def check(text: str, page_count: int) -> tuple[Literal["pass", "fail"], str]
```

- Composes `HeuristicEvaluator` → `LLMJudge`.
- Returns `(verdict, reason)` — the reason string is used for logging.
- `LLMJudge` is injected at construction (allows mocking in tests).

### `ConversionPipeline`

```python
def convert(file_path: str) -> str
```

- Runs MarkItDown on the file.
- Extracts page count via `pypdfium2` (already available in the project).
- Calls `QualityGate.check(text, page_count)`.
- On `"pass"` → returns MarkItDown result.
- On `"fail"` → calls existing `convert_pdf_to_md()` (EasyOCR path).

## Constants

```python
CHARS_PER_PAGE_FAIL = 100   # below → clearly bad
CHARS_PER_PAGE_PASS = 500   # above → clearly good

VISUAL_PATTERNS = [
    r"the image (shows|displays|contains|depicts)",
    r"on the (left|right) side",
    r"appears to be",
    r"the (slide|document|page) (shows|displays|contains)",
    r"bullet points (indicating|showing|listing)",
]
```

These are module-level constants; adjust thresholds without touching logic.

## Error Handling

| Failure point | Behavior |
|---|---|
| MarkItDown raises | Log, go directly to EasyOCR; skip QualityGate |
| LLMJudge unparseable response | Return `"fail"` conservatively |
| LLMJudge raises | Return `"fail"` conservatively |
| EasyOCR also fails | Raise with message identifying which stage failed |

## Out of Scope

- Confidence scores or probability outputs from the judge
- Per-document-type threshold tuning
- Caching of judge results
- Support for non-PDF file types (future extension)
