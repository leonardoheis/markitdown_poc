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
