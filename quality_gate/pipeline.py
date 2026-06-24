import logging
from typing import Callable
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
        try:
            return len(pdf)
        finally:
            pdf.close()
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
