import logging
import os
import tempfile
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


class ExtractionError(Exception):
    pass


class RobustConversionPipeline:
    def __init__(
        self,
        gate: QualityGate | None = None,
        md_client=None,
        ocr_reader=None,
        ocr_languages: list[str] | None = None,
    ):
        if md_client is None:
            client = OpenAI(base_url=_OLLAMA_BASE_URL, api_key="ollama")
            self._md = MarkItDown(llm_client=client, llm_model=_VISION_MODEL)
        else:
            self._md = md_client
        self._gate = gate if gate is not None else QualityGate()
        self._ocr_reader = ocr_reader
        self._ocr_languages = ocr_languages or ["en"]

    def _get_ocr_reader(self):
        if self._ocr_reader is None:
            import easyocr
            self._ocr_reader = easyocr.Reader(self._ocr_languages)
        return self._ocr_reader

    def _try_markitdown(self, file_path: str) -> str | None:
        try:
            return self._md.convert(file_path).text_content
        except Exception as exc:
            logger.warning("MarkItDown failed for %s: %s", file_path, exc)
            return None

    def _run_easyocr(self, file_path: str) -> str | None:
        try:
            reader = self._get_ocr_reader()
            if not file_path.lower().endswith(".pdf"):
                return "".join(reader.readtext(file_path, detail=0, paragraph=True))
            pdf = pdfium.PdfDocument(file_path)
            pages = []
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    for i, page in enumerate(pdf):
                        img_path = os.path.join(tmp_dir, f"page_{i+1}.png")
                        page.render(scale=2).to_pil().save(img_path)
                        pages.append("".join(reader.readtext(img_path, detail=0, paragraph=True)))
            finally:
                pdf.close()
            return "\n\n".join(pages)
        except Exception as exc:
            logger.warning("EasyOCR failed for %s: %s", file_path, exc)
            return None

    def convert(self, file_path: str) -> str:
        page_count = _get_page_count(file_path)

        md_text = self._try_markitdown(file_path)
        if md_text is not None:
            verdict, reason = self._gate.check(md_text, page_count)
            logger.info("MarkItDown gate [%s]: %s", verdict, reason)
            if verdict == "pass":
                return md_text

        logger.info("Falling back to EasyOCR for %s", file_path)
        ocr_text = self._run_easyocr(file_path)
        if ocr_text is not None:
            verdict, reason = self._gate.check(ocr_text, page_count)
            logger.info("EasyOCR gate [%s]: %s", verdict, reason)
            if verdict == "pass":
                return ocr_text

        raise ExtractionError(
            f"Both MarkItDown and EasyOCR failed quality check for {file_path}"
        )
