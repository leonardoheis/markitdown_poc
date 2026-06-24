import pytest
from unittest.mock import MagicMock, patch
from quality_gate.pipeline import RobustConversionPipeline, ExtractionError
from quality_gate.gate import QualityGate

def _make_robust(md_text=None, md_raises=False, ocr_text=None,
                 md_gate_verdict="pass", ocr_gate_verdict="pass"):
    mock_md = MagicMock()
    if md_raises:
        mock_md.convert.side_effect = Exception("md error")
    else:
        mock_md.convert.return_value = MagicMock(text_content=md_text)

    mock_gate = MagicMock(spec=QualityGate)
    mock_gate.check.side_effect = [
        (md_gate_verdict, "md reason"),
        (ocr_gate_verdict, "ocr reason"),
    ]

    mock_ocr = MagicMock()
    mock_ocr.readtext.return_value = [ocr_text or ""]

    return RobustConversionPipeline(gate=mock_gate, md_client=mock_md, ocr_reader=mock_ocr)


def test_robust_returns_markitdown_when_gate_passes():
    pipeline = _make_robust(md_text="full text", md_gate_verdict="pass")
    with patch("quality_gate.pipeline._get_page_count", return_value=1):
        assert pipeline.convert("data/test.jpg") == "full text"


def test_robust_falls_to_easyocr_when_markitdown_gate_fails():
    pipeline = _make_robust(md_text="bad", md_gate_verdict="fail",
                            ocr_text="ocr text", ocr_gate_verdict="pass")
    with patch("quality_gate.pipeline._get_page_count", return_value=1):
        assert pipeline.convert("data/test.jpg") == "ocr text"


def test_robust_raises_when_both_fail():
    pipeline = _make_robust(md_text="bad", md_gate_verdict="fail",
                            ocr_text="also bad", ocr_gate_verdict="fail")
    with patch("quality_gate.pipeline._get_page_count", return_value=1):
        with pytest.raises(ExtractionError):
            pipeline.convert("data/test.jpg")


def test_robust_skips_to_easyocr_when_markitdown_raises():
    pipeline = _make_robust(md_raises=True, ocr_text="ocr text", ocr_gate_verdict="pass")
    with patch("quality_gate.pipeline._get_page_count", return_value=1):
        # gate.check should only be called once (for OCR result, not for failed MD)
        assert pipeline.convert("data/test.jpg") == "ocr text"
        assert pipeline._gate.check.call_count == 1
