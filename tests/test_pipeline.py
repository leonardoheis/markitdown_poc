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
