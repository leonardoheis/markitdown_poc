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
