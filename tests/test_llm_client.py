"""
Unit tests for Stage 3: LLM Integration.
"""

from unittest.mock import MagicMock, patch
import pytest
import openai
from pipeline.llm_client import LLMClient


def test_llm_client_missing_api_key_returns_error():
    client = LLMClient(api_key="")
    result = client.generate(messages=[{"role": "user", "content": "Hi"}], runtime_api_key="")
    assert result.success is False
    assert "API key is missing" in result.error_message


@patch("pipeline.llm_client.OpenAI")
def test_llm_client_successful_response(mock_openai_class):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    # Mock response object
    mock_choice = MagicMock()
    mock_choice.message.content = "This is a real model response."
    mock_choice.finish_reason = "stop"

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 15
    mock_usage.completion_tokens = 7
    mock_usage.total_tokens = 22

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    mock_response.model = "gpt-4o-mini-test"

    mock_client.chat.completions.create.return_value = mock_response

    llm = LLMClient(api_key="sk-test-key")
    result = llm.generate(
        messages=[{"role": "user", "content": "Hello"}],
        model="gpt-4o-mini",
        temperature=0.5,
    )

    assert result.success is True
    assert result.raw_response == "This is a real model response."
    assert result.prompt_tokens == 15
    assert result.completion_tokens == 7
    assert result.total_tokens == 22
    assert result.finish_reason == "stop"
    assert result.latency_seconds >= 0.0


@patch("pipeline.llm_client.OpenAI")
def test_llm_client_handles_authentication_error(mock_openai_class):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.chat.completions.create.side_effect = openai.AuthenticationError(
        message="Invalid credentials", response=MagicMock(), body=None
    )

    llm = LLMClient(api_key="sk-invalid")
    result = llm.generate(messages=[{"role": "user", "content": "Hello"}])

    assert result.success is False
    assert "Authentication Error" in result.error_message


@patch("pipeline.llm_client.OpenAI")
def test_llm_client_handles_rate_limit_error(mock_openai_class):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.chat.completions.create.side_effect = openai.RateLimitError(
        message="Rate limit reached", response=MagicMock(), body=None
    )

    llm = LLMClient(api_key="sk-test")
    result = llm.generate(messages=[{"role": "user", "content": "Hello"}])

    assert result.success is False
    assert "Rate Limit Exceeded" in result.error_message
