"""
Tests for the free LLM backend used by OpenClaw agents.

The old Anthropic API path is replaced by FreeLLMClient (Ollama/Groq).
These tests verify that _llm_analyze delegates correctly to FreeLLMClient
and that the free LLM client routes to the configured Ollama backend.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from nexus_alpha.config import LLMConfig
from nexus_alpha.intelligence.openclaw_agents import BreakingNewsAgent
from nexus_alpha.intelligence.free_llm import FreeLLMClient


@pytest.mark.asyncio
async def test_llm_analyze_uses_free_llm_client() -> None:
    """_llm_analyze should delegate to FreeLLMClient.complete, not Anthropic API."""
    agent = BreakingNewsAgent(llm_config=LLMConfig())

    mock_client = AsyncMock()
    mock_client.complete.return_value = "analysis result"

    with patch(
        "nexus_alpha.intelligence.free_llm.FreeLLMClient.from_config",
        return_value=mock_client,
    ):
        result = await agent._llm_analyze("system prompt", "user content")  # noqa: SLF001

    assert result == "analysis result"
    mock_client.complete.assert_called_once_with("user content", system="system prompt")


@pytest.mark.asyncio
async def test_llm_analyze_returns_empty_on_error() -> None:
    """_llm_analyze should return empty string (not raise) if LLM is unavailable."""
    agent = BreakingNewsAgent(llm_config=LLMConfig())

    mock_client = AsyncMock()
    mock_client.complete.side_effect = RuntimeError("ollama not reachable")

    with patch(
        "nexus_alpha.intelligence.free_llm.FreeLLMClient.from_config",
        return_value=mock_client,
    ):
        result = await agent._llm_analyze("system", "user")  # noqa: SLF001

    assert result == ""


def test_free_llm_client_from_config_uses_ollama_defaults() -> None:
    """FreeLLMClient.from_config should use Ollama URL and model from LLMConfig."""
    cfg = LLMConfig()
    client = FreeLLMClient.from_config(cfg)
    assert client._ollama_url == cfg.ollama_base_url  # noqa: SLF001
    assert client._primary == cfg.ollama_primary_model  # noqa: SLF001
    assert client._groq_key == cfg.groq_api_key.get_secret_value()  # noqa: SLF001
    assert client._use_groq_fallback is False  # noqa: SLF001


def test_llm_config_model_name_respects_explicit_override() -> None:
    """model_name property falls back to explicit primary_model override."""
    cfg = LLMConfig(primary_model="custom-model")
    assert cfg.model_name == "custom-model"


def test_llm_config_model_name_default_returns_ollama_primary() -> None:
    """model_name property returns ollama_primary_model by default."""
    cfg = LLMConfig()
    assert cfg.model_name == cfg.ollama_primary_model
