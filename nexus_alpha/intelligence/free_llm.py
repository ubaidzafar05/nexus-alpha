"""
Free LLM client — Ollama local primary, with optional Groq fallback.

Priority order:
  1. Ollama local server (qwen3:8b / deepseek-r1:8b) — zero cost, always-on
  2. Optional Groq fallback if explicitly enabled
  3. Hard error if Ollama is unavailable and fallback is disabled

Usage::
    client = FreeLLMClient.from_config(config.llm)
    result = await client.complete("Analyze this news...")
    embedding = await client.embed("Some text")
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

import httpx

from nexus_alpha.config import LLMConfig
from nexus_alpha.logging import get_logger

logger = get_logger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _strip_json_fences(text: str) -> str:
    match = _JSON_FENCE_RE.search(text)
    return match.group(1) if match else text.strip()


class FreeLLMClient:
    """
    Unified async LLM client using only free backends.

    Ollama is the primary backend. Groq is optional and disabled by default.
    """

    def __init__(
        self,
        ollama_base_url: str,
        primary_model: str,
        fast_model: str,
        reasoning_model: str,
        embed_model: str,
        use_groq_fallback: bool = False,
        groq_api_key: str = "",
        groq_model: str = "llama-3.3-70b-versatile",
        timeout: float = 60.0,
    ) -> None:
        self._ollama_url = ollama_base_url.rstrip("/")
        self._primary = primary_model
        self._fast = fast_model
        self._reasoning = reasoning_model
        self._embed_model = embed_model
        self._use_groq_fallback = use_groq_fallback
        self._groq_key = groq_api_key
        self._groq_model = groq_model
        self._timeout = timeout
        self._last_ollama_warning_at = 0.0

    @classmethod
    def from_config(cls, cfg: LLMConfig) -> "FreeLLMClient":
        return cls(
            ollama_base_url=cfg.ollama_base_url,
            primary_model=cfg.ollama_primary_model,
            fast_model=cfg.ollama_fast_model,
            reasoning_model=cfg.ollama_reasoning_model,
            embed_model=cfg.ollama_embed_model,
            use_groq_fallback=cfg.use_groq_fallback,
            groq_api_key=cfg.groq_api_key.get_secret_value(),
            groq_model=cfg.groq_model,
        )

    # ── Core completion ──────────────────────────────────────────────────────

    async def complete(
        self,
        prompt: str,
        system: str = "You are a financial analysis assistant. Be precise and factual.",
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a text completion. Uses Groq only if explicitly enabled."""
        chosen_model = model or self._primary
        try:
            return await self._ollama_complete(prompt, system, chosen_model, temperature, max_tokens)
        except Exception as err:
            self._log_ollama_warning(err)
            if not (self._use_groq_fallback and self._groq_key):
                raise RuntimeError("Ollama unavailable and Groq fallback is disabled") from err
            return await self._groq_complete(prompt, system, temperature, max_tokens)

    async def complete_json(
        self,
        prompt: str,
        system: str = "Output ONLY valid JSON. No markdown. No explanation.",
        model: str | None = None,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """Complete and parse JSON response."""
        raw = await self.complete(prompt, system=system, model=model, temperature=temperature)
        clean = _strip_json_fences(raw)
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            logger.warning("json_parse_failed", raw_preview=raw[:200])
            raise

    async def complete_fast(self, prompt: str, system: str = "", temperature: float = 0.1) -> str:
        """Use the fast model (Mistral) for low-latency structured tasks."""
        return await self.complete(prompt, system=system, model=self._fast, temperature=temperature)

    async def complete_reasoning(self, prompt: str, system: str = "", temperature: float = 0.1) -> str:
        """Use the reasoning model (DeepSeek-R1) for debate/cross-validation."""
        return await self.complete(prompt, system=system, model=self._reasoning, temperature=temperature)

    # ── Embeddings ───────────────────────────────────────────────────────────

    async def embed(self, text: str) -> list[float]:
        """Generate embeddings using nomic-embed-text via Ollama (free, 768-dim)."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._ollama_url}/api/embeddings",
                json={"model": self._embed_model, "prompt": text},
            )
            resp.raise_for_status()
            return resp.json()["embedding"]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embeddings — sequential calls to Ollama."""
        results = []
        for text in texts:
            results.append(await self.embed(text))
        return results

    # ── Health check ─────────────────────────────────────────────────────────

    async def health_check(self) -> dict[str, Any]:
        """Verify Ollama is running and models are available."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._ollama_url}/api/tags")
                resp.raise_for_status()
                models = [m["name"] for m in resp.json().get("models", [])]
                return {"status": "ok", "backend": "ollama", "models": models}
        except Exception as err:
            backend = "groq" if self._use_groq_fallback and self._groq_key else "none"
            return {"status": "degraded", "backend": backend, "error": repr(err)}

    def _log_ollama_warning(self, err: Exception) -> None:
        now = time.monotonic()
        if now - self._last_ollama_warning_at < 30.0:
            return
        self._last_ollama_warning_at = now
        logger.warning(
            "ollama_unavailable",
            error=repr(err),
            fallback="groq" if self._use_groq_fallback and self._groq_key else "disabled",
        )

    # ── Ollama backend ───────────────────────────────────────────────────────

    async def _ollama_complete(
        self,
        prompt: str,
        system: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "stream": False,
        }
        timeout = httpx.Timeout(self._timeout, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            last_error: Exception | None = None
            for attempt in range(2):
                try:
                    resp = await client.post(f"{self._ollama_url}/api/chat", json=payload)
                    resp.raise_for_status()
                    return resp.json()["message"]["content"]
                except Exception as err:
                    last_error = err
                    if attempt == 0:
                        await asyncio.sleep(0.5)
                        continue
                    raise
            if last_error is not None:
                raise last_error
            raise RuntimeError("Ollama completion failed without an exception")

    # ── Groq fallback ─────────────────────────────────────────────────────────

    async def _groq_complete(
        self,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self._groq_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._groq_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
