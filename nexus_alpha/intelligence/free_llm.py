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
import os
import re
import time
from typing import Any

import httpx

from nexus_alpha.config import LLMConfig
from nexus_alpha.logging import get_logger

logger = get_logger(__name__)

# Optional Prometheus metrics. If prometheus_client is not available we provide
# no-op fallbacks so the module can still be used without adding a hard dependency.
try:
    from prometheus_client import Counter, Histogram

    METRICS_REQUESTS = Counter(
        "nexus_ollama_requests_total",
        "Total Ollama requests attempted",
        ["model", "type"],
    )
    METRICS_FAILURES = Counter(
        "nexus_ollama_failures_total",
        "Total Ollama failures",
        ["model", "type"],
    )
    METRICS_RETRIES = Counter(
        "nexus_ollama_retries_total",
        "Ollama retry attempts",
        ["model"],
    )
    METRICS_LATENCY = Histogram(
        "nexus_ollama_request_latency_seconds",
        "Latency of Ollama requests",
        ["model", "type"],
    )
except Exception:
    class _Noop:
        def labels(self, *_, **__):
            return self

        def inc(self, *_, **__):
            return None

        def observe(self, *_, **__):
            return None

    METRICS_REQUESTS = METRICS_FAILURES = METRICS_RETRIES = METRICS_LATENCY = _Noop()


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
        strict_fail: bool = True,
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
        # If strict_fail is True, failures raise RuntimeError when no fallback is configured.
        # If False, the client degrades gracefully and returns an empty result instead.
        self._strict_fail = strict_fail
        # Internal tuning: increase retries/backoff for local model latency
        self._ollama_max_attempts = max(3, int(os.getenv('OLLAMA_MAX_ATTEMPTS','5')))
        self._ollama_backoff_base = float(os.getenv('OLLAMA_BACKOFF_BASE','1.0'))
        self._ollama_connect_timeout = float(os.getenv('OLLAMA_CONNECT_TIMEOUT','30.0'))

    @classmethod
    def from_config(cls, cfg: LLMConfig) -> "FreeLLMClient":
        # Allow operator to relax strict failure behaviour via env var LLM_ALLOW_DEGRADE=1
        allow_degrade = os.getenv("LLM_ALLOW_DEGRADE", "0") in ("1", "true", "True")
        return cls(
            ollama_base_url=cfg.ollama_base_url,
            primary_model=cfg.ollama_primary_model,
            fast_model=cfg.ollama_fast_model,
            reasoning_model=cfg.ollama_reasoning_model,
            embed_model=cfg.ollama_embed_model,
            use_groq_fallback=cfg.use_groq_fallback,
            groq_api_key=cfg.groq_api_key.get_secret_value(),
            groq_model=cfg.groq_model,
            strict_fail=not allow_degrade,
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
            # If Groq fallback is configured and a key is present, try it.
            if self._use_groq_fallback and self._groq_key:
                return await self._groq_complete(prompt, system, temperature, max_tokens)
            # If strict failures are enabled, raise to preserve legacy/test behaviour.
            if self._strict_fail:
                raise RuntimeError("Ollama unavailable and Groq fallback is disabled") from err
            # Otherwise degrade gracefully and return empty string; callers may handle this.
            logger.warning("ollama_degraded", error=repr(err))
            return ""

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
        """Use the fast model (Mistral) for low-latency structured tasks.

        If the fast model itself fails (rare), return empty string or raise depending on strict_fail.
        """
        try:
            return await self.complete(prompt, system=system, model=self._fast, temperature=temperature)
        except Exception as err:
            logger.warning("fast_model_failed", error=repr(err))
            if self._strict_fail:
                raise
            return ""

    async def complete_reasoning(self, prompt: str, system: str = "", temperature: float = 0.1) -> str:
        """Use the reasoning model (DeepSeek-R1) for debate/cross-validation.

        This method prefers the reasoning model but falls back to the fast model if the heavy model
        times out or fails, to ensure downstream tasks receive a result.
        """
        try:
            return await self.complete(prompt, system=system, model=self._reasoning, temperature=temperature)
        except Exception as err:
            logger.warning("reasoning_model_failed", error=repr(err))
            # fall back to fast model for best-effort; record a fallback metric if available
            try:
                from nexus_alpha.monitoring.metrics import LLM_FALLBACKS
                try:
                    LLM_FALLBACKS.labels(from_model=self._reasoning, to_model=self._fast).inc()
                except Exception:
                    LLM_FALLBACKS.inc()
            except Exception:
                pass
            try:
                return await self.complete_fast(prompt, system=system, temperature=temperature)
            except Exception:
                if self._strict_fail:
                    raise
                return ""

    # ── Embeddings ───────────────────────────────────────────────────────────

    async def embed(self, text: str) -> list[float]:
        """Generate embeddings using nomic-embed-text via Ollama (free, 768-dim)."""
        # Use a semaphore to limit concurrent embedding requests (embeddings can be heavy)
        if not hasattr(self, "_embed_semaphore"):
            self._embed_semaphore = asyncio.Semaphore(int(os.getenv("OLLAMA_EMBED_CONCURRENCY", "4")))
        async with self._embed_semaphore:
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
        # Instrumentation: record request attempt and measure latency
        try:
            METRICS_REQUESTS.labels(model=model, type="completion").inc()
        except Exception:
            pass
        start_time = time.monotonic()

        timeout = httpx.Timeout(self._timeout, connect=self._ollama_connect_timeout)
        async with httpx.AsyncClient(timeout=timeout) as client:
            last_error: Exception | None = None
            for attempt in range(self._ollama_max_attempts):
                try:
                    resp = await client.post(f"{self._ollama_url}/api/chat", json=payload)
                    resp.raise_for_status()
                    content = resp.json()["message"]["content"]
                    # Record latency
                    try:
                        METRICS_LATENCY.labels(model=model, type="completion").observe(time.monotonic() - start_time)
                    except Exception:
                        pass
                    return content
                except httpx.HTTPStatusError as err:
                    # Permanent (4xx) errors — log and propagate
                    body = err.response.text[:500] if err.response is not None else None
                    logger.warning("ollama_http_error", status=err.response.status_code if err.response is not None else None, body=body)
                    try:
                        METRICS_FAILURES.labels(model=model, type="http").inc()
                    except Exception:
                        pass
                    raise
                except (httpx.RequestError, httpx.TransportError, asyncio.TimeoutError) as err:
                    last_error = err
                    # count this retry attempt
                    try:
                        METRICS_RETRIES.labels(model=model).inc()
                        METRICS_FAILURES.labels(model=model, type="transport").inc()
                    except Exception:
                        pass
                    if attempt < self._ollama_max_attempts - 1:
                        jitter = float(os.getenv("OLLAMA_BACKOFF_JITTER", "0.3"))
                        sleep_for = self._ollama_backoff_base * (2 ** attempt) + (jitter * attempt)
                        logger.info("ollama_retry", attempt=attempt + 1, sleep_for=f"{sleep_for:.2f}s", error=repr(err))
                        await asyncio.sleep(sleep_for)
                        continue
                    # exhausted
                    logger.warning("ollama_retries_exhausted", error=repr(last_error))
                    try:
                        METRICS_FAILURES.labels(model=model, type="exhausted").inc()
                    except Exception:
                        pass
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
