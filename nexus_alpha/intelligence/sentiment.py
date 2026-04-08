"""
Hybrid sentiment analysis pipeline — free, production-grade.

Architecture:
  Fast path  → FinBERT (110M params, CPU, ~30ms/batch) — handles 90% of articles
  Deep path  → Qwen3:8b via Ollama — handles high-stakes or ambiguous articles

FinBERT is trained specifically on financial text (ProsusAI/finbert, Apache 2.0).
It outperforms generic LLMs on pure sentiment classification at 1/100th the cost.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

# Must be set before transformers is imported anywhere — prevents HuggingFace
# tokenizers from spawning subprocesses that crash when stdin is closed (daemon mode).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from nexus_alpha.logging import get_logger

if TYPE_CHECKING:
    from nexus_alpha.intelligence.free_llm import FreeLLMClient

logger = get_logger(__name__)

_DEEP_ANALYSIS_KEYWORDS = frozenset({
    "sec", "hack", "ban", "etf", "regulation", "arrest", "lawsuit",
    "crash", "bankrupt", "liquidat", "exploit", "rug", "flash crash",
    "federal reserve", "interest rate", "cpi", "inflation",
})

_SENTIMENT_SYSTEM = (
    "You are a financial sentiment analyst. Output ONLY valid JSON. "
    "No preamble. No markdown. No explanation."
)

_SENTIMENT_PROMPT = """\
Analyze this crypto news for trading sentiment:

{text}

Output ONLY this JSON:
{{
  "sentiment_score": <float -1.0 to 1.0>,
  "confidence": <float 0.0 to 1.0>,
  "affected_assets": ["BTC", "ETH"],
  "event_type": "regulatory|macro|hack|partnership|technical|other",
  "time_horizon": "immediate|short|medium|long",
  "reasoning": "<1 sentence factual>"
}}
"""


@dataclass
class SentimentResult:
    sentiment_score: float   # -1.0 (bearish) to 1.0 (bullish)
    confidence: float        # 0.0 to 1.0
    label: str               # positive / neutral / negative
    method: str              # finbert_fast | qwen3_deep | groq_fallback
    affected_assets: list[str] = None  # type: ignore[assignment]
    event_type: str = "other"
    time_horizon: str = "short"
    reasoning: str = ""

    def __post_init__(self) -> None:
        if self.affected_assets is None:
            self.affected_assets = []


class FinBERTAnalyzer:
    """
    FinBERT: financial sentiment, 110M params, Apache 2.0.
    Runs on CPU in ~30ms per batch. No API cost.
    """

    _LABEL_MAP = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

    def __init__(self, model_name: str = "ProsusAI/finbert") -> None:
        self._model_name = model_name
        self._pipeline: Any = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            from transformers import (  # type: ignore[import]
                AutoModelForSequenceClassification,
                AutoTokenizer,
                pipeline,
            )

            tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=-1,  # CPU
                num_workers=0,  # no worker subprocesses — avoids bad-fd on closed stdin
            )
            self._loaded = True
            logger.info("finbert_loaded", model=self._model_name)
        except ImportError:
            logger.warning("transformers_not_installed", hint="pip install transformers")
            raise

    def score(self, text: str) -> SentimentResult:
        self._ensure_loaded()
        result = self._pipeline(text[:512])[0]
        sentiment = self._LABEL_MAP[result["label"]] * result["score"]
        return SentimentResult(
            sentiment_score=round(sentiment, 4),
            confidence=round(result["score"], 4),
            label=result["label"],
            method="finbert_fast",
        )

    def score_batch(self, texts: list[str]) -> list[SentimentResult]:
        self._ensure_loaded()
        truncated = [t[:512] for t in texts]
        raw = self._pipeline(truncated, batch_size=16)
        return [
            SentimentResult(
                sentiment_score=round(self._LABEL_MAP[r["label"]] * r["score"], 4),
                confidence=round(r["score"], 4),
                label=r["label"],
                method="finbert_fast",
            )
            for r in raw
        ]


class HybridSentimentPipeline:
    """
    Two-stage sentiment pipeline:
      1. FinBERT (fast, free, CPU) — scores all articles instantly
      2. Qwen3:8b via Ollama — deep analysis for high-stakes articles only

    This minimises Ollama load while maximising signal quality.
    """

    def __init__(
        self,
        llm_client: "FreeLLMClient",
        finbert_model: str = "ProsusAI/finbert",
        deep_analysis_confidence_threshold: float = 0.85,
        deep_analysis_sentiment_threshold: float = 0.6,
        max_concurrent_deep_analyses: int = 2,
        deep_analysis_enabled: bool = True,
    ) -> None:
        self._llm = llm_client
        self._finbert = FinBERTAnalyzer(finbert_model)
        self._deep_conf_threshold = deep_analysis_confidence_threshold
        self._deep_sent_threshold = deep_analysis_sentiment_threshold
        self._deep_analysis_semaphore = asyncio.Semaphore(max_concurrent_deep_analyses)
        self._deep_analysis_enabled = deep_analysis_enabled

    def _needs_deep_analysis(self, article: dict[str, Any], fb: SentimentResult) -> bool:
        if not self._deep_analysis_enabled:
            return False
        title_lower = article.get("title", "").lower()
        has_key_term = any(kw in title_lower for kw in _DEEP_ANALYSIS_KEYWORDS)
        high_confidence = fb.confidence > self._deep_conf_threshold
        extreme_sentiment = abs(fb.sentiment_score) > self._deep_sent_threshold
        return has_key_term or high_confidence or extreme_sentiment

    async def _deep_analyze(self, article: dict[str, Any]) -> SentimentResult:
        text = f"{article.get('title', '')}\n{article.get('text', '')[:500]}"
        try:
            async with self._deep_analysis_semaphore:
                # Use reasoning model but allow fallback to fast model if heavy model times out
                try:
                    data = await self._llm.complete_json(
                        _SENTIMENT_PROMPT.format(text=text),
                        system=_SENTIMENT_SYSTEM,
                        model=self._llm._reasoning,
                    )
                except Exception as err:
                    # log and fallback to fast model (best-effort)
                    logger.warning("deep_reasoning_failed_try_fast", error=repr(err))
                    data_raw = await self._llm.complete_fast(_SENTIMENT_PROMPT.format(text=text), system=_SENTIMENT_SYSTEM)
                    try:
                        import json as _json
                        data = _json.loads(_strip_json_fences(data_raw))
                    except Exception:
                        data = {"sentiment_score": 0.0, "confidence": 0.0, "affected_assets": [], "event_type": "other", "time_horizon": "short", "reasoning": ""}
            return SentimentResult(
                sentiment_score=float(data.get("sentiment_score", 0.0)),
                confidence=float(data.get("confidence", 0.5)),
                label="positive" if data.get("sentiment_score", 0) > 0.1 else
                      "negative" if data.get("sentiment_score", 0) < -0.1 else "neutral",
                method="qwen3_deep",
                affected_assets=data.get("affected_assets", []),
                event_type=data.get("event_type", "other"),
                time_horizon=data.get("time_horizon", "short"),
                reasoning=data.get("reasoning", ""),
            )
        except Exception as err:
            logger.warning("deep_analysis_failed", error=repr(err))
            return SentimentResult(
                sentiment_score=0.0, confidence=0.0, label="neutral", method="error"
            )

    async def process_articles(self, articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Score all articles. Deep-analyze high-stakes ones via Ollama."""
        if not articles:
            return []

        texts = [f"{a.get('title', '')}. {a.get('text', '')[:200]}" for a in articles]
        try:
            fb_scores = self._finbert.score_batch(texts)
        except Exception as err:
            logger.warning("finbert_failed", error=str(err))
            fb_scores = [
                SentimentResult(0.0, 0.5, "neutral", "fallback") for _ in articles
            ]

        enriched = []
        deep_tasks = []
        deep_indices: list[int] = []

        for i, (article, fb) in enumerate(zip(articles, fb_scores)):
            article = dict(article)
            article["sentiment_finbert"] = {
                "score": fb.sentiment_score,
                "confidence": fb.confidence,
                "label": fb.label,
            }
            if self._needs_deep_analysis(article, fb):
                deep_indices.append(i)
                deep_tasks.append(self._deep_analyze(article))
            enriched.append(article)

        if deep_tasks:
            deep_results = await asyncio.gather(*deep_tasks, return_exceptions=True)
            for idx, result in zip(deep_indices, deep_results):
                if isinstance(result, SentimentResult):
                    enriched[idx]["sentiment"] = {
                        "score": result.sentiment_score,
                        "confidence": result.confidence,
                        "label": result.label,
                        "method": result.method,
                        "affected_assets": result.affected_assets,
                        "event_type": result.event_type,
                        "time_horizon": result.time_horizon,
                        "reasoning": result.reasoning,
                    }

        # Fill non-deep-analyzed articles with FinBERT result
        for i, article in enumerate(enriched):
            if "sentiment" not in article:
                fb = fb_scores[i]
                article["sentiment"] = {
                    "score": fb.sentiment_score,
                    "confidence": fb.confidence,
                    "label": fb.label,
                    "method": fb.method,
                    "affected_assets": [],
                    "event_type": "other",
                    "time_horizon": "short",
                    "reasoning": "",
                }

        return enriched

    def score_single(self, text: str) -> SentimentResult:
        """Fast single-article scoring via FinBERT (synchronous)."""
        return self._finbert.score(text)
