"""NEXUS-ALPHA: Self-Evolving Autonomous Trading Intelligence v3.0"""

__version__ = "3.0.0"

# Prewarm FinBERT and other heavy models if configured to reduce first-call latency.
try:
    from .config import load_config
    cfg = load_config('.env')
    if getattr(cfg, 'llm', None) and cfg.llm.finbert_enabled:
        # Importing the module triggers lazy model loading elsewhere; pre-import reduces latency
        import threading
        def _prewarm():
            try:
                # Import FinBERT stub — actual implementation in nexus_alpha.intelligence.sentiment
                from .intelligence.sentiment import FinBERT
                FinBERT.prewarm()
            except Exception:
                pass
        t = threading.Thread(target=_prewarm, daemon=True)
        t.start()
except Exception:
    pass
