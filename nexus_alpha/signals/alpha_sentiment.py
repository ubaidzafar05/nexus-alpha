import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class SentimentEngine:
    """Consolidated Sentiment Engine using FinBERT with Lazy-Loading."""
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.tokenizer = None
        self.model = None
        self.nlp = None
        
        # Sources
        self.news_rss_urls = [
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "https://cointelegraph.com/rss"
        ]
        
    def _lazy_init(self):
        """Load transformers models and dependencies only when needed."""
        if self.nlp is not None:
            return
            
        try:
            import feedparser
            self._feedparser = feedparser
        except ImportError:
            logger.error("Missing dependency: feedparser")
            self._feedparser = None

        # V6 ULTRA: Skip initialization if no token provided to prevent hung states
        from nexus_alpha.config import LLMConfig
        if not LLMConfig().hf_token:
            logger.warning("skipping_finbert_load_no_token")
            # We use a dummy NLP that returns neutral
            self.nlp = lambda x: [{"label": "neutral", "score": 0.0}] * (len(x) if isinstance(x, list) else 1)
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
        except ImportError as e:
            logger.error(f"Missing sentiment dependencies: {e}. Run 'pip install transformers torch'")
            raise
            
        logger.info("loading_finbert_model_lazy")
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        device = 0 if self.use_gpu and torch.cuda.is_available() else -1
        self.nlp = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer, device=device)

    def get_news_sentiment(self) -> float:
        """Fetch latest news headlines and score them."""
        self._lazy_init()
        headlines = []
        try:
            for url in self.news_rss_urls:
                feed = self._feedparser.parse(url)
                for entry in feed.entries[:5]: # Take top 5 from each
                    headlines.append(entry.title)
                    
            if not headlines:
                return 0.0
                
            results = self.nlp(headlines)
            # FinBERT labels: positive, negative, neutral
            scores = []
            for res in results:
                if res['label'] == 'positive':
                    scores.append(res['score'])
                elif res['label'] == 'negative':
                    scores.append(-res['score'])
                else:
                    scores.append(0.0)
            
            return sum(scores) / len(scores)
        except Exception as e:
            logger.error(f"News sentiment error: {e}")
            return 0.0

    def get_social_sentiment(self, symbol: str) -> float:
        """Mock social 'hype' detection - can be expanded with Twitter/Reddit APIs."""
        # For now, we simulate social buzz by looking for 'breakout' keywords in a mock stream
        # In production, this would use a real-time stream search
        base = symbol.split('/')[0]
        mock_buzz = 0.1 # Baseline
        
        # Simulate hype detection
        if base in ["BTC", "ETH", "SOL"]:
            mock_buzz += 0.05
            
        return mock_buzz

    def generate_signal(self, symbol: str) -> Dict[str, Any]:
        """Produce a combined sentiment signal."""
        news_score = self.get_news_sentiment()
        social_score = self.get_social_sentiment(symbol)
        
        # Combine: News is 70% weight (stable), Social is 30% (hype)
        combined_score = (news_score * 0.7) + (social_score * 0.3)
        
        direction = 0
        if combined_score > 0.15:
            direction = 1
        elif combined_score < -0.15:
            direction = -1
            
        return {
            "direction": direction,
            "confidence": min(abs(combined_score), 1.0),
            "metadata": {
                "news_score": float(news_score),
                "social_score": float(social_score),
                "combined_score": float(combined_score)
            }
        }
