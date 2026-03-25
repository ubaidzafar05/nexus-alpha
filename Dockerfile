FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY nexus_alpha/ nexus_alpha/

RUN pip install --no-cache-dir -e ".[dev]"

# ─── Production image ─────────────────────────────────────────────────
FROM base AS production

ENV ENVIRONMENT=production
ENV LOG_LEVEL=INFO
ENV TRADING_MODE=paper

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["python", "-m", "nexus_alpha.cli"]
CMD ["run"]
