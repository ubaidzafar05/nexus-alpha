FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY README.md ./
COPY nexus_alpha/ nexus_alpha/

RUN pip install --no-cache-dir -e .

# ─── Production image ─────────────────────────────────────────────────
FROM base AS production

ENV ENVIRONMENT=production
ENV LOG_LEVEL=INFO
ENV TRADING_MODE=paper

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=3 \
    CMD python -m nexus_alpha.cli health || exit 1

ENTRYPOINT ["python", "-m", "nexus_alpha.cli"]
CMD ["run"]
