#!/usr/bin/env python3
"""Validate important environment defaults for Nexus Alpha.
Exits with non-zero on no-go settings.
"""
import os
import sys

RECOMMENDED = {
    'OLLAMA_EMBED_CONCURRENCY': '1',
    'OLLAMA_CONNECT_TIMEOUT': '30',
    'OLLAMA_MAX_ATTEMPTS': '6',
    'OLLAMA_BACKOFF_BASE': '1.5',
    'LLM_ALLOW_CLOUD': '0',
}

bad = []
for k, want in RECOMMENDED.items():
    val = os.environ.get(k)
    if val is None:
        print(f'WARN: {k} is not set; recommended default: {want}')
    else:
        try:
            # basic numeric checks for numeric-looking vars
            if k in ('OLLAMA_EMBED_CONCURRENCY', 'OLLAMA_MAX_ATTEMPTS'):
                if int(val) > int(want):
                    bad.append((k, val, want))
            elif k in ('OLLAMA_CONNECT_TIMEOUT',):
                if float(val) < float(want):
                    bad.append((k, val, want))
            elif k == 'LLM_ALLOW_CLOUD':
                if val not in ('0', 'false', 'False'):
                    bad.append((k, val, want))
        except Exception:
            print(f'Could not validate {k} value: {val}')

if bad:
    print('\nThe following environment settings are outside conservative recommendations:')
    for k, val, want in bad:
        print(f' - {k}: {val} (recommended: {want})')
    print('\nPlease review infra/monitoring/ollama_env_defaults.env and adjust before running training/eval on constrained hosts.')
    sys.exit(2)

print('Environment validation passed (or only warnings).')
