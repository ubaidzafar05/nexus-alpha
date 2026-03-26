#!/usr/bin/env bash
set -euo pipefail

INPUT_FILE="${1:-.env}"
OUTPUT_FILE="${2:-${INPUT_FILE}.enc}"

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "Input file not found: $INPUT_FILE" >&2
  exit 1
fi

openssl enc -aes-256-cbc -salt -pbkdf2 -in "$INPUT_FILE" -out "$OUTPUT_FILE"
chmod 600 "$OUTPUT_FILE"
echo "Encrypted $INPUT_FILE -> $OUTPUT_FILE"
