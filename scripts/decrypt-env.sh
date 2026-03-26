#!/usr/bin/env bash
set -euo pipefail

INPUT_FILE="${1:-.env.enc}"
OUTPUT_FILE="${2:-.env}"

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "Encrypted file not found: $INPUT_FILE" >&2
  exit 1
fi

openssl enc -d -aes-256-cbc -pbkdf2 -in "$INPUT_FILE" -out "$OUTPUT_FILE"
chmod 600 "$OUTPUT_FILE"
echo "Decrypted $INPUT_FILE -> $OUTPUT_FILE"
