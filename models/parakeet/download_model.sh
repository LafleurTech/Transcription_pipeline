#!/usr/bin/env bash
set -euo pipefail

# Script to download the Parakeet .nemo model into this directory.
# Usage: ./download_model.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_URL="https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/resolve/main/parakeet-tdt-0.6b-v2.nemo"
FILENAME="$(basename "${RAW_URL}")"
OUT_PATH="${SCRIPT_DIR}/${FILENAME}"

if [[ -f "${OUT_PATH}" ]]; then
  echo "File already exists: ${OUT_PATH} (skipping download)"
  exit 0
fi

echo "Downloading model: ${RAW_URL}" 
echo "Destination: ${OUT_PATH}" 

TMP_FILE="${OUT_PATH}.part"
trap 'rm -f "${TMP_FILE}"' INT TERM EXIT

download() {
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --progress-bar -o "${TMP_FILE}" "${RAW_URL}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${TMP_FILE}" "${RAW_URL}"
  else
    echo "ERROR: Neither curl nor wget is installed." >&2
    return 1
  fi
}

if ! download; then
  echo "Download failed." >&2
  exit 1
fi

# Basic size check (> 1MB) to reduce chance of saving an HTML error page
SIZE=$(wc -c < "${TMP_FILE}" || echo 0)
if [[ ${SIZE} -lt 1000000 ]]; then
  echo "ERROR: Downloaded file is unexpectedly small (${SIZE} bytes). Aborting." >&2
  exit 1
fi

mv "${TMP_FILE}" "${OUT_PATH}"
trap - INT TERM EXIT
echo "Download complete: ${OUT_PATH}"

exit 0
