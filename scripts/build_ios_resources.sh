#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "Building iOS resources..."
mkdir -p GuidelineAssistant/GuidelineAssistant/Resources

echo "[1/5] Generating TurboQuant codebook..."
"${PYTHON_BIN}" scripts/build_codebook.py

echo "[2/5] Building compressed vector database..."
"${PYTHON_BIN}" scripts/build_ios_vector_db.py

echo "[3/5] Converting BGE-small to Core ML..."
"${PYTHON_BIN}" scripts/convert_bge_to_coreml.py

echo "[4/5] Extracting BGE vocabulary..."
"${PYTHON_BIN}" scripts/extract_bge_vocab.py

echo "[5/5] Copying table images and source PDFs..."
"${PYTHON_BIN}" scripts/copy_asset_dirs.py

echo
echo "Done. Output files in GuidelineAssistant/GuidelineAssistant/Resources/:"
ls -lh GuidelineAssistant/GuidelineAssistant/Resources/
