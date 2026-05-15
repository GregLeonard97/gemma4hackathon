# Setup Guide

This guide explains how to run NeoGuide with your own clinical guideline corpus.

## Prerequisites

- macOS (for ingestion and iOS resource builds)
- Python 3.11
- Xcode 16+
- iOS 18+ device with high-memory profile recommended for Gemma 4 class models

## 1. Install dependencies

From repository root:

pip install -r scripts/requirements_resources.txt

## 2. Bring your own guideline corpus

Create/place your own PDF corpus at:

data/guidelines/

Important: this repository does not ship the development corpus.

The clinical PDF corpus used during development consists of UCLH-internal neonatal guidelines and is not redistributable. You must provide your own PDFs.

## 3. Run ingestion

python ingestion/ingest.py

This builds retrieval-ready artifacts from your corpus.

## 4. Build iOS resources

bash scripts/build_ios_resources.sh

This orchestrates:
- codebook/vector DB build
- embedding model conversion
- tokenizer/vocab asset extraction
- resource copy/validation steps

Note: `GuidelineAssistant/GuidelineAssistant/Resources/BGESmall.mlpackage` is intentionally excluded from version control and is generated/downloaded locally by this pipeline.

## 5. Build and run iOS app

1. Open [GuidelineAssistant/GuidelineAssistant.xcodeproj](../GuidelineAssistant/GuidelineAssistant.xcodeproj).
2. Select a physical device target.
3. Build and run.

On first launch, Gemma 4 model files are downloaded from Hugging Face.

## 6. Validation checks

Before deployment with a new corpus:

- Verify retrieval returns correct local guideline sections.
- Verify table requests surface expected table artifacts.
- Verify refusal behavior on out-of-corpus questions.
- Run unit/integration tests in [GuidelineAssistant/GuidelineAssistantTests/](../GuidelineAssistant/GuidelineAssistantTests/).

## Troubleshooting

- If Chroma collection is not found, ensure commands are run from repository root so relative paths resolve to root data directories.
- If iOS resources are missing, rerun scripts in order and confirm output in app Resources group.
- If model load fails on device, verify memory entitlements and available storage.
