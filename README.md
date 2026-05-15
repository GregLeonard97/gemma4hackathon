# NeoGuide - On-Device Clinical Guideline RAG for Hospital Clinicians

A native iOS assistant that answers guideline questions on-device with source-linked evidence, built to work under strict clinical privacy and connectivity constraints.

![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg) ![Gemma 4 Good Hackathon](https://img.shields.io/badge/Gemma%204%20Good-Hackathon%20Submission-0A7B83)

## What this is

NeoGuide is a clinical retrieval-augmented generation (RAG) system for point-of-care guideline lookup on iPhone. It combines a Mac-side ingestion pipeline (PDF parsing, table extraction, vector indexing) with an on-device Swift app that retrieves relevant chunks and generates grounded natural-language answers. The app is designed for scenarios where cloud access is undesirable or restricted, and where clinicians need transparent traceability back to source documents. The architecture is corpus-agnostic: users can supply their own local guideline PDFs and rebuild resources for their institution.

## Demo

- Demo video: to be added before final submission
- Representative app visuals are documented in the hackathon writeup: [WRITEUP.md](./WRITEUP.md)

## Key Features

- Source-grounded responses with guideline/page provenance shown in the app
- Fully on-device inference and retrieval workflow after initial setup
- Tap-through citation flow from answer to PDF source context
- Table-aware retrieval with image surfacing for full-table queries
- Hospital-agnostic ingestion pipeline for bring-your-own corpus deployment

## Architecture

Architecture details are documented in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

The system has two stages: a Mac resource build pipeline and an iOS runtime pipeline. Mac-side scripts ingest PDF guidelines, extract text and tables, and produce app resources (embedding model assets plus vector database artifacts). On device, the Swift app embeds the query, retrieves top-k chunks, and performs grounded generation with citation metadata exposed in the UI.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for implementation-level details.

## Quick Start

### Prerequisites

- macOS for the ingestion/resource pipeline
- Xcode 16+ for iOS build/deploy
- Python 3.11
- iPhone 17 Pro Max or equivalent class device (12GB RAM, iOS 18+ recommended)

### Bringing your own clinical guidelines

Important: this repository does not include the clinical PDF corpus used during development.

The clinical PDF corpus used during development consists of UCLH-internal neonatal guidelines and is not redistributable. Users must supply their own clinical guideline PDFs to run the system.

Place your own PDFs at `data/guidelines/` before running ingestion.

### Steps

1. Clone the repository.
2. Install Python dependencies:
   `pip install -r scripts/requirements_resources.txt`
3. Place your clinical PDFs in `data/guidelines/`.
4. Run ingestion:
   `python ingestion/ingest.py`
5. Build iOS resources:
   `bash scripts/build_ios_resources.sh`
6. Open the Xcode project and deploy to a physical iOS device.

Note: `BGESmall.mlpackage` is intentionally not committed in this public repository. It is generated/downloaded locally by the resource build pipeline.

Note: the Gemma 4 model is downloaded on first app launch from Hugging Face.

## Engineering

- `ingestion/` - Mac-side document processing and retrieval preparation
- `scripts/` - Build scripts for iOS resource generation
- `GuidelineAssistant/` - Native iOS Swift app and tests
- `docs/` - Architecture and setup documentation

## Evaluation

The system was evaluated against a 50-case clinical test set written by a paediatric clinician, scored via blinded human review and an LLM-as-judge workflow. Quantitative findings are reported in [WRITEUP.md](./WRITEUP.md).

The specific test set used during development was UCLH-specific and is not included in this repository. Users evaluating against their own corpus should construct an equivalent local test set.

## Hackathon Submission

This repository was submitted to the [Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon) (Kaggle, May 2026) in the Health category.

The technical writeup ([WRITEUP.md](./WRITEUP.md)) describes design decisions, evaluation methodology, and key engineering challenges, including an upstream issue in mlx-swift-lm Gemma 4 KV-cache handling.

## Acknowledgements

- Gemma 4 E2B model: Google DeepMind, released under Apache 2.0
- BGE-small-en-v1.5 embeddings: BAAI
- TurboQuant compression: Zandieh et al., 2025 (arXiv:2504.19874)
- MLX framework: Apple ML Research

## License

Apache License 2.0. See [LICENSE](./LICENSE).

## Author

Greg Leonard - Paediatric doctor and NIHR Academic Clinical Fellow, LSHTM
