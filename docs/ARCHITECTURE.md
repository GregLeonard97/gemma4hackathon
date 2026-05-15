# NeoGuide Architecture

NeoGuide is split into two execution domains: a Mac-side resource build pipeline and an iOS on-device runtime pipeline.

## 1) Mac-side build pipeline

Input:
- A local corpus of clinical guideline PDFs supplied by the user

Processing:
- Ingestion and extraction in [ingestion/](../ingestion/)
  - PDF parsing and text extraction
  - table detection and row-level chunking
  - optional rendered table image outputs
- Resource build scripts in [scripts/](../scripts/)
  - compressed vector DB build
  - Core ML embedding model conversion
  - vocabulary extraction and resource validation

Output:
- App-bundle resources consumed by iOS runtime

## 2) iOS runtime pipeline

Core app and ML stack live in [GuidelineAssistant/](../GuidelineAssistant/):

- Query input and UI event flow
- Local embedding generation (BGE-small Core ML package)
- On-device vector retrieval (top-k chunk selection)
- Grounded answer generation with Gemma 4 class model on MLX
- Source metadata for citation and PDF page navigation

## 3) Safety and transparency

The runtime behavior is constrained by a clinical safety prompt strategy and refusal rules. Answers are generated from retrieved excerpts and linked to source context in the app UI for clinician verification.

## 4) Portability model

The architecture is corpus-agnostic. To adapt NeoGuide to another institution or specialty:

1. Replace guideline PDFs with a local corpus.
2. Re-run ingestion and resource build scripts.
3. Rebuild and deploy the iOS app.

No cloud retraining is required for baseline operation.
