# NeoGuide Hackathon Technical Writeup

## Introduction

Postgraduate medical training in the UK often requires frequent rotations between hospitals and NHS trusts. Core clinical principles remain stable, but local medication choices, dosing protocols, and follow-up pathways vary across institutions. In practice, local guidance is frequently fragmented across intranet systems, and delays in accessing the right document can add friction during time-critical workflows.

NeoGuide was built to reduce that friction using a clinically grounded, source-cited, on-device RAG workflow. The project uses Gemma 4 class models (primarily E2B in production configuration) to answer process-driven clinical queries in natural language while preserving a transparent path back to source guideline documents.

## Design

The system was designed to run locally after initial model setup, to align with strict healthcare governance constraints and reduce external dependency risk. The iOS app is implemented in Swift and tested on high-memory iPhone hardware. It returns natural-language responses, cites source guideline context, and supports PDF page-level navigation from citations.

The development corpus was 164 neonatal guideline PDFs, processed on Mac into a compressed on-device retrieval corpus containing 10,022 chunks. The built iOS assets included:

- a compressed SQLite retrieval database (~9.69 MB)
- rendered table/flowchart images (~60 MB)

At runtime, clinical queries are embedded locally (BGE-small-en-v1.5), top-k retrieval is performed over TurboQuant-compressed vectors, and Gemma 4 E2B on MLX generates grounded answers under clinical safety constraints.

E4B was investigated during development but did not fit stable on-device memory constraints for clinically realistic prompt lengths in this implementation window.

## Technical Execution

### Ingestion and Chunking

A Mac-side ingestion pipeline transforms PDF guidelines into retrieval-ready artifacts:

- text extraction with PyMuPDF (selected for strong table/layout handling)
- context-preserving page concatenation
- chunking with LangChain RecursiveCharacterTextSplitter (800 chars, 150 overlap)

This configuration balanced context retention with retrieval precision.

### Table-Aware Retrieval

Because many clinical protocols are table-driven, table extraction is first-class. Each detected table yields:

- row-level chunks (for precise value lookups)
- a table-summary chunk
- a rendered table image (for full-table requests)

This enables different interaction modes:

- direct lookups (for example, dose queries) return row-level evidence
- full-table requests surface table image artifacts in the UI

### Hospital-Agnostic Pipeline

The pipeline is corpus-agnostic by construction. Replacing the PDF corpus and rebuilding resources produces a functional system for another hospital/service context with minimal prompt adaptation.

### Embeddings and Compression

A single embedding model is used for indexing and query-time embedding: BAAI/bge-small-en-v1.5 (384-dim), selected for quality-memory tradeoff on mobile. TurboQuant was implemented at multiple bit-depths. On internal retrieval experiments, 3-bit compression achieved the strongest R@5 among tested settings while preserving compact on-device footprint.

### Query Normalization and Retrieval Heuristics

Bidirectional abbreviation expansion improves retrieval robustness for clinical shorthand (for example, acronym and expanded forms). Query intent heuristics plus retrieval metadata support better handling of algorithm/table-style requests in the UI.

### Safety Prompting and Refusal Logic

Safety behavior is enforced with prompt constraints rather than task-specific fine-tuning to preserve portability across institutions. Rules include:

- interpretation conventions for clinical wording
- strict verbatim behavior for doses/thresholds
- refusal when evidence is absent or insufficient
- calculation constraints requiring explicit available inputs

### On-Device Generation and Systems Work

Gemma 4 E2B (4-bit variant) was used as the grounded generator with conservative decoding settings. During optimization, multiple memory-reduction and runtime-stability interventions were tested, including token limits, retrieval depth constraints, memory entitlement updates, and KV-cache investigations.

A reproducible failure path was traced to upstream Gemma 4 KV-cache handling in mlx-swift-lm when quantized cache routes are invoked. A workaround was applied by disabling KV-cache quantization in production configuration pending upstream resolution.

## Evaluation

A 50-case clinical test set was authored to stress practical usage categories:

- dosing and interval questions
- contact/process lookups
- table retrieval interactions
- deliberate failure-mode probes

Two model configurations (E2B and E4B) were blind-compared with clinician scoring and an LLM-as-judge path used as a secondary signal.

Reported outcomes in development experiments:

- mean quality score: E4B 2.42/3.00, E2B 2.44/3.00
- both scored 3/3 on 31 of 50 cases
- one clinically dangerous answer was observed in each model variant
- no clear hallucination advantage was observed for either model in final blind pass

The evaluation framework was still valuable for iterative improvement. Earlier configurations had lower quality and worse safety outcomes; retrieval and prompting improvements materially improved final behavior.

## Limitations

- E2B-first optimization constrains potential capability headroom versus larger variants.
- The evaluation dataset is limited (50 cases), and not sufficient for deployment-level validation.
- Clinical deployment would require substantially larger, external, and prospective validation.
- Retrieval quality can be strong while generation quality remains variable in difficult edge cases.

## Data and Redistribution

Code and architecture are open-source under Apache 2.0.

The clinical corpus used during development consists of UCLH-internal neonatal guidelines and is not redistributable. Derived evaluation artifacts tied to that corpus are also excluded from this public repository.

## Conclusion

NeoGuide targets a practical healthcare workflow problem under strict privacy and usability constraints. The system is designed for transparent, source-grounded assistance with refusal-by-default behavior when evidence is missing.

The technical architecture is intentionally reusable: users can substitute local guideline corpora, regenerate resources, and deploy an institution-specific on-device assistant without changing the core app architecture.

## References

1. Hilton J. Careless costs related to inefficient technology used within NHS England. Clin Med (Lond). 2020 Jan;20(1):115. doi: 10.7861/clinmed.2019-0340.
2. Zandieh A, Daliri M, Hadian M, Mirrokni V. TurboQuant: Online vector quantization with near-optimal distortion rate. arXiv:2504.19874, 2025.
