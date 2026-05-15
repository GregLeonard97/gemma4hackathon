# GuidelineAssistant Backend (iOS)

Native iOS backend for an on-device clinical guideline RAG assistant.

This folder contains backend-only Swift code:
- ML stack (embedding + Gemma generation)
- TurboQuant compression/decompression and compressed similarity search
- SQLite vector retrieval
- Query orchestration pipeline with event streaming
- Unit tests for parity, compression, retrieval, and integration flow

No SwiftUI view layer is included.

## Requirements

- Xcode 16+ (Swift 6 mode)
- iOS deployment target 17.0+
- iPhone 16 Pro target profile (or any iOS 17+ device with sufficient memory)
- Swift Package dependencies:
  - mlx-swift (`from: 0.31.3`)
    - `MLX`
    - `MLXNN`
  - mlx-swift-lm (`from: 3.31.3`)
    - `MLXVLM`
    - `MLXLMCommon`
  - SQLite.swift (`0.15+`)

## Setup

### 1. Add Package Dependencies

Add the following in Xcode (File -> Add Package Dependencies):

- `https://github.com/ml-explore/mlx-swift` (from `0.31.3`)
- `https://github.com/ml-explore/mlx-swift-lm` (from `3.31.3`)
- `https://github.com/stephencelis/SQLite.swift` (0.15+)

Link products to the app target:
- `MLX`
- `MLXNN`
- `MLXVLM`
- `MLXLMCommon`
- `SQLite`

### 2. Configure Entitlements

In target entitlements, add:

```xml
<key>com.apple.developer.kernel.increased-memory-limit</key>
<true/>
<key>com.apple.developer.kernel.extended-virtual-addressing</key>
<true/>
```

### 3. No Post-Build Dylib Codesign Script Required

MLX Swift dependencies are regular Swift Package targets and do not require a post-build dylib codesign script.

## Resource Placement

Place these files into the app target resources (the `GuidelineAssistant/Resources` group in Xcode):

- `BGESmall.mlpackage`
- `bge_vocab.txt`
- `codebook.json`
- `guidelines.db`
- `extracted_tables/` directory
- `Guidelines/` directory (source PDFs)

Optional test resource:
- `embedding_reference.json` (Python reference vector for `"neonatal sepsis early-onset"`)

## Build

1. Open your Xcode project/workspace that includes this source tree.
2. Confirm Signing & Capabilities is configured.
3. Select an iOS 17+ simulator/device target.
4. Build the app target.

## Run Tests

Run all backend tests from Xcode or CLI:

```bash
xcodebuild test \
  -scheme GuidelineAssistant \
  -destination 'platform=iOS Simulator,name=iPhone 16 Pro'
```

Included tests:
- `EmbeddingTests`
- `TurboQuantTests`
- `VectorStoreTests`
- `RAGPipelineIntegrationTests`

## Backend API Surface

Primary integration points for frontend:
- `RAGPipeline.query(_:) -> AsyncThrowingStream<QueryEvent, Error>`
- Domain models:
  - `Message`
  - `Source`
  - `RetrievedChunk`
  - `QueryEvent`

## Notes on Behavior

- Retrieval is stateless per query.
- LLM generation uses a fresh MLXVLM chat session every call.
- `VectorStore` performs an O(n) scan with compressed cosine scoring.
- `EmbeddingEngine` applies L2 normalization before returning embeddings.

## Deviations from Spec

1. `RAGPipeline` uses protocol-based dependency injection (`LLMEngineProtocol`, `EmbeddingEngineProtocol`, `VectorStoreProtocol`) to make integration tests deterministic and avoid hard coupling to heavyweight runtime dependencies.
2. Memory warning handling in `LLMEngine` is conservative: because each generation uses a short-lived conversation, there is no persistent conversation KV cache to reset between turns. On serious/critical thermal states, the model is unloaded.
3. `EmbeddingTests` and parity validation are resource-gated: tests skip when `BGESmall`, `bge_vocab.txt`, or `embedding_reference.json` are not present in test resources.
4. `VectorStoreTests` seed a temporary SQLite test DB with the production schema to validate retrieval logic without requiring a full bundled production database artifact.
