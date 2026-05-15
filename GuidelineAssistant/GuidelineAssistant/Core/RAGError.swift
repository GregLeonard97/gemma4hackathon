import Foundation

enum RAGError: LocalizedError, Sendable {
    case modelNotDownloaded
    case modelNotLoaded
    case embeddingFailed(String)
    case retrievalFailed(String)
    case generationFailed(String)
    case resourceMissing(String)
    case invalidData(String)

    var errorDescription: String? {
        switch self {
        case .modelNotDownloaded:
            return "The Gemma 4 E2B model has not been downloaded yet."
        case .modelNotLoaded:
            return "The language model is not loaded."
        case .embeddingFailed(let reason):
            return "Embedding failed: \(reason)"
        case .retrievalFailed(let reason):
            return "Retrieval failed: \(reason)"
        case .generationFailed(let reason):
            return "Generation failed: \(reason)"
        case .resourceMissing(let name):
            return "Missing required resource: \(name)"
        case .invalidData(let details):
            return "Invalid data: \(details)"
        }
    }
}
