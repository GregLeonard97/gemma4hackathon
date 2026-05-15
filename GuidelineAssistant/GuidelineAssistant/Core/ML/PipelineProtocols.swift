import Foundation

protocol EmbeddingEngineProtocol: Sendable {
    func embed(_ text: String) async throws -> [Float]
}

protocol VectorStoreProtocol: Sendable {
    func search(queryEmbedding: [Float], topK: Int) throws -> [RetrievedChunk]
}

protocol LLMEngineProtocol: Sendable {
    func generate(
        systemPrompt: String,
        userPrompt: String
    ) -> AsyncThrowingStream<String, Error>
}