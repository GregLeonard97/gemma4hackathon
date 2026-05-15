import Foundation
import XCTest
@testable import GuidelineAssistant

final class RAGPipelineIntegrationTests: XCTestCase {
    func testPipelineEventOrderAndExpandedPromptUsage() async throws {
        let query = "show me the full LP table"
        let expanded = AbbreviationExpander.expand(query)

        let embedder = MockEmbedder(embedding: makeUnitQueryEmbedding())
        let vectorStore = MockVectorStore(chunks: [
            RetrievedChunk(
                id: 1,
                source: "Antibiotics_Guideline.pdf",
                page: 4,
                section: "Dosing",
                chunkType: .tableRow,
                content: "LP dosing row",
                tableImage: "lp_table.png",
                imageRefs: [],
                similarity: 0.93
            )
        ])
        let llm = MemoryAwareMockLLM(tokens: ["Use ", "the table."])

        let pipeline = RAGPipeline(
            llm: llm,
            embedder: embedder,
            vectorStore: vectorStore,
            repository: GuidelineRepository()
        )

        var events: [QueryEvent] = []
        for try await event in pipeline.query(query) {
            events.append(event)
        }

        XCTAssertGreaterThanOrEqual(events.count, 5)

        guard case .retrievalStarted = events[0] else {
            XCTFail("Expected retrievalStarted as first event")
            return
        }

        guard case .retrievalComplete(let sources, let tableImage) = events[1] else {
            XCTFail("Expected retrievalComplete as second event")
            return
        }

        XCTAssertEqual(tableImage, "lp_table.png")
        XCTAssertEqual(sources.count, 1)

        guard case .generationStarted = events[2] else {
            XCTFail("Expected generationStarted as third event")
            return
        }

        guard case .complete = events.last else {
            XCTFail("Expected complete as final event")
            return
        }

        let tokenEvents = events.compactMap { event -> String? in
            if case .token(let token) = event {
                return token
            }
            return nil
        }
        XCTAssertEqual(tokenEvents.joined(), "Use the table.")

        XCTAssertEqual(await embedder.lastQueryText(), expanded)

        let userPrompt = await llm.lastUserPrompt()
        XCTAssertTrue(userPrompt.contains("QUESTION: \(expanded)"))
        XCTAssertFalse(userPrompt.contains("QUESTION: \(query)\n"))
    }

    func testMemoryWarningDuringGenerationDoesNotBreakSubsequentQuery() async throws {
        let query = "LP indications"

        let embedder = MockEmbedder(embedding: makeUnitQueryEmbedding())
        let vectorStore = MockVectorStore(chunks: [
            RetrievedChunk(
                id: 7,
                source: "Neonatal_Sepsis.pdf",
                page: 2,
                section: "Investigations",
                chunkType: .text,
                content: "Indications for lumbar puncture",
                tableImage: nil,
                imageRefs: [],
                similarity: 0.91
            )
        ])
        let llm = MemoryAwareMockLLM(tokens: ["First ", "answer."])

        let pipeline = RAGPipeline(
            llm: llm,
            embedder: embedder,
            vectorStore: vectorStore,
            repository: GuidelineRepository()
        )

        var firstCompleted = false
        var firstSawToken = false

        for try await event in pipeline.query(query) {
            if case .token = event, !firstSawToken {
                firstSawToken = true
                await llm.handleMemoryWarning()
            }
            if case .complete = event {
                firstCompleted = true
            }
        }

        XCTAssertTrue(firstCompleted)

        var secondCompleted = false
        for try await event in pipeline.query("CRP threshold") {
            if case .complete = event {
                secondCompleted = true
            }
        }

        XCTAssertTrue(secondCompleted)
        XCTAssertEqual(await llm.memoryWarningCount(), 1)
        XCTAssertEqual(await llm.generationCount(), 2)
    }

    private func makeUnitQueryEmbedding() -> [Float] {
        var vector = Array(repeating: Float(0), count: 384)
        vector[0] = 1
        return vector
    }
}

private final class MockEmbedder: EmbeddingEngineProtocol, @unchecked Sendable {
    private let embedding: [Float]
    private let state = LockedState<String?>(nil)

    init(embedding: [Float]) {
        self.embedding = embedding
    }

    func embed(_ text: String) async throws -> [Float] {
        state.set(text)
        return embedding
    }

    func lastQueryText() async -> String? {
        state.get()
    }
}

private final class MockVectorStore: VectorStoreProtocol, @unchecked Sendable {
    private let chunks: [RetrievedChunk]

    init(chunks: [RetrievedChunk]) {
        self.chunks = chunks
    }

    func search(queryEmbedding: [Float], topK: Int) throws -> [RetrievedChunk] {
        Array(chunks.prefix(topK))
    }
}

private final class MemoryAwareMockLLM: LLMEngineProtocol, @unchecked Sendable {
    private let tokens: [String]

    private let userPromptState = LockedState<String>("")
    private let warningState = LockedState<Int>(0)
    private let generationState = LockedState<Int>(0)

    init(tokens: [String]) {
        self.tokens = tokens
    }

    func generate(systemPrompt: String, userPrompt: String) -> AsyncThrowingStream<String, Error> {
        userPromptState.set(userPrompt)
        generationState.modify { $0 += 1 }

        return AsyncThrowingStream { continuation in
            Task {
                for token in tokens {
                    continuation.yield(token)
                    try? await Task.sleep(for: .milliseconds(20))
                }
                continuation.finish()
            }
        }
    }

    func handleMemoryWarning() async {
        warningState.modify { $0 += 1 }
    }

    func lastUserPrompt() async -> String {
        userPromptState.get()
    }

    func memoryWarningCount() async -> Int {
        warningState.get()
    }

    func generationCount() async -> Int {
        generationState.get()
    }
}

private final class LockedState<Value>: @unchecked Sendable {
    private var value: Value
    private let lock = NSLock()

    init(_ value: Value) {
        self.value = value
    }

    func get() -> Value {
        lock.lock()
        defer { lock.unlock() }
        return value
    }

    func set(_ newValue: Value) {
        lock.lock()
        value = newValue
        lock.unlock()
    }

    func modify(_ body: (inout Value) -> Void) {
        lock.lock()
        body(&value)
        lock.unlock()
    }
}
