import XCTest
@testable import GuidelineAssistant

final class LLMEngineSmokeTest: XCTestCase {
    /// Skipped unless model is downloaded. Verifies we can load Gemma 4 E2B
    /// and stream at least one non-empty output chunk.
    func testLoadAndGenerate() async throws {
        guard ModelStorage.shared.isModelDownloaded else {
            throw XCTSkip("Model not downloaded. Run onboarding download first.")
        }

        let engine = LLMEngine()
        try await engine.loadModel()

        XCTAssertEqual(engine.state, .ready)

        let stream = engine.generate(
            systemPrompt: "You are a helpful assistant.",
            userPrompt: "Say hello in one word."
        )

        var receivedAny = false
        for try await token in stream {
            if !token.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                receivedAny = true
                break
            }
        }

        XCTAssertTrue(receivedAny)
        await engine.unload()
    }
}
