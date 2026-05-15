import Accelerate
import XCTest
@testable import GuidelineAssistant

final class EmbeddingTests: XCTestCase {
    func testEmbeddingParityWithPythonReference() async throws {
        guard let modelURL = locateResource(named: "BGESmall", ext: "mlpackage")
                ?? locateResource(named: "BGESmall", ext: "mlmodelc") else {
            throw XCTSkip("BGESmall model resource is not available in the test bundle")
        }

        guard let vocabURL = locateResource(named: "bge_vocab", ext: "txt") else {
            throw XCTSkip("bge_vocab.txt is not available in the test bundle")
        }

        guard let referenceURL = locateResource(named: "embedding_reference", ext: "json") else {
            throw XCTSkip("embedding_reference.json is not available in the test bundle")
        }

        let reference = try JSONDecoder().decode([Float].self, from: Data(contentsOf: referenceURL))
        let engine = try EmbeddingEngine(modelURL: modelURL, vocabURL: vocabURL)

        let embedding = try await engine.embed("neonatal sepsis early-onset")
        XCTAssertEqual(embedding.count, reference.count)

        let cosine = cosineSimilarity(embedding, reference)
        XCTAssertGreaterThan(cosine, 0.999)
    }

    private func locateResource(named name: String, ext: String) -> URL? {
        let bundles = [Bundle(for: Self.self), Bundle.main]
        for bundle in bundles {
            if let url = bundle.url(forResource: name, withExtension: ext) {
                return url
            }
        }
        return nil
    }

    private func cosineSimilarity(_ lhs: [Float], _ rhs: [Float]) -> Float {
        guard lhs.count == rhs.count, !lhs.isEmpty else { return 0 }

        var dot: Float = 0
        var lhsNormSq: Float = 0
        var rhsNormSq: Float = 0

        vDSP_dotpr(lhs, 1, rhs, 1, &dot, vDSP_Length(lhs.count))
        vDSP_svesq(lhs, 1, &lhsNormSq, vDSP_Length(lhs.count))
        vDSP_svesq(rhs, 1, &rhsNormSq, vDSP_Length(rhs.count))

        let denom = sqrtf(lhsNormSq) * sqrtf(rhsNormSq)
        return denom > 0 ? dot / denom : 0
    }
}
