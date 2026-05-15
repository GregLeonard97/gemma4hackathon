import XCTest
@testable import GuidelineAssistant

final class TurboQuantTests: XCTestCase {
    func testRoundTripMSEAt3BitCompression() {
        let dimension = 384
        let codebook = makeTestCodebook(dimension: dimension, bits: 3)
        let turboQuant = TurboQuant(codebook: codebook)

        var rng = SeededGenerator(seed: 1337)
        var totalSquaredError: Float = 0
        var totalCount: Int = 0

        for _ in 0..<100 {
            let vector = randomUnitVector(dimension: dimension, rng: &rng)
            let compressed = turboQuant.compress(vector)
            let reconstructed = turboQuant.decompress(compressed)

            for i in 0..<dimension {
                let delta = vector[i] - reconstructed[i]
                totalSquaredError += delta * delta
                totalCount += 1
            }
        }

        let mse = totalSquaredError / Float(totalCount)
        XCTAssertLessThan(mse, 0.05)
    }

    private func makeTestCodebook(dimension: Int, bits: Int) -> TurboQuantCodebook {
        let levels = 1 << bits
        let sigma = 1.0 / sqrt(Float(dimension))
        let minValue = -2.5 * sigma
        let maxValue = 2.5 * sigma
        let step = (maxValue - minValue) / Float(levels - 1)

        let centroids = (0..<levels).map { minValue + Float($0) * step }

        var rotationMatrix = Array(repeating: Float(0), count: dimension * dimension)
        for i in 0..<dimension {
            // Column-major identity matrix.
            rotationMatrix[i * dimension + i] = 1
        }

        return TurboQuantCodebook(
            centroids: centroids,
            bitsPerCoordinate: bits,
            rotationMatrix: rotationMatrix,
            dimension: dimension
        )
    }

    private func randomUnitVector(dimension: Int, rng: inout SeededGenerator) -> [Float] {
        var vector = Array(repeating: Float(0), count: dimension)
        var normSq: Float = 0

        for i in 0..<dimension {
            let value = Float.random(in: -1...1, using: &rng)
            vector[i] = value
            normSq += value * value
        }

        let norm = sqrtf(normSq)
        guard norm > 0 else { return vector }
        return vector.map { $0 / norm }
    }
}

private struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        state = 6364136223846793005 &* state &+ 1
        return state
    }
}
