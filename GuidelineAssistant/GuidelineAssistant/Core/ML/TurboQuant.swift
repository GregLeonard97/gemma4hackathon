import Accelerate
import Foundation

struct TurboQuantCodebook: Codable, Sendable {
    let centroids: [Float]
    let bitsPerCoordinate: Int
    let rotationMatrix: [Float]
    let dimension: Int
}

struct CompressedVector: Sendable {
    let indices: [UInt8]
    let norm: Float
}

final class TurboQuant: @unchecked Sendable {
    let codebook: TurboQuantCodebook

    private let levels: Int
    private let thresholds: [Float]
    private let rotationRowMajor: [Float]
    private let rotationTransposeRowMajor: [Float]

    init(codebook: TurboQuantCodebook) {
        precondition(codebook.dimension > 0, "Codebook dimension must be positive")
        precondition((1...8).contains(codebook.bitsPerCoordinate), "bitsPerCoordinate must be in [1, 8]")

        let expectedLevels = 1 << codebook.bitsPerCoordinate
        precondition(codebook.centroids.count == expectedLevels, "Centroid count must equal 2^bitsPerCoordinate")
        precondition(codebook.rotationMatrix.count == codebook.dimension * codebook.dimension, "Rotation matrix must be dimension x dimension")

        self.codebook = codebook
        self.levels = expectedLevels

        self.thresholds = zip(codebook.centroids, codebook.centroids.dropFirst()).map { lhs, rhs in
            0.5 * (lhs + rhs)
        }

        self.rotationRowMajor = TurboQuant.columnMajorToRowMajor(
            codebook.rotationMatrix,
            dimension: codebook.dimension
        )
        self.rotationTransposeRowMajor = TurboQuant.transposeRowMajor(
            rotationRowMajor,
            dimension: codebook.dimension
        )
    }

    static func load(from url: URL) throws -> TurboQuant {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        let codebook = try decoder.decode(TurboQuantCodebook.self, from: data)
        return TurboQuant(codebook: codebook)
    }

    func compress(_ vector: [Float]) -> CompressedVector {
        precondition(vector.count == codebook.dimension, "Vector dimension mismatch")

        let norm = l2Norm(vector)
        guard norm > 0 else {
            return CompressedVector(indices: Array(repeating: 0, count: codebook.dimension), norm: 0)
        }

        let unit = vector.map { $0 / norm }
        let rotated = matrixVectorMultiply(
            matrixRowMajor: rotationRowMajor,
            vector: unit,
            rows: codebook.dimension,
            cols: codebook.dimension
        )

        let quantized = rotated.map { value in
            quantize(value)
        }

        return CompressedVector(indices: quantized, norm: norm)
    }

    func decompress(_ compressed: CompressedVector) -> [Float] {
        guard compressed.norm > 0 else {
            return Array(repeating: 0, count: codebook.dimension)
        }

        let indices = normalizeIndices(compressed.indices)
        let rotated = indices.map { index in
            codebook.centroids[Int(index)]
        }

        let unitApprox = matrixVectorMultiply(
            matrixRowMajor: rotationTransposeRowMajor,
            vector: rotated,
            rows: codebook.dimension,
            cols: codebook.dimension
        )

        return unitApprox.map { $0 * compressed.norm }
    }

    func cosineSimilarity(query: [Float], compressed: CompressedVector) -> Float {
        guard query.count == codebook.dimension else {
            return 0
        }

        let queryNorm = l2Norm(query)
        guard queryNorm > 0 else {
            return 0
        }

        let queryUnit = query.map { $0 / queryNorm }
        let queryRotated = matrixVectorMultiply(
            matrixRowMajor: rotationRowMajor,
            vector: queryUnit,
            rows: codebook.dimension,
            cols: codebook.dimension
        )

        let indices = normalizeIndices(compressed.indices)
        let rotatedApprox = indices.map { index in
            codebook.centroids[Int(index)]
        }

        let denom = l2Norm(rotatedApprox)
        guard denom > 0 else {
            return 0
        }

        var dot: Float = 0
        vDSP_dotpr(queryRotated, 1, rotatedApprox, 1, &dot, vDSP_Length(codebook.dimension))
        return dot / denom
    }

    private func normalizeIndices(_ indices: [UInt8]) -> [UInt8] {
        if indices.count == codebook.dimension {
            return indices.map { min($0, UInt8(levels - 1)) }
        }

        var output = Array(repeating: UInt8(0), count: codebook.dimension)
        for i in 0..<min(indices.count, codebook.dimension) {
            output[i] = min(indices[i], UInt8(levels - 1))
        }
        return output
    }

    private func quantize(_ value: Float) -> UInt8 {
        var low = 0
        var high = thresholds.count

        while low < high {
            let mid = (low + high) / 2
            if value >= thresholds[mid] {
                low = mid + 1
            } else {
                high = mid
            }
        }

        return UInt8(min(low, levels - 1))
    }

    private func matrixVectorMultiply(
        matrixRowMajor: [Float],
        vector: [Float],
        rows: Int,
        cols: Int
    ) -> [Float] {
        var output = Array(repeating: Float(0), count: rows)
        var vectorCopy = vector

        matrixRowMajor.withUnsafeBufferPointer { matrixPtr in
            vectorCopy.withUnsafeMutableBufferPointer { vectorPtr in
                output.withUnsafeMutableBufferPointer { outputPtr in
                    vDSP_mmul(
                        matrixPtr.baseAddress!,
                        1,
                        vectorPtr.baseAddress!,
                        1,
                        outputPtr.baseAddress!,
                        1,
                        vDSP_Length(rows),
                        1,
                        vDSP_Length(cols)
                    )
                }
            }
        }

        return output
    }

    private func l2Norm(_ vector: [Float]) -> Float {
        var squaresSum: Float = 0
        vDSP_svesq(vector, 1, &squaresSum, vDSP_Length(vector.count))
        return sqrtf(squaresSum)
    }

    private static func columnMajorToRowMajor(_ matrix: [Float], dimension: Int) -> [Float] {
        var rowMajor = Array(repeating: Float(0), count: matrix.count)
        for row in 0..<dimension {
            for col in 0..<dimension {
                rowMajor[row * dimension + col] = matrix[col * dimension + row]
            }
        }
        return rowMajor
    }

    private static func transposeRowMajor(_ matrix: [Float], dimension: Int) -> [Float] {
        var transposed = Array(repeating: Float(0), count: matrix.count)
        for row in 0..<dimension {
            for col in 0..<dimension {
                transposed[col * dimension + row] = matrix[row * dimension + col]
            }
        }
        return transposed
    }
}
