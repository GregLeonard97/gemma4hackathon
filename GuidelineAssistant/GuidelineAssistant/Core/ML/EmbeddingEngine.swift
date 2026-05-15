import Accelerate
import CoreML
import Foundation
import os

actor EmbeddingEngine: EmbeddingEngineProtocol {
    private let model: MLModel
    private let tokenizer: WordPieceTokenizer
    private let maxSequenceLength: Int
    private let logger = Logger(subsystem: "GuidelineAssistant", category: "EmbeddingEngine")

    init() throws {
        let bundle = Bundle.main
        guard let modelURL = bundle.url(forResource: "BGESmall", withExtension: "mlpackage")
            ?? bundle.url(forResource: "BGESmall", withExtension: "mlmodelc") else {
            throw RAGError.resourceMissing("BGESmall.mlpackage")
        }
        guard let vocabURL = bundle.url(forResource: "bge_vocab", withExtension: "txt") else {
            throw RAGError.resourceMissing("bge_vocab.txt")
        }

        try self.init(modelURL: modelURL, vocabURL: vocabURL)
    }

    init(
        modelURL: URL,
        vocabURL: URL,
        maxSequenceLength: Int = 512,
        configuration: MLModelConfiguration = MLModelConfiguration()
    ) throws {
        let loadableModelURL = try Self.makeLoadableModelURL(from: modelURL)
        self.model = try MLModel(contentsOf: loadableModelURL, configuration: configuration)
        self.tokenizer = try WordPieceTokenizer(vocabURL: vocabURL)
        self.maxSequenceLength = maxSequenceLength
    }

    private static func makeLoadableModelURL(from modelURL: URL) throws -> URL {
        let ext = modelURL.pathExtension.lowercased()
        if ext == "mlmodelc" {
            return modelURL
        }
        if ext == "mlpackage" || ext == "mlmodel" {
            return try MLModel.compileModel(at: modelURL)
        }
        return modelURL
    }

    func embed(_ text: String) async throws -> [Float] {
        do {
            let encoded = tokenizer.encode(text: text, maxLength: maxSequenceLength)
            let provider = try makeInputProvider(encoded: encoded)
            let prediction = try await model.prediction(from: provider)
            let embedding = try extractEmbedding(from: prediction)
            let normalized = l2Normalize(embedding)

            let norm = l2Norm(normalized)
            logger.debug("Embedding generated with norm: \(norm, format: .fixed(precision: 6))")

            return normalized
        } catch {
            throw RAGError.embeddingFailed(error.localizedDescription)
        }
    }

    private func makeInputProvider(encoded: EncodedInput) throws -> MLFeatureProvider {
        let modelInputs = model.modelDescription.inputDescriptionsByName

        guard let inputIDsName = resolveInputName(candidates: ["input_ids", "inputIds"], available: modelInputs),
              let maskName = resolveInputName(candidates: ["attention_mask", "attentionMask"], available: modelInputs) else {
            throw RAGError.invalidData("Could not resolve model input names for input_ids/attention_mask")
        }

        let inputIDs = try multiArray(from: encoded.inputIDs)
        let attentionMask = try multiArray(from: encoded.attentionMask)

        var values: [String: MLFeatureValue] = [
            inputIDsName: MLFeatureValue(multiArray: inputIDs),
            maskName: MLFeatureValue(multiArray: attentionMask)
        ]

        if let tokenTypeName = resolveInputName(candidates: ["token_type_ids", "tokenTypeIds"], available: modelInputs) {
            let tokenType = try multiArray(from: encoded.tokenTypeIDs)
            values[tokenTypeName] = MLFeatureValue(multiArray: tokenType)
        }

        return try MLDictionaryFeatureProvider(dictionary: values)
    }

    private func resolveInputName(
        candidates: [String],
        available: [String: MLFeatureDescription]
    ) -> String? {
        for candidate in candidates where available[candidate] != nil {
            return candidate
        }

        for key in available.keys {
            let lower = key.lowercased()
            if candidates.contains(where: { lower.contains($0.lowercased()) }) {
                return key
            }
        }

        return nil
    }

    private func multiArray(from values: [Int32]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, NSNumber(value: values.count)], dataType: .int32)
        for (index, value) in values.enumerated() {
            array[index] = NSNumber(value: value)
        }
        return array
    }

    private func extractEmbedding(from provider: MLFeatureProvider) throws -> [Float] {
        for key in provider.featureNames {
            guard let feature = provider.featureValue(for: key),
                  let array = feature.multiArrayValue else {
                continue
            }

            let embedding = firstTokenEmbedding(from: array)
            if !embedding.isEmpty {
                return embedding
            }
        }

        throw RAGError.invalidData("No MLMultiArray output found for embedding")
    }

    private func firstTokenEmbedding(from array: MLMultiArray) -> [Float] {
        let shape = array.shape.map { $0.intValue }
        let expectedDimension = 384

        switch shape.count {
        case 1:
            return readVector(from: array, count: min(shape[0], expectedDimension))

        case 2:
            if shape[1] == expectedDimension {
                return readVector(from: array, baseIndices: [0], length: expectedDimension)
            }
            if shape[0] == expectedDimension {
                return (0..<expectedDimension).map { idx in
                    readValue(from: array, indices: [idx, 0])
                }
            }

        case 3:
            if shape[2] == expectedDimension {
                return (0..<expectedDimension).map { idx in
                    readValue(from: array, indices: [0, 0, idx])
                }
            }
            if shape[1] == expectedDimension {
                return (0..<expectedDimension).map { idx in
                    readValue(from: array, indices: [0, idx, 0])
                }
            }
            if shape[0] == expectedDimension {
                return (0..<expectedDimension).map { idx in
                    readValue(from: array, indices: [idx, 0, 0])
                }
            }

        default:
            break
        }

        let fallback = readVector(from: array, count: min(array.count, expectedDimension))
        return fallback
    }

    private func readVector(
        from array: MLMultiArray,
        baseIndices: [Int] = [],
        length: Int
    ) -> [Float] {
        var output = Array(repeating: Float(0), count: length)
        for i in 0..<length {
            output[i] = readValue(from: array, indices: baseIndices + [i])
        }
        return output
    }

    private func readVector(from array: MLMultiArray, count: Int) -> [Float] {
        var output = Array(repeating: Float(0), count: count)
        for i in 0..<count {
            output[i] = readValue(from: array, flatIndex: i)
        }
        return output
    }

    private func readValue(from array: MLMultiArray, indices: [Int]) -> Float {
        let strides = array.strides.map { $0.intValue }
        var offset = 0

        for i in 0..<min(indices.count, strides.count) {
            offset += indices[i] * strides[i]
        }

        return readValue(from: array, flatIndex: offset)
    }

    private func readValue(from array: MLMultiArray, flatIndex: Int) -> Float {
        switch array.dataType {
        case .float32:
            return array.dataPointer.bindMemory(to: Float.self, capacity: array.count)[flatIndex]
        case .double:
            let value = array.dataPointer.bindMemory(to: Double.self, capacity: array.count)[flatIndex]
            return Float(value)
        case .float16:
            let raw = array.dataPointer.bindMemory(to: UInt16.self, capacity: array.count)[flatIndex]
            return Float(Float16(bitPattern: raw))
        default:
            return array[flatIndex].floatValue
        }
    }

    private func l2Normalize(_ vector: [Float]) -> [Float] {
        let norm = l2Norm(vector)
        guard norm > 0 else { return vector }

        var divisor = norm
        var output = Array(repeating: Float(0), count: vector.count)
        vDSP_vsdiv(vector, 1, &divisor, &output, 1, vDSP_Length(vector.count))
        return output
    }

    private func l2Norm(_ vector: [Float]) -> Float {
        var sumSquares: Float = 0
        vDSP_svesq(vector, 1, &sumSquares, vDSP_Length(vector.count))
        return sqrtf(sumSquares)
    }
}

private struct EncodedInput {
    let inputIDs: [Int32]
    let attentionMask: [Int32]
    let tokenTypeIDs: [Int32]
}

private struct WordPieceTokenizer {
    private let vocab: [String: Int32]

    private let clsToken = "[CLS]"
    private let sepToken = "[SEP]"
    private let unkToken = "[UNK]"
    private let padToken = "[PAD]"

    init(vocabURL: URL) throws {
        let contents = try String(contentsOf: vocabURL, encoding: .utf8)
        var loaded: [String: Int32] = [:]

        for (index, line) in contents.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
            loaded[String(line)] = Int32(index)
        }

        guard !loaded.isEmpty else {
            throw RAGError.invalidData("Vocabulary file is empty")
        }

        self.vocab = loaded
    }

    func encode(text: String, maxLength: Int) -> EncodedInput {
        let normalized = text.lowercased()
        let basic = basicTokenize(normalized)

        var wordPieces: [String] = []
        wordPieces.reserveCapacity(maxLength)

        for token in basic {
            wordPieces.append(contentsOf: wordPieceTokenize(token))
        }

        var tokens = [clsToken] + wordPieces + [sepToken]
        if tokens.count > maxLength {
            tokens = Array(tokens.prefix(maxLength))
            if tokens.last != sepToken {
                tokens[maxLength - 1] = sepToken
            }
        }

        var inputIDs = tokens.map { tokenID($0) }
        var attentionMask = Array(repeating: Int32(1), count: inputIDs.count)
        var tokenTypeIDs = Array(repeating: Int32(0), count: inputIDs.count)

        if inputIDs.count < maxLength {
            let padCount = maxLength - inputIDs.count
            inputIDs += Array(repeating: tokenID(padToken), count: padCount)
            attentionMask += Array(repeating: 0, count: padCount)
            tokenTypeIDs += Array(repeating: 0, count: padCount)
        }

        return EncodedInput(
            inputIDs: inputIDs,
            attentionMask: attentionMask,
            tokenTypeIDs: tokenTypeIDs
        )
    }

    private func tokenID(_ token: String) -> Int32 {
        vocab[token] ?? vocab[unkToken] ?? 100
    }

    private func basicTokenize(_ text: String) -> [String] {
        var tokens: [String] = []
        var current = ""

        for scalar in text.unicodeScalars {
            if CharacterSet.whitespacesAndNewlines.contains(scalar) {
                if !current.isEmpty {
                    tokens.append(current)
                    current.removeAll(keepingCapacity: true)
                }
                continue
            }

            if CharacterSet.alphanumerics.contains(scalar) {
                current.unicodeScalars.append(scalar)
            } else {
                if !current.isEmpty {
                    tokens.append(current)
                    current.removeAll(keepingCapacity: true)
                }
                tokens.append(String(scalar))
            }
        }

        if !current.isEmpty {
            tokens.append(current)
        }

        return tokens
    }

    private func wordPieceTokenize(_ token: String) -> [String] {
        if vocab[token] != nil {
            return [token]
        }

        let chars = Array(token)
        guard chars.count <= 100 else {
            return [unkToken]
        }

        var output: [String] = []
        var start = 0

        while start < chars.count {
            var end = chars.count
            var currentPiece: String?

            while start < end {
                let substring = String(chars[start..<end])
                let candidate = start == 0 ? substring : "##\(substring)"
                if vocab[candidate] != nil {
                    currentPiece = candidate
                    break
                }
                end -= 1
            }

            guard let piece = currentPiece else {
                return [unkToken]
            }

            output.append(piece)
            start = end
        }

        return output
    }
}
