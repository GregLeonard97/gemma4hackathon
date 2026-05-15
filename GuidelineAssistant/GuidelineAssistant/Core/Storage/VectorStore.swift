import Accelerate
import Foundation
import SQLite
import os

final class VectorStore: VectorStoreProtocol, @unchecked Sendable {
    private static let DB_VERSION: Int = 1
    
    private let turboQuant: TurboQuant
    private let db: Connection
    private let logger = Logger(subsystem: "GuidelineAssistant", category: "VectorStore")

    private let guidelineChunks = Table("guideline_chunks")
    private let compressedVectors = Table("compressed_vectors")
    private let metadata = Table("metadata")

    private let chunkID = Expression<Int64>("id")
    private let source = Expression<String>("source")
    private let page = Expression<Int64>("page")
    private let section = Expression<String?>("section")
    private let chunkType = Expression<String>("chunk_type")
    private let content = Expression<String>("content")
    private let tableImage = Expression<String?>("table_image")
    private let imageRefs = Expression<String?>("image_refs")

    private let vectorChunkID = Expression<Int64>("chunk_id")
    private let indicesBlob = Expression<Blob>("indices")
    private let norm = Expression<Double>("norm")

    private let metadataKey = Expression<String>("key")
    private let metadataValue = Expression<String>("value")

    convenience init(
        turboQuant: TurboQuant,
        bundle: Bundle = .main,
        fileManager: FileManager = .default
    ) throws {
        let url = try Self.prepareDatabase(bundle: bundle, fileManager: fileManager)
        try self.init(turboQuant: turboQuant, databaseURL: url)
    }

    init(turboQuant: TurboQuant, databaseURL: URL) throws {
        self.turboQuant = turboQuant
        self.db = try Connection(databaseURL.path, readonly: true)
        self.db.busyTimeout = 3.0
    }

    func search(queryEmbedding: [Float], topK: Int) throws -> [RetrievedChunk] {
        guard topK > 0 else { return [] }
        guard queryEmbedding.count == turboQuant.codebook.dimension else {
            throw RAGError.retrievalFailed("Query embedding dimension does not match codebook dimension")
        }

        let queryNorm = l2Norm(queryEmbedding)
        guard queryNorm > 0 else { return [] }
        let queryUnit = queryEmbedding.map { $0 / queryNorm }

        let query = guidelineChunks
            .join(compressedVectors, on: guidelineChunks[chunkID] == compressedVectors[vectorChunkID])

        var retrieved: [RetrievedChunk] = []
        retrieved.reserveCapacity(2048)

        for row in try db.prepare(query) {
            let chunkIDValue = Int(row[chunkID])
            let compressed = CompressedVector(
                indices: decodeIndices(row[indicesBlob]),
                norm: Float(row[norm])
            )
            let similarity = turboQuant.cosineSimilarity(query: queryUnit, compressed: compressed)

            retrieved.append(
                RetrievedChunk(
                    id: chunkIDValue,
                    source: row[source],
                    page: Int(row[page]),
                    section: row[section],
                    chunkType: Source.ChunkType(dbValue: row[chunkType]),
                    content: row[content],
                    tableImage: row[tableImage],
                    imageRefs: GuidelineRepository.parseImageRefs(row[imageRefs]),
                    similarity: similarity
                )
            )
        }

        return retrieved
            .sorted { $0.similarity > $1.similarity }
            .prefix(topK)
            .map { $0 }
    }

    func chunkCount() throws -> Int {
        try db.scalar(guidelineChunks.count)
    }

    func sourceList() throws -> [String] {
        var values: [String] = []
        for row in try db.prepare(guidelineChunks.select(distinct: source)) {
            values.append(row[source])
        }
        return values.sorted()
    }

    private func decodeIndices(_ blob: Blob) -> [UInt8] {
        let bytes = Array(blob.bytes)
        let dimension = turboQuant.codebook.dimension
        let bits = turboQuant.codebook.bitsPerCoordinate

        if bytes.count == dimension {
            return bytes
        }

        let packedBytes = Int(ceil(Double(dimension * bits) / 8.0))
        if bytes.count == packedBytes {
            return unpackBitPackedIndices(bytes: bytes, bitsPerCoordinate: bits, dimension: dimension)
        }

        logger.warning("Unexpected indices blob length: \(bytes.count), expected \(dimension) or \(packedBytes)")

        var padded = Array(repeating: UInt8(0), count: dimension)
        for i in 0..<min(bytes.count, dimension) {
            padded[i] = bytes[i]
        }
        return padded
    }

    private func unpackBitPackedIndices(
        bytes: [UInt8],
        bitsPerCoordinate: Int,
        dimension: Int
    ) -> [UInt8] {
        var result = Array(repeating: UInt8(0), count: dimension)

        for coordinate in 0..<dimension {
            var value = 0
            for bit in 0..<bitsPerCoordinate {
                let bitIndex = coordinate * bitsPerCoordinate + bit
                let byteIndex = bitIndex / 8
                let intraByte = bitIndex % 8
                let bitValue = (bytes[byteIndex] >> UInt8(intraByte)) & 0x01
                value |= Int(bitValue) << bit
            }
            result[coordinate] = UInt8(value)
        }

        return result
    }

    private func l2Norm(_ vector: [Float]) -> Float {
        var sumSquares: Float = 0
        vDSP_svesq(vector, 1, &sumSquares, vDSP_Length(vector.count))
        return sqrtf(sumSquares)
    }

    private static func getDBVersion(at url: URL) -> Int? {
        do {
            let db = try Connection(url.path, readonly: true)
            
            let metadata = Table("metadata")
            let key = Expression<String>("key")
            let value = Expression<String>("value")
            
            if let row = try db.pluck(metadata.where(key == "version")) {
                let versionStr = try row.get(value)
                if let version = Int(versionStr) {
                    return version
                }
            }
        } catch {
            return nil
        }
        return nil
    }

    private static func prepareDatabase(bundle: Bundle, fileManager: FileManager) throws -> URL {
        guard let bundledDB = bundle.url(forResource: "guidelines", withExtension: "db") else {
            throw RAGError.resourceMissing("guidelines.db")
        }

        let supportRoot = try fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )

        let targetDirectory = supportRoot.appendingPathComponent("GuidelineAssistant", isDirectory: true)
        if !fileManager.fileExists(atPath: targetDirectory.path) {
            try fileManager.createDirectory(at: targetDirectory, withIntermediateDirectories: true)
        }

        let targetDB = targetDirectory.appendingPathComponent("guidelines.db")
        
        let bundledVersion = getDBVersion(at: bundledDB) ?? DB_VERSION
        let installedVersion = getDBVersion(at: targetDB)
        
        // If installed DB exists but has mismatched version, overwrite it
        if fileManager.fileExists(atPath: targetDB.path) && installedVersion != bundledVersion {
            try fileManager.removeItem(at: targetDB)
            try fileManager.copyItem(at: bundledDB, to: targetDB)
        } else if !fileManager.fileExists(atPath: targetDB.path) {
            // If installed DB doesn't exist, copy bundled DB
            try fileManager.copyItem(at: bundledDB, to: targetDB)
        }

        return targetDB
    }
}
