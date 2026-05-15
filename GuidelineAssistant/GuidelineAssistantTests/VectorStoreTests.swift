import Foundation
import SQLite
import XCTest
@testable import GuidelineAssistant

final class VectorStoreTests: XCTestCase {
    func testSearchFindsAntibioticsGuideline() throws {
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("guidelines-\(UUID().uuidString).db")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let dimension = 384
        let turboQuant = TurboQuant(codebook: makeTestCodebook(dimension: dimension, bits: 3))
        try seedDatabase(at: tempURL, turboQuant: turboQuant, dimension: dimension)

        let store = try VectorStore(turboQuant: turboQuant, databaseURL: tempURL)

        var queryEmbedding = Array(repeating: Float(0), count: dimension)
        queryEmbedding[0] = 1

        let results = try store.search(queryEmbedding: queryEmbedding, topK: 1)

        XCTAssertEqual(try store.chunkCount(), 2)
        XCTAssertEqual(try store.sourceList().count, 2)
        XCTAssertEqual(results.count, 1)
        XCTAssertTrue(results[0].source.localizedCaseInsensitiveContains("antibiotics"))
    }

    private func seedDatabase(at url: URL, turboQuant: TurboQuant, dimension: Int) throws {
        let db = try Connection(url.path)

        let guidelineChunks = Table("guideline_chunks")
        let compressedVectors = Table("compressed_vectors")

        let chunkID = Expression<Int64>("id")
        let source = Expression<String>("source")
        let page = Expression<Int64>("page")
        let section = Expression<String?>("section")
        let chunkType = Expression<String>("chunk_type")
        let content = Expression<String>("content")
        let tableImage = Expression<String?>("table_image")
        let imageRefs = Expression<String?>("image_refs")
        let chunkIndex = Expression<Int64>("chunk_index")

        let vectorChunkID = Expression<Int64>("chunk_id")
        let indices = Expression<Blob>("indices")
        let norm = Expression<Double>("norm")

        try db.run(guidelineChunks.create { table in
            table.column(chunkID, primaryKey: true)
            table.column(source)
            table.column(page)
            table.column(section)
            table.column(chunkType)
            table.column(content)
            table.column(tableImage)
            table.column(imageRefs)
            table.column(chunkIndex)
        })

        try db.run(compressedVectors.create { table in
            table.column(vectorChunkID, primaryKey: true)
            table.column(indices)
            table.column(norm)
            table.foreignKey(vectorChunkID, references: guidelineChunks, chunkID)
        })

        var antibioticsVector = Array(repeating: Float(0), count: dimension)
        antibioticsVector[0] = 1

        var fluidsVector = Array(repeating: Float(0), count: dimension)
        fluidsVector[1] = 1

        let antibioticsCompressed = turboQuant.compress(antibioticsVector)
        let fluidsCompressed = turboQuant.compress(fluidsVector)

        try db.run(guidelineChunks.insert(
            chunkID <- 1,
            source <- "Antibiotics_Guideline.pdf",
            page <- 4,
            section <- "Gentamicin",
            chunkType <- "table_row",
            content <- "Gentamicin dosing by gestation",
            tableImage <- "gentamicin_table.png",
            imageRefs <- "[]",
            chunkIndex <- 0
        ))

        try db.run(guidelineChunks.insert(
            chunkID <- 2,
            source <- "Fluids_Guideline.pdf",
            page <- 2,
            section <- "Maintenance",
            chunkType <- "text",
            content <- "Maintenance fluid rates by age",
            tableImage <- nil,
            imageRefs <- "[]",
            chunkIndex <- 1
        ))

        try db.run(compressedVectors.insert(
            vectorChunkID <- 1,
            indices <- Blob(bytes: antibioticsCompressed.indices),
            norm <- Double(antibioticsCompressed.norm)
        ))

        try db.run(compressedVectors.insert(
            vectorChunkID <- 2,
            indices <- Blob(bytes: fluidsCompressed.indices),
            norm <- Double(fluidsCompressed.norm)
        ))
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
            rotationMatrix[i * dimension + i] = 1
        }

        return TurboQuantCodebook(
            centroids: centroids,
            bitsPerCoordinate: bits,
            rotationMatrix: rotationMatrix,
            dimension: dimension
        )
    }
}
