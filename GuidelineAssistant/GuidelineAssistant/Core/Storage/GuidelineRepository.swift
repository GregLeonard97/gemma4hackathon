import Foundation

final class GuidelineRepository: @unchecked Sendable {
    private let bundle: Bundle

    init(bundle: Bundle = .main) {
        self.bundle = bundle
    }

    func sources(from chunks: [RetrievedChunk], includeTableImage: Bool) -> [Source] {
        chunks.map { chunk in
            Source(
                guideline: chunk.source,
                page: chunk.page,
                chunkType: chunk.chunkType,
                excerpt: chunk.content,
                section: chunk.section,
                tableImageFilename: includeTableImage ? chunk.tableImage : nil,
                imageRefs: chunk.imageRefs
            )
        }
    }

    func tableImageURL(filename: String) -> URL? {
        resourceURL(path: "extracted_tables", file: filename)
    }

    func guidelinePDFURL(filename: String) -> URL? {
        resourceURL(path: "Guidelines", file: filename)
    }

    private func resourceURL(path: String, file: String) -> URL? {
        guard let base = bundle.resourceURL else { return nil }
        return base
            .appendingPathComponent(path, isDirectory: true)
            .appendingPathComponent(file)
    }

    static func parseImageRefs(_ json: String?) -> [String] {
        guard let json,
              let data = json.data(using: .utf8),
              let values = try? JSONDecoder().decode([String].self, from: data) else {
            return []
        }
        return values
    }
}
