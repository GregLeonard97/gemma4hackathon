import Foundation
import Observation

@MainActor
@Observable
final class QueryViewModel {
    enum Phase: Equatable {
        case loading
        case answered
        case failed(String)
    }

    let question: String

    var phase: Phase = .loading
    var answerText: String = ""
    var sources: [Source] = []
    var expandedSourceID: UUID?
    var tableImageFilename: String?

    private var streamTask: Task<Void, Never>?

    init(question: String) {
        self.question = question
    }

    func start(pipeline: RAGPipeline) {
        guard streamTask == nil else { return }

        phase = .loading
        answerText = ""
        sources = []
        expandedSourceID = nil
        tableImageFilename = nil

        streamTask = Task { [weak self] in
            guard let self else { return }

            do {
                for try await event in pipeline.query(question) {
                    handle(event)
                }
            } catch {
                phase = .failed(error.localizedDescription)
            }
        }
    }

    func stop() {
        streamTask?.cancel()
        streamTask = nil
    }

    var answerBlocks: [AnswerBlock] {
        Self.parseBlocks(from: answerText)
    }

    var tableImageFilenames: [String] {
        let candidates = sources.compactMap(\.tableImageFilename) + [tableImageFilename].compactMap { $0 }
        var seen = Set<String>()
        var unique: [String] = []

        for candidate in candidates {
            let trimmed = candidate.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty, seen.insert(trimmed).inserted else { continue }
            unique.append(trimmed)
        }

        return unique
    }

    private func handle(_ event: QueryEvent) {
        switch event {
        case .retrievalStarted:
            phase = .loading

        case .retrievalComplete(let retrievedSources, let tableImage):
            sources = retrievedSources
            tableImageFilename = tableImage

        case .generationStarted:
            phase = .loading

        case .token(let token):
            answerText += token

        case .complete:
            phase = .answered
            streamTask = nil
        }
    }

    private static func parseBlocks(from text: String) -> [AnswerBlock] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }

        var blocks: [AnswerBlock] = []

        let paragraphs = trimmed
            .components(separatedBy: "\n\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        for paragraph in paragraphs {
            let lower = paragraph.lowercased()
            if lower.hasPrefix("tip:") {
                let note = paragraph.dropFirst(4).trimmingCharacters(in: .whitespacesAndNewlines)
                blocks.append(.note(note))
                continue
            }

            if lower.hasPrefix("note:") {
                let note = paragraph.dropFirst(5).trimmingCharacters(in: .whitespacesAndNewlines)
                blocks.append(.note(note))
                continue
            }

            let lines = paragraph
                .components(separatedBy: "\n")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }

            let bulletPrefixSet = ["- ", "* ", "• "]
            let bulletLines = lines.filter { line in
                bulletPrefixSet.contains { line.hasPrefix($0) }
            }

            if !bulletLines.isEmpty {
                let heading: String?
                if let first = lines.first,
                   !bulletPrefixSet.contains(where: { first.hasPrefix($0) }) {
                    heading = first
                } else {
                    heading = nil
                }

                let items = lines
                    .filter { line in
                        bulletPrefixSet.contains { line.hasPrefix($0) }
                    }
                    .map { line in
                        var output = line
                        for prefix in bulletPrefixSet where output.hasPrefix(prefix) {
                            output.removeFirst(prefix.count)
                            break
                        }
                        return output
                    }

                if !items.isEmpty {
                    blocks.append(.list(heading: heading, items: items))
                    continue
                }
            }

            blocks.append(.paragraph(paragraph))
        }

        return blocks
    }
}
