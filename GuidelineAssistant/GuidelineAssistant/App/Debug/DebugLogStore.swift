import Foundation
import Observation

@Observable
final class DebugLogStore {
    static let shared = DebugLogStore()

    struct LogEntry: Identifiable, Sendable {
        let id = UUID()
        let timestamp: Date
        let level: String
        let category: String
        let message: String
    }

    private(set) var entries: [LogEntry] = []

    private let maxEntries = 1_000
    private let queue = DispatchQueue(label: "DebugLogStore", qos: .utility)

    private let logFileURL: URL = {
        let docs = FileManager.default
            .urls(for: .documentDirectory, in: .userDomainMask)
            .first!
        return docs.appendingPathComponent("debug_log.txt")
    }()

    private let timestampFormatter: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()

    private init() {
        loadFromDisk()
    }

    func log(level: String, category: String, message: String) {
        let entry = LogEntry(
            timestamp: Date(),
            level: level,
            category: category,
            message: message
        )

        // Keep writes synchronous so the latest entry is likely to survive a sudden jetsam kill.
        let line = "[\(timestampFormatter.string(from: entry.timestamp))] [\(level)] [\(category)] \(message)\n"
        if let data = line.data(using: .utf8) {
            queue.sync {
                if FileManager.default.fileExists(atPath: self.logFileURL.path),
                   let handle = try? FileHandle(forWritingTo: self.logFileURL) {
                    defer { try? handle.close() }
                    try? handle.seekToEnd()
                    try? handle.write(contentsOf: data)
                } else {
                    try? data.write(to: self.logFileURL)
                }
            }
        }

        Task { @MainActor in
            self.entries.append(entry)
            if self.entries.count > self.maxEntries {
                self.entries.removeFirst(self.entries.count - self.maxEntries)
            }
        }
    }

    func clear() {
        queue.sync {
            try? FileManager.default.removeItem(at: self.logFileURL)
        }

        Task { @MainActor in
            self.entries.removeAll()
        }
    }

    func exportAsText() -> String {
        guard let data = try? Data(contentsOf: logFileURL),
              let text = String(data: data, encoding: .utf8) else {
            return entries.map { entry in
                "[\(timestampFormatter.string(from: entry.timestamp))] [\(entry.level)] [\(entry.category)] \(entry.message)"
            }.joined(separator: "\n")
        }

        return text
    }

    private func loadFromDisk() {
        guard let data = try? Data(contentsOf: logFileURL),
              let text = String(data: data, encoding: .utf8) else {
            return
        }

        let lines = text.split(separator: "\n", omittingEmptySubsequences: true)
        let recentLines = lines.suffix(maxEntries)

        var loaded: [LogEntry] = []
        for line in recentLines {
            // Expected format: [timestamp] [LEVEL] [Category] message
            let lineString = String(line)
            guard lineString.hasPrefix("[") else { continue }

            let parts = lineString.components(separatedBy: "] ")
            guard parts.count >= 4 else { continue }

            let timestampPart = String(parts[0].dropFirst())
            let levelPart = String(parts[1].dropFirst())
            let categoryPart = String(parts[2].dropFirst())
            let messagePart = parts[3...].joined(separator: "] ")

            let timestamp = timestampFormatter.date(from: timestampPart) ?? Date()
            loaded.append(
                LogEntry(
                    timestamp: timestamp,
                    level: levelPart,
                    category: categoryPart,
                    message: messagePart
                )
            )
        }

        let captured = loaded
        Task { @MainActor in
            self.entries = captured
        }
    }
}
