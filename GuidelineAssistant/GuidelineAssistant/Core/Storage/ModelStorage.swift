import Foundation
import Hub
import Observation
import os

@Observable
final class ModelStorage {
    static let shared = ModelStorage()

    private let modelRepoID = "mlx-community/gemma-4-e2b-it-4bit"
    private let logger = Logger(subsystem: "GuidelineAssistant", category: "ModelStorage")
    private let requiredModelFiles: [String] = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "processor_config.json",
        "model.safetensors.index.json"
    ]
    private let minimumSafetensorBytes: Int64 = 3_000_000_000

    var isModelDownloaded: Bool {
        guard let modelPath else {
            return false
        }

        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: modelPath.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            return false
        }

        for filename in requiredModelFiles {
            let fileURL = modelPath.appendingPathComponent(filename, isDirectory: false)
            guard FileManager.default.fileExists(atPath: fileURL.path) else {
                return false
            }
            guard fileSize(at: fileURL) > 0 else {
                return false
            }
        }

        let contents: [URL]
        do {
            contents = try FileManager.default.contentsOfDirectory(
                at: modelPath,
                includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
                options: []
            )
        } catch {
            return false
        }

        let safetensorFiles = contents.filter { $0.pathExtension == "safetensors" }
        guard !safetensorFiles.isEmpty else {
            return false
        }

        let totalSafetensorBytes = safetensorFiles.reduce(Int64(0)) { partial, fileURL in
            partial + fileSize(at: fileURL)
        }

        return totalSafetensorBytes >= minimumSafetensorBytes
    }

    var modelPath: URL? {
        guard let base = applicationSupportDirectory else {
            return nil
        }

        return repoPath(base: base.appendingPathComponent("models", isDirectory: true))
    }

    private var applicationSupportDirectory: URL? {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
    }

    private func repoPath(base: URL) -> URL {
        modelRepoID.split(separator: "/").reduce(base) { partial, component in
            partial.appendingPathComponent(String(component), isDirectory: true)
        }
    }

    func reportModelStorage() -> String {
        guard let modelPath else {
            return "Model storage: no model path configured"
        }

        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: modelPath.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            return "Model storage: model directory missing at \(modelPath.path)"
        }

        let stats = directoryStats(at: modelPath)
        let sizeGB = Double(stats.totalBytes) / 1_000_000_000
        let integrity = isModelDownloaded ? "integrity=ok" : "integrity=incomplete"

        return "Model storage: path=\(modelPath.path), files=\(stats.fileCount), sizeGB=\(String(format: "%.2f", sizeGB)), \(integrity)"
    }

    func reportModelStorageDetailed() -> String {
        guard let path = modelPath else {
            return "No model path configured"
        }

        guard FileManager.default.fileExists(atPath: path.path) else {
            return "Model not downloaded"
        }

        var lines: [String] = []
        lines.append("Model directory: \(path.path)")

        let contents = (try? FileManager.default.contentsOfDirectory(
            at: path,
            includingPropertiesForKeys: [.fileSizeKey],
            options: []
        )) ?? []

        let sortedFiles = contents.sorted { lhs, rhs in
            let s1 = (try? lhs.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            let s2 = (try? rhs.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            return s1 > s2
        }

        var totalBytes: Int64 = 0
        for file in sortedFiles {
            let size = (try? file.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
            let sizeMB = Double(size) / 1_000_000
            let sizeStr = sizeMB >= 100
                ? String(format: "%.0f MB", sizeMB)
                : String(format: "%.1f MB", sizeMB)
            lines.append("  \(file.lastPathComponent): \(sizeStr)")
            totalBytes += Int64(size)
        }

        let totalGB = Double(totalBytes) / 1_000_000_000
        lines.append("Total: \(String(format: "%.2f GB", totalGB)) across \(contents.count) files")

        return lines.joined(separator: "\n")
    }

    func downloadModel(progress: @escaping (Double) -> Void) async throws {
        if isModelDownloaded {
            await MainActor.run {
                progress(1.0)
            }
            logger.info("Model already present: \(self.modelRepoID)")
            return
        }

        guard let base = applicationSupportDirectory else {
            throw RAGError.invalidData("Application Support directory unavailable")
        }

        let hub = HubApi(downloadBase: base)
        let repo = Hub.Repo(id: modelRepoID)

        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        logger.info("Starting model download for \(self.modelRepoID, privacy: .public) into \(base.path, privacy: .public)")

        await MainActor.run {
            progress(0.0)
        }

        let snapshotURL = try await hub.snapshot(
            from: repo,
            matching: ["*.safetensors", "*.json", "*.model", "*.txt", "*.jinja"],
            progressHandler: { snapshotProgress in
                Task { @MainActor in
                    progress(snapshotProgress.fractionCompleted)
                }
            }
        )

        logger.info("Model snapshot downloaded to \(snapshotURL.path, privacy: .public)")
        logger.info("\(self.reportModelStorage(), privacy: .public)")

        await MainActor.run {
            progress(1.0)
        }

        guard isModelDownloaded else {
            logger.error("Model download completed but validation failed. Expected model path: \(self.modelPath?.path ?? "nil", privacy: .public)")
            throw RAGError.modelNotDownloaded
        }

        logger.info("Model downloaded: \(self.modelRepoID)")
    }

    func deleteModel() throws {
        guard let modelPath else { return }
        guard FileManager.default.fileExists(atPath: modelPath.path) else { return }
        try FileManager.default.removeItem(at: modelPath)
        logger.info("Deleted model: \(self.modelRepoID)")
    }

    private func fileSize(at url: URL) -> Int64 {
        guard
            let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
            values.isRegularFile == true
        else {
            return 0
        }
        return Int64(values.fileSize ?? 0)
    }

    private func directoryStats(at directory: URL) -> (fileCount: Int, totalBytes: Int64) {
        guard let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
            options: []
        ) else {
            return (0, 0)
        }

        var fileCount = 0
        var totalBytes: Int64 = 0

        for case let fileURL as URL in enumerator {
            guard
                let values = try? fileURL.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
                values.isRegularFile == true
            else {
                continue
            }

            fileCount += 1
            totalBytes += Int64(values.fileSize ?? 0)
        }

        return (fileCount, totalBytes)
    }
}
