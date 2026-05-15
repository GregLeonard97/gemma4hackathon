import Foundation
import Observation
import UIKit
import os

@MainActor
@Observable
final class AppState {
    enum LaunchState: Sendable {
        case initialising
        case needsModelDownload(progress: Double)
        case loadingModel
        case ready(RAGPipeline)
        case failed(String)
    }

    var launchState: LaunchState = .initialising

    private let logger = Logger(subsystem: "GuidelineAssistant", category: "AppState")
    private let modelStorage: ModelStorage
    private let llm: LLMEngine
    private var memoryWarningObserverTask: Task<Void, Never>?
    private var memoryPressureSource: DispatchSourceMemoryPressure?
    private var activeDownloadID: UUID?

    init(
        modelStorage: ModelStorage = .shared,
        llm: LLMEngine? = nil
    ) {
        self.modelStorage = modelStorage
        self.llm = llm ?? LLMEngine(modelStorage: modelStorage)
    }

    func bootstrap() async {
        activeDownloadID = nil
        launchState = .initialising

        observeMemoryPressureIfNeeded()
        MemoryDiagnostics.log(stage: "bootstrap-start", logger: logger)

        let storageReport = self.modelStorage.reportModelStorage()
        logger.info("\(storageReport, privacy: .public)")
        DebugLogStore.shared.log(level: "INFO", category: "Bootstrap", message: storageReport)

        let detailedStorageReport = self.modelStorage.reportModelStorageDetailed()
        logger.info("\(detailedStorageReport, privacy: .public)")
        DebugLogStore.shared.log(level: "INFO", category: "ModelInventory", message: detailedStorageReport)

        do {
            if !self.modelStorage.isModelDownloaded {
                DebugLogStore.shared.log(level: "INFO", category: "Bootstrap", message: "Model not downloaded; prompting download")
                launchState = .needsModelDownload(progress: 0)
                return
            }

            let embeddingEngine = try EmbeddingEngine()
            MemoryDiagnostics.log(stage: "bootstrap-after-embedding-init", logger: logger)

            guard let codebookURL = Bundle.main.url(forResource: "codebook", withExtension: "json") else {
                throw RAGError.resourceMissing("codebook.json")
            }

            let turboQuant = try TurboQuant.load(from: codebookURL)
            let vectorStore = try VectorStore(turboQuant: turboQuant)
            MemoryDiagnostics.log(stage: "bootstrap-after-vector-store-init", logger: logger)

            launchState = .loadingModel
            try await llm.loadModel()
            MemoryDiagnostics.log(stage: "bootstrap-after-model-load", logger: logger)

            let pipeline = RAGPipeline(
                llm: llm,
                embedder: embeddingEngine,
                vectorStore: vectorStore
            )

            observeMemoryWarningsIfNeeded()
            launchState = .ready(pipeline)
            logger.info("App bootstrap complete")
            DebugLogStore.shared.log(level: "INFO", category: "Bootstrap", message: "App bootstrap complete")
            MemoryDiagnostics.log(stage: "bootstrap-ready", logger: logger)
        } catch {
            logger.error("Bootstrap failed: \(error.localizedDescription)")
            DebugLogStore.shared.log(level: "ERROR", category: "Bootstrap", message: "Bootstrap failed: \(error.localizedDescription)")
            MemoryDiagnostics.log(stage: "bootstrap-failed", logger: logger)
            launchState = .failed(error.localizedDescription)
        }
    }

    func downloadModel(progress: @escaping (Double) -> Void) async {
        let downloadID = UUID()
        activeDownloadID = downloadID
        logger.info("User initiated model download")
        DebugLogStore.shared.log(level: "INFO", category: "ModelDownload", message: "User initiated model download")

        do {
            try await modelStorage.downloadModel { [weak self] value in
                progress(value)
                Task { @MainActor in
                    guard let self, self.activeDownloadID == downloadID else { return }
                    self.launchState = .needsModelDownload(progress: value)
                    self.logger.debug("Model download progress: \(Int(value * 100), privacy: .public)%")
                }
            }

            activeDownloadID = nil
            logger.info("Model download finished, restarting bootstrap")
            DebugLogStore.shared.log(level: "INFO", category: "ModelDownload", message: "Model download finished; restarting bootstrap")
            await bootstrap()
        } catch {
            activeDownloadID = nil
            logger.error("Model download failed: \(error.localizedDescription, privacy: .public)")
            DebugLogStore.shared.log(level: "ERROR", category: "ModelDownload", message: "Model download failed: \(error.localizedDescription)")
            launchState = .failed(error.localizedDescription)
        }
    }

    func handleMemoryWarning() async {
        let message = "UIKit memory warning received - \(MemoryDiagnostics.report())"
        logger.warning("\(message, privacy: .public)")
        DebugLogStore.shared.log(level: "WARN", category: "Memory", message: message)
        await llm.handleMemoryWarning()
    }

    func handleCriticalMemoryPressure() async {
        let startMessage = "Critical memory pressure detected; attempting model unload"
        logger.warning("\(startMessage, privacy: .public)")
        DebugLogStore.shared.log(level: "WARN", category: "Memory", message: startMessage)

        guard case .ready = launchState else {
            let skipped = "Critical memory pressure handler skipped unload because pipeline is not ready"
            logger.warning("\(skipped, privacy: .public)")
            DebugLogStore.shared.log(level: "WARN", category: "Memory", message: skipped)
            return
        }

        await llm.unload()
        launchState = .failed("Critical memory pressure: model unloaded. Restart the app to retry.")
        MemoryDiagnostics.log(stage: "after-critical-memory-unload", logger: logger)
    }

    private func observeMemoryPressureIfNeeded() {
        guard memoryPressureSource == nil else { return }

        let source = DispatchSource.makeMemoryPressureSource(
            eventMask: [.warning, .critical],
            queue: .main
        )

        source.setEventHandler { [weak self] in
            Task { @MainActor in
                guard let self else { return }

                let event = self.memoryPressureSource?.data ?? []
                let label: String
                if event.contains(.critical) {
                    label = "CRITICAL"
                } else if event.contains(.warning) {
                    label = "WARNING"
                } else {
                    label = "UNKNOWN"
                }

                let message = "Memory pressure: \(label) - \(MemoryDiagnostics.report())"
                self.logger.warning("\(message, privacy: .public)")
                DebugLogStore.shared.log(level: "WARN", category: "Memory", message: message)

                if event.contains(.critical) {
                    await self.handleCriticalMemoryPressure()
                }
            }
        }

        source.activate()
        memoryPressureSource = source

        logger.info("Memory pressure source activated")
        DebugLogStore.shared.log(level: "INFO", category: "Memory", message: "Memory pressure source activated")
    }

    private func observeMemoryWarningsIfNeeded() {
        guard memoryWarningObserverTask == nil else { return }

        logger.info("UIKit memory warning observer activated")
        DebugLogStore.shared.log(level: "INFO", category: "Memory", message: "UIKit memory warning observer activated")

        memoryWarningObserverTask = Task { [weak self] in
            guard let self else { return }
            for await _ in NotificationCenter.default.notifications(
                named: UIApplication.didReceiveMemoryWarningNotification
            ) {
                await self.handleMemoryWarning()
            }
        }
    }
}
