import Foundation
import Hub
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM
import Observation
import os
import Tokenizers

@Observable
final class LLMEngine: LLMEngineProtocol, @unchecked Sendable {
    enum State: Equatable, Sendable {
        case notLoaded
        case loading(progress: Double)
        case ready
        case failed(String)
    }

    private(set) var state: State = .notLoaded

    private let modelStorage: ModelStorage
    private let logger = Logger(subsystem: "GuidelineAssistant", category: "LLMEngine")
    private let holder = ModelHolder()

    init(modelStorage: ModelStorage = .shared) {
        self.modelStorage = modelStorage
    }

    func loadModel() async throws {
        if case .ready = state {
            return
        }

        MemoryDiagnostics.log(stage: "before-model-load", logger: logger)

        let cacheLimitBytes = 512 * 1024 * 1024
        MLX.Memory.cacheLimit = cacheLimitBytes
        let cacheMessage = "MLX cache limit set to 512 MB via MLX.Memory.cacheLimit"
        logger.info("\(cacheMessage, privacy: .public)")
        DebugLogStore.shared.log(level: "INFO", category: "MLX", message: cacheMessage)

        guard modelStorage.isModelDownloaded else {
            state = .failed("Model not downloaded")
            throw RAGError.modelNotDownloaded
        }

        guard let modelPath = modelStorage.modelPath else {
            state = .failed("Model path unavailable")
            throw RAGError.modelNotDownloaded
        }

    #if targetEnvironment(simulator)
        let message = "Gemma 4 E2B inference is not supported in iOS Simulator. Run on a physical iPhone or My Mac destination."
        state = .failed(message)
        DebugLogStore.shared.log(level: "ERROR", category: "LLMEngine", message: message)
        throw RAGError.generationFailed(message)
    #endif

        state = .loading(progress: 0.0)

        do {
            let tokenizerLoader = SwiftTransformersTokenizerLoader(hubApi: HubApi())

            let aboutToLoadMessage = "About to call LLMModelFactory.loadContainer"
            logger.info("\(aboutToLoadMessage, privacy: .public)")
            DebugLogStore.shared.log(level: "INFO", category: "ModelLoad", message: aboutToLoadMessage)

            let preLoadReport = MemoryDiagnostics.report()
            let preLoadMessage = "pre-load: \(preLoadReport)"
            logger.info("\(preLoadMessage, privacy: .public)")
            DebugLogStore.shared.log(level: "INFO", category: "ModelLoad", message: preLoadMessage)

            let container = try await LLMModelFactory.shared.loadContainer(
                from: modelPath,
                using: tokenizerLoader
            )

            let returnedMessage = "LLMModelFactory.loadContainer returned"
            logger.info("\(returnedMessage, privacy: .public)")
            DebugLogStore.shared.log(level: "INFO", category: "ModelLoad", message: returnedMessage)

            let postLoadReport = MemoryDiagnostics.report()
            let postLoadMessage = "post-load: \(postLoadReport)"
            logger.info("\(postLoadMessage, privacy: .public)")
            DebugLogStore.shared.log(level: "INFO", category: "ModelLoad", message: postLoadMessage)

            await holder.setContainer(container)
            state = .ready
            logger.info("Gemma 4 E2B MLX model loaded")
            DebugLogStore.shared.log(level: "INFO", category: "LLMEngine", message: "Gemma 4 E2B MLX model loaded")
            MemoryDiagnostics.log(stage: "after-model-load", logger: logger)
        } catch {
            let message = "Failed to load model: \(error.localizedDescription)"
            state = .failed(message)
            DebugLogStore.shared.log(level: "ERROR", category: "LLMEngine", message: message)
            MemoryDiagnostics.log(stage: "model-load-failed", logger: logger)
            throw RAGError.generationFailed(message)
        }
    }

    func unload() async {
        _ = await holder.takeContainer()
        clearMLXCache()
        state = .notLoaded
        logger.info("Model unloaded")
        DebugLogStore.shared.log(level: "INFO", category: "LLMEngine", message: "Model unloaded")
        MemoryDiagnostics.log(stage: "after-model-unload", logger: logger)
    }

    func generate(
        systemPrompt: String,
        userPrompt: String
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task.detached(priority: .userInitiated) { [holder, logger] in
                do {
                    guard let container = await holder.getContainer() else {
                        throw RAGError.modelNotLoaded
                    }

                    MemoryDiagnostics.log(stage: "before-generation", logger: logger)

                    let kvWorkaroundMessage = "Gemma4 crash workaround active: KV cache quantization disabled (kvBits=nil)"
                    logger.info("\(kvWorkaroundMessage, privacy: .public)")
                    DebugLogStore.shared.log(level: "INFO", category: "KVCache", message: kvWorkaroundMessage)

                    let parameters = GenerateParameters(
                        maxTokens: 1024,
                        kvBits: nil,
                        kvGroupSize: 64,
                        quantizedKVStart: 0,
                        temperature: 0.1,
                        topP: 0.95
                    )

                    // Keep conversations stateless: new session for each query.
                    let session = ChatSession(
                        container,
                        instructions: systemPrompt,
                        generateParameters: parameters
                    )

                    logger.info("=== PROMPT DIAGNOSTIC ===")
                    logger.info("System prompt chars: \(systemPrompt.count)")
                    logger.info("User prompt chars: \(userPrompt.count)")
                    logger.info("Total assembled prompt chars: \(systemPrompt.count + userPrompt.count)")
                    logger.info("First 200 chars of user prompt: \(String(userPrompt.prefix(200)))")
                    logger.info("Last 200 chars of user prompt: \(String(userPrompt.suffix(200)))")
                    DebugLogStore.shared.log(level: "INFO", category: "PromptDiag", message: "system=\(systemPrompt.count) chars, user=\(userPrompt.count) chars")
                    do {
                        let tokenizer = await container.tokenizer
                        let messages: [[String: any Sendable]] = [
                            ["role": "system", "content": systemPrompt],
                            ["role": "user", "content": userPrompt],
                        ]
                        let promptTokens = try tokenizer.applyChatTemplate(messages: messages)
                        logger.info("Tokenizer prompt token count: \(promptTokens.count)")
                        DebugLogStore.shared.log(level: "INFO", category: "PromptDiag", message: "prompt_tokens=\(promptTokens.count)")
                    } catch {
                        let message = "Tokenizer prompt token count unavailable: \(error.localizedDescription)"
                        logger.info("\(message, privacy: .public)")
                        DebugLogStore.shared.log(level: "INFO", category: "PromptDiag", message: message)
                    }
                    logger.info("=== END PROMPT DIAGNOSTIC ===")

                    let preGenTime = Date()
                    var tokenCount = 0
                    var lastMemLog = Date()

                    let stream = session.streamResponse(to: userPrompt)
                    for try await token in stream {
                        tokenCount += 1

                        // Log memory at first token, then every 5 seconds.
                        if tokenCount == 1 {
                            let ttft = Date().timeIntervalSince(preGenTime)
                            MemoryDiagnostics.log(stage: "first-token-after-\(Int(ttft * 1000))ms", logger: logger)
                            lastMemLog = Date()
                        } else if Date().timeIntervalSince(lastMemLog) > 5.0 {
                            MemoryDiagnostics.log(stage: "during-gen-\(tokenCount)tokens", logger: logger)
                            lastMemLog = Date()
                        }

                        if !token.isEmpty {
                            continuation.yield(token)
                        }
                    }

                    MemoryDiagnostics.log(stage: "post-gen-\(tokenCount)tokens", logger: logger)
                    continuation.finish()
                } catch {
                    let failureMessage = "Generation failed: \(error.localizedDescription)"
                    logger.error("\(failureMessage, privacy: .public)")
                    DebugLogStore.shared.log(level: "ERROR", category: "LLMEngine", message: failureMessage)
                    MemoryDiagnostics.log(stage: "generation-failed", logger: logger)
                    continuation.finish(throwing: RAGError.generationFailed(error.localizedDescription))
                }
            }
        }
    }

    func handleMemoryWarning() async {
        clearMLXCache()

        switch ProcessInfo.processInfo.thermalState {
        case .critical:
            let message = "Critical thermal state detected; unloading model"
            logger.warning("\(message, privacy: .public)")
            DebugLogStore.shared.log(level: "WARN", category: "Memory", message: message)
            await unload()
        default:
            let message = "Memory warning received; cleared MLX cache"
            logger.info("\(message, privacy: .public)")
            DebugLogStore.shared.log(level: "INFO", category: "Memory", message: message)
        }
    }

    private func clearMLXCache() {
        MLX.GPU.clearCache()
    }
}

private actor ModelHolder {
    private var container: ModelContainer?

    func getContainer() -> ModelContainer? {
        container
    }

    func setContainer(_ container: ModelContainer) {
        self.container = container
    }

    func takeContainer() -> ModelContainer? {
        defer { container = nil }
        return container
    }
}

private struct SwiftTransformersTokenizerLoader: TokenizerLoader {
    let hubApi: HubApi

    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let upstream = try await AutoTokenizer.from(modelFolder: directory, hubApi: hubApi)
        return SwiftTransformersTokenizerBridge(upstream: upstream)
    }
}

private struct SwiftTransformersTokenizerBridge: MLXLMCommon.Tokenizer {
    let upstream: any Tokenizers.Tokenizer

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? {
        upstream.convertTokenToId(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        upstream.convertIdToToken(id)
    }

    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        try upstream.applyChatTemplate(
            messages: messages,
            tools: tools,
            additionalContext: additionalContext
        )
    }
}
