import Foundation
import llama

/// 进程内 GGUF 主/辅 LLM。权重与 KV cache 常驻，不启动任何子进程。
public final class LlamaCppLLMService: LLMService, PreloadableLLMService,
                                       WarmupLLMService, ClosableLLMService,
                                       KVCacheInvalidating, @unchecked Sendable {
    public let identifier: String
    public let modelPath: String
    public let contextWindow: Int
    public let supportsTools = true

    private let maxTokens: Int
    private let topP: Double
    private let minP: Double
    private let topK: Int
    private let repetitionPenalty: Double
    private let enableThinking: Bool
    private let gpuLayers: Int
    /// 跨轮 KV/前缀复用开关（llm_kv_reuse）。关闭时每轮全量重建，
    /// 与 MLX 后端对齐。
    private let kvReuse: Bool
    /// KV cache int8 量化（ggml q8_0），KV 内存约减半。
    private let kvInt8: Bool

    private let gate = AsyncGate()
    private let lock = NSLock()
    private var loaded: Loaded?
    private var loadFailure: Error?
    private var cachedTokens: [llama_token] = []
    private var cachedToolsKey: String?

    init(modelPath: String,
         contextWindow: Int,
         maxTokens: Int,
         topP: Double,
         minP: Double,
         topK: Int,
         repetitionPenalty: Double,
         enableThinking: Bool,
         gpuLayers: Int,
         kvReuse: Bool = true,
         kvInt8: Bool = true) {
        self.modelPath = modelPath
        self.identifier = "llama.cpp:\((modelPath as NSString).lastPathComponent)"
        self.contextWindow = max(1_024, contextWindow)
        self.maxTokens = max(1, maxTokens)
        self.topP = topP
        self.minP = minP
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
        self.enableThinking = enableThinking
        self.gpuLayers = gpuLayers
        self.kvReuse = kvReuse
        self.kvInt8 = kvInt8
    }

    // MARK: - LLMService

    public func chatStream(messages: [ChatMessage],
                           tools: [JSONValue],
                           temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.run(messages: messages, tools: tools,
                                       temperature: temperature,
                                       continuation: continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: Self.mapError(error, modelPath: modelPath))
                }
            }
            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    // MARK: - 加载

    public func preload() async throws {
        try await gate.withLock {
            lock.lock()
            defer { lock.unlock() }
            _ = try ensureLoadedLocked()
        }
    }

    public func close() {
        lock.lock()
        loaded = nil
        loadFailure = nil
        cachedTokens = []
        cachedToolsKey = nil
        lock.unlock()
    }

    public func invalidateKVCache() {
        lock.lock()
        defer { lock.unlock() }
        cachedTokens = []
        cachedToolsKey = nil
        if let loaded, let memory = loaded.memory {
            llama_memory_clear(memory, true)
        }
    }

    private func ensureLoadedLocked() throws -> Loaded {
        if let loaded { return loaded }
        if let loadFailure { throw loadFailure }
        do {
            let value = try Loaded(modelPath: modelPath,
                                   contextWindow: contextWindow,
                                   gpuLayers: gpuLayers,
                                   kvInt8: kvInt8)
            loaded = value
            return value
        } catch {
            loadFailure = error
            throw error
        }
    }

    // MARK: - 预热

    /// 预填充 system + tools 前缀，不生成 token；正式请求可直接复用该段 KV。
    /// KV 复用关闭时预热无意义，直接跳过。
    public func warmup(systemPrompt: String, tools: [JSONValue]) async throws {
        guard kvReuse, !systemPrompt.isEmpty else { return }
        try await gate.withLock {
            try Task.checkCancellation()
            lock.lock()
            defer { lock.unlock() }

            let loaded = try ensureLoadedLocked()
            cachedTokens = []
            cachedToolsKey = nil

            // 部分模板要求至少有一条 user 消息；多渲染一条占位 user，
            // system+tools 前缀仍然可复用，正式请求会裁掉占位尾巴。
            let marker = "__WARMUP_\(UUID().uuidString)__"
            let rendered: String
            do {
                rendered = try LlamaCppPromptRenderer.render(
                    template: loaded.model.chatTemplate,
                    model: loaded.model,
                    messages: [
                        ChatMessage(role: .system, content: systemPrompt),
                        ChatMessage(role: .user, content: marker),
                    ],
                    tools: tools,
                    enableThinking: enableThinking,
                    addGenerationPrompt: false)
            } catch {
                AgentLog.warning("[gguf-warmup] 模板不支持前缀预热，跳过："
                    + error.localizedDescription)
                return
            }
            // 只保留到占位 user 内容之前：system + tools + user 头。
            // 正式请求的第一条 user 消息天然以这段为前缀，可精确续写，
            // 对混合 SSM 模型也无需回滚 KV。
            guard let markerRange = rendered.range(of: marker) else {
                AgentLog.warning("[gguf-warmup] 渲染结果缺少占位标记，跳过")
                return
            }
            let prompt = String(rendered[..<markerRange.lowerBound])
            let tokens = try loaded.model.tokenize(prompt, addSpecial: true)
            guard tokens.count >= 32, tokens.count <= contextWindow - 1_024 else {
                AgentLog.warning("[gguf-warmup] 跳过：前缀 token 数 \(tokens.count) 不适合预热")
                return
            }
            llama_memory_clear(loaded.memory, true)
            try LlamaCppContext.decode(tokens: tokens,
                                       context: loaded.context,
                                       chunkSize: loaded.batchSize)
            cachedTokens = tokens
            cachedToolsKey = Self.toolsKey(tools)
            AgentLog.info("[gguf-warmup] prefill tokens=\(tokens.count) tools=\(tools.count)")
        }
    }

    // MARK: - 生成

    private func run(messages: [ChatMessage],
                     tools: [JSONValue],
                     temperature: Double,
                     continuation: AsyncThrowingStream<LLMStreamEvent, Error>.Continuation) async throws {
        try await gate.withLock {
            try Task.checkCancellation()
            lock.lock()
            defer { lock.unlock() }

            let loaded = try ensureLoadedLocked()
            let model = loaded.model
            let basePrompt = try LlamaCppPromptRenderer.render(
                template: model.chatTemplate,
                model: model,
                messages: messages,
                tools: tools,
                enableThinking: enableThinking,
                addGenerationPrompt: false)
            let prompt = try LlamaCppPromptRenderer.render(
                template: model.chatTemplate,
                model: model,
                messages: messages,
                tools: tools,
                enableThinking: enableThinking,
                addGenerationPrompt: true)
            let tokens = try model.tokenize(prompt, addSpecial: true)
            let generationPrefix = LlamaCppPromptRenderer.generationPrefix(
                full: prompt, base: basePrompt)
            guard !tokens.isEmpty else {
                throw AgentError.llm("GGUF prompt 为空")
            }
            guard tokens.count <= contextWindow - 64 else {
                throw AgentError.llm(
                    "GGUF prompt \(tokens.count) tokens 超过上下文预算 \(contextWindow)")
            }

            let toolsKey = Self.toolsKey(tools)
            let toolsMatch = kvReuse && cachedToolsKey == toolsKey
            var common = toolsMatch
                ? LlamaCppContext.commonPrefix(cachedTokens, tokens) : 0
            if !kvReuse, cachedTokens.count > 0 {
                // 开关关闭：丢弃上一轮缓存，整段重建（关闭前已缓存的 KV 一并清掉）。
                llama_memory_clear(loaded.memory, true)
                cachedTokens = []
                cachedToolsKey = nil
                common = 0
            }
            if common == tokens.count {
                common = max(0, tokens.count - 1)
            }
            AgentLog.info("[gguf-kv] common=\(common) cached=\(cachedTokens.count) "
                + "prompt=\(tokens.count) tools_match=\(toolsMatch)")
            if common > 0, common < cachedTokens.count, let memory = loaded.memory {
                let removed = llama_memory_seq_rm(memory, 0, llama_pos(common), -1)
                if !removed {
                    // 混合 SSM 模型不支持按位置裁剪 KV；只能整段重建。
                    AgentLog.warning("[gguf-kv] seq_rm(\(common)) 失败，重建缓存")
                    common = 0
                }
            }
            if common == 0 {
                llama_memory_clear(loaded.memory, true)
            }

            let suffix = Array(tokens[common...])
            let prefillStart = Date()
            try LlamaCppContext.decode(tokens: suffix,
                                       context: loaded.context,
                                       chunkSize: loaded.batchSize)
            let prefillMilliseconds = Int(Date().timeIntervalSince(prefillStart) * 1_000)
            let cachedPromptTokens = common

            let generationBudget = max(1, min(maxTokens, contextWindow - tokens.count - 8))
            let sampling = LlamaCppSampling(
                temperature: temperature,
                topK: topK,
                topP: topP,
                minP: minP,
                repetitionPenalty: repetitionPenalty)
            let sampler = LlamaCppSampler(vocabSize: model.vocabSize, config: sampling)
            let startInThink = prompt
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .hasSuffix("<think>")
            var thinkSplitter = ThinkSplitter(startInThink: startInThink)
            var toolSplitter = ToolTagSplitter()
            var decoder = LlamaCppUTF8StreamDecoder()
            var rawContent = generationPrefix
            var generated: [llama_token] = []
            var nPast = tokens.count
            let allowedToolNames = Set(tools.compactMap {
                $0["function"]?["name"]?.stringValue
            })
            let generationStart = Date()

            func emit(_ text: String) {
                rawContent += text
                for piece in thinkSplitter.feed(text) {
                    if piece.reasoning {
                        if !piece.text.isEmpty {
                            continuation.yield(LLMStreamEvent(
                                kind: .reasoning, content: piece.text))
                        }
                    } else {
                        for visible in toolSplitter.feed(piece.text) where !visible.isEmpty {
                            continuation.yield(LLMStreamEvent(
                                kind: .content, content: visible))
                        }
                    }
                }
            }

            do {
                while generated.count < generationBudget {
                    try Task.checkCancellation()
                    if nPast >= contextWindow { break }
                    let token = sampler.sample(context: loaded.context)
                    sampler.accept(token)
                    generated.append(token)
                    try LlamaCppContext.decode(token: token, context: loaded.context)
                    nPast += 1
                    if model.isEndOfGeneration(token) { break }

                    let bytes = model.pieceBytes(token)
                    if !bytes.isEmpty {
                        let text = decoder.feed(bytes)
                        if !text.isEmpty { emit(text) }
                    }
                }

                let tail = decoder.flush()
                if !tail.isEmpty { emit(tail) }
                for piece in thinkSplitter.flush() {
                    if piece.reasoning {
                        if !piece.text.isEmpty {
                            continuation.yield(LLMStreamEvent(
                                kind: .reasoning, content: piece.text))
                        }
                    } else {
                        for visible in toolSplitter.feed(piece.text) where !visible.isEmpty {
                            continuation.yield(LLMStreamEvent(
                                kind: .content, content: visible))
                        }
                    }
                }
                for visible in toolSplitter.flush() where !visible.isEmpty {
                    continuation.yield(LLMStreamEvent(kind: .content, content: visible))
                }
            } catch {
                llama_memory_clear(loaded.memory, true)
                cachedTokens = []
                cachedToolsKey = nil
                throw error
            }

            let toolCalls = LocalToolCallParser.extract(
                raw: rawContent, allowedNames: allowedToolNames)
            let elapsed = max(0.001, Date().timeIntervalSince(generationStart))
            let usage: [String: Int] = [
                "prompt_tokens": tokens.count,
                "completion_tokens": generated.count,
                "total_tokens": tokens.count + generated.count,
                "tokens_per_second": Int(Double(generated.count) / elapsed),
                "cached_prompt_tokens": cachedPromptTokens,
            ]
            AgentLog.info("[gguf] prompt=\(tokens.count) cached=\(cachedPromptTokens) "
                + "generated=\(generated.count) prefill_ms=\(prefillMilliseconds) "
                + "tps=\(usage["tokens_per_second"] ?? 0)")

            cachedTokens = tokens + generated
            cachedToolsKey = toolsKey
            continuation.yield(LLMStreamEvent(
                kind: .done,
                toolCalls: toolCalls,
                usage: usage,
                rawContent: rawContent))
        }
    }

    // MARK: - 状态

    private final class Loaded {
        let model: LlamaCppModel
        let context: OpaquePointer
        let memory: OpaquePointer?
        let batchSize: Int

        init(modelPath: String, contextWindow: Int, gpuLayers: Int, kvInt8: Bool) throws {
            model = try LlamaCppModel(path: modelPath, gpuLayers: gpuLayers)
            batchSize = min(2_048, max(64, contextWindow))
            context = try LlamaCppContext.make(
                model: model,
                contextWindow: contextWindow,
                batchSize: batchSize,
                embeddings: false,
                kvInt8: kvInt8)
            memory = llama_get_memory(context)
        }

        deinit {
            llama_free(context)
        }
    }

    private static func toolsKey(_ tools: [JSONValue]) -> String {
        tools.map { $0.jsonString() }.joined(separator: "\n")
    }

    private static func mapError(_ error: Error, modelPath: String) -> Error {
        if error is CancellationError { return AgentError.cancelled }
        if let agentError = error as? AgentError { return agentError }
        return AgentError.llm(
            "本地 GGUF 模型 \((modelPath as NSString).lastPathComponent) 推理失败："
            + error.localizedDescription)
    }
}
