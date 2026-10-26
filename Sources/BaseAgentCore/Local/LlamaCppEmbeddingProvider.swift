import Foundation
import llama

/// 进程内 GGUF Embedding：直接加载模型并用 llama.cpp 的 pooled embedding。
public final class LlamaCppEmbeddingProvider: EmbeddingProvider, @unchecked Sendable {
    public let identifier: String
    public let modelPath: String

    private let modelName: String
    private let usesNomicPrefix: Bool
    private let gate = AsyncGate()
    private let stateLock = NSLock()
    private var runtime: LlamaCppEmbeddingRuntime?
    private var loadFailure: Error?
    private let dimensionLock = NSLock()
    private var storedDimension = 768

    public var dimension: Int {
        dimensionLock.lock()
        defer { dimensionLock.unlock() }
        return storedDimension
    }

    init(modelPath: String, modelName: String) {
        self.modelPath = modelPath
        self.modelName = modelName
        self.identifier = "gguf:\(modelName)"
        self.usesNomicPrefix = modelName.lowercased().contains("nomic")
            || modelPath.lowercased().contains("nomic")
    }

    public func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        guard !texts.isEmpty else { return [] }
        let prefixed = texts.map {
            usesNomicPrefix ? addPrefix($0, prefix: "search_document: ") : $0
        }
        return try await gate.withLock {
            let runtime = try self.ensureRuntime()
            var result: [[Float]] = []
            result.reserveCapacity(prefixed.count)
            for text in prefixed {
                try Task.checkCancellation()
                let vector = try runtime.embed(text)
                updateDimension(vector.count)
                result.append(vector)
            }
            return result
        }
    }

    public func embedQuery(_ text: String) async throws -> [Float] {
        let prefixed = usesNomicPrefix ? addPrefix(text, prefix: "search_query: ") : text
        return try await gate.withLock {
            let runtime = try self.ensureRuntime()
            let vector = try runtime.embed(prefixed)
            updateDimension(vector.count)
            return vector
        }
    }

    public func unload() async {
        await gate.acquire()
        stateLock.lock()
        let current = runtime
        runtime = nil
        loadFailure = nil
        stateLock.unlock()
        current?.close()
        gate.release()
    }

    private func ensureRuntime() throws -> LlamaCppEmbeddingRuntime {
        stateLock.lock()
        defer { stateLock.unlock() }
        if let runtime { return runtime }
        if let loadFailure { throw loadFailure }
        do {
            let value = try LlamaCppEmbeddingRuntime(modelPath: modelPath)
            runtime = value
            updateDimension(value.dimension)
            return value
        } catch {
            loadFailure = error
            throw error
        }
    }

    private func updateDimension(_ value: Int) {
        guard value > 0 else { return }
        dimensionLock.lock()
        storedDimension = value
        dimensionLock.unlock()
    }

    private func addPrefix(_ text: String, prefix: String) -> String {
        text.hasPrefix(prefix) ? text : prefix + text
    }
}

/// 进程内 GGUF Reranker：要求模型使用 RANK pooling（cross-encoder）。
public final class LlamaCppReranker: RerankProvider, @unchecked Sendable {
    public let identifier: String
    public let modelPath: String

    private let gate = AsyncGate()
    private let stateLock = NSLock()
    private var runtime: LlamaCppRerankerRuntime?
    private var loadFailure: Error?

    init(modelPath: String) {
        self.modelPath = modelPath
        self.identifier = "gguf-rerank:\((modelPath as NSString).lastPathComponent)"
    }

    public func rerank(query: String, documents: [String]) async throws -> [Double] {
        guard !documents.isEmpty else { return [] }
        return try await gate.withLock {
            let runtime = try self.ensureRuntime()
            var scores: [Double] = []
            scores.reserveCapacity(documents.count)
            for document in documents {
                try Task.checkCancellation()
                scores.append(try runtime.score(query: query, document: document))
            }
            return scores
        }
    }

    public func unload() async {
        await gate.acquire()
        stateLock.lock()
        let current = runtime
        runtime = nil
        loadFailure = nil
        stateLock.unlock()
        current?.close()
        gate.release()
    }

    private func ensureRuntime() throws -> LlamaCppRerankerRuntime {
        stateLock.lock()
        defer { stateLock.unlock() }
        if let runtime { return runtime }
        if let loadFailure { throw loadFailure }
        do {
            let value = try LlamaCppRerankerRuntime(modelPath: modelPath)
            runtime = value
            return value
        } catch {
            loadFailure = error
            throw error
        }
    }
}

// MARK: - 运行时

final class LlamaCppEmbeddingRuntime: @unchecked Sendable {
    private let model: LlamaCppModel
    private let context: OpaquePointer
    private let memory: OpaquePointer?
    let dimension: Int
    private let batchSize = 512
    private var closed = false

    init(modelPath: String) throws {
        model = try LlamaCppModel(path: modelPath, gpuLayers: 0)
        context = try LlamaCppContext.make(
            model: model, contextWindow: 2_048, batchSize: 512, embeddings: true)
        memory = llama_get_memory(context)
        dimension = Int(llama_model_n_embd_out(model.raw))
        guard dimension > 0 else {
            closed = true
            llama_free(context)
            throw LlamaCppError.unsupported("Embedding 模型没有输出维度")
        }
        llama_set_embeddings(context, true)
    }

    func close() {
        guard !closed else { return }
        closed = true
        llama_free(context)
    }

    deinit {
        close()
    }

    func embed(_ text: String) throws -> [Float] {
        let tokens = try model.tokenize(text, addSpecial: true)
        guard !tokens.isEmpty else {
            return [Float](repeating: 0, count: dimension)
        }
        let truncated = Array(tokens.prefix(2_048 - 2))
        if let memory {
            llama_memory_clear(memory, true)
        }
        try LlamaCppContext.decode(tokens: truncated, context: context, chunkSize: batchSize)

        let pooling = llama_pooling_type(context)
        let pointer: UnsafeMutablePointer<Float>?
        if pooling == LLAMA_POOLING_TYPE_NONE {
            pointer = llama_get_embeddings_ith(context, -1)
        } else {
            pointer = llama_get_embeddings_seq(context, 0)
        }
        guard let pointer else {
            throw LlamaCppError.decode("Embedding 模型没有返回向量")
        }
        var vector = Array(UnsafeBufferPointer(start: pointer, count: dimension))
        Self.normalize(&vector)
        return vector
    }

    static func normalize(_ vector: inout [Float]) {
        var norm: Float = 0
        for value in vector { norm += value * value }
        norm = sqrt(norm)
        guard norm > 0 else { return }
        for index in vector.indices { vector[index] /= norm }
    }
}

final class LlamaCppRerankerRuntime: @unchecked Sendable {
    private let model: LlamaCppModel
    private let context: OpaquePointer
    private let memory: OpaquePointer?
    private let batchSize = 512
    private let rerankTemplate: String?
    private var closed = false

    init(modelPath: String) throws {
        model = try LlamaCppModel(path: modelPath, gpuLayers: 0)
        context = try LlamaCppContext.make(
            model: model, contextWindow: 2_048, batchSize: 512, embeddings: true)
        memory = llama_get_memory(context)
        llama_set_embeddings(context, true)
        let pooling = llama_pooling_type(context)
        if pooling != LLAMA_POOLING_TYPE_RANK {
            closed = true
            llama_free(context)
            throw LlamaCppError.unsupported(
                "该 GGUF 不是 reranker（pooling 类型为 \(pooling.rawValue)，需要 RANK pooling）")
        }
        if let raw = llama_model_chat_template(model.raw, "rerank") {
            let value = String(cString: raw)
            rerankTemplate = value.isEmpty ? nil : value
        } else {
            rerankTemplate = nil
        }
    }

    func close() {
        guard !closed else { return }
        closed = true
        llama_free(context)
    }

    deinit {
        close()
    }

    func score(query: String, document: String) throws -> Double {
        let tokens = try makeTokens(query: query, document: document)
        guard !tokens.isEmpty else { return 0 }
        let truncated = Array(tokens.prefix(2_048 - 2))
        if let memory {
            llama_memory_clear(memory, true)
        }
        try LlamaCppContext.decode(tokens: truncated, context: context, chunkSize: batchSize)
        let pointer = llama_get_embeddings_seq(context, 0)
            ?? llama_get_embeddings_ith(context, -1)
        guard let pointer else {
            throw LlamaCppError.decode("Reranker 没有返回分数")
        }
        return Double(pointer[0])
    }

    private func makeTokens(query: String, document: String) throws -> [llama_token] {
        if var template = rerankTemplate {
            template = template.replacingOccurrences(of: "{query}", with: query)
            template = template.replacingOccurrences(of: "{document}", with: document)
            return try model.tokenize(template, addSpecial: false)
        }

        var tokens: [llama_token] = []
        let queryTokens = try model.tokenize(query, addSpecial: false)
        let documentTokens = try model.tokenize(document, addSpecial: false)
        if model.addBos { tokens.append(model.bosToken) }
        tokens.append(contentsOf: queryTokens)
        if model.addEos { tokens.append(model.eosToken) }
        if model.addSep { tokens.append(model.sepToken) }
        tokens.append(contentsOf: documentTokens)
        if model.addEos { tokens.append(model.eosToken) }
        return tokens
    }
}
