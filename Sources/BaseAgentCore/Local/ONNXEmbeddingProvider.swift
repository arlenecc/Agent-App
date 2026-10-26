import Foundation

/// nomic-embed-text-v1.5 的 ONNX 实现。
///
/// - document 输入统一加 `search_document: ` 前缀，query 加 `search_query: ` 前缀；
/// - 输出做 attention-mask 加权 mean pooling，再 L2 归一化；
/// - 按子批执行，避免长文本批量推理一次性占用过多内存。
public final class ONNXEmbeddingProvider: EmbeddingProvider, @unchecked Sendable {
    public let identifier: String
    public let modelName: String
    public let directory: URL

    private let modelURL: URL?
    private let preflightError: Error?
    private let maxLength: Int
    private let batchSize: Int
    private let state = EmbeddingLoadState()

    private let dimensionLock = NSLock()
    private var storedDimension = 768

    /// nomic 的标准输出维度；实际值会在首次推理后按输出张量更新。
    public var dimension: Int {
        dimensionLock.lock()
        defer { dimensionLock.unlock() }
        return storedDimension
    }

    /// ONNXRuntime session.run 线程安全：同实例（共享权重）允许 2 路并发，
    /// 覆盖多文件并发 + 语义切片同时请求 embedding 的场景。
    public var maxConcurrentRuns: Int { 2 }

    private static let documentPrefix = "search_document: "
    private static let queryPrefix = "search_query: "

    public init(directory: URL, modelName: String, batchSize: Int = 32) {
        self.directory = directory
        self.modelName = modelName
        self.identifier = "local:onnx:\(modelName)"
        self.maxLength = LocalModelRegistry.resolveMaxSequenceLength(directory: directory, defaultValue: 8192)
        self.batchSize = max(1, batchSize)

        var resolved: URL?
        var loadError: Error?
        do {
            resolved = try LocalModelRegistry.resolveONNXModel(in: directory, kind: .embedding)
        } catch {
            loadError = error
        }
        self.modelURL = resolved
        self.preflightError = loadError

        // 从 config.json 预读真实维度，避免非 768 模型建库时维度错配。
        if let data = FileManager.default.contents(atPath: directory.appendingPathComponent("config.json").path),
           let config = JSONValue.parse(String(decoding: data, as: UTF8.self))?.objectValue {
            let hidden = config["hidden_size"]?.intValue
                ?? config["text_config"]?.objectValue?["hidden_size"]?.intValue
            if let hidden, hidden > 0 {
                self.storedDimension = hidden
            }
        }
    }

    // MARK: - EmbeddingProvider

    public func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        guard !texts.isEmpty else { return [] }
        let prefixed = texts.map { addPrefix($0, prefix: Self.documentPrefix) }
        return try await embed(prefixed)
    }

    public func embedQuery(_ text: String) async throws -> [Float] {
        let result = try await embed([addPrefix(text, prefix: Self.queryPrefix)])
        return result.first ?? []
    }

    // MARK: - 推理

    private func embed(_ texts: [String]) async throws -> [[Float]] {
        if let preflightError { throw preflightError }
        guard let modelURL else {
            throw LocalModelError.runtime("Embedding 模型路径未解析：\(directory.path)")
        }

        let (session, tokenizer) = try await state.handles(
            modelURL: modelURL,
            directory: directory
        )
        let outputName = preferredOutputName(in: session)
        var result: [[Float]] = []
        result.reserveCapacity(texts.count)

        var start = 0
        while start < texts.count {
            try Task.checkCancellation()
            let end = min(start + batchSize, texts.count)
            let batch = Array(texts[start..<end])
            let encoded = tokenizer.encodeBatch(batch, maxLength: maxLength)
            let inputs = try makeInputs(encoded, inputNames: session.inputNames)
            let outputs = try session.run(inputs: inputs, outputNames: [outputName])
            guard let output = outputs[outputName] else {
                throw ONNXRuntimeError.output("模型没有返回输出 \(outputName)")
            }
            let tensor = try output.floatTensor()
            result.append(contentsOf: try pool(tensor, mask: encoded))
            start = end
        }
        return result
    }

    private func makeInputs(
        _ encoded: TokenizedBatch,
        inputNames: [String]
    ) throws -> [String: OrtValue] {
        let batch = encoded.batchSize
        let length = encoded.sequenceLength
        let inputIDs = encoded.inputIDs.flatMap { row in row.map(Int64.init) }
        let mask = encoded.attentionMask.flatMap { row in row.map(Int64.init) }

        var inputs: [String: OrtValue] = [:]
        for name in inputNames {
            switch name {
            case "input_ids":
                inputs[name] = try OrtValue(int64: inputIDs, shape: [batch, length])
            case "attention_mask":
                inputs[name] = try OrtValue(int64: mask, shape: [batch, length])
            case "token_type_ids":
                inputs[name] = try OrtValue(
                    int64: Array(repeating: 0, count: batch * length),
                    shape: [batch, length]
                )
            case "position_ids":
                inputs[name] = try OrtValue(
                    int64: Self.positionIDs(mask: encoded.attentionMask),
                    shape: [batch, length]
                )
            default:
                throw ONNXRuntimeError.input("Embedding 模型要求未知输入 `\(name)`，当前封装未实现")
            }
        }
        return inputs
    }

    /// mean pooling + L2 归一化；输出维度以运行时张量最后一维为准。
    private func pool(_ tensor: OrtTensor, mask: TokenizedBatch) throws -> [[Float]] {
        let shape = tensor.shape
        if shape.count == 2 {
            let batch = shape[0]
            let hidden = shape[1]
            guard batch == mask.batchSize else {
                throw ONNXRuntimeError.shape("输出 batch=\(batch) 与输入 batch=\(mask.batchSize) 不一致")
            }
            updateDimension(hidden)
            return (0..<batch).map { row in
                var vector = Array(tensor.values[(row * hidden)..<((row + 1) * hidden)])
                Self.normalize(&vector)
                return vector
            }
        }

        guard shape.count == 3 else {
            throw ONNXRuntimeError.shape("Embedding 输出维度应为 [B,H] 或 [B,T,H]，实际为 \(shape)")
        }
        let batch = shape[0]
        let length = shape[1]
        let hidden = shape[2]
        guard batch == mask.batchSize, length == mask.sequenceLength else {
            throw ONNXRuntimeError.shape(
                "输出 shape=\(shape) 与输入 batch/token=\(mask.batchSize)/\(mask.sequenceLength) 不一致"
            )
        }
        updateDimension(hidden)

        var result: [[Float]] = []
        result.reserveCapacity(batch)
        for row in 0..<batch {
            var vector = [Float](repeating: 0, count: hidden)
            var count = 0
            for token in 0..<length where mask.attentionMask[row][token] != 0 {
                let offset = (row * length + token) * hidden
                for index in 0..<hidden {
                    vector[index] += tensor.values[offset + index]
                }
                count += 1
            }
            if count > 0 {
                let scale = Float(count)
                for index in 0..<hidden {
                    vector[index] /= scale
                }
            }
            Self.normalize(&vector)
            result.append(vector)
        }
        return result
    }

    private func preferredOutputName(in session: OrtSession) -> String {
        let preferred = ["last_hidden_state", "token_embeddings", "sentence_embedding", "embeddings"]
        for name in preferred where session.outputNames.contains(name) {
            return name
        }
        return session.outputNames[0]
    }

    private func addPrefix(_ text: String, prefix: String) -> String {
        text.hasPrefix(prefix) ? text : prefix + text
    }

    private func updateDimension(_ value: Int) {
        guard value > 0 else { return }
        dimensionLock.lock()
        storedDimension = value
        dimensionLock.unlock()
    }

    /// 由 attention_mask 生成 position_ids：每条有效 token 递增，padding 位置保持 0。
    private static func positionIDs(mask: [[Int]]) -> [Int64] {
        var result: [Int64] = []
        result.reserveCapacity(mask.reduce(0) { $0 + $1.count })
        for row in mask {
            var position: Int64 = -1
            for flag in row {
                if flag != 0 {
                    position += 1
                    result.append(max(position, 0))
                } else {
                    result.append(0)
                }
            }
        }
        return result
    }

    private static func normalize(_ vector: inout [Float]) {
        var sum: Float = 0
        for value in vector {
            sum += value * value
        }
        guard sum > 0 else { return }
        let norm = sqrt(sum)
        for index in vector.indices {
            vector[index] /= norm
        }
    }

    /// 释放 ONNX session/tokenizer；provider 保留，下次 embed 时按需重载。
    public func unload() async {
        await state.unload()
    }
}

/// 会话与 tokenizer 只加载一次；失败后缓存错误，重复调用会得到同一明确错误。
private actor EmbeddingLoadState {
    private var session: OrtSession?
    private var tokenizer: LocalTokenizer?
    private var failure: Error?

    func handles(modelURL: URL, directory: URL) async throws -> (OrtSession, LocalTokenizer) {
        if let session, let tokenizer {
            return (session, tokenizer)
        }
        if let failure {
            throw failure
        }

        do {
            // 8 线程吃满多核：embedding 推理是切片阶段的主要耗时，
            // 单会话内并行比跨会话并发更高效。
            let session = try OrtSession(modelPath: modelURL.path, intraOpThreads: 8)
            let tokenizer = try await LocalTokenizer.load(directory: directory)
            self.session = session
            self.tokenizer = tokenizer
            return (session, tokenizer)
        } catch {
            self.failure = error
            throw error
        }
    }

    func unload() {
        session = nil
        tokenizer = nil
        failure = nil
    }
}
