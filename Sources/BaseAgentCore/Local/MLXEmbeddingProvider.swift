import Foundation
import MLX
import MLXEmbedders
import MLXLMCommon

/// 基于 MLXEmbedders 的本地 Embedding 实现。
///
/// 与 ONNX 实现保持同样的接口：document/query 前缀、pooling + L2 归一化。
/// 模型专属处理：
/// - nomic：`search_document:` / `search_query:` 前缀；
/// - bge-zh：query 侧追加官方检索指令（document 不加），其余模型原样。
public final class MLXEmbeddingProvider: EmbeddingProvider, @unchecked Sendable {
    public static let bgeQueryInstruction = "为这个句子生成表示以用于检索相关文章："

    public let identifier: String
    public let modelName: String
    public let directory: URL

    private let maxLength: Int
    private let usesNomicPrefix: Bool
    private let usesBGEQueryInstruction: Bool
    private let state = MLXEmbeddingLoadState()
    private let dimensionLock = NSLock()
    private var storedDimension = 768

    public var dimension: Int {
        dimensionLock.lock()
        defer { dimensionLock.unlock() }
        return storedDimension
    }

    public init(directory: URL, modelName: String) {
        self.directory = directory
        self.modelName = modelName
        self.identifier = "local:mlx-embedding:\(modelName)"
        let lowerName = modelName.lowercased()
        self.usesNomicPrefix = lowerName.contains("nomic")
        // bge-zh 官方检索用法：短 query 加指令前缀以对齐检索任务（en 版与 m3 不加）。
        self.usesBGEQueryInstruction = lowerName.contains("bge") && lowerName.contains("zh")
        self.maxLength = min(
            max(LocalModelRegistry.resolveMaxSequenceLength(
                directory: directory, defaultValue: 512), 128),
            8_192)

        if let data = FileManager.default.contents(
            atPath: directory.appendingPathComponent("config.json").path),
           let config = JSONValue.parse(String(decoding: data, as: UTF8.self))?.objectValue {
            let hidden = config["hidden_size"]?.intValue
                ?? config["text_config"]?.objectValue?["hidden_size"]?.intValue
            if let hidden, hidden > 0 {
                storedDimension = hidden
            }
        }
    }

    public func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        guard !texts.isEmpty else { return [] }
        let prefixed = texts.map {
            usesNomicPrefix ? addPrefix($0, prefix: "search_document: ") : $0
        }
        return try await embed(prefixed)
    }

    public func embedQuery(_ text: String) async throws -> [Float] {
        var prefixed = usesNomicPrefix ? addPrefix(text, prefix: "search_query: ") : text
        if usesBGEQueryInstruction {
            prefixed = addPrefix(prefixed, prefix: Self.bgeQueryInstruction)
        }
        return try await embed([prefixed]).first ?? []
    }

    private func embed(_ texts: [String]) async throws -> [[Float]] {
        let container = try await state.container(directory: directory)
        let maxLength = self.maxLength
        let vectors = try await container.perform { context -> [[Float]] in
            let tokenizer = context.tokenizer
            let eos = tokenizer.eosTokenId ?? 0
            let encoded = texts.map { text -> [Int] in
                Array(tokenizer.encode(text: text, addSpecialTokens: true)
                    .prefix(maxLength))
            }
            let width = max(16, encoded.map(\.count).max() ?? 16)
            let padded = stacked(encoded.map { ids in
                MLXArray(ids + Array(repeating: eos, count: width - ids.count))
            })
            let mask = padded .!= eos
            let tokenTypes = MLXArray.zeros(like: padded)
            let output = context.model(padded, positionIds: nil,
                                       tokenTypeIds: tokenTypes,
                                       attentionMask: mask)
            let pooled = context.pooling(output, normalize: true,
                                         applyLayerNorm: true)
            pooled.eval()
            return pooled.map { $0.asArray(Float.self) }
        }
        if let hidden = vectors.first?.count, hidden > 0 {
            dimensionLock.lock()
            storedDimension = hidden
            dimensionLock.unlock()
        }
        return vectors
    }

    private func addPrefix(_ text: String, prefix: String) -> String {
        text.hasPrefix(prefix) ? text : prefix + text
    }

    /// 释放 embedding 权重；provider 保留，下次 embed 时按需重载。
    public func unload() async {
        await state.unload()
    }
}

private actor MLXEmbeddingLoadState {
    private var container: EmbedderModelContainer?
    private var failure: Error?

    func container(directory: URL) async throws -> EmbedderModelContainer {
        if let container { return container }
        if let failure { throw failure }
        do {
            let container = try await EmbedderModelFactory.shared.loadContainer(
                from: directory, using: LocalTokenizerLoader())
            self.container = container
            return container
        } catch {
            let wrapped = LocalModelError.runtime(
                "MLX Embedding 加载失败（\(directory.path)）：\(error.localizedDescription)")
            failure = wrapped
            throw wrapped
        }
    }

    func unload() {
        container = nil
        failure = nil
    }
}
