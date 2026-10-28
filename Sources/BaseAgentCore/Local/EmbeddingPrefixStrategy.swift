import Foundation

/// Embedding 模型族。不同模型系列对输入前缀有不同要求。
enum EmbeddingModelFamily: Equatable, Sendable {
    case nomic
    case bgeZh
    case jina
    case e5
    case generic
}

/// 向量化任务类型。
///
/// 当前业务只使用 document / query；clustering、classification 先作为
/// 前缀策略的一部分定义，后续需要任务型 embedding API 时可直接复用。
enum EmbeddingTask: Equatable, Sendable {
    case retrievalDocument
    case retrievalQuery
    case clustering
    case classification
}

/// 按模型族生成 embedding 输入前缀。
///
/// 前缀只在 provider 的 `embedDocuments` / `embedQuery` 内添加，调用方不要
/// 重复拼；`apply` 是幂等的，文本已带目标前缀时原样返回。
struct EmbeddingPrefixStrategy: Equatable, Sendable {
    /// bge-zh 官方检索指令（document 侧不加）。
    static let bgeQueryInstruction = "为这个句子生成表示以用于检索相关文章："

    let family: EmbeddingModelFamily

    init(family: EmbeddingModelFamily) {
        self.family = family
    }

    init(modelName: String, modelPath: String? = nil) {
        self.family = Self.detectFamily(modelName: modelName, modelPath: modelPath)
    }

    /// 指定任务的前缀；nil 表示该模型族在该任务下不加前缀。
    func prefix(for task: EmbeddingTask) -> String? {
        switch (family, task) {
        case (.nomic, .retrievalDocument):
            return "search_document: "
        case (.nomic, .retrievalQuery):
            return "search_query: "
        case (.bgeZh, .retrievalQuery):
            return Self.bgeQueryInstruction
        case (.jina, .retrievalDocument):
            return "passage: "
        case (.jina, .retrievalQuery):
            return "query: "
        case (.jina, .clustering):
            return "clustering: "
        case (.jina, .classification):
            return "classification: "
        case (.e5, .retrievalDocument):
            return "passage: "
        case (.e5, .retrievalQuery):
            return "query: "
        default:
            return nil
        }
    }

    func apply(_ text: String, task: EmbeddingTask) -> String {
        guard let prefix = prefix(for: task) else { return text }
        return text.hasPrefix(prefix) ? text : prefix + text
    }

    /// document 侧前缀需要额外预留的 token 数（特殊 token 由调用方另扣）。
    /// 切片预算用这个值给前缀留余量，避免长块尾部被模型截断。
    var documentPrefixTokenReserve: Int {
        switch family {
        case .nomic:
            return 8
        case .jina, .e5:
            return 4
        case .bgeZh, .generic:
            return 0
        }
    }

    // MARK: - 模型族识别

    /// 同时看模型名与模型路径的最后一段，避免用户名/上级目录里的字符串干扰。
    static func detectFamily(modelName: String,
                             modelPath: String? = nil) -> EmbeddingModelFamily {
        let tokens = Set([modelName, modelPath]
            .compactMap { $0 }
            .flatMap { nameTokens($0) })

        if tokens.contains("nomic") { return .nomic }
        if tokens.contains("bge") {
            return tokens.contains("zh") ? .bgeZh : .generic
        }
        if tokens.contains("jina") { return .jina }
        if tokens.contains("e5") { return .e5 }
        return .generic
    }

    private static func nameTokens(_ value: String) -> [String] {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        // 只看文件名/目录名与上一级：GGUF 可能叫 model.gguf，模型族在父目录；
        // 同时排除用户名等更上层的路径，避免 "zhang" 被当成 zh。
        let parts = trimmed.split(whereSeparator: { $0 == "/" || $0 == "\\" })
        let leaf = parts.suffix(2).joined(separator: " ")
        return leaf.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }
    }
}
