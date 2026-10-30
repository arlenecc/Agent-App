import Foundation

/// jina-embeddings-v5-text-nano（EuroBERT-210M）模型族识别与共享变换。
///
/// V5 的查询/文档区分不在模型内做任务头，而是靠输入前缀：
/// 文档分片用 `Document: `，检索查询用 `Query: `。原生输出 768 维，
/// 业务统一截前 128 维后**重新 L2 归一化**，再交给 Storage 量化成 int8。
enum JinaV5Profile {
    /// 原生输出维度。
    static let nativeDimension = 768

    /// 业务检索维度（Matryoshka 截断后）。
    static let retrievalDimension = 128

    static let queryPrefix = "Query: "
    static let documentPrefix = "Document: "

    /// 名称/标识符是否指向 jina-embeddings-v5。
    static func matches(_ text: String) -> Bool {
        let lower = text.lowercased()
        return lower.contains("jina-embeddings-v5")
            || lower.contains("jina_v5")
            || lower.contains("jina-v5")
    }

    /// 目录是否为 jina-embeddings-v5（优先看目录名，其次看 config.json）。
    static func isJinaV5(directory: URL) -> Bool {
        if matches(directory.lastPathComponent) { return true }
        guard let config = readConfig(directory: directory) else { return false }
        if let base = config["base_model"]?.stringValue, matches(base) {
            return true
        }
        if let type = config["model_type"]?.stringValue, matches(type) {
            return true
        }
        // 目录名被重命名时，用 V5 独有的配置特征兜底：
        // EuroBERT 架构 + 大写 Query/Document 前缀。
        if let type = config["model_type"]?.stringValue?.lowercased(),
           type == "eurobert" {
            if let prefixes = config["prefixes"]?.objectValue,
               prefixes["query"]?.stringValue == queryPrefix,
               prefixes["doc"]?.stringValue?.hasPrefix("Document") == true {
                return true
            }
            if config["matryoshka_dimensions"] != nil,
               config["pooling"]?.stringValue?.lowercased().contains("last") == true {
                return true
            }
        }
        return false
    }

    static func readConfig(directory: URL) -> [String: JSONValue]? {
        let url = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: url),
              let text = String(data: data, encoding: .utf8),
              let object = JSONValue.parse(text)?.objectValue else {
            return nil
        }
        return object
    }

    /// MLX 权重文件：V5 转换目录统一为 `model.safetensors`。
    static func weightsURL(in directory: URL) -> URL? {
        for name in ["weights.safetensors", "model.safetensors"] {
            let url = directory.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        return nil
    }

    /// 截前 `dimension` 维后重新 L2 归一化。
    ///
    /// 截断会破坏单位范数，必须重算；零向量保持原样，避免产生 NaN。
    static func truncateAndNormalize(_ vector: [Float],
                                     dimension: Int = retrievalDimension) -> [Float] {
        guard !vector.isEmpty else { return vector }
        var output = Array(vector.prefix(max(1, dimension)))
        var norm: Float = 0
        for value in output { norm += value * value }
        norm = norm.squareRoot()
        guard norm > 0 else { return output }
        for index in output.indices { output[index] /= norm }
        return output
    }
}
