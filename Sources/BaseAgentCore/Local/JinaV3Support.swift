import Foundation

/// jina-embeddings-v3（XLM-RoBERTa + RoPE + task LoRA）模型族的识别。
///
/// MLXEmbedders 的 `jina-v3-mlx` 架构不在默认工厂里，需要走自定义推理；
/// ONNX 单文件模型则复用通用 provider，只额外提供 `task_id` 输入。
enum JinaV3Profile {
    /// 原生输出维度。
    static let nativeDimension = 1024

    /// 名称/标识符是否指向 jina-embeddings-v3。
    static func matches(_ text: String) -> Bool {
        let lower = text.lowercased()
        return lower.contains("jina-embeddings-v3")
            || lower.contains("jina_v3")
            || lower.contains("jina-v3")
    }

    /// 目录是否为 jina-embeddings-v3（优先看 config.json，其次看目录名）。
    static func isJinaV3(directory: URL) -> Bool {
        if matches(directory.lastPathComponent) { return true }
        guard let config = readConfig(directory: directory) else { return false }
        if let modelType = config["model_type"]?.stringValue, matches(modelType) {
            return true
        }
        if config["task_id_map"] != nil { return true }
        if let architectures = config["architectures"]?.arrayValue?
            .compactMap({ $0.stringValue }),
           architectures.contains(where: matches) {
            return true
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

    /// MLX 权重文件：转换后的统一文件优先，其次兼容 `model.safetensors`。
    static func weightsURL(in directory: URL) -> URL? {
        let fm = FileManager.default
        for name in ["weights.safetensors", "model.safetensors"] {
            let url = directory.appendingPathComponent(name)
            if fm.fileExists(atPath: url.path) { return url }
        }
        return nil
    }
}

/// 任务类型 → task_id。与 `jina-embeddings-v3-onnx-int4/config.json` 的
/// `task_id_map` 以及 MLX unified 权重的 5 组 LoRA 顺序一致。
enum JinaV3Task: Int, Sendable {
    case retrievalQuery = 0
    case retrievalPassage = 1
    case separation = 2
    case classification = 3
    case textMatching = 4

    static func resolve(_ task: EmbeddingTask) -> JinaV3Task {
        switch task {
        case .retrievalQuery: return .retrievalQuery
        case .retrievalDocument: return .retrievalPassage
        case .clustering: return .separation
        case .classification: return .classification
        }
    }

    /// 文档分片必须用 `retrieval.passage`，检索查询用 `retrieval.query`。
    static func id(for task: EmbeddingTask) -> Int64 {
        Int64(resolve(task).rawValue)
    }
}

/// MLX unified 权重文件的结构参数（来自模型目录的 config.json）。
struct JinaV3ModelConfiguration: Sendable {
    let hiddenSize: Int
    let layerCount: Int
    let headCount: Int
    let intermediateSize: Int
    let layerNormEps: Float
    let vocabularySize: Int
    let typeVocabularySize: Int
    let rotaryBase: Float
    let loraRank: Int
    let loraAlpha: Float
    let unified: Bool

    let groupSize: Int
    let bits: Int
    let mixedBits: Int?
    let mixedPatterns: [String]
    let quantizationMode: String

    var headDimension: Int { hiddenSize / headCount }
    var loraScaling: Float { loraAlpha / Float(loraRank) }

    /// 指定权重的量化位数：命中 mixed pattern（如 mixer）时用 mixedBits。
    func bits(for key: String) -> Int {
        guard let mixedBits else { return bits }
        return mixedPatterns.contains(where: { key.contains($0) }) ? mixedBits : bits
    }

    static func load(from directory: URL) throws -> JinaV3ModelConfiguration {
        guard let config = JinaV3Profile.readConfig(directory: directory) else {
            throw LocalModelError.invalidConfiguration(
                "Jina v3 模型缺少可解析的 config.json（目录：\(directory.path)）")
        }
        guard JinaV3Profile.isJinaV3(directory: directory) else {
            throw LocalModelError.unsupportedArchitecture(
                "不是 jina-embeddings-v3 模型目录：\(directory.path)")
        }

        let quantization = config["quantization"]?.objectValue
            ?? config["quantization_config"]?.objectValue
        let mixed = quantization?["mixed"]?.objectValue

        return JinaV3ModelConfiguration(
            hiddenSize: config["hidden_size"]?.intValue ?? JinaV3Profile.nativeDimension,
            layerCount: config["num_hidden_layers"]?.intValue ?? 24,
            headCount: config["num_attention_heads"]?.intValue ?? 16,
            intermediateSize: config["intermediate_size"]?.intValue ?? 4_096,
            layerNormEps: Float(config["layer_norm_eps"]?.doubleValue ?? 1e-5),
            vocabularySize: config["vocab_size"]?.intValue ?? 250_002,
            typeVocabularySize: config["type_vocab_size"]?.intValue ?? 1,
            rotaryBase: Float(config["rotary_emb_base"]?.doubleValue ?? 20_000),
            loraRank: config["lora_rank"]?.intValue ?? 4,
            loraAlpha: Float(config["lora_alpha"]?.doubleValue ?? 1),
            unified: config["unified"]?.boolValue ?? true,
            groupSize: quantization?["group_size"]?.intValue ?? 32,
            bits: quantization?["bits"]?.intValue ?? 4,
            mixedBits: mixed?["bits"]?.intValue,
            mixedPatterns: mixed?["patterns"]?.arrayValue?
                .compactMap { $0.stringValue } ?? [],
            quantizationMode: quantization?["mode"]?.stringValue ?? "affine"
        )
    }
}
