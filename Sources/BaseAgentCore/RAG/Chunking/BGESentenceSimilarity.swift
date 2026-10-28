import Foundation
import Tokenizers

/// 语义切片的句子相似度：内置 bge-small-zh-v1.5（int8 ONNX，纯 CPU）。
///
/// 取每个句子的 [CLS] 向量并 L2 归一化，再算相邻句余弦。
/// 模型随 App 资源 bundle 分发，不联网、不写用户目录。
struct BGESentenceSimilarity {
    static let resourceName = "bge-small-zh-v1.5-int8"
    static let modelFileName = "model_int8.onnx"
    static let maxSequenceLength = 512
    static let batchSize = 32
    static let intraOpThreads = 4

    private static let runtime = BGERuntime()

    /// 资源目录；环境变量仅用于开发/基准替换模型。
    static func modelDirectory() -> URL? {
        let override = ProcessInfo.processInfo.environment["BASE_AGENT_SEMANTIC_MODEL"] ?? ""
        if !override.isEmpty {
            let url = URL(fileURLWithPath: override).standardizedFileURL
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        return Bundle.module.url(forResource: resourceName, withExtension: nil)
    }

    /// HuggingFace 默认编码（含 [CLS]/[SEP]）；供 golden 对齐测试使用。
    func tokenIDs(for sentence: String) -> [Int]? {
        Self.runtime.tokenIDs(for: sentence)
    }

    /// 每个句子的 [CLS] 向量（已 L2 归一化）。任一句无法编码/推理失败时返回 nil。
    func sentenceVectors(_ sentences: [String]) -> [[Float]]? {
        Self.runtime.sentenceVectors(sentences)
    }

    /// 相邻句余弦（长度 = 句数 - 1）。不足两句或任一句无法编码时返回 nil。
    func adjacentCosines(_ sentences: [String]) -> [Double]? {
        guard sentences.count >= 2 else { return nil }
        guard let vectors = sentenceVectors(sentences) else { return nil }
        return (0..<(vectors.count - 1)).map {
            Self.cosine(vectors[$0], vectors[$0 + 1])
        }
    }

    private static func cosine(_ a: [Float], _ b: [Float]) -> Double {
        guard a.count == b.count else { return 0 }
        var dot = 0.0
        var normA = 0.0
        var normB = 0.0
        for index in a.indices {
            dot += Double(a[index]) * Double(b[index])
            normA += Double(a[index]) * Double(a[index])
            normB += Double(b[index]) * Double(b[index])
        }
        guard normA > 0, normB > 0 else { return 0 }
        return dot / (sqrt(normA) * sqrt(normB))
    }
}

/// 模型与分词器只加载一次；失败后缓存失败状态，避免每个章节重复尝试。
private final class BGERuntime {
    private let lock = NSLock()
    private var state: State?
    private var failed = false
    private var warned = false

    private struct State {
        let session: OrtSession
        let tokenizer: BertTokenizer
        let clsID: Int
        let sepID: Int
        let unkID: Int
        let padID: Int
    }

    func tokenIDs(for sentence: String) -> [Int]? {
        lock.lock()
        defer { lock.unlock() }
        guard let state = loadIfNeeded() else { return nil }
        return encode(sentence, state: state)
    }

    /// 批量编码并推理：batch 32 纯 CPU，失败统一返回 nil 让调用方回退。
    func sentenceVectors(_ sentences: [String]) -> [[Float]]? {
        guard !sentences.isEmpty else { return [] }
        if Task.isCancelled { return nil }
        lock.lock()
        defer { lock.unlock() }
        guard let state = loadIfNeeded() else { return nil }

        var encoded: [[Int]] = []
        encoded.reserveCapacity(sentences.count)
        for sentence in sentences {
            guard let ids = encode(sentence, state: state) else { return nil }
            encoded.append(ids)
        }

        var result: [[Float]] = []
        result.reserveCapacity(encoded.count)
        var start = 0
        while start < encoded.count {
            if Task.isCancelled { return nil }
            let end = min(start + BGESentenceSimilarity.batchSize, encoded.count)
            guard let vectors = run(Array(encoded[start..<end]), state: state) else { return nil }
            result.append(contentsOf: vectors)
            start = end
        }
        return result
    }

    // MARK: - 加载

    private func loadIfNeeded() -> State? {
        if let state { return state }
        guard !failed else { return nil }
        guard let directory = BGESentenceSimilarity.modelDirectory() else {
            failed = true
            warnOnce("未找到内置语义分片模型")
            return nil
        }
        guard let vocabText = try? String(
            contentsOf: directory.appendingPathComponent("vocab.txt"), encoding: .utf8
        ) else {
            failed = true
            warnOnce("语义分片模型缺少 vocab.txt")
            return nil
        }
        // 只按 LF 切分：词表里存在 U+2028（行分隔符）这样的 token 本身，
        // 用 `\.isNewline` 会把它当换行吞掉，导致后续 token id 整体错位。
        var lines = vocabText.components(separatedBy: "\n")
        if lines.last?.isEmpty == true { lines.removeLast() }
        var vocab: [String: Int] = [:]
        vocab.reserveCapacity(lines.count)
        for (index, line) in lines.enumerated() {
            vocab[line] = index
        }
        guard let clsID = vocab["[CLS]"], let sepID = vocab["[SEP]"],
              let unkID = vocab["[UNK]"], let padID = vocab["[PAD]"] else {
            failed = true
            warnOnce("语义分片模型词表缺少特殊 token")
            return nil
        }
        do {
            // 不挂 CoreML EP：24 MB 小模型在 CPU 上更稳定，也避免和 LLM 抢占显存。
            let session = try OrtSession(
                modelPath: directory
                    .appendingPathComponent(BGESentenceSimilarity.modelFileName).path,
                intraOpThreads: BGESentenceSimilarity.intraOpThreads
            )
            let state = State(
                session: session,
                tokenizer: BertTokenizer(
                    vocab: vocab, merges: nil,
                    tokenizeChineseChars: true,
                    bosToken: nil, eosToken: nil,
                    fuseUnknownTokens: false,
                    doLowerCase: false
                ),
                clsID: clsID, sepID: sepID, unkID: unkID, padID: padID
            )
            self.state = state
            return state
        } catch {
            failed = true
            warnOnce("语义分片模型加载失败：\(error.localizedDescription)")
            return nil
        }
    }

    // MARK: - 分词

    private func encode(_ sentence: String, state: State) -> [Int]? {
        let tokens = state.tokenizer.tokenize(text: sentence)
        guard !tokens.isEmpty else { return nil }
        let content = tokens
            .prefix(BGESentenceSimilarity.maxSequenceLength - 2)
            .map { state.tokenizer.convertTokenToId($0) ?? state.unkID }
        // 整句都是 [UNK] 时没有有效语义，交给调用方回退词法相似度。
        guard content.contains(where: { $0 != state.unkID }) else { return nil }
        return [state.clsID] + content + [state.sepID]
    }

    // MARK: - 推理

    private func run(_ batch: [[Int]], state: State) -> [[Float]]? {
        let width = batch.map(\.count).max() ?? 0
        guard width > 0 else { return nil }
        var ids: [Int64] = []
        var mask: [Int64] = []
        ids.reserveCapacity(batch.count * width)
        mask.reserveCapacity(batch.count * width)
        for row in batch {
            for index in 0..<width {
                ids.append(index < row.count ? Int64(row[index]) : Int64(state.padID))
                mask.append(index < row.count ? 1 : 0)
            }
        }
        do {
            var inputs: [String: OrtValue] = [:]
            for name in state.session.inputNames {
                switch name {
                case "input_ids":
                    inputs[name] = try OrtValue(int64: ids, shape: [batch.count, width])
                case "attention_mask":
                    inputs[name] = try OrtValue(int64: mask, shape: [batch.count, width])
                case "token_type_ids":
                    inputs[name] = try OrtValue(
                        int64: Array(repeating: 0, count: ids.count),
                        shape: [batch.count, width]
                    )
                default:
                    throw ONNXRuntimeError.input("语义分片模型要求未知输入 \(name)")
                }
            }
            let outputName = state.session.outputNames.contains("last_hidden_state")
                ? "last_hidden_state" : state.session.outputNames[0]
            let outputs = try state.session.run(inputs: inputs, outputNames: [outputName])
            guard let output = outputs[outputName] else {
                throw ONNXRuntimeError.output("语义分片模型没有返回 \(outputName)")
            }
            let tensor = try output.floatTensor()
            let shape = tensor.shape
            guard shape.count == 3, shape[0] == batch.count,
                  shape[1] == width, shape[2] > 0 else {
                throw ONNXRuntimeError.shape("语义分片模型输出 shape=\(shape)")
            }
            // CLS pooling + L2 归一化：bge 系列的标准句向量取法。
            let hidden = shape[2]
            return (0..<batch.count).map { row in
                let offset = row * width * hidden
                var vector = Array(tensor.values[offset..<(offset + hidden)])
                Self.normalize(&vector)
                return vector
            }
        } catch {
            warnOnce("语义分片推理失败：\(error.localizedDescription)")
            return nil
        }
    }

    private static func normalize(_ vector: inout [Float]) {
        var sum: Float = 0
        for value in vector { sum += value * value }
        guard sum > 0 else { return }
        let norm = sqrt(sum)
        for index in vector.indices { vector[index] /= norm }
    }

    private func warnOnce(_ message: String) {
        guard !warned else { return }
        warned = true
        AgentLog.warning("semantic_chunk: \(message)，回退词法相似度")
    }
}
