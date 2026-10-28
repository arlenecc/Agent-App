import Foundation

/// 句子级语义切片：在相邻句相似度低谷处断开。
/// 相似度由内置 bge-small-zh-v1.5（int8 ONNX，纯 CPU）计算；
/// 不可用时退化为词重叠相似度（仍可用，只是更粗糙）。
struct SemanticChunker {

    /// 可注入的相邻句相似度（测试用）；返回 nil 表示不可用。
    typealias AdjacentSimilarity = ([String]) -> [Double]?

    /// 一个语义片。`boundary` 是该片末尾的相邻句相似度：
    /// nil 表示末尾是长度/段落边界，不能跨过去合并。
    private struct Piece {
        var text: String
        var tokens: Int
        var boundary: Double?
    }

    static func split(_ text: String, options: ChunkingOptions,
                      similarity: @escaping AdjacentSimilarity = {
                          BGESentenceSimilarity().adjacentCosines($0)
                      }) async -> [String]? {
        let sentences = TextAnalyzer.sentences(text)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard sentences.count >= 2 else { return nil }

        let similarities: [Double]
        if let value = similarity(sentences), value.count == sentences.count - 1 {
            similarities = value
        } else {
            if Task.isCancelled { return nil }
            warnFallbackOnce()
            similarities = (0..<(sentences.count - 1)).map {
                lexicalSimilarity(sentences[$0], sentences[$0 + 1])
            }
        }

        let threshold = adaptiveThreshold(similarities, fallback: options.semanticThreshold)
        let budget = max(1, options.chunkSizeTokens)
        var pieces: [Piece] = []
        var current: [String] = []
        var currentTokens = 0

        func flush(boundary: Double?) {
            guard !current.isEmpty else { return }
            let value = joinSentences(current)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !value.isEmpty {
                pieces.append(Piece(text: value,
                                    tokens: options.tokenCounter(value),
                                    boundary: boundary))
            }
            current.removeAll(keepingCapacity: true)
            currentTokens = 0
        }

        for index in sentences.indices {
            let sentence = sentences[index]
            let sentenceTokens = options.tokenCounter(sentence)
            let separator = current.last.map { joinSeparator(after: $0, before: sentence) } ?? ""
            // 先判断再追加：保证整片不超过 token 预算；
            // 单句超预算时先独占一片，交给 TextChunker 的递归回退再拆。
            if !current.isEmpty,
               currentTokens + options.tokenCounter(separator) + sentenceTokens > budget {
                flush(boundary: nil)
            }
            currentTokens += (current.isEmpty ? 0 : options.tokenCounter(separator)) + sentenceTokens
            current.append(sentence)

            let hasNext = index + 1 < sentences.count
            let boundary = hasNext && similarities[index] <= threshold
            if boundary {
                flush(boundary: similarities[index])
            }
        }
        flush(boundary: nil)

        guard !pieces.isEmpty else { return nil }
        // 短语义片在「弱边界」上合并：强主题边界（相似度明显低于阈值）保留，
        // 避免一本书被切成大量单句碎片，同时不抹掉真正的语义转折。
        let merged = mergeShortPieces(pieces, options: options,
                                      budget: budget, threshold: threshold)
        let chunks = merged.map(\.text)
        // 单块仍超限（例如整段只有一句超长文本）→ 交给递归切片。
        if chunks.count == 1, options.tokenCounter(chunks[0]) > budget {
            return RecursiveChunker.split(text, options: options)
        }
        return chunks
    }

    // MARK: 拼接

    /// 按语言边界拼接句子：中文（含全角标点）之间不插空格，
    /// 英文等拉丁文本仍以空格分隔，保证切片是原文连续子串。
    private static func joinSentences(_ sentences: [String]) -> String {
        guard var value = sentences.first else { return "" }
        for sentence in sentences.dropFirst() {
            value += joinSeparator(after: value, before: sentence) + sentence
        }
        return value
    }

    private static func joinSeparator(after left: String, before right: String) -> String {
        guard let last = left.last, let first = right.first else { return "" }
        if last.isWhitespace || first.isWhitespace { return "" }
        if isCJK(last) || isCJK(first) { return "" }
        return " "
    }

    /// CJK 文字与全角/中文标点：拼接时不插入空格。
    private static func isCJK(_ character: Character) -> Bool {
        guard character.unicodeScalars.count == 1,
              let scalar = character.unicodeScalars.first else { return false }
        switch scalar.value {
        case 0x3000...0x303F,   // CJK 标点（。、「」等）
             0x3040...0x30FF,   // 日文假名
             0x3400...0x4DBF,   // CJK 扩展 A
             0x4E00...0x9FFF,   // CJK 统一表意
             0xAC00...0xD7AF,   // 韩文音节
             0xF900...0xFAFF,   // CJK 兼容表意
             0xFF00...0xFFEF:   // 全角符号
            return true
        default:
            return false
        }
    }

    // MARK: 合并

    /// 合并短语义片：只跨「弱边界」（相似度仍接近阈值）合并，
    /// 且至少一侧短于软下限、合并后不超预算；强边界永远保留。
    private static func mergeShortPieces(_ pieces: [Piece], options: ChunkingOptions,
                                         budget: Int, threshold: Double) -> [Piece] {
        guard pieces.count > 1, options.minChunkTokens > 0 else { return pieces }
        // 软下限至少是用户设置的最小切片；但长切片下不能无限抬高，
        // 否则语义片会被大量合并、语义粒度退化成按长度切片。
        let softMin = min(budget, max(options.minChunkTokens, min(240, budget / 4)))
        let weakBoundary = max(0, threshold) * 0.75
        var merged: [Piece] = []
        merged.reserveCapacity(pieces.count)

        for piece in pieces {
            guard var last = merged.last,
                  let boundary = last.boundary,
                  boundary >= weakBoundary,
                  last.tokens < softMin || piece.tokens < softMin else {
                merged.append(piece)
                continue
            }
            let separator = joinSeparator(after: last.text, before: piece.text)
            let combined = last.text + separator + piece.text
            guard options.tokenCounter(combined) <= budget else {
                merged.append(piece)
                continue
            }
            last.text = combined
            last.tokens = options.tokenCounter(combined)
            last.boundary = piece.boundary
            merged[merged.count - 1] = last
        }
        return merged
    }

    // MARK: 相似度

    private static func lexicalSimilarity(_ a: String, _ b: String) -> Double {
        let setA = Set(TextAnalyzer.words(a))
        let setB = Set(TextAnalyzer.words(b))
        guard !setA.isEmpty, !setB.isEmpty else { return 0 }
        let intersection = setA.intersection(setB).count
        let union = setA.union(setB).count
        return union > 0 ? Double(intersection) / Double(union) : 0
    }

    /// 词向量不可用时只警告一次：模型缺失/解码失败时每个章节都会失败，
    /// 逐节告警会刷屏。
    private static let fallbackWarningLock = NSLock()
    private static var fallbackWarningLogged = false

    private static func warnFallbackOnce() {
        fallbackWarningLock.lock()
        let shouldLog = !fallbackWarningLogged
        if shouldLog { fallbackWarningLogged = true }
        fallbackWarningLock.unlock()
        if shouldLog {
            AgentLog.warning("语义切片内置模型不可用，回退词法相似度（仅提示一次）")
        }
    }

    /// 用相似度分布的低分位作为断点阈值：语义切分关心「相对低谷」，
    /// 固定阈值在长文档/短句上会要么切得过碎、要么完全不断。
    private static func adaptiveThreshold(_ values: [Double], fallback: Double) -> Double {
        guard values.count >= 4 else { return fallback }
        let sorted = values.sorted()
        let index = max(0, min(sorted.count - 1, Int(Double(sorted.count) * 0.25)))
        return max(0.15, min(fallback, sorted[index]))
    }
}
