import Foundation

/// 句子级语义切片：在相邻句相似度低谷处断开。
/// 相似度默认由 Apple NaturalLanguage 词向量（本地 CPU）计算，
/// 不调用 embedding 模型；不可用时退化为词重叠相似度（仍可用，只是更粗糙）。
struct SemanticChunker {

    /// 可注入的相邻句相似度（测试用）；返回 nil 表示不可用。
    typealias AdjacentSimilarity = ([String]) -> [Double]?

    static func split(_ text: String, options: ChunkingOptions,
                      similarity: @escaping AdjacentSimilarity = {
                          NLSentenceSimilarity().adjacentCosines($0)
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
            AgentLog.warning("语义切片 Apple NL 词向量不可用，回退词法相似度")
            similarities = (0..<(sentences.count - 1)).map {
                lexicalSimilarity(sentences[$0], sentences[$0 + 1])
            }
        }

        let threshold = adaptiveThreshold(similarities, fallback: options.semanticThreshold)
        var chunks: [String] = []
        var current: [String] = []
        var currentTokens = 0

        func flush() {
            let value = current.joined(separator: current.first?.hasSuffix("\n") == true ? "" : " ")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !value.isEmpty { chunks.append(value) }
            current.removeAll(keepingCapacity: true)
            currentTokens = 0
        }

        for index in sentences.indices {
            current.append(sentences[index])
            currentTokens += RecursiveChunker.tokenCount(sentences[index])

            let hasNext = index + 1 < sentences.count
            let overBudget = currentTokens >= options.chunkSizeTokens ||
                current.joined(separator: " ").count >= options.maxChunkChars
            let boundary = hasNext && similarities[index] <= threshold
            if overBudget || boundary {
                flush()
            }
        }
        flush()

        guard !chunks.isEmpty else { return nil }
        // 单块仍超限 → 交给递归切片，避免返回一个巨型 chunk。
        if chunks.count == 1,
           RecursiveChunker.tokenCount(chunks[0]) > options.chunkSizeTokens {
            return RecursiveChunker.split(text, options: options)
        }
        return chunks
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

    /// 用相似度分布的低分位作为断点阈值：语义切分关心「相对低谷」，
    /// 固定阈值在长文档/短句上会要么切得过碎、要么完全不断。
    private static func adaptiveThreshold(_ values: [Double], fallback: Double) -> Double {
        guard values.count >= 4 else { return fallback }
        let sorted = values.sorted()
        let index = max(0, min(sorted.count - 1, Int(Double(sorted.count) * 0.25)))
        return max(0.15, min(fallback, sorted[index]))
    }
}
