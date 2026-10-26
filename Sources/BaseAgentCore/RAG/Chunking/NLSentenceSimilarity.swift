import Foundation
import NaturalLanguage

/// 句子相似度：用 Apple NaturalLanguage 的词向量（本地 CPU）求均值后算余弦。
/// 语义切片不再调用 embedding 模型，避免与切片后的向量化争抢同一实例。
struct NLSentenceSimilarity {

    /// 相邻句余弦相似度；词向量不可用或覆盖不足时返回 nil，由调用方回退词法相似度。
    func adjacentCosines(_ sentences: [String]) -> [Double]? {
        guard sentences.count >= 2 else { return nil }
        let language: NLLanguage = Self.prefersChinese(sentences.joined())
            ? .simplifiedChinese : .english
        guard let embedding = Self.embedding(for: language) else { return nil }

        var vectors: [[Double]] = []
        vectors.reserveCapacity(sentences.count)
        for (index, sentence) in sentences.enumerated() {
            if index % 32 == 0, Task.isCancelled { return nil }
            guard let vector = Self.averageVector(of: sentence, embedding: embedding) else {
                return nil
            }
            vectors.append(vector)
        }
        return (0..<(vectors.count - 1)).map {
            Self.cosine(vectors[$0], vectors[$0 + 1])
        }
    }

    // MARK: 内部

    /// 中文字符占比超过 15% 视为中文文本（中英混排也走中文词表）。
    private static func prefersChinese(_ text: String) -> Bool {
        var han = 0
        var letters = 0
        for scalar in text.unicodeScalars {
            if (0x4E00...0x9FFF).contains(scalar.value) {
                han += 1
            } else if CharacterSet.letters.contains(scalar) {
                letters += 1
            }
        }
        guard han > 0 else { return false }
        return Double(han) / Double(han + letters) >= 0.15
    }

    private static let cache = NSCache<NSString, NLEmbedding>()

    private static func embedding(for language: NLLanguage) -> NLEmbedding? {
        let key = language.rawValue as NSString
        if let cached = cache.object(forKey: key) { return cached }
        guard let value = NLEmbedding.wordEmbedding(for: language) else { return nil }
        cache.setObject(value, forKey: key)
        return value
    }

    /// 句子向量 = 已知词向量的均值。
    /// 已知词过少（少于 2 个或覆盖率低于 25%）视为不可用，避免噪声向量误导边界。
    private static func averageVector(of sentence: String,
                                      embedding: NLEmbedding) -> [Double]? {
        let tokens = TextAnalyzer.words(sentence)
        guard !tokens.isEmpty else { return nil }
        let dimension = embedding.dimension
        var sum = [Double](repeating: 0, count: dimension)
        var known = 0
        for token in tokens {
            guard let vector = embedding.vector(for: token),
                  vector.count == dimension else { continue }
            for index in 0..<dimension { sum[index] += vector[index] }
            known += 1
        }
        guard known >= 2, Double(known) / Double(tokens.count) >= 0.25 else {
            return nil
        }
        return sum.map { $0 / Double(known) }
    }

    private static func cosine(_ a: [Double], _ b: [Double]) -> Double {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot = 0.0
        var normA = 0.0
        var normB = 0.0
        for index in 0..<a.count {
            dot += a[index] * b[index]
            normA += a[index] * a[index]
            normB += b[index] * b[index]
        }
        guard normA > 0, normB > 0 else { return 0 }
        return dot / (sqrt(normA) * sqrt(normB))
    }
}
