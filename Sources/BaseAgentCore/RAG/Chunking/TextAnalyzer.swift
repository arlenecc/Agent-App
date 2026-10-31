import Foundation
import NaturalLanguage

/// 基于 Apple NaturalLanguage 的统一文本分析器：
/// - 句子切分（chunking）
/// - 词/词组切分（FTS5 预分词 + 关键词查询）
///
/// SQLite FTS5 的 unicode61 分词器对中文只会按连续 CJK 串切分，
/// 因此这里用 NLTokenizer 先做词级切分，并对 CJK 追加 bigram，
/// 让关键词召回在中文上仍然可用。
public enum TextAnalyzer {

    // MARK: 句子切分

    /// 按自然语言句子切分，保留原文（不裁剪空白之外的字符）。
    public static func sentences(_ text: String) -> [String] {
        guard !text.isEmpty else { return [] }
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = text
        var result: [String] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let sentence = String(text[range]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !sentence.isEmpty { result.append(sentence) }
            return true
        }
        if result.isEmpty {
            let fallback = text.split(whereSeparator: \.isNewline)
                .map { String($0).trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }
            return fallback
        }
        return result
    }

    /// 词级切分（去标点、统一小写）。
    public static func words(_ text: String) -> [String] {
        guard !text.isEmpty else { return [] }
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.string = text
        var result: [String] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let token = String(text[range]).lowercased()
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !token.isEmpty, token.unicodeScalars.contains(where: isWordScalar) {
                result.append(token)
            }
            return true
        }
        return result
    }

    // MARK: FTS5 预分词

    /// 写入 FTS 与查询共用的 token 序列。
    /// CJK 词额外展开 bigram，改善「整词不匹配 → 完全召回不到」的问题。
    public static func ftsTokens(_ text: String) -> [String] {
        let base = words(text)
        var output: [String] = []
        output.reserveCapacity(base.count * 2)
        for token in base {
            output.append(token)
            if containsCJK(token), token.count > 2 {
                let chars = Array(token)
                for i in 0..<(chars.count - 1) {
                    output.append(String(chars[i...i + 1]))
                }
            }
        }
        return output
    }

    public static func ftsIndexText(_ text: String) -> String {
        ftsTokens(text).joined(separator: " ")
    }

    /// 批量预分词：切片之间相互独立，按索引并行回填，输出与逐条 `ftsIndexText` 一致。
    /// 写入路径在拿到存储写锁之前调用，避免整批分词阻塞检索。
    static func ftsIndexTexts(_ texts: [String]) -> [String] {
        guard !texts.isEmpty else { return [] }
        // 小批量串行：线程调度开销高于分词本身。
        guard texts.count >= 16 else { return texts.map(ftsIndexText) }
        var output = [String](repeating: "", count: texts.count)
        output.withUnsafeMutableBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return }
            DispatchQueue.concurrentPerform(iterations: texts.count) { index in
                base[index] = ftsIndexText(texts[index])
            }
        }
        return output
    }

    /// 查询词里最多参与 MATCH 的显著 token 数。自然语言问句常带几十个词，
    /// 全量交给 AND 会几乎不可能命中；按信息量截断后再逐级回退 OR。
    public static let maxSignificantTokens = 32

    public enum FTSQueryMode: Sendable {
        case or
        case and
    }

    /// 停用词：高频虚词几乎不提供区分度，参与 AND 只会把结果筛空。
    private static let stopwords: Set<String> = [
        // 英文
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
        "do", "does", "did", "for", "from", "had", "has", "have", "he", "her",
        "his", "how", "i", "if", "in", "into", "is", "it", "its", "me", "my",
        "no", "not", "of", "on", "or", "our", "she", "so", "than", "that",
        "the", "their", "them", "then", "there", "these", "they", "this",
        "those", "to", "was", "we", "were", "what", "when", "where", "which",
        "who", "why", "will", "with", "would", "you", "your",
        // 中文功能词 / 常见填充词
        "的", "了", "着", "过", "是", "在", "和", "与", "及", "或", "也",
        "就", "都", "而", "被", "把", "对", "从", "向", "给", "为", "以",
        "于", "其", "之", "这", "那", "这个", "那个", "这些", "那些",
        "一个", "一些", "什么", "怎么", "怎样", "如何", "为什么", "哪",
        "哪些", "我", "我们", "你", "你们", "他", "他们", "她", "她们",
        "它", "它们", "可以", "能够", "应该", "已经", "正在", "还是",
        "但是", "因为", "所以", "如果", "然后", "以及", "对于", "关于",
        "通过", "进行", "使用", "目前", "其中", "并", "并且", "且", "等",
        "等等", "吗", "呢", "吧", "啊", "呀", "哦",
    ]

    /// 查询用显著 token：去停用词、去重，按词长（信息量）取前 `limit` 个。
    ///
    /// - 长 CJK 词只保留 bigram：分词边界不稳定，整词 AND 常常因文档里
    ///   只有更长/更短的词而全灭；bigram 已在 `ftsTokens` 里展开。
    public static func ftsSignificantTokens(_ text: String,
                                            limit: Int = maxSignificantTokens) -> [String] {
        guard limit > 0 else { return [] }
        var seen = Set<String>()
        var tokens: [String] = []
        for token in ftsTokens(text) {
            guard !stopwords.contains(token) else { continue }
            if containsCJK(token), token.count > 2 { continue }
            guard seen.insert(token).inserted else { continue }
            tokens.append(token)
        }
        guard tokens.count > limit else { return tokens }
        // 长词信息量更高；同长度保持原顺序（稳定排序）。
        return tokens.enumerated()
            .sorted { lhs, rhs in
                lhs.element.count == rhs.element.count
                    ? lhs.offset < rhs.offset
                    : lhs.element.count > rhs.element.count
            }
            .prefix(limit)
            .map(\.element)
    }

    /// 构造 FTS5 MATCH 表达式：所有 token 加引号，避免特殊字符注入语法错误。
    /// `mode` 决定 token 之间是 AND 还是 OR；调用方通常先试 AND，无结果再退 OR。
    public static func ftsQuery(_ query: String,
                                mode: FTSQueryMode = .or,
                                limit: Int = maxSignificantTokens) -> String {
        let tokens = ftsSignificantTokens(query, limit: limit)
        guard !tokens.isEmpty else { return "" }
        let joiner = mode == .and ? " AND " : " OR "
        return tokens.map { term -> String in
            let escaped = term.replacingOccurrences(of: "\"", with: "\"\"")
            return "\"\(escaped)\""
        }.joined(separator: joiner)
    }

    // MARK: Helpers

    private static func isWordScalar(_ scalar: Unicode.Scalar) -> Bool {
        if CharacterSet.alphanumerics.contains(scalar) { return true }
        switch scalar.value {
        case 0x4E00...0x9FFF, 0x3400...0x4DBF, 0x3040...0x30FF, 0xAC00...0xD7AF:
            return true
        default:
            return false
        }
    }

    private static func containsCJK(_ text: String) -> Bool {
        text.unicodeScalars.contains { scalar in
            switch scalar.value {
            case 0x4E00...0x9FFF, 0x3400...0x4DBF:
                return true
            default:
                return false
            }
        }
    }
}
