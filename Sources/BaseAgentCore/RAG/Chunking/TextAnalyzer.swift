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

    /// 构造 FTS5 MATCH 表达式：带引号的 OR 查询，避免特殊字符注入语法错误。
    public static func ftsQuery(_ query: String) -> String {
        let tokens = Array(Set(ftsTokens(query))).sorted()
        guard !tokens.isEmpty else { return "" }
        return tokens.map { term -> String in
            let escaped = term.replacingOccurrences(of: "\"", with: "\"\"")
            return "\"\(escaped)\""
        }.joined(separator: " OR ")
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
