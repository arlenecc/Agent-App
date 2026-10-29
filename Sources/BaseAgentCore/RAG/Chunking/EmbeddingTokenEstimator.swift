import Foundation

/// 近似中文分词器的 token 计数器（按 bge 词表标定，对当前 embedding 模型偏保守）。
///
/// 切片发生在 embedding 之前，用标定过的估算器控制预算；估算取偏保守的上界
/// （实测中文、英文、代码、哈希、数字表格等语料，估算均 ≥ 实际 token），
/// 宁可切小一点，也不让块超出 embedding 模型的输入窗口。
/// 实际 ingest 时若模型分词器可用，`RAGEngine` 会改用精确计数。
public enum EmbeddingTokenEstimator {
    /// 估算文本 token 数（空文本为 0）。
    public static func count(_ text: String) -> Int {
        guard !text.isEmpty else { return 0 }
        var total = 0.0
        for scalar in text.unicodeScalars {
            total += weight(scalar)
        }
        return max(1, Int(total.rounded(.up)))
    }

    /// 单个 Unicode 标量的近似 token 权重（偏保守）。
    public static func weight(_ scalar: Unicode.Scalar) -> Double {
        switch scalar.value {
        case 0x4E00...0x9FFF,   // CJK 统一表意文字
             0x3400...0x4DBF,   // 扩展 A
             0xF900...0xFAFF,   // 兼容表意文字
             0x3040...0x30FF,   // 日文假名
             0xAC00...0xD7AF,   // 谚文
             0x3000...0x303F,   // 中文标点（、。「」等）
             0x20000...0x2FA1F: // 扩展 B 及以后
            return 1.0
        case 0xFF00...0xFFEF:
            return 1.1           // 全角字符
        case 0x30...0x39, 0x41...0x5A, 0x61...0x7A:
            return 0.7           // ASCII 字母与数字：按最坏情况（哈希/编号）取整
        case 0x20, 0x09, 0x0A, 0x0D:
            return 0.25          // 空白通常并入相邻词
        case 0x2000...0x206F:
            return 1.1           // 中文引号、省略号、破折号
        case 0x1F000...0x1FAFF, 0x2600...0x27BF:
            return 2.0           // emoji / 装饰符号
        case 0x21...0x2F, 0x3A...0x40, 0x5B...0x60, 0x7B...0x7E:
            return 1.0           // ASCII 标点：代码/URL 中多为单独 token
        default:
            return 0.7           // 带音标拉丁、西里尔等
        }
    }

    /// 按 token 预算取尾部文本（按标量累加），不处理句界。
    static func suffix(_ text: String, tokens: Int) -> String {
        guard tokens > 0, !text.isEmpty else { return "" }
        var used = 0.0
        var index = text.endIndex
        while index > text.startIndex {
            let previous = text.index(before: index)
            used += text[previous].unicodeScalars.reduce(0.0) { $0 + weight($1) }
            index = previous
            if used >= Double(tokens) { break }
        }
        return String(text[index...])
    }
}
