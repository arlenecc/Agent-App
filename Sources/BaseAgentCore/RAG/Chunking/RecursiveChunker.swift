import Foundation

/// 递归回退切片：段落 → 换行 → 句号 → 空格 → token 窗口。
/// 正文每片不超过 `contentBudgetTokens`（已为 overlap 预留预算），
/// 不负责 overlap（由 TextChunker 统一在最终阶段添加，避免重叠复合）。
struct RecursiveChunker {
    private static let separators = ["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ", ""]

    static func split(_ text: String, options: ChunkingOptions) -> [String] {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return [] }
        let budget = options.contentBudgetTokens
        // 先按 token 预算硬切，防止单行超长（代码/表格）。
        let capped = splitByTokens(text, maxTokens: budget)
        var output: [String] = []
        for piece in capped {
            output.append(contentsOf: splitRecursive(piece, separators: separators,
                                                     options: options, depth: 0))
        }
        return output.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    private static func splitRecursive(_ text: String, separators: [String],
                                       options: ChunkingOptions, depth: Int) -> [String] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        let budget = options.contentBudgetTokens
        if options.tokenCounter(trimmed) <= budget {
            return [trimmed]
        }
        guard depth < separators.count else {
            return splitByTokens(trimmed, maxTokens: budget)
        }

        let sep = separators[depth]
        if sep.isEmpty {
            return splitByTokens(trimmed, maxTokens: budget)
        }
        guard trimmed.contains(sep) else {
            return splitRecursive(trimmed, separators: separators, options: options, depth: depth + 1)
        }

        let parts = componentsKeeping(trimmed, separator: sep)
        var chunks: [String] = []
        var current = ""

        func flush() {
            let value = current.trimmingCharacters(in: .whitespacesAndNewlines)
            if !value.isEmpty { chunks.append(value) }
            current = ""
        }

        for part in parts {
            let candidate = current + part
            if options.tokenCounter(candidate) <= budget {
                current = candidate
                continue
            }
            flush()
            // 单个 part 仍然超限 → 用更细的分隔符继续递归。
            if options.tokenCounter(part) > budget {
                chunks.append(contentsOf: splitRecursive(part, separators: separators,
                                                         options: options, depth: depth + 1))
            } else {
                current = part
            }
        }
        flush()
        return chunks
    }

    /// 按分隔符切分，但把分隔符保留在前一段末尾。
    /// 若直接 `components(separatedBy:)` 再拼回分隔符，最后一段会丢掉结尾的
    /// 句号/换行，切片就变成「半句」——这正是要求切片在句界断开时要避免的。
    private static func componentsKeeping(_ text: String, separator: String) -> [String] {
        guard !separator.isEmpty else { return [text] }
        var parts: [String] = []
        var start = text.startIndex
        while let range = text.range(of: separator, range: start..<text.endIndex) {
            parts.append(String(text[start..<range.upperBound]))
            start = range.upperBound
        }
        if start < text.endIndex {
            parts.append(String(text[start...]))
        }
        return parts
    }

    /// 按 token 预算硬切；到达预算时优先回退到自然句界，避免从半句断开。
    private static func splitByTokens(_ text: String, maxTokens: Int) -> [String] {
        guard !text.isEmpty else { return [] }
        let budget = max(1, maxTokens)
        var chunks: [String] = []
        var current = ""
        var used = 0.0
        for char in text {
            current.append(char)
            used += char.unicodeScalars.reduce(0.0) { $0 + EmbeddingTokenEstimator.weight($1) }
            if used >= Double(budget) {
                let breakIndex = lastNaturalBreak(in: current)
                if breakIndex > current.count / 2 {
                    let head = String(current.prefix(breakIndex))
                    let tail = String(current.dropFirst(breakIndex))
                    if !head.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        chunks.append(head)
                    }
                    current = tail
                    used = tail.unicodeScalars.reduce(0.0) { $0 + EmbeddingTokenEstimator.weight($1) }
                } else {
                    chunks.append(current)
                    current = ""
                    used = 0
                }
            }
        }
        if !current.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            chunks.append(current)
        }
        return chunks
    }

    private static func lastNaturalBreak(in text: String) -> Int {
        let chars = Array(text)
        guard !chars.isEmpty else { return 0 }
        let lower = max(1, chars.count / 2)
        var index = chars.count - 1
        while index > lower {
            let ch = chars[index]
            if ch == "\n" || ch == "。" || ch == "！" || ch == "？" || ch == "." {
                return index + 1
            }
            index -= 1
        }
        return 0
    }
}
