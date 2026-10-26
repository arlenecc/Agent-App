import Foundation

/// 递归回退切片：段落 → 换行 → 句号 → 空格 → 字符窗口。
/// 保证每片不超过 `chunkSizeTokens` 与 `maxChunkChars`，不负责 overlap
/// （由 TextChunker 统一在最终阶段添加，避免重叠复合）。
struct RecursiveChunker {
    private static let separators = ["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ", ""]

    static func split(_ text: String, options: ChunkingOptions) -> [String] {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return [] }
        // 先按最大字符数硬切，防止单行超长（代码/表格）。
        let capped = splitByCharacters(text, maxChars: max(200, options.maxChunkChars),
                                       chunkSize: options.chunkSizeTokens)
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
        if tokenCount(trimmed) <= options.chunkSizeTokens && trimmed.count <= options.maxChunkChars {
            return [trimmed]
        }
        guard depth < separators.count else {
            return splitByCharacters(trimmed, maxChars: options.maxChunkChars,
                                     chunkSize: options.chunkSizeTokens)
        }

        let sep = separators[depth]
        if sep.isEmpty {
            return splitByCharacters(trimmed, maxChars: options.maxChunkChars,
                                     chunkSize: options.chunkSizeTokens)
        }
        guard trimmed.contains(sep) else {
            return splitRecursive(trimmed, separators: separators, options: options, depth: depth + 1)
        }

        let parts = trimmed.components(separatedBy: sep)
        var chunks: [String] = []
        var current = ""

        func flush() {
            let value = current.trimmingCharacters(in: .whitespacesAndNewlines)
            if !value.isEmpty { chunks.append(value) }
            current = ""
        }

        for part in parts {
            let candidate = current.isEmpty ? part : current + sep + part
            if tokenCount(candidate) <= options.chunkSizeTokens && candidate.count <= options.maxChunkChars {
                current = candidate
                continue
            }
            flush()
            // 单个 part 仍然超限 → 用更细的分隔符继续递归。
            if tokenCount(part) > options.chunkSizeTokens || part.count > options.maxChunkChars {
                chunks.append(contentsOf: splitRecursive(part, separators: separators,
                                                         options: options, depth: depth + 1))
            } else {
                current = part
            }
        }
        flush()
        return chunks
    }

    private static func splitByCharacters(_ text: String, maxChars: Int, chunkSize: Int) -> [String] {
        guard !text.isEmpty else { return [] }
        var chunks: [String] = []
        var current = ""
        for char in text {
            current.append(char)
            let overChars = current.count >= maxChars
            let overTokens = current.count % 64 == 0 && tokenCount(current) >= chunkSize
            if overChars || overTokens {
                // 在自然边界处回退断句。
                let breakIndex = lastNaturalBreak(in: current)
                if breakIndex > current.count / 2 {
                    let head = String(current.prefix(breakIndex))
                    let tail = String(current.dropFirst(breakIndex))
                    if !head.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        chunks.append(head)
                    }
                    current = tail
                } else {
                    chunks.append(current)
                    current = ""
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

    static func tokenCount(_ text: String) -> Int {
        max(1, TokenEstimator.estimate(text))
    }
}
