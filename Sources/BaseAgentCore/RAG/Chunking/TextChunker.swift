import Foundation

/// 文本切片主入口：结构优先 → 语义切分 → 递归回退。
/// 对应原项目 hybrid_chunker / semantic_chunker / chunker 的三级链路。
public final class TextChunker: TextChunkingStrategy {
    public let options: ChunkingOptions
    private let similarity: SemanticChunker.AdjacentSimilarity

    public init(options: ChunkingOptions) {
        self.options = options
        self.similarity = { PotionSentenceSimilarity().adjacentCosines($0) }
    }

    /// 测试注入相似度实现，避免依赖内置模型。
    init(options: ChunkingOptions,
         similarity: @escaping SemanticChunker.AdjacentSimilarity) {
        self.options = options
        self.similarity = similarity
    }

    /// 协议要求的签名（无进度回调）。
    public func chunk(document: ParsedDocument, variant: ChunkVariant) async -> [Chunk] {
        await chunk(document: document, variant: variant, progress: nil)
    }

    public func chunk(document: ParsedDocument,
                      variant: ChunkVariant = .hybrid,
                      progress: ((Int, Int) -> Void)? = nil) async -> [Chunk] {
        let text = document.text
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return [] }

        let nodes = DocumentOutline.extract(text, source: document.source)
        let sections = SectionSplitter.sections(text, nodes: nodes)
        var raw: [(text: String, heading: String,
                   sectionIndex: Int, mergeable: Bool)] = []

        for (sectionIndex, section) in sections.enumerated() {
            let split = await splitSection(section.text, variant: variant)
            guard !split.texts.isEmpty else { continue }
            // 章节正文本身已包含标题行；不要再拼接 heading path，否则
            // chunk 不再是原文连续子串，且标题会重复。
            for piece in split.texts {
                raw.append((piece, section.heading, sectionIndex, split.mergeable))
            }
            progress?(sectionIndex + 1, sections.count)
        }

        guard !raw.isEmpty else { return [] }
        let withOverlap = applyOverlap(raw, overlapTokens: options.overlapTokens)
        var chunks: [Chunk] = []
        for item in withOverlap {
            let trimmed = item.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            // 分页标记与页码一律不进切片：溯源只保留文件名（source），
            // 不再记录页码。
            let clean = DocumentCleaner.stripPageMarkers(trimmed)
            guard !clean.isEmpty else { continue }
            chunks.append(Chunk(text: clean, source: document.source,
                                chunkIndex: chunks.count, heading: item.heading))
        }
        return chunks
    }

    // MARK: Section

    /// 单个章节的切分结果。
    /// `mergeable` 表示片段是否允许与相邻片段合并：整章/语义切片是完整语义单元
    /// （过短的语义片已在 `SemanticChunker` 内部沿弱边界合并），绝不能再被
    /// `minChunkTokens` 合并回去，否则语义边界会被抹掉。
    private struct SectionPieces {
        var texts: [String]
        var mergeable: Bool
    }

    private func splitSection(_ text: String, variant: ChunkVariant) async -> SectionPieces {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return SectionPieces(texts: [], mergeable: false) }

        // 短章节整章即一个切片（按 token 预算对齐切片大小设置，默认 800 token）。
        // 长章节必须走语义/递归切分，禁止退化为按页切。
        if options.tokenCounter(trimmed) <= options.chunkSizeTokens {
            return SectionPieces(texts: [trimmed], mergeable: false)
        }

        switch variant {
        case .recursive, .structured:
            return SectionPieces(texts: RecursiveChunker.split(trimmed, options: options),
                                 mergeable: true)
        case .semantic:
            if let semantic = await SemanticChunker.split(trimmed, options: options,
                                                          similarity: similarity),
               !semantic.isEmpty {
                return SectionPieces(texts: enforceMaxTokens(semantic), mergeable: false)
            }
            return SectionPieces(texts: RecursiveChunker.split(trimmed, options: options),
                                 mergeable: true)
        case .hybrid:
            if options.enableSemantic {
                if let semantic = await SemanticChunker.split(trimmed, options: options,
                                                              similarity: similarity),
                   !semantic.isEmpty {
                    return SectionPieces(texts: enforceMaxTokens(semantic), mergeable: false)
                }
            }
            return SectionPieces(texts: RecursiveChunker.split(trimmed, options: options),
                                 mergeable: true)
        }
    }

    private func enforceMaxTokens(_ pieces: [String]) -> [String] {
        var output: [String] = []
        for piece in pieces {
            if options.tokenCounter(piece) <= options.chunkSizeTokens {
                output.append(piece)
            } else {
                output.append(contentsOf: RecursiveChunker.split(piece, options: options))
            }
        }
        return output
    }

    // MARK: Overlap

    /// 把上一块尾部约 `overlapTokens` token 的内容前置到当前块。
    /// 使用「原始块」而不是已加重叠的块，避免重叠逐段复合膨胀。
    ///
    /// 合并与 overlap 都严格限制在同一章节内、且片段本身允许合并：
    /// 整章切片与语义切片是完整语义单元，不能被 `minChunkTokens` 或 overlap
    /// 跨边界拼接（否则语义边界会被抹掉，或重新退化成按页/跨主题切片）。
    private func applyOverlap(
        _ pieces: [(text: String, heading: String,
                    sectionIndex: Int, mergeable: Bool)],
        overlapTokens: Int
    ) -> [(text: String, heading: String)] {
        guard !pieces.isEmpty else { return [] }

        // 先合并过短块，再做 overlap：反过来会让 overlap 后的块再次被合并，
        // 最终突破 chunkSize 预算。
        var merged = pieces
        var index = 1
        while index < merged.count {
            let piece = merged[index]
            var previous = merged[index - 1]
            if piece.mergeable, previous.mergeable,
               piece.sectionIndex == previous.sectionIndex,
               options.tokenCounter(piece.text) < options.minChunkTokens,
               options.tokenCounter(previous.text + "\n" + piece.text) <= options.contentBudgetTokens {
                previous.text += "\n" + piece.text
                merged[index - 1] = previous
                merged.remove(at: index)
            } else {
                index += 1
            }
        }

        guard overlapTokens > 0, merged.count > 1 else {
            return merged.map { (text: $0.text, heading: $0.heading) }
        }
        var output = [merged[0]]
        for index in 1..<merged.count {
            let previous = merged[index - 1]
            let current = merged[index]
            let sameSection = previous.sectionIndex == current.sectionIndex
            let canOverlap = sameSection && previous.mergeable && current.mergeable
            let tail = canOverlap ? overlapTail(previous.text, tokens: overlapTokens) : ""
            if tail.isEmpty || current.text.hasPrefix(tail) {
                output.append(current)
            } else if options.tokenCounter(tail + "\n" + current.text) <= options.chunkSizeTokens {
                var value = current
                value.text = tail + "\n" + current.text
                output.append(value)
            } else {
                // 当前块已接近 token 预算：宁可不要 overlap，也不产生超长块。
                output.append(current)
            }
        }
        return output.map { (text: $0.text, heading: $0.heading) }
    }

    /// 取尾部约 `tokens` token 的文本，并向后挪到自然句界，避免从半句开始。
    private func overlapTail(_ text: String, tokens: Int) -> String {
        guard tokens > 0, !text.isEmpty else { return "" }
        if options.tokenCounter(text) <= tokens {
            let value = text.trimmingCharacters(in: .whitespacesAndNewlines)
            return value.count >= 12 ? value : ""
        }
        let raw = EmbeddingTokenEstimator.suffix(text, tokens: tokens)
        let all = Array(raw)
        var start = 0
        // 向后挪到自然句界/空白，避免从半个词开始。
        let limit = min(all.count, 40)
        while start < limit {
            let ch = all[start]
            if ch == "\n" || ch == "。" || ch == "！" || ch == "？" || ch == "." || ch == " " {
                start += 1
                break
            }
            start += 1
        }
        let tail = String(all[start...]).trimmingCharacters(in: .whitespacesAndNewlines)
        return tail.count >= 12 ? tail : ""
    }
}
