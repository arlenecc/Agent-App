import Foundation

/// 纯文本结构还原器：对无标记的纯文本，通过目录块、编号模式、
/// 标题与段落的格式差异（长度对比、句末标点、独立成块）等启发式信号，
/// 还原标题层级结构，并把纯文本转换成 Markdown。
///
/// 转换后的 Markdown 会自然进入 `DocumentOutline.extract` 与切片管线，
/// 因此下游（digest、SectionSplitter、heading path）无需感知还原过程。
public enum PlainTextStructure {

    public struct Result {
        public let markdown: String
        public let entries: [OutlineEntry]
        public let tocLinesRemoved: Int
    }

    /// 文本已带有 Markdown 标题结构（≥2 行 `#`）时无需还原。
    public static func isMarkdownLike(_ text: String) -> Bool {
        guard let regex = try? NSRegularExpression(
            pattern: "(?m)^[ \\t]{0,3}#{1,6}[ \\t]") else { return false }
        let ns = text as NSString
        let range = NSRange(location: 0, length: ns.length)
        var count = 0
        regex.enumerateMatches(in: text, options: [], range: range) { _, _, _ in
            count += 1
        }
        return count >= 2
    }

    // MARK: - 行模型

    private struct Line {
        let index: Int
        let raw: String
        let trimmed: String
        let indent: Int

        var isEmpty: Bool { trimmed.isEmpty }
        var displayLength: Int { trimmed.count }
    }

    private static func makeLines(_ text: String) -> [Line] {
        text.split(separator: "\n", omittingEmptySubsequences: false).enumerated().map {
            let raw = String($0.element)
            let leading = raw.prefix { $0 == " " || $0 == "\t" || $0 == "\u{3000}" }
            let indent = leading.reduce(0) { $0 + ($1 == "\u{3000}" ? 2 : 1) }
            return Line(index: $0.offset, raw: raw,
                        trimmed: raw.trimmingCharacters(in: .whitespacesAndNewlines),
                        indent: indent)
        }
    }

    // MARK: - 入口

    /// 还原纯文本结构。没有足够把握（无 TOC 且无强编号模式）时返回 nil，
    /// 调用方保持原文不做改写。
    public static func restore(_ raw: String) -> Result? {
        let lines = makeLines(raw)
        guard lines.count >= 10 else { return nil }

        var tocMarks = [Bool](repeating: false, count: lines.count)
        var tocEntries: [(index: Int, level: Int, title: String)] = []
        var tocHits = [Int: String]()
        detectTableOfContents(lines, marks: &tocMarks, entries: &tocEntries, hits: &tocHits)

        var headings = [Int: (level: Int, title: String)]()
        detectHeadings(lines, tocMarks: tocMarks, headings: &headings)

        // 强模式与 TOC 都没有 → 无法可靠还原，保持原样。
        guard !headings.isEmpty || !tocEntries.isEmpty else { return nil }

        var out: [String] = []
        out.reserveCapacity(lines.count)
        var removedToc = 0
        for line in lines {
            if tocMarks[line.index] {
                // 目录行转为 Markdown 列表：保留可检索的章节信息，
                // 但不产生标题结构（避免把目录页切成一堆小章节）。
                if let title = tocHits[line.index] {
                    out.append("- \(title)")
                    removedToc += 1
                } else if !line.isEmpty {
                    removedToc += 1
                }
                continue
            }
            if let heading = headings[line.index] {
                out.append(String(repeating: "#", count: max(1, min(6, heading.level)))
                           + " " + heading.title)
                continue
            }
            out.append(line.raw)
        }

        var entries = tocEntries.sorted { $0.index < $1.index }
            .map { OutlineEntry(level: $0.level, title: $0.title) }
        entries.append(contentsOf: headings.sorted { $0.key < $1.key }
            .map { OutlineEntry(level: $0.value.level, title: $0.value.title) })
        return Result(markdown: out.joined(separator: "\n"),
                      entries: entries,
                      tocLinesRemoved: removedToc)
    }

    // MARK: - 目录块识别

    /// 目录行：标题 + 点线/连续空白 + 页码。
    private static let tocPatterns: [(String, NSRegularExpression.Options)] = [
        // 「第一章 …… …… 12」：点线或空白 ≥2 后跟页码收尾。
        ("^(\\S.{0,78}?)[ \\t\\u00A0\\u3000.．·…‥⋯_—\\-]{2,}(\\d{1,4})[ \\t]*$",
         [.anchorsMatchLines]),
        // 「1.2 标题    33」：正文与页码之间 ≥3 个连续空白。
        ("^(\\S.{0,78}?)\\t?[ \\t]{3,}(\\d{1,4})[ \\t]*$", [.anchorsMatchLines]),
    ]

    /// 连续 ≥3 行目录行才视为目录块，避免把「……详见 42 页」类偶发行判进去。
    private static let minTocBlockLines = 3

    private static func detectTableOfContents(_ lines: [Line],
                                              marks: inout [Bool],
                                              entries: inout [(index: Int, level: Int, title: String)],
                                              hits: inout [Int: String]) {
        var found = [Int: (title: String, page: String)]()
        for line in lines where !line.isEmpty {
            guard line.displayLength <= 90,
                  !hasTrailingSentencePunctuation(line.trimmed) else { continue }
            let s = line.trimmed as NSString
            for (pattern, options) in tocPatterns {
                guard let regex = try? NSRegularExpression(pattern: pattern, options: options),
                      let match = regex.firstMatch(
                        in: line.trimmed, options: [],
                        range: NSRange(location: 0, length: s.length)),
                      match.numberOfRanges >= 3 else { continue }
                let title = s.substring(with: match.range(at: 1))
                    .trimmingCharacters(in: CharacterSet(
                        charactersIn: " \t\u{00A0}\u{3000}.．·…‥⋯_-—"))
                guard title.count >= 2, !title.allSatisfy({ $0.isNumber }) else { continue }
                found[line.index] = (title, s.substring(with: match.range(at: 2)))
                break
            }
        }
        guard found.count >= minTocBlockLines else { return }

        // 连续命中段（允许中间夹 ≤1 行空行/页眉）。
        var run: [Int] = []
        func flush() {
            guard run.count >= minTocBlockLines else {
                run.removeAll()
                return
            }
            for index in run {
                marks[index] = true
                if let hit = found[index] {
                    hits[index] = hit.title
                    entries.append((index, tocLevel(lines[index], title: hit.title), hit.title))
                }
            }
            run.removeAll()
        }
        for line in lines {
            if found[line.index] != nil {
                run.append(line.index)
            } else if line.isEmpty, let last = run.last, found[last] != nil {
                run.append(line.index) // 空行挂在段尾，flush 时不会产出条目
            } else {
                flush()
            }
        }
        flush()
    }

    /// 目录条目层级：优先编号模式，其次缩进深度。
    private static func tocLevel(_ line: Line, title: String) -> Int {
        if let classified = DocumentOutline.classify(title) { return classified.level }
        return max(1, min(4, line.indent / 2 + 1))
    }

    // MARK: - 正文标题识别

    private static func detectHeadings(_ lines: [Line],
                                       tocMarks: [Bool],
                                       headings: inout [Int: (level: Int, title: String)]) {
        var candidateLines: [Int] = []
        var weakCandidates = [Int: Int]() // 行号 -> 建议层级（不含强模式）

        for line in lines where !line.isEmpty && !tocMarks[line.index] {
            // PDF 分页标记（<!-- page N -->）不是文档结构，保持原样。
            if DocumentCleaner.isPageMarker(line.trimmed) { continue }
            // 页码/页眉（`2 / 164`、`第 2 页`）既不是强标题也不是弱标题。
            if DocumentOutline.isPageHeaderLine(line.trimmed) { continue }
            // 强模式：编号/章节词直接命中。
            if let classified = DocumentOutline.classify(line.trimmed) {
                headings[line.index] = classified
                continue
            }
            if let level = weakHeadingLevel(line, lines: lines) {
                weakCandidates[line.index] = level
                candidateLines.append(line.index)
            }
        }

        // 弱模式靠格式对比，天然易误判：占比超过 max(3, 8%) 时整体放弃，
        // 宁可漏掉也不要把正文切得支离破碎。
        let contentLines = lines.filter { !$0.isEmpty && !tocMarks[$0.index] }.count
        let weakLimit = max(3, Int(Double(contentLines) * 0.08))
        guard contentLines > 0, weakCandidates.count <= weakLimit else {
            return
        }

        // 相邻弱候选（中间只隔空行）构成「短行链」（诗歌/列表）——整链拒绝。
        let rejected = shortLineChainMembers(weakCandidates, lines: lines)
        var lastLevel = 1
        for index in candidateLines.sorted() {
            guard let suggested = weakCandidates[index], !rejected.contains(index) else { continue }
            let level = max(2, min(4, suggested))
            headings[index] = (level, cleanedTitle(lines[index].trimmed))
            lastLevel = level
            _ = lastLevel
        }
    }

    /// 找出短行链成员：弱候选与其相邻弱候选（中间只隔空行）链长 ≥2 时整链拒绝。
    private static func shortLineChainMembers(_ candidates: [Int: Int],
                                              lines: [Line]) -> Set<Int> {
        let sorted = candidates.keys.sorted()
        var rejected = Set<Int>()
        var chain: [Int] = []
        func flush() {
            if chain.count >= 2 { rejected.formUnion(chain) }
            chain.removeAll()
        }
        for index in sorted {
            if let previous = chain.last {
                // 与上一候选之间只有空行 → 同一条链。
                let between = lines[(previous + 1)..<index]
                if between.allSatisfy({ $0.isEmpty }) {
                    chain.append(index)
                    continue
                }
            }
            flush()
            chain = [index]
        }
        flush()
        return rejected
    }

    /// 弱标题判定：短、无句末标点、独立成块（邻居是空行或明显更长的段落行）。
    private static func weakHeadingLevel(_ line: Line, lines: [Line]) -> Int? {
        let text = line.trimmed
        guard text.count >= 2, text.count <= 40 else { return nil }
        // 页码/页眉不是标题。
        guard !DocumentOutline.isPageHeaderLine(text) else { return nil }
        // 纯数字/页码/分隔线不是标题。
        guard !text.allSatisfy({ $0.isNumber || $0.isWhitespace
            || "-—=_.·•*#[]()（）/／".contains($0) })
        else { return nil }

        // 全角括号/书名号整体包裹的短行是强标题信号（【第X节】……）。
        let bracketed = (text.hasPrefix("【") && text.hasSuffix("】"))
            || (text.hasPrefix("〔") && text.hasSuffix("〕"))

        guard !hasSentencePunctuationBody(text) || bracketed else { return nil }
        // 句中含逗号/分号几乎总是正文；允许书名号、引号、冒号。
        if (text.contains("，") || text.contains(",")) && !bracketed { return nil }

        guard let prev = neighbor(lines, index: line.index, offset: -1),
              let next = neighbor(lines, index: line.index, offset: 1) else { return nil }

        // 标题与正文的分隔信号：一侧有空行，或两侧都是明显更长的段落行
        // （PDF 文本层按排版行提取，段落间通常没有空行）。
        let prevFree = prev.isEmpty || line.index == 0
        let nextFree = next.isEmpty
        let separated = prevFree || nextFree
        let longBothSides = prev.displayLength >= 30
            && next.displayLength >= 30
            && prev.displayLength > text.count * 2
            && next.displayLength > text.count * 2
        guard separated || longBothSides else { return nil }

        let longNeighbor = (prev.isEmpty ? false : prev.displayLength >= 50)
            || (next.isEmpty ? false : next.displayLength >= 50)
        // 空行场景：另一侧也是短行（上下文全是短行堆）时不可信。
        if separated, !bracketed, !longNeighbor, !(prevFree && nextFree) {
            return nil
        }
        let neighborShort = (!prev.isEmpty && prev.displayLength <= 40
                             && !hasSentencePunctuationBody(prev.trimmed))
            || (!next.isEmpty && next.displayLength <= 40
                && !hasSentencePunctuationBody(next.trimmed))
        // 两侧都是短行 → 连续短行堆（诗歌/列表/通讯录），不判标题。
        if !bracketed, neighborShort, separated, !(prevFree && longNeighbor) { return nil }
        if !bracketed, !longBothSides, !longNeighbor, !(prevFree && nextFree) { return nil }

        if bracketed { return 2 }
        // 有编号残留提示时靠近上级；否则默认次级标题。
        return 2
    }

    private static func neighbor(_ lines: [Line], index: Int, offset: Int) -> Line? {
        let target = index + offset
        guard target >= 0, target < lines.count else { return nil }
        return lines[target]
    }

    /// 清理标题文本：去掉尾部冒号、收尾空白。
    private static func cleanedTitle(_ text: String) -> String {
        var t = text
        while t.hasSuffix("：") || t.hasSuffix(":") || t.hasSuffix(" ")
            || t.hasSuffix("　") {
            t.removeLast()
        }
        return t
    }

    private static func hasSentencePunctuationBody(_ text: String) -> Bool {
        text.contains("。") || text.contains("！") || text.contains("？")
            || text.contains("；") || text.contains("…")
    }

    private static func hasTrailingSentencePunctuation(_ text: String) -> Bool {
        guard let last = text.last else { return false }
        return "。！？；，,.!?；".contains(last)
    }
}
