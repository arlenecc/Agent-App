import AppKit
import Foundation
import SwiftSoup

/// AppKit 的同步转换无法被中断，只能在阶段边界和循环内主动观察取消信号。
private func checkWordCancellation() throws {
    if Task.isCancelled { throw AgentError.cancelled }
}

/// Word 97–2003 `.doc` 提取。
///
/// 文本、字体、表格和列表格式交给 AppKit 的 OLE2 Word 导入器；标题没有
/// 可靠的命名样式，只能用「字号明显大于正文 + 短段落」启发式识别，
/// 再把导出的 HTML 中对应 `<p>` 提升为 `<h1>`–`<h3>`，复用 HTMLMarkdown。
struct LegacyWordExtractor {

    /// HTML 导出前允许的最大 UTF-16 字符数；默认与最终正文上限一致。
    var htmlCharacterLimit: Int = LegacyOfficeLimits.maxTextCharacters

    struct Output {
        let markdown: String
        let outline: [OutlineEntry]
        let metadata: [String: String]
    }

    private struct Paragraph {
        let text: String
        let fontSize: CGFloat
        let isBold: Bool
        let isItalic: Bool
        let isInTable: Bool
        let listMarker: String?
    }

    private struct HeadingCandidate {
        let text: String
        let level: Int
    }

    func extract(url: URL) throws -> Output {
        do {
            return try extractInternal(url: url)
        } catch let error as AgentError {
            throw error
        } catch {
            let message = error.localizedDescription
            let lower = message.lowercased()
            if lower.contains("password") || lower.contains("encrypt")
                || message.contains("密码") || message.contains("加密") {
                throw AgentError.unsupported(
                    "Word 文档已加密或受密码保护（\(url.lastPathComponent)）")
            }
            if lower.contains("not supported") || lower.contains("unsupported")
                || message.contains("不支持") {
                throw AgentError.unsupported(
                    "不支持该 Word 版本（\(url.lastPathComponent)）：\(message)")
            }
            throw AgentError.parse(
                "旧版 Word 解析失败（\(url.lastPathComponent)）：\(message)")
        }
    }

    static func htmlExportRange(for attributed: NSAttributedString,
                                characterLimit: Int = LegacyOfficeLimits.maxTextCharacters) -> NSRange {
        let string = attributed.string as NSString
        let length = LegacyOfficeLimits.safeUTF16PrefixLength(string, limit: characterLimit)
        return NSRange(location: 0, length: length)
    }

    private func extractInternal(url: URL) throws -> Output {
        try checkWordCancellation()
        let values = try url.resourceValues(forKeys: [.fileSizeKey])
        if let size = values.fileSize, size > LegacyOfficeLimits.maxFileSize {
            throw AgentError.parse(
                "Word 文档超过 \(LegacyOfficeLimits.maxFileSize / 1024 / 1024)MB 上限"
                + "（\(url.lastPathComponent)）")
        }

        let attributed = try autoreleasepool {
            try NSAttributedString(url: url, options: [:], documentAttributes: nil)
        }
        try checkWordCancellation()
        guard attributed.length > 0 else {
            throw AgentError.parse("Word 文档未解析出正文（\(url.lastPathComponent)）")
        }

        // HTML 导出的体积远大于正文字符数；先按最终正文上限截断，避免为一个
        // 注定会被 limitText 丢弃的巨型文档生成超大 HTML 再检查 64MB 上限。
        let exportRange = Self.htmlExportRange(for: attributed,
                                               characterLimit: htmlCharacterLimit)
        let inputTruncated = exportRange.length < attributed.length

        let paragraphs = try analyzeParagraphs(attributed, limit: exportRange.length)
        guard !paragraphs.isEmpty else {
            throw AgentError.parse("Word 文档未解析出正文（\(url.lastPathComponent)）")
        }

        let bodyFontSize = weightedBodyFontSize(paragraphs)
        let candidates = headingCandidates(paragraphs, bodyFontSize: bodyFontSize)
        let listMarkers = listMarkersByText(paragraphs)

        try checkWordCancellation()
        let htmlAttributes: [NSAttributedString.DocumentAttributeKey: Any] = [
            .documentType: NSAttributedString.DocumentType.html,
            .characterEncoding: String.Encoding.utf8.rawValue,
        ]
        let htmlData = try attributed.data(from: exportRange,
                                           documentAttributes: htmlAttributes)
        try checkWordCancellation()
        guard htmlData.count <= LegacyOfficeLimits.maxDocHTMLBytes else {
            throw AgentError.parse(
                "Word 文档导出的 HTML 超过 "
                + "\(LegacyOfficeLimits.maxDocHTMLBytes / 1024 / 1024)MB 上限"
                + "（\(url.lastPathComponent)）")
        }
        guard let html = String(data: htmlData, encoding: .utf8) else {
            throw AgentError.parse("Word 文档导出的 HTML 编码无效（\(url.lastPathComponent)）")
        }

        try checkWordCancellation()
        let document = try SwiftSoup.parse(html)
        try checkWordCancellation()
        let tableCount = try document.select("table").count
        let outline = try applyHeadings(candidates, listMarkers: listMarkers, to: document)
        try checkWordCancellation()
        let converted = HTMLMarkdown.convert(document, removeBoilerplate: false)
        try checkWordCancellation()
        // 已有 h 标签时 converted.headings 会非空；不能因此丢掉启发式识别出的
        // 条目（HTMLMarkdown 会忽略引用块等上下文里的标题）。
        let finalOutline = Self.mergeOutline(converted: converted.headings,
                                             heuristic: outline,
                                             markdown: converted.markdown)

        var metadata: [String: String] = [
            "engine": "doc",
            "legacy_format": "word-97",
            "paragraphs": String(paragraphs.count),
            "tables": String(tableCount),
            "heading_detection": "font-size",
            "headings": String(finalOutline.count),
        ]

        var markdown = converted.markdown
        try checkWordCancellation()
        var truncated = inputTruncated
        let limited = LegacyOfficeLimits.limitText(markdown)
        if limited.truncated {
            markdown = limited.text
            truncated = true
        }
        if truncated {
            metadata["truncated"] = "true"
            metadata["truncated_reason"] = "text_limit"
            if inputTruncated {
                AgentLog.warning("Word 文档正文超过 \(htmlCharacterLimit) 字符，已在导出 HTML 前截断：\(url.lastPathComponent)")
            } else {
                AgentLog.warning("Word 文档正文超过 \(LegacyOfficeLimits.maxTextCharacters) 字符，已截断：\(url.lastPathComponent)")
            }
        }
        guard !markdown.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AgentError.parse("Word 文档未解析出正文（\(url.lastPathComponent)）")
        }

        return Output(markdown: markdown,
                      outline: finalOutline,
                      metadata: metadata)
    }

    // MARK: - NSAttributedString 分析

    private func analyzeParagraphs(_ attributed: NSAttributedString,
                                   limit: Int? = nil) throws -> [Paragraph] {
        let string = attributed.string as NSString
        guard string.length > 0 else { return [] }
        let scanEnd = min(string.length, max(0, limit ?? string.length))
        var result: [Paragraph] = []
        var index = 0
        var paragraphCount = 0
        while index < scanEnd {
            paragraphCount += 1
            if paragraphCount % 4096 == 0 { try checkWordCancellation() }
            var start = 0
            var end = 0
            var contentsEnd = 0
            string.getParagraphStart(&start, end: &end,
                                     contentsEnd: &contentsEnd,
                                     for: NSRange(location: index, length: 0))
            let contentRange = NSRange(location: start,
                                       length: max(0, min(contentsEnd, scanEnd) - start))
            let rawText = string.substring(with: contentRange)
            let text = rawText.trimmingCharacters(in: .whitespacesAndNewlines)
            if !text.isEmpty {
                result.append(paragraph(from: attributed, string: string,
                                        range: contentRange, text: text))
            }
            guard end > index else { break }
            index = end
        }
        return result
    }

    private func paragraph(from attributed: NSAttributedString, string: NSString,
                           range: NSRange, text: String) -> Paragraph {
        let visible = string.rangeOfCharacter(from: CharacterSet.whitespacesAndNewlines.inverted,
                                              options: [], range: range)
        let location = visible.location == NSNotFound ? range.location : visible.location
        var effective = NSRange()
        let attributes = attributed.attributes(at: location, effectiveRange: &effective)
        let font = attributes[.font] as? NSFont
        let style = attributes[.paragraphStyle] as? NSParagraphStyle
        let traits = font?.fontDescriptor.symbolicTraits ?? []
        let inTable = style?.textBlocks.contains { $0 is NSTextTableBlock } ?? false

        var marker: String?
        if let list = style?.textLists.last, !startsWithListMarker(text) {
            marker = listMarker(for: list.markerFormat)
        }

        return Paragraph(text: text,
                         fontSize: font?.pointSize ?? 0,
                         isBold: traits.contains(.bold),
                         isItalic: traits.contains(.italic),
                         isInTable: inTable,
                         listMarker: marker)
    }

    private func weightedBodyFontSize(_ paragraphs: [Paragraph]) -> CGFloat {
        var counts: [CGFloat: Int] = [:]
        for paragraph in paragraphs where paragraph.fontSize > 0 {
            let size = (paragraph.fontSize * 2).rounded() / 2
            counts[size, default: 0] += max(1, paragraph.text.count)
        }
        var bestSize: CGFloat = 0
        var bestCount = -1
        for (size, count) in counts {
            if count > bestCount || (count == bestCount && size > bestSize) {
                bestSize = size
                bestCount = count
            }
        }
        return bestSize
    }

    private func headingCandidates(_ paragraphs: [Paragraph],
                                   bodyFontSize: CGFloat) -> [HeadingCandidate] {
        guard bodyFontSize > 0 else { return [] }
        let terminators: Set<Character> = ["。", "！", "？", "；", ".", "!", "?", ";"]
        let eligible = paragraphs.filter { paragraph in
            guard !paragraph.isInTable, paragraph.listMarker == nil else { return false }
            guard paragraph.fontSize >= bodyFontSize + 1 else { return false }
            guard paragraph.text.count >= 2, paragraph.text.count <= 80 else { return false }
            if let last = paragraph.text.last, terminators.contains(last) { return false }
            return true
        }
        guard !eligible.isEmpty else { return [] }

        let sizes = Set(eligible.map { ($0.fontSize * 2).rounded() / 2 })
            .sorted(by: >)
        var levelBySize: [CGFloat: Int] = [:]
        for (index, size) in sizes.enumerated() {
            levelBySize[size] = min(3, index + 1)
        }
        return eligible.map { paragraph in
            let size = (paragraph.fontSize * 2).rounded() / 2
            return HeadingCandidate(text: normalize(paragraph.text),
                                    level: levelBySize[size] ?? 3)
        }
    }

    /// 列表项如果 HTML 导出没有变成 `<ul>/<ol>`，按文本补 Markdown 前缀。
    private func listMarkersByText(_ paragraphs: [Paragraph]) -> [String: String] {
        var result: [String: String] = [:]
        for paragraph in paragraphs {
            guard let marker = paragraph.listMarker else { continue }
            let key = normalize(paragraph.text)
            if result[key] == nil { result[key] = marker }
        }
        return result
    }

    // MARK: - HTML 后处理

    private func applyHeadings(_ candidates: [HeadingCandidate],
                               listMarkers: [String: String],
                               to document: Document) throws -> [OutlineEntry] {
        var outline: [OutlineEntry] = []
        var nextCandidate = 0
        var elementCount = 0
        for element in try document.select("p").array() {
            elementCount += 1
            if elementCount % 512 == 0 { try checkWordCancellation() }
            guard !isInsideTableCell(element) else { continue }
            let text = normalize(try element.text())

            if nextCandidate < candidates.count,
               text == candidates[nextCandidate].text {
                let candidate = candidates[nextCandidate]
                let heading = try document.createElement("h\(candidate.level)")
                try heading.text(candidate.text)
                try element.replaceWith(heading)
                outline.append(OutlineEntry(level: candidate.level,
                                            title: candidate.text))
                nextCandidate += 1
                continue
            }

            if let marker = listMarkers[text] {
                try element.prependText(marker)
                continue
            }
            if let bullet = markdownBullet(text) {
                try element.text(bullet)
            }
        }
        return outline
    }

    private func isInsideTableCell(_ element: Element) -> Bool {
        for parent in element.parents() {
            let tag = parent.tagName().lowercased()
            if tag == "td" || tag == "th" { return true }
        }
        return false
    }

    // MARK: - 大纲合并

    /// `HTMLMarkdown` 只收集能渲染成 Markdown 标题的条目；它忽略的上下文
    /// （如引用块）仍可能包含启发式识别出的标题。合并时以转换结果为准，
    /// 再把启发式里缺失的条目录回原位置。
    static func mergeOutline(converted: [OutlineEntry],
                             heuristic: [OutlineEntry],
                             markdown: String) -> [OutlineEntry] {
        guard !converted.isEmpty else { return heuristic }
        guard !heuristic.isEmpty else { return converted }

        // 按标题计数而不是集合去重：同名章节可能重复出现，转换结果里的
        // 数量应与启发式结果一一抵消，不能把第二个同名标题误判为已存在。
        var unmatchedCounts = Dictionary(grouping: converted,
                                         by: { normalizedHeading($0.title) })
            .mapValues(\.count)
        var missing: [OutlineEntry] = []
        for entry in heuristic {
            let key = normalizedHeading(entry.title)
            if let count = unmatchedCounts[key], count > 0 {
                unmatchedCounts[key] = count - 1
            } else {
                missing.append(entry)
            }
        }
        guard !missing.isEmpty else { return converted }

        // 只扫描一次 Markdown 建立「标题 → 首次出现位置」，避免逐条扫描
        // 造成大纲条目多时的 O(n²)。
        let offsets = headingOffsets(in: markdown)
        var result = converted
        var positions = converted.map { offsets[normalizedHeading($0.title)] }
        for entry in missing {
            let position = offsets[normalizedHeading(entry.title)]
            let insertion: Int
            if let position,
               let index = positions.firstIndex(where: {
                   $0.map { $0 > position } ?? false
               }) {
                insertion = index
            } else {
                insertion = result.endIndex
            }
            result.insert(entry, at: insertion)
            positions.insert(position, at: insertion)
        }
        return result
    }

    /// 大纲去重键：折叠空白并去掉 HTMLMarkdown 给标题加上的强调标记，
    /// 例如已有 `<h1><b>第一章</b></h1>` 的标题会转成 `**第一章**`。
    private static func normalizedHeading(_ title: String) -> String {
        title
            .replacingOccurrences(of: "\u{00A0}", with: " ")
            .replacingOccurrences(of: "\\s+", with: " ",
                                  options: .regularExpression)
            // 标题里的局部加粗会变成 `**前缀** 后缀`，不能只处理整段包裹。
            .replacingOccurrences(of: "*", with: "")
            .replacingOccurrences(of: "`", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Markdown 标题行的「去重键 → 字符偏移」；用于把转换结果忽略的启发式
    /// 条目插回文档顺序，而不是一律追加到末尾。引用块标题写作 `> ### 标题`。
    private static func headingOffsets(in markdown: String) -> [String: Int] {
        var offsets: [String: Int] = [:]
        var offset = 0
        for line in markdown.components(separatedBy: "\n") {
            var stripped = line.trimmingCharacters(in: .whitespaces)
            if stripped.hasPrefix(">") {
                stripped = String(stripped.dropFirst())
                    .trimmingCharacters(in: .whitespaces)
            }
            let hashes = stripped.prefix { $0 == "#" }.count
            if hashes > 0, hashes <= 6 {
                let text = stripped.dropFirst(hashes)
                    .trimmingCharacters(in: .whitespaces)
                let key = normalizedHeading(text)
                if !key.isEmpty, offsets[key] == nil {
                    offsets[key] = offset
                }
            }
            offset += line.count + 1
        }
        return offsets
    }

    // MARK: - 列表

    private func startsWithListMarker(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if let first = trimmed.first, bulletCharacters.contains(first) {
            return true
        }
        if trimmed.range(of: "^\\d{1,3}[.、)]\\s*", options: .regularExpression) != nil {
            return true
        }
        return false
    }

    private func markdownBullet(_ text: String) -> String? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let first = trimmed.first, bulletCharacters.contains(first) else { return nil }
        let rest = String(trimmed.dropFirst()).trimmingCharacters(in: .whitespaces)
        guard !rest.isEmpty else { return nil }
        return "- " + rest
    }

    private func listMarker(for format: NSTextList.MarkerFormat) -> String {
        if format == .decimal { return "1. " }
        if format == .lowercaseAlpha || format == .lowercaseLatin { return "a. " }
        if format == .uppercaseAlpha || format == .uppercaseLatin { return "A. " }
        if format == .lowercaseRoman { return "i. " }
        if format == .uppercaseRoman { return "I. " }
        return "- "
    }

    private let bulletCharacters: Set<Character> = [
        "•", "·", "▪", "▫", "●", "○", "◦", "■", "□", "◆", "◇", "➢", "➤",
    ]

    private func normalize(_ text: String) -> String {
        text.replacingOccurrences(of: "\u{00A0}", with: " ")
            .replacingOccurrences(of: "\\s+", with: " ",
                                  options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
