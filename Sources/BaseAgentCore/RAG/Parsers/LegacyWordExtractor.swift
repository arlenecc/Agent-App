import AppKit
import Foundation
import SwiftSoup

/// Word 97–2003 `.doc` 提取。
///
/// 文本、字体、表格和列表格式交给 AppKit 的 OLE2 Word 导入器；标题没有
/// 可靠的命名样式，只能用「字号明显大于正文 + 短段落」启发式识别，
/// 再把导出的 HTML 中对应 `<p>` 提升为 `<h1>`–`<h3>`，复用 HTMLMarkdown。
struct LegacyWordExtractor {

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

    private func extractInternal(url: URL) throws -> Output {
        let values = try url.resourceValues(forKeys: [.fileSizeKey])
        if let size = values.fileSize, size > LegacyOfficeLimits.maxFileSize {
            throw AgentError.parse(
                "Word 文档超过 \(LegacyOfficeLimits.maxFileSize / 1024 / 1024)MB 上限"
                + "（\(url.lastPathComponent)）")
        }

        let attributed = try autoreleasepool {
            try NSAttributedString(url: url, options: [:], documentAttributes: nil)
        }
        guard attributed.length > 0 else {
            throw AgentError.parse("Word 文档未解析出正文（\(url.lastPathComponent)）")
        }

        let paragraphs = analyzeParagraphs(attributed)
        guard !paragraphs.isEmpty else {
            throw AgentError.parse("Word 文档未解析出正文（\(url.lastPathComponent)）")
        }

        let bodyFontSize = weightedBodyFontSize(paragraphs)
        let candidates = headingCandidates(paragraphs, bodyFontSize: bodyFontSize)
        let listMarkers = listMarkersByText(paragraphs)

        let htmlAttributes: [NSAttributedString.DocumentAttributeKey: Any] = [
            .documentType: NSAttributedString.DocumentType.html,
            .characterEncoding: String.Encoding.utf8.rawValue,
        ]
        let fullRange = NSRange(location: 0, length: attributed.length)
        let htmlData = try attributed.data(from: fullRange,
                                           documentAttributes: htmlAttributes)
        guard htmlData.count <= LegacyOfficeLimits.maxDocHTMLBytes else {
            throw AgentError.parse(
                "Word 文档导出的 HTML 超过 "
                + "\(LegacyOfficeLimits.maxDocHTMLBytes / 1024 / 1024)MB 上限"
                + "（\(url.lastPathComponent)）")
        }
        guard let html = String(data: htmlData, encoding: .utf8) else {
            throw AgentError.parse("Word 文档导出的 HTML 编码无效（\(url.lastPathComponent)）")
        }

        let document = try SwiftSoup.parse(html)
        let tableCount = try document.select("table").count
        let outline = try applyHeadings(candidates, listMarkers: listMarkers, to: document)
        let converted = HTMLMarkdown.convert(document, removeBoilerplate: false)

        var metadata: [String: String] = [
            "engine": "doc",
            "legacy_format": "word-97",
            "paragraphs": String(paragraphs.count),
            "tables": String(tableCount),
            "heading_detection": "font-size",
            "headings": String(outline.count),
        ]

        var markdown = converted.markdown
        let limited = LegacyOfficeLimits.limitText(markdown)
        if limited.truncated {
            markdown = limited.text
            metadata["truncated"] = "true"
            metadata["truncated_reason"] = "text_limit"
            AgentLog.warning("Word 文档正文超过 \(LegacyOfficeLimits.maxTextCharacters) 字符，已截断：\(url.lastPathComponent)")
        }
        guard !markdown.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AgentError.parse("Word 文档未解析出正文（\(url.lastPathComponent)）")
        }

        return Output(markdown: markdown,
                      outline: converted.headings.isEmpty ? outline : converted.headings,
                      metadata: metadata)
    }

    // MARK: - NSAttributedString 分析

    private func analyzeParagraphs(_ attributed: NSAttributedString) -> [Paragraph] {
        let string = attributed.string as NSString
        guard string.length > 0 else { return [] }
        var result: [Paragraph] = []
        var index = 0
        while index < string.length {
            var start = 0
            var end = 0
            var contentsEnd = 0
            string.getParagraphStart(&start, end: &end,
                                     contentsEnd: &contentsEnd,
                                     for: NSRange(location: index, length: 0))
            let contentRange = NSRange(location: start, length: max(0, contentsEnd - start))
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
        for element in try document.select("p").array() {
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
