import Foundation
import SwiftSoup

/// HTML / XHTML → Markdown 风格正文。EPUBParser 复用同一套转换逻辑。
enum HTMLMarkdown {

    /// 转换整篇文档，并返回标题大纲。
    static func convert(_ document: Document,
                        removeBoilerplate: Bool) -> (markdown: String, headings: [OutlineEntry]) {
        guard let root = document.body() ?? document.children().first() else {
            return ("", [])
        }

        // 网页解析去掉导航/页脚；EPUB 只去掉脚本样式，避免误删章节页眉内容。
        let selectors = removeBoilerplate
            ? "script,style,noscript,nav,footer,header"
            : "script,style,noscript"
        _ = try? root.select(selectors).remove()

        var blocks: [String] = []
        var headings: [OutlineEntry] = []
        renderChildren(root, blocks: &blocks, headings: &headings, listDepth: 0)

        var markdown = blocks
            .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            .joined(separator: "\n\n")
        markdown = markdown.replacingOccurrences(of: "\n{3,}", with: "\n\n",
                                                options: .regularExpression)
        return (markdown.trimmingCharacters(in: .whitespacesAndNewlines), headings)
    }

    // MARK: - 块级元素

    private static func renderChildren(_ element: Element,
                                       blocks: inout [String],
                                       headings: inout [OutlineEntry],
                                       listDepth: Int) {
        for child in element.children() {
            renderBlock(child, blocks: &blocks, headings: &headings, listDepth: listDepth)
        }
    }

    private static func renderBlock(_ element: Element,
                                    blocks: inout [String],
                                    headings: inout [OutlineEntry],
                                    listDepth: Int) {
        let tag = element.tagName().lowercased()

        switch tag {
        case "h1", "h2", "h3", "h4", "h5", "h6":
            let level = Int(tag.dropFirst()) ?? 1
            let text = inlineText(of: element).trimmingCharacters(in: .whitespacesAndNewlines)
            if !text.isEmpty {
                blocks.append(String(repeating: "#", count: level) + " " + text)
                headings.append(OutlineEntry(level: level, title: text))
            }

        case "p", "address", "figcaption":
            if let heading = boldOnlyHeading(of: element) {
                blocks.append(heading.markdown)
                headings.append(OutlineEntry(level: heading.level, title: heading.title))
            } else {
                let text = inlineText(of: element).trimmingCharacters(in: .whitespacesAndNewlines)
                if !text.isEmpty { blocks.append(text) }
            }

        case "ul", "ol":
            let list = renderList(element, ordered: tag == "ol", depth: listDepth)
            if !list.isEmpty { blocks.append(list) }

        case "table":
            let table = renderTable(element)
            if !table.isEmpty { blocks.append(table) }

        case "pre":
            let code = renderPre(element)
            if !code.isEmpty { blocks.append(code) }

        case "blockquote":
            var inner: [String] = []
            var ignoredHeadings: [OutlineEntry] = []
            renderChildren(element, blocks: &inner,
                           headings: &ignoredHeadings, listDepth: listDepth)
            let text = inner.joined(separator: "\n\n")
            if !text.isEmpty {
                blocks.append(prefixLines(text, prefix: "> "))
            }

        case "br", "hr", "script", "style", "noscript", "nav":
            break

        default:
            if hasBlockChildren(element) {
                renderChildren(element, blocks: &blocks,
                               headings: &headings, listDepth: listDepth)
            } else if let heading = boldOnlyHeading(of: element) {
                // Calibre 等工具常用 <div><b>…</b></div> 排版小节标题。
                blocks.append(heading.markdown)
                headings.append(OutlineEntry(level: heading.level, title: heading.title))
            } else {
                let text = inlineText(of: element).trimmingCharacters(in: .whitespacesAndNewlines)
                if !text.isEmpty { blocks.append(text) }
            }
        }
    }

    private static func hasBlockChildren(_ element: Element) -> Bool {
        element.children().contains { blockTags.contains($0.tagName().lowercased()) }
    }

    /// 识别「整个块只有一个 <b>/<strong> 子元素」的伪标题。
    ///
    /// Calibre 转制的 EPUB 普遍用 `<p><b>…</b></p>` 排版小节标题（正文没有 h 标签），
    /// 不识别这类结构会导致大纲几乎为空。仅当块内无其他可见文本、
    /// 文本短且不含句末标点时才视为标题，避免误伤段内强调。
    private static func boldOnlyHeading(of element: Element) -> (markdown: String,
                                                                  level: Int,
                                                                  title: String)? {
        let children = element.children()
        guard children.count == 1 else { return nil }
        let only = children[0]
        let tag = only.tagName().lowercased()
        guard tag == "b" || tag == "strong" else { return nil }

        // 块内除 <b> 外不允许再有可见文本，否则是「段首强调」而不是标题。
        let leadingText = element.getChildNodes()
            .compactMap { $0 as? TextNode }
            .map { $0.text() }
            .joined()
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard leadingText.isEmpty else { return nil }

        let title = rawText(only).trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "[ \\t\\u00A0\\u3000]+", with: " ",
                                  options: .regularExpression)
        guard title.count >= 2, title.count <= 60 else { return nil }
        guard !title.contains("。"), !title.contains("！"), !title.contains("？"),
              !title.contains("；"), !title.contains("，") else { return nil }
        return (markdown: "### \(title)", level: 3, title: title)
    }

    // MARK: - 列表 / 表格 / 代码

    private static func renderList(_ element: Element, ordered: Bool, depth: Int) -> String {
        var lines: [String] = []
        var index = 1
        let indent = String(repeating: "  ", count: max(0, depth))

        for item in element.children() where item.tagName().lowercased() == "li" {
            var pieces: [String] = []
            for node in item.getChildNodes() {
                if let child = node as? Element {
                    let tag = child.tagName().lowercased()
                    if tag == "ul" || tag == "ol" { continue }
                }
                pieces.append(inlineText(of: node, skipNestedLists: true))
            }
            let text = normalizeInlineWhitespace(pieces.joined())
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let prefix = ordered ? "\(index). " : "- "
            lines.append(indent + prefix + text)

            for child in item.children() {
                let tag = child.tagName().lowercased()
                if tag == "ul" || tag == "ol" {
                    let nested = renderList(child, ordered: tag == "ol", depth: depth + 1)
                    if !nested.isEmpty { lines.append(nested) }
                }
            }
            index += 1
        }
        return lines.joined(separator: "\n")
    }

    private static func renderTable(_ element: Element) -> String {
        var rowElements: [Element] = []
        for child in element.children() {
            let tag = child.tagName().lowercased()
            if tag == "tr" {
                rowElements.append(child)
            } else if tag == "thead" || tag == "tbody" || tag == "tfoot" {
                rowElements.append(contentsOf: child.children().filter {
                    $0.tagName().lowercased() == "tr"
                })
            }
        }

        var rows: [[String]] = []
        for row in rowElements {
            let cells = row.children().filter {
                let tag = $0.tagName().lowercased()
                return tag == "th" || tag == "td"
            }
            rows.append(cells.map { escapeTableCell(inlineText(of: $0)) })
        }
        return renderMarkdownTable(rows)
    }

    private static func renderPre(_ element: Element) -> String {
        let code = rawText(element)
            .replacingOccurrences(of: "\r\n", with: "\n")
            .trimmingCharacters(in: .newlines)
        guard !code.isEmpty else { return "" }

        var language = ""
        if let codeElement = try? element.select("code").first(),
           let classValue = try? codeElement.attr("class"),
           let match = languageRegex.firstMatch(
                in: classValue,
                range: NSRange(classValue.startIndex..<classValue.endIndex, in: classValue)),
           let range = Range(match.range(at: 1), in: classValue) {
            language = String(classValue[range])
        }
        return "```\(language)\n\(code)\n```"
    }

    // MARK: - 行内文本

    private static func inlineText(of element: Element, skipNestedLists: Bool = false) -> String {
        var result = ""
        for node in element.getChildNodes() {
            result += inlineText(of: node, skipNestedLists: skipNestedLists)
        }
        return normalizeInlineWhitespace(result)
    }

    private static func inlineText(of node: Node, skipNestedLists: Bool = false) -> String {
        if let textNode = node as? TextNode {
            return textNode.text()
        }
        guard let element = node as? Element else { return "" }

        let tag = element.tagName().lowercased()
        if skipNestedLists && (tag == "ul" || tag == "ol") { return "" }

        switch tag {
        case "br":
            return "\n"
        case "script", "style", "noscript":
            return ""
        case "code":
            let code = rawText(element).trimmingCharacters(in: .newlines)
            return code.isEmpty ? "" : "`\(code)`"
        case "strong", "b":
            let inner = inlineText(of: element).trimmingCharacters(in: .whitespacesAndNewlines)
            return inner.isEmpty ? "" : "**\(inner)**"
        case "em", "i":
            let inner = inlineText(of: element).trimmingCharacters(in: .whitespacesAndNewlines)
            return inner.isEmpty ? "" : "*\(inner)*"
        case "img":
            return (try? element.attr("alt")) ?? ""
        case "p", "div", "section", "article", "li", "tr",
             "h1", "h2", "h3", "h4", "h5", "h6":
            let inner = inlineText(of: element)
            return "\n" + inner + "\n"
        default:
            return inlineText(of: element, skipNestedLists: skipNestedLists)
        }
    }

    private static func rawText(_ node: Node) -> String {
        if let textNode = node as? TextNode {
            return textNode.text()
        }
        guard let element = node as? Element else { return "" }
        if element.tagName().lowercased() == "br" { return "\n" }
        return element.getChildNodes().map { rawText($0) }.joined()
    }

    private static func normalizeInlineWhitespace(_ text: String) -> String {
        var result = text.replacingOccurrences(of: "[ \t]+", with: " ",
                                               options: .regularExpression)
        result = result.replacingOccurrences(of: " *\n *", with: "\n",
                                             options: .regularExpression)
        return result.trimmingCharacters(in: .whitespaces)
    }

    // MARK: - Markdown 表格

    static func renderMarkdownTable(_ rows: [[String]]) -> String {
        guard let columnCount = rows.map(\.count).max(), columnCount > 0 else { return "" }
        func line(_ values: [String]) -> String {
            var padded = values
            if padded.count < columnCount {
                padded += Array(repeating: "", count: columnCount - padded.count)
            }
            return "| " + padded.joined(separator: " | ") + " |"
        }
        let header = line(rows.first ?? [])
        let separator = "| " + Array(repeating: "---", count: columnCount).joined(separator: " | ") + " |"
        let body = rows.dropFirst().map(line)
        return ([header, separator] + body).joined(separator: "\n")
    }

    static func escapeTableCell(_ value: String) -> String {
        value.replacingOccurrences(of: "|", with: "\\|")
            .replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func prefixLines(_ text: String, prefix: String) -> String {
        text.components(separatedBy: "\n").map { prefix + $0 }.joined(separator: "\n")
    }

    private static let blockTags: Set<String> = [
        "address", "article", "aside", "blockquote", "details", "div", "dl", "dd", "dt",
        "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4",
        "h5", "h6", "header", "hr", "li", "main", "nav", "ol", "p", "pre", "section",
        "table", "ul", "tr", "td", "th",
    ]

    private static let languageRegex = try! NSRegularExpression(
        pattern: #"(?:language-|lang-)([A-Za-z0-9_+-]+)"#)
}

// MARK: - HTML 解析器

public final class HTMLParser: DocumentParser {

    public let name = "HTML"
    public let supportedExtensions = ["html", "htm", "xhtml"]

    public init() {}

    public func parse(url: URL,
                      options: ParserOptions,
                      progress: ((ParserProgress) -> Void)?) async throws -> ParsedDocument {
        guard let data = try? Data(contentsOf: url) else {
            throw AgentError.parse("无法读取文件：\(url.lastPathComponent)")
        }
        let html = DocumentTextDecoder.decode(data)

        let document: Document
        do {
            document = try SwiftSoup.parse(html, url.absoluteString)
        } catch {
            throw AgentError.parse("HTML 解析失败（\(url.lastPathComponent)）：\(error.localizedDescription)")
        }

        let title = (try? document.title()) ?? ""
        let (markdown, headings) = HTMLMarkdown.convert(document, removeBoilerplate: true)

        var metadata: [String: String] = ["engine": "html"]
        if !title.isEmpty { metadata["title"] = title }

        var parsed = ParsedDocument(text: markdown,
                                    source: url.lastPathComponent,
                                    outline: MarkdownOutline.build(headings),
                                    metadata: metadata)
        parsed = finish(parsed, options: options)
        return parsed
    }
}
