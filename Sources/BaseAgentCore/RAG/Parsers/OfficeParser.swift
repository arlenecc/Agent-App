import Foundation
import ZIPFoundation
import AEXML

/// Office Open XML 解析：.docx / .xlsx / .pptx。
///
/// 仅使用 ZipFoundation + AEXML，不引入新依赖：
/// - docx：word/document.xml 的段落与表格；
/// - xlsx：sharedStrings + 各 sheet 的行列；
/// - pptx：ppt/slides 的文本框与表格。
public final class OfficeParser: DocumentParser {

    public let name = "Office"
    public let supportedExtensions = ["docx", "xlsx", "pptx"]

    public init() {}

    public func parse(url: URL,
                      options: ParserOptions,
                      progress: ((ParserProgress) -> Void)?) async throws -> ParsedDocument {
        let archive = try RAGArchiveReader.open(url)

        do {
            let extensionName = url.pathExtension.lowercased()
            let result: (text: String, outline: [OutlineEntry], metadata: [String: String])
            switch extensionName {
            case "docx":
                result = try parseDocx(archive: archive)
            case "xlsx":
                result = try parseXlsx(archive: archive)
            case "pptx":
                result = try parsePptx(archive: archive)
            default:
                throw AgentError.unsupported("不支持的 Office 格式：\(extensionName)")
            }

            guard !result.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw AgentError.parse("Office 文档未解析出正文：\(url.lastPathComponent)")
            }

            var metadata = result.metadata
            metadata["engine"] = extensionName

            var parsed = ParsedDocument(text: result.text,
                                        source: url.lastPathComponent,
                                        outline: MarkdownOutline.build(result.outline),
                                        metadata: metadata)
            parsed = finish(parsed, options: options)
            return parsed
        } catch let error as AgentError {
            throw error
        } catch {
            throw AgentError.parse("Office 解析失败（\(url.lastPathComponent)）：\(error.localizedDescription)")
        }
    }

    // MARK: - DOCX

    private func parseDocx(archive: Archive) throws -> (String, [OutlineEntry], [String: String]) {
        guard let data = RAGArchiveReader.data(archive, path: "word/document.xml") else {
            throw AgentError.parse("DOCX 缺少 word/document.xml")
        }
        let document = try RAGXMLHelper.parse(data, source: "word/document.xml")
        guard let body = RAGXMLHelper.descendants(document.root, "body").first else {
            throw AgentError.parse("DOCX 缺少文档正文（w:body）")
        }

        var blocks: [String] = []
        var headings: [OutlineEntry] = []
        renderDocxChildren(body, blocks: &blocks, headings: &headings)

        return (blocks.joined(separator: "\n\n"), headings, [:])
    }

    private func renderDocxChildren(_ element: AEXMLElement,
                                    blocks: inout [String],
                                    headings: inout [OutlineEntry]) {
        for child in element.children {
            switch RAGXMLHelper.localName(child.name).lowercased() {
            case "p":
                let text = docxParagraphText(child)
                guard !text.isEmpty else { continue }
                if let level = docxHeadingLevel(child) {
                    blocks.append(String(repeating: "#", count: level) + " " + text)
                    headings.append(OutlineEntry(level: level, title: text))
                } else {
                    blocks.append(text)
                }

            case "tbl":
                let table = docxTable(child)
                if !table.isEmpty { blocks.append(table) }

            case "sectpr", "tblpr", "trpr", "tcpr", "ppr", "rpr", "props":
                continue

            case "sdt", "sdtcontent", "body":
                renderDocxChildren(child, blocks: &blocks, headings: &headings)

            default:
                renderDocxChildren(child, blocks: &blocks, headings: &headings)
            }
        }
    }

    private func docxParagraphText(_ paragraph: AEXMLElement) -> String {
        RAGXMLHelper.descendants(paragraph, "t")
            .compactMap(\.value)
            .joined()
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// 优先读取 outlineLvl，其次从 pStyle 的 Heading N / 标题 N 读取。
    private func docxHeadingLevel(_ paragraph: AEXMLElement) -> Int? {
        guard let properties = RAGXMLHelper.firstChild(paragraph, "pPr") else { return nil }

        if let outline = RAGXMLHelper.firstChild(properties, "outlineLvl"),
           let raw = RAGXMLHelper.attribute(outline, ["val"]),
           let level = Int(raw) {
            return min(6, max(1, level + 1))
        }

        if let style = RAGXMLHelper.firstChild(properties, "pStyle"),
           let value = RAGXMLHelper.attribute(style, ["val"]) {
            if value.lowercased() == "title" { return 1 }
            let range = NSRange(value.startIndex..<value.endIndex, in: value)
            if let match = Self.docxHeadingStyleRegex.firstMatch(in: value, range: range),
               let levelRange = Range(match.range(at: 1), in: value),
               let level = Int(value[levelRange]) {
                return level
            }
        }
        return nil
    }

    private func docxTable(_ table: AEXMLElement) -> String {
        var rows: [[String]] = []
        for row in RAGXMLHelper.children(table, "tr") {
            let cells = RAGXMLHelper.children(row, "tc")
            rows.append(cells.map { cell in
                let paragraphs = RAGXMLHelper.descendants(cell, "p")
                    .map { docxParagraphText($0) }
                    .filter { !$0.isEmpty }
                return HTMLMarkdown.escapeTableCell(paragraphs.joined(separator: " "))
            })
        }
        return HTMLMarkdown.renderMarkdownTable(rows)
    }

    // MARK: - XLSX

    private func parseXlsx(archive: Archive) throws -> (String, [OutlineEntry], [String: String]) {
        guard let workbookData = RAGArchiveReader.data(archive, path: "xl/workbook.xml") else {
            throw AgentError.parse("XLSX 缺少 xl/workbook.xml")
        }
        let workbook = try RAGXMLHelper.parse(workbookData, source: "xl/workbook.xml")
        let sheets = RAGXMLHelper.descendants(workbook.root, "sheet")
        guard !sheets.isEmpty else {
            throw AgentError.parse("XLSX 未声明任何工作表")
        }

        var relationships: [String: String] = [:]
        if let relsData = RAGArchiveReader.data(archive, path: "xl/_rels/workbook.xml.rels"),
           let rels = try? AEXMLDocument(xml: relsData) {
            for relationship in RAGXMLHelper.descendants(rels.root, "Relationship") {
                if let id = RAGXMLHelper.attribute(relationship, ["Id"]),
                   let target = RAGXMLHelper.attribute(relationship, ["Target"]) {
                    relationships[id] = target
                }
            }
        }

        var sharedStrings: [String] = []
        if let sharedData = RAGArchiveReader.data(archive, path: "xl/sharedStrings.xml"),
           let shared = try? AEXMLDocument(xml: sharedData) {
            sharedStrings = RAGXMLHelper.descendants(shared.root, "si").map { item in
                RAGXMLHelper.descendants(item, "t").compactMap(\.value).joined()
            }
        }

        var sections: [String] = []
        var outline: [OutlineEntry] = []
        for (index, sheet) in sheets.enumerated() {
            let sheetName = RAGXMLHelper.attribute(sheet, ["name"]) ?? "Sheet\(index + 1)"
            var sheetPath: String?
            if let relationshipID = RAGXMLHelper.attribute(sheet, ["r:id"]),
               let target = relationships[relationshipID] {
                sheetPath = resolvePartTarget(target, baseDirectory: "xl")
            }
            if let candidate = sheetPath,
               RAGArchiveReader.data(archive, path: candidate) != nil {
                sheetPath = candidate
            } else {
                sheetPath = "xl/worksheets/sheet\(index + 1).xml"
            }
            guard let path = sheetPath,
                  let sheetData = RAGArchiveReader.data(archive, path: path),
                  let sheetDocument = try? AEXMLDocument(xml: sheetData) else {
                continue
            }

            var rows: [[String]] = []
            for row in RAGXMLHelper.descendants(sheetDocument.root, "row") {
                if rows.count >= Self.maxXlsxRows { break }
                var values: [Int: String] = [:]
                for (cellIndex, cell) in RAGXMLHelper.children(row, "c").enumerated() {
                    let column = RAGXMLHelper.attribute(cell, ["r"])
                        .map { columnIndex(from: $0) } ?? cellIndex
                    // 远端列（异常引用 / 格式残留）会把行宽撑到上万列，
                    // 限制到常见工作表范围，避免生成巨型空表。
                    guard column >= 0, column < Self.maxXlsxColumns else { continue }
                    let value = xlsxCellValue(cell, sharedStrings: sharedStrings)
                    if !value.isEmpty { values[column] = value }
                }
                guard !values.isEmpty else { continue }
                let width = (values.keys.max() ?? 0) + 1
                rows.append((0..<width).map { values[$0] ?? "" })
            }

            let table = HTMLMarkdown.renderMarkdownTable(rows)
            guard !table.isEmpty else { continue }
            sections.append("## Sheet: \(sheetName)\n\n\(table)")
            outline.append(OutlineEntry(level: 2, title: "Sheet: \(sheetName)"))
        }

        guard !sections.isEmpty else {
            throw AgentError.parse("XLSX 未读取到任何工作表数据")
        }
        return (sections.joined(separator: "\n\n"), outline,
                ["sheets": String(sections.count)])
    }

    private func xlsxCellValue(_ cell: AEXMLElement, sharedStrings: [String]) -> String {
        let type = RAGXMLHelper.attribute(cell, ["t"]) ?? ""
        switch type {
        case "s":
            guard let raw = RAGXMLHelper.firstChild(cell, "v")?.value,
                  let index = Int(raw), sharedStrings.indices.contains(index) else { return "" }
            return sharedStrings[index]
        case "inlineStr":
            return RAGXMLHelper.descendants(cell, "t").compactMap(\.value).joined()
        case "b":
            guard let raw = RAGXMLHelper.firstChild(cell, "v")?.value else { return "" }
            return raw == "1" ? "TRUE" : "FALSE"
        default:
            return RAGXMLHelper.firstChild(cell, "v")?.value ?? ""
        }
    }

    /// "C12" → 2（零基列号）。
    private func columnIndex(from reference: String) -> Int {
        var value = 0
        for scalar in reference.uppercased().unicodeScalars {
            guard scalar.value >= 65, scalar.value <= 90 else { break }
            value = value * 26 + Int(scalar.value - 64)
        }
        return max(0, value - 1)
    }

    // MARK: - PPTX

    private func parsePptx(archive: Archive) throws -> (String, [OutlineEntry], [String: String]) {
        var relationships: [String: String] = [:]
        if let relsData = RAGArchiveReader.data(archive, path: "ppt/_rels/presentation.xml.rels"),
           let rels = try? AEXMLDocument(xml: relsData) {
            for relationship in RAGXMLHelper.descendants(rels.root, "Relationship") {
                if let id = RAGXMLHelper.attribute(relationship, ["Id"]),
                   let target = RAGXMLHelper.attribute(relationship, ["Target"]) {
                    relationships[id] = target
                }
            }
        }

        var slidePaths: [String] = []
        if let presentationData = RAGArchiveReader.data(archive, path: "ppt/presentation.xml"),
           let presentation = try? AEXMLDocument(xml: presentationData) {
            for slideID in RAGXMLHelper.descendants(presentation.root, "sldId") {
                guard let relationshipID = RAGXMLHelper.attribute(slideID, ["r:id"]),
                      let target = relationships[relationshipID] else { continue }
                slidePaths.append(resolvePartTarget(target, baseDirectory: "ppt"))
            }
        }

        // 缺失 presentation 关系时回退到归档内全部 slideN.xml。
        if slidePaths.isEmpty {
            slidePaths = RAGArchiveReader.filePaths(archive)
                .filter { $0.range(of: #"^ppt/slides/slide\d+\.xml$"#,
                                   options: .regularExpression) != nil }
                .sorted { slideNumber($0) < slideNumber($1) }
        }
        guard !slidePaths.isEmpty else {
            throw AgentError.parse("PPTX 未找到任何幻灯片")
        }

        var sections: [String] = []
        var outline: [OutlineEntry] = []
        for (index, path) in slidePaths.enumerated() {
            guard let data = RAGArchiveReader.data(archive, path: path),
                  let slide = try? AEXMLDocument(xml: data) else { continue }

            let (content, title) = renderSlide(slide.root)
            var section = "## Slide \(index + 1)"
            let body = content.trimmingCharacters(in: .whitespacesAndNewlines)
            if !body.isEmpty {
                section += "\n\n" + body
            }
            sections.append(section)
            outline.append(OutlineEntry(level: 2,
                                        title: title.map { "Slide \(index + 1)：\($0)" }
                                            ?? "Slide \(index + 1)"))
        }

        guard !sections.isEmpty else {
            throw AgentError.parse("PPTX 未读取到任何幻灯片内容")
        }
        return (sections.joined(separator: "\n\n"), outline,
                ["slides": String(sections.count)])
    }

    /// 按文档顺序收集 sp / graphicFrame（不深入其内部，避免重复取文本）。
    private func collectShapes(_ element: AEXMLElement) -> [AEXMLElement] {
        var result: [AEXMLElement] = []
        func walk(_ node: AEXMLElement) {
            for child in node.children {
                let name = RAGXMLHelper.localName(child.name).lowercased()
                if name == "sp" || name == "graphicframe" {
                    result.append(child)
                } else {
                    walk(child)
                }
            }
        }
        walk(element)
        return result
    }

    private func renderSlide(_ root: AEXMLElement) -> (String, String?) {
        var blocks: [String] = []
        var title: String?

        for shape in collectShapes(root) {
            let name = RAGXMLHelper.localName(shape.name).lowercased()
            if name == "sp" {
                let isTitle = shapeIsTitle(shape)
                for paragraph in RAGXMLHelper.descendants(shape, "p") {
                    let text = RAGXMLHelper.descendants(paragraph, "t")
                        .compactMap(\.value)
                        .joined()
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !text.isEmpty else { continue }
                    if isTitle, title == nil { title = text }
                    blocks.append(text)
                }
            } else if name == "graphicframe" {
                for table in RAGXMLHelper.descendants(shape, "tbl") {
                    let markdown = pptxTable(table)
                    if !markdown.isEmpty { blocks.append(markdown) }
                }
            }
        }
        return (blocks.joined(separator: "\n\n"), title)
    }

    private func shapeIsTitle(_ shape: AEXMLElement) -> Bool {
        for placeholder in RAGXMLHelper.descendants(shape, "ph") {
            guard let type = RAGXMLHelper.attribute(placeholder, ["type"]) else { continue }
            if type == "title" || type == "ctrTitle" { return true }
        }
        return false
    }

    private func pptxTable(_ table: AEXMLElement) -> String {
        var rows: [[String]] = []
        for row in RAGXMLHelper.children(table, "tr") {
            let cells = RAGXMLHelper.children(row, "tc")
            rows.append(cells.map { cell in
                let text = RAGXMLHelper.descendants(cell, "t")
                    .compactMap(\.value)
                    .joined(separator: " ")
                return HTMLMarkdown.escapeTableCell(text)
            })
        }
        return HTMLMarkdown.renderMarkdownTable(rows)
    }

    private func slideNumber(_ path: String) -> Int {
        let name = (path as NSString).lastPathComponent
        let digits = name.filter(\.isNumber)
        return Int(digits) ?? 0
    }

    // MARK: - 路径

    /// 将关系文件里的相对 Target 解析为归档绝对路径。
    private func resolvePartTarget(_ target: String, baseDirectory: String) -> String {
        let decoded = target.removingPercentEncoding ?? target
        if decoded.hasPrefix("/") {
            return RAGArchivePath.normalize(String(decoded.dropFirst()))
        }
        if decoded.hasPrefix(baseDirectory + "/") {
            return RAGArchivePath.normalize(decoded)
        }
        return RAGArchivePath.normalize(baseDirectory + "/" + decoded)
    }

    // MARK: - 常量

    private static let docxHeadingStyleRegex = try! NSRegularExpression(
        pattern: #"(?i)(?:heading|标题)[\s_-]*([1-6])"#)

    /// XLSX 安全上限：防止异常列引用/超长表把内存与 Markdown 撑爆。
    private static let maxXlsxColumns = 256
    private static let maxXlsxRows = 20_000
}
