import Foundation

/// 文本文件解码：BOM 优先，其次 UTF-8，再尝试 GB18030，最后有损 UTF-8。
enum DocumentTextDecoder {
    static func decode(_ data: Data) -> String {
        if data.starts(with: [0xEF, 0xBB, 0xBF]) {
            return String(data: data.dropFirst(3), encoding: .utf8)
                ?? String(decoding: data.dropFirst(3), as: UTF8.self)
        }
        if data.starts(with: [0xFF, 0xFE]) {
            if let text = String(data: data.dropFirst(2), encoding: .utf16LittleEndian) {
                return text
            }
        }
        if data.starts(with: [0xFE, 0xFF]) {
            if let text = String(data: data.dropFirst(2), encoding: .utf16BigEndian) {
                return text
            }
        }
        if let text = String(data: data, encoding: .utf8) {
            return text
        }
        let gb18030 = String.Encoding(
            rawValue: CFStringConvertEncodingToNSStringEncoding(
                CFStringEncoding(CFStringEncodings.GB_18030_2000.rawValue)))
        if let text = String(data: data, encoding: gb18030) {
            return text
        }
        return String(decoding: data, as: UTF8.self)
    }

    /// HTML 解码：在通用逻辑之前先看 `<meta charset>`（老网页常见 big5 /
    /// shift_jis / gb2312，用 UTF-8 兜底会整篇乱码）。
    ///
    /// 顺序：BOM → 严格 UTF-8（合法 UTF-8 一定就是正确解码）→ 声明编码
    /// → GB18030 → 有损 UTF-8。
    static func decodeHTML(_ data: Data) -> String {
        let hasBOM = data.starts(with: [0xEF, 0xBB, 0xBF])
            || data.starts(with: [0xFF, 0xFE])
            || data.starts(with: [0xFE, 0xFF])
        if !hasBOM {
            if let text = String(data: data, encoding: .utf8) { return text }
            if let declared = declaredEncoding(in: data),
               let encoding = stringEncoding(named: declared),
               encoding != .utf8,
               let text = String(data: data, encoding: encoding) {
                return text
            }
        }
        return decode(data)
    }

    /// 从文件头部（前 8KB）读取 `<meta charset>` / `<?xml encoding>` 声明。
    private static func declaredEncoding(in data: Data) -> String? {
        let prefix = data.prefix(8192)
        let ascii = String(decoding: prefix, as: UTF8.self)
        for regex in declaredEncodingRegexes {
            let range = NSRange(ascii.startIndex..<ascii.endIndex, in: ascii)
            guard let match = regex.firstMatch(in: ascii, range: range),
                  match.numberOfRanges > 1,
                  let valueRange = Range(match.range(at: 1), in: ascii) else { continue }
            let value = String(ascii[valueRange]).trimmingCharacters(in: .whitespaces)
            if !value.isEmpty { return value }
        }
        return nil
    }

    /// 编码名 → Foundation 编码（先查常见别名，再交给 CoreFoundation 的 IANA 映射）。
    private static func stringEncoding(named name: String) -> String.Encoding? {
        let normalized = name.lowercased()
            .trimmingCharacters(in: CharacterSet(charactersIn: "\"' ;"))
        func cf(_ encoding: CFStringEncodings) -> String.Encoding {
            String.Encoding(rawValue: CFStringConvertEncodingToNSStringEncoding(
                CFStringEncoding(encoding.rawValue)))
        }
        switch normalized {
        case "utf-8", "utf8", "unicode-1-1-utf-8": return .utf8
        case "gb2312", "gbk", "gb18030", "x-gbk", "csgb2312", "hz-gb-2312":
            return cf(.GB_18030_2000)
        case "big5", "big-5", "big5-hkscs", "csbig5", "x-x-big5": return cf(.big5)
        case "shift_jis", "shift-jis", "sjis", "x-sjis", "windows-31j", "ms_kanji":
            return .shiftJIS
        case "euc-jp", "x-euc-jp", "eucjp": return cf(.EUC_JP)
        case "euc-kr", "x-euc-kr", "ks_c_5601-1987", "windows-949":
            return cf(.EUC_KR)
        case "iso-8859-1", "latin1", "latin-1", "iso8859-1": return .isoLatin1
        case "windows-1252", "cp1252": return .windowsCP1252
        case "utf-16", "utf-16le", "utf-16be": return .utf16
        case "koi8-r": return cf(.KOI8_R)
        default: break
        }
        let cfEncoding = CFStringConvertIANACharSetNameToEncoding(normalized as CFString)
        guard cfEncoding != kCFStringEncodingInvalidId else { return nil }
        return String.Encoding(rawValue: CFStringConvertEncodingToNSStringEncoding(cfEncoding))
    }

    private static let declaredEncodingRegexes: [NSRegularExpression] = [
        try! NSRegularExpression(pattern: #"(?i)charset\s*=\s*["']?\s*([A-Za-z0-9_\-:.]+)"#),
        try! NSRegularExpression(pattern: #"(?i)encoding\s*=\s*["']\s*([A-Za-z0-9_\-:.]+)"#),
    ]
}

/// 纯文本 / Markdown / CSV 等解析器。
/// CSV/TSV 转换为 Markdown 表格；Markdown 额外生成标题大纲。
public final class PlainTextParser: DocumentParser {

    public let name = "PlainText"
    public let supportedExtensions = [
        "txt", "text", "log", "md", "markdown", "rst", "csv", "tsv", "json", "jsonl",
        "tex", "latex",
    ]

    public init() {}

    public func parse(url: URL,
                      options: ParserOptions,
                      progress: ((ParserProgress) -> Void)?) async throws -> ParsedDocument {
        guard let data = try? Data(contentsOf: url) else {
            throw AgentError.parse("无法读取文件：\(url.lastPathComponent)")
        }
        let raw = DocumentTextDecoder.decode(data)
        let ext = url.pathExtension.lowercased()

        var text = raw
        if ext == "csv" || ext == "tsv" {
            let delimiter: Character = ext == "csv" ? "," : "\t"
            let rows = CSVParser.parse(raw, delimiter: delimiter)
            let table = HTMLMarkdown.renderMarkdownTable(
                rows.map { $0.map(HTMLMarkdown.escapeTableCell) })
            if !table.isEmpty {
                text = table
            }
        } else if ext == "json" || ext == "jsonl" {
            // JSON 保留原始结构，但包成 Markdown 代码围栏，便于后续检索与展示。
            let body = raw.trimmingCharacters(in: .newlines)
            text = "```json\n\(body)\n```"
        } else if ext != "md" && ext != "markdown" {
            // 非Markdown 纯文本：尝试启发式还原标题结构（目录块、编号、
            // 格式对比），转换成 Markdown。无把握时保持原文。
            if let restored = PlainTextStructure.restore(raw) {
                text = restored.markdown
                let outline = MarkdownOutline.build(restored.entries)
                var parsed = ParsedDocument(text: text,
                                            source: url.lastPathComponent,
                                            outline: outline,
                                            metadata: ["engine": "text", "format": ext,
                                                       "structure_restored": "true",
                                                       "toc_lines_removed":
                                                        String(restored.tocLinesRemoved)])
                parsed = finish(parsed, options: options,
                                profile: DocumentProfile.defaultProfile(for: url))
                return parsed
            }
        }

        let outline = (ext == "md" || ext == "markdown")
            ? MarkdownOutline.build(MarkdownOutline.scan(raw))
            : ""

        var parsed = ParsedDocument(text: text,
                                    source: url.lastPathComponent,
                                    outline: outline,
                                    metadata: ["engine": "text", "format": ext])
        // JSON 保留原始结构，避免清洗破坏字符串内容。
        if ext != "json" && ext != "jsonl" {
            parsed = finish(parsed, options: options,
                            profile: DocumentProfile.defaultProfile(for: url))
        }
        return parsed
    }
}

/// 轻量 RFC 4180 CSV/TSV 解析（支持引号、转义引号与 CRLF）。
enum CSVParser {
    static func parse(_ text: String, delimiter: Character) -> [[String]] {
        var rows: [[String]] = []
        var row: [String] = []
        var field = ""
        var inQuotes = false

        var index = text.startIndex
        while index < text.endIndex {
            let character = text[index]

            if inQuotes {
                if character == "\"" {
                    let next = text.index(after: index)
                    if next < text.endIndex, text[next] == "\"" {
                        field.append("\"")
                        index = next
                    } else {
                        inQuotes = false
                    }
                } else {
                    field.append(character)
                }
            } else {
                switch character {
                case "\"":
                    inQuotes = true
                case delimiter:
                    row.append(field)
                    field = ""
                case "\n":
                    row.append(field)
                    rows.append(row)
                    row = []
                    field = ""
                case "\r":
                    break
                default:
                    field.append(character)
                }
            }
            index = text.index(after: index)
        }

        if !field.isEmpty || !row.isEmpty {
            row.append(field)
            rows.append(row)
        }

        // 丢弃全空行。
        return rows.filter { line in
            !line.allSatisfy { $0.trimmingCharacters(in: .whitespaces).isEmpty }
        }
    }
}
