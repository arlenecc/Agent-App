import Foundation
import CryptoKit

/// 大纲择优：
/// - 解析器自带大纲（EPUB 目录 / Office 标题）先逐条锚定到 Markdown；
/// - 锚定率、层级、噪声、正文覆盖率综合评分；
/// - 解析器完全无法锚定时必须回退正则大纲；
/// - 胜者同时用于 digest / chapters / 章节切分。
public enum OutlineResolver {

    public struct Candidate: Sendable, Equatable {
        public var nodes: [OutlineNode]
        public var totalEntries: Int
        public var anchoredEntries: Int
        public var noiseEntries: Int
        public var source: String
        public var score: Double

        public init(nodes: [OutlineNode], totalEntries: Int, anchoredEntries: Int,
                    noiseEntries: Int, source: String, score: Double) {
            self.nodes = nodes
            self.totalEntries = totalEntries
            self.anchoredEntries = anchoredEntries
            self.noiseEntries = noiseEntries
            self.source = source
            self.score = score
        }
    }

    public static func resolve(markdown: String, parsedOutline: String,
                               docName: String) -> ResolvedBookOutline {
        let regexNodes = DocumentOutline.extract(markdown, source: docName)
        let regexCandidate = makeCandidate(nodes: regexNodes,
                                           totalEntries: regexNodes.count,
                                           anchoredEntries: regexNodes.count,
                                           source: "regex", markdown: markdown)
        let parserCandidate = makeParserCandidate(parsedOutline, markdown: markdown)

        let chosen: Candidate
        if parserCandidate.anchoredEntries > 0,
           parserCandidate.score >= regexCandidate.score {
            chosen = parserCandidate
        } else if !regexCandidate.nodes.isEmpty {
            chosen = regexCandidate
        } else if parserCandidate.anchoredEntries > 0 {
            chosen = parserCandidate
        } else {
            chosen = Candidate(nodes: [], totalEntries: 0, anchoredEntries: 0,
                               noiseEntries: 0, source: "none", score: 0)
        }

        let repaired = repairTitles(chosen.nodes, markdown: markdown)
        let sections = makeSections(repaired.nodes, markdown: markdown)
        let lastPath = (docName as NSString).lastPathComponent
        let digest = repaired.nodes.isEmpty
            ? "# 目录\n\n\(lastPath)"
            : "# 目录\n\n\(DocumentOutline.render(repaired.nodes))"
        let chaptersJSON = encodeChapters(repaired.nodes)
        return ResolvedBookOutline(
            nodes: repaired.nodes, sections: sections,
            source: chosen.nodes.isEmpty ? "none" : chosen.source,
            digest: digest, chaptersJSON: chaptersJSON,
            hadRepairs: repaired.repaired)
    }

    // MARK: - 质量评分

    public static func qualityScore(nodes: [OutlineNode], totalEntries: Int,
                                    anchoredEntries: Int, noiseEntries: Int,
                                    textLength: Int) -> Double {
        guard anchoredEntries > 0, totalEntries > 0 else { return 0 }
        let valid = nodes.filter {
            !$0.title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                && $0.level >= 1 && $0.level <= 6
                && max(0, $0.end - $0.start) > 0
        }
        guard !valid.isEmpty else { return 0 }
        let anchor = min(1.0, Double(anchoredEntries) / Double(totalEntries))
        let noise = min(1.0, Double(noiseEntries) / Double(totalEntries))
        let depth = Double(valid.map(\.level).max() ?? 1)
        let span = valid.reduce(0) { $0 + max(0, $1.end - $1.start) }
        let coverage = textLength > 0
            ? min(1.0, Double(span) / Double(textLength))
            : 0
        return Double(valid.count)
            * (1 + anchor)
            * (1 + 0.1 * depth)
            * max(0.05, 1 - noise)
            * (0.6 + 0.4 * coverage)
    }

    private static func makeCandidate(nodes: [OutlineNode], totalEntries: Int,
                                      anchoredEntries: Int, source: String,
                                      markdown: String) -> Candidate {
        let noise = nodes.filter { isBadTitle($0.title) }.count
        let score = qualityScore(nodes: nodes, totalEntries: totalEntries,
                                 anchoredEntries: anchoredEntries,
                                 noiseEntries: noise,
                                 textLength: (markdown as NSString).length)
        return Candidate(nodes: nodes, totalEntries: totalEntries,
                         anchoredEntries: anchoredEntries, noiseEntries: noise,
                         source: source, score: score)
    }

    // MARK: - 解析器大纲锚定

    private struct LineEntry {
        var location: Int
        var text: String
        var normalized: String
        var headingTitle: String?
    }

    private static func makeParserCandidate(_ outline: String,
                                            markdown: String) -> Candidate {
        let entries = flatten(outline)
        guard !entries.isEmpty else {
            return Candidate(nodes: [], totalEntries: 0, anchoredEntries: 0,
                             noiseEntries: 0, source: "parser", score: 0)
        }
        let lines = lineIndex(markdown)
        let textLength = (markdown as NSString).length
        var used = Set<Int>()
        var anchors: [(line: Int, level: Int, title: String)] = []
        var searchStart = 0

        for entry in entries {
            var found: Int?
            if searchStart < lines.count {
                for index in searchStart..<lines.count where !used.contains(index) {
                    if matches(entry.title, lines[index]) {
                        found = index
                        break
                    }
                }
            }
            if found == nil {
                for index in 0..<min(searchStart, lines.count) where !used.contains(index) {
                    if matches(entry.title, lines[index]) {
                        found = index
                        break
                    }
                }
            }
            guard let index = found else { continue }
            used.insert(index)
            anchors.append((index, entry.level, entry.title))
            searchStart = index + 1
        }

        anchors.sort { $0.line < $1.line }
        var nodes: [OutlineNode] = []
        for (offset, anchor) in anchors.enumerated() {
            let start = lines[anchor.line].location
            let end = offset + 1 < anchors.count
                ? min(lines[anchors[offset + 1].line].location, textLength)
                : textLength
            nodes.append(OutlineNode(level: max(1, min(6, anchor.level)),
                                     title: anchor.title, start: start, end: end))
        }
        let noise = entries.filter { isBadTitle($0.title) }.count
        let score = qualityScore(nodes: nodes, totalEntries: entries.count,
                                 anchoredEntries: anchors.count,
                                 noiseEntries: noise, textLength: textLength)
        return Candidate(nodes: nodes, totalEntries: entries.count,
                         anchoredEntries: anchors.count, noiseEntries: noise,
                         source: "parser", score: score)
    }

    private static func flatten(_ outline: String) -> [(level: Int, title: String)] {
        outline.components(separatedBy: "\n").compactMap { line in
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty, !trimmed.hasPrefix("…") else { return nil }
            let leading = line.prefix { $0 == " " || $0 == "\t" }.count
            let level = max(1, min(6, leading / 2 + 1))
            return (level, trimmed)
        }
    }

    private static func lineIndex(_ markdown: String) -> [LineEntry] {
        var entries: [LineEntry] = []
        var location = 0
        for raw in markdown.components(separatedBy: "\n") {
            let line = raw.hasSuffix("\r") ? String(raw.dropLast()) : raw
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            entries.append(LineEntry(
                location: location,
                text: trimmed,
                normalized: normalize(trimmed),
                headingTitle: DocumentOutline.classify(trimmed)?.title))
            location += (raw as NSString).length + 1
        }
        return entries
    }

    private static func matches(_ parserTitle: String, _ line: LineEntry) -> Bool {
        let target = normalize(parserTitle)
        guard !target.isEmpty else { return false }
        if line.normalized == target { return true }
        // 解析器目录可能只保留「引言」，Markdown 标题是「第一章 引言」；
        // 只对可识别为标题的行做包含匹配，避免命中正文句子。
        guard let heading = line.headingTitle else { return false }
        let candidate = normalize(heading)
        let shorter = min(candidate.count, target.count)
        guard shorter >= 2 else { return false }
        return candidate.hasPrefix(target) || target.hasPrefix(candidate)
    }

    private static func normalize(_ line: String) -> String {
        var text = line.trimmingCharacters(in: .whitespacesAndNewlines)
        if let range = text.range(of: #"^#{1,6}\s+"#, options: .regularExpression) {
            text.removeSubrange(range)
        }
        text = text
            .replacingOccurrences(of: "\u{200B}", with: "")
            .replacingOccurrences(of: "\u{FEFF}", with: "")
        return text
            .split(whereSeparator: { $0.isWhitespace })
            .joined(separator: " ")
            .lowercased()
    }

    // MARK: - 保守修复

    private static func repairTitles(_ nodes: [OutlineNode],
                                     markdown: String) -> (nodes: [OutlineNode], repaired: Bool) {
        var repaired = false
        let result = nodes.map { node -> OutlineNode in
            guard isBadTitle(node.title) else { return node }
            let derived = firstPlausibleLine(after: node, markdown: markdown)
            var title = node.title.trimmingCharacters(in: .whitespacesAndNewlines)
            if !derived.isEmpty {
                if isDanglingMarker(title) {
                    title = "\(title) \(derived)"
                } else {
                    title = derived
                }
            } else if title.isEmpty {
                title = "未命名章节"
            }
            if title.count > 80 { title = String(title.prefix(80)) }
            if title != node.title { repaired = true }
            return OutlineNode(level: node.level, title: title,
                               start: node.start, end: node.end)
        }
        return (result, repaired)
    }

    private static func firstPlausibleLine(after node: OutlineNode,
                                           markdown: String) -> String {
        let ns = markdown as NSString
        let length = max(0, min(node.end, ns.length) - node.start)
        guard length > 0 else { return "" }
        let chunk = ns.substring(with: NSRange(location: node.start, length: length))
        guard let newline = chunk.firstIndex(of: "\n") else { return "" }
        let body = chunk[chunk.index(after: newline)...]
        for raw in body.components(separatedBy: "\n") {
            var line = raw.trimmingCharacters(in: .whitespaces)
            while let first = line.first, "-*•·".contains(first) {
                line.removeFirst()
                line = line.trimmingCharacters(in: .whitespaces)
            }
            if let range = line.range(of: #"^#{1,6}\s+"#, options: .regularExpression) {
                line.removeSubrange(range)
            }
            guard !line.isEmpty, line.count >= 2, line.count <= 60,
                  !line.contains("|"), !line.hasPrefix("http"),
                  !isBadTitle(line) else { continue }
            return line
        }
        return ""
    }

    private static func isBadTitle(_ title: String) -> Bool {
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return true }
        if DocumentOutline.isPageHeaderLine(trimmed) { return true }
        if DocumentOutline.isBareNumberTitle(trimmed) { return true }
        if DocumentOutline.looksLikeDateProse(trimmed) { return true }
        if DocumentOutline.hasSentencePunctuation(trimmed) { return true }
        return isDanglingMarker(trimmed)
    }

    private static func isDanglingMarker(_ title: String) -> Bool {
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        return danglingRegex.firstMatch(
            in: trimmed, range: NSRange(trimmed.startIndex..<trimmed.endIndex, in: trimmed)) != nil
    }

    private static let danglingRegex = try! NSRegularExpression(
        pattern: #"^(?:第[一二三四五六七八九十百千零〇0-9]+[章节篇部卷]|chapter|section|part|(?:chapter|section|part)\s+[0-9ivxlcdm.]+|[0-9]+(?:\.[0-9]+)*[、.．]?)$"#,
        options: [.caseInsensitive])

    // MARK: - 章节切分

    private static func makeSections(_ nodes: [OutlineNode],
                                     markdown: String) -> [BookSection] {
        var stack: [(level: Int, title: String)] = []
        return nodes.enumerated().map { index, node in
            while let last = stack.last, last.level >= node.level {
                stack.removeLast()
            }
            stack.append((node.level, node.title))
            let heading = stack.map(\.title).joined(separator: " > ")
            let body = bodyAfterHeading(node, markdown: markdown)
            return BookSection(
                index: index, level: node.level, title: node.title,
                heading: heading, body: body,
                start: node.start, end: node.end,
                contentHash: contentHash(heading: heading, body: body))
        }
    }

    private static func bodyAfterHeading(_ node: OutlineNode, markdown: String) -> String {
        let ns = markdown as NSString
        let length = max(0, min(node.end, ns.length) - node.start)
        guard length > 0 else { return "" }
        let chunk = ns.substring(with: NSRange(location: node.start, length: length))
        guard let newline = chunk.firstIndex(of: "\n") else { return "" }
        return String(chunk[chunk.index(after: newline)...])
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func contentHash(heading: String, body: String) -> String {
        let data = Data((heading + "\n" + body).utf8)
        return SHA256.hash(data: data).prefix(8)
            .map { String(format: "%02x", $0) }
            .joined()
    }

    private static func encodeChapters(_ nodes: [OutlineNode]) -> String {
        let array = nodes.map { ["level": $0.level, "title": $0.title] }
        guard let data = try? JSONSerialization.data(withJSONObject: array, options: []),
              let text = String(data: data, encoding: .utf8) else { return "[]" }
        return text
    }
}
