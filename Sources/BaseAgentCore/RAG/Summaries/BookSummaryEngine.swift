import Foundation
import CryptoKit

/// 逐章/全书摘要引擎。
///
/// - 章节结构由 `OutlineResolver` 从 Markdown 与已存目录择优生成；
/// - 章节正文按辅助模型上下文预算分段，段结果持久化后增量复用；
/// - reduce 生成 50–100 字章节摘要，全部章节完成后生成 300–500 字全书摘要；
/// - 任何模型调用前后都检查取消标记，中断时只保留已完成分段。
public final class BookSummaryEngine: @unchecked Sendable {

    private let store: VectorStore
    /// 预留 system/指令与输出 token 后的分段预算上限。
    private static let maxSegmentBudget = 8_000

    public init(store: VectorStore) {
        self.store = store
    }

    // MARK: - Token 估算与分段

    /// CJK 按 1 token，其它字符按 0.25 token；向上取整以偏保守。
    public static func estimateTokens(_ text: String) -> Int {
        guard !text.isEmpty else { return 0 }
        let tokens = text.reduce(0.0) { partial, character in
            partial + (isCJK(character) ? 1.0 : 0.25)
        }
        return Int(ceil(tokens))
    }

    /// 按空行分段；累计超出预算时切段，超长段落再按句末标点/字符硬切。
    public static func splitSegments(_ body: String, tokenBudget: Int) -> [String] {
        let text = body.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return [] }
        let budget = max(64, tokenBudget)
        let paragraphs = text.components(separatedBy: "\n\n").compactMap { raw -> String? in
            let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : trimmed
        }

        var segments: [String] = []
        var current = ""
        for paragraph in paragraphs {
            if estimateTokens(paragraph) > budget {
                if !current.isEmpty {
                    segments.append(current)
                    current = ""
                }
                var remainder = paragraph
                while !remainder.isEmpty {
                    let piece = takePrefix(remainder, tokenBudget: budget)
                    if !piece.text.isEmpty { segments.append(piece.text) }
                    remainder = piece.rest
                }
                continue
            }
            let candidate = current.isEmpty ? paragraph : current + "\n\n" + paragraph
            if estimateTokens(candidate) <= budget {
                current = candidate
            } else {
                if !current.isEmpty { segments.append(current) }
                current = paragraph
            }
        }
        if !current.isEmpty { segments.append(current) }
        return segments
    }

    private static func takePrefix(_ text: String, tokenBudget: Int) -> (text: String, rest: String) {
        let characters = Array(text)
        var tokens = 0.0
        var lastBreak = -1
        var index = 0
        while index < characters.count {
            let cost = tokenCost(characters[index])
            if tokens + cost > Double(tokenBudget), index > 0 {
                let cut: Int
                if lastBreak >= 0, lastBreak >= index / 2 {
                    cut = lastBreak + 1
                } else {
                    cut = index
                }
                let prefix = String(characters[0..<cut])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                let rest = String(characters[cut...])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !prefix.isEmpty {
                    return (prefix, rest)
                }
            }
            tokens += cost
            if isSentenceEnd(characters[index]) { lastBreak = index }
            index += 1
        }
        return (text, "")
    }

    // MARK: - 摘要规范化

    public static func normalizeChapterSummary(_ raw: String) -> String {
        normalizeSummary(raw, maxLength: 100, preferredCut: 30)
    }

    public static func normalizeBookSummary(_ raw: String) -> String {
        normalizeSummary(raw, maxLength: 500, preferredCut: 250)
    }

    private static func normalizeSummary(_ raw: String, maxLength: Int,
                                         preferredCut: Int) -> String {
        let lines = raw.replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "```", with: "")
            .components(separatedBy: "\n")
            .compactMap(stripSummaryLine)
        var text = joinSummaryLines(lines)
        text = stripSummaryLabels(text)
        text = text.replacingOccurrences(of: "\\s+", with: " ",
                                         options: .regularExpression)
        text = text.replacingOccurrences(of: " ([。，！？；：、])", with: "$1",
                                         options: .regularExpression)
        text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        while let first = text.first, "\"'“”‘’「」『』".contains(first) {
            text.removeFirst()
        }
        while let last = text.last, "\"'“”‘’「」『』".contains(last) {
            text.removeLast()
        }
        text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard text.count > maxLength else { return text }

        let prefix = String(text.prefix(maxLength))
        if let punctuation = prefix.lastIndex(where: { isSentenceEnd($0) }) {
            let length = prefix.distance(from: prefix.startIndex, to: punctuation) + 1
            if length >= preferredCut {
                return String(prefix[..<prefix.index(after: punctuation)])
            }
        }
        return prefix.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func stripSummaryLine(_ raw: String) -> String? {
        var line = raw.trimmingCharacters(in: .whitespaces)
        line = line
            .replacingOccurrences(of: "**", with: "")
            .replacingOccurrences(of: "__", with: "")
            .replacingOccurrences(of: "`", with: "")
        while line.hasPrefix("#") || line.hasPrefix(">") {
            line.removeFirst()
            line = line.trimmingCharacters(in: .whitespaces)
        }
        if let range = line.range(of: #"^[-*+]\s+"#, options: .regularExpression) {
            line.removeSubrange(range)
        }
        if let range = line.range(of: #"^[0-9]+[.、)]\s*"#, options: .regularExpression) {
            line.removeSubrange(range)
        }
        line = line.trimmingCharacters(in: .whitespaces)
        guard !line.isEmpty else { return nil }
        let labels = ["本章摘要", "章节摘要", "全书摘要", "内容梗概", "内容概要", "摘要", "总结"]
        if labels.contains(where: { line == $0 || line == "\($0)：" || line == "\($0):" }) {
            return nil
        }
        return line
    }

    private static func joinSummaryLines(_ lines: [String]) -> String {
        var result = ""
        for line in lines {
            guard !result.isEmpty else {
                result = line
                continue
            }
            if let last = result.last, let first = line.first,
               last.isASCII && first.isASCII,
               last.isLetter || last.isNumber,
               first.isLetter || first.isNumber {
                result += " " + line
            } else {
                result += line
            }
        }
        return result
    }

    private static func stripSummaryLabels(_ raw: String) -> String {
        var text = raw
        let pattern = #"^(?:本章摘要|章节摘要|全书摘要|内容梗概|内容概要|摘要|总结)\s*[：:]\s*"#
        for _ in 0..<3 {
            guard let range = text.range(of: pattern, options: .regularExpression) else {
                break
            }
            text.removeSubrange(range)
            text = text.trimmingCharacters(in: .whitespaces)
        }
        return text
    }

    // MARK: - 章节结构

    public func resolve(_ document: DocumentRecord, markdown: String) -> ResolvedBookOutline {
        OutlineResolver.resolve(markdown: markdown,
                                parsedOutline: Self.outlineFromDigest(document.digest),
                                docName: document.docName)
    }

    /// `documents.digest` 首行固定为「# 目录」，其余是入库时择优的目录。
    private static func outlineFromDigest(_ digest: String) -> String {
        let lines = digest.components(separatedBy: "\n")
        guard let first = lines.first else { return "" }
        let body = first.trimmingCharacters(in: .whitespaces).hasPrefix("#")
            ? lines.dropFirst().joined(separator: "\n")
            : digest
        return body.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - 逐章处理

    /// 处理 `next_section` 指向的章节；已摘要且内容未变的章节自动跳过。
    public func processNextChapter(document: DocumentRecord,
                                   markdown: String,
                                   llm: LLMService,
                                   now: Date,
                                   isCancelled: () -> Bool) async throws -> ChapterProcessResult {
        let resolved = resolve(document, markdown: markdown)
        let total = resolved.sections.count
        if resolved.digest != document.digest || resolved.chaptersJSON != document.chapters {
            try? store.updateDocumentOutline(docID: document.docID,
                                             digest: resolved.digest,
                                             chapters: resolved.chaptersJSON)
        }
        guard total > 0 else {
            try store.upsertSummaryJob(SummaryJobRecord(
                docID: document.docID, status: .completed, nextSection: 0,
                totalSections: 0, lastError: "", updatedAt: now.timeIntervalSince1970))
            return ChapterProcessResult(
                sectionIndex: 0, totalSections: 0, chapterCompleted: true,
                bookCompleted: true, titleRepaired: resolved.hadRepairs, summary: "")
        }

        let job = try store.summaryJob(docID: document.docID)
        if job?.status == .completed, job?.totalSections == total {
            return ChapterProcessResult(
                sectionIndex: max(0, total - 1), totalSections: total,
                chapterCompleted: true, bookCompleted: true,
                titleRepaired: resolved.hadRepairs, summary: "")
        }

        var stored = try store.chapterSummaries(docID: document.docID)
        // 目录修复可能收缩章节数：越界的旧摘要不会再被覆盖，先清掉。
        if stored.contains(where: { $0.sectionIndex >= total }) {
            try? store.deleteChapterSummaries(docID: document.docID,
                                             fromSectionIndex: total)
            stored.removeAll { $0.sectionIndex >= total }
        }
        let storedByIndex = Dictionary(
            stored.map { ($0.sectionIndex, $0) },
            uniquingKeysWith: { first, _ in first })
        let missing = Self.firstMissingSectionIndex(
            sections: resolved.sections, storedByIndex: storedByIndex)
        // next_section 只在正常推进时可信；目录/正文变化或任务缺失时，
        // 以第一个真正缺少有效摘要的章节为准，避免跳过需要重做的章节。
        var index = max(0, job?.nextSection ?? 0)
        if missing < index { index = missing }
        if job == nil || job?.totalSections != total || job?.nextSection != index {
            try store.upsertSummaryJob(SummaryJobRecord(
                docID: document.docID, status: .pending,
                nextSection: index, totalSections: total,
                lastError: job?.lastError ?? "", updatedAt: now.timeIntervalSince1970))
        }

        while index < total {
            let section = resolved.sections[index]
            if let existing = try store.chapterSummary(docID: document.docID, sectionIndex: index),
               existing.contentHash == section.contentHash,
               !existing.summary.isEmpty {
                index += 1
                continue
            }
            try Self.ensureNotCancelled(isCancelled)
            do {
                let summary = try await summarizeSection(
                    document: document, section: section, llm: llm,
                    now: now, isCancelled: isCancelled)
                try store.upsertChapterSummary(ChapterSummaryRecord(
                    docID: document.docID, sectionIndex: index,
                    heading: section.heading, title: section.title,
                    level: section.level, summary: summary,
                    contentHash: section.contentHash,
                    updatedAt: now.timeIntervalSince1970))
                let next = index + 1
                let chapterCompleted = next >= total
                try store.upsertSummaryJob(SummaryJobRecord(
                    docID: document.docID,
                    status: chapterCompleted ? .summarizingBook : .pending,
                    nextSection: next, totalSections: total,
                    lastError: "", updatedAt: now.timeIntervalSince1970))
                return ChapterProcessResult(
                    sectionIndex: index, totalSections: total,
                    chapterCompleted: chapterCompleted, bookCompleted: false,
                    titleRepaired: resolved.hadRepairs, summary: summary)
            } catch {
                if !Self.isCancellation(error, isCancelled: isCancelled) {
                    try? store.upsertSummaryJob(SummaryJobRecord(
                        docID: document.docID, status: .failed, nextSection: index,
                        totalSections: total,
                        lastError: error.localizedDescription,
                        updatedAt: now.timeIntervalSince1970))
                }
                throw error
            }
        }

        // 所有章节都已有可用摘要，进入全书摘要阶段。
        try store.upsertSummaryJob(SummaryJobRecord(
            docID: document.docID, status: .summarizingBook, nextSection: total,
            totalSections: total, lastError: "", updatedAt: now.timeIntervalSince1970))
        return ChapterProcessResult(
            sectionIndex: max(0, total - 1), totalSections: total,
            chapterCompleted: true, bookCompleted: false,
            titleRepaired: resolved.hadRepairs, summary: "")
    }

    private func summarizeSection(document: DocumentRecord,
                                  section: BookSection,
                                  llm: LLMService,
                                  now: Date,
                                  isCancelled: () -> Bool) async throws -> String {
        let body = section.body.trimmingCharacters(in: .whitespacesAndNewlines)
        if body.count < 40 {
            return Self.fallbackSummary(for: section)
        }
        let budget = Self.segmentBudget(contextWindow: llm.contextWindow)
        let segments = Self.splitSegments(body, tokenBudget: budget)
        guard !segments.isEmpty else { return Self.fallbackSummary(for: section) }

        let persisted = try store.chapterSegments(
            docID: document.docID, sectionIndex: section.index)
        var segmentSummaries: [String] = []
        for (segmentIndex, segment) in segments.enumerated() {
            try Self.ensureNotCancelled(isCancelled)
            let hash = Self.shortHash(segment)
            if let reused = persisted.first(where: {
                $0.segmentIndex == segmentIndex
                    && $0.segmentHash == hash
                    && !$0.summary.isEmpty
            }) {
                segmentSummaries.append(reused.summary)
                continue
            }
            let prompt = """
            请用 30-60 字概括下面章节片段的核心内容，只输出摘要正文，不要标题或解释。

            章节：\(section.heading)
            片段：
            \(segment)
            """
            let raw = try await complete(
                system: "你是专业的书籍摘要助手，只输出摘要正文。",
                prompt: prompt,
                llm: llm, temperature: 0.2,
                isCancelled: isCancelled)
            let summary = Self.normalizeSummary(raw, maxLength: 80, preferredCut: 20)
            guard !summary.isEmpty else {
                throw AgentError.llm("辅助模型未返回有效的分段摘要")
            }
            try store.upsertChapterSegment(ChapterSegmentRecord(
                docID: document.docID, sectionIndex: section.index,
                segmentIndex: segmentIndex, segmentHash: hash,
                summary: summary, updatedAt: now.timeIntervalSince1970))
            segmentSummaries.append(summary)
        }

        var reduced = try await reduceChapter(
            section: section, segmentSummaries: segmentSummaries,
            llm: llm, isCancelled: isCancelled)
        if reduced.count < 30 {
            let retryPrompt = """
            上一次摘要过短。请严格输出 50-100 字的中文摘要，只输出摘要正文。

            章节：\(section.heading)
            各片段摘要：
            \(segmentSummaries.joined(separator: "\n"))

            上一次结果：\(reduced)
            """
            let retryRaw = try await complete(
                system: "你是专业的书籍摘要助手，只输出摘要正文。",
                prompt: retryPrompt, llm: llm, temperature: 0.2,
                isCancelled: isCancelled)
            let retry = Self.normalizeChapterSummary(retryRaw)
            if retry.count > reduced.count { reduced = retry }
        }
        return reduced.isEmpty ? Self.fallbackSummary(for: section) : reduced
    }

    private func reduceChapter(section: BookSection, segmentSummaries: [String],
                               llm: LLMService,
                               isCancelled: () -> Bool) async throws -> String {
        let prompt = """
        以下是章节《\(section.heading)》各片段的内容概括。请综合成 50-100 字的中文摘要，只输出摘要正文，不要标题或解释。

        \(segmentSummaries.enumerated().map { "\($0.offset + 1). \($0.element)" }
            .joined(separator: "\n"))
        """
        let raw = try await complete(
            system: "你是专业的书籍摘要助手，只输出摘要正文。",
            prompt: prompt, llm: llm, temperature: 0.2,
            isCancelled: isCancelled)
        return Self.normalizeChapterSummary(raw)
    }

    private static func fallbackSummary(for section: BookSection) -> String {
        let body = section.body
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !body.isEmpty else { return "\(section.title)（正文较短）" }
        let excerpt = String(body.prefix(60))
        return normalizeChapterSummary("\(section.title)：\(excerpt)")
    }

    // MARK: - 全书摘要

    /// 全部章节完成后的 map-reduce 全书摘要；组结果持久化后增量复用。
    public func processBookSummary(document: DocumentRecord,
                                   markdown: String,
                                   llm: LLMService,
                                   now: Date,
                                   isCancelled: () -> Bool) async throws -> BookSummaryRecord {
        let resolved = resolve(document, markdown: markdown)
        if resolved.digest != document.digest || resolved.chaptersJSON != document.chapters {
            try? store.updateDocumentOutline(docID: document.docID,
                                             digest: resolved.digest,
                                             chapters: resolved.chaptersJSON)
        }
        let sections = resolved.sections
        guard !sections.isEmpty else {
            try store.upsertSummaryJob(SummaryJobRecord(
                docID: document.docID, status: .completed, nextSection: 0,
                totalSections: 0, lastError: "", updatedAt: now.timeIntervalSince1970))
            throw AgentError.storage("本书没有可识别章节，跳过全书摘要")
        }
        let stored = try store.chapterSummaries(docID: document.docID)
        if stored.contains(where: { $0.sectionIndex >= sections.count }) {
            try? store.deleteChapterSummaries(docID: document.docID,
                                             fromSectionIndex: sections.count)
        }
        let storedByIndex = Dictionary(
            stored.map { ($0.sectionIndex, $0) },
            uniquingKeysWith: { first, _ in first })
        var chapters: [ChapterSummaryRecord] = []
        var firstMissing: Int?
        for (offset, section) in sections.enumerated() {
            if let row = storedByIndex[section.index],
               row.contentHash == section.contentHash,
               !row.summary.isEmpty {
                chapters.append(row)
            } else if firstMissing == nil {
                firstMissing = offset
            }
        }
        if let firstMissing {
            // 目录修复或正文变化导致摘要与当前章节对不上：退回逐章阶段，
            // 不能拿旧摘要生成全书摘要。
            try store.upsertSummaryJob(SummaryJobRecord(
                docID: document.docID, status: .pending, nextSection: firstMissing,
                totalSections: sections.count,
                lastError: "章节摘要与当前目录不一致，需要重新摘要",
                updatedAt: now.timeIntervalSince1970))
            throw AgentError.storage("章节摘要与当前目录不一致，已回到逐章处理")
        }
        guard chapters.count == sections.count else {
            throw AgentError.storage("本书尚未生成完整的章节摘要")
        }

        let lines = chapters.map { "- [\($0.heading)] \($0.summary)" }
        let budget = Self.segmentBudget(contextWindow: llm.contextWindow)
        let groups = Self.splitSegments(lines.joined(separator: "\n\n"),
                                        tokenBudget: budget)
        guard !groups.isEmpty else {
            throw AgentError.storage("本书没有可汇总的章节摘要")
        }

        let persisted = try store.bookSummarySegments(docID: document.docID)
        var groupSummaries: [String] = []
        for (groupIndex, group) in groups.enumerated() {
            try Self.ensureNotCancelled(isCancelled)
            let hash = Self.shortHash(group)
            if let reused = persisted.first(where: {
                $0.groupIndex == groupIndex
                    && $0.groupHash == hash
                    && !$0.summary.isEmpty
            }) {
                groupSummaries.append(reused.summary)
                continue
            }
            let prompt = """
            以下是本书部分章节的摘要。请合并为 150-250 字的内容梗概，保留主线、结构与关键结论，只输出正文。

            \(group)
            """
            let raw = try await complete(
                system: "你是专业的书籍摘要助手，只输出正文。",
                prompt: prompt, llm: llm, temperature: 0.2,
                isCancelled: isCancelled)
            let summary = Self.normalizeSummary(raw, maxLength: 300, preferredCut: 100)
            guard !summary.isEmpty else {
                throw AgentError.llm("辅助模型未返回有效的分段摘要")
            }
            try store.upsertBookSummarySegment(BookSummarySegmentRecord(
                docID: document.docID, groupIndex: groupIndex, groupHash: hash,
                summary: summary, updatedAt: now.timeIntervalSince1970))
            groupSummaries.append(summary)
        }

        try Self.ensureNotCancelled(isCancelled)
        let reducePrompt = """
        以下是全书各部分的梗概。请生成 300-500 字的中文全书摘要，覆盖主题、结构、核心观点与结论，只输出正文。

        \(groupSummaries.enumerated().map { "\($0.offset + 1). \($0.element)" }
            .joined(separator: "\n\n"))
        """
        let raw = try await complete(
            system: "你是专业的书籍摘要助手，只输出正文。",
            prompt: reducePrompt, llm: llm, temperature: 0.2,
            isCancelled: isCancelled)
        var summary = Self.normalizeBookSummary(raw)
        if summary.count < 100 {
            // 模型偶尔只回一行标题或极短概览；再给一次明确的长摘要机会，
            // 但只接受更长的结果，避免重试反而更差。
            try Self.ensureNotCancelled(isCancelled)
            let retryPrompt = """
            上一次全书摘要过短。请输出 300-500 字的中文全书摘要，覆盖主题、结构、核心观点与结论，只输出正文。

            上一次结果：\(summary)
            """
            let retryRaw = try await complete(
                system: "你是专业的书籍摘要助手，只输出正文。",
                prompt: retryPrompt, llm: llm, temperature: 0.2,
                isCancelled: isCancelled)
            let retry = Self.normalizeBookSummary(retryRaw)
            if retry.count > summary.count { summary = retry }
        }
        // 只返回标签（如「摘要：」）或空内容时不能落库，否则任务会被当成完成，
        // 候选队列永远不再重试这本书。
        guard !summary.isEmpty else {
            throw AgentError.llm("辅助模型未返回有效的全书摘要")
        }
        let sourceHash = Self.shortHash(chapters.map {
            "\($0.sectionIndex):\($0.contentHash):\($0.summary)"
        }.joined(separator: "\n"))
        let record = BookSummaryRecord(
            docID: document.docID, summary: summary, outline: resolved.digest,
            chapterCount: sections.count, sourceHash: sourceHash,
            updatedAt: now.timeIntervalSince1970)
        try store.upsertBookSummary(record)
        try store.upsertSummaryJob(SummaryJobRecord(
            docID: document.docID, status: .completed,
            nextSection: max(sections.count, chapters.count),
            totalSections: sections.count,
            lastError: "", updatedAt: now.timeIntervalSince1970))
        return record
    }

    // MARK: - 辅助模型调用

    private func complete(system: String, prompt: String, llm: LLMService,
                          temperature: Double,
                          isCancelled: () -> Bool) async throws -> String {
        try Self.ensureNotCancelled(isCancelled)
        let messages = [
            ChatMessage(role: .system, content: system),
            ChatMessage(role: .user, content: prompt),
        ]
        var output = ""
        for try await event in llm.chatStream(messages: messages, tools: [],
                                              temperature: temperature) {
            try Self.ensureNotCancelled(isCancelled)
            if event.kind == .content { output += event.content }
        }
        let trimmed = output.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw AgentError.llm("辅助模型未返回摘要内容")
        }
        return trimmed
    }

    // MARK: - 工具

    /// 第一个没有“内容哈希匹配且非空”摘要的章节下标；全部有效时返回总数。
    static func firstMissingSectionIndex(sections: [BookSection],
                                         storedByIndex: [Int: ChapterSummaryRecord]) -> Int {
        for (offset, section) in sections.enumerated() {
            guard let row = storedByIndex[section.index],
                  row.contentHash == section.contentHash,
                  !row.summary.isEmpty else { return offset }
        }
        return sections.count
    }

    private static func segmentBudget(contextWindow: Int) -> Int {
        min(maxSegmentBudget, max(1024, contextWindow - 1400))
    }

    private static func ensureNotCancelled(_ isCancelled: () -> Bool) throws {
        if Task.isCancelled || isCancelled() {
            throw AgentError.cancelled
        }
    }

    private static func isCancellation(_ error: Error,
                                       isCancelled: () -> Bool) -> Bool {
        if isCancelled() || Task.isCancelled { return true }
        if case AgentError.cancelled = error { return true }
        return false
    }

    private static func shortHash(_ text: String) -> String {
        let data = Data(text.utf8)
        return SHA256.hash(data: data).prefix(8)
            .map { String(format: "%02x", $0) }
            .joined()
    }

    private static func tokenCost(_ character: Character) -> Double {
        isCJK(character) ? 1.0 : 0.25
    }

    private static func isSentenceEnd(_ character: Character) -> Bool {
        "。！？；.!?;\n".contains(character)
    }

    private static func isCJK(_ character: Character) -> Bool {
        guard character.unicodeScalars.count == 1,
              let scalar = character.unicodeScalars.first else { return false }
        switch scalar.value {
        case 0x2E80...0x9FFF, 0xF900...0xFAFF, 0xFF00...0xFFEF,
             0x3040...0x30FF, 0xAC00...0xD7AF, 0x20000...0x2FA1F:
            return true
        default:
            return false
        }
    }
}
