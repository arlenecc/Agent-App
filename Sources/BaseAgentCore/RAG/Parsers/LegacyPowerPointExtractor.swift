import Foundation

/// PowerPoint 97–2003 `.ppt` 提取。
///
/// 只读取 `PowerPoint Document` 流的记录树：`SlideListWithText` 里的
/// `SlidePersistAtom` 决定幻灯片顺序，`TextHeaderAtom` 区分标题/正文/备注，
/// `TextCharsAtom` / `TextBytesAtom` 提供文本。列表缺失时回退到文件中的
/// `Slide` 容器顺序。所有偏移、递归深度和记录数量都有明确上界。
struct LegacyPowerPointExtractor {

    struct Output {
        let markdown: String
        let outline: [OutlineEntry]
        let metadata: [String: String]
    }

    // MARK: - 记录类型

    private enum RecordType {
        static let document: UInt16 = 0x03E8
        static let slide: UInt16 = 0x03EE
        static let slideAtom: UInt16 = 0x03EF
        static let notes: UInt16 = 0x03F0
        static let notesAtom: UInt16 = 0x03F1
        static let slidePersistAtom: UInt16 = 0x03F3
        static let textHeaderAtom: UInt16 = 0x0F9F
        static let textCharsAtom: UInt16 = 0x0FA0
        static let textBytesAtom: UInt16 = 0x0FA8
        static let slideListWithText: UInt16 = 0x0FF0
        static let containerVersion: UInt8 = 0x0F
    }

    /// TextHeaderAtom 的标题类型（0 = 标题，4 = 其他标题占位符）。
    private static let titleTextTypes: Set<UInt32> = [0, 4]
    /// TextHeaderAtom 的备注类型。
    private static let notesTextType: UInt32 = 2

    private struct TextBlock {
        let headerType: UInt32?
        let paragraphs: [String]
        let atomCount: Int
    }

    private struct SlideContainer {
        let slideId: UInt32
        let blocks: [TextBlock]
    }

    private struct NotesContainer {
        let slideIdRef: UInt32
        let blocks: [TextBlock]
    }

    private struct SlideTextGroup {
        let slideId: UInt32
        let blocks: [TextBlock]
    }

    private struct ScanState {
        var recordCount = 0
        var slides: [SlideContainer] = []
        var notes: [NotesContainer] = []
        var slideGroups: [SlideTextGroup] = []
    }

    private struct RecordHeader {
        let version: UInt8
        let instance: UInt16
        let type: UInt16
        let payload: Range<Int>
    }

    // MARK: - 入口

    func extract(url: URL) throws -> Output {
        do {
            return try extractInternal(url: url)
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as AgentError {
            throw error
        } catch {
            throw AgentError.parse(
                "旧版 PowerPoint 解析失败（\(url.lastPathComponent)）：\(error.localizedDescription)")
        }
    }

    private func extractInternal(url: URL) throws -> Output {
        let fileName = url.lastPathComponent
        let values = try url.resourceValues(forKeys: [.fileSizeKey])
        if let size = values.fileSize, size > LegacyOfficeLimits.maxFileSize {
            throw AgentError.parse(
                "PowerPoint 文档超过 \(LegacyOfficeLimits.maxFileSize / 1024 / 1024)MB 上限"
                + "（\(fileName)）")
        }

        let compound: OLE2CompoundFile
        do {
            compound = try OLE2CompoundFile.open(url: url)
        } catch {
            throw AgentError.parse(
                "旧版 PowerPoint 解析失败（\(fileName)）：\(error.localizedDescription)")
        }

        if compound.containsStream(named: "EncryptionInfo")
            || compound.containsStream(named: "EncryptedSummary") {
            throw AgentError.unsupported(
                "PowerPoint 文档已加密或受密码保护（\(fileName)）")
        }
        guard compound.containsStream(named: "PowerPoint Document") else {
            throw AgentError.parse(
                "不是有效的 PowerPoint 97–2003 文件（\(fileName)）："
                + "缺少 PowerPoint Document 流")
        }

        let stream: Data
        do {
            stream = try compound.streamData(named: "PowerPoint Document")
        } catch {
            throw AgentError.parse(
                "PowerPoint 文档流读取失败（\(fileName)）：\(error.localizedDescription)")
        }
        guard stream.count >= 8 else {
            throw AgentError.parse("PowerPoint 记录头不完整（\(fileName)）")
        }
        guard let firstHeader = Self.uint16(stream, at: 0),
              let firstType = Self.uint16(stream, at: 2),
              firstType == RecordType.document,
              UInt8(firstHeader & 0x000F) == RecordType.containerVersion else {
            throw AgentError.unsupported(
                "仅支持 PowerPoint 97–2003（PPT）格式，"
                + "PowerPoint 95 及更早版本暂不支持（\(fileName)）")
        }

        var state = ScanState()
        try scan(stream, range: 0..<stream.count, depth: 0,
                 state: &state, fileName: fileName)

        let allSlides = selectSlides(from: state)
        let slideLimitHit = allSlides.count > LegacyOfficeLimits.maxSlides
        let slides = Array(allSlides.prefix(LegacyOfficeLimits.maxSlides))
        let notes = notesForSlides(slides, notes: state.notes)

        var sections: [String] = []
        var outline: [OutlineEntry] = []
        var notesSlides = 0
        var textAtomCount = 0
        var hasContent = false

        for (index, slide) in slides.enumerated() {
            let rendered = renderSlide(slide, number: index + 1, notes: notes[index])
            sections.append(rendered.section)
            outline.append(OutlineEntry(level: 2, title: rendered.outlineTitle))
            if rendered.hasNotes { notesSlides += 1 }
            if rendered.hasContent { hasContent = true }
            textAtomCount += slide.blocks.reduce(0) { $0 + $1.atomCount }
            textAtomCount += notes[index].reduce(0) { $0 + $1.atomCount }
        }

        guard hasContent else {
            throw AgentError.parse("PowerPoint 文档未解析出正文（\(fileName)）")
        }

        var metadata: [String: String] = [
            "engine": "ppt",
            "legacy_format": "ppt-97",
            "slides": String(sections.count),
            "text_atoms": String(textAtomCount),
        ]
        if notesSlides > 0 { metadata["notes_slides"] = String(notesSlides) }
        if slideLimitHit {
            metadata["slide_limit_hit"] = "true"
            metadata["truncated"] = "true"
            metadata["truncated_reason"] = "slide_limit"
            AgentLog.warning(
                "PowerPoint 幻灯片超过 \(LegacyOfficeLimits.maxSlides) 页，已截断：\(fileName)")
        }

        var markdown = sections.joined(separator: "\n\n")
        let limited = LegacyOfficeLimits.limitText(markdown)
        if limited.truncated {
            markdown = limited.text
            metadata["truncated"] = "true"
            if metadata["truncated_reason"] == nil {
                metadata["truncated_reason"] = "text_limit"
            }
            AgentLog.warning(
                "PowerPoint 正文超过 \(LegacyOfficeLimits.maxTextCharacters) 字符，已截断："
                + fileName)
        }

        return Output(markdown: markdown, outline: outline, metadata: metadata)
    }

    // MARK: - 记录树遍历

    private func scan(_ data: Data, range: Range<Int>, depth: Int,
                      state: inout ScanState, fileName: String) throws {
        guard depth <= LegacyOfficeLimits.maxRecordDepth else {
            throw depthLimitError(fileName)
        }
        var offset = range.lowerBound
        while offset < range.upperBound {
            let header = try readHeader(data, at: offset, end: range.upperBound,
                                         fileName: fileName)
            try countRecord(&state, fileName: fileName)

            if header.version == RecordType.containerVersion {
                switch header.type {
                case RecordType.document:
                    try scan(data, range: header.payload, depth: depth + 1,
                             state: &state, fileName: fileName)
                case RecordType.slide:
                    try processSlide(data, payload: header.payload, depth: depth + 1,
                                     state: &state, fileName: fileName)
                case RecordType.notes:
                    try processNotes(data, payload: header.payload, depth: depth + 1,
                                     state: &state, fileName: fileName)
                case RecordType.slideListWithText where header.instance == 0:
                    try processSlideList(data, payload: header.payload, depth: depth + 1,
                                         state: &state, fileName: fileName)
                default:
                    try scan(data, range: header.payload, depth: depth + 1,
                             state: &state, fileName: fileName)
                }
            }
            offset = header.payload.upperBound
        }
    }

    /// 遍历一个容器内的文本记录；`onAtom` 用于收集 SlideAtom / NotesAtom。
    private func walkText(_ data: Data, range: Range<Int>, depth: Int,
                          state: inout ScanState, fileName: String,
                          collector: inout TextCollector,
                          onAtom: (UInt16, Data) -> Void = { _, _ in }) throws {
        guard depth <= LegacyOfficeLimits.maxRecordDepth else {
            throw depthLimitError(fileName)
        }
        var offset = range.lowerBound
        while offset < range.upperBound {
            let header = try readHeader(data, at: offset, end: range.upperBound,
                                         fileName: fileName)
            try countRecord(&state, fileName: fileName)

            if header.version == RecordType.containerVersion {
                try walkText(data, range: header.payload, depth: depth + 1,
                             state: &state, fileName: fileName,
                             collector: &collector, onAtom: onAtom)
            } else {
                let payload = data[header.payload]
                switch header.type {
                case RecordType.textHeaderAtom:
                    if let type = Self.uint32(payload, at: 0) {
                        collector.begin(type)
                    }
                case RecordType.textCharsAtom:
                    collector.append(Self.decodeChars(payload))
                case RecordType.textBytesAtom:
                    collector.append(Self.decodeBytes(payload))
                default:
                    onAtom(header.type, payload)
                }
            }
            offset = header.payload.upperBound
        }
    }

    private func processSlide(_ data: Data, payload: Range<Int>, depth: Int,
                              state: inout ScanState, fileName: String) throws {
        var slideId: UInt32 = 0
        var collector = TextCollector()
        try walkText(data, range: payload, depth: depth, state: &state,
                     fileName: fileName, collector: &collector) { type, atom in
            if type == RecordType.slideAtom, let value = Self.uint32(atom, at: 8) {
                slideId = value
            }
        }
        state.slides.append(SlideContainer(slideId: slideId,
                                           blocks: collector.finish()))
    }

    private func processNotes(_ data: Data, payload: Range<Int>, depth: Int,
                              state: inout ScanState, fileName: String) throws {
        var slideIdRef: UInt32 = 0
        var collector = TextCollector()
        try walkText(data, range: payload, depth: depth, state: &state,
                     fileName: fileName, collector: &collector) { type, atom in
            if type == RecordType.notesAtom, let value = Self.uint32(atom, at: 0) {
                slideIdRef = value
            }
        }
        state.notes.append(NotesContainer(slideIdRef: slideIdRef,
                                          blocks: collector.finish()))
    }

    private func processSlideList(_ data: Data, payload: Range<Int>, depth: Int,
                                  state: inout ScanState, fileName: String) throws {
        guard depth <= LegacyOfficeLimits.maxRecordDepth else {
            throw depthLimitError(fileName)
        }

        var groups: [SlideTextGroup] = []
        var currentId: UInt32?
        var collector = TextCollector()

        func flush() {
            guard let id = currentId else { return }
            groups.append(SlideTextGroup(slideId: id, blocks: collector.finish()))
            collector = TextCollector()
        }

        var offset = payload.lowerBound
        while offset < payload.upperBound {
            let header = try readHeader(data, at: offset, end: payload.upperBound,
                                         fileName: fileName)
            try countRecord(&state, fileName: fileName)

            if header.version == RecordType.containerVersion {
                try walkText(data, range: header.payload, depth: depth + 1,
                             state: &state, fileName: fileName,
                             collector: &collector)
            } else if header.type == RecordType.slidePersistAtom {
                flush()
                currentId = Self.uint32(data[header.payload], at: 12) ?? 0
            } else {
                let atom = data[header.payload]
                switch header.type {
                case RecordType.textHeaderAtom:
                    if let type = Self.uint32(atom, at: 0) { collector.begin(type) }
                case RecordType.textCharsAtom:
                    collector.append(Self.decodeChars(atom))
                case RecordType.textBytesAtom:
                    collector.append(Self.decodeBytes(atom))
                default:
                    break
                }
            }
            offset = header.payload.upperBound
        }
        flush()
        state.slideGroups.append(contentsOf: groups)
    }

    private func countRecord(_ state: inout ScanState, fileName: String) throws {
        state.recordCount += 1
        guard state.recordCount <= LegacyOfficeLimits.maxRecordCount else {
            throw AgentError.parse(
                "PowerPoint 记录数量超过 \(LegacyOfficeLimits.maxRecordCount) 条上限"
                + "（\(fileName)）")
        }
        if state.recordCount % 4096 == 0, Task.isCancelled {
            throw CancellationError()
        }
    }

    /// 读取记录头并校验负载边界。
    private func readHeader(_ data: Data, at offset: Int, end: Int,
                            fileName: String) throws -> RecordHeader {
        guard offset >= 0, end <= data.count, offset <= end else {
            throw AgentError.parse("PowerPoint 记录偏移越界（\(fileName)）")
        }
        if end - offset < 8 {
            if Self.isZeroPadding(data, from: offset, to: end) {
                return RecordHeader(version: 0, instance: 0, type: 0,
                                    payload: end..<end)
            }
            throw AgentError.parse("PowerPoint 记录头不完整（\(fileName)）")
        }
        guard let raw = Self.uint16(data, at: offset),
              let type = Self.uint16(data, at: offset + 2),
              let length = Self.uint32(data, at: offset + 4) else {
            throw AgentError.parse("PowerPoint 记录头不完整（\(fileName)）")
        }
        let payloadStart = offset + 8
        let payloadLength = Int(length)
        guard payloadLength <= end - payloadStart else {
            throw AgentError.parse("PowerPoint 记录长度越界（\(fileName)）")
        }
        return RecordHeader(version: UInt8(raw & 0x000F),
                            instance: raw >> 4,
                            type: type,
                            payload: payloadStart..<(payloadStart + payloadLength))
    }

    private func depthLimitError(_ fileName: String) -> AgentError {
        AgentError.parse(
            "PowerPoint 记录嵌套深度超过 \(LegacyOfficeLimits.maxRecordDepth) 层上限"
            + "（\(fileName)）")
    }

    // MARK: - 幻灯片与备注组装

    /// 优先使用 SlideListWithText 的顺序；该列表缺失时按文件中的 Slide 容器顺序。
    private func selectSlides(from state: ScanState) -> [SlideContainer] {
        guard !state.slideGroups.isEmpty else {
            return state.slides
        }

        var containerByID: [UInt32: SlideContainer] = [:]
        for slide in state.slides where slide.slideId != 0 {
            if containerByID[slide.slideId] == nil {
                containerByID[slide.slideId] = slide
            }
        }

        return state.slideGroups.enumerated().map { index, group in
            if !group.blocks.isEmpty {
                return SlideContainer(slideId: group.slideId, blocks: group.blocks)
            }
            if let container = containerByID[group.slideId] {
                return container
            }
            if group.slideId == 0, index < state.slides.count {
                return state.slides[index]
            }
            return SlideContainer(slideId: group.slideId, blocks: [])
        }
    }

    /// 备注按 NotesAtom 的 slideIdRef 挂到对应幻灯片；全部缺失关联时按顺序兜底。
    private func notesForSlides(_ slides: [SlideContainer],
                                notes: [NotesContainer]) -> [[TextBlock]] {
        var result = Array(repeating: [TextBlock](), count: slides.count)
        var used = Set<Int>()

        for (slideIndex, slide) in slides.enumerated() where slide.slideId != 0 {
            for (noteIndex, note) in notes.enumerated()
            where !used.contains(noteIndex) && note.slideIdRef == slide.slideId {
                result[slideIndex].append(contentsOf: note.blocks)
                used.insert(noteIndex)
            }
        }

        if used.isEmpty, notes.count == slides.count {
            for index in slides.indices {
                result[index] = notes[index].blocks
            }
        }
        return result
    }

    private func renderSlide(_ slide: SlideContainer, number: Int,
                             notes: [TextBlock])
        -> (section: String, outlineTitle: String,
            hasContent: Bool, hasNotes: Bool) {
        var title: String?
        var body: [String] = []

        for block in slide.blocks {
            if block.headerType == Self.notesTextType { continue }
            if let header = block.headerType, Self.titleTextTypes.contains(header),
               title == nil {
                let candidate = Self.titleText(block.paragraphs)
                if !candidate.isEmpty {
                    title = candidate
                    continue
                }
            }
            body.append(contentsOf: block.paragraphs)
        }

        var section = "## " + Self.outlineTitle(slideNumber: number, title: title)
        if !body.isEmpty {
            section += "\n\n" + body.joined(separator: "\n\n")
        }

        let noteParagraphs = notes.flatMap(\.paragraphs)
        if !noteParagraphs.isEmpty {
            section += "\n\n**演讲者备注**\n\n"
                + noteParagraphs.joined(separator: "\n\n")
        }

        return (section,
                Self.outlineTitle(slideNumber: number, title: title),
                title != nil || !body.isEmpty || !noteParagraphs.isEmpty,
                !noteParagraphs.isEmpty)
    }

    private static func outlineTitle(slideNumber: Int, title: String?) -> String {
        guard let title, !title.isEmpty else { return "Slide \(slideNumber)" }
        return "Slide \(slideNumber)：\(title)"
    }

    private static func titleText(_ paragraphs: [String]) -> String {
        paragraphs.joined(separator: " ")
            .replacingOccurrences(of: "\\s+", with: " ",
                                  options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - 文本 atom

    /// 文本块：一个 TextHeaderAtom 后紧跟的所有文本 atom 按顺序拼接。
    private struct TextCollector {
        private var blocks: [TextBlock] = []
        private var headerType: UInt32?
        private var pending = ""
        private var atomCount = 0

        mutating func begin(_ type: UInt32) {
            flush()
            headerType = type
        }

        mutating func append(_ text: String) {
            pending += text
            atomCount += 1
        }

        mutating func finish() -> [TextBlock] {
            flush()
            return blocks
        }

        private mutating func flush() {
            defer {
                headerType = nil
                pending = ""
                atomCount = 0
            }
            let paragraphs = Self.paragraphs(from: pending)
            guard !paragraphs.isEmpty else { return }
            blocks.append(TextBlock(headerType: headerType,
                                    paragraphs: paragraphs,
                                    atomCount: atomCount))
        }

        private static func paragraphs(from text: String) -> [String] {
            let normalized = text
                .replacingOccurrences(of: "\r\n", with: "\n")
                .replacingOccurrences(of: "\r", with: "\n")
                .replacingOccurrences(of: "\u{000B}", with: "\n")
            return normalized
                .components(separatedBy: "\n")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
        }
    }

    private static func decodeChars(_ payload: Data) -> String {
        let count = payload.count / 2
        guard count > 0 else { return "" }
        var units: [UInt16] = []
        units.reserveCapacity(count)
        for index in 0..<count {
            if let unit = uint16(payload, at: index * 2) {
                units.append(unit)
            }
        }
        return String(decoding: units, as: UTF16.self)
    }

    private static func decodeBytes(_ payload: Data) -> String {
        if let text = String(data: payload, encoding: .windowsCP1252) {
            return text
        }
        if let text = String(data: payload, encoding: .utf8) {
            return text
        }
        return String(data: payload, encoding: .isoLatin1) ?? ""
    }

    // MARK: - 无符号读取（相对 Data.startIndex）

    private static func uint16(_ data: Data, at offset: Int) -> UInt16? {
        let start = data.startIndex + offset
        guard offset >= 0, start >= data.startIndex, start + 2 <= data.endIndex else {
            return nil
        }
        return UInt16(data[start]) | (UInt16(data[start + 1]) << 8)
    }

    private static func uint32(_ data: Data, at offset: Int) -> UInt32? {
        let start = data.startIndex + offset
        guard offset >= 0, start >= data.startIndex, start + 4 <= data.endIndex else {
            return nil
        }
        return UInt32(data[start])
            | (UInt32(data[start + 1]) << 8)
            | (UInt32(data[start + 2]) << 16)
            | (UInt32(data[start + 3]) << 24)
    }

    private static func isZeroPadding(_ data: Data, from offset: Int, to end: Int) -> Bool {
        var index = data.startIndex + offset
        let endIndex = data.startIndex + end
        while index < endIndex {
            if data[index] != 0 { return false }
            index += 1
        }
        return true
    }
}
