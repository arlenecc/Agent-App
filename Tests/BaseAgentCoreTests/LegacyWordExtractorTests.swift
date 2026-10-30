import AppKit
import XCTest
@testable import BaseAgentCore

final class LegacyWordExtractorTests: XCTestCase {

    private var directory: URL!

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-legacy-word-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory,
                                                withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    func testHeadingsBoldItalicAndChinese() throws {
        let url = try makeDocument([
            ("旧版 Word 测试标题", 18, true, false),
            ("这是正文第一段，包含中文内容。", 12, false, false),
            ("粗体强调", 12, true, false),
            ("斜体文字", 12, false, true),
        ])
        let output = try LegacyWordExtractor().extract(url: url)

        XCTAssertTrue(output.markdown.contains("# 旧版 Word 测试标题"), output.markdown)
        XCTAssertTrue(output.markdown.contains("这是正文第一段，包含中文内容。"), output.markdown)
        XCTAssertTrue(output.markdown.contains("**粗体强调**"), output.markdown)
        XCTAssertTrue(output.markdown.contains("*斜体文字*"), output.markdown)
        XCTAssertEqual(output.metadata["engine"], "doc")
        XCTAssertEqual(output.metadata["legacy_format"], "word-97")
        XCTAssertEqual(output.metadata["heading_detection"], "font-size")
        XCTAssertEqual(output.metadata["headings"], "1")
        XCTAssertGreaterThanOrEqual(Int(output.metadata["paragraphs"] ?? "0") ?? 0, 4)
    }

    func testHeadingLevelsFollowFontSize() throws {
        let url = try makeDocument([
            ("第一章 概述", 20, true, false),
            ("1.1 背景", 16, true, false),
            ("正文介绍项目背景与目标。", 12, false, false),
        ])
        let output = try LegacyWordExtractor().extract(url: url)

        XCTAssertTrue(output.markdown.contains("# 第一章 概述"), output.markdown)
        XCTAssertTrue(output.markdown.contains("## 1.1 背景"), output.markdown)
        XCTAssertTrue(output.outline.contains { $0.level == 1 && $0.title == "第一章 概述" })
    }

    /// `HTMLMarkdown` 也会把纯粗体段落（`<p><b>…</b></p>`）识别成标题；
    /// 这时 converted.headings 非空，metadata 和最终大纲必须使用同一个结果。
    func testHeadingMetadataMatchesFinalOutline() throws {
        let url = try makeDocument([
            ("第一章 概述", 18, true, false),
            ("正文第一段。", 12, false, false),
            ("独立粗体短句", 12, true, false),
        ])
        let output = try LegacyWordExtractor().extract(url: url)

        XCTAssertEqual(output.outline.map(\.title), ["第一章 概述", "独立粗体短句"])
        XCTAssertEqual(output.metadata["headings"], "2")
    }

    /// `HTMLMarkdown` 会忽略引用块里的标题；这些标题仍在启发式大纲中，
    /// 不能因为最终 HTML 已有其他 h 标签就把启发式结果整体丢掉。
    func testOutlineMergeKeepsHeuristicHeadingsMissingFromConverted() {
        let converted = [
            OutlineEntry(level: 1, title: "第一章 概述"),
            OutlineEntry(level: 3, title: "结尾"),
        ]
        let heuristic = [
            OutlineEntry(level: 2, title: "引用块标题"),
            OutlineEntry(level: 3, title: "结尾"),
        ]
        let markdown = """
        # 第一章 概述

        > ### 引用块标题

        ### 结尾
        """

        let merged = LegacyWordExtractor.mergeOutline(converted: converted,
                                                      heuristic: heuristic,
                                                      markdown: markdown)
        XCTAssertEqual(merged.map(\.title), ["第一章 概述", "引用块标题", "结尾"])
        XCTAssertEqual(merged.map(\.level), [1, 2, 3])
    }

    /// 转换后的标题可能带 Markdown 强调标记，去重时不能把它当成另一个标题。
    func testOutlineMergeTreatsMarkdownEmphasisAsSameHeading() {
        let converted = [OutlineEntry(level: 1, title: "**第一章**")]
        let heuristic = [OutlineEntry(level: 1, title: "第一章")]

        let merged = LegacyWordExtractor.mergeOutline(converted: converted,
                                                      heuristic: heuristic,
                                                      markdown: "# **第一章**")
        XCTAssertEqual(merged, converted)

        // 局部加粗会生成 `**前缀** 后缀`，同样不能重复计入大纲。
        let partial = LegacyWordExtractor.mergeOutline(
            converted: [OutlineEntry(level: 1, title: "**第一章** 概述")],
            heuristic: [OutlineEntry(level: 1, title: "第一章 概述")],
            markdown: "# **第一章** 概述")
        XCTAssertEqual(partial.map(\.title), ["**第一章** 概述"])
    }

    func testUniformBodyTextProducesNoHeading() throws {
        let url = try makeDocument([
            ("第一段都是正文，没有标题。", 12, false, false),
            ("第二段也使用同样的字号。", 12, false, false),
        ])
        let output = try LegacyWordExtractor().extract(url: url)

        XCTAssertFalse(output.markdown.split(separator: "\n").contains { line in
            line.hasPrefix("#")
        }, output.markdown)
        XCTAssertTrue(output.outline.isEmpty)
    }

    func testBulletPrefixBecomesMarkdownList() throws {
        let url = try makeDocument([
            ("• 条目一", 12, false, false),
            ("• 条目二", 12, false, false),
            ("正文继续。", 12, false, false),
        ])
        let output = try LegacyWordExtractor().extract(url: url)

        XCTAssertTrue(output.markdown.contains("- 条目一"), output.markdown)
        XCTAssertTrue(output.markdown.contains("- 条目二"), output.markdown)
    }

    /// HTML 导出前必须先按正文上限截断，否则超大文档会先生成完整个 HTML
    /// 再被 64MB 上限拒绝，内存峰值完全失控。
    func testHTMLExportTruncatesBeforeConversion() throws {
        let tail = "尾部标记不应出现。"
        let body = String(repeating: "正文内容。", count: 30) + tail
        let url = try makeDocument([
            ("第一章 概述", 18, true, false),
            (body, 12, false, false),
        ])

        var extractor = LegacyWordExtractor()
        extractor.htmlCharacterLimit = 60
        let output = try extractor.extract(url: url)

        XCTAssertEqual(output.metadata["truncated"], "true")
        XCTAssertEqual(output.metadata["truncated_reason"], "text_limit")
        XCTAssertFalse(output.markdown.contains(tail), output.markdown)
    }

    /// 截断 UTF-16 长度时不能把 emoji 的代理对切成两半。
    func testHTMLExportRangeDoesNotSplitSurrogatePair() {
        let attributed = NSAttributedString(string: "A😀B")
        XCTAssertEqual(attributed.length, 4)
        XCTAssertEqual(LegacyWordExtractor.htmlExportRange(
            for: attributed, characterLimit: 2).length, 1)
        XCTAssertEqual(LegacyWordExtractor.htmlExportRange(
            for: attributed, characterLimit: 3).length, 3)
    }

    /// AppKit 转换无法中断，但解析器至少要在进入/离开耗时阶段时观察取消信号。
    func testExtractionStopsWhenTaskIsCancelled() async throws {
        let url = try makeDocument([
            ("第一章 概述", 18, true, false),
            ("正文内容。", 12, false, false),
        ])

        let task = Task<Void, Error> {
            withUnsafeCurrentTask { $0?.cancel() }
            _ = try LegacyWordExtractor().extract(url: url)
        }
        do {
            try await task.value
            XCTFail("取消后不应继续解析 Word")
        } catch AgentError.cancelled {
            // 预期：取消信号被解析器观察到。
        } catch {
            XCTFail("应抛出 AgentError.cancelled，实际：\(error)")
        }
    }

    func testEmptyDocumentThrows() throws {
        let url = directory.appendingPathComponent("empty.doc")
        try Data().write(to: url)

        XCTAssertThrowsError(try LegacyWordExtractor().extract(url: url))
    }

    // MARK: - helpers

    private func makeDocument(_ paragraphs: [(String, CGFloat, Bool, Bool)]) throws -> URL {
        let attributed = NSMutableAttributedString()
        for (index, item) in paragraphs.enumerated() {
            let attributes: [NSAttributedString.Key: Any] = [
                .font: font(size: item.1, bold: item.2, italic: item.3),
            ]
            attributed.append(NSAttributedString(string: item.0, attributes: attributes))
            if index + 1 < paragraphs.count {
                attributed.append(NSAttributedString(
                    string: "\n", attributes: [.font: font(size: 12, bold: false, italic: false)]))
            }
        }

        let fullRange = NSRange(location: 0, length: attributed.length)
        let data = try attributed.data(from: fullRange,
                                       documentAttributes: [
                                           .documentType: NSAttributedString.DocumentType.docFormat,
                                       ])
        let url = directory.appendingPathComponent("sample-\(UUID().uuidString).doc")
        try data.write(to: url)
        return url
    }

    private func font(size: CGFloat, bold: Bool, italic: Bool) -> NSFont {
        var font = NSFont.systemFont(ofSize: size)
        if bold {
            font = NSFontManager.shared.convert(font, toHaveTrait: .boldFontMask)
        }
        if italic {
            font = NSFontManager.shared.convert(font, toHaveTrait: .italicFontMask)
        }
        return font
    }
}
