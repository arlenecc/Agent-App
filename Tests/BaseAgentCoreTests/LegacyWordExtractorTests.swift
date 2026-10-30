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
