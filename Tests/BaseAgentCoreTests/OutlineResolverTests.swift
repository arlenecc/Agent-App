import XCTest
@testable import BaseAgentCore

final class OutlineResolverTests: XCTestCase {

    func testParserOutlineWinsWhenAnchoredToMarkdown() throws {
        let markdown = """
        # 第一章 引言

        第一段正文。

        ## 1.1 背景

        背景正文。

        # 第二章 方法

        方法正文。
        """
        let parsed = """
        第一章 引言
          1.1 背景
        第二章 方法
        """
        let resolved = OutlineResolver.resolve(markdown: markdown,
                                               parsedOutline: parsed,
                                               docName: "book.md")
        XCTAssertEqual(resolved.source, "parser")
        XCTAssertEqual(resolved.nodes.map(\.title),
                       ["第一章 引言", "1.1 背景", "第二章 方法"])
        XCTAssertEqual(resolved.sections.count, 3)
    }

    func testParserOutlineFallsBackToRegexWhenUnanchored() throws {
        let markdown = """
        # 第一章 引言

        正文。

        ## 1.1 背景

        更多正文。
        """
        let parsed = """
        完全不存在的目录
          另一个不存在的条目
        """
        let resolved = OutlineResolver.resolve(markdown: markdown,
                                               parsedOutline: parsed,
                                               docName: "book.md")
        XCTAssertEqual(resolved.source, "regex")
        XCTAssertEqual(resolved.nodes.map(\.title), ["第一章 引言", "1.1 背景"])
    }

    func testConservativeRepairFillsDanglingAndDummyTitles() throws {
        let danglingMarkdown = """
        # 第一章

        研究方法概述

        正文内容。
        """
        let dangling = OutlineResolver.resolve(markdown: danglingMarkdown,
                                               parsedOutline: "第一章",
                                               docName: "book.md")
        XCTAssertEqual(dangling.nodes.first?.title, "第一章 研究方法概述")
        XCTAssertTrue(dangling.hadRepairs)

        let dummyMarkdown = """
        # 2 / 164

        真正的标题是这一行

        正文内容。
        """
        let dummy = OutlineResolver.resolve(markdown: dummyMarkdown,
                                            parsedOutline: "2 / 164",
                                            docName: "book.md")
        XCTAssertEqual(dummy.nodes.first?.title, "真正的标题是这一行")
        XCTAssertTrue(dummy.hadRepairs)
    }

    func testSectionsHaveStableIndexHeadingPathAndContentHash() throws {
        let markdown = """
        # 第一章 引言

        第一段。

        ## 1.1 背景

        第二段。

        # 第二章 方法

        第三段。
        """
        let first = OutlineResolver.resolve(markdown: markdown,
                                            parsedOutline: "",
                                            docName: "book.md")
        XCTAssertEqual(first.sections.map(\.index), [0, 1, 2])
        XCTAssertEqual(first.sections.map(\.heading),
                       ["第一章 引言", "第一章 引言 > 1.1 背景", "第二章 方法"])
        XCTAssertEqual(first.sections[0].body, "第一段。")
        XCTAssertEqual(first.sections[1].body, "第二段。")

        let second = OutlineResolver.resolve(markdown: markdown,
                                             parsedOutline: "",
                                             docName: "book.md")
        XCTAssertEqual(first.sections.map(\.contentHash), second.sections.map(\.contentHash))

        let changed = OutlineResolver.resolve(
            markdown: markdown.replacingOccurrences(of: "第二段。", with: "完全不同的段落。"),
            parsedOutline: "", docName: "book.md")
        XCTAssertNotEqual(first.sections[1].contentHash, changed.sections[1].contentHash)
        XCTAssertEqual(first.sections[0].contentHash, changed.sections[0].contentHash)
    }

    func testQualityScoreRewardsDepthAndRejectsNoise() throws {
        let shallow = [OutlineNode(level: 1, title: "第一章", start: 0, end: 100)]
        let deep = [
            OutlineNode(level: 1, title: "第一章", start: 0, end: 50),
            OutlineNode(level: 2, title: "1.1 小节", start: 50, end: 75),
            OutlineNode(level: 2, title: "1.2 小节", start: 75, end: 100),
        ]
        let shallowScore = OutlineResolver.qualityScore(
            nodes: shallow, totalEntries: 1, anchoredEntries: 1,
            noiseEntries: 0, textLength: 100)
        let deepScore = OutlineResolver.qualityScore(
            nodes: deep, totalEntries: 3, anchoredEntries: 3,
            noiseEntries: 0, textLength: 100)
        XCTAssertGreaterThan(deepScore, shallowScore)

        let noisyScore = OutlineResolver.qualityScore(
            nodes: deep, totalEntries: 3, anchoredEntries: 1,
            noiseEntries: 2, textLength: 100)
        XCTAssertLessThan(noisyScore, deepScore)
        XCTAssertEqual(OutlineResolver.qualityScore(
            nodes: deep, totalEntries: 3, anchoredEntries: 0,
            noiseEntries: 0, textLength: 100), 0)
    }
}
