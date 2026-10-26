import XCTest
@testable import BaseAgentCore

final class ChunkingTests: XCTestCase {
    func testOutlineAndStructuredChunks() async {
        let text = """
        # 第一章 概述

        这是第一章的第一段。人工智能正在快速发展，很多应用已经落地。第二句话继续补充内容。

        ## 1.1 背景

        背景部分包含许多细节。我们需要理解历史、现状和未来趋势。这里再补充一些内容，让文本长度超过切片阈值。

        # 第二章 方法

        第二章介绍方法。这里包含实验设计、数据采集和评估指标。最后给出结论。
        """
        let nodes = DocumentOutline.extract(text)
        XCTAssertEqual(nodes.count, 3)
        XCTAssertEqual(nodes[0].title, "第一章 概述")
        XCTAssertEqual(nodes[2].level, 1)

        let options = ChunkingOptions(chunkSizeTokens: 12, overlapTokens: 3,
                                      minChunkChars: 10, maxChunkChars: 500,
                                      enableSemantic: false)
        let document = ParsedDocument(text: text, source: "doc.md")
        let chunks = await TextChunker(options: options).chunk(document: document, variant: .hybrid)
        XCTAssertGreaterThan(chunks.count, 1)
        XCTAssertEqual(chunks.map(\.chunkIndex), Array(0..<chunks.count))
        XCTAssertTrue(chunks.contains { $0.heading.contains("第一章") })
        XCTAssertTrue(chunks.allSatisfy { !$0.text.isEmpty })
        XCTAssertTrue(chunks.allSatisfy { $0.text.count <= options.maxChunkChars + 80 },
                      "chunk 不应突破字符上限")
    }

    /// 分页标记与页码不进入切片，溯源只保留文件名（source）。
    func testChunksStripPageMarkersWithoutPagesMetadata() async {
        let text = """
        <!-- page 3 -->

        这是第一页的正文内容，讲述基本概念与背景信息，长度足够形成切片。

        # 第一章 概述

        <!-- page 4 -->

        这是第二页的正文内容，继续补充细节与示例说明，长度同样足够。

        <!-- page 8 -->

        这是第八页的正文内容，与前文不连续，页码应压缩为独立区间。
        """
        let options = ChunkingOptions(chunkSizeTokens: 200, overlapTokens: 0,
                                      minChunkChars: 10, maxChunkChars: 500,
                                      enableSemantic: false)
        let document = ParsedDocument(text: text, source: "book.pdf")
        let chunks = await TextChunker(options: options).chunk(document: document)
        XCTAssertFalse(chunks.isEmpty)
        XCTAssertTrue(chunks.allSatisfy { !$0.text.contains("<!-- page") },
                      "切片文本不应包含分页标记")
        XCTAssertTrue(chunks.allSatisfy { $0.metadata.isEmpty },
                      "切片不再记录页码元数据")
        XCTAssertTrue(chunks.allSatisfy { $0.source == "book.pdf" },
                      "溯源只保留文件名")
        // 正文完整保留。
        XCTAssertTrue(chunks.contains { $0.text.contains("第一页的正文") })
    }

    /// 页码/年份正文不得被识别成章节标题（曾导致整本书退化为按页切片）。
    func testOutlineRejectsPageHeadersAndDateProse() {
        let text = """
        # 2 / 164

        # 164

        1898 年拜耶林克发现烟草花叶病毒

        # 1898 年，烟草花叶病毒的发现过程

        2 / 164

        第 3 页

        Page 4 of 10

        # 第一章 绪论
        """
        let nodes = DocumentOutline.extract(text)
        XCTAssertEqual(nodes.map(\.title), ["第一章 绪论"])
    }

    /// 长章节必须走语义切片；语义片即使短于 minChunkChars 也不能被合并。
    func testSemanticBoundariesSurviveMinChunkChars() async {
        let text = """
        苹果是一种常见的水果，富含维生素和膳食纤维。香蕉也是一种常见的水果，口感香甜且价格实惠。今天天气晴朗，阳光明媚，适合外出散步。天空没有云朵，微风轻拂，让人心情愉快。
        """
        let options = ChunkingOptions(chunkSizeTokens: 80, overlapTokens: 0,
                                      minChunkChars: 80, maxChunkChars: 500,
                                      semanticThreshold: 0.55, enableSemantic: true)
        let document = ParsedDocument(text: text, source: "fruit.md")
        // 前两句水果、后两句天气：主题之间相似度为 0，主题内部为 1。
        let similarity: SemanticChunker.AdjacentSimilarity = { sentences in
            (0..<(sentences.count - 1)).map { index in
                let a = sentences[index]
                let b = sentences[index + 1]
                let fruitA = a.contains("苹果") || a.contains("香蕉")
                let fruitB = b.contains("苹果") || b.contains("香蕉")
                return fruitA == fruitB ? 1.0 : 0.0
            }
        }
        let chunks = await TextChunker(options: options, similarity: similarity)
            .chunk(document: document, variant: .hybrid)
        XCTAssertEqual(chunks.count, 2, chunks.map(\.text).joined(separator: " | "))
        XCTAssertTrue(chunks[0].text.contains("苹果"))
        XCTAssertTrue(chunks[1].text.contains("天空"))
    }

    /// 不超过切片大小设置的短章节整章作为一个切片，不做二次切分。
    func testShortSectionKeptWhole() async {
        let text = """
        苹果是一种常见的水果。香蕉也是一种常见的水果。今天天气晴朗。天空没有云朵。
        """
        let options = ChunkingOptions(chunkSizeTokens: 200, overlapTokens: 40,
                                      minChunkChars: 1, maxChunkChars: 500,
                                      enableSemantic: true)
        let document = ParsedDocument(text: text, source: "short.md")
        let chunks = await TextChunker(options: options)
            .chunk(document: document, variant: .hybrid)
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks[0].text, text.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    /// 页码标记与各种页码形态在清洗阶段全部删除，任意位置都不保留。
    func testCleanerRemovesPageMarkersAndPageNumbersAnywhere() {
        let input = """
        <!-- page 1 -->
        正文第一段。

        2 / 164

        # 2 / 164

        第 3 页

        Page 4 of 10

        <!-- page 2 -->
        正文第二段。
        """
        let cleaned = DocumentCleaner.clean(input)
        XCTAssertFalse(cleaned.contains("<!-- page"), cleaned)
        XCTAssertFalse(cleaned.contains("2 / 164"), cleaned)
        XCTAssertFalse(cleaned.contains("第 3 页"), cleaned)
        XCTAssertFalse(cleaned.contains("Page 4"), cleaned)
        XCTAssertTrue(cleaned.contains("正文第一段"), cleaned)
        XCTAssertTrue(cleaned.contains("正文第二段"), cleaned)
    }

    func testSemanticChunkerSplitsAtLowSimilarity() async {
        let text = """
        苹果是一种水果。香蕉也是一种水果。今天天气晴朗。天空没有云朵。
        """
        let options = ChunkingOptions(chunkSizeTokens: 8, overlapTokens: 2,
                                      minChunkChars: 1, maxChunkChars: 400,
                                      semanticThreshold: 0.9, enableSemantic: true)
        let document = ParsedDocument(text: text, source: "fruit.md")
        let similarity: SemanticChunker.AdjacentSimilarity = { sentences in
            Array(repeating: 0.0, count: max(0, sentences.count - 1))
        }
        let chunks = await TextChunker(options: options, similarity: similarity)
            .chunk(document: document, variant: .semantic)
        XCTAssertFalse(chunks.isEmpty)
        XCTAssertTrue(chunks.allSatisfy { !$0.text.isEmpty })
    }

    func testDocumentCleaner() {
        let dirty = "第 1 页\n正文内容第一行。\n正文内容第二行。\n\n\n\n第 2 页\n更多内容。\n第 3 页"
        let cleaned = DocumentCleaner.clean(dirty, keepHeadersFooters: false)
        XCTAssertFalse(cleaned.contains("第 1 页"))
        XCTAssertFalse(cleaned.contains("第 3 页"))
        XCTAssertTrue(cleaned.contains("正文内容第一行"))
        XCTAssertFalse(cleaned.contains("\n\n\n\n"))
    }

    func testOutlineEnglishBracketAndCommaPatterns() {
        let text = """
        Chapter 1 Introduction

        第一章 引言，研究背景

        （一）数据准备

        Part II Methods

        Section 3.1 Sampling

        第二章开始，我们讨论了实验设计、数据采集和评估指标的完整流程，并给出了分析结果与改进方向
        """
        let nodes = DocumentOutline.extract(text)
        let titles = nodes.map(\.title)
        XCTAssertTrue(titles.contains("Chapter 1 Introduction"), "英文 Chapter 应识别")
        XCTAssertTrue(titles.contains { $0.contains("引言") && $0.contains("第一章") },
                      "含逗号的短中文标题应保留")
        XCTAssertTrue(titles.contains { $0.contains("数据准备") }, "括号序号（一）应识别")
        XCTAssertTrue(titles.contains("Part II Methods"), "英文 Part + 罗马数字应识别")
        XCTAssertTrue(titles.contains { $0.contains("Sampling") }, "英文 Section 应识别为二级")
        XCTAssertFalse(titles.contains {
            $0.contains("实验设计") && $0.contains("分析结果")
        }, "长正文句子不应被误判为标题")
        let section = nodes.first { $0.title.contains("Sampling") }
        XCTAssertEqual(section?.level, 2)
    }
}
