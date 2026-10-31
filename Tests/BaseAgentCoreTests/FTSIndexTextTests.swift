import XCTest
@testable import BaseAgentCore

final class FTSIndexTextTests: XCTestCase {
    func testBatchMatchesPerTextResult() {
        let texts = [
            "检索增强生成把向量检索与大语言模型结合起来。",
            "cloud native container orchestration",
            "",
            "   ",
            "第 3 章 部署 运维 监控",
        ]
        XCTAssertEqual(TextAnalyzer.ftsIndexTexts(texts),
                       texts.map(TextAnalyzer.ftsIndexText))
    }

    func testBatchPreservesOrderOnParallelPath() {
        let texts = (0..<64).map { "第\($0) 节 云计算 容器 网络 存储 安全" }
        XCTAssertEqual(TextAnalyzer.ftsIndexTexts(texts),
                       texts.map(TextAnalyzer.ftsIndexText),
                       "并行分词必须按原索引回填，顺序不能乱")
    }

    func testBatchHandlesEmptyInput() {
        XCTAssertEqual(TextAnalyzer.ftsIndexTexts([]), [])
    }
}
