import XCTest
@testable import BaseAgentCore

private struct GoldenFixture: Decodable {
    let sentences: [String]
    let tokenIDs: [[Int]]
    let vectors: [[Float]]

    enum CodingKeys: String, CodingKey {
        case sentences
        case tokenIDs = "token_ids"
        case vectors
    }

    static func load() throws -> GoldenFixture {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/bge-small-zh-golden.json")
        return try JSONDecoder().decode(GoldenFixture.self, from: Data(contentsOf: url))
    }
}

final class BGESentenceSimilarityTests: XCTestCase {
    func testBundledModelResourceIsPresent() throws {
        let directory = try XCTUnwrap(BGESentenceSimilarity.modelDirectory())
        let fm = FileManager.default
        for name in ["model_int8.onnx", "vocab.txt", "tokenizer.json",
                     "tokenizer_config.json", "config.json", "LICENSE"] {
            XCTAssertTrue(fm.fileExists(atPath: directory.appendingPathComponent(name).path), name)
        }
        let size = try fm.attributesOfItem(
            atPath: directory.appendingPathComponent("model_int8.onnx").path
        )[.size] as? Int64
        XCTAssertGreaterThan(size ?? 0, 20_000_000)
    }

    func testTokenizerMatchesHuggingFaceGoldenIDs() throws {
        let fixture = try GoldenFixture.load()
        let provider = BGESentenceSimilarity()
        for (sentence, expected) in zip(fixture.sentences, fixture.tokenIDs) {
            XCTAssertEqual(provider.tokenIDs(for: sentence), expected, sentence)
        }
    }

    /// 向量逐句对齐：不同 ONNX Runtime 版本的 int8 量化内核存在约 0.5% 的
    /// 数值差异（Swift 包 1.24 / Python 1.31 实测余弦 0.993~0.996），
    /// 因此这里校验「方向 + 相邻句余弦结构」，而不是逐元素精确相等。
    func testSentenceVectorsMatchPythonGolden() throws {
        let fixture = try GoldenFixture.load()
        let vectors = try XCTUnwrap(BGESentenceSimilarity().sentenceVectors(fixture.sentences))
        XCTAssertEqual(vectors.count, fixture.vectors.count)
        for (index, pair) in zip(vectors, fixture.vectors).enumerated() {
            XCTAssertEqual(pair.0.count, 512)
            XCTAssertGreaterThan(Self.cosine(pair.0, pair.1), 0.98, "sentence \(index)")
        }
        for i in 0..<vectors.count {
            for j in (i + 1)..<vectors.count {
                let actual = Self.cosine(vectors[i], vectors[j])
                let expected = Self.cosine(fixture.vectors[i], fixture.vectors[j])
                XCTAssertEqual(actual, expected, accuracy: 0.02,
                               "cosine(\(i), \(j))")
            }
        }
    }

    func testChineseRelatedSentencesScoreHigherThanUnrelated() throws {
        let sentences = ["我喜欢吃蛋糕。", "甜点是我最爱的食物。", "首相在议会发表演讲。"]
        let vectors = try XCTUnwrap(BGESentenceSimilarity().sentenceVectors(sentences))
        let related = Self.cosine(vectors[0], vectors[1])
        let unrelated = Self.cosine(vectors[0], vectors[2])
        XCTAssertGreaterThan(related, unrelated + 0.2)
    }

    func testAdjacentCosinesReturnsCountAndFiniteValues() throws {
        let fixture = try GoldenFixture.load()
        let values = try XCTUnwrap(BGESentenceSimilarity().adjacentCosines(fixture.sentences))
        XCTAssertEqual(values.count, fixture.sentences.count - 1)
        for value in values {
            XCTAssertTrue(value.isFinite)
            XCTAssertGreaterThanOrEqual(value, -1.0001)
            XCTAssertLessThanOrEqual(value, 1.0001)
        }
    }

    func testEmptySentenceReturnsNil() {
        XCTAssertNil(BGESentenceSimilarity().adjacentCosines([""]))
        XCTAssertNil(BGESentenceSimilarity().adjacentCosines(["我喜欢吃蛋糕。", ""]))
    }

    /// 纯 CPU 批量推理的宽松吞吐门槛：防止误挂 CoreML/线程数退化为串行。
    func testSentenceVectorsRunOnCPUWithinBudget() throws {
        let sentences = (0..<200).map {
            "第\($0)段测试文本，用于评估语义切片模型在纯 CPU 上的吞吐表现。"
        }
        let start = Date()
        let vectors = BGESentenceSimilarity().sentenceVectors(sentences)
        let elapsed = Date().timeIntervalSince(start)
        XCTAssertEqual(vectors?.count, sentences.count)
        XCTAssertLessThan(elapsed, 5.0, "200 句耗时 \(elapsed) s")
    }

    private static func cosine(_ a: [Float], _ b: [Float]) -> Double {
        var dot = 0.0
        for index in a.indices {
            dot += Double(a[index]) * Double(b[index])
        }
        return dot
    }
}
