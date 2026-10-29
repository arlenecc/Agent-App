import XCTest
@testable import BaseAgentCore

private struct PotionGoldenFixture: Decodable {
    let sentences: [String]
    let tokenIDs: [[Int]]
    let vectors: [[Float]]
    let dimension: Int

    enum CodingKeys: String, CodingKey {
        case sentences
        case tokenIDs = "token_ids"
        case vectors
        case dimension
    }

    static func load() throws -> PotionGoldenFixture {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/potion-multilingual-golden.json")
        return try JSONDecoder().decode(PotionGoldenFixture.self, from: Data(contentsOf: url))
    }
}

final class PotionSentenceSimilarityTests: XCTestCase {
    func testBundledModelResourceIsPresent() throws {
        let directory = try XCTUnwrap(PotionSentenceSimilarity.modelDirectory())
        let fm = FileManager.default
        for name in ["model.onnx", "tokenizer.json", "config.json",
                     "LICENSE", "MODEL_INFO.md"] {
            XCTAssertTrue(fm.fileExists(atPath: directory.appendingPathComponent(name).path), name)
        }
        let size = try fm.attributesOfItem(
            atPath: directory.appendingPathComponent("model.onnx").path
        )[.size] as? Int64
        XCTAssertGreaterThan(size ?? 0, 70_000_000, "int4 权重约 80 MB")
    }

    func testTokenizerMatchesHuggingFaceGoldenIDs() throws {
        let fixture = try PotionGoldenFixture.load()
        let provider = PotionSentenceSimilarity()
        for (sentence, expected) in zip(fixture.sentences, fixture.tokenIDs) {
            XCTAssertEqual(provider.tokenIDs(for: sentence), expected,
                           "token ids 应与 HF tokenizer.json 一致：\(sentence.prefix(20))")
        }
    }

    /// potion 是静态词向量模型：输出已 mean-pool + L2 归一化。
    /// 与 Python onnxruntime 逐句对齐，校验方向与相邻句余弦结构。
    func testSentenceVectorsMatchPythonGolden() throws {
        let fixture = try PotionGoldenFixture.load()
        let vectors = try XCTUnwrap(
            PotionSentenceSimilarity().sentenceVectors(fixture.sentences))
        XCTAssertEqual(vectors.count, fixture.vectors.count)
        for (index, pair) in zip(vectors, fixture.vectors).enumerated() {
            XCTAssertEqual(pair.0.count, fixture.dimension)
            XCTAssertEqual(Self.norm(pair.0), 1, accuracy: 1e-4, "sentence \(index)")
            XCTAssertGreaterThan(Self.cosine(pair.0, pair.1), 0.99,
                                 "sentence \(index) 应与 Python golden 同向")
        }
        for i in 0..<vectors.count {
            for j in (i + 1)..<vectors.count {
                let actual = Self.cosine(vectors[i], vectors[j])
                let expected = Self.cosine(fixture.vectors[i], fixture.vectors[j])
                XCTAssertEqual(actual, expected, accuracy: 0.01,
                               "cosine(\(i), \(j))")
            }
        }
    }

    /// 超过 512 token 的长句必须完整编码：potion 没有位置编码，
    /// 不能像 bge-small-zh 那样在 512 token 处截断。
    func testLongSentenceBeyond512TokensIsNotTruncated() throws {
        let fixture = try PotionGoldenFixture.load()
        let longIndex = fixture.sentences.count - 1
        XCTAssertGreaterThan(fixture.tokenIDs[longIndex].count, 512,
                             "夹具应覆盖超过 512 token 的长句")
        let ids = try XCTUnwrap(
            PotionSentenceSimilarity().tokenIDs(for: fixture.sentences[longIndex]))
        XCTAssertEqual(ids.count, fixture.tokenIDs[longIndex].count,
                       "长句不能被截断到 512 token")
        let vector = try XCTUnwrap(
            PotionSentenceSimilarity().sentenceVectors([fixture.sentences[longIndex]])?.first)
        XCTAssertEqual(vector.count, fixture.dimension)
        XCTAssertEqual(Self.norm(vector), 1, accuracy: 1e-4)
        XCTAssertGreaterThan(Self.cosine(vector, fixture.vectors[longIndex]), 0.99)
    }

    func testChineseRelatedSentencesScoreHigherThanUnrelated() throws {
        let sentences = ["我喜欢吃蛋糕。", "甜点是我最爱的食物。", "首相在议会发表演讲。"]
        let vectors = try XCTUnwrap(PotionSentenceSimilarity().sentenceVectors(sentences))
        let related = Self.cosine(vectors[0], vectors[1])
        let unrelated = Self.cosine(vectors[0], vectors[2])
        XCTAssertGreaterThan(related, unrelated + 0.2)
    }

    func testAdjacentCosinesReturnsCountAndFiniteValues() throws {
        let fixture = try PotionGoldenFixture.load()
        let values = try XCTUnwrap(
            PotionSentenceSimilarity().adjacentCosines(fixture.sentences))
        XCTAssertEqual(values.count, fixture.sentences.count - 1)
        for value in values {
            XCTAssertTrue(value.isFinite)
            XCTAssertGreaterThanOrEqual(value, -1.0001)
            XCTAssertLessThanOrEqual(value, 1.0001)
        }
    }

    func testEmptySentenceReturnsNil() {
        XCTAssertNil(PotionSentenceSimilarity().adjacentCosines([""]))
        XCTAssertNil(PotionSentenceSimilarity().adjacentCosines(["我喜欢吃蛋糕。", ""]))
    }

    /// 纯 CPU 批量推理的宽松吞吐门槛：防止误挂 CoreML/线程数退化为串行。
    func testSentenceVectorsRunOnCPUWithinBudget() throws {
        let sentences = (0..<200).map {
            "第\($0)段测试文本，用于评估语义切片模型在纯 CPU 上的吞吐表现。"
        }
        let start = Date()
        let vectors = PotionSentenceSimilarity().sentenceVectors(sentences)
        let elapsed = Date().timeIntervalSince(start)
        XCTAssertEqual(vectors?.count, sentences.count)
        XCTAssertLessThan(elapsed, 5.0, "200 句耗时 \(elapsed) s")
    }

    /// 大批次长句不能把内存撑爆：按 token 总量切子批后结果仍逐句正确。
    func testLargeBatchWithMixedLengthsKeepsPerSentenceVectors() throws {
        let short = "苹果是一种常见的水果。"
        let long = String(repeating: "多肉植物喜欢充足的散射光。", count: 200)
        let sentences = [short, long, short, long, short, long]
        guard let vectors = PotionSentenceSimilarity().sentenceVectors(sentences) else {
            return XCTFail("批量推理失败")
        }
        XCTAssertEqual(vectors.count, sentences.count)
        // 相同文本的向量必须一致，且短句与长句互不串位。
        XCTAssertGreaterThan(Self.cosine(vectors[0], vectors[2]), 0.9999)
        XCTAssertGreaterThan(Self.cosine(vectors[1], vectors[3]), 0.9999)
        XCTAssertGreaterThan(Self.cosine(vectors[3], vectors[5]), 0.9999)
        XCTAssertLessThan(Self.cosine(vectors[0], vectors[1]), 0.5)
    }

    private static func cosine(_ a: [Float], _ b: [Float]) -> Double {
        var dot = 0.0
        for index in a.indices {
            dot += Double(a[index]) * Double(b[index])
        }
        return dot
    }

    private static func norm(_ vector: [Float]) -> Double {
        var sum = 0.0
        for value in vector { sum += Double(value) * Double(value) }
        return sqrt(sum)
    }
}
