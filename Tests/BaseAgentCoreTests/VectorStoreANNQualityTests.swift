import XCTest
@testable import BaseAgentCore

/// ANN 离线质量评测：同一批查询对比分块 IVF 与精确检索的 recall@20 / nDCG@20。
/// 合成语料使用重叠的簇（真实簇数多于索引簇数），让 nprobe 的取舍有实际差异。
final class VectorStoreANNQualityTests: XCTestCase {
    private var tempDir: URL!

    override func setUpWithError() throws {
        tempDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ann-quality-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tempDir)
    }

    // MARK: - 夹具

    private let dimension = 32
    private let trueClusters = 12
    private let rowsPerCluster = 40
    private let queriesPerCluster = 2

    /// 固定种子的伪随机数：评测结果可复现。
    private struct LCG {
        var state: UInt64

        mutating func next() -> Float {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float((state >> 40) & 0xFFFFFF) / Float(0x7F_FFFF) * 2 - 1
        }
    }

    private func noise(_ rng: inout LCG) -> [Float] {
        (0..<dimension).map { _ in rng.next() }
    }

    private func blend(_ base: [Float], _ noise: [Float], weight: Float) -> [Float] {
        VectorCodec.normalize(zip(base, noise).map { $0 + $1 * weight })
    }

    private func makeStore() throws -> VectorStore {
        try VectorStore(dbPath: tempDir.appendingPathComponent("index.sqlite").path,
                        dimension: dimension)
    }

    func testANNQualityAgainstExactSearch() throws {
        var rng = LCG(state: 42)
        let centers = (0..<trueClusters).map { _ in
            VectorCodec.normalize(noise(&rng))
        }
        let store = try makeStore()
        var queries: [[Float]] = []
        for cluster in 0..<trueClusters {
            var chunks: [Chunk] = []
            var vectors: [[Float]] = []
            for index in 0..<rowsPerCluster {
                chunks.append(Chunk(text: "cluster \(cluster) row \(index)",
                                    source: "/kb/cluster-\(cluster).txt",
                                    chunkIndex: index))
                vectors.append(blend(centers[cluster], noise(&rng), weight: 0.55))
            }
            try store.add(chunks: chunks, vectors: vectors)
            for _ in 0..<queriesPerCluster {
                queries.append(blend(centers[cluster], noise(&rng), weight: 0.45))
            }
        }
        let build = try store.buildANNIndex(clusters: 8, clusterSize: 128, iterations: 10)
        XCTAssertFalse(build.cancelled)
        XCTAssertTrue(store.annIndexStatus().isReady)

        var reports: [ANNQualityReport] = []
        for nprobe in [1, 2, 3, 4] {
            for rescore in [0, 64, 128] {
                let report = try XCTUnwrap(store.evaluateANNQuality(
                    queries: queries, topK: 20, nprobe: nprobe, rescore: rescore))
                reports.append(report)
                print("ANN 质量评测 \(report.summary)")
            }
        }
        XCTAssertEqual(reports.count, 12)
        XCTAssertTrue(reports.allSatisfy { $0.queries == queries.count })
        XCTAssertTrue(reports.allSatisfy { $0.emptyQueries == 0 })

        func report(_ nprobe: Int, _ rescore: Int) -> ANNQualityReport {
            reports.first { $0.nprobe == nprobe && $0.rescore == rescore }!
        }
        // nprobe 增加不应降低召回。该语料刻意让 12 个真实簇被 8 个索引簇
        // 覆盖，nprobe=4 才能覆盖查询附近的大部分精确结果。
        XCTAssertGreaterThanOrEqual(report(2, 64).recall, report(1, 64).recall - 0.001)
        XCTAssertGreaterThanOrEqual(report(4, 64).recall, report(2, 64).recall - 0.001)
        XCTAssertGreaterThanOrEqual(report(4, 64).recall, 0.7)
        XCTAssertGreaterThanOrEqual(report(4, 64).ndcg, 0.8)
        // 精确重打分不应降低召回；int8 量化在归一化向量上排序误差很小。
        XCTAssertGreaterThanOrEqual(report(2, 64).recall, report(2, 0).recall - 0.001)
        XCTAssertGreaterThanOrEqual(report(4, 128).recall, report(4, 64).recall - 0.001)
    }

    func testQualityEvaluationRequiresBuiltIndex() throws {
        let store = try makeStore()
        var rowRng = LCG(state: 7)
        let chunk = Chunk(text: "only", source: "/kb/only.txt", chunkIndex: 0)
        try store.add(chunks: [chunk],
                      vectors: [VectorCodec.normalize(noise(&rowRng))])
        XCTAssertNil(try store.evaluateANNQuality(queries: [], topK: 20))
        var queryRng = LCG(state: 9)
        let built = try store.evaluateANNQuality(
            queries: [VectorCodec.normalize(noise(&queryRng))], topK: 5)
        XCTAssertNil(built, "未构建索引时应返回 nil，由调用方回退精确检索")
    }
}
