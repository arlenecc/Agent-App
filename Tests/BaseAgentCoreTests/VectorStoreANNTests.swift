import XCTest
@testable import BaseAgentCore

/// ANN（分块 IVF）索引测试：全部用例都在临时库上跑，不依赖真实 2100 万切片库。
final class VectorStoreANNTests: XCTestCase {
    private var tempDir: URL!

    override func setUpWithError() throws {
        tempDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ann-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tempDir)
    }

    // MARK: - 夹具

    private let dimension = 16
    /// 簇中心：dim=16 上互不相似的单位向量方向（cos=0）。
    private let clusterCount = 6

    private func makeStore() throws -> VectorStore {
        try VectorStore(dbPath: tempDir.appendingPathComponent("index.sqlite").path,
                        dimension: dimension)
    }

    private func center(_ index: Int) -> [Float] {
        var vector = [Float](repeating: 0, count: dimension)
        vector[index % dimension] = 1
        return VectorCodec.normalize(vector)
    }

    private func vector(cluster: Int, offset: Int) -> [Float] {
        var vector = center(cluster)
        // 轻微扰动：同簇向量彼此相似但不完全相同。
        let noise = Float(offset % 7) * 0.01
        vector[(cluster + 3) % dimension] += noise
        vector[(cluster + 5) % dimension] -= noise / 2
        return VectorCodec.normalize(vector)
    }

    private func source(cluster: Int) -> String { "/kb/cluster-\(cluster).txt" }

    /// 写入若干簇的切片，返回 rowid → 簇的映射。
    @discardableResult
    private func seed(_ store: VectorStore, perCluster: Int,
                      clusters: Range<Int>? = nil) throws -> [Int64: Int] {
        let range = clusters ?? 0..<clusterCount
        var rowids: [Int64: Int] = [:]
        for cluster in range {
            var chunks: [Chunk] = []
            var vectors: [[Float]] = []
            for offset in 0..<perCluster {
                chunks.append(Chunk(text: "cluster \(cluster) item \(offset)",
                                    source: source(cluster: cluster), chunkIndex: offset))
                vectors.append(vector(cluster: cluster, offset: offset))
            }
            try store.add(chunks: chunks, vectors: vectors)
        }
        // 回读 rowid（add 后按来源查询）。
        for cluster in range {
            let ids = try store.rowIDs(source: source(cluster: cluster))
            for id in ids { rowids[id] = cluster }
        }
        return rowids
    }

    private func cluster(of rowID: Int64, in map: [Int64: Int]) -> Int? { map[rowID] }

    // MARK: - 用例

    func testBuildProducesReadyStatus() throws {
        let store = try makeStore()
        try seed(store, perCluster: 40)
        XCTAssertFalse(store.annIndexStatus().isReady)

        let result = try store.buildANNIndex(clusters: 4, clusterSize: 32, iterations: 8)

        XCTAssertFalse(result.cancelled)
        XCTAssertEqual(result.clusters, 4)
        let status = store.annIndexStatus()
        XCTAssertTrue(status.isReady)
        XCTAssertEqual(status.clusters, 4)
        XCTAssertEqual(status.blockRows, 128)
        XCTAssertEqual(status.blocks, 2)
        XCTAssertEqual(status.indexedRows, 240)
        XCTAssertEqual(status.tailRows, 0)
        XCTAssertEqual(status.coverage, 1.0, accuracy: 0.0001)
    }

    /// 探测全部簇时，ANN 结果应与暴力检索落在同一批簇里。
    func testProbingAllClustersMatchesBruteForce() throws {
        let store = try makeStore()
        let map = try seed(store, perCluster: 40)
        _ = try store.buildANNIndex(clusters: 4, clusterSize: 32, iterations: 8)

        let query = vector(cluster: 2, offset: 0)
        let brute = try store.search(vector: query, topK: 5)
        let ann = try XCTUnwrap(store.annSearch(vector: query, topK: 5, nprobe: 4))

        XCTAssertEqual(brute.count, 5)
        XCTAssertEqual(ann.count, 5)
        XCTAssertEqual(Set(ann.map { cluster(of: $0.rowID, in: map) ?? -1 }), [2],
                       "ANN 召回应全部来自查询所在的簇")
        // 分数口径一致（int8 量化误差可忽略）。
        XCTAssertEqual(ann[0].score, brute[0].score, accuracy: 0.02)
    }

    /// nprobe=1 时也应命中查询所在的簇（簇之间正交，最近簇唯一）。
    func testSingleProbeSelectsNearestCluster() throws {
        let store = try makeStore()
        let map = try seed(store, perCluster: 40)
        _ = try store.buildANNIndex(clusters: 4, clusterSize: 32, iterations: 8)

        let query = vector(cluster: 4, offset: 3)
        let ann = try XCTUnwrap(store.annSearch(vector: query, topK: 3, nprobe: 1))
        XCTAssertFalse(ann.isEmpty)
        XCTAssertEqual(Set(ann.map { cluster(of: $0.rowID, in: map) ?? -1 }), [4])
    }

    /// 索引不可用时返回 nil，检索自动回退暴力路径。
    func testSearchFallsBackWhenIndexMissing() throws {
        let store = try makeStore()
        try seed(store, perCluster: 20)

        XCTAssertNil(try store.annSearch(vector: vector(cluster: 1, offset: 0), topK: 3, nprobe: 4))

        let query = vector(cluster: 1, offset: 0)
        let brute = try store.search(vector: query, topK: 3)
        let withANN = try store.search(vector: query, topK: 3,
                                      ann: ANNQueryOptions(nprobe: 4))
        XCTAssertEqual(withANN.map(\.rowID), brute.map(\.rowID))
    }

    /// 索引就绪后新增切片进尾块，检索能看到它们。
    /// 用新来源（cluster 6）：同一来源再次写入是「先删后插」，旧切片会被替换。
    func testNewChunksAreIndexedIntoTail() throws {
        let store = try makeStore()
        let map = try seed(store, perCluster: 40)
        _ = try store.buildANNIndex(clusters: 4, clusterSize: 32, iterations: 8)

        let newMap = try seed(store, perCluster: 12, clusters: 6..<7)
        let status = store.annIndexStatus()
        XCTAssertEqual(status.tailRows, 12)
        XCTAssertEqual(status.indexedRows, 240)

        let query = vector(cluster: 6, offset: 1)
        let ann = try XCTUnwrap(store.annSearch(vector: query, topK: 3, nprobe: 4))
        let clusters = Set(ann.map { newMap[$0.rowID] ?? map[$0.rowID] ?? -1 })
        XCTAssertEqual(clusters, [6])
    }

    /// 同一来源重新同步（行数变少）后，新切片仍应被尾块补齐检索到。
    /// 这类「先删后插」会让计数器口径的尾块大小为 0，必须按 rowid 水位判断。
    func testResyncedSourceIsStillSearchable() throws {
        let store = try makeStore()
        try seed(store, perCluster: 40)
        _ = try store.buildANNIndex(clusters: 4, clusterSize: 32, iterations: 8)

        // cluster 5 已有 40 行，重新同步为 12 行（旧行删除、新 rowid 追加）。
        let newMap = try seed(store, perCluster: 12, clusters: 5..<6)
        let status = store.annIndexStatus()
        XCTAssertEqual(status.tailRows, 12)
        XCTAssertEqual(status.isUsable, true)

        let ann = try XCTUnwrap(store.annSearch(vector: vector(cluster: 5, offset: 1),
                                                topK: 3, nprobe: 4))
        XCTAssertEqual(Set(ann.map { newMap[$0.rowID] ?? -1 }), [5])
    }

    /// 取消后已完成的块保留，续跑能补齐并达到 ready。
    func testBuildIsCancellableAndResumable() throws {
        let store = try makeStore()
        try seed(store, perCluster: 40)
        let total = 240

        var scannedBlocks = 0
        let partial = try store.buildANNIndex(clusters: 4, clusterSize: 32, iterations: 8,
                                              progress: { _ in },
                                              isCancelled: {
            scannedBlocks += 1
            return scannedBlocks > 1
        })
        XCTAssertTrue(partial.cancelled)
        let partialStatus = store.annIndexStatus()
        XCTAssertEqual(partialStatus.blocks, 1)
        XCTAssertEqual(partialStatus.indexedRows, 128)
        XCTAssertFalse(partialStatus.isReady)
        XCTAssertEqual(partialStatus.coverage, Double(128) / Double(total), accuracy: 0.0001)

        let resumed = try store.buildANNIndex(clusters: 4, clusterSize: 32, iterations: 8)
        XCTAssertFalse(resumed.cancelled)
        let status = store.annIndexStatus()
        XCTAssertEqual(status.blocks, 2)
        XCTAssertEqual(status.indexedRows, total)
        XCTAssertTrue(status.isReady)
    }

    /// 清空 / 显式删除都应移除索引。
    func testClearAndDropRemoveIndex() throws {
        let store = try makeStore()
        try seed(store, perCluster: 20)
        _ = try store.buildANNIndex(clusters: 4, clusterSize: 16, iterations: 4)
        XCTAssertTrue(store.annIndexStatus().isReady)

        try store.dropANNIndex()
        XCTAssertFalse(store.annIndexStatus().isReady)
        XCTAssertNil(try store.annSearch(vector: vector(cluster: 0, offset: 0), topK: 3, nprobe: 4))

        _ = try store.buildANNIndex(clusters: 4, clusterSize: 16, iterations: 4)
        try store.clear()
        let status = store.annIndexStatus()
        XCTAssertFalse(status.isReady)
        XCTAssertEqual(status.indexedRows, 0)
    }

    /// 删除切片后索引里的陈旧 rowid 不应污染检索结果。
    func testStaleRowIDsAreFilteredAfterDelete() throws {
        let store = try makeStore()
        try seed(store, perCluster: 30)
        _ = try store.buildANNIndex(clusters: 4, clusterSize: 32, iterations: 8)

        try store.deleteBySource(source(cluster: 1))
        let hits = try store.search(vector: vector(cluster: 1, offset: 0), topK: 5,
                                    ann: ANNQueryOptions(nprobe: 4))
        XCTAssertFalse(hits.isEmpty)
        XCTAssertFalse(hits.contains { $0.source == source(cluster: 1) },
                       "已删除来源不应出现在结果里")
    }
}