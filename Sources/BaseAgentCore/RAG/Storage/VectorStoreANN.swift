import Foundation
import Accelerate

// MARK: - 公开类型

/// ANN 检索参数。
public struct ANNQueryOptions: Sendable {
    /// 每块探测的簇数。越大召回越高、读放大越多（1 = 最快）。
    public var nprobe: Int
    /// 用点查精确向量重打分的候选数（0 = 关闭）。开启后最终排序用真实
    /// 余弦分，消除 int8 量化带来的排序误差。
    public var rescore: Int

    public init(nprobe: Int = 1, rescore: Int = 0) {
        self.nprobe = max(1, nprobe)
        self.rescore = max(0, rescore)
    }
}

/// ANN 构建进度。
public struct ANNBuildProgress: Sendable {
    public var stage: String
    public var blocks: Int
    public var totalBlocks: Int
    public var rows: Int
    public var totalRows: Int
    public var message: String

    public init(stage: String, blocks: Int = 0, totalBlocks: Int = 0,
                rows: Int = 0, totalRows: Int = 0, message: String = "") {
        self.stage = stage
        self.blocks = blocks
        self.totalBlocks = totalBlocks
        self.rows = rows
        self.totalRows = totalRows
        self.message = message
    }

    public var fraction: Double {
        guard totalRows > 0 else { return 0 }
        return min(1, max(0, Double(rows) / Double(totalRows)))
    }
}

/// ANN 构建结果。
public struct ANNBuildResult: Sendable {
    public var clusters: Int = 0
    public var blockRows: Int = 0
    public var blocks: Int = 0
    /// 已写入块的切片数（尾块不计）。
    public var indexedRows: Int = 0
    /// 索引之外的新切片（检索时由精确扫描补齐）。
    public var tailRows: Int = 0
    public var cancelled: Bool = false
    public var elapsed: Double = 0
    public var summary: String = ""

    public init() {}
}

/// ANN 索引状态。
public struct ANNIndexStatus: Sendable {
    public var clusters: Int = 0
    public var blockRows: Int = 0
    public var blocks: Int = 0
    /// 已写入块的切片数。
    public var indexedRows: Int = 0
    /// 索引之外的新切片数（检索时精确补齐）。
    public var tailRows: Int = 0
    /// 尾块里超出精确补齐上限的部分：这些切片检索不到，需要重建索引。
    public var unindexedRows: Int = 0
    public var dimension: Int = 0
    public var builtAt: Double = 0
    /// 整库都在块内（无尾块）才算完全就绪。
    public var isReady: Bool = false
    /// 当前能否用索引检索（尾块仍在上限内）。
    public var isUsable: Bool = false
    public var needsRebuild: Bool = false
    public var coverage: Double = 0
    public var summary: String = ""

    public init() {}
}

// MARK: - 内部结构

/// ANN 元数据（存 meta 表，键前缀 `ann_`）。
struct ANNStoredMeta {
    var version = 0
    var clusters = 0
    var blockRows = 0
    var dim = 0
    var blocks = 0
    var rows = 0
    var lastRowID: Int64 = 0
    var builtAt: Double = 0
}

/// 构建时按 rowid 顺序攒下的一块切片（int8 码 + 每行尺度 + rowid）。
struct ANNBlockBuffer {
    private(set) var rowids: [Int64] = []
    private(set) var codes: [Int8] = []
    private(set) var scales: [Float] = []
    private(set) var lastRowID: Int64 = 0

    var count: Int { rowids.count }

    mutating func reset(maxRows: Int, dim: Int) {
        rowids.removeAll(keepingCapacity: true)
        codes.removeAll(keepingCapacity: true)
        scales.removeAll(keepingCapacity: true)
        rowids.reserveCapacity(maxRows)
        scales.reserveCapacity(maxRows)
        codes.reserveCapacity(maxRows * dim)
        lastRowID = 0
    }

    /// 量化一行向量：`scale = max|v|`，码值 = round(v × 127 / scale) ∈ [-127,127]。
    /// 打分时按 `dot(q, codes) × scale / 127` 还原（对归一化向量即余弦）。
    mutating func append(rowID: Int64, floats: UnsafePointer<Float>, dim: Int,
                         scratch: inout [Int8]) {
        var maxAbs: Float = 0
        for i in 0..<dim {
            let value = abs(floats[i])
            if value > maxAbs { maxAbs = value }
        }
        let scale: Float = maxAbs > 0 ? maxAbs : 1
        let factor = 127 / scale
        scratch.withUnsafeMutableBufferPointer { destination in
            for i in 0..<dim {
                let value = (floats[i] * factor).rounded()
                destination[i] = Int8(max(-127, min(127, value)))
            }
        }
        rowids.append(rowID)
        scales.append(scale)
        codes.append(contentsOf: scratch)
        lastRowID = rowID
    }
}

/// int8 量化后的簇心集合（按块、按簇行优先）。
final class ANNCentroidCache {
    let blocks: Int
    let clusters: Int
    let dim: Int
    /// blocks × clusters × dim。
    private let codes: [Int8]
    /// blocks × clusters。
    private let scales: [Float]

    init(blocks: Int, clusters: Int, dim: Int, codes: [Int8], scales: [Float]) {
        self.blocks = blocks
        self.clusters = clusters
        self.dim = dim
        self.codes = codes
        self.scales = scales
    }

    /// 某块里与查询最接近的 nprobe 个簇（降序）。
    func nearest(query: UnsafePointer<Float>, block: Int, nprobe: Int,
                 scratch: inout [Float]) -> [Int] {
        guard clusters > 0, dim > 0, nprobe > 0 else { return [] }
        let base = block * clusters * dim
        guard base >= 0, base + clusters * dim <= codes.count else { return [] }
        if scratch.count < clusters * dim {
            scratch = [Float](repeating: 0, count: clusters * dim)
        }
        codes.withUnsafeBufferPointer { source in
            guard let sourceBase = source.baseAddress else { return }
            vDSP_vflt8(sourceBase + base, 1, &scratch, 1, vDSP_Length(clusters * dim))
        }
        var probe: [Int] = []
        var probeScores: [Float] = []
        probe.reserveCapacity(nprobe)
        probeScores.reserveCapacity(nprobe)
        scratch.withUnsafeBufferPointer { floats in
            guard let base = floats.baseAddress else { return }
            for cluster in 0..<clusters {
                var dot: Float = 0
                vDSP_dotpr(base + cluster * dim, 1, query, 1, &dot, vDSP_Length(dim))
                let score = dot * scales[block * clusters + cluster] / 127
                if probe.count == nprobe, score <= probeScores[probe.count - 1] { continue }
                var slot = probe.count
                while slot > 0, probeScores[slot - 1] < score { slot -= 1 }
                if probe.count == nprobe {
                    probe.removeLast()
                    probeScores.removeLast()
                }
                probe.insert(cluster, at: slot)
                probeScores.insert(score, at: slot)
            }
        }
        return probe
    }
}

/// 容量受限的 top-K 收集器（小顶堆，O(log k) 更新）。
struct ANNTopKCollector {
    let capacity: Int
    private var rowids: [Int64] = []
    private var scores: [Double] = []

    init(capacity: Int) {
        self.capacity = max(1, capacity)
        rowids.reserveCapacity(self.capacity)
        scores.reserveCapacity(self.capacity)
    }

    var count: Int { rowids.count }

    mutating func offer(rowid: Int64, score: Double) {
        if rowids.count < capacity {
            rowids.append(rowid)
            scores.append(score)
            var index = rowids.count - 1
            while index > 0 {
                let parent = (index - 1) / 2
                if scores[parent] <= scores[index] { break }
                scores.swapAt(parent, index)
                rowids.swapAt(parent, index)
                index = parent
            }
            return
        }
        guard score > scores[0] else { return }
        scores[0] = score
        rowids[0] = rowid
        var index = 0
        while true {
            let left = index * 2 + 1
            let right = left + 1
            var smallest = index
            if left < rowids.count, scores[left] < scores[smallest] { smallest = left }
            if right < rowids.count, scores[right] < scores[smallest] { smallest = right }
            if smallest == index { break }
            scores.swapAt(index, smallest)
            rowids.swapAt(index, smallest)
            index = smallest
        }
    }

    /// 按分数降序返回全部候选。
    func sortedItems() -> [(Int64, Double)] {
        var items = Array(zip(rowids, scores))
        items.sort { $0.1 > $1.1 }
        return items
    }
}

/// 影子表快速通道不可用（回退官方读取路径，尚未投递任何数据）。
private struct ANNShadowUnavailable: Error {
    let reason: String
}

// MARK: - VectorStore 的 ANN 扩展

/// 分块 IVF 向量索引。
///
/// **为什么自建**：sqlite-vec 0.1.9 没有 ANN——`WHERE embedding MATCH ?` 是
/// 暴力扫描，实测外置 USB SSD 上 200k 行 40s（≈10MB/s），2100 万切片要
/// 70 分钟；只读查询有 5s 超时，等于大库上检索永远被跳过。
///
/// **布局**（与索引库同文件，便于随库迁移/删除）：
/// - 每块 `clusters × clusterSize` 行，按 rowid 顺序切分；块内 k-means 训练
///   簇心，向量与簇心都量化成 int8（`scale` + 码，512 维每行 516B）；
/// - `ann_cells(block_id, cluster_id, seq, count, payload)`：一个簇一段连续
///   行，按簇顺序写入 → 检索只读命中的簇，磁盘上是顺序读（外置 USB SSD
///   实测顺序 115MB/s，随机点查只有 87 次/s，所以必须聚簇落盘）；
/// - `ann_blocks(block_id, clusters, dim, payload)`：该块簇心（同样 int8）；
/// - 索引之外的新切片（尾块）用顺序扫描精确补齐：同步完立刻可检索，
///   超过读取预算时由调用方触发重建。
extension VectorStore {

    static let annFormatVersion = 1
    /// 尾块精确补齐的读取预算：超出则不再用索引检索（回退精确路径并要求重建）。
    static let annTailScanByteBudget = 128 * 1024 * 1024
    static let annCellsTable = "ann_cells"
    static let annBlocksTable = "ann_blocks"
    private var annMetaPrefix: String { "ann_" }

    // MARK: 状态

    public func annIndexStatus() -> ANNIndexStatus {
        lock.lock(); defer { lock.unlock() }
        return annIndexStatusLocked()
    }

    func annIndexStatusLocked() -> ANNIndexStatus {
        var status = ANNIndexStatus()
        guard let meta = try? annStoredMetaLocked() else { return status }
        if meta.blocks == 0 {
            status.summary = "ANN 索引未构建"
            return status
        }
        let total = count
        let tail = annTailRowCountLocked(after: meta.lastRowID, dim: meta.dim, total: total,
                                         indexedRows: meta.rows)
        let cap = annTailScanRowCap(dim: meta.dim)
        let unindexed = max(0, tail - cap)
        let dimOK = meta.dim > 0 && meta.dim == dimension
        status.clusters = meta.clusters
        status.blockRows = meta.blockRows
        status.blocks = meta.blocks
        status.indexedRows = meta.rows
        status.tailRows = tail
        status.unindexedRows = unindexed
        status.dimension = meta.dim
        status.builtAt = meta.builtAt
        status.isReady = meta.blocks > 0 && dimOK && tail == 0
        status.isUsable = meta.blocks > 0 && dimOK && unindexed == 0
        status.coverage = total > 0 ? min(1, Double(meta.rows) / Double(total)) : 0
        status.needsRebuild = meta.blocks > 0 && tail > annRebuildTailThreshold(dim: meta.dim)
        status.summary = annSummaryLocked(status)
        return status
    }

    /// 尾块精确补齐的行数上限（按读取字节预算折算）。
    func annTailScanRowCap(dim: Int) -> Int {
        guard dim > 0 else { return 0 }
        return max(1024, Self.annTailScanByteBudget / (dim * 4))
    }

    /// 尾块大小：rowid 水位之后的切片数。
    ///
    /// 不能用 `count - indexedRows` 单独判断——同步是「按来源先删后插」，
    /// 某个来源重新同步且行数变少时，计数器口径会算出 0 甚至负数，
    /// 而水位之后其实还有新切片。两个口径取大，并且统计封顶在 `cap + 1`
    /// 行（只为判断是否超出补齐预算，不需要精确值；大库上也不必扫全表）。
    func annTailRowCountLocked(after rowID: Int64, dim: Int, total: Int,
                               indexedRows: Int) -> Int {
        let byCount = max(0, total - indexedRows)
        let cap = annTailScanRowCap(dim: dim)
        let counted = (try? db.scalar("""
        SELECT COUNT(*) FROM (
            SELECT 1 FROM \(Self.chunksTable) WHERE rowid > ? LIMIT ?
        )
        """, [.int(rowID), .int(Int64(cap + 1))]).intValue).flatMap { Int($0) } ?? 0
        return max(byCount, counted)
    }

    /// 建议重建的尾块规模：超过四分之一预算时提示（读取已明显变慢）。
    func annRebuildTailThreshold(dim: Int) -> Int {
        max(8_192, annTailScanRowCap(dim: dim) / 4)
    }

    private func annSummaryLocked(_ status: ANNIndexStatus) -> String {
        guard status.blocks > 0, status.dimension > 0 else { return "ANN 索引未构建" }
        if status.dimension != dimension {
            return "ANN 索引维度 \(status.dimension) 与当前向量维度 \(dimension) 不符，需重建"
        }
        var parts = [
            "分块 IVF \(status.clusters) 簇/块 × \(status.blockRows / max(1, status.clusters)) 行/簇",
            "\(status.blocks) 块",
            "已索引 \(status.indexedRows) 切片",
        ]
        if status.tailRows > 0 { parts.append("尾块 \(status.tailRows) 切片（精确补齐）") }
        if status.unindexedRows > 0 { parts.append("尾块超出补齐上限，需重建索引") }
        parts.append(String(format: "覆盖 %.1f%%", status.coverage * 100))
        if status.needsRebuild { parts.append("建议重建") }
        return parts.joined(separator: "，")
    }

    // MARK: 构建

    /// 构建（或增量续跑）ANN 索引。
    ///
    /// 已建好的块直接复用（meta 记录块数、行数与 rowid 水位），因此取消/中断
    /// 后再次调用会从断点继续；簇数、块大小或维度变化时整体重建。
    /// 每块一个事务：中途杀进程只丢当前块。
    @discardableResult
    public func buildANNIndex(clusters: Int = 256,
                              clusterSize: Int = 512,
                              iterations: Int = 8,
                              progress: ((ANNBuildProgress) -> Void)? = nil,
                              isCancelled: (() -> Bool)? = nil) throws -> ANNBuildResult {
        lock.lock(); defer { lock.unlock() }
        let started = Date()
        let dim = dimension
        guard dim > 0 else { throw AgentError.storage("向量维度无效，无法构建 ANN 索引") }
        let clusterCount = min(max(2, clusters), 4096)
        let rowsPerCluster = min(max(1, clusterSize), 8192)
        let blockRows = clusterCount * rowsPerCluster

        try ensureANNSchemaLocked()
        var meta = try annStoredMetaLocked()
        if meta.version != Self.annFormatVersion || meta.dim != dim
            || meta.clusters != clusterCount || meta.blockRows != blockRows {
            if meta.version != 0 {
                AgentLog.info("ANN: 索引参数变化（\(meta.clusters) 簇 / \(meta.blockRows) 行/块 / dim \(meta.dim)），整体重建")
            }
            try resetANNIndexLocked()
            meta = ANNStoredMeta()
        }

        var result = ANNBuildResult()
        result.clusters = clusterCount
        result.blockRows = blockRows
        var blockID = meta.blocks
        var indexedRows = meta.rows
        var watermark = meta.lastRowID
        let total = count
        let remaining = max(0, total - indexedRows)
        let estimatedBlocks = max(1, (remaining + blockRows - 1) / blockRows)
        if indexedRows > 0 {
            AgentLog.info("ANN: 从断点续建（已有 \(blockID) 块 / \(indexedRows) 切片）")
        }
        progress?(ANNBuildProgress(
            stage: "start", blocks: blockID, totalBlocks: estimatedBlocks,
            rows: indexedRows, totalRows: max(total, indexedRows),
            message: "开始构建 ANN 索引（\(clusterCount) 簇/块 × \(rowsPerCluster) 行/簇）"))

        while true {
            var buffer = ANNBlockBuffer()
            try gatherANNBlockLocked(after: watermark, dim: dim, maxRows: blockRows, into: &buffer)
            if buffer.count == 0 { break }
            if let isCancelled, isCancelled() {
                result.cancelled = true
                break
            }
            try writeANNBlockLocked(blockID: blockID, dim: dim, clusters: clusterCount,
                                    blockRows: blockRows, iterations: iterations, buffer: buffer)
            blockID += 1
            indexedRows += buffer.count
            watermark = buffer.lastRowID
            progress?(ANNBuildProgress(
                stage: "blocks", blocks: blockID,
                totalBlocks: max(estimatedBlocks, blockID),
                rows: indexedRows, totalRows: max(total, indexedRows),
                message: "已写入 \(blockID) 块（\(indexedRows) 切片）"))
        }

        result.blocks = blockID
        result.indexedRows = indexedRows
        result.tailRows = max(0, count - indexedRows)
        result.elapsed = Date().timeIntervalSince(started)
        var summary = "ANN 索引：\(blockID) 块 / \(indexedRows) 切片，耗时 "
            + String(format: "%.1fs", result.elapsed)
        if result.tailRows > 0 { summary += "，尾块 \(result.tailRows) 切片" }
        if result.cancelled { summary += "（已取消，下次可续建）" }
        result.summary = summary
        AgentLog.info(summary)
        return result
    }

    /// 删除 ANN 索引（表一起删掉，释放占用；随时可重建）。
    public func dropANNIndex() throws {
        lock.lock(); defer { lock.unlock() }
        try db.transaction {
            try db.execute("DROP TABLE IF EXISTS \(Self.annCellsTable)")
            try db.execute("DROP TABLE IF EXISTS \(Self.annBlocksTable)")
            try db.execute("DELETE FROM meta WHERE key GLOB '\(annMetaPrefix)*'")
        }
        annCentroidCache = nil
    }

    /// 清空索引内容、保留表结构（清空知识库 / 维度变化时调用）。
    /// 表未建立时会抛错，因此「清空」场景用 `try?` 包裹。
    func resetANNIndexLocked() throws {
        try db.execute("DELETE FROM \(Self.annCellsTable)")
        try db.execute("DELETE FROM \(Self.annBlocksTable)")
        try db.execute("DELETE FROM meta WHERE key GLOB '\(annMetaPrefix)*'")
        annCentroidCache = nil
    }

    // MARK: 检索

    /// ANN 检索：返回 nil 表示索引不可用，调用方应回退精确检索。
    public func annSearch(vector: [Float], topK: Int, nprobe: Int = 1, rescore: Int = 0,
                          sourceFilter: String? = nil,
                          sourceContains: String? = nil) throws -> [SearchHit]? {
        guard topK > 0, vector.count == dimension else { return nil }
        let options = ANNQueryOptions(nprobe: nprobe, rescore: rescore)
        lock.lock(); defer { lock.unlock() }
        guard hasChunks else { return nil }
        return try annSearchLocked(vector: vector, topK: topK, options: options,
                                   sourceFilter: sourceFilter, sourceContains: sourceContains)
    }

    /// 调用方需已持有 `lock`。
    func annSearchLocked(vector: [Float], topK: Int, options: ANNQueryOptions,
                         sourceFilter: String?,
                         sourceContains: String?) throws -> [SearchHit]? {
        guard let meta = try? annStoredMetaLocked(), meta.blocks > 0,
              meta.dim == dimension, meta.dim == vector.count else { return nil }
        let cap = annTailScanRowCap(dim: meta.dim)
        let tailRows = annTailRowCountLocked(after: meta.lastRowID, dim: meta.dim,
                                             total: count, indexedRows: meta.rows)
        guard tailRows <= cap else {
            AgentLog.info("ANN: 尾块 \(tailRows) 切片超出补齐上限，本次走精确检索")
            return nil
        }
        do {
            let query = VectorCodec.normalize(vector)
            let targeted = sourceFilter != nil || sourceContains != nil
            // 候选池必须留余量：索引里的 rowid 可能已被删除（同步是「先删后插」），
            // hydrate 会丢弃这些失效行；池子太小会让有效结果被挤空。
            let capacity = targeted
                ? max(topK * 20, 512)
                : max(topK * 8, 256)
            var pool = ANNTopKCollector(capacity: capacity)
            let centroids = try annCentroidCacheLocked(blocks: meta.blocks,
                                                       clusters: meta.clusters,
                                                       dim: meta.dim)
            var scratch: [Float] = []
            try query.withUnsafeBufferPointer { queryPointer in
                guard let queryBase = queryPointer.baseAddress else { return }
                for block in 0..<meta.blocks {
                    let probes = centroids.nearest(query: queryBase, block: block,
                                                   nprobe: options.nprobe, scratch: &scratch)
                    for cluster in probes {
                        try scoreANNCellsLocked(block: block, cluster: cluster, dim: meta.dim,
                                                query: queryBase, pool: &pool)
                    }
                }
            }
            if tailRows > 0 {
                try scanANNTailLocked(query: query, after: meta.lastRowID, dim: meta.dim,
                                      rowLimit: tailRows, pool: &pool)
            }
            var candidates = pool.sortedItems()
            if options.rescore > 0, !candidates.isEmpty {
                candidates = try rescoreANNCandidatesLocked(candidates, query: query,
                                                            limit: max(topK, options.rescore))
            }
            // 一个候选都没有：多半是索引里的 rowid 全被删了（清空/去重/整库重同步后
            // 水位之前的内容都不存在了）。此时不能返回空结果——交给精确检索作答。
            guard !candidates.isEmpty else { return nil }
            let hits = try hydrate(candidates: candidates, sourceFilter: sourceFilter,
                                   sourceContains: sourceContains, limit: topK)
            // 有候选但一条都 hydrate 不出来：候选 rowid 全失效，或定向过滤把候选全
            // 排除了。ANN 的「近似」只应体现在排序上，不该把它答不出的问题答成空，
            // 回退精确检索核实一次。
            return hits.isEmpty ? nil : hits
        } catch {
            AgentLog.warning("ANN: 检索失败，回退精确检索: \(error.localizedDescription)")
            return nil
        }
    }

    /// 读取一个簇的打包行并打分。
    private func scoreANNCellsLocked(block: Int, cluster: Int, dim: Int,
                                     query: UnsafePointer<Float>,
                                     pool: inout ANNTopKCollector) throws {
        let rows = try db.query("""
        SELECT payload FROM \(Self.annCellsTable)
        WHERE block_id = ? AND cluster_id = ? ORDER BY seq
        """, [.int(Int64(block)), .int(Int64(cluster))])
        guard !rows.isEmpty else { return }
        let rowBytes = 12 + dim
        var scratch = [Float](repeating: 0, count: dim)
        for row in rows {
            guard let payload = row["payload"]?.dataValue else { continue }
            let count = payload.count / rowBytes
            guard count > 0 else { continue }
            payload.withUnsafeBytes { raw in
                guard let base = raw.baseAddress else { return }
                for index in 0..<count {
                    let offset = index * rowBytes
                    let rowID = raw.loadUnaligned(fromByteOffset: offset, as: Int64.self)
                    let scale = raw.loadUnaligned(fromByteOffset: offset + 8, as: Float.self)
                    let codes = base.advanced(by: offset + 12).assumingMemoryBound(to: Int8.self)
                    vDSP_vflt8(codes, 1, &scratch, 1, vDSP_Length(dim))
                    var dot: Float = 0
                    vDSP_dotpr(scratch, 1, query, 1, &dot, vDSP_Length(dim))
                    pool.offer(rowid: rowID, score: Double(dot * scale / 127))
                }
            }
        }
    }

    /// 索引之外的新切片：按 rowid 顺序精确补齐（顺序读，实测 115MB/s）。
    private func scanANNTailLocked(query: [Float], after rowID: Int64, dim: Int,
                                   rowLimit: Int, pool: inout ANNTopKCollector) throws {
        var scanned = 0
        try query.withUnsafeBufferPointer { queryPointer in
            guard let queryBase = queryPointer.baseAddress else { return }
            try streamStoredVectors(from: rowID, dim: dim) { hitRowID, pointer in
                let floats = pointer.assumingMemoryBound(to: Float.self)
                var dot: Float = 0
                vDSP_dotpr(floats, 1, queryBase, 1, &dot, vDSP_Length(dim))
                pool.offer(rowid: hitRowID, score: Double(dot))
                scanned += 1
                return scanned < rowLimit
            }
        }
    }

    /// 用点查到的精确向量给候选重打分（消除量化误差）。
    private func rescoreANNCandidatesLocked(_ candidates: [(Int64, Double)], query: [Float],
                                            limit: Int) throws -> [(Int64, Double)] {
        let head = candidates.prefix(limit).map(\.0)
        let exact = try exactVectorsLocked(rowIDs: Array(head))
        guard !exact.isEmpty else { return candidates }
        var rescored: [(Int64, Double)] = []
        rescored.reserveCapacity(candidates.count)
        for item in candidates {
            if let vector = exact[item.0], vector.count == query.count {
                rescored.append((item.0, VectorCodec.cosine(query, vector)))
            } else {
                rescored.append(item)
            }
        }
        rescored.sort { $0.1 > $1.1 }
        return rescored
    }

    // MARK: 簇心缓存

    /// 懒加载所有块的簇心（int8 码 + 尺度）。2100 万切片约 21MB，
    /// 省掉每次检索重复读 21MB 元数据。构建/删除后置空。
    private func annCentroidCacheLocked(blocks: Int, clusters: Int,
                                        dim: Int) throws -> ANNCentroidCache {
        if let cache = annCentroidCache, cache.blocks == blocks,
           cache.clusters == clusters, cache.dim == dim {
            return cache
        }
        var codes = [Int8](repeating: 0, count: blocks * clusters * dim)
        var scales = [Float](repeating: 0, count: blocks * clusters)
        for block in 0..<blocks {
            guard let payload = try db.scalar(
                "SELECT payload FROM \(Self.annBlocksTable) WHERE block_id = ?",
                [.int(Int64(block))]).dataValue else {
                throw AgentError.storage("ANN 索引块 \(block) 缺失，需要重建")
            }
            try decodeANNCentroidPayload(payload, clusters: clusters, dim: dim,
                                         block: block, codes: &codes, scales: &scales)
        }
        let cache = ANNCentroidCache(blocks: blocks, clusters: clusters, dim: dim,
                                     codes: codes, scales: scales)
        annCentroidCache = cache
        return cache
    }

    // MARK: 块构建

    /// 按 rowid 顺序攒下一块（最多 maxRows 行）。已建块通过水位跳过，
    /// 跳过时只顺序扫 `_rowids`（不读向量数据）。
    func gatherANNBlockLocked(after rowID: Int64, dim: Int, maxRows: Int,
                              into buffer: inout ANNBlockBuffer) throws {
        buffer.reset(maxRows: maxRows, dim: dim)
        var scratch = [Int8](repeating: 0, count: dim)
        try streamStoredVectors(from: rowID, dim: dim) { hitRowID, pointer in
            buffer.append(rowID: hitRowID,
                          floats: pointer.assumingMemoryBound(to: Float.self),
                          dim: dim, scratch: &scratch)
            return buffer.count < maxRows
        }
        // 兜底：官方 vec0 读取路径不保证 rowid 顺序，排序后再切块/记录水位。
        if buffer.needsSort {
            buffer.sortByRowID()
        }
    }

    /// 训练簇心 → 分配 → 按簇打包写入（单事务）。失败整体回滚、水位不前进，
    /// 下次构建重做这一块。
    func writeANNBlockLocked(blockID: Int, dim: Int, clusters: Int, blockRows: Int,
                             iterations: Int, buffer: ANNBlockBuffer) throws {
        let rows = buffer.count
        guard rows > 0 else { return }
        let activeClusters = min(clusters, rows)
        var centroids = [Float](repeating: 0, count: clusters * dim)
        var counts = [Int32](repeating: 0, count: clusters)
        if let seeds = annSeedCentroids(codes: buffer.codes, scales: buffer.scales, rows: rows,
                                        dim: dim, clusters: activeClusters),
           let trained = ANNClustering.train(codes: buffer.codes, scales: buffer.scales,
                                             rows: rows, dim: dim, clusters: activeClusters,
                                             iterations: iterations, seeds: seeds) {
            for i in 0..<(activeClusters * dim) { centroids[i] = trained.centroids[i] }
            for cluster in 0..<activeClusters { counts[cluster] = trained.counts[cluster] }
        }
        let assignments = ANNClustering.assign(codes: buffer.codes, scales: buffer.scales,
                                               rows: rows, dim: dim, clusters: activeClusters,
                                               centroids: centroids)
        // 计数排序：簇内保持 rowid 顺序，按簇顺序写盘 → 检索时顺序读。
        var offsets = [Int](repeating: 0, count: activeClusters + 1)
        for assignment in assignments { offsets[Int(assignment) + 1] += 1 }
        for cluster in 0..<activeClusters { offsets[cluster + 1] += offsets[cluster] }
        var order = [Int32](repeating: 0, count: rows)
        var cursor = offsets
        for (index, assignment) in assignments.enumerated() {
            let cluster = Int(assignment)
            order[cursor[cluster]] = Int32(index)
            cursor[cluster] += 1
        }

        try db.transaction {
            for cluster in 0..<activeClusters {
                let start = offsets[cluster]
                let end = offsets[cluster + 1]
                guard end > start else { continue }
                let payload = ANNPacking.packCell(rowids: buffer.rowids, codes: buffer.codes,
                                                  scales: buffer.scales, order: order,
                                                  range: start..<end, dim: dim)
                try db.execute("""
                INSERT OR REPLACE INTO \(Self.annCellsTable)
                    (block_id, cluster_id, seq, count, payload)
                VALUES(?, ?, 0, ?, ?)
                """, [.int(Int64(blockID)), .int(Int64(cluster)), .int(Int64(end - start)),
                      .blob(payload)])
            }
            let centroidPayload = ANNPacking.packCentroids(centroids: centroids, counts: counts,
                                                           clusters: activeClusters, dim: dim)
            try db.execute("""
            INSERT OR REPLACE INTO \(Self.annBlocksTable)(block_id, clusters, dim, payload)
            VALUES(?, ?, ?, ?)
            """, [.int(Int64(blockID)), .int(Int64(activeClusters)), .int(Int64(dim)),
                  .blob(centroidPayload)])
            try writeANNMetaLocked(ANNStoredMeta(
                version: Self.annFormatVersion, clusters: clusters, blockRows: blockRows,
                dim: dim, blocks: blockID + 1, rows: rows,
                lastRowID: buffer.lastRowID, builtAt: Date().timeIntervalSince1970))
        }
        annCentroidCache = nil
    }

    /// 写索引 meta。`meta.rows` 是**本块新增行数**：`ann_rows` 在已有值上累加
    ///（块数、水位、簇数等其余键直接覆盖）。
    private func writeANNMetaLocked(_ meta: ANNStoredMeta) throws {
        let stored = try annStoredMetaLocked()
        let rows = stored.rows + meta.rows
        let values: [(String, String)] = [
            ("ann_ver", String(meta.version)),
            ("ann_clusters", String(meta.clusters)),
            ("ann_block_rows", String(meta.blockRows)),
            ("ann_dim", String(meta.dim)),
            ("ann_blocks", String(meta.blocks)),
            ("ann_rows", String(rows)),
            ("ann_last_rowid", String(meta.lastRowID)),
            ("ann_built_at", String(meta.builtAt)),
        ]
        for (key, value) in values {
            try db.execute("""
            INSERT INTO meta(key, value) VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, [.text(key), .text(value)])
        }
    }

    /// 初始簇心：块内均匀采样若干行（k-means 起点）。
    private func annSeedCentroids(codes: [Int8], scales: [Float], rows: Int, dim: Int,
                                  clusters: Int) -> [Float]? {
        guard rows > 0, clusters > 0 else { return nil }
        let sampleCount = min(rows, max(clusters * 4, 1024))
        var seeds = [Float](repeating: 0, count: clusters * dim)
        var row = [Float](repeating: 0, count: dim)
        var state: UInt64 = 0x9E37_79B9_7F4A_7C15
        var used = Set<Int>()
        for cluster in 0..<clusters {
            var pick = 0
            var attempts = 0
            repeat {
                state = state &* 6364136223846793005 &+ 1442695040888963407
                pick = Int((state >> 33) % UInt64(sampleCount))
                attempts += 1
            } while used.contains(pick) && attempts < 32
            used.insert(pick)
            let source = sampleCount >= rows
                ? min(pick, rows - 1)
                : min(Int(Double(pick) * Double(rows) / Double(sampleCount)), rows - 1)
            ANNClustering.dequantize(codes: codes, scales: scales, row: source, dim: dim, into: &row)
            VectorCodec.normalizeInPlace(&row)
            for i in 0..<dim { seeds[cluster * dim + i] = row[i] }
        }
        return seeds
    }

    // MARK: meta / schema

    func ensureANNSchemaLocked() throws {
        try db.execScript("""
        CREATE TABLE IF NOT EXISTS \(Self.annBlocksTable) (
            block_id INTEGER PRIMARY KEY,
            clusters INTEGER NOT NULL,
            dim INTEGER NOT NULL,
            payload BLOB NOT NULL
        );
        CREATE TABLE IF NOT EXISTS \(Self.annCellsTable) (
            block_id INTEGER NOT NULL,
            cluster_id INTEGER NOT NULL,
            seq INTEGER NOT NULL,
            count INTEGER NOT NULL,
            payload BLOB NOT NULL,
            PRIMARY KEY (block_id, cluster_id, seq)
        ) WITHOUT ROWID;
        """)
    }

    func annStoredMetaLocked() throws -> ANNStoredMeta {
        var meta = ANNStoredMeta()
        let rows = try db.query("SELECT key, value FROM meta WHERE key GLOB '\(annMetaPrefix)*'")
        var values: [String: String] = [:]
        for row in rows {
            if let key = row["key"]?.stringValue, let value = row["value"]?.stringValue {
                values[key] = value
            }
        }
        meta.version = Int(values["ann_ver"] ?? "") ?? 0
        meta.clusters = Int(values["ann_clusters"] ?? "") ?? 0
        meta.blockRows = Int(values["ann_block_rows"] ?? "") ?? 0
        meta.dim = Int(values["ann_dim"] ?? "") ?? 0
        meta.blocks = Int(values["ann_blocks"] ?? "") ?? 0
        meta.rows = Int(values["ann_rows"] ?? "") ?? 0
        meta.lastRowID = Int64(values["ann_last_rowid"] ?? "") ?? 0
        meta.builtAt = Double(values["ann_built_at"] ?? "") ?? 0
        return meta
    }

    /// 解码一块的簇心（int8 码 + 尺度）到扁平缓存。
    private func decodeANNCentroidPayload(_ payload: Data, clusters: Int, dim: Int, block: Int,
                                          codes: inout [Int8], scales: inout [Float]) throws {
        let headerBytes = 12
        let entryBytes = 8 + dim
        guard clusters > 0, dim > 0,
              payload.count >= headerBytes + clusters * entryBytes else {
            throw AgentError.storage("ANN 索引块 \(block) 数据损坏（长度 \(payload.count)）")
        }
        payload.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            var offset = headerBytes
            for cluster in 0..<clusters {
                let scale = raw.loadUnaligned(fromByteOffset: offset + 4, as: Float.self)
                let codeBase = base.advanced(by: offset + 8).assumingMemoryBound(to: Int8.self)
                let destination = block * clusters * dim + cluster * dim
                for i in 0..<dim { codes[destination + i] = codeBase[i] }
                scales[block * clusters + cluster] = scale
                offset += entryBytes
            }
        }
    }

    // MARK: 向量表读取

    /// 按 rowid 升序流式读取向量表。
    ///
    /// vec0 库走影子表快通道：`_rowids` 顺序扫描 + 按 chunk_id 懒读打包
    /// blob（实测 115MB/s）；官方 `SELECT ... FROM chunk_vectors` 只有
    /// 10MB/s（每行都要从打包块里解出独立 blob）。影子表缺失或布局校验
    /// 失败时回退官方路径。
    func streamStoredVectors(from rowID: Int64, dim: Int,
                             _ body: (Int64, UnsafeRawPointer) throws -> Bool) throws {
        if vectorEnabled, let shadow = try annShadowTablesLocked() {
            do {
                try streamShadowVectors(shadow, from: rowID, dim: dim, body)
                return
            } catch let error as ANNShadowUnavailable {
                AgentLog.warning("ANN: vec0 影子表快速通道不可用（\(error.reason)），改用官方读取路径")
            }
        }
        try streamDirectVectors(from: rowID, dim: dim, body)
    }

    /// vec0 影子表：`<table>_rowids` 与 `<table>_vector_chunks*`。
    private func annShadowTablesLocked() throws -> (rowids: String, chunks: String)? {
        let names = try db.query(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name GLOB ?",
            [.text(Self.vectorTable + "_*")]).compactMap { $0["name"]?.stringValue }
        guard let rowids = names.first(where: { $0.hasSuffix("_rowids") }),
              let chunks = names.first(where: { $0.contains("_vector_chunks") }) else { return nil }
        let columns = try db.query("PRAGMA table_info(\(rowids))")
            .compactMap { $0["name"]?.stringValue }
        guard columns.contains("chunk_id"), columns.contains("chunk_offset") else { return nil }
        return (rowids, chunks)
    }

    private func streamShadowVectors(_ shadow: (rowids: String, chunks: String),
                                     from rowID: Int64, dim: Int,
                                     _ body: (Int64, UnsafeRawPointer) throws -> Bool) throws {
        // 先做一次布局校验（第一行）：不合预期就直接回退，避免投递半截数据。
        if let probe = try db.query("""
            SELECT rowid, chunk_id, chunk_offset FROM \(shadow.rowids)
            WHERE rowid > ? ORDER BY rowid LIMIT 1
            """, [.int(rowID)]).first,
           let probeRowID = probe["rowid"]?.intValue,
           let probeChunk = probe["chunk_id"]?.intValue,
           let probeOffset = probe["chunk_offset"]?.intValue,
           let chunkData = try db.scalar(
            "SELECT vectors FROM \(shadow.chunks) WHERE rowid = ?",
            [.int(probeChunk)]).dataValue {
            let byteOffset = Int(probeOffset) * dim * 4
            guard byteOffset >= 0, byteOffset + dim * 4 <= chunkData.count else {
                throw ANNShadowUnavailable(reason: "chunk 偏移越界")
            }
            let valid = chunkData.withUnsafeBytes { raw -> Bool in
                guard let base = raw.baseAddress else { return false }
                return validateShadowVector(rowID: probeRowID,
                                            pointer: UnsafeRawPointer(base.advanced(by: byteOffset)),
                                            dim: dim)
            }
            guard valid else { throw ANNShadowUnavailable(reason: "影子表向量与点查结果不一致") }
        }

        var chunkID: Int64 = -1
        var chunkData = Data()
        var bodyError: Error?
        let rowBytes = dim * 4
        let sql = "SELECT rowid, chunk_id, chunk_offset FROM \(shadow.rowids) "
            + "WHERE rowid > ? ORDER BY rowid"
        try db.stream(sql, [.int(rowID)]) { row in
            guard bodyError == nil else { return false }
            let hitRowID = row.int(0)
            let currentChunk = row.int(1)
            let chunkOffset = Int(row.int(2))
            if currentChunk != chunkID {
                guard let data = try? db.scalar(
                    "SELECT vectors FROM \(shadow.chunks) WHERE rowid = ?",
                    [.int(currentChunk)]).dataValue else {
                    throw AgentError.storage(
                        "ANN: vec0 影子表缺少 chunk \(currentChunk)，需要重建向量表")
                }
                chunkData = data
                chunkID = currentChunk
            }
            let byteOffset = chunkOffset * rowBytes
            guard byteOffset >= 0, byteOffset + rowBytes <= chunkData.count else {
                throw AgentError.storage(
                    "ANN: vec0 影子表 chunk \(currentChunk) 偏移越界，需要重建向量表")
            }
            var proceed = true
            chunkData.withUnsafeBytes { raw in
                guard let base = raw.baseAddress else { return }
                do {
                    proceed = try body(hitRowID, UnsafeRawPointer(base.advanced(by: byteOffset)))
                } catch {
                    bodyError = error
                    proceed = false
                }
            }
            return proceed
        }
        if let bodyError { throw bodyError }
    }

    /// 校验影子打包 blob 的布局假设（`chunk_offset × dim × 4` 处的向量应与
    /// 点查一致）。每块只做一次，代价可忽略。
    private func validateShadowVector(rowID: Int64, pointer: UnsafeRawPointer,
                                      dim: Int) -> Bool {
        guard let data = try? db.scalar(
            "SELECT embedding FROM \(Self.vectorTable) WHERE rowid = ?",
            [.int(rowID)]).dataValue else { return false }
        let direct = VectorCodec.decode(data)
        guard direct.count == dim else { return false }
        let shadow = UnsafeBufferPointer(start: pointer.assumingMemoryBound(to: Float.self),
                                         count: dim)
        var dot: Float = 0
        var shadowNorm: Float = 0
        var directNorm: Float = 0
        direct.withUnsafeBufferPointer { reference in
            guard let base = reference.baseAddress else { return }
            vDSP_dotpr(shadow.baseAddress!, 1, base, 1, &dot, vDSP_Length(dim))
            vDSP_svesq(shadow.baseAddress!, 1, &shadowNorm, vDSP_Length(dim))
            vDSP_svesq(base, 1, &directNorm, vDSP_Length(dim))
        }
        let denominator = sqrt(shadowNorm) * sqrt(directNorm)
        guard denominator > 0 else { return true }
        return Double(dot / denominator) > 0.999
    }

    /// 官方读取路径：vec0 全表扫（约束只在 SQLite 侧后置过滤）或普通表顺序读。
    private func streamDirectVectors(from rowID: Int64, dim: Int,
                                     _ body: (Int64, UnsafeRawPointer) throws -> Bool) throws {
        let order = vectorEnabled ? "" : " ORDER BY rowid"
        var previous = Int64.min
        var outOfOrder = 0
        try db.stream("SELECT rowid, embedding FROM \(Self.vectorTable) WHERE rowid > ?\(order)",
                      [.int(rowID)]) { row in
            let hitRowID = row.int(0)
            if hitRowID < previous { outOfOrder += 1 }
            previous = hitRowID
            guard let blob = row.blob(1), blob.count >= dim * 4 else { return true }
            return try body(hitRowID, blob.pointer)
        }
        if outOfOrder > 0 {
            AgentLog.warning("ANN: 向量表物理顺序与 rowid 不一致（\(outOfOrder) 行），"
                + "已按 rowid 排序兜底；建议重建向量表以恢复影子表快速通道")
        }
    }
}

extension ANNBlockBuffer {
    /// 官方读取路径下可能乱序（vec0 按打包块顺序返回）。
    var needsSort: Bool {
        guard rowids.count > 1 else { return false }
        for index in 1..<rowids.count where rowids[index] < rowids[index - 1] { return true }
        return false
    }

    mutating func sortByRowID() {
        let rows = rowids.count
        let dim = rows > 0 ? codes.count / rows : 0
        guard rows > 1, dim > 0 else { return }
        let order = (0..<rows).sorted { rowids[$0] < rowids[$1] }
        var newRowids = [Int64](repeating: 0, count: rows)
        var newScales = [Float](repeating: 0, count: rows)
        var newCodes = [Int8](repeating: 0, count: rows * dim)
        for (target, source) in order.enumerated() {
            newRowids[target] = rowids[source]
            newScales[target] = scales[source]
            let from = source * dim
            let to = target * dim
            for i in 0..<dim { newCodes[to + i] = codes[from + i] }
        }
        rowids = newRowids
        scales = newScales
        codes = newCodes
        lastRowID = newRowids[rows - 1]
    }
}

// MARK: - 打包与聚类

enum ANNPacking {
    /// 一个簇的打包行：`[rowid Int64][scale Float32][int8 × dim]`。
    static func packCell(rowids: [Int64], codes: [Int8], scales: [Float],
                         order: [Int32], range: Range<Int>, dim: Int) -> Data {
        let rowBytes = 12 + dim
        let count = range.count
        var bytes = [UInt8](repeating: 0, count: count * rowBytes)
        bytes.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress else { return }
            codes.withUnsafeBufferPointer { codePointer in
                guard let codeBase = codePointer.baseAddress else { return }
                var offset = 0
                for slot in range {
                    let index = Int(order[slot])
                    raw.storeBytes(of: rowids[index], toByteOffset: offset, as: Int64.self)
                    raw.storeBytes(of: scales[index], toByteOffset: offset + 8, as: Float.self)
                    base.advanced(by: offset + 12)
                        .copyMemory(from: UnsafeRawPointer(codeBase + index * dim), byteCount: dim)
                    offset += rowBytes
                }
            }
        }
        return Data(bytes)
    }

    /// 块头：magic + 簇数 + 维度；随后每簇 `[count UInt32][scale Float32][int8 × dim]`。
    static func packCentroids(centroids: [Float], counts: [Int32],
                              clusters: Int, dim: Int) -> Data {
        var bytes = [UInt8](repeating: 0, count: 12 + clusters * (8 + dim))
        bytes.withUnsafeMutableBytes { raw in
            raw.storeBytes(of: UInt32(0x414E_4E31), toByteOffset: 0, as: UInt32.self)  // "ANN1"
            raw.storeBytes(of: UInt32(clusters), toByteOffset: 4, as: UInt32.self)
            raw.storeBytes(of: UInt32(dim), toByteOffset: 8, as: UInt32.self)
            guard let base = raw.baseAddress else { return }
            var offset = 12
            for cluster in 0..<clusters {
                let start = cluster * dim
                let end = min(start + dim, centroids.count)
                var maxAbs: Float = 0
                for i in start..<end { maxAbs = max(maxAbs, abs(centroids[i])) }
                let scale: Float = maxAbs > 0 ? maxAbs : 1
                let factor = 127 / scale
                let clusterCount = cluster < counts.count ? counts[cluster] : 0
                raw.storeBytes(of: UInt32(bitPattern: clusterCount), toByteOffset: offset,
                               as: UInt32.self)
                raw.storeBytes(of: scale, toByteOffset: offset + 4, as: Float.self)
                let codes = base.advanced(by: offset + 8).assumingMemoryBound(to: Int8.self)
                for i in 0..<dim {
                    let value = start + i < end ? centroids[start + i] : 0
                    let quantized = (value * factor).rounded()
                    codes[i] = Int8(max(-127, min(127, quantized)))
                }
                offset += 8 + dim
            }
        }
        return Data(bytes)
    }
}

enum ANNClustering {
    struct Result {
        var centroids: [Float]
        var counts: [Int32]
    }

    /// 从 int8 码还原一行（乘回量化尺度）。
    static func dequantize(codes: [Int8], scales: [Float], row: Int, dim: Int,
                           into output: inout [Float]) {
        if output.count != dim { output = [Float](repeating: 0, count: dim) }
        let scale = (row >= 0 && row < scales.count) ? scales[row] / 127 : 1
        codes.withUnsafeBufferPointer { source in
            guard let sourceBase = source.baseAddress else { return }
            output.withUnsafeMutableBufferPointer { destination in
                guard let destinationBase = destination.baseAddress else { return }
                vDSP_vflt8(sourceBase + row * dim, 1, destinationBase, 1, vDSP_Length(dim))
                var factor = scale
                vDSP_vsmul(destinationBase, 1, &factor, destinationBase, 1, vDSP_Length(dim))
            }
        }
    }

    /// 块内 k-means（余弦口径：向量与簇心都归一化，点积即余弦）。
    static func train(codes: [Int8], scales: [Float], rows: Int, dim: Int,
                      clusters: Int, iterations: Int, seeds: [Float]) -> Result? {
        guard rows > 0, clusters > 0, dim > 0, seeds.count >= clusters * dim else { return nil }
        // 训练样本最多 8192 行，均匀取样覆盖整块（避免只学到块首的内容）。
        let sampleRows = min(rows, 8192)
        var sample = [Float](repeating: 0, count: sampleRows * dim)
        var row = [Float](repeating: 0, count: dim)
        for index in 0..<sampleRows {
            let source = sampleRows >= rows
                ? min(index, rows - 1)
                : min(Int(Double(index) * Double(rows) / Double(sampleRows)), rows - 1)
            dequantize(codes: codes, scales: scales, row: source, dim: dim, into: &row)
            for i in 0..<dim { sample[index * dim + i] = row[i] }
        }
        var centroids = Array(seeds[0..<(clusters * dim)])
        for cluster in 0..<clusters {
            normalize(&centroids, offset: cluster * dim, dim: dim)
        }
        var assignments = [Int32](repeating: 0, count: sampleRows)
        var counts = [Int32](repeating: 0, count: clusters)
        for _ in 0..<max(1, iterations) {
            assignMatrix(matrix: sample, rows: sampleRows, dim: dim, clusters: clusters,
                         centroids: centroids, output: &assignments)
            var sums = [Float](repeating: 0, count: clusters * dim)
            counts = [Int32](repeating: 0, count: clusters)
            for index in 0..<sampleRows {
                let cluster = Int(assignments[index])
                counts[cluster] += 1
                for i in 0..<dim { sums[cluster * dim + i] += sample[index * dim + i] }
            }
            for cluster in 0..<clusters where counts[cluster] > 0 {
                let offset = cluster * dim
                normalize(&sums, offset: offset, dim: dim)
                for i in 0..<dim { centroids[offset + i] = sums[offset + i] }
            }
        }
        // 用最终簇心统计样本分布（供打包记录 count）。
        assignMatrix(matrix: sample, rows: sampleRows, dim: dim, clusters: clusters,
                     centroids: centroids, output: &assignments)
        counts = [Int32](repeating: 0, count: clusters)
        for index in 0..<sampleRows { counts[Int(assignments[index])] += 1 }
        return Result(centroids: centroids, counts: counts)
    }

    /// 全量分配：分批评分（Accelerate sgemm），返回每行最近簇。
    static func assign(codes: [Int8], scales: [Float], rows: Int, dim: Int,
                       clusters: Int, centroids: [Float]) -> [Int32] {
        var output = [Int32](repeating: 0, count: rows)
        guard rows > 0, clusters > 0, dim > 0 else { return output }
        // 每批控制在几十 MB（SGERA 工作集），8192 档位兼顾 sgemm 效率。
        let batch = max(1, min(rows, min(8192, max(256, 4_000_000 / max(1, dim + clusters)))))
        var matrix = [Float](repeating: 0, count: batch * dim)
        var scratch = [Int32](repeating: 0, count: batch)
        var row = [Float](repeating: 0, count: dim)
        var start = 0
        while start < rows {
            let count = min(batch, rows - start)
            for index in 0..<count {
                dequantize(codes: codes, scales: scales, row: start + index, dim: dim, into: &row)
                for i in 0..<dim { matrix[index * dim + i] = row[i] }
            }
            if scratch.count != count { scratch = [Int32](repeating: 0, count: count) }
            assignMatrix(matrix: matrix, rows: count, dim: dim, clusters: clusters,
                         centroids: centroids, output: &scratch)
            for index in 0..<count { output[start + index] = scratch[index] }
            start += count
        }
        return output
    }

    /// X(rows×dim) · C(clusters×dim)ᵀ → 每行取最大点积。
    /// 用 vDSP_mmul（不是 cblas_sgemm）：后者在 macOS 13.3+ 头文件里已标注
    /// 弃用，会持续产生编译警告。转置簇心（dim×clusters）即可复用矩阵乘法。
    static func assignMatrix(matrix: [Float], rows: Int, dim: Int, clusters: Int,
                             centroids: [Float], output: inout [Int32]) {
        guard rows > 0, clusters > 0, dim > 0, matrix.count >= rows * dim,
              centroids.count >= clusters * dim else { return }
        if output.count != rows { output = [Int32](repeating: 0, count: rows) }
        var transposed = [Float](repeating: 0, count: dim * clusters)
        for cluster in 0..<clusters {
            let source = cluster * dim
            for i in 0..<dim { transposed[i * clusters + cluster] = centroids[source + i] }
        }
        var scores = [Float](repeating: 0, count: rows * clusters)
        matrix.withUnsafeBufferPointer { matrixPointer in
            transposed.withUnsafeBufferPointer { centroidPointer in
                scores.withUnsafeMutableBufferPointer { scorePointer in
                    guard let matrixBase = matrixPointer.baseAddress,
                          let centroidBase = centroidPointer.baseAddress,
                          let scoreBase = scorePointer.baseAddress else { return }
                    vDSP_mmul(matrixBase, 1, centroidBase, 1, scoreBase, 1,
                              vDSP_Length(rows), vDSP_Length(clusters), vDSP_Length(dim))
                }
            }
        }
        scores.withUnsafeBufferPointer { pointer in
            guard let base = pointer.baseAddress else { return }
            for row in 0..<rows {
                var best = 0
                var bestScore = -Float.greatestFiniteMagnitude
                let offset = row * clusters
                for cluster in 0..<clusters {
                    let score = base[offset + cluster]
                    if score > bestScore {
                        bestScore = score
                        best = cluster
                    }
                }
                output[row] = Int32(best)
            }
        }
    }

    private static func normalize(_ vector: inout [Float], offset: Int, dim: Int) {
        var norm: Float = 0
        vector.withUnsafeMutableBufferPointer { pointer in
            guard let base = pointer.baseAddress else { return }
            vDSP_svesq(base + offset, 1, &norm, vDSP_Length(dim))
            guard norm > 0 else { return }
            var scale = 1 / sqrt(norm)
            vDSP_vsmul(base + offset, 1, &scale, base + offset, 1, vDSP_Length(dim))
        }
    }
}

extension VectorCodec {
    /// 原地归一化（避免大向量反复分配）。
    static func normalizeInPlace(_ vector: inout [Float]) {
        let count = vector.count
        guard count > 0 else { return }
        var norm: Float = 0
        vector.withUnsafeMutableBufferPointer { pointer in
            guard let base = pointer.baseAddress else { return }
            vDSP_svesq(base, 1, &norm, vDSP_Length(count))
            guard norm > 0 else { return }
            var scale = 1 / sqrt(norm)
            vDSP_vsmul(base, 1, &scale, base, 1, vDSP_Length(count))
        }
    }
}