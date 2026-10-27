import Foundation
import CryptoKit

/// SQLite + sqlite-vec 向量库。
///
/// 设计要点（对应原 LanceDB 实现的替换方案）：
/// - **主表与向量表分离**：`chunks` 存正文/元数据，`chunk_vectors` 只存向量。
/// - **单事务写入**：正文 + FTS5 + 向量在同一个 `BEGIN IMMEDIATE` 事务内完成，
///   失败整体回滚。
/// - **单事务删除**：先取 rowid，再按批删除三张表，仍在一个事务内。
/// - **两段式 KNN**：先查向量表拿 rowid，再回主表按 rowid 取正文；绝不 JOIN
///   虚拟表。
/// - **过滤**：SQLite 虚拟表过滤不可靠，统一用 over-fetch 后在第二阶段过滤；
///   定向检索（source=...）时自动放大召回数。
/// - **混合检索**：FTS5 + 向量，RRF(rank Reciprocal Fusion) 融合。
public final class VectorStore: @unchecked Sendable {

    public static let chunksTable = "chunks"
    public static let vectorTable = "chunk_vectors"
    public static let ftsTable = "chunks_fts"
    public static let documentsTable = "documents"

    public let db: SQLiteDatabase
    /// 当前向量维度。极少数模型（缺少 config.json）会在首次推理后才发现
    /// 真实维度；若库还是空的，允许自动按真实维度重建。
    public private(set) var dimension: Int
    public private(set) var vectorEnabled: Bool = false
    /// 维度变化导致旧索引被清空；RAGEngine 据此同步失效 manifest。
    public private(set) var didResetForDimensionChange = false
    private let lock = NSRecursiveLock()
    private let deleteBatchSize = 400
    private let insertBatchSize = 200
    /// sqlite-vec 的 KNN k 上限为 4096；分批/过滤不能越过该值。
    private let maxVectorK = 4096
    /// 维护的切片总数，持久化在 meta 表。写操作增量更新，避免在状态/检索
    /// 热路径上执行 `COUNT(*)`：千万级切片的库上实测需要数分钟。
    /// 旧库没有该键时用 `MAX(rowid)` 作为基线（O(1)，可能略高但不阻塞）。
    private var cachedChunkCount: Int?
    private static let chunkCountKey = "chunks_count"

    public init(dbPath: String, dimension: Int, enableSQLiteVec: Bool = true) throws {
        self.dimension = max(1, dimension)
        try FileManager.default.createDirectory(
            atPath: (dbPath as NSString).deletingLastPathComponent,
            withIntermediateDirectories: true
        )
        self.db = try SQLiteDatabase(path: dbPath)
        var enabled = enableSQLiteVec && db.vectorExtensionLoaded
        try createBaseSchema()
        self.cachedChunkCount = Self.readStoredCount(db, key: Self.chunkCountKey)
        if enabled {
            do {
                try ensureVectorTable(dimension: self.dimension, useVec0: true)
            } catch {
                AgentLog.warning("vec0 建表失败，退化为精确向量扫描: \(error.localizedDescription)")
                enabled = false
                _ = try? db.execute("DROP TABLE IF EXISTS \(Self.vectorTable)")
                try ensureVectorTable(dimension: self.dimension, useVec0: false)
            }
        } else {
            try ensureVectorTable(dimension: self.dimension, useVec0: false)
        }
        self.vectorEnabled = enabled
    }

    public func close() { db.close() }

    /// 在期限内执行只读查询：先限时获取切片锁（避免大批量写入把读路径无限期
    /// 挡住），再把剩余时间交给底层语句级期限。超时抛 `AgentError.timeout`，
    /// 调用方跳过本次查询即可。
    public func withQueryDeadline<T>(seconds: Double, _ body: () throws -> T) throws -> T {
        guard seconds > 0 else { throw AgentError.timeout("查询超出时限被跳过") }
        let deadline = Date().addingTimeInterval(seconds)
        guard lock.lock(before: deadline) else {
            throw AgentError.timeout("知识库正在写入，本次查询已跳过")
        }
        defer { lock.unlock() }
        let remaining = deadline.timeIntervalSinceNow
        guard remaining > 0 else { throw AgentError.timeout("查询超出时限被跳过") }
        return try db.withDeadline(seconds: remaining, body)
    }

    // MARK: - Schema

    private func createBaseSchema() throws {
        try db.execScript("""
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            rowid INTEGER PRIMARY KEY AUTOINCREMENT,
            id TEXT NOT NULL UNIQUE,
            source TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            heading TEXT NOT NULL DEFAULT '',
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_source_hash ON chunks(source_hash, chunk_index);
        CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            tokens,
            source_hash UNINDEXED,
            chunk_index UNINDEXED,
            tokenize = 'unicode61 remove_diacritics 2'
        );

        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            doc_name TEXT NOT NULL,
            digest TEXT NOT NULL DEFAULT '',
            chapters TEXT NOT NULL DEFAULT '',
            md_path TEXT NOT NULL DEFAULT '',
            updated_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_documents_name ON documents(doc_name);
        CREATE INDEX IF NOT EXISTS idx_documents_md_path ON documents(md_path);
        """)

        let stored = try db.scalar("SELECT value FROM meta WHERE key = 'embedding_dim'").stringValue
        if let stored, Int(stored) != dimension {
            AgentLog.warning("向量维度从 \(stored) 变为 \(dimension)，清空旧索引以避免维度不匹配")
            didResetForDimensionChange = true
            try db.transaction {
                try db.execute("DROP TABLE IF EXISTS \(Self.vectorTable)")
                try db.execute("DELETE FROM \(Self.chunksTable)")
                try db.execute("DELETE FROM \(Self.ftsTable)")
                try db.execute("DELETE FROM \(Self.documentsTable)")
                // 维度重建后切片清空，计数器同步归零，避免状态栏显示旧数量。
                try db.execute(
                    "INSERT INTO meta(key, value) VALUES('\(Self.chunkCountKey)', '0') "
                    + "ON CONFLICT(key) DO UPDATE SET value = excluded.value")
            }
        }
        try db.execute("INSERT INTO meta(key, value) VALUES('embedding_dim', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                       [.text(String(dimension))])
    }

    private func ensureVectorTable(dimension: Int, useVec0: Bool) throws {
        let existingSQL = try db.scalar(
            "SELECT sql FROM sqlite_master WHERE type IN ('table','view') AND name = ?",
            [.text(Self.vectorTable)]
        ).stringValue

        if let existingSQL, existingSQL.contains("USING vec0") != useVec0 {
            try db.execute("DROP TABLE IF EXISTS \(Self.vectorTable)")
        } else if let existingSQL, useVec0 {
            // 维度已由 meta 校验；这里只确认表存在且是 vec0。
            _ = existingSQL
            return
        } else if existingSQL != nil, !useVec0 {
            // 普通表已存在，确认列结构即可。
            return
        }

        if useVec0 {
            try db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS \(Self.vectorTable)
            USING vec0(embedding float[\(dimension)] distance_metric=cosine)
            """)
        } else {
            try db.execScript("""
            CREATE TABLE IF NOT EXISTS \(Self.vectorTable) (
                rowid INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL
            );
            """)
        }
    }

    // MARK: - Write

    /// 批量写入（按 source upsert）：同一 source 的旧切片先删后插。
    /// 所有 source 的全部改动都在**一个事务**内，任意失败整体回滚。
    @discardableResult
    public func add(chunks: [Chunk], vectors: [[Float]]) throws -> Int {
        guard !chunks.isEmpty else { return 0 }
        guard chunks.count == vectors.count else {
            throw AgentError.storage("chunks 与 vectors 数量不一致: \(chunks.count) vs \(vectors.count)")
        }
        lock.lock(); defer { lock.unlock() }
        // 先确定计数基线：旧库没有 meta 计数器时用 MAX(rowid) 兜底。
        ensureChunkCounterLocked()

        // 模型未声明维度时，首次推理才发现真实值：空库允许直接重建；
        // 已有数据的库必须报错，避免静默清空用户的知识库。
        if let first = vectors.first, first.count != dimension {
            guard !hasChunks else {
                throw AgentError.storage(
                    "向量维度 \(first.count) != \(dimension)；请重建索引后再同步")
            }
            AgentLog.warning("VectorStore: 按实际输出维度 \(first.count) 重建空库（原 \(dimension)）")
            try rebuildDimensionLocked(first.count)
        }

        let uniqueSources = Set(chunks.map(\.source))
        var inserted = 0
        try withChunkCounterRollback {
            try db.transaction {
                // 1) 删除这些 source 的旧数据（分批，事务内）。
                for source in uniqueSources {
                    _ = try deleteBySourceLocked(source)
                }
                // 2) 分批插入正文 + FTS + 向量。
                var offset = 0
                while offset < chunks.count {
                    let end = min(offset + insertBatchSize, chunks.count)
                    for i in offset..<end {
                        let chunk = chunks[i]
                        let vector = VectorCodec.normalize(vectors[i])
                        guard vector.count == dimension else {
                            throw AgentError.storage("向量维度 \(vector.count) != \(dimension)")
                        }
                        let metadataJSON = JSONValue.object(chunk.metadata.mapValues { .string($0) }).jsonString()
                        let returned = try db.query("""
                        INSERT INTO chunks(id, source, source_hash, chunk_index, text, heading, metadata, created_at)
                        VALUES(?, ?, ?, ?, ?, ?, ?, ?) RETURNING rowid
                        """, [
                            .text(chunk.id), .text(chunk.source),
                            .text(Self.hashSource(chunk.source)),
                            .int(Int64(chunk.chunkIndex)), .text(chunk.text),
                            .text(chunk.heading), .text(metadataJSON),
                            .double(Date().timeIntervalSince1970),
                        ])
                        guard let rowID = returned.first?["rowid"]?.intValue else {
                            throw AgentError.storage("写入 chunks 后未返回 rowid: \(chunk.id)")
                        }
                        try db.execute(
                            "INSERT INTO chunks_fts(rowid, tokens, source_hash, chunk_index) VALUES(?, ?, ?, ?)",
                            [.int(rowID), .text(TextAnalyzer.ftsIndexText(chunk.text)),
                             .text(Self.hashSource(chunk.source)), .int(Int64(chunk.chunkIndex))]
                        )
                        try insertVector(rowID: rowID, vector: vector)
                        inserted += 1
                    }
                    offset = end
                }
                // 删除阶段已扣减，这里只补新增，随事务一起提交。
                if let cachedChunkCount {
                    self.cachedChunkCount = cachedChunkCount + inserted
                    try persistChunkCounterLocked()
                }
            }
        }
        AgentLog.info("VectorStore: 写入 \(inserted) 个切片（\(uniqueSources.count) 个来源，单事务）")
        return inserted
    }

    /// 仅在库为空时重建向量表（调用方已持锁）。
    private func rebuildDimensionLocked(_ newDimension: Int) throws {
        let resolved = max(1, newDimension)
        try db.transaction {
            try db.execute("DROP TABLE IF EXISTS \(Self.vectorTable)")
            try ensureVectorTable(dimension: resolved, useVec0: vectorEnabled)
            try db.execute(
                "INSERT INTO meta(key, value) VALUES('embedding_dim', ?) "
                + "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                [.text(String(resolved))])
            // 只在空库上重建；计数器同步归零，避免自增 rowid 残留造成虚高。
            try db.execute(
                "INSERT INTO meta(key, value) VALUES('\(Self.chunkCountKey)', '0') "
                + "ON CONFLICT(key) DO UPDATE SET value = excluded.value")
        }
        // 事务成功后再更新内存状态：失败时仍与磁盘上的旧维度一致。
        dimension = resolved
        didResetForDimensionChange = true
        cachedChunkCount = 0
    }

    private func insertVector(rowID: Int64, vector: [Float]) throws {
        let blob = VectorCodec.encode(vector)
        if vectorEnabled {
            try db.execute("INSERT INTO \(Self.vectorTable)(rowid, embedding) VALUES(?, ?)",
                           [.int(rowID), .blob(blob)])
        } else {
            try db.execute("INSERT OR REPLACE INTO \(Self.vectorTable)(rowid, embedding) VALUES(?, ?)",
                           [.int(rowID), .blob(blob)])
        }
    }

    /// 删除一个 source 的全部切片。单事务、分批。
    @discardableResult
    public func deleteBySource(_ source: String) throws -> Int {
        lock.lock(); defer { lock.unlock() }
        return try withChunkCounterRollback {
            try db.transaction {
                try deleteBySourceLocked(source)
            }
        }
    }

    private func deleteBySourceLocked(_ source: String) throws -> Int {
        let hash = Self.hashSource(source)
        return try deleteRowsLocked(
            selectSQL: "SELECT rowid FROM chunks WHERE source_hash = ?",
            bindings: [.text(hash)]
        )
    }

    private func deleteRowsLocked(selectSQL: String, bindings: [SQLValue]) throws -> Int {
        let rows = try db.query(selectSQL, bindings)
        let rowIDs = rows.compactMap { $0["rowid"]?.intValue }
        return try deleteRowIDsLocked(rowIDs)
    }

    private func deleteRowIDsLocked(_ rowIDs: [Int64]) throws -> Int {
        guard !rowIDs.isEmpty else { return 0 }
        // 删除前先确定基线；旧库没有计数器时用 MAX(rowid) 兜底。
        ensureChunkCounterLocked()
        var index = 0
        while index < rowIDs.count {
            let batch = Array(rowIDs[index..<min(index + deleteBatchSize, rowIDs.count)])
            let placeholders = Array(repeating: "?", count: batch.count).joined(separator: ",")
            let args = batch.map { SQLValue.int($0) }
            try db.execute("DELETE FROM \(Self.chunksTable) WHERE rowid IN (\(placeholders))", args)
            try db.execute("DELETE FROM \(Self.ftsTable) WHERE rowid IN (\(placeholders))", args)
            try db.execute("DELETE FROM \(Self.vectorTable) WHERE rowid IN (\(placeholders))", args)
            index += batch.count
        }
        if let cachedChunkCount {
            self.cachedChunkCount = max(0, cachedChunkCount - rowIDs.count)
            try persistChunkCounterLocked()
        }
        return rowIDs.count
    }

    /// 按 rowid 删除切片/全文索引/向量（单事务）。
    @discardableResult
    public func deleteRowIDs(_ rowIDs: [Int64]) throws -> Int {
        guard !rowIDs.isEmpty else { return 0 }
        lock.lock(); defer { lock.unlock() }
        return try withChunkCounterRollback {
            try db.transaction {
                try deleteRowIDsLocked(rowIDs)
            }
        }
    }

    /// 向量去重结果。
    public struct ChunkDedupeResult: Sendable {
        /// 扫描过的切片数。
        public var scannedRows: Int
        /// 发现的重复切片数（预览时为检测数，清理时为删除数）。
        public var duplicateRows: Int
        /// 实际删除的切片数；预览模式为 0。
        public var removedRows: Int
        public var removedBySource: [String: Int]
        /// 被清理来源 → 保留切片所属来源（用于写 tombstone）。
        public var canonicalBySource: [String: String]

        public init(scannedRows: Int = 0,
                    duplicateRows: Int = 0,
                    removedRows: Int = 0,
                    removedBySource: [String: Int] = [:],
                    canonicalBySource: [String: String] = [:]) {
            self.scannedRows = scannedRows
            self.duplicateRows = duplicateRows
            self.removedRows = removedRows
            self.removedBySource = removedBySource
            self.canonicalBySource = canonicalBySource
        }
    }

    /// 按切片正文哈希清理重复向量，保留 `created_at` 最早的一份。
    ///
    /// 相同正文在同一 Embedding 模型下会产生相同向量；用正文 SHA256 前 128 bit
    /// 作为键，避免把整个向量读入内存。分页扫描 + 单事务删除，保持 rowid 顺序稳定。
    @discardableResult
    public func deduplicateTextChunks(pageSize: Int = 512) throws -> ChunkDedupeResult {
        try scanDuplicateTextChunks(pageSize: pageSize, deleteDuplicates: true)
    }

    /// 只统计重复切片，不删除任何数据。用于清理前的确认弹窗。
    public func previewDuplicateTextChunks(pageSize: Int = 512) throws -> ChunkDedupeResult {
        try scanDuplicateTextChunks(pageSize: pageSize, deleteDuplicates: false)
    }

    private func scanDuplicateTextChunks(pageSize: Int,
                                         deleteDuplicates: Bool) throws -> ChunkDedupeResult {
        lock.lock(); defer { lock.unlock() }
        var result = ChunkDedupeResult()
        var firstSourceByHash: [Data: String] = [:]
        var cursorCreatedAt = -Double.greatestFiniteMagnitude
        var cursorRowID = Int64.min

        try withChunkCounterRollback {
            try db.transaction {
                while true {
                    let rows = try db.query("""
                        SELECT rowid, source, created_at, text FROM \(Self.chunksTable)
                        WHERE (created_at > ? OR (created_at = ? AND rowid > ?))
                        ORDER BY created_at ASC, rowid ASC
                        LIMIT ?
                        """, [
                            .double(cursorCreatedAt),
                            .double(cursorCreatedAt),
                            .int(cursorRowID),
                            .int(Int64(max(32, pageSize))),
                        ])
                    if rows.isEmpty { break }
                    result.scannedRows += rows.count

                    var duplicateRowIDs: [Int64] = []
                    for row in rows {
                        guard let rowID = row["rowid"]?.intValue,
                              let createdAt = row["created_at"]?.doubleValue else {
                            // 游标无法前进时直接结束，避免死循环。
                            return
                        }
                        cursorCreatedAt = createdAt
                        cursorRowID = rowID

                        guard let source = row["source"]?.stringValue,
                              let text = row["text"]?.stringValue,
                              !text.isEmpty else { continue }

                        let digest = SHA256.hash(data: Data(text.utf8))
                        let key = Data(digest.prefix(16))
                        if let keptSource = firstSourceByHash[key] {
                            duplicateRowIDs.append(rowID)
                            result.duplicateRows += 1
                            result.removedBySource[source, default: 0] += 1
                            if result.canonicalBySource[source] == nil {
                                result.canonicalBySource[source] = keptSource
                            }
                        } else {
                            firstSourceByHash[key] = source
                        }
                    }
                    if deleteDuplicates, !duplicateRowIDs.isEmpty {
                        result.removedRows += try deleteRowIDsLocked(duplicateRowIDs)
                    }
                }
            }
        }
        return result
    }

    /// 清空全部切片/向量/文档元数据（单事务）。
    public func clear() throws {
        lock.lock(); defer { lock.unlock() }
        try withChunkCounterRollback {
            try db.transaction {
                try db.execute("DELETE FROM \(Self.chunksTable)")
                try db.execute("DELETE FROM \(Self.ftsTable)")
                try db.execute("DELETE FROM \(Self.vectorTable)")
                try db.execute("DELETE FROM \(Self.documentsTable)")
                self.cachedChunkCount = 0
                try persistChunkCounterLocked()
            }
        }
    }

    // MARK: - Read / Search

    /// 切片总数。优先返回持久化计数器，旧库回退到 `MAX(rowid)`。
    /// 两者都是 O(1)，绝不在热路径上执行 `COUNT(*)` 全表扫描。
    public var count: Int {
        lock.lock(); defer { lock.unlock() }
        if let cachedChunkCount { return cachedChunkCount }
        return maxRowIDLocked
    }

    /// 是否存在至少一条切片。用 `LIMIT 1` 探测（O(1)，实测 2100 万切片约 0.01s），
    /// 不信任可能来自旧库 `MAX(rowid)` 的自增基线，避免空库被误判为非空。
    public var hasChunks: Bool {
        lock.lock(); defer { lock.unlock() }
        return (try? db.scalar("SELECT 1 FROM \(Self.chunksTable) LIMIT 1").intValue) != nil
    }

    private var maxRowIDLocked: Int {
        (try? db.scalar("SELECT COALESCE(MAX(rowid), 0) FROM \(Self.chunksTable)").intValue)
            .flatMap { Int($0) } ?? 0
    }

    private func ensureChunkCounterLocked() {
        if cachedChunkCount == nil {
            cachedChunkCount = maxRowIDLocked
        }
    }

    /// 事务失败时恢复内存计数器：SQLite 已回滚磁盘写入，内存值也必须回滚，
    /// 否则后续 count/status 会与库内实际切片数不一致。
    private func withChunkCounterRollback<T>(_ body: () throws -> T) rethrows -> T {
        let before = cachedChunkCount
        do {
            return try body()
        } catch {
            cachedChunkCount = before
            throw error
        }
    }

    /// 计数器写入失败必须让所在事务一起回滚，避免磁盘与内存分叉。
    private func persistChunkCounterLocked() throws {
        guard let cachedChunkCount else { return }
        try db.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            + "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            [.text(Self.chunkCountKey), .text(String(cachedChunkCount))])
    }

    private static func readStoredCount(_ db: SQLiteDatabase, key: String) -> Int? {
        guard let value = try? db.scalar(
            "SELECT value FROM meta WHERE key = ?", [.text(key)]).stringValue,
            let count = Int(value) else { return nil }
        return count
    }

    /// 已索引来源总数。**全表扫描**，只在非热路径/诊断时使用；
    /// 状态展示优先使用 manifest 中的来源数（见 `RAGEngine.status`）。
    public func sourceCount() -> Int {
        (try? db.scalar("SELECT COUNT(DISTINCT source) FROM \(Self.chunksTable)").intValue)
            .flatMap { Int($0) } ?? 0
    }

    public func listSources(limit: Int = 0) -> [String] {
        let rows: [[String: SQLValue]]?
        if limit > 0 {
            rows = try? db.query(
                "SELECT DISTINCT source FROM \(Self.chunksTable) ORDER BY source LIMIT ?",
                [.int(Int64(limit))])
        } else {
            rows = try? db.query(
                "SELECT DISTINCT source FROM \(Self.chunksTable) ORDER BY source")
        }
        return rows?.compactMap { $0["source"]?.stringValue } ?? []
    }

    /// 单个来源实际入库的切片数，用于校验 manifest 的完成状态。
    public func chunkCount(source: String) -> Int {
        let value = try? db.scalar(
            "SELECT COUNT(*) FROM \(Self.chunksTable) WHERE source = ?",
            [.text(source)]).intValue
        return value.flatMap { Int($0) } ?? 0
    }

    /// 两段式向量 KNN：
    /// 1) 向量表 KNN 取 rowid（over-fetch）；
    /// 2) 主表按 rowid 批量取正文（无 JOIN）。
    public func search(vector: [Float], topK: Int, sourceFilter: String? = nil,
                       sourceContains: String? = nil) throws -> [SearchHit] {
        guard topK > 0 else { return [] }
        guard vector.count == dimension else {
            throw AgentError.storage("查询向量维度 \(vector.count) != \(dimension)")
        }
        // 与写入/删除共用一把锁：两段式查询之间不能有事务提交，
        // 否则 KNN 拿到的 rowid 可能在 hydrate 时已被删掉。
        lock.lock(); defer { lock.unlock() }
        guard hasChunks else { return [] }
        let query = VectorCodec.normalize(vector)
        let targeted = sourceFilter != nil || sourceContains != nil

        if vectorEnabled {
            // 首轮召回也必须先夹到 sqlite-vec 的 k 上限内。
            var k = min(max(topK, targeted ? topK * 20 : topK), maxVectorK)
            while true {
                let candidates = try knnVec0(query: query, k: k)
                let hits = try hydrate(candidates: candidates, sourceFilter: sourceFilter,
                                       sourceContains: sourceContains, limit: topK)
                if hits.count >= topK || !targeted || k >= maxVectorK {
                    return hits
                }
                k = min(k * 2, maxVectorK)
            }
        } else {
            let candidates = try knnExact(query: query, sourceFilter: sourceFilter)
            return try hydrate(candidates: candidates, sourceFilter: sourceFilter,
                               sourceContains: sourceContains, limit: topK)
        }
    }

    private func knnVec0(query: [Float], k: Int) throws -> [(Int64, Double)] {
        let rows = try db.query(
            "SELECT rowid, distance FROM \(Self.vectorTable) WHERE embedding MATCH ? AND k = ?",
            [.blob(VectorCodec.encode(query)), .int(Int64(k))]
        )
        return rows.compactMap { row in
            guard let rowID = row["rowid"]?.intValue else { return nil }
            let distance = row["distance"]?.doubleValue ?? 2.0
            return (rowID, max(0, 1.0 - distance))
        }
    }

    private func knnExact(query: [Float], sourceFilter: String?) throws -> [(Int64, Double)] {
        var rows: [[String: SQLValue]]
        if let sourceFilter {
            // 仍然是两段式：先取该 source 的 rowid，再读向量，不做 JOIN。
            let ids = try db.query("SELECT rowid FROM chunks WHERE source = ?", [.text(sourceFilter)])
                .compactMap { $0["rowid"]?.intValue }
            guard !ids.isEmpty else { return [] }
            let placeholders = Array(repeating: "?", count: ids.count).joined(separator: ",")
            rows = try db.query("SELECT rowid, embedding FROM \(Self.vectorTable) WHERE rowid IN (\(placeholders))",
                                ids.map { SQLValue.int($0) })
        } else {
            rows = try db.query("SELECT rowid, embedding FROM \(Self.vectorTable)")
        }
        var scored: [(Int64, Double)] = []
        scored.reserveCapacity(rows.count)
        for row in rows {
            guard let rowID = row["rowid"]?.intValue,
                  let data = row["embedding"]?.dataValue else { continue }
            let candidate = VectorCodec.decode(data)
            scored.append((rowID, VectorCodec.cosine(query, candidate)))
        }
        return scored.sorted { $0.1 > $1.1 }
    }

    private func hydrate(candidates: [(Int64, Double)], sourceFilter: String?,
                         sourceContains: String? = nil, limit: Int) throws -> [SearchHit] {
        guard !candidates.isEmpty else { return [] }
        var result: [SearchHit] = []
        result.reserveCapacity(min(limit, candidates.count))
        var index = 0
        while index < candidates.count && result.count < limit {
            let batch = Array(candidates[index..<min(index + deleteBatchSize, candidates.count)])
            let placeholders = Array(repeating: "?", count: batch.count).joined(separator: ",")
            let rows = try db.query("""
            SELECT rowid, text, source, chunk_index, heading
            FROM \(Self.chunksTable) WHERE rowid IN (\(placeholders))
            """, batch.map { SQLValue.int($0.0) })
            var byID: [Int64: [String: SQLValue]] = [:]
            for row in rows {
                if let rowID = row["rowid"]?.intValue { byID[rowID] = row }
            }
            for (rowID, score) in batch {
                guard let row = byID[rowID] else { continue }
                let source = row["source"]?.stringValue ?? ""
                if let sourceFilter, source != sourceFilter { continue }
                if let sourceContains, !source.localizedCaseInsensitiveContains(sourceContains) { continue }
                result.append(SearchHit(
                    rowID: rowID,
                    text: row["text"]?.stringValue ?? "",
                    source: source,
                    chunkIndex: Int(row["chunk_index"]?.intValue ?? 0),
                    heading: row["heading"]?.stringValue ?? "",
                    score: score,
                    vectorScore: score
                ))
                if result.count >= limit { break }
            }
            index += batch.count
        }
        return result
    }

    /// FTS5 关键词检索，BM25 分数归一化为 [0,1)。
    public func searchFTS(query: String, topK: Int, sourceFilter: String? = nil,
                          sourceContains: String? = nil) throws -> [SearchHit] {
        guard topK > 0 else { return [] }
        let match = TextAnalyzer.ftsQuery(query)
        guard !match.isEmpty else { return [] }
        // 同 search：FTS 命中与主表 hydrate 必须处于同一写入快照。
        lock.lock(); defer { lock.unlock() }

        // 定向检索时扩大 FTS 召回，再在第二阶段过滤（FTS 表不参与 JOIN）。
        let fetchLimit = sourceContains == nil ? topK : topK * 20
        let sql: String
        let bindings: [SQLValue]
        if let sourceFilter {
            sql = """
            SELECT rowid, bm25(\(Self.ftsTable)) AS rank
            FROM \(Self.ftsTable)
            WHERE tokens MATCH ? AND source_hash = ?
            ORDER BY rank LIMIT ?
            """
            bindings = [.text(match), .text(Self.hashSource(sourceFilter)), .int(Int64(fetchLimit))]
        } else {
            sql = """
            SELECT rowid, bm25(\(Self.ftsTable)) AS rank
            FROM \(Self.ftsTable)
            WHERE tokens MATCH ?
            ORDER BY rank LIMIT ?
            """
            bindings = [.text(match), .int(Int64(fetchLimit))]
        }

        let rows: [[String: SQLValue]]
        do {
            rows = try db.query(sql, bindings)
        } catch {
            AgentLog.warning("FTS5 检索失败，退化为纯向量检索: \(error.localizedDescription)")
            return []
        }

        let ranked = rows.compactMap { row -> (Int64, Double)? in
            guard let rowID = row["rowid"]?.intValue else { return nil }
            // bm25() 越小越相关，通常是负数。
            let raw = -(row["rank"]?.doubleValue ?? 0)
            return (rowID, raw)
        }
        let hydrated = try hydrateRaw(ranked: ranked, sourceFilter: sourceFilter,
                                      sourceContains: sourceContains, limit: topK)
        return hydrated.map { hit in
            var copy = hit
            let raw = hit.score
            copy.score = raw > 0 ? raw / (1 + raw) : 0
            copy.ftsScore = raw
            return copy
        }
    }

    private func hydrateRaw(ranked: [(Int64, Double)], sourceFilter: String?,
                            sourceContains: String? = nil, limit: Int) throws -> [SearchHit] {
        guard !ranked.isEmpty else { return [] }
        let ids = ranked.map(\.0)
        let placeholders = Array(repeating: "?", count: ids.count).joined(separator: ",")
        let rows = try db.query("""
        SELECT rowid, text, source, chunk_index, heading
        FROM \(Self.chunksTable) WHERE rowid IN (\(placeholders))
        """, ids.map { SQLValue.int($0) })
        var byID: [Int64: [String: SQLValue]] = [:]
        for row in rows {
            if let rowID = row["rowid"]?.intValue { byID[rowID] = row }
        }
        var result: [SearchHit] = []
        for (rowID, score) in ranked {
            guard let row = byID[rowID] else { continue }
            let source = row["source"]?.stringValue ?? ""
            if let sourceFilter, source != sourceFilter { continue }
            if let sourceContains, !source.localizedCaseInsensitiveContains(sourceContains) { continue }
            result.append(SearchHit(
                rowID: rowID,
                text: row["text"]?.stringValue ?? "",
                source: source,
                chunkIndex: Int(row["chunk_index"]?.intValue ?? 0),
                heading: row["heading"]?.stringValue ?? "",
                score: score,
                ftsScore: score
            ))
            if result.count >= limit { break }
        }
        return result
    }

    /// 混合检索：FTS5 + 向量 → RRF 融合。
    /// - 两路各自 over-fetch（`topK * overfetch`）；
    /// - 定向检索时向量通道自动放大召回；
    /// - 融合只按 rowid，不 JOIN 虚拟表。
    public func hybridSearch(vector: [Float], query: String, topK: Int,
                             sourceFilter: String? = nil,
                             sourceContains: String? = nil,
                             rrfK: Int = 60, overfetch: Int = 4) throws -> [SearchHit] {
        let candidateCount = max(topK * max(1, overfetch) * 4, 20)
        let vectorHits = try search(vector: vector, topK: candidateCount,
                                    sourceFilter: sourceFilter, sourceContains: sourceContains)
        let ftsHits = try searchFTS(query: query, topK: candidateCount,
                                    sourceFilter: sourceFilter, sourceContains: sourceContains)

        if vectorHits.isEmpty && ftsHits.isEmpty { return [] }
        if vectorHits.isEmpty { return Array(ftsHits.prefix(topK)) }
        if ftsHits.isEmpty { return Array(vectorHits.prefix(topK)) }

        var fused: [Int64: SearchHit] = [:]
        var rrfScores: [Int64: Double] = [:]
        let k = Double(max(1, rrfK))

        for (rank, hit) in vectorHits.enumerated() {
            let contribution = 1.0 / (k + Double(rank + 1))
            rrfScores[hit.rowID, default: 0] += contribution
            fused[hit.rowID] = hit
        }
        for (rank, hit) in ftsHits.enumerated() {
            let contribution = 1.0 / (k + Double(rank + 1))
            rrfScores[hit.rowID, default: 0] += contribution
            if var existing = fused[hit.rowID] {
                existing.ftsScore = hit.ftsScore
                fused[hit.rowID] = existing
            } else {
                fused[hit.rowID] = hit
            }
        }

        var output = fused.values.map { hit -> SearchHit in
            var copy = hit
            copy.rrfScore = rrfScores[hit.rowID]
            copy.score = rrfScores[hit.rowID] ?? 0
            return copy
        }
        output.sort { ($0.rrfScore ?? 0) > ($1.rrfScore ?? 0) }
        return Array(output.prefix(topK))
    }

    // MARK: - Documents

    public func upsertDocument(docName: String, digest: String, chapters: String = "", mdPath: String = "") throws {
        let docID = Self.hashSource(mdPath.isEmpty ? docName : mdPath)
        let now = Date().timeIntervalSince1970
        try db.transaction {
            try db.execute("DELETE FROM \(Self.documentsTable) WHERE doc_id = ?", [.text(docID)])
            if !mdPath.isEmpty {
                try db.execute("DELETE FROM \(Self.documentsTable) WHERE doc_name = ? AND md_path != ?",
                               [.text(docName), .text(mdPath)])
            }
            try db.execute("""
            INSERT INTO \(Self.documentsTable)(doc_id, doc_name, digest, chapters, md_path, updated_at)
            VALUES(?, ?, ?, ?, ?, ?)
            """, [.text(docID), .text(docName), .text(digest), .text(chapters),
                  .text(mdPath), .double(now)])
        }
    }

    public func getDocumentDigest(docName: String) -> DocumentRecord? {
        let rows = allDocumentRows()
        let target = docName.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !target.isEmpty else { return nil }
        if let exact = rows.first(where: {
            $0.docName.lowercased() == target ||
            ($0.docName as NSString).lastPathComponent.lowercased() == target
        }) { return exact }
        return rows.first {
            let name = $0.docName.lowercased()
            return name.contains(target) || target.contains(name)
        }
    }

    public func allDocumentRows() -> [DocumentRecord] {
        guard let rows = try? db.query("""
        SELECT doc_id, doc_name, digest, chapters, md_path FROM \(Self.documentsTable)
        WHERE doc_name != '' ORDER BY doc_name
        """) else { return [] }
        return rows.map {
            DocumentRecord(
                docID: $0["doc_id"]?.stringValue ?? "",
                docName: $0["doc_name"]?.stringValue ?? "",
                digest: $0["digest"]?.stringValue ?? "",
                chapters: $0["chapters"]?.stringValue ?? "",
                mdPath: $0["md_path"]?.stringValue ?? ""
            )
        }
    }

    public func listDocuments() -> [String] {
        allDocumentRows().map(\.docName)
    }

    public func getDocumentByMdPath(_ mdPath: String) -> DocumentRecord? {
        guard !mdPath.isEmpty else { return nil }
        let target = (mdPath as NSString).standardizingPath
        return allDocumentRows().first {
            !$0.mdPath.isEmpty && ($0.mdPath as NSString).standardizingPath == target
        }
    }

    public func deleteDocument(docName: String) throws {
        let docID = Self.hashSource(docName)
        try db.execute("DELETE FROM \(Self.documentsTable) WHERE doc_id = ? OR doc_name = ?",
                       [.text(docID), .text(docName)])
    }

    public func deleteDocumentByMdPath(_ mdPath: String) throws -> Bool {
        guard !mdPath.isEmpty else { return false }
        let target = (mdPath as NSString).standardizingPath
        let before = try db.scalar("SELECT COUNT(*) FROM \(Self.documentsTable) WHERE md_path = ?",
                                   [.text(target)]).intValue ?? 0
        try db.execute("DELETE FROM \(Self.documentsTable) WHERE md_path = ?", [.text(target)])
        return before > 0
    }

    // MARK: - Helpers

    /// 原项目使用 md5(source)[:12] 作为稳定短哈希，这里保持一致。
    public static func hashSource(_ source: String) -> String {
        let digest = Insecure.MD5.hash(data: Data(source.utf8))
        return digest.prefix(6).map { String(format: "%02x", $0) }.joined()
    }
}
