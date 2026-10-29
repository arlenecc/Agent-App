import Foundation
import SQLite3
import CSQLiteVec

public enum SQLValue: Equatable, Sendable {
    case null
    case int(Int64)
    case double(Double)
    case text(String)
    case blob(Data)

    public init(_ value: Int) { self = .int(Int64(value)) }
    public init(_ value: Int64) { self = .int(value) }
    public init(_ value: Double) { self = .double(value) }
    public init(_ value: String) { self = .text(value) }
    public init(_ value: Bool) { self = .int(value ? 1 : 0) }
    public init(_ value: Data) { self = .blob(value) }

    public static func vector(_ floats: [Float]) -> SQLValue {
        .blob(floats.withUnsafeBufferPointer { Data(buffer: $0) })
    }

    public var intValue: Int64? {
        switch self {
        case .int(let v): return v
        case .double(let v): return Int64(v)
        case .text(let s): return Int64(s)
        case .blob: return nil
        case .null: return nil
        }
    }

    public var doubleValue: Double? {
        switch self {
        case .double(let v): return v
        case .int(let v): return Double(v)
        case .text(let s): return Double(s)
        default: return nil
        }
    }

    public var stringValue: String? {
        switch self {
        case .text(let s): return s
        case .int(let v): return String(v)
        case .double(let v): return String(v)
        default: return nil
        }
    }

    public var dataValue: Data? {
        if case .blob(let d) = self { return d }
        return nil
    }

    var isNull: Bool {
        if case .null = self { return true }
        return false
    }
}

private let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

/// 轻量 SQLite 封装：WAL、busy timeout、事务、预编译语句、sqlite-vec 注册。
public final class SQLiteDatabase {
    private var handle: OpaquePointer?
    private let lock = NSRecursiveLock()
    private var transactionDepth = 0
    public let path: String
    /// sqlite-vec 是否可用（vec0 虚拟表 + KNN）。
    public private(set) var vectorExtensionLoaded = false
    /// 当前生效的期限（`DispatchTime` 纳秒，0 = 不设期限）。只在 `lock` 内读写。
    private var deadlineNanos: UInt64 = 0
    /// 进度回调间隔（VM 指令数）：越小越灵敏、开销越大。2000 条指令约
    /// 几十微秒的量级，配合 5s 级期限足够灵敏。
    private static let progressInstructionCount = 2_000
    /// 单连接预编译语句缓存上限。检索/写入反复使用同一批 SQL，缓存后
    /// 省掉每次 prepare/finalize；64 条足够覆盖 RAG 的全部热路径。
    private static let statementCacheLimit = 64

    private struct CachedStatement {
        let stmt: OpaquePointer
        var inUse: Bool
        var lastUsed: UInt64
    }

    private var statementCache: [String: CachedStatement] = [:]
    private var statementClock: UInt64 = 0

    /// 当前缓存的语句数（测试/诊断用）。
    var statementCacheSize: Int {
        lock.lock(); defer { lock.unlock() }
        return statementCache.count
    }

    public init(path: String, readOnly: Bool = false) throws {
        self.path = path
        var db: OpaquePointer?
        let flags = readOnly ? SQLITE_OPEN_READONLY : (SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE)
        let rc = sqlite3_open_v2(path, &db, flags | SQLITE_OPEN_FULLMUTEX, nil)
        guard rc == SQLITE_OK, let db else {
            let msg = db.map { String(cString: sqlite3_errmsg($0)) } ?? "open failed"
            if let db { sqlite3_close(db) }
            throw AgentError.storage("无法打开数据库 \(path): \(msg)")
        }
        self.handle = db
        sqlite3_busy_timeout(db, 5_000)
        try execute("PRAGMA journal_mode=WAL")
        try execute("PRAGMA synchronous=NORMAL")
        try execute("PRAGMA foreign_keys=ON")
        try execute("PRAGMA temp_store=MEMORY")
        try execute("PRAGMA cache_size=-40000")
        // 将 sqlite-vec 静态注册到本连接；失败不致命，由 VectorStore 退化到精确扫描。
        if sqlite3_vec_init(db, nil, nil) == SQLITE_OK {
            vectorExtensionLoaded = true
        } else {
            AgentLog.warning("sqlite-vec 初始化失败，将使用精确向量扫描")
        }
    }

    deinit { close() }

    public func close() {
        lock.lock(); defer { lock.unlock() }
        for cached in statementCache.values {
            sqlite3_finalize(cached.stmt)
        }
        statementCache.removeAll()
        if let handle {
            sqlite3_close_v2(handle)
            self.handle = nil
        }
    }

    // MARK: Deadline

    /// 在期限内执行一批语句：到期后正在执行的语句会被 VM 层中断
    /// （`sqlite3_progress_handler` 返回非 0 → SQLITE_INTERRUPT）并抛 `AgentError.timeout`。
    ///
    /// 这是掐断长查询的唯一手段：sqlite-vec 的 KNN 循环在虚拟表内部，Swift 侧
    /// 插不进检查点；56GB / 2100 万切片的库上一次全量 KNN 可能耗时数分钟。
    /// `seconds` 必须为正，否则视为「期限已过」直接抛超时（配置里 0 = 关闭
    /// 超时由调用方决定，不要在期限内传 0）。
    public func withDeadline<T>(seconds: Double, _ body: () throws -> T) throws -> T {
        guard seconds > 0 else { throw AgentError.timeout("查询超出时限被跳过") }
        lock.lock(); defer { lock.unlock() }
        let deadline = DispatchTime.now().uptimeNanoseconds
            + UInt64(seconds * 1_000_000_000)
        let previous = deadlineNanos
        deadlineNanos = deadline
        if let handle {
            sqlite3_progress_handler(handle, Int32(Self.progressInstructionCount),
                                     { context in
                guard let context else { return 0 }
                let database = Unmanaged<SQLiteDatabase>
                    .fromOpaque(context).takeUnretainedValue()
                return database.deadlineExpired ? 1 : 0
            }, Unmanaged.passUnretained(self).toOpaque())
        }
        defer {
            deadlineNanos = previous
            if previous == 0, let handle {
                sqlite3_progress_handler(handle, 0, nil, nil)
            }
        }
        // 期限可能在进入前就已耗尽（例如「打开库」已经花光预算）：直接跳过，
        // 保证不产生任何副作用。
        try checkDeadline()
        return try body()
    }

    private var deadlineExpired: Bool {
        let deadline = deadlineNanos
        return deadline != 0 && DispatchTime.now().uptimeNanoseconds >= deadline
    }

    /// 语句入口的廉价检查：期限已过就不再开始新语句。
    private func checkDeadline() throws {
        guard deadlineExpired else { return }
        throw AgentError.timeout("查询超出时限被跳过")
    }

    /// 把「被期限中断」与普通 SQL 错误区分开：前者是预期行为，不该报成存储错误。
    private func statementError(_ handle: OpaquePointer, stage: String, sql: String) -> AgentError {
        let code = sqlite3_errcode(handle)
        let message = String(cString: sqlite3_errmsg(handle))
        if code == SQLITE_INTERRUPT {
            return AgentError.timeout("查询超出时限被中断")
        }
        return AgentError.storage("\(stage): \(message)\n\(sql.prefix(300))")
    }

    // MARK: Statement cache

    /// 取一条可执行语句。命中缓存时 reset/clear bindings 后复用；同一 SQL
    /// 正在被外层 `stream` 的递归调用使用时，另建临时语句，避免 reset 正在
    /// step 的语句。返回值第二项表示是否属于缓存（决定 release 时是否 finalize）。
    private func acquireStatement(_ sql: String) throws -> (stmt: OpaquePointer, cached: Bool) {
        guard let handle else { throw AgentError.storage("数据库已关闭") }
        statementClock &+= 1
        if var entry = statementCache[sql], !entry.inUse {
            sqlite3_reset(entry.stmt)
            sqlite3_clear_bindings(entry.stmt)
            entry.inUse = true
            entry.lastUsed = statementClock
            statementCache[sql] = entry
            return (entry.stmt, true)
        }

        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(handle, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK, let prepared = stmt else {
            throw statementError(handle, stage: "SQL 准备失败", sql: sql)
        }

        // 只有同一 SQL 尚未缓存时才可能放进去；缓存满时淘汰最久未用且
        // 不在执行中的语句。全部在执行中（递归查询）则本次不缓存。
        var cacheable = statementCache[sql] == nil
        if cacheable, statementCache.count >= Self.statementCacheLimit {
            let victim = statementCache
                .filter { !$0.value.inUse }
                .min { $0.value.lastUsed < $1.value.lastUsed }?.key
            if let victim, let entry = statementCache.removeValue(forKey: victim) {
                sqlite3_finalize(entry.stmt)
            } else {
                cacheable = false
            }
        }
        if cacheable {
            statementCache[sql] = CachedStatement(stmt: prepared, inUse: true,
                                                 lastUsed: statementClock)
        }
        return (prepared, cacheable)
    }

    private func releaseStatement(_ sql: String, _ stmt: OpaquePointer, cached: Bool) {
        if cached, var entry = statementCache[sql], entry.stmt == stmt {
            // 必须 reset 后再放回缓存：提前 break 的 `stream` 可能停在
            // SQLITE_ROW，未 reset 的语句会让同一连接上的 COMMIT 报
            // "SQL statements in progress"。
            sqlite3_reset(stmt)
            sqlite3_clear_bindings(stmt)
            entry.inUse = false
            statementClock &+= 1
            entry.lastUsed = statementClock
            statementCache[sql] = entry
        } else {
            sqlite3_finalize(stmt)
        }
    }

    // MARK: Core execution

    @discardableResult
    public func execute(_ sql: String, _ bindings: [SQLValue] = []) throws -> Int {
        lock.lock(); defer { lock.unlock() }
        guard let handle else { throw AgentError.storage("数据库已关闭") }
        try checkDeadline()
        let (stmt, cached) = try acquireStatement(sql)
        defer { releaseStatement(sql, stmt, cached: cached) }
        try bind(stmt, bindings)
        let step = sqlite3_step(stmt)
        guard step == SQLITE_DONE || step == SQLITE_ROW else {
            throw statementError(handle, stage: "SQL 执行失败", sql: sql)
        }
        return Int(sqlite3_changes(handle))
    }

    public func query(_ sql: String, _ bindings: [SQLValue] = []) throws -> [[String: SQLValue]] {
        lock.lock(); defer { lock.unlock() }
        guard let handle else { throw AgentError.storage("数据库已关闭") }
        try checkDeadline()
        let (stmt, cached) = try acquireStatement(sql)
        defer { releaseStatement(sql, stmt, cached: cached) }
        try bind(stmt, bindings)
        var rows: [[String: SQLValue]] = []
        while true {
            let step = sqlite3_step(stmt)
            if step == SQLITE_DONE { break }
            guard step == SQLITE_ROW else {
                throw statementError(handle, stage: "SQL 查询失败", sql: sql)
            }
            var row: [String: SQLValue] = [:]
            let count = sqlite3_column_count(stmt)
            for i in 0..<count {
                let name = String(cString: sqlite3_column_name(stmt, i))
                row[name] = columnValue(stmt, Int32(i))
            }
            rows.append(row)
        }
        return rows
    }

    public func scalar(_ sql: String, _ bindings: [SQLValue] = []) throws -> SQLValue {
        try query(sql, bindings).first?.values.first ?? .null
    }

    // MARK: Streaming

    /// 流式读取的轻量行视图：按列序号直接取值，不在 Swift 侧构造字典
    /// 或整行数组。`blob` 返回的是 SQLite 内部缓冲区指针，**只在本次回调
    /// 内有效**（下一次 step/finalize 后失效），调用方不得持有。
    public struct StreamRow {
        let stmt: OpaquePointer?

        public var columnCount: Int { Int(sqlite3_column_count(stmt)) }
        public func isNull(_ index: Int32) -> Bool { sqlite3_column_type(stmt, index) == SQLITE_NULL }
        public func int(_ index: Int32) -> Int64 { sqlite3_column_int64(stmt, index) }
        public func double(_ index: Int32) -> Double { sqlite3_column_double(stmt, index) }

        public func text(_ index: Int32) -> String? {
            guard let cString = sqlite3_column_text(stmt, index) else { return nil }
            return String(cString: cString)
        }

        /// blob 的原始指针与字节数；指针只在回调内有效。
        public func blob(_ index: Int32) -> (pointer: UnsafeRawPointer, count: Int)? {
            guard let bytes = sqlite3_column_blob(stmt, index) else { return nil }
            return (UnsafeRawPointer(bytes), Int(sqlite3_column_bytes(stmt, index)))
        }
    }

    /// 逐行流式执行查询，`row` 返回 false 时提前结束。
    ///
    /// 用于千万级切片的顺序扫描（构建 ANN 索引）：`query` 会把每一行都
    /// 变成 `[String: SQLValue]`，2100 万行时仅字典分配就要数分钟、内存
    /// 也放不下。回调内可以再调用本对象的其它方法（同一线程递归加锁）。
    public func stream(_ sql: String, _ bindings: [SQLValue] = [],
                       row: (StreamRow) throws -> Bool) throws {
        lock.lock(); defer { lock.unlock() }
        guard let handle else { throw AgentError.storage("数据库已关闭") }
        try checkDeadline()
        let (stmt, cached) = try acquireStatement(sql)
        defer { releaseStatement(sql, stmt, cached: cached) }
        try bind(stmt, bindings)
        while true {
            let step = sqlite3_step(stmt)
            if step == SQLITE_DONE { break }
            guard step == SQLITE_ROW else {
                throw statementError(handle, stage: "SQL 查询失败", sql: sql)
            }
            if try !row(StreamRow(stmt: stmt)) { break }
        }
    }

    @discardableResult
    public func execScript(_ sql: String) throws -> Int {
        lock.lock(); defer { lock.unlock() }
        guard let handle else { throw AgentError.storage("数据库已关闭") }
        try checkDeadline()
        var err: UnsafeMutablePointer<CChar>?
        let rc = sqlite3_exec(handle, sql, nil, nil, &err)
        if rc != SQLITE_OK {
            let msg = err.map { String(cString: $0) } ?? "unknown"
            sqlite3_free(err)
            throw AgentError.storage("SQL 脚本失败: \(msg)")
        }
        return Int(sqlite3_changes(handle))
    }

    // MARK: Transactions

    /// 写事务（BEGIN IMMEDIATE）。可嵌套（savepoint）。
    @discardableResult
    public func transaction<T>(_ body: () throws -> T) throws -> T {
        lock.lock(); defer { lock.unlock() }
        guard let handle else { throw AgentError.storage("数据库已关闭") }
        let nested = transactionDepth > 0
        let name = "sp_\(transactionDepth)"
        if nested {
            try execute("SAVEPOINT \(name)")
        } else {
            try execute("BEGIN IMMEDIATE")
        }
        transactionDepth += 1
        do {
            let value = try body()
            if nested {
                try execute("RELEASE SAVEPOINT \(name)")
            } else {
                try execute("COMMIT")
            }
            // 只有提交/释放成功后才递减；COMMIT 抛错时由 catch 统一递减，
            // 避免 transactionDepth 变成负数、后续事务被误判为嵌套。
            transactionDepth -= 1
            return value
        } catch {
            transactionDepth -= 1
            if nested {
                try? execute("ROLLBACK TO SAVEPOINT \(name)")
                try? execute("RELEASE SAVEPOINT \(name)")
            } else {
                try? execute("ROLLBACK")
            }
            throw error
        }
    }

    // MARK: Bind / Read

    private func bind(_ stmt: OpaquePointer?, _ values: [SQLValue]) throws {
        for (offset, value) in values.enumerated() {
            let index = Int32(offset + 1)
            switch value {
            case .null:
                sqlite3_bind_null(stmt, index)
            case .int(let v):
                sqlite3_bind_int64(stmt, index, v)
            case .double(let v):
                sqlite3_bind_double(stmt, index, v)
            case .text(let s):
                sqlite3_bind_text(stmt, index, s, -1, SQLITE_TRANSIENT)
            case .blob(let d):
                d.withUnsafeBytes { raw in
                    sqlite3_bind_blob(stmt, index, raw.baseAddress, Int32(d.count), SQLITE_TRANSIENT)
                }
            }
        }
    }

    private func columnValue(_ stmt: OpaquePointer?, _ index: Int32) -> SQLValue {
        switch sqlite3_column_type(stmt, index) {
        case SQLITE_INTEGER:
            return .int(sqlite3_column_int64(stmt, index))
        case SQLITE_FLOAT:
            return .double(sqlite3_column_double(stmt, index))
        case SQLITE_TEXT:
            guard let cString = sqlite3_column_text(stmt, index) else { return .null }
            return .text(String(cString: cString))
        case SQLITE_BLOB:
            guard let bytes = sqlite3_column_blob(stmt, index) else { return .blob(Data()) }
            let count = Int(sqlite3_column_bytes(stmt, index))
            return .blob(Data(bytes: bytes, count: count))
        default:
            return .null
        }
    }
}

// MARK: - Vector blob helpers

public enum VectorCodec {
    public static func encode(_ vector: [Float]) -> Data {
        vector.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    public static func decode(_ data: Data) -> [Float] {
        guard data.count >= 4 else { return [] }
        return data.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self))
        }
    }

    public static func cosine(_ a: [Float], _ b: [Float]) -> Double {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0..<a.count {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        let denom = sqrt(na) * sqrt(nb)
        return denom > 0 ? Double(dot / denom) : 0
    }

    public static func normalize(_ vector: [Float]) -> [Float] {
        var norm: Float = 0
        for v in vector { norm += v * v }
        norm = sqrt(norm)
        guard norm > 0 else { return vector }
        return vector.map { $0 / norm }
    }

    /// L2 距离 → 相似度（sqlite-vec 的 vec_distance_l2）。
    public static func l2ToSimilarity(_ distance: Double) -> Double {
        1.0 / (1.0 + max(0, distance))
    }
}
