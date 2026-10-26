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
        if let handle {
            sqlite3_close_v2(handle)
            self.handle = nil
        }
    }

    // MARK: Core execution

    @discardableResult
    public func execute(_ sql: String, _ bindings: [SQLValue] = []) throws -> Int {
        lock.lock(); defer { lock.unlock() }
        guard let handle else { throw AgentError.storage("数据库已关闭") }
        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(handle, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else {
            throw AgentError.storage("SQL 准备失败: \(String(cString: sqlite3_errmsg(handle)))\n\(sql.prefix(300))")
        }
        defer { sqlite3_finalize(stmt) }
        try bind(stmt, bindings)
        let step = sqlite3_step(stmt)
        guard step == SQLITE_DONE || step == SQLITE_ROW else {
            throw AgentError.storage("SQL 执行失败: \(String(cString: sqlite3_errmsg(handle)))\n\(sql.prefix(300))")
        }
        return Int(sqlite3_changes(handle))
    }

    public func query(_ sql: String, _ bindings: [SQLValue] = []) throws -> [[String: SQLValue]] {
        lock.lock(); defer { lock.unlock() }
        guard let handle else { throw AgentError.storage("数据库已关闭") }
        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(handle, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else {
            throw AgentError.storage("SQL 准备失败: \(String(cString: sqlite3_errmsg(handle)))\n\(sql.prefix(300))")
        }
        defer { sqlite3_finalize(stmt) }
        try bind(stmt, bindings)
        var rows: [[String: SQLValue]] = []
        while true {
            let step = sqlite3_step(stmt)
            if step == SQLITE_DONE { break }
            guard step == SQLITE_ROW else {
                throw AgentError.storage("SQL 查询失败: \(String(cString: sqlite3_errmsg(handle)))")
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

    @discardableResult
    public func execScript(_ sql: String) throws -> Int {
        lock.lock(); defer { lock.unlock() }
        guard let handle else { throw AgentError.storage("数据库已关闭") }
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
