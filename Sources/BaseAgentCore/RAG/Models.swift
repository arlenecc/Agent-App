import Foundation

// MARK: - 文档与切片

/// 解析器输出的统一文档结构。
public struct ParsedDocument: Sendable {
    public var text: String
    public var source: String
    /// 章节/标题大纲（供 rag_outline 使用），例如 "第一章 ...\n  1.1 ..."。
    public var outline: String
    /// 解析器用于日志/诊断的元信息。
    public var metadata: [String: String]
    public var pageCount: Int

    public init(text: String, source: String, outline: String = "",
                metadata: [String: String] = [:], pageCount: Int = 0) {
        self.text = text
        self.source = source
        self.outline = outline
        self.metadata = metadata
        self.pageCount = pageCount
    }
}

public struct Chunk: Sendable, Identifiable {
    public var id: String
    public var text: String
    public var source: String
    public var chunkIndex: Int
    /// 可选章节/标题路径，用于过滤与展示。
    public var heading: String
    public var metadata: [String: String]

    public init(text: String, source: String, chunkIndex: Int,
                heading: String = "", metadata: [String: String] = [:]) {
        self.id = "\(VectorStore.hashSource(source))_\(chunkIndex)"
        self.text = text
        self.source = source
        self.chunkIndex = chunkIndex
        self.heading = heading
        self.metadata = metadata
    }
}

public struct SearchHit: Sendable, Identifiable {
    /// chunks 表主键（rowid）。
    public var rowID: Int64
    public var text: String
    public var source: String
    public var chunkIndex: Int
    public var heading: String
    /// 最终分数（RRF 融合值或 rerank 语义分）。
    public var score: Double
    /// 各路原始分数，用于调试。
    public var vectorScore: Double?
    public var ftsScore: Double?
    public var rrfScore: Double?

    public var id: Int64 { rowID }

    public init(rowID: Int64, text: String, source: String, chunkIndex: Int,
                heading: String = "", score: Double = 0,
                vectorScore: Double? = nil, ftsScore: Double? = nil,
                rrfScore: Double? = nil) {
        self.rowID = rowID
        self.text = text
        self.source = source
        self.chunkIndex = chunkIndex
        self.heading = heading
        self.score = score
        self.vectorScore = vectorScore
        self.ftsScore = ftsScore
        self.rrfScore = rrfScore
    }
}

public struct DocumentRecord: Sendable, Equatable {
    public var docID: String
    public var docName: String
    public var digest: String
    public var chapters: String
    public var mdPath: String

    public init(docID: String, docName: String, digest: String,
                chapters: String = "", mdPath: String = "") {
        self.docID = docID
        self.docName = docName
        self.digest = digest
        self.chapters = chapters
        self.mdPath = mdPath
    }
}

// MARK: - 模型协议

public protocol EmbeddingProvider: AnyObject, Sendable {
    var identifier: String { get }
    var dimension: Int { get }
    func embedDocuments(_ texts: [String]) async throws -> [[Float]]
    func embedQuery(_ text: String) async throws -> [Float]
    /// 释放模型权重，provider 本身保留；下次调用会按需重新加载。
    func unload() async
    /// 底层模型可安全并发推理的调用数：ONNX session.run 线程安全可并发；
    /// MLX 受全局 Metal 队列限制只能串行。调度器据此限流。
    var maxConcurrentRuns: Int { get }
}

public extension EmbeddingProvider {
    func unload() async {}
    var maxConcurrentRuns: Int { 1 }
}

public protocol RerankProvider: AnyObject, Sendable {
    var identifier: String { get }
    /// 返回与 candidates 等长的相关性分数（越大越相关）。
    func rerank(query: String, documents: [String]) async throws -> [Double]
    /// 释放模型权重，provider 本身保留；下次调用会按需重新加载。
    func unload() async
}

public extension RerankProvider {
    func unload() async {}
}

// MARK: - 进度回调

public struct IngestProgress: Sendable {
    public var stage: String
    public var current: Int
    public var total: Int
    public var message: String

    public init(stage: String, current: Int = 0, total: Int = 0, message: String = "") {
        self.stage = stage
        self.current = current
        self.total = total
        self.message = message
    }
}
