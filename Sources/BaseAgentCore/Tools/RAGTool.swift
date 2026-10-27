import Foundation

// MARK: - 参数辅助

private extension Dictionary where Key == String, Value == JSONValue {
    func trimmedString(_ key: String) -> String {
        (self[key]?.stringValue ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func boundedInt(_ key: String, default defaultValue: Int, range: ClosedRange<Int>) -> Int {
        let value = self[key]?.intValue ?? defaultValue
        return Swift.min(Swift.max(value, range.lowerBound), range.upperBound)
    }
}

// MARK: - rag_search

/// 知识库检索工具：等价于 Python `RagSearchTool`。
public final class RagSearchTool: AgentTool {
    public let name = "rag_search"
    public let description = "从本地知识库（RAG）检索与 query 相关的内容，可限定来源文件。query 请使用用户原始措辞，不要翻译或音译。返回带来源的 Markdown 片段。"
    public let parameters: JSONValue = .object([
        "type": .string("object"),
        "properties": .object([
            "query": .object([
                "type": .string("string"),
                "description": .string("检索问题或关键词"),
            ]),
            "source": .object([
                "type": .string("string"),
                "description": .string("只检索指定来源文件（可选，支持文件名子串）"),
            ]),
            "top_k": .object([
                "type": .string("integer"),
                "description": .string("候选片段数，默认 5，范围 1-10；开启重排序时"
                    + "最终条数由设置里的 TopN 决定"),
            ]),
        ]),
        "required": .array([.string("query")]),
    ])

    private let engine: RAGEngine

    public init(config: AgentConfig,
                callbacks: AgentCallbacks? = nil,
                registry: ToolRegistry? = nil,
                engine: RAGEngine) {
        self.engine = engine
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        let query = args.trimmedString("query")
        guard !query.isEmpty else { return .fail("缺少参数: query") }
        let topK = args.boundedInt("top_k", default: 5, range: 1...10)
        let source = args.trimmedString("source")
        do {
            let output = try await engine.searchFormatted(
                query: query,
                topK: topK,
                source: source.isEmpty ? nil : source)
            return .ok(output)
        } catch {
            return .fail("知识库检索失败: \(error.localizedDescription)")
        }
    }
}

// MARK: - rag_outline

/// 文档大纲工具：等价于 Python `RagOutlineTool`。
public final class RagOutlineTool: AgentTool {
    public let name = "rag_outline"
    public let description = "查看已索引文档的章节结构（目录）与内容摘要，用于先整体了解一本书再定位细节；不传 book_name 时列出全部已索引文档。"
    public let parameters: JSONValue = .object([
        "type": .string("object"),
        "properties": .object([
            "book_name": .object([
                "type": .string("string"),
                "description": .string("文档名称（支持部分匹配）；留空则列出所有文档"),
            ]),
        ]),
        "required": .array([]),
    ])

    private let engine: RAGEngine

    public init(config: AgentConfig,
                callbacks: AgentCallbacks? = nil,
                registry: ToolRegistry? = nil,
                engine: RAGEngine) {
        self.engine = engine
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        let name = args.trimmedString("book_name")
        if name.isEmpty {
            let documents = engine.listDocuments()
            guard !documents.isEmpty else {
                return .ok("知识库中还没有已索引文档。")
            }
            let lines = ["已索引文档（共 \(documents.count) 个）:"]
                + documents.prefix(100).map { "  - \(($0 as NSString).lastPathComponent)" }
            return .ok(lines.joined(separator: "\n"))
        }

        let documents = engine.listDocuments()
        let matched = documents.first {
            ($0 as NSString).lastPathComponent.localizedCaseInsensitiveContains(name)
        } ?? name
        guard let record = engine.getDocumentOutline(docName: matched) else {
            return .fail("未找到文档: \(name)")
        }
        let digest = record.digest.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !digest.isEmpty else {
            return .fail("文档 \(record.docName) 没有可用大纲")
        }
        let chapters = record.chapters.trimmingCharacters(in: .whitespacesAndNewlines)
        var lines = ["文档: \(record.docName)", "文档 ID: \(record.docID)"]
        if !chapters.isEmpty {
            lines.append("")
            lines.append("章节结构:")
            lines.append(String(chapters.prefix(20_000)))
        }
        lines.append("")
        lines.append("内容摘要:")
        lines.append(String(digest.prefix(20_000)))
        return .ok(lines.joined(separator: "\n"))
    }
}

// MARK: - rag_status

/// 知识库状态工具：等价于 Python `RagStatusTool`。
public final class RagStatusTool: AgentTool {
    public let name = "rag_status"
    public let description = "查看知识库索引状态：知识库目录、RAG 目录、切片数、已索引文件列表。"
    public let parameters: JSONValue = .object([
        "type": .string("object"),
        "properties": .object([:]),
        "required": .array([]),
    ])

    private let engine: RAGEngine

    public init(config: AgentConfig,
                callbacks: AgentCallbacks? = nil,
                registry: ToolRegistry? = nil,
                engine: RAGEngine) {
        self.engine = engine
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        let status = engine.status()
        let knowledgeBase = status["knowledge_base"]?.stringValue ?? ""
        let ragDir = status["rag_dir"]?.stringValue ?? ""
        let chunks = status["chunks_stored"]?.intValue ?? 0
        let backend = status["vector_backend"]?.stringValue ?? ""
        let sources = status["sources"]?.arrayValue?.compactMap(\.stringValue) ?? []
        let sourceCount = status["source_count"]?.intValue ?? sources.count
        let annIndex = status["ann_index"]?.stringValue ?? ""
        let elementType = status["vector_element_type"]?.stringValue ?? ""
        let matryoshkaDim = status["matryoshka_dim"]?.intValue ?? 0
        let rebuildRequired = status["vector_rebuild_required"]?.boolValue ?? false
        let rebuildReason = status["rebuild_reason"]?.stringValue ?? ""

        var lines: [String] = ["知识库状态:"]
        lines.append("  知识库目录: \(knowledgeBase.isEmpty ? "（未设置）" : knowledgeBase)")
        lines.append("  RAG 目录: \(ragDir)")
        lines.append("  已存储切片数: \(chunks)")
        if !backend.isEmpty { lines.append("  向量后端: \(backend)") }
        if !elementType.isEmpty {
            lines.append("  向量元素类型: \(elementType)"
                + (matryoshkaDim > 0 ? "，Matryoshka 截断 \(matryoshkaDim) 维" : ""))
        }
        if rebuildRequired {
            lines.append("  向量库待重建: \(rebuildReason.isEmpty ? "配置已变化" : rebuildReason)")
            lines.append("    提示: 向量检索暂停（全文检索仍可用），请在设置中执行「重建向量库」")
        }
        if !annIndex.isEmpty { lines.append("  ANN 索引: \(annIndex)") }
        lines.append("  已索引文件数: \(sourceCount)")
        if !sources.isEmpty {
            lines.append("  已索引文件:")
            for source in sources.prefix(50) {
                lines.append("    - \(source)")
            }
            if sourceCount > min(50, sources.count) {
                lines.append("    …（仅显示前 \(min(50, sources.count)) 个，共 \(sourceCount) 个）")
            }
        }
        if let error = status["error"]?.stringValue, !error.isEmpty {
            lines.append("  错误: \(error)")
        }
        return .ok(lines.joined(separator: "\n"))
    }
}

// MARK: - rag_ingest

/// 知识库构建工具：等价于 Python `RagIngestTool`。
public final class RagIngestTool: AgentTool {
    public let name = "rag_ingest"
    public let description = "把知识库目录中的文档构建/更新到 RAG 索引。增量处理，force=true 时强制重处理全部文件。"
    public let parameters: JSONValue = .object([
        "type": .string("object"),
        "properties": .object([
            "force": .object([
                "type": .string("boolean"),
                "description": .string("是否强制重建全部索引（默认 false，仅处理变更文件）"),
            ]),
        ]),
        "required": .array([]),
    ])

    private let engine: RAGEngine

    public init(config: AgentConfig,
                callbacks: AgentCallbacks? = nil,
                registry: ToolRegistry? = nil,
                engine: RAGEngine) {
        self.engine = engine
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        let force = args["force"]?.boolValue ?? false
        let stats = await engine.ingest(force: force)

        if stats.processed == 0, stats.skipped == 0, stats.chunks == 0,
           !stats.errors.isEmpty {
            return .fail(stats.errors.joined(separator: "\n"))
        }
        if stats.cancelled {
            return .fail("RAG 同步已取消。\(stats.summary)")
        }

        var lines: [String] = ["知识库索引完成:"]
        lines.append("  新处理文件数: \(stats.processed)")
        lines.append("  跳过(未变更): \(stats.skipped)")
        lines.append("  删除文件数: \(stats.removed)")
        lines.append("  去重副本数: \(stats.deduplicated)")
        lines.append("  生成切片数: \(stats.chunks)")
        if !stats.errors.isEmpty {
            lines.append("  失败: \(stats.errors.count) 个")
            for error in stats.errors.prefix(5) {
                lines.append("    - \(error)")
            }
        }
        return .ok(lines.joined(separator: "\n"))
    }
}
