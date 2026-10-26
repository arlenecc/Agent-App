import Foundation

/// 一条长期记忆命中。
public struct LongTermMemoryHit: Sendable, Equatable {
    public var text: String
    public var score: Double

    public init(text: String, score: Double) {
        self.text = text
        self.score = score
    }
}

/// 长期记忆（扁平事实列表）。
///
/// 持久化格式与 Python 旧版一致：`{"facts": ["...", "..."]}`，
/// 便于已有 `long_memory.json` 直接迁移。向量只在内存中缓存；
/// 重启后首次检索会重新 embed 缺失条目。
///
/// - 超过容量上限时按最旧优先淘汰；
/// - 写入时按文本去重；
/// - embedding 不可用 / 调用失败时自动回退到词法匹配。
public final class LongTermMemory {
    public let path: String
    /// 容量上限（按插入顺序淘汰最旧条目）。
    public var maxFacts: Int = 5_000

    private let embedding: EmbeddingProvider?
    private let lock = NSRecursiveLock()
    private var facts: [String] = []
    private var vectors: [String: [Float]] = [:]
    private var embeddingDisabled = false

    public init(path: String, embedding: EmbeddingProvider? = nil) {
        self.path = path
        self.embedding = embedding
        load()
    }

    /// 便捷构造：`<workspace>/.agent/long_memory.json`。
    public convenience init(workspace: String, embedding: EmbeddingProvider? = nil) {
        self.init(path: (workspace as NSString)
            .appendingPathComponent(".agent")
            .appending("/long_memory.json"), embedding: embedding)
    }

    // MARK: - 写入

    public func add(_ fact: String) {
        addMany([fact])
    }

    public func addMany(_ newFacts: [String]) {
        let cleaned = newFacts
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !cleaned.isEmpty else { return }

        lock.lock(); defer { lock.unlock() }
        var existing = Set(facts)
        for fact in cleaned where !existing.contains(fact) {
            facts.append(fact)
            existing.insert(fact)
        }
        pruneLocked()
        saveLocked()
    }

    public func all() -> [String] {
        lock.lock(); defer { lock.unlock() }
        return facts
    }

    public func clear() {
        lock.lock(); defer { lock.unlock() }
        facts = []
        vectors = [:]
        saveLocked()
    }

    // MARK: - 检索

    /// 语义检索；embedding 不可用时回退为词法匹配。
    public func search(query: String, topK: Int = 8) async -> [LongTermMemoryHit] {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { return [] }
        let limit = max(1, topK)

        let snapshot: [String]
        lock.lock(); snapshot = facts; lock.unlock()
        guard !snapshot.isEmpty else { return [] }

        if let embedding, !isEmbeddingDisabled {
            do {
                try await ensureVectors(snapshot, embedding: embedding)
                let queryVector = try await embedding.embedQuery(q)
                let scored = score(snapshot, queryVector: queryVector)
                return Array(scored.filter { $0.score >= 0.5 }
                    .sorted { $0.score > $1.score }
                    .prefix(limit))
            } catch {
                AgentLog.warning("long_memory: 语义检索不可用，回退词法匹配: \(error.localizedDescription)")
                lock.lock(); embeddingDisabled = true; lock.unlock()
            }
        }
        return lexicalSearch(snapshot, query: q, topK: limit)
    }

    /// 用于 prompt 注入的紧凑摘要。
    public func snapshot(maxItems: Int = 10) -> String {
        lock.lock(); let items = facts; lock.unlock()
        guard !items.isEmpty else { return "" }
        return items.prefix(max(0, maxItems)).map { "- \($0)" }.joined(separator: "\n")
    }

    // MARK: - 向量

    private var isEmbeddingDisabled: Bool {
        lock.lock(); defer { lock.unlock() }
        return embeddingDisabled
    }

    private func ensureVectors(_ snapshot: [String], embedding: EmbeddingProvider) async throws {
        lock.lock()
        let missing = snapshot.filter { vectors[$0] == nil }
        lock.unlock()
        guard !missing.isEmpty else { return }

        let embedded = try await embedding.embedDocuments(missing)
        guard embedded.count == missing.count else {
            throw AgentError.storage("long_memory embedding 数量不一致")
        }
        lock.lock()
        for (text, vector) in zip(missing, embedded) {
            vectors[text] = vector
        }
        lock.unlock()
    }

    private func score(_ snapshot: [String], queryVector: [Float]) -> [LongTermMemoryHit] {
        var out: [LongTermMemoryHit] = []
        lock.lock()
        for fact in snapshot {
            guard let vector = vectors[fact] else { continue }
            out.append(LongTermMemoryHit(text: fact,
                                         score: Self.cosine(queryVector, vector)))
        }
        lock.unlock()
        return out
    }

    // MARK: - 词法回退

    private func lexicalSearch(_ snapshot: [String], query: String, topK: Int) -> [LongTermMemoryHit] {
        let queryTokens = Set(Self.tokens(query))
        guard !queryTokens.isEmpty else { return [] }
        let lowered = query.lowercased()
        var hits: [LongTermMemoryHit] = []
        for fact in snapshot {
            let factTokens = Set(Self.tokens(fact))
            let overlap = queryTokens.intersection(factTokens).count
            var score = queryTokens.isEmpty ? 0 : Double(overlap) / Double(queryTokens.count)
            if fact.lowercased().contains(lowered) { score = max(score, 0.5) }
            if score > 0 {
                hits.append(LongTermMemoryHit(text: fact, score: score))
            }
        }
        return Array(hits.sorted { $0.score > $1.score }.prefix(topK))
    }

    // MARK: - 持久化

    private func load() {
        lock.lock(); defer { lock.unlock() }
        let fm = FileManager.default
        guard fm.fileExists(atPath: path),
              let raw = try? String(contentsOfFile: path, encoding: .utf8),
              let root = JSONValue.parse(raw)?.objectValue else {
            facts = []
            return
        }
        var loaded: [String] = []
        if let list = root["facts"]?.arrayValue {
            loaded = list.compactMap(\.stringValue)
        }
        // 兼容我们自己的 observations 结构（若用户手工扩展过）。
        if let observations = root["observations"]?.arrayValue {
            for item in observations {
                if let text = item.objectValue?["text"]?.stringValue {
                    loaded.append(text)
                } else if let text = item.stringValue {
                    loaded.append(text)
                }
            }
        }
        facts = Array(loaded
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .prefix(maxFacts))
    }

    private func pruneLocked() {
        if facts.count > maxFacts {
            let dropped = facts.count - maxFacts
            let removed = facts.prefix(dropped)
            facts.removeFirst(dropped)
            for text in removed { vectors.removeValue(forKey: text) }
        }
        // 向量缓存只保留仍在 facts 中的条目。
        if vectors.count > facts.count * 2 {
            let alive = Set(facts)
            vectors = vectors.filter { alive.contains($0.key) }
        }
    }

    private func saveLocked() {
        let payload: [String: JSONValue] = [
            "facts": .array(facts.map { .string($0) }),
        ]
        do {
            let text = JSONValue.object(payload).jsonString(pretty: true) + "\n"
            try AtomicFile.write(Data(text.utf8), to: path)
        } catch {
            AgentLog.warning("long_memory: 写入失败 \(path): \(error.localizedDescription)")
        }
    }

    // MARK: - 工具函数

    private static func cosine(_ a: [Float], _ b: [Float]) -> Double {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0..<a.count {
            let x = Double(a[i]), y = Double(b[i])
            dot += x * y
            na += x * x
            nb += y * y
        }
        let denom = (na.squareRoot() * nb.squareRoot())
        return denom > 0 ? max(0, min(1, dot / denom)) : 0
    }

    /// 英文按词、中文按连续片段 + 二元组切分（与技能关键词策略一致）。
    private static func tokens(_ text: String) -> [String] {
        var out: [String] = []
        var current = ""
        var cjkRun = ""

        func flushWord() {
            if !current.isEmpty { out.append(current.lowercased()); current = "" }
        }
        func flushCJK() {
            guard !cjkRun.isEmpty else { return }
            out.append(cjkRun)
            let chars = Array(cjkRun)
            if chars.count > 1 {
                for i in 0..<(chars.count - 1) {
                    out.append(String(chars[i...i + 1]))
                }
            }
            cjkRun = ""
        }

        for scalar in text.unicodeScalars {
            let value = scalar.value
            if (0x4E00...0x9FFF).contains(value) {
                flushWord()
                cjkRun.unicodeScalars.append(scalar)
                continue
            }
            flushCJK()
            if CharacterSet.alphanumerics.contains(scalar) || scalar == "_" {
                current.unicodeScalars.append(scalar)
            } else {
                flushWord()
            }
        }
        flushWord()
        flushCJK()
        return out
    }
}
