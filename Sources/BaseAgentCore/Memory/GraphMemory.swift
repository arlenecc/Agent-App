import Foundation

// MARK: - 数据模型

/// 图谱实体（与 Python `graph_memory.py` 的 entity 结构一致）。
public struct GraphEntity: Codable, Sendable, Equatable, Identifiable {
    public var name: String
    public var type: String
    public var observations: [String]
    public var createdAt: String

    public var id: String { name.lowercased() }

    public init(name: String, type: String = "", observations: [String] = [],
                createdAt: String = "") {
        self.name = name
        self.type = type
        self.observations = observations
        self.createdAt = createdAt
    }

    enum CodingKeys: String, CodingKey {
        case name, type, observations
        case createdAt = "created_at"
    }
}

/// 有向关系：source --[label]--> target。
public struct GraphRelation: Codable, Sendable, Equatable {
    public var source: String
    public var target: String
    public var label: String
    public var created: String?

    public init(source: String, target: String, label: String, created: String? = nil) {
        self.source = source
        self.target = target
        self.label = label
        self.created = created
    }
}

/// 一次观察检索命中。
public struct GraphObservationHit: Sendable, Equatable {
    public var entity: String
    public var text: String
    public var score: Double

    public init(entity: String, text: String, score: Double) {
        self.entity = entity
        self.text = text
        self.score = score
    }
}

/// 邻居查询结果（中心实体 + 半径内的实体与关系）。
public struct GraphNeighborhood: Sendable, Equatable {
    public var center: GraphEntity?
    public var entities: [GraphEntity]
    public var relations: [GraphRelation]

    public init(center: GraphEntity?, entities: [GraphEntity], relations: [GraphRelation]) {
        self.center = center
        self.entities = entities
        self.relations = relations
    }
}

// MARK: - 图谱存储

/// 线程安全的知识图谱存储：实体 + 关系 + 观察。
///
/// 持久化文件为单个 JSON：
/// `{"entities": {"<lower-name>": {name,type,observations,created_at}},
///   "relations": [{source,target,label,created}]}`
/// 与 Python 版本格式兼容。观察向量只缓存在内存中，首次检索时懒计算；
/// embedding 不可用或失败时自动回退到关键词子串匹配。
public final class GraphMemoryStore {
    public let path: String

    /// 单实体观察上限（保留最新）。
    public var maxObservationsPerEntity: Int = 200
    /// 实体总数上限（按 created_at 淘汰最旧）。
    public var maxEntities: Int = 5_000
    /// 关系总数上限（淘汰最旧）。
    public var maxRelations: Int = 20_000

    private let embedding: EmbeddingProvider?
    private let lock = NSRecursiveLock()
    private var dirty = false
    private var pendingSave: DispatchWorkItem?
    private var entities: [String: GraphEntity] = [:]   // key = name.lowercased()
    private var relations: [GraphRelation] = []
    private var vectors: [String: [Float]] = [:]        // key = vectorKey(entity, text)
    private var embeddingDisabled = false

    public init(path: String, embedding: EmbeddingProvider? = nil) {
        self.path = path
        self.embedding = embedding
        load()
    }

    /// 便捷构造：`<workspace>/.agent/long_memory.graph.json`。
    public convenience init(workspace: String, embedding: EmbeddingProvider? = nil) {
        self.init(path: (workspace as NSString)
            .appendingPathComponent(".agent")
            .appending("/long_memory.graph.json"), embedding: embedding)
    }

    private static let sharedLock = NSLock()
    private static var sharedStores: [String: GraphMemoryStore] = [:]

    /// 同一 workspace 的默认共享实例，确保多个工具看到同一份内存状态。
    public static func shared(workspace: String,
                              embedding: EmbeddingProvider? = nil) -> GraphMemoryStore {
        let path = (workspace as NSString)
            .appendingPathComponent(".agent")
            .appending("/long_memory.graph.json")
        sharedLock.lock(); defer { sharedLock.unlock() }
        if let existing = sharedStores[path] { return existing }
        let store = GraphMemoryStore(path: path, embedding: embedding)
        sharedStores[path] = store
        return store
    }

    /// 进程退出/切换 workspace 前确保共享图谱已落盘。
    public static func flushShared() {
        sharedLock.lock()
        let stores = Array(sharedStores.values)
        sharedLock.unlock()
        for store in stores { store.flush() }
    }

    /// 关闭 Agent 时释放该 workspace 的共享实例。
    ///
    /// 只从注册表移除并落盘：仍持有该实例的工具可以继续使用，
    /// 但下次 `shared()` 会重新从磁盘加载，避免跨 workspace 切换后
    /// 旧图谱常驻内存。
    public static func releaseShared(workspace: String) {
        let path = (workspace as NSString)
            .appendingPathComponent(".agent")
            .appending("/long_memory.graph.json")
        sharedLock.lock()
        let store = sharedStores.removeValue(forKey: path)
        sharedLock.unlock()
        store?.flush()
    }

    // MARK: - 实体

    /// 创建实体；同名（大小写不敏感）时合并观察。
    @discardableResult
    public func upsertEntity(_ name: String, type entityType: String = "",
                             observations: [String] = []) -> GraphEntity {
        let normalized = name.trimmingCharacters(in: .whitespacesAndNewlines)
        let cleaned = observations
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !normalized.isEmpty else {
            return GraphEntity(name: "", type: entityType, observations: [], createdAt: Self.now())
        }

        lock.lock()
        let key = normalized.lowercased()
        var entity: GraphEntity
        if var existing = entities[key] {
            if entityType != "" && existing.type.isEmpty {
                existing.type = entityType
            }
            var known = Set(existing.observations)
            for obs in cleaned where !known.contains(obs) {
                existing.observations.append(obs)
                known.insert(obs)
            }
            entity = existing
            entities[key] = existing
        } else {
            var known = Set<String>()
            var unique: [String] = []
            for obs in cleaned where !known.contains(obs) {
                unique.append(obs)
                known.insert(obs)
            }
            entity = GraphEntity(name: normalized, type: entityType,
                                 observations: unique, createdAt: Self.now())
            entities[key] = entity
        }
        pruneLocked()
        saveLocked()
        let result = entity
        lock.unlock()
        return result
    }

    @discardableResult
    public func deleteEntity(_ name: String) -> Bool {
        let key = name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !key.isEmpty else { return false }
        lock.lock()
        guard entities.removeValue(forKey: key) != nil else {
            lock.unlock()
            return false
        }
        relations.removeAll {
            $0.source.lowercased() == key || $0.target.lowercased() == key
        }
        vectors = vectors.filter { !$0.key.hasPrefix(key + "|") }
        saveLocked()
        lock.unlock()
        return true
    }

    public func getEntity(_ name: String) -> GraphEntity? {
        let key = name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !key.isEmpty else { return nil }
        lock.lock(); defer { lock.unlock() }
        return entities[key]
    }

    public func listEntities() -> [GraphEntity] {
        lock.lock(); defer { lock.unlock() }
        return entities.values.sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    }

    // MARK: - 观察

    /// 添加观察；实体不存在时自动创建。返回实际新增条数。
    @discardableResult
    public func addObservations(_ name: String, _ observations: [String]) -> Int {
        let normalized = name.trimmingCharacters(in: .whitespacesAndNewlines)
        let cleaned = observations
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !normalized.isEmpty, !cleaned.isEmpty else { return 0 }

        lock.lock()
        let key = normalized.lowercased()
        var entity: GraphEntity
        if let existing = entities[key] {
            entity = existing
        } else {
            entity = GraphEntity(name: normalized, type: "", observations: [],
                                 createdAt: Self.now())
        }
        var known = Set(entity.observations)
        var added = 0
        for obs in cleaned where !known.contains(obs) {
            entity.observations.append(obs)
            known.insert(obs)
            added += 1
        }
        entities[key] = entity
        pruneLocked()
        saveLocked()
        lock.unlock()
        return added
    }

    // MARK: - 关系

    /// 创建有向边；两端实体不存在时自动创建。重复边返回 false。
    @discardableResult
    public func createRelation(source: String, target: String, label: String) -> Bool {
        let src = source.trimmingCharacters(in: .whitespacesAndNewlines)
        let tgt = target.trimmingCharacters(in: .whitespacesAndNewlines)
        let lbl = label.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !src.isEmpty, !tgt.isEmpty, !lbl.isEmpty else { return false }

        lock.lock()
        for name in [src, tgt] where entities[name.lowercased()] == nil {
            entities[name.lowercased()] = GraphEntity(name: name, type: "",
                                                      observations: [], createdAt: Self.now())
        }
        let duplicate = relations.contains {
            $0.source.lowercased() == src.lowercased()
                && $0.target.lowercased() == tgt.lowercased()
                && $0.label.lowercased() == lbl.lowercased()
        }
        if duplicate {
            lock.unlock()
            return false
        }
        relations.append(GraphRelation(source: src, target: tgt, label: lbl))
        pruneLocked()
        saveLocked()
        lock.unlock()
        return true
    }

    public func listRelations() -> [GraphRelation] {
        lock.lock(); defer { lock.unlock() }
        return relations
    }

    /// 返回中心实体及其 `depth` 跳内的邻居（默认 1 跳）。
    public func neighbors(of name: String, depth: Int = 1) -> GraphNeighborhood {
        let key = name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !key.isEmpty else {
            return GraphNeighborhood(center: nil, entities: [], relations: [])
        }
        let limit = max(1, min(depth, 3))

        lock.lock(); defer { lock.unlock() }
        guard let center = entities[key] else {
            return GraphNeighborhood(center: nil, entities: [], relations: [])
        }
        var visited: Set<String> = [key]
        var frontier: Set<String> = [key]
        var collectedRelations: [GraphRelation] = []
        var seenRelations = Set<String>()

        for _ in 0..<limit {
            var next: Set<String> = []
            for relation in relations {
                let src = relation.source.lowercased()
                let tgt = relation.target.lowercased()
                guard frontier.contains(src) || frontier.contains(tgt) else { continue }
                let relKey = "\(src)|\(tgt)|\(relation.label.lowercased())"
                if seenRelations.insert(relKey).inserted {
                    collectedRelations.append(relation)
                }
                if !visited.contains(src) { next.insert(src) }
                if !visited.contains(tgt) { next.insert(tgt) }
            }
            if next.isEmpty { break }
            visited.formUnion(next)
            frontier = next
        }

        let found = visited.compactMap { entities[$0] }
            .sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
        return GraphNeighborhood(center: center, entities: found, relations: collectedRelations)
    }

    // MARK: - 检索

    /// 语义检索观察；向量不可用时回退为关键词子串匹配。
    public func search(query: String, topK: Int = 8) async -> [GraphObservationHit] {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { return [] }
        let limit = max(1, topK)

        let pairs: [(entity: String, text: String)] = {
            lock.lock(); defer { lock.unlock() }
            return entities.values
                .sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
                .flatMap { entity in
                    entity.observations.map { (entity: entity.name, text: $0) }
                }
        }()
        guard !pairs.isEmpty else { return [] }

        if let embedding, !isEmbeddingDisabled {
            do {
                try await ensureVectors(pairs, embedding: embedding)
                let queryVector = try await embedding.embedQuery(q)
                var hits: [GraphObservationHit] = []
                lock.lock()
                for pair in pairs {
                    guard let vector = vectors[Self.vectorKey(pair.entity, pair.text)] else { continue }
                    let score = Self.cosine(queryVector, vector)
                    if score >= 0.5 {
                        hits.append(GraphObservationHit(entity: pair.entity,
                                                        text: pair.text,
                                                        score: score))
                    }
                }
                lock.unlock()
                return Array(hits.sorted { $0.score > $1.score }.prefix(limit))
            } catch {
                AgentLog.warning("graph_memory: 语义检索不可用，回退关键词匹配: \(error.localizedDescription)")
                // 不永久禁用：模型可能随后加载成功（例如用户刚下载/切换模型），
                // 下次检索应重新尝试语义通道。
            }
        }
        return keywordSearch(pairs, query: q, topK: limit)
    }

    /// `search` 的语义化别名（任务契约中的 query）。
    public func query(_ text: String, topK: Int = 8) async -> [GraphObservationHit] {
        await search(query: text, topK: topK)
    }

    private func keywordSearch(_ pairs: [(entity: String, text: String)],
                               query: String, topK: Int) -> [GraphObservationHit] {
        let needle = query.lowercased()
        var hits: [GraphObservationHit] = []
        for pair in pairs where pair.text.lowercased().contains(needle) {
            hits.append(GraphObservationHit(entity: pair.entity, text: pair.text, score: 0.5))
            if hits.count >= topK { break }
        }
        return hits
    }

    private var isEmbeddingDisabled: Bool {
        lock.lock(); defer { lock.unlock() }
        return embeddingDisabled
    }

    private func ensureVectors(_ pairs: [(entity: String, text: String)],
                               embedding: EmbeddingProvider) async throws {
        lock.lock()
        var missingTexts: [String] = []
        var seen = Set<String>()
        for pair in pairs where vectors[Self.vectorKey(pair.entity, pair.text)] == nil {
            // 同一段观察文本可能挂在多个实体下，按文本去重只嵌入一次。
            if seen.insert(pair.text).inserted { missingTexts.append(pair.text) }
        }
        lock.unlock()
        guard !missingTexts.isEmpty else { return }

        var embeddedByText: [String: [Float]] = [:]
        var offset = 0
        while offset < missingTexts.count {
            try Task.checkCancellation()
            let batch = Array(missingTexts[offset..<min(offset + 32, missingTexts.count)])
            let embedded = try await embedding.embedDocuments(batch)
            guard embedded.count == batch.count else {
                throw AgentError.storage("graph_memory embedding 数量不一致")
            }
            for (text, vector) in zip(batch, embedded) { embeddedByText[text] = vector }
            offset += batch.count
        }
        lock.lock()
        for pair in pairs where vectors[Self.vectorKey(pair.entity, pair.text)] == nil {
            if let vector = embeddedByText[pair.text] {
                vectors[Self.vectorKey(pair.entity, pair.text)] = vector
            }
        }
        lock.unlock()
    }

    // MARK: - 摘要

    /// 用于 prompt 注入的紧凑摘要。
    public func snapshot(maxItems: Int = 10) -> String {
        lock.lock()
        let entityList = entities.values.sorted {
            if $0.observations.count != $1.observations.count {
                return $0.observations.count > $1.observations.count
            }
            return $0.name.localizedStandardCompare($1.name) == .orderedAscending
        }
        let relationList = relations
        lock.unlock()

        guard !entityList.isEmpty || !relationList.isEmpty else { return "" }
        var lines: [String] = []
        var shown = 0
        for entity in entityList {
            if shown >= maxItems { break }
            let observations = entity.observations
            guard !observations.isEmpty else { continue }
            let type = entity.type.isEmpty ? "" : " (\(entity.type))"
            lines.append("- \(entity.name)\(type): \(observations.prefix(3).joined(separator: "; "))")
            shown += 1
        }
        for relation in relationList.prefix(max(0, maxItems / 2)) {
            lines.append("- \(relation.source) --[\(relation.label)]--> \(relation.target)")
        }
        return lines.joined(separator: "\n")
    }

    // MARK: - 生命周期

    public func clear() {
        lock.lock()
        entities = [:]
        relations = []
        vectors = [:]
        saveLocked()
        lock.unlock()
    }

    /// 立即落盘（本实现每次修改都已落盘）。
    public func flush() {
        lock.lock(); defer { lock.unlock() }
        saveLocked(force: true)
    }

    /// 释放资源；本实现无外部句柄，仅确保落盘。
    public func close() {
        flush()
    }

    // MARK: - 裁剪

    private func pruneLocked() {
        var pruned = false

        // 1) 每实体观察上限：保留最新（先收集 key，避免边遍历边改字典）。
        let oversized = entities.compactMap { (key, entity) -> String? in
            entity.observations.count > maxObservationsPerEntity ? key : nil
        }
        if !oversized.isEmpty {
            pruned = true
            for key in oversized {
                guard var entity = entities[key] else { continue }
                let dropCount = entity.observations.count - maxObservationsPerEntity
                let dropped = entity.observations.prefix(dropCount)
                entity.observations.removeFirst(dropCount)
                entities[key] = entity
                for text in dropped {
                    vectors.removeValue(forKey: Self.vectorKey(entity.name, text))
                }
            }
        }
        // 2) 实体总数上限：按 created_at 淘汰最旧。
        if entities.count > maxEntities {
            pruned = true
            let ordered = entities.sorted {
                if $0.value.createdAt != $1.value.createdAt {
                    return $0.value.createdAt < $1.value.createdAt
                }
                return $0.key < $1.key
            }
            let excess = entities.count - maxEntities
            for (key, _) in ordered.prefix(excess) {
                entities.removeValue(forKey: key)
                vectors = vectors.filter { !$0.key.hasPrefix(key + "|") }
                relations.removeAll {
                    $0.source.lowercased() == key || $0.target.lowercased() == key
                }
            }
        }
        // 3) 关系上限：淘汰最旧。
        if relations.count > maxRelations {
            pruned = true
            relations.removeFirst(relations.count - maxRelations)
        }
        // 向量缓存只在确实裁剪过时重建存活集合：每次写入都全量扫描
        // 会让 upsert/addObservations 随着图谱变大而线性变慢。
        if pruned, !vectors.isEmpty {
            var alive = Set<String>()
            for entity in entities.values {
                for text in entity.observations {
                    alive.insert(Self.vectorKey(entity.name, text))
                }
            }
            vectors = vectors.filter { alive.contains($0.key) }
        }
    }

    // MARK: - 持久化

    private func load() {
        lock.lock(); defer { lock.unlock() }
        let fm = FileManager.default
        guard fm.fileExists(atPath: path),
              let raw = try? String(contentsOfFile: path, encoding: .utf8),
              let root = JSONValue.parse(raw)?.objectValue else {
            entities = [:]
            relations = []
            return
        }

        var loadedEntities: [String: GraphEntity] = [:]
        if let rawEntities = root["entities"]?.objectValue {
            for (key, value) in rawEntities {
                guard let object = value.objectValue else { continue }
                let name = (object["name"]?.stringValue ?? key)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                guard !name.isEmpty else { continue }
                let type = object["type"]?.stringValue ?? ""
                let observations = (object["observations"]?.arrayValue ?? [])
                    .compactMap(\.stringValue)
                    .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                    .filter { !$0.isEmpty }
                let createdAt = object["created_at"]?.stringValue ?? ""
                loadedEntities[name.lowercased()] = GraphEntity(
                    name: name, type: type, observations: observations,
                    createdAt: createdAt.isEmpty ? Self.now() : createdAt)
            }
        }

        var loadedRelations: [GraphRelation] = []
        if let rawRelations = root["relations"]?.arrayValue {
            for item in rawRelations {
                guard let object = item.objectValue else { continue }
                let source = (object["source"]?.stringValue ?? "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                let target = (object["target"]?.stringValue ?? "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                let label = (object["label"]?.stringValue ?? "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                guard !source.isEmpty, !target.isEmpty, !label.isEmpty else { continue }
                let created = object["created"]?.stringValue
                loadedRelations.append(GraphRelation(source: source, target: target,
                                                     label: label, created: created))
            }
        }

        entities = loadedEntities
        relations = loadedRelations
        pruneLocked()
    }

    /// 写入落盘。默认防抖 0.5s：事实抽取一次会写 5 个实体 + 3 条关系，
    /// 每次都全量序列化大 JSON 会造成明显写放大。`force=true` 用于 flush/close。
    private func saveLocked(force: Bool = false) {
        dirty = true
        guard !force else {
            pendingSave?.cancel()
            pendingSave = nil
            dirty = false
            writeLocked()
            return
        }
        guard pendingSave == nil else { return }
        let work = DispatchWorkItem { [weak self] in
            guard let self else { return }
            self.lock.lock()
            self.pendingSave = nil
            if self.dirty {
                self.dirty = false
                self.writeLocked()
            }
            self.lock.unlock()
        }
        pendingSave = work
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 0.5, execute: work)
    }

    private func writeLocked() {
        var entityObjects: [String: JSONValue] = [:]
        for (key, entity) in entities {
            var object: [String: JSONValue] = [
                "name": .string(entity.name),
                "type": .string(entity.type),
                "observations": .array(entity.observations.map { .string($0) }),
                "created_at": .string(entity.createdAt),
            ]
            if !entity.createdAt.isEmpty { object["created_at"] = .string(entity.createdAt) }
            entityObjects[key] = .object(object)
        }
        let relationValues: [JSONValue] = relations.map { relation in
            var object: [String: JSONValue] = [
                "source": .string(relation.source),
                "target": .string(relation.target),
                "label": .string(relation.label),
            ]
            if let created = relation.created { object["created"] = .string(created) }
            return .object(object)
        }
        let payload: JSONValue = .object([
            "entities": .object(entityObjects),
            "relations": .array(relationValues),
        ])
        do {
            let text = payload.jsonString(pretty: true) + "\n"
            try AtomicFile.write(Data(text.utf8), to: path)
        } catch {
            AgentLog.warning("graph_memory: 写入失败 \(path): \(error.localizedDescription)")
        }
    }

    // MARK: - 工具函数

    private static func now() -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
        return formatter.string(from: Date())
    }

    private static func vectorKey(_ entity: String, _ text: String) -> String {
        "\(entity.lowercased())|\(text)"
    }

    private static func cosine(_ a: [Float], _ b: [Float]) -> Double {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0..<a.count {
            let x = Double(a[i]), y = Double(b[i])
            dot += x * y
            na += x * x
            nb += y * y
        }
        let denom = na.squareRoot() * nb.squareRoot()
        return denom > 0 ? max(0, min(1, dot / denom)) : 0
    }
}
