import Foundation

// MARK: - 检查点状态文件

/// 工作记忆检查点的最小可行落盘实现。
///
/// Python 版把 checkpoint 写进内核 `Session.working`，跨会话部分交给
/// work memory。Swift 工具层不直接依赖 Memory 层，因此统一写入
/// `<workspace>/.agent/working_checkpoint.json`：文件天然跨会话存活，
/// 同时也能被 `start_long_term_update` 读取并作为蒸馏材料。
enum CheckpointStore {
    static let maxKeyInfo = 4_000

    static func path(_ config: AgentConfig) -> String {
        ToolSupport.stateFile("working_checkpoint.json", workspace: config.workspace)
    }

    static func read(_ config: AgentConfig) -> String {
        readRecord(config)?.text ?? ""
    }

    /// 仅返回标记了 `persist=true` 的跨会话检查点。
    static func readPersisted(_ config: AgentConfig) -> String {
        guard let record = readRecord(config), record.persist else { return "" }
        return record.text
    }

    static func readRecord(_ config: AgentConfig) -> (text: String, persist: Bool)? {
        guard let data = FileManager.default.contents(atPath: path(config)),
              let raw = try? JSONDecoder().decode(JSONValue.self, from: data),
              let text = raw["key_info"]?.stringValue else {
            return nil
        }
        return (text, raw["persist"]?.boolValue ?? false)
    }

    static func write(_ text: String, persist: Bool, config: AgentConfig) throws {
        let payload = JSONValue.object([
            "key_info": .string(text),
            "persist": .bool(persist),
            "updated_at": .string(ISO8601DateFormatter().string(from: Date())),
        ])
        try AtomicFile.write(payload.jsonString(pretty: true), to: path(config))
    }

    static func clear(_ config: AgentConfig) throws {
        let file = path(config)
        guard FileManager.default.fileExists(atPath: file) else { return }
        try FileManager.default.removeItem(atPath: file)
    }
}

// MARK: - update_working_checkpoint

/// `update_working_checkpoint`：短期工作记忆便签。
///
/// 只记「非显然、后续必须知道」的状态（已验证的路径/ID/端口、决策、已排除
/// 的方案），不记流水账。空字符串表示清空。
public final class UpdateWorkingCheckpointTool: AgentTool {
    public let name = "update_working_checkpoint"
    public let description =
        "Short-term working notepad, auto-injected into every later turn. "
        + "Call early/mid task ONLY when there is non-obvious state worth keeping: "
        + "verified paths/IDs/ports, decisions, approaches already ruled out. "
        + "Do NOT store transcripts or facts re-derivable in one tool call. "
        + "Empty key_info clears it."

    private let config: AgentConfig
    private let callbacks: AgentCallbacks?
    private weak var registry: ToolRegistry?

    public var parameters: JSONValue {
        ToolSupport.objectSchema(
            properties: [
                "key_info": ToolSupport.stringProperty(
                    "Compact notes (<=\(CheckpointStore.maxKeyInfo) chars). "
                    + "Empty string clears the notepad."),
                "persist": ToolSupport.boolProperty(
                    "Also persist across sessions (default false)."),
            ],
            required: ["key_info"])
    }

    public init(config: AgentConfig, callbacks: AgentCallbacks?, registry: ToolRegistry? = nil) {
        self.config = config
        self.callbacks = callbacks
        self.registry = registry
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        do {
            let keyInfo = try ToolSupport.requireString(args, "key_info")
            let persist = ToolSupport.optionalBool(args, "persist") ?? false
            let text = String(keyInfo
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .prefix(CheckpointStore.maxKeyInfo))

            guard let session = registry?.session else {
                return .fail("working checkpoint unavailable (no kernel session bound)")
            }
            session.setKeyInfo(text)

            if persist {
                try CheckpointStore.write(text, persist: true, config: config)
            }
            if text.isEmpty {
                try CheckpointStore.clear(config)
                return .ok("working checkpoint cleared")
            }
            let hint = session.currentTurn <= 2
                ? " (tip: skip re-calling unless new non-obvious facts appear)" : ""
            return .ok("working checkpoint updated (\(text.count) chars)\(hint)")
        } catch {
            return ToolSupport.failure(error)
        }
    }
}

// MARK: - start_long_term_update

/// `start_long_term_update`：触发 L4 经验蒸馏。
///
/// 不直接改长期记忆文件，只把「本会话检查点 + 待提纯材料」返回给模型，
/// 并向内核发出 distill 标记（内核按工具名 `start_long_term_update` 识别）。
public final class StartLongTermUpdateTool: AgentTool {
    public let name = "start_long_term_update"
    public let description =
        "After finishing a task or reaching a checkpoint, distill durable experience "
        + "into long-term memory. Only for verified facts, user preferences and lessons "
        + "that are expensive to rediscover. Then (optionally) call memory_graph to add "
        + "the distilled entities, or skill_save to freeze a reusable procedure. "
        + "Skip if nothing durable was learned."

    private let config: AgentConfig
    private let callbacks: AgentCallbacks?

    public var parameters: JSONValue {
        ToolSupport.objectSchema(
            properties: [
                "focus": ToolSupport.stringProperty(
                    "What to distill, e.g. 'python env setup' (optional)."),
            ])
    }

    public init(config: AgentConfig, callbacks: AgentCallbacks?, registry: ToolRegistry? = nil) {
        self.config = config
        self.callbacks = callbacks
        _ = registry
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        let focus = ToolSupport.optionalString(args, "focus") ?? ""
        var transcript = CheckpointStore.read(config)
        if !transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            transcript = "[checkpoint]\n\(transcript)"
        } else {
            transcript = "(当前会话尚无有效记录)"
        }

        let payload = JSONValue.object([
            "status": .string("DISTILL_REQUESTED"),
            "focus": .string(focus),
            "transcript": .string(String(transcript.suffix(3_000))),
            "next": .array([
                .string("memory_graph(op=create_entity/add_observations) 写入验证过的事实与偏好"),
                .string("skill_save(...) 若该流程可复用，固化为 SKILL.md"),
                .string("rag_ingest() 若有外部语料需要灌入本地知识库"),
            ]),
            "rules": .array([
                .string("只写经工具验证成功的信息；禁止写入推测、临时状态、通用常识"),
                .string("合并重复、压缩冗述，不堆叠流水账"),
                .string("无提纯价值则跳过，不要为了调用而调用"),
            ]),
        ])
        return .ok(compactJSON(payload, maxChars: 2_600))
    }
}
