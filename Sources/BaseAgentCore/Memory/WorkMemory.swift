import Foundation

/// 工作记忆（任务内 scratchpad）。
///
/// 文件位于 `workspace/.agent/work_memory.json`，内容为扁平的
/// `{key: string}` JSON，与 Python 版 `WorkMemory` 完全兼容。
/// 所有修改都在锁内完成并原子写盘，避免并发写坏文件。
public final class WorkMemory {
    /// 存储文件绝对路径。
    public let path: String

    private let lock = NSLock()
    private var data: [String: String] = [:]

    public init(path: String) {
        self.path = path
        load()
    }

    /// 便捷构造：`<workspace>/.agent/work_memory.json`。
    public convenience init(workspace: String) {
        self.init(path: (workspace as NSString)
            .appendingPathComponent(".agent")
            .appending("/work_memory.json"))
    }

    // MARK: - 读取

    public func get(_ key: String) -> String? {
        lock.lock(); defer { lock.unlock() }
        return data[key]
    }

    public func list() -> [String: String] {
        lock.lock(); defer { lock.unlock() }
        return data
    }

    // MARK: - 写入

    public func set(_ key: String, _ value: String) {
        lock.lock(); defer { lock.unlock() }
        data[key] = value
        saveLocked()
    }

    public func delete(_ key: String) {
        lock.lock(); defer { lock.unlock() }
        data.removeValue(forKey: key)
        saveLocked()
    }

    public func clear() {
        lock.lock(); defer { lock.unlock() }
        data = [:]
        saveLocked()
    }

    /// 立即落盘（本实现每次写入都已落盘，保留该方法仅为对齐 Python API）。
    public func flush() {
        lock.lock(); defer { lock.unlock() }
        saveLocked()
    }

    /// 从磁盘重新读取，丢弃内存副本。
    public func reload() {
        lock.lock(); defer { lock.unlock() }
        data = [:]
        loadLocked()
    }

    // MARK: - 持久化

    private func load() {
        lock.lock(); defer { lock.unlock() }
        loadLocked()
    }

    private func loadLocked() {
        let fm = FileManager.default
        guard fm.fileExists(atPath: path),
              let raw = try? String(contentsOfFile: path, encoding: .utf8),
              !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            data = [:]
            return
        }
        guard let root = JSONValue.parse(raw)?.objectValue else {
            backupCorrupt()
            data = [:]
            return
        }
        var parsed: [String: String] = [:]
        for (key, value) in root {
            if let s = value.stringValue {
                parsed[key] = s
            }
        }
        data = parsed
    }

    private func saveLocked() {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        guard let encoded = try? encoder.encode(data) else { return }
        do {
            try AtomicFile.write(encoded, to: path)
        } catch {
            AgentLog.warning("work_memory: 写入失败 \(path): \(error.localizedDescription)")
        }
    }

    private func backupCorrupt() {
        let stamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "")
        let backup = "\(path).corrupt-\(stamp)"
        try? FileManager.default.moveItem(atPath: path, toPath: backup)
        AgentLog.warning("work_memory: 文件损坏，已备份到 \(backup)")
    }
}
