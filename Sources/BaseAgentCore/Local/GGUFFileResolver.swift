import Foundation

/// GGUF 文件发现。接受 `.gguf` 文件、包含 `.gguf` 的目录，或模型名。
enum GGUFFileResolver {
    static func resolve(kind: LocalModelKind, value: String, config: AgentConfig) throws -> URL {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw LocalModelError.notConfigured(
                kind, hint: "请在配置中填写 \(kind.displayName) 的 GGUF 路径")
        }
        let expanded = (trimmed as NSString).expandingTildeInPath
        let fm = FileManager.default

        if fm.fileExists(atPath: expanded), let file = ggufFile(atPath: expanded) {
            return file
        }

        let candidates = [
            (config.modelsDirectory as NSString).appendingPathComponent(expanded),
            (NSHomeDirectory() as NSString).appendingPathComponent("base-agent/models/" + expanded),
            (NSHomeDirectory() as NSString).appendingPathComponent("models/" + expanded),
            (NSHomeDirectory() as NSString).appendingPathComponent(".cache/huggingface/hub/" + expanded),
            (NSHomeDirectory() as NSString).appendingPathComponent("models/" + Self.stripModelsPrefix(expanded)),
        ]
        for path in candidates where fm.fileExists(atPath: path) {
            if let file = ggufFile(atPath: path) { return file }
        }
        throw LocalModelError.notFound(
            kind, value: trimmed, searched: candidates,
            hint: "请把 GGUF 文件放到 \(config.modelsDirectory)/<name>，或填写完整的 .gguf 路径。")
    }

    private static func stripModelsPrefix(_ path: String) -> String {
        path.hasPrefix("models/") ? String(path.dropFirst("models/".count)) : path
    }

    static func ggufFile(atPath path: String) -> URL? {
        let fm = FileManager.default
        var isDirectory: ObjCBool = false
        guard fm.fileExists(atPath: path, isDirectory: &isDirectory) else { return nil }
        if !isDirectory.boolValue {
            return path.lowercased().hasSuffix(".gguf")
                ? URL(fileURLWithPath: path) : nil
        }
        let url = URL(fileURLWithPath: path)
        let contents = (try? fm.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)) ?? []
        let files = contents.filter { $0.pathExtension.lowercased() == "gguf" }
        let main = files.filter { !$0.lastPathComponent.lowercased().contains("mmproj") }
        guard !main.isEmpty else { return nil }
        let preferences = ["q4_k_m", "q4_k_s", "q5_k_m", "q6_k", "q8_0", "q4_0"]
        for preference in preferences {
            if let match = main.first(where: {
                $0.lastPathComponent.lowercased().contains(preference)
            }) {
                return match
            }
        }
        return main.sorted { $0.lastPathComponent < $1.lastPathComponent }.first
    }

    static func resolveMMProj(modelPath: String, explicit: String?) -> String? {
        if let explicit, !explicit.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return (explicit as NSString).expandingTildeInPath
        }
        let directory = URL(fileURLWithPath: modelPath).deletingLastPathComponent()
        let contents = (try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)) ?? []
        return contents.first {
            $0.pathExtension.lowercased() == "gguf"
                && $0.lastPathComponent.lowercased().contains("mmproj")
        }?.path
    }
}
