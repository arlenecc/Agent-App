import Foundation

/// 把 MCP 服务暴露的工具包装成 `AgentTool`。
///
/// 默认工具名为 `mcp_<server>_<tool>`，避免多个 MCP 服务之间重名；
/// `prefixName=false` 时沿用原始工具名（与 Python 版一致）。
public final class MCPTool: AgentTool, ShutdownableTool {
    public let name: String
    public let description: String
    public let parameters: JSONValue
    public let destructive: Bool

    private let client: MCPClient
    private let toolName: String
    private let required: [String]
    private let propertyNames: Set<String>

    /// 从 `tools/list` 返回的工具定义构造；名称为空时返回 nil。
    public init?(client: MCPClient,
                 definition: JSONValue,
                 serverName: String? = nil,
                 prefixName: Bool = true) {
        guard let object = definition.objectValue,
              let rawName = object["name"]?.stringValue?
                .trimmingCharacters(in: .whitespacesAndNewlines),
              !rawName.isEmpty else {
            return nil
        }

        let server = Self.sanitize(serverName ?? client.name)
        if prefixName && !server.isEmpty {
            self.name = "mcp_\(server)_\(Self.sanitize(rawName))"
        } else {
            self.name = rawName
        }
        self.toolName = rawName

        let rawDescription = object["description"]?.stringValue?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        self.description = rawDescription.isEmpty
            ? "MCP 工具 \(rawName)（服务 \(server)）"
            : rawDescription

        let schema = object["inputSchema"]?.objectValue
        let properties = schema?["properties"]?.objectValue ?? [:]
        let required = (schema?["required"]?.arrayValue ?? []).compactMap(\.stringValue)
        self.required = required
        self.propertyNames = Set(properties.keys)
        let rawType = schema?["type"]?.stringValue ?? ""
        self.parameters = .object([
            "type": .string(rawType.isEmpty ? "object" : rawType),
            "properties": .object(properties),
            "required": .array(required.map { .string($0) }),
        ])

        // MCP 工具是外部能力，默认需要确认（与 Python 版一致）。
        self.destructive = true
        self.client = client
    }

    /// 从已解析的字典定义构造。
    public convenience init?(client: MCPClient,
                            definition: [String: JSONValue],
                            serverName: String? = nil,
                            prefixName: Bool = true) {
        self.init(client: client,
                  definition: .object(definition),
                  serverName: serverName,
                  prefixName: prefixName)
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        let missing = required.filter { args[$0] == nil }
        guard missing.isEmpty else {
            return .fail("Invalid arguments for \(name): missing required argument(s): \(missing.joined(separator: ", "))")
        }
        if !propertyNames.isEmpty {
            let unknown = args.keys.filter { !propertyNames.contains($0) }.sorted()
            guard unknown.isEmpty else {
                return .fail("Invalid arguments for \(name): unknown argument(s): \(unknown.joined(separator: ", "))")
            }
        }
        do {
            let output = try await client.callTool(name: toolName, arguments: args)
            return .ok(output)
        } catch {
            return .fail(error.localizedDescription)
        }
    }

    public func shutdown() {
        client.stop()
    }

    private static func sanitize(_ raw: String) -> String {
        var out = ""
        var lastUnderscore = false
        for scalar in raw.unicodeScalars {
            let value = scalar.value
            let isAllowed = (value >= 48 && value <= 57)
                || (value >= 65 && value <= 90)
                || (value >= 97 && value <= 122)
                || scalar == "_" || scalar == "-"
            if isAllowed {
                out.unicodeScalars.append(scalar)
                lastUnderscore = false
            } else if !lastUnderscore {
                out.append("_")
                lastUnderscore = true
            }
        }
        return out.trimmingCharacters(in: CharacterSet(charactersIn: "_-"))
    }
}
