import Foundation

/// 从模型输出文本中提取工具调用。
///
/// 兼容三类常见格式：
/// - `<tool_call>{"name":"...","arguments":{...}}</tool_call>`
/// - Qwen 的 `<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`
/// - 纯 JSON / JSON 围栏
///
/// 只接受请求里声明过的工具名，避免普通回答中的 JSON 被误判成工具调用。
enum LocalToolCallParser {
    static func extract(raw: String, allowedNames: Set<String>) -> [ToolCall] {
        guard !raw.isEmpty, !allowedNames.isEmpty else { return [] }
        var payloads: [String] = []
        var seen = Set<String>()

        func collect(_ pattern: String, group: Int = 1) {
            guard let regex = try? NSRegularExpression(
                pattern: pattern, options: [.dotMatchesLineSeparators]) else { return }
            let range = NSRange(raw.startIndex..<raw.endIndex, in: raw)
            for match in regex.matches(in: raw, options: [], range: range) {
                guard match.numberOfRanges > group,
                      let matchRange = Range(match.range(at: group), in: raw) else { continue }
                let text = String(raw[matchRange])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !text.isEmpty, seen.insert(text).inserted {
                    payloads.append(text)
                }
            }
        }

        collect("<tool_call>\\s*([\\s\\S]*?)\\s*</tool_call>")
        collect("```(?:json)?\\s*([\\s\\S]*?)```")
        if payloads.isEmpty, let value = JSONValue.parse(raw),
           value.objectValue != nil || value.arrayValue != nil {
            payloads.append(raw)
        }

        var calls: [ToolCall] = []
        for payload in payloads {
            if let value = JSONValue.parse(payload) {
                calls.append(contentsOf: callsFromJSON(value, allowedNames: allowedNames))
                continue
            }
            calls.append(contentsOf: callsFromXML(payload, allowedNames: allowedNames))
        }
        return calls
    }

    // MARK: - JSON

    private static func callsFromJSON(_ value: JSONValue,
                                      allowedNames: Set<String>) -> [ToolCall] {
        if let array = value.arrayValue {
            return array.flatMap { callsFromJSON($0, allowedNames: allowedNames) }
        }
        guard let object = value.objectValue else { return [] }
        // 有些模型输出 {"tool_calls":[...]}。
        if let nested = object["tool_calls"]?.arrayValue {
            return nested.flatMap { callsFromJSON($0, allowedNames: allowedNames) }
        }

        let function = object["function"]?.objectValue
        let name = (object["name"]?.stringValue
            ?? object["tool"]?.stringValue
            ?? function?["name"]?.stringValue ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty, allowedNames.contains(name) else { return [] }

        var arguments: [String: JSONValue] = [:]
        if let rawArgs = object["arguments"] ?? object["parameters"] ?? function?["arguments"] {
            if let text = rawArgs.stringValue, let parsed = JSONValue.parse(text)?.objectValue {
                arguments = parsed
            } else if let parsed = rawArgs.objectValue {
                arguments = parsed
            }
        }
        let id = object["id"]?.stringValue
            ?? "call_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(12))"
        return [ToolCall(id: id, name: name, arguments: arguments,
                         rawArguments: JSONValue.object(arguments).jsonString())]
    }

    // MARK: - Qwen XML

    private static func callsFromXML(_ payload: String,
                                     allowedNames: Set<String>) -> [ToolCall] {
        guard let functionRange = payload.range(
            of: #"<function(?:=|\s+name=)["']?([^"'>\s]+)"#, options: .regularExpression)
        else { return [] }
        let functionText = String(payload[functionRange])
        let name = functionText
            .replacingOccurrences(of: #"<function(?:=|\s+name=)["']?"#,
                                  with: "", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "<>\"' "))
        guard !name.isEmpty, allowedNames.contains(name) else { return [] }

        var arguments: [String: JSONValue] = [:]
        let pattern = #"<parameter(?:=|\s+name=)["']?([^"'>\s]+)["']?>\s*([\s\S]*?)\s*</parameter>"#
        if let regex = try? NSRegularExpression(pattern: pattern) {
            let range = NSRange(payload.startIndex..<payload.endIndex, in: payload)
            for match in regex.matches(in: payload, options: [], range: range) {
                guard match.numberOfRanges > 2,
                      let keyRange = Range(match.range(at: 1), in: payload),
                      let valueRange = Range(match.range(at: 2), in: payload) else { continue }
                let key = String(payload[keyRange]).trimmingCharacters(in: .whitespacesAndNewlines)
                let value = String(payload[valueRange])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                guard !key.isEmpty else { continue }
                arguments[key] = parseScalar(value)
            }
        }
        let id = "call_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(12))"
        return [ToolCall(id: id, name: name, arguments: arguments,
                         rawArguments: JSONValue.object(arguments).jsonString())]
    }

    /// XML 参数没有类型标注：能按 JSON 解析就保留类型，否则当字符串。
    private static func parseScalar(_ value: String) -> JSONValue {
        if let parsed = JSONValue.parse(value) {
            switch parsed {
            case .object, .array, .bool, .int, .double, .null:
                return parsed
            case .string(let text):
                return .string(text)
            }
        }
        if value == "true" { return .bool(true) }
        if value == "false" { return .bool(false) }
        if value == "null" { return .null }
        return .string(value)
    }
}
