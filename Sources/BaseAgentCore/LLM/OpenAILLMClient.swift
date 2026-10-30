import Foundation

/// OpenAI 兼容的流式聊天客户端。
///
/// 支持：
/// - 列出模型（GET /v1/models）
/// - 流式 chat completions，含 tool_calls 与 reasoning_content（deepseek 风格）
public final class OpenAILLMClient: LLMService, ClosableLLMService {
    public let baseURL: String
    public let apiKey: String
    public let model: String
    public let timeout: Double
    public let maxTokens: Int
    public let topP: Double
    public let minP: Double
    public let topK: Int
    public let repetitionPenalty: Double
    public let contextWindow: Int

    public var identifier: String { "openai:\(model)" }
    /// HTTP 请求彼此无共享会话状态，可并发提交。
    public var supportsConcurrentIsolatedRequests: Bool { true }

    /// 官方 OpenAI（含 Azure）会拒绝未声明的请求字段，且没有通用的「关闭思考」
    /// 参数；vLLM / llama.cpp / DashScope 等兼容端点支持 Qwen 模板的
    /// `enable_thinking`，此时才允许携带请求级 thinking 开关。
    var supportsTemplateThinkingControl: Bool {
        guard let host = URL(string: baseURL)?.host?.lowercased() else { return false }
        return !(host == "api.openai.com" || host.hasSuffix(".openai.azure.com"))
    }

    private let http: URLSession

    public init(baseURL: String,
                apiKey: String,
                model: String,
                timeout: Double = 120.0,
                maxTokens: Int = 32_768,
                topP: Double = 1.0,
                minP: Double = 0.0,
                topK: Int = 0,
                repetitionPenalty: Double = 1.0,
                contextWindow: Int = 32_768) {
        self.baseURL = baseURL.hasSuffix("/") ? String(baseURL.dropLast()) : baseURL
        self.apiKey = apiKey
        self.model = model
        self.timeout = timeout
        self.maxTokens = maxTokens
        self.topP = topP
        self.minP = minP
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
        self.contextWindow = contextWindow

        // 结构化超时：`read` 是 chunk 之间的最大间隔。一次性的 `timeout` 只约束
        // 整个请求；流式场景下服务器中途停顿会永久阻塞，必须给有限上界。
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = max(timeout, 60.0)
        configuration.timeoutIntervalForResource = 0
        configuration.waitsForConnectivity = false
        self.http = URLSession(configuration: configuration)
    }

    public convenience init(config: AgentConfig) {
        self.init(
            baseURL: config.baseURL,
            apiKey: config.apiKey,
            model: config.model,
            timeout: config.requestTimeout,
            maxTokens: config.maxTokens,
            topP: config.topP,
            minP: config.minP,
            topK: config.topK,
            repetitionPenalty: config.repetitionPenalty,
            contextWindow: config.maxContextTokens
        )
    }

    // MARK: - 请求头

    private func headers() -> [String: String] {
        var h = ["Content-Type": "application/json"]
        if !apiKey.isEmpty {
            h["Authorization"] = "Bearer \(apiKey)"
        }
        return h
    }

    private func makeURL(_ path: String) throws -> URL {
        guard let url = URL(string: baseURL + path) else {
            throw AgentError.llm("非法的 base_url: \(baseURL)")
        }
        return url
    }

    // MARK: - 列表模型

    public func listModels() async throws -> [String] {
        let url = try makeURL("/models")
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        for (key, value) in headers() {
            request.setValue(value, forHTTPHeaderField: key)
        }
        let (data, response) = try await http.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw AgentError.llm("list_models failed: 无 HTTP 响应")
        }
        if httpResponse.statusCode >= 400 {
            let body = String(data: data.prefix(200), encoding: .utf8) ?? ""
            throw AgentError.llm("list_models failed: HTTP \(httpResponse.statusCode) \(body)")
        }
        let parsed = JSONValue.parse(String(data: data, encoding: .utf8) ?? "")
        let items = parsed?["data"]?.arrayValue ?? parsed?.arrayValue ?? []
        let ids = items.compactMap { $0["id"]?.stringValue }.filter { !$0.isEmpty }
        return ids.sorted()
    }

    // MARK: - 流式聊天

    public func chatStream(messages: [ChatMessage],
                           tools: [JSONValue],
                           temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        chatStream(messages: messages, tools: tools, temperature: temperature,
                   options: .default)
    }

    public func chatStream(messages: [ChatMessage],
                           tools: [JSONValue],
                           temperature: Double,
                           options: LLMRequestOptions) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task { [weak self] in
                guard let self else {
                    continuation.finish()
                    return
                }
                do {
                    try await self.streamChat(messages: messages, tools: tools,
                                              temperature: temperature,
                                              options: options,
                                              continuation: continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func streamChat(messages: [ChatMessage],
                            tools: [JSONValue],
                            temperature: Double,
                            options: LLMRequestOptions,
                            continuation: AsyncThrowingStream<LLMStreamEvent, Error>.Continuation) async throws {
        let url = try makeURL("/chat/completions")

        // 双视图历史的第二视图（raw_content）只存在于本地 MLX 后端；Swift 的
        // ChatMessage 契约没有该字段，因此无需剥离。
        let wireMessages: [[String: Any]] = messages.map { message in
            var wire = message.wire()
            // assistant 带 tool_calls 且无文本时，Python 发的是 content=None。
            if message.role == .assistant,
               (wire["content"] as? String)?.isEmpty == true,
               !(message.toolCalls ?? []).isEmpty {
                wire["content"] = NSNull()
            }
            return wire
        }

        var payload: [String: JSONValue] = [
            "model": .string(model),
            "messages": .array(wireMessages.map(JSONValue.from)),
            "temperature": .double(temperature),
            "max_tokens": .int(maxTokens),
            "top_p": .double(topP),
            "stream": .bool(true),
            "stream_options": .object(["include_usage": .bool(true)]),
        ]
        if minP > 0 {
            payload["min_p"] = .double(minP)
        }
        if topK > 0 {
            payload["top_k"] = .int(topK)
        }
        if repetitionPenalty != 1.0 {
            payload["repetition_penalty"] = .double(repetitionPenalty)
        }
        if !tools.isEmpty {
            payload["tools"] = .array(tools)
        }
        if let enableThinking = options.enableThinking,
           supportsTemplateThinkingControl {
            // 不同兼容服务把 Qwen 模板开关放在不同位置：DashScope 等用顶层
            // `enable_thinking`，vLLM / llama.cpp 用 `chat_template_kwargs`；
            // 同时发送覆盖两类端点，服务端通常忽略不认识的字段。官方 OpenAI
            // 不接受这些字段，已在 supportsTemplateThinkingControl 中排除。
            payload["enable_thinking"] = .bool(enableThinking)
            payload["chat_template_kwargs"] = .object([
                "enable_thinking": .bool(enableThinking),
            ])
        }

        let body = try JSONEncoder().encode(JSONValue.object(payload))
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.httpBody = body
        for (key, value) in headers() {
            request.setValue(value, forHTTPHeaderField: key)
        }

        let (bytes, response) = try await http.bytes(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw AgentError.llm("chat_stream failed: 无 HTTP 响应")
        }
        if httpResponse.statusCode >= 400 {
            var text = ""
            for try await line in bytes.lines {
                text += line
                if text.count >= 300 { break }
            }
            throw AgentError.llm(
                "chat_stream failed: HTTP \(httpResponse.statusCode) \(String(text.prefix(300)))"
            )
        }

        var toolAcc: [Int: ToolAccumulator] = [:]
        var finalUsage: [String: Int] = [:]
        var nextToolSlot = 0
        var lastToolSlot: Int?
        var sseBuffer: [String] = []
        var streamDone = false

        func processEvent(_ obj: [String: JSONValue]) {
            if let usage = obj["usage"]?.objectValue {
                for (key, item) in usage {
                    if let number = item.intValue {
                        finalUsage[key] = number
                    }
                    // completion_tokens_details.reasoning_tokens 摊平成
                    // reasoning_tokens，供内核的 tok/s 统计使用。
                    if let details = item.objectValue,
                       let reasoning = details["reasoning_tokens"]?.intValue {
                        finalUsage["reasoning_tokens"] = reasoning
                    }
                }
            }

            guard let choice = obj["choices"]?.arrayValue?.first,
                  let delta = choice["delta"]?.objectValue else {
                return
            }

            if let reasoning = delta["reasoning_content"]?.stringValue, !reasoning.isEmpty {
                continuation.yield(LLMStreamEvent(kind: .reasoning, content: reasoning))
            }
            if let content = delta["content"]?.stringValue, !content.isEmpty {
                continuation.yield(LLMStreamEvent(kind: .content, content: content))
            }

            if let deltas = delta["tool_calls"]?.arrayValue {
                for tc in deltas {
                    let fn = tc["function"]?.objectValue
                    let id = tc["id"]?.stringValue ?? ""
                    let name = fn?["name"]?.stringValue ?? ""
                    let hasIdentity = !id.isEmpty || !name.isEmpty

                    // 少数兼容服务省略 index：
                    // - 带 id / name 的 delta 视为新调用，分配递增槽位；
                    // - 只有 arguments 的 delta 视为上一调用的参数片段。
                    let index: Int
                    if let explicit = tc["index"]?.intValue {
                        index = explicit
                        nextToolSlot = max(nextToolSlot, explicit + 1)
                    } else if hasIdentity {
                        index = nextToolSlot
                        nextToolSlot += 1
                    } else {
                        index = lastToolSlot ?? 0
                    }
                    lastToolSlot = index

                    var slot = toolAcc[index]
                        ?? ToolAccumulator(id: "", type: "function", name: "", arguments: "")
                    if !id.isEmpty {
                        slot.id = id
                    }
                    if let type = tc["type"]?.stringValue, !type.isEmpty {
                        slot.type = type
                    }
                    if !name.isEmpty {
                        // 有的服务按 fragment 发 name；拼接与 Python 一致。
                        slot.name += name
                    }
                    if let args = fn?["arguments"]?.stringValue {
                        slot.arguments += args
                    }
                    toolAcc[index] = slot
                }
            }
        }

        /// SSE 的一个事件可以由多行 `data:` 组成（规范用空行分隔）。
        /// 必须在空行处再解析，逐行解析会丢掉跨行拼接的 JSON。
        func flushEvent() {
            guard !sseBuffer.isEmpty else { return }
            let payload = sseBuffer.joined(separator: "\n")
            sseBuffer.removeAll(keepingCapacity: true)
            guard payload != "[DONE]",
                  let object = JSONValue.parse(payload)?.objectValue else { return }
            processEvent(object)
        }

        for try await rawLine in bytes.lines {
            if Task.isCancelled { throw CancellationError() }
            // 空行 = 事件边界；不能先 trim 再判断，否则整行丢失。
            if rawLine.isEmpty {
                flushEvent()
                continue
            }
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            if line.isEmpty {
                flushEvent()
                continue
            }
            // SSE 注释/心跳行。
            if line.hasPrefix(":") { continue }
            guard line.hasPrefix("data:") else { continue }
            let payload = String(line.dropFirst("data:".count))
                .trimmingCharacters(in: .whitespaces)
            if payload == "[DONE]" {
                flushEvent()
                streamDone = true
                break
            }
            // 兼容缺少空行分隔的服务：若缓冲已经是完整 JSON，先处理它。
            if !sseBuffer.isEmpty,
               JSONValue.parse(sseBuffer.joined(separator: "\n")) != nil {
                flushEvent()
            }
            sseBuffer.append(payload)
        }
        if !streamDone { flushEvent() }

        // sorted() 要求 key 可比：index 已在上面统一成 Int。
        let toolCalls = toolAcc.keys.sorted().map { index -> ToolCall in
            let slot = toolAcc[index]!
            var arguments: [String: JSONValue] = [:]
            if !slot.arguments.isEmpty {
                if let parsed = JSONValue.parse(slot.arguments), let obj = parsed.objectValue {
                    arguments = obj
                } else if JSONValue.parse(slot.arguments) != nil {
                    // 合法 JSON 但不是对象 → 与 Python 一致，按空参数处理。
                    arguments = [:]
                } else {
                    // 解析失败必须报回模型，静默当 {} 会变成莫名的「缺少必填参数」。
                    arguments = [
                        "__parse_error__": .string("Invalid JSON in tool arguments"),
                        "__raw__": .string(slot.arguments),
                    ]
                }
            }
            return ToolCall(id: slot.id, name: slot.name,
                            arguments: arguments, rawArguments: slot.arguments)
        }
        continuation.yield(LLMStreamEvent(kind: .done, toolCalls: toolCalls,
                                          usage: finalUsage, rawContent: ""))
    }

    /// 释放网络资源。
    public func close() {
        http.invalidateAndCancel()
    }

    private struct ToolAccumulator {
        var id: String
        var type: String
        var name: String
        var arguments: String
    }
}
