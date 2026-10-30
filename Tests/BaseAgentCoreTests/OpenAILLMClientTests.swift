import XCTest
import Network
@testable import BaseAgentCore

/// 本地 OpenAI 兼容 server：记录 `/chat/completions` 请求体，并返回固定 SSE 流。
/// 用于验证请求级 `LLMRequestOptions` 是否真正进入 HTTP payload。
private final class RecordingSSEServer: @unchecked Sendable {
    private let listener: NWListener
    private let queue = DispatchQueue(label: "openai-client-test-server")
    private let lock = NSLock()
    private var lastBodyStorage: [String: JSONValue]?
    private let responseData: Data

    init(sse: String) throws {
        listener = try NWListener(using: NWParameters.tcp, on: .any)
        responseData = Data(sse.utf8)
        let queue = self.queue
        listener.newConnectionHandler = { [weak self] connection in
            guard let self else { return }
            connection.start(queue: queue)
            self.readRequest(connection, buffer: Data())
        }
    }

    var baseURL: String {
        "http://127.0.0.1:\(listener.port?.rawValue ?? 0)/v1"
    }

    var lastBody: [String: JSONValue]? {
        lock.lock()
        defer { lock.unlock() }
        return lastBodyStorage
    }

    func start() {
        listener.start(queue: queue)
    }

    func stop() {
        listener.cancel()
    }

    private func readRequest(_ connection: NWConnection, buffer: Data) {
        connection.receive(minimumIncompleteLength: 1,
                           maximumLength: 128 * 1024) { [weak self] data, _, isComplete, error in
            guard let self else {
                connection.cancel()
                return
            }
            var buffer = buffer
            if let data { buffer.append(data) }
            if let body = Self.completeBody(in: buffer) {
                self.store(body: body)
                self.respond(connection)
                return
            }
            if error != nil || isComplete {
                connection.cancel()
                return
            }
            self.readRequest(connection, buffer: buffer)
        }
    }

    /// 头部完整且 Content-Length 满足后才返回 body，避免只读到半截 JSON。
    private static func completeBody(in data: Data) -> Data? {
        guard let headerEnd = data.range(of: Data("\r\n\r\n".utf8)) else { return nil }
        let header = String(decoding: data[..<headerEnd.lowerBound], as: UTF8.self)
        let length = header.split(separator: "\r\n")
            .first { $0.lowercased().hasPrefix("content-length:") }
            .flatMap { line -> Int? in
                let value = line.split(separator: ":", maxSplits: 1)
                return Int(value.last?.trimmingCharacters(in: .whitespaces) ?? "")
            } ?? 0
        let body = data[headerEnd.upperBound...]
        guard body.count >= length else { return nil }
        return Data(body.prefix(length))
    }

    private func store(body: Data) {
        guard let parsed = JSONValue.parse(String(decoding: body, as: UTF8.self))?.objectValue
        else { return }
        lock.lock()
        lastBodyStorage = parsed
        lock.unlock()
    }

    private func respond(_ connection: NWConnection) {
        let header = "HTTP/1.1 200 OK\r\n"
            + "Content-Type: text/event-stream; charset=utf-8\r\n"
            + "Cache-Control: no-cache\r\n"
            + "Connection: close\r\n\r\n"
        connection.send(content: Data(header.utf8) + responseData,
                        completion: .contentProcessed { _ in connection.cancel() })
    }
}

final class OpenAILLMClientTests: XCTestCase {
    private var server: RecordingSSEServer?

    override func tearDown() {
        server?.stop()
        server = nil
        super.tearDown()
    }

    private func makeClient(baseURL: String) -> OpenAILLMClient {
        OpenAILLMClient(baseURL: baseURL, apiKey: "test-key",
                        model: "qwen3-test", timeout: 10,
                        maxTokens: 64, contextWindow: 4_096)
    }

    private func startServer() throws -> (RecordingSSEServer, OpenAILLMClient) {
        let server = try RecordingSSEServer(sse: """
        data: {"choices":[{"delta":{"content":"摘要内容"}}]}

        data: {"choices":[{"delta":{}}],"usage":{"total_tokens":9}}

        data: [DONE]

        """)
        server.start()
        Thread.sleep(forTimeInterval: 0.2)
        self.server = server
        return (server, makeClient(baseURL: server.baseURL))
    }

    private func drain(_ client: OpenAILLMClient,
                       options: LLMRequestOptions) async throws -> String {
        var content = ""
        for try await event in client.chatStream(
            messages: [ChatMessage(role: .user, content: "请总结这段正文")],
            tools: [], temperature: 0, options: options) {
            if event.kind == .content { content += event.content }
        }
        return content
    }

    /// 摘要等辅助任务显式要求关闭 thinking 时，远端兼容 API 必须收到开关；
    /// 只实现三参数 `chatStream` 会被协议默认实现静默丢弃。
    func testThinkingOverrideIsSentInRequestPayload() async throws {
        let (server, client) = try startServer()
        let content = try await drain(client, options: LLMRequestOptions(
            enableThinking: false, isolated: true))
        XCTAssertEqual(content, "摘要内容")

        let body = try XCTUnwrap(server.lastBody)
        XCTAssertEqual(body["enable_thinking"]?.boolValue, false)
        XCTAssertEqual(body["chat_template_kwargs"]?["enable_thinking"]?.boolValue, false)
    }

    /// 未显式覆盖时不能改变主对话请求：OpenAI 官方端点会拒绝未声明字段。
    func testDefaultRequestDoesNotSendThinkingControls() async throws {
        let (server, client) = try startServer()
        _ = try await drain(client, options: .default)

        let body = try XCTUnwrap(server.lastBody)
        XCTAssertNil(body["enable_thinking"])
        XCTAssertNil(body["chat_template_kwargs"])
    }

    /// 官方 OpenAI / Azure 不接受模板 thinking 字段；兼容端点才发送。
    func testTemplateThinkingControlIsLimitedToCompatibleEndpoints() {
        XCTAssertFalse(makeClient(baseURL: "https://api.openai.com/v1")
            .supportsTemplateThinkingControl)
        XCTAssertFalse(makeClient(baseURL: "https://demo.openai.azure.com/openai/v1")
            .supportsTemplateThinkingControl)
        XCTAssertTrue(makeClient(baseURL: "http://127.0.0.1:8000/v1")
            .supportsTemplateThinkingControl)
        XCTAssertTrue(makeClient(baseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1")
            .supportsTemplateThinkingControl)
    }
}
