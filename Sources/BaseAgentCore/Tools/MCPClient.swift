import Darwin
import Foundation

// MARK: - 错误

/// MCP 客户端 / 协议错误。
public struct MCPError: Error, LocalizedError, Sendable {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var errorDescription: String? { message }
}

// MARK: - JSON-RPC 响应

public struct MCPResponse: Sendable {
    public var id: Int
    public var result: JSONValue?
    public var errorCode: Int?
    public var errorMessage: String?

    public init(id: Int, result: JSONValue? = nil,
                errorCode: Int? = nil, errorMessage: String? = nil) {
        self.id = id
        self.result = result
        self.errorCode = errorCode
        self.errorMessage = errorMessage
    }
}

// MARK: - stdio JSON-RPC 客户端

/// 基于 stdio 的 MCP（Model Context Protocol）客户端。
///
/// 消息为换行分隔的 JSON-RPC 2.0：
/// - 启动时完成 `initialize` 握手并发送 `notifications/initialized`；
/// - `tools/list` 拉取工具清单，`tools/call` 调用工具；
/// - 后台线程读取 stdout，按 `id` 匹配请求并支持超时；
/// - 处理服务端通知（`notifications/*`）与反向请求（统一返回 -32601）；
/// - `stop()` 终止进程，SIGTERM 后 5 秒升级为 SIGKILL。
public final class MCPClient: @unchecked Sendable {
    public let name: String
    public let command: String
    public let args: [String]
    public let env: [String: String]
    public let timeout: TimeInterval
    public let maxOutputChars: Int

    private final class PendingRequest {
        var continuation: CheckedContinuation<MCPResponse, Error>?
        var result: Result<MCPResponse, Error>?
        var completed = false
    }

    private var process = Process()
    private var stdinPipe = Pipe()
    private var stdoutPipe = Pipe()
    private var stderrPipe = Pipe()
    private let lock = NSRecursiveLock()
    private let writeLock = NSLock()
    private let writeQueue = DispatchQueue(label: "baseagent.mcp.write")
    private let toolCallTimeout: TimeInterval = 300

    private final class WriteOutcome {
        var error: Error?
    }

    private var running = false
    private var startInProgress = false
    /// shutdown() 之后永久关闭：callTool 不允许再把它「复活」。
    private var closed = false
    private var nextID = 1
    private var pending: [Int: PendingRequest] = [:]
    private var receiveBuffer = Data()
    private var serverInfo: [String: JSONValue] = [:]
    private var tools: [JSONValue] = []
    /// 服务端通知 tools/list_changed 后回调最新工具清单。
    public var onToolsChanged: (([JSONValue]) -> Void)?
    /// 单条未换行缓冲的上限；超过视为畸形消息，避免无限增长。
    private let maxReceiveBuffer = 8 * 1024 * 1024

    public init(name: String, command: String, args: [String] = [],
                env: [String: String] = [:], timeout: TimeInterval = 30,
                maxOutputChars: Int = 200_000) {
        self.name = name
        self.command = command
        self.args = args
        self.env = env
        self.timeout = max(1, timeout)
        self.maxOutputChars = max(1, maxOutputChars)
    }

    public convenience init(config: MCPServerConfig, timeout: TimeInterval = 30) {
        self.init(name: config.name, command: config.command,
                  args: config.args, env: config.env, timeout: timeout)
    }

    deinit {
        stop()
    }

    // MARK: - 生命周期

    public var isRunning: Bool {
        lock.lock(); defer { lock.unlock() }
        return running
    }

    public var serverInfoValue: [String: JSONValue] {
        lock.lock(); defer { lock.unlock() }
        return serverInfo
    }

    /// 启动进程并完成 MCP 握手。
    ///
    /// 并发调用会等待同一个启动过程完成；`running` 只在握手期间置位，
    /// 避免第二个调用在 initialize 完成前就发出 tools/call。
    public func start() async throws {
        while true {
            lock.lock()
            if closed {
                lock.unlock()
                throw MCPError("MCP 服务 '\(name)' 已关闭")
            }
            if running && !startInProgress {
                lock.unlock()
                return
            }
            if startInProgress {
                lock.unlock()
                do {
                    try await Task.sleep(nanoseconds: 20_000_000)
                } catch {
                    throw CancellationError()
                }
                continue
            }
            startInProgress = true
            lock.unlock()
            break
        }
        defer {
            lock.lock(); startInProgress = false; lock.unlock()
        }
        try await performStart()
    }

    private func performStart() async throws {
        lock.lock()
        receiveBuffer = Data()
        pending.removeAll()
        // 重连时必须使用全新的 Process / Pipe：Process 不能重复 run，
        // Pipe 在 stop() 中已被关闭。
        process = Process()
        stdinPipe = Pipe()
        stdoutPipe = Pipe()
        stderrPipe = Pipe()
        lock.unlock()

        process.environment = ProcessInfo.processInfo.environment.merging(env) { _, new in new }
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe
        if command.contains("/") {
            process.executableURL = URL(fileURLWithPath: command)
            process.arguments = args
        } else {
            // 与 Python Popen 一致：无路径分隔符时使用 PATH 查找。
            process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            process.arguments = [command] + args
        }

        lock.lock(); running = true; lock.unlock()
        do {
            try process.run()
        } catch {
            lock.lock(); running = false; lock.unlock()
            throw MCPError("启动 MCP 服务 '\(name)' 失败: \(error.localizedDescription)")
        }

        installReadHandlers()

        do {
            let response = try await call("initialize", params: [
                "protocolVersion": .string("2024-11-05"),
                "capabilities": .object([:]),
                "clientInfo": .object([
                    "name": .string("base-agent"),
                    "version": .string("1.0"),
                ]),
            ])
            if let message = response.errorMessage {
                throw MCPError("MCP 服务 '\(name)' initialize 失败: \(message)")
            }
            lock.lock()
            serverInfo = response.result?.objectValue ?? [:]
            lock.unlock()

            try sendNotification("notifications/initialized", params: [:])
            try await refreshTools()
        } catch {
            stop()
            throw error
        }
    }

    /// 拉取并缓存工具清单。
    public func refreshTools() async throws {
        let response = try await call("tools/list", params: [:])
        if let message = response.errorMessage {
            throw MCPError("MCP 服务 '\(name)' tools/list 失败: \(message)")
        }
        let list = response.result?.objectValue?["tools"]?.arrayValue ?? []
        lock.lock()
        tools = list
        lock.unlock()
    }

    public func listTools() -> [JSONValue] {
        lock.lock(); defer { lock.unlock() }
        return tools
    }

    /// 终止进程并清理所有挂起请求。
    public func stop() {
        lock.lock()
        running = false
        let requests = pending
        pending.removeAll()
        let pid = process.processIdentifier
        lock.unlock()

        stdoutPipe.fileHandleForReading.readabilityHandler = nil
        stderrPipe.fileHandleForReading.readabilityHandler = nil

        if process.isRunning {
            // MCP 子进程通常是 npx/node 包装：必须连同子进程树一起终止。
            if pid > 0 {
                ToolProcessRunner.terminateProcessTree(pid: pid, signal: SIGTERM)
                ToolProcessRunner.terminateProcessTree(pid: pid, signal: SIGKILL)
            } else {
                process.terminate()
            }
        }
        try? stdinPipe.fileHandleForWriting.close()
        try? stdoutPipe.fileHandleForReading.close()
        try? stderrPipe.fileHandleForReading.close()

        let error = MCPError("MCP 服务 '\(name)' 已停止")
        for request in requests.values {
            lock.lock()
            guard !request.completed else { lock.unlock(); continue }
            request.completed = true
            request.result = .failure(error)
            let continuation = request.continuation
            request.continuation = nil
            lock.unlock()
            continuation?.resume(throwing: error)
        }
    }

    public func shutdown() {
        lock.lock()
        closed = true
        lock.unlock()
        stop()
    }

    // MARK: - 调用

    /// 调用 JSON-RPC 方法。超时或进程退出时抛错。
    public func call(_ method: String, params: [String: JSONValue],
                     timeout: TimeInterval? = nil) async throws -> MCPResponse {
        guard isRunning else {
            throw MCPError("MCP 服务 '\(name)' 未启动")
        }

        let request = PendingRequest()
        lock.lock()
        let id = nextID
        nextID += 1
        pending[id] = request
        lock.unlock()

        let message: [String: JSONValue] = [
            "jsonrpc": .string("2.0"),
            "id": .int(id),
            "method": .string(method),
            "params": .object(params),
        ]
        do {
            try send(message)
        } catch {
            removePending(id)
            throw error
        }

        let seconds = max(1, timeout ?? self.timeout)
        DispatchQueue.global().asyncAfter(deadline: .now() + seconds) { [weak self] in
            self?.failRequest(id, MCPError(
                "MCP 请求超时（\(method), \(Int(seconds))s）"))
        }

        return try await withCheckedThrowingContinuation {
            (continuation: CheckedContinuation<MCPResponse, Error>) in
            lock.lock()
            if request.completed {
                let result = request.result ?? .failure(MCPError("MCP 请求失败"))
                lock.unlock()
                continuation.resume(with: result)
            } else {
                request.continuation = continuation
                lock.unlock()
            }
        }
    }

    /// 调用服务端工具，返回文本结果；`isError=true` 时抛错。
    public func callTool(name: String, arguments: [String: JSONValue]) async throws -> String {
        // 服务进程异常退出后自动重连一次，避免工具永久失效；
        // 但已被 shutdown() 的客户端不允许借 callTool 复活。
        if !isRunning {
            lock.lock()
            let rejected = closed
            lock.unlock()
            if rejected {
                throw MCPError("MCP 服务 '\(self.name)' 已关闭")
            }
            try await start()
        }
        let response = try await call("tools/call", params: [
            "name": .string(name),
            "arguments": .object(arguments),
        ], timeout: max(timeout, toolCallTimeout))

        if let message = response.errorMessage {
            throw MCPError("MCP 工具 '\(name)' 调用失败: \(message)")
        }
        guard let result = response.result else {
            throw MCPError("MCP 工具 '\(name)' 未返回结果")
        }
        // 渲染时就限制长度：避免先拼出完整（可能几十 MB）的字符串再截断。
        let text = Self.render(result: result, limit: maxOutputChars + 1)
        if result.objectValue?["isError"]?.boolValue == true {
            throw MCPError("MCP 工具 '\(name)' 返回错误: \(text)")
        }
        guard text.count > maxOutputChars else { return text }
        return String(text.prefix(maxOutputChars))
            + "\n… [MCP 工具输出超过 \(maxOutputChars) 字符，已截断]"
    }

    // MARK: - 发送

    private func sendNotification(_ method: String, params: [String: JSONValue]) throws {
        var message: [String: JSONValue] = [
            "jsonrpc": .string("2.0"),
            "method": .string(method),
        ]
        if !params.isEmpty { message["params"] = .object(params) }
        try send(message)
    }

    private func send(_ message: [String: JSONValue]) throws {
        guard isRunning else {
            throw MCPError("MCP 服务 '\(name)' 未启动")
        }
        let line = JSONValue.object(message).jsonString() + "\n"
        guard let data = line.data(using: .utf8) else {
            throw MCPError("MCP 消息编码失败")
        }
        writeLock.lock()
        defer { writeLock.unlock() }
        // 写入放到专用串行队列并设超时：server 不读 stdin 时避免永久阻塞调用线程。
        let semaphore = DispatchSemaphore(value: 0)
        let outcome = WriteOutcome()
        writeQueue.async { [stdinPipe] in
            do {
                try stdinPipe.fileHandleForWriting.write(contentsOf: data)
            } catch {
                outcome.error = error
            }
            semaphore.signal()
        }
        if semaphore.wait(timeout: .now() + 10) == .timedOut {
            throw MCPError("写入 MCP 服务 '\(name)' 超时")
        }
        if let error = outcome.error {
            throw MCPError("写入 MCP 服务 '\(name)' 失败: \(error.localizedDescription)")
        }
    }

    // MARK: - 接收

    private func installReadHandlers() {
        stdoutPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if data.isEmpty {
                handle.readabilityHandler = nil
                self?.handleEOF()
                return
            }
            self?.consume(data)
        }
        stderrPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if data.isEmpty {
                handle.readabilityHandler = nil
                return
            }
            guard let text = String(data: data, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines),
                !text.isEmpty else { return }
            AgentLog.warning("MCP[stderr]: \(String(text.prefix(500)))")
        }
    }

    private func consume(_ data: Data) {
        lock.lock()
        if receiveBuffer.count + data.count > maxReceiveBuffer {
            // 没有换行的超长缓冲通常意味着协议损坏；直接丢弃，避免内存无界增长。
            receiveBuffer.removeAll(keepingCapacity: false)
            AgentLog.warning("MCP[\(name)]: 接收缓冲超过 \(maxReceiveBuffer) 字节，已丢弃")
        }
        receiveBuffer.append(data)
        var lines: [Data] = []
        while let newline = receiveBuffer.firstIndex(of: 0x0A) {
            let line = receiveBuffer[..<newline]
            let data = Data(line)
            receiveBuffer.removeSubrange(...newline)
            if !data.isEmpty { lines.append(data) }
        }
        lock.unlock()

        for line in lines { handleLine(line) }
    }

    private func handleLine(_ data: Data) {
        guard let text = String(data: data, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines),
            !text.isEmpty,
            let value = JSONValue.parse(text),
            let object = value.objectValue else {
            AgentLog.warning("MCP[\(name)]: 无法解析消息")
            return
        }

        // 服务端主动请求（含 id）：统一返回 method not found。
        if let method = object["method"]?.stringValue {
            let idValue = object["id"]
            let hasID: Bool = {
                guard let idValue else { return false }
                if case .null = idValue { return false }
                return true
            }()
            if hasID {
                var reply: [String: JSONValue] = [
                    "jsonrpc": .string("2.0"),
                    "id": idValue ?? .null,
                ]
                if method == "ping" {
                    reply["result"] = .object([:])
                } else {
                    reply["error"] = .object([
                        "code": .int(-32601),
                        "message": .string("Method not supported: \(method)"),
                    ])
                }
                try? send(reply)
                return
            }
            handleNotification(method: method, params: object["params"]?.objectValue ?? [:])
            return
        }

        guard let id = object["id"]?.intValue else { return }
        let error = object["error"]?.objectValue
        let response = MCPResponse(
            id: id,
            result: object["result"],
            errorCode: error?["code"]?.intValue,
            errorMessage: error?["message"]?.stringValue)
        complete(id, response)
    }

    private func handleNotification(method: String, params: [String: JSONValue]) {
        if method == "notifications/tools/list_changed" {
            Task { [weak self] in
                guard let self, self.isRunning else { return }
                guard (try? await self.refreshTools()) != nil else { return }
                // 让上层更新 ToolRegistry / schema；否则工具清单变了，
                // 模型侧仍然只能看到旧工具。
                self.onToolsChanged?(self.listTools())
            }
        }
        // 其他通知按协议忽略。
    }

    private func handleEOF() {
        failAllPending("MCP 服务 '\(name)' 连接已关闭")
    }

    // MARK: - 挂起请求

    private func complete(_ id: Int, _ response: MCPResponse) {
        lock.lock()
        guard let request = pending.removeValue(forKey: id), !request.completed else {
            lock.unlock()
            return
        }
        request.completed = true
        if let continuation = request.continuation {
            request.continuation = nil
            lock.unlock()
            continuation.resume(returning: response)
        } else {
            request.result = .success(response)
            lock.unlock()
        }
    }

    private func failRequest(_ id: Int, _ error: Error) {
        lock.lock()
        guard let request = pending.removeValue(forKey: id), !request.completed else {
            lock.unlock()
            return
        }
        request.completed = true
        request.result = .failure(error)
        let continuation = request.continuation
        request.continuation = nil
        lock.unlock()
        continuation?.resume(throwing: error)
    }

    private func removePending(_ id: Int) {
        lock.lock()
        pending.removeValue(forKey: id)
        lock.unlock()
    }

    private func failAllPending(_ message: String) {
        lock.lock()
        running = false
        let requests = pending
        pending.removeAll()
        let error = MCPError(message)
        for request in requests.values where !request.completed {
            request.completed = true
            request.result = .failure(error)
        }
        let continuations = requests.values.compactMap(\.continuation)
        for request in requests.values { request.continuation = nil }
        lock.unlock()
        for continuation in continuations {
            continuation.resume(throwing: error)
        }
    }

    // MARK: - 结果渲染

    private static func render(result: JSONValue, limit: Int) -> String {
        if let object = result.objectValue,
           let content = object["content"]?.arrayValue {
            var parts: [String] = []
            var total = 0
            for item in content {
                let piece: String
                if let block = item.objectValue {
                    switch block["type"]?.stringValue {
                    case "text":
                        piece = block["text"]?.stringValue ?? ""
                    case "resource":
                        piece = "[resource] \(item.jsonString())"
                    default:
                        piece = item.jsonString()
                    }
                } else {
                    piece = item.jsonString()
                }
                guard !piece.isEmpty else { continue }
                parts.append(piece)
                total += piece.count + 1
                if total >= limit { break }
            }
            let joined = parts.joined(separator: "\n")
            if !joined.isEmpty { return String(joined.prefix(limit)) }
        }
        return String(result.jsonString(pretty: true).prefix(limit))
    }
}
