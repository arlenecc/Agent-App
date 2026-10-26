import Foundation

// MARK: - shell_run

/// `shell_run`：以 workspace 为 cwd 执行 shell 命令，捕获 stdout/stderr、
/// 退出码，并支持超时与 UI「终止对话」取消。
///
/// 确认策略对齐 Python `tools/risk.py`：默认直接执行；sudo / 系统与电源控制 /
/// 进程杀伤 / 磁盘操作 / 管道下载执行 / `git push --force` / 写工作区之外
/// 路径才弹人工确认。`config.alwaysConfirmCode = true` 可恢复逐次确认。
public final class ShellRunTool: AgentTool {
    public let name = "shell_run"
    public let description =
        "Run a shell command in the workspace, return stdout+stderr. "
        + "For CLI tools, git, pip, build scripts, etc. "
        + "This is code_run's system-side extension: use it whenever a task needs "
        + "OS-level commands rather than Python logic."
    public let destructive = true

    private let config: AgentConfig
    private let callbacks: AgentCallbacks?
    private let defaultTimeout: Double = 120

    public var parameters: JSONValue {
        ToolSupport.objectSchema(
            properties: [
                "command": ToolSupport.stringProperty(
                    "Shell command to execute (may include pipes, redirects, etc.)."),
                "timeout": ToolSupport.numberProperty(
                    "Timeout in seconds (default \(defaultTimeout))."),
                "workdir": ToolSupport.stringProperty(
                    "Optional working directory inside the workspace (default: workspace root)."),
            ],
            required: ["command"])
    }

    public init(config: AgentConfig, callbacks: AgentCallbacks?, registry: ToolRegistry? = nil) {
        self.config = config
        self.callbacks = callbacks
        _ = registry
    }

    public func shouldConfirm(_ args: [String: JSONValue]) -> Bool {
        if config.alwaysConfirmCode { return true }
        let command = args["command"].map(ToolSupport.coerceString) ?? ""
        return ToolRiskPolicy.isRiskyCommand(command, workspace: config.workspace)
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        do {
            let command = try ToolSupport.requireString(args, "command")
            let timeout = ToolSupport.clampTimeout(
                ToolSupport.optionalDouble(args, "timeout"),
                default: defaultTimeout)

            let workspace = ToolSupport.realPath(config.workspace)
            let workdirArg = ToolSupport.optionalString(args, "workdir") ?? ""
            let workdir: String
            if workdirArg.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                workdir = workspace
            } else {
                workdir = try ToolSupport.resolveWorkspacePath(workdirArg,
                                                              workspace: config.workspace)
            }

            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: workdir, isDirectory: &isDirectory),
                  isDirectory.boolValue else {
                return .fail("workdir does not exist: \(workdirArg)")
            }

            let callbacks = self.callbacks
            let outcome = await ToolProcessRunner.run(
                executable: "/bin/sh",
                arguments: ["-c", command],
                currentDirectory: workdir,
                timeout: timeout,
                shouldCancel: { callbacks?.isCancelled() ?? false })

            if let launchError = outcome.launchError {
                return .fail(launchError)
            }

            let output = ToolSupport.combineOutput(stdout: outcome.stdout,
                                                   stderr: outcome.stderr)

            // ToolResult.toMessage 失败时只渲染 error；把截断后的输出并入
            // error，否则模型只看到「Exit code 1」而无从定位（对齐 code_run）。
            func withOutput(_ message: String) -> String {
                let tail = output.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !tail.isEmpty else { return message }
                return message + "\n[命令输出（尾部）]\n\(String(tail.suffix(3_000)))"
            }

            if outcome.cancelled {
                return ToolResult(success: false, output: output,
                                  error: withOutput("用户取消了对话，命令被中断"))
            }

            if outcome.timedOut {
                return ToolResult(
                    success: false,
                    output: output,
                    error: withOutput("Command timed out after \(Int(timeout))s: "
                        + String(command.prefix(200))))
            }

            if outcome.exitCode == 0 {
                return .ok(output.isEmpty ? "(no output)" : output)
            }

            let detail = outcome.exitCode.map { "Exit code \($0)" }
                ?? "进程未能被回收（可能仍在运行或被信号终止）"
            return ToolResult(success: false, output: output, error: withOutput(detail))
        } catch {
            return ToolSupport.failure(error)
        }
    }
}
