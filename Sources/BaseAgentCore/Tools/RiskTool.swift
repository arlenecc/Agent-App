import Foundation

// MARK: - code_run / shell_run 的风险分级

/// 风险判定是纯辅助函数（不做成 AgentTool），供 shell_run / code_run 的
/// `shouldConfirm` 调用。策略对齐 Python `tools/risk.py`：
///
/// - 默认直接执行：只读操作、工作区内的读写、工作区内子命令；
/// - 弹确认：sudo / 系统与电源控制 / 进程杀伤 / 磁盘操作 /
///   管道下载执行 / `git push --force` / 写工作区之外的路径。
///
/// 判定是静态启发式（只读参数文本，不执行）。宁可漏报不可误杀：误报的确认
/// 弹窗会让用户养成无脑点「Yes」的习惯，确认机制反而失效。
public enum ToolRiskPolicy {

    /// shell_run 的风险判定。
    public static func isRiskyCommand(_ command: String, workspace: String) -> Bool {
        let lowered = command.lowercased()
        if matches(controlRegex, in: lowered) { return true }
        if matches(pipeExecRegex, in: lowered) { return true }
        if matches(gitForceRegex, in: lowered) { return true }
        // 破坏性动词（或向绝对路径重定向）+ 工作区之外的路径 → 风险
        if matches(shellVerbRegex, in: lowered) || matches(shellRedirectRegex, in: lowered) {
            if mentionsOutsidePath(command, workspace: workspace) { return true }
        }
        return false
    }

    /// code_run 的风险判定。
    public static func isRiskyCode(_ code: String, workspace: String) -> Bool {
        let source = code
        if matches(sudoRegex, in: source) { return true }
        if matches(pipeExecRegex, in: source) { return true }
        if matches(gitForceRegex, in: source) { return true }
        // 破坏性调用 + 工作区之外的路径字面量同现才判风险：
        // shutil.rmtree("build") 在工作区内是安全的；
        // subprocess.run(["pip", "install", "x"]) 无外部路径也安全。
        if matches(pyCallRegex, in: source),
           mentionsOutsidePath(source, workspace: workspace) {
            return true
        }
        return false
    }

    /// 文本中出现**工作区之外**的路径字面量 → true。
    ///
    /// 相对路径默认落在工作区内（两个工具都以 workspace 为 cwd），视为安全；
    /// 绝对路径与 `~` 展开做 realpath 前缀比对。
    public static func mentionsOutsidePath(_ text: String, workspace: String) -> Bool {
        guard !text.isEmpty, !workspace.isEmpty else { return false }

        // 先剥掉 URL，避免 http://x/y 被当成 //x/y 的绝对路径。
        let cleaned = replacingMatches(urlRegex, in: text, with: " ")
        let ws = ToolSupport.realPath(workspace)
        let nsText = cleaned as NSString
        let fullRange = NSRange(location: 0, length: nsText.length)

        for match in pathTokenRegex.matches(in: cleaned, range: fullRange) {
            let rawToken = nsText.substring(with: match.range)
            var token = rawToken.trimmingCharacters(in: pathTrimCharacters)
            // 裸 `~` / `/` / `..` 会被 trim 成空串（"." 在裁剪集合里）；
            // 它们恰好是最危险的目标之一，必须保留原样继续判定。
            if token.isEmpty,
               ["~", "/", "..", "$HOME", "${HOME}"].contains(rawToken) {
                token = rawToken
            }
            guard !token.isEmpty else { continue }

            // ../ 逃逸：按 workspace 解析后必须仍在工作区内。
            if token.hasPrefix("../") || token == ".." {
                let joined = (ws as NSString).appendingPathComponent(token)
                let resolved = (joined as NSString).standardizingPath
                if resolved != ws && !resolved.hasPrefix(ws + "/") { return true }
                continue
            }

            let expanded: String
            if token == "$HOME" || token == "${HOME}" {
                expanded = NSHomeDirectory()
            } else {
                expanded = (token as NSString).expandingTildeInPath
            }
            guard expanded.hasPrefix("/") else { continue }
            let normalized = (expanded as NSString).standardizingPath
            if safeAbsolutePaths.contains(normalized) { continue }
            let real = ToolSupport.realPath(normalized)
            if real != ws && !real.hasPrefix(ws + "/") { return true }
        }
        return false
    }

    // MARK: 内部工具

    private static func matches(_ regex: NSRegularExpression, in text: String) -> Bool {
        guard !text.isEmpty else { return false }
        let range = NSRange(location: 0, length: (text as NSString).length)
        return regex.firstMatch(in: text, options: [], range: range) != nil
    }

    private static func replacingMatches(_ regex: NSRegularExpression,
                                         in text: String,
                                         with template: String) -> String {
        let range = NSRange(location: 0, length: (text as NSString).length)
        return regex.stringByReplacingMatches(in: text, options: [], range: range,
                                              withTemplate: template)
    }

    // MARK: 规则表

    /// 常见重定向惯用法的安全特殊路径。
    private static let safeAbsolutePaths: Set<String> = [
        "/dev/null", "/dev/stdout", "/dev/stderr", "/dev/stdin",
        "/dev/zero", "/dev/urandom", "/dev/tty",
    ]

    /// ~ 开头的 HOME 展开、/ 开头的绝对路径、../ 逃逸、$HOME。
    ///
    /// 用 lookbehind 排除紧贴在单词/路径字符后的片段，避免把 `./` 里的 `/`
    /// 误当成根目录（`rm -rf ./` 是工作区内操作）。末尾允许 0 个字符，
    /// 以覆盖裸 `~`、`/`、`..`。
    private static let pathTokenRegex = try! NSRegularExpression(
        pattern: #"(?<![A-Za-z0-9_.~/\\])(?:[~/][^\s'"`|;&<>()\[\]{}]*"#
            + #"|\.\.[^\s'"`|;&<>()\[\]{}]*|\$\{?HOME\}?)"#)

    private static let urlRegex = try! NSRegularExpression(pattern: #"\b\w+://\S+"#)

    /// 尾部标点全部剥掉：避免 `~/foo!` / `../x)` 被当成不存在的路径而漏判。
    private static let pathTrimCharacters = CharacterSet(
        charactersIn: ".,;:)\"'!?]}>")

    /// Python 侧的「破坏性调用」记号：与危险路径同现才判风险。
    private static let pyCallRegex = try! NSRegularExpression(
        pattern: #"\b(os\.system|os\.popen|os\.remove|os\.unlink|os\.rmdir|os\.removedirs"# +
            #"|os\.chmod|os\.chown|os\.symlink|os\.link|os\.rename|os\.replace"# +
            #"|os\.truncate|os\.makedirs|os\.mkdir|os\.open|os\.fdopen"# +
            #"|io\.open|pathlib\.Path|Path\.open|write_text|write_bytes|touch"# +
            #"|shutil\.rmtree|shutil\.chown|shutil\.move|shutil\.copy|shutil\.copy2"# +
            #"|shutil\.copytree|subprocess)\b"#)

    /// shell 侧的破坏性动词（读文件不算：cat/ls/grep/head/tail 不在表里）。
    private static let shellVerbRegex = try! NSRegularExpression(
        pattern: #"\b(rm|rmdir|mv|cp|chmod|chown|ln|truncate|shred|tee|install|mkfile"# +
            #"|hdiutil|defaults"# +
            #"|dd|mkfs|fdisk|parted|wipefs|badblocks|mkswap|swapon|swapoff"# +
            #"|rmtree|removedirs|unlink|os\.system|os\.popen|os\.remove|os\.unlink"# +
            #"|os\.rmdir|os\.removedirs|os\.chmod|os\.chown|os\.rename|os\.replace"# +
            #"|os\.symlink|os\.link|os\.truncate|os\.open|os\.fdopen"# +
            #"|io\.open|pathlib\.Path|Path\.open|write_text|write_bytes|touch"# +
            #"|shutil\.rmtree|shutil\.move"# +
            #"|shutil\.copy|shutil\.copy2|shutil\.copytree"# +
            #"|subprocess\.(run|call|check_call|check_output|Popen))\b"#)

    /// 重定向到工作区之外的路径（> /etc/hosts、> ~/.bashrc、>> ../outside）。
    /// `./` 是工作区内相对路径，不能算风险。
    private static let shellRedirectRegex = try! NSRegularExpression(
        pattern: #">>?\s*(?:~|\.\./|/)"#)

    /// sudo / 系统-电源-进程-磁盘控制。
    private static let controlRegex = try! NSRegularExpression(
        pattern: #"\b(sudo|doas|su\s+-|shutdown|reboot|halt|poweroff|pmset|csrutil"# +
            #"|launchctl|nvram|diskutil|mkfs|kill|killall|pkill|spctl"# +
            #"|systemctl|service|ufw|iptables|pfctl|crontab)\b"#)

    /// 远程脚本直接执行（curl … | sh / | python3 / | node …）。
    private static let pipeExecRegex = try! NSRegularExpression(
        pattern: #"(curl|wget)\b[^|\n]*\|\s*(sudo\s+)?"# +
            #"((ba|z|da|fi)?sh|python[\d.]*|node|perl|ruby|php)\b"#)

    /// 改写远端历史。
    private static let gitForceRegex = try! NSRegularExpression(
        pattern: #"\bgit\s+push\b[^\n]*(--force\b|--force-with-lease\b|-f\b)"#)

    private static let sudoRegex = try! NSRegularExpression(pattern: #"\bsudo\b"#)
}
