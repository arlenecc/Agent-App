import Foundation
import AppKit
import BaseAgentCore

// MARK: - UI 模型

struct ChatItem: Identifiable, Equatable {
    enum Role: String { case user, assistant, system }
    let id = UUID()
    var role: Role
    var content: String
}

struct ToolEvent: Identifiable {
    let id = UUID()
    var name: String
    var arguments: String
    var success: Bool?
    var output: String
    var timestamp = Date()
}

struct LogLine: Identifiable {
    let id = UUID()
    var text: String
    var timestamp = Date()
}

/// 协作式取消标志：Agent 回调在后台线程读取，UI 在主线程写入。
final class CancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    var isCancelled: Bool {
        lock.lock(); defer { lock.unlock() }
        return value
    }
    func set(_ newValue: Bool) {
        lock.lock(); value = newValue; lock.unlock()
    }
}

// MARK: - Callbacks

/// 把 AgentCallbacks 的事件桥接到主线程 UI。
final class UICallbacks: AgentCallbacks, @unchecked Sendable {
    private weak var model: AppViewModel?
    private let cancel: CancelFlag

    init(model: AppViewModel, cancel: CancelFlag) {
        self.model = model
        self.cancel = cancel
    }

    func isCancelled() -> Bool { cancel.isCancelled }

    func onContent(_ text: String) {
        DispatchQueue.main.async { [weak self] in self?.model?.appendContent(text) }
    }

    func onReasoning(_ text: String) {
        DispatchQueue.main.async { [weak self] in self?.model?.appendReasoning(text) }
    }

    func onToolStart(_ name: String, _ args: [String: JSONValue]) {
        DispatchQueue.main.async { [weak self] in
            self?.model?.beginTool(name: name, args: args)
        }
    }

    func onToolEnd(_ name: String, _ result: ToolResult) {
        DispatchQueue.main.async { [weak self] in
            self?.model?.finishTool(name: name, result: result)
        }
    }

    func onLog(_ line: String) {
        DispatchQueue.main.async { [weak self] in self?.model?.appendLog(line) }
    }

    func onUsage(_ usage: [String: Int]) {
        DispatchQueue.main.async { [weak self] in
            self?.model?.totalTokens = usage["total_tokens"] ?? usage["total"] ?? 0
        }
    }

    func onTokenSpeed(_ totalTokens: Int, _ speed: Double) {
        DispatchQueue.main.async { [weak self] in
            self?.model?.tokenSpeed = speed
        }
    }

    func onPlanUpdated(_ plan: Plan?) {
        DispatchQueue.main.async { [weak self] in self?.model?.plan = plan }
    }

    func onError(_ message: String) {
        DispatchQueue.main.async { [weak self] in
            self?.model?.appendLog("ERROR \(message)")
            self?.model?.appendSystem(message)
        }
    }

    func onContextShrunk(_ summary: String, _ reason: String) {
        DispatchQueue.main.async { [weak self] in
            self?.model?.appendLog("[context] 已收缩（\(reason)）：\(summary.prefix(200))")
        }
    }

    func onSkillSuggested(_ skill: SkillSuggestion) {
        DispatchQueue.main.async { [weak self] in
            self?.model?.appendLog("[skill] 重复请求，可固化为技能：\(skill.name)")
        }
    }

    func confirm(_ message: String) -> Bool {
        runOnMain {
            let alert = NSAlert()
            alert.messageText = "需要确认"
            alert.informativeText = message
            alert.alertStyle = .warning
            // 默认按钮是「拒绝」：回车/误触不会自动放行高风险操作。
            alert.addButton(withTitle: "拒绝")
            alert.addButton(withTitle: "允许")
            return alert.runModal() == .alertSecondButtonReturn
        }
    }

    func askUser(_ prompt: String) -> String? {
        runOnMain {
            let alert = NSAlert()
            alert.messageText = "需要补充信息"
            alert.informativeText = prompt
            alert.addButton(withTitle: "提交")
            alert.addButton(withTitle: "取消")
            let field = NSTextField(frame: NSRect(x: 0, y: 0, width: 420, height: 24))
            field.placeholderString = "请输入…"
            alert.accessoryView = field
            guard alert.runModal() == .alertFirstButtonReturn else { return nil }
            let value = field.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
            return value.isEmpty ? nil : value
        }
    }

    func onFinished() {
        DispatchQueue.main.async { [weak self] in self?.model?.finishRun() }
    }

    private func runOnMain<T>(_ body: () -> T) -> T {
        if Thread.isMainThread { return body() }
        return DispatchQueue.main.sync(execute: body)
    }
}

// MARK: - AppViewModel

@MainActor
final class AppViewModel: ObservableObject {
    @Published var config: AgentConfig
    /// 识别测试窗口使用的临时配置（含尚未保存的自定义 OCR 设置）；为空时回退 `config`。
    @Published var ocrLabConfig: AgentConfig?
    @Published var chatItems: [ChatItem] = []
    @Published var reasoning = ""
    @Published var logs: [LogLine] = []
    @Published var toolEvents: [ToolEvent] = []
    @Published var plan: Plan?
    @Published var isRunning = false
    @Published var statusText = "就绪"
    @Published var totalTokens = 0
    @Published var tokenSpeed: Double = 0
    @Published var ragStatusText = "RAG 未就绪"
    @Published var isSyncing = false
    /// 已请求停止、正在等当前解析/embedding 批次收尾。
    @Published var isStoppingSync = false
    /// 正在手动清理向量库内的重复切片。
    @Published var isDeduplicating = false
    @Published var isModelLoading = false
    /// 主模型加载完成后的 KV 预热阶段；期间禁止发送消息，避免和 prefill 争夺模型。
    @Published var isWarmingUp = false
    @Published var mainModelLoaded = false
    @Published var auxModelLoaded = false

    let cancelFlag = CancelFlag()
    private(set) var agent: Agent?
    private var streamingID: UUID?
    private var pendingToolIndices: [String: [Int]] = [:]
    private var runTask: Task<Void, Never>?
    private var syncTask: Task<Void, Never>?
    /// 空闲内存回收：无前台活动一段时间后释放 aux 权重与 Embedding/Reranker/OCR。
    private var idleUnloadTask: Task<Void, Never>?
    /// 每次同步/取消同步递增，旧同步的收尾不能清掉新同步的状态。
    private var syncGeneration = 0
    private var runGeneration = 0
    private var contentBuffer = ""
    private var reasoningBuffer = ""
    private var summaryFilter = SummaryStreamFilter()
    private var flushScheduled = false
    private var ragStatusRefreshing = false
    private var configGeneration = 0
    /// `applyConfig { sync }` 路径：记住应用配置前已加载的模型，同步结束后再恢复。
    private var pendingSyncRestore: (main: Bool, aux: Bool)?

    static var configPath: String {
        (NSHomeDirectory() as NSString).appendingPathComponent(".base-agent/config.json")
    }

    init() {
        config = AgentConfig.load(Self.configPath)
        config.ensureWorkspace()
        buildAgent()
        appendLog("base-agent 启动，workspace=\(config.workspace)")
        // 上次退出时未完成的计划不跨启动恢复：直接丢弃，避免开盘即自动续跑。
        if agent?.discardPersistedPlan() == true {
            appendLog("[plan] 已清除上次未完成的计划")
        }
        // 默认在后台加载主/辅模型；模型缺失时只记录可诊断错误，不阻塞启动。
        Task { await preloadModels(main: config.llmProvider == "local", aux: true) }
    }

    // MARK: Agent 生命周期

    /// 丢弃当前流式缓冲。配置切换/清空对话后，旧任务迟到的 flush 不应写进新气泡。
    private func clearStreamingState() {
        contentBuffer = ""
        reasoningBuffer = ""
        streamingID = nil
        flushScheduled = false
        summaryFilter.reset()
    }

    private func buildAgent() {
        // 新 Agent 的模型服务尚未预加载；成功 preload 后再置为已加载。
        mainModelLoaded = false
        auxModelLoaded = false
        LocalModelFactory.shared.setCacheLimitMB(config.mlxCacheLimitMB)
        LocalModelFactory.shared.setMemoryLimitMB(config.mlxMemoryLimitMB)
        if config.effectiveMainContextTokens < config.maxContextTokens {
            appendLog("[memory] 主模型上下文按 KV 上限收紧到 "
                + "\(config.effectiveMainContextTokens) tokens")
        }
        if config.effectiveAuxContextTokens < config.auxMaxContextTokens {
            appendLog("[memory] 辅助模型上下文按 KV 上限收紧到 "
                + "\(config.effectiveAuxContextTokens) tokens")
        }
        let llm = LocalModelFactory.shared.makeLLM(config: config)
        let callbacks = UICallbacks(model: self, cancel: cancelFlag)
        let newAgent = Agent(llm: llm, config: config, callbacks: callbacks)
        newAgent.auxLLM = LocalModelFactory.shared.makeAuxLLM(config: config)
        agent = newAgent
        refreshRAGStatus()
        Task { [weak newAgent] in
            await newAgent?.connectMCP()
        }
    }

    /// 应用完整配置（知识库、Agent、模型等）。重建 Agent 会关闭旧模型实例；
    /// 这里记住此前的加载状态，应用配置后自动恢复。
    /// `afterApply` 用于「先应用知识库配置、再同步」这类链式操作。
    func applyConfig(_ newConfig: AgentConfig,
                     afterApply: (@MainActor () -> Void)? = nil) {
        // 先停掉在途对话，再关闭 Agent，避免旧 Task 继续使用已释放/已换配置的模型。
        let wasMainLoaded = mainModelLoaded
        let wasAuxLoaded = auxModelLoaded
        let oldTask = runTask
        runGeneration += 1
        configGeneration += 1
        let generation = configGeneration
        isModelLoading = false
        cancelFlag.set(true)
        oldTask?.cancel()
        Task { [weak self] in
            _ = await oldTask?.value
            await MainActor.run {
                guard let self else { return }
                // 连续修改配置时只应用最后一次，避免旧任务晚到把新 agent 覆盖掉。
                guard self.configGeneration == generation else { return }
                self.cancelIdleUnload()
                self.clearStreamingState()
                self.isWarmingUp = false
                self.config = newConfig
                do {
                    try self.config.save(Self.configPath)
                } catch {
                    self.appendLog("配置保存失败: \(error.localizedDescription)")
                }
                // 取消旧知识库同步，避免旧 agent 的 ingest 在 close 时占着锁，
                // 导致连接迟迟不释放（close 会在同步结束后补关）。
                self.agent?.ragEngine.cancel()
                self.syncGeneration += 1
                self.syncTask?.cancel()
                self.syncTask = nil
                self.isStoppingSync = false
                self.agent?.close()
                LocalModelFactory.shared.clearCaches()
                self.buildAgent()
                self.plan = nil
                self.isRunning = false
                self.isSyncing = false
                self.pendingToolIndices.removeAll()
                self.toolEvents.removeAll()
                self.cancelFlag.set(false)
                self.appendLog("配置已应用："
                    + (self.config.llmProvider == "local"
                       ? self.config.localLLMModel : self.config.model))
                self.appendLog("知识库目录：\(self.config.resolvedKnowledgeBase)"
                    + (self.config.knowledgeBase.isEmpty ? "（跟随工作目录）" : ""))
                if let afterApply {
                    // 同步会先卸载模型、结束后再恢复；这里不要同时预加载，
                    // 否则同步期间模型又常驻内存，抵消资源释放。
                    self.pendingSyncRestore = (wasMainLoaded, wasAuxLoaded)
                    afterApply()
                } else if wasMainLoaded || wasAuxLoaded {
                    Task {
                        await self.preloadModels(main: wasMainLoaded,
                                                 aux: wasAuxLoaded)
                    }
                }
            }
        }
    }

    // MARK: 单模型重载

    private enum ModelSlot { case main, aux }

    /// 只重载主模型，保留已加载的辅助模型与当前会话。
    func reloadMainModel(config newConfig: AgentConfig) {
        swapModel(.main, config: newConfig)
    }

    /// 只重载辅助模型，保留已加载的主模型与当前会话。
    func reloadAuxModel(config newConfig: AgentConfig) {
        swapModel(.aux, config: newConfig)
    }

    /// 重建 Agent 会同时关闭主/辅模型；这里只替换目标槽位，另一个模型保持常驻。
    private func swapModel(_ slot: ModelSlot, config newConfig: AgentConfig) {
        guard !isSyncing, !isDeduplicating else {
            isModelLoading = false
            appendLog("[model] 知识同步进行中，暂不加载模型")
            return
        }
        let oldTask = runTask
        runGeneration += 1
        configGeneration += 1
        let generation = configGeneration
        cancelFlag.set(true)
        oldTask?.cancel()
        cancelIdleUnload()
        isRunning = false
        isModelLoading = true
        statusText = "模型加载中…"

        Task { [weak self] in
            _ = await oldTask?.value
            guard let self, self.configGeneration == generation else { return }
            guard let agent = self.agent else {
                self.isModelLoading = false
                return
            }

            self.clearStreamingState()
            self.isWarmingUp = false
            self.config = newConfig
            do {
                try self.config.save(Self.configPath)
            } catch {
                self.appendLog("配置保存失败: \(error.localizedDescription)")
            }
            // 温度等请求参数由 KernelLoop 读取 agent.config，必须同步。
            agent.config = newConfig
            self.pendingToolIndices.removeAll()
            self.cancelFlag.set(false)

            switch slot {
            case .main:
                // 先释放旧主模型，再按新参数创建；辅助模型不受影响。
                agent.switchLLM(UnloadedLLMService())
                LocalModelFactory.shared.unloadLLMCaches()
                let service = LocalModelFactory.shared.makeLLM(config: newConfig)
                do {
                    if let preloadable = service as? PreloadableLLMService {
                        try await preloadable.preload()
                    }
                    guard self.configGeneration == generation else { return }
                    agent.switchLLM(service)
                    self.mainModelLoaded = true
                    let label = newConfig.localLLMModel.isEmpty
                        ? "默认主模型" : newConfig.localLLMModel
                    self.appendLog("[model] 主模型已加载：\(label)")
                    self.statusText = "模型加载成功"
                    try? await Task.sleep(nanoseconds: 500_000_000)
                    guard self.configGeneration == generation else { return }
                    if let warmable = service as? WarmupLLMService {
                        await self.warmupMainModel(warmable, generation: generation)
                    } else if !self.isRunning {
                        self.statusText = "就绪"
                    }
                } catch {
                    self.mainModelLoaded = false
                    self.appendLog("[model] 主模型加载失败：\(error.localizedDescription)")
                    if !self.isRunning { self.statusText = "模型加载失败" }
                }

            case .aux:
                // 先释放旧辅助模型，再按新参数创建；主模型不受影响。
                agent.unloadAuxLLM()
                LocalModelFactory.shared.unloadLLMCaches()
                let service = LocalModelFactory.shared.makeAuxLLM(config: newConfig)
                do {
                    if let preloadable = service as? PreloadableLLMService {
                        try await preloadable.preload()
                    }
                    guard self.configGeneration == generation else { return }
                    agent.auxLLM = service
                    self.auxModelLoaded = true
                    let label = newConfig.auxLocalLLMModel.isEmpty
                        ? "默认辅助模型" : newConfig.auxLocalLLMModel
                    self.appendLog("[model] 辅助模型已加载：\(label)")
                    self.statusText = "模型加载成功"
                    try? await Task.sleep(nanoseconds: 500_000_000)
                    if !self.isRunning { self.statusText = "就绪" }
                } catch {
                    self.auxModelLoaded = false
                    self.appendLog("[model] 辅助模型加载失败：\(error.localizedDescription)")
                    if !self.isRunning { self.statusText = "模型加载失败" }
                }
            }

            if self.configGeneration == generation {
                self.isModelLoading = false
            }
        }
    }

    // MARK: 模型加载 / 卸载

    /// 主动把主/辅模型权重加载到内存。API 模式的主模型不需要预加载。
    func preloadModels(main: Bool, aux: Bool) async {
        guard main || aux else { return }
        let generation = configGeneration
        isModelLoading = true
        statusText = "模型加载中…"
        var hadError = false
        var loadedAny = false
        var warmable: WarmupLLMService?
        defer {
            // 配置已切换时，旧预加载无权清掉新一次加载的状态。
            if configGeneration == generation {
                isModelLoading = false
                isWarmingUp = false
            }
        }

        if main, config.llmProvider.lowercased() == "local",
           let service = agent?.llm as? PreloadableLLMService {
            do {
                try await service.preload()
                guard configGeneration == generation else { return }
                mainModelLoaded = true
                loadedAny = true
                warmable = service as? WarmupLLMService
                let label = config.localLLMModel.isEmpty
                    ? "默认主模型" : config.localLLMModel
                appendLog("[model] 主模型已加载：\(label)")
            } catch {
                guard configGeneration == generation else { return }
                hadError = true
                mainModelLoaded = false
                appendLog("[model] 主模型加载失败：\(error.localizedDescription)")
            }
        }

        if aux, let service = agent?.auxLLM as? PreloadableLLMService {
            do {
                try await service.preload()
                guard configGeneration == generation else { return }
                auxModelLoaded = true
                loadedAny = true
                let label = config.auxLocalLLMModel.isEmpty
                    ? "默认辅助模型" : config.auxLocalLLMModel
                appendLog("[model] 辅助模型已加载：\(label)")
            } catch {
                guard configGeneration == generation else { return }
                hadError = true
                auxModelLoaded = false
                appendLog("[model] 辅助模型加载失败：\(error.localizedDescription)")
            }
        }

        guard configGeneration == generation else { return }
        if loadedAny {
            statusText = "模型加载成功"
            // 让「加载成功」在状态栏可见，再进入预热。
            try? await Task.sleep(nanoseconds: 500_000_000)
        }
        guard configGeneration == generation else { return }
        // 辅助模型失败不应阻止主模型预热。
        if let warmable {
            await warmupMainModel(warmable, generation: generation)
        } else if !isRunning {
            statusText = hadError ? "模型加载失败" : "就绪"
        }
        if !isRunning { scheduleIdleUnload() }
    }

    /// 用稳定的 system prompt + 工具清单预热主模型 KV cache，并更新状态栏。
    private func warmupMainModel(_ service: WarmupLLMService,
                                 generation: Int? = nil) async {
        guard let agent else { return }
        if let generation, configGeneration != generation { return }
        isWarmingUp = true
        defer {
            if generation == nil || configGeneration == generation {
                isWarmingUp = false
            }
        }
        statusText = "模型预热中…"
        // MCP 工具清单会进入 system 前缀，预热前先等连接完成。
        await agent.connectMCP()
        guard generation == nil || configGeneration == generation else { return }
        do {
            try await service.warmup(systemPrompt: agent.systemPrompt(),
                                     tools: agent.dispatcher.schemas())
            guard generation == nil || configGeneration == generation else { return }
            appendLog("[model] 主模型预热结束")
            if !isRunning { statusText = "预热结束" }
            try? await Task.sleep(nanoseconds: 1_200_000_000)
            if !isRunning, statusText == "预热结束" { statusText = "就绪" }
        } catch {
            guard generation == nil || configGeneration == generation else { return }
            appendLog("[model] 主模型预热失败：\(error.localizedDescription)")
            if !isRunning { statusText = "预热失败" }
        }
    }

    /// 卸载主模型，保留会话与 RAG；再次对话前需要在配置窗口重新加载。
    func unloadMainModel() {
        // 生成/加载/预热期间切换 LLM 会让在途请求使用已关闭的模型。
        guard !isRunning, !isModelLoading, !isWarmingUp, !isDeduplicating else { return }
        agent?.switchLLM(UnloadedLLMService())
        LocalModelFactory.shared.unloadLLMCaches()
        mainModelLoaded = false
        appendLog("[model] 主模型已卸载")
        statusText = "主模型未加载"
    }

    /// 卸载辅助模型；主模型不受影响。
    func unloadAuxModel() {
        guard !isRunning, !isModelLoading, !isWarmingUp, !isDeduplicating else { return }
        agent?.unloadAuxLLM()
        LocalModelFactory.shared.unloadLLMCaches()
        auxModelLoaded = false
        appendLog("[model] 辅助模型已卸载")
    }

    func resetConversation() {
        cancelIdleUnload()
        let oldTask = runTask
        runGeneration += 1
        cancelFlag.set(true)
        agent?.ragEngine.cancel()
        oldTask?.cancel()
        Task { [weak self] in
            _ = await oldTask?.value
            await MainActor.run {
                guard let self else { return }
                self.chatItems.removeAll()
                self.reasoning = ""
                self.toolEvents.removeAll()
                self.pendingToolIndices.removeAll()
                self.plan = nil
                self.clearStreamingState()
                self.agent?.reset()
                self.cancelFlag.set(false)
                self.isRunning = false
                self.totalTokens = 0
                self.tokenSpeed = 0
                self.statusText = "就绪"
                self.refreshRAGStatus()
                self.appendLog("对话已清空")
                self.scheduleIdleUnload()
            }
        }
    }

    func cancel() {
        guard isRunning else { return }
        cancelFlag.set(true)
        runTask?.cancel()
        statusText = "正在终止…"
        appendLog("用户请求终止")
        agent?.ragEngine.cancel()
    }

    /// 清空任务计划：运行中先请求终止，结束后再次清掉迟到的计划更新。
    func clearPlan() {
        cancelIdleUnload()
        let wasRunning = isRunning
        if wasRunning { cancel() }
        let task = runTask
        agent?.discardPersistedPlan()
        plan = nil
        appendLog(wasRunning ? "已请求终止，并清空任务计划" : "任务计划已清空")
        guard let task else { return }
        Task { [weak self] in
            _ = await task.value
            await MainActor.run {
                guard let self else { return }
                self.agent?.discardPersistedPlan()
                self.plan = nil
            }
        }
    }

    // MARK: 发送

    func send(_ input: String) {
        let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isRunning, !isModelLoading, !isWarmingUp,
              !isSyncing, !isDeduplicating, let agent else { return }
        cancelIdleUnload()
        runGeneration += 1
        let generation = runGeneration
        // 长会话只保留最近 1000 条消息，避免 UI 数组与布局无限增长。
        if chatItems.count > 1_000 { chatItems.removeFirst(chatItems.count - 1_000) }
        clearStreamingState()
        chatItems.append(ChatItem(role: .user, content: text))
        let item = ChatItem(role: .assistant, content: "")
        chatItems.append(item)
        streamingID = item.id
        reasoning = ""
        toolEvents.removeAll()
        pendingToolIndices.removeAll()
        isRunning = true
        statusText = "思考中…"
        totalTokens = 0
        tokenSpeed = 0
        cancelFlag.set(false)

        runTask = Task { [weak self, weak agent] in
            guard let self, let agent else { return }
            let answer = await agent.run(text)
            await MainActor.run {
                guard self.runGeneration == generation else { return }
                self.finishSummaryFilter()
                self.flushBuffers()
                self.finalizeStreaming(with: answer)
                self.isRunning = false
                self.statusText = "就绪"
                self.cancelFlag.set(false)
                self.refreshRAGStatus()
                self.scheduleIdleUnload()
            }
        }
    }

    /// 退出前的有序收尾：停止在途生成/同步，等待 GPU 工作结束并释放模型。
    /// 不做这步直接退出，MLX 的 asyncEval 会与进程退出并发而崩溃。
    func prepareForTermination() async {
        cancelIdleUnload()
        cancelFlag.set(true)
        runGeneration += 1
        let running = runTask
        runTask = nil
        running?.cancel()
        agent?.ragEngine.cancel()
        syncTask?.cancel()
        _ = await running?.value
        if let agent {
            await agent.shutdownForTermination()
        }
        await LocalModelFactory.shared.unloadAllForTermination()
        LocalModelFactory.shared.unloadLLMCaches()
        LocalModelFactory.shared.synchronizeGPU()
    }

    // MARK: 空闲内存回收

    /// 延迟 3 分钟无前台活动后，释放所有可重建的内存：
    /// 主模型对话 KV、辅助模型权重、Embedding/Reranker/OCR 权重。
    /// provider / 服务对象保留，下次使用自动按需重载。
    private func scheduleIdleUnload() {
        idleUnloadTask?.cancel()
        idleUnloadTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 180_000_000_000)
            guard let self, !Task.isCancelled else { return }
            guard !self.isRunning, !self.isModelLoading, !self.isWarmingUp,
                  !self.isSyncing, !self.isDeduplicating,
                  (self.agent?.isBackgroundIdle() ?? true) else { return }
            // 主模型 / 辅助模型 / Reranker 常驻；只回收按需使用的 Embedding 与 OCR。
            await LocalModelFactory.shared.unloadIdleRAGModelWeights()
        }
    }

    private func cancelIdleUnload() {
        idleUnloadTask?.cancel()
        idleUnloadTask = nil
    }

    // MARK: 流式更新

    func appendContent(_ text: String) {
        guard streamingID != nil else { return }
        // `<summary>` 是内核协议标签：正文照常进对话，摘要转进思考区。
        let parts = summaryFilter.feed(text)
        if !parts.content.isEmpty { contentBuffer += parts.content }
        if !parts.summary.isEmpty { reasoningBuffer += parts.summary }
        if !parts.content.isEmpty || !parts.summary.isEmpty {
            scheduleFlush()
        }
        statusText = "生成中…"
    }

    func appendReasoning(_ text: String) {
        reasoningBuffer += text
        scheduleFlush()
    }

    /// 每 50ms 合并一次流式增量，避免逐 token 修改整个数组触发重绘。
    private func scheduleFlush() {
        guard !flushScheduled else { return }
        flushScheduled = true
        let generation = runGeneration
        Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: 50_000_000)
            guard let self else { return }
            // 配置切换/清空对话后，旧任务不能再把缓冲写进新气泡。
            guard self.runGeneration == generation else {
                self.flushScheduled = false
                return
            }
            self.flushBuffers()
        }
    }

    /// 流结束时取出过滤器里尚未闭合的残留内容。
    private func finishSummaryFilter() {
        let tail = summaryFilter.finish()
        if !tail.content.isEmpty { contentBuffer += tail.content }
        if !tail.summary.isEmpty { reasoningBuffer += tail.summary }
    }

    func flushBuffers() {
        flushScheduled = false
        if !contentBuffer.isEmpty, let id = streamingID,
           let index = chatItems.firstIndex(where: { $0.id == id }) {
            let existing = chatItems[index].content
            if existing.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                // 工具调用前的中间回复可能只剩被剥掉 <tool_call> 后的空行；
                // 气泡第一个可见字符之前不应保留这些空白，否则会出现巨大的空背景框。
                chatItems[index].content = String(contentBuffer.drop(while: \.isWhitespace))
            } else {
                chatItems[index].content += contentBuffer
            }
            contentBuffer = ""
        }
        if !reasoningBuffer.isEmpty {
            reasoning += reasoningBuffer
            reasoningBuffer = ""
        }
    }

    func appendSystem(_ text: String) {
        chatItems.append(ChatItem(role: .system, content: text))
    }

    func appendLog(_ text: String) {
        let line = LogLine(text: text)
        logs.append(line)
        if logs.count > 2_000 { logs.removeFirst(logs.count - 2_000) }
    }

    /// 手动清空日志面板（日志与工具调用记录都是内存态，进程真正退出
    /// 后本就清空；此按钮用于不重启时手动清理）。
    func clearLogs() {
        logs.removeAll()
        toolEvents.removeAll()
    }

    func beginTool(name: String, args: [String: JSONValue]) {
        // 上一次流式回复到此结束；若模型在 <summary> 内被截断，先落盘并复位，
        // 避免把下一段回复的正文一直吞进思考区。
        finishSummaryFilter()
        flushBuffers()
        var arguments = JSONValue.object(args).jsonString()
        if arguments.count > 500 { arguments = String(arguments.prefix(500)) + "…" }
        let event = ToolEvent(name: name, arguments: arguments, success: nil, output: "")
        toolEvents.append(event)
        pendingToolIndices[name, default: []].append(toolEvents.count - 1)
        statusText = "调用工具 \(name)…"
    }

    func finishTool(name: String, result: ToolResult) {
        if var indices = pendingToolIndices[name], let index = indices.first,
           index < toolEvents.count {
            toolEvents[index].success = result.success
            let body = result.success ? result.output : result.error
            toolEvents[index].output = String(body.prefix(600))
            indices.removeFirst()
            pendingToolIndices[name] = indices.isEmpty ? nil : indices
        }
    }

    func finishRun() {
        isRunning = false
        statusText = "就绪"
        cancelFlag.set(false)
    }

    private func finalizeStreaming(with answer: String) {
        // 兜底答案同样剥掉协议标签，避免摘要回落到对话区。
        let cleaned = stripSummaryTags(answer)
        guard let id = streamingID,
              let index = chatItems.firstIndex(where: { $0.id == id }) else {
            if !cleaned.isEmpty { chatItems.append(ChatItem(role: .assistant, content: cleaned)) }
            return
        }
        if chatItems[index].content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            chatItems[index].content = cleaned
        }
        streamingID = nil
    }

    // MARK: RAG

    /// 点击「向量去重」：先扫描并把重复数量反馈给用户，确认后再清理。
    func requestVectorDedupe() {
        guard let agent else {
            showInfoAlert(title: "向量去重", message: "RAG 尚未就绪，请先在知识同步配置中应用配置。")
            return
        }
        if isSyncing {
            showInfoAlert(title: "暂不能向量去重", message: "知识同步正在进行，请等同步结束后再试。")
            return
        }
        if isDeduplicating {
            showInfoAlert(title: "向量去重进行中", message: "正在排查或清理重复向量，请稍候。")
            return
        }
        if isModelLoading || isWarmingUp {
            showInfoAlert(title: "暂不能向量去重", message: "模型正在加载或预热，请稍后再试。")
            return
        }
        if isRunning {
            showInfoAlert(title: "暂不能向量去重", message: "对话正在生成，请等本轮结束后再试。")
            return
        }
        cancelIdleUnload()
        isDeduplicating = true
        statusText = "正在扫描重复向量…"
        let target = ObjectIdentifier(agent)
        Task.detached(priority: .utility) {
            let result: Result<VectorDedupPreview, Error>
            do {
                result = .success(try agent.ragEngine.previewVectorDedupe())
            } catch {
                result = .failure(error)
            }
            await MainActor.run { [weak self] in
                guard let self else { return }
                self.isDeduplicating = false
                guard let current = self.agent, ObjectIdentifier(current) == target else {
                    self.statusText = "就绪"
                    self.scheduleIdleUnload()
                    return
                }
                switch result {
                case .failure(let error):
                    self.appendLog("RAG: 向量去重扫描失败 — \(error.localizedDescription)")
                    self.statusText = "向量去重失败"
                    self.showInfoAlert(title: "向量去重失败", message: error.localizedDescription)
                    self.scheduleIdleUnload()
                case .success(let preview):
                    if preview.duplicateChunks == 0 {
                        self.appendLog("RAG: 向量库共 \(preview.totalChunks) 条切片，未发现重复向量")
                        self.statusText = "未发现重复向量"
                        self.showInfoAlert(
                            title: "向量去重",
                            message: "向量库共 \(preview.totalChunks) 条切片，未发现重复向量。")
                        self.scheduleIdleUnload()
                        return
                    }
                    if self.confirmVectorDedupe(preview) {
                        self.performVectorDedupe(agent: agent, target: target)
                    } else {
                        self.appendLog("[rag] 已取消向量去重")
                        self.statusText = "就绪"
                        self.scheduleIdleUnload()
                    }
                }
            }
        }
    }

    /// 用户确认后执行实际删除。
    private func performVectorDedupe(agent: Agent, target: ObjectIdentifier) {
        guard !isSyncing, !isDeduplicating, !isModelLoading, !isRunning, !isWarmingUp else {
            showInfoAlert(title: "暂不能向量去重", message: "当前有任务正在进行，请稍后再试。")
            return
        }
        cancelIdleUnload()
        isDeduplicating = true
        statusText = "正在清理重复向量…"
        appendLog("[rag] 开始向量去重：清理重复切片，保留入库时间最早的一份")
        Task.detached(priority: .utility) {
            let result: Result<VectorDedupStats, Error>
            do {
                result = .success(try agent.ragEngine.deduplicateVectors())
            } catch {
                result = .failure(error)
            }
            await MainActor.run { [weak self] in
                guard let self else { return }
                self.isDeduplicating = false
                if let current = self.agent, ObjectIdentifier(current) == target {
                    self.refreshRAGStatus()
                }
                switch result {
                case .success(let stats):
                    self.appendLog("RAG: 向量去重完成 — \(stats.summary)")
                    self.statusText = "向量去重完成"
                    self.showInfoAlert(title: "向量去重完成", message: stats.summary + "。")
                case .failure(let error):
                    self.appendLog("RAG: 向量去重失败 — \(error.localizedDescription)")
                    self.statusText = "向量去重失败"
                    self.showInfoAlert(title: "向量去重失败", message: error.localizedDescription)
                }
                self.scheduleIdleUnload()
            }
        }
    }

    private func confirmVectorDedupe(_ preview: VectorDedupPreview) -> Bool {
        let alert = NSAlert()
        alert.messageText = "确认清理重复向量"
        alert.informativeText = """
        向量库共 \(preview.totalChunks) 条切片，检测到 \(preview.duplicateChunks) 条重复向量，涉及 \(preview.affectedSources) 个来源。
        每个重复分组只保留入库时间最早的一份，删除后不可恢复。
        """
        alert.alertStyle = .warning
        // 默认按钮是「取消」，避免误触删除。
        alert.addButton(withTitle: "取消")
        alert.addButton(withTitle: "清理")
        NSApp.activate(ignoringOtherApps: true)
        return alert.runModal() == .alertSecondButtonReturn
    }

    private func showInfoAlert(title: String, message: String) {
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = message
        alert.addButton(withTitle: "好")
        NSApp.activate(ignoringOtherApps: true)
        alert.runModal()
    }

    func syncKnowledgeBase(force: Bool) {
        let restore = pendingSyncRestore
        pendingSyncRestore = nil
        guard let agent, !isSyncing, !isDeduplicating else { return }
        cancelIdleUnload()
        if isRunning {
            appendLog("[rag] 对话进行中，跳过本次同步")
            if let restore, restore.main || restore.aux {
                Task { await resumeModelsAfterSync(main: restore.main, aux: restore.aux) }
            }
            return
        }
        // 同步已被接受：立即进入同步态，按钮马上锁定（「同步知识」变灰、
        // 「停止同步」可用），不存在模型加载等待/卸载期间的空窗。
        isSyncing = true
        isStoppingSync = false
        syncGeneration += 1
        statusText = force ? "强制同步知识库…" : "同步知识库…"

        // 启动加载/预热是短暂状态：等它结束再按「先卸载、后同步」执行。
        // 不要直接丢弃用户点击。最多等 5 分钟，避免异常卡死时无限挂起。
        if isModelLoading || isWarmingUp {
            appendLog("[rag] 模型正在加载/预热，完成后自动开始同步")
            pendingSyncRestore = restore
            let generation = syncGeneration
            Task { @MainActor [weak self] in
                for _ in 0..<3_000 {
                    guard let self else { return }
                    if !self.isModelLoading && !self.isWarmingUp { break }
                    // 等待期间用户点了「停止同步」：放弃本次同步。
                    if self.isStoppingSync { break }
                    try? await Task.sleep(nanoseconds: 100_000_000)
                }
                guard let self else { return }
                self.pendingSyncRestore = nil
                guard self.syncGeneration == generation else { return }
                guard !self.isStoppingSync else {
                    self.appendLog("[rag] 同步已在模型就绪前被停止")
                    self.isSyncing = false
                    self.isStoppingSync = false
                    self.syncTask = nil
                    if let restore, restore.main || restore.aux {
                        Task { await self.resumeModelsAfterSync(main: restore.main, aux: restore.aux) }
                    }
                    return
                }
                if self.isModelLoading || self.isWarmingUp {
                    self.appendLog("[rag] 模型加载/预热仍未结束，取消本次同步")
                    self.isSyncing = false
                    self.isStoppingSync = false
                    self.syncTask = nil
                    if let restore, restore.main || restore.aux {
                        Task { await self.resumeModelsAfterSync(main: restore.main, aux: restore.aux) }
                    }
                    return
                }
                self.startIngest(agent: agent, force: force,
                                 restore: restore, generation: generation)
            }
            return
        }
        startIngest(agent: agent, force: force, restore: restore, generation: syncGeneration)
    }

    /// 已处于同步态（isSyncing == true）：卸载本地模型并启动 ingest 任务。
    private func startIngest(agent: Agent, force: Bool,
                             restore: (main: Bool, aux: Bool)?, generation: Int) {
        // applyConfig 后的同步：模型尚未加载，按应用配置前的状态在同步结束后恢复。
        let wasMain = restore?.main ?? mainModelLoaded
        let wasAux = restore?.aux ?? auxModelLoaded
        appendLog("[rag] 同步开始：先卸载主/辅助模型，结束后重新加载并预热主模型")
        // 同步期间卸载已加载的主/辅模型，释放权重与 KV cache。
        suspendModelsForSync()

        statusText = force ? "强制同步知识库…" : "同步知识库…"
        let syncGen = generation
        syncTask = Task.detached(priority: .userInitiated) { [weak self, weak agent] in
            guard let agent else {
                await MainActor.run { [weak self] in
                    guard let self, self.syncGeneration == syncGen else { return }
                    self.isSyncing = false
                    self.isStoppingSync = false
                    self.syncTask = nil
                }
                return
            }
            // 给 MLX/GGUF 的异步卸载一点收尾时间，避免刚释放的权重与
            // RAG 的 embedding 模型抢内存。GGUF 是同步释放，这里几乎无感知。
            try? await Task.sleep(nanoseconds: 200_000_000)
            let stats = await agent.ragEngine.ingest(force: force) { current, total, file in
                Task { @MainActor [weak self] in
                    guard let self, self.syncGeneration == syncGen else { return }
                    self.statusText = "同步 \(current)/\(total) \(file)"
                }
            } log: { line in
                Task { @MainActor [weak self] in self?.appendLog(line) }
            }
            // 只重载实际执行同步的引擎：配置切换后 self.agent 可能已经是新实例，
            // 绝不能把旧同步的结果作用到新 agent 上。
            agent.reloadRAG()
            await MainActor.run { [weak self] in
                guard let self, self.syncGeneration == syncGen else { return }
                self.isSyncing = false
                self.isStoppingSync = false
                self.syncTask = nil
                self.appendLog("RAG: \(stats.summary)")
                for error in stats.errors.prefix(20) {
                    self.appendLog("RAG 错误: \(error)")
                }
                if self.agent === agent {
                    self.refreshRAGStatus()
                } else {
                    self.statusText = "就绪"
                }
                self.scheduleIdleUnload()
            }
            // 同步结束（成功/失败/取消）后按同步前状态恢复模型。
            let shouldRestore = await MainActor.run(body: { [weak self] in
                guard let self else { return false }
                return self.syncGeneration == syncGen && self.agent === agent
            })
            if shouldRestore {
                // 同步 Task 可能刚被「停止同步」取消；恢复必须放到新 Task，
                // 否则 warmup 中的 Task.checkCancellation 会立即抛错显示「预热失败」。
                Task.detached { [weak self] in
                    await self?.resumeModelsAfterSync(main: wasMain, aux: wasAux)
                }
            }
        }
    }

    /// 知识同步前卸载主/辅模型，释放权重、KV cache 与 MLX 缓冲。
    private func suspendModelsForSync() {
        // 只卸载本地可预加载模型；API 模式的 OpenAI 客户端不占本地权重，保持原样。
        let hasMain = mainModelLoaded || agent?.llm is PreloadableLLMService
        let hasAux = auxModelLoaded || agent?.auxLLM is PreloadableLLMService
        guard hasMain || hasAux else {
            appendLog("[model] 知识同步前模型未加载，无需卸载")
            return
        }
        if hasMain {
            agent?.switchLLM(UnloadedLLMService())
            appendLog("[model] 已卸载主模型，释放权重与 KV cache")
        }
        if hasAux {
            agent?.unloadAuxLLM()
            appendLog("[model] 已卸载辅助模型，释放权重与 KV cache")
        }
        LocalModelFactory.shared.unloadLLMCaches()
        mainModelLoaded = false
        auxModelLoaded = false
        appendLog("[model] 主/辅助模型卸载完成，开始知识同步")
    }

    /// 同步结束后恢复同步前已加载的模型；主模型会重新预热。
    private func resumeModelsAfterSync(main: Bool, aux: Bool) async {
        guard main || aux, let agent else { return }
        appendLog("[model] 知识同步结束：重新加载"
            + "\(main ? "主模型" : "")\(main && aux ? "与" : "")\(aux ? "辅助模型" : "")"
            + (main ? "，主模型加载后自动预热" : ""))
        statusText = "同步结束，重新加载模型…"
        if main, agent.llm is UnloadedLLMService {
            agent.switchLLM(LocalModelFactory.shared.makeLLM(config: config))
        }
        if aux, agent.auxLLM == nil {
            agent.auxLLM = LocalModelFactory.shared.makeAuxLLM(config: config)
        }
        await preloadModels(main: main, aux: aux)
    }

    func stopSync() {
        guard isSyncing, !isStoppingSync else { return }
        isStoppingSync = true
        statusText = "正在停止同步…"
        appendLog("用户请求停止知识库同步")
        // 内部标志让 RAG 在文件/embedding 批次边界退出；Task 取消让
        // SemanticChunker 等支持 Task.isCancelled 的步骤立即返回。
        agent?.ragEngine.cancel()
        syncTask?.cancel()
    }

    /// 缓存 RAG 统计，避免 SwiftUI body 每次重绘都查库。
    /// 查库可能触发建库 + COUNT/DISTINCT，必须放到后台，不能卡主线程。
    func refreshRAGStatus() {
        guard let agent else {
            ragStatusText = "RAG 未就绪"
            return
        }
        guard !ragStatusRefreshing else { return }
        ragStatusRefreshing = true
        // 用 ObjectIdentifier 判新旧实例，避免在并发闭包里捕获会变化的 agent 引用。
        let targetID = ObjectIdentifier(agent)
        let engine = agent.ragEngine
        Task.detached(priority: .utility) { [weak self] in
            let status = engine.status()
            let chunks = status["chunks_stored"]?.intValue ?? 0
            let backend = status["vector_backend"]?.stringValue ?? "sqlite"
            let sources = status["source_count"]?.intValue
                ?? status["sources"]?.arrayValue?.count ?? 0
            let text = "\(chunks) 切片 · \(sources) 文件 · \(backend)"
            await MainActor.run { [weak self] in
                guard let self else { return }
                self.ragStatusRefreshing = false
                if let current = self.agent,
                   ObjectIdentifier(current) == targetID {
                    self.ragStatusText = text
                } else {
                    // agent 已切换：丢弃旧结果，重新为新实例取一次状态。
                    self.refreshRAGStatus()
                }
            }
        }
    }
}
