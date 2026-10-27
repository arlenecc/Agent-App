import Foundation
import CryptoKit
import Darwin

// MARK: - MCP

public struct MCPServerConfig: Codable, Sendable, Equatable {
    public var name: String
    public var command: String
    public var args: [String]
    public var env: [String: String]

    public init(name: String, command: String, args: [String] = [], env: [String: String] = [:]) {
        self.name = name
        self.command = command
        self.args = args
        self.env = env
    }
}

// MARK: - PDF 解析引擎

/// PDF 文档解析引擎。docling / marker 已移除：
/// - ovis：OvisOCR3 视觉语言模型逐页转 Markdown
/// - vision：Apple Vision 逐页 OCR（默认，最快，无表格结构）
/// - hybrid：Vision 为主，仅表格页交给 OvisOCR3（扫描书籍最快）
/// - custom：使用「高级 RAG / OCR 设置」中自定义的 OCR 模型逐页转 Markdown
public enum PDFEngine: String, Codable, Sendable, CaseIterable {
    case ovis
    case vision
    case hybrid
    case custom

    public init(stored: String) {
        let v = stored.lowercased()
        if v == "vision" || v == "pdfkit_vision" || v == "apple_vision" {
            self = .vision
        } else if v == "ovis" || v == "ovisocr3" {
            self = .ovis
        } else if v == "hybrid" || v == "mixed" || v == "auto" {
            self = .hybrid
        } else if v == "custom" || v == "custom_vlm" {
            self = .custom
        } else {
            // 未显式配置 ovis / hybrid / custom 时一律先走 Apple Vision，
            // 避免旧值（如 docling）把 OCR 默默带回纯 Ovis 路径。
            self = .vision
        }
    }
}

// MARK: - AgentConfig

/// 运行时配置，持久化到 `~/.base-agent/config.json`。
/// 字段命名对齐原 Python `AgentConfig`（JSON 中为 snake_case）。
public struct AgentConfig: Sendable {
    public var baseURL: String = "https://api.openai.com/v1"
    public var apiKey: String = ""
    public var model: String = "gpt-4o-mini"
    public var temperature: Double = 0.7
    public var maxTokens: Int = 32768
    public var topP: Double = 0.95
    public var minP: Double = 0.05
    public var topK: Int = 20
    public var repetitionPenalty: Double = 1.0
    public var workspace: String = expandPath("~/base-agent-workspace")
    public var knowledgeBase: String = ""
    public var requestTimeout: Double = 120.0
    public var maxIterations: Int = 15
    public var maxHistory: Int = 50
    public var maxContextTokens: Int = 32768
    /// prompt prefill 每次前向的 token 上限（0 = 使用 MLX 默认值）。
    public var promptBatchSize: Int = 1024
    /// MLX 空闲 buffer cache 上限（MB）。只限制可复用 buffer，不含模型权重与 KV；
    /// 运行时夹在 256...1024，默认 512。
    public var mlxCacheLimitMB: Int = 512
    /// MLX 全局内存硬上限（MB，含权重/KV/buffer）；0 = 自动取
    /// min(物理内存 65%, 9.6GB)。
    public var mlxMemoryLimitMB: Int = 0
    /// 主模型 KV cache 的 token 硬上限；0 表示跟随 maxContextTokens。
    /// 32k 约对应 1GB KV（Qwen3.5 只有 1/4 层是全注意力）。
    public var kvCacheMaxTokens: Int = 32_768
    /// 主 LLM 跨轮 KV/前缀复用开关。开启时 system+tools 等不变前缀的 KV
    /// 跨轮保留，每轮只 prefill 新增消息；关闭时每轮全量重建。
    /// 同时作用于 MLX 与 GGUF (llama.cpp) 后端；辅助模型不受影响（其缓存
    /// 按约定每次后台任务后即失效）。
    public var llmKVReuse: Bool = true
    /// 辅助模型 KV cache 的 token 硬上限；0 表示跟随 auxMaxContextTokens。
    public var auxKvCacheMaxTokens: Int = 16_384
    /// KV cache int8 量化（MLX 用 kvBits=8 的 QuantizedKVCache，GGUF 用
    /// ggml q8_0 KV）。q8_0 对生成质量几乎无损，KV 内存约减半；
    /// 同时作用于 MLX 与 GGUF 后端的主模型和辅助模型。
    public var kvCacheInt8: Bool = true
    public var contextShrinkRatio: Double = 0.9

    /// 实际生效的主模型上下文：min(配置上下文, KV 硬上限)。
    public var effectiveMainContextTokens: Int {
        let cap = kvCacheMaxTokens > 0 ? kvCacheMaxTokens : maxContextTokens
        // 下限 8k：system + 工具 schema 已约 4.7k，再低会每轮都撑爆 prompt。
        return max(8_192, min(maxContextTokens, cap))
    }

    /// 实际生效的辅助模型上下文：min(配置上下文, KV 硬上限)。
    public var effectiveAuxContextTokens: Int {
        let cap = auxKvCacheMaxTokens > 0 ? auxKvCacheMaxTokens : auxMaxContextTokens
        return max(4_096, min(auxMaxContextTokens, cap))
    }

    // 内核：三态自动机 + 五级压缩
    public var kernelMaxContextTokens: Int = 30_000
    public var kernelRawTurns: Int = 2
    public var kernelConsolidateEvery: Int = 4
    public var kernelMaxToolChars: Int = 6_000
    public var kernelMaxDigestLines: Int = 40

    // RAG
    /// 切片大小（token，含 overlap）。默认 500：bge-small-zh 上下文 512 token，
    /// 留出 [CLS]/[SEP] 与估算误差的余量。实测纯中文约 1 字 ≈ 1 token，
    /// 中英混排约 1.8 字符/token（500 token ≈ 600-900 字）。
    public var ragChunkSize: Int = 500
    /// 默认 chunk_size 的 10%（token）。计入 chunk_size 总预算，
    /// 正文按 `chunk_size - overlap` 切片，拼好后仍不超过 500 token。
    public var ragChunkOverlap: Int = 50
    /// 语义切片开关：开启时大段落在句间相似度低谷处断开（切片阶段需把
    /// 每个候选句嵌入一遍，显著慢于纯结构切片）；关闭时按标题结构 +
    /// 字符边界递归切片，速度快一个数量级，检索质量损失有限。
    public var ragSemanticChunking: Bool = true
    /// 默认 bge-small-zh-v1.5（MLX fp16）：中文检索全面优于 nomic
    /// （实测 Recall@1 9/10 vs 5/10、吞吐 8-10 倍），query 侧自动加官方检索指令。
    public var ragEmbeddingModel: String = "bge-small-zh-v1.5-mlx"
    /// 默认 ONNX int8 版 jina reranker：CPU 推理吞吐约为 MLX mxfp8 的 2 倍，
    /// 1024 token 上下文。实测中文重排分数整体偏低（目标块 0.25-0.68），
    /// 需配合 ragRerankThreshold=0.45 使用（原 0.7 阈值会滤掉全部结果）。
    /// 追求排序质量可改回 bge-reranker-v2-m3-mlx-mxfp8（MLX 后端）并上调阈值。
    public var ragRerankModel: String = "jina-reranker-v2-base-multilingual-onnx"
    /// 重排分数过滤阈值（0-1）。全部低于阈值时回退原排序 topK，不会空结果。
    public var ragRerankThreshold: Double = 0.45
    public var ragRerankEnabled: Bool = true
    /// 检索结果格式化时单条片段的字符上限（超出截断加省略号）。
    public var ragSearchMaxCharsPerResult: Int = 1_500
    /// 检索结果总字符上限（超出截断）。
    public var ragSearchMaxTotalChars: Int = 12_000
    /// 辅助模型精简触发下限：检索结果总字数低于该值时原样返回，
    /// 不调用辅助模型（内容不多时总结只会丢细节、平白增加延迟）。
    public var ragAuxSummarizeMinChars: Int = 4_000
    /// 辅助模型精简触发上限：超过该值的结果不做精简
    ///（内容过大时一次性摘要不可靠）。
    public var ragAuxSummarizeMaxChars: Int = 12_000
    /// 启动时自动同步知识库。默认关闭：首次全量同步会长时间占用
    /// embedding 模型与磁盘 IO，容易与对话争抢资源；需要时手动触发。
    public var ragAutoIngest: Bool = false
    public var ragPDFEngine: PDFEngine = .vision
    /// 扫描版 EPUB 的 OCR 后端：vision（Apple Vision）/ ovis（OvisOCR3）。
    /// 两条路径都会先做统一图像归一化。
    public var ragEPUBOCREngine: PDFEngine = .vision
    public var ragOCREngine: String = "mlx-vlm"
    public var ragOCRModelName: String = "OvisOCR3-MLX-4bit-mixed46"
    public var ragOCRModelPath: String = "models/OvisOCR3-MLX-4bit-mixed46"
    /// GGUF 视觉模型的多模态投影文件；留空时自动查找模型同目录的 mmproj*.gguf。
    public var ragOCRMMProjPath: String = ""
    /// 预留：OCR ONNX 模型的 tokenizer 目录（ONNX VLM 尚未接入）。
    public var ragOCRTokenizerPath: String = ""
    /// 预留：主模型 GGUF 的 mmproj 路径（当前主/辅 LLM 为纯文本，未接入视觉输入）。
    public var llmMmprojPath: String = ""
    /// 主模型 ONNX 的 tokenizer 目录；留空时从模型目录读取。
    public var llmTokenizerPath: String = ""
    /// 预留：辅助模型 GGUF 的 mmproj 路径（当前主/辅 LLM 为纯文本，未接入视觉输入）。
    public var auxMmprojPath: String = ""
    /// 辅助模型 ONNX 的 tokenizer 目录；留空时从模型目录读取。
    public var auxTokenizerPath: String = ""
    public var ragOCRCoreML: Bool = false
    public var ragOCRPreprocess: Bool = true
    public var ragOCRBinarize: Bool = true
    /// llama.cpp 的 GPU 层数（-ngl）；内存紧张时可设为 0 改用 CPU。
    public var llamaGpuLayers: Int = 99
    /// 同时处理的文件数（1...10）。解析/OCR 与 embedding 流水线并行；
    /// 纯文本文件不经过 OCR 闸门，多 worker 可把等待空档填满。
    public var ragIngestConcurrency: Int = 10
    /// Apple Vision OCR 的全局并发闸门（跨文件共享）。
    /// 实测 4 路已接近吞吐拐点，再多不一定快甚至变慢；0 也按 4 处理，
    /// 保留覆盖入口便于实验。
    public var ragVisionConcurrency: Int = 4
    /// OvisOCR3 并行实例数（1...2）。4bit 权重约 0.7GB，单实例运行约 1GB，
    /// 第二实例懒加载（仅在真正并发时才占内存），可提高扫描页吞吐。
    public var ragOvisInstances: Int = 2

    // Embedding 子批自适应（AdaptiveEmbeddingScheduler）
    /// 子批下限（2...64），默认 2；运行时还会夹到不超过上限。
    public var ragEmbeddingMinBatch: Int = 2
    /// 子批上限（2...256）。默认 32：实测 477 token 长块在 4-16 已达吞吐平台，
    /// 32 兼顾短块；64 以上只增加内存与单批延迟。
    public var ragEmbeddingMaxBatch: Int = 32
    /// 每次同步的起始子批（2...256），默认 16；运行时夹到 [下限, 上限]。
    public var ragEmbeddingInitialBatch: Int = 16
    /// 每轮上探/下探的步进（1...16），默认 2。
    public var ragEmbeddingBatchStep: Int = 2
    /// 子批调参的进程 footprint 上限（MB，1024...32768），默认 6144。
    /// 超过该值立即下调子批；bge 单独运行时峰值约 1.6GB。
    public var ragEmbeddingMemoryLimitMB: Int = 6_144
    /// 吞吐变化容差（0.01...0.5），默认 0.03。变化小于该比例视为测量噪声，
    /// 不调整子批；实测同 batch 重复测量的噪声可达 ±20%，可调到 0.08-0.10。
    public var ragEmbeddingThroughputTolerance: Double = 0.03
    /// 每档至少观察的批次数（1...16，默认 2），与 probeSeconds 满足其一即评估。
    public var ragEmbeddingProbeBatches: Int = 2
    /// 每档至少观察的秒数（0.5...30，默认 2）。
    public var ragEmbeddingProbeSeconds: Double = 2.0

    // SQLite + sqlite-vec 存储
    public var ragStorageBackend: String = "sqlite"
    public var ragSQLiteVecEnabled: Bool = true
    /// 混合检索 RRF 常数（原项目 RRF k=60）。
    public var ragRRFK: Int = 60
    /// 向量召回 over-fetch 倍数（先多取再过滤，避免 join 虚拟表）。
    public var ragVectorOverfetch: Int = 4
    /// FTS5 召回 over-fetch 倍数。
    public var ragFTSOverfetch: Int = 4

    // 本地模型 provider
    // 模型类型统一使用 mlx-lm / mlx-vlm / onnx / gguf；旧值 mlx 迁移为 mlx-lm。
    public var llmProvider: String = "local"
    public var localLLMModel: String = "models/qwen3.5-4b-agentic-coder-v4-4bit-mlx"
    public var localLLMBackend: String = "mlx-lm"
    public var localLLMEnableThinking: Bool = true
    /// 启动时是否预加载主模型。默认开启；关闭后启动只创建 Agent，不加载权重，
    /// 需在配置窗口手动「加载模型」。不影响配置应用/同步结束后的模型恢复。
    public var autoPreloadMain: Bool = true

    /// 是否启用辅助模型。默认关闭：启动和知识同步前后都不加载辅助模型；
    /// RAG 精简直接跳过，后台记忆/总结回退主模型（若主模型可用）。
    public var auxEnabled: Bool = false
    public var auxLLMProvider: String = "local"
    public var auxLocalLLMModel: String = "models/Qwen3.5-2B-MLX-4bit"
    public var auxLocalLLMBackend: String = "mlx-lm"
    public var auxTemperature: Double = 0.7
    public var auxMaxTokens: Int = 32768
    public var auxMaxContextTokens: Int = 32768
    public var auxPromptBatchSize: Int = 512
    public var auxTopP: Double = 0.95
    public var auxMinP: Double = 0.05
    public var auxTopK: Int = 20
    public var auxRepetitionPenalty: Double = 1.0
    public var auxRequestTimeout: Double = 120.0
    public var auxLocalLLMEnableThinking: Bool = false

    public var embeddingProvider: String = "local"
    public var localEmbeddingModel: String = "bge-small-zh-v1.5-mlx"
    /// 显式本地目录；非空时优先于模型名。
    /// 使用相对模型根目录的路径，由 LocalModelRegistry 统一解析。
    public var localEmbeddingModelPath: String = "models/bge-small-zh-v1.5-mlx"
    /// onnx / fastembed / mlx / gguf
    /// Embedding 后端：mlx（Metal GPU，默认，使用本地 8bit 量化权重）
    /// / onnx（CPU int8）/ fastembed / gguf。
    public var localEmbeddingBackend: String = "mlx"

    public var rerankerProvider: String = "local"
    public var localRerankerModel: String = "jina-reranker-v2-base-multilingual-onnx"
    public var localRerankerModelPath: String = "models/jina-reranker-v2-base-multilingual-onnx"
    /// onnx / fastembed / mlx / gguf
    /// 默认 ONNX：与默认模型 jina-reranker-v2-base-multilingual-onnx 的权重格式一致。
    public var localRerankerBackend: String = "onnx"

    public var vlmProvider: String = "api"
    public var localVLMModel: String = ""
    public var localVLMBackend: String = "mlx"

    public var browserEndpoint: String = ""
    public var alwaysConfirmCode: Bool = false
    public var mcpServers: [MCPServerConfig] = []

    public init() {}

    // MARK: Paths

    public var resolvedKnowledgeBase: String {
        knowledgeBase.isEmpty
            ? (workspace as NSString).appendingPathComponent("knowledge_base")
            : expandPath(knowledgeBase)
    }

    /// RAG 索引与产物目录：按知识库目录隔离（`rag_data/<目录名-路径哈希>/`）。
    /// 历史版本所有知识库共用 `rag_data` 根目录，切换知识库目录后互相把对方
    /// 的条目当「已删除文件」清掉，导致增量同步失效；按知识库隔离后各自
    /// 保留 manifest 与索引。知识库目录是只读的原始文件目录，绝不向其写入。
    public var ragIndexDirectory: String {
        Self.ragIndexDirectory(forKB: resolvedKnowledgeBase, workspace: workspace)
    }

    /// 按知识库目录推导索引子目录：真实路径规范化 + 目录名 + 绝对路径哈希，
    /// 保证同名不同位置的目录各自隔离，同一路径多次推导结果稳定。
    public static func ragIndexDirectory(forKB kb: String, workspace: String) -> String {
        let base = (workspace as NSString).appendingPathComponent("rag_data")
        let canonical = canonicalizePath(kb)
        let name = (canonical as NSString).lastPathComponent
        let digest = SHA256.hash(data: Data(canonical.utf8))
        let short = digest.prefix(4).map { String(format: "%02x", $0) }.joined()
        return (base as NSString).appendingPathComponent("\(name.isEmpty ? "kb" : name)-\(short)")
    }

    /// `realpath` 规范化：解析 `/var` → `/private/var` 与其它符号链接。
    public static func canonicalizePath(_ path: String) -> String {
        let expanded = (path as NSString).expandingTildeInPath
        guard let resolved = realpath(expanded, nil) else {
            return URL(fileURLWithPath: expanded).standardizedFileURL.path
        }
        defer { free(resolved) }
        return String(cString: resolved)
    }

    public var dbPath: String {
        (ragIndexDirectory as NSString).appendingPathComponent("index.sqlite")
    }

    public var modelsDirectory: String {
        (workspace as NSString).appendingPathComponent("models")
    }

    @discardableResult
    public func ensureWorkspace() -> String {
        try? FileManager.default.createDirectory(atPath: workspace, withIntermediateDirectories: true)
        return workspace
    }

    // MARK: Load / Save

    public static func load(_ path: String) -> AgentConfig {
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: path) else { return AgentConfig() }
        guard let data = fileManager.contents(atPath: path),
              let raw = try? JSONDecoder().decode(JSONValue.self, from: data),
              let dict = raw.objectValue else {
            // 配置文件损坏时备份后使用默认值启动。
            let backup = path + ".corrupt"
            try? fileManager.removeItem(atPath: backup)
            try? fileManager.moveItem(atPath: path, toPath: backup)
            AgentLog.warning("config 无法解析，已备份到 \(backup)，使用默认配置")
            return AgentConfig()
        }

        var cfg = AgentConfig()
        cfg.apply(dict)
        cfg.migrate()
        return cfg
    }

    private mutating func apply(_ d: [String: JSONValue]) {
        func string(_ key: String) -> String? {
            guard let v = d[key] else { return nil }
            return v.stringValue
        }
        func int(_ key: String) -> Int? {
            guard let v = d[key] else { return nil }
            return v.intValue
        }
        func double(_ key: String) -> Double? {
            guard let v = d[key] else { return nil }
            return v.doubleValue
        }
        func bool(_ key: String) -> Bool? {
            guard let v = d[key] else { return nil }
            return v.boolValue
        }

        if let v = string("base_url") { baseURL = v }
        if let v = string("api_key") { apiKey = v }
        if let v = string("model") { model = v }
        if let v = double("temperature") { temperature = v }
        if let v = int("max_tokens") { maxTokens = v }
        if let v = double("top_p") { topP = v }
        if let v = double("min_p") { minP = v }
        if let v = int("top_k") { topK = v }
        if let v = double("repetition_penalty") { repetitionPenalty = v }
        if let v = string("workspace"), !v.isEmpty { workspace = expandPath(v) }
        if let v = string("knowledge_base") { knowledgeBase = v }
        if let v = double("request_timeout") { requestTimeout = v }
        if let v = int("max_iterations") { maxIterations = v }
        if let v = int("max_history") { maxHistory = v }
        if let v = int("max_context_tokens") { maxContextTokens = v }
        if let v = double("context_shrink_ratio") { contextShrinkRatio = v }

        if let v = int("prompt_batch_size") { promptBatchSize = v }
        if let v = int("mlx_cache_limit_mb") { mlxCacheLimitMB = v }
        if let v = int("mlx_memory_limit_mb") { mlxMemoryLimitMB = v }
        if let v = int("kv_cache_max_tokens") { kvCacheMaxTokens = v }
        if let v = bool("llm_kv_reuse") { llmKVReuse = v }
        if let v = bool("kv_cache_int8") { kvCacheInt8 = v }
        if let v = int("aux_kv_cache_max_tokens") { auxKvCacheMaxTokens = v }
        if let v = int("kernel_max_context_tokens") { kernelMaxContextTokens = v }
        if let v = int("kernel_raw_turns") { kernelRawTurns = v }
        if let v = int("kernel_consolidate_every") { kernelConsolidateEvery = v }
        if let v = int("kernel_max_tool_chars") { kernelMaxToolChars = v }
        if let v = int("kernel_max_digest_lines") { kernelMaxDigestLines = v }

        if let v = int("rag_chunk_size") { ragChunkSize = v }
        if let v = int("rag_chunk_overlap") { ragChunkOverlap = v }
        if let v = bool("rag_semantic_chunking") { ragSemanticChunking = v }
        if let v = int("rag_ingest_concurrency") {
            ragIngestConcurrency = min(max(1, v), 10)
        }
        if let v = int("rag_vision_concurrency") { ragVisionConcurrency = max(0, v) }
        if let v = int("rag_ovis_instances") { ragOvisInstances = min(max(1, v), 2) }
        if let v = int("rag_embedding_min_batch") { ragEmbeddingMinBatch = v }
        if let v = int("rag_embedding_max_batch") { ragEmbeddingMaxBatch = v }
        if let v = int("rag_embedding_initial_batch") { ragEmbeddingInitialBatch = v }
        if let v = int("rag_embedding_batch_step") { ragEmbeddingBatchStep = v }
        if let v = int("rag_embedding_memory_limit_mb") { ragEmbeddingMemoryLimitMB = v }
        if let v = double("rag_embedding_throughput_tolerance") {
            ragEmbeddingThroughputTolerance = v
        }
        if let v = int("rag_embedding_probe_batches") { ragEmbeddingProbeBatches = v }
        if let v = double("rag_embedding_probe_seconds") { ragEmbeddingProbeSeconds = v }
        if let v = string("rag_embedding_model") { ragEmbeddingModel = v }
        if let v = string("rag_rerank_model") { ragRerankModel = v }
        if let v = bool("rag_rerank_enabled") { ragRerankEnabled = v }
        if let v = bool("rag_auto_ingest") { ragAutoIngest = v }
        if let v = string("rag_pdf_engine") { ragPDFEngine = PDFEngine(stored: v) }
        if let v = string("rag_epub_ocr_engine") { ragEPUBOCREngine = PDFEngine(stored: v) }
        if let v = string("rag_ocr_engine") { ragOCREngine = v }
        if let v = string("rag_ocr_model_name") { ragOCRModelName = v }
        if let v = string("rag_ocr_model_path") { ragOCRModelPath = v }
        if let v = string("rag_ocr_mmproj_path") { ragOCRMMProjPath = v }
        if let v = string("rag_ocr_tokenizer_path") { ragOCRTokenizerPath = v }
        if let v = string("llm_mmproj_path") { llmMmprojPath = v }
        if let v = string("llm_tokenizer_path") { llmTokenizerPath = v }
        if let v = string("aux_mmproj_path") { auxMmprojPath = v }
        if let v = string("aux_tokenizer_path") { auxTokenizerPath = v }
        if let v = bool("rag_ocr_coreml") { ragOCRCoreML = v }
        if let v = bool("rag_ocr_preprocess") { ragOCRPreprocess = v }
        if let v = bool("rag_ocr_binarize") { ragOCRBinarize = v }
        if let v = int("llama_gpu_layers") { llamaGpuLayers = max(0, v) }

        if let v = string("rag_storage_backend") { ragStorageBackend = v }
        if let v = bool("rag_sqlite_vec_enabled") { ragSQLiteVecEnabled = v }
        if let v = int("rag_rrf_k") { ragRRFK = v }
        if let v = double("rag_rerank_threshold") { ragRerankThreshold = v }
        if let v = int("rag_search_max_chars_per_result") { ragSearchMaxCharsPerResult = v }
        if let v = int("rag_search_max_total_chars") { ragSearchMaxTotalChars = v }
        if let v = int("rag_aux_summarize_min_chars") { ragAuxSummarizeMinChars = v }
        if let v = int("rag_aux_summarize_max_chars") { ragAuxSummarizeMaxChars = v }
        if let v = int("rag_vector_overfetch") { ragVectorOverfetch = v }
        if let v = int("rag_fts_overfetch") { ragFTSOverfetch = v }

        if let v = string("llm_provider") { llmProvider = v }
        if let v = string("local_llm_model") { localLLMModel = v }
        if let v = string("local_llm_backend") { localLLMBackend = v }
        if let v = bool("local_llm_enable_thinking") { localLLMEnableThinking = v }
        if let v = bool("auto_preload_main") { autoPreloadMain = v }

        if let v = bool("aux_enabled") { auxEnabled = v }
        if let v = string("aux_llm_provider") { auxLLMProvider = v }
        if let v = string("aux_local_llm_model") { auxLocalLLMModel = v }
        if let v = string("aux_local_llm_backend") { auxLocalLLMBackend = v }
        if let v = double("aux_temperature") { auxTemperature = v }
        if let v = int("aux_max_tokens") { auxMaxTokens = v }
        if let v = double("aux_top_p") { auxTopP = v }
        if let v = double("aux_min_p") { auxMinP = v }
        if let v = int("aux_top_k") { auxTopK = v }
        if let v = double("aux_repetition_penalty") { auxRepetitionPenalty = v }
        if let v = double("aux_request_timeout") { auxRequestTimeout = v }
        if let v = bool("aux_local_llm_enable_thinking") { auxLocalLLMEnableThinking = v }
        if let v = int("aux_max_context_tokens") { auxMaxContextTokens = v }
        if let v = int("aux_prompt_batch_size") { auxPromptBatchSize = v }

        if let v = string("embedding_provider") { embeddingProvider = v }
        if let v = string("local_embedding_model") { localEmbeddingModel = v }
        if let v = string("local_embedding_model_path") { localEmbeddingModelPath = v }
        if let v = string("local_embedding_backend") { localEmbeddingBackend = v }
        if let v = string("reranker_provider") { rerankerProvider = v }
        if let v = string("local_reranker_model") { localRerankerModel = v }
        if let v = string("local_reranker_model_path") { localRerankerModelPath = v }
        if let v = string("local_reranker_backend") { localRerankerBackend = v }
        if let v = string("vlm_provider") { vlmProvider = v }
        if let v = string("local_vlm_model") { localVLMModel = v }
        if let v = string("local_vlm_backend") { localVLMBackend = v }

        // 兼容其他版本/参考实现的字段命名（仅当本版本规范键缺失时生效）。
        if d["llm_provider"] == nil, let v = string("llm_mode") {
            llmProvider = v
        }
        if d["local_llm_backend"] == nil, let v = string("llm_backend") {
            localLLMBackend = v
        }
        if d["local_llm_model"] == nil, let v = string("llm_model_path") {
            localLLMModel = v
        }
        if d["local_llm_enable_thinking"] == nil,
           let v = bool("llm_enable_thinking") {
            localLLMEnableThinking = v
        }
        if d["aux_local_llm_backend"] == nil, let v = string("aux_llm_backend") {
            auxLocalLLMBackend = v
        }
        if d["aux_local_llm_model"] == nil, let v = string("aux_llm_model_path") {
            auxLocalLLMModel = v
        }
        if d["aux_local_llm_enable_thinking"] == nil,
           let v = bool("aux_llm_enable_thinking") {
            auxLocalLLMEnableThinking = v
        }
        if d["local_embedding_backend"] == nil,
           let v = string("rag_embedding_backend") {
            localEmbeddingBackend = v
        }
        if d["local_embedding_model_path"] == nil,
           let v = string("rag_embedding_model_path") {
            localEmbeddingModelPath = v
        }
        if d["local_reranker_backend"] == nil,
           let v = string("rag_rerank_backend") {
            localRerankerBackend = v
        }
        // 旧配置为 MLX / ONNX 各保存一份路径；按当前类型选择对应路径，避免 MLX
        // 后端误用 ONNX 文件（反之亦然）。
        if d["local_reranker_model_path"] == nil {
            if Self.normalizeEmbeddingBackend(localRerankerBackend) == "mlx",
               let v = string("rag_rerank_mlx_model_path"), !v.isEmpty {
                localRerankerModelPath = v
            } else if let v = string("rag_rerank_model_path") {
                localRerankerModelPath = v
            }
        }

        if let v = string("browser_endpoint") { browserEndpoint = v }
        if let v = bool("always_confirm_code") { alwaysConfirmCode = v }
        if let servers = d["mcp_servers"]?.arrayValue {
            mcpServers = servers.compactMap { item in
                guard let o = item.objectValue,
                      let name = o["name"]?.stringValue,
                      let command = o["command"]?.stringValue else { return nil }
                let args = o["args"]?.arrayValue?.compactMap(\.stringValue) ?? []
                var env: [String: String] = [:]
                if let e = o["env"]?.objectValue {
                    env = e.compactMapValues(\.stringValue)
                }
                return MCPServerConfig(name: name, command: command, args: args, env: env)
            }
        }
    }

    /// 旧版本配置迁移（docling/marker → ovis、旧 embedding/reranker 名）。
    public mutating func migrate() {
        // docling/marker 已移除：旧配置值统一由 PDFEngine(stored:) 归一化为 ovis。
        let deprecatedEmbedding: Set<String> = [
            "all-MiniLM-L6-v2",
            "sentence-transformers/all-MiniLM-L6-v2",
            "BAAI/bge-small-zh-v1.5",
            "BAAI/bge-large-zh-v1.5",
            "",
        ]
        if deprecatedEmbedding.contains(ragEmbeddingModel) ||
            ragEmbeddingModel == "nomic-ai/nomic-embed-text-v1.5" {
            ragEmbeddingModel = "nomic-ai/nomic-embed-text-v1.5-Q"
        }
        // 旧 MLX fp16 embedding 名称/路径迁移到 8bit 量化版（167MB，实测与
        // fp16 余弦相似度 >0.999）。迁移为目录名而非绝对路径，由
        // LocalModelRegistry 在 workspace/models 下解析。
        let legacyMLXEmbeddingName = "nomic-ai/nomic-embed-text-v1.5-mlx"
        let mlx8bitName = "nomic-ai/nomic-embed-text-v1.5-mlx-8bit"
        if ragEmbeddingModel == legacyMLXEmbeddingName {
            ragEmbeddingModel = mlx8bitName
        }
        if localEmbeddingModel == legacyMLXEmbeddingName {
            localEmbeddingModel = mlx8bitName
        }
        if localEmbeddingModelPath == "nomic-embed-text-v1.5-mlx"
            || localEmbeddingModelPath.hasSuffix("/nomic-embed-text-v1.5-mlx") {
            localEmbeddingModelPath = mlx8bitName
        }
        // 旧 reranker 默认（HF 名）迁移到本地 MLX mxfp8 目录名。
        if ragRerankModel == "BAAI/bge-reranker-v2-m3"
            || ragRerankModel == "BAAI/bge-reranker-v2-m3-int8" {
            ragRerankModel = "bge-reranker-v2-m3-mlx-mxfp8"
        }
        // 上一版默认（bge mxfp8 + MLX）迁移到 jina ONNX：仅当模型与后端
        // 都恰好等于旧默认时才迁移，用户显式选择的其他组合保持不变。
        let legacyRerankerName = "bge-reranker-v2-m3-mlx-mxfp8"
        if localRerankerBackend == "mlx",
           localRerankerModel == legacyRerankerName,
           localRerankerModelPath.isEmpty
            || localRerankerModelPath == "models/\(legacyRerankerName)"
            || localRerankerModelPath == legacyRerankerName
            || localRerankerModelPath.hasSuffix("/\(legacyRerankerName)") {
            localRerankerBackend = "onnx"
            localRerankerModel = "jina-reranker-v2-base-multilingual-onnx"
            localRerankerModelPath = "models/jina-reranker-v2-base-multilingual-onnx"
            if ragRerankModel == legacyRerankerName {
                ragRerankModel = "jina-reranker-v2-base-multilingual-onnx"
            }
        }
        // 上一版默认（nomic 8bit）迁移到 bge-small-zh（MLX fp16）：中文检索
        // 质量与速度全面占优。仅迁移恰好等于旧默认名/路径的配置。
        let bgeName = "bge-small-zh-v1.5-mlx"
        for legacy in ["nomic-embed-text-v1.5-mlx-8bit",
                       "nomic-ai/nomic-embed-text-v1.5-mlx-8bit"] {
            if ragEmbeddingModel == legacy { ragEmbeddingModel = bgeName }
            if localEmbeddingModel == legacy { localEmbeddingModel = bgeName }
        }
        if localEmbeddingModelPath == "models/nomic-embed-text-v1.5-mlx-8bit"
            || localEmbeddingModelPath == "nomic-embed-text-v1.5-mlx-8bit"
            || localEmbeddingModelPath.hasSuffix("/nomic-embed-text-v1.5-mlx-8bit") {
            localEmbeddingModelPath = "models/\(bgeName)"
        }
        if maxTokens == 4096 { maxTokens = 32768 }
        if maxContextTokens < 8192 { maxContextTokens = maxTokens }
        // 旧默认迁移到 500/50（10%）：800/40 是上一版默认，800/80 更早。
        // 只迁移「恰好等于旧默认值」的组合，用户自定义值保持不变。
        if ragChunkSize == 800, ragChunkOverlap == 40 {
            ragChunkSize = 500
            ragChunkOverlap = 50
        } else if ragChunkSize == 800, ragChunkOverlap == 80 {
            ragChunkSize = 500
            ragChunkOverlap = 50
        }
        if ragChunkOverlap >= ragChunkSize {
            ragChunkOverlap = max(0, ragChunkSize / 10)
        }
        if ragStorageBackend != "sqlite" { ragStorageBackend = "sqlite" }

        // 旧配置使用 openai / openai_compatible 表示 API 模式。
        let provider = llmProvider.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if provider == "openai" || provider == "openai_compatible" || provider == "openai-compatible" {
            llmProvider = "api"
        }
        let auxProvider = auxLLMProvider.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if auxProvider == "openai" || auxProvider == "openai_compatible" || auxProvider == "openai-compatible" {
            auxLLMProvider = "api"
        }

        localLLMBackend = Self.normalizeLLMBackend(localLLMBackend)
        auxLocalLLMBackend = Self.normalizeLLMBackend(auxLocalLLMBackend)
        localEmbeddingBackend = Self.normalizeEmbeddingBackend(localEmbeddingBackend)
        localRerankerBackend = Self.normalizeEmbeddingBackend(localRerankerBackend)
        if auxMaxContextTokens < 8192 { auxMaxContextTokens = max(8192, auxMaxTokens) }
        if auxMaxContextTokens <= 0 { auxMaxContextTokens = 32768 }
        // MLX 空闲 buffer cache 运行时上限 1GB：旧配置可能残留 2048/4096，
        // 加载时一并压回，避免配置文件里的值与实际生效值不一致。
        mlxCacheLimitMB = max(256, min(1_024, mlxCacheLimitMB))
        // Embedding 子批自适应参数：手改 config.json 或旧配置可能越界。
        ragEmbeddingMaxBatch = max(2, min(256, ragEmbeddingMaxBatch))
        ragEmbeddingMinBatch = max(2, min(64, ragEmbeddingMinBatch))
        ragEmbeddingMinBatch = min(ragEmbeddingMinBatch, ragEmbeddingMaxBatch)
        ragEmbeddingInitialBatch = max(ragEmbeddingMinBatch,
                                       min(ragEmbeddingInitialBatch, ragEmbeddingMaxBatch))
        ragEmbeddingBatchStep = max(1, min(16, ragEmbeddingBatchStep))
        ragEmbeddingMemoryLimitMB = max(1_024, min(32_768, ragEmbeddingMemoryLimitMB))
        ragEmbeddingThroughputTolerance = max(0.01,
                                              min(0.5, ragEmbeddingThroughputTolerance))
        ragEmbeddingProbeBatches = max(1, min(16, ragEmbeddingProbeBatches))
        ragEmbeddingProbeSeconds = max(0.5, min(30, ragEmbeddingProbeSeconds))
    }

    /// 统一主/辅模型类型：mlx → mlx-lm；llama.cpp / llama_cpp → gguf。
    public static func normalizeLLMBackend(_ raw: String) -> String {
        switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "", "mlx", "mlx-lm", "mlx_lm":
            return "mlx-lm"
        case "mlx-vlm", "mlx_vlm":
            return "mlx-vlm"
        case "onnx":
            return "onnx"
        case "gguf", "llama_cpp", "llama.cpp", "llama-cpp":
            return "gguf"
        default:
            return raw.lowercased()
        }
    }

    /// 统一 Embedding / Reranker 类型：onnx / fastembed / mlx / gguf。
    public static func normalizeEmbeddingBackend(_ raw: String) -> String {
        switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "", "onnx":
            return "onnx"
        case "fastembed":
            return "fastembed"
        case "mlx", "mlx_lm", "mlx-lm":
            return "mlx"
        case "gguf", "llama_cpp", "llama.cpp", "llama-cpp":
            return "gguf"
        default:
            return raw.lowercased()
        }
    }

    /// 导出全部配置为字典（save 落盘与测试断言共用）。
    public func export() -> [String: Any] {
        return [
            "base_url": baseURL, "api_key": apiKey, "model": model,
            "temperature": temperature, "max_tokens": maxTokens,
            "top_p": topP, "min_p": minP, "top_k": topK,
            "repetition_penalty": repetitionPenalty,
            "workspace": workspace, "knowledge_base": knowledgeBase,
            "request_timeout": requestTimeout, "max_iterations": maxIterations,
            "max_history": maxHistory, "max_context_tokens": maxContextTokens,
            "context_shrink_ratio": contextShrinkRatio,
            "prompt_batch_size": promptBatchSize,
            "mlx_cache_limit_mb": mlxCacheLimitMB,
            "mlx_memory_limit_mb": mlxMemoryLimitMB,
            "kv_cache_max_tokens": kvCacheMaxTokens,
            "llm_kv_reuse": llmKVReuse,
            "kv_cache_int8": kvCacheInt8,
            "aux_kv_cache_max_tokens": auxKvCacheMaxTokens,
            "kernel_max_context_tokens": kernelMaxContextTokens,
            "kernel_raw_turns": kernelRawTurns,
            "kernel_consolidate_every": kernelConsolidateEvery,
            "kernel_max_tool_chars": kernelMaxToolChars,
            "kernel_max_digest_lines": kernelMaxDigestLines,
            "rag_chunk_size": ragChunkSize, "rag_chunk_overlap": ragChunkOverlap,
            "rag_semantic_chunking": ragSemanticChunking,
            "rag_ingest_concurrency": ragIngestConcurrency,
            "rag_vision_concurrency": ragVisionConcurrency,
            "rag_ovis_instances": ragOvisInstances,
            "rag_embedding_min_batch": ragEmbeddingMinBatch,
            "rag_embedding_max_batch": ragEmbeddingMaxBatch,
            "rag_embedding_initial_batch": ragEmbeddingInitialBatch,
            "rag_embedding_batch_step": ragEmbeddingBatchStep,
            "rag_embedding_memory_limit_mb": ragEmbeddingMemoryLimitMB,
            "rag_embedding_throughput_tolerance": ragEmbeddingThroughputTolerance,
            "rag_embedding_probe_batches": ragEmbeddingProbeBatches,
            "rag_embedding_probe_seconds": ragEmbeddingProbeSeconds,
            "rag_embedding_model": ragEmbeddingModel,
            "rag_rerank_model": ragRerankModel,
            "rag_rerank_enabled": ragRerankEnabled,
            "rag_auto_ingest": ragAutoIngest,
            "rag_pdf_engine": ragPDFEngine.rawValue,
            "rag_epub_ocr_engine": ragEPUBOCREngine.rawValue,
            "rag_ocr_engine": ragOCREngine,
            "rag_ocr_model_name": ragOCRModelName,
            "rag_ocr_model_path": ragOCRModelPath,
            "rag_ocr_mmproj_path": ragOCRMMProjPath,
            "rag_ocr_tokenizer_path": ragOCRTokenizerPath,
            "llm_mmproj_path": llmMmprojPath,
            "llm_tokenizer_path": llmTokenizerPath,
            "aux_mmproj_path": auxMmprojPath,
            "aux_tokenizer_path": auxTokenizerPath,
            "rag_ocr_coreml": ragOCRCoreML,
            "rag_ocr_preprocess": ragOCRPreprocess,
            "rag_ocr_binarize": ragOCRBinarize,
            "llama_gpu_layers": llamaGpuLayers,
            "rag_storage_backend": ragStorageBackend,
            "rag_sqlite_vec_enabled": ragSQLiteVecEnabled,
            "rag_rrf_k": ragRRFK,
            "rag_rerank_threshold": ragRerankThreshold,
            "rag_search_max_chars_per_result": ragSearchMaxCharsPerResult,
            "rag_search_max_total_chars": ragSearchMaxTotalChars,
            "rag_aux_summarize_min_chars": ragAuxSummarizeMinChars,
            "rag_aux_summarize_max_chars": ragAuxSummarizeMaxChars,
            "rag_vector_overfetch": ragVectorOverfetch,
            "rag_fts_overfetch": ragFTSOverfetch,
            "llm_provider": llmProvider,
            "local_llm_model": localLLMModel,
            "local_llm_backend": localLLMBackend,
            "local_llm_enable_thinking": localLLMEnableThinking,
            "auto_preload_main": autoPreloadMain,
            "aux_enabled": auxEnabled,
            "aux_llm_provider": auxLLMProvider,
            "aux_local_llm_model": auxLocalLLMModel,
            "aux_local_llm_backend": auxLocalLLMBackend,
            "aux_temperature": auxTemperature,
            "aux_max_tokens": auxMaxTokens,
            "aux_top_p": auxTopP,
            "aux_min_p": auxMinP,
            "aux_top_k": auxTopK,
            "aux_repetition_penalty": auxRepetitionPenalty,
            "aux_request_timeout": auxRequestTimeout,
            "aux_local_llm_enable_thinking": auxLocalLLMEnableThinking,
            "aux_max_context_tokens": auxMaxContextTokens,
            "aux_prompt_batch_size": auxPromptBatchSize,
            "embedding_provider": embeddingProvider,
            "local_embedding_model": localEmbeddingModel,
            "local_embedding_model_path": localEmbeddingModelPath,
            "local_embedding_backend": localEmbeddingBackend,
            "reranker_provider": rerankerProvider,
            "local_reranker_model": localRerankerModel,
            "local_reranker_model_path": localRerankerModelPath,
            "local_reranker_backend": localRerankerBackend,
            "vlm_provider": vlmProvider,
            "local_vlm_model": localVLMModel,
            "local_vlm_backend": localVLMBackend,
            "browser_endpoint": browserEndpoint,
            "always_confirm_code": alwaysConfirmCode,
            "mcp_servers": mcpServers.map {
                ["name": $0.name, "command": $0.command, "args": $0.args, "env": $0.env]
            },
        ]
    }

    public func save(_ path: String) throws {
        let data = try JSONSerialization.data(withJSONObject: export(),
                                              options: [.prettyPrinted, .sortedKeys])
        try AtomicFile.write(data, to: path)
    }
}
