import SwiftUI
import AppKit
import BaseAgentCore

struct SettingsView: View {
    @ObservedObject var model: AppViewModel
    @Environment(\.dismissWindow) private var dismissWindow
    @Environment(\.openWindow) private var openWindow

    // MARK: 主模型

    @State private var mainMode = "local"
    @State private var mainBackend = "mlx-lm"
    @State private var mainPath = ""
    @State private var mainContext = 32768
    @State private var mainKVCache = 32_768
    @State private var kvInt8Enabled = true
    @State private var mainTokenizer = ""
    @State private var auxTokenizer = ""
    @State private var mainTemperature = 0.7
    @State private var mainTopP = 0.95
    @State private var mainMinP = 0.05
    @State private var mainTopK = 20
    @State private var mainRepetitionPenalty = 1.0
    @State private var mainTimeout = 120.0
    @State private var mainPromptBatch = 512
    @State private var mainThinking = true
    @State private var mainKVReuse = true
    /// 应用启动时是否预加载主模型（auto_preload_main）。
    @State private var autoPreloadMain = true

    // MARK: API 模式

    @State private var baseURL = ""
    @State private var apiKey = ""
    @State private var apiModel = ""
    @State private var apiModels: [String] = []
    @State private var fetchingModels = false

    // MARK: 辅助模型

    @State private var auxEnabled = false
    @State private var auxBackend = "mlx-lm"
    @State private var auxPath = ""
    @State private var auxContext = 32768
    @State private var auxKVCache = 16384
    @State private var auxTemperature = 0.7
    @State private var auxTopP = 0.95
    @State private var auxMinP = 0.05
    @State private var auxTopK = 20
    @State private var auxRepetitionPenalty = 1.0
    @State private var auxTimeout = 120.0
    @State private var auxPromptBatch = 512
    @State private var auxThinking = false
    @State private var auxKVReuse = false
    @State private var auxKVInt8Enabled = true

    // MARK: 知识同步

    @State private var workspace = ""
    @State private var knowledgeBase = ""
    @State private var embeddingBackend = "onnx"
    @State private var embeddingPath = ""
    @State private var rerankerBackend = "onnx"
    @State private var rerankerPath = ""
    /// 当前 Embedding 后端下「切片实际生效值」的 token 上限（nil = 无精确换算）。
    @State private var embeddingContentLimit: Int?

    // MARK: 高级 RAG

    @State private var pdfEngine: PDFEngine = .vision
    @State private var epubOCREngine: PDFEngine = .vision
    @State private var ocrModelPath = ""
    @State private var ocrMMProjPath = ""
    @State private var vlmBackend = "mlx"
    @State private var ocrPreprocess = true
    @State private var ocrBinarize = true
    @State private var chunkSize = 800
    @State private var chunkOverlap = 80
    /// 向量落盘元素类型与 Matryoshka 截断策略。
    @State private var vectorElementType: VectorElementType = .int8
    @State private var matryoshkaMode: MatryoshkaMode = .auto
    @State private var matryoshkaDim = 512
    @State private var rerankEnabled = true
    @State private var rerankTopN = 5
    @State private var semanticChunking = true
    @State private var autoIngest = false
    @State private var embeddingMinBatch = 2
    @State private var embeddingMaxBatch = 32
    @State private var embeddingBatchStep = 2
    @State private var embeddingInitialBatch = 16
    @State private var embeddingMemoryLimitMB = 6_144
    @State private var embeddingThroughputTolerance = 0.03
    @State private var embeddingProbeBatches = 2
    @State private var embeddingProbeSeconds = 2.0
    @State private var showAdvanced = false

    @State private var statusMessage = ""

    private let llmTypes: [(id: String, label: String)] = [
        ("mlx-lm", "MLX-LM"),
        ("mlx-vlm", "MLX-VLM"),
        ("onnx", "ONNX"),
        ("gguf", "GGUF"),
    ]
    private let embeddingTypes: [(id: String, label: String)] = [
        ("fastembed", "FastEmbed"),
        ("mlx", "MLX"),
        ("onnx", "ONNX"),
        ("gguf", "GGUF"),
    ]

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    agentSection
                    knowledgeSection
                }
                .padding(16)
            }

            Divider()

            HStack(spacing: 8) {
                if model.isModelLoading {
                    ProgressView().controlSize(.small)
                    Text("模型加载中…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Text(statusMessage)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                Spacer()
                Button("关闭") { dismissWindow() }
                    .keyboardShortcut(.cancelAction)
            }
            .padding(12)
        }
        .frame(minWidth: 880, minHeight: 760)
        .onAppear(perform: load)
        .onChange(of: embeddingBackend) { refreshEmbeddingLimit() }
        .onChange(of: embeddingPath) { refreshEmbeddingLimit() }
    }

    // MARK: - Agent 模型配置区

    private var agentSection: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 10) {
                mainModelBlock
                Divider()
                auxModelBlock
            }
            .padding(8)
        } label: {
            Text("Agent 模型配置").font(.headline)
        }
    }

    private var mainModelBlock: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                Text("主模型").font(.subheadline).bold()
                if mainMode == "local" {
                    Toggle("启动时预加载", isOn: $autoPreloadMain)
                        .toggleStyle(.checkbox)
                        .help("默认开启：应用启动时在后台加载主模型；关闭后需手动点「加载模型」。修改后需点「应用配置」保存；同步结束后的模型恢复不受影响")
                }
                Spacer()
                Picker("", selection: $mainMode) {
                    Text("本地模式").tag("local")
                    Text("OpenAI 兼容接口").tag("api")
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .frame(width: 260)
            }

            if mainMode == "local" {
                localModelRows(
                    backend: $mainBackend,
                    path: $mainPath,
                    pathHint: "默认：models/qwen3.5-4b-agentic-coder-v4-4bit-mlx",
                    tokenizer: $mainTokenizer,
                    context: $mainContext,
                    kvCap: $mainKVCache,
                    temperature: $mainTemperature,
                    topP: $mainTopP,
                    minP: $mainMinP,
                    topK: $mainTopK,
                    repetitionPenalty: $mainRepetitionPenalty,
                    promptBatch: $mainPromptBatch,
                    thinking: $mainThinking,
                    kvReuse: $mainKVReuse,
                    kvInt8: $kvInt8Enabled
                ) {
                    HStack(spacing: 8) {
                        if model.mainModelLoaded {
                            Button("重新加载") { loadMainModel() }
                                .buttonStyle(.borderedProminent)
                                .tint(.green)
                                .disabled(model.isModelLoading || model.isRunning
                                    || model.isWarmingUp || model.isSyncing
                                    || model.isBuildingANN)
                        } else {
                            Button("加载模型") { loadMainModel() }
                                .disabled(model.isModelLoading || model.isRunning
                                    || model.isWarmingUp || model.isSyncing
                                    || model.isBuildingANN)
                        }
                        Button("卸载") { model.unloadMainModel() }
                            .disabled(model.isModelLoading || model.isRunning
                                || model.isWarmingUp || model.isSyncing)
                    }
                }
            } else {
                apiModelRows
            }
        }
    }

    private var apiModelRows: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Text("Base URL").font(.caption)
                TextField("https://api.openai.com/v1", text: $baseURL)
                    .textFieldStyle(.roundedBorder)
                    .frame(minWidth: 180)
                Text("API Key").font(.caption)
                SecureField("sk-…", text: $apiKey)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 140)
                Text("模型").font(.caption)
                if apiModels.isEmpty {
                    TextField("模型名", text: $apiModel)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 170)
                } else {
                    Picker("", selection: $apiModel) {
                        ForEach(apiModelChoices, id: \.self) { name in
                            Text(name).tag(name)
                        }
                    }
                    .labelsHidden()
                    .frame(width: 170)
                }
                Button(fetchingModels ? "获取中…" : "获取模型") { fetchModels() }
                    .disabled(fetchingModels)
            }

            HStack(alignment: .center, spacing: 6) {
                parameterFields(
                    context: $mainContext,
                    temperature: $mainTemperature,
                    topP: $mainTopP,
                    minP: $mainMinP,
                    topK: $mainTopK,
                    repetitionPenalty: $mainRepetitionPenalty
                )
                Spacer(minLength: 4)
                Button("应用配置") { applyAPIConfig() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(model.isSyncing || model.isDeduplicating || model.isBuildingANN)
            }
        }
    }

    private var auxModelBlock: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                Text("辅助模型").font(.subheadline).bold()
                Toggle("启用", isOn: $auxEnabled)
                    .toggleStyle(.checkbox)
                    .help("默认关闭：启动和知识同步前后都不加载辅助模型；关闭时 RAG 精简跳过，后台记忆/总结回退主模型。修改后需点击「应用配置」或「同步知识」保存")
                Spacer()
            }
            localModelRows(
                backend: $auxBackend,
                path: $auxPath,
                pathHint: "默认：models/Qwen3.5-2B-MLX-4bit",
                tokenizer: $auxTokenizer,
                context: $auxContext,
                kvCap: $auxKVCache,
                temperature: $auxTemperature,
                topP: $auxTopP,
                minP: $auxMinP,
                topK: $auxTopK,
                repetitionPenalty: $auxRepetitionPenalty,
                promptBatch: $auxPromptBatch,
                thinking: $auxThinking,
                kvReuse: $auxKVReuse,
                kvInt8: $auxKVInt8Enabled
            ) {
                HStack(spacing: 8) {
                    if model.auxModelLoaded {
                        Button("重新加载") { loadAuxModel() }
                            .buttonStyle(.borderedProminent)
                            .tint(.green)
                            .disabled(!auxEnabled || model.isModelLoading || model.isRunning
                                || model.isWarmingUp || model.isSyncing
                                || model.isBuildingANN)
                    } else {
                        Button("加载模型") { loadAuxModel() }
                            .disabled(!auxEnabled || model.isModelLoading || model.isRunning
                                || model.isWarmingUp || model.isSyncing
                                || model.isBuildingANN)
                    }
                    Button("卸载") { model.unloadAuxModel() }
                        .disabled(model.isModelLoading || model.isRunning
                            || model.isWarmingUp || model.isSyncing)
                }
            }
        }
    }

    /// 本地模型：类型 + 路径 + 参数 + 操作按钮（两行紧贴，参数标签与文本框同行）。
    private func localModelRows(
        backend: Binding<String>,
        path: Binding<String>,
        pathHint: String,
        tokenizer: Binding<String>? = nil,
        context: Binding<Int>,
        kvCap: Binding<Int>,
        temperature: Binding<Double>,
        topP: Binding<Double>,
        minP: Binding<Double>,
        topK: Binding<Int>,
        repetitionPenalty: Binding<Double>,
        promptBatch: Binding<Int>,
        thinking: Binding<Bool>,
        kvReuse: Binding<Bool>,
        kvInt8: Binding<Bool>,
        @ViewBuilder actions: () -> some View
    ) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text("模型类型").font(.caption)
                Picker("", selection: backend) {
                    ForEach(llmTypes, id: \.id) { type in
                        Text(type.label).tag(type.id)
                    }
                }
                .labelsHidden()
                .frame(width: 140)

                Text("本地路径").font(.caption)
                TextField(pathHint, text: path)
                    .textFieldStyle(.roundedBorder)
                Button("选择…") { chooseDirectory(path, title: "选择模型目录") }
            }
            if let tokenizer, backend.wrappedValue == "onnx" {
                HStack(spacing: 8) {
                    Text("tokenizer路径").font(.caption)
                    TextField("可选：含 tokenizer.json 的目录，留空从模型目录读取",
                              text: tokenizer)
                        .textFieldStyle(.roundedBorder)
                    Button("选择…") {
                        chooseDirectory(tokenizer, title: "选择 tokenizer 目录")
                    }
                }
            }

            // 参数行（含「思考 / KV复用 / KV量化」开关）尽量与操作按钮同行；
            // 窗口较窄放不下时按钮自动落到下一行右端，参数行不会被挤压变形。
            ViewThatFits(in: .horizontal) {
                HStack(alignment: .center, spacing: 6) {
                    parameterFields(
                        context: context,
                        kvCap: kvCap,
                        temperature: temperature,
                        topP: topP,
                        minP: minP,
                        topK: topK,
                        repetitionPenalty: repetitionPenalty,
                        promptBatch: backend.wrappedValue == "mlx-lm" ? promptBatch : nil,
                        thinking: thinking,
                        // ONNX 自动复用 past_key_values、MLX-VLM 每轮重建 prompt，
                        // 都不消费这两个开关，故仅在支持的后端上显示。
                        kvReuse: supportsKVToggles(backend.wrappedValue) ? kvReuse : nil,
                        kvInt8: supportsKVToggles(backend.wrappedValue) ? kvInt8 : nil
                    )
                    Spacer(minLength: 4)
                    actions()
                }
                VStack(alignment: .leading, spacing: 6) {
                    parameterFields(
                        context: context,
                        kvCap: kvCap,
                        temperature: temperature,
                        topP: topP,
                        minP: minP,
                        topK: topK,
                        repetitionPenalty: repetitionPenalty,
                        promptBatch: backend.wrappedValue == "mlx-lm" ? promptBatch : nil,
                        thinking: thinking,
                        kvReuse: supportsKVToggles(backend.wrappedValue) ? kvReuse : nil,
                        kvInt8: supportsKVToggles(backend.wrappedValue) ? kvInt8 : nil
                    )
                    HStack(spacing: 6) {
                        Spacer(minLength: 4)
                        actions()
                    }
                }
            }
        }
    }

    // MARK: - 知识同步配置区

    private var knowledgeSection: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 8) {
                    Text("工作目录").font(.caption)
                    TextField("~/base-agent-workspace", text: $workspace)
                        .textFieldStyle(.roundedBorder)
                    Button("选择…") { chooseDirectory($workspace, title: "选择工作目录") }
                    Text("知识库目录").font(.caption)
                    TextField("留空 = workspace/knowledge_base", text: $knowledgeBase)
                        .textFieldStyle(.roundedBorder)
                    Button("选择…") { chooseDirectory($knowledgeBase, title: "选择知识库目录") }
                }

                // Embedding 行：模型类型 / 本地路径之后紧接切片设置（语义切片 +
                // 切片大小 + overlap），再是「应用配置 / 同步知识」。优先单行；
                // 放不下时先把按钮换到下一行（切片设置仍在 Embedding 这一行），
                // 窗口极窄时再把切片设置换行，避免控件被裁剪。
                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 8) {
                        embeddingPathControls
                        chunkControls()
                        Spacer(minLength: 8)
                        knowledgeButtons()
                    }
                    VStack(alignment: .leading, spacing: 6) {
                        HStack(spacing: 8) {
                            embeddingPathControls
                            chunkControls()
                            Spacer(minLength: 8)
                        }
                        HStack(spacing: 8) {
                            Spacer(minLength: 8)
                            knowledgeButtons()
                        }
                    }
                    VStack(alignment: .leading, spacing: 6) {
                        HStack(spacing: 8) {
                            embeddingPathControls
                            Spacer(minLength: 8)
                        }
                        HStack(spacing: 8) {
                            chunkControls()
                            Spacer(minLength: 8)
                        }
                        HStack(spacing: 8) {
                            Spacer(minLength: 8)
                            knowledgeButtons()
                        }
                    }
                }

                // Reranker 行：模型类型 / 本地路径之后是「启用重排序 + TopN」，
                // 再是「向量去重 / 停止同步」，同样按可用宽度自动折行。
                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 8) {
                        rerankerPathControls
                        rerankControls()
                        Spacer(minLength: 8)
                        dedupeButtons()
                    }
                    VStack(alignment: .leading, spacing: 6) {
                        HStack(spacing: 8) {
                            rerankerPathControls
                            rerankControls()
                            Spacer(minLength: 8)
                        }
                        HStack(spacing: 8) {
                            Spacer(minLength: 8)
                            dedupeButtons()
                        }
                    }
                }

                // 向量索引行：ANN 索引摘要 + 构建/停止/删除。
                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 8) {
                        annStatusLabel
                        Spacer(minLength: 8)
                        annButtons()
                    }
                    VStack(alignment: .leading, spacing: 6) {
                        annStatusLabel
                        HStack(spacing: 8) {
                            Spacer(minLength: 8)
                            annButtons()
                        }
                    }
                }

                // 向量存储行：元素类型 / Matryoshka / 显式重建。
                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 8) {
                        vectorStorageControls
                        Spacer(minLength: 8)
                        vectorRebuildButtons()
                    }
                    VStack(alignment: .leading, spacing: 6) {
                        vectorStorageControls
                        HStack(spacing: 8) {
                            Spacer(minLength: 8)
                            vectorRebuildButtons()
                        }
                    }
                }
                if model.vectorRebuildRequired {
                    HStack(alignment: .firstTextBaseline, spacing: 6) {
                        Image(systemName: "exclamationmark.triangle.fill")
                        Text("向量库需重建：\(model.vectorRebuildReason)。"
                            + "旧切片仍保留，全文检索可用；重建会按当前配置清空并全量重新同步。")
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .font(.caption2)
                    .foregroundStyle(.red)
                }

                DisclosureGroup("高级 RAG / OCR 设置", isExpanded: $showAdvanced) {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack(spacing: 8) {
                            Text("自定义OCR模型").font(.caption)
                            Picker("", selection: $vlmBackend) {
                                Text("MLX").tag("mlx")
                                Text("GGUF").tag("gguf")
                            }
                            .labelsHidden()
                            .frame(width: 80)
                            TextField("模型路径（MLX 填目录，GGUF 填文件）", text: $ocrModelPath)
                                .textFieldStyle(.roundedBorder)
                                .frame(minWidth: 160)
                            Button("选择…") {
                                if vlmBackend == "mlx" {
                                    chooseDirectory($ocrModelPath, title: "选择 OCR 模型目录")
                                } else {
                                    chooseFile($ocrModelPath, title: "选择 OCR 模型文件")
                                }
                            }
                            Button("保存设置") { saveAdvancedSettings() }
                                .disabled(model.isSyncing || model.isDeduplicating
                                    || model.isBuildingANN)
                                .help("保存并应用高级 RAG / OCR 设置")
                            Button("识别测试") { openOCRLab() }
                                .disabled(model.isSyncing || model.isDeduplicating
                                    || model.isBuildingANN)
                                .help("打开 OCR 识别测试窗口，用指定模型识别图片或 PDF 并对比结果")
                        }
                        if vlmBackend == "gguf" {
                            HStack(spacing: 8) {
                                Text("mmproj路径").font(.caption)
                                TextField("可选：多模态投影文件，留空自动查找模型同目录的 mmproj*.gguf",
                                          text: $ocrMMProjPath)
                                    .textFieldStyle(.roundedBorder)
                                Button("选择…") { chooseFile($ocrMMProjPath, title: "选择 mmproj 文件") }
                                Spacer(minLength: 8)
                            }
                        }
                        HStack(spacing: 10) {
                            Text("PDF OCR").font(.caption)
                            Picker("", selection: $pdfEngine) {
                                Text("OvisOCR3").tag(PDFEngine.ovis)
                                Text("Vision OCR").tag(PDFEngine.vision)
                                Text("混合（Vision + Ovis 表格）").tag(PDFEngine.hybrid)
                                if customOCRAvailable {
                                    Text("自定义：\(customOCRName)").tag(PDFEngine.custom)
                                }
                            }
                            .labelsHidden()
                            .frame(width: 240)
                            Text("EPUB OCR").font(.caption)
                            Picker("", selection: $epubOCREngine) {
                                Text("Apple Vision").tag(PDFEngine.vision)
                                Text("OvisOCR3").tag(PDFEngine.ovis)
                                if customOCRAvailable {
                                    Text("自定义：\(customOCRName)").tag(PDFEngine.custom)
                                }
                            }
                            .labelsHidden()
                            .frame(width: 170)
                            Toggle("图像归一化", isOn: $ocrPreprocess)
                            Toggle("Otsu 二值化", isOn: $ocrBinarize)
                            Button("转换工具") { openConverter() }
                                .help("打开转换工具窗口：把 PDF/EPUB/Office/HTML/图片/文本等批量转换为 Markdown，"
                                    + "沿用这里的 OCR 与切片前的解析设置")
                        }
                        HStack(spacing: 14) {
                            Text("同步参数设置")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.secondary)
                            Toggle("启动时自动同步", isOn: $autoIngest)
                                .help("启动后自动执行一次知识增量同步")
                        }
                        HStack(spacing: 8) {
                            Text("自适应子批参数")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.secondary)
                            Stepper("下限：\(embeddingMinBatch)",
                                    value: $embeddingMinBatch, in: 2...64)
                                .help("自适应下调的最小值，默认 2；超过上限时按上限收口")
                            Stepper("上限：\(embeddingMaxBatch)",
                                    value: $embeddingMaxBatch, in: 2...256)
                                .help("embedding 自适应子批的最大值。实测 477 token 长块在 batch 4-16 已到吞吐平台，默认 32 兼顾短块；64 以上只增加内存与单批延迟")
                            Stepper("步进：\(embeddingBatchStep)",
                                    value: $embeddingBatchStep, in: 1...16)
                                .help("每次上探/下探的调整幅度，默认 2")
                            Stepper("启动预设：\(embeddingInitialBatch)",
                                    value: $embeddingInitialBatch, in: 2...256)
                                .help("同步开始时的初始子批，默认 16；会被夹在下限与上限之间")
                            Stepper("探测批次：\(embeddingProbeBatches)",
                                    value: $embeddingProbeBatches, in: 1...16)
                                .help("每档至少观察多少个 embedding 批后评估一次，默认 2")
                            Stepper("探测窗口：\(String(format: "%.1f", embeddingProbeSeconds))s",
                                    value: $embeddingProbeSeconds, in: 0.5...30, step: 0.5)
                                .help("每档至少观察多少秒后评估一次，默认 2s；与探测批次数满足其一即评估")
                            Stepper("内存阈值：\(embeddingMemoryLimitMB)MB",
                                    value: $embeddingMemoryLimitMB,
                                    in: 1_024...32_768, step: 512)
                                .help("进程 footprint 超过该值立即下调子批；bge 单独运行峰值约 1.6GB，默认 6144MB")
                            Stepper("吞吐容差：\(Int((embeddingThroughputTolerance * 100).rounded()))%",
                                    value: $embeddingThroughputTolerance,
                                    in: 0.01...0.50, step: 0.01)
                                .help("吞吐变化小于该比例视为测量噪声，不调整子批；默认 3%，实测噪声较大时可调到 8-10%")
                        }
                        .font(.caption)
                    }
                    .padding(.top, 8)
                }
            }
            .padding(8)
        } label: {
            Text("知识同步配置").font(.headline)
        }
    }

    // MARK: - 知识同步配置的分段控件

    /// Embedding 的「模型类型 + 本地路径 + 选择…」。
    @ViewBuilder
    private var embeddingPathControls: some View {
        modelPathControls(title: "Embedding",
                          backend: $embeddingBackend,
                          path: $embeddingPath,
                          chooseTitle: "选择 Embedding 目录")
    }

    /// Reranker 的「模型类型 + 本地路径 + 选择…」。
    @ViewBuilder
    private var rerankerPathControls: some View {
        modelPathControls(title: "Reranker",
                          backend: $rerankerBackend,
                          path: $rerankerPath,
                          chooseTitle: "选择 Reranker 目录")
    }

    @ViewBuilder
    private func modelPathControls(title: String,
                                   backend: Binding<String>,
                                   path: Binding<String>,
                                   chooseTitle: String) -> some View {
        Text(title).font(.caption)
        Picker("", selection: backend) {
            ForEach(embeddingTypes, id: \.id) { type in
                Text(type.label).tag(type.id)
            }
        }
        .labelsHidden()
        .frame(width: 130)
        Text("本地路径").font(.caption)
        TextField("可选", text: path)
            .textFieldStyle(.roundedBorder)
            .frame(minWidth: 140)
        Button("选择…") { chooseDirectory(path, title: chooseTitle) }
    }

    /// 切片设置：语义切片开关 + 切片大小（含模型窗口收口提示）+ overlap。
    @ViewBuilder
    private func chunkControls() -> some View {
        Toggle("语义切片", isOn: $semanticChunking)
            .help("开启后大段落在句间相似度低谷处断开，切片阶段需嵌入全部候选句，明显更慢；"
                + "关闭则按结构+token 边界递归切片")
        VStack(alignment: .leading, spacing: 1) {
            Stepper("切片大小：\(chunkSize) token", value: $chunkSize,
                    in: 100...4000, step: 50)
                .help("含 overlap 的总预算，全链路（切分、入库、检索呈现）统一按它计算；"
                    + "超过 embedding 模型输入窗口时会自动压到窗口上限。"
                    + "纯中文约 1 字 ≈ 1 token，中英混排约 1.8 字符/token")
            if let limit = embeddingContentLimit, chunkSize > limit {
                Text("实际生效：\(limit) token（受 embedding 模型输入窗口限制）")
                    .font(.caption2)
                    .foregroundStyle(.orange)
            }
        }
        Stepper("overlap：\(chunkOverlap) token", value: $chunkOverlap,
                in: 0...1000, step: 10)
            .help("占切片大小的 10%，计入总预算：正文按 chunk_size − overlap 切片")
    }

    /// 重排序开关与 TopN：开启后最终返回分数最高的 TopN 条。
    @ViewBuilder
    private func rerankControls() -> some View {
        Toggle("启用重排序", isOn: $rerankEnabled)
            .help("开启后先用混合检索取候选，再由 reranker 重新打分排序，"
                + "按下面的 TopN 返回最终结果；关闭则直接用 RRF 排序结果")
        Stepper("TopN：\(rerankTopN)", value: $rerankTopN, in: 1...20)
            .help("重排序之后保留的结果数量，默认 5；候选池会按它放大后再打乱重排。"
                + "关闭重排序时按检索调用方的 top_k 返回")
    }

    @ViewBuilder
    private func knowledgeButtons() -> some View {
        Button("应用配置") { applyKnowledgeConfig() }
            .disabled(model.isSyncing || model.isDeduplicating || model.isBuildingANN
                || model.isModelLoading || model.isRunning || model.isWarmingUp)
        Button("同步知识") { applyAndSync() }
            .help("同步期间会自动等待模型加载/预热结束")
            .disabled(model.isSyncing || model.isDeduplicating || model.isBuildingANN
                || model.isRunning)
    }

    @ViewBuilder
    private func dedupeButtons() -> some View {
        Button(model.isDeduplicating ? "去重中…" : "向量去重") {
            model.requestVectorDedupe()
        }
        .help("排查并清理向量库内重复的向量，保留入库时间最早的一份")
        .disabled(model.isDeduplicating || model.isBuildingANN)
        Button("停止同步") { model.stopSync() }
            .disabled(!model.isSyncing)
    }

    /// ANN 索引状态：分块 IVF 摘要（未构建 / 已索引切片数 / 尾块 / 需重建）。
    @ViewBuilder
    private var annStatusLabel: some View {
        HStack(spacing: 6) {
            Text("ANN 索引").font(.caption)
            Text(model.annStatusText)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(2)
                .help("分块 IVF 索引：块内 k-means 聚类 + int8 量化按簇落盘，检索只读命中的簇，"
                    + "避免在大库上全表扫描；索引水位之后的新切片按 rowid 顺序精确补齐。"
                    + "库内切片较少时精确扫描更快，可先不构建")
        }
    }

    @ViewBuilder
    private func annButtons() -> some View {
        Button(model.isBuildingANN ? "索引维护中…" : "构建 ANN 索引") {
            model.requestANNIndexBuild()
        }
        .help("构建/续建向量库的分块 IVF 索引（可中断，已写完的块保留）；"
            + "构建期间知识检索会按只读超时跳过，建议在空闲时执行")
        .disabled(model.isBuildingANN || model.isSyncing || model.isDeduplicating
            || model.isModelLoading || model.isWarmingUp || model.isRunning)
        Button("停止构建") { model.stopANNIndexBuild() }
            .disabled(!model.isBuildingANN)
        Button("删除索引") { model.requestDropANNIndex() }
            .help("删除 ANN 索引，检索回退到精确扫描（2100 万切片级别的大库上会明显变慢）")
            .disabled(model.isBuildingANN || model.isSyncing || model.isDeduplicating
                || model.isRunning)
    }

    // MARK: - 向量存储 / Matryoshka

    /// 向量落盘元素类型 + Matryoshka 模式/截断维 + 当前生效摘要。
    @ViewBuilder
    private var vectorStorageControls: some View {
        Text("向量库").font(.caption)
        Picker("", selection: $vectorElementType) {
            ForEach(VectorElementType.allCases, id: \.rawValue) { type in
                Text(vectorElementLabel(type)).tag(type)
            }
        }
        .labelsHidden()
        .frame(width: 120)
        .help("向量落盘元素类型：int8 体积是 fp32 的 1/4（默认）；bin 体积最小但只有汉明排序。"
            + "fp16 在 sqlite-vec 0.1.9 中不可用，选择后应用配置会直接报错。")
        Text("Matryoshka").font(.caption)
        Picker("", selection: $matryoshkaMode) {
            ForEach(MatryoshkaMode.allCases, id: \.rawValue) { mode in
                Text(mode.displayName).tag(mode)
            }
        }
        .labelsHidden()
        .frame(width: 135)
        .help("截断向量后强制重新归一化，入库与查询使用同一套变换。"
            + "自动模式只对 nomic 等已知支持该特性的模型启用。")
        Picker("", selection: $matryoshkaDim) {
            Text("128 维").tag(128)
            Text("256 维").tag(256)
            Text("512 维").tag(512)
        }
        .labelsHidden()
        .frame(width: 90)
        .disabled(matryoshkaMode == .off)
        .help("Matryoshka 截断维：越小向量库越小、召回精度越低；默认 512。")
        Text(model.vectorStatusText)
            .font(.caption2)
            .foregroundStyle(model.vectorRebuildRequired ? .red : .secondary)
            .lineLimit(1)
    }

    @ViewBuilder
    private func vectorRebuildButtons() -> some View {
        Button("重建向量库") { model.requestVectorRebuild() }
            .help("按当前元素类型/Matryoshka 配置重建向量表：清空旧切片、全文索引与 ANN 索引后"
                + "全量重新同步；可中断，知识库原始文件不受影响")
            .disabled(model.isSyncing || model.isDeduplicating || model.isBuildingANN
                || model.isModelLoading || model.isWarmingUp || model.isRunning)
    }

    private func vectorElementLabel(_ type: VectorElementType) -> String {
        switch type {
        case .fp32: return "FP32"
        case .int8: return "INT8（推荐）"
        case .fp16: return "FP16（不支持）"
        case .bin: return "BIN（最小）"
        }
    }

    // MARK: - 参数控件

    /// 参数区：优先把数字字段与开关放在同一行；窗口较窄时拆成「数字字段」
    /// +「思考 / KV复用 / KV量化」两行，开关始终紧挨在一起且不会被裁剪。
    @ViewBuilder
    private func parameterFields(
        context: Binding<Int>,
        kvCap: Binding<Int>? = nil,
        temperature: Binding<Double>,
        topP: Binding<Double>,
        minP: Binding<Double>,
        topK: Binding<Int>,
        repetitionPenalty: Binding<Double>,
        promptBatch: Binding<Int>? = nil,
        thinking: Binding<Bool>? = nil,
        kvReuse: Binding<Bool>? = nil,
        kvInt8: Binding<Bool>? = nil
    ) -> some View {
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .center, spacing: 6) {
                numberFields(
                    context: context, kvCap: kvCap, temperature: temperature,
                    topP: topP, minP: minP, topK: topK,
                    repetitionPenalty: repetitionPenalty, promptBatch: promptBatch
                )
                parameterToggles(thinking: thinking, kvReuse: kvReuse, kvInt8: kvInt8)
            }
            VStack(alignment: .leading, spacing: 3) {
                numberFields(
                    context: context, kvCap: kvCap, temperature: temperature,
                    topP: topP, minP: minP, topK: topK,
                    repetitionPenalty: repetitionPenalty, promptBatch: promptBatch
                )
                parameterToggles(thinking: thinking, kvReuse: kvReuse, kvInt8: kvInt8)
            }
        }
    }

    @ViewBuilder
    private func numberFields(
        context: Binding<Int>,
        kvCap: Binding<Int>?,
        temperature: Binding<Double>,
        topP: Binding<Double>,
        minP: Binding<Double>,
        topK: Binding<Int>,
        repetitionPenalty: Binding<Double>,
        promptBatch: Binding<Int>?
    ) -> some View {
        HStack(alignment: .center, spacing: 3) {
            numberField("Context", value: context, width: 60)
            if let kvCap {
                numberField("KV 上限", value: kvCap, width: 66)
            }
            numberField("Temp", value: temperature, width: 48)
            numberField("top_p", value: topP, width: 48)
            numberField("min_p", value: minP, width: 48)
            numberField("top_k", value: topK, width: 48)
            numberField("repeat", value: repetitionPenalty, width: 52)
            if let promptBatch {
                numberField("batch", value: promptBatch, width: 52)
            }
        }
    }

    /// 「思考 / KV复用 / KV量化」开关组；不支持的组合（nil）不渲染。
    @ViewBuilder
    private func parameterToggles(
        thinking: Binding<Bool>?,
        kvReuse: Binding<Bool>?,
        kvInt8: Binding<Bool>?
    ) -> some View {
        if thinking != nil || kvReuse != nil || kvInt8 != nil {
            HStack(alignment: .center, spacing: 10) {
                if let thinking {
                    Toggle("思考", isOn: thinking)
                        .toggleStyle(.checkbox)
                        .font(.caption2)
                        .fixedSize()
                }
                if let kvReuse {
                    Toggle("KV复用", isOn: kvReuse)
                        .toggleStyle(.checkbox)
                        .font(.caption2)
                        .fixedSize()
                        .help("关闭后每轮全量重建 KV（等效禁用前缀缓存）：主模型与辅助模型各自独立")
                }
                if let kvInt8 {
                    Toggle("KV量化", isOn: kvInt8)
                        .toggleStyle(.checkbox)
                        .font(.caption2)
                        .fixedSize()
                        .help("KV cache int8 量化：KV 内存约减半，质量几乎无损；主模型与辅助模型各自独立")
                }
            }
        }
    }

    private func numberField(_ title: String, value: Binding<Int>,
                             width: CGFloat) -> some View {
        HStack(spacing: 3) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .fixedSize(horizontal: true, vertical: false)
            TextField("", value: value, format: .number)
                .textFieldStyle(.roundedBorder)
                .frame(width: width)
        }
    }

    private func numberField(_ title: String, value: Binding<Double>,
                             width: CGFloat) -> some View {
        HStack(spacing: 3) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .fixedSize(horizontal: true, vertical: false)
            TextField("", value: value, format: .number)
                .textFieldStyle(.roundedBorder)
                .frame(width: width)
        }
    }

    // MARK: - 加载 / 应用

    private func load() {
        let config = model.config
        mainMode = config.llmProvider.lowercased() == "api" ? "api" : "local"
        mainBackend = normalizedLLMType(config.localLLMBackend)
        mainPath = config.localLLMModel
        mainTokenizer = config.llmTokenizerPath
        mainContext = config.maxContextTokens
        mainKVCache = config.kvCacheMaxTokens > 0
            ? config.kvCacheMaxTokens : config.maxContextTokens
        kvInt8Enabled = config.kvCacheInt8
        mainTemperature = config.temperature
        mainTopP = config.topP
        mainMinP = config.minP
        mainTopK = config.topK
        mainRepetitionPenalty = config.repetitionPenalty
        mainTimeout = config.requestTimeout
        mainPromptBatch = config.promptBatchSize
        mainThinking = config.localLLMEnableThinking
        mainKVReuse = config.llmKVReuse
        autoPreloadMain = config.autoPreloadMain

        baseURL = config.baseURL
        apiKey = config.apiKey
        apiModel = config.model

        auxEnabled = config.auxEnabled
        auxBackend = normalizedLLMType(config.auxLocalLLMBackend)
        auxPath = config.auxLocalLLMModel
        auxTokenizer = config.auxTokenizerPath
        auxContext = config.auxMaxContextTokens
        auxKVCache = config.auxKvCacheMaxTokens > 0
            ? config.auxKvCacheMaxTokens : config.auxMaxContextTokens
        auxTemperature = config.auxTemperature
        auxTopP = config.auxTopP
        auxMinP = config.auxMinP
        auxTopK = config.auxTopK
        auxRepetitionPenalty = config.auxRepetitionPenalty
        auxTimeout = config.auxRequestTimeout
        auxPromptBatch = config.auxPromptBatchSize
        auxThinking = config.auxLocalLLMEnableThinking
        auxKVReuse = config.auxKvReuse
        auxKVInt8Enabled = config.auxKvCacheInt8

        workspace = config.workspace
        knowledgeBase = config.knowledgeBase
        embeddingBackend = normalizedEmbeddingType(config.localEmbeddingBackend)
        embeddingPath = config.localEmbeddingModelPath
        rerankerBackend = normalizedEmbeddingType(config.localRerankerBackend)
        rerankerPath = config.localRerankerModelPath

        ocrModelPath = config.ragOCRModelPath
        let hasCustomOCR = !config.ragOCRModelPath
            .trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        pdfEngine = (!hasCustomOCR && config.ragPDFEngine == .custom)
            ? .vision : config.ragPDFEngine
        epubOCREngine = (!hasCustomOCR && config.ragEPUBOCREngine == .custom)
            ? .vision : config.ragEPUBOCREngine
        ocrMMProjPath = config.ragOCRMMProjPath
        vlmBackend = config.localVLMBackend.lowercased() == "gguf" ? "gguf" : "mlx"
        ocrPreprocess = config.ragOCRPreprocess
        ocrBinarize = config.ragOCRBinarize
        chunkSize = config.ragChunkSize
        chunkOverlap = config.ragChunkOverlap
        vectorElementType = config.ragVectorElementType
        matryoshkaMode = config.ragMatryoshka
        matryoshkaDim = config.ragMatryoshkaDim
        rerankEnabled = config.ragRerankEnabled
        rerankTopN = config.ragRerankTopN
        semanticChunking = config.ragSemanticChunking
        autoIngest = config.ragAutoIngest
        embeddingMinBatch = config.ragEmbeddingMinBatch
        embeddingMaxBatch = config.ragEmbeddingMaxBatch
        embeddingBatchStep = config.ragEmbeddingBatchStep
        embeddingInitialBatch = config.ragEmbeddingInitialBatch
        embeddingMemoryLimitMB = config.ragEmbeddingMemoryLimitMB
        embeddingThroughputTolerance = config.ragEmbeddingThroughputTolerance
        embeddingProbeBatches = config.ragEmbeddingProbeBatches
        embeddingProbeSeconds = config.ragEmbeddingProbeSeconds
        refreshEmbeddingLimit()
    }

    /// 刷新「切片实际生效值」提示：embedding 模型输入窗口小于配置的切片大小时，
    /// ingest 会把整块收口到窗口内（同一份收口逻辑在 RAGEngine 与 provider 里共用）。
    /// 这里只做一次「面板选择 → 配置」的换算，不改变当前状态提示。
    private func refreshEmbeddingLimit() {
        let saved = statusMessage
        defer { statusMessage = saved }
        // 路径还没填完整时不要去解析模型目录：既算不出上限，又会刷无效路径告警。
        let path = embeddingPath.trimmingCharacters(in: .whitespacesAndNewlines)
        if !path.isEmpty {
            let isGGUF = AgentConfig.normalizeEmbeddingBackend(embeddingBackend) == "gguf"
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: (path as NSString).expandingTildeInPath,
                                                isDirectory: &isDirectory),
                  isDirectory.boolValue != isGGUF else {
                embeddingContentLimit = nil
                return
            }
        }
        guard let config = buildConfig() else {
            embeddingContentLimit = nil
            return
        }
        embeddingContentLimit = LocalModelFactory.shared
            .embeddingContentTokenLimit(config: config)
    }

    private func loadMainModel() {
        guard !model.isSyncing, !model.isDeduplicating, !model.isBuildingANN else {
            statusMessage = "知识同步/索引构建/向量去重进行中，暂不能加载模型"
            return
        }
        guard let config = buildConfig() else { return }
        statusMessage = "正在加载主模型…"
        model.reloadMainModel(config: config)
    }

    private func loadAuxModel() {
        guard !model.isSyncing, !model.isDeduplicating, !model.isBuildingANN else {
            statusMessage = "知识同步/索引构建/向量去重进行中，暂不能加载模型"
            return
        }
        guard let config = buildConfig() else { return }
        statusMessage = "正在加载辅助模型…"
        model.reloadAuxModel(config: config)
    }

    private func applyAPIConfig() {
        guard let config = buildConfig() else { return }
        statusMessage = "配置已应用"
        model.applyConfig(config)
    }

    private func applyKnowledgeConfig() {
        guard !model.isSyncing, !model.isDeduplicating, !model.isBuildingANN else {
            statusMessage = "知识同步/索引构建/向量去重进行中，暂不能应用配置"
            return
        }
        guard let config = buildConfig() else { return }
        statusMessage = "知识同步配置已应用并保存"
        model.applyConfig(config)
    }

    private func applyAndSync() {
        guard let config = buildConfig() else { return }
        // 知识库目录切换后旧索引不迁移：直接对新目录强制全量重建，
        // 避免增量同步对着已有 manifest 判定「无变化」而秒完成。
        let kbChanged = model.config.resolvedKnowledgeBase != config.resolvedKnowledgeBase
        statusMessage = kbChanged
            ? "知识库目录已切换为 \(config.resolvedKnowledgeBase)，将强制全量同步…"
            : "配置已应用，正在同步知识库…"
        model.applyConfig(config) {
            model.syncKnowledgeBase(force: kbChanged)
        }
    }

    /// 自定义 OCR 模型的显示名：MLX 取模型目录名，GGUF 取文件名去掉扩展名。
    private var customOCRName: String {
        let path = ocrModelPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !path.isEmpty else { return "" }
        let url = URL(fileURLWithPath: path)
        if vlmBackend == "mlx" {
            return url.lastPathComponent
        }
        let name = url.deletingPathExtension().lastPathComponent
        return name.isEmpty ? url.lastPathComponent : name
    }

    private var customOCRAvailable: Bool {
        !customOCRName.isEmpty
    }

    /// KV 复用 / 量化只对 MLX-LM 与 GGUF 生效：
    /// ONNX 自动复用 past_key_values，MLX-VLM 每轮重建 prompt，均不消费这两个开关。
    private func supportsKVToggles(_ backend: String) -> Bool {
        backend == "mlx-lm" || backend == "gguf"
    }

    /// 保存并应用高级 RAG / OCR 设置（包含自定义 OCR 模型、PDF/EPUB OCR 引擎等）。
    private func saveAdvancedSettings() {
        guard !model.isSyncing, !model.isDeduplicating, !model.isBuildingANN else {
            statusMessage = "知识同步/索引构建/向量去重进行中，暂不能保存设置"
            return
        }
        if !customOCRAvailable {
            if pdfEngine == .custom { pdfEngine = .vision }
            if epubOCREngine == .custom { epubOCREngine = .vision }
        }
        guard let config = buildConfig() else { return }
        statusMessage = "高级 RAG / OCR 设置已保存并应用"
        model.applyConfig(config)
    }

    /// 打开识别测试窗口：把当前面板上的 OCR 设置（含尚未保存的修改）带过去。
    private func openOCRLab() {
        var config = model.config
        let path = ocrModelPath.trimmingCharacters(in: .whitespacesAndNewlines)
        if !path.isEmpty {
            config.ragOCRModelName = customOCRName
        }
        config.ragOCRModelPath = path
        config.ragOCRMMProjPath = ocrMMProjPath.trimmingCharacters(in: .whitespacesAndNewlines)
        config.localVLMBackend = vlmBackend
        config.ragOCRPreprocess = ocrPreprocess
        config.ragOCRBinarize = ocrBinarize
        config.ragPDFEngine = pdfEngine
        config.ragEPUBOCREngine = epubOCREngine
        model.ocrLabConfig = config
        openWindow(id: "ocr-lab")
    }

    /// 打开转换工具窗口：使用当前面板上的解析/OCR 设置快照（含尚未保存的修改），
    /// 只影响这一窗口内的转换参数，不写回 config.json、不重建 Agent。
    private func openConverter() {
        var config = model.config
        let path = ocrModelPath.trimmingCharacters(in: .whitespacesAndNewlines)
        if !path.isEmpty {
            config.ragOCRModelName = customOCRName
        }
        config.ragOCRModelPath = path
        config.ragOCRMMProjPath = ocrMMProjPath.trimmingCharacters(in: .whitespacesAndNewlines)
        config.localVLMBackend = vlmBackend
        config.ragOCRPreprocess = ocrPreprocess
        config.ragOCRBinarize = ocrBinarize
        config.ragPDFEngine = pdfEngine
        config.ragEPUBOCREngine = epubOCREngine
        // 快照里若指定了自定义 OCR 却没有模型路径，回退 Apple Vision，
        // 避免转换窗口因空模型路径直接失败。
        if !customOCRAvailable {
            if config.ragPDFEngine == .custom { config.ragPDFEngine = .vision }
            if config.ragEPUBOCREngine == .custom { config.ragEPUBOCREngine = .vision }
        } else {
            config.ragOCRModelName = customOCRName
        }
        model.converterConfig = config
        openWindow(id: "markdown-converter")
    }

    private func buildConfig() -> AgentConfig? {
        let mb = AgentConfig.normalizeLLMBackend(mainBackend)
        let ab = AgentConfig.normalizeLLMBackend(auxBackend)
        let supportedLLM = ["mlx-lm", "mlx-vlm", "gguf", "onnx"]
        if mainMode == "local", !supportedLLM.contains(mb) {
            statusMessage = "主模型类型 \(mainBackend) 尚未接入，请选择 MLX-LM / MLX-VLM / GGUF / ONNX"
            return nil
        }
        if !supportedLLM.contains(ab) {
            statusMessage = "辅助模型类型 \(auxBackend) 尚未接入，请选择 MLX-LM / MLX-VLM / GGUF / ONNX"
            return nil
        }
        if mainMode == "api" {
            guard !baseURL.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                statusMessage = "API 模式需要填写 Base URL"
                return nil
            }
            guard !apiModel.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                statusMessage = "API 模式需要选择或填写模型"
                return nil
            }
        }
        let eb = AgentConfig.normalizeEmbeddingBackend(embeddingBackend)
        guard eb == "onnx" || eb == "fastembed" || eb == "mlx" || eb == "gguf" else {
            statusMessage = "Embedding 类型 \(embeddingBackend) 当前版本尚未接入，请选择 ONNX / FastEmbed / MLX / GGUF"
            return nil
        }
        // fp16：sqlite-vec 0.1.9 没有该元素类型，直接拒绝应用并给出明确提示。
        if let reason = vectorElementType.unsupportedReason {
            statusMessage = reason
            return nil
        }
        let rb = AgentConfig.normalizeEmbeddingBackend(rerankerBackend)
        if rerankEnabled, rb != "onnx", rb != "fastembed", rb != "mlx", rb != "gguf" {
            statusMessage = "Reranker 类型 \(rerankerBackend) 当前版本尚未接入，请选择 ONNX / FastEmbed / MLX / GGUF"
            return nil
        }

        var config = model.config
        config.llmProvider = mainMode
        config.localLLMBackend = AgentConfig.normalizeLLMBackend(mainBackend)
        config.localLLMModel = mainPath.trimmingCharacters(in: .whitespacesAndNewlines)
        config.llmTokenizerPath = mainTokenizer.trimmingCharacters(in: .whitespacesAndNewlines)
        config.maxContextTokens = max(8192, mainContext)
        config.kvCacheMaxTokens = max(0, mainKVCache)
        config.kvCacheInt8 = kvInt8Enabled
        config.temperature = clamp(mainTemperature, 0, 2)
        config.topP = clamp(mainTopP, 0, 1)
        config.minP = clamp(mainMinP, 0, 1)
        config.topK = max(0, mainTopK)
        config.repetitionPenalty = max(0, mainRepetitionPenalty)
        config.requestTimeout = max(10, mainTimeout)
        config.promptBatchSize = max(0, mainPromptBatch)
        config.localLLMEnableThinking = mainThinking
        config.llmKVReuse = mainKVReuse
        config.autoPreloadMain = autoPreloadMain

        if mainMode == "api" {
            config.baseURL = baseURL.trimmingCharacters(in: .whitespacesAndNewlines)
            config.apiKey = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
            config.model = apiModel.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        config.auxEnabled = auxEnabled
        config.auxLLMProvider = "local"
        config.auxLocalLLMBackend = AgentConfig.normalizeLLMBackend(auxBackend)
        config.auxLocalLLMModel = auxPath.trimmingCharacters(in: .whitespacesAndNewlines)
        config.auxTokenizerPath = auxTokenizer.trimmingCharacters(in: .whitespacesAndNewlines)
        config.auxMaxContextTokens = max(8192, auxContext)
        config.auxKvCacheMaxTokens = max(0, auxKVCache)
        config.auxTemperature = clamp(auxTemperature, 0, 2)
        config.auxTopP = clamp(auxTopP, 0, 1)
        config.auxMinP = clamp(auxMinP, 0, 1)
        config.auxTopK = max(0, auxTopK)
        config.auxRepetitionPenalty = max(0, auxRepetitionPenalty)
        config.auxRequestTimeout = max(10, auxTimeout)
        config.auxPromptBatchSize = max(0, auxPromptBatch)
        config.auxLocalLLMEnableThinking = auxThinking
        config.auxKvReuse = auxKVReuse
        config.auxKvCacheInt8 = auxKVInt8Enabled

        config.workspace = (workspace.trimmingCharacters(in: .whitespacesAndNewlines) as NSString)
            .expandingTildeInPath
        config.knowledgeBase = knowledgeBase.trimmingCharacters(in: .whitespacesAndNewlines)

        config.embeddingProvider = "local"
        config.localEmbeddingBackend = eb
        config.localEmbeddingModelPath = embeddingPath.trimmingCharacters(in: .whitespacesAndNewlines)
        // 模型名不再由用户编辑：填了本地路径时从目录名派生；
        // 未填路径则保留配置文件里的既有名字（默认 8bit nomic）。
        if !config.localEmbeddingModelPath.isEmpty {
            let derived = URL(fileURLWithPath: config.localEmbeddingModelPath).lastPathComponent
            config.localEmbeddingModel = derived
            config.ragEmbeddingModel = derived
        }

        config.rerankerProvider = "local"
        config.localRerankerBackend = rb
        config.localRerankerModelPath = rerankerPath.trimmingCharacters(in: .whitespacesAndNewlines)
        if !config.localRerankerModelPath.isEmpty {
            let derived = URL(fileURLWithPath: config.localRerankerModelPath).lastPathComponent
            config.localRerankerModel = derived
            config.ragRerankModel = derived
        }

        let ocrPath = ocrModelPath.trimmingCharacters(in: .whitespacesAndNewlines)
        config.ragPDFEngine = (pdfEngine == .custom && !customOCRAvailable)
            ? .vision : pdfEngine
        config.ragEPUBOCREngine = (epubOCREngine == .custom && !customOCRAvailable)
            ? .vision : epubOCREngine
        if !ocrPath.isEmpty {
            // 模型名不再由用户填写：MLX 取目录名，GGUF 取文件名去掉扩展名。
            config.ragOCRModelName = customOCRName
        }
        config.ragOCRModelPath = ocrPath
        config.ragOCRMMProjPath = ocrMMProjPath.trimmingCharacters(in: .whitespacesAndNewlines)
        config.localVLMBackend = vlmBackend
        config.ragOCRPreprocess = ocrPreprocess
        config.ragOCRBinarize = ocrBinarize
        config.ragChunkSize = max(100, chunkSize)
        config.ragChunkOverlap = min(max(0, chunkOverlap), config.ragChunkSize / 2)
        config.ragRerankEnabled = rerankEnabled
        config.ragRerankTopN = min(max(1, rerankTopN), 20)
        config.ragSemanticChunking = semanticChunking
        config.ragAutoIngest = autoIngest
        config.ragVectorElementType = vectorElementType
        config.ragMatryoshka = matryoshkaMode
        config.ragMatryoshkaDim = [128, 256, 512].contains(matryoshkaDim) ? matryoshkaDim : 512
        config.ragEmbeddingMaxBatch = min(max(2, embeddingMaxBatch), 256)
        config.ragEmbeddingMinBatch = min(max(2, embeddingMinBatch), 64)
        config.ragEmbeddingMinBatch = min(config.ragEmbeddingMinBatch,
                                          config.ragEmbeddingMaxBatch)
        config.ragEmbeddingInitialBatch = min(max(config.ragEmbeddingMinBatch,
                                                  embeddingInitialBatch),
                                              config.ragEmbeddingMaxBatch)
        config.ragEmbeddingBatchStep = min(max(1, embeddingBatchStep), 16)
        config.ragEmbeddingMemoryLimitMB = min(max(1_024, embeddingMemoryLimitMB), 32_768)
        config.ragEmbeddingThroughputTolerance = min(max(0.01, embeddingThroughputTolerance),
                                                      0.5)
        config.ragEmbeddingProbeBatches = min(max(1, embeddingProbeBatches), 16)
        config.ragEmbeddingProbeSeconds = min(max(0.5, embeddingProbeSeconds), 30)

        if mainMode == "local", !config.localLLMModel.isEmpty,
           config.localLLMModel != model.config.localLLMModel {
            statusMessage = "已应用本地模型路径"
        }
        return config
    }

    private var apiModelChoices: [String] {
        var choices = apiModels
        let current = apiModel.trimmingCharacters(in: .whitespacesAndNewlines)
        if !current.isEmpty, !choices.contains(current) {
            choices.insert(current, at: 0)
        }
        return choices
    }

    private func fetchModels() {
        let base = baseURL.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !base.isEmpty else {
            statusMessage = "请先填写 Base URL"
            return
        }
        fetchingModels = true
        statusMessage = "正在获取模型列表…"
        let client = OpenAILLMClient(baseURL: base,
                                     apiKey: apiKey.trimmingCharacters(in: .whitespacesAndNewlines),
                                     model: apiModel)
        Task {
            do {
                let ids = try await client.listModels()
                client.close()
                await MainActor.run {
                    fetchingModels = false
                    apiModels = ids
                    if apiModel.isEmpty, let first = ids.first { apiModel = first }
                    statusMessage = ids.isEmpty
                        ? "接口没有返回模型"
                        : "已获取 \(ids.count) 个模型"
                }
            } catch {
                client.close()
                await MainActor.run {
                    fetchingModels = false
                    statusMessage = "获取模型失败：\(error.localizedDescription)"
                }
            }
        }
    }

    private func chooseDirectory(_ binding: Binding<String>, title: String) {
        let panel = NSOpenPanel()
        panel.title = title
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        let current = binding.wrappedValue.trimmingCharacters(in: .whitespacesAndNewlines)
        if !current.isEmpty {
            panel.directoryURL = URL(fileURLWithPath: (current as NSString).expandingTildeInPath)
        }
        if panel.runModal() == .OK, let path = panel.url?.path {
            binding.wrappedValue = path
        }
    }

    private func chooseFile(_ binding: Binding<String>, title: String) {
        let panel = NSOpenPanel()
        panel.title = title
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        let current = binding.wrappedValue.trimmingCharacters(in: .whitespacesAndNewlines)
        if !current.isEmpty {
            panel.directoryURL = URL(fileURLWithPath: (current as NSString).expandingTildeInPath)
                .deletingLastPathComponent()
        }
        if panel.runModal() == .OK, let path = panel.url?.path {
            binding.wrappedValue = path
        }
    }

    private func clamp(_ value: Double, _ lower: Double, _ upper: Double) -> Double {
        min(max(value, lower), upper)
    }

    /// 旧配置可能包含 openai / llama_cpp 等值；Picker 只能选中已声明的选项。
    private func normalizedLLMType(_ raw: String) -> String {
        let value = AgentConfig.normalizeLLMBackend(raw)
        return llmTypes.contains { $0.id == value } ? value : "mlx-lm"
    }

    private func normalizedEmbeddingType(_ raw: String) -> String {
        let value = AgentConfig.normalizeEmbeddingBackend(raw)
        return embeddingTypes.contains { $0.id == value } ? value : "onnx"
    }
}
