import XCTest
import AppKit
import PDFKit
import llama
@testable import BaseAgentCore

final class BackendVerificationTests: XCTestCase {

    func testGGUFEmbedding() async throws {
        guard let path = ProcessInfo.processInfo.environment["GGUF_EMBED_MODEL"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("未设置 GGUF_EMBED_MODEL")
        }
        let provider = LlamaCppEmbeddingProvider(modelPath: path,
                                                 modelName: "nomic-embed-text-v1.5")
        let vectors = try await provider.embedDocuments(["hello world", "你好世界"])
        XCTAssertEqual(vectors.count, 2)
        XCTAssertEqual(vectors[0].count, 768)
        XCTAssertEqual(vectors[1].count, 768)
        await provider.unload()
    }

    func testGGUFLLMChat() async throws {
        guard let path = ProcessInfo.processInfo.environment["GGUF_LLM_MODEL"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("未设置 GGUF_LLM_MODEL")
        }
        let gpuLayers = Int(ProcessInfo.processInfo.environment["GGUF_GPU_LAYERS"] ?? "0") ?? 0
        let service = LlamaCppLLMService(
            modelPath: path,
            contextWindow: 1_024,
            maxTokens: 64,
            topP: 0.95, minP: 0.0, topK: 20,
            repetitionPenalty: 1.0,
            enableThinking: false,
            gpuLayers: gpuLayers
        )
        try await service.preload()
        var content = ""
        var calls: [ToolCall] = []
        let tools: [JSONValue] = [
            .object([
                "type": .string("function"),
                "function": .object([
                    "name": .string("get_weather"),
                    "description": .string("查询城市天气"),
                    "parameters": .object([
                        "type": .string("object"),
                        "properties": .object([
                            "city": .object(["type": .string("string")]),
                        ]),
                        "required": .array([.string("city")]),
                    ]),
                ]),
            ]),
        ]
        for try await event in service.chatStream(
            messages: [ChatMessage(role: .user,
                                   content: "查询北京天气。必须调用 get_weather 工具，不要直接回答。")],
            tools: tools,
            temperature: 0
        ) {
            content += event.content
            if !event.toolCalls.isEmpty { calls = event.toolCalls }
        }
        print("GGUF content: \(content), calls: \(calls.map(\.name))")
        XCTAssertFalse(content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                       && calls.isEmpty,
                       "既没有正文也没有工具调用")
        service.close()
    }

    func testGGUFMultiTurnReuse() async throws {
        guard let path = ProcessInfo.processInfo.environment["GGUF_LLM_MODEL"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("未设置 GGUF_LLM_MODEL")
        }
        let gpuLayers = Int(ProcessInfo.processInfo.environment["GGUF_GPU_LAYERS"] ?? "0") ?? 0
        let service = LlamaCppLLMService(
            modelPath: path, contextWindow: 2_048, maxTokens: 16,
            topP: 0.95, minP: 0.0, topK: 20, repetitionPenalty: 1.0,
            enableThinking: false, gpuLayers: gpuLayers)
        try await service.preload()

        let system = ChatMessage(role: .system, content: "你是一个简洁的中文助手。")
        let firstUser = ChatMessage(role: .user, content: "用一句话介绍北京。")
        var firstRaw = ""
        for try await event in service.chatStream(
            messages: [system, firstUser], tools: [], temperature: 0) {
            if event.kind == .done { firstRaw = event.rawContent }
        }
        XCTAssertFalse(firstRaw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

        let secondUser = ChatMessage(role: .user, content: "再用一句话介绍上海。")
        var secondRaw = ""
        var cached = 0
        for try await event in service.chatStream(
            messages: [system, firstUser,
                       ChatMessage(role: .assistant, content: firstRaw),
                       secondUser],
            tools: [], temperature: 0) {
            if event.kind == .done {
                secondRaw = event.rawContent
                cached = event.usage["cached_prompt_tokens"] ?? 0
            }
        }
        print("GGUF second turn cached=\(cached), raw=\(secondRaw)")
        // 混合 SSM 模型无法按位置裁剪 KV；模板归一化 thinking 后若前缀不一致，
        // 服务会安全地整段重建，这里只要求第二轮正常产出。
        XCTAssertFalse(secondRaw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        service.close()
    }

    func testGGUFWarmupReuse() async throws {
        guard let path = ProcessInfo.processInfo.environment["GGUF_LLM_MODEL"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("未设置 GGUF_LLM_MODEL")
        }
        let gpuLayers = Int(ProcessInfo.processInfo.environment["GGUF_GPU_LAYERS"] ?? "0") ?? 0
        let service = LlamaCppLLMService(
            modelPath: path, contextWindow: 2_048, maxTokens: 8,
            topP: 0.95, minP: 0.0, topK: 20, repetitionPenalty: 1.0,
            enableThinking: false, gpuLayers: gpuLayers)
        try await service.preload()
        let system = String(
            repeating: "你是一个严谨的中文助手，回答前先确认事实，避免臆测。",
            count: 20)
        try await service.warmup(systemPrompt: system, tools: [])

        var cached = 0
        var raw = ""
        for try await event in service.chatStream(
            messages: [ChatMessage(role: .system, content: system),
                       ChatMessage(role: .user, content: "回复一个字：好")],
            tools: [], temperature: 0) {
            if event.kind == .done {
                cached = event.usage["cached_prompt_tokens"] ?? 0
                raw = event.rawContent
            }
        }
        print("GGUF warmup cached=\(cached), raw=\(raw)")
        XCTAssertGreaterThan(cached, 0, "预热后首次请求应命中 system 前缀")
        service.close()
    }

    func testMLXVLMAsTextLLM() async throws {
        guard let path = ProcessInfo.processInfo.environment["MLX_VLM_MODEL"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("未设置 MLX_VLM_MODEL")
        }
        let service = MLXVLMLLMService(
            directory: URL(fileURLWithPath: path),
            modelName: "Qwen3.5-2B",
            maxTokens: 64,
            topP: 0.95, minP: 0.0, topK: 20,
            repetitionPenalty: 1.0,
            enableThinking: false,
            contextWindow: 2048
        )
        try await service.preload()
        var content = ""
        for try await event in service.chatStream(
            messages: [ChatMessage(role: .user, content: "用一句话介绍你自己。")],
            tools: [],
            temperature: 0
        ) {
            content += event.content
        }
        print("MLX-VLM content: \(content)")
        XCTAssertFalse(content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        service.close()
    }

    /// 用一个合成的 decoder ONNX 图验证 onnxruntime-swift 的
    /// input_ids / attention_mask / past_key_values / present 管线。
    /// 图只复制最后一个输入 token 的 one-hot，不验证生成质量。
    func testONNXToyDecoderPipeline() async throws {
        guard let path = ProcessInfo.processInfo.environment["ONNX_TOY_MODEL"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("未设置 ONNX_TOY_MODEL")
        }
        let service = ONNXLLMService(
            directory: URL(fileURLWithPath: path),
            modelName: "toy-decoder",
            contextWindow: 1_024,
            maxTokens: 2,
            topP: 1.0, minP: 0.0, topK: 1,
            repetitionPenalty: 1.0,
            enableThinking: false)
        try await service.preload()
        var done = false
        for try await event in service.chatStream(
            messages: [ChatMessage(role: .user, content: "hi")],
            tools: [], temperature: 0) {
            if event.kind == .done { done = true }
        }
        XCTAssertTrue(done)
        service.close()
    }

    func testLlamaCppFrameworkLoaded() {
        let version = String(cString: llama_version())
        print("llama.cpp version: \(version)")
        XCTAssertFalse(version.isEmpty)
    }

    func testGGUFPathAndMMProjResolution() throws {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("gguf-resolve-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data().write(to: directory.appendingPathComponent("model-Q4_K_M.gguf"))
        try Data().write(to: directory.appendingPathComponent("mmproj-model-F16.gguf"))

        var config = AgentConfig()
        config.workspace = directory.path
        config.knowledgeBase = directory.path

        let resolved = try GGUFFileResolver.resolve(kind: .llm,
                                                    value: directory.path,
                                                    config: config)
        XCTAssertEqual(resolved.lastPathComponent, "model-Q4_K_M.gguf")
        let mmproj = GGUFFileResolver.resolveMMProj(modelPath: resolved.path, explicit: nil)
        XCTAssertEqual((mmproj as NSString?)?.lastPathComponent, "mmproj-model-F16.gguf")
    }

    func testONNXWithoutWeightsFailsClearly() async throws {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("onnx-no-weights-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let configJSON = """
            {"model_type":"qwen","num_hidden_layers":2,
             "num_attention_heads":4,"num_key_value_heads":2,"hidden_size":64}
            """
        try configJSON.write(to: directory.appendingPathComponent("config.json"),
                            atomically: true, encoding: .utf8)

        var config = AgentConfig()
        config.llmProvider = "local"
        config.localLLMBackend = "onnx"
        config.localLLMModel = directory.path
        let service = LocalModelFactory.shared.makeLLM(config: config)
        XCTAssertTrue(service.identifier.contains("onnx"))

        var failed = false
        do {
            for try await _ in service.chatStream(
                messages: [ChatMessage(role: .user, content: "hi")],
                tools: [],
                temperature: 0) {}
        } catch {
            failed = true
            XCTAssertTrue(error.localizedDescription.contains("ONNX")
                          || error.localizedDescription.contains("tokenizer"),
                          error.localizedDescription)
        }
        XCTAssertTrue(failed)
    }

    func testGGUFVLMRecognize() async throws {
        guard let model = ProcessInfo.processInfo.environment["GGUF_VLM_MODEL"],
              let mmproj = ProcessInfo.processInfo.environment["GGUF_VLM_MMPROJ"],
              FileManager.default.fileExists(atPath: model),
              FileManager.default.fileExists(atPath: mmproj) else {
            throw XCTSkip("未设置 GGUF_VLM_MODEL / GGUF_VLM_MMPROJ")
        }
        let image = try makeTextImage("Hello 世界 123")
        let gpuLayers = Int(ProcessInfo.processInfo.environment["GGUF_GPU_LAYERS"] ?? "0") ?? 0
        let service = LlamaCppVLMService(modelPath: model,
                                         mmprojPath: mmproj,
                                         contextWindow: 1_024,
                                         gpuLayers: gpuLayers)
        let text = try await service.recognize(
            image: image,
            prompt: "转写图片中的文字，只输出文字。",
            maxTokens: 32
        )
        print("GGUF VLM text: \(text)")
        XCTAssertFalse(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        await service.unload()
    }

    func testPDFRendererOrientation() throws {
        guard let path = ProcessInfo.processInfo.environment["PDF_ORIENT_TEST"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("未设置 PDF_ORIENT_TEST")
        }
        let document = PDFDocument(url: URL(fileURLWithPath: path))
        let page = document?.page(at: 3)
        XCTAssertNotNil(page)
        guard let image = page.flatMap({ PDFPageRenderer.render(page: $0, maxLongSide: 1408) }) else {
            XCTFail("渲染失败")
            return
        }
        let text = try VisionPDFParser.recognizeText(image: image,
                                                     languages: ["zh-Hans", "en-US"])
        let cleaned = DocumentCleaner.clean(text)
        print("PDF render OCR: \(String(cleaned.prefix(160)))")
        XCTAssertTrue(cleaned.contains("出版说明"), cleaned)
        XCTAssertTrue(cleaned.contains("汉译世界学术名著丛书"), cleaned)
        XCTAssertTrue(cleaned.contains("我馆历来重视移译世界各国学术名著"), cleaned)
        XCTAssertFalse(cleaned.contains("牛丛暮各"), cleaned)
    }

    private func makeTextImage(_ text: String) throws -> CGImage {
        let size = NSSize(width: 640, height: 200)
        let image = NSImage(size: size)
        image.lockFocus()
        NSColor.white.setFill()
        NSRect(origin: .zero, size: size).fill()
        text.draw(at: NSPoint(x: 40, y: 80),
                  withAttributes: [.font: NSFont.systemFont(ofSize: 48),
                                   .foregroundColor: NSColor.black])
        image.unlockFocus()
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw AgentError.parse("无法创建测试图片")
        }
        return cgImage
    }
}
