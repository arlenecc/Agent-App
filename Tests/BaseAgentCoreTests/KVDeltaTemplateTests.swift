import XCTest
import Tokenizers
@testable import BaseAgentCore

/// 回归：KV 复用只渲染增量时，tool 结果必须与完整 prompt 的尾部逐 token 对齐。
///
/// 只加载 tokenizer（不加载模型权重）；模型目录不存在时跳过。
final class KVDeltaTemplateTests: XCTestCase {
    private let context: [String: any Sendable] = ["enable_thinking": true]
    private let tools: [[String: any Sendable]] = [[
        "type": "function",
        "function": [
            "name": "shell_run",
            "description": "run shell",
            "parameters": ["type": "object", "properties": [:]],
        ],
    ]]

    func testSingleToolDeltaMatchesFullPromptTail() async throws {
        let base = try await loadTokenizer()

        let full: [[String: any Sendable]] = [
            ["role": "system", "content": "S"],
            ["role": "user", "content": "U"],
            ["role": "assistant", "content": "calling", "tool_calls": [[
                "function": ["name": "shell_run", "arguments": ["command": "ls"]],
            ]]],
            ["role": "tool", "content": "ok", "name": "shell_run", "tool_call_id": "c1"],
            ["role": "user", "content": "U2"],
        ]
        let delta: [[String: any Sendable]] = [
            ["role": "tool", "content": "ok", "name": "shell_run", "tool_call_id": "c1"],
            ["role": "user", "content": "U2"],
        ]
        let wrapped: [[String: any Sendable]] = [
            ["role": "user", "content": "<tool_response>\nok\n</tool_response>"],
            ["role": "user", "content": "U2"],
        ]

        try assertRewrittenDelta(full: full, delta: delta, wrapped: wrapped, base: base)
    }

    func testConsecutiveToolDeltaMatchesFullPromptTail() async throws {
        let base = try await loadTokenizer()

        let full: [[String: any Sendable]] = [
            ["role": "system", "content": "S"],
            ["role": "user", "content": "U"],
            ["role": "assistant", "content": "calling", "tool_calls": [
                ["function": ["name": "shell_run", "arguments": ["command": "ls"]]],
                ["function": ["name": "shell_run", "arguments": ["command": "pwd"]]],
            ]],
            ["role": "tool", "content": "a", "name": "shell_run", "tool_call_id": "c1"],
            ["role": "tool", "content": "b", "name": "shell_run", "tool_call_id": "c2"],
            ["role": "user", "content": "U2"],
        ]
        let delta: [[String: any Sendable]] = [
            ["role": "tool", "content": "a", "name": "shell_run", "tool_call_id": "c1"],
            ["role": "tool", "content": "b", "name": "shell_run", "tool_call_id": "c2"],
            ["role": "user", "content": "U2"],
        ]
        let wrapped: [[String: any Sendable]] = [[
            "role": "user",
            "content": "<tool_response>\na\n</tool_response>\n"
                + "<tool_response>\nb\n</tool_response>",
        ], [
            "role": "user", "content": "U2",
        ]]

        try assertRewrittenDelta(full: full, delta: delta, wrapped: wrapped, base: base)
    }

    func testWarmupSystemPrefixMatchesFirstPrompt() async throws {
        let base = try await loadTokenizer()
        let system: [String: any Sendable] = [
            "role": "system",
            "content": "You are base-agent. " + String(repeating: "稳定系统提示。", count: 200),
        ]
        let user: [String: any Sendable] = ["role": "user", "content": "U"]
        let full = try base.applyChatTemplate(
            messages: [system, user], tools: tools, additionalContext: context)
        let prefix = try base.applyChatTemplate(
            messages: [system],
            chatTemplate: nil,
            addGenerationPrompt: false,
            truncation: false,
            maxLength: nil,
            tools: tools,
            additionalContext: context)

        XCTAssertGreaterThan(prefix.count, 100)
        XCTAssertLessThan(prefix.count, full.count)
        XCTAssertEqual(Array(full.prefix(prefix.count)), prefix,
                       "预热前缀必须是首次正式 prompt 的逐 token 前缀")
    }

    private func assertRewrittenDelta(
        full: [[String: any Sendable]],
        delta: [[String: any Sendable]],
        wrapped: [[String: any Sendable]],
        base: any Tokenizers.Tokenizer
    ) throws {
        let fullTokens = try base.applyChatTemplate(
            messages: full, tools: tools, additionalContext: context)
        let deltaTokens = try base.applyChatTemplate(
            messages: delta, tools: nil, additionalContext: context)
        let wrappedTokens = try base.applyChatTemplate(
            messages: wrapped, tools: nil, additionalContext: context)

        XCTAssertEqual(Array(fullTokens.suffix(wrappedTokens.count)), wrappedTokens,
                       "改写后的增量应与完整 prompt 尾部逐 token 一致")
        XCTAssertEqual(wrappedTokens.count, deltaTokens.count + 2,
                       "改写后只应多出 <|im_start|>user 头")
        XCTAssertEqual(Array(wrappedTokens.suffix(deltaTokens.count)), deltaTokens,
                       "除头部外应与直接渲染完全一致")
    }

    private func loadTokenizer() async throws -> any Tokenizers.Tokenizer {
        let directory = URL(fileURLWithPath:
            "/Users/arlenecc/codebase/Base-Aug-Agent/models/qwen3.5-4b-agentic-coder-v4-4bit-mlx")
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw XCTSkip("default model not installed")
        }
        return try await LocalTokenizer.load(directory: directory).base
    }
}
