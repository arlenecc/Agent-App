# 内置 jina-reranker：RAG 开箱可用设计

## 1. 背景与目标

当前 App 已内置默认 Embedding `jina-embeddings-v5-text-nano-mlx-int4`，但未内置默认 Reranker
`jina-reranker-v2-base-multilingual-onnx`。打包后的 App 缺少该模型目录，默认配置解析失败，
`LocalModelFactory` 返回 `FailingRerankProvider`，检索时重排异常并回退到 RRF 排序。

目标：把 Reranker 一并打包进 App，使首次启动用户只需配置主模型与辅助模型，RAG 的
“切片 → 向量入库 → 混合检索 → 重排”全链路即可工作。

已确认的体积取舍：接受 App 从约 356MB 增至约 640MB、DMG 从约 235MB 增至约 480–520MB。

## 2. 方案

### 2.1 打包清单

`scripts/package-app.sh` 新增 reranker 打包块，目标目录：

```
BaseAgent.app/Contents/Resources/models/jina-reranker-v2-base-multilingual-onnx/
```

只复制运行时必需文件：

| 文件 | 用途 |
|---|---|
| `model_quantized.onnx` | ONNX int8 交叉编码器权重 |
| `tokenizer.json` | pair 编码 |
| `tokenizer_config.json` | XLM-R 最大长度与 tokenizer 元数据 |
| `special_tokens_map.json` | 特殊 token 映射 |
| `config.json` | 模型配置 |

排除 Python 参考脚本、README、`.gitignore`、`.DS_Store` 等非运行时文件。源目录缺失时
输出警告，避免生成“看似可用但实际缺模型”的包。

### 2.2 默认解析与配置

沿用现有默认值，不新增配置项：

- `rerankerProvider = "local"`
- `localRerankerBackend = "onnx"`
- `localRerankerModel = "jina-reranker-v2-base-multilingual-onnx"`
- `localRerankerModelPath = "models/jina-reranker-v2-base-multilingual-onnx"`
- `ragRerankEnabled = true`，阈值 `0.45`，TopN `5`

`LocalModelRegistry.applicationModelsDirectory()` 已优先返回
`Contents/Resources/models`，`resolveDirectory(kind: .reranker, value: "models/jina-...")`
会锚定到该目录。路径留空时，`LocalModelFactory.makeReranker` 回落到
`LocalModelRegistry.defaultName(for: .reranker)`，同样命中内置模型。

设置界面保留预填路径，并补充“留空/默认即使用 App 内置 reranker”的说明。

### 2.3 错误处理

内置文件缺失、ONNX Runtime 加载失败或 tokenizer 异常时，继续使用现有
`FailingRerankProvider`；`RAGEngine.search` 捕获后记录警告并回退 RRF 排序，检索不中断。

## 3. 测试

1. **Bundle 解析单测**：在临时目录模拟 `Contents/Resources/models`，用
   `BASE_AGENT_MODELS_DIR` 锚定模型根，验证默认配置经 `resolveDirectory` 与
   `makeReranker` 得到 `ONNXReranker` 而非 `FailingRerankProvider`。
2. **真实模型 E2E**：模型存在时加载 jina reranker，断言输出 3 个 `[0,1]` 分数，
   相关段落分数高于算式短句基线；模型缺失时 `XCTSkip`。
3. **回归**：`swift build && swift test` 全量通过。
4. **打包验证**：执行 `scripts/package-app.sh release`，确认 App 内 reranker 文件齐全，
   记录 App 体积；生成 DMG 并用 `hdiutil verify` + 挂载检查。

## 4. 验收标准

- 打包后的 `BaseAgent.app` 含完整 reranker 运行时文件。
- 默认配置无需用户填写 reranker 路径即可创建可用 `ONNXReranker`。
- 模型缺失时检索仍可回退 RRF，并给出明确日志。
- 全量测试通过；DMG 可校验、可挂载，内含 App 与内置 reranker。
- README 与设计文档说明内置模型与包体积变化。
