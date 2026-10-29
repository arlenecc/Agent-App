# 入库前丢弃过短正文设计

日期：2026-10-29

## 背景

部分 EPUB/PDF 抽取失败后只剩书名、目录名或图片元数据，生成的 Markdown 只有很少的字节。这类碎片会被正常切片、向量化并写入知识库，检索时占用结果位、污染召回，还会进入后台逐章摘要的候选队列。

## 目标

1. 在切片/向量化之前识别「正文文字过短」的文档，整份丢弃，不写切片、向量与 `documents`。
2. 丢弃决定可复用：文件内容未变化时后续同步直接跳过，不重复解析。
3. 文件之后被修复、内容变长时，下一次同步能自动重新解析入库。
4. 丢弃结果在同步统计与日志中可见。

## 非目标

- 不主动清理已经入库的历史短文档（只对本次及以后解析的文件生效）。
- 不提供可配置阈值；阈值固定为 100。
- 不修改解析器或清洗规则。

## 判定规则

「正文文字」= 去掉标题/目录行后的非空白字符数：

- 用 `DocumentOutline.extract` 识别标题行（Markdown `#` 标题、`第X章/节`、`Chapter N`、数字编号、中文序号等），这些行整行不计入正文；
- 其余行按非空白字符（`unicodeScalars` 忽略 `isWhitespace`）累加，含标点与数字；
- **正文文字 < 100 → 丢弃**；≥ 100 正常入库。

因此只有书名/目录名的文件正文为 0 或个位数，必然丢弃；正常书籍即使标题很长，正文也远超阈值。

## 流程

插入点：`RAGEngine.processFile` 中解析、清洗得到 `text` 且非空之后，正文去重 claim 与切片之前。

1. `DocumentOutline.extract(text)` 计算一次标题节点，供丢弃判定与 `buildDigest` 复用（避免重复抽取）。
2. 统计正文文字数。
3. 若 `< 100`：
   - 写 manifest 标记：`chunkCount: 0, complete: true, mdPath: "", digest: "", discarded: true`；
   - `digest` 留空，避免该条目参与全局正文去重（否则多个同文碎片会互相写 tombstone，下一轮又被重新解析）；
   - 若该来源此前已入库（例如文件被改短），调用 `removeIndexedSource` 清掉旧切片、Markdown 缓存、`documents` 行与摘要；
   - 设置 `outcome.discarded`，记日志后返回，不进入切片/embedding。
4. `canSkip` 在 `ingestSignature` / `complete` 检查之后、`chunkCount` 检查之前识别 `discarded == true`：mtime、size（以及开启 `rag_verify_fingerprint` 时的采样指纹）都未变化时直接跳过并记「内容过短，已丢弃」；任一项变化则按普通文件重新解析。该分支必须自己校验 mtime/size，不能提前到这些检查之前。
5. `force` 重建会重新解析并再次丢弃，行为一致；文件 mtime/size 或入库签名变化后自动重新解析。

## 数据模型

- `RAGManifestEntry` 新增可选字段 `discarded: Bool?`（默认 `nil`，兼容旧 manifest）。
- `IngestStats` 新增 `discarded` 计数，摘要中显示「丢弃过短 N」。
- `IngestOutcome` 新增 `discarded` 标记，用于汇总统计。

## 日志

- 首次丢弃：`RAG: 丢弃 xxx（正文仅 N 字，视为无效碎片）`。
- 后续跳过：`RAG: 跳过（内容过短，已丢弃）`。
- 不写入 `stats.errors`，不算失败。

## 边界

- 有失败页但正文 `< 100` 的文件同样丢弃并标记完成，不再重试；若文件后来变化（含解析器/配置签名变化）会重新解析。
- 正文去重登记在丢弃判定之后，不会产生指向空来源的 tombstone。
- 丢弃文件不会进入 `documents`，因此也不会成为后台摘要候选。

## 测试

在 `RAGEngineTests` 中覆盖：

1. 短文件被丢弃：`stats.discarded == 1`、源无切片、无文档记录；再次同步命中 manifest 跳过，不重新解析。
2. 已入库文件被改短后重新解析：旧切片、文档记录与 Markdown 缓存被清除。
3. 阈值边界：正文 99 字丢弃，100 字正常入库。

既有用例中的合成文档原本普遍不足 100 字（超时、ANN、大纲、向量配置、Embedding E2E 等）。为保持这些用例各自的验证目标，已把它们的夹具正文补足到阈值以上；新增的 3 个用例专门覆盖丢弃行为。
