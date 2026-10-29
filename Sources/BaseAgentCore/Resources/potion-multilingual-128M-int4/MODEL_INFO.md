# potion-multilingual-128M (int4 ONNX)

- 上游模型：https://huggingface.co/minishlab/potion-multilingual-128M（MIT，见 `LICENSE`）
- ONNX 来源：本地 `models/potion-multilingual-128M-onnx/int4`（由 fp32 导出图转换的分块量化版本）
- SHA-256：
  - `model.onnx`：`0ab52d61b8319abe832664236ab02af1e8d6016251bf45eb4586da0a3b124877`
  - `tokenizer.json`：`19f1909063da3cfe3bd83a782381f040dccea475f4816de11116444a73e1b6a1`
  - `config.json`：`c5e22af4f7bf7c0ffbc8361728930d1860ef5daf45fb739c75316962b856b077`
- 结构：model2vec 静态词向量表（vocab 500353 × dim 256），int4 分块非对称量化
  （block=32，2 nibbles/byte 打包，约 80.07 MB），opset 18
- 推理图：`input_ids` / `attention_mask`（int64，动态 batch × seq）
  → `embeddings`（float32 `[batch, 256]`，已 mean-pool + L2 归一化）
- 分词：Unigram（`tokenizer.json`，500353 词表）；无特殊 token，`unk_token_id = 1`，
  与官方 model2vec 一致直接丢弃 unk；`config.json` 的 `max_length: 512` 只是官方示例的
  截断值，静态词向量没有位置编码，本实现不使用该截断
- 长度：无 512 token 位置上限（实测 3 万/10 万 token 输入仍输出单位向量，纯 CPU 约
  0.09 s / 0.30 s）；代码仅保留 8192 token 的防御性上限
- 用途：RAG 语义切片的句子相似度（纯 CPU，不联网、不占用 MLX 资源）
- 精度/实测：int4 与 fp32 余弦均值 0.998110、最小 0.996149；中文相关句余弦约 0.52，
  无关句约 0.037（bge-small-zh 约为 0.72 / 0.38；potion 中文区分度略低但速度更快、
  无长度限制）
