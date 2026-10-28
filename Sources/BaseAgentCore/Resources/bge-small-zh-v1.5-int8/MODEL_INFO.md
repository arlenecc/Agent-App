# bge-small-zh-v1.5 (int8 ONNX)

- ONNX 来源：https://huggingface.co/Xenova/bge-small-zh-v1.5（`onnx/model_int8.onnx`）
- 上游权重：https://huggingface.co/BAAI/bge-small-zh-v1.5（MIT，见 `LICENSE`）
- SHA-256：`b9837c19ce154ff0726d398ee77abbc03a7faf0476c6f93016c84e531be7ebb5`
- 结构：4 层 BERT，hidden 512，最大 512 token
- 分词：`vocab.txt` + `bert-base-chinese` 词表（WordPiece，`do_lower_case=false`，中文逐字切分）
- 池化：`[CLS]` 向量 + L2 归一化；相邻句相似度取余弦
- 用途：RAG 语义切片（纯 CPU，不联网、不占用 MLX 资源）
- 实测：中文相关句余弦约 0.72，无关句约 0.38；Swift/ORT 纯 CPU 200 句约 0.6 s（batch 32）
