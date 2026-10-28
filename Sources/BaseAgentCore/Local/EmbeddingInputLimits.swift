import Foundation

/// Embedding 推理的输入长度上限。
///
/// jina-embeddings-v3 声明的窗口是 8192 token，但实际切片预算默认只有
/// 800 token；推理前统一把输入压到 1024 token，只为真正会入库的切片
/// 计算向量，避免为用不到的窗口容量做无效计算。切片收口读取同一上限，
/// 防止「设置里给了长切片、推理阶段却被静默截断」。
enum EmbeddingInputLimits {
    /// 单条 embedding 输入（含特殊 token）的硬上限。
    static let maxTokens = 1_024

    /// 模型声明的窗口收口到上限；原本更小的模型保持原窗口。
    static func cappedWindow(_ window: Int) -> Int {
        max(16, min(window, maxTokens))
    }
}
