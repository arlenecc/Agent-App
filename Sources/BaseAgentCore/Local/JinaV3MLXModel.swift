import Foundation
import MLX

/// jina-embeddings-v3 的 MLX 推理实现。
///
/// 结构对应模型自带的 `example/jina_v3_mlx.py`：
/// `word_emb + token_type_emb[0] → emb_ln → 24 × post-norm block`
/// `（RoPE + task LoRA 的 MHA、GELU MLP）→ mean pooling → L2 归一化`。
///
/// MLXEmbedders 默认不支持 `jina-v3-mlx` 架构，因此这里直接读取
/// `weights.safetensors`（int4/int8 混合量化 + 5 组 task LoRA）。
final class JinaV3MLXModel: @unchecked Sendable {
    private struct QuantizedLinear: @unchecked Sendable {
        let weight: MLXArray
        let scales: MLXArray
        let biases: MLXArray
        let bias: MLXArray?
        let groupSize: Int
        let bits: Int
        let mode: QuantizationMode

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var output = quantizedMM(
                x, weight,
                scales: scales, biases: biases,
                transpose: true,
                groupSize: groupSize, bits: bits, mode: mode
            )
            output = output.asType(.float32)
            if let bias { output = output + bias }
            return output
        }
    }

    /// Linear LoRA：`A` 为 [tasks, rank, in]，`B` 为 [tasks, out, rank]，
    /// delta = B[t] @ A[t]，与官方实现的 `((x @ A.T) @ B.T)` 等价。
    private struct LinearLoRA: @unchecked Sendable {
        let a: MLXArray
        let b: MLXArray

        func apply(_ x: MLXArray, task: Int, scaling: Float) -> MLXArray {
            let aT = a[task].asType(.float32).transposed()
            let bT = b[task].asType(.float32).transposed()
            return matmul(matmul(x, aT), bT) * scaling
        }
    }

    /// Embedding LoRA：`A` 为 [tasks, vocab, rank]，`B` 为 [tasks, rank, dim]，
    /// delta = A[t][ids] @ B[t]。
    private struct EmbeddingLoRA: @unchecked Sendable {
        let a: MLXArray
        let b: MLXArray

        func apply(ids: MLXArray, task: Int, scaling: Float) -> MLXArray {
            let embedding = a[task].asType(.float32)[ids]
            let projection = b[task].asType(.float32)
            return matmul(embedding, projection) * scaling
        }
    }

    /// token_type embedding 的 LoRA（type vocab 只有 1，固定取第 0 行）。
    private struct TokenTypeLoRA: @unchecked Sendable {
        let a: MLXArray
        let b: MLXArray

        func apply(task: Int, scaling: Float) -> MLXArray {
            let row = a[task].asType(.float32)[0].reshaped([1, -1])
            let projection = b[task].asType(.float32)
            return matmul(row, projection).reshaped([-1]) * scaling
        }
    }

    private struct Layer: @unchecked Sendable {
        let wqkv: QuantizedLinear
        let outProjection: QuantizedLinear
        let fc1: QuantizedLinear
        let fc2: QuantizedLinear

        let norm1Weight: MLXArray
        let norm1Bias: MLXArray
        let norm2Weight: MLXArray
        let norm2Bias: MLXArray

        let loraWqkv: LinearLoRA
        let loraOut: LinearLoRA
        let loraFc1: LinearLoRA
        let loraFc2: LinearLoRA

        func callAsFunction(
            _ x: MLXArray,
            additiveMask: MLXArray,
            task: Int,
            configuration: JinaV3ModelConfiguration,
            inverseFrequencies: MLXArray
        ) -> MLXArray {
            let batch = x.dim(0)
            let sequence = x.dim(1)
            let heads = configuration.headCount
            let headDimension = configuration.headDimension
            let scaling = configuration.loraScaling

            var qkv = wqkv(x)
            qkv = qkv + loraWqkv.apply(x, task: task, scaling: scaling)
            qkv = qkv.reshaped([batch, sequence, 3, heads, headDimension])
                .asType(.float32)

            let parts = split(qkv, parts: 3, axis: 2)
            var queries = parts[0].reshaped([batch, sequence, heads, headDimension])
            var keys = parts[1].reshaped([batch, sequence, heads, headDimension])
            let values = parts[2].reshaped([batch, sequence, heads, headDimension])

            // GPT-NeoX 半旋转 RoPE：只作用于 Q/K。
            let half = headDimension / 2
            let positions = MLXArray(0 ..< sequence).asType(.float32)
            let frequencies = positions.expandedDimensions(axis: 1)
                * inverseFrequencies.expandedDimensions(axis: 0)
            let cosine = frequencies.cos().expandedDimensions(axes: [0, 2])
            let sine = frequencies.sin().expandedDimensions(axes: [0, 2])
            queries = Self.rotateHalf(queries, cosine: cosine, sine: sine, half: half)
            keys = Self.rotateHalf(keys, cosine: cosine, sine: sine, half: half)

            let queriesH = queries.transposed(0, 2, 1, 3)
            let keysH = keys.transposed(0, 2, 1, 3)
            let valuesH = values.transposed(0, 2, 1, 3)

            let scale = 1 / Float(headDimension).squareRoot()
            var scores = matmul(queriesH * scale, keysH.transposed(0, 1, 3, 2))
            scores = scores + additiveMask
            let attention = softmax(scores, axis: -1)
            let attended = matmul(attention, valuesH)
                .transposed(0, 2, 1, 3)
                .reshaped([batch, sequence, heads * headDimension])

            var projected = outProjection(attended)
            projected = projected + loraOut.apply(attended, task: task, scaling: scaling)

            // post-norm：norm1(x + MHA(x))
            var hidden = layerNorm(
                x + projected,
                weight: norm1Weight, bias: norm1Bias,
                eps: configuration.layerNormEps
            )

            var expanded = fc1(hidden)
            expanded = expanded + loraFc1.apply(hidden, task: task, scaling: scaling)
            expanded = Self.gelu(expanded)

            var mlp = fc2(expanded)
            mlp = mlp + loraFc2.apply(expanded, task: task, scaling: scaling)

            hidden = layerNorm(
                hidden + mlp,
                weight: norm2Weight, bias: norm2Bias,
                eps: configuration.layerNormEps
            )
            return hidden
        }

        private static func rotateHalf(
            _ x: MLXArray, cosine: MLXArray, sine: MLXArray, half: Int
        ) -> MLXArray {
            let first = x[.ellipsis, 0 ..< half]
            let second = x[.ellipsis, half ..< (half * 2)]
            return concatenated(
                [first * cosine - second * sine, second * cosine + first * sine],
                axis: -1
            )
        }

        /// 精确 GELU（erf），对应 `F.gelu(approximate='none')`。
        private static func gelu(_ x: MLXArray) -> MLXArray {
            x * 0.5 * (1 + erf(x * (1 / Float(2).squareRoot())))
        }
    }

    let configuration: JinaV3ModelConfiguration

    private let wordEmbedding: MLXArray
    private let wordScales: MLXArray
    private let wordBiases: MLXArray
    private let tokenTypeEmbedding: MLXArray
    private let embeddingNormWeight: MLXArray
    private let embeddingNormBias: MLXArray
    private let loraEmbedding: EmbeddingLoRA
    private let loraTokenType: TokenTypeLoRA
    private let layers: [Layer]
    private let inverseFrequencies: MLXArray
    private let quantizationMode: QuantizationMode

    init(directory: URL) throws {
        let configuration = try JinaV3ModelConfiguration.load(from: directory)
        guard configuration.unified else {
            throw LocalModelError.unsupportedArchitecture(
                "Jina v3 需要 unified 权重（config.json 中 unified 应为 true）：\(directory.path)")
        }
        guard let weightsURL = JinaV3Profile.weightsURL(in: directory) else {
            throw LocalModelError.runtime(
                "Jina v3 缺少 MLX 权重 weights.safetensors / model.safetensors（目录：\(directory.path)）")
        }

        let weights: [String: MLXArray]
        do {
            // safetensors 的 Load primitive 仅在 CPU 实现；放到 GPU 流上求值会
            // 直接触发 `Load::eval_gpu Not implemented`。默认流就是 CPU。
            weights = try MLX.loadArrays(url: weightsURL)
        } catch {
            throw LocalModelError.runtime(
                "Jina v3 权重加载失败（\(weightsURL.path)）：\(error.localizedDescription)")
        }

        let mode = QuantizationMode(rawValue: configuration.quantizationMode) ?? .affine
        let hidden = configuration.hiddenSize

        self.configuration = configuration
        self.quantizationMode = mode
        self.wordEmbedding = try Self.require(weights, "word_embeddings.weight")
        self.wordScales = try Self.require(weights, "word_embeddings.scales")
        self.wordBiases = try Self.require(weights, "word_embeddings.biases")

        // type vocab 为 1，直接存成 [hidden] 便于广播。
        self.tokenTypeEmbedding = try Self.require(
            weights, "token_type_embeddings.weight").reshaped([hidden]).asType(.float32)
        self.embeddingNormWeight = try Self.require(
            weights, "emb_ln.weight").asType(.float32)
        self.embeddingNormBias = try Self.require(
            weights, "emb_ln.bias").asType(.float32)

        self.loraEmbedding = EmbeddingLoRA(
            a: try Self.require(weights, "lora_emb_A"),
            b: try Self.require(weights, "lora_emb_B"))
        self.loraTokenType = TokenTypeLoRA(
            a: try Self.require(weights, "lora_tt_A"),
            b: try Self.require(weights, "lora_tt_B"))

        self.layers = try (0 ..< configuration.layerCount).map { index in
            try Self.makeLayer(
                index: index, weights: weights,
                configuration: configuration, mode: mode)
        }

        let half = configuration.headDimension / 2
        let frequencies = (0 ..< half).map { index -> Float in
            1 / pow(configuration.rotaryBase, Float(index * 2) / Float(configuration.headDimension))
        }
        self.inverseFrequencies = MLXArray(frequencies)
    }

    /// 执行一次前向并返回已 L2 归一化的句向量。
    ///
    /// - Parameters:
    ///   - inputIDs / attentionMask: 等长的 batch（右 padding）。
    ///   - task: 决定使用哪组 task LoRA。
    func encode(inputIDs: [[Int]], attentionMask: [[Int]], task: JinaV3Task) -> [[Float]] {
        guard let first = inputIDs.first, !first.isEmpty else { return [] }
        let ids = MLXArray(inputIDs.map { row in row.map(Int32.init) })
        let mask = MLXArray(attentionMask.map { row in row.map(Float32.init) })

        var hidden = dequantized(
            wordEmbedding[ids],
            scales: wordScales[ids], biases: wordBiases[ids],
            groupSize: configuration.groupSize, bits: configuration.bits,
            mode: quantizationMode, dtype: .float32
        )
        hidden = hidden + loraEmbedding.apply(
            ids: ids, task: task.rawValue, scaling: configuration.loraScaling)

        var typeVector = tokenTypeEmbedding
        typeVector = typeVector + loraTokenType.apply(
            task: task.rawValue, scaling: configuration.loraScaling)
        hidden = hidden + typeVector
        hidden = layerNorm(
            hidden,
            weight: embeddingNormWeight, bias: embeddingNormBias,
            eps: configuration.layerNormEps
        )

        // (B,S) 的 0/1 mask → (B,1,1,S) 的加性 mask（与官方 -10000 偏置一致）。
        let additive = MLXArray(attentionMask.map { row in
            row.map { Float32($0 == 0 ? -10_000 : 0) }
        }).expandedDimensions(axes: [1, 2])

        for layer in layers {
            hidden = layer(
                hidden,
                additiveMask: additive,
                task: task.rawValue,
                configuration: configuration,
                inverseFrequencies: inverseFrequencies
            )
        }

        hidden = hidden.asType(.float32)
        let floatMask = mask.expandedDimensions(axis: -1)
        let pooled = (hidden * floatMask).sum(axis: 1)
            / MLX.maximum(
                floatMask.sum(axis: 1),
                MLXArray(1e-9, dtype: .float32))
        let normalized = pooled.l2Normalized(axis: -1, eps: 1e-12)
        normalized.eval()
        return normalized.map { $0.asArray(Float.self) }
    }

    // MARK: - 权重装配

    private static func require(
        _ weights: [String: MLXArray], _ key: String
    ) throws -> MLXArray {
        guard let value = weights[key] else {
            throw LocalModelError.runtime("Jina v3 权重缺少 `\(key)`")
        }
        return value
    }

    private static func makeLayer(
        index: Int,
        weights: [String: MLXArray],
        configuration: JinaV3ModelConfiguration,
        mode: QuantizationMode
    ) throws -> Layer {
        let prefix = "layers.\(index)"
        let mixer = prefix + ".mixer"
        let mlp = prefix + ".mlp"

        return Layer(
            wqkv: try makeLinear(mixer + ".Wqkv", weights: weights,
                                 configuration: configuration, mode: mode),
            outProjection: try makeLinear(mixer + ".out_proj", weights: weights,
                                          configuration: configuration, mode: mode),
            fc1: try makeLinear(mlp + ".fc1", weights: weights,
                                configuration: configuration, mode: mode),
            fc2: try makeLinear(mlp + ".fc2", weights: weights,
                                configuration: configuration, mode: mode),
            norm1Weight: try require(weights, prefix + ".norm1.weight").asType(.float32),
            norm1Bias: try require(weights, prefix + ".norm1.bias").asType(.float32),
            norm2Weight: try require(weights, prefix + ".norm2.weight").asType(.float32),
            norm2Bias: try require(weights, prefix + ".norm2.bias").asType(.float32),
            loraWqkv: LinearLoRA(
                a: try require(weights, mixer + ".lora_Wqkv_A"),
                b: try require(weights, mixer + ".lora_Wqkv_B")),
            loraOut: LinearLoRA(
                a: try require(weights, mixer + ".lora_out_A"),
                b: try require(weights, mixer + ".lora_out_B")),
            loraFc1: LinearLoRA(
                a: try require(weights, mlp + ".lora_fc1_A"),
                b: try require(weights, mlp + ".lora_fc1_B")),
            loraFc2: LinearLoRA(
                a: try require(weights, mlp + ".lora_fc2_A"),
                b: try require(weights, mlp + ".lora_fc2_B"))
        )
    }

    private static func makeLinear(
        _ prefix: String,
        weights: [String: MLXArray],
        configuration: JinaV3ModelConfiguration,
        mode: QuantizationMode
    ) throws -> QuantizedLinear {
        QuantizedLinear(
            weight: try require(weights, prefix + ".weight"),
            scales: try require(weights, prefix + ".scales"),
            biases: try require(weights, prefix + ".biases"),
            bias: try require(weights, prefix + ".bias").asType(.float32),
            groupSize: configuration.groupSize,
            bits: configuration.bits(for: prefix),
            mode: mode
        )
    }
}
