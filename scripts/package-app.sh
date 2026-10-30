#!/bin/bash
# 把 SwiftPM 产物打包成可分发的 BaseAgent.app。
#
# 用法：scripts/package-app.sh [debug|release]（默认 release）
# 产物：dist/BaseAgent.app（含 App 图标、资源 bundle 与 llama/onnxruntime 框架）
set -euo pipefail

CONFIG="${1:-release}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PRODUCTS="$ROOT/.build/out/Products/$([[ "$CONFIG" == "release" ]] && echo Release || echo Debug)"
BINARY="$PRODUCTS/BaseAgent"
DIST="$ROOT/dist"
APP="$DIST/BaseAgent.app"

if [[ ! -x "$BINARY" ]]; then
  echo "未找到可执行文件：$BINARY（先运行 swift build -c $CONFIG --disable-sandbox）" >&2
  exit 1
fi

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources" "$APP/Contents/Frameworks"

cp "$BINARY" "$APP/Contents/MacOS/BaseAgent"

# SwiftPM 资源 bundle：Bundle.module 会依次查找 Bundle.main.resourceURL
# （即 Contents/Resources）与 Bundle.main.bundleURL，因此放在 Resources 下。
for bundle in "$PRODUCTS"/*.bundle; do
  [[ -e "$bundle" ]] || continue
  cp -R "$bundle" "$APP/Contents/Resources/"
done

# 默认 Embedding：jina-embeddings-v5-text-nano int4 MLX 权重随 App 分发
#（约 143MB）。未显式配置 embedding 模型时开箱可用；只拷贝运行时必需文件，
# 不把示例脚本与 __pycache__ 打进包。
EMBEDDING_NAME="jina-embeddings-v5-text-nano-mlx-int4"
EMBEDDING_SRC="$ROOT/models/$EMBEDDING_NAME"
EMBEDDING_DST="$APP/Contents/Resources/models/$EMBEDDING_NAME"
if [[ -d "$EMBEDDING_SRC" ]]; then
  mkdir -p "$EMBEDDING_DST"
  for file in config.json model.safetensors weights.safetensors \
              tokenizer.json tokenizer_config.json README.md; do
    if [[ -f "$EMBEDDING_SRC/$file" ]]; then
      cp "$EMBEDDING_SRC/$file" "$EMBEDDING_DST/"
    fi
  done
  echo "已打包默认 embedding：$EMBEDDING_NAME"
else
  echo "警告：未找到 $EMBEDDING_SRC，App 内不含默认 embedding 模型" >&2
fi

# 默认 Reranker：jina-reranker-v2-base-multilingual ONNX int8 随 App 分发
#（model_quantized.onnx 约 267MB）。与 embedding 一样只复制运行时必需文件，
# 不把 Python 参考脚本与缓存文件打进包。
RERANKER_NAME="jina-reranker-v2-base-multilingual-onnx"
RERANKER_SRC="$ROOT/models/$RERANKER_NAME"
RERANKER_DST="$APP/Contents/Resources/models/$RERANKER_NAME"
if [[ -d "$RERANKER_SRC" ]]; then
  mkdir -p "$RERANKER_DST"
  for file in model_quantized.onnx tokenizer.json tokenizer_config.json \
              special_tokens_map.json config.json; do
    if [[ -f "$RERANKER_SRC/$file" ]]; then
      cp "$RERANKER_SRC/$file" "$RERANKER_DST/"
    fi
  done
  echo "已打包默认 reranker：$RERANKER_NAME"
else
  echo "警告：未找到 $RERANKER_SRC，App 内不含默认 reranker" >&2
fi

# 动态框架：llama（GGUF 进程内推理）。onnxruntime 在本项目中静态链接，
# 不需要单独拷贝。
if [[ -e "$PRODUCTS/llama.framework" ]]; then
  cp -R "$PRODUCTS/llama.framework" "$APP/Contents/Frameworks/"
fi

# 图标：优先用项目根目录的 icns
if [[ -f "$ROOT/BaseAgent.icns" ]]; then
  cp "$ROOT/BaseAgent.icns" "$APP/Contents/Resources/AppIcon.icns"
fi

cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>BaseAgent</string>
  <key>CFBundleDisplayName</key><string>BaseAgent</string>
  <key>CFBundleExecutable</key><string>BaseAgent</string>
  <key>CFBundleIdentifier</key><string>com.baseagent.desktop</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSPrincipalClass</key><string>NSApplication</string>
  <key>LSApplicationCategoryType</key><string>public.app-category.productivity</string>
</dict>
</plist>
PLIST

# 依赖的框架通过 @rpath 解析：可执行文件与框架都在包内。
install_name_tool -add_rpath "@executable_path/../Frameworks" "$APP/Contents/MacOS/BaseAgent" 2>/dev/null || true

# 本地运行需要放宽沙箱（模型/知识库位于外部卷），如要上架需替换为正式签名。
codesign --force --deep --sign - "$APP" 2>&1 | tail -2

echo "已生成：$APP"
du -sh "$APP"