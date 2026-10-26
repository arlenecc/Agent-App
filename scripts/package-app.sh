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