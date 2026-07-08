#!/usr/bin/env bash
# Vendor the Silero-VAD web assets + matching onnxruntime-web wasm into static/.
# vad-web pins onnxruntime-web@1.14.0; loading these from CDN is version-fragile,
# so we serve them from our own origin. Run once after cloning.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p static
tmp=$(mktemp -d)
mkdir "$tmp/vad" "$tmp/ort"
( cd "$tmp/vad" && npm pack @ricky0123/vad-web@0.0.22 >/dev/null && tar xzf *.tgz )
( cd "$tmp/ort" && npm pack onnxruntime-web@1.14.0     >/dev/null && tar xzf *.tgz )
cp "$tmp"/vad/package/dist/bundle.min.js \
   "$tmp"/vad/package/dist/vad.worklet.bundle.min.js \
   "$tmp"/vad/package/dist/silero_vad_legacy.onnx \
   "$tmp"/vad/package/dist/silero_vad_v5.onnx static/
cp "$tmp"/ort/package/dist/ort.min.js \
   "$tmp"/ort/package/dist/ort-wasm*.wasm \
   "$tmp"/ort/package/dist/ort-wasm*.worker.js static/
rm -rf "$tmp"
echo "static/ populated:"; ls static/
