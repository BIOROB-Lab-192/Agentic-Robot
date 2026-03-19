#!/bin/bash

THINKING=false

for arg in "$@"; do
  if [ "$arg" = "--thinking" ]; then
    THINKING=true
  fi
done

if [ "$THINKING" = true ]; then
  TEMP=1.0
  TOP_P=0.95
  TOP_K=20
  MIN_P=0.0
  PRESENCE_PENALTY=1.5
  THINKING_FLAG='{"enable_thinking":true}'
else
  TEMP=0.7
  TOP_P=0.8
  TOP_K=20
  MIN_P=0.0
  PRESENCE_PENALTY=1.5
  THINKING_FLAG='{"enable_thinking":false}'
fi

ARGS=(
  -m models/Qwen3.5-4B-Q4_K_M.gguf
  --mmproj models/mmproj-F16.gguf
  --ctx-size 16384
  --temp "$TEMP"
  --top-p "$TOP_P"
  --top-k "$TOP_K"
  --min-p "$MIN_P"
  --presence-penalty "$PRESENCE_PENALTY"
  --port 8080
  --jinja
  --chat-template-kwargs "$THINKING_FLAG"
)

llama-server "${ARGS[@]}"
