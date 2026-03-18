llama-server -m models/Youtu-VL-4B-Instruct-Q8_0.gguf \
  --mmproj models/mmproj-Youtu-VL-4b-Instruct-BF16.gguf \
  --port 8080 \
  --image-max-tokens 2048 \
  --temp 0.1 \
  --top-p 0.001 \
  --repeat-penalty 1.05 \
  -n 12280 \
  --host 0.0.0.0