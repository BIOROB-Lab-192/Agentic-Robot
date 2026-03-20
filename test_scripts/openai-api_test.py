import base64
from io import BytesIO

from openai import OpenAI
from PIL import Image


def preprocess_image(image_path, max_size=(1000, 1000)):
    with Image.open(image_path) as img:
        img = img.convert("RGB")

        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


image_path = "test_images/robot_test2.png"
base64_image = preprocess_image(image_path)

openai_client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="sk-no-key-required",
)
completion = openai_client.chat.completions.create(
    model="models/Qwen3.5-4B-Q4_K_M.gguf",
    # messages=[
    #     {"role": "user", "content": "What is 2+2?"},
    # ],
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is going on in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }
    ],
)
print(completion.choices[0].message.content)
# print(completion.choices[0].message.reasoning_content)
