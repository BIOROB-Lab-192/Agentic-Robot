import base64
from io import BytesIO

from openai import OpenAI
from PIL import Image


def preprocess_image(img, max_size=(1024, 1024)):
    img = img.convert("RGB")
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class LLMinterface:
    def __init__(self, model):
        self.openai_client = OpenAI(
            base_url="http://127.0.0.1:8080/v1",
            api_key="sk-no-key-required",
        )

        self.model = model
        
        self.completion = None
        self.reply = None
        self.messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            }
        ]

    def get_text(self):
        self.text = input("Enter command: ")

    def get_image(self, image):
        self.proc_image = preprocess_image(image)


    def send_message_and_image(self):
        self.completion = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{self.text}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{self.proc_image}"
                            },
                        },
                    ],
                }
            ],
        )

    def send_message(self):
        
        self.messages.append({
            "role": "user",
            "content": self.text
        })
        
        self.completion = self.openai_client.chat.completions.create(
            model=self.model,
            messages=self.messages
        )
        
        self.reply = self.completion.choices[0].message.content
        
        self.messages.append({
            "role": "assistant",
            "content": self.reply
        })

    def print_message(self):
        print(self.completion.choices[0].message.content)
