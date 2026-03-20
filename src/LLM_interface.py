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

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_webcam_frame",
                    "description": "Capture a single frame from the webcam for visual analysis.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        ]

        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]

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

        self.messages.append({"role": "user", "content": self.text})

        self.completion = self.openai_client.chat.completions.create(
            model=self.model, messages=self.messages
        )

        self.reply = self.completion.choices[0].message.content

        self.messages.append({"role": "assistant", "content": self.reply})

    def add_image(self):
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{self.proc_image}"
                        },
                    }
                ],
            }
        )

    def send_message_with_tools(self, webcam):
        self.messages.append({"role": "user", "content": self.text})

        while True:
            self.completion = self.openai_client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.tools,
                tool_choice="auto",
            )

            msg = self.completion.choices[0].message

            # No tool call — we have a final answer
            if not msg.tool_calls:
                self.reply = msg.content
                self.messages.append({"role": "assistant", "content": self.reply})
                break

            # Append the assistant's tool-call request to history
            self.messages.append(msg)

            # Handle each requested tool call
            for tool_call in msg.tool_calls:
                if tool_call.function.name == "get_webcam_frame":
                    # Capture and preprocess the frame
                    frame = webcam.get_frame()
                    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    self.get_image(pil_image)

                    # Tool result must be a string — use a placeholder
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": "Webcam frame captured successfully.",
                        }
                    )

                    # Inject the actual image as a follow-up user message
                    self.messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{self.proc_image}"
                                    },
                                }
                            ],
                        }
                    )

    def print_message(self):
        print(self.completion.choices[0].message.content)


if __name__ == "__main__":
    import cv2

    from src.webcam_capture import Webcam

    cam = Webcam(0, (1920, 1080))
    llm = LLMinterface(model="models/Qwen3.5-4B-Q4_K_M.gguf")

    llm.get_text()  # prompts: "Enter command: "
    llm.send_message_with_tools(cam)
    llm.print_message()

    cam.stop_webcam()
