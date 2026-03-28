import json
from io import BytesIO

from openai import OpenAI

import src.tools as tools


class LLMinterface:
    def __init__(self, model, tools):
        self.openai_client = OpenAI(
            base_url="http://127.0.0.1:8080/v1",
            api_key="sk-no-key-required",
        )
        self.tools = tools
        self.model = model

        self.completion = None
        self.reply = None

        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]

    def get_text(self):
        self.text = input("Enter command: ")

    def send_message(self):

        self.messages.append({"role": "user", "content": self.text})

        self.completion = self.openai_client.chat.completions.create(
            model=self.model, messages=self.messages
        )

        self.reply = self.completion.choices[0].message.content

        self.messages.append({"role": "assistant", "content": self.reply})

    def send_message_with_tools(self, webcam, depthcam):
        self.messages.append({"role": "user", "content": self.text})
        while True:
            self.completion = self.openai_client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.tools,
                tool_choice="auto",
            )
            msg = self.completion.choices[0].message
            if not msg.tool_calls:
                self.reply = msg.content
                self.messages.append({"role": "assistant", "content": self.reply})
                break
            self.messages.append(msg)
            for tool_call in msg.tool_calls:
                args = json.loads(tool_call.function.arguments or "{}")
                result, extra = tools.dispatch(tool_call.function.name, args, webcam, depthcam)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
                if extra:
                    self.messages.append(extra)
                    
    def prune_image_history(self):
        """Remove image injections from history, keeping only text."""
        self.messages = [
            m for m in self.messages
            if not (
                isinstance(m.get("content"), list) and
                any(c.get("type") == "image_url" for c in m["content"])
            )
        ]

    def print_message(self):
        print(self.completion.choices[0].message.content)


if __name__ == "__main__":
    from webcam_capture import Webcam

    cam = Webcam(0, (1920, 1080))
    llm = LLMinterface(
        model="models/Qwen3.5-4B-Q4_K_M.gguf", tools=tools.tool_json_list
    )

    llm.get_text() 
    llm.send_message_with_tools(cam)
    llm.prune_image_history()
    llm.print_message()

    cam.stop_webcam()
