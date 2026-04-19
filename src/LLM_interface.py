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

        self.messages = [
            {
                "role": "system",
                "content": "You control a robot arm. You have a camera and depth cam to see the workspace. For coordinate requests, do not keep searching for a better frame. Use the current saved frame, estimate the target pixel once, call get_xyz_coords once, and then answer. If the point is invalid or depth is missing, explain that the coordinate could not be read from the saved frame.",
            }
        ]

    def get_text(self):
        self.text = input("Enter command: ")

    def send_message(self):

        self.messages.append({"role": "user", "content": self.text})

        self.completion = self.openai_client.chat.completions.create(
            model=self.model, messages=self.messages
        )

        self.reply = self.completion.choices[0].message.content

        self.messages.append({"role": "assistant", "content": self.reply})

    def send_message_with_tools(self, webcam, depthcam, max_rounds=4):
        # Start a fresh request
        self.messages.append({"role": "user", "content": self.text})

        # Reset saved depth frame for each new user command
        if depthcam is not None:
            depthcam.last_depth_rs = None

        tool_counts = {}

        for _ in range(max_rounds):
            self.completion = self.openai_client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.tools,
                tool_choice="auto",
            )

            msg = self.completion.choices[0].message

            # No tool calls -> final assistant reply
            if not msg.tool_calls:
                self.reply = msg.content or "No answer produced."
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": self.reply,
                    }
                )
                return

            # Convert SDK message object to a plain dict before storing
            assistant_msg = {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    }
                    for tc in (msg.tool_calls or [])
                ],
            }
            self.messages.append(assistant_msg)

            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                tool_counts[name] = tool_counts.get(name, 0) + 1

                # Hard limits to stop loops
                if name == "get_depth_frames" and tool_counts[name] > 1:
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": (
                                "ERROR: get_depth_frames may be called only once per user request. "
                                "Use the already captured frame."
                            ),
                        }
                    )
                    continue

                if name == "get_xyz_coords" and tool_counts[name] > 1:
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": (
                                "ERROR: get_xyz_coords may be called only once per user request. "
                                "Use the returned coordinate if valid; otherwise explain that the "
                                "coordinate could not be read from the saved frame."
                            ),
                        }
                    )
                    continue

                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                result, extra = tools.dispatch(name, args, webcam, depthcam)

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

                if extra:
                    self.messages.append(extra)

        # Failsafe if the model keeps trying to call tools
        self.reply = (
            "Stopped after too many tool rounds. "
            "Use the current result, or say the target could not be localized from the current frame."
        )
        self.messages.append(
            {
                "role": "assistant",
                "content": self.reply,
            }
        )

    def prune_image_history(self):
        """Remove image injections from history, keeping only text."""
        self.messages = [
            m
            for m in self.messages
            if not (
                isinstance(m.get("content"), list)
                and any(c.get("type") == "image_url" for c in m["content"])
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
    llm.send_message_with_tools(cam, depthcam=None)
    llm.prune_image_history()
    llm.print_message()

    cam.stop_webcam()
