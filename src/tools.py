import base64
import json
from io import BytesIO

import cv2
from PIL import Image

tool_json_list = [
    {
        "type": "function",
        "function": {
            "name": "get_webcam_frame",
            "description": "Capture a single frame from the webcam for visual analysis.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
]


def get_webcam_frame(webcam):
    frame = webcam.get_frame()
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame = frame.convert("RGB")
    frame = frame.resize((640, 480))  # drastically reduces token count
    buffer = BytesIO()
    frame.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def dispatch(tool_name: str, tool_args: dict, webcam) -> str:
    """Returns the tool result string for the LLM history."""
    if tool_name == "get_webcam_frame":
        return get_webcam_frame(webcam)
    raise ValueError(f"Unknown tool: {tool_name}")


def dispatch_side_effect(tool_name: str, tool_args: dict, webcam) -> dict | None:
    """Returns any extra message to inject (e.g. an image), or None."""
    if tool_name == "get_webcam_frame":
        image = get_webcam_frame(webcam)
        return {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            ],
        }
    return None
