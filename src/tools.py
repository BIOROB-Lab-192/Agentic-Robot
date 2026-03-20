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
    print("capturing image)")
    frame = webcam.get_frame()
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame = frame.convert("RGB")
    # frame = frame.resize((640, 480))
    buffer = BytesIO()
    frame.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def dispatch(tool_name: str, tool_args: dict, webcam) -> tuple[str, dict | None]:
    """Returns (tool_result_string, optional_extra_message)"""
    if tool_name == "get_webcam_frame":
        image = get_webcam_frame(webcam)
        extra = {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}]
        }
        return "Webcam frame captured successfully.", extra
    raise ValueError(f"Unknown tool: {tool_name}")

