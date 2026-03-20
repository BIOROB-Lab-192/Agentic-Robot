import base64

import cv2
from toolregistry import ToolRegistry

registry = ToolRegistry()


def make_registry(webcam):
    registry = ToolRegistry()

    @registry.register
    def get_image() -> str:
        """Captures the latest frame from the webcam and returns it as a base64-encoded JPEG string."""
        frame = webcam.get_frame()
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return base64.b64encode(buffer.tobytes()).decode("utf-8")

    return registry