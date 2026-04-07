import base64
import json
import time
from io import BytesIO

import cv2
import numpy as np
import pyrealsnse2 as rs
from PIL import Image

tool_json_list = [
    {
        "type": "function",
        "function": {
            "name": "get_webcam_frame",
            "description": "Capture a single frame from the webcam for visual analysis.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_depth_frames",
            "description": "Capture a pair of aligned RGB and depth frames from a depth camera for 3D visual analysis. Also captures the x, y, z coordinates of pixels in the image, in meters.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def get_webcam_frame(webcam):
    print("capturing image")
    frame = webcam.get_frame()
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame = frame.convert("RGB")
    # frame = frame.resize((640, 480))
    buffer = BytesIO()
    frame.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_depth_frames(depthcam):
    print("capturing depth images")
    for _ in range(100):
        rgb, depth, depth_rs = depthcam.get_frames()
        if rgb is not None and depth is not None:
            break
        time.sleep(0.01)
    else:
        raise RuntimeError("Timed out waiting for camera frames")

    rgb_img = Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    rgb_buffer = BytesIO()
    rgb_img.save(rgb_buffer, format="JPEG", quality=85)
    rgb_b64 = base64.b64encode(rgb_buffer.getvalue()).decode("utf-8")

    depth_display = cv2.convertScaleAbs(depth, alpha=0.03)
    depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

    depth_img = Image.fromarray(cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB))
    depth_buffer = BytesIO()
    depth_img.save(depth_buffer, format="JPEG", quality=85)
    depth_b64 = base64.b64encode(depth_buffer.getvalue()).decode("utf-8")

    xyz = depthcam.get_xyz_image()

    return rgb_b64, depth_b64, xyz, rgb, depth, depth_rs


def get_xyz_cords(depthcam, coords, depth_rs):
    if depth_rs is None:
        return None

    coords = np.asarray(coords, dtype=np.int32).reshape(-1, 2)

    intrinsics = depth_rs.profile.as_video_stream_profile().intrinsics
    h = depth_rs.get_height()
    w = depth_rs.get_width()

    out = []
    for x, y in coords:
        if not (0 <= x < w and 0 <= y < h):
            out.append([np.nan, np.nan, np.nan])
            continue

        z = depth_rs.get_distance(x, y)
        if z <= 0:
            out.append([np.nan, np.nan, np.nan])
            continue

        xyz = rs.rs2_deproject_pixel_to_point(intrinsics, [float(x), float(y)], z)
        out.append(xyz)

    return np.asarray(out, dtype=np.float32)


def dispatch(
    tool_name: str, tool_args: dict, webcam, depthcam
) -> tuple[str, dict | None]:
    """Returns (tool_result_string, optional_extra_message)"""
    print(f"Selected Tool: {tool_name}")
    if tool_name == "get_webcam_frame":
        image = get_webcam_frame(webcam)
        extra = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            ],
        }
        return "Webcam frame captured successfully.", extra

    elif tool_name == "get_depth_frames":
        rgb, depth, xyz = get_depth_frames(depthcam)

        xyz_payload = None
        if xyz is not None:
            xyz_small = xyz[::16, ::16, :]  # downsample
            xyz_payload = {
                "units": "meters",
                "original_shape": list(xyz.shape),
                "sampled_shape": list(xyz_small.shape),
                "sampling_stride": 16,
                "data": xyz_small.tolist(),
            }

        extra = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{rgb}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{depth}"},
                },
            ],
        }

        return "Depth frames captured successfully.", extra

    raise ValueError(f"Unknown tool: {tool_name}")
