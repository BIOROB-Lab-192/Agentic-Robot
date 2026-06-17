import base64
import json
import math
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image

from src.mapping import Calibration

robot = "dummy robot"
_calibration: Optional[Calibration] = None


def load_calibration(path: str | None = None) -> bool:
    """Load a calibration file produced by mapping.py.

    Once loaded, every call to get_xyz_coords will automatically transform
    camera-frame coordinates into robot-base coordinates before returning.

    If *path* is None or the file does not exist, prints a single warning
    and leaves the system in raw-camera-coords mode. Returns True if a
    calibration was loaded, False otherwise.
    """
    global _calibration
    if path is None:
        print(
            "[tools] No calibration path provided - using raw camera coordinates."
        )
        return False
    if not Path(path).is_file():
        print(
            f"[tools] No calibration found at {path} - using raw camera coordinates."
        )
        return False
    _calibration = Calibration.load(path)
    print(
        f"[tools] Loaded calibration from {path}  "
        f"(RMS: {_calibration.rms_error * 1000:.1f} mm)"
    )
    return True


def build_tools(webcam_res=(1920, 1080), depth_res=(1280, 720)):
    return [
        {
            "type": "function",
            "function": {
                "name": "get_webcam_frame",
                "description": f"Capture a single {webcam_res[0]}x{webcam_res[1]} frame from the webcam for visual analysis.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_depth_frames",
                "description": f"Captures a {depth_res[0]}x{depth_res[1]} RGB frame and aligned depth data. Use this to identify objects before calling get_xyz_coords.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_xyz_coords",
                "description": (
                    "Convert object locations into XYZ meters. "
                    "Pass the object's pixel coordinates as NORMALIZED FLOATS in the range [0, 1], "
                    f"where [0, 0] is the top-left of the {depth_res[0]}x{depth_res[1]} depth frame "
                    "and [1, 1] is the bottom-right. "
                    "Example: [0.5, 0.5] is the exact center of the frame. "
                    "The returned coordinates are in the depth-camera frame by default; "
                    "if a calibration has been loaded, they are transformed into the robot-base frame. "
                    "If a point returns 'invalid' (status: invalid), do not retry the same pixel. "
                    "Instead, pick a new location 5-10% of the frame away to bypass depth sensor noise."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "coords": {
                            "type": "array",
                            "description": "List of [x, y] NORMALIZED coordinates (floats in [0, 1]).",
                            "items": {
                                "type": "array",
                                "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                        }
                    },
                    "required": ["coords"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "robot_control",
                "description": (
                    "Sends waypoints in XYZ meters. Only use coordinates obtained via get_xyz_coords. "
                    "Never estimate coordinates or use raw pixel values here."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "waypoints": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "x": {"type": "number"},
                                    "y": {"type": "number"},
                                    "z": {"type": "number"},
                                },
                                "required": ["x", "y", "z"],
                            },
                        }
                    },
                    "required": ["waypoints"],
                },
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
        if rgb is not None and depth is not None and depth_rs is not None:
            break
        time.sleep(0.01)
    else:
        raise RuntimeError("Timed out waiting for camera frames")

    # Preserve the exact captured depth frame for later tool calls
    depth_rs.keep()
    depthcam.last_depth_rs = depth_rs

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

    depthcam.last_rgb = rgb.copy()

    return rgb_b64, depth_b64, xyz, rgb, depth, depth_rs


def get_xyz_coords(depthcam, coords, depth_rs):
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


def robot_control(waypoints, robot):

    for wp in waypoints:
        if not all(k in wp for k in ("x", "y", "z")):
            raise ValueError("Each waypoint must contain x, y, z.")
        if any(abs(wp[k]) > 5 for k in ("x", "y", "z")):
            raise ValueError("Waypoint values look invalid for meters.")

    print("Sending commands to robot")
    print(robot)
    print(f"Waypoints: \n {waypoints}")

    status = f"VIRTUAL MOVE: Robot would move through {len(waypoints)} points."
    for i, wp in enumerate(waypoints):
        status += (
            f"\n Point {i + 1}: X={wp['x']:.3f}m, Y={wp['y']:.3f}m, Z={wp['z']:.3f}m"
        )
    print(status)
    return status


def dispatch(
    tool_name: str, tool_args: dict, webcam, depthcam
) -> tuple[str, dict | None]:
    """Returns (tool_result_string, optional_extra_message)"""
    print(f"[DISPATCH] tool={tool_name} args={json.dumps(tool_args, indent=2)}")
    print(f"Selected Tool: {tool_name}")

    if tool_name == "get_depth_frames" and depthcam.last_depth_rs is not None:
        return (
            "Depth frame already captured. Use get_xyz_coords with the existing frame. "
            "Do NOT call get_depth_frames again unless explicitly told to refresh.",
            None,
        )

    elif tool_name == "get_xyz_coords":
        raw_coords = tool_args.get("coords", [])
        if depthcam.last_depth_rs is None:
            return "ERROR: No saved depth frame. Call get_depth_frames first.", None

        # Clamp into [0, 1] floats; guards against the LLM returning integers.
        norm_coords: list[list[float]] = []
        for c in raw_coords:
            try:
                nx = max(0.0, min(1.0, float(c[0])))
                ny = max(0.0, min(1.0, float(c[1])))
                norm_coords.append([nx, ny])
            except (TypeError, ValueError, IndexError):
                continue

        if hasattr(depthcam, "last_rgb"):
            debug_img = depthcam.last_rgb.copy()
            rgb_h, rgb_w = debug_img.shape[:2]
            for nx, ny in norm_coords:
                u = int(round(nx * (rgb_w - 1)))
                v = int(round(ny * (rgb_h - 1)))
                cv2.drawMarker(debug_img, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.imwrite("last_ai_aim.jpg", debug_img)
            print("[DEBUG] Saved AI target visualization to last_ai_aim.jpg")

        depth_h = depthcam.last_depth_rs.get_height()
        depth_w = depthcam.last_depth_rs.get_width()
        pixel_coords = [
            [int(round(nx * (depth_w - 1))), int(round(ny * (depth_h - 1)))]
            for nx, ny in norm_coords
        ]

        xyz = get_xyz_coords(depthcam, pixel_coords, depthcam.last_depth_rs)

        # Transform camera → robot if calibration is loaded
        if _calibration is not None:
            xyz = _calibration.camera_to_robot(xyz)

        points = xyz.tolist()

        # Tell the agent explicitly which coords failed — don't silently return nan
        results = []
        for (nx, ny), pt in zip(norm_coords, points):
            if any(v is not None and math.isnan(v) for v in pt):
                results.append({"norm": [nx, ny], "status": "invalid", "xyz": None})
            else:
                results.append({"norm": [nx, ny], "status": "ok", "xyz": pt})

        return json.dumps({"units": "meters", "points": results}), None

    elif tool_name == "get_webcam_frame":
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
        rgb_b64, depth_b64, xyz, rgb, depth, depth_rs = get_depth_frames(depthcam)

        extra = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{rgb_b64}"},
                },
            ],
        }

        return "Depth frames captured successfully.", extra

    elif tool_name == "robot_control":
        waypoints = tool_args.get("waypoints", [])
        success = robot_control(waypoints, robot)
        return (
            "Robot commands sent successfully." if success else "Robot control failed."
        ), None

    raise ValueError(f"Unknown tool: {tool_name}")
