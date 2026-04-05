import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs


class RealSense:
    def __init__(
        self, resolution=(1280, 720), fps=30, enable_depth=True, enable_rgb=True
    ):
        if not enable_depth and not enable_rgb:
            raise ValueError("At least one stream must be enabled.")

        self.resolution = resolution
        self.fps = fps
        self.enable_depth = enable_depth
        self.enable_rgb = enable_rgb

        self._lock = threading.Lock()
        self._rgb_frame = None
        self._depth_frame = None
        self.running = False
        self._depth_rs_frame = None

        self.pipeline = rs.pipeline()
        config = rs.config()

        if enable_rgb:
            config.enable_stream(
                rs.stream.color, resolution[0], resolution[1], rs.format.bgr8, fps
            )
        if enable_depth:
            config.enable_stream(
                rs.stream.depth, resolution[0], resolution[1], rs.format.z16, fps
            )

        self.profile = self.pipeline.start(config)

        # Depth scale: multiply raw depth values by this to get meters
        self.depth_scale = None
        if enable_depth:
            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()

        # Only align if both streams are enabled
        self.align = (
            rs.align(rs.stream.color) if (enable_depth and enable_rgb) else None
        )

        self.running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        self.display_thread = None

    def get_xyz_image(self):
        with self._lock:
            if self._depth_rs_frame is None:
                return None
            depth_frame = self._depth_rs_frame

        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)

        # Returns a flat list of vertices, one per pixel
        verts = np.asanyarray(points.get_vertices())
        h, w = depth_frame.get_height(), depth_frame.get_width()

        # Reshape to image layout: [row, col, xyz]
        xyz = verts.view(np.float32).reshape(h, w, 3)
        return xyz.copy()

    def _capture_loop(self):
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames()

                if self.align is not None:
                    frames = self.align.process(frames)

                with self._lock:
                    if self.enable_rgb:
                        color = frames.get_color_frame()
                        if color:
                            self._rgb_frame = np.asanyarray(color.get_data())

                    if self.enable_depth:
                        depth = frames.get_depth_frame()
                        if depth:
                            self._depth_rs_frame = depth
                            self._depth_frame = np.asanyarray(depth.get_data())

            except RuntimeError as e:
                print(e)
                # Can happen during shutdown
                break

    def get_rgb(self):
        with self._lock:
            return self._rgb_frame.copy() if self._rgb_frame is not None else None

    def get_depth(self):
        # Raw uint16 depth image in camera units
        with self._lock:
            return self._depth_frame.copy() if self._depth_frame is not None else None

    def get_depth_meters(self):
        with self._lock:
            if self._depth_frame is None or self.depth_scale is None:
                return None
            return self._depth_frame.astype(np.float32) * self.depth_scale

    def get_frames(self):
        with self._lock:
            rgb = self._rgb_frame.copy() if self._rgb_frame is not None else None
            depth = self._depth_frame.copy() if self._depth_frame is not None else None
        return rgb, depth

    def _display_loop(self):
        while self.running:
            rgb, depth = self.get_frames()

            if rgb is not None:
                cv2.imshow("RGB", rgb)

            if depth is not None:
                depth_display = cv2.convertScaleAbs(depth, alpha=0.03)
                depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
                cv2.imshow("Depth", depth_colormap)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.stop()
                break

    def start_display(self):
        if self.display_thread is None or not self.display_thread.is_alive():
            self.display_thread = threading.Thread(
                target=self._display_loop, daemon=True
            )
            self.display_thread.start()

    def stop(self):
        if not self.running:
            return

        self.running = False

        try:
            self.pipeline.stop()
        except Exception:
            pass

        if self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.0)

        if self.display_thread is not None and self.display_thread.is_alive():
            self.display_thread.join(timeout=1.0)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    cam = RealSense(resolution=(1280, 720), fps=30, enable_rgb=True, enable_depth=True)
    cam.start_display()

    for i in range(10):
        rgb, depth = cam.get_frames()
        if rgb is not None:
            print(f"[{i}] RGB shape: {rgb.shape}")
        if depth is not None:
            print(f"[{i}] Depth shape: {depth.shape}, dtype={depth.dtype}")
        time.sleep(1)

    cam.stop()
