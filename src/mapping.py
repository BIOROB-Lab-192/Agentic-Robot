"""
Self-contained calibration module.  No dependencies on other project
modules — only needs ``cv2``, ``numpy``, and optionally ``pyrealsense2``
for the live camera capture.

Parts:
  1. ArUco marker detection  (RGB → pixel → camera-frame XYZ)
  2. Kabsch SVD solver       (find R, T from corresponding 3D pairs)
  3. Calibration class        (store, apply, save/load the transform)
  4. Interactive routine      (robot moves → auto-capture → solve)
"""

from __future__ import annotations

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None  # Live capture functions will raise at call time.

# ---------------------------------------------------------------------------
# ArUco marker detection
# ---------------------------------------------------------------------------


def default_aruco_dict() -> cv2.aruco.Dictionary:
    """Return a pre-defined ArUco dictionary suitable for calibration."""
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)


def default_aruco_params() -> cv2.aruco.DetectorParameters:
    """Return default detector parameters."""
    return cv2.aruco.DetectorParameters()


def detect_marker_corners(
    rgb: np.ndarray,
    dictionary: cv2.aruco.Dictionary | None = None,
    params: cv2.aruco.DetectorParameters | None = None,
    marker_id: int = 0,
) -> np.ndarray | None:
    """Search *rgb* for ArUco *marker_id* and return its 4 corner pixels.

    Parameters
    ----------
    rgb : (H, W, 3) BGR uint8 image.
    dictionary : ArUco dictionary to use (defaults to DICT_4X4_1000).
    params : detector parameters (defaults to sensible defaults).
    marker_id : the specific marker ID we're looking for.

    Returns
    -------
    (4, 2) float32 array of pixel coordinates [u, v], ordered
    top-left, top-right, bottom-right, bottom-left.
    Returns *None* if the marker ID is not found.
    """
    if dictionary is None:
        dictionary = default_aruco_dict()
    if params is None:
        params = default_aruco_params()

    detector = cv2.aruco.ArucoDetector(dictionary, params)
    corners, ids, _ = detector.detectMarkers(rgb)
    if ids is None or marker_id not in ids.flatten():
        return None

    idx = int(np.where(ids.flatten() == marker_id)[0][0])
    return corners[idx][0].astype(np.float32)  # (4, 2)


def marker_center_pixel(corners: np.ndarray) -> tuple[int, int]:
    """Return the centre (u, v) of the 4 marker corners."""
    u = int(round(float(np.mean(corners[:, 0]))))
    v = int(round(float(np.mean(corners[:, 1]))))
    return u, v


def camera_xyz_at_pixel(
    u: int,
    v: int,
    depth_frame: rs.depth_frame,
    intrinsics: rs.intrinsics,
    patch_radius: int = 3,
) -> np.ndarray | None:
    """Deproject pixel (u, v) to camera-frame (X, Y, Z) in metres.

    If the exact centre pixel has invalid depth (0 or negative), the function
    samples a small (2*radius+1)-sized patch around it and takes the median
    valid depth.  This handles the common case where the marker centre lands
    on a depth hole or edge.

    Returns a (3,) float32 array, or *None* if no valid depth is found.
    """
    h, w = depth_frame.get_height(), depth_frame.get_width()

    def _depth_at(px, py) -> float:
        if 0 <= px < w and 0 <= py < h:
            return depth_frame.get_distance(px, py)
        return 0.0

    z = _depth_at(u, v)
    if z > 0:
        xyz = rs.rs2_deproject_pixel_to_point(intrinsics, [float(u), float(v)], z)
        return np.asarray(xyz, dtype=np.float32)

    # Fallback: sample a patch and take the median valid depth
    depths = []
    for du in range(-patch_radius, patch_radius + 1):
        for dv in range(-patch_radius, patch_radius + 1):
            d = _depth_at(u + du, v + dv)
            if d > 0:
                depths.append(d)
    if not depths:
        return None

    z_med = float(np.median(depths))
    xyz = rs.rs2_deproject_pixel_to_point(intrinsics, [float(u), float(v)], z_med)
    return np.asarray(xyz, dtype=np.float32)


def detect_marker_camera_xyz(
    rgb: np.ndarray,
    depth_frame: rs.depth_frame,
    intrinsics: rs.intrinsics,
    marker_id: int = 0,
) -> tuple[tuple[int, int], np.ndarray] | None:
    """High-level helper: detect *marker_id* in *rgb*, return its centre
    pixel and camera-frame (X, Y, Z).

    Returns ((u, v), camera_xyz) on success, or *None* if the marker
    wasn't found or the depth was invalid.
    """
    corners = detect_marker_corners(rgb, marker_id=marker_id)
    if corners is None:
        return None

    u, v = marker_center_pixel(corners)
    cam_xyz = camera_xyz_at_pixel(u, v, depth_frame, intrinsics)
    if cam_xyz is None:
        return None

    return (u, v), cam_xyz


# ---------------------------------------------------------------------------
# Kabsch solver (rigid alignment via SVD)
# ---------------------------------------------------------------------------


def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Kabsch (SVD) absolute-orientation solver.

    Finds the rigid rotation R and translation T that best align the
    (N, 3) source cloud *P* to the target cloud *Q* in a least-squares
    sense:

        Q ~ R @ P + T

    Both arrays must have the same shape and at least 3 rows.

    Returns
    -------
    R : (3, 3) orthonormal rotation matrix.
    T : (3,)   translation vector.
    rms_error_m : root-mean-square residual after alignment (metres).
    """
    assert P.shape == Q.shape and P.shape[0] >= 3

    # Centre the point clouds
    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    P_centred = P - cP
    Q_centred = Q - cQ

    # Cross-covariance matrix
    H = P_centred.T @ Q_centred

    # SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure a right-handed coordinate system (no reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    T = cQ - R @ cP

    # RMS error
    residual = Q - (P @ R.T + T)
    rms = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))

    return R, T, rms


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _pairwise_distance_check(
    camera_pts: np.ndarray, robot_pts: np.ndarray, tolerance_mm: float = 30.0
) -> list[tuple[int, int, float]]:
    """Compare the distance between every pair of points in both frames.

    Rigid transforms preserve distances, so for correctly matched points the
    camera-frame and robot-frame pairwise distances must agree within noise.

    Returns a list of (i, j, diff_mm) for pairs that exceed *tolerance_mm*.
    """
    n = len(camera_pts)
    bad_pairs: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            d_cam = np.linalg.norm(camera_pts[i] - camera_pts[j]) * 1000
            d_rob = np.linalg.norm(robot_pts[i] - robot_pts[j]) * 1000
            diff = abs(d_cam - d_rob)
            if diff > tolerance_mm:
                bad_pairs.append((i, j, diff))
    return bad_pairs


def _leave_one_out_errors(
    camera_pts: np.ndarray, robot_pts: np.ndarray
) -> list[tuple[int, float]]:
    """For each pose, solve without it and measure how far its prediction is.

    Returns a sorted list of (pose_index, error_mm), worst-first.
    A single large error points to a likely bad measurement.
    """
    n = len(camera_pts)
    errors: list[tuple[int, float]] = []
    for k in range(n):
        mask = [i for i in range(n) if i != k]
        R_k, T_k, _ = kabsch(camera_pts[mask], robot_pts[mask])
        pred = R_k @ camera_pts[k] + T_k
        err = np.linalg.norm(pred - robot_pts[k]) * 1000
        errors.append((k, err))
    errors.sort(key=lambda x: -x[1])  # worst first
    return errors


def validate_correspondences(
    camera_pts: np.ndarray,
    robot_pts: np.ndarray,
    distance_tolerance_mm: float = 30.0,
    outlier_threshold_mm: float = 50.0,
) -> list[int]:
    """Run pairwise and leave-one-out checks, report results.

    Parameters
    ----------
    camera_pts : (N, 3) array from the depth camera.
    robot_pts  : (N, 3) array from the robot controller.
    distance_tolerance_mm : max acceptable pairwise distance mismatch.
    outlier_threshold_mm : LOO error above this flags a pose as suspect.

    Returns
    -------
    Sorted list of suspect pose indices (worst first).  Empty if all clean.
    """
    print("\n  ── Running validation ──")

    # 1. Pairwise distance check
    bad = _pairwise_distance_check(camera_pts, robot_pts, distance_tolerance_mm)
    if bad:
        print(f"\n  ⚠  {len(bad)} pair(s) have mismatched distances:")
        suspect_count = {i: 0 for i in range(len(camera_pts))}
        for i, j, diff in bad:
            suspect_count[i] += 1
            suspect_count[j] += 1
            print(f"     Pose {i + 1} ↔ Pose {j + 1}:  Δ = {diff:.0f} mm")
    else:
        print("\n  ✓ All pairwise distances match (within tolerance).")
        suspect_count = {i: 0 for i in range(len(camera_pts))}

    # 2. Leave-one-out
    loo = _leave_one_out_errors(camera_pts, robot_pts)
    print(f"\n  Leave-one-out errors (worst first):")
    for idx, err in loo:
        flag = " ⚠" if err > outlier_threshold_mm else ""
        print(f"     Pose {idx + 1}:  {err:.0f} mm{flag}")

    # 3. Identify suspects
    suspects = []
    for idx, err in loo:
        score = suspect_count.get(idx, 0) + (1 if err > outlier_threshold_mm else 0)
        if score >= 1:
            suspects.append((idx, score, err))
    suspects.sort(key=lambda x: -x[1])  # worst first

    if suspects:
        print(f"\n  ⚠  Suspect poses (likely bad readings):")
        for idx, score, err in suspects:
            print(
                f"     Pose {idx + 1}:  mismatch score={score}, LOO error={err:.0f} mm"
            )
    else:
        print("\n  ✓ All poses look consistent.")

    return [s[0] for s in suspects]


def find_best_subset(
    camera_pts: np.ndarray,
    robot_pts: np.ndarray,
    min_inliers: int = 3,
    inlier_threshold_mm: float = 40.0,
) -> tuple[list[int], list[int]]:
    """Find the best consistent subset of poses via RANSAC.

    Enumerates every 3-pose combination, solves R,T, then checks how many
    of the remaining poses fit within *inlier_threshold_mm*.  Returns the
    largest inlier set.

    Parameters
    ----------
    camera_pts : (N, 3)
    robot_pts  : (N, 3)
    min_inliers : minimum poses to form a valid solution (default 3).
    inlier_threshold_mm : max error for a pose to be considered an inlier.

    Returns
    -------
    (inlier_indices, outlier_indices) — 0-based.  If no combination
    meets *min_inliers*, returns (all, []).
    """
    from itertools import combinations

    n = len(camera_pts)
    best_inliers: list[int] = list(range(n))
    best_outliers: list[int] = []
    best_score = (0, 0.0)  # (inlier_count, negative_max_err)

    for combo in combinations(range(n), 3):
        c_a = camera_pts[list(combo)]
        r_a = robot_pts[list(combo)]

        try:
            R, T, _ = kabsch(c_a, r_a)
        except (np.linalg.LinAlgError, AssertionError):
            continue

        errors = []
        for k in range(n):
            pred = R @ camera_pts[k] + T
            err = np.linalg.norm(pred - robot_pts[k]) * 1000
            errors.append(err)

        inliers = [k for k, e in enumerate(errors) if e < inlier_threshold_mm]
        outliers = [k for k, e in enumerate(errors) if e >= inlier_threshold_mm]
        max_err = max(errors) if errors else 0.0
        score = (len(inliers), -max_err)

        if score > best_score:
            best_score = score
            best_inliers = inliers
            best_outliers = outliers

    if len(best_inliers) < min_inliers:
        return list(range(n)), []

    return sorted(best_inliers), sorted(best_outliers)


# ---------------------------------------------------------------------------
# Calibration class
# ---------------------------------------------------------------------------


class Calibration:
    """Stores the rigid transform from the depth-camera frame to the robot
    base frame.

    .. code-block:: text

        P_robot = R @ P_camera + T

    Usage
    -----
    Collect corresponding 3D point pairs, then solve::

        cal = Calibration()
        cal.calibrate(camera_pts, robot_pts)  # each shape (N, 3)
        cal.save("calibration.json")

    At runtime::

        cal = Calibration.load("calibration.json")
        robot_xyz = cal.camera_to_robot(camera_xyz)
    """

    def __init__(self) -> None:
        self.R: np.ndarray = np.eye(3, dtype=np.float64)
        self.T: np.ndarray = np.zeros(3, dtype=np.float64)
        self.rms_error: float | None = None
        self.calibrated: bool = False

    # -- Solving -------------------------------------------------------------

    def calibrate(self, camera_pts: np.ndarray, robot_pts: np.ndarray) -> None:
        """Solve R, T from N >= 3 corresponding (camera, robot) point pairs.

        Parameters
        ----------
        camera_pts : (N, 3) array  — points in the depth-camera frame (metres).
        robot_pts  : (N, 3) array  — same physical points in the robot base
                                     frame (metres).
        """
        assert camera_pts.ndim == 2 and camera_pts.shape[1] == 3
        assert robot_pts.ndim == 2 and robot_pts.shape[1] == 3
        assert camera_pts.shape[0] == robot_pts.shape[0]
        assert camera_pts.shape[0] >= 3, "Need at least 3 non-collinear points."

        self.R, self.T, self.rms_error = kabsch(camera_pts, robot_pts)
        self.calibrated = True

        rms_mm = self.rms_error * 1000
        n = camera_pts.shape[0]
        print(f"[Calibration] RMS residual: {rms_mm:.1f} mm (from {n} point pairs)")

    # -- Transform -----------------------------------------------------------

    def camera_to_robot(self, pts: np.ndarray) -> np.ndarray:
        """Transform point(s) from camera frame to robot frame.

        Accepts shape (3,) for a single point or (N, 3) for many.
        Returns the same shape as the input.
        """
        assert self.calibrated, "Calibration has not been performed yet."
        single = pts.ndim == 1
        pts = pts.reshape(-1, 3) if single else pts
        result = pts @ self.R.T + self.T
        return result.ravel() if single else result

    def robot_to_camera(self, pts: np.ndarray) -> np.ndarray:
        """Transform point(s) from robot frame back to camera frame.

        Accepts shape (3,) for a single point or (N, 3) for many.
        Returns the same shape as the input.
        """
        assert self.calibrated, "Calibration has not been performed yet."
        single = pts.ndim == 1
        pts = pts.reshape(-1, 3) if single else pts
        result = (pts - self.T) @ self.R
        return result.ravel() if single else result

    # -- Persistence ---------------------------------------------------------

    def save(self, path: str) -> None:
        """Write R, T to a JSON file."""
        import json
        from pathlib import Path

        data = {
            "R": self.R.tolist(),
            "T": self.T.tolist(),
            "rms_error_m": self.rms_error,
            "calibrated": self.calibrated,
        }
        Path(path).write_text(json.dumps(data, indent=2))
        print(f"[Calibration] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "Calibration":
        """Load R, T from a JSON file written by *save*."""
        import json
        from pathlib import Path

        data = json.loads(Path(path).read_text())
        obj = cls()
        obj.R = np.asarray(data["R"], dtype=np.float64)
        obj.T = np.asarray(data["T"], dtype=np.float64)
        obj.rms_error = data.get("rms_error_m")
        obj.calibrated = data.get("calibrated", True)
        return obj


# ---------------------------------------------------------------------------
# Interactive calibration routine (manual robot positioning)
# ---------------------------------------------------------------------------


def run_interactive_calibration(
    robot_poses: list[tuple[float, float, float]],
    marker_id: int = 0,
    marker_size_m: float = 0.04,
    depth_resolution: tuple[int, int] = (1280, 720),
    fps: int = 30,
    output_dir: str = "calibration_out",
) -> Calibration:
    """Walk the operator through N manual robot positions.

    For each position the operator moves the robot so the ArUco marker on
    the end-effector is visible in the workspace.  Pressing Enter captures
    the camera-frame XYZ of the marker.  After all N poses are captured the
    transform is solved and saved.

    All outputs (calibration.json, per-pose debug images, and a log file)
    are written to *output_dir*, which is created if it doesn't exist.

    Parameters
    ----------
    robot_poses : list of (x, y, z) metres
        The known robot-base coordinates for each position.
    marker_id : int
        ArUco marker ID attached to the end-effector.
    marker_size_m : float
        Physical side length of the marker (used for visualisation).
    depth_resolution : (int, int)
        (width, height) for the RealSense streams.
    fps : int
        Frame rate for the RealSense streams.
    output_dir : str
        Folder where all calibration artifacts are written.

    Returns
    -------
    Calibration instance.
    """
    from pathlib import Path

    if rs is None:
        raise RuntimeError("pyrealsense2 is not installed — cannot capture frames.")

    # Prepare output directory
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_path = str(out / "calibration.json")
    log_path = out / "captures.csv"

    # Start the camera
    pipeline = rs.pipeline()
    config = rs.config()
    w, h = depth_resolution
    config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)

    profile = pipeline.start(config)

    # Align depth to colour so pixels correspond
    align = rs.align(rs.stream.color)

    # Get depth intrinsics (needed for deprojection)
    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    intrinsics = depth_profile.get_intrinsics()

    cal = Calibration()
    camera_pts: list[np.ndarray] = []
    robot_pts: list[np.ndarray] = []

    print("\n" + "=" * 60)
    print("  Camera-to-Robot Calibration")
    print("=" * 60)
    print(f"  Marker ID:      {marker_id}")
    print(f"  Robot poses:    {len(robot_poses)}")
    print(f"  Resolution:     {w}x{h}")
    print()
    print("  Instructions:")
    print("    For each pose below, move the robot so the centre of its")
    print(f"    end-effector (with marker {marker_id}) is at the specified")
    print("    robot-base coordinates.  Then press Enter to capture.")
    print()
    for i, (rx, ry, rz) in enumerate(robot_poses):
        print(f"    Pose {i + 1}:  X={rx:.3f}  Y={ry:.3f}  Z={rz:.3f}")
    print()
    input("  Press Enter when ready to begin...")

    try:
        for i, (rx, ry, rz) in enumerate(robot_poses):
            print(f"\n{'─' * 60}")
            print(f"  Pose {i + 1} / {len(robot_poses)}")
            print(f"  Robot coordinates: ({rx:.3f}, {ry:.3f}, {rz:.3f}) m")
            print(f"  {'─' * 60}")

            # Wait for the operator to position the robot and press Enter
            input("  Move the robot to this position → Press Enter to capture: ")

            # Grab a frame
            rgb = None
            depth_rs = None
            for _ in range(60):
                frames = pipeline.wait_for_frames()
                frames = align.process(frames)
                color_frame = frames.get_color_frame()
                depth_rs = frames.get_depth_frame()
                if color_frame and depth_rs:
                    rgb = np.asanyarray(color_frame.get_data()).copy()
                    break

            if rgb is None or depth_rs is None:
                print("  ✗ Failed to get camera frames. Skipping.")
                continue

            # Detect marker
            corners = detect_marker_corners(rgb, marker_id=marker_id)
            if corners is None:
                print(f"  ✗ Marker {marker_id} not detected. Skipping.")
                continue

            u, v = marker_center_pixel(corners)
            cam_xyz = camera_xyz_at_pixel(u, v, depth_rs, intrinsics)
            if cam_xyz is None:
                print(f"  ✗ Invalid depth at centre pixel ({u}, {v}). Skipping.")
                continue

            camera_pts.append(cam_xyz)
            robot_pts.append(np.array([rx, ry, rz], dtype=np.float64))

            # Log to console
            print(f"  ✓ Captured:  pixel=({u}, {v})")
            print(f"  Robot : ({rx:8.4f}, {ry:8.4f}, {rz:8.4f}) m")
            print(
                f"  Camera: ({cam_xyz[0]:8.4f}, {cam_xyz[1]:8.4f}, {cam_xyz[2]:8.4f}) m"
            )
            d = np.linalg.norm(cam_xyz - np.array([rx, ry, rz])) * 1000
            print(f"  Raw Δ : {d:6.0f} mm  (robot - camera, before calibration)")

            # Append to CSV log
            with open(log_path, "a") as f:
                if i == 0:
                    f.write(
                        "pose,robot_x,robot_y,robot_z,camera_x,camera_y,camera_z,pixel_u,pixel_v\n"
                    )
                f.write(
                    f"{i + 1},"
                    f"{rx:.6f},{ry:.6f},{rz:.6f},"
                    f"{cam_xyz[0]:.6f},{cam_xyz[1]:.6f},{cam_xyz[2]:.6f},"
                    f"{u},{v}\n"
                )

            # Save a debug image with the marker centre marked
            debug = rgb.copy()
            cv2.circle(debug, (u, v), 5, (0, 0, 255), -1)
            cv2.circle(debug, (u, v), 25, (0, 255, 0), 2)
            cv2.imwrite(str(out / f"pose_{i + 1:02d}.jpg"), debug)

        if len(camera_pts) < 3:
            raise RuntimeError(
                f"Only {len(camera_pts)} valid captures — need at least 3."
            )

        # --- Validation ---
        ca = np.array(camera_pts)
        ro = np.array(robot_pts)

        # Auto-select the best consistent subset
        inliers, outliers = find_best_subset(ca, ro)

        print(f"\n{'=' * 60}")
        print(f"  Collected {len(camera_pts)} / {len(robot_poses)} poses")
        print(f"  Best subset: {len(inliers)} inlier(s), {len(outliers)} outlier(s)")

        if outliers:
            print(f"\n  Outlier poses (excluded):")
            for idx in outliers:
                rx, ry, rz = robot_poses[idx]
                R, T, _ = kabsch(ca[inliers], ro[inliers])
                pred = R @ ca[idx] + T
                err = np.linalg.norm(pred - ro[idx]) * 1000
                print(
                    f"     Pose {idx + 1}:  target=({rx:.3f}, {ry:.3f}, {rz:.3f})  "
                    f"error={err:.0f} mm"
                )

            # Offer to re-capture outliers or accept subset
            print(
                f"\n  You can re-capture {len(outliers)} outlier pose(s), "
                f"or press Enter to solve with the {len(inliers)} good poses."
            )
            choice = input(
                "  Enter a pose number to re-capture, or Enter to accept subset: "
            ).strip()
            if not choice:
                # Drop outliers and keep going
                camera_pts = [camera_pts[i] for i in inliers]
                robot_pts = [robot_pts[i] for i in inliers]
            else:
                try:
                    redo = int(choice) - 1
                except ValueError:
                    redo = -1
                if redo < 0 or redo >= len(robot_poses):
                    camera_pts = [camera_pts[i] for i in inliers]
                    robot_pts = [robot_pts[i] for i in inliers]
                else:
                    # Re-capture one pose
                    rx, ry, rz = robot_poses[redo]
                    print(
                        f"\n  Re-capturing Pose {redo + 1}:  ({rx:.3f}, {ry:.3f}, {rz:.3f})"
                    )
                    input(
                        "  Move the robot to this position → Press Enter to capture: "
                    )

                    rgb = None
                    depth_rs = None
                    for _ in range(60):
                        frames = pipeline.wait_for_frames()
                        frames = align.process(frames)
                        color_frame = frames.get_color_frame()
                        depth_rs = frames.get_depth_frame()
                        if color_frame and depth_rs:
                            rgb = np.asanyarray(color_frame.get_data()).copy()
                            break

                    if rgb is None or depth_rs is None:
                        print("  ✗ Failed to get frames. Using best subset.")
                        camera_pts = [camera_pts[i] for i in inliers]
                        robot_pts = [robot_pts[i] for i in inliers]
                    else:
                        corners = detect_marker_corners(rgb, marker_id=marker_id)
                        if corners is None:
                            print(
                                f"  ✗ Marker {marker_id} not detected. Using best subset."
                            )
                            camera_pts = [camera_pts[i] for i in inliers]
                            robot_pts = [robot_pts[i] for i in inliers]
                        else:
                            u, v = marker_center_pixel(corners)
                            cam_xyz = camera_xyz_at_pixel(u, v, depth_rs, intrinsics)
                            if cam_xyz is None:
                                print("  ✗ Invalid depth. Using best subset.")
                                camera_pts = [camera_pts[i] for i in inliers]
                                robot_pts = [robot_pts[i] for i in inliers]
                            else:
                                camera_pts[redo] = cam_xyz
                                robot_pts[redo] = np.array(
                                    [rx, ry, rz], dtype=np.float64
                                )
                                print(f"  ✓ Re-captured:  pixel=({u}, {v})")
                                print(
                                    f"     Camera: ({cam_xyz[0]:8.4f}, {cam_xyz[1]:8.4f}, {cam_xyz[2]:8.4f}) m"
                                )

                                # Update CSV
                                rows = []
                                with open(log_path) as f:
                                    rows = f.readlines()
                                rows[redo + 1] = (
                                    f"{redo + 1},"
                                    f"{rx:.6f},{ry:.6f},{rz:.6f},"
                                    f"{cam_xyz[0]:.6f},{cam_xyz[1]:.6f},{cam_xyz[2]:.6f},"
                                    f"{u},{v}\n"
                                )
                                with open(log_path, "w") as f:
                                    f.writelines(rows)

                                # Overwrite debug image
                                debug = rgb.copy()
                                cv2.circle(debug, (u, v), 5, (0, 0, 255), -1)
                                cv2.circle(debug, (u, v), 25, (0, 255, 0), 2)
                                cv2.imwrite(
                                    str(out / f"pose_{redo + 1:02d}.jpg"), debug
                                )

        print(f"\n{'=' * 60}")
        print(f"  Solving with {len(camera_pts)} poses...")

        cal.calibrate(np.array(camera_pts), np.array(robot_pts))
        cal.save(save_path)

        print(f"\n  Calibration complete. Saved to {save_path}")
        print(f"  RMS residual: {cal.rms_error * 1000:.1f} mm")
        print(f"{'=' * 60}")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return cal


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Default: run interactive calibration
    # Pass --test to run the solver self-test instead
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Verify the solver with a known ground-truth transform
        np.random.seed(42)
        R_true, _ = np.linalg.qr(np.random.randn(3, 3))  # orthonormal
        T_true = np.random.uniform(-0.5, 0.5, 3)

        pts_cam = np.random.uniform(-0.3, 0.3, (50, 3))
        pts_robot = pts_cam @ R_true.T + T_true

        cal = Calibration()
        cal.calibrate(pts_cam, pts_robot)

        recovered = cal.camera_to_robot(pts_cam[0])
        err_mm = np.linalg.norm(pts_robot[0] - recovered) * 1000
        print(f"Single-point error: {err_mm:.3f} mm")
        print(f"RMS residual:       {cal.rms_error * 1000:.3f} mm")

        assert cal.rms_error < 1e-6, (
            "Solver should recover exact transform from noise-free data"
        )
        print("\n✓ Self-test passed — Kabsch solver is correct.")
    else:
        # Example poses — operator edits these before running
        poses = [
            (290 / 1000, -10 / 1000, 600 / 1000),
            (-160 / 1000, 125 / 1000, 750 / 1000),
            (100 / 1000, 315 / 1000, 500 / 1000),
            (290 / 1000, 240 / 1000, 650 / 1000),
            (160 / 1000, 330 / 1000, 400 / 1000),
        ]
        run_interactive_calibration(poses, output_dir="calibration_out")
