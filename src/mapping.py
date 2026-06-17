"""
Self-contained calibration module.  No dependencies on other project
modules — only needs ``cv2``, ``numpy``, and ``pyrealsense2`` for the
live camera capture.

Parts:
  1. ArUco marker detection  (RGB → 4 corner pixels)
  2. 6-DoF pose recovery     (corners + marker size → rvec, tvec)
  3. Kabsch SVD solver       (find R, T from corresponding 3D pairs)
  4. Calibration class        (store, apply, save/load the transform)
  5. Interactive routine      (robot moves → auto-capture → solve)
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
    """Return tuned detector parameters for sub-pixel corner accuracy."""
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize = 5
    p.cornerRefinementMaxIterations = 50
    p.cornerRefinementMinAccuracy = 0.01
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 23
    p.adaptiveThreshConstant = 7
    p.minMarkerPerimeterRate = 0.02
    p.maxMarkerPerimeterRate = 0.30
    p.errorCorrectionRate = 0.6
    return p


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
    c = corners[idx][0]
    sides = [
        np.linalg.norm(c[0] - c[1]),
        np.linalg.norm(c[1] - c[2]),
        np.linalg.norm(c[2] - c[3]),
        np.linalg.norm(c[3] - c[0]),
    ]
    if max(sides) / min(sides) > 1.6:
        return None  # one corner likely on gripper edge, not the marker
    return c.astype(np.float32)  # (4, 2)


def marker_center_pixel(corners: np.ndarray) -> tuple[int, int]:
    """Return the centre (u, v) of the 4 marker corners."""
    u = int(round(float(np.mean(corners[:, 0]))))
    v = int(round(float(np.mean(corners[:, 1]))))
    return u, v


# ---------------------------------------------------------------------------
# 6-DoF marker pose recovery (calibration-time helpers)
# ---------------------------------------------------------------------------


def intrinsics_to_KD(intr: "rs.intrinsics") -> tuple[np.ndarray, np.ndarray]:
    """Convert a pyrealsense2 intrinsics struct into OpenCV K, D arrays."""
    K = np.array(
        [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]],
        dtype=np.float64,
    )
    D = np.asarray(intr.coeffs, dtype=np.float64)
    return K, D


def marker_pose_from_corners(
    corners: np.ndarray,
    marker_size_m: float,
    K: np.ndarray,
    D: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve 6-DoF pose of a square marker from its 4 image corners.

    Uses ``cv2.aruco.estimatePoseSingleMarkers`` (internally ``solvePnP``) to
    recover the marker's full pose in the camera frame from a known-size 2D
    target.  Returns the rvec, tvec, and the worst-case per-corner reprojection
    error in pixels.

    Parameters
    ----------
    corners : (4, 2) array of pixel coordinates in OpenCV ArUco order
        (top-left, top-right, bottom-right, bottom-left).
    marker_size_m : physical side length of the marker, in metres.
    K : (3, 3) camera intrinsic matrix.
    D : (5,) distortion coefficients, or *None* to assume zero distortion.

    Returns
    -------
    rvec : (3,) Rodrigues rotation vector.
    tvec : (3,) translation vector (marker centre in camera frame, metres).
    reproj_err_px : max per-corner reprojection error in pixels.
    """
    if D is None:
        D = np.zeros(5, dtype=np.float64)
    s = float(marker_size_m)
    obj = np.array(
        [
            [-s / 2, s / 2, 0],
            [s / 2, s / 2, 0],
            [s / 2, -s / 2, 0],
            [-s / 2, -s / 2, 0],
        ],
        dtype=np.float32,
    )
    inp = corners.reshape(4, 1, 2).astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(
        obj, inp, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not ok:
        raise RuntimeError("solvePnP failed for marker corners")
    rvec = rvec.reshape(3)
    tvec = tvec.reshape(3)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    err = float(np.linalg.norm(proj.reshape(4, 2) - corners, axis=1).max())
    return rvec, tvec, err


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
    rvecs: np.ndarray | None = None,
    distance_tolerance_mm: float = 30.0,
    outlier_threshold_mm: float = 50.0,
) -> list[int]:
    """Run pairwise and leave-one-out checks, report results.

    Parameters
    ----------
    camera_pts : (N, 3) array from the depth camera.
    robot_pts  : (N, 3) array from the robot controller.
    rvecs : (N, 3) array of per-pose marker rvecs in the camera frame, or
        *None* to skip the cross-pose orientation consistency check.
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
    loo_err_by_idx = {idx: err for idx, err in loo}
    suspects = []
    for idx, err in loo:
        score = suspect_count.get(idx, 0) + (1 if err > outlier_threshold_mm else 0)
        if score >= 1:
            suspects.append((idx, score, err))

    # 4. Cross-pose rvec consistency (no-rotation gripper: rvec should be
    # near-constant across all poses; threshold = median_dev + 5*1.4826*MAD,
    # which adapts to the rig's natural arm-compliance drift instead of
    # using a fixed angle)
    if rvecs is not None and len(rvecs) == len(camera_pts) and len(rvecs) >= 3:
        median_rvec = np.median(rvecs, axis=0)
        dev_deg = np.linalg.norm(rvecs - median_rvec, axis=1) * (180.0 / np.pi)
        median_dev = float(np.median(dev_deg))
        mad = float(np.median(np.abs(dev_deg - median_dev)))
        threshold = median_dev + 5.0 * 1.4826 * mad
        print(f"\n  Per-pose rvec deviation from median (deg):")
        for i, d in enumerate(dev_deg):
            flag = " ⚠" if d > threshold else ""
            print(f"     Pose {i + 1}:  {d:.2f}°{flag}")
        for i, d in enumerate(dev_deg):
            if d > threshold:
                loo_err_mm = loo_err_by_idx.get(i, 0.0)
                suspects.append((i, 1, loo_err_mm))

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
    inlier_threshold_mm: float = 25.0,
) -> tuple[list[int], list[int]]:
    """Find the most self-consistent subset of poses.

    Enumerates every 3-pose combination, solves R,T, then scores each
    subset by how many of the remaining poses fit within
    *inlier_threshold_mm* (tie-broken by lowest RMS).  Returns the
    largest / tightest inlier set.

    Parameters
    ----------
    camera_pts : (N, 3)
    robot_pts  : (N, 3)
    inlier_threshold_mm : max error for a pose to be considered an inlier.

    Returns
    -------
    (inlier_indices, outlier_indices) — 0-based.  If every subset scores
    0 inliers, returns the subset with the lowest RMS error (and marks
    the worst-fitting pose as the sole outlier, if N > 3).
    """
    from itertools import combinations

    n = len(camera_pts)
    best_inliers: list[int] = list(range(n))
    best_outliers: list[int] = []
    best_score = (-1, 0.0)  # (inlier_count, -RMS_mm) — higher is better

    for combo in combinations(range(n), 3):
        idx = list(combo)
        c_a = camera_pts[idx]
        r_a = robot_pts[idx]

        try:
            R, T, rms = kabsch(c_a, r_a)
        except (np.linalg.LinAlgError, AssertionError):
            continue

        errors = []
        for k in range(n):
            pred = R @ camera_pts[k] + T
            err = np.linalg.norm(pred - robot_pts[k]) * 1000
            errors.append(err)

        inliers = [k for k, e in enumerate(errors) if e < inlier_threshold_mm]
        outliers = [k for k, e in enumerate(errors) if e >= inlier_threshold_mm]
        score = (len(inliers), -rms * 1000)  # higher inlier count + lower RMS

        if score > best_score:
            best_score = score
            best_inliers = inliers
            best_outliers = outliers

    # If no subset has ≥3 inliers, fall back: return the tightest 3-combo
    # and mark the single worst overall pose as outlier.
    if len(best_inliers) < 3:
        # Try again scoring purely by lowest RMS
        best_rms = float("inf")
        best_combo = None
        best_all_errors = None
        for combo in combinations(range(n), 3):
            idx = list(combo)
            try:
                R, T, rms = kabsch(camera_pts[idx], robot_pts[idx])
            except (np.linalg.LinAlgError, AssertionError):
                continue
            errors = [
                np.linalg.norm(R @ camera_pts[k] + T - robot_pts[k]) * 1000
                for k in range(n)
            ]
            total_rms = float(np.sqrt(np.mean(np.array(errors) ** 2)))
            if total_rms < best_rms:
                best_rms = total_rms
                best_combo = idx
                best_all_errors = errors

        if best_combo and best_all_errors:
            best_inliers = list(range(n))
            # Kick out the single worst pose if N > 3
            if n > 3:
                worst = int(np.argmax(best_all_errors))
                best_inliers = [k for k in range(n) if k != worst]
                best_outliers = [worst]
            else:
                best_outliers = []
        else:
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
    marker_size_m: float = 0.03,
    color_resolution: tuple[int, int] = (1280, 720),
    fps: int = 30,
    output_dir: str = "calibration_out",
    num_captures: int = 30,
) -> Calibration:
    """Walk the operator through N manual robot positions.

    For each position the operator moves the robot so the ArUco marker on
    the end-effector is visible in the workspace.  Pressing Enter captures
    the marker's 6-DoF pose in the camera frame (recovered via solvePnP on
    the RGB image, no depth required).  After all N poses are captured the
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
        Physical side length of the marker, in metres.  Black-edge to
        black-edge.  Must match OpenCV's ``markerLength`` convention.
    color_resolution : (int, int)
        (width, height) for the RealSense color stream.
    fps : int
        Frame rate for the color stream.
    output_dir : str
        Folder where all calibration artifacts are written.
    num_captures : int
        Number of frames to capture per pose.  Each frame yields a
        6-DoF marker pose; 3σ filtering and median across frames is used
        to pick the per-pose camera point.

    Returns
    -------
    Calibration instance.
    """
    from pathlib import Path

    if rs is None:
        raise RuntimeError("pyrealsense2 is not installed — cannot capture frames.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_path = str(out / "calibration.json")
    log_path = out / "captures.csv"

    pipeline = rs.pipeline()
    config = rs.config()
    w, h = color_resolution
    config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)

    profile = pipeline.start(config)

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_profile.get_intrinsics()
    K, D = intrinsics_to_KD(intrinsics)

    cal = Calibration()
    camera_pts: list[np.ndarray] = []
    robot_pts: list[np.ndarray] = []
    rvecs: list[np.ndarray] = []

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

            # Capture num_captures frames; per frame, recover 6-DoF pose
            # via solvePnP (cv2.SOLVEPNP_IPPE_SQUARE) on the RGB image.
            rvec_samples: list[np.ndarray] = []
            tvec_samples: list[np.ndarray] = []
            reproj_errs: list[float] = []
            best_rgb: np.ndarray | None = None
            best_corners: np.ndarray | None = None
            best_reproj: float = float("inf")

            for n in range(num_captures):
                rgb = None
                for _ in range(60):
                    frames = pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        rgb = np.asanyarray(color_frame.get_data()).copy()
                        break

                if rgb is None:
                    continue

                corners = detect_marker_corners(rgb, marker_id=marker_id)
                if corners is None:
                    continue

                rvec, tvec, err = marker_pose_from_corners(
                    corners, marker_size_m, K, D
                )
                if err > 0.6:
                    continue

                rvec_samples.append(rvec)
                tvec_samples.append(tvec)
                reproj_errs.append(err)

                if err < best_reproj:
                    best_reproj = err
                    best_rgb = rgb
                    best_corners = corners

                print(f"    {n + 1}/{num_captures}", end="\r")

            n_total = len(rvec_samples)
            if n_total == 0:
                print("  ✗ No valid frames captured. Skipping.")
                continue

            if n_total < num_captures:
                print(
                    f"  ⚠ Only {n_total}/{num_captures} frames valid "
                    f"(reproj_err ≤ 0.6 px)."
                )
            else:
                print(f"  ✓ Captured {n_total} frames." + " " * 20)

            rvecs_arr = np.array(rvec_samples)
            tvecs_arr = np.array(tvec_samples)

            def _three_sigma_filter(arr: np.ndarray) -> np.ndarray:
                if len(arr) < 3:
                    return np.ones(len(arr), dtype=bool)
                med = np.median(arr, axis=0)
                std = np.std(arr, axis=0)
                std = np.where(std < 1e-9, 1e-9, std)
                return np.all(np.abs(arr - med) <= 3.0 * std, axis=1)

            keep_t = _three_sigma_filter(tvecs_arr)
            keep_r = _three_sigma_filter(rvecs_arr)
            keep = keep_t & keep_r
            n_accepted = int(np.sum(keep))
            if n_accepted < 3:
                keep = np.ones(n_total, dtype=bool)
                n_accepted = n_total

            tvec_med = np.median(tvecs_arr[keep], axis=0)
            rvec_med = np.median(rvecs_arr[keep], axis=0)
            reproj_med = float(np.median(np.array(reproj_errs)[keep]))

            tvec_std_mm = np.std(tvecs_arr[keep], axis=0) * 1000
            rvec_std_deg = (
                np.std(rvecs_arr[keep], axis=0) * (180.0 / np.pi)
            )
            print(
                f"    tvec (median): ({tvec_med[0]:+.4f}, {tvec_med[1]:+.4f}, {tvec_med[2]:+.4f}) m"
            )
            print(
                f"    tvec std (mm): ({tvec_std_mm[0]:.1f}, {tvec_std_mm[1]:.1f}, {tvec_std_mm[2]:.1f})"
            )
            print(
                f"    rvec std (deg): ({rvec_std_deg[0]:.2f}, {rvec_std_deg[1]:.2f}, {rvec_std_deg[2]:.2f})"
            )
            print(
                f"    reproj_err (med): {reproj_med:.3f} px   "
                f"accepted: {n_accepted}/{n_total}"
            )

            camera_pts.append(tvec_med.astype(np.float64))
            robot_pts.append(np.array([rx, ry, rz], dtype=np.float64))
            rvecs.append(rvec_med.astype(np.float64))

            best_u, best_v = (
                marker_center_pixel(best_corners) if best_corners is not None else (0, 0)
            )

            if i == 0:
                csv_header = (
                    "pose,robot_x,robot_y,robot_z,"
                    "tvec_x,tvec_y,tvec_z,rvec_x,rvec_y,rvec_z,reproj_err_px,"
                    "accepted,"
                    "u0,v0,u1,v1,u2,v2,u3,v3\n"
                )
                with open(log_path, "w") as f:
                    f.write(csv_header)

            if best_corners is not None:
                c_flat = [f"{c[0]:.2f},{c[1]:.2f}" for c in best_corners]
            else:
                c_flat = ["0,0"] * 4
            with open(log_path, "a") as f:
                f.write(
                    f"{i + 1},"
                    f"{rx:.6f},{ry:.6f},{rz:.6f},"
                    f"{tvec_med[0]:.6f},{tvec_med[1]:.6f},{tvec_med[2]:.6f},"
                    f"{rvec_med[0]:.6f},{rvec_med[1]:.6f},{rvec_med[2]:.6f},"
                    f"{reproj_med:.4f},"
                    f"{n_accepted},"
                    + ",".join(c_flat)
                    + "\n"
                )

            if best_rgb is not None:
                debug = best_rgb.copy()
                if best_corners is not None:
                    for (cx, cy) in best_corners:
                        cv2.circle(debug, (int(cx), int(cy)), 4, (0, 255, 0), -1)
                    pts = best_corners.astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(debug, [pts], True, (0, 255, 255), 2)
                cv2.circle(debug, (int(best_u), int(best_v)), 5, (0, 0, 255), -1)
                cv2.imwrite(str(out / f"pose_{i + 1:02d}.jpg"), debug)


        if len(camera_pts) < 3:
            raise RuntimeError(
                f"Only {len(camera_pts)} valid captures — need at least 3."
            )

        # --- Validation ---
        ca = np.array(camera_pts)
        ro = np.array(robot_pts)
        rv = np.array(rvecs)

        inliers, outliers = find_best_subset(ca, ro)
        print(
            f"\n  Cross-pose rvec consistency (no-rotation gripper, "
            f"all poses should agree):"
        )
        rvec_outliers: list[int] = []
        if len(rv) >= 3:
            median_rv = np.median(rv, axis=0)
            dev_deg = np.linalg.norm(rv - median_rv, axis=1) * (180.0 / np.pi)
            median_dev = float(np.median(dev_deg))
            mad = float(np.median(np.abs(dev_deg - median_dev)))
            threshold = median_dev + 5.0 * 1.4826 * mad
            for k, d in enumerate(dev_deg):
                flag = " ⚠" if d > threshold else ""
                print(f"     Pose {k + 1}:  {d:.2f}°{flag}")
                if d > threshold:
                    rvec_outliers.append(k)
            print(
                f"  (rvec threshold: median {median_dev:.2f}° + "
                f"5×1.4826×MAD {mad:.2f}° = {threshold:.2f}°)"
            )
        validate_correspondences(ca, ro, rvecs=rv)

        if rvec_outliers:
            outliers = sorted(set(outliers) | set(rvec_outliers))
            inliers = sorted(set(range(len(camera_pts))) - set(outliers))
            print(
                f"\n  ⚠  Excluding {len(rvec_outliers)} pose(s) on rvec grounds: "
                f"{[i + 1 for i in rvec_outliers]}"
            )

        print(f"\n{'=' * 60}")
        print(f"  Collected {len(camera_pts)} / {len(robot_poses)} poses")
        print(f"  Best subset: {len(inliers)} inlier(s), {len(outliers)} outlier(s)")

        if outliers:
            print("\n  Outlier poses (excluded):")
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
                rvecs = [rvecs[i] for i in inliers]
            else:
                try:
                    redo = int(choice) - 1
                except ValueError:
                    redo = -1
                if redo < 0 or redo >= len(robot_poses):
                    camera_pts = [camera_pts[i] for i in inliers]
                    robot_pts = [robot_pts[i] for i in inliers]
                    rvecs = [rvecs[i] for i in inliers]
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
                    for _ in range(60):
                        frames = pipeline.wait_for_frames()
                        color_frame = frames.get_color_frame()
                        if color_frame:
                            rgb = np.asanyarray(color_frame.get_data()).copy()
                            break

                    if rgb is None:
                        print("  ✗ Failed to get frames. Using best subset.")
                        camera_pts = [camera_pts[i] for i in inliers]
                        robot_pts = [robot_pts[i] for i in inliers]
                        rvecs = [rvecs[i] for i in inliers]
                    else:
                        corners = detect_marker_corners(rgb, marker_id=marker_id)
                        if corners is None:
                            print(
                                f"  ✗ Marker {marker_id} not detected. Using best subset."
                            )
                            camera_pts = [camera_pts[i] for i in inliers]
                            robot_pts = [robot_pts[i] for i in inliers]
                            rvecs = [rvecs[i] for i in inliers]
                        else:
                            rvec_new, tvec_new, err_new = marker_pose_from_corners(
                                corners, marker_size_m, K, D
                            )
                            if err_new > 0.6:
                                print(
                                    f"  ✗ Bad re-capture "
                                    f"(reproj={err_new:.3f} px). Using best subset."
                                )
                                camera_pts = [camera_pts[i] for i in inliers]
                                robot_pts = [robot_pts[i] for i in inliers]
                                rvecs = [rvecs[i] for i in inliers]
                            else:
                                camera_pts[redo] = tvec_new.astype(np.float64)
                                robot_pts[redo] = np.array(
                                    [rx, ry, rz], dtype=np.float64
                                )
                                rvecs[redo] = rvec_new.astype(np.float64)
                                print(
                                    f"  ✓ Re-captured:  reproj={err_new:.3f} px"
                                )
                                print(
                                    f"     tvec : ({tvec_new[0]:+.4f}, {tvec_new[1]:+.4f}, {tvec_new[2]:+.4f}) m"
                                )

                                u, v = marker_center_pixel(corners)
                                c_flat = [f"{c[0]:.2f},{c[1]:.2f}" for c in corners]
                                with open(log_path) as f:
                                    rows = f.readlines()
                                rows[redo + 1] = (
                                    f"{redo + 1},"
                                    f"{rx:.6f},{ry:.6f},{rz:.6f},"
                                    f"{tvec_new[0]:.6f},{tvec_new[1]:.6f},{tvec_new[2]:.6f},"
                                    f"{rvec_new[0]:.6f},{rvec_new[1]:.6f},{rvec_new[2]:.6f},"
                                    f"{err_new:.4f},"
                                    f"1,"
                                    + ",".join(c_flat)
                                    + "\n"
                                )
                                with open(log_path, "w") as f:
                                    f.writelines(rows)

                                debug = rgb.copy()
                                for (cx, cy) in corners:
                                    cv2.circle(
                                        debug, (int(cx), int(cy)), 4, (0, 255, 0), -1
                                    )
                                pts = corners.astype(np.int32).reshape(-1, 1, 2)
                                cv2.polylines(debug, [pts], True, (0, 255, 255), 2)
                                cv2.circle(debug, (int(u), int(v)), 5, (0, 0, 255), -1)
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
            (150 / 1000, -100 / 1000, 550 / 1000),
        ]
        run_interactive_calibration(
            poses, marker_size_m=0.03, output_dir="calibration_out"
        )
