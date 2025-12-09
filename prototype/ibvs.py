from depth_pro.depth_pro import DepthProConfig
import depth_pro
from machinevisiontoolbox import CentralCamera
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks import python as mp_python
import mediapipe as mp
from lerobot.cameras.configs import ColorMode  # type: ignore
from lerobot.cameras.opencv.configuration_opencv import (
    OpenCVCameraConfig,
)
from lerobot.robots.so101_follower.config_so101_follower import (
    SO101FollowerConfig,
)
from lerobot.robots import make_robot_from_config
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np
import torch
from lerobot.model.kinematics import RobotKinematics


sys.path.append(os.path.join(os.path.dirname(__file__), "lerobot", "src"))


@dataclass
class Params:

    port: str = os.environ.get("SO101_PORT", "/dev/ttyACM0")
    max_relative_target: Optional[float] = (
        float(os.environ.get("MAX_RELATIVE_TARGET", "10"))
    )

    # Camera (OpenCV)
    cam_index: int = int(os.environ.get("CV_INDEX", 3))
    width: int = int(os.environ.get("CAM_WIDTH", 640))
    height: int = int(os.environ.get("CAM_HEIGHT", 480))
    fps: int = int(os.environ.get("CAM_FPS", 30))
    default_fov_deg: float = float(os.environ.get("DEFAULT_FOV_DEG", 60))

    # Depth Pro
    dp_ckpt: str = os.path.join(
        os.path.dirname(
            __file__), "ml-depth-pro", "checkpoints", "depth_pro.pt"
    )
    infer_every_n: int = int(os.environ.get("INFER_EVERY_N", 30))

    # MediaPipe
    hand_model_path: str = os.environ.get(
        "HAND_MODEL_PATH", os.path.join(
            os.path.dirname(__file__), "hand_landmarker.task")
    )
    landmark_id: int = int(os.environ.get("LANDMARK_ID", 9))
    depth_roi_px: int = int(os.environ.get("DEPTH_ROI_PX", 5))

    # IBVS
    control_dt: float = float(os.environ.get("CONTROL_DT", 0.033))
    ibvs_gain: float = float(os.environ.get("IBVS_GAIN", 5))
    dls_lambda: float = float(os.environ.get("DLS_LAMBDA", 0.01))
    control_dof_mask: str = os.environ.get("CONTROL_DOF_MASK", "1,1,1,0,0,0")
    target_x: float = float(os.environ.get("TARGET_X", 0.5))
    target_y: float = float(os.environ.get("TARGET_Y", 0.5))

    # Calibration and rectification
    calib_path: Optional[str] = os.environ.get("CALIB_PATH", None)
    fisheye: bool = os.environ.get(
        "FISHEYE", "true").lower() in ("1", "true", "yes")
    knew_scale: float = float(os.environ.get(
        "KNEW_SCALE", 0.8))  # zoom-in to remove borders
    rotate: str = os.environ.get(
        "ROTATE", "none").lower()  # none|90cw|90ccw|180
    mirror: bool = os.environ.get(
        "MIRROR", "false").lower() in ("1", "true", "yes")

    # IK and safety
    urdf_path: str = os.environ.get(
        "SO101_URDF", os.path.join(os.path.dirname(
            __file__), "SO-ARM100/Simulation/SO101/so101_new_calib.urdf")
    )
    target_frame: str = os.environ.get("TARGET_FRAME", "gripper_frame_link")
    max_step_deg: float = float(os.environ.get("MAX_STEP_DEG", 8.0))

    # Eye-in-hand extrinsics (camera relative to EE)
    use_camera_to_ee_adjoint: bool = os.environ.get(
        "USE_CAMERA_TO_EE_ADJOINT", "true").lower() in ("1", "true", "yes")
    eih_tx: float = float(os.environ.get("EIH_TX", 0.0))
    eih_ty: float = float(os.environ.get("EIH_TY", 0.0))
    eih_tz: float = float(os.environ.get("EIH_TZ", 0.0))
    eih_rx_deg: float = float(os.environ.get("EIH_RX_DEG", 0))
    eih_ry_deg: float = float(os.environ.get("EIH_RY_DEG", 0.0))
    eih_rz_deg: float = float(os.environ.get("EIH_RZ_DEG", -90))

    # Depth filtering
    depth_ema_alpha: float = float(os.environ.get("DEPTH_EMA_ALPHA", 0.6))
    depth_min_m: float = float(os.environ.get("DEPTH_MIN_M", 0.15))
    depth_max_m: float = float(os.environ.get("DEPTH_MAX_M", 2.0))

    # Depth-based joint-2 bias
    depth_bias_enable: bool = os.environ.get(
        "DEPTH_BIAS_ENABLE", "true").lower() in ("1", "true", "yes")
    depth_bias_target_m: float = float(
        os.environ.get("DEPTH_BIAS_TARGET_M", 0.5))
    depth_bias_gain_deg_per_m: float = float(os.environ.get(
        "DEPTH_BIAS_GAIN_DEG_PER_M", 20.0))
    depth_bias_pos_scale: float = float(os.environ.get(
        "DEPTH_BIAS_POS_SCALE", 0.4))
    depth_bias_neg_scale: float = float(os.environ.get(
        "DEPTH_BIAS_NEG_SCALE",1.5))

    # Hardware configuration
    use_gripper: bool = os.environ.get(
        "USE_GRIPPER", "false").lower() in ("1", "true", "yes")

    # Homing before IBVS
    home_before_ibvs: bool = os.environ.get(
        "HOME_BEFORE_IBVS", "true").lower() in ("1", "true", "yes")
    home_pos_deg: str = os.environ.get("HOME_POS_DEG", "0,-30,10,0,0")
    home_hold_sec: float = float(os.environ.get("HOME_HOLD_SEC", 0.033))
    home_tol_deg: float = float(os.environ.get("HOME_TOL_DEG", 1.0))

    # Force calibration (ignore existing file and rerun interactive calibration)
    force_calibrate: bool = os.environ.get(
        "FORCE_CALIBRATE", "false").lower() in ("1", "true", "yes")

    # Debugging
    debug_signs: bool = os.environ.get(
        "DEBUG_SIGNS", "false").lower() in ("1", "true", "yes")

    # Test/monitor: report how often send_action is called (calls/sec)
    monitor_send_action_rate: bool = os.environ.get(
        "MONITOR_SEND_ACTION_RATE", "true").lower() in ("1", "true", "yes")


def get_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_depth_pro(params: Params):
    config = DepthProConfig(
        patch_encoder_preset="dinov2l16_384",
        image_encoder_preset="dinov2l16_384",
        decoder_features=256,
        use_fov_head=True,
        fov_encoder_preset="dinov2l16_384",
        checkpoint_uri=params.dp_ckpt,
    )

    device = get_torch_device()
    precision = torch.half if device.type in {"cuda", "mps"} else torch.float32

    model, transform = depth_pro.create_model_and_transforms(
        config=config, device=device, precision=precision
    )
    model.eval()
    if device.type == "cuda" and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True
    print(f"Depth Pro on device: {device} (precision={precision})")
    return model, transform, device


def parse_dof_mask(mask_str: str) -> np.ndarray:
    values = [int(x) for x in mask_str.split(",")]
    if len(values) != 6:
        raise ValueError(
            "CONTROL_DOF_MASK must have 6 comma-separated 0/1 values")
    return np.array(values, dtype=float)


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def adjoint_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    upper = np.hstack((R, np.zeros((3, 3))))
    lower = np.hstack((skew(t) @ R, R))
    return np.vstack((upper, lower))


def euler_rpy_to_rotation(rx_rad: float, ry_rad: float, rz_rad: float) -> np.ndarray:
    cx, sx = np.cos(rx_rad), np.sin(rx_rad)
    cy, sy = np.cos(ry_rad), np.sin(ry_rad)
    cz, sz = np.cos(rz_rad), np.sin(rz_rad)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0],
                  [0.0, 0.0, 1.0]], dtype=float)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0],
                  [-sy, 0.0, cy]], dtype=float)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx],
                  [0.0, sx, cx]], dtype=float)
    return Rz @ Ry @ Rx


def draw_crosshair(img: np.ndarray, x: int, y: int, size: int = 10, color: tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> None:
    x = int(x)
    y = int(y)
    cv2.line(img, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)


class SendActionRateMonitor:
    """Lightweight wrapper to measure send_action call rate.

    Install with SendActionRateMonitor(robot).install() and it will print calls/sec
    every second and a final average on uninstall(). Designed to be easily removed.
    """

    def __init__(self, robot) -> None:
        self.robot = robot
        self._original_send_action = getattr(robot, "send_action")
        self._total_calls = 0
        self._window_calls = 0
        self._t_start = time.perf_counter()
        self._t_window = self._t_start

        def _wrapped_send_action(action):
            self._total_calls += 1
            self._window_calls += 1
            now = time.perf_counter()
            if now - self._t_window >= 1.0:
                rate = self._window_calls / (now - self._t_window)
                print(
                    f"[SEND_RATE] {rate:.1f} calls/s (total={self._total_calls})")
                self._window_calls = 0
                self._t_window = now
            return self._original_send_action(action)

        self._wrapped_send_action = _wrapped_send_action

    def install(self) -> None:
        # type: ignore[attr-defined]
        self.robot.send_action = self._wrapped_send_action

    def uninstall(self) -> None:
        # type: ignore[attr-defined]
        self.robot.send_action = self._original_send_action
        t_elapsed = max(1e-9, time.perf_counter() - self._t_start)
        avg_rate = self._total_calls / t_elapsed
        print(
            f"[SEND_RATE] avg={avg_rate:.1f} calls/s over {t_elapsed:.2f}s (total={self._total_calls})")


def se3_exp_body(twist6: np.ndarray, dt: float) -> np.ndarray:
    v = np.asarray(twist6[:3], dtype=float)
    w = np.asarray(twist6[3:], dtype=float)
    theta = np.linalg.norm(w) * dt
    if theta < 1e-9:
        R = np.eye(3) + skew(w) * dt
        t = v * dt
    else:
        w_unit = w / np.linalg.norm(w)
        w_hat = skew(w_unit)
        R = (
            np.eye(3)
            + np.sin(theta) * w_hat
            + (1.0 - np.cos(theta)) * (w_hat @ w_hat)
        )
        I = np.eye(3)
        V = (
            I * dt
            + (1 - np.cos(theta)) / (np.linalg.norm(w) ** 2) * skew(w)
            + (theta - np.sin(theta)) /
            (np.linalg.norm(w) ** 3) * (skew(w) @ skew(w))
        )
        t = V @ v
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def compute_ibvs_twist(
    camera_model: CentralCamera,
    landmark_u: float,
    landmark_v: float,
    target_u: float,
    target_v: float,
    depth_m: float,
    gain: float,
    dof_mask_cam6: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pixel_error = np.array(
        [landmark_u - target_u, landmark_v - target_v], dtype=float)
    J = camera_model.visjac_p((landmark_u, landmark_v), depth_m)
    allowed_idx = np.where(dof_mask_cam6 > 0.5)[0]
    if allowed_idx.size == 0:
        return np.zeros(6, dtype=float), np.array([0.0, 0.0]), pixel_error

    J_red = J[:, allowed_idx]
    lam = 0.01
    JJt = J_red @ J_red.T
    inv_term = np.linalg.inv(JJt + (lam ** 2) * np.eye(J_red.shape[0]))
    J_pinv = J_red.T @ inv_term
    v_red = -gain * (J_pinv @ pixel_error)
    v = np.zeros(6, dtype=float)
    v[allowed_idx] = v_red
    normalized_error = np.array([pixel_error[0], pixel_error[1]])
    return v, normalized_error, pixel_error


def main():
    p = Params()

    # Build robot with OpenCV camera
    cam_cfg = OpenCVCameraConfig(
        index_or_path=p.cam_index,
        fps=p.fps,
        width=p.width,
        height=p.height,
        color_mode=ColorMode.RGB,
    )
    robot_cfg = SO101FollowerConfig(
        id="so101_ibvs",
        port=p.port,
        use_degrees=True,
        max_relative_target=p.max_relative_target,
        cameras={"front": cam_cfg},
        use_gripper=p.use_gripper,
    )
    robot = make_robot_from_config(robot_cfg)
    if p.force_calibrate:
        # Clear any loaded calibration so connect(calibrate=True) performs a fresh calibration
        robot.calibration = {}
    robot.connect()
    robot.configure()

    # Optional: monitor send_action call rate
    monitor = None
    if p.monitor_send_action_rate:
        monitor = SendActionRateMonitor(robot)
        monitor.install()

    # Optional homing to a defined upright, extended pose
    if p.home_before_ibvs:
        homing_joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ]
        try:
            target_vals = [float(x) for x in p.home_pos_deg.split(",")]
        except Exception:
            raise ValueError(
                "HOME_POS_DEG must be 5 comma-separated angles in degrees")
        if len(target_vals) != len(homing_joint_names):
            raise ValueError(
                "HOME_POS_DEG must have 5 values for shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll")

        print(f"Homing to: {dict(zip(homing_joint_names, target_vals))}")
        obs = robot.get_observation()
        q_curr_deg = np.array(
            [float(obs[f"{jn}.pos"]) for jn in homing_joint_names], dtype=float)
        max_step = p.max_step_deg
        for _ in range(100):
            delta = np.array(target_vals, dtype=float) - q_curr_deg
            if np.max(np.abs(delta)) <= p.home_tol_deg:
                break
            step = np.clip(delta, -max_step, max_step)
            q_cmd = q_curr_deg + step
            action = {f"{jn}.pos": float(
                q_cmd[i]) for i, jn in enumerate(homing_joint_names)}
            robot.send_action(action)
            time.sleep(p.home_hold_sec)
            obs = robot.get_observation()
            q_curr_deg = np.array(
                [float(obs[f"{jn}.pos"]) for jn in homing_joint_names], dtype=float)
        print("Homing complete.")

    if p.debug_signs:
        print(
            "DEBUG_SIGNS: rotate=", p.rotate,
            "mirror=", p.mirror,
            "use_adj=", p.use_camera_to_ee_adjoint,
            "EIH(deg)=(", p.eih_rx_deg, p.eih_ry_deg, p.eih_rz_deg, ")",
            "DOF_MASK=", p.control_dof_mask,
        )

    cam = robot.cameras.get("front") if hasattr(robot, "cameras") else None
    if cam is None:
        cam = next(iter(robot.cameras.values()))

    H = p.height
    W = p.width

    # Optional calibration/rectification setup
    rect_map1 = rect_map2 = None
    use_rect = False
    fx = fy = None
    cx = cy = None

    if p.calib_path and os.path.exists(p.calib_path):
        data = np.load(p.calib_path, allow_pickle=True)
        K = data["K"].astype(np.float64)
        D = data["D"].astype(np.float64)
        # Build rectification maps for current DIM
        DIM = (W, H)
        Knew = K.copy()
        Knew[0, 0] = float(K[0, 0] * p.knew_scale)
        Knew[1, 1] = float(K[1, 1] * p.knew_scale)
        # Keep principal point near center (can be refined from calibration)
        if p.fisheye:
            rect_map1, rect_map2 = cv2.fisheye.initUndistortRectifyMap(
                K, D, np.eye(3), Knew, DIM, cv2.CV_16SC2
            )
        else:
            rect_map1, rect_map2 = cv2.initUndistortRectifyMap(
                K, D, None, Knew, DIM, cv2.CV_16SC2
            )
        use_rect = True
        fx = float(Knew[0, 0])
        fy = float(Knew[1, 1])
        cx = float(Knew[0, 2]) if Knew[0, 2] != 0 else W / 2.0
        cy = float(Knew[1, 2]) if Knew[1, 2] != 0 else H / 2.0
        print(
            f"Loaded calibration from {p.calib_path}. Using fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

    if not use_rect:
        # Fallback to pinhole FOV guess
        fx = fy = 0.5 * W / np.tan(0.5 * np.deg2rad(p.default_fov_deg))
        cx = W / 2.0
        cy = H / 2.0
        print(f"No calibration. Using FOV={p.default_fov_deg}° → fx≈{fx:.1f}")

    # DepthPro focal length in px
    fpx = fx

    # Camera model for IBVS on rectified frames
    camera_model = CentralCamera(f=(fx, fy), pp=(cx, cy), imagesize=(W, H))

    # Depth Pro
    model, transform, device = load_depth_pro(p)

    # MediaPipe Hand Landmarker
    base_opts = mp_python.BaseOptions(model_asset_path=p.hand_model_path)
    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
    )
    hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_opts)

    dof_mask = parse_dof_mask(p.control_dof_mask)
    ema_depth = None
    last_depth_map = None
    # kept for visualization only; actions require current-frame detection
    last_landmark = None

    # Kinematics
    kin = RobotKinematics(
        urdf_path=p.urdf_path,
        target_frame_name=p.target_frame,
        joint_names=[
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ],
    )
    # Eye-in-hand adjoint
    rx = np.deg2rad(p.eih_rx_deg)
    ry = np.deg2rad(p.eih_ry_deg)
    rz = np.deg2rad(p.eih_rz_deg)
    R_e_c = euler_rpy_to_rotation(rx, ry, rz)
    t_e_c = np.array([p.eih_tx, p.eih_ty, p.eih_tz], dtype=float)
    Ad_e_c = adjoint_transform(R_e_c, t_e_c)
    R_c_e = R_e_c.T
    t_c_e = -R_c_e @ t_e_c
    Ad_c_e = adjoint_transform(R_c_e, t_c_e)

    # Parse home target once for loss-recovery homing
    try:
        _home_targets_list = [float(x) for x in p.home_pos_deg.split(",")]
    except Exception:
        raise ValueError(
            "HOME_POS_DEG must be 5 comma-separated angles in degrees")
    if len(_home_targets_list) != 5:
        raise ValueError(
            "HOME_POS_DEG must have 5 values for shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll"
        )
    home_targets_deg = np.array(_home_targets_list, dtype=float)

    print(
        f"Starting IBVS loop (ACTIVE) — Press ESC to quit."
    )

    last_infer_frame_idx = -1
    frame_idx = 0
    t_last = time.perf_counter()
    last_det_time = t_last
    # Depth inference timing counters
    depth_win_count = 0
    depth_win_ms = 0.0
    depth_win_start = time.perf_counter()

    while True:
        # Read frame
        frame = cam.async_read()  # RGB
        if frame is None:
            continue
        frame_rgb = frame
        # Reset current-frame detection
        current_landmark = None

        # Optional orientation fixes
        if p.rotate == "180":
            frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_180)
        elif p.rotate == "90cw":
            frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_90_CLOCKWISE)
        elif p.rotate == "90ccw":
            frame_rgb = cv2.rotate(
                frame_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if p.mirror:
            frame_rgb = cv2.flip(frame_rgb, 1)

        # Undistort
        if use_rect and rect_map1 is not None and rect_map2 is not None:
            frame_rgb = cv2.remap(
                frame_rgb, rect_map1, rect_map2, interpolation=cv2.INTER_LINEAR)

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Draw target crosshair (green)
        target_u = int(p.target_x * W)
        target_v = int(p.target_y * H)
        draw_crosshair(frame_bgr, target_u, target_v, size=10,
                       color=(0, 255, 0), thickness=2)

        # Landmark detection
        ts_ms = int(time.perf_counter() * 1000)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = hand_landmarker.detect_for_video(mp_img, ts_ms)
        if result and result.hand_landmarks:
            lm = result.hand_landmarks[0]
            if 0 <= p.landmark_id < len(lm):
                nx = float(lm[p.landmark_id].x)
                ny = float(lm[p.landmark_id].y)
                u_px = int(np.clip(nx * W, 0, W - 1))
                v_px = int(np.clip(ny * H, 0, H - 1))
                current_landmark = (u_px, v_px)
                last_landmark = current_landmark
                cv2.circle(frame_bgr, (u_px, v_px), 6, (0, 0, 255), -1)

        # Run Depth Pro every N frames
        if frame_idx % max(1, p.infer_every_n) == 0:
            with torch.no_grad():
                _t0 = time.perf_counter()
                inp = transform(frame_rgb)
                # type: ignore[attr-defined]
                model_dtype = next(model.parameters()).dtype
                fpx_tensor = torch.tensor(
                    float(fpx), device=device, dtype=model_dtype)
                pred = model.infer(inp, f_px=fpx_tensor)
                depth = pred["depth"].detach().cpu().numpy().squeeze()
                _t1 = time.perf_counter()
                infer_ms = (_t1 - _t0) * 1000.0
                print(
                    f"[DEPTH] infer took {infer_ms:.1f} ms (frame={frame_idx})")
                depth_win_count += 1
                depth_win_ms += infer_ms
                now = time.perf_counter()
                if now - depth_win_start >= 1.0:
                    rate = depth_win_count / (now - depth_win_start)
                    avg_ms = depth_win_ms / max(1, depth_win_count)
                    print(
                        f"[DEPTH] rate={rate:.1f} inf/s avg={avg_ms:.1f} ms over {now - depth_win_start:.2f}s")
                    depth_win_count = 0
                    depth_win_ms = 0.0
                    depth_win_start = now
            last_depth_map = depth
            last_infer_frame_idx = frame_idx
        frame_idx += 1

        # If we have current-frame landmark and depth, compute median depth in ROI and IBVS twist
        if current_landmark is not None and last_depth_map is not None:
            # update last detection timestamp
            last_det_time = time.perf_counter()
            u, v = current_landmark
            x0, x1 = max(0, u - p.depth_roi_px), min(W,
                                                     u + p.depth_roi_px + 1)
            y0, y1 = max(0, v - p.depth_roi_px), min(H,
                                                     v + p.depth_roi_px + 1)
            roi = last_depth_map[y0:y1, x0:x1]
            finite = np.isfinite(roi)
            valid_vals = roi[finite]
            if valid_vals.size > 0:
                depth_m = float(np.median(valid_vals))
                # Clamp and EMA
                depth_m = float(
                    np.clip(depth_m, p.depth_min_m, p.depth_max_m))
                if ema_depth is None:
                    ema_depth = depth_m
                else:
                    ema_depth = p.depth_ema_alpha * depth_m + \
                        (1.0 - p.depth_ema_alpha) * ema_depth

                # IBVS twist (camera frame)
                target_u = p.target_x * W
                target_v = p.target_y * H
                v_cam, norm_err, pix_err = compute_ibvs_twist(
                    camera_model,
                    float(u),
                    float(v),
                    float(target_u),
                    float(target_v),
                    ema_depth,
                    p.ibvs_gain,
                    dof_mask,
                )
                # Map camera twist to EE twist if configured
                v_ee = v_cam.copy()
                if p.use_camera_to_ee_adjoint:
                    v_ee = Ad_e_c @ v_cam
                if p.debug_signs:
                    print(
                        "DBG err_px=", np.round(pix_err, 1),
                        " v_cam=", np.round(v_cam, 3),
                        " v_ee=", np.round(v_ee, 3),
                    )

                # Read current joint positions (degrees)
                obs = robot.get_observation()
                joint_names = ["shoulder_pan", "shoulder_lift",
                               "elbow_flex", "wrist_flex", "wrist_roll"]
                q_curr_deg = np.array(
                    [float(obs[f"{jn}.pos"]) for jn in joint_names], dtype=float)

                # Integrate body twist to a target pose
                T_curr = kin.forward_kinematics(q_curr_deg)
                T_delta = se3_exp_body(v_ee, p.control_dt)
                T_next = T_curr @ T_delta

                # Differential IK to get next joint config (degrees)
                q_next_deg = kin.inverse_kinematics(
                    q_curr_deg, T_next, position_weight=1.0, orientation_weight=1e-3)

                # Add depth-based bias on joint 2 (shoulder_lift)
                if p.depth_bias_enable and ema_depth is not None and np.isfinite(ema_depth):
                    depth_err_m = float(ema_depth - p.depth_bias_target_m)
                    bias_deg = p.depth_bias_gain_deg_per_m * depth_err_m
                    # scale positive bias (farther than target) or negative bias (closer)
                    if bias_deg > 0.0:
                        bias_deg *= float(p.depth_bias_pos_scale)
                    elif bias_deg < 0.0:
                        bias_deg *= float(p.depth_bias_neg_scale)
                    pre_j2 = float(q_next_deg[1])
                    post_j2 = float(pre_j2 + bias_deg)
                    print(
                        f"[DEPTH_BIAS] ema={ema_depth:.3f}m target={p.depth_bias_target_m:.3f} err={depth_err_m:.3f}m gain={p.depth_bias_gain_deg_per_m:.2f} pos_scale={p.depth_bias_pos_scale:.2f} neg_scale={p.depth_bias_neg_scale:.2f} -> bias={bias_deg:.2f} deg | j2 pre={pre_j2:.2f} post={post_j2:.2f}"
                    )
                    # joint index 1 corresponds to shoulder_lift
                    q_next_deg[1] = post_j2

                # Clamp per-joint step
                max_step = p.max_step_deg
                step = np.clip(q_next_deg - q_curr_deg, -
                               max_step, max_step)
                q_cmd_deg = q_curr_deg + step
                print(
                    f"[DEPTH_BIAS] j2 curr={q_curr_deg[1]:.2f} cmd={q_cmd_deg[1]:.2f} step={q_cmd_deg[1]-q_curr_deg[1]:.2f} (max_step={p.max_step_deg:.2f})"
                )

                if p.debug_signs:
                    print("DBG q_curr=", np.round(q_curr_deg, 2),
                          " q_cmd=", np.round(q_cmd_deg, 2))

                # Freeze joint 4 (index 3) before sending
                q_cmd_deg[3] = q_curr_deg[3]
                # Build action and optionally send (skip joint index 3)
                action = {f"{jn}.pos": float(
                    q_cmd_deg[i]) for i, jn in enumerate(joint_names) if i != 3}
                sent = robot.send_action(action)
                cv2.putText(
                    frame_bgr,
                    f"depth={ema_depth:.3f}m err_px=({int(pix_err[0])},{int(pix_err[1])})",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
        else:
            cv2.putText(
                frame_bgr,
                "landmark: LOST",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            if time.perf_counter() - last_det_time > 4.0:
                obs = robot.get_observation()
                joint_names = [
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                ]
                q_curr_deg = np.array(
                    [float(obs[f"{jn}.pos"]) for jn in joint_names], dtype=float)

                delta = home_targets_deg - q_curr_deg
                max_step = p.max_step_deg
                step = np.clip(delta, -max_step, max_step)
                q_cmd_deg = q_curr_deg + step

                q_cmd_deg[3] = q_curr_deg[3]
                action = {f"{jn}.pos": float(
                    q_cmd_deg[i]) for i, jn in enumerate(joint_names) if i != 3}
                robot.send_action(action)

        # Show
        dt = time.perf_counter() - t_last
        fps = 1.0 / dt if dt > 0 else 0.0
        t_last = time.perf_counter()
        cv2.putText(
            frame_bgr,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("SO101 IBVS (preview)", frame_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        time.sleep(max(0.0, p.control_dt - (time.perf_counter() - t_last)))

    hand_landmarker.close()
    if p.monitor_send_action_rate and monitor is not None:
        monitor.uninstall()
    robot.disconnect()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()