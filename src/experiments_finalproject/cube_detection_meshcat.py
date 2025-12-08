"""
Meshcat-enabled cube detection demo on the table-to-table scenario.

- Loads `table_to_table_scene_0.yaml` (or a provided scene) from generated_scenarios.
- Builds plant + scene_graph from directives, adds eye-in-hand RgbdSensor (as in main_sim).
- Starts Meshcat and attaches MeshcatVisualizer so you can view the scene/camera pose.
- Captures one RGB/depth frame from the camera, runs HSV-based cube detection with
  a depth-guided fallback, draws a bounding box, and optionally saves the result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import manipulation
from manipulation.station import ApplyMultibodyPlantConfig, LoadScenario
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    CameraInfo,
    ClippingRange,
    ColorRenderCamera,
    DepthRange,
    DepthRenderCamera,
    DiagramBuilder,
    ImageDepth32F,
    ImageRgba8U,
    MakeRenderEngineVtk,
    MeshcatVisualizer,
    RenderEngineVtkParams,
    RenderCameraCore,
    RgbdSensor,
    RigidTransform,
    RollPitchYaw,
    StartMeshcat,
)
from pydrake.multibody.parsing import ModelDirectives, Parser, ProcessModelDirectives

# Ensure local src modules (cameras.py) are importable
ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from cameras import add_eye_in_hand_camera  # noqa: E402

SCENARIO_DIR = ROOT / "src" / "experiments_finalproject" / "generated_scenarios"
DEFAULT_SCENE = "table_to_table_scene_0.yaml"
OUTPUT_DIR = ROOT / "src" / "experiments_finalproject" / "camera-output"


def drake_image_to_rgb(img: ImageRgba8U) -> np.ndarray:
    if img.size() == 0:
        raise ValueError("Empty image")
    arr = np.array(img.data, copy=False).reshape(img.height(), img.width(), 4)
    return arr[:, :, :3]


def drake_depth_to_array(img: ImageDepth32F) -> np.ndarray | None:
    if img.size() == 0:
        return None
    return np.array(img.data, copy=False).reshape(img.height(), img.width())


def detect_cube(
    rgb: np.ndarray,
    depth: np.ndarray | None,
    lower_hsv,
    upper_hsv,
    lower_hsv2=None,
    upper_hsv2=None,
    min_area_frac: float = 0.0003,
    center_bias: float = 0.001,
) -> dict:
    """
    HSV detect cube; fallback to nearest-depth region if HSV empty.
    Supports optional second HSV range (useful for red wraparound).
    
    Args:
        rgb: RGB image array
        depth: depth array (or None)
        lower_hsv/upper_hsv: primary HSV bounds
        lower_hsv2/upper_hsv2: optional secondary HSV bounds for wraparound colors
        min_area_frac: discard contours smaller than this fraction of image area
        center_bias: weight for favoring contours near image center (higher = more bias)
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    mask1 = cv2.inRange(hsv, lower_hsv, upper_hsv)
    if lower_hsv2 is not None and upper_hsv2 is not None:
        mask2 = cv2.inRange(hsv, lower_hsv2, upper_hsv2)
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = mask1

    # Clean up mask to reduce tiny speckles
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        if depth is None:
            raise RuntimeError("No cube-like contour found; adjust HSV bounds.")
        valid = np.isfinite(depth) & (depth > 0)
        if not np.any(valid):
            raise RuntimeError("No valid depth pixels; cannot fallback.")
        min_depth = np.min(depth[valid])
        near_mask = (depth <= min_depth + 0.05).astype(np.uint8) * 255
        contours, _ = cv2.findContours(near_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise RuntimeError("No cube-like contour found; adjust HSV bounds or scene.")

    # Filter by minimum area
    img_area = rgb.shape[0] * rgb.shape[1]
    area_thresh = max(10, min_area_frac * img_area)
    filtered = [c for c in contours if cv2.contourArea(c) >= area_thresh]
    if not filtered:
        raise RuntimeError("Contours found but all below min area threshold.")

    # Pick contour by combined score: area minus center distance penalty
    h_img, w_img = rgb.shape[:2]
    cX_img = w_img / 2.0
    cY_img = h_img / 2.0

    def score_contour(c):
        area = cv2.contourArea(c)
        M = cv2.moments(c)
        if M["m00"] == 0:
            return -np.inf
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        dist2 = (cx - cX_img) ** 2 + (cy - cY_img) ** 2
        return area - center_bias * dist2

    main = max(filtered, key=score_contour)

    # Axis-aligned bbox (for legacy logging)
    x, y, w, h = cv2.boundingRect(main)
    cx = int(x + w / 2)
    cy = int(y + h / 2)
    z = float(depth[cy, cx]) if depth is not None else None

    # Rotated bbox to capture orientation
    rect = cv2.minAreaRect(main)  # ((cx, cy), (w, h), angle)
    box = cv2.boxPoints(rect)
    box = box.astype(np.int32)
    raw_angle = rect[2]
    w_r, h_r = rect[1]
    # Normalize angle so that it represents the long side orientation
    angle_deg = raw_angle if w_r >= h_r else raw_angle + 90.0

    vis = rgb.copy()
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.drawContours(vis, [box], 0, (0, 255, 255), 2)  # rotated box in yellow
    cv2.circle(vis, (cx, cy), 4, (255, 0, 0), -1)
    cv2.putText(vis, f"{angle_deg:.1f} deg", (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

    return {
        "bbox": (x, y, w, h),
        "rotated_box": box.tolist(),
        "angle_deg": float(angle_deg),
        "centroid": (cx, cy),
        "depth_m": z,
        "vis": vis,
    }


def _add_overhead_camera(builder, scene_graph, camera_xy=(0.5, 0.3), camera_z=.75):
    """World-mounted top-down camera looking straight down at the table."""
    # Renderer
    renderer_name = "vtk_renderer_overhead"
    scene_graph.AddRenderer(renderer_name, MakeRenderEngineVtk(RenderEngineVtkParams()))

    width = 720
    height = 480
    fov_y = np.pi / 4  # 45 deg
    intrinsics = CameraInfo(width, height, fov_y)
    clipping = ClippingRange(0.01, 5.0)
    X_BS = RigidTransform()
    camera_core = RenderCameraCore(renderer_name, intrinsics, clipping, X_BS)
    depth_range = DepthRange(0.1, 5.0)
    depth_camera = DepthRenderCamera(camera_core, depth_range)
    color_camera = ColorRenderCamera(camera_core, False)

    parent_id = scene_graph.world_frame_id()

    # Point optical axis along -Z (downwards) from a spot above the table.
    rpy = RollPitchYaw(np.pi, 0, 0)  # flip to look down
    p = np.array([camera_xy[0], camera_xy[1], camera_z])
    X_PB = RigidTransform(rpy, p)

    rgbd_sensor = builder.AddSystem(
        RgbdSensor(
            parent_id=parent_id,
            X_PB=X_PB,
            color_camera=color_camera,
            depth_camera=depth_camera,
        )
    )
    # Connect sensor to scene graph
    builder.Connect(
        scene_graph.get_query_output_port(),
        rgbd_sensor.query_object_input_port(),
    )
    return rgbd_sensor


def build_diagram(scene_path: Path, use_overhead: bool):
    scenario = LoadScenario(filename=str(scene_path))

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(
        builder, time_step=scenario.plant_config.time_step
    )
    ApplyMultibodyPlantConfig(scenario.plant_config, plant)
    parser = Parser(plant, scene_graph)

    manipulation_models_path = Path(manipulation.__file__).parent / "models"
    parser.package_map().Add("manipulation", str(manipulation_models_path))
    drake_models_path = manipulation_models_path / "drake_models"
    if drake_models_path.is_dir():
        parser.package_map().Add("drake_models", str(drake_models_path))

    directives = ModelDirectives(directives=scenario.directives)
    ProcessModelDirectives(directives, parser)

    plant.Finalize()
    ee_frame = plant.GetFrameByName("iiwa_link_7")

    if use_overhead:
        rgbd_sensor = _add_overhead_camera(builder, scene_graph)
    else:
        rgbd_sensor, _image_converter = add_eye_in_hand_camera(
            builder, plant, scene_graph, ee_frame
        )

    # Meshcat
    meshcat = StartMeshcat()
    print(f"[Meshcat] Listening at {meshcat.web_url()}")
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    diagram = builder.Build()
    return diagram, rgbd_sensor, meshcat


def capture_and_detect(
    scene: str | Path | None = None,
    use_overhead: bool = True,
    hsv_lower=(0, 30, 30),
    hsv_upper=(25, 255, 255),
    hsv_lower2=(160, 30, 30),
    hsv_upper2=(180, 255, 255),
    min_area_frac: float = 0.0003,
    center_bias: float = 0.001,
    save_path: Path | None = None,
):
    """
    One-shot capture + detect helper for programmatic use.

    Returns a tuple (result_dict, meshcat_url) where result_dict includes
    bbox, rotated_box, angle_deg, centroid, depth_m, and vis (RGB for logging).
    """
    available = sorted(p.name for p in SCENARIO_DIR.glob("*.yaml"))
    if not available:
        raise FileNotFoundError(f"No YAML files in {SCENARIO_DIR}")
    scene_name = scene if scene is not None else DEFAULT_SCENE
    if isinstance(scene_name, Path):
        scene_path = scene_name
    else:
        if scene_name not in available:
            raise FileNotFoundError(f"Scene '{scene_name}' not found. Available: {available}")
        scene_path = SCENARIO_DIR / scene_name

    diagram, rgbd_sensor, meshcat = build_diagram(scene_path, use_overhead=use_overhead)
    context = diagram.CreateDefaultContext()
    diagram.ForcedPublish(context)

    sensor_context = rgbd_sensor.GetMyContextFromRoot(context)
    rgb_img = rgbd_sensor.color_image_output_port().Eval(sensor_context)
    depth_img = rgbd_sensor.depth_image_32F_output_port().Eval(sensor_context)

    rgb = drake_image_to_rgb(rgb_img)
    depth = drake_depth_to_array(depth_img)

    lower = np.array(hsv_lower, dtype=np.uint8)
    upper = np.array(hsv_upper, dtype=np.uint8)
    lower2 = np.array(hsv_lower2, dtype=np.uint8) if hsv_lower2 else None
    upper2 = np.array(hsv_upper2, dtype=np.uint8) if hsv_upper2 else None

    result = detect_cube(
        rgb,
        depth,
        lower,
        upper,
        lower2,
        upper2,
        min_area_frac=float(min_area_frac),
        center_bias=float(center_bias),
    )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), cv2.cvtColor(result["vis"], cv2.COLOR_RGB2BGR))

    return result, meshcat.web_url()


def main():
    parser = argparse.ArgumentParser(description="Meshcat cube detection demo.")
    parser.add_argument("--scene", type=str, default=None, help="Scenario YAML filename (from generated_scenarios)")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save visualization PNG")
    # Wider defaults for red objects (dual range to handle hue wraparound).
    parser.add_argument("--hsv-lower", type=int, nargs=3, default=[0, 30, 30], help="HSV lower bound (H S V)")
    parser.add_argument("--hsv-upper", type=int, nargs=3, default=[25, 255, 255], help="HSV upper bound (H S V)")
    parser.add_argument("--hsv-lower2", type=int, nargs=3, default=[160, 30, 30], help="HSV lower bound 2 (for red wraparound)")
    parser.add_argument("--hsv-upper2", type=int, nargs=3, default=[180, 255, 255], help="HSV upper bound 2 (for red wraparound)")
    parser.add_argument("--min-area-frac", type=float, default=0.0003, help="Minimum contour area as fraction of image area")
    parser.add_argument("--center-bias", type=float, default=0.001, help="Penalty weight for distance from image center")
    parser.add_argument(
        "--hold-secs",
        type=float,
        default=300.0,
        help="Keep Meshcat server alive for N seconds after capture (0 = exit immediately).",
    )
    parser.add_argument(
        "--eye-in-hand",
        action="store_true",
        help="Use the end-effector camera (original) instead of overhead top-down view.",
    )
    args = parser.parse_args()
    hold_secs = float(np.clip(args.hold_secs, 0.0, 600.0))
    use_overhead = not args.eye_in_hand

    available = sorted(p.name for p in SCENARIO_DIR.glob("*.yaml"))
    if not available:
        raise FileNotFoundError(f"No YAML files in {SCENARIO_DIR}")
    scene_name = args.scene or DEFAULT_SCENE
    if scene_name not in available:
        raise FileNotFoundError(f"Scene '{scene_name}' not found. Available: {available}")
    scene_path = SCENARIO_DIR / scene_name

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = Path(args.save) if args.save else OUTPUT_DIR / "cube_vis_meshcat.png"

    diagram, rgbd_sensor, meshcat = build_diagram(scene_path, use_overhead=use_overhead)
    context = diagram.CreateDefaultContext()

    # Publish once so Meshcat gets geometry and to get images
    diagram.ForcedPublish(context)

    sensor_context = rgbd_sensor.GetMyContextFromRoot(context)
    rgb_img = rgbd_sensor.color_image_output_port().Eval(sensor_context)
    depth_img = rgbd_sensor.depth_image_32F_output_port().Eval(sensor_context)

    rgb = drake_image_to_rgb(rgb_img)
    depth = drake_depth_to_array(depth_img)
    print(f"RGB shape: {rgb.shape}, depth shape: {None if depth is None else depth.shape}")

    lower = np.array(args.hsv_lower, dtype=np.uint8)
    upper = np.array(args.hsv_upper, dtype=np.uint8)
    lower2 = np.array(args.hsv_lower2, dtype=np.uint8) if args.hsv_lower2 else None
    upper2 = np.array(args.hsv_upper2, dtype=np.uint8) if args.hsv_upper2 else None

    result = detect_cube(
        rgb,
        depth,
        lower,
        upper,
        lower2,
        upper2,
        min_area_frac=float(args.min_area_frac),
        center_bias=float(args.center_bias),
    )
    x, y, w, h = result["bbox"]
    cx, cy = result["centroid"]
    z = result["depth_m"]
    angle = result.get("angle_deg")
    print(f"Detected bbox x={x}, y={y}, w={w}, h={h}; centroid=({cx},{cy}); depth={z}; angle={angle:.1f} deg")

    cv2.imwrite(str(save_path), cv2.cvtColor(result["vis"], cv2.COLOR_RGB2BGR))
    print(f"Saved visualization to {save_path}")
    print(f"Meshcat URL: {meshcat.web_url()}")

    if hold_secs > 0:
        import time

        print(f"Holding Meshcat open for {hold_secs} seconds... (Ctrl+C to stop)")
        try:
            time.sleep(hold_secs)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

