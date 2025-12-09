"""
Cube detection demo using Drake scenarios + OpenCV.

Loads a scenario YAML from generated_scenarios, builds a station from directives,
adds an eye-in-hand RgbdSensor (same pattern as main_sim), pulls one RGB/depth
frame, and runs a simple HSV threshold to find an orange cube. Depth is sampled
at the detected centroid if available.
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
    DiagramBuilder,
    ImageDepth32F,
    ImageRgba8U,
)
from pydrake.multibody.parsing import ModelDirectives, Parser, ProcessModelDirectives


ROOT = Path(__file__).resolve().parents[2]
# Ensure local src/ is on sys.path for module imports (cameras, etc.)
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cameras import add_eye_in_hand_camera  # noqa: E402
SCENARIO_DIR = ROOT / "src" / "experiments_finalproject" / "generated_scenarios"


def drake_image_to_rgb(img: ImageRgba8U) -> np.ndarray:
    """Convert Drake ImageRgba8U to HxWx3 uint8 RGB array."""
    if img.size() == 0:
        raise ValueError("Empty image")
    arr = np.array(img.data, copy=False).reshape(img.height(), img.width(), 4)
    return arr[:, :, :3]


def drake_depth_to_array(img: ImageDepth32F) -> np.ndarray | None:
    """Convert Drake ImageDepth32F to HxW float array in meters (or None)."""
    if img.size() == 0:
        return None
    return np.array(img.data, copy=False).reshape(img.height(), img.width())


def detect_cube(rgb: np.ndarray, depth: np.ndarray | None, lower_hsv, upper_hsv) -> dict:
    """Run HSV threshold to find cube; fallback to depth-near region if empty."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        if depth is None:
            raise RuntimeError("No cube-like contour found; adjust HSV bounds.")
        # Depth-guided fallback: keep pixels within 5cm of nearest depth
        valid = np.isfinite(depth) & (depth > 0)
        if not np.any(valid):
            raise RuntimeError("No valid depth pixels; cannot fallback.")
        min_depth = np.min(depth[valid])
        near_mask = (depth <= min_depth + 0.05).astype(np.uint8) * 255
        contours, _ = cv2.findContours(near_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise RuntimeError("No cube-like contour found; adjust HSV bounds or scene.")
    main = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main)
    cx = int(x + w / 2)
    cy = int(y + h / 2)

    z = float(depth[cy, cx]) if depth is not None else None

    vis = rgb.copy()
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(vis, (cx, cy), 4, (255, 0, 0), -1)

    return {
        "bbox": (x, y, w, h),
        "centroid": (cx, cy),
        "depth_m": z,
        "mask": mask,
        "vis": vis,
    }


def main():
    parser = argparse.ArgumentParser(description="Cube detection demo on Drake scenarios.")
    parser.add_argument("--scene", type=str, default=None, help="Scenario YAML filename (from generated_scenarios)")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save visualization PNG")
    parser.add_argument("--hsv-lower", type=int, nargs=3, default=[0, 40, 40], help="HSV lower bound (H S V)")
    parser.add_argument("--hsv-upper", type=int, nargs=3, default=[40, 255, 255], help="HSV upper bound (H S V)")
    args = parser.parse_args()

    available = sorted(p.name for p in SCENARIO_DIR.glob("*.yaml"))
    if not available:
        raise FileNotFoundError(f"No YAML files in {SCENARIO_DIR}")
    scene_name = args.scene or available[0]
    scene_path = SCENARIO_DIR / scene_name
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene file not found: {scene_path}")

    print(f"Loading scene: {scene_name}")
    scenario = LoadScenario(filename=str(scene_path))

    # Build plant+scene_graph from directives (mirrors main_sim wiring)
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
    ee_frame = plant.GetFrameByName("iiwa_link_7")  # matches scenario

    # Add eye-in-hand camera (same helper as main_sim)
    rgbd_sensor, _image_converter = add_eye_in_hand_camera(
        builder, plant, scene_graph, ee_frame
    )

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

    # Render once to populate images
    diagram.ForcedPublish(context)

    # Evaluate sensor outputs
    sensor_context = rgbd_sensor.GetMyContextFromRoot(context)
    rgb_img = rgbd_sensor.color_image_output_port().Eval(sensor_context)
    depth_img = rgbd_sensor.depth_image_32F_output_port().Eval(sensor_context)

    rgb = drake_image_to_rgb(rgb_img)
    depth = drake_depth_to_array(depth_img)
    print(f"RGB shape: {rgb.shape}, depth shape: {None if depth is None else depth.shape}")

    lower = np.array(args.hsv_lower, dtype=np.uint8)
    upper = np.array(args.hsv_upper, dtype=np.uint8)
    result = detect_cube(rgb, depth, lower, upper)
    x, y, w, h = result["bbox"]
    cx, cy = result["centroid"]
    z = result["depth_m"]
    print(f"Detected bbox x={x}, y={y}, w={w}, h={h}; centroid=({cx},{cy}); depth={z}")

    if args.save:
        out_path = Path(args.save)
        cv2.imwrite(str(out_path), cv2.cvtColor(result["vis"], cv2.COLOR_RGB2BGR))
        print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()

