# IBVS Simulation

Drake-based Image-Based Visual Servoing (IBVS) simulation for Kuka IIWA14 robot arm with Schunk WSG gripper.

## Overview

This project implements a closed-loop, image-driven controller that drives the IIWA end-effector velocity toward a target object based purely on image feature error.

## Features

- **IBVS Control Law**: Implements `v = -λ · L⁺ · e` with Damped Least Squares (DLS) for robustness
- **Feature Tracking**: Shi-Tomasi corner detection for stable feature points
- **Depth Estimation**: Real-time depth extraction with Exponential Moving Average (EMA) filtering
- **Joint Limits**: Velocity clamping and position-based constraints
- **Null-Space Control**: Biases joints towards center of range without affecting primary task
- **DOF Masking**: Enable/disable specific degrees of freedom (translational/rotational)
- **Eye-in-Hand Configuration**: Camera mounted on end-effector with adjoint transform

## Requirements

- Python 3.8+
- Drake (pydrake) >= 1.46.0
- NumPy >= 1.20.0, < 2.3.0
- OpenCV >= 4.5.0
- Matplotlib >= 3.5.0 (optional, for visualization)

## Installation

1. Install Python dependencies (including Drake):
   ```bash
   pip install -r requirements.txt
   ```


## Usage

Run the main simulation (uses venv Python if available):
```bash
python src/main_sim.py --scene stairs_scene_0.yaml --sim-time 5.0
```

### Scene loading
- Scenes live in `src/experiments_finalproject/generated_scenarios/` as YAML.
- Use `--scene <file.yaml>` to pick a scenario; defaults to the first YAML found.
- Station is built directly from the scenario directives (`LoadScenario`).

### Meshcat
- Meshcat auto-starts; URL is printed (e.g., `http://localhost:7000`).
- Port overrides are ignored in this Drake build; it binds to the default port.

### Visuals & logging
- Camera overlay frames save to `src/camera_screenshots/` (e.g., `camera_view_*.png`).
- Verbose q̇ prints come from `JointVelocityController`; silence by commenting prints in `main_sim.py` if desired.

## Project Structure

- `main_sim.py` - Main entry; IBVS loop + scenario-based scene loading + Meshcat
- `cameras.py` - Eye-in-hand RGB-D camera system
- `features.py` - Feature extraction and tracking
- `ibvs_controller.py` - IBVS control law implementation
- `jacobians.py` - Jacobian computation and inverse kinematics
- `state_machine.py` - Hierarchical state machine (for future use)
- `transforms.py` - Spatial transform utilities
- `visualization.py` - Camera view visualization with feature overlays

