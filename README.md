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

Run the main simulation:
```bash
python main_sim.py
```

## Project Structure

- `main_sim.py` - Main entry point, orchestrates the complete IBVS control loop
- `cameras.py` - Eye-in-hand RGB-D camera system
- `features.py` - Feature extraction and tracking
- `ibvs_controller.py` - IBVS control law implementation
- `jacobians.py` - Jacobian computation and inverse kinematics
- `state_machine.py` - Hierarchical state machine (for future use)
- `transforms.py` - Spatial transform utilities
- `visualization.py` - Camera view visualization with feature overlays

