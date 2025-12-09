# IBVS State Machine Framework Guide

## Overview

The visual servoing system now uses a **dictionary-based state machine** where each state defines its own objective, control parameters, and transition conditions.

## Project Structure

```
src/
├── ibvs_states.py          # State definitions and configuration dictionary
├── state_machine.py        # State machine implementation (uses STATE_CONFIGS)
├── ibvs_controller.py      # IBVS controller (now accepts dynamic DOF mask)
└── main_sim.py             # Main simulation (wires everything together)
```

## How It Works

### 1. State Configuration (`ibvs_states.py`)

Each state is defined in the `STATE_CONFIGS` dictionary with:

```python
STATE_CONFIGS = {
    IBVSState.APPROACH: StateConfig(
        name="Approach Target",
        desired_features_fn=lambda ctx, N: _center_features(N),  # Where to move features
        dof_mask=np.array([1, 0, 0, 0, 0, 0]),  # Which DOFs to control [vx,vy,vz,wx,wy,wz]
        error_threshold=15.0,  # Convergence threshold (pixels)
        convergence_time=0.3,  # Time to stay converged before transition
        next_state=IBVSState.CENTER_ON_TARGET,  # Where to go next
    ),
    # ... more states
}
```

### 2. State Machine (`state_machine.py`)

The `IBVSStateMachine` class:
- Tracks current state
- Outputs desired features based on current state
- Outputs DOF mask based on current state
- Handles state transitions based on convergence criteria

### 3. Integration (`main_sim.py`)

The state machine is connected to:
- **Input**: Current features (for transitions)
- **Input**: Convergence flag from IBVS controller
- **Output**: Desired features → IBVS controller
- **Output**: DOF mask → IBVS controller (dynamic control)

## Current States

1. **WAIT_FOR_FEATURES**: Wait until features are detected
2. **APPROACH**: Move forward to approach target (vx only)
3. **CENTER_ON_TARGET**: Center features in image (vx, vy)
4. **DESCEND_TO_GRASP**: Move down to grasp height (vz only)
5. **GRASP**: Close gripper (no visual servoing)
6. **LIFT**: Move up after grasping (vz only)
7. **MOVE_TO_GOAL**: Move to goal position (vx, vy)
8. **TASK_COMPLETE**: Terminal state

## Adding a New State

1. **Add state enum** in `ibvs_states.py`:
```python
class IBVSState(Enum):
    MY_NEW_STATE = "my_new_state"
```

2. **Add state config** in `STATE_CONFIGS`:
```python
IBVSState.MY_NEW_STATE: StateConfig(
    name="My New State",
    desired_features_fn=lambda ctx, N: _my_custom_features(N),
    dof_mask=np.array([1, 1, 1, 0, 0, 0]),  # Enable x, y, z translation
    error_threshold=12.0,
    convergence_time=0.4,
    next_state=IBVSState.NEXT_STATE,
),
```

3. **Add transition logic** in `state_machine.py` `_update_state()` method:
```python
elif current_state == IBVSState.MY_NEW_STATE:
    if is_converged:
        # Your transition logic
        should_transition = True
        next_state = config.next_state
```

## DOF Mask Reference

The DOF mask is a 6D array: `[vx, vy, vz, wx, wy, wz]`

- `vx, vy, vz`: Translation velocities (forward/back, left/right, up/down)
- `wx, wy, wz`: Rotation velocities (roll, pitch, yaw)

Examples:
- `[1, 0, 0, 0, 0, 0]`: Only forward/backward movement
- `[1, 1, 0, 0, 0, 0]`: Only x-y translation (2D positioning)
- `[0, 0, 1, 0, 0, 0]`: Only up/down movement
- `[1, 1, 1, 0, 0, 0]`: Full 3D translation, no rotation
- `[1, 1, 1, 1, 1, 1]`: Full 6DOF control

## Usage

The state machine is **enabled by default** in `main_sim.py`:

```python
use_state_machine = True  # Line 467
```

To disable and use simple constant desired features:
```python
use_state_machine = False
```

## Debugging

The state machine prints state transitions:
```
[StateMachine] Transitioned to: center
```

You can also access the current state name via the `state_name_output` port for visualization.

## What to edit

- Per-state objectives, masks, thresholds: `src/ibvs_states.py` (edit `STATE_CONFIGS`: `desired_features_fn`, `dof_mask`, `error_threshold`, `convergence_time`, `next_state`).
- Transition logic: `src/state_machine.py` (`_update_state` uses convergence/feature presence to move through states; add custom conditions here).
- Controller behavior: `src/ibvs_controller.py` (gain `lambda_gain`, DLS `dls_lambda`, convergence window/threshold).
- Visualization overlay (state label, RMS error text): `src/visualization.py` (uses `state_name_input` and `desired_features_input`).
- Wiring/integration: `main_sim.py` (connects the state machine, controller, visualizer; loads scenarios).
- Unused/legacy: root-level `visualization.py` and `prototype/` scripts are not used by `main_sim.py`; keep only for reference.

