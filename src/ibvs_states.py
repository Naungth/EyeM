"""
IBVS State Machine Framework

Dictionary-based state machine where each state defines:
- Desired features (where to move features in image)
- DOF mask (which degrees of freedom to control)
- Convergence criteria (when to transition to next state)
- Next state (what state to transition to)

This allows different objectives for different states:
- APPROACH: Move forward to get closer to target
- CENTER: Center features in image (full 2D control)
- DESCEND: Move down to grasp height (z-axis control)
- GRASP: Close gripper (no visual servoing)
- LIFT: Move up (z-axis control)
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np


class IBVSState(Enum):
    """Visual servoing states."""
    WAIT_FOR_FEATURES = "wait_for_features"
    APPROACH = "approach"              # Move forward to approach target
    CENTER_ON_TARGET = "center"        # Center features in image (2D positioning)
    DESCEND_TO_GRASP = "descend"      # Move down to grasp height
    GRASP = "grasp"                    # Close gripper (no visual servoing)
    LIFT = "lift"                      # Move up after grasping
    MOVE_TO_GOAL = "move_to_goal"      # Move to goal position
    TASK_COMPLETE = "task_complete"    # Task finished


@dataclass
class StateConfig:
    """
    Configuration for a visual servoing state.
    
    Attributes:
        name: State name
        desired_features_fn: Function that returns desired feature coordinates (2N array)
                            Takes (context, N_features) as arguments
        dof_mask: 6D array [vx, vy, vz, wx, wy, wz] - which DOFs to enable
        error_threshold: Feature error threshold for convergence (pixels, RMS)
        convergence_time: Time to stay converged before transitioning (seconds)
        next_state: State to transition to when converged
        transition_condition: Optional function(context) -> bool for custom transitions
    """
    name: str
    desired_features_fn: Callable
    dof_mask: np.ndarray  # [vx, vy, vz, wx, wy, wz]
    error_threshold: float = 10.0  # pixels (RMS)
    convergence_time: float = 0.5  # seconds
    next_state: Optional[IBVSState] = None
    transition_condition: Optional[Callable] = None


# State configuration dictionary
# Each state defines its objective and control parameters
STATE_CONFIGS = {
    IBVSState.WAIT_FOR_FEATURES: StateConfig(
        name="Wait for Features",
        desired_features_fn=lambda ctx, N: np.zeros(2 * N),  # No desired features yet
        dof_mask=np.array([0, 0, 0, 0, 0, 0]),  # No movement
        error_threshold=float('inf'),  # Never converges
        next_state=IBVSState.APPROACH,
    ),
    
    IBVSState.APPROACH: StateConfig(
        name="Approach Target",
        desired_features_fn=lambda ctx, N: _center_features(N),  # Center of image
        dof_mask=np.array([1, 0, 0, 0, 0, 0]),  # Only forward/backward (vx)
        error_threshold=15.0,  # pixels
        convergence_time=0.3,
        next_state=IBVSState.CENTER_ON_TARGET,
    ),
    
    IBVSState.CENTER_ON_TARGET: StateConfig(
        name="Center on Target",
        desired_features_fn=lambda ctx, N: _center_features(N),  # Center of image
        dof_mask=np.array([1, 1, 0, 0, 0, 0]),  # x and y translation (vx, vy)
        error_threshold=10.0,  # pixels
        convergence_time=0.5,
        next_state=IBVSState.DESCEND_TO_GRASP,
    ),
    
    IBVSState.DESCEND_TO_GRASP: StateConfig(
        name="Descend to Grasp",
        desired_features_fn=lambda ctx, N: _center_features(N),  # Keep centered
        dof_mask=np.array([0, 0, 1, 0, 0, 0]),  # Only z translation (vz, down)
        error_threshold=10.0,  # pixels (maintain centering)
        convergence_time=0.2,
        next_state=IBVSState.GRASP,
    ),
    
    IBVSState.GRASP: StateConfig(
        name="Grasp Object",
        desired_features_fn=lambda ctx, N: np.zeros(2 * N),  # No visual servoing
        dof_mask=np.array([0, 0, 0, 0, 0, 0]),  # No movement (gripper closes separately)
        error_threshold=float('inf'),  # Never converges (time-based transition)
        convergence_time=1.0,  # Time to close gripper
        next_state=IBVSState.LIFT,
    ),
    
    IBVSState.LIFT: StateConfig(
        name="Lift Object",
        desired_features_fn=lambda ctx, N: _center_features(N),  # Keep centered
        dof_mask=np.array([0, 0, 1, 0, 0, 0]),  # Only z translation (vz, up)
        error_threshold=10.0,  # pixels
        convergence_time=0.3,
        next_state=IBVSState.MOVE_TO_GOAL,
    ),
    
    IBVSState.MOVE_TO_GOAL: StateConfig(
        name="Move to Goal",
        desired_features_fn=lambda ctx, N: _goal_features(N),  # Goal position in image
        dof_mask=np.array([1, 1, 0, 0, 0, 0]),  # x and y translation
        error_threshold=15.0,  # pixels
        convergence_time=0.5,
        next_state=IBVSState.TASK_COMPLETE,
    ),
    
    IBVSState.TASK_COMPLETE: StateConfig(
        name="Task Complete",
        desired_features_fn=lambda ctx, N: np.zeros(2 * N),  # No desired features
        dof_mask=np.array([0, 0, 0, 0, 0, 0]),  # No movement
        error_threshold=float('inf'),
        next_state=None,  # Terminal state
    ),
}


def _center_features(N_features: int) -> np.ndarray:
    """Return desired features at center of image."""
    desired_u = 320.0  # Center of 640px image
    desired_v = 240.0  # Center of 480px image
    features = np.zeros(2 * N_features)
    for i in range(N_features):
        features[2*i] = desired_u + (i % 2) * 20  # Slight offset for multiple features
        features[2*i + 1] = desired_v + (i // 2) * 20
    return features


def _goal_features(N_features: int) -> np.ndarray:
    """Return desired features at goal position in image."""
    # Goal: slightly to the right and up (example)
    goal_u = 400.0
    goal_v = 200.0
    features = np.zeros(2 * N_features)
    for i in range(N_features):
        features[2*i] = goal_u + (i % 2) * 20
        features[2*i + 1] = goal_v + (i // 2) * 20
    return features


def get_state_config(state: IBVSState) -> StateConfig:
    """Get configuration for a state."""
    if state not in STATE_CONFIGS:
        raise ValueError(f"Unknown state: {state}")
    return STATE_CONFIGS[state]


def get_initial_state() -> IBVSState:
    """Get the initial state."""
    return IBVSState.WAIT_FOR_FEATURES

