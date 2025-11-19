"""
IBVS State Machine

Implements a hierarchical state machine for IBVS control, similar to the Planner
pattern in bin.py. States manage different phases of the visual servoing task.

Future states:
    - APPROACH_PREGRASP: Move to initial position
    - CENTER_ON_TARGET: Use IBVS to center on target
    - DESCEND_TO_GRASP: Move down to grasp height
    - GRASP: Close gripper
    - LIFT: Move up
    - MOVE_TO_GOAL: Move to goal position
"""

from enum import Enum
import numpy as np
from pydrake.all import (
    AbstractValue,
    InputPortIndex,
    LeafSystem,
    RigidTransform,
)


class IBVSState(Enum):
    """States for IBVS control task."""
    WAIT_FOR_FEATURES = 1
    CENTER_ON_TARGET = 2  # Main IBVS state
    TASK_COMPLETE = 3
    GO_HOME = 4


class IBVSStateMachine(LeafSystem):
    """
    State machine for IBVS control.
    
    Manages state transitions and provides desired features based on current state.
    Similar to Planner class in bin.py.
    """
    
    def __init__(self, N_features=4):
        """
        Initialize IBVS state machine.
        
        Args:
            N_features: Number of feature points
        """
        LeafSystem.__init__(self)
        self.N_features = N_features
        
        # Input: current feature error (for state transitions)
        self.error_input = self.DeclareVectorInputPort("feature_error", size=2 * N_features)
        
        # Input: convergence flag from IBVS controller
        self.converged_input = self.DeclareVectorInputPort("converged", size=1)
        
        # State: current mode
        self._mode_index = self.DeclareAbstractState(
            AbstractValue.Make(IBVSState.WAIT_FOR_FEATURES)
        )
        
        # Output: desired features (2N vector)
        self.desired_features_output = self.DeclareVectorOutputPort(
            "desired_features",
            size=2 * N_features,
            calc=self.CalcDesiredFeatures
        )
        
        # Output: control mode (for future PortSwitch integration)
        self.control_mode_output = self.DeclareAbstractOutputPort(
            "control_mode",
            lambda: AbstractValue.Make(InputPortIndex(0)),
            self.CalcControlMode
        )
        
        # Periodic update for state transitions
        self.DeclarePeriodicUnrestrictedUpdateEvent(0.1, 0.0, self.Update)
    
    def Update(self, context, state):
        """Update state machine based on current conditions."""
        mode = context.get_abstract_state(int(self._mode_index)).get_value()
        current_time = context.get_time()
        
        if mode == IBVSState.WAIT_FOR_FEATURES:
            # Wait until we have valid features
            error = self.error_input.Eval(context)
            if np.any(error != 0):  # Have some features
                state.get_mutable_abstract_state(int(self._mode_index)).set_value(
                    IBVSState.CENTER_ON_TARGET
                )
        
        elif mode == IBVSState.CENTER_ON_TARGET:
            # Check if converged
            converged = self.converged_input.Eval(context)
            if converged[0] > 0.5:  # Converged
                state.get_mutable_abstract_state(int(self._mode_index)).set_value(
                    IBVSState.TASK_COMPLETE
                )
        
        elif mode == IBVSState.TASK_COMPLETE:
            # Stay in this state
            pass
        
        elif mode == IBVSState.GO_HOME:
            # Future: implement go home logic
            pass
    
    def CalcDesiredFeatures(self, context, output):
        """Compute desired features based on current state."""
        mode = context.get_abstract_state(int(self._mode_index)).get_value()
        
        if mode == IBVSState.CENTER_ON_TARGET:
            # Center features in image (main IBVS task)
            desired_u = 320.0
            desired_v = 240.0
            
            # Set all features to same target (can be customized)
            for i in range(self.N_features):
                output[2*i] = desired_u + (i % 2) * 20  # Slight offset for multiple features
                output[2*i + 1] = desired_v + (i // 2) * 20
        else:
            # For other states, use current features (no change)
            # This will be set by the system using current features
            output.SetFromVector(np.zeros(2 * self.N_features))
    
    def CalcControlMode(self, context, output):
        """Output control mode for PortSwitch (future use)."""
        mode = context.get_abstract_state(int(self._mode_index)).get_value()
        
        if mode == IBVSState.GO_HOME:
            output.set_value(InputPortIndex(1))  # Position control
        else:
            output.set_value(InputPortIndex(0))  # IBVS control

