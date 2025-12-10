"""
IBVS State Machine

Dictionary-based state machine framework for hierarchical visual servoing control.
Each state has its own objective defined in ibvs_states.py.

Uses STATE_CONFIGS dictionary to define:
- Desired features for each state
- DOF mask (which degrees of freedom to control)
- Convergence criteria
- State transitions
"""

import numpy as np
from pydrake.all import AbstractValue, LeafSystem

from ibvs_states import (
    IBVSState,
    STATE_CONFIGS,
    get_state_config,
    get_initial_state,
)


class IBVSStateMachine(LeafSystem):
    """
    Dictionary-based state machine for IBVS control.
    
    Each state has its own objective defined in STATE_CONFIGS:
    - Desired features (where to move features in image)
    - DOF mask (which degrees of freedom to control)
    - Convergence criteria (when to transition)
    - Next state (where to go next)
    """
    
    def __init__(self, N_features=4):
        """
        Initialize IBVS state machine.
        
        Args:
            N_features: Number of feature points
        """
        LeafSystem.__init__(self)
        self.N_features = N_features
        
        # Input: current features (for error calculation)
        self.current_features_input = self.DeclareVectorInputPort(
            "current_features",
            size=2 * N_features
        )
        
        # Input: convergence flag from IBVS controller
        self.converged_input = self.DeclareVectorInputPort("converged", size=1)
        
        # State: current mode
        initial_state = get_initial_state()
        self._mode_index = self.DeclareAbstractState(
            AbstractValue.Make(initial_state)
        )
        # Track previous state name as an abstract state (used by some visualizers)
        self._previous_state_name_index = self.DeclareAbstractState(
            AbstractValue.Make(initial_state.value)
        )
        
        # Track convergence time for each state
        self._convergence_start_time_index = self.DeclareDiscreteState(1)  # Time when converged
        
        # Output: desired features (2N vector) - based on current state
        # These outputs only depend on state (read in calc), not inputs, so no direct feedthrough
        self.desired_features_output = self.DeclareVectorOutputPort(
            "desired_features",
            size=2 * N_features,
            calc=self._calc_desired_features
        )
        
        # Output: DOF mask (6D vector) - which DOFs to enable for current state
        self.dof_mask_output = self.DeclareVectorOutputPort(
            "dof_mask",
            size=6,
            calc=self._calc_dof_mask
        )
        
        # Output: state name (for debugging/visualization)
        self.state_name_output = self.DeclareAbstractOutputPort(
            "state_name",
            lambda: AbstractValue.Make(""),
            self._calc_state_name
        )
        
        # Periodic update for state transitions (every 0.1 seconds)
        self.DeclarePeriodicUnrestrictedUpdateEvent(0.1, 0.0, self._update_state)
        # Track last printed state locally to avoid repeated logs
        self._last_state_logged = None
    
    def _update_state(self, context, state):
        """Update state machine based on current conditions."""
        current_state = context.get_abstract_state(int(self._mode_index)).get_value()
        current_time = context.get_time()
        config = get_state_config(current_state)
        
        # Check if this is a new state (first entry)
        def _get_discrete_scalar(values, idx, default=-1.0):
            """Safely extract scalar from Drake discrete values (array or scalar)."""
            if len(values) <= idx:
                return default
            val = values[idx]
            if isinstance(val, np.ndarray):
                return float(val.flat[0]) if val.size > 0 else default
            try:
                return float(val)
            except Exception:
                return default

        def _get_convergence_start():
            discrete_state = state.get_discrete_state()
            discrete_values = discrete_state.value()
            return _get_discrete_scalar(discrete_values, int(self._convergence_start_time_index), default=-1.0)

        if self._last_state_logged != current_state.value:
            # New state entered - print entry message
            print(f"[StateMachine] >>> ENTERED STATE: {config.name} ({current_state.value})", flush=True)
            print(f"    Objective: {config.name}", flush=True)
            print(f"    DOF Mask: {config.dof_mask}", flush=True)
            print(f"    Error Threshold: {config.error_threshold} px", flush=True)
            self._last_state_logged = current_state.value
            # Update previous state tracker for downstream consumers
            state.get_mutable_abstract_state(int(self._previous_state_name_index)).set_value(current_state.value)
        
        # Check if we should transition
        should_transition = False
        
        # Get convergence status
        converged = self.converged_input.Eval(context)
        is_converged = converged[0] > 0.5
        
        # Get current features to check if we have valid features
        current_features = self.current_features_input.Eval(context)
        has_features = np.any(current_features != 0)
        
        # Get mutable discrete state once for all updates
        mutable_discrete_state = state.get_mutable_discrete_state()
        
        # State-specific transition logic
        if current_state == IBVSState.WAIT_FOR_FEATURES:
            if has_features:
                should_transition = True
                next_state = config.next_state
        
        elif current_state == IBVSState.APPROACH:
            if is_converged:
                convergence_start = _get_convergence_start()
                if convergence_start < 0:  # Not yet tracking convergence
                    mutable_discrete_state.set_value(int(self._convergence_start_time_index), np.array([current_time]))
                elif current_time - convergence_start >= config.convergence_time:
                    should_transition = True
                    next_state = config.next_state
            else:
                # Reset convergence timer if not converged
                mutable_discrete_state.set_value(int(self._convergence_start_time_index), np.array([-1.0]))
        
        elif current_state == IBVSState.CENTER_ON_TARGET:
            if is_converged:
                convergence_start = _get_convergence_start()
                if convergence_start < 0:
                    mutable_discrete_state.set_value(int(self._convergence_start_time_index), np.array([current_time]))
                elif current_time - convergence_start >= config.convergence_time:
                    should_transition = True
                    next_state = config.next_state
            else:
                mutable_discrete_state.set_value(int(self._convergence_start_time_index), np.array([-1.0]))
        
        elif current_state == IBVSState.DESCEND_TO_GRASP:
            if is_converged:
                convergence_start = _get_convergence_start()
                if convergence_start < 0:
                    mutable_discrete_state.set_value(int(self._convergence_start_time_index), np.array([current_time]))
                elif current_time - convergence_start >= config.convergence_time:
                    should_transition = True
                    next_state = config.next_state
            else:
                mutable_discrete_state.set_value(int(self._convergence_start_time_index), np.array([-1.0]))
        
        elif current_state == IBVSState.GRASP:
            # Time-based transition (gripper closing time)
            convergence_start = _get_convergence_start()
            if convergence_start < 0:
                mutable_discrete_state.set_value(int(self._convergence_start_time_index), np.array([current_time]))
            elif current_time - convergence_start >= config.convergence_time:
                should_transition = True
                next_state = config.next_state
        
        elif current_state == IBVSState.LIFT:
            if is_converged:
                convergence_start = _get_convergence_start()
                if convergence_start < 0:
                    mutable_discrete_state.set_value(int(self._convergence_start_time_index), np.array([current_time]))
                elif current_time - convergence_start >= config.convergence_time:
                    should_transition = True
                    next_state = config.next_state
            else:
                mutable_discrete_state.set_value(int(self._convergence_start_time_index), np.array([-1.0]))
        
        elif current_state == IBVSState.MOVE_TO_GOAL:
            if is_converged:
                convergence_start = _get_convergence_start()
                if convergence_start < 0:
                    mutable_discrete_state.set_value(int(self._convergence_start_time_index), np.array([current_time]))
                elif current_time - convergence_start >= config.convergence_time:
                    should_transition = True
                    next_state = config.next_state
            else:
                mutable_discrete_state.set_value(int(self._convergence_start_time_index), np.array([-1.0]))
        
        elif current_state == IBVSState.TASK_COMPLETE:
            # Terminal state - no transitions
            pass
        
        # Perform transition
        if should_transition and next_state is not None:
            next_config = get_state_config(next_state)
            print(f"[StateMachine] <<< TRANSITIONING FROM: {config.name} ({current_state.value})", flush=True)
            print(f"    To: {next_config.name} ({next_state.value})", flush=True)
            print(f"    Reason: {'Converged' if is_converged else 'Time-based' if current_state == IBVSState.GRASP else 'Features detected'}", flush=True)
            
            state.get_mutable_abstract_state(int(self._mode_index)).set_value(next_state)
            # Reset convergence timer
            mutable_discrete_state.set_value(int(self._convergence_start_time_index), np.array([-1.0]))
    
    def _calc_desired_features(self, context, output):
        """Compute desired features based on current state."""
        # IMPORTANT: Only read from state, NOT from inputs (to avoid direct feedthrough)
        # This function is called during output evaluation, so it must not read inputs
        current_state = context.get_abstract_state(int(self._mode_index)).get_value()
        config = get_state_config(current_state)
        
        # Call the desired features function for this state
        # Note: desired_features_fn takes context but doesn't use it - it's just for future extensibility
        desired = config.desired_features_fn(context, self.N_features)
        output.SetFromVector(desired)
    
    def _calc_dof_mask(self, context, output):
        """Output DOF mask for current state."""
        # IMPORTANT: Only read from state, NOT from inputs
        current_state = context.get_abstract_state(int(self._mode_index)).get_value()
        config = get_state_config(current_state)
        output.SetFromVector(config.dof_mask)
    
    def _calc_state_name(self, context, output):
        """Output state name for debugging."""
        # IMPORTANT: Only read from state, NOT from inputs
        current_state = context.get_abstract_state(int(self._mode_index)).get_value()
        config = get_state_config(current_state)
        output.set_value(config.name)
