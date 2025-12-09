"""
Mapping Twist to Joint Velocities

Converts desired end-effector spatial velocity (twist) to joint velocities
using the manipulator Jacobian and pseudoinverse.

    q̇ = J⁺ · V_ee

where:
    - q̇: joint velocities
    - J: spatial velocity Jacobian
    - V_ee: desired end-effector spatial velocity
"""

import numpy as np
from pydrake.all import (
    Context,
    Frame,
    JacobianWrtVariable,
    MultibodyPlant,
    SpatialVelocity,
)


def solve_ik_qdot(plant, context, Vee_des, ee_frame=None, max_velocity=2.0, enforce_position_limits=True):
    """
    Solve for joint velocities given desired end-effector spatial velocity.
    
    Uses the spatial velocity Jacobian and pseudoinverse:
        q̇ = J⁺ · V_ee
    
    Applies joint velocity and position limit constraints.
    
    Args:
        plant: MultibodyPlant
        context: Context for the plant
        Vee_des: Desired end-effector spatial velocity (6D numpy array)
                 [vx, vy, vz, wx, wy, wz]
        ee_frame: End-effector frame (if None, uses iiwa_link_7)
        max_velocity: Maximum joint velocity in rad/s (default: 2.0)
        enforce_position_limits: If True, prevent motion towards position limits
    
    Returns:
        numpy array: Joint velocities q̇ (clamped to limits)
    """
    # Get end-effector frame
    if ee_frame is None:
        # Try to find IIWA end-effector
        try:
            ee_frame = plant.GetFrameByName("iiwa_link_7")
        except:
            raise ValueError("Could not find end-effector frame. Please specify ee_frame.")
    
    # Get world frame
    world_frame = plant.world_frame()
    
    # Compute spatial velocity Jacobian (full plant)
    J_v = plant.CalcJacobianSpatialVelocity(
        context,
        JacobianWrtVariable.kV,
        ee_frame,
        np.zeros(3),  # p_BoBp_B (point in frame, use origin)
        world_frame,
        world_frame
    )
    
    # Convert to numpy (J_v might already be a numpy array or a matrix)
    if hasattr(J_v, 'toarray'):
        J_v_np = J_v.toarray()  # If it's a sparse matrix or matrix object
    else:
        J_v_np = np.array(J_v)  # If it's already a numpy array or can be converted
    
    # Restrict Jacobian to actuated DOFs (order matches actuation input port)
    actuated_vel_indices = []
    actuated_pos_indices = []
    for actuator_index in plant.GetJointActuatorIndices():
        actuator = plant.get_joint_actuator(actuator_index)
        joint = actuator.joint()
        for k in range(joint.num_velocities()):
            actuated_vel_indices.append(joint.velocity_start() + k)
        for k in range(joint.num_positions()):
            actuated_pos_indices.append(joint.position_start() + k)
    
    J_act = J_v_np[:, actuated_vel_indices]
    
    # Compute pseudoinverse on actuated subset
    J_pinv = np.linalg.pinv(J_act)
    
    # Solve for joint velocities (primary task) for actuated DOFs only
    qdot_task = J_pinv @ Vee_des
    
    # Check if primary task is zero (or very small) - if so, disable null-space control
    # This prevents unwanted movement when IBVS is disabled (e.g., DOF mask all zeros)
    task_magnitude = np.linalg.norm(Vee_des)
    qdot_task_mag = np.linalg.norm(qdot_task)
    use_null_space = task_magnitude > 1e-6  # Only use null-space if task is active
    
    # Debug output
    if task_magnitude < 1e-6:
        if not hasattr(solve_ik_qdot, '_zero_task_warned'):
            print(f"[JACOBIANS] Primary task is zero (magnitude={task_magnitude:.2e}), disabling null-space control")
            print(f"[JACOBIANS] qdot_task magnitude: {qdot_task_mag:.6e}, qdot_task: {qdot_task}")
            solve_ik_qdot._zero_task_warned = True
        elif qdot_task_mag > 1e-6:
            # This shouldn't happen - if Vee_des is zero, qdot_task should be zero
            print(f"[JACOBIANS] ERROR: Zero Vee_des but non-zero qdot_task! qdot_task_mag={qdot_task_mag:.6e}, qdot_task={qdot_task}")
    
    qdot = qdot_task.copy()
    
    # Get actuated joint positions (used for limits and optional null-space)
    q_full = plant.GetPositions(context)
    q = q_full[actuated_pos_indices]
    
    # Add null-space component to avoid extreme configurations (only if task is active)
    if use_null_space and enforce_position_limits:
        # Null-space projector: (I - J⁺J)
        num_joints = J_act.shape[1]
        I = np.eye(num_joints)
        N = I - J_pinv @ J_act  # Null-space projector
        
        # Null-space velocity: bias joints towards center of their range
        num_iiwa_joints = min(7, len(q), num_joints)
        
        # Compute desired null-space velocity (bias towards center)
        qdot_null = np.zeros(num_joints)
        q_min = np.array([-2.967, -2.094, -2.967, -2.094, -2.967, -2.094, -3.054])
        q_max = np.array([2.967, 2.094, 2.967, 2.094, 2.967, 2.094, 3.054])
        
        for i in range(num_iiwa_joints):
            range_center = (q_min[i] + q_max[i]) / 2.0
            # Bias towards center, stronger if further from center
            error = range_center - q[i]
            # Use stronger bias to actively avoid extreme configurations
            qdot_null[i] = 0.5 * error  # Gain for null-space bias
        
        # Project null-space velocity into null-space
        qdot_null_projected = N @ qdot_null
        
        # Combine primary task and null-space
        qdot = qdot_task + qdot_null_projected
    
    # Apply joint limits
    qdot_before_limits = qdot.copy()
    qdot_before_mag = np.linalg.norm(qdot_before_limits)
    qdot = apply_joint_limits(
        plant,
        context,
        qdot,
        q_actuated=q,
        max_velocity=max_velocity,
        enforce_position_limits=enforce_position_limits,
    )
    qdot_after_mag = np.linalg.norm(qdot)
    
    # Debug: check if joint limits are modifying zero velocities
    if task_magnitude < 1e-6:
        if qdot_after_mag > 1e-6:
            if not hasattr(solve_ik_qdot, '_limits_adding_motion_warned'):
                print(f"[JACOBIANS] WARNING: apply_joint_limits is adding motion!")
                print(f"[JACOBIANS]   Before limits: mag={qdot_before_mag:.6e}, qdot={qdot_before_limits}")
                print(f"[JACOBIANS]   After limits:  mag={qdot_after_mag:.6e}, qdot={qdot}")
                solve_ik_qdot._limits_adding_motion_warned = True
        else:
            # Good - zero input, zero output
            if not hasattr(solve_ik_qdot, '_zero_verified'):
                print(f"[JACOBIANS] Verified: Zero input -> Zero output (qdot_mag={qdot_after_mag:.6e})")
                solve_ik_qdot._zero_verified = True
    
    return qdot


def apply_joint_limits(plant, context, qdot, q_actuated=None, max_velocity=2.0, enforce_position_limits=True):
    """
    Apply joint velocity and position limits to joint velocities.
    
    Args:
        plant: MultibodyPlant
        context: Context for the plant
        qdot: Unconstrained joint velocities
        max_velocity: Maximum joint velocity in rad/s
        enforce_position_limits: If True, prevent motion towards position limits
    
    Returns:
        numpy array: Constrained joint velocities
    """
    input_qdot_mag = np.linalg.norm(qdot)
    qdot_constrained = qdot.copy()
    
    # Debug: if input is zero, output should be zero
    if input_qdot_mag < 1e-6:
        if not hasattr(apply_joint_limits, '_zero_input_warned'):
            print(f"[APPLY_JOINT_LIMITS] Input qdot is zero (magnitude={input_qdot_mag:.6e})")
            apply_joint_limits._zero_input_warned = True
    
    # Build actuated position indices to align with qdot
    actuated_pos_indices = []
    for actuator_index in plant.GetJointActuatorIndices():
        actuator = plant.get_joint_actuator(actuator_index)
        joint = actuator.joint()
        for k in range(joint.num_positions()):
            actuated_pos_indices.append(joint.position_start() + k)
    
    # Get current joint positions for actuated joints
    if q_actuated is None:
        q_full = plant.GetPositions(context)
        q = q_full[actuated_pos_indices]
    else:
        q = q_actuated
    
    # Get number of actuated joints (e.g., iiwa + wsg)
    num_actuators = len(qdot)
    
    # Only apply limits to IIWA joints (first 7), not gripper
    num_iiwa_joints = min(7, num_actuators, len(q))
    
    # IIWA joint limits (from Kuka IIWA14 specifications)
    # Position limits in radians
    q_min = np.array([-2.967, -2.094, -2.967, -2.094, -2.967, -2.094, -3.054])
    q_max = np.array([2.967, 2.094, 2.967, 2.094, 2.967, 2.094, 3.054])
    
    # Velocity limits (rad/s) - IIWA can go up to ~2.0 rad/s
    qdot_max = np.full(num_actuators, max_velocity)
    
    # Apply velocity limits (clamp to max velocity)
    qdot_constrained = np.clip(qdot_constrained, -qdot_max, qdot_max)
    
    # Apply position limit constraints (prevent moving towards limits)
    if enforce_position_limits:
        # Use larger buffer zone (0.3 rad = ~17 degrees) to prevent awkward configurations
        buffer_zone = 0.3
        
        for i in range(min(num_iiwa_joints, len(q), num_actuators)):
            # Check if approaching lower limit
            if q[i] <= q_min[i] + buffer_zone:
                # Scale down velocity as we approach limit, or reverse if too close
                if q[i] <= q_min[i] + 0.1:
                    qdot_constrained[i] = max(0, qdot_constrained[i])  # Hard stop
                else:
                    # Gradually reduce velocity as approaching limit
                    scale = (q[i] - q_min[i]) / buffer_zone
                    if qdot_constrained[i] < 0:  # Moving towards limit
                        qdot_constrained[i] *= scale
            
            # Check if approaching upper limit
            if q[i] >= q_max[i] - buffer_zone:
                # Scale down velocity as we approach limit, or reverse if too close
                if q[i] >= q_max[i] - 0.1:
                    qdot_constrained[i] = min(0, qdot_constrained[i])  # Hard stop
                else:
                    # Gradually reduce velocity as approaching limit
                    scale = (q_max[i] - q[i]) / buffer_zone
                    if qdot_constrained[i] > 0:  # Moving towards limit
                        qdot_constrained[i] *= scale
            
            # Additional: Penalize extreme configurations (bias towards center of range)
            # This helps avoid awkward overlapping configurations
            # BUT: Only apply bias if there's actual motion (don't add motion when qdot is zero)
            input_joint_qdot = qdot[i]  # Original input velocity for this joint
            if abs(input_joint_qdot) > 1e-6:  # Only bias if there's actual motion
                range_center = (q_min[i] + q_max[i]) / 2.0
                range_size = q_max[i] - q_min[i]
                
                # If joint is far from center, add small bias towards center
                if abs(q[i] - range_center) > range_size * 0.3:  # More than 30% from center
                    bias_strength = 0.1  # Small bias
                    bias_direction = np.sign(range_center - q[i])  # Direction towards center
                    qdot_constrained[i] += bias_strength * bias_direction
                    if input_qdot_mag < 1e-6:
                        print(f"[APPLY_JOINT_LIMITS] WARNING: Adding bias to joint {i} even though input is zero!")
    
    output_qdot_mag = np.linalg.norm(qdot_constrained)
    if input_qdot_mag < 1e-6 and output_qdot_mag > 1e-6:
        if not hasattr(apply_joint_limits, '_zero_to_nonzero_warned'):
            print(f"[APPLY_JOINT_LIMITS] ERROR: Zero input but non-zero output!")
            print(f"[APPLY_JOINT_LIMITS]   Input:  mag={input_qdot_mag:.6e}, qdot={qdot}")
            print(f"[APPLY_JOINT_LIMITS]   Output: mag={output_qdot_mag:.6e}, qdot={qdot_constrained}")
            apply_joint_limits._zero_to_nonzero_warned = True
    
    return qdot_constrained


def get_end_effector_jacobian(plant, context, ee_frame=None):
    """
    Get the spatial velocity Jacobian for the end-effector.
    
    Args:
        plant: MultibodyPlant
        context: Context for the plant
        ee_frame: End-effector frame (if None, uses iiwa_link_7)
    
    Returns:
        numpy array: (6, nq) Jacobian matrix
    """
    # Get end-effector frame
    if ee_frame is None:
        try:
            ee_frame = plant.GetFrameByName("iiwa_link_7")
        except:
            raise ValueError("Could not find end-effector frame. Please specify ee_frame.")
    
    # Get world frame
    world_frame = plant.world_frame()
    
    # Compute spatial velocity Jacobian
    J_v = plant.CalcJacobianSpatialVelocity(
        context,
        JacobianWrtVariable.kV,
        ee_frame,
        np.zeros(3),  # p_BoBp_B (point in frame, use origin)
        world_frame,
        world_frame
    )
    
    return J_v.toarray()

