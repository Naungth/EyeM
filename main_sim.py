"""
Main Simulation Integration

Wires together the complete IBVS control loop:
    Camera → FeatureTracker → IBVS → Vee_des → q̇ → plant

Simulates for 5 seconds with a stationary cube and constant target features.

HIERARCHICAL STATE MACHINE (future):
States such as:
    APPROACH_PREGRASP → CENTER_ON_TARGET → DESCEND_TO_GRASP → 
    GRASP → LIFT → MOVE_TO_GOAL

Each state sets a different u_desired or feature set.
For now, just scaffold the interface.
"""

import os
import manipulation
import numpy as np
from pydrake.all import (
    DiagramBuilder,
    LeafSystem,
    Simulator,
    VectorSystem,
)

from cameras import add_eye_in_hand_camera
from features import FeatureTracker
from ibvs_controller import IBVSController
from jacobians import solve_ik_qdot
from state_machine import IBVSStateMachine, IBVSState
from transforms import adjoint_transform, euler_rpy_to_rotation

from pydrake.all import AbstractValue, ImageDepth32F, PortSwitch


class DesiredFeaturesSource(VectorSystem):
    """
    System that provides desired feature coordinates.
    
    For now, provides constant desired features.
    Later, this will be replaced by a state machine that sets
    different targets based on the current state.
    """
    
    def __init__(self, N_features=4):
        """
        Initialize desired features source.
        
        Args:
            N_features: Number of feature points
        """
        super().__init__(input_size=0, output_size=2 * N_features, direct_feedthrough=False)  # No inputs, 2N outputs
        self.N_features = N_features
    
    def DoCalcVectorOutput(self, context, unused, unused2, output):
        """Output constant desired feature coordinates."""
        # Set desired features (e.g., center of image for each feature)
        # For a 640x480 image, center is (320, 240)
        desired_u = 320.0
        desired_v = 240.0
        
        # Set all features to same target (can be customized)
        for i in range(self.N_features):
            output[2*i] = desired_u + (i % 2) * 20  # Slight offset for multiple features
            output[2*i + 1] = desired_v + (i // 2) * 20


class DepthEstimator(LeafSystem):
    """
    System that estimates depth for each feature point by sampling the depth image.
    
    Extracts depth values from the depth image at the feature pixel coordinates.
    Uses Exponential Moving Average (EMA) filtering for smoother depth estimates.
    """
    
    def __init__(self, N_features=4, image_width=640, image_height=480, default_depth=0.5,
                 depth_ema_alpha=0.6, depth_min_m=0.15, depth_max_m=2.0):
        """
        Initialize depth estimator.
        
        Args:
            N_features: Number of feature points
            image_width: Width of depth image in pixels
            image_height: Height of depth image in pixels
            default_depth: Default depth estimate in meters (used if feature is out of bounds or invalid)
            depth_ema_alpha: EMA smoothing factor (0-1, higher = more responsive, lower = smoother)
            depth_min_m: Minimum valid depth in meters
            depth_max_m: Maximum valid depth in meters
        """
        LeafSystem.__init__(self)
        self.N_features = N_features
        self.image_width = image_width
        self.image_height = image_height
        self.default_depth = default_depth
        self.depth_ema_alpha = depth_ema_alpha
        self.depth_min_m = depth_min_m
        self.depth_max_m = depth_max_m
        
        # Track EMA depth for each feature
        self.ema_depths = None
        
        # Input: depth image (ImageDepth32F from RgbdSensor)
        self.depth_input = self.DeclareAbstractInputPort(
            "depth_image",
            AbstractValue.Make(ImageDepth32F())
        )
        
        # Input: feature locations (2N vector: [u1, v1, u2, v2, ...])
        self.features_input = self.DeclareVectorInputPort("features", size=2 * N_features)
        
        # Output: depth estimates for each feature (N values)
        self.depth_output = self.DeclareVectorOutputPort(
            "depth_estimates",
            size=N_features,
            calc=self._calc_depth
        )
    
    def _calc_depth(self, context, output):
        """Extract depth at feature locations from depth image."""
        # Get depth image (ImageDepth32F)
        depth_image_obj = self.depth_input.Eval(context)
        
        # Convert to numpy array
        if depth_image_obj.size() == 0:
            # No depth data, use default
            output.SetFromVector(np.full(self.N_features, self.default_depth))
            return
        
        # Get actual image dimensions
        img_height = depth_image_obj.height()
        img_width = depth_image_obj.width()
        
        depth_array = np.array(depth_image_obj.data).reshape(
            img_height, img_width
        )
        depth_image = depth_array
        
        # Get feature locations
        features = self.features_input.Eval(context)
        
        # Extract depth for each feature with EMA filtering
        depth_estimates = np.zeros(self.N_features)
        
        # Initialize EMA depths if needed
        if self.ema_depths is None:
            self.ema_depths = np.full(self.N_features, self.default_depth)
        
        for i in range(self.N_features):
            u = int(features[2 * i])      # u coordinate
            v = int(features[2 * i + 1])  # v coordinate
            
            # Check if feature is valid and within image bounds
            if u > 0 and v > 0 and u < img_width and v < img_height:
                # Sample depth at feature location (use nearest neighbor)
                depth_value = depth_image[v, u]
                
                # Check if depth is valid (not NaN, not zero, within reasonable range)
                if np.isfinite(depth_value) and depth_value > 0.01 and depth_value < 10.0:
                    # Clamp to valid range
                    depth_value = np.clip(depth_value, self.depth_min_m, self.depth_max_m)
                    
                    # Apply EMA filtering for smooth depth estimates
                    if self.ema_depths[i] is None or not np.isfinite(self.ema_depths[i]):
                        self.ema_depths[i] = depth_value
                    else:
                        self.ema_depths[i] = (
                            self.depth_ema_alpha * depth_value +
                            (1.0 - self.depth_ema_alpha) * self.ema_depths[i]
                        )
                    
                    depth_estimates[i] = self.ema_depths[i]
                else:
                    # Invalid depth, use previous EMA value or default
                    if self.ema_depths[i] is not None and np.isfinite(self.ema_depths[i]):
                        depth_estimates[i] = self.ema_depths[i]
                    else:
                        depth_estimates[i] = self.default_depth
                        self.ema_depths[i] = self.default_depth
            else:
                # Out of bounds, use previous EMA value or default
                if self.ema_depths[i] is not None and np.isfinite(self.ema_depths[i]):
                    depth_estimates[i] = self.ema_depths[i]
                else:
                    depth_estimates[i] = self.default_depth
                    self.ema_depths[i] = self.default_depth
        
        output.SetFromVector(depth_estimates)


class JointVelocityController(LeafSystem):
    """
    System that converts desired spatial velocity to joint velocity commands.
    
    This wraps the jacobians.solve_ik_qdot function and applies it in the
    control loop.
    """
    
    def __init__(self, plant, ee_frame, diagram=None, root_context_ref=None):
        """
        Initialize joint velocity controller.
        
        Args:
            plant: MultibodyPlant
            ee_frame: End-effector frame
            diagram: Optional Diagram reference (for context access)
            root_context_ref: Optional mutable reference to root context (will be set by simulator)
        """
        LeafSystem.__init__(self)
        self.plant = plant
        self.ee_frame = ee_frame
        self.diagram = diagram  # Store diagram reference for context access
        self.root_context_ref = root_context_ref  # Mutable reference to root context
        
        # Declare input port for desired twist (6D)
        self.twist_input = self.DeclareVectorInputPort("twist", size=6)
        
        # Declare output port for joint velocities
        self.qdot_output = self.DeclareVectorOutputPort(
            "qdot",
            size=plant.num_actuators(),
            calc=self._calc_qdot
        )
    
    def _calc_qdot(self, context, output):
        """
        Compute joint velocities from desired twist.
        
        Improved context management following bin.py patterns:
        - Properly accesses root context through the diagram
        - Better error handling
        """
        # CONSTANT PRINT: Entry point (with flush to ensure immediate output)
        import sys
        print(f"[ROBOT_COMMAND] _calc_qdot called at time={context.get_time():.4f}", flush=True)
        sys.stdout.flush()
        
        # Get desired twist from input
        Vee_des = self.twist_input.Eval(context)
        
        # Debug: check if we're getting zero twist
        twist_mag = np.linalg.norm(Vee_des)
        if twist_mag < 1e-6:
            if not hasattr(self, '_zero_twist_warned'):
                print(f"[JOINT_VEL_CONTROLLER] Received zero twist (magnitude={twist_mag:.2e})")
                self._zero_twist_warned = True
        
        # Get plant context from root context
        # In Drake, GetMyContextFromRoot requires a root (Diagram) context, not a LeafContext.
        # Solution: Use stored root context reference if available, otherwise try to get it from context tree.
        try:
            root_context = None
            
            # Method 1: Use stored root context reference (set by simulator)
            if self.root_context_ref is not None and self.root_context_ref[0] is not None:
                root_context = self.root_context_ref[0]
            
            # Method 2: Try to get root context from LeafContext's parent
            if root_context is None:
                # In Drake, LeafContext is a child of DiagramContext.
                # We can try to access the parent context through the context's internal structure.
                # However, LeafContext doesn't expose parent directly in pydrake.
                # 
                # Alternative: Use GetParentDiagram to get diagram, but we still need its context.
                # The context passed to calc is a LeafContext, which is part of a DiagramContext.
                # We need to get the DiagramContext (root) from the LeafContext.
                #
                # WORKAROUND: For now, we'll use a stored reference set by the simulator.
                # This is set in main() after creating the simulator.
                print(f"[ROBOT_COMMAND] WARNING: No root context available. Outputting zeros.", flush=True)
                output.SetFromVector(np.zeros(self.plant.num_actuators()))
                return
            
            # Now we have root context - get plant context
            plant_context = self.plant.GetMyContextFromRoot(root_context)
            
            # Solve for joint velocities with joint limits enforced
            qdot_all = solve_ik_qdot(
                self.plant, 
                plant_context, 
                Vee_des, 
                self.ee_frame,
                max_velocity=1.5,  # Maximum joint velocity in rad/s
                enforce_position_limits=True
            )
            
            # Only use IIWA joint velocities (first 7 joints)
            # Set gripper joint velocities to zero to prevent gripper from moving/disconnecting
            num_actuators = self.plant.num_actuators()
            qdot = np.zeros(num_actuators)
            num_iiwa_joints = min(7, len(qdot_all), num_actuators)
            qdot[:num_iiwa_joints] = qdot_all[:num_iiwa_joints]
            # Gripper joints (if any) remain at zero
            
            # Debug: check if we're outputting non-zero velocities when twist is zero
            qdot_mag = np.linalg.norm(qdot)
            if twist_mag < 1e-6:
                if qdot_mag > 1e-6:
                    if not hasattr(self, '_nonzero_qdot_warned'):
                        print(f"[JOINT_VEL_CONTROLLER] ERROR: Zero twist but non-zero qdot!")
                        print(f"[JOINT_VEL_CONTROLLER]   Twist: mag={twist_mag:.6e}, Vee_des={Vee_des}")
                        print(f"[JOINT_VEL_CONTROLLER]   Qdot:  mag={qdot_mag:.6e}, qdot={qdot}")
                        self._nonzero_qdot_warned = True
                else:
                    # Good - zero twist, zero qdot
                    if not hasattr(self, '_zero_verified'):
                        print(f"[JOINT_VEL_CONTROLLER] Verified: Zero twist -> Zero qdot (qdot_mag={qdot_mag:.6e})")
                        self._zero_verified = True
            
            # CONSTANT PRINT: Always print before sending command to robot (with flush)
            import sys
            print(f"[ROBOT_COMMAND] Sending qdot: mag={qdot_mag:.6f}, qdot={qdot}", flush=True)
            sys.stdout.flush()
            
            output.SetFromVector(qdot)
        except Exception as e:
            # If computation fails, output zero velocities
            # This can happen during initialization or if context access fails
            import sys
            print(f"[ROBOT_COMMAND] ERROR in _calc_qdot: {e}", flush=True)
            sys.stdout.flush()
            output.SetFromVector(np.zeros(self.plant.num_actuators()))


def build_ibvs_diagram():
    """
    Build the complete IBVS control diagram.
    
    Returns:
        tuple: (diagram, plant, ee_frame)
    """
    builder = DiagramBuilder()
    
    # Build the station environment (plant, scene_graph, table, cube)
    from pydrake.all import AddMultibodyPlantSceneGraph, Box, GeometryInstance
    from pydrake.all import MakePhongIllustrationProperties, Parser, RigidTransform, Sphere
    from pydrake.all import SceneGraph, SpatialInertia, UnitInertia
    
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant, scene_graph)
    

    manipulation_models_path = os.path.join(os.path.dirname(manipulation.__file__), "models")
    parser.package_map().Add("manipulation", manipulation_models_path)
    
    # Add IIWA
    iiwa_model = parser.AddModelsFromUrl(
        "package://drake_models/iiwa_description/sdf/iiwa14_no_collision.sdf"
    )[0]
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("iiwa_link_0", iiwa_model)
    )
    
    # Add Schunk WSG gripper (matching stairs_scene_0.yaml)
    wsg_model = parser.AddModelsFromUrl(
        "package://manipulation/hydro/schunk_wsg_50_with_tip.sdf"
    )[0]
    iiwa_ee_frame = plant.GetFrameByName("iiwa_link_7", iiwa_model)
    wsg_base_frame = plant.GetFrameByName("body", wsg_model)
    # Match the transform from stairs_scene_0.yaml: translation [0, 0, 0.114] and rotation [90, 0, 90] degrees
    from pydrake.all import RollPitchYaw
    rpy = RollPitchYaw(np.deg2rad([90.0, 0.0, 90.0]))  # Roll, Pitch, Yaw in degrees
    X_WSG_EE = RigidTransform(rpy.ToRotationMatrix(), p=[0, 0, 0.114])
    plant.WeldFrames(iiwa_ee_frame, wsg_base_frame, X_WSG_EE)
    
    # Toggle camera marker visualization in Meshcat
    show_camera_marker = True
    if show_camera_marker:
        # Must match the mount in cameras.py: 20cm forward, 0cm lateral, 10cm up, 0 deg pitch (straight forward)
        camera_rpy = RollPitchYaw(0, 0, 0)
        camera_p = np.array([0.20, 0.0, 0.10]).reshape(3, 1)
        X_camera_EE = RigidTransform(camera_rpy, camera_p)
        camera_marker_color = np.array([0.1, 0.8, 1.0, 0.9])
        # Use diffuse color directly (RegisterVisualGeometry expects 4D color, not IllustrationProperties)
        plant.RegisterVisualGeometry(
            iiwa_ee_frame.body(),  # Register on the end-effector body
            X_camera_EE,
            Sphere(0.015),
            "camera_marker",
            camera_marker_color,
        )

    # Create a model instance for environment objects (cube)
    env_model_instance = plant.AddModelInstance("Environment")
    
    # Add cube
    cube_size = 0.05
    cube_mass = 0.1  # 100g cube
    cube_inertia = SpatialInertia(
        mass=cube_mass,
        p_PScm_E=np.zeros(3),  # Center of mass at origin
        G_SP_E=UnitInertia.SolidBox(cube_size, cube_size, cube_size)
    )
    cube_body = plant.AddRigidBody(
        "cube",
        env_model_instance,
        cube_inertia,
    )
    # Cube geometry (position relative to body frame, which will be at origin)
    cube_geometry = GeometryInstance(
        RigidTransform(),  # Geometry at body origin
        Box(cube_size, cube_size, cube_size),
        "cube_visual"
    )
    cube_geometry.set_illustration_properties(
        MakePhongIllustrationProperties(np.array([0.8, 0.2, 0.2, 1.0]))
    )
    plant.RegisterVisualGeometry(cube_body, cube_geometry)
    
    # Weld cube to world at desired position
    cube_position = np.array([0, 1, cube_size / 2]).reshape(3, 1)  # On the ground
    cube_pose = RigidTransform(cube_position)
    plant.WeldFrames(plant.world_frame(), cube_body.body_frame(), cube_pose)
    
    # Disable gravity for testing (robot should stay still when qdot=0)
    # TODO: Replace with proper velocity-to-force controller later
    # Get the gravity field and set it to zero
    plant.mutable_gravity_field().set_gravity_vector(np.array([0, 0, 0]))
    
    plant.Finalize()
    ee_frame = plant.GetFrameByName("iiwa_link_7", iiwa_model)
    
    # Add camera
    rgbd_sensor, image_converter = add_eye_in_hand_camera(
        builder, plant, scene_graph, ee_frame
    )
    
    # Add feature tracker
    N_features = 4
    feature_tracker = builder.AddSystem(FeatureTracker(N_features=N_features))
    
    # Connect camera RGB output to feature tracker
    builder.Connect(
        rgbd_sensor.color_image_output_port(),
        feature_tracker.image_input
    )
    
    # Add camera feature visualizer (overlays blue dots on features)
    from visualization import CameraFeatureVisualizer
    camera_visualizer = builder.AddSystem(
        CameraFeatureVisualizer(
            N_features=N_features,
            save_dir="camera_screenshots",  # Directory to save screenshots
            save_interval=0.25  # Save every 0.25 seconds
        )
    )
    
    # Connect camera image and features to visualizer
    builder.Connect(
        rgbd_sensor.color_image_output_port(),
        camera_visualizer.image_input
    )
    builder.Connect(
        feature_tracker.features_output,
        camera_visualizer.features_input
    )
    
    # Connect desired features to visualizer (for error visualization)
    # This will be connected after desired_features is created below
    
    # Add IBVS controller with DLS and DOF mask (following real-world patterns)
    # DOF mask: [vx, vy, vz, wx, wy, wz] - 1.0 to enable, 0.0 to disable
    # Example: [1,1,1,0,0,0] = only translation, no rotation
    # NOTE: Setting all to 0 should disable all movement (for testing)
    dof_mask = np.array([1, 0, 0, 0, 0, 0])  # All DOFs enabled
    
    ibvs_controller = builder.AddSystem(
        IBVSController(
            N_features=N_features, 
            lambda_gain=1000.0,
            error_threshold=10.0,  # Stop when feature error < 10 pixels (RMS)
            convergence_window=10,  # Require 10 consecutive frames below threshold
            dls_lambda=0.01,  # DLS regularization (higher = more damping, more robust)
            dof_mask=dof_mask  # Control which DOFs are active
        )
    )
    
    # Connect feature tracker to IBVS controller
    builder.Connect(
        feature_tracker.features_output,
        ibvs_controller.current_uv_input
    )
    
    # Add state machine (for future hierarchical control)
    # For now, we can use either the simple DesiredFeaturesSource or the state machine
    use_state_machine = False  # Set to True to enable state machine
    
    if use_state_machine:
        state_machine = builder.AddSystem(IBVSStateMachine(N_features=N_features))
        # Connect feature error to state machine
        builder.Connect(
            feature_tracker.features_output,
            state_machine.error_input
        )
        # Connect convergence flag
        builder.Connect(
            ibvs_controller.converged_output,
            state_machine.converged_input
        )
        # Connect desired features from state machine
        builder.Connect(
            state_machine.desired_features_output,
            ibvs_controller.desired_uv_input
        )
        
        # Connect desired features to visualizer for error visualization
        builder.Connect(
            state_machine.desired_features_output,
            camera_visualizer.desired_features_input
        )
    else:
        # Use simple desired features source (current implementation)
        desired_features = builder.AddSystem(DesiredFeaturesSource(N_features=N_features))
        builder.Connect(
            desired_features.get_output_port(0),
            ibvs_controller.desired_uv_input
        )
        
        # Connect desired features to visualizer for error visualization
        builder.Connect(
            desired_features.get_output_port(0),
            camera_visualizer.desired_features_input
        )
    
    # Add depth estimator with EMA filtering (following real-world patterns)
    depth_estimator = builder.AddSystem(
        DepthEstimator(
            N_features=N_features, 
            image_width=640, 
            image_height=480,
            depth_ema_alpha=0.6,  # EMA smoothing (0-1, higher = more responsive)
            depth_min_m=0.01,     # Minimum valid depth (matches DepthRange)
            depth_max_m=2.0       # Maximum valid depth
        )
    )
    
    # Connect depth image from camera to depth estimator
    builder.Connect(
        rgbd_sensor.depth_image_32F_output_port(),
        depth_estimator.depth_input
    )
    
    # Connect feature locations to depth estimator (so it knows where to sample)
    builder.Connect(
        feature_tracker.features_output,
        depth_estimator.features_input
    )
    
    # Connect depth estimates to IBVS controller
    builder.Connect(
        depth_estimator.depth_output,
        ibvs_controller.depth_input
    )
    

    use_eye_in_hand_transform = True  # Set to False to disable transform
    eih_rx_deg = 0.0    # Roll in degrees (matches cameras.py)
    eih_ry_deg = 0.0    # Pitch in degrees (matches cameras.py: 0° = straight forward)
    eih_rz_deg = 0.0    # Yaw in degrees (matches cameras.py)
    eih_tx = 0.20       # Translation x (meters) - 20cm forward (matches cameras.py)
    eih_ty = 0.0        # Translation y (meters) - 0cm lateral (matches cameras.py)
    eih_tz = 0.10       # Translation z (meters) - 10cm up (matches cameras.py)
    
    if use_eye_in_hand_transform:
        # Create eye-in-hand adjoint transform
        # NOTE: X_Camera_EE in cameras.py is the transform from EE to Camera
        # We need the inverse (Camera to EE) for the adjoint transform
        from transforms import euler_rpy_to_rotation
        # Rotation from EE to Camera (as defined in cameras.py)
        R_ee_cam = euler_rpy_to_rotation(
            np.deg2rad(eih_rx_deg),
            np.deg2rad(eih_ry_deg),
            np.deg2rad(eih_rz_deg)
        )
        t_ee_cam = np.array([eih_tx, eih_ty, eih_tz], dtype=float)
        
        # Inverse transform: from Camera to EE
        R_cam_ee = R_ee_cam.T  # Inverse rotation
        t_cam_ee = -R_ee_cam.T @ t_ee_cam  # Inverse translation
        
        # Adjoint transform: transforms twist from Camera to EE
        # adjoint_transform(R, t) where R and t are from source to target
        Ad_cam_ee = adjoint_transform(R_cam_ee, t_cam_ee)
        
        # Create a system to apply the adjoint transform
        class EyeInHandTransform(LeafSystem):
            def __init__(self, Ad_cam_ee):
                LeafSystem.__init__(self)
                self.Ad_cam_ee = Ad_cam_ee
                self.twist_input = self.DeclareVectorInputPort("twist_camera", size=6)
                self.twist_output = self.DeclareVectorOutputPort(
                    "twist_ee", size=6, calc=self._transform
                )
            
            def _transform(self, context, output):
                v_cam = self.twist_input.Eval(context)
                v_ee = self.Ad_cam_ee @ v_cam
                output.SetFromVector(v_ee)
        
        eih_transform = builder.AddSystem(EyeInHandTransform(Ad_cam_ee))
        builder.Connect(ibvs_controller.velocity_output, eih_transform.twist_input)
        twist_source = eih_transform.twist_output
    else:
        # No transform, use camera twist directly
        twist_source = ibvs_controller.velocity_output
    
    # Add joint velocity controller
    # We'll set root context reference after simulator is created
    # Use a mutable list to store root context reference
    root_context_ref = [None]  # Mutable reference to root context
    
    joint_vel_controller = builder.AddSystem(
        JointVelocityController(plant, ee_frame, diagram=None, root_context_ref=root_context_ref)
    )
    builder.Connect(
        twist_source,
        joint_vel_controller.twist_input
    )
    
    # Connect joint velocities to plant
    builder.Connect(
        joint_vel_controller.qdot_output,
        plant.get_actuation_input_port()
    )
    
    # Add visualizer
    from pydrake.all import MeshcatVisualizer, StartMeshcat
    meshcat = StartMeshcat()
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    
    diagram = builder.Build()
    
    # Set diagram reference in joint_vel_controller for context access
    joint_vel_controller.diagram = diagram
    
    # Return root_context_ref so it can be set in main() after simulator is created
    return diagram, plant, ee_frame, root_context_ref


def main():
    """Run the IBVS simulation."""
    print("Building IBVS diagram...")
    diagram, plant, ee_frame, root_context_ref = build_ibvs_diagram()
    
    print("Creating simulator...")
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    
    # Set root context reference in joint_vel_controller
    # This allows the controller to access the plant context
    root_context_ref[0] = context
    
    # Set initial robot pose
    plant_context = plant.GetMyContextFromRoot(context)
    
    # Check actual number of positions
    num_pos = plant.num_positions()
    print(f"Plant has {num_pos} positions")
    
    # Initial joint positions (home position)
    # Start with IIWA positions
    q0_iiwa = np.array([-0.92, 0.08, -0.34, 0.76, -0.28, -1.5, 0])
    
    # Pad with zeros for gripper and any oßther positions
    q0 = np.zeros(num_pos)
    q0[:7] = q0_iiwa  # Set IIWA positions
    # Remaining positions (gripper, etc.) stay at zero
    # This ensures gripper joints (if any) are initialized to zero
    
    plant.SetPositions(plant_context, q0)
    
    # Explicitly set gripper joint velocities to zero to prevent detachment
    # (Gripper base is welded, but finger joints might exist and need to be controlled)
    qdot0 = np.zeros(plant.num_velocities())
    plant.SetVelocities(plant_context, qdot0)
    
    # Set initial velocities to zero
    plant.SetVelocities(plant_context, np.zeros(plant.num_velocities()))
    
    print("Running simulation for 5 seconds...")
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(5.0)
    
    print("Simulation complete!")


if __name__ == "__main__":
    main()

