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

import argparse
import os
import sys
from pathlib import Path

import manipulation
import numpy as np
from pydrake.all import (
    DiagramBuilder,
    JacobianWrtVariable,
    LeafSystem,
    MathematicalProgram,
    Simulator,
    Solve,
    VectorSystem,
)
from pydrake.all import AbstractValue, ImageDepth32F

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cameras import add_eye_in_hand_camera  # noqa: E402
from features import FeatureTracker  # noqa: E402
from ibvs_controller import IBVSController  # noqa: E402
from transforms import adjoint_transform, euler_rpy_to_rotation  # noqa: E402
from experiments_finalproject.cube_detection_meshcat import capture_and_detect  # noqa: E402

XY_BOUNDS = {
    "xmin": -0.2,
    "xmax": 0.8,
    "ymin": 0.6,
    "ymax": 1.4,
}
QP_DT = 0.02
VEL_LIMIT = 1.5
QP_REG_ALPHA = 1e-3


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


class DetectedFeaturesSource(VectorSystem):
    """
    System that outputs detected (u, v) feature coordinates.
    """

    def __init__(self, uv: list[float], N_features=4):
        super().__init__(input_size=0, output_size=2 * N_features, direct_feedthrough=False)
        self.N_features = N_features
        if len(uv) < 2 * N_features:
            # Repeat last provided coordinate if fewer than needed
            last_u = uv[-2] if len(uv) >= 2 else 320.0
            last_v = uv[-1] if len(uv) >= 1 else 240.0
            while len(uv) < 2 * N_features:
                uv.extend([last_u, last_v])
        self.uv = np.array(uv[: 2 * N_features], dtype=float)

    def DoCalcVectorOutput(self, context, unused, unused2, output):
        output[:] = self.uv


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
        twist_mag = np.linalg.norm(Vee_des)

        try:
            root_context = None
            if self.root_context_ref is not None and self.root_context_ref[0] is not None:
                root_context = self.root_context_ref[0]
            if root_context is None:
                print("[ROBOT_COMMAND] WARNING: No root context available. Outputting zeros.", flush=True)
                output.SetFromVector(np.zeros(self.plant.num_actuators()))
                return

            plant_context = self.plant.GetMyContextFromRoot(root_context)
            num_actuators = self.plant.num_actuators()

            # Spatial Jacobian (6xN) at EE origin, expressed in world
            J_spatial = self.plant.CalcJacobianSpatialVelocity(
                plant_context,
                JacobianWrtVariable.kQDot,
                self.ee_frame,
                np.zeros(3),
                self.plant.world_frame(),
                self.plant.world_frame(),
            )
            J_spatial = J_spatial[:, :num_actuators]

            # Translational Jacobian for xy constraints
            J_trans = self.plant.CalcJacobianTranslationalVelocity(
                plant_context,
                JacobianWrtVariable.kQDot,
                self.ee_frame,
                np.zeros(3),
                self.plant.world_frame(),
                self.plant.world_frame(),
            )
            J_trans = J_trans[:2, :num_actuators]  # x, y rows

            X_WE = self.plant.CalcRelativeTransform(
                plant_context, self.plant.world_frame(), self.ee_frame
            )
            p_WE = X_WE.translation()

            prog = MathematicalProgram()
            qdot = prog.NewContinuousVariables(num_actuators, "qdot")

            # Tracking cost: ||J*qdot - V_des||^2
            prog.AddQuadraticErrorCost(J_spatial, Vee_des, qdot)
            # Regularization
            prog.AddQuadraticCost(QP_REG_ALPHA * np.eye(num_actuators), np.zeros(num_actuators), qdot)

            # Joint velocity bounds
            prog.AddLinearConstraint(qdot >= -VEL_LIMIT)
            prog.AddLinearConstraint(qdot <= VEL_LIMIT)

            # EE xy bounds (linearized with small dt)
            dt = QP_DT
            x_row = J_trans[0, :]
            y_row = J_trans[1, :]
            prog.AddLinearConstraint(
                dt * x_row @ qdot,
                XY_BOUNDS["xmin"] - p_WE[0],
                XY_BOUNDS["xmax"] - p_WE[0],
            )
            prog.AddLinearConstraint(
                dt * y_row @ qdot,
                XY_BOUNDS["ymin"] - p_WE[1],
                XY_BOUNDS["ymax"] - p_WE[1],
            )

            result = Solve(prog)
            if not result.is_success():
                print("[ROBOT_COMMAND] QP solve failed, outputting zeros.", flush=True)
                output.SetFromVector(np.zeros(num_actuators))
                return

            qdot_sol = result.GetSolution(qdot)

            qdot_mag = np.linalg.norm(qdot_sol)
            if twist_mag < 1e-6 and qdot_mag > 1e-6 and not hasattr(self, "_nonzero_qdot_warned"):
                print(f"[JOINT_VEL_CONTROLLER] WARN: zero twist but non-zero qdot (mag={qdot_mag:.2e})")
                self._nonzero_qdot_warned = True

            print(f"[ROBOT_COMMAND] Sending qdot: mag={qdot_mag:.6f}, qdot={qdot_sol}", flush=True)
            output.SetFromVector(qdot_sol)
        except Exception as e:
            print(f"[ROBOT_COMMAND] ERROR in _calc_qdot: {e}", flush=True)
            output.SetFromVector(np.zeros(self.plant.num_actuators()))


def build_ibvs_diagram(detected_uv=None, detected_depth=None):
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
    # Enable x-y translation for camera/gripper motion.
    dof_mask = np.array([1, 1, 0, 0, 0, 0])
    
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
    
    # Desired features: use detection if provided, else default center targets
    if detected_uv is not None:
        desired_features = builder.AddSystem(
            DetectedFeaturesSource(list(detected_uv), N_features=N_features)
        )
    else:
        desired_features = builder.AddSystem(DesiredFeaturesSource(N_features=N_features))

    builder.Connect(
        desired_features.get_output_port(0),
        ibvs_controller.desired_uv_input
    )
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
            default_depth=float(detected_depth) if detected_depth is not None else 0.5,
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
    parser = argparse.ArgumentParser(description="Run IBVS demo with optional cube detection seeding.")
    parser.add_argument("--detect-once", action="store_true", help="Run cube detection once to seed desired features/depth.")
    parser.add_argument("--sim-time", type=float, default=5.0, help="Simulation duration in seconds.")
    parser.add_argument("--scene", type=str, default=None, help="Optional scene name for detection helper.")
    args = parser.parse_args()

    detected_uv = None
    detected_depth = None

    if args.detect_once:
        print("[Detect] Running cube detection to seed desired features...")
        result, meshcat_url = capture_and_detect(scene=args.scene)
        cx, cy = result["centroid"]
        detected_depth = result.get("depth_m", None)
        # Repeat centroid for four features
        detected_uv = [cx, cy, cx, cy, cx, cy, cx, cy]
        print(f"[Detect] centroid=({cx:.1f},{cy:.1f}), depth={detected_depth}, meshcat={meshcat_url}")

    print("Building IBVS diagram...")
    diagram, plant, ee_frame, root_context_ref = build_ibvs_diagram(
        detected_uv=detected_uv,
        detected_depth=detected_depth,
    )
    
    print("Creating simulator...")
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    
    # Set root context reference in joint_vel_controller
    root_context_ref[0] = context
    
    # Set initial robot pose
    plant_context = plant.GetMyContextFromRoot(context)
    
    num_pos = plant.num_positions()
    print(f"Plant has {num_pos} positions")
    
    # Initial joint positions (home position)
    q0_iiwa = np.array([-0.92, 0.08, -0.34, 0.76, -0.28, -1.5, 0])
    q0 = np.zeros(num_pos)
    q0[:7] = q0_iiwa
    plant.SetPositions(plant_context, q0)
    plant.SetVelocities(plant_context, np.zeros(plant.num_velocities()))
    
    run_duration = max(0.0, float(args.sim_time))
    print(f"Running simulation for {run_duration:.2f} seconds...")
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(run_duration)
    
    print("Simulation complete!")


if __name__ == "__main__":
    main()

