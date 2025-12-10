"""
Main Simulation Integration with cube detection seeding and xy-bounded IBVS.
"""

import argparse
import os
import sys
from pathlib import Path

import manipulation
import numpy as np
from manipulation.station import ApplyMultibodyPlantConfig, LoadScenario
from pydrake.all import (
    AbstractValue,
    DiagramBuilder,
    ImageDepth32F,
    JacobianWrtVariable,
    AddMultibodyPlantSceneGraph,
    LeafSystem,
    Simulator,
    VectorSystem,
    RotationMatrix,
    RigidTransform,
    RollPitchYaw,
    Solve,
    ZeroOrderHold,
    Adder,
    PublishEvent,
)
from pydrake.geometry import Box, Rgba
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.multibody.parsing import ModelDirectives, Parser, ProcessModelDirectives

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cameras import add_eye_in_hand_camera  # noqa: E402
from features import FeatureTracker  # noqa: E402
from ibvs_controller import IBVSController  # noqa: E402
from state_machine import IBVSStateMachine  # noqa: E402
from transforms import adjoint_transform, euler_rpy_to_rotation  # noqa: E402
from experiments_finalproject.cube_detection_meshcat import capture_and_detect  # noqa: E402

SCENARIO_DIR = ROOT / "src" / "experiments_finalproject" / "generated_scenarios"
XY_BOUNDS = {"xmin": -2.0, "xmax": 2.0, "ymin": -0.5, "ymax": 2.0}
QP_DT = 0.02
VEL_LIMIT = 1.5
QP_REG_ALPHA = 1e-3


def _ensure_scenario_dir() -> None:
    if not SCENARIO_DIR.exists():
        raise FileNotFoundError(f"Scenario directory '{SCENARIO_DIR}' does not exist")


def _available_scenes() -> list[str]:
    _ensure_scenario_dir()
    return sorted(p.name for p in SCENARIO_DIR.glob("*.yaml"))


def _resolve_scene(name: str | None) -> Path:
    available = _available_scenes()
    if not available:
        raise RuntimeError(f"No scenario YAML files found in '{SCENARIO_DIR}'")
    scene_name = name or available[0]
    scene_path = SCENARIO_DIR / scene_name
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene '{scene_path}' not found. Available: {available}")
    return scene_path


def _set_hover_start_over_cube(plant, plant_context, ee_frame, cube_xy=(0.5, 0.3), hover_z=0.35) -> bool:
    """
    Solve a small IK to place the end-effector above the cube before starting.
    Returns True on success; leaves positions unchanged on failure.
    """
    target_p = np.array([cube_xy[0], cube_xy[1], hover_z])
    # Point EE -Z down; yaw 90° so x-axis points along +Y (roughly aligns with table frame).
    R_W = RollPitchYaw(np.pi, 0.0, np.deg2rad(90.0)).ToRotationMatrix()

    ik = InverseKinematics(plant, plant_context)
    ik.AddPositionConstraint(
        frameB=ee_frame,
        p_BQ=np.zeros(3),
        frameA=plant.world_frame(),
        p_AQ_lower=target_p - np.array([0.01, 0.01, 0.01]),
        p_AQ_upper=target_p + np.array([0.01, 0.01, 0.01]),
    )
    ik.AddOrientationConstraint(
        plant.world_frame(),  # frameAbar
        R_W,                  # R_AbarA
        ee_frame,             # frameBbar
        RotationMatrix(),     # R_BbarB (identity)
        theta_bound=0.15,
    )

    prog = ik.prog()
    prog.SetInitialGuess(ik.q(), plant.GetPositions(plant_context))
    result = Solve(prog)
    if not result.is_success():
        print("[InitPose] IK failed; keeping scenario defaults.")
        return False
    q_sol = result.GetSolution(ik.q())
    plant.SetPositions(plant_context, q_sol)
    plant.SetVelocities(plant_context, np.zeros(plant.num_velocities()))
    print(f"[InitPose] Set EE above cube at {target_p}.")
    return True


class DesiredFeaturesSource(VectorSystem):
    def __init__(self, N_features=4):
        super().__init__(input_size=0, output_size=2 * N_features, direct_feedthrough=False)
        self.N_features = N_features

    def DoCalcVectorOutput(self, context, unused, unused2, output):
        desired_u = 320.0
        desired_v = 240.0
        for i in range(self.N_features):
            output[2 * i] = desired_u + (i % 2) * 20
            output[2 * i + 1] = desired_v + (i // 2) * 20


class DetectedFeaturesSource(VectorSystem):
    def __init__(self, uv: list[float], N_features=4):
        super().__init__(input_size=0, output_size=2 * N_features, direct_feedthrough=False)
        self.N_features = N_features
        if len(uv) < 2 * N_features:
            last_u = uv[-2] if len(uv) >= 2 else 320.0
            last_v = uv[-1] if len(uv) >= 1 else 240.0
            while len(uv) < 2 * N_features:
                uv.extend([last_u, last_v])
        self.uv = np.array(uv[: 2 * N_features], dtype=float)

    def DoCalcVectorOutput(self, context, unused, unused2, output):
        output[:] = self.uv


class DepthEstimator(LeafSystem):
    def __init__(
        self,
        N_features=4,
        image_width=640,
        image_height=480,
        default_depth=0.5,
        depth_ema_alpha=0.6,
        depth_min_m=0.15,
        depth_max_m=2.0,
    ):
        LeafSystem.__init__(self)
        self.N_features = N_features
        self.image_width = image_width
        self.image_height = image_height
        self.default_depth = default_depth
        self.depth_ema_alpha = depth_ema_alpha
        self.depth_min_m = depth_min_m
        self.depth_max_m = depth_max_m
        self.ema_depths = None

        self.depth_input = self.DeclareAbstractInputPort("depth_image", AbstractValue.Make(ImageDepth32F()))
        self.features_input = self.DeclareVectorInputPort("features", size=2 * N_features)
        self.depth_output = self.DeclareVectorOutputPort(
            "depth_estimates", size=N_features, calc=self._calc_depth
        )

    def _calc_depth(self, context, output):
        depth_image_obj = self.depth_input.Eval(context)
        if depth_image_obj.size() == 0:
            output.SetFromVector(np.full(self.N_features, self.default_depth))
            return

        img_height = depth_image_obj.height()
        img_width = depth_image_obj.width()
        depth_array = np.array(depth_image_obj.data).reshape(img_height, img_width)

        features = self.features_input.Eval(context)
        depth_estimates = np.zeros(self.N_features)
        if self.ema_depths is None:
            self.ema_depths = np.full(self.N_features, self.default_depth)

        for i in range(self.N_features):
            u = int(features[2 * i])
            v = int(features[2 * i + 1])
            if u > 0 and v > 0 and u < img_width and v < img_height:
                depth_value = depth_array[v, u]
                if np.isfinite(depth_value) and depth_value > 0.01 and depth_value < 10.0:
                    depth_value = np.clip(depth_value, self.depth_min_m, self.depth_max_m)
                    if self.ema_depths[i] is None or not np.isfinite(self.ema_depths[i]):
                        self.ema_depths[i] = depth_value
                    else:
                        self.ema_depths[i] = (
                            self.depth_ema_alpha * depth_value + (1.0 - self.depth_ema_alpha) * self.ema_depths[i]
                        )
                    depth_estimates[i] = self.ema_depths[i]
                else:
                    depth_estimates[i] = self.ema_depths[i] if np.isfinite(self.ema_depths[i]) else self.default_depth
            else:
                depth_estimates[i] = self.ema_depths[i] if np.isfinite(self.ema_depths[i]) else self.default_depth

        output.SetFromVector(depth_estimates)


class JointVelocityController(LeafSystem):
    def __init__(self, plant, ee_frame, diagram=None, root_context_ref=None):
        LeafSystem.__init__(self)
        self.plant = plant
        self.ee_frame = ee_frame
        self.diagram = diagram
        self.root_context_ref = root_context_ref
        self.twist_input = self.DeclareVectorInputPort("twist", size=6)
        self.qdot_output = self.DeclareVectorOutputPort("qdot", size=plant.num_actuators(), calc=self._calc_qdot)
        self._dbg_count = 0  # limited debug prints

    def _calc_qdot(self, context, output):
        Vee_des = self.twist_input.Eval(context)
        twist_mag = np.linalg.norm(Vee_des)
        try:
            root_context = None
            if self.root_context_ref is not None and self.root_context_ref[0] is not None:
                root_context = self.root_context_ref[0]
            if root_context is None:
                #print("[ROBOT_COMMAND] WARNING: No root context available. Outputting zeros.", flush=True)
                output.SetFromVector(np.zeros(self.plant.num_actuators()))
                return

            plant_context = self.plant.GetMyContextFromRoot(root_context)
            num_vars = self.plant.num_velocities()  # matches Jacobian columns
            num_actuators = self.plant.num_actuators()

            # Transform desired twist from end-effector frame to world frame.
            # The incoming twist is expressed in the EE frame; the Jacobian below expects
            # a spatial velocity expressed in the world frame.
            X_WE = self.plant.CalcRelativeTransform(plant_context, self.plant.world_frame(), self.ee_frame)
            R_WE = X_WE.rotation().matrix()
            p_WE = X_WE.translation()
            Ad_WE = adjoint_transform(R_WE, p_WE)
            V_world = Ad_WE @ Vee_des

            # Debug: print a few samples to trace frame transforms
            if self._dbg_count < 6:
                print(
                    f"[JVEC] dbg#{self._dbg_count}: V_cam_in={Vee_des}, V_ee_in={Vee_des}, "
                    f"V_world={V_world}, p_WE={p_WE}, R_WE[0,0]={R_WE[0,0]:.3f}",
                    flush=True,
                )
                self._dbg_count += 1

            J_spatial = self.plant.CalcJacobianSpatialVelocity(
                plant_context,
                JacobianWrtVariable.kQDot,
                self.ee_frame,
                np.zeros(3),
                self.plant.world_frame(),
                self.plant.world_frame(),
            )[:, :num_vars]

            J_trans = self.plant.CalcJacobianTranslationalVelocity(
                plant_context,
                JacobianWrtVariable.kQDot,
                self.ee_frame,
                np.zeros(3),
                self.plant.world_frame(),
                self.plant.world_frame(),
            )[:2, :num_vars]

            V_des = np.asarray(V_world, dtype=float).reshape(-1)
            if V_des.size != J_spatial.shape[0]:
                print(
                    f"[QP] Warning: V_des size {V_des.size} != J rows {J_spatial.shape[0]}, padding/truncating",
                    flush=True,
                )
                if V_des.size < J_spatial.shape[0]:
                    V_des = np.pad(V_des, (0, J_spatial.shape[0] - V_des.size))
                else:
                    V_des = V_des[: J_spatial.shape[0]]

            # Damped least squares (no Drake QP): qdot = Jᵀ (J Jᵀ + λ² I)⁻¹ V
            lam = 1e-3
            JJt = J_spatial @ J_spatial.T + (lam ** 2) * np.eye(J_spatial.shape[0])
            v = np.linalg.solve(JJt, V_des)
            qdot_sol = J_spatial.T @ v

            # Clip joint velocities to limits
            qdot_sol = np.clip(qdot_sol, -VEL_LIMIT, VEL_LIMIT)

            # Enforce xy bounds via simple step scaling
            dt = QP_DT
            dp = dt * (J_trans @ qdot_sol)
            alpha = 1.0
            x_next = p_WE[0] + dp[0]
            y_next = p_WE[1] + dp[1]
            if dp[0] < 0 and x_next < XY_BOUNDS["xmin"]:
                alpha = min(alpha, (XY_BOUNDS["xmin"] - p_WE[0]) / dp[0])
            if dp[0] > 0 and x_next > XY_BOUNDS["xmax"]:
                alpha = min(alpha, (XY_BOUNDS["xmax"] - p_WE[0]) / dp[0])
            if dp[1] < 0 and y_next < XY_BOUNDS["ymin"]:
                alpha = min(alpha, (XY_BOUNDS["ymin"] - p_WE[1]) / dp[1])
            if dp[1] > 0 and y_next > XY_BOUNDS["ymax"]:
                alpha = min(alpha, (XY_BOUNDS["ymax"] - p_WE[1]) / dp[1])
            alpha = max(0.0, min(1.0, alpha))
            qdot_sol *= alpha

            qdot_cmd = np.zeros(num_actuators)
            count = min(num_vars, num_actuators)
            qdot_cmd[:count] = qdot_sol[:count]
            qdot_mag = np.linalg.norm(qdot_cmd)
            if twist_mag < 1e-6 and qdot_mag > 1e-6 and not hasattr(self, "_nonzero_qdot_warned"):
                print(f"[JOINT_VEL_CONTROLLER] WARN: zero twist but non-zero qdot (mag={qdot_mag:.2e})")
                self._nonzero_qdot_warned = True

            #print(f"[ROBOT_COMMAND] Sending qdot: mag={qdot_mag:.6f}, qdot={qdot_cmd}", flush=True)
            output.SetFromVector(qdot_cmd)
        except Exception as e:
            #print(f"[ROBOT_COMMAND] ERROR in _calc_qdot: {e}", flush=True)
            output.SetFromVector(np.zeros(self.plant.num_actuators()))


def build_ibvs_diagram(scene_name: str | None = None, detected_uv=None, detected_depth=None):
    builder = DiagramBuilder()

    scene_path = _resolve_scene(scene_name)
    scenario = LoadScenario(filename=str(scene_path))

    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=scenario.plant_config.time_step)
    ApplyMultibodyPlantConfig(scenario.plant_config, plant)
    parser = Parser(plant, scene_graph)

    manipulation_models_path = os.path.join(os.path.dirname(manipulation.__file__), "models")
    parser.package_map().Add("manipulation", manipulation_models_path)
    drake_models_path = os.path.join(manipulation_models_path, "drake_models")
    if os.path.isdir(drake_models_path):
        parser.package_map().Add("drake_models", drake_models_path)

    directives = ModelDirectives(directives=scenario.directives)
    ProcessModelDirectives(directives, parser)

    plant.Finalize()
    iiwa_model = plant.GetModelInstanceByName("iiwa")
    ee_frame = plant.GetFrameByName("iiwa_link_7", iiwa_model)

    rgbd_sensor, image_converter, X_Camera_EE = add_eye_in_hand_camera(builder, plant, scene_graph, ee_frame)

    N_features = 1  # Single feature: midpoint of cube corners
    feature_tracker = builder.AddSystem(FeatureTracker(N_features=N_features))
    builder.Connect(rgbd_sensor.color_image_output_port(), feature_tracker.image_input)

    from visualization import CameraFeatureVisualizer
    camera_visualizer = builder.AddSystem(
        CameraFeatureVisualizer(N_features=N_features, save_dir="camera_screenshots", save_interval=0.25)
    )
    builder.Connect(rgbd_sensor.color_image_output_port(), camera_visualizer.image_input)
    builder.Connect(feature_tracker.features_output, camera_visualizer.features_input)

    # Camera intrinsics (match add_eye_in_hand_camera)
    width = 640
    height = 480
    fov_y = np.pi / 4
    fy = height / (2.0 * np.tan(fov_y / 2.0))
    fx = fy  # square pixels assumed
    cx = width / 2.0
    cy = height / 2.0

    # Add IBVS controller (DOF mask will come from state machine)
    ibvs_controller = builder.AddSystem(
        IBVSController(
            N_features=N_features,
            lambda_gain= 0.0,  # was 1000.0; lower to avoid overshoot with correct scaling
            focal_length=fx,
            cx=cx,
            cy=cy,
            error_threshold=10.0,  # Base threshold (states can override)
            convergence_window=10,
            dls_lambda=0.01,
            dof_mask=np.ones(6)  # Default: all DOFs enabled (will be overridden by state machine)
        )
    )
    builder.Connect(feature_tracker.features_output, ibvs_controller.current_uv_input)

    # Add state machine (dictionary-based hierarchical control)
    use_state_machine = True  # Enable state machine for multi-stage control
    
    if use_state_machine:
        state_machine = builder.AddSystem(IBVSStateMachine(N_features=N_features))
        
        # Connect current features to state machine (for state transitions)
        builder.Connect(
            feature_tracker.features_output,
            state_machine.current_features_input
        )
        
        # Connect convergence flag from IBVS controller to state machine
        # Add ZeroOrderHold to break algebraic loop (converged output is direct feedthrough)
        converged_zoh = builder.AddSystem(ZeroOrderHold(period_sec=0.01, vector_size=1))
        builder.Connect(ibvs_controller.converged_output, converged_zoh.get_input_port())
        builder.Connect(converged_zoh.get_output_port(), state_machine.converged_input)
        
        # Connect desired features from state machine to IBVS controller
        builder.Connect(
            state_machine.desired_features_output,
            ibvs_controller.desired_uv_input
        )
        
        # Connect desired features to visualizer for error display
        builder.Connect(
            state_machine.desired_features_output,
            camera_visualizer.desired_features_input
        )
        
        # Connect state name to visualizer for state display
        builder.Connect(
            state_machine.state_name_output,
            camera_visualizer.state_name_input
        )
        
        # Connect DOF mask from state machine to IBVS controller (dynamic control)
        builder.Connect(
            state_machine.dof_mask_output,
            ibvs_controller.dof_mask_input
        )
    else:
        # Use simple desired features source (fallback, no state machine)
        if detected_uv is not None:
            desired_features = builder.AddSystem(DetectedFeaturesSource(list(detected_uv), N_features=N_features))
        else:
            desired_features = builder.AddSystem(DesiredFeaturesSource(N_features=N_features))
        builder.Connect(desired_features.get_output_port(0), ibvs_controller.desired_uv_input)
        # Connect desired features to visualizer for error display
        builder.Connect(desired_features.get_output_port(0), camera_visualizer.desired_features_input)

    depth_estimator = builder.AddSystem(
        DepthEstimator(
            N_features=N_features,
            image_width=640,
            image_height=480,
            default_depth=float(detected_depth) if detected_depth is not None else 0.5,
            depth_ema_alpha=0.6,
            depth_min_m=0.01,
            depth_max_m=2.0,
        )
    )
    builder.Connect(rgbd_sensor.depth_image_32F_output_port(), depth_estimator.depth_input)
    builder.Connect(feature_tracker.features_output, depth_estimator.features_input)
    builder.Connect(depth_estimator.depth_output, ibvs_controller.depth_input)

    use_eye_in_hand_transform = True

    if use_eye_in_hand_transform:
        # Use the actual camera-on-EE pose from add_eye_in_hand_camera for frame mapping.
        R_ee_cam = X_Camera_EE.rotation().matrix()
        t_ee_cam = X_Camera_EE.translation()
        # Map twist from camera frame to EE frame: use Adjoint of X_EC (EE to Camera).
        Ad_ee_cam = adjoint_transform(R_ee_cam, t_ee_cam)

        class EyeInHandTransform(LeafSystem):
            def __init__(self, Ad_ee_cam):
                LeafSystem.__init__(self)
                self.Ad_ee_cam = Ad_ee_cam
                self.twist_input = self.DeclareVectorInputPort("twist_camera", size=6)
                self.twist_output = self.DeclareVectorOutputPort("twist_ee", size=6, calc=self._transform)
                self._dbg_count = 0  # limited debug prints

            def _transform(self, context, output):
                v_cam = self.twist_input.Eval(context)
                v_ee = self.Ad_ee_cam @ v_cam
                if self._dbg_count < 6:
                    print(
                        f"[EIH] cam->ee dbg#{self._dbg_count}: v_cam={v_cam}, v_ee={v_ee}",
                        flush=True,
                    )
                    self._dbg_count += 1
                output.SetFromVector(v_ee)

        eih_transform = builder.AddSystem(EyeInHandTransform(Ad_ee_cam))
        builder.Connect(ibvs_controller.velocity_output, eih_transform.twist_input)
        twist_source = eih_transform.twist_output
    else:
        twist_source = ibvs_controller.velocity_output

    root_context_ref = [None]
    joint_vel_controller = builder.AddSystem(
        JointVelocityController(plant, ee_frame, diagram=None, root_context_ref=root_context_ref)
    )
    builder.Connect(twist_source, joint_vel_controller.twist_input)

    # Gravity compensation + torque summation
    class GravityCompensator(LeafSystem):
        def __init__(self, plant, root_context_ref):
            LeafSystem.__init__(self)
            self.plant = plant
            self.root_context_ref = root_context_ref
            self.torque_output = self.DeclareVectorOutputPort(
                "gravity_torque",
                size=plant.num_actuators(),
                calc=self._calc_torque,
            )

        def _calc_torque(self, context, output):
            try:
                root_context = self.root_context_ref[0]
                if root_context is None:
                    output.SetFromVector(np.zeros(self.plant.num_actuators()))
                    return
                plant_context = self.plant.GetMyContextFromRoot(root_context)
                g_forces = self.plant.CalcGravityGeneralizedForces(plant_context)
                # Map to actuators (assumes ordering aligns)
                n = min(len(g_forces), self.plant.num_actuators())
                tau = np.zeros(self.plant.num_actuators())
                tau[:n] = g_forces[:n]
                output.SetFromVector(tau)
            except Exception:
                output.SetFromVector(np.zeros(self.plant.num_actuators()))

    # Convert joint velocities to torques using damping + gravity compensation
    class VelocityToTorqueConverter(LeafSystem):
        def __init__(self, plant, root_context_ref, damping=2.0, vel_gain=25.0, torque_limit=60.0):
            """
            Tracks desired joint velocities with a simple PD-like torque law:
                tau = vel_gain * (qdot_des - qdot) - damping * qdot + tau_gravity
            This is consistent with the IIWA's torque-controlled actuation and avoids
            fighting the commanded velocities (previous code applied a damping term
            directly to qdot_des).
            """
            LeafSystem.__init__(self)
            self.plant = plant
            self.root_context_ref = root_context_ref
            self.damping = damping
            self.vel_gain = vel_gain
            self.torque_limit = torque_limit
            self.qdot_input = self.DeclareVectorInputPort("qdot", size=plant.num_actuators())
            self.torque_output = self.DeclareVectorOutputPort(
                "torque",
                size=plant.num_actuators(),
                calc=self._calc_torque,
            )

            # Pre-compute mapping from actuators to velocity indices for speed/clarity.
            actuated_indices = []
            for actuator_index in self.plant.GetJointActuatorIndices():
                actuator = self.plant.get_joint_actuator(actuator_index)
                joint = actuator.joint()
                for k in range(joint.num_velocities()):
                    actuated_indices.append(joint.velocity_start() + k)
            self._actuated_indices = actuated_indices[: self.plant.num_actuators()]

        def _calc_torque(self, context, output):
            try:
                root_context = self.root_context_ref[0]
                if root_context is None:
                    output.SetFromVector(np.zeros(self.plant.num_actuators()))
                    return

                plant_context = self.plant.GetMyContextFromRoot(root_context)
                qdot_des = np.asarray(self.qdot_input.Eval(context)).flatten()

                num_actuators = self.plant.num_actuators()
                if qdot_des.size < num_actuators:
                    qdot_des = np.pad(qdot_des, (0, num_actuators - qdot_des.size))
                elif qdot_des.size > num_actuators:
                    qdot_des = qdot_des[:num_actuators]

                # Current joint velocities for the actuated joints (IIWA is fixed-base).
                qdot_full = self.plant.GetVelocities(plant_context)
                qdot_current = np.zeros(num_actuators)
                for i, vel_idx in enumerate(self._actuated_indices):
                    if vel_idx < len(qdot_full):
                        qdot_current[i] = qdot_full[vel_idx]

                # Velocity tracking torque (positive when desired > current).
                tau_track = self.vel_gain * (qdot_des - qdot_current)

                # Viscous damping to keep motion stable.
                tau_damp = -self.damping * qdot_current

                # Gravity compensation.
                g_forces = self.plant.CalcGravityGeneralizedForces(plant_context)
                tau_grav = np.zeros(num_actuators)
                n = min(len(g_forces), num_actuators)
                # Compensate gravity: apply equal and opposite joint torques.
                tau_grav[:n] = -g_forces[:n]

                tau_total = tau_track + tau_damp + tau_grav
                tau_total = np.clip(tau_total, -self.torque_limit, self.torque_limit)

                output.SetFromVector(tau_total)
            except Exception:
                # Fallback: just gravity compensation or zeros if unavailable.
                try:
                    root_context = self.root_context_ref[0]
                    if root_context is not None:
                        plant_context = self.plant.GetMyContextFromRoot(root_context)
                        g_forces = self.plant.CalcGravityGeneralizedForces(plant_context)
                        n = min(len(g_forces), self.plant.num_actuators())
                        tau = np.zeros(self.plant.num_actuators())
                        tau[:n] = -g_forces[:n]
                        output.SetFromVector(tau)
                    else:
                        output.SetFromVector(np.zeros(self.plant.num_actuators()))
                except Exception:
                    output.SetFromVector(np.zeros(self.plant.num_actuators()))

    vel_to_torque = builder.AddSystem(VelocityToTorqueConverter(plant, root_context_ref, damping=5.0))
    builder.Connect(joint_vel_controller.qdot_output, vel_to_torque.qdot_input)
    builder.Connect(vel_to_torque.torque_output, plant.get_actuation_input_port())

    from pydrake.all import MeshcatVisualizer, StartMeshcat

    meshcat = StartMeshcat()
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    class CameraMarkerPublisher(LeafSystem):
        """Publishes a small cyan box at the camera pose so it is visible in Meshcat."""

        def __init__(self, plant, ee_frame, X_Camera_EE, meshcat, root_context_ref, path="camera_marker"):
            LeafSystem.__init__(self)
            self.plant = plant
            self.ee_frame = ee_frame
            self.X_Camera_EE = X_Camera_EE
            self.meshcat = meshcat
            self.root_context_ref = root_context_ref
            self.path = path
            self.box = Box(0.03, 0.02, 0.02)
            self.meshcat.SetObject(self.path, self.box, Rgba(0.0, 1.0, 1.0, 0.6))
            self.DeclarePeriodicEvent(period_sec=0.02, offset_sec=0.0, event=PublishEvent(self.DoPublish))

        def DoPublish(self, context, event):
            # Use the stored root context (set in main) to access the plant context.
            root_context = self.root_context_ref[0]
            if root_context is None:
                return
            plant_context = self.plant.GetMyContextFromRoot(root_context)
            X_WE = self.plant.CalcRelativeTransform(
                plant_context, self.plant.world_frame(), self.ee_frame
            )
            X_WC = X_WE @ self.X_Camera_EE
            self.meshcat.SetTransform(self.path, X_WC.GetAsMatrix4())

    builder.AddSystem(CameraMarkerPublisher(plant, ee_frame, X_Camera_EE, meshcat, root_context_ref))

    diagram = builder.Build()
    joint_vel_controller.diagram = diagram
    return diagram, plant, ee_frame, root_context_ref


def main():
    parser = argparse.ArgumentParser(description="Run IBVS demo with optional cube detection seeding.")
    parser.add_argument("--detect-once", action="store_true", help="Run cube detection once to seed desired features/depth.")
    parser.add_argument(
        "--detect-eye-in-hand",
        action="store_true",
        help="Use eye-in-hand view for one-shot detection (default: overhead).",
    )
    parser.add_argument("--sim-time", type=float, default=5.0, help="Simulation duration in seconds.")
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Scenario YAML from src/experiments_finalproject/generated_scenarios/",
    )
    args = parser.parse_args()

    available = _available_scenes()
    print(f"Available scenes ({len(available)}): {available}")
    scene_path = _resolve_scene(args.scene)
    print(f"Selected scene: {scene_path.name}")

    detected_uv = None
    detected_depth = None
    if args.detect_once:
        print("[Detect] Running cube detection to seed desired features...")
        result, meshcat_url = capture_and_detect(scene=scene_path.name, use_overhead=not args.detect_eye_in_hand)
        cx, cy = result["centroid"]
        detected_depth = result.get("depth_m", None)
        detected_uv = [cx, cy, cx, cy, cx, cy, cx, cy]
        print(f"[Detect] centroid=({cx:.1f},{cy:.1f}), depth={detected_depth}, meshcat={meshcat_url}")

    print("Building IBVS diagram...")
    diagram, plant, ee_frame, root_context_ref = build_ibvs_diagram(
        scene_name=scene_path.name, detected_uv=detected_uv, detected_depth=detected_depth
    )

    print("Creating simulator...")
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    root_context_ref[0] = context

    plant_context = plant.GetMyContextFromRoot(context)
    num_pos = plant.num_positions()
    print(f"Plant has {num_pos} positions")
    placed = _set_hover_start_over_cube(plant, plant_context, ee_frame, cube_xy=(0.5, 0.3), hover_z=0.35)
    if not placed and num_pos >= 7:
        # Fallback to a sane seed if IK fails.
        q0_iiwa = np.array([-0.92, 0.08, -0.34, 0.76, -0.28, -1.5, 0])
        q0 = plant.GetPositions(plant_context).copy()
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
