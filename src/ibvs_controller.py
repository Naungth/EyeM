"""
IBVS Control Law

Implements the image-based visual servoing control law:
    v = -λ · L⁺ · e

where:
    - v: desired spatial velocity (6D twist)
    - λ: gain parameter
    - L: image Jacobian (interaction matrix)
    - e: feature error (u_current - u_desired)
    - L⁺: pseudoinverse of L

The controller computes the desired end-effector twist based on image feature error.
"""

import numpy as np
from pydrake.all import LeafSystem


class IBVSController(LeafSystem):
    """
    LeafSystem implementing IBVS control law.
    
    Inputs:
        - current_uv: Vector of size 2N (current feature coordinates)
        - desired_uv: Vector of size 2N (desired feature coordinates)
        - depth_estimates: Vector of size N (depth Z for each feature)
    
    Output:
        - desired_spatial_velocity: Vector of size 6 (twist: [vx, vy, vz, wx, wy, wz])
    """
    
    def __init__(
        self,
        N_features=4,
        lambda_gain=1.0,
        focal_length=320.0,
        error_threshold=10.0,
        convergence_window=10,
        dls_lambda=0.01,
        dof_mask=None,
        cx=None,
        cy=None,
    ):
        """
        Initialize IBVS controller.
        
        Args:
            N_features: Number of feature points
            lambda_gain: Control gain λ
            focal_length: Camera focal length in pixels (assumed fx = fy if cx/cy not provided)
            cx, cy: Principal point in pixels (defaults to center if None)
            error_threshold: Feature error threshold in pixels - if error below this, stop (default: 10.0)
            convergence_window: Number of consecutive frames below threshold to confirm convergence (default: 10)
            dls_lambda: Damped Least Squares regularization parameter (default: 0.01)
                       Higher values = more damping, more robust but slower
            dof_mask: 6D array [vx,vy,vz,wx,wy,wz] with 1.0 to enable, 0.0 to disable (default: all enabled)
        """
        LeafSystem.__init__(self)
        
        self.N_features = N_features
        self.lambda_gain = lambda_gain
        self.fx = focal_length
        self.fy = focal_length
        self.cx = 320.0 if cx is None else float(cx)
        self.cy = 240.0 if cy is None else float(cy)
        self.error_threshold = error_threshold  # Stop if error below this (pixels)
        self.convergence_window = convergence_window
        self.dls_lambda = dls_lambda  # DLS regularization parameter
        
        # DOF mask: which degrees of freedom to control
        if dof_mask is None:
            self.dof_mask = np.ones(6, dtype=float)  # All DOFs enabled by default
        else:
            self.dof_mask = np.array(dof_mask, dtype=float)
            if len(self.dof_mask) != 6:
                raise ValueError("dof_mask must be a 6-element array")
        
        # Track error history for convergence detection
        self.error_history = []
        self.converged = False
        self._debug_count = 0  # for limited debug printing
        
        # Output port for convergence flag (for state machine)
        self.converged_output = self.DeclareVectorOutputPort(
            "converged",
            size=1,
            calc=self._calc_converged
        )
        
        # Input ports
        self.current_uv_input = self.DeclareVectorInputPort(
            "current_uv",
            size=2 * N_features
        )
        
        self.desired_uv_input = self.DeclareVectorInputPort(
            "desired_uv",
            size=2 * N_features
        )
        
        self.depth_input = self.DeclareVectorInputPort(
            "depth_estimates",
            size=N_features
        )
        
        # Optional input: dynamic DOF mask (if not connected, uses constructor default)
        self.dof_mask_input = self.DeclareVectorInputPort(
            "dof_mask",
            size=6
        )
        
        # Output port: 6D spatial velocity (twist)
        self.velocity_output = self.DeclareVectorOutputPort(
            "desired_spatial_velocity",
            size=6,
            calc=self._calc_velocity
        )
    
    def compute_interaction_matrix(self, uv, depth_along_optical_axis):
        """
        Compute the image Jacobian (interaction matrix) L for a camera whose
        optical axis is +Z (Drake RgbdSensor convention). Pixel coordinates are
        (u, v) in pixels, depth is along +Z.

        Pixel coordinates are converted to normalized image coordinates using
        the intrinsics (fx, fy, cx, cy). The interaction matrix is first built
        in normalized units, then scaled back to pixel units so it matches the
        pixel-space error we feed in.

        For normalized coordinates (x = (u-cx)/fx, y = (v-cy)/fy):
            [ẋ]   [ -1/Z   0     x/Z    x*y   -(1 + x^2)    y ] [vx]
            [ẏ] = [  0    -1/Z   y/Z   1 + y^2   -x*y      -x ] [vy]
                  [                                  angular terms           ] [vz wx wy wz]^T
        We scale rows by fx / fy to return to pixel-space Jacobian.

        Args:
            uv: np.ndarray, shape (N, 2), pixel coordinates
            depth_along_optical_axis: np.ndarray, shape (N,), depth Z per feature (meters)

        Returns:
            L: np.ndarray, shape (2N, 6)
        """
        N = len(uv)
        L = np.zeros((2 * N, 6))

        for i in range(N):
            u, v = uv[i, 0], uv[i, 1]
            Z = depth_along_optical_axis[i] if depth_along_optical_axis[i] > 1e-6 else 1e-6
            x = (u - self.cx) / self.fx
            y = (v - self.cy) / self.fy

            # Row for u (image x)
            L[2 * i, 0] = -1.0 / Z             # vx
            L[2 * i, 1] = 0.0                  # vy
            L[2 * i, 2] = x / Z                # vz
            L[2 * i, 3] = x * y                # wx
            L[2 * i, 4] = -(1.0 + x * x)       # wy
            L[2 * i, 5] = y                    # wz

            # Row for v (image y)
            L[2 * i + 1, 0] = 0.0                # vx
            L[2 * i + 1, 1] = -1.0 / Z           # vy
            L[2 * i + 1, 2] = y / Z              # vz
            L[2 * i + 1, 3] = 1.0 + y * y        # wx
            L[2 * i + 1, 4] = -x * y             # wy
            L[2 * i + 1, 5] = -x                 # wz

            # Scale back to pixel units
            L[2 * i, :] *= self.fx
            L[2 * i + 1, :] *= self.fy

        return L
    
    def compute_twist(self, L, error, lam=None, dof_mask=None):
        """
        Compute desired spatial velocity from interaction matrix and error.
        
        Uses Damped Least Squares (DLS) for robustness:
        v = -λ · L_DLS⁺ · e
        
        where L_DLS⁺ = L^T (L L^T + λ² I)^(-1)
        
        This is more robust than simple pseudoinverse when L is near-singular.
        
        Args:
            L: Interaction matrix (2N, 6)
            error: Feature error vector (2N,)
            lam: Control gain (defaults to self.lambda_gain)
            dof_mask: Optional DOF mask (if None, uses self.dof_mask)
        
        Returns:
            numpy array: (6,) desired spatial velocity
        """
        if lam is None:
            lam = self.lambda_gain
        
        # Use provided mask or default
        if dof_mask is None:
            dof_mask = self.dof_mask
        
        # Apply DOF mask: only use enabled degrees of freedom
        allowed_idx = np.where(dof_mask > 0.5)[0]
        if allowed_idx.size == 0:
            # All DOFs disabled - return zero velocity
            if not hasattr(self, '_dof_mask_warned'):
                print(f"[IBVS] WARNING: All DOFs disabled (mask={dof_mask}), returning zero velocity")
                self._dof_mask_warned = True
            return np.zeros(6)
        
        # Reduce L to only enabled DOFs
        L_red = L[:, allowed_idx]
        
        # Damped Least Squares (DLS) - more robust than pseudoinverse
        # L_DLS⁺ = L^T (L L^T + λ² I)^(-1)
        LLt = L_red @ L_red.T
        dls_term = LLt + (self.dls_lambda ** 2) * np.eye(L_red.shape[0])
        L_dls_pinv = L_red.T @ np.linalg.inv(dls_term)
        
        # Compute twist for reduced DOFs
        v_red = -lam * (L_dls_pinv @ error)
        
        # Map back to full 6D twist
        v = np.zeros(6, dtype=float)
        v[allowed_idx] = v_red
        
        return v
    
    def _calc_velocity(self, context, output):
        """Compute desired spatial velocity from inputs."""
        # Get inputs
        current_uv = self.current_uv_input.Eval(context)
        desired_uv = self.desired_uv_input.Eval(context)
        depth_estimates = self.depth_input.Eval(context)
        
        # Get DOF mask (use dynamic input if connected, otherwise use default)
        try:
            dynamic_mask = self.dof_mask_input.Eval(context)
            # Always use dynamic mask if input is connected (even if all zeros)
            active_dof_mask = dynamic_mask
        except:
            # Input not connected, use default from constructor
            active_dof_mask = self.dof_mask
        
        # Reshape to (N, 2)
        current_uv_2d = current_uv.reshape(self.N_features, 2)
        desired_uv_2d = desired_uv.reshape(self.N_features, 2)
        
        # Compute error (current - desired), standard IBVS convention:
        #   v = -λ · L⁺ · (s - s*)
        # Using (current - desired) ensures the sign matches the image Jacobian columns
        # (e.g., du/dt = -f/Z * vx, so if u is too large, vx will reduce it).
        error = (current_uv_2d - desired_uv_2d).flatten()
        
        # Filter out invalid features (zero coordinates indicate no feature)
        valid_mask = np.any(current_uv_2d != 0, axis=1) & (depth_estimates > 0)
        
        if not np.any(valid_mask):
            # No valid features, output zero velocity
            output.SetFromVector(np.zeros(6))
            return
        
        # Use only valid features
        valid_uv = current_uv_2d[valid_mask]
        valid_desired_uv = desired_uv_2d[valid_mask]
        valid_Z = depth_estimates[valid_mask]
        valid_error = (valid_uv - valid_desired_uv).flatten()
        
        # Check if error is below threshold (stopping condition)
        # Compute RMS error in pixels
        error_magnitude = np.linalg.norm(valid_error) / np.sqrt(len(valid_error))
        
        # Track error history for convergence detection
        self.error_history.append(error_magnitude)
        if len(self.error_history) > self.convergence_window:
            self.error_history.pop(0)  # Keep only recent history
        
        # Check convergence: error must be below threshold for several consecutive frames
        if len(self.error_history) >= self.convergence_window:
            recent_errors = self.error_history[-self.convergence_window:]
            if all(e < self.error_threshold for e in recent_errors):
                self.converged = True
        
        # If converged, stop moving
        if self.converged:
            output.SetFromVector(np.zeros(6))
            return
        
        # Also stop if current error is very small (immediate stop for very good alignment)
        if error_magnitude < self.error_threshold * 0.5:  # Half the threshold
            output.SetFromVector(np.zeros(6))
            return
        
        # Compute interaction matrix
        L = self.compute_interaction_matrix(valid_uv, valid_Z)
        
        # Compute twist with active DOF mask
        v = self.compute_twist(L, valid_error, dof_mask=active_dof_mask)

        # Debug: print a few samples to verify direction
        if self._debug_count < 5:
            print(
                f"[IBVS] err_pix={valid_error} | rms={error_magnitude:.2f} | "
                f"dof_mask={active_dof_mask} | twist_cam={v}",
                flush=True,
            )
            self._debug_count += 1
        
        # Debug: check if DOF mask is causing zero output
        if np.allclose(v, 0) and not np.allclose(self.dof_mask, 0):
            # This shouldn't happen - if DOF mask has enabled DOFs, we should get non-zero output
            pass  # Could add debug here if needed
        
        output.SetFromVector(v)
    
    def _calc_converged(self, context, output):
        """Output convergence flag (1.0 if converged, 0.0 otherwise)."""
        output.SetFromVector(np.array([1.0 if self.converged else 0.0]))
