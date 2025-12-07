"""
Camera View Visualization with Feature Overlay

Provides a system to visualize the camera view with detected features overlaid
as blue dots, and optionally save screenshots.
"""

import cv2
import numpy as np
import os
from pydrake.all import (
    AbstractValue,
    ImageRgba8U,
    LeafSystem,
)


class CameraFeatureVisualizer(LeafSystem):
    """
    LeafSystem that visualizes camera view with feature overlays.
    
    Inputs:
        - rgb_image: AbstractValue containing ImageRgba8U
        - features: Vector of size 2N (feature coordinates [u1, v1, u2, v2, ...])
    
    Outputs:
        - visualized_image: AbstractValue containing ImageRgba8U with blue dots
    """
    
    def __init__(self, N_features=4, save_dir="camera_screenshots", save_interval=1.0):
        """
        Initialize visualizer.
        
        Args:
            N_features: Number of features to visualize
            save_dir: Directory to save screenshots (None to disable saving)
            save_interval: Save screenshot every N seconds (0 to save every frame)
        """
        LeafSystem.__init__(self)
        
        self.N_features = N_features
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.last_save_time = -float('inf')
        self.frame_count = 0
        
        # Create save directory if specified
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
        
        # Input ports
        self.image_input = self.DeclareAbstractInputPort(
            "rgb_image",
            AbstractValue.Make(ImageRgba8U())
        )
        
        self.features_input = self.DeclareVectorInputPort(
            "features",
            size=2 * N_features
        )
        
        # Optional input port for desired features (for error visualization)
        self.desired_features_input = self.DeclareVectorInputPort(
            "desired_features",
            size=2 * N_features
        )
        
        # Output port: visualized image (optional, for display)
        self.visualized_output = self.DeclareAbstractOutputPort(
            "visualized_image",
            lambda: AbstractValue.Make(ImageRgba8U()),
            self._visualize
        )
        
        # Use periodic update to ensure _visualize is called even if output isn't connected
        # This ensures screenshots are saved regardless of output connections
        self.DeclarePeriodicPublishEvent(
            0.01,  # period_sec: Update every 10ms (100 Hz)
            0.0,   # offset_sec
            self._publish_visualization  # publish callback
        )
    
    def _get_image_and_features(self, context):
        """Helper to get image, current features, and desired features from inputs."""
        # Get input image
        image_value = self.image_input.Eval(context)
        
        # Convert to numpy array
        if isinstance(image_value, ImageRgba8U):
            if image_value.size() == 0:
                return None, None, None
            
            image_array = np.array(image_value.data).reshape(
                image_value.height(), image_value.width(), 4
            )[:, :, :3]  # Drop alpha channel, keep RGB
        else:
            return None, None, None
        
        # Get current features
        features_flat = self.features_input.Eval(context)
        features = features_flat.reshape(self.N_features, 2)  # (N, 2) array of [u, v]
        
        # Get desired features (optional - may not be connected)
        desired_features = None
        try:
            desired_features_flat = self.desired_features_input.Eval(context)
            desired_features = desired_features_flat.reshape(self.N_features, 2)  # (N, 2) array
        except:
            # Desired features not connected, that's okay
            pass
        
        return image_array, features, desired_features
    
    def _visualize(self, context, output):
        """Visualize features on camera image (for output port)."""
        image_array, features, desired_features = self._get_image_and_features(context)
        
        if image_array is None:
            output.set_value(ImageRgba8U())
            return
        
        # Process and save (reuse the same logic)
        self._process_and_save(context, image_array, features, desired_features, output)
    
    def _publish_visualization(self, context):
        """Periodic publish event to ensure visualization runs even if output isn't connected."""
        image_array, features, desired_features = self._get_image_and_features(context)
        
        if image_array is None:
            return
        
        # Process and save (without setting output)
        self._process_and_save(context, image_array, features, desired_features, None)
    
    def _process_and_save(self, context, image_array, features, desired_features, output):
        """Process image with features and optionally save screenshot."""
        # Convert RGB to BGR for OpenCV drawing
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Colors (BGR format)
        current_color = (255, 0, 0)    # Blue for current features
        desired_color = (0, 255, 0)    # Green for desired features
        text_color = (255, 255, 255)   # White for text
        
        dot_radius = 5
        dot_thickness = -1  # Filled circle
        
        # Calculate total error magnitude if desired features are available
        sum_squared_errors = 0.0
        valid_error_count = 0
        
        if desired_features is not None:
            for i in range(self.N_features):
                u, v = features[i, 0], features[i, 1]
                u_des, v_des = desired_features[i, 0], desired_features[i, 1]
                
                # Only calculate error if both current and desired features are valid
                if u > 0 and v > 0 and u_des > 0 and v_des > 0:
                    error_squared = (u - u_des)**2 + (v - v_des)**2
                    sum_squared_errors += error_squared
                    valid_error_count += 1
        
        # Calculate RMS error (root mean square)
        if valid_error_count > 0:
            rms_error = np.sqrt(sum_squared_errors / valid_error_count)
        else:
            rms_error = 0.0
        
        # Draw desired features first (so they appear behind current features)
        if desired_features is not None:
            for i in range(self.N_features):
                u_des, v_des = desired_features[i, 0], desired_features[i, 1]
                
                # Only draw if desired feature is valid (non-zero)
                if u_des > 0 and v_des > 0:
                    u_des_int = int(np.round(u_des))
                    v_des_int = int(np.round(v_des))
                    
                    # Draw desired feature as green circle
                    cv2.circle(image_bgr, (u_des_int, v_des_int), dot_radius, desired_color, dot_thickness)
                    
                    # Draw desired feature label
                    cv2.putText(
                        image_bgr,
                        f"D{i+1}",
                        (u_des_int + 8, v_des_int - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        desired_color,
                        1,
                        cv2.LINE_AA
                    )
        
        # Draw current features
        for i in range(self.N_features):
            u, v = features[i, 0], features[i, 1]
            
            # Only draw if feature is valid (non-zero)
            if u > 0 and v > 0:
                u_int = int(np.round(u))
                v_int = int(np.round(v))
                
                # Draw current feature as blue circle
                cv2.circle(image_bgr, (u_int, v_int), dot_radius, current_color, dot_thickness)
                
                # Draw current feature number
                cv2.putText(
                    image_bgr,
                    str(i + 1),
                    (u_int + 8, v_int + 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    current_color,
                    1,
                    cv2.LINE_AA
                )
        
        # Display error magnitude in top-left corner
        if desired_features is not None and valid_error_count > 0:
            error_text = f"RMS Error: {rms_error:.2f} px"
            # Draw text with background for better visibility
            text_size = cv2.getTextSize(error_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x, text_y = 10, 30
            
            # Draw semi-transparent background rectangle
            overlay = image_bgr.copy()
            cv2.rectangle(
                overlay,
                (text_x - 5, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5),
                (0, 0, 0),
                -1
            )
            cv2.addWeighted(overlay, 0.6, image_bgr, 0.4, 0, image_bgr)
            
            # Draw error text
            cv2.putText(
                image_bgr,
                error_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                text_color,
                2,
                cv2.LINE_AA
            )
        
        # Save screenshot if enabled
        if self.save_dir is not None:
            current_time = context.get_time()
            if current_time - self.last_save_time >= self.save_interval:
                self.frame_count += 1
                filename = os.path.join(
                    self.save_dir,
                    f"camera_view_{self.frame_count:05d}_t{current_time:.3f}.png"
                )
                # Save as BGR (OpenCV format)
                success = cv2.imwrite(filename, image_bgr)
                if success:
                    self.last_save_time = current_time
                    n_features = np.sum((features[:, 0] > 0) & (features[:, 1] > 0))
                    print(f"[VISUALIZER] Saved screenshot: {filename} (features: {n_features})")
                else:
                    print(f"[VISUALIZER] ERROR: Failed to save screenshot: {filename}")
        
        # Set output if provided (for output port)
        if output is not None:
            # Convert back to RGB
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            # Convert to RGBA for Drake ImageRgba8U
            h, w = image_rgb.shape[:2]
            image_rgba = np.zeros((h, w, 4), dtype=np.uint8)
            image_rgba[:, :, :3] = image_rgb
            image_rgba[:, :, 3] = 255  # Alpha channel
            
            # Create Drake image
            drake_image = ImageRgba8U(w, h)
            drake_image.mutable_data()[:] = image_rgba.flatten()
            
            output.set_value(drake_image)

