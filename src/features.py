"""
Feature Extraction from RGB Images

Implements a Drake LeafSystem that extracts 2D feature points from RGB images
using HSV-based red cube detection. Extracts the 4 corners of the detected red cube.
"""

import cv2
import numpy as np
from pydrake.all import (
    AbstractValue,
    ImageRgba8U,
    LeafSystem,
)


class FeatureTracker(LeafSystem):
    """
    LeafSystem that extracts N feature points from an RGB image by detecting
    a red cube and extracting its corners.
    
    Input:
        - rgb_image: AbstractValue containing ImageRgba8U or numpy array
    
    Output:
        - features: Vector of size 2N representing N feature points [u1, v1, u2, v2, ...]
        - feature_count: Scalar indicating number of features found
    """
    
    def __init__(self, N_features=4, max_corners=100):
        """
        Initialize feature tracker for red cube detection.
        
        Args:
            N_features: Target number of features to extract (typically 4 for cube corners)
            max_corners: Not used (kept for compatibility)
        """
        LeafSystem.__init__(self)
        
        self.N_features = N_features
        
        # HSV color ranges for red cube detection
        # Red wraps around hue=0, so we need two ranges
        self.hsv_lower1 = np.array([0, 30, 30], dtype=np.uint8)    # Lower red range
        self.hsv_upper1 = np.array([25, 255, 255], dtype=np.uint8)
        self.hsv_lower2 = np.array([160, 30, 30], dtype=np.uint8)  # Upper red range (wraparound)
        self.hsv_upper2 = np.array([180, 255, 255], dtype=np.uint8)
        
        # Contour filtering parameters
        self.min_area_frac = 0.0003  # Minimum contour area as fraction of image area
        self.center_bias = 0.001      # Weight for favoring contours near image center
        
        # Input port: RGB image (can be AbstractValue or numpy array)
        self.image_input = self.DeclareAbstractInputPort(
            "rgb_image",
            AbstractValue.Make(ImageRgba8U())
        )
        
        # Output port: Feature coordinates [u1, v1, u2, v2, ...]
        self.features_output = self.DeclareVectorOutputPort(
            "features",
            size=2 * N_features,
            calc=self._calc_features
        )
        
        # Output port: Number of features found
        self.feature_count_output = self.DeclareVectorOutputPort(
            "feature_count",
            size=1,
            calc=self._calc_feature_count
        )
    
    def _extract_features(self, image):
        """
        Extract features from image by detecting red cube and extracting its corners.
        
        Args:
            image: numpy array (H, W, 3) uint8 RGB image
        
        Returns:
            numpy array: (N, 2) array of [u, v] coordinates, padded to N_features
        """
        # Convert RGB to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Create mask for red color (handling hue wraparound)
        mask1 = cv2.inRange(hsv, self.hsv_lower1, self.hsv_upper1)
        mask2 = cv2.inRange(hsv, self.hsv_lower2, self.hsv_upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Clean up mask to reduce noise
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours or len(contours) == 0:
            # No cube detected - return zeros
            return np.zeros((self.N_features, 2))
        
        # Filter by minimum area
        img_area = image.shape[0] * image.shape[1]
        area_thresh = max(10, self.min_area_frac * img_area)
        filtered = [c for c in contours if cv2.contourArea(c) >= area_thresh]
        
        if not filtered:
            # No valid contours - return zeros
            return np.zeros((self.N_features, 2))
        
        # Select best contour (largest area, biased toward center)
        h_img, w_img = image.shape[:2]
        cX_img = w_img / 2.0
        cY_img = h_img / 2.0
        
        def score_contour(c):
            area = cv2.contourArea(c)
            M = cv2.moments(c)
            if M["m00"] == 0:
                return -np.inf
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            dist2 = (cx - cX_img) ** 2 + (cy - cY_img) ** 2
            return area - self.center_bias * dist2
        
        main_contour = max(filtered, key=score_contour)
        
        # Get rotated bounding box to extract 4 corners
        rect = cv2.minAreaRect(main_contour)
        box_points = cv2.boxPoints(rect)  # Returns 4 corners: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        
        # Compute midpoint (centroid) of the 4 corners
        midpoint = np.mean(box_points, axis=0)  # Shape: (2,) - [u, v]
        
        # Return as single feature point (reshape to (1, 2) for consistency)
        # If N_features > 1, pad with zeros; if N_features == 1, return just the midpoint
        if self.N_features == 1:
            return midpoint.reshape(1, 2)
        else:
            # For N_features > 1, return midpoint as first feature, pad rest with zeros
            features = np.zeros((self.N_features, 2))
            features[0] = midpoint
            return features
    
    def _calc_features(self, context, output):
        """Compute feature coordinates from input image."""
        image_value = self.image_input.Eval(context)
        
        # Handle different input types
        if isinstance(image_value, ImageRgba8U):
            # Convert Drake image to numpy
            if image_value.size() == 0:
                output.SetFromVector(np.zeros(2 * self.N_features))
                return
            
            image_array = np.array(image_value.data).reshape(
                image_value.height(), image_value.width(), 4
            )[:, :, :3]  # Drop alpha channel
        elif isinstance(image_value, np.ndarray):
            image_array = image_value
        else:
            # Try to convert to numpy
            try:
                image_array = np.array(image_value)
            except:
                output.SetFromVector(np.zeros(2 * self.N_features))
                return
        
        # Extract features
        features = self._extract_features(image_array)
        
        # Flatten to [u1, v1, u2, v2, ...]
        features_flat = features.flatten()
        output.SetFromVector(features_flat)
    
    def _calc_feature_count(self, context, output):
        """Compute number of features found."""
        image_value = self.image_input.Eval(context)
        
        # Handle different input types
        if isinstance(image_value, ImageRgba8U):
            if image_value.size() == 0:
                output.SetFromVector(np.array([0.0]))
                return
            
            image_array = np.array(image_value.data).reshape(
                image_value.height(), image_value.width(), 4
            )[:, :, :3]
        elif isinstance(image_value, np.ndarray):
            image_array = image_value
        else:
            try:
                image_array = np.array(image_value)
            except:
                output.SetFromVector(np.array([0.0]))
                return
        
        # Extract features to count them
        features = self._extract_features(image_array)
        
        # Count non-zero features
        n_features = np.sum(np.any(features != 0, axis=1))
        output.SetFromVector(np.array([float(n_features)]))

