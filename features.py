"""
Feature Extraction from RGB Images

Implements a Drake LeafSystem that extracts 2D feature points from RGB images
using OpenCV. Currently uses Shi-Tomasi corner detection for stable features.

Later, this will be extended to track specific object corners (e.g., cube vertices).
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
    LeafSystem that extracts N feature points from an RGB image.
    
    Input:
        - rgb_image: AbstractValue containing ImageRgba8U or numpy array
    
    Output:
        - features: Vector of size 2N representing N feature points [u1, v1, u2, v2, ...]
        - feature_count: Scalar indicating number of features found
    """
    
    def __init__(self, N_features=4, max_corners=100):
        """
        Initialize feature tracker.
        
        Args:
            N_features: Target number of features to extract
            max_corners: Maximum corners to detect (will select top N_features)
        """
        LeafSystem.__init__(self)
        
        self.N_features = N_features
        self.max_corners = max_corners
        
        # Shi-Tomasi corner detection parameters
        self.max_corners_to_detect = max(max_corners, N_features * 2)
        self.quality_level = 0.01
        self.min_distance = 10
        self.block_size = 3
        self.use_harris = False
        self.k = 0.04
        
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
        Extract features from image using Shi-Tomasi corner detection.
        
        Args:
            image: numpy array (H, W, 3) uint8 RGB image
        
        Returns:
            numpy array: (N, 2) array of [u, v] coordinates, padded to N_features
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Detect corners
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners_to_detect,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size,
            useHarrisDetector=self.use_harris,
            k=self.k
        )
        
        if corners is None or len(corners) == 0:
            # Return zeros if no features found
            return np.zeros((self.N_features, 2))
        
        # Convert to (N, 2) format
        corners = corners.reshape(-1, 2)
        
        # Select top N_features by quality (already sorted by quality)
        n_found = min(len(corners), self.N_features)
        selected = corners[:n_found]
        
        # Pad to N_features if needed
        if n_found < self.N_features:
            padding = np.zeros((self.N_features - n_found, 2))
            selected = np.vstack([selected, padding])
        
        return selected
    
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

