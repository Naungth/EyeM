"""
Eye-in-Hand RGB-D Camera System

Implements a Drake RgbdSensor attached to the IIWA end-effector frame.
Provides a LeafSystem wrapper that converts Drake's ImageRgba8U to numpy arrays.

The camera is mounted on the end-effector (eye-in-hand configuration) to enable
image-based visual servoing.
"""

import numpy as np
from pydrake.all import (
    AbstractValue,
    CameraInfo,
    ClippingRange,
    ColorRenderCamera,
    DepthRange,
    DepthRenderCamera,
    DiagramBuilder,
    Frame,
    ImageDepth32F,
    ImageRgba8U,
    LeafSystem,
    MakeRenderEngineVtk,
    MultibodyPlant,
    RenderCameraCore,
    RenderEngineVtkParams,
    RgbdSensor,
    RigidTransform,
    RollPitchYaw,
    SceneGraph,
)


class CameraImageConverter(LeafSystem):
    """
    LeafSystem that converts Drake Image types to numpy arrays.
    
    Input ports:
        - rgb_image: AbstractValue containing ImageRgba8U
        - depth_image: AbstractValue containing ImageDepth32F
    
    Output ports:
        - rgb: numpy array (H, W, 3) uint8
        - depth: numpy array (H, W) float32
        - point_cloud: numpy array (H*W, 3) float32 (optional, placeholder)
    """
    
    def __init__(self):
        LeafSystem.__init__(self)
        
        # Input ports
        self.rgb_input = self.DeclareAbstractInputPort(
            "rgb_image",
            AbstractValue.Make(ImageRgba8U())
        )
        
        self.depth_input = self.DeclareAbstractInputPort(
            "depth_image",
            AbstractValue.Make(ImageDepth32F())
        )
        
        # Output ports
        self.rgb_output = self.DeclareVectorOutputPort(
            "rgb",
            size=0,  # Will be set dynamically
            calc=self._calc_rgb
        )
        
        self.depth_output = self.DeclareVectorOutputPort(
            "depth",
            size=0,  # Will be set dynamically
            calc=self._calc_depth
        )
        
        # Placeholder for point cloud (not implemented yet)
        self.point_cloud_output = self.DeclareVectorOutputPort(
            "point_cloud",
            size=0,
            calc=self._calc_point_cloud_placeholder
        )
    
    def _calc_rgb(self, context, output):
        """Convert ImageRgba8U to numpy array."""
        image = self.rgb_input.Eval(context)
        if image.size() == 0:
            output.SetFromVector(np.array([]))
            return
        
        # Convert to numpy
        rgb_array = np.array(image.data).reshape(
            image.height(), image.width(), 4
        )[:, :, :3]  # Drop alpha channel
        
        # Flatten for vector output (will need to reshape in consumer)
        output.SetFromVector(rgb_array.flatten())
    
    def _calc_depth(self, context, output):
        """Convert ImageDepth32F to numpy array."""
        image = self.depth_input.Eval(context)
        if image.size() == 0:
            output.SetFromVector(np.array([]))
            return
        
        # Convert to numpy
        depth_array = np.array(image.data).reshape(
            image.height(), image.width()
        )
        
        # Flatten for vector output
        output.SetFromVector(depth_array.flatten())
    
    def _calc_point_cloud_placeholder(self, context, output):
        """Placeholder for point cloud computation."""
        # TODO: Implement point cloud generation from depth + intrinsics
        output.SetFromVector(np.array([]))


def add_eye_in_hand_camera(
    builder: DiagramBuilder,
    plant: MultibodyPlant,
    scene_graph: SceneGraph,
    ee_frame: Frame,
    camera_name: str = "eye_in_hand_camera"
):
    """
    Add an RGB-D camera mounted on the end-effector frame.
    
    Args:
        builder: DiagramBuilder to add systems to
        plant: MultibodyPlant
        scene_graph: SceneGraph
        ee_frame: Frame to mount camera on (end-effector frame)
        camera_name: Name for the camera system
    
    Returns:
        tuple: (rgbd_sensor, image_converter)
            - rgbd_sensor: The RgbdSensor system
            - image_converter: The CameraImageConverter system
    """
    # Camera pose relative to end-effector
    # Position camera looking straight forward
    rpy = RollPitchYaw(0, 0, 0)  # No rotation - looking straight forward
    p = np.array([0.20, 0.0, 0.10]).reshape(3, 1)  # 20cm forward, 0cm lateral, 10cm up
    X_Camera_EE = RigidTransform(rpy, p)
    
    # Create render engine
    render_engine = MakeRenderEngineVtk(RenderEngineVtkParams())
    renderer_name = "vtk_renderer"
    scene_graph.AddRenderer(renderer_name, render_engine)
    
    # Camera parameters
    width = 640
    height = 480
    fov_y = np.pi / 4  # 45 degrees
    
    # Create camera intrinsics
    intrinsics = CameraInfo(width, height, fov_y)
    
    # Create clipping range (near and far planes)
    clipping = ClippingRange(0.01, 5.0)  # 1cm to 5m
    
    # Camera pose (identity for now, X_PB will handle the transform)
    X_BS = RigidTransform()
    
    # Create render camera core
    camera_core = RenderCameraCore(
        renderer_name,
        intrinsics,
        clipping,
        X_BS
    )
    
    # Create depth range for depth camera
    depth_range = DepthRange(0.01, 5.0)  # 1cm to 5m range
    
    # Create depth render camera
    depth_camera = DepthRenderCamera(camera_core, depth_range)
    
    # Create color render camera
    color_camera = ColorRenderCamera(camera_core, False)
    
    # Get the geometry frame ID for the end-effector body
    # The body should already have a frame registered with the scene graph
    ee_body = ee_frame.body()
    
    # Get the body's frame ID from the scene graph
    # Each body in the plant has a corresponding geometry frame
    body_frame_id = plant.GetBodyFrameIdOrThrow(ee_body.index())
    
    # Add RGB-D sensor attached to the end-effector body's frame
    rgbd_sensor = builder.AddSystem(
        RgbdSensor(
            parent_id=body_frame_id,
            X_PB=X_Camera_EE,
            color_camera=color_camera,
            depth_camera=depth_camera,
        )
    )
    
    # Connect sensor to scene graph
    builder.Connect(
        scene_graph.get_query_output_port(),
        rgbd_sensor.query_object_input_port()
    )
    
    # Add image converter
    image_converter = builder.AddSystem(CameraImageConverter())
    
    # Connect RGB-D sensor outputs to converter
    builder.Connect(
        rgbd_sensor.color_image_output_port(),
        image_converter.rgb_input
    )
    builder.Connect(
        rgbd_sensor.depth_image_32F_output_port(),
        image_converter.depth_input
    )
    
    return rgbd_sensor, image_converter

