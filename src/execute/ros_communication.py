import threading
import numpy as np

from rclpy.node import Node
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped, Pose, Quaternion
from sensor_msgs.msg import Image, CameraInfo


class RosVlmNode(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        
        # Create publishers
        self.target_pub = self.create_publisher(PoseStamped, '/target/pose', 1)

        # Create subscribers
        self.rgb_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw',
            self.rgb_callback, 1)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw',
            self.depth_callback, 1)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/info',
            self.camera_info_callback, 1)
        self.position_sub = self.create_subscription(
            PoseStamped, '/pose',
            self.position_callback, 1)
        self.quat_sub = self.create_subscription(
            Quaternion, '/quat',
            self.quat_callback, 1)
        self.trigger_sub = self.create_subscription(
            Bool, '/mpc/flag',
            self.trigger_callback, 1)
        
        # Thread-safe storage for received data
        self.rgb = None
        self.depth = None
        self.camera_info = None
        self.pose = None
        self.mpc_running = False

        self.data_lock = threading.Lock()
        self.trigger_lock = threading.Lock()

    def rgb_callback(self, msg: Image):
        with self.data_lock:
            self.rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

    def depth_callback(self, msg: Image):
        with self.data_lock:
            self.depth = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

    def camera_info_callback(self, msg: CameraInfo):
        with self.data_lock:
            self.camera_info = msg
        
        try:
            self.destroy_subscription(self.camera_info_sub)
            self.camera_info_sub = None
        except Exception as e:
            self.get_logger().error(f"Error destroying camera_info_sub: {e}")

    def position_callback(self, msg: PoseStamped):
        with self.data_lock:
            self.pose.position = msg.pose.position

    def quat_callback(self, msg: Quaternion):
        with self.data_lock:
            self.pose.orientation = msg

    def trigger_callback(self, msg: Bool):
        with self.trigger_lock:
            self.mpc_running = msg.data

    def publish_target_pose(self, pose: Pose):
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.header.frame_id = "map"
        pose_stamped.pose = pose
        self.target_pub.publish(pose_stamped)
        self.get_logger().info(f"Published target position: {pose.position} and quat {pose.orientation}")

    def get_current_data(self) -> tuple[np.ndarray, np.ndarray, CameraInfo, Pose]:
        """Thread-safe getter for current RGB and depth images"""
        with self.data_lock:
            return self.rgb, self.depth, self.camera_info, self.pose

    def get_path_execution_running(self):
        """Thread-safe getter for MPC running state"""
        with self.trigger_lock:
            return self.mpc_running
