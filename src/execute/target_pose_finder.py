import numpy as np

from geometry_msgs.msg import Pose, Quaternion, Point
from sensor_msgs.msg import CameraInfo


def get_relative_target_pose_from_image(
        pixel_position: tuple[int, int],
        rgb: np.ndarray, 
        depth: np.ndarray, 
        cam_info: CameraInfo,
        window_size: int = 10,
        depth_scale: float | None = None
) -> Point | None:
    if rgb is None or depth is None:
        return None

    if rgb.shape[:2] != depth.shape[:2]:
        raise ValueError("RGB and depth images must have the same dimensions.")
    
    if depth_scale is None:
        if depth.dtype == np.uint16:
            depth_scale = 0.001
        else:
            depth_scale = 1.0

    x, y = pixel_position
    h, w = depth.shape[:2]

    half = window_size // 2
    x0 = max(0, x - half)
    x1 = min(w, x + half + 1)
    y0 = max(0, y - half)
    y1 = min(h, y + half + 1)
    window = depth[y0:y1, x0:x1]
    valid_mask = np.isfinite(window) & (window >= 0)

    if not np.any(valid_mask):
        return None

    mean_raw = float(np.mean(window[valid_mask]))
    z = mean_raw * float(depth_scale)
    K = cam_info.k
    fx, fy = K[0], K[4]
    cx, cy = K[2], K[5]

    if fy == 0 or fx == 0:
        return None

    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy
    return Point(x=X, y=Y, z=z)


def transform_relative_to_map(robot_pose: Pose, relative_point: Point) -> Pose:
    if robot_pose is None or relative_point is None:
        print("Robot Pose or Relative Point is None.")
        return None

    robot_pos = np.array([robot_pose.position.x, robot_pose.position.y, robot_pose.position.z])
    robot_quat = np.array([robot_pose.orientation.x, robot_pose.orientation.y, robot_pose.orientation.z, robot_pose.orientation.w])
    relative_vec = np.array([relative_point.x, relative_point.y, relative_point.z])

    q_vec = robot_quat[:3]
    q_w = robot_quat[3]
    t = 2 * np.cross(q_vec, relative_vec)
    rotated_vec = relative_vec + q_w * t + np.cross(q_vec, t)

    absolute_pos = robot_pos + rotated_vec
    absolute_target_pose = Pose(
        position=Point(x=absolute_pos[0], y=absolute_pos[1], z=absolute_pos[2]),
        orientation=robot_pose.orientation
    )
    
    return absolute_target_pose


if __name__ == "__main__":
    # Example usage
    rgb_image = np.zeros((480, 640, 3), dtype=np.uint8)
    depth_image = np.zeros((480, 640), dtype=np.float32)
    pixel_coords = (640, 480)

    caminfo = CameraInfo()

    target_pose = get_relative_target_pose_from_image(pixel_coords, rgb_image, depth_image, caminfo)
    print("Estimated target pose:", target_pose)