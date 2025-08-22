import time
import rclpy
import socket
import time
import threading

from dataclasses import dataclass, field
from enum import Enum 
from copy import deepcopy
from PIL import Image
from geometry_msgs.msg import Pose, Point, Quaternion

try:
    from src.execute.ros_communication import RosVlmNode
    from src.execute.target_pose_finder import get_relative_target_pose_from_image, relative_to_map_pose
except ImportError:
    from ros_communication import RosVlmNode
    from target_pose_finder import get_relative_target_pose_from_image, relative_to_map_pose

try:    
    from src.VLM_agent.agent import VLM_agent
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(Path(__file__).parent.parent.parent.as_posix())
    from src.VLM_agent.agent import VLM_agent


class Direction(Enum):
    LEFT       = "left"
    RIGHT      = "right"
    FORWARD    = "forward"
    BACKWARD   = "backward"


@dataclass
class PerceiveAction:
    target: str    = "Default Target"

@dataclass
class StopAction:
    duration: float = 1.0

@dataclass
class MoveAction:
    speed: float                          = 0.5
    distance: float                       = 0.0
    direction: Direction                  = Direction.FORWARD
    execution_time: float                 = field(init=False, repr=False)
    udp_cmd: tuple[float, float, float]   = field(init=False, repr=False)

    def __post_init__(self):
        self.speed = abs(self.speed)
        self.execution_time = self.distance / self.speed
        match self.direction:
            case Direction.FORWARD:
                self.udp_cmd = (self.speed, 0.0, 0.0)
            case Direction.BACKWARD:
                self.udp_cmd = (-self.speed, 0.0, 0.0)
            case Direction.LEFT:
                self.udp_cmd = (0.0, self.speed, 0.0)
            case Direction.RIGHT:
                self.udp_cmd = (0.0, -self.speed, 0.0)
            case _:
                self.udp_cmd = (0.0, 0.0, 0.0)


@dataclass
class TurnAction:
    speed: float                          = 0.5
    angle: float                          = 0.0
    direction: Direction                  = Direction.LEFT
    execution_time: float                 = field(init=False, repr=False)
    udp_cmd: tuple[float, float, float]   = field(init=False, repr=False)

    def __post_init__(self):
        self.angle = abs(self.angle)
        angle_radians = self.angle / 180 * 3.14
        self.execution_time = angle_radians / self.speed

        if self.direction == Direction.LEFT:
            self.udp_cmd = (0.0, 0.0, self.speed)
        elif self.direction == Direction.RIGHT:
            self.udp_cmd = (0.0, 0.0, -self.speed)
        else:
            self.udp_cmd = (0.0, 0.0, 0.0)


class ActionExecutor:
    def __init__(self, address: str = "127.0.0.1", port: int = 5555, image_path: str = "images/example.jpg"):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = (address, port)
        self.image_path = image_path
        self.retry_count = 0

        rclpy.init()
        self.node = RosVlmNode()
        self._spin_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self._spin_thread.start()

    def execute_sequence(self, actions: list[TurnAction | MoveAction | PerceiveAction | StopAction]) -> bool:
        success = False
        for action in actions:
            print(f"Executing -> {action}")
            match action:
                case PerceiveAction():
                    success = self.execute_preceive(action)
                case MoveAction() | TurnAction():
                    success = self.execute_move(action)
                case StopAction():
                    success = self.execute_stop(action)
                case _:
                    success = False

            if not success:
                print(f"Aborting action sequence due to failure at action {action}.")
                return False
        
            time.sleep(0.5)
        
        print("Action sequence completed.")
        return True

    def execute_preceive(self, action: PerceiveAction, max_retries=6):
        while True:
            rgb, depth, cam_info, robot_pose = self.node.get_current_data()
            
            # Demo robot pose
            # robot_pose = Pose(position=Point(x=0.0, y=1.0, z=1.0), orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0))

            none_vals = [key for key, val in zip(
                ["rgb", "depth", "cam_info", "robot_pose"], [rgb, depth, cam_info, robot_pose]) if val is None]

            if not none_vals:
                break
            else:
                print(f"Not all relevant data available: {none_vals}. Waiting...")
                time.sleep(1)

        rgb_im = Image.fromarray(rgb)
        rgb_im.save(self.image_path)
        
        success, image_target_point = VLM_agent(action.target, image_path=self.image_path)

        if success: 
            rel_target_pose = get_relative_target_pose_from_image(image_target_point, rgb, depth, cam_info, window_size=10)
            abs_target_pose = relative_to_map_pose(robot_pose, rel_target_pose)

            print(f"Robot pose: {robot_pose}")
            print(f"Relative target pose: {rel_target_pose}")
            print(f"Absolute target pose: {abs_target_pose}")
            self.node.publish_target_pose(abs_target_pose)
            self.retry_count = 0

            while self.node.get_path_execution_running():
                time.sleep(1)
            return True
    
        if self.retry_count < max_retries:
            self.retry_count += 1
            return self.execute_sequence([TurnAction(angle=60), action])

        return False

    def execute_move(self, action: MoveAction | TurnAction) -> bool:
        self.send_udp_cmd(*action.udp_cmd)
        time.sleep(action.execution_time)
        self.send_udp_cmd(0.04, 0.0, 0.0)
        time.sleep(1)
        return True

    def execute_stop(self, action: StopAction) -> bool:
        self.send_udp_cmd(0.0, 0.0, 0.0)
        time.sleep(action.duration)
        return True

    def send_udp_cmd(self, vx, vy, wz):
        try:
            message = f"{vx} {vy} {wz}"
            data = message.encode('utf-8')
            self.sock.sendto(data, self.addr)
        except Exception as e:
            print(f"Error sending UDP command: {e}")

    def shutdown(self):
        """Clean shutdown of ros spinners"""
        try:
            rclpy.shutdown()
        except Exception as e:
            print(f"Warning during rclpy.shutdown(): {e}")
        try:
            self._spin_thread.join(timeout=1.0)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()





if __name__ == "__main__":
    actions = [
        PerceiveAction(target="plant"),
        MoveAction(speed=0.5, distance=1.0, direction=Direction.FORWARD),
        TurnAction(speed=0.2, angle=90.0, direction=Direction.LEFT),
        StopAction(duration=1.0),
    ]
    with ActionExecutor(image_path="images/rgb.jpg", port=12345) as action_exec:
        action_exec.execute_sequence(actions)