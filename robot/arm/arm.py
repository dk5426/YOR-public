import time
from pathlib import Path
from typing import Optional

import mink
import numpy as np

from piperlib import ControllerConfig, PiperController
from robot.arm.gripper import Gripper
from robot.arm.ik_solver import SingleArmIK

GRIPPER_OPEN = 0.07
# this is caliberated in the dxl.py file
# DYNAMIXEL_GRIPPER_OPEN = 3100
# DYNAMIXEL_GRIPPER_CLOSED = 1200
# GRIPPER_RANGE = DYNAMIXEL_GRIPPER_OPEN - DYNAMIXEL_GRIPPER_CLOSED

# this comes from the dynamixel wizard
BAUDRATE = 1000000 #115200
DXL_ID_RIGHT = 3
DXL_ID_LEFT = 2


class ArmNode:
    def __init__(
        self,
        can_port: str,
        mjcf_path: str,
        solver_dt: float = 0.01,
        is_left_arm: bool = True,
        dynamixel_gripper: bool = True,
    ):
        _HERE = Path(__file__).parent
        self.can_port = can_port
        self.is_left_arm = is_left_arm
        if is_left_arm:
            self.urdf_path = (_HERE / "../../piperlib/urdf/piper_cone-e_left.urdf").as_posix()
        else:
            self.urdf_path = (_HERE / "../../piperlib/urdf/piper_cone-e_right.urdf").as_posix()
        self.solver_dt = solver_dt

        # initialize arm
        self.controller_config = ControllerConfig()
        self.controller_config.interface_name = can_port
        self.controller_config.urdf_path = self.urdf_path
        self.controller_config.gravity_compensation = False
        self.controller_config.default_kp = np.array([15.0, 15.0, 15.0, 15.0, 15.0, 15.0])
        self.controller_config.controller_freq_hz = 200
        self.controller_config.gripper_on = (
            False
        ) 
        self.controller_config.home_position = (
            [0.0, 1.58065, -0.578175, 0.0, -0.912, 0.78]
            if is_left_arm
            else [0.0, 1.58065, -0.578175, 0.0, -0.912, -0.78]
        )
        self.piper = PiperController(self.controller_config)

        self.target: Optional[mink.SE3] = None
        self.gripper_target: Optional[float] = None
        self.q_offset = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        if is_left_arm:
            joint_names = [
                "left_arm_joint1",
                "left_arm_joint2",
                "left_arm_joint3",
                "left_arm_joint4",
                "left_arm_joint5",
                "left_arm_joint6",
            ]
            ee_frame = "left_arm_ee_iphone"
        else:
            joint_names = [
                "right_arm_joint1",
                "right_arm_joint2",
                "right_arm_joint3",
                "right_arm_joint4",
                "right_arm_joint5",
                "right_arm_joint6",
            ]
            ee_frame = "right_arm_ee_iphone"

        self.ik_solver = SingleArmIK(
            mjcf_path,
            solver_dt=self.solver_dt,
            joint_names=joint_names,
            ee_frame=ee_frame,
        )

        self.dynamixel_gripper = dynamixel_gripper
        if self.dynamixel_gripper:
            # print("skip gripper")
            dxl_id = DXL_ID_LEFT if is_left_arm else DXL_ID_RIGHT
            self.gripper = Gripper(baudrate=BAUDRATE, dxl_id=dxl_id)
            open_gripper_value, close_gripper_value = self.gripper.dxl.calibrate_motor()
            self.open_gripper_value = open_gripper_value
            self.close_gripper_value = close_gripper_value
            self.gripper_range = open_gripper_value - close_gripper_value

    def init(self):
        
        if not self.piper.start():
            raise RuntimeError("Failed to start PiperController")
        self.piper.reset_to_home()
        time.sleep(1.0)

        q = self.piper.get_current_state().pos - self.q_offset
        
        print(f"q_reached: {np.round(q, 4)}")
        
        self.ik_solver.init(q)
        
        self.target = self.ik_solver.forward_kinematics()
        

    def home(self, gripper_target: float = 1.0):
        self.piper.reset_to_home()
        if self.dynamixel_gripper:
            self.gripper.move_to_pos(int(gripper_target * self.gripper_range + self.close_gripper_value))
        time.sleep(2.0)
        self.update_joint_positions()

    def tuck_arms(self):
        self.set_joint_target(np.zeros(6), gripper_target=1.00, preview_time=2.0)

    def set_joint_target(
        self, joint_target: np.ndarray, gripper_target: float | None = None, preview_time: float = 0.1
    ):
        self.piper.set_target(joint_target.tolist(), gripper_target, preview_time)
        if gripper_target is not None and self.dynamixel_gripper:
            self.gripper.move_to_pos(int(gripper_target * self.gripper_range + self.close_gripper_value))

    def open_gripper(self):
        if not self.dynamixel_gripper:
            q = self.get_joint_positions()
            self.set_joint_target(q, gripper_target=1.0)  # WHY IS THIS NOT 1.0 * GRIPPER_OPEN?
            time.sleep(0.5)
        else:
            self.gripper.move_to_pos(self.open_gripper_value)

    def close_gripper(self):
        if not self.dynamixel_gripper:
            q = self.get_joint_positions()
            self.set_joint_target(q, gripper_target=0.0)
            time.sleep(0.5)
        else:
            self.gripper.move_to_pos(self.close_gripper_value)

    def set_ee_target(self, ee_target: mink.SE3, gripper_target: Optional[float] = None, preview_time: float = 0.0):
        self.target = ee_target
        qd, _ = self.ik_solver.solve_ik(self.target)
        self.set_joint_target(qd, gripper_target, preview_time)
        if gripper_target is not None and self.dynamixel_gripper:
            past = time.time()
            self.gripper.move_to_pos(int(gripper_target * (self.gripper_range) + self.close_gripper_value))
            print(f"it took {past - time.time()})")

    def get_joint_positions(self) -> np.ndarray:
        return self.piper.get_current_state().pos - self.q_offset

    def get_ee_pose(self) -> mink.SE3:
        self.update_joint_positions()
        return self.ik_solver.forward_kinematics()

    def get_gripper_pose(self):
        if not self.dynamixel_gripper:
            joint_state = self.piper.get_current_state()
            return joint_state.gripper_pos
        else:
            unnormalized_pos = self.gripper.dxl.get_present_position()
            return float(unnormalized_pos - self.close_gripper_value) / float(
                self.gripper_range
            )  # TODO: why is this not 0.0-1.0?

    def update_joint_positions(self):
        joint_state = self.piper.get_current_state()
        q = joint_state.pos - self.q_offset
        self.ik_solver.update_configuration(q)

    def set_q_offset(self, q_offset: np.ndarray):
        self.q_offset = q_offset

    def stop(self):
        print("called stop")
        pass


if __name__ == "__main__":
    _HERE = Path(__file__).parent.parent
    left_arm = ArmNode(
        can_port="can_left",
        mjcf_path=(_HERE / "cone-e-description/robot-welded-base-and-lift.mjcf").as_posix(),
        is_left_arm=True,
    )
    right_arm = ArmNode(
        can_port="can_right",
        mjcf_path=(_HERE / "cone-e-description/robot-welded-base-and-lift.mjcf").as_posix(),
        is_left_arm=False,
    )
    input("Press enter to init left arm")
    left_arm.init()
    input("Press enter to init right arm")
    #right_arm.init()
    input("Press enter to close gripper")
    left_arm.close_gripper()
    #right_arm.close_gripper()
    input("Press enter to open gripper")
    left_arm.open_gripper()
    #right_arm.open_gripper()
    input("Press enter to exit")
    exit()
