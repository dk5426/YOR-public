import time
import numpy as np
from typing import Optional
from pathlib import Path

from piperlib import PiperJointController, RobotConfigFactory, ControllerConfigFactory, JointState, Gain
import mink

from robot.arm.ik_solver import SingleArmIK

from robot.arm.gripper import Gripper

GRIPPER_OPEN = 0.07
# this is caliberated in the dxl.py file
# DYNAMIXEL_GRIPPER_OPEN = 3100
# DYNAMIXEL_GRIPPER_CLOSED = 1200
# GRIPPER_RANGE = DYNAMIXEL_GRIPPER_OPEN - DYNAMIXEL_GRIPPER_CLOSED

# this comes from the dynamixel wizard
BAUDRATE = 115200
DXL_ID_RIGHT = 2
DXL_ID_LEFT = 3

class ArmNode:
    def __init__(
        self,
        can_port: str,
        mjcf_path: str,
        urdf_path: Optional[str] = None,
        solver_dt: float = 0.01,
        is_left_arm: bool = True,
        dynamixel_gripper: bool = True,
    ):
        _HERE = Path(__file__).parent
        self.can_port = can_port
        self.is_left_arm = is_left_arm
        # if mjcf_path is None:
        #     if is_left_arm:
        #         self.mjcf_path = (_HERE / "mujoco/scene_piper_left.xml").as_posix()
        #     else:
        #         self.mjcf_path = (_HERE / "mujoco/scene_piper_right.xml").as_posix()
        # else:
        #     self.mjcf_path = mjcf_path
        if urdf_path is None:
            if is_left_arm:
                self.urdf_path = (_HERE / "urdf/piper_description_left.xml").as_posix()
            else:
                self.urdf_path = (_HERE / "urdf/piper_description_right.xml").as_posix()
        else:
            self.urdf_path = urdf_path
        self.solver_dt = solver_dt

        # initialize arm
        self.robot_config = RobotConfigFactory.get_instance().get_config("piper")
        self.controller_config = ControllerConfigFactory.get_instance().get_config("joint_controller")
        self.robot_config.urdf_path = self.urdf_path
        self.robot_config.joint_vel_max = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        self.controller_config.controller_dt = 0.003
        self.controller_config.default_kp = np.array([2.5, 2.5, 2.5, 2.5, 2.5, 2.5])
        self.controller_config.default_kd = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        self.controller_config.gravity_compensation = True
        self.controller_config.interpolation_method = "linear"
        self.piper = PiperJointController(self.robot_config, self.controller_config, self.can_port)
        self.target: Optional[mink.SE3] = None
        self.gripper_target: Optional[float] = None
        self.target_timestamp: Optional[int] = None
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
            self.home_q = np.array([0.0, 1.58065, -0.578175, 0.0, -0.912, 0.78])
            # self.home_q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            joint_names = [
                "right_arm_joint1",
                "right_arm_joint2",
                "right_arm_joint3",
                "right_arm_joint4",
                "right_arm_joint5",
                "right_arm_joint6",
            ]
            ee_frame = "right_arm_ee"
            self.home_q = np.array([0.0, 1.58065, -0.578175, 0.0, -0.912, -0.78])

        self.ik_solver = SingleArmIK(
            mjcf_path,
            solver_dt=self.solver_dt,
            joint_names=joint_names,
            ee_frame=ee_frame,
        )

        self.dynamixel_gripper = dynamixel_gripper
        if self.dynamixel_gripper:
            dxl_id = DXL_ID_LEFT if is_left_arm else DXL_ID_RIGHT
            self.gripper = Gripper(baudrate=BAUDRATE, dxl_id=dxl_id)
            open_gripper_value, close_gripper_value = self.gripper.dxl.calibrate_motor()
            self.open_gripper_value = open_gripper_value
            self.close_gripper_value = close_gripper_value
            self.gripper_range = open_gripper_value - close_gripper_value

    def init(self):
        self.reset()
        q = self.piper.get_joint_state().pos - self.q_offset
        # q = self.piper.get_joint_state().pos
        print(f"q_reached: {np.round(q, 4)}")
        self.ik_solver.init(q)
        self.target = self.ik_solver.forward_kinematics()

    def reset(self):
        self.piper.reset_to_home()
        time.sleep(1.0)

    def home(self, gripper_target: float = 1.0):
        q = self.home_q.copy()
        cmd = JointState(self.robot_config.joint_dof)
        cmd.timestamp = self.piper.get_timestamp() + 1.0
        cmd.pos = q + self.q_offset
        cmd.gripper_pos = gripper_target * GRIPPER_OPEN
        self.piper.set_joint_cmd(cmd)
        if self.dynamixel_gripper:
            self.gripper.move_to_pos(int(gripper_target * self.gripper_range + self.close_gripper_value))
        time.sleep(2.0)
        self.update_joint_positions()
        # curr_pos = self.get_ee_pose()
        # if self.is_left_arm:
        #     translation = mink.SE3.from_translation(np.array([0.05,-0.05,0]))
        # else:
        #     translation = mink.SE3.from_translation(np.array([0.05,0.05,0]))
        # self.set_ee_target(translation @ curr_pos,preview_time = 1)

    def tuck_arms(self):
        self.set_joint_target(np.zeros(6), gripper_target=1.00, preview_time=2.0)

    def set_joint_target(
        self, joint_target: np.ndarray, gripper_target: float | None = None, preview_time: float = 0.1
    ):
        cmd = JointState(self.robot_config.joint_dof)
        cmd.pos = joint_target + self.q_offset
        cmd.timestamp = self.piper.get_timestamp() + preview_time
        if gripper_target is not None:
            cmd.gripper_pos = gripper_target * GRIPPER_OPEN
        self.piper.set_joint_cmd(cmd)
        if gripper_target is not None and self.dynamixel_gripper:
            self.gripper.move_to_pos(int(gripper_target * self.gripper_range + self.close_gripper_value))

    def open_gripper(self):
        if not self.dynamixel_gripper:
            q = self.get_joint_positions()
            self.set_joint_target(q, gripper_target=1.0) #WHY IS THIS NOT 1.0 * GRIPPER_OPEN?
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

    def set_ee_target(self, ee_target: mink.SE3, gripper_target: Optional[float] = None, preview_time: float = 0.01):
        self.target = ee_target
        qd, _ = self.ik_solver.solve_ik(self.target)
        cmd = JointState(self.robot_config.joint_dof)
        cmd.pos = qd + self.q_offset
        if gripper_target is not None:
            cmd.gripper_pos = gripper_target * GRIPPER_OPEN
        cmd.timestamp = self.piper.get_timestamp() + preview_time
        self.piper.set_joint_cmd(cmd)
        if gripper_target is not None and self.dynamixel_gripper:
            past = time.time()
            self.gripper.move_to_pos(int(gripper_target * (self.gripper_range) + self.close_gripper_value))
            print(f"it took {past - time.time()})")

    def set_gain(self, kp: np.ndarray, kd: np.ndarray):
        gain = Gain(kp, kd)
        self.piper.set_gain(gain)

    def get_joint_positions(self) -> np.ndarray:
        return self.piper.get_joint_state().pos - self.q_offset
        # return self.piper.get_joint_state().pos

    def get_ee_pose(self) -> mink.SE3:
        self.update_joint_positions()
        return self.ik_solver.forward_kinematics()
    
    def get_gripper_pose(self):
        if not self.dynamixel_gripper:
            gripper_pos = JointState(self.robot_config.joint_dof).gripper_pos
            return gripper_pos
        else:
            unnormalized_pos = self.gripper.dxl.get_present_position()
            return float(unnormalized_pos - self.close_gripper_value)/float(self.gripper_range)

    def update_joint_positions(self):
        q = self.piper.get_joint_state().pos - self.q_offset
        # q = self.piper.get_joint_state().pos
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
        mjcf_path=(_HERE / "cone-e-description/robot-welded-base-and-lift.mjcf").as_posix()
    )
    left_arm.init()
    input("Press enter to home left arm")
    left_arm.home()
    input("Press enter to close gripper")
    left_arm.close_gripper()
    input("Press enter to open gripper")
    left_arm.open_gripper()
    input("Press enter to exit")
    exit()