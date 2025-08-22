import numpy as np
from scipy.spatial.transform import Rotation


class FK:
    def __init__(self):
        transformations = np.array([
            [0.0, 0.0, 0.267, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -1.5708, 0.0, 0.0],
            [0.0, -0.293, 0.0, 1.5708, 0.0, 0.0],
            [0.0525, 0.0, 0.0, 1.5708, 0.0, 0.0],
            [0.0775, -0.3425, 0.0, 1.5708, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.5708, 0.0, 0.0],
            [0.076, 0.097, 0.0, -1.5708, 0.0, 0.0]
        ], dtype=np.float32)

        self.rot_pos_transformations = []
        for transformation in transformations:
            rotvec, pos = transformation[3:], transformation[:3]
            rot = Rotation.from_rotvec(rotvec, degrees=False)
            self.rot_pos_transformations.append((rot, pos))


        self.arm_in_robot_base_pos = np.array([0.425, 0.0, 0.451])

    def __call__(self, joint_angles):
        joint_rotations = [
            Rotation.from_euler('z', angle, degrees=False) for angle in joint_angles
        ]
        rot_result = Rotation.identity()
        pos_result = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        for i, (rot, pos) in enumerate(self.rot_pos_transformations):
            pos_result = pos_result + rot_result.apply(pos)
            rot_result = rot_result * rot
            joint_rot = joint_rotations[i]
            rot_result = rot_result * joint_rot
        pos_result += self.arm_in_robot_base_pos

        return pos_result, rot_result