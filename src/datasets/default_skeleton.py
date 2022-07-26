import pickle
import numpy as np

from fairmotion.core.motion import Motion
from fairmotion.ops.conversions import T2Rp


class DefaultSkeleton:
    def __init__(self, skl_path):
        self._master_skeleton = self.load_master_skeleton(skl_path)

    @property
    def skeleton(self):
        return self._master_skeleton

    @property
    def skeleton_joints_names(self):
        return [joint.name for joint in self._master_skeleton.joints]

    @property
    def joints_parent(self):
        return np.array([-1] +
                        [self.skeleton_joints_names.index(joint.parent_joint.name)
                         for joint in self.skeleton.joints[1:]])

    @property
    def skeleton_joints_ordered(self):
        return {joint_name: idx for idx, joint_name in enumerate(self.skeleton_joints_names)}

    @staticmethod
    def load_master_skeleton(skl_path):
        with open(skl_path, 'rb') as f:
            skl = pickle.load(f)
        return skl

    @staticmethod
    def save_master_skeleton(skl, skl_path):
        with open(skl_path, 'wb') as handle:
            pickle.dump(skl, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def compare_joints_order(skl_a, skl_b):
        bones_names_a = [joint.name for joint in skl_a.joints]
        bones_names_b = [joint.name for joint in skl_b.joints]

        return bones_names_a == bones_names_b

    def sort_bones_by_base_skeleton_order(self, a_motion):
        a_skl = a_motion.skel
        joint_index = self.skeleton_joints_ordered
        joints_order = []
        a_motion_matrix = a_motion.to_matrix()
        new_motion_matrix = np.zeros(a_motion_matrix.shape)

        for old_idx, joint in enumerate(a_skl.joints):
            new_idx = joint_index[joint.name]
            joints_order.append(new_idx)
            new_motion_matrix[:, new_idx, :, :] = a_motion_matrix[:, old_idx, :, :]

        new_motion = Motion.from_matrix(data=new_motion_matrix, skel=self.skeleton)

        return new_motion

    @property
    def skeleton_offsets(self):
        xform_from_parent_joint = []
        for joint in self.skeleton.joints:
            R, p = T2Rp(joint.xform_from_parent_joint)
            xform_from_parent_joint.append(p)
        # xform_from_parent_joint = xform_from_parent_joint / np.linalg.norm(xform_from_parent_joint[1])
        return np.array(xform_from_parent_joint)
