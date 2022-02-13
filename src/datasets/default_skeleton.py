# import os
import pickle
import numpy as np

from fairmotion.core.motion import Motion
# from fairmotion.data import bvh
# from fairmotion.ops.motion import position_wrt_root
#
# from src.kinematic.kinematic_consts import datagen_kinematic_chain
# from src.visualization.viz import plot_3d_motion


class DefaultSkeleton:
    def __init__(self, skl_path):
        self._master_skeleton = self._generate_master_skeleton(skl_path)

    @property
    def skeleton(self):
        return self._master_skeleton

    @property
    def skeleton_joints_names(self):
        return [joint.name for joint in self._master_skeleton.joints]

    @property
    def skeleton_joints_ordered(self):
        return {joint_name: idx for idx, joint_name in enumerate(self.skeleton_joints_names)}

    @staticmethod
    def _generate_master_skeleton(skl_path):
        with open(skl_path, 'rb') as f:
            skl = pickle.load(f)
        return skl

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


# if __name__ == '__main__':
#     motion_b = bvh.load('/bvh_motion_files/03_Coffee_Mug.bvh')
#     # motion_b = bvh.load('/Users/paul.yudkin/dev/motion_synthesis/bvh_motion_files/DGS_X0026.34_Kick_front.bvh')
#     # motion_b = bvh.load('/Users/paul.yudkin/dev/motion_synthesis/bvh_motion_files/13_Agree.bvh')
#     d_skeleton = DefaultSkeleton('master_data/skeleton.pkl')
#     if not d_skeleton.compare_joints_order(motion_b.skel, d_skeleton.skeleton):
#         motion_b = d_skeleton.sort_bones_by_base_skeleton_order(motion_b)
#
#     # datagen_kinematic_chain = build_kinematic_chain(motion_a.skel)
#     # print(datagen_kinematic_chain)
#     matrix = position_wrt_root(motion_b)
#
#     # (frames, joints, xzy) matrix.shape
#     matrix = np.moveaxis(matrix, 0, 2)
#     output_path = '/output'
#     file_name = '11_Take_a_break'
#     animation_path = os.path.join(output_path, file_name + '.gif')
#     plot_3d_motion(matrix, motion_b.num_frames(), animation_path, kinematic_tree=datagen_kinematic_chain)