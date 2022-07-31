import pickle as pkl
import numpy as np
import os
from .dataset import Dataset
from ..datasets.default_skeleton import DefaultSkeleton


class Datagen(Dataset):
    dataname = "datagen"

    def __init__(self, datapath="data/Datagen", **kargs):
        self.datapath = datapath

        super().__init__(**kargs)
        skl_location = 'src/datasets/skeleton.pkl'
        pkldatafilepath = os.path.join(datapath, "skeleton_motion.pkl")
        dataset = pkl.load(open(pkldatafilepath, "rb"))
        data = dataset['motion_data']
        self._pose = [x["rotations"] for x in data]
        self._num_frames_in_video = [p.shape[0] for p in self._pose]
        self._joints = [x["joints3D"] for x in data]

        self._actions = [x["y"] for x in data]
        self._joints_number = len(dataset['base_skeleton'].joints)
        self.base_skeleton = dataset['base_skeleton']
        DefaultSkeleton.save_master_skeleton(self.base_skeleton, skl_location)
        total_num_actions = 1
        self.num_classes = total_num_actions

        self._train = list(range(len(self._pose)))

        keep_actions = np.arange(0, total_num_actions)

        self._action_to_label = {x: i for i, x in enumerate(keep_actions)}
        self._label_to_action = {i: x for i, x in enumerate(keep_actions)}

        self.action_classes = datagen_coarse_action_enumerator

    def _load_joints3D(self, ind, frame_ix):
        return self._joints[ind][frame_ix]

    def _load_rotvec(self, ind, frame_ix):
        pose = self._pose[ind][frame_ix].reshape(-1, self._joints_number, 3)
        return pose


datagen_coarse_action_enumerator = {
    0: "walking"
}
