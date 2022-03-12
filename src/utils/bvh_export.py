import os.path

from fairmotion.core.motion import Motion
from fairmotion.data import bvh
from fairmotion.ops.conversions import R2T

from src.datasets.default_skeleton import DefaultSkeleton
import src.utils.rotation_conversions as geometry


def save_generated_motion(generation, mask, save_path, params):
    if not ((params['dataset'] == 'datagen') & (params['pose_rep'] == 'rot6d')):
        return
    skl = DefaultSkeleton('src/datasets/skeleton.pkl')
    if params['translation']:
        x_translations = generation[:, -1, :3]
        x_rotations = generation[:, :-1]
    else:
        x_rotations = generation

    x_rotations = x_rotations.permute(0, 3, 1, 2)
    nsamples, time, njoints, feats = x_rotations.shape
    for i in range(nsamples):
        rotations = geometry.rotation_6d_to_matrix(x_rotations[i][mask[i]])
        rotations = rotations.clone().detach().cpu().numpy()
        new_motion = Motion.from_matrix(R2T(rotations), skl.skeleton)
        bvh.save(new_motion, os.path.join(save_path, f'gen_motion_{i}.bvh'))


