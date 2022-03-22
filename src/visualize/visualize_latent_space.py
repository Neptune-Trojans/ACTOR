import os

import imageio
import numpy as np
import torch
from scipy.spatial.distance import pdist

from src.parser.visualize import parser
from src.utils.bvh_export import save_generated_motion
from src.utils.get_model_and_data import get_model_and_data
from src.visualize.anim import plot_3d_motion


class VisualizeLatentSpace:
    def __init__(self, model, dataset, base_path, reconstruct_pool, synth_to_visualize):
        self._model = model
        self._dataset = dataset
        self._base_path = base_path
        self.reconstruct_pool = reconstruct_pool
        self.synth_to_visualize = synth_to_visualize

    def _visualize_real_image(self, data, idx=0):
        # Get xyz for the real ones
        data["output_xyz"] = self._model.rot2xyz(data["output"], data["mask"])

        motion = data["output_xyz"]
        batch, joints, _, length = motion.shape
        save_path = os.path.join(self._base_path, 'real.gif')
        params = {"pose_rep": 'xyz'}

        plot_3d_motion(motion[idx], length, save_path, params, title='real')
        return save_path

    def _visualize_reconstructed_image(self, reconstructions, idx=0):

        # Reconstruction of the real data
        model(reconstructions['ntf'])  # update reconstruction dicts
        reconstruction = reconstructions[list(reconstructions.keys())[0]]
        motion = reconstruction['output_xyz']
        batch, joints, _, length = motion.shape
        save_path = os.path.join(self._base_path, 'reconstruction.gif')
        params = {"pose_rep": 'xyz'}
        plot_3d_motion(motion[idx], length, save_path, params, title='reconstruction')
        return save_path, reconstruction['z']

    def _generate_new_motion(self, z=None):

        if z is None:
            nspa = self.synth_to_visualize
            z = torch.randn(nspa, 256, device=self._model.device)
        else:
            nspa, latent_space = z.shape

        duration = torch.as_tensor([200])
        durations = duration.repeat((nspa, 1))
        y = torch.as_tensor([0]).to(self._model.device).repeat(nspa)
        lengths = durations.to(self._model.device).reshape(y.shape)
        mask = torch.ones((nspa, 200)).type(torch.bool)

        batch = {"z": z, "y": y, "mask": mask, "lengths": lengths}

        batch = self._model.decoder(batch)
        batch['output_xyz'] = self._model.rot2xyz(batch["output"], batch["mask"])

        return batch

    def save_bvh_motion(self, generated_motion):
        params = {'translation': True,
                  'dataset': 'datagen',
                  "pose_rep": 'rot6d'}
        bvh_base_path = os.path.join(self._base_path, 'gen_bvh')
        save_generated_motion(generated_motion['output'], generated_motion['mask'], bvh_base_path, params)

    def _visualize_synthetic(self, generated_motion):
        params = {'translation': True,
                  'dataset': 'datagen',
                  "pose_rep": 'xyz'}
        batch_size, joints, _, frames = generated_motion['output_xyz'].shape
        motion = generated_motion['output_xyz']
        paths = []
        indexes = np.random.randint(0, batch_size, self.synth_to_visualize)
        for i in indexes:
            gif_path = os.path.join(self._base_path, f'gen_{i}.gif')
            plot_3d_motion(motion[i], frames, gif_path, params, title='gen')
            paths.append(gif_path)
        return paths

    def explore_latent_space(self):

        classes = torch.as_tensor(np.zeros(self.reconstruct_pool), dtype=torch.int64)
        real_samples, mask_real, real_lengths = dataset.get_label_sample_batch(classes.numpy())

        real = {"x": real_samples.to(model.device),
                "y": classes.to(model.device),
                "mask": mask_real.to(model.device),
                "lengths": real_lengths.to(model.device),
                "output": real_samples.to(model.device)}

        reconstructions = {'ntf': {"x": real_samples.to(model.device),
                                   "y": classes.to(model.device),
                                   "lengths": real_lengths.to(model.device),
                                   "mask": mask_real.to(model.device),
                                   "teacher_force": 'ntf' == "tf"}}

        model.eval()
        with torch.no_grad():
            real_path = self._visualize_real_image(real)
            recon_path, recon_z = self._visualize_reconstructed_image(reconstructions)
            z = self._explore_z(recon_z)
            generated_data = self._generate_new_motion(z=z)

        self.save_bvh_motion(generated_data)
        generated_path = self._visualize_synthetic(generated_data)
            # self.compute_diversity(motion)

        all_path = [real_path] + [recon_path] + generated_path
        visualize_gen_data(200, all_path, os.path.join(self._base_path, 'output.gif'))

    @staticmethod
    def compute_diversity(pred, *args):
        if pred.shape[0] == 1:
            return 0.0
        dist = pdist(pred.reshape(pred.shape[0], -1))
        diversity = dist.mean().item()
        return diversity

    def _explore_z(self, rec_z):
        z = torch.randn(self.reconstruct_pool, 256, device=self._model.device)

        for i in range(self.reconstruct_pool):
            indices = np.random.randint(0, self.reconstruct_pool, 4)
            a, b, c, d = indices
            z[i] = torch.cat((rec_z[d][0:64], rec_z[b][64:128], rec_z[c][128:192], rec_z[a][192:256]), 0)
        return z


def visualize_gen_data(number_of_frames, gen_path, output_path):
    # Create reader object for the gif
    gif_readers = [imageio.get_reader(path) for path in gen_path]

    # Create writer object
    new_gif = imageio.get_writer(output_path)

    for frame_number in range(number_of_frames):
        imgs = []
        for reader in gif_readers:
            img = reader.get_next_data()
            imgs.append(img)
        # here is the magic
        new_image = np.hstack(imgs)
        new_gif.append_data(new_image)

    [reader.close() for reader in gif_readers]
    new_gif.close()


if __name__ == '__main__':
    base_path = '/Users/paul.yudkin/Datagen/Applications/visualize_latent_space'
    # parse options
    parameters, folder, checkpointname, epoch = parser()
    parameters['pose_rep'] = 'rot6d'
    parameters['num_frames'] = 200

    parameters['decoder_test'] = "new"
    parameters["noise_same_action"] = 'random'
    parameters['fps'] = 10

    parameters['dataset'] = 'datagen'
    parameters['jointstype'] = 'datagen_skeleton'
    parameters["num_actions_to_sample"] = 1

    model, datasets = get_model_and_data(parameters)
    dataset = datasets["train"]

    checkpointpath = os.path.join(folder, checkpointname)
    state_dict = torch.load(checkpointpath, map_location=parameters["device"])
    model.load_state_dict(state_dict)

    v = VisualizeLatentSpace(model, dataset, base_path, reconstruct_pool=100, synth_to_visualize=6)
    v.explore_latent_space()
