import os

import matplotlib.pyplot as plt
import torch
from src.utils.get_model_and_data import get_model_and_data
from src.parser.visualize import parser
from visualize import viz_epoch

# import src.utils.fixseed  # noqa

plt.switch_backend('agg')


def main():
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
    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    state_dict = torch.load(checkpointpath, map_location=parameters["device"])
    model.load_state_dict(state_dict)
    
    # visualize_params
    viz_epoch(model, dataset, epoch, parameters, folder=folder, writer=None)


if __name__ == '__main__':
    main()
