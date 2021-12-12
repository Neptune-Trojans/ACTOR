import matplotlib.pyplot as plt
# import torch
import os

from src.datasets.get_dataset import get_dataset
from src.parser.visualize import parser
# from src.utils import optutils

import src.utils.fixseed  # noqa
from src.visualize.visualize import viz_dataset

plt.switch_backend('agg')


if __name__ == '__main__':
    # parse options
    # parameters = optutils.visualize_dataset_parser()
    parameters = parser(checkpoint=False)
    parameters['num_frames'] = 55
    parameters['fps'] = 10
    # get device
    device = parameters["device"]

    # get data
    DATA = get_dataset(name=parameters["dataset"])
    dataset = DATA(split="train", **parameters)

    # add specific parameters from the dataset loading
    dataset.update_parameters(parameters)

    name = f"{parameters['dataset']}_{parameters['pose_rep']}"
    folder = os.path.join("datavisualize", name)
    viz_dataset(dataset, parameters, folder)
