import matplotlib.pyplot as plt
import os
import pandas as pd
import plotly.express as px

from src.datasets.get_dataset import get_dataset
from src.parser.visualize import parser
from src.visualize.visualize import viz_dataset

plt.switch_backend('agg')


def build_dataset_dist(dataset):
    frames_by_class = zip(dataset._actions, dataset._num_frames_in_video)
    frames_by_class_df = pd.DataFrame(frames_by_class, columns=['action', 'frames'])
    frames_by_class_df.replace({"action":  dataset._action_classes}, inplace=True)
    fig = px.histogram(frames_by_class_df, x="frames", color="action", barmode='overlay', title='frames by action')
    fig.write_html(os.path.join("datavisualize", 'HumanAct12_frames_by_action.html'))


if __name__ == '__main__':
    # parse options
    # parameters = optutils.visualize_dataset_parser()
    parameters = parser(checkpoint=False)
    parameters['num_frames'] = 55
    parameters['fps'] = 10
    # parameters['pose_rep'] = 'xyz'
    # parameters['pose_rep'] = 'rot6d'
    # get device
    device = parameters["device"]

    # get data
    DATA = get_dataset(name=parameters["dataset"])
    dataset = DATA(split="train", **parameters)
    # build_dataset_dist(dataset)
    # add specific parameters from the dataset loading
    dataset.update_parameters(parameters)

    name = f"{parameters['dataset']}_{parameters['pose_rep']}"
    folder = os.path.join("datavisualize", name)
    viz_dataset(dataset, parameters, folder)
