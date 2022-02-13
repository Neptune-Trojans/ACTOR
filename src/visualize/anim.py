import numpy as np
import torch
import imageio

# from action2motion
# Define a kinematic tree for the skeletal struture
humanact12_kinematic_chain = [[0, 1, 4, 7, 10],
                              [0, 2, 5, 8, 11],
                              [0, 3, 6, 9, 12, 15],
                              [9, 13, 16, 18, 20, 22],
                              [9, 14, 17, 19, 21, 23]]  # same as smpl

smpl_kinematic_chain = humanact12_kinematic_chain

mocap_kinematic_chain = [[0, 1, 2, 3],
                         [0, 12, 13, 14, 15],
                         [0, 16, 17, 18, 19],
                         [1, 4, 5, 6, 7],
                         [1, 8, 9, 10, 11]]

vibe_kinematic_chain = [[0, 12, 13, 14, 15],
                        [0, 9, 10, 11, 16],
                        [0, 1, 8, 17],
                        [1, 5, 6, 7],
                        [1, 2, 3, 4]]

datagen_kinematic_chain = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9],
    [0, 10],
    [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    [17, 22, 23, 24, 25],
    [17, 26, 27, 28, 29],
    [17, 30, 31, 32, 33],
    [30, 34, 35, 36],
    [13, 37, 38, 39, 40, 41, 42, 43, 44],
    [40, 45, 46, 47, 48],
    [40, 49, 50, 51, 52],
    [40, 53, 54, 55, 56],
    [53, 57, 58, 59],
    [13, 60, 61]]

action2motion_kinematic_chain = vibe_kinematic_chain


def add_shadow(img, shadow=15):
    img = np.copy(img)
    mask = img > shadow
    img[mask] = img[mask] - shadow
    img[~mask] = 0
    return img


def load_anim(path, timesize=None):
    data = np.array(imageio.mimread(path, memtest=False))[..., :3]
    if timesize is None:
        return data
    # take the last frame and put shadow repeat the last frame but with a little shadow
    lastframe = add_shadow(data[-1])
    alldata = np.tile(lastframe, (timesize, 1, 1, 1))

    # copy the first frames
    lenanim = data.shape[0]
    alldata[:lenanim] = data[:lenanim]
    return alldata


def plot_3d_motion(motion, length, save_path, params, title="", interval=50):
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401
    from matplotlib.animation import FuncAnimation, writers  # noqa: F401
    # import mpl_toolkits.mplot3d.axes3d as p3
    matplotlib.use('Agg')
    pose_rep = params["pose_rep"]

    fig = plt.figure(figsize=[2.6, 2.8])
    ax = fig.add_subplot(111, projection='3d')
    # ax = p3.Axes3D(fig)
    # ax = fig.gca(projection='3d')

    def init():
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.4, 0.4)
        ax.set_zlim(-0.3, 0.3)

        ax.view_init(azim=-90, elev=110)
        # ax.set_axis_off()
        ax.xaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.25)
        ax.yaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.25)
        ax.zaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.25)

    colors = ['red', 'magenta', 'black', 'green', 'blue', 'cyan', 'magenta', 'DarkSeaGreen', 'azure', 'red',
              'orange', 'plum', 'salmon', 'orchid', 'sienna']

    if pose_rep != "xyz":
        raise ValueError("It should already be xyz.")

    if torch.is_tensor(motion):
        motion = motion.numpy()

    # invert axis
    # motion[:, 1, :] = -motion[:, 1, :]
    # motion[:, 2, :] = -motion[:, 2, :]

    """
    Debug: to rotate the bodies
    import src.utils.rotation_conversions as geometry
    glob_rot = [0, 1.5707963267948966, 0]
    global_orient = torch.tensor(glob_rot)
    rotmat = geometry.axis_angle_to_matrix(global_orient)
    motion = np.einsum("ikj,ko->ioj", motion, rotmat)
    """

    if motion.shape[0] == 18:
        kinematic_tree = action2motion_kinematic_chain
    elif motion.shape[0] == 24:
        kinematic_tree = smpl_kinematic_chain
    elif motion.shape[0] == 62:
        kinematic_tree = datagen_kinematic_chain
    else:
        kinematic_tree = None

    def update(index):
        ax.lines = []
        ax.collections = []
        if kinematic_tree is not None:
            for chain, color in zip(kinematic_tree, colors):
                ax.plot(motion[chain, 0, index],
                        motion[chain, 1, index],
                        motion[chain, 2, index], linewidth=4.0, color=color)
        else:
            ax.scatter(motion[1:, 0, index], motion[1:, 1, index],
                       motion[1:, 2, index], c="red")
            ax.scatter(motion[:1, 0, index], motion[:1, 1, index],
                       motion[:1, 2, index], c="blue")

    ax.set_title(title + f', frames: {length}')

    ani = FuncAnimation(fig, update, frames=length, interval=interval, repeat=False, init_func=init)

    plt.tight_layout()
    # pillow have problem droping frames
    ani.save(save_path, writer='ffmpeg', fps=1000/interval)
    plt.close()


def plot_3d_motion_dico(x):
    motion, length, save_path, params, kargs = x
    plot_3d_motion(motion, length, save_path, params, **kargs)
