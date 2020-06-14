import os, sys
import pprint
import tensorflow as tf
import numpy as np
import imageio
import pprint
import re
import matplotlib.pyplot as plt

import run_nerf as rn
import run_nerf_helpers as rnh
import load_blender
import MODELS.rotations
import MODELS.tf_rotations
import OUTPUT.image_tools
import load_llff

tf.compat.v1.enable_eager_execution()


def render_depth(args, T, H, W, focal):
    # c2w in network
    args.c2w = tf.Variable(
        MODELS.rotations.np_transformation_matrix_to_9d_flat(
            np.expand_dims(T, axis=0)
        ),
        name='c2w')

    render_kwargs_train, render_kwargs_test, start, _, models = rn.create_nerf(args)
    #render original
    render_kwargs_test.update({'near':args.near, 'far':args.far})
    rgb, disp_map, _, extras_network = rn.render(H, W, focal, c2w=np.eye(4, dtype=np.float32)[:3, :4], **render_kwargs_test)
    extras_network.update({'disp_map': disp_map})
    extras_network.update({'rgb': rgb})
    return extras_network


def linemod_dpt(path):
    """
    read a depth image

    @return uint16 image of distance in [mm]"""
    dpt = open(path, "rb")
    rows = np.frombuffer(dpt.read(4), dtype=np.int32)[0]
    cols = np.frombuffer(dpt.read(4), dtype=np.int32)[0]

    return np.fromfile(dpt, dtype=np.uint16).reshape((rows, cols))


def load_linemod_depth_images(base_directory, depth_directory):
    list_of_image_filenames = load_llff.get_list_of_image_filenames(base_directory)
    depth_images = []
    for meta_filename in list_of_image_filenames:
        frame_number = int(re.findall(r'\d+', meta_filename)[-1])
        depth_filename = os.path.join(depth_directory, f'depth{frame_number}.dpt')
        assert os.path.isfile(depth_filename), f'mask_filename not found at: {depth_filename}'
        actual_depth = linemod_dpt(depth_filename) / 1000
        depth_images.append(actual_depth)

    depth_images = np.stack(depth_images, axis=0)
    return depth_images


def plot_depth_diff(depth_map, rendered_depth):
    plt.imshow(tf.abs(depth_map - rendered_depth))
    plt.colorbar()
    plt.show()


def render_example(Ts):
    import open3d as o3d
    import copy
    pcd = o3d.io.read_point_cloud('/home/adam/DATA/SH/objects/spraybottle/spraybottle-100000_centred.ply')
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))
    geoms = [coord]
    for T in Ts:
        print(T)
        geoms.append(copy.deepcopy(pcd).transform(T))

    o3d.visualization.draw_geometries(geoms)


if __name__ == '__main__':
    parser = rn.config_parser()
    args = parser.parse_args()

    images, poses, render_poses, hwf, i_split, depth_maps = load_blender.load_blender_data(
        args.datadir, args.half_res, args.testskip, args.image_extn, get_depths=True)

    print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
    i_train, i_val, i_tests = i_split

    test_example = 0
    i_test = i_tests[test_example]
    near = 0.
    far = 6.
    args.near = tf.cast(near, tf.float32),
    args.far = tf.cast(far, tf.float32),

    H, W, focal = images[0].shape[0], images[0].shape[1], hwf[2]
    test_pose = poses[i_test, :4, :4]
    render_example([test_pose])
    extras = render_depth(args, test_pose, H, W, focal)

    rendered_depth = extras['depth_map'].numpy()
    rendered_disp = extras['disp_map'].numpy()
    inf_map = np.where(rendered_disp == 1.e10)
    rendered_disp[inf_map] = 0
    rgb = extras['rgb']
    plt.imshow(rgb)
    plt.colorbar()
    plt.show()

    plt.imshow(rendered_depth)
    plt.colorbar()
    plt.show()

    disp_map = 1. / depth_maps[test_example]

    plt.imshow(disp_map)
    plt.colorbar()
    plt.show()

    plt.imshow(rendered_depth)
    plt.colorbar()
    plt.show()
    rendered_depth = rendered_depth.numpy()

    plot_depth_diff(disp_map, rendered_disp)

    # inf_map = np.where(depth_map == np.inf)
    # rendered_depth[inf_map] = 0
    # depth_map[inf_map] = 0
    #
    # plt.imshow(depth_map)
    # plt.colorbar()
    # plt.show()
    # #
    # # depth_map, rendered_depth
    # plot_depth_diff(depth_map, rendered_depth)