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
import load_blender
from OUTPUT.image_tools import tile_images
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


def load_linemod_depth_images(base_directory, depth_directory):
    list_of_image_filenames = load_llff.get_list_of_image_filenames(base_directory)
    depth_images = []
    for meta_filename in list_of_image_filenames:
        frame_number = int(re.findall(r'\d+', meta_filename)[-1])
        depth_filename = os.path.join(depth_directory, f'depth{frame_number}.dpt')
        assert os.path.isfile(depth_filename), f'mask_filename not found at: {depth_filename}'
        actual_depth = load_blender.linemod_dpt(depth_filename)
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
    for i, T in enumerate(Ts):
        print(T)
        this_bottle = copy.deepcopy(pcd)
        this_bottle.paint_uniform_color([i / len(Ts), i / len(Ts), 0])
        geoms.append(this_bottle.transform(T))
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.])).transform(T))

    o3d.visualization.draw_geometries(geoms)



def test_linemod(args):
    images, poses, render_poses, hwf, i_split, extras = load_blender.load_blender_data(
        args.datadir, args.half_res, args.testskip, args.image_extn, get_depths=False)

    print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
    i_train, i_val, i_tests = i_split

    images_drill, poses_drill, render_poses_drill, hwf_drill, i_split_drill, extras = load_blender.load_blender_data(
        '/home/adam/shared/LINEMOD/nerf/driller', args.half_res, args.testskip, '.jpg', get_depths=True)

    i_train_drill, i_val_drill, i_tests_drill = i_split_drill

    near = 0.
    far = 6.
    args.near = tf.cast(near, tf.float32),
    args.far = tf.cast(far, tf.float32),

    for i_test, i_test_drill in zip(i_tests, i_tests_drill):
        test_pose = poses[i_test, :4, :4]
        plt.imshow(images[i_test,:, :, :3])
        plt.show()
        plt.imshow(images_drill[i_test_drill])
        plt.show()
        test_pose_drill = poses_drill[i_test_drill]
        render_example([test_pose, test_pose_drill])


if __name__ == '__main__':
    parser = rn.config_parser()
    args = parser.parse_args()
    # test_linemod(args)
    images, poses, render_poses, hwf, i_split, extras = load_blender.load_blender_data(
        args.datadir, args.half_res, args.testskip, args.image_extn, get_depths=False)

    print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
    i_train, i_val, i_tests = i_split
    test_pose = poses[i_tests[0]]
    H, W, focal = images[0].shape[0], images[0].shape[1], hwf[2]
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