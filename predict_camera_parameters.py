import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

import numpy as np
import imageio
import pprint

import matplotlib.pyplot as plt

import run_nerf as rn
import run_nerf_helpers as rnh
from load_llff import load_llff_data


def optimize_model_to_single_image(args, test_image, render_kwargs_test):
    H, W, focal = 480, 640, 584.
    rays_in_camera_frame = rnh.get_rays_np(H, W, focal, np.eye(4))
    optimizer = tf.keras.optimizers.Adam(args.lrate)
    with tf.GradientTape() as tape:
        c2w = tf.eye(3, 4)
        grad_vars = [c2w]

        # Make predictions for color, disparity, accumulated opacity.
        rgb, disp, acc, extras = rn.render(
            H, W, focal, chunk=args.chunk, rays=rays_in_camera_frame, retraw=True, **render_kwargs_test)

        # Compute MSE loss between predicted and true RGB.
        img_loss = rnh.img2mse(rgb, test_image)
        trans = extras['raw'][..., -1]
        loss = img_loss
        psnr = rnh.mse2psnr(img_loss)

        # Add MSE loss for coarse-grained model
        if 'rgb0' in extras:
            img_loss0 = rnh.img2mse(extras['rgb0'], test_image)
            loss += img_loss0

    gradients = tape.gradient(loss, grad_vars)
    optimizer.apply_gradients(zip(gradients, grad_vars))


if __name__ == '__main__':
    # load_nerf given config
    basedir = './logs'
    expname = 'linemod_driller_back_to_eggbox_similar_layout'

    exp_dir = os.path.join(basedir, expname)

    config = os.path.join(basedir, expname, 'config.txt')
    print('Args:')
    print(open(config, 'r').read())

    parser = rn.config_parser()
    ft_str = ''
    ft_str = '--ft_path {}'.format(os.path.join(basedir, expname, f'model_{680000}.npy'))
    args = parser.parse_args('--config {} '.format(config) + ft_str)
    args.chunk = 1000
    args.N_importance = 0
    images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                              recenter=True, bd_factor=.75,
                                                              spherify=args.spherify)

    # Create nerf model
    _, render_kwargs_test, start, grad_vars, models = rn.create_nerf(args)
    # randomly initialize camera parameters
    random_camera_init = np.eye(4, dtype=np.float32)
    # per step:
    test_image = images[i_test]
    optimize_model_to_single_image(args, test_image, render_kwargs_test)
    # rotate batched camera rays to world
    # forward pass nerf to generate image given input
    # get loss with test image
    # run optimizer until no difference with test image
