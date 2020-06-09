import os, sys
import pprint
import tensorflow as tf
import numpy as np
import imageio
import pprint

import matplotlib.pyplot as plt

import run_nerf as rn
import run_nerf_helpers as rnh
from load_llff import load_llff_data
import MODELS.rotations
import MODELS.tf_rotations
import OUTPUT.image_tools

tf.compat.v1.enable_eager_execution()


def create_ray_batches(H, W, focal, images, poses, i_train=[0], seed=0, shuffle=True):
        # For random ray batching.
        #
        # Constructs an array 'rays_rgb' of shape [N*H*W, 3, 3] where axis=1 is
        # interpreted as,
        #   axis=0: ray origin in world space
        #   axis=1: ray direction in world space
        #   axis=2: observed RGB color of pixel
        print('get rays')
        # get_rays_np() returns rays_origin=[H, W, 3], rays_direction=[H, W, 3]
        # for each pixel in the image. This stack() adds a new dimension.
        rays = [rn.get_rays_np(H, W, focal, p) for p in poses[:, :3, :4]]
        rays = np.stack(rays, axis=0)  # [N, ro+rd, H, W, 3]
        print('done, concats')
        # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.concatenate([rays, images[:, None, ...]], 1)
        # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], axis=0)  # train images only
        # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
        rays_rgb = rays_rgb.astype(np.float32)
        if shuffle:
            print('shuffle rays')
            np.random.RandomState(seed).shuffle(rays_rgb)
            print('done')
        return rays_rgb


def optimize_model_to_single_image(args, test_image, render_kwargs, grad_vars, test_T, epochs=100, down=16):
    H, W, focal = test_image.shape[0]//down, test_image.shape[1]//down, 584./down
    optimizer = tf.keras.optimizers.Adam(args.lrate)
    identity_transformation = np.eye(4, dtype=np.float32)[:3, :4] # identity pose matrix
    loss_list = []
    rotation_difference = []
    translation_difference = []
    downsampled_image = downsample_image(np.copy(test_image), down)
    for epoch in range(epochs):
        rays_rgb = create_ray_batches(H, W, focal, np.expand_dims(downsampled_image, axis=0),
                                      np.expand_dims(identity_transformation, axis=0), seed=epoch, shuffle=True)
        number_of_batches = max(rays_rgb.shape[0] // args.N_rand, 1)
        for i_batch in range(number_of_batches):
            with tf.GradientTape() as tape:
                # Random over all image
                batch = rays_rgb[i_batch:i_batch+args.N_rand]  # [B, 2+1, 3*?]
                batch = tf.transpose(batch, [1, 0, 2])
                # batch_rays[i, n, xyz] = ray origin or direction, example_id, 3D position
                # target_s[n, rgb] = example_id, observed color.
                batch_rays, target_s = batch[:2], batch[2]

                rgb, disp, acc, extras = rn.render(
                    H, W, focal, rays=batch_rays, retraw=True, **render_kwargs)

                # Compute MSE loss between predicted and true RGB.
                img_loss = rnh.img2mse(rgb, target_s)
                trans = extras['raw'][..., -1]
                loss = img_loss
                psnr = rnh.mse2psnr(img_loss)

                # Add MSE loss for coarse-grained model
                if 'rgb0' in extras:
                    img_loss0 = rnh.img2mse(extras['rgb0'], target_s)
                    loss += img_loss0

            gradients = tape.gradient(loss, grad_vars)
            optimizer.apply_gradients(zip(gradients, grad_vars))
            T = np.squeeze(MODELS.rotations.np_rotation_9d_flat_to_transformation_matrix(grad_vars[0].numpy())).astype(np.float32)
            diff = MODELS.rotations.compare_poses(T, test_T)
            loss_list.append(loss.numpy())
            rotation_difference.append(diff[0])
            translation_difference.append(diff[1])

            print(f'epoch: {epoch+1}/{epochs} batch: {i_batch+1}/{number_of_batches} loss: {loss:1f} psnr: {psnr:1f} diff: {diff[0] * 180. / np.pi:1f}deg {diff[1]:1f}m')

    plt.plot(loss_list, label='loss')
    plt.plot(rotation_difference, label='rotation_difference')
    plt.plot(translation_difference, label='translation_difference')
    plt.legend()
    plt.show()
    return T


def render_both_ways(args, T, H, W, focal):
    # c2w in rays
    render_kwargs_train, render_kwargs_test, start, _, models = rn.create_nerf(args)

    bds_dict = {
        'near': args.near,
        'far': args.far,
    }
    render_kwargs_test.update(bds_dict)

    down = 8
    c2w_first_rgb, _, _, _ = rn.render(H // down, W // down, focal // down, c2w=T, **render_kwargs_test)

    # c2w in network
    args.c2w = tf.Variable(
                MODELS.rotations.np_transformation_matrix_to_9d_flat(
                    np.expand_dims(T, axis=0)
                ),
                name='c2w')

    eye = np.eye(4, dtype=np.float32)[:3, :4]
    render_kwargs_train, render_kwargs_test, start, _, models = rn.create_nerf(args)
    bds_dict = {
        'near': args.near,
        'far': args.far,
    }
    render_kwargs_test.update(bds_dict)

    c2w_network_rgb, _, _, _ = rn.render(H // down, W // down, focal // down, c2w=eye, **render_kwargs_test)

    tiled_images = OUTPUT.image_tools.tile_images(np.array([[255.*np.clip(c2w_first_rgb.numpy(), 0, 1), 255.*np.clip(c2w_network_rgb.numpy(), 0, 1)]]))
    # tiled_images = OUTPUT.image_tools.tile_images(
    #     np.array([[255. * c2w_first_rgb.numpy(), 255. * c2w_network_rgb.numpy()]]))
    plt.imshow(tiled_images)
    plt.suptitle(','.join(['c2w_first_rgb', 'c2w_network_rgb']))
    plt.show()


def add_trainable_c2w_to_nerf(render_kwargs_test, input_ch=3, input_ch_views=3,):
    keras_model_object = render_kwargs_test['network_fn']
    original_inputs = keras_model_object.input
    inputs_pts, inputs_views = tf.split(original_inputs, [input_ch, input_ch_views], -1)
    c2w = tf.Variable(tf.eye(3, 4))
    world_input_pts = inputs_pts @ c2w
    world_input_views = inputs_views @ c2w
    world_inputs = tf.concat([world_input_pts, world_input_views])
    render_kwargs_test['network_fn'] = tf.keras.Model(input=world_inputs, output=keras_model_object.input)
    grad_vars = [c2w]
    return render_kwargs_test, grad_vars


def try_render_demo():
    basedir = './logs'
    expname = 'linemod_driller_back_to_eggbox_similar_layout'

    exp_dir = os.path.join(basedir, expname)
    config = os.path.join(exp_dir, 'config.txt')
    print('Args:')
    print(open(config, 'r').read())
    parser = rn.config_parser()

    args = parser.parse_args(
        '--config {} --ft_path {}'.format(config, os.path.join(basedir, expname, 'model_680000.npy')))
    print('loaded args')

    images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                              recenter=True, bd_factor=.75,
                                                              spherify=args.spherify)
    H, W, focal = poses[0, :3, -1].astype(np.float32)
    print(focal)
    H = int(H)
    W = int(W)
    hwf = [H, W, focal]

    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    if args.no_ndc:
        near = tf.reduce_min(bds) * .9
        far = tf.reduce_max(bds) * 1.
    else:
        near = 0.
        far = 1.

    # In[3]:

    # Create nerf model
    _, render_kwargs_test, start, grad_vars, models = rn.create_nerf(args)

    bds_dict = {
        'near' : tf.cast(near, tf.float32),
        'far' : tf.cast(far, tf.float32),
    }
    render_kwargs_test.update(bds_dict)

    print('Render kwargs:')
    pprint.pprint(render_kwargs_test)


    down = 1
    render_kwargs_fast = {k : render_kwargs_test[k] for k in render_kwargs_test}
    render_kwargs_fast['N_importance'] = 0

    c2w = np.eye(4)[:3,:4].astype(np.float32) # identity pose matrix
    test = rn.render(H//down, W//down, focal/down, c2w=c2w, **render_kwargs_fast)


def show_test_images_at_c2w(Ts, test_image, render_kwargs, titles=[]):
    H, W, focal = test_image.shape[0], test_image.shape[1], 584. * test_image.shape[0] / 480.
    image_list = [255. * test_image]
    for T in Ts:
        # render_kwargs.c2w = np.eye(4, dtype=np.float32)
        rgb, disp, acc, extras = rn.render(H, W, focal, c2w=T, **render_kwargs)
        clipped_rgb = np.clip(rgb, 0, 1)
        image_list.append(255. * clipped_rgb)

    tiled_images = OUTPUT.image_tools.tile_images(np.array([image_list]))
    plt.imshow(tiled_images)
    plt.suptitle(','.join(titles))
    plt.show()


def downsample_image(image, down):
    import tensorflow as tf
    H, W = image.shape[0], image.shape[1]
    downsampled_image = tf.image.resize(np.expand_dims(image, axis=0), [H // down, W // down]).numpy()
    return downsampled_image[0]


def train_on_one_image(args, i_test):
    identity_pose = np.eye(4, dtype=np.float32)
    tf_identity_pose = tf.Variable(
        MODELS.rotations.np_transformation_matrix_to_9d_flat(
            np.expand_dims(identity_pose, axis=0)
        ),
        name='c2w')
    test_pose = np.r_[poses[i_test][:, :4], [[0., 0., 0., 1.]]].astype(np.float32)
    initial_pose = MODELS.rotations.translate_Z(MODELS.rotations.rotate_about_X(test_pose, 8.).astype(np.float32), .0)[
                :3, :4]
    # initial_pose = test_pose
    args.c2w = tf.Variable(
        MODELS.rotations.np_transformation_matrix_to_9d_flat(
            np.expand_dims(initial_pose, axis=0)
        ),
        name='c2w')
    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, _, models = rn.create_nerf(args)
    # per step:
    test_image = images[i_test]

    bds_dict = {
        'near': args.near,
        'far': args.far
    }
    render_kwargs_test.update(bds_dict)

    pprint.pprint(render_kwargs_test)
    # print(f'diff: {MODELS.rotations.compare_rotations(test_pose, initial_pose) * 180. / np.pi:.2f}deg')
    for downsample in [4, 2, 1]:
        predicted_c2w = optimize_model_to_single_image(args, test_image, render_kwargs_test, [args.c2w], test_pose, epochs=99, down=downsample)
        downsampled_image = downsample_image(np.copy(test_image), downsample)
        args.c2w = tf_identity_pose
        show_test_images_at_c2w([initial_pose, test_pose, predicted_c2w],
                                downsampled_image,
                                render_kwargs_test,
                                titles=[f'downsample:{downsample}', 'test_image', 'initial_pose', 'test_pose', 'predicted_pose'])

        args.c2w = tf.Variable(
            MODELS.rotations.np_transformation_matrix_to_9d_flat(
                np.expand_dims(predicted_c2w, axis=0)
            ),
            name='c2w')

    args.c2w = tf_identity_pose
    show_test_images_at_c2w([initial_pose, test_pose, predicted_c2w],
                            test_image,
                            render_kwargs_test,
                            titles=[f'i_test:{i_test}', 'test_image', 'initial_pose', 'test_pose', 'predicted_pose'])

    diff = MODELS.rotations.compare_poses(test_pose, predicted_c2w)
    print(test_pose)
    print(predicted_c2w)
    print(f'i_test: {i_test} diff: {diff[0] * 180. / np.pi:1f}deg {diff[1]:1f}')


if __name__ == '__main__':
    # try_render_demo()
    # load_nerf given config
    parser = rn.config_parser()
    args = parser.parse_args()

    images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                              recenter=True, bd_factor=.75,
                                                              spherify=args.spherify)

    i_test = 2
    # test_pose = poses[i_test, :3, :4]
    test_pose = np.eye(4, dtype=np.float32)
    test_pose = MODELS.rotations.translate_Z(MODELS.rotations.rotate_about_X(test_pose, -10.).astype(np.float32), .0)[:3, :4]

    test_image = images[i_test]
    H, W, focal = test_image.shape[0], test_image.shape[1], 584.
    # Create nerf model
    # render_kwargs_train, render_kwargs_test, start, _, models = rn.create_nerf(args)
    # per step:
    test_image = images[i_test]
    near = tf.reduce_min(bds) * .9
    far = tf.reduce_max(bds) * 1.
    args.near = tf.cast(near, tf.float32)
    args.far = tf.cast(far, tf.float32)
    print('Render kwargs:')

    # show_test_images_at_c2w([poses[i_test, :3, :4]], test_image, render_kwargs_test)
    # render_both_ways(args, test_pose, H, W, focal)
    train_on_one_image(args, i_test)
