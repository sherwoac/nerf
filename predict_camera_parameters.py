import os, sys
import pprint
import tensorflow as tf
import numpy as np
import scipy
import imageio
import pprint
import re
import matplotlib.pyplot as plt
import json

import run_nerf as rn
import run_nerf_helpers as rnh
import load_llff # load_llff_data
import load_blender
import MODELS.rotations
import MODELS.tf_rotations
import OUTPUT.image_tools
import LOADER.dumb_loader as dl
import create_linemod_data
import MODELS.open3d_point_cloud_distance

tf.compat.v1.enable_eager_execution()
degree_sign= u'\N{DEGREE SIGN}'

def create_ray_batches(H, W, focal, images, poses, i_train=[0], seed=0, shuffle=True, masks=None, depth_images=None, K=None):
        # For random ray batching.
        #
        # Constructs an array 'rays_rgb' of shape [N*H*W, 3, 3] where axis=1 is
        # interpreted as,
        #   axis=0: ray origin in world space
        #   axis=1: ray direction in world space
        #   axis=2: observed RGB color of pixel

        # get_rays_np() returns rays_origin=[H, W, 3], rays_direction=[H, W, 3]
        # for each pixel in the image. This stack() adds a new dimension.
        if K is not None:
            # print('get rays K')
            rays = [rn.get_rays_np_K(H, W, K, p) for p in poses[:, :3, :4]]
        else:
            # print('get rays')
            rays = [rn.get_rays_np(H, W, focal, p) for p in poses[:, :3, :4]]

        rays = np.stack(rays, axis=0)  # [N, ro+rd, H, W, 3]
        # print('done, concats')
        # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.concatenate([rays, images[:, None, ...]], 1)
        # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], axis=0)  # train images only
        # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
        rays_rgb = rays_rgb.astype(np.float32)
        if shuffle:
            arrays_to_shuffle = [rays_rgb]
            if masks is not None:
                arrays_to_shuffle.append(masks.ravel())

            coordinates = np.arange(images.size // 3)
            arrays_to_shuffle.append(coordinates)

            if depth_images is not None:
                arrays_to_shuffle.append(depth_images.ravel())

            shuffled_arrays = rn.unison_shuffled_copies(arrays_to_shuffle)
            return tuple(shuffled_arrays)
            # return masks, rays
            # else:
            #     print('shuffle rays')
            #     np.random.RandomState(seed).shuffle(rays_rgb)
            #     print('done')
            #     return rays_rgb
            # masks, rays = rn.unison_shuffled_copies(ravelled_masks, rays_rgb)


def create_masked_ray_batches(H, W, focal, ray_mask, images, poses, i_train=[0], seed=0, shuffle=True, masks=None, depth_images=None,
                       K=None):
    # For random ray batching.
    #
    # Constructs an array 'rays_rgb' of shape [N*H*W, 3, 3] where axis=1 is
    # interpreted as,
    #   axis=0: ray origin in world space
    #   axis=1: ray direction in world space
    #   axis=2: observed RGB color of pixel

    # get_rays_np() returns rays_origin=[H, W, 3], rays_direction=[H, W, 3]
    # for each pixel in the image. This stack() adds a new dimension.
    if K is not None:
        print('get rays K')
        rays = [rn.get_rays_np_K(H, W, K, p) for p in poses[:, :3, :4]]
    else:
        print('get rays')
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
        arrays_to_shuffle = [rays_rgb]
        if masks is not None:
            arrays_to_shuffle.append(masks.ravel())

        coordinates = np.arange(images.size // 3)
        arrays_to_shuffle.append(coordinates)

        if depth_images is not None:
            arrays_to_shuffle.append(depth_images.ravel())

        shuffled_arrays = rn.unison_shuffled_copies(arrays_to_shuffle)
        return tuple(shuffled_arrays)
        # return masks, rays
        # else:
        #     print('shuffle rays')
        #     np.random.RandomState(seed).shuffle(rays_rgb)
        #     print('done')
        #     return rays_rgb
        # masks, rays = rn.unison_shuffled_copies(ravelled_masks, rays_rgb)


def find_rough_mask(args, render_kwargs, test_image, down, K):
    H, W = test_image.shape[:2]
    H_down, W_down, K_down = test_image.shape[0] // down, test_image.shape[1] // down, K / down
    rays = rn.get_rays_np_K(H_down, W_down, K_down, c2w=np.eye(4, dtype=np.float32)[:3, :4])
    _, _, acc, _ = rn.render(H_down, W_down, focal=-1.0, rays=rays, **render_kwargs)
    threshold_full_mask = tf.where(acc > args.sigma_threshold,
                                   tf.ones(shape=acc.shape[:2]),
                                   tf.zeros(shape=acc.shape[:2]))

    dilated_threshold_mask = scipy.ndimage.binary_dilation(threshold_full_mask.numpy()).astype(threshold_full_mask.numpy().dtype)
    resized_threshold_mask = np.squeeze(tf.image.resize(np.expand_dims(dilated_threshold_mask, axis=-1), (H, W)).numpy())
    return np.where(resized_threshold_mask.astype(np.bool).ravel())


def optimize_model_to_single_image(args,
                                   test_image,
                                   render_kwargs,
                                   grad_vars,
                                   test_T,
                                   epochs=100,
                                   down=1,
                                   image_mask=None,
                                   test_depth_image=None,
                                   K=None,
                                   pcd=None):
    H, W, focal = test_image.shape[0]//down, test_image.shape[1]//down, 584./down
    optimizer = tf.keras.optimizers.Adam(args.lrate)
    identity_transformation = np.eye(4, dtype=np.float32)[:3, :4] # identity pose matrix
    img_loss_list = []
    depth_loss_list = []
    pcd_list = []

    downsampled_image = downsample_image(np.copy(test_image), down)

    loss_min = (float("inf"), )
    img_loss_min = (float("inf"), )
    depth_loss_min = (float("inf"), )

    for epoch in range(epochs):
        up_mask = find_rough_mask(args, render_kwargs, test_image, 16, K)
        ret = create_ray_batches(H,
                                 W,
                                 focal,
                                 np.expand_dims(downsampled_image, axis=0),
                                 np.expand_dims(identity_transformation, axis=0),
                                 seed=epoch,
                                 shuffle=True,
                                 masks=np.expand_dims(image_mask, axis=0) if image_mask is not None else None,
                                 depth_images=np.expand_dims(test_depth_image, axis=0) if test_depth_image is not None else None,
                                 K=K)

        rays_rgb = ret[0]
        rays_rgb = rays_rgb[up_mask]
        if image_mask is not None:
            masks = ret[1]
            masks = masks[up_mask]

        if test_depth_image is not None:
            depths = ret[-1]
            depths = depths[up_mask]

        # shuffled_coordinates = ret[-2]

        number_of_batches = max(rays_rgb.shape[0] // args.N_rand, 1)
        for i_batch in range(number_of_batches):
            with tf.GradientTape() as tape:
                # Random over all image
                batch_indices = slice(i_batch * args.N_rand, i_batch * args.N_rand + args.N_rand)
                batch = rays_rgb[batch_indices] # [B, 2+1, 3*?]
                if image_mask is not None:
                    mask_batch = masks[batch_indices]  # [B, 2+1, 3*?]

                if test_depth_image is not None:
                    depth_batch = depths[batch_indices]  # [B, 2+1, 3*?]

                batch = tf.transpose(batch, [1, 0, 2])
                # batch_rays[i, n, xyz] = ray origin or direction, example_id, 3D position
                # target_s[n, rgb] = example_id, observed color.
                batch_rays, target_s = batch[:2], batch[2]

                rgb, disp, acc, extras = rn.render(
                    H, W, focal, rays=batch_rays, retraw=True, **render_kwargs)

                # Compute MSE loss between predicted and true RGB.
                if args.sigma_threshold > 0.:
                    threshold_mask = tf.where(acc > args.sigma_threshold)
                    # img_loss = 1 / (int(threshold_mask.shape[0]) + .001)
                    img_loss = rnh.img2mse(tf.gather_nd(rgb, threshold_mask), tf.gather_nd(target_s, threshold_mask))
                    if test_depth_image is not None:
                        # print(tf.reduce_mean(extras['depth_map']), extras['depth_map'].numpy().min(), extras['depth_map'].numpy().max())
                        depth_loss = tf.reduce_mean(tf.abs(tf.gather_nd(extras['depth_map'], threshold_mask) - tf.gather_nd(depth_batch, threshold_mask)))
                    else:
                        depth_loss = 0.
                else:
                    img_loss = rnh.img2mse(rgb, target_s)

                trans = extras['raw'][..., -1]
                loss = img_loss + 100.*depth_loss
                # psnr = rnh.mse2psnr(img_loss)

                # Add MSE loss for coarse-grained model
                # if 'rgb0' in extras:
                #     if args.sigma_threshold > 0.:
                #         img_loss0 = rnh.img2mse(tf.gather_nd(extras['rgb0'], threshold_mask), tf.gather_nd(target_s, threshold_mask))
                #     else:
                #         img_loss0 = rnh.img2mse(extras['rgb0'], target_s)
                #
                #     loss += img_loss0

            gradients = tape.gradient(loss, grad_vars)
            optimizer.apply_gradients(zip(gradients, grad_vars))
            T = np.squeeze(MODELS.rotations.np_rotation_9d_flat_to_transformation_matrix(grad_vars[0].numpy())).astype(np.float32)
            pcd_distance = pcd.get_distance_to_transformation(T, test_T)

            img_loss_list.append(img_loss.numpy())
            depth_loss_list.append(depth_loss.numpy())
            pcd_list.append(pcd_distance)

            print(f'epoch: {epoch+1}/{epochs} batch: {i_batch+1}/{number_of_batches} loss: {loss:1f} img_loss: {img_loss:1f} depth loss: {depth_loss:1f} pcd: {pcd_distance}m')
            if loss.numpy() < loss_min[0]:
                loss_min = (loss.numpy(), pcd_distance)

            if img_loss.numpy() < img_loss_min[0]:
                img_loss_min = (img_loss.numpy(), pcd_distance)

            if depth_loss.numpy() < depth_loss_min[0]:
                depth_loss_min = (depth_loss.numpy(), pcd_distance)

            if args.visualize_optimization:
                tiled = output_interim_results(test_image,
                                               render_kwargs,
                                               test_depth_image,
                                               image_mask,
                                               shuffled_coordinates,
                                               batch_indices,
                                               threshold_mask)
                imageio.imwrite(f'/home/adam/CODE/nerf/logs/camera_optimize/tiled_{str(epoch * 100 + i_batch).zfill(4)}.png', (255. * np.clip(tiled, 0., 1.)).astype(np.uint8))

    # plt.plot(loss_list, label='loss')
    # plt.plot(rotation_difference, label='rotation_difference')
    # plt.plot(translation_difference, label='translation_difference')
    # plt.legend()
    # plt.show()
    results = {'img_loss_min': img_loss_list, 'depth_loss_min': depth_loss_list, 'pcd': pcd_list}
    return T, results


def output_interim_results(test_image, render_kwargs, test_depth_image, image_mask, shuffled_coordinates, batch_indices, threshold_mask):
    H, W =test_image.shape[:2]
    rgb, disp, acc, extras = rn.render(H, W, 1.0, c2w=np.eye(4, dtype=np.float32)[:3, :4], **render_kwargs)
    threshold_full_mask = tf.where(acc > args.sigma_threshold, tf.ones(shape=rgb.shape[:2]),
                                   tf.zeros(shape=rgb.shape[:2]))
    rgb_thresholded = (rgb * threshold_full_mask[:, :, tf.newaxis]).numpy()
    depth_thresholded = threshold_full_mask * extras['depth_map']
    depth3 = get_array_norm(extras['depth_map'].numpy())
    depth_th3 = depth_thresholded.numpy()
    thresholded_gt_depth = threshold_full_mask * test_depth_image
    overlap = np.concatenate([threshold_full_mask.numpy()[:, :, np.newaxis],
                              image_mask.astype(np.float32)[:, :, np.newaxis],
                              np.zeros(shape=threshold_full_mask.shape[:2] + [1])], axis=-1)
    batch_shuffle_mask = np.zeros(shape=test_image.shape[:2]).astype(np.bool)
    in_object_batch_shuffle_mask = np.zeros(shape=test_image.shape[:2]).astype(np.bool)
    batch_shuffle_mask.ravel()[shuffled_coordinates[batch_indices]] = True
    in_object_batch_shuffle_mask.ravel()[shuffled_coordinates[batch_indices][threshold_mask]] = True
    # show_batch_image = np.copy(rgb.numpy())
    # show_batch_image[batch_shuffle_mask] = 1
    test_depth_image_norm = get_mpl_array_plot(get_array_norm(test_depth_image))
    test_depth_image_norm[threshold_full_mask.numpy().astype(np.bool)] = np.array([1., 0., 0.])

    test_image_output = np.copy(test_image)
    test_image_output[batch_shuffle_mask] = np.array([1., 0., 0.])
    test_image_output[in_object_batch_shuffle_mask] = np.array([0., 1., 0.])

    rgb_thresholded[batch_shuffle_mask] = np.array([1., 0., 0.])
    rgb_thresholded[in_object_batch_shuffle_mask] = np.array([0., 1., 0.])
    tiled = OUTPUT.image_tools.tile_images(
        np.array(
            [[make_rgb(acc), test_image_output, get_mpl_array_plot(depth3), test_depth_image_norm],
             [overlap, rgb_thresholded, get_mpl_array_plot(depth_th3),
              get_mpl_array_plot(depth_thresholded - thresholded_gt_depth)]]
        ),
    )
    return tiled


def get_array_norm(array):
    return (array - array.min()) / (array.max() - array.min())


def get_mpl_array_plot(array):
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    fig = Figure(frameon=False)
    fig.add_axes([0, 0, 1, 1])
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    ax.axis('off')
    ax.imshow(array)
    canvas.draw()  # draw the canvas, cache the renderer
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = (np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3) / 255.).astype(np.float32)
    return image


def make_rgb(single_channel_image):
    return np.concatenate([single_channel_image[:, :, np.newaxis], single_channel_image[:, :, np.newaxis], single_channel_image[:, :, np.newaxis]], axis=-1)


def render_both_ways(args, T, H, W, K):
    # c2w in rays
    render_kwargs_train, render_kwargs_test, start, _, models = rn.create_nerf(args)

    bds_dict = {
        'near': args.near,
        'far': args.far,
    }
    render_kwargs_test.update(bds_dict)
    down = 16
    render_kwargs_test.update({'K': K / down})

    c2w_first_rgb, _, _, _ = rn.render(H // down, W // down, 1.0 // down, c2w=T, **render_kwargs_test)

    # c2w in network
    args.c2w = tf.Variable(
                MODELS.rotations.np_transformation_matrix_to_9d_flat(
                    np.expand_dims(T, axis=0)
                ),
                name='c2w')

    identity_transformation = np.eye(4, dtype=np.float32)[:3, :4]
    render_kwargs_train, render_kwargs_test, start, _, models = rn.create_nerf(args)
    bds_dict = {
        'near': args.near,
        'far': args.far,
    }
    render_kwargs_test.update(bds_dict)
    render_kwargs_test.update({'K': K / down})
    c2w_network_rgb, _, _, _ = rn.render(H // down, W // down, 1.0 // down, c2w=identity_transformation, **render_kwargs_test)

    tiled_images = OUTPUT.image_tools.tile_images(np.clip(np.array([[255.*np.clip(c2w_first_rgb.numpy(), 0., 1.), 255.*np.clip(c2w_network_rgb.numpy(), 0., 1.)]]), 0., 1.))
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

    images, poses, bds, render_poses, i_test = load_llff.load_llff_data(args.datadir, args.factor,
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

    tiled_images = OUTPUT.image_tools.tile_images(np.array([image_list]).reshape((2, 2, 480, 640, 3)))
    plt.imshow(tiled_images)
    plt.suptitle(','.join(titles))
    plt.show()

    plt.imshow(acc)
    plt.show()


def downsample_image(image, down):
    import tensorflow as tf
    H, W = image.shape[0], image.shape[1]
    if down > 1:
        downsampled_image = tf.image.resize(np.expand_dims(image, axis=0), [H // down, W // down]).numpy()
        return downsampled_image[0]
    else:
        return image


def test_perturbation_range(args, images, poses, depth_images, K):
    for test_pose in poses:
        for x_deg in [0., 1., 2., 4., 8.]:
            for z_offset in [-8, -4, -2, 0., 2, 4, 8]:
                z = z_offset / 100.
                initial_pose = MODELS.rotations.translate_Z(MODELS.rotations.rotate_about_X(test_pose, x_deg).astype(np.float32), z)[:3, :4]
                # initial_pose = identity_pose[:3, :4]
                args.c2w = tf.Variable(
                    MODELS.rotations.np_transformation_matrix_to_9d_flat(
                        np.expand_dims(initial_pose, axis=0)
                    ),
                    name='c2w')
                # Create nerf model
                render_kwargs_train, render_kwargs_test, start, _, models = rn.create_nerf(args)
                # per step:
                test_image = images[i_test]
                test_mask = masks[i_test]
                # test_image = images[i_test]
                if depth_images is not None:
                    depth_image = depth_images[i_test]
                    depth_image = np.where(depth_image == 0., args.far.numpy(), depth_image)
                else:
                    depth_image = None

                bds_dict = {
                    'near': args.near,
                    'far': args.far
                }

                render_kwargs_test.update(bds_dict)

                pprint.pprint(render_kwargs_test)
                # print(f'diff: {MODELS.rotations.compare_rotations(test_pose, initial_pose) * 180. / np.pi:.2f}deg')
                for downsample in [1]:
                    predicted_c2w, results = optimize_model_to_single_image(args,
                                                                   test_image,
                                                                   render_kwargs_test,
                                                                   [args.c2w],
                                                                   test_pose,
                                                                   epochs=10,
                                                                   down=downsample,
                                                                   image_mask=test_mask,
                                                                   test_depth_image=depth_image,
                                                                   K=K)


def train_on_one_image(args, test_image, initial_pose, test_pose, test_mask, test_depth_image, K, pcd=None):
    args.c2w = tf.Variable(
        MODELS.rotations.np_transformation_matrix_to_9d_flat(
            np.expand_dims(initial_pose, axis=0)
        ),
        name='c2w')
    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, _, models = rn.create_nerf(args)
    bds_dict = {
        'near': args.near,
        'far': args.far
    }

    render_kwargs_test.update(bds_dict)
    # pprint.pprint(render_kwargs_test)
    predicted_c2w, results = optimize_model_to_single_image(args,
                                                   test_image,
                                                   render_kwargs_test,
                                                   [args.c2w],
                                                   test_pose,
                                                   epochs=10,
                                                   image_mask=test_mask,
                                                   test_depth_image=test_depth_image,
                                                   K=K,
                                                            pcd=pcd)

    return predicted_c2w, results


def show_depth_map(depth_map):
    plt.imshow(depth_map)
    plt.colorbar()
    plt.show()


def loss_plot(values, which_loss, measure_name, x_ticks, y_ticks, x_axis_label, y_axis_label):
    fig, ax = plt.subplots(1, 1)
    from matplotlib.pylab import cm
    from matplotlib.colors import LogNorm
    plt.title(f'loss: {which_loss} measure_name: {measure_name}')
    # img = ax.imshow(values)
    img = ax.matshow(values, cmap=cm.gray_r, norm=LogNorm(vmin=0.01, vmax=1))
    ax.set_xticks(np.arange(len(x_ticks)))
    ax.set_xticklabels([str(x_tick) for x_tick in x_ticks])
    ax.set_xlabel(x_axis_label)
    ax.set_yticks(np.arange(len(y_ticks)))
    ax.set_yticklabels([str(y_tick) for y_tick in y_ticks])
    ax.set_ylabel(y_axis_label)
    fig.colorbar(img)
    plt.show()


def plot_results():
    import pickle
    assert os.path.isfile('overall_results.pkl'), f'file not found at: {"overall_results.pkl"}'
    input = open('overall_results.pkl', 'rb')
    overall_results = pickle.load(input)

    factor = 180. / np.pi

    x_degs = [0., 1., 2., 4., 8.]
    zs = [-8., -4., -2., 0., 2., 4., 8.]
    for i, measure in enumerate(['val', 'angle', 'translation']):
        loss_mins = np.zeros(shape=(len(x_degs), len(zs)))
        img_loss_mins = np.zeros_like(loss_mins)
        depth_loss_mins = np.zeros_like(loss_mins)
        for key, value in overall_results.items():
            x_deg, z = key
            z *= 100.
            loss_mins[x_degs.index(x_deg), zs.index(z)] = value['loss_min'][i] * (factor if measure == 'angle' else 1.)
            img_loss_mins[x_degs.index(x_deg), zs.index(z)] = value['img_loss_min'][i] * (factor if measure == 'angle' else 1.)
            depth_loss_mins[x_degs.index(x_deg), zs.index(z)] = value['depth_loss_min'][i] * (factor if measure == 'angle' else 1.)

        loss_plot(loss_mins, 'overall', measure, zs, x_degs, 'z displacement (m)', 'x rotation (deg)')
        loss_plot(img_loss_mins, 'image', measure, zs, x_degs, 'z displacement (m)', 'x rotation (deg)')
        loss_plot(depth_loss_mins, 'depth', measure, zs, x_degs, 'z displacement (m)', 'x rotation (deg)')


def render_depth_mask(args, T, H, W, focal, depth_image, mask):
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
    #render original
    rgb, _, _, extras_network = rn.render(H, W, focal, c2w=eye, **render_kwargs_test)
    # print(f'near:{args.near} far: {args.far} depth near:{extras_network["depth_map"].numpy().min()} far:{extras_network["depth_map"].numpy().max()} mask depth near:{extras_network["depth_map"].numpy()[mask].min()} far:{extras_network["depth_map"].numpy()[mask].max()} loss:{rnh.img2mse(depth_image, extras_network["depth_map"])} mask loss:{rnh.img2mse(depth_image[mask], extras_network["depth_map"][mask])}')
    #
    # plt.hist(extras_network['depth_map'].numpy()[mask].ravel())
    # plt.show()
    # plt.hist(depth_image[mask].ravel())
    # plt.show()
    #
    # show_depth_map(rgb)
    #
    # show_depth_map(extras_network['depth_map'].numpy())
    # show_depth_map(depth_image)

    diff_map = depth_image - extras_network['depth_map'].numpy()
    # show_depth_map(diff_map)
    diff_map[~mask] = 0
    show_depth_map(diff_map)
    # depth_image[~mask] = 0
    # show_depth_map(depth_image)


def render_depth_masks(args, images, poses, depth_images, masks):
    H, W, focal = images[0].shape[0], images[0].shape[1], 584.
    for i_test in i_tests:
        render_depth_mask(args, poses[i_test, :3, :4], H, W, focal, depth_images[i_test], masks[i_test])


def load_centring_transforms(args):
    assert os.path.exists(args.centring_transforms), f'args.centring_transforms not found at: {args.centring_transforms}'
    with open(args.centring_transforms, 'r') as fp:
        centring_transforms_dict = json.load(fp)
        centring_transforms_dict_out = {key: np.array(value) for key, value in centring_transforms_dict.items()}
    return centring_transforms_dict_out


def get_object_name_from_filename(filename):
    return str(filename).split('_')[0]


def load_approximate_poses(approximate_poses_filename, centring_transforms_dict):
    assert os.path.isfile(approximate_poses_filename), \
        f'approximate_poses_filename not found at: {approximate_poses_filename}'
    df = dl.open_dataframe(approximate_poses_filename)
    object_name = get_object_name_from_filename(df['filename'].iloc[0])
    centring_transformation = centring_transforms_dict[object_name]
    return {int(df_row['frame_number']):
        transformation_to_nerf_sense(
            df_row['revised_pose'],
            centring_transformation).astype(np.float32)
            for _, df_row in df.iterrows()}


def transformation_to_nerf_sense(T, centring_transformation):
    return np.linalg.inv(create_linemod_data.rub_to_rdf(np.dot(T, centring_transformation)))


def transformation_from_nerf_sense(T, centring_transformation):
    return np.dot(create_linemod_data.rub_to_rdf(np.linalg.inv(T)), np.linalg.inv(centring_transformation))


def print_string_diff(T1, T2):
    diff_T1T2 = MODELS.rotations.compare_poses(T1, T2)
    return f'{diff_T1T2[0] * 180./np.pi:.2g}{degree_sign} {diff_T1T2[1]:.2g}m'


if __name__ == '__main__':
    # try_render_demo()
    # load_nerf given config
    parser = rn.config_parser()
    args = parser.parse_args()
    masks = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_tests = load_llff.load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        masks = rn.load_masks(args.datadir, args.mask_directory)
    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split, extras = load_blender.load_blender_data(
            args.datadir,
            args.half_res,
            args.testskip,
            args.image_extn,
            mask_directory=args.mask_directory,
            get_depths=args.get_depth_maps)

        i_train, i_val, i_tests = i_split

        if args.mask_directory is not None:
            masks = extras['masks']

        depth_images = None
        if args.get_depth_maps:
            depth_images = extras['depth_maps']

        K = None
        if args.use_K:
            K = extras['K']

    if args.Z_limits_from_pose:
        Zs = poses[:, 2, 3]
        near = np.min(np.abs(Zs)) * 0.9
        far = np.max(np.abs(Zs)) * 1.5
        args.near = tf.cast(near, tf.float32)
        args.far = tf.cast(far, tf.float32)

    H, W = images[0].shape[:2]
    # render_depth_masks(args, images, poses, depth_images, masks)
    # show_test_images_at_c2w([poses[i_test, :3, :4]], test_image, render_kwargs_test)
    # render_both_ways(args, test_pose, H, W, extras['K'])
    if args.centring_transforms:
        centring_transforms_dict = load_centring_transforms(args)

    if args.approximate_poses_filename:
        approximate_pose_dict = load_approximate_poses(args.approximate_poses_filename, centring_transforms_dict)
        df = dl.open_dataframe(args.approximate_poses_filename)
        transformation_pcd_function = lambda T: transformation_from_nerf_sense(T, centring_transformation)
        pcd_distance = MODELS.open3d_point_cloud_distance.PointCloudDistance(df['shape_filename'].iloc[0],
                                                                             transformation_function=transformation_pcd_function)
        object_name = get_object_name_from_filename(df['filename'].iloc[0])
        centring_transformation = centring_transforms_dict[object_name]

    overall_results = {}
    predicted_poses = []
    test_to_initial_pcd_distances = []

    for i, i_test in enumerate(i_tests):
        filename = extras['filenames'][2][i]
        frame_number = int(re.findall(r'\d+', filename)[-1])
        test_image = images[i_test]
        test_mask = masks[i]
        test_depth_image = depth_images[i_test]
        bb8_pose = approximate_pose_dict[frame_number]
        gt_pose = poses[i_test]
        predicted_c2w, results = train_on_one_image(args, test_image, bb8_pose, gt_pose, test_mask, test_depth_image, K, pcd_distance)

        gt_to_predicted = pcd_distance.get_distance_to_transformation(
            transformation_from_nerf_sense(gt_pose, centring_transformation),
            transformation_from_nerf_sense(predicted_c2w, centring_transformation)
        )

        gt_to_bb8 = pcd_distance.get_distance_to_transformation(
            transformation_from_nerf_sense(gt_pose, centring_transformation),
            transformation_from_nerf_sense(bb8_pose, centring_transformation)
        )
        predicted_poses.append(predicted_c2w)
        test_to_initial_pcd_distances.append(gt_to_predicted)
        overall_results[filename] = results
        result_str = 'worse'
        if gt_to_bb8 > gt_to_predicted:
            result_str = 'better'

        print(f'filename: {os.path.basename(filename)} {result_str} '
              f'pcd GTvsNLM: {gt_to_predicted:.2g} '
              f'GTvsBB8: {gt_to_bb8:.2g} '
              f'{results}')

# f'BB8->NLM: {print_string_diff(bb8_pose, predicted_c2w)} '
# f'GT->NLM: {print_string_diff(gt_pose, predicted_c2w)} '
# f'GT->BB8: {print_string_diff(gt_pose, bb8_pose)}

    test_to_initial_pcd_distances = np.stack(test_to_initial_pcd_distances)
    print(f'ADD measure: {np.sum(test_to_initial_pcd_distances < 0.1 * 0.259425) / test_to_initial_pcd_distances.shape[0]}')

    import pickle
    output = open('overall_results.pkl', 'wb')
    pickle.dump(overall_results, output)
    output.close()
    # plot_results()