# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from collections import defaultdict
import time
import tqdm

from args_and_config import config_parser
from run_nerf_helpers import *
from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
import NERFCO.nerf_keypoint_network
import NERFCO.nerf_renderer


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = defaultdict(list)
    for i in range(0, rays_flat.shape[0], chunk):
        # print(f'i:{i} / {rays_flat.shape[0]}')
        ret = NERFCO.nerf_renderer.render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret.keys():
            all_ret[k].append(ret[k])

    all_ret = {k: tf.concat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, focal,
           chunk=1024*32, rays=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):
    """Render rays

    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.

    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """

    if c2w is not None:
        if 'K' in kwargs:
            rays_o, rays_d = NERFCO.nerf_renderer.get_rays_tf_K(H, W, kwargs['K'], c2w)
        else:
            # special case to render full image
            rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        if len(rays) == 3:
            rays_o, rays_d, depth = rays
            far = depth[:, tf.newaxis] * 1.1
            near = depth[:, tf.newaxis] * 0.9
        else:
            rays_o, rays_d = rays

    # hack for K not an argument of render_rays
    # TODO: can only be solved with total refactor
    kwargs_copy = kwargs.copy()
    if 'K' in kwargs_copy:
        del kwargs_copy['K']

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)

        # Make all directions unit magnitude.
        # shape: [batch_size, 3]
        viewdirs = viewdirs / tf.linalg.norm(viewdirs, axis=-1, keepdims=True)
        viewdirs = tf.cast(tf.reshape(viewdirs, [-1, 3]), dtype=tf.float32)

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(
            H, W, focal, tf.cast(1., tf.float32), rays_o, rays_d)

    # Create ray batch
    rays_o = tf.cast(tf.reshape(rays_o, [-1, 3]), dtype=tf.float32)
    rays_d = tf.cast(tf.reshape(rays_d, [-1, 3]), dtype=tf.float32)
    if not (type(near) == np.ndarray and type(far) == np.ndarray):
        near, far = near * \
            tf.ones_like(rays_d[..., :1]), far * tf.ones_like(rays_d[..., :1])

    # (ray origin, ray direction, min dist, max dist) for each ray
    rays = tf.concat([rays_o, rays_d, near, far], axis=-1)
    if use_viewdirs:
        # (ray origin, ray direction, min dist, max dist, normalized viewing direction)
        rays = tf.concat([rays, viewdirs], axis=-1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs_copy)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = tf.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
    from collections import defaultdict
    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []

    t = time.time()
    extrass = defaultdict(list)
    for i, c2w in tqdm.tqdm(enumerate(render_poses)):
        # print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, extras = render(
            H, W, focal, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)

        extras['disp'] = disp.numpy()

        rgbs.append(rgb.numpy())
        if i == 0:
            print(rgb.shape, disp.shape)

        if gt_imgs is not None and render_factor == 0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            print(p)

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

        _ = [extrass[k].append(extras[k]) for k in extras]

    rgbs = np.stack(rgbs, 0)
    extrass = {k: np.stack(extrass[k], 0) for k in extrass}
    return rgbs, extrass


def create_nerf(args):
    """Instantiate NeRF's MLP model."""

    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(
            args.multires_views, args.i_embed)
    output_ch = 4
    skips = [4]
    model = init_nerf_model(
        D=args.netdepth, W=args.netwidth,
        input_ch=input_ch, output_ch=output_ch, skips=skips,
        input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)

    grad_vars = model.trainable_variables
    models = {'model': model}

    model_fine = None
    if args.N_importance > 0:
        if args.keypoint_embedding_size is not None and args.keypoint_embedding_size > 0:
            model_fine = NERFCO.nerf_keypoint_network.init_nerf_model_with_categories(
                D=args.netdepth_fine, W=args.netwidth_fine,
                input_ch=input_ch, output_ch=output_ch, skips=skips,
                input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs,
                keypoint_embedding_size=args.keypoint_embedding_size,
                category_activation=args.category_activation,
                keypoint_regularize=args.keypoint_regularize,
                keypoint_dropout=args.keypoint_dropout,
            )
        else:
            model_fine = init_nerf_model(
                D=args.netdepth_fine, W=args.netwidth_fine,
                input_ch=input_ch, output_ch=output_ch, skips=skips,
                input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs
            )

        if args.learnable_embeddings_filename is not None:
            models['keypoint_embeddings'] = NERFCO.nerf_keypoint_network.get_learnable_embeddings(args)

        grad_vars += model_fine.trainable_variables
        models['model_fine'] = model_fine

    def network_query_fn(inputs, viewdirs, network_fn): return NERFCO.nerf_renderer.run_network(
        inputs, viewdirs, network_fn,
        embed_fn=embed_fn,
        embeddirs_fn=embeddirs_fn,
        netchunk=args.netchunk,
        c2w=args.c2w if hasattr(args, 'c2w') else None)

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
        'choose_keypoint_closest_depth': args.choose_keypoint_closest_depth,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        # print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    start = 0
    basedir = args.basedir
    expname = args.expname

    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 ('model_' in f and 'fine' not in f and 'optimizer' not in f)]
    # print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ft_weights = ckpts[-1]
        print('Reloading from', ft_weights)
        model.set_weights(np.load(ft_weights, allow_pickle=True))
        start = int(ft_weights[-10:-4]) + 1
        # print('Resetting step to', start)

        if model_fine is not None:
            ft_weights_fine = '{}_fine_{}'.format(
                ft_weights[:-11], ft_weights[-10:])
            print('Reloading fine from', ft_weights_fine)
            model_fine.set_weights(np.load(ft_weights_fine, allow_pickle=True))

    return render_kwargs_train, render_kwargs_test, start, grad_vars, models


def shuffler(array, permutations=None):
    len_first = len(array)
    if permutations is None:
        permutations = np.random.permutation(len_first)
        return array[permutations], permutations
    else:
        return array[permutations]


def unison_shuffled_copies(list_of_arrays: list):
    len_first = len(list_of_arrays[0])
    for _array in list_of_arrays:
        assert len_first == len(_array), f'len_first:{len_first} != len(_array): {len(_array)}'

    p = np.random.permutation(len_first)
    return [_array[p] for _array in list_of_arrays]


def render_test_data(args, render_poses, images, i_test, start, render_kwargs_test, hwf, K=None, embeddings=None, gt_keypoint_map=None):
    print('RENDER ONLY')
    if args.render_test:
        # render_test switches to test poses
        images = images[i_test]
    else:
        # Default is smoother render_poses path
        images = None

    testsavedir = os.path.join(args.basedir, args.expname, 'renderonly_{}_{:06d}'.format(
        'test' if args.render_test else 'path', start))
    os.makedirs(testsavedir, exist_ok=True)
    print('test poses shape', render_poses.shape)
    if args.use_K:
        print('get rays K')
        render_kwargs_test['K'] = K

    rgbs, extras = render_path(render_poses, hwf, args.chunk, render_kwargs_test,
                          gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
    print('Done rendering', testsavedir)
    for i, depth_map in enumerate(extras['depth_map']):
        imageio.imsave(os.path.join(testsavedir, f'depth_map_{i}.jpg'), depth_map)

    imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'),
                     to8b(rgbs), fps=30, quality=8)

    if 'keypoint_map' in extras:
        for i, (inferred_rgb_map, inferred_keypoint_map) in enumerate(zip(rgbs, extras['keypoint_map'])):
            if len(embeddings.shape) > 1:
                kp_map, kp_acc_map = NERFCO.nerf_keypoint_network.create_embedded_keypoint_image(inferred_keypoint_map,
                                                                                                 embeddings,
                                                                                                 gt_keypoint_map[i])
            else:
                kp_map, kp_acc_map = NERFCO.nerf_keypoint_network.create_binary_keypoint_image(np.squeeze(inferred_keypoint_map),
                                                                                               inferred_rgb_map,
                                                                                               gt_keypoint_map[i])
            imageio.imsave(os.path.join(testsavedir, f'kp_map_{i}.jpg'), to8b(kp_map))
            imageio.imsave(os.path.join(testsavedir, f'kp_acc_map_{i}.jpg'), to8b(kp_acc_map))


def train():

    parser = config_parser()
    args = parser.parse_args()
    
    if args.random_seed is not None:
        print('Fixing random seed', args.random_seed)
        np.random.seed(args.random_seed)
        tf.compat.v1.set_random_seed(args.random_seed)

    # Load data

    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)

        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape,
              render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            args.near = tf.reduce_min(bds) * .9
            args.far = tf.reduce_max(bds) * 1.
        else:
            args.near = 0.
            args.far = 1.
        print('NEAR FAR', args.near, args.far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split, extras = load_blender_data(
            args.datadir,
            args.half_res,
            args.testskip,
            args.image_extn,
            mask_directory=args.mask_directory,
            get_depths=args.get_depth_maps if args.near is None and args.far is None else False,
            image_field=args.image_fieldname,
            image_dir_override=args.image_dir_override,
            trainskip=args.trainskip,
            train_frames_field=args.frames_field)

        if args.mask_directory is not None:
            masks = extras['masks']

        if args.get_depth_maps:
            depth_maps = extras['depth_maps']

        K = None
        if args.use_K:
            K = extras['K']

        print('Loaded blender', images.shape,
              render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split
        if args.near is None and args.far is None:
            if args.Z_limits_from_pose:
                Zs = poses[:, 2, 3]
                args.near = np.min(np.abs(Zs)) * 0.5
                args.far = np.max(np.abs(Zs)) * 1.5

            elif args.get_depth_maps and args.mask_directory is not None:
                args.near = np.min(np.mean(depth_maps[masks])) * 0.9
                args.far = np.max(depth_maps[masks]) * 1.1
                print('using masked depth')

            elif args.get_depth_maps:
                args.near = np.min(depth_maps) * 0.9
                args.far = np.max(depth_maps) * 1.1
                print('using masked depth')
            else:
                args.near = 0.
                args.far = 2.

        print(f'args.near: {args.near} far: {args.far}')

        if args.mask_directory is not None and images.shape[-1] == 3:
            images = np.concatenate([images, masks[..., np.newaxis]], axis=-1)

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == 'deepvoxels':


        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape,
              render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        args.near = hemi_R-1.
        args.far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    if args.mask_directory and not args.white_bkgd:
        assert os.path.isdir(args.mask_directory), f'args.mask_directory not found at: {args.mask_directory}'
        if args.mask_images:
            images *= masks[..., np.newaxis]

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    filename_splits = extras['filenames']
    enable_keypoints = False
    keypoint_loss_coeff = 0
    if args.colmap_keypoints_filename is not None:
        enable_keypoints = True
        from NERFCO.nerf_keypoint_network import get_keypoint_masks
        import MODELS.random_embeddings

        train_keypoint_masks, args.number_of_keypoints = \
            get_keypoint_masks(args.colmap_keypoints_filename,
                               i_train,
                               filename_splits,
                               H, W)

        train_keypoint_masks = train_keypoint_masks.ravel()
        test_keypoint_masks, test_keypoint_count = \
            get_keypoint_masks(args.colmap_keypoints_filename,
                               i_test,
                               filename_splits,
                               H, W)

        number_of_keypoints = np.sum(train_keypoint_masks > 0)

        assert test_keypoint_count == args.number_of_keypoints, \
            f'test_keypoint_count: {test_keypoint_count} == args.number_of_keypoints: {args.number_of_keypoints}'

        print(train_keypoint_masks.shape, train_keypoint_masks.min(), train_keypoint_masks.max(), number_of_keypoints)
        keypoint_embeddings_filename = os.path.join(args.basedir, args.expname, 'keypoint_embeddings.pkl')
        random_keypoint_embeddings_object = MODELS.random_embeddings.StaticRandomEmbeddings(
            args.number_of_keypoints + 1,
            args.keypoint_embedding_size,
            embedding_filename=keypoint_embeddings_filename,
            zero_embedding_origin=args.zero_embedding_origin)

        keypoint_embeddings = random_keypoint_embeddings_object.embeddings

    import NERFCO.extract_keypoints
    if args.autoencoded_keypoints_filename is not None:
        enable_keypoints = True
        assert os.path.isfile(args.autoencoded_keypoints_filename), \
            f'autoencoded_keypoints_filename not found at: {args.autoencoded_keypoints_filename}'

        import pickle
        input = open(args.autoencoded_keypoints_filename, 'rb')
        data = pickle.load(input)
        input.close()

        train_keypoint_masks = data['train_keypoint_masks']
        test_keypoint_masks = data['test_keypoint_masks']
        keypoint_embeddings = data['encoded_embeddings']

        NERFCO.extract_keypoints.test_keypoint_masks(train_keypoint_masks, keypoint_embeddings)
        NERFCO.extract_keypoints.test_keypoint_masks(test_keypoint_masks, keypoint_embeddings)

        args.keypoint_embedding_size = keypoint_embeddings.shape[-1]
        train_keypoint_masks = train_keypoint_masks.ravel()
        print(f'loaded {args.keypoint_detector} keypoints: {keypoint_embeddings.shape} from: {args.autoencoded_keypoints_filename}')

    elif args.keypoints_filename is not None and args.keypoint_detector in ['SIFT', 'ORB']:
        enable_keypoints = True

        train_keypoint_masks, test_keypoint_masks, keypoint_embeddings = \
            NERFCO.extract_keypoints.get_keypoints_and_maps(
                args.keypoints_filename,
                i_test,
                i_train,
                filename_splits,
                H, W)

        NERFCO.extract_keypoints.test_keypoint_masks(train_keypoint_masks, keypoint_embeddings)
        NERFCO.extract_keypoints.test_keypoint_masks(test_keypoint_masks, keypoint_embeddings)

        args.keypoint_embedding_size = keypoint_embeddings.shape[-1]
        train_keypoint_masks = train_keypoint_masks.ravel()
        print(f'loaded {args.keypoint_detector} keypoints: {keypoint_embeddings.shape} from: {args.keypoints_filename}')

    elif args.learnable_embeddings_filename is not None:
        enable_keypoints = True

        train_keypoint_masks, test_keypoint_masks, static_keypoints = \
            NERFCO.extract_keypoints.get_keypoints_and_maps(
                args.keypoints_filename,
                i_test,
                i_train,
                filename_splits,
                H, W)

        args.number_of_keypoints = static_keypoints.shape[0]

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, models = create_nerf(
        args)

    if args.use_K:
        K_dict = {
            'K': K,
        }

        render_kwargs_train.update(K_dict)
        render_kwargs_test.update(K_dict)

    if args.learnable_embeddings_filename is not None:
        keypoint_embeddings = models['keypoint_embeddings']

        NERFCO.extract_keypoints.test_keypoint_masks(train_keypoint_masks, keypoint_embeddings[:])
        NERFCO.extract_keypoints.test_keypoint_masks(test_keypoint_masks, keypoint_embeddings[:])

        args.keypoint_embedding_size = keypoint_embeddings.shape[-1]
        train_keypoint_masks = train_keypoint_masks.ravel()
        print(f'loaded {args.keypoint_detector} keypoints: {keypoint_embeddings.shape} from: {args.learnable_embeddings_filename}')


    bds_dict = {
        'near': tf.cast(args.near, tf.float32),
        'far': tf.cast(args.far, tf.float32),
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Short circuit if only rendering out from trained model
    if args.render_only:
        # render_poses = poses[::args.testskip]
        render_test_data(args,
                         render_poses,
                         images,
                         i_test,
                         start,
                         render_kwargs_test,
                         hwf,
                         K if K is not None else None,
                         embeddings=keypoint_embeddings if enable_keypoints else None,
                         gt_keypoint_map=test_keypoint_masks if enable_keypoints else None)
        exit(0)

    # Create optimizer
    lrate = args.lrate
    if args.lrate_decay > 0:
        lrate = tf.keras.optimizers.schedules.ExponentialDecay(lrate,
                                                               decay_steps=args.lrate_decay * 1000, decay_rate=0.1)
    optimizer = tf.keras.optimizers.Adam(lrate)
    models['optimizer'] = optimizer

    global_step = tf.compat.v1.train.get_or_create_global_step()
    global_step.assign(start)

    photometric_loss_function = img2mse
    if args.use_huber_loss:
        photometric_loss_function = tf.compat.v1.losses.huber_loss

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching.
        #
        # Constructs an array 'rays_rgb' of shape [N*H*W, 3, 3] where axis=1 is
        # interpreted as,
        #   axis=0: ray origin in world space
        #   axis=1: ray direction in world space
        #   axis=2: observed RGB color of pixel

        # get_rays_np() returns rays_origin=[H, W, 3], rays_direction=[H, W, 3]
        # for each pixel in the image. This stack() adds a new dimension.
        if args.use_K:
            print('get rays K')
            rays = [NERFCO.nerf_renderer.get_rays_tf_K(H, W, K, p) for p in poses[:, :3, :4]]
        else:
            print('get rays')
            rays = [get_rays_np(H, W, focal, p) for p in poses[:, :3, :4]]

        rays = np.stack(rays, axis=0)  # [N, ro+rd, H, W, 3]
        print('done, concats')
        # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.concatenate([rays, images[:, None, ...]], 1)
        # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], axis=0)  # train images only
        #
        if args.depth_from_camera and args.depth_loss:
            train_depths = np.stack([depth_maps[i] for i in i_train], axis=0).ravel()  # train images only

        rays_rgb = np.stack([rays_rgb[i] for i in i_train], axis=0)  # train images only
        if args.mask_directory and not args.white_bkgd:
            train_masks = np.stack([masks[i] for i in i_train], axis=0)
            pixel_train_masks = train_masks.ravel()
            if args.ray_masking:
                rays_rgb = rays_rgb[np.where(train_masks)]

        # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
        rays_rgb = rays_rgb.astype(np.float32)

        if args.keypoint_oversample:
            keypoint_mask = train_keypoint_masks.astype(np.bool)
            over_sample = train_keypoint_masks.shape[0] // np.sum(keypoint_mask) - 1
            keypoint_rays_rgb = rays_rgb[keypoint_mask]
            non_zero_keypoints = train_keypoint_masks[keypoint_mask]
            assert non_zero_keypoints[0] == train_keypoint_masks[np.argmax(keypoint_mask)], 'first true keypoint should match'

            # repeat rgb
            repeated_keypoint_rays_rgb = np.repeat(keypoint_rays_rgb, over_sample, axis=0)
            assert np.all(keypoint_rays_rgb[0] == repeated_keypoint_rays_rgb[0]), 'repeats should be the same'
            rays_rgb = np.concatenate([rays_rgb, repeated_keypoint_rays_rgb])

            # repeat keypoints
            repeated_keypoint_masks = np.repeat(non_zero_keypoints, over_sample, axis=0)
            assert repeated_keypoint_masks[0] == train_keypoint_masks[np.argmax(keypoint_mask)]
            train_keypoint_masks = np.concatenate([train_keypoint_masks, repeated_keypoint_masks])

            # check rays and keypoint masks are the same size
            assert rays_rgb.shape[0] == train_keypoint_masks.shape[0]

    N_iters = 1000000
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    writer = tf.contrib.summary.create_file_writer(
        os.path.join(basedir, 'summaries', expname))
    writer.set_as_default()
    i_batch = 0
    for i in range(start, N_iters):
        time0 = time.time()
        if i >= args.keypoint_iterations_start:
            keypoint_loss_coeff = args.keypoint_loss_coeff
        # Sample random ray batch

        if use_batching:
            # shuffle
            if i == start or i_batch >= rays_rgb.shape[0]:
                rays_rgb, permutations = shuffler(rays_rgb)
                if args.mask_directory and args.sigma_masking:
                    pixel_train_masks = shuffler(pixel_train_masks, permutations)

                if enable_keypoints:
                    train_keypoint_masks = shuffler(train_keypoint_masks, permutations)

                if args.depth_from_camera and args.depth_loss:
                    train_depths = shuffler(train_depths, permutations)

                i_batch = 0

            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand]  # [B, 2+1, 3*?]
            batch = tf.transpose(batch, [1, 0, 2])


            if args.sigma_masking:
                in_mask_pixels_batch = pixel_train_masks[i_batch: i_batch + N_rand]

            if enable_keypoints:
                keypoint_masks_batch = train_keypoint_masks[i_batch: i_batch + N_rand]

            if args.depth_from_camera and args.depth_loss:
                train_depths_batch = train_depths[i_batch: i_batch + N_rand]

            i_batch += N_rand

            # batch_rays[i, n, xyz] = ray origin or direction, example_id, 3D position
            # target_s[n, rgb] = example_id, observed color.
            if args.depth_from_camera and args.depth_loss:
                batch_rays, target_s = (batch[0], batch[0], train_depths_batch), batch[2]
            else:
                batch_rays, target_s = batch[:2], batch[2]



        else:
            # Random from one image
            test_frame_number = np.random.choice(i_train)
            target = images[test_frame_number]
            pose = poses[test_frame_number, :3, :4]

            if N_rand is not None:
                if args.use_K:
                    rays_o, rays_d = NERFCO.nerf_renderer.get_rays_K(H, W, K, pose)
                else:
                    rays_o, rays_d = get_rays(H, W, focal, pose)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = tf.stack(tf.meshgrid(
                        tf.range(H//2 - dH, H//2 + dH), 
                        tf.range(W//2 - dW, W//2 + dW), 
                        indexing='ij'), -1)
                    if i < 10:
                        print('precrop', dH, dW, coords[0,0], coords[-1,-1])
                else:
                    coords = tf.stack(tf.meshgrid(
                        tf.range(H), tf.range(W), indexing='ij'), -1)
                coords = tf.reshape(coords, [-1, 2])
                select_inds = np.random.choice(
                    coords.shape[0], size=[N_rand], replace=False)
                select_inds = tf.gather_nd(coords, select_inds[:, tf.newaxis])
                rays_o = tf.gather_nd(rays_o, select_inds)
                rays_d = tf.gather_nd(rays_d, select_inds)
                batch_rays = tf.stack([rays_o, rays_d], 0)
                target_s = tf.gather_nd(target, select_inds)

        #####  Core optimization loop  #####

        with tf.GradientTape() as nerf_gradient_tape, tf.GradientTape() as embedding_gradient_tape:

            # Make predictions for color, disparity, accumulated opacity.
            rgb, disp, acc, extras = render(
                H, W, focal, chunk=args.chunk, rays=batch_rays,
                verbose=i < 10, retraw=True, **render_kwargs_train)
            # Compute MSE loss between predicted and true RGB.

            if args.sigma_masking:
                loss = photometric_loss_function(rgb[in_mask_pixels_batch], target_s[in_mask_pixels_batch])
            else:
                loss = photometric_loss_function(rgb, target_s)

            if args.force_black_background:
                loss += 0.1 * photometric_loss_function(rgb[~in_mask_pixels_batch], 0.)

            if 'keypoint_map' in extras:
                keypoint_loss = NERFCO.nerf_keypoint_network.get_keypoint_loss(keypoint_embeddings,
                                                                               keypoint_masks_batch,
                                                                               extras['keypoint_map'])
                loss += keypoint_loss_coeff * keypoint_loss

                if args.learnable_embeddings_filename is not None:
                    embedding_loss = keypoint_loss_coeff * keypoint_embeddings.distance_correlation_loss(extras['xyz_map'],
                                                                                                         extras['keypoint_map'])

            trans = extras['raw'][..., -1]
            psnr = mse2psnr(loss)

            if args.sigma_masking:
                loss += 0.01 * tf.reduce_sum(acc[~in_mask_pixels_batch]) / N_rand

            # Add MSE loss for coarse-grained model
            if 'rgb0' in extras:
                if args.sigma_masking:
                    img_loss0 = photometric_loss_function(extras['rgb0'][in_mask_pixels_batch], target_s[in_mask_pixels_batch])
                    psnr0 = mse2psnr(img_loss0)
                    loss += img_loss0 + 0.01 * tf.reduce_sum(extras['acc0'][~in_mask_pixels_batch]) / N_rand
                else:
                    img_loss0 = photometric_loss_function(extras['rgb0'], target_s)
                    psnr0 = mse2psnr(img_loss0)
                    loss += img_loss0

                if args.force_black_background:
                    loss += 0.1 * photometric_loss_function(extras['rgb0'][~in_mask_pixels_batch], 0.)

                if 'keypoint_map_0' in extras:
                    coarse_keypoint_loss = NERFCO.nerf_keypoint_network.get_keypoint_loss(keypoint_embeddings,
                                                                                          keypoint_masks_batch,
                                                                                          extras['keypoint_map_0'])
                    loss += keypoint_loss_coeff * coarse_keypoint_loss

                if args.depth_from_camera and args.depth_loss:
                    loss += tf.losses.huber_loss(train_depths_batch, extras['depth_map'])

        gradients = nerf_gradient_tape.gradient(loss, grad_vars)
        optimizer.apply_gradients(zip(gradients, grad_vars))

        if args.learnable_embeddings_filename is not None:
            gradients = embedding_gradient_tape.gradient(embedding_loss, models['keypoint_embeddings'].trainable_variables)
            optimizer.apply_gradients(zip(gradients, models['keypoint_embeddings'].trainable_variables))

        dt = time.time()-time0

        #####           end            #####

        # Rest is logging

        def save_weights(net, prefix, i):
            path = os.path.join(
                basedir, expname, '{}_{:06d}.npy'.format(prefix, i))
            np.save(path, net.get_weights())
            print('saved weights at', path)

        if i % args.i_weights == 0:
            for k in models:
                save_weights(models[k], k, i)

        if i % args.i_video == 0 and i > 0:

            rgbs, test_extras = render_path(
                render_poses, hwf, args.chunk, render_kwargs_test)
            disps = test_extras['disp']
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(
                basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4',
                             to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4',
                             to8b(disps / np.max(disps)), fps=30, quality=8)

            if args.use_viewdirs:
                render_kwargs_test['c2w_staticcam'] = render_poses[0][:3, :4]
                rgbs_still, _ = render_path(
                    render_poses, hwf, args.chunk, render_kwargs_test)
                render_kwargs_test['c2w_staticcam'] = None
                imageio.mimwrite(moviebase + 'rgb_still.mp4',
                                 to8b(rgbs_still), fps=30, quality=8)

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(
                basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            render_path(poses[i_test], hwf, args.chunk, render_kwargs_test,
                        gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

        if i % args.i_print == 0 or i < 10:
            output_line = f'{expname} {i}, {psnr.numpy():.3g}, {loss.numpy():.3g} '
            # report loss
            if enable_keypoints:
                gt_kp_mask = keypoint_masks_batch.astype(np.bool)
                network_output = np.squeeze(extras['keypoint_map'])
                output_line += f'kp loss:{keypoint_loss:.2g} '
                if len(network_output.shape) == 1:
                    output_line+= f'GT kp#:{int(np.sum(keypoint_masks_batch.astype(np.bool).astype(np.float))):d} '\
                                  f'TP+TN:{np.sum(np.isclose(gt_kp_mask.astype(np.float), network_output)) / network_output.shape[0]:.2g} '\
                                  f'TP:{np.sum(np.isclose(gt_kp_mask[gt_kp_mask].astype(np.float), network_output[gt_kp_mask])) / np.sum(gt_kp_mask):.2g} '\
                                  f'TN:{np.sum(np.isclose(gt_kp_mask[~gt_kp_mask].astype(np.float), network_output[~gt_kp_mask])) / np.sum(~gt_kp_mask):.2g} '

            print(output_line)
            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)

                if enable_keypoints:
                    tf.contrib.summary.histogram('keypoint output', network_output)
                    tf.contrib.summary.scalar('keypoint_loss', keypoint_loss)

            if i % args.i_img == 0:
                # report accuracy

                # Log a rendered validation view to Tensorboard
                test_frame_number = np.random.choice(i_val)
                target = images[test_frame_number]
                pose = poses[test_frame_number, :3, :4]

                rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))
                
                # Save out the validation image for Tensorboard-free monitoring
                testimgdir = os.path.join(basedir, expname, 'tboard_val_imgs')
                os.makedirs(testimgdir, exist_ok=True)
                imageio.imwrite(os.path.join(testimgdir, '{:06d}.png'.format(i)), to8b(rgb))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                    output_image = to8b(rgb)[tf.newaxis]
                    tf.contrib.summary.image('rgb', output_image)
                    tf.contrib.summary.image(
                        'disp', disp[tf.newaxis, ..., tf.newaxis])
                    tf.contrib.summary.image(
                        'acc', acc[tf.newaxis, ..., tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])
                    if enable_keypoints:
                        # TODO: test why the train renders seem to be less lossy than the test renders
                        # the test renders highlight the entire object whereas
                        # the train renders don't, but have a good loss
                        test_frame_in_split = np.where(i_val == test_frame_number)[0][0]
                        gt_keypoint_mask = test_keypoint_masks[test_frame_in_split]
                        gt_non_zeros_kp_mask = gt_keypoint_mask.astype(np.bool)
                        test_keypoint_loss = NERFCO.nerf_keypoint_network.get_keypoint_loss(keypoint_embeddings,
                                                                                            gt_keypoint_mask,
                                                                                            extras['keypoint_map'])
                        tf.contrib.summary.scalar('test_keypoint_loss', test_keypoint_loss)

                        if len(keypoint_embeddings.shape) == 1:
                            network_output = extras['keypoint_map']
                            network_output_image = np.reshape(network_output.numpy(), (H, W))
                            keypoint_accuracy_image = network_output_image == gt_non_zeros_kp_mask
                            tf.contrib.summary.histogram('keypointyness', network_output)
                            tf.contrib.summary.scalar('keypoint_acc',
                                                      np.sum(keypoint_accuracy_image) / (H*W))
                            tf.contrib.summary.scalar('non_zero_keypoint_acc',
                                                      np.sum(keypoint_accuracy_image[gt_non_zeros_kp_mask]) / np.sum(
                                                          gt_non_zeros_kp_mask))

                            tf.contrib.summary.image('keypoint_image_acc',
                                                     to8b(keypoint_accuracy_image)[tf.newaxis, ..., tf.newaxis])

                        else:
                            inferred_keypoint_image, keypoint_accuracy_image = \
                                NERFCO.nerf_keypoint_network.create_embedded_keypoint_image(extras['keypoint_map'],
                                                                                            keypoint_embeddings,
                                                                                            gt_keypoint_mask)

                            tf.contrib.summary.scalar('keypoint_acc', np.sum(keypoint_accuracy_image) / (H * W))
                            tf.contrib.summary.scalar('non_zero_keypoint_acc',
                                                      np.sum(keypoint_accuracy_image[gt_non_zeros_kp_mask]) / np.sum(gt_non_zeros_kp_mask))

                            tf.contrib.summary.image('keypoint_image_acc',
                                                     to8b(keypoint_accuracy_image)[tf.newaxis, ..., tf.newaxis])
                            tf.contrib.summary.image('keypoint_image',
                                                     to8b(inferred_keypoint_image)[tf.newaxis, ..., tf.newaxis])

                if args.N_importance > 0:
                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image(
                            'rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image(
                            'disp0', extras['disp0'][tf.newaxis, ..., tf.newaxis])
                        tf.contrib.summary.image(
                            'z_std', extras['z_std'][tf.newaxis, ..., tf.newaxis])

        global_step.assign_add(1)


if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    train()
