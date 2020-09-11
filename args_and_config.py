import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str,
                        default='./data/llff/fern', help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int,
                        default=8, help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float,
                        default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000s)')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--random_seed", type=int, default=None,
                        help='fix random seed for repeatability')

    # pre-crop options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    # deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    # blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    # llff flags
    parser.add_argument("--factor", type=int,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000,
                        help='frequency of render_poses video saving')

    parser.add_argument("--image_extn",   type=str, default='.png',
                        help='training image extension')

    parser.add_argument("--mask_directory",   type=str, default=None,
                        help='mask_directory')

    parser.add_argument("--mask_images", action='store_true',
                        help='mask_images')

    parser.add_argument("--ray_masking", action='store_true',
                        help='ray_masking')

    parser.add_argument("--sigma_masking", action='store_true',
                        help='sigma_masking')

    parser.add_argument("--sigma_threshold", type=float, default=0., help='sigma_threshold')

    parser.add_argument("--get_depth_maps", action='store_true', help='get_depth_maps')

    parser.add_argument("--Z_limits_from_pose", action='store_true', help='Z_limits_from_pose')

    parser.add_argument("--force_black_background", action='store_true', help='force_black_background')

    parser.add_argument("--visualize_optimization", action='store_true', help='visualize_optimization')
    parser.add_argument("--visualize_results", action='store_true', help='visualize_results')
    parser.add_argument("--use_K", action='store_true', help='use_K - full camera model')

    parser.add_argument("--image_fieldname", type=str, default='file_path', help='image_fieldname')

    parser.add_argument("--approximate_poses_filename", type=str, default='', help='approximate_poses_filename')

    parser.add_argument("--centring_transforms", type=str, help='centring_transforms')

    parser.add_argument("--img_loss_threshold", type=float, help='img_loss_threshold')

    parser.add_argument("--depth_loss_threshold", type=float, help='depth_loss_threshold')

    parser.add_argument("--epochs", type=int, default=10, help='number of epochs')

    parser.add_argument("--output_directory", type=str, default='./', help='output_directory')

    parser.add_argument("--use_huber_loss", action='store_true', help='use_huber_loss')

    parser.add_argument("--image_dir_override", type=str, default=None, help='image_dir_override')

    parser.add_argument("--resample", type=float, default=1.0, help='resample')

    parser.add_argument("--object_radius", type=float, default=0.259425, help='object_radius default driller: 0.259425')

    parser.add_argument("--colmap_db_filename", type=str, default='./data/linemod_driller_all_llff/database.db', help='location of colmap db')

    parser.add_argument("--colmap_keypoints_filename", type=str, default=None, help='location of colmap keypoints pkl')

    parser.add_argument("--number_of_keypoints", type=int, default=None, help='number of unique keypoints')

    parser.add_argument("--keypoint_embedding_size", type=int, default=None, help='keypoint_embedding_size')
    return parser