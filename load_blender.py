import os
import tensorflow as tf
import numpy as np
import imageio 
import json
import tqdm

from load_llff import load_masks
from NERFCO.create_linemod_data import linemod_camera_intrinsics


trans_t = lambda t : tf.convert_to_tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
], dtype=tf.float32)

rot_phi = lambda phi : tf.convert_to_tensor([
    [1,0,0,0],
    [0,tf.cos(phi),-tf.sin(phi),0],
    [0,tf.sin(phi), tf.cos(phi),0],
    [0,0,0,1],
], dtype=tf.float32)

rot_theta = lambda th : tf.convert_to_tensor([
    [tf.cos(th),0,-tf.sin(th),0],
    [0,1,0,0],
    [tf.sin(th),0, tf.cos(th),0],
    [0,0,0,1],
], dtype=tf.float32)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w
    

def linemod_dpt(path):
    """
    read a depth image

    @return uint16 image of distance in [mm]"""
    dpt = open(path, "rb")
    rows = np.frombuffer(dpt.read(4), dtype=np.int32)[0]
    cols = np.frombuffer(dpt.read(4), dtype=np.int32)[0]

    return (np.fromfile(dpt, dtype=np.uint16).reshape((rows, cols)) / 1000.).astype(np.float32)


def smooth_kinect_depth(depth_raw):
    import scipy.interpolate
    depth = depth_raw / 1000.
    x = np.arange(depth_raw.shape[1])
    y = np.arange(depth_raw.shape[0])
    # mask invalid values
    array = np.ma.masked_where((depth_raw == 2 ** 16 - 1) | (depth_raw == 0), depth)
    # array.mask = invalid_depth_mask
    xx, yy = np.meshgrid(x, y)
    # get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]
    depth = scipy.interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method='nearest')
    return depth.astype(np.float32)


def load_blender_data(basedir, half_res=False, testskip=1, image_extn='.png', get_depths=False, mask_directory=None,
                      image_field='file_path', image_dir_override=None, trainskip=1, test_finish=None, train_frames_field='frames'):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        json_filename = os.path.join(basedir, f'transforms_{s}.json')
        with open(json_filename, 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    all_depth_maps = []
    counts = [0]
    all_filenames = []
    all_depth_filenames = []
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        depth_maps = []
        if s == 'train' or testskip == 0:
            frame_slice = slice(None, None, trainskip)
            frames_field = train_frames_field
            assert frames_field in meta, f'frames_field: {frames_field} not in meta: {meta.keys()}'
        else:
            frame_slice = slice(None, test_finish, testskip)
            frames_field = 'frames'
        filenames = []
        depth_filenames = []
        for frame in tqdm.tqdm(meta[frames_field][frame_slice]):
            if any(ext in frame[image_field] for ext in ['png', 'jpg']):
                image_filename = os.path.join(basedir, frame[image_field])
            else:
                image_filename = os.path.join(basedir, frame[image_field] + image_extn)

            if image_dir_override is not None:
                image_filename = os.path.join(image_dir_override, os.path.basename(image_filename))

            assert os.path.isfile(image_filename), f'fname not found at: {image_filename}'
            filenames.append(image_filename)

            image = imageio.imread(image_filename)
            imgs.append(image)

            pose = np.array(frame['transform_matrix'])
            poses.append(pose)

            # depth
            if 'depth_path' in frame:
                #eg. r_0_depth_0001.png
                depth_filename = frame['depth_path']
                depth_filenames.append(depth_filename)
                if get_depths:
                    assert os.path.isfile(depth_filename), f'filename:{depth_filename} not found'
                    if '.png' in depth_filename:
                        # kinect depth
                        depth_raw = imageio.imread(depth_filename)
                        if s == 'train':
                            depth = smooth_kinect_depth(depth_raw)
                        else:
                            depth = depth_raw / 1000.

                    elif '.dpt' in depth_filename:
                        depth = linemod_dpt(depth_filename)

                    depth_maps.append(depth)

        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
        all_depth_maps.append(depth_maps)
        all_filenames.append(filenames)
        all_depth_filenames.append(depth_filenames)

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    extras = {}
    if mask_directory is not None:
        list_of_image_files = [item for sublist in all_filenames for item in sublist]
        extras['masks'] = load_masks(basedir, mask_directory, list_of_image_files)

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = tf.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]],0)
    
    if half_res:
        imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
        H = H//2
        W = W//2
        focal = focal/2.

    if get_depths:
        all_depth_maps = np.concatenate(all_depth_maps, 0)
        extras['depth_maps'] = np.stack(all_depth_maps, axis=0)

    if 'K' in meta:
        extras['K'] = np.array(meta['K'], dtype=np.float32)

    extras['filenames'] = all_filenames
    extras['depth_filenames'] = all_depth_filenames
    return imgs, poses, render_poses, [H, W, focal], i_split, extras



