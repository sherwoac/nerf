#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import tensorflow as tf

import numpy as np
import imageio
import json
import random
import time
import pprint

import matplotlib.pyplot as plt

import run_nerf

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
import OUTPUT.image_tools
tf.compat.v1.enable_eager_execution()


def show_test_images_at_c2w(Ts, test_image, render_kwargs, titles=[]):
    H, W, focal = test_image.shape[0], test_image.shape[1], 584.
    image_list = [255. * test_image]
    for T in Ts:
        # render_kwargs.c2w = np.eye(4, dtype=np.float32)
        rgb, disp, acc, extras = run_nerf.render(H, W, focal, c2w=T, **render_kwargs)
        clipped_rgb = np.clip(rgb, 0, 1)
        image_list.append(255. * clipped_rgb)

    tiled_images = OUTPUT.image_tools.tile_images(np.array([image_list]))
    plt.imshow(tiled_images)
    plt.suptitle(','.join(titles))
    plt.show()


basedir = './logs'
expname = 'linemod_driller_back_to_eggbox_similar_layout_no_ndc'

exp_dir = os.path.join(basedir, expname) 
config = os.path.join(exp_dir, 'config.txt')
print('Args:')
print(open(config, 'r').read())
parser = run_nerf.config_parser()

args = parser.parse_args('--config {} --ft_path {}'.format(config, os.path.join(basedir, expname, 'model_470000.npy')))
print('loaded args')

images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor, 
                                                          recenter=True, bd_factor=.75, 
                                                          spherify=args.spherify)
H, W, focal = poses[0,:3,-1].astype(np.float32)
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
import json
json.dump(vars(args), open("/tmp/args.json", 'w'))
_, render_kwargs_test, start, grad_vars, models = run_nerf.create_nerf(args)


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
# c2w = poses[i_test, :3, :4].astype(np.float32)

print('Render kwargs:')
pprint.pprint(render_kwargs_fast)

test = run_nerf.render(H//down, W//down, focal/down, c2w=c2w, **render_kwargs_fast)

img = np.clip(test[0],0,1)
plt.imshow(img)
plt.show()

test_image = images[i_test]
show_test_images_at_c2w([poses[i_test, :3, :4]], test_image, render_kwargs_fast)

# In[4]:


# down = 8 # trade off resolution+aliasing for render speed to make this video faster
# frames = []
# for i, c2w in enumerate(render_poses):
#     if i%8==0: print(i)
#     test = run_nerf.render(H//down, W//down, focal/down, c2w=c2w[:3,:4], **render_kwargs_fast)
#     frames.append((255*np.clip(test[0],0,1)).astype(np.uint8))
    
# print('done, saving')

# f = os.path.join(exp_dir, 'video.mp4')
# imageio.mimwrite(f, frames, fps=30, quality=8)

# from IPython.display import Video
# Video(f, height=320)


# In[5]:


# %matplotlib inline
# from ipywidgets import interactive, widgets
# import matplotlib.pyplot as plt
# import numpy as np


# def f(x, y, z):
    
#     c2w = tf.convert_to_tensor([
#         [1,0,0,x],
#         [0,1,0,y],
#         [0,0,1,z],
#         [0,0,0,1],
#     ], dtype=tf.float32)
    
#     test = run_nerf.render(H//down, W//down, focal/down, c2w=c2w, **render_kwargs_fast)
#     img = np.clip(test[0],0,1)
    
#     plt.figure(2, figsize=(20,6))
#     plt.imshow(img)
#     plt.show()
    

# sldr = lambda : widgets.FloatSlider(
#     value=0.,
#     min=-1.,
#     max=1.,
#     step=.01,
# )

# names = ['x', 'y', 'z']
    
# interactive_plot = interactive(f, **{n : sldr() for n in names})
# interactive_plot


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
from ipywidgets import interactive, widgets

down = 1

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


def f(**kwargs):
    print(f'**kwargs:{kwargs}')
    focal = kwargs['focal']
    del kwargs['focal']
    c2w = pose_spherical(**kwargs)
    test = run_nerf.render(H//down, W//down, focal/down, c2w=c2w, **render_kwargs_fast)
    img = np.clip(test[0],0,1)
#     img = depth
    # img = sigma
    plt.figure(2, figsize=(20,6))
    plt.imshow(img)
    plt.show()
    

sldr = lambda v, mi, ma: widgets.FloatSlider(
    value=v,
    min=mi,
    max=ma,
    step=.01,
    continuous_update=False
)

names = [
    ['theta', [180., 0., 360]],
    ['phi', [-80., -90, 0]],
    ['radius', [1.5, 1., 10.]],
    ['focal', [584., 10., 1000.]],
]

interactive_plot = interactive(f, **{s[0] : sldr(*s[1]) for s in names})
output = interactive_plot.children[-1]
interactive_plot


# In[ ]:




