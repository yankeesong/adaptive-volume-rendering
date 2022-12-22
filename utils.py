import torch, torchvision
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Optional, Tuple, Generator, Dict
import lpips
import os
import sys
import gdown
import h5py
import random

import imageio
import skimage
import io
import warnings
import time

import math
import collections
import traceback
from einops import rearrange, repeat

from dotmap import DotMap
from pyhocon import ConfigFactory
from glob import glob
from skimage.transform import resize

# Indexing functions
def batched_index_select_nd(t, inds):
    """
    Index select on dim 1 of a n-dimensional batched tensor. From PixelNeRF
    :param t (batch, n, ...)
    :param inds (batch, k)
    :return (batch, k, ...)
    """
    return t.gather(
        1, inds[(...,) + (None,) * (len(t.shape) - 2)].expand(-1, -1, *t.shape[2:])
    )

def batched_index_select_nd_second(t, inds):
    """
    Index select on dim 2 of a >=2D multi-batched tensor. inds assumed
    to have all batch dimensions except one data dimension 'n'
    :param t (batch, n, m, ...)
    :param inds (batch, n, k)
    :return (batch, n, k, ...)
    """
    return t.gather(
        2, inds[(...,) + (None,) * (len(t.shape) - 3)].expand(-1, -1, -1, *t.shape[3:])
    )

def data_loop(dl):
    """
    Loop an iterable infinitely
    """
    while True:
        for x in iter(dl):
            yield x

# NN functions
def to_gpu(ob):
    if isinstance(ob, collections.Mapping):
        return {k: to_gpu(v) for k, v in ob.items()}
    elif isinstance(ob, tuple):
        return tuple(to_gpu(k) for k in ob)
    elif isinstance(ob, list):
        return [to_gpu(k) for k in ob]
    else:
        try:
            return ob.cuda()
        except:
            return ob

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)

def init_recurrent_weights(self):
    for m in self.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)


def lstm_forget_gate_init(lstm_layer):
    for name, parameter in lstm_layer.named_parameters():
        if not "bias" in name: continue
        n = parameter.size(0)
        start, end = n // 4, n // 2
        parameter.data[start:end].fill_(1.)


def clip_grad_norm_hook(x, max_norm=10):
    total_norm = x.norm()
    total_norm = total_norm ** (1 / 2.)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        return x * clip_coef


# Geometry
def homogenize_points(points: torch.Tensor):
    """Appends a "1" to the coordinates of a (batch of) points of dimension DIM.

    Args:
        points: points of shape (..., DIM)

    Returns:
        points_hom: points with appended "1" dimension.
    """
    ones = torch.ones_like(points[..., :1], device=points.device)
    return torch.cat((points, ones), dim=-1)


def homogenize_vecs(vectors: torch.Tensor):
    """Appends a "0" to the coordinates of a (batch of) vectors of dimension DIM.

    Args:
        vectors: vectors of shape (..., DIM)

    Returns:
        vectors_hom: points with appended "0" dimension.
    """
    zeros = torch.zeros_like(vectors[..., :1], device=vectors.device)
    return torch.cat((vectors, zeros), dim=-1)


def unproject(
    xy_pix: torch.Tensor, 
    z: torch.Tensor, 
    intrinsics: torch.Tensor
    ) -> torch.Tensor:
    """Unproject (lift) 2D pixel coordinates x_pix and per-pixel z coordinate
    to 3D points in camera coordinates.

    Args:
        xy_pix: 2D pixel coordinates of shape (..., 2)
        z: per-pixel depth, defined as z coordinate of shape (..., 1) 
        intrinscis: camera intrinscics of shape (..., 3, 3)

    Returns:
        xyz_cam: points in 3D camera coordinates.
    """
    xy_pix_hom = homogenize_points(xy_pix)
    xyz_cam = torch.einsum('...ij,...kj->...ki', intrinsics.inverse(), xy_pix_hom)
    x_cam = xyz_cam[...,0]
    xyz_cam[...,0] = -x_cam
    xyz_cam *= z
    return xyz_cam
    

def transform_world2cam(xyz_world_hom: torch.Tensor, cam2world: torch.Tensor) -> torch.Tensor:
    """Transforms points from 3D world coordinates to 3D camera coordinates.

    Args:
        xyz_world_hom: homogenized 3D points of shape (..., 4)
        cam2world: camera pose of shape (..., 4, 4)

    Returns:
        xyz_cam: points in camera coordinates.
    """
    world2cam = torch.inverse(cam2world)
    return transform_rigid(xyz_world_hom, world2cam)


def transform_cam2world(xyz_cam_hom: torch.Tensor, cam2world: torch.Tensor) -> torch.Tensor:
    """Transforms points from 3D world coordinates to 3D camera coordinates.

    Args:
        xyz_cam_hom: homogenized 3D points of shape (..., 4)
        cam2world: camera pose of shape (..., 4, 4)

    Returns:
        xyz_world: points in camera coordinates.
    """
    return transform_rigid(xyz_cam_hom, cam2world)


def transform_rigid(xyz_hom: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Apply a rigid-body transform to a (batch of) points / vectors.

    Args:
        xyz_hom: homogenized 3D points of shape (..., 4)
        T: rigid-body transform matrix of shape (..., 4, 4)

    Returns:
        xyz_trans: transformed points.
    """ 
    return torch.einsum('...ij,...kj->...ki', T, xyz_hom)
       
def get_normalized_cam_ray_directions(xy_pix:torch.Tensor,
                                        intrinsics:torch.Tensor) -> torch.Tensor:
    unnormalized_rays = unproject(xy_pix, -torch.ones_like(xy_pix[..., :1], device=xy_pix.device),  intrinsics=intrinsics)
    return unnormalized_rays/torch.norm(unnormalized_rays, dim=-1).unsqueeze(-1)


def get_world_rays(xy_pix: torch.Tensor, 
                   intrinsics: torch.Tensor,
                   cam2world: torch.Tensor,
                   ) -> torch.Tensor:
    # Get camera origin of camera 1
    cam_origin_world = cam2world[..., :3, -1]

    # Get ray directions in cam coordinates
    ray_dirs_cam = get_normalized_cam_ray_directions(xy_pix, intrinsics)

    # Homogenize ray directions
    rd_cam_hom = homogenize_vecs(ray_dirs_cam)

    # Transform ray directions to world coordinates
    rd_world_hom = transform_cam2world(rd_cam_hom, cam2world)

    # Tile the ray origins to have the same shape as the ray directions.
    # Currently, ray origins have shape (SB, NV, 3), while ray directions have shape (SB, NV, num_rays, 3)
    cam_origin_world = repeat(cam_origin_world, 'b nv ch -> b nv num_rays ch', num_rays=ray_dirs_cam.shape[2])

    # Return tuple of cam_origins, ray_world_directions
    return cam_origin_world, rd_world_hom[..., :3]


def get_opencv_pixel_coordinates(
    y_resolution: int,
    x_resolution: int,
    ):
    """For an image with y_resolution and x_resolution, return a tensor of pixel coordinates
    normalized to lie in [0, 1], with the origin (0, 0) in the top left corner,
    the x-axis pointing right, the y-axis pointing down, and the bottom right corner
    being at (1, 1).

    Returns:
        xy_pix: a meshgrid of values from [0, 1] of shape 
                (y_resolution, x_resolution, 2)
    """
    i, j = torch.meshgrid(torch.linspace(0, 1-1/x_resolution, steps=x_resolution), 
                          torch.linspace(0, 1-1/x_resolution, steps=y_resolution))

    xy_pix = torch.stack([i.float(), j.float()], dim=-1).permute(1, 0, 2)
    return xy_pix

def depth_from_world(world_coords, cam2world):
    points_hom = homogenize_points(world_coords)  # (batch, NV, num_samples, 4)
    points_cam = transform_world2cam(points_hom,cam2world)  # (batch, NV, num_samples, 4)
    return -points_cam[...,2]  # (batch, NV, num_samples, 1)

# Loss functions
def mse_loss(mlp_out, gt):
    _, img, _ = mlp_out
    return ((img - gt)**2).mean()

def mse_loss_fine(mlp_out, gt):
    img_coarse, img_fine, _ = mlp_out
    return ((img_coarse - gt)**2).mean() + ((img_fine - gt)**2).mean()

def mse_regularization_loss(mlp_out, gt, near=0.5, far=2.0):
    # add regularization loss
    _, img, depth = mlp_out
    penalty = (torch.min(depth-near, torch.zeros_like(depth)) ** 2) + (torch.max(depth-far, torch.zeros_like(depth)) ** 2)
    return ((img - gt)**2).mean() + torch.mean(penalty) * 10000

# Plotting functions
def plot_output_ground_truth(vis_output, vis_gt, resolution, src_idx):
    _, vis_img, vis_depth = vis_output

    SB = vis_img.shape[0]

    _, axes = plt.subplots(SB, 3, figsize=(18, 6*SB), squeeze=False)

    for sb in range(SB):
        img = vis_img[sb,0]
        depth = vis_depth[sb,0]
        gt = vis_gt[sb,0]

        axes[sb, 0].imshow(img.cpu().view(*resolution).detach().numpy())
        if src_idx[sb,0] == 0:
            title = "Trained MLP (same view as src)"
        else:
            title = "Trained MLP"
        axes[sb, 0].set_title(title)
        axes[sb, 1].imshow(gt.cpu().view(*resolution).detach().numpy())
        axes[sb, 1].set_title("Ground Truth")
        axes[sb, 2].imshow(depth.cpu().view(*resolution[:2]).detach().numpy(), cmap='Greys')
        axes[sb, 2].set_title("Depth")
        
        for j in range(3):
            axes[sb,j].set_axis_off()
    plt.show()

def get_metrics(mlp_out, gts): # rgb and gt are both 0 to 1 already
    _, rgbs, _ = mlp_out
    SB, NV, sl2, _ = rgbs.shape
    sl = int(np.sqrt(sl2))

    rgbs = rgbs.reshape(SB, NV, sl, sl, 3).detach().cpu().numpy()
    gts = gts.reshape(SB, NV, sl, sl, 3).detach().cpu().numpy()

    psnrs = []
    ssims = []

    for sb in range(SB):
        total_psnr = 0
        total_ssim = 0

        for nv in range(NV):
            ssim = skimage.measure.structural_similarity(rgbs[sb,nv,...], gts[sb,nv,...], channel_axis=-1, data_range=1)
            psnr = skimage.measure.peak_signal_noise_ratio(rgbs[sb,nv,...], gts[sb,nv,...], data_range=1)
            total_ssim += ssim
            total_psnr += psnr
        
        psnrs.append(total_psnr/NV)
        ssims.append(total_ssim/NV)

    return np.mean(psnr), np.mean(ssim)

def get_lpips(mlp_out, gts):
    _, rgbs, _ = mlp_out
    SB, NV, sl2, _ = rgbs.shape
    sl = int(np.sqrt(sl2))

    rgbs = rgbs.reshape(SB, NV, sl, sl, 3).permute(0,1,4,2,3).detach() * 2 - 1 # (SB, NV, 3, sl, sl)
    gts = gts.reshape(SB, NV, sl, sl, 3).permute(0,1,4,2,3).detach() * 2 - 1 # (SB, NV, 3, sl, sl)
    lpips_vgg = to_gpu(lpips.LPIPS(net="vgg"))

    lpipss = []

    for sb in range(SB):
        lpipss.append(lpips_vgg(rgbs[sb],gts[sb]))
    lpips_all = torch.cat(lpipss)

    return lpips_all.mean().item().cpu().numpy()

# utils for generating videos
def get_R(x,y,z):
    camera_position = torch.Tensor([x,y,z]).reshape(1,3)
    at = torch.Tensor([0,0,0]).reshape(1,3)
    up = torch.Tensor([0,0,-1]).reshape(1,3)

    z_axis = F.normalize(at - camera_position, eps=1e-5)
    x_axis = F.normalize(torch.cross(up, z_axis, dim=1), eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)
    is_close = torch.isclose(x_axis, torch.tensor(0.0), atol=5e-3).all(
        dim=1, keepdim=True
    )
    if is_close.any():
        replacement = F.normalize(torch.cross(y_axis, z_axis, dim=1), eps=1e-5)
        x_axis = torch.where(is_close, replacement, x_axis)
    R = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)
    return R.transpose(1, 2)
    
def generate_video(model_input, num_frames, radius, net, model):
    # Encode the model
    model_input = to_gpu(model_input)
    ground_truth = model_input["images"] # (NV, H*W, 3), values are between 0 and 1

    NV, sl2, _ = ground_truth.shape
    sl = int(np.sqrt(sl2))
    idx = [0]

    src_images = ground_truth[idx,...].reshape(-1,sl,sl,3).permute(0,3,1,2)
    poses = model_input['cam2world'][idx,...]
    focal = model_input['focal'][idx,...]
    c = model_input['c'][idx,...]
    intrinsics = model_input['intrinsics'][idx,...]
    net.encode(src_images, poses, focal, c)
    
    # Generate a round of camera views
    angles = torch.linspace(0,2*np.pi,num_frames)
    cam2world = []
    for i in range(num_frames):
        angle = angles[i]
        tx = radius * np.sin(angle)
        ty = radius * np.cos(angle)
        tz = 0.4
        R = get_R(tx,ty,tz)
        c2w = np.zeros((4,4))
        c2w[:3,:3] = R
        c2w[0,3] = tx
        c2w[1,3] = ty
        c2w[2,3] = tz
        c2w[3,3] = 1
        c2w = torch.Tensor(c2w).float() @ torch.diag(torch.tensor([1, -1, -1, 1], dtype=torch.float32))
        cam2world.append(c2w)
                                                  
    # Render out images
    x_pix = get_opencv_pixel_coordinates(sl, sl)
    x_pix = rearrange(x_pix, 'i j c -> (i j) c')
    with torch.no_grad():
        frames = []
        for i in range(len(cam2world)):
            model_input = {'cam2world': cam2world[i].reshape(1,4,4), 'intrinsics': intrinsics, 
                        'focal':focal,'c':c,'x_pix': x_pix.reshape(1,-1,2)}
            model_input = to_gpu(model_input)
            model_output = model(model_input)
            img, depth = model_output
            img = img[0].reshape(sl, sl, 3).cpu().numpy()
            img *= 255
            img = np.clip(img, 0, 255).astype(np.uint8)
            frames.append(img)
    return frames