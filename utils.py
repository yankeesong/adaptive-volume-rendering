import torch, torchvision
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Optional, Tuple, Generator, Dict
import os
import sys
import gdown
import h5py
import random

import imageio
import skimage
import io
import warnings

import math
import collections
import traceback
from einops import rearrange, repeat

from dotmap import DotMap
from pyhocon import ConfigFactory
from glob import glob
from skimage.transform import resize

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




# Multiview helper functions
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


def get_unnormalized_cam_ray_directions(xy_pix:torch.Tensor,
                                        intrinsics:torch.Tensor) -> torch.Tensor:
    return unproject(xy_pix, -torch.ones_like(xy_pix[..., :1], device=xy_pix.device),  intrinsics=intrinsics)  
       
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
    # Currently, ray origins have shape (batch, 3), while ray directions have shape
    cam_origin_world = repeat(cam_origin_world, 'b ch -> b num_rays ch', num_rays=ray_dirs_cam.shape[1])

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

# functions from SRN
def parse_intrinsics(intrinsics):
    # intrinsics = intrinsics.cuda()

    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    return fx, fy, cx, cy


def expand_as(x, y):
    if len(x.shape) == len(y.shape):
        return x

    for i in range(len(y.shape) - len(x.shape)):
        x = x.unsqueeze(-1)

    return x


def lift(x, y, z, intrinsics, homogeneous=False):
    '''

    :param self:
    :param x: Shape (batch_size, num_points)
    :param y:
    :param z:
    :param intrinsics:
    :return:
    '''
    fx, fy, cx, cy = parse_intrinsics(intrinsics)

    x_lift = (x - expand_as(cx, x)) / expand_as(fx, x) * z
    y_lift = (y - expand_as(cy, y)) / expand_as(fy, y) * z

    if homogeneous:
        return torch.stack((x_lift, y_lift, z, torch.ones_like(z).to(z.device)), dim=-1)
    else:
        return torch.stack((x_lift, y_lift, z), dim=-1)

def world_from_xy_depth(xy, depth, cam2world, intrinsics):
    '''Translates meshgrid of xy pixel coordinates plus depth to  world coordinates.
    '''
    batch_size, _, _ = cam2world.shape

    x_cam = xy[:, :, 0].view(batch_size, -1)
    y_cam = xy[:, :, 1].view(batch_size, -1)
    z_cam = depth.view(batch_size, -1)

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics, homogeneous=True)  # (batch_size, -1, 4)

    # permute for batch matrix product
    pixel_points_cam = pixel_points_cam.permute(0, 2, 1)

    world_coords = torch.bmm(cam2world, pixel_points_cam).permute(0, 2, 1)[:, :, :3]  # (batch_size, -1, 3)

    return world_coords

def get_ray_directions(xy, cam2world, intrinsics):
    '''Translates meshgrid of xy pixel coordinates to normalized directions of rays through these pixels.
    '''
    batch_size, num_samples, _ = xy.shape

    z_cam = torch.ones((batch_size, num_samples)).to(xy.device)
    pixel_points = world_from_xy_depth(xy, z_cam, intrinsics=intrinsics, cam2world=cam2world)  # (batch, num_samples, 3)

    cam_pos = cam2world[:, :3, 3]
    ray_dirs = pixel_points - cam_pos[:, None, :]  # (batch, num_samples, 3)
    ray_dirs = F.normalize(ray_dirs, dim=2)
    return ray_dirs


def depth_from_world(world_coords, cam2world):
    batch_size, num_samples, _ = world_coords.shape

    points_hom = torch.cat((world_coords, torch.ones((batch_size, num_samples, 1)).to(world_coords.device)),
                           dim=2)  # (batch, num_samples, 4)

    # permute for bmm
    points_hom = points_hom.permute(0, 2, 1)

    points_cam = torch.inverse(cam2world).bmm(points_hom)  # (batch, 4, num_samples)
    depth = -points_cam[:, 2, :][:, :, None]  # (batch, num_samples, 1)
    return depth

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

def calculate_psnr(rgb, gt):
    rgb = rgb.squeeze().detach().cpu().numpy()
    rgb = (rgb/ 2.) + 0.5
    rgb = np.clip(rgb, a_min=0., a_max=1.)
    
    gt = gt.squeeze().detach().cpu().numpy()
    gt = (gt / 2.) + 0.5
    
    ssim = skimage.measure.compare_ssim(rgb, gt, multichannel=True, data_range=1)
    psnr = skimage.measure.compare_psnr(rgb, gt, data_range=1)

    return psnr, ssim