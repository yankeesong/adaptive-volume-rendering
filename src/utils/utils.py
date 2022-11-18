import torch, torchvision
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import IterableDataset
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Optional, Tuple, Generator, Dict
import os
import gdown
import h5py
import random

import imageio
import os
import skimage
import h5py
import io

import math
from einops import rearrange, repeat


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
    return unproject(xy_pix, torch.ones_like(xy_pix[..., :1], device=xy_pix.device),  intrinsics=intrinsics)     


def get_world_rays(xy_pix: torch.Tensor, 
                   intrinsics: torch.Tensor,
                   cam2world: torch.Tensor,
                   ) -> torch.Tensor:
    # Get camera origin of camera 1
    cam_origin_world = cam2world[..., :3, -1]

    # Get ray directions in cam coordinates
    ray_dirs_cam = get_unnormalized_cam_ray_directions(xy_pix, intrinsics)

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
    i, j = torch.meshgrid(torch.linspace(0, 1, steps=x_resolution), 
                          torch.linspace(0, 1, steps=y_resolution))

    xy_pix = torch.stack([i.float(), j.float()], dim=-1).permute(1, 0, 2)
    return xy_pix