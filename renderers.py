from utils import *

def sample_points_along_rays(
    near_depth: float,
    far_depth: float,
    num_samples: int,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    device: torch.device
):
    # Compute a linspace of num_samples depth values beetween near_depth and far_depth.
    z_vals = torch.linspace(near_depth, far_depth, num_samples, device=device)

    # Using the ray_origins, ray_directions, generate 3D points along
    # the camera rays according to the z_vals.
    pts = ray_origins[...,None,:] + ray_directions[...,None,:] * z_vals[...,:,None]

    return pts, z_vals

def volume_integral(
    z_vals: torch.tensor,
    sigmas: torch.tensor,
    radiances: torch.tensor
) -> Tuple[torch.tensor, torch.tensor]:

    # Compute the deltas in depth between the points.
    dists = torch.cat([
        z_vals[..., 1:] - z_vals[..., :-1], 
        torch.broadcast_to(torch.Tensor([1e10]).to(z_vals.device), z_vals[...,:1].shape)
        ], -1) 

    # Compute the alpha values from the densities and the dists.
    # Tip: use torch.einsum for a convenient way of multiplying the correct 
    # dimensions of the sigmas and the dists.
    alpha = 1.- torch.exp(-torch.einsum('brzs, z -> brzs', sigmas, dists))

    # Compute the Ts from the alpha values. Use torch.cumprod.
    Ts = torch.cumprod(1.-alpha+1e-10, -2)

    # Compute the weights from the Ts and the alphas.
    weights = alpha * Ts
    
    # Compute the pixel color as the weighted sum of the radiance values.
    rgb = torch.einsum('brzs, brzs -> brs', weights, radiances)

    # Compute the depths as the weighted sum of z_vals.
    # Tip: use torch.einsum for a convenient way of computing the weighted sum,
    # without the need to reshape the z_vals.
    depth_map = torch.einsum('brzs, z -> brs', weights, z_vals)

    return rgb, depth_map, weights


class VolumeRenderer(nn.Module):
    def __init__(self, near, far, n_coarse=32, n_fine=16, n_fine_depth=8, depth_std = 0.01, white_back=True):
        super().__init__()
        self.near = near
        self.far = far
        self.n_coarse = n_coarse
        self.n_fine = n_fine
        self.n_fine_depth = n_fine_depth
        self.depth_std = depth_std
        self.white_back = white_back

    def forward(
        self, 
        cam2world,
        intrinsics,
        x_pix,
        radiance_field: nn.Module
        ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Takes as inputs ray origins and directions - samples points along the 
        rays and then calculates the volume rendering integral. 

        Params:
            input_dict: Dictionary with keys 'cam2world', 'intrinsics', and 'x_pix'
                here x_pix has size (B,2)
            radiance_field: nn.Module instance of the radiance field we want to render. This should by default the PixelNeRF. 
                Params: 
                    xyz: points in space, with shape (SB,B,3)
                        SB is batch of objects
                        B is batch of points (in rays)
                Return:
                    (SB, B, 4) of r g b sigma

        Returns:
            Tuple of rgb, depth_map
            rgb: for each pixel coordinate x_pix, the color of the respective ray.
            depth_map: for each pixel coordinate x_pix, the depth of the respective ray.
 
        """
        batch_size, num_rays = x_pix.shape[0], x_pix.shape[1], x_pix.shape[2]  # _ should be 2, since each pixel coordinate has 2 inputs

        # Compute the ray directions in world coordinates.
        # Use the function get_world_rays.
        ros, rds = get_world_rays(x_pix, intrinsics, cam2world)

        # Generate the points along rays and their depth values
        # Use the function sample_points_along_rays.
        pts, z_vals = sample_points_along_rays(self.near, self.far, self.n_coarse, 
                                                ros, rds, device=x_pix.device)

        # Reshape pts to (batch_size, -1, 3).
        pts = pts.reshape(1, batch_size, -1, 3)  # To comply with pixelnerf input

        # Sample the radiance field with the points along the rays.
        sigma_rad = radiance_field(pts)
        sigma_rad = sigma_rad.squeeze(0)   # Get rid of SB dimension
        sigma = sigma_rad[...,3]
        rad = sigma_rad[...,:3]


        # Reshape sigma and rad back to (batch_size, num_rays, self.n_samples, -1)
        sigma = sigma.view(batch_size, num_rays, self.n_coarse, 1)
        rad = rad.view(batch_size, num_rays, self.n_coarse, 3)

        # Compute pixel colors, depths, and weights via the volume integral.
        rgb, depth_map, weights = volume_integral(z_vals, sigma, rad)

        if self.white_back:
            accum = weights.sum(dim=-2)
            rgb = rgb + (1. - accum)

        return rgb, depth_map

    @classmethod
    def from_conf(cls, conf, white_back=False):
        return cls(
            conf.get_int("n_coarse", 32),
            conf.get_int("n_fine", 16),
            n_fine_depth=conf.get_int("n_fine_depth", 8),
            depth_std=conf.get_float("depth_std", 0.01),
            white_back=conf.get_float("white_back", white_back),
        )
    
