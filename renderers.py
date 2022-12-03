from utils import *
import sys
#root_dir = "/Users/jameszli/desktop/MIT/6.S980/"  
#sys.path.insert(0, f"{root_dir}/scene-representation-networks/")
from geometry import *
from util import *
#from pytorch_prototyping import pytorch_prototyping

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

class DepthSampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                xy,
                depth,
                cam2world,
                intersection_net,
                intrinsics):
        self.logs = list()

        batch_size, _, _ = cam2world.shape

        intersections = world_from_xy_depth(xy=xy, depth=depth, cam2world=cam2world, intrinsics=intrinsics)

        depth = depth_from_world(intersections, cam2world)

        if self.training:
            print(depth.min(), depth.max())

        return intersections, depth

def sample_points_along_rays(
    near_depth: float,
    far_depth: float,
    num_samples: int,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    device: torch.device
):
    # Compute a linspace of num_samples depth values beetween near_depth and far_depth.
    z_vals = torch.linspace(near_depth, far_depth - (far_depth-near_depth)/num_samples, num_samples, device=device)

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
    def __init__(self, near=0.8, far=1.8, n_coarse=32, n_fine=16, n_fine_depth=8, depth_std = 0.01, white_back=True):
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
        NV, num_rays, _ = x_pix.shape  # _ should be 2, since each pixel coordinate has 2 inputs

        # Compute the ray directions in world coordinates.
        # Use the function get_world_rays.
        ros, rds = get_world_rays(x_pix, intrinsics, cam2world)

        # Generate the points along rays and their depth values
        # Use the function sample_points_along_rays.
        pts, z_vals = sample_points_along_rays(self.near, self.far, self.n_coarse, 
                                                ros, rds, device=x_pix.device) # pts has shape (NV, num_rays, 3)
        # print(f'pts has shape {pts.shape}')
        # print(f'rds has shape {rds.shape}')
        rds = rds.unsqueeze(2).expand(NV, num_rays, self.n_coarse, -1)

        # 
        pts = pts.reshape(1,-1,3)  # (1, NV*num_ways, 3)
        rds = rds.reshape(1,-1,3)

        # Input view directions
        # (NV*num_ways, 3)

        # Sample the radiance field with the points along the rays.
        sigma_rad = radiance_field(pts, viewdirs=rds) # (SB, NV*num_rays, 4)
        sigma_rad = sigma_rad.squeeze(0)   # Get rid of SB dimension
        sigma = sigma_rad[...,3]
        rad = sigma_rad[...,:3]


        # Reshape sigma and rad back to (NV, num_rays, self.n_samples, -1)
        sigma = sigma.view(NV, num_rays, self.n_coarse, 1)
        rad = rad.view(NV, num_rays, self.n_coarse, 3)

        # Compute pixel colors, depths, and weights via the volume integral.
        rgb, depth_map, weights = volume_integral(z_vals, sigma, rad)

        if self.white_back:
            accum = weights.sum(dim=-2)
            rgb = rgb + (1. - accum)

        return rgb, depth_map

    @classmethod
    def from_conf(cls, conf, white_back=False):
        return cls(
            near=conf.get_float("near", 0.8),
            far=conf.get_float("far", 1.8),
            n_coarse=conf.get_int("n_coarse", 32),
            n_fine=conf.get_int("n_fine", 16),
            n_fine_depth=conf.get_int("n_fine_depth", 8),
            depth_std=conf.get_float("depth_std", 0.01),
            white_back=conf.get_float("white_back", white_back),
        )

class AdaptiveVolumeRenderer(nn.Module):
    def __init__(self, near=1.0, far=2.5, n_coarse=32, n_fine=16, n_fine_depth=8, depth_std = 0.01, white_back=True):
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
        NV, num_rays, _ = x_pix.shape  # _ should be 2, since each pixel coordinate has 2 inputs

        # Compute the ray directions in world coordinates.
        # Use the function get_world_rays.
        ros, rds = get_world_rays(x_pix, intrinsics, cam2world)

        # Generate the points along rays and their depth values
        # Use the function sample_points_along_rays.
        pts, z_vals = sample_points_along_rays(self.near, self.far, self.n_coarse, 
                                                ros, rds, device=x_pix.device) # pts has shape (NV, num_rays, 3)
        # print(f'pts has shape {pts.shape}')
        # print(f'rds has shape {rds.shape}')
        rds = rds.unsqueeze(2).expand(NV, num_rays, self.n_coarse, -1)

        # 
        pts = pts.reshape(1,-1,3)  # (1, NV*num_ways, 3)
        rds = rds.reshape(1,-1,3)

        # Input view directions
        # (NV*num_ways, 3)

        # Sample the radiance field with the points along the rays.
        sigma_rad = radiance_field(pts, viewdirs=rds)
        sigma_rad = sigma_rad.squeeze(0)   # Get rid of SB dimension
        sigma = sigma_rad[...,3]
        rad = sigma_rad[...,:3]


        # Reshape sigma and rad back to (NV, num_rays, self.n_samples, -1)
        sigma = sigma.view(NV, num_rays, self.n_coarse, 1)
        rad = rad.view(NV, num_rays, self.n_coarse, 3)

        # Compute pixel colors, depths, and weights via the volume integral.
        rgb, depth_map, weights = volume_integral(z_vals, sigma, rad)

        if self.white_back:
            accum = weights.sum(dim=-2)
            rgb = rgb + (1. - accum)

        return rgb, depth_map

    @classmethod
    def from_conf(cls, conf, white_back=False):
        return cls(
            near=conf.get_float("near", 1.0),
            far=conf.get_float("far", 2.5),
            n_coarse=conf.get_int("n_coarse", 32),
            n_fine=conf.get_int("n_fine", 16),
            n_fine_depth=conf.get_int("n_fine_depth", 8),
            depth_std=conf.get_float("depth_std", 0.01),
            white_back=conf.get_float("white_back", white_back),
        )

# Question: Should we train raymarcher first then train constant step volume renderer or train both at the same time?
class OldAdaptiveVolumeRenderer(nn.Module):
    def __init__(self, near, far, num_feature_channels, raymarch_steps, hidden_size = 16, n_coarse=32, n_fine=16, n_fine_depth=8, depth_std = 0.01, white_back=True):
        super().__init__()
        self.near = near
        self.far = far
        self.n_coarse = n_coarse
        self.n_fine = n_fine
        self.n_fine_depth = n_fine_depth
        self.depth_std = depth_std
        self.white_back = white_back

        self.n_feature_channels = num_feature_channels
        self.steps = raymarch_steps

        # hidden_size = 16
        self.lstm = nn.LSTMCell(input_size=self.n_feature_channels,
                                hidden_size=hidden_size)

        self.lstm.apply(init_recurrent_weights)
        lstm_forget_gate_init(self.lstm)

        self.out_layer = nn.Linear(hidden_size, 1)
        self.counter = 0
        

    def forward(
        self, 
        phi,
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
            phi: FC block if in SRN (NOTE: could also be radiance field here)

        Returns:
            Tuple of rgb, depth_map
            rgb: for each pixel coordinate x_pix, the color of the respective ray.
            depth_map: for each pixel coordinate x_pix, the depth of the respective ray.
 
        """
        batch_size, num_rays = x_pix.shape[0], x_pix.shape[1]  # _ should be 2, since each pixel coordinate has 2 inputs

        log = list()

        ray_dirs = get_ray_directions(x_pix,
                                               cam2world=cam2world,
                                               intrinsics=intrinsics) # from geometry (SRN)

        initial_depth = torch.zeros((batch_size, num_rays, 1)).normal_(mean=0.05, std=5e-4).cuda()
        init_world_coords = world_from_xy_depth(x_pix,
                                                initial_depth,
                                                intrinsics=intrinsics,
                                                cam2world=cam2world)  # from geometry (SRN)

        world_coords = [init_world_coords]
        depths = [initial_depth]
        states = [None]

        for step in range(self.steps):
            v = phi(world_coords[-1]) # could replace by radiance_field instead of phi

            state = self.lstm(v.view(-1, self.n_feature_channels), states[-1])

            if state[0].requires_grad:
                state[0].register_hook(lambda x: x.clamp(min=-10, max=10))

            signed_distance = self.out_layer(state[0]).view(batch_size, num_rays, 1)
            new_world_coords = world_coords[-1] + ray_dirs * signed_distance

            states.append(state)
            world_coords.append(new_world_coords)

            depth = depth_from_world(world_coords[-1], cam2world)  # from geometry (SRN)

            if self.training:
                print("Raymarch step %d: Min depth %0.6f, max depth %0.6f" %
                      (step, depths[-1].min().detach().cpu().numpy(), depths[-1].max().detach().cpu().numpy()))

            depths.append(depth)

        if not self.counter % 100:
            # Write tensorboard summary for each step of ray-marcher.
            drawing_depths = torch.stack(depths, dim=0)[:, 0, :, :]
            drawing_depths = lin2img(drawing_depths).repeat(1, 3, 1, 1) # from util (SRN)
            log.append(('image', 'raycast_progress',
                        torch.clamp(torchvision.utils.make_grid(drawing_depths, scale_each=False, normalize=True), 0.0,
                                    5),
                        100))

            # Visualize residual step distance (i.e., the size of the final step)
            fig = show_images([lin2img(signed_distance)[i, :, :, :].detach().cpu().numpy().squeeze()
                                    for i in range(batch_size)]) # from util (SRN)
            log.append(('figure', 'stopping_distances', fig, 100))
        self.counter += 1

        # Compute the ray directions in world coordinates.
        # Use the function get_world_rays.
        ros, rds = get_world_rays(x_pix, intrinsics, cam2world)

        # Generate the points along rays and their depth values
        # Use the function sample_points_along_rays.

        pts, z_vals = sample_points_along_rays(depth[-1].min(), depth[-1].max(), self.n_coarse, 
                                                ros, rds, device=x_pix.device) # might not be right.

        # Reshape pts to (batch_size, -1, 3).
        pts = pts.reshape(batch_size, -1,3)  # To comply with pixelnerf input

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

        return log, rgb, depth_map # world_coords[-1], depths[-1] (might not need)

    @classmethod
    def from_conf(cls, conf, white_back=False):
        return cls(
            conf.get_int("n_coarse", 32),
            conf.get_int("n_fine", 16),
            n_fine_depth=conf.get_int("n_fine_depth", 8),
            depth_std=conf.get_float("depth_std", 0.01),
            white_back=conf.get_float("white_back", white_back),
        )
    
