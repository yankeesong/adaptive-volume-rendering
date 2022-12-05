from utils import *
# import sys
# #root_dir = "/Users/jameszli/desktop/MIT/6.S980/"  
# #sys.path.insert(0, f"{root_dir}/scene-representation-networks/")
# from geometry import *
# from util import *
# #from pytorch_prototyping import pytorch_prototyping

import time
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

def linspace(start: torch.Tensor, stop: torch.Tensor, num: int):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)
    
    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)
    
    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps*(stop - start)[None]
    
    return out

def batch_sample_points_along_rays(
    near_depth: torch.Tensor,
    far_depth: torch.Tensor,
    num_samples: int,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    device: torch.device
):
    # Compute a linspace of num_samples depth values beetween near_depth and far_depth.
    z_vals = linspace(near_depth, far_depth - (far_depth-near_depth)/num_samples, num_samples)

    z_vals = z_vals.permute(1, 2, 0, 3).squeeze()

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

def batch_volume_integral(
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
    alpha = 1.- torch.exp(-torch.einsum('brzs, brz -> brzs', sigmas, dists))

    # Compute the Ts from the alpha values. Use torch.cumprod.
    Ts = torch.cumprod(1.-alpha+1e-10, -2)

    # Compute the weights from the Ts and the alphas.
    weights = alpha * Ts
    
    # Compute the pixel color as the weighted sum of the radiance values.
    rgb = torch.einsum('brzs, brzs -> brs', weights, radiances)

    # Compute the depths as the weighted sum of z_vals.
    # Tip: use torch.einsum for a convenient way of computing the weighted sum,
    # without the need to reshape the z_vals.
    depth_map = torch.einsum('brzs, brz -> brs', weights, z_vals)

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
        ros, rds = get_world_rays(x_pix, intrinsics, cam2world) # (NV, num_ways, 3)

        z_scales = rds[...,2].unsqueeze(2) # (NV, num_ways, 1)


        # Generate the points along rays and their depth values
        # Use the function sample_points_along_rays.
        pts, z_vals = sample_points_along_rays(self.near, self.far, self.n_coarse, 
                                                ros, rds, device=x_pix.device) # pts has shape (NV, num_rays, 3)
        rds = rds.unsqueeze(2).expand(NV, num_rays, self.n_coarse, -1)

        # 
        pts = pts.reshape(1,-1,3)  # (1, NV*num_rays*self.n_coarse, 3)
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
        # print(z_vals.shape)
        # print(f'sigma has shape {sigma.shape}')
        # print(f'rad has shape {rad.shape}')

        # Compute pixel colors, depths, and weights via the volume integral.
        rgb, depth_map, weights = volume_integral(z_vals, sigma, rad) # (NV, num_rays, _)

        if self.white_back:
            accum = weights.sum(dim=-2)
            rgb = rgb + (1. - accum)
        
        # Re-calculate depth map since rds now does not have z=1
        depth_map = depth_map * torch.abs(z_scales)

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


class Raymarcher(nn.Module):
    def __init__(self,
                 num_feature_channels,
                 raymarch_steps):
        super().__init__()

        self.n_feature_channels = num_feature_channels
        self.steps = raymarch_steps

        hidden_size = 16
        self.lstm = nn.LSTMCell(input_size=self.n_feature_channels,
                                hidden_size=hidden_size)

        self.lstm.apply(init_recurrent_weights)
        lstm_forget_gate_init(self.lstm)

        self.out_layer = nn.Linear(hidden_size, 1)
        self.counter = 0

    def forward(self,
                cam2world,
                intrinsics,
                xy_pix, # xy_pix has shape 
                phi):  # Here phi is pixelnerf net
        NV, num_rays, _ = xy_pix.shape

        #ray_dirs = get_ray_directions(uv, cam2world=cam2world, intrinsics=intrinsics)
        ros, rds = get_world_rays(xy_pix, intrinsics=intrinsics, cam2world=cam2world) # (NV, num_rays, 3)

        initial_depth = torch.zeros((NV, num_rays, 1)).normal_(mean=0.8, std=5e-4).to(xy_pix.device)

        init_world_coords = ros + rds * initial_depth

        world_coords = [init_world_coords]
        depths = [initial_depth]
        states = [None]


        for step in range(self.steps):

            v = phi(world_coords[-1].reshape(1,-1,3), viewdirs = rds.reshape(1,-1,3), return_features = True)   # (1, NV*num_rays, self.n_feature_channels)

            state = self.lstm(v.squeeze(0), states[-1])  # (NV*num_rays, self.n_feature_channels)

            if state[0].requires_grad:
                state[0].register_hook(lambda x: x.clamp(min=-10, max=10))

            signed_distance = self.out_layer(state[0]).view(NV, num_rays, 1) # (NV, num_rays, 1)

            new_world_coords = world_coords[-1] + rds * signed_distance

            states.append(state)
            world_coords.append(new_world_coords)

            depth = depth_from_world(world_coords[-1], cam2world)
            # commented out for now

            print("Raymarch step %d: Min depth %0.6f, max depth %0.6f" %
                      (step, depths[-1].min().detach().cpu().numpy(), depths[-1].max().detach().cpu().numpy()))

            depths.append(depth)

        # commented out for now
        # if not self.counter % 100:
        #     # Write tensorboard summary for each step of ray-marcher.
        #     drawing_depths = torch.stack(depths, dim=0)[:, 0, :, :]
        #     drawing_depths = lin2img(drawing_depths).repeat(1, 3, 1, 1)
        #     log.append(('image', 'raycast_progress',
        #                 torch.clamp(torchvision.utils.make_grid(drawing_depths, scale_each=False, normalize=True), 0.0,
        #                             5),
        #                 100))

        #     # Visualize residual step distance (i.e., the size of the final step)
        #     fig = show_images([lin2img(signed_distance)[i, :, :, :].detach().cpu().numpy().squeeze()
        #                             for i in range(batch_size)])
        #     log.append(('figure', 'stopping_distances', fig, 100))
        self.counter += 1

        output = phi(world_coords[-1].reshape(1,-1,3), viewdirs = rds.reshape(1,-1,3), return_features = False)


        rgb = output[..., :3].reshape(NV, num_rays,3)
        sigma = output[..., 3:4].reshape(NV, num_rays,1) 
        # return world_coords[-1], depths[-1], log
        return rgb, sigma # sigma isn't depth, but should affect loss calculation

    @classmethod
    def from_conf(cls, conf):
        return cls(
            num_feature_channels=conf.get_int("num_feature_channels", 512),
            raymarch_steps = conf.get_int("raymarch_steps", 10),
        )

class AdaptiveVolumeRenderer(nn.Module):
    def __init__(self, num_feature_channels, raymarch_steps, epsilon, n_coarse=32, white_back=True):
        super().__init__()
        self.epsilon = epsilon
        self.n_coarse = n_coarse
        # self.n_fine = n_fine
        # self.n_fine_depth = n_fine_depth
        # self.depth_std = depth_std
        self.white_back = white_back

        self.n_feature_channels = num_feature_channels
        self.steps = raymarch_steps

        hidden_size = 16
        self.lstm = nn.LSTMCell(input_size=self.n_feature_channels,
                                hidden_size=hidden_size)

        self.lstm.apply(init_recurrent_weights)
        lstm_forget_gate_init(self.lstm)

        self.out_layer = nn.Linear(hidden_size, 1)
        self.counter = 0

    def forward(
        self,
        cam2world,
        intrinsics,
        uv,
        phi
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

        batch_size, num_samples, _ = uv.shape

        ray_dirs = get_ray_directions(uv, cam2world=cam2world, intrinsics=intrinsics)

        initial_depth = torch.zeros((batch_size, num_samples, 1)).normal_(mean=0.05, std=5e-4).to(uv.device)
        init_world_coords = world_from_xy_depth(uv, initial_depth, intrinsics=intrinsics, cam2world=cam2world)

        world_coords = [init_world_coords]
        depths = [initial_depth]
        states = [None]


        for step in range(self.steps):
            world_coords[-1] = world_coords[-1].reshape(1,-1,3)  # (1, batch_size*num_samples, 3)
            ray_dirs = ray_dirs.reshape(1,-1,3)

            v = phi(world_coords[-1], viewdirs = ray_dirs, return_features = True)

            state = self.lstm(v.reshape(-1, self.n_feature_channels), states[-1])

            if state[0].requires_grad:
                state[0].register_hook(lambda x: x.clamp(min=-10, max=10))

            signed_distance = self.out_layer(state[0]).view(batch_size, num_samples, 1)
            ray_dirs = ray_dirs.reshape(batch_size, num_samples,3)
            world_coords[-1] = world_coords[-1].reshape(batch_size, num_samples,3)
            new_world_coords = world_coords[-1] + ray_dirs * signed_distance

            states.append(state)
            world_coords.append(new_world_coords)

            depth = depth_from_world(world_coords[-1], cam2world)
            # commented out for now
            # if self.training:
            #     print("Raymarch step %d: Min depth %0.6f, max depth %0.6f" %
            #           (step, depths[-1].min().detach().cpu().numpy(), depths[-1].max().detach().cpu().numpy()))

            depths.append(depth)
            
        # Compute the ray directions in world coordinates.
        # Use the function get_world_rays.
        ros, rds = get_world_rays(uv, intrinsics, cam2world)

        # Generate the points along rays and their depth values
        # Use the function sample_points_along_rays.
        # print("start")
        # a = time.time()
        pts, z_vals = batch_sample_points_along_rays(depths[-1] - self.epsilon, depth[-1] + self.epsilon, self.n_coarse, 
                                                ros, rds, device=uv.device) # pts has shape (NV, num_rays, 3)
        # print("end", time.time() - a)


        rds = rds.unsqueeze(2).expand(batch_size, num_samples, self.n_coarse, -1)

        # 
        pts = pts.reshape(1,-1,3)  # (1, NV*num_ways, 3)
        rds = rds.reshape(1,-1,3)

        # Input view directions
        # (NV*num_ways, 3)

        # Sample the radiance field with the points along the rays.
        sigma_rad = phi(pts, viewdirs=rds)
        sigma_rad = sigma_rad.squeeze(0)   # Get rid of SB dimension
        sigma = sigma_rad[...,3]
        rad = sigma_rad[...,:3]


        # Reshape sigma and rad back to (NV, num_rays, self.n_samples, -1)
        sigma = sigma.view(batch_size, num_samples, self.n_coarse, 1)
        rad = rad.view(batch_size, num_samples, self.n_coarse, 3)

        # print(f'z_vals has shape {z_vals.shape}')
        # print(f'sigma has shape {sigma.shape}')
        # Compute pixel colors, depths, and weights via the volume integral.
        rgb, depth_map, weights = batch_volume_integral(z_vals, sigma, rad)

        if self.white_back:
            accum = weights.sum(dim=-2)
            rgb = rgb + (1. - accum)

        return rgb, depth_map

    @classmethod
    def from_conf(cls, conf, white_back=False):
        return cls(
            num_feature_channels=conf.get_int("num_feature_channels", 512),
            raymarch_steps = conf.get_int("raymarch_steps", 10),
            epsilon = conf.get_float("epsilon", 0.5),
            n_coarse=conf.get_int("n_coarse", 32),
            # near=conf.get_float("near", 1.0),
            # far=conf.get_float("far", 2.5),
            # n_coarse=conf.get_int("n_coarse", 32),
            # n_fine=conf.get_int("n_fine", 16),
            # n_fine_depth=conf.get_int("n_fine_depth", 8),
            # depth_std=conf.get_float("depth_std", 0.01),
            white_back=conf.get_float("white_back", white_back),
        )