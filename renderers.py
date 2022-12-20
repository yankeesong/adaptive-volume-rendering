from utils import *


def sample_coarse(
    near_depth, # (SB, NV, num_ways)
    far_depth, # (SB, NV, num_ways)
    num_samples: int,
    device: torch.device,
    infinity = -1
):
    # Compute a linspace of num_samples depth values beetween near_depth and far_depth.
    steps = torch.arange(num_samples, dtype=torch.float32, device=device) / num_samples # n_coarse
    z_vals = near_depth.unsqueeze(-1) + torch.einsum('bns,j->bnsj', far_depth-near_depth, steps)  # (SB, NV, num_rays, n_coarse)

    if infinity != -1:
        zz_vals = torch.cat([z_vals[..., 1:], 
        torch.broadcast_to(torch.tensor([infinity]).to(z_vals.device), z_vals[...,:1].shape)
        ], -1) 
    else: 
        zz_vals = z_vals # (SB, NV, num_rays, n_coarse)
    return zz_vals

def sample_fine(
    near_depth, # (SB, NV, num_ways)
    far_depth, # (SB, NV, num_ways)
    num_samples: int,
    weights, # # (SB, NV, num_rays, n_coarse, 1)
    device: torch.device,
):


def volume_integral(
    z_vals: torch.tensor, # (SB, NV, num_rays, n_coarse)
    sigmas: torch.tensor,  # (SB, NV, num_rays, n_coarse, 1)
    radiances: torch.tensor,  # (SB, NV, num_rays, n_coarse, 3)
    white_back=True
) -> Tuple[torch.tensor, torch.tensor]:

    # Compute the deltas in depth between the points.
    dists = torch.cat([
        z_vals[..., 1:] - z_vals[..., :-1], 
        torch.broadcast_to(torch.Tensor([1e10]).to(z_vals.device), z_vals[...,:1].shape)
        ], -1) # (SB, NV, num_rays, n_coarse)

    # Compute the alpha values from the densities and the dists.
    # Tip: use torch.einsum for a convenient way of multiplying the correct 
    # dimensions of the sigmas and the dists.
    alpha = 1.- torch.exp(-torch.einsum('bnrzs, bnrz -> bnrzs', sigmas, dists)) # (SB, NV, num_rays, n_coarse, 1)

    # Compute the Ts from the alpha values. Use torch.cumprod.
    
    Ts = torch.cat([
        torch.broadcast_to(torch.Tensor([1]).to(alpha.device), alpha[...,:1,:].shape),
        torch.cumprod(1.-alpha+1e-10, -2)[...,:-1,:],
        ], -2) 

    # Compute the weights from the Ts and the alphas.    
    weights = alpha * Ts # (SB, NV, num_rays, n_coarse, 1)
    
    # Compute the pixel color as the weighted sum of the radiance values.
    rgb = torch.einsum('bnrzs, bnrzs -> bnrs', weights, radiances) # (SB, NV, num_rays, 3)

    # Compute the depths as the weighted sum of z_vals.
    # Tip: use torch.einsum for a convenient way of computing the weighted sum,
    # without the need to reshape the z_vals.
    depth_map = torch.einsum('bnrzs, bnrz -> bnrs', weights, z_vals) # (SB, NV, num_rays, 1)

    if white_back:
        accum = weights.sum(dim=-2)
        rgb = rgb + (1. - accum)

    return rgb, depth_map, weights

class VolumeRenderer(nn.Module):
    def __init__(self, near, far, n_coarse, n_fine, n_fine_depth, depth_std, white_back=True):
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
                here x_pix has size (SB, NV, num_rays, 2)
            radiance_field: nn.Module instance of the radiance field we want to render. This should by default the PixelNeRF. 
                Params: 
                    xyz: points in space, with shape (SB, NV, num_rays, 3)
                        SB is batch of objects
                        NV is num of views
                        num_rays is batch of points (in rays)
                Return:
                    (SB, NV, num_rays, 4) of r g b sigma

        Returns:
            Tuple of rgb, depth_map
            rgb: for each pixel coordinate x_pix, the color of the respective ray.
            depth_map: for each pixel coordinate x_pix, the depth of the respective ray.
 
        """
        SB, NV, num_rays, _ = x_pix.shape  # _ should be 2, since each pixel coordinate has 2 inputs

        # Compute the ray directions in world coordinates.
        # Use the function get_world_rays.
        ros, rds = get_world_rays(x_pix, intrinsics, cam2world) # (SB, NV, num_ways, 3)

        ### Coarse sampling
        z_vals_coarse = sample_coarse(torch.tensor(self.near).expand_as(ros[...,0]), torch.tensor(self.far).expand_as(ros[...,0]), 
                                                self.n_coarse, device=x_pix.device) # pts has shape (SB, NV, num_rays, n_coarse, 3)
        pts_coarse = ros.unsqueeze(-2) + torch.einsum('bnsi,bnsj->bnsji', rds, z_vals_coarse)  # (SB, NV, num_rays, n_coarse, 3)
        
        sigma_rad = radiance_field(pts_coarse.reshape(SB,-1,3), 
            viewdirs=rds.unsqueeze(3).expand(SB, NV, num_rays, self.n_coarse, -1).reshape(SB,-1,3),
            coarse=True
            ) # (SB, NV*num_rays, 4)
        sigma = sigma_rad[...,3].view(SB, NV, num_rays, self.n_coarse, 1) # (SB, NV, num_rays, n_coarse, 1)
        rad = sigma_rad[...,:3].view(SB, NV, num_rays, self.n_coarse, 3) # (SB, NV, num_rays, n_coarse, 3)

        rgb_coarse, distance_map_corase, weights_coarse = volume_integral(z_vals_coarse, sigma, rad, white_back=self.white_back) 
        # (SB, NV, num_rays, _) for rgb, (SB, NV, num_rays, n_coarse, 1) for weights

        ### Fine sampling
        z_vals_fine = sample_fine(torch.tensor(self.near).expand_as(ros[...,0]), torch.tensor(self.far).expand_as(ros[...,0]),
                    self.n_fine-self.n_fine_depth, weights_coarse, device=x_pix.device) # z_vals has shape (SB, NV, num_rays, n)
        z_vals_depth = sample_depth(distance_map_corase, self.n_fine_depth, self.depth_std, device=x_pix.device)

        z_vals = torch.cat([z_vals_coarse,z_vals_fine,z_vals_depth],dim=-1)
        z_vals_sorted, _ = torch.sort(z_vals, dim=-1)

        pts_sorted = ros.unsqueeze(-2) + torch.einsum('bnsi,bnsj->bnsji', rds, z_vals_sorted)  # (SB, NV, num_rays, n, 3)


        sigma_rad = radiance_field(pts_sorted.reshape(SB,-1,3), 
            viewdirs=rds.unsqueeze(3).expand(SB, NV, num_rays, self.n_coarse+self.n_fine, -1).reshape(SB,-1,3),
            coarse=False
            ) # (SB, NV*num_rays, 4)
        sigma = sigma_rad[...,3].view(SB, NV, num_rays, self.n_coarse+self.n_fine, 1) # (SB, NV, num_rays, n_coarse, 1)
        rad = sigma_rad[...,:3].view(SB, NV, num_rays, self.n_coarse+self.n_fine, 3) # (SB, NV, num_rays, n_coarse, 3)

        rgb_fine, distance_map_fine, _ = volume_integral(z_vals_sorted, sigma, rad, white_back=self.white_back) 
        # (SB, NV, num_rays, _) for rgb, (SB, NV, num_rays, n_coarse+n_fine, 1) for weights
        
        # Re-calculate depth map since rds now does not have z=1
        world_coordinates_fine = ros + rds * distance_map_fine
        depth_map_fine = depth_from_world(world_coordinates_fine, cam2world)

        return rgb_coarse, rgb_fine, depth_map_fine

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
                 raymarch_steps, near = 0.5, far = 2.0):
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

        self.near = near
        self.far = far

    def forward(self,
                cam2world,
                intrinsics,
                xy_pix, # xy_pix has shape 
                phi):  # Here phi is pixelnerf net
        NV, num_rays, _ = xy_pix.shape

        ros, rds = get_world_rays(xy_pix, intrinsics=intrinsics, cam2world=cam2world) # (NV, num_rays, 3)

        initial_depth = torch.zeros((NV, num_rays, 1)).normal_(mean=0.8, std=5e-2).to(xy_pix.device)

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
        final_depth = depths[-1].reshape(NV, num_rays, -1)

        # return world_coords[-1], depths[-1], log
        return rgb, final_depth # See here the output is a bit different

    @classmethod
    def from_conf(cls, conf):
        return cls(
            num_feature_channels=conf.get_int("num_feature_channels", 512),
            raymarch_steps = conf.get_int("raymarch_steps", 30),
        )

class AdaptiveVolumeRenderer(nn.Module):
    def __init__(self, num_feature_channels, raymarch_steps, epsilon, n_coarse=16, white_back=True):
        super().__init__()
        self.epsilon = epsilon
        self.n_coarse = n_coarse
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
        xy_pix,
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
        NV, num_rays, _ = xy_pix.shape

        ros, rds = get_world_rays(xy_pix, intrinsics=intrinsics, cam2world=cam2world) # (NV, num_rays, 3)

        initial_depth = torch.zeros((NV, num_rays, 1)).normal_(mean=0.8, std=5e-2).to(xy_pix.device)

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

            # print("Raymarch step %d: Min depth %0.6f, max depth %0.6f" %
            #           (step, depths[-1].min().detach().cpu().numpy(), depths[-1].max().detach().cpu().numpy()))

            depths.append(depth)
        
#         print(f'depth out of raymarcher: {depths[-1].min().detach().cpu().numpy()},{depths[-1].max().detach().cpu().numpy()}')
            
        # Compute the ray directions in world coordinates.
        # Use the function get_world_rays.
        ros, rds = get_world_rays(xy_pix, intrinsics, cam2world)
#         print(world_coords[-1][0][528])
        
        z_scales = rds[...,2].unsqueeze(2) # (NV, num_ways, 1)

        # Generate the points along rays and their depth values
        # print("start")
        # a = time.time()
        distance = ((world_coords[-1][...,0] - ros[...,0]) / rds[...,0]).unsqueeze(2)

        pts, z_vals = sample_coarse(distance - self.epsilon,
                                                     distance + self.epsilon, 
                                                     self.n_coarse, ros, rds, device=xy_pix.device) # pts has shape (NV, num_rays, 3)
        # print("end", time.time() - a)


        rds = rds.unsqueeze(2).expand(NV, num_rays, self.n_coarse, -1)

        # 

        # Input view directions
        # (NV*num_ways, 3)

        # Sample the radiance field with the points along the rays.
        sigma_rad = phi(pts.reshape(1,-1,3), viewdirs=rds.reshape(1,-1,3), return_features = False)
        sigma = sigma_rad[...,3].reshape(NV, num_rays, self.n_coarse, 1)
        rad = sigma_rad[...,:3].reshape(NV, num_rays, self.n_coarse, 3)

        # Compute pixel colors, depths, and weights via the volume integral.
        rgb, distance_map, weights = batch_volume_integral(z_vals, sigma, rad)

        if self.white_back:
            accum = weights.sum(dim=-2)
            rgb = rgb + (1. - accum)
        
        # Re-calculate depth map since rds now does not have z=1
        world_coordinates = ros + rds.reshape(NV, num_rays, self.n_coarse, -1)[...,0,:].squeeze() * distance_map
        depth_map = depth_from_world(world_coordinates, cam2world)

        return rgb, depth_map

    @classmethod
    def from_conf(cls, conf, white_back=False):
        return cls(
            num_feature_channels=conf.get_int("num_feature_channels", 512),
            raymarch_steps = conf.get_int("raymarch_steps", 30),
            epsilon = conf.get_float("epsilon", 0.5),
            n_coarse=conf.get_int("n_coarse", 16),
            # near=conf.get_float("near", 1.0),
            # far=conf.get_float("far", 2.5),
            white_back=conf.get_float("white_back", white_back),
        )