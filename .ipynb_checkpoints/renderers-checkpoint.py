from utils import *


def sample_coarse(
    near_depth, # (SB, num_ways)
    far_depth, # (SB, num_ways)
    num_samples: int,
    device: torch.device,
    infinity = -1
):
    # Compute a linspace of num_samples depth values beetween near_depth and far_depth.
    steps = torch.arange(num_samples, dtype=torch.float32, device=device) / num_samples # n_coarse
    z_vals = near_depth.unsqueeze(-1) + torch.einsum('bs,j->bsj', far_depth-near_depth, steps)  # (SB, num_rays, n_coarse)
    z_vals += torch.einsum('bsi,bs->bsi',torch.rand_like(z_vals), far_depth-near_depth) / num_samples 

    if infinity != -1:
        zz_vals = torch.cat([z_vals[..., 1:], 
        torch.broadcast_to(torch.tensor([infinity]).to(z_vals.device), z_vals[...,:1].shape)
        ], -1) 
    else: 
        zz_vals = z_vals # (SB, num_rays, n_coarse)
    steps = None
    z_vals = None
    return zz_vals


def sample_fine(
    near_depth, # (SB, num_ways)
    far_depth, # (SB, num_ways)
    num_samples: int,
    weights, # (SB, num_rays, n_coarse, 1)
    device: torch.device,
):
    SB, num_rays, n_coarse, _ = weights.shape

    weights = weights.squeeze(-1).detach() + 1e-5  # Prevent division by zero, (SB, num_rays, n_coarse)
    pdf = weights / torch.sum(weights, -1, keepdim=True)  # (SB, num_rays, n_coarse)
    cdf = torch.cumsum(pdf, -1)  # (SB, num_rays, n_coarse)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (SB, num_rays, n_coarse+1)

    u = torch.rand(SB, num_rays, num_samples, dtype=torch.float32, device=device)  # (SB, num_rays, n_fine)
    inds = torch.searchsorted(cdf, u, right=True).float() - 1.0  # (SB, num_rays, n_fine)
    inds = torch.clamp_min(inds, 0.0)

    z_steps = (inds + torch.rand_like(inds)) / n_coarse  # (SB, num_rays, n_fine)
    z_samp = near_depth.unsqueeze(-1) + torch.einsum('bs,bsj->bsj', far_depth-near_depth, z_steps)  # (SB, num_rays, n_fine)
    weights = None
    pdf = None
    cdf = None
    u = None
    inds = None
    z_steps = None

    return z_samp

def sample_depth(
    depth, # (SB, num_rays, 1)
    num_samples: int,
    depth_std,
):  
    SB, num_rays, _ = depth.shape
    z_samp = depth.expand((SB, num_rays, num_samples))
    z_vals = torch.randn_like(z_samp) * depth_std
    
    z_samp = None
    return z_vals


def volume_integral(
    z_vals: torch.tensor, # (SB, num_rays, n_coarse)
    sigmas: torch.tensor,  # (SB, num_rays, n_coarse, 1)
    radiances: torch.tensor,  # (SB, num_rays, n_coarse, 3)
    white_back=True,
    infinity=1.8
) -> Tuple[torch.tensor, torch.tensor]:

    # Compute the deltas in depth between the points.
    dists = torch.cat([
        z_vals[..., 1:] - z_vals[..., :-1], 
        torch.broadcast_to(torch.Tensor([1e10]).to(z_vals.device), z_vals[...,:1].shape)
        ], -1) # (SB, num_rays, n_coarse)

    # Compute the alpha values from the densities and the dists.
    # Tip: use torch.einsum for a convenient way of multiplying the correct 
    # dimensions of the sigmas and the dists.
    alpha = 1.- torch.exp(-torch.einsum('brzs, brz -> brzs', sigmas, dists)) # (SB, num_rays, n_coarse, 1)

    # Compute the Ts from the alpha values. Use torch.cumprod.
    
    Ts = torch.cat([
        torch.broadcast_to(torch.Tensor([1]).to(alpha.device), alpha[...,:1,:].shape),
        torch.cumprod(1.-alpha+1e-10, -2)[...,:-1,:],
        ], -2) 

    # Compute the weights from the Ts and the alphas.    
    weights = alpha * Ts # (SB, num_rays, n_coarse, 1)
    
    # Compute the pixel color as the weighted sum of the radiance values.
    rgb = torch.einsum('brzs, brzs -> brs', weights, radiances) # (SB, num_rays, 3)

    # Compute the depths as the weighted sum of z_vals.
    # Tip: use torch.einsum for a convenient way of computing the weighted sum,
    # without the need to reshape the z_vals.
    zz_vals = torch.cat([z_vals[..., 1:], 
        torch.broadcast_to(torch.tensor([infinity]).to(z_vals.device), z_vals[...,:1].shape)
        ], -1) 
    
    depth_map = torch.einsum('brzs, brz -> brs', weights, zz_vals) # (SB, num_rays, 1)

    if white_back:
        accum = weights.sum(dim=-2)
        rgb = rgb + (1. - accum)
        accum = None
        
    dists = None
    alpha = None
    Ts = None

    return rgb, depth_map, weights

class VolumeRenderer(nn.Module):
    def __init__(self, near, far, n_coarse, n_fine, n_fine_depth, depth_std, white_back=True):
        super().__init__()
        self.near = to_gpu(torch.tensor([near]))
        self.far = to_gpu(torch.tensor([far]))

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
        SB, num_rays, _ = x_pix.shape  # _ should be 2, since each pixel coordinate has 2 inputs

        # Compute the ray directions in world coordinates.
        # Use the function get_world_rays.
        ros, rds = get_world_rays(x_pix, intrinsics, cam2world) # (SB, num_ways, 3)

        ### Coarse sampling
        z_vals_coarse = sample_coarse(self.near.expand_as(ros[...,0]), self.far.expand_as(ros[...,0]), 
                                                self.n_coarse, device=x_pix.device) # pts has shape (SB, num_rays, n_coarse)
        pts_coarse = ros.unsqueeze(-2) + torch.einsum('bsi,bsj->bsji', rds, z_vals_coarse)  # (SB, num_rays, n_coarse, 3)    
        
        sigma_rad = radiance_field(pts_coarse.reshape(SB,-1,3), 
            viewdirs=rds.unsqueeze(-2).expand(SB, num_rays, self.n_coarse, -1).reshape(SB,-1,3),
            coarse=True
            ) # (SB, num_rays, 4)
        sigma = sigma_rad[...,3].view(SB, num_rays, self.n_coarse, 1) # (SB, num_rays, n_coarse, 1)
        rad = sigma_rad[...,:3].view(SB, num_rays, self.n_coarse, 3) # (SB, num_rays, n_coarse, 3)

        rgb_coarse, distance_map_corase, weights_coarse = volume_integral(z_vals_coarse, sigma, rad, white_back=self.white_back) 
        # (SB, num_rays, _) for rgb, (SB, num_rays, n_coarse, 1) for weights

        
        
#         ### pixelnerf codes, for comparison
#         # We are starting with x_pix, intrinsics, cam2world
          


#         temp_rays = torch.cat([ros,rds],-1).reshape(-1, 6)   # (SB*NV*num_rays, 6)
#         rays = torch.cat([temp_rays, 0.8*torch.ones_like(temp_rays[...,0:1]), 1.8*torch.ones_like(temp_rays[...,0:1])],-1)  # (SB*NV*num_rays, 8)

#         # Sample coarse
#         near, far = rays[:, -2:-1], rays[:, -1:]  # (B, 1)
#         device=rays.device
#         step = 1.0 / self.n_coarse
#         B = rays.shape[0]
#         z_steps = torch.linspace(0, 1 - step, self.n_coarse, device=device)  # (Kc)
#         z_steps = z_steps.unsqueeze(0).repeat(B, 1)  # (B, Kc)
#         z_steps += torch.rand_like(z_steps) * step
#         z_samp = near * (1 - z_steps) + far * z_steps  # (B, Kc)
        
#         print(f'z val difference: {((z_vals_coarse - z_samp.reshape(SB, NV, num_rays, -1))**2).mean()}')
#         print(z_vals_coarse[0,0,0])
#         print(z_samp[0])
        
#         points = rays[:, None, :3] + z_samp.unsqueeze(2) * rays[:, None, 3:6]
#         points = points.reshape(SB, -1, 3)  # (B*K, 3)
        
#         print(f'points difference: {((pts_coarse - points.reshape(SB, NV, num_rays, -1, 3))**2).mean()}')
#         print(pts_coarse[0,0,0,10])
#         print(points.reshape(SB, NV, num_rays, -1, 3)[0,0,0,10])
        
#         deltas = z_samp[:, 1:] - z_samp[:, :-1]  # (B, K-1)
#         delta_inf = rays[:, -1:] - z_samp[:, -1:]
#         deltas = torch.cat([deltas, delta_inf], -1)  # (B, K)
        
#         out = radiance_field(points, coarse=True, viewdirs=rds.unsqueeze(3).expand(SB, NV, num_rays, self.n_coarse, -1).reshape(SB,-1,3))
        
#         print(f'out rgb difference: {((sigma_rad[...,:3] - out[...,:3])**2).mean()}')
        
#         out = out.reshape(B, self.n_coarse, -1)   # (SB*NV*num_rays, n_coarse, 4)
        
#         rgbs = out[..., :3]  # (B, K, 3)
#         sigmas = out[..., 3]  # (B, K)
        
#         alphas = 1 - torch.exp(-deltas * torch.relu(sigmas))  # (B, K)
#         deltas = None
#         sigmas = None
#         alphas_shifted = torch.cat(
#                 [torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1
#             )  # (B, K+1) = [1, a1, a2, ...]
#         T = torch.cumprod(alphas_shifted, -1)  # (B)
#         weights = alphas * T[:, :-1]  # (B, K)
#         alphas = None
#         alphas_shifted = None

#         rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)  # (B, 3)
#         depth_final = torch.sum(weights * z_samp, -1)  # (B)
#         pix_alpha = weights.sum(dim=1)  # (B), pixel alpha
#         rgb_final = rgb_final + 1 - pix_alpha.unsqueeze(-1)  # (B, 3)
        
#         print(f'rgb difference: {((rgb_coarse - rgb_final.reshape(SB, NV, num_rays, 3))**2).mean()}')
        
#         print(f'distance difference: {((distance_map_corase - depth_final.reshape(SB, NV, num_rays, 1))**2).mean()}')
        
        
        

        
        ### Fine sampling
        z_vals_fine = sample_fine(self.near.expand_as(ros[...,0]), self.far.expand_as(ros[...,0]),
                    self.n_fine-self.n_fine_depth, weights_coarse, device=x_pix.device) # z_vals has shape (SB, num_rays, n)
        z_vals_depth = sample_depth(distance_map_corase, self.n_fine_depth, self.depth_std)
        z_vals_depth = torch.clamp(z_vals_depth,self.near,self.far)

        z_vals = torch.cat([z_vals_coarse,z_vals_fine,z_vals_depth],dim=-1)
        z_vals_sorted, _ = torch.sort(z_vals, dim=-1)

        pts_sorted = ros.unsqueeze(-2) + torch.einsum('bsi,bsj->bsji', rds, z_vals_sorted)  # (SB, num_rays, n_coarse, 3)


        sigma_rad = radiance_field(pts_sorted.reshape(SB,-1,3), 
            viewdirs=rds.unsqueeze(-2).expand(SB, num_rays, self.n_coarse+self.n_fine, -1).reshape(SB,-1,3),
            coarse=False
            ) # (SB, num_rays*n_coarse, 4)
        sigma = sigma_rad[...,3].view(SB, num_rays, self.n_coarse+self.n_fine, 1) # (SB, num_rays, n_coarse, 1)
        rad = sigma_rad[...,:3].view(SB, num_rays, self.n_coarse+self.n_fine, 3) # (SB, num_rays, n_coarse, 3)

        rgb_fine, distance_map_fine, _ = volume_integral(z_vals_sorted, sigma, rad, white_back=self.white_back) 
        # (SB, num_rays, _) for rgb, (SB, num_rays, n_coarse+n_fine, 1) for weights
        
        # Re-calculate depth map since rds now does not have z=1
        world_coordinates_fine = ros + rds * distance_map_fine
        depth_map_fine = depth_from_world(world_coordinates_fine, cam2world)
        
        return rgb_coarse, rgb_fine, depth_map_fine, depth_map_fine

    @classmethod
    def from_conf(cls, conf, white_back=True):
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
        
        print(f'initializing raymarcher with {raymarch_steps} steps')

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
        SB, num_rays, _ = xy_pix.shape

        ros, rds = get_world_rays(xy_pix, intrinsics=intrinsics, cam2world=cam2world) # (SB, num_rays, 3)

        initial_distance = torch.zeros((SB, num_rays, 1)).normal_(mean=0.8, std=5e-2).to(xy_pix.device)
        init_world_coords = ros + rds * initial_distance

        world_coords = [init_world_coords]
        states = [None]


        for step in range(self.steps):

            v = phi(world_coords[-1].reshape(SB,-1,3), viewdirs = rds.reshape(SB,-1,3), return_features = True)   # (SB, num_rays, self.n_feature_channels)

            state = self.lstm(v.reshape(-1,self.n_feature_channels), states[-1])  # (SB*num_rays, self.n_feature_channels)

            if state[0].requires_grad:
                state[0].register_hook(lambda x: x.clamp(min=-10, max=10))

            signed_distance = self.out_layer(state[0]).view(SB, num_rays, 1) # (SB, num_rays, 1)

            new_world_coords = world_coords[-1] + rds * signed_distance # (SB, num_rays, 3)

            states.append(state)
            world_coords.append(new_world_coords)

        self.counter += 1
        output = phi(world_coords[-1].reshape(SB,-1,3), viewdirs = rds.reshape(SB,-1,3), coarse = True, return_features = False)

        rgb = output[..., :3].reshape(SB, num_rays, 3)
        final_depth = depth_from_world(world_coords[-1], cam2world).reshape(SB, num_rays, -1)

        return rgb, None, final_depth, final_depth

    @classmethod
    def from_conf(cls, conf, raymarch_steps):
        return cls(
            num_feature_channels=conf.get_int("num_feature_channels", 512),
            raymarch_steps = raymarch_steps,
        )

class AdaptiveVolumeRenderer(nn.Module):
    def __init__(self, num_feature_channels, raymarch_steps, epsilon, n_coarse, white_back):
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
        phi,
        debug=False
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
        SB, num_rays, _ = xy_pix.shape

        ros, rds = get_world_rays(xy_pix, intrinsics=intrinsics, cam2world=cam2world) # (SB, num_rays, 3)

        initial_distance = torch.zeros((SB, num_rays, 1)).normal_(mean=0.8, std=5e-2).to(xy_pix.device)

        init_world_coords = ros + rds * initial_distance

        world_coords = [init_world_coords]
        states = [None]


        for _ in range(self.steps):

            v = phi(world_coords[-1].reshape(SB,-1,3), viewdirs = rds.reshape(SB,-1,3), return_features = True)   # (1, num_rays, self.n_feature_channels)

            state = self.lstm(v.reshape(-1,self.n_feature_channels), states[-1])  # (SB*num_rays, self.n_feature_channels)

            if state[0].requires_grad:
                state[0].register_hook(lambda x: x.clamp(min=-10, max=10))

            signed_distance = self.out_layer(state[0]).view(SB, num_rays, 1) # (SB, num_rays, 1)

            new_world_coords = world_coords[-1] + rds * signed_distance

            states.append(state)
            world_coords.append(new_world_coords)
            
        # debug mode
#         if debug:
#             z_vals_debug = sample_coarse(to_gpu(torch.tensor([0.8])).expand_as(ros[...,0]), to_gpu(torch.tensor([1.8])).expand_as(ros[...,0]), 
#                                                 64, device=xy_pix.device) # pts has shape (SB, num_rays, n_coarse)
#             pts_debug = ros.unsqueeze(-2) + torch.einsum('bsi,bsj->bsji', rds, z_vals_debug)  # (SB, num_rays, n_coarse, 3)    
        
#             sigma_rad = phi(pts_debug.reshape(SB,-1,3), 
#                 viewdirs=rds.unsqueeze(-2).expand(SB, num_rays, 64, -1).reshape(SB,-1,3),
#                 coarse=True
#                 ) # (SB, num_rays, 4)
#             sigma = sigma_rad[...,3].view(SB, num_rays, 64, 1) # (SB, num_rays, n_coarse, 1)
#             rad = sigma_rad[...,:3].view(SB, num_rays, 64, 3) # (SB, num_rays, n_coarse, 3)
#             rgb_fine, distance_map_fine, _ = volume_integral(z_vals_debug, sigma, rad, white_back=True)
#             world_coordinates = ros + rds * distance_map_fine
#             depth_map_debug = depth_from_world(world_coordinates, cam2world)
#             random_idx = 8320
#             print(f' pixel location is {xy_pix[0][random_idx]}')
#             print(f' densities are {sigma[0][random_idx].squeeze()}')
#             print(f' distance is {distance_map_fine[0][random_idx]}')
#             print(f' depth is {depth_map_debug[0][random_idx]}')
#             print(sigma.min(),sigma.max())
#             x_debug = xy_pix[0,:,0:1].expand(num_rays, 64)      # (num_rays, n_coarse)
#             y_debug = xy_pix[0,:,1:2].expand(num_rays, 64) 
#             z_debug = depth_from_world(pts_debug, cam2world.unsqueeze(2).expand(1,num_rays,64,-1,-1))[0,...].squeeze()  # (num_rays, n_coarse)
            
#             import plotly.graph_objects as go
#             import plotly.offline as pyo
#             pyo.init_notebook_mode()

#             fig = go.Figure(data=go.Volume(
#                 x=x_debug.flatten().cpu().numpy(),
#                 y=y_debug.flatten().cpu().numpy(),
#                 z=z_debug.flatten().cpu().numpy(),
#                 value=sigma.flatten().cpu().numpy(),
#                 opacity=0.1,
#                 isomin=0,
#                 isomax=100,
#                 surface_count=20,
#                 caps=dict(x_show=False, y_show=False, z_show=False)
#                 ))
#             #
#             fig.show()
            
            
            
        # Coarse image
        output_coarse = phi(world_coords[-1].reshape(SB,-1,3), viewdirs = rds.reshape(SB,-1,3), coarse = True, return_features = False)

        rgb_coarse = output_coarse[..., :3].reshape(SB, num_rays, 3)
        depth_coarse = depth_from_world(world_coords[-1], cam2world).reshape(SB, num_rays, -1)
            

        # Generate the points along rays and their depth values
        final_distance = ((world_coords[-1][...,0] - ros[...,0]) / rds[...,0]) # (SB, num_rays)

        z_vals = sample_coarse(final_distance - self.epsilon, final_distance + self.epsilon, 
                                    self.n_coarse, device=xy_pix.device) # z_vals has shape (SB, num_rays, n_coarse)
        z_vals_sorted, _ = torch.sort(z_vals, dim=-1)

        pts_sorted = ros.unsqueeze(-2) + torch.einsum('bsi,bsj->bsji', rds, z_vals_sorted)  # (SB, num_rays, n, 3)

        # Sample the radiance field with the points along the rays.
        sigma_rad = phi(pts_sorted.reshape(SB,-1,3), coarse=False,
                    viewdirs=rds.unsqueeze(-2).expand(SB, num_rays, self.n_coarse, -1).reshape(SB,-1,3), return_features = False)
        sigma = sigma_rad[...,3].reshape(SB, num_rays, self.n_coarse, 1)
        rad = sigma_rad[...,:3].reshape(SB, num_rays, self.n_coarse, 3)

        # Compute pixel colors, depths, and weights via the volume integral.
        rgb, distance_map, _ = volume_integral(z_vals_sorted, sigma, rad, white_back=self.white_back)

        # Re-calculate depth map since rds now does not have z=1
        world_coordinates = ros + rds * distance_map
        depth_map = depth_from_world(world_coordinates, cam2world)
        
        if debug:
            random_idx = 64 # 8256: center, 8320: edge_center, 0: corner
            
            print('now AVR')
            
            print(f' pixel location is {xy_pix[0][random_idx]}')
            print(f' distances are {z_vals_sorted[0][random_idx].squeeze()}')
            print(f' colors are {rad[0][random_idx][:,0].squeeze()}')
            print(f' densities are {sigma[0][random_idx].squeeze()}')
            print(f' distance is {distance_map[0][random_idx]}')
            print(f' color is {rgb[0][random_idx][0]}')
            print(f' depth is {depth_map[0][random_idx]}')
            
            print(f'fine grid for debugging')
            z_vals_debug = sample_coarse(to_gpu(torch.tensor([0.8])).expand_as(ros[...,0]), to_gpu(torch.tensor([1.8])).expand_as(ros[...,0]), 
                                                64, device=xy_pix.device) # pts has shape (SB, num_rays, n_coarse)
            pts_debug = ros.unsqueeze(-2) + torch.einsum('bsi,bsj->bsji', rds, z_vals_debug)  # (SB, num_rays, n_coarse, 3)    
        
            sigma_rad = phi(pts_debug.reshape(SB,-1,3), 
                viewdirs=rds.unsqueeze(-2).expand(SB, num_rays, 64, -1).reshape(SB,-1,3),
                coarse=False
                ) # (SB, num_rays, 4)
            sigma = sigma_rad[...,3].view(SB, num_rays, 64, 1) # (SB, num_rays, n_coarse, 1)
            rad = sigma_rad[...,:3].view(SB, num_rays, 64, 3) # (SB, num_rays, n_coarse, 3)
            rgb_fine, distance_map_fine, _ = volume_integral(z_vals_debug, sigma, rad, white_back=True)
            world_coordinates = ros + rds * distance_map_fine
            depth_map_debug = depth_from_world(world_coordinates, cam2world)

            print(f' pixel location is {xy_pix[0][random_idx]}')
            print(f' distances are {z_vals_debug[0][random_idx].squeeze()}')
            print(f' colors are {rad[0][random_idx][:,0].squeeze()}')
            print(f' densities are {sigma[0][random_idx].squeeze()}')
            print(f' distance is {distance_map_fine[0][random_idx]}')
            print(f' color is {rgb_fine[0][random_idx][0]}')
            print(f' depth is {depth_map_debug[0][random_idx]}')

        return rgb_coarse, rgb, depth_coarse, depth_map

    @classmethod
    def from_conf(cls, conf, white_back=False):
        return cls(
            num_feature_channels=conf.get_int("num_feature_channels", 512),
            raymarch_steps = conf.get_int("raymarch_steps", 10),
            epsilon = conf.get_float("epsilon", 0.05),
            n_coarse=conf.get_int("n_coarse", 20),
            white_back=conf.get_float("white_back", white_back)
        )