from utils import *
from dataset import *

def fit(
    net: nn.Module,
    model: nn.Module,
    data_loader,
    loss_fn,
    resolution: Tuple,
    optimizer,
    plotting_function = None,
    steps_til_summary = 500,
    total_steps=2001
   ):

    losses = []
    data_iterator = iter(data_loader)
    for step in range(total_steps):
        # Get the next batch of data and move it to the GPU
        try:
            model_input = next(data_iterator)
        except StopIteration:
            data_iterator = iter(data_loader)
            model_input = next(data_iterator)

        model_input = to_gpu(model_input)
        ground_truth = model_input["images"] # (NV, H*W, 3), values are between 0 and 1

        NV, sl2, _ = ground_truth.shape
        NS = 1 # Number of source views, san only handle one now
        sl = int(np.sqrt(sl2))

        # Extract source images (randomly determine)
        src_idx = to_gpu(np.random.choice(NV, NS, replace=False))
        


        # Encode
        # :param images (NS, 3, H, W)
        # NS is number of source views, now can only deal with 1
        # :param poses (NS, 4, 4)
        # :param focal focal length () or (2) or (NS) or (NS, 2) [fx, fy]
        # :param z_bounds ignored argument (used in the past)
        # :param c principal point None or () or (2) or (NS) or (NS, 2) [cx, cy],
        # default is center of image
        src_images = ground_truth[src_idx,...].reshape(-1,sl,sl,3).permute(0,3,1,2)
        #print(f'source image has size {src_images.shape}')
        poses = model_input['cam2world'][src_idx,...]
        #print(f'poses has size {poses.shape}')
        focal = model_input['focal'][src_idx,...]
        c = model_input['c'][src_idx,...]
        
        net.encode(src_images, poses, focal, c)


        

        # Compute the MLP output for the given input data and compute the loss
        model_output = model(model_input)

        # Implement a simple mean-squared-error loss between the 
        loss = loss_fn(model_output, ground_truth) # Note: loss now takes "model" as input.

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate the losses so that we can plot them later
        # losses.append(loss.detach().cpu().numpy())            

        # Every so often, we want to show what our model has learned.
        # It would be boring otherwise!
        if not step % steps_til_summary:
            print(f"Step {step}: loss = {float(loss.detach().cpu()):.5f}")
            print(f"car number {model_input['idx'][0]}")
#             print("Min depth %0.6f, max depth %0.6f" %
#                           (model_output[1].min().detach().cpu().numpy(), model_output[1].max().detach().cpu().numpy()))
#             print("Min color %0.6f, max color %0.6f" %
#                           (model_output[0].min().detach().cpu().numpy(), model_output[0].max().detach().cpu().numpy()))
            if src_idx == 0:
                print(f'same view dir as src')

            if plotting_function is not None: # Note: we now call the "plotting function" instead of hard-coding the plotting here.
                plotting_function(model_output, ground_truth, resolution)

    return model_output

def mse_loss(mlp_out, gt):
    img, depth = mlp_out
    return ((img - gt)**2).mean()

def mse_regularization_loss(mlp_out, gt, near=0.5, far=2.0):
    # add regularization loss
    img, depth = mlp_out
    penalty = (torch.min(depth-near, torch.zeros_like(depth)) ** 2) + (torch.max(depth-far, torch.zeros_like(depth)) ** 2)
    return ((img - gt)**2).mean() + torch.mean(penalty) * 10000

def plot_output_ground_truth(model_output, ground_truth, resolution):
    img, depth = model_output

    img = img[0]
    depth = depth[0]
    gt = ground_truth[0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), squeeze=False)
    axes[0, 0].imshow(img.cpu().view(*resolution).detach().numpy())
    axes[0, 0].set_title("Trained MLP")
    axes[0, 1].imshow(gt.cpu().view(*resolution).detach().numpy())
    axes[0, 1].set_title("Ground Truth")
    
    depth = depth.cpu().view(*resolution[:2]).detach().numpy()
    axes[0, 2].imshow(depth, cmap='Greys')
    axes[0, 2].set_title("Depth")
    
    for i in range(3):
        axes[0, i].set_axis_off()

    plt.show()
    
def get_psnr(net, rf_and_renderer, val_dir, n_val, sl, specific_observation_idcs):
    obj_dirs = sorted(glob(os.path.join(val_dir, "*/")))

    assert (len(obj_dirs) != 0), "No objects in the data directory"

    obj_dirs = obj_dirs[:n_val]
    
    psnrs = []
    ssims = []
    
    for idx, dir in enumerate(obj_dirs):
        print(f'object number {idx}')
        # Encode
        source_data = SceneInstanceDataset(instance_idx=idx,
                                               instance_dir=dir,
                                               specific_observation_idcs=[64],
                                               img_sidelength=sl,
                                               num_images=-1)
        model_input = to_gpu(source_data[0])
        src_images = model_input['images'].reshape(-1,sl,sl,3).permute(0,3,1,2)
        poses = model_input['cam2world'].unsqueeze(0)
        focal = model_input['focal'].unsqueeze(0)
        c = model_input['c'].unsqueeze(0)
        intrinsics = model_input['intrinsics'].unsqueeze(0)
        net.encode(src_images, poses, focal, c)
        
        # Calculation
        obj_data = SceneInstanceDataset(instance_idx=idx,
                                               instance_dir=dir,
                                               specific_observation_idcs=None,
                                               img_sidelength=sl,
                                               num_images=-1)
        counter = 0
        total_psnr = 0
        total_ssim = 0
        with torch.no_grad():
            for mi in obj_data:
                counter += 1
                mi = to_gpu(mi)
                for key in mi:
                    mi[key] = mi[key].unsqueeze(0)
                mo = rf_and_renderer(mi)
                psnr, ssim = calculate_psnr(mo[0][0],mi['images'][0])
                total_psnr += psnr
                total_ssim += ssim
            psnrs.append(total_psnr/counter)
            ssims.append(total_ssim/counter)
    return np.mean(psnr), np.mean(ssim)