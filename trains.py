from utils import *

def fit(
    net: nn.Module,
    model: nn.Module,
    data_iterator,
    loss_fn,
    resolution: Tuple,
    optimizer,
    plotting_function = None,
    steps_til_summary = 500,
    total_steps=2001
   ):

    losses = []
    for step in range(total_steps):
        # Get the next batch of data and move it to the GPU
        model_input = next(data_iterator)
        model_input = to_gpu(model_input)
        ground_truth = model_input["images"] # (NV, H*W, 3)

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
        print(f'source image has size {src_images.shape}')
        poses = model_input['cam2world'][src_idx,...]
        print(f'poses has size {poses.shape}')
        focal = model_input['focal'][src_idx,...]
        c = model_input['c'][src_idx,...]
        
        net.encode(src_images, poses, focal, c)

        # Compute the MLP output for the given input data and compute the loss
        model_output = model(model_input)

        # Implement a simple mean-squared-error loss between the 
        loss = loss_fn(model_output, ground_truth, model) # Note: loss now takes "model" as input.

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate the losses so that we can plot them later
        # losses.append(loss.detach().cpu().numpy())            

        # Every so often, we want to show what our model has learned.
        # It would be boring otherwise!
        if not step % steps_til_summary:
            print(f"Step {step}: loss = {float(loss.detach().cpu()):.5f}")

            if plotting_function is not None: # Note: we now call the "plotting function" instead of hard-coding the plotting here.
                plotting_function(model_output, ground_truth, resolution)

    return model_output

def mse_loss(mlp_out, gt, model):
    img, depth = mlp_out
    return ((img - gt)**2).mean()

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