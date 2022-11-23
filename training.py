from utils import *

def fit(
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
        model_input, ground_truth = next(data_iterator)
        model_input, ground_truth = to_gpu(model_input), to_gpu(ground_truth)

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