from utils import *
from dataset import *

def fit(
    net: nn.Module,
    model: nn.Module,
    train_dset,
    val_dset,
    batch_size,
    ray_batch_size,
    loss_fn,
    optimizer,
    epochs,
    steps_til_summary,
    save_info
   ):
    (print_steps, vis_steps, val_epochs, save_epochs) = steps_til_summary
    train_dataloader = DataLoader(train_dset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      drop_last=True,
                                      collate_fn=train_dset.collate_fn
                                      )

    val_dataloader = DataLoader(val_dset,
                                      batch_size=min(batch_size,16),
                                      shuffle=True,
                                      drop_last=True,
                                      collate_fn=val_dset.collate_fn
                                      )
    val_dataiter = data_loop(val_dataloader)

    for e in range(epochs):
        step = 0
        for all_input in train_dataloader:
            all_input = to_gpu(all_input)
            all_images = all_input["images"] # (SB, NV, H*W, 3), values are between 0 and 1

            SB, NV, sl2, _ = all_images.shape
            NS = 1 # Number of source views, can only handle one now
            sl = int(np.sqrt(sl2))

            # Sample source indices
            src_idx = to_gpu(torch.randint(0, NV, (SB, NS)))

            # Encode
            src_images = batched_index_select_nd(all_images,src_idx).reshape(SB,NS,sl,sl,3).permute(0,1,4,2,3) # (SB, NS, 3, sl, sl)
            poses = batched_index_select_nd(all_input['cam2world'],src_idx) # (SB, NS, 4, 4)
            focal = batched_index_select_nd(all_input['focal'],src_idx)[0,0] # scalar
            c = batched_index_select_nd(all_input['c'],src_idx)[0,0,:] # (2)
            net.encode(src_images, poses, focal, c)

            # Sample rays
            rays_idx = torch.randint(0, sl2, (SB, NV, ray_batch_size))
            model_input = {}
            model_input['x_pix'] = batched_index_select_nd_second(all_input['x_pix'], rays_idx)
            model_input['cam2world'] = all_input['cam2world']
            model_input['intrinsics'] = all_input['intrinsics']
            ground_truth = 0.5 * batched_index_select_nd_second(all_input['images'], rays_idx) + 0.5

            # Compute the MLP output for the given input data and compute the loss
            model_output = model(model_input)

            # Implement a simple mean-squared-error loss
            loss = loss_fn(model_output, ground_truth)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()         

            # Every so often, we want to show what our model has learned.
            # It would be boring otherwise!
            if not step % print_steps:
                print(f"Epoch {e} Step {step}: loss = {float(loss.detach().cpu()):.5f}")
    #             print("Min depth %0.6f, max depth %0.6f" %
    #                           (model_output[1].min().detach().cpu().numpy(), model_output[1].max().detach().cpu().numpy()))
    #             print("Min color %0.6f, max color %0.6f" %
    #                           (model_output[0].min().detach().cpu().numpy(), model_output[0].max().detach().cpu().numpy()))
                

            if not step % vis_steps: # visualization
                model.eval()
                vis_idx =  to_gpu(torch.tensor([0]).expand(SB, 1))
                vis_gt = 0.5 * batched_index_select_nd(all_images,vis_idx) + 0.5
                all_input['x_pix'] = batched_index_select_nd(all_input['x_pix'], vis_idx)
                all_input['cam2world'] = batched_index_select_nd(all_input['cam2world'], vis_idx)
                all_input['intrinsics'] = batched_index_select_nd(all_input['intrinsics'], vis_idx)

                with torch.no_grad():
                    vis_output = model(all_input)
                    plot_output_ground_truth(vis_output, vis_gt, (sl,sl,3), src_idx)
                model.train()
            
            step += 1

        if not (e+1) % val_epochs: # validation
            model.eval()
            val_input = to_gpu(next(val_dataiter))
            val_images = val_input['images']
            SB, NV, sl2, _ = val_images.shape
            NS = 1 # Number of source views, can only handle one now
            sl = int(np.sqrt(sl2))

            # Sample source indices
            src_idx = to_gpu(torch.randint(0, NV, (SB, NS)))

            # Encode
            src_images = batched_index_select_nd(val_images,src_idx).reshape(SB,NS,sl,sl,3).permute(0,1,4,2,3) # (SB, NS, 3, sl, sl)
            poses = batched_index_select_nd(all_input['cam2world'],src_idx) # (SB, NS, 4, 4)
            focal = batched_index_select_nd(all_input['focal'],src_idx)[0,0] # scalar
            c = batched_index_select_nd(all_input['c'],src_idx)[0,0,:] # (2)
            net.encode(src_images, poses, focal, c)

            # Sample rays
            val_gt = 0.5 * val_images + 0.5

            # Compute the MLP output for the given input data and compute the loss
            with torch.no_grad():
                val_output = model(val_input)

            # Implement a simple mean-squared-error loss
            loss = loss_fn(val_output, val_gt)
            psnr, ssim = get_metrics(val_output, val_gt)
            model.train()
            print(f"Validation: Epoch {e}: loss = {float(loss.detach().cpu()):.5f}, psnr = {float(psnr):.5f}, ssim = {float(ssim):.5f}")

        if not (e+1) % save_epochs: # save model
            root_dir, model_name = save_info
            save_path = f"{root_dir}checkpoints/experiments/{model_name}_epoch{e}.pt"
            model.save_weights(save_path)
    return model_output

# Loss functions
def mse_loss(mlp_out, gt):
    img_coarse, _, _ = mlp_out
    return ((img_coarse - gt)**2).mean()

def mse_loss_fine(mlp_out, gt):
    img_coarse, img_fine, _ = mlp_out
    return ((img_coarse - gt)**2).mean() + ((img_fine - gt)**2).mean()

def mse_regularization_loss(mlp_out, gt, near=0.5, far=2.0):
    # add regularization loss
    img, _, depth = mlp_out
    penalty = (torch.min(depth-near, torch.zeros_like(depth)) ** 2) + (torch.max(depth-far, torch.zeros_like(depth)) ** 2)
    return ((img - gt)**2).mean() + torch.mean(penalty) * 10000

# Plotting functions
def plot_output_ground_truth(vis_output, vis_gt, resolution, src_idx):
    _, vis_img, vis_depth = vis_output

    SB = vis_img.shape[0]

    _, axes = plt.subplots(SB, 3, figsize=(18, 6*SB), squeeze=False)

    for sb in range(SB):
        img = vis_img[sb,0]
        depth = vis_depth[sb,0]
        gt = vis_gt[sb,0]

        axes[sb, 0].imshow(img.cpu().view(*resolution).detach().numpy())
        if src_idx[sb,0] == 0:
            title = "Trained MLP (same view as src)"
        else:
            title = "Trained MLP"
        axes[sb, 0].set_title(title)
        axes[sb, 1].imshow(gt.cpu().view(*resolution).detach().numpy())
        axes[sb, 1].set_title("Ground Truth")
        axes[sb, 2].imshow(depth.cpu().view(*resolution[:2]).detach().numpy(), cmap='Greys')
        axes[sb, 2].set_title("Depth")
        
        for j in range(3):
            axes[sb,j].set_axis_off()
    plt.show()

def get_metrics(mlp_out, gts): # rgb and gt are both 0 to 1 already
    _, rgbs, _ = mlp_out
    SB, NV, sl2, _ = rgbs.shape
    sl = int(np.sqrt(sl2))

    rgbs = rgbs.reshape(SB, NV, sl, sl, 3).detach().cpu().numpy()
    gts = gts.reshape(SB, NV, sl, sl, 3).detach().cpu().numpy()

    psnrs = []
    ssims = []

    for sb in range(SB):
        total_psnr = 0
        total_ssim = 0

        for nv in range(NV):
            ssim = skimage.measure.structural_similarity(rgbs[sb,nv,...], gts[sb,nv,...], channel_axis=-1, data_range=1)
            psnr = skimage.measure.peak_signal_noise_ratio(rgbs[sb,nv,...], gts[sb,nv,...], data_range=1)
            total_ssim += ssim
            total_psnr += psnr
        
        psnrs.append(total_psnr/NV)
        ssims.append(total_ssim/NV)

    return np.mean(psnr), np.mean(ssim)