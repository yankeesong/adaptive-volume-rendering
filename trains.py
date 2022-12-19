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
   ):
    (print_steps, vis_steps, val_steps, save_epochs) = steps_til_summary
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
                dummy = batched_index_select_nd(all_images,vis_idx)
                vis_gt = 0.5 * batched_index_select_nd(all_images,vis_idx) + 0.5
                all_input['x_pix'] = batched_index_select_nd(all_input['x_pix'], vis_idx)
                all_input['cam2world'] = batched_index_select_nd(all_input['cam2world'], vis_idx)
                all_input['intrinsics'] = batched_index_select_nd(all_input['intrinsics'], vis_idx)

                with torch.no_grad():
                    vis_output = model(all_input)
                    plot_output_ground_truth(vis_output, vis_gt, (sl,sl,3), src_idx)
                model.train()

            if not step % val_steps:
                pass # validation
            
            step += 1
        if not (e+1) % save_epochs:
            pass # save model
    return model_output

# Loss functions
def mse_loss(mlp_out, gt):
    img, depth = mlp_out
    return ((img - gt)**2).mean()

def mse_regularization_loss(mlp_out, gt, near=0.5, far=2.0):
    # add regularization loss
    img, depth = mlp_out
    penalty = (torch.min(depth-near, torch.zeros_like(depth)) ** 2) + (torch.max(depth-far, torch.zeros_like(depth)) ** 2)
    return ((img - gt)**2).mean() + torch.mean(penalty) * 10000

# Plotting functions
def plot_output_ground_truth(vis_output, vis_gt, resolution, src_idx):
    vis_img, vis_depth = vis_output

    SB = vis_img.shape[0]

    fig, axes = plt.subplots(SB, 3, figsize=(18, 6*SB), squeeze=False)

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

# Validation functions
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