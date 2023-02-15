import sys
import os
import configargparse
from pyhocon import ConfigFactory
root_dir = "/nobackup/users/yankeson/AVR/" 
sys.path.insert(0, f"{root_dir}/adaptive-volume-rendering/")
from dataset import *
from models import *
from renderers import *
import utils
from utils import *

def fit(
    net,
    model,
    train_dset,
    val_dset,
    batch_size,
    ray_batch_size,
    loss_params,
    optimizer,
    epochs,
    with_bbox,
    steps_til_summary,
    save_info,
    anomaly_detection=False
   ):
    (print_steps, val_steps, save_epochs, no_visualization) = steps_til_summary
    if no_visualization:
        print('no visualization')
    train_dataloader = DataLoader(train_dset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      drop_last=True,
                                      collate_fn=train_dset.collate_fn
                                      )

    val_dataloader = DataLoader(val_dset,
                                      batch_size=1,   # Only one instance for validation, for saving memory
                                      shuffle=True,
                                      drop_last=True,
                                      collate_fn=val_dset.collate_fn
                                      )
    val_dataiter = data_loop(val_dataloader)
    
    e = save_info[2]
    mean_losses = []
    for i in range(epochs):
        e += 1
        step = 1
        losses = []
        for all_input in train_dataloader:
            all_images = all_input["images"] # (SB, NV, H*W, 3), values are between 0 and 1

            SB, NV, sl2, _ = all_images.shape
            NS = 1 # Number of source views, can only handle one now
            sl = int(np.sqrt(sl2))

            # Sample source indices
            src_idx = torch.randint(0, NV, (SB, NS))

            # Encode
            src_images = batched_index_select_nd(all_images,src_idx).reshape(SB,NS,sl,sl,3).permute(0,1,4,2,3) # (SB, NS, 3, sl, sl)
            poses = batched_index_select_nd(all_input['cam2world'],src_idx) # (SB, NS, 4, 4)
            focal = batched_index_select_nd(all_input['focal'],src_idx)[0,0] # scalar
            c = batched_index_select_nd(all_input['c'],src_idx)[0,0,:] # (2)
            
            net.encode(to_gpu(src_images), to_gpu(poses), to_gpu(focal), to_gpu(c))
            
            # Sample rays
            if with_bbox:
                rays_idx = []
                for sb in range(SB):
                    pix = bbox_sample(all_input['bbox'][sb], ray_batch_size)
                    rays_idx.append(pix[..., 0] * sl2 + pix[..., 1] * sl + pix[..., 2])
                rays_idx = torch.stack(rays_idx)
            else:
                rays_idx = torch.randint(0, NV*sl2, (SB, ray_batch_size))    # (SB, ray_batch_size)
            
            
            model_input = {}
            model_input['x_pix'] = batched_index_select_nd(all_input['x_pix'].reshape(SB,-1,2), rays_idx)  # (SB, ray_batch_size, 2)
            model_input['cam2world'] = batched_index_select_nd(all_input['cam2world'].unsqueeze(2).expand(SB,NV,sl2,4,4).reshape(SB,-1,4,4), rays_idx) # (SB, ray_batch_size, 4, 4)
            model_input['intrinsics'] = all_input['intrinsics'][:,0,...] #(SB, 3, 3)
            ground_truth = 0.5 * to_gpu(batched_index_select_nd(all_input['images'].reshape(SB,-1,3), rays_idx)) + 0.5  # (SB, ray_batch_size, 3)
            
            if torch.sum(ground_truth) == 0:
                print('ray all has color 0')
            
            
            ############ Compare with pixelnerf
            # Generate rays for only the first scene
#             cam_rays = gen_rays(
#                 all_input['cam2world'][0], sl, sl, all_input['focal'][0,0], 0.8, 1.8, c = None
#             ).reshape(1,NV,sl2,-1) #(NV, sl, sl, 8)
#             ros, rds = get_world_rays(all_input['x_pix'][0:1], all_input['intrinsics'][0:1], all_input['cam2world'][0:1]) # (SB, NV, num_ways, 3)
#             print(f'ros diff: {((cam_rays[...,:3] - ros)**2).mean()}')
#             print(f'rds diff: {((cam_rays[...,3:6] - rds)**2).mean()}')
#             cam_rays = None
#             ros = None
#             rds = None

            
        
            
            torch.autograd.set_detect_anomaly(anomaly_detection)
            # Compute the MLP output for the given input data and compute the loss
            model_output = model(to_gpu(model_input))

            # Implement a simple mean-squared-error loss
            loss = loss_fn(model_output, ground_truth,loss_params)
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
                

            if not step % val_steps: # visualization
                model.eval()
                val_input = next(val_dataiter)
                val_images = val_input['images']
                SB, NV, sl2, _ = val_images.shape
                NS = 1 # Number of source views, can only handle one now
                sl = int(np.sqrt(sl2))

                # Sample source indices
                src_idx = torch.tensor([0]).unsqueeze(1).expand(SB, NS)

                # Encode
                src_images = batched_index_select_nd(val_images,src_idx).reshape(SB,NS,sl,sl,3).permute(0,1,4,2,3) # (SB, NS, 3, sl, sl)
                poses = batched_index_select_nd(val_input['cam2world'],src_idx) # (SB, NS, 4, 4)
                focal = batched_index_select_nd(val_input['focal'],src_idx)[0,0] # scalar
                c = batched_index_select_nd(val_input['c'],src_idx)[0,0,:] # (2)
                net.encode(to_gpu(src_images), to_gpu(poses), to_gpu(focal), to_gpu(c))

                # Sample rays
                val_gt = 0.5 * to_gpu(val_images[:,1,...]) + 0.5 # (SB, 1, sl2, 3)

                # Compute the MLP output for the given input data and compute the loss
                vval_input = {}
                vval_input['x_pix'] = val_input['x_pix'][:,1,...]  # (SB, sl2, 2)
                vval_input['cam2world'] = val_input['cam2world'][:,1,...].unsqueeze(1).expand(SB,sl2,4,4) # (SB, sl2, 4, 4)
                vval_input['intrinsics'] = val_input['intrinsics'][:,1,...] # (SB, 3, 3)
                with torch.no_grad():
                    val_output = model(to_gpu(vval_input))
                    psnr, ssim = get_metrics(val_output, val_gt, fine=(loss_params[0]!='coarse'))
                    val_loss = loss_fn(val_output, val_gt, loss_params)
                    if not no_visualization:
                        plot_output_ground_truth(val_output, val_gt, (sl,sl,3), fine=(loss_params[0]!='coarse'))
                model.train()
                if net.stop_encoder_grad:
                    net.encoder.eval()
                    
                print(f"Validation: Epoch {e} Step {step}: loss = {float(val_loss):.5f}, psnr = {float(psnr):.5f}, ssim = {float(ssim):.5f}")
            
            step += 1
            losses.append(loss.detach().cpu().item())


        if not e % save_epochs: # save model
            root_dir, model_name, _ = save_info
            save_path = f"{root_dir}checkpoints/experiments/{model_name}_epoch{e}.pt"
            model.save_weights(save_path)
        mean_losses.append(np.mean(losses))
    return mean_losses

if __name__ == "__main__":
    p = configargparse.ArgumentParser()
    p.add_argument('--root_dir', type=str, required=True,
                   help='Root directory')
    p.add_argument('--loss_mode', type=str, required=True,
                   help='Loss mode for training: corase/fine/both')
    p.add_argument('--depth_regularization', action='store_true', required=False,
                   help='whether to apply depth regularization')
    p.add_argument('--renderer', type=str, required=True,
                   help='what renderer to train')
    p.add_argument('--starting_epoch', type=int, required=True,
                   help='which epoch to resume from')

    p.add_argument('--sl', type=int, default=128, required=False,
                   help='Sidelength of image to be trained on')
    p.add_argument('--lr', type=float, default=1e-4, required=False,
                   help='Learning rate')
    p.add_argument('--max_num_instances', type=int, default=-1, required=False,
                   help='max number of instance for training, default -1, only change for debugging')
    p.add_argument('--batch_size', type=int, default=4, required=False,
                   help='batch size for training')
    p.add_argument('--epochs', type=int, default=50, required=False,
                   help='how many epochs to train')
    p.add_argument('--with_bbox', action='store_true', required=False,
                   help='whether to sample from bboxes')
    p.add_argument('--samples_per_instance', type=int, default=50, required=False,
                   help='number of views per instance per pass')
    p.add_argument('--ray_batch_size', type=int, default=512, required=False,
                   help='ray batch size')
    p.add_argument('--raymarch_steps', type=int, default=10, required=False,
                   help='raymarch steps')
    p.add_argument('--stop_encoder_grad', action='store_true', required=False,
                   help='Whether to stop Conv encoder gradient')
    p.add_argument('--anomaly_detection', action='store_true', required=False,
                   help='Whether to enable anomaly detection for debugging')
    p.add_argument('--bn', action='store_true', required=False,
                   help='Whether to add bn layer in mlp')
    p.add_argument('--no_visualization', action='store_true', default=True, required=False,
                   help='Whether to stop visualization')
    p.add_argument('--steps_print', type=int, default=5, required=False,
                   help='how many steps to print progress')
    p.add_argument('--steps_val', type=int, default=50, required=False,
                   help='how many steps to do validation and visualization')
    p.add_argument('--epochs_save', type=int, default=10, required=False,
                   help='how many epochs to save the model')


    opt = p.parse_args()

    root_dir = opt.root_dir

    os.chdir(root_dir)

    # Import everything

    sys.path.insert(0, f"{opt.root_dir}/adaptive-volume-rendering/")

    from dataset import *
    from models import *
    from renderers import *
    import utils
    from utils import *

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print(device)

    stop_encoder_grad = opt.stop_encoder_grad
    sl = opt.sl
    max_num_instances = opt.max_num_instances
    batch_size = opt.batch_size
    loss_params = (opt.loss_mode, opt.depth_regularization)
    samples_per_instance = opt.samples_per_instance
    ray_batch_size = opt.ray_batch_size
    raymarch_steps = opt.raymarch_steps
    epochs = opt.epochs
    with_bbox = opt.with_bbox
    print(f'bbox sampling status is {with_bbox}')
    steps_til_summary=(opt.steps_print,opt.steps_val,opt.epochs_save, opt.no_visualization)  # print, val, save (epoch)
    starting_epoch = opt.starting_epoch
    anomaly_detection = opt.anomaly_detection
    save_info = (root_dir, opt.renderer, starting_epoch)
    
    # Create a custom conf
    conf = ConfigFactory.parse_file(f"adaptive-volume-rendering/conf/default_mv.conf")

    # Create a pixelnerf net
    net = make_new_model(conf["model"], bn=opt.bn).to(device=device)

    # Combine with volumerenderer
    if 'Raymarcher' in save_info[1]:
        renderer = Raymarcher.from_conf(conf["raymarcher"],raymarch_steps).to(device=device)
    elif save_info[1][0:2] == 'VR':
        renderer = VolumeRenderer.from_conf(conf["normal_renderer"]).to(device=device)
    else:
        renderer = AdaptiveVolumeRenderer.from_conf(conf["adaptive_renderer"]).to(device=device)
    rf_and_renderer = RadFieldAndRenderer(net, renderer).to(device=device)

    # Load pretrained weights
    model_path = f"{root_dir}checkpoints/experiments/{save_info[1]}_epoch{starting_epoch}.pt"
    rf_and_renderer.load_weights(model_path, strict=False, device=device)
    net.stop_encoder_grad = stop_encoder_grad
    if stop_encoder_grad:
        net.encoder.eval()

    # Train dataset
    train_dset = SceneClassDataset(filename=f"{root_dir}data/cars_train.hdf5",
                                                 max_num_instances=max_num_instances,
                                                 max_observations_per_instance=-1,
                                                 img_sidelength=sl,
                                                 specific_observation_idcs=None,
                                                 samples_per_instance=samples_per_instance)

    val_dset = SceneClassDataset(filename=f"{root_dir}data/cars_val.hdf5",
                                                 max_num_instances=max_num_instances,
                                                 max_observations_per_instance=-1,
                                                 img_sidelength=sl,
                                                 specific_observation_idcs=[64],
                                                 samples_per_instance=2)

    # Configurations
    optim = torch.optim.Adam(lr=opt.lr, params=rf_and_renderer.parameters())
    print(f'lr is {opt.lr}')
    
    try:
        mean_losses = fit(net,
            rf_and_renderer, 
            train_dset, 
            val_dset,
            batch_size=batch_size,
            ray_batch_size=ray_batch_size,
            loss_params=loss_params, 
            optimizer=optim, 
            epochs=epochs,
            with_bbox=with_bbox,
            steps_til_summary=steps_til_summary,
            save_info=save_info
            )
        plt.plot(range(save_info[2],save_info[2]+epochs),mean_losses)
        plt.savefig(f"{root_dir}logs/losses_{opt.renderer}_epoch{starting_epoch}.png")

    except Exception:
        print(traceback.format_exc())