from utils import *
from dataset import *

def test_approximate(
    net: nn.Module,
    model: nn.Module,
    test_dset,
    loss_params,
    observation_idx
   ):

    test_dataloader = DataLoader(test_dset,
                                      batch_size=1,
                                      shuffle=True,
                                      drop_last=True,
                                      collate_fn=test_dset.collate_fn
                                      )


    model.eval()
    psnr_list = []
    ssim_list = []
    lpips_list = []
    loss_list = []
    lpips_vgg = to_gpu(lpips.LPIPS(net="vgg"))
    count = 0
    for test_input in test_dataloader:
        print(count)
        test_images = test_input['images']
        SB, NV, sl2, _ = test_images.shape
        NS = 1 # Number of source views, can only handle one now
        sl = int(np.sqrt(sl2))

        # Sample source indices
        src_idx = torch.tensor([observation_idx]).unsqueeze(1).expand(SB, NS)

        # Encode
        src_images = batched_index_select_nd(test_images,src_idx).reshape(SB,NS,sl,sl,3).permute(0,1,4,2,3) # (SB, NS, 3, sl, sl)
        poses = batched_index_select_nd(test_input['cam2world'],src_idx) # (SB, NS, 4, 4)
        focal = batched_index_select_nd(test_input['focal'],src_idx)[0,0] # scalar
        c = batched_index_select_nd(test_input['c'],src_idx)[0,0,:] # (2)
        net.encode(to_gpu(src_images), to_gpu(poses), to_gpu(focal), to_gpu(c))
        
#         total_psnr = 0
#         total_ssim = 0
#         total_lpips = 0
        
        with torch.no_grad():
            nv = torch.randint(0,NV,(1,))
            ttest_input = {}
            ttest_input['x_pix'] = test_input['x_pix'][:,nv,...].squeeze(1)
            ttest_input['cam2world'] = test_input['cam2world'][:,nv,...].expand(1,sl2,4,4)
            ttest_input['intrinsics'] = test_input['intrinsics'][:,nv,...].squeeze(1)
            test_gt = 0.5 * to_gpu(test_images[:,nv,...].squeeze(1)) + 0.5
            test_output = model(to_gpu(ttest_input))

            psnr, ssim = get_metrics(test_output, test_gt)
            loss = loss_fn(test_output, test_gt, loss_params)
            rgbs_coarse, rgbs_fine, _, _ = test_output
            rgbs = rgbs_fine

            rgbs = rgbs.reshape(sl, sl, 3).permute(2,0,1).detach() * 2 - 1 # 
            test_gt = test_gt.reshape(sl, sl, 3).permute(2,0,1).detach() * 2 - 1 # 
            this_lpips = lpips_vgg(rgbs,test_gt).cpu().numpy()
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        lpips_list.append(this_lpips)
        loss_list.append(loss.cpu().numpy())
        count += 1
    psnr_mean = np.mean(psnr_list)
    ssim_mean = np.mean(ssim_list)
    lpips_mean = np.mean(lpips_list)
    loss_mean = np.mean(loss_list)
    print(f"Test: psnr = {float(psnr_mean):.5f}, ssim = {float(ssim_mean):.5f}, lpips = {float(lpips_mean):.5f}, loss = {float(loss_mean):.5f}")

    return None

