from utils import *
from dataset import *

def fit(
    net: nn.Module,
    model: nn.Module,
    test_dset,
    batch_size,
   ):

    test_dataloader = DataLoader(test_dset,
                                      batch_size=16,
                                      shuffle=True,
                                      drop_last=True,
                                      collate_fn=test_dset.collate_fn
                                      )


    model.eval()
    psnr_list = []
    ssim_list = []
    lpips_list = []
    for test_input in test_dataloader:
        test_images = test_input['images']
        SB, NV, sl2, _ = test_images.shape
        NS = 1 # Number of source views, can only handle one now
        sl = int(np.sqrt(sl2))

        # Sample source indices
        src_idx = to_gpu(torch.tensor([64]).unsqueeze(1).expand(SB, NS))

        # Encode
        src_images = batched_index_select_nd(test_images,src_idx).reshape(SB,NS,sl,sl,3).permute(0,1,4,2,3) # (SB, NS, 3, sl, sl)
        poses = batched_index_select_nd(test_input['cam2world'],src_idx) # (SB, NS, 4, 4)
        focal = batched_index_select_nd(test_input['focal'],src_idx)[0,0] # scalar
        c = batched_index_select_nd(test_input['c'],src_idx)[0,0,:] # (2)
        net.encode(src_images, poses, focal, c)

        # Sample rays
        val_gt = 0.5 * test_images + 0.5

        # Compute the MLP output for the given input data and compute the loss
        with torch.no_grad():
            val_output = model(test_input)

        psnr, ssim = get_metrics(val_output, val_gt)
        lpips = get_lpips(val_output, val_gt)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        lpips_list.append(lpips)
    model.train()
    psnr_mean = np.mean(psnr_list)
    ssim_mean = np.mean(ssim_list)
    lpips_mean = np.mean(lpips_list)
    print(f"Test: psnr = {float(psnr_mean):.5f}, ssim = {float(ssim_mean):.5f}, lpips = {float(lpips_mean):.5f}")

    return None

