# adaptive-volume-rendering

## What pixelnerf is doing
- SB is 4, NV is all
- For non-visualization steps, only sample a batch of 128 rays
- Loss function for both coarse and fine images, equal weights
- Fine sampling: fine-depth samples, importance sampling
- Depth sampling: depth samples, random sampling
- Collect all samples, sort, then pass through fine network

## Training code:
- sbatch --export=renderer='AVR',lr=1e-5,starting_epoch=54,loss_mode='fine' train.sh 

## TODO list
- Other ideas for raw training:
    - Separate the coarse and fine network
- Other ideas for fine training:
    - bbox sampling: seems working, AVR and Raymarcher 210-240 epochs, lr=1e-5. Trying 240-270 epochs lr=5e-5
    - drop raymarcher loss?
    - sample wider
- More than 10 raymarcher steps.
- Make the whole thing into a script and write results to tensorboard


## Results: aim for 0.01 training loss (both sum), 0.05 with bbox (both sum)
- Random test:
    - SRN (reported):           psnr = 22.25000, ssim = 0.89000
    - VR:  10000 epochs:        psnr = 23.33842, ssim = 0.90478, lpips = 0.11172, loss = 0.01209
           10000 epochs coarse: psnr = 23.31252, ssim = 0.90418, lpips = 0.11463, loss = 0.00618
    - AVR:    30 epochs:        psnr = 18.37844, ssim = 0.80092, lpips = 0.26828
              80 epochs:        psnr = 20.01904, ssim = 0.83636, lpips = 0.22749, loss = 0.02453
    - AVR: 10030 epochs:        psnr = 20.36204, ssim = 0.84194, lpips = 0.21646
           10060 epochs:        psnr = 21.59525, ssim = 0.87302, lpips = 0.17355, loss = 0.01761
           10090 epochs:        psnr = 22.00173, ssim = 0.87827, lpips = 0.16816, loss = 0.01613
           10120 epochs:        psnr = 22.03870, ssim = 0.88107, lpips = 0.16035, loss = 0.01640
           10150 epochs:        psnr = 22.17718, ssim = 0.88307, lpips = 0.15826, loss = 0.01567
           10180 epochs:        psnr = 22.29578, ssim = 0.88452, lpips = 0.15685, loss = 0.00725
           10210 epochs:        psnr = 22.20050, ssim = 0.88350, lpips = 0.15767, loss = 0.00737
           10240 epochs:        psnr = 22.32232, ssim = 0.88466, lpips = 0.15735, loss = 0.00719
    - Raymarcher: 10030 epochs: psnr = 18.61872, ssim = 0.80009, lpips = 0.28302
                  10060 epochs: psnr = 20.95928, ssim = 0.85699, lpips = 0.19574
                  10090 epochs: psnr = 21.25482, ssim = 0.86238, lpips = 0.19464
                  10120 epochs: psnr = 21.37475, ssim = 0.86694, lpips = 0.18478, loss = 0.00860
                  10150 epochs: psnr = 21.47335, ssim = 0.86738, lpips = 0.18222, loss = 0.00839
                  10180 epochs: psnr = 21.58362, ssim = 0.86951, lpips = 0.18005, loss = 0.00821
                  10240 epochs: psnr = 21.63945, ssim = 0.86932, lpips = 0.18283, loss = 0.00810
                  
                  
                  

          
          
## Test pixelnerf package: No need now!!!!!!!!
- Train: python train/train.py -n srn_car_test -c conf/exp/srn.conf -D ../data/cars --gpu_id='0' --resume --epochs=1. Currently cannot deal with 50 vs 250 images.
- Evaluation: python eval/eval_approx.py -D ../data/cars -n srn_car -P '64'. Currently has empty images, will deal with it later
    - default: psnr 23.535904424880005 ssim 0.9084699700276057
