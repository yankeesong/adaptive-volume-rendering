# adaptive-volume-rendering

## What pixelnerf is doing
- SB is 4, NV is 
- For non-visualization steps, only sample a batch of 128 rays
- Loss function for both coarse and fine images, equal weights
- Fine sampling: fine-depth samples, importance sampling
- Depth sampling: depth samples, random sampling
- Collect all samples, sort, then pass through fine network

## TODO list
- Rewrite the dataloader so we have train, val, test
- Distinguish between nn.train and nn.eval
- Allow superbatch
- Rewrite the trainer so we are sampling a subset of rays, but include all views
- Rewrite the trainder so we we have visualization step and validation step
- Do coarse + fine sampling
- For visualization, only do fine