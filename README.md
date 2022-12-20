# adaptive-volume-rendering

## What pixelnerf is doing
- SB is 4, NV is all
- For non-visualization steps, only sample a batch of 128 rays
- Loss function for both coarse and fine images, equal weights
- Fine sampling: fine-depth samples, importance sampling
- Depth sampling: depth samples, random sampling
- Collect all samples, sort, then pass through fine network

## TODO list
- Do coarse + fine sampling
- For visualization and validation, only do fine sampling
- Make the whole thing into a script and write results to tensorboard
- Add test step