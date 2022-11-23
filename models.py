from utils import *

class RadFieldAndRenderer(nn.Module):
    def __init__(self, rf, renderer):
        super().__init__()
        self.rf = rf
        self.renderer = renderer

    def forward(self, model_input):
        xy_pix = model_input['x_pix']
        intrinsics = model_input['intrinsics']
        c2w = model_input['cam2world']

        print(f'pixels shape: {xy_pix.shape}')

        rgb, depth = self.renderer(
                              c2w,
                              intrinsics,
                              xy_pix, 
                              self.rf
                              )
        
        return rgb, depth