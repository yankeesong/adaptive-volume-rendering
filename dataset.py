from utils import *

def parse_rgb(hdf5_dataset):
    s = hdf5_dataset[...].tobytes()
    f = io.BytesIO(s)

    img = imageio.imread(f)[:, :, :3]
    img = skimage.img_as_float32(img)
    return img


def parse_intrinsics(hdf5_dataset):
    s = hdf5_dataset[...].tobytes()
    s = s.decode('utf-8')

    lines = s.split('\n')
    f, cx, cy, _ = map(float, lines[0].split())
    full_intrinsic = torch.tensor([[f, 0., cx],
                                    [0., f, cy],
                                    [0., 0, 1]])

    return full_intrinsic


def parse_pose(hdf5_dataset):
    raw = hdf5_dataset[...]
    ba = bytearray(raw)
    s = ba.decode('ascii')

    lines = s.splitlines()
    pose = np.zeros((4, 4), dtype=np.float32)

    for i in range(16):
        pose[i // 4, i % 4] = lines[0].split(" ")[i]
    
    pose = torch.from_numpy(pose.squeeze())
    return pose
    

class SRNsCars(IterableDataset):
    def __init__(self, working_dir, max_num_instances=None, img_sidelength=None):
        self.f = h5py.File(f'{working_dir}data/cars_train.hdf5', 'r')
        self.instances = sorted(list(self.f.keys()))

        self.img_sidelength = img_sidelength

        if max_num_instances:
            self.instances = self.instances[:max_num_instances]

    def __len__(self):
        return len(self.instances)

    def __iter__(self, override_idx=None):
        while True:
            if override_idx is not None:
                idx = override_idx
            else:
                idx = random.randint(0, len(self.instances)-1)
                
            key = self.instances[idx]
            
            instance = self.f[key]
            rgbs_ds = instance['rgb']
            c2ws_ds = instance['pose']

            rgb_keys = list(rgbs_ds.keys())
            c2w_keys = list(c2ws_ds.keys())

            observation_idx = np.random.randint(0, len(rgb_keys))
            rgb = parse_rgb( rgbs_ds[rgb_keys[observation_idx]] )

            x_pix = get_opencv_pixel_coordinates(*rgb.shape[:2])

            # There is a lot of white-space around the cars - we'll thus crop the images a bit:
            rgb = rgb[32:-32, 32:-32]
            x_pix = x_pix[32:-32, 32:-32]

            # Nearest-neighbor downsampling of *both* the
            # RGB image and the pixel coordinates. This is better than down-
            # sampling RGB only and then generating corresponding pixel coordinates,
            # which generates "fake rays", i.e., rays that the camera
            # didn't actually capture with wrong colors. Instead, this simply picks a 
            # subset of the "true" camera rays.
            if self.img_sidelength is not None and rgb.shape[0] != self.img_sidelength:
                rgb = resize(rgb, 
                             (self.img_sidelength, self.img_sidelength), 
                             anti_aliasing=False,
                             order=0)
                rgb = torch.from_numpy(rgb)
                x_pix = resize(x_pix, 
                               (self.img_sidelength, self.img_sidelength), 
                               anti_aliasing=False,
                               order=0)
                
            x_pix = rearrange(x_pix, 'i j c -> (i j) c')
            c2w = parse_pose( c2ws_ds[c2w_keys[observation_idx]] )

            rgb = rearrange(rgb, 'i j c -> (i j) c')

            intrinsics = parse_intrinsics( instance['intrinsics.txt'] )
            intrinsics[:2, :3] /= 128. # Normalize intrinsics from resolution-specific intrinsics for 128x128

            model_input = {"cam2world": c2w,
                           "intrinsics": intrinsics,
                           "focal": intrinsics[0,0],
                           "c":torch.tensor([intrinsics[0,2], intrinsics[1,2]], dtype=torch.float32),
                           "x_pix": x_pix,
                           "idx": torch.tensor([idx]),
                           "images": rgb}

            yield model_input