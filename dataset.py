from utils import *
from glob import glob

# def parse_rgb(hdf5_dataset):
#     s = hdf5_dataset[...].tobytes()
#     f = io.BytesIO(s)

#     img = imageio.imread(f)[:, :, :3]
#     img = skimage.img_as_float32(img)
#     return img


# def parse_intrinsics(hdf5_dataset):
#     s = hdf5_dataset[...].tobytes()
#     s = s.decode('utf-8')

#     lines = s.split('\n')
#     f, cx, cy, _ = map(float, lines[0].split())
#     full_intrinsic = torch.tensor([[f, 0., cx],
#                                     [0., f, cy],
#                                     [0., 0, 1]])

#     return full_intrinsic


# def parse_pose(hdf5_dataset):
#     raw = hdf5_dataset[...]
#     ba = bytearray(raw)
#     s = ba.decode('ascii')

#     lines = s.splitlines()
#     pose = np.zeros((4, 4), dtype=np.float32)

#     for i in range(16):
#         pose[i // 4, i % 4] = lines[0].split(" ")[i]
    
#     pose = torch.from_numpy(pose.squeeze())
#     pose = pose @ torch.diag(
#             torch.tensor([1, -1, -1, 1], dtype=torch.float32))
#     return pose
    

# class SRNsCars(IterableDataset):
#     def __init__(self, working_dir, max_num_instances=None, img_sidelength=None):
#         self.f = h5py.File(f'{working_dir}data/cars_train.hdf5', 'r')
#         self.instances = sorted(list(self.f.keys()))

#         self.img_sidelength = img_sidelength

#         if max_num_instances:
#             self.instances = self.instances[:max_num_instances]

#     def __len__(self):
#         return len(self.instances)

#     def __iter__(self, override_idx=None):
#         while True:
#             if override_idx is not None:
#                 idx = override_idx
#             else:
#                 idx = random.randint(0, len(self.instances)-1)
                
#             key = self.instances[idx]
            
#             instance = self.f[key]
#             rgbs_ds = instance['rgb']
#             c2ws_ds = instance['pose']

#             rgb_keys = list(rgbs_ds.keys())
#             c2w_keys = list(c2ws_ds.keys())

#             observation_idx = np.random.randint(0, len(rgb_keys))
#             rgb = parse_rgb( rgbs_ds[rgb_keys[observation_idx]] )

#             x_pix = get_opencv_pixel_coordinates(*rgb.shape[:2])

#             # There is a lot of white-space around the cars - we'll thus crop the images a bit:
#             # rgb = rgb[32:-32, 32:-32]
#             # x_pix = x_pix[32:-32, 32:-32]

#             # Nearest-neighbor downsampling of *both* the
#             # RGB image and the pixel coordinates. This is better than down-
#             # sampling RGB only and then generating corresponding pixel coordinates,
#             # which generates "fake rays", i.e., rays that the camera
#             # didn't actually capture with wrong colors. Instead, this simply picks a 
#             # subset of the "true" camera rays.
#             if self.img_sidelength is not None and rgb.shape[0] != self.img_sidelength:
#                 rgb = resize(rgb, 
#                              (self.img_sidelength, self.img_sidelength), 
#                              anti_aliasing=False,
#                              order=0)
#                 rgb = torch.from_numpy(rgb)
#                 x_pix = resize(x_pix, 
#                                (self.img_sidelength, self.img_sidelength), 
#                                anti_aliasing=False,
#                                order=0)
                
#             x_pix = rearrange(x_pix, 'i j c -> (i j) c')
#             c2w = parse_pose( c2ws_ds[c2w_keys[observation_idx]] )

#             rgb = rearrange(rgb, 'i j c -> (i j) c')

#             # print(f'key is {key}')
#             # print(f'observation idx is {observation_idx}')
#             # print(f'pose key is {c2w_keys[observation_idx]}')

#             intrinsics = parse_intrinsics( instance['intrinsics.txt'] )
#             intrinsics[:2, :3] /= 128. # Normalize intrinsics from resolution-specific intrinsics for 128x128

#             model_input = {"cam2world": c2w,
#                            "intrinsics": intrinsics,
#                            "focal": intrinsics[0,0]*self.img_sidelength, # For pixelnerf
#                            "c":torch.tensor([intrinsics[0,2]*self.img_sidelength, intrinsics[1,2]*self.img_sidelength], dtype=torch.float32), # For pixelnerf
#                            "x_pix": x_pix, # For volume renderer
#                            "idx": torch.tensor([idx]),
#                            "images": rgb}

#             yield model_input


def pick(list, item_idcs):
    if not list:
        return list
    return [list[i] for i in item_idcs]

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs


def parse_intrinsics(filepath, trgt_sidelength=None, invert_y=False):
    # Get camera intrinsics
    with open(filepath, 'r') as file:
        f, cx, cy, _ = map(float, file.readline().split())
        grid_barycenter = torch.Tensor(list(map(float, file.readline().split())))
        scale = float(file.readline())
        height, width = map(float, file.readline().split())

        try:
            world2cam_poses = int(file.readline())
        except ValueError:
            world2cam_poses = None

    if world2cam_poses is None:
        world2cam_poses = False

    world2cam_poses = bool(world2cam_poses)

    cx = cx/width
    cy = cy/height
    f = f/height

    # Build the intrinsic matrices
    intrinsic = np.array([[f, 0., cx],
                               [0., f, cy],
                               [0., 0, 1]])

    return intrinsic, grid_barycenter, scale, world2cam_poses

def load_pose(filename):
    lines = open(filename).read().splitlines()
    if len(lines) == 1:
        pose = np.zeros((4, 4), dtype=np.float32)
        for i in range(16):
            pose[i // 4, i % 4] = lines[0].split(" ")[i]
        return pose.squeeze()
    else:
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines[:4])]
        return np.asarray(lines).astype(np.float32).squeeze()



class SceneInstanceDataset():
    """This creates a dataset class for a single object instance (such as a single car)."""

    def __init__(self,
                 instance_idx,
                 instance_dir,
                 specific_observation_idcs=None,  # For few-shot case: Can pick specific observations only
                 img_sidelength=None,
                 num_images=-1):
        self.instance_idx = instance_idx
        self.img_sidelength = img_sidelength
        self.instance_dir = instance_dir

        color_dir = os.path.join(instance_dir, "rgb")
        pose_dir = os.path.join(instance_dir, "pose")
        param_dir = os.path.join(instance_dir, "params")

        if not os.path.isdir(color_dir):
            print("Error! root dir %s is wrong" % instance_dir)
            return

        self.has_params = os.path.isdir(param_dir)
        self.color_paths = sorted(glob_imgs(color_dir))
        self.pose_paths = sorted(glob(os.path.join(pose_dir, "*.txt")))

        if self.has_params:
            self.param_paths = sorted(glob(os.path.join(param_dir, "*.txt")))
        else:
            self.param_paths = []

        # Filter out bad images
        idcs = []
        for i in range(len(self.pose_paths)):
            if torch.from_numpy(load_pose(self.pose_paths[i]))[2,3]>0:
                idcs.append(i)
        self.color_paths = pick(self.color_paths, idcs)
        self.pose_paths = pick(self.pose_paths, idcs)
        self.param_paths = pick(self.param_paths, idcs)
        
        if specific_observation_idcs is not None:
            self.color_paths = pick(self.color_paths, specific_observation_idcs)
            self.pose_paths = pick(self.pose_paths, specific_observation_idcs)
            self.param_paths = pick(self.param_paths, specific_observation_idcs)
        elif num_images != -1:
            idcs = np.linspace(0, stop=len(self.color_paths), num=num_images, endpoint=False, dtype=int)
            self.color_paths = pick(self.color_paths, idcs)
            self.pose_paths = pick(self.pose_paths, idcs)
            self.param_paths = pick(self.param_paths, idcs)

    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with whichimages are loaded."""
        self.img_sidelength = new_img_sidelength

    def __len__(self):
        return len(self.pose_paths)

    def __getitem__(self, idx): # This index means observation index
        intrinsics, _, _, _ = parse_intrinsics(os.path.join(self.instance_dir, "intrinsics.txt"),
                                                                  trgt_sidelength=self.img_sidelength)
        intrinsics = torch.Tensor(intrinsics).float()

        img = imageio.imread(self.color_paths[idx])[:, :, :3]
        rgb = skimage.img_as_float32(img)

        x_pix = get_opencv_pixel_coordinates(*rgb.shape[:2])

        # rgb = rgb[32:-32, 32:-32]
        # x_pix = x_pix[32:-32, 32:-32]

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
            
        x_pix = torch.from_numpy(rearrange(x_pix, 'i j c -> (i j) c'))

        rgb = rearrange(rgb, 'i j c -> (i j) c')

        c2w = torch.from_numpy(load_pose(self.pose_paths[idx])) @ torch.diag(torch.tensor([1, -1, -1, 1], dtype=torch.float32))


        model_input = {"cam2world": c2w,
                           "intrinsics": intrinsics,
                           "focal": intrinsics[0,0]*self.img_sidelength, # For pixelnerf
                           "c":torch.tensor([intrinsics[0,2]*self.img_sidelength, intrinsics[1,2]*self.img_sidelength], dtype=torch.float32), # For pixelnerf
                           "x_pix": x_pix, # For volume renderer
                           "idx": torch.tensor([self.instance_idx]),
                           "images": rgb}

        return model_input


class SceneClassDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 root_dir,
                 img_sidelength=None,
                 max_num_instances=-1,
                 max_observations_per_instance=-1,
                 specific_observation_idcs=None,  # For few-shot case: Can pick specific observations only
                 samples_per_instance=2):

        self.samples_per_instance = samples_per_instance
        self.instance_dirs = sorted(glob(os.path.join(root_dir, "*/")))

        assert (len(self.instance_dirs) != 0), "No objects in the data directory"

        if max_num_instances != -1:
            self.instance_dirs = self.instance_dirs[:max_num_instances]

        self.all_instances = [SceneInstanceDataset(instance_idx=idx,
                                                   instance_dir=dir,
                                                   specific_observation_idcs=specific_observation_idcs,
                                                   img_sidelength=img_sidelength,
                                                   num_images=max_observations_per_instance)
                              for idx, dir in enumerate(self.instance_dirs)]

        self.num_per_instance_observations = [len(obj) for obj in self.all_instances]
        self.num_instances = len(self.all_instances)

    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with whichimages are loaded."""
        for instance in self.all_instances:
            instance.set_img_sidelength(new_img_sidelength)

    def __len__(self):
        return np.sum(self.num_per_instance_observations)

    def get_instance_idx(self, idx):
        """Maps an index into all tuples of all objects to the idx of the tuple relative to the other tuples of that
        object
        """
        obj_idx = 0
        while idx >= 0:
            idx -= self.num_per_instance_observations[obj_idx]
            obj_idx += 1
        return obj_idx - 1, int(idx + self.num_per_instance_observations[obj_idx - 1])

    def collate_fn(self,batch_list):
        batch_num = len(batch_list)
        all_dict = {}
        # key_list = ['focal','intrinsics','c','idx']
        for key in batch_list[0][0].keys():
            temp = torch.stack([torch.stack([observation[key] for observation in scene]) for scene in batch_list ])
            # if key in key_list:
            #     temp = temp[:,0,...].unsqueeze(1)
            if batch_num == 1:
                temp = temp.squeeze(0)
            all_dict[key] = temp
        return all_dict

    def __getitem__(self, idx): # This idx is total index
        """Each __getitem__ call yields a list of self.samples_per_instance observations of a single scene (each a dict),
        as well as a list of ground-truths for each observation (also a dict)."""
        obj_idx, rel_idx = self.get_instance_idx(idx)

        observations = []
        observations.append(self.all_instances[obj_idx][rel_idx])

        for i in range(self.samples_per_instance - 1):
            observations.append(self.all_instances[obj_idx][np.random.randint(len(self.all_instances[obj_idx]))])

        #ground_truth = [{'rgb':ray_bundle['rgb']} for ray_bundle in observations]

        return observations