from utils import *

def pick(list, item_idcs):
    if not list:
        return list
    return [list[i] for i in item_idcs]

class SceneInstanceDataset():
    """This creates a dataset class for a single object instance (such as a single car)."""

    def __init__(self,
                 filename,
                 instance_idx,
                 instance_key,
                 specific_observation_idcs=None,  # For few-shot case: Can pick specific observations only
                 img_sidelength=None,
                 num_images=-1):
        self.f = h5py.File(filename, 'r')
        self.instance_idx = instance_idx
        self.img_sidelength = img_sidelength
        self.instance_key = instance_key

        self.color_keys = self.f[self.instance_key]['rgb'].keys()
        self.pose_keys = self.f[self.instance_key]['pose'].keys()
        
        if specific_observation_idcs is not None:
            self.color_keys = pick(self.color_keys, specific_observation_idcs)
            self.pose_keys = pick(self.pose_keys, specific_observation_idcs)

        if specific_observation_idcs is None and num_images != -1:
            idcs = np.linspace(0, stop=len(self.color_paths), num=num_images, endpoint=False, dtype=int)
            self.color_keys = pick(self.color_keys, idcs)
            self.pose_keys = pick(self.pose_keys, idcs)

    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with whichimages are loaded."""
        self.img_sidelength = new_img_sidelength

    def __len__(self):
        return len(self.pose_keys)

    def __getitem__(self, idx): # This index means observation index
        focal, cx, cy, width, height = self.f[self.instance_key]['intrinsics'][...]
        cx = cx/width
        cy = cy/height
        focal = focal/height
        intrinsics = np.array([[focal, 0., cx],
                                [0., focal, cy],
                                [0., 0, 1]])

        intrinsics = torch.Tensor(intrinsics).float()

        img = self.f[self.instance_key]['rgb'][self.color_keys[idx]][...]
        ops = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
        

        x_pix = get_opencv_pixel_coordinates(*img.shape[:2])


        if self.img_sidelength is not None and img.shape[0] != self.img_sidelength:
            ops.append(transforms.Resize(self.img_sidelength))
            x_pix = resize(x_pix, 
                            (self.img_sidelength, self.img_sidelength), 
                            anti_aliasing=False,
                            order=0)
            
        x_pix = torch.from_numpy(rearrange(x_pix, 'i j c -> (i j) c'))

        op = transforms.Compose(ops)
        rgb = op(img).permute(1,2,0).reshape(self.img_sidelength*self.img_sidelength,-1)

        c2w = torch.from_numpy(self.f[self.instance_key]['pose'][self.pose_keys[idx]][...]) \
                    @ torch.diag(torch.tensor([1, -1, -1, 1], dtype=torch.float32))


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
                 filename,
                 img_sidelength=None,
                 max_num_instances=-1,
                 max_observations_per_instance=-1,
                 specific_observation_idcs=None,  # For few-shot case: Can pick specific observations only
                 samples_per_instance=2):

        self.f = h5py.File(filename, 'r')
        self.samples_per_instance = samples_per_instance
        self.instance_keys = sorted(self.f.keys())

        assert (len(self.instance_keys) != 0), "No objects in the data directory"
        

        if max_num_instances != -1:
            self.instance_keys = self.instance_keys[:max_num_instances]

        self.all_instances = [SceneInstanceDataset(filename,
                                                   instance_idx=idx,
                                                   instance_key=key,
                                                   specific_observation_idcs=specific_observation_idcs,
                                                   img_sidelength=img_sidelength,
                                                   num_images=max_observations_per_instance)
                              for idx, key in enumerate(self.instance_keys)]

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
        for key in batch_list[0][0].keys():
            temp = torch.stack([torch.stack([observation[key] for observation in scene]) for scene in batch_list ])
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