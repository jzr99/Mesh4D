from tkinter import S
import torch
import numpy as np
import os
from util.console import print
import glob
import re
import trimesh
import imageio
from im2mesh.utils.libmesh import check_mesh_contains
from transformers import BitImageProcessor
from munch import Munch
import util.misc as misc
import random
import cv2
import yaml
import torch.utils.data
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_info
from typing import Union, List, Tuple
from torch.utils.data.distributed import DistributedSampler


def get_loc_scale(mesh):
    # Determine bounding box

    bbox = mesh.bounding_box.bounds # bounds Bounding box with [min, max] coordinates
    # Compute location and scale
    loc = (bbox[0] + bbox[1]) / 2 
    scale = (bbox[1] - bbox[0]).max() / (1 - 0.1)

    return loc, scale

def get_loc_scale_min0(mesh):
    # Determine bounding box

    bbox = mesh.bounding_box.bounds # bounds Bounding box with [min, max] coordinates
    # Compute location and scale
    # loc = (bbox[0] + bbox[1]) / 2 
    loc = bbox[0]
    scale = (bbox[1] - bbox[0]).max() / (1 - 0.1)

    return loc, scale

def padding(image, mask, center=True, padding_ratio_range=[1.15, 1.15]):
    """
    Pad the input image and mask to a square shape with padding ratio.

    Args:
        image (np.ndarray): Input image array of shape (H, W, C).
        mask (np.ndarray): Corresponding mask array of shape (H, W).
        center (bool): Whether to center the original image in the padded output.
        padding_ratio_range (list): Range [min, max] to randomly select padding ratio.

    Returns:
        newimg (np.ndarray): Padded image of shape (resize_side, resize_side, 3).
        newmask (np.ndarray): Padded mask of shape (resize_side, resize_side).
    """
    h, w = image.shape[:2]
    max_side = max(h, w)

    # Select padding ratio either fixed or randomly within the given range
    if padding_ratio_range[0] == padding_ratio_range[1]:
        padding_ratio = padding_ratio_range[0]
    else:
        padding_ratio = random.uniform(padding_ratio_range[0], padding_ratio_range[1])
    resize_side = int(max_side * padding_ratio)
    # resize_side = int(max_side * 1.15)

    pad_h = resize_side - h
    pad_w = resize_side - w
    if center:
        start_h = pad_h // 2
    else:
        start_h = pad_h - resize_side // 20
        
    start_w = pad_w // 2

    # Create new white image and black mask with padded size
    newimg = np.ones((resize_side, resize_side, 3), dtype=np.uint8) * 255
    newmask = np.zeros((resize_side, resize_side), dtype=np.uint8)
    
    # Place original image and mask into the padded canvas
    newimg[start_h:start_h + h, start_w:start_w + w] = image
    newmask[start_h:start_h + h, start_w:start_w + w] = mask
    
    return newimg, newmask


class AxisScaling(object):
    def __init__(self, interval=(0.75, 1.25), jitter=True):
        assert isinstance(interval, tuple)
        self.interval = interval
        self.jitter = jitter
        
    def __call__(self, surface, point):
        scaling = torch.rand(1, 3) * 0.5 + 0.75
        surface = surface * scaling
        point = point * scaling

        scale = (0.5 / torch.abs(surface).max().item()) * 0.999999
        surface *= scale
        point *= scale

        if self.jitter:
            surface += 0.005 * torch.randn_like(surface)
            surface.clamp_(min=-0.5, max=0.5)

        return surface, point
    
class OBJVERSES(torch.utils.data.Dataset):
    def __init__(self, 
        split, cfg, log_dir=None,
        image_size=224,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        padding=True,
        padding_ratio_range=[1.15, 1.15],
        image_transform=None,
        dataset_path=None,
    ):
        if dataset_path is not None:
            self.dataset_path = dataset_path
        else:
            self.dataset_path = "../DATA"

        self.split = split
        if self.split == 'train':
            # self.select_index_list = list(range(0, 10000))
            self.select_index_list = list(range(0, 5000))

        if isinstance(cfg, str):
            config_yaml = yaml.load(open(cfg, 'r'), Loader=yaml.FullLoader)
            config_yaml = Munch(config_yaml)
            default_cfg = misc.dummy_cfg()
            config_yaml = misc.merge_munch(config_yaml, default_cfg)
            cfg = config_yaml

        self.stage = cfg.stage # 'shape' or 'deform'
        self.mode = cfg.mode # 'ae' or 'diff'
        self.repeat = cfg.repeat
        self.seed = cfg.seed 
        self.use_depth = cfg.use_depth
        self.pc_size = cfg.pc_size
        self.num_samples = cfg.num_samples
        self.n_inputs = cfg.n_inputs
        self.inputs_noise_std = cfg.inputs_noise_std
        self.n_sample_pro_model = cfg.n_sample_pro_model
        self.interval_between_frames = cfg.interval_between_frames
        self.n_selected_frames = cfg.n_selected_frames
        self.length_sequence = cfg.length_sequence
        self.n_files_per_sequence = cfg.n_files_per_sequence
        self.offset_sequence = cfg.offset_sequence
        self.n_training_frames = cfg.n_training_frames
        self.overfitting = cfg.get('overfitting', False)
        self.normalize_first_frame = cfg.get('normalize_first_frame', False)
        self.interval_between_sequence = cfg.get('interval_between_sequence', 1)
        self.revert = cfg.get('revert', False)
        self.surface_sampling = cfg.get('surface_sampling', True)
        self.augment_points = cfg.get('augment_points', True)
        self.return_normal = cfg.get('return_normal', True)
        self.filter_num = cfg.get('filter_num', 1048)
        # self.feature_extractor_dinov2 = BitImageProcessor.from_pretrained(
        #     "VAST-AI/TripoSG",
        #     subfolder="feature_extractor_dinov2",  # reads preprocessor_config.json
        #     trust_remote_code=True,                # safe to keep; some processors require it
        # )

        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.padding = padding
        self.padding_ratio_range = padding_ratio_range


        if image_transform is not None:
            self.image_transform = image_transform
        else:
            self.image_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.image_size),
                transforms.Normalize(mean=self.mean, std=self.std)]
            )

        
        if self.split.find('train') == -1:
            self.repeat = 1
        
        self.split_lst = []
        self.test_lst = []
        for folder in os.listdir(self.dataset_path):
            for name in os.listdir(os.path.join(self.dataset_path, folder)):
                self.split_lst.append(os.path.join(folder, name))
        print("test_lst: ", self.test_lst)
        print("Number of objects: ", len(self.split_lst))
        
        # Sort by name
        self.split_lst.sort()
        if self.overfitting:
            self.split_lst = self.test_lst
        if self.split != 'train':
            if len(self.test_lst) > 0:
                self.split_lst = self.test_lst
        self.split_lst_save = self.split_lst.copy()
        # write the training split_lst to a file 
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, 'split_lst.txt'), 'w') as f:
                for line in self.split_lst:
                    f.write(line + '\n')
        self.prepare_data_lst()
        
        if self.split == 'train': # train
            self.transform = AxisScaling((0.75, 1.25), True)
            if self.mode == 'diff' or self.mode == 'diffMulti':
                self.transform = None
            self.sampling = True
            self.surface_sampling = True
        else: # val and test
            self.transform = None
            self.sampling = False
            self.surface_sampling = True


    def load_render(self, imgs_paths):
        # imgs_choice = self.rng.sample(imgs_path, 1)
        images_before_padding, masks_before_padding = [], []
        images, masks = [], []
        for image_path in imgs_paths:
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            assert image.shape[2] == 4
            alpha = image[:, :, 3:4].astype(np.float32) / 255
            forground = image[:, :, :3]
            background = np.ones_like(forground) * 255
            img_new = forground * alpha + background * (1 - alpha)
            image = img_new.astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = (alpha[:, :, 0] * 255).astype(np.uint8)

            images_before_padding.append(image)
            masks_before_padding.append(mask)

        for i, (image, mask) in enumerate(zip(images_before_padding, masks_before_padding)):
            if i == 0:
                x_min_all, y_min_all, x_max_all, y_max_all = 999999, 999999, 0, 0


            if self.padding:
                h, w = image.shape[:2]
                binary = mask > 0.3
                non_zero_coords = np.argwhere(binary)
                if len(non_zero_coords) == 0:
                    non_zero_coords = np.array([[0, 0], [h-1, w-1]])

                x_min, y_min = non_zero_coords.min(axis=0)
                x_max, y_max = non_zero_coords.max(axis=0)
                x_min_all, y_min_all, x_max_all, y_max_all = min(x_min_all, x_min), min(y_min_all, y_min), max(x_max_all, x_max), max(y_max_all, y_max)
                
        for i, (image, mask) in enumerate(zip(images_before_padding, masks_before_padding)):
            if self.padding:
                image, mask = padding(
                    image[max(x_min_all - 5, 0):min(x_max_all + 5, h), max(y_min_all - 5, 0):min(y_max_all + 5, w)],
                    mask[max(x_min_all - 5, 0):min(x_max_all + 5, h), max(y_min_all - 5, 0):min(y_max_all + 5, w)],
                    center=True, padding_ratio_range=self.padding_ratio_range)
            
            if self.image_transform:
                image = self.image_transform(image)
                mask = np.stack((mask, mask, mask), axis=-1)
                mask = self.image_transform(mask)
                    
                images.append(image)
                masks.append(mask)
            
        images = torch.stack(images, dim=0) # [T, 3, H, W]
        masks = torch.stack(masks, dim=0)[:, :1, ...] # [T, 1, H, W]
        return images, masks
        
    def prepare_data_lst(self):
        self.split_lst_path = [os.path.join(self.dataset_path, line) for line in self.split_lst]
        # self.split_lst_pcl_path = [sorted(glob.glob(os.path.join(self.dataset_path, line, 'frame_*.ply'))) for line in self.split_lst]
        def natural_sort_key(path):
            match = re.search(r'(\d+)\.png$', path)
            return int(match.group(1)) if match else float('inf')
        self.split_lst_rgb_path = [sorted(glob.glob(os.path.join(self.dataset_path, line, '*.png')), key=natural_sort_key) for line in self.split_lst]

        if self.repeat > 1:
            # self.split_lst_pcl_path = self.split_lst_pcl_path * self.repeat
            self.split_lst_rgb_path = self.split_lst_rgb_path * self.repeat
        # self.split_lst_pcl_len = [len(pcl_path) for pcl_path in self.split_lst_pcl_path]
        self.split_lst_rgb_len = [len(rgb_path) for rgb_path in self.split_lst_rgb_path]
            
        self.models = []
        if self.stage == 'deform':
            if self.mode == 'ae':
                models_out, indices_out, stride_list = self.subdivide_into_sequences(self.split_lst_save * self.repeat, self.split_lst_rgb_len * self.repeat)
                self.models += [
                    {"model": m, "start_idx": indices_out[i], "stride": stride_list[i]} for i, m in enumerate(models_out)
                ]
            elif self.mode == 'diff':
                models_out, indices_out, stride_list = self.subdivide_into_sequences(self.split_lst_save * self.repeat, self.split_lst_rgb_len * self.repeat)
                self.models += [
                    {"model": m, "start_idx": indices_out[i], "stride": stride_list[i]} for i, m in enumerate(models_out)
                ]
            elif self.mode == 'aeMulti':
                models_out, indices_out, stride_list = self.subdivide_into_sequences(self.split_lst_save * self.repeat, self.split_lst_rgb_len * self.repeat)
                self.models += [
                    {"model": m, "start_idx": indices_out[i], "stride": stride_list[i]} for i, m in enumerate(models_out)
                ]
            elif self.mode == 'diffMulti':
                models_out, indices_out, stride_list = self.subdivide_into_sequences(self.split_lst_save * self.repeat, self.split_lst_rgb_len * self.repeat)
                self.models += [
                    {"model": m, "start_idx": indices_out[i], "stride": stride_list[i]} for i, m in enumerate(models_out)
                ]
            else:
                raise NotImplementedError("mode not implemented")
            
        print(f"OBJVERSE {self.stage} dataset is loaded :smile:")
        print("Current stage: [bold red]{}[/bold red]".format(self.stage))
        print("Current mode: [bold red]{}[/bold red]".format(self.mode))
        print("Current split: [bold red]{}[/bold red]".format(self.split))
        print("Dataset is repeated [bold cyan]{}[/bold cyan] times".format(self.repeat))
        print("Depth is used: {}".format(self.use_depth), "with camera angle: {}".format(45))
        print("Length of split: {}".format(len(self.split_lst) if self.stage == 'shape' else len(self.models)))
    
    def __getitem__(self, index):
        if self.split == 'train':
            if index == 100:
                print("*******************************************************************")
                print(f"fake index: {index}, real index: {self.select_index_list[index]}")
                print("*******************************************************************")
            index = self.select_index_list[index]
        if self.stage == 'shape':
            if self.mode == 'ae':
                return self.get_shape_ae(index)
            elif self.mode == 'diff':
                return self.get_shape_diff(index)
            else:
                raise NotImplementedError("mode not implemented")
        elif self.stage == 'deform':
            if self.mode == 'ae':
                # return self.get_deform_ae(index)
                # return self.get_deform_ae_tripoSG(index)
                if self.revert:
                    return self.get_deform_ae_tripoSG(index)
                else:
                    return self.get_deform_ae_tripoSG_sample(index)
            elif self.mode == 'diff':
                # return self.get_deform_diff(index)
                return self.get_deform_ae_tripoSG_sample(index)
            elif self.mode == 'aeMulti':
                # return self.get_deform_ae_tripoSG(index)
                return self.get_deform_ae_tripoSG_sample(index)
            elif self.mode == 'diffMulti':
                # return self.get_deform_diff_multi(index)
                return self.get_deform_ae_tripoSG_sample(index)
            else:
                raise NotImplementedError("mode not implemented")
    
    def __len__(self):
        if self.split == 'train':
            return len(self.select_index_list)
        if self.stage == 'shape':
            return sum(self.split_lst_pcl_len) * self.repeat 
        elif self.stage == 'deform':
            return len(self.models)
        else:
            raise NotImplementedError("stage not implemented")
    
    def random_sample_indexes(self, models, models_len):
        n_sample_pro_model = self.n_sample_pro_model
        interval_between_frames = self.interval_between_frames
        n_selected_frames = self.n_selected_frames

        # Initialize output lists
        models_out = []
        indexes_out = []

        # Loop over each model
        for idx, model in enumerate(models):
            # For each sample per model
            for n in range(n_sample_pro_model):
                # Initialize indices list for current sample
                indexes = []

                # Select n_selected_frames number of indices
                for i in range(n_selected_frames):
                    # If first index, randomly select from range
                    if i == 0:
                        indexes.append(np.random.randint(0, models_len[idx] - interval_between_frames))
                    else:
                        # For subsequent indices, select within interval_between_frames from the previous index
                        indexes.append(indexes[-1] + np.random.randint(0, interval_between_frames))
                    
                # Append the selected indices and corresponding model to output lists
                indexes_out.append(sorted(indexes))
                models_out.append(model)
        
        return models_out, indexes_out  
    
    def subdivide_into_sequences(self, models, models_len):
        """Subdivides model sequence into smaller sequences.

        Args:
            models (list): list of model names
            models_len (list): list of lengths of model sequences
        """
        length_sequence = self.length_sequence
        n_files_per_sequence = self.n_files_per_sequence
        offset_sequence = self.offset_sequence
        interval_between_sequence = self.interval_between_sequence

        # Remove files before offset
        models_len = [l - offset_sequence for l in models_len]

        # Reduce to maximum number of files that should be considered
        if n_files_per_sequence > 0:
            models_len = [min(n_files_per_sequence, l) for l in models_len]

        models_out = []
        start_idx = []
        stride_list = []
        if self.split == "train":
            if length_sequence > 10:
                avai_stride_list = [1,2]
            else:
                avai_stride_list = [1,2,3,4]
        else:
            avai_stride_list = [1,]
        for idx, model in enumerate(models):
            for s in avai_stride_list:
                for n in range(0, models_len[idx] - (length_sequence * s) + 1, interval_between_sequence):
                    models_out.append(model)
                    start_idx.append(n + offset_sequence)
                    stride_list.append(s)


        return models_out, start_idx, stride_list

    def get_shape_ae(self, index):
        pcl_path = self.split_lst_pcl_path[index]
        points_path = self.split_lst_points_path[index]
        points_near_path = self.split_lst_points_near_path[index]
        
        meta_info = {
            "pcl_file": pcl_path,
            "points_file": points_path,
            "points_nearsurface_file": points_near_path,
        }        
        
        pcl = np.load(pcl_path)
        points = np.load(points_path)
        points_nearsurface = np.load(points_near_path)
        
        surface = pcl['points']
        
        src = surface
        
        # Val and Test will do surface sampling also
        if self.surface_sampling:
            if self.split.find('train') == -1: # val and test
                ind = np.random.default_rng(seed=self.seed).choice(src.shape[0], self.pc_size, replace=False)
            ind = np.random.default_rng().choice(src.shape[0], self.pc_size, replace=False)
            src = src[ind]
            
        src = torch.from_numpy(src).float()
        vol_points = points['points']
        vol_label = np.unpackbits(points['occupancies'])

        near_points = points_nearsurface['points']
        near_label = np.unpackbits(points_nearsurface['occupancies'])
        
        if self.sampling: # val and test won't do sampling
            ind = np.random.default_rng().choice(vol_points.shape[0], self.num_samples, replace=False) # downsample to 1024
            vol_points = vol_points[ind]
            vol_label = vol_label[ind]

            ind = np.random.default_rng().choice(near_points.shape[0], self.num_samples, replace=False)# downsample to 1024
            near_points = near_points[ind]
            near_label = near_label[ind]
            
        vol_points = torch.from_numpy(vol_points).float()
        vol_label = torch.from_numpy(vol_label).float()

        if self.split.find('train') != -1: # train
            near_points = torch.from_numpy(near_points).float()
            near_label = torch.from_numpy(near_label).float()

            points = torch.cat([vol_points, near_points], dim=0) # uniform sampled points + surface and near surface points 2048
            labels = torch.cat([vol_label, near_label], dim=0)
        else: # val and test
            points = vol_points
            labels = vol_label
        if self.transform:
            src, points = self.transform(src, points)
        
        model_data = {}
        model_data["points"] = points
        model_data["labels"] = labels
        model_data["surface"] = src
        
        return model_data, meta_info

    def get_shape_diff(self, index):
        pcl_path = self.split_lst_pcl_path[index]
        points_path = self.split_lst_points_path[index]
        depth_path = self.split_lst_depth_path[index] if self.use_depth else ""
        
        meta_info = {
            "pcl_file": pcl_path,
            "points_file": points_path,
            "depth_file": depth_path,
            "index": pcl_path.split('/')[-1].split('.')[0],
            "model": pcl_path.split('/')[-2]
        }
        
        pcl = np.load(pcl_path)
        surface = pcl['points']
        
        points = np.load(points_path)
        occ = np.unpackbits(points['occupancies'])
        points = points['points']
        

        if self.split.find('train') == -1: # val and test
            rng = np.random.default_rng(seed=self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)
        else:
            rng = np.random.default_rng()
            
        if self.use_depth:
            depth = np.load(depth_path)
            depth = depth['points']
            src = depth
        else:
            src = surface
            
        surface_org = surface.copy()
        
        choice_inputs = np.random.choice(src.shape[0], self.n_inputs, replace=False)
        inputs = src[choice_inputs]
        
        if self.surface_sampling:
            ind = rng.choice(surface.shape[0], self.pc_size, replace=False)
            surface = surface[ind]
        
        noise = self.inputs_noise_std * np.random.randn(*inputs.shape)
        noise = noise.astype(np.float32)
        inputs = inputs + noise
        inputs = torch.from_numpy(inputs).float()
        surface = torch.from_numpy(surface).float()
        
        model_data = {}
        model_data["inputs"] = inputs
        model_data["surface"] = surface
        
        if not self.split == "train":
            surface_org = torch.from_numpy(surface_org).float()
            points = torch.from_numpy(points).float()
            occ = torch.from_numpy(occ).float()
            
            model_data["points"] = points
            model_data["labels"] = occ
            model_data["surface"] = surface_org
        
        return model_data, meta_info
    
    def get_deform_ae(self, index):
        # No evaluation implemented for this stage
        model = self.models[index]
        model_name = model["model"]
        model_indices = model["indices"]
        
        model_info = {}
        model_info["model"] = model_name
        model_info["indices"] = model_indices
        
        model_pcl_pathes = [os.path.join(self.dataset_path, model_name, "pcl_seq", f"{i:08d}.npz") for i in model_indices]
        model_near_pathes = [os.path.join(self.dataset_path, model_name, "points_near_seq", f"{i:08d}.npz") for i in model_indices]
        
        model_pcl_np = [np.load(model_pcl_path) for model_pcl_path in model_pcl_pathes]
        model_near_np = [np.load(model_near_path) for model_near_path in model_near_pathes]

        surface = np.array([model_pcl["points"] for model_pcl in model_pcl_np])
        near_points = np.array([model_near["points"] for model_near in model_near_np])

        ind = np.random.default_rng().choice(surface.shape[1], self.pc_size, replace=False)
        surface = surface[:,ind,:]

        surface_src = surface[0,:,:]
        surface_tgt = surface[1,:,:]

        surface_src = torch.from_numpy(surface_src)
        surface_tgt = torch.from_numpy(surface_tgt)

        ind = np.random.default_rng().choice(near_points.shape[1], self.num_samples, replace=False)# downsample to 1024
        near_points = near_points[:,ind,:]
        near_points_src = near_points[0,:,:]
        near_points_tgt = near_points[1,:,:]
        near_points_src = torch.from_numpy(near_points_src)
        near_points_tgt = torch.from_numpy(near_points_tgt)

        points_src = torch.cat([surface_src, near_points_src], dim=0)
        points_tgt = torch.cat([surface_tgt, near_points_tgt], dim=0)

        model_data = {}
        model_data["points_src"] = points_src.float()
        model_data["points_tgt"] = points_tgt.float()
        model_data["surface_src"] = surface_src.float()
        model_data["surface_tgt"] = surface_tgt.float()

        return model_data, model_info

    

    def filter_split_lst(self):
        print(f"filtering split_lst")
        print(f"before filtering: {len(self.split_lst)}")
        new_split_lst = []
        num_vertices = []
        for model_name in self.split_lst:
            if os.path.exists(os.path.join(self.dataset_path, model_name, "frame_0000.ply")):
                model_pcl_path = os.path.join(self.dataset_path, model_name, "frame_0000.ply")
                model_pcl = trimesh.load(model_pcl_path, process=False)
                if model_pcl.vertices.shape[0] > self.filter_num and model_pcl.vertices.shape[0] < 160000: # if the first frame has less than self.pc_size points, remove the model from the split_lst
                    new_split_lst.append(model_name)
                    num_vertices.append(model_pcl.vertices.shape[0])

            else:
                pass
        # print the percentile of the num_vertices
        if len(self.split_lst) > 0:
            print(f"percentile of the num_vertices: {np.percentile(num_vertices, [25, 50, 75, 90, 95, 99])}") # [  2918.     8168.5   22687.5   73502.9  159471.2  760212.91]
            print(f"after filtering: {len(new_split_lst)}")
        return new_split_lst

    def set_epoch(self, epoch):
        self.epoch = epoch
        num_models = len(self.models)
        start_index = ((epoch + 1) * 5000) % (num_models - 5001)
        print("*******************************************************************")
        print(f"start_index: {start_index}")
        print(f"num_models: {num_models}")
        print(f"epoch: {epoch}")
        print("*******************************************************************")
        self.select_index_list = list(range(start_index, start_index + 5000))
        
    
    def get_deform_ae_tripoSG(self, index):

        model = self.models[index]
        model_info = {}
        
        model_name = model["model"]
        start_idx = model["start_idx"]
        
        model_info["model"] = model_name
        model_info["start_idx"] = start_idx
        model_data = {}

       
        end_idx = start_idx + self.length_sequence - 1 # [0, ..., 16] => 17 frames


        sampled_idx = range(start_idx, end_idx + 1)

        # Sort the sampled indices and preappend 0 to the list
        sampled_idx = sorted(sampled_idx)
        sampled_idx = [start_idx] + sampled_idx
        # import pdb;pdb.set_trace()

        model_pcl_pathes = [os.path.join(self.dataset_path, model_name, f"frame_{i:04d}.ply") for i in sampled_idx]
        model_rgb_pathes = [os.path.join(self.dataset_path, model_name, f"front_frame{i}.png") for i in sampled_idx]
        
        model_pcl_np = [trimesh.load(model_pcl_path, process=False) for model_pcl_path in model_pcl_pathes]
        model_rgb_np = [imageio.imread(model_rgb_path) for model_rgb_path in model_rgb_pathes]

        faces = np.array([model_pcl.faces for model_pcl in model_pcl_np])
        
        surface = np.array([model_pcl.vertices for model_pcl in model_pcl_np]) 

        # normalize the surface points
        # calculate the max and min of the surface pints
        max_points = np.max(surface, axis=1)
        min_points = np.min(surface, axis=1)
        max_points = np.max(max_points, axis=0)
        min_points = np.min(min_points, axis=0)
        surface = (surface - min_points) / np.linalg.norm(max_points - min_points)





        if self.split == "test":
            vol_points_list = []
            vol_occ_list = []

            for mesh in model_pcl_np:
                if self.normalize_first_frame:
                    mesh.apply_translation(- loc)
                    mesh.apply_scale(1/scale)
                else:
                    mesh.apply_translation(- min_points)
                    mesh.apply_scale(1/np.linalg.norm(max_points - min_points))
                # TODO change the mesh data range
                boxsize = 1 + 0.1
                points_uniform = np.random.rand(100000, 3)
                if self.normalize_first_frame:
                    points_uniform = boxsize * (points_uniform - 0.5)
                else:
                    points_uniform = boxsize * (points_uniform) # should be (points_uniform - 0.5) after change all the stuff to -1, 1
                points = points_uniform
                vol_points_list.append(points)

                occupancies = check_mesh_contains(mesh, points)
                vol_occ_list.append(occupancies)
            vol_points = np.stack(vol_points_list, axis=0)
            vol_occ = np.stack(vol_occ_list, axis=0)
            # to tensor
            vol_points = torch.from_numpy(vol_points).float()
            vol_occ = torch.from_numpy(vol_occ).float()
            model_data["points"] = vol_points
            model_data["labels"] = vol_occ

            model_data["normalized_surface"] = torch.from_numpy(surface.copy()).float()
            model_data["normalized_faces"] = torch.from_numpy(faces.copy()).float()



        if surface.shape[1] < self.num_samples:
            print(f"surface.shape[1] < self.num_samples: {surface.shape[1]} < {self.num_samples}")
            print(f"model_name: {model_name}")
            return self.get_deform_ae_tripoSG((index + 3000) % len(self.models))
        ind = np.random.default_rng().choice(surface.shape[1], self.num_samples, replace=False)# downsample to 1024
        near_points = surface[:,ind,:]
        near_points_src = near_points[[0],:,:]
        near_points_tgt = near_points[1:,:,:]
        near_points_src = torch.from_numpy(near_points_src)
        near_points_tgt = torch.from_numpy(near_points_tgt)

        

        ind = np.random.default_rng().choice(surface.shape[1], self.pc_size, replace=False)
        surface = surface[:,ind,:]

        surface_src = surface[[0],:,:]
        surface_tgt = surface[1:,:,:]

        surface_src = torch.from_numpy(surface_src)
        surface_tgt = torch.from_numpy(surface_tgt)


        points_src = torch.cat([surface_src, near_points_src], dim=1)
        points_tgt = torch.cat([surface_tgt, near_points_tgt], dim=1)

        # model_data = {}
        model_data["points_src"] = points_src.float().clone()
        model_data["points_tgt"] = points_tgt.float().clone()
        model_data["surface_src"] = surface_src.float().clone()
        model_data["surface_tgt"] = surface_tgt.float().clone()

        model_data["rgb_path"] = model_rgb_pathes[0]
        model_data["ply_path"] = model_pcl_pathes[0]

        return model_data, model_info



    def get_bone_information(self, base_path):
        # base_path = base_path.replace("component1", "component1_skin")
        # bone_names = os.path.join(base_path, "bone_names.npy")
        # bone_parents = os.path.join(base_path, "bone_parents.npy")
        bone_names = None
        bone_parents = None
        bone_positions_sequence = os.path.join(base_path, "bone_positions_sequence.npy") # [T, B, 3]
        # bone_world_matrices_sequence = os.path.join(base_path, "bone_world_matrices_sequence.npy") # [T, B, 4, 4]
        bone_world_matrices_sequence = None
        bone_tail_positions_sequence = os.path.join(base_path, "bone_tail_positions_sequence.npy") # [T, B, 3]
        weights = os.path.join(base_path, "weights.npy") # [V, B]
        # weights = None
        # bone_names = np.load(bone_names)
        # bone_parents = np.load(bone_parents)
        bone_positions_sequence = np.load(bone_positions_sequence)
        # bone_world_matrices_sequence = np.load(bone_world_matrices_sequence)
        bone_tail_positions_sequence = np.load(bone_tail_positions_sequence)
        weights = np.load(weights)
        if weights.shape[1] > 64:
            weights = weights[:,:64]
        # import pdb;pdb.set_trace()
        return bone_names, bone_parents, bone_positions_sequence, bone_world_matrices_sequence, weights, bone_tail_positions_sequence

    def get_deform_ae_tripoSG_sample(self, index):
        # No evaluation implemented for this stage
        model = self.models[index]
        model_info = {}
        
        model_name = model["model"]
        start_idx = model["start_idx"]
        stride = model["stride"]

        model_info["model"] = model_name
        model_info["start_idx"] = start_idx
        model_info["stride"] = stride
        model_data = {}

       
        end_idx = start_idx + self.length_sequence * stride - 1 # [0, ..., 16] => 17 frames

        sampled_idx = range(start_idx, end_idx + 1, stride)

        # Sort the sampled indices and preappend 0 to the list
        sampled_idx = sorted(sampled_idx)
        sampled_idx = [start_idx] + sampled_idx
        # import pdb;pdb.set_trace()

        # model_pcl_pathes = [os.path.join(self.dataset_path, model_name, f"frame_{i:04d}.ply") for i in sampled_idx]
        model_rgb_pathes = [os.path.join(self.dataset_path, model_name, f"{i}.png") for i in sampled_idx]
        print("loading model_rgb_pathes: ", model_rgb_pathes)


        if self.mode == "diffMulti":
            images, masks = self.load_render(model_rgb_pathes)
            model_data["image"] = images
            model_data["mask"] = masks
        
        if not os.path.exists(os.path.join(self.dataset_path, model_name, "bone_names.npy")):
            # print(f"bone_names.npy not found for {model_name}")
            bone_positions_sequence=None
            bone_tail_positions_sequence=None
            weights = None
        else:
            bone_names, bone_parents, bone_positions_sequence, bone_world_matrices_sequence, weights, bone_tail_positions_sequence = self.get_bone_information(os.path.join(self.dataset_path, model_name))
            bone_positions_sequence = bone_positions_sequence[sampled_idx]
            bone_tail_positions_sequence = bone_tail_positions_sequence[sampled_idx]



        model_data["rgb_path"] = model_rgb_pathes[0]
        model_data["rgb_path_list"] = model_rgb_pathes
        model_data["model_name"] = model_name + "_" + str(start_idx) + "_" + str(stride)

        return model_data


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id

    # dataset = worker_info.dataset
    # split_size = dataset.num_records // worker_info.num_workers
    # # reset num_records to the true number to retain reliable length information
    # dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
    # current_id = np.random.choice(len(np.random.get_state()[1]), 1)
    # return np.random.seed(np.random.get_state()[1][current_id] + worker_id)

    return np.random.seed(np.random.get_state()[1][0] + worker_id)

class OBJVERSESModule(LightningDataModule):
    def __init__(
        self,
        cfg: str,
        log_dir: str,
        batch_size: int = 1,
        num_workers: int = 4,
        val_num_workers: int = 2,
        # train_data_list: str = None,
        # val_data_list: str = None,
        # cond_stage_key: str = "all",
        image_size: int = 224,
        mean: Union[List[float], Tuple[float]] = (0.485, 0.456, 0.406),
        std: Union[List[float], Tuple[float]] = (0.229, 0.224, 0.225),
        # pc_size: int = 2048,
        # pc_sharpedge_size: int = 2048,
        # sharpedge_label: bool = False,
        # return_normal: bool = False, 
        padding = True,
        padding_ratio_range=[1.15, 1.15],
        dataset_path=None,
    ):

        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_num_workers = val_num_workers

        # self.train_data_list = train_data_list
        # self.val_data_list = val_data_list
        
        # self.cond_stage_key = cond_stage_key
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.train_image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.image_size),
            transforms.Normalize(mean=self.mean, std=self.std)])
        self.val_image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.image_size),
            transforms.Normalize(mean=self.mean, std=self.std)])

        # self.pc_size = pc_size
        # self.pc_sharpedge_size = pc_sharpedge_size
        # self.sharpedge_label = sharpedge_label
        # self.return_normal = return_normal

        self.padding = padding
        self.padding_ratio_range = padding_ratio_range
        self.cfg = cfg
        self.log_dir = log_dir


        train_asl_params = {
            "split": "train",
            'cfg': self.cfg,
            'log_dir': self.log_dir,
            # "data_list": self.train_data_list,
            # "cond_stage_key": self.cond_stage_key,
            "image_transform": self.train_image_transform,
            # "pc_size": self.pc_size,
            # "pc_sharpedge_size": self.pc_sharpedge_size,
            # "sharpedge_label": self.sharpedge_label,
            # "return_normal": self.return_normal,
            "padding": self.padding,
            "padding_ratio_range": self.padding_ratio_range,
            "dataset_path": dataset_path,
        }
        self.train_dataset = OBJVERSES(**train_asl_params)

        val_asl_params = {
            "split": "test",
            'cfg': self.cfg,
            'log_dir': self.log_dir,
            # "data_list": self.train_data_list,
            # "cond_stage_key": self.cond_stage_key,
            "image_transform": self.val_image_transform,
            # "pc_size": self.pc_size,
            # "pc_sharpedge_size": self.pc_sharpedge_size,
            # "sharpedge_label": self.sharpedge_label,
            # "return_normal": self.return_normal,
            "padding": self.padding,
            "padding_ratio_range": self.padding_ratio_range,
            "dataset_path": dataset_path,
        }
        self.val_dataset = OBJVERSES(**val_asl_params)
        
    def train_dataloader(self):
        # asl_params = {
        #     "split": "train",
        #     'cfg': self.cfg,
        #     'log_dir': self.log_dir,
        #     # "data_list": self.train_data_list,
        #     # "cond_stage_key": self.cond_stage_key,
        #     "image_transform": self.train_image_transform,
        #     # "pc_size": self.pc_size,
        #     # "pc_sharpedge_size": self.pc_sharpedge_size,
        #     # "sharpedge_label": self.sharpedge_label,
        #     # "return_normal": self.return_normal,
        #     "padding": self.padding,
        #     "padding_ratio_range": self.padding_ratio_range
        # }
        # dataset = OBJVERSES(**asl_params)
        sampler = DistributedSampler(self.train_dataset)
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=sampler,
            # drop_last=True,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self):
        # asl_params = {
        #     "data_list": self.val_data_list,
        #     "cond_stage_key": self.cond_stage_key,
        #     "image_transform": self.val_image_transform,
        #     "pc_size": self.pc_size,
        #     "pc_sharpedge_size": self.pc_sharpedge_size,
        #     "sharpedge_label": self.sharpedge_label,
        #     "return_normal": self.return_normal, 
        #     "padding": self.padding,
        #     "padding_ratio_range": self.padding_ratio_range
        # }
        # asl_params = {
        #     "split": "test",
        #     'cfg': self.cfg,
        #     'log_dir': self.log_dir,
        #     # "data_list": self.train_data_list,
        #     # "cond_stage_key": self.cond_stage_key,
        #     "image_transform": self.val_image_transform,
        #     # "pc_size": self.pc_size,
        #     # "pc_sharpedge_size": self.pc_sharpedge_size,
        #     # "sharpedge_label": self.sharpedge_label,
        #     # "return_normal": self.return_normal,
        #     "padding": self.padding,
        #     "padding_ratio_range": self.padding_ratio_range
        # }
        # dataset = OBJVERSES(**asl_params)
        # sampler = DistributedSampler(dataset)
        return torch.utils.data.DataLoader(
            self.val_dataset,
            # batch_size=self.batch_size,
            batch_size=1,
            num_workers=self.val_num_workers,
            # sampler=sampler,
            pin_memory=True,
            # drop_last=True,
            worker_init_fn=worker_init_fn,
        )



def safe_load_obj(path: str, process: bool = False) -> trimesh.Trimesh:
    vertices = []
    faces = []

    with open(path, 'r', errors='ignore') as f:
        for line in f:
            if not line or line[0] == '#':
                continue
            parts = line.strip().split()
            if not parts:
                continue

            # vertex
            if parts[0] == 'v' and len(parts) >= 4:
                try:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except ValueError:
                    continue

            # face (supports n-gons, triangulate by fan)
            elif parts[0] == 'f' and len(parts) >= 4:
                # OBJ face tokens can be like: v, v/vt, v//vn, v/vt/vn
                def parse_v_idx(tok: str) -> int:
                    v_str = tok.split('/')[0]
                    return int(v_str)

                try:
                    # Convert to 0-based indices, handle negative indices per OBJ spec
                    raw = [parse_v_idx(t) for t in parts[1:]]
                except ValueError:
                    continue

                # Handle negative indices (relative to end of vertex list)
                num_v = len(vertices)
                idxs = []
                for vi in raw:
                    if vi > 0:
                        idxs.append(vi - 1)
                    else:
                        idxs.append(num_v + vi)  # vi is negative

                # Filter out faces with any invalid indices
                if any((i < 0 or i >= num_v) for i in idxs):
                    continue

                # Triangulate fan: (0, i, i+1)
                for i in range(1, len(idxs) - 1):
                    faces.append([idxs[0], idxs[i], idxs[i + 1]])

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    # # Build mesh
    # mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=process)

    # # Optional cleanup to be extra robust
    # mesh.remove_degenerate_faces()
    # mesh.remove_unreferenced_vertices()
    # mesh.remove_duplicate_faces()
    # mesh.remove_infinite_values()
    # mesh.fix_normals()

    return vertices, faces




