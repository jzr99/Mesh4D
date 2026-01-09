# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import copy
import importlib
import inspect
import os
from typing import List, Optional, Union

import numpy as np
import torch
import trimesh
import yaml
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils.import_utils import is_accelerate_version, is_accelerate_available
from tqdm import tqdm

from .models.autoencoders import ShapeVAE
from .models.autoencoders import SurfaceExtractors
from .utils import logger, synchronize_timer, smart_load_model
from pycpd import RigidRegistration
# from hy3dshape.utils.simplify_mesh_utils import mesh_simplify_trimesh

import pymeshlab
from im2mesh.eval import MeshEvaluator

evaluator = MeshEvaluator(n_points=100000)


def mesh_simplify_trimesh(inputpath, outputpath, target_count=40000):
    # 先去除离散面
    ms = pymeshlab.MeshSet()
    if inputpath.endswith(".glb"):
        ms.load_new_mesh(inputpath, load_in_a_single_layer=True)
    else:
        ms.load_new_mesh(inputpath)
    ms.save_current_mesh(outputpath.replace(".glb", ".obj"), save_textures=False)
    # 调用减面函数
    courent = trimesh.load(outputpath.replace(".glb", ".obj"), force="mesh")
    face_num = courent.faces.shape[0]

    if face_num > target_count:
        courent = courent.simplify_quadric_decimation(target_count)
    return courent.export(outputpath)



def eval_oflow_all(
    pcl_tgt, points_tgt, occ_tgt, mesh_t_list, evaluator, corr_project_to_final_mesh, eval_corr=True, return_colored_pcd = False
):
    try:
        eval_dict_mean, eval_dict_t = {}, {}
        eval_dict_mesh = {}
        T = pcl_tgt.shape[0]
        # eval IOU and CD
        for t, mesh in enumerate(mesh_t_list):
            _eval_dict_mesh = evaluator.eval_mesh(mesh, pcl_tgt[t], None, points_tgt[t], occ_tgt[t])
            for k, v in _eval_dict_mesh.items():
                # ! Modify here 2021.10.5, skip the normal metrics, to avoid ignoring nan in other metrics
                if k.startswith("normal"):
                    continue
                if not np.isnan(v):
                    if k not in eval_dict_mesh.keys():
                        eval_dict_mesh[k] = [v]
                    else:
                        eval_dict_mesh[k].append(v)
                else:
                    # raise ValueError("Evaluator meets nan")
                    if k not in eval_dict_mesh.keys():
                        eval_dict_mesh[k] = [v]
                    else:
                        eval_dict_mesh[k].append(v)
                    print(f"Evaluator meets nan for {k}")
                    print(f"Evaluator meets nan for {k}")
                    print(f"Evaluator meets nan for {k}")
                    print(f"Evaluator meets nan for {k}")
                    print(f"Evaluator meets nan for {k}")
                    print(f"Evaluator meets nan for {k}")
                    print(f"Evaluator meets nan for {k}")
                    print(f"Evaluator meets nan for {k}")
                    print(f"Evaluator meets nan for {k}")
                    print(f"Evaluator meets nan for {k}")
                    print(f"Evaluator meets nan for {k}")
                    print(f"Evaluator meets nan for {k}")
        for k, v in eval_dict_mesh.items():
            mean_v = np.array(v).mean()
            eval_dict_mean["{}".format(k)] = mean_v
            for t in range(T):
                eval_dict_t["{}_t{}".format(k, t)] = v[t]
        if eval_corr:
            if not return_colored_pcd:
                # eval correspondence
                eval_dict_corr = evaluator.eval_correspondences_mesh(
                    mesh_t_list,
                    pcl_tgt,
                    project_to_final_mesh=corr_project_to_final_mesh,
                    return_colored_pcd = False
                )
            else:
                eval_dict_corr,_ = evaluator.eval_correspondences_mesh(
                    mesh_t_list,
                    pcl_tgt,
                    project_to_final_mesh=corr_project_to_final_mesh,
                    return_colored_pcd = True)
                
                colored_pcd = evaluator.vis_chamfer(
                    mesh_t_list,
                    pcl_tgt)
            
            
            corr_list = []
            for k, v in eval_dict_corr.items():
                t = int(k.split(" ")[1])
                eval_dict_t["corr_l2_t%d" % t] = v
                corr_list.append(v)
            eval_dict_mean["corr_l2"] = np.array(corr_list).mean()
        
        if return_colored_pcd:
            return eval_dict_mean, eval_dict_t, colored_pcd
        else:
            return eval_dict_mean, eval_dict_t
    except Exception as e:
        print(f"Error in eval_oflow_all: {e}")
        return None, None, None

def eval_color_pcd(batch, deformed_mesh_list, evaluator, output_dir, prefix):
    if batch.get('points') is not None:
        eval_dict_mean, eval_dict_t, colored_pcd = eval_oflow_all(
                                pcl_tgt=batch['normalized_surface_all'][:,1:].squeeze(0).detach().to(dtype=torch.float32, device='cpu').numpy(),
                                points_tgt=batch['points'][:,1:].squeeze(0).detach().to(dtype=torch.float32, device='cpu').numpy(),
                                occ_tgt=batch['labels'][:,1:].squeeze(0).detach().to(dtype=torch.float32, device='cpu').numpy(),
                                mesh_t_list=deformed_mesh_list,
                                evaluator=evaluator,
                                corr_project_to_final_mesh=False,
                                return_colored_pcd=True
                            )
        vis_dir = output_dir
        
        os.makedirs(vis_dir, exist_ok=True)
        for j in range(len(deformed_mesh_list)):
            defomed_colored_mesh = trimesh.Trimesh(vertices=deformed_mesh_list[j].vertices, faces=deformed_mesh_list[j].faces, vertex_colors=colored_pcd[j], process=False)
            # defomed_colored_mesh = trimesh.Trimesh(vertices=deformed_mesh_list[j].vertices, faces=deformed_mesh_list[j].faces, process=False)
            defomed_colored_mesh.export(f'{vis_dir}/{prefix}_{j:02d}.obj')
        # import pdb;pdb.set_trace()
        # wirte the metric in to a file
        with open(f'{vis_dir}/metric.txt', 'w') as f:
            for k, v in eval_dict_mean.items():
                f.write(f'{k}: {v}\n')
                print(f'prefix: {prefix}, {k}: {v}')
        with open(f'{vis_dir}/metric_t.txt', 'w') as f:
            for k, v in eval_dict_t.items():
                f.write(f'{k}: {v}\n')
                # print(f'prefix: {prefix}, {k}: {v}')
        # import pdb;pdb.set_trace()
        return eval_dict_mean, eval_dict_t, colored_pcd
    else:
        vis_dir = output_dir 
        os.makedirs(vis_dir, exist_ok=True)
        for j in range(len(deformed_mesh_list)):
            # defomed_colored_mesh = trimesh.Trimesh(vertices=deformed_mesh_list[j].vertices, faces=deformed_mesh_list[j].faces, vertex_colors=colored_pcd[j], process=False)
            defomed_colored_mesh = trimesh.Trimesh(vertices=deformed_mesh_list[j].vertices, faces=deformed_mesh_list[j].faces, process=False)
            defomed_colored_mesh.export(f'{vis_dir}/{prefix}_{j:02d}.obj')


def rewrite_obj_with_material(input_path, output_path):
    """
    Opens an OBJ file, skips the first three lines, 
    writes the rest to a new file, and adds:
        mtllib textured_gen_mesh.mtl
        usemtl Material
    to the top of the new file.
    """
    with open(input_path, 'r') as infile:
        lines = infile.readlines()

    # Take lines starting from the 4th line (index 3)
    content_from_4th = lines[3:]

    # Lines to add at the top
    header = [
        "mtllib textured_gen_mesh.mtl\n",
        "usemtl Material\n"
    ]

    # Write to the new file
    with open(output_path, 'w') as outfile:
        outfile.writelines(header + content_from_4th)

    print(f"✅ Wrote modified file to: {output_path}")

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@synchronize_timer('Export to trimesh')
def export_to_trimesh(mesh_output):
    if isinstance(mesh_output, list):
        outputs = []
        for mesh in mesh_output:
            if mesh is None:
                outputs.append(None)
            else:
                mesh.mesh_f = mesh.mesh_f[:, ::-1]
                mesh_output = trimesh.Trimesh(mesh.mesh_v, mesh.mesh_f)
                outputs.append(mesh_output)
        return outputs
    else:
        mesh_output.mesh_f = mesh_output.mesh_f[:, ::-1]
        mesh_output = trimesh.Trimesh(mesh_output.mesh_v, mesh_output.mesh_f)
        return mesh_output


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config, **kwargs):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    cls = get_obj_from_str(config["target"])
    params = config.get("params", dict())
    kwargs.update(params)
    instance = cls(**kwargs)
    return instance


class Hunyuan3DDiTPipeline:
    model_cpu_offload_seq = "conditioner->model->vae"
    _exclude_from_cpu_offload = []

    @classmethod
    @synchronize_timer('Hunyuan3DDiTPipeline Model Loading')
    def from_single_file(
        cls,
        ckpt_path,
        config_path,
        device='cuda',
        dtype=torch.float16,
        use_safetensors=None,
        **kwargs,
    ):
        # load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # load ckpt
        if use_safetensors:
            ckpt_path = ckpt_path.replace('.ckpt', '.safetensors')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model file {ckpt_path} not found")
        logger.info(f"Loading model from {ckpt_path}")

        if use_safetensors:
            # parse safetensors
            import safetensors.torch
            safetensors_ckpt = safetensors.torch.load_file(ckpt_path, device='cpu')
            ckpt = {}
            for key, value in safetensors_ckpt.items():
                model_name = key.split('.')[0]
                new_key = key[len(model_name) + 1:]
                if model_name not in ckpt:
                    ckpt[model_name] = {}
                ckpt[model_name][new_key] = value
        else:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        # load model later
        # model = instantiate_from_config(config['model'])
        # model.load_state_dict(ckpt['model'])
        raise NotImplementedError('Model loading is not implemented for video pipelines, the vae should be conditioner')
        vae = instantiate_from_config(config['vae'])
        vae.load_state_dict(ckpt['vae'], strict=False)
        conditioner = instantiate_from_config(config['conditioner'])
        if 'conditioner' in ckpt:
            conditioner.load_state_dict(ckpt['conditioner'])
        image_processor = instantiate_from_config(config['image_processor'])
        scheduler = instantiate_from_config(config['scheduler'])

        model_kwargs = dict(
            vae=vae,
            model=None,
            scheduler=scheduler,
            conditioner=conditioner,
            image_processor=image_processor,
            device=device,
            dtype=dtype,
        )
        model_kwargs.update(kwargs)

        return cls(
            **model_kwargs
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        config_path,
        device='cuda',
        dtype=torch.float16,
        use_safetensors=False,
        variant='fp16',
        subfolder='hunyuan3d-dit-v2-1',
        **kwargs,
    ):
        kwargs['from_pretrained_kwargs'] = dict(
            model_path=model_path,
            subfolder=subfolder,
            use_safetensors=use_safetensors,
            variant=variant,
            dtype=dtype,
            device=device,
        )
        _, ckpt_path = smart_load_model(
            model_path,
            subfolder=subfolder,
            use_safetensors=use_safetensors,
            variant=variant
        )
        return cls.from_single_file(
            ckpt_path,
            config_path,
            device=device,
            dtype=dtype,
            use_safetensors=use_safetensors,
            **kwargs
        )

    def __init__(
        self,
        vae,
        model,
        scheduler,
        conditioner,
        image_processor,
        cond_stage_model_2,
        z_scale_factor,
        device='cuda',
        dtype=torch.float16,
        latent_shape=(6,320,64),
        nocfg=False,
        **kwargs
    ):
        self.vae = vae
        self.model = model
        self.scheduler = scheduler
        self.conditioner = conditioner
        self.image_processor = image_processor
        self.kwargs = kwargs
        self.latent_shape = latent_shape
        self.cond_stage_model_2 = cond_stage_model_2
        self.z_scale_factor = z_scale_factor
        self.nocfg = nocfg
        self.to(device, dtype)

    def compile(self):
        self.vae = torch.compile(self.vae)
        self.model = torch.compile(self.model)
        self.conditioner = torch.compile(self.conditioner)

    def enable_flashvdm(
        self,
        enabled: bool = True,
        adaptive_kv_selection=True,
        topk_mode='mean',
        mc_algo='mc',
        replace_vae=True,
    ):
        if enabled:
            model_path = self.kwargs['from_pretrained_kwargs']['model_path']
            turbo_vae_mapping = {
                'Hunyuan3D-2': ('tencent/Hunyuan3D-2', 'hunyuan3d-vae-v2-0-turbo'),
                'Hunyuan3D-2mv': ('tencent/Hunyuan3D-2', 'hunyuan3d-vae-v2-0-turbo'),
                'Hunyuan3D-2mini': ('tencent/Hunyuan3D-2mini', 'hunyuan3d-vae-v2-mini-turbo'),
            }
            model_name = model_path.split('/')[-1]
            if replace_vae and model_name in turbo_vae_mapping:
                model_path, subfolder = turbo_vae_mapping[model_name]
                self.vae = ShapeVAE.from_pretrained(
                    model_path, subfolder=subfolder,
                    use_safetensors=self.kwargs['from_pretrained_kwargs']['use_safetensors'],
                    device=self.device,
                )
            self.vae.enable_flashvdm_decoder(
                enabled=enabled,
                adaptive_kv_selection=adaptive_kv_selection,
                topk_mode=topk_mode,
                mc_algo=mc_algo
            )
        else:
            model_path = self.kwargs['from_pretrained_kwargs']['model_path']
            vae_mapping = {
                'Hunyuan3D-2': ('tencent/Hunyuan3D-2', 'hunyuan3d-vae-v2-0'),
                'Hunyuan3D-2mv': ('tencent/Hunyuan3D-2', 'hunyuan3d-vae-v2-0'),
                'Hunyuan3D-2mini': ('tencent/Hunyuan3D-2mini', 'hunyuan3d-vae-v2-mini'),
            }
            model_name = model_path.split('/')[-1]
            if model_name in vae_mapping:
                model_path, subfolder = vae_mapping[model_name]
                self.vae = ShapeVAE.from_pretrained(model_path, subfolder=subfolder)
            self.vae.enable_flashvdm_decoder(enabled=False)

    def to(self, device=None, dtype=None):
        if dtype is not None:
            self.dtype = dtype
            self.vae.to(dtype=dtype)
            self.model.to(dtype=dtype)
            self.conditioner.to(dtype=dtype)
            self.cond_stage_model_2.to(dtype=dtype)
        if device is not None:
            self.device = torch.device(device)
            self.vae.to(device)
            self.model.to(device)
            self.conditioner.to(device)
            self.cond_stage_model_2.to(device)
            

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        [`~DiffusionPipeline.enable_sequential_cpu_offload`] the execution device can only be inferred from
        Accelerate's module hooks.
        """
        for name, model in self.components.items():
            if not isinstance(model, torch.nn.Module) or name in self._exclude_from_cpu_offload:
                continue

            if not hasattr(model, "_hf_hook"):
                return self.device
            for module in model.modules():
                if (
                    hasattr(module, "_hf_hook")
                    and hasattr(module._hf_hook, "execution_device")
                    and module._hf_hook.execution_device is not None
                ):
                    return torch.device(module._hf_hook.execution_device)
        return self.device

    def enable_model_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = "cuda"):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.

        Arguments:
            gpu_id (`int`, *optional*):
                The ID of the accelerator that shall be used in inference. If not specified, it will default to 0.
            device (`torch.Device` or `str`, *optional*, defaults to "cuda"):
                The PyTorch device type of the accelerator that shall be used in inference. If not specified, it will
                default to "cuda".
        """
        if self.model_cpu_offload_seq is None:
            raise ValueError(
                "Model CPU offload cannot be enabled because no `model_cpu_offload_seq` class attribute is set."
            )

        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        torch_device = torch.device(device)
        device_index = torch_device.index

        if gpu_id is not None and device_index is not None:
            raise ValueError(
                f"You have passed both `gpu_id`={gpu_id} and an index as part of the passed device `device`={device}"
                f"Cannot pass both. Please make sure to either not define `gpu_id` or not pass the index as part of "
                f"the device: `device`={torch_device.type}"
            )

        # _offload_gpu_id should be set to passed gpu_id (or id in passed `device`)
        # or default to previously set id or default to 0
        self._offload_gpu_id = gpu_id or torch_device.index or getattr(self, "_offload_gpu_id", 0)

        device_type = torch_device.type
        device = torch.device(f"{device_type}:{self._offload_gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu")
            device_mod = getattr(torch, self.device.type, None)
            if hasattr(device_mod, "empty_cache") and device_mod.is_available():
                device_mod.empty_cache()  
                # otherwise we don't see the memory savings (but they probably exist)

        all_model_components = {k: v for k, v in self.components.items() if isinstance(v, torch.nn.Module)}

        self._all_hooks = []
        hook = None
        for model_str in self.model_cpu_offload_seq.split("->"):
            model = all_model_components.pop(model_str, None)
            if not isinstance(model, torch.nn.Module):
                continue

            _, hook = cpu_offload_with_hook(model, device, prev_module_hook=hook)
            self._all_hooks.append(hook)

        # CPU offload models that are not in the seq chain unless they are explicitly excluded
        # these models will stay on CPU until maybe_free_model_hooks is called
        # some models cannot be in the seq chain because they are iteratively called, 
        # such as controlnet
        for name, model in all_model_components.items():
            if not isinstance(model, torch.nn.Module):
                continue

            if name in self._exclude_from_cpu_offload:
                model.to(device)
            else:
                _, hook = cpu_offload_with_hook(model, device)
                self._all_hooks.append(hook)

    def maybe_free_model_hooks(self):
        r"""
        Function that offloads all components, removes all model hooks that were added when using
        `enable_model_cpu_offload` and then applies them again. In case the model has not been offloaded this function
        is a no-op. Make sure to add this function to the end of the `__call__` function of your pipeline so that it
        functions correctly when applying enable_model_cpu_offload.
        """
        if not hasattr(self, "_all_hooks") or len(self._all_hooks) == 0:
            # `enable_model_cpu_offload` has not be called, so silently do nothing
            return

        for hook in self._all_hooks:
            # offload model and remove hook from model
            hook.offload()
            hook.remove()

        # make sure the model is in the same state as before calling it
        self.enable_model_cpu_offload()

    @synchronize_timer('Encode cond')
    def encode_cond(self, image, additional_cond_inputs, do_classifier_free_guidance, dual_guidance):
        bsz = image.shape[0]
        cond = self.conditioner(image=image, **additional_cond_inputs)

        if do_classifier_free_guidance:
            un_cond = self.conditioner.unconditional_embedding(bsz, **additional_cond_inputs)

            if dual_guidance:
                un_cond_drop_main = copy.deepcopy(un_cond)
                un_cond_drop_main['additional'] = cond['additional']

                def cat_recursive(a, b, c):
                    if isinstance(a, torch.Tensor):
                        return torch.cat([a, b, c], dim=0).to(self.dtype)
                    out = {}
                    for k in a.keys():
                        out[k] = cat_recursive(a[k], b[k], c[k])
                    return out

                cond = cat_recursive(cond, un_cond_drop_main, un_cond)
            else:
                def cat_recursive(a, b):
                    if isinstance(a, torch.Tensor):
                        return torch.cat([a, b], dim=0).to(self.dtype)
                    out = {}
                    for k in a.keys():
                        out[k] = cat_recursive(a[k], b[k])
                    return out

                cond = cat_recursive(cond, un_cond)
        return cond

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(self, batch_size, dtype, device, generator, latents=None):
        # shape = (batch_size, *self.vae.latent_shape)
        shape = (batch_size, *self.latent_shape)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * getattr(self.scheduler, 'init_noise_sigma', 1.0)
        return latents

    def prepare_image(self, image, mask=None) -> dict:
        if isinstance(image, torch.Tensor) and isinstance(mask, torch.Tensor):
            outputs = {
                'image': image,
                'mask': mask
            }
            return outputs
            
        if isinstance(image, str) and not os.path.exists(image):
            raise FileNotFoundError(f"Couldn't find image at path {image}")

        if not isinstance(image, list):
            image = [image]

        outputs = []
        for img in image:
            output = self.image_processor(img)
            outputs.append(output)

        cond_input = {k: [] for k in outputs[0].keys()}
        for output in outputs:
            for key, value in output.items():
                cond_input[key].append(value)
        for key, value in cond_input.items():
            if isinstance(value[0], torch.Tensor):
                cond_input[key] = torch.cat(value, dim=0)

        return cond_input

    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            timesteps (`torch.Tensor`):
                generate embedding vectors at these timesteps
            embedding_dim (`int`, *optional*, defaults to 512):
                dimension of the embeddings to generate
            dtype:
                data type of the generated embeddings

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    def set_surface_extractor(self, mc_algo):
        if mc_algo is None:
            return
        logger.info('The parameters `mc_algo` is deprecated, and will be removed in future versions.\n'
                    'Please use: \n'
                    'from hy3dshape.models.autoencoders import SurfaceExtractors\n'
                    'pipeline.vae.surface_extractor = SurfaceExtractors[mc_algo]() instead\n')
        if mc_algo not in SurfaceExtractors.keys():
            raise ValueError(f"Unknown mc_algo {mc_algo}")
        self.vae.surface_extractor = SurfaceExtractors[mc_algo]()

    @torch.no_grad()
    def __call__(
        self,
        image: Union[str, List[str], Image.Image] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        eta: float = 0.0,
        guidance_scale: float = 7.5,
        dual_guidance_scale: float = 10.5,
        dual_guidance: bool = True,
        generator=None,
        box_v=1.01,
        octree_resolution=384,
        mc_level=-1 / 512,
        num_chunks=8000,
        mc_algo=None,
        output_type: Optional[str] = "trimesh",
        enable_pbar=True,
        **kwargs,
    ) -> List[List[trimesh.Trimesh]]:
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        self.set_surface_extractor(mc_algo)

        device = self.device
        dtype = self.dtype
        do_classifier_free_guidance = guidance_scale >= 0 and \
                                      getattr(self.model, 'guidance_cond_proj_dim', None) is None
        dual_guidance = dual_guidance_scale >= 0 and dual_guidance

        if isinstance(image, torch.Tensor):
            pass
        else:
            cond_inputs = self.prepare_image(image)
            image = cond_inputs.pop('image')
        
        cond = self.encode_cond(
            image=image,
            additional_cond_inputs=cond_inputs,
            do_classifier_free_guidance=do_classifier_free_guidance,
            dual_guidance=False,
        )
        batch_size = image.shape[0]

        t_dtype = torch.long
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas)

        latents = self.prepare_latents(batch_size, dtype, device, generator)
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        guidance_cond = None
        if getattr(self.model, 'guidance_cond_proj_dim', None) is not None:
            logger.info('Using lcm guidance scale')
            guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(batch_size)
            guidance_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.model.guidance_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)
        with synchronize_timer('Diffusion Sampling'):
            for i, t in enumerate(tqdm(timesteps, disable=not enable_pbar, desc="Diffusion Sampling:", leave=False)):
                # expand the latents if we are doing classifier free guidance
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * (3 if dual_guidance else 2))
                else:
                    latent_model_input = latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                timestep_tensor = torch.tensor([t], dtype=t_dtype, device=device)
                timestep_tensor = timestep_tensor.expand(latent_model_input.shape[0])
                noise_pred = self.model(latent_model_input, timestep_tensor, cond, guidance_cond=guidance_cond)

                # no drop, drop clip, all drop
                if do_classifier_free_guidance:
                    if dual_guidance:
                        noise_pred_clip, noise_pred_dino, noise_pred_uncond = noise_pred.chunk(3)
                        noise_pred = (
                            noise_pred_uncond
                            + guidance_scale * (noise_pred_clip - noise_pred_dino)
                            + dual_guidance_scale * (noise_pred_dino - noise_pred_uncond)
                        )
                    else:
                        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                outputs = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
                latents = outputs.prev_sample

                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, outputs)

        return self._export(
            latents,
            output_type,
            box_v, mc_level, num_chunks, octree_resolution, mc_algo,
        )

    def _export(
        self,
        latents,
        output_type='trimesh',
        box_v=1.01,
        mc_level=0.0,
        num_chunks=20000,
        octree_resolution=256,
        mc_algo='mc',
        enable_pbar=True
    ):
        if not output_type == "latent":
            latents = 1. / self.vae.scale_factor * latents
            latents = self.vae(latents)
            outputs = self.vae.latents2mesh(
                latents,
                bounds=box_v,
                mc_level=mc_level,
                num_chunks=num_chunks,
                octree_resolution=octree_resolution,
                mc_algo=mc_algo,
                enable_pbar=enable_pbar,
            )
        else:
            outputs = latents

        if output_type == 'trimesh':
            outputs = export_to_trimesh(outputs)

        return outputs

def get_loc_scale_min0(mesh):
    # Determine bounding box

    bbox = mesh.bounding_box.bounds # bounds Bounding box with [min, max] coordinates
    # Compute location and scale
    # loc = (bbox[0] + bbox[1]) / 2 
    loc = bbox[0]
    scale = (bbox[1] - bbox[0]).max() / (1 - 0.1)

    return loc, scale

class Hunyuan3DDiTFlowMatchingPipeline(Hunyuan3DDiTPipeline):

    @torch.inference_mode()
    def __call__(
        self,
        batch: dict,
        output_path: str,
        # image: Union[str, List[str], Image.Image, dict, List[dict], torch.Tensor] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        eta: float = 0.0,
        guidance_scale: float = 5.0,
        generator=None,
        box_v=1.01,
        octree_resolution=384,
        mc_level=0.0,
        mc_algo=None,
        num_chunks=8000,
        output_type: Optional[str] = "trimesh",
        enable_pbar=True,
        mask = None,
        gen_mesh=None,
        gen_mesh_list=None,
        not_simplify=False,
        combine_glb=False,
        **kwargs,
    ) -> List[List[trimesh.Trimesh]]:
        mean_metrics = {'name': batch['model_name'][0]}
        converted_mesh = None
        registered_gen_mesh_list = None
        # import pdb;pdb.set_trace()
        if gen_mesh is not None:
            vis_dir_gen_mesh = os.path.join(output_path, batch['model_name'][0], 'gen_mesh')
            os.makedirs(vis_dir_gen_mesh, exist_ok=True)

            convert_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

            converted_vertices = convert_matrix @ gen_mesh.vertices.T
            converted_vertices = converted_vertices.T

            # # normalize the vertices
            # max_points = np.max(converted_vertices, axis=1)
            # min_points = np.min(converted_vertices, axis=1)
            # max_points = np.max(max_points, axis=0)
            # min_points = np.min(min_points, axis=0) # could be wrong here, min points should be 3 dimension intead of just 1 dimension
            # converted_vertices = (converted_vertices - min_points) / np.linalg.norm(max_points - min_points)
            converted_mesh = trimesh.Trimesh(vertices=converted_vertices, faces=gen_mesh.faces, process=False)
            loc, scale = get_loc_scale_min0(converted_mesh)
            converted_vertices = (converted_vertices - loc) / scale

            converted_mesh = trimesh.Trimesh(vertices=converted_vertices, faces=gen_mesh.faces, process=False)


            converted_mesh.export(f'{vis_dir_gen_mesh}/normalized_gen_mesh.obj')
            if not not_simplify:
                mesh_simplify_trimesh(f'{vis_dir_gen_mesh}/normalized_gen_mesh.obj', f'{vis_dir_gen_mesh}/simplified_normalized_gen_mesh.obj', target_count=10000)
                converted_mesh = trimesh.load(f'{vis_dir_gen_mesh}/simplified_normalized_gen_mesh.obj')

            print('before registration')

            # import pdb;pdb.set_trace()
            reg = RigidRegistration(X=batch['surface_src'][0,0,:,:3].cpu().numpy(), Y=np.array(converted_mesh.vertices), w=0.1, tolerance=0.01) # tolerance, max_iterations
            TY, (s_reg, R_reg, t_reg) = reg.register()
            # import pdb;pdb.set_trace()

            converted_mesh = trimesh.Trimesh(vertices=TY, faces=converted_mesh.faces, process=False)
            converted_mesh.export(f'{vis_dir_gen_mesh}/registered_gen_mesh.obj')

            print('after registration')


            if combine_glb:
                gen_mesh_mtl_path = os.path.join(output_path, batch['model_name'][0], 'registered_gen_mesh_with_mtl.obj')
                gen_mesh.vertices = TY
                gen_mesh.export(gen_mesh_mtl_path)
                gen_mesh_pbr_mtl_path = gen_mesh_mtl_path.replace('.obj', '_with_pbr_mtl.obj')
                rewrite_obj_with_material(gen_mesh_mtl_path, gen_mesh_pbr_mtl_path)
                from DifferentiableRenderer.mesh_utils import convert_obj_to_glb
                convert_obj_to_glb(gen_mesh_pbr_mtl_path, gen_mesh_mtl_path.replace('.obj', '.glb'))

            print('after converting glb')

            if gen_mesh_list is not None:
                vis_dir_gen_mesh_list = os.path.join(output_path, batch['model_name'][0], 'gen_mesh_list')
                os.makedirs(vis_dir_gen_mesh_list, exist_ok=True)
                registered_gen_mesh_list = [converted_mesh]
                max_num_vertex = converted_mesh.vertices.shape[0]
                converted_mesh.export(f'{vis_dir_gen_mesh_list}/registered_gen_mesh_00.obj')
                for j in range(1, len(gen_mesh_list)):
                    gen_mesh_list_i = gen_mesh_list[j]
                    convert_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
                    gen_mesh_list_i_converted_vertices = convert_matrix @ gen_mesh_list_i.vertices.T
                    gen_mesh_list_i_converted_vertices = gen_mesh_list_i_converted_vertices.T# could be wrong here, min points should be 3 dimension intead of just 1 dimension
                    # gen_mesh_list_i_converted_vertices = (gen_mesh_list_i_converted_vertices - min_points) / np.linalg.norm(max_points - min_points)
                    gen_mesh_list_i_converted_vertices = (gen_mesh_list_i_converted_vertices - loc) / scale
                    gen_mesh_list_i_converted_mesh = trimesh.Trimesh(vertices=gen_mesh_list_i_converted_vertices, faces=gen_mesh_list_i.faces, process=False)
                    gen_mesh_list_i_converted_mesh.export(f'{vis_dir_gen_mesh_list}/normalized_gen_mesh_{j:02d}.obj')
                    mesh_simplify_trimesh(f'{vis_dir_gen_mesh_list}/normalized_gen_mesh_{j:02d}.obj', f'{vis_dir_gen_mesh_list}/simplified_normalized_gen_mesh_{j:02d}.obj', target_count=10000)
                    gen_mesh_list_i_converted_mesh = trimesh.load(f'{vis_dir_gen_mesh_list}/simplified_normalized_gen_mesh_{j:02d}.obj')
                    gen_mesh_list_i_converted_mesh_vertices = reg.transform_point_cloud(Y=np.array(gen_mesh_list_i_converted_mesh.vertices))
                    gen_mesh_list_i_converted_mesh = trimesh.Trimesh(vertices=gen_mesh_list_i_converted_mesh_vertices, faces=gen_mesh_list_i_converted_mesh.faces, process=False)
                    gen_mesh_list_i_converted_mesh.export(f'{vis_dir_gen_mesh_list}/registered_gen_mesh_{j:02d}.obj')
                    registered_gen_mesh_list.append(gen_mesh_list_i_converted_mesh)
                    if gen_mesh_list_i_converted_mesh.vertices.shape[0] > max_num_vertex:
                        max_num_vertex = gen_mesh_list_i_converted_mesh.vertices.shape[0]
            
            # import pdb;pdb.set_trace()
            # iterate over the registered_gen_mesh_list and paading the vertices to the max_num_vertex
            if registered_gen_mesh_list is not None:
                for i in range(len(registered_gen_mesh_list)):
                    if registered_gen_mesh_list[i].vertices.shape[0] < max_num_vertex:
                        registered_gen_mesh_list_vertices = np.concatenate([registered_gen_mesh_list[i].vertices, np.zeros((max_num_vertex - registered_gen_mesh_list[i].vertices.shape[0], 3))], axis=0)
                        # registered_gen_mesh_list_vertices = np.concatenate([registered_gen_mesh_list[5].vertices, np.zeros((2, 3))], axis=0)
                        # registered_gen_mesh_list[5] = trimesh.Trimesh(vertices=registered_gen_mesh_list_vertices, faces=registered_gen_mesh_list[5].faces, process=False)
                        registered_gen_mesh_list[i] = trimesh.Trimesh(vertices=registered_gen_mesh_list_vertices, faces=registered_gen_mesh_list[i].faces, process=False)







            # import pdb;pdb.set_trace()

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        self.set_surface_extractor(mc_algo)

        device = self.device
        dtype = self.dtype
        do_classifier_free_guidance = guidance_scale >= 0 and not (
            hasattr(self.model, 'guidance_embed') and
            self.model.guidance_embed is True
        )
        if self.nocfg:
            do_classifier_free_guidance = False

        # print('image', type(image), 'mask', type(mask))
        # cond_inputs = self.prepare_image(image, mask)
        # image = cond_inputs.pop('image')
        video = batch['image'][:,1:]
        mask = batch['mask'][:,1:]
        B, T, C, H, W = video.shape
        video = video.reshape(B*T, C, H, W)
        mask = mask.reshape(B*T, 1, H, W)
        cond_inputs = {'mask': mask}
        cond = self.encode_cond(
            image=video,
            additional_cond_inputs=cond_inputs,
            do_classifier_free_guidance=do_classifier_free_guidance,
            dual_guidance=False,
        )
        if do_classifier_free_guidance:
            cond['main'] = cond['main'].reshape(2*B, T, cond['main'].shape[-2], cond['main'].shape[-1])
        else:
            cond['main'] = cond['main'].reshape(B, T, cond['main'].shape[-2], cond['main'].shape[-1])

        
        # get static cond
        if converted_mesh is not None:
            vertex = torch.from_numpy(converted_mesh.vertices).float().to(device).unsqueeze(0)
            vertex_normals = torch.from_numpy(converted_mesh.vertex_normals).float().to(device).unsqueeze(0)
            shape_input = torch.cat([vertex, vertex_normals, torch.zeros_like(vertex[:,:,:1])], dim=-1)
            shape_input = self.vae.encoder._sample_features(shape_input, num_tokens=2048)
            # shape_input = self.vae.model._sample_features(shape_input, num_tokens=2048)
            # torch.cat([batch['surface_src'][:,0], torch.zeros_like(batch['surface_src'][:,0,:,-1:])], dim=-1)
        else:
            shape_input = torch.cat([batch['surface_src'][:,0], torch.zeros_like(batch['surface_src'][:,0,:,-1:])], dim=-1)
        shape_latent_cond = self.cond_stage_model_2.encode(shape_input, sample_posterior=False)
        shape_latent_cond = shape_latent_cond.unsqueeze(1)
        shape_latent_cond = shape_latent_cond.repeat(1, T, 1, 1)

        batch_size = B


        latents_gt, latents_dist, canonical_points = self.vae.encode(batch['surface_src'], batch['surface_tgt'],inference_mode=True, bone_positions_sequence = batch['bone_positions_sequence'])

        # latents_gt, latents_dist, canonical_points = self.vae.encode(batch['surface_src'], batch['surface_tgt'],return_dist=True, return_canonial=True)
        # latents = self.z_scale_factor * latents
        if converted_mesh is not None:
            # canonical_points = self.vae.model._sample_features(vertex, num_tokens=256)
            canonical_points = self.vae.encoder._sample_features(vertex, num_tokens=256)

        B, T, N, C = latents_gt.shape

        # 5. Prepare timesteps
        # NOTE: this is slightly different from common usage, we start from 0.
        sigmas = np.linspace(0, 1, num_inference_steps) if sigmas is None else sigmas
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
        )
        latents = self.prepare_latents(batch_size, dtype, device, generator)

        guidance = None
        if hasattr(self.model, 'guidance_embed') and \
            self.model.guidance_embed is True:
            guidance = torch.tensor([guidance_scale] * batch_size, device=device, dtype=dtype)
            # logger.info(f'Using guidance embed with scale {guidance_scale}')

        with synchronize_timer('Diffusion Sampling'):
            for i, t in enumerate(tqdm(timesteps, disable=not enable_pbar, desc="Diffusion Sampling:")):
                # expand the latents if we are doing classifier free guidance
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * 2)
                    shape_latent_cond_input = shape_latent_cond.repeat(2, 1, 1, 1)
                    canonical_points_input = canonical_points.repeat(2, 1, 1)
                else:
                    latent_model_input = latents
                    shape_latent_cond_input = shape_latent_cond
                    canonical_points_input = canonical_points

                # NOTE: we assume model get timesteps ranged from 0 to 1
                timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)
                timestep = timestep / self.scheduler.config.num_train_timesteps
                # import pdb;pdb.set_trace()


                # TODO TODO TODO TODOTODO TODOTODO TODOTODO TODOTODO TODOTODO TODO here, we need to get 320 token number to align the point cloud and shape toaken
                noise_pred = self.model(latent_model_input, timestep, cond, guidance=guidance, shape_latent_cond=shape_latent_cond_input, canonical_points=canonical_points_input)

                if do_classifier_free_guidance:
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                outputs = self.scheduler.step(noise_pred, t, latents)
                latents = outputs.prev_sample

                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, outputs)

        latents = 1. / self.z_scale_factor * latents
        # import pdb;pdb.set_trace()

        normalized_surface_all = batch['normalized_surface_all'].repeat(1,1,1,1) # (B, T, N, 3)
        normalized_surface = batch['normalized_surface'].repeat(1,T,1,1)
        normalized_faces = batch['normalized_faces']
        if 'valid_surface_number' in batch.keys():
            valid_surface_number = batch['valid_surface_number'][0].int()
            normalized_surface_all = normalized_surface_all[:,:,0:valid_surface_number]
            normalized_surface = normalized_surface[:,:,0:valid_surface_number]
            # normalized_surface = normalized_surface[[0]]
        if 'valid_faces_number' in batch.keys():
            valid_faces_number = batch['valid_faces_number'][0].int()
            normalized_faces = normalized_faces[:,:,0:valid_faces_number]

        faces = normalized_faces[0,0]
        deformed_verts_vae = self.vae.decode(latents_gt[[0]], normalized_surface[[0]])['deformed'].squeeze(0)
        deformed_verts = self.vae.decode(latents[[0]], normalized_surface[[0]])['deformed'].squeeze(0)
        if converted_mesh is not None:
            deformed_verts_gen = self.vae.decode(latents[[0]], vertex)['deformed'].squeeze(0)
            vis_dir_gen = os.path.join(output_path, batch['model_name'][0], 'gen')
            os.makedirs(vis_dir_gen, exist_ok=True)
            deformed_mesh_list = []
            for j in range(T):
                defomed_colored_mesh = trimesh.Trimesh(vertices=deformed_verts_gen[j].to(torch.float32).cpu().numpy(), faces=converted_mesh.faces, process=False)
                deformed_mesh_list.append(defomed_colored_mesh)
                # defomed_colored_mesh.export(f'{vis_dir_gen}/pred_{j:02d}_gen.obj')
            ret = eval_color_pcd(batch, deformed_mesh_list, evaluator, vis_dir_gen, 'gen')
            mean_metrics.update({'gen': ret[0]})
            if combine_glb:
                obj_paths = []
                for j, deformed_mesh_list in enumerate(deformed_mesh_list):
                    obj_paths.append(f'{vis_dir_gen}/gen_{j:02d}.obj')
                obj_paths[0] = os.path.join(output_path, batch['model_name'][0], 'registered_gen_mesh_with_mtl.obj')
                print(obj_paths)
                from .combine_glb_anim import combine_glb_objs
                combine_glb_objs(obj_paths, os.path.join(output_path, batch['model_name'][0], 'combined_gen_mtl.glb'), fps=24, frame_duration=0, transition_frames=2)

            # if combine_glb:
            #     first_obj_path = os.path.join(vis_dir_gen, 'gen_00.obj')
            #     from DifferentiableRenderer.mesh_utils import convert_obj_to_glb
            #     convert_obj_to_glb(first_obj_path, first_obj_path.replace('.obj', '.glb'))
        else:
            vis_dir = os.path.join(output_path, batch['model_name'][0], 'diffusion')
            os.makedirs(vis_dir, exist_ok=True)
            deformed_mesh_list = []
            for j in range(T):
                defomed_colored_mesh = trimesh.Trimesh(vertices=deformed_verts[j].to(torch.float32).cpu().numpy(), faces=faces.to(torch.int32).cpu().numpy(), process=False)
                # defomed_colored_mesh.export(f'{vis_dir}/pred_{j:02d}_old.obj')
                deformed_mesh_list.append(defomed_colored_mesh)
            ret = eval_color_pcd(batch, deformed_mesh_list, evaluator, vis_dir, 'diffusion')
            mean_metrics.update({'diffusion': ret[0]})
            # import pdb;pdb.set_trace()
    
        vis_dir_vae = os.path.join(output_path, batch['model_name'][0], 'vae')
        os.makedirs(vis_dir_vae, exist_ok=True)
        deformed_mesh_list = []
        for j in range(T):
            defomed_colored_mesh = trimesh.Trimesh(vertices=deformed_verts_vae[j].to(torch.float32).cpu().numpy(), faces=faces.to(torch.int32).cpu().numpy(), process=False)
            deformed_mesh_list.append(defomed_colored_mesh)
        ret =eval_color_pcd(batch, deformed_mesh_list, evaluator, vis_dir_vae, 'vae')
        mean_metrics.update({'vae': ret[0]})
            # defomed_colored_mesh.export(f'{vis_dir_vae}/pred_{j:02d}_vae.obj')
        
        vis_dir_gt = os.path.join(output_path, batch['model_name'][0], 'gt')
        os.makedirs(vis_dir_gt, exist_ok=True)
        deformed_mesh_list = []
        for j in range(T):
            # import pdb;pdb.set_trace()
            # TODO check why the chamfer distance for GT is large, hypo: the number of points is different, we should filter out the valid points
            # checked it is because we sample the point on the predicted mesh
            defomed_colored_mesh = trimesh.Trimesh(vertices=normalized_surface_all[0,j+1,:,:3].to(torch.float32).cpu().numpy(), faces=faces.to(torch.int32).cpu().numpy(), process=False)
            deformed_mesh_list.append(defomed_colored_mesh)
            # defomed_colored_mesh.export(f'{vis_dir_gt}/pred_{j:02d}_gt.obj')
        # import pdb;pdb.set_trace()
        ret = eval_color_pcd(batch, deformed_mesh_list, evaluator, vis_dir_gt, 'gt')
        mean_metrics.update({'gt': ret[0]})

        if registered_gen_mesh_list is not None:
            vis_dir_gen_mesh_list = os.path.join(output_path, batch['model_name'][0], 'gen_mesh_list')
            ret = eval_color_pcd(batch, registered_gen_mesh_list, evaluator, vis_dir_gen_mesh_list, 'gen_mesh_list')
            mean_metrics.update({'gen_mesh_list': ret[0]})
        return mean_metrics


        # return deformed_verts
        
        
        
        
        
        # return self._export(
        #     latents,
        #     output_type,
        #     box_v, mc_level, num_chunks, octree_resolution, mc_algo,
        #     enable_pbar=enable_pbar,
        # )
