import os
from contextlib import contextmanager
from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities import rank_zero_only

from ...utils.ema import LitEma
from ...utils.misc import instantiate_from_config, instantiate_non_trainable_model

import trimesh



class Diffuser(pl.LightningModule):
    def __init__(
        self,
        *,
        first_stage_config,
        cond_stage_config,
        denoiser_cfg,
        scheduler_cfg,
        optimizer_cfg,
        cond_stage_config_2,
        # cond_stage_ckpt_2,
        pipeline_cfg=None,
        image_processor_cfg=None,
        lora_config=None,
        ema_config=None,
        first_stage_key: str = "surface",
        cond_stage_key: str = "image",
        cond_stage_key_2: str = "normalized_surface",
        scale_by_std: bool = False,
        z_scale_factor: float = 1.0,
        ckpt_path: Optional[str] = None,
        ignore_keys: Union[Tuple[str], List[str]] = (),
        torch_compile: bool = False,
    ):
        super().__init__()
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.cond_stage_key_2 = cond_stage_key_2

        # ========= init optimizer config ========= #
        self.optimizer_cfg = optimizer_cfg

        # ========= init diffusion scheduler ========= #
        self.scheduler_cfg = scheduler_cfg
        self.sampler = None
        if 'transport' in scheduler_cfg:
            self.transport = instantiate_from_config(scheduler_cfg.transport)
            self.sampler = instantiate_from_config(scheduler_cfg.sampler, transport=self.transport)
            self.sample_fn = self.sampler.sample_ode(**scheduler_cfg.sampler.ode_params)

        # ========= init the model ========= #
        self.denoiser_cfg = denoiser_cfg
        self.model = instantiate_from_config(denoiser_cfg, device=None, dtype=None)
        self.cond_stage_model = instantiate_from_config(cond_stage_config)

        

        self.ckpt_path = ckpt_path
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            # todo check here

        # ========= config lora model ========= #
        if lora_config is not None:
            from peft import LoraConfig, get_peft_model
            loraconfig = LoraConfig(
                r=lora_config.rank,
                lora_alpha=lora_config.rank,
                target_modules=lora_config.get('target_modules')
            )
            self.model = get_peft_model(self.model, loraconfig)

        # ========= config ema model ========= #
        self.ema_config = ema_config
        if self.ema_config is not None:
            if self.ema_config.ema_model == 'DSEma':
                # from michelangelo.models.modules.ema_deepspeed import DSEma
                from ..utils.ema_deepspeed import DSEma
                self.model_ema = DSEma(self.model, decay=self.ema_config.ema_decay)
            else:
                self.model_ema = LitEma(self.model, decay=self.ema_config.ema_decay)
            #do not initilize EMA weight from ckpt path, since I need to change moe layers
            if ckpt_path is not None:
                self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        # ========= init vae at last to prevent it is overridden by loaded ckpt ========= #
        self.first_stage_model = instantiate_non_trainable_model(first_stage_config)

        self.cond_stage_model_2 = instantiate_non_trainable_model(cond_stage_config_2)
        
        # if cond_stage_ckpt_2 is not None:
        #     self.cond_stage_model_2.load_state_dict(torch.load(cond_stage_ckpt_2, map_location="cpu")["state_dict"])

        self.scale_by_std = scale_by_std
        if scale_by_std:
            self.register_buffer("z_scale_factor", torch.tensor(z_scale_factor))
        else:
            self.z_scale_factor = z_scale_factor

        # ========= init pipeline for inference ========= #
        self.image_processor_cfg = image_processor_cfg
        self.image_processor = None
        if self.image_processor_cfg is not None:
            self.image_processor = instantiate_from_config(self.image_processor_cfg)
        self.pipeline_cfg = pipeline_cfg
        from ...schedulers import FlowMatchEulerDiscreteScheduler
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
        self.pipeline = instantiate_from_config(
            pipeline_cfg,
            vae=self.first_stage_model,
            model=self.model,
            scheduler=scheduler, # self.sampler,
            conditioner=self.cond_stage_model,
            image_processor=self.image_processor,
            cond_stage_model_2=self.cond_stage_model_2,
            z_scale_factor = self.z_scale_factor,
        )

        # ========= torch compile to accelerate ========= #
        self.torch_compile = torch_compile
        if self.torch_compile:
            torch.nn.Module.compile(self.model)
            torch.nn.Module.compile(self.first_stage_model)
            torch.nn.Module.compile(self.cond_stage_model)
            print(f'*' * 100)
            print(f'Compile model for acceleration')
            print(f'*' * 100)

    @contextmanager
    def ema_scope(self, context=None):
        if self.ema_config is not None and self.ema_config.get('ema_inference', False):
            self.model_ema.store(self.model)
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.ema_config is not None and self.ema_config.get('ema_inference', False):
                self.model_ema.restore(self.model)
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=()):
        ckpt = torch.load(path, map_location="cpu")
        if 'state_dict' not in ckpt:
            # deepspeed ckpt
            state_dict = {}
            for k in ckpt.keys():
                new_k = k.replace('_forward_module.', '')
                state_dict[new_k] = ckpt[k]
        else:
            state_dict = ckpt["state_dict"]

        keys = list(state_dict.keys())
        for k in keys:
            for ik in ignore_keys:
                if ik in k:
                    print("Deleting key {} from state_dict.".format(k))
                    del state_dict[k]

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_load_checkpoint(self, checkpoint):
        """
        The pt_model is trained separately, so we already have access to its
        checkpoint and load it separately with `self.set_pt_model`.

        However, the PL Trainer is strict about
        checkpoint loading (not configurable), so it expects the loaded state_dict
        to match exactly the keys in the model state_dict.

        So, when loading the checkpoint, before matching keys, we add all pt_model keys
        from self.state_dict() to the checkpoint state dict, so that they match
        """
        for key in self.state_dict().keys():
            if key.startswith("model_ema") and key not in checkpoint["state_dict"]:
                checkpoint["state_dict"][key] = self.state_dict()[key]

    def configure_optimizers(self) -> Tuple[List, List]:
        lr = self.learning_rate

        params_list = []
        trainable_parameters = list(self.model.parameters())
        params_list.append({'params': trainable_parameters, 'lr': lr})

        no_decay = ['bias', 'norm.weight', 'norm.bias', 'norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias']


        if self.optimizer_cfg.get('train_image_encoder', False):
            image_encoder_parameters = list(self.cond_stage_model.named_parameters())
            image_encoder_parameters_decay = [param for name, param in image_encoder_parameters if
                                              not any((no_decay_name in name) for no_decay_name in no_decay)]
            image_encoder_parameters_nodecay = [param for name, param in image_encoder_parameters if
                                                any((no_decay_name in name) for no_decay_name in no_decay)]
            # filter trainable params
            image_encoder_parameters_decay = [param for param in image_encoder_parameters_decay if
                                              param.requires_grad]
            image_encoder_parameters_nodecay = [param for param in image_encoder_parameters_nodecay if
                                                param.requires_grad]

            print(f"Image Encoder Params: {len(image_encoder_parameters_decay)} decay, ")
            print(f"Image Encoder Params: {len(image_encoder_parameters_nodecay)} nodecay, ")

            image_encoder_lr = self.optimizer_cfg['image_encoder_lr']
            image_encoder_lr_multiply = self.optimizer_cfg.get('image_encoder_lr_multiply', 1.0)
            image_encoder_lr = image_encoder_lr if image_encoder_lr is not None else lr * image_encoder_lr_multiply
            params_list.append(
                {'params': image_encoder_parameters_decay, 'lr': image_encoder_lr,
                 'weight_decay': 0.05})
            params_list.append(
                {'params': image_encoder_parameters_nodecay, 'lr': image_encoder_lr,
                 'weight_decay': 0.})

        optimizer = instantiate_from_config(self.optimizer_cfg.optimizer, params=params_list, lr=lr)
        if hasattr(self.optimizer_cfg, 'scheduler'):
            scheduler_func = instantiate_from_config(
                self.optimizer_cfg.scheduler,
                max_decay_steps=self.trainer.max_steps,
                lr_max=lr
            )
            scheduler = {
                "scheduler": lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_func.schedule),
                "interval": "step",
                "frequency": 1
            }
            schedulers = [scheduler]
        else:
            schedulers = []
        optimizers = [optimizer]

        return optimizers, schedulers

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 \
            and batch_idx == 0 and self.ckpt_path is None:
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")

            z_q = self.encode_first_stage(batch[self.first_stage_key])
            z = z_q.detach()

            del self.z_scale_factor
            self.register_buffer("z_scale_factor", 1. / z.flatten().std())
            print(f"setting self.z_scale_factor to {self.z_scale_factor}")

            print("### USING STD-RESCALING ###")

    def on_train_batch_end(self, *args, **kwargs):
        if self.ema_config is not None:
            self.model_ema(self.model)

    def on_train_epoch_start(self) -> None:
        pl.seed_everything(self.trainer.global_rank)
        # print the current epoch
        print(f"current epoch: {self.current_epoch}")
        train_dataset = self.trainer.datamodule.train_dataset
        train_dataset.set_epoch(self.current_epoch)
        print("finished setting epoch for dataset")
        # call set_epoch for train dataset


    def forward(self, batch):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): #float32 for text
            # contexts = self.cond_stage_model(image=batch.get('image'), text=batch.get('text'), mask=batch.get('mask'))
            video = batch.get('image')[:,1:]
            mask = batch.get('mask')[:,1:]
            B, T, C, H, W = video.shape
            video = video.reshape(B*T, C, H, W)
            mask = mask.reshape(B*T, 1, H, W)
            contexts = self.cond_stage_model(image=video, mask=mask) # ([T, 1370, 1024]) 
            # import pdb;pdb.set_trace()
            contexts['main'] = contexts['main'].reshape(B, T, contexts['main'].shape[-2], contexts['main'].shape[-1])

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                weights = batch['weights']


                # latents, latents_dist, canonical_points = self.first_stage_model.encode(batch['surface_src'], batch['surface_tgt'],return_dist=True, return_canonial=True)
                latents, posterior, canonical_points = self.first_stage_model.encode(batch['surface_src'], batch['surface_tgt'], bone_positions_sequence=batch['bone_positions_sequence'], sample_posterior=True, weights=weights,  inference_mode=True)
                latents = self.z_scale_factor * latents

                B, T, N, C = latents.shape




                # shape_input = torch.cat([batch['normalized_surface'][:,0], batch['normalized_vertex_normals'][:,0]], dim=-1)
                # concat 0 indicate sharpedge label
                shape_input = torch.cat([batch['surface_src'][:,0], torch.zeros_like(batch['surface_src'][:,0,:,-1:])], dim=-1)
                shape_latent_cond = self.cond_stage_model_2.encode(shape_input, sample_posterior=False)
                shape_latent_cond = shape_latent_cond.unsqueeze(1)
                shape_latent_cond = shape_latent_cond.repeat(1, T, 1, 1)
                # import pdb;pdb.set_trace()


                # latents = self.first_stage_model.encode(batch[self.first_stage_key], sample_posterior=True)
                # latents = self.z_scale_factor * latents

                # check vae encode and decode is ok? the answer is OK!
                if False:
                    latents = 1. / self.z_scale_factor * latents
                    # import pdb;pdb.set_trace()
                    normalized_surface = batch['normalized_surface'].repeat(1,T,1,1)
                    normalized_faces = batch['normalized_faces']
                    faces = normalized_faces[0,0]
                    deformed_verts = self.first_stage_model.decode(latents[[0]], normalized_surface[[0]])['deformed'].squeeze(0)
                    deformed_mesh_list = []
                    for j in range(deformed_verts.shape[0]):
                        defomed_mesh = trimesh.Trimesh(vertices= deformed_verts[j].cpu().numpy(), faces=faces.cpu().numpy(), process=False)
                        deformed_mesh_list.append(defomed_mesh)
                    vis_dir = f'./sanity_check/error_map'
                    os.makedirs(vis_dir, exist_ok=True)
                    # import pdb;pdb.set_trace()
                    # TODO: check here I use the length of colored_pcd is not same as the length of deformed_verts, need to fix this
                    for j in range(T):
                        defomed_colored_mesh = trimesh.Trimesh(vertices=deformed_verts[j].cpu().numpy(), faces=faces.cpu().numpy(), process=False)
                        defomed_colored_mesh.export(f'{vis_dir}/pred_{j:02d}.obj')
                    import pdb;pdb.set_trace()




                # print(latents.shape)

                # check vae encode and decode is ok? answer is ok!
                # import time
                # from hy3dshape.pipelines import export_to_trimesh
                # latents = 1. / self.z_scale_factor * latents
                # latents = self.first_stage_model(latents)
                # outputs = self.first_stage_model.latents2mesh(
                #     latents,
                #     bounds=1.01,
                #     mc_level=0.0,
                #     num_chunks=20000,
                #     octree_resolution=256,
                #     mc_algo='mc',
                #     enable_pbar=True
                # )
                # mesh = export_to_trimesh(outputs)
                # if isinstance(mesh, list):
                #     for midx, m in enumerate(mesh):
                #         m.export(f"check_{midx}_{time.time()}.glb")
                # else:
                #     mesh.export(f"check_{time.time()}.glb")
                
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # import pdb;pdb.set_trace()
            loss = self.transport.training_losses(self.model, latents, dict(contexts=contexts, shape_latent_cond=shape_latent_cond, canonical_points=canonical_points))["loss"].mean()
        return loss

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        loss = self.forward(batch)
        split = 'train'
        loss_dict = {
            f"{split}/simple": loss.detach(),
            f"{split}/total_loss": loss.detach(),
            f"{split}/lr_abs": self.optimizers().param_groups[0]['lr'],
        }
        self.log_dict(loss_dict, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)

        return loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        loss = self.forward(batch)
        split = 'val'
        loss_dict = {
            f"{split}/simple": loss.detach(),
            f"{split}/total_loss": loss.detach(),
            f"{split}/lr_abs": self.optimizers().param_groups[0]['lr'],
        }
        self.log_dict(loss_dict, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)

        return loss

    @torch.no_grad()
    def sample(self, batch, output_type='trimesh', **kwargs):
        self.cond_stage_model.disable_drop = True

        generator = torch.Generator().manual_seed(0)

        with self.ema_scope("Sample"):
            with torch.amp.autocast(device_type='cuda'):
                try:
                    self.pipeline.device = self.device
                    self.pipeline.dtype = self.dtype
                    print("### USING PIPELINE ###")
                    print(f'device: {self.device} dtype : {self.dtype}')
                    additional_params = {'output_type':output_type}

                    image = batch.get("image", None)
                    mask = batch.get('mask', None)
                    
                    outputs = self.pipeline(image=image, 
                                            mask=mask,
                                            generator=generator,
                                            **additional_params)

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"Unexpected {e=}, {type(e)=}")
                    with open("error.txt", "a") as f:
                        f.write(str(e))
                        f.write(traceback.format_exc())
                        f.write("\n")
                    outputs = [None]

        self.cond_stage_model.disable_drop = False
        return [outputs]

    @torch.no_grad()
    def test_overfit(self, batch, output_folder, output_type='trimesh', **kwargs):
        self.cond_stage_model.disable_drop = True

        generator = torch.Generator().manual_seed(0)

        with self.ema_scope("Sample"):
            with torch.amp.autocast(device_type='cuda'):
                try:
                    self.pipeline.device = self.device
                    self.pipeline.dtype = self.dtype
                    print("### USING PIPELINE ###")
                    print(f'device: {self.device} dtype : {self.dtype}')
                    # additional_params = {'output_type':output_type}

                    # image = batch.get("image", None)
                    # mask = batch.get('mask', None)
                    
                    # outputs = self.pipeline(image=image, 
                    #                         mask=mask,
                    #                         generator=generator,
                    #                         **additional_params)
                    self.pipeline(batch=batch, output_path=output_folder, generator=generator)

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"Unexpected {e=}, {type(e)=}")
                    with open("error.txt", "a") as f:
                        f.write(str(e))
                        f.write(traceback.format_exc())
                        f.write("\n")
                    outputs = [None]

        self.cond_stage_model.disable_drop = False
        # return [outputs]