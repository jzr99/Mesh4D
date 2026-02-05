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



class VAEPL(pl.LightningModule):
    def __init__(
        self,
        *,
        first_stage_config,
        optimizer_cfg,
        ema_config=None,
        ckpt_path: Optional[str] = None,
        ignore_keys: Union[Tuple[str], List[str]] = (),
        torch_compile: bool = False,
        output_folder: Optional[str] = None,
        clip_loss_threshold: float = 0.01,
    ):
        super().__init__()
        if output_folder is not None:
            os.makedirs(output_folder, exist_ok=True)
            self.output_folder = output_folder
        else:
            self.output_folder = None

        self.model = instantiate_from_config(first_stage_config, device=None, dtype=None)

        self.optimizer_cfg = optimizer_cfg
        self.criterion = torch.nn.MSELoss()
        self.kl_weight = first_stage_config.kl_weight

        self.clip_loss_threshold = clip_loss_threshold

        self.ckpt_path = ckpt_path
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            # todo check here
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
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
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

    # @rank_zero_only
    # @torch.no_grad()
    # def on_train_batch_start(self, batch, batch_idx):
    #     # only for very first batch
    #     if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 \
    #         and batch_idx == 0 and self.ckpt_path is None:
    #         # set rescale weight to 1./std of encodings
    #         print("### USING STD-RESCALING ###")

    #         z_q = self.encode_first_stage(batch[self.first_stage_key])
    #         z = z_q.detach()

    #         del self.z_scale_factor
    #         self.register_buffer("z_scale_factor", 1. / z.flatten().std())
    #         print(f"setting self.z_scale_factor to {self.z_scale_factor}")

    #         print("### USING STD-RESCALING ###")

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
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            surface_src = batch['surface_src']
            surface_tgt = batch['surface_tgt']
            points_src = batch['points_src']
            points_tgt = batch['points_tgt']
            bone_positions_sequence = batch['bone_positions_sequence']
            points_src_normal = batch['points_src_normal']
            weights = batch['weights']
            outputs = self.model(surface_src, surface_tgt, points_src, bone_positions_sequence=bone_positions_sequence, weights=weights, points_src_normal=points_src_normal)
            loss_kl = outputs['kl']
            loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
            loss_def = self.criterion(outputs['deformed'], points_tgt)
            loss_kl = loss_kl * self.kl_weight
            loss = loss_def + loss_kl
        return loss, loss_def, loss_kl

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        loss, loss_def, loss_kl = self.forward(batch)
        loss_value = loss.item()
        if loss_value > self.clip_loss_threshold: # 0.002
            if self.output_folder is not None:
                large_loss_log_path = os.path.join(self.output_folder, 'large_loss_log.txt')
                # create the file if it doesn't exist
                if not os.path.exists(large_loss_log_path):
                    with open(large_loss_log_path, 'w') as f:
                        f.write(f"model: loss\n")
                with open(large_loss_log_path, 'a') as f:
                    # meta_info['model'] is a list, join them with comma
                    model_name = ', '.join(batch['model_name'])
                    f.write(f"{model_name}: {loss_value}\n")
            # Use a very small weight instead of zero to maintain gradient flow
            print("************************************************large loss************************************************")
            print(f"model_name: {model_name}, loss_value: {loss_value}")
            print("************************************************large loss************************************************")
            loss = loss * 1e-8
            loss_value = loss.item()
        split = 'train'
        loss_dict = {
            f"{split}/total_loss": loss.detach(),
            f"{split}/loss_def": loss_def.detach(),
            f"{split}/loss_kl": loss_kl.detach(),
            f"{split}/lr_abs": self.optimizers().param_groups[0]['lr'],
        }
        self.log_dict(loss_dict, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)

        return loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        loss, loss_def, loss_kl = self.forward(batch)
        split = 'val'
        loss_dict = {
            f"{split}/total_loss": loss.detach(),
            f"{split}/loss_def": loss_def.detach(),
            f"{split}/loss_kl": loss_kl.detach(),
            f"{split}/lr_abs": self.optimizers().param_groups[0]['lr'],
        }
        self.log_dict(loss_dict, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)

        return loss

    # @torch.no_grad()
    # def sample(self, batch, output_type='trimesh', **kwargs):
    #     self.cond_stage_model.disable_drop = True

    #     generator = torch.Generator().manual_seed(0)

    #     with self.ema_scope("Sample"):
    #         with torch.amp.autocast(device_type='cuda'):
    #             try:
    #                 self.pipeline.device = self.device
    #                 self.pipeline.dtype = self.dtype
    #                 print("### USING PIPELINE ###")
    #                 print(f'device: {self.device} dtype : {self.dtype}')
    #                 additional_params = {'output_type':output_type}

    #                 image = batch.get("image", None)
    #                 mask = batch.get('mask', None)
                    
    #                 outputs = self.pipeline(image=image, 
    #                                         mask=mask,
    #                                         generator=generator,
    #                                         **additional_params)

    #             except Exception as e:
    #                 import traceback
    #                 traceback.print_exc()
    #                 print(f"Unexpected {e=}, {type(e)=}")
    #                 with open("error.txt", "a") as f:
    #                     f.write(str(e))
    #                     f.write(traceback.format_exc())
    #                     f.write("\n")
    #                 outputs = [None]

    #     self.cond_stage_model.disable_drop = False
    #     return [outputs]

    # @torch.no_grad()
    # def test_overfit(self, batch, output_folder, output_type='trimesh', **kwargs):
    #     self.cond_stage_model.disable_drop = True

    #     generator = torch.Generator().manual_seed(0)

    #     with self.ema_scope("Sample"):
    #         with torch.amp.autocast(device_type='cuda'):
    #             try:
    #                 self.pipeline.device = self.device
    #                 self.pipeline.dtype = self.dtype
    #                 print("### USING PIPELINE ###")
    #                 print(f'device: {self.device} dtype : {self.dtype}')
    #                 # additional_params = {'output_type':output_type}

    #                 # image = batch.get("image", None)
    #                 # mask = batch.get('mask', None)
                    
    #                 # outputs = self.pipeline(image=image, 
    #                 #                         mask=mask,
    #                 #                         generator=generator,
    #                 #                         **additional_params)
    #                 self.pipeline(batch=batch, output_path=output_folder, generator=generator)

    #             except Exception as e:
    #                 import traceback
    #                 traceback.print_exc()
    #                 print(f"Unexpected {e=}, {type(e)=}")
    #                 with open("error.txt", "a") as f:
    #                     f.write(str(e))
    #                     f.write(traceback.format_exc())
    #                     f.write("\n")
    #                 outputs = [None]

    #     self.cond_stage_model.disable_drop = False
    #     # return [outputs]