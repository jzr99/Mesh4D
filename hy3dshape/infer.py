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
import sys
sys.path.insert(0, '../hy3dpaint')


from PIL import Image
import numpy as np
import pandas as pd
from hy3dshape.rembg import BackgroundRemover
# from hy3dshape.pipelines import instantiate_non_trainable_model
from hy3dshape.utils.misc import instantiate_from_config, instantiate_non_trainable_model
# from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

# model_path = 'tencent/Hunyuan3D-2.1'
# pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
import os

def save_tensor_as_gif(tensor_frames, save_path, duration=200):
    """
    Save a tensor of frames as a GIF.
    
    Args:
        tensor_frames (torch.Tensor): Tensor of shape [T, C, H, W] or [T, H, W] for masks
        save_path (str): Path to save the GIF
        duration (int): Duration of each frame in milliseconds
    """
    # Convert tensor to numpy and handle different tensor shapes
    if tensor_frames.dim() == 4:  # [T, C, H, W] for images
        # Denormalize if needed (assuming ImageNet normalization)
        if tensor_frames.min() < 0:  # Likely normalized
            mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(tensor_frames.device)
            std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(tensor_frames.device)
            tensor_frames = tensor_frames * std + mean
        tensor_frames = torch.clamp(tensor_frames, 0, 1)
        frames = (tensor_frames.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    elif tensor_frames.dim() == 3:  # [T, H, W] for masks
        frames = (tensor_frames.cpu().numpy() * 255).astype(np.uint8)
        # Convert grayscale to RGB for better visualization
        frames = np.stack([frames, frames, frames], axis=-1)
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor_frames.shape}")
    
    # Convert numpy arrays to PIL Images
    pil_frames = [Image.fromarray(frame) for frame in frames]
    
    # Save as GIF
    pil_frames[0].save(
        save_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0
    )


import torch
import yaml
from hy3dshape.utils import instantiate_from_config 
import pytorch_lightning as pl
import os
import trimesh

from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig


try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")                                      
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")


max_num_view = 6  # can be 6 to 9
resolution = 512  # can be 768 or 512
conf = Hunyuan3DPaintConfig(max_num_view, resolution)
conf.realesrgan_ckpt_path = "../hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
conf.multiview_cfg_path = "../hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
conf.custom_pipeline = "../hy3dpaint/hunyuanpaintpbr"
paint_pipeline = Hunyuan3DPaintPipeline(conf)

# shape
model_path = 'tencent/Hunyuan3D-2.1'
pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)

ckpt_cfg_path = './configs/infer.yaml'
# ckpt_path = '/scratch/shared/beegfs/zeren/Hunyuan3D-2.1/hy3dshape/output_folder/dit/overfitting_depth_16_token_4096_lr1e4_motion_2k_pretrain_bone_resume_5_batt/ckpt/ckpt-step=00004000.ckpt'
ckpt_path = '../ckpt/denoiser.ckpt'
config = yaml.safe_load(open(ckpt_cfg_path, 'r'))
model = instantiate_from_config(config['model']['params']['denoiser_cfg'])
sd = torch.load(ckpt_path)
sd = sd['state_dict']
sd = {k.replace('model.', ''):v for k,v in sd.items()}
msg = model.load_state_dict(sd, strict=False)
print(msg)
model = model.cuda().half()


cond_stage_model = instantiate_from_config(config['model']['params']['cond_stage_config'])
first_stage_model = instantiate_non_trainable_model(config['model']['params']['first_stage_config'])
cond_stage_model_2 = instantiate_non_trainable_model(config['model']['params']['cond_stage_config_2'])
image_processor = instantiate_from_config(config['model']['params']['image_processor_cfg'])
from hy3dshape.schedulers import FlowMatchEulerDiscreteScheduler
scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
z_scale_factor = config['model']['params']['z_scale_factor']


pipeline = instantiate_from_config(
    config['model']['params']['pipeline_cfg'],
    vae=first_stage_model,
    model=model,
    scheduler=scheduler, # self.sampler,
    conditioner=cond_stage_model,
    image_processor=image_processor,
    cond_stage_model_2=cond_stage_model_2,
    z_scale_factor = z_scale_factor,
)
cond_stage_model.disable_drop = True

generator = torch.Generator().manual_seed(0)

data: pl.LightningDataModule = instantiate_from_config(config['dataset'])
output_folder = 'output_folder/test/infer'
mean_metrics_list = []
for batch in data.val_dataloader():

    only_first_frame = True
    image_paths = batch['rgb_path_list'][1:]
    gen_mesh_list = []
    init_latents = None
    for image_path in image_paths:
        image_path = image_path[0]
        image = Image.open(image_path).convert("RGBA")
        if image.mode == 'RGB':
            rembg = BackgroundRemover()
            image = rembg(image)
        # preset_latent=None,
        # return_init_latent=False,
        mesh, init_latents = pipeline_shapegen(image=image, preset_latent=init_latents, return_init_latent=True, generator=generator)
        mesh = mesh[0]
        gen_mesh_list.append(mesh)
        if only_first_frame:
            break


    mesh = gen_mesh_list[0]
    os.makedirs(os.path.join(output_folder, batch['model_name'][0]), exist_ok=True)
    vis_dir_gt = os.path.join(output_folder, batch['model_name'][0], 'temp_raw_gen_mesh.glb')
    mesh.export(vis_dir_gt)


    output_mesh_path = os.path.join(output_folder, batch['model_name'][0], 'textured_gen_mesh.obj')
    output_mesh_path = paint_pipeline(
        mesh_path = vis_dir_gt, 
        image_path = image_paths[0][0],
        output_mesh_path = output_mesh_path
    )

    mesh = trimesh.load(output_mesh_path, process=False) # we should stop trimesh from processing the mesh!!!!!!!
    

    # load the data in batch to the device
    batch = {k: v.to('cuda') if hasattr(v, 'to') else v for k,v in batch.items()}

    with torch.amp.autocast(device_type='cuda'):
        pipeline.device = torch.device('cuda')
        pipeline.dtype = torch.float16
        print("### USING PIPELINE ###")
        # print(f'device: {model.device} dtype : {model.dtype}')
        if only_first_frame:
            mean_metrics = pipeline(batch=batch, output_path=output_folder, generator=generator, gen_mesh=mesh, gen_mesh_list=None, not_simplify=True, combine_glb=True)
        else:
            mean_metrics = pipeline(batch=batch, output_path=output_folder, generator=generator, gen_mesh=mesh, gen_mesh_list=gen_mesh_list, not_simplify=True, combine_glb=True) # here for gen_mesh I use the textured mesh, which is a simplified version of the original mesh
        mean_metrics_list.append(mean_metrics)
    vis_dir = os.path.join(output_folder, batch['model_name'][0])
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create subdirectories for images and masks
    image_gif_dir = os.path.join(vis_dir, "image")
    mask_gif_dir = os.path.join(vis_dir, "mask")
    os.makedirs(image_gif_dir, exist_ok=True)
    os.makedirs(mask_gif_dir, exist_ok=True)
    
    # Save images and masks as GIFs if they exist in the batch
    if 'image' in batch:
        # Extract just the filename part from model_name to avoid nested directories
        model_filename = os.path.basename(batch['model_name'][0])
        image_gif_path = os.path.join(image_gif_dir, f"{model_filename}_images.gif")
        save_tensor_as_gif(batch['image'][0], image_gif_path, duration=200)
        print(f"Saved image GIF: {image_gif_path}")
    
    if 'mask' in batch:
        # Extract just the filename part from model_name to avoid nested directories
        model_filename = os.path.basename(batch['model_name'][0])
        mask_gif_path = os.path.join(mask_gif_dir, f"{model_filename}_masks.gif")
        # Squeeze the channel dimension for masks if it exists
        mask_tensor = batch['mask'][0]
        if mask_tensor.dim() == 4 and mask_tensor.shape[1] == 1:
            mask_tensor = mask_tensor.squeeze(1)
        save_tensor_as_gif(mask_tensor, mask_gif_path, duration=200)
        print(f"Saved mask GIF: {mask_gif_path}")

    

    # clean cuda cache
    torch.cuda.empty_cache()
    # break

# Process and save metrics
if mean_metrics_list:
    print("\n" + "="*80)
    print("PROCESSING METRICS")
    print("="*80)
    
    # Identify all model names (e.g., 'gen', 'vae', 'gt', 'diffusion', 'gen_mesh_list')
    model_names = set()
    for entry in mean_metrics_list:
        for key in entry.keys():
            if key != 'name':
                model_names.add(key)
    
    model_names = sorted(list(model_names))
    
    # Process each model separately
    for model_name in model_names:
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name.upper()}")
        print(f"{'='*80}")
        
        # Collect all metrics for this model
        model_data = []
        for entry in mean_metrics_list:
            if model_name in entry:
                row = {'sequence_name': entry['name']}
                row.update(entry[model_name])
                model_data.append(row)
        
        if not model_data:
            print(f"No data found for model: {model_name}")
            continue
        
        # Create DataFrame
        df_model = pd.DataFrame(model_data)
        
        # Save all metrics for this model to CSV
        model_csv_path = os.path.join(output_folder, f'{model_name}_all_metrics.csv')
        df_model.to_csv(model_csv_path, index=False)
        print(f"\nAll {model_name} metrics saved to: {model_csv_path}")
        
        # Calculate averages (ignoring NaN values)
        avg_metrics = {}
        for column in df_model.columns:
            if column != 'sequence_name':
                try:
                    # Convert to float and calculate mean ignoring NaN
                    values = pd.to_numeric(df_model[column], errors='coerce')
                    avg_value = values.mean(skipna=True)
                    avg_metrics[column] = avg_value
                except:
                    # Skip columns that can't be converted to numeric
                    pass
        
        # Print average metrics for this model
        print(f"\nAverage metrics for {model_name}:")
        print("-" * 60)
        for metric_name, avg_value in sorted(avg_metrics.items()):
            if not np.isnan(avg_value):
                print(f"  {metric_name}: {avg_value:.6f}")
            else:
                print(f"  {metric_name}: NaN (all values were NaN)")
        
        # Save average metrics to CSV
        avg_csv_path = os.path.join(output_folder, f'{model_name}_average_metrics.csv')
        df_avg = pd.DataFrame([avg_metrics])
        df_avg.to_csv(avg_csv_path, index=False)
        print(f"\nAverage {model_name} metrics saved to: {avg_csv_path}")
    
    print("\n" + "="*80)
    print("METRICS PROCESSING COMPLETE")
    print("="*80 + "\n")

# pipeline_shapegen.model = model


# image = 'tools/mini_testset/images/015.png'

# image = Image.open(image_path).convert("RGBA")
# if image.mode == 'RGB':
#     rembg = BackgroundRemover()
#     image = rembg(image)

# mesh = pipeline_shapegen(image=image, guidance_scale=1.0)[0]
# mesh = pipeline_shapegen(image=image)[0]
# mesh.export('demo.glb')
