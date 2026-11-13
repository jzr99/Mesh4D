#!/usr/bin/env python3
"""Run the 16-frame Mesh4D R3 denoiser and VAE on RGBA video frames."""
import argparse
import gc
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.append(str(REPO))
from inference_utils import (checkpoint_model, input_sequences, load_batch, load_config,
                             reference_mesh_path, reference_surface, validate_images, write_json)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input-dir', type=Path, help='One sequence of segmented RGBA PNGs.')
    source.add_argument('--benchmark-root', type=Path, help='Use the manifest\'s 48-clip sixteen-frame split.')
    parser.add_argument('--input-root', type=Path, help='Override <benchmark-root>/input_images.')
    parser.add_argument('--manifest', type=Path, help='Override <benchmark-root>/manifest.json.')
    parser.add_argument('--clip', action='append', help='Select a benchmark clip_path; repeat for a subset.')
    parser.add_argument('--reference-geometry-root', type=Path,
                        help='Evaluation only: root of original per-frame PLYs for research-style GT registration.')
    parser.add_argument('--image-pattern', help='Custom sequence filenames, e.g. "{frame:04d}.png".')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--config', type=Path, default=HERE / 'configs/infer_16.yaml')
    parser.add_argument('--vae-checkpoint', type=Path,
                        default=REPO / 'ckpt/16frames/vae_bone_r2_16_00001500.ckpt')
    parser.add_argument('--denoiser-checkpoint', type=Path,
                        default=REPO / 'ckpt/16frames/diffuse_r3_ckpt-step=00003000.ckpt')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'output_folder/test/infer_16')
    parser.add_argument('--initial-mesh', type=Path,
                        help='Use an existing canonical mesh for one clip; bypass shape generation and painting.')
    parser.add_argument('--skip-texture', action='store_true', help='Generate untextured geometry for a faster check.')
    parser.add_argument('--num-inference-steps', type=int, default=50)
    parser.add_argument('--guidance-scale', type=float, default=5.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', action='store_true', help='Resume the same saved run settings and completed stages.')
    parser.add_argument('--dry-run', action='store_true', help='Validate images, paths and config without CUDA or model downloads.')
    parser.add_argument('--validate-weights', action='store_true',
                        help='Also strictly validate checkpoint tensor names/shapes on CPU; useful with --dry-run.')
    return parser, parser.parse_args(argv)


def read_mesh(path):
    import numpy as np
    import trimesh
    mesh = trimesh.load(str(path), force='mesh', process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) < 256 or not len(mesh.faces):
        raise ValueError(f'Expected a triangle mesh with at least 256 vertices: {path}')
    if not np.isfinite(mesh.vertices).all():
        raise ValueError(f'Nonfinite mesh vertices: {path}')
    return mesh


def release_cuda():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def prepare_meshes(sequences, args):
    import torch
    from PIL import Image

    needed = [s for s in sequences if not (args.output_dir / s.name / 'mesh_ready.json').is_file()]
    if not needed:
        return
    if args.initial_mesh:
        directory = args.output_dir / needed[0].name
        read_mesh(args.initial_mesh).export(directory / 'textured_gen_mesh.obj')
        write_json(directory / 'mesh_ready.json', {'source': str(args.initial_mesh.resolve())})
        return

    raw_needed = [s for s in needed if not (args.output_dir / s.name / 'shape_ready.json').is_file()]
    if raw_needed:
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2.1')
        for sequence in raw_needed:
            directory = args.output_dir / sequence.name
            torch.manual_seed(args.seed)
            with Image.open(sequence.paths[0]) as image:
                mesh = shape_pipeline(image=image.convert('RGBA'),
                                      generator=torch.Generator().manual_seed(args.seed))[0]
            mesh.export(directory / 'temp_raw_gen_mesh.glb')
            write_json(directory / 'shape_ready.json', {'source_image': str(sequence.paths[0])})
        del shape_pipeline
        release_cuda()

    if args.skip_texture:
        for sequence in needed:
            directory = args.output_dir / sequence.name
            read_mesh(directory / 'temp_raw_gen_mesh.glb').export(directory / 'textured_gen_mesh.obj')
            write_json(directory / 'mesh_ready.json', {'textured': False})
        return

    # This import must precede basicsr/realesrgan in the paint pipeline.
    from torchvision_fix import apply_fix
    apply_fix()
    sys.path.insert(0, str(REPO / 'hy3dpaint'))
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline
    config = Hunyuan3DPaintConfig(6, 512)
    config.realesrgan_ckpt_path = str(REPO / 'hy3dpaint/ckpt/RealESRGAN_x4plus.pth')
    config.multiview_cfg_path = str(REPO / 'hy3dpaint/cfgs/hunyuan-paint-pbr.yaml')
    config.custom_pipeline = str(REPO / 'hy3dpaint/hunyuanpaintpbr')
    paint_pipeline = Hunyuan3DPaintPipeline(config)
    for sequence in needed:
        directory = args.output_dir / sequence.name
        torch.manual_seed(args.seed)
        output = paint_pipeline(mesh_path=str(directory / 'temp_raw_gen_mesh.glb'),
                                image_path=str(sequence.paths[0]),
                                output_mesh_path=str(directory / 'textured_gen_mesh.obj'))
        read_mesh(output)
        write_json(directory / 'mesh_ready.json', {'textured': True})
    del paint_pipeline
    release_cuda()


def run_motion(sequences, config, args):
    import numpy as np
    import torch
    from hy3dshape.schedulers import FlowMatchEulerDiscreteScheduler
    from hy3dshape.utils.misc import instantiate_from_config, instantiate_non_trainable_model
    from hy3dshape.combine_glb_anim import combine_glb_objs

    params = config['model']
    denoiser = checkpoint_model(params['denoiser_cfg'], args.denoiser_checkpoint).to('cuda', torch.float16)
    vae = checkpoint_model(params['first_stage_config'], args.vae_checkpoint).to('cuda', torch.float16)
    conditioner = instantiate_non_trainable_model(params['cond_stage_config'])
    conditioner.disable_drop = True
    shape_vae = instantiate_non_trainable_model(params['cond_stage_config_2'])
    pipeline = instantiate_from_config(
        params['pipeline_cfg'], model=denoiser, vae=vae, conditioner=conditioner,
        cond_stage_model_2=shape_vae, z_scale_factor=params['z_scale_factor'],
        image_processor=instantiate_from_config(params['image_processor_cfg']),
        scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000),
    )
    for sequence in sequences:
        directory = args.output_dir / sequence.name
        print(f'Inferring 16 frames: {sequence.name}', flush=True)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        batch = load_batch(sequence, config['image_size'], config['padding_ratio'])
        batch = {key: value.to('cuda') if hasattr(value, 'to') else value for key, value in batch.items()}
        mesh = read_mesh(directory / 'textured_gen_mesh.obj')
        alignment_points = (reference_surface(reference_mesh_path(sequence, args.reference_geometry_root), args.seed)
                            if args.reference_geometry_root else None)
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
            pipeline(batch=batch, output_path=str(args.output_dir), gen_mesh=mesh,
                     num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
                     generator=torch.Generator().manual_seed(args.seed),
                     not_simplify=True, combine_glb=False, alignment_points=alignment_points)

        # Preserve the canonical mesh's texture and the research coordinate transform.
        # The original first animation pose is the canonical mesh; the other 15
        # poses are gen_01.obj ... gen_15.obj at timeline frames 3, 5, ... 31.
        registered = read_mesh(directory / 'gen_mesh/registered_gen_mesh.obj')
        if len(mesh.vertices) != len(registered.vertices):
            raise ValueError('Canonical mesh topology changed during inference.')
        mesh.vertices = registered.vertices
        base = directory / 'registered_gen_mesh_with_mtl.obj'
        mesh.export(base)
        poses = [directory / 'gen' / f'gen_{t:02d}.obj' for t in range(1, 16)]
        if not all(path.is_file() for path in poses):
            raise RuntimeError(f'Missing predicted geometry for {sequence.name}')
        output = directory / 'combined_gen_mtl.glb'
        combine_glb_objs([str(base), *(str(p) for p in poses)], str(output),
                         fps=24, frame_duration=0, transition_frames=2)
        write_json(directory / 'inference.json', dict(
            sequence.record(), frames=16, animation_frames=list(range(1, 32, 2)), fps=24,
            first_pose='canonical mesh', alignment=('ground-truth first-pose CPD registration'
                if args.reference_geometry_root else 'generated mesh bounding box; no ground-truth registration'),
            glb=output.name, complete=True))
        del batch, mesh, registered
        release_cuda()


def main(argv=None):
    parser, args = parse_args(argv)
    try:
        if args.num_inference_steps < 1 or not 0 <= args.guidance_scale < float('inf'):
            raise ValueError('Use positive inference steps and finite, nonnegative guidance scale.')
        if not 0 <= args.seed < 2**32:
            raise ValueError('--seed must be in [0, 2**32).')
        if args.benchmark_root and (args.image_pattern or args.start_frame != 0 or args.stride != 1):
            raise ValueError('Benchmark frame selection comes from its manifest; do not override it.')
        config = load_config(args.config)
        sequences = input_sequences(args)
        validate_images(sequences)
        if args.reference_geometry_root:
            if not args.benchmark_root:
                raise ValueError('--reference-geometry-root is only for benchmark evaluation.')
            for sequence in sequences:
                path = reference_mesh_path(sequence, args.reference_geometry_root)
                if not path.is_file():
                    raise FileNotFoundError(path)
        if args.initial_mesh and (len(sequences) != 1 or not args.initial_mesh.is_file()):
            raise ValueError('--initial-mesh requires an existing file and exactly one selected sequence.')
        for path in (args.vae_checkpoint, args.denoiser_checkpoint):
            if not path.is_file():
                raise FileNotFoundError(f'{path}; run python checkpoints/download.py from the repository root.')
        plan = dict(frames=16, sequences=[s.record() for s in sequences], config=config,
                    vae_checkpoint=str(args.vae_checkpoint.resolve()),
                    denoiser_checkpoint=str(args.denoiser_checkpoint.resolve()),
                    initial_mesh=str(args.initial_mesh.resolve()) if args.initial_mesh else None,
                    reference_geometry_root=(str(args.reference_geometry_root.resolve())
                                             if args.reference_geometry_root else None),
                    skip_texture=args.skip_texture, num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale, seed=args.seed)
        print(json.dumps({'frames': 16, 'clips': len(sequences), 'images_per_batch': 17,
                          'output_dir': str(args.output_dir), 'sequences': [s.record() for s in sequences]}, indent=2), flush=True)
        if args.validate_weights:
            for name, path in [('first_stage_config', args.vae_checkpoint), ('denoiser_cfg', args.denoiser_checkpoint)]:
                print(json.dumps(checkpoint_model(config['model'][name], path, validate_only=True)), flush=True)
        if args.dry_run:
            return 0

        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for inference. Use --dry-run for input checks on CPU.')
        # Fail before generating canonical meshes if the Blender exporter is absent.
        import bpy  # noqa: F401
        run_file = args.output_dir / 'run.json'
        if run_file.exists():
            previous = json.loads(run_file.read_text())
            if not args.resume or previous['settings'] != plan:
                raise FileExistsError('Output already exists. Use --resume with identical settings or a new --output-dir.')
        elif args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError('Output directory is nonempty and has no matching run.json.')
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for sequence in sequences:
            (args.output_dir / sequence.name).mkdir(parents=True, exist_ok=True)
        pending = []
        for sequence in sequences:
            directory = args.output_dir / sequence.name
            if (directory / 'inference.json').is_file():
                if not (directory / 'combined_gen_mtl.glb').is_file():
                    raise FileNotFoundError(f'Missing completed result in {directory}; choose a new output directory.')
                continue
            pending.append(sequence)
        write_json(run_file, {'settings': plan, 'status': 'running'})
        if pending:
            prepare_meshes(pending, args)
            run_motion(pending, config, args)
        write_json(run_file, {'settings': plan, 'status': 'complete'})
        print(f'Completed {len(sequences)} sixteen-frame sequence(s): {args.output_dir}', flush=True)
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
