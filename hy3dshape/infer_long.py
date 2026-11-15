#!/usr/bin/env python3
"""Run the six-frame Mesh4D model chunk-wise on 16-frame and longer RGBA videos.

The six-frame model predicts six poses from a canonical mesh and six frames. A
longer video is covered by six-frame windows that overlap by one frame: frames
0-5, 5-10, 10-15, ... Each window starts from the pose the previous window
predicted for its last frame. A final window with fewer than six frames repeats
its last frame, and the repeated poses are discarded. Every window is prepared
and sampled like a six-frame clip (its own foreground crop, the same seed), and
all poses are returned in the first window's coordinates. The output has the
layout of infer_16.py.
"""
import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.append(str(REPO))
from inference_utils import (Sequence, checkpoint_model, image_pattern, input_sequences, load_batch, load_config,
                             reference_mesh_path, reference_surface, validate_images, write_json)

WINDOW = 6
# The motion pipeline turns the input mesh into its Z-up frame with this matrix
# before normalising its bounding box (pipelines_video_newvae_all_nonalign_infer).
CONVERT = ((1, 0, 0), (0, 0, -1), (0, 1, 0))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input-dir', type=Path, help='One sequence of segmented RGBA PNGs.')
    source.add_argument('--benchmark-root', type=Path, help='Use the manifest\'s 6- or 16-frame split.')
    parser.add_argument('--num-frames', default='16',
                        help='Frames to reconstruct (at least 2), or "all" for every consecutive frame from '
                             '--start-frame. Benchmark splits have 6 or 16.')
    parser.add_argument('--input-root', type=Path, help='Override <benchmark-root>/input_images.')
    parser.add_argument('--manifest', type=Path, help='Override <benchmark-root>/manifest.json.')
    parser.add_argument('--clip', action='append', help='Select a benchmark clip_path; repeat for a subset.')
    parser.add_argument('--reference-geometry-root', type=Path,
                        help='Evaluation only: root of original per-frame PLYs for research-style GT registration.')
    parser.add_argument('--image-pattern', help='Custom sequence filenames, e.g. "{frame:04d}.png".')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--config', type=Path, default=HERE / 'configs/infer_6.yaml')
    parser.add_argument('--vae-checkpoint', type=Path, default=REPO / 'ckpt/deform_vae.ckpt')
    parser.add_argument('--denoiser-checkpoint', type=Path, default=REPO / 'ckpt/denoiser.ckpt')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'output_folder/test/infer_long')
    parser.add_argument('--initial-mesh', type=Path,
                        help='Use an existing canonical mesh for one clip; bypass shape generation and painting.')
    parser.add_argument('--skip-texture', action='store_true', help='Generate untextured geometry for a faster check.')
    parser.add_argument('--num-inference-steps', type=int, default=50)
    parser.add_argument('--guidance-scale', type=float, default=5.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--keep-chunks', action='store_true',
                        help='Keep each window\'s own pipeline outputs in <sequence>/chunks.')
    parser.add_argument('--resume', action='store_true', help='Resume the same saved run settings and completed stages.')
    parser.add_argument('--dry-run', action='store_true', help='Validate images, paths and config without CUDA or model downloads.')
    parser.add_argument('--validate-weights', action='store_true',
                        help='Also strictly validate checkpoint tensor names/shapes on CPU; useful with --dry-run.')
    return parser, parser.parse_args(argv)


def chunk_windows(num_frames, window=WINDOW):
    """Inclusive (first, last) frame indices of the overlapping windows covering num_frames frames."""
    if window < 2:
        raise ValueError('A window needs at least two frames.')
    if num_frames < 2:
        raise ValueError('Chunk-wise inference needs at least two frames.')
    step = window - 1
    return [(start, min(start + step, num_frames - 1)) for start in range(0, num_frames - 1, step)]


def window_sequence(sequence, index, first, last, window=WINDOW):
    """One window as a six-frame sequence; a short final window repeats its last frame."""
    count = last - first + 1
    paths = sequence.paths[first:last + 1] + (sequence.paths[last],) * (window - count)
    numbers = sequence.frame_numbers[first:last + 1] + (sequence.frame_numbers[last],) * (window - count)
    return Sequence(f'window_{index:03d}', paths, numbers, sequence.clip_path)


def window_input(vertices):
    """A pose in the first window's coordinates as a pipeline input mesh's vertices.

    The pipeline applies CONVERT and then normalises the bounding box, so this
    input reaches the model as the same pose, normalised again.
    """
    import numpy as np
    return np.asarray(vertices, dtype=np.float64) @ np.asarray(CONVERT, dtype=np.float64)


def frame_count(args):
    if args.num_frames != 'all':
        try:
            return int(args.num_frames)
        except ValueError:
            raise ValueError('--num-frames must be an integer or "all".') from None
    if args.benchmark_root:
        raise ValueError('Benchmark splits have 6 or 16 frames; "all" applies to --input-dir.')
    if args.start_frame < 0 or args.stride < 1:
        raise ValueError('Use a nonnegative --start-frame and positive --stride.')
    pattern = image_pattern(args.input_dir, args.start_frame, args.image_pattern)
    count = 0
    while (args.input_dir / pattern.format(frame=args.start_frame + count * args.stride)).is_file():
        count += 1
        if count > 100000:
            raise ValueError('--image-pattern must include {frame}.')
    return count


def release_cuda():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def read_vertices(path, count):
    import numpy as np
    from infer_16 import read_mesh
    vertices = np.asarray(read_mesh(path).vertices, dtype=np.float64)
    if len(vertices) != count:
        raise ValueError(f'Canonical mesh topology changed during inference: {path}')
    return vertices


def export_pose(vertices, faces, path):
    import trimesh
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(path)


def run_motion(sequences, config, args):
    import numpy as np
    import torch
    import trimesh
    from hy3dshape.schedulers import FlowMatchEulerDiscreteScheduler
    from hy3dshape.utils.misc import instantiate_from_config, instantiate_non_trainable_model
    from hy3dshape.combine_glb_anim import combine_glb_objs
    from hy3dshape.pipelines_video_newvae_all_nonalign_infer import get_loc_scale_min0
    from infer_16 import read_mesh

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
    window = config['frames']
    for sequence in sequences:
        directory = args.output_dir / sequence.name
        work = directory / 'chunks'
        if work.exists():
            shutil.rmtree(work)  # an interrupted attempt
        count = len(sequence.paths)
        windows = chunk_windows(count, window)
        print(f'Inferring {count} frames in {len(windows)} six-frame window(s): {sequence.name}', flush=True)
        mesh = read_mesh(directory / 'textured_gen_mesh.obj')
        faces = np.asarray(mesh.faces)
        alignment_points = (reference_surface(reference_mesh_path(sequence, args.reference_geometry_root), args.seed)
                            if args.reference_geometry_root else None)
        poses = [None] * count  # every pose in the first window's coordinates
        records = []
        for index, (first, last) in enumerate(windows):
            part = window_sequence(sequence, index, first, last, window)
            if index == 0:
                given = mesh.copy()
            else:
                given = trimesh.Trimesh(vertices=window_input(poses[first]), faces=faces, process=False)
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            batch = load_batch(part, config['image_size'], config['padding_ratio'])
            batch = {key: value.to('cuda') if hasattr(value, 'to') else value for key, value in batch.items()}
            with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
                pipeline(batch=batch, output_path=str(work), gen_mesh=given,
                         num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
                         generator=torch.Generator().manual_seed(args.seed), sample_seed=args.seed,
                         not_simplify=True, combine_glb=False,
                         alignment_points=alignment_points if index == 0 else None)

            output = work / part.name
            start = read_vertices(output / 'gen_mesh/registered_gen_mesh.obj', len(mesh.vertices))
            decoded = [read_vertices(output / 'gen' / f'gen_{t:02d}.obj', len(mesh.vertices)) for t in range(window)]
            if index == 0:
                # The canonical mesh after normalisation (and the optional registration).
                poses[0] = start
                loc, scale = np.zeros(3), 1.0
            else:
                # Undo this window's own bounding-box normalisation.
                loc, scale = get_loc_scale_min0(trimesh.Trimesh(vertices=poses[first], faces=faces, process=False))
                loc, scale = np.asarray(loc, dtype=np.float64), float(scale)
            extent = float(np.ptp(poses[first], axis=0).max())
            mismatch = float(np.abs(start * scale + loc - poses[first]).max()) / extent
            if mismatch > 1e-5:
                raise RuntimeError(f'Window {index} of {sequence.name} does not start from its given pose '
                                   f'({mismatch:.2e} of the extent).')
            for t in range(1, last - first + 1):
                poses[first + t] = decoded[t] * scale + loc
            records.append(dict(
                window=index, frames=[first, last], source_frames=list(part.frame_numbers),
                repeated_last_frame=window - (last - first + 1),
                start_pose='canonical mesh' if index == 0 else f'predicted pose {first}',
                normalization={'loc': loc.tolist(), 'scale': scale},
                # How far the model's own decoding of the start pose is from it, relative to the extent.
                decoded_start_offset=float(np.linalg.norm(decoded[0] * scale + loc - poses[first], axis=1).mean()) / extent,
            ))
            if index == 0:
                shutil.copytree(output / 'gen_mesh', directory / 'gen_mesh', dirs_exist_ok=True)
                first_decoded = decoded[0]
            del batch, given, decoded
            release_cuda()

        width = max(2, len(str(count - 1)))
        pose_dir = directory / 'gen'
        pose_dir.mkdir(exist_ok=True)
        paths = [pose_dir / f'gen_{t:0{width}d}.obj' for t in range(count)]
        # As in infer_16.py, gen_00 is the decoded first pose; the animation starts from the canonical mesh.
        export_pose(first_decoded, faces, paths[0])
        for t in range(1, count):
            export_pose(poses[t], faces, paths[t])

        # Preserve the canonical mesh's texture; the other poses follow at timeline frames 3, 5, ...
        mesh.vertices = poses[0]
        base = directory / 'registered_gen_mesh_with_mtl.obj'
        mesh.export(base)
        output = directory / 'combined_gen_mtl.glb'
        combine_glb_objs([str(base), *(str(p) for p in paths[1:])], str(output),
                         fps=24, frame_duration=0, transition_frames=2)
        write_json(directory / 'inference.json', dict(
            sequence.record(), frames=count, model='six-frame', chunking=dict(window=window, overlap=1, windows=records),
            animation_frames=list(range(1, 2 * count, 2)), fps=24, first_pose='canonical mesh',
            alignment=('ground-truth first-pose CPD registration on the first window'
                       if args.reference_geometry_root else 'generated mesh bounding box; no ground-truth registration'),
            glb=output.name, complete=True))
        if not args.keep_chunks:
            shutil.rmtree(work)
        del mesh, poses
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
        frames = frame_count(args)
        if frames < 2:
            raise ValueError(f'Found {frames} frame(s); chunk-wise inference needs at least two.')
        config = load_config(args.config, frames=WINDOW)
        sequences = input_sequences(args, frames)
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
                raise FileNotFoundError(f'{path}; download the six-frame checkpoints as described in README.md.')
        windows = chunk_windows(frames, config['frames'])
        plan = dict(frames=frames, model='six-frame', windows=windows, sequences=[s.record() for s in sequences],
                    config=config, vae_checkpoint=str(args.vae_checkpoint.resolve()),
                    denoiser_checkpoint=str(args.denoiser_checkpoint.resolve()),
                    initial_mesh=str(args.initial_mesh.resolve()) if args.initial_mesh else None,
                    reference_geometry_root=(str(args.reference_geometry_root.resolve())
                                             if args.reference_geometry_root else None),
                    skip_texture=args.skip_texture, num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale, seed=args.seed)
        print(json.dumps({'frames': frames, 'windows': windows, 'clips': len(sequences),
                          'output_dir': str(args.output_dir), 'sequences': [s.record() for s in sequences]},
                         indent=2), flush=True)
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
        from infer_16 import prepare_meshes
        run_file = args.output_dir / 'run.json'
        if run_file.exists():
            previous = json.loads(run_file.read_text())
            if not args.resume or previous['settings'] != json.loads(json.dumps(plan)):
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
        print(f'Completed {len(sequences)} sequence(s) of {frames} frames: {args.output_dir}', flush=True)
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
