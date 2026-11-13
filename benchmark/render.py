#!/usr/bin/env python3
"""Render a Mesh4D benchmark split with its saved cameras and prediction timing."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(REPO))

from benchmark.evaluate import VIEWS, load_clips
from visualization.render import read_cameras


def render_jobs(args):
    manifest = args.manifest or args.benchmark_root / 'manifest.json'
    cameras_root = args.camera_root or args.benchmark_root / 'evaluation_images'
    jobs = []
    for clip in load_clips(manifest, args.frames, args.clip):
        name = f'{clip["object_path"]}_{clip["start_frame"]}_{clip["stride"]}'
        if any(Path(p).is_absolute() or '..' in Path(p).parts for p in (name, clip['clip_path'])):
            raise ValueError('Manifest paths must stay within the benchmark directories.')
        model = args.mesh_root / name / 'combined_gen_mtl.glb'
        if not model.is_file():
            raise FileNotFoundError(model)
        info = model.parent / 'inference.json'
        if info.is_file() and json.loads(info.read_text())['frames'] != args.frames:
            raise ValueError(f'{model} does not contain {args.frames}-frame predictions.')
        camera_dir = cameras_root / clip['clip_path']
        read_cameras(camera_dir, VIEWS)
        output = args.output_dir / name / 'combined_gen_mtl'
        expected = [output / f'{view}_frame{2*t+1:04d}.png'
                    for view in VIEWS for t in range(args.frames)]
        if not args.overwrite and any(p.exists() for p in [output / 'render.json', *expected]):
            raise FileExistsError(f'{output} already has renders; select remaining --clip values or use --overwrite.')
        renderer = REPO / 'visualization/render.py'
        command = ([args.blender, '--background', '--python-exit-code', '1', '--python', str(renderer), '--']
                   if args.blender else [sys.executable, str(renderer)])
        command += ['--input', str(model.resolve()), '--output-dir', str(output.resolve()),
                    '--camera-dir', str(camera_dir.resolve()), '--normalize', 'none',
                    '--frame-start', '1', '--frame-end', str(2 * args.frames - 1), '--frame-step', '2',
                    '--fps', '24', '--resolution', str(args.resolution), '--samples', str(args.samples),
                    '--device', args.device]
        if args.overwrite:
            command.append('--overwrite')
        if not args.preserve_coordinates:
            command.append('--mesh4d-coordinates')
        for name in args.exclude_object:
            command += ['--exclude-object', name]
        jobs.append({'clip': clip['clip_path'], 'command': command})
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-root', type=Path, default=Path('data/mesh4d_benchmark'))
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--camera-root', type=Path, help='Override <benchmark-root>/evaluation_images.')
    parser.add_argument('--mesh-root', type=Path, required=True, help='Inference output directory containing group/clip/combined_gen_mtl.glb.')
    parser.add_argument('--output-dir', type=Path, required=True, help='Prediction render root for benchmark/evaluate.py.')
    parser.add_argument('--frames', type=int, choices=(6, 16), default=16)
    parser.add_argument('--clip', action='append')
    parser.add_argument('--blender', help='Blender executable; omitted runs with this Python environment\'s bpy.')
    parser.add_argument('--device', choices=('CPU', 'CUDA', 'OPTIX', 'HIP', 'ONEAPI', 'METAL'), default='CPU')
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--exclude-object', action='append', default=[])
    parser.add_argument('--preserve-coordinates', action='store_true',
                        help='Skip the research Mesh4D GLB axis conversion/normalization for meshes already in camera coordinates.')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Check all meshes and cameras, then print commands without rendering.')
    args = parser.parse_args(argv)
    try:
        if args.resolution < 16 or args.samples < 1:
            raise ValueError('Use resolution >= 16 and samples >= 1.')
        jobs = render_jobs(args)
        if args.dry_run:
            print(json.dumps({'frames': args.frames, 'clips': len(jobs),
                              'images': len(jobs) * args.frames * 4, 'jobs': jobs}, indent=2))
            return 0
        for index, job in enumerate(jobs, 1):
            print(f'[{index}/{len(jobs)}] Rendering {job["clip"]}', flush=True)
            subprocess.run(job['command'], check=True)
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
