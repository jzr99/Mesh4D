#!/usr/bin/env python3
"""Assemble rendered PNG views into a white-background GIF or MP4 preview."""
import argparse
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def collect_frames(directory, views=None):
    metadata_path = directory / 'render.json'
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        views = views or metadata['views']
        frames = metadata['frames']
    else:
        found = {}
        for path in directory.glob('*_frame*.png'):
            match = re.fullmatch(r'([a-z]+)_frame(\d+)\.png', path.name)
            if match:
                found.setdefault(match[1], set()).add(int(match[2]))
        order = ('front', 'right', 'back', 'left', 'turntable')
        views = views or [view for view in order if view in found]
        if not views or any(view not in found for view in views):
            raise ValueError('No rendered frames found for the requested views.')
        frames = sorted(found[views[0]])
        if any(found[view] != set(frames) for view in views):
            raise ValueError('Views contain different frame numbers; refusing to drop incomplete frames.')
    if not views or not frames or any(not re.fullmatch('[a-z]+', view) for view in views):
        raise ValueError('Invalid or empty render manifest.')
    paths = [[directory / f'{view}_frame{frame:04d}.png' for view in views] for frame in frames]
    missing = [str(path) for row in paths for path in row if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'Missing rendered frames: {missing[:8]}')
    return views, frames, paths


def composite(paths, views, frame, tile_size):
    from PIL import Image, ImageDraw
    canvas = Image.new('RGB', (tile_size * len(views), tile_size + 24), 'white')
    draw = ImageDraw.Draw(canvas)
    for index, (path, view) in enumerate(zip(paths, views)):
        with Image.open(path) as source:
            rgba = source.convert('RGBA')
            rgba.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
            tile = Image.new('RGBA', rgba.size, 'white')
            tile.alpha_composite(rgba)
            x = index * tile_size + (tile_size - rgba.width) // 2
            canvas.paste(tile.convert('RGB'), (x, 24 + (tile_size - rgba.height) // 2))
        draw.text((index * tile_size + 8, 6), f'{view} | frame {frame}', fill='black')
    return canvas


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='.gif or .mp4; MP4 requires ffmpeg on PATH.')
    parser.add_argument('--views', nargs='+', choices=('front', 'right', 'back', 'left', 'turntable'))
    parser.add_argument('--fps', type=float, default=12, help='Preview playback rate, independent of glTF import FPS.')
    parser.add_argument('--tile-size', type=int, default=256)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    try:
        suffix = args.output.suffix.lower()
        if suffix not in ('.gif', '.mp4') or not math.isfinite(args.fps) or not 0 < args.fps <= 100 or args.tile_size < 16:
            raise ValueError('Use .gif/.mp4, 0 < FPS <= 100, and tile size >= 16.')
        if args.output.exists() and not args.overwrite:
            raise FileExistsError('Preview already exists; use another output or --overwrite.')
        if suffix == '.mp4' and shutil.which('ffmpeg') is None:
            raise RuntimeError('MP4 needs ffmpeg on PATH; use a .gif output or install ffmpeg.')
        views, frames, rows = collect_frames(args.input_dir, args.views)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if suffix == '.gif':
            images = [composite(row, views, frame, args.tile_size) for frame, row in zip(frames, rows)]
            images[0].save(args.output, save_all=True, append_images=images[1:],
                           duration=round(1000 / args.fps), loop=0, disposal=2)
        else:
            with tempfile.TemporaryDirectory(prefix='mesh4d-preview-') as temp:
                for index, (frame, row) in enumerate(zip(frames, rows)):
                    composite(row, views, frame, args.tile_size).save(Path(temp) / f'frame_{index:06d}.png')
                subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
                                '-framerate', str(args.fps), '-i', str(Path(temp) / 'frame_%06d.png'),
                                '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white',
                                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                                str(args.output.resolve())], check=True)
        print(f'Saved {len(frames)} frames ({", ".join(views)}) to {args.output}')
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, ImportError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
