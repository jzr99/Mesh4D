#!/usr/bin/env python3
"""Evaluate rendered Mesh4D predictions against the benchmark's four-view RGBs."""
import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VIEWS = ('front', 'right', 'back', 'left')
PRED_TEMPLATE = '{object_path}_{start}_{stride}/combined_gen_mtl/{view}_frame{pred_frame:04d}.png'


@dataclass(frozen=True)
class FramePair:
    view: str
    t: int
    pred: Path
    gt: Path


def load_clips(manifest_path, frames, selected=None):
    manifest = json.loads(manifest_path.read_text())
    flag = 'six_frame_evaluation' if frames == 6 else 'sixteen_frame_evaluation'
    clips = [c for c in manifest['clips'] if c[flag]]
    names = [c['clip_path'] for c in clips]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate clips in manifest.')
    if selected:
        unknown = set(selected) - set(names)
        if unknown:
            raise ValueError(f'Clips unavailable for the {frames}-frame split: {sorted(unknown)}')
        clips = [c for c in clips if c['clip_path'] in selected]
    if not clips:
        raise ValueError('No clips selected.')
    return clips


def frame_pairs(clip, gt_root, pred_root, frames, template=PRED_TEMPLATE, offset=1, step=2):
    pairs = []
    for view in VIEWS:
        for t in range(frames):
            gt_frame = clip['start_frame'] + t * clip['stride']
            context = dict(clip, start=clip['start_frame'], view=view, t=t,
                           gt_frame=gt_frame, pred_frame=offset + t * step)
            relative = Path(template.format(**context))
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('--pred-template must produce a path inside --pred-root.')
            gt = gt_root / clip['clip_path'] / f'{view}_frame{gt_frame:04d}.png'
            pairs.append(FramePair(view, t, pred_root / relative, gt))
    return pairs


def prepare_pairs(clips, args):
    ready, missing = [], []
    for clip in clips:
        pairs = frame_pairs(clip, args.gt_root, args.pred_root, args.frames,
                            args.pred_template, args.pred_frame_offset, args.pred_frame_step)
        absent = sorted({str(path) for pair in pairs for path in (pair.pred, pair.gt) if not path.is_file()})
        if absent:
            missing.append({'clip': clip['clip_path'], 'missing_files': absent})
        else:
            ready.append((clip, pairs))
    return ready, missing


def average(rows, keys):
    result = {}
    for key in keys:
        values = [row[key] for row in rows]
        value = sum(values) / len(values)
        if math.isnan(value):
            raise ValueError(f'Undefined average for {key}; inspect the per-frame scores.')
        result[key] = value
    return result


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            raise ValueError('Cannot serialize an undefined metric.')
        return 'Infinity' if value > 0 else '-Infinity'
    return value


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-root', type=Path, default=Path('data/mesh4d_benchmark'))
    parser.add_argument('--manifest', type=Path, help='Defaults to <benchmark-root>/manifest.json.')
    parser.add_argument('--gt-root', type=Path, help='Defaults to <benchmark-root>/evaluation_images.')
    parser.add_argument('--pred-root', type=Path, required=True)
    parser.add_argument('--frames', type=int, choices=(6, 16), default=6)
    parser.add_argument('--clip', action='append', help='Evaluate one manifest clip_path; repeat for a subset.')
    parser.add_argument('--pred-template', default=PRED_TEMPLATE)
    parser.add_argument('--pred-frame-offset', type=int, default=1)
    parser.add_argument('--pred-frame-step', type=int, default=2)
    parser.add_argument('--metrics', nargs='+', choices=('psnr', 'ssim', 'lpips', 'clip', 'legacy_fvd'),
                        default=['psnr', 'ssim', 'lpips', 'clip'])
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N')
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--data-range', type=float, help='PSNR/SSIM data range; omitted matches legacy dynamic range.')
    parser.add_argument('--clip-model', default='openai/clip-vit-base-patch32')
    parser.add_argument('--i3d-weights', type=Path, help='Original checkpoint required for legacy_fvd.')
    parser.add_argument('--output-dir', type=Path, default=Path('benchmark/results'))
    parser.add_argument('--dry-run', action='store_true', help='Validate all file pairs without loading metric models.')
    parser.add_argument('--allow-missing', action='store_true', help='Skip incomplete clips and record them in summary.json.')
    parser.add_argument('--overwrite', action='store_true', help='Replace results in --output-dir.')
    args = parser.parse_args(argv)
    args.manifest = args.manifest or args.benchmark_root / 'manifest.json'
    args.gt_root = args.gt_root or args.benchmark_root / 'evaluation_images'
    try:
        if args.pred_frame_step < 1 or args.pred_frame_offset < 0 or args.image_size < 16:
            raise ValueError('Use a positive frame step, nonnegative offset, and image size >= 16.')
        if args.data_range is not None and (not math.isfinite(args.data_range) or args.data_range <= 0):
            raise ValueError('--data-range must be finite and positive.')
        clips = load_clips(args.manifest, args.frames, args.clip)
        split_size = len(load_clips(args.manifest, args.frames))
        ready, missing = prepare_pairs(clips, args)
        coverage = {'frames_per_clip': args.frames, 'views': list(VIEWS), 'split_clips': split_size,
                    'requested_clips': len(clips), 'full_split': len(ready) == split_size,
                    'complete_clips': len(ready), 'image_pairs': sum(len(pairs) for _, pairs in ready),
                    'skipped_clips': missing}
        print(json.dumps(coverage, indent=2), flush=True)
        if missing and not args.allow_missing:
            raise ValueError('Missing benchmark/prediction images; no metrics computed. Fix paths or explicitly use --allow-missing.')
        if not ready:
            raise ValueError('No complete clips available.')
        if args.dry_run:
            return 0
        names = ('per_frame.csv', 'per_clip.csv', 'summary.json')
        if not args.overwrite and any((args.output_dir / n).exists() for n in names):
            raise FileExistsError('Results already exist; choose another --output-dir or use --overwrite.')
        if 'legacy_fvd' in args.metrics and (args.i3d_weights is None or not args.i3d_weights.is_file()):
            raise ValueError('legacy_fvd requires --i3d-weights /path/to/i3d_pretrained_400.pt.')
        from benchmark.metrics import MetricComputer
        scorer = MetricComputer(args.metrics, args.device, args.image_size, args.data_range,
                                args.clip_model, args.i3d_weights)
        image_keys = [key for key in dict.fromkeys(args.metrics) if key != 'legacy_fvd']
        frame_rows, clip_rows = [], []
        for index, (clip, pairs) in enumerate(ready, 1):
            clip_frames, view_scores = [], []
            if image_keys:
                for pair in pairs:
                    scores = scorer.image(pair.pred, pair.gt)
                    if any(math.isnan(v) for v in scores.values()):
                        raise ValueError(f'NaN metric for {pair.pred}')
                    clip_frames.append(dict(clip=clip['clip_path'], view=pair.view, frame=pair.t, **scores))
                frame_rows.extend(clip_frames)
            row = {'clip': clip['clip_path'], 'frames': args.frames, 'views': len(VIEWS)}
            row.update(average(clip_frames, image_keys))
            if 'legacy_fvd' in args.metrics:
                for view in VIEWS:
                    view_pairs = [p for p in pairs if p.view == view]
                    view_scores.append(scorer.sequence([p.pred for p in view_pairs], [p.gt for p in view_pairs]))
                row.update(average(view_scores, ['legacy_fvd']))
            clip_rows.append(row)
            print(f'[{index}/{len(ready)}] {clip["clip_path"]}: {row}', flush=True)
        summary = dict(coverage, metrics=average(clip_rows, list(dict.fromkeys(args.metrics))),
                       protocol={'manifest': str(args.manifest.resolve()), 'gt_root': str(args.gt_root.resolve()),
                                 'pred_root': str(args.pred_root.resolve()), 'pred_template': args.pred_template,
                                 'pred_frame_offset': args.pred_frame_offset, 'pred_frame_step': args.pred_frame_step,
                                 'image_size': args.image_size, 'data_range': args.data_range,
                                 'device': scorer.device, 'clip_model': args.clip_model,
                                 'i3d_weights': str(args.i3d_weights) if args.i3d_weights else None,
                                 'aggregation': 'mean of per-clip means; equal frames and views per clip'})
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_dir / 'per_frame.csv', frame_rows)
        write_csv(args.output_dir / 'per_clip.csv', clip_rows)
        (args.output_dir / 'summary.json').write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False) + '\n')
        print(f'Results saved to {args.output_dir}')
        return 0
    except (OSError, KeyError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
