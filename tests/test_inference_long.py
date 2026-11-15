import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'hy3dshape'))

from infer_long import CONVERT, chunk_windows, frame_count, main, parse_args, window_input, window_sequence
from inference_utils import input_sequences, load_config


def write_frames(root, count, name='{frame}.png'):
    for frame in range(count):
        Image.new('RGBA', (16, 16), (frame, 0, 0, 255)).save(root / name.format(frame=frame))


class WindowTests(unittest.TestCase):
    def test_sixteen_frames_use_three_windows_sharing_one_frame(self):
        self.assertEqual(chunk_windows(16), [(0, 5), (5, 10), (10, 15)])
        self.assertEqual(chunk_windows(6), [(0, 5)])
        self.assertEqual(chunk_windows(2), [(0, 1)])
        self.assertEqual(chunk_windows(13), [(0, 5), (5, 10), (10, 12)])
        with self.assertRaises(ValueError):
            chunk_windows(1)

    def test_windows_chain_and_predict_every_later_frame_once(self):
        for count in range(2, 60):
            windows = chunk_windows(count)
            self.assertEqual(windows[0][0], 0)
            self.assertEqual(windows[-1][1], count - 1)
            for (_, last), (first, _) in zip(windows, windows[1:]):
                self.assertEqual(first, last)
            predicted = [t for first, last in windows for t in range(first + 1, last + 1)]
            self.assertEqual(predicted, list(range(1, count)))
            self.assertTrue(all(1 <= last - first <= 5 for first, last in windows))

    def test_short_final_window_repeats_its_last_frame(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_frames(root, 13)
            _, args = parse_args(['--input-dir', str(root), '--num-frames', '13'])
            sequence = input_sequences(args, 13)[0]
            part = window_sequence(sequence, 2, 10, 12)
            self.assertEqual(part.name, 'window_002')
            self.assertEqual(part.frame_numbers, (10, 11, 12, 12, 12, 12))
            self.assertEqual([p.name for p in part.paths], ['10.png', '11.png', '12.png', '12.png', '12.png', '12.png'])

    def test_window_input_reaches_the_model_as_the_renormalized_pose(self):
        # The two steps the pipeline applies to an input mesh: CONVERT, then
        # the minimum corner and longest extent / 0.9 of the bounding box.
        rng = np.random.default_rng(0)
        pose = rng.normal(size=(500, 3)) * [0.3, 0.5, 0.2] + [0.4, 0.1, 0.6]
        converted = (np.asarray(CONVERT) @ window_input(pose).T).T
        np.testing.assert_allclose(converted, pose, atol=1e-12)
        loc = converted.min(axis=0)
        scale = np.ptp(converted, axis=0).max() / 0.9
        normalized = (converted - loc) / scale
        np.testing.assert_allclose(normalized * scale + loc, pose, atol=1e-12)


class InputTests(unittest.TestCase):
    def test_defaults_are_the_six_frame_release_files(self):
        _, args = parse_args(['--input-dir', '/unused'])
        self.assertEqual(args.vae_checkpoint, REPO / 'ckpt/deform_vae.ckpt')
        self.assertEqual(args.denoiser_checkpoint, REPO / 'ckpt/denoiser.ckpt')
        self.assertEqual(args.num_frames, '16')
        self.assertEqual(load_config(args.config, frames=6)['frames'], 6)

    def test_six_frame_config_matches_the_released_model(self):
        released = yaml.safe_load((REPO / 'hy3dshape/configs/infer.yaml').read_text())['model']['params']
        flat = load_config(REPO / 'hy3dshape/configs/infer_6.yaml', frames=6)['model']
        for name in ('denoiser_cfg', 'first_stage_config', 'cond_stage_config', 'cond_stage_config_2',
                     'image_processor_cfg', 'pipeline_cfg'):
            expected = copy.deepcopy(released[name])
            expected.get('params', {}).pop('ckpt_path', None)
            if name == 'first_stage_config':
                expected['params'].setdefault('num_frames', 6)  # DeformVAEHY's default
            self.assertEqual(flat[name], expected, name)
        self.assertEqual(flat['z_scale_factor'], released['z_scale_factor'])
        with self.assertRaises(ValueError):
            load_config(REPO / 'hy3dshape/configs/infer_16.yaml', frames=6)

    def test_all_counts_consecutive_frames_from_the_start(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_frames(root, 23, '{frame:04d}.png')
            _, args = parse_args(['--input-dir', str(root), '--num-frames', 'all', '--image-pattern', '{frame:04d}.png'])
            self.assertEqual(frame_count(args), 23)
            _, args = parse_args(['--input-dir', str(root), '--num-frames', 'all', '--image-pattern', '{frame:04d}.png',
                                  '--start-frame', '3', '--stride', '2'])
            self.assertEqual(frame_count(args), 10)
            (root / '0007.png').unlink()
            self.assertEqual(frame_count(args), 2)
            sequence = input_sequences(args, 2)[0]
            self.assertEqual(sequence.frame_numbers, (3, 5))
            _, args = parse_args(['--input-dir', str(root), '--num-frames', 'some'])
            with self.assertRaises(ValueError):
                frame_count(args)

    def test_benchmark_uses_the_manifest_six_and_sixteen_frame_splits(self):
        manifest = str(REPO / 'benchmark/manifest.json')
        for frames, clips in (('16', 48), ('6', 50)):
            _, args = parse_args(['--benchmark-root', '/unused', '--manifest', manifest, '--num-frames', frames])
            sequences = input_sequences(args, frame_count(args))
            self.assertEqual(len(sequences), clips)
            self.assertTrue(all(len(s.paths) == int(frames) for s in sequences))
        _, args = parse_args(['--benchmark-root', '/unused', '--num-frames', 'all'])
        with self.assertRaises(ValueError):
            frame_count(args)

    def test_dry_run_plans_the_windows_without_cuda(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frames = root / 'frames'
            frames.mkdir()
            write_frames(frames, 16)
            for name in ('vae.ckpt', 'denoiser.ckpt'):
                (root / name).touch()
            argv = ['--input-dir', str(frames), '--dry-run', '--output-dir', str(root / 'out'),
                    '--vae-checkpoint', str(root / 'vae.ckpt'), '--denoiser-checkpoint', str(root / 'denoiser.ckpt')]
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(main(argv), 0)
            plan = json.loads(stdout.getvalue())
            self.assertEqual(plan['frames'], 16)
            self.assertEqual(plan['windows'], [[0, 5], [5, 10], [10, 15]])
            self.assertFalse((root / 'out').exists())
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                main(argv[:-4] + ['--vae-checkpoint', str(root / 'missing.ckpt'),
                                  '--denoiser-checkpoint', str(root / 'denoiser.ckpt')])


if __name__ == '__main__':
    unittest.main()
