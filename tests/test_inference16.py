import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'hy3dshape'))

from infer_16 import parse_args
from inference_utils import (checkpoint_model, input_sequences, load_batch, load_config,
                             reference_mesh_path, reference_surface, validate_images)


class CheckpointTests(unittest.TestCase):
    def test_default_checkpoint_paths_match_release_manifest(self):
        release = json.loads((REPO / 'checkpoints/manifest_16.json').read_text())
        _, args = parse_args(['--input-dir', '/unused'])
        self.assertEqual([args.vae_checkpoint.name, args.denoiser_checkpoint.name],
                         [record['name'] for record in release['checkpoints']])
        self.assertEqual(args.vae_checkpoint.parent, REPO / 'ckpt/16frames')
        self.assertEqual(args.denoiser_checkpoint.parent, REPO / 'ckpt/16frames')

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('accelerate'), 'requires ML environment')
    def test_strict_component_loading_rejects_missing_or_wrong_tensors(self):
        import torch
        config = {'target': 'torch.nn.Linear', 'params': {'in_features': 3, 'out_features': 2}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'small.ckpt'
            state = {'model.weight': torch.ones(2, 3), 'model.bias': torch.zeros(2),
                     'cond_stage_model.frozen_weight': torch.ones(4)}
            torch.save({'state_dict': state}, path)
            actual = checkpoint_model(config, path)
            torch.testing.assert_close(actual(torch.ones(1, 3)), torch.full((1, 2), 3.0))
            state['model.weight'] = torch.ones(6, 3)
            torch.save({'state_dict': state}, path)
            with self.assertRaisesRegex(ValueError, 'shape_mismatch'):
                checkpoint_model(config, path)
            del state['model.bias']
            torch.save({'state_dict': state}, path)
            with self.assertRaisesRegex(ValueError, 'missing='):
                checkpoint_model(config, path)


class InferenceInputTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('trimesh'), 'requires trimesh')
    def test_reference_registration_points_are_scale_and_translation_invariant(self):
        import numpy as np
        import trimesh
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mesh = trimesh.creation.box(extents=[2, 4, 1])
            mesh.export(root / 'first.ply')
            mesh.vertices = mesh.vertices * 3 + [9, -3, 10]
            mesh.export(root / 'transformed.ply')
            points = reference_surface(root / 'first.ply', seed=7)
            self.assertEqual(points.shape, (2048, 3))
            np.testing.assert_allclose(points, reference_surface(root / 'transformed.ply', seed=7), atol=1e-6)
            self.assertGreaterEqual(points.min(), 0)
            self.assertLessEqual(points.max(), 0.9 + 1e-6)

    def test_benchmark_uses_the_48_clip_split_and_absolute_input_frames(self):
        _, args = parse_args(['--benchmark-root', '/unused', '--manifest', str(REPO / 'benchmark/manifest.json')])
        sequences = input_sequences(args)
        self.assertEqual(len(sequences), 48)
        later = next(s for s in sequences if s.frame_numbers[0] == 40)
        self.assertEqual(later.frame_numbers, tuple(range(40, 56)))
        self.assertEqual(later.paths[-1].name, 'front_frame55.png')
        self.assertTrue(later.name.endswith('_40_1'))
        self.assertEqual(reference_mesh_path(later, Path('/geometry')).name, 'frame_0040.ply')

    def test_custom_inputs_reject_short_rgb_and_ambiguous_sequences(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for frame in range(16):
                Image.new('RGBA', (16, 16), (frame, 0, 0, 255)).save(root / f'{frame}.png')
            _, args = parse_args(['--input-dir', str(root)])
            sequences = input_sequences(args)
            validate_images(sequences)
            (root / '15.png').unlink()
            with self.assertRaises(FileNotFoundError):
                validate_images(sequences)
            Image.new('RGB', (16, 16)).save(root / '15.png')
            with self.assertRaisesRegex(ValueError, 'RGBA'):
                validate_images(sequences)
            Image.new('RGBA', (16, 16)).save(root / 'front_frame0.png')
            with self.assertRaises(ValueError):
                input_sequences(args)
            args.image_pattern = '0.png'
            with self.assertRaises(ValueError):
                input_sequences(args)

    def test_config_rejects_mixed_six_and_sixteen_frame_models(self):
        import yaml
        config = load_config(REPO / 'hy3dshape/configs/infer_16.yaml')
        config['model']['first_stage_config']['params']['num_frames'] = 6
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'bad.yaml'
            path.write_text(yaml.safe_dump(config))
            with self.assertRaises(ValueError):
                load_config(path)

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('cv2'), 'requires ML environment')
    def test_canonical_duplicate_and_motion_preserving_crop(self):
        import torch
        from PIL import ImageDraw
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for frame in range(16):
                image = Image.new('RGBA', (64, 32))
                ImageDraw.Draw(image).rectangle((frame * 3, 10, frame * 3 + 5, 20), fill=(200, 30, 10, 255))
                image.save(root / f'{frame}.png')
            _, args = parse_args(['--input-dir', str(root)])
            batch = load_batch(input_sequences(args)[0], image_size=32)
            self.assertEqual(batch['image'].shape, (1, 17, 3, 32, 32))
            self.assertEqual(batch['mask'].shape, (1, 17, 1, 32, 32))
            torch.testing.assert_close(batch['image'][:, 0], batch['image'][:, 1])
            first = torch.nonzero(batch['mask'][0, 1, 0] > 0).float()[:, 1].mean()
            last = torch.nonzero(batch['mask'][0, -1, 0] > 0).float()[:, 1].mean()
            self.assertGreater(float(last - first), 15)
            self.assertEqual(float(batch['mask'].min()), -1)


if __name__ == '__main__':
    unittest.main()
