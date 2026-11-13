import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
from PIL import Image

from benchmark.download import extract_archives, verify_archives
from benchmark.evaluate import frame_pairs, load_clips, main
from benchmark.metrics import legacy_frechet_score, load_rgb, psnr
from benchmark.render import render_jobs


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.clips = [
            dict(clip_path='000-001/long_040', object_path='000-001/long',
                 start_frame=40, stride=1, six_frame_evaluation=True, sixteen_frame_evaluation=True),
            dict(clip_path='000-002/short_000', object_path='000-002/short',
                 start_frame=0, stride=1, six_frame_evaluation=True, sixteen_frame_evaluation=False),
        ]
        (self.root / 'manifest.json').write_text(json.dumps({'clips': self.clips}))
        self.gt = self.root / 'evaluation_images'
        self.pred = self.root / 'pred'

    def make_images(self):
        image = np.zeros((16, 16, 4), dtype=np.uint8)
        image[..., 0] = np.arange(16, dtype=np.uint8)[None, :] * 16
        image[..., 3] = 255
        for clip in self.clips:
            for pair in frame_pairs(clip, self.gt, self.pred, 6):
                for path in (pair.gt, pair.pred):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(image).save(path)

    def arguments(self, *extra):
        return ['--benchmark-root', str(self.root), '--pred-root', str(self.pred), *extra]

    def test_nonzero_start_and_transition_frames(self):
        pairs = frame_pairs(self.clips[0], self.gt, self.pred, 6)
        self.assertEqual(len(pairs), 24)
        self.assertEqual(pairs[0].gt, self.gt / '000-001/long_040/front_frame0040.png')
        self.assertEqual(pairs[5].gt.name, 'front_frame0045.png')
        self.assertEqual(pairs[0].pred, self.pred / '000-001/long_40_1/combined_gen_mtl/front_frame0001.png')
        self.assertEqual(pairs[5].pred.name, 'front_frame0011.png')

    def test_sixteen_frame_split_excludes_short_clip(self):
        self.assertEqual(len(load_clips(self.root / 'manifest.json', 6)), 2)
        self.assertEqual(load_clips(self.root / 'manifest.json', 16), self.clips[:1])
        with self.assertRaises(ValueError):
            load_clips(self.root / 'manifest.json', 16, ['000-002/short_000'])

    def test_sixteen_frame_evaluation_includes_final_pose(self):
        clip = self.clips[0]
        pairs = frame_pairs(clip, self.gt, self.pred, 16)
        self.assertEqual(len(pairs), 64)
        self.assertEqual(pairs[15].pred.name, 'front_frame0031.png')
        self.assertEqual(pairs[15].gt.name, 'front_frame0055.png')
        image = Image.fromarray(np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3))
        for pair in pairs:
            for path in (pair.pred, pair.gt):
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path)
        out = self.root / 'results16'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(self.arguments('--frames', '16', '--metrics', 'psnr',
                                                 '--image-size', '16', '--output-dir', str(out))), 0)
        result = json.loads((out / 'summary.json').read_text())
        self.assertEqual(result['image_pairs'], 64)
        self.assertTrue(result['full_split'])
        self.assertEqual(result['metrics']['psnr'], 'Infinity')
        pairs[-1].pred.unlink()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(self.arguments('--frames', '16', '--dry-run'))

    def test_sixteen_frame_render_plan_and_mismatched_prediction_rejection(self):
        clip = self.clips[0]
        model = self.root / 'meshes/000-001/long_40_1/combined_gen_mtl.glb'
        model.parent.mkdir(parents=True)
        model.touch()
        camera_dir = self.gt / clip['clip_path']
        camera_dir.mkdir(parents=True)
        camera = {'matrix_world': [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 2], [0, 0, 0, 1]],
                  'x_fov': 0.69, 'y_fov': 0.69}
        for view in ('front', 'right', 'back', 'left'):
            (camera_dir / f'{view}.json').write_text(json.dumps(camera))
        args = SimpleNamespace(manifest=None, benchmark_root=self.root, camera_root=None, frames=16,
                               clip=None, mesh_root=self.root / 'meshes', output_dir=self.pred,
                               overwrite=False, blender='/path/with spaces/blender', resolution=512,
                               samples=64, device='CPU', preserve_coordinates=False, exclude_object=[])
        jobs = render_jobs(args)
        self.assertEqual(len(jobs), 1)
        command = jobs[0]['command']
        self.assertEqual(command[0], '/path/with spaces/blender')
        self.assertEqual(command[command.index('--frame-end') + 1], '31')
        self.assertIn('--mesh4d-coordinates', command)
        self.assertIn(str(camera_dir.resolve()), command)
        (model.parent / 'inference.json').write_text(json.dumps({'frames': 6}))
        with self.assertRaises(ValueError):
            render_jobs(args)

    def test_matching_ground_truth_template(self):
        pairs = frame_pairs(self.clips[0], self.gt, self.gt, 6,
                            '{clip_path}/{view}_frame{gt_frame:04d}.png')
        self.assertTrue(all(pair.pred == pair.gt for pair in pairs))

    def test_missing_prediction_fails_instead_of_changing_split(self):
        self.make_images()
        frame_pairs(self.clips[0], self.gt, self.pred, 6)[3].pred.unlink()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                main(self.arguments('--dry-run'))
        self.assertEqual(raised.exception.code, 1)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(self.arguments('--dry-run', '--allow-missing'))
        coverage = json.loads(output.getvalue())
        self.assertEqual(coverage['complete_clips'], 1)
        self.assertEqual(coverage['image_pairs'], 24)
        self.assertEqual(coverage['skipped_clips'][0]['clip'], '000-001/long_040')

    def test_psnr_cli_writes_complete_results_and_preserves_existing(self):
        self.make_images()
        out = self.root / 'results'
        args = self.arguments('--metrics', 'psnr', '--image-size', '16', '--output-dir', str(out))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 0)
        summary = json.loads((out / 'summary.json').read_text())
        self.assertEqual(summary['complete_clips'], 2)
        self.assertEqual(summary['image_pairs'], 48)
        self.assertEqual(summary['metrics']['psnr'], 'Infinity')
        self.assertEqual(len((out / 'per_frame.csv').read_text().splitlines()), 49)
        original = (out / 'summary.json').read_bytes()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(args)
        self.assertEqual((out / 'summary.json').read_bytes(), original)

    def test_metric_values_and_alpha_handling(self):
        self.assertAlmostEqual(psnr(np.zeros((2, 2, 3)), np.full((2, 2, 3), 255), 255), 0)
        self.assertAlmostEqual(psnr(np.full((2, 2, 3), 229.5), np.full((2, 2, 3), 255), 255), 20)
        path = self.root / 'alpha.png'
        rgba = np.array([[[0, 0, 0, 0], [10, 20, 30, 128]]], dtype=np.uint8)
        Image.fromarray(rgba).save(path)
        actual = np.asarray(load_rgb(path))
        np.testing.assert_array_equal(actual, [[[255, 255, 255], [10, 20, 30]]])

    def test_legacy_i3d_distance_of_identical_outputs(self):
        values = np.arange(400, dtype=float).reshape(1, 400) / 100
        self.assertAlmostEqual(legacy_frechet_score(values, values), 0, places=8)
        self.assertAlmostEqual(legacy_frechet_score(values + 1, values), 400, places=8)


class ArchiveTests(unittest.TestCase):
    def test_verified_archive_extracts_and_corruption_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'small.zip'
            with ZipFile(archive, 'w') as z:
                z.writestr('mesh4d_benchmark/manifest.json', '{}')
            checksums = root / 'SHA256SUMS'
            checksums.write_text(hashlib.sha256(archive.read_bytes()).hexdigest() + '  small.zip\n')
            with contextlib.redirect_stdout(io.StringIO()):
                verified = verify_archives(root, checksums)
                destination = extract_archives(verified, root / 'data')
            self.assertEqual((destination / 'manifest.json').read_text(), '{}')
            with self.assertRaises(FileExistsError):
                extract_archives(verified, root / 'data')
            archive.write_bytes(b'incomplete download')
            with self.assertRaises(ValueError):
                verify_archives(root, checksums)

    def test_archive_cannot_escape_dataset_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'bad.zip'
            with ZipFile(archive, 'w') as z:
                z.writestr('mesh4d_benchmark/../../escaped.txt', 'bad')
            with self.assertRaises(ValueError):
                extract_archives([archive], root / 'data')
            self.assertFalse((root / 'escaped.txt').exists())


if __name__ == '__main__':
    unittest.main()
