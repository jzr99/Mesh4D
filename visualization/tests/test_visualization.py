import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from visualization import preview, render


class VisualizationTests(unittest.TestCase):
    def test_frame_selection_and_invalid_ranges(self):
        self.assertEqual(render.frame_numbers(1, 11, 2), [1, 3, 5, 7, 9, 11])
        self.assertEqual(render.frame_numbers(40, 42, 1), [40, 41, 42])
        for args in ((2, 1, 1), (0, 2, 0), (-1, 2, 1)):
            with self.assertRaises(ValueError):
                render.frame_numbers(*args)

    def test_preview_requires_all_views_and_composites_alpha(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for view in ('front', 'right'):
                for frame in (1, 3):
                    Image.new('RGBA', (16, 16), (0, 0, 0, 0)).save(root / f'{view}_frame{frame:04d}.png')
            views, frames, paths = preview.collect_frames(root)
            self.assertEqual(frames, [1, 3])
            image = preview.composite(paths[0], views, 1, 16)
            self.assertEqual(image.getpixel((8, 32)), (255, 255, 255))
            (root / 'right_frame0003.png').unlink()
            with self.assertRaises(ValueError):
                preview.collect_frames(root)

    def test_preview_manifest_ignores_stale_files_and_rejects_missing_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'render.json').write_text(json.dumps({'views': ['front'], 'frames': [3]}))
            for frame in (1, 3):
                Image.new('RGBA', (16, 16), 'red').save(root / f'front_frame{frame:04d}.png')
            self.assertEqual(preview.collect_frames(root)[1], [3])
            (root / 'front_frame0003.png').unlink()
            with self.assertRaises(FileNotFoundError):
                preview.collect_frames(root)

    def test_gif_and_optional_mp4(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for frame, color in ((1, 'red'), (3, 'blue')):
                Image.new('RGBA', (17, 17), color).save(root / f'front_frame{frame:04d}.png')
            output = root / 'preview.gif'
            args = ['--input-dir', str(root), '--output', str(output), '--tile-size', '17']
            self.assertEqual(preview.main(args), 0)
            with Image.open(output) as image:
                self.assertEqual(image.n_frames, 2)
                self.assertEqual(image.size, (17, 41))
            original = output.read_bytes()
            with self.assertRaises(SystemExit):
                preview.main(args)
            self.assertEqual(output.read_bytes(), original)
            if shutil.which('ffmpeg'):
                self.assertEqual(preview.main(['--input-dir', str(root), '--output', str(root / 'preview.mp4'),
                                              '--tile-size', '17']), 0)
                self.assertGreater((root / 'preview.mp4').stat().st_size, 0)

    def test_camera_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = {'matrix_world': [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 2], [0, 0, 0, 1]],
                    'x_fov': 0.69, 'y_fov': 0.69}
            (root / 'front.json').write_text(json.dumps(data))
            self.assertEqual(render.read_cameras(root, ['front'])['front'], data)
            data['y_fov'] = 0.5
            (root / 'front.json').write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                render.read_cameras(root, ['front'])

    @unittest.skipUnless(importlib.util.find_spec('bpy'), 'requires Blender bpy')
    def test_blender_animation_render_and_camera_import(self):
        import bpy
        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene = bpy.context.scene
        scene.render.fps = 24
        scene.frame_start, scene.frame_end = 0, 2
        parent = bpy.data.objects.new('MovingRoot', None)
        scene.collection.objects.link(parent)
        for frame, x in ((0, -0.25), (2, 0.25)):
            parent.location.x = x
            parent.keyframe_insert(data_path='location', frame=frame)
        bpy.ops.mesh.primitive_cube_add(size=0.5)
        cube = bpy.context.object
        cube.parent = parent
        cube.shape_key_add(name='Basis')
        key = cube.shape_key_add(name='Stretch')
        for vertex in key.data:
            vertex.co.z *= 1.5
        for frame, value in ((0, 0), (2, 1)):
            key.value = value
            key.keyframe_insert(data_path='value', frame=frame)
        bpy.ops.mesh.primitive_cube_add(size=5)
        bpy.context.object.name = 'Occluder'
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / 'animated.glb'
            bpy.ops.export_scene.gltf(filepath=str(model), export_format='GLB', export_animations=True)
            output = root / 'renders'
            args = ['--input', str(model), '--output-dir', str(output), '--frame-start', '0',
                    '--frame-end', '2', '--frame-step', '2', '--resolution', '32', '--samples', '1',
                    '--shading', 'clay', '--exclude-object', 'Occluder']
            self.assertEqual(render.main(args), 0)
            metadata = json.loads((output / 'render.json').read_text())
            self.assertEqual(metadata['frames'], [0, 2])
            self.assertEqual(metadata['excluded_objects'], ['Occluder'])
            self.assertEqual(len(list(output.glob('*.png'))), 8)
            for path in output.glob('*.png'):
                with Image.open(path) as image:
                    self.assertEqual(image.size, (32, 32))
                    self.assertIsNotNone(image.getchannel('A').getbbox())
            lower, upper = render.scene_bounds(bpy, [0, 2])
            self.assertAlmostEqual(max(b - a for a, b in zip(lower, upper)), 1.0, places=5)
            first, last = render.scene_bounds(bpy, [0]), render.scene_bounds(bpy, [2])
            self.assertGreater(last[0][0] - first[0][0], 0.4)
            saved = (output / 'front_frame0000.png').read_bytes()
            with self.assertRaises(SystemExit):
                render.main(args)
            self.assertEqual((output / 'front_frame0000.png').read_bytes(), saved)
            cameras = root / 'cameras'
            cameras.mkdir()
            camera = metadata['cameras'][0]
            (cameras / 'front.json').write_text(json.dumps(camera))
            fixed = root / 'fixed'
            self.assertEqual(render.main(['--input', str(model), '--output-dir', str(fixed),
                                         '--camera-dir', str(cameras), '--normalize', 'none', '--views', 'front',
                                         '--exclude-object', 'Occluder',
                                         '--frame-start', '0', '--frame-end', '0', '--resolution', '32', '--samples', '1']), 0)
            actual = json.loads((fixed / 'render.json').read_text())['cameras'][0]
            for row1, row2 in zip(actual['matrix_world'], camera['matrix_world']):
                for a, b in zip(row1, row2):
                    self.assertAlmostEqual(a, b, places=5)
            self.assertAlmostEqual(actual['x_fov'], camera['x_fov'], places=5)


if __name__ == '__main__':
    unittest.main()
