import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from checkpoints.download import main, obtain_checkpoint, read_manifest

REPO = Path(__file__).resolve().parents[1]


class CheckpointDownloadTests(unittest.TestCase):
    def setUp(self):
        self.content = b'a small complete checkpoint'
        self.record = dict(name='model.ckpt', size=len(self.content),
                           sha256=hashlib.sha256(self.content).hexdigest())
        self.manifest = dict(format='huggingface-v1', frames=16, repo_id='owner/model',
                             revision='a' * 40, checkpoints=[self.record])

    def hub(self, callback):
        client = Mock(side_effect=callback)
        module = types.ModuleType('huggingface_hub')
        module.hf_hub_download = client
        return patch.dict(sys.modules, huggingface_hub=module), client

    def download(self, **kwargs):
        path = Path(kwargs['local_dir']) / kwargs['filename']
        path.write_bytes(self.content)
        return str(path)

    def test_downloads_pinned_file_and_reuses_verified_output(self):
        context, client = self.hub(self.download)
        with tempfile.TemporaryDirectory() as temp, context:
            root = Path(temp)
            output = obtain_checkpoint(self.record, self.manifest, root)
            self.assertEqual(output.read_bytes(), self.content)
            self.assertFalse((root / '.download' / self.record['name']).exists())
            self.assertEqual(client.call_args.kwargs['repo_id'], 'owner/model')
            self.assertEqual(client.call_args.kwargs['revision'], 'a' * 40)
            self.assertIs(client.call_args.kwargs['token'], False)
            self.assertEqual(obtain_checkpoint(self.record, self.manifest, root), output)
            client.assert_called_once()

    def test_same_size_corruption_is_never_published(self):
        def corrupt(**kwargs):
            path = Path(self.download(**kwargs))
            path.write_bytes(b'x' * self.record['size'])
            return str(path)
        context, _ = self.hub(corrupt)
        with tempfile.TemporaryDirectory() as temp, context:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                obtain_checkpoint(self.record, self.manifest, root)
            self.assertFalse((root / self.record['name']).exists())

    def test_invalid_existing_checkpoint_is_preserved(self):
        context, client = self.hub(self.download)
        with tempfile.TemporaryDirectory() as temp, context:
            root = Path(temp)
            output = root / self.record['name']
            output.write_bytes(b'x' * self.record['size'])
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                obtain_checkpoint(self.record, self.manifest, root)
            self.assertEqual(output.read_bytes(), b'x' * self.record['size'])
            self.assertFalse((root / '.download').exists())
            client.assert_not_called()

    def test_file_created_during_download_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / self.record['name']
            def interrupted_by_existing_file(**kwargs):
                staged = self.download(**kwargs)
                output.write_bytes(b'existing user file')
                return staged
            context, _ = self.hub(interrupted_by_existing_file)
            with context, self.assertRaises(ValueError):
                obtain_checkpoint(self.record, self.manifest, root)
            self.assertEqual(output.read_bytes(), b'existing user file')

    def test_interrupted_download_retains_resume_directory(self):
        def interrupt(**kwargs):
            partial = Path(kwargs['local_dir']) / 'partial-download'
            partial.write_bytes(b'partial bytes')
            raise OSError('connection interrupted')
        context, client = self.hub(interrupt)
        with tempfile.TemporaryDirectory() as temp, context:
            root = Path(temp)
            with self.assertRaisesRegex(RuntimeError, 'rerun to resume'):
                obtain_checkpoint(self.record, self.manifest, root)
            self.assertFalse((root / self.record['name']).exists())
            self.assertEqual((root / '.download/partial-download').read_bytes(), b'partial bytes')
            first_directory = client.call_args.kwargs['local_dir']
            client.side_effect = self.download
            self.assertEqual(obtain_checkpoint(self.record, self.manifest, root).read_bytes(), self.content)
            self.assertEqual(client.call_args.kwargs['local_dir'], first_directory)

    def test_verify_only_cli_is_offline_and_does_not_create_output(self):
        context, client = self.hub(self.download)
        with tempfile.TemporaryDirectory() as temp, context:
            root = Path(temp)
            manifest_path = root / 'manifest.json'
            manifest_path.write_text(json.dumps(self.manifest))
            output_dir = root / 'output'
            argv = ['--manifest', str(manifest_path), '--output-dir', str(output_dir), '--verify-only']
            with self.assertRaises(SystemExit) as error:
                main(argv)
            self.assertEqual(error.exception.code, 1)
            self.assertFalse(output_dir.exists())
            output_dir.mkdir()
            (output_dir / self.record['name']).write_bytes(self.content)
            self.assertEqual(main(argv), 0)
            self.assertEqual(sorted(p.name for p in output_dir.iterdir()), [self.record['name']])
            client.assert_not_called()

    def test_release_manifest_and_invalid_metadata(self):
        release = read_manifest(REPO / 'checkpoints/manifest_16.json')
        self.assertEqual(release['repo_id'], 'zerenjiang/Mesh4D_16')
        self.assertEqual(release['revision'], 'a2d8b4a2bed73b6e52a8866ca95055898c3d0a91')
        self.assertEqual([c['name'] for c in release['checkpoints']],
                         ['vae_bone_r2_16_00001500.ckpt', 'diffuse_r3_ckpt-step=00003000.ckpt'])
        self.assertEqual(sum(c['size'] for c in release['checkpoints']), 27709068732)
        cases = [
            ('name', '../escape.ckpt'),
            ('name', 'folder\\escape.ckpt'),
            ('size', 0),
            ('sha256', 'z' * 64),
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'manifest.json'
            for key, value in cases:
                with self.subTest(key=key, value=value):
                    manifest = json.loads(json.dumps(self.manifest))
                    manifest['checkpoints'][0][key] = value
                    path.write_text(json.dumps(manifest))
                    with self.assertRaises(ValueError):
                        read_manifest(path)
            manifest = dict(self.manifest, revision='main')
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                read_manifest(path)
            manifest = dict(self.manifest, checkpoints=[self.record, self.record])
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                read_manifest(path)


if __name__ == '__main__':
    unittest.main()
