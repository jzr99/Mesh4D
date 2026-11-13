#!/usr/bin/env python3
"""Download and verify the complete 16-frame Mesh4D checkpoints from Hugging Face."""
import argparse
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
BLOCK_SIZE = 8 * 1024 * 1024


def safe_name(name):
    if not isinstance(name, str) or name in ('', '.', '..') or Path(name).name != name or '\\' in name:
        raise ValueError(f'Invalid checkpoint filename: {name!r}')
    return name


def read_manifest(path):
    manifest = json.loads(path.read_text())
    if manifest['format'] != 'huggingface-v1' or manifest['frames'] != 16:
        raise ValueError('Expected the huggingface-v1 16-frame checkpoint manifest.')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', manifest['repo_id']):
        raise ValueError('Invalid Hugging Face repository ID.')
    if not re.fullmatch(r'[0-9a-f]{40}', manifest['revision']):
        raise ValueError('The Hugging Face revision must be a full commit SHA.')
    names = set()
    for checkpoint in manifest['checkpoints']:
        name = safe_name(checkpoint['name'])
        if name in names:
            raise ValueError(f'Duplicate manifest filename: {name}')
        names.add(name)
        if (type(checkpoint['size']) is not int or checkpoint['size'] <= 0
                or not re.fullmatch(r'[0-9a-f]{64}', checkpoint['sha256'])):
            raise ValueError(f'Invalid size or SHA-256 for {name}')
    if not manifest['checkpoints']:
        raise ValueError('The manifest contains no checkpoints.')
    return manifest


def verify_file(path, record):
    if not path.is_file() or path.stat().st_size != record['size']:
        raise ValueError(f'Missing or incomplete file: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(BLOCK_SIZE), b''):
            digest.update(block)
    if digest.hexdigest() != record['sha256']:
        raise ValueError(f'SHA-256 mismatch: {path}')


def obtain_checkpoint(checkpoint, manifest, output_dir, verify_only=False):
    """Resume a download in staging and publish it only after checksum validation."""
    destination = output_dir / safe_name(checkpoint['name'])
    if destination.exists() or verify_only:
        verify_file(destination, checkpoint)
        return destination
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("Install the client: python -m pip install 'huggingface-hub==0.30.2'") from exc

    staging_dir = output_dir / '.download'
    staging_dir.mkdir(parents=True, exist_ok=True)
    try:
        staged = Path(hf_hub_download(
            repo_id=manifest['repo_id'],
            filename=checkpoint['name'],
            repo_type='model',
            revision=manifest['revision'],
            local_dir=staging_dir,
            token=False,
        ))
    except Exception as exc:
        raise RuntimeError(
            f'Unable to download {checkpoint["name"]} from {manifest["repo_id"]}; '
            f'rerun to resume. {exc}'
        ) from exc
    verify_file(staged, checkpoint)
    # Preserve a file that appeared while downloading instead of overwriting it.
    if destination.exists():
        verify_file(destination, checkpoint)
    else:
        staged.replace(destination)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=HERE / 'manifest_16.json')
    parser.add_argument('--output-dir', type=Path, default=HERE.parent / 'ckpt' / '16frames')
    parser.add_argument('--verify-only', action='store_true',
                        help='Verify existing checkpoints without downloading or writing.')
    args = parser.parse_args(argv)
    try:
        manifest = read_manifest(args.manifest)
        for checkpoint in manifest['checkpoints']:
            print(f'Checking {checkpoint["name"]}', flush=True)
            output = obtain_checkpoint(checkpoint, manifest, args.output_dir, args.verify_only)
            print(f'Verified {output}', flush=True)
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
