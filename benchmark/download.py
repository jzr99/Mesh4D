#!/usr/bin/env python3
"""Download, verify, and extract the three Mesh4D benchmark archives."""
import argparse
import hashlib
import stat
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
DRIVE_URL = 'https://drive.google.com/drive/folders/12kzy9YTctkNNIWELhXboiaM0vNhK9nFn'


def verify_archives(directory, checksums):
    verified = []
    for line in checksums.read_text().splitlines():
        expected, name = line.split()
        if Path(name).name != name:
            raise ValueError(f'Invalid archive name in checksum file: {name}')
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(
                f'Missing {path}. Download the archives from {DRIVE_URL}. '
                'The Drive release may still be pending.'
            )
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f'SHA256 mismatch for {path}; download this archive again.')
        print(f'Verified {name}', flush=True)
        verified.append(path)
    if not verified:
        raise ValueError('The checksum file is empty.')
    return verified


def extract_archives(archives, output_dir, overwrite=False):
    destination = output_dir / 'mesh4d_benchmark'
    if destination.exists() and not overwrite:
        raise FileExistsError(f'{destination} exists; choose another --output-dir or use --overwrite.')
    root = output_dir.resolve()
    # Validate every member before writing any archive contents.
    for archive in archives:
        with ZipFile(archive) as z:
            for member in z.infolist():
                path = PurePosixPath(member.filename)
                if (not path.parts or path.parts[0] != 'mesh4d_benchmark'
                        or '..' in path.parts or '\\' in member.filename
                        or stat.S_ISLNK(member.external_attr >> 16)):
                    raise ValueError(f'Unsafe archive member: {member.filename}')
                target = root.joinpath(*path.parts).resolve()
                if not target.is_relative_to(root):
                    raise ValueError(f'Archive member escapes output directory: {member.filename}')
    output_dir.mkdir(parents=True, exist_ok=True)
    for archive in archives:
        print(f'Extracting {archive.name}', flush=True)
        with ZipFile(archive) as z:
            z.extractall(output_dir)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-dir', type=Path, default=HERE / 'downloads')
    parser.add_argument('--output-dir', type=Path, default=Path('data'))
    parser.add_argument('--skip-download', action='store_true', help='Use archives already on disk.')
    parser.add_argument('--verify-only', action='store_true', help='Check hashes without extracting.')
    parser.add_argument('--overwrite', action='store_true', help='Allow extraction over an existing benchmark.')
    args = parser.parse_args(argv)
    try:
        checksums = HERE / 'SHA256SUMS'
        names = [line.split()[1] for line in checksums.read_text().splitlines()]
        if not args.skip_download and not all((args.archive_dir / n).is_file() for n in names):
            try:
                import gdown
            except ImportError as exc:
                raise RuntimeError('Install the downloader with: python -m pip install gdown') from exc
            args.archive_dir.mkdir(parents=True, exist_ok=True)
            gdown.download_folder(url=DRIVE_URL, output=str(args.archive_dir), quiet=False)
        archives = verify_archives(args.archive_dir, checksums)
        if not args.verify_only:
            destination = extract_archives(archives, args.output_dir, args.overwrite)
            print(f'Benchmark ready at {destination}')
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
