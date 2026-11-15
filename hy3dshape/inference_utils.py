"""Input preparation and strict checkpoint loading for six- and sixteen-frame inference.

RGBA compositing, shared foreground crop, padding and normalization follow
Hunyuan3D-2.1/hy3dshape/dataset/OBJVERSES.py and custom_dataloader.py.
"""
import copy
import json
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FRAMES = 16  # default sequence length (infer_16.py)


def relative_name(name):
    path = Path(name)
    if not name or path.is_absolute() or '..' in path.parts or '\\' in name or name == '.':
        raise ValueError(f'Invalid relative sequence name: {name!r}')
    return name


@dataclass(frozen=True)
class Sequence:
    name: str
    paths: tuple
    frame_numbers: tuple
    clip_path: str = None

    def record(self):
        return {'name': self.name, 'images': [str(p) for p in self.paths],
                'source_frames': list(self.frame_numbers), 'clip_path': self.clip_path}


def input_sequences(args, frames=FRAMES):
    if args.benchmark_root is not None:
        from benchmark.evaluate import load_clips
        manifest = args.manifest or args.benchmark_root / 'manifest.json'
        root = args.input_root or args.benchmark_root / 'input_images'
        sequences = []
        for clip in load_clips(manifest, frames, args.clip):
            object_path = relative_name(clip['object_path'])
            relative_name(clip['clip_path'])
            start, stride = clip['start_frame'], clip['stride']
            if start < 0 or stride < 1:
                raise ValueError(f'Invalid frame interval: {clip["clip_path"]}')
            numbers = tuple(start + t * stride for t in range(frames))
            paths = tuple((root / object_path / f'front_frame{t}.png').resolve() for t in numbers)
            sequences.append(Sequence(f'{object_path}_{start}_{stride}', paths, numbers, clip['clip_path']))
        return sequences
    if args.clip or args.input_root or args.manifest:
        raise ValueError('--clip, --input-root and --manifest require --benchmark-root.')
    if args.start_frame < 0 or args.stride < 1:
        raise ValueError('Use a nonnegative --start-frame and positive --stride.')
    numbers = tuple(args.start_frame + t * args.stride for t in range(frames))
    pattern = image_pattern(args.input_dir, numbers[0], args.image_pattern)
    names = [relative_name(pattern.format(frame=t)) for t in numbers]
    if len(set(names)) != frames:
        raise ValueError(f'--image-pattern must produce {frames} distinct paths; include {{frame}}.')
    paths = tuple((args.input_dir / name).resolve() for name in names)
    name = f'{args.input_dir.resolve().name}_{args.start_frame}_{args.stride}'
    return [Sequence(relative_name(name), paths, numbers)]


def image_pattern(input_dir, first_frame, pattern=None):
    """The given filename pattern, or the one default pattern whose first frame exists."""
    if pattern is None:
        candidates = ('{frame}.png', 'front_frame{frame}.png')
        matches = [p for p in candidates if (input_dir / p.format(frame=first_frame)).is_file()]
        if len(matches) != 1:
            raise ValueError('Cannot determine image names; provide --image-pattern, e.g. "{frame:04d}.png".')
        pattern = matches[0]
    return pattern


def validate_images(sequences):
    from PIL import Image
    for sequence in sequences:
        dimensions = set()
        for path in sequence.paths:
            with Image.open(path) as image:
                if image.mode != 'RGBA':
                    raise ValueError(f'Expected segmented RGBA input: {path} (got {image.mode})')
                dimensions.add(image.size)
                image.verify()
        if len(dimensions) != 1:
            raise ValueError(f'All frames must have the same dimensions: {sequence.name}')


def load_batch(sequence, image_size=518, padding_ratio=1.15):
    import cv2
    import numpy as np
    import torch
    import torchvision.transforms as transforms

    # The reference prepends the canonical frame: 17 inputs, 16 motion targets.
    paths = (sequence.paths[0], *sequence.paths)
    images, masks = [], []
    bounds = []
    for path in paths:
        rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
            raise ValueError(f'Cannot read RGBA image: {path}')
        alpha = rgba[:, :, 3:4].astype(np.float32) / 255
        rgb = (rgba[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
        images.append(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        masks.append((alpha[:, :, 0] * 255).astype(np.uint8))
        coords = np.argwhere(masks[-1] > 0.3)
        if len(coords) == 0:
            coords = np.array([[0, 0], [rgba.shape[0] - 1, rgba.shape[1] - 1]])
        bounds.append((*coords.min(axis=0), *coords.max(axis=0)))
    height, width = images[0].shape[:2]
    y0 = max(min(b[0] for b in bounds) - 5, 0)
    x0 = max(min(b[1] for b in bounds) - 5, 0)
    y1 = min(max(b[2] for b in bounds) + 5, height)
    x1 = min(max(b[3] for b in bounds) + 5, width)
    side = int(max(y1 - y0, x1 - x0) * padding_ratio)
    top, left = (side - (y1 - y0)) // 2, (side - (x1 - x0)) // 2
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(image_size),
                                    transforms.Normalize([0.5] * 3, [0.5] * 3)])
    rgb_tensors, mask_tensors = [], []
    for rgb, mask in zip(images, masks):
        padded = np.full((side, side, 3), 255, dtype=np.uint8)
        padded_mask = np.zeros((side, side, 3), dtype=np.uint8)
        padded[top:top + y1 - y0, left:left + x1 - x0] = rgb[y0:y1, x0:x1]
        padded_mask[top:top + y1 - y0, left:left + x1 - x0] = mask[y0:y1, x0:x1, None]
        rgb_tensors.append(transform(padded))
        mask_tensors.append(transform(padded_mask)[:1])
    return {'image': torch.stack(rgb_tensors).unsqueeze(0),
            'mask': torch.stack(mask_tensors).unsqueeze(0), 'model_name': [sequence.name]}


def load_config(path, frames=FRAMES):
    import yaml
    config = yaml.safe_load(path.read_text())
    model = config['model']
    counts = [config['frames'], model['denoiser_cfg']['params']['num_frames'],
              model['first_stage_config']['params']['num_frames'],
              model['pipeline_cfg']['params']['latent_shape'][0]]
    if counts != [frames] * 4 or model['pipeline_cfg']['params']['latent_shape'][1:] != [256, 64]:
        raise ValueError(f'The input, VAE, denoiser and latent sequence must all use {frames} frames and 256x64 latents.')
    if config['image_size'] != 518 or config['padding_ratio'] < 1:
        raise ValueError('Use image_size=518 and padding_ratio >= 1.')
    return config


def reference_mesh_path(sequence, root):
    object_path = sequence.name.rsplit('_', 2)[0]
    return root / object_path / f'frame_{sequence.frame_numbers[0]:04d}.ply'


def reference_surface(path, seed=0):
    """Research evaluation alignment: normalize the first GT pose and sample 2048 points."""
    import numpy as np
    import trimesh
    mesh = trimesh.load(str(path), force='mesh', process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f'Expected a ground-truth triangle mesh: {path}')
    lower, upper = mesh.bounds
    scale = (upper - lower).max() / 0.9
    if not np.isfinite(mesh.vertices).all() or scale <= 0:
        raise ValueError(f'Invalid ground-truth geometry: {path}')
    mesh.vertices = (mesh.vertices - lower) / scale
    points, _ = trimesh.sample.sample_surface(mesh, 2048, seed=seed)
    return points


def checkpoint_model(model_config, checkpoint, validate_only=False):
    """Check all tensor names/shapes, and load without allocating optimizer state."""
    import torch
    from accelerate import init_empty_weights
    from hy3dshape.utils.misc import instantiate_from_config

    config = copy.deepcopy(model_config)
    config['params'].pop('ckpt_path', None)  # Bypass the old permissive CUDA loader.
    with init_empty_weights():
        model = instantiate_from_config(config)
    # The published Lightning files include OmegaConf/NumPy training metadata.
    # Use only trusted checkpoints (the downloader verifies the release hashes).
    checkpoint_data = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    state = checkpoint_data.get('state_dict', checkpoint_data)
    # The diffusion Lightning checkpoint also stores frozen image/shape encoders.
    # Its motion denoiser is the model.* component, as in the research runner.
    if any(key.startswith('model.') for key in state):
        state = {key[len('model.'):]: value for key, value in state.items() if key.startswith('model.')}
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatched = [key for key in set(expected) & set(state) if expected[key].shape != state[key].shape]
    if missing or unexpected or mismatched:
        raise ValueError(f'Incompatible checkpoint {checkpoint}: '
                         f'missing={missing[:5]}, unexpected={unexpected[:5]}, shape_mismatch={mismatched[:5]}')
    # assign=True preserves memory-mapped CPU tensors until moved to the GPU.
    model.load_state_dict(state, strict=True, assign=True)
    if validate_only:
        return {'file': str(checkpoint), 'tensors': len(state), 'strict_load': True}
    return model.eval().requires_grad_(False)


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)
