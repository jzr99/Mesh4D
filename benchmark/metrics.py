"""Image metrics from the original hy3dshape evaluation, with lazy model loading."""
import math
from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path, size=None):
    """Match the research evaluator: replace fully transparent pixels with white."""
    with Image.open(path) as image:
        rgba = np.array(image.convert('RGBA'))
    rgb = rgba[..., :3].copy()
    rgb[rgba[..., 3] == 0] = 255
    image = Image.fromarray(rgb)
    if size is not None:
        image = image.resize((size, size), Image.Resampling.BICUBIC)
    return image


def psnr(prediction, target, data_range=None):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mse = np.square(prediction - target).mean()
    if mse == 0:
        return float('inf')
    peak = float(target.max() - target.min()) if data_range is None else data_range
    return 10 * math.log10(peak * peak / mse) if peak > 0 else float('-inf')


def legacy_frechet_score(prediction, target):
    """Preserve the original per-clip I3D-logit calculation, not corpus FVD."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != (1, 400) or target.shape != (1, 400):
        raise ValueError('legacy_fvd expects one 400-class I3D logit vector per clip.')
    # In the research environment (NumPy 1.24), np.cov((1, 400), rowvar=False)
    # collapsed to scalar variance across classes. Newer NumPy instead returns
    # an undefined 400 x 400 covariance from one observation. Spell out the old
    # scalar Fréchet formula so this compatibility score remains reproducible.
    difference = prediction[0] - target[0]
    std1 = np.std(prediction[0], ddof=1)
    std2 = np.std(target[0], ddof=1)
    return float(difference.dot(difference) + (std1 - std2) ** 2)


class MetricComputer:
    def __init__(self, metrics, device='auto', image_size=512, data_range=None,
                 clip_model='openai/clip-vit-base-patch32', i3d_weights=None):
        self.metrics = set(metrics)
        self.image_size = image_size
        self.data_range = data_range
        self.device = 'cpu'
        self.torch = None
        if self.metrics - {'psnr'}:
            import torch
            self.torch = torch
            self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
            if self.device.startswith('cuda') and not torch.cuda.is_available():
                raise RuntimeError('CUDA is unavailable. Use --device cpu.')
        if 'ssim' in self.metrics:
            from torchmetrics.functional.image import structural_similarity_index_measure
            self.ssim = structural_similarity_index_measure
        if 'lpips' in self.metrics:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', reduction='mean').to(self.device).eval()
        if 'clip' in self.metrics:
            from transformers import CLIPImageProcessor, CLIPModel
            self.clip = CLIPModel.from_pretrained(clip_model).to(self.device).eval()
            self.clip_processor = CLIPImageProcessor.from_pretrained(clip_model)
        if 'legacy_fvd' in self.metrics:
            if i3d_weights is None or not Path(i3d_weights).is_file():
                raise ValueError('--i3d-weights must point to the original i3d_pretrained_400.pt checkpoint.')
            from .inception3d import InceptionI3d
            self.i3d = InceptionI3d(400, in_channels=3)
            self.i3d.load_state_dict(self.torch.load(i3d_weights, map_location='cpu', weights_only=True))
            self.i3d = self.i3d.to(self.device).eval()

    def image(self, pred_path, gt_path):
        prediction = load_rgb(pred_path, self.image_size)
        target = load_rgb(gt_path, self.image_size)
        scores = {}
        if 'psnr' in self.metrics:
            scores['psnr'] = psnr(prediction, target, self.data_range)
        if self.metrics & {'ssim', 'lpips'}:
            pred = self.torch.from_numpy(np.array(prediction)).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
            gt = self.torch.from_numpy(np.array(target)).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
            with self.torch.inference_mode():
                if 'ssim' in self.metrics:
                    scores['ssim'] = self.ssim(pred, gt, data_range=self.data_range).item()
                if 'lpips' in self.metrics:
                    scores['lpips'] = self.lpips(pred / 127.5 - 1, gt / 127.5 - 1).item()
                    self.lpips.reset()
        if 'clip' in self.metrics:
            inputs = self.clip_processor(images=[prediction, target], return_tensors='pt').to(self.device)
            with self.torch.inference_mode():
                features = self.clip.get_image_features(**inputs)
                scores['clip'] = self.torch.nn.functional.cosine_similarity(features[:1], features[1:]).item()
        return scores

    def video(self, paths):
        frames = []
        for path in paths:
            rgb = np.asarray(load_rgb(path), dtype=np.float32)
            # Float-channel resizing matches eval_vid.py; resizing uint8 differs.
            channels = [np.asarray(Image.fromarray(rgb[..., c]).resize((224, 224), Image.Resampling.BICUBIC))
                        for c in range(3)]
            frames.append(np.stack(channels, axis=-1))
        tensor = self.torch.from_numpy(np.stack(frames).copy()).float() / 127.5 - 1
        tensor = tensor.unsqueeze(0).permute(0, 4, 1, 2, 3).to(self.device)
        with self.torch.inference_mode():
            return self.i3d(tensor).cpu().numpy()

    def sequence(self, pred_paths, gt_paths):
        if 'legacy_fvd' not in self.metrics:
            return {}
        return {'legacy_fvd': legacy_frechet_score(self.video(pred_paths), self.video(gt_paths))}
