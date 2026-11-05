# Mesh4D benchmark

Run the commands below from the Mesh4D repository root with Python 3.10 or newer.

## Download and extract

Benchmark folder: **[Google Drive](https://drive.google.com/drive/folders/12kzy9YTctkNNIWELhXboiaM0vNhK9nFn)**.

Publication status, 2026-11-05: the archives have been prepared and verified
locally, but their presence in the Drive folder has not yet been verified.
The download command expects the following files to be published there.

| Archive | Contents | Download size |
| --- | --- | --- |
| `animated_glbs.zip` | 40 original animated GLBs | 126 MB |
| `input_images.zip` | 1,331 front-view input PNGs and 110 JSON metadata files | 943 MB |
| `evaluation_images.zip` | 3,168 ground-truth RGB PNGs and 250 JSON metadata files | 2.20 GB |

Allow about 7 GB for the archives and extracted data. Each archive includes the
same `manifest.json`, `objects.csv`, `clips.csv`, and `README.txt`. GLBs are the
complete source animations; a GLB can supply several evaluation clips.

In the existing Mesh4D environment, install the evaluation dependencies:

```bash
python -m pip install -r benchmark/requirements.txt
```

Once the files are published, download, verify, and extract with:

```bash
python benchmark/download.py
```

The downloader uses [gdown](https://github.com/wkentaro/gdown) and requires that
your account or the public link can download the files. If automated access
fails, download the three archives using the Drive web interface and run:

```bash
python benchmark/download.py --archive-dir /path/to/downloaded/archives --skip-download
```

The script verifies every archive against the repository's [SHA256SUMS](SHA256SUMS)
before extraction. To verify without extracting, add `--verify-only`. Use
`--output-dir /path/to/data` to change the destination. An existing extraction
is preserved unless `--overwrite` is explicitly supplied.

```text
data/mesh4d_benchmark/
├── manifest.json
├── objects.csv
├── clips.csv
├── glbs/<group>/<object-id>.glb
├── input_images/<group>/<object-id>/
│   ├── front_frame0.png
│   ├── front_frame1.png
│   └── ...
└── evaluation_images/<group>/<object-id>_<start:03d>/
    ├── front_frame0000.png
    ├── right_frame0000.png
    ├── back_frame0000.png
    ├── left_frame0000.png
    └── ... camera and object JSON files
```

Images and GLBs retain their original bytes, alpha channels, and filenames.
Normal-map renders, per-frame PLY ground truth, model predictions, and model
checkpoints are not part of these three archives.

## Objects and clips

The source collection has 40 objects. The six-frame split has **50 clips from
39 objects**: 33 objects contribute one clip, one contributes two, and five
contribute three. Start frames are spaced by 40, and frames within a clip have
stride one. One object has only one geometry frame and is excluded from both
evaluation splits, while its GLB and input image remain in the download.

| Protocol | Clips | Unique objects | Images scored across four views |
| --- | --- | --- | --- |
| Six frames | 50 | 39 | 1,200 |
| Sixteen frames | 48 | 37 | 3,072 |

Use the `six_frame_evaluation` or `sixteen_frame_evaluation` flags in
[clips.csv](clips.csv), also present in [manifest.json](manifest.json).
Do not derive the split by counting render files: the archived folders contain
9, 15, or 16 frames per view, and not every stored frame belongs to a protocol.
Ground-truth filenames use absolute benchmark frame numbers, so a clip starting
at 40 uses `front_frame0040.png`, not `front_frame0000.png`.

## Evaluate rendered predictions

The evaluator scores RGB renders. Render your reconstructed sequence using the
benchmark's front, right, back, and left cameras and matching frame timing,
normalization, and alignment. Camera JSON files accompany the reference images.
The evaluator does not render GLBs or align geometry for you.

The separate [visualization renderer](../visualization/README.md#use-benchmark-cameras)
can render animated Mesh4D GLBs using these camera JSON files and the filename
layout below. It uses fixed studio lighting; the original randomized benchmark
lighting is not stored in the camera files. Keep that difference in mind when
comparing RGB metrics, and use the released PNGs as ground truth.

The default prediction layout matches the original research evaluator:

```text
prediction_renders/<group>/<object-id>_<start>_1/combined_gen_mtl/
├── front_frame0001.png
├── front_frame0003.png
├── front_frame0005.png
└── ... plus right/back/left views
```

Prediction frame `2*t + 1` is compared with ground-truth frame `start + t`.
This accounts for the transition frames in the original animated Mesh4D output.
Use frame offsets appropriate to your renderer rather than renaming an arbitrary
animation frame to satisfy the pattern.

Check the complete split before loading any metric models:

```bash
python benchmark/evaluate.py \
  --benchmark-root data/mesh4d_benchmark \
  --pred-root /path/to/prediction_renders \
  --frames 6 --dry-run
```

Then evaluate:

```bash
python benchmark/evaluate.py \
  --benchmark-root data/mesh4d_benchmark \
  --pred-root /path/to/prediction_renders \
  --frames 6 --device cuda \
  --output-dir benchmark/results/mesh4d_6
```

Use `--frames 16` with sixteen-frame predictions to score the 48-clip split.
`--device auto` (the default) uses CUDA when available, otherwise CPU. CLIP and
LPIPS may download pretrained weights on first use. For a lightweight check
without model downloads, use `--metrics psnr`; this only needs NumPy and Pillow.
`--metrics psnr ssim` additionally needs PyTorch and TorchMetrics.

Alternate render names can be specified explicitly. For predictions numbered
0, 1, 2, ... directly inside `<group>/<object-id>_<start>_1/`, use:

```bash
python benchmark/evaluate.py \
  --benchmark-root data/mesh4d_benchmark \
  --pred-root /path/to/prediction_renders \
  --pred-template '{object_path}_{start}_{stride}/{view}_frame{pred_frame:04d}.png' \
  --pred-frame-offset 0 --pred-frame-step 1 \
  --frames 6 --output-dir benchmark/results/another_method
```

Available template fields are `object_path`, `clip_path`, `glb`, `start`,
`start_frame`, `stride`, `view`, `t`, `pred_frame`, and `gt_frame`.
Use `--manifest /path/to/manifest.json` and `--gt-root /path/to/reference_renders`
to evaluate existing directories without rearranging them.

Missing images cause an error before scoring. `--allow-missing` explicitly skips
whole incomplete clips and records them in the result. Such a result is a subset
evaluation, not the full benchmark. `--clip <clip_path>` selects an explicit clip
and can be repeated for debugging. Existing result files require `--overwrite`.

## Metrics and outputs

Default metrics are PSNR (higher is better), SSIM (higher), VGG LPIPS (lower),
and CLIP ViT-B/32 image-embedding cosine similarity (higher). Images are resized
to 512 x 512 with bicubic interpolation. As in the original code, pixels with
alpha exactly zero become white; other RGB pixels are retained.

The default PSNR/SSIM data range follows the original evaluator: it is inferred
from each image pair (PSNR uses the target range). Use `--data-range 255` for
a fixed range, and record this change when comparing results. Exact identical
images have infinite PSNR; JSON stores infinity as the string `"Infinity"`.
PSNR uses NumPy double precision, so tiny numerical differences from the old
Torch float32 implementation are expected.

The script writes:

- `per_frame.csv`: one row per clip/view/frame for image metrics.
- `per_clip.csv`: averages over frames and the four views for each clip.
- `summary.json`: equal-weight averages across clips, coverage, skipped files,
  and the evaluation settings. No NaN scores are silently dropped.

The two frame protocols have different clip sets. Match frame count, selected
clips, rendering/alignment, preprocessing, and metric settings when comparing
methods or historical CSVs.

### Historical I3D score

The old `eval_vid.py` called its per-clip I3D classifier-logit distance `fvd`.
It computes a score separately for each clip/view and then averages these
scores. It is not a distribution-level FVD over the dataset. To retain this
calculation without confusing the two metrics, the optional output is named
`legacy_fvd`:

```bash
python benchmark/evaluate.py \
  --benchmark-root data/mesh4d_benchmark \
  --pred-root /path/to/prediction_renders \
  --metrics psnr ssim lpips clip legacy_fvd \
  --i3d-weights /path/to/i3d_pretrained_400.pt \
  --output-dir benchmark/results/mesh4d_6_legacy
```

This requires the original 400-class I3D checkpoint, which is not bundled with
the image/GLB archives. `inception3d.py` preserves the implementation from the
research checkout. A different checkpoint changes the score. For six frames,
the old `fvd` and `fvd_6` columns used identical inputs; only one is emitted here.
The scalar covariance behavior of the original NumPy 1.24 implementation is
explicitly preserved, including when running with NumPy 2.

### Geometry and tracking

This entry point evaluates rendered appearance. It does not reproduce Chamfer,
occupancy IoU, or correspondence L2 directly from the source GLBs. Those research
metrics need consistently indexed per-frame ground-truth meshes, the matching
coordinate normalization, and an explicit alignment protocol. The original
geometry evaluator is in `hy3dshape/im2mesh/eval.py`; the per-frame PLY sequences
and occupancy queries are not part of this image/GLB release. A static GLB load
does not evaluate its animation.

## Local checks

Run the small dependency-light regression suite with:

```bash
python -m unittest discover -s benchmark/tests -v
```

The tests cover clip/frame mapping, split selection, missing predictions,
metric edge cases, and archive validation. Full learned-metric evaluation
requires the dependencies and pretrained weights described above.
