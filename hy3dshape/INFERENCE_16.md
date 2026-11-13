# Sixteen-frame inference

Use the existing Mesh4D CUDA environment and Blender 4.0's `bpy` Python module.
The deformation VAE and motion denoiser must both use the newer 16-frame
checkpoint pair from [Hugging Face](https://huggingface.co/zerenjiang/Mesh4D_16).
[Download and verify them](../checkpoints/README.md) into `ckpt/16frames/`. The original six-frame `infer.py` remains available.

Run these commands from the repository root:

```bash
# Validate a sequence and checkpoint paths without CUDA or model downloads.
python hy3dshape/infer_16.py --input-dir /path/to/rgba_frames --dry-run

# Generate a textured canonical mesh, predict motion, and export an animated GLB.
python hy3dshape/infer_16.py \
  --input-dir /path/to/rgba_frames \
  --output-dir hy3dshape/output_folder/test/my_video_16
```

Input frames must be segmented RGBA PNGs with equal image dimensions. The
default filenames are `0.png` through `15.png`, or `front_frame0.png` through
`front_frame15.png`. Larger sequences are accepted; the command selects one
16-frame window. Use `--start-frame 40 --stride 1` to select frames 40–55, or
`--image-pattern '{frame:04d}.png'` for zero-padded names. It rejects missing
frames rather than silently shortening or repeating the video.

Preprocessing follows the reference dataset: composite onto white, use one
foreground crop across the whole sequence, pad by 1.15, resize to 518×518 and
normalize to [-1, 1]. The first image is also prepended as the canonical image,
so the batch contains 17 images and 16 motion targets.

`configs/infer_16.yaml` matches the reference `batt16_pretrain` model settings:
both models use 16 frames, with motion latents shaped `[16, 256, 64]`. The loader
checks every model tensor name and shape. It selects the denoiser's `model.*`
component from the full Lightning checkpoint and ignores the separately stored
frozen encoders. VAE loading also checks every model tensor. Both models run in
evaluation mode. Add `--validate-weights` to `--dry-run` to perform these checks
on CPU without loading the pretrained image encoders or generating a mesh.

Shape generation, painting, and motion inference run in separate stages so their
large models do not all occupy GPU memory together. The first two stages still
use the original Hunyuan3D pretrained assets and the RealESRGAN weights described
in the main setup instructions. Each inference batch contains one clip. Full
inference requires CUDA; a successful dry run does not measure peak GPU memory
or validate reconstruction quality.

Useful options:

| Option | Effect |
| --- | --- |
| `--vae-checkpoint PATH --denoiser-checkpoint PATH` | Use checkpoint files outside the default directory. |
| `--initial-mesh PATH` | Use an existing canonical mesh for a single selected clip, bypassing shape generation and painting. Supply the original Hunyuan canonical orientation, before the motion pipeline's coordinate conversion. |
| `--skip-texture` | Generate untextured geometry for a smoke test; appearance scores will not be comparable to textured output. |
| `--num-inference-steps 50 --guidance-scale 5 --seed 0` | Explicit defaults for motion sampling. |
| `--resume` | Resume completed stages with identical run settings, inputs and checkpoints. |

Each sequence produces:

```text
<output-dir>/<sequence-name>_<start>_<stride>/
├── textured_gen_mesh.obj          # canonical mesh, with associated texture files
├── gen/gen_00.obj ... gen_15.obj  # sixteen decoded motion poses
├── combined_gen_mtl.glb           # textured animation
└── inference.json                # source frames, animation timing and alignment
```

The GLB follows the research export: the canonical mesh is evaluated at Blender
frame 1, and `gen_01.obj` through `gen_15.obj` at frames 3, 5, …, 31. It uses
24 FPS and interpolation between poses. The last keyframe is included in the
export. The decoded `gen_00.obj` is retained on disk; the animated first pose
uses the canonical mesh, as in the reference. `run.json` records the configuration
and selected images. Existing outputs require an identical `--resume` run or a
new output directory.

## Benchmark inputs and alignment

```bash
python hy3dshape/infer_16.py \
  --benchmark-root data/mesh4d_benchmark \
  --output-dir hy3dshape/output_folder/test/benchmark_16
```

This selects the manifest's **48 clips from 37 objects**, including nonzero start
frames. Use `--clip 000-000/01f1cc659cb847a4a7366b84844641c2_000` for a single-clip
check. `--manifest` and `--input-root` allow the original benchmark image tree
to be used without copying it.

Image-only inference normalizes the generated canonical mesh without using
ground-truth geometry. For evaluation with the original per-frame PLY data,
add:

```bash
--reference-geometry-root /path/to/Motion2VecSets/obj_test_33
```

This explicitly enables the research pipeline's first-pose registration:
normalize the first ground-truth mesh using its minimum corner and longest
extent divided by 0.9, sample 2,048 surface points, then run rigid CPD with
`w=0.1` and `tolerance=0.01` before motion inference. Sampling is seeded here;
the older research loader used unseeded sampling, so alignment need not be
bit-for-bit identical. The PLYs are not included in the image/GLB benchmark
archives. Their expected path is `<root>/<group>/<object-id>/frame_<start:04d>.ply`.
The registration transform is saved in `gen_mesh/alignment.json`, and
`inference.json` records that ground truth was used. This option is for evaluation,
not inference on an unseen video.

Follow the [rendering and evaluation commands](../benchmark/README.md#sixteen-frame-inference-rendering-and-evaluation)
to obtain four-view appearance metrics. Geometry/tracking metrics still require
the separate per-frame ground truth and the original geometry evaluator.

## Checks

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s benchmark/tests -v
python -m unittest discover -s visualization/tests -v
```

Run these in the Mesh4D environment to include checkpoint loading, tensor
preprocessing, and Blender integration tests. Optional dependency tests are
reported as skipped in environments lacking PyTorch or `bpy`. The Blender test
exports and reimports all sixteen poses, including the last pose at frame 31.
