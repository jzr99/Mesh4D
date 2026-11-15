# Long-video inference

For 16-frame and longer videos, the default is to run the original six-frame
model chunk-wise. `hy3dshape/infer_long.py` covers the video with six-frame
windows that overlap by one frame: frames 0–5, 5–10, 10–15, and so on. The
first window starts from the canonical mesh. Each later window starts from the
pose that the previous window predicted for its last frame. A final window
shorter than six frames repeats its last frame, and the poses predicted for the
repeated frames are discarded. In our tests on 16-frame clips, chaining the
six-frame model gave lower geometry and motion errors than the single-pass
[sixteen-frame R3 model](INFERENCE_16.md), which remains available.

It uses the six-frame checkpoints from the setup in the [main README](../README.md),
`ckpt/deform_vae.ckpt` and `ckpt/denoiser.ckpt`, with
[`configs/infer_6.yaml`](configs/infer_6.yaml). That file is the model section of
`configs/infer.yaml`, in the layout of `infer_16.yaml`; a test checks that both
describe the same model. Use `--vae-checkpoint` and `--denoiser-checkpoint` for
other locations.

Run these commands from the repository root:

```bash
# Validate frames, windows, paths and config without CUDA or model downloads.
python hy3dshape/infer_long.py --input-dir /path/to/rgba_frames --num-frames 16 --dry-run

# Generate a textured canonical mesh, predict 16 poses chunk-wise, and export an animated GLB.
python hy3dshape/infer_long.py --input-dir /path/to/rgba_frames --num-frames 16 \
  --output-dir hy3dshape/output_folder/test/my_video

# Use every consecutive frame from the start frame.
python hy3dshape/infer_long.py --input-dir /path/to/rgba_frames --num-frames all \
  --output-dir hy3dshape/output_folder/test/my_long_video
```

Inputs follow the [sixteen-frame guide](INFERENCE_16.md): segmented RGBA PNGs
with equal image dimensions, named `0.png`, `1.png`, … or `front_frame0.png`, …
by default. `--image-pattern '{frame:04d}.png'` selects other names, and
`--start-frame` / `--stride` select the frames. `--num-frames` takes any number
from 2; missing frames are an error. With `--num-frames all`, counting stops at
the first missing frame.

Each window is prepared and sampled like a six-frame clip. The window's first
frame is prepended as its canonical image, one foreground crop covers the
window's frames, and every window uses the same seed. The motion pipeline
normalizes each window's start mesh by its bounding box. The script maps every
window's poses back into the first window's coordinates, so the sequence is
continuous and in the coordinates of a six- or sixteen-frame run. It also checks
that each window starts exactly from the pose it was given.

Useful options:

| Option | Effect |
| --- | --- |
| `--num-frames N` or `all` | Number of frames to reconstruct. |
| `--initial-mesh PATH` | Use an existing canonical mesh for a single selected clip, bypassing shape generation and painting. Supply the original Hunyuan canonical orientation. |
| `--skip-texture` | Generate untextured geometry for a smoke test. |
| `--num-inference-steps 50 --guidance-scale 5 --seed 0` | Motion sampling defaults, applied to every window. The seed fixes both the sampling noise and the mesh points sampled for the model's shape inputs, so a run is repeatable on the same hardware. |
| `--keep-chunks` | Keep each window's own pipeline outputs in `<sequence>/chunks/`. |
| `--resume` | Resume completed sequences with identical run settings, inputs and checkpoints. |

Each sequence produces the same layout as `infer_16.py`:

```text
<output-dir>/<sequence-name>_<start>_<stride>/
├── textured_gen_mesh.obj             # canonical mesh, with associated texture files
├── gen_mesh/registered_gen_mesh.obj  # canonical mesh in the output coordinates
├── gen/gen_00.obj ... gen_15.obj     # one pose per frame
├── combined_gen_mtl.glb              # textured animation
└── inference.json                    # source frames, windows, animation timing and alignment
```

The GLB evaluates the canonical mesh at Blender frame 1 and the predicted poses
at frames 3, 5, …, 2N−1, at 24 FPS. As in the sixteen-frame output, `gen_00.obj`
is the model's decoding of the canonical mesh, while the animation's first pose
is the canonical mesh itself. Pose filenames use three digits from 101 frames.
`inference.json` records each window's frames, the number of repeated frames, its
normalization, and how far the model's decoding of the start pose lies from that
pose, relative to the mesh extent.

## Benchmark inputs

```bash
python hy3dshape/infer_long.py \
  --benchmark-root data/mesh4d_benchmark --num-frames 16 \
  --output-dir hy3dshape/output_folder/test/benchmark_16
```

`--num-frames 16` selects the manifest's 48 sixteen-frame clips. `--num-frames 6`
selects the 50 six-frame clips; each clip is then a single window, which is the
original six-frame inference. The output layout matches `infer_16.py`, so the
[rendering and evaluation commands](../benchmark/README.md#sixteen-frame-inference-rendering-and-evaluation)
apply unchanged. With `--reference-geometry-root`, the first-pose CPD
registration described in the sixteen-frame guide aligns the first window; later
windows continue in its registered coordinates.

## Checks

```bash
python -m unittest tests/test_inference_long.py -v
```

The tests cover the window plan, short final windows, the coordinate round
trip, the configuration's match with `configs/infer.yaml`, the benchmark splits
and a CPU dry run.
