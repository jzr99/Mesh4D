# Render and visualize Mesh4D outputs

Run these commands from the repository root. The renderer accepts animated
`.glb`/`.gltf` files (including Mesh4D's `combined_gen_mtl.glb`) and static `.obj`
files with their MTL/textures. It renders transparent PNG sequences without
loading model checkpoints or PyTorch. A separate tool makes GIF and MP4 previews.

## Requirements

Use Blender 4.0, either its command-line executable or an environment containing
the `bpy` Python module. CPU rendering is the default. The preview tool needs
Pillow; MP4 output additionally requires `ffmpeg` with the `libx264` encoder.
Pillow is already included in the Mesh4D and benchmark requirements.

## Render an animated result

Mesh4D inference writes results under `hy3dshape/output_folder/test/infer/`.
For example, the existing blue jay export contains an extra mesh named `Cube`,
which can be excluded explicitly:

```bash
blender --background --python-exit-code 1 --python visualization/render.py -- \
  --input hy3dshape/output_folder/test/infer/in-the-wild/pecking_blue_jay_0_1/combined_gen_mtl.glb \
  --output-dir visualization/results/blue_jay --exclude-object Cube
```

If `bpy` is installed in your Python environment, the equivalent command is:

```bash
python visualization/render.py \
  --input /path/to/combined_gen_mtl.glb \
  --output-dir visualization/results/example
```

The default output has four fixed views, 512 x 512 pixels, 64 Cycles samples,
imported materials, and deterministic studio lighting:

```text
visualization/results/example/
├── front_frame0000.png
├── right_frame0000.png
├── back_frame0000.png
├── left_frame0000.png
├── ...
└── render.json
```

Frames use their actual Blender timeline indices, padded to at least four
digits. `render.json` records the selected frames, camera matrices and fields
of view, normalization transform, Blender version and render settings.
Existing outputs require `--overwrite`.

Useful options:

| Option | Effect |
| --- | --- |
| `--frame-start 1 --frame-end 11 --frame-step 2` | Render frames 1, 3, 5, 7, 9, 11; bounds are inclusive. |
| `--views front` | Render only the front camera. |
| `--resolution 256 --samples 16` | Faster previews. |
| `--device CUDA` or `--device OPTIX` | Use a compatible GPU; unavailable devices cause an error. |
| `--shading clay` | Replace materials with a uniform blue-gray material to inspect geometry. |
| `--exclude-object Cube` | Hide an unwanted imported mesh by exact name; repeat to exclude more meshes. |
| `--turntable` | Orbit once while playing the selected animation frames. |
| `--normalize first-frame` | Fit the first selected pose into a unit cube. |
| `--normalize none` | Preserve imported coordinates. |

By default, `--normalize sequence` measures the evaluated mesh bounds across
all selected frames and applies one fixed transform to the entire animation.
This preserves motion and avoids camera framing changes between frames. The
default cameras are at front `(0, -2, 0)`, right `(2, 0, 0)`, back `(0, 2, 0)`
and left `(-2, 0, 0)` in Blender coordinates, aimed at the origin with a 50 mm
lens and 36 mm sensor.

The default timeline comes from the imported animation actions; static meshes
render frame zero. Blender's active imported animation is used. For GLBs with
multiple alternative animation clips, export the desired clip separately first.
`--fps 24` is set **before** glTF import because glTF animation times are stored
in seconds. Keep this at 24 for Mesh4D exports. Explicit frame bounds can extend
beyond the last keyframe, where Blender evaluates the animation's extrapolation.

For a static mesh turntable, supply a frame range:

```bash
blender --background --python-exit-code 1 --python visualization/render.py -- \
  --input /path/to/mesh.obj --output-dir visualization/results/turntable \
  --turntable --frame-start 0 --frame-end 47 --shading clay
```

## Make a GIF or MP4

```bash
python visualization/preview.py \
  --input-dir visualization/results/example \
  --output visualization/results/example.gif --fps 12

python visualization/preview.py \
  --input-dir visualization/results/example \
  --output visualization/results/example.mp4 --fps 12
```

The preview places views side by side with frame labels, composites transparency
onto white, and uses `--tile-size 256` per view by default. Use `--views front`
for a single-view preview. Playback FPS is independent of the renderer's import
FPS: for source frames sampled every two steps at 24 FPS, use 12 FPS here.

For directories produced by the renderer, `render.json` determines the exact
frame list. Older directories with names such as `front_frame0001.png` also
work without a manifest. Missing views or frames cause an error instead of
silently shortening the preview. GIFs are assembled in memory; use MP4 for long
sequences, which are encoded from temporary PNGs.

## Use benchmark cameras

The [benchmark evaluator](../benchmark/README.md) consumes four-view images.
To use a reference clip's saved camera poses and fields of view, pass its
evaluation-image directory:

```bash
blender --background --python-exit-code 1 --python visualization/render.py -- \
  --input '/path/to/predictions/<group>/<object-id>_<start>_1/combined_gen_mtl.glb' \
  --output-dir '/path/to/prediction_renders/<group>/<object-id>_<start>_1/combined_gen_mtl' \
  --camera-dir 'data/mesh4d_benchmark/evaluation_images/<group>/<object-id>_<start-padded-to-3-digits>' \
  --normalize none --frame-start 1 --frame-end 11 --frame-step 2
```

The input geometry must already be aligned, oriented and scaled to those
cameras. `--camera-dir` requires `--normalize none` to keep that alignment.
For sixteen-frame Mesh4D output, use `--frame-end 31` with the same start and
step. Check that these timeline indices correspond to your model's predictions;
the original evaluator pairs prediction `2*t + 1` with reference `start + t`.

For raw `infer_16.py` output, also add `--mesh4d-coordinates`. This reproduces
the research prediction renderer's normalization of the canonical pose at frame
0 followed by a -90-degree X rotation. It requires `--normalize none` to avoid
applying a second normalization, and records the transform in `render.json`.
Omit this flag for geometry already transformed into the saved camera coordinates.
The [benchmark batch renderer](../benchmark/README.md#sixteen-frame-inference-rendering-and-evaluation)
applies this convention by default and renders the entire selected split.

This provides the camera and filename interface to the evaluator. It does **not**
reproduce the original benchmark lighting: the research rendering scripts in
Motion2VecSets randomized their lights, and the archived camera JSON files do
not record those lights. This renderer uses fixed studio lighting, so RGB scores
depend on that difference. Use the released reference PNGs for evaluation;
rendering the source GLBs here does not regenerate the exact reference images.

If an old export contains a default Blender cube that obscures the reconstruction,
use `--exclude-object Cube`. Excluded meshes are omitted from both rendering and
bounds calculation; their names are recorded in `render.json`. Names must exist,
and no objects are excluded automatically. This does not modify the input file.

## Existing Hunyuan visualization helpers

Inspection of Hunyuan3D-2.1 found these helpers already present, byte-for-byte,
in this repository:

- `hy3dshape/tools/render/render.py`: static multiview rendering with depth,
  normals, albedo and material passes; intended for preprocessing.
- `hy3dshape/utils/render_utils.py`: mesh, joint, skinning, turntable and
  animation utilities with CUDA and research-environment assumptions.
- `hy3dshape/hy3dshape/utils/visualizers/`: notebook/HTML mesh visualization.

The standalone command follows their Blender scene/camera/animation workflow,
with explicit input/output paths, CPU support and animation-aware framing.
The original helpers and their license notices remain in their existing files.

## Validation

```bash
python -m unittest discover -s visualization/tests -v
```

The Blender integration test runs when `bpy` is available. It exports an
animated fixture, renders actual PNGs, and checks motion, normalization, camera
metadata and overwrite protection. Run the same command in your `bpy`
environment to include it; otherwise it is reported as skipped.
