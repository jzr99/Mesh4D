# Sixteen-frame checkpoints

The 16-frame VAE and R3 denoiser are available as complete files in
[zerenjiang/Mesh4D_16 on Hugging Face](https://huggingface.co/zerenjiang/Mesh4D_16).
Use them together with [infer_16.yaml](../hy3dshape/configs/infer_16.yaml).

| Component | Filename | Training step | Size |
| --- | --- | --- | --- |
| Deformation VAE | `vae_bone_r2_16_00001500.ckpt` | 1,500 | 3.10 GiB |
| R3 motion denoiser | `diffuse_r3_ckpt-step=00003000.ckpt` | 3,000 | 22.71 GiB |

Each checkpoint is a single `.ckpt` file; no chunk reassembly is required.
The weights total 25.81 GiB and preserve the original tensors and training
metadata. These are the pair selected by Hunyuan3D-2.1's
`minimal_video_with_ckpt_pretrain_predict_bone_test_texture_batt_16_pretrained.py`;
the VAE is also known locally as `vae_bone_r2_ckpt-step=00001500.ckpt`.

Run from the repository root:

```bash
python -m pip install 'huggingface-hub==0.30.2'
python checkpoints/download.py
```

The client version matches the main `requirements.txt`. The public release
can be downloaded without a Hugging Face login.
[manifest_16.json](manifest_16.json) pins the uploaded release commit and records
the full-file sizes and SHA-256 checksums. The downloader verifies each complete
file before placing it under `ckpt/16frames/`:

```text
ckpt/16frames/
├── vae_bone_r2_16_00001500.ckpt
└── diffuse_r3_ckpt-step=00003000.ckpt
```

Interrupted downloads and their metadata are retained under
`ckpt/16frames/.download/`; rerun the same command to resume.
Completed checkpoints are verified and reused. Existing or downloaded files
with incorrect sizes or checksums cause an error; remove or replace only the
reported bad file before retrying.

You can also download the two complete files from the Hugging Face repository
and place them in `ckpt/16frames/`. Verify them without downloading or writing:

```bash
python checkpoints/download.py --verify-only
```

Use `--output-dir` to choose another checkpoint directory, then pass
`--vae-checkpoint` and `--denoiser-checkpoint` to the inference command.
These are full Lightning checkpoints containing pickled training metadata;
load checkpoints from trusted sources. Downloaded weights and the default
download cache are ignored by Git.

## Switching from R6

R3 is now the default sixteen-frame denoiser; the training-run label does not
refer to the number of frames. Rerun the downloader to obtain R3 and reuse the
same verified VAE. Existing R6 files can remain on disk, and R6 remains available
on Hugging Face for reproducing previous experiments.

Use a new inference output directory, for example:

```bash
python hy3dshape/infer_16.py --input-dir /path/to/rgba_frames \
  --output-dir hy3dshape/output_folder/test/my_video_16_r3
```

Do not resume an R6 run with R3: inference checks that saved checkpoint paths
match. Update any explicit `--denoiser-checkpoint` arguments to the R3 filename.

The change follows an audit of 48 stored research sequences:

| Denoiser | Median frame-0 offset | Median frame-1 similarity residual |
| --- | --- | --- |
| R3 | 0.40% | 0.96% |
| R6 | 12.94% | 11.69% |

Both measurements are percentages of the canonical mesh's longest side. The
frame-1 residual removes the best global scale, rotation and translation; it
can include genuine deformation. These were existing research outputs using
ground-truth registration and Hunyuan canonical meshes. This audit does not
establish six-frame performance parity or validate a fresh run on other
canonical meshes.

See [sixteen-frame inference](../hy3dshape/INFERENCE_16.md) and the
[benchmark workflow](../benchmark/README.md#sixteen-frame-inference-rendering-and-evaluation).
