# Python API

Everything the GUI does is available from Python. The package exports three
classes:

- `Cabana`: analyse one image.
- `BatchProcessor`: analyse a folder of images in batches, with checkpoints.
- `TMAPreprocessor`: cut a tissue micro-array slide into per-core images and
  masks.

All three take the same YAML parameter file the GUI exports (see
[Parameter Details](parameters.md)). Activate the environment Cabana was
installed in first.

## One image

```python
from cabana import Cabana

analyzer = Cabana(
    "Parameters.yml",
    "data/Picrocirius/540.vsi - 20x_BF multi-band_01Annotation (Ellipse) (Tumor)_0.tif",
    "out/single",
    ignore_large=True,      # False: analyse the top-left block of an oversized image (BatchProcessor analyses every block)
    mask_dir=None,          # optional folder of ROI masks named like the image
)
ok = analyzer.run()         # False when the image was skipped (too dark, too small)
if ok:
    analyzer.export_results()          # <stem>_QuantificationResults.csv
    print(analyzer.stats.T)            # the same numbers as a DataFrame
```

The output folder receives the same sub-folders as a batch run (`ROIs`,
`Masks`, `Exports`, `Colors`, `HDM`, …); see
[Cabana Outputs](workflow.md#cabana-outputs).

## A folder of images

```python
from cabana import BatchProcessor

bp = BatchProcessor(
    batch_size=5,
    param_file="Parameters.yml",
    input_folder="cores/BF/Patients/Images",
    output_folder="cores/BF/Patients/Output",
    mask_dir="cores/BF/Patients/Masks",   # optional ROI masks
    ignore_large=False,
    generate_stats=True,                  # QuantificationResults_MEAN_STD_SEM.csv
    generate_scores=True,                 # QuantificationResults_SCORES.csv
)
bp.progress_callback = lambda pct: print(f"{pct}%")
bp.status_callback = print              # "Batch 1/9: Segmenting … (1/5)"
bp.run()
```

When `param_file`, `input_folder` or `output_folder` is omitted the class asks
for them with file dialogs, and it offers to resume when the output folder
holds a checkpoint from an interrupted run. `cabana.batch.BatchProcessor` is
the same class without the dialogs; it takes `resume=True` to continue from a
checkpoint programmatically.

## TMA slides

```python
from cabana import TMAPreprocessor

pre = TMAPreprocessor("APGI TMA 4 PicRed.vsi", array_number=4, slide_name="TMA4")
cores = pre.fit()                 # Core objects: centre, radius, grid position, fill
pre.map_to_array()                # orientation and map positions (patient IDs)
print(pre.orientation_method, pre.orientation_ties)
if not pre.orientation_resolved:  # tie that neither control nor replicates resolved
    pre.orientation = "270"       # set it explicitly after checking a control core
    pre.map_to_array()
pre.export("cores", channels=["BF", "POL"])
pre.close()
```

Fit and filter settings are constructor arguments with the same names and
defaults as the `cabana-tma` command line options (`core_diameter_um`,
`sat_thresh`, `min_fill`, `max_grid_offset`, `min_diameter_frac`,
`max_diameter_frac`, `min_stain_frac`, …). `export()` writes
`<channel>/<Patients|Controls|Unmapped>/{Images,Masks}/`, `cores.csv` and
`overlay.png`; the `Images` and `Masks` folders of a group are the
`input_folder` and `mask_dir` of a `BatchProcessor` run.

## Where the numbers go

- `QuantificationResults.csv`: one row per analysed image (or block of a
  split image), columns described in [Read-outs](readouts.md).
- `QuantificationResults_MEAN_STD_SEM.csv`: per-patient mean, standard
  deviation and standard error over the replicate images of a patient (see
  [Image naming](workflow.md#image-naming-for-per-patient-statistics)). Patients
  are identified from the image name: the prefix before `.vsi` (which is how
  both QuPath slide exports and the TMA core export name their files, for
  example `8010718.vsi - TMA1_BF_A3Annotation (Tumor)_1`), otherwise the
  first token of the name.
- `QuantificationResults_SCORES.csv`: Rigidity and Bundling collagen risk
  scores computed from the per-patient means.
- `version_params.yaml`: the parameters, Cabana version, git commit and the
  folders (including the ROI mask folder) used for the run.

## Choosing the compute device

The segmentation network runs on CUDA when available, on the Apple GPU (MPS)
on macOS, else on the CPU. Set the environment variable `CABANA_DEVICE` to
`cpu`, `cuda` or `mps` before importing `cabana` to override the choice.
