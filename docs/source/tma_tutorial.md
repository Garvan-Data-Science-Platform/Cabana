# TMA Tutorial: from slide to results

This walkthrough takes one Olympus `.vsi` scan of an ICGC/APGI tissue
micro-array through to `QuantificationResults.csv`, using the GUI. It assumes
Cabana is [installed](installation.md) and that you know which printed array
map (1 to 8) the slide was built from. Every control mentioned here is
described in detail on the [TMA preprocessing](tma.md) page.

## 1. Cut the slide into cores

1. Start Cabana and click **Open TMA Slide…** on the Start page (or **File >
   Open TMA Slide**). Choose the `.vsi` file. A preview appears within a few
   seconds and the status line reports the slide size, pixel size and the
   channels found (`BF`, `POL`).
2. Set **Array Map** to the slide's array. For a TMA that is not one of the
   built-in ICGC/APGI arrays, choose **Load map file…** and pick your map (see
   [Array maps](tma.md#array-maps); **File > Generate TMA Map Template…** writes
   a template to type or paste it into). Check the map in the preview and
   click **Use This Map**. Leave **Orientation** on *Auto*.
3. Leave the fit settings at their defaults on a first pass (**Core Ø**
   *Auto*, **Sensitivity** 15, **Recover faint** on) and click **Fit Cores**.
   With *Auto* the measured core diameter appears in the Core Ø box.
   Fitting takes 5 to 60 seconds depending on the machine.
4. Read the status line under the settings. It reports the number of cores,
   the grid size, how many will be exported, the orientation and which rule
   chose it, how many cores each filter removed, the Core Ø used and any
   patient left without a core. Check the overlay: green circles are
   exported, purple were recovered at empty grid positions, grey with a cross
   were excluded by the filters, plain grey lie outside the printed map, and
   dashed circles mark map positions that hold no core. Each core is
   labelled with its map position and number; control cores also show their
   tissue, and hovering a core gives its patient ID and measurements.
   - If the status says the orientation is **unresolved**, look at the
     control cores (for example Brain, which is nearly unstained) and switch
     **Orientation** until their names land on the right tissue; the labels
     update instantly, no refit. Export is disabled until the orientation is
     resolved.
   - With a typed **Core Ø**, the status line warns when the median fitted
     diameter differs from it by more than 10 %; tick *Auto* or set Core Ø to
     the reported value and refit.
   - Torn or partial cores fit a smaller circle; lower **Min Ø** (for example
     70 %) to keep them. Filter changes apply instantly without a refit.
   - For the odd circle the fit got wrong, tick **Edit cores**: drag a circle
     to move it, drag its rim to resize it, double-click it to include or
     exclude it, double-click a dashed circle to add a core there. The core
     takes the patient ID of the cell it lands in (see
     [Editing cores by hand](tma.md#editing-cores-by-hand)).
5. Check the **Output Folder** (defaults to `<slide>_cores` next to the slide),
   tick the **Channels** to export and click **Export Cores**. A 20x slide
   with 80 cores exports in one to two minutes. The dialog at the end offers
   **Use BF in Batch Run** and **Use POL in Batch Run**.

The export folder now looks like this:

```
APGI_TMA_4_PicRed_cores/
  cores.csv               one row per core and per map position without a core: position, IDs, QC metrics, flag, group
  overlay.png             the annotated overlay
  cores_edits.json        the cores as exported; a later refit offers to reinstate hand edits from it
  BF/
    Patients/  Images/    2119.vsi - TMA4_BF_A3Annotation (Tumor)_1.png …
               Masks/     2119.vsi - TMA4_BF_A3Annotation (Tumor)_1.png … (white disc = analyse)
    Controls/  Images/, Masks/   Liver.vsi - TMA4_BF_A1Annotation (Liver)_1.png …
    Unmapped/  Images/, Masks/   cores in cells the map does not have, or marks ?
  POL/  (same layout)
```

## 2. Prepare the parameters

1. Open one exported core with **Open Image…** and tune the Segmentation,
   Fibre Detection and Gap Analysis pages on it as described in the
   [workflow](workflow.md). Bright-field cores use **Dark Line** on; polarised
   cores use it off.
2. Keep **Patch Size** on the Segmentation page at 1024 px (the default), so
   that the segmentation network sees the tissue at a useful resolution rather
   than the whole core shrunk to 512 px.
3. Export the parameters with **Parameters > Export Parameters**. The default
   Max Size (5000 px, shown on the Batch Run page) analyses a 4000 to 5000 px
   core as one image; lower it on a machine with little memory.

## 3. Run the batch

1. Click **Use BF in Batch Run** in the export dialog, or go to **Analysis >
   Batch Run** and select `BF/Patients/Images` as **Input Folder** and
   `BF/Patients/Masks` as **ROI Masks** yourself.
2. Select the parameter file from step 2 and an **Output Folder**.
3. Tick **Stats** and **Scores** to get per-patient statistics and collagen
   risk scores; the TMA core names give the patient IDs.
4. Click **Process Batch**. The status line above the progress bar names the
   batch, stage and image being processed. On an Apple M-series Mac a whole
   4400 px core takes about 2.5 minutes with nine line widths: segmentation
   about 13 seconds with 1024 px patches, fibre detection about 1.8 minutes
   and skeleton analysis about 16 seconds. A CPU-only machine is slower.
5. When the run finishes, **Open Folder** shows the results. The `Batches`
   folder and checkpoint are removed automatically. If the run stops with an
   error (for example a full disk) or is cancelled, they are kept: fix the
   cause and run again with the same output folder to resume.

Repeat step 3 for `POL/Patients` with the polarised parameter file, and for
the `Controls` folders if you want to compare staining between slides.

## 4. Read the results

In the output folder:

- `QuantificationResults.csv`: one row per core, columns explained in
  [Read-outs](readouts.md).
- `QuantificationResults_MEAN_STD_SEM.csv`: per patient, over the replicate
  cores.
- `QuantificationResults_SCORES.csv`: Rigidity and Bundling scores per
  patient.
- `Exports/<core>/`, `Colors/<core>/`: the per-core maps and visualisations;
  `Masks/GapAnalysis/`: gap images and per-gap tables.
- `version_params.yaml`: the parameters, Cabana version and folders used.

## Command line equivalent

```bash
cabana-tma "APGI TMA 4 PicRed.vsi" cores --array 4 --slide-name TMA4
cabana-tma "my slide.vsi" cores --map my_maps.csv     # your own map file
# then, in Python:
from cabana import BatchProcessor
BatchProcessor(param_file="Parameters_BF.yml",
               input_folder="cores/BF/Patients/Images",
               output_folder="cores/BF/Patients/Output",
               mask_dir="cores/BF/Patients/Masks",
               generate_stats=True, generate_scores=True).run()
```
