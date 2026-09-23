# Tissue Micro-Array (TMA) Preprocessing

Cabana can turn a whole-slide scan of a tissue micro-array into one image and
one binary mask per core, named by patient, ready for batch analysis. The
step lives on the **TMA** page of the GUI and in the `cabana-tma` command.

## What it does

1. **Load slide**: Olympus VS200 `.vsi` scans are read natively (no Java or
   Bio-Formats needed) from the `.vsi` file and its `_<name>_/stack*/` tile
   folders. Plain whole-slide TIFF/PNG exports are accepted too. Bright-field
   and polarised layers of a `.vsi` are exposed as the channels `BF` and `POL`.
   As soon as a slide is chosen a low-resolution preview appears in the image
   panel and the pixel size and available channels are filled in.
2. **Fit cores**: tissue is thresholded on a low-resolution level, a circle is
   fitted to every core, the circles are snapped to the array grid, and the
   grid is matched to the printed ICGC/APGI array map (arrays 1 to 8 are
   bundled). A numbered overlay is shown: green circles will be exported,
   purple were recovered at empty grid positions (also exported), grey with a
   cross were excluded by the quality filters, and grey lie outside the printed
   map; neither grey kind is exported.
3. **Export cores**: for every core and selected channel a square crop is
   written to `Images/` and a circular mask (white inside the fitted circle,
   shrunk by *Mask Shrink*, black in the corners) to `Masks/`. A `cores.csv`
   manifest and `overlay.png` are written alongside.

The three stages run independently: fitting can be repeated with new settings
before exporting, and exporting can be repeated into a different folder.

## How core fitting works and how to tune it

1. **Tissue mask.** On a level of about 4 µm/px, a pixel is tissue when its
   HSV saturation exceeds *Sensitivity* (default 15) or it is darker than
   96.5% of the median brightness of its image column (the per-column
   reference cancels scanner banding).
2. **Clean-up.** Gaps of up to 30% of a core diameter are closed so a core
   becomes one blob; an opening of 10% of a core diameter then cuts off thin
   structures such as coverslip edges, scratches and streaks.
3. **Components.** Blobs smaller than 5% of the largest are dropped. Inside
   each blob detached debris smaller than 5% of the main tissue mass is
   removed, and the minimum enclosing circle of what remains is the core.
   Circles larger than 1.5 or smaller than 0.2 nominal core diameters are
   rejected (fused neighbours, dust).
4. **Grid.** Circle centres are snapped to a lattice whose pitch is the
   median neighbour distance; fragments falling in one cell are merged.
5. **Recovery.** With *Recover faint* on, every empty grid position is tested
   with a permissive threshold (half the saturation, twice the brightness
   margin). If at least 3% of the expected disc is tissue, a core is added
   there with the median radius and flagged `recovered` (purple on the
   overlay). Positions with less tissue stay empty.
6. **Fill.** The fraction of each circle covered by tissue is recorded in
   `cores.csv` (`fill`) for reference.

Tuning: lower *Sensitivity* (for example 8) when very pale cores are missed
and debris is not a problem; raise it (25 to 30) when shading or dust on the
glass is being fitted. *Core Ø* sets the scale of every morphological step
and of the size gates, so set it to the real core diameter first. On the
command line the same knobs are `--sat-thresh`, `--val-ratio`, `--min-fill`
and `--no-recover`.

## Quality filters

After the cores are fitted and patient IDs assigned, each core is measured and
checked against the **Filter** settings. Excluded cores stay in `cores.csv`
(columns `excluded`, `reason`, plus the measured `grid_offset`, `diameter_um`,
`fill`, `stain_frac`), are drawn grey with a cross on the overlay, and are not
exported. Filters update instantly; no refit is needed.

| Setting | Excludes a core when | Default |
|---|---|---|
| Grid Offset | its centre is further than this from its grid position, in core spacings | 0.35 |
| Min Ø / Max Ø | its fitted diameter is outside this range, as % of Core Ø | 70 % / 120 % |
| Min Stain | less than this fraction of the circle has HSV saturation above 40, i.e. the core is empty or unstained (control cores exempt) | 2 % |

Filters never change patient IDs. The orientation match ignores only objects
more than half a core spacing from any grid position, a fixed rule, and all
filters run after IDs are assigned. The status line reports how many cores each
filter removed, the median fitted diameter (with a hint when Core Ø differs by
more than 20 %), and any patient left with no core.

On the APGI slides the weakest genuine patient cores have 3 to 5 % stained
area and the Brain and Muscle controls about 0 to 1 %, which is why the stain
default is 2 %. Core Ø defaults to 1250 µm, the fitted size of the APGI
cores; the diameter range is only meaningful when Core Ø matches the array, so
change it for other arrays (the status line warns when the median fitted
diameter differs from Core Ø by more than 10 %). Min Ø is 70 % so partial or
torn cores, which fit a smaller circle, are kept.

CLI: `--max-grid-offset`, `--diameter-range MIN MAX`, `--min-stain`, and
`--stain-sat` (the saturation threshold, 40, not exposed in the GUI).

## Output naming

```
<slide>_<position>_<patientID>_<icgcID>_<channel>.png   e.g. TMA1_A3_8010718_1734_BF.png
<slide>_<position>_<tissue>_<channel>.png               e.g. TMA1_A1_Liver_BF.png   (control cores)
<slide>_r<row>c<col>_unknown_<channel>.png              (no array map selected or unmapped cell)
```

`position` is the flat map position (rows A to L, columns 1 to 8) after the
printed sector offsets are resolved. `cores.csv` records the map label,
sector, patient ID, ICGC ID, tissue, circle centre and radius (level-0 pixels),
tissue fill, QC metrics, flag, exclusion reason and the orientation used.

## Orientation

The scan is typically rotated relative to the printed map. *Auto* compares the
pattern of missing cores with the map for all eight rotations and mirror
images. A fully populated array cannot distinguish a rotation from its mirror
image; when several orientations fit equally well the status line lists them.
Check a few control cores on the overlay against the map (for example the
collagen-free Brain core) and, if the labels are mirrored, choose the
orientation explicitly and fit again.

## Using the export in Cabana

`Images/` is the input folder and `Masks/` the ROI-mask folder of **Batch
Run** (the GUI offers both after an export). With a mask, the corners of each
crop are excluded from segmentation, fibre detection, HDM, fibre-area and gap
metrics, and the percentage metrics are relative to the circle area. The
masks are honoured whether segmentation is enabled or not.

Cores scanned at 20x are about 4000 to 5000 pixels across. Raise
`Segmentation: Max Size` in the parameter file (for example to 6000) so that
each core is analysed as one image, and consider `Segmentation: Patch Size`
(for example 1024) so the segmentation network sees the tissue at a useful
resolution instead of the whole core shrunk to 512 pixels.

## Command line

```bash
cabana-tma "APGI TMA 1 PicRed.vsi" out/TMA1 --array 1 --slide-name TMA1
cabana-tma slide.tif out/slide --pixel-size 0.5 --fit-only        # cores.csv + overlay only
cabana-tma slide.vsi out/x --array 4 --orientation 90+flip --channels BF
```

Run `cabana-tma --help` for all options (core diameter, crop margin, mask
shrink, channel selection).

## Python API

```python
from cabana import TMAPreprocessor

pre = TMAPreprocessor("APGI TMA 1 PicRed.vsi", array_number=1, slide_name="TMA1")
pre.fit()                      # list of Core objects with centre, radius, grid position
pre.map_to_array()             # orientation, sets map positions; pre.orientation_ties lists ambiguities
pre.export("out/TMA1", channels=["BF", "POL"])
```
