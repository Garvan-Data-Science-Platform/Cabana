# Changelog

## Unreleased

### Segmentation

- **Results change for colour images.** The colour threshold is now compared
  with each region's absolute colour score (1 = exactly the colour of
  interest) instead of a score rescaled to the image's, or in patch mode to
  each patch's, own range. With Patch Size set, the old per-patch rescaling
  marked faint or empty patches as collagen (68 % of an APGI Brain control
  core, which has almost none) and left square seams; results also depended
  on the patch size. Typical stained images change by a few points; weakly
  stained ones can drop a lot (an APGI tumour core from 48 % to 15 %).
  Results from runs with a Patch Size set should be re-run.
- Greyscale images (SHG, fluorescence) keep the earlier behaviour: they are
  scored by brightness relative to the image, rescaled once per image (never
  per patch). Unpatched greyscale results are unchanged.
- Patch mode trains the segmentation network once on the whole image and
  labels every patch with it, instead of training a network per patch: about
  11 s instead of 127 s for a 4400 px TMA core at 1024 px patches.
- Patch Size defaults to 1024 px; images up to 1024 px are unaffected.
- After a segmentation in the GUI, moving the Colour Threshold slider updates
  the result immediately, without segmenting again.

### Speed and memory

- A whole 4400 px TMA core is analysed about 7 times faster with the same
  settings (125 s instead of 857 s on an Apple M-series Mac) and with a lower
  memory peak (7.0 GB instead of 9.7 GB):
  - Skeleton analysis worked on the full image once per skeleton piece; it
    now works within each piece's bounding box (533 s to 16 s on a core).
  - Fibre detection keeps only each pixel's best line-width scale instead of
    full-size arrays for every scale, so its memory no longer grows with the
    number of line widths, and frees intermediate arrays between its steps.
    Width polygons are filled directly instead of through a full-size mask
    per fibre.
  - Results are unchanged: detected fibres, skeleton graphs and all metrics
    are identical, except fibre widths, which differ by at most 0.03 px (and
    in all but a handful of points by less than 0.001 px) because the
    per-scale gradients used to locate fibre edges are stored in single
    precision.

### Storage

- A batch of whole TMA cores needs about a fifth of the storage (300 MB per
  4400 px core instead of 1.35 GB; the measurements are unchanged):
  - The maps in `Exports/` (Energy, Coherency, Orientation, curvature and
    length maps, colour survey) are TIFFs with lossless deflate compression,
    stored as float32 (relative rounding below 1e-7; they were uncompressed
    float64, 158 MB each).
  - The gap overlay images are kept once, as `Colors/.../all_gaps.png` and
    `intra_gaps.png`; the copies in `Exports/` are gone.
  - The working copies of the inputs (`Eligible/`) are deleted at the end
    of a run; `IgnoredImages.txt` moved to the output folder.
  - Figures with a colour bar draw the image at its own pixel size (they
    were enlarged 2 to 3 times because importing seaborn_image changes
    matplotlib's save resolution), as RGB without transparency; layout and
    colour bar are unchanged.
- Batch runs no longer write `Fibres/`: its fibre overlay showed the same as
  `Colors/.../color_mask.png` (24 MB per core).
- New `Configs: Save Visualisations` (default on), with a **Visualisations**
  checkbox on the Batch Run page: off keeps only the measurements (about
  60 MB per core).

### Batch

- Oversized images (larger than Max Size) are split into blocks again in GUI
  batch runs. From 0.3.1 to 0.3.8 they were skipped, so batches of TMA cores
  (4000 to 5000 px) produced no results.
- **Max Size moved to `Configs: Max Size` and defaults to 5000 px** (was
  `Segmentation: Max Size`, 2048 px, which the segmenter never used). Images
  between 2048 and 5000 px, such as TMA cores, are now analysed whole instead
  of in blocks, so their results change (fibres and gaps are no longer cut at
  block edges). Parameter files with the old location are still read, and
  are written in the new layout when re-exported.
- The Batch Run page shows the parameter file's Max Size and can override it
  for a run; the value is kept in the checkpoint, so a resumed run splits
  images as it started. 0 (*No limit*) analyses every image whole and the
  run warns about the largest image's memory need (a 4400 px core needs about
  7 GB); 0 used to fail.
- The log reports how many eligible images have an external ROI mask, and
  says when no image was eligible, instead of "0 external ROI masks found".

## 0.3.8

### GUI

- The side panel's scroll bar is drawn in its own slim column beside the
  page, with a gap, instead of as an overlay on the page's right edge next
  to the splitter; it appears only when the page shown does not fit. Short
  pages such as Start no longer scroll just because the TMA Dearrayer page
  is taller.
- Quitting asks for confirmation only when a job is running or TMA hand
  edits are unsaved; otherwise the window closes at once.

### TMA

- Slides printed as several blocks whose spacing is not a whole number of
  grid pitches are handled: a block sitting out of step with the rest of the
  array (e.g. the bottom-right quadrant of the neuroblastoma TMA, 0.4 pitch
  off) gets its own offset, so its missing-core placeholders no longer
  overlap real cores and its cores pass the grid-offset check. With a map,
  such a block also moves by one cell when that fits the map better, which
  fixes patient IDs that were one column off.
- The faint-core pass no longer adds a core where its disc would overlap an
  already fitted core.
- *Core Ø* has an *Auto* tick box, on by default: the core diameter is
  measured from the slide at every fit (median of the well-formed cores,
  refitting once with it) and shown in the box; untick it to type a value.
  `cabana-tma --core-diameter` defaults to `auto`, and `TMAPreprocessor`'s
  `core_diameter_um` to None (measure). `cores.csv` records the value used
  and its source. Previously Core Ø defaulted to 1250 µm, which excluded
  every core of arrays with smaller cores.
- Fixed a crash in the circle fit when debris pieces of different sizes were
  dropped from an oversized circle.
- With *Edit cores* on, clicking a core only selects it: a press must drag at
  least 4 screen pixels to move or resize the core. Previously every click
  marked the core hand-edited and added an undo step.

## 0.3.7

### GUI

- After fitting TMA cores, the long status line no longer widens the side
  panel's page past the panel, which hid its right edge behind the image
  panel (a regression in 0.3.6); the line wraps to the panel again.
- Quitting asks for confirmation (Confirm or Cancel), naming any running job
  and TMA hand edits not yet exported or saved.

## 0.3.6

### TMA Dearrayer

- Your own TMA maps: **Array Map > Load map file…** reads a map laid out like
  the printed array (`ctrl:<tissue>` for controls, `?` for a core with unknown
  ID, blank for no core), from `.csv`, `.tsv`, `.txt` or QuPath `.qpmap`
  files, with several maps per file. Tables with one line per core are also
  accepted. A preview shows each map as a coloured grid with any problems
  before it is used. **File > Generate TMA Map Template…** writes a ready-labelled
  12 x 12 grid to type or paste into; its **cell lines** row lets cells pasted
  from a PDF, holding a position, patient ID, block ID and class on separate
  lines, be read directly. Maps may have any number of rows and columns; the
  built-in ICGC/APGI arrays are unchanged.
- Cores a map lists without an identity are exported to `Unmapped` as
  `<slide>-<position>.vsi - …Annotation (Unknown)_1`. Extra map columns are
  copied to `cores.csv`. The Brain control is recognised regardless of case.
- Saved edits record the map's name and a fingerprint of its contents.
- Command line: `--map FILE` and `--map-name NAME`, and
  `cabana-tma map check|convert|template`.
- Cells holding several lines without a cell lines row are refused with a
  message naming the cell, instead of being misread as a grid of fragments.
  A labelled grid must give every row its label.
- The first hand edit no longer freezes the window for a few seconds: the
  tissue mask used to measure edited cores is now kept from Fit Cores
  instead of being rebuilt on the first edit.

### GUI

- On a short screen the side panel scrolls instead of squeezing its text
  boxes, spin boxes and buttons below their natural height.

## 0.3.5

### TMA Dearrayer

- *Auto* and an explicitly chosen orientation can now place the map with an
  empty edge row or column (row L of arrays 1 and 2) outside the fitted grid.
  Stray tissue beside APGI TMA 2 had widened the grid on the other side, so
  every label was shifted by a row whatever the orientation; TMA 2 is now
  placed correctly by *Auto* (270, decided by the Brain control). Fully
  populated arrays (3 to 8) are matched exactly as before. Re-export TMA 2
  if it was exported with an earlier version.

## 0.3.4

### TMA Dearrayer

- A wrong map placement can be corrected by hand: with **Edit cores** on,
  right-click a recognisable core (for example a control) and choose
  **Set map position…** to give it its true position. The whole map is placed
  from that core, the orientation is kept, and choosing another orientation
  turns the map about it. A hand-set placement enables Export, is recorded as
  `orientation_method = anchor`, is saved in `cores_edits.json` and can be
  undone or cleared. Rotating alone could not fix a map that was shifted by a
  row, as happened on APGI TMA 2.
- Hovering a core reminds you that a wrong label can be corrected by
  right-clicking it.
- **Save Edits…** saves the cores, orientation and hand-set map position
  without exporting, and **Load Edits…** reinstates any saved edits file,
  asking first if it was saved for another slide name.
- After a fit, the offer to reinstate saved edits also appears when the only
  saved edit is a hand-set map position.
- Export asks before replacing saved edits of the slide with a fit that has no
  hand edits, so a declined reinstate offer no longer loses them silently.
- Opening a slide always sets the output folder to that slide's own
  `<slide>_cores` folder. It used to keep the previous slide's folder, so a
  second slide was exported under the first slide's name. The folder is
  placed next to the output folder chosen last (remembered between sessions),
  or beside the slide until one has been chosen, and Export asks before
  writing into a folder that holds another slide's export.

## 0.3.3

### TMA Dearrayer

- Exported core images are named by the 4-digit patient ID printed on the
  ICGC map (for example `1734.vsi - …`) instead of the longer APGI block ID.
  Replicate numbering and the replicate orientation check group cores by the
  same patient ID. The block ID is in `cores.csv` in a new `block_id` column,
  which replaces `icgc_id`. Exports made with earlier versions use block IDs
  as names and should be re-exported before they are combined with new ones.
- Running Fit Cores again on the same slide no longer closes the slide file:
  exporting after a refit used to fail with "seek of closed file".

## 0.3.2

- The automatic orientation entry on the TMA Dearrayer page is labelled
  "Auto"; the tooltip and the status line explain how it is chosen.
- The GUI silences Qt's harmless macOS "Back buffer dpr ... contents scale"
  log message at start-up.

## 0.3.1

### TMA Dearrayer

- The array's rotation is estimated and removed before the cores are snapped
  to the grid, and the grid is set by well-formed cores only, so a tilted
  slide or debris no longer shifts rows or adds a spurious one.
- Map positions without a core are listed in `cores.csv` (flag `missing`) and
  drawn as dashed circles joined into the lattice.
- Fitted cores can be corrected by hand (move, resize, add, delete, include or
  exclude, undo); a moved or added core takes the patient ID of the grid cell
  it lands in and two cores can never share a cell. Edits are saved with the
  export (`cores_edits.json`) and offered back after a refit, keeping the
  confirmed orientation.
- Changing the orientation or array relabels the cores instantly; control
  cores show their tissue name and hovering a core shows its details.
- The Brain tie-break rule ignores slivers of lost cores, which could mimic
  an unstained core.
- The page is called TMA Dearrayer.

### Measurements

- `Cabana` (single image) now reads the pixel size from the image metadata
  like the batch pipeline, so its µm metrics are no longer in pixels.
- A skeleton mask without any fibre reports zero area, length, lacunarity and
  fractal dimension instead of the whole image area and a lacunarity of 1.
- Closed cells and loops in a fibre network keep their full length: paths
  running between the same two junctions, or back to one junction, were
  previously collapsed to one segment.
- Closed ridge contours are recognised as closed (they were traced twice and
  given a false self-junction).
- `Fibre Area (HDM, µm²)` is relative to the analysed ROI-mask area when a
  mask is used, matching `% HDM Area`; a uniform tile reports 0 % HDM instead
  of 100 % with Dark Line on; an empty ROI mask gives 0 % rather than the
  whole image.
- Per-patient STD and SEM use the sample standard deviation; a patient with a
  single image has no STD or SEM instead of 0.
- The channel suffix on patient IDs (`_red`, `_green`, …) requires a whole
  token, so names such as `Fred` or `registered` are no longer split off, and
  the result does not depend on the Python hash seed.
- Orientation metrics on an empty mask are 0, not NaN; the orientation
  randomness measure no longer fails on images that do not populate every
  angular bin; a flat image no longer leaves the previous image's tensors in
  place.
- Detection sanitises NaN pixels in float images and validates step sizes in
  the parameter file.

### GUI

- The Segment, Detect and Analyze buttons recover and report the error when
  their worker fails instead of the application closing.
- Images are drawn through the visible region only, so zooming far into a
  large image no longer allocates a gigantic scaled copy.
- The interactive Detect preview uses the same line-width scales as the batch
  run.
- `BatchProcessor` raises on a missing parameter file or input folder instead
  of terminating the process.
