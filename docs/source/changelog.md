# Changelog

## 0.4.1

- Every message and question (Run Folder and Batch Run results, the
  questions before writing into a folder, TMA export, Check for Updates,
  errors and Save Image) now uses the card style of the About and What's
  New dialogs. Check for Updates shows pip's output behind Show Details.
- What's New after an upgrade also lists the releases that were skipped.
- Fixed: the last line of a wrapped note in What's New could be cut off.
- The screenshots in the documentation show the Dark theme.

## 0.4.0

- New look for the side panel: each page's settings are grouped in cards
  that fold away (click a card's title), with the step's main action at the
  bottom of its card. Path fields show the file or folder name with its
  location beside it; click the field to choose. Spin boxes have small
  up/down arrows.
- An icon bar at the left edge switches pages; clicking the open page's
  icon folds the panel to give the image the whole window. It replaces the
  panel toggle button.
- Help > Take the Tour walks through the analysis one card at a time; it
  is offered once on the first start. After an upgrade a short What's New
  note is shown once (Help > What's New shows it again).
- Starting a new Batch Run asks before deleting a results folder that
  already holds files. The batch progress is shown in the Run card.
- TMA Dearrayer page: short settings share rows in two aligned columns;
  Load Edits and Save Edits are in the Edit card's title bar and the BF and
  POL channels in the Export card's title bar.
- Birefringence page: a band edge dragged into the neighbouring band stops
  there instead of moving that band; bands may leave a gap but never
  overlap. The rows are named Red-orange, Yellow and Green.
- Keyboard: Tab reaches every control and shows a focus ring; Enter or
  Space opens a path field's file dialog, Delete clears the ROI mask
  folder, Enter or Space folds a card, and the arrow keys move range slider
  handles (Space switches between them).
- Easier to read: larger text, higher-contrast secondary text, placeholders
  and field outlines in both themes. Only the Light and Dark themes remain.
- Slider handles are no longer cut off at the ends of their track.
- The licence holder is now the Garvan Institute of Medical Research.

## 0.3.17

- New Birefringence page for polarised-light images: each pixel is
  classified into the red-orange, yellow and green bands of the PR-BRF
  Analyser ImageJ macro, with the same pixel counts as the macro at its
  default settings. A Grid view shows the original image and the three
  bands side by side. Run Folder separates every image in a folder, inside
  optional ROI masks, and writes the band images and tables (sorted into
  `Total`, `Red`, `Yellow`, `Green` and `Stats` by default); the band
  images can then be analysed in a Batch Run like any other images;
  per-patient statistics keep each band (`red`, `yellow`, `green`,
  `total`) separate.
- The ROI mask folder on the Batch Run page is cleared with the icon in its
  field; an empty field means no masks.
- Min Branch Length and the curvature windows are now set on the Fibre
  Detection page, and HDM Saturation (Contrast Enhancement) on the Gap
  Analysis page; before, they could only be edited in the parameter file.

## 0.3.16

- Max Size is set on the Segmentation page, next to Patch Size, and saved
  with the other parameters by Parameters > Export. The Batch Run page shows
  the parameter file's value but no longer changes it, so the file kept with
  the results describes the run. A note on the Segmentation page says when
  the open image would be split in a batch run, and when Patch Size is too
  large to apply to the blocks. The Max Size arrows step in round thousands.
- A resumed batch run always uses the parameters and ROI mask folder it
  started with, which are now saved in the output folder; the resume prompt
  lists any differences from the selected ones. Before, a resume used
  whatever parameter file was selected, mixing two sets of parameters in one
  run.
- Scrolling the side panel with the mouse wheel no longer changes a spin
  box, slider or drop-down list under the pointer unless it was clicked
  first.
- Fixed: gap analysis ran out of memory on images with tens of thousands of
  small gaps (a small Minimum Gap Diameter on large blocks). It is also about
  three times faster on such images. The circles found are unchanged.
- Inputs above 89 megapixels no longer trigger PIL's "decompression bomb"
  warning, and inputs above 179 megapixels no longer fail to open.
- Fixed: Cabana failed to start with `cannot import name 'get_cmap' from
  'matplotlib.cm'` on a fresh install with matplotlib 3.11. The `seaborn-image`
  package that caused it is no longer required.
- The empty image panel now points to File > Open Image… (there is no
  Load Image button since the menu bar replaced it).
- Fixed: an input or output folder whose name contains `[`, `]`, `*` or `?`
  (for example `Run [2]`) was refused as having no supported images, or
  produced empty results, because those characters were read as file
  patterns.

## 0.3.15

- `QuantificationResults_Compact.csv` (and `<image>_QuantificationResults_Compact.csv`
  for single-image runs) holds the same rows as the full results with the
  core columns, dropping those that restate others (absolute areas, raw counts,
  gap areas, gap percentiles and normalized gap statistics, whole-image and
  HDM orientation). With Stats or Scores on, `QuantificationResults_Compact_MEAN_STD_SEM.csv`
  and `QuantificationResults_Compact_SCORES.csv` give the per-patient statistics
  and the (identical) risk scores over the same columns. The full files are
  unchanged.
- Fixed: resuming a batch run whose final "Putting together everything" step
  had been interrupted deleted the images already gathered into the `ROIs`,
  `Bins`, `Masks`, `HDM`, `Exports` and `Colors` folders, leaving only the
  CSV results.

## 0.3.14

- The colour bars of the saved maps have larger text, scaled to the image
  size, so they stay readable on large images such as whole TMA cores.

## 0.3.13

- Length maps use a wider colour range (blue for short to red for long
  fibres); curvature maps show each fibre as straight (< 15°), wavy (15 to
  45°) or curly (> 45°), the same unit as the curvature columns of the
  results.

## 0.3.12

- Fixed: batch runs stopped at the colour maps when an input image had a
  transparency (alpha) channel.
- With visualisations off, the output no longer contains an empty `Colors`
  folder.
- **Check for Updates** detects a leftover install record from an
  interrupted upgrade, which made Cabana show the wrong version, and explains
  how to clean it up.
- The project links (Report an Issue, licence, package home page) point to
  the repository's new address.

## 0.3.11

- **Check for Updates** (Help menu; Cabana menu on macOS) installs a newer
  release from PyPI into the running environment; restart Cabana afterwards
  to use it. Cabana also checks once a day at startup and offers the update,
  or lets you skip that version.
- Fixed: on Windows the taskbar showed a generic or Python icon instead of
  Cabana's logo.
- Fixed: on Windows, batch runs failed when Cabana was started with
  `cabana-gui` (no console window).

## 0.3.10

### Measurements

- **Results in µm change slightly.** The pixel size is no longer rounded to
  0.01 µm (up to about 2 % in lengths and 4 % in areas). Re-run analyses
  whose pixel size is not a multiple of 0.01 µm.
- Fixed: images with a resolution in DPI got a wrong pixel size; 1.0 µm/pixel
  is now used, with a warning. Units in ImageJ and QuPath TIFFs are converted.
- Fixed: images without gaps could get blank gap metrics instead of 0.

### Visualisations

- Clearer colour maps: fibre length and curvature are drawn as wider lines
  over the faded image with a colour scale that ignores outliers; the
  orientation vector field shows lines only on fibres, coloured by direction.
  Gap overlays use the same colours as the GUI.
- Single-image runs now write the same colour maps as batch runs.
- Fixed: the orientation vector field averaged fibre directions wrongly near
  horizontal.

### Batch

- Max Size is at least 2048 px (or *No limit*); smaller values are raised.
- Fixed: a batch run could delete its inputs when the output folder
  contained them; such output folders are now refused.
- Fixed: the GUI could close on a failed batch setup, a damaged checkpoint or
  an input path containing a comma.
- Fixed: an input folder without supported images was reported as a
  successful run.
- Single-image runs no longer keep a working copy of the input (`Eligible/`).
- Fixed: the single-image `.txt` results file could fail on Windows.

### GUI

- The Gap Analysis page shows intra-fibre and other gaps in different
  colours over clearer fibres, with a summary of the gaps found.

### Logging

- Each run writes a log file to `<output>/Logs/`; the console shows only
  progress, warnings and errors.
- Scripts: call `cabana.log.setup_logging()` to see progress on the console.

## 0.3.9

### Segmentation

- **Results change for colour images.** The colour threshold now applies to
  the absolute colour score instead of a score rescaled per image or per
  patch. Per-patch rescaling marked faint or empty patches as collagen and
  left seams at patch borders. Weakly stained images can change
  substantially; re-run results produced with a Patch Size set.
- Greyscale images (SHG, fluorescence) are unchanged without patches; with
  patches they are rescaled once per image instead of per patch.
- Patch mode trains the network once per image instead of once per patch
  (about 11 times faster on a 4400 px test image).
- Patch Size defaults to 1024 px.
- In the GUI, the Colour Threshold slider updates the result instantly after
  segmentation.

### Speed, memory and storage

- On a 4400 px test image, a full analysis is about 7 times faster (125 s
  instead of 857 s), peaks at 7 GB of memory instead of 9.7 GB, and needs
  about 300 MB of storage instead of 1.35 GB. Measurements are unchanged
  (fibre widths differ by less than 0.03 px).
- Analysis maps are saved as compressed TIFFs. Duplicate gap overlays,
  `Fibres/` (batch runs) and the temporary `Eligible/` folder are no longer
  kept; `IgnoredImages.txt` is in the output folder.
- Figures with a colour bar are saved at the image's own resolution (they
  were enlarged 2 to 3 times).
- New **Save Visualisations** option (`Configs: Save Visualisations`, also on
  the Batch Run page): turn it off to keep only the measurements.

### Batch

- Fixed: images larger than Max Size were skipped in GUI batch runs since
  0.3.1. They are split into blocks again.
- **Max Size moved to `Configs: Max Size` and defaults to 5000 px** (was
  2048 px). Images between 2048 and 5000 px are now analysed whole, so their
  results change. Old parameter files still load.
- The Batch Run page can override Max Size per run (kept when resuming).
  0 means no limit and warns about the memory needed.
- Clearer log message about how many images have an external ROI mask.

## 0.3.8

### GUI

- The side panel's scroll bar has its own column and appears only when
  needed.
- Quitting asks for confirmation only when a job is running or TMA edits are
  unsaved.

### TMA

- Arrays printed as blocks that are out of step with the grid are handled.
  With a map, such a block is shifted by one cell when that fits better,
  fixing patient IDs that were one column off.
- Core Ø is measured from the slide by default (*Auto*; command line
  `--core-diameter auto`). The previous 1250 µm default excluded every core
  of arrays with smaller cores.
- The faint-core pass no longer adds cores that overlap existing ones.
- Fixed a crash in the circle fit when removing debris.
- In *Edit cores*, a click selects a core; moving or resizing needs a drag.
  Previously every click counted as an edit.

## 0.3.7

### GUI

- Fixed: after fitting TMA cores, the status line widened the side panel and
  hid its edge (regression in 0.3.6).
- Quitting asks for confirmation, naming running jobs and unsaved TMA edits.

## 0.3.6

### TMA Dearrayer

- Custom TMA maps: **Array Map > Load map file…** reads grid- or list-style
  maps from `.csv`, `.tsv`, `.txt` or QuPath `.qpmap` files, with a preview
  that shows any problems. **File > Generate TMA Map Template…** writes a
  labelled grid to fill in, which also accepts cells pasted from a PDF. Maps
  can have any number of rows and columns.
- Cores without an identity in the map are exported to `Unmapped`; extra map
  columns are copied to `cores.csv`.
- Command line: `--map`, `--map-name` and
  `cabana-tma map check|convert|template`.
- Malformed map cells are reported by position instead of being misread.
- The first hand edit no longer freezes the window.

### GUI

- On short screens the side panel scrolls instead of squeezing its controls.

## 0.3.5

### TMA Dearrayer

- Fixed: stray tissue beside an array with an empty edge row could shift
  every label by one row. Re-export affected slides.

## 0.3.4

### TMA Dearrayer

- A wrong map placement can be fixed by hand: with **Edit cores** on,
  right-click a known core and choose **Set map position…**. The placement is
  saved with the edits and can be undone.
- **Save Edits…** and **Load Edits…** save and restore edits without
  exporting.
- Export asks before replacing saved edits with an unedited fit.
- Fixed: opening a second slide kept the first slide's output folder, so it
  was exported under the wrong name. Each slide now gets its own
  `<slide>_cores` folder, and Export asks before writing into another slide's
  folder.

## 0.3.3

### TMA Dearrayer

- Exported cores are named by the 4-digit patient ID from the map instead of
  the block ID, which moves to a `block_id` column in `cores.csv`. Re-export
  older exports before combining them with new ones.
- Fixed: exporting after a second Fit Cores failed with "seek of closed
  file".

## 0.3.2

- The automatic orientation option is labelled "Auto", with an explanation
  in its tooltip.
- A harmless Qt warning at start-up on macOS is silenced.

## 0.3.1

### TMA Dearrayer

- Slide rotation is corrected before grid fitting, so tilted slides or debris
  no longer shift rows or add spurious ones.
- Map positions without a core are listed in `cores.csv` and drawn as dashed
  circles.
- Fitted cores can be edited by hand (move, resize, add, delete, include or
  exclude, undo). Edits are saved with the export and offered back after a
  refit.
- Changing the orientation or array relabels cores instantly; hovering a core
  shows its details.
- The orientation tie-break ignores slivers of lost cores.
- The page is renamed TMA Dearrayer.

### Measurements

- Single-image `Cabana` reads the pixel size from the image metadata, so its
  µm metrics are no longer in pixels.
- Empty masks report zeros instead of whole-image values or NaN.
- Closed loops in fibre networks keep their full length, and closed ridge
  contours are no longer traced twice.
- HDM area is relative to the ROI mask when one is used; uniform tiles report
  0 % HDM.
- Per-patient STD and SEM use the sample standard deviation; patients with a
  single image have none instead of 0.
- Channel suffixes (`_red`, `_green`, …) are stripped from patient IDs only as
  whole tokens.
- Orientation randomness no longer fails on sparse histograms.
- Detection handles NaN pixels and validates step sizes.

### GUI

- Segment, Detect and Analyze report worker errors instead of closing the
  application.
- Zooming far into large images no longer uses excessive memory.
- The Detect preview uses the same line-width scales as batch runs.
- `BatchProcessor` raises an error on missing inputs instead of exiting the
  process.
