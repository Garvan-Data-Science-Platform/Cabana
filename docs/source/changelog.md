# Changelog

## Unreleased

### Logging

- Batch and single-image runs write a log file to `<output>/Logs/`, with
  timestamps and one file per run. Repeated runs in the GUI no longer write
  into each other's log.
- The console shows only run progress, warnings, errors and where results
  were saved; stage details and parameters go to the log file. Per-image
  warnings are summarised as counts.
- Scripts: importing `cabana` no longer prints anything by itself. Call
  `cabana.log.setup_logging()` to see progress on the console; CABANA's
  messages also reach your own logging setup.
- Messages from other GUI actions no longer appear in a running batch's log.
- Fixed: the GUI could close when a batch run failed during setup (for
  example, a missing parameter file). The error is now reported.
- Batch runs no longer re-read every input image's metadata for each batch.

### Batch

- Fixed: a batch run could delete its own inputs. A new run clears the output
  folder, so an output folder that is, or contains, the input or ROI mask
  folder (for example a TMA export folder) is now refused.
- Fixed: an input folder path containing a comma, or a damaged checkpoint
  file, made the GUI close when Process was pressed.
- Fixed: an input folder without supported images reported a successful run
  and could remove an interrupted run's checkpoint. It is now refused.
- Fixed: on Windows, the single-image `.txt` results file failed partway
  through.
- Blank 16-bit images are listed in `IgnoredImages.txt`.

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
