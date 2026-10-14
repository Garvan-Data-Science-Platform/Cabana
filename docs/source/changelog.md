# Changelog

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
