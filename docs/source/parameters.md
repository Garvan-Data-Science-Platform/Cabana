# Parameter Details

## Quick reference

All parameters live in the YAML parameter file (`Parameters.yml`, exported from **Parameters > Export Parameters**). The "Set in" column says whether a control exists in the GUI; the others are edited in the file.

| Section | Parameter | Default | Unit | Set in |
|---|---|---|---|---|
| Configs | Segmentation | true | on/off | Segmentation page toggle |
| Configs | Quantification | true | on/off | file only |
| Configs | Gap Analysis | true | on/off | Gap Analysis page toggle |
| Configs | Max Size | 5000 (0 = no limit) | px (side) | Batch Run page (per run) |
| Configs | Save Visualisations | true | on/off | Batch Run page (per run) |
| Segmentation | Number of Labels | 32 | count | slider |
| Segmentation | Max Iterations | 30 | count | slider |
| Segmentation | Normalized Hue Value | 0.96 | 0 to 1 | colour picker |
| Segmentation | Color Threshold | 0.2 | 0 to 1 | slider |
| Segmentation | Min Size | 64 | px | file only |
| Segmentation | Patch Size | 1024 (0 = off) | px | spinner |
| Detection | Dark Line | true | on/off | checkbox |
| Detection | Min Line Width / Max Line Width | 5 / 13 | px | range slider |
| Detection | Line Width Step | 2 | px | slider |
| Detection | Low Contrast / High Contrast | 100 / 200 | grey levels | range slider |
| Detection | Extend Line | false | on/off | checkbox |
| Detection | Minimum Line Length | 5 | px | slider |
| Detection | Maximum Line Length | 0 (no limit) | px | file only |
| Quantification | Maximum Display HDM | 230 | grey level | slider |
| Quantification | Contrast Enhancement | 0.1 | fraction | file only |
| Quantification | Minimum Branch Length | 5 | px | file only |
| Quantification | Minimum / Maximum Curvature Window, Curvature Window Step | 10 / 30 / 10 | px | file only |
| Gap Analysis | Minimum Gap Diameter | 20 | px | slider |

The folders of a run (parameter file, input, output and the optional ROI mask folder) are chosen on the Batch Run page and are not part of the parameter file; the folders used are recorded in `version_params.yaml` in the output folder.

## Configurations

The user can set the parameters in this section to enable (setting to `true`) or disable (setting to `false`) the components to be used in the `Configs` section.

For instance, if the background is clean, segmentation may not be needed and therefore can be disabled.

`Max Size` (also in `Configs`) sets the largest image analysed as one piece: images with more pixels than its *square* are split into blocks of at most this size, and each block is analysed as a separate image, which keeps memory use down but cuts fibres and gaps at the block edges. The smallest value is 2048. 0 means no limit: every image is analysed whole, and very large images may run out of memory; a batch run warns when an image is larger than 5000 px. The default value, 5000, analyses TMA cores scanned at 20x whole. The Batch Run page shows the parameter file's value and can override it for a run. Parameter files written before version 0.3.9 kept it under `Segmentation`; that location is still read.

`Save Visualisations` (also in `Configs`, default `true`) decides whether batch runs write the visualisations: the colour maps and the gap overlay images (`Colors/`). The measurements (CSV files, the TIFF maps in `Exports/`, masks) are always written. Turning it off saves about four fifths of the storage for TMA cores; the Batch Run page shows the file's value and can change it for a run.

## Segmentation Parameters

This component of Cabana aims to extract collagen fibre areas determined by Picrosirius Red staining or SHG in an image from cluttered background based on colour and other low-level features. It relies on a self-supervised semantic segmentation model based on convolutional neural networks to group semantically similar neighbouring pixels. The mean colour of the pixels in the same segment will be compared with a user-specified threshold to determine whether the segment is the region of interest (ROI).

The following image segmentation parameters for ROI extraction can be adjusted:

1. **Number of Labels**

   The number of labels for semantic segmentation. It controls the level of granularity of segmentation. A higher number of labels will result in segments of smaller size. The default value is 32.

2. **Max Iterations**

   The maximal number of iterations for semantic segmentation. This value should be large enough to generate reliable segmentation results but not too high to avoid grouping all pixels into a single segment. The default value is 30.

3. **Normalized Hue Value**

   The normalized hue value in [0,1] for the colour of interest in HSB/HSV colour space. The typical hue values for green, blue, and red colour are 0.33, 0.66, and 1.0, respectively.

4. **Colour Threshold**

   Minimum colour score of a segment for it to count as ROI. Each segment is scored by how close its mean colour is to the colour of interest, from 1 (exactly that colour) down to 0. For colour images the score is absolute, so the same threshold means the same on every image and in every patch; below about 0.12 pale tissue and glass start to count. Greyscale images are scored by brightness relative to the image (see the note below). The default value is 0.2. After a segmentation in the GUI, moving the slider updates the result immediately without segmenting again.

5. **Min Size**

   The minimal size of segments. Any segment with a size smaller than this parameter will be ignored. The default value is 64.

6. **Patch Size**

   Images larger than this (in pixels, on the longer side) are segmented in square patches of this size; 0 segments every image whole after shrinking it to 512 px. The segmentation network is trained once on the whole image shrunk to 512 px and then labels every patch, itself shrunk to 512 px, without further training, so a large image is seen at a higher resolution at little extra cost. Patches overlap by one eighth of their size and are edge-anchored so every patch is full-sized; their colour scores are blended across the overlaps and the image is thresholded once. The default value is 1024.

Note:

The program automatically detects whether the image is grayscale or in colour. If it's grayscale, it'll disregard the normalized hue value parameter and solely rely on the colour threshold for segmentation: segments are scored by brightness, rescaled from the darkest to the brightest segment of the image (once per image, also when it is segmented in patches), so the threshold is relative to each image. To extract ROIs of colour images, you may need to adjust the Normalized Hue Value under the `Segmentation` tab or directly modify the `Segmentation` section in the exported `Parameters.yml` to optimise the parameters for your image**. To segment SHG images, it is recommended to increase the number of iterations from the default value 30 to 50. This needs to be tested and optimised by the user.

When the boundaries between regions of interest (SHG signal or Picrosirius Red areas) and background are not clearly distinguishable, obtaining satisfactory segmentation results can be difficult. To address this, the `Number of Labels` and `Max Iterations` parameters can be optimised. Increasing the number of labels generally leads to finer granularity in segmentations, which may result in a better segmentation result. Avoid increasing the number of labels beyond 64, otherwise GPU Memory might overflow. On the other hand, decreasing the max iterations may result in premature segmentation, reducing the likelihood of mixing fibres and background. Adjusting the HUE parameter may not have much of an effect as the colour of Picrosirius stained collagen fibres is already red/pink.

## Fibre Detection and Quantification

This component is designed to detect and quantify fibre structures in images. The file Parameters.yml contains two dedicated sections for controlling the outcomes of collagen fibre analysis:

1. **Detection**: Parameters for detecting fibres.
2. **Quantification**: Parameters for quantifying fibres.

### Detection

1. **Dark Line**

   Set to false = the program assumes that fibres are light on a dark background (fluorescence, birefringence, SHG etc).

   Set to true = Dark fibres on a light background (Picrosirius Red IHC).

2. **Min Line Width** $\mathbf{\omega}_{\mathbf{\min}}$

   It defines the minimum line (ridge) width in pixels that the ridge detection algorithm can detect. The line width $\omega$ is used to estimate the `Sigma` $\sigma$ parameter of Gaussian filtering kernel: $\sigma = \frac{\omega}{2\sqrt{3}} + 0.5$.

   Note on the width floor:

   Fibre width is measured from the two edges (gradient maxima) on either side of the ridge, located in the image smoothed at the scale assigned to that fibre (the median of the per-pixel best scales along the ridge). For a fibre thinner than the smoothing kernel, the two edges are found about one $\sigma$ either side of the ridge regardless of the true width, so the reported width cannot fall below roughly $2\sigma$ of the smallest scale. With $\omega_{\min} = 5$ this floor is about 3.9 px. Widths near this value should be read as "at most this wide". Lower $\omega_{\min}$ to lower the floor, at the cost of more texture being detected as ridges.

3. **Max Line Width** $\mathbf{\omega}_{\mathbf{\max}}$

   The maximum line (ridge) width in pixels that the ridge detection algorithm can detect.

   Twombli runs ridge detection repeatedly with every value between the minimum and maximum line width and calculates the combination of the detected ridges.

   Note:

   Setting a large line width (e.g., >15) gives rise to the chance of *straight line artefacts*, which are caused by the small (close to zero) upper threshold of filtering response.

4. **Line Width Step**

   This parameter controls the sampling factor for line widths between the minimum and the maximum line widths. It allows for increasing the line width by step larger than 1.

   For instance, if you want to detect ridges with multiple line widths 5, 7, and 9, you can specify a line width step to 2 with the min line width $\omega_{\min} = 5$ and the max line width $\omega_{\max} = 9$. The default value is 2.

5. **Low Contrast** $\mathbf{b}_{\mathbf{l}}$

   Defines the lowest grayscale contrast between a line (collagen fibre) and background (non-fibre area).

   This parameter is used to estimate the lower threshold for the filtering response: $T_{L} = \frac{0.17\omega b_{l}}{\sqrt{2\pi}\sigma^{3}}e^{- \frac{\omega^{2}}{8\sigma^{2}}}$. Line points with a filtering response lower than the low contrast threshold $T_{L}$ are discarded.

6. **High Contrast** $\mathbf{b}_{\mathbf{u}}$

   The highest grayscale contrast between a line (collagen fibre) and background (non-fibre area). This parameter is used to estimate the upper threshold for filtering response: $T_{U} = \frac{0.17\omega b_{u}}{\sqrt{2\pi}\sigma^{3}}e^{- \frac{\omega^{2}}{8\sigma^{2}}}$. Line points with a response larger than the high contrast threshold $T_{U}$ are accepted. Line points with a response in [$T_{L},\ T_{U}$] are added to the accepted line points if line structures are reasonably formed.

   Note:

   Not the absolute intensity is important, but the difference/contrast of a pixel with its neighbours. This applies to low and high contrast. 

   If a pixel with an intensity of 230 appears among neighbouring pixels with intensity of 230, the filtering response will be zero because this point is not visually salient. If this pixel appears among pixels with an intensity of 10, then the filtering response will be strong enough to signify it as a salient line/ridge point. For example, pixels values > 200 might or might not be accepted depending on their filtering responses. But the higher the contrast threshold, the criteria for a pixel becoming a line point becomes more stringent, therefore less line points will be detected. Also note that,
if the `Dark Line` is set to `true` (for Picrosirius Red), the low contrast and high contrast will be calculated as $255 - b_{u}$ and $255 - b_{l}$, respectively.

### Quantification

1. **Contrast Enhancement**

   A value between [0,1] shows the percent of pixels that will be saturated for contrast enhancement.

2. **Min/Max Curvature Window and Curvature Window Step**

   The curvature of ridges/lines in curvature windows bounded by the minimum and maximum curvature windows with a window step size.

3. **Minimum Branch Length**

   Any line/ridge with a length smaller than this value is ignored.

   Note: If the minimum branch length value is too low, short branch artefacts are introduced.

4. **Maximum Display HDM**

   Pixels outside [0, Maximum Display HDM] are set to 0 before estimating the high-density matrix (HDM) area.

## Gap Analysis Parameters

1. **Minimum Gap Diameter**

   The minimal gap diameter for gap analysis.
