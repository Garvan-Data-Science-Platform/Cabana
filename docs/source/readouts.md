# List of Cabana Read-outs (QuantificationResults.csv)

Values referring to *WIDTH*, *ROI* and *HDM* are based on the following outputs:

- WIDTH: Values based on ridge detection (found in images named \`\*\_Width.png\` within the `Exports` folder).
- ROI: Values based on segmented images (located in the `ROI` folder).
- HDM: Values derived from the High-Density Matrix (HDM), represented as non-black areas/pixels in images from the `HDM` folder.

## Statistics Interpretations

### Image and fibre area

| Result | Meaning | Calculation method |
|---|---|---|
| Total Image Area (µm²) | Total image area in microns² | Total number of pixels × (Image Resolution (µm/pp))² |
| Image Res. (µm/pp) | Image resolution in µm per pixel | Read out from image metadata. |
| Fibre Area (HDM, µm²) | Projected area of detected ridge spines, typically one pixel wide. | Number of HDM pixels × (Image Resolution (µm/pp))² |
| % HDM Area | Percentage of the image area covered by HDM. | High-density matrix areas are obtained using the following three steps: <br>1. Set pixels outside the range [0, maxDisHDM] to 0, where maxDisHDM is the value of the `Maximum Display HDM` parameter in Parameters.yml<br>2. Apply contrast enhancement to the resultant image using the `Contrast Enhancement` parameter in Parameters.yml.<br>3. Calculate % HDM as the portion of pixels in HDM area (% total area), where HDM areas are defined by the non-black pixels in the contrast enhanced image. |
| Avg Thickness (HDM, µm) | Average fibre thickness based on HDM. | HDM area / Total fibre length |
| Mean Fibre Intensity (HDM) | Mean intensity of red (PicrosiriusRed) or gray (SHG) pixels in HDM. | Measured in HDM area. |
| Fibre Area (ROI, µm²) | Area within the segmented ROI. | ROI pixel count × (Image Resolution (µm/pp))² |
| % ROI Area | ROI area as a percentage of the total image area. | ROI area / Total area |
| Avg Thickness (ROI, µm) | Average fibre thickness based on the ROI. | ROI area / Total fibre length |
| Fibre Area (WIDTH, µm²) | Fibre area calculated using ridge detection. | Ridge-detected pixel count × (Image Resolution (µm/pp))² |
| % Width Area | Fibre coverage based on ridge detection. | WIDTH area / Total area |
| Fibre Coverage (WIDTH/ROI) | Fibre coverage in ROI based on ridge detection. | WIDTH area / ROI area |
| Avg Thickness (WIDTH, µm) | Average fibre thickness using ridge detection. | WIDTH area / Total fibre length |

### Skeleton and texture

| Result | Meaning | Calculation method |
|---|---|---|
| Total Length (µm) | Total length of fibres derived from ridge detection. | Total fibre length in µm. Pruned ridges (removing artifacts) used to calculate length in microns. |
| Avg Length (µm) | Average fibre length. | Total fibre length / Number of (endpoints + branchpoints) |
| Endpoints | Number of fibre endpoints. | Count of endpoints in ridge detection. |
| Endpoint Density (1/µm) | Endpoint count normalized to fibre length. | Number of endpoints / Total fibre length (µm) |
| Branchpoints | Number of fibre branchpoints. | Count of branchpoints in detected ridges. |
| Branchpoint Density (1/µm) | Branchpoint count normalized to fibre length. | Number of branchpoints / Total fibre length (µm) |
| Box-Counting Fractal Dimension | Indicator of structural complexity. Values in [1, 2] | It is an indicator for the capability of filling space. A more complex structure has a higher tendency to fill up the whole space, thus a higher fractal dimension.<br><br>The space filling tendency is measured as the slope of a linear fit to points represented as $(\log(1/\epsilon_{i}),\ \log(N(\epsilon_{i}))$ where $\epsilon_{i}\ $is the box size, and $N(\epsilon_{i})$ represents the count of boxes containing the structure (i.e. ridges). Typically, $\epsilon_{i}$ spans from 2 to one-quarter of the image\'s largest dimension. Refer to [here](https://imagej.nih.gov/ij/plugins/fraclac/FLHelp/BoxCounting.htm) for more information. |
| Curvature | Mean angle change across a user-defined window size. | Example: Curvature_20 refers to curvature within a 20px sliding window. |
| Lacunarity | Measure of gaps in the matrix. Larger values indicate larger gaps. | Calculated as $\ |
| Normalized Lacunarity | Lacunarity normalized to total image area (value in [0, 1]). | (Lacunarity - 1) / ((1/area ratio) - 1), where area ratio is the ratio of the number of ridge pixels to the total number of pixels. Refer to [here](https://sci-hub.mksa.top/10.1016/j.jsg.2010.08.010) for more detail. |

### Orientation

| Result | Meaning | Calculation method |
|---|---|---|
| Orient. Alignment | Coherency of fibre orientation across the entire image. | Orientation coherency is calculated as ${(\lambda}_{\max} - \lambda_{\min})\ /\ (\lambda_{\max} + \lambda_{\min})$ where $\lambda_{\min}$ and $\lambda_{\max}$ are respectively the largest eigenvalue (major axis) and smallest eigenvalue (minor axis) of the global gradient structure tensor. Refer to the [source code](https://github.com/Biomedical-Imaging-Group/OrientationJ/blob/master/src/main/java/OrientationJ_Dominant_Direction.java#L102) for more info. |
| Orient. Variance | Spread of fibre orientations (circular variation). | Using scipy.stats.circvar on blockwise orientations. |

## Gap Analysis Results

Gap sizes come from the maximum inscribed circles fitted between fibres; *All gaps* are measured on the whole image and *ROI gaps* only inside the segmented fibre region (intra-fibre gaps). Normalised values are divided by the image area or by the mean fibre width.

### All gaps

| Result | Meaning | Calculation method |
|---|---|---|
| Mean (All gaps area in µm²) | Mean Max Inscribed Circle area in entire image | Refer to [here](https://imagej.net/plugins/max-inscribed-circles) for more info about Max Inscribed Circles. |
| Normalized Mean (All gaps area) | Mean area of Max Inscribed Circle normalised to total image area |  |
| Std (All gaps area in µm²) | StDev of Max Inscribed Circle area in entire image (variability measurement of the Max Inscribed Circles) |  |
| Normalised Std (All gap area) | Normalised StDev of Max Inscribed Circle area to total image area |  |
| Median (All gaps area in µm²) | Median Inscribed Circle area in entire image |  |
| Percentile 5 (All gaps area in µm²) | $5^{th}$ percentile of median Max Inscribed Circle area |  |
| Percentile 95 (All gaps area in µm²) | $95^{th}$ percentile of median Max Inscribed Circle area |  |
| Mean (All gaps radius in µm) | Mean Max Inscribed Circle radius in entire image |  |
| Normalised Mean (All gaps radius) | Mean Max Inscribed Circle radius normalised to the square root of the total image area |  |
| Std (All gaps radius in µm) | StDev of Max Inscribed Circle radius in entire image (variability measurement of the Max Inscribed Circle radius) |  |
| Normalised Std (All gaps radius) | Normalised StDev of Max Inscribed Circle radius to the square root of the total image area |  |
| Median (All gaps radius in µm) | Median Max Inscribed Circle radius in entire image |  |
| Percentile 5 (All gaps radius in µm) | $5^{th}$ percentile of Max Inscribed Circle radius in entire image |  |
| Percentile 95 (All gaps radius in µm) | $95^{th}$ percentile of Max Inscribed Circle radius in entire image |  |
| Gap Circles Count (All) | Total number of Max Inscribed Circles in entire image |  |
| Gap density (Total, number/ µm²) | Number of Max Inscribed Circle per µm² |  |

### ROI (intra-fibre) gaps

| Result | Meaning | Calculation method |
|---|---|---|
| Mean (ROI gap area in µm²) | Mean Max Inscribed Circle area in ROI (segmented image) |  |
| Normalised Mean (ROI gap area) | Mean Max Inscribed Circle area normalised to ROI area |  |
| Std (ROI gap area in µm²) | StDev of Max Inscribed Circle area in ROI (variability measurement of the Max Inscribed Circle areas) |  |
| Normalised Std (ROI gap area) | Normalised StDev of Max Inscribed Circle area to ROI area |  |
| Percentile5 (ROI gaps area in µm) | $5^{th}$ percentile of Max Inscribed Circle area in ROI |  |
| Percentile95 (ROI gaps area in µm) | $95^{th}$ percentile of Max Inscribed Circle area in ROI |  |
| Median (ROI gaps area in µm) | Median Max Inscribed Circle area in ROI |  |
| Mean (ROI gap radius in µm) | Mean Max Inscribed Circle radius in ROI |  |
| Normalised Mean (ROI gap radius) | Mean Max Inscribed Circle radius normalised to the square root of the ROI area |  |
| Median (ROI gaps radius in µm) | Median Max Inscribed Circle radius in ROI |  |
| Std (ROI gap radius in µm) | StDev of Max Inscribed Circle radius in ROI (variability measurement of the Max Inscribed Circle radius) |  |
| Normalised Std (ROI gap radius) | Normalised StDev of Max Inscribed Circle radius to the square root of the ROI area |  |
| Percentile5 (ROI gaps radius in µm) | $5^{th}$ percentile of Max Inscribed Circle radius in ROI |  |
| Percentile95 (ROI gaps radius in µm) | $95^{th}$ percentile of Max Inscribed Circle radius in ROI |  |
| Gap Circles Count (ROI) | Total number of Max Inscribed Circles in ROI |  |
| Gap density (ROI, number/µm²) | Number of Max Inscribed Circles per µm² in ROI |  |
