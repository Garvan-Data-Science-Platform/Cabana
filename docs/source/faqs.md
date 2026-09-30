# FAQs

**1. Can I analyze images with different pixel resolutions?**  
Yes, but it's not recommended. While Cabana can process images of varying resolutions, ridge detection results may vary significantly. For consistent and comparable analysis, use images with the same pixel resolution.

**2. What is the maximum supported image size?**  
Images larger than `Segmentation: Max Size` squared (2048×2048 pixels by default) are split into blocks, or skipped when large images are ignored. Raise `Max Size` in the parameter file (for example to 6000 for TMA cores) to analyse each image whole, and consider `Segmentation: Patch Size` so the segmentation network sees large images at a useful resolution.

**3. Why was my image rejected due to a dark background?**  
Images with more than 99% of pixels having intensity values below 5 are automatically rejected. Such images are considered to lack sufficient regions of interest for analysis.

**4. How are images processed in Cabana?**  
Images are processed in batches (5 by default, adjustable on the Batch Run page). This setup allows efficient processing and easier error recovery. A status line above the progress bar shows the current batch, stage and image.

**5. What happens if Cabana crashes during processing?**  
If the program crashes before finishing, it can be restarted and will resume from the last successfully processed batch. Cabana automatically creates a checkpoint file to support this recovery.

**6. How should I name my images for the Stats and Scores outputs?**  
Follow the QuPath export pattern `<patient>.vsi - <anything>_<BF|POL>_<anything>Annotation (<sample type>)_<ROI number>`, for example `K324.vsi - 20x_BF_01Annotation (Tumor)_1.tif`. The patient ID is read from before `.vsi`, the image type from `BF`/`POL`, the sample type from the last bracket and the ROI number from the end. Cores exported by the TMA page are already named this way. See [Image naming for per-patient statistics](workflow.md#image-naming-for-per-patient-statistics).

**7. Does Cabana use my GPU?**  
The segmentation network runs on an NVIDIA GPU when CUDA is available, on the Apple GPU (MPS) on Apple silicon Macs, and otherwise on the CPU, which is much slower for large images. Set the environment variable `CABANA_DEVICE` to `cpu`, `cuda` or `mps` to override the choice.
