[//]: # (# CABANA - Collagen Fibre Analyzer)
![image](cabana.png)
[![PyPI version](https://badge.fury.io/py/cabana.svg)](https://pypi.org/project/cabana/)
[![Documentation Status](https://readthedocs.org/projects/cabana/badge/?version=latest)](https://cabana.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

CABANA (CollAgen FiBre ANAlyzer) is a comprehensive toolkit for analyzing collagen fibre architecture in immunohistochemistry (IHC) and fluorescence microscopy images. It provides:

- Automated detection and quantification of collagen fibres
- Analysis of fibre orientation, length, and thickness
- Gap analysis for inter-fibre and intra-fibre spaces
- Interactive parameter optimization interface
- Batch processing for large datasets
- Tissue micro-array (TMA) preprocessing: native Olympus VSI reading, core fitting, per-core images and masks named by patient (`cabana-tma`)
- Optional ROI masks that confine every measurement to a region of interest

CABANA combines advanced computer vision algorithms with an intuitive user interface, making it accessible to researchers without extensive programming experience while providing detailed quantitative data for comprehensive analysis of collagen architecture.

## System Requirements

- **Processor (CPU):** Intel Core i7, AMD Ryzen 7, or higher recommended
- **Memory (RAM):** Minimum 16 GB, 32 GB recommended
- **GPU:** NVIDIA GeForce GTX/RTX series with at least 10 GB VRAM recommended
- **Python:** 3.12
- **Apple silicon:** the segmentation network runs on the Apple GPU (MPS) automatically


## Quick Installation

Install CABANA using pip:

```bash
pip install -U cabana
```
or install the latest version on GitHub:

```bash
pip install git+https://github.com/lxfhfut/Cabana.git
```

After installation, you can launch the CABANA GUI by running:

```bash
python -m cabana        # or: cabana-gui
```

Tissue micro-array slides can also be cut into per-core images from the command line:

```bash
cabana-tma "APGI TMA 1 PicRed.vsi" out/TMA1 --array 1
```

Alternatively, you can import CABANA in your Python code for customized analysis (see the [Python API](https://cabana.readthedocs.io/en/latest/api.html) page):

```python
from cabana import Cabana, BatchProcessor, TMAPreprocessor
```

## Documentation

Comprehensive documentation is available at:
[https://cabana.readthedocs.io/en/latest/](https://cabana.readthedocs.io/en/latest/)

## Tutorials

We provide an interactive Jupyter notebook tutorial to help you get started:
[CABANA Tutorial](https://cabana.readthedocs.io/en/latest/_static/tutorial.ipynb)

## Citation

If you use CABANA in your research, please cite:
<!--
```
Magenau A, et al. (2025). CABANA: A comprehensive toolkit for collagen fibre analysis in biomedical imaging. Journal of XYZ. DOI: XYZ
```-->

## License

CABANA is released under the MIT License.

## Contact

For questions and support, please contact:
- Xufeng Lin - <x.lin@garvan.org.au>
- Astrid Magenau - <a.magenau@garvan.org.au>
