# What is Cabana?

**CABANA** (**C**oll**A**gen Fi**b**re **ANA**lyzer) is a specialized software tool developed for the analysis of collagen fiber architecture in biological images. It provides a comprehensive solution for researchers studying collagen structures in tissues through immunohistochemistry (IHC) and fluorescence microscopy.

## Overview

Cabana combines several image processing techniques to detect, quantify, and analyze collagen fibers in microscopy images. The software processes images through multiple stages:

1. **Segmentation**: Extracts regions of interest (ROIs) from the background based on color and other low-level features
2. **Ridge Detection**: Identifies fiber structures using advanced ridge detection algorithms
3. **Orientation Analysis**: Quantifies the orientation and alignment of fibers
4. **Skeleton Analysis**: Evaluates the topological properties of fiber networks
5. **Gap Analysis**: Measures spaces between and within fiber structures

For tissue micro-arrays, a preprocessing step reads the whole-slide scan, fits every core and exports per-core images and masks named by patient, ready for batch analysis.

## Key Features

- Compatible with both IHC (Picrosirius Red) and fluorescence microscopy images
- Automated detection and quantification of collagen fibers
- Analysis of fiber orientation, alignment, and distribution
- Calculation of fiber thickness, length, and branching patterns
- Gap analysis to quantify spaces within the fiber network
- Comprehensive statistical outputs for quantitative analysis
- Tissue micro-array preprocessing: native Olympus VSI reading, core fitting and per-core export named by patient
- Optional ROI masks that confine every measurement to a region of interest
- Available in both desktop and cloud-based versions

## Application Areas

Cabana is particularly useful for researchers in:

- Cancer biology (studying tumor microenvironment)
- Wound healing and tissue repair
- Fibrosis research
- Connective tissue disorders
- Biomaterial development and evaluation
- Regenerative medicine

## Versions and Availability

Cabana is available in two versions:

1. **Desktop Version**: For processing smaller datasets on local machines
2. **Google Cloud Version**: For large-scale processing of datasets exceeding 10,000 images (currently available only to Garvan Institute staff and students)
