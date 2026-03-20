"""
Per-patient statistics and collagen risk score computation.

Provides:
- parse_image_name(): extract patient ID, image type, tissue type, ROI number
- generate_mean_std_sem(): aggregate ROI-level results to per-patient stats
- compute_scores(): compute Rigidity and Bundling collagen risk scores
"""

import re
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CHANNEL_NAMES = {'original', 'red', 'yellow', 'green'}
SHAPE_TYPES = {'ellipse', 'rectangle', 'polygon', 'line', 'polyline', 'points'}


def parse_image_name(filename):
    """
    Parse a Cabana Image column filename into structured components.

    Parameters
    ----------
    filename : str
        e.g. 'K324.vsi - 20x_BF_01Annotation (Tumor)_1_roi.png'

    Returns
    -------
    dict with keys: patient_id, image_type, tissue_type, roi_number
    """
    result = {
        'patient_id': '',
        'image_type': '',
        'tissue_type': '',
        'roi_number': '',
    }

    # Extract patient_id from .vsi or .czi prefix, else first token
    vsi_match = re.match(r'^(.+?)\.vsi', filename, re.IGNORECASE)
    if vsi_match:
        result['patient_id'] = vsi_match.group(1).strip()
    else:
        czi_match = re.match(r'^(.+?)\.czi', filename, re.IGNORECASE)
        if czi_match:
            result['patient_id'] = czi_match.group(1).strip()
        else:
            # First token before space or underscore
            token_match = re.match(r'^([^\s_]+)', filename)
            result['patient_id'] = token_match.group(1) if token_match else filename

    # Extract image type (BF or POL)
    type_match = re.search(r'[_\s](BF|POL|XPL)[_\s]', filename, re.IGNORECASE)
    if type_match:
        result['image_type'] = type_match.group(1).upper()

    # Extract tissue type: find all (...) groups after 'Annotation' and take the last one.
    # This handles cases like 'Annotation (Ellipse) (Tumor)' where the shape type precedes
    # the tissue class, as well as the simple 'Annotation (Tumor)' case.
    annotation_match = re.search(r'Annotation', filename, re.IGNORECASE)
    if annotation_match:
        after_annotation = filename[annotation_match.end():]
        paren_groups = re.findall(r'\(([^()]+)\)', after_annotation)
        if paren_groups:
            tissue = next(
                (g.strip() for g in reversed(paren_groups)
                 if g.strip().lower() not in SHAPE_TYPES),
                paren_groups[-1].strip()  # fallback to last if all are shape types
            )
            result['tissue_type'] = tissue

    # Extract ROI number: find the first _<digits> after the last ')' in the filename.
    # This anchors the search to after the tissue annotation parentheses.
    last_paren = filename.rfind(')')
    search_from = filename[last_paren:] if last_paren != -1 else filename
    roi_match = re.search(r'_(\d+)', search_from)
    if roi_match:
        result['roi_number'] = roi_match.group(1)

    # Check for known channel names anywhere in the filename (case-insensitive substring match).
    filename_lower = filename.lower()
    for channel in CHANNEL_NAMES:
        if channel in filename_lower:
            result['patient_id'] = f"{result['patient_id']}_{channel}"
            break

    return result


def generate_mean_std_sem(df, output_path=None):
    """
    Aggregate ROI-level results to per-patient MEAN, STD, SEM statistics.

    Parameters
    ----------
    df : pd.DataFrame
        ROI-level results with an 'Image' column.
    output_path : str or None
        If provided, write aggregated CSV to this path.

    Returns
    -------
    pd.DataFrame
        Per-patient aggregated DataFrame.
    """
    if 'Image' not in df.columns:
        logger.warning("No 'Image' column found in results. Skipping aggregation.")
        return None

    # Parse image names
    parsed = df['Image'].apply(parse_image_name)
    df = df.copy()
    df['_patient_id'] = parsed.apply(lambda x: x['patient_id'])
    df['_image_type'] = parsed.apply(lambda x: x['image_type'])
    df['_tissue_type'] = parsed.apply(lambda x: x['tissue_type'])

    # Identify numeric columns (exclude Image and internal columns)
    numeric_cols = [c for c in df.columns
                    if c not in ('Image', '_patient_id', '_image_type', '_tissue_type')
                    and pd.api.types.is_numeric_dtype(df[c])]

    rows = []
    for patient_id, group in df.groupby('_patient_id', sort=False):
        n = len(group)
        row = {
            'Patient': patient_id,
            'No. of ROIs': n,
            'Image Type': group['_image_type'].iloc[0],
            'Tissue Type': group['_tissue_type'].iloc[0],
        }
        for col in numeric_cols:
            values = group[col].dropna()
            mean = values.mean()
            std = values.std(ddof=0)
            sem = std / np.sqrt(len(values)) if len(values) > 0 else np.nan
            row[f'{col} MEAN'] = mean
            row[f'{col} STD'] = std
            row[f'{col} SEM'] = sem
        rows.append(row)

    agg_df = pd.DataFrame(rows)

    if output_path is not None:
        agg_df.to_csv(output_path, index=False)
        logger.info(f"Per-patient statistics written to {output_path}")

    return agg_df


def compute_scores(agg_df, output_path):
    """
    Compute Rigidity and Bundling collagen risk scores from patient-level means.

    Parameters
    ----------
    agg_df : pd.DataFrame
        Per-patient aggregated DataFrame from generate_mean_std_sem().
    output_path : str
        Path to write the scores CSV.
    """
    def z_score(series):
        std = series.std(ddof=0)
        return np.where(std == 0, 0.0, (series - series.mean()) / std)

    # Column mapping
    col_map = {
        'thickness': 'Avg Thickness (WIDTH, µm) MEAN',
        'alignment': 'Orient. Alignment (WIDTH) MEAN',
        'coverage': 'Fibre Coverage (WIDTH/ROI) MEAN',
        'endpoint_density': 'Endpoints Density (µm⁻¹) MEAN',
        'fractal_dim': 'Box-Counting Fractal Dimension MEAN',
    }

    # Check required columns exist
    missing = [k for k, v in col_map.items() if v not in agg_df.columns]
    if missing:
        logger.warning(f"Missing columns for score computation: {missing}. Skipping.")
        return

    # Extract variables
    thickness = agg_df[col_map['thickness']].astype(float)
    alignment = agg_df[col_map['alignment']].astype(float)
    coverage = agg_df[col_map['coverage']].astype(float)
    endpoint_density = agg_df[col_map['endpoint_density']].astype(float)
    fractal_dim = agg_df[col_map['fractal_dim']].astype(float)

    # Curvature: mean of all Curvature (win_sz=X) MEAN columns
    curvature_cols = [c for c in agg_df.columns if 'Curvature' in c and c.endswith('MEAN')]
    if curvature_cols:
        curvature = agg_df[curvature_cols].astype(float).mean(axis=1)
    else:
        logger.warning("No curvature columns found. Setting curvature to 0.")
        curvature = pd.Series(0.0, index=agg_df.index)

    # Z-score each variable
    z_thickness = z_score(thickness)
    z_alignment = z_score(alignment)
    z_coverage = z_score(coverage)
    z_endpoint = z_score(endpoint_density)
    z_fractal = z_score(fractal_dim)
    z_curvature = z_score(curvature)

    # Compute scores
    rigidity = z_thickness - z_curvature + z_alignment + z_coverage - z_endpoint
    bundling = z_thickness + z_alignment + z_coverage + z_fractal - z_endpoint

    # Build output DataFrame
    meta_cols = ['Patient', 'No. of ROIs', 'Image Type', 'Tissue Type']
    mean_cols = [c for c in agg_df.columns if c.endswith(' MEAN')]

    out_df = agg_df[meta_cols + mean_cols].copy()
    out_df['Rigidity Score'] = rigidity
    out_df['Bundling Score'] = bundling

    out_df.to_csv(output_path, index=False)
    logger.info(f"Collagen risk scores written to {output_path}")
