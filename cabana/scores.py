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

CHANNEL_NAMES = ('original', 'red', 'yellow', 'green')     # fixed order: matching must be deterministic
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

    # Export scripts sometimes prefix the name with '_' or a space; never let
    # that leak into the patient ID.
    result['patient_id'] = result['patient_id'].strip(' _-')

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

    # A channel token ("red", "green", ...) delimited by separators, e.g. "_1_red_roi.png";
    # a substring match would turn "Fred" or "registered" into a channel.
    m = re.search(r'(?:^|[\s_.-])(' + '|'.join(CHANNEL_NAMES) + r')(?=[\s_.-]|$)', filename, re.IGNORECASE)
    if m:
        result['patient_id'] = f"{result['patient_id']}_{m.group(1).lower()}"

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
            # sample statistics: a single ROI has no spread estimate (NaN), not zero
            std = values.std(ddof=1) if len(values) > 1 else np.nan
            sem = std / np.sqrt(len(values)) if len(values) > 1 else np.nan
            row[f'{col} MEAN'] = mean
            row[f'{col} STD'] = std
            row[f'{col} SEM'] = sem
        rows.append(row)

    agg_df = pd.DataFrame(rows)

    if output_path is not None:
        # Match QuantificationResults.csv float formatting so the three
        # result CSVs render with identical decimal precision.
        agg_df.to_csv(output_path, index=False, float_format="%.3f")
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

    EPS = 1e-6

    # Column mapping (fractal dimension not used in final score formulas)
    col_map = {
        'thickness': 'Avg Thickness (WIDTH, µm) MEAN',
        'alignment': 'Orient. Alignment (WIDTH) MEAN',
        'coverage': 'Fibre Coverage (WIDTH/ROI) MEAN',
        'endpoint_density': 'Endpoints Density (µm⁻¹) MEAN',
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

    # κ_ms: mean of all Curvature (win_sz=X) MEAN columns
    curvature_cols = [c for c in agg_df.columns if 'Curvature' in c and c.endswith('MEAN')]
    if curvature_cols:
        curvature = agg_df[curvature_cols].astype(float).mean(axis=1)
    else:
        logger.warning("No curvature columns found. Setting curvature to 0.")
        curvature = pd.Series(0.0, index=agg_df.index)

    # Z-score with log transforms per spec:
    # R* = z(log(T+ε)) - z(log(κ_ms+ε)) + z(A) + z(log(C+ε)) - z(E)
    # B* = z(log(T+ε)) + z(A) + z(log(C+ε)) - z(E)
    z_log_thickness = z_score(np.log(thickness + EPS))
    z_alignment     = z_score(alignment)
    z_log_coverage  = z_score(np.log(coverage + EPS))
    z_endpoint      = z_score(endpoint_density)
    z_log_curvature = z_score(np.log(curvature + EPS))

    # Compute scores
    rigidity = z_log_thickness - z_log_curvature + z_alignment + z_log_coverage - z_endpoint
    bundling = z_log_thickness + z_alignment + z_log_coverage - z_endpoint

    # Build output DataFrame
    meta_cols = ['Patient', 'No. of ROIs', 'Image Type', 'Tissue Type']
    mean_cols = [c for c in agg_df.columns if c.endswith(' MEAN')]

    out_df = agg_df[meta_cols + mean_cols].copy()
    out_df['Rigidity Score'] = rigidity
    out_df['Bundling Score'] = bundling

    out_df.to_csv(output_path, index=False, float_format="%.3f")
    logger.info(f"Collagen risk scores written to {output_path}")
