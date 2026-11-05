"""
Per-patient statistics and collagen risk score computation.

Provides:
- parse_image_name(): extract patient ID, image type, tissue type, ROI number
- generate_mean_std_sem(): aggregate ROI-level results to per-patient stats
- compute_scores(): compute Rigidity and Bundling collagen risk scores
- compact_results(): the short column set of QuantificationResults_Compact.csv
- combine_modalities(): one row per sample from a bright-field and a polarised-light run
"""

import re
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns of QuantificationResults_Compact.csv, in this order. Each dropped
# column of the full file restates one kept here: absolute areas follow from
# the percentages and the total image area, raw endpoint/branchpoint counts
# from their densities, gap areas from gap radii (area = pi r^2), the
# percentile and "normalized" gap statistics from the mean/std/median, and
# the whole-image/HDM orientation pairs rarely differ from the ROI/WIDTH ones.
# The scores (compute_scores) use only columns kept here.
COMPACT_COLUMNS = (
    'Image',
    'Total Image Area (µm²)',
    '% HDM Area',
    'Avg Thickness (HDM, µm)',
    'Mean Fibre Intensity (HDM)',
    '% ROI Area',
    'Avg Thickness (ROI, µm)',
    'Mean Fibre Intensity (ROI)',
    '% WIDTH Area',
    'Fibre Coverage (WIDTH/ROI)',
    'Avg Thickness (WIDTH, µm)',
    'Mean Fibre Intensity (WIDTH)',
    'Lacunarity',
    'Total Length (µm)',
    'Avg Length (µm)',
    'Endpoints Density (µm⁻¹)',
    'Branchpoints Density (µm⁻¹)',
    'Box-Counting Fractal Dimension',
    'Curvature (win_sz=*)',          # every curvature window, in file order
    'Orient. Alignment (ROI)',
    'Orient. Variance (ROI)',
    'Orient. Alignment (WIDTH)',
    'Orient. Variance (WIDTH)',
    'Mean (All gaps radius in µm)',
    'Std (All gaps radius in µm)',
    'Median (All gaps radius in µm)',
    'Gap Circles Count (All)',
    'Gap density (All, µm⁻²)',
    'Mean (ROI gaps radius in µm)',
    'Std (ROI gaps radius in µm)',
    'Median (ROI gaps radius in µm)',
    'Gap Circles Count (ROI)',
    'Gap density (ROI, µm⁻²)',
    'BRF Red-Orange (% Region)',
    'BRF Yellow (% Region)',
    'BRF Green (% Region)',
    'BRF Total (% Region)',
    'BRF Red-Orange (% Signal)',
    'BRF Yellow (% Signal)',
    'BRF Green (% Signal)',
    'Image Res. (µm/pixel)',
)
COMPACT_SUFFIX = '_Compact'


def compact_results(df):
    """Return the columns of ``COMPACT_COLUMNS`` present in ``df``, in that
    order. ``Curvature (win_sz=*)`` expands to every curvature window column.
    Columns a run did not produce (segmentation, HDM or gap analysis off)
    are left out rather than filled."""
    cols = []
    for name in COMPACT_COLUMNS:
        if name == 'Curvature (win_sz=*)':
            cols.extend(c for c in df.columns if c.startswith('Curvature (win_sz='))
        elif name in df.columns:
            cols.append(name)
    return df[cols]


def write_compact_results(df, results_csv):
    """Write ``compact_results(df)`` beside ``results_csv`` (``...Results.csv``
    → ``...Results_Compact.csv``) and return the path."""
    stem, ext = results_csv.rsplit('.', 1)
    path = f"{stem}{COMPACT_SUFFIX}.{ext}"
    compact_results(df).to_csv(path, index=False, float_format='%.3f')
    return path


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


# Modality token of an image name (bright-field or polarised light), as in
# parse_image_name; XPL (cross-polarised) counts as polarised light.
_MODALITY_TOKEN = re.compile(r'(?<=[_\s])(BF|POL|XPL)(?=[_\s])', re.IGNORECASE)


def modality_key(image_name):
    """The name shared by the bright-field and polarised-light images of one
    sample: ``image_name`` without its BF/POL/XPL token and the ``_roi.png``
    suffix Cabana adds, e.g. ``A1420.vsi_POL_S01_roi.png`` -> ``A1420.vsi_S01``
    and ``1734.vsi - TMA1_BF_A4Annotation (Tumor)_1_roi.png`` ->
    ``1734.vsi - TMA1_A4Annotation (Tumor)_1``. None without a token."""
    stem = re.sub(r'(_roi)?\.(png|tif|tiff|jpg|jpeg)$', '', str(image_name), flags=re.IGNORECASE)
    m = _MODALITY_TOKEN.search(stem)
    if m is None:
        return None
    key = stem[:m.start()] + stem[m.end():]
    return re.sub(r'([_\s])\1+', r'\1', key).strip(' _')


def combine_modalities(bf_csv, pol_csv, output_path=None):
    """Join the results of a bright-field run and a polarised-light run of the
    same samples into one row per sample.

    Rows are matched on :func:`modality_key` of their ``Image``; the columns
    of each run are prefixed ``BF: `` and ``POL: ``. Images without a partner
    keep a row with the other side empty. Returns ``(combined, counts)`` with
    counts of ``matched``, ``bf_only`` and ``pol_only`` rows; writes
    ``output_path`` when given. Raises ValueError when an image name has no
    BF/POL token or two images of one run share a sample name.
    """
    sides = []
    for path, prefix in ((bf_csv, 'BF'), (pol_csv, 'POL')):
        df = pd.read_csv(path)
        keys = df['Image'].map(modality_key)
        if keys.isna().any():
            bad = df.loc[keys.isna(), 'Image'].tolist()
            raise ValueError(f"{prefix} results: no BF/POL in the image name of {', '.join(bad[:3])}"
                             + (" ..." if len(bad) > 3 else ""))
        dup = keys[keys.duplicated()].unique().tolist()
        if dup:
            raise ValueError(f"{prefix} results: several images of sample {', '.join(dup[:3])}"
                             + (" ..." if len(dup) > 3 else ""))
        df = df.add_prefix(f'{prefix}: ')
        df.insert(0, 'Sample', keys.values)
        sides.append(df)
    combined = sides[0].merge(sides[1], on='Sample', how='outer', indicator=True)
    counts = {'matched': int((combined['_merge'] == 'both').sum()),
              'bf_only': int((combined['_merge'] == 'left_only').sum()),
              'pol_only': int((combined['_merge'] == 'right_only').sum())}
    combined = combined.drop(columns='_merge')
    combined.insert(1, 'Patient', combined['Sample'].map(lambda s: parse_image_name(s)['patient_id']))
    if output_path:
        combined.to_csv(output_path, index=False, float_format='%.3f')
    if counts['bf_only'] or counts['pol_only']:
        logger.warning(f"{counts['bf_only']} BF and {counts['pol_only']} POL images have no partner "
                       f"in the other run.")
    return combined, counts
