"""Tests for cabana/scores.py — parse_image_name, generate_mean_std_sem, compute_scores."""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cabana.scores import parse_image_name, generate_mean_std_sem, compute_scores


# ---------------------------------------------------------------------------
# parse_image_name
# ---------------------------------------------------------------------------

class TestParseImageName:
    def test_vsi_patient_id(self):
        result = parse_image_name("K324.vsi - 20x_BF_01Annotation (Tumor)_1_roi.png")
        assert result['patient_id'] == 'K324'

    def test_czi_patient_id(self):
        result = parse_image_name("patient001.czi - 10x_POL_01Annotation (Stroma)_2_roi.png")
        assert result['patient_id'] == 'patient001'

    def test_image_type_bf(self):
        result = parse_image_name("K324.vsi - 20x_BF_01Annotation (Tumor)_1_roi.png")
        assert result['image_type'] == 'BF'

    def test_image_type_pol(self):
        result = parse_image_name("K324.vsi - 20x_POL_01Annotation (Tumor)_1_roi.png")
        assert result['image_type'] == 'POL'

    def test_tissue_type_tumor(self):
        result = parse_image_name("K324.vsi - 20x_BF_01Annotation (Tumor)_1_roi.png")
        assert result['tissue_type'] == 'Tumor'

    def test_tissue_type_stroma(self):
        result = parse_image_name("K324.vsi - 20x_BF_01Annotation (Stroma)_1_roi.png")
        assert result['tissue_type'] == 'Stroma'

    def test_roi_number(self):
        result = parse_image_name("K324.vsi - 20x_BF_01Annotation (Tumor)_3_roi.png")
        assert result['roi_number'] == '3'

    def test_shape_type_skipped(self):
        # 'Annotation (Ellipse) (Tumor)' — should skip Ellipse, take Tumor
        result = parse_image_name("K324.vsi - 20x_BF_01Annotation (Ellipse) (Tumor)_1_roi.png")
        assert result['tissue_type'] == 'Tumor'

    def test_returns_dict_with_all_keys(self):
        result = parse_image_name("anything.png")
        assert 'patient_id' in result
        assert 'image_type' in result
        assert 'tissue_type' in result
        assert 'roi_number' in result

    def test_channel_name_appended(self):
        result = parse_image_name("K324.vsi - 20x_BF_01Annotation (Tumor)_1_red_roi.png")
        assert 'red' in result['patient_id']

    def test_no_vsi_czi_uses_first_token(self):
        result = parse_image_name("Patient123_BF_01_roi.png")
        assert result['patient_id'] == 'Patient123'

    def test_empty_image_type_if_not_found(self):
        result = parse_image_name("K324.vsi - 20x_01Annotation (Tumor)_1_roi.png")
        assert result['image_type'] == ''


# ---------------------------------------------------------------------------
# generate_mean_std_sem
# ---------------------------------------------------------------------------

class TestGenerateMeanStdSem:
    def _make_df(self):
        return pd.DataFrame({
            'Image': [
                'K1.vsi - BF _01Annotation (Tumor)_1_roi.png',
                'K1.vsi - BF _01Annotation (Tumor)_2_roi.png',
                'K2.vsi - BF _01Annotation (Tumor)_1_roi.png',
            ],
            'Length': [10.0, 20.0, 30.0],
            'Width': [1.0, 2.0, 3.0],
        })

    def test_returns_dataframe(self):
        df = self._make_df()
        result = generate_mean_std_sem(df)
        assert isinstance(result, pd.DataFrame)

    def test_has_patient_column(self):
        df = self._make_df()
        result = generate_mean_std_sem(df)
        assert 'Patient' in result.columns

    def test_aggregates_by_patient(self):
        df = self._make_df()
        result = generate_mean_std_sem(df)
        assert len(result) == 2  # K1 and K2

    def test_mean_column_exists(self):
        df = self._make_df()
        result = generate_mean_std_sem(df)
        assert 'Length MEAN' in result.columns

    def test_std_column_exists(self):
        df = self._make_df()
        result = generate_mean_std_sem(df)
        assert 'Length STD' in result.columns

    def test_sem_column_exists(self):
        df = self._make_df()
        result = generate_mean_std_sem(df)
        assert 'Length SEM' in result.columns

    def test_mean_value_correct(self):
        df = self._make_df()
        result = generate_mean_std_sem(df)
        k1_row = result[result['Patient'] == 'K1']
        assert abs(k1_row['Length MEAN'].iloc[0] - 15.0) < 1e-6

    def test_missing_image_column_returns_none(self):
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        result = generate_mean_std_sem(df)
        assert result is None

    def test_writes_csv_if_output_path(self, tmp_path):
        df = self._make_df()
        out = str(tmp_path / "agg.csv")
        generate_mean_std_sem(df, output_path=out)
        assert os.path.exists(out)

    def test_no_of_rois_correct(self):
        df = self._make_df()
        result = generate_mean_std_sem(df)
        k1_row = result[result['Patient'] == 'K1']
        assert k1_row['No. of ROIs'].iloc[0] == 2


# ---------------------------------------------------------------------------
# compute_scores
# ---------------------------------------------------------------------------

class TestComputeScores:
    def _make_agg_df(self, n=5):
        """Create a minimal aggregated DataFrame with required columns."""
        np.random.seed(42)
        return pd.DataFrame({
            'Patient': [f'P{i}' for i in range(n)],
            'No. of ROIs': [3] * n,
            'Image Type': ['BF'] * n,
            'Tissue Type': ['Tumor'] * n,
            'Avg Thickness (WIDTH, µm) MEAN': np.random.rand(n) * 5,
            'Orient. Alignment (WIDTH) MEAN': np.random.rand(n),
            'Fibre Coverage (WIDTH/ROI) MEAN': np.random.rand(n),
            'Endpoints Density (µm⁻¹) MEAN': np.random.rand(n) * 0.1,
            'Box-Counting Fractal Dimension MEAN': np.random.rand(n) + 1,
            'Curvature (win_sz=10) MEAN': np.random.rand(n) * 0.5,
        })

    def test_creates_csv(self, tmp_path):
        df = self._make_agg_df()
        out = str(tmp_path / "scores.csv")
        compute_scores(df, out)
        assert os.path.exists(out)

    def test_rigidity_score_column_in_csv(self, tmp_path):
        df = self._make_agg_df()
        out = str(tmp_path / "scores.csv")
        compute_scores(df, out)
        result = pd.read_csv(out)
        assert 'Rigidity Score' in result.columns

    def test_bundling_score_column_in_csv(self, tmp_path):
        df = self._make_agg_df()
        out = str(tmp_path / "scores.csv")
        compute_scores(df, out)
        result = pd.read_csv(out)
        assert 'Bundling Score' in result.columns

    def test_scores_are_numeric(self, tmp_path):
        df = self._make_agg_df()
        out = str(tmp_path / "scores.csv")
        compute_scores(df, out)
        result = pd.read_csv(out)
        assert pd.api.types.is_numeric_dtype(result['Rigidity Score'])
        assert pd.api.types.is_numeric_dtype(result['Bundling Score'])

    def test_missing_columns_no_crash(self, tmp_path):
        # Missing required columns → should log warning and return without crashing
        df = pd.DataFrame({'Patient': ['A', 'B'], 'No. of ROIs': [1, 1]})
        out = str(tmp_path / "scores.csv")
        compute_scores(df, out)  # Should not raise

    def test_output_row_count_matches_input(self, tmp_path):
        n = 6
        df = self._make_agg_df(n=n)
        out = str(tmp_path / "scores.csv")
        compute_scores(df, out)
        result = pd.read_csv(out)
        assert len(result) == n


# ---------------------------------------------------------------------------
# Decimal formatting consistency with QuantificationResults.csv
# ---------------------------------------------------------------------------

def _decimal_places(text):
    """Return the number of digits after the decimal point in a numeric token,
    or 0 if no decimal point is present."""
    if '.' not in text:
        return 0
    return len(text.split('.', 1)[1])


def _max_decimals_in_csv(path):
    """Return the largest number of decimal places observed in any numeric
    cell across the file (excluding scientific-notation / non-numeric)."""
    max_dp = 0
    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline()  # skip column names
        for line in f:
            for cell in line.rstrip('\n').split(','):
                cell = cell.strip()
                if not cell or cell.lower() in {'nan', 'inf', '-inf'}:
                    continue
                # Skip scientific-notation tokens; %.3f never produces them.
                if 'e' in cell or 'E' in cell:
                    continue
                # Only consider tokens that parse as a float.
                try:
                    float(cell)
                except ValueError:
                    continue
                max_dp = max(max_dp, _decimal_places(cell))
    return max_dp


class TestCsvDecimalFormatting:
    """Per-patient stats and scores CSVs use %.3f, matching QuantificationResults.csv."""

    def _make_df(self):
        return pd.DataFrame({
            'Image': [
                'K1.vsi - BF _01Annotation (Tumor)_1_roi.png',
                'K1.vsi - BF _01Annotation (Tumor)_2_roi.png',
                'K2.vsi - BF _01Annotation (Tumor)_1_roi.png',
                'K2.vsi - BF _01Annotation (Tumor)_2_roi.png',
            ],
            # Values that produce many decimals when written without format.
            'Length': [10.123456789, 20.987654321, 30.111111111, 40.222222222],
            'Width':  [1.0 / 3.0, 2.0 / 3.0, 1.0 / 7.0, 2.0 / 7.0],
        })

    def test_mean_std_sem_csv_uses_three_decimals(self, tmp_path):
        out = str(tmp_path / "stats.csv")
        generate_mean_std_sem(self._make_df(), output_path=out)
        # Every numeric cell that has decimals should have exactly 3.
        # %.3f format guarantees no cell exceeds 3 decimals.
        assert _max_decimals_in_csv(out) <= 3

    def test_mean_std_sem_no_full_precision_floats(self, tmp_path):
        """Without the float_format fix the file would contain values like
        '0.4444444444444444' (16 decimal places). Verify the cap is < 4."""
        out = str(tmp_path / "stats.csv")
        generate_mean_std_sem(self._make_df(), output_path=out)
        with open(out, 'r', encoding='utf-8') as f:
            content = f.read()
        # No cell should have 4+ decimal places.
        # A simple regex: digit, dot, then 4+ digits = bad.
        import re
        assert re.search(r'\d\.\d{4,}', content) is None, (
            f"Found floats with >3 decimals in {out}:\n{content[:500]}"
        )

    def test_scores_csv_uses_three_decimals(self, tmp_path):
        # Build aggregated DF with required columns, then write scores.
        np.random.seed(7)
        n = 6
        agg_df = pd.DataFrame({
            'Patient': [f'P{i}' for i in range(n)],
            'No. of ROIs': [3] * n,
            'Image Type': ['BF'] * n,
            'Tissue Type': ['Tumor'] * n,
            'Avg Thickness (WIDTH, µm) MEAN': np.random.rand(n) * 5,
            'Orient. Alignment (WIDTH) MEAN': np.random.rand(n),
            'Fibre Coverage (WIDTH/ROI) MEAN': np.random.rand(n),
            'Endpoints Density (µm⁻¹) MEAN': np.random.rand(n) * 0.1,
            'Box-Counting Fractal Dimension MEAN': np.random.rand(n) + 1,
            'Curvature (win_sz=10) MEAN': np.random.rand(n) * 0.5,
        })
        out = str(tmp_path / "scores.csv")
        compute_scores(agg_df, out)
        assert _max_decimals_in_csv(out) <= 3

    def test_scores_csv_no_full_precision_floats(self, tmp_path):
        np.random.seed(7)
        n = 6
        agg_df = pd.DataFrame({
            'Patient': [f'P{i}' for i in range(n)],
            'No. of ROIs': [3] * n,
            'Image Type': ['BF'] * n,
            'Tissue Type': ['Tumor'] * n,
            'Avg Thickness (WIDTH, µm) MEAN': np.random.rand(n) * 5,
            'Orient. Alignment (WIDTH) MEAN': np.random.rand(n),
            'Fibre Coverage (WIDTH/ROI) MEAN': np.random.rand(n),
            'Endpoints Density (µm⁻¹) MEAN': np.random.rand(n) * 0.1,
            'Box-Counting Fractal Dimension MEAN': np.random.rand(n) + 1,
            'Curvature (win_sz=10) MEAN': np.random.rand(n) * 0.5,
        })
        out = str(tmp_path / "scores.csv")
        compute_scores(agg_df, out)
        with open(out, 'r', encoding='utf-8') as f:
            content = f.read()
        import re
        assert re.search(r'\d\.\d{4,}', content) is None, (
            f"Found floats with >3 decimals in {out}:\n{content[:500]}"
        )

    def test_integer_columns_unchanged(self, tmp_path):
        """%.3f only applies to floats; integer columns like 'No. of ROIs'
        must still serialize as integers (no '3.000')."""
        out = str(tmp_path / "stats.csv")
        generate_mean_std_sem(self._make_df(), output_path=out)
        df_back = pd.read_csv(out)
        assert df_back['No. of ROIs'].dtype.kind in ('i', 'u')

    def test_string_columns_unchanged(self, tmp_path):
        out = str(tmp_path / "stats.csv")
        generate_mean_std_sem(self._make_df(), output_path=out)
        df_back = pd.read_csv(out)
        assert df_back['Patient'].iloc[0] == 'K1'


class TestTMACoreNames:
    """Stems written by the TMA export follow the slide-export pattern."""

    def test_patient_core(self):
        r = parse_image_name("8010718.vsi - TMA1_BF_A3Annotation (Tumour)_2_roi.png")
        assert (r["patient_id"], r["image_type"], r["tissue_type"], r["roi_number"]) == ("8010718", "BF", "Tumour", "2")

    def test_sanitised_by_batch_pipeline(self):
        r = parse_image_name("8010718.vsi_-_TMA1_POL_A3Annotation_(PNET)_1_roi.png")
        assert (r["patient_id"], r["image_type"], r["tissue_type"]) == ("8010718", "POL", "PNET")

    def test_control_and_unmapped(self):
        assert parse_image_name("Liver.vsi - TMA1_BF_A1Annotation (Liver)_1_roi.png")["patient_id"] == "Liver"
        assert parse_image_name("TMA1-r2c5.vsi - TMA1_BF_r2c5Annotation (Unmapped)_1_roi.png")["patient_id"] == "TMA1-r2c5"

    def test_replicates_aggregate_per_patient(self):
        import pandas as pd
        df = pd.DataFrame({
            "Image": [f"8010718.vsi - TMA1_BF_{p}Annotation (Tumour)_{i}_roi.png" for i, p in enumerate(("A3", "E3", "I3"), 1)]
                     + ["8010720.vsi - TMA1_BF_A4Annotation (Tumour)_1_roi.png"],
            "Total Length (µm)": [10.0, 12.0, 14.0, 5.0],
        })
        agg = generate_mean_std_sem(df)
        assert sorted(agg["Patient"]) == ["8010718", "8010720"]
        assert agg.set_index("Patient").loc["8010718", "Total Length (µm) MEAN"] == 12.0
