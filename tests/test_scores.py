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
