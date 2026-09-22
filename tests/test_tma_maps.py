"""Tests for cabana/tma_maps.py: ICGC array map loading and core naming."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cabana.tma_maps import (MAP_COLS, MAP_ROWS, CoreInfo, available_arrays, core_stem,
                             load_array_map, occupancy_grid, sanitize_token)


class TestMapFile:
    def test_all_eight_arrays_present(self):
        assert available_arrays() == [1, 2, 3, 4, 5, 6, 7, 8]

    @pytest.mark.parametrize("array", range(1, 9))
    def test_every_cell_present(self, array):
        cells = load_array_map(array)
        assert len(cells) == MAP_ROWS * MAP_COLS
        assert set(cells) == {(r, c) for r in range(MAP_ROWS) for c in range(MAP_COLS)}

    @pytest.mark.parametrize("array,n_cores", [(1, 88), (2, 88), (3, 96), (4, 96),
                                               (5, 96), (6, 96), (7, 96), (8, 96)])
    def test_core_counts(self, array, n_cores):
        assert occupancy_grid(array).sum() == n_cores

    def test_arrays_1_and_2_last_row_empty(self):
        for array in (1, 2):
            grid = occupancy_grid(array)
            assert not grid[MAP_ROWS - 1].any()
            assert grid[:MAP_ROWS - 1].all()

    @pytest.mark.parametrize("array", range(1, 9))
    def test_filled_cells_have_identity(self, array):
        for info in load_array_map(array).values():
            if info.empty:
                continue
            assert info.tissue or (info.patient_id and info.icgc_id), info

    def test_spot_checks_against_pdf(self):
        a1 = load_array_map(1)
        assert a1[(0, 0)].tissue == "Liver"
        assert a1[(0, 3)].patient_id == "8010718" and a1[(0, 3)].icgc_id == "1734"
        assert a1[(3, 6)].tissue == "Brain"
        assert a1[(4, 0)].label == "A5" and a1[(4, 0)].sector == 2
        a6 = load_array_map(6)
        assert a6[(0, 3)].patient_id == "8046501" and a6[(0, 3)].note == "MCN"
        assert a6[(8, 6)].tissue == "Kidney/ muscle"
        a8 = load_array_map(8)
        assert a8[(0, 0)].label == "D3" and a8[(0, 0)].patient_id == "8070444"

    def test_unknown_array_raises(self):
        with pytest.raises(KeyError):
            load_array_map(9)


class TestNaming:
    def test_sanitize_token(self):
        assert sanitize_token("Salivary gland") == "Salivary-gland"
        assert sanitize_token("Kidney/ muscle") == "Kidney-muscle"
        assert sanitize_token("  ") == "unknown"

    def test_patient_core_stem(self):
        info = load_array_map(1)[(0, 3)]
        assert core_stem("TMA1", 0, 3, info, "BF") == "TMA1_A4_8010718_1734_BF"

    def test_control_core_stem(self):
        info = load_array_map(1)[(0, 0)]
        assert core_stem("TMA1", 0, 0, info, "POL") == "TMA1_A1_Liver_POL"

    def test_unmapped_core_stem(self):
        assert core_stem("TMA1", 1, 4, None, "BF") == "TMA1_r2c5_unknown_BF"
        assert core_stem("TMA1", 1, 4, None) == "TMA1_r2c5_unknown"

    def test_slide_name_is_sanitized(self):
        info = load_array_map(3)[(0, 1)]
        assert core_stem("APGI TMA 3 PicRed", 0, 1, info, "BF").startswith("APGI-TMA-3-PicRed_A2_")

    def test_coreinfo_properties(self):
        info = CoreInfo(1, 1, "B3", "B", 3, "8012191", "2113", "", "", False)
        assert info.row_index == 1 and info.col_index == 2 and info.position == "B3"
        assert not info.is_control and info.identity() == "8012191_2113"
