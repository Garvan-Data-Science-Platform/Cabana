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
        assert core_stem("TMA1", 0, 3, info, "BF") == "8010718.vsi - TMA1_BF_A4Annotation (Tumour)_1"
        assert core_stem("TMA1", 0, 3, info, "POL", replicate=3) == "8010718.vsi - TMA1_POL_A4Annotation (Tumour)_3"

    def test_control_core_stem(self):
        info = load_array_map(1)[(0, 0)]
        assert core_stem("TMA1", 0, 0, info, "POL") == "Liver.vsi - TMA1_POL_A1Annotation (Liver)_1"

    def test_unmapped_core_stem(self):
        assert core_stem("TMA1", 1, 4, None, "BF") == "TMA1-r2c5.vsi - TMA1_BF_r2c5Annotation (Unmapped)_1"
        assert core_stem("TMA1", 1, 4, None) == "TMA1-r2c5.vsi - TMA1_r2c5Annotation (Unmapped)_1"

    def test_slide_name_is_sanitized(self):
        info = load_array_map(3)[(0, 1)]
        assert " - APGI-TMA-3-PicRed_BF_A2Annotation" in core_stem("APGI TMA 3 PicRed", 0, 1, info, "BF")

    def test_coreinfo_properties(self):
        info = CoreInfo(1, 1, "B3", "B", 3, "8012191", "2113", "", "", False)
        assert info.row_index == 1 and info.col_index == 2 and info.position == "B3"
        assert not info.is_control and info.identity() == "8012191_2113"


class TestMapConsistency:
    """Every (patient, ICGC sample) pair appears exactly three times per array,
    one core per sector on arrays 3 to 8 (arrays 1 and 2 shuffle patients
    between sectors). Two exceptions are printed in the PDF and verified
    against its page images: Array 6 patient 8070344 has two samples (3368
    and 3541), each in triplicate; Array 7 sample 8062699/3290 appears twice
    because a Kidney control occupies its sector-3 slot."""

    EXCEPTIONS = {(7, "8062699", "3290"): 2}

    @pytest.mark.parametrize("array", range(1, 9))
    def test_each_sample_in_triplicate(self, array):
        from collections import Counter
        cells = [c for c in load_array_map(array).values() if not c.empty and c.patient_id]
        counts = Counter((c.patient_id, c.icgc_id) for c in cells)
        for (pid, icgc), n in counts.items():
            assert n == self.EXCEPTIONS.get((array, pid, icgc), 3), (array, pid, icgc, n)

    @pytest.mark.parametrize("array", range(3, 9))
    def test_one_core_per_sector_on_later_arrays(self, array):
        from collections import Counter
        cells = [c for c in load_array_map(array).values() if not c.empty and c.patient_id]
        per_sector = Counter((c.patient_id, c.icgc_id, c.sector) for c in cells)
        assert max(per_sector.values()) == 1

    def test_control_counts(self):
        for array in range(1, 9):
            controls = [c for c in load_array_map(array).values() if c.is_control]
            assert 3 <= len(controls) <= 9, (array, len(controls))
