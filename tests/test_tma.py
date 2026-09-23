"""Tests for cabana/tma.py: core fitting, grid inference, orientation, export."""

import csv
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cabana.tma import (TMAPreprocessor, fit_cores, infer_grid, match_orientation,
                        merge_grid_duplicates)
from cabana.tma_maps import occupancy_grid

PX_UM = 4.0            # synthetic slide resolution
CORE_UM = 1000.0
PITCH_PX = 400         # 1.6 mm pitch at 4 µm/px
RADIUS_PX = CORE_UM / PX_UM / 2


def synthetic_slide(occupancy, pitch=PITCH_PX, radius=RADIUS_PX, jitter=0, seed=0):
    """White slide with pink discs where ``occupancy`` is True. Returns (img, centres)."""
    rng = np.random.default_rng(seed)
    n_rows, n_cols = occupancy.shape
    h = int((n_rows + 1) * pitch)
    w = int((n_cols + 1) * pitch)
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    centres = {}
    for r in range(n_rows):
        for c in range(n_cols):
            if not occupancy[r, c]:
                continue
            cx = int((c + 1) * pitch + rng.integers(-jitter, jitter + 1))
            cy = int((r + 1) * pitch + rng.integers(-jitter, jitter + 1))
            cv2.circle(img, (cx, cy), int(radius), (190, 150, 230), -1)   # BGR pink
            centres[(r, c)] = (cx, cy)
    return img, centres


class TestFitCores:
    def test_finds_every_disc(self):
        occ = np.ones((3, 4), dtype=bool)
        img, centres = synthetic_slide(occ)
        circles = fit_cores(img, PX_UM, CORE_UM)
        assert len(circles) == 12
        found = np.array([[c[0], c[1]] for c in circles])
        for (cx, cy) in centres.values():
            d = np.sqrt(((found - [cx, cy]) ** 2).sum(1)).min()
            assert d < 3
        for c in circles:
            assert abs(c[2] - RADIUS_PX) < 0.06 * RADIUS_PX
            assert c[3] > 0.95

    def test_empty_slide(self):
        img = np.full((800, 800, 3), 245, dtype=np.uint8)
        assert fit_cores(img, PX_UM, CORE_UM) == []

    def test_ignores_small_debris(self):
        occ = np.ones((2, 2), dtype=bool)
        img, _ = synthetic_slide(occ)
        cv2.circle(img, (60, 60), 6, (190, 150, 230), -1)   # speck far from cores
        circles = fit_cores(img, PX_UM, CORE_UM)
        assert len(circles) == 4


class TestRobustness:
    def test_core_fused_with_edge_strip_is_found(self):
        occ = np.ones((2, 3), dtype=bool)
        img, centres = synthetic_slide(occ)
        # thin coloured strip (a coverslip edge) along the bottom touching the bottom-left core
        cx, cy = centres[(1, 0)]
        y = cy + int(RADIUS_PX) - 3
        img[y:y + 8, :] = (200, 200, 230)
        from cabana.tma import infer_grid, merge_grid_duplicates, recover_faint_cores
        circles = fit_cores(img, PX_UM, CORE_UM)
        rows, cols, nr, nc = infer_grid(circles, CORE_UM / PX_UM)
        circles, rows, cols = merge_grid_duplicates(circles, rows, cols)
        circles, rows, cols, rec = recover_faint_cores(img, circles, rows, cols, max(nr, 2), max(nc, 3),
                                                       PX_UM, CORE_UM)
        assert len(circles) == 6
        found = np.array([[c[0], c[1]] for c in circles])
        for (x, y) in centres.values():
            assert np.sqrt(((found - [x, y]) ** 2).sum(1)).min() < 0.25 * RADIUS_PX

    def test_faint_core_recovered_at_grid_position(self):
        occ = np.ones((2, 3), dtype=bool)
        occ[0, 1] = False
        img, centres = synthetic_slide(occ)
        # a very pale core: barely below the background, low saturation
        cx, cy = 2 * PITCH_PX, PITCH_PX
        cv2.circle(img, (cx, cy), int(RADIUS_PX), (232, 228, 240), -1)
        pre_circles = fit_cores(img, PX_UM, CORE_UM)
        assert len(pre_circles) == 5                       # too pale for the main pass
        from cabana.tma import infer_grid, recover_faint_cores
        rows, cols, nr, nc = infer_grid(pre_circles, CORE_UM / PX_UM)
        circles, rows, cols, rec = recover_faint_cores(img, pre_circles, rows, cols, nr, nc, PX_UM, CORE_UM)
        assert len(circles) == 6 and sum(rec) == 1
        mx, my, mr, fill = circles[-1]
        assert abs(mx - cx) < 0.25 * RADIUS_PX and abs(my - cy) < 0.25 * RADIUS_PX
        assert fill > 0.5

    def test_truly_empty_cell_is_not_recovered(self):
        occ = np.ones((2, 3), dtype=bool)
        occ[0, 1] = False
        img, _ = synthetic_slide(occ)
        from cabana.tma import infer_grid, recover_faint_cores
        circles = fit_cores(img, PX_UM, CORE_UM)
        rows, cols, nr, nc = infer_grid(circles, CORE_UM / PX_UM)
        circles, rows, cols, rec = recover_faint_cores(img, circles, rows, cols, nr, nc, PX_UM, CORE_UM)
        assert len(circles) == 5 and not any(rec)

    def test_preprocessor_flags_recovered(self, tmp_path):
        occ = np.ones((2, 3), dtype=bool)
        occ[1, 2] = False
        img, _ = synthetic_slide(occ)
        cv2.circle(img, (3 * PITCH_PX, 2 * PITCH_PX), int(RADIUS_PX), (232, 228, 240), -1)
        path = str(tmp_path / "s.png")
        cv2.imwrite(path, img)
        pre = TMAPreprocessor(path, pixel_size_um=PX_UM, core_diameter_um=CORE_UM, fit_pixel_size_um=PX_UM)
        pre.fit()
        assert [c.flag for c in pre.cores].count("recovered") == 1
        pre_off = TMAPreprocessor(path, pixel_size_um=PX_UM, core_diameter_um=CORE_UM,
                                  fit_pixel_size_um=PX_UM, recover_faint=False)
        pre_off.fit()
        assert len(pre_off.cores) == 5


class TestGrid:
    def test_lattice_with_gaps_and_jitter(self):
        occ = np.ones((4, 6), dtype=bool)
        occ[1, 2] = occ[2, 5] = occ[0, 0] = False
        img, centres = synthetic_slide(occ, jitter=40)
        circles = fit_cores(img, PX_UM, CORE_UM)
        rows, cols, n_rows, n_cols = infer_grid(circles, CORE_UM / PX_UM)
        assert (n_rows, n_cols) == (4, 6)
        for c, r, k in zip(circles, rows, cols):
            cx, cy = centres[(r, k)]
            assert abs(c[0] - cx) < 3 and abs(c[1] - cy) < 3

    def test_whole_empty_interior_column_is_kept(self):
        occ = np.ones((3, 5), dtype=bool)
        occ[:, 2] = False
        img, centres = synthetic_slide(occ)
        circles = fit_cores(img, PX_UM, CORE_UM)
        rows, cols, n_rows, n_cols = infer_grid(circles, CORE_UM / PX_UM)
        assert n_cols == 5 and 2 not in set(cols.tolist())

    def test_merge_duplicates(self):
        circles = [(100.0, 100.0, 40.0, 0.9), (130.0, 100.0, 20.0, 0.3), (500.0, 100.0, 40.0, 1.0)]
        rows = np.array([0, 0, 0])
        cols = np.array([0, 0, 1])
        merged, r, c = merge_grid_duplicates(circles, rows, cols)
        assert len(merged) == 2 and list(r) == [0, 0] and list(c) == [0, 1]
        mx, my, mr, fill = merged[0]
        assert 60 < mx < 130 and mr >= 40 and 0.3 < fill < 0.9


class TestOrientation:
    @pytest.mark.parametrize("orientation", ["0", "90", "180", "270", "0+flip", "90+flip",
                                             "180+flip", "270+flip"])
    def test_recovers_applied_transform(self, orientation):
        from cabana.tma import _transform_grid
        occ = occupancy_grid(1)                       # 11 filled rows, 1 empty
        observed = _transform_grid(occ, orientation).copy()
        # drop a few cores as a section would
        observed[0, 0] = False
        observed[observed.shape[0] // 2, observed.shape[1] // 2] = False
        best_o, lookup, score, ties = match_orientation(observed, 1, "auto")
        # Occupancy of array 1 is mirror-symmetric, so the applied transform
        # must be among the tied best candidates and the tie must be reported.
        assert orientation in ties
        assert best_o in ties
        assert lookup.shape[:2] == observed.shape
        # rotation part is unambiguous thanks to the empty last row
        assert all(t.split("+")[0] in {orientation.split("+")[0],
                                       str((int(orientation.split("+")[0]) + 180) % 360)}
                   for t in ties)

    def test_observed_smaller_than_map(self):
        occ = occupancy_grid(1)
        observed = occ[4:12, :].copy()       # lower part, includes the empty row
        o, lookup, score, ties = match_orientation(observed, 1, "0")
        assert o == "0" and ties == ["0"]
        assert tuple(lookup[0, 0]) == (4, 0)
        assert tuple(lookup[-1, -1]) == (11, 7)

    def test_observed_larger_than_map_marks_outside(self):
        occ = occupancy_grid(3)
        observed = np.zeros((14, 8), dtype=bool)
        observed[1:13] = occ
        o, lookup, score, ties = match_orientation(observed, 3, "0")
        assert tuple(lookup[0, 0]) == (-1, -1) and tuple(lookup[1, 0]) == (0, 0)

    def test_forced_orientation(self):
        occ = occupancy_grid(3)
        o, lookup, _, ties = match_orientation(occ, 3, "180")
        assert o == "180" and ties == ["180"] and tuple(lookup[0, 0]) == (11, 7)

    def test_full_array_reports_ties(self):
        occ = occupancy_grid(4)
        o, lookup, score, ties = match_orientation(occ, 4, "auto")
        assert o == "0" and ties == ["0", "180", "0+flip", "180+flip"]


class TestPreprocessor:
    def _write_slide(self, tmp_path, occ, name="TMA1.png"):
        img, centres = synthetic_slide(occ, jitter=10)
        path = str(tmp_path / name)
        cv2.imwrite(path, img)
        return path, centres

    def test_end_to_end_export(self, tmp_path):
        # array 1 rotated by 90 degrees like the scanner does, with 2 cores dropped
        from cabana.tma import _transform_grid
        occ = _transform_grid(occupancy_grid(1), "90").copy()
        occ[0, 1] = False
        occ[3, 5] = False
        path, centres = self._write_slide(tmp_path, occ)
        out = tmp_path / "out"
        pre = TMAPreprocessor(path, array_number=1, slide_name="TMA1", pixel_size_um=PX_UM,
                              core_diameter_um=CORE_UM, margin_um=40, erode_px=2,
                              fit_pixel_size_um=PX_UM)
        progress = []
        ok = pre.run(str(out), progress=lambda d, t, s: progress.append((d, t)))
        assert ok
        assert pre.grid_shape == (occ.shape[0], occ.any(axis=0).sum())   # empty column trimmed
        assert len(pre.cores) == occ.sum()
        assert pre.matched_orientation is not None
        imgs = sorted(os.listdir(out / "Images"))
        masks = sorted(os.listdir(out / "Masks"))
        assert imgs == masks and len(imgs) == occ.sum()
        assert all(n.startswith("TMA1_") and n.endswith("_BF.png") for n in imgs)
        assert len(set(imgs)) == len(imgs)                     # unique names
        assert any("_Liver_" in n or "_Muscle_" in n or "_Brain_" in n for n in imgs)
        assert progress[-1] == (len(imgs), len(imgs))
        # mask is a centred disc with black corners; image is square and same size
        m = cv2.imread(str(out / "Masks" / masks[0]), 0)
        im = cv2.imread(str(out / "Images" / imgs[0]))
        assert m.shape == im.shape[:2] and m.shape[0] == m.shape[1]
        assert m[0, 0] == 0 and m[m.shape[0] // 2, m.shape[1] // 2] == 255
        expected_r = RADIUS_PX - 2
        assert abs(np.sqrt((m > 0).sum() / np.pi) - expected_r) < 2
        with open(out / "cores.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == occ.sum()
        assert all(r["patient_id"] or r["tissue"] for r in rows)
        assert (out / "overlay.png").exists()
        # exported images carry the pixel size so Cabana recovers µm/pixel
        from cabana.io import split2batches
        _, res = split2batches([str(out / "Images" / imgs[0])])
        assert res[0] == pytest.approx(PX_UM, abs=0.01)
        pre.close()

    def test_no_array_names_by_grid_position(self, tmp_path):
        occ = np.ones((2, 3), dtype=bool)
        path, _ = self._write_slide(tmp_path, occ, "slideX.png")
        pre = TMAPreprocessor(path, array_number=None, pixel_size_um=PX_UM,
                              core_diameter_um=CORE_UM, fit_pixel_size_um=PX_UM)
        pre.run(str(tmp_path / "out"))
        names = sorted(os.listdir(tmp_path / "out" / "Images"))
        assert names == sorted(f"slideX_r{r}c{c}_unknown_BF.png" for r in (1, 2) for c in (1, 2, 3))

    def test_debris_outside_map_is_not_exported(self, tmp_path):
        from cabana.tma import _transform_grid
        occ = _transform_grid(occupancy_grid(3), "90")                    # 8 x 12
        wide = np.zeros((occ.shape[0], occ.shape[1] + 2), dtype=bool)
        wide[:, :occ.shape[1]] = occ
        wide[2, -1] = True                                                 # debris column
        path, _ = self._write_slide(tmp_path, wide)
        pre = TMAPreprocessor(path, array_number=3, slide_name="TMA3", pixel_size_um=PX_UM,
                              core_diameter_um=CORE_UM, fit_pixel_size_um=PX_UM)
        pre.fit()
        pre.map_to_array()
        flags = [c.flag for c in pre.cores]
        assert flags.count("outside_map") == 1
        assert len(pre.exportable_cores()) == occ.sum()

    def test_cancel_stops_export(self, tmp_path):
        occ = np.ones((2, 2), dtype=bool)
        path, _ = self._write_slide(tmp_path, occ)
        pre = TMAPreprocessor(path, pixel_size_um=PX_UM, core_diameter_um=CORE_UM,
                              fit_pixel_size_um=PX_UM)
        pre.fit()
        calls = []
        ok = pre.export(str(tmp_path / "out"), progress=lambda *a: calls.append(a),
                        cancel=lambda: len(calls) >= 1)
        assert ok is False and len(calls) == 1

    def test_unknown_pixel_size_raises(self, tmp_path):
        path, _ = self._write_slide(tmp_path, np.ones((1, 1), dtype=bool))
        with pytest.raises(ValueError):
            TMAPreprocessor(path)


class TestFilters:
    def _pre(self, tmp_path, img, array=None, **kw):
        path = str(tmp_path / "s.png")
        cv2.imwrite(path, img)
        pre = TMAPreprocessor(path, array_number=array, slide_name="S", pixel_size_um=PX_UM,
                              core_diameter_um=CORE_UM, fit_pixel_size_um=PX_UM, **kw)
        pre.fit()
        pre.map_to_array()
        return pre

    def test_filters_flag_each_reason(self, tmp_path):
        occ = np.ones((3, 4), dtype=bool)
        img, centres = synthetic_slide(occ)
        # (0,0): grey, unstained but tissue-like disc (dark, low saturation)
        cx, cy = centres[(0, 0)]
        cv2.circle(img, (cx, cy), int(RADIUS_PX), (245, 245, 245), -1)
        cv2.circle(img, (cx, cy), int(RADIUS_PX), (200, 200, 200), -1)
        # (1,1): small core
        cx, cy = centres[(1, 1)]
        cv2.circle(img, (cx, cy), int(RADIUS_PX), (245, 245, 245), -1)
        cv2.circle(img, (cx, cy), int(0.45 * RADIUS_PX), (190, 150, 230), -1)
        # (2,3): shifted off its grid position by 0.4 pitch
        cx, cy = centres[(2, 3)]
        cv2.circle(img, (cx, cy), int(RADIUS_PX), (245, 245, 245), -1)
        cv2.circle(img, (cx + int(0.4 * PITCH_PX), cy), int(RADIUS_PX * 0.9), (190, 150, 230), -1)
        pre = self._pre(tmp_path, img)
        reasons = {(c.row, c.col): c.reason for c in pre.cores}
        assert reasons[(0, 0)] == "stain"
        assert reasons[(1, 1)] == "diameter"
        assert reasons[(2, 3)] == "off_grid"
        assert sum(1 for r in reasons.values() if r) == 3
        assert len(pre.exportable_cores()) == 9
        assert pre.exclusion_summary() == {"stain": 1, "diameter": 1, "off_grid": 1}

    def test_relaxing_filters_restores_cores_without_refit(self, tmp_path):
        occ = np.ones((2, 3), dtype=bool)
        img, centres = synthetic_slide(occ)
        cx, cy = centres[(0, 0)]
        cv2.circle(img, (cx, cy), int(RADIUS_PX), (200, 200, 200), -1)   # unstained
        pre = self._pre(tmp_path, img)
        assert pre.cores[0].reason == "stain"
        pre.min_stain_frac = 0.0
        pre.apply_filters()
        assert not any(c.excluded for c in pre.cores)

    def test_controls_exempt_from_stain(self, tmp_path):
        from cabana.tma import _transform_grid
        occ = occupancy_grid(1)
        img, centres = synthetic_slide(occ)
        # A1 is Liver (control), A2 a patient: make both unstained grey discs
        for pos in [(0, 0), (0, 1)]:
            cx, cy = centres[pos]
            cv2.circle(img, (cx, cy), int(RADIUS_PX), (200, 200, 200), -1)
        pre = self._pre(tmp_path, img, array=1, orientation="0")
        by_pos = {(c.map_row, c.map_col): c for c in pre.cores}
        assert not by_pos[(0, 0)].excluded                    # Liver control kept
        assert by_pos[(0, 1)].reason == "stain"               # patient core excluded

    def test_filter_values_never_change_patient_ids(self, tmp_path):
        from cabana.tma import _transform_grid
        occ = _transform_grid(occupancy_grid(1), "90").copy()
        img, _ = synthetic_slide(occ, jitter=10)
        pre = self._pre(tmp_path, img, array=1)
        ids = {c.index: (c.map_row, c.map_col) for c in pre.cores}
        o = pre.matched_orientation
        for kw in (dict(max_diameter_frac=0.8), dict(min_diameter_frac=1.2),
                   dict(max_grid_offset=0.01), dict(min_stain_frac=1.01), dict(min_tissue_fill=1.0)):
            for k, v in kw.items():
                setattr(pre, k, v)
            pre.map_to_array()
            assert pre.matched_orientation == o
            assert {c.index: (c.map_row, c.map_col) for c in pre.cores} == ids
            assert any(c.excluded for c in pre.cores)
            pre.max_diameter_frac, pre.min_diameter_frac, pre.max_grid_offset = 1.4, 0.6, 0.35
            pre.min_stain_frac, pre.min_tissue_fill = 0.02, 0.2

    def test_manifest_records_exclusions(self, tmp_path):
        occ = np.ones((2, 2), dtype=bool)
        img, centres = synthetic_slide(occ)
        cx, cy = centres[(1, 1)]
        cv2.circle(img, (cx, cy), int(RADIUS_PX), (200, 200, 200), -1)
        pre = self._pre(tmp_path, img)
        pre.export(str(tmp_path / "out"))
        with open(tmp_path / "out" / "cores.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 4
        ex = [r for r in rows if r["excluded"] == "1"]
        assert len(ex) == 1 and ex[0]["reason"] == "stain" and ex[0]["flag"] == "excluded"
        assert float(ex[0]["stain_frac"]) < 0.02 and float(ex[0]["diameter_um"]) > 0
        assert len(os.listdir(tmp_path / "out" / "Images")) == 3
