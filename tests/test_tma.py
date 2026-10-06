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
from cabana.tma_maps import occupancy_grid, load_array_map

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


class TestTissueMask:
    """Scanner artefacts must not be mistaken for tissue or bias the background."""

    @staticmethod
    def _scanned_block(h, w, bg=243, seed=0):
        rng = np.random.default_rng(seed)
        block = np.full((h, w, 3), bg, dtype=np.uint8)
        return np.clip(block.astype(int) + rng.integers(-2, 3, (h, w, 1)), 0, 255).astype(np.uint8)

    def test_core_inside_partly_scanned_columns(self):
        # Columns 0..600 are mostly unscanned (white 255) with a small scanned
        # block holding one core; an off-white noisy stripe runs through the
        # white area, as VS200 scans show.
        from cabana.tma import tissue_mask
        img = np.full((2000, 1400, 3), 255, dtype=np.uint8)
        img[:, 700:] = self._scanned_block(2000, 700)
        img[1500:1900, 100:500] = self._scanned_block(400, 400, seed=1)
        cv2.circle(img, (300, 1700), int(RADIUS_PX), (190, 150, 230), -1)
        rng = np.random.default_rng(2)
        img[:1500, 280:286] = rng.integers(253, 256, (1500, 6, 1))
        mask = tissue_mask(img)
        assert mask[1500:1900, 100:500].mean() > 0.3          # the core is there
        bg_block = mask[1500:1900, 100:500].copy()
        yy, xx = np.ogrid[1500:1900, 100:500]
        outside = (xx - 300) ** 2 + (yy - 1700) ** 2 > (RADIUS_PX + 6) ** 2
        assert bg_block[outside].mean() < 0.01                  # scanned background is clean
        assert mask[:1500, 270:300].mean() < 0.01               # the stripe is not tissue
        assert mask[:, 700:].mean() < 0.01
        circles = fit_cores(img, PX_UM, CORE_UM)
        assert len(circles) == 1 and abs(circles[0][2] - RADIUS_PX) < 0.1 * RADIUS_PX

    def test_flat_padding_strip_is_not_tissue(self):
        # A bright slide (background 254) with the constant grey padding the
        # scanner writes below the acquired frame.
        from cabana.tma import tissue_mask
        img = self._scanned_block(1200, 1600, bg=253)
        cv2.circle(img, (800, 600), int(RADIUS_PX), (190, 150, 230), -1)
        img[1184:, :] = 238
        mask = tissue_mask(img)
        assert mask[1184:, :].sum() == 0
        assert len(fit_cores(img, PX_UM, CORE_UM)) == 1

    def test_flat_synthetic_background_is_still_background(self):
        # Flat backgrounds at the slide's own level are legitimate (synthetic
        # slides, very clean scans) and must keep working as before.
        from cabana.tma import tissue_mask
        img, _ = synthetic_slide(np.ones((1, 2), dtype=bool))
        mask = tissue_mask(img)
        assert 0.1 < mask.mean() < 0.4
        cv2.circle(img, (PITCH_PX, PITCH_PX), int(RADIUS_PX), (232, 228, 240), -1)   # pale core
        pale = tissue_mask(img, sat_thresh=7.5, val_ratio=0.93)
        assert pale[PITCH_PX, PITCH_PX] == 1


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
        assert set(c.flag for c in pre.cores) <= {"ok", "recovered"}
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
        # identical synthetic cores cannot break the occupancy tie, so fix the orientation
        pre = TMAPreprocessor(path, array_number=1, slide_name="TMA1", pixel_size_um=PX_UM,
                              core_diameter_um=CORE_UM, margin_um=40, erode_px=2,
                              fit_pixel_size_um=PX_UM, orientation="90")
        progress = []
        ok = pre.run(str(out), progress=lambda d, t, s: progress.append((d, t)))
        assert ok
        assert pre.grid_shape == (occ.shape[0], occ.any(axis=0).sum())   # empty column trimmed
        assert len(pre.cores) == occ.sum()
        assert pre.matched_orientation is not None
        assert sorted(p.name for p in out.iterdir()) == ["BF", "cores.csv", "cores_edits.json", "overlay.png"]
        assert sorted(p.name for p in (out / "BF").iterdir()) == ["Controls", "Patients"]
        ctrl = sorted(os.listdir(out / "BF" / "Controls" / "Images"))
        pats = sorted(os.listdir(out / "BF" / "Patients" / "Images"))
        assert ctrl == sorted(os.listdir(out / "BF" / "Controls" / "Masks"))
        assert pats == sorted(os.listdir(out / "BF" / "Patients" / "Masks"))
        assert ctrl and all(n.split(".vsi")[0] in ("Liver", "Muscle", "Brain", "Placenta",
                                                    "Salivary-gland", "Lung") for n in ctrl)
        assert all(n.split(".vsi")[0].isdigit() for n in pats)          # patient IDs only
        imgs = sorted(pats + ctrl)
        masks = imgs
        assert len(imgs) == occ.sum()
        assert all(" - TMA1_BF_" in n and n.endswith(".png") for n in imgs)
        assert len(set(imgs)) == len(imgs)                     # unique names
        assert any(n.startswith(("Liver.vsi", "Muscle.vsi", "Brain.vsi")) for n in imgs)
        assert progress[-1] == (len(imgs), len(imgs))
        # mask is a centred disc with black corners; image is square and same size
        m = cv2.imread(str(out / "BF" / "Patients" / "Masks" / pats[0]), 0)
        im = cv2.imread(str(out / "BF" / "Patients" / "Images" / pats[0]))
        assert m.shape == im.shape[:2] and m.shape[0] == m.shape[1]
        assert m[0, 0] == 0 and m[m.shape[0] // 2, m.shape[1] // 2] == 255
        expected_r = RADIUS_PX - 2
        assert abs(np.sqrt((m > 0).sum() / np.pi) - expected_r) < 2
        with open(out / "cores.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        present = [r for r in rows if r["flag"] != "missing"]
        missing = [r for r in rows if r["flag"] == "missing"]
        assert len(present) == occ.sum()
        # the two dropped cores appear as "missing" rows at their scan cell, so the
        # manifest covers every position of the printed map
        assert len(missing) == 2
        assert {(int(r["scan_row"]) - 1, int(r["scan_col"]) - 1) for r in missing} == {(0, 1), (3, 5)}
        assert all(r["patient_id"] or r["tissue"] for r in rows)
        assert all(r["excluded"] == "1" and r["reason"] == "missing" and r["cx"] for r in missing)
        assert {r["group"] for r in rows} == {"Patients", "Controls"}
        assert (out / "overlay.png").exists()
        assert (out / "cores_edits.json").exists()
        # exported images carry the pixel size so Cabana recovers µm/pixel
        from cabana.io import split2batches
        _, res = split2batches([str(out / "BF" / "Patients" / "Images" / pats[0])])
        assert res[0] == pytest.approx(PX_UM, abs=0.01)
        pre.close()

    def test_no_array_names_by_grid_position(self, tmp_path):
        occ = np.ones((2, 3), dtype=bool)
        path, _ = self._write_slide(tmp_path, occ, "slideX.png")
        pre = TMAPreprocessor(path, array_number=None, pixel_size_um=PX_UM,
                              core_diameter_um=CORE_UM, fit_pixel_size_um=PX_UM)
        pre.run(str(tmp_path / "out"))
        names = sorted(os.listdir(tmp_path / "out" / "BF" / "Unmapped" / "Images"))
        assert names == sorted(f"slideX-r{r}c{c}.vsi - slideX_BF_r{r}c{c}Annotation (Unmapped)_1.png"
                               for r in (1, 2) for c in (1, 2, 3))

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
                   dict(max_grid_offset=0.01), dict(min_stain_frac=1.01)):
            for k, v in kw.items():
                setattr(pre, k, v)
            pre.map_to_array()
            assert pre.matched_orientation == o
            assert {c.index: (c.map_row, c.map_col) for c in pre.cores} == ids
            assert any(c.excluded for c in pre.cores)
            pre.max_diameter_frac, pre.min_diameter_frac, pre.max_grid_offset = 1.2, 0.8, 0.35
            pre.min_stain_frac = 0.02

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
        assert len(os.listdir(tmp_path / "out" / "BF" / "Unmapped" / "Images")) == 3


class TestDebrisRobustness:
    def test_debris_beside_core_does_not_inflate_circle(self):
        occ = np.ones((2, 3), dtype=bool)
        img, centres = synthetic_slide(occ)
        cx, cy = centres[(0, 1)]
        # sizeable debris blob just outside the core, bridged by the gap closing
        cv2.circle(img, (cx + int(1.25 * RADIUS_PX), cy - int(0.6 * RADIUS_PX)), int(0.3 * RADIUS_PX),
                   (190, 150, 230), -1)
        circles = fit_cores(img, PX_UM, CORE_UM)
        c = min(circles, key=lambda q: np.hypot(q[0] - cx, q[1] - cy))
        assert c[2] < 1.12 * RADIUS_PX
        assert np.hypot(c[0] - cx, c[1] - cy) < 0.1 * RADIUS_PX

    def test_fused_debris_is_split_off(self):
        occ = np.ones((2, 3), dtype=bool)
        img, centres = synthetic_slide(occ)
        cx, cy = centres[(1, 2)]
        # debris attached to the core by a thin bridge
        cv2.line(img, (cx + int(RADIUS_PX) - 2, cy), (cx + int(1.4 * RADIUS_PX), cy), (190, 150, 230), 3)
        cv2.circle(img, (cx + int(1.6 * RADIUS_PX), cy), int(0.3 * RADIUS_PX), (190, 150, 230), -1)
        circles = fit_cores(img, PX_UM, CORE_UM)
        c = min(circles, key=lambda q: np.hypot(q[0] - cx, q[1] - cy))
        assert c[2] < 1.12 * RADIUS_PX

    def test_fragmented_core_keeps_all_fragments(self):
        occ = np.ones((2, 3), dtype=bool)
        img, centres = synthetic_slide(occ)
        cx, cy = centres[(0, 0)]
        # split one core into two halves separated by a gap
        cv2.rectangle(img, (cx - 6, cy - int(RADIUS_PX) - 2), (cx + 6, cy + int(RADIUS_PX) + 2),
                      (245, 245, 245), -1)
        circles = fit_cores(img, PX_UM, CORE_UM)
        c = min(circles, key=lambda q: np.hypot(q[0] - cx, q[1] - cy))
        assert abs(c[2] - RADIUS_PX) < 0.08 * RADIUS_PX and np.hypot(c[0] - cx, c[1] - cy) < 5

    def test_debris_circle_in_same_cell_is_not_merged(self):
        # a small core plus a separate debris speck in the same grid cell
        circles = [(400.0, 400.0, 100.0, 0.9), (560.0, 470.0, 26.0, 0.8)]
        merged, r, c = merge_grid_duplicates(circles, np.array([0, 0]), np.array([0, 0]), max_radius=137.5)
        assert len(merged) == 1 and merged[0][:3] == (400.0, 400.0, 100.0)

    def test_fragments_in_same_cell_are_merged_within_bound(self):
        circles = [(400.0, 400.0, 60.0, 0.9), (470.0, 400.0, 50.0, 0.8)]
        merged, r, c = merge_grid_duplicates(circles, np.array([0, 0]), np.array([0, 0]), max_radius=137.5)
        assert len(merged) == 1 and merged[0][2] > 60 and merged[0][2] <= 137.5


class TestChannelFolders:
    def test_each_channel_gets_its_own_images_and_masks(self, tmp_path):
        """Two-channel export: <out>/<ch>/Images and <out>/<ch>/Masks, matching stems."""
        occ = np.ones((2, 2), dtype=bool)
        img, _ = synthetic_slide(occ)
        path = str(tmp_path / "s.png")
        cv2.imwrite(path, img)
        pre = TMAPreprocessor(path, slide_name="S", pixel_size_um=PX_UM, core_diameter_um=CORE_UM,
                              fit_pixel_size_um=PX_UM)
        pre.reader.channels = ("BF", "POL")          # second channel served by the same raster
        orig = pre.reader.read_region
        pre.reader.read_region = lambda x, y, w, h, channel=0, level=0: orig(x, y, w, h, 0, level)
        pre.fit()
        pre.map_to_array()
        out = tmp_path / "out"
        assert pre.export(str(out))
        for ch in ("BF", "POL"):
            imgs = sorted(os.listdir(out / ch / "Unmapped" / "Images"))
            masks = sorted(os.listdir(out / ch / "Unmapped" / "Masks"))
            assert len(imgs) == 4 and imgs == masks
            assert all(f"_{ch}_" in n for n in imgs)
        assert TMAPreprocessor.channel_dirs(str(out), "POL", "Controls") == (
            str(out / "POL" / "Controls" / "Images"), str(out / "POL" / "Controls" / "Masks"))


class TestTieBreaking:
    """Orientation ties on fully populated arrays are broken by appearance."""

    def _slide_with_patient_signatures(self, array, orientation, brain_pale=True, seed=0):
        """Synthetic slide of ``array`` laid out under ``orientation`` where every
        patient's three cores share a distinctive colour and the Brain control
        (if any) is nearly unstained."""
        from cabana.tma import _transform_grid
        amap = load_array_map(array)
        idx = np.stack(np.meshgrid(np.arange(12), np.arange(8), indexing="ij"), -1)
        placed = _transform_grid(idx, orientation)
        occ = _transform_grid(occupancy_grid(array), orientation)
        img, centres = synthetic_slide(occ)
        rng = np.random.default_rng(seed)
        colours = {}
        for (r, c), (cx, cy) in centres.items():
            info = amap[tuple(int(v) for v in placed[r, c])]
            if info.is_control:
                colour = (236, 232, 240) if (info.tissue == "Brain" and brain_pale) else (200, 170, 235)
            else:
                if info.patient_id not in colours:
                    colours[info.patient_id] = tuple(int(v) for v in rng.integers(60, 230, 3))
                colour = colours[info.patient_id]
            cv2.circle(img, (cx, cy), int(RADIUS_PX), colour, -1)
        return img

    def _pre(self, tmp_path, img, array):
        path = str(tmp_path / "s.png")
        cv2.imwrite(path, img)
        pre = TMAPreprocessor(path, array_number=array, slide_name="S", pixel_size_um=PX_UM,
                              core_diameter_um=CORE_UM, fit_pixel_size_um=PX_UM, min_stain_frac=0.0)
        pre.fit()
        pre.map_to_array()
        return pre

    @pytest.mark.parametrize("orientation", ["90", "270", "90+flip"])
    def test_brain_breaks_tie(self, tmp_path, orientation):
        pre = self._pre(tmp_path, self._slide_with_patient_signatures(4, orientation), 4)
        assert len(pre.orientation_ties) > 1
        assert pre.matched_orientation == orientation
        assert pre.orientation_method == "brain" and pre.orientation_resolved

    @pytest.mark.parametrize("orientation", ["90", "270"])
    def test_replicates_break_tie_without_brain(self, tmp_path, orientation):
        pre = self._pre(tmp_path, self._slide_with_patient_signatures(7, orientation), 7)   # no Brain on array 7
        assert len(pre.orientation_ties) > 1
        assert pre.matched_orientation == orientation
        assert pre.orientation_method == "replicates"
        assert pre.orientation_margin >= 0.10

    def test_unresolved_blocks_export(self, tmp_path):
        # all cores identical: neither Brain nor replicates can separate the ties
        occ = occupancy_grid(7)
        img, _ = synthetic_slide(occ)
        pre = self._pre(tmp_path, img, 7)
        assert not pre.orientation_resolved and pre.orientation_method == "unresolved"
        with pytest.raises(RuntimeError):
            pre.export(str(tmp_path / "out"))
        pre.orientation = "90"
        pre.map_to_array()
        assert pre.orientation_method == "manual" and pre.orientation_resolved
        assert pre.export(str(tmp_path / "out"))


def rotate_slide(img, degrees):
    """Rotate a synthetic slide about its centre, padding with the slide background."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(245, 245, 245))


class TestDeskew:
    """A tilted array must snap to the same grid as an upright one."""

    @pytest.mark.parametrize("degrees", [-4.0, 3.0])
    def test_rotated_grid_shape(self, degrees):
        from cabana.tma import estimate_rotation, well_formed
        occ = np.ones((5, 8), dtype=bool)
        occ[1, 3] = occ[4, 0] = False
        img, _ = synthetic_slide(occ)
        img = rotate_slide(img, degrees)
        circles = fit_cores(img, PX_UM, CORE_UM)
        pts = np.array([[c[0], c[1]] for c in circles])
        angle = np.degrees(estimate_rotation(pts[well_formed(circles)], CORE_UM / PX_UM))
        # image rotation is counter-clockwise for positive degrees in image coordinates (y down)
        assert abs(angle + degrees) < 0.5
        rows, cols, n_rows, n_cols = infer_grid(circles, CORE_UM / PX_UM)
        assert (n_rows, n_cols) == (5, 8)
        assert len({(r, c) for r, c in zip(rows, cols)}) == len(circles)   # one circle per cell

    def test_off_lattice_debris_does_not_widen_grid(self):
        occ = np.ones((4, 6), dtype=bool)
        img, centres = synthetic_slide(occ)
        # a small blob half a pitch above the top row, between two columns
        cx, cy = centres[(0, 2)]
        cv2.circle(img, (cx + int(0.55 * PITCH_PX), cy - int(0.55 * PITCH_PX)), 30, (190, 150, 230), -1)
        circles = fit_cores(img, PX_UM, CORE_UM)
        assert len(circles) == 25
        rows, cols, n_rows, n_cols = infer_grid(circles, CORE_UM / PX_UM)
        assert (n_rows, n_cols) == (4, 6)
        small = int(np.argmin([c[2] for c in circles]))
        assert rows[small] < 0 or rows[small] >= n_rows or cols[small] < 0 or cols[small] >= n_cols
        assert all(0 <= r < 4 and 0 <= c < 6 for i, (r, c) in enumerate(zip(rows, cols)) if i != small)

    def test_pitch_ignores_malformed_circles(self):
        from cabana.tma import well_formed
        circles = [(0, 0, 50, 0.9), (200, 0, 50, 0.9), (400, 0, 48, 0.8), (600, 0, 52, 0.95),
                   (700, 0, 15, 0.9),   # debris: wrong radius
                   (100, 300, 50, 0.1)]  # hollow ring: low fill
        assert set(well_formed(circles).tolist()) == {0, 1, 2, 3}


class TestEditing:
    """Hand edits re-snap to the lattice, re-map and survive a save/load."""

    def _pre(self, tmp_path, name="e.png"):
        from cabana.tma import _transform_grid
        occ = _transform_grid(occupancy_grid(1), "90").copy()
        occ[0, 1] = False
        occ[3, 5] = False
        img, centres = synthetic_slide(occ, jitter=5)
        path = str(tmp_path / name)
        cv2.imwrite(path, img)
        pre = TMAPreprocessor(path, array_number=1, slide_name="E", pixel_size_um=PX_UM,
                              core_diameter_um=CORE_UM, fit_pixel_size_um=PX_UM, orientation="90")
        pre.fit()
        pre.map_to_array()
        return pre, centres, occ

    def test_map_and_scan_cells_are_inverse(self, tmp_path):
        pre, _, _ = self._pre(tmp_path)
        for c in pre.cores:
            assert pre.map_cell(c.row, c.col) == (c.map_row, c.map_col)
            assert pre.scan_cell(c.map_row, c.map_col) == (c.row, c.col)
        assert pre.map_cell(-5, -5) == (-1, -1)
        assert pre.map_cell(50, 50) == (-1, -1)
        pre.close()

    def test_missing_positions_and_links(self, tmp_path):
        pre, _, occ = self._pre(tmp_path)
        miss = pre.missing_positions()
        assert {(r, c) for _, r, c, _ in miss} == {(0, 1), (3, 5)}
        assert all(centre is not None for _, _, _, centre in miss)
        n_rows, n_cols = occ.shape
        expected = sum(1 for r in range(n_rows) for c in range(n_cols)
                       if occ[r, c] and ((c + 1 < n_cols and occ[r, c + 1])))
        expected += sum(1 for r in range(n_rows) for c in range(n_cols)
                        if occ[r, c] and ((r + 1 < n_rows and occ[r + 1, c])))
        assert len(pre.grid_links()) == expected
        pre.close()

    def test_add_move_remove(self, tmp_path):
        pre, centres, _ = self._pre(tmp_path)
        n0 = len(pre.cores)
        info_missing = {(r, c): info for info, r, c, _ in pre.missing_positions()}
        # add a core at the lattice-predicted centre of an empty cell
        x, y = pre.predict_centre(0, 1)
        core = pre.add_core(x, y)
        assert core.manual and (core.row, core.col) == (0, 1)
        assert pre.core_info(core).position == info_missing[(0, 1)].position
        assert {(r, c) for _, r, c, _ in pre.missing_positions()} == {(3, 5)}
        assert len(pre.cores) == n0 + 1 and [c.index for c in pre.cores] == list(range(1, n0 + 2))
        assert core.diameter_um == pytest.approx(2 * core.radius * PX_UM)
        # move it to the other empty cell: map position follows the lattice cell
        x2, y2 = pre.predict_centre(3, 5)
        pre.move_core(core, x2 + 3, y2 - 2)
        assert (core.row, core.col) == (3, 5)
        assert pre.core_info(core).position == info_missing[(3, 5)].position
        assert core.grid_offset < 0.05
        assert {(r, c) for _, r, c, _ in pre.missing_positions()} == {(0, 1)}
        # move it far off the array: outside the map, still listed
        pre.move_core(core, x2 + 6 * PITCH_PX * 1.0, y2)
        assert core.outside_map
        pre.remove_core(core)
        assert len(pre.cores) == n0
        assert {(r, c) for _, r, c, _ in pre.missing_positions()} == {(0, 1), (3, 5)}
        pre.close()

    def test_resize_and_override(self, tmp_path):
        pre, _, _ = self._pre(tmp_path)
        core = next(c for c in pre.cores if not c.excluded and pre.core_info(c).patient_id)
        pre.resize_core(core, core.radius * 0.5)
        assert core.excluded and core.reason == "diameter" and core.manual
        pre.set_override(core, "include")
        assert not core.excluded and core.reason == ""
        pre.map_to_array()                       # re-mapping keeps the override
        assert not core.excluded
        pre.apply_filters()
        assert not core.excluded
        pre.set_override(core, "")
        assert core.excluded and core.reason == "diameter"
        other = next(c for c in pre.cores if not c.excluded)
        pre.set_override(other, "exclude")
        assert other.excluded and other.reason == "manual" and other not in pre.exportable_cores()
        assert pre.manual_count() == 2
        pre.close()

    def test_snapshot_restore(self, tmp_path):
        pre, _, _ = self._pre(tmp_path)
        before = pre.snapshot()
        core = pre.cores[0]
        pre.move_core(core, core.cx + PITCH_PX, core.cy)
        pre.remove_core(pre.cores[-1])
        pre.restore(before)
        assert len(pre.cores) == len(before)
        assert not any(c.manual for c in pre.cores)
        assert [(c.row, c.col, c.map_row, c.map_col) for c in pre.cores] == \
               [(c.row, c.col, c.map_row, c.map_col) for c in before]
        pre.close()

    def test_save_and_load_edits(self, tmp_path):
        pre, _, _ = self._pre(tmp_path)
        x, y = pre.predict_centre(0, 1)
        added = pre.add_core(x, y)
        pre.set_override(added, "include")       # the synthetic cell holds no tissue
        added_pos = pre.core_info(added).position
        victim = next(c for c in pre.cores if not c.excluded and c is not added)
        pre.set_override(victim, "exclude")
        victim_pos = pre.core_info(victim).position
        path = str(tmp_path / "edits.json")
        pre.save_edits(path)
        n = len(pre.cores)
        pre.close()

        pre2, _, _ = self._pre(tmp_path, "e2.png")
        assert len(pre2.cores) == n - 1
        assert pre2.load_edits(path) == n
        assert len(pre2.cores) == n and pre2.manual_count() == 2
        assert pre2.matched_orientation == "90"
        by_pos = {pre2.core_info(c).position: c for c in pre2.cores}
        assert by_pos[added_pos].manual and by_pos[added_pos].override == "include"
        assert not by_pos[added_pos].excluded
        assert by_pos[victim_pos].override == "exclude" and by_pos[victim_pos].excluded
        assert {(r, c) for _, r, c, _ in pre2.missing_positions()} == {(3, 5)}
        pre2.close()

    def test_empty_positions_without_map_and_labels(self, tmp_path):
        occ = np.ones((2, 3), dtype=bool)
        occ[1, 1] = False
        img, _ = synthetic_slide(occ)
        path = str(tmp_path / "nomap.png")
        cv2.imwrite(path, img)
        pre = TMAPreprocessor(path, array_number=None, pixel_size_um=PX_UM,
                              core_diameter_um=CORE_UM, fit_pixel_size_um=PX_UM)
        pre.fit()
        pre.map_to_array()
        empties = pre.empty_positions()
        assert [(r, c) for _, r, c, _ in empties] == [(1, 1)] and empties[0][0] is None
        assert empties[0][3] is not None
        c = next(c for c in pre.cores if (c.row, c.col) == (0, 2))
        assert pre.core_labels(c) == [f"r1c3 #{c.index}"]
        assert "r1c3" in pre.core_tooltip(c) and "exported" in pre.core_tooltip(c)
        a = next(c for c in pre.cores if (c.row, c.col) == (0, 0))
        b = next(c for c in pre.cores if (c.row, c.col) == (0, 1))
        (x1, y1), (x2, y2) = pre.link_segment(a, b)
        assert abs(x1 - (a.cx + a.radius)) < 1e-6 and abs(x2 - (b.cx - b.radius)) < 1e-6
        # the lattice lines also reach the empty cell: every adjacent pair of a 2x3 grid
        assert len(pre.grid_links()) == 4 and len(pre.grid_segments()) == 7
        pre.close()

    def test_duplicate_cell_is_excluded(self, tmp_path):
        pre, _, _ = self._pre(tmp_path)
        a = next(c for c in pre.cores if (c.row, c.col) == (1, 1))
        b = next(c for c in pre.cores if (c.row, c.col) == (1, 2))
        pre.move_core(a, b.cx + 3, b.cy)
        assert (a.row, a.col) == (1, 2) and a.excluded and a.reason == "duplicate" and not b.excluded
        stems = [pre.core_stem(c) for c in pre.exportable_cores()]
        assert len(stems) == len(set(stems))
        pre.set_override(a, "include")          # an override cannot create a duplicate export
        assert a.excluded and a.reason == "duplicate"
        pre.close()

    def test_manual_core_fill_is_measured(self, tmp_path):
        pre, _, _ = self._pre(tmp_path)
        x, y = pre.predict_centre(0, 1)         # an empty cell on the synthetic slide
        empty = pre.add_core(x, y)
        assert empty.fill < 0.05
        full = pre.cores[0]
        r0 = full.radius
        pre.resize_core(full, r0 * 0.5)         # still entirely on tissue
        assert full.fill > 0.9
        pre.close()

    def test_load_edits_restores_saved_orientation(self, tmp_path):
        pre, _, _ = self._pre(tmp_path)
        path = str(tmp_path / "edits.json")
        pre.save_edits(path)
        pre.close()
        img_path = str(tmp_path / "e.png")
        pre2 = TMAPreprocessor(img_path, array_number=1, slide_name="E", pixel_size_um=PX_UM,
                               core_diameter_um=CORE_UM, fit_pixel_size_um=PX_UM)   # orientation auto
        pre2.fit()
        pre2.map_to_array()
        pre2.load_edits(path)
        assert pre2.orientation == "90" and pre2.matched_orientation == "90"
        assert pre2.orientation_method == "manual"
        with pytest.raises(ValueError):
            pre3 = TMAPreprocessor(img_path, array_number=2, slide_name="E", pixel_size_um=PX_UM,
                                   core_diameter_um=CORE_UM, fit_pixel_size_um=PX_UM)
            pre3.fit()
            pre3.load_edits(path)
        pre2.close()

    def test_load_edits_rejects_bad_records(self, tmp_path):
        import json
        pre, _, _ = self._pre(tmp_path)
        bad = {"slide": "E", "array": 1, "orientation": "90",
               "cores": [{"cx": 10.0, "cy": 10.0, "radius": 5.0, "override": "maybe"}]}
        path = str(tmp_path / "bad.json")
        json.dump(bad, open(path, "w"))
        with pytest.raises(ValueError):
            pre.load_edits(path)
        pre.close()

    def test_fit_resets_previous_match(self, tmp_path):
        pre, _, _ = self._pre(tmp_path)
        assert pre.matched_orientation == "90"
        pre.fit()
        assert pre.matched_orientation is None and pre.orientation_method is None and not pre.missing_positions()
        pre.map_to_array()
        assert pre.matched_orientation == "90"
        pre.close()

    def test_include_override_exports_outside_map_core(self, tmp_path):
        pre, _, _ = self._pre(tmp_path)
        core = pre.add_core(pre.cores[0].cx + 20 * PITCH_PX, pre.cores[0].cy)
        assert core.outside_map and core not in pre.exportable_cores()
        pre.set_override(core, "include")
        assert core in pre.exportable_cores() and pre.core_group(core) == "Unmapped"
        pre.close()

    def test_export_rejects_unknown_channel(self, tmp_path):
        pre, _, _ = self._pre(tmp_path)
        with pytest.raises(ValueError):
            pre.export(str(tmp_path / "out"), channels=["../x"])
        pre.close()

    def test_manifest_marks_manual(self, tmp_path):
        pre, _, _ = self._pre(tmp_path)
        x, y = pre.predict_centre(0, 1)
        pre.add_core(x, y)
        out = tmp_path / "out"
        out.mkdir()
        pre.write_manifest(str(out / "cores.csv"))
        with open(out / "cores.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        assert sum(int(r["manual"]) for r in rows) == 1
        assert sum(r["flag"] == "missing" for r in rows) == 1
        pre.close()
