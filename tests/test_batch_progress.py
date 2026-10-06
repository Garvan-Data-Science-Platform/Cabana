"""Tests for the batch-processing progress callback (Step 1).

These tests target the small, isolated invariants of the new callback wiring:

- ``BatchProcessor.update_progress`` coerces float values to int, clamps to
  [0, 100], drops duplicates, and is monotone non-decreasing.
- ``BatchCabana._tick`` reports a fraction in [0, 1] to its callback and
  clamps overshoots.
- ``BatchCabana._estimate_progress_total`` reflects which Configs stages run.

The full pipeline is not invoked here — those paths are exercised by the
existing test suite. The progress wiring is verified by patching
``BatchCabana`` so we can drive the callback directly without running CNN
segmentation or any other heavy work.
"""

import os
import shutil
import tempfile
import types

import pytest

from cabana.batch import BatchCabana, BatchProcessor


# ----------------------------- update_progress ----------------------------- #


def _make_processor(tmpdir):
    """Construct a BatchProcessor that does not execute __init__ validation.

    BatchProcessor.__init__ does heavy validation (checking the param file and
    input folder paths). For unit tests we only care about update_progress, so
    we bypass __init__ and set the few attributes the method touches.
    """
    bp = BatchProcessor.__new__(BatchProcessor)
    bp.progress_callback = None
    bp._last_progress = 0
    return bp


def test_update_progress_coerces_floats_to_int():
    bp = _make_processor(None)
    seen = []
    bp.progress_callback = seen.append
    bp.update_progress(12.7)
    bp.update_progress(13.2)
    assert seen == [13, 13] or seen == [13]  # 13.2 may be deduped against 13


def test_update_progress_is_monotone_nondecreasing():
    bp = _make_processor(None)
    seen = []
    bp.progress_callback = seen.append
    bp.update_progress(40)
    bp.update_progress(20)   # dropped
    bp.update_progress(40)   # dropped (equal)
    bp.update_progress(50)
    bp.update_progress(49.4) # rounds to 49 -> dropped
    bp.update_progress(75.6) # rounds to 76
    assert seen == [40, 50, 76]


def test_update_progress_clamps_to_0_100():
    bp = _make_processor(None)
    seen = []
    bp.progress_callback = seen.append
    bp.update_progress(-5)
    bp.update_progress(150)
    assert seen == [100]  # negative is dropped (last=0); 150 clamps to 100


def test_update_progress_handles_invalid_silently():
    bp = _make_processor(None)
    seen = []
    bp.progress_callback = seen.append
    bp.update_progress(None)
    bp.update_progress("nope")
    assert seen == []


def test_update_progress_no_callback_is_noop():
    bp = _make_processor(None)
    # No callback set; must not raise
    bp.update_progress(42)
    assert bp._last_progress == 42  # state still updates so subsequent calls obey monotonicity


# ------------------------------- BatchCabana ------------------------------- #


def _make_batch_cabana(tmpdir):
    """Construct a BatchCabana without running the side-effecting __init__.

    __init__ creates several output sub-directories. For unit tests of _tick
    and _estimate_progress_total we don't need those; we instantiate the
    object minimally via __new__.
    """
    bc = BatchCabana.__new__(BatchCabana)
    bc.progress_callback = None
    bc._progress_done = 0
    bc._progress_total = 1
    bc.args = None
    return bc


def test_tick_reports_fraction_to_callback():
    bc = _make_batch_cabana(None)
    seen = []
    bc.progress_callback = seen.append
    bc._progress_total = 4
    bc._tick()
    bc._tick()
    bc._tick()
    bc._tick()
    assert seen == [0.25, 0.5, 0.75, 1.0]


def test_tick_clamps_overshoot_at_one():
    bc = _make_batch_cabana(None)
    seen = []
    bc.progress_callback = seen.append
    bc._progress_total = 2
    bc._tick()
    bc._tick()
    bc._tick()  # overshoot -> still clamped to 1.0
    bc._tick(5) # large overshoot -> still 1.0
    assert seen == [0.5, 1.0, 1.0, 1.0]


def test_tick_no_callback_is_noop():
    bc = _make_batch_cabana(None)
    # No callback -> no error, no state change for _progress_done either
    bc._progress_total = 10
    bc._tick()
    bc._tick(3)
    # Without a callback we never bother counting (intentional: avoids overhead
    # in CLI usage where progress is unused). Confirm no error and no callback.
    # _progress_done should stay at 0 because the callback gate short-circuits.
    assert bc._progress_done == 0


def test_estimate_progress_total_full_pipeline():
    bc = _make_batch_cabana(None)
    bc.args = {"Configs": {"Quantification": True, "Gap Analysis": True}}
    # 5 eligible, 5 input
    total = bc._estimate_progress_total(n_eligible=5, n_input=5)
    # generate_rois (5) + hdm (5) + detect (5) + skel (5) + orient (5) +
    # visualize (5) + calc_areas (5) + gap (5) + intra_gap (5) + colormaps (5)
    assert total == 50


def test_estimate_progress_total_no_quantification():
    bc = _make_batch_cabana(None)
    bc.args = {"Configs": {"Quantification": False, "Gap Analysis": True}}
    total = bc._estimate_progress_total(n_eligible=5, n_input=5)
    # Only generate_rois runs
    assert total == 5


def test_estimate_progress_total_no_gap_analysis():
    bc = _make_batch_cabana(None)
    bc.args = {"Configs": {"Quantification": True, "Gap Analysis": False}}
    total = bc._estimate_progress_total(n_eligible=5, n_input=5)
    # Full pipeline minus gap (5) and intra-gap (5) = 50 - 10 = 40
    assert total == 40


def test_estimate_progress_total_weights_segmentation_by_patches(tmp_path):
    import numpy as np, cv2, types
    from cabana.batch import SEG_TICKS_PER_PATCH
    bc = _make_batch_cabana(None)
    bc.args = {"Configs": {"Segmentation": True, "Quantification": False, "Gap Analysis": False}}
    cv2.imwrite(str(tmp_path / "small.png"), np.zeros((300, 300, 3), np.uint8))
    cv2.imwrite(str(tmp_path / "big.png"), np.zeros((2000, 1200, 3), np.uint8))
    bc.input_folder = str(tmp_path)
    # patch-wise segmentation off: one CNN run per image
    bc.seg_args = types.SimpleNamespace(patch_size=0)
    assert bc._estimate_progress_total(n_eligible=2, n_input=2) == 2 * SEG_TICKS_PER_PATCH
    # 1024-px patches with 1/8 overlap: the small image fits in one patch, the
    # big one needs 3 x 2 windows
    bc.seg_args = types.SimpleNamespace(patch_size=1024, patch_overlap=0.125)
    assert bc._seg_weight(str(tmp_path / "small.png")) == SEG_TICKS_PER_PATCH
    assert bc._seg_weight(str(tmp_path / "big.png")) == 6 * SEG_TICKS_PER_PATCH
    assert bc._estimate_progress_total(n_eligible=2, n_input=2) == 7 * SEG_TICKS_PER_PATCH


def test_status_callback_receives_batch_prefix(tmp_path, monkeypatch):
    import numpy as np, cv2
    input_dir = tmp_path / "in"; input_dir.mkdir()
    for i in range(2):
        cv2.imwrite(str(input_dir / f"img_{i}.png"), np.ones((4, 4, 3), np.uint8) * 200)
    out_dir = tmp_path / "out"; out_dir.mkdir()
    param_file = tmp_path / "params.yml"
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    shutil.copy(os.path.join(proj_root, "cabana", "default_params.yml"), str(param_file))
    import cabana.batch as batch_mod

    class _Chatty(_FakeBatchCabana):
        def run(self):
            self.status_callback("Segmenting img_0.png (1/2)")
            super().run()

    monkeypatch.setattr(batch_mod, "BatchCabana", _Chatty)
    bp = BatchProcessor(str(param_file), str(input_dir), str(out_dir),
                        batch_size=2, batch_num=0, resume=False, ignore_large=True)
    messages = []
    bp.status_callback = messages.append
    bp.process()
    assert messages[0] == "Batch 1/1: Segmenting img_0.png (1/2)"


def test_estimate_progress_total_no_eligible_returns_one():
    bc = _make_batch_cabana(None)
    bc.args = {"Configs": {"Quantification": True, "Gap Analysis": True}}
    total = bc._estimate_progress_total(n_eligible=0, n_input=5)
    # No eligible images -> only the final 1.0 emit; total normalized to 1
    assert total == 1


# --------------------- BatchProcessor.process integration ------------------ #


class _FakeBatchCabana:
    """Stand-in for BatchCabana that emits a few progress fractions."""

    instances = []

    def __init__(self, param_file, input_folder, batch_folder, batch_size,
                 batch_idx, ignore_large, progress_callback=None, mask_dir=None,
                 status_callback=None):
        self.batch_idx = batch_idx
        self.status_callback = status_callback
        self.progress_callback = progress_callback
        type(self).instances.append(self)

    def run(self):
        # Simulate per-image-stage ticks; final 1.0 mirrors real BatchCabana.
        for frac in (0.25, 0.5, 0.75, 1.0):
            if self.progress_callback:
                self.progress_callback(frac)


def test_process_emits_smooth_per_image_progress(tmp_path, monkeypatch):
    # Build a minimal valid param file and image inputs.
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    # Create 6 fake PNGs (split into 2 batches of 3 each at batch_size=3)
    import numpy as np, cv2
    for i in range(6):
        cv2.imwrite(str(input_dir / f"img_{i}.png"),
                    (np.ones((4, 4, 3), dtype=np.uint8) * 200))

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    param_file = tmp_path / "params.yml"
    # Copy the project default params so file existence checks pass.
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    shutil.copy(os.path.join(proj_root, "cabana", "default_params.yml"),
                str(param_file))

    # Patch BatchCabana so we don't run heavy work; only progress callback
    # behavior is under test here.
    import cabana.batch as batch_mod
    monkeypatch.setattr(batch_mod, "BatchCabana", _FakeBatchCabana)
    _FakeBatchCabana.instances = []

    seen = []
    bp = BatchProcessor(str(param_file), str(input_dir), str(out_dir),
                        batch_size=3, batch_num=0, resume=False,
                        ignore_large=True)
    bp.progress_callback = seen.append
    # Force 'ignore_large' path to skip oversized check side-effects.
    bp.process()

    # 1. All values are integers in [0, 100]
    assert all(isinstance(v, int) for v in seen)
    assert all(0 <= v <= 100 for v in seen)
    # 2. Monotone non-decreasing (no reversals)
    assert seen == sorted(seen)
    # 3. 2% (initial) and 5% (after setup) reported
    assert 2 in seen and 5 in seen
    # 4. Final per-batch endpoint is the global 5+80 = 85
    assert seen[-1] == 85
    # 5. We saw at least one mid-batch tick between 5 and 85, exclusive,
    #    proving sub-batch progress is reported (not just per-batch).
    mid_ticks = [v for v in seen if 5 < v < 85]
    assert len(mid_ticks) >= 4
    # 6. Two batches were created (6 images / batch_size=3)
    assert len(_FakeBatchCabana.instances) == 2


def test_segmenter_accepts_iter_callback_param():
    """segment_single_image must expose iter_callback as a keyword arg."""
    import inspect
    from cabana.segmenter import segment_single_image
    sig = inspect.signature(segment_single_image)
    assert 'iter_callback' in sig.parameters
    assert sig.parameters['iter_callback'].default is None


def test_generate_rois_emits_sub_image_progress(monkeypatch, tmp_path):
    """During segmentation, the iter_callback should drive multiple progress
    fractions per image — not just one tick at the end."""
    import cabana.batch as batch_mod

    # Stand in for segment_single_image: invoke iter_callback 5 times per image
    # to simulate an in-progress CNN training loop.
    def fake_segment(args, iter_callback=None):
        if iter_callback is not None:
            for it in range(5):
                iter_callback(it, 5)

    monkeypatch.setattr(batch_mod, "segment_single_image", fake_segment)

    bc = _make_batch_cabana(None)
    bc.args = {"Configs": {"Segmentation": True, "Quantification": False, "Gap Analysis": False}}

    # Construct a fake input folder with 2 images by patching get_img_paths.
    fake_paths = ["/fake/img_0.png", "/fake/img_1.png"]
    monkeypatch.setattr(batch_mod, "get_img_paths", lambda *_a, **_kw: fake_paths)

    # Set tick budget to 2 (one per image) so fractions land in [0, 1].
    bc._progress_total = 2
    bc._progress_done = 0
    bc.input_folder = "/fake"
    bc.seg_args = types.SimpleNamespace(roi_dir="/tmp", bin_dir="/tmp", input=None)

    fractions = []
    bc.progress_callback = fractions.append
    bc.generate_rois()

    # We expect at least 2 sub-image emissions per image (5 iter callbacks +
    # one final _tick per image = 6 emissions per image, 12 total). Even if
    # _tick deduplicates against the last iter emission, we should be well
    # above the per-image-only count of 2.
    assert len(fractions) >= 10, f"expected >=10 progress emissions, got {len(fractions)}: {fractions}"
    # Monotone non-decreasing
    assert fractions == sorted(fractions), f"non-monotone: {fractions}"
    # Final emission lands at exactly 1.0 (both images fully ticked).
    assert fractions[-1] == 1.0


def test_generate_rois_no_segmentation_skips_iter_callback(monkeypatch, tmp_path):
    """When Configs.Segmentation is False, generate_rois copies images via
    a trivial mask instead of running the CNN. Per-image ticks still fire,
    but no iter_callback is involved."""
    import cabana.batch as batch_mod
    import numpy as np
    import cv2

    # Build a small input image so generate_rois' fallback branch can read it.
    src = tmp_path / "src"
    src.mkdir()
    paths = []
    for i in range(2):
        p = src / f"img_{i}.png"
        cv2.imwrite(str(p), np.ones((4, 4, 3), dtype=np.uint8) * 200)
        paths.append(str(p))

    out = tmp_path / "out"
    (out / "roi").mkdir(parents=True)
    (out / "bin").mkdir(parents=True)

    bc = _make_batch_cabana(None)
    bc.args = {"Configs": {"Segmentation": False, "Quantification": False, "Gap Analysis": False}}
    bc.input_folder = str(src)
    bc.seg_args = types.SimpleNamespace(
        roi_dir=str(out / "roi"), bin_dir=str(out / "bin"), input=None
    )
    bc._progress_total = 2
    bc._progress_done = 0

    fractions = []
    bc.progress_callback = fractions.append
    bc.generate_rois()

    # Exactly one tick per image -> two emissions, ending at 1.0
    assert fractions == [0.5, 1.0]


def test_process_resume_advances_progress_immediately(tmp_path, monkeypatch):
    """Resuming from batch 1 of 3 should jump the bar to ~57% before any work."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    import numpy as np, cv2
    for i in range(9):
        cv2.imwrite(str(input_dir / f"img_{i}.png"),
                    (np.ones((4, 4, 3), dtype=np.uint8) * 200))

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # Pre-create the batches subfolder + checkpoint so resume path validates.
    (out_dir / "Batches").mkdir()

    param_file = tmp_path / "params.yml"
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    shutil.copy(os.path.join(proj_root, "cabana", "default_params.yml"),
                str(param_file))

    # Pre-write the checkpoint file so the resume path can read it.
    with open(str(out_dir / ".CheckPoint.txt"), "w") as f:
        f.write(f"Input Folder,{input_dir}\n")
        f.write("Batch Size,3\n")
        f.write("Batch Number,0\n")  # batch 0 already done
        f.write("Ignore Large,True\n")

    import cabana.batch as batch_mod
    monkeypatch.setattr(batch_mod, "BatchCabana", _FakeBatchCabana)
    _FakeBatchCabana.instances = []

    seen = []
    bp = BatchProcessor(str(param_file), str(input_dir), str(out_dir),
                        batch_size=3, batch_num=0, resume=True,
                        ignore_large=True)
    bp.progress_callback = seen.append
    bp.process()

    # With 9 images, batch 0 done = 3 images done before resume.
    # Resumed initial advance: 5 + 3/9 * 80 ≈ 32%
    # Confirm the bar jumps past 5% before BatchCabana runs.
    assert any(v >= 30 for v in seen[:6]), f"resume did not advance bar early: {seen[:6]}"
    # Final endpoint is still 85.
    assert seen[-1] == 85


def test_post_process_moves_results_and_merges_summaries(tmp_path):
    """Consolidation must not duplicate results on disk, must merge every
    batch's summary CSVs, and must remove the Batches folder afterwards."""
    import pandas as pd
    out = tmp_path / "out"
    for b in range(2):
        bf = out / "Batches" / f"batch_{b}"
        for sub in ("ROIs", "Bins", "Masks", "HDM", "Exports", "Fibres", "Eligible", "Colors"):
            (bf / sub).mkdir(parents=True)
            (bf / sub / f"img{b}.png").write_bytes(b"png")
        (bf / "Masks" / "GapAnalysis").mkdir()
        (bf / "Masks" / "GapAnalysis" / f"gaps{b}.png").write_bytes(b"png")
        (bf / "Masks" / "GapAnalysis" / f"IndividualGaps_img{b}_roi.csv").write_text("a\n1\n")
        pd.DataFrame({"Image": [f"img{b}"], "Mean": [b]}).to_csv(
            bf / "Masks" / "GapAnalysis" / "GapAnalysisSummary.csv", index=False)
        pd.DataFrame({"Image": [f"img{b}"], "Mean": [b]}).to_csv(
            bf / "Masks" / "GapAnalysis" / "IntraGapAnalysisSummary.csv", index=False)
        (bf / "Exports" / f"img{b}_roi").mkdir()
        (bf / "Exports" / f"img{b}_roi" / "Curve_Map_10.tif").write_bytes(b"tif")
        (bf / "Colors" / f"img{b}_roi").mkdir()
        (bf / "Colors" / f"img{b}_roi" / "all_gaps.png").write_bytes(b"png")
        (bf / "Eligible" / "IgnoredImages.txt").write_text("")
        pd.DataFrame({"Image": [f"img{b}"], "Value": [b]}).to_csv(bf / "QuantificationResults.csv", index=False)
    bp = _make_processor(None)
    bp.output_folder = str(out)
    bp.batch_num = 1
    bp.generate_stats = bp.generate_scores = False
    bp.status_callback = None
    bp.post_process()
    assert not (out / "Batches").exists()
    assert sorted(p.name for p in (out / "ROIs").iterdir()) == ["img0.png", "img1.png"]
    assert (out / "Exports" / "img0_roi" / "Curve_Map_10.tif").exists()
    assert (out / "Exports" / "img1_roi" / "Curve_Map_10.tif").exists()
    assert (out / "Colors" / "img1_roi" / "all_gaps.png").exists()
    assert (out / "Masks" / "GapAnalysis" / "IndividualGaps_img0_roi.csv").exists()
    for name in ("GapAnalysisSummary.csv", "IntraGapAnalysisSummary.csv"):
        assert len(pd.read_csv(out / "Masks" / "GapAnalysis" / name)) == 2
    assert len(pd.read_csv(out / "QuantificationResults.csv")) == 2


def test_bad_paths_raise_instead_of_exiting(tmp_path):
    """A missing folder must raise so the GUI worker can report it; os._exit killed the GUI."""
    param_file = tmp_path / "p.yml"
    param_file.write_text("Configs: {}\n")
    with pytest.raises(FileNotFoundError):
        BatchProcessor(str(param_file), str(tmp_path / "missing"), str(tmp_path / "out"),
                       batch_size=2, batch_num=0, resume=False, ignore_large=True)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        BatchProcessor(str(param_file), str(empty), str(tmp_path / "out"),
                       batch_size=2, batch_num=0, resume=False, ignore_large=True)
    with pytest.raises(FileNotFoundError):
        BatchProcessor(str(tmp_path / "nope.yml"), str(empty), str(tmp_path / "out"),
                       batch_size=2, batch_num=0, resume=False, ignore_large=True)
