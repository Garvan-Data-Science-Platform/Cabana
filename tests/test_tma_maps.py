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
            assert info.tissue or (info.patient_id and info.block_id), info

    def test_spot_checks_against_pdf(self):
        a1 = load_array_map(1)
        assert a1[(0, 0)].tissue == "Liver"
        assert a1[(0, 3)].patient_id == "1734" and a1[(0, 3)].block_id == "8010718"
        assert a1[(3, 6)].tissue == "Brain"
        assert a1[(4, 0)].label == "A5" and a1[(4, 0)].sector == 2
        a6 = load_array_map(6)
        assert a6[(0, 3)].patient_id == "2144" and a6[(0, 3)].block_id == "8046501"
        assert a6[(0, 3)].note == "MCN"
        assert a6[(8, 6)].tissue == "Kidney/ muscle"
        a8 = load_array_map(8)
        assert a8[(0, 0)].label == "D3" and a8[(0, 0)].patient_id == "3804" and a8[(0, 0)].block_id == "8070444"

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
        assert core_stem("TMA1", 0, 3, info, "BF") == "1734.vsi - TMA1_BF_A4Annotation (Tumor)_1"
        assert core_stem("TMA1", 0, 3, info, "POL", replicate=3) == "1734.vsi - TMA1_POL_A4Annotation (Tumor)_3"

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
        info = CoreInfo(1, 1, "B3", "B", 3, "2113", "8012191", "", "", False)
        assert info.row_index == 1 and info.col_index == 2 and info.position == "B3"
        assert not info.is_control and info.identity() == "2113_8012191"


class TestMapConsistency:
    """Every (patient, block) pair appears exactly three times per array,
    one core per sector on arrays 3 to 8 (arrays 1 and 2 shuffle patients
    between sectors). Two exceptions are printed in the PDF and verified
    against its page images: on Array 6 block 8070344 carries two patient
    IDs (3368 and 3541), each in triplicate; Array 7 patient 3290 (block
    8062699) appears twice because a Kidney control occupies its sector-3
    slot."""

    EXCEPTIONS = {(7, "3290", "8062699"): 2}

    @pytest.mark.parametrize("array", range(1, 9))
    def test_each_sample_in_triplicate(self, array):
        from collections import Counter
        cells = [c for c in load_array_map(array).values() if not c.empty and c.patient_id]
        counts = Counter((c.patient_id, c.block_id) for c in cells)
        for (pid, block), n in counts.items():
            assert n == self.EXCEPTIONS.get((array, pid, block), 3), (array, pid, block, n)

    @pytest.mark.parametrize("array", range(3, 9))
    def test_one_core_per_sector_on_later_arrays(self, array):
        from collections import Counter
        cells = [c for c in load_array_map(array).values() if not c.empty and c.patient_id]
        per_sector = Counter((c.patient_id, c.block_id, c.sector) for c in cells)
        assert max(per_sector.values()) == 1

    def test_patient_id_is_four_digits_and_has_one_block(self):
        blocks = {}
        for array in range(1, 9):
            for c in load_array_map(array).values():
                if c.patient_id:
                    assert len(c.patient_id) == 4 and len(c.block_id) in (6, 7), c
                    blocks.setdefault(c.patient_id, set()).add(c.block_id)
        assert all(len(b) == 1 for b in blocks.values())

    def test_control_counts(self):
        for array in range(1, 9):
            controls = [c for c in load_array_map(array).values() if c.is_control]
            assert 3 <= len(controls) <= 9, (array, len(controls))


# ---------------------------------------------------------------------------
# User map files
# ---------------------------------------------------------------------------

from cabana.tma_maps import (TMAMap, as_map, builtin_map, describe_map, load_map_file,  # noqa: E402
                             match_map_to_slide, write_map_list, write_templates)


def _write(tmp_path, name, text, encoding="utf-8"):
    path = tmp_path / name
    path.write_text(text, encoding=encoding)
    return str(path)


class TestBuiltinMapObject:
    def test_matches_legacy_loader(self):
        for n in range(1, 9):
            m = builtin_map(n)
            assert m.shape == (MAP_ROWS, MAP_COLS) and m.array == n and m.is_builtin
            assert (m.occupancy() == occupancy_grid(n)).all()
            assert m.cells == load_array_map(n)

    def test_as_map(self):
        assert as_map(None) is None
        assert as_map(3) is builtin_map(3)
        m = builtin_map(4)
        assert as_map(m) is m

    def test_digest_distinguishes_maps(self):
        assert builtin_map(1).digest() != builtin_map(2).digest()
        assert builtin_map(1).digest() == builtin_map(1).digest()


class TestGridFiles:
    def test_template(self, tmp_path):
        (grid,) = write_templates(str(tmp_path))
        assert os.path.basename(grid) == "tma_map_template.csv"
        g = load_map_file(grid)
        assert g.ok and not g.problems and [m.name for m in g.maps] == ["TMA1"]
        m = g.maps[0]
        assert m.shape == (12, 12) and m.n_cores == 22                    # ready-labelled 12 x 12 grid
        assert m.cells[(0, 0)].is_control and m.cells[(0, 0)].tissue == "Liver"
        assert m.cells[(1, 2)].is_unknown and m.cells[(4, 0)].empty
        assert m.cells[(0, 1)].patient_id == "1001" and m.cells[(0, 1)].position == "A2"
        pasted = m.cells[(2, 2)]                                          # a multi-line example cell
        assert (pasted.patient_id, pasted.block_id, pasted.note) == ("1011", "801011", "PNET")
        assert m.cells[(2, 0)].tissue == "Lung"

    def test_qupath_grid_with_tabs(self, tmp_path):
        mf = load_map_file(_write(tmp_path, "slide.qpmap", "\t1\t2\t3\nA\t1599\t\t1734\nB\t?\tctrl:Liver\t2000\n"))
        assert mf.ok and mf.maps[0].name == "slide"
        m = mf.maps[0]
        assert m.shape == (2, 3) and m.cells[(0, 1)].empty and m.cells[(1, 1)].tissue == "Liver"

    def test_grid_without_headers(self, tmp_path):
        mf = load_map_file(_write(tmp_path, "g.csv", "1599,1734,ctrl:Lung\n,2001,2002\n2003,,2004\n"))
        m = mf.maps[0]
        assert m.shape == (3, 3) and m.cells[(1, 0)].empty and m.cells[(0, 0)].patient_id == "1599"
        assert m.row_labels == ["A", "B", "C"] and m.col_labels == [1, 2, 3]

    def test_header_keeps_empty_edge_row(self, tmp_path):
        mf = load_map_file(_write(tmp_path, "g.csv", ",1,2\nA,P1,P2\nB,P3,P4\nC,,\n"))
        assert mf.maps[0].shape == (3, 2) and mf.maps[0].n_cores == 4

    def test_several_maps_and_prefix_warning(self, tmp_path):
        text = "# comment\nmap,TMA2\n,1,2\nA,P1,P2\n\nmap,TMA3\n,1,2\nA,P3,ctl:Liver\n"
        mf = load_map_file(_write(tmp_path, "m.csv", text))
        assert mf.ok and [m.name for m in mf.maps] == ["TMA2", "TMA3"]
        assert any("ctl:Liver" in p.message for p in mf.warnings)
        assert mf.maps[1].cells[(0, 1)].patient_id == "ctl:Liver"

    def test_semicolons_and_excel_bom(self, tmp_path):
        mf = load_map_file(_write(tmp_path, "s.csv", "﻿,1;2\n".replace(",", ";") + "A;P1;ctrl:Liver\n",
                                  encoding="utf-8"))
        assert mf.ok and mf.maps[0].cells[(0, 1)].tissue == "Liver"

    def test_numbered_rows_position(self, tmp_path):
        mf = load_map_file(_write(tmp_path, "n.csv", ",1,2\n1,P1,P2\n2,ctrl:Liver,\n"))
        assert mf.maps[0].cells[(1, 0)].position == "R2C1"

    @pytest.mark.parametrize("text,needle", [
        (",1,2\nA,P1,P2,P3\n", "more cells than column labels"),
        ("map,X\n,1\nA,P1\nmap,X\n,1\nA,P2\n", "appears twice"),
        (",1,2\nA,ctrl:,P1\n", "names no control tissue"),
        ("", "holds no map"),
    ])
    def test_grid_errors(self, tmp_path, text, needle):
        mf = load_map_file(_write(tmp_path, "e.csv", text))
        assert not mf.ok and not mf.maps
        assert any(needle in p.message for p in mf.errors), [str(p) for p in mf.problems]


class TestListFiles:
    def test_list_with_extras_and_unknown(self, tmp_path):
        text = "map,row,col,patient_id,tissue,class,block_id,site\nT,A,1,P1,,PNET,B1,head\nT,A,2,,Liver,,,\nT,B,1,,,,,\n"
        m = load_map_file(_write(tmp_path, "l.csv", text)).maps[0]
        assert m.cells[(0, 0)].note == "PNET" and m.cells[(0, 0)].block_id == "B1"
        assert dict(m.cells[(0, 0)].extras) == {"site": "head"} and m.extra_columns() == ["site"]
        assert m.cells[(1, 0)].is_unknown and m.cells[(1, 1)].empty

    def test_map_column_optional(self, tmp_path):
        mf = load_map_file(_write(tmp_path, "solo.csv", "row,col,patient_id\nA,1,P1\n"))
        assert [m.name for m in mf.maps] == ["solo"]

    @pytest.mark.parametrize("text,needle", [
        ("row,col,patient_id\nA,1,P1\nA,1,P2\n", "twice"),
        ("row,col,patient_id\nA,1,P1\n2,2,P2\n", "mixes letters and numbers"),
        ("row,col,patient_id\nA,x1,P1\n", "not a row or column label"),
        ("row,col,patient_id,status\nA,1,P1,gone\n", "status"),
    ])
    def test_list_errors(self, tmp_path, text, needle):
        mf = load_map_file(_write(tmp_path, "e.csv", text))
        assert not mf.ok
        assert any(needle in p.message for p in mf.errors), [str(p) for p in mf.problems]

    def test_warnings(self, tmp_path):
        rows = "".join(f"A,{k},P1\n" for k in range(1, 5)) + "".join(f"B,{k},P{k + 1}\n" for k in range(1, 5))
        mf = load_map_file(_write(tmp_path, "w.csv", "row,col,patient_id,tissue\n" + rows + "C,1,P9,Liver\n"))
        assert mf.ok
        msgs = " ".join(p.message for p in mf.warnings)
        assert "P1 appears 4 times" in msgs and "both a patient ID and a tissue" in msgs

    def test_builtin_round_trip_through_list_file(self, tmp_path):
        path = str(tmp_path / "icgc.csv")
        write_map_list([builtin_map(1), builtin_map(2)], path)
        back = load_map_file(path)
        assert back.ok and [m.name for m in back.maps] == ["ICGC Array 1", "ICGC Array 2"]
        for m, n in zip(back.maps, (1, 2)):
            assert m.shape == (12, 8)                                     # empty row L kept
            assert (m.occupancy() == occupancy_grid(n)).all()
            for rc, info in load_array_map(n).items():
                got = m.cells[rc]
                assert (got.patient_id, got.block_id, got.tissue, got.note) == \
                       (info.patient_id, info.block_id, info.tissue, info.note)


class TestHelpers:
    def test_match_map_to_slide(self, tmp_path):
        mf = load_map_file(_write(tmp_path, "m.csv", "map,TMA1\n,1\nA,P1\nmap,TMA12\n,1\nA,P2\n"))
        assert match_map_to_slide(mf.maps, "APGI TMA 12 PicRed.vsi").name == "TMA12"
        assert match_map_to_slide(mf.maps, "APGI TMA 1 PicRed.vsi").name == "TMA1"
        assert match_map_to_slide(mf.maps, "other.vsi") is None

    def test_describe_map(self):
        text = describe_map(builtin_map(1))
        assert "Map 'ICGC Array 1': 12 rows x 8 columns, 88 cores" in text and "ctrl:Liver" in text

    def test_unknown_core_stem_and_identity(self):
        info = CoreInfo("T", 0, "B3", "B", 3, "", "", "", "", False, 1, 2)
        assert info.is_unknown and info.identity() == "unknown-B3"
        assert core_stem("S", 1, 2, info, "BF") == "S-B3.vsi - S_BF_B3Annotation (Unknown)_1"

    def test_user_map_fills_missing_cells(self):
        m = TMAMap("X", {}, (2, 2), ["A", "B"], [1, 2])
        assert len(m.cells) == 4 and all(i.empty for i in m.cells.values()) and m.n_cores == 0


class TestCommentsInSpreadsheets:
    def test_template_comments_stay_in_one_cell(self):
        from cabana.tma_maps import GRID_TEMPLATE
        for text in (GRID_TEMPLATE,):
            for line in text.splitlines():
                if line.startswith("#"):
                    assert not any(d in line for d in ",;\t"), line   # a spreadsheet would split it

    def test_quoted_and_split_comments_are_ignored(self, tmp_path):
        text = ('"# a comment, quoted by Excel"\n'
                '# a comment,split,into cells\n'
                ',1,2\nA,P1,P2\nB,P3,ctrl:Liver\n')
        mf = load_map_file(_write(tmp_path, "c.csv", text))
        assert mf.ok and mf.maps[0].shape == (2, 2) and mf.maps[0].n_cores == 4


class TestPastedFromPdf:
    """Cells pasted from a PDF hold several lines; such files must be refused, not misread."""

    PASTED = ('map,TMA6\n,1,2\nA,"A1\nLiver","A2\n3000\n8064399"\n'
              'B,"B1\n2133\n8032572","B2\n2343\n8053543"\n')

    def test_multiline_grid_cells_are_errors(self, tmp_path):
        mf = load_map_file(_write(tmp_path, "p.csv", self.PASTED))
        assert not mf.ok and not mf.maps
        msgs = [p.message for p in mf.errors]
        assert any("'A2 / 3000 / 8064399'" in m and "row A, column 2" in m for m in msgs)
        assert not any("has no cores" in m for m in msgs)                 # no redundant follow-up error

    def test_multiline_cells_keep_line_numbers(self, tmp_path):
        mf = load_map_file(_write(tmp_path, "p.csv", self.PASTED))
        b1 = next(p.message for p in mf.errors if "'B1 / 2133 / 8032572'" in p.message)
        assert b1.startswith("Line 7,")                                   # the A row spans lines 3 to 6

    def test_multiline_list_cells_are_errors(self, tmp_path):
        mf = load_map_file(_write(tmp_path, "l.csv", 'row,col,patient_id\nA,1,"3000\n8064399"\n'))
        assert not mf.ok and any("several lines" in p.message for p in mf.errors)

    def test_unlabelled_rows_in_a_labelled_grid(self, tmp_path):
        mf = load_map_file(_write(tmp_path, "u.csv", ",1,2\nA,P1,P2\n,P3,P4\n"))
        assert not mf.ok
        assert any("instead of a row label" in p.message for p in mf.errors)

    def test_too_few_column_labels_says_how_to_fix(self, tmp_path):
        mf = load_map_file(_write(tmp_path, "c.csv", ",1,2\nA,P1,P2,P3\n"))
        assert any("add the missing column numbers" in p.message for p in mf.errors)

    def test_quoted_single_line_cells_still_fine(self, tmp_path):
        mf = load_map_file(_write(tmp_path, "q.csv", ',1,2\nA,"P1","ctrl:Kidney/ muscle"\n'))
        assert mf.ok and mf.maps[0].cells[(0, 1)].tissue == "Kidney/ muscle"


class TestCellLines:
    """A 'cell lines' row describes pasted cells holding several lines."""

    def _load(self, tmp_path, body, spec="cell lines,position patient block class\n"):
        return load_map_file(_write(tmp_path, "p.csv", "map,T\n" + spec + body))

    def test_pasted_pdf_cells(self, tmp_path):
        body = (',1,2,3\nA,"A1\nLiver","A2\n3000\n8064399","A4\n2144\n8046501\n(MCN)"\n'
                'B,"B1\nKidney/\nmuscle","B2","B3\n?"\n')
        mf = self._load(tmp_path, body)
        assert mf.ok, [str(p) for p in mf.problems]
        c = mf.maps[0].cells
        assert c[(0, 0)].tissue == "Liver" and c[(0, 0)].is_control
        assert (c[(0, 1)].patient_id, c[(0, 1)].block_id) == ("3000", "8064399")
        assert (c[(0, 2)].patient_id, c[(0, 2)].note) == ("2144", "MCN")
        assert c[(1, 0)].tissue == "Kidney/ muscle"
        assert c[(1, 1)].patient_id == "B2"                               # single value: read as typed
        assert any("looks like a printed position" in p.message for p in mf.warnings)
        assert c[(1, 2)].is_unknown

    def test_lone_position_is_kept_with_a_warning(self, tmp_path):
        # a pasted empty position arrives as its bare label; it is never dropped silently
        mf = self._load(tmp_path, ',1,2\nA,"A1\n","A2\n3000"\nB,1,2\n')
        assert mf.ok and mf.maps[0].cells[(0, 0)].patient_id == "A1" and mf.maps[0].cells[(0, 1)].patient_id == "3000"
        assert any("'A1' looks like a printed position" in p.message for p in mf.warnings)

    def test_order_without_position(self, tmp_path):
        mf = self._load(tmp_path, ',1\nA,"8064399\n3000"\nB,7\n', spec="cell lines,block patient\n")
        assert mf.ok and (mf.maps[0].cells[(0, 0)].patient_id, mf.maps[0].cells[(0, 0)].block_id) == \
            ("3000", "8064399")

    def test_wrong_order_is_caught(self, tmp_path):
        mf = self._load(tmp_path, ',1\nA,"3000\n8064399"\n')              # no position line in the cells
        assert not mf.ok and any("is not a position such as A1" in p.message for p in mf.errors)

    def test_skip_and_extra_lines(self, tmp_path):
        mf = self._load(tmp_path, ',1\nA,"A1\nx9\n3000\n77\n88"\n', spec="cell lines,position skip patient block\n")
        assert mf.ok and mf.maps[0].cells[(0, 0)].patient_id == "3000" and mf.maps[0].cells[(0, 0)].block_id == "77"
        assert any("more lines than" in p.message for p in mf.warnings)

    @pytest.mark.parametrize("spec,needle", [("cell lines,position block\n", "include patient"),
                                             ("cell lines,position patient id\n", "'id' is not one of them")])
    def test_bad_spec(self, tmp_path, spec, needle):
        mf = self._load(tmp_path, ',1\nA,"A1\n3000"\n', spec=spec)
        assert not mf.ok and any(needle in p.message for p in mf.errors)

    def test_spec_after_header_is_an_error(self, tmp_path):
        mf = load_map_file(_write(tmp_path, "p.csv", "map,T\n,1\ncell lines,position patient\nA,P1\n"))
        assert any("must come before the row of column numbers" in p.message for p in mf.errors)

    def test_file_level_spec_applies_to_every_map(self, tmp_path):
        text = ('cell lines,position patient\nmap,T1\n,1\nA,"A1\n11"\nmap,T2\n,1\nA,"A1\n22"\n')
        mf = load_map_file(_write(tmp_path, "p.csv", text))
        assert mf.ok and [m.cells[(0, 0)].patient_id for m in mf.maps] == ["11", "22"]

    def test_without_spec_multiline_mentions_the_option(self, tmp_path):
        mf = load_map_file(_write(tmp_path, "p.csv", ',1\nA,"A1\n3000"\n'))
        assert any("cell lines" in p.message for p in mf.errors)
