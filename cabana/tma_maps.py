"""ICGC/APGI tissue micro-array (TMA) layout maps.

The maps are transcribed from the APGI "ICGC TMA maps" PDF into
``cabana/data/icgc_arrays.csv``. Each array is printed as three sectors of
4 rows x 8 columns stacked vertically, giving a flat grid of 12 rows (A..L)
by 8 columns (1..8). The PDF labels cells with sector-local names (e.g. the
second sector starts at column 5), so both the original ``label`` and the
flat ``row``/``col`` position are stored.

Arrays 1 and 2 have an empty last row (88 cores); arrays 3 to 8 have 96.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ARRAY_MAP_CSV = Path(__file__).parent / "data" / "icgc_arrays.csv"
ROW_LETTERS = "ABCDEFGHIJKL"
MAP_ROWS = 12
MAP_COLS = 8


@dataclass(frozen=True)
class CoreInfo:
    """One cell of a TMA map."""
    array: int
    sector: int
    label: str
    row: str          # flat row letter A..L
    col: int          # flat column 1..8
    patient_id: str
    icgc_id: str
    tissue: str       # control tissue name; empty for patient cores
    note: str         # e.g. PNET, MCN
    empty: bool

    @property
    def row_index(self):
        return ROW_LETTERS.index(self.row)

    @property
    def col_index(self):
        return self.col - 1

    @property
    def position(self):
        return f"{self.row}{self.col}"

    @property
    def is_control(self):
        return bool(self.tissue) and not self.patient_id

    def identity(self):
        """Filename-safe identity string: patient and ICGC IDs, or tissue name."""
        if self.empty:
            return "empty"
        if self.patient_id:
            ident = self.patient_id
            if self.icgc_id:
                ident += f"_{self.icgc_id}"
            return ident
        return sanitize_token(self.tissue)


def sanitize_token(text):
    """Make a map field safe for use inside a filename."""
    out = []
    for ch in text.strip():
        out.append(ch if ch.isalnum() else "-")
    token = "".join(out)
    while "--" in token:
        token = token.replace("--", "-")
    return token.strip("-") or "unknown"


def _read_rows():
    with open(ARRAY_MAP_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def available_arrays():
    """Sorted list of array numbers present in the map file."""
    return sorted({int(r["array"]) for r in _read_rows()})


def load_array_map(array_number):
    """Return ``{(row_index, col_index): CoreInfo}`` for one array.

    Every one of the 12 x 8 cells is present; empty cells have ``empty=True``.
    """
    cells = {}
    for r in _read_rows():
        if int(r["array"]) != int(array_number):
            continue
        info = CoreInfo(array=int(r["array"]), sector=int(r["sector"]), label=r["label"],
                        row=r["row"], col=int(r["col"]), patient_id=r["patient_id"],
                        icgc_id=r["icgc_id"], tissue=r["tissue"], note=r["note"],
                        empty=bool(int(r["empty"])))
        cells[(info.row_index, info.col_index)] = info
    if not cells:
        raise KeyError(f"Array {array_number} is not in {ARRAY_MAP_CSV.name}; "
                       f"available: {available_arrays()}")
    return cells


def occupancy_grid(array_number):
    """Boolean ``(12, 8)`` array, True where the map has a core."""
    grid = np.zeros((MAP_ROWS, MAP_COLS), dtype=bool)
    for (r, c), info in load_array_map(array_number).items():
        grid[r, c] = not info.empty
    return grid


def core_stem(slide, row_index, col_index, info=None, channel=None):
    """Build the output filename stem for one core.

    ``TMA1_A3_8010718_1734_BF`` for patient cores, ``TMA1_A1_Liver_BF`` for
    controls and ``TMA1_r2c5_unknown_BF`` when no map entry is available.
    """
    slide = sanitize_token(str(slide))
    if info is None or info.empty:
        stem = f"{slide}_r{row_index + 1}c{col_index + 1}_unknown"
    else:
        stem = f"{slide}_{info.position}_{info.identity()}"
    if channel:
        stem += f"_{sanitize_token(channel)}"
    return stem
