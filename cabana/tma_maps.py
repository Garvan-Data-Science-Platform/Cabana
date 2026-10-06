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
from functools import lru_cache
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


@lru_cache(maxsize=None)
def load_array_map(array_number):
    """Return ``{(row_index, col_index): CoreInfo}`` for one array.

    Every one of the 12 x 8 cells is present; empty cells have ``empty=True``.
    The result is cached (the maps are static and read-only): the GUI overlay
    looks a core's entry up on every repaint.
    """
    array_number = int(array_number)
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


PATIENT_CLASS = "Tumor"    # annotation class written for patient cores without a map note


def core_stem(slide, row_index, col_index, info=None, channel=None, replicate=1):
    """Build the output filename stem for one core.

    The stem follows the naming of the QuPath slide exports Cabana was built
    around, ``<patient>.vsi - <description>Annotation (<class>)_<n>``, so the
    per-patient statistics (:func:`cabana.scores.parse_image_name`) read the
    patient ID from the prefix, the channel from ``_BF_``/``_POL_``, the class
    from the last bracket and the replicate number from the trailing ``_<n>``:

        8010718.vsi - TMA1_BF_A3Annotation (Tumor)_1      patient core
        Liver.vsi - TMA1_BF_A1Annotation (Liver)_1          control core
        TMA1-r2c5.vsi - TMA1_BF_r2c5Annotation (Unmapped)_1 no map entry

    ``replicate`` numbers the cores of one patient (or control tissue) on the
    slide. The class is the map note (e.g. PNET) when present, else
    :data:`PATIENT_CLASS`.
    """
    slide = sanitize_token(str(slide))
    ch = f"{sanitize_token(channel)}_" if channel else ""
    if info is None or info.empty:
        position = f"r{row_index + 1}c{col_index + 1}"
        subject, klass = f"{slide}-{position}", "Unmapped"
    elif info.patient_id:
        position = info.position
        subject = sanitize_token(info.patient_id)
        klass = sanitize_token(info.note) if info.note else PATIENT_CLASS
    else:
        position = info.position
        subject = klass = sanitize_token(info.tissue)
    return f"{subject}.vsi - {slide}_{ch}{position}Annotation ({klass})_{replicate}"
