"""Tissue micro-array (TMA) layout maps.

Two sources of maps share one representation, :class:`TMAMap`:

* the built-in ICGC/APGI arrays 1 to 8 (:func:`builtin_map`), and
* map files supplied by the user (:func:`load_map_file`), in a grid layout
  that looks like the printed map or a list layout with one line per core.

The rest of this docstring describes the built-in maps.

The maps are transcribed from the APGI "ICGC TMA maps" PDF into
``cabana/data/icgc_arrays.csv``. Each array is printed as three sectors of
4 rows x 8 columns stacked vertically, giving a flat grid of 12 rows (A..L)
by 8 columns (1..8). The PDF labels cells with sector-local names (e.g. the
second sector starts at column 5), so both the original ``label`` and the
flat ``row``/``col`` position are stored.

Each patient cell in the PDF prints two numbers: the APGI block ID (6 or 7 digits)
above the 4-digit patient ID. The patient ID names the exported images and
groups the replicate cores; the block ID is kept for traceability.

Arrays 1 and 2 have an empty last row (88 cores); arrays 3 to 8 have 96.
"""

import csv
import hashlib
import io
import json
import os
import re
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
    """One cell of a TMA map.

    A cell is a patient core (``patient_id``), a control core (``tissue``
    without ``patient_id``), a core whose identity is unknown (neither), or
    an empty position (``empty``).
    """
    array: object     # map: the array number of a built-in map, else the map name
    sector: int       # printed sector of the ICGC maps; 0 for user maps
    label: str        # printed label (ICGC: sector-local); the position for user maps
    row: str          # row label, e.g. "A" (or "3" for numbered rows)
    col: object       # column label, e.g. 1
    patient_id: str   # names the exported images and groups the replicates
    block_id: str     # tissue block ID, kept for traceability
    tissue: str       # control tissue name; empty for patient cores
    note: str         # annotation class, e.g. PNET, MCN; default Tumor
    empty: bool
    r: int = -1       # 0-based grid row; -1 derives it from an ICGC row letter
    c: int = -1       # 0-based grid column; -1 derives it from the column number
    extras: tuple = ()   # ((column, value), ...) further columns of a map file

    @property
    def row_index(self):
        return self.r if self.r >= 0 else ROW_LETTERS.index(self.row)

    @property
    def col_index(self):
        return self.c if self.c >= 0 else self.col - 1

    @property
    def position(self):
        return position_label(self.row, self.col)

    @property
    def is_control(self):
        return bool(self.tissue) and not self.patient_id

    @property
    def is_unknown(self):
        """A core whose identity the map does not give."""
        return not self.empty and not self.patient_id and not self.tissue

    def identity(self):
        """Filename-safe identity string: patient and block IDs, or tissue name."""
        if self.empty:
            return "empty"
        if self.is_unknown:
            return f"unknown-{sanitize_token(self.position)}"
        if self.patient_id:
            ident = self.patient_id
            if self.block_id:
                ident += f"_{self.block_id}"
            return ident
        return sanitize_token(self.tissue)


def position_label(row, col):
    """Printed position of a cell, e.g. ``A3``; ``R3C12`` for numbered rows."""
    if str(row).isdigit():
        return f"R{row}C{col}"     # numbered rows: keep row and column apart
    return f"{row}{col}"


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
                        block_id=r["block_id"], tissue=r["tissue"], note=r["note"],
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

        1734.vsi - TMA1_BF_A4Annotation (Tumor)_1         patient core
        Liver.vsi - TMA1_BF_A1Annotation (Liver)_1          control core
        TMA1-r2c5.vsi - TMA1_BF_r2c5Annotation (Unmapped)_1 no map entry
        TMA1-B3.vsi - TMA1_BF_B3Annotation (Unknown)_1      core of unknown identity

    ``replicate`` numbers the cores of one patient (or control tissue) on the
    slide. The class is the map note (e.g. PNET) when present, else
    :data:`PATIENT_CLASS`.
    """
    slide = sanitize_token(str(slide))
    ch = f"{sanitize_token(channel)}_" if channel else ""
    if info is None or info.empty:
        position = f"r{row_index + 1}c{col_index + 1}"
        subject, klass = f"{slide}-{position}", "Unmapped"
    elif info.is_unknown:
        position = sanitize_token(info.position)
        subject, klass, replicate = f"{slide}-{position}", "Unknown", 1
    elif info.patient_id:
        position = info.position
        subject = sanitize_token(info.patient_id)
        klass = sanitize_token(info.note) if info.note else PATIENT_CLASS
    else:
        position = info.position
        subject = klass = sanitize_token(info.tissue)
    return f"{subject}.vsi - {slide}_{ch}{position}Annotation ({klass})_{replicate}"


# ---------------------------------------------------------------------------
# Map objects (built-in and user maps)
# ---------------------------------------------------------------------------

CONTROL_PREFIX = "ctrl:"     # grid cell text marking a control core, e.g. ctrl:Liver
UNKNOWN_TOKEN = "?"          # grid cell text marking a core of unknown identity
MAP_FILE_FILTER = "TMA maps (*.csv *.tsv *.txt *.qpmap);;All files (*)"


class TMAMap:
    """A TMA layout: a rectangular grid of :class:`CoreInfo` cells.

    ``cells`` holds every position of the grid; positions a map file does
    not list are empty. ``array`` is the array number of a built-in map and
    the name of a user map.
    """

    def __init__(self, name, cells, shape, row_labels, col_labels, source="", array=None):
        self.name = str(name)
        self.array = array if array is not None else self.name
        self.shape = (int(shape[0]), int(shape[1]))
        self.row_labels = list(row_labels)
        self.col_labels = list(col_labels)
        self.source = source
        self.cells = dict(cells)
        for r in range(self.shape[0]):
            for c in range(self.shape[1]):
                if (r, c) not in self.cells:
                    row, col = self.row_labels[r], self.col_labels[c]
                    self.cells[(r, c)] = CoreInfo(self.array, 0, position_label(row, col), row, col,
                                                  "", "", "", "", True, r, c)
        self._digest = None

    def __repr__(self):
        return f"TMAMap({self.name!r}, {self.shape[0]}x{self.shape[1]}, {self.n_cores} cores)"

    @property
    def is_builtin(self):
        return isinstance(self.array, int)

    @property
    def n_cores(self):
        return sum(1 for i in self.cells.values() if not i.empty)

    def occupancy(self):
        """Boolean array of the map's shape, True where the map has a core."""
        grid = np.zeros(self.shape, dtype=bool)
        for (r, c), info in self.cells.items():
            grid[r, c] = not info.empty
        return grid

    def index_grid(self):
        """``(n_rows, n_cols, 2)`` array of each cell's own ``(row, col)``."""
        return np.stack(np.meshgrid(np.arange(self.shape[0]), np.arange(self.shape[1]), indexing="ij"),
                        axis=-1)

    def extra_columns(self):
        """Names of the extra map-file columns, in first-seen order."""
        names = []
        for info in self.cells.values():
            for k, _ in info.extras:
                if k not in names:
                    names.append(k)
        return names

    def digest(self):
        """Short fingerprint of the layout and identities; edits files record
        it so they are never applied to a different map."""
        if self._digest is None:
            rows = [[r, c, i.patient_id, i.block_id, i.tissue, i.note, i.empty]
                    for (r, c), i in sorted(self.cells.items())]
            blob = json.dumps([list(self.shape), rows], separators=(",", ":"))
            self._digest = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]
        return self._digest


@lru_cache(maxsize=None)
def builtin_map(array_number):
    """The built-in ICGC/APGI array ``array_number`` as a :class:`TMAMap`."""
    n = int(array_number)
    return TMAMap(f"ICGC Array {n}", load_array_map(n), (MAP_ROWS, MAP_COLS),
                  list(ROW_LETTERS), list(range(1, MAP_COLS + 1)), source=ARRAY_MAP_CSV.name, array=n)


def as_map(value):
    """``None``, a :class:`TMAMap`, or a built-in array number, as a map."""
    if value is None or isinstance(value, TMAMap):
        return value
    return builtin_map(int(value))


def match_map_to_slide(maps, slide_name):
    """The map whose name appears in ``slide_name`` (letters and digits only,
    ignoring case; the longest such name wins), or None."""
    norm = lambda t: re.sub(r"[^0-9a-z]", "", str(t).lower())  # noqa: E731
    slide = norm(slide_name)
    hits = [m for m in maps if norm(m.name) and norm(m.name) in slide]
    return max(hits, key=lambda m: len(norm(m.name))) if hits else None


# ---------------------------------------------------------------------------
# Map files
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MapProblem:
    """A problem found while reading a map file; errors make the file unusable."""
    level: str        # "error" or "warning"
    message: str

    def __str__(self):
        return f"{self.level.upper()}: {self.message}"


class MapFile:
    """Result of :func:`load_map_file`: the maps and the problems found."""

    def __init__(self, path, maps, problems):
        self.path = path
        self.maps = maps
        self.problems = problems

    @property
    def errors(self):
        return [p for p in self.problems if p.level == "error"]

    @property
    def warnings(self):
        return [p for p in self.problems if p.level == "warning"]

    @property
    def ok(self):
        return bool(self.maps) and not self.errors


def _letters(index):
    """0 -> A, 25 -> Z, 26 -> AA."""
    out = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        out = chr(65 + rem) + out
    return out


def _parse_label(text):
    """``("number", index)`` for 1, 2, ..., 200, ``("letter", index)`` for
    A, B, ..., ZZ; None otherwise (so patient IDs and tissue names are never
    mistaken for labels)."""
    t = str(text).strip()
    if t.isdigit() and 1 <= int(t) <= 200:
        return "number", int(t) - 1
    if t.isascii() and t.isalpha() and len(t) <= 2:
        idx = 0
        for ch in t.upper():
            idx = idx * 26 + (ord(ch) - 64)
        return "letter", idx - 1
    return None


def _read_table(path):
    """``[(line_number, cells)]`` of a delimited text file, ``line_number``
    being the line a row starts on. Rows whose first cell starts with ``#``
    are comments; the delimiter is a tab, semicolon or comma (detected); an
    Excel byte-order mark is dropped; cells are stripped. Quoted cells may
    span several lines (they are then reported by the parsers)."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        text = f.read()
    # a comment line starts with '#', possibly quoted by a spreadsheet program
    sample = [line for line in text.splitlines()
              if line.strip() and not line.lstrip().lstrip('"').startswith("#")][:50]
    if any("\t" in line for line in sample):
        delim = "\t"
    elif sum(line.count(";") for line in sample) > sum(line.count(",") for line in sample):
        delim = ";"
    else:
        delim = ","
    out = []
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delim)
    start = 1
    for row in reader:
        line, start = start, reader.line_num + 1
        cells = [c.strip() for c in row]
        if cells and cells[0].startswith("#"):
            continue
        out.append((line, cells))
    return out


def _multiline(text):
    return "\n" in text or "\r" in text


def _lines_of(text):
    return " / ".join(part.strip() for part in text.splitlines() if part.strip())


_LONG_COLUMNS = {"map": "map", "array": "map", "row": "row", "col": "col", "column": "col",
                 "patient_id": "patient_id", "patient": "patient_id", "case_id": "patient_id",
                 "tissue": "tissue", "control": "tissue", "class": "class", "note": "class",
                 "status": "status", "block_id": "block_id", "block": "block_id"}


def _is_blank(cells):
    return not any(cells)


def load_map_file(path):
    """Read a TMA map file (``.csv``, ``.tsv``, ``.txt`` or QuPath ``.qpmap``)
    in grid or list layout, detected from the header. Returns a
    :class:`MapFile` with the maps and any problems; a file with errors
    yields no maps."""
    default = os.path.splitext(os.path.basename(path))[0]
    problems = []
    try:
        rows = _read_table(path)
    except (OSError, UnicodeDecodeError, csv.Error) as e:
        return MapFile(path, [], [MapProblem("error", f"Cannot read {path}: {e}")])
    first = next((cells for _, cells in rows if not _is_blank(cells)), None)
    if first is None:
        return MapFile(path, [], [MapProblem("error", "The file holds no map.")])
    header = {_LONG_COLUMNS.get(c.strip().lower().replace(" ", "_")) for c in first}
    if {"row", "col"} <= header:
        entries = _parse_list(rows, default, problems)
    else:
        entries = _parse_grid(rows, default, problems)
    maps = []
    for name, (items, min_shape) in entries.items():
        m = _build_map(name, items, min_shape, path, problems)
        if m is not None:
            maps.append(m)
    if any(p.level == "error" for p in problems):
        maps = []
    return MapFile(path, maps, problems)


def _parse_list(rows, default, problems):
    """List layout: one line per core. Returns ``{map: (entries, (0, 0))}``."""
    it = iter(rows)
    for line, header in it:
        if not _is_blank(header):
            break
    keys = [_LONG_COLUMNS.get(h.strip().lower().replace(" ", "_")) for h in header]
    extras = [(i, h) for i, (h, k) in enumerate(zip(header, keys)) if k is None and h]
    out = {}
    for line, cells in it:
        if _is_blank(cells):
            continue
        multi = [c for c in cells if _multiline(c)]
        if multi:
            problems.append(MapProblem("error", f"Line {line}: a cell holds several lines ('{_lines_of(multi[0])}'); "
                                                f"put a single value in each cell."))
            continue
        get = lambda k: next((cells[i] for i, key in enumerate(keys) if key == k and i < len(cells)), "")  # noqa: E731
        name = get("map") or default
        status = get("status").lower() or "core"
        if status not in ("core", "empty"):
            problems.append(MapProblem("error", f"Line {line}: status must be empty or 'empty', not '{status}'."))
            continue
        out.setdefault(name, ([], (0, 0)))[0].append({
            "line": line, "row": get("row"), "col": get("col"), "patient_id": get("patient_id"),
            "tissue": get("tissue"), "class": get("class"), "block_id": get("block_id"),
            "empty": status == "empty",
            "extras": tuple((h, cells[i]) for i, h in extras if i < len(cells) and cells[i])})
    return out


CELL_LINES_KEY = "cell lines"
CELL_LINE_WORDS = ("position", "patient", "block", "class", "skip")
_POSITION_RE = re.compile(r"[A-Za-z]{1,2}\s*-?\s*\d{1,3}")


def _parse_cell_lines(cells, line, problems):
    """Words of a ``cell lines`` row (in its second cell, or spread over the
    following cells); None when invalid."""
    words = " ".join(cells[1:]).lower().split()
    bad = [w for w in words if w not in CELL_LINE_WORDS]
    if bad or "patient" not in words:
        problems.append(MapProblem("error", f"Line {line}: the cell lines row must name the lines of a pasted "
                                            f"cell in order using {', '.join(CELL_LINE_WORDS)} and include "
                                            f"patient" + (f"; '{bad[0]}' is not one of them." if bad else ".")))
        return None
    return words


def _pasted_cell(text, line, where, problems, spec):
    """Interpret a cell holding several lines with the map's ``cell lines``
    description: a printed position, the patient ID, the block ID, a class in
    brackets, or a control tissue (lines after the position without digits)."""
    parts = [p.strip() for p in text.splitlines() if p.strip()]
    entry = {"patient_id": "", "tissue": "", "class": "", "block_id": "", "empty": False, "extras": ()}
    order = [w for w in spec if w != "class"]
    if "class" in spec:
        bracketed = [p for p in parts if re.fullmatch(r"\(.*\)", p)]
        if bracketed:
            entry["class"] = bracketed[0][1:-1].strip()
            parts = [p for p in parts if p not in bracketed]
    if order and order[0] == "position":
        if not parts or not _POSITION_RE.fullmatch(parts[0]):
            problems.append(MapProblem("error", f"Line {line}, {where}: the first line '{parts[0] if parts else ''}' "
                                                f"of '{_lines_of(text)}' is not a position such as A1; check "
                                                f"the cell lines row."))
            return None
        parts, order = parts[1:], order[1:]
    if not parts:
        return None                                    # only a printed position: no core here
    joined = " ".join(parts)
    if joined == UNKNOWN_TOKEN:
        return entry
    if joined.lower().startswith(CONTROL_PREFIX) or not any(ch.isdigit() for ch in joined):
        tissue = joined[len(CONTROL_PREFIX):].strip() if joined.lower().startswith(CONTROL_PREFIX) else joined
        entry["tissue"] = tissue
        return entry
    values = {}
    for word, value in zip(order, parts):
        if word != "skip":
            values.setdefault(word, value)
    if len(parts) > len(order):
        problems.append(MapProblem("warning", f"Line {line}, {where}: '{_lines_of(text)}' has more lines than "
                                              f"the cell lines row describes; the extra lines are ignored."))
    if not values.get("patient"):
        problems.append(MapProblem("error", f"Line {line}, {where}: no patient ID in '{_lines_of(text)}' "
                                            f"according to the cell lines row."))
        return None
    entry["patient_id"] = values["patient"]
    entry["block_id"] = values.get("block", "")
    return entry


def _grid_cell(text, line, where, problems, spec=None):
    """Interpret one grid cell; None for an empty position. ``spec`` is the
    map's ``cell lines`` description, used for cells holding several lines."""
    if not text:
        return None
    if _multiline(text):
        if spec:
            return _pasted_cell(text, line, where, problems, spec)
        problems.append(MapProblem("error", f"Line {line}, {where}: the cell holds several lines "
                                            f"('{_lines_of(text)}'). Put a single value in each cell (the "
                                            f"patient ID, {CONTROL_PREFIX}<tissue>, {UNKNOWN_TOKEN} or nothing), "
                                            f"or describe the lines with a '{CELL_LINES_KEY}' row."))
        return None
    if spec and spec[0] == "position" and _POSITION_RE.fullmatch(text):
        problems.append(MapProblem("warning", f"Line {line}, {where}: '{text}' looks like a printed position; "
                                              f"it is read as a patient ID. Clear the cell if there is no core."))
    entry = {"patient_id": "", "tissue": "", "class": "", "block_id": "", "empty": False, "extras": ()}
    if text == UNKNOWN_TOKEN:
        return entry
    if text.lower().startswith(CONTROL_PREFIX):
        tissue = text[len(CONTROL_PREFIX):].strip()
        if not tissue:
            problems.append(MapProblem("error", f"Line {line}, {where}: '{text}' names no control tissue."))
            return None
        entry["tissue"] = tissue
        return entry
    if re.match(r"^[A-Za-z]+\s*:", text):
        problems.append(MapProblem("warning", f"Line {line}, {where}: '{text}' looks like a prefix other "
                                              f"than '{CONTROL_PREFIX}'; it is read as a patient ID."))
    entry["patient_id"] = text
    return entry


def _parse_grid(rows, default, problems):
    """Grid layout: the map drawn as a table. Returns ``{map: (entries, min_shape)}``."""
    blocks, current, file_spec = [], None, None
    titled = any(cells and cells[0].lower() == "map" and len(cells) > 1 and cells[1]
                 and not any(cells[2:]) for _, cells in rows)
    for line, cells in rows:
        is_title = cells and cells[0].lower() == "map" and len(cells) > 1 and cells[1] and not any(cells[2:])
        if cells and cells[0].strip().lower() == CELL_LINES_KEY:
            spec = _parse_cell_lines(cells, line, problems)
            if current is None:
                file_spec = spec                       # before any map: applies to every map
            elif any(not _is_blank(c) for _, c in current[1]):
                problems.append(MapProblem("error", f"Line {line}: the cell lines row must come before the "
                                                    f"row of column numbers."))
            else:
                current[2][0] = spec
            continue
        if is_title:
            current = (cells[1], [], [None])
            blocks.append(current)
        elif titled and current is None:
            if not _is_blank(cells):
                problems.append(MapProblem("error", f"Line {line}: grid rows before the first 'map,<name>' line."))
        else:
            if current is None:
                current = (default, [], [None])
                blocks.append(current)
            current[1].append((line, cells))
    out = {}
    for name, block, spec in blocks:
        while block and _is_blank(block[0][1]):
            block.pop(0)
        while block and _is_blank(block[-1][1]):
            block.pop()
        if name in out:
            problems.append(MapProblem("error", f"Map '{name}' appears twice in the file."))
            continue
        if not block:
            problems.append(MapProblem("error", f"Map '{name}' has no grid."))
            continue
        out[name] = _grid_entries(name, block, problems, spec[0] or file_spec)
    return out


def _grid_entries(name, block, problems, spec=None):
    head_line, head = block[0]
    body = block[1:]
    col_labels = [h for h in head[1:]]
    has_header = (bool(head) and head[0] == "" and any(col_labels)
                  and all(_parse_label(h) for h in col_labels if h)
                  and len({_parse_label(h)[0] for h in col_labels if h}) == 1)
    entries = []
    if has_header:
        # with column labels, every row needs its row label
        labelled = []
        for line, cells in body:
            if _is_blank(cells):
                continue
            if not _parse_label(cells[0]):
                shown = f"'{_lines_of(cells[0])}'" if cells[0] else "nothing"
                problems.append(MapProblem("error", f"Line {line}: map '{name}' has column labels, but this row's "
                                                    f"first cell holds {shown} instead of a row label; give every "
                                                    f"row its letter (A, B, C, ...)."))
                continue
            labelled.append((line, cells))
        body = labelled
        n_header = max(i for i, h in enumerate(col_labels) if h) + 1
        for line, cells in body:
            if _is_blank(cells):
                continue
            values = cells[1:]
            if any(values[n_header:]):
                problems.append(MapProblem("error", f"Line {line}: map '{name}' has more cells than column labels "
                                                    f"({n_header}); add the missing column numbers to the header row."))
                continue
            for k, text in enumerate(values[:n_header]):
                if text and not col_labels[k]:
                    problems.append(MapProblem("error", f"Line {line}: a cell under an empty column label."))
                    continue
                e = _grid_cell(text, line, f"row {cells[0]}, column {col_labels[k]}", problems, spec)
                if e is not None:
                    e.update(line=line, row=cells[0], col=col_labels[k])
                    entries.append(e)
        rows_idx = [_parse_label(cells[0])[1] for _, cells in body if not _is_blank(cells)]
        cols_idx = [_parse_label(h)[1] for h in col_labels if h]
        min_shape = (max(rows_idx, default=-1) + 1, max(cols_idx, default=-1) + 1)
        # labels define the grid even where no core is listed
        for line, cells in body:
            if not _is_blank(cells):
                entries.append({"line": line, "row": cells[0], "col": next(h for h in col_labels if h),
                                "marker": True})
        for h in col_labels:
            if h:
                entries.append({"line": head_line, "row": body[0][1][0] if body else "A", "col": h,
                                "marker": True})
    else:
        n_cols = max(len(cells) for _, cells in block)
        for r, (line, cells) in enumerate(block):
            for k, text in enumerate(cells):
                e = _grid_cell(text, line, f"row {r + 1}, column {k + 1}", problems, spec)
                if e is not None:
                    e.update(line=line, row=_letters(r), col=str(k + 1))
                    entries.append(e)
        min_shape = (len(block), n_cols)
    return entries, min_shape


def _build_map(name, items, min_shape, path, problems):
    """Assemble one :class:`TMAMap` from parsed entries; None on errors."""
    n_errors = sum(p.level == "error" for p in problems)
    kinds = {"row": set(), "col": set()}
    parsed = []
    for e in items:
        rl, cl = _parse_label(e["row"]), _parse_label(e["col"])
        if rl is None or cl is None:
            bad = e["row"] if rl is None else e["col"]
            problems.append(MapProblem("error", f"Line {e['line']}: '{bad}' is not a row or column label "
                                                f"(use letters A, B, ... or numbers 1, 2, ...)."))
            continue
        kinds["row"].add(rl[0])
        kinds["col"].add(cl[0])
        parsed.append((e, rl[1], cl[1]))
    for axis in ("row", "col"):
        if len(kinds[axis]) > 1:
            problems.append(MapProblem("error", f"Map '{name}' mixes letters and numbers as {axis} labels."))
    if sum(p.level == "error" for p in problems) > n_errors:
        return None
    if not parsed:
        problems.append(MapProblem("error", f"Map '{name}' lists no positions."))
        return None
    row_kind = next(iter(kinds["row"]))
    col_kind = next(iter(kinds["col"]))
    n_rows = max(max(r for _, r, _ in parsed) + 1, min_shape[0])
    n_cols = max(max(c for _, _, c in parsed) + 1, min_shape[1])
    row_labels = [_letters(i) if row_kind == "letter" else str(i + 1) for i in range(n_rows)]
    col_labels = [i + 1 if col_kind == "number" else _letters(i) for i in range(n_cols)]
    cells, seen = {}, {}
    for e, r, c in parsed:
        if e.get("marker"):
            continue
        if (r, c) in seen:
            problems.append(MapProblem("error", f"Line {e['line']}: map '{name}' lists position "
                                                f"{position_label(row_labels[r], col_labels[c])} twice (also line {seen[(r, c)]})."))
            continue
        seen[(r, c)] = e["line"]
        pid, tissue = e["patient_id"], e["tissue"]
        if pid and tissue:
            problems.append(MapProblem("warning", f"Line {e['line']}: both a patient ID and a tissue; "
                                                  f"the core is read as patient {pid}."))
        cells[(r, c)] = CoreInfo(name, 0, position_label(row_labels[r], col_labels[c]), row_labels[r],
                                 col_labels[c], pid, e["block_id"], tissue, e["class"], e["empty"], r, c,
                                 e["extras"])
    if sum(p.level == "error" for p in problems) > n_errors:
        return None
    m = TMAMap(name, cells, (n_rows, n_cols), row_labels, col_labels, source=os.path.basename(path))
    if m.n_cores == 0:
        if not any(p.level == "error" for p in problems):    # otherwise the errors already say why
            problems.append(MapProblem("error", f"Map '{name}' has no cores."))
        return None
    if m.n_cores < 6:
        problems.append(MapProblem("warning", f"Map '{name}' has only {m.n_cores} cores; the orientation "
                                              f"cannot be matched reliably. Check it by hand."))
    counts = {}
    for i in m.cells.values():
        if i.patient_id:
            counts[i.patient_id] = counts.get(i.patient_id, 0) + 1
    for pid, n in sorted(counts.items()):
        if n > 3:
            problems.append(MapProblem("warning", f"Map '{name}': patient {pid} appears {n} times; "
                                                  f"check for a typing error."))
    return m


def describe_map(m):
    """Plain-text drawing of a map for checking against its source:
    patient IDs, ``ctrl:<tissue>``, ``?`` for unknown and ``.`` for empty."""
    def text(i):
        if i.empty:
            return "."
        if i.patient_id:
            return i.patient_id
        if i.tissue:
            return CONTROL_PREFIX + i.tissue
        return UNKNOWN_TOKEN
    table = [[""] + [str(c) for c in m.col_labels]]
    for r in range(m.shape[0]):
        table.append([m.row_labels[r]] + [text(m.cells[(r, c)]) for c in range(m.shape[1])])
    widths = [max(len(row[k]) for row in table) for k in range(len(table[0]))]
    lines = [f"Map '{m.name}': {m.shape[0]} rows x {m.shape[1]} columns, {m.n_cores} cores "
             f"({sum(i.is_control for i in m.cells.values())} controls, "
             f"{sum(i.is_unknown for i in m.cells.values())} unknown)"]
    lines += ["  ".join(v.rjust(w) for v, w in zip(row, widths)) for row in table]
    return "\n".join(lines)


def write_map_list(maps, path):
    """Write maps in the list layout (one line per core). Each map's
    bottom-right position is written with status ``empty`` when it has no
    core, so the map keeps its size when read back."""
    extra = []
    for m in maps:
        for k in m.extra_columns():
            if k not in extra:
                extra.append(k)
    fields = ["map", "row", "col", "patient_id", "tissue", "class", "status", "block_id"] + extra
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for m in maps:
            last = (m.shape[0] - 1, m.shape[1] - 1)
            for (r, c), i in sorted(m.cells.items()):
                if i.empty and (r, c) != last:
                    continue
                ex = dict(i.extras)
                w.writerow([m.name, i.row, i.col, i.patient_id, i.tissue, i.note,
                            "empty" if i.empty else "", i.block_id] + [ex.get(k, "") for k in extra])


GRID_TEMPLATE = """\
# CABANA TMA map. Lines starting with # are ignored. Replace the example map below with yours.
# One cell is one position on the array where it physically sits (not where a printed label says it is).
# A cell holds the patient ID / ctrl:<tissue> for a control such as ctrl:Liver / ? for a core whose ID is unknown / nothing when there is no core.
# The row letters and column numbers are ready for up to 12 x 12 positions. Add more if needed. Unused rows and columns do no harm.
# Pasting from a PDF or another table is fine: a pasted cell may hold several lines.
# The cell lines row names those lines in order using position / patient / block / class / skip.
# A line in brackets such as (PNET) is the class. A pasted cell with no digits after its position is a control.
# Change the cell lines row to match your pasted cells. Cells holding a single value are read as described above.
# For several maps in one file start each with a row holding map in the first cell and the map name in the second.
map,TMA1
cell lines,position patient block class
,1,2,3,4,5,6,7,8,9,10,11,12
A,ctrl:Liver,1001,1002,1003,1004,ctrl:Brain,,,,,,
B,1005,1006,?,1007,1008,1009,,,,,,
C,"C1
Lung","C2
1010
801010","C3
1011
801011
(PNET)",1012,,,,,,,,
D,1001,1002,1003,1004,1005,1006,,,,,,
E,,,,,,,,,,,,
F,,,,,,,,,,,,
G,,,,,,,,,,,,
H,,,,,,,,,,,,
I,,,,,,,,,,,,
J,,,,,,,,,,,,
K,,,,,,,,,,,,
L,,,,,,,,,,,,
"""


TEMPLATE_FILE = "tma_map_template.csv"


def write_templates(folder):
    """Write the map template (grid layout) into ``folder``; returns ``[path]``."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, TEMPLATE_FILE)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(GRID_TEMPLATE)
    return [path]
