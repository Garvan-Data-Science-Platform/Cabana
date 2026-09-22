"""Whole-slide image readers used by the TMA preprocessing step.

Two readers share one small interface:

* :class:`VsiReader` reads Olympus VS200 ``.vsi`` slides natively by parsing
  the companion ``_<name>_/stack*/frame_t.ets`` tile pyramids. No Java or
  Bio-Formats is required.
* :class:`FlatReader` wraps an ordinary TIFF/PNG/JPEG whole-slide export and
  builds a pyramid on demand.

Coordinates are always given in level-0 pixels. ``read_region`` returns BGR
``uint8`` arrays (OpenCV convention) so the rest of CABANA can consume them
unchanged.
"""

import os
import re
import struct
from collections import OrderedDict, defaultdict
from glob import glob

import cv2
import numpy as np

_PIXEL_TYPES = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
                5: np.int32, 6: np.uint32, 9: np.float32, 10: np.float64}
_COMPRESSION = {0: "raw", 2: "jpeg", 3: "jpeg2000", 5: "jpeg-lossless", 8: "png", 9: "bmp"}


def open_slide(path, pixel_size_um=None):
    """Return a reader for ``path`` (``.vsi`` or a flat image)."""
    if path.lower().endswith(".vsi"):
        return VsiReader(path, pixel_size_um=pixel_size_um)
    return FlatReader(path, pixel_size_um=pixel_size_um)


class SlideReader:
    """Common interface. Subclasses fill ``channels``, ``pixel_size_um``,
    ``_level_shapes`` (list of (h, w) at each level) and ``_read``."""

    channels = ()
    pixel_size_um = None
    _level_shapes = ()

    @property
    def level_count(self):
        return len(self._level_shapes)

    def level_shape(self, level):
        return self._level_shapes[level]

    def level_downsample(self, level):
        h0, w0 = self._level_shapes[0]
        h, w = self._level_shapes[level]
        return 0.5 * (h0 / h + w0 / w)

    def best_level_for_pixel_size(self, target_um):
        """Finest level whose pixel size is at least ``target_um``."""
        if not self.pixel_size_um:
            return self.level_count - 1
        best = 0
        for lv in range(self.level_count):
            if self.pixel_size_um * self.level_downsample(lv) <= target_um:
                best = lv
        return best

    def read_level(self, level, channel=0):
        h, w = self._level_shapes[level]
        return self.read_region(0, 0, w, h, channel=channel, level=level)

    def read_region(self, x, y, w, h, channel=0, level=0):
        """Read a ``w`` x ``h`` window whose top-left is ``(x, y)`` in the pixel
        coordinates of ``level``. Windows may extend past the slide; the
        overhang is filled with the channel background."""
        raise NotImplementedError

    def channel_index(self, channel):
        if isinstance(channel, str):
            return list(self.channels).index(channel)
        return int(channel)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------
# Olympus VSI / ETS
# ---------------------------------------------------------------------------

class EtsStack:
    """One ``frame_t.ets`` tile pyramid."""

    def __init__(self, path):
        self.path = path
        self._fh = open(path, "rb")
        head = self._fh.read(0x40)
        if head[:3] != b"SIS":
            raise ValueError(f"{path} is not an ETS file")
        _, _, ndim = struct.unpack("<iii", head[4:16])
        add_off, = struct.unpack("<q", head[16:24])
        add_size, = struct.unpack("<i", head[24:28])
        chunk_off, = struct.unpack("<q", head[32:40])
        n_chunks, = struct.unpack("<i", head[40:44])
        self._fh.seek(add_off)
        add = self._fh.read(add_size)
        if add[:3] != b"ETS":
            raise ValueError(f"{path}: missing ETS header")
        (_, ptype, self.size_c, self.colorspace, comp, _,
         self.tile_w, self.tile_h, _) = struct.unpack("<9i", add[4:40])
        self.dtype = _PIXEL_TYPES.get(ptype, np.uint8)
        self.compression = _COMPRESSION.get(comp, str(comp))
        self.ndim = ndim

        self._fh.seek(chunk_off)
        rec = struct.Struct(f"<i{ndim}iqii")
        raw = self._fh.read(rec.size * n_chunks)
        self.tiles = defaultdict(dict)   # level -> {(tx, ty): (offset, size)}
        for i in range(n_chunks):
            vals = rec.unpack_from(raw, i * rec.size)
            coords = vals[1:1 + ndim]
            off, size = vals[1 + ndim], vals[2 + ndim]
            if ndim > 3 and coords[2] != 0:
                continue   # only z/c plane 0 is used
            self.tiles[coords[-1]][(coords[0], coords[1])] = (off, size)
        self.levels = sorted(self.tiles)
        self.grid = {lv: (max(t[0] for t in self.tiles[lv]) + 1,
                          max(t[1] for t in self.tiles[lv]) + 1) for lv in self.levels}
        self._cache = OrderedDict()

    def close(self):
        self._fh.close()

    def level_extent(self, level):
        """(height, width) covered by the tile grid at ``level``."""
        nx, ny = self.grid[level]
        return ny * self.tile_h, nx * self.tile_w

    def read_tile(self, level, tx, ty, fill):
        key = (level, tx, ty)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        entry = self.tiles[level].get((tx, ty))
        if entry is None:
            tile = np.full((self.tile_h, self.tile_w, 3), fill, dtype=np.uint8)
        else:
            off, size = entry
            self._fh.seek(off)
            buf = np.frombuffer(self._fh.read(size), dtype=np.uint8)
            if self.compression == "raw":
                tile = buf.reshape(self.tile_h, self.tile_w, -1)
                if tile.shape[2] == 1:
                    tile = np.repeat(tile, 3, axis=2)
            else:
                tile = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if tile is None:
                    tile = np.full((self.tile_h, self.tile_w, 3), fill, dtype=np.uint8)
            if tile.shape[:2] != (self.tile_h, self.tile_w):
                padded = np.full((self.tile_h, self.tile_w, 3), fill, dtype=np.uint8)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded
        self._cache[key] = tile
        if len(self._cache) > 256:
            self._cache.popitem(last=False)
        return tile


class VsiReader(SlideReader):
    """Reader for Olympus VS200 ``.vsi`` slides.

    The full-resolution image layers are the ETS stacks that share the largest
    tile grid. Channel names are assigned by brightness: a bright layer is
    brightfield (``BF``), a dark layer is polarised light (``POL``).
    """

    def __init__(self, path, pixel_size_um=None):
        self.path = path
        folder = os.path.join(os.path.dirname(path),
                              "_" + os.path.splitext(os.path.basename(path))[0] + "_")
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Companion folder not found: {folder}")
        stacks = []
        for ets in sorted(glob(os.path.join(folder, "stack*", "frame_t.ets"))):
            try:
                stacks.append(EtsStack(ets))
            except (ValueError, struct.error):
                continue
        if not stacks:
            raise FileNotFoundError(f"No tile pyramids under {folder}")
        largest = max(np.prod(s.level_extent(0)) for s in stacks)
        self._stacks = [s for s in stacks if np.prod(s.level_extent(0)) >= 0.5 * largest]
        self._overview = min(stacks, key=lambda s: np.prod(s.level_extent(0)))
        for s in stacks:
            if s not in self._stacks:
                s.close()
        self._stacks.sort(key=lambda s: s.path)

        self._image_shape = self._read_image_shape()
        base = self._stacks[0]
        self._level_shapes = []
        for lv in base.levels:
            ds = 2 ** lv
            self._level_shapes.append((int(np.ceil(self._image_shape[0] / ds)),
                                       int(np.ceil(self._image_shape[1] / ds))))

        # Classify each layer as bright-field or polarised by the mean of a
        # coarse level restricted to the image extent (the padded part of the
        # coarsest JPEG tile is not representative).
        probe_level = next((lv for lv in range(self.level_count)
                            if max(self._level_shapes[lv]) <= 1024), self.level_count - 1)
        self._fill = [255] * len(self._stacks)
        names = []
        for ci, s in enumerate(self._stacks):
            thumb = self.read_region(0, 0, self._level_shapes[probe_level][1],
                                     self._level_shapes[probe_level][0], channel=ci, level=probe_level)
            bright = float(thumb.mean()) > 100
            self._fill[ci] = 255 if bright else 0
            names.append("BF" if bright else "POL")
        if len(set(names)) != len(names):
            names = [f"{n}{i}" for i, n in enumerate(names)]
        self.channels = tuple(names)
        self.slide_name = self._read_slide_name()
        self.pixel_size_um = pixel_size_um or self._guess_pixel_size()

    # -- metadata -----------------------------------------------------------
    def _read_vsi_bytes(self):
        with open(self.path, "rb") as f:
            return f.read()

    def _read_image_shape(self):
        """Full-resolution (h, w). Prefer the Exif PixelX/YDimension written by
        VS200; fall back to the tile grid extent."""
        try:
            import tifffile
            with tifffile.TiffFile(self.path) as t:
                exif = t.pages[0].tags.get("ExifTag")
                if exif is not None:
                    w = exif.value.get("PixelXDimension")
                    h = exif.value.get("PixelYDimension")
                    ext_h, ext_w = self._stacks[0].level_extent(0)
                    if w and h and ext_w - self._stacks[0].tile_w < w <= ext_w \
                            and ext_h - self._stacks[0].tile_h < h <= ext_h:
                        return int(h), int(w)
        except Exception:
            pass
        return self._stacks[0].level_extent(0)

    def _read_slide_name(self):
        data = self._read_vsi_bytes()
        stem = os.path.splitext(os.path.basename(self.path))[0]
        for m in re.finditer(rb"(?:[\x20-\x7e]\x00){3,80}", data):
            s = m.group().decode("utf-16le")
            if s == stem:
                return s
        return stem

    def _guess_pixel_size(self):
        """Look for the calibration pair (two nearly equal doubles) in the
        ``.vsi`` tag stream. Returns the smallest plausible value in microns,
        which belongs to the highest-magnification layer, or ``None``."""
        data = self._read_vsi_bytes()
        cands = []
        for shift in range(0, 8):
            view = np.frombuffer(data[shift:shift + ((len(data) - shift) // 8) * 8], dtype="<f8")
            a, b = view[:-1], view[1:]
            with np.errstate(all="ignore"):
                ok = (a > 0.02) & (a < 50) & (b > 0.02) & (b < 50)
                ok &= np.abs(a - b) < 1e-3 * a
            cands.extend(a[ok].tolist())
        if not cands:
            return None
        return float(min(cands))

    # -- pixel access -------------------------------------------------------
    def read_region(self, x, y, w, h, channel=0, level=0):
        ci = self.channel_index(channel)
        stack = self._stacks[ci]
        fill = self._fill[ci]
        out = np.full((h, w, 3), fill, dtype=np.uint8)
        tw, th = stack.tile_w, stack.tile_h
        lv_h, lv_w = self._level_shapes[level]
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, lv_w), min(y + h, lv_h)
        if x1 <= x0 or y1 <= y0:
            return out
        for ty in range(y0 // th, (y1 - 1) // th + 1):
            for tx in range(x0 // tw, (x1 - 1) // tw + 1):
                tile = stack.read_tile(stack.levels[level], tx, ty, fill)
                sx0, sy0 = max(x0, tx * tw), max(y0, ty * th)
                sx1, sy1 = min(x1, (tx + 1) * tw), min(y1, (ty + 1) * th)
                out[sy0 - y:sy1 - y, sx0 - x:sx1 - x] = \
                    tile[sy0 - ty * th:sy1 - ty * th, sx0 - tx * tw:sx1 - tx * tw]
        return out

    def read_overview(self):
        """Low-resolution slide overview (label + macro layer), BGR."""
        s = self._overview
        lv = s.levels[0]
        h, w = s.level_extent(lv)
        out = np.full((h, w, 3), 255, dtype=np.uint8)
        for (tx, ty) in s.tiles[lv]:
            out[ty * s.tile_h:(ty + 1) * s.tile_h, tx * s.tile_w:(tx + 1) * s.tile_w] = \
                s.read_tile(lv, tx, ty, 255)
        return out

    def close(self):
        for s in self._stacks:
            s.close()
        self._overview.close()


# ---------------------------------------------------------------------------
# Flat images
# ---------------------------------------------------------------------------

class FlatReader(SlideReader):
    """Reader for a whole-slide image stored as one plain raster file."""

    def __init__(self, path, pixel_size_um=None, min_level_px=256):
        self.path = path
        self.slide_name = os.path.splitext(os.path.basename(path))[0]
        img = self._load(path)
        self._levels = [img]
        while min(self._levels[-1].shape[:2]) > 2 * min_level_px:
            self._levels.append(cv2.resize(self._levels[-1], None, fx=0.5, fy=0.5,
                                           interpolation=cv2.INTER_AREA))
        self._level_shapes = [im.shape[:2] for im in self._levels]
        self.channels = ("BF",) if float(img.mean()) > 100 else ("POL",)
        self._fill = 255 if self.channels[0] == "BF" else 0
        self.pixel_size_um = pixel_size_um or self._read_pixel_size(path)

    @staticmethod
    def _load(path):
        if path.lower().endswith((".tif", ".tiff")):
            import tifffile
            img = tifffile.imread(path)
            if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[0] < img.shape[-1]:
                img = np.moveaxis(img, 0, -1)
            if img.ndim == 3:
                img = img[..., :3][..., ::-1]
        else:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        if img.dtype != np.uint8:
            img = cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return np.ascontiguousarray(img)

    @staticmethod
    def _read_pixel_size(path):
        if not path.lower().endswith((".tif", ".tiff")):
            return None
        try:
            import tifffile
            with tifffile.TiffFile(path) as t:
                page = t.pages[0]
                unit = page.tags.get("ResolutionUnit")
                xres = page.tags.get("XResolution")
                if unit is None or xres is None:
                    return None
                num, den = xres.value
                if not num:
                    return None
                per_unit = num / den
                unit_um = {2: 25400.0, 3: 10000.0}.get(int(unit.value))
                if unit_um is None:
                    return None
                px = unit_um / per_unit
                return px if 0.02 < px < 50 else None
        except Exception:
            return None

    def read_region(self, x, y, w, h, channel=0, level=0):
        img = self._levels[level]
        out = np.full((h, w, 3), self._fill, dtype=np.uint8)
        lv_h, lv_w = img.shape[:2]
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, lv_w), min(y + h, lv_h)
        if x1 > x0 and y1 > y0:
            out[y0 - y:y1 - y, x0 - x:x1 - x] = img[y0:y1, x0:x1]
        return out

    def read_overview(self):
        return self._levels[-1].copy()
