"""I/O helpers: paths, folders, image-path globs, batch splitting, parameter
file conversion. Extracted from the monolithic ``cabana.utils`` for
readability. ``cabana.utils`` re-exports the public names below so existing
``from .utils import join_path`` style imports continue to work.
"""

import os
import re
import shutil
from glob import glob
from PIL import Image, ExifTags

from .log import Log


read_bar_format = "%s{l_bar}%s{bar}%s{r_bar}" % (
    "\033[0;34m", "\033[0;34m", "\033[0;34m")


def join_path(*args):
    return os.path.join(*args).replace("\\", "/")


def create_folder(folder, overwrite=True):
    if os.path.exists(folder):
        if overwrite:
            shutil.rmtree(folder)
            os.mkdir(folder)
    else:
        os.makedirs(folder)


def get_img_paths(folder,
                  image_types=('*.[Tt][Ii][Ff]*', '*.[Pp][Nn][Gg]',
                               '*.[Jj][Pp][Gg]', '*.[Jj][Pp][Ee][Gg]')):
    img_paths = []
    for image_type in image_types:
        img_paths.extend(glob(join_path(folder, image_type)))
    return img_paths


def sanitize_filename(filename):
    forbidden_chars = r"[ ,:?\/*]"
    return re.sub(forbidden_chars, '_', filename)


# Default of ``Configs: Max Size``: images with more pixels than its square
# are split into blocks. It covers TMA cores scanned at 20x (4000 to 5000 px).
DEFAULT_MAX_SIZE = 5000


def max_image_size(params):
    """``Max Size`` of a parameter dict: ``Configs: Max Size``, else the
    pre-0.3.9 location ``Segmentation: Max Size``, else :data:`DEFAULT_MAX_SIZE`.
    0 means no limit."""
    for section in ("Configs", "Segmentation"):
        value = (params.get(section) or {}).get("Max Size")
        if value is not None:
            return int(value or 0)
    return DEFAULT_MAX_SIZE


def save_visualisations(params):
    """``Configs: Save Visualisations`` of a parameter dict (default True):
    whether batch runs write the colour maps and gap overlay images. The
    measurement data (TIFF maps, masks, CSV) is always written."""
    return bool((params.get("Configs") or {}).get("Save Visualisations", True))


def migrate_max_size(params):
    """Move a pre-0.3.9 ``Segmentation: Max Size`` to ``Configs: Max Size``
    (in place) so that files are written in the current layout."""
    seg = params.get("Segmentation") or {}
    if "Max Size" in seg:
        value = seg.pop("Max Size")
        params.setdefault("Configs", {}).setdefault("Max Size", value)
    return params


def is_oversized(height, width, max_size):
    """Whether an image has more than ``max_size`` squared pixels (the
    ``Configs: Max Size`` parameter). 0 or None means no limit."""
    return bool(max_size) and height * width > int(max_size) ** 2


def contains_oversized(img_paths, max_res=2048):
    for img_path in img_paths:
        with Image.open(img_path) as image:
            w, h = image.size
        if is_oversized(h, w, max_res):
            return True
    return False


def largest_image(img_paths):
    """``(path, height, width)`` of the image with the most pixels, or None."""
    best = None
    for img_path in img_paths:
        with Image.open(img_path) as image:
            w, h = image.size
        if best is None or h * w > best[1] * best[2]:
            best = (img_path, h, w)
    return best


# Micrometres per unit, for the unit an ImageJ description names
# (``unit=µm`` in the TIFF ImageDescription; QuPath exports write it too).
# Inch is left out: ImageJ sets it from an image's DPI, which is not a pixel size.
_IMAGEJ_UNITS_UM = {'µm': 1.0, 'um': 1.0, 'micron': 1.0, 'microns': 1.0, 'nm': 1e-3,
                    'mm': 1e3, 'cm': 1e4, 'm': 1e6}
_TIFF_RESOLUTION_UNIT = 296
_TIFF_IMAGE_DESCRIPTION = 270


def pixel_size_um(img_path):
    """``(x, y)`` pixel size in µm from an image's resolution metadata, or None.

    The TIFF resolution is pixels per unit. The unit is the one an ImageJ
    description names (``unit=µm``); without one, ResolutionUnit 3 is a
    centimetre and "none" is taken as a micrometre (as before). Inch-based
    resolutions (DPI) are written by image editors as a default and say
    nothing about the specimen, so they count as no pixel size, as does an
    ImageJ ``unit=pixel``.
    """
    with Image.open(img_path) as img:
        exif = img.getexif()
    xres = exif.get(ExifTags.Base.XResolution)
    yres = exif.get(ExifTags.Base.YResolution, xres)
    if not xres or not yres or float(xres) <= 0 or float(yres) <= 0:
        return None
    desc = exif.get(_TIFF_IMAGE_DESCRIPTION) or ''
    match = re.search(r'^unit=(.+)$', desc, re.M) if desc.startswith('ImageJ') else None
    if match:
        unit = match.group(1).strip().replace('\\u00B5', 'µ').replace('\\u00b5', 'µ').lower()
        um_per_unit = _IMAGEJ_UNITS_UM.get(unit)
    else:
        # ResolutionUnit: 1 none (taken as µm), 2 inch (DPI: no pixel size), 3 cm
        um_per_unit = {2: None, 3: 1e4}.get(exif.get(_TIFF_RESOLUTION_UNIT, 1), 1.0)
    if um_per_unit is None:
        return None
    return um_per_unit / float(xres), um_per_unit / float(yres)


def split2batches(img_paths, max_batch_size=5):
    pixel_res = []
    no_res, mismatched = 0, 0
    for img_path in img_paths:
        size = pixel_size_um(img_path)
        if size is None:
            no_res += 1
            pixel_res.append(1.0)
            continue
        xres, yres = size
        if abs(xres - yres) > 1e-3 * xres:
            mismatched += 1
        # 4 significant figures: exact enough to measure with (< 0.05 %),
        # coarse enough that one slide's images share a batch
        pixel_res.append(float(f'{xres:.4g}'))
    assert len(pixel_res) == len(img_paths)
    if mismatched:
        Log.logger.warning(f'{mismatched} of {len(img_paths)} images have different X and Y resolutions '
                           f'in their metadata; using XResolution.')
    if no_res:
        Log.logger.warning(f'{no_res} of {len(img_paths)} images have no usable pixel size in their metadata '
                           f'(missing, or in DPI); using 1.0 µm/pixel.')

    img_paths = [x for _, x in sorted(zip(pixel_res, img_paths))]
    pixel_res = [y for y, _ in sorted(zip(pixel_res, img_paths))]
    path_batches = []
    res_batches = []

    pres_value = pixel_res[0]
    path_batch = [img_paths[0]]

    for res, img_path in zip(pixel_res[1:], img_paths[1:]):
        if pres_value == res:
            if len(path_batch) < max_batch_size:
                path_batch.append(img_path)
            else:
                path_batches.append(path_batch)
                res_batches.append(pres_value)
                path_batch = [img_path]
        else:
            path_batches.append(path_batch)
            res_batches.append(pres_value)
            path_batch = [img_path]
            pres_value = res

    if len(path_batch) > 0:
        path_batches.append(path_batch)
        res_batches.append(pres_value)

    return path_batches, res_batches


def export_parameters(param_path, out_file):
    if not os.path.exists(param_path):
        Log.logger.error("{} not exists.".format(param_path))
        return
    with open(out_file, 'a+') as hf:
        if os.path.basename(param_path).endswith('.txt'):
            str_header = f"\n******{os.path.basename(param_path)}******\n"
            hf.write(str_header)
            with open(param_path) as f:
                for line in f:
                    key, _, value = line.rstrip().partition(",")
                    hf.write(f"{key}:   {value}\n")
            str_footer = '*' * ((len(str_header) - 3) // 2) + "End" + '*' * ((len(str_header) - 3) // 2) + "\n"
            hf.write(str_footer)


def convert_parameters(param_file_in_micros, param_file_in_pixels, ims_res):
    with open(param_file_in_micros, 'r') as rf, open(param_file_in_pixels, 'w+') as wf:
        for line in rf:
            key, _, value = line.rstrip().partition(",")
            kl = key.lower()
            if kl.startswith("dark line") or kl.startswith("contrast saturation") \
                    or kl.startswith("low contrast") or kl.startswith("high contrast") \
                    or kl.startswith("maximum display hdm"):
                wf.write(line)
            elif kl.startswith("min line width"):
                wf.write("Min Line Width,{:d}\n".format(int(float(value) / ims_res)))
            elif kl.startswith("max line width"):
                wf.write("Max Line Width,{:d}\n".format(int(float(value) / ims_res)))
            elif kl.startswith("line width step"):
                wf.write("Line Width Step,{:d}\n".format(int(float(value) / ims_res)))
            elif kl.startswith("min curvature window"):
                wf.write("Min Curvature Window,{:d}\n".format(int(float(value) / ims_res)))
            elif kl.startswith("max curvature window"):
                wf.write("Max Curvature Window,{:d}\n".format(int(float(value) / ims_res)))
            elif kl.startswith("minimum branch length"):
                wf.write("Minimum Branch Length,{:d}\n".format(int(float(value) / ims_res)))
            elif kl.startswith("minimum gap diameter"):
                wf.write("Minimum Gap Diameter,{:d}\n".format(int(float(value) / ims_res)))
            else:
                Log.logger.warning('Invalid parameter {}'.format(key))
