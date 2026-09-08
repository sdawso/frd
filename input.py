import logging
from PIL import Image
import numpy as np
import uncertainties.unumpy as unp
from uncertainties.umath import *
from uncertainties import ufloat, correlated_values
from astropy.stats import sigma_clip
from scipy.ndimage import binary_opening, median_filter
from pathlib import Path

## set log level for compatibility with logging
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(funcName)s: %(message)s")
logger = logging.getLogger(__name__)

# helper to compile reduced photometric data cubes into a single master reduction
def _master_reduction(master_path, res_sigma=3, prefix="", filetype="tiff"):
    """
    helper function, reduces multiple independent frames into one median master frame
    :param master_path: - path to input frames directory
    :param res_sigma: - sigma parameter for data reduction
    :param prefix: - prefix for input frames
    :param filetype: - file type for input frames

    :return: - master median, master standard deviation
    """
    ## load and sigma clip data files per res_sigma
    master_dir = Path(master_path)
    files = [p for p in master_dir.glob(f'{prefix}*.{filetype}')
             if p.is_file() and not p.stem.endswith("-bad")]
    stack = []
    for p in files:
        try:
            arr = np.array(Image.open(str(p)), dtype=np.float32)
        except Exception:
            continue
        stack.append(sigma_clip(arr, sigma=res_sigma).filled(np.nan))
    if not stack:
        return None, None
    cube = np.stack(stack)

    ## find master median
    master_median = np.nanmedian(cube, axis=0).astype(np.float32)
    master_std = (np.nanstd(cube, axis=0) if len(stack) > 1
                  else np.zeros_like(master_median)).astype(np.float32)

    ## filter out bad pixels
    bad = ~np.isfinite(master_median)
    if bad.any():
        master_median[bad] = np.nanmedian(master_median)
        master_std[bad] = np.nanmedian(master_std)

    ## return statistical coefficient arrays (master image and error likelihood)
    return master_median, master_std

##
def _load(frame, prefix="", filetype=""):
    """
    helper function to load an image, images, or reduced ndarray(s) depending on the input
    """
    if frame is None:
        return None ## reject bad frames
    if isinstance(frame, np.ndarray):
        return frame ## pass pre-reduced data
    if Path(frame).is_dir(): ## parse directories of photos into a single reduction master
        mas, _ = _master_reduction(frame, prefix=prefix, filetype=filetype)
    else:
        try:
            mas = np.array(Image.open(str(frame))) ## load image directly
        except Exception:
            mas = None ## return None if none of the above apply
    if mas is None:
        raise FileNotFoundError(f"Could not read image: {frame}")
    return mas

## accepts image and bias, dark, exposures -- add flat capability later
##
def input_reduction(img_path, master_bias_in=None, master_dark_in=None,
                    exp_time=None, dark_exp_time=None, filetype="tiff",
                    saturation_adu=None):
    """
    returns a combined photometric reduction of science data using the above helper functions
    :param img_path: - path to input frames directory
    :param master_bias_in: - master bias (optional)
    :param master_dark_in: - master dark (optional)
    :param exp_time: - exposure time (optional)
    :param dark_exp_time: - dark exposure time (optional)
    :param filetype: - file type for input frames
    :param saturation_adu: - saturation adu (optional)

    :return: median-scaled science reduction of input data, pixel saturation information
    """

    img = _load(img_path, filetype=filetype).astype(np.float32)

    # ingesting raw data
    if isinstance(img_path, np.ndarray):
        raw = img_path
    elif Path(img_path).is_file():
        raw = np.array(Image.open(str(img_path)))
    else:
        raw = None
        logger.debug(f"{img_path}: master-combined input; saturation detection skipped.")

    # saturation detection (hot pixels)
    sat_mask = np.zeros(img.shape, dtype=bool)
    if raw is not None:
        if saturation_adu is None:
            if np.issubdtype(raw.dtype, np.integer):
                saturation_adu = 0.99 * np.iinfo(raw.dtype).max # adaptive bit depth
            else:
                logger.warning(f"{img_path}: non-integer dtype {raw.dtype} and no explicit "
                               f"saturation_adu; saturation detection disabled.")
        if saturation_adu is not None:
            sat_mask = raw >= saturation_adu

    n_sat = int(sat_mask.sum())
    if n_sat / sat_mask.size > 0.01:  # flagging uncalibrated bit depth
        logger.error(f"{img_path}: {100 * n_sat / raw.size:.1f}% of frame above "
                     f"saturation_adu={saturation_adu:.0f}; threshold likely wrong "
                     f"for this data scaling — check bit packing.")

    # hot pixel/oversaturation reporting and patching
    if n_sat:
        clumped_mask = binary_opening(sat_mask, structure=np.ones((3, 3), dtype=bool))
        isolated_mask = sat_mask & ~clumped_mask
        n_clumped = int(clumped_mask.sum())
        n_isolated = int(isolated_mask.sum())

        if n_clumped > 0:
            logger.warning(f"{n_clumped} clumped saturated pixels in {img_path}; "
                            f"ring fit may be biased.")
        if n_isolated > 0:
            logger.info(f"{n_isolated} isolated hot pixel in {img_path}; "
                         f"patching local median.")

        # patch with median filter
        if n_isolated or n_clumped:
            med = median_filter(img, size=5)
            repair_mask = isolated_mask | clumped_mask
            img[repair_mask] = med[repair_mask]

    bias = _load(master_bias_in, prefix="bias", filetype=filetype)
    dark = _load(master_dark_in, prefix="dark", filetype=filetype)

    # apply bias and dark data
    if bias is not None:
        img -= bias
    if dark is not None:
        d = dark - bias if bias is not None else dark
        scale = (exp_time / dark_exp_time) if (exp_time and dark_exp_time) else 1.0
        img -= d * scale

    return img, n_sat