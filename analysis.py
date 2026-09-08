import cv2
import logging
import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from scipy.signal import find_peaks
from scipy.stats import median_abs_deviation
from lmfit.models import MoffatModel

import input

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(funcName)s: %(message)s")
logger = logging.getLogger(__name__)

# primary annulus identification function, very first point of contact for establishing analysis axes
# takes an image path and optionally a path to a series of darks
# TODO: integrate bias, flat reduction

def _hwhm_crossings(r_prof, r_px, peak_r_px,
                    debounce_px=20.0, bin_step=5.0, h_frac=0.05):
    """
    computes HWHM offset deltas from a specified peak with debouncing and hysteresis.
    :param r_prof: - radial profile y data
    :param r_px: - radial profile x data
    :param peak_r_px: - peak index of y data
    :param debounce_px: - minimum HWHM gap for debouncing
    :param bin_step: - radial bin size
    :param h_frac: - +/- percentage deviation from 50% for hysteresis bands

    :return: - left_deltas, right_deltas, peak_r (origin)
    """

    # instantiating profile copies for splitting, binning
    full_prof = r_prof

    n_bins = len(r_prof)
    peak_val, peak_r = r_prof[peak_r_px], r_px[peak_r_px]
    min_gap = int(debounce_px / bin_step)

    side_configs = (
        ('left', r_prof[:peak_r_px + 1], 0),
        ('right', r_prof[peak_r_px:], peak_r_px),
    )

    # delta HWHM dictionary (peak origin)
    deltas_by_side = {'left': np.array([]), 'right': np.array([])}

    # side-biased delta calculation loop
    for side, rp, offset in side_configs:
        if len(rp) < 2:
            continue

        amp = peak_val - rp.min() if len(rp) > 0 else 0
        thresh = rp.min() + 0.5 * amp if len(rp) > 0 else 0
        thresh_high = thresh + h_frac * amp
        thresh_low  = thresh - h_frac * amp

        # debounced HWHM crossing candidates
        idx = np.where(np.diff(np.signbit(rp - thresh)))[0]
        if len(idx) == 0:
            continue

        # hysteresis band block
        idx = np.sort(idx)[::-1] if side == 'left' else np.sort(idx)
        valid = [idx[0]]
        for i in idx[1:]:
            prof_inward  = rp[i + 1:] if side == 'left' else rp[:i + 1]
            prof_outward = rp[:i + 1] if side == 'left' else rp[i + 1:]
            if np.any(prof_inward > thresh_high) and np.any(prof_outward < thresh_low):
                if not valid or abs(i - valid[-1]) >= min_gap:
                    valid.append(i)

        # calculation of HWHM offsets from peak
        deltas = []
        for i in valid:
            fi = offset + i
            if fi + 1 >= n_bins:
                continue
            y0, y1 = full_prof[fi], full_prof[fi + 1]
            if y1 == y0:
                continue
            x0, x1 = r_px[fi], r_px[fi + 1]
            r_cross = x0 + (thresh - y0) * (x1 - x0) / (y1 - y0)
            deltas.append(peak_r - r_cross if side == 'left' else r_cross - peak_r)

        deltas_by_side[side] = np.array(deltas)

    return deltas_by_side['left'], deltas_by_side['right'], peak_r

def _find_primary_peak(prof, stderr):
    """
    locates the primary interior peak in a profile wrt stderr noise floor
    :param prof: xy data of radial profile
    :param stderr: stderr of radial profile

    :return: (peak_r_px, peaks, props)
    """
    n_bins = len(prof)
    if n_bins == 0:
        return None, np.array([]), {}

    # calculate noise floor
    fin = stderr[np.isfinite(stderr)]
    noise_floor = np.median(fin) if fin.size else 0.0

    # Use Block 2's safe lower bound, but keep Block 1's warning trigger
    prom = max((3.0 * noise_floor), 1e-12)
    if prom == 1e-12 and logger:
        logger.warning("Prominence of peaks estimated to be zero, stderr has collapsed")

    peaks, props = find_peaks(prof, prominence=prom)

    # primary interior peak selection
    interior = [p for p in peaks if 0 < p < n_bins - 1]

    if interior:
        peak_idx = interior[int(np.argmax(prof[interior]))]
    else:
        peak_idx = int(np.argmax(prof))
        if peak_idx in (0, n_bins - 1):  # rejection of no-local-maxima case
            logger.debug("No interior peak found; profile may be noise-dominated.")
            return None, peaks, props

    return peak_idx, peaks, props

def radial_profile(data, center, axes, angle_deg, debounce_px=20.0, bin_step=5.0,
                   arc_start=0.0, arc_end=360.0, h_frac=0.05, geometry=None):
    """
    calculates a radial profile and associated physical properties across a bin of image pixels (2 dimensional
    ndarray) using scipy.stats' median_abs_deviation()

    :param data: input data (wedge bin)
    :param center: center of ellipse
    :param axes: axes of ellipse
    :param angle_deg: rotation angle of ellipse
    :param debounce_px: fit debounce pixel magnitude
    :param bin_step: bin size in pixels
    :param arc_start: arc start (degrees)
    :param arc_end: arc end (degrees)
    :param geometry: can optionally input (r_px, pixel_angles) to use a precalculated annular plane

    :return: MAD-adjusted radial profile y data, encircled energy across profile, radial x data, HWHM crossings, peak
    x-coordinate, stdev error introduced by binning
    """

    # preload ring coordinates if available
    if geometry is None:
        geometry = _ring_polar_coordinates(data.shape, center, axes, angle_deg)
    r_pixel, pixel_angles = geometry

    # slice a the pizza (azimuthal binning based on number of requested slices in circlefinder)
    arc_start = arc_start % 360.0
    arc_end = 360.0 if (arc_end % 360.0 == 0.0 and arc_end != 0.0) else arc_end % 360.0
    if arc_start <= arc_end:
        arc_mask = (pixel_angles >= arc_start) & (pixel_angles <= arc_end)
    else:
        arc_mask = (pixel_angles >= arc_start) | (pixel_angles <= arc_end)

    # instantiate bin axis and data
    flat_r = r_pixel[arc_mask]
    flat_d = data[arc_mask].astype(float)
    if len(flat_r) == 0:
        logger.warning("Pizza slice yielded 0 pixels.")
        empty = np.array([])
        return empty, empty, empty, empty, empty, None, empty

    # radial binning based on bin_step
    r_bins = (flat_r / bin_step).astype(np.int32)
    n_bins = int(r_bins.max()) + 1
    order = np.argsort(r_bins, kind='stable')
    rb_sorted, d_sorted = r_bins[order], flat_d[order]
    uniq, starts = np.unique(rb_sorted, return_index=True)
    chunks = np.split(d_sorted, starts[1:])

    # calculating stderr and radial mad contour
    radial_prof = np.zeros(n_bins)
    bin_stderr = np.full(n_bins, np.inf)
    nr = np.zeros(n_bins, dtype=np.int64)
    for u, chunk in zip(uniq, chunks):
        nr[u] = len(chunk)
        radial_prof[u] = np.median(chunk)
        mad = median_abs_deviation(chunk, scale='normal')
        if mad > 0 and len(chunk) > 1:
            bin_stderr[u] = mad / np.sqrt(len(chunk))

    # remapping from bin to image coordinates
    r_px_coords = bin_step * (np.arange(n_bins) + 0.5)

    # encircled energy calculation
    tbin_signal = np.clip(radial_prof, 0, None) * nr
    total = tbin_signal.sum()
    ee = np.maximum.accumulate(np.cumsum(tbin_signal) / total) if total > 0 else np.zeros(n_bins)

    # filtering noise from bin_stderr floor, locating peaks
    noise_floor = np.median(bin_stderr[np.isfinite(bin_stderr)]) if np.any(np.isfinite(bin_stderr)) else 0.0
    prom = max((3.0 * noise_floor), 0.0)
    if prom == 0.0:
        logger.warning("Prominence of peaks estimated to be negative, stderr has collapsed")

    # filtering noise from bin_stderr floor, locating peaks
    peak_idx, peaks, props = _find_primary_peak(radial_prof, bin_stderr)

    if peak_idx is None:
        # rejection of no-local-maxima case
        return radial_prof, ee, r_px_coords, np.array([]), np.array([]), None, bin_stderr

    peak_val, peak_r = radial_prof[peak_idx], r_px_coords[peak_idx]

    left_deltas, right_deltas, _ = _hwhm_crossings(
        radial_prof, r_px_coords, peak_idx,
        debounce_px=debounce_px, bin_step=bin_step, h_frac=h_frac
    )

    return radial_prof, ee, r_px_coords, left_deltas, right_deltas, peak_r, bin_stderr

def _mad_crest_clip(radii, keep_sigma, scale_floor ):
    """
    MAD-clip crest radii distribution against median to prune speckle outliers
    :param radii: list of radii from circlefinder
    :param keep_sigma: keep abs radii within the median +/- sigma (prune outlier noise)
    :param scale_floor: scaling floor applied to the crest channel thickness to prevent quantization errors

    :return: keep mask for annular crest
    """

    # MAD mask calculation
    med = np.median(radii)
    mad = median_abs_deviation(radii, scale='normal')

    # scale floor application (recommended ~ 2*crest_bin_px for sigma flooring to prevent quantization breakdown)
    thresh = max(keep_sigma * mad, scale_floor) if np.isfinite(mad) else scale_floor
    if thresh <= 0: # flag negative thresholds
        logger.warning(f"MAD clip degenerate (mad={mad}, n={radii.size}, "
                       f"med={med:.2f}); keeping all crests.")
        return np.ones(radii.shape, dtype=bool)
    return np.abs(radii - med) <= thresh

def _ring_polar_coordinates(shape, center, axes, angle_deg):
    """
    compute elliptical-frame coordinates of annular crest

    :param shape: - dimension of frame (px)
    :param center: - center of ellipse in frame
    :param axes: - ellipse sma and smi axes ((d1, d2) in circlefinder) in px
    :param angle_deg: - fitted rotation angle of ellipse (deg)

    :return: (r_px, pixel_angles_deg) for ring mathematics
    """

    # row-column index instantiation, origin
    y, x = np.indices(shape)
    x_c, y_c = x - center[0], y - center[1]

    # angular coordinate instantiation for rotational basis
    r_ang = np.radians(angle_deg)
    cr_ang, sr_ang = np.cos(r_ang), np.sin(r_ang)
    x_rot = x_c * cr_ang + y_c * sr_ang
    y_rot = -x_c * sr_ang + y_c * cr_ang

    # semi-major and minor scaling, drawing of ellipse across polar plane at r_elliptical
    a, b = axes[0] / 2, axes[1] / 2
    r_elliptical = np.sqrt((x_rot / a) ** 2 + (y_rot / b) ** 2)

    # pixel coordinates of ellipse mask in rotational basis
    r_px = r_elliptical * ((a + b) / 2)
    pixel_angles = np.degrees(np.arctan2(y_rot, x_rot)) % 360.0

    return r_px, pixel_angles

def _ring_xy_coordinates(r_polar, theta_deg, center, axes, angle_deg):
    """
    inverse of _ring_polar_coordinates -- compute xy coordinates from elliptical frame

    :param r_polar: norm profile of ellipse r vector in ellipse basis (NOT pixel basis)
    :param theta_deg: - integrating rotation angle (deg)
    :param center: - center of ellipse in pixel basis
    :param axes: - ellipse sma and smi axes from fit
    :param angle_deg: - fitted rotation angle of ellipse (deg)

    :return: pixel basis profile coordinates (x_px, y_px)
    """

    # define integrating ellipse plane (center, axis, theta)
    x_c, y_c = center
    a, b = axes[0]/2.0, axes[1]/2.0
    theta_rad = np.radians(theta_deg)

    # map r_polar to pixel basis as x_r, y_r
    r_px = r_polar / ((a + b) / 2.0) / np.hypot(np.cos(theta_rad) / a, np.sin(theta_rad) / b)
    angle_rad = np.radians(angle_deg)
    x_r, y_r = r_px*np.cos(theta_rad), r_px*np.sin(theta_rad)

    # rotate x_r, y_r into image basis and center on (x_c, y_c)
    x_px, y_px = (x_c + x_r * np.cos(angle_rad) - y_r * np.sin(angle_rad),
                  y_c + x_r * np.sin(angle_rad) + y_r * np.cos(angle_rad))

    return x_px, y_px


def circlefinder(img_path, detection_sigma=3.0, crest_sigma=3.0, dark_path=None,
                 max_iter=3, crest_bin_px=10.0, tol_px=0.1,
                 n_wedge=72, min_crest_px=5.0, min_ring_px=10.0):
    """

    """

    wedge_deg = 360.0 / n_wedge

    # load image, get background info
    img, n_sat = input.input_reduction(img_path, master_dark_in=dark_path)
    _, bkg_med, bkg_std = sigma_clipped_stats(img, sigma=detection_sigma)

    img_sub = img.astype(float) - bkg_med
    img_smooth = cv2.GaussianBlur(img_sub, (15, 15), 0)
    _, sm_med, sm_std = sigma_clipped_stats(img_smooth, sigma=detection_sigma)

    # create binary mask against background using detection_sigma and bkg info
    binary = np.ascontiguousarray(
        (img_smooth > sm_med + detection_sigma * sm_std).astype(np.uint8) * 255)

    # identify contours in binary mask
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contours found above intensity threshold.")

    # outer contour
    outer = max(contours, key=cv2.contourArea)
    if len(outer) < 5:  # fitEllipse needs >=5 points (ellipse has 5 DOF)
        raise ValueError(f"Contour has {len(outer)} points; fitEllipse requires 5.")
    if cv2.contourArea(outer) < np.pi * (min_ring_px / 2) ** 2:
        raise ValueError(f"Contour area {cv2.contourArea(outer):.0f} square px below "
                         f"min_ring_px={min_ring_px} threshold.")

    # contour center and sma/smi
    (cx, cy), (d1, d2), angle = cv2.fitEllipse(outer)
    if d1 < min_ring_px or d2 < min_ring_px:
        raise ValueError(f"Fitted ellipse impossibly small (d1={d1:.1f}, d2={d2:.1f}).")

    # derive ring coordinates from ellipse fit
    pts = None
    shifts = []
    ee_wedge_avg = None
    r_px_master = None

    # refinement loop -- retarget until variance of fit is stabilized
    for it in range(max_iter):
        geom = _ring_polar_coordinates(img.shape, (cx, cy), (d1, d2), angle)
        crest_xy, crest_r, crest_ee, crest_r_px = [], [], [], []

        # generate wedge masks for crest location
        for k in range(n_wedge):
            a0 = k * wedge_deg
            _, ee_w, r_px, _, _, rp, _ = radial_profile(img_sub, None, None, None,
                                                        bin_step=crest_bin_px, arc_start=a0,
                                                        arc_end=a0 + wedge_deg, geometry=geom)

            if rp is None or rp <= min_crest_px:
                continue

            # resample per wedge back to xy for final contour
            x_img, y_img = _ring_xy_coordinates(rp, a0 + 0.5 * wedge_deg, (cx, cy), (d1, d2), angle)
            crest_xy.append([x_img, y_img])
            crest_r.append(rp)
            crest_ee.append(ee_w)
            crest_r_px.append(r_px)

        if len(crest_xy) < max(5, int(0.25 * n_wedge)):  # flag sparse refinement fit
            logger.warning(f"{img_path}: only {len(crest_xy)} usable wedges on "
                           f"iter {it}; stopping crest refinement.")
            break

        crest_xy = np.asarray(crest_xy, dtype=float)
        keep = _mad_crest_clip(np.asarray(crest_r, dtype=float), keep_sigma=crest_sigma,
                               scale_floor=2.0 * crest_bin_px)

        if keep.sum() >= max(5, int(0.25 * n_wedge)):  # flag overpruned selections in MAD refinement mask
            n_rej = int((~keep).sum())
            if n_rej:
                logger.info(f"{img_path}: iter {it} rejected {n_rej} outlier crests.")
            crest_xy = crest_xy[keep]

        # refit pruned and reduced selection back through fitEllipse
        (ncx, ncy), (nd1, nd2), nangle = cv2.fitEllipse(crest_xy.astype(np.float32).reshape(-1, 1, 2))
        if nd1 < min_ring_px or nd2 < min_ring_px:
            logger.warning(f"{img_path}: iter {it} ellipse degenerate; keeping previous.")
            break

        # characterize center shift of refinement
        shift = float(np.hypot(ncx - cx, ncy - cy))
        shifts.append(shift)
        cx, cy, d1, d2, angle = ncx, ncy, nd1, nd2, nangle
        pts = crest_xy

        # calculate average EE for the valid crests in this iteration
        if len(crest_ee) > 0:
            max_bins = max(len(r) for r in crest_r_px)
            r_px_master = crest_bin_px * (np.arange(max_bins) + 0.5)
            # interpolate to a common axis before averaging, wedges might have slightly different max radii at frame edge
            ee_interp = [np.interp(r_px_master, r, e, left=0, right=1.0) for r, e in zip(crest_r_px, crest_ee)]
            ee_wedge_avg = np.mean(ee_interp, axis=0) # ee normalized to 1.0 so mean is taken

        if shift < tol_px:  # break once stabilized
            break

    # fallback for refinement failure/reporting
    if pts is None:
        logger.warning(f"Crest refit failed for {img_path}; falling back to edge-contour ellipse.")
        pts = outer.squeeze(axis=1).astype(float)
    else:
        logger.info(f"{img_path}: center shifts {['%.2f' % s for s in shifts]} px")
        if len(shifts) >= 2 and shifts[-1] > shifts[-2]:
            logger.warning(f"{img_path}: crest refinement diverging "
                           f"({shifts[-2]:.2f} -> {shifts[-1]:.2f} px); center suspect.")

    # characterize the spread of the crest against an idealized ellipse contour - error residual
    a, b = d1 / 2, d2 / 2
    r_ang = np.radians(angle);
    cr_ang, sr_ang = np.cos(r_ang), np.sin(r_ang)
    xr = (pts[:, 0] - cx) * cr_ang + (pts[:, 1] - cy) * sr_ang
    yr = -(pts[:, 0] - cx) * sr_ang + (pts[:, 1] - cy) * cr_ang
    ang_p = np.arctan2(yr, xr)
    r_ideal = 1.0 / np.sqrt((np.cos(ang_p) / a) ** 2 + (np.sin(ang_p) / b) ** 2)
    contour_rms = float(np.sqrt(np.mean((np.hypot(xr, yr) - r_ideal) ** 2)))

    # Return the empirical average wedge EE alongside the image metadata
    return (cx, cy), (d1, d2), angle, img_sub, contour_rms, ee_wedge_avg, r_px_master

def _ring_moffat_fit(fit_data, x_coords, peak_r, ee_integration_radius=None,
                     weights=None, hwhm_seed_px=None, fit_window=None):

    """Symmetric Moffat + linear bkg. EE from background-free Moffat over a
    fixed integration radius. Slope bounded <=0 and sigma seeded to block the
    collapsed basin (narrow Moffat riding a steep rising background)."""

    from lmfit.models import LinearModel
    peak_model = MoffatModel()
    model = peak_model + LinearModel(prefix='bkg_')
    params = peak_model.guess(fit_data, x=x_coords)
    params['center'].set(value=peak_r, min=0)
    params['beta'].set(min=1.05, max=10.0)
    sig_floor = max(0.5 * float(x_coords[1] - x_coords[0]), 0.01)
    if hwhm_seed_px is not None and np.isfinite(hwhm_seed_px) and hwhm_seed_px > 0:
        sig_floor = max(sig_floor, 0.15 * hwhm_seed_px)
    params['sigma'].set(min=sig_floor, max=x_coords[-1] - x_coords[0])

    if fit_window is None and hwhm_seed_px and np.isfinite(hwhm_seed_px):
        lo = max(0.0, peak_r - 8.0 * hwhm_seed_px)
        hi = peak_r + 16.0 * hwhm_seed_px
        m_win = (x_coords >= lo) & (x_coords <= hi)
        if m_win.sum() >= 20:
            x_coords, fit_data = x_coords[m_win], fit_data[m_win]
            if weights is not None:
                weights = weights[m_win]
        else:
            logger.debug(f"Fit window {lo:.0f}-{hi:.0f} px kept only "
                         f"{m_win.sum()} bins; using full range.")

    if hwhm_seed_px is not None and np.isfinite(hwhm_seed_px) and hwhm_seed_px > 0:
        b0 = params['beta'].value if params['beta'].value else 2.5
        s0 = hwhm_seed_px / np.sqrt(2.0 ** (1.0 / b0) - 1.0)
        params['sigma'].set(value=float(np.clip(s0, 0.01,
                                                x_coords[-1] - x_coords[0])))

    params.add('bkg_slope', value=0.0, max=0.0)
    params.add('bkg_intercept', value=float(np.min(fit_data)))
    result = model.fit(fit_data, params, x=x_coords, weights=weights)

    k0 = np.sqrt(2.0 ** (1.0 / result.params['beta'].value) - 1.0)
    h0 = result.params['sigma'].value * k0
    if hwhm_seed_px and np.isfinite(hwhm_seed_px) and h0 < 0.3 * hwhm_seed_px:
        logger.warning(f"Moffat collapsed (h={h0:.2f} vs seed {hwhm_seed_px:.2f}); "
                       f"refitting with flat background.")
        p2 = params.copy()
        p2['bkg_slope'].set(value=0.0, vary=False)
        alt = model.fit(fit_data, p2, x=x_coords, weights=weights)
        if alt.redchi < result.redchi or h0 < 1.0:
            result = alt

    r_max = ee_integration_radius or 3.0 * result.params['center'].value
    r_th = np.linspace(0, r_max, 4000)
    ring_only = np.clip(peak_model.eval(result.params, x=r_th), 0, None)
    flux = ring_only * r_th
    ee = np.cumsum(flux)
    ee /= (ee[-1] + 1e-12)
    ee = np.maximum.accumulate(ee)
    ee_half = np.cumsum(flux[r_th <= r_max / 2])
    trunc_sens = 1.0 - ee_half[-1] / (np.sum(flux) + 1e-12)

    sig = result.params['sigma'].value
    bet = result.params['beta'].value
    k = np.sqrt(2.0 ** (1.0 / bet) - 1.0)
    hwhm_px = sig * k
    fw = result.params.get('fwhm')
    if fw is not None and fw.stderr is not None:
        hwhm_err = fw.stderr / 2.0
    else:
        hwhm_err = (result.params['sigma'].stderr or 0.05) * k

    A = result.params['height'].value
    b = result.params['bkg_intercept'].value
    m = result.params['bkg_slope'].value
    c = result.params['center'].value
    beta_railed = bool(bet <= 1.06 or bet >= 9.9)
    slope_frac = abs(m * c) / (abs(b) + 1e-12)
    q_fit = (b - float(np.min(fit_data))) / (A + 1e-12)
    rms_frac = float(np.sqrt(max(result.redchi, 0.0)) / (abs(A) + 1e-12))

    diag = {
        'slope_frac': float(slope_frac), 'q_fit': float(q_fit),
        'rms_frac': rms_frac,
        'hwhm_px': hwhm_px, 'hwhm_err_px': hwhm_err,
        'beta_railed': beta_railed, 'redchi': float(result.redchi),
        'trunc_sens': float(trunc_sens),
        'collapsed': bool(hwhm_px < 1.0),
    }
    return result, c, ee, r_th, trunc_sens, diag


def profile_analysis(image_path, dark_path=None, bin_step=5.0, num_slices=10,
                     shared_center=None, ee_integration_radius=None, debounce_px=20):
    logger.debug(f"Processing: {image_path}")

    # Unpack the new empirical ee arrays from circlefinder
    (cxf, cyf), axes_, angle, image_sub, contour_rms, ee_emp, ee_r_px = circlefinder(
        image_path, dark_path=dark_path)

    cx, cy = shared_center if shared_center is not None else (cxf, cyf)

    geometry = _ring_polar_coordinates(image_sub.shape, (cx, cy), axes_, angle)

    theta_edges = np.linspace(0, 360, num_slices + 1)
    theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2.0
    slice_data, r_master = [], None

    for i in range(num_slices):
        prof, ee_a, r_px, ld, rd, pk, berr = radial_profile(
            image_sub, (cx, cy), axes_, angle, bin_step=bin_step,
            arc_start=theta_edges[i], arc_end=theta_edges[i + 1],
            geometry=geometry)
        if r_master is None and len(r_px) > 0:
            r_master = r_px
        slice_data.append({'theta': theta_centers[i], 'r_px': r_px,
                           'prof': prof, 'raw_peak_r': pk,
                           'left_deltas': ld, 'right_deltas': rd,
                           'ee': ee_a, 'bin_stderr': berr})

    peaks = [s['raw_peak_r'] for s in slice_data if s['raw_peak_r'] is not None]
    if not peaks:
        raise ValueError("No valid peaks in any slice.")
    mean_peak_r = float(np.mean(peaks))
    peak_scatter = float(np.std(peaks))  # real azimuthal wobble, reported

    def _stack(align):
        rows = []
        for s in slice_data:
            if s['raw_peak_r'] is None or len(s['r_px']) == 0:
                continue
            shift = (mean_peak_r - s['raw_peak_r']) if align else 0.0
            rows.append(np.interp(r_master, s['r_px'] + shift,
                                  s['prof'], left=0, right=0))
        return np.array(rows)

    prof_aligned = np.median(_stack(True), axis=0)
    prof_unaligned = np.median(_stack(False), axis=0)

    # Replaced redundant _empirical_hwhm and _median_profile_hwhm calls
    # utilizing the unified crossing logic:
    master_peak = int(np.argmax(prof_aligned))
    left_deltas, right_deltas, pk_m = _hwhm_crossings(
        prof_aligned, r_master, master_peak,
        debounce_px=debounce_px, bin_step=bin_step
    )

    # Compute h_emp as the average of the inner-most crossing deltas
    l_val = left_deltas[0] if len(left_deltas) > 0 else np.nan
    r_val = right_deltas[0] if len(right_deltas) > 0 else np.nan
    h_emp = np.nanmean([l_val, r_val])

    seed = h_emp if np.isfinite(h_emp) else None

    fit_a, _, ee_m, r_th, trunc, diag_a = _ring_moffat_fit(
        prof_aligned, r_master, mean_peak_r, ee_integration_radius,
        hwhm_seed_px=seed)
    fit_u, _, _, _, _, _ = _ring_moffat_fit(
        prof_unaligned, r_master, mean_peak_r, ee_integration_radius,
        hwhm_seed_px=seed)

    boot_s, boot_c = [], []
    for s in slice_data:
        s['fit_result'] = None
        if s['raw_peak_r'] is None or len(s['prof']) == 0:
            continue
        try:
            n = min(len(s['prof']), len(s['bin_stderr']))
            w = 1.0 / np.clip(s['bin_stderr'][:n], 1e-3, np.inf)
            res_s, _, _, _, _, _ = _ring_moffat_fit(
                s['prof'][:n], s['r_px'][:n],
                s['raw_peak_r'], ee_integration_radius, weights=w,
                hwhm_seed_px=seed)
            s['fit_result'] = res_s
            boot_s.append(res_s.params['sigma'].value)
            boot_c.append(res_s.params['center'].value)
        except Exception as e:
            logger.debug(f"Slice fit failed: {e}")
            boot_s.append(res_s.params['sigma'].value)
            boot_c.append(res_s.params['center'].value)

    if len(boot_s) > 1:
        sigma_err = float(np.std(boot_s) / np.sqrt(len(boot_s)))
        center_err = float(np.std(boot_c) / np.sqrt(len(boot_c)))
    else:
        sigma_err = fit_a.params['sigma'].stderr or 0.5
        center_err = fit_a.params['center'].stderr or 0.5

    y, x = np.indices(image_sub.shape)
    ih, iw = image_sub.shape[:2]
    if (iw, ih) != (5472, 3648):
        logger.warning(f"Frame {iw}x{ih} not native 5472x3648")
    r_full = float(min(cx, cy, iw - cx, ih - cy))
    w_fit = fit_a.params['sigma'].value
    coverage_ok = bool(mean_peak_r + 4.0 * w_fit <= r_full)
    if not coverage_ok:
        logger.warning(f"Ring wing exceeds inscribed radius {r_full:.0f}px "
                       f"(peak {mean_peak_r:.0f}, sigma {w_fit:.1f}) — "
                       f"outer bins are corner-only")
    ap = np.hypot(x - cx, y - cy) <= mean_peak_r + 6 * fit_a.params['sigma'].value
    _, bkg_med, _ = sigma_clipped_stats(image_sub[~ap], sigma=3.0)
    total_flux = float(np.sum(image_sub[ap] - bkg_med))

    # Calculate both 95% radii
    ee95_moffat = float(np.interp(0.95, ee_m, r_th))
    ee95_emp = float(np.interp(0.95, ee_emp, ee_r_px)) if ee_emp is not None else np.nan

    return {
        'center': (cx, cy), 'axes': axes_, 'angle': angle, 'image': image_sub,
        'avg_data': prof_aligned, 'avg_data_unaligned': prof_unaligned,
        'r_px': r_master, 'ee': ee_m, 'ee_r_theoretical': r_th,

        'ee_empirical': ee_emp,
        'ee_r_px': ee_r_px,
        'ee95_radius_moffat': ee95_moffat,
        'ee95_radius_emp': ee95_emp,
        'ee95_ratio': float(ee95_emp / ee95_moffat) if ee95_moffat > 0 else np.nan,

        'result': fit_a, 'result_unaligned': fit_u,
        'sigma': fit_a.params['sigma'].value, 'sigma_err': sigma_err,
        'sigma_unaligned': fit_u.params['sigma'].value,
        'peak_r': fit_a.params['center'].value, 'peak_r_err': center_err,
        'peak_scatter': peak_scatter, 'raw_peak_r': mean_peak_r,
        'r_full_px': r_full, 'coverage_ok': coverage_ok,
        'beta': fit_a.params['beta'].value,
        'beta_err': fit_a.params['beta'].stderr or 0.0,
        'ee95_radius': ee95_moffat,
        'ee_truncation_sensitivity': trunc,
        'hwhm_px': diag_a['hwhm_px'],
        'hwhm_err_px': diag_a['hwhm_err_px'], # covariance-only
        'hwhm_total_err_px': max(float(np.hypot(diag_a['hwhm_err_px'],
                                                max(sigma_err, 0.0) *
                                                np.sqrt(2 ** (1 / fit_a.params['beta'].value) - 1))), 0.05),
        'h_emp_px': float(h_emp),
        'fit_flags': diag_a,
        'hwhm_ratio': (float(h_emp / diag_a['hwhm_px'])
                       if np.isfinite(h_emp) and diag_a['hwhm_px'] > 0
                       else np.nan),
        'total_flux': total_flux, 'contour_rms': contour_rms,
        'theta_centers': theta_centers, 'slice_data': slice_data,
        'left_deltas': left_deltas, 'right_deltas': right_deltas,
    }


def calculate_frd(results, input_angles, pixel_size=4.8e-3, d_tol=0.10,
                  camera_dist_mm=10.0, core_D_um = 150.0, collimator_f_mm = 18.24):

    input_angles = np.array(input_angles, dtype=float)
    floor_deg = np.degrees(np.arctan((core_D_um * 1e-3 / 2) / collimator_f_mm))

    logger.info(f"Instrument angular floor: {floor_deg:.3f} deg HWHM "
                f"({core_D_um:.0f} um source / f={collimator_f_mm} mm). "
                f"Widths at or below this are source-limited.")

    tan_in = np.tan(np.radians(input_angles))
    peak_px = np.array([ufloat(r['peak_r'], max(r['peak_r_err'], 0.1))
                        if r else ufloat(np.nan, np.nan) for r in results])

    r0 = ufloat(0.0, 0.0)
    D = ufloat(camera_dist_mm, d_tol * camera_dist_mm) # mechanical and human error tolerance

    sigma_unaligned_px = np.array([r['sigma_unaligned'] if r else np.nan for r in results])

    def _hwhm_ufloat(r):
        if not r:
            return ufloat(np.nan, np.nan)
        k = np.sqrt(2 ** (1 / r['beta']) - 1)
        fit_err = r.get('hwhm_err_px', np.nan)
        if not np.isfinite(fit_err):
            fit_err = max(r['sigma_err'], 0.05) * k
        azim_err = max(r['sigma_err'], 0.0) * k
        return ufloat(r['hwhm_px'], np.hypot(fit_err, azim_err))

    hwhm_px = np.array([_hwhm_ufloat(r) for r in results])
    theta_peak = unp.degrees(unp.arctan((peak_px * pixel_size - r0) / D))
    theta_hwhm_outer = unp.degrees(unp.arctan(
        ((peak_px + hwhm_px) * pixel_size - r0) / D))
    sigma_deg = theta_hwhm_outer - theta_peak
    sigma_unaligned_deg = np.degrees(np.arctan(
        ((unp.nominal_values(peak_px) + sigma_unaligned_px) * pixel_size
         - r0.nominal_value) / D.nominal_value)) - unp.nominal_values(theta_peak)

    ee95_moffat_px = np.array([ufloat(r['ee95_radius_moffat'],
                                      r['ee95_radius_moffat'] * r['ee_truncation_sensitivity']
                                      + r['peak_r_err']) if r
                               else ufloat(np.nan, np.nan) for r in results])

    ee95_emp_px = np.array([ufloat(r['ee95_radius_emp'],
                                   r['ee95_radius_emp'] * r['ee_truncation_sensitivity']
                                   + r['peak_r_err']) if r and np.isfinite(r.get('ee95_radius_emp', np.nan))
                            else ufloat(np.nan, np.nan) for r in results])

    theta_out95_moffat = unp.degrees(unp.arctan((ee95_moffat_px * pixel_size - r0) / D))
    theta_out95_emp = unp.degrees(unp.arctan((ee95_emp_px * pixel_size - r0) / D))

    f_in = 1 / (2 * tan_in)

    f_out95_moffat = 1 / (2 * unp.tan(unp.radians(theta_out95_moffat)))
    f_out95_emp = 1 / (2 * unp.tan(unp.radians(theta_out95_emp)))

    tan_out95 = unp.tan(unp.radians(theta_out95_moffat))
    tan_d = unp.sqrt(np.maximum(tan_out95 ** 2 - tan_in ** 2, 0))
    theta_d = unp.degrees(unp.arctan(tan_d))

    flux = np.array([r['total_flux'] if r else np.nan for r in results])
    flux_rel = flux / np.nanmax(flux) if np.any(np.isfinite(flux)) else flux

    Dn, r0n = D.nominal_value, r0.nominal_value

    def _to_deg(pk, deltas, sign):
        th_pk = np.degrees(np.arctan((pk * pixel_size - r0n) / Dn))
        th = np.degrees(np.arctan(
            ((pk + sign * np.asarray(deltas)) * pixel_size - r0n) / Dn))
        return sign * (th - th_pk)

    def _slice_deltas_deg(r, key, sign):
        if not r:
            return None
        out = []
        for s in r.get('slice_data', []):
            if s.get(key) is not None and len(s[key]) > 0 and s.get('raw_peak_r') is not None:
                out.extend(_to_deg(s['raw_peak_r'], s[key], sign))
        return out if out else None

    def _med_delta_deg(r, key, sign):
        if not r or r.get(key) is None or len(r[key]) == 0:
            return np.nan
        return float(_to_deg(r['peak_r'], r[key], sign)[0])

    hwhm_left_deg = [_slice_deltas_deg(r, 'left_deltas', -1) for r in results]
    hwhm_right_deg = [_slice_deltas_deg(r, 'right_deltas', +1) for r in results]

    med_hwhm_left_deg = np.array([_med_delta_deg(r, 'left_deltas', -1) for r in results])
    med_hwhm_right_deg = np.array([_med_delta_deg(r, 'right_deltas', +1) for r in results])

    rows = []
    for a, r in zip(input_angles, results):
        d = r.get('fit_flags', {}) if r else {}
        ratio = r.get('hwhm_ratio', np.nan) if r else np.nan
        ee_rat = r.get('ee95_ratio', np.nan) if r else np.nan

        f = []
        if r:
            if d.get('collapsed'): f.append('COLLAPSED')
            if d.get('beta_railed'): f.append('BETA_RAIL')
            if d.get('rms_frac', 0) > 0.05: f.append('RESID')
            if d.get('slope_frac', 0) > 0.6: f.append('HALO')
            if d.get('q_fit', 0) > 0.4: f.append('Q_HI')
            if d.get('trunc_sens', 0) > 0.1: f.append('TRUNC')
            if np.isfinite(ratio) and not 0.9 <= ratio <= 1.1:
                f.append('RATIO')
            if np.isfinite(ee_rat) and not 0.95 <= ee_rat <= 1.05:
                f.append('EE_DIV')

        rows.append({
            'ang': a,
            'h_fit': r.get('hwhm_px', np.nan) if r else np.nan,
            'h_emp': r.get('h_emp_px', np.nan) if r else np.nan,
            'ratio': ratio,
            'beta': r['beta'] if r else np.nan,
            'redchi': d.get('redchi', np.nan),
            'ee_mof': r.get('ee95_radius_moffat', np.nan) if r else np.nan,
            'ee_emp': r.get('ee95_radius_emp', np.nan) if r else np.nan,
            'ee_rat': ee_rat,
            'flags': ','.join(f) if f else ('ok' if r else 'NO FIT'),
        })

    tbl = Table(rows=rows)
    for c in ('ang', 'h_fit', 'h_emp', 'ratio', 'beta', 'redchi',
              'ee_mof', 'ee_emp', 'ee_rat'):
        tbl[c].format = '.2f'
    tbl['ang'].unit = 'deg'
    for c in ('h_fit', 'h_emp', 'ee_mof', 'ee_emp'):
        tbl[c].unit = 'pix'
    tbl.pprint(max_lines=-1, max_width=-1)

    return {
        'camera_dist_mm': D, 'intercept_mm': r0,
        'input_angles': input_angles,
        'theta_peak_deg': theta_peak,

        'theta_out95_deg': theta_out95_moffat,
        'theta_out95_emp_deg': theta_out95_emp,

        'theta_d_deg': theta_d,
        'f_in': f_in,

        'f_out95': f_out95_moffat,
        'f_out95_emp': f_out95_emp,

        'sigma_deg': sigma_deg,
        'sigma_unaligned_deg': sigma_unaligned_deg,
        'sigma_deg_nom': unp.nominal_values(sigma_deg),
        'sigma_deg_std': unp.std_devs(sigma_deg),
        'hwhm_left_deg': hwhm_left_deg,
        'hwhm_right_deg': hwhm_right_deg,
        'med_hwhm_left_deg': med_hwhm_left_deg,
        'med_hwhm_right_deg': med_hwhm_right_deg,
        'flux_rel': flux_rel,
    }