import logging
import uncertainties.unumpy as unp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import analysis

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(funcName)s: %(message)s")
logger = logging.getLogger(__name__)

def plot_stats(input_angles, sigma_deg=None, sigma_err=None,
               hwhm_left_deg=None, hwhm_right_deg=None,
               flux_rel=None, image=None, title_prefix="",
               med_hwhm_left_deg=None, med_hwhm_right_deg=None):

    _hwhm_sides = {
        'left': [-1, 'dodgerblue', 'navy', '<'],
        'right': [+1, 'crimson', 'darkred', '>']
    }

    plots = []
    if sigma_deg is not None: plots.append('sigma')
    if hwhm_left_deg is not None or hwhm_right_deg is not None: plots.append('hwhm')
    if flux_rel is not None: plots.append('flux')
    if image is not None: plots.append('histogram')

    ncols = len(plots)
    if ncols == 0: return

    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 5), tight_layout=True)
    axes = np.atleast_1d(axes)
    axes = {name: axes[i] for i, name in enumerate(plots)}

    if 'sigma' in plots:
        ax = axes['sigma']

        if flux_rel is not None and sigma_err is not None:
            valid = (~np.isnan(sigma_deg) & ~np.isnan(sigma_err)
                     & ~np.isnan(flux_rel) & (sigma_err > 0))
            x_val, y_val = input_angles[valid], sigma_deg[valid]

            if len(x_val) > 2:
                w = (1.0 / sigma_err[valid]) * flux_rel[valid] ** 2

                coeffs, cov = np.polyfit(x_val, y_val, deg=1, w=w, cov=True)
                m, c = coeffs
                m_err = np.sqrt(cov[0, 0])

                x_fit = np.linspace(x_val.min(), x_val.max(), 100)
                ax.plot(x_fit, m * x_fit + c, 'r-', lw=2, zorder=4,
                        label=f'Fit Slope = {m:3e} ± {m_err:.3e}°/°')

        if sigma_err is not None:
            ax.errorbar(input_angles, sigma_deg, yerr=sigma_err, fmt='D', color='steelblue',
                        capsize = 3, label = 'Moffat σ')
        else:
            ax.scatter(input_angles, sigma_deg, marker='D', color='steelblue', label = 'Moffat σ')

        ax.set(xlabel='Input angle (°)', ylabel='Ring σ (°)', title=f'{title_prefix}Broadening')
        ax.legend()

    if 'hwhm' in plots:
        ax = axes['hwhm']
        x_unique = np.asarray(input_angles)
        pk = np.zeros(len(x_unique))

        for (hwhm_deg, (side, (sign, color, edge, marker))) in zip((hwhm_left_deg, hwhm_right_deg), _hwhm_sides.items()):
            if hwhm_deg is None:
                continue

            x_pts, y_pts, band_lo, band_hi = [], [], [], []
            for i, (ang, deltas) in enumerate(zip(input_angles, hwhm_deg)):
                if deltas is not None and len(deltas) > 0:
                    band_lo.append(pk[i] + sign * min(deltas))
                    band_hi.append(pk[i] + sign * max(deltas))
                    x_pts.extend([ang] * len(deltas))
                    y_pts.extend(pk[i] + sign * np.asarray(deltas, dtype=float))
                else:
                    band_lo.append(np.nan)
                    band_hi.append(np.nan)

            band_lo, band_hi = np.array(band_lo), np.array(band_hi)
            valid = ~np.isnan(band_lo)
            if np.any(valid):
                ax.fill_between(x_unique[valid], band_lo[valid], band_hi[valid],
                                color=color, alpha=0.2, zorder=2)
                for band in (band_lo, band_hi):
                    ax.plot(x_unique[valid], band[valid], linestyle='--',
                            color=color, linewidth=1, zorder=3)

            if x_pts:
                ax.scatter(x_pts, y_pts, marker=marker, color=color, edgecolors=edge,
                           zorder=5, label=f'{side} HWHM Δr')

        for (med_deg, (side, (sign, _, _, _))) in zip((med_hwhm_left_deg, med_hwhm_right_deg), _hwhm_sides.items()):
            if med_deg is None:
                continue
            valid = ~np.isnan(med_deg)
            if not np.any(valid):
                continue
            y_med = pk[valid] + sign * med_deg[valid]
            ax.plot(x_unique[valid], y_med, 'g--', linewidth=1.5, zorder=6)
            ax.scatter(x_unique[valid], y_med, marker='s', color='green',
                       edgecolors='darkgreen', zorder=7, label=f'Med. {side} HWHM')

        if sigma_deg is not None:
            valid_s = ~np.isnan(sigma_deg)
            if np.any(valid_s):
                xs = np.asarray(input_angles)[valid_s]
                ys = np.asarray(sigma_deg)[valid_s]
                pks = pk[valid_s]
                ye = np.asarray(sigma_err)[valid_s] if sigma_err is not None else None
                for sign, lbl in ((+1, 'Moffat σ'), (-1, None)):
                    ax.plot(xs, pks + sign * ys, 'b-', linewidth=1.8, zorder=8)
                    ax.errorbar(xs, pks + sign * ys, yerr=ye, fmt='o', color='blue',
                                markeredgecolor='navy', markersize=4,
                                capsize=2, zorder=9, label=lbl)

        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, zorder=1)
        ax.set(xlabel='Input angle (°)', ylabel='HWHM angle (° from aligned peak)',
               title=f'{title_prefix}Wavelet HWHM')
        ax.legend()

    if 'flux' in plots:
        ax = axes['flux']
        ax.plot(input_angles, flux_rel, 'k--', alpha=0.5)
        ax.scatter(input_angles, flux_rel, marker='s', color='orchid', edgecolors='purple',
                   zorder=5, label='Total Flux (Norm)')
        ax.set(xlabel='Input angle (°)', ylabel='Relative Flux (Normalized)',
               title=f'{title_prefix}Throughput', ylim=(0, 1.1))
        ax.legend()

    if 'histogram' in plots:
        ax = axes['histogram']
        ax.hist(image.ravel(), bins=256, color='dimgray', edgecolor='black', linewidth=0.5)
        ax.set_yscale('log')
        ax.set(xlabel='Pixel Intensity (ADU)', ylabel='Count (Log Scale)',
               title=f'{title_prefix}Intensity')

    plt.show(block=False)


def plot_image(result):
    """Plots the raw image with the fitted ellipse at the fitted ring radius."""
    img = result['image']
    cx, cy = result['center']
    ax_a, ax_b = result['axes']
    img_angle = result['angle']
    r_mean = result['result'].params['center'].value

    fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
    ax.imshow(img, cmap='magma', origin='lower')
    ax.set_title('Far-Field Annulus & Ellipse Fit')

    scale = r_mean / ((ax_a + ax_b) / 4)
    ax.add_patch(Ellipse((cx, cy), ax_a * scale, ax_b * scale, angle=img_angle,
                         color='red', fill=False, lw=1.5, linestyle='--'))

    plt.show()
    plt.close(fig)


def plot_eccentricity(results, angles, title_prefix=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=True)

    scatter = [r['peak_scatter'] if r else np.nan for r in results]
    colors = plt.cm.tab10(np.linspace(0, 1, len(angles)))
    ax1.bar([f'{a:.0f}°' for a in angles], scatter, color=colors)
    ax1.set(xlabel='Input angle (°)', ylabel='Ring-peak scatter (px, std)',
            title=f'{title_prefix}Azimuthal wobble')

    for r, ang in zip(results, angles):
        if not r or not r.get('slice_data'):
            continue
        pts = [(s['theta'], s['raw_peak_r']) for s in r['slice_data']
               if s.get('raw_peak_r') is not None]
        if pts:
            th, rp = zip(*pts)
            ax2.plot(th, rp, 'o-', alpha=0.6, label=f'{ang:.0f}°')
    ax2.set(xlabel='Azimuth (°)', ylabel='Ring-peak radius (px)',
            title=f'{title_prefix}Peak radius vs. azimuth')
    ax2.legend(fontsize=7)
    return fig


def plot_hwhm_channels(results, angles, pixel_size_mm=2.4e-3, camera_dist_mm=10,
                       ax1_x0=None, ax1_x1=None, ax2_x0=None, ax2_x1=None,
                       title_prefix="", show_ee=True):

    valid_data = [(res, ang) for res, ang in zip(results, angles) if res]
    N = len(valid_data)
    if N == 0:
        return

    nrows = 2 if show_ee else 1
    fig, axes = plt.subplots(nrows, N, figsize=(8 * N, 8 * nrows), tight_layout=True)
    axes = np.atleast_1d(axes).reshape(nrows, N)

    def _plot_deltas(ax, peak_r, deltas, sign, x_px, y_data, color, **kwargs):
        if deltas is None or len(deltas) == 0 or peak_r is None:
            return False
        x_cross = peak_r + sign * np.asarray(deltas, dtype=float)
        ax.plot(x_cross, np.interp(x_cross, x_px, y_data), color, **kwargs)
        return True

    for i, (result, ang) in enumerate(valid_data):
        logger.debug(f"Processing subplot for angle {ang}...")
        ax_rad = axes[0, i]

        r_mean = result['result'].params['center'].value
        raw_peak_r = result['raw_peak_r']

        x_px = result['r_px']
        y_data = result['avg_data']

        for si, s in enumerate(result.get('slice_data', [])):
            s_x_px, s_y_data, s_peak_r = s['r_px'], s['prof'], s['raw_peak_r']

            ax_rad.plot(s_x_px, s_y_data, color='goldenrod', alpha=0.6, linewidth=1, zorder=1,
                        label='Individual Slices' if si == 0 else None)

            if s_peak_r is not None:
                ax_rad.plot(s_peak_r, np.interp(s_peak_r, s_x_px, s_y_data),
                            'k.', markersize=4, alpha=0.4, zorder=2)

            for sign, fmt in ((-1, 'b.'), (+1, 'm.')):
                key = 'left_deltas' if sign < 0 else 'right_deltas'
                _plot_deltas(ax_rad, s_peak_r, s.get(key), sign, s_x_px, s_y_data, fmt,
                             markersize=3, alpha=0.4, zorder=2)

        ax_rad.plot(x_px, y_data, color='green', lw=2, zorder=5, label='Azimuthal Average')
        ax_rad.plot(x_px, result['result'].eval(x=x_px), color='red', linestyle='--', lw=2,
                    zorder=5, label='Pseudo-Voigt Fit')

        ax_rad.plot(raw_peak_r, np.interp(raw_peak_r, x_px, y_data),
                    'k*', markersize=8, zorder=10, label='Raw Peak (Avg)')

        for sign, fmt, lbl in ((-1, 'bo', 'HWHM (Avg)'), (+1, 'mo', None)):
            key = 'left_deltas' if sign < 0 else 'right_deltas'
            _plot_deltas(ax_rad, raw_peak_r, result.get(key), sign, x_px, y_data, fmt,
                         markersize=6, zorder=10, label=lbl)

        master_peak = int(np.argmax(y_data))
        left_deltas, right_deltas, _ = analysis._hwhm_crossings(
            y_data, x_px, master_peak, debounce_px=20, bin_step=5.0
        )

        peak_val = y_data[master_peak]
        for sign, deltas, floor_slice, lbl in (
                (-1, left_deltas, y_data[:master_peak + 1], 'HWHM (Median)'),
                (+1, right_deltas, y_data[master_peak:], None)):
            if len(deltas) == 0:
                continue
            floor = np.min(floor_slice) if len(floor_slice) > 1 else 0
            thresh = floor + 0.5 * (peak_val - floor)
            ax_rad.plot(x_px[master_peak] + sign * deltas[0], thresh,
                        'gs', markersize=6, zorder=12, label=lbl)

        ax_rad.set(xlabel='Radius (pixels)', ylabel='Intensity',
                   title=f'{title_prefix}{ang}° Radial Profile',
                   xlim=(ax1_x0 or 0, ax1_x1 or r_mean * 2.5))
        ax_rad.legend()

        if not show_ee:
            continue

        ax_ee = axes[1, i]
        ee_curves = (
            ('ee', 'ee_r_theoretical', 'blue', '-', 'Moffat', 'bo'),
            ('ee_empirical', 'ee_r_px', 'purple', '--', 'Empirical', 'ro'),
        )

        def _px_to_deg(r_px, psmm=pixel_size_mm, cdmm=camera_dist_mm):
            return np.degrees(np.arctan((np.asarray(r_px, dtype=float) * psmm) / cdmm))

        for y_key, x_key, color, style, name, marker in ee_curves:
            y_ee, x_ee_px = result.get(y_key), result.get(x_key)
            if y_ee is None or x_ee_px is None:
                continue
            x_ee_deg = _px_to_deg(x_ee_px, pixel_size_mm, camera_dist_mm)
            ax_ee.plot(x_ee_deg, y_ee, color=color, lw=2, linestyle=style, label=f'{name} EE')
            p85_x = np.interp(0.85, y_ee, x_ee_deg)
            ax_ee.plot(p85_x, 0.85, marker, markersize=5, label=f'{name} 85% ({p85_x:.2f}°)')

        xlim_max_ee_deg = _px_to_deg(ax2_x1 or r_mean * 2.0, pixel_size_mm, camera_dist_mm)
        ax_ee.axhline(0.85, color='gray', linestyle=':')
        ax_ee.set(xlabel='Output Angle (degrees)', ylabel='Fraction of Total Flux',
                  title=f'{title_prefix}{ang}° Combined EE Profiles',
                  xlim=(ax2_x0 or 0, xlim_max_ee_deg))
        ax_ee.legend()

    plt.show()
    plt.close(fig)

def plot_numerical_aperture(frd, n_index=1.0, title_prefix="", na_ref = 0.22):

    """experimental characterization of NA internal flux relative to a wide NA normal flux"""

    input_angles = np.asarray(frd['input_angles'], dtype=float)

    def _na(theta_deg):
        return n_index * unp.sin(unp.radians(theta_deg))

    fig, ax = plt.subplots(figsize=(7, 6), tight_layout=True)

    theta_peak = frd['theta_peak_deg']
    na_peak = _na(theta_peak)
    na_peak_nom, na_peak_err = unp.nominal_values(na_peak), unp.std_devs(na_peak)

    ax.plot(input_angles, na_peak_nom, '--', color='black', lw=1.5,
                label='Peak NA (median)')

    sigma_deg = frd['sigma_deg']
    theta_peak_nom = unp.nominal_values(theta_peak)
    na_mof_outer = unp.nominal_values(_na(theta_peak + sigma_deg))
    na_mof_inner = unp.nominal_values(_na(theta_peak - sigma_deg))

    ax.plot(input_angles, na_mof_outer, '--', color='crimson', lw=1.5,
            label='Moffat σ NA (outer)')

    ax.plot(input_angles, na_mof_inner, '--', color='dodgerblue', lw=1.5,
            label='Moffat σ NA (inner)')

    med_left = np.asarray(frd['med_hwhm_left_deg'], dtype=float)
    med_right = np.asarray(frd['med_hwhm_right_deg'], dtype=float)

    valid_l = ~np.isnan(med_left)
    valid_r = ~np.isnan(med_right)

    if np.any(valid_l):
        na_hwhm_inner = n_index * np.sin(np.radians(theta_peak_nom[valid_l] - med_left[valid_l]))
        ax.plot(input_angles[valid_l], na_hwhm_inner, ':', color='navy', lw=1.5,
                label='Empirical HWHM NA (inner)')
    if np.any(valid_r):
        na_hwhm_outer = n_index * np.sin(np.radians(theta_peak_nom[valid_r] + med_right[valid_r]))
        ax.plot(input_angles[valid_r], na_hwhm_outer, ':', color='darkred', lw=1.5,
                label='Empirical HWHM NA (outer)')

    theta_out85 = frd['theta_out85_deg']
    theta_out15 = frd['theta_out15_deg']
    na_ee85 = _na(theta_out85)
    na_ee15 = _na(theta_out15)
    na_ee85_nom, na_ee85_err = unp.nominal_values(na_ee85), unp.std_devs(na_ee85)
    na_ee15_nom, na_ee15_err = unp.nominal_values(na_ee15), unp.std_devs(na_ee15)

    ax.errorbar(input_angles, na_ee85_nom, yerr=na_ee85_err, fmt='--', color='purple',
                capsize=3, label='85% EE NA (median)')

    ax.errorbar(input_angles, na_ee15_nom, yerr=na_ee15_err, fmt='--', color='indigo',
                capsize=3, label='15% EE NA (median)')

    if na_ref is not None:
        ax.axhline(y=na_ref, color='gray', linestyle='--', lw=1.5, label=f'Fiber NA spec ({na_ref:.2f})')
    ax.set(xlabel='Input angle (°)', ylabel='Numerical Aperture (NA)',
           title=f'{title_prefix}Ring NA vs. Input Angle')

    ax.legend(fontsize=8)
    plt.show(block=False)

def _ee_curve(res, prefer='empirical'):
    order = [('ee_empirical', 'ee_r_px'), ('ee', 'ee_r_theoretical')]
    if prefer == 'moffat': order = order[::-1]
    for ykey, xkey in order:
        ee, rp = res.get(ykey), res.get(xkey)
        if ee is None or rp is None: continue
        ee, rp = np.asarray(ee, float).ravel(), np.asarray(rp, float).ravel()
        if ee.size != rp.size: continue
        m = np.isfinite(ee) & np.isfinite(rp)
        if m.sum() < 3: continue
        o = np.argsort(rp[m])
        return ee[m][o], rp[m][o]
    return None, None


def spec_enclosed_flux(results, angles, na_values, frd=None, n_index=1.0,
                       pixel_size_mm=2.4e-3, camera_dist_mm=13.0, d_tol=0.10,
                       use_frd_geometry=True, prefer='empirical', na_norm=None,
                       flux_rel_min=0.5, weight_by_flux=False):
    from uncertainties import ufloat

    na_values = np.atleast_1d(np.asarray(na_values, float))
    na_norm = float(np.max(na_values)) if na_norm is None else float(na_norm)

    if frd is not None and use_frd_geometry:
        D = frd.get('camera_dist_mm', ufloat(camera_dist_mm, d_tol * camera_dist_mm))
        r0 = frd.get('intercept_mm', 0.0)
    else:
        D, r0 = ufloat(camera_dist_mm, d_tol * camera_dist_mm), 0.0

    tan_ref = np.tan(np.arcsin(np.clip(np.append(na_values, na_norm) / n_index, -1.0, 1.0)))
    r_ref = np.atleast_1d((r0 + D * tan_ref) / pixel_size_mm)
    r_nom = np.array([getattr(v, 'nominal_value', v) for v in r_ref])
    r_sig = np.array([getattr(v, 'std_dev', 0.0) for v in r_ref])

    flux_rel = (np.asarray(frd['flux_rel'], float)
                if frd is not None and frd.get('flux_rel') is not None else None)

    angles = np.asarray(angles, float)
    f = np.full((na_values.size, angles.size), np.nan)
    f_err = np.full_like(f, np.nan)

    for i, res in enumerate(results):
        if res is None: continue
        if flux_rel is not None and np.isfinite(flux_rel[i]) and flux_rel[i] < flux_rel_min: continue
        ee, rp = _ee_curve(res, prefer)
        if ee is None: continue
        v = np.array([np.where(r <= rp[-1], np.interp(r, rp, ee), np.nan)
                      for r in (r_nom, r_nom - r_sig, r_nom + r_sig)])
        v = v[:, :-1] / v[:, -1:]
        w = (flux_rel[i] if (weight_by_flux and flux_rel is not None
                             and np.isfinite(flux_rel[i])) else 1.0)
        f[:, i] = w * v[0]
        f_err[:, i] = w * 0.5 * np.abs(v[2] - v[1])

    return {'na_values': na_values, 'angles': angles, 'f': f, 'f_err': f_err,
            'na_norm': na_norm}

def plot_spec_enclosed_flux(datasets, na_ref=0.22, na_start=None, na_stop=None,
                            na_step=None, na_values=None, n_index=1.0,
                            pixel_size_mm=2.4e-3, camera_dist_mm=13.0, d_tol=0.10,
                            use_frd_geometry=True, prefer='empirical', na_norm=None,
                            title_prefix="", show_input_limit=True, ncols=None,
                            flux_rel_min=0.5, weight_by_flux=False):

    if na_values is None:
        if na_start is None and na_stop is None and na_step is None:
            na_values = np.atleast_1d(na_ref)
        else:
            a, b = (0.10 if na_start is None else na_start), (na_ref if na_stop is None else na_stop)
            s = 0.02 if na_step is None else na_step
            na_values = np.arange(a, b + 0.5 * s, s)
    na_values = np.atleast_1d(np.asarray(na_values, float))
    n_na = na_values.size

    ncols = ncols or min(3, n_na)
    nrows = int(np.ceil(n_na / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.6 * nrows),
                             tight_layout=True, squeeze=False, sharex=True, sharey=True)
    axes = axes.ravel()
    for ax in axes[n_na:]: ax.set_visible(False)

    colors = plt.get_cmap('viridis')(np.linspace(0.05, 0.85, max(len(datasets), 1)))
    out = {}

    for (label, entry), color in zip(datasets.items(), colors):
        results, angles, frd = entry if (isinstance(entry, tuple) and len(entry) == 3) else (*entry, None)
        r = spec_enclosed_flux(results, angles, na_values, frd=frd, n_index=n_index,
                               pixel_size_mm=pixel_size_mm, camera_dist_mm=camera_dist_mm,
                               d_tol=d_tol, use_frd_geometry=use_frd_geometry, prefer=prefer,
                               na_norm=na_norm, flux_rel_min=flux_rel_min,
                               weight_by_flux=weight_by_flux)
        out[label] = r
        for k in range(n_na):
            axes[k].errorbar(r['angles'], r['f'][k], yerr=r['f_err'][k], fmt='o-',
                             color=color, capsize=3, lw=1.5, label=label)

    for k, na in enumerate(na_values):
        theta = np.degrees(np.arcsin(np.clip(na / n_index, -1.0, 1.0)))
        axes[k].axhline(1.0, color='black', lw=1.0, alpha=0.5)
        if show_input_limit:
            axes[k].axvline(theta, color='gray', ls='--', lw=1.5)
        axes[k].set(title=f'NA = {na:.3f}  ({theta:.1f}°)', ylim=(0, 1.05))
        axes[k].grid(alpha=0.3)
        if k % ncols == 0: axes[k].set_ylabel('Flux fraction inside NA')
        if k >= n_na - ncols: axes[k].set_xlabel('Input angle (°)')

    axes[0].legend(fontsize=8, loc='lower left')
    fig.suptitle(f'{title_prefix}Spec-enclosed flux vs. input angle', y=1.0)
    plt.show(block=False)
    return out


def plot_2d_contours(result, title="2D Planar Intensity with Contours",
                     na_ref=0.22, pixel_size_mm=2.4e-3, camera_dist_mm=10.0, n_index=1.0):

    logger.debug(f"Starting 2D rendering for '{title}'...")

    image = result['image']
    center, axes, angle = result['center'], result['axes'], result['angle']

    r_peak = result['raw_peak_r']
    hwhm = result['h_emp_px']

    r_ee85 = result.get('ee85_radius_emp')
    if r_ee85 is None or np.isnan(r_ee85):
        r_ee85 = result.get('ee85_radius_moffat')

    r_ee15 = result.get('ee85_emp')
    if r_ee15 is None or np.isnan(r_ee15):
        r_ee15 = result.get('ee15_moffat')

    r_na_ref = None
    if na_ref is not None:
        theta_ref = np.arcsin(na_ref/n_index)
        r_na_ref = camera_dist_mm * np.tan(theta_ref)/ pixel_size_mm

    radii_dict = {
        'EE85 (Outer)': (r_ee85, 'red', '--'),
        'HWHM (Outer)': (r_peak + hwhm, 'cyan', '-.'),
        'Peak Crest': (r_peak, 'green', '-'),
        'HWHM (Inner)': (r_peak - hwhm, 'cyan', '-.'),
        'EE15 (Inner)': (r_ee15, 'magenta', '--'),
        f'NA={na_ref:.2f} spec': (r_na_ref, 'gray', '-'),
    }

    fig, ax = plt.subplots(figsize=(10, 8), tight_layout=True)
    im = ax.imshow(image, cmap='magma', origin='upper', interpolation='none')
    fig.colorbar(im, ax=ax, shrink=0.7, label='Intensity')
    ax.set(title=title, xlabel='X Pixel', ylabel='Y Pixel')

    theta_dense = np.linspace(0, 360, 720)
    for label, (R, color, style) in radii_dict.items():
        if R is None or not np.isfinite(R):
            continue
        x_contour, y_contour = analysis._ring_xy_coordinates(R, theta_dense, center, axes, angle)
        ax.plot(x_contour, y_contour, color=color, linestyle=style, lw=2,
                label=f"{label} (R={R:.1f})")

    ax.legend(loc='upper right')

    a, b = axes[0] / 2, axes[1] / 2
    max_r = max(R for R, _, _ in radii_dict.values() if R is not None and np.isfinite(R))
    pad = max_r * max(a, b) / ((a + b) / 2) + 50

    h, w = image.shape[:2]
    ax.set_xlim(max(0, center[0] - pad), min(w, center[0] + pad))
    ax.set_ylim(min(h, center[1] + pad), max(0, center[1] - pad))

    plt.show(block=False)


def plot_3d_intensity(image, title="3D Pixel Intensity Map", save_path=None, mesh_profiles=None,
                      theta_centers=None, r_coords_master=None, center=None, axes=None, angle=None):

    X, Y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))

    extrude = mesh_profiles is not None and center is not None
    if extrude:
        num_slices = len(mesh_profiles)
        ncols = 3
        nrows = int(np.ceil((num_slices + 1) / ncols))
        fig = plt.figure(figsize=(6 * ncols, 5 * nrows), tight_layout=True)
        ax = fig.add_subplot(nrows, ncols, 1, projection='3d')
    else:
        fig = plt.figure(figsize=(10, 8), tight_layout=True)
        ax = fig.add_subplot(111, projection='3d')

    stride = max(1, image.shape[0] // 100)
    surf = ax.plot_surface(X[::stride, ::stride], Y[::stride, ::stride], image[::stride, ::stride],
                           cmap='magma', edgecolor='none', alpha=0.9)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Intensity')
    ax.set(title=title + " (Raw)", xlabel='X Pixel', ylabel='Y Pixel', zlabel='Intensity')

    if extrude:
        theta_dense = np.linspace(0, 360, 100)
        R, THETA = np.meshgrid(r_coords_master, theta_dense)
        x_reconstructed, y_reconstructed = analysis._ring_xy_coordinates(R, THETA, center, axes, angle)

        for i, profile_1d in enumerate(mesh_profiles):
            ax_slice = fig.add_subplot(nrows, ncols, i + 2, projection='3d')
            slice_angle = theta_centers[i] if theta_centers is not None else (i * (360.0 / num_slices))

            ax_slice.plot_surface(x_reconstructed, y_reconstructed,
                                  np.tile(profile_1d, (len(theta_dense), 1)),
                                  cmap='magma', edgecolor='none', alpha=0.9)
            ax_slice.set(title=f"Extruded Slice @ {slice_angle:.1f}°",
                         xlabel='X Pixel', ylabel='Y Pixel', zlabel='Intensity')

    if save_path:
        plt.savefig(save_path)

    plt.show(block=False)
    plt.close(fig)
