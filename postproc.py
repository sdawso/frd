import logging

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

    logger.debug(f"Starting setup for {title_prefix}")
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
        logger.debug("Rendering 'sigma' subplot...")
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
        logger.debug("Rendering 'hwhm' subplot...")
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
        logger.debug("Rendering 'flux' subplot...")
        ax = axes['flux']
        ax.plot(input_angles, flux_rel, 'k--', alpha=0.5)
        ax.scatter(input_angles, flux_rel, marker='s', color='orchid', edgecolors='purple',
                   zorder=5, label='Total Flux (Norm)')
        ax.set(xlabel='Input angle (°)', ylabel='Relative Flux (Normalized)',
               title=f'{title_prefix}Throughput', ylim=(0, 1.1))
        ax.legend()

    if 'histogram' in plots:
        logger.debug("Rendering 'histogram' subplot...")
        ax = axes['histogram']
        ax.hist(image.ravel(), bins=256, color='dimgray', edgecolor='black', linewidth=0.5)
        ax.set_yscale('log')
        ax.set(xlabel='Pixel Intensity (ADU)', ylabel='Count (Log Scale)',
               title=f'{title_prefix}Intensity')

    logger.debug("Showing plot (block=False)...")
    plt.show(block=False)


def plot_image(result):
    """Plots the raw image with the fitted ellipse at the fitted ring radius."""
    logger.debug("Initializing figure rendering...")
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

    logger.debug("Displaying image...")
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
    logger.debug(f"Setting up subplots for {title_prefix}")

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
            p95_x = np.interp(0.95, y_ee, x_ee_deg)
            ax_ee.plot(p95_x, 0.95, marker, markersize=5, label=f'{name} 95% ({p95_x:.2f}°)')

        xlim_max_ee_deg = _px_to_deg(ax2_x1 or r_mean * 2.0, pixel_size_mm, camera_dist_mm)
        ax_ee.axhline(0.95, color='gray', linestyle=':')
        ax_ee.set(xlabel='Output Angle (degrees)', ylabel='Fraction of Total Flux',
                  title=f'{title_prefix}{ang}° Combined EE Profiles',
                  xlim=(ax2_x0 or 0, xlim_max_ee_deg))
        ax_ee.legend()

    logger.debug("Rendering final figure...")
    plt.show()
    plt.close(fig)


def plot_2d_contours(result, title="2D Planar Intensity with Contours", save_path=None):

    logger.debug(f"Starting 2D planar rendering for '{title}'...")

    image = result['image']
    center, axes, angle = result['center'], result['axes'], result['angle']

    r_peak = result['raw_peak_r']
    hwhm = result['h_emp_px']

    ee_emp = result.get('ee_empirical')
    ee_r_px = result.get('ee_r_px')

    r_ee95 = result.get('ee95_radius_emp')
    if r_ee95 is None or np.isnan(r_ee95):
        r_ee95 = result.get('ee95_radius_moffat')

    r_ee15 = None
    if ee_emp is not None and ee_r_px is not None:
        r_ee15 = float(np.interp(0.15, ee_emp, ee_r_px))

    radii_dict = {
        'EE95 (Outer)': (r_ee95, 'red', '--'),
        'HWHM (Outer)': (r_peak + hwhm, 'cyan', '-.'),
        'Peak Crest': (r_peak, 'green', '-'),
        'HWHM (Inner)': (r_peak - hwhm, 'cyan', '-.'),
        'EE15 (Inner)': (r_ee15, 'magenta', '--'),
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

    if save_path:
        logger.debug(f"Saving to {save_path}...")
        plt.savefig(save_path)

    logger.debug("Displaying plot and cleaning up memory...")
    plt.show(block=False)


def plot_3d_intensity(image, title="3D Pixel Intensity Map", save_path=None, mesh_profiles=None,
                      theta_centers=None, r_coords_master=None, center=None, axes=None, angle=None):

    logger.debug(f"Starting 3D rendering for '{title}'")
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

    logger.debug("Generating surface plot data geometry...")
    stride = max(1, image.shape[0] // 100)
    surf = ax.plot_surface(X[::stride, ::stride], Y[::stride, ::stride], image[::stride, ::stride],
                           cmap='magma', edgecolor='none', alpha=0.9)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Intensity')
    ax.set(title=title + " (Raw)", xlabel='X Pixel', ylabel='Y Pixel', zlabel='Intensity')

    if extrude:
        logger.debug(f"Generating {num_slices} azimuthally extruded radial profile subplots...")

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
        logger.debug(f"Saving to {save_path}...")
        plt.savefig(save_path)

    logger.debug("Displaying plot and cleaning up memory...")
    plt.show(block=False)
    plt.close(fig)
