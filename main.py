import logging
import re
import numpy as np
import uncertainties.unumpy as unp
from pathlib import Path

import input, analysis, postproc

logging.basicConfig(level=logging.WARN, format="%(levelname)s %(funcName)s: %(message)s")
logger = logging.getLogger(__name__)

if __name__ == '__main__':

    base_dir = Path(__file__).parent / 'data' # change to parent directory of image data files
    fiber_names = sorted(p.name for p in base_dir.glob('control-2*') # directory/directories of focus
                         if p.is_dir() and not p.name.endswith("-bad"))
    if not fiber_names:
        raise SystemExit(f"No fiber directories found under {base_dir}")

    logging.getLogger().setLevel(logging.INFO)   # DEBUG, INFO, WARNING, ERROR, CRITICAL (sensitivity level tags)

    camera_dist_mm = 10.6
    plot_full_fibers = True

    pixel_size_mm = 2.4e-3
    (h, w) = (3660, 5480)

    all_results, frd_results = {}, {}
    for fiber in fiber_names:
        logger.info(f"Processing {fiber}")
        fiber_dir = base_dir / fiber
        m = re.search(r'(\d+(?:\.\d+)?)mm', fiber.replace('_', '-'))
        camera_dist_mm = float(m.group(1)) if m else 10.6
        angle_files = np.array(sorted(
            int(p.stem) for p in fiber_dir.glob('*.tiff') if p.stem.isdigit()))
        input_angles = angle_files / 100.0
        logger.info(f"  [*] Found input angles: {input_angles}")
        master_dark, _ = input._master_reduction(fiber_dir, prefix="dark")
        if master_dark is None:
            logger.warning(f"{fiber}: no dark frames found, proceeding uncalibrated")
        paths = [fiber_dir / f"{a}.tiff" for a in angle_files]

        # pass 1: shared center
        centers = []
        for p in paths:
            logger.info(f"circlefinder: {p.name}")
            try:
                (c_x, c_y), *_ = analysis.circlefinder(str(p), dark_path=master_dark)
                centers.append((c_x, c_y))
            except (FileNotFoundError, ValueError) as e:
                logger.debug(f"Center pass skipped {p.name}: {e}")
        if not centers:
            logger.warning(f"  [!] Skipped {fiber}: no valid centers")
            all_results[fiber] = [None] * len(paths)
            continue
        shared_center = tuple(np.median(np.array(centers), axis=0))
        logger.info(f"  [*] Shared center: {shared_center}")

        # pass 2: full analysis
        results = []
        for p in paths:
            logger.info(f"Analyzing {p.name}")
            try:
                res = analysis.profile_analysis(str(p), dark_path=master_dark,
                                       shared_center=shared_center)
                results.append(res)

            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"  [!] Skipped {p.name}: {e}")
                results.append(None)

        all_results[fiber] = results

        logger.info(f"{fiber} done")

        frd = analysis.calculate_frd(results, input_angles,
                                     pixel_size=pixel_size_mm,
                                     camera_dist_mm=camera_dist_mm)
        frd_results[fiber] = frd
        D = frd['camera_dist_mm']
        a0 = frd['intercept_mm']
        r0_deg = np.degrees(np.arctan(a0.nominal_value / D.nominal_value))
        print(f"  Calibrated distance: {D.nominal_value:.3f} ± {D.std_dev:.3f} mm "
              f"(nominal {camera_dist_mm} mm); zero-angle footprint"
              f"{a0.nominal_value * 1e3:.1f} µm ({r0_deg:.2f}° divergence)")

        logger.info(f"Plotting {fiber}")
        results, frd = all_results[fiber], frd_results.get(fiber)
        if frd is None:
            continue
        sel = results if plot_full_fibers else [results[2]]
        ang = input_angles if plot_full_fibers else [input_angles[2]]
        for res, a in zip(sel, ang):
            if res is not None:
                postproc.plot_2d_annulus_contours(res, title=f"{fiber} {a}° 2D Annulus Contours")

        postproc.plot_hwhm_channels(sel, ang, title_prefix=f"{fiber} ",
                                    show_ee=True, camera_dist_mm=camera_dist_mm)
        postproc.plot_stats(input_angles=input_angles,
                            sigma_deg=frd['sigma_deg_nom'], sigma_err=frd['sigma_deg_std'],
                            flux_rel=frd['flux_rel'], title_prefix=f"{fiber} FRD: ",
                            hwhm_left_deg=frd['hwhm_left_deg'],
                            hwhm_right_deg=frd['hwhm_right_deg'],
                            med_hwhm_left_deg=frd['med_hwhm_left_deg'],
                            med_hwhm_right_deg=frd['med_hwhm_right_deg'],
                            theta_peak_deg=unp.nominal_values(frd['theta_peak_deg']),
                            )
        postproc.plot_eccentricity(results, input_angles, title_prefix=f"{fiber} ")


