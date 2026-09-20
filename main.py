import logging
import numpy as np
import uncertainties.unumpy as unp
from pathlib import Path

import input, analysis, postproc

logging.basicConfig(level=logging.WARN, format="%(levelname)s %(funcName)s: %(message)s")
logger = logging.getLogger(__name__)

if __name__ == '__main__':

    base_dir = Path(__file__).parent / 'data' # change to parent directory of image data files
    fiber_patterns = ['control-2*']  # directory names
    fiber_names = sorted({p.name for pat in fiber_patterns
                          for p in base_dir.glob(pat)
                          if p.is_dir() and not p.name.endswith("-bad")})
    if not fiber_names:
        raise SystemExit(f"No fiber directories found under {base_dir}")

    logging.getLogger().setLevel(logging.INFO)   # DEBUG, INFO, WARNING, ERROR, CRITICAL (sensitivity level tags)

    camera_dist_mm = 3
    plot_full_fibers = True

    pixel_size_mm = 2.4e-3
    (h, w) = (3660, 5480)

    all_results, frd_results = {}, {}
    angles_by_fiber = {}
    for fiber in fiber_names:
        logger.info(f"Processing {fiber}")
        fiber_dir = base_dir / fiber
        angle_files = np.array(sorted(int(p.stem) for p in fiber_dir.glob('*.tiff') if p.stem.isdigit()))
        input_angles = angle_files / 100.0
        angles_by_fiber[fiber] = input_angles
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
                (c_x, c_y), *_ = analysis.circlefinder(str(p), dark_path=master_dark,
                                                       use_cache=True, regenerate_cache=False)
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
                res = analysis.profile_analysis(str(p), dark_path=master_dark, shared_center=shared_center,
                                                use_cache=True, regenerate_cache=False)
                results.append(res)

            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"  [!] Skipped {p.name}: {e}")
                results.append(None)

        all_results[fiber] = results
        logger.info(f"{fiber} done")

        frd = analysis.calculate_frd(results, input_angles, pixel_size=pixel_size_mm, camera_dist_mm=camera_dist_mm)
        frd_results[fiber] = frd
        D = frd['camera_dist_mm']
        a0 = frd['intercept_mm']
        r0_deg = np.degrees(np.arctan(a0.nominal_value / D.nominal_value))

        logger.info(f"Plotting {fiber}")
        results, frd = all_results[fiber], frd_results.get(fiber)
        if frd is None:
            continue

        sel = results if plot_full_fibers else [results[2]]
        ang = input_angles if plot_full_fibers else [input_angles[2]]
        # for res, a in zip(sel, ang):
        #     if res is not None:
        #         postproc.plot_2d_contours(res, title=f"{fiber} {a}° 2D Annulus Contours", pixel_size_mm=pixel_size_mm,
        #                                   camera_dist_mm=camera_dist_mm,)

        postproc.plot_hwhm_channels(sel, ang, title_prefix=f"{fiber} ",
                                    show_ee=True, camera_dist_mm=camera_dist_mm)
        postproc.plot_stats(input_angles=input_angles,
                            sigma_deg=frd['sigma_deg_nom'], sigma_err=frd['sigma_deg_std'],
                            flux_rel=frd['flux_rel'], title_prefix=f"{fiber} FRD: ",
                            hwhm_left_deg=frd['hwhm_left_deg'],
                            hwhm_right_deg=frd['hwhm_right_deg'],
                            med_hwhm_left_deg=frd['med_hwhm_left_deg'],
                            med_hwhm_right_deg=frd['med_hwhm_right_deg'],
                            )
        postproc.plot_numerical_aperture(frd, title_prefix=f"{fiber} ")

    datasets = {f: (all_results[f], angles_by_fiber[f], frd_results.get(f))
                for f in fiber_names if any(r is not None for r in all_results.get(f, []))}

    shared = sorted(set.intersection(*(set(np.round(a, 2))
                                       for _, a, _ in datasets.values())))
    logger.info(f"Shared angles: {shared}")

    trimmed = {}
    for name, (res, ang, frd) in datasets.items():
        keep = [i for i, a in enumerate(np.round(ang, 2)) if a in shared]
        if frd is not None and frd.get('flux_rel') is not None:
            frd = dict(frd)
            frd['flux_rel'] = np.asarray(frd['flux_rel'])[keep]
        trimmed[name] = ([res[i] for i in keep], np.asarray(ang)[keep], frd)

    postproc.plot_spec_enclosed_flux(trimmed, na_start=0.15, na_stop=0.23, na_step=0.01, na_norm=0.30,
                                     pixel_size_mm=pixel_size_mm, camera_dist_mm=camera_dist_mm,
                                     title_prefix='Control vs stressed: ')
