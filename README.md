Fiber optic signal analysis for characterization of focal ratio degradation in a wide array of fibers, 
Intended to be paired with optical table capable of producing a collimated beam annulus on a bare CCD sensor of choice.

[An example of conducting a collimated beam FRD test (not exactly this setup but conceptually similar)](https://opg.optica.org/viewmedia.cfm?r=1&rwjcode=ao&URI=ao-55-25-6829&seq=0&origin=search)

***Be sure to build the uv environment before running the function on the included test datafile, unless you want to install everything yourself***

### circlefinder/
* main.py - execution file, most tweaks happen here
* input.py - intake and reduction functions
* analysis.py - annular contour generation, elliptical change-of-basis and crest data analysis, radial profile generation. meat and potatoes file
* postproc.py - a bunch of matplotlib functions, some useful some not. functions of importance are plot_hwhm_channels and plot_stats. everything else is largely experimental


Anthropic's Claude and Google's Gemini were used for generating matplotlib functions and general debugging passes. All code drafted by me (Simon) unless otherwise credited.
