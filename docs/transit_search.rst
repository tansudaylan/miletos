TESS transit search and vetting
===============================

Miletos searches public Quick-Look Pipeline (QLP) light curves from the
Transiting Exoplanet Survey Satellite (TESS) using Astropy Box Least Squares
(BLS). QLP products are retrieved from the Mikulski Archive for Space Telescopes
(MAST) by exact TIC identifier and sector. The FITS reader applies QLP quality
flags, normalizes the selected aperture flux, converts BJD to BTJD, and retains
sector and target identifiers.

A target-catalog CSV must contain integer ``tic_id`` and ``sector`` columns. It
may also contain ``tmag``, ``toi``, ``reference_disposition``, and
``sample_is_parent_population``. Each row is one TIC-sector search. The flag
``sample_is_parent_population`` must be true only when the catalog contains the
complete target selection used to define the survey denominator. A list of TOIs
alone is a validation subset and cannot define an occurrence-rate denominator.

Run a supplied catalog with:

.. code-block:: console

   python -m miletos.tess_transit_search --catalog path/to/targets.csv

The command retrieves QLP FITS products, searches each available curve, records
missing products and search failures, writes candidate and injection-recovery
CSV files, and generates separate yield, completeness, inverted-signal control,
magnitude, and TOI-disposition figures. ``--offline`` reads FITS products from
``--data-directory`` without querying MAST. Set ``--injection-trials`` to control
the number of independent epoch draws in each period-depth cell.

One real-data pilot is available with:

.. code-block:: console

   python examples/tess_transit_search/run_pilot.py

It analyzes TOI 1338.01 (TIC 260128333), Sector 10, using the public QLP FITS
light curve. Its outputs are saved in
``examples/tess_transit_search/visuals/``. The pilot is a single-target software
and vetting demonstration, not a survey-level completeness or occurrence
measurement. Its illustrative injection grid uses a fixed 0.1-day box duration
across periods and depths. It assigns no prior to planet populations.

For a retrospective comparison with the archived TOI process, run:

.. code-block:: console

   python examples/tess_transit_search/run_toi_validation.py

The example selects 50 Sector 10 TOI entries in the 10.5--13.5 TESS-magnitude
range from the local 2020-09-16 ExoFOP snapshot, collapses duplicate TICs, and
searches the 49 unique targets with cached QLP products. Add ``--download`` to
retrieve missing curves from MAST. The current run retrieves 48 curves, passes
17 through initial triage, and retains 31 for review. Because the sample is
selected from previously identified TOIs, these counts compare workflows and
do not measure precision, recall, completeness, or occurrence.

.. figure:: ../examples/tess_transit_search/visuals/tess_toi1338_sector10_target_search_20261026_055259.png
   :alt: Observed Sector 10 QLP light curve and folded BLS review signal
   :width: 100%

   The strongest positive-flux BLS event is weaker than the inverted-light-curve
   control and is retained for review.

.. figure:: ../examples/tess_transit_search/visuals/tess_toi1338_sector10_completeness_20261026_055259.png
   :alt: Injection recovery versus transit depth for the observed pilot target
   :width: 100%

   Eight injections per depth cell demonstrate the recovery calculation for one
   target. Population completeness requires a larger parent sample and injection
   grid.

Search and vetting
------------------

The initial search uses Miletos's Astropy BLS implementation and reports the
highest-SNR periodic box in the configured period range. The vetter records the
odd-even depth difference, the signal near secondary phase, and the strongest
BLS signal in an inverted flux series. Signals failing a metric remain in the
``review`` disposition. These tests are triage measurements, not an automated
astrophysical validation. Pixel-level localization, centroid shifts, dilution,
stellar multiplicity, and human follow-up remain necessary for candidate
validation.

Inverted curves provide a comparison sample of instrumental or astrophysical
signals that can resemble transits. This sample alone does not estimate a
false-discovery rate. A survey estimate also needs an independently labeled
validation sample or a model describing how inverted curves represent false
alarms.

Native Albus-compatible analysis
--------------------------------

Miletos does not depend on Lightkurve. Normal workflows use NumPy, Astropy,
and Astroquery. ``miletos.light_curve.LightCurve`` stores time, measured flux,
and uncertainties. It supports column access, selection, normalization,
Astropy sigma clipping, folding, and binning with independent-error propagation.
Missing uncertainties remain missing.

``miletos.photometry.prepare_light_curve_collection`` accepts measured array
columns. It removes NaNs, normalizes each curve and its uncertainties, removes
outliers, stitches sectors, and subtracts a 501-sample median filter. Its
``injection`` option centers the flux on zero rather than one. Normalized
errors are retained. The previously unused flatten calculation is omitted.

``miletos.box_least_squares.fixed_grid_box_least_squares`` returns an Astropy
BLS result using Albus's 2,048 logarithmic trial periods from 1 to 10 days
and duration trials equal to one percent of those periods. It retains the
likelihood objective and best-fit attributes used by Albus. Missing errors
disable weighting for the entire fit, matching the historical convention.
Native fit failures propagate. ``search_box_least_squares`` retains its
separate automatic grid and SNR objective.

``select_lightkurve_bls_candidates`` in the same module returns the raw-power
maximum followed by up to three median-detrended peaks outside the inclusive
0.8--1.2 period interval around each previously selected peak. The first power
remains raw and subsequent powers are detrended, matching the Albus convention.
Returned indices align each candidate with its fitted epoch and duration.

``miletos.transit_search.transit_mask`` converts Astropy time quantities to
days and absolute ``Time`` objects to Julian dates. Numeric inputs must share
a day-based origin. Transit boundaries are excluded by default, as in Albus.
``include_edges=True`` retains Miletos's inclusive repeating-search convention.
Periods must be finite and positive, epochs finite, and durations finite and
nonnegative. Zero duration returns an empty mask with the default convention.

``preprocess_tess_light_curve`` in ``miletos.photometry`` retrieves TESS
light-curve FITS products directly through MAST. The native reader supports
SPOC and QLP flux columns and applies the default TESS quality bitmask.
Failed or empty downloads return ``None``. Query and FITS parsing failures
propagate. Retrieved absolute times use BJD in the TDB scale.
``lightkurve_bls_results`` and ``fold_lightkurve`` in
``miletos.box_least_squares`` provide Albus-compatible periodogram and folded
plots, including binned curves, fitted candidate ephemerides, white and dark
themes, and the existing return conventions.

Older Lightkurve-named functions remain compatibility names for native
operations. They neither import Lightkurve nor call its curve methods.
The BLS and light-curve return types are now native Astropy/Miletos objects.

Lightkurve is imported only by the explicit comparison function
``miletos.benchmarks.compare_lightkurve_bls``. Install Lightkurve separately
to run that benchmark. It compares both implementations on identical data
and trial grids and reports their results and elapsed times. Normal imports,
preprocessing, searches, folds, and injections work with Lightkurve unavailable.

``miletos.transit_metrics`` contains the shared depth, shape, scatter, SNR,
fitted-parameter extraction, and quantile-depth measurements. The simple SNR
divides mean depth by out-of-transit population scatter without a sample-count
factor. The shape statistic retains Albus's flux thresholds over the supplied
time span. The quantile estimator uses period-normalized phase and reports
its clipped baseline scatter and quantile-weighted support count. None of
these measurements replaces the fitted BLS depth SNR.

``miletos.simulation`` provides the shared bounded ``sample_power_law``,
``generate_batman_light_curve``, ``inject_batman_transit``, and
``minimum_transit_inclination`` functions. Sampling retains NumPy's global
random stream. The BATMAN model retains the original Albus circular,
uniform-disk assumptions and zero-duration placeholder, with no exposure
integration. Radii are solar radii, stellar mass is solar masses, time is
days, inclination is degrees, and optional semimajor axis is meters.
``semimajor_axis_from_period`` takes solar masses, while
``semimajor_axis_from_mass_kg`` retains the archived Albus kilogram convention.

Completeness and occurrence
---------------------------

Miletos injects box-shaped transits at random epochs into the observed flux and
repeats BLS. A recovery requires period agreement within the configured
tolerance, a fitted depth SNR above the stated threshold, and passing the
odd-even and secondary-event vetting. Completeness is the
recovery fraction over injected epochs. The pilot grid uses a 7-SNR threshold.
Larger surveys should increase injection counts and test transit duration,
stellar variability, sector gaps, and the adopted candidate-vetting procedure.

For a complete parent target sample, Miletos can combine per-target recovery
efficiencies with ``pergamon.estimate_occurrence_rate`` and
``pergamon.log_likelihood_occurrence_rate``. By default, it averages
completeness uniformly over the discrete period-depth injection grid. This
corresponds to a uniform discrete occurrence distribution over the tested grid.
Supply ``injection_grid_weights`` when the scientific occurrence domain requires
another distribution. The estimate applies only to that stated domain and to
systems satisfying the at-most-one-event Bernoulli model in Pergamon. TOI-only
catalogs must not be used as the denominator.

Occurrence tables must contain one row per unique TIC. Multiple sectors need
a justified combined per-target selection model rather than repeated target
counts. The parent-population, searched, and detection flags accept only
Boolean or numeric 0/1 values. Each target needs exactly one finite completeness
fraction between zero and one per period-depth cell. Injection target identifiers
must match the parent sample, and inverted-control dispositions must be
``candidate`` or ``review``. Missing values, duplicate records, and unknown
control labels are rejected before inference.

Scientific context
------------------

The TESS Faint-star Search reported 1,617 TOIs from the Primary Mission after
independent vetting of QLP transit-search results, extending vetting beyond the
bright-star regime emphasized in early follow-up (Kunimoto et al. 2022). A later
study showed that inverted TESS light curves contain structured false alarms in
BLS searches (Kunimoto et al. 2023). Miletos makes these ideas reproducible in a
single workflow by keeping search outputs, inverted-curve comparisons,
injection recoveries, and target-list provenance together. QLP users should cite the QLP
products and the relevant reduction papers, as well as the Faint-star Search
when comparing its catalog-level results.
