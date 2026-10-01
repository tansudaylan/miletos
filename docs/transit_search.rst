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
measurement. Its illustrative injection grid tests a 0.1 day box duration over
periods and depths; it is not a planet-population prior.

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

Inverted curves provide a negative-control distribution for instrumental or
astrophysical signals that resemble transits. This distribution is not by itself
a calibrated false-discovery rate. A survey false-discovery estimate also needs
an independently labeled validation sample or an explicit model for how the
negative control represents the false-alarm population.

Completeness and occurrence
---------------------------

Miletos injects box-shaped transits at random epochs into the observed flux and
repeats BLS. A recovery requires both period agreement within the configured
tolerance and a fitted depth SNR above the stated threshold. Completeness is the
recovery fraction over injected epochs. The pilot grid uses a 7-SNR threshold;
larger surveys should increase injection counts and test transit duration,
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

Scientific context
------------------

The TESS Faint-star Search reported 1,617 TOIs from the Primary Mission after
independent vetting of QLP transit-search results, extending vetting beyond the
bright-star regime emphasized in early follow-up (Kunimoto et al. 2022). A later
study showed that inverted TESS light curves contain structured false alarms in
BLS searches (Kunimoto et al. 2023). Miletos makes these ideas reproducible in a
single workflow by keeping search outputs, negative controls, injection
recoveries, and target-list provenance together. QLP users should cite the QLP
products and the relevant reduction papers, as well as the Faint-star Search
when comparing its catalog-level results.
