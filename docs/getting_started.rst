Getting started
===============

Miletos processes photometric and radial-velocity time series with a shared
analysis pipeline. The package supports file-backed and supplied arrays,
transit and phase-curve models, period searches, and plotting.

Installation
------------

Install Miletos and its declared dependencies from the repository root:

.. code-block:: console

   python -m pip install -e .

Set ``MILETOS_PATH`` to the repository root. Runtime data are read from its
``data/`` directory and pipeline figures are written below the selected target
in ``examples/`` or ``visuals/``.

.. code-block:: console

   export MILETOS_PATH=/path/to/miletos

First run
---------

The self-contained simulated-transit example runs without downloading
observations:

.. code-block:: console

   python examples/simulated_transit/run.py --typefileplot png

For an observed target, the TOI-1233 example reads the cached TESS Sector 10
photometry and PFS radial velocities. The command-line options are available
without starting an analysis:

.. code-block:: console

   python examples/TOI-1233/run.py --help

The example notebooks display these workflows and their generated figures. The
individual target guides document their inputs, model choices, and any required
network access.

Stellar activity analysis
------------------------

The Hattusa analysis functions now live in ``miletos.stellar_activity`` and
call Ephesos for forward predictions. Use ``parse_spot_flare_parameters`` to
unpack six stellar entries, three entries per spot (five for evolving spots),
and four entries per flare. Times use days and angles use degrees. The Gaussian
likelihood accepts independent flux uncertainties. The radius-ordering prior
and likelihood can be used separately for posterior inference.

``fit_spot_flare_light_curve`` performs a fixed-count, bounded local fit from
supplied initial parameters. It does not select spot multiplicity or provide
posterior uncertainty. Stellar spot geometry can be degenerate.

.. code-block:: console

   python examples/stellar_activity/run.py --typefileplot png

This seeded synthetic example fits an explicitly assumed flare and flux offset.
It contains no spots and does not constrain stellar rotation. Models with spots
use Ephesos's native spherical-cap integration and need no optional spot engine.

``select_observed_variable_stars`` selects aligned catalog rows from one-based
TESS-sector coverage flags. For the optional toy period-regression experiment:

.. code-block:: console

   python -m pip install -e '.[rotation_ml]'

.. code-block:: python

   from miletos import run_rotation_regression_benchmark

   products = run_rotation_regression_benchmark()

The random forest is trained on synthetic sinusoidal curves. Both subsets use
the same noise prescription, and returned predictions concern held-out targets.
No convolutional-network implementation or observational performance is claimed.

Sanity checks
-------------

``miletos.init`` accepts ``boolsanity=True`` to check loaded time series,
companion orbital parameters, photometric model fluxes, and phase-folded arrays.
The additional checks are disabled by default. The existing ``booldiag`` option
controls Miletos's earlier structural and execution diagnostics independently.