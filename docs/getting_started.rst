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

Sanity checks
-------------

``miletos.init`` accepts ``boolsanity=True`` to check loaded time series,
companion orbital parameters, photometric model fluxes, and phase-folded arrays.
The additional checks are disabled by default. The existing ``booldiag`` option
controls Miletos's earlier structural and execution diagnostics independently.