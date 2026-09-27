# Miletos

## Scientific purpose
Miletos is a time-series analysis and forward-modeling pipeline for astrophysical systems. It is designed to interpret time-domain photometry and related observations by combining detrending, period searches, diagnostics, and Bayesian forward modeling.

## Scope
Miletos sits in the time-domain exoplanet and stellar-variability layer of the broader scientific ecosystem. It consumes observational time-series data, performs relevant preliminary analyses, and produces scientifically interpretable plots and summary outputs for subsequent inference.

## Installation

```bash
cd /path/to/miletos
python -m pip install -e .
export MILETOS_PATH=/path/to/miletos
```

`MILETOS_PATH` identifies the repository root. Runtime inputs belong under `data/` and generated pipeline outputs belong under `visuals/`. Both directories are ignored by Git. Existing deployments may continue to use `MILETOS_DATA_PATH` for an external data root while migrating.

## Minimal usage
Set `MILETOS_PATH` to the repository root and run the deterministic diagnostic:

```bash
python examples/simulated_transit_diagnostic.py --typefileplot png
```

The example injects an Ephesos transit into a deterministic simulated light curve, then uses Miletos to mask the transits during spline detrending, phase-fold the result, and calculate uncertainty-aware phase bins. It writes `visuals/simulated_transit_diagnostic.png`; use `--typefileplot pdf` for a vector figure.

![Miletos simulated transit diagnostic showing the input light curve, spline detrending, and phase-folded output](docs/simulated_transit_diagnostic.png)

The figure is generated from an explicitly simulated benchmark, not observational data. The reusable API is also available directly:

```python
from pathlib import Path

from miletos.diagnostics import run_simulated_transit_diagnostic

result = run_simulated_transit_diagnostic(Path("visuals/transit_diagnostic.png"))
```

## What the workflow does
Miletos can be used to:

- detrend and diagnose time-series measurements;
- perform period searches and candidate identification;
- inspect intermediate model-building diagnostics;
- compare forward models against observed data;
- produce final plots and stored summary products that make the modeling chain inspectable.

## Development status
Miletos remains a maintained research-grade pipeline rather than a broad generic library. The supported interface is the importable workflow and the documented model entry points; legacy or exploratory fragments should be treated as historical unless explicitly migrated into the active API.


## Input Data
The input time-series data can include photometry, spectroscopy, radial velocity, or astrometry. Examples are time-series data from the Transiting Exoplanet Survey Satellite (TESS) and JWST, Legacy Survey for Space and Time (LSST), radial velocity surveys such as HARPS, PFS, and NEID.


## Data gathering
Given a target, Miletos searches for time-series data using MAST (e.g., TESS, Kepler, HST, and JWST).


## Analyses
Miletos performs preliminary analyses such as detrending, phase-folding, producing Lomb-Scargle periodograms (via Astropy), and performing Box Least Squares (BLS) searches. The outcomes are plotted, written to disk, and returned to the user. They are also used as priors for subsequent generative modeling of the data.


## Model
Miletos is inherently a Bayesian framework that takes fair samples from the posterior probability distribution of the forward model. The suite of forward models is obtained via [Ephesos](https://github.com/tansudaylan/ephesos). These include potentially flaring or spotted stars with stellar, compact, or planetary companions; and exploding stars with companions.

Miletos allows the user to marginalize over the model parameters, including those that characterize limb darkening using an parametrization that is efficient to sample from (Kipping 2013).


### red noise
Data collected in the real Universe, unlike many of our simulations, contain features that are not drawn from, and hence cannot be explained by, our fitting models. This requires a prescription for modeling unknown components in a way that is minimally degenerate with the signal of interest. Miletos uses Gaussian Processes (GP) as implemented in [celerite](https://github.com/dfm/celerite) (Foreman-Mackey et al. 2017) to model the baseline of the time-series data to account for systematics in the form of red noise.


### priors
In order to determine the priors on the model parameters, Miletos either performs fast analyses on the time-series or fetches those priors from relevant databases.

When modeling exoplanetary systems (e.g., known exoplanets or TESS Objects of Interest) Miletos can either perform a box least squares (BLS) search to find candidates of transiting exoplanets and perform an Lomb-Scargle search to find radial-velocity candidates, querry the NASA Exoplanet Archive or the TOI catalog, respectively, to retrieve priors on the epoch of mid-transit time and orbital period.


## Implementation and performance
As a high-level pipeline, Miletos is written in Python 3. Model evaluations, which are the bottleneck of forward modeling, can be accelerated with just-in-time compilation or GPUs when necessary.


## Usage

### JWST Early Release Science (ERS)

Miletos's functionality has been significantly enhanced as part of the JWST Early Release Science (ERS) effort in order to provide the JWST exoplanet research community with a fast and robust analysis and modeling tool for time-series data from NIRSpec and NIRISS. Used in this mode, Miletos first performs a fit of the white light curve on a given target, obtaining the posterior on the system parameters. Then, it uses these posteriors as priors to the system parameters in subsequent fits to data at each wavelength. Note that Miletos does not perform Stage 1 or 2 reductions, which can be separately performed by the JWST pipeline. The functionality of Miletos is focused on the accurate modeling of the resulting spectral light curves.

When fitting spectral light curves over a wavelength interval, an important consideration is the modeling of limb darkening and how it changes with wavelength. In NIRSpec, the marginalization provided by Miletos is more important in modeling Prism data compard to NIRSpec 395H with a smaller wavelength coverage. 


### JWST ERS Observations of WASP-39b

Will be public soon...


### TESS Observations of WASP-121b

Here is an example usage of Miletos for analyzing the TESS data on an ultra-hot Jupiet, WASP-121b.

```
def cnfg_WASP0121():
    
    # a string that will appear as the base of the file names
    strgtarg = 'wasp0121'
    
    # a string that will be used to search for data on MAST 
    strgmast = 'WASP-121'
    
    # a string for labeling the target in the plots
    labltarg = strgmast
    
    # a string indicating the TOI number for querying initial conditions
    strgtoii = '495.01'
    
    # also do a phase curve analysis
    boolphascurv = True
    
    # call Miletos
    Miletos.main( \
                 strgtarg=strgtarg, \
                 strgmast=strgmast, \
                 labltarg=labltarg, \
                 strgtoii=strgtoii, \
                 boolphascurv=boolphascurv, \
                )
```

You can find more example uses of Miletos under the examples folder.
