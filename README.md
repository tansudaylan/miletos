# Miletos

## Introduction
Miletos is a pipeline to analyze and forward-model time-series data. Using these time-series data, it can be used to partially characterize (i.e., to the extent allowed by the information content in the time-series data at hand)
- systems of planets, stars, and compact objects including their
    - orbits (e.g., periods, semi-major axes, inclination, spin-orbit alignments), bulk structural features (e.g., radii, masses, densities), and surface brightness distributions of the planets and stars and
    - properties of spots and flares on the stars (e.g., sizes and evolution time scales);
- explosive astrophysical phenomena such as supernovae including
    - brightnening profile and the size of any companions.

It is written in python3.


## Input Data
The time-series data can include photometry, spectroscopy, radial velocity, or astrometry. Examples are time-series data from the Transiting Exoplanet Survey Satellite (TESS) and JWST, Legacy Survey for Space and Time (LSST), radial velocity surveys such as HARPS, PFS, and NEID.


## Analysis
The suite of analyses are Lomb-Scargle periodograms (via astropy) and Box Least Squares (BLS) via ephesus. The outcome of these analyses are used as priors for generative modeling of the data. But before that, the results of the analyses are plotted, written on the disc, and eventually returned to the user. 


## Model
The suite of models include potentially flaring or spotted stars with stellar, compact, or planetary companions; and exploding stars with companions. To forward-model transiting systems with potentially flaring or spotted stars, it uses Ephesus.

### red noise
Data collected in the real Universe, unlike many of our simulations, contain features and components that are not drawn from, and hence cannot be explained by, our fitting models. This requires a prescription for modeling unknown components in a way that is minimally degenerate with the signal of interest. Miletos uses Gaussian Processes (GP) as implemented in celerite to model the baseline of the time-series data to account for systematics in the form of red noise.


### priors
Miletos either performs fast analyses on the time-series to obtain priors on the model parameters or fetches those priors from relevant data bases.

When modeling exoplanetary systems (e.g., known exoplanets or TESS Objects of Interest) Miletos can either perform a box least squares (BLS) search to find candidates of transiting exoplanets and perform an Lomb-Scargle search to find radial-velocity candidates, querry the NASA Exoplanet Archive or the TOI catalog, respectively, to retrieve priors on the epoch of mid-transit time and orbital period.



## Usage
Miletos is an end-to-end pipeline that takes its inputs and configuration parameters from the user in a single function call, fetches the data, performs the relevant analyses and modeling, plots and saves the results to the disk, and returns a dictionary that contains all of the relevant intermediate and output variables. Therefore, running Miletos requires a single function even though the arguments to the function make it possible to extensively customize the behavior of the pipeline.

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
