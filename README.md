# miletos

## Introduction
miletos is a pipeline to analyze and forward-model time-series data. It can be used to characterize orbital parameters, stellar companions, spots, and flares.


## Input Data
The time-series data include photometric and radial velocity. It is designed to be used on time-series data from the Transiting Exoplanet Survey Satellite (TESS) and James Webb Space Telescope (JWST), radial velocity surveys such as HARPS, PFS, and NEID.


## Analysis
The suite of analyses are Lomb-Scargle periodograms (via astropy) and Box Least Squares (BLS) via ephesus. 


## Model
The suite of models include potentially flaring or spotted stars with stellar, compact, or planetary companions; and exploding stars with companions.

It uses Gaussian Process (GP) to model the baseline of the time-series data to account for systematics in the form of red noise.


### priors
For known exoplanets or TESS Objects of Interest, miletos querries the NASA Exoplanet Archive or the TOI catalog, respectively, to retrieve priors on the epoch of mid-transit time and orbital period.



## Usage
Here is an example usage of miletos for analyzing the TESS data on an ultra-hot Jupiet, WASP-121b.

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
    
    # call miletos
    miletos.main( \
                 strgtarg=strgtarg, \
                 strgmast=strgmast, \
                 labltarg=labltarg, \
                 strgtoii=strgtoii, \
                 boolphascurv=boolphascurv, \
                )
```
