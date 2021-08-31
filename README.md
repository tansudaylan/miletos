# miletos

## Introduction

miletos is a pipeline to find and characterize stellar companions, spots, and flares by analyzing and modeling time-series photometric and radial velocity data.

It querries the NASA Exoplanet Archive and the TOI catalog to retrieve orbital priors.


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
