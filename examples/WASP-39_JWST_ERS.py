import sys
import os
import numpy as np
import pickle

import xarray as xr
import json

import miletos

'''
This example shows how miletos can be called to analyze and model JWST data.
It uses the Early Release Science (ERS) data collected on WASP-39b as an example.
'''

# label for the wavelength axis
lablwlen = r'Wavelength [$\mu$m]'

# path of the input stage 3 reduction
## Please modify this path to reflect the folder where the relevant data files exist
patherss = os.environ['DATA'] + '/other/JWST_ERS/WASP-39b/'

# a string used to search for the target on MAST
strgmast = 'WASP-39'

# type of inference
## 'opti': find the maximum likelihood solution (much faster)
## 'samp': sample from the posterior (slow)
typeinfe = 'opti'

# type of data
liststrgtypedata = [['inpt'], []]

# a string describing the miletos run
strgcnfg = 'ERS_obsd_%s' % (typeinfe)

# Boolean flag to write over previous output files
boolwritover = False

# dictionary for the fitting model
dictfitt = dict()

# choose the type of forward-model for the time-series data
## in this case, the only model is 'psys', which stands for planetary system
dictfitt['typemodl'] = 'psys'

# make the fitting baseline 'step', which is a smooth step function
dictfitt['typemodlbase'] = 'step'

# type of iteration over energy bins
dictfitt['typemodlener'] = 'iter'

# instrument label for the data set
listlablinst = [['NIRSpec G395H NRS1', 'NIRSpec G395H NRS2', 'NIRSPec PRISM'], []]

# list of strings indicating the experiments
liststrginst = [['JWST_nirspec_f290lp-g395h-s1600a1-sub2048_NRS1', \
                 'JWST_nirspec_f290lp-g395h-s1600a1-sub2048_NRS2', \
                 'JWST_nirspec_clear-prism-s1600a1-sub512'], []]

# turn off LS periodogram and BLS spectrum analyses for estimating the priors
listtypeanls = []

# turn off light curve detrending for estimating the priors
boolbdtr = None#

# turn on neglecting the lowest likelihood data sample (outlier rejection)
boolrejeoutlllik = True

# priors for WASP-39b
#pericompprio = np.array([4.0552941]) # [days] Macini+2018
#rsmacompprio = np.array([(14.34 / gdat.dictfact['rsre'] + 0.939) / (0.04828 * gdat.dictfact['aurs'])]) (0.103125)
#epocmtracompprio = np.array([791.112])
#cosicompprio = np.array([0.0414865])

dictoutp = miletos.main.init( \
                             strgmast=strgmast, \
                             listlablinst=listlablinst, \
                             liststrginst=liststrginst, \
                             #liststrgtypedata=liststrgtypedata, \
                             strgcnfg=strgcnfg, \
                             listtypeanls=listtypeanls, \
                             boolbdtr=boolbdtr, \
                             typeinfe=typeinfe, \
                             boolrejeoutlllik=boolrejeoutlllik, \
                             boolwritover=boolwritover, \
                             dictfitt=dictfitt, \
                             booldiag=True, \
                             #pericompprio=pericompprio, \
                             #epocmtracompprio=epocmtracompprio, \
                             #rsmacompprio=rsmacompprio, \
                             #cosicompprio=cosicompprio, \

                             #listarrytser=listarrytser, \
                             #typemodlener=typemodlener, \
                             #listener=wlen, \
                             #lablener=lablwlen, \
                            )





