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

# determine the type of the instrument based on the user input
## Options are:
### 'JWST_NIRCam'
### 'JWST_NIRSpec_Prism'
### 'JWST_NIRSpec_G395H_NRS1'
### 'JWST_NIRSpec_G395H_NRS2'
typeinst = sys.argv[1]

# read the data (wavelengths, times, relative fluxes, and relative flux errors)
#if typeinst == 'JWST_NIRCam':
#
#    # flux
#    path = patherss + 'NIRCam/star1_flux_resampled.pickle'
#    objtfile = open(path, 'rb')
#    flux = pickle.load(objtfile, )
#    
#    # flux error
#    path = patherss + 'NIRCam/star1_error_resampled.pickle'
#    objtfile = open(path, 'rb')
#    errr = pickle.load(objtfile)
#    
#    # wavelength solution [um]
#    path = patherss + 'NIRCam/wvl_solution.pickle'
#    objtfile = open(path, 'rb')
#    wlen = pickle.load(objtfile)
#    
#    # times [BJD]
#    path = patherss + 'NIRCam/BJD_TDB_time.pickle'
#    objtfile = open(path, 'rb')
#    time = pickle.load(objtfile) + 2.4e6
#
#elif typeinst.startswith('JWST_NIRSpec'):
#    
#    strg = typeinst.split('_')[3]
#    
#    path = patherss + 'NIRSpec/raw-binned-light-curves-W39-G395H-%s-10pix-custom-Alam_v4.xc' % strg 
#    
#    print('Reading from %s...' % path)
#    objt = xr.open_dataset(path)
#    
#    # wavelength solution [um]
#    wlen = objt['central_wavelength'].values
#    # flux
#    flux = objt['raw_flux'].T.values
#    # flux error
#    errr = objt['raw_flux_error'].T.values
#    
#    # times [BJD]
#    time = objt['time_flux'].values - 59000.
#
#else:
#    print('An argument is needed: JWST_NIRCAM, JWST_NIRSpec_Prism, JWST_NIRSpec_NRS1, or JWST_NIRSpec_NRS2.')
#    raise Exception('')

#numbener = 5
#wlen = wlen[:numbener]
#flux = flux[:, :numbener]
#errr = errr[:, :numbener]
#
## number of time samples
#numbtime = time.size
## indices of the time samples
#indxtime = np.arange(numbtime)
#
## number of wavelength samples
#numbwlen = wlen.size
## indices of the wavelength samples
#indxwlen = np.arange(numbwlen)
#
#print('Number of time bins in the file: %d' % numbtime)
#print('Number of wavelength bins in the file: %d' % numbwlen)
#
## put the data into a format suitable for miletos
#listarrytser = dict()
#listarrytser['raww'] = [[[np.empty((numbtime, numbwlen, 3))]], []]
#listarrytser['raww'][0][0][0][:, :, 0] = time[:, None]
#listarrytser['raww'][0][0][0][:, :, 1] = flux
#listarrytser['raww'][0][0][0][:, :, 2] = errr
#
## normalize the spectral light curve by the medians
#medi = np.nanmedian(listarrytser['raww'][0][0][0][:int(time.size/10), :, 1], axis=0)
#listarrytser['raww'][0][0][0][:, :, 1:3] /= medi[None, :, None]
#
## half of the wavelenth bin width
#diffwlenhalf = (wlen[1:] - wlen[:-1]) / 2.

# a string used to search for the target on MAST
strgmast = 'WASP-39'

# type of inference
## 'opti': find the maximum likelihood solution (much faster)
## 'samp': sample from the posterior (slow)
typeinfe = 'opti'

# instrument label for the data set
listlablinst = [[typeinst], []]

# type of data
liststrgtypedata = [['inpt'], []]

# a string describing the miletos run
strgcnfg = 'ERS_%s_inpt_%s' % (typeinst, typeinfe)

# Boolean flag to write over previous output files
boolwritover = False

# dictionary for the fitting model
dictfitt = dict()

# choose the type of forward-model for the time-series data
## in this case, the only model is 'psys', which stands for planetary system
dictfitt['listtypemodl'] = ['psys']

# make the fitting baseline 'step', which is a smooth step function
dictfitt['typemodlbase'] = 'step'

# type of iteration over energy bins
dictfitt['typemodlener'] = 'iter'

# list of strings indicating the experiments
liststrgexpr = ['JWST_NIRSpec']

# turn off LS periodogram and BLS spectrum analyses for estimating the priors
listtypeanls = []

# turn off light curve detrending for estimating the priors
boolbdtr = False

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
                             #liststrgtypedata=liststrgtypedata, \
                             strgcnfg=strgcnfg, \
                             listtypeanls=listtypeanls, \
                             boolbdtr=boolbdtr, \
                             liststrgexpr=liststrgexpr, \
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


# write the output to xarray for the ERS Team
ds = xr.Dataset( \
                data_vars=dict(
                               transit_depth=(["central_wavelength"], dictoutp['pmedrratcompspec'], {'units': '(R_jup*R_jup)/(R_jup*R_jup)'}),
                               transit_depth_error=(["central_wavelength"], dictoutp['perrrratcompspec'], {'units': '(R_jup*R_jup)/(R_jup*R_jup)'}),
                              ),
                
                coords=dict(
                            central_wavelength=(["central_wavelength"], wlen, {'units': 'micron'}),#required*
                            bin_half_width=(["bin_half_width"], diffwlenhalf, {'units': 'micron'})#required*
                           ),
                
                attrs=dict(
                           author="Tansu Daylan",
                           
                           contact="tansu.daylan@gmail.com",
                           
                           code="miletos, https://github.com/tdaylan/miletos",
                           
                           data_origin=json.dumps({
                                                   'stellar_spec': 'zenodo.org/xxxx',
                                                   'raw_light_curve': 'www.drive.google.com/someonegavemethis',
                                                   'fitted_light_curve': 'malam@carnegiescience.edu'
                                                 }),
                           
                           doi="",
                           
                          )
               )


path = patherss + 'Transmission_Spectrum_W39_ERS_%s_miletos_TansuDaylan.nc' % typeinst
print('Writing to %s...' % path)
ds.to_netcdf(path)



