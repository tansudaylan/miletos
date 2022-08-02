import sys
from tqdm import tqdm
import os
import numpy as np
import pickle

import miletos
import tdpy
from tdpy.util import summgene

import xarray as xr
import json

import astropy

import ephesus

'''
This example shows how miletos can be called to analyze and model JWST data. It uses WASP-39b as an example.
'''

# type of the data input to Miletos
## Miletos accepts data either as
### a list of time-series collected at different wavelengths
### an array of time-series collected at different wavelengths
typeinpt = 'arry'

# choose planetary system as the type of model for the time-series data
listtypemodl = ['psys']

# label for the Wavelength axis
lablwlen = r'Wavelength [$\mu$m]'

# path of the input stage 3 reduction
patherss = os.environ['MILETOS_DATA_PATH'] + '/data/JWST_ERS/WASP-39_Kirk_Stage3_v0/'

# flux
path = patherss + 'star1_flux_resampled.pickle'
objtfile = open(path, 'rb')
flux = pickle.load(objtfile, )

# flux error
path = patherss + 'star1_error_resampled.pickle'
objtfile = open(path, 'rb')
errr = pickle.load(objtfile)

# wavelength solution [um]
path = patherss + 'wvl_solution.pickle'
objtfile = open(path, 'rb')
wlen = pickle.load(objtfile)

# times [BJD]
path = patherss + 'BJD_TDB_time.pickle'
objtfile = open(path, 'rb')
time = pickle.load(objtfile) + 2.4e6

numbener = 3
wlen = wlen[:numbener]
flux = flux[:, :numbener]
errr = errr[:, :numbener]

# number of time samples
numbtime = time.size
# indices of the time samples
indxtime = np.arange(numbtime)

# number of wavelength samples
numbwlen = wlen.size
# indices of the wavelength samples
indxwlen = np.arange(numbwlen)

print('Number of wavelength bins in the file: %d' % numbwlen)

listarrytser = dict()
if typeinpt == 'arry':
    listarrytser['raww'] = [[[np.empty((numbtime, numbwlen, 3))]], []]
    listarrytser['raww'][0][0][0][:, :, 0] = time[:, None]
    listarrytser['raww'][0][0][0][:, :, 1] = flux
    listarrytser['raww'][0][0][0][:, :, 2] = errr
        
    medi = np.nanmedian(listarrytser['raww'][0][0][0][:int(time.size/10), :, 1], axis=0)
    listarrytser['raww'][0][0][0][:, :, 1:3] /= medi[None, :, None]

if typeinpt == 'list':
    listarrytser['raww'] = [[[np.empty((numbtime, 1, 3))] for e in indxwlen]]
    for k in tqdm(range(2, len(listhdun) - 1)):
        t = k - 2
        for e in indxwlen:
            listarrytser['raww'][0][e][0][t, 0, 0] = time[t]
            listarrytser['raww'][0][e][0][t, 0, 1] = listhdun[k].data['FLUX'][e]
            listarrytser['raww'][0][e][0][t, 0, 2] = listhdun[k].data['FLUX_ERROR'][e]
    
    print('Deleting wavelength bins, where light curve is all infinite...')
    listarrytsertemp = dict()
    listarrytsertemp['raww'] = [[]]
    for e in indxwlen:
        if (np.isfinite(listarrytser['raww'][0][e][0][:, 0, 1])).any():
            listarrytsertemp['raww'][0].append(listarrytser['raww'][0][e])
    listarrytser['raww'] = listarrytsertemp['raww']

# half of the wavelenth bin width
diffwlenhalf = (wlen[1:] - wlen[:-1]) / 2.

# type of iteration over energy bins
typemodlener = 'iter'
#typemodlener = 'full'

# a string used to search for the target on MAST
strgmast = 'WASP-39'

# type of inference
## 'opti': optimize
#typeinfe = 'opti'
typeinfe = 'samp'

# a string describing the miletos run
strgcnfg = 'ERS_JWST_inpt'

# instrument label for the data set
listlablinst = [[strgcnfg.split('_')[1]], []]
liststrgtypedata = [[strgcnfg.split('_')[2]], []]

strgcnfgfull = '%s_%s_%s' % (strgcnfg, typemodlener, typeinfe)

print('Rebinning the light curve...')
for e in indxwlen:
    temp = ephesus.rebn_tser(listarrytser['raww'][0][0][0][:, e, :], delt=1e-3)
    if e == 0:
        arrytser = np.empty((temp.shape[0], numbwlen, 3))
        arrytser[:, e, :] = temp
    else:
        arrytser[:, e, :] = ephesus.rebn_tser(listarrytser['raww'][0][0][0][:, e, :], delt=1e-3)
    print('arrytser[:, e, 2]')
    summgene(arrytser[:, e, 2])
listarrytser['raww'][0][0][0] = arrytser

dictoutp = miletos.main.init( \
                             strgmast=strgmast, \
                             
                             #labltarg=labltarg, \
                             
                             listlablinst=listlablinst, \
                             liststrgtypedata=liststrgtypedata, \

                             strgcnfg=strgcnfgfull, \
                             
                             # turn off LS periodogram and BLS analysis for estimating the priors
                             listtypeanls=[], \
                             
                             booldiag=False, \

                             # turn off detrending for estimating the priors
                             boolbdtr=False, \
                               
                             typeinfe=typeinfe, \

                             listarrytser=listarrytser, \
                               
                             typemodlener=typemodlener, \

                             listtypemodl=listtypemodl, \
                             
                             #boolbdtr=False, \

                             listener=wlen, \
                             lablener=lablwlen, \
                             
                             typeverb=0, \

                            )

if listlablinst[0][0] == 'JWST' and liststrgtypedata[0][0] == 'inpt':
    
    import astropy.units as u
    from astropy.utils.misc import JsonCustomEncoder

    # write the output to xarray for the ERS Team
    ds = xr.Dataset( \
                    data_vars=dict(
                                   transit_depth=(["central_wavelength"], dictoutp['perrrratcompspec'], {'units': '(R_jup*R_jup)/(R_jup*R_jup)'}),
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
                               
                               system_params=json.dumps({ 
                                                         'rp': 1*u.Unit('R_jup'), 
                                                         'rs':1*u.Unit('R_sun'), 
                                                         'a_rs': 0.1, 
                                                         'tc': 0.9
                                                        },
                                                        cls=JsonCustomEncoder),
                               
                               limb_darkening_params=json.dumps({'u1':1, 
                                                                 'u2':1,'u3':1, 'u4':1}) #optional in accordance with model runs
                              )
                   )
    
    
    ds.to_netcdf("fitted-light-lurve_target_W39_mode_G395H_code_miletos_author_Daylan.nc")



