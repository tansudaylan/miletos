import sys
from tqdm import tqdm
import os
import numpy as np

import miletos
import tdpy
from tdpy.util import summgene

import xarray as xr
import json

import astropy

'''
This example shows how miletos can be called to analyze and model JWST data. It uses WASP-39b as an example.
'''

# type of the run
## 'SimGenWhite': using simulated data produced by injecting a transit to a flat baseline and adding only white noise
## 'SimGenRed': using simulated data produced by injecting a transit to a flat baseline and adding both red and white noise
## 'ERS': using the ERS data
strgtyperuns = sys.argv[1]

# number of wavelength bins
numbwlen = 10
# indices of the wavelength bins
indxwlen = np.arange(numbwlen)
#listlablinst = [[], []]
#for k in indxwlen:
#    listlablinst[0].append('JWST_NIRSpec_G395H_%04d' % k)
#    listlablinst[0].append('JWST_NIRSpec_G395H_%04d' % k)

# instrument label for the data set
listlablinst = [['JWST_NIRSpec_G395H'], []]

# type of the data input to Miletos
## Miletos accepts data either as
### a list of time-series collected at different wavelengths
### an array of time-series collected at different wavelengths
typeinpt = 'list'

# choose planetary system as the type of model for the time-series data
listtypemodl = ['psys']

# label for the Wavelength axis
lablwlen = r'Wavelength [$\mu$m]'

# values of the wavelength axis [um]
listwlen = np.linspace(0.6,  5., 2)

if strgtyperuns == 'simuerss':
    patherss = os.environ['DATA'] + '/other/ERS/'
    path = patherss + 'NIRSpec/Stage2/jwdata0010010_11010_0001_NRS1_uncal_updatedHDR_MOD_injected_x1dints.fits'
    
    listarrytser = dict()
    
    #tdpy.read_fits(path)
    
    listhdun = astropy.io.fits.open(path)
    
    ds_ex = xr.open_dataset("stellar_spectra_target_FAKE_mode_G395H_code_eureka.nc")

    #time = listhdun[1].data['int_mid_BJD_TDB']
    #print('time')
    #summgene(time)
    numbtime = len(listhdun) - 3
    time = np.arange(numbtime).astype(float)
    
    wlen = listhdun[2].data['WAVELENGTH']
    
    print('len(listhdun)')
    print(len(listhdun))
    print('numbtime')
    print(numbtime)
    
    numbwlen = wlen.size
    indxwlen = np.arange(numbwlen)
    
    print('Number of wavelength bins in the file: %d' % numbwlen)

    if typeinpt == 'arry':
        listarrytser['raww'] = [[[np.empty((numbtime, numbwlen, 3))]]]
        for k in range(2, len(listhdun) - 1):
            t = k - 2
            listarrytser['raww'][0][0][0][t, :, 0] = listhdun[k].data['EXPMID']
            listarrytser['raww'][0][0][0][t, :, 1] = listhdun[k].data['FLUX']
            listarrytser['raww'][0][0][0][t, :, 2] = listhdun[k].data['FLUX_ERROR']
            
        medi = np.nanmedian(listarrytser['raww'][0][0][0][:, :, 1], axis=0)
        print('medi')
        summgene(medi)
        print('listarrytser[raww][0][0][0]')
        summgene(listarrytser['raww'][0][0][0])
        listarrytser['raww'][0][0][0][:, :, 1:3] /= medi[None, :, None]

        print('listarrytser[raww][0][0][0][:, :, 0]')
        summgene(listarrytser['raww'][0][0][0][:, :, 0])
        print('listarrytser[raww][0][0][0][:, :, 1]')
        summgene(listarrytser['raww'][0][0][0][:, :, 1])
        print('listarrytser[raww][0][0][0][:, :, 2]')
        summgene(listarrytser['raww'][0][0][0][:, :, 2])
    
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

        numbwlen = len(listarrytser['raww'][0])
        indxwlen = np.arange(numbwlen)
        for e in indxwlen:
            print('e: %d' % e)
            print('listarrytser[raww][0][e][0][:, 0, 0]')
            summgene(listarrytser['raww'][0][e][0][:, 0, 0])
            print('listarrytser[raww][0][e][0][:, 0, 1]')
            summgene(listarrytser['raww'][0][e][0][:, 0, 1])
            print('listarrytser[raww][0][e][0][:, 0, 2]')
            summgene(listarrytser['raww'][0][e][0][:, 0, 2])
            print('')
    
    print('Number of wavelength bins remaining: %d' % numbwlen)

    strgmast = None
    labltarg = 'WASP-39'

if strgtyperuns == 'SimGeneWhite':
    strgmast = 'WASP-39'
    listarrytser = None
    labltarg = None
    
dictoutp = miletos.main.init( \
                  strgmast=strgmast, \
                  labltarg=labltarg, \
                  strgtyperuns=strgtyperuns, \
                  
                  listarrytser=listarrytser, \

                  listtypemodl=listtypemodl, \
                  
                  #boolbdtr=False, \

                  listener=listwlen, \
                  lablener=lablwlen, \

                  listlablinst=listlablinst, \
                 )


# write the output to xarray for the ERS Team

ds = xr.Dataset( \
                data_vars=dict(
                               transit_depth=(["central_wavelength"], transit_depth, {'units': '(R_jup*R_jup)/(R_jup*R_jup)'}),
                               
                               transit_depth_error=(["central_wavelength"], transit_depth_error, {'units': '(R_jup*R_jup)/(R_jup*R_jup)'}),
                              ),
                
                coords=dict(
                            central_wavelength=(["central_wavelength"], central_wavelength,{'units': 'micron'}),#required*
                            
                            bin_half_width=(["bin_half_width"], bin_half_width,{'units': 'micron'})#required*
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
                                                     'a_rs':0.1, 
                                                     'tc':0.9
                                                    },
                                                    cls=JsonCustomEncoder),
                           
                           limb_darkening_params=json.dumps({'u1':1, 'u2':1,'u3':1, 'u4':1}) #optional in accordance with model runs
                          )
               )


ds.to_netcdf("fitted-light-lurve_target_W39_mode_G395H_code_miletos_author_Daylan.nc")



