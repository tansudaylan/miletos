import os
import pandas as pd
from tdpy.util import summgene
import matplotlib.pyplot as plt
import numpy as np

pathbase = os.environ['PEXO_DATA_PATH'] + '/'
pathimag = pathbase + 'imag/'

path = pathbase + 'data/exofop_toilists.csv'
print('Reading from %s...' % path)
objtexof = pd.read_csv(path, skiprows=0)
print(type(objtexof))
k = 0
l = 0
indx = np.where((objtexof['Planet Radius (R_Earth)'] < 2.) & (objtexof['Planet Equil Temp (K)'] < 370.) & (objtexof['Stellar Eff Temp (K)'] > 1000.))[0]
print(indx)
print('Earth-like around the Sun-like')
print(objtexof['TOI'].to_numpy()[indx])
print('Planet Radius (R_Earth)')
print(objtexof['Planet Radius (R_Earth)'].to_numpy()[indx])
print('Planet Equil Temp (K)')
print(objtexof['Planet Equil Temp (K)'].to_numpy()[indx])
print('Stellar Eff Temp (K)')
print(objtexof['Stellar Eff Temp (K)'].to_numpy()[indx])


for strgfrst in objtexof:
    for strgseco in objtexof:
        if strgfrst != 'Planet Equil Temp (K)':
            continue
        if strgseco == 'Planet Equil Temp (K)':
            continue

        if (objtexof[strgfrst].dtype == np.dtype('object')):
            continue
        if (objtexof[strgseco].dtype == np.dtype('object')):
            continue
        figr, axis = plt.subplots()
        axis.scatter(objtexof[strgfrst], objtexof[strgseco], s=1, color='k')
        axis.set_xlabel(strgfrst)
        axis.set_ylabel(strgseco)
        axis.set_xlim([None, 373])
        for n in range(len(objtexof['TOI'][indxhabi])):
            axis.text(objtexof[strgfrst][indxhabi][n], objtexof[strgfrst][indxhabi][n], '%f' % objtexof['TOI'][indxhabi][n])
        if strgseco == 'Stellar Eff Temp (K)':
            axis.set_ylim([None, 8000.])

        if strgseco == 'Planet Radius (R_Earth)':
            axis.set_ylim([None, 10.])

        path = pathimag + f'{k:02d}{l:02d}.pdf'
        plt.savefig(path)
        plt.close()
        l += 1
    k += 1
#if gdat.epocprio is None:
#    gdat.epocprio = objtexof['Epoch (BJD)'].values[indx]
#if gdat.periprio is None:
#    gdat.periprio = objtexof['Period (days)'].values[indx]
#gdat.deptprio = objtexof['Depth (ppm)'].values[indx] * 1e-6
#gdat.duraprio = objtexof['Duration (hours)'].values[indx] / 24. # [days]

