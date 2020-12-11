import allesfitter
import allesfitter.config
import allesfitter.priors
#from allesfitter.prior import simulate_PDF


from tqdm import tqdm

import tdpy.util
import tdpy.mcmc
from tdpy.util import prnt_list
from tdpy.util import summgene
import lygos.main

import pickle

import os, fnmatch
import sys, datetime
import numpy as np
import scipy.interpolate
import scipy.stats

import tesstarg.util

import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import adjustText 

import seaborn as sns

mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amssymb}"]

import astroquery
import astropy

import emcee

'''
Given a target, pexo is an interface to allesfitter that allows 
1) automatic search for, download and process TESS and Kepler data via MAST or use user-provided data
2) impose priors based on custom inputs, ExoFOP or NASA Exoplanet Archive
3) configure and run allesfitter on the target
4) Make characterization plots of the target after the analysis
'''


def retr_listcolrplan(numbplan):
    
    listcolrplan = np.array(['magenta', 'orange', 'red', 'green', 'purple', 'cyan'])[:numbplan]

    return listcolrplan


def retr_liststrgplan(numbplan):
    
    liststrgplan = np.array(['b', 'c', 'd', 'e', 'f', 'g'])[:numbplan]

    return liststrgplan


def retr_llik_albbemisepsi(gdat, para):
    
    albb = para[0]
    emis = para[1]
    epsi = para[2]
    

    psiimodl = ((1 - albb) / emis)**.25
    #tmptequiprim = gdat.dictlist['tmptequi'][:, 0] * psiimodl
    tmptequiprim = gdat.gmeatmptequi * psiimodl
    tmptdayy = tmptequiprim * (2. / 3. - 5. / 12. * epsi)**.25
    tmptnigh = tmptequiprim * (epsi / 4.)**.25
    #print('para')
    #print(para)
    #print('psiimodl')
    #print(psiimodl)
    #print('gdat.dictlist[tmptequi][:, 0]')
    #summgene(gdat.dictlist['tmptequi'][:, 0])
    #print('tmptequiprim')
    #summgene(tmptequiprim)
    #print('tmptdayy')
    #summgene(tmptdayy)
    #print('gdat.dictlist[tmptdayy][:, 0]')
    #summgene(gdat.dictlist['tmptdayy'][:, 0])
    #print('gdat.dictlist[tmptnigh][:, 0]')
    #summgene(gdat.dictlist['tmptnigh'][:, 0])
    #print('')
    #print('')
    #print('')
    
    #llik = np.zeros(gdat.numbsamp)
    #llik += -0.5 * (tmptdayy - gdat.dictlist['tmptdayy'][:, 0])**2
    #llik += -0.5 * (tmptnigh - gdat.dictlist['tmptnigh'][:, 0])**2
    #llik += -0.5 * (psiimodl - gdat.listpsii)**2 * 1e6
    #llik = np.sum(llik)
    
    llik = 0.
    llik += -0.5 * (tmptdayy - gdat.gmeatmptdayy)**2 / gdat.gstdtmptdayy**2
    llik += -0.5 * (tmptnigh - gdat.gmeatmptnigh)**2 / gdat.gstdtmptnigh**2
    llik += -0.5 * (psiimodl - gdat.gmeapsii)**2 / gdat.gstdpsii**2 * 1e3
    
    return llik


def retr_dilu(tmpttarg, tmptcomp, strgwlentype='tess'):
    
    if strgwlentype != 'tess':
        raise Exception('')
    else:
        binswlen = np.linspace(0.6, 1.)
    meanwlen = (binswlen[1:] + binswlen[:-1]) / 2.
    diffwlen = (binswlen[1:] - binswlen[:-1]) / 2.
    
    fluxtarg = tdpy.util.retr_specbbod(tmpttarg, meanwlen)
    fluxtarg = np.sum(diffwlen * fluxtarg)
    
    fluxcomp = tdpy.util.retr_specbbod(tmptcomp, meanwlen)
    fluxcomp = np.sum(diffwlen * fluxcomp)
    
    dilu = 1. - fluxtarg / (fluxtarg + fluxcomp)
    
    return dilu


def retr_modl_spec(gdat, tmpt, booltess=False, strgtype='intg'):
    
    if booltess:
        thpt = scipy.interpolate.interp1d(gdat.meanwlenband, gdat.thptband)(wlen)
    else:
        thpt = 1.
    
    if strgtype == 'intg':
        spec = tdpy.util.retr_specbbod(tmpt, gdat.meanwlen)
        spec = np.sum(gdat.diffwlen * spec)
    if strgtype == 'diff' or strgtype == 'logt':
        spec = tdpy.util.retr_specbbod(tmpt, gdat.cntrwlen)
        if strgtype == 'logt':
            spec *= gdat.cntrwlen
    
    return spec


def retr_llik_spec(gdat, para):
    
    tmpt = para[0]
    specboloplan = retr_modl_spec(gdat, tmpt, booltess=False, strgtype='intg')
    deptplan = 1e6 * gdat.rratmedi[0]**2 * specboloplan / gdat.specstarintg # [ppm]
    
    #print('tmpt')
    #print(tmpt)
    #print('specboloplan')
    #summgene(specboloplan)
    #print('gdat.specstarintg')
    #summgene(gdat.specstarintg)
    #print('deptplan')
    #print(deptplan)
    #print('gdat.deptobsd')
    #print(gdat.deptobsd)
    #print('gdat.varideptobsd')
    #print(gdat.varideptobsd)
    #print('')
    #print('')
    #print('')
    llik = -0.5 * np.sum((deptplan - gdat.deptobsd)**2 / gdat.varideptobsd)
    
    return llik


def icdf(para):
    
    icdf = gdat.limtpara[0, :] + para * (gdat.limtpara[1, :] - gdat.limtpara[0, :])

    return icdf


def retr_reso(listperi, maxmordr=10):
    
    if np.where(listperi == 0)[0].size > 0:
        raise Exception('')

    numbsamp = listperi.shape[0]
    numbplan = listperi.shape[1]
    indxplan = np.arange(numbplan)
    listratiperi = np.zeros((numbsamp, numbplan, numbplan))
    intgreso = np.zeros((numbplan, numbplan, 2))
    for j in indxplan:
        for jj in indxplan:
            if j >= jj:
                continue
                
            rati = listperi[:, j] / listperi[:, jj]
            #print('listperi')
            #print(listperi)
            #print('rati')
            #print(rati)
            if rati < 1:
                listratiperi[:, j, jj] = 1. / rati
            else:
                listratiperi[:, j, jj] = rati

            minmdiff = 1e100
            for a in range(1, maxmordr):
                for aa in range(1, maxmordr):
                    diff = abs(float(a) / aa - listratiperi[:, j, jj])
                    if np.mean(diff) < minmdiff:
                        minmdiff = np.mean(diff)
                        minmreso = a, aa
            intgreso[j, jj, :] = minmreso
            #print('minmdiff') 
            #print(minmdiff)
            #print('minmreso')
            #print(minmreso)
            #print
    
    return intgreso, listratiperi


def writ_filealle(gdat, namefile, pathalle, dictalle, dictalledefa, verbtype=1):
    
    listline = []
    # add the lines
    if namefile == 'params.csv':
        listline.append('#name,value,fit,bounds,label,unit\n')
    
    if dictalle is not None:
        for strg, varb in dictalle.items():
            if namefile == 'params.csv':
                line = strg
                for k, varbtemp in enumerate(varb):
                    if varbtemp is not None:
                        line += ',' + varbtemp
                    else:
                        line += ',' + dictalledefa[strg][k]
                line += '\n'
            else:
                line = strg + ',' + varb + '\n'
            listline.append(line)
    for strg, varb in dictalledefa.items():
        if dictalle is None or strg not in dictalle:
            if namefile == 'params.csv':
                line = strg
                for varbtemp in varb:
                    line += ',' + varbtemp
                line += '\n'
            else:
                line = strg + ',' + varb + '\n'
            listline.append(line)
    
    # write
    pathfile = pathalle + namefile
    print('Writing to %s...' % pathfile)
    objtfile = open(pathfile, 'w')
    for line in listline:
        objtfile.write('%s' % line)
    objtfile.close()


def retr_exof(gdat, strgtoii=None):
    
    path = gdat.pathbase + 'data/TOI_augmented.csv'
    if os.path.exists(path):
        print('Reading from %s...' % path)
        objtexof = pd.read_csv(path, skiprows=0)
        dictexof = objtexof.to_dict('list')
        for attr, varb in dictexof.items():
            dictexof[attr] = np.array(varb)
        #raise Exception('')
    else:
    
        pathexof = gdat.pathbase + 'data/exofop_toilists_20200916.csv'
        print('Reading from %s...' % pathexof)
        objtexof = pd.read_csv(pathexof, skiprows=0)
        if strgtoii is None:
            indx = np.arange(objtexof['TOI'].size)
        else:
            indx = np.where(objtexof['TOI'] == strgtarg)[0]
            
        if indx.size == 0:
            print('The planet name, %s, was not found in the NASA Exoplanet Archive composite table.' % strgtarg)
            return None
        else:
            dictexof = {}
            dictexof['toii'] = objtexof['TOI'][indx].values
            numbplan = dictexof['toii'].size
            indxplan = np.arange(numbplan)
            dictexof['namestar'] = np.empty(numbplan, dtype=object)
            dictexof['nameplan'] = np.empty(numbplan, dtype=object)
            for k in indxplan:
                dictexof['nameplan'][k] = 'TOI ' + str(dictexof['toii'][k])
                dictexof['namestar'][k] = 'TOI ' + str(dictexof['toii'][k])[:-3]
            dictexof['rrat'] = np.sqrt(objtexof['Depth (ppm)'][indx].values * 1e-6)
            dictexof['radistar'] = objtexof['Stellar Radius (R_Sun)'][indx].values * gdat.factrsrj
            dictexof['radiplan'] = dictexof['rrat'] * dictexof['radistar']
            
            dictexof['rascstar'] = objtexof['RA'][indx].values
            dictexof['declstar'] = objtexof['Dec'][indx].values
            
            dictexof['stdvradiplan'] = objtexof['Planet Radius (R_Earth) err'][indx].values * gdat.factrjre
            dictexof['peri'] = objtexof['Period (days)'][indx].values
            dictexof['epoc'] = objtexof['Epoch (BJD)'][indx].values
            
            dictexof['stdvradistar'] = objtexof['Stellar Radius (R_Sun) err'][indx].values * gdat.factrsrj
            dictexof['tmptstar'] = objtexof['Stellar Eff Temp (K)'][indx].values
            dictexof['stdvtmptstar'] = objtexof['Stellar Eff Temp (K) err'][indx].values
            dictexof['loggstar'] = objtexof['Stellar log(g) (cm/s^2)'][indx].values
            #dictexof['stdvloggstar'] = objtexof['Stellar log(g) (cm/s^2) err'][indx].values
            
            dictexof['vmagstar'] = np.zeros(numbplan)
            dictexof['jmagstar'] = np.zeros(numbplan)
            dictexof['hmagstar'] = np.zeros(numbplan)
            dictexof['kmagstar'] = np.zeros(numbplan)
            dictexof['boolfrst'] = np.zeros(numbplan)
            dictexof['numbplanstar'] = np.zeros(numbplan)
            for k in indxplan:
                tici = objtexof['TIC ID'].values[k]
                strgmast = 'TIC %d' % tici
                catalogData = astroquery.mast.Catalogs.query_object(strgmast, catalog='TIC', radius='1s')
                if catalogData[0]['dstArcSec'] > 0.1:
                    print('The nearest source is more than 0.1 arcsec away from the target!')
                
                #dictexof['namestar'] = 'TOI ' + str(tici)
                #dictexof['radistar'] = catalogData[0]['rad'] * gdat.factrsrj
                #dictexof['stdvradistar'] = catalogData[0]['e_rad'] * gdat.factrsrj
                #dictexof['massstar'] = catalogData[0]['mass'] * gdat.factmsmj
                #dictexof['stdvmassstar'] = catalogData[0]['e_mass'] * gdat.factmsmj
                #dictexof['tmptstar'] = catalogData[0]['Teff']
                #dictexof['stdvtmptstar'] = catalogData[0]['e_Teff']
                
                print('k')
                print(k)
                dictexof['vmagstar'][k] = catalogData[0]['Vmag']
                dictexof['jmagstar'][k] = catalogData[0]['Jmag']
                dictexof['hmagstar'][k] = catalogData[0]['Hmag']
                dictexof['kmagstar'][k] = catalogData[0]['Kmag']
                #if k == 5:
                #    break
            
            # augment
            dictexof['numbplanstar'] = np.empty(numbplan)
            dictexof['boolfrst'] = np.zeros(numbplan, dtype=bool)
            for k in indxplan:
                indxplanthis = np.where(dictexof['namestar'][k] == dictexof['namestar'])[0]
                if k == indxplanthis[0]:
                    dictexof['boolfrst'][k] = True
                dictexof['numbplanstar'][k] = indxplanthis.size
            
            dictexof['numbplantranstar'] = dictexof['numbplanstar']
            dictexof['lumistar'] = (dictexof['radistar'] / gdat.factrsrj)**2 * (dictexof['tmptstar'] / 5778.)**4
            dictexof['massstar'] = dictexof['loggstar'] * dictexof['radistar']**2
            dictexof['smax'] = (dictexof['peri']**2)**(1. / 3.) * dictexof['massstar']
            dictexof['smaxasun'] = dictexof['smax'] / gdat.factaurj
            dictexof['inso'] = objtexof['Planet Insolation (Earth Flux)'][indx].values
            #dictexof['inso'] = dictexof['lumistar'] / dictexof['smaxasun']**2
            print('dictexof[tmptstar]')
            summgene(dictexof['tmptstar'][:5])
            print('dictexof[radistar]')
            summgene(dictexof['radistar'][:5])
            print('dictexof[lumistar]')
            summgene(dictexof['lumistar'][:5])
            print('dictexof[smaxasun]')
            summgene(dictexof['smaxasun'][:5])
            #raise Exception('')
            dictexof['tmptplan'] = dictexof['tmptstar'] * np.sqrt(dictexof['radistar'] / dictexof['smax'] / 2.)
            # temp check if factor of 2 is right
            dictexof['stdvtmptplan'] = np.sqrt(dictexof['stdvtmptstar']**2 + 0.5 * dictexof['stdvradistar']**2) / np.sqrt(2.)
            print('dictexof[radiplan]')
            summgene(dictexof['radiplan'])
            dictexof['massplan'] = np.ones_like(dictexof['radiplan']) + np.nan
            indx = np.isfinite(dictexof['radiplan'])
            dictexof['massplan'][indx] = tesstarg.util.retr_massfromradi(dictexof['radiplan'][indx])
            # temp
            dictexof['stdvmassplan'] = dictexof['massplan'] * 0.3
            dictexof['densplan'] = dictexof['massplan'] / dictexof['radiplan']**3
            dictexof['vesc'] = tesstarg.util.retr_vesc(dictexof['massplan'], dictexof['radiplan'])
            dictexof['booltran'] = np.ones_like(dictexof['toii'])
        
            dictexof['projoblq'] = np.ones_like(dictexof['vesc']) + np.nan
            
        df = pd.DataFrame.from_dict(dictexof)#)orient="namestar")
        print('Writing to %s...' % path)
        df.to_csv(path)

    return dictexof


def retr_exarcomp(strgexar=None):
    
    # get NASA Exoplanet Archive data
    path = os.environ['PEXO_DATA_PATH'] + '/data/PSCompPars_2020.11.20_17.58.11.csv'
    print('Reading %s...' % path)
    objtexarcomp = pd.read_csv(path, skiprows=330)
    if strgexar is None:
        indx = np.arange(objtexarcomp['hostname'].size)
        #indx = np.where(objtexarcomp['default_flag'].values == 1)[0]
    else:
        indx = np.where(objtexarcomp['hostname'] == strgexar)[0]
        #indx = np.where((objtexarcomp['hostname'] == strgexar) & (objtexarcomp['default_flag'].values == 1))[0]
    print('strgexar')
    print(strgexar)
    
    factrsrj, factmsmj, factrjre, factmjme, factaurj = tesstarg.util.retr_factconv()

    if indx.size == 0:
        print('The planet name, %s, was not found in the NASA Exoplanet Archive composite table.' % strgexar)
        return None
    else:
        dictexarcomp = {}
        dictexarcomp['namestar'] = objtexarcomp['hostname'][indx].values
        dictexarcomp['nameplan'] = objtexarcomp['pl_name'][indx].values
        
        dictexarcomp['rascstar'] = objtexarcomp['ra'][indx].values
        dictexarcomp['declstar'] = objtexarcomp['dec'][indx].values
        
        dictexarcomp['radistar'] = objtexarcomp['st_rad'][indx].values * factrsrj # [R_J]
        radistarstd1 = objtexarcomp['st_raderr1'][indx].values * factrsrj # [R_J]
        radistarstd2 = objtexarcomp['st_raderr2'][indx].values * factrsrj # [R_J]
        dictexarcomp['stdvradistar'] = (radistarstd1 + radistarstd2) / 2.
        dictexarcomp['massstar'] = objtexarcomp['st_mass'][indx].values * factmsmj # [M_J]
        massstarstd1 = objtexarcomp['st_masserr1'][indx].values * factmsmj # [M_J]
        massstarstd2 = objtexarcomp['st_masserr2'][indx].values * factmsmj # [M_J]
        dictexarcomp['stdvmassstar'] = (massstarstd1 + massstarstd2) / 2.
        dictexarcomp['tmptstar'] = objtexarcomp['st_teff'][indx].values # [K]
        tmptstarstd1 = objtexarcomp['st_tefferr1'][indx].values # [K]
        tmptstarstd2 = objtexarcomp['st_tefferr2'][indx].values # [K]
        dictexarcomp['stdvtmptstar'] = (tmptstarstd1 + tmptstarstd2) / 2.
        
        dictexarcomp['inso'] = objtexarcomp['pl_insol'][indx].values
        dictexarcomp['peri'] = objtexarcomp['pl_orbper'][indx].values # [days]
        
        dictexarcomp['smax'] = objtexarcomp['pl_orbsmax'][indx].values # [AU]
        
        dictexarcomp['radiplan'] = objtexarcomp['pl_radj'][indx].values # [R_J]
        dictexarcomp['stdvradiplan'] = np.maximum(objtexarcomp['pl_radjerr1'][indx].values, -objtexarcomp['pl_radjerr2'][indx].values) # [R_J]
        dictexarcomp['massplan'] = objtexarcomp['pl_bmassj'][indx].values # [M_J]
        dictexarcomp['stdvmassplan'] = np.maximum(objtexarcomp['pl_bmassjerr1'][indx].values, -objtexarcomp['pl_bmassjerr2'][indx].values) # [M_J]
        dictexarcomp['tmptplan'] = objtexarcomp['pl_eqt'][indx].values # [K]
        dictexarcomp['stdvtmptplan'] = np.maximum(objtexarcomp['pl_eqterr1'][indx].values, -objtexarcomp['pl_eqterr2'][indx].values) # [K]
        
        #print('dictexarcomp[stdvradiplan][np.where(dictexarcomp[stdvradiplan])]')
        #summgene(dictexarcomp['stdvradiplan'][np.where(np.isfinite(dictexarcomp['stdvradiplan']))])
        #print('dictexarcomp[stdvmassplan][np.where(dictexarcomp[stdvmassplan])]')
        #summgene(dictexarcomp['stdvmassplan'][np.where(np.isfinite(dictexarcomp['stdvmassplan']))])
        #print('dictexarcomp[stdvtmptplan][np.where(dictexarcomp[stdvtmptplan])]')
        #summgene(dictexarcomp['stdvtmptplan'][np.where(np.isfinite(dictexarcomp['stdvtmptplan']))])
        #print('np.isfinite(dictexarcomp[radiplan]')
        #summgene(np.isfinite(dictexarcomp['radiplan']))
        #print('np.isfinite(dictexarcomp[stdvradiplan]')
        #summgene(np.isfinite(dictexarcomp['stdvradiplan']))
        #print('np.where(dictexarcomp[stdvradiplan] == 0)')
        #summgene(np.where(dictexarcomp['stdvradiplan'] == 0))
        #print('np.isfinite(dictexarcomp[massplan]')
        #summgene(np.isfinite(dictexarcomp['massplan']))
        #print('np.isfinite(dictexarcomp[stdvmassplan]')
        #summgene(np.isfinite(dictexarcomp['stdvmassplan']))
        #print('np.where(dictexarcomp[stdvmassplan] == 0)')
        #summgene(np.where(dictexarcomp['stdvmassplan'] == 0))
        #print('np.isfinite(dictexarcomp[tmptplan]')
        #summgene(np.isfinite(dictexarcomp['tmptplan']))
        #print('np.isfinite(dictexarcomp[stdvtmptplan]')
        #summgene(np.isfinite(dictexarcomp['stdvtmptplan']))
        #print('np.where(dictexarcomp[stdvtmptplan] == 0)')
        #summgene(np.where(dictexarcomp['stdvtmptplan'] == 0))
        
        dictexarcomp['booltran'] = objtexarcomp['tran_flag'][indx].values # [K]
        dictexarcomp['booltran'] = dictexarcomp['booltran'].astype(bool)
        dictexarcomp['vmagstar'] = objtexarcomp['sy_vmag'][indx].values
        dictexarcomp['jmagstar'] = objtexarcomp['sy_jmag'][indx].values # [K]
        dictexarcomp['hmagstar'] = objtexarcomp['sy_hmag'][indx].values # [K]
        dictexarcomp['kmagstar'] = objtexarcomp['sy_kmag'][indx].values # [K]
        dictexarcomp['densplan'] = objtexarcomp['pl_dens'][indx].values / 5.51 # [d_E]
        dictexarcomp['loggstar'] = dictexarcomp['massstar'] / dictexarcomp['radistar']**2
        dictexarcomp['vsiistar'] = objtexarcomp['st_vsin'][indx].values # [km/s]
        dictexarcomp['projoblq'] = objtexarcomp['pl_projobliq'][indx].values # [deg]
        
        dictexarcomp['stdvradiplan'][~np.isfinite(dictexarcomp['stdvradiplan'])] = 0.
        dictexarcomp['stdvmassplan'][~np.isfinite(dictexarcomp['stdvmassplan'])] = 0.
        dictexarcomp['stdvtmptplan'][~np.isfinite(dictexarcomp['stdvtmptplan'])] = 0.
        
        numbplanexar = len(dictexarcomp['nameplan'])
        
        dictexarcomp['numbplanstar'] = np.empty(numbplanexar)
        dictexarcomp['numbplantranstar'] = np.empty(numbplanexar)
        dictexarcomp['boolfrst'] = np.zeros(numbplanexar, dtype=bool)
        #dictexarcomp['booltrantotl'] = np.empty(numbplanexar, dtype=bool)
        for k, namestar in enumerate(dictexarcomp['namestar']):
            indxexarstar = np.where(namestar == dictexarcomp['namestar'])[0]
            if k == indxexarstar[0]:
                dictexarcomp['boolfrst'][k] = True
            dictexarcomp['numbplanstar'][k] = indxexarstar.size
            indxexarstartran = np.where((namestar == dictexarcomp['namestar']) & dictexarcomp['booltran'])[0]
            dictexarcomp['numbplantranstar'][k] = indxexarstartran.size
            #dictexarcomp['booltrantotl'][k] = dictexarcomp['booltran'][indxexarstar].all()
    
    return dictexarcomp


def get_color(color):

    if isinstance(color, tuple) and len(color) == 3: # already a tuple of RGB values
        return color

    import matplotlib.colors as mplcolors
    
    if color == 'r':
        color = 'red'
    if color == 'g':
        color = 'green'
    if color == 'y':
        color = 'yellow'
    if color == 'c':
        color = 'cyan'
    if color == 'm':
        color = 'magenta'
    if color == 'b':
        color = 'blue'
    if color == 'o':
        color = 'orange'
    hexcolor = mplcolors.cnames[color]

    hexcolor = hexcolor.lstrip('#')
    lv = len(hexcolor)
    
    return tuple(int(hexcolor[i:i + lv // 3], 16)/255. for i in range(0, lv, lv // 3)) # tuple of rgb values


def retr_objtlinefade(x, y, colr='black', initalph=1., alphfinl=0.):
    
    colr = get_color(colr)
    cdict = {'red':   ((0.,colr[0],colr[0]),(1.,colr[0],colr[0])),
             'green': ((0.,colr[1],colr[1]),(1.,colr[1],colr[1])),
             'blue':  ((0.,colr[2],colr[2]),(1.,colr[2],colr[2])),
             'alpha': ((0.,initalph, initalph), (1., alphfinl, alphfinl))}
    
    Npts = len(x)
    if len(y) != Npts:
        raise AttributeError("x and y must have same dimension.")
   
    segments = np.zeros((Npts-1,2,2))
    segments[0][0] = [x[0], y[0]]
    for i in range(1,Npts-1):
        pt = [x[i], y[i]]
        segments[i-1][1] = pt
        segments[i][0] = pt 
    segments[-1][1] = [x[-1], y[-1]]

    individual_cm = mpl.colors.LinearSegmentedColormap('indv1', cdict)
    lc = mpl.collections.LineCollection(segments, cmap=individual_cm)
    lc.set_array(np.linspace(0.,1.,len(segments)))
    
    return lc


def plot_pser(gdat, strgproc):
    
    for b in gdat.indxdatatser:
        arrypcur = gdat.arrypcurprim[strgproc]
        arrypcurbindtotl = gdat.arrypcurprimbindtotl[strgproc]
        arrypcurbindzoom = gdat.arrypcurprimbindzoom[strgproc]
            
        # plot individual phase curves
        for p in gdat.indxinst[b]:
            for j in gdat.indxplan:
                # phase on the horizontal axis
                figr, axis = plt.subplots(1, 1, figsize=gdat.figrsizeydob)
                if b == 0:
                    yerr = None
                if b == 1:
                    yerr = arrypcur[b][p][j][:, 2]
                axis.errorbar(arrypcur[b][p][j][:, 0], arrypcur[b][p][j][:, 1], yerr=yerr, elinewidth=1, capsize=2, zorder=1, \
                                                            color='grey', alpha=gdat.alphraww, marker='o', ls='', ms=1, rasterized=gdat.boolrastraww)
                if b == 0:
                    yerr = None
                if b == 1:
                    yerr = arrypcurbindzoom[b][p][j][:, 2]
                axis.errorbar(arrypcurbindzoom[b][p][j][:, 0], arrypcurbindzoom[b][p][j][:, 1], color=gdat.listcolrplan[j], elinewidth=1, capsize=2, \
                                                                                                                 zorder=2, marker='o', ls='', ms=3)
                if gdat.boolwritplan:
                    axis.text(0.9, 0.9, r'\textbf{%s}' % gdat.liststrgplan[j], \
                                        color=gdat.listcolrplan[j], va='center', ha='center', transform=axis.transAxes)
                axis.set_ylabel(gdat.listlabltser[b])
                axis.set_xlabel('Phase')
                # overlay the posterior model
                if strgproc.startswith('adtr'):
                    axis.plot(gdat.arrypcurprimmodl[b][p][j][:, 0], gdat.arrypcurprimmodl[b][p][j][:, 1], color='b', zorder=3)
                if gdat.listdeptdraw is not None:
                    for k in range(len(gdat.listdeptdraw)):  
                        axis.axhline(1. - gdat.listdeptdraw[k], ls='-', color='grey')
                path = gdat.pathimag + 'pcurphas_%s_%s_%s_%s_%s.%s' % (gdat.liststrginst[b][p], gdat.liststrgplan[j], \
                                                                                            strgproc, gdat.strgtarg, gdat.strgprio, gdat.typefileplot)
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()
            
                # time on the horizontal axis
                figr, axis = plt.subplots(1, 1, figsize=gdat.figrsize)
                if b == 0:
                    yerr = None
                if b == 1:
                    yerr = arrypcur[b][p][j][:, 2]
                axis.errorbar(gdat.periprio[j] * arrypcur[b][p][j][:, 0] * 24., arrypcur[b][p][j][:, 1], yerr=yerr, elinewidth=1, capsize=2, \
                                                    zorder=1, color='grey', alpha=gdat.alphraww, marker='o', ls='', ms=1, rasterized=gdat.boolrastraww)
                if b == 0:
                    yerr = None
                if b == 1:
                    yerr = arrypcurbindzoom[b][p][j][:, 2]
                axis.errorbar(gdat.periprio[j] * arrypcurbindzoom[b][p][j][:, 0] * 24., arrypcurbindzoom[b][p][j][:, 1], zorder=2, \
                                                                                                yerr=yerr, elinewidth=1, capsize=2, \
                                                                                    color=gdat.listcolrplan[j], marker='o', ls='', ms=3)
                if strgproc.startswith('adtr'):
                    axis.plot(gdat.periprio[j] * 24. * gdat.arrypcurprimmodl[b][p][j][:, 0], gdat.arrypcurprimmodl[b][p][j][:, 1], color='b', zorder=3)
                if gdat.boolwritplan:
                    axis.text(0.9, 0.9, \
                                    r'\textbf{%s}' % gdat.liststrgplan[j], color=gdat.listcolrplan[j], va='center', ha='center', transform=axis.transAxes)
                # temp these are prior
                #axis.axvline(-np.amax(gdat.duraprio) * 24., alpha=gdat.alphraww, ls='--', color=gdat.listcolrplan[j])
                #axis.axvline(np.amax(gdat.duraprio) * 24., alpha=gdat.alphraww, ls='--', color=gdat.listcolrplan[j])
                axis.set_ylabel(gdat.listlabltser[b])
                axis.set_xlabel('Time [hours]')
                axis.set_xlim([-np.amax(gdat.duramask) * 24., np.amax(gdat.duramask) * 24.])
                if gdat.listdeptdraw is not None:
                    for k in range(len(gdat.listdeptdraw)):  
                        axis.axhline(1. - gdat.listdeptdraw[k], ls='--', color='grey')
                plt.subplots_adjust(hspace=0., bottom=0.25, left=0.25)
                path = gdat.pathimag + 'pcurtime_%s_%s_%s_%s_%s.%s' % (gdat.liststrginst[b][p], gdat.liststrgplan[j], \
                                                                                strgproc, gdat.strgtarg, gdat.strgprio, gdat.typefileplot)
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()
            
            # plot all phase curves
            if gdat.numbplan > 1:
                figr, axis = plt.subplots(gdat.numbplan, 1, figsize=gdat.figrsizeydob, sharex=True)
                if gdat.numbplan == 1:
                    axis = [axis]
                for jj, j in enumerate(gdat.indxplan):
                    axis[jj].plot(arrypcur[b][p][j][:, 0], arrypcur[b][p][j][:, 1], color='grey', alpha=gdat.alphraww, \
                                                                                        marker='o', ls='', ms=1, rasterized=gdat.boolrastraww)
                    axis[jj].plot(arrypcurbindtotl[b][p][j][:, 0], arrypcurbindtotl[b][p][j][:, 1], color=gdat.listcolrplan[j], marker='o', ls='', ms=1)
                    if gdat.boolwritplan:
                        axis[jj].text(0.97, 0.8, r'\textbf{%s}' % gdat.liststrgplan[j], transform=axis[jj].transAxes, \
                                                                                            color=gdat.listcolrplan[j], va='center', ha='center')
                axis[0].set_ylabel(gdat.listlabltser[b])
                axis[0].set_xlim(-0.5, 0.5)
                axis[0].yaxis.set_label_coords(-0.08, 1. - 0.5 * gdat.numbplan)
                axis[gdat.numbplan-1].set_xlabel('Phase')
                
                plt.subplots_adjust(hspace=0., bottom=0.2)
                path = gdat.pathimag + 'pcurphastotl_%s_%s_%s_%s.%s' % (gdat.liststrginst[b][p], strgproc, gdat.strgtarg, gdat.strgprio, gdat.typefileplot)
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()
    

def retr_albg(amplplanrefl, radiplan, smax):
    
    albg = amplplanrefl / (radiplan / smax)**2
    
    return albg


def calc_prop(gdat, strgpdfn):

    gdat.liststrgfeat = ['epoc', 'peri', 'rrat', 'rsma', 'cosi', 'ecos', 'esin', 'rvsa']
    if strgpdfn == '0003' or strgpdfn == '0004' or strgpdfn == '0006':
        gdat.liststrgfeat += ['sbrtrati', 'amplelli']
    if strgpdfn == '0003' or strgpdfn == '0006':
        gdat.liststrgfeat += ['amplplan', 'timeshftplan']
    if strgpdfn == '0004':
        gdat.liststrgfeat += ['amplplanther', 'amplplanrefl', 'timeshftplanther', 'timeshftplanrefl']
    
    gdat.dictlist = {}
    gdat.dictpost = {}
    gdat.dicterrr = {}
    for strgfeat in gdat.liststrgfeat:
        gdat.dictlist[strgfeat] = np.empty((gdat.numbsamp, gdat.numbplan))

        for j in gdat.indxplanalle:
            if strgfeat == 'epoc':
                strg = '%s_epoch' % gdat.liststrgplan[j]
            if strgfeat == 'peri':
                strg = '%s_period' % gdat.liststrgplan[j]
            if strgfeat == 'rrat':
                strg = '%s_rr' % gdat.liststrgplan[j]
            if strgfeat == 'rsma':
                strg = '%s_rsuma' % gdat.liststrgplan[j]
            if strgfeat == 'cosi':
                strg = '%s_cosi' % gdat.liststrgplan[j]
    
            if strgpdfn == '0003' or strgpdfn == '0004' or strgpdfn == '0006':
                if strgfeat == 'sbrtrati':
                    strg = '%s_sbratio_TESS' % gdat.liststrgplan[j]
                if strgfeat == 'amplelli':
                    strg = '%s_phase_curve_ellipsoidal_TESS' % gdat.liststrgplan[j]
            if strgpdfn == '0003' or strgpdfn == '0006':
                if strgfeat == 'amplplan':
                    strg = '%s_phase_curve_atmospheric_TESS' % gdat.liststrgplan[j]
                if strgfeat == 'timeshftplan':
                    strg = '%s_phase_curve_atmospheric_shift_TESS' % gdat.liststrgplan[j]
            if strgpdfn == '0004':
                if strgfeat == 'amplplanther':
                    strg = '%s_phase_curve_atmospheric_thermal_TESS' % gdat.liststrgplan[j]
                if strgfeat == 'amplplanrefl':
                    strg = '%s_phase_curve_atmospheric_reflected_TESS' % gdat.liststrgplan[j]
                if strgfeat == 'timeshftplanther':
                    strg = '%s_phase_curve_atmospheric_thermal_shift_TESS' % gdat.liststrgplan[j]
                if strgfeat == 'timeshftplanrefl':
                    strg = '%s_phase_curve_atmospheric_reflected_shift_TESS' % gdat.liststrgplan[j]
            
            if strgpdfn == 'prio':
                gdat.dictlist[strgfeat][:, j] = getattr(gdat, strgfeat + 'prio')[j] + np.random.randn(gdat.numbsamp) * \
                                                                                            getattr(gdat, 'stdv' + strgfeat + 'prio')[j]
            else:
                if strg in gdat.objtalle[strgpdfn].posterior_params.keys():
                    gdat.dictlist[strgfeat][:, j] = gdat.objtalle[strgpdfn].posterior_params[strg][gdat.indxsamp]
                else:
                    gdat.dictlist[strgfeat][:, j] = np.zeros(gdat.numbsamp) + allesfitter.config.BASEMENT.params[strg]

    # allesfitter phase curve depths are in ppt
    for strgfeat in gdat.liststrgfeat:
        if strgfeat.startswith('ampl'):
            gdat.dictlist[strgfeat] *= 1e-3
    
    print('Calculating derived variables...')
    # derived variables
    ## get samples from the star's variables

    # stellar properties
    for featstar in gdat.listfeatstar:
        stdvtemp = getattr(gdat, 'stdv' + featstar)
        meantemp = getattr(gdat, featstar)
        if stdvtemp == 0.:
            a = -np.inf
        else:
            a = -meantemp / stdvtemp

        gdat.dictlist[featstar] = allesfitter.priors.simulate_PDF.simulate_PDF(meantemp, stdvtemp, stdvtemp, size=gdat.numbsamp, plot=False)
        #gdat.dictlist[featstar] = r = scipy.stats.truncnorm.rvs(a, np.inf, size=gdat.numbsamp) * stdvtemp + meantemp
        
        gdat.dictlist[featstar] = np.vstack([gdat.dictlist[featstar]] * gdat.numbplan).T
    
    if strgpdfn == '0003' or strgpdfn == '0004' or strgpdfn == '0006':
        gdat.dictlist['amplnigh'] = gdat.dictlist['sbrtrati'] * gdat.dictlist['rrat']**2
    if strgpdfn == '0003' or strgpdfn == '0006':
        gdat.dictlist['phasshftplan'] = gdat.dictlist['timeshftplan'] * 360. / gdat.dictlist['peri']
    if strgpdfn == '0004':
        gdat.dictlist['phasshftplanther'] = gdat.dictlist['timeshftplanther'] * 360. / gdat.dictlist['peri']
        gdat.dictlist['phasshftplanrefl'] = gdat.dictlist['timeshftplanrefl'] * 360. / gdat.dictlist['peri']

    print('Calculating inclinations...')
    # inclination [degree]
    gdat.dictlist['incl'] = np.arccos(gdat.dictlist['cosi']) * 180. / np.pi
    
    # radius of the planets
    gdat.dictlist['radiplan'] = gdat.dictlist['radistar'] * gdat.dictlist['rrat']
    
    # semi-major axis
    gdat.dictlist['smax'] = (gdat.dictlist['radiplan'] + gdat.dictlist['radistar']) / gdat.dictlist['rsma']
    gdat.dictlist['smaxasun'] = gdat.dictlist['smax'] / gdat.factaurj
    
    print('Calculating equilibrium temperatures...')
    # planet equilibrium temperature
    gdat.dictlist['tmptplan'] = gdat.dictlist['tmptstar'] * np.sqrt(gdat.dictlist['radistar'] / 2. / gdat.dictlist['smax'])
    
    # stellar luminosity
    gdat.dictlist['lumistar'] = (gdat.dictlist['radistar'] / gdat.factrsrj)**2 * (gdat.dictlist['tmptstar'] / 5778.)**4
    
    # insolation
    gdat.dictlist['inso'] = gdat.dictlist['lumistar'] / gdat.dictlist['smaxasun']**2
    
    # predicted planet mass
    print('Calculating predicted masses...')
    gdat.dictlist['massplanpredchen'] = np.empty_like(gdat.dictlist['radiplan'])
    gdat.dictlist['massplanpredwolf'] = np.empty_like(gdat.dictlist['radiplan'])
    for j in gdat.indxplanalle:
        print('j')
        print(j)
        print('gdat.dictlist[radiplan][:, j]')
        summgene(gdat.dictlist['radiplan'][:, j])
        if not np.isfinite(gdat.dictlist['radiplan'][:, j]).all():
            raise Exception('')
        gdat.dictlist['massplanpredchen'][:, j] = tesstarg.util.retr_massfromradi(gdat.dictlist['radiplan'][:, j])
        gdat.dictlist['massplanpredwolf'][:, j] = tesstarg.util.retr_massfromradi(gdat.dictlist['radiplan'][:, j], strgtype='wolf2016')
    gdat.dictlist['massplanpred'] = gdat.dictlist['massplanpredchen']
    gdat.dictlist['massplanpredeartchen'] = gdat.dictlist['massplanpredchen'] * gdat.factmjme
    gdat.dictlist['massplanpredeartwolf'] = gdat.dictlist['massplanpredwolf'] * gdat.factmjme
    
    # mass used for later calculations
    gdat.dictlist['massplanused'] = np.empty_like(gdat.dictlist['massplanpredchen'])
    
    # temp
    gdat.dictlist['massplan'] = np.zeros_like(gdat.dictlist['esin'])
    gdat.dictlist['massplanused'] = gdat.dictlist['massplanpredchen']
    #for j in gdat.indxplanalle:
    #    if 
    #        gdat.dictlist['massplanused'][:, j] = 
    #    else:
    #        gdat.dictlist['massplanused'][:, j] = 
    
    # conversions
    gdat.dictlist['radistarsunn'] = gdat.dictlist['radistar'] / gdat.factrsrj # R_S
    gdat.dictlist['radiplaneart'] = gdat.dictlist['radiplan'] * gdat.factrjre # R_E
    gdat.dictlist['massplaneart'] = gdat.dictlist['massplan'] * gdat.factmjme # M_E

    # density of the planet
    gdat.dictlist['densplan'] = gdat.dictlist['massplanused'] / gdat.dictlist['radiplan']**3

    # log g of the host star
    gdat.dictlist['loggstar'] = gdat.dictlist['massstar'] / gdat.dictlist['radistar']**2

    # escape velocity
    gdat.dictlist['vesc'] = tesstarg.util.retr_vesc(gdat.dictlist['massplanused'], gdat.dictlist['radiplan'])
    
    print('Calculating radius and period ratios...')
    for j in gdat.indxplanalle:
        strgratiperi = 'ratiperi_%s' % gdat.liststrgplan[j]
        strgratiradi = 'ratiradi_%s' % gdat.liststrgplan[j]
        for jj in gdat.indxplanalle:
            gdat.dictlist[strgratiperi] = gdat.dictlist['peri'][:, j] / gdat.dictlist['peri'][:, jj]
            gdat.dictlist[strgratiradi] = gdat.dictlist['radiplan'][:, j] / gdat.dictlist['radiplan'][:, jj]
    
    gdat.dictlist['ecce'] = gdat.dictlist['esin']**2 + gdat.dictlist['ecos']**2
    print('Calculating RV semi-amplitudes...')
    # RV semi-amplitude
    gdat.dictlist['rvsapred'] = tesstarg.util.retr_rvelsema(gdat.dictlist['peri'], gdat.dictlist['massplanpred'], \
                                                                                            gdat.dictlist['massstar'] / gdat.factmsmj, \
                                                                                                gdat.dictlist['incl'], gdat.dictlist['ecce'])
    
    print('Calculating TSMs...')
    # TSM
    gdat.dictlist['tsmm'] = tesstarg.util.retr_tsmm(gdat.dictlist['radiplan'], gdat.dictlist['tmptplan'], \
                                                                                gdat.dictlist['massplanused'], gdat.dictlist['radistar'], gdat.jmagstar)
    
    # ESM
    gdat.dictlist['esmm'] = tesstarg.util.retr_esmm(gdat.dictlist['tmptplan'], gdat.dictlist['tmptstar'], \
                                                                                gdat.dictlist['radiplan'], gdat.dictlist['radistar'], gdat.kmagstar)
        
    gdat.dictlist['logttsmm'] = np.log(gdat.dictlist['tsmm']) 
    gdat.dictlist['logtesmm'] = np.log(gdat.dictlist['esmm']) 

    # temp
    gdat.dictlist['sini'] = np.sqrt(1. - gdat.dictlist['cosi']**2)
    gdat.dictlist['omeg'] = 180. / np.pi * np.mod(np.arctan2(gdat.dictlist['esin'], gdat.dictlist['ecos']), 2 * np.pi)
    gdat.dictlist['rs2a'] = gdat.dictlist['rsma'] / (1. + gdat.dictlist['rrat'])
    gdat.dictlist['dept'] = gdat.dictlist['rrat']**2
    gdat.dictlist['sinw'] = np.sin(np.pi / 180. * gdat.dictlist['omeg'])
    
    gdat.dictlist['imfa'] = retr_imfa(gdat.dictlist['cosi'], gdat.dictlist['rs2a'], gdat.dictlist['ecce'], gdat.dictlist['sinw'])
   
    ## expected ellipsoidal variation (EV)
    gdat.dictlist['deptelli'] = gdat.alphelli * gdat.dictlist['massplanused'] * np.sin(gdat.dictlist['incl'] / 180. * np.pi)**2 / \
                                                                  gdat.dictlist['massstar']* (gdat.dictlist['radistar'] / gdat.dictlist['smax'])**3
    ## expected Doppler beaming (DB)
    deptbeam = 4. * gdat.dictlist['rvsapred'] / 3e8 * np.sum(gdat.diffwlenbeam * gdat.funcpcurmodu)

    print('Calculating durations...')
    

    gdat.dictlist['durafull'] = retr_durafull(gdat.dictlist['peri'], gdat.dictlist['rs2a'], gdat.dictlist['sini'], \
                                                                                    gdat.dictlist['rrat'], gdat.dictlist['imfa'])
    gdat.dictlist['duratotl'] = retr_duratotl(gdat.dictlist['peri'], gdat.dictlist['rs2a'], gdat.dictlist['sini'], \
                                                                                    gdat.dictlist['rrat'], gdat.dictlist['imfa'])
    #gdat.dictlist['durafull'] = gdat.dictlist['peri'] / np.pi * np.arcsin(gdat.dictlist['rs2a'] / gdat.dictlist['sini'] * \
    #                    np.sqrt((1. - gdat.dictlist['rrat'])**2 - gdat.dictlist['imfa']**2))
    #gdat.dictlist['duratotl'] = gdat.dictlist['peri'] / np.pi * np.arcsin(gdat.dictlist['rs2a'] / gdat.dictlist['sini'] * \
    #                    np.sqrt((1. + gdat.dictlist['rrat'])**2 - gdat.dictlist['imfa']**2))
    
    gdat.dictlist['maxmdeptblen'] = (1. - gdat.dictlist['durafull'] / gdat.dictlist['duratotl'])**2 / \
                                                                    (1. + gdat.dictlist['durafull'] / gdat.dictlist['duratotl'])**2
    gdat.dictlist['minmdilu'] = gdat.dictlist['dept'] / gdat.dictlist['maxmdeptblen']
    gdat.dictlist['minmratiflux'] = gdat.dictlist['minmdilu'] / (1. - gdat.dictlist['minmdilu'])
    gdat.dictlist['maxmdmag'] = -2.5 * np.log10(gdat.dictlist['minmratiflux'])
    
    # orbital
    ## RM effect
    gdat.dictlist['amplrmef'] = 2. / 3. * gdat.dictlist['vsiistar'] * gdat.dictlist['dept'] * np.sqrt(1. - gdat.dictlist['imfa'])
    gdat.dictlist['stnormefpfss'] = (gdat.dictlist['amplrmef'] / 0.9) * np.sqrt(gdat.dictlist['durafull'] / (10. / 60. / 24.))
    
    # 0003 single component, offset
    # 0004 double component, offset
    # 0006 single component, GP
    if strgpdfn == '0003' or strgpdfn == '0006':
        frac = np.random.rand(gdat.dictlist['amplplan'].size).reshape(gdat.dictlist['amplplan'].shape)
        gdat.dictlist['amplplanther'] = gdat.dictlist['amplplan'] * frac
        gdat.dictlist['amplplanrefl'] = gdat.dictlist['amplplan'] * (1. - frac)
    
    if strgpdfn == '0004':
        # temp -- this does not work for two component (thermal + reflected)
        gdat.dictlist['amplseco'] = gdat.dictlist['amplnigh'] + gdat.dictlist['amplplanther'] + gdat.dictlist['amplplanrefl']
    if strgpdfn == '0003' or strgpdfn == '0006':
        # temp -- this does not work when phase shift is nonzero
        gdat.dictlist['amplseco'] = gdat.dictlist['amplnigh'] + gdat.dictlist['amplplan']
    
    if strgpdfn == '0003' or strgpdfn == '0004' or strgpdfn == '0006':
        gdat.dictlist['albg'] = retr_albg(gdat.dictlist['amplplanrefl'], gdat.dictlist['radiplan'], gdat.dictlist['smax'])

    print('Calculating the equilibrium temperature of the planets...')
    gdat.dictlist['tmptequi'] = gdat.dictlist['tmptstar'] * np.sqrt(gdat.dictlist['radistar'] / gdat.dictlist['smax'] / 2.)
    
    if gdat.labltarg == 'WASP-121':# and strgpdfn == '0003':
        # read and parse ATMO posterior
        ## get secondary depth data from Tom
        path = gdat.pathdata + 'ascii_output/EmissionDataArray.txt'
        print('Reading from %s...' % path)
        arrydata = np.loadtxt(path)
        print('arrydata')
        summgene(arrydata)
        print('arrydata[0, :]')
        print(arrydata[0, :])
        path = gdat.pathdata + 'ascii_output/EmissionModelArray.txt'
        print('Reading from %s...' % path)
        arrymodl = np.loadtxt(path)
        print('arrymodl')
        summgene(arrymodl)
        print('Secondary eclipse depth mean and standard deviation:')
        # get wavelengths
        path = gdat.pathdata + 'ascii_output/ContribFuncWav.txt'
        print('Reading from %s...' % path)
        wlen = np.loadtxt(path)
        path = gdat.pathdata + 'ascii_output/ContribFuncWav.txt'
        print('Reading from %s...' % path)
        wlenctrb = np.loadtxt(path, skiprows=1)
   
        ### spectrum of the host star
        gdat.meanwlenthomraww = arrymodl[:, 0]
        gdat.specstarthomraww = arrymodl[:, 9]
        
        ## calculate the geometric albedo "informed" by the ATMO posterior
        wlenmodl = arrymodl[:, 0]
        deptmodl = arrymodl[:, 1]
        indxwlenmodltess = np.where((wlenmodl > 0.6) & (wlenmodl < 0.95))[0]
        gdat.amplplantheratmo = np.mean(1e-6 * deptmodl[indxwlenmodltess])
        print('gdat.amplplantheratmo')
        print(gdat.amplplantheratmo)
        gdat.dictlist['amplplanreflatmo'] = 1e-6 * arrydata[0, 2] + np.random.randn(gdat.numbsamp).reshape((gdat.numbsamp, 1)) \
                                                                                                    * arrydata[0, 3] * 1e-6 - gdat.amplplantheratmo
        #gdat.dictlist['amplplanreflatmo'] = gdat.dictlist['amplplan'] - gdat.amplplantheratmo
        gdat.dictlist['albginfo'] = retr_albg(gdat.dictlist['amplplanreflatmo'], gdat.dictlist['radiplan'], gdat.dictlist['smax'])
        
        ## update Tom's secondary (dayside) with the posterior secondary depth, since Tom's secondary was preliminary (i.e., 490+-50 ppm)
        print('Updating the multiband depth array with dayside and adding the nightside...')
        medideptseco = np.median(gdat.dictlist['amplseco'][:, 0])
        stdvdeptseco = (np.percentile(gdat.dictlist['amplseco'][:, 0], 84.) - np.percentile(gdat.dictlist['amplseco'][:, 0], 16.)) / 2.
        arrydata[0, 2] = medideptseco * 1e6 # [ppm]
        arrydata[0, 3] = stdvdeptseco * 1e6 # [ppm]
        ## add the nightside depth
        medideptnigh = np.median(gdat.dictlist['amplnigh'][:, 0])
        stdvdeptnigh = (np.percentile(gdat.dictlist['amplnigh'][:, 0], 84.) - np.percentile(gdat.dictlist['amplnigh'][:, 0], 16.)) / 2.
        arrydata = np.concatenate((arrydata, np.array([[arrydata[0, 0], arrydata[0, 1], medideptnigh * 1e6, stdvdeptnigh * 1e6, 0, 0, 0, 0]])), axis=0)
        
        print('arrydata[0, :]')
        print(arrydata[0, :])
        print('arrydata[-1, :]')
        print(arrydata[-1, :])
        
        # calculate brightness temperatures
        listlablpara = [['Temperature', 'K']]
        gdat.rratmedi = np.median(gdat.dictlist['rrat'], axis=0)
        listscalpara = ['self']
        listminmpara = np.array([1000.])
        listmaxmpara = np.array([4000.])
        listmeangauspara = None
        liststdvgauspara = None
        numbpara = len(listlablpara)
        numbdata = 0
        numbsampwalk = 10000
        numbsampburnwalk = 0
        numbsampburnwalkseco = 1000
        gdat.numbdatatmpt = arrydata.shape[0]
        gdat.indxdatatmpt = np.arange(gdat.numbdatatmpt)
        listtmpt = []
        specarry = np.empty((2, 3, gdat.numbdatatmpt))
        for k in gdat.indxdatatmpt:
            
            if not (k == 0 or k == gdat.numbdatatmpt - 1):
                continue
            gdat.minmwlen = arrydata[k, 0] - arrydata[k, 1]
            gdat.maxmwlen = arrydata[k, 0] + arrydata[k, 1]
            gdat.binswlen = np.linspace(gdat.minmwlen, gdat.maxmwlen, 100)
            gdat.meanwlen = (gdat.binswlen[1:] + gdat.binswlen[:-1]) / 2.
            gdat.diffwlen = (gdat.binswlen[1:] - gdat.binswlen[:-1]) / 2.
            gdat.cntrwlen = np.mean(gdat.meanwlen)
            strgextn = 'tmpt_%d' % k
            gdat.indxenerdata = k

            gdat.specstarintg = retr_modl_spec(gdat, gdat.tmptstar, strgtype='intg')
            
            gdat.specstarthomlogt = scipy.interpolate.interp1d(gdat.meanwlenthomraww, gdat.specstarthomraww)(gdat.cntrwlen)
            gdat.specstarthomdiff = gdat.specstarthomlogt / gdat.cntrwlen
            gdat.specstarthomintg = np.sum(gdat.diffwlen * \
                                    scipy.interpolate.interp1d(gdat.meanwlenthomraww, gdat.specstarthomraww)(gdat.meanwlen) / gdat.meanwlen)

            print('k')
            print(k)
            gdat.deptobsd = arrydata[k, 2]
            gdat.stdvdeptobsd = arrydata[k, 3]
            gdat.varideptobsd = gdat.stdvdeptobsd**2
            print('gdat.deptobsd')
            print(gdat.deptobsd)
            print('gdat.varideptobsd')
            print(gdat.varideptobsd)
            print('')
            listtmpttemp = tdpy.mcmc.samp(gdat, gdat.pathalle[strgpdfn], numbsampwalk, numbsampburnwalk, numbsampburnwalkseco, retr_llik_spec, \
                             listlablpara, listscalpara, listminmpara, listmaxmpara, listmeangauspara, liststdvgauspara, numbdata, strgextn=strgextn)[:, 0]
            listtmpt.append(listtmpttemp)
        listtmpt = np.vstack(listtmpt).T
        print('listtmpt')
        summgene(listtmpt)
        indxsamp = np.random.choice(np.arange(listtmpt.shape[0]), size=gdat.numbsamp, replace=False)
        # dayside and nightside temperatures to be used for albedo, emissivity, and circulation efficiency calculation
        gdat.dictlist['tmptdayy'] = listtmpt[indxsamp, 0, None]
        gdat.dictlist['tmptnigh'] = listtmpt[indxsamp, -1, None]
        print('indxsamp')
        summgene(indxsamp)
        print('listtmpt')
        summgene(listtmpt)
        print('gdat.dictlist[tmptdayy]')
        summgene(gdat.dictlist['tmptdayy'])
        # dayside/nightside temperature contrast
        gdat.dictlist['tmptcont'] = (gdat.dictlist['tmptdayy'] - gdat.dictlist['tmptnigh']) / gdat.dictlist['tmptdayy']
        
    # copy the prior
    gdat.dictlist['projoblq'] = np.random.randn(gdat.numbsamp)[:, None] * gdat.stdvprojoblqprio[None, :] + gdat.projoblqprio[None, :]
    
    gdat.boolsampbadd = np.zeros(gdat.numbsamp, dtype=bool)
    for j in gdat.indxplanalle:
        boolsampbaddtemp = ~np.isfinite(gdat.dictlist['maxmdmag'][:, j])
        gdat.boolsampbadd = gdat.boolsampbadd | boolsampbaddtemp
    gdat.indxsampbadd = np.where(gdat.boolsampbadd)[0]
    gdat.indxsamptran = np.setdiff1d(gdat.indxsamp, gdat.indxsampbadd)

    gdat.liststrgfeat = np.array(list(gdat.dictlist.keys()))
    for strgfeat in gdat.liststrgfeat:
        errrshap = list(gdat.dictlist[strgfeat].shape)
        errrshap[0] = 3
        gdat.dictpost[strgfeat] = np.empty(errrshap)
        gdat.dicterrr[strgfeat] = np.empty(errrshap)
        
        # transit duration can be NaN when not transiting
        indxfini = np.isfinite(gdat.dictlist[strgfeat])
        if not indxfini.all():
            print('Warning! %s are not all finite!' % strgfeat)
            print('gdat.dictlist[strgfeat]')
            summgene(gdat.dictlist[strgfeat])
            print('indxfini')
            summgene(indxfini)
        gdat.dictpost[strgfeat][0, ...] = np.nanpercentile(gdat.dictlist[strgfeat], 16., 0)
        gdat.dictpost[strgfeat][1, ...] = np.nanpercentile(gdat.dictlist[strgfeat], 50., 0)
        gdat.dictpost[strgfeat][2, ...] = np.nanpercentile(gdat.dictlist[strgfeat], 84., 0)
        gdat.dicterrr[strgfeat][0, ...] = gdat.dictpost[strgfeat][1, ...]
        gdat.dicterrr[strgfeat][1, ...] = gdat.dictpost[strgfeat][1, ...] - gdat.dictpost[strgfeat][0, ...]
        gdat.dicterrr[strgfeat][2, ...] = gdat.dictpost[strgfeat][2, ...] - gdat.dictpost[strgfeat][1, ...]
        
    for strgfeat in np.sort(gdat.liststrgfeat):
        print(strgfeat)
        print(gdat.dicterrr[strgfeat])
        print('')
    # augment
    gdat.dictfeatobjt['radistar'] = gdat.dicterrr['radistar'][0, :]
    gdat.dictfeatobjt['radiplan'] = gdat.dicterrr['radiplan'][0, :]
    gdat.dictfeatobjt['massplan'] = gdat.dicterrr['massplan'][0, :]
    gdat.dictfeatobjt['stdvradistar'] = np.mean(gdat.dicterrr['radistar'][1:, :], 0)
    gdat.dictfeatobjt['stdvmassstar'] = np.mean(gdat.dicterrr['massstar'][1:, :], 0)
    gdat.dictfeatobjt['stdvtmptstar'] = np.mean(gdat.dicterrr['tmptstar'][1:, :], 0)
    gdat.dictfeatobjt['stdvloggstar'] = np.mean(gdat.dicterrr['loggstar'][1:, :], 0)
    gdat.dictfeatobjt['stdvradiplan'] = np.mean(gdat.dicterrr['radiplan'][1:, :], 0)
    gdat.dictfeatobjt['stdvmassplan'] = np.mean(gdat.dicterrr['massplan'][1:, :], 0)
    gdat.dictfeatobjt['stdvtmptplan'] = np.mean(gdat.dicterrr['tmptplan'][1:, :], 0)
    gdat.dictfeatobjt['stdvesmm'] = np.mean(gdat.dicterrr['esmm'][1:, :], 0)
    gdat.dictfeatobjt['stdvtsmm'] = np.mean(gdat.dicterrr['tsmm'][1:, :], 0)
    
    gdat.dictfeatobjt['stdvradiplaneart'] = gdat.dictfeatobjt['stdvradiplan'] * gdat.factrjre
    gdat.dictfeatobjt['stdvmassplaneart'] = gdat.dictfeatobjt['stdvmassplan'] * gdat.factmjme


def retr_durafull(peri, rs2a, sini, rrat, imfa):
    
    durafull = peri / np.pi * np.arcsin(rs2a / sini * np.sqrt((1. - rrat)**2 - imfa**2))

    return durafull 


def retr_duratotl(peri, rs2a, sini, rrat, imfa):
    
    duratotl = peri / np.pi * np.arcsin(rs2a / sini * np.sqrt((1. + rrat)**2 - imfa**2))
    
    return duratotl

    
def retr_imfa(cosi, rs2a, ecce, sinw):
    
    imfa = cosi / rs2a * (1. - ecce)**2 / (1. + ecce * sinw)

    return imfa


def proc_alle(gdat, typeallemodl):
    
    print('Processing allesfitter model %s...' % typeallemodl)
    # allesfit run folder
    gdat.pathalle[typeallemodl] = gdat.pathallebase + 'allesfit_%s/' % typeallemodl
    
    strgproc = 'adtr' + typeallemodl
    
    # make sure the folder exists
    cmnd = 'mkdir -p %s' % gdat.pathalle[typeallemodl]
    os.system(cmnd)
    
    # write the input data file
    if typeallemodl != 'pfss':
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                path = gdat.pathalle[typeallemodl] + gdat.liststrginst[b][p] + '.csv'
                listarrytserbdtrtemp = gdat.arrytser['bdtr'][b][p]
                indx = np.argsort(listarrytserbdtrtemp[:, 0])
                listarrytserbdtrtemp = listarrytserbdtrtemp[indx, :]
                print('Writing to %s...' % path)
                np.savetxt(path, listarrytserbdtrtemp, delimiter=',', header='time,%s,%s_err' % (gdat.liststrgtseralle[b], gdat.liststrgtseralle[b]))
    
    ## params_star
    pathparastar = gdat.pathalle[typeallemodl] + 'params_star.csv'
    if not os.path.exists(pathparastar):
        print('Writing to %s...' % pathparastar)
        objtfile = open(pathparastar, 'w')
        objtfile.write('#R_star,R_star_lerr,R_star_uerr,M_star,M_star_lerr,M_star_uerr,Teff_star,Teff_star_lerr,Teff_star_uerr\n')
        objtfile.write('#R_sun,R_sun,R_sun,M_sun,M_sun,M_sun,K,K,K\n')
        objtfile.write('%g,%g,%g,%g,%g,%g,%g,%g,%g' % (gdat.radistar / gdat.factrsrj, gdat.stdvradistar / gdat.factrsrj, gdat.stdvradistar / gdat.factrsrj, \
                                                       gdat.massstar / gdat.factmsmj, gdat.stdvmassstar / gdat.factmsmj, gdat.stdvmassstar / gdat.factmsmj, \
                                                                                                      gdat.tmptstar, gdat.stdvtmptstar, gdat.stdvtmptstar))
        objtfile.close()

    ## params
    dictalleparadefa = dict()
    pathpara = gdat.pathalle[typeallemodl] + 'params.csv'
    if not os.path.exists(pathpara):
        cmnd = 'touch %s' % (pathpara)
        print(cmnd)
        os.system(cmnd)
    
        for j in gdat.indxplanalle:
            strgrrat = '%s_rr' % gdat.liststrgplan[j]
            strgrsma = '%s_rsuma' % gdat.liststrgplan[j]
            strgcosi = '%s_cosi' % gdat.liststrgplan[j]
            strgepoc = '%s_epoch' % gdat.liststrgplan[j]
            strgperi = '%s_period' % gdat.liststrgplan[j]
            strgecos = '%s_f_c' % gdat.liststrgplan[j]
            strgesin = '%s_f_s' % gdat.liststrgplan[j]
            strgrvsa = '%s_K' % gdat.liststrgplan[j]
            dictalleparadefa[strgrrat] = ['%f' % gdat.rratprio[j], '1', 'uniform 0 %f' % (4 * gdat.rratprio[j]), \
                                                                            '$R_{%s} / R_\star$' % gdat.liststrgplan[j], '']
            
            dictalleparadefa[strgrsma] = ['%f' % gdat.rsmaprio[j], '1', 'uniform 0 %f' % (4 * gdat.rsmaprio[j]), \
                                                                      '$(R_\star + R_{%s}) / a_{%s}$' % (gdat.liststrgplan[j], gdat.liststrgplan[j]), '']
            dictalleparadefa[strgcosi] = ['%f' % gdat.cosiprio[j], '1', 'uniform 0 %f' % max(0.1, 4 * gdat.cosiprio[j]), \
                                                                                        '$\cos{i_{%s}}$' % gdat.liststrgplan[j], '']
            dictalleparadefa[strgepoc] = ['%f' % gdat.epocprio[j], '1', \
                                            'uniform %f %f' % (gdat.epocprio[j] - gdat.stdvepocprio[j], gdat.epocprio[j] + gdat.stdvepocprio[j]), \
                                                                    '$T_{0;%s}$' % gdat.liststrgplan[j], '$\mathrm{BJD}$']
            dictalleparadefa[strgperi] = ['%f' % gdat.periprio[j], '1', \
                                            'uniform %f %f' % (gdat.periprio[j] - gdat.stdvperiprio[j], gdat.periprio[j] + gdat.stdvperiprio[j]), \
                                                                    '$P_{%s}$' % gdat.liststrgplan[j], 'days']
            dictalleparadefa[strgecos] = ['%f' % gdat.ecosprio[j], '0', 'uniform -0.9 0.9', \
                                                                '$\sqrt{e_{%s}} \cos{\omega_{%s}}$' % (gdat.liststrgplan[j], gdat.liststrgplan[j]), '']
            dictalleparadefa[strgesin] = ['%f' % gdat.esinprio[j], '0', 'uniform -0.9 0.9', \
                                                                '$\sqrt{e_{%s}} \sin{\omega_{%s}}$' % (gdat.liststrgplan[j], gdat.liststrgplan[j]), '']
            dictalleparadefa[strgrvsa] = ['%f' % gdat.rvsaprio[j], '0', \
                                    'uniform %f %f' % (max(0, gdat.rvsaprio[j] - 5 * gdat.stdvrvsaprio[j]), gdat.rvsaprio[j] + 5 * gdat.stdvrvsaprio[j]), \
                                                                '$K_{%s}$' % gdat.liststrgplan[j], '']
            if typeallemodl == '0003' or typeallemodl == '0004' or typeallemodl == '0006':
                for b in gdat.indxdatatser:
                    for p in gdat.indxinst[b]:
                        strgsbrt = '%s_sbratio_' % gdat.liststrgplan[j] + gdat.liststrginst[b][p]
                        dictalleparadefa[strgsbrt] = '%s_sbratio_%s,1e-3,1,uniform 0 1,$J_{%s; \mathrm{%s}}$,' % \
                                    (gdat.liststrgplan[j], gdat.liststrginst[b][p], gdat.liststrgplan[j], gdat.listlablinst[b][p])
                        
                        dictalleparadefa['%s_phase_curve_beaming_%s' % (gdat.liststrgplan[j], gdat.liststrginst[b][p])] = \
                                                                                                        '0,1,uniform 0 10,$A_\mathrm{b}$,'
                        dictalleparadefa['%s_phase_curve_atmospheric_%s' % (gdat.liststrgplan[j], gdat.liststrginst[b][p])] = \
                                                                                                        '0,1,uniform 0 10,$A_\mathrm{b}$,'
                        dictalleparadefa['%s_phase_curve_ellipsoidal_%s' % (gdat.liststrgplan[j], gdat.liststrginst[b][p])] = \
                                                                                                        '0,1,uniform 0 10,$A_\mathrm{b}$,'

        if typeallemodl == 'pfss':
            for p in gdat.indxinst[1]:
                lineadde.extend([ \
                                ['', 'host_vsini,%g,1,uniform %g %g,$v \sin i$$,\n' % (gdat.vsiistarprio, 0, \
                                                                                                                            10 * gdat.vsiistarprio)], \
                                ['', 'host_lambda_%s,%g,1,uniform %g %g,$v \sin i$$,\n' % (gdat.liststrginst[1][p], gdat.lambstarprio, 0, \
                                                                                                                            10 * gdat.lambstarprio)], \
                                ])
        
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                strgldc1 = 'host_ldc_q1_%s' % gdat.liststrginst[b][p]
                strgldc2 = 'host_ldc_q2_%s' % gdat.liststrginst[b][p]
                strgscal = 'ln_err_flux_%s' % gdat.liststrginst[b][p]
                strgbaseoffs = 'baseline_offset_flux_%s' % gdat.liststrginst[b][p]
                strggprosigm = 'baseline_gp_matern32_lnsigma_flux_%s' % gdat.liststrginst[b][p]
                strggprorhoo = 'baseline_gp_matern32_lnrho_flux_%s' % gdat.liststrginst[b][p]
                dictalleparadefa[strgldc1] = ['0.5', '1', 'uniform 0 1', '$q_{1; \mathrm{%s}}$' % gdat.listlablinst[b][p], '']
                dictalleparadefa[strgldc2] = ['0.5', '1', 'uniform 0 1', '$q_{2; \mathrm{%s}}$' % gdat.listlablinst[b][p], '']
                dictalleparadefa[strgscal] = ['-7', '1', 'uniform -10 -4', '$\ln{\sigma_\mathrm{%s}}$' % gdat.listlablinst[b][p], '']
                dictalleparadefa[strgbaseoffs] = ['0', '1', 'uniform -1 1', '$O_{\mathrm{%s}}$' % gdat.listlablinst[b][p], '']
                if b == 1:
                    dictalleparadefa['ln_jitter_rv_%s' % gdat.liststrginst[b][p]] = ['-10', '1', 'uniform -20 20', \
                                                                            '$\ln{\sigma_{\mathrm{RV;%s}}}$' % gdat.listlablinst[b][p], '']
                #lineadde.extend([ \
                #            ['', '%s,%f,1,uniform %f %f,$\ln{\sigma_{GP;\mathrm{TESS}}}$,\n' % \
                #                 (strggprosigm, -6, -12, 12)], \
                #            ['', '%s,%f,1,uniform %f %f,$\ln{\\rho_{GP;\mathrm{TESS}}}$,\n' % \
                #                 (strggprorhoo, -2, -12, 12)], \
                #           ])
                
        writ_filealle(gdat, 'params.csv', gdat.pathalle[typeallemodl], gdat.dictdictallepara[typeallemodl], dictalleparadefa)
    
    ## settings
    dictallesettdefa = dict()
    if typeallemodl == 'pfss':
        for j in gdat.indxplanalle:
            dictallesettdefa['%s_flux_weighted_PFS' % gdat.liststrgplan[j]] = 'True'
    
    pathsett = gdat.pathalle[typeallemodl] + 'settings.csv'
    if not os.path.exists(pathsett):
        cmnd = 'touch %s' % (pathsett)
        print(cmnd)
        os.system(cmnd)
        
        dictallesettdefa['fast_fit_width'] = '%.3g' % np.amax(gdat.duramask)
        dictallesettdefa['multiprocess'] = 'True'
        dictallesettdefa['multiprocess_cores'] = 'all'

        dictallesettdefa['mcmc_nwalkers'] = '100'
        dictallesettdefa['mcmc_total_steps'] = '100000'
        dictallesettdefa['mcmc_burn_steps'] = '10000'
        dictallesettdefa['mcmc_thin_by'] = '100'
        
        for p in gdat.indxinst[0]:
            dictallesettdefa['inst_phot'] = '%s' % gdat.liststrginst[0][p]
        
        for b in gdat.indxdatatser:
            if b == 0:
                strg = 'phot'
            if b == 1:
                strg = 'rv'
            for p in gdat.indxinst[b]:
                dictallesettdefa['inst_%s' % strg] = '%s' % gdat.liststrginst[b][p]
                dictallesettdefa['host_ld_law_%s' % gdat.liststrginst[b][p]] = 'quad'
                dictallesettdefa['host_grid_%s' % gdat.liststrginst[b][p]] = 'very_sparse'
                dictallesettdefa['baseline_flux_%s' % gdat.liststrginst[b][p]] = 'sample_offset'
        
        #dictallesettdefa['use_host_density_prior'] = 'False'
        
        if typeallemodl == '0003' or typeallemodl == '0004' or typeallemodl == '0006':
            dictallesettdefa['phase_curve'] = 'True'
            dictallesettdefa['phase_curve_style'] = 'sine_physical'
        
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                for j in gdat.indxplanalle:
                    dictallesettdefa['%s_grid_%s' % (gdat.liststrgplan[j], gdat.liststrginst[b][p])] = 'very_sparse'
            
            if gdat.numbinst[b] > 0:
                if b == 0:
                    strg = 'companions_phot'
                if b == 1:
                    strg = 'companions_rv'
                varb = ''
                cntr = 0
                for j in gdat.indxplanalle:
                    if cntr != 0:
                        varb += ' '
                    varb += '%s' % gdat.liststrgplan[j]
                    cntr += 1
                dictallesettdefa[strg] = varb
        
        dictallesettdefa['fast_fit'] = 'True'

        writ_filealle(gdat, 'settings.csv', gdat.pathalle[typeallemodl], gdat.dictdictallesett[typeallemodl], dictallesettdefa)
    
    ## initial plot
    path = gdat.pathalle[typeallemodl] + 'results/initial_guess_b.pdf'
    if not os.path.exists(path):
        allesfitter.show_initial_guess(gdat.pathalle[typeallemodl])
    
    ## do the run
    path = gdat.pathalle[typeallemodl] + 'results/mcmc_save.h5'
    if not os.path.exists(path):
        allesfitter.mcmc_fit(gdat.pathalle[typeallemodl])
    else:
        print('%s exists... Skipping the orbit run.' % path)

    if typeallemodl == 'pfss':
        return

    ## make the final plots
    path = gdat.pathalle[typeallemodl] + 'results/mcmc_corner.pdf'
    if not os.path.exists(path):
        allesfitter.mcmc_output(gdat.pathalle[typeallemodl])
        
    # read the allesfitter posterior
    print('Reading from %s...' % gdat.pathalle[typeallemodl])
    gdat.objtalle[typeallemodl] = allesfitter.allesclass(gdat.pathalle[typeallemodl])
    
    gdat.numbsampalle = allesfitter.config.BASEMENT.settings['mcmc_total_steps']
    gdat.numbwalkalle = allesfitter.config.BASEMENT.settings['mcmc_nwalkers']
    gdat.numbsampalleburn = allesfitter.config.BASEMENT.settings['mcmc_burn_steps']
    gdat.numbsampallethin = allesfitter.config.BASEMENT.settings['mcmc_thin_by']

    print('gdat.numbwalkalle')
    print(gdat.numbwalkalle)
    print('gdat.numbsampalle')
    print(gdat.numbsampalle)
    print('gdat.numbsampalleburn')
    print(gdat.numbsampalleburn)
    print('gdat.numbsampallethin')
    print(gdat.numbsampallethin)

    gdat.numbsamp = gdat.objtalle[typeallemodl].posterior_params[list(gdat.objtalle[typeallemodl].posterior_params.keys())[0]].size
    
    print('gdat.numbsamp')
    print(gdat.numbsamp)

    # temp 
    if gdat.numbsamp > 10000:
        print('Thinning down the allesfitter chain!')
        gdat.indxsamp = np.random.choice(np.arange(gdat.numbsamp), size=10000, replace=False)
        gdat.numbsamp = 10000
    else:
        gdat.indxsamp = np.arange(gdat.numbsamp)
    
    print('gdat.numbsamp')
    print(gdat.numbsamp)

    calc_prop(gdat, typeallemodl)

    gdat.arrytsermodlmedi = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            gdat.arrytsermodlmedi[b][p] = np.empty((gdat.time[b][p].size, 3))
            gdat.arrytsermodlmedi[b][p][:, 0] = gdat.time[b][p]
            gdat.arrytsermodlmedi[b][p][:, 1] = gdat.objtalle[typeallemodl].get_posterior_median_model(gdat.liststrginst[b][p], gdat.liststrgtseralle[b], \
                                                                                                                                        xx=gdat.time[b][p])
            gdat.arrytsermodlmedi[b][p][:, 2] = 0.

    # write the model to file
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            path = gdat.pathdata + 'arry%smodl_%s.csv' % (gdat.liststrgdatatser[b], gdat.liststrginst[b][p])
            print('Writing to %s...' % path)
            np.savetxt(path, gdat.arrytsermodlmedi[b][p], delimiter=',', header='time,%s,%s_err' % (gdat.liststrgtseralle[b], gdat.liststrgtseralle[b]))

    # number of samples to plot
    gdat.numbsampplot = min(10, gdat.numbsamp)
    gdat.indxsampplot = np.random.choice(gdat.indxsamp, gdat.numbsampplot, replace=False)
    
    gdat.listarrypserquadmodl = [[[[] for j in gdat.indxplan] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            for j in gdat.indxplan:
                gdat.listarrypserquadmodl[b][p][j] = np.empty((gdat.numbsampplot, gdat.numbtimeclen[b][p][j], 3))
    
    print('gdat.numbsampplot')
    print(gdat.numbsampplot)
    gdat.arrypcurprimmodl = [[[[] for j in gdat.indxplan] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gdat.arrypcurquadmodl = [[[[] for j in gdat.indxplan] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gdat.arrypcurprim[strgproc] = [[[[] for j in gdat.indxplan] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gdat.arrypcurprimbindtotl[strgproc] = [[[[] for j in gdat.indxplan] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gdat.arrypcurprimbindzoom[strgproc] = [[[[] for j in gdat.indxplan] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    
    gdat.arrytser[strgproc] = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gdat.arrytser['abas'] = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    # these are just for plotting -- evaluated at times of each chunck
    gdat.listarrytser[strgproc] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gdat.listarrytser['abas'] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    
    gdat.listarrytsermodl = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            gdat.listarrytsermodl[b][p] = np.empty((gdat.numbsampplot, gdat.numbtime[b][p], 3))
    
    gdat.arrypcurquadadtr = [[[[] for j in gdat.indxplan] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gdat.arrypcurquadadtrbindtotl = [[[[] for j in gdat.indxplan] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gdat.arrypcurquadadtrbindzoom = [[[[] for j in gdat.indxplan] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            gdat.listlcurmodl = np.empty((gdat.numbsampplot, gdat.time[b][p].size))
            print('Phase-folding the posterior samples from the model light curve...')
            for ii in tqdm(range(gdat.numbsampplot)):
                i = gdat.indxsampplot[ii]
                
                # this is only the physical model and excludes the baseline, which is available separately via get_one_posterior_baseline()
                gdat.listarrytsermodl[b][p][ii, :, 1] = gdat.objtalle[typeallemodl].get_one_posterior_model(gdat.liststrginst[b][p], \
                                                                        gdat.liststrgtseralle[b], xx=gdat.time[b][p], sample_id=i)
                
                for j in gdat.indxplan:
                    gdat.listarrypserquadmodl[b][p][j][ii, :, :] = \
                                            tesstarg.util.fold_tser(gdat.listarrytsermodl[b][p][ii, gdat.listindxtimeclen[j][b][p], :], \
                                                                                   gdat.dicterrr['epoc'][0, j], gdat.dicterrr['peri'][0, j], phasshft=0.25)

            
            # get allesfitter baseline model
            gdat.arrytser['abas'][b][p] = np.copy(gdat.arrytser['bdtr'][b][p])
            gdat.arrytser['abas'][b][p][:, 1] = gdat.objtalle[typeallemodl].get_posterior_median_baseline(gdat.liststrginst[b][p], 'flux', \
                                                                                                                                xx=gdat.time[b][p])
            # get allesfitter-detrended data
            gdat.arrytser[strgproc][b][p] = np.copy(gdat.arrytser['bdtr'][b][p])
            gdat.arrytser[strgproc][b][p][:, 1] = gdat.arrytser['bdtr'][b][p][:, 1] - gdat.arrytser['abas'][b][p][:, 1]
            for y in gdat.indxchun[b][p]:
                # get allesfitter baseline model
                gdat.listarrytser['abas'][b][p][y] = np.copy(gdat.listarrytser['bdtr'][b][p][y])
                gdat.listarrytser['abas'][b][p][y][:, 1] = gdat.objtalle[typeallemodl].get_posterior_median_baseline(gdat.liststrginst[b][p], 'flux', \
                                                                                                        xx=gdat.listarrytser['abas'][b][p][y][:, 0])
                # get allesfitter-detrended data
                gdat.listarrytser[strgproc][b][p][y] = np.copy(gdat.listarrytser['bdtr'][b][p][y])
                gdat.listarrytser[strgproc][b][p][y][:, 1] = gdat.listarrytser[strgproc][b][p][y][:, 1] - gdat.listarrytser['abas'][b][p][y][:, 1]
            
            for j in gdat.indxplan:
                delt = gdat.delttimebindzoom / gdat.periprio[j]
                gdat.arrypcurprimmodl[b][p][j] = tesstarg.util.fold_tser(gdat.arrytsermodlmedi[b][p][gdat.listindxtimeclen[j][b][p], :], \
                                                                                                gdat.dicterrr['epoc'][0, j], gdat.dicterrr['peri'][0, j])
                gdat.arrypcurquadmodl[b][p][j] = tesstarg.util.fold_tser(gdat.arrytsermodlmedi[b][p][gdat.listindxtimeclen[j][b][p], :], \
                                                                                    gdat.dicterrr['epoc'][0, j], gdat.dicterrr['peri'][0, j], phasshft=0.25)
                gdat.arrypcurprim[strgproc][b][p][j] = tesstarg.util.fold_tser(gdat.arrytser[strgproc][b][p][gdat.listindxtimeclen[j][b][p], :], \
                                                                                                gdat.dicterrr['epoc'][0, j], gdat.dicterrr['peri'][0, j])
                gdat.arrypcurprimbindtotl[strgproc][b][p][j] = tesstarg.util.rebn_tser(gdat.arrypcurprim[strgproc][b][p][j], numbbins=gdat.numbbinspcurtotl)
                gdat.arrypcurprimbindzoom[strgproc][b][p][j] = tesstarg.util.rebn_tser(gdat.arrypcurprim[strgproc][b][p][j], delt=delt)
                gdat.arrypcurquadadtr[b][p][j] = tesstarg.util.fold_tser(gdat.arrytser[strgproc][b][p][gdat.listindxtimeclen[j][b][p], :], \
                                                                                    gdat.dicterrr['epoc'][0, j], gdat.dicterrr['peri'][0, j], phasshft=0.25)
                gdat.arrypcurquadadtrbindtotl[b][p][j] = tesstarg.util.rebn_tser(gdat.arrypcurquadadtr[b][p][j], numbbins=gdat.numbbinspcurtotl)
                gdat.arrypcurquadadtrbindzoom[b][p][j] = tesstarg.util.rebn_tser(gdat.arrypcurquadadtr[b][p][j], delt=delt)
    
    # plots
    ## plot GP-detrended phase curves
    if gdat.booldatatser:
        plot_tser(gdat, strgproc)
        plot_pser(gdat, strgproc)
    if gdat.boolplotprop:
        plot_prop(gdat, gdat.strgprio + typeallemodl)
    
    # print out transit times
    for j in gdat.indxplan:
        print(gdat.liststrgplan[j])
        time = np.empty(500)
        for n in range(500):
            time[n] = gdat.dicterrr['epoc'][0, j] + gdat.dicterrr['peri'][0, j] * n
        objttime = astropy.time.Time(time, format='jd', scale='utc', out_subfmt='date_hm')
        listtimelabl = objttime.iso
        for n in range(500):
            if time[n] > 2458788 and time[n] < 2458788 + 200:
                print('%f, %s' % (time[n], listtimelabl[n]))


    if typeallemodl == '0003' or typeallemodl == '0004' or typeallemodl == '0006':
        gdat.arrytsermodlmedicomp = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.arrypcurquadmodlcomp = [[[[] for j in gdat.indxplan] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                if typeallemodl == '0003' or typeallemodl == '0006': 
                    listlablpara = [['Nightside', 'ppm'], ['Secondary', 'ppm'], ['Planetary Modulation', 'ppm'], ['Thermal', 'ppm'], \
                                                                ['Reflected', 'ppm'], ['Phase shift', 'deg'], ['Geometric Albedo', '']]
                else:
                    listlablpara = [['Nightside', 'ppm'], ['Secondary', 'ppm'], ['Thermal', 'ppm'], \
                                          ['Reflected', 'ppm'], ['Thermal Phase shift', 'deg'], ['Reflected Phase shift', 'deg'], ['Geometric Albedo', '']]
                numbpara = len(listlablpara)
                indxpara = np.arange(numbpara)
                listpost = np.empty((gdat.numbsamp, numbpara))
                
                for j in gdat.indxplan:
                    listpost[:, 0] = gdat.dictlist['amplnigh'][:, j] * 1e6 # [ppm]
                    listpost[:, 1] = gdat.dictlist['amplseco'][:, j] * 1e6 # [ppm]
                    if typeallemodl == '0003' or typeallemodl == '0006': 
                        listpost[:, 2] = gdat.dictlist['amplplan'][:, j] * 1e6 # [ppm]
                        listpost[:, 3] = gdat.dictlist['amplplanther'][:, j] * 1e6 # [ppm]
                        listpost[:, 4] = gdat.dictlist['amplplanrefl'][:, j] * 1e6 # [ppm]
                        listpost[:, 5] = gdat.dictlist['phasshftplan'][:, j]
                        listpost[:, 6] = gdat.dictlist['albg'][:, j]
                    else:
                        listpost[:, 2] = gdat.dictlist['amplplanther'][:, j] * 1e6 # [ppm]
                        listpost[:, 3] = gdat.dictlist['amplplanrefl'][:, j] * 1e6 # [ppm]
                        listpost[:, 4] = gdat.dictlist['phasshftplanther'][:, j]
                        listpost[:, 5] = gdat.dictlist['phasshftplanrefl'][:, j]
                        listpost[:, 6] = gdat.dictlist['albg'][:, j]
                    tdpy.mcmc.plot_grid(gdat.pathalle[typeallemodl], 'pcur_%s' % typeallemodl, listpost, listlablpara, plotsize=2.5)

                # plot phase curve
                ## determine data gaps for overplotting model without the data gaps
                gdat.indxtimegapp = np.argmax(gdat.time[b][p][1:] - gdat.time[b][p][:-1]) + 1
                figr = plt.figure(figsize=(10, 12))
                axis = [[] for k in range(3)]
                axis[0] = figr.add_subplot(3, 1, 1)
                axis[1] = figr.add_subplot(3, 1, 2)
                axis[2] = figr.add_subplot(3, 1, 3, sharex=axis[1])
                
                for k in range(len(axis)):
                    
                    ## unbinned data
                    if k < 2:
                        if k == 0:
                            xdat = gdat.time[b][p] - gdat.timetess
                            ydat = gdat.arrytser[strgproc][b][p][:, 1] + gdat.dicterrr['amplnigh'][0, 0]
                        if k == 1:
                            xdat = gdat.arrypcurquadadtr[b][p][j][:, 0]
                            ydat = gdat.arrypcurquadadtr[b][p][j][:, 1] + gdat.dicterrr['amplnigh'][0, 0]
                        axis[k].plot(xdat, ydat, '.', color='grey', alpha=0.3, label='Raw data')
                    
                    ## binned data
                    if k > 0:
                        xdat = gdat.arrypcurquadadtrbindtotl[b][p][j][:, 0]
                        ydat = gdat.arrypcurquadadtrbindtotl[b][p][j][:, 1] + gdat.dicterrr['amplnigh'][0, 0]
                        yerr = np.copy(gdat.arrypcurquadadtrbindtotl[b][p][j][:, 2])
                    else:
                        xdat = None
                        ydat = None
                        yerr = None
                    if k == 2:
                        ydat = (ydat - 1) * 1e6
                        yerr *= 1e6
                    # temp - add offset to bring the base of secondary to 0 
                    axis[k].errorbar(xdat, ydat, marker='o', yerr=yerr, capsize=0, ls='', color='k', label='Binned data')
                    
                    ## model
                    if k > 0:
                        xdat = gdat.arrypcurquadmodl[b][p][j][:, 0]
                        ydat = gdat.arrypcurquadmodl[b][p][j][:, 1] + gdat.dicterrr['amplnigh'][0, 0]
                    else:
                        xdat = gdat.arrytsermodlmedi[b][p][:, 0] - gdat.timetess
                        ydat = gdat.arrytsermodlmedi[b][p][:, 1] + gdat.dicterrr['amplnigh'][0, 0]
                    if k == 2:
                        ydat = (ydat - 1) * 1e6
                    if k == 0:
                        axis[k].plot(xdat[:gdat.indxtimegapp], ydat[:gdat.indxtimegapp], color='b', lw=2, label='Total Model', zorder=10)
                        axis[k].plot(xdat[gdat.indxtimegapp:], ydat[gdat.indxtimegapp:], color='b', lw=2, zorder=10)
                    else:
                        axis[k].plot(xdat, ydat, color='b', lw=2, label='Model', zorder=10)
                    
                    # add Vivien's result
                    if k == 2 and gdat.labltarg == 'WASP-121':
                        axis[k].plot(gdat.phasvivi, gdat.deptvivi*1e6, color='orange', lw=2, label='GCM (Parmentier+2018)')
                        axis[k].axhline(0., ls='-.', alpha=0.3, color='grey')

                    if k == 0:
                        axis[k].set(xlabel='Time [BJD - %d]' % gdat.timetess)
                    if k > 0:
                        axis[k].set(xlabel='Phase')
                axis[0].set(ylabel='Relative Flux')
                axis[1].set(ylabel='Relative Flux')
                axis[2].set(ylabel='Relative Flux - 1 [ppm]')
                
                if gdat.labltarg == 'WASP-121':
                    ylimpcur = [-400, 1000]
                else:
                    ylimpcur = [-100, 300]
                axis[2].set_ylim(ylimpcur)
                
                gdat.arrypcurquadmodlcomp[b][p][j] = dict()
                gdat.arrypcurquadmodlcomp[b][p][j]['totl'] = gdat.arrypcurquadmodl[b][p][j]

                ## plot components in the zoomed panel
                arrytsermodlmeditemp = np.copy(gdat.arrytsermodlmedi[b][p])
                
                ### stellar baseline
                gdat.objtalle[typeallemodl] = allesfitter.allesclass(gdat.pathalle[typeallemodl])
                gdat.objtalle[typeallemodl].posterior_params_median['b_phase_curve_ellipsoidal_TESS'] = 0
                if typeallemodl == '0003' or typeallemodl == '0006':
                    gdat.objtalle[typeallemodl].posterior_params_median['b_phase_curve_atmospheric_TESS'] = 0
                if typeallemodl == '0004':
                    gdat.objtalle[typeallemodl].posterior_params_median['b_phase_curve_atmospheric_thermal_TESS'] = 0
                    gdat.objtalle[typeallemodl].posterior_params_median['b_phase_curve_atmospheric_reflected_TESS'] = 0
                gdat.objtalle[typeallemodl].posterior_params_median['b_sbratio_TESS'] = 0
                arrytsermodlmeditemp[:, 1] = gdat.objtalle[typeallemodl].get_posterior_median_model(strginst, 'flux', xx=gdat.time[b][p])
                gdat.arrypcurquadmodlcomp[b][p][j]['stel'] = tesstarg.util.fold_tser(arrytsermodlmeditemp[gdat.listindxtimeclen[j][b][p], :], \
                                                                        gdat.dicterrr['epoc'][0, j], gdat.dicterrr['peri'][0, j], phasshft=0.25)
                xdat = gdat.arrypcurquadmodlcomp[b][p][j]['stel'][:, 0]
                ydat = (gdat.arrypcurquadmodlcomp[b][p][j]['stel'][:, 1] - 1.) * 1e6
                axis[2].plot(xdat, ydat, lw=2, color='orange', label='Stellar baseline', ls='--', zorder=11)
                
                ### EV
                gdat.objtalle[typeallemodl] = allesfitter.allesclass(gdat.pathalle[typeallemodl])
                gdat.objtalle[typeallemodl].posterior_params_median['b_sbratio_TESS'] = 0
                if typeallemodl == '0003' or typeallemodl == '0006':
                    gdat.objtalle[typeallemodl].posterior_params_median['b_phase_curve_atmospheric_TESS'] = 0
                if typeallemodl == '0004':
                    gdat.objtalle[typeallemodl].posterior_params_median['b_phase_curve_atmospheric_thermal_TESS'] = 0
                    gdat.objtalle[typeallemodl].posterior_params_median['b_phase_curve_atmospheric_reflected_TESS'] = 0
                arrytsermodlmeditemp[:, 1] = gdat.objtalle[typeallemodl].get_posterior_median_model(strginst, 'flux', xx=gdat.time[b][p])
                gdat.arrypcurquadmodlcomp[b][p][j]['elli'] = tesstarg.util.fold_tser(arrytsermodlmeditemp[gdat.listindxtimeclen[j][b][p], :], \
                                                                          gdat.dicterrr['epoc'][0, j], gdat.dicterrr['peri'][0, j], phasshft=0.25)
                gdat.arrypcurquadmodlcomp[b][p][j]['elli'][:, 1] -= gdat.arrypcurquadmodlcomp[b][p][j]['stel'][:, 1]
                xdat = gdat.arrypcurquadmodlcomp[b][p][j]['elli'][:, 0]
                ydat = (gdat.arrypcurquadmodlcomp[b][p][j]['elli'][:, 1] - 1.) * 1e6
                axis[2].plot(xdat, ydat, lw=2, color='r', ls='--', label='Ellipsoidal variation')
                
                # planetary
                gdat.objtalle[typeallemodl] = allesfitter.allesclass(gdat.pathalle[typeallemodl])
                gdat.objtalle[typeallemodl].posterior_params_median['b_phase_curve_ellipsoidal_TESS'] = 0
                arrytsermodlmeditemp[:, 1] = gdat.objtalle[typeallemodl].get_posterior_median_model(strginst, 'flux', xx=gdat.time[b][p])
                gdat.arrypcurquadmodlcomp[b][p][j]['plan'] = tesstarg.util.fold_tser(arrytsermodlmeditemp[gdat.listindxtimeclen[j][b][p], :], \
                                                                        gdat.dicterrr['epoc'][0, j], gdat.dicterrr['peri'][0, j], phasshft=0.25)
                gdat.arrypcurquadmodlcomp[b][p][j]['plan'] += gdat.dicterrr['amplnigh'][0, 0]
                gdat.arrypcurquadmodlcomp[b][p][j]['plan'][:, 1] -= gdat.arrypcurquadmodlcomp[b][p][j]['stel'][:, 1]
                
                xdat = gdat.arrypcurquadmodlcomp[b][p][j]['plan'][:, 0]
                ydat = (gdat.arrypcurquadmodlcomp[b][p][j]['plan'][:, 1] - 1.) * 1e6
                axis[2].plot(xdat, ydat, lw=2, color='g', label='Planetary', ls='--')
    
                # planetary nightside
                gdat.objtalle[typeallemodl] = allesfitter.allesclass(gdat.pathalle[typeallemodl])
                gdat.objtalle[typeallemodl].posterior_params_median['b_phase_curve_ellipsoidal_TESS'] = 0
                if typeallemodl == '0003' or typeallemodl == '0006':
                    gdat.objtalle[typeallemodl].posterior_params_median['b_phase_curve_atmospheric_TESS'] = 0
                if typeallemodl == '0004':
                    gdat.objtalle[typeallemodl].posterior_params_median['b_phase_curve_atmospheric_thermal_TESS'] = 0
                    gdat.objtalle[typeallemodl].posterior_params_median['b_phase_curve_atmospheric_reflected_TESS'] = 0
                arrytsermodlmeditemp[:, 1] = gdat.objtalle[typeallemodl].get_posterior_median_model(strginst, 'flux', xx=gdat.time[b][p])
                gdat.arrypcurquadmodlcomp[b][p][j]['nigh'] = tesstarg.util.fold_tser(arrytsermodlmeditemp[gdat.listindxtimeclen[j][b][p], :], \
                                                                        gdat.dicterrr['epoc'][0, j], gdat.dicterrr['peri'][0, j], phasshft=0.25)
                gdat.arrypcurquadmodlcomp[b][p][j]['nigh'] += gdat.dicterrr['amplnigh'][0, 0]
                gdat.arrypcurquadmodlcomp[b][p][j]['nigh'][:, 1] -= gdat.arrypcurquadmodlcomp[b][p][j]['stel'][:, 1]
                xdat = gdat.arrypcurquadmodlcomp[b][p][j]['nigh'][:, 0]
                ydat = (gdat.arrypcurquadmodlcomp[b][p][j]['nigh'][:, 1] - 1.) * 1e6
                axis[2].plot(xdat, ydat, lw=2, color='olive', label='Planetary baseline', ls='--', zorder=11)
    
                ### planetary modulation
                gdat.arrypcurquadmodlcomp[b][p][j]['pmod'] = np.copy(gdat.arrypcurquadmodlcomp[b][p][j]['plan'])
                gdat.arrypcurquadmodlcomp[b][p][j]['pmod'][:, 1] -= gdat.arrypcurquadmodlcomp[b][p][j]['nigh'][:, 1]
                xdat = gdat.arrypcurquadmodlcomp[b][p][j]['pmod'][:, 0]
                ydat = (gdat.arrypcurquadmodlcomp[b][p][j]['pmod'][:, 1] - 1.) * 1e6
                axis[2].plot(xdat, ydat, lw=2, color='m', label='Planetary modulation', ls='--', zorder=11)
                 
                ## legend
                axis[2].legend(ncol=3)
                
                path = gdat.pathalle[typeallemodl] + 'pcur_grid_%s_%s_%s.%s' % (typeallemodl, gdat.strgtarg, gdat.strgprio, gdat.typefileplot)
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()
                
                
                # replot phase curve
                ### sample model phas
                #numbphasfine = 1000
                #gdat.meanphasfine = np.linspace(np.amin(gdat.arrypcurquad['bdtr'][0][gdat.indxphasotpr, 0]), \
                #                                np.amax(gdat.arrypcurquad['bdtr'][0][gdat.indxphasotpr, 0]), numbphasfine)
                #indxphasfineinse = np.where(abs(gdat.meanphasfine - 0.5) < phasseco)[0]
                #indxphasfineotprleft = np.where(-gdat.meanphasfine > phasmask)[0]
                #indxphasfineotprrght = np.where(gdat.meanphasfine > phasmask)[0]
       
                indxphasmodlouttprim = [[] for a in range(2)]
                indxphasdatabindouttprim = [[] for a in range(2)]
                indxphasmodlouttprim[0] = np.where(gdat.arrypcurquadmodlcomp[b][p][j]['totl'][:, 0] < -0.05)[0]
                indxphasdatabindouttprim[0] = np.where(gdat.arrypcurquadbind['bdtr'][b][p][j][:, 0] < -0.05)[0]
                indxphasmodlouttprim[1] = np.where(gdat.arrypcurquadmodlcomp[b][p][j]['totl'][:, 0] > 0.05)[0]
                indxphasdatabindouttprim[1] = np.where(gdat.arrypcurquadbind['bdtr'][b][p][j][:, 0] > 0.05)[0]

                # plot the phase curve with components
                figr, axis = plt.subplots(figsize=gdat.figrsizeydob)
                ## data
                axis.errorbar(gdat.arrypcurquadbind['bdtr'][b][p][j][:, 0], \
                               (gdat.arrypcurquadbind['bdtr'][b][p][j][:, 1] + gdat.dicterrr['amplnigh'][0, 0] - 1) * 1e6, \
                               yerr=1e6*gdat.arrypcurquadbind['bdtr'][b][p][j][:, 2], color='k', marker='o', ls='', markersize=2, lw=1, label='Data')
                ## total model
                axis.plot(gdat.arrypcurquadmodlcomp[b][p][j]['totl'][:, 0], \
                                                1e6*(gdat.arrypcurquadmodlcomp[b][p][j]['totl'][:, 1]+gdat.dicterrr['amplnigh'][0, 0]-1), \
                                                                                                                color='b', lw=3, label='Model')
                
                #axis.plot(gdat.arrypcurquadmodlcomp[b][p][j]['plan'][:, 0], 1e6*(gdat.arrypcurquadmodlcomp[b][p][j]['plan'][:, 1]), \
                #                                                                                              color='g', label='Planetary', lw=1, ls='--')
                
                axis.plot(gdat.arrypcurquadmodlcomp[b][p][j]['pmod'][:, 0], 1e6*(gdat.arrypcurquadmodlcomp[b][p][j]['pmod'][:, 1]), \
                                                                                                      color='m', label='Planetary modulation', lw=2, ls='--')
                axis.plot(gdat.arrypcurquadmodlcomp[b][p][j]['nigh'][:, 0], 1e6*(gdat.arrypcurquadmodlcomp[b][p][j]['nigh'][:, 1]), \
                                                                                                      color='olive', label='Planetary baseline', lw=2, ls='--')
                
                axis.plot(gdat.arrypcurquadmodlcomp[b][p][j]['elli'][:, 0], 1e6*(gdat.arrypcurquadmodlcomp[b][p][j]['elli'][:, 1]), \
                                                                                                      color='r', label='Ellipsoidal variation', lw=2, ls='--')
                
                axis.plot(gdat.arrypcurquadmodlcomp[b][p][j]['stel'][:, 0], 1e6*(gdat.arrypcurquadmodlcomp[b][p][j]['stel'][:, 1]-1.), \
                                                                                                      color='orange', label='Stellar baseline', lw=2, ls='--')
                
                axis.set_ylim(ylimpcur)
                axis.set_ylabel('Relative Flux [ppm]')
                axis.set_xlabel('Phase')
                axis.legend(ncol=3)
                plt.tight_layout()
                path = gdat.pathalle[typeallemodl] + 'pcur_comp_%s_%s_%s.%s' % (typeallemodl, gdat.strgtarg, gdat.strgprio, gdat.typefileplot)
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()

                # plot the phase curve with samples
                figr, axis = plt.subplots(figsize=gdat.figrsizeydob)
                axis.errorbar(gdat.arrypcurquadbind['bdtr'][b][p][j][:, 0], (gdat.arrypcurquadbind['bdtr'][b][p][j][:, 1] + gdat.dicterrr['amplnigh'][0, 0] - 1) * 1e6, \
                                             yerr=1e6*gdat.arrypcurquadbind['bdtr'][b][p][j][:, 2], color='k', marker='o', ls='', markersize=2, lw=1)
                for ii, i in enumerate(gdat.indxsampplot):
                    axis.plot(gdat.arrypcurquadmodlcomp[b][p][j]['totl'][:, 0], \
                                                1e6 * (gdat.listarrypserquadmodl[b][p][j][ii, :] + gdat.dicterrr['amplnigh'][0, 0] - 1.), \
                                                                                                                                  alpha=0.1, color='b')
                axis.set_ylabel('Relative Flux [ppm]')
                axis.set_xlabel('Phase')
                axis.set_ylim(ylimpcur)
                plt.tight_layout()
                path = gdat.pathalle[typeallemodl] + 'pcur_samp_%s_%s_%s.%s' % (typeallemodl, gdat.strgtarg, gdat.strgprio, gdat.typefileplot)
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()

                # plot all along with residuals
                #figr, axis = plt.subplots(3, 1, figsize=gdat.figrsizeydob)
                #axis.errorbar(gdat.arrypcurquadbind['bdtr'][b][p][j][:, 0], (gdat.arrypcurquadbind['bdtr'][b][p][j][:, 1]) * 1e6, \
                #                             yerr=1e6*gdat.arrypcurquadbind['bdtr'][b][p][j][:, 2], color='k', marker='o', ls='', markersize=2, lw=1)
                #for kk, k in enumerate(gdat.indxsampplot):
                #    axis.plot(gdat.meanphasfine[indxphasfineotprleft], (listmodltotl[k, indxphasfineotprleft] - listoffs[k]) * 1e6, \
                #                                                                                                            alpha=0.1, color='b')
                #    axis.plot(gdat.meanphasfine[indxphasfineotprrght], (listmodltotl[k, indxphasfineotprrght] - listoffs[k]) * 1e6, \
                #                                                                                                            alpha=0.1, color='b')
                #axis.set_ylabel('Relative Flux - 1 [ppm]')
                #axis.set_xlabel('Phase')
                #plt.tight_layout()
                #path = gdat.pathalle[typeallemodl] + 'pcur_resi_%s_%s_%s.%s' % (typeallemodl, gdat.strgtarg, gdat.strgprio, gdat.typefileplot)
                #print('Writing to %s...' % path)
                #plt.savefig(path)
                #plt.close()

                # write to text file
                fileoutp = open(gdat.pathalle[typeallemodl] + 'post_pcur_%s_tabl.csv' % (typeallemodl), 'w')
                for strgfeat in gdat.dictlist:
                    if gdat.dictlist[strgfeat].ndim == 2:
                        for j in gdat.indxplan:
                            fileoutp.write('%s,%s,%g,%g,%g,%g,%g\\\\\n' % (strgfeat, gdat.liststrgplan[j], gdat.dictlist[strgfeat][0, j], gdat.dictlist[strgfeat][1, j], \
                                                                        gdat.dictlist[strgfeat][2, j], gdat.dicterrr[strgfeat][1, j], gdat.dicterrr[strgfeat][2, j]))
                    else:
                        fileoutp.write('%s,,%g,%g,%g,%g,%g\\\\\n' % (strgfeat, gdat.dictlist[strgfeat][0], gdat.dictlist[strgfeat][1], \
                                                                        gdat.dictlist[strgfeat][2], gdat.dicterrr[strgfeat][1], gdat.dicterrr[strgfeat][2]))
                    #fileoutp.write('\\\\\n')
                fileoutp.close()
                
                fileoutp = open(gdat.pathalle[typeallemodl] + 'post_pcur_%s_cmnd.csv' % (typeallemodl), 'w')
                for strgfeat in gdat.dictlist:
                    if gdat.dictlist[strgfeat].ndim == 2:
                        for j in gdat.indxplan:
                            fileoutp.write('%s,%s,$%.3g \substack{+%.3g \\\\ -%.3g}$\\\\\n' % (strgfeat, gdat.liststrgplan[j], gdat.dicterrr[strgfeat][0, j], \
                                                                                        gdat.dicterrr[strgfeat][1, j], gdat.dicterrr[strgfeat][2, j]))
                    else:
                        fileoutp.write('%s,,$%.3g \substack{+%.3g \\\\ -%.3g}$\\\\\n' % (strgfeat, gdat.dicterrr[strgfeat][0], \
                                                                                                    gdat.dicterrr[strgfeat][1], gdat.dicterrr[strgfeat][2]))
                    #fileoutp.write('\\\\\n')
                fileoutp.close()

                if gdat.labltarg == 'WASP-121' and typeallemodl == '0003':
                    
                    # wavelength axis
                    gdat.conswlentmpt = 0.0143877735e6 # [um K]

                    print('gdat.dictlist[albg]')
                    summgene(gdat.dictlist['albg'])
                    print('gdat.dictlist[albginfo]')
                    summgene(gdat.dictlist['albginfo'])
                    minmalbg = min(np.amin(gdat.dictlist['albginfo']), np.amin(gdat.dictlist['albg']))
                    maxmalbg = max(np.amax(gdat.dictlist['albginfo']), np.amax(gdat.dictlist['albg']))
                    binsalbg = np.linspace(minmalbg, maxmalbg, 100)
                    meanalbg = (binsalbg[1:] + binsalbg[:-1]) / 2.
                    print('binsalbg')
                    summgene(binsalbg)
                    print('meanalbg')
                    summgene(meanalbg)
                    pdfnalbg = tdpy.util.retr_kdegpdfn(gdat.dictlist['albg'][:, 0], binsalbg, 0.02)
                    pdfnalbginfo = tdpy.util.retr_kdegpdfn(gdat.dictlist['albginfo'][:, 0], binsalbg, 0.02)
                    
                    figr, axis = plt.subplots(figsize=gdat.figrsizeydob)
                    axis.plot(meanalbg, pdfnalbg, label='TESS only', lw=2)
                    axis.plot(meanalbg, pdfnalbginfo, label='TESS + ATMO', lw=2)
                    axis.set_xlabel('$A_g$')
                    axis.set_ylabel('$P(A_g)$')
                    axis.legend()
                    axis.set_xlim([0, None])
                    plt.subplots_adjust()
                    path = gdat.pathalle[typeallemodl] + 'pdfn_albg_%s_%s.%s' % (gdat.strgtarg, gdat.strgprio, gdat.typefileplot)
                    print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()
                
                    figr, axis = plt.subplots(figsize=gdat.figrsizeydob)
                    axis.hist(gdat.dictlist['albg'][:, 0], label='TESS only', bins=binsalbg)
                    axis.hist(gdat.dictlist['albginfo'][:, 0], label='TESS + ATMO', bins=binsalbg)
                    axis.set_xlabel('$A_g$')
                    axis.set_ylabel('$N(A_g)$')
                    axis.legend()
                    plt.subplots_adjust()
                    path = gdat.pathalle[typeallemodl] + 'hist_albg_%s_%s.%s' % (gdat.strgtarg, gdat.strgprio, gdat.typefileplot)
                    print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()
                
                    #liststrgfile = ['ContribFuncArr.txt', \
                    #                'EmissionDataArray.txt', \
                    #                #'RetrievalParamSamples.txt', \
                    #                'ContribFuncWav.txt', \
                    #                'EmissionModelArray.txt', \
                    #                'RetrievalPTSamples.txt', \
                    #                'pdependent_abundances/', \
                    #                ]
                    
                    # get the ATMO posterior
                    path = gdat.pathdata + 'ascii_output/RetrievalParamSamples.txt'
                    listsampatmo = np.loadtxt(path)
                    
                    # plot ATMO posterior
                    listlablpara = [['$\kappa_{IR}$', ''], ['$\gamma$', ''], ['$\psi$', ''], ['[M/H]', ''], \
                                                                                                    ['[C/H]', ''], ['[O/H]', '']]
                    tdpy.mcmc.plot_grid(gdat.pathalle[typeallemodl], 'post_atmo', listsampatmo, listlablpara, plotsize=2.5)
   
                    # get the ATMO posterior on irradiation efficiency, psi
                    indxsampatmo = np.random.choice(np.arange(listsampatmo.shape[0]), size=gdat.numbsamp, replace=False)
                    gdat.listpsii = listsampatmo[indxsampatmo, 2]
                    
                    gdat.gmeatmptequi = np.percentile(gdat.dictlist['tmptequi'][:, 0], 50.)
                    gdat.gstdtmptequi = (np.percentile(gdat.dictlist['tmptequi'][:, 0], 84.) - np.percentile(gdat.dictlist['tmptequi'][:, 0], 16.)) / 2.
                    gdat.gmeatmptdayy = np.percentile(gdat.dictlist['tmptdayy'][:, 0], 50.)
                    gdat.gstdtmptdayy = (np.percentile(gdat.dictlist['tmptdayy'][:, 0], 84.) - np.percentile(gdat.dictlist['tmptdayy'][:, 0], 16.)) / 2.
                    gdat.gmeatmptnigh = np.percentile(gdat.dictlist['tmptnigh'][:, 0], 50.)
                    gdat.gstdtmptnigh = (np.percentile(gdat.dictlist['tmptnigh'][:, 0], 84.) - np.percentile(gdat.dictlist['tmptnigh'][:, 0], 16.)) / 2.
                    gdat.gmeapsii = np.percentile(gdat.listpsii, 50.)
                    gdat.gstdpsii = (np.percentile(gdat.listpsii, 84.) - np.percentile(gdat.listpsii, 16.)) / 2.
                
                    histpsii, gdat.binspsii = np.histogram(gdat.listpsii, 1001)
                    gdat.meanpsii = (gdat.binspsii[1:] + gdat.binspsii[:-1]) / 2.
                    
                    gdat.kdegstdvpsii = 0.01
                    figr, axis = plt.subplots(figsize=gdat.figrsizeydob)
                    gdat.kdegpsii = tdpy.util.retr_kdeg(gdat.listpsii, gdat.meanpsii, gdat.kdegstdvpsii)
                    axis.plot(gdat.meanpsii, gdat.kdegpsii)
                    axis.set_xlabel('$\psi$')
                    axis.set_ylabel('$K_\psi$')
                    plt.subplots_adjust()
                    path = gdat.pathalle[typeallemodl] + 'kdeg_psii_%s_%s.%s' % (gdat.strgtarg, gdat.strgprio, gdat.typefileplot)
                    print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()
                
                    # use psi posterior to infer Bond albedo and heat circulation efficiency
                    numbsampwalk = 10000
                    numbsampburnwalk = 0
                    numbsampburnwalkseco = 1000
                    listlablpara = [['$A_b$', ''], ['$E$', ''], [r'$\varepsilon$', '']]
                    listscalpara = ['self', 'self', 'self']
                    listminmpara = np.array([0., 0., 0.])
                    listmaxmpara = np.array([1., 1., 1.])
                    listmeangauspara = None
                    liststdvgauspara = None
                    numbdata = 0
                    strgextn = 'albbemisepsi'
                    listpostheat = tdpy.mcmc.samp(gdat, gdat.pathalle[typeallemodl], numbsampwalk, numbsampburnwalk, \
                                                                                numbsampburnwalkseco, retr_llik_albbemisepsi, \
                                         listlablpara, listscalpara, listminmpara, listmaxmpara, listmeangauspara, liststdvgauspara, numbdata, strgextn=strgextn)

                    # plot emission spectra, secondary eclipse depth, and brightness temperature
                    #listcolr = ['k', 'm', 'purple', 'olive', 'olive', 'r', 'g']
                    listcolr = ['k', 'm', 'purple', 'olive', 'olive', 'r', 'g']
                    for i in range(15):
                        listcolr.append('r')
                    for i in range(28):
                        listcolr.append('g')
                    figr, axis = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
                    ## stellar emission spectrum and TESS throughput
                    axis[0].plot(arrymodl[:, 0], 1e-9 * arrymodl[:, 9], label='Host star', color='grey')
                    axis[0].plot(0., 0., ls='--', label='TESS Throughput', color='grey')
                    axis[0].set_ylabel(r'$\nu F_{\nu}$ [10$^9$ erg/s/cm$^2$]')
                    axis[0].legend(fancybox=True, bbox_to_anchor=[0.7, 0.22, 0.2, 0.2])
                    axistwin = axis[0].twinx()
                    axistwin.plot(gdat.meanwlenband, gdat.thptband, color='grey', ls='--', label='TESS')
                    axistwin.set_ylabel(r'Throughput')
                    
                    ## secondary eclipse depths
                    ### model
                    print('arrydata[0, 0]')
                    print(arrydata[0, 0])
                    print('gdat.amplplantheratmo')
                    print(gdat.amplplantheratmo)
                    print('')
                    objtplotmodllavgd, = axis[1].plot(arrydata[0, 0], 1e6*gdat.amplplantheratmo, color='b', marker='D')
                    axis[1].plot(arrymodl[:, 0], arrymodl[:, 1], label='1D Retrieval (This work)', color='b')
                    axis[1].plot(arrymodl[:, 0], arrymodl[:, 2], label='Blackbody (This work)', alpha=0.3, color='deepskyblue')
                    axis[1].fill_between(arrymodl[:, 0], arrymodl[:, 3], arrymodl[:, 4], alpha=0.3, color='deepskyblue')
                    objtplotvivi, = axis[1].plot(gdat.wlenvivi, gdat.specvivi * 1e6, color='orange', alpha=0.6, lw=2)
                    ### data
                    for k in range(5):
                        axis[1].errorbar(arrydata[k, 0], arrydata[k, 2], xerr=arrydata[k, 1], yerr=arrydata[k, 3], ls='', marker='o', color=listcolr[k])
                    axis[1].errorbar(arrydata[5:22, 0], arrydata[5:22, 2], xerr=arrydata[5:22, 1], yerr=arrydata[5:22, 3], ls='', marker='o', color='r')
                    axis[1].errorbar(arrydata[22:-1, 0], arrydata[22:-1, 2], xerr=arrydata[22:-1, 1], yerr=arrydata[22:-1, 3], ls='', marker='o', color='g')
                    axis[1].set_ylabel(r'Depth [ppm]')
                    axis[1].set_xticklabels([])
                    
                    ## planetary emission spectra
                    ### model
                    objtplotretr, = axis[2].plot(arrymodl[:, 0], 1e-9 * arrymodl[:, 5], label='1D Retrieval (This work)', color='b')
                    objtplotmblc, = axis[2].plot(arrymodl[:, 0], 1e-9 * arrymodl[:, 6], label='Blackbody (This work)', color='deepskyblue', alpha=0.3)
                    objtploteblc = axis[2].fill_between(arrymodl[:, 0], 1e-9 * arrymodl[:, 7], 1e-9 * arrymodl[:, 8], color='deepskyblue', alpha=0.3)
                    axis[2].legend([objtplotretr, objtplotmodllavgd, (objtplotmblc, objtploteblc), objtplotvivi], \
                                               ['1D Retrieval (This work)', '1D Retrieval (This work), Avg', 'Blackbody (This work)', 'GCM (Parmentier+2018)'], \
                                                                                            bbox_to_anchor=[0.8, 1.4, 0.2, 0.2])
                    ### data
                    for k in range(5):
                        axis[2].errorbar(arrydata[k, 0],  1e-9 * arrydata[k, 6], xerr=arrydata[k, 1], yerr=1e-9*arrydata[k, 7], ls='', marker='o', color=listcolr[k])
                    axis[2].errorbar(arrydata[5:22, 0], 1e-9 * arrydata[5:22, 6], xerr=arrydata[5:22, 1], yerr=1e-9*arrydata[5:22, 7], ls='', marker='o', color='r')
                    axis[2].errorbar(arrydata[22:-1, 0], 1e-9 * arrydata[22:-1, 6], xerr=arrydata[22:-1, 1], \
                                                                    yerr=1e-9*arrydata[22:-1, 7], ls='', marker='o', color='g')
                    
                    axis[2].set_ylabel(r'$\nu F_{\nu}$ [10$^9$ erg/s/cm$^2$]')
                    axis[2].set_xticklabels([])
                    
                    ## brightness temperature
                    ### data
                    for k in range(5):
                        if k == 0:
                            labl = 'TESS (This work)'
                        if k == 1:
                            labl = 'Z$^\prime$ (Delrez+2016)'
                        if k == 2:
                            labl = '$K_s$ (Kovacs\&Kovacs2019)'
                        if k == 3:
                            labl = 'IRAC $\mu$m (Garhart+2019)'
                        #if k == 4:
                        #    labl = 'IRAC 4.5 $\mu$m (Garhart+2019)'
                        axis[3].errorbar(arrydata[k, 0], arrydata[k, 4], xerr=arrydata[k, 1], yerr=arrydata[k, 5], label=labl, ls='', marker='o', color=listcolr[k])
                    axis[3].errorbar(arrydata[5:22, 0], arrydata[5:22, 4], xerr=arrydata[5:22, 1], \
                                                         yerr=arrydata[5:22, 5], label='HST G102 (Evans+2019)', ls='', marker='o', color='r')
                    axis[3].errorbar(arrydata[22:-1, 0], arrydata[22:-1, 4], xerr=arrydata[22:-1, 1], \
                                                        yerr=arrydata[22:-1, 5], label='HST G141 (Evans+2017)', ls='', marker='o', color='g')
                    #axis[3].errorbar(arrydata[:, 0], np.median(tmpt, 0), xerr=arrydata[:, 1], yerr=np.std(tmpt, 0), label='My calc', ls='', marker='o', color='c')
                    axis[3].set_ylabel(r'$T_B$ [K]')
                    axis[3].set_xlabel(r'$\lambda$ [$\mu$m]')
                    axis[3].legend(fancybox=True, bbox_to_anchor=[0.8, 3.8, 0.2, 0.2], ncol=2)
                    
                    axis[1].set_ylim([20, None])
                    axis[1].set_yscale('log')
                    for i in range(4):
                        axis[i].set_xscale('log')
                    axis[3].set_xlim([0.5, 5])
                    axis[3].xaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())
                    axis[3].xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
                    plt.subplots_adjust(hspace=0., wspace=0.)
                    path = gdat.pathalle[typeallemodl] + 'spec_%s_%s.%s' % (gdat.strgtarg, gdat.strgprio, gdat.typefileplot)
                    print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()
                    
                    # get contribution function
                    path = gdat.pathdata + 'ascii_output/ContribFuncArr.txt'
                    print('Reading from %s...' % path)
                    ctrb = np.loadtxt(path)
                    presctrb = ctrb[0, :]
                    # interpolate the throughput
                    gdat.thptbandctrb = scipy.interpolate.interp1d(gdat.meanwlenband, gdat.thptband, fill_value=0, bounds_error=False)(wlenctrb)
                    numbwlenctrb = wlenctrb.size
                    indxwlenctrb = np.arange(numbwlenctrb)
                    numbpresctrb = presctrb.size
                    indxpresctrb = np.arange(numbpresctrb)

                    # plot pressure-temperature, contribution function, abundances
                    ## get ATMO posterior
                    path = gdat.pathdata + 'ascii_output/RetrievalPTSamples.txt'
                    dataptem = np.loadtxt(path)
                    liststrgcomp = ['CH4.txt', 'CO.txt', 'FeH.txt', 'H+.txt', 'H.txt', 'H2.txt', 'H2O.txt', 'H_.txt', 'He.txt', 'K+.txt', \
                                                                        'K.txt', 'NH3.txt', 'Na+.txt', 'Na.txt', 'TiO.txt', 'VO.txt', 'e_.txt']
                    listlablcomp = ['CH$_4$', 'CO', 'FeH', 'H$^+$', 'H', 'H$_2$', 'H$_2$O', 'H$^-$', 'He', 'K$^+$', \
                                                                        'K', 'NH$_3$', 'Na$^+$', 'Na', 'TiO', 'VO', 'e$^-$']
                    listdatacomp = []
                    for strg in liststrgcomp:
                        path = gdat.pathdata + 'ascii_output/pdependent_abundances/' + strg
                        listdatacomp.append(np.loadtxt(path))
                    ## plot
                    figr, axis = plt.subplots(nrows=1, ncols=2, sharey=True, gridspec_kw={'width_ratios': [1, 2]}, figsize=gdat.figrsizeydob)
                    ### pressure temperature
                    numbsamp = dataptem.shape[0] - 1
                    indxsamp = np.arange(numbsamp)
                    for i in indxsamp[::100]:
                        axis[0].plot(dataptem[i, :], dataptem[0, :], color='b', alpha=0.1)
                    axis[0].plot(np.percentile(dataptem, 10, axis=0), dataptem[0, :], color='g')
                    axis[0].plot(np.percentile(dataptem, 50, axis=0), dataptem[0, :], color='r')
                    axis[0].plot(np.percentile(dataptem, 90, axis=0), dataptem[0, :], color='g')
                    axis[0].set_xlim([1500, 3700])
                    axis[0].set_xlabel('$T$ [K]')
                    axis[0].set_yscale('log')
                    axis[0].set_ylabel('$P$ [bar]')
                    axis[0].invert_yaxis()
                    axis[0].set_ylim([10., 1e-5])
                    ### contribution function
                    axistwin = axis[0].twiny()
                    ctrbtess = np.empty(numbpresctrb)
                    for k in indxpresctrb:
                        ctrbtess[k] = np.sum(ctrb[1:, k] * gdat.thptbandctrb)
                    ctrbtess *= 1e-12 / np.amax(ctrbtess)
                    axistwin.fill(ctrbtess, presctrb, alpha=0.5, color='grey')
                    axistwin.set_xticklabels([])
                    ## abundances
                    numbcomp = len(listdatacomp)
                    indxcomp = np.arange(numbcomp)
                    listobjtcolr = sns.color_palette('hls', numbcomp)
                    axis[1].set_prop_cycle('color', listobjtcolr)
                    listcolr = []
                    for k in indxcomp:
                        objt, = axis[1].plot(listdatacomp[k][:, 1], listdatacomp[k][:, 0])
                        listcolr.append(objt.get_color())

                    axis[1].xaxis.tick_top()
                    
                    arry = np.logspace(-16., 0., 21) # x 0.8
                    for k in range(21):
                        axis[1].axvline(arry[k], ls='--', alpha=0.1, color='k')
                    arry = np.logspace(-5., 1., 11) # y 0.6
                    for k in range(11):
                        axis[1].axhline(arry[k], ls='--', alpha=0.1, color='k')
                    listobjtcolr = sns.color_palette('hls', numbcomp)
                    axis[1].set_prop_cycle('color', listobjtcolr)
                    for k in indxcomp:
                        if k == 0: # CH4
                            xpos, ypos = 10**-12.8, 10**-2.3
                        elif k == 1: # CO
                            xpos, ypos = 10**-2.8, 10**-3.5
                        elif k == 2: # FeH
                            xpos, ypos = 10**-10.8, 10**-3.5
                        elif k == 3: # H+
                            xpos, ypos = 10**-12.8, 10**-4.1
                        elif k == 4: # H
                            xpos, ypos = 10**-1.6, 10**-2
                        elif k == 5: # H2
                            xpos, ypos = 10**-1.6, 10**-2.6
                        elif k == 6: # H20
                            xpos, ypos = 10**-8.8, 10**-4.1
                        elif k == 7: # H_
                            xpos, ypos = 10**-10., 10**0.4
                        elif k == 8: # He
                            xpos, ypos = 10**-1.6, 10**-4.1
                        elif k == 9: # K+
                            xpos, ypos = 10**-4.4, 10**-4.8
                        elif k == 10: # K
                            xpos, ypos = 10**-8.4, 10**-4.8
                        elif k == 11: # Nh3
                            xpos, ypos = 10**-13.6, 10**-4.1
                        elif k == 12: # Na+
                            xpos, ypos = 10**-4.4, 10**-3.8
                        elif k == 13: # Na
                            xpos, ypos = 10**-6, 10**-3.8
                        elif k == 14: # TiO
                            xpos, ypos = 10**-7.6, 10**-2
                        elif k == 15: # VO
                            xpos, ypos = 10**-6, 10**-2
                        elif k == 16: # e-
                            xpos, ypos = 10**-5.6, 10**-0.8
                        else:
                            xpos = 10**(np.random.rand() * 16. - 16.)
                            ypos = 10**(np.random.rand() * 6. - 5.)
                        axis[1].text(xpos, ypos, '%s' % listlablcomp[k], color=listcolr[k], size=10, va='center', ha='center')
                    axis[1].set_xscale('log')
                    axis[1].set_xlabel('Volume Mixing Ratio')
                    axis[1].set_yscale('log')
                    axis[1].set_xlim([1e-16, 1])
                    plt.subplots_adjust(hspace=0., wspace=0., bottom=0.15)
                    path = gdat.pathalle[typeallemodl] + 'ptem_%s_%s.%s' % (gdat.strgtarg, gdat.strgprio, gdat.typefileplot)
                    print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()
  

def retr_rflxtranmodl(time, radiplan, radistar, rsma, epoc, peri, cosi):
    
    minmtime = np.amin(time)
    maxmtime = np.amax(time)
    
    
    ecce = 0
    sinw = 0
    factrsrj, factmsmj, factrjre, factmjme, factaurj = tesstarg.util.retr_factconv()
    smax = (radistar * factrsrj + radiplan / factrjre) / rsma / factaurj
    rs2a = radistar / smax
    imfa = retr_imfa(cosi, rs2a, ecce, sinw)
    sini = np.sqrt(1. - cosi**2)
    rrat = radiplan / factrsrj / factrjre / radistar
    durafull = retr_durafull(peri, rs2a, sini, rrat, imfa)
    duratotl = retr_durafull(peri, rs2a, sini, rrat, imfa)
    dept = rrat**2
    duraineg = (duratotl - durafull) / 2.
    rflxtranmodl = np.ones_like(time)
    
    numbplan = radiplan.size
    indxplan = np.arange(numbplan)
    for j in indxplan:
        minmindxtran = np.floor(epoc[j] - minmtime) / peri[j]
        maxmindxtran = np.ceil(epoc[j] - maxmtime) / peri[j]
        for n in np.arange(minmindxtran, maxmindxtran + 1):
            timetran = epoc - peri * n
            indxtimetotl = np.where(abs(timetran - time) < duratotl)[0]
            indxtimeinre = indxtimetotl[np.where((timetran - time[indxtimetotl] > durafull / 2.) & (timetran - time[indxtimetotl] < duratotl / 2.))[0]]
            indxtimeegre = indxtimetotl[np.where((time[indxtimetotl] - timetran > durafull / 2.) & (time[indxtimetotl] - timetran < duratotl / 2.))[0]]
            indxtimetemp = np.concatenate((indxtimeinre, indxtimeegre))
            indxtimefull = np.setdiff1d(indxtimetotl, indxtimetemp)
            rflxtranmodl[indxtimeinre] -= dept[j] * ((time[indxtimeinre] - timetran - duratotl / 2.) / duraineg)
            rflxtranmodl[indxtimeegre] += dept[j] * ((timetran + duratotl / 2. - time[indxtimeinre]) / duraineg)
            rflxtranmodl[indxtimefull] -= dept[j]
    
    return rflxtranmodl


def plot_orbt( \
              # path to write the plot
              path, \
              # radius of the planets [R_E]
              radiplan, \
              # sum of radius of planet and star divided by the semi-major axis
              rsma, \
              # epoc of the planets [BJD]
              epoc, \
              # orbital periods of the planets [days]
              peri, \
              # cosine of the inclination
              cosi, \
              # type of visualization: 
              ## 'realblac': dark background, black planet
              ## 'realblaclcur': dark backgound, luminous planet, 
              ## 'realillu': dark background, illuminated planet, 
              ## 'cart': cartoon, 'realdark' 
              ## 'cartmerc': cartoon, 'realdark' 
              typevisu, \
              
              # radius of the star [R_S]
              radistar=1., \
              # mass of the star [M_S]
              massstar=1., \
              # Boolean flag to produce an animation
              boolanim=False, \

              # angle of view with respect to the orbital plane [deg]
              anglpers=5., \

              # size of the figure
              sizefigr=(8, 8), \
              listcolrplan=None, \
              liststrgplan=None, \
              boolsingside=True, \
              typefileplot='pdf', \
             ):

    from allesfitter.v2.classes import allesclass2
    from allesfitter.v2.translator import translate
    
    factrsrj, factmsmj, factrjre, factmjme, factaurj = tesstarg.util.retr_factconv()
    
    mpl.use('Agg')

    numbplan = len(radiplan)
    
    if isinstance(radiplan, list):
        radiplan = np.array(radiplan)

    if isinstance(rsma, list):
        rsma = np.array(rsma)

    if isinstance(epoc, list):
        epoc = np.array(epoc)

    if isinstance(peri, list):
        peri = np.array(peri)

    if isinstance(cosi, list):
        cosi = np.array(cosi)

    if listcolrplan is None:
        listcolrplan = retr_listcolrplan(numbplan)

    if liststrgplan is None:
        liststrgplan = retr_liststrgplan(numbplan)
    
    # semi-major axes of the planets [AU]
    smaxasun = (radiplan / factrjre + radistar * factrsrj) / rsma / factaurj
    indxplan = np.arange(numbplan)
    
    # perspective factor
    factpers = np.sin(anglpers * np.pi / 180.)

    ## scale factor for the star
    factstar = 5.
    
    ## scale factor for the planets
    factplan = 5.
    
    if typevisu == 'cartmerc':
        # Mercury
        smaxasunmerc = 0.387 # [AU]
        radiplanmerc = 0.0349 # [R_J]
    
    # scaled radius of the star
    radistarscal = radistar * factrsrj / factaurj * factstar
    
    if boolanim:
        numbiter = 30
    else:
        numbiter = 1
    indxiter = np.arange(numbiter)
    
    xposmaxm = smaxasun
    yposmaxm = factpers * xposmaxm
    numbtimequad = 10
    
    if typevisu == 'realblaclcur':
        numbtimespan = 100

    time = np.arange(0., 30., 2. / 60. / 24.)
    numbtime = time.size
    indxtime = np.arange(numbtime)
   
    # get transit model based on TESS ephemerides
    rrat = radiplan / radistar
    alles = allesclass2()
    
    rflxtranmodl = retr_rflxtranmodl(time, radiplan, radistar, rsma, epoc, peri, cosi)
    print('rflxtranmodl')
    summgene(rflxtranmodl)
    #alles.settings = {'inst_phot':['telescope'], 'host_ld_law_telescope':'quad'}
    #alles.settings['companions_phot'] = []
    #alles.params['host_ldc_q1_telescope'] = 0.5
    #alles.params['host_ldc_q2_telescope'] = 0.2
    #for j, strgplan in enumerate(liststrgplan):
    #    alles.settings['companions_phot'].append(strgplan)
    #    alles.params['%s_rr' % strgplan] = rrat[j]
    #    alles.params['%s_rsuma' % strgplan] = rsma[j]
    #    alles.params['%s_epoch' % strgplan] = epoc[j]
    #    alles.params['%s_period' % strgplan] = peri[j]
    #    alles.params['%s_cosi' % strgplan] = cosi[j]
    #alles.params_host = {'R_host':radistar, 'M_host':massstar}
    #alles.fill()
    #model_flux = alles.generate_model(time, inst='telescope', key='flux')
    lcur = rflxtranmodl + np.random.randn(numbtime) * 1e-4
    ylimrflx = [np.amin(lcur), np.amax(lcur)]
    
    phas = 2. * np.pi * time[:, None] / peri[None, :]
    yposelli = yposmaxm[None, :] * np.sin(phas)
    xposelli = xposmaxm[None, :] * np.cos(phas)
    
    # time indices for iterations
    indxtimeiter = np.linspace(0., numbtime - numbtime / numbiter, numbiter).astype(int)
    
    print('xposelli, yposelli')
    for k in range(len(xposelli)):
        print(xposelli[k, 0], yposelli[k, 0])
    
    if typevisu.startswith('cart'):
        colrstar = 'k'
        colrface = 'w'
        plt.style.use('default')
    else:
        colrface = 'k'
        colrstar = 'w'
        plt.style.use('dark_background')
    
    if boolanim:
        cmnd = 'convert -delay 20'
        listpathtemp = []
    for k in indxiter:
        
        if typevisu == 'realblaclcur':
            numbrows = 2
        else:
            numbrows = 1
        figr, axis = plt.subplots(numbrows, 1, figsize=sizefigr)
        if numbrows == 1:
            axis = [axis]

        ### lower half of the star
        #w1 = mpl.patches.Wedge((0, 0), radistarscal, 180, 360, fc=colrstar, zorder=1, edgecolor=colrstar)
        #axis[0].add_artist(w1)
        #
        #for jj, j in enumerate(indxplan):
        #    print('xposelli')
        #    summgene(xposelli)
        #    print('indxtimeiter[k]')
        #    print(indxtimeiter[k])
        #    xposellishft = np.roll(xposelli[:, j], -indxtimeiter[k])[-numbtimequad:][::-1]
        #    yposellishft = np.roll(yposelli[:, j], -indxtimeiter[k])[-numbtimequad:][::-1]
        #
        #    # trailing lines
        #    if typevisu.startswith('cart'):
        #        objt = retr_objtlinefade(xposellishft, yposellishft, colr=listcolrplan[j], initalph=1., alphfinl=0.)
        #        axis[0].add_collection(objt)
        #    
        #    # add planets
        #    if typevisu.startswith('cart'):
        #        colrplan = listcolrplan[j]
        #        # add planet labels
        #        axis[0].text(.15 + 0.02 * jj, -0.03, liststrgplan[j], color=listcolrplan[j])
        #
        #    if typevisu.startswith('real'):
        #        if typevisu == 'realillu':
        #            colrplan = 'k'
        #        else:
        #            colrplan = 'black'
        #    radi = radiplan[j] / factrjre / factaurj * factplan
        #    w1 = mpl.patches.Circle((xposelli[indxtimeiter[k], j], yposelli[indxtimeiter[k], j], 0), radius=radi, color=colrplan, zorder=3)
        #    axis[0].add_artist(w1)
        #    
        ### upper half of the star
        #w1 = mpl.patches.Wedge((0, 0), radistarscal, 0, 180, fc=colrstar, zorder=4, edgecolor=colrstar)
        #axis[0].add_artist(w1)
        
        if typevisu == 'cartmerc':
            ## add Mercury
            axis[0].text(.387, 0.01, 'Mercury', color='grey', ha='right')
            radi = radiplanmerc / factaurj * factplan
            w1 = mpl.patches.Circle((smaxasunmerc, 0), radius=radi, color='grey')
            axis[0].add_artist(w1)
        
        # temperature axis
        #axistwin = axis.twiny()
        ##axistwin.set_xlim(axis.get_xlim())
        #xpostemp = axistwin.get_xticks()
        ##axistwin.set_xticks(xpostemp[1:])
        #axistwin.set_xticklabels(['%f' % tmpt for tmpt in listtmpt])
        
        # temperature contours
        #for tmpt in [500., 700,]:
        #    smaj = tmpt
        #    axis[0].axvline(smaj, ls='--')
        
        axis[0].get_yaxis().set_visible(False)
        axis[0].set_aspect('equal')
        
        if typevisu == 'cartmerc':
            maxmxaxi = max(1.2 * np.amax(smaxasun), 0.4)
        else:
            maxmxaxi = 1.2 * np.amax(smaxasun)
        
        if boolsingside:
            minmxaxi = 0.
        else:
            minmxaxi = -maxmxaxi

        axis[0].set_xlim([minmxaxi, maxmxaxi])
        axis[0].set_ylim([-0.05, 0.05])
        axis[0].set_xlabel('Distance from the star [AU]')
        
        if typevisu == 'realblaclcur':
        #if False and typevisu == 'realblaclcur':
            minmindxtime = max(0, indxtimeiter[k]-numbtimespan)
            print('time[minmindxtime:indxtimeiter[k]]')
            summgene(time[minmindxtime:indxtimeiter[k]])
            print('lcur[minmindxtime:indxtimeiter[k]]')
            summgene(lcur[minmindxtime:indxtimeiter[k]])
            print('minmindxtime')
            print(minmindxtime)
            print('indxtimeiter[k]')
            print(indxtimeiter[k])
            #axis[1].scatter(time[minmindxtime:indxtimeiter[k]], lcur[minmindxtime:indxtimeiter[k]])
            #axis[1].set_ylim(ylimrflx)
            #axis[1].plot(time[minmindxtime:indxtimeiter[k]], lcur[minmindxtime:indxtimeiter[k]], ls='', marker='o', color='cyan', rasterized=True)

        #plt.subplots_adjust()
        #axis.legend()
        
        if boolanim:
            pathtemp = '%s_%s_%04d.%s' % (path, typevisu, k, typefileplot)
        else:
            pathtemp = '%s_%s.%s' % (path, typevisu, typefileplot)
        print('Writing to %s...' % pathtemp)
        plt.savefig(pathtemp)
        plt.close()
        
        if boolanim:
            listpathtemp.append(pathtemp)
            cmnd += ' %s' % pathtemp 
    cmnd += ' %s_%s.gif' % (path, typevisu)
    os.system(cmnd)
    for pathtemp in listpathtemp:
        cmnd = 'rm %s' % pathtemp
        os.system(cmnd)


def plot_prop(gdat, strgpdfn):
    
    pathimagprop = getattr(gdat, 'pathimagprop' + strgpdfn)
    pathimagheke = getattr(gdat, 'pathimagheke' + strgpdfn)
    pathimagmagt = getattr(gdat, 'pathimagmagt' + strgpdfn)
    
    if gdat.boolobjt:
        
        path = pathimagprop + 'orbt_%s_%s' % (dat.strgtarg, strgpdfn)
        plot_orbt( \
                  path, \
                  gdat.dicterrr['radiplan'][0, :], \
                  gdat.dicterrr['peri'][0, :], \
                  gdat.dicterrr['smaxasun'][0, :], \
                  radistar=gdat.radistar, \
                  sizefigr=gdat.figrsizeydob, \
                  typefileplot=gdat.typefileplot, \
                 )

    ## occurence rate as a function of planet radius with highlighted radii of the system's planets
    ### get the CKS occurence rate as a function of planet radius
    path = gdat.pathbase + 'data/Fulton+2017/Means.csv'
    data = np.loadtxt(path, delimiter=',')
    timeoccu = data[:, 0]
    occumean = data[:, 1]
    path = gdat.pathbase + 'data/Fulton+2017/Lower.csv'
    occulowr = np.loadtxt(path, delimiter=',')
    occulowr = occulowr[:, 1]
    path = gdat.pathbase + 'data/Fulton+2017/Upper.csv'
    occuuppr = np.loadtxt(path, delimiter=',')
    occuuppr = occuuppr[:, 1]
    occuyerr = np.empty((2, occumean.size))
    occuyerr[0, :] = occuuppr - occumean
    occuyerr[1, :] = occumean - occulowr
    
    figr, axis = plt.subplots(figsize=gdat.figrsize)
    
    if gdat.boolobjt:
        # this system
        for jj, j in enumerate(gdat.indxplan):
            xposlowr = gdat.factrjre * gdat.dictpost['radiplan'][0, j]
            xposuppr = gdat.factrjre * gdat.dictpost['radiplan'][2, j]
            axis.axvspan(xposlowr, xposuppr, alpha=0.5, color=gdat.listcolrplan[j])
            axis.axvline(gdat.factrjre * gdat.dicterrr['radiplan'][0, j], color=gdat.listcolrplan[j], ls='--', label=gdat.liststrgplan[j])
            axis.text(0.7, 0.9 - jj * 0.07, r'\textbf{%s}' % gdat.liststrgplan[j], color=gdat.listcolrplan[j], \
                                                                                        va='center', ha='center', transform=axis.transAxes)
    xerr = (timeoccu[1:] - timeoccu[:-1]) / 2.
    xerr = np.concatenate([xerr[0, None], xerr])
    axis.errorbar(timeoccu, occumean, yerr=occuyerr, xerr=xerr, color='black', ls='', marker='o', lw=1, zorder=10)
    axis.set_xlabel('Radius [$R_E$]')
    axis.set_ylabel('Occurrence rate of planets per star')
    plt.subplots_adjust(bottom=0.2)
    plt.subplots_adjust(left=0.2)
    path = pathimagprop + 'occuradi_%s_%s.%s' % (gdat.strgtarg, strgpdfn, gdat.typefileplot)
    print('Writing to %s...' % path)
    plt.savefig(path)
    plt.close()
   
    for strgpopl in gdat.liststrgpopl:
        
        if strgpopl == 'exarcomp':
            dictpopl = gdat.dictexarcomp
        else:
            dictpopl = gdat.dictexof
        
        print('strgpopl')
        print(strgpopl)

        ### TSM and ESM
        numbsamppopl = 100
        listradiplan = dictpopl['radiplan'][None, :] + np.random.randn(numbsamppopl)[:, None] * dictpopl['stdvradiplan'][None, :]
        listtmptplan = dictpopl['tmptplan'][None, :] + np.random.randn(numbsamppopl)[:, None] * dictpopl['stdvtmptplan'][None, :]
        listmassplan = dictpopl['massplan'][None, :] + np.random.randn(numbsamppopl)[:, None] * dictpopl['stdvmassplan'][None, :]
        listradistar = dictpopl['radistar'][None, :] + np.random.randn(numbsamppopl)[:, None] * dictpopl['stdvradistar'][None, :]
        listkmagstar = dictpopl['kmagstar'][None, :] + np.random.randn(numbsamppopl)[:, None] * 0.
        listjmagstar = dictpopl['jmagstar'][None, :] + np.random.randn(numbsamppopl)[:, None] * 0.
        listtmptstar = dictpopl['tmptstar'][None, :] + np.random.randn(numbsamppopl)[:, None] * dictpopl['stdvtmptstar'][None, :]
        
        dictlistplan = dict()
        for strgfeat in ['radiplan', 'massplan', 'tmptplan']:
            dictlistplan[strgfeat] = np.zeros((numbsamppopl, dictpopl['massplan'].size)) + np.nan
            for k in range(dictpopl[strgfeat].size):
                meanvarb = dictpopl[strgfeat][k]
                if not np.isfinite(meanvarb):
                    continue
                if np.isfinite(dictpopl['stdv' + strgfeat][k]):
                    stdvvarb = dictpopl['stdv' + strgfeat][k]
                else:
                    stdvvarb = 0.
                
                dictlistplan[strgfeat][:, k] = scipy.stats.truncnorm.rvs(-meanvarb/stdvvarb, np.inf, size=numbsamppopl) * stdvvarb + meanvarb
                if (dictlistplan[strgfeat][:, k] < 0).any():
                    raise Exception('')

        #### TSM
        listtsmm = tesstarg.util.retr_tsmm(dictlistplan['radiplan'], dictlistplan['tmptplan'], dictlistplan['massplan'], listradistar, listjmagstar)
        dictpopl['stdvtsmm'] = np.std(listtsmm, 0)
        dictpopl['tsmm'] = np.median(listtsmm, 0)
        #### ESM
        listesmm = tesstarg.util.retr_esmm(dictlistplan['tmptplan'], listtmptstar, dictlistplan['radiplan'], listradistar, listkmagstar)
        dictpopl['stdvesmm'] = np.std(listesmm, 0)
        dictpopl['esmm'] = np.median(listesmm, 0)
        
        ## augment
        dictpopl['logttsmm'] = np.log(dictpopl['tsmm'])
        dictpopl['logtesmm'] = np.log(dictpopl['esmm'])
        ### radii
        dictpopl['radistarsunn'] = dictpopl['radistar'] / gdat.factrsrj # R_S
        dictpopl['radiplaneart'] = dictpopl['radiplan'] * gdat.factrjre # R_E
        dictpopl['massplaneart'] = dictpopl['massplan'] * gdat.factmjme # M_E
        dictpopl['stdvradiplaneart'] = dictpopl['stdvradiplan'] * gdat.factrjre # R_E
        dictpopl['stdvmassplaneart'] = dictpopl['stdvmassplan'] * gdat.factmjme # M_E

        numbtargpopl = dictpopl['radiplan'].size
        indxtargpopl = np.arange(numbtargpopl)
        
        indxpoplcuttradi = dict()
        indxpoplcuttradi['allr'] = indxtargpopl
        if (gdat.dicterrr['radiplaneart'][0, :] < 4).all():
            indxpoplcuttradi['rall'] = np.where((dictpopl['radiplaneart'] > 0))[0]
            indxpoplcuttradi['r004'] = np.where((dictpopl['radiplaneart'] < 4))[0]
            indxpoplcuttradi['r154'] = np.where((dictpopl['radiplaneart'] > 1.5) & (dictpopl['radiplaneart'] < 4))[0]
            indxpoplcuttradi['r204'] = np.where((dictpopl['radiplaneart'] > 2) & (dictpopl['radiplaneart'] < 4))[0]
        #if ((gdat.dicterrr['radiplaneart'][0, :] < 4) & (gdat.dicterrr['radiplaneart'][0, :] > 2)).all():
        #    indxpoplcuttradi['rb24'] = np.where((dictpopl['radiplaneart'] < 4) & (dictpopl['radiplaneart'] > 2.))[0]
        
        indxpoplcuttmass = dict()
        indxpoplcuttmass['allm'] = indxtargpopl
        # 5 sigma mass measurement
        stnomass = dictpopl['massplan'] / dictpopl['stdvmassplan']
        indxpoplcuttmass['gmtr'] = np.where(np.isfinite(stnomass) & (stnomass > 5) & (dictpopl['booltran']))[0]
        indxpoplcuttmult = dict()
        indxpoplcuttmult['alln'] = indxtargpopl
        indxpoplcuttmult['mult'] = np.where(dictpopl['numbplantranstar'] > 3)[0]
    
        liststrgcuttmult = indxpoplcuttmult.keys()
        liststrgcuttradi = indxpoplcuttradi.keys()
        liststrgcuttmass = indxpoplcuttmass.keys()
        
        # period ratios
        ## all 
        gdat.listratiperi = []
        gdat.intgreso = []
        liststrgstarcomp = []
        for m in indxtargpopl:
            strgstar = dictpopl['namestar'][m]
            if not strgstar in liststrgstarcomp:
                indxexarstar = np.where(dictpopl['namestar'] == strgstar)[0]
                if indxexarstar[0] != m:
                    raise Exception('')
                
                listperi = dictpopl['peri'][None, indxexarstar]
                if not np.isfinite(listperi).all() or np.where(listperi == 0)[0].size > 0:
                    liststrgstarcomp.append(strgstar)
                    continue
                intgreso, ratiperi = retr_reso(listperi)
                
                numbplan = indxexarstar.size
                
                gdat.listratiperi.append(ratiperi[0, :, :][np.triu_indices(numbplan, k=1)])
                gdat.intgreso.append(intgreso)
                
                liststrgstarcomp.append(strgstar)
        
        gdat.listratiperi = np.concatenate(gdat.listratiperi)
        if gdat.boolplotprop:
            figr, axis = plt.subplots(figsize=gdat.figrsize)
            bins = np.linspace(1., 10., 400)
            axis.hist(gdat.listratiperi, bins=bins, rwidth=1)
            if gdat.boolobjt and gdat.numbplan > 1:
                ## this system
                for j in gdat.indxplan:
                    for jj in gdat.indxplan:
                        if gdat.dicterrr['peri'][0, j] > gdat.dicterrr['peri'][0, jj]:
                            ratiperi = gdat.dicterrr['peri'][0, j] / gdat.dicterrr['peri'][0, jj]
                            axis.axvline(ratiperi, color=gdat.listcolrplan[jj])
                            axis.axvline(ratiperi, color=gdat.listcolrplan[j], ls='--')
            
            ylim = axis.get_ylim()
            ydatlabl = 0.9 * ylim[1] + ylim[0]
            ## resonances
            for perifrst, periseco in [[2., 1.], [3., 2.], [4., 3.], [5., 4.], [5., 3.], [5., 2.]]:
                rati = perifrst / periseco
                axis.text(rati + 0.05, ydatlabl, '%d:%d' % (perifrst, periseco), size=8, color='grey', va='center', ha='center')
                axis.axvline(perifrst / periseco, color='grey', ls='--', alpha=0.5)
            #axis.set_xscale('log')
            axis.set_xlim([0.9, 2.7])
            axis.set_ylabel('N')
            axis.set_xlabel('Period ratio')
            plt.subplots_adjust(bottom=0.2)
            plt.subplots_adjust(left=0.2)
            path = pathimagprop + 'histratiperi_%s_%s_%s.%s' % (gdat.strgtarg, strgpdfn, strgpopl, gdat.typefileplot)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
        
        if gdat.boolobjt:
            # metastable helium absorption
            path = gdat.pathbase + '/data/wasp107b_transmission_spectrum.dat'
            print('Reading from %s...' % path)
            arry = np.loadtxt(path, delimiter=',', skiprows=1)
            wlenwasp0107 = arry[:, 0]
            deptwasp0107 = arry[:, 1]
            deptstdvwasp0107 = arry[:, 2]
            
            stdvnirs = 0.24e-2
            for a in range(2):
                duratranplanwasp0107 = 2.74 / 24.
                jmagstarwasp0107 = 9.4
                if a == 1:
                    radiplan = gdat.dicterrr['radiplan'][0, :]
                    massplan = gdat.dicterrr['massplanused'][0, :]
                    tmptplan = gdat.dicterrr['tmptplan'][0, :]
                    duratranplan = gdat.dicterrr['duratotl'][0, :]
                    radistar = gdat.radistar
                    jmagstar = gdat.jmagstar
                else:
                    print('WASP-107')
                    radiplan = 0.924
                    massplan = 0.119
                    tmptplan = 736
                    radistar = 0.66 * gdat.factrsrj
                    jmagstar = jmagstarwasp0107
                    duratranplan = duratranplanwasp0107
                scalheig = tesstarg.util.retr_scalheig(tmptplan, massplan, radiplan)
            
                print('radiplan')
                print(radiplan)
                print('massplan')
                print(massplan)
                print('duratranplan')
                print(duratranplan)
                print('tmptplan')
                print(tmptplan)
                print('jmagstar')
                print(jmagstar)
                print('jmagstarwasp0107')
                print(jmagstarwasp0107)
                print('scalheig [R_J]')
                print(scalheig)
                print('scalheig [km]')
                print(scalheig * 71398)
                deptscal = 2. * radiplan * scalheig / radistar**2
                print('deptscal')
                print(deptscal)
                dept = 80. * deptscal
                print('dept')
                print(dept)
                print('duratranplanwasp0107')
                print(duratranplanwasp0107)
                print('duratranplan')
                print(duratranplan)
                factstdv = np.sqrt(10**((-jmagstarwasp0107 + jmagstar) / 2.5) * duratranplanwasp0107 / duratranplan)
                print('factstdv')
                print(factstdv)
                stdvnirsthis = factstdv * stdvnirs
                print('stdvnirsthis')
                print(stdvnirsthis)
                for b in np.arange(1, 6):
                    print('With %d transits:' % b)
                    stdvnirsscal = stdvnirsthis / np.sqrt(float(b))
                    print('stdvnirsscal')
                    print(stdvnirsscal)
                    sigm = dept / stdvnirsscal
                    print('sigm')
                    print(sigm)
                print('')
            print('James WASP107b scale height: 855 km')
            print('James WASP107b scale height: %g [R_J]' % (855. / 71398))
            print('James WASP107b depth per scale height: 5e-4')
            fact = deptscal / 500e-6
            print('ampltide rattio fact: deptthis / 500e-6')
            print(fact)
            # 2 A * Rp * H / Rs**2
            figr, axis = plt.subplots(figsize=gdat.figrsize)
            #axis.errorbar(wlenwasp0107, deptwasp0107, yerr=deptstdvwasp0107, ls='', ms=1, lw=1, marker='o', color='k', alpha=1)
            axis.errorbar(wlenwasp0107-10833, deptwasp0107*fact[0], yerr=deptstdvwasp0107*factstdv[0], ls='', ms=1, lw=1, marker='o', color='k', alpha=1)
            axis.set_xlabel(r'Wavelength - 10,833 [$\AA$]')
            axis.set_ylabel('Depth [\%]')
            plt.subplots_adjust(bottom=0.2, left=0.2)
            path = pathimagheke + 'dept_%s_%s.%s' % (gdat.strgtarg, strgpdfn, gdat.typefileplot)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()

        numbtext = min(30, numbtargpopl)
        liststrgtext = ['notx', 'text']
        
        liststrgvarb = ['peri', 'inso', 'vesc', 'massplan', 'massplaneart', \
                                        'tmptstar', 'declstar', 'radistarsunn', \
                                                          'radiplan', 'radiplaneart', 'tmptplan', \
                                                           'logttsmm', 'logtesmm', \
                                                           'tsmm', 'esmm', \
                                                           'vsiistar', 'projoblq', 'jmagstar']
        listlablvarb = [['P', 'days'], ['F', '$F_E$'], ['$v_{esc}$', 'kms$^{-1}$'], ['$M_p$', '$M_J$'], ['$M_p$', '$M_E$'], \
                                   ['$T_{eff}$', 'K'], ['Dec', 'deg'], ['$R_s$', '$R_S$'], \
                                   ['$R_p$', '$R_J$'], ['$R_p$', '$R_E$'], ['$T_p$', 'K'], \
                                   ['$\ln$ TSM', ''], ['$\ln$ ESM', ''], \
                                   ['TSM', ''], ['ESM', ''], \
                                   ['$v$sin$i$', 'kms$^{-1}$'], ['$\lambda$', 'deg'], ['J', 'mag']] 
        
        numbvarb = len(liststrgvarb)
        indxvarb = np.arange(numbvarb)
        listlablvarbtotl = []
        for k in indxvarb:
            if listlablvarb[k][1] == '':
                listlablvarbtotl.append('%s' % (listlablvarb[k][0]))
            else:
                listlablvarbtotl.append('%s [%s]' % (listlablvarb[k][0], listlablvarb[k][1]))
       
        # optical magnitude vs number of planets
        if gdat.boolplotprop:
            os.system('mkdir -p %s' % pathimagmagt)
            #for b in range(4):
            for b in np.arange(3, 4):
                if b == 0:
                    strgvarbmagt = 'vmag'
                    lablxaxi = 'V Magnitude'
                    if gdat.boolobjt:
                        varbtarg = gdat.vmagstar
                    varb = dictpopl['vmagstar']
                if b == 1:
                    strgvarbmagt = 'jmag'
                    lablxaxi = 'J Magnitude'
                    if gdat.boolobjt:
                        varbtarg = gdat.jmagstar
                    varb = dictpopl['jmagstar']
                if b == 2:
                    strgvarbmagt = 'rvsascal_vmag'
                    lablxaxi = '$K^{\prime}_{V}$'
                    if gdat.boolobjt:
                        varbtarg = np.sqrt(10**(-gdat.vmagstar / 2.5)) / gdat.massstar**(2. / 3.)
                    varb = np.sqrt(10**(-dictpopl['vmagstar'] / 2.5)) / dictpopl['massstar']**(2. / 3.)
                if b == 3:
                    strgvarbmagt = 'rvsascal_jmag'
                    lablxaxi = '$K^{\prime}_{J}$'
                    if gdat.boolobjt:
                        varbtarg = np.sqrt(10**(-gdat.vmagstar / 2.5)) / gdat.massstar**(2. / 3.)
                    varb = np.sqrt(10**(-dictpopl['jmagstar'] / 2.5)) / dictpopl['massstar']**(2. / 3.)
                #for a in range(3):
                for a in np.arange(2, 3):
                    figr, axis = plt.subplots(figsize=gdat.figrsize)
                    if a == 0:
                        indx = np.where(dictpopl['boolfrst'])[0]
                    if a == 1:
                        indx = np.where(dictpopl['boolfrst'] & (dictpopl['numbplanstar'] > 3))[0]
                    if a == 2:
                        indx = np.where(dictpopl['boolfrst'] & (dictpopl['numbplantranstar'] > 3))[0]
                    
                    if gdat.boolobjt and (b == 2 or b == 3):
                        print('varbtarg')
                        print(varbtarg)
                        print('np.amax(varb[indx])')
                        print(np.amax(varb[indx]))
                        normfact = max(varbtarg, np.nanmax(varb[indx]))
                    else:
                        normfact = 1.
                    if gdat.boolobjt:
                        varbtargnorm = varbtarg / normfact
                    varbnorm = varb[indx] / normfact
                    axis.scatter(varbnorm, dictpopl['numbplanstar'][indx], s=1, color='black')
                    
                    indxsort = np.argsort(varbnorm)
                    if b == 2 or b == 3:
                        indxsort = indxsort[::-1]

                    listnameaddd = []
                    cntr = 0
                    maxmnumbname = min(5, varbnorm.size)
                    while True:
                        k = indxsort[cntr]
                        nameadd = dictpopl['namestar'][indx][k]
                        if not nameadd in listnameaddd:
                            axis.text(varbnorm[k], dictpopl['numbplanstar'][indx][k] + 0.5, nameadd, size=6, \
                                                                                                    va='center', ha='right', rotation=45)
                            listnameaddd.append(nameadd)
                        cntr += 1
                        if len(listnameaddd) == maxmnumbname: 
                            break
                    if gdat.boolobjt:
                        axis.scatter(varbtargnorm, gdat.numbplan, s=5, color='black', marker='x')
                        axis.text(varbtargnorm, gdat.numbplan + 0.5, gdat.labltarg, size=8, color='black', \
                                                                                                    va='center', ha='center', rotation=45)
                    axis.set_ylabel(r'Number of transiting planets')
                    axis.set_xlabel(lablxaxi)
                    plt.subplots_adjust(bottom=0.2)
                    plt.subplots_adjust(left=0.2)
                    path = pathimagmagt + '%snumb_%s_%s_%s_%d.%s' % (strgvarbmagt, gdat.strgtarg, strgpdfn, strgpopl, a, gdat.typefileplot)
                    print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()

                    figr, axis = plt.subplots(figsize=gdat.figrsize)
                    axis.hist(varbnorm, 50)
                    if gdat.boolobjt:
                        axis.axvline(varbtargnorm, color='black', ls='--')
                        axis.text(0.3, 0.9, gdat.labltarg, size=8, color='black', transform=axis.transAxes, va='center', ha='center')
                    axis.set_ylabel(r'Number of systems')
                    axis.set_xlabel(lablxaxi)
                    plt.subplots_adjust(bottom=0.2)
                    plt.subplots_adjust(left=0.2)
                    path = pathimagmagt + 'hist_%s_%s_%s_%s_%d.%s' % (strgvarbmagt, gdat.strgtarg, strgpdfn, strgpopl, a, gdat.typefileplot)
                    print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()
        indxinit = np.where(dictpopl['peri'] < 100)[0]
            
        
        # distribution plots
        boolwritordr = True
        for k, strgxaxi in enumerate(liststrgvarb):
            for m, strgyaxi in enumerate(liststrgvarb):
                if k >= m:
                    continue
                
                booltemp = False
                # temp
                if (strgxaxi == 'radiplaneart' and strgyaxi == 'tmptplan'):
                    booltemp = True
                if (strgxaxi == 'radiplaneart' and strgyaxi == 'logttsmm'):
                    booltemp = True
                #if (strgxaxi == 'tmptplan' and strgyaxi == 'vesc'):
                #    booltemp = True
                #if (strgxaxi == 'peri' and strgyaxi == 'inso'):
                #    booltemp = True
                #if (strgxaxi == 'radistarsunn' and strgyaxi == 'radiplaneart'):
                #    booltemp = True
                #if (strgxaxi == 'tmptplan' and strgyaxi == 'radistarsunn'):
                #    booltemp = True
                #if (strgxaxi == 'projoblq' or strgyaxi == 'projoblq'):
                #    booltemp = True
                #if (strgxaxi == 'vsiistar' or strgyaxi == 'vsiistar'):
                #    booltemp = True
                
                if not booltemp:
                    continue

                for strgcuttmult in liststrgcuttmult:
                    
                    for strgcuttradi in liststrgcuttradi:
                        
                        # ensure allr plots do not come with R_E units
                        if strgcuttradi == 'allr' and (strgxaxi == 'radiplaneart' or strgyaxi == 'radiplaneart'):
                            continue

                        for strgcuttmass in liststrgcuttmass:
                            
                            # impose radius cut
                            indx = np.intersect1d(indxpoplcuttmult[strgcuttmult], indxpoplcuttmass[strgcuttmass])
                            indx = np.intersect1d(indxpoplcuttradi[strgcuttradi], indx)
                            indx = np.intersect1d(indx, indxinit)
                            
                            # write out
                            if boolwritordr:
                            
                                for y in range(1):
                                    
                                    if y == 0:
                                        strgstrgsort = 'tsmm'
                                    else:
                                        strgstrgsort = 'esmm'
                                    print('strgcuttmult')
                                    print(strgcuttmult)
                                    print('strgcuttradi')
                                    print(strgcuttradi)
                                    print('strgcuttmass')
                                    print(strgcuttmass)
                                    print('strgstrgsort')
                                    print(strgstrgsort)
                                    # sort
                                    dicttemp = dict()
                                    
                                    for strgfeat, valu in dictpopl.items():
                                       
                                        #if strgfeat.startswith('stdv'):
                                        #    print('Skipping feature plot for %s...' % strgfeat)
                                        #    continue
                                        if strgfeat.startswith('Unnamed'):
                                            continue
                                        if strgfeat == 'toii':
                                            continue
                                        #if strgfeat.startswith('stdv'):
                                        #    continue
                                        if gdat.boolobjt:
                                            if strgfeat in gdat.dictfeatobjt.keys():
                                                dicttemp[strgfeat] = np.concatenate([dictpopl[strgfeat][indx], gdat.dictfeatobjt[strgfeat]])
                                            else:
                                                dicttemp[strgfeat] = np.concatenate([dictpopl[strgfeat][indx], gdat.dicterrr[strgfeat][0, :]])
                                        else:
                                            dicttemp[strgfeat] = np.copy(dictpopl[strgfeat][indx])
                                    indxgood = np.where(np.isfinite(dicttemp[strgstrgsort]))[0]
                                    indxsort = np.argsort(dicttemp[strgstrgsort][indxgood])[::-1]
                                    
                                    path = gdat.pathdata + '%s_%s_%s_%s_%s.csv' % (strgpopl, strgcuttmult, strgcuttradi, strgcuttmass, strgstrgsort)
                                    objtfile = open(path, 'w')

                                    strghead = '%4s, %20s, %7s, %12s, %12s, %12s, %12s, %12s, %12s, %12s, %12s, %12s, %12s\n' % \
                                                ('Rank', 'Name', 'Transit', 'Jmag', 'R [R_E]', 'e_R [R_E]', 'M [M_E]', \
                                                                                    'e_M [M_E]', 'T [K]', 'e_T [K]', 'TSM', 'e_TSM', 'e_TSM [%]')
                                    objtfile.write(strghead)
                                    cntr = 0
                                    for l in indxgood[indxsort]:
                                        strgline = '%4d, %20s, %7d, %12.3f, %12.3f, %12.3f, %12.3f, %12.3f, %12.3f, %12.3f, %12.3f, %12.3f, %12.3f\n' % \
                                                                                    (cntr, dicttemp['nameplan'][l], dicttemp['booltran'][l], \
                                                        dicttemp['jmagstar'][l], \
                                        dicttemp['radiplaneart'][l], dicttemp['stdvradiplaneart'][l], \
                                        dicttemp['massplaneart'][l], dicttemp['stdvmassplaneart'][l], \
                                        dicttemp['tmptplan'][l], dicttemp['stdvtmptplan'][l], \
                                                                                    dicttemp[strgstrgsort][l], \
                                                                                    dicttemp['stdv' + strgstrgsort][l], \
                                                                                100. * dicttemp['stdv' + strgstrgsort][l] / dicttemp[strgstrgsort][l], \
                                                                                    )
                                        objtfile.write(strgline)
                                        cntr += 1 
                                    print('Writing to %s...' % path)
                                    objtfile.close()
                            
                            if gdat.boolplotprop:
                                # repeat, one without text, one with text
                                for b, strgtext in enumerate(liststrgtext):
                                    figr, axis = plt.subplots(figsize=gdat.figrsize)
                                    
                                    ## population
                                    axis.errorbar(dictpopl[strgxaxi][indx], dictpopl[strgyaxi][indx], ls='', ms=1, lw=1, marker='o', \
                                                                                                            color='k', alpha=1., zorder=1)
                                    
                                    ## this system
                                    if gdat.boolobjt:
                                        for j in gdat.indxplan:
                                            if strgxaxi in gdat.dicterrr:
                                                xdat = gdat.dicterrr[strgxaxi][0, j, None]
                                                xerr = gdat.dicterrr[strgxaxi][1:3, j, None]
                                            if strgyaxi in gdat.dicterrr:
                                                ydat = gdat.dicterrr[strgyaxi][0, j, None]
                                                yerr = gdat.dicterrr[strgyaxi][1:3, j, None]
                                            
                                            if strgcuttradi == 'r004' and strgxaxi == 'radiplaneart' and xdat > 4.:
                                                continue
                                            if strgcuttradi == 'r204' and strgxaxi == 'radiplaneart' and (xdat > 4. or xdat < 2):
                                                continue
                                            if strgcuttradi == 'r154' and strgxaxi == 'radiplaneart' and (xdat > 4. or xdat < 1.5):
                                                continue
                                            if strgcuttradi == 'r004' and strgyaxi == 'radiplaneart' and ydat > 4.:
                                                continue
                                            if strgcuttradi == 'r204' and strgyaxi == 'radiplaneart' and (ydat > 4. or ydat < 2):
                                                continue
                                            if strgcuttradi == 'r154' and strgyaxi == 'radiplaneart' and (ydat > 4. or ydat < 1.5):
                                                continue
                                            
                                            if strgxaxi in gdat.listfeatstar and strgyaxi in gdat.listfeatstar:
                                                axis.errorbar(xdat, ydat, color='k', lw=1, xerr=xerr, yerr=yerr, ls='', marker='o', ms=6, zorder=2)
                                                axis.text(0.85, 0.9 - j * 0.08, gdat.labltarg, color='k', \
                                                                                                      va='center', ha='center', transform=axis.transAxes)
                                                break
                                            else:
                                                
                                                if not strgxaxi in gdat.dicterrr and strgyaxi in gdat.dicterrr:
                                                    if strgyaxi in gdat.listfeatstar:
                                                        axis.axhline(ydat, color='k', lw=1, ls='--', zorder=2)
                                                        axis.text(0.85, 0.9 - j * 0.08, gdat.labltarg, color='k', \
                                                                                                      va='center', ha='center', transform=axis.transAxes)
                                                        break
                                                    else:
                                                        axis.axhline(ydat, color=gdat.listcolrplan[j], lw=1, ls='--', zorder=2)
                                                if not strgyaxi in gdat.dicterrr and strgxaxi in gdat.dicterrr:
                                                    if strgxaxi in gdat.listfeatstar:
                                                        axis.axvline(xdat, color='k', lw=1, ls='--', zorder=2)
                                                        axis.text(0.85, 0.9 - j * 0.08, gdat.labltarg, color='k', \
                                                                                                      va='center', ha='center', transform=axis.transAxes)
                                                        break
                                                    else:
                                                        axis.axvline(xdat, color=gdat.listcolrplan[j], lw=1, ls='--')
                                                if strgxaxi in gdat.dicterrr and strgyaxi in gdat.dicterrr:
                                                    axis.errorbar(xdat, ydat, color=gdat.listcolrplan[j], lw=1, xerr=xerr, yerr=yerr, ls='', marker='o', \
                                                                                                                                        zorder=2, ms=6)
                                                
                                                if strgxaxi in gdat.dicterrr or strgyaxi in gdat.dicterrr:
                                                    axis.text(0.85, 0.9 - j * 0.08, r'\textbf{%s}' % gdat.liststrgplan[j], color=gdat.listcolrplan[j], \
                                                                                                        va='center', ha='center', transform=axis.transAxes)
                                    
                                    # include text
                                    if b == 1:
                                        boolfinitsmm = np.isfinite(dictpopl['tsmm'][indx])
                                        boolfinixdat = np.isfinite(dictpopl[strgxaxi][indx])
                                        boolfiniydat = np.isfinite(dictpopl[strgyaxi][indx])
                                        indxfinitsmm = np.where(boolfinitsmm & boolfinixdat & boolfiniydat)[0]
                                        indxinfitsmm = np.where((~boolfinitsmm) & boolfinixdat & boolfiniydat)[0]

                                        indxtemp = np.argsort(dictpopl['tsmm'][indx][indxfinitsmm])[::-1]
                                        indxextr = np.concatenate((indx[indxfinitsmm[indxtemp]], indx[indxinfitsmm]))
                                        indxextr = indxextr[:numbtext]
                                        for l in indxextr:
                                            text = '%s, R=%.3g$\pm$%.3g, M=%.3g$\pm$%.3g, TSM=%.3g' % (dictpopl['nameplan'][l], \
                                                                                             dictpopl['radiplaneart'][l], \
                                                                                             dictpopl['stdvradiplaneart'][l], \
                                                                                             dictpopl['massplaneart'][l], \
                                                                                             dictpopl['stdvmassplaneart'][l], \
                                                                                             dictpopl['tsmm'][l])
                                            xdat = dictpopl[strgxaxi][l]
                                            ydat = dictpopl[strgyaxi][l]
                                            if np.isfinite(xdat) and np.isfinite(ydat):
                                                objttext = axis.text(xdat, ydat, text, size=1, ha='center', va='center')
                                    
                                    if strgxaxi == 'tmptplan' and strgyaxi == 'vesc':
                                        xlim = axis.get_xlim()
                                        arrytmptplan = np.linspace(xlim[0], xlim[1], 1000)
                                        #h20 = 2+16=18
                                        #nh3 = 14
                                        #ch4 = 12
                                        cons = [2., 4., 18]
                                        for i in range(len(cons)):
                                            arryyaxi = (2. * arrytmptplan / 2000. / cons[i])**0.5 * 50.
                                            axis.plot(arrytmptplan, arryyaxi, color='grey', alpha=0.5)
                                        axis.set_xlim(xlim)

                                    if strgxaxi == 'radiplaneart' and strgyaxi == 'massplaneart':
                                        gdat.listlabldenscomp = ['Earth-like', 'Pure Water', 'Pure Iron']
                                        listdenscomp = [1., 0.1813, 1.428]
                                        listposicomp = [[13., 2.6], [4.7, 3.5], [13., 1.9]]
                                        gdat.numbdenscomp = len(gdat.listlabldenscomp)
                                        gdat.indxdenscomp = np.arange(gdat.numbdenscomp)
                                        masscompdens = np.linspace(0.5, 16.) # M_E
                                        for i in gdat.indxdenscomp:
                                            radicompdens = (masscompdens / listdenscomp[i])**(1. / 3.)
                                            axis.plot(masscompdens, radicompdens, color='grey')
                                        for i in gdat.indxdenscomp:
                                            axis.text(listposicomp[i][0], listposicomp[i][1], gdat.listlabldenscomp[i])
                                    
                                    axis.set_xlabel(listlablvarbtotl[k])
                                    axis.set_ylabel(listlablvarbtotl[m])
                                    #plt.subplots_adjust(left=0.2)
                                    #plt.subplots_adjust(bottom=0.2)
                                    pathimagprop = getattr(gdat, 'pathimagprop' + strgtext + strgpdfn)
                                    path = pathimagprop + 'feat_%s_%s_%s_%s_%s_%s_%s_%s_%s.%s' % \
                                                            (strgxaxi, strgyaxi, gdat.strgtarg, strgcuttmult, strgcuttradi, strgcuttmass, strgpopl, \
                                                                                       strgtext, strgpdfn, gdat.typefileplot)
                                    print('Writing to %s...' % path)
                                    plt.savefig(path)
                                    plt.close()
                boolwritordr = False

   
def bdtr_wrap(gdat, epocmask, perimask, duramask, indxdatatser):
    
    gdat.listobjtspln = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gdat.indxsplnregi = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gdat.listindxtimeregi = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gdat.indxtimeregioutt = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    
    b = indxdatatser
    for p in gdat.indxinst[b]:
        for y in gdat.indxchun[b][p]:
            if gdat.boolbdtr:
                gdat.rflxbdtrregi, gdat.listindxtimeregi[b][p][y], gdat.indxtimeregioutt[b][p][y], gdat.listobjtspln[b][p][y], timeedge = \
                                 tesstarg.util.bdtr_tser(gdat.listarrytser['raww'][b][p][y][:, 0], gdat.listarrytser['raww'][b][p][y][:, 1], \
                                                            epocmask=epocmask, perimask=perimask, duramask=duramask, \
                                                            verbtype=gdat.verbtype, durabrek=gdat.durabrek, ordrspln=gdat.ordrspln, bdtrtype=gdat.bdtrtype)
                gdat.listarrytser['bdtr'][b][p][y] = np.copy(gdat.listarrytser['raww'][b][p][y])
                gdat.listarrytser['bdtr'][b][p][y][:, 1] = np.concatenate(gdat.rflxbdtrregi)
                numbsplnregi = len(gdat.rflxbdtrregi)
                gdat.indxsplnregi[b][p][y] = np.arange(numbsplnregi)
            else:
                gdat.listarrytser['bdtr'][b][p][y] = gdat.listarrytser['raww'][b][p][y]
        # merge chuncks
        gdat.arrytser['bdtr'][b][p] = np.concatenate(gdat.listarrytser['bdtr'][b][p], axis=0)


def plot_tserwrap(gdat, strgproc, boolchun=True, boolcolr=True):
    
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            
            if boolchun and gdat.numbchun[b][p] == 1:
                continue
            
            for y in gdat.indxchun[b][p]:
                figr, axis = plt.subplots(figsize=gdat.figrsizeydobskin)
                
                if boolchun:
                    arrytser = gdat.listarrytser[strgproc][b][p][y]
                else:
                    if y > 0:
                        continue
                    arrytser = gdat.arrytser[strgproc][b][p]
                
                print('arrytser')
                print(arrytser)
                axis.plot(arrytser[:, 0] - gdat.timetess, arrytser[:, 1], color='grey', marker='.', ls='', ms=1, rasterized=True)
                if gdat.listlimttimemask is not None:
                    axis.plot(arrytser[strgproc][b][p][listindxtimegood, 0] - gdat.timetess, \
                                                            arrytser[strgproc][b][p][listindxtimegood, 1], color='k', marker='.', ls='', ms=1)
                if boolcolr:
                    # color and name transits
                    ylim = axis.get_ylim()
                    listtimetext = []
                    for j in gdat.indxplan:
                        if boolchun:
                            indxtime = gdat.listindxtimetranchun[j][b][p][y] 
                        else:
                            if y > 0:
                                continue
                            indxtime = gdat.listindxtimetran[j][b][p] 
                        
                        colr = gdat.listcolrplan[j]
                        axis.plot(arrytser[indxtime, 0] - gdat.timetess, arrytser[indxtime, 1], \
                                                                                           color=colr, marker='.', ls='', ms=1, rasterized=True)
                        # draw planet names
                        for n in np.linspace(-gdat.numbcyclcolrplot, gdat.numbcyclcolrplot, 2 * gdat.numbcyclcolrplot + 1):
                            time = gdat.epocprio[j] + n * gdat.periprio[j] - gdat.timetess
                            if np.where(abs(arrytser[:, 0] - gdat.timetess - time) < 0.1)[0].size > 0:
                                
                                # add a vertical offset if overlapping
                                if np.where(abs(np.array(listtimetext) - time) < 0.5)[0].size > 0:
                                    ypostemp = ylim[0] + (ylim[1] - ylim[0]) * 0.95
                                else:
                                    ypostemp = ylim[0] + (ylim[1] - ylim[0]) * 0.9

                                # draw the planet letter
                                axis.text(time, ypostemp, r'\textbf{%s}' % gdat.liststrgplan[j], color=gdat.listcolrplan[j], va='center', ha='center')
                                listtimetext.append(time)
                
                axis.set_xlabel('Time [BJD - %d]' % gdat.timetess)
                axis.set_ylabel(gdat.listlabltser[b])
                plt.subplots_adjust(bottom=0.2)
                strgprio = ''
                if strgproc == 'raww':
                    strgprio = ''
                else:
                    strgprio = '_%s' % gdat.strgprio
                strgcolr = ''
                if boolcolr:
                    strgcolr = '_colr'
                strgchun = ''
                if boolchun:
                    strgchun = '_chu%d' % y
                path = gdat.pathimag + '%s%s%s_%s%s_%s%s.%s' % (gdat.liststrgtser[b], strgproc, strgcolr, gdat.liststrginst[b][p], \
                                                                                            strgchun, gdat.strgtarg, strgprio, gdat.typefileplot)
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()


def plot_tser(gdat, strgproc):
    
    plot_tserwrap(gdat, strgproc, boolchun=False, boolcolr=False)
    plot_tserwrap(gdat, strgproc, boolchun=True, boolcolr=False)
    if strgproc != 'raww':
        if gdat.numbplan > 0:
            
            plot_tserwrap(gdat, strgproc, boolchun=False, boolcolr=True)
            plot_tserwrap(gdat, strgproc, boolchun=True, boolcolr=True)

            for b in gdat.indxdatatser:
                if b == 1:
                    continue
                for p in gdat.indxinst[b]:
                    # plot only the in-transit data
                    figr, axis = plt.subplots(gdat.numbplan, 1, figsize=gdat.figrsizeydobskin, sharex=True)
                    if gdat.numbplan == 1:
                        axis = [axis]
                    for jj, j in enumerate(gdat.indxplan):
                        axis[jj].plot(gdat.arrytser[strgproc][b][p][gdat.listindxtimetran[j][b][p], 0] - gdat.timetess, \
                                                                                      gdat.arrytser[strgproc][b][p][gdat.listindxtimetran[j][b][p], 1], \
                                                                                               color=gdat.listcolrplan[j], marker='o', ls='', ms=0.2)
                    
                    axis[-1].set_ylabel('Relative Flux')
                    #axis[-1].yaxis.set_label_coords(0, gdat.numbplan * 0.5)
                    axis[-1].set_xlabel('Time [BJD - %d]' % gdat.timetess)
                    
                    #plt.subplots_adjust(bottom=0.2)
                    path = gdat.pathimag + 'rflx%s_intr_%s_%s_%s.%s' % (strgproc, gdat.liststrginst[b][p], gdat.strgtarg, gdat.strgprio, gdat.typefileplot)
                    print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()
        
    if gdat.boolbdtr and strgproc == 'bdtr':
        for b in gdat.indxdatatser:
            if b == 1:
                continue
            for p in gdat.indxinst[b]:
                for y in gdat.indxchun[b][p]:
                    # plot baseline-detrending
                    figr, axis = plt.subplots(2, 1, figsize=gdat.figrsizeydob)
                    for i in gdat.indxsplnregi[b][p][y]:
                        ## masked and non-baseline-detrended light curve
                        indxtimetemp = gdat.listindxtimeregi[b][p][y][i][gdat.indxtimeregioutt[b][p][y][i]]
                        axis[0].plot(gdat.listarrytser['raww'][b][p][y][indxtimetemp, 0] - gdat.timetess, \
                                                         gdat.listarrytser['raww'][b][p][y][indxtimetemp, 1], rasterized=True, alpha=gdat.alphraww, \
                                                                                                        marker='o', ls='', ms=1, color='grey')
                        ## spline
                        if gdat.listobjtspln[b][p][y] is not None and gdat.listobjtspln[b][p][y][i] is not None:
                            timesplnregifine = np.linspace(gdat.listarrytser[strgproc][b][p][y][gdat.listindxtimeregi[b][p][y][i], 0][0], \
                                                                 gdat.listarrytser[strgproc][b][p][y][gdat.listindxtimeregi[b][p][y][i], 0][-1], 1000)
                            axis[0].plot(timesplnregifine - gdat.timetess, gdat.listobjtspln[b][p][y][i](timesplnregifine), 'b-', lw=3, rasterized=True)
                        ## baseline-detrended light curve
                        indxtimetemp = gdat.listindxtimeregi[b][p][y][i]
                        axis[1].plot(gdat.listarrytser[strgproc][b][p][y][indxtimetemp, 0] - gdat.timetess, \
                                                                                    gdat.listarrytser[strgproc][b][p][y][indxtimetemp, 1], rasterized=True, \
                                                                                                          marker='o', ms=1, ls='', color='grey')
                    for a in range(2):
                        axis[a].set_ylabel('Relative Flux')
                    axis[0].set_xticklabels([])
                    axis[1].set_xlabel('Time [BJD - %d]' % gdat.timetess)
                    plt.subplots_adjust(hspace=0.)
                    path = gdat.pathimag + 'rflxbdtr_bdtr_%s_%s_%s_%s.%s' % (gdat.liststrginst[b][p], \
                                                        gdat.liststrgchun[b][p][y], gdat.strgtarg, gdat.strgprio, gdat.typefileplot)
                    print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()
                            

def init( \
         strgexar=None, \
         toiitarg=None, \
         ticitarg=None, \

         # stellar prior
         strgmast=None, \
         
         # a string for the label of the target
         labltarg=None, \
         
         # a string that describes the prior
         strgprio=None, \

         # a string for the folder name and file name extensions
         strgtarg=None, \
         
         # mode of operation
         ## Boolean flag to use and plot time-series data
         booldatatser=True, \
         ## Boolean flag to plot the properties of exoplanets
         boolplotprop=False, \
         ## Boolean flag to enforce offline operation
         boolforcoffl=False, \
        
         listdatatype=None, \

         # input
         listpathdatainpt=None, \
         ## type of TESS light curve to be used for analysis: 'SPOC', 'lygos'
         typedatatess='SPOC', \
         ## type of SPOC light curve to be used for analysis: 'PDC', 'SAP'
         typedataspoc='PDC', \
         ## maximum number of stars to fit in lygos
         maxmnumbstarlygo=1, \
         # list of labels indicating instruments
         listlablinst=[['TESS'], []], \
         # list of strings indicating instruments
         liststrginst=None, \
         # list of strings indicating chunks
         liststrgchun=None, \
         # list of chunk indices for each instrument
         listindxchuninst=None, \
            
         indxplanalle=None, \

         # planet names
         ## list of letters to be assigned to planets
         liststrgplan=None, \
         ## list of colors to be assigned to planets
         listcolrplan=None, \
         ## Boolean flag to assign them letters *after* ordering them in orbital period, unless liststrgplan is specified by the user
         boolordrplanname=True, \

         # preprocessing
         boolbdtr=True, \
         boolclip=False, \
         ## dilution: None (no correction), 'lygos' for estimation via lygos, or float value
         dilu=None, \
         ## baseline detrending
         durabrek=1., \
         ordrspln=3, \
         bdtrtype='spln', \
         durakernbdtrmedi=1., \
         ## Boolean flag to mask bad data
         boolmaskqual=True, \
         ## time limits to mask
         listlimttimemask=None, \
    
         # planet search
         ## maximum number of planets for TLS
         maxmnumbplantlsq=None, \
        
         # include the ExoFOP catalog in the comparisons to exoplanet population
         boolexof=True, \

         # model
         ## priors
         ### type of priors for stars: 'tici', 'exar', 'inpt'
         typepriostar=None, \

         # type of priors for planets
         typeprioplan=None, \
         
         ## prior values
         ### photometric and RV model
         #### means
         rratprio=None, \
         rsmaprio=None, \
         epocprio=None, \
         periprio=None, \
         cosiprio=None, \
         ecosprio=None, \
         esinprio=None, \
         rvsaprio=None, \
         #### uncertainties
         stdvrratprio=None, \
         stdvrsmaprio=None, \
         stdvepocprio=None, \
         stdvperiprio=None, \
         stdvcosiprio=None, \
         stdvecosprio=None, \
         stdvesinprio=None, \
         stdvrvsaprio=None, \
        
         ### others 
         #### mean
         projoblqprio=None, \
         #### uncertainties
         stdvprojoblqprio=None, \

         radistar=None, \
         massstar=None, \
         tmptstar=None, \
         rascstar=None, \
         declstar=None, \
         vsiistarprio=None, \
         stdvradistar=None, \
         stdvmassstar=None, \
         stdvtmptstar=None, \
         stdvrascstar=None, \
         stdvdeclstar=None, \
         stdvvsiistarprio=None, \

         # type of inference, alle for allesfitter, trap for trapezoidal fit
         infetype='alle', \

         # allesfitter settings
         boolexecalle=False, \
         boolallebkgdgaus=False, \
         boolalleorbt=True, \
         ## allesfitter analysis type
         listtypeallemodl=['orbt'], \
         dictdictallesett=None, \
         dictdictallepara=None, \
         # output
         ## Boolean flag to process the prior
         boolprocprio=True, \

         ## Boolean flag to write planet name on plots
         boolwritplan=True, \
         ## plotting
         makeprioplot=True, \
         ## Boolean flag to rasterize the raw time-series on plots
         boolrastraww=True, \
         ## list of transit depths to indicate on the light curve and phase curve plots
         listdeptdraw=None, \
         ### list of offsets for the planet annotations in the TSM/ESM plot
         offstextatmoraditmpt=None, \
         offstextatmoradimetr=None, \

         booldiagmode=True, \
         
         verbtype=1, \

         **args \
        ):
    
    # construct global object
    gdat = tdpy.util.gdatstrt()
    
    # copy unnamed inputs to the global object
    #for attr, valu in locals().iter():
    for attr, valu in locals().items():
        if '__' not in attr and attr != 'gdat':
            setattr(gdat, attr, valu)

    # copy named arguments to the global object
    for strg, valu in args.items():
        setattr(gdat, strg, valu)

    # string for date and time
    gdat.strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # paths
    gdat.pathbase = os.environ['PEXO_DATA_PATH'] + '/'
    print('PEXO initialized at %s...' % gdat.strgtimestmp)
    
    # check input
    if gdat.boolexecalle and not gdat.booldatatser:
        raise Exception('')

    # parse inpt
    ## determine whether pexo will work on a specific target
    if gdat.strgexar is None and gdat.toiitarg is None and gdat.ticitarg is None:
        gdat.boolobjt = False
    else:
        gdat.boolobjt = True
    print('gdat.boolobjt')
    print(gdat.boolobjt)
    
    gdat.strgtoiibase = None

    if gdat.boolobjt:
        if gdat.toiitarg is not None:
            gdat.strgtoiibase = str(gdat.toiitarg)

        if gdat.typepriostar is None:
            if gdat.radistar is not None:
                gdat.typepriostar = 'inpt'
            else:
                gdat.typepriostar = 'tici'
        print('Stellar parameter prior type: %s' % gdat.typepriostar)
        
        if gdat.typeprioplan is None:
            if gdat.epocprio is not None:
                gdat.typeprioplan = 'inpt'
            else:
                if gdat.ticitarg is not None:
                    gdat.typeprioplan = 'tlsq'
                if gdat.strgexar is not None:
                    gdat.typeprioplan = 'exar'
                if gdat.toiitarg is not None:
                    gdat.typeprioplan = 'exof'
        print('Planetary parameter prior type: %s' % gdat.typeprioplan)
        
        if gdat.labltarg is None:
            if gdat.strgmast is not None:
                gdat.labltarg = gdat.strgmast
        
            elif gdat.strgtoiibase is not None:
                gdat.labltarg = 'TOI ' + gdat.strgtoiibase
        
            elif gdat.typeprioplan == 'exar':
                gdat.labltarg = gdat.strgexar
        
            elif gdat.ticitarg is not None:
                gdat.labltarg = 'TIC %d' % gdat.ticitarg
        
        if gdat.strgtarg is None:
            gdat.strgtarg = ''.join(gdat.labltarg.split(' '))
        
        print('gdat.labltarg')
        print(gdat.labltarg)
        
        if gdat.strgtarg is None:
            raise Exception('')
        print('gdat.strgtarg')
        print(gdat.strgtarg)

        gdat.pathobjt = gdat.pathbase + '%s/' % gdat.strgtarg
    else:
        gdat.pathobjt = gdat.pathbase + 'gene/'
    
    gdat.pathtoii = gdat.pathbase + 'data/exofop_toilists_20200916.csv'
    print('Reading from %s...' % gdat.pathtoii)
    objtexof = pd.read_csv(gdat.pathtoii, skiprows=0)
    
    if gdat.boolobjt and gdat.booldatatser:
        gdat.liststrgdatatser = ['lcur', 'rvel']
        gdat.numbdatatser = len(gdat.liststrgdatatser)
        gdat.indxdatatser = np.arange(gdat.numbdatatser)

        gdat.numbinst = np.empty(gdat.numbdatatser, dtype=int)
        gdat.indxinst = [[] for b in gdat.indxdatatser]
        for b in gdat.indxdatatser:
            gdat.numbinst[b] = len(gdat.listlablinst[b])
            gdat.indxinst[b] = np.arange(gdat.numbinst[b])
        
        print('gdat.numbinst')
        print(gdat.numbinst)
        print('gdat.indxinst')
        print(gdat.indxinst)
        
        if gdat.liststrginst is None:
            gdat.liststrginst = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
            for b in gdat.indxdatatser:
                for p in gdat.indxinst[b]:
                    gdat.liststrginst[b][p] = ''.join(gdat.listlablinst[b][p].split(' '))

        if 'TESS' in gdat.liststrginst[0]:
            print('gdat.typedatatess')
            print(gdat.typedatatess)
            gdat.arrytsersapp = [[]]
            gdat.arrytserpdcc = [[]]
            gdat.listarrytsersapp = [[]]
            gdat.listarrytserpdcc = [[]]
            arrylcurtess, gdat.arrytsersapp[0], \
                    gdat.arrytserpdcc[0], listarrylcurtess, gdat.listarrytsersapp[0], \
                            gdat.listarrytserpdcc[0], gdat.listisec, gdat.listicam, gdat.listiccd = \
                                  tesstarg.util.retr_data(gdat.strgmast, gdat.pathobjt, \
                                                typedatatess=typedatatess, \
                                                typedataspoc=typedataspoc, \
                                                boolmaskqual=gdat.boolmaskqual, \
                                                labltarg=gdat.labltarg, \
                                                strgtarg=gdat.strgtarg, \
                                                maxmnumbstarlygo=gdat.maxmnumbstarlygo)
        
        # determine number of chunks
        gdat.numbchun = [np.empty(gdat.numbinst[b], dtype=int) for b in gdat.indxdatatser]
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                if gdat.listpathdatainpt is not None:
                    gdat.numbchun[b][p] = len(gdat.listpathdatainpt[b][p])
                
                if b == 0 and gdat.liststrginst[b][p] == 'TESS':
                    gdat.numbchun[b][p] = len(listarrylcurtess)
        
                if b == 1 and gdat.liststrginst[b][p] == 'PFS' and gdat.listdatatype[b][p] == 'mock':
                    gdat.numbchun[b][p] = 1
        
        gdat.indxchun = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                gdat.indxchun[b][p] = np.arange(gdat.numbchun[b][p], dtype=int)

        gdat.liststrgchun = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                for y in gdat.indxchun[b][p]:
                    gdat.liststrgchun[b][p][y] = 'chu%d' % y
        
        print('gdat.liststrginst')
        print(gdat.liststrginst)
        print('gdat.liststrgchun')
        print(gdat.liststrgchun)
        
        # check the user-defined gdat.listpathdatainpt
        if gdat.listpathdatainpt is not None:
            for b in gdat.indxdatatser:
                for p in gdat.indxinst[b]:
                    if not isinstance(gdat.listpathdatainpt[b][p], list):
                        raise Exception('')

        # list of data types (real or mock) for each instrument for both light curve and RV data
        if gdat.listdatatype is None:
            gdat.listdatatype = [[] for b in gdat.indxdatatser]
            for b in gdat.indxdatatser:
                gdat.listdatatype[b] = ['real' for p in gdat.indxinst[b]]

    ### Boolean flag to plot the initial guess
    gdat.boolprioalle = False
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            if gdat.listdatatype[b][p] == 'mock':
                gdat.boolprioalle = True
    
    if gdat.typeprioplan == 'tlsq' and gdat.boolprioalle:
        raise Exception('')

    # settings
    ## plotting
    gdat.numbcyclcolrplot = 300
    gdat.alphraww = 0.2
    ### percentile for zoom plots of relative flux
    gdat.pctlrflx = 95.
    
    if gdat.dictdictallesett is None:
        gdat.dictdictallesett = dict()
        for typeallemodl in gdat.listtypeallemodl:
            gdat.dictdictallesett[typeallemodl] = None

    if gdat.dictdictallepara is None:
        gdat.dictdictallepara = dict()
        for typeallemodl in gdat.listtypeallemodl:
            gdat.dictdictallepara[typeallemodl] = None

    gdat.listlabltser = ['Relative Flux', 'Radial Velocity [km/s]']
    gdat.liststrgtser = ['rflx', 'rvel']
    gdat.liststrgtseralle = ['flux', 'rv']
    if gdat.boolobjt:
        print('Target identifier inputs:')
        print('strgexar')
        print(strgexar)
        print('toiitarg')
        print(toiitarg)
        
        if gdat.booldatatser:
            gdat.timetess = 2457000

        if gdat.strgexar is not None and gdat.toiitarg is not None:
            raise Exception('')
       
        if gdat.booldatatser:
            print('Light curve data: %s' % gdat.listlablinst[0])
            print('RV data: %s' % gdat.listlablinst[1])
    
    if gdat.strgprio is None:
        gdat.strgprio = gdat.typeprioplan

    if gdat.typeprioplan != 'inpt' and gdat.epocprio is not None:
        raise Exception('')

    # plotting
    gdat.typefileplot = 'pdf'
    gdat.figrsize = [4.5, 3.5]
    gdat.figrsizeydob = [8., 4.]
    gdat.figrsizeydobskin = [8., 2.5]
    boolpost = False
    if boolpost:
        gdat.figrsize /= 1.5
        
    if gdat.offstextatmoraditmpt is None:
        gdat.offstextatmoraditmpt = [[0.3, -0.5], [0.3, -0.5], [0.3, -0.5], [0.3, 0.5]]
    if gdat.offstextatmoradimetr is None:
        gdat.offstextatmoradimetr = [[0.3, -0.5], [0.3, -0.5], [0.3, -0.5], [0.3, 0.5]]
    
    # conversion factors
    gdat.factrsrj, gdat.factmsmj, gdat.factrjre, gdat.factmjme, gdat.factaurj = tesstarg.util.retr_factconv()
    
    gdat.liststrgpopl = ['exarcomp']
    if gdat.boolexof:
        gdat.liststrgpopl += ['exof']
    gdat.numbpopl = len(gdat.liststrgpopl)
    
    gdat.listfeatstar = ['radistar', 'massstar', 'tmptstar', 'rascstar', 'declstar', 'vsiistar', 'jmagstar']
   
    if gdat.boolobjt:
        dictexartarg = retr_exarcomp(strgexar=gdat.strgexar)
        gdat.boolexar = gdat.strgexar is not None and dictexartarg is not None
    
        if gdat.boolexar:
            print('The planet name was found in the NASA Exoplanet Archive "composite" table.')
            # stellar properties
            
            if gdat.periprio is None:
                gdat.periprio = dictexartarg['peri']
            
        else:
            print('The planet name was *not* found in the Exoplanet Archive "composite" table.')
    
        # read the NASA Exoplanet Archive planets
        path = gdat.pathbase + 'data/PS_2020.11.18_20.29.19.csv'
        print('Reading %s...' % path)
        objtexarplan = pd.read_csv(path, skiprows=302, low_memory=False)
        indx = np.where((objtexarplan['hostname'].values == gdat.strgexar) & (objtexarplan['default_flag'].values == 1))[0]
        print('gdat.strgexar')
        print(gdat.strgexar)

        if indx.size == 0:
            print('The planet name was *not* found in the NASA Exoplanet Archive "planets" table.')
            gdat.boolexar = False
        else:
            gdat.deptprio = objtexarplan['pl_trandep'][indx].values * 1e-2
            print('gdat.deptprio')
            summgene(gdat.deptprio)
            if gdat.cosiprio is None:
                gdat.cosiprio = np.cos(objtexarplan['pl_orbincl'][indx].values / 180. * np.pi)
            if gdat.epocprio is None:
                gdat.epocprio = objtexarplan['pl_tranmid'][indx].values # [days]
            gdat.duraprio = objtexarplan['pl_trandur'][indx].values / 24. # [days]
        
        if gdat.ticitarg is None and gdat.strgmast is None:
            raise Exception('')

        # set ticitarg
        if gdat.ticitarg is None and gdat.toiitarg is not None:
            if gdat.typeprioplan == 'exof':
                gdat.ticitarg = objtexof['TIC ID'].values[indx[0]]
            
            if gdat.typepriostar == 'tici':
                indx = []
                for k, strg in enumerate(objtexof['TOI']):
                    if str(strg).split('.')[0] == gdat.strgtoiibase:
                        indx.append(k)
                indx = np.array(indx)
                if indx.size == 0:
                    print('Did not find the TOI in the ExoFOP-TESS TOI list.')
                    raise Exception('')
            
                gdat.ticitarg = objtexof['TIC ID'].values[indx[0]]


            if gdat.strgmast is not None:
                print('ticitarg was not provided. Using the strgmast to find ticitarg of the closest match...')
                catalogData = astroquery.mast.Catalogs.query_object(gdat.strgmast, catalog='TIC', radius='40s')
                gdat.ticitarg = '%d' % catalogData[0]['ID']
        print('gdat.ticitarg')
        print(gdat.ticitarg)
            
        if gdat.strgmast is None:
            print('gdat.strgmast was not provided as input. Using the TIC ID to construct gdat.strgmast.')
            gdat.strgmast = 'TIC %d' % gdat.ticitarg
        print('gdat.strgmast')
        print(gdat.strgmast)
    
        if gdat.typeprioplan == 'exof':
            print('A TOI number is provided. Retreiving the TCE attributes from ExoFOP-TESS...')
            
            # find the indices of the target in the TOI catalog
            objtexof = pd.read_csv(gdat.pathtoii, skiprows=0)
            
            if gdat.epocprio is None:
                gdat.epocprio = objtexof['Epoch (BJD)'].values[indx]
            if gdat.periprio is None:
                gdat.periprio = objtexof['Period (days)'].values[indx]
            gdat.deptprio = objtexof['Depth (ppm)'].values[indx] * 1e-6
            gdat.duraprio = objtexof['Duration (hours)'].values[indx] / 24. # [days]
            if gdat.cosiprio is None:
                gdat.cosiprio = np.zeros_like(gdat.epocprio)
            
        if gdat.typeprioplan == 'inpt':
            if gdat.rratprio is None:
                gdat.rratprio = 0.1 + np.zeros_like(gdat.epocprio)
            if gdat.rsmaprio is None:
                gdat.rsmaprio = 0.2 * gdat.periprio**(-2. / 3.)
            print('gdat.cosiprio')
            print(gdat.cosiprio)

            if gdat.cosiprio is None:
                gdat.cosiprio = np.zeros_like(gdat.epocprio)
            gdat.duraprio = tesstarg.util.retr_dura(gdat.periprio, gdat.rsmaprio, gdat.cosiprio)
            gdat.deptprio = gdat.rratprio**2
        
        # check MAST
        if gdat.strgmast is None:
            gdat.strgmast = gdat.labltarg

        print('gdat.strgmast')
        print(gdat.strgmast)
        
        if not gdat.boolforcoffl and gdat.strgmast is not None:
            catalogData = astroquery.mast.Catalogs.query_object(gdat.strgmast, catalog='TIC', radius='40s')
            if catalogData[0]['dstArcSec'] > 0.1:
                print('The nearest source is more than 0.1 arcsec away from the target!')
            print('Found the target on MAST!')
            gdat.rascstar = catalogData[0]['ra']
            gdat.declstar = catalogData[0]['dec']
            gdat.stdvrascstar = 0.
            gdat.stdvdeclstar = 0.
            if gdat.radistar is None:
                print('Setting the stellar radius from the TIC.')
                gdat.radistar = catalogData[0]['rad'] * gdat.factrsrj
                gdat.stdvradistar = catalogData[0]['e_rad'] * gdat.factrsrj
                if not np.isfinite(gdat.radistar):
                    raise Exception('TIC stellar radius is not finite.')
                if not np.isfinite(gdat.radistar):
                    raise Exception('TIC stellar radius uncertainty is not finite.')
            if gdat.massstar is None:
                print('Setting the stellar mass from the TIC.')
                gdat.massstar = catalogData[0]['mass'] * gdat.factmsmj
                gdat.stdvmassstar = catalogData[0]['e_mass'] * gdat.factmsmj
                if not np.isfinite(gdat.massstar):
                    raise Exception('TIC stellar mass is not finite.')
                if not np.isfinite(gdat.massstar):
                    raise Exception('TIC stellar mass uncertainty is not finite.')
            if gdat.tmptstar is None:
                print('Setting the stellar temperature from the TIC.')
                gdat.tmptstar = catalogData[0]['Teff']
                gdat.stdvtmptstar = catalogData[0]['e_Teff']
                if not np.isfinite(gdat.tmptstar):
                    raise Exception('TIC stellar temperature is not finite.')
                if not np.isfinite(gdat.tmptstar):
                    raise Exception('TIC stellar temperature uncertainty is not finite.')
            gdat.jmagstar = catalogData[0]['Jmag']
            gdat.hmagstar = catalogData[0]['Hmag']
            gdat.kmagstar = catalogData[0]['Kmag']
            gdat.vmagstar = catalogData[0]['Vmag']
            
            # check that the closest TIC to a given TIC is itself
            if gdat.ticitarg is not None:
                strgtici = '%s' % catalogData[0]['ID']
                if strgtici != str(gdat.ticitarg):
                    print('strgtici')
                    print(strgtici)
                    print('gdat.ticitarg')
                    print(gdat.ticitarg)
                    raise Exception('')
    if gdat.booldatatser and (gdat.boolprioalle or gdat.boolexecalle):
        gdat.pathallebase = gdat.pathobjt + 'allesfits/'
    gdat.pathdata = gdat.pathobjt + 'data/'
    gdat.pathimag = gdat.pathobjt + 'imag/'
    
    gdat.liststrgpdfn = [gdat.strgprio] + [gdat.strgprio + typeallemodl for typeallemodl in gdat.listtypeallemodl]
    
    gdat.pathimagprop = gdat.pathimag + 'prop/'
    
    for strgpdfn in gdat.liststrgpdfn:
        pathimagpdfn = gdat.pathimagprop + strgpdfn + '/'
        setattr(gdat, 'pathimagprop' + strgpdfn, pathimagpdfn)
        setattr(gdat, 'pathimagproptext' + strgpdfn, pathimagpdfn + 'text/')
        setattr(gdat, 'pathimagpropnotx' + strgpdfn, pathimagpdfn + 'notx/')
        setattr(gdat, 'pathimagmagt' + strgpdfn, pathimagpdfn + 'magt/')
        setattr(gdat, 'pathimagheke' + strgpdfn, pathimagpdfn + 'heke/')
    
    os.system('mkdir -p %s' % gdat.pathdata)
    for attr, valu in gdat.__dict__.items():
        if attr.startswith('pathimag'):
            os.system('mkdir -p %s' % valu)
    
    gdat.arrytser = dict()
    gdat.listarrytser = dict()
    
    print('gdat.typeprioplan')
    print(gdat.typeprioplan)
    print('gdat.booldatatser')
    print(gdat.booldatatser)

    if gdat.boolobjt and gdat.booldatatser:
        
        gdat.pathalle = dict()
        gdat.objtalle = dict()
        
        print('gdat.numbchun')
        print(gdat.numbchun)
        print('gdat.indxchun')
        print(gdat.indxchun)
        
        gdat.arrytser['raww'] = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.arrytser['bdtr'] = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.listarrytser['raww'] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.listarrytser['bdtr'] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        
        # load TESS data
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                if b == 0 and gdat.liststrginst[b][p] == 'TESS':
                    gdat.arrytser['raww'][b][p] = arrylcurtess
                    for y in gdat.indxchun[b][p]:
                        gdat.listarrytser['raww'][b][p] = listarrylcurtess
        
        # load input data
        if gdat.listpathdatainpt is not None:
            for b in gdat.indxdatatser:
                for p in gdat.indxinst[b]:
                    for y in gdat.indxchun[b][p]:
                        arry = np.loadtxt(gdat.listpathdatainpt[b][p][y], delimiter=',', skiprows=1)
                        gdat.listarrytser['raww'][b][p][y] = np.empty((arry.shape[0], 3))
                        gdat.listarrytser['raww'][b][p][y][:, 0:2] = arry[:, 0:2]
                        gdat.listarrytser['raww'][b][p][y][:, 2] = 1e-4 * arry[:, 1]
                        indx = np.argsort(gdat.listarrytser['raww'][b][p][y][:, 0])
                        gdat.listarrytser['raww'][b][p][y] = gdat.listarrytser['raww'][b][p][y][indx, :]
                        indx = np.where(gdat.listarrytser['raww'][b][p][y][:, 1] < 1e6)[0]
                        gdat.listarrytser['raww'][b][p][y] = gdat.listarrytser['raww'][b][p][y][indx, :]
                        gdat.listisec = None
                    gdat.arrytser['raww'][b][p] = np.concatenate(gdat.listarrytser['raww'][b][p])

        plot_tser(gdat, 'raww')
        
        if gdat.typeprioplan == 'tlsq':

            # temp
            for b in gdat.indxdatatser:
                if b == 0:
                    for p in gdat.indxinst[b]:
                        strgextn = '%s_%s' % (gdat.liststrginst[b][p], gdat.strgtarg)
                        epocmask = np.array([])
                        perimask = np.array([])
                        duramask = np.array([])
                        bdtr_wrap(gdat, epocmask, perimask, duramask, indxdatatser=b)
                        dicttlsq = tesstarg.util.exec_tlsq(gdat.arrytser['bdtr'][b][p], gdat.pathimag, maxmnumbplantlsq=gdat.maxmnumbplantlsq, \
                                                        strgextn=strgextn, \
                                                        strgplotextn=gdat.typefileplot, figrsize=gdat.figrsizeydob, figrsizeydobskin=gdat.figrsizeydobskin, \
                                                        alphraww=gdat.alphraww, \
                                                        )
                        if gdat.epocprio is None:
                            gdat.epocprio = dicttlsq['epoc']
                        if gdat.periprio is None:
                            gdat.periprio = dicttlsq['peri']
                        gdat.deptprio = 1. - dicttlsq['dept']
                        gdat.duraprio = dicttlsq['dura']
                        gdat.cosiprio = np.zeros_like(dicttlsq['epoc']) 
                        gdat.rratprio = np.sqrt(gdat.deptprio)
                        gdat.rsmaprio = np.sin(np.pi * gdat.duraprio / gdat.periprio)

    if gdat.boolobjt:
        # planet priors
        gdat.numbplan = gdat.epocprio.size
        gdat.indxplan = np.arange(gdat.numbplan)
        
        if gdat.indxplanalle is None:
            gdat.indxplanalle = gdat.indxplan
        print('gdat.numbplan')
        print(gdat.numbplan)
        
        if gdat.liststrgplan is None:
            gdat.liststrgplan = retr_liststrgplan(gdat.numbplan)
        if gdat.listcolrplan is None:
            gdat.listcolrplan = retr_listcolrplan(gdat.numbplan)
        print('Planet letters: ')
        print(gdat.liststrgplan)

    if gdat.boolobjt:
        
        if gdat.duraprio is None:
            gdat.duraprio = tesstarg.util.retr_dura(gdat.periprio, gdat.rsmaprio, gdat.cosiprio)
        
        if gdat.rratprio is None:
            gdat.rratprio = np.sqrt(gdat.deptprio)
        if gdat.rsmaprio is None:
            gdat.rsmaprio = np.sqrt(np.sin(np.pi * gdat.duraprio / gdat.periprio)**2 + gdat.cosiprio**2)
        if gdat.ecosprio is None:
            gdat.ecosprio = np.zeros(gdat.numbplan)
        if gdat.esinprio is None:
            gdat.esinprio = np.zeros(gdat.numbplan)
        if gdat.rvsaprio is None:
            gdat.rvsaprio = np.zeros(gdat.numbplan)
        
        if gdat.stdvrratprio is None:
            gdat.stdvrratprio = 0.01 + np.zeros(gdat.numbplan)
        if gdat.stdvrsmaprio is None:
            gdat.stdvrsmaprio = 0.01 + np.zeros(gdat.numbplan)
        if gdat.stdvepocprio is None:
            gdat.stdvepocprio = 0.1 + np.zeros(gdat.numbplan)
        if gdat.stdvperiprio is None:
            gdat.stdvperiprio = 0.01 + np.zeros(gdat.numbplan)
        if gdat.stdvcosiprio is None:
            gdat.stdvcosiprio = 0.05 + np.zeros(gdat.numbplan)
        if gdat.stdvecosprio is None:
            gdat.stdvecosprio = 0.1 + np.zeros(gdat.numbplan)
        if gdat.stdvesinprio is None:
            gdat.stdvesinprio = 0.1 + np.zeros(gdat.numbplan)
        if gdat.stdvrvsaprio is None:
            gdat.stdvrvsaprio = 0.001 + np.zeros(gdat.numbplan)
        
        # others
        if gdat.projoblqprio is None:
            gdat.projoblqprio = 0. + np.zeros(gdat.numbplan)
        if gdat.stdvprojoblqprio is None:
            gdat.stdvprojoblqprio = 10. + np.zeros(gdat.numbplan)
        
        # order planets with respect to period
        if gdat.typeprioplan != 'inpt':
            print('Sorting the planets with respect to orbital period...')
            indxplansort = np.argsort(gdat.periprio)
            gdat.rratprio = gdat.rratprio[indxplansort]
            gdat.rsmaprio = gdat.rsmaprio[indxplansort]
            gdat.epocprio = gdat.epocprio[indxplansort]
            gdat.periprio = gdat.periprio[indxplansort]
            gdat.cosiprio = gdat.cosiprio[indxplansort]
            gdat.ecosprio = gdat.ecosprio[indxplansort]
            gdat.esinprio = gdat.esinprio[indxplansort]
            gdat.rvsaprio = gdat.rvsaprio[indxplansort]
        
            gdat.duraprio = gdat.duraprio[indxplansort]
        
        for featstar in gdat.listfeatstar:
            if not hasattr(gdat, 'stdv' + featstar):
                setattr(gdat, 'stdv' + featstar, 0.)
            if not hasattr(gdat, featstar):
                if featstar == 'radistar':
                    setattr(gdat, featstar, gdat.factrsrj)
                if featstar == 'massstar':
                    setattr(gdat, featstar, gdat.factmsmj)
                if featstar == 'tmptstar':
                    setattr(gdat, featstar, 5778.)
                if featstar == 'vsiistar':
                    setattr(gdat, featstar, 1e3)
                print('Setting a property from Solar default!')

        print('Stellar priors:')
        print('gdat.rascstar')
        print(gdat.rascstar)
        print('gdat.declstar')
        print(gdat.declstar)
        print('gdat.radistar')
        print(gdat.radistar)
        print('gdat.stdvradistar')
        print(gdat.stdvradistar)
        print('gdat.radistar [R_S]')
        print(gdat.radistar / gdat.factrsrj)
        print('gdat.stdvradistar [R_S]')
        print(gdat.stdvradistar / gdat.factrsrj)
        print('gdat.massstar')
        print(gdat.massstar)
        print('gdat.stdvmassstar')
        print(gdat.stdvmassstar)
        print('gdat.vsiistar')
        print(gdat.vsiistar)
        print('gdat.stdvvsiistar')
        print(gdat.stdvvsiistar)
        print('gdat.massstar [M_S]')
        print(gdat.massstar / gdat.factmsmj)
        print('gdat.stdvmassstar [M_S]')
        print(gdat.stdvmassstar / gdat.factmsmj)
        print('gdat.tmptstar')
        print(gdat.tmptstar)
        print('gdat.stdvtmptstar')
        print(gdat.stdvtmptstar)
        
        print('Planetary priors:')
        print('gdat.rratprio')
        print(gdat.rratprio)
        print('gdat.rsmaprio')
        print(gdat.rsmaprio)
        print('gdat.epocprio')
        print(gdat.epocprio)
        print('gdat.periprio')
        print(gdat.periprio)
        print('gdat.cosiprio')
        print(gdat.cosiprio)
        print('gdat.ecosprio')
        print(gdat.ecosprio)
        print('gdat.esinprio')
        print(gdat.esinprio)
        print('gdat.rvsaprio')
        print(gdat.rvsaprio)
        print('gdat.stdvrratprio')
        print(gdat.stdvrratprio)
        print('gdat.stdvrsmaprio')
        print(gdat.stdvrsmaprio)
        print('gdat.stdvepocprio')
        print(gdat.stdvepocprio)
        print('gdat.stdvperiprio')
        print(gdat.stdvperiprio)
        print('gdat.stdvcosiprio')
        print(gdat.stdvcosiprio)
        print('gdat.stdvecosprio')
        print(gdat.stdvecosprio)
        print('gdat.stdvesinprio')
        print(gdat.stdvesinprio)
        print('gdat.stdvrvsaprio')
        print(gdat.stdvrvsaprio)
        #print('gdat.deptprio')
        #print(gdat.deptprio)
        #print('gdat.duraprio')
        #print(gdat.duraprio)
        
        if not np.isfinite(gdat.rratprio).all():
            print('rrat is infinite!')
        if not np.isfinite(gdat.rsmaprio).all():
            print('rsma is infinite!')
        if not np.isfinite(gdat.epocprio).all():
            print('epoc is infinite!')
        if not np.isfinite(gdat.periprio).all():
            print('peri is infinite!')
        if not np.isfinite(gdat.cosiprio).all():
            print('cosi is infinite!')
        if not np.isfinite(gdat.ecosprio).all():
            print('ecos is infinite!')
        if not np.isfinite(gdat.esinprio).all():
            print('esin is infinite!')
        if not np.isfinite(gdat.rvsaprio).all():
            print('evsa is infinite!')

    if gdat.boolobjt and gdat.booldatatser:
        # determine the baseline-detrend mask
        if gdat.duraprio is not None:
            gdat.epocmask = gdat.epocprio
            gdat.perimask = gdat.periprio
            gdat.duramask = 2. * gdat.duraprio
        else:
            print('Did not find any transit-like features in the light curve...')
            return
            gdat.epocmask = None
            gdat.perimask = None
            gdat.duramask = None
        print('gdat.epocmask')
        print(gdat.epocmask)
        print('gdat.perimask')
        print(gdat.perimask)
        print('gdat.duramask')
        print(gdat.duramask)
        
    if gdat.boolobjt and gdat.booldatatser:
        # baseline-detrend with the ephemeris prior
        if gdat.numbinst[0] > 0:
            bdtr_wrap(gdat, gdat.epocmask, gdat.perimask, gdat.duramask, indxdatatser=0)
        
            # sigma-clip the light curve
            # temp -- this does not work properly!
            for b in gdat.indxdatatser:
                for p in gdat.indxinst[b]:
                    for y in gdat.indxchun[b][p]:
                        if gdat.boolclip:
                            lcurclip, lcurcliplowr, lcurclipuppr = scipy.stats.sigmaclip(gdat.listarrytser['bdtr'][b][p][y][:, 1], low=5., high=5.)
                            print('Clipping the light curve at %g and %g...' % (lcurcliplowr, lcurclipuppr))
                            indx = np.where((gdat.listarrytser['bdtr'][b][p][y][:, 1] < lcurclipuppr) & \
                                                                        (gdat.listarrytser['bdtr'][b][p][y][:, 1] > lcurcliplowr))[0]
                            gdat.listarrytser['bdtr'][b][p][y] = gdat.listarrytser['bdtr'][b][p][y][indx, :]
                
            # write baseline-detrended light curve
            for b in gdat.indxdatatser:
                for p in gdat.indxinst[b]:
                    path = gdat.pathdata + 'arrytserbdtr%s.csv' % (gdat.liststrginst[b][p])
                    print('Writing to %s...' % path)
                    np.savetxt(path, gdat.arrytser['bdtr'][b][p], delimiter=',', header='time,flux,flux_err')
        
        # make mock PFS data for RM
        if gdat.listpathdatainpt is None:
            boolpfss = False
            for b in gdat.indxdatatser:
                for p in gdat.indxinst[b]:
                    if b == 1 and gdat.liststrginst[b][p] == 'PFS' and gdat.listdatatype[b][p] == 'mock':
                        boolpfss = True
                        # trick allesfitter by a dummy data set
                        arry = np.zeros((2, 3))
                        arry[0, 0] = 2459000
                        arry[1, 0] = 2459500
                        arry[0, 1] = 1e-4
                        arry[1, 1] = 1e-4
            
            for b in gdat.indxdatatser:
                for p in gdat.indxinst[b]:
                    if b == 1 and gdat.liststrginst[b][p] == 'PFS' and gdat.listdatatype[b][p] == 'mock':

                        objtalle = allesfitter.allesclass(gdat.pathalle['pfss'])

                        xx = np.arange(2459322.5675 - 2. / 24., 2459322.7248 + 2. / 24., 10. / 60. / 24.)
                        arry = np.zeros((xx.size, 3))
                        arry[:, 0] = xx
                        tserrvel = objtalle.get_initial_guess_model(gdat.liststrginst[b][p], 'rv', xx=xx)
                        stdv = 0.9e-3 * 1.4
                        tserrvel += np.random.randn(tserrvel.size) * stdv
                        arry[:, 1] = tserrvel
                        arry[:, 2] = stdv
                        gdat.listarrytser['raww'][b][p] = [arry]
                        gdat.arrytser['raww'][b][p] = arry
        
        #bdtr_wrap(gdat, gdat.epocmask, gdat.perimask, gdat.duramask, indxdatatser=1)
        gdat.arrytser['bdtr'][1] = gdat.arrytser['raww'][1]
        gdat.listarrytser['bdtr'][1] = gdat.listarrytser['raww'][1]
        
        gdat.arrytsertotl = [[] for b in gdat.indxdatatser]
        for b in gdat.indxdatatser:
            if gdat.numbinst[b] > 0:
                gdat.arrytsertotl[b] = np.concatenate(gdat.arrytser['raww'][b], axis=0)
        
        # plot LS periodogram
        for b in gdat.indxdatatser:
            if b == 1 and gdat.numbinst[b] > 0:
                strgextn = '%s' % (gdat.liststrgtser[b])
                listperilspe = tesstarg.util.plot_lspe(gdat.pathimag, gdat.arrytsertotl[b], strgextn=strgextn)
                for p in gdat.indxinst[b]:
                    strgextn = '%s_%s' % (gdat.liststrgtser[b], gdat.liststrginst[b][p]) 
                    listperilspe = tesstarg.util.plot_lspe(gdat.pathimag, gdat.arrytser['raww'][b][p], strgextn=strgextn)
        
        # determine transit masks
        gdat.listindxtimeoutt = [[[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser] for j in gdat.indxplan]
        gdat.listindxtimetran = [[[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser] for j in gdat.indxplan]
        gdat.listindxtimetranchun = [[[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser] for j in gdat.indxplan]
        gdat.listindxtimeclen = [[[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser] for j in gdat.indxplan]
        gdat.numbtimeclen = [[np.empty((gdat.numbplan), dtype=int) for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                for j in gdat.indxplan:
                    # determine time mask
                    if gdat.booldiagmode:
                        if not np.isfinite(gdat.duramask[j]):
                            raise Exception('')
                    for y in gdat.indxchun[b][p]:
                        gdat.listindxtimetranchun[j][b][p][y] = tesstarg.util.retr_indxtimetran(gdat.listarrytser['raww'][b][p][y][:, 0], gdat.epocprio[j], \
                                                                                                                    gdat.periprio[j], gdat.duramask[j])
                    
                    gdat.listindxtimetran[j][b][p] = tesstarg.util.retr_indxtimetran(gdat.arrytser['bdtr'][b][p][:, 0], \
                                                                                                    gdat.epocprio[j], gdat.periprio[j], gdat.duramask[j])
                    gdat.listindxtimeoutt[j][b][p] = np.setdiff1d(np.arange(gdat.arrytser['bdtr'][b][p].shape[0]), gdat.listindxtimetran[j][b][p])
            
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                for j in gdat.indxplan:
                    # clean times for each planet
                    listindxtimetemp = []
                    for jj in gdat.indxplan:
                        if jj != j:
                            listindxtimetemp.append(gdat.listindxtimetran[jj][b][p])
                    if len(listindxtimetemp) > 0:
                        listindxtimetemp = np.concatenate(listindxtimetemp)
                        listindxtimetemp = np.unique(listindxtimetemp)
                    else:
                        listindxtimetemp = np.array([])
                    gdat.listindxtimeclen[j][b][p] = np.setdiff1d(np.arange(gdat.arrytser['bdtr'][b][p].shape[0]), listindxtimetemp)
                    gdat.numbtimeclen[b][p][j] = gdat.listindxtimeclen[j][b][p].size
                if not np.isfinite(gdat.arrytser['bdtr'][b][p]).all():
                    raise Exception('')
                
                #for y in gdat.indxchun[b][p]:
                #    for i in gdat.indxsplnregi[b][p][y]:
                #        if gdat.listobjtspln[b][p][y] is not None and gdat.listobjtspln[b][p][y][i] is not None:
                #            # produce a table for the spline coefficients
                #            fileoutp = open(gdat.pathdata + 'coefbdtr.csv', 'w')
                #            fileoutp.write(' & ')
                #            for i in gdat.indxsplnregi[b][p][y]:
                #                print('$\beta$:', gdat.listobjtspln[b][p][y][i].get_coeffs())
                #                print('$t_k$:', gdat.listobjtspln[b][p][y][i].get_knots())
                #                print
                #            fileoutp.write('\\hline\n')
                #            fileoutp.close()

        plot_tser(gdat, 'bdtr')

        gdat.time = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.indxtime = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.numbtime = [np.empty(gdat.numbinst[b], dtype=int) for b in gdat.indxdatatser]
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                gdat.time[b][p] = gdat.arrytser['bdtr'][b][p][:, 0]
                gdat.numbtime[b][p] = gdat.time[b][p].size
                gdat.indxtime[b][p] = np.arange(gdat.numbtime[b][p])
    
        if gdat.booldatatser:
            if gdat.listindxchuninst is None:
                gdat.listindxchuninst = [gdat.indxchun]
    
    if gdat.boolobjt:
        if gdat.booldatatser:
            # plot raw data
            if 'TESS' in gdat.liststrginst[0] and (gdat.typedataspoc == 'PDC' or gdat.typedataspoc == 'SAP'):
                for b in gdat.indxdatatser:
                    for p in gdat.indxinst[b]:
                        if gdat.liststrginst[b][p] != 'TESS':
                            continue
                        for y in gdat.indxchun[0]:
                            path = gdat.pathdata + gdat.liststrgchun[b][p][y] + '_SAP.csv'
                            print('Writing to %s...' % path)
                            np.savetxt(path, gdat.arrytsersapp[0], delimiter=',', header='time,flux,flux_err')
                            path = gdat.pathdata + gdat.liststrgchun[b][p][y] + '_PDCSAP.csv'
                            print('Writing to %s...' % path)
                            np.savetxt(path, gdat.arrytserpdcc[0], delimiter=',', header='time,flux,flux_err')
                
                # plot PDCSAP and SAP light curves
                figr, axis = plt.subplots(2, 1, figsize=gdat.figrsizeydob)
                axis[0].plot(gdat.arrytsersapp[0][:, 0] - gdat.timetess, gdat.arrytsersapp[0][:, 1], color='k', marker='.', ls='', ms=1)
                if gdat.listlimttimemask is not None:
                    axis[0].plot(gdat.arrytsersapp[0][listindxtimegood, 0] - gdat.timetess, \
                                                    gdat.arrytsersapp[0][listindxtimegood, 1], color='k', marker='.', ls='', ms=1)
                axis[1].plot(gdat.arrytserpdcc[0][:, 0] - gdat.timetess, gdat.arrytserpdcc[0][:, 1], color='k', marker='.', ls='', ms=1)
                if gdat.listlimttimemask is not None:
                    axis[1].plot(gdat.arrytserpdcc[0][listindxtimegood, 0] - gdat.timetess, \
                                                    gdat.arrytserpdcc[0][listindxtimegood, 1], color='k', marker='.', ls='', ms=1)
                #axis[0].text(.97, .97, 'SAP', transform=axis[0].transAxes, size=20, color='r', ha='right', va='top')
                #axis[1].text(.97, .97, 'PDC', transform=axis[1].transAxes, size=20, color='r', ha='right', va='top')
                axis[1].set_xlabel('Time [BJD - %d]' % gdat.timetess)
                for a in range(2):
                    axis[a].set_ylabel('Relative Flux')
                
                #for j in gdat.indxplan:
                #    colr = gdat.listcolrplan[j]
                #    axis[1].plot(gdat.arrytserpdcc[0][gdat.listindxtimetran[j], 0] - gdat.timetess, gdat.arrytserpdcc[0][gdat.listindxtimetran[j], 1], \
                #                                                                                                         color=colr, marker='.', ls='', ms=1)
                plt.subplots_adjust(hspace=0.)
                path = gdat.pathimag + 'lcurspoc_%s.%s' % (gdat.strgtarg, gdat.typefileplot)
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()
            
            # injection recovery test
        
    # grab all exoplanet properties
    ## ExoFOP
    if gdat.boolexof:
        gdat.dictexof = retr_exof(gdat)
    
    #print('dictexof[toii][np.where(dictexof[toii] % 1. > 0.02)[0]]')
    #print(gdat.dictexof['toii'][np.where(gdat.dictexof['toii'] % 1. > 0.02)[0]])
    #raise Exception('')
    
    # ellipsoidal variations
    ## limb and gravity darkening coefficients from Claret2017
    u = 0.4
    g = 0.2
    gdat.alphelli = 0.15 * (15 + u) * (1 + g) / (3 - u)
    ### DB
    gdat.binswlenbeam = np.linspace(0.6, 1., 101)
    gdat.meanwlenbeam = (gdat.binswlenbeam[1:] + gdat.binswlenbeam[:-1]) / 2.
    gdat.diffwlenbeam = (gdat.binswlenbeam[1:] - gdat.binswlenbeam[:-1]) / 2.
    x = 2.248 / gdat.meanwlenbeam
    gdat.funcpcurmodu = .25 * x * np.exp(x) / (np.exp(x) - 1.)
    
    ## NASA Exoplanet Archive
    gdat.dictexarcomp = retr_exarcomp()
    numbplanexar = gdat.dictexarcomp['radiplan'].size
    gdat.indxplanexar = np.arange(numbplanexar)
    ### augment the catalog
    gdat.dictexarcomp['vesc'] = tesstarg.util.retr_vesc(gdat.dictexarcomp['massplan'], gdat.dictexarcomp['radiplan'])
    
    if gdat.boolobjt:
        #if ''.join(gdat.liststrgplan) != ''.join(sorted(gdat.liststrgplan)):
        #    print('Provided planet letters are not in order. Changing the TCE order to respect the letter order in plots (b, c, d, e)...')
        #    gdat.indxplan = np.argsort(np.array(gdat.liststrgplan))
        print('gdat.indxplan') 
        print(gdat.indxplan)

        gdat.liststrgplanfull = np.empty(gdat.numbplan, dtype='object')
        print('gdat.liststrgplan')
        print(gdat.liststrgplan)
        for j in gdat.indxplan:
            gdat.liststrgplanfull[j] = gdat.labltarg + ' ' + gdat.liststrgplan[j]

        ## augment object dictinary
        gdat.dictfeatobjt = dict()
        gdat.dictfeatobjt['namestar'] = np.array([gdat.labltarg] * gdat.numbplan)
        gdat.dictfeatobjt['nameplan'] = gdat.liststrgplanfull
        # temp
        gdat.dictfeatobjt['booltran'] = np.array([True] * gdat.numbplan, dtype=bool)
        gdat.dictfeatobjt['boolfrst'] = np.array([True] + [False] * (gdat.numbplan - 1), dtype=bool)
        gdat.dictfeatobjt['vmagstar'] = np.zeros(gdat.numbplan) + gdat.vmagstar
        gdat.dictfeatobjt['jmagstar'] = np.zeros(gdat.numbplan) + gdat.jmagstar
        gdat.dictfeatobjt['hmagstar'] = np.zeros(gdat.numbplan) + gdat.hmagstar
        gdat.dictfeatobjt['kmagstar'] = np.zeros(gdat.numbplan) + gdat.kmagstar
        gdat.dictfeatobjt['numbplanstar'] = np.zeros(gdat.numbplan) + gdat.numbplan
        gdat.dictfeatobjt['numbplantranstar'] = np.zeros(gdat.numbplan) + gdat.numbplan
        
        if gdat.booldatatser:
            if gdat.dilu == 'lygos':
                print('Calculating the contamination ratio...')
                gdat.contrati = lygos.calc_contrati()

                
        if gdat.booldatatser:
            for b in gdat.indxdatatser:
                for p in gdat.indxinst[b]:
                    for y in gdat.indxchun[b][p]:
                        if gdat.listlimttimemask is not None:
                            # mask the data
                            print('Masking the data...')
                            numbmask = gdat.listlimttimemask.shape[0]
                            listindxtimemask = []
                            for k in range(numbmask):
                                indxtimemask = np.where((gdat.listarrytser['bdtr'][b][p][y][:, 0] < gdat.listlimttimemask[k, 1]) & \
                                                            (gdat.listarrytser['bdtr'][b][p][y][:, 0] > gdat.listlimttimemask[k, 0]))[0]
                                listindxtimemask.append(indxtimemask)
                            listindxtimemask = np.concatenate(listindxtimemask)
                            listindxtimegood = np.setdiff1d(gdat.indxtime, listindxtimemask)
                            gdat.listarrytser['bdtr'][b][p][y] = gdat.listarrytser['bdtr'][b][p][y][listindxtimegood, :]
            
            # correct for dilution
            #print('Correcting for dilution!')
            #if gdat.dilucorr is not None:
            #    gdat.arrytserdilu = np.copy(gdat.listarrytser['bdtr'][b][p][y])
            #if gdat.dilucorr is not None:
            #    gdat.arrytserdilu[:, 1] = 1. - gdat.dilucorr * (1. - gdat.listarrytser['bdtr'][b][p][y][:, 1])
            #gdat.arrytserdilu[:, 1] = 1. - gdat.contrati * gdat.contrati * (1. - gdat.listarrytser['bdtr'][b][p][y][:, 1])
            #if gdat.dilucorr is not NoneL
            #    gdat.listarrytser['bdtr'][b][p][y] = np.copy(gdat.arrytserdilu) 
            #    figr, axis = plt.subplots(1, 1, figsize=gdat.figrsizeydob)
            #    axis.plot(gdat.arrytserdilu[:, 0] - gdat.timetess, gdat.arrytserdilu[:, 1], color='grey', marker='.', ls='', ms=1)
            #    for j in gdat.indxplan:
            #        colr = gdat.listcolrplan[j]
            #        axis.plot(gdat.arrytserdilu[gdat.listindxtimetran[j], 0] - gdat.timetess, gdat.arrytserdilu[gdat.listindxtimetran[j], 1], \
            #                                                                                                color=colr, marker='.', ls='', ms=1)
            #    axis.set_xlabel('Time [BJD - %d]' % gdat.timetess)
            #    axis.set_ylabel('Relative Flux')
            #    plt.subplots_adjust(hspace=0.)
            #    path = gdat.pathimag + 'lcurdilu_%s.%s' % (gdat.strgtarg, gdat.typefileplot)
            #    print('Writing to %s...' % path)
            #    plt.savefig(path)
            #    plt.close()
                
            gdat.delttimebind = 1. # [days]
            gdat.delttimebindzoom = 20. / 60. / 24. # [days]
            gdat.numbbinspcurtotl = 300
                
            ## bin the light curve
            for b in gdat.indxdatatser:
                for p in gdat.indxinst[b]:
                    for y in gdat.indxchun[b][p]:
                        delt = gdat.delttimebind
                        lcurbind = tesstarg.util.rebn_tser(gdat.listarrytser['bdtr'][b][p][y], delt=gdat.delttimebind)
                        
                        path = gdat.pathdata + 'arrytserbdtrbind%s%s.csv' % (gdat.liststrginst[b][p], gdat.liststrgchun[b][p][y])
                        print('Writing to %s' % path)
                        np.savetxt(path, lcurbind, delimiter=',', header='time,flux,flux_err')
                    
            ## phase-fold and save the baseline-detrended light curve
            gdat.arrypcurprim = dict()
            gdat.arrypcurprimbindtotl = dict()
            gdat.arrypcurprimbindzoom = dict()
            gdat.arrypcurquad = dict()
            gdat.arrypcurquadbindtotl = dict()
            gdat.arrypcurquadbindzoom = dict()
    
            gdat.arrypcurprim['bdtr'] = [[[[] for j in gdat.indxplan] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
            gdat.arrypcurprimbindtotl['bdtr'] = [[[[] for j in gdat.indxplan] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
            gdat.arrypcurprimbindzoom['bdtr'] = [[[[] for j in gdat.indxplan] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
            gdat.arrypcurquad['bdtr'] = [[[[] for j in gdat.indxplan] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
            gdat.arrypcurquadbindtotl['bdtr'] = [[[[] for j in gdat.indxplan] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
            gdat.arrypcurquadbindzoom['bdtr'] = [[[[] for j in gdat.indxplan] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
            for b in gdat.indxdatatser:
                for p in gdat.indxinst[b]:
                    for j in gdat.indxplan:
                        delt = gdat.delttimebindzoom / gdat.periprio[j]
                        gdat.arrypcurprim['bdtr'][b][p][j] = tesstarg.util.fold_tser(gdat.arrytser['bdtr'][b][p][gdat.listindxtimeclen[j][b][p], :], \
                                                                                                                     gdat.epocprio[j], gdat.periprio[j])
                        gdat.arrypcurprimbindtotl['bdtr'][b][p][j] = tesstarg.util.rebn_tser(gdat.arrypcurprim['bdtr'][b][p][j], numbbins=gdat.numbbinspcurtotl)
                        gdat.arrypcurprimbindzoom['bdtr'][b][p][j] = tesstarg.util.rebn_tser(gdat.arrypcurprim['bdtr'][b][p][j], delt=delt)
                        gdat.arrypcurquad['bdtr'][b][p][j] = tesstarg.util.fold_tser(gdat.arrytser['bdtr'][b][p][gdat.listindxtimeclen[j][b][p], :], \
                                                                                                gdat.epocprio[j], gdat.periprio[j], phasshft=0.25)
                        gdat.arrypcurquadbindtotl['bdtr'][b][p][j] = tesstarg.util.rebn_tser(gdat.arrypcurquad['bdtr'][b][p][j], numbbins=gdat.numbbinspcurtotl)
                        gdat.arrypcurquadbindzoom['bdtr'][b][p][j] = tesstarg.util.rebn_tser(gdat.arrypcurquad['bdtr'][b][p][j], delt=delt)
                        
                        # write (good for Vespa)
                        path = gdat.pathdata + 'arrypcurprimbdtrbind_%s_%s.csv' % (gdat.liststrgplan[j], gdat.liststrginst[b][p])
                        print('Writing to %s...' % path)
                        temp = np.copy(gdat.arrypcurprimbindtotl['bdtr'][b][p][j])
                        temp[:, 0] *= gdat.periprio[j]
                        np.savetxt(path, temp, delimiter=',')
                
        # number of samples to draw from the prior
        gdat.numbsamp = 10000
    
        if gdat.booldatatser:
            plot_pser(gdat, 'bdtr')
        if gdat.boolprocprio:
            for typeallemodl in gdat.listtypeallemodl:
                calc_prop(gdat, 'prio')
                if gdat.boolplotprop:
                    plot_prop(gdat, 'prio')
    
    if not gdat.boolobjt or gdat.numbplan == 0:
        return

    # look for single transits using matched filter
    
    if not gdat.boolexecalle:
        return

    #gdat.boolalleprev = {}
    #for typeallemodl in gdat.listtypeallemodl:
    #    gdat.boolalleprev[typeallemodl] = {}
    #
    #for strgfile in ['params.csv', 'settings.csv', 'params_star.csv']:
    #    
    #    for typeallemodl in gdat.listtypeallemodl:
    #        pathinit = '%sdata/allesfit_templates/%s/%s' % (gdat.pathbase, typeallemodl, strgfile)
    #        pathfinl = '%sallesfits/allesfit_%s/%s' % (gdat.pathobjt, typeallemodl, strgfile)

    #        if not os.path.exists(pathfinl):
    #            cmnd = 'cp %s %s' % (pathinit, pathfinl)
    #            print(cmnd)
    #            os.system(cmnd)
    #            if strgfile == 'params.csv':
    #                gdat.boolalleprev[typeallemodl]['para'] = False
    #            if strgfile == 'settings.csv':
    #                gdat.boolalleprev[typeallemodl]['sett'] = False
    #            if strgfile == 'params_star.csv':
    #                gdat.boolalleprev[typeallemodl]['pars'] = False
    #        else:
    #            if strgfile == 'params.csv':
    #                gdat.boolalleprev[typeallemodl]['para'] = True
    #            if strgfile == 'settings.csv':
    #                gdat.boolalleprev[typeallemodl]['sett'] = True
    #            if strgfile == 'params_star.csv':
    #                gdat.boolalleprev[typeallemodl]['pars'] = True

    if gdat.boolexecalle and gdat.booldatatser:
        if gdat.boolallebkgdgaus:
            # background allesfitter run
            print('Setting up the background allesfitter run...')
            
            if not gdat.boolalleprev['bkgd']['para']:
                writ_filealle(gdat, 'params.csv', gdat.pathallebkgd, gdat.dictdictallepara[typeallemodl], dictalleparadefa)
            
            ## mask out the transits for the background run
            for b in gdat.indxdatatser:
                for p in gdat.indxinst[b]:
                    path = gdat.pathallebkgd + gdat.liststrgchun[b][p][y]  + '.csv'
                    if not os.path.exists(path):
                        indxtimebkgd = np.setdiff1d(gdat.indxtime, np.concatenate(gdat.listindxtimetran))
                        gdat.arrytserbkgd = gdat.listarrytser['bdtr'][b][p][y][indxtimebkgd, :]
                        print('Writing to %s...' % path)
                        np.savetxt(path, gdat.arrytserbkgd, delimiter=',', header='time,%s,%s_err' % (gdat.liststrgtseralle[b], gdat.liststrgtseralle[b]))
                    else:
                        print('OoT light curve available for the background allesfitter run at %s.' % path)
                    
                    #liststrg = list(gdat.objtallebkgd.posterior_params.keys())
                    #for k, strg in enumerate(liststrg):
                    #   post = gdat.objtallebkgd.posterior_params[strg]
                    #   linesplt = '%s' % gdat.objtallebkgd.posterior_params_at_maximum_likelihood[strg][0]
    
    if not gdat.booldatatser:
        return

    for typeallemodl in gdat.listtypeallemodl:
        
        #_0003: single component offset baseline
        #_0004: multiple components, offset baseline
        #_0006: multiple components, GP baseline
        
        # setup the orbit run
        print('Setting up the orbit allesfitter run...')

        proc_alle(gdat, typeallemodl)
        
        gdat.arrytsermodlinit = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                gdat.arrytsermodlinit[b][p] = np.empty((gdat.arrytser['bdtr'][b][p].shape[0], 3))
                gdat.arrytsermodlinit[b][p][:, 0] = gdat.arrytser['bdtr'][b][p][:, 0]
                gdat.arrytsermodlinit[b][p][:, 1] = gdat.objtalle[typeallemodl].get_initial_guess_model(gdat.liststrginst[b][p], 'flux', \
                                                                                                                xx=gdat.arrytser['bdtr'][b][p][:, 0])
                gdat.arrytsermodlinit[b][p][:, 2] = 0.

        if gdat.labltarg == 'WASP-121':
            # get Vivien's GCM model
            path = gdat.pathdata + 'PC-Solar-NEW-OPA-TiO-LR.dat'
            arryvivi = np.loadtxt(path, delimiter=',')
            gdat.phasvivi = (arryvivi[:, 0] / 360. + 0.75) % 1. - 0.25
            gdat.deptvivi = arryvivi[:, 4]
            print('gdat.phasvivi')
            summgene(gdat.phasvivi)
            print('gdat.deptvivi')
            summgene(gdat.deptvivi)
            indxphasvivisort = np.argsort(gdat.phasvivi)
            gdat.phasvivi = gdat.phasvivi[indxphasvivisort]
            gdat.deptvivi = gdat.deptvivi[indxphasvivisort]
            path = gdat.pathdata + 'PC-Solar-NEW-OPA-TiO-LR-AllK.dat'
            arryvivi = np.loadtxt(path, delimiter=',')
            gdat.wlenvivi = arryvivi[:, 1]
            gdat.specvivi = arryvivi[:, 2]
        
            ## TESS throughput 
            gdat.data = np.loadtxt(gdat.pathdata + 'band.csv', delimiter=',', skiprows=9)
            gdat.meanwlenband = gdat.data[:, 0] * 1e-3
            gdat.thptband = gdat.data[:, 1]
    
        ## plot the spherical limits
        #figr, axis = plt.subplots(figsize=gdat.figrsizeydob)
        #
        #gdat.objtalle[typeallemodl] = allesfitter.allesclass(gdat.pathallepcur)
        #gdat.objtalle[typeallemodl].posterior_params_median['b_sbratio_TESS'] = 0
        #gdat.objtalle[typeallemodl].settings['host_shape_TESS'] = 'sphere'
        #gdat.objtalle[typeallemodl].settings['b_shape_TESS'] = 'roche'
        #gdat.objtalle[typeallemodl].posterior_params_median['host_gdc_TESS'] = 0
        #gdat.objtalle[typeallemodl].posterior_params_median['host_bfac_TESS'] = 0
        #lcurmodltemp = gdat.objtalle[typeallemodl].get_posterior_median_model(strgchun, 'flux', xx=gdat.time)
        #axis.plot(gdat.arrypcurquad['bdtr'][b][p][j][:, 0], (gdat.lcurmodlevvv - lcurmodltemp) * 1e6, lw=2, label='Spherical star')
        #
        #gdat.objtalle[typeallemodl] = allesfitter.allesclass(gdat.pathallepcur)
        #gdat.objtalle[typeallemodl].posterior_params_median['b_sbratio_TESS'] = 0
        #gdat.objtalle[typeallemodl].settings['host_shape_TESS'] = 'roche'
        #gdat.objtalle[typeallemodl].settings['b_shape_TESS'] = 'sphere'
        #gdat.objtalle[typeallemodl].posterior_params_median['host_gdc_TESS'] = 0
        #gdat.objtalle[typeallemodl].posterior_params_median['host_bfac_TESS'] = 0
        #lcurmodltemp = gdat.objtalle[typeallemodl].get_posterior_median_model(strgchun, 'flux', xx=gdat.time)
        #axis.plot(gdat.arrypcurquad['bdtr'][b][p][j][:, 0], (gdat.lcurmodlevvv - lcurmodltemp) * 1e6, lw=2, label='Spherical planet')
        #axis.legend()
        #axis.set_ylim([-100, 100])
        #axis.set(xlabel='Phase')
        #axis.set(ylabel='Relative flux [ppm]')
        #plt.subplots_adjust(hspace=0.)
        #path = pathimag + 'pcurquadmodldiff_%s.%s' % (gdat.strgtarg, gdat.typefileplot)
        #plt.savefig(path)
        #plt.close()

