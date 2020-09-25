import allesfitter
import allesfitter.config

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


def evol_file(gdat, namefile, pathalle, lineadde, verbtype=1):
    
    ## read the CSV file
    pathfile = pathalle + namefile
    objtfile = open(pathfile, 'r')
    listline = []
    for line in objtfile:
        listline.append(line)
    objtfile.close()
            
    numbline = len(lineadde)
    indxline = np.arange(numbline)
    
    if verbtype > 1:
        print('numbline')
        print(numbline)

    # delete the lines by leaving out
    listlineneww = []
    for k, line in enumerate(listline):
        linesplt = line.split(',')
        booltemp = True
        if verbtype > 1:
            print('linesplt')
            print(linesplt)
        for m in indxline:
            if len(lineadde[m][0]) > 0:
                if verbtype > 1:
                    print('lineadde[m][0]')
                    print(lineadde[m][0])
                if lineadde[m][0].endswith('*'):
                    if verbtype > 1:
                        print('line[:-1]')
                        print(line[:-1])
                        print('lineadde[m][0][:-1]')
                        print(lineadde[m][0][:-1])
                    if line[:-1].startswith(lineadde[m][0][:-2]):
                        booltemp = False
                else:
                    if line[:-1] == lineadde[m][0]:
                        booltemp = False
                if verbtype > 1:
                    print('booltemp')
                    print(booltemp)
                    print('')
        if booltemp:
            listlineneww.append(line)
        if verbtype > 1:
            print('')
            print('')
    # add the lines
    for m in indxline:
        if lineadde[m][1] != []:
            listlineneww.append(lineadde[m][1])
    
    # rewrite
    pathfile = pathalle + namefile
    print('Writing to %s...' % pathfile)
    objtfile = open(pathfile, 'w')
    for lineneww in listlineneww:
        objtfile.write('%s' % lineneww)
    objtfile.close()


def retr_exof(gdat, strgtoii=None):
    
    path = gdat.pathbase + 'data/TOI_augmented.csv'
    if os.path.exists(path):
        print('Reading from %s...' % path)
        objtexof = pd.read_csv(path, skiprows=0)
        dictexof = objtexof.to_dict('list')
        for attr, varb in dictexof.items():
            dictexof[attr] = np.array(varb)
    else:
    
        pathexof = gdat.pathbase + 'data/exofop_toilists_20200916.csv'
        print('Reading from %s...' % pathexof)
        objtexof = pd.read_csv(pathexof, skiprows=0)
        if strgtoii is None:
            indx = np.arange(objtexof['TOI'].size)
        else:
            indx = np.where(objtexof['TOI'] == strgtarg)[0]
        print('strgtoii')
        print(strgtoii)
            
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
            dictexof['radistar'] = objtexof['Stellar Radius (R_Sun)'][indx].values / gdat.factrsrj
            dictexof['radiplan'] = dictexof['rrat'] * dictexof['radistar']
            
            dictexof['stdvradiplan'] = objtexof['Planet Radius (R_Earth) err'][indx].values * gdat.factrjre
            dictexof['peri'] = objtexof['Period (days)'][indx].values
            dictexof['epoc'] = objtexof['Epoch (BJD)'][indx].values
            
            dictexof['stdvradistar'] = objtexof['Stellar Radius (R_Sun) err'][indx].values / gdat.factrsrj
            dictexof['tmptstar'] = objtexof['Stellar Radius (R_Sun)'][indx].values
            dictexof['stdvtmptstar'] = objtexof['Stellar Radius (R_Sun) err'][indx].values
            dictexof['loggstar'] = objtexof['Stellar log(g) (cm/s^2)'][indx].values
            dictexof['stdvloggstar'] = objtexof['Stellar log(g) (cm/s^2) err'][indx].values
            
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
            dictexof['lumistar'] = dictexof['radistar']**2 * (dictexof['tmptstar'] / 5778.)**4
            dictexof['massstar'] = dictexof['loggstar'] * dictexof['radistar']**2
            dictexof['smax'] = (dictexof['peri']**2)**(1. / 3.) * dictexof['massstar']
            dictexof['smaxasun'] = dictexof['smax'] / gdat.factaurj
            dictexof['inso'] = dictexof['lumistar'] / dictexof['smaxasun']**2
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
            
        df = pd.DataFrame.from_dict(dictexof)#)orient="namestar")
        print('Writing to %s...' % path)
        df.to_csv(path)
    
    return dictexof


def retr_exarcomp(gdat, strgtarg=None):
    
    # get NASA Exoplanet Archive data
    path = gdat.pathbase + 'data/PSCompPars_2020.09.15_17.22.26.csv'
    print('Reading %s...' % path)
    objtexarcomp = pd.read_csv(path, skiprows=330)
    if strgtarg is None:
        indx = np.arange(objtexarcomp['hostname'].size)
    else:
        indx = np.where(objtexarcomp['hostname'] == strgtarg)[0]
    print('strgtarg')
    print(strgtarg)
    
    if indx.size == 0:
        print('The planet name, %s, was not found in the NASA Exoplanet Archive composite table.' % strgtarg)
        return None
    else:
        dictexarcomp = {}
        dictexarcomp['namestar'] = objtexarcomp['hostname'][indx].values
        
        dictexarcomp['radistar'] = objtexarcomp['st_rad'][indx].values * gdat.factrsrj # [R_J]
        radistarstd1 = objtexarcomp['st_raderr1'][indx].values * gdat.factrsrj # [R_J]
        radistarstd2 = objtexarcomp['st_raderr2'][indx].values * gdat.factrsrj # [R_J]
        dictexarcomp['stdvradistar'] = (radistarstd1 + radistarstd2) / 2.
        dictexarcomp['massstar'] = objtexarcomp['st_mass'][indx].values * gdat.factmsmj # [M_J]
        massstarstd1 = objtexarcomp['st_masserr1'][indx].values * gdat.factmsmj # [M_J]
        massstarstd2 = objtexarcomp['st_masserr2'][indx].values * gdat.factmsmj # [M_J]
        dictexarcomp['stdvmassstar'] = (massstarstd1 + massstarstd2) / 2.
        dictexarcomp['tmptstar'] = objtexarcomp['st_teff'][indx].values # [K]
        tmptstarstd1 = objtexarcomp['st_tefferr1'][indx].values # [K]
        tmptstarstd2 = objtexarcomp['st_tefferr2'][indx].values # [K]
        dictexarcomp['stdvtmptstar'] = (tmptstarstd1 + tmptstarstd2) / 2.
        
        dictexarcomp['inso'] = objtexarcomp['pl_insol'][indx].values
        dictexarcomp['nameplan'] = objtexarcomp['pl_name'][indx].values
        dictexarcomp['peri'] = objtexarcomp['pl_orbper'][indx].values # [days]
        dictexarcomp['radiplan'] = objtexarcomp['pl_radj'][indx].values # [R_J]
        dictexarcomp['smax'] = objtexarcomp['pl_orbsmax'][indx].values # [AU]
        dictexarcomp['massplan'] = objtexarcomp['pl_bmassj'][indx].values # [M_J]
        dictexarcomp['stdvradiplan'] = np.maximum(objtexarcomp['pl_radjerr1'][indx].values, objtexarcomp['pl_radjerr2'][indx].values) # [R_J]
        dictexarcomp['stdvmassplan'] = np.maximum(objtexarcomp['pl_bmassjerr1'][indx].values, objtexarcomp['pl_bmassjerr2'][indx].values) # [M_J]
        dictexarcomp['tmptplan'] = objtexarcomp['pl_eqt'][indx].values # [K]
        dictexarcomp['stdvtmptplan'] = (objtexarcomp['pl_eqterr1'][indx].values + objtexarcomp['pl_eqterr2'][indx].values) # [K]
        dictexarcomp['booltran'] = objtexarcomp['tran_flag'][indx].values # [K]
        dictexarcomp['booltran'] = dictexarcomp['booltran'].astype(bool)
        dictexarcomp['vmagstar'] = objtexarcomp['sy_vmag'][indx].values
        dictexarcomp['jmagstar'] = objtexarcomp['sy_jmag'][indx].values # [K]
        dictexarcomp['hmagstar'] = objtexarcomp['sy_hmag'][indx].values # [K]
        dictexarcomp['kmagstar'] = objtexarcomp['sy_kmag'][indx].values # [K]
        dictexarcomp['densplan'] = objtexarcomp['pl_dens'][indx].values / 5.51 # [d_E]
        
        numbplanexar = len(dictexarcomp['nameplan'])
        print('numbplanexar')
        print(numbplanexar)
        
        dictexarcomp['numbplanstar'] = np.empty(numbplanexar)
        dictexarcomp['numbplantranstar'] = np.empty(numbplanexar)
        dictexarcomp['boolfrst'] = np.zeros(numbplanexar, dtype=bool)
        dictexarcomp['booltrantotl'] = np.empty(numbplanexar, dtype=bool)
        for k, namestar in enumerate(dictexarcomp['namestar']):
            indxexarstar = np.where(namestar == dictexarcomp['namestar'])[0]
            if k == indxexarstar[0]:
                dictexarcomp['boolfrst'][k] = True
            dictexarcomp['numbplanstar'][k] = indxexarstar.size
            indxexarstartran = np.where((namestar == dictexarcomp['namestar']) & dictexarcomp['booltran'])[0]
            dictexarcomp['numbplantranstar'][k] = indxexarstartran.size
            dictexarcomp['booltrantotl'][k] = dictexarcomp['booltran'][indxexarstar].all()
    
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


def retr_objtlinefade(x, y, color='black', alpha_initial=1., alpha_final=0.):
    
    color = get_color(color)
    cdict = {'red': ((0.,color[0],color[0]),(1.,color[0],color[0])),
             'green': ((0.,color[1],color[1]),(1.,color[1],color[1])),
             'blue': ((0.,color[2],color[2]),(1.,color[2],color[2])),
             'alpha': ((0.,alpha_initial, alpha_initial), (1., alpha_final, alpha_final))}
    
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


def plot_pcur(gdat, strgpdfn):
    
    if strgpdfn == 'prio':
        arrypcur = gdat.arrypcurprimbdtr
        arrypcurbind = gdat.arrypcurprimbdtrbind
    else:
        arrypcur = gdat.arrypcurprimadtr
        arrypcurbind = gdat.arrypcurprimadtrbind

    for a in range(2):
        # plot individual phase curves
        for p in gdat.indxinst:
            for j in gdat.indxplan:
                # phase on the horizontal axis
                figr, axis = plt.subplots(1, 1, figsize=gdat.figrsizeydob)
                axis.plot(arrypcur[p][j][:, 0], arrypcur[p][j][:, 1], color='grey', alpha=0.2, marker='o', ls='', ms=0.5, rasterized=True)
                axis.plot(arrypcurbind[p][j][:, 0], arrypcurbind[p][j][:, 1], color=gdat.listcolrplan[j], marker='o', ls='', ms=2)
                if gdat.boolwritplan:
                    axis.text(0.9, 0.9, r'\textbf{%s}' % gdat.liststrgplan[j], color=gdat.listcolrplan[j], va='center', ha='center', transform=axis.transAxes)
                axis.set_ylabel('Relative Flux')
                axis.set_xlabel('Phase')
                
                if a == 1:
                    ylim = [np.percentile(arrypcur[p][j][:, 1], 0.3), np.percentile(arrypcur[p][j][:, 1], 99.7)]
                    axis.set_ylim(ylim)
                
                # overlay the posterior model
                if strgpdfn == 'post':
                    axis.plot(gdat.arrypcurprimmodl[p][j][:, 0], gdat.arrypcurprimmodl[p][j][:, 1], color='b')
                    
                path = gdat.pathimag + 'pcurphas_%s_pla%d_%s_%d.%s' % (gdat.liststrginst[p], j, strgpdfn, a, gdat.strgplotextn)
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()
            
                # time on the horizontal axis
                figr, axis = plt.subplots(1, 1, figsize=gdat.figrsize)
                axis.plot(gdat.periprio[j] * arrypcur[p][j][:, 0] * 24., arrypcur[p][j][:, 1], \
                                                                    color='grey', alpha=0.2, marker='o', ls='', ms=0.5, rasterized=True)
                axis.plot(gdat.periprio[j] * arrypcurbind[p][j][:, 0] * 24., arrypcurbind[p][j][:, 1], \
                                                                                    color=gdat.listcolrplan[j], marker='o', ls='', ms=2)
                if strgpdfn == 'post':
                    axis.plot(gdat.periprio[j] * 24. * gdat.arrypcurprimmodl[p][j][:, 0], gdat.arrypcurprimmodl[p][j][:, 1], color='b')
                    
                if gdat.boolwritplan:
                    axis.text(0.9, 0.9, \
                                    r'\textbf{%s}' % gdat.liststrgplan[j], color=gdat.listcolrplan[j], va='center', ha='center', transform=axis.transAxes)
                axis.set_ylabel('Relative Flux')
                axis.set_xlabel('Time [hours]')
                axis.set_xlim([-2 * np.amax(gdat.duraprio) * 24., 2 * np.amax(gdat.duraprio) * 24.])
                
                if a == 1:
                    axis.set_ylim(ylim)
                
                plt.subplots_adjust(hspace=0., bottom=0.25, left=0.25)
                path = gdat.pathimag + 'pcurtime_%s_pla%d_%s_%d.%s' % (gdat.liststrginst[p], j, strgpdfn, a, gdat.strgplotextn)
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()
            
            # plot all phase curves
            if gdat.numbplan > 1:
                figr, axis = plt.subplots(gdat.numbplan, 1, figsize=gdat.figrsizeydob, sharex=True)
                if gdat.numbplan == 1:
                    axis = [axis]
                for jj, j in enumerate(gdat.indxplan):
                    axis[jj].plot(arrypcur[p][j][:, 0], arrypcur[p][j][:, 1], color='grey', alpha=0.2, marker='o', ls='', ms=0.5, rasterized=True)
                    axis[jj].plot(arrypcurbind[p][j][:, 0], arrypcurbind[p][j][:, 1], color=gdat.listcolrplan[j], marker='o', ls='', ms=2)
                    if gdat.boolwritplan:
                        axis[jj].text(0.97, 0.8, r'\textbf{%s}' % gdat.liststrgplan[j], transform=axis[jj].transAxes, \
                                                                                            color=gdat.listcolrplan[j], va='center', ha='center')
                    if a == 1:
                        ylim = [np.percentile(arrypcur[p][j][:, 1], 0.3), np.percentile(arrypcur[p][j][:, 1], 99.7)]
                        axis[jj].set_ylim(ylim)
                
                axis[0].set_ylabel('Relative Flux')
                axis[0].set_xlim(-0.5, 0.5)
                axis[0].yaxis.set_label_coords(-0.08, 1. - 0.5 * gdat.numbplan)
                axis[gdat.numbplan-1].set_xlabel('Phase')
                
                plt.subplots_adjust(hspace=0., bottom=0.2)
                path = gdat.pathimag + 'pcur_%s_%s_%d.%s' % (gdat.liststrginst[p], strgpdfn, a, gdat.strgplotextn)
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()
    

def retr_albg(amplplanrefl, radiplan, smax):
    
    albg = amplplanrefl / (radiplan / smax)**2
    
    return albg


def calc_prop(gdat, strgpdfn, typeparacalc):

    gdat.liststrgfeat = ['epoc', 'peri', 'rrat', 'rsma', 'cosi', 'ecos', 'esin', 'rvsa']
    if typeparacalc == '0003' or typeparacalc == '0004' or typeparacalc == '0006':
        gdat.liststrgfeat += ['sbrtrati', 'amplelli']
    if typeparacalc == '0003' or typeparacalc == '0006':
        gdat.liststrgfeat += ['amplplan', 'timeshftplan']
    if typeparacalc == '0004':
        gdat.liststrgfeat += ['amplplanther', 'amplplanrefl', 'timeshftplanther', 'timeshftplanrefl']
    
    gdat.dictlist = {}
    gdat.dictpost = {}
    gdat.dicterrr = {}
    for strgfeat in gdat.liststrgfeat:
        gdat.dictlist[strgfeat] = np.empty((gdat.numbsamp, gdat.numbplan))

        for j in gdat.indxplan:
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
    
            if typeparacalc == '0003' or typeparacalc == '0004' or typeparacalc == '0006':
                if strgfeat == 'sbrtrati':
                    strg = '%s_sbratio_TESS' % gdat.liststrgplan[j]
                if strgfeat == 'amplelli':
                    strg = '%s_phase_curve_ellipsoidal_TESS' % gdat.liststrgplan[j]
            if typeparacalc == '0003' or typeparacalc == '0006':
                if strgfeat == 'amplplan':
                    strg = '%s_phase_curve_atmospheric_TESS' % gdat.liststrgplan[j]
                if strgfeat == 'timeshftplan':
                    strg = '%s_phase_curve_atmospheric_shift_TESS' % gdat.liststrgplan[j]
            if typeparacalc == '0004':
                if strgfeat == 'amplplanther':
                    strg = '%s_phase_curve_atmospheric_thermal_TESS' % gdat.liststrgplan[j]
                if strgfeat == 'amplplanrefl':
                    strg = '%s_phase_curve_atmospheric_reflected_TESS' % gdat.liststrgplan[j]
                if strgfeat == 'timeshftplanther':
                    strg = '%s_phase_curve_atmospheric_thermal_shift_TESS' % gdat.liststrgplan[j]
                if strgfeat == 'timeshftplanrefl':
                    strg = '%s_phase_curve_atmospheric_reflected_shift_TESS' % gdat.liststrgplan[j]
            
            if strgpdfn == 'prio':
                gdat.dictlist[strgfeat][:, j] = getattr(gdat, strgfeat + 'prio')[j] + np.random.randn(gdat.numbsamp) * getattr(gdat, strgfeat + 'stdvprio')[j]
            else:
                if strg in gdat.objtalle[typeparacalc].posterior_params.keys():
                    gdat.dictlist[strgfeat][:, j] = gdat.objtalle[typeparacalc].posterior_params[strg][gdat.indxsamp]
                else:
                    gdat.dictlist[strgfeat][:, j] = np.zeros(gdat.numbsamp) + allesfitter.config.BASEMENT.params[strg]

    # allesfitter phase curve depths are in ppt
    for strgfeat in gdat.liststrgfeat:
        if strgfeat.startswith('ampl'):
            gdat.dictlist[strgfeat] *= 1e-3
    
    print('Calculating derived variables...')
    # derived variables
    ## get samples from the star's variables
    gdat.dictlist['radistar'] = np.random.randn(gdat.numbsamp) * gdat.stdvradistar + gdat.radistar
    gdat.dictlist['massstar'] = np.random.randn(gdat.numbsamp) * gdat.stdvmassstar + gdat.massstar
    gdat.dictlist['tmptstar'] = np.random.randn(gdat.numbsamp) * gdat.stdvtmptstar + gdat.tmptstar
    gdat.dictlist['radistar'] = np.vstack([gdat.dictlist['radistar']] * gdat.numbplan).T
    gdat.dictlist['massstar'] = np.vstack([gdat.dictlist['massstar']] * gdat.numbplan).T
    gdat.dictlist['tmptstar'] = np.vstack([gdat.dictlist['tmptstar']] * gdat.numbplan).T
    print('gdat.dictlist[radistar]')
    summgene(gdat.dictlist['radistar'])
    print('gdat.dictlist[massstar]')
    summgene(gdat.dictlist['massstar'])
    print('gdat.dictlist[tmptstar]')
    summgene(gdat.dictlist['tmptstar'])
    
    if typeparacalc == '0003' or typeparacalc == '0004' or typeparacalc == '0006':
        gdat.dictlist['amplnigh'] = gdat.dictlist['sbrtrati'] * gdat.dictlist['rrat']**2
    if typeparacalc == '0003' or typeparacalc == '0006':
        gdat.dictlist['phasshftplan'] = gdat.dictlist['timeshftplan'] * 360. / gdat.dictlist['peri']
    if typeparacalc == '0004':
        gdat.dictlist['phasshftplanther'] = gdat.dictlist['timeshftplanther'] * 360. / gdat.dictlist['peri']
        gdat.dictlist['phasshftplanrefl'] = gdat.dictlist['timeshftplanrefl'] * 360. / gdat.dictlist['peri']

    print('Calculating inclinations...')
    # inclination [degree]
    gdat.dictlist['incl'] = np.arccos(gdat.dictlist['cosi']) * 180. / np.pi
    
    # radius of the planets
    gdat.dictlist['radiplan'] = gdat.dictlist['radistar'] * gdat.dictlist['rrat']
    gdat.dictlist['radiplaneart'] = gdat.dictlist['radiplan'] * gdat.factrjre
    
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
    for j in gdat.indxplan:
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
    gdat.dictlist['massplan'] = np.empty_like(gdat.dictlist['esin']) + np.nan
    gdat.dictlist['massplanused'] = gdat.dictlist['massplanpredchen']
    #for j in gdat.indxplan:
    #    if 
    #        gdat.dictlist['massplanused'][:, j] = 
    #    else:
    #        gdat.dictlist['massplanused'][:, j] = 
            
    # density of the planet
    gdat.dictlist['densplan'] = gdat.dictlist['massplanused'] / gdat.dictlist['radiplan']**3

    # escape velocity
    gdat.dictlist['vesc'] = tesstarg.util.retr_vesc(gdat.dictlist['massplanused'], gdat.dictlist['radiplan'])
    
    print('Calculating radius and period ratios...')
    for j in gdat.indxplan:
        strgratiperi = 'ratiperipla%d' % j
        strgratiradi = 'ratiradipla%d' % j
        for jj in gdat.indxplan:
            gdat.dictlist[strgratiperi] = gdat.dictlist['peri'][:, j] / gdat.dictlist['peri'][:, jj]
            gdat.dictlist[strgratiradi] = gdat.dictlist['radiplan'][:, j] / gdat.dictlist['radiplan'][:, jj]
    
    gdat.dictlist['ecce'] = gdat.dictlist['esin']**2 + gdat.dictlist['ecos']**2
    print('Calculating RV semi-amplitudes...')
    # RV semi-amplitude
    gdat.dictlist['rvsapred'] = tesstarg.util.retr_radvsema(gdat.dictlist['peri'], gdat.dictlist['massplanpred'], \
                                                                                            gdat.dictlist['massstar'] / gdat.factmsmj, \
                                                                                                gdat.dictlist['incl'], gdat.dictlist['ecce'])
    
    print('Calculating TSMs...')
    # TSM
    gdat.dictlist['tsmm'] = tesstarg.util.retr_tsmm(gdat.dictlist['radiplan'], gdat.dictlist['tmptplan'], \
                                                                                gdat.dictlist['massplanused'], gdat.dictlist['radistar'], gdat.jmagstar)
    
    # ESM
    gdat.dictlist['esmm'] = tesstarg.util.retr_esmm(gdat.dictlist['tmptplan'], gdat.dictlist['tmptstar'], \
                                                                                gdat.dictlist['radiplan'], gdat.dictlist['radistar'], gdat.kmagstar)
        
    gdat.dictlist['ltsm'] = np.log(gdat.dictlist['tsmm']) 
    gdat.dictlist['lesm'] = np.log(gdat.dictlist['esmm']) 

    # temp
    gdat.dictlist['sini'] = np.sqrt(1. - gdat.dictlist['cosi']**2)
    gdat.dictlist['omeg'] = 180. / np.pi * np.mod(np.arctan2(gdat.dictlist['esin'], gdat.dictlist['ecos']), 2 * np.pi)
    gdat.dictlist['rs2a'] = gdat.dictlist['rsma'] / (1. + gdat.dictlist['rrat'])
    gdat.dictlist['dept'] = gdat.dictlist['rrat']**2
    gdat.dictlist['sinw'] = np.sin(np.pi / 180. * gdat.dictlist['omeg'])
    gdat.dictlist['imfa'] = 1. / gdat.dictlist['rs2a'] * gdat.dictlist['cosi'] * (1. - gdat.dictlist['ecce'])**2 / \
                                                                        (1. + gdat.dictlist['ecce'] * gdat.dictlist['sinw'])
    
    print('Calculating durations...')
    gdat.dictlist['durafull'] = gdat.dictlist['peri'] / np.pi * np.arcsin(gdat.dictlist['rs2a'] / gdat.dictlist['sini'] * \
                        np.sqrt((1. - gdat.dictlist['rrat'])**2 - gdat.dictlist['imfa']**2))
    gdat.dictlist['duratotl'] = gdat.dictlist['peri'] / np.pi * np.arcsin(gdat.dictlist['rs2a'] / gdat.dictlist['sini'] * \
                        np.sqrt((1. + gdat.dictlist['rrat'])**2 - gdat.dictlist['imfa']**2))
    
    gdat.dictlist['maxmdeptblen'] = (1. - gdat.dictlist['durafull'] / gdat.dictlist['duratotl'])**2 / \
                                                                    (1. + gdat.dictlist['durafull'] / gdat.dictlist['duratotl'])**2
    gdat.dictlist['minmdilu'] = gdat.dictlist['dept'] / gdat.dictlist['maxmdeptblen']
    gdat.dictlist['minmratiflux'] = gdat.dictlist['minmdilu'] / (1. - gdat.dictlist['minmdilu'])
    gdat.dictlist['maxmdmag'] = -2.5 * np.log10(gdat.dictlist['minmratiflux'])
    
    # 0003 single component, offset
    # 0004 double component, offset
    # 0006 single component, GP
    if typeparacalc == '0003' or typeparacalc == '0006':
        frac = np.random.rand(gdat.dictlist['amplplan'].size).reshape(gdat.dictlist['amplplan'].shape)
        gdat.dictlist['amplplanther'] = gdat.dictlist['amplplan'] * frac
        gdat.dictlist['amplplanrefl'] = gdat.dictlist['amplplan'] * (1. - frac)
    
    if typeparacalc == '0004':
        # temp -- this does not work for two component (thermal + reflected)
        gdat.dictlist['amplseco'] = gdat.dictlist['amplnigh'] + gdat.dictlist['amplplanther'] + gdat.dictlist['amplplanrefl']
    if typeparacalc == '0003' or typeparacalc == '0006':
        # temp -- this does not work when phase shift is nonzero
        gdat.dictlist['amplseco'] = gdat.dictlist['amplnigh'] + gdat.dictlist['amplplan']
    
    if typeparacalc == '0003' or typeparacalc == '0004' or typeparacalc == '0006':
        gdat.dictlist['albg'] = retr_albg(gdat.dictlist['amplplanrefl'], gdat.dictlist['radiplan'], gdat.dictlist['smax'])

    print('Calculating the equilibrium temperature of the planets...')
    gdat.dictlist['tmptequi'] = gdat.dictlist['tmptstar'] * np.sqrt(gdat.dictlist['radistar'] / gdat.dictlist['smax'] / 2.)
    
    if gdat.labltarg == 'WASP-121':# and typeparacalc == '0003':
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
            listtmpttemp = tdpy.mcmc.samp(gdat, gdat.pathimagpcur, numbsampwalk, numbsampburnwalk, numbsampburnwalkseco, retr_llik_spec, \
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

    gdat.indxsamp = np.arange(gdat.numbsamp)
    gdat.boolsampbadd = np.zeros(gdat.numbsamp, dtype=bool)
    for j in gdat.indxplan:
        boolsampbaddtemp = ~np.isfinite(gdat.dictlist['maxmdmag'][:, j])
        gdat.boolsampbadd = gdat.boolsampbadd | boolsampbaddtemp
    gdat.indxsampbadd = np.where(gdat.boolsampbadd)[0]
    gdat.indxsamptran = np.setdiff1d(gdat.indxsamp, gdat.indxsampbadd)

    gdat.liststrgfeat = gdat.dictlist.keys()
    for strgfeat in gdat.liststrgfeat:
        print('strgfeat')
        print(strgfeat)
        errrshap = list(gdat.dictlist[strgfeat].shape)
        errrshap[0] = 3
        gdat.dictpost[strgfeat] = np.empty(errrshap)
        gdat.dicterrr[strgfeat] = np.empty(errrshap)
        gdat.dictpost[strgfeat][0, ...] = np.percentile(gdat.dictlist[strgfeat], 16., 0)
        gdat.dictpost[strgfeat][1, ...] = np.percentile(gdat.dictlist[strgfeat], 50., 0)
        gdat.dictpost[strgfeat][2, ...] = np.percentile(gdat.dictlist[strgfeat], 84., 0)
        gdat.dicterrr[strgfeat][0, ...] = gdat.dictpost[strgfeat][1, ...]
        gdat.dicterrr[strgfeat][1, ...] = gdat.dictpost[strgfeat][1, ...] - gdat.dictpost[strgfeat][0, ...]
        gdat.dicterrr[strgfeat][2, ...] = gdat.dictpost[strgfeat][2, ...] - gdat.dictpost[strgfeat][1, ...]


def proc_alle(gdat, typeparacalc):
    
    # allesfit run folder
    gdat.pathalle[typeparacalc] = gdat.pathallebase + 'allesfit_%s/' % typeparacalc
    
    # make sure the folder exists
    cmnd = 'mkdir -p %s' % gdat.pathalle[typeparacalc]
    os.system(cmnd)
    
    # write the input data file
    for p in gdat.indxinst:
        path = gdat.pathalle[typeparacalc] + gdat.liststrginst[p] + '.csv'
        indxchuninst = gdat.listindxchuninst[p]
        print('indxchuninst')
        summgene(indxchuninst)
        print(indxchuninst)
        listarrylcurbdtrtemp = []
        for y in gdat.indxchun[p]:
            listarrylcurbdtrtemp.append(gdat.listarrylcurbdtr[p][y])
        listarrylcurbdtrtemp = np.concatenate(listarrylcurbdtrtemp)
        indx = np.argsort(listarrylcurbdtrtemp[:, 0])
        listarrylcurbdtrtemp = listarrylcurbdtrtemp[indx, :]
        print('listarrylcurbdtrtemp[0, :]')
        summgene(listarrylcurbdtrtemp[0, :])
        print('listarrylcurbdtrtemp[1, :]')
        summgene(listarrylcurbdtrtemp[1, :])
        print('listarrylcurbdtrtemp[2, :]')
        summgene(listarrylcurbdtrtemp[2, :])
        print('Writing to %s...' % path)
        np.savetxt(path, listarrylcurbdtrtemp, delimiter=',', header='time,flux,flux_err')
    
    if gdat.peristdvprio is None:
        gdat.peristdvprio = 0.01

    if gdat.epocstdvprio is None:
        gdat.epocstdvprio = 0.5

    if typeparacalc == 'orbt':
        pass
    pathpara = gdat.pathalle[typeparacalc] + 'params.csv'
    if not os.path.exists(pathpara):
        cmnd = 'touch %s' % (pathpara)
        print(cmnd)
        os.system(cmnd)
    
        lineadde = []
        lineadde.extend([ \
                        ['', '#name,value,fit,bounds,label,unit\n'], \
                        ])
        for j in gdat.indxplan:
            strgrrat = '%s_rr' % gdat.liststrgplan[j]
            strgrsma = '%s_rsuma' % gdat.liststrgplan[j]
            strgcosi = '%s_cosi' % gdat.liststrgplan[j]
            strgepoc = '%s_epoch' % gdat.liststrgplan[j]
            strgperi = '%s_period' % gdat.liststrgplan[j]
            strgecos = '%s_f_c' % gdat.liststrgplan[j]
            strgesin = '%s_f_s' % gdat.liststrgplan[j]
            lineadde.extend([ \
                        ['', '%s,%f,1,uniform %f %f,$R_{%s} / R_\star$,\n' % \
                             (strgrrat, gdat.rratprio[j], 0, 2 * gdat.rratprio[j], gdat.liststrgplan[j])], \
                        ['', '%s,%f,1,uniform %f %f,$(R_\star + R_{%s}) / a_{%s}$,\n' % \
                             (strgrsma, gdat.rsmaprio[j], 0, 2 * gdat.rsmaprio[j], gdat.liststrgplan[j], gdat.liststrgplan[j])], \
                        ['', '%s,%f,1,uniform %f %f,$\cos{i_{%s}}$,\n' % \
                             (strgcosi, gdat.cosiprio[j], 0, max(0.1, gdat.cosiprio[j] * 2), gdat.liststrgplan[j])], \
                        ['', '%s,%f,1,uniform %f %f,$T_{0;%s}$,$\mathrm{BJD}$\n' % \
                             (strgepoc, gdat.epocprio[j], gdat.epocprio[j] - gdat.epocstdvprio, gdat.epocprio[j] + gdat.epocstdvprio, \
                                                                                                                            gdat.liststrgplan[j])], \
                        ['', '%s,%f,1,uniform %f %f,$P_{%s}$,days\n' % \
                             (strgperi, gdat.periprio[j], gdat.periprio[j] - gdat.peristdvprio, gdat.periprio[j] + gdat.peristdvprio, \
                                                                                                                            gdat.liststrgplan[j])], \
                        #['', '%s,%f,1,uniform %f %f,$\sqrt{e_b} \cos{\omega_b}$,\n' % \
                        #     (strgecos, gdat.ecosprio[j], -0.9, 0.9, gdat.liststrgplan[j])], \
                        #['', '%s,%f,1,uniform %f %f,$\sqrt{e_b} \sin{\omega_b}$,\n' % \
                        #     (strgesin, gdat.esinprio[j], -0.9, 0.9, gdat.liststrgplan[j])], \
                       ])
            if typeparacalc == '0003' or typeparacalc == '0004' or typeparacalc == '0006':
                for p in gdat.indxinst:
                    strgsbrt = '%s_sbratio_' % gdat.liststrgplan[j] + gdat.liststrginst[p]
                    lineadde.extend([ \
                            ['', '%s_sbratio_%s,1e-3,1,uniform 0 1,$J_{%s; \mathrm{%s}}$,\n' % (gdat.liststrgplan[j], \
                                                                                    gdat.liststrginst[p], gdat.liststrgplan[j]), gdat.listlablinst[p]], \
                            ['', '%s_phase_curve_beaming_%s,0,1,uniform 0 10,$A_\mathrm{b}$,\n' % gdat.liststrgplan[j], gdat.liststrginst[p]], \
                            ['', '%s_phase_curve_atmospheric_%s,0,1,uniform 0 10,$A_\mathrm{p}$,\n' % gdat.liststrgplan[j], gdat.liststrginst[p]], \
                            ['', '%s_phase_curve_ellipsoidal_%s,0,1,uniform 0 10,$A_\mathrm{e}$,\n' % gdat.liststrgplan[j], gdat.liststrginst[p]], \
                           ])

        for p in gdat.indxinst:
            strgldc1 = 'host_ldc_q1_%s' % gdat.liststrginst[p]
            strgldc2 = 'host_ldc_q2_%s' % gdat.liststrginst[p]
            strgscal = 'ln_err_flux_%s' % gdat.liststrginst[p]
            strgbaseoffs = 'baseline_offset_flux_%s' % gdat.liststrginst[p]
            strggprosigm = 'baseline_gp_matern32_lnsigma_flux_%s' % gdat.liststrginst[p]
            strggprorhoo = 'baseline_gp_matern32_lnrho_flux_%s' % gdat.liststrginst[p]
            lineadde.extend([ \
                        ['', '%s,0.5,1,uniform 0 1,$q_{1; \mathrm{%s}}$,\n' % \
                             (strgldc1, gdat.listlablinst[p])], \
                        ['', '%s,0.5,1,uniform 0 1,$q_{2; \mathrm{%s}}$,\n' % \
                             (strgldc2, gdat.listlablinst[p])], \
                        ['', '%s,%f,1,uniform %f %f,$\ln{\sigma_\mathrm{%s}}$,\n' % \
                             (strgscal, -8, -12, 0, gdat.listlablinst[p])], \
                       ])
            lineadde.extend([ \
                        ['', '%s,%f,1,uniform %f %f,$O_{\mathrm{%s}}$,\n' % \
                             (strgbaseoffs, 0, -1, 1, gdat.listlablinst[p])], \
                       ])
            #lineadde.extend([ \
            #            ['', '%s,%f,1,uniform %f %f,$\ln{\sigma_{GP;\mathrm{TESS}}}$,\n' % \
            #                 (strggprosigm, -6, -12, 12)], \
            #            ['', '%s,%f,1,uniform %f %f,$\ln{\\rho_{GP;\mathrm{TESS}}}$,\n' % \
            #                 (strggprorhoo, -2, -12, 12)], \
            #           ])
            
            evol_file(gdat, 'params.csv', gdat.pathalle[typeparacalc], lineadde)
    
    pathsett = gdat.pathalle[typeparacalc] + 'settings.csv'
    if not os.path.exists(pathsett):
        cmnd = 'touch %s' % (pathsett)
        print(cmnd)
        os.system(cmnd)
        
        lineadde = []
        # add user-defined settings.csv inputs
        for strg, varb in gdat.dictsettalle.items():
            lineadde.extend([ \
                            ['', '%s,%s\n' % (strg, varb)] \
                            ])

        if not 'fast_fit' in [lineadde[k][1].split(',')[0] for k in range(len(lineadde))]:
            lineadde = [ \
                         ['', 'fast_fit,True\n'], \
                       ]
        lineadde += [ \
                     ['', 'fast_fit_width,%.3g\n' % np.amax(gdat.duramask)], \
                     ['', 'multiprocess,True\n'], \
                     ['', 'multiprocess_cores,all\n'], \
                     ['', 'mcmc_nwalkers,100\n'], \
                     ['', 'mcmc_total_steps,400000\n'], \
                     ['', 'mcmc_burn_steps,10000\n'], \
                     ['', 'mcmc_thin_by,100\n'], \
                   ]
        for p in gdat.indxinst:
            lineadde += [ \
                         ['', 'inst_phot,%s\n' % gdat.liststrginst[p]], \
                         ['', 'host_ld_law_%s,quad\n' % gdat.liststrginst[p]], \
                         ['', 'host_grid_%s,very_sparse\n' % gdat.liststrginst[p]], \
                         ['', 'baseline_flux_%s,sample_offset\n' % gdat.liststrginst[p]], \
                         ['', 'use_host_density_prior,False\n'], \
                        ]
        
        if typeparacalc == '0003' or typeparacalc == '0004' or typeparacalc == '0006':
            lineadde.extend([ \
                            ['', 'phase_curve,True\n'], \
                            ['', 'phase_curve_style,sine_physical\n'], \
                            ])
        
        for p in gdat.indxinst:
            for j in gdat.indxplan:
                lineadde.extend([ \
                                ['', '%s_grid_%s,very_sparse\n' % (gdat.liststrgplan[j], gdat.liststrginst[p])], \
                                ])
        linetemp = 'companions_phot,'
        for j in gdat.indxplan:
            if j != 0:
                linetemp += ' '
            linetemp += '%s' % gdat.liststrgplan[j]
        linetemp += '\n'
        lineadde.extend([ \
                        ['', linetemp] \
                        ])
        
        evol_file(gdat, 'settings.csv', gdat.pathalle[typeparacalc], lineadde)
    
    ## initial plot
    path = gdat.pathalle[typeparacalc] + 'results/initial_guess_b.pdf'
    if not os.path.exists(path):
        allesfitter.show_initial_guess(gdat.pathalle[typeparacalc])
    
    ## do the run
    path = gdat.pathalle[typeparacalc] + 'results/mcmc_save.h5'
    if not os.path.exists(path):
        allesfitter.mcmc_fit(gdat.pathalle[typeparacalc])
    else:
        print('%s exists... Skipping the orbit run.' % path)

    ## make the final plots
    path = gdat.pathalle[typeparacalc] + 'results/mcmc_corner.pdf'
    if not os.path.exists(path):
        allesfitter.mcmc_output(gdat.pathalle[typeparacalc])
        
    # read the allesfitter posterior
    print('Reading from %s...' % gdat.pathalle[typeparacalc])
    gdat.objtalle[typeparacalc] = allesfitter.allesclass(gdat.pathalle[typeparacalc])
    
    gdat.numbsampalle = allesfitter.config.BASEMENT.settings['mcmc_total_steps']
    numbwalkalle = allesfitter.config.BASEMENT.settings['mcmc_nwalkers']
    gdat.numbsampalleburn = allesfitter.config.BASEMENT.settings['mcmc_burn_steps']
    gdat.numbsampallethin = allesfitter.config.BASEMENT.settings['mcmc_thin_by']

    print('numbwalkalle')
    print(numbwalkalle)
    print('gdat.numbsampalle')
    print(gdat.numbsampalle)
    print('gdat.numbsampalleburn')
    print(gdat.numbsampalleburn)
    print('gdat.numbsampallethin')
    print(gdat.numbsampallethin)

    gdat.numbsamp = gdat.objtalle[typeparacalc].posterior_params[list(gdat.objtalle[typeparacalc].posterior_params.keys())[0]].size
    
    if gdat.numbsamp > 10000:
        print('Thinning down the allesfitter chain!')
        gdat.indxsamp = np.random.choice(np.arange(gdat.numbsamp), size=10000, replace=False)
        gdat.numbsamp = 10000
    else:
        gdat.indxsamp = np.arange(gdat.numbsamp)

    calc_prop(gdat, 'post', typeparacalc)

    gdat.arrylcurmodl = [[[] for p in gdat.liststrginst] for p in gdat.indxinst]
    for p, strginst in enumerate(gdat.liststrginst):
        gdat.arrylcurmodl[p] = np.empty((gdat.time[p].size, 3))
        gdat.arrylcurmodl[p][:, 0] = gdat.time[p]
        gdat.arrylcurmodl[p][:, 1] = gdat.objtalle[typeparacalc].get_posterior_median_model(strginst, 'flux', xx=gdat.time[p])
        gdat.arrylcurmodl[p][:, 2] = 0.

    # write the model to file
    for p in gdat.indxinst:
        
        path = gdat.pathdata + 'arrylcurmodl' + gdat.liststrgchun[y] + '.csv'
        print('Writing to %s...' % path)
        np.savetxt(path, gdat.arrylcurmodl[p], delimiter=',', header='time,flux,flux_err')

    # number of samples to plot
    gdat.numbsampplot = min(100, gdat.numbsamp)
    gdat.indxsampplot = np.random.choice(gdat.indxsamp, gdat.numbsampplot, replace=False)
    gdat.listlcurmodl = np.empty((gdat.numbsampplot, gdat.time[p].size))
    gdat.listpcurquadmodl = [[[] for j in gdat.indxplan] for p in gdat.liststrginst]
    for p in gdat.indxinst:
        for j in gdat.indxplan:
            gdat.listpcurquadmodl[p][j] = np.empty((gdat.numbsampplot, gdat.numbtime[p]))
        gdat.arrylcurmodl[p] = np.zeros((gdat.time[p].size, 3))
        gdat.arrylcurmodl[p][:, 0] = gdat.time[p]
    gdat.arrypcurprimmodl = [[[] for j in gdat.indxplan] for p in gdat.liststrginst]
    gdat.arrypcurquadmodl = [[[] for j in gdat.indxplan] for p in gdat.liststrginst]
    gdat.arrypcurprimadtr = [[[] for j in gdat.indxplan] for p in gdat.liststrginst]
    gdat.arrypcurprimadtrbind = [[[] for j in gdat.indxplan] for p in gdat.liststrginst]
    gdat.arrypcurprimadtrbindfine = [[[] for j in gdat.indxplan] for p in gdat.liststrginst]
    gdat.arrypcurquadadtr = [[[] for j in gdat.indxplan] for p in gdat.liststrginst]
    gdat.arrypcurquadadtrbind = [[[] for j in gdat.indxplan] for p in gdat.liststrginst]
    gdat.arrypcurquadadtrbindfine = [[[] for j in gdat.indxplan] for p in gdat.liststrginst]
    for p, strginst in enumerate(gdat.liststrginst):
        for ii, i in enumerate(gdat.indxsampplot):
            # this is only the physical model and excludes the baseline, which is available separately via get_one_posterior_baseline()
            gdat.listlcurmodl[ii, :] = gdat.objtalle[typeparacalc].get_one_posterior_model(strginst, 'flux', xx=gdat.time[p], sample_id=i)
            print('gdat.arrylcurmodl[p]')
            summgene(gdat.arrylcurmodl[p])
            gdat.arrylcurmodl[p][:, 1] = gdat.listlcurmodl[ii, :]
            print('gdat.arrylcurmodl[p]')
            summgene(gdat.arrylcurmodl[p])
            gdat.listpcurquadmodl[p][j][ii, :] = tesstarg.util.fold_lcur(gdat.arrylcurmodl[p], \
                                                                                gdat.dicterrr['epoc'][0, j], gdat.dicterrr['peri'][0, j], phasshft=0.25)[:, 1]

        # get allesfitter baseline model
        gdat.lcurbasealle = gdat.objtalle[typeparacalc].get_posterior_median_baseline(strginst, 'flux', xx=gdat.time[p])
        # get allesfitter-detrended data
        gdat.lcuradtr = gdat.arrylcurbdtr[p][:, 1] - gdat.lcurbasealle
        gdat.arrylcuradtr = np.copy(gdat.arrylcurbdtr[p])
        gdat.arrylcuradtr[:, 1] = gdat.lcuradtr
        
        for j in gdat.indxplan:
            gdat.arrypcurprimmodl[p][j] = tesstarg.util.fold_lcur(gdat.arrylcurmodl[p][gdat.listindxtimeclen[j][p], :], \
                                                                                            gdat.dicterrr['epoc'][0, j], gdat.dicterrr['peri'][0, j])
            gdat.arrypcurquadmodl[p][j] = tesstarg.util.fold_lcur(gdat.arrylcurmodl[p][gdat.listindxtimeclen[j][p], :], \
                                                                                gdat.dicterrr['epoc'][0, j], gdat.dicterrr['peri'][0, j], phasshft=0.25)
            gdat.arrypcurprimadtr[p][j] = tesstarg.util.fold_lcur(gdat.arrylcuradtr[gdat.listindxtimeclen[j][p], :], \
                                                                                            gdat.dicterrr['epoc'][0, j], gdat.dicterrr['peri'][0, j])
            gdat.arrypcurprimadtrbind[p][j] = tesstarg.util.rebn_lcur(gdat.arrypcurprimadtr[p][j], gdat.numbbinspcur)
            gdat.arrypcurprimadtrbindfine[p][j] = tesstarg.util.rebn_lcur(gdat.arrypcurprimadtr[p][j], gdat.numbbinspcurfine)
            gdat.arrypcurquadadtr[p][j] = tesstarg.util.fold_lcur(gdat.arrylcuradtr[gdat.listindxtimeclen[j][p], :], \
                                                                                      gdat.dicterrr['epoc'][0, j], gdat.dicterrr['peri'][0, j], phasshft=0.25)
            gdat.arrypcurquadadtrbind[p][j] = tesstarg.util.rebn_lcur(gdat.arrypcurquadadtr[p][j], gdat.numbbinspcur)
            gdat.arrypcurquadadtrbindfine[p][j] = tesstarg.util.rebn_lcur(gdat.arrypcurquadadtr[p][j], gdat.numbbinspcurfine)
    
    # plots
    ## plot GP-detrended phase curves
    plot_pcur(gdat, 'post')
    
    plot_prop(gdat, 'post')

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

    if typeparacalc == '0003' or typeparacalc == '0004' or typeparacalc == '0006':
        gdat.arrylcurmodlcomp = [[] for p in gdat.indxinst]
        gdat.arrypcurquadmodlcomp = [[[] for j in gdat.indxplan] for p in gdat.indxinst]
        for p in gdat.indxinst:
            if typeparacalc == '0003' or typeparacalc == '0006': 
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
                if typeparacalc == '0003' or typeparacalc == '0006': 
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
                print('gdat.pathimagpcur')
                print(gdat.pathimagpcur)
                tdpy.mcmc.plot_grid(gdat.pathimagpcur, 'pcur_%s' % typeparacalc, listpost, listlablpara, plotsize=2.5)

            # plot phase curve
            ## determine data gaps for overplotting model without the data gaps
            gdat.indxtimegapp = np.argmax(gdat.time[p][1:] - gdat.time[p][:-1]) + 1
            figr = plt.figure(figsize=(10, 12))
            axis = [[] for k in range(3)]
            axis[0] = figr.add_subplot(3, 1, 1)
            axis[1] = figr.add_subplot(3, 1, 2)
            axis[2] = figr.add_subplot(3, 1, 3, sharex=axis[1])
            
            for k in range(len(axis)):
                
                ## unbinned data
                if k < 2:
                    if k == 0:
                        xdat = gdat.time[p] - gdat.timetess
                        ydat = gdat.arrylcuradtr[:, 1] + gdat.dicterrr['amplnigh'][0, 0]
                    if k == 1:
                        xdat = gdat.arrypcurquadadtr[p][j][:, 0]
                        ydat = gdat.arrypcurquadadtr[p][j][:, 1] + gdat.dicterrr['amplnigh'][0, 0]
                    axis[k].plot(xdat, ydat, '.', color='grey', alpha=0.3, label='Raw data')
                
                ## binned data
                if k > 0:
                    xdat = gdat.arrypcurquadadtrbind[p][j][:, 0]
                    ydat = gdat.arrypcurquadadtrbind[p][j][:, 1] + gdat.dicterrr['amplnigh'][0, 0]
                    yerr = np.copy(gdat.arrypcurquadadtrbind[p][j][:, 2])
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
                    xdat = gdat.arrypcurquadmodl[p][j][:, 0]
                    ydat = gdat.arrypcurquadmodl[p][j][:, 1] + gdat.dicterrr['amplnigh'][0, 0]
                else:
                    xdat = gdat.arrylcurmodl[p][:, 0] - gdat.timetess
                    ydat = gdat.arrylcurmodl[p][:, 1] + gdat.dicterrr['amplnigh'][0, 0]
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
            
            gdat.arrypcurquadmodlcomp[p][j] = dict()
            gdat.arrypcurquadmodlcomp[p][j]['totl'] = gdat.arrypcurquadmodl[p][j]

            ## plot components in the zoomed panel
            arrylcurmodltemp = np.copy(gdat.arrylcurmodl[p])
            
            ### stellar baseline
            gdat.objtalle[typeparacalc] = allesfitter.allesclass(gdat.pathalle[typeparacalc])
            gdat.objtalle[typeparacalc].posterior_params_median['b_phase_curve_ellipsoidal_TESS'] = 0
            if typeparacalc == '0003' or typeparacalc == '0006':
                gdat.objtalle[typeparacalc].posterior_params_median['b_phase_curve_atmospheric_TESS'] = 0
            if typeparacalc == '0004':
                gdat.objtalle[typeparacalc].posterior_params_median['b_phase_curve_atmospheric_thermal_TESS'] = 0
                gdat.objtalle[typeparacalc].posterior_params_median['b_phase_curve_atmospheric_reflected_TESS'] = 0
            gdat.objtalle[typeparacalc].posterior_params_median['b_sbratio_TESS'] = 0
            arrylcurmodltemp[:, 1] = gdat.objtalle[typeparacalc].get_posterior_median_model(strginst, 'flux', xx=gdat.time[p])
            gdat.arrypcurquadmodlcomp[p][j]['stel'] = tesstarg.util.fold_lcur(arrylcurmodltemp, \
                                                                    gdat.dicterrr['epoc'][0, j], gdat.dicterrr['peri'][0, j], phasshft=0.25)
            xdat = gdat.arrypcurquadmodlcomp[p][j]['stel'][:, 0]
            ydat = (gdat.arrypcurquadmodlcomp[p][j]['stel'][:, 1] - 1.) * 1e6
            axis[2].plot(xdat, ydat, lw=2, color='orange', label='Stellar baseline', ls='--', zorder=11)
            
            ### EV
            gdat.objtalle[typeparacalc] = allesfitter.allesclass(gdat.pathalle[typeparacalc])
            gdat.objtalle[typeparacalc].posterior_params_median['b_sbratio_TESS'] = 0
            if typeparacalc == '0003' or typeparacalc == '0006':
                gdat.objtalle[typeparacalc].posterior_params_median['b_phase_curve_atmospheric_TESS'] = 0
            if typeparacalc == '0004':
                gdat.objtalle[typeparacalc].posterior_params_median['b_phase_curve_atmospheric_thermal_TESS'] = 0
                gdat.objtalle[typeparacalc].posterior_params_median['b_phase_curve_atmospheric_reflected_TESS'] = 0
            arrylcurmodltemp[:, 1] = gdat.objtalle[typeparacalc].get_posterior_median_model(strginst, 'flux', xx=gdat.time[p])
            gdat.arrypcurquadmodlcomp[p][j]['elli'] = tesstarg.util.fold_lcur(arrylcurmodltemp, \
                                                                      gdat.dicterrr['epoc'][0, j], gdat.dicterrr['peri'][0, j], phasshft=0.25)
            gdat.arrypcurquadmodlcomp[p][j]['elli'][:, 1] -= gdat.arrypcurquadmodlcomp[p][j]['stel'][:, 1]
            xdat = gdat.arrypcurquadmodlcomp[p][j]['elli'][:, 0]
            ydat = (gdat.arrypcurquadmodlcomp[p][j]['elli'][:, 1] - 1.) * 1e6
            axis[2].plot(xdat, ydat, lw=2, color='r', ls='--', label='Ellipsoidal variation')
            
            # planetary
            gdat.objtalle[typeparacalc] = allesfitter.allesclass(gdat.pathalle[typeparacalc])
            gdat.objtalle[typeparacalc].posterior_params_median['b_phase_curve_ellipsoidal_TESS'] = 0
            arrylcurmodltemp[:, 1] = gdat.objtalle[typeparacalc].get_posterior_median_model(strginst, 'flux', xx=gdat.time[p])
            gdat.arrypcurquadmodlcomp[p][j]['plan'] = tesstarg.util.fold_lcur(arrylcurmodltemp, \
                                                                    gdat.dicterrr['epoc'][0, j], gdat.dicterrr['peri'][0, j], phasshft=0.25)
            gdat.arrypcurquadmodlcomp[p][j]['plan'] += gdat.dicterrr['amplnigh'][0, 0]
            gdat.arrypcurquadmodlcomp[p][j]['plan'][:, 1] -= gdat.arrypcurquadmodlcomp[p][j]['stel'][:, 1]
            
            xdat = gdat.arrypcurquadmodlcomp[p][j]['plan'][:, 0]
            ydat = (gdat.arrypcurquadmodlcomp[p][j]['plan'][:, 1] - 1.) * 1e6
            axis[2].plot(xdat, ydat, lw=2, color='g', label='Planetary', ls='--')
    
            # planetary nightside
            gdat.objtalle[typeparacalc] = allesfitter.allesclass(gdat.pathalle[typeparacalc])
            gdat.objtalle[typeparacalc].posterior_params_median['b_phase_curve_ellipsoidal_TESS'] = 0
            if typeparacalc == '0003' or typeparacalc == '0006':
                gdat.objtalle[typeparacalc].posterior_params_median['b_phase_curve_atmospheric_TESS'] = 0
            if typeparacalc == '0004':
                gdat.objtalle[typeparacalc].posterior_params_median['b_phase_curve_atmospheric_thermal_TESS'] = 0
                gdat.objtalle[typeparacalc].posterior_params_median['b_phase_curve_atmospheric_reflected_TESS'] = 0
            arrylcurmodltemp[:, 1] = gdat.objtalle[typeparacalc].get_posterior_median_model(strginst, 'flux', xx=gdat.time[p])
            gdat.arrypcurquadmodlcomp[p][j]['nigh'] = tesstarg.util.fold_lcur(arrylcurmodltemp, \
                                                                    gdat.dicterrr['epoc'][0, j], gdat.dicterrr['peri'][0, j], phasshft=0.25)
            gdat.arrypcurquadmodlcomp[p][j]['nigh'] += gdat.dicterrr['amplnigh'][0, 0]
            gdat.arrypcurquadmodlcomp[p][j]['nigh'][:, 1] -= gdat.arrypcurquadmodlcomp[p][j]['stel'][:, 1]
            xdat = gdat.arrypcurquadmodlcomp[p][j]['nigh'][:, 0]
            ydat = (gdat.arrypcurquadmodlcomp[p][j]['nigh'][:, 1] - 1.) * 1e6
            axis[2].plot(xdat, ydat, lw=2, color='olive', label='Planetary baseline', ls='--', zorder=11)
    
            ### planetary modulation
            gdat.arrypcurquadmodlcomp[p][j]['pmod'] = np.copy(gdat.arrypcurquadmodlcomp[p][j]['plan'])
            gdat.arrypcurquadmodlcomp[p][j]['pmod'][:, 1] -= gdat.arrypcurquadmodlcomp[p][j]['nigh'][:, 1]
            xdat = gdat.arrypcurquadmodlcomp[p][j]['pmod'][:, 0]
            ydat = (gdat.arrypcurquadmodlcomp[p][j]['pmod'][:, 1] - 1.) * 1e6
            axis[2].plot(xdat, ydat, lw=2, color='m', label='Planetary modulation', ls='--', zorder=11)
             
            ## legend
            axis[2].legend(ncol=3)
            
            path = gdat.pathimagpcur + 'pcur_grid_%s.%s' % (typeparacalc, gdat.strgplotextn)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
            
            
            # replot phase curve
            ### sample model phas
            #numbphasfine = 1000
            #gdat.meanphasfine = np.linspace(np.amin(gdat.arrypcurquadbdtr[0][gdat.indxphasotpr, 0]), \
            #                                np.amax(gdat.arrypcurquadbdtr[0][gdat.indxphasotpr, 0]), numbphasfine)
            #indxphasfineinse = np.where(abs(gdat.meanphasfine - 0.5) < phasseco)[0]
            #indxphasfineotprleft = np.where(-gdat.meanphasfine > phasmask)[0]
            #indxphasfineotprrght = np.where(gdat.meanphasfine > phasmask)[0]
       
            indxphasmodlouttprim = [[] for a in range(2)]
            indxphasdatabindouttprim = [[] for a in range(2)]
            indxphasmodlouttprim[0] = np.where(gdat.arrypcurquadmodlcomp[p][j]['totl'][:, 0] < -0.05)[0]
            indxphasdatabindouttprim[0] = np.where(gdat.arrypcurquadbdtrbind[p][j][:, 0] < -0.05)[0]
            indxphasmodlouttprim[1] = np.where(gdat.arrypcurquadmodlcomp[p][j]['totl'][:, 0] > 0.05)[0]
            indxphasdatabindouttprim[1] = np.where(gdat.arrypcurquadbdtrbind[p][j][:, 0] > 0.05)[0]

            # plot the phase curve with components
            figr, axis = plt.subplots(figsize=gdat.figrsizeydob)
            ## data
            axis.errorbar(gdat.arrypcurquadbdtrbind[p][j][:, 0], \
                           (gdat.arrypcurquadbdtrbind[p][j][:, 1] + gdat.dicterrr['amplnigh'][0, 0] - 1) * 1e6, \
                           yerr=1e6*gdat.arrypcurquadbdtrbind[p][j][:, 2], color='k', marker='o', ls='', markersize=2, lw=1, label='Data')
            ## total model
            axis.plot(gdat.arrypcurquadmodlcomp[p][j]['totl'][:, 0], \
                                            1e6*(gdat.arrypcurquadmodlcomp[p][j]['totl'][:, 1]+gdat.dicterrr['amplnigh'][0, 0]-1), \
                                                                                                            color='b', lw=3, label='Model')
            
            #axis.plot(gdat.arrypcurquadmodlcomp[p][j]['plan'][:, 0], 1e6*(gdat.arrypcurquadmodlcomp[p][j]['plan'][:, 1]), \
            #                                                                                              color='g', label='Planetary', lw=1, ls='--')
            
            axis.plot(gdat.arrypcurquadmodlcomp[p][j]['pmod'][:, 0], 1e6*(gdat.arrypcurquadmodlcomp[p][j]['pmod'][:, 1]), \
                                                                                                  color='m', label='Planetary modulation', lw=2, ls='--')
            axis.plot(gdat.arrypcurquadmodlcomp[p][j]['nigh'][:, 0], 1e6*(gdat.arrypcurquadmodlcomp[p][j]['nigh'][:, 1]), \
                                                                                                  color='olive', label='Planetary baseline', lw=2, ls='--')
            
            axis.plot(gdat.arrypcurquadmodlcomp[p][j]['elli'][:, 0], 1e6*(gdat.arrypcurquadmodlcomp[p][j]['elli'][:, 1]), \
                                                                                                  color='r', label='Ellipsoidal variation', lw=2, ls='--')
            
            axis.plot(gdat.arrypcurquadmodlcomp[p][j]['stel'][:, 0], 1e6*(gdat.arrypcurquadmodlcomp[p][j]['stel'][:, 1]-1.), \
                                                                                                  color='orange', label='Stellar baseline', lw=2, ls='--')
            
            axis.set_ylim(ylimpcur)
            axis.set_ylabel('Relative Flux [ppm]')
            axis.set_xlabel('Phase')
            axis.legend(ncol=3)
            plt.tight_layout()
            path = gdat.pathimagpcur + 'pcur_comp_%s.%s' % (typeparacalc, gdat.strgplotextn)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()

            # plot the phase curve with samples
            figr, axis = plt.subplots(figsize=gdat.figrsizeydob)
            axis.errorbar(gdat.arrypcurquadbdtrbind[p][j][:, 0], (gdat.arrypcurquadbdtrbind[p][j][:, 1] + gdat.dicterrr['amplnigh'][0, 0] - 1) * 1e6, \
                                         yerr=1e6*gdat.arrypcurquadbdtrbind[p][j][:, 2], color='k', marker='o', ls='', markersize=2, lw=1)
            for ii, i in enumerate(gdat.indxsampplot):
                axis.plot(gdat.arrypcurquadmodlcomp[p][j]['totl'][:, 0], 1e6 * (gdat.listpcurquadmodl[p][j][ii, :] + gdat.dicterrr['amplnigh'][0, 0] - 1.), \
                                                                                                                                    alpha=0.1, color='b')
            axis.set_ylabel('Relative Flux [ppm]')
            axis.set_xlabel('Phase')
            axis.set_ylim(ylimpcur)
            plt.tight_layout()
            path = gdat.pathimagpcur + 'pcur_samp_%s.%s' % (typeparacalc, gdat.strgplotextn)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()

            # plot all along with residuals
            #figr, axis = plt.subplots(3, 1, figsize=gdat.figrsizeydob)
            #axis.errorbar(gdat.arrypcurquadbdtrbind[p][j][:, 0], (gdat.arrypcurquadbdtrbind[p][j][:, 1]) * 1e6, \
            #                             yerr=1e6*gdat.arrypcurquadbdtrbind[p][j][:, 2], color='k', marker='o', ls='', markersize=2, lw=1)
            #for kk, k in enumerate(gdat.indxsampplot):
            #    axis.plot(gdat.meanphasfine[indxphasfineotprleft], (listmodltotl[k, indxphasfineotprleft] - listoffs[k]) * 1e6, \
            #                                                                                                            alpha=0.1, color='b')
            #    axis.plot(gdat.meanphasfine[indxphasfineotprrght], (listmodltotl[k, indxphasfineotprrght] - listoffs[k]) * 1e6, \
            #                                                                                                            alpha=0.1, color='b')
            #axis.set_ylabel('Relative Flux - 1 [ppm]')
            #axis.set_xlabel('Phase')
            #plt.tight_layout()
            #path = gdat.pathimagpcur + 'pcur_resi_%s.%s' % (typeparacalc, gdat.strgplotextn)
            #print('Writing to %s...' % path)
            #plt.savefig(path)
            #plt.close()

            # write to text file
            fileoutp = open(gdat.pathdatapcur + 'post_pcur_%s_tabl.csv' % (typeparacalc), 'w')
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
            
            fileoutp = open(gdat.pathdatapcur + 'post_pcur_%s_cmnd.csv' % (typeparacalc), 'w')
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

            if gdat.labltarg == 'WASP-121' and typeparacalc == '0003':
                
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
                path = gdat.pathimagpcur + 'pdfn_albg.%s' % gdat.strgplotextn
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
                path = gdat.pathimagpcur + 'hist_albg.%s' % gdat.strgplotextn
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
                tdpy.mcmc.plot_grid(gdat.pathimagpcur, 'post_atmo', listsampatmo, listlablpara, plotsize=2.5)
   
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
                path = gdat.pathimagpcur + 'kdeg_psii.%s' % gdat.strgplotextn
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
                listpostheat = tdpy.mcmc.samp(gdat, gdat.pathimagpcur, numbsampwalk, numbsampburnwalk, numbsampburnwalkseco, retr_llik_albbemisepsi, \
                                     listlablpara, listscalpara, listminmpara, listmaxmpara, listmeangauspara, liststdvgauspara, numbdata, strgextn=strgextn)

                # plot emission spectra, secondary eclipse depth, and brightness temperature
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
                path = gdat.pathimagpcur + 'spec.%s' % gdat.strgplotextn
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
                path = gdat.pathimagpcur + 'ptem.%s' % gdat.strgplotextn
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()
  
    
def plot_prop(gdat, strgpdfn):
    
    if gdat.boolobjt:
        # pretty orbit plot
        fact = 50.
        factstar = 5.
        figr, axis = plt.subplots(figsize=gdat.figrsizeydob)
        ## star 1
        w1 = mpl.patches.Wedge((0, 0), gdat.radistar/gdat.factaurj*factstar, 270, 360, fc='k', zorder=1, edgecolor='k')
        axis.add_artist(w1)
        for jj, j in enumerate(gdat.indxplan):
            xposmaxm = gdat.dicterrr['smaxasun'][0, j]
            yposmaxm = 0.3 * xposmaxm
            yposelli = yposmaxm * np.concatenate([np.linspace(0., -1., 100), np.linspace(1., 0., 100)])
            xposelli = xposmaxm * np.sqrt(1. - (yposelli / yposmaxm)**2)
            objt = retr_objtlinefade(xposelli, yposelli, color=gdat.listcolrplan[j], alpha_initial=1., alpha_final=0.)
            axis.add_collection(objt)
            # temperature contours?
            #for tmpt in [500., 700,]:
            #    smaj = tmpt
            #    axis.axvline(smaj, ls='--')
            # planet
            w1 = mpl.patches.Circle((gdat.dicterrr['smaxasun'][0, j], 0), \
                                                radius=gdat.dicterrr['radiplan'][0, j]/gdat.factaurj*fact, color=gdat.listcolrplan[j], zorder=3)
            axis.add_artist(w1)
            axis.text(.15 + 0.02 * jj, -0.03, gdat.liststrgplan[j], color=gdat.listcolrplan[j])
        ## add Mercury
        axis.text(.387, 0.01, 'Mercury', color='grey', ha='right')
        w1 = mpl.patches.Circle((0.387, 0), radius=0.0349/gdat.factaurj*fact, color='grey')
        axis.add_artist(w1)
        ##axistwin.set_xlim(axis.get_xlim())
        #xpostemp = axistwin.get_xticks()
        ##axistwin.set_xticks(xpostemp[1:])
        #axistwin.set_xticklabels(['%f' % tmpt for tmpt in listtmpt])
        ## star 2
        w1 = mpl.patches.Wedge((0, 0), gdat.radistar/gdat.factaurj*factstar, 0, 90, fc='k', zorder=4, edgecolor='k')
        axis.add_artist(w1)
        #axistwin = axis.twiny()
        axis.get_yaxis().set_visible(False)
        axis.set_aspect('equal')
        axis.set_xlim([0, 0.4])
        axis.set_ylim([-0.05, 0.05])
        ## centerline
        axis.set_xlabel('Distance from the star [AU]')
        plt.subplots_adjust()
        #axis.legend()
        path = gdat.pathimag + 'orbt_%s.%s' % (strgpdfn, gdat.strgplotextn)
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()
   
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
    
    xerr = (timeoccu[1:] - timeoccu[:-1]) / 2.
    xerr = np.concatenate([xerr[0, None], xerr])
    axis.errorbar(timeoccu, occumean, yerr=occuyerr, xerr=xerr, color='black', ls='', marker='o', lw=1)
    if gdat.boolobjt:
        # this system
        for jj, j in enumerate(gdat.indxplan):
            xposlowr = gdat.factrjre * gdat.dictpost['radiplan'][0, j]
            xposuppr = gdat.factrjre * gdat.dictpost['radiplan'][2, j]
            axis.axvspan(xposlowr, xposuppr, alpha=0.5, color=gdat.listcolrplan[j])
            axis.axvline(gdat.factrjre * gdat.dicterrr['radiplan'][0, j], color=gdat.listcolrplan[j], ls='--', label=gdat.liststrgplan[j])
            axis.text(0.7, 0.9 - jj * 0.07, r'\textbf{%s}' % gdat.liststrgplan[j], color=gdat.listcolrplan[j], \
                                                                                        va='center', ha='center', transform=axis.transAxes)
    axis.set_xlabel('Radius [$R_E$]')
    axis.set_ylabel('Occurrence rate of planets per star')
    plt.subplots_adjust(bottom=0.2)
    path = gdat.pathimag + 'occuradi_%s.%s' % (strgpdfn, gdat.strgplotextn)
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
        
        ## augment
        ### radii
        dictpopl['radistarsunn'] = dictpopl['radistar'] / gdat.factrsrj # R_S
        dictpopl['radiplaneart'] = dictpopl['radiplan'] * gdat.factrjre # R_E
        dictpopl['massplaneart'] = dictpopl['massplan'] * gdat.factmjme # M_E
        
        ### ESM
        dictpopl['tsmm'] = tesstarg.util.retr_esmm(dictpopl['tmptplan'], dictpopl['tmptstar'], dictpopl['radiplan'], \
                                                                                                    dictpopl['radistar'], dictpopl['kmagstar'])
        
        numbsamppopl = 100
        listradiplan = dictpopl['radiplan'][None, :] + np.random.randn(numbsamppopl)[:, None] * dictpopl['stdvradiplan'][None, :]
        listtmptplan = dictpopl['tmptplan'][None, :] + np.random.randn(numbsamppopl)[:, None] * dictpopl['stdvtmptplan'][None, :]
        listmassplan = dictpopl['massplan'][None, :] + np.random.randn(numbsamppopl)[:, None] * dictpopl['stdvmassplan'][None, :]
        listradistar = dictpopl['radistar'][None, :] + np.random.randn(numbsamppopl)[:, None] * dictpopl['stdvradistar'][None, :]
        listjmagstar = dictpopl['jmagstar'][None, :] + np.random.randn(numbsamppopl)[:, None] * 0.
        
        ### TSM
        listtsmm = tesstarg.util.retr_tsmm(listradiplan, listtmptplan, listmassplan, listradistar, listjmagstar)
       
        dictpopl['stdvtsmm'] = np.std(listtsmm, 0)
        dictpopl['tsmm'] = np.median(listtsmm, 0)

        numbtargpopl = dictpopl['radiplan'].size
        indxtargpopl = np.arange(numbtargpopl)
        
        indxpoplcuttradi = dict()
        indxpoplcuttradi['allr'] = indxtargpopl
        indxpoplcuttradi['rb04'] = np.where((dictpopl['radiplaneart'] < 4))[0]
        indxpoplcuttradi['rb24'] = np.where((dictpopl['radiplaneart'] < 4) & (dictpopl['radiplaneart'] > 2.))[0]
        indxpoplcuttmass = dict()
        indxpoplcuttmass['allm'] = indxtargpopl
        indxpoplcuttmass['gmas'] = np.where(dictpopl['stdvmassplan'] / dictpopl['massplan'] < 0.4)[0]
    
        liststrgcuttradi = ['allr', 'rb24', 'rb04']
        
        # period ratios
        figr, axis = plt.subplots(figsize=gdat.figrsize)
        ## all 
        gdat.listratiperi = []
        gdat.intgreso = []
        liststrgstarcomp = []
        for m in gdat.indxplanexar:
            strgstar = dictpopl['namestar'][m]
            if not strgstar in liststrgstarcomp:
                indxexarstar = np.where(dictpopl['namestar'] == strgstar)[0]
                if indxexarstar[0] != m:
                    raise Exception('')
                
                listperi = dictpopl['peri'][None, indxexarstar]
                if not np.isfinite(listperi).all():
                    liststrgstarcomp.append(strgstar)
                    continue
                intgreso, ratiperi = retr_reso(listperi)
                
                numbplan = indxexarstar.size
                
                gdat.listratiperi.append(ratiperi[0, :, :][np.triu_indices(numbplan, k=1)])
                gdat.intgreso.append(intgreso)
                
                liststrgstarcomp.append(strgstar)
        
        gdat.listratiperi = np.concatenate(gdat.listratiperi)
        bins = np.linspace(1., 10., 400)
        axis.hist(gdat.listratiperi, bins=bins)
        if gdat.boolobjt and gdat.numbplan > 1:
            ## this system
            for j in gdat.indxplan:
                for jj in gdat.indxplan:
                    if gdat.dicterrr['peri'][0, j] > gdat.dicterrr['peri'][0, jj]:
                        ratiperi = gdat.dicterrr['peri'][0, j] / gdat.dicterrr['peri'][0, jj]
                        axis.axvline(ratiperi, color='k')
        ## resonances
        for perifrst, periseco in [[2., 1.], [3., 2.], [4., 3.], [5., 4.], [5., 3.], [5., 2.]]:
            axis.axvline(perifrst / periseco, color='grey', ls='--', alpha=0.5)
        #axis.set_xscale('log')
        axis.set_xlim([0.9, 2.7])
        axis.set_xlabel('Period ratio')
        #plt.subplots_adjust()
        path = gdat.pathimag + 'histratiperi_%s_%s.%s' % (strgpdfn, strgpopl, gdat.strgplotextn)
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()
        
        numbtext = min(5, numbtargpopl)
        liststrgcuttmass = ['allm', 'gmas']
        liststrgtext = ['notx', 'text']
        
        liststrgvarb = ['peri', 'inso', 'tmptplan', 'vesc', 'massplan', 'tmptstar', 'radistarsunn', 'tsmm', 'esmm', 'radiplan', 'radiplaneart']
        listlablvarb = [['P', 'days'], ['F', '$F_E$'], ['$T_p$', 'K'], ['$v_{esc}$', 'kms$^{-1}$'], ['$M_p$', '$M_E$'], \
                                                 ['$T_s$', '$T_{eff}$'], ['$R_s$', '$R_S$'], ['TSM', ''], ['ESM', ''], ['R', '$R_J$'], ['R', '$R_E$']] 
        liststrgvarb.append('esmm')
        listlablvarb.append(['ESM', ''])
        
        numbvarb = len(liststrgvarb)
        indxvarb = np.arange(numbvarb)
        listlablvarbtotl = []
        print('listlablvarb')
        print(listlablvarb)
        for k in indxvarb:
            if listlablvarb[k][1] == '':
                listlablvarbtotl.append('%s' % (listlablvarb[k][0]))
            else:
                listlablvarbtotl.append('%s [%s]' % (listlablvarb[k][0], listlablvarb[k][1]))
                
        listscalvarb = ['self'] * numbvarb
       
        if gdat.boolobjt:
            # metastable helium absorption
            path = gdat.pathbase + '/data/wasp107b_transmission_spectrum.dat'
            print('Reading from %s...' % path)
            arry = np.loadtxt(path, delimiter=',', skiprows=1)
            wlenwasp0107 = arry[:, 0]
            deptwasp0107 = arry[:, 1]
            deptstdvwasp0107 = arry[:, 2]
            
            print('')
            print('')
            print('')
            stdvnirs = 0.24e-2
            print('gdat.dicterrr[radiplan]')
            print(gdat.dicterrr['radiplan'])
            summgene(gdat.dicterrr['radiplan'])
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
                print('10**((-jmagstarwasp0107 + jmagstar) / 2.5)')
                print(10**((-jmagstarwasp0107 + jmagstar) / 2.5))
                print('duratranplan / duratranplanwasp0107')
                print('duratranplan / duratranplanwasp0107')
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
            figr, axis = plt.subplots(figsize=(8, 6))
            #axis.errorbar(wlenwasp0107, deptwasp0107, yerr=deptstdvwasp0107, ls='', ms=1, lw=1, marker='o', color='k', alpha=1)
            axis.errorbar(wlenwasp0107-10833, deptwasp0107*fact[0], yerr=deptstdvwasp0107*factstdv[0], ls='', ms=1, lw=1, marker='o', color='k', alpha=1)
            axis.set_xlabel(r'Wavelength - 10,833 [$\AA$]')
            axis.set_ylabel('Depth [\%]')
            plt.subplots_adjust(bottom=0.2, left=0.2)
            path = gdat.pathimag + 'heli_%s.%s' % (strgpdfn, gdat.strgplotextn)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()

        # optical magnitude vs number of planets
        if gdat.boolplot:
            gdat.pathimagmagt = gdat.pathimag + 'magt/'
            os.system('mkdir -p %s' % gdat.pathimagmagt)
            for b in range(4):
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
                    strgvarbmagt = 'msnrvmag'
                    lablxaxi = 'Relative mass SNR in the V band'
                    if gdat.boolobjt:
                        varbtarg = np.sqrt(10**(-gdat.vmagstar / 2.5)) / gdat.massstar**(2. / 3.)
                    varb = np.sqrt(10**(-dictpopl['vmagstar'] / 2.5)) / dictpopl['massstar']**(2. / 3.)
                if b == 3:
                    strgvarbmagt = 'msnrjmag'
                    lablxaxi = 'Relative mass SNR in the J band'
                    if gdat.boolobjt:
                        varbtarg = np.sqrt(10**(-gdat.vmagstar / 2.5)) / gdat.massstar**(2. / 3.)
                    varb = np.sqrt(10**(-dictpopl['jmagstar'] / 2.5)) / dictpopl['massstar']**(2. / 3.)
                for a in range(3):
                    figr, axis = plt.subplots(figsize=gdat.figrsize)
                    if a == 0:
                        indx = np.where(dictpopl['boolfrst'])[0]
                    if a == 1:
                        indx = np.where(dictpopl['boolfrst'] & (dictpopl['numbplanstar'] > 3))[0]
                    if a == 2:
                        indx = np.where(dictpopl['boolfrst'] & (dictpopl['numbplantranstar'] > 3))[0]
                    
                    if gdat.boolobjt and (b == 2 or b == 3):
                        normfact = max(varbtarg, np.amax(varb[indx]))
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
                    path = gdat.pathimagmagt + '%snumb_%d_%s_%s.%s' % (strgvarbmagt, a, strgpopl, strgpdfn, gdat.strgplotextn)
                    print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()

                    figr, axis = plt.subplots(figsize=gdat.figrsize)
                    axis.hist(varbnorm, 50)
                    if gdat.boolobjt:
                        axis.axvline(varbtargnorm, color='black', ls='--')
                        axis.text(0.9, 0.9, gdat.labltarg, size=8, color='black', transform=axis.transAxes, va='center', ha='center')
                    axis.set_ylabel(r'Number of systems')
                    axis.set_xlabel(lablxaxi)
                    plt.subplots_adjust(bottom=0.2)
                    path = gdat.pathimagmagt + 'hist%s_%d_%s_%s.%s' % (strgvarbmagt, a, strgpopl, strgpdfn, gdat.strgplotextn)
                    print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()
                
        indxinit = np.where(dictpopl['peri'] < 100)[0]
            
        # distribution plots
        for k, strgxaxi in enumerate(liststrgvarb):
            for m, strgyaxi in enumerate(liststrgvarb):
                if k >= m:
                    continue
                
                # temp
                #if not (strgxaxi == 'tsmm' or strgyaxi == 'tsmm'):
                #    continue
                # temp
                if not (strgxaxi == 'peri' and strgyaxi == 'inso'):
                    continue
                # temp
                #if not (strgxaxi == 'radistarsunn' and strgyaxi == 'radiplan' or strgxaxi == 'tmptplan' and strgyaxi == 'radistarsunn'):
                #    continue

                if gdat.boolobjt:
                    varbxaxithis = gdat.dicterrr[strgxaxi][0, :]
                    varbyaxithis = gdat.dicterrr[strgyaxi][0, :]
                    errrxaxithis = gdat.dicterrr[strgxaxi]
                    errryaxithis = gdat.dicterrr[strgyaxi]
                
                # finiteness cut
                #indxinit = np.where(np.isfinite(dictpopl[strgxaxi]) & np.isfinite(dictpopl[strgyaxi]))[0]
                
                for strgcuttradi in liststrgcuttradi:
                    
                    #if (strgcuttradi == 'rb04' or strgcuttradi == 'rb24') and (strgxaxi == 'radiplan' or strgyaxi == 'radiplan'):
                    #    continue

                    # ensure allr plots do not come with R_E units
                    if strgcuttradi == 'allr' and (strgxaxi == 'radiplaneart' or strgyaxi == 'radiplaneart'):
                        continue

                    for strgcuttmass in liststrgcuttmass:
                        
                        # impose radius cut
                        indx = np.intersect1d(indxpoplcuttradi[strgcuttradi], indxpoplcuttmass[strgcuttmass])
                        indx = np.intersect1d(indx, indxinit)
                        
                        # write out
                        if (k == 0 and m == 1) and strgcuttmass == 'allm':
                            
                            for y in range(1):
                                
                                if y == 0:
                                    strgstrgsort = 'tsmm'
                                else:
                                    strgstrgsort = 'esmm'
                                print('strgcuttradi')
                                print(strgcuttradi)
                                print('strgcuttmass')
                                print(strgcuttmass)
                                print('strgstrgsort')
                                print(strgstrgsort)
                                # sort
                                dicttemp = dict()
                                
                                for strgfeat, valu in dictpopl.items():
                                    if strgfeat.startswith('stdv'):
                                        continue
                                    if gdat.boolobjt:
                                        if strgfeat in gdat.dictfeatobjt.keys():
                                            print('strgfeat')
                                            print(strgfeat)
                                            print('dictpopl[strgfeat][indx]')
                                            summgene(dictpopl[strgfeat][indx])
                                            print('gdat.dictfeatobjt[strgfeat]')
                                            summgene(gdat.dictfeatobjt[strgfeat])
                                            print('')
                                            dicttemp[strgfeat] = np.concatenate([dictpopl[strgfeat][indx], gdat.dictfeatobjt[strgfeat]])
                                        else:
                                            dicttemp[strgfeat] = np.concatenate([dictpopl[strgfeat][indx], gdat.dicterrr[strgfeat][0, :]])
                                            #dicttemp[strgfeat] = np.concatenate([dictpopl['stdv' + strgfeat][indx], \
                                            #                                                            np.mean(gdat.dicterrr[strgfeat][1:3, :], 0)])
                                    else:
                                        dicttemp[strgfeat] = np.copy(dictpopl[strgfeat][indx])
                                indxgood = np.where(np.isfinite(dicttemp[strgstrgsort]))[0]
                                indxsort = np.argsort(dicttemp[strgstrgsort][indxgood])[::-1]
                                
                                path = gdat.pathdata + '%s_%s_%s_%s.csv' % (strgpopl, strgcuttradi, strgcuttmass, strgstrgsort)
                                objtfile = open(path, 'w')


                                #print('%20s, %12s, %12s, %12s, %12s, %12s, %12s, %12s, %12s' % \
                                
                                #('Name', 'TSM', 'e_TSM', 'e_TSM [%]', 'R [R_J]', 'M [M_J]', 'e_M [M_J]', 'T_p [K]', 'Jmag'))
                                strghead = '%20s, %12s, %12s, %12s, %12s, %12s, %12s, %12s, %12s\n' % \
                                            ('Name', 'TSM', 'e_TSM', 'e_TSM [%]', 'R [R_J]', 'M [M_J]', 'e_M [M_J]', 'T_p [K]', 'Jmag')
                                objtfile.write(strghead)
                                for l in indxgood[indxsort]:
                                    strgline = '%20s, %12.3f, %12.3f, %12.3f, %12.3f, %12.3f, %12.3f, %12.3f, %12.3f\n' % (dicttemp['nameplan'][l], dicttemp[strgstrgsort][l], \
                                    dicttemp['stdv' + strgstrgsort][l], 100. * dicttemp['stdv' + strgstrgsort][l] / dicttemp[strgstrgsort][l], \
                                    dicttemp['radiplan'][l], dicttemp['massplan'][l], dicttemp['stdvmassplan'][l], \
                                                    dicttemp['tmptplan'][l], dicttemp['jmagstar'][l])
                                    objtfile.write(strgline)
                                print('Writing to %s...' % path)
                                objtfile.close()

                        if gdat.boolplot:
                            # repeat, one without text, one with text
                            for b, strgtext in enumerate(liststrgtext):
                                if gdat.boolplot:
                                    figr, axis = plt.subplots(figsize=gdat.figrsize)
                                    
                                    ## population
                                    axis.errorbar(dictpopl[strgxaxi][indx], dictpopl[strgyaxi][indx], ls='', ms=1, lw=1, marker='o', color='k', alpha=0.5)
                                    if gdat.boolobjt:
                                        ## this system
                                        for j in gdat.indxplan:
                                            xdat = gdat.dicterrr[strgxaxi][0, j, None]
                                            ydat = gdat.dicterrr[strgyaxi][0, j, None]
                                            xerr = gdat.dicterrr[strgxaxi][1:3, j, None]
                                            yerr = gdat.dicterrr[strgyaxi][1:3, j, None]
                                            axis.errorbar(xdat, ydat, color=gdat.listcolrplan[j], lw=1, xerr=xerr, yerr=yerr, ls='', marker='o', ms=6)
                                            axis.text(0.85, 0.9 - j * 0.08, r'\textbf{%s}' % gdat.liststrgplan[j], color=gdat.listcolrplan[j], \
                                                                                                        va='center', ha='center', transform=axis.transAxes)
                                    
                                    # include text
                                    if b == 1:
                                        indxsortxaxi = np.argsort(dictpopl[strgxaxi][indx])
                                        indxsortyaxi = np.argsort(dictpopl[strgyaxi][indx])
                                        print('indxsortxaxi[:numbtext]')
                                        print(indxsortxaxi[:numbtext])
                                        print('indxsortxaxi[-numbtext:]')
                                        print(indxsortxaxi[-numbtext:])
                                        print('indxsortyaxi[:numbtext]')
                                        print(indxsortyaxi[:numbtext])
                                        print('indxsortyaxi[-numbtext:]')
                                        print(indxsortyaxi[-numbtext:])
                                        indxextr = np.concatenate([indxsortxaxi[:numbtext], indxsortxaxi[-numbtext:], \
                                                                            indxsortyaxi[:numbtext], indxsortyaxi[-numbtext:]])
                                        indxextr = np.unique(indxextr)
                                        print('indxextr')
                                        summgene(indxextr)
                                        for l in indxextr:
                                            objttext = axis.text(dictpopl[strgxaxi][indx][l], dictpopl[strgyaxi][indx][l], \
                                                                    '%s, R=%g, M=%g' % (dictpopl['nameplan'][indx][l], dictpopl['radiplan'][indx][l], \
                                                                        dictpopl['massplan'][indx][l]), size=1, ha='center', va='center')
                                    
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

                                    if strgxaxi == 'radiplan' and strgyaxi == 'massplan':
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
                                    
                                    if strgyaxi == 'vesc':
                                        axis.set_ylim([0., 60.])

                                    axis.set_xlabel(listlablvarbtotl[k])
                                    axis.set_ylabel(listlablvarbtotl[m])
                                    plt.subplots_adjust(left=0.2)
                                    plt.subplots_adjust(bottom=0.2)
                                    path = gdat.pathimag + 'feat_%s_%s_%s_%s_%s_%s_%s.%s' % \
                                                    (strgxaxi, strgyaxi, strgcuttradi, strgcuttmass, strgpopl, strgtext, strgpdfn, gdat.strgplotextn)
                                    print('Writing to %s...' % path)
                                    plt.savefig(path)
                                    plt.close()

   
def bdtr_wrap(gdat, epocmask, perimask, duramask):
    
    gdat.listobjtspln = [[[] for y in gdat.indxchun[p]] for p in gdat.indxinst]
    gdat.indxsplnregi = [[[] for y in gdat.indxchun[p]] for p in gdat.indxinst]
    gdat.listindxtimeregi = [[[] for y in gdat.indxchun[p]] for p in gdat.indxinst]
    gdat.indxtimeregioutt = [[[] for y in gdat.indxchun[p]] for p in gdat.indxinst]
    gdat.listarrylcurbdtr = [[[] for y in gdat.indxchun[p]] for p in gdat.indxinst]
    gdat.arrylcurbdtr = [[] for p in gdat.indxinst]
    for p in gdat.indxinst:
        for y in gdat.indxchun[p]:
            if gdat.boolbdtr:
                gdat.lcurbdtrregi, gdat.listindxtimeregi[p][y], gdat.indxtimeregioutt[p][y], gdat.listobjtspln[p][y], timeedge = \
                                 tesstarg.util.bdtr_lcur(gdat.listarrylcur[p][y][:, 0], gdat.listarrylcur[p][y][:, 1], weigsplnbdtr=gdat.weigsplnbdtr, \
                                                            epocmask=epocmask, perimask=perimask, duramask=duramask, \
                                                            verbtype=gdat.verbtype, durabrek=gdat.durabrek, ordrspln=gdat.ordrspln, bdtrtype=gdat.bdtrtype)
                gdat.listarrylcurbdtr[p][y] = np.copy(gdat.listarrylcur[p][y])
                gdat.listarrylcurbdtr[p][y][:, 1] = np.concatenate(gdat.lcurbdtrregi)
                numbsplnregi = len(gdat.lcurbdtrregi)
                gdat.indxsplnregi[p][y] = np.arange(numbsplnregi)
            else:
                gdat.listarrylcurbdtr[p][y] = gdat.listarrylcur[p][y]
        # merge chuncks
        gdat.arrylcurbdtr[p] = np.concatenate(gdat.listarrylcurbdtr[p], axis=0)


def init( \
         ticitarg=None, \
         strgmast=None, \
         toiitarg=None, \
         
         # a string for the label of the target
         labltarg=None, \
         
         # a string for the folder name and file name extensions
         strgtarg=None, \
         
         # mode of operation
         booldatatser=True, \
         
         boolplot=False, \

         # input
         boolforcoffl=False, \
         listpathdatainpt=None, \
         ## type of light curve to be used for analysis: pdcc, sapp, lygo
         datatype=None, \
         ## maximum number of stars to fit in lygos
         maxmnumbstarlygo=1, \
         # list of strings indicating instruments
         liststrginst=['TESS'], \
         # list of labels indicating instruments
         listlablinst=['TESS'], \
         # list of strings indicating chunks
         liststrgchun=None, \
         # list of chunk indices for each instrument
         listindxchuninst=None, \

         # preprocessing
         boolclip=False, \
         ## dilution: None (no correction), 'lygo' for estimation via lygos, or float value
         dilu=None, \
         ## baseline detrending
         durabrek=1., \
         ordrspln=3, \
         bdtrtype='spln', \
         weigsplnbdtr=1., \
         durakernbdtrmedi=1., \
         ## Boolean flag to mask bad data
         boolmaskqual=True, \
         ## time limits to mask
         listlimttimemask=None, \
    
         # planet search
         ## maximum number of planets for TLS
         maxmnumbplantlss=None, \
         
         liststrgplan=None, \
         
         # include the ExoFOP catalog in the comparisons to exoplanet population
         boolexof=True, \

         # model
         # priors
         ## type of priors used for allesfitter
         priotype=None, \
         ## prior values
         rratprio=None, \
         rsmaprio=None, \
         epocprio=None, \
         periprio=None, \
         cosiprio=None, \
         ecosprio=None, \
         esinprio=None, \
         rvsaprio=None, \
         #massplanprio=None, \
         #ecceprio=None, \
         
         rratstdvprio=None, \
         rsmastdvprio=None, \
         epocstdvprio=None, \
         peristdvprio=None, \
         cosistdvprio=None, \
         ecosstdvprio=None, \
         esinstdvprio=None, \
         rvsastdvprio=None, \
         #massplanstdvprio=None, \
         #eccestdvprio=None, \

         radistar=None, \
         massstar=None, \
         tmptstar=None, \
         stdvradistar=None, \
    
         boolphascurv=False, \
         
         # type of inference, alle for allesfitter, trap for trapezoidal fit
         infetype='alle', \

         # allesfitter settings
         boolallebkgdgaus=False, \
         boolalleorbt=True, \
         ## allesfitter analysis type
         typeparacalctype = 'ther', \
        
         # output
         boolwritplan=True, \
         ## plotting
         makeprioplot=True, \
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
    
    # parse inpt
    ## determine whether pexo will work on a specific target
    if gdat.ticitarg is None and gdat.strgmast is None and gdat.toiitarg is None:
        gdat.boolobjt = False
    else:
        gdat.boolobjt = True
    
    # settings
    ## plotting
    if gdat.boolobjt:
        if gdat.booldatatser:
            gdat.timetess = 2457000.
    
        if gdat.ticitarg is not None and gdat.strgmast is not None:
            raise Exception('')
        if gdat.ticitarg is not None and gdat.toiitarg is not None:
            raise Exception('')
        if gdat.strgmast is not None and gdat.toiitarg is not None:
            raise Exception('')
        if gdat.ticitarg is not None:
            gdat.inpttype = 'tici'
        if gdat.strgmast is not None:
            gdat.inpttype = 'mast'
        if gdat.toiitarg is not None:
            gdat.inpttype = 'toii'

    # plotting
    gdat.strgplotextn = 'pdf'
    gdat.figrsize = [4., 3.]
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
    gdat.factmjme = 317.907
    gdat.factmsmj = 1048.
    gdat.factrjre = 11.2
    gdat.factrsrj = 9.95
    gdat.factaurj = 2093.
    print('Target identifier inputs:')
    print('ticitarg')
    print(ticitarg)
    print('strgmast')
    print(strgmast)
    print('toiitarg')
    print(toiitarg)
    
    gdat.liststrgpopl = ['exarcomp']
    if gdat.boolexof:
        gdat.liststrgpopl += ['exof']
    gdat.numbpopl = len(gdat.liststrgpopl)
    
    if gdat.boolobjt:
        dictexartarg = retr_exarcomp(gdat, strgtarg=gdat.strgmast)
        gdat.boolexar = gdat.strgmast is not None and dictexartarg is not None
    
        # determine the type of prior to be fed into allesfitter
        if gdat.priotype is None:
            if gdat.inpttype == 'toii':
                gdat.priotype = 'exof'
            elif gdat.boolexar:
                gdat.priotype = 'exar'
        print('gdat.priotype')
        print(gdat.priotype)
    
        if gdat.boolexar:
            print('The planet name was found in the NASA Exoplanet Archive "composite" table.')
            # stellar properties
            
            if gdat.periprio is None:
                gdat.periprio = dictexartarg['peri']
            gdat.smaxprio = dictexartarg['smax']
            gdat.massplanprio = dictexartarg['massplan']
            
        else:
            print('The planet name was *not* found in the Exoplanet Archive "composite" table.')
    
        if gdat.booldatatser:
            gdat.numbinst = len(gdat.liststrginst)
            gdat.indxinst = np.arange(gdat.numbinst)

        # read the NASA Exoplanet Archive planets
        path = gdat.pathbase + 'data/PS_2020.09.15_17.21.46.csv'
        print('Reading %s...' % path)
        objtexarplan = pd.read_csv(path, skiprows=302, low_memory=False)
        indx = np.where(objtexarplan['hostname'].values == gdat.strgmast)[0]
        if indx.size == 0:
            print('The planet name was *not* found in the NASA Exoplanet Archive "planets" table.')
            gdat.boolexar = False
        else:
            gdat.deptprio = objtexarplan['pl_trandep'][indx].values
            if gdat.cosiprio is None:
                gdat.cosiprio = np.cos(objtexarplan['pl_orbincl'][indx].values / 180. * np.pi)
            if gdat.epocprio is None:
                gdat.epocprio = objtexarplan['pl_tranmid'][indx].values # [days]
            gdat.duraprio = objtexarplan['pl_trandur'][indx].values # [days]
    
        if gdat.inpttype == 'toii':
            print('A TOI number is provided. Retreiving the TCE attributes from ExoFOP-TESS...')
            
            # find the indices of the target in the TOI catalog
            path = gdat.pathbase + 'data/exofop_toilists_20200916.csv'
            print('Reading from %s...' % path)
            objtexof = pd.read_csv(path, skiprows=0)
            gdat.strgtoiibase = str(gdat.toiitarg)
            indx = []
            for k, strg in enumerate(objtexof['TOI']):
                if str(strg).split('.')[0] == gdat.strgtoiibase:
                    indx.append(k)
            indx = np.array(indx)
            if indx.size == 0:
                print('Did not find the TOI in the ExoFOP-TESS TOI list.')
                raise Exception('')
            
            if gdat.ticitarg is not None:
                raise Exception('')
            else:
                gdat.ticitarg = objtexof['TIC ID'].values[indx[0]]
            gdat.strgmast  = 'TIC %d' % gdat.ticitarg
            if gdat.epocprio is None:
                gdat.epocprio = objtexof['Epoch (BJD)'].values[indx]
            if gdat.periprio is None:
                gdat.periprio = objtexof['Period (days)'].values[indx]
            gdat.deptprio = objtexof['Depth (ppm)'].values[indx] * 1e-6
            gdat.duraprio = objtexof['Duration (hours)'].values[indx] / 24. # [days]
            if gdat.cosiprio is None:
                gdat.cosiprio = np.zeros_like(gdat.epocprio)
            
            if gdat.strgtarg is None:
                gdat.strgtarg = 'TOI' + gdat.strgtoiibase
            if gdat.labltarg is None:
                gdat.labltarg = 'TOI ' + gdat.strgtoiibase
        if gdat.strgmast is not None:
            if gdat.strgtarg is None:
                gdat.strgtarg = ''
                strgtargsplt = gdat.strgmast.split(' ')
                for strgtemp in strgtargsplt:
                    gdat.strgtarg += strgtemp
            if gdat.labltarg is None:
                gdat.labltarg = gdat.strgmast
        if gdat.ticitarg is not None:
            if gdat.strgtarg is None:
                gdat.strgtarg = 'TIC%d' % gdat.ticitarg
            if gdat.labltarg is None:
                gdat.labltarg = 'TIC %d' % gdat.ticitarg
        print('gdat.strgtarg')
        print(gdat.strgtarg)
        print('gdat.labltarg')
        print(gdat.labltarg)
        if gdat.priotype == 'inpt':
            if gdat.rratprio is None:
                gdat.rratprio = 0.1 + np.zeros(gdat.numbplan)
            if gdat.rsmaprio is None:
                gdat.rsmaprio = 0.1 + np.zeros(gdat.numbplan)
            if gdat.cosiprio is None:
                gdat.cosiprio = np.zeros(gdat.numbplan)
            gdat.duraprio = tesstarg.util.retr_dura(gdat.periprio, gdat.rsmaprio, gdat.cosiprio)
            gdat.deptprio = gdat.rratprio**2
        
        gdat.numbplan = None
        
        # check MAST
        print('gdat.strgmast')
        print(gdat.strgmast)
        if gdat.strgmast is None:
            print('gdat.strgmast was not provided as input. Using the TIC ID to construct gdat.strgmast.')
            gdat.strgmast = 'TIC %d' % gdat.ticitarg
        
        if not gdat.boolforcoffl:
            catalogData = astroquery.mast.Catalogs.query_object(gdat.strgmast, catalog='TIC', radius='40s')
            if catalogData[0]['dstArcSec'] > 0.1:
                print('The nearest source is more than 0.1 arcsec away from the target!')
            print('Found the target on MAST!')
            rasc = catalogData[0]['ra']
            decl = catalogData[0]['dec']
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
            print('rasc')
            print(rasc)
            print('decl')
            print(decl)
            
            # check that the closest TIC to a given TIC is itself
            if gdat.inpttype == 'tici' and gdat.ticitarg is not None:
                strgtici = '%s' % catalogData[0]['ID']
                if strgtici != str(gdat.ticitarg):
                    print('strgtici')
                    print(strgtici)
                    print('gdat.ticitarg')
                    print(gdat.ticitarg)
                    raise Exception('')
        gdat.pathobjt = gdat.pathbase + '%s/' % gdat.strgtarg
    else:
        gdat.pathobjt = gdat.pathbase + 'gene/'
    gdat.pathdata = gdat.pathobjt + 'data/'
    gdat.pathimag = gdat.pathobjt + 'imag/'
    os.system('mkdir -p %s' % gdat.pathdata)
    os.system('mkdir -p %s' % gdat.pathimag)

    if gdat.boolobjt and gdat.booldatatser:
        print('gdat.datatype')
        print(gdat.datatype)
        if gdat.listpathdatainpt is None:
            gdat.arrylcur = [[]]
            gdat.arrylcursapp = [[]]
            gdat.arrylcurpdcc = [[]]
            gdat.listarrylcur = [[]]
            gdat.listarrylcursapp = [[]]
            gdat.listarrylcurpdcc = [[]]
            gdat.datatype, gdat.arrylcur[0], gdat.arrylcursapp[0], gdat.arrylcurpdcc[0], gdat.listarrylcur[0], gdat.listarrylcursapp[0], \
                                       gdat.listarrylcurpdcc[0], gdat.listisec, gdat.listicam, gdat.listiccd = \
                                            tesstarg.util.retr_data(gdat.datatype, gdat.strgmast, gdat.pathobjt, boolmaskqual=gdat.boolmaskqual, \
                                            labltarg=gdat.labltarg, strgtarg=gdat.strgtarg, ticitarg=gdat.ticitarg, maxmnumbstarlygo=gdat.maxmnumbstarlygo)
            gdat.numbchun = np.array([len(gdat.listarrylcur[0])])
        
        else:
            gdat.datatype = 'inpt'
            gdat.arrylcur = [[] for p in gdat.indxinst]
            gdat.numbchun = np.empty(gdat.numbinst, dtype=int)
            for p in gdat.indxinst:
                gdat.numbchun[p] = len(gdat.listpathdatainpt[p])
        print('gdat.numbchun')
        print(gdat.numbchun)
        print('gdat.datatype')
        print(gdat.datatype)
        
        # determine whether to baseline-detrend
        gdat.boolbdtr = gdat.datatype != 'pdcc'
            
        gdat.indxchun = [[] for p in gdat.indxinst]
        for p in gdat.indxinst:
            gdat.indxchun[p] = np.arange(gdat.numbchun[p], dtype=int)

        if gdat.listpathdatainpt is not None:
            gdat.listarrylcur = [[[] for y in gdat.indxchun[p]] for p in gdat.indxinst]
            for p in gdat.indxinst:
                for y in gdat.indxchun[p]:
                    print('gdat.listpathdatainpt[p][y]')
                    print(gdat.listpathdatainpt[p][y])
                    arry = np.loadtxt(gdat.listpathdatainpt[p][y], delimiter=',', skiprows=1)
                    gdat.listarrylcur[p][y] = np.empty((arry.shape[0], 3))
                    gdat.listarrylcur[p][y][:, 0:2] = arry[:, 0:2]
                    gdat.listarrylcur[p][y][:, 2] = 1e-4 * arry[:, 1]
                    indx = np.argsort(gdat.listarrylcur[p][y][:, 0])
                    gdat.listarrylcur[p][y] = gdat.listarrylcur[p][y][indx, :]
                    indx = np.where(gdat.listarrylcur[p][y][:, 1] < 1e6)[0]
                    gdat.listarrylcur[p][y] = gdat.listarrylcur[p][y][indx, :]
                    gdat.listisec = None
                gdat.arrylcur[p] = np.concatenate(gdat.listarrylcur[p])
        if gdat.listindxchuninst is None:
            gdat.listindxchuninst = [gdat.indxchun]

        if gdat.liststrgchun is None:
            gdat.liststrgchun = []
            for p in gdat.indxinst:
                for o in gdat.indxchun[p]:
                    strgchun = 'chu%d' % o
                    gdat.liststrgchun.append(strgchun)

        for p in gdat.indxinst:
            for y in gdat.indxchun[p]:
                # plot raw light curve
                figr, axis = plt.subplots(1, 1, figsize=gdat.figrsizeydobskin)
                axis.plot(gdat.listarrylcur[p][y][:, 0] - gdat.timetess, gdat.listarrylcur[p][y][:, 1], color='grey', marker='.', ls='', ms=1)
                axis.set_xlabel('Time [BJD - 2457000]')
                axis.set_ylabel('Relative Flux')
                plt.subplots_adjust(bottom=0.2)
                path = gdat.pathimag + 'lcurraww%s%s.%s' % (gdat.liststrginst[p], gdat.liststrgchun[y], gdat.strgplotextn)
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()

        gdat.arrylcurtotl = np.concatenate(gdat.arrylcur, axis=0)
        listperilspe = tesstarg.util.plot_lspe(gdat.pathimag, gdat.arrylcurtotl)
        print('listperilspe')
        print(listperilspe)

        if gdat.priotype == 'tlss':
            epocmask = None
            perimask = None
            duramask = None
            bdtr_wrap(gdat, epocmask, perimask, duramask)
            dicttlss = tesstarg.util.exec_tlss(gdat.arrylcurbdtrtotl, gdat.pathimag, maxmnumbplantlss=gdat.maxmnumbplantlss, \
                                            strgplotextn=gdat.strgplotextn, figrsize=gdat.figrsizeydob, figrsizeydobskin=gdat.figrsizeydobskin)
            if gdat.epocprio is None:
                gdat.epocprio = dicttlss['epoc']
            if gdat.periprio is None:
                gdat.periprio = dicttlss['peri']
            gdat.deptprio = dicttlss['dept']
            gdat.duraprio = dicttlss['dura']
            
            gdat.rratprio = np.sqrt(gdat.deptprio)
            gdat.rsmaprio = np.sin(np.pi * gdat.duraprio / gdat.periprio)
        
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
        print('gdat.duramask')
        print(gdat.duramask)
        
        # baseline-detrend with the ephemeris prior
        bdtr_wrap(gdat, gdat.epocmask, gdat.perimask, gdat.duramask)
        
        # sigma-clip the light curve
        # temp -- this does not work properly!
        for p in gdat.indxinst:
            for y in gdat.indxchun[p]:
                if gdat.boolclip:
                    lcurclip, lcurcliplowr, lcurclipuppr = scipy.stats.sigmaclip(gdat.listarrylcurbdtr[p][y][:, 1], low=5., high=5.)
                    print('Clipping the light curve at %g and %g...' % (lcurcliplowr, lcurclipuppr))
                    indx = np.where((gdat.listarrylcurbdtr[p][y][:, 1] < lcurclipuppr) & (gdat.listarrylcurbdtr[p][y][:, 1] > lcurcliplowr))[0]
                    gdat.listarrylcurbdtr[p][y] = gdat.listarrylcurbdtr[p][y][indx, :]
        
                path = gdat.pathdata + 'arrylcurbdtr%s%s.csv' % (gdat.liststrginst[p], gdat.liststrgchun[y])
                print('Writing to %s...' % path)
                np.savetxt(path, gdat.listarrylcurbdtr[p][y], delimiter=',', header='time,flux,flux_err')
        
        # write baseline-detrended light curve
        for p in gdat.indxinst:
            path = gdat.pathdata + 'arrylcurbdtr%s.csv' % (gdat.liststrginst[p])
            print('Writing to %s...' % path)
            np.savetxt(path, gdat.arrylcurbdtr[p], delimiter=',', header='time,flux,flux_err')
        
        gdat.time = [[] for p in gdat.indxinst]
        gdat.indxtime = [[] for p in gdat.indxinst]
        gdat.numbtime = np.empty(gdat.numbinst, dtype=int)
        for p in gdat.indxinst:
            gdat.time[p] = gdat.arrylcurbdtr[p][:, 0]
            gdat.numbtime[p] = gdat.time[p].size
            gdat.indxtime[p] = np.arange(gdat.numbtime[p])
    
    if gdat.boolobjt:
        #if gdat.duraprio is None:
        #    gdat.duraprio = tesstarg.util.retr_dura(gdat.periprio, gdat.rsmaprio, gdat.cosiprio)
        
        if gdat.rratprio is None:
            gdat.rratprio = np.sqrt(gdat.deptprio)

        if gdat.rsmaprio is None:
            gdat.rsmaprio = np.sqrt(np.sin(np.pi * gdat.duraprio / gdat.periprio)**2 + gdat.cosiprio**2)
            
        if gdat.booldatatser:
            # plot raw data
            if gdat.datatype == 'pdcc' or gdat.datatype == 'sapp':
                for y in gdat.indxchun[0]:
                    path = gdat.pathdata + gdat.liststrgchun[y] + '_SAP.csv'
                    print('Writing to %s...' % path)
                    np.savetxt(path, gdat.arrylcursapp[0], delimiter=',', header='time,flux,flux_err')
                    path = gdat.pathdata + gdat.liststrgchun[y] + '_PDCSAP.csv'
                    print('Writing to %s...' % path)
                    np.savetxt(path, gdat.arrylcurpdcc[0], delimiter=',', header='time,flux,flux_err')
                
                # plot PDCSAP and SAP light curves
                figr, axis = plt.subplots(2, 1, figsize=gdat.figrsizeydob)
                axis[0].plot(gdat.arrylcursapp[0][:, 0] - gdat.timetess, gdat.arrylcursapp[0][:, 1], color='k', marker='.', ls='', ms=1)
                if listlimttimemask is not None:
                    axis[0].plot(gdat.arrylcursapp[0][listindxtimegood, 0] - gdat.timetess, \
                                                    gdat.arrylcursapp[0][listindxtimegood, 1], color='k', marker='.', ls='', ms=1)
                axis[1].plot(gdat.arrylcurpdcc[0][:, 0] - gdat.timetess, gdat.arrylcurpdcc[0][:, 1], color='k', marker='.', ls='', ms=1)
                if listlimttimemask is not None:
                    axis[1].plot(gdat.arrylcurpdcc[0][listindxtimegood, 0] - gdat.timetess, \
                                                    gdat.arrylcurpdcc[0][listindxtimegood, 1], color='k', marker='.', ls='', ms=1)
                #axis[0].text(.97, .97, 'SAP', transform=axis[0].transAxes, size=20, color='r', ha='right', va='top')
                #axis[1].text(.97, .97, 'PDC', transform=axis[1].transAxes, size=20, color='r', ha='right', va='top')
                axis[1].set_xlabel('Time [BJD - 2457000]')
                for a in range(2):
                    axis[a].set_ylabel('Relative Flux')
                
                #for j in gdat.indxplan:
                #    colr = gdat.listcolrplan[j]
                #    axis[1].plot(gdat.arrylcurpdcc[0][gdat.listindxtimetran[j], 0] - gdat.timetess, gdat.arrylcurpdcc[0][gdat.listindxtimetran[j], 1], \
                #                                                                                                         color=colr, marker='.', ls='', ms=1)
                plt.subplots_adjust(hspace=0.)
                path = gdat.pathimag + 'lcurspoc.%s' % gdat.strgplotextn
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()
            
            # injection recovery test
        
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
        print('gdat.massstar [M_S]')
        print(gdat.massstar / gdat.factmsmj)
        print('gdat.stdvmassstar [M_S]')
        print(gdat.stdvmassstar / gdat.factmsmj)
        print('gdat.tmptstar')
        print(gdat.tmptstar)
        print('gdat.stdvtmptstar')
        print(gdat.stdvtmptstar)
        
        gdat.numbplan = gdat.epocprio.size
        gdat.indxplan = np.arange(gdat.numbplan)
        
        if gdat.ecosprio is None:
            gdat.ecosprio = np.zeros(gdat.numbplan)
        if gdat.esinprio is None:
            gdat.esinprio = np.zeros(gdat.numbplan)
        if gdat.rvsaprio is None:
            gdat.rvsaprio = np.zeros(gdat.numbplan)
        #if gdat.massplanprio is None:
        #    gdat.massplanprio = np.zeros(gdat.numbplan)
        #if gdat.ecceprio is None:
        #    gdat.ecceprio = np.zeros(gdat.numbplan)
        
        if gdat.rratstdvprio is None:
            gdat.rratstdvprio = np.zeros(gdat.numbplan)
        if gdat.rsmastdvprio is None:
            gdat.rsmastdvprio = np.zeros(gdat.numbplan)
        if gdat.epocstdvprio is None:
            gdat.epocstdvprio = np.zeros(gdat.numbplan)
        if gdat.peristdvprio is None:
            gdat.peristdvprio = np.zeros(gdat.numbplan)
        if gdat.cosistdvprio is None:
            gdat.cosistdvprio = np.zeros(gdat.numbplan)
        if gdat.ecosstdvprio is None:
            gdat.ecosstdvprio = np.zeros(gdat.numbplan)
        if gdat.esinstdvprio is None:
            gdat.esinstdvprio = np.zeros(gdat.numbplan)
        if gdat.rvsastdvprio is None:
            gdat.rvsastdvprio = np.zeros(gdat.numbplan)
        #if gdat.massplanprio is None:
        #    gdat.massplanprio = np.zeros_like(gdat.epocprio)
        #if gdat.ecceprio is None:
        #    gdat.ecceprio = np.zeros_like(gdat.epocprio)
        
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
        #print('gdat.massplanprio')
        #print(gdat.massplanprio)
        #print('gdat.ecceprio')
        #print(gdat.ecceprio)
        #print('gdat.deptprio')
        #print(gdat.deptprio)
        #print('gdat.duraprio')
        #print(gdat.duraprio)
        

        if not np.isfinite(gdat.rratprio).all():
            raise Exception('')
        if not np.isfinite(gdat.rsmaprio).all():
            raise Exception('')
        if not np.isfinite(gdat.epocprio).all():
            raise Exception('')
        if not np.isfinite(gdat.periprio).all():
            raise Exception('')
        if not np.isfinite(gdat.cosiprio).all():
            raise Exception('')
        if not np.isfinite(gdat.ecosprio).all():
            raise Exception('')
        if not np.isfinite(gdat.esinprio).all():
            raise Exception('')
        if not np.isfinite(gdat.rvsaprio).all():
            raise Exception('')
    
    # grab all exoplanet properties
    ## ExoFOP
    if gdat.boolexof:
        gdat.dictexof = retr_exof(gdat)
    ## NASA Exoplanet Archive
    gdat.dictexarcomp = retr_exarcomp(gdat)
    numbplanexar = gdat.dictexarcomp['radiplan'].size
    gdat.indxplanexar = np.arange(numbplanexar)
    ### augment the catalog
    gdat.dictexarcomp['vesc'] = tesstarg.util.retr_vesc(gdat.dictexarcomp['massplan'], gdat.dictexarcomp['radiplan'])
    
    if gdat.boolobjt:
        # settings
        if gdat.liststrgplan is None:
            gdat.liststrgplan = ['b', 'c', 'd', 'e', 'f', 'g'][:gdat.numbplan]
        print('Planet letters: ')
        print(gdat.liststrgplan)
        if ''.join(gdat.liststrgplan) != ''.join(sorted(gdat.liststrgplan)):
            print('Provided planet letters are not in order. Changing the TCE order to respect the letter order in plots (b, c, d, e)...')
            gdat.indxplan = np.argsort(np.array(gdat.liststrgplan))
        print('gdat.indxplan') 
        print(gdat.indxplan)

        gdat.liststrgplanfull = np.empty(gdat.numbplan, dtype='object')
        print('gdat.liststrgplan')
        print(gdat.liststrgplan)
        print('gdat.numbplan')
        print(gdat.numbplan)
        for j in gdat.indxplan:
            gdat.liststrgplanfull[j] = gdat.labltarg + ' ' + gdat.liststrgplan[j]

        gdat.listcolrplan = ['red', 'green', 'orange', 'magenta', 'yellow', 'cyan']
        
        if gdat.booldatatser:
            if gdat.dilu == 'lygo':
                print('Calculating the contamination ratio...')
                gdat.contrati = lygos.calc_contrati()

        if gdat.boolexar:
            ## expected ellipsoidal variation (EV) and Doppler beaming (DB)
            print('Predicting the ellipsoidal variation and Doppler beaming amplitudes...')
            ### EV
            #### limb and gravity darkening coefficients from Claret2017
            
            if gdat.massplanprio is not None:
                u = 0.4
                g = 0.2
                alphelli = 0.15 * (15 + u) * (1 + g) / (3 - u)
                deptelli = alphelli * gdat.massplanprio * np.sin(gdat.inclprio / 180. * np.pi)**2 / gdat.massstar * (gdat.radistar / gdat.smaxprio)**3
                ### DB
                kkrv = 181. # [m/s]
                binswlenbeam = np.linspace(0.6, 1., 101)
                meanwlenbeam = (binswlenbeam[1:] + binswlenbeam[:-1]) / 2.
                diffwlenbeam = (binswlenbeam[1:] - binswlenbeam[:-1]) / 2.
                x = 2.248 / meanwlenbeam
                f = .25 * x * np.exp(x) / (np.exp(x) - 1.)
                deptbeam = 4. * kkrv / 3e8 * np.sum(diffwlenbeam * f)
                
                print('Expected Doppler beaming amplitude:')
                print(deptbeam)
                print('Expected EV amplitude:')
                print(deptelli)
        
        if gdat.booldatatser:
            for p in gdat.indxinst:
                for y in gdat.indxchun[p]:
                    if listlimttimemask is not None:
                        # mask the data
                        print('Masking the data...')
                        numbmask = listlimttimemask.shape[0]
                        listindxtimemask = []
                        for k in range(numbmask):
                            indxtimemask = np.where((gdat.listarrylcurbdtr[p][y][:, 0] < listlimttimemask[k, 1]) & (gdat.listarrylcurbdtr[p][y][:, 0] > listlimttimemask[k, 0]))[0]
                            listindxtimemask.append(indxtimemask)
                        listindxtimemask = np.concatenate(listindxtimemask)
                        listindxtimegood = np.setdiff1d(gdat.indxtime, listindxtimemask)
                        gdat.listarrylcurbdtr[p][y] = gdat.listarrylcurbdtr[p][y][listindxtimegood, :]
            
            # correct for dilution
            #print('Correcting for dilution!')
            #if gdat.dilucorr is not None:
            #    gdat.arrylcurdilu = np.copy(gdat.listarrylcurbdtr[p][y])
            #if gdat.dilucorr is not None:
            #    gdat.arrylcurdilu[:, 1] = 1. - gdat.dilucorr * (1. - gdat.listarrylcurbdtr[p][y][:, 1])
            #gdat.arrylcurdilu[:, 1] = 1. - gdat.contrati * gdat.contrati * (1. - gdat.listarrylcurbdtr[p][y][:, 1])
            #if gdat.dilucorr is not NoneL
            #    gdat.listarrylcurbdtr[p][y] = np.copy(gdat.arrylcurdilu) 
            #    figr, axis = plt.subplots(1, 1, figsize=gdat.figrsizeydob)
            #    axis.plot(gdat.arrylcurdilu[:, 0] - gdat.timetess, gdat.arrylcurdilu[:, 1], color='grey', marker='.', ls='', ms=1)
            #    for j in gdat.indxplan:
            #        colr = gdat.listcolrplan[j]
            #        axis.plot(gdat.arrylcurdilu[gdat.listindxtimetran[j], 0] - gdat.timetess, gdat.arrylcurdilu[gdat.listindxtimetran[j], 1], \
            #                                                                                                color=colr, marker='.', ls='', ms=1)
            #    axis.set_xlabel('Time [BJD - 2457000]')
            #    axis.set_ylabel('Relative Flux')
            #    plt.subplots_adjust(hspace=0.)
            #    path = gdat.pathimag + 'lcurdilu.%s' % (gdat.strgplotextn)
            #    print('Writing to %s...' % path)
            #    plt.savefig(path)
            #    plt.close()
                
            gdat.numbbinspcurfine = 1000
            gdat.numbbinspcur = 100
            gdat.numbbinslcur = 1000
                
            ## bin the light curve
            gdat.listarrylcurbdtrbind = [[[] for y in gdat.indxchun[p]] for p in gdat.indxinst]
            for p in gdat.indxinst:
                for y in gdat.indxchun[p]:
                    gdat.listarrylcurbdtrbind[p][y] = tesstarg.util.rebn_lcur(gdat.listarrylcurbdtr[p][y], gdat.numbbinslcur)
                    
                    path = gdat.pathdata + 'arrylcurbdtrbind%s%s.csv' % (gdat.liststrginst[p], gdat.liststrgchun[y])
                    print('Writing to %s' % path)
                    np.savetxt(path, gdat.listarrylcurbdtrbind[p][y], delimiter=',', header='time,flux,flux_err')
                    
            ## phase-fold and save the baseline-detrended light curve
            gdat.arrypcurprimbdtr = [[[] for j in gdat.indxplan] for p in gdat.indxinst]
            gdat.arrypcurprimbdtrbind = [[[] for j in gdat.indxplan] for p in gdat.indxinst]
            gdat.arrypcurprimbdtrbindfine = [[[] for j in gdat.indxplan] for p in gdat.indxinst]
            gdat.arrypcurquadbdtr = [[[] for j in gdat.indxplan] for p in gdat.indxinst]
            gdat.arrypcurquadbdtrbind = [[[] for j in gdat.indxplan] for p in gdat.indxinst]
            gdat.arrypcurquadbdtrbindfine = [[[] for j in gdat.indxplan] for p in gdat.indxinst]
            for p in gdat.indxinst:
                for j in gdat.indxplan:
                    gdat.arrypcurprimbdtr[p][j] = tesstarg.util.fold_lcur(gdat.listarrylcurbdtr[p][y], gdat.epocprio[j], gdat.periprio[j])
                    gdat.arrypcurprimbdtrbind[p][j] = tesstarg.util.rebn_lcur(gdat.arrypcurprimbdtr[p][j], gdat.numbbinspcur)
                    gdat.arrypcurprimbdtrbindfine[p][j] = tesstarg.util.rebn_lcur(gdat.arrypcurprimbdtr[p][j], gdat.numbbinspcurfine)
                    gdat.arrypcurquadbdtr[p][j] = tesstarg.util.fold_lcur(gdat.listarrylcurbdtr[p][y], gdat.epocprio[j], gdat.periprio[j], phasshft=0.25)
                    gdat.arrypcurquadbdtrbind[p][j] = tesstarg.util.rebn_lcur(gdat.arrypcurquadbdtr[p][j], gdat.numbbinspcur)
                    gdat.arrypcurquadbdtrbindfine[p][j] = tesstarg.util.rebn_lcur(gdat.arrypcurquadbdtr[p][j], gdat.numbbinspcurfine)
                    
                    # write (good for Vespa)
                    path = gdat.pathdata + 'arrypcurprimbdtrbindpla%d%s.csv' % (j, gdat.liststrginst[p])
                    print('Writing to %s...' % path)
                    temp = np.copy(gdat.arrypcurprimbdtrbind[p][j])
                    temp[:, 0] *= gdat.periprio[j]
                    np.savetxt(path, temp, delimiter=',')
                
                # determine time mask
                gdat.listindxtimeoutt = [[[] for p in gdat.indxinst] for j in gdat.indxplan]
                gdat.listindxtimetran = [[[] for p in gdat.indxinst] for j in gdat.indxplan]
                gdat.listindxtimetranchun = [[[[] for y in gdat.indxchun[p]] for p in gdat.indxinst] for j in gdat.indxplan]
                for j in gdat.indxplan:
                    if gdat.booldiagmode:
                        if not np.isfinite(gdat.duramask[j]):
                            raise Exception('')
                    for p in gdat.indxinst:
                        for y in gdat.indxchun[p]:
                            gdat.listindxtimetranchun[j][p][y] = tesstarg.util.retr_indxtimetran(gdat.listarrylcur[p][y][:, 0], gdat.epocprio[j], \
                                                                                                                        gdat.periprio[j], gdat.duramask[j])
                        gdat.listindxtimetran[j][p] = tesstarg.util.retr_indxtimetran(gdat.arrylcurbdtr[p][:, 0], \
                                                                                                        gdat.epocprio[j], gdat.periprio[j], gdat.duramask[j])
                        gdat.listindxtimeoutt[j][p] = np.setdiff1d(np.arange(gdat.arrylcurbdtr[p].shape[0]), gdat.listindxtimetran[j][p])
                
                # clean times for each planet
                gdat.listindxtimeclen = [[[] for p in gdat.indxinst] for j in gdat.indxplan]
                for j in gdat.indxplan:
                    listindxtimetemp = []
                    for jj in gdat.indxplan:
                        for p in gdat.indxinst:
                            if jj == j:
                                continue
                            listindxtimetemp.append(gdat.listindxtimetran[jj][p])
                    if len(listindxtimetemp) > 0:
                        listindxtimetemp = np.concatenate(listindxtimetemp)
                        listindxtimetemp = np.unique(listindxtimetemp)
                    else:
                        listindxtimetemp = np.array([])
                    gdat.listindxtimeclen[j][p] = np.setdiff1d(np.arange(gdat.arrylcurbdtr[p].shape[0]), listindxtimetemp)
            
                for p in gdat.indxinst:
                
                    if not np.isfinite(gdat.arrylcurbdtr[p]).all():
                        raise Exception('')
                
                    figr, axis = plt.subplots(1, 1, figsize=gdat.figrsizeydobskin)
                    axis.plot(gdat.arrylcurbdtr[p][:, 0] - gdat.timetess, gdat.arrylcurbdtr[p][:, 1], color='grey', marker='.', ls='', ms=1)
                    if listlimttimemask is not None:
                        axis.plot(gdat.arrylcurbdtr[p][listindxtimegood, 0] - gdat.timetess, \
                                                                gdat.arrylcurbdtr[p][listindxtimegood, 1], color='k', marker='.', ls='', ms=1)
                    for j in gdat.indxplan:
                        colr = gdat.listcolrplan[j]
                        axis.plot(gdat.arrylcurbdtr[p][gdat.listindxtimetran[j], 0] - gdat.timetess, gdat.arrylcurbdtr[p][gdat.listindxtimetran[j], 1], \
                                                                                                             color=colr, marker='.', ls='', ms=1)
                    axis.set_xlabel('Time [BJD - 2457000]')
                    for a in range(2):
                        axis.set_ylabel('Relative Flux')
                    plt.subplots_adjust(bottom=0.2)
                    path = gdat.pathimag + 'lcur%s.%s' % (gdat.liststrginst[p], gdat.strgplotextn)
                    print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()
   
                    if gdat.numbplan > 1:
                        figr, axis = plt.subplots(gdat.numbplan, 1, figsize=gdat.figrsizeydobskin)
                        for jj, j in enumerate(gdat.indxplan):
                            axis[jj].plot(gdat.arrylcurbdtr[p][gdat.listindxtimetran[j], 0] - gdat.timetess, \
                                                                                                gdat.arrylcurbdtr[p][gdat.listindxtimetran[j], 1], \
                                                                                                              color=gdat.listcolrplan[j], marker='o', ls='', ms=0.2)
                            axis[jj].set_ylabel('Relative Flux')
                        axis[-1].set_xlabel('Time [BJD - 2457000]')
                        plt.subplots_adjust(bottom=0.2)
                        path = gdat.pathimag + 'lcurplanmask%s.%s' % (gdat.liststrginst[p], gdat.strgplotextn)
                        print('Writing to %s...' % path)
                        plt.savefig(path)
                        plt.close()
                    
                    if gdat.numbchun[p] > 1:
                        for y in gdat.indxchun[p]:
                            figr, axis = plt.subplots(1, 1, figsize=gdat.figrsizeydobskin)
                            
                            axis.plot(gdat.listarrylcur[p][y][:, 0] - gdat.timetess, gdat.listarrylcur[p][y][:, 1], \
                                                                                            color='grey', marker='.', ls='', ms=1, rasterized=True)
                            
                            if listlimttimemask is not None:
                                axis.plot(gdat.listarrylcur[p][y][listindxtimegood, 0] - gdat.timetess, \
                                                            gdat.listarrylcur[p][y][listindxtimegood, 1], color='k', marker='.', ls='', ms=1, rasterized=True)
                            
                            # color transits
                            ylim = axis.get_ylim()
                            listtimetext = []
                            for j in gdat.indxplan:
                                colr = gdat.listcolrplan[j]
                                axis.plot(gdat.listarrylcur[p][y][gdat.listindxtimetranchun[j][p][y], 0] - gdat.timetess, \
                                                                                           gdat.listarrylcur[p][y][gdat.listindxtimetranchun[j][p][y], 1], \
                                                                                                          color=colr, marker='.', ls='', ms=1, rasterized=True)
                                # draw planet names
                                for n in np.linspace(-30, 30, 61):
                                    time = gdat.epocprio[j] + n * gdat.periprio[j] - gdat.timetess
                                    if np.where(abs(gdat.listarrylcur[p][y][:, 0] - gdat.timetess - time) < 0.1)[0].size > 0:
                                        
                                        # add a vertical offset if overlapping
                                        if np.where(abs(np.array(listtimetext) - time) < 0.5)[0].size > 0:
                                            ypostemp = ylim[0] + (ylim[1] - ylim[0]) * 0.95
                                        else:
                                            ypostemp = ylim[0] + (ylim[1] - ylim[0]) * 0.9

                                        # draw the planet letter
                                        axis.text(time, ypostemp, r'\textbf{%s}' % gdat.liststrgplan[j], color=gdat.listcolrplan[j], va='center', ha='center')
                                        listtimetext.append(time)
                            #axis.set_xlim(xlim)

                            axis.set_xlabel('Time [BJD - 2457000]')
                            for a in range(2):
                                axis.set_ylabel('Relative Flux')
                            
                            plt.subplots_adjust(hspace=0., bottom=0.2)
                            path = gdat.pathimag + 'lcurtran%s%s.%s' % (gdat.liststrginst[p], gdat.liststrgchun[y], gdat.strgplotextn)
                            print('Writing to %s...' % path)
                            plt.savefig(path)
                            plt.close()
                    
                    if gdat.boolbdtr:
                        for y in gdat.indxchun[p]:
                            # plot baseline-detrending
                            figr, axis = plt.subplots(2, 1, figsize=gdat.figrsizeydob)
                            for i in gdat.indxsplnregi[p][y]:
                                ## masked and non-baseline-detrended light curve
                                
                                indxtimetemp = gdat.listindxtimeregi[p][y][i][gdat.indxtimeregioutt[p][y][i]]
                                print('indxtimetemp')
                                summgene(indxtimetemp)
                                print('gdat.listarrylcur[p][y][indxtimetemp, 1]')
                                summgene(gdat.listarrylcur[p][y][indxtimetemp, 1])
                                axis[0].plot(gdat.listarrylcur[p][y][indxtimetemp, 0] - gdat.timetess, gdat.listarrylcur[p][y][indxtimetemp, 1], \
                                                                                                                    marker='o', ls='', ms=1, color='k')
                                ## spline
                                if gdat.listobjtspln[p][y] is not None and gdat.listobjtspln[p][y][i] is not None:
                                    timesplnregifine = np.linspace(gdat.listarrylcur[p][y][gdat.listindxtimeregi[p][y][i], 0][0], \
                                                                                            gdat.listarrylcur[p][y][gdat.listindxtimeregi[p][y][i], 0][-1], 1000)
                                    axis[0].plot(timesplnregifine - gdat.timetess, gdat.listobjtspln[p][y][i](timesplnregifine), 'b-', lw=3)
                                ## baseline-detrended light curve
                                indxtimetemp = gdat.listindxtimeregi[p][y][i]
                                axis[1].plot(gdat.listarrylcurbdtr[p][y][indxtimetemp, 0] - gdat.timetess, gdat.listarrylcurbdtr[p][y][indxtimetemp, 1], \
                                                                                                                    marker='o', ms=1, ls='', color='k')
                            for a in range(2):
                                axis[a].set_ylabel('Relative Flux')
                            axis[0].set_xticklabels([])
                            axis[1].set_xlabel('Time [BJD - 2457000]')
                            plt.subplots_adjust(hspace=0.)
                            path = gdat.pathimag + 'lcurbdtr%s%s.%s' % (gdat.liststrginst[p], gdat.liststrgchun[y], gdat.strgplotextn)
                            print('Writing to %s...' % path)
                            plt.savefig(path)
                            plt.close()
                            
                            if gdat.listobjtspln[p][y] is not None and gdat.listobjtspln[p][y][i] is not None:
                                # produce a table for the spline coefficients
                                fileoutp = open(gdat.pathdata + 'coefbdtr.csv', 'w')
                                fileoutp.write(' & ')
                                for i in gdat.indxsplnregi[p][y]:
                                    print('$\beta$:', gdat.listobjtspln[p][y][i].get_coeffs())
                                    print('$t_k$:', gdat.listobjtspln[p][y][i].get_knots())
                                    print
                                fileoutp.write('\\hline\n')
                                fileoutp.close()

        # number of samples to draw from the prior
        gdat.numbsamp = 10000
    
        if gdat.booldatatser:
            plot_pcur(gdat, 'prio')
        calc_prop(gdat, 'prio', 'orbt')
    
    ## augment object dictinary
    if gdat.boolobjt:
        gdat.dictfeatobjt = dict()
        gdat.dictfeatobjt['namestar'] = np.array([gdat.labltarg] * gdat.numbplan)
        gdat.dictfeatobjt['nameplan'] = gdat.liststrgplanfull
        gdat.dictfeatobjt['booltran'] = np.array([True], dtype=bool)
        gdat.dictfeatobjt['vmagstar'] = np.zeros(gdat.numbplan) + gdat.vmagstar
        gdat.dictfeatobjt['jmagstar'] = np.zeros(gdat.numbplan) + gdat.jmagstar
        gdat.dictfeatobjt['hmagstar'] = np.zeros(gdat.numbplan) + gdat.hmagstar
        gdat.dictfeatobjt['kmagstar'] = np.zeros(gdat.numbplan) + gdat.kmagstar
        gdat.dictfeatobjt['numbplanstar'] = np.zeros(gdat.numbplan) + gdat.numbplan
        gdat.dicterrr['radistarsunn'] = gdat.dicterrr['radistar'] / gdat.factrsrj # R_S
        #gdat.dicterrr['stdvradistar'] = gdat.dicterrr['radistar'] / gdat.factrsrj # R_S
    
    plot_prop(gdat, 'prio')
    
    if not gdat.boolobjt:
        return

    # look for single transits using matched filter
    

    if gdat.booldatatser:
        gdat.pathallebase = gdat.pathobjt + 'allesfits/'
    
    #gdat.boolalleprev = {}
    #for typeparacalc in gdat.listtypeparacalc:
    #    gdat.boolalleprev[typeparacalc] = {}
    #
    #for strgfile in ['params.csv', 'settings.csv', 'params_star.csv']:
    #    
    #    for typeparacalc in gdat.listtypeparacalc:
    #        pathinit = '%sdata/allesfit_templates/%s/%s' % (gdat.pathbase, typeparacalc, strgfile)
    #        pathfinl = '%sallesfits/allesfit_%s/%s' % (gdat.pathobjt, typeparacalc, strgfile)

    #        if not os.path.exists(pathfinl):
    #            cmnd = 'cp %s %s' % (pathinit, pathfinl)
    #            print(cmnd)
    #            os.system(cmnd)
    #            if strgfile == 'params.csv':
    #                gdat.boolalleprev[typeparacalc]['para'] = False
    #            if strgfile == 'settings.csv':
    #                gdat.boolalleprev[typeparacalc]['sett'] = False
    #            if strgfile == 'params_star.csv':
    #                gdat.boolalleprev[typeparacalc]['pars'] = False
    #        else:
    #            if strgfile == 'params.csv':
    #                gdat.boolalleprev[typeparacalc]['para'] = True
    #            if strgfile == 'settings.csv':
    #                gdat.boolalleprev[typeparacalc]['sett'] = True
    #            if strgfile == 'params_star.csv':
    #                gdat.boolalleprev[typeparacalc]['pars'] = True

    if gdat.booldatatser:
        if gdat.boolallebkgdgaus:
            # background allesfitter run
            print('Setting up the background allesfitter run...')
            
            if not gdat.boolalleprev['bkgd']['para']:
                evol_file(gdat, 'params.csv', gdat.pathallebkgd, lineadde)
            
            ## mask out the transits for the background run
            path = gdat.pathallebkgd + gdat.liststrgchun[y]  + '.csv'
            if not os.path.exists(path):
                indxtimebkgd = np.setdiff1d(gdat.indxtime, np.concatenate(gdat.listindxtimetran))
                gdat.arrylcurbkgd = gdat.listarrylcurbdtr[p][y][indxtimebkgd, :]
                print('Writing to %s...' % path)
                np.savetxt(path, gdat.arrylcurbkgd, delimiter=',', header='time,flux,flux_err')
            else:
                print('OoT light curve available for the background allesfitter run at %s.' % path)
            
            #liststrg = list(gdat.objtallebkgd.posterior_params.keys())
            #for k, strg in enumerate(liststrg):
            #   post = gdat.objtallebkgd.posterior_params[strg]
            #   linesplt = '%s' % gdat.objtallebkgd.posterior_params_at_maximum_likelihood[strg][0]
    
    gdat.pathalle = dict()
    gdat.objtalle = dict()
    if not gdat.boolphascurv:
        # setup the orbit run
        print('Setting up the orbit allesfitter run...')

        proc_alle(gdat, 'orbt')
        
    # phase curve
    else:
        
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
    
        ## update the transit duration for fastfit
        #if not gdat.boolalleprev['pcur']['sett']:
        #    lineadde = [['fast_fit,*', 'fast_fit,False']]
        #    evol_file(gdat, 'settings.csv', gdat.pathallepcur, lineadde)
        #
        ## update the parameters
        #lineadde = []
        #for j in gdat.indxplan:
        #    strgrrat = '%s_rr' % gdat.liststrgplan[j]
        #    strgrsma = '%s_rsuma' % gdat.liststrgplan[j]
        #    strgcosi = '%s_cosi' % gdat.liststrgplan[j]
        #    strgepoc = '%s_epoch' % gdat.liststrgplan[j]
        #    strgperi = '%s_period' % gdat.liststrgplan[j]
        #    lineadde.extend([ \
        #                [strgrrat + '*', '%s,%f,1,uniform %f %f,$R_{%s} / R_\star$,\n' % \
        #                              (strgrrat, gdat.dicterrr['rrat'][0, j], 0, 2 * gdat.rratprio[j], gdat.liststrgplan[j])], \
        #                [strgrsma + '*', '%s,%f,1,uniform %f %f,$(R_\star + R_{%s}) / a_{%s}$,\n' % \
        #                              (strgrsma, gdat.dicterrr['rsma'][0, j], 0, 2 * gdat.rsmaprio[j], gdat.liststrgplan[j], gdat.liststrgplan[j])], \
        #                [strgcosi + '*', '%s,%f,1,uniform %f %f,$\cos{i_{%s}}$,\n' % \
        #                              (strgcosi, gdat.dicterrr['cosi'][0, j], 0, max(0.1, gdat.cosiprio[j] * 2), gdat.liststrgplan[j])], \
        #                [strgepoc + '*', '%s,%f,1,uniform %f %f,$T_{0;%s}$,$\mathrm{BJD}$\n' % \
        #                (strgepoc, gdat.dictpost['epoc'][1, j], gdat.dictpost['epoc'][0, j], gdat.dictpost['epoc'][2, j], gdat.liststrgplan[j])], \
        #                [strgperi + '*', '%s,%f,1,uniform %f %f,$P_{%s}$,$\mathrm{d}$\n' % \
        #                      (strgperi, gdat.dicterrr['peri'][0, j], gdat.periprio[j] - 0.01, gdat.periprio[j] + 0.01, gdat.liststrgplan[j])], \
        #               ])
        #if not gdat.boolalleprev['pcur']['para']:
        #    evol_file(gdat, 'params.csv', gdat.pathallepcur, lineadde)
        #
        
        #_0003: single component offset baseline
        #_0004: multiple components, offset baseline
        #_0006: multiple components, GP baseline
        
        gdat.pathimagpcur = gdat.pathimag + 'pcur/'
        gdat.pathdatapcur = gdat.pathdata + 'pcur/'
        os.system('mkdir -p %s' % gdat.pathimagpcur)
        os.system('mkdir -p %s' % gdat.pathdatapcur)
        
        proc_alle(gdat, '0003')
        #proc_alle(gdat, '0004')
        #proc_alle(gdat, '0006')
        
        ## plot the spherical limits
        #figr, axis = plt.subplots(figsize=gdat.figrsizeydob)
        #
        #gdat.objtalle[typeparacalc] = allesfitter.allesclass(gdat.pathallepcur)
        #gdat.objtalle[typeparacalc].posterior_params_median['b_sbratio_TESS'] = 0
        #gdat.objtalle[typeparacalc].settings['host_shape_TESS'] = 'sphere'
        #gdat.objtalle[typeparacalc].settings['b_shape_TESS'] = 'roche'
        #gdat.objtalle[typeparacalc].posterior_params_median['host_gdc_TESS'] = 0
        #gdat.objtalle[typeparacalc].posterior_params_median['host_bfac_TESS'] = 0
        #lcurmodltemp = gdat.objtalle[typeparacalc].get_posterior_median_model(strgchun, 'flux', xx=gdat.time)
        #axis.plot(gdat.arrypcurquadbdtr[p][j][:, 0], (gdat.lcurmodlevvv - lcurmodltemp) * 1e6, lw=2, label='Spherical star')
        #
        #gdat.objtalle[typeparacalc] = allesfitter.allesclass(gdat.pathallepcur)
        #gdat.objtalle[typeparacalc].posterior_params_median['b_sbratio_TESS'] = 0
        #gdat.objtalle[typeparacalc].settings['host_shape_TESS'] = 'roche'
        #gdat.objtalle[typeparacalc].settings['b_shape_TESS'] = 'sphere'
        #gdat.objtalle[typeparacalc].posterior_params_median['host_gdc_TESS'] = 0
        #gdat.objtalle[typeparacalc].posterior_params_median['host_bfac_TESS'] = 0
        #lcurmodltemp = gdat.objtalle[typeparacalc].get_posterior_median_model(strgchun, 'flux', xx=gdat.time)
        #axis.plot(gdat.arrypcurquadbdtr[p][j][:, 0], (gdat.lcurmodlevvv - lcurmodltemp) * 1e6, lw=2, label='Spherical planet')
        #axis.legend()
        #axis.set_ylim([-100, 100])
        #axis.set(xlabel='Phase')
        #axis.set(ylabel='Relative flux [ppm]')
        #plt.subplots_adjust(hspace=0.)
        #path = pathimag + 'pcurquadmodldiff.%s' % gdat.strgplotextn
        #plt.savefig(path)
        #plt.close()

