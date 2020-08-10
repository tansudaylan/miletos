import allesfitter
import allesfitter.config

import tdpy.util
import tdpy.mcmc
from tdpy.util import prnt_list
from tdpy.util import summgene
import pandora.main

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

            minmdiff = 1e12
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


def retr_exarcomp(gdat, strgtarg=None):
    
    pathbase = os.environ['PEXO_DATA_PATH'] + '/'

    # get NASA Exoplanet Archive data
    path = pathbase + 'data/compositepars_2020.01.23_16.22.13.csv'
    print('Reading %s...' % path)
    objtexarcomp = pd.read_csv(path, skiprows=124)
    if strgtarg is None:
        indx = np.arange(objtexarcomp['fpl_hostname'].size)
    else:
        indx = np.where(objtexarcomp['fpl_hostname'] == strgtarg)[0]
    print('strgtarg')
    print(strgtarg)
    print('indx')
    summgene(indx)
    
    if indx.size == 0:
        print('The planet name, %s, was not found in the NASA Exoplanet Archive composite table.' % strgtarg)
        return None
    else:
        dictexarcomp = {}
        dictexarcomp['namestar'] = objtexarcomp['fpl_hostname'][indx].values
        
        dictexarcomp['radistar'] = objtexarcomp['fst_rad'][indx].values * gdat.factrsrj # [R_J]
        dictexarcomp['radistarstd1'] = objtexarcomp['fst_raderr1'][indx].values * gdat.factrsrj # [R_J]
        dictexarcomp['radistarstd2'] = objtexarcomp['fst_raderr2'][indx].values * gdat.factrsrj # [R_J]
        dictexarcomp['radistarstdv'] = (dictexarcomp['radistarstd1'] + dictexarcomp['radistarstd2']) / 2.
        dictexarcomp['massstar'] = objtexarcomp['fst_mass'][indx].values * gdat.factmsmj # [M_J]
        dictexarcomp['massstarstd1'] = objtexarcomp['fst_masserr1'][indx].values * gdat.factmsmj # [M_J]
        dictexarcomp['massstarstd2'] = objtexarcomp['fst_masserr2'][indx].values * gdat.factmsmj # [M_J]
        dictexarcomp['massstarstdv'] = (dictexarcomp['massstarstd1'] + dictexarcomp['massstarstd2']) / 2.
        dictexarcomp['tmptstar'] = objtexarcomp['fst_teff'][indx].values # [K]
        dictexarcomp['tmptstarstd1'] = objtexarcomp['fst_tefferr1'][indx].values # [K]
        dictexarcomp['tmptstarstd2'] = objtexarcomp['fst_tefferr2'][indx].values # [K]
        dictexarcomp['tmptstarstdv'] = (dictexarcomp['tmptstarstd1'] + dictexarcomp['tmptstarstd2']) / 2.
        
        dictexarcomp['nameplan'] = objtexarcomp['fpl_name'][indx].values
        dictexarcomp['peri'] = objtexarcomp['fpl_orbper'][indx].values # [days]
        dictexarcomp['radiplan'] = objtexarcomp['fpl_radj'][indx].values # [R_J]
        dictexarcomp['smax'] = objtexarcomp['fpl_smax'][indx].values # [AU]
        dictexarcomp['massplan'] = objtexarcomp['fpl_bmassj'][indx].values # [M_J]
        dictexarcomp['stdvradiplan'] = np.maximum(objtexarcomp['fpl_radjerr1'][indx].values, objtexarcomp['fpl_radjerr2'][indx].values) # [R_J]
        dictexarcomp['stdvmassplan'] = np.maximum(objtexarcomp['fpl_bmassjerr1'][indx].values, objtexarcomp['fpl_bmassjerr2'][indx].values) # [M_J]
        dictexarcomp['tmptplan'] = objtexarcomp['fpl_eqt'][indx].values # [K]
        dictexarcomp['omagstar'] = objtexarcomp['fst_optmag'][indx].values
        dictexarcomp['booltran'] = objtexarcomp['fpl_tranflag'][indx].values # [K]
        dictexarcomp['booltran'] = dictexarcomp['booltran'].astype(bool)
        # temp
        dictexarcomp['kmagstar'] = objtexarcomp['fst_nirmag'][indx].values # [K]
        dictexarcomp['imagstar'] = objtexarcomp['fst_nirmag'][indx].values # [K]
        
        numbplanexar = len(dictexarcomp['nameplan'])
        print('numbplanexar')
        print(numbplanexar)
        
        dictexarcomp['numbplanstar'] = np.empty(numbplanexar)
        dictexarcomp['numbplantranstar'] = np.empty(numbplanexar)
        # 
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
                axis.text(0.9, 0.9, r'\textbf{%s}' % gdat.liststrgplan[j], color=gdat.listcolrplan[j], va='center', ha='center', transform=axis.transAxes)
                axis.set_ylabel('Relative Flux')
                axis.set_xlabel('Phase')
                
                if a == 1:
                    ylim = [np.percentile(arrypcur[p][j][:, 1], 0.3), np.percentile(arrypcur[p][j][:, 1], 99.7)]
                    axis.set_ylim(ylim)
                
                # overlay the posterior model
                if strgpdfn == 'post':
                    axis.plot(gdat.arrypcurprimmodl[p][j][:, 0], gdat.arrypcurprimmodl[p][j][:, 1], color='b')
                    
                path = gdat.pathimag + 'pcurphasplan%04d_%s_%d.%s' % (j + 1, strgpdfn, a, gdat.strgplotextn)
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
                    
                axis.text(0.9, 0.9, \
                                    r'\textbf{%s}' % gdat.liststrgplan[j], color=gdat.listcolrplan[j], va='center', ha='center', transform=axis.transAxes)
                axis.set_ylabel('Relative Flux')
                axis.set_xlabel('Time [hours]')
                axis.set_xlim([-2 * np.amax(gdat.duraprio) * 24., 2 * np.amax(gdat.duraprio) * 24.])
                
                if a == 1:
                    axis.set_ylim(ylim)
                
                plt.subplots_adjust(hspace=0., bottom=0.25, left=0.25)
                path = gdat.pathimag + 'pcurtimeplan%04d_%s_%d.%s' % (j + 1, strgpdfn, a, gdat.strgplotextn)
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
                path = gdat.pathimag + 'pcur_%s_%d.%s' % (strgpdfn, a, gdat.strgplotextn)
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()
    

def retr_albg(amplplanrefl, radiplan, smax):
    
    albg = amplplanrefl / (radiplan / smax)**2
    
    return albg


def proc_alle(gdat, strgalle):
    
    # allesfit run folder
    gdat.pathalle[strgalle] = gdat.pathallebase + 'allesfit_%s/' % strgalle
    
    # make sure the folder exists
    cmnd = 'mkdir -p %s' % gdat.pathalle[strgalle]
    os.system(cmnd)
    
    # write the input data file
    for p in gdat.indxinst:
        path = gdat.pathalle[strgalle] + gdat.liststrginst[p] + '.csv'
        indxchuninst = gdat.listindxchuninst[p]
        print('indxchuninst')
        summgene(indxchuninst)
        print(indxchuninst)
        listarrylcurbdtrtemp = []
        for y in gdat.indxchun[p]:
            listarrylcurbdtrtemp.append(gdat.listarrylcurbdtr[y])
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
    
    if strgalle == 'orbt':
        pass
    pathpara = gdat.pathalle[strgalle] + 'params.csv'
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
                             (strgepoc, gdat.epocprio[j], gdat.epocprio[j] - 0.5, gdat.epocprio[j] + 0.5, gdat.liststrgplan[j])], \
                        ['', '%s,%f,1,uniform %f %f,$P_{%s}$,days\n' % \
                             (strgperi, gdat.periprio[j], gdat.periprio[j] - 0.01, gdat.periprio[j] + 0.01, gdat.liststrgplan[j])], \
                        #['', '%s,%f,1,uniform %f %f,$\sqrt{e_b} \cos{\omega_b}$,\n' % \
                        #     (strgecos, gdat.ecosprio[j], -0.9, 0.9, gdat.liststrgplan[j])], \
                        #['', '%s,%f,1,uniform %f %f,$\sqrt{e_b} \sin{\omega_b}$,\n' % \
                        #     (strgesin, gdat.esinprio[j], -0.9, 0.9, gdat.liststrgplan[j])], \
                       ])
            if strgalle == '0003' or strgalle == '0004' or strgalle == '0006':
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
            
            evol_file(gdat, 'params.csv', gdat.pathalle[strgalle], lineadde)
    
    pathsett = gdat.pathalle[strgalle] + 'settings.csv'
    if not os.path.exists(pathsett):
        cmnd = 'touch %s' % (pathsett)
        print(cmnd)
        os.system(cmnd)

        lineadde = [ \
                     ['', 'fast_fit,True\n'], \
                    #['', 'fast_fit_width,%.3g\n' % np.amax(gdat.duramask)], \
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
        
        if strgalle == '0003' or strgalle == '0004' or strgalle == '0006':
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
            
        evol_file(gdat, 'settings.csv', gdat.pathalle[strgalle], lineadde)
    
    ## initial plot
    path = gdat.pathalle[strgalle] + 'results/initial_guess_b.pdf'
    if not os.path.exists(path):
        allesfitter.show_initial_guess(gdat.pathalle[strgalle])
    
    ## do the run
    path = gdat.pathalle[strgalle] + 'results/mcmc_save.h5'
    if not os.path.exists(path):
        allesfitter.mcmc_fit(gdat.pathalle[strgalle])
    else:
        print('%s exists... Skipping the orbit run.' % path)

    ## make the final plots
    path = gdat.pathalle[strgalle] + 'results/mcmc_corner.pdf'
    if not os.path.exists(path):
        allesfitter.mcmc_output(gdat.pathalle[strgalle])
        
    # read the allesfitter posterior
    print('Reading from %s...' % gdat.pathalle[strgalle])
    gdat.objtalle[strgalle] = allesfitter.allesclass(gdat.pathalle[strgalle])
    
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

    gdat.numbsamp = gdat.objtalle[strgalle].posterior_params[list(gdat.objtalle[strgalle].posterior_params.keys())[0]].size
    
    if gdat.numbsamp > 10000:
        print('Thinning down the allesfitter chain!')
        gdat.indxsamp = np.random.choice(np.arange(gdat.numbsamp), size=10000, replace=False)
        gdat.numbsamp = 10000
    else:
        gdat.indxsamp = np.arange(gdat.numbsamp)

    gdat.liststrgfeat = ['epoc', 'peri', 'rrat', 'rsma', 'cosi', 'ecce', 'smax']
    if strgalle == '0003' or strgalle == '0004' or strgalle == '0006':
        gdat.liststrgfeat += ['sbrtrati', 'amplelli']
    if strgalle == '0003' or strgalle == '0006':
        gdat.liststrgfeat += ['amplplan', 'timeshftplan']
    if strgalle == '0004':
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
    
            if strgalle == '0003' or strgalle == '0004' or strgalle == '0006':
                if strgfeat == 'sbrtrati':
                    strg = '%s_sbratio_TESS' % gdat.liststrgplan[j]
                if strgfeat == 'amplelli':
                    strg = '%s_phase_curve_ellipsoidal_TESS' % gdat.liststrgplan[j]
            if strgalle == '0003' or strgalle == '0006':
                if strgfeat == 'amplplan':
                    strg = '%s_phase_curve_atmospheric_TESS' % gdat.liststrgplan[j]
                if strgfeat == 'timeshftplan':
                    strg = '%s_phase_curve_atmospheric_shift_TESS' % gdat.liststrgplan[j]
            if strgalle == '0004':
                if strgfeat == 'amplplanther':
                    strg = '%s_phase_curve_atmospheric_thermal_TESS' % gdat.liststrgplan[j]
                if strgfeat == 'amplplanrefl':
                    strg = '%s_phase_curve_atmospheric_reflected_TESS' % gdat.liststrgplan[j]
                if strgfeat == 'timeshftplanther':
                    strg = '%s_phase_curve_atmospheric_thermal_shift_TESS' % gdat.liststrgplan[j]
                if strgfeat == 'timeshftplanrefl':
                    strg = '%s_phase_curve_atmospheric_reflected_shift_TESS' % gdat.liststrgplan[j]
            
            if strg in gdat.objtalle[strgalle].posterior_params.keys():
                gdat.dictlist[strgfeat][:, j] = gdat.objtalle[strgalle].posterior_params[strg][gdat.indxsamp]
            else:
                gdat.dictlist[strgfeat][:, j] = np.zeros(gdat.numbsamp) + allesfitter.config.BASEMENT.params[strg]
    

    # allesfitter phase curve depths are in ppt
    for strgfeat in gdat.liststrgfeat:
        if strgfeat.startswith('ampl'):
            gdat.dictlist[strgfeat] *= 1e-3
    
    print('Calculating derived variables...')
    # derived variables
    ## get samples from the star's variables
    gdat.listradistar = np.random.randn(gdat.numbsamp) * gdat.radistarstdv + gdat.radistar
    gdat.listmassstar = np.random.randn(gdat.numbsamp) * gdat.massstarstdv + gdat.massstar
    gdat.listtmptstar = np.random.randn(gdat.numbsamp) * gdat.tmptstarstdv + gdat.tmptstar
    print('gdat.listradistar')
    summgene(gdat.listradistar)
    print('gdat.listmassstar')
    summgene(gdat.listmassstar)
    print('gdat.listtmptstar')
    summgene(gdat.listtmptstar)
    
    print('gdat.dictlist[timeshftplan]')
    summgene(gdat.dictlist['timeshftplan'])
    if strgalle == '0003' or strgalle == '0004' or strgalle == '0006':
        gdat.dictlist['amplnigh'] = gdat.dictlist['sbrtrati'] * gdat.dictlist['rrat']**2
    if strgalle == '0003' or strgalle == '0006':
        gdat.dictlist['phasshftplan'] = gdat.dictlist['timeshftplan'] * 360. / gdat.dictlist['peri']
    if strgalle == '0004':
        gdat.dictlist['phasshftplanther'] = gdat.dictlist['timeshftplanther'] * 360. / gdat.dictlist['peri']
        gdat.dictlist['phasshftplanrefl'] = gdat.dictlist['timeshftplanrefl'] * 360. / gdat.dictlist['peri']

    print('Calculating inclinations...')
    # inclination [degree]
    gdat.dictlist['incl'] = np.arccos(gdat.dictlist['cosi']) * 180. / np.pi
    
    # radius of the planets
    gdat.dictlist['radiplan'] = gdat.listradistar[:, None] * gdat.dictlist['rrat']
    gdat.dictlist['radiplaneart'] = gdat.dictlist['radiplan'] * gdat.factrjre
    
    # semi-major axis
    gdat.dictlist['smax'] = (gdat.dictlist['radiplan'] + gdat.listradistar[:, None]) / gdat.dictlist['rsma']
    gdat.dictlist['smaxasun'] = gdat.dictlist['smax'] / gdat.factaurj
    
    print('Calculating equilibrium temperatures...')
    # planet equilibrium temperature
    gdat.dictlist['tmptplan'] = gdat.listtmptstar[:, None] * np.sqrt(gdat.listradistar[:, None] / 2. / gdat.dictlist['smax'])
    
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
    
    print('Calculating radius and period ratios...')
    for j in gdat.indxplan:
        strgratiperi = 'ratiperipla%s' % gdat.liststrgplan[j]
        strgratiradi = 'ratiradipla%s' % gdat.liststrgplan[j]
        for jj in gdat.indxplan:
            gdat.dictlist[strgratiperi] = gdat.dictlist['peri'][:, j] / gdat.dictlist['peri'][:, jj]
            gdat.dictlist[strgratiradi] = gdat.dictlist['radiplan'][:, j] / gdat.dictlist['radiplan'][:, jj]
        
    print('Calculating RV semi-amplitudes...')
    # RV semi-amplitude
    gdat.dictlist['rvsapred'] = tesstarg.util.retr_radvsema(gdat.dictlist['peri'], gdat.dictlist['massplanpred'], \
                                                                                            gdat.listmassstar[:, None] / gdat.factmsmj, \
                                                                                                gdat.dictlist['incl'], gdat.dictlist['ecce'])
    
    print('Calculating TSMs...')
    # TSM
    gdat.dictlist['tsmm'] = tesstarg.util.retr_tsmm(gdat.dictlist['radiplan'], gdat.dictlist['tmptplan'], \
                                                                                gdat.dictlist['massplanpred'], gdat.listradistar[:, None], gdat.imagstar)
    
    # ESM
    gdat.dictlist['esmm'] = tesstarg.util.retr_esmm(gdat.dictlist['tmptplan'], gdat.listtmptstar[:, None], \
                                                                                gdat.dictlist['radiplan'], gdat.listradistar[:, None], gdat.imagstar)
        
    gdat.dictlist['ltsm'] = np.log(gdat.dictlist['tsmm']) 
    gdat.dictlist['lesm'] = np.log(gdat.dictlist['esmm']) 

    # temp
    gdat.dictlist['fsin'] = np.zeros_like(gdat.dictlist['rrat'])
    gdat.dictlist['fcos'] = np.zeros_like(gdat.dictlist['rrat'])
    gdat.dictlist['sini'] = np.sqrt(1. - gdat.dictlist['cosi']**2)
    gdat.dictlist['ecce'] = gdat.dictlist['fsin']**2 + gdat.dictlist['fcos']**2
    gdat.dictlist['omeg'] = 180. / np.pi * np.mod(np.arctan2(gdat.dictlist['fsin'], gdat.dictlist['fcos']), 2 * np.pi)
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
    if strgalle == '0003' or strgalle == '0006':
        frac = np.random.rand(gdat.dictlist['amplplan'].size).reshape(gdat.dictlist['amplplan'].shape)
        gdat.dictlist['amplplanther'] = gdat.dictlist['amplplan'] * frac
        gdat.dictlist['amplplanrefl'] = gdat.dictlist['amplplan'] * (1. - frac)
    
    if strgalle == '0004':
        # temp -- this does not work for two component (thermal + reflected)
        gdat.dictlist['amplseco'] = gdat.dictlist['amplnigh'] + gdat.dictlist['amplplanther'] + gdat.dictlist['amplplanrefl']
    if strgalle == '0003' or strgalle == '0006':
        # temp -- this does not work when phase shift is nonzero
        gdat.dictlist['amplseco'] = gdat.dictlist['amplnigh'] + gdat.dictlist['amplplan']
    
    if strgalle == '0003' or strgalle == '0004' or strgalle == '0006':
        gdat.dictlist['albg'] = retr_albg(gdat.dictlist['amplplanrefl'], gdat.dictlist['radiplan'], gdat.dictlist['smax'])

    print('Calculating the equilibrium temperature of the planets...')
    gdat.dictlist['tmptequi'] = gdat.listtmptstar[:, None] * np.sqrt(gdat.listradistar[:, None] / gdat.dictlist['smax'] / 2.)
    
    if gdat.labltarg == 'WASP-121':# and strgalle == '0003':
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
        print('gdat.dicterrr[strgfeat]')
        print(gdat.dicterrr[strgfeat])
        print('')

    gdat.arrylcurmodl = [[[] for p in gdat.liststrginst] for p in gdat.indxinst]
    for p, strginst in enumerate(gdat.liststrginst):
        gdat.arrylcurmodl[p] = np.empty((gdat.time.size, 3))
        gdat.arrylcurmodl[p][:, 0] = gdat.time
        gdat.arrylcurmodl[p][:, 1] = gdat.objtalle[strgalle].get_posterior_median_model(strginst, 'flux', xx=gdat.time)
        gdat.arrylcurmodl[p][:, 2] = 0.

    # write the model to file
    for p in gdat.indxinst:
        
        path = gdat.pathdata + 'arrylcurmodl' + gdat.liststrgchun[y] + '.csv'
        print('Writing to %s...' % path)
        np.savetxt(path, gdat.arrylcurmodl[p], delimiter=',', header='time,flux,flux_err')

    # number of samples to plot
    gdat.numbsampplot = min(100, gdat.numbsamp)
    gdat.indxsampplot = np.random.choice(gdat.indxsamp, gdat.numbsampplot, replace=False)
    gdat.listlcurmodl = np.empty((gdat.numbsampplot, gdat.time.size))
    gdat.listpcurquadmodl = [[[] for j in gdat.indxplan] for p in gdat.liststrginst]
    for p in gdat.indxinst:
        for j in gdat.indxplan:
            gdat.listpcurquadmodl[p][j] = np.empty((gdat.numbsampplot, gdat.numbtime))
        gdat.arrylcurmodl[p] = np.zeros((gdat.time.size, 3))
        gdat.arrylcurmodl[p][:, 0] = gdat.time
    
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
            gdat.listlcurmodl[ii, :] = gdat.objtalle[strgalle].get_one_posterior_model(strginst, 'flux', xx=gdat.time, sample_id=i)
            gdat.arrylcurmodl[p][:, 1] = gdat.listlcurmodl[ii, :]
            gdat.listpcurquadmodl[p][j][ii, :] = tesstarg.util.fold_lcur(gdat.arrylcurmodl[p], \
                                                                                gdat.dicterrr['epoc'][0, j], gdat.dicterrr['peri'][0, j], phasshft=0.25)[:, 1]

        # get allesfitter baseline model
        gdat.lcurbasealle = gdat.objtalle[strgalle].get_posterior_median_baseline(strginst, 'flux', xx=gdat.time)
        # get allesfitter-detrended data
        gdat.lcuradtr = gdat.listarrylcurbdtr[y][:, 1] - gdat.lcurbasealle
        gdat.arrylcuradtr = np.copy(gdat.listarrylcurbdtr[y])
        gdat.arrylcuradtr[:, 1] = gdat.lcuradtr
    
        for j in gdat.indxplan:
            gdat.arrypcurprimmodl[p][j] = tesstarg.util.fold_lcur(gdat.arrylcurmodl[p][gdat.listindxtimeclen[j], :], \
                                                                                            gdat.dicterrr['epoc'][0, j], gdat.dicterrr['peri'][0, j])
            gdat.arrypcurquadmodl[p][j] = tesstarg.util.fold_lcur(gdat.arrylcurmodl[p][gdat.listindxtimeclen[j], :], \
                                                                                gdat.dicterrr['epoc'][0, j], gdat.dicterrr['peri'][0, j], phasshft=0.25)
            gdat.arrypcurprimadtr[p][j] = tesstarg.util.fold_lcur(gdat.arrylcuradtr[gdat.listindxtimeclen[j], :], \
                                                                                            gdat.dicterrr['epoc'][0, j], gdat.dicterrr['peri'][0, j])
            gdat.arrypcurprimadtrbind[p][j] = tesstarg.util.rebn_lcur(gdat.arrypcurprimadtr[p][j], gdat.numbbinspcur)
            gdat.arrypcurprimadtrbindfine[p][j] = tesstarg.util.rebn_lcur(gdat.arrypcurprimadtr[p][j], gdat.numbbinspcurfine)
            gdat.arrypcurquadadtr[p][j] = tesstarg.util.fold_lcur(gdat.arrylcuradtr[gdat.listindxtimeclen[j], :], \
                                                                                      gdat.dicterrr['epoc'][0, j], gdat.dicterrr['peri'][0, j], phasshft=0.25)
            gdat.arrypcurquadadtrbind[p][j] = tesstarg.util.rebn_lcur(gdat.arrypcurquadadtr[p][j], gdat.numbbinspcur)
            gdat.arrypcurquadadtrbindfine[p][j] = tesstarg.util.rebn_lcur(gdat.arrypcurquadadtr[p][j], gdat.numbbinspcurfine)
    
    # plot GP-detrended phase curves
    plot_pcur(gdat, 'post')
    
    # pretty orbit plot
    fact = 50.
    factstar = 5.
    # plot the orbit
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
    
    # add Mercury
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
    # centerline
    axis.set_xlabel('Distance from the star [AU]')
    plt.subplots_adjust()
    #axis.legend()
    path = gdat.pathimag + 'orbt.%s' % gdat.strgplotextn
    print('Writing to %s...' % path)
    plt.savefig(path)
    plt.close()
   
    # plot all exoplanets and overplot this one
    dictexarcomp = retr_exarcomp(gdat)
    
    numbplanexar = dictexarcomp['radiplan'].size
    indxplanexar = np.arange(numbplanexar)
    
    # optical magnitude vs number of planets
    gdat.pathimagmagt = gdat.pathimag + 'magt/'
    os.system('mkdir -p %s' % gdat.pathimagmagt)
    for b in range(4):
        if b == 0:
            strgvarbmagt = 'omag'
            lablxaxi = 'V Magnitude'
            varbtarg = gdat.omagstar
            varb = dictexarcomp['omagstar']
        if b == 1:
            strgvarbmagt = 'imag'
            lablxaxi = 'J Magnitude'
            varbtarg = gdat.imagstar
            varb = dictexarcomp['imagstar']
        if b == 2:
            strgvarbmagt = 'msnromag'
            lablxaxi = 'Relative mass SNR in the V band'
            varbtarg = np.sqrt(10**(-gdat.omagstar / 2.5)) / gdat.massstar**(2. / 3.)
            varb = np.sqrt(10**(-dictexarcomp['omagstar'] / 2.5)) / dictexarcomp['massstar']**(2. / 3.)
        if b == 3:
            strgvarbmagt = 'msnrimag'
            lablxaxi = 'Relative mass SNR in the J band'
            varbtarg = np.sqrt(10**(-gdat.omagstar / 2.5)) / gdat.massstar**(2. / 3.)
            varb = np.sqrt(10**(-dictexarcomp['imagstar'] / 2.5)) / dictexarcomp['massstar']**(2. / 3.)
        for a in range(3):
            figr, axis = plt.subplots(figsize=gdat.figrsize)
            if a == 0:
                indx = np.where(dictexarcomp['boolfrst'])[0]
            if a == 1:
                indx = np.where(dictexarcomp['boolfrst'] & (dictexarcomp['numbplanstar'] > 3))[0]
            if a == 2:
                indx = np.where(dictexarcomp['boolfrst'] & (dictexarcomp['numbplantranstar'] > 3))[0]
            
            if b == 2 or b == 3:
                normfact = max(varbtarg, np.amax(varb[indx]))
            else:
                normfact = 1.
            varbtargnorm = varbtarg / normfact
            varbnorm = varb[indx] / normfact
            
            axis.scatter(varbnorm, dictexarcomp['numbplanstar'][indx], s=1, color='black')
                
            indxsort = np.argsort(varbnorm)
            if b == 2 or b == 3:
                indxsort = indxsort[::-1]

            listnameaddd = []
            cntr = 0
            while True:
                k = indxsort[cntr]
                nameadd = dictexarcomp['namestar'][indx][k]
                if not nameadd in listnameaddd:
                    axis.text(varbnorm[k], dictexarcomp['numbplanstar'][indx][k] + 0.5, nameadd, size=6, \
                                                                                            va='center', ha='right', rotation=45)
                    listnameaddd.append(nameadd)
                cntr += 1
                if len(listnameaddd) == 5: 
                    break
            axis.scatter(varbtargnorm, gdat.numbplan, s=5, color='black', marker='x')
            axis.text(varbtargnorm, gdat.numbplan + 0.5, gdat.labltarg, size=8, color='black', \
                                                                                            va='center', ha='center', rotation=45)
            axis.set_ylabel(r'Number of transiting planets')
            axis.set_xlabel(lablxaxi)
            plt.subplots_adjust(bottom=0.2)
            path = gdat.pathimagmagt + '%snumb_%d.%s' % (strgvarbmagt, a, gdat.strgplotextn)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()

            figr, axis = plt.subplots(figsize=gdat.figrsize)
            axis.hist(varbnorm, 50)
            axis.axvline(varbtargnorm, color='black', ls='--')
            #ypos = axis.get_ylim()[0] + (axis.get_ylim()[1] - axis.get_ylim()[0]) * 0.8
            
            axis.text(0.9, 0.9, gdat.labltarg, size=8, color='black', transform=axis.transAxes, va='center', ha='center')
            axis.set_ylabel(r'Number of systems')
            axis.set_xlabel(lablxaxi)
            plt.subplots_adjust(bottom=0.2)
            path = gdat.pathimagmagt + 'hist%s_%d.%s' % (strgvarbmagt, a, gdat.strgplotextn)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()

    gdat.liststrgzoom = ['allr']
    if np.where((gdat.dicterrr['radiplan'][0, :] * gdat.factrjre > 2.) & (gdat.dicterrr['radiplan'][0, :] * gdat.factrjre < 4.))[0].size > 0:
        gdat.liststrgzoom.append('rb24')
    if np.where((gdat.dicterrr['radiplan'][0, :] * gdat.factrjre > 1.5) & (gdat.dicterrr['radiplan'][0, :] * gdat.factrjre < 4.))[0].size > 0:
        gdat.liststrgzoom.append('rb14')
        
    # mass radius
    for strgzoom in gdat.liststrgzoom:
        
        figr, axis = plt.subplots(figsize=gdat.figrsize)
        indx = np.where(np.isfinite(dictexarcomp['stdvmassplan']) & np.isfinite(dictexarcomp['stdvradiplan']))[0]
        
        xerr = dictexarcomp['stdvmassplan'][indx] * gdat.factmjme
        yerr = dictexarcomp['stdvradiplan'][indx] * gdat.factrjre
        # temp
        xerr = None
        yerr = None
        axis.errorbar(dictexarcomp['massplan'][indx] * gdat.factmjme, dictexarcomp['radiplan'][indx] * gdat.factrjre, lw=1, ls='', ms=1, marker='o', \
                                                                                            xerr=xerr, yerr=yerr, color='black')
        
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
        for jj, j in enumerate(gdat.indxplan):
            xerr = gdat.dicterrr['massplanpred'][1:3, j, None] * gdat.factmjme
            yerr = gdat.dicterrr['radiplan'][1:3, j, None] * gdat.factrjre
            axis.errorbar(gdat.dicterrr['massplanpred'][0, j, None] * gdat.factmjme, gdat.dicterrr['radiplan'][0, j, None] * gdat.factrjre, marker='o', \
                                                   xerr=xerr, ls='', yerr=yerr, lw=1, color=gdat.listcolrplan[j])
            axis.text(0.1, 0.9 - jj * 0.07, r'\textbf{%s}' % gdat.liststrgplan[j], color=gdat.listcolrplan[j], \
                                                                                        va='center', ha='center', transform=axis.transAxes)
        
        axis.set_ylabel(r'Radius [$R_E$]')
        axis.set_xlabel(r'Mass [$M_E$]')
        if strgzoom == 'rb14':
            axis.set_xlim([1.5, 16])
            axis.set_ylim([1.5, 4.])
        if strgzoom == 'rb24':
            axis.set_xlim([2., 16])
            axis.set_ylim([2., 4.])
        
        plt.subplots_adjust(bottom=0.2)
        path = gdat.pathimag + 'massradii_%s.%s' % (strgzoom, gdat.strgplotextn)
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()

    if gdat.numbplan > 1:
        # period ratios
        figr, axis = plt.subplots(figsize=gdat.figrsize)
        ## all 
        
        gdat.listratiperi = []
        gdat.intgreso = []
        
        liststrgstarcomp = []
        for m in indxplanexar:
            strgstar = dictexarcomp['namestar'][m]
            if not strgstar in liststrgstarcomp:
                indxexarstar = np.where(dictexarcomp['namestar'] == strgstar)[0]
                if indxexarstar[0] != m:
                    raise Exception('')
                
                listperi = dictexarcomp['peri'][None, indxexarstar]
                if not np.isfinite(listperi).all():
                    liststrgstarcomp.append(strgstar)
                    continue
                intgreso, ratiperi = retr_reso(listperi)
                
                numbplan = indxexarstar.size
                
                #print('strgstar')
                #print(strgstar)
                #if (ratiperi[0, :, :][np.tril_indices(numbplan, k=-1)] == 0).any():
                #    print('listperi')
                #    summgene(listperi)
                #    print('gdat.numbplan')
                #    print(gdat.numbplan)
                #    print('ratiperi')
                #    print(ratiperi)
                #    print('np.triu_indices(numbplan, k=1)')
                #    print(np.triu_indices(numbplan, k=1))
                #    print('ratiperi[0, :, :]')
                #    print(ratiperi[0, :, :])
                #    print('ratiperi[0, :, :][np.tril_indices(numbplan, k=-1)]')
                #    print(ratiperi[0, :, :][np.tril_indices(numbplan, k=-1)])
                #    print('ratiperi[0, :, :][np.triu_indices(numbplan, k=1)]')
                #    print(ratiperi[0, :, :][np.triu_indices(numbplan, k=1)])
                #print('')
                gdat.listratiperi.append(ratiperi[0, :, :][np.triu_indices(numbplan, k=1)])
                gdat.intgreso.append(intgreso)
                
                liststrgstarcomp.append(strgstar)
        
        gdat.listratiperi = np.concatenate(gdat.listratiperi)
        bins = np.linspace(1., 10., 200)
        axis.hist(gdat.listratiperi, bins=bins)
        
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
        plt.subplots_adjust()
        path = gdat.pathimag + 'ratiperi.%s' % gdat.strgplotextn
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()

    # radius histogram
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
    
    #listradickss = []
    ##path = gdat.pathbase + 'data/cks.txt'
    #objtfile = open(path, 'r')
    #cntr = 0
    #for line in objtfile:
    #    if cntr > 75:
    #        strgtemp = line.split('|')[8]
    #        try:
    #            listradickss.append(float(line.split('|')[8]))
    #        except:
    #            pass
    #    cntr += 1
    #listradickss = np.array(listradickss) # [R_E]
    
    print('Plotting the radii...')
    for strgzoom in gdat.liststrgzoom:
        figr, axis = plt.subplots(figsize=gdat.figrsize)
        
        # NASA Exoplanet Archive
        #temp = dictexarcomp['radiplan'] * gdat.factrjre
        
        # CKS sample
        xerr = (timeoccu[1:] - timeoccu[:-1]) / 2.
        xerr = np.concatenate([xerr[0, None], xerr])
        axis.errorbar(timeoccu, occumean, yerr=occuyerr, xerr=xerr, color='black', ls='', marker='o', lw=1)
        #temp = listradickss
        #indx = np.where(np.isfinite(temp))[0]
        #if strgzoom == 'rb14':
        #    binsradi = np.linspace(1.5, 4., 40)
        #elif strgzoom == 'rb24':
        #    binsradi = np.linspace(2., 4., 40)
        #else:
        #    radi = temp[indx]
        #    binsradi = np.linspace(np.amin(radi), np.amax(radi), 40)
        #meanradi = (binsradi[1:] + binsradi[:-1]) / 2.
        #deltradi = binsradi[1] - binsradi[0]
        #axis.bar(meanradi, np.histogram(temp[indx], bins=binsradi)[0], width=deltradi, color='grey')
        
        # this system
        for jj, j in enumerate(gdat.indxplan):
            xposlowr = gdat.factrjre *  gdat.dictpost['radiplan'][0, j]
            xposuppr = gdat.factrjre *  gdat.dictpost['radiplan'][2, j]
            axis.axvspan(xposlowr, xposuppr, alpha=0.5, color=gdat.listcolrplan[j])
            print('jj, j')
            print(jj, j)
            print('gdat.factrjre * gdat.dicterrr[radiplan][0, j]')
            print(gdat.factrjre * gdat.dicterrr['radiplan'][0, j])
            print('')
            axis.axvline(gdat.factrjre * gdat.dicterrr['radiplan'][0, j], color=gdat.listcolrplan[j], ls='--', label=gdat.liststrgplan[j])
            axis.text(0.7, 0.9 - jj * 0.07, r'\textbf{%s}' % gdat.liststrgplan[j], color=gdat.listcolrplan[j], \
                                                                                        va='center', ha='center', transform=axis.transAxes)
        axis.set_xlabel('Radius [$R_E$]')
        axis.set_ylabel('Occurrence rate of planets per star')
        plt.subplots_adjust(bottom=0.2)
        path = gdat.pathimag + 'histradi%s.%s' % (strgzoom, gdat.strgplotextn)
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()
   
        # radius vs period
        #figr, axis = plt.subplots(figsize=gdat.figrsize)
        #axis.errorbar(dictexarcomp['peri'][indx], temp[indx], ls='', lw=1)
        #for j in gdat.indxplan:
        #    xerr = gdat.dicterrr['peri'][1:3, j, None]
        #    yerr = gdat.factrjre * gdat.dicterrr['radiplan'][1:3, j, None]
        #    axis.errorbar(gdat.dicterrr['peri'][0, j, None], gdat.factrjre * gdat.dicterrr['radiplan'][0, j, None], color=gdat.listcolrplan[j], \
        #                                    lw=1, xerr=xerr, yerr=yerr, ls='', marker='o')
        #axis.set_xlabel('Period [days]')
        #axis.set_ylabel('Radius [$R_E$]')
        #if strgzoom == 'rb14':
        #    axis.set_ylim([1.5, 4.])
        #if strgzoom == 'rb24':
        #    axis.set_ylim([2., 4.])
        #axis.set_xlim([None, 100.])
        #plt.subplots_adjust(bottom=0.2)
        #path = gdat.pathimag + 'radiperi%s.%s' % (strgzoom, gdat.strgplotextn)
        #print('Writing to %s...' % path)
        #plt.savefig(path)
        #plt.close()
    
    tmptplan = dictexarcomp['tmptplan']
    radiplan = dictexarcomp['radiplan'] * gdat.factrjre # R_E
    
    # known planets
    ## ESM
    esmm = tesstarg.util.retr_esmm(dictexarcomp['tmptplan'], dictexarcomp['tmptstar'], dictexarcomp['radiplan'], dictexarcomp['radistar'], \
                                                                                                        dictexarcomp['kmagstar'])
    
    ## TSM
    tsmm = tesstarg.util.retr_tsmm(dictexarcomp['radiplan'], dictexarcomp['tmptplan'], dictexarcomp['massplan'], dictexarcomp['radistar'], \
                                                                                                        dictexarcomp['imagstar'])
    numbtext = 30
    liststrgmetr = ['tsmm']
    if np.isfinite(gdat.tmptstar):
        liststrgmetr.append('esmm')
    
    liststrglimt = ['allm', 'gmas']
    
    for strgmetr in liststrgmetr:
        
        if strgmetr == 'tsmm':
            metr = np.log(tsmm)
            lablmetr = '$\log$ TSM'
            metrthis = gdat.dicterrr['ltsm'][0, :]
            errrmetrthis = gdat.dicterrr['ltsm']
        else:
            metr = np.log(esmm)
            lablmetr = '$\log$ ESM'
            metrthis = gdat.dicterrr['lesm'][0, :]
            errrmetrthis = gdat.dicterrr['lesm']
        for strgzoom in gdat.liststrgzoom:
            if strgzoom == 'allr':
                indxzoom = np.where(np.isfinite(radiplan) & np.isfinite(tmptplan) \
                                                                            & np.isfinite(metr) & (dictexarcomp['booltran'] == 1.))[0]
            if strgzoom == 'rb14':
                indxzoom = np.where((radiplan < 4) & (radiplan > 1.5) & np.isfinite(radiplan) & np.isfinite(tmptplan) \
                                                                            & np.isfinite(metr) & (dictexarcomp['booltran'] == 1.))[0]
            if strgzoom == 'rb24':
                indxzoom = np.where((radiplan < 4) & (radiplan > 2.) & np.isfinite(radiplan) & np.isfinite(tmptplan) \
                                                                            & np.isfinite(metr) & (dictexarcomp['booltran'] == 1.))[0]
    
            for strglimt in liststrglimt:
                if strglimt == 'gmas':
                    indxlimt = np.where(dictexarcomp['stdvmassplan'] / dictexarcomp['massplan'] < 0.4)[0]
                if strglimt == 'allm':
                    indxlimt = np.arange(esmm.size)
                
                indx = np.intersect1d(indxzoom, indxlimt)
                
                maxmmetr = max(np.amax(metrthis), np.amax(metr[indx]))
                minmmetr = min(np.amin(metrthis), np.amin(metr[indx]))
                
                metr -= minmmetr
                metrthis -= minmmetr
                
                # sort
                indxsort = np.argsort(metr[indx])[::-1]
                
                if strgmetr == 'tsmm' and strgzoom == 'rb24' and strglimt == 'gmas':
                    print('strgmetr')
                    print(strgmetr)
                    print('strgzoom')
                    print(strgzoom)
                    print('strglimt')
                    print(strglimt)
                    listname = []
                    listmetrtemp = np.copy(metr[indx])
                    listnametemp = np.copy(dictexarcomp['nameplan'][indx])
                    listmetrtemp = np.concatenate((listmetrtemp, metrthis))
                    listnametemp = np.concatenate((listnametemp, gdat.liststrgplan))
                    print('listmetrtemp')
                    summgene(listmetrtemp)
                    print('listnametemp')
                    summgene(listnametemp)
                    indxsorttemp = np.argsort(listmetrtemp)[::-1]
                    print('')
                    for k in indxsorttemp:
                        print('%s: %g' % (listnametemp[k], listmetrtemp[k]))
                    print('')

                # repeat, one without text, one with text
                for b in range(2):
                    # radius vs equilibrium temperature
                    figr, axis = plt.subplots(figsize=gdat.figrsize)
                    
                    ## known planets
                    axis.errorbar(radiplan[indx], tmptplan[indx], ms=1, ls='', marker='o', color='black')
                    if b == 1:
                        for k in indxsort[:numbtext]:
                            objttext = axis.text(radiplan[indx[k]], tmptplan[indx[k]], '%s' % dictexarcomp['nameplan'][indx[k]], size=1, \
                                                                                                                               ha='center', va='center')
                    
                    ## this system
                    for j in gdat.indxplan:
                        xerr = gdat.factrjre * gdat.dicterrr['radiplan'][1:3, j, None]
                        yerr = gdat.dicterrr['tmptplan'][1:3, j, None]
                        axis.errorbar(gdat.dicterrr['radiplan'][0, j, None] * gdat.factrjre, gdat.dicterrr['tmptplan'][0, j, None], ls='', \
                                    ms=6, lw=1, xerr=xerr, yerr=yerr, marker='o', \
                                                                                                color=gdat.listcolrplan[j])
                        for jj, j in enumerate(gdat.indxplan):
                            axis.text(0.85, 0.9 - jj * 0.07, r'\textbf{%s}' % gdat.liststrgplan[j], color=gdat.listcolrplan[j], \
                                                                                    va='center', ha='center', transform=axis.transAxes)

                    if strgzoom == 'rb14':
                        axis.set_xlim([1.5, 4.])
                    if strgzoom == 'rb24':
                        axis.set_xlim([2., 4.])
                    
                    axis.set_ylabel(r'Planet Equilibrium Temperature [K]')
                    axis.set_xlabel('Radius [$R_E$]')
                    plt.tight_layout()
                    path = gdat.pathimag + 'radiplan_tmptplan_%s_%s_targ_%d.%s' % (strgzoom, strglimt, b, gdat.strgplotextn)
                    print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()
        
                    # metric vs radius
                    figr, axis = plt.subplots(figsize=gdat.figrsize)
                    
                    ## known planets
                    axis.errorbar(radiplan[indx], metr[indx], ms=1, ls='', marker='o', color='black')
                    if b == 1:
                        for k in indxsort[:numbtext]:
                            axis.text(radiplan[indx[k]], metr[indx[k]], '%s' % dictexarcomp['nameplan'][indx[k]], ha='center', va='center', size=1)
                    
                    ## this system
                    for jj, j in enumerate(gdat.indxplan):
                        xerr = gdat.factrjre * gdat.dicterrr['radiplan'][1:3, j, None]
                        axis.errorbar(gdat.dicterrr['radiplan'][0, j, None] * gdat.factrjre, \
                                                metrthis[j, None], lw=1, ms=6, ls='', marker='o', \
                                xerr=xerr, yerr=errrmetrthis[1:3, j, None], color=gdat.listcolrplan[j])
                        if not (strgzoom == 'rb24' and gdat.dicterrr['radiplan'][0, j, None] * gdat.factrjre < 2):
                            axis.text(0.85, 0.9 - jj * 0.07, r'\textbf{%s}' % gdat.liststrgplan[j], color=gdat.listcolrplan[j], \
                                                                                        va='center', ha='center', transform=axis.transAxes)
                    
                    if strgzoom == 'rb14':
                        axis.set_xlim([1.5, 4.])
                    if strgzoom == 'rb24':
                        axis.set_xlim([2., 4.])
                    axis.set_ylabel(lablmetr)
                    axis.set_xlabel('Radius [$R_E$]')
                    #axis.set_yscale('log')
                    plt.tight_layout()
                    path = gdat.pathimag + '%s_radiplan_%s_%s_targ_%d.%s' % (strgmetr, strgzoom, strglimt, b, gdat.strgplotextn)
                    print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()
        
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

    if strgalle == '0003' or strgalle == '0004' or strgalle == '0006':
        gdat.arrylcurmodlcomp = [[] for p in gdat.indxinst]
        gdat.arrypcurquadmodlcomp = [[[] for j in gdat.indxplan] for p in gdat.indxinst]
        for p in gdat.indxinst:
            if strgalle == '0003' or strgalle == '0006': 
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
                if strgalle == '0003' or strgalle == '0006': 
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
                tdpy.mcmc.plot_grid(gdat.pathimagpcur, 'pcur_%s' % strgalle, listpost, listlablpara, plotsize=2.5)

            # plot phase curve
            ## determine data gaps for overplotting model without the data gaps
            gdat.indxtimegapp = np.argmax(gdat.time[1:] - gdat.time[:-1]) + 1
            figr = plt.figure(figsize=(10, 12))
            axis = [[] for k in range(3)]
            axis[0] = figr.add_subplot(3, 1, 1)
            axis[1] = figr.add_subplot(3, 1, 2)
            axis[2] = figr.add_subplot(3, 1, 3, sharex=axis[1])
            
            for k in range(len(axis)):
                
                ## unbinned data
                if k < 2:
                    if k == 0:
                        xdat = gdat.time - gdat.timetess
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
            gdat.objtalle[strgalle] = allesfitter.allesclass(gdat.pathalle[strgalle])
            gdat.objtalle[strgalle].posterior_params_median['b_phase_curve_ellipsoidal_TESS'] = 0
            if strgalle == '0003' or strgalle == '0006':
                gdat.objtalle[strgalle].posterior_params_median['b_phase_curve_atmospheric_TESS'] = 0
            if strgalle == '0004':
                gdat.objtalle[strgalle].posterior_params_median['b_phase_curve_atmospheric_thermal_TESS'] = 0
                gdat.objtalle[strgalle].posterior_params_median['b_phase_curve_atmospheric_reflected_TESS'] = 0
            gdat.objtalle[strgalle].posterior_params_median['b_sbratio_TESS'] = 0
            arrylcurmodltemp[:, 1] = gdat.objtalle[strgalle].get_posterior_median_model(strginst, 'flux', xx=gdat.time)
            gdat.arrypcurquadmodlcomp[p][j]['stel'] = tesstarg.util.fold_lcur(arrylcurmodltemp, \
                                                                    gdat.dicterrr['epoc'][0, j], gdat.dicterrr['peri'][0, j], phasshft=0.25)
            xdat = gdat.arrypcurquadmodlcomp[p][j]['stel'][:, 0]
            ydat = (gdat.arrypcurquadmodlcomp[p][j]['stel'][:, 1] - 1.) * 1e6
            axis[2].plot(xdat, ydat, lw=2, color='orange', label='Stellar baseline', ls='--', zorder=11)
            
            ### EV
            gdat.objtalle[strgalle] = allesfitter.allesclass(gdat.pathalle[strgalle])
            gdat.objtalle[strgalle].posterior_params_median['b_sbratio_TESS'] = 0
            if strgalle == '0003' or strgalle == '0006':
                gdat.objtalle[strgalle].posterior_params_median['b_phase_curve_atmospheric_TESS'] = 0
            if strgalle == '0004':
                gdat.objtalle[strgalle].posterior_params_median['b_phase_curve_atmospheric_thermal_TESS'] = 0
                gdat.objtalle[strgalle].posterior_params_median['b_phase_curve_atmospheric_reflected_TESS'] = 0
            arrylcurmodltemp[:, 1] = gdat.objtalle[strgalle].get_posterior_median_model(strginst, 'flux', xx=gdat.time)
            gdat.arrypcurquadmodlcomp[p][j]['elli'] = tesstarg.util.fold_lcur(arrylcurmodltemp, \
                                                                      gdat.dicterrr['epoc'][0, j], gdat.dicterrr['peri'][0, j], phasshft=0.25)
            gdat.arrypcurquadmodlcomp[p][j]['elli'][:, 1] -= gdat.arrypcurquadmodlcomp[p][j]['stel'][:, 1]
            xdat = gdat.arrypcurquadmodlcomp[p][j]['elli'][:, 0]
            ydat = (gdat.arrypcurquadmodlcomp[p][j]['elli'][:, 1] - 1.) * 1e6
            axis[2].plot(xdat, ydat, lw=2, color='r', ls='--', label='Ellipsoidal variation')
            
            # planetary
            gdat.objtalle[strgalle] = allesfitter.allesclass(gdat.pathalle[strgalle])
            gdat.objtalle[strgalle].posterior_params_median['b_phase_curve_ellipsoidal_TESS'] = 0
            arrylcurmodltemp[:, 1] = gdat.objtalle[strgalle].get_posterior_median_model(strginst, 'flux', xx=gdat.time)
            gdat.arrypcurquadmodlcomp[p][j]['plan'] = tesstarg.util.fold_lcur(arrylcurmodltemp, \
                                                                    gdat.dicterrr['epoc'][0, j], gdat.dicterrr['peri'][0, j], phasshft=0.25)
            gdat.arrypcurquadmodlcomp[p][j]['plan'] += gdat.dicterrr['amplnigh'][0, 0]
            gdat.arrypcurquadmodlcomp[p][j]['plan'][:, 1] -= gdat.arrypcurquadmodlcomp[p][j]['stel'][:, 1]
            
            xdat = gdat.arrypcurquadmodlcomp[p][j]['plan'][:, 0]
            ydat = (gdat.arrypcurquadmodlcomp[p][j]['plan'][:, 1] - 1.) * 1e6
            axis[2].plot(xdat, ydat, lw=2, color='g', label='Planetary', ls='--')
    
            # planetary nightside
            gdat.objtalle[strgalle] = allesfitter.allesclass(gdat.pathalle[strgalle])
            gdat.objtalle[strgalle].posterior_params_median['b_phase_curve_ellipsoidal_TESS'] = 0
            if strgalle == '0003' or strgalle == '0006':
                gdat.objtalle[strgalle].posterior_params_median['b_phase_curve_atmospheric_TESS'] = 0
            if strgalle == '0004':
                gdat.objtalle[strgalle].posterior_params_median['b_phase_curve_atmospheric_thermal_TESS'] = 0
                gdat.objtalle[strgalle].posterior_params_median['b_phase_curve_atmospheric_reflected_TESS'] = 0
            arrylcurmodltemp[:, 1] = gdat.objtalle[strgalle].get_posterior_median_model(strginst, 'flux', xx=gdat.time)
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
            
            path = gdat.pathimagpcur + 'pcur_grid_%s.%s' % (strgalle, gdat.strgplotextn)
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
            path = gdat.pathimagpcur + 'pcur_comp_%s.%s' % (strgalle, gdat.strgplotextn)
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
            path = gdat.pathimagpcur + 'pcur_samp_%s.%s' % (strgalle, gdat.strgplotextn)
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
            #path = gdat.pathimagpcur + 'pcur_resi_%s.%s' % (strgalle, gdat.strgplotextn)
            #print('Writing to %s...' % path)
            #plt.savefig(path)
            #plt.close()

            # write to text file
            fileoutp = open(gdat.pathdatapcur + 'post_pcur_%s_tabl.csv' % (strgalle), 'w')
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
            
            fileoutp = open(gdat.pathdatapcur + 'post_pcur_%s_cmnd.csv' % (strgalle), 'w')
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

            if gdat.labltarg == 'WASP-121' and strgalle == '0003':
                
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
  
    
def init( \
         ticitarg=None, \
         strgmast=None, \
         toiitarg=None, \
         
         # a string for the label of the target
         labltarg=None, \
         
         # a string for the folder name and file name extensions
         strgtarg=None, \
        
         booloffl=False, \

         boolphascurv=False, \
        
         boolclip=False, \

         # type of light curve to be used for analysis
         datatype=None, \
                    
         # Boolean flag to use SAP instead of PDC by default, when SPOC data is being used.
         boolsapp=False, \
         
         boolexar=None, \
        
         maxmnumbstarpand=1, \
         
         # whether to mask bad data
         boolmaskqual=True, \

         dilucorr=None, \

         # type of priors used for allesfitter
         priotype=None, \
    
         durabrek=1., \
         ordrspln=3, \

         rratprio=None, \
         rsmaprio=None, \
         epocprio=None, \
         periprio=None, \
         cosiprio=None, \
         ecosprio=None, \
         esinprio=None, \
         rvsaprio=None, \
        
         radistarstdv=None, \

         listpathdatainpt=None, \
        
         radistar=None, \
         massstar=None, \
         tmptstar=None, \
    
         liststrgchun=None, \
         liststrginst=['TESS'], \
         listlablinst=['TESS'], \
         listindxchuninst=None, \

         # type of inference, alle for allesfitter, trap for trapezoidal fit
         infetype='alle', \

         boolcontrati=False, \
        
         makeprioplot=True, \
         
         # list of offsets for the planet annotations in the TSM/ESM plot
         offstextatmoraditmpt=None, \
         offstextatmoradimetr=None, \

         ## baseline
         bdtrtype='spln', \
         weigsplnbdtr=1., \
         durakernbdtrmedi=1., \
         
         boolallebkgdgaus=False, \
         boolalleorbt=True, \

         maxmnumbplantlss=None, \

         liststrgplan=None, \

         booldiagmode=True, \
         
         # allesfitter analysis type
         strgalletype = 'ther', \
    
         listlimttimemask=None, \
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

    # settings
    ## plotting
    gdat.timetess = 2457000.
    
    print('PEXO initialized at %s...' % gdat.strgtimestmp)
    
    if gdat.ticitarg is None and gdat.strgmast is None and gdat.toiitarg is None:
        raise Exception('')
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
    
    dictexartarg = retr_exarcomp(gdat, strgtarg=gdat.strgmast)
    gdat.boolexar = gdat.strgmast is not None and dictexartarg is not None
    
    if gdat.boolexar and gdat.priotype is None:
        print('The planet name was found in the NASA Exoplanet Archive "composite" table.')
        # stellar properties
        
        gdat.priotype = 'exar'
        if gdat.periprio is None:
            gdat.periprio = dictexartarg['peri']
        gdat.smaxprio = dictexartarg['smax']
        gdat.massplanprio = dictexartarg['massplan']
        
        #gdat.radistar = dictexartarg['radistar']
        #gdat.massstar = dictexartarg['massstar']
        #gdat.tmptstar = dictexartarg['tmptstar']
        #gdat.radistarstdv = dictexartarg['radistarstdv']
        #gdat.massstarstdv = dictexartarg['massstarstdv']
        #gdat.tmptstarstdv = dictexartarg['tmptstarstdv']
        
    else:
        print('The planet name was *not* found in the Exoplanet Archive "composite" table.')
    
    gdat.numbinst = len(gdat.liststrginst)
    gdat.indxinst = np.arange(gdat.numbinst)

    #if len(gdat.radistar) > 0:
    #    raise Exception('')

    # read the NASA Exoplanet Archive planets
    path = gdat.pathbase + 'data/planets_2020.01.23_16.21.55.csv'
    print('Reading %s...' % path)
    objtexarplan = pd.read_csv(path, skiprows=359, low_memory=False)
    indx = np.where(objtexarplan['pl_hostname'].values == gdat.strgmast)[0]
    if indx.size == 0:
        print('The planet name was *not* found in the NASA Exoplanet Archive "planets" table.')
        gdat.boolexar = False
    else:
        gdat.priotype = 'exar'
        gdat.deptprio = objtexarplan['pl_trandep'][indx].values
        if gdat.cosiprio is None:
            gdat.cosiprio = np.cos(objtexarplan['pl_orbincl'][indx].values / 180. * np.pi)
        if gdat.epocprio is None:
            gdat.epocprio = objtexarplan['pl_tranmid'][indx].values # [days]
        gdat.duraprio = objtexarplan['pl_trandur'][indx].values # [days]
    
    if gdat.toiitarg is not None:
        
        gdat.priotype = 'exof'
        print('A TOI number is provided. Retreiving the TCE attributes from ExoFOP-TESS...')
        # read ExoFOP-TESS
        path = gdat.pathbase + 'data/exofop_toilists.csv'
        print('Reading from %s...' % path)
        objtexof = pd.read_csv(path, skiprows=0)
        indx = []
        gdat.strgtoiibase = str(gdat.toiitarg)
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
    print('gdat.priotype')
    print(gdat.priotype)
    if gdat.priotype == 'inpt':
        if gdat.rratprio is None:
            gdat.rratprio = 0.1 + np.zeros(gdat.numbplan)
        if gdat.rsmaprio is None:
            gdat.rsmaprio = 0.1 + np.zeros(gdat.numbplan)
        if gdat.cosiprio is None:
            gdat.cosiprio = np.zeros(gdat.numbplan)
        gdat.duraprio = gdat.periprio / np.pi * np.arcsin(gdat.rsmaprio**2 - gdat.cosiprio**2)
        gdat.deptprio = gdat.rratprio**2
    
    if gdat.ecosprio is None:
        gdat.ecosprio = np.zeros_like(gdat.periprio)
    if gdat.esinprio is None:
        gdat.esinprio = np.zeros_like(gdat.periprio)
    if gdat.rvsaprio is None:
        gdat.rvsaprio = np.zeros_like(gdat.periprio)
    
    if gdat.priotype == None:
        if gdat.epocprio is None:
            gdat.priotype = 'tlss'
        else:
            gdat.priotype = 'inpt'
    
    if gdat.priotype == 'exof' and gdat.toiitarg is None:
        raise Exception('')

    gdat.numbplan = None
    
    # check MAST
    print('gdat.strgmast')
    print(gdat.strgmast)
    if gdat.strgmast is None:
        print('gdat.strgmast was not provided as input. Using the TIC ID to construct gdat.strgmast.')
        gdat.strgmast = 'TIC %d' % gdat.ticitarg
    
    if gdat.booloffl:
        catalogData = astroquery.mast.Catalogs.query_object(gdat.strgmast, catalog='TIC', radius='5m')
        if catalogData[0]['dstArcSec'] < 0.002:
            print('Found the target on MAST!')
            rasc = catalogData[0]['ra']
            decl = catalogData[0]['dec']
            if gdat.radistar is None:
                print('Setting the stellar radius from the TIC.')
                gdat.radistar = catalogData[0]['rad'] * gdat.factrsrj
                gdat.radistarstdv = catalogData[0]['e_rad'] * gdat.factrsrj
                if not np.isfinite(gdat.radistar):
                    raise Exception('TIC stellar radius is not finite.')
                if not np.isfinite(gdat.radistar):
                    raise Exception('TIC stellar radius uncertainty is not finite.')
            if gdat.massstar is None:
                print('Setting the stellar mass from the TIC.')
                gdat.massstar = catalogData[0]['mass'] * gdat.factmsmj
                gdat.massstarstdv = catalogData[0]['e_mass'] * gdat.factmsmj
                if not np.isfinite(gdat.massstar):
                    raise Exception('TIC stellar mass is not finite.')
                if not np.isfinite(gdat.massstar):
                    raise Exception('TIC stellar mass uncertainty is not finite.')
            if gdat.tmptstar is None:
                print('Setting the stellar temperature from the TIC.')
                gdat.tmptstar = catalogData[0]['Teff']
                gdat.tmptstarstdv = catalogData[0]['e_Teff']
                if not np.isfinite(gdat.tmptstar):
                    raise Exception('TIC stellar temperature is not finite.')
                if not np.isfinite(gdat.tmptstar):
                    raise Exception('TIC stellar temperature uncertainty is not finite.')
            gdat.imagstar = catalogData[0]['Jmag']
            gdat.omagstar = catalogData[0]['Vmag']
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
    gdat.pathdata = gdat.pathobjt + 'data/'
    gdat.pathimag = gdat.pathobjt + 'imag/'
    os.system('mkdir -p %s' % gdat.pathdata)
    os.system('mkdir -p %s' % gdat.pathimag)

    print('gdat.datatype')
    print(gdat.datatype)
    if gdat.listpathdatainpt is None:
        gdat.numbchun = 1
        gdat.arrylcur = [[]]
        gdat.datatype, gdat.arrylcur[0], gdat.arrylcursapp, gdat.arrylcurpdcc, gdat.listarrylcur, gdat.listarrylcursapp, \
                                   gdat.listarrylcurpdcc, gdat.listisec, gdat.listicam, gdat.listiccd = \
                                        tesstarg.util.retr_data(gdat.datatype, gdat.strgmast, gdat.pathobjt, gdat.boolsapp, boolmaskqual=gdat.boolmaskqual, \
                                        labltarg=gdat.labltarg, strgtarg=gdat.strgtarg, ticitarg=gdat.ticitarg, maxmnumbstarpand=gdat.maxmnumbstarpand)
    
    else:
        gdat.numbchun = len(gdat.listpathdatainpt)
    
    gdat.indxchun = [[] for p in gdat.indxinst]
    for p in gdat.indxinst:
        gdat.indxchun[p] = np.arange(gdat.numbchun, dtype=int)

    if gdat.listpathdatainpt is not None:
        gdat.datatype = 'inpt'
        gdat.listarrylcur = []
        gdat.arrylcur = [[] for  y in range(len(gdat.listpathdatainpt))]
        for y in range(len(gdat.listpathdatainpt)):
            print('gdat.listpathdatainpt[y]')
            print(gdat.listpathdatainpt[y])
            arry = np.loadtxt(gdat.listpathdatainpt[y], delimiter=',', skiprows=1)
            gdat.numbtime = arry.shape[0]
            gdat.arrylcur[y] = np.empty((gdat.numbtime, 3))
            gdat.arrylcur[y][:, 0:2] = arry[:, 0:2]
            gdat.arrylcur[y][:, 2] = 1e-4 * arry[:, 1]
            indx = np.argsort(gdat.arrylcur[y][:, 0])
            gdat.arrylcur[y] = gdat.arrylcur[y][indx, :]
            indx = np.where(gdat.arrylcur[y][:, 1] < 1e6)[0]
            gdat.arrylcur[y] = gdat.arrylcur[y][indx, :]
            gdat.listarrylcur.append(gdat.arrylcur[y])
            gdat.listisec = None
            
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
            axis.plot(gdat.arrylcur[y][:, 0] - gdat.timetess, gdat.arrylcur[y][:, 1], color='grey', marker='.', ls='', ms=1)
            axis.set_xlabel('Time [BJD - 2457000]')
            axis.set_ylabel('Relative Flux')
            plt.subplots_adjust(bottom=0.2)
            path = gdat.pathimag + 'lcurraww%s.%s' % (gdat.liststrgchun[y], gdat.strgplotextn)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
   
    print('gdat.datatype')
    print(gdat.datatype)
    gdat.numbsect = len(gdat.listarrylcur)
    gdat.indxsect = np.arange(gdat.numbsect)
    gdat.boolbdtr = gdat.datatype != 'pdcc'
    if gdat.duraprio is not None:
        epocmask = gdat.epocprio
        perimask = gdat.periprio
        duramask = 2. * gdat.duraprio
    else:
        epocmask = None
        perimask = None
        duramask = None
    gdat.listarrylcurbdtr = [[] for y in gdat.indxchun]
    for p in gdat.indxinst:
        for y in gdat.indxchun[p]:
            if gdat.boolbdtr:
                
                lcurbdtrregi, indxtimeregi, indxtimeregioutt, listobjtspln, timeedge = \
                                       tesstarg.util.bdtr_lcur(gdat.listarrylcur[y][:, 0], gdat.listarrylcur[y][:, 1], weigsplnbdtr=gdat.weigsplnbdtr, \
                                                                 epocmask=epocmask, perimask=perimask, verbtype=1, durabrek=durabrek, ordrspln=ordrspln, \
                                                                        duramask=duramask)#, durabrek=gdat.durakernbdtrmedi)
                gdat.listarrylcurbdtr[y] = np.copy(gdat.arrylcur[y])
                gdat.listarrylcurbdtr[y][:, 1] = np.concatenate(lcurbdtrregi)
                numbsplnregi = len(lcurbdtrregi)
                indxsplnregi = np.arange(numbsplnregi)
                
            else:
                gdat.listarrylcurbdtr[y] = gdat.arrylcur[y]
            
            if not np.isfinite(gdat.listarrylcurbdtr[y]).all():
                raise Exception('')

            if False and gdat.booldiagmode:
                for a in range(gdat.arrylcur[y][:, 0].size):
                    if a != gdat.arrylcur[y][:, 0].size - 1 and gdat.arrylcur[y][a, 0] >= gdat.arrylcur[y][a+1, 0]:
                        raise Exception('')
                    if a != gdat.listarrylcurbdtr[y][:, 0].size - 1 and gdat.listarrylcurbdtr[y][a, 0] >= gdat.listarrylcurbdtr[y][a+1, 0]:
                        raise Exception('')
    
    if isinstance(gdat.listarrylcurbdtr[y], list):
        raise(Exception)

    print('gdat.priotype')
    print(gdat.priotype)
    
    # sigma-clip the light curve
    # temp -- this does not work properly!
    for p in gdat.indxinst:
        for y in gdat.indxchun[p]:
            if gdat.boolclip:
                lcurclip, lcurcliplowr, lcurclipuppr = scipy.stats.sigmaclip(gdat.listarrylcurbdtr[y][:, 1], low=5., high=5.)
                print('Clipping the light curve at %g and %g...' % (lcurcliplowr, lcurclipuppr))
                indx = np.where((gdat.listarrylcurbdtr[y][:, 1] < lcurclipuppr) & (gdat.listarrylcurbdtr[y][:, 1] > lcurcliplowr))[0]
                gdat.listarrylcurbdtr[y] = gdat.listarrylcurbdtr[y][indx, :]
            
                path = gdat.pathdata + 'arrylcurclip.csv'
                print('Writing to %s...' % path)
                np.savetxt(path, gdat.listarrylcurbdtr[y], delimiter=',', header='time,flux,flux_err')
    
            if gdat.priotype == 'tlss':
                dicttlss = tesstarg.util.exec_tlss(gdat.listarrylcurbdtr[y], gdat.pathimag, numbplan=gdat.numbplan, maxmnumbplantlss=gdat.maxmnumbplantlss, \
                                                strgplotextn=gdat.strgplotextn, figrsize=gdat.figrsizeydob, figrsizeydobskin=gdat.figrsizeydobskin)
                if gdat.epocprio is None:
                    gdat.epocprio = dicttlss['epoc']
                if gdat.periprio is None:
                    gdat.periprio = dicttlss['peri']
                gdat.deptprio = dicttlss['dept']
                gdat.duraprio = dicttlss['dura']
                
                gdat.rratprio = np.sqrt(gdat.deptprio)
                gdat.rsmaprio = np.sin(np.pi * gdat.duraprio / gdat.periprio)
    
    if not np.isfinite(gdat.duraprio).all():
        gdat.duraprio = tesstarg.util.retr_dura(gdat.periprio, gdat.rsmaprio, gdat.cosiprio)

    if gdat.rratprio is None:
        gdat.rratprio = np.sqrt(gdat.deptprio)

    if gdat.rsmaprio is None:
        gdat.rsmaprio = np.sqrt(np.sin(np.pi * gdat.duraprio / gdat.periprio)**2 + gdat.cosiprio**2)
        
    # plot raw data
    if gdat.datatype == 'pdcc' or gdat.datatype == 'sapp':
        path = gdat.pathdata + gdat.liststrgchun[y] + '_SAP.csv'
        print('Writing to %s...' % path)
        np.savetxt(path, gdat.arrylcursapp, delimiter=',', header='time,flux,flux_err')
        path = gdat.pathdata + gdat.liststrgchun[y] + '_PDCSAP.csv'
        print('Writing to %s...' % path)
        np.savetxt(path, gdat.arrylcurpdcc, delimiter=',', header='time,flux,flux_err')
        
        # plot PDCSAP and SAP light curves
        figr, axis = plt.subplots(2, 1, figsize=gdat.figrsizeydob)
        axis[0].plot(gdat.arrylcursapp[:, 0] - gdat.timetess, gdat.arrylcursapp[:, 1], color='k', marker='.', ls='', ms=1)
        if listlimttimemask is not None:
            axis[0].plot(gdat.arrylcursapp[listindxtimegood, 0] - gdat.timetess, gdat.arrylcursapp[listindxtimegood, 1], color='k', marker='.', ls='', ms=1)
        axis[1].plot(gdat.arrylcurpdcc[:, 0] - gdat.timetess, gdat.arrylcurpdcc[:, 1], color='k', marker='.', ls='', ms=1)
        if listlimttimemask is not None:
            axis[1].plot(gdat.arrylcurpdcc[listindxtimegood, 0] - gdat.timetess, gdat.arrylcurpdcc[listindxtimegood, 1], color='k', marker='.', ls='', ms=1)
        #axis[0].text(.97, .97, 'SAP', transform=axis[0].transAxes, size=20, color='r', ha='right', va='top')
        #axis[1].text(.97, .97, 'PDC', transform=axis[1].transAxes, size=20, color='r', ha='right', va='top')
        axis[1].set_xlabel('Time [BJD - 2457000]')
        for a in range(2):
            axis[a].set_ylabel('Relative Flux')
        
        #for j in gdat.indxplan:
        #    colr = gdat.listcolrplan[j]
        #    axis[1].plot(gdat.arrylcurpdcc[gdat.listindxtimetran[j], 0] - gdat.timetess, gdat.arrylcurpdcc[gdat.listindxtimetran[j], 1], \
        #                                                                                                         color=colr, marker='.', ls='', ms=1)
        plt.subplots_adjust(hspace=0.)
        path = gdat.pathimag + 'lcurspoc.%s' % gdat.strgplotextn
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()
    
    # injection recovery test
    #periods = [1,2,3,4,5] #list of injection periods in days
    #rplanets = [1,2,3,4,5] #list of injection rplanets in Rearth
    #logfname = 'injection_recovery_test.csv'
    
    print('gdat.radistar')
    print(gdat.radistar)
    print('gdat.radistarstdv')
    print(gdat.radistarstdv)
    print('gdat.radistar [R_S]')
    print(gdat.radistar / gdat.factrsrj)
    print('gdat.radistarstdv [R_S]')
    print(gdat.radistarstdv / gdat.factrsrj)
    print('gdat.massstar')
    print(gdat.massstar)
    print('gdat.massstarstdv')
    print(gdat.massstarstdv)
    print('gdat.massstar [M_S]')
    print(gdat.massstar / gdat.factmsmj)
    print('gdat.massstarstdv [M_S]')
    print(gdat.massstarstdv / gdat.factmsmj)
    print('gdat.tmptstar')
    print(gdat.tmptstar)
    print('gdat.tmptstarstdv')
    print(gdat.tmptstarstdv)
    
    print('gdat.deptprio')
    print(gdat.deptprio)
    print('gdat.duraprio')
    print(gdat.duraprio)
    
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
    
    #gdat.massplanprio = np.zeros_like(gdat.epocprio)
    gdat.massplanprio = None
    
    gdat.ecceprio = np.zeros_like(gdat.epocprio)
    print('gdat.ecceprio')
    print(gdat.ecceprio)
    

    if not np.isfinite(gdat.rratprio).all() or \
           not np.isfinite(gdat.rsmaprio).all() or \
           not np.isfinite(gdat.epocprio).all() or \
           not np.isfinite(gdat.periprio).all() or \
           not np.isfinite(gdat.cosiprio).all() or \
           not np.isfinite(gdat.ecosprio).all() or \
           not np.isfinite(gdat.esinprio).all() or \
           not np.isfinite(gdat.rvsaprio).all():
            raise Exception('')
    
    # settings
    gdat.numbplan = gdat.epocprio.size
    gdat.indxplan = np.arange(gdat.numbplan)
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
    
    if gdat.dilucorr is not None and gdat.boolcontrati:
        print('Calculating the contamination ratio...')
        tmagfrst = catalogData[0]['Tmag']
        frstflux = 10**(-tmagfrst)
        totlflux = 0.
        for k in range(1, len(catalogData)):
            dist = np.sqrt((catalogData[k]['ra'] - rasc)**2 + (catalogData[k]['dec'] - decl)**2) * 3600.
            tmag = catalogData[k]['Tmag']
            totlfluxtemp = 10**(-tmag) * np.exp(-(dist / 13.)**2)
            totlflux += totlfluxtemp
        gdat.contrati = totlflux / frstflux

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
    
    gdat.duramask = 2. * gdat.duraprio
    print('gdat.duramask')
    print(gdat.duramask)
    
    for p in gdat.indxinst:
        for y in gdat.indxchun[p]:
            if listlimttimemask is not None:
                # mask the data
                print('Masking the data...')
                gdat.arrylcurumsk = np.copy(gdat.listarrylcurbdtr[y])
                numbmask = listlimttimemask.shape[0]
                listindxtimemask = []
                for k in range(numbmask):
                    indxtimemask = np.where((gdat.listarrylcurbdtr[y][:, 0] < listlimttimemask[k, 1]) & (gdat.listarrylcurbdtr[y][:, 0] > listlimttimemask[k, 0]))[0]
                    listindxtimemask.append(indxtimemask)
                listindxtimemask = np.concatenate(listindxtimemask)
                listindxtimegood = np.setdiff1d(gdat.indxtime, listindxtimemask)
                gdat.listarrylcurbdtr[y] = gdat.listarrylcurbdtr[y][listindxtimegood, :]
    
    # correct for dilution
    #print('Correcting for dilution!')
    #if gdat.dilucorr is not None or gdat.boolcontrati:
    #    gdat.arrylcurdilu = np.copy(gdat.listarrylcurbdtr[y])
    #if gdat.dilucorr is not None:
    #    gdat.arrylcurdilu[:, 1] = 1. - gdat.dilucorr * (1. - gdat.listarrylcurbdtr[y][:, 1])
    #if gdat.boolcontrati:
    #    gdat.arrylcurdilu[:, 1] = 1. - gdat.contrati * gdat.contrati * (1. - gdat.listarrylcurbdtr[y][:, 1])
    #if gdat.dilucorr is not None or gdat.boolcontrati:
    #    gdat.listarrylcurbdtr[y] = np.copy(gdat.arrylcurdilu) 
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
        
        
    ## phase-fold and save the baseline-detrended light curve
    gdat.listarrylcurbdtrbind = [[] for y in gdat.indxchun]
    for p in gdat.indxinst:
        for y in gdat.indxchun[p]:
            gdat.numbbinspcurfine = 1000
            gdat.numbbinspcur = 100
            gdat.numbbinslcur = 1000
            gdat.listarrylcurbdtrbind[y] = tesstarg.util.rebn_lcur(gdat.listarrylcurbdtr[y], gdat.numbbinslcur)
            
            path = gdat.pathdata + 'arrylcurbdtrchu%d.csv' % y
            print('Writing to %s' % path)
            np.savetxt(path, gdat.listarrylcurbdtr[y], delimiter=',', header='time,flux,flux_err')
            
            path = gdat.pathdata + 'arrylcurbdtrbindchu%d.csv' % y
            print('Writing to %s' % path)
            np.savetxt(path, gdat.listarrylcurbdtrbind[y], delimiter=',', header='time,flux,flux_err')
            
            gdat.arrypcurprimbdtr = [[[] for j in gdat.indxplan] for p in gdat.indxinst]
            gdat.arrypcurprimbdtrbind = [[[] for j in gdat.indxplan] for p in gdat.indxinst]
            gdat.arrypcurprimbdtrbindfine = [[[] for j in gdat.indxplan] for p in gdat.indxinst]
            gdat.arrypcurquadbdtr = [[[] for j in gdat.indxplan] for p in gdat.indxinst]
            gdat.arrypcurquadbdtrbind = [[[] for j in gdat.indxplan] for p in gdat.indxinst]
            gdat.arrypcurquadbdtrbindfine = [[[] for j in gdat.indxplan] for p in gdat.indxinst]
            for j in gdat.indxplan:
                gdat.arrypcurprimbdtr[p][j] = tesstarg.util.fold_lcur(gdat.listarrylcurbdtr[y], gdat.epocprio[j], gdat.periprio[j])
                gdat.arrypcurprimbdtrbind[p][j] = tesstarg.util.rebn_lcur(gdat.arrypcurprimbdtr[p][j], gdat.numbbinspcur)
                gdat.arrypcurprimbdtrbindfine[p][j] = tesstarg.util.rebn_lcur(gdat.arrypcurprimbdtr[p][j], gdat.numbbinspcurfine)
                gdat.arrypcurquadbdtr[p][j] = tesstarg.util.fold_lcur(gdat.listarrylcurbdtr[y], gdat.epocprio[j], gdat.periprio[j], phasshft=0.25)
                gdat.arrypcurquadbdtrbind[p][j] = tesstarg.util.rebn_lcur(gdat.arrypcurquadbdtr[p][j], gdat.numbbinspcur)
                gdat.arrypcurquadbdtrbindfine[p][j] = tesstarg.util.rebn_lcur(gdat.arrypcurquadbdtr[p][j], gdat.numbbinspcurfine)
                
                # write to disc for vespa
                path = gdat.pathdata + 'arrypcurprimbdtrbindpla%d%s.csv' % (j, gdat.liststrgchun[y])
                print('Writing to %s' % path)
                temp = np.copy(gdat.arrypcurprimbdtrbind[p][j])
                temp[:, 0] *= gdat.periprio[j]
                np.savetxt(path, temp, delimiter=',')
            
            # determine time mask
            gdat.listindxtimetran = [[] for j in gdat.indxplan]
            gdat.listindxtimeoutt = [[] for j in gdat.indxplan]
            gdat.listindxtimetransect = [[[] for o in gdat.indxsect] for j in gdat.indxplan]
            for j in gdat.indxplan:
                for o in gdat.indxsect:
                    if gdat.booldiagmode:
                        if not np.isfinite(gdat.duramask[j]):
                            raise Exception('')
                    gdat.listindxtimetransect[j][o] = tesstarg.util.retr_indxtimetran(gdat.listarrylcur[o][:, 0], gdat.epocprio[j], \
                                                                                                                        gdat.periprio[j], gdat.duramask[j])
                gdat.listindxtimetran[j] = tesstarg.util.retr_indxtimetran(gdat.listarrylcurbdtr[y][:, 0], gdat.epocprio[j], gdat.periprio[j], gdat.duramask[j])
                gdat.listindxtimeoutt[j] = np.setdiff1d(np.arange(gdat.listarrylcurbdtr[y].shape[0]), gdat.listindxtimetran[j])
            
            # clean times for each planet
            gdat.listindxtimeclen = [[] for j in gdat.indxplan]
            for j in gdat.indxplan:
                listindxtimetemp = []
                for jj in gdat.indxplan:
                    if jj == j:
                        continue
                    listindxtimetemp.append(gdat.listindxtimetran[jj])
                if len(listindxtimetemp) > 0:
                    listindxtimetemp = np.concatenate(listindxtimetemp)
                    listindxtimetemp = np.unique(listindxtimetemp)
                else:
                    listindxtimetemp = np.array([])
                gdat.listindxtimeclen[j] = np.setdiff1d(np.arange(gdat.listarrylcurbdtr[y].shape[0]), listindxtimetemp)

            gdat.time = gdat.listarrylcurbdtr[y][:, 0]
            gdat.numbtime = gdat.time.size
            gdat.indxtime = np.arange(gdat.numbtime)
            
            if not np.isfinite(gdat.arrylcur[y]).all():
                raise Exception('')
            
            if not np.isfinite(gdat.listarrylcurbdtr[y]).all():
                raise Exception('')
            
            figr, axis = plt.subplots(1, 1, figsize=gdat.figrsizeydobskin)
            axis.plot(gdat.listarrylcurbdtr[y][:, 0] - gdat.timetess, gdat.listarrylcurbdtr[y][:, 1], color='grey', marker='.', ls='', ms=1)
            if listlimttimemask is not None:
                axis.plot(gdat.listarrylcurbdtr[y][listindxtimegood, 0] - gdat.timetess, gdat.listarrylcurbdtr[y][listindxtimegood, 1], color='k', marker='.', ls='', ms=1)
            for j in gdat.indxplan:
                colr = gdat.listcolrplan[j]
                print('gdat.listarrylcurbdtr[y]')
                summgene(gdat.listarrylcurbdtr[y])
                print('gdat.listindxtimetran[j]')
                summgene(gdat.listindxtimetran[j])
                axis.plot(gdat.listarrylcurbdtr[y][gdat.listindxtimetran[j], 0] - gdat.timetess, gdat.listarrylcurbdtr[y][gdat.listindxtimetran[j], 1], \
                                                                                                     color=colr, marker='.', ls='', ms=1)
            axis.set_xlabel('Time [BJD - 2457000]')
            for a in range(2):
                axis.set_ylabel('Relative Flux')
            plt.subplots_adjust(bottom=0.2)
            path = gdat.pathimag + 'lcur%s.%s' % (gdat.liststrgchun[y], gdat.strgplotextn)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
   
            if gdat.numbplan > 1:
                figr, axis = plt.subplots(gdat.numbplan, 1, figsize=gdat.figrsizeydobskin)
                for jj, j in enumerate(gdat.indxplan):
                    axis[jj].plot(gdat.listarrylcurbdtr[y][gdat.listindxtimetran[j], 0] - gdat.timetess, gdat.listarrylcurbdtr[y][gdat.listindxtimetran[j], 1], \
                                                                                                          color=gdat.listcolrplan[j], marker='o', ls='', ms=0.2)
                    axis[jj].set_ylabel('Relative Flux')
                axis[-1].set_xlabel('Time [BJD - 2457000]')
                plt.subplots_adjust(bottom=0.2)
                path = gdat.pathimag + 'lcurplanmask%s.%s' % (gdat.liststrgchun[y], gdat.strgplotextn)
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()
            
            if gdat.numbsect > 1:
                for o in gdat.indxsect:
                    figr, axis = plt.subplots(1, 1, figsize=gdat.figrsizeydobskin)
                    
                    axis.plot(gdat.listarrylcur[o][:, 0] - gdat.timetess, gdat.listarrylcur[o][:, 1], color='grey', marker='.', ls='', ms=1, rasterized=True)
                    
                    if listlimttimemask is not None:
                        axis.plot(gdat.listarrylcur[o][listindxtimegood, 0] - gdat.timetess, \
                                                    gdat.listarrylcur[o][listindxtimegood, 1], color='k', marker='.', ls='', ms=1, rasterized=True)
                    
                    ylim = axis.get_ylim()
                    # draw planet names
                    xlim = axis.get_xlim()
                    listtimetext = []
                    for j in gdat.indxplan:
                        colr = gdat.listcolrplan[j]
                        axis.plot(gdat.listarrylcur[o][gdat.listindxtimetransect[j][o], 0] - gdat.timetess, \
                                                                                                gdat.listarrylcur[o][gdat.listindxtimetransect[j][o], 1], \
                                                                                                           color=colr, marker='.', ls='', ms=1, rasterized=True)
                        for n in np.linspace(-30, 30, 61):
                            time = gdat.epocprio[j] + n * gdat.periprio[j] - gdat.timetess
                            if np.where(abs(gdat.listarrylcur[o][:, 0] - gdat.timetess - time) < 0.1)[0].size > 0:
                                
                                # add a vertical offset if overlapping
                                if np.where(abs(np.array(listtimetext) - time) < 0.5)[0].size > 0:
                                    ypostemp = ylim[0] + (ylim[1] - ylim[0]) * 0.95
                                else:
                                    ypostemp = ylim[0] + (ylim[1] - ylim[0]) * 0.9

                                # draw the planet letter
                                axis.text(time, ypostemp, r'\textbf{%s}' % gdat.liststrgplan[j], color=gdat.listcolrplan[j], va='center', ha='center')
                                listtimetext.append(time)
                    axis.set_xlim(xlim)

                    axis.set_xlabel('Time [BJD - 2457000]')
                    for a in range(2):
                        axis.set_ylabel('Relative Flux')
                    
                    plt.subplots_adjust(hspace=0., bottom=0.2)
                    path = gdat.pathimag + 'lcur%s.%s' % (gdat.liststrgchun[o], gdat.strgplotextn)
                    print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()
            
            if gdat.boolbdtr:
                # plot baseline-detrending
                figr, axis = plt.subplots(2, 1, figsize=gdat.figrsizeydob)
                for i in indxsplnregi:
                    ## masked and non-baseline-detrended light curve
                    #indxtimetemp = indxtimeregi[i]
                    #axis[0].plot(gdat.arrylcur[y][indxtimetemp, 0] - gdat.timetess, gdat.arrylcur[y][indxtimetemp, 1], marker='o', ls='', ms=1, color='grey')
                    indxtimetemp = indxtimeregi[i][indxtimeregioutt[i]]
                    axis[0].plot(gdat.arrylcur[y][indxtimetemp, 0] - gdat.timetess, gdat.arrylcur[y][indxtimetemp, 1], marker='o', ls='', ms=1, color='k')
                    ## spline
                    if listobjtspln is not None and listobjtspln[i] is not None:
                        timesplnregifine = np.linspace(gdat.arrylcur[y][indxtimeregi[i], 0][0], gdat.arrylcur[y][indxtimeregi[i], 0][-1], 1000)
                        axis[0].plot(timesplnregifine - gdat.timetess, listobjtspln[i](timesplnregifine), 'b-', lw=3)
                    ## baseline-detrended light curve
                    indxtimetemp = indxtimeregi[i]
                    axis[1].plot(gdat.arrylcur[y][indxtimetemp, 0] - gdat.timetess, lcurbdtrregi[i], marker='o', ms=1, ls='', color='k')
                for a in range(2):
                    axis[a].set_ylabel('Relative Flux')
                axis[0].set_xticklabels([])
                axis[1].set_xlabel('Time [BJD - 2457000]')
                plt.subplots_adjust(hspace=0.)
                path = gdat.pathimag + 'lcurbdtr%s.%s' % (gdat.liststrgchun[y], gdat.strgplotextn)
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()
                
                if listobjtspln is not None and listobjtspln[i] is not None:
                    # produce a table for the spline coefficients
                    fileoutp = open(gdat.pathdata + 'coefbdtr.csv', 'w')
                    fileoutp.write(' & ')
                    for i in indxsplnregi:
                        print('$\beta$:', listobjtspln[i].get_coeffs())
                        print('$t_k$:', listobjtspln[i].get_knots())
                        print
                    fileoutp.write('\\hline\n')
                    fileoutp.close()

    plot_pcur(gdat, 'prio')
   
    # look for single transits using matched filter
    

    gdat.pathallebase = gdat.pathobjt + 'allesfits/'
    
    #gdat.boolalleprev = {}
    #for strgalle in gdat.liststrgalle:
    #    gdat.boolalleprev[strgalle] = {}
    #
    #for strgfile in ['params.csv', 'settings.csv', 'params_star.csv']:
    #    
    #    for strgalle in gdat.liststrgalle:
    #        pathinit = '%sdata/allesfit_templates/%s/%s' % (gdat.pathbase, strgalle, strgfile)
    #        pathfinl = '%sallesfits/allesfit_%s/%s' % (gdat.pathobjt, strgalle, strgfile)

    #        if not os.path.exists(pathfinl):
    #            cmnd = 'cp %s %s' % (pathinit, pathfinl)
    #            print(cmnd)
    #            os.system(cmnd)
    #            if strgfile == 'params.csv':
    #                gdat.boolalleprev[strgalle]['para'] = False
    #            if strgfile == 'settings.csv':
    #                gdat.boolalleprev[strgalle]['sett'] = False
    #            if strgfile == 'params_star.csv':
    #                gdat.boolalleprev[strgalle]['pars'] = False
    #        else:
    #            if strgfile == 'params.csv':
    #                gdat.boolalleprev[strgalle]['para'] = True
    #            if strgfile == 'settings.csv':
    #                gdat.boolalleprev[strgalle]['sett'] = True
    #            if strgfile == 'params_star.csv':
    #                gdat.boolalleprev[strgalle]['pars'] = True

    if gdat.boolallebkgdgaus:
        # background allesfitter run
        print('Setting up the background allesfitter run...')
        
        if not gdat.boolalleprev['bkgd']['para']:
            evol_file(gdat, 'params.csv', gdat.pathallebkgd, lineadde)
        
        ## mask out the transits for the background run
        path = gdat.pathallebkgd + gdat.liststrgchun[y]  + '.csv'
        if not os.path.exists(path):
            indxtimebkgd = np.setdiff1d(gdat.indxtime, np.concatenate(gdat.listindxtimetran))
            gdat.arrylcurbkgd = gdat.listarrylcurbdtr[y][indxtimebkgd, :]
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
        #gdat.objtalle[strgalle] = allesfitter.allesclass(gdat.pathallepcur)
        #gdat.objtalle[strgalle].posterior_params_median['b_sbratio_TESS'] = 0
        #gdat.objtalle[strgalle].settings['host_shape_TESS'] = 'sphere'
        #gdat.objtalle[strgalle].settings['b_shape_TESS'] = 'roche'
        #gdat.objtalle[strgalle].posterior_params_median['host_gdc_TESS'] = 0
        #gdat.objtalle[strgalle].posterior_params_median['host_bfac_TESS'] = 0
        #lcurmodltemp = gdat.objtalle[strgalle].get_posterior_median_model(strgchun, 'flux', xx=gdat.time)
        #axis.plot(gdat.arrypcurquadbdtr[p][j][:, 0], (gdat.lcurmodlevvv - lcurmodltemp) * 1e6, lw=2, label='Spherical star')
        #
        #gdat.objtalle[strgalle] = allesfitter.allesclass(gdat.pathallepcur)
        #gdat.objtalle[strgalle].posterior_params_median['b_sbratio_TESS'] = 0
        #gdat.objtalle[strgalle].settings['host_shape_TESS'] = 'roche'
        #gdat.objtalle[strgalle].settings['b_shape_TESS'] = 'sphere'
        #gdat.objtalle[strgalle].posterior_params_median['host_gdc_TESS'] = 0
        #gdat.objtalle[strgalle].posterior_params_median['host_bfac_TESS'] = 0
        #lcurmodltemp = gdat.objtalle[strgalle].get_posterior_median_model(strgchun, 'flux', xx=gdat.time)
        #axis.plot(gdat.arrypcurquadbdtr[p][j][:, 0], (gdat.lcurmodlevvv - lcurmodltemp) * 1e6, lw=2, label='Spherical planet')
        #axis.legend()
        #axis.set_ylim([-100, 100])
        #axis.set(xlabel='Phase')
        #axis.set(ylabel='Relative flux [ppm]')
        #plt.subplots_adjust(hspace=0.)
        #path = pathimag + 'pcurquadmodldiff.%s' % gdat.strgplotextn
        #plt.savefig(path)
        #plt.close()

