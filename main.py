import allesfitter
import allesfitter.config

import tdpy.util
import tdpy.mcmc
from tdpy.util import prnt_list
from tdpy.util import summgene
import tcat.main

import pickle

import os, fnmatch
import sys, datetime
import numpy as np
import scipy.interpolate

import tesstarg.util

import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amssymb}"]

import astroquery
import astropy

import transitleastsquares

import emcee

'''
Given a target, pexo is an interface to allesfitter that allows 
1) automatic search for, download and process available TESS and Kepler data via MAST
2) impose priors based on custom inputs, ExoFOP or Exoplanet Archive
i) custom: 
3) automatic search for, download and process available TESS and Kepler data via MAST
4) configure and run allesfitter on the target
5) Make characterization plots of the target after the analysis
'''


def retr_lpos_albb(para, gdat):
    
    albb = para[0]
    epsi = para[1]
   
    if albb < 0 or albb > 1 or epsi > 1 or epsi < 0:
        lpos = -np.inf
    else:
        psiimodl = (1 - albb)**.25 * (1. - 5. / 8. * epsi)**.25 / 1.5**.25
        lpos = gdat.likeintp(psiimodl)

    return lpos


def retr_dilu(tmpttarg, tmptcomp, strgwlentype='tess'):
    
    if strgwlentype != 'tess':
        raise Exception('')
    else:
        binswlen = np.linspace(0.6, 1.)
    meanwlen = (binswlen[1:] + binswlen[:-1]) / 2.
    diffwlen = (binswlen[1:] - binswlen[:-1]) / 2.
    
    fluxtarg = retr_specbbod(tmpttarg, meanwlen)
    fluxtarg = np.sum(diffwlen * fluxtarg)
    
    fluxcomp = retr_specbbod(tmptcomp, meanwlen)
    fluxcomp = np.sum(diffwlen * fluxcomp)
    
    dilu = 1. - fluxtarg / (fluxtarg + fluxcomp)
    
    return dilu


def retr_modl_spec(gdat, tmpt, booltess=False, strgtype='intg'):
    
    if booltess:
        thpt = scipy.interpolate.interp1d(gdat.meanwlenband, gdat.thptband)(wlen)
    else:
        thpt = 1.
    
    if strgtype == 'intg':
        spec = retr_specbbod(gdat, tmpt, gdat.meanwlen)
        spec = np.sum(gdat.diffwlen * spec)
    if strgtype == 'diff' or strgtype == 'logt':
        spec = retr_specbbod(gdat, tmpt, gdat.cntrwlen)
        if strgtype == 'logt':
            spec *= gdat.cntrwlen
    
    return spec


def retr_lpos_spec(para, gdat):
    
    if ((para < gdat.limtpara[0, :]) | (para > gdat.limtpara[1, :])).any():
        lpos = -np.inf
    else:
        tmpt = para[0]
        specboloplan = retr_modl_spec(gdat, tmpt, booltess=False, strgtype='intg')
        deptplan = gdat.rratpost**2 * specboloplan / gdat.specstarintg
        llik = -0.5 * np.sum((deptplan - gdat.deptobsd)**2 / gdat.varideptobsd)
        lpos = llik
    
    return lpos


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

    # get Exoplanet Archive data
    path = pathbase + 'data/compositepars_2020.01.23_16.22.13.csv'
    print('Reading %s...' % path)
    NASA_Arc = pd.read_csv(path, skiprows=124)
    if strgtarg is None:
        indx = np.arange(NASA_Arc['fpl_hostname'].size)
    else:
        indx = np.where(NASA_Arc['fpl_hostname'] == strgtarg)[0]
    
    if indx.size == 0:
        print('The planet name, %s, was not found in the Exoplanet Archive composite table.' % strgtarg)
        return None
    elif strgtarg is not None and indx.size > 1:
        raise Exception('Should not be more than 1 match.')
    else:
        dictexarcomp = {}
        dictexarcomp['namestar'] = NASA_Arc['fpl_hostname'][indx].values
        dictexarcomp['nameplan'] = NASA_Arc['fpl_name'][indx].values
        dictexarcomp['peri'] = NASA_Arc['fpl_orbper'][indx].values # [days]
        dictexarcomp['radiplan'] = NASA_Arc['fpl_radj'][indx].values # [R_J]
        dictexarcomp['radistar'] = NASA_Arc['fst_rad'][indx].values * gdat.factrsrj # [R_J]
        dictexarcomp['smax'] = NASA_Arc['fpl_smax'][indx].values # [AU]
        dictexarcomp['massplan'] = NASA_Arc['fpl_bmassj'][indx].values # [M_J]
        dictexarcomp['stdvradiplan'] = np.maximum(NASA_Arc['fpl_radjerr1'][indx].values, NASA_Arc['fpl_radjerr2'][indx].values) # [R_J]
        dictexarcomp['stdvmassplan'] = np.maximum(NASA_Arc['fpl_bmassjerr1'][indx].values, NASA_Arc['fpl_bmassjerr2'][indx].values) # [M_J]
        dictexarcomp['massstar'] = NASA_Arc['fst_mass'][indx].values * 1047.9 # [M_J]
        dictexarcomp['tmptplan'] = NASA_Arc['fpl_eqt'][indx].values # [K]
        dictexarcomp['tmptstar'] = NASA_Arc['fst_teff'][indx].values # [K]
        dictexarcomp['booltran'] = NASA_Arc['fpl_tranflag'][indx].values # [K]
        # temp
        dictexarcomp['kmag'] = NASA_Arc['fst_nirmag'][indx].values # [K]
        dictexarcomp['jmag'] = NASA_Arc['fst_nirmag'][indx].values # [K]
    
    return dictexarcomp


def retr_modl_trap(gdat, para):
    
    dilu = para[0]
    offs = para[1]
    dura = para[2]
    dept = dilu * 0.6 
    numbphas = gdat.arrylcur.shape[0]
    indxphas = np.arange(numbphas)
    lcurmodl = np.ones(numbphas) + offs
    indxtimetran = tesstarg.util.retr_indxtimetran(gdat.arrylcurdetr[:, 0], gdat.epocprio[0], gdat.periprio[0], dura)
    lcurmodl[indxtimetran] -= dept
    
    return lcurmodl, []
        
        
def retr_llik_trap(gdat, para):
    
    lcurmodl, _ = retr_modl_trap(gdat, para)
    llik = -0.5 * np.sum(((gdat.arrylcurdetr[:, 1] - lcurmodl) / gdat.arrylcurdetr[:, 2])**2)
    
    return llik


def retr_llik_sinu(gdat, para):
    
    llik = 0.
    for j in gdat.indxplan:
        modl, _ = retr_modl_sinu(gdat, para, gdat.arrypcurcentdetr[j][gdat.indxphasotpr[j], 0], gdat.indxphasotprotse[j])
        llik += -0.5 * np.sum((modl - gdat.arrypcurcentdetr[j][gdat.indxphasotpr[j], 1])**2 / gdat.arrypcurcentdetr[j][gdat.indxphasotpr[j], 2]**2)
    
    return llik


def retr_modl_sinu(gdat, para, phas, indxphasotse):
    
    numbphas = phas.size

    offs = para[0]
    deptnigh = para[1]
    deptther = para[2]
    deptrefl = para[3] 
    deptelli = para[4] # [ppm]
    if gdat.pcursinetype == 'sinushft':
        phasshft = para[5] # [degrees between 0 and 360]
    else:
        phasshft = 0.

    modl = offs + np.ones(numbphas)
    modl[indxphasotse] += deptnigh * 1e-6
    
    phasaxis = phas + phasshft / 360.
    deptpmod = deptther + deptrefl
    modl += 0.5 * deptpmod * np.cos(2. * np.pi * phasaxis) * 1e-6
    
    modl += 0.5 * deptelli * np.cos(2. * np.pi * phas) * 1e-6
    
    return modl, deptpmod


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


def exec_tlss(gdat):
    
    # setup TLS
    # temp
    #ab, mass, mass_min, mass_max, radius, radius_min, radius_max = transitleastsquares.catalog_info(TIC_ID=int(gdat.ticitarg))

    j = 0
    gdat.epocprio = []
    gdat.periprio = []
    gdat.deptprio = []
    gdat.duraprio = []

    while True:
        
        # mask
        if j == 0:
            timetlssmeta = gdat.arrylcurdetr[:, 0]
            lcurtlssmeta = gdat.arrylcurdetr[:, 1]
        else:
            # mask out the detected transit
            listtimetrantemp = results.transit_times
            indxtimetran = []
            for timetrantemp in listtimetrantemp:
                indxtimetran.append(np.where(abs(timetlssmeta - timetrantemp) < results.duration / 2.)[0])
            indxtimetran = np.concatenate(indxtimetran)
            if indxtimetran.size != np.unique(indxtimetran).size:
                raise Exception('')
            indxtimegood = np.setdiff1d(np.arange(timetlssmeta.size), indxtimetran)
            timetlssmeta = timetlssmeta[indxtimegood]
            lcurtlssmeta = lcurtlssmeta[indxtimegood]
        
        # transit search
        objtmodltlss = transitleastsquares.transitleastsquares(timetlssmeta, lcurtlssmeta)
        # temp
        #results = objtmodltlss.power(u=ab, use_threads=1)
        results = objtmodltlss.power(period_min=0.4)
        
        print('results.period')
        print(results.period)
        print('results.T0')
        print(results.T0)
        print('results.duration')
        print(results.duration)
        print('results.depth')
        print(results.depth)
        print('np.amax(results.power)')
        print(np.amax(results.power))
        print('results.SDE')
        print(results.SDE)
        print('FAP: %g' % results.FAP) 
        
        # plot TLS power spectrum
        figr, axis = plt.subplots(figsize=gdat.figrsize)
        axis.axvline(results.period, alpha=0.4, lw=3)
        axis.set_xlim(np.min(results.periods), np.max(results.periods))
        for n in range(2, 10):
            axis.axvline(n*results.period, alpha=0.4, lw=1, linestyle='dashed')
            axis.axvline(results.period / n, alpha=0.4, lw=1, linestyle='dashed')
        axis.set_ylabel(r'SDE')
        axis.set_xlabel('Period (days)')
        axis.plot(results.periods, results.power, color='black', lw=0.5)
        axis.set_xlim(0, max(results.periods));
        plt.subplots_adjust()
        path = gdat.pathimag + 'sdeetls%d.%s' % (j, gdat.strgplotextn)
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()
        
        # plot light curve + TLS model
        figr, axis = plt.subplots(figsize=gdat.figrsizeydobskin)
        axis.scatter(timetlssmeta, lcurtlssmeta, alpha=0.5, s = 0.8, zorder=0)
        axis.plot(results.model_lightcurve_time, results.model_lightcurve_model, alpha=0.5, color='red', zorder=1)
        axis.set_xlabel('Time (days)')
        axis.set_ylabel('Relative flux');
        plt.subplots_adjust()
        path = gdat.pathimag + 'lcurtls%d.%s' % (j, gdat.strgplotextn)
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()

        # plot phase curve + TLS model
        figr, axis = plt.subplots(figsize=gdat.figrsizeydobskin)
        axis.plot(results.model_folded_phase, results.model_folded_model, color='red')
        axis.scatter(results.folded_phase, results.folded_y, s=0.8, alpha=0.5, zorder=2)
        axis.set_xlabel('Phase')
        axis.set_ylabel('Relative flux');
        plt.subplots_adjust()
        path = gdat.pathimag + 'pcurtls%d.%s' % (j, gdat.strgplotextn)
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()
        
        if gdat.numbplan is None:
            if results.SDE > 7.1 and not (gdat.maxmnumbplantlss is not None and j >= gdat.maxmnumbplantlss):
                gdat.periprio.append(results.period)
                gdat.epocprio.append(results.T0)
                gdat.duraprio.append(results.duration)
                gdat.deptprio.append(results.depth)
            else:
                break
        else:
            if j == gdat.numbplan:
                break
        j += 1

    gdat.epocprio = np.array(gdat.epocprio)
    gdat.periprio = np.array(gdat.periprio)
    gdat.deptprio = np.array(gdat.deptprio)
    gdat.duraprio = np.array(gdat.duraprio)
            

def plot_pcur(gdat, strgpdfn):
    
    if strgpdfn == 'prio':
        arrypcur = gdat.arrypcurdetr
        arrypcurbind = gdat.arrypcurdetrbind
    else:
        arrypcur = gdat.arrypcurdetrgaus
        arrypcurbind = gdat.arrypcurdetrgausbind

    # plot individual phase curves
    for j in gdat.indxplan:
        # phase on the horizontal axis
        figr, axis = plt.subplots(1, 1, figsize=gdat.figrsizeydob)
        axis.plot(arrypcur[j][:, 0], arrypcur[j][:, 1], color='grey', alpha=0.3, marker='o', ls='', ms=1)
        axis.plot(arrypcurbind[j][:, 0], arrypcurbind[j][:, 1], color=gdat.listcolrplan[j], marker='o', ls='', ms=4)
        axis.text(0.9, 0.9, r'\textbf{%s}' % gdat.liststrgplan[j], color=gdat.listcolrplan[j], va='center', ha='center', transform=axis.transAxes)
        axis.set_ylabel('Relative Flux')
        axis.set_xlabel('Phase')
        path = gdat.pathimag + 'pcurphasplan%04d_%s.%s' % (j + 1, strgpdfn, gdat.strgplotextn)
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()
    
        # time on the horizontal axis
        figr, axis = plt.subplots(1, 1, figsize=gdat.figrsize)
        axis.plot(gdat.periprio[j] * arrypcur[j][:, 0] * 24., arrypcur[j][:, 1], color='grey', alpha=0.3, marker='o', ls='', ms=1)
        axis.plot(gdat.periprio[j] * arrypcurbind[j][:, 0] * 24., arrypcurbind[j][:, 1], \
                                                                            color=gdat.listcolrplan[j], marker='o', ls='', ms=4)
        axis.text(0.9, 0.9, \
                                    r'\textbf{%s}' % gdat.liststrgplan[j], color=gdat.listcolrplan[j], va='center', ha='center', transform=axis.transAxes)
        axis.set_ylabel('Relative Flux')
        axis.set_xlabel('Time [hours]')
        axis.set_xlim([-2 * gdat.duraprio[j] * 24., 2 * gdat.duraprio[j] * 24.])
        plt.subplots_adjust(hspace=0., bottom=0.2, left=0.2)
        path = gdat.pathimag + 'pcurtimeplan%04d_%s.%s' % (j + 1, strgpdfn, gdat.strgplotextn)
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()
    
    # plot all phase curves
    if gdat.numbplan > 1:
        figr, axis = plt.subplots(gdat.numbplan, 1, figsize=gdat.figrsizeydob)
        if gdat.numbplan == 1:
            axis = [axis]
        for j in gdat.indxplan:
            axis[j].plot(arrypcur[j][:, 0], arrypcur[j][:, 1], color='grey', alpha=0.3, marker='o', ls='', ms=1)
            axis[j].plot(arrypcurbind[j][:, 0], arrypcurbind[j][:, 1], color=gdat.listcolrplan[j], marker='o', ls='', ms=4)
            axis[j].text(0.47, np.percentile(arrypcur[j][:, 1], 99.9), r'\textbf{%s}' % gdat.liststrgplan[j], \
                                                                                    color=gdat.listcolrplan[j], va='center', ha='center')
            axis[j].minorticks_on()
            axis[j].set_ylabel('Relative Flux')
        axis[0].set_xlabel('Phase')
        plt.subplots_adjust(hspace=0., bottom=0.3)
        path = gdat.pathimag + 'pcur_%s.%s' % (strgpdfn, gdat.strgplotextn)
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()
    

def main( \
         ticitarg=None, \
         strgmast=None, \
         toiitarg=None, \
         
         # a string for the label of the target
         labltarg=None, \
         
         # a string for the folder name and file name extensions
         strgtarg=None, \
         
         boolphascurv=True, \
         
         # type of light curve to be used for analysis
         datatype=None, \
                    
         # Boolean flag to use SAP instead of PDC by default, when SPOC data is being used.
         boolsapp=False, \
         
         boolexar=None, \
        
         maxmnumbstartcat=1, \
         
         pcurtype='alle', \

         # star properties
         jmag=None, \

         dilucorr=None, \

         epocprio=None, \
         periprio=None, \
         radiplanprio=None, \
         inclprio=None, \
         duraprio=None, \
         smaxprio=None, \
         timedetr=None, \
    
         # type of inference, alle for allesfitter, trap for trapezoidal fit
         infetype='alle', \

         boolcontrati=False, \
        
         makeprioplot=True, \
         
         # list of offsets for the planet annotations in the TSM/ESM plot
         offstextatmo=None, \

         ## baseline
         detrtype='spln', \
         weigsplndetr=1., \
         durakerndetrmedi=1., \
         
         # type of priors used for allesfitter
         priotype=None, \

         boolallebkgdgaus=False, \
    
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

    print('TESS TOI/allesfitter pipeline initialized at %s...' % gdat.strgtimestmp)
    
    # plotting
    gdat.strgplotextn = 'pdf'
    gdat.figrsize = [4., 3.]
    gdat.figrsizeydob = [8., 4.]
    gdat.figrsizeydobskin = [8., 2.5]
    boolpost = False
    if boolpost:
        gdat.figrsize /= 1.5
    
    if gdat.offstextatmo is None:
        gdat.offstextatmo = [[0.3, -0.5], [0.3, -0.5], [0.3, -0.5], [0.3, 0.5]]
    
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

    dictexarcomp = retr_exarcomp(gdat, strgtarg=gdat.strgmast)
    gdat.boolexar = dictexarcomp is not None
    
    if gdat.boolexar:
        print('Found the target on the Exoplanet Archive.')
        # stellar properties
        
        gdat.priotype = 'exar'
        gdat.periprio = dictexarcomp['peri']
        gdat.radiplanprio = dictexarcomp['radiplan']
        gdat.radistarprio = dictexarcomp['radistar']
        gdat.smaxprio = dictexarcomp['smax']
        gdat.massplanprio = dictexarcomp['massplan']
        gdat.massstarprio = dictexarcomp['massstar']
    else:
        print('Could not find the target on the Exoplanet Archive.')

    # read the Exoplanet Archive planets
    path = gdat.pathbase + 'data/planets_2020.01.23_16.21.55.csv'
    print('Reading %s...' % path)
    NASA_Arc = pd.read_csv(path, skiprows=359)
    indx = np.where(NASA_Arc['pl_hostname'].values == gdat.strgmast)[0]
    if indx.size == 0:
        print('The planet name was not found in the Exoplanet Archive planet table')
        gdat.boolexar = False
    elif indx.size > 1:
        raise Exception('Should not be more than 1 match.')
    else:
        #indx = indx[0]
        gdat.priotype = 'exar'
        gdat.inclprio = NASA_Arc['pl_orbincl'][indx].values # [deg]
        gdat.epocprio = NASA_Arc['pl_tranmid'][indx].values # [days]
        gdat.duraprio = NASA_Arc['pl_trandur'][indx].values # [days]
    
    if gdat.toiitarg is not None:
        
        gdat.priotype = 'exof'
        print('A TOI number is provided. Retreiving the TCE attributes from ExoFOP-TESS...')
        # read ExoFOP-TESS
        path = gdat.pathbase + 'data/exofop_toilists.csv'
        print('Reading from %s...' % path)
        NASA_Arc = pd.read_csv(path, skiprows=0)
        indx = []
        gdat.strgtoiibase = str(gdat.toiitarg)
        for k, strg in enumerate(NASA_Arc['TOI']):
            if str(strg).split('.')[0] == gdat.strgtoiibase:
                indx.append(k)
        indx = np.array(indx)
        if indx.size == 0:
            print('Did not find the TOI in the ExoFOP-TESS TOI list.')
            raise Exception('')
        
        if gdat.ticitarg is not None:
            raise Exception('')
        else:
            gdat.ticitarg = NASA_Arc['TIC ID'].values[indx[0]]
        gdat.strgmast  = 'TIC %d' % gdat.ticitarg
        gdat.epocprio = NASA_Arc['Epoch (BJD)'].values[indx]
        gdat.periprio = NASA_Arc['Period (days)'].values[indx]
        gdat.deptprio = NASA_Arc['Depth (ppm)'].values[indx] * 1e-6
        gdat.radiplanprio = NASA_Arc['Planet Radius (R_Earth)'].values[indx] / gdat.factrjre # [R_J]
        gdat.duraprio = NASA_Arc['Duration (hours)'].values[indx] / 24. # [days]
        gdat.radistar = NASA_Arc['Stellar Radius (R_Sun)'].values[indx][0] * gdat.factrsrj # [R_J]
        gdat.tmptplanprio = NASA_Arc['Planet Equil Temp (K)'].values[indx] # [K]
        gdat.tmptstar = NASA_Arc['Stellar Eff Temp (K)'].values[indx[0]] # [K]
        if not np.isfinite(gdat.tmptstar):
            raise Exception('ExoFOP stellar temperature is not finite.')
        gdat.cosiprio = np.zeros_like(gdat.epocprio)
        
        if gdat.strgtarg is None:
            gdat.strgtarg = 'TOI' + gdat.strgtoiibase
        if gdat.labltarg is None:
            gdat.labltarg = 'TOI ' + gdat.strgtoiibase
        print('gdat.labltarg')
        print(gdat.labltarg)
        gdat.loggstar = NASA_Arc['Stellar log(g) (cm/s^2)'].values[indx] # [K]
        if np.unique(gdat.loggstar).size != 1:
            raise Exception('')
        gdat.loggstar = gdat.loggstar[0]

        gdat.massstar = 1. / 2.7e4 * 10**(gdat.loggstar) * (gdat.radistar / gdat.factrsrj)**2 * gdat.factmsmj

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

    catalogData = astroquery.mast.Catalogs.query_object(gdat.strgmast, catalog='TIC', radius='5m')
    if catalogData[0]['dstArcSec'] < 0.002:
        print('Found the target on MAST!')
        rasc = catalogData[0]['ra']
        decl = catalogData[0]['dec']
        if gdat.tmptstar is None:
            print('Setting the stellar temperature from the TIC.')
            gdat.tmptstar = catalogData[0]['Teff']
        gdat.jmag = catalogData[0]['Jmag']
    print('rasc')
    print(rasc)
    print('decl')
    print(decl)
    strgtici = '%s' % catalogData[0]['ID']
    if strgtici != str(gdat.ticitarg):
        raise Exception('')
    
    gdat.pathobjt = gdat.pathbase + '%s/' % gdat.strgtarg
    gdat.pathdata = gdat.pathobjt + 'data/'
    gdat.pathimag = gdat.pathobjt + 'imag/'
    os.system('mkdir -p %s' % gdat.pathdata)
    os.system('mkdir -p %s' % gdat.pathimag)

    print('gdat.datatype')
    print(gdat.datatype)
    gdat.datatype, gdat.arrylcur, gdat.arrylcursapp, gdat.arrylcurpdcc, gdat.listarrylcur, gdat.listarrylcursapp, \
                                                    gdat.listarrylcurpdcc, gdat.listisec, gdat.listicam, gdat.listiccd = \
                                                     tesstarg.util.retr_data(gdat.datatype, gdat.strgmast, gdat.pathobjt, gdat.boolsapp, \
                                          labltarg=gdat.labltarg, strgtarg=gdat.strgtarg, ticitarg=gdat.ticitarg, maxmnumbstartcat=gdat.maxmnumbstartcat)
    
    gdat.numbsect = len(gdat.listarrylcur)
    gdat.indxsect = np.arange(gdat.numbsect)
    gdat.booldetr = gdat.datatype != 'pdcc'
    if gdat.duraprio is not None:
        epocmask = gdat.epocprio
        perimask = gdat.periprio
        duramask = 2. * gdat.duraprio
    else:
        epocmask = None
        perimask = None
        duramask = None
    print('gdat.booldetr')
    print(gdat.booldetr)
    if gdat.booldetr:
        
        lcurdetrregi, indxtimeregi, indxtimeregioutt, listobjtspln = \
                                                     tesstarg.util.detr_lcur(gdat.arrylcur[:, 0], gdat.arrylcur[:, 1], weigsplndetr=gdat.weigsplndetr, \
                                                         epocmask=epocmask, perimask=perimask, duramask=duramask, durakerndetrmedi=gdat.durakerndetrmedi)
        gdat.arrylcurdetr = np.copy(gdat.arrylcur)
        gdat.arrylcurdetr[:, 1] = np.concatenate(lcurdetrregi)
        numbsplnregi = len(lcurdetrregi)
        indxsplnregi = np.arange(numbsplnregi)

    else:
        gdat.arrylcurdetr = gdat.arrylcur
    
    if not np.isfinite(gdat.arrylcurdetr).all():
        raise Exception('')

    if gdat.booldiagmode:
        for a in range(gdat.arrylcur[:, 0].size):
            if a != gdat.arrylcur[:, 0].size - 1 and gdat.arrylcur[a, 0] >= gdat.arrylcur[a+1, 0]:
                raise Exception('')
    
    print('gdat.priotype')
    print(gdat.priotype)

    if gdat.priotype == 'tlss':
        exec_tlss(gdat)
        gdat.cosiprio = np.zeros_like(gdat.periprio)
        gdat.rratprio = np.sqrt(gdat.deptprio)
        gdat.rsmaprio = np.sin(np.pi * gdat.duraprio / gdat.periprio)

    gdat.rratprio = np.sqrt(gdat.deptprio)

    if gdat.priotype == 'exof' or gdat.priotype == 'tlss' or gdat.priotype == 'inpt':
        gdat.radiplanprio = gdat.rratprio * gdat.radistar
   
    if gdat.priotype == 'exar' or gdat.priotype == 'exof' or gdat.priotype == 'tlss':
        gdat.rsmaprio = np.sqrt(np.sin(np.pi * gdat.duraprio / gdat.periprio)**2 + gdat.cosiprio**2)
        #gdat.duraprio = gdat.periprio / np.pi * np.arcsin(np.sqrt(gdat.rsmaprio**2 - gdat.cosiprio**2))
    
    print('gdat.radistar')
    print(gdat.radistar)
    print('gdat.tmptstar')
    print(gdat.tmptstar)
    print('gdat.massstar')
    print(gdat.massstar)
    print('gdat.epocprio')
    print(gdat.epocprio)
    print('gdat.periprio')
    print(gdat.periprio)
    print('gdat.radiplanprio')
    print(gdat.radiplanprio)
    print('gdat.rratprio')
    print(gdat.rratprio)
    print('gdat.rsmaprio')
    print(gdat.rsmaprio)
    print('gdat.cosiprio')
    print(gdat.cosiprio)
    print('gdat.duraprio')
    print(gdat.duraprio)
    
    gdat.massplanprio = np.zeros_like(gdat.epocprio)
    fracmass = gdat.massplanprio / gdat.massstar
    print('fracmass')
    print(fracmass)
    
    gdat.ecceprio = np.zeros_like(gdat.epocprio)
    print('gdat.ecceprio')
    print(gdat.ecceprio)
    

    if gdat.booldiagmode:
        if not np.isfinite(gdat.duraprio).all():
            raise Exception('')
    

    gdat.boolplotspec = False
        
    # settings
    gdat.numbplan = gdat.epocprio.size
    gdat.indxplan = np.arange(gdat.numbplan)
    if gdat.liststrgplan is None:
        gdat.liststrgplan = ['b', 'c', 'd', 'e', 'f', 'g'][:gdat.numbplan]
    
    gdat.liststrgplanfull = np.empty(gdat.numbplan, dtype='object')
    print('gdat.liststrgplan')
    print(gdat.liststrgplan)
    print('gdat.numbplan')
    print(gdat.numbplan)
    for j in gdat.indxplan:
        gdat.liststrgplanfull[j] = gdat.labltarg + ' ' + gdat.liststrgplan[j]

    #gdat.listcolrplan = ['r', 'g', 'b', 'm', 'y', 'c']
    gdat.listcolrplan = ['red', 'green', 'blue', 'magenta', 'yellow', 'cyan']
    gdat.listlablplan = ['04', '03', '01', '02']
    
    gdat.timetess = 2457000.
    
    gdat.indxplan = np.arange(gdat.numbplan)
    
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
    # determine time mask
    gdat.listindxtimetran = [[] for j in gdat.indxplan]
    gdat.listindxtimetransect = [[[] for o in gdat.indxsect] for j in gdat.indxplan]
    for j in gdat.indxplan:
        for o in gdat.indxsect:
            if gdat.booldiagmode:
                if not np.isfinite(gdat.duramask[j]):
                    raise Exception('')
            gdat.listindxtimetransect[j][o] = tesstarg.util.retr_indxtimetran(gdat.listarrylcur[o][:, 0], gdat.epocprio[j], \
                                                                                                                gdat.periprio[j], gdat.duramask[j])
        gdat.listindxtimetran[j] = tesstarg.util.retr_indxtimetran(gdat.arrylcur[:, 0], gdat.epocprio[j], gdat.periprio[j], gdat.duramask[j])
    gdat.time = gdat.arrylcur[:, 0]
    gdat.numbtime = gdat.time.size
    gdat.indxtime = np.arange(gdat.numbtime)
    
    if listlimttimemask is not None:
        # mask the data
        print('Masking the data...')
        gdat.arrylcurumsk = np.copy(gdat.arrylcur)
        numbmask = listlimttimemask.shape[0]
        listindxtimemask = []
        for k in range(numbmask):
            indxtimemask = np.where((gdat.arrylcur[:, 0] < listlimttimemask[k, 1]) & (gdat.arrylcur[:, 0] > listlimttimemask[k, 0]))[0]
            listindxtimemask.append(indxtimemask)
        listindxtimemask = np.concatenate(listindxtimemask)
        listindxtimegood = np.setdiff1d(gdat.indxtime, listindxtimemask)
        gdat.arrylcur = gdat.arrylcur[listindxtimegood, :]
    
    # correct for dilution
    print('Correcting for dilution!')
    if gdat.dilucorr is not None or gdat.boolcontrati:
        gdat.arrylcurdilu = np.copy(gdat.arrylcurdetr)
    if gdat.dilucorr is not None:
        gdat.arrylcurdilu[:, 1] = 1. - gdat.dilucorr * (1. - gdat.arrylcurdetr[:, 1])
    if gdat.boolcontrati:
        gdat.arrylcurdilu[:, 1] = 1. - gdat.contrati * gdat.contrati * (1. - gdat.arrylcurdetr[:, 1])
    if gdat.dilucorr is not None or gdat.boolcontrati:
        gdat.arrylcurdetr = np.copy(gdat.arrylcurdilu) 
        figr, axis = plt.subplots(1, 1, figsize=gdat.figrsizeydob)
        axis.plot(gdat.arrylcurdilu[:, 0] - gdat.timetess, gdat.arrylcurdilu[:, 1], color='grey', marker='.', ls='', ms=1)
        for j in gdat.indxplan:
            colr = gdat.listcolrplan[j]
            axis.plot(gdat.arrylcurdilu[gdat.listindxtimetran[j], 0] - gdat.timetess, gdat.arrylcurdilu[gdat.listindxtimetran[j], 1], \
                                                                                                    color=colr, marker='.', ls='', ms=1)
        axis.set_xlabel('Time [BJD - 2457000]')
        axis.minorticks_on()
        axis.set_ylabel('Relative Flux')
        plt.subplots_adjust(hspace=0.)
        path = gdat.pathimag + 'lcurdilu.%s' % (gdat.strgplotextn)
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()
        
        
    ## phase-fold and save the detrended light curve
    numbbins = 2000
    gdat.arrylcurdetrbind = tesstarg.util.rebn_lcur(gdat.arrylcurdetr, numbbins)
    
    path = gdat.pathobjt + 'arrylcurdetrbind.csv'
    print('Writing to %s' % path)
    np.savetxt(path, gdat.arrylcurdetrbind, delimiter=',', header='time,flux,flux_err')
    
    gdat.arrypcurdetr = [[] for j in gdat.indxplan]
    gdat.arrypcurdetrbind = [[] for j in gdat.indxplan]
    for j in gdat.indxplan:
        gdat.arrypcurdetr[j] = tesstarg.util.fold_lcur(gdat.arrylcurdetr, gdat.epocprio[j], gdat.periprio[j])
        gdat.arrypcurdetrbind[j] = tesstarg.util.rebn_lcur(gdat.arrypcurdetr[j], numbbins)
        
        # write to disc for vespa
        path = gdat.pathobjt + 'arrypcurdetrbind%04d.csv' % (j + 1)
        print('Writing to %s' % path)
        temp = np.copy(gdat.arrypcurdetrbind[j])
        temp[:, 0] *= gdat.periprio[j]
        np.savetxt(path, temp, delimiter=',')
        
    
    liststrgzoom = ['allr']
    if np.where((gdat.radiplanprio * gdat.factrjre > 1.5) & (gdat.radiplanprio * gdat.factrjre < 4.))[0].size > 0:
        liststrgzoom.append('rb14')
        
    print('gdat.datatype')
    print(gdat.datatype)
    if gdat.boolplotspec:
        ## TESS throughput 
        gdat.data = np.loadtxt(pathdata + 'band.csv', delimiter=',', skiprows=9)
        gdat.meanwlenband = gdat.data[:, 0] * 1e-3
        gdat.thptband = gdat.data[:, 1]
    
    if gdat.datatype == 'spoc':
        # plot PDCSAP and SAP light curves
        figr, axis = plt.subplots(2, 1, figsize=gdat.figrsizeydob)
        axis[0].plot(gdat.arrylcursapp[:, 0] - gdat.timetess, gdat.arrylcursapp[:, 1], color='grey', marker='.', ls='', ms=1)
        if listlimttimemask is not None:
            axis[0].plot(gdat.arrylcursapp[listindxtimegood, 0] - gdat.timetess, gdat.arrylcursapp[listindxtimegood, 1], color='k', marker='.', ls='', ms=1)
        axis[1].plot(gdat.arrylcurpdcc[:, 0] - gdat.timetess, gdat.arrylcurpdcc[:, 1], color='grey', marker='.', ls='', ms=1)
        if listlimttimemask is not None:
            axis[1].plot(gdat.arrylcurpdcc[listindxtimegood, 0] - gdat.timetess, gdat.arrylcurpdcc[listindxtimegood, 1], color='k', marker='.', ls='', ms=1)
        #axis[0].text(.97, .97, 'SAP', transform=axis[0].transAxes, size=20, color='r', ha='right', va='top')
        #axis[1].text(.97, .97, 'PDC', transform=axis[1].transAxes, size=20, color='r', ha='right', va='top')
        axis[1].set_xlabel('Time [BJD - 2457000]')
        for a in range(2):
            axis[a].minorticks_on()
            axis[a].set_ylabel('Relative Flux')
        for j in gdat.indxplan:
            colr = gdat.listcolrplan[j]
            axis[1].plot(gdat.arrylcurpdcc[gdat.listindxtimetran[j], 0] - gdat.timetess, gdat.arrylcurpdcc[gdat.listindxtimetran[j], 1], \
                                                                                                                 color=colr, marker='.', ls='', ms=1)
        plt.subplots_adjust(hspace=0.)
        path = gdat.pathimag + 'lcurspoc.%s' % gdat.strgplotextn
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()
    
    if not np.isfinite(gdat.arrylcur).all():
        raise Exception('')
    
    if gdat.booldiagmode:
        for a in range(gdat.arrylcur[:, 0].size):
            if a != gdat.arrylcur[:, 0].size - 1 and gdat.arrylcur[a, 0] >= gdat.arrylcur[a+1, 0]:
                raise Exception('')
    

    figr, axis = plt.subplots(1, 1, figsize=gdat.figrsizeydobskin)
    axis.plot(gdat.arrylcur[:, 0] - gdat.timetess, gdat.arrylcur[:, 1], color='grey', marker='.', ls='', ms=1)
    if listlimttimemask is not None:
        axis.plot(gdat.arrylcur[listindxtimegood, 0] - gdat.timetess, gdat.arrylcur[listindxtimegood, 1], color='k', marker='.', ls='', ms=1)
    for j in gdat.indxplan:
        colr = gdat.listcolrplan[j]
        axis.plot(gdat.arrylcur[gdat.listindxtimetran[j], 0] - gdat.timetess, gdat.arrylcur[gdat.listindxtimetran[j], 1], \
                                                                                                color=colr, marker='.', ls='', ms=1)
    axis.set_xlabel('Time [BJD - 2457000]')
    for a in range(2):
        axis.minorticks_on()
        axis.set_ylabel('Relative Flux')
    plt.subplots_adjust(bottom=0.2)
    path = gdat.pathimag + 'lcur.%s' % (gdat.strgplotextn)
    print('Writing to %s...' % path)
    plt.savefig(path)
    plt.close()
    
    if gdat.numbsect > 1:
        for o in gdat.indxsect:
            figr, axis = plt.subplots(1, 1, figsize=gdat.figrsizeydobskin)
            
            axis.plot(gdat.listarrylcur[o][:, 0] - gdat.timetess, gdat.listarrylcur[o][:, 1], color='grey', marker='.', ls='', ms=1)
            
            if listlimttimemask is not None:
                axis.plot(gdat.listarrylcur[o][listindxtimegood, 0] - gdat.timetess, \
                                            gdat.listarrylcur[o][listindxtimegood, 1], color='k', marker='.', ls='', ms=1)
            
            ylim = axis.get_ylim()
            # draw planet names
            xlim = axis.get_xlim()
            listtimetext = []
            for j in gdat.indxplan:
                colr = gdat.listcolrplan[j]
                axis.plot(gdat.listarrylcur[o][gdat.listindxtimetransect[j][o], 0] - gdat.timetess, \
                                                                                        gdat.listarrylcur[o][gdat.listindxtimetransect[j][o], 1], \
                                                                                                        color=colr, marker='.', ls='', ms=1)
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
                axis.minorticks_on()
                axis.set_ylabel('Relative Flux')
            
            plt.subplots_adjust(hspace=0., bottom=0.2)
            path = gdat.pathimag + 'lcursc%02d.%s' % (gdat.listisec[o], gdat.strgplotextn)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
    
    if gdat.booldetr:
        # plot detrending
        figr, axis = plt.subplots(2, 1, figsize=gdat.figrsizeydob)
        for i in indxsplnregi:
            # plot the masked and detrended light curves
            indxtimetemp = indxtimeregi[i]
            axis[0].plot(gdat.arrylcur[indxtimetemp, 0] - gdat.timetess, gdat.arrylcur[indxtimetemp, 1], marker='o', ls='', ms=1, color='grey')
            indxtimetemp = indxtimeregi[i][indxtimeregioutt[i]]
            axis[0].plot(gdat.arrylcur[indxtimetemp, 0] - gdat.timetess, gdat.arrylcur[indxtimetemp, 1], marker='o', ls='', ms=1, color='k')
            
            if listobjtspln is not None and listobjtspln[i] != []:
                timesplnregifine = np.linspace(gdat.arrylcur[indxtimeregi[i], 0][0], gdat.arrylcur[indxtimeregi[i], 0][-1], 1000)
                axis[0].plot(timesplnregifine - gdat.timetess, listobjtspln[i](timesplnregifine), 'b-', lw=3)
            
            indxtimetemp = indxtimeregi[i]
            axis[1].plot(gdat.arrylcur[indxtimetemp, 0] - gdat.timetess, lcurdetrregi[i], marker='o', ms=1, ls='', color='grey')
        for a in range(2):
            axis[a].set_ylabel('Relative Flux')
        axis[0].set_xticklabels([])
        axis[1].set_xlabel('Time [BJD - 2457000]')
        path = gdat.pathimag + 'lcurdetr.%s' % gdat.strgplotextn
        plt.subplots_adjust(hspace=0.)
        plt.savefig(path)
        plt.close()
        
        if listobjtspln is not None:
            # produce a table for the spline coefficients
            fileoutp = open(gdat.pathdata + 'coef.csv', 'w')
            fileoutp.write(' & ')
            for j in indxsplnregi:
                print('$\beta$:', listobjtspln[i].get_coeffs())
                print('$t_k$:', listobjtspln[i].get_knots())
                print
            fileoutp.write('\\hline\n')
            fileoutp.close()

    plot_pcur(gdat, 'prio')
    
    if gdat.infetype == 'alle':
        gdat.pathalle = gdat.pathobjt + 'allesfits/'
        gdat.pathalleorbt = gdat.pathalle + 'allesfit_orbt/'
        
        if gdat.boolallebkgdgaus:
            gdat.pathallebkgd = gdat.pathalle + 'allesfit_bkgd/'

        cmnd = 'mkdir -p %s' % gdat.pathalleorbt
        os.system(cmnd)
        if gdat.boolallebkgdgaus:
            cmnd = 'mkdir -p %s' % gdat.pathallebkgd
            os.system(cmnd)
    
        gdat.liststrgalle = []
        if gdat.boolallebkgdgaus:
            gdat.liststrgalle += ['bkgd']
        gdat.liststrgalle += ['orbt']
        if gdat.boolphascurv and gdat.pcurtype == 'alle':
            gdat.liststrgalle += ['pcur']
            gdat.pathallepcur = gdat.pathalle + 'allesfit_pcur/'
            cmnd = 'mkdir -p %s' % gdat.pathallepcur
            os.system(cmnd)
        print('gdat.liststrgalle')
        print(gdat.liststrgalle)

        gdat.boolalleprev = {}
        for strgalle in gdat.liststrgalle:
            gdat.boolalleprev[strgalle] = {}
        for strgfile in ['params.csv', 'settings.csv', 'params_star.csv']:
            
            for strgalle in gdat.liststrgalle:
                pathinit = '%sdata/allesfit_templates/%s/%s' % (gdat.pathbase, strgalle, strgfile)
                pathfinl = '%sallesfits/allesfit_%s/%s' % (gdat.pathobjt, strgalle, strgfile)

                if not os.path.exists(pathfinl):
                    cmnd = 'cp %s %s' % (pathinit, pathfinl)
                    print(cmnd)
                    os.system(cmnd)
                    if strgfile == 'params.csv':
                        gdat.boolalleprev[strgalle]['para'] = False
                    if strgfile == 'settings.csv':
                        gdat.boolalleprev[strgalle]['sett'] = False
                    if strgfile == 'params_star.csv':
                        gdat.boolalleprev[strgalle]['pars'] = False
                else:
                    if strgfile == 'params.csv':
                        gdat.boolalleprev[strgalle]['para'] = True
                    if strgfile == 'settings.csv':
                        gdat.boolalleprev[strgalle]['sett'] = True
                    if strgfile == 'params_star.csv':
                        gdat.boolalleprev[strgalle]['pars'] = True
        print('gdat.boolalleprev')
        print(gdat.boolalleprev)

        if gdat.boolallebkgdgaus:
            # background allesfitter run
            print('Setting up the background allesfitter run...')
            
            if not gdat.boolalleprev['bkgd']['para']:
                evol_file(gdat, 'params.csv', gdat.pathallebkgd, lineadde)
            
            ## mask out the transits for the background run
            path = gdat.pathallebkgd + 'TESS.csv'
            if not os.path.exists(path):
                indxtimebkgd = np.setdiff1d(gdat.indxtime, np.concatenate(gdat.listindxtimetran))
                gdat.arrylcurbkgd = gdat.arrylcurdetr[indxtimebkgd, :]
                print('Writing to %s...' % path)
                np.savetxt(path, gdat.arrylcurbkgd, delimiter=',', header='time,flux,flux_err')
            else:
                print('OoT light curve available for the background allesfitter run at %s.' % path)
            
            ## initial plot
            path = gdat.pathallebkgd + 'results/initial_guess_b.pdf' 
            if not os.path.exists(path):
                allesfitter.show_initial_guess(gdat.pathallebkgd)
            
            ## do the run
            path = gdat.pathallebkgd + 'results/mcmc_save.h5'
            if not os.path.exists(path):
                allesfitter.mcmc_fit(gdat.pathallebkgd)
            else:
                print('%s exists... Skipping the background run.' % path)

            ## make the final plots
            path = gdat.pathallebkgd + 'results/mcmc_corner.pdf'
            if not os.path.exists(path):
                allesfitter.mcmc_output(gdat.pathallebkgd)

            # read the background run output
            gdat.objtallebkgd = allesfitter.allesclass(gdat.pathallebkgd)
            allesfitter.config.init(gdat.pathallebkgd)
            
            numbsamp = gdat.objtallebkgd.posterior_params[list(gdat.objtallebkgd.posterior_params.keys())[0]].size
            liststrg = list(gdat.objtallebkgd.posterior_params.keys())
            for k, strg in enumerate(liststrg):
               post = gdat.objtallebkgd.posterior_params[strg]
               linesplt = '%s' % gdat.objtallebkgd.posterior_params_at_maximum_likelihood[strg][0]
        
        # setup the orbit run
        print('Setting up the orbit allesfitter run...')

        path = gdat.pathalleorbt + 'TESS.csv'
        print('Writing to %s...' % path)
        np.savetxt(path, gdat.arrylcurdetr, delimiter=',', header='time,flux,flux_err')
        
        # update the transit duration for fastfit
        lineadde = [ \
                    [[], 'fast_fit_width,%.3g\n' % np.amax(gdat.duramask)], \
                   ]
        linetemp = 'companions_phot,'
        for j in gdat.indxplan:
            if j != 0:
                linetemp += ' '
            linetemp += '%s' % gdat.liststrgplan[j]
        linetemp += '\n'
        lineadde.extend([ \
                        ['companions_phot*', linetemp] \
                        ])
        if not gdat.boolalleprev['orbt']['sett']:
            evol_file(gdat, 'settings.csv', gdat.pathalleorbt, lineadde)
        
        lineadde = []
        for j in gdat.indxplan:
            strgrrat = '%s_rr' % gdat.liststrgplan[j]
            strgrsma = '%s_rsuma' % gdat.liststrgplan[j]
            strgcosi = '%s_cosi' % gdat.liststrgplan[j]
            strgepoc = '%s_epoch' % gdat.liststrgplan[j]
            strgperi = '%s_period' % gdat.liststrgplan[j]
            lineadde.extend([ \
                        [strgrrat + '*', '%s,%f,1,uniform %f %f,$R_{%s} / R_\star$,\n' % \
                                        (strgrrat, gdat.rratprio[j], 0, 2 * gdat.rratprio[j], gdat.liststrgplan[j])], \
                        [strgrsma + '*', '%s,%f,1,uniform %f %f,$(R_\star + R_{%s}) / a_{%s}$,\n' % \
                                        (strgrsma, gdat.rsmaprio[j], 0, 2 * gdat.rsmaprio[j], gdat.liststrgplan[j], gdat.liststrgplan[j])], \
                        [strgcosi + '*', '%s,%f,1,uniform %f %f,$\cos{i_{%s}}$,\n' % \
                                        (strgcosi, gdat.cosiprio[j], 0, max(0.1, gdat.cosiprio[j] * 2), gdat.liststrgplan[j])], \
                        [strgepoc + '*', '%s,%f,1,uniform %f %f,$T_{0;%s}$,$\mathrm{BJD}$\n' % \
                                        (strgepoc, gdat.epocprio[j], gdat.epocprio[j] - 0.5, gdat.epocprio[j] + 0.5, gdat.liststrgplan[j])], \
                        [strgperi + '*', '%s,%f,1,uniform %f %f,$P_{%s}$,$\mathrm{d}$\n' % \
                                        (strgperi, gdat.periprio[j], gdat.periprio[j] - 0.01, gdat.periprio[j] + 0.01, gdat.liststrgplan[j])], \
                       ])
        print('lineadde')
        print(lineadde)
        if not gdat.boolalleprev['orbt']['para']:
            evol_file(gdat, 'params.csv', gdat.pathalleorbt, lineadde)
            
        ## initial plot
        path = gdat.pathalleorbt + 'results/initial_guess_b.pdf'
        if not os.path.exists(path):
            allesfitter.show_initial_guess(gdat.pathalleorbt)
        
        ## do the run
        path = gdat.pathalleorbt + 'results/mcmc_save.h5'
        if not os.path.exists(path):
            allesfitter.mcmc_fit(gdat.pathalleorbt)
        else:
            print('%s exists... Skipping the orbit run.' % path)

        ## make the final plots
        path = gdat.pathalleorbt + 'results/mcmc_corner.pdf'
        # temp
        #if not os.path.exists(path):
        #    allesfitter.mcmc_output(gdat.pathalleorbt)
        
        # read the allesfitter posterior
        strginst = 'TESS'
        print('Reading from %s...' % gdat.pathalleorbt)
        alles = allesfitter.allesclass(gdat.pathalleorbt)
        allesfitter.config.init(gdat.pathalleorbt)
        
        numbsamp = alles.posterior_params[list(alles.posterior_params.keys())[0]].size

        gdat.liststrgfeat = ['epoc', 'peri', 'rrat', 'rsma', 'cosi', 'ecce', 'smax']
        gdat.dictlist = {}
        gdat.dictpost = {}
        gdat.dicterrr = {}
        for strgfeat in gdat.liststrgfeat:
            gdat.dictlist[strgfeat] = np.empty((numbsamp, gdat.numbplan))

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
    
                print('alles')
                print(alles)
                print('alles.posterior_params.keys()')
                print(alles.posterior_params.keys())
                if strg in alles.posterior_params.keys():
                    gdat.dictlist[strgfeat][:, j] = alles.posterior_params[strg]
                else:
                    gdat.dictlist[strgfeat][:, j] = np.zeros(numbsamp) + allesfitter.config.BASEMENT.params[strg]

        # inclination [degree]
        gdat.dictlist['incl'] = np.arccos(gdat.dictlist['cosi']) * 180. / np.pi
        
        # radius of the planets
        gdat.dictlist['radiplan'] = gdat.radistar * gdat.dictlist['rrat']
        
        # semi-major axis
        gdat.dictlist['smax'] = (gdat.dictlist['radiplan'] + gdat.radistar) / gdat.dictlist['rsma']
        
        # planet equilibrium temperature
        gdat.dictlist['tmptplan'] = gdat.tmptstar * np.sqrt(gdat.radistar / 2. / gdat.dictlist['smax'])
        
        # predicted planet mass
        gdat.dictlist['massplan'] = tesstarg.util.retr_massfromradi(gdat.dictlist['radiplan'])
            
        # RV semi-amplitude
        gdat.dictlist['rvsapred'] = tesstarg.util.retr_rvsa(gdat.dictlist['peri'], gdat.dictlist['massplan'], gdat.massstar, \
                                                                                                    gdat.dictlist['incl'], gdat.dictlist['ecce'])
        
        # TSM
        gdat.dictlist['tsmm'] = tesstarg.util.retr_tsmm(gdat.dictlist['radiplan'], gdat.dictlist['tmptplan'], \
                                                                                    gdat.dictlist['massplan'], gdat.radistar, gdat.jmag)
        
        # ESM
        gdat.dictlist['esmm'] = tesstarg.util.retr_esmm(gdat.dictlist['tmptplan'], gdat.tmptstar, gdat.dictlist['radiplan'], gdat.radistar, gdat.jmag)
            
        gdat.dictlist['ltsm'] = np.log(gdat.dictlist['tsmm']) 
        gdat.dictlist['lesm'] = np.log(gdat.dictlist['esmm']) 

        gdat.dictlist['dura'] = np.empty((numbsamp, gdat.numbplan))
        path = gdat.pathalleorbt + 'results/mcmc_derived_samples.pickle'
        posteriors = pickle.load(open(path, 'rb'))
        for j in gdat.indxplan:
            fimp = posteriors[gdat.liststrgplan[j] + '_b_tra']
            #depth_obs = posteriors[gdat.liststrgplan[j] + '_depth_diluted_TESS'] / 1000.
            duratotl = posteriors[gdat.liststrgplan[j] + '_T_tra_tot'] / 24.
            durafull = posteriors[gdat.liststrgplan[j] + '_T_tra_full'] / 24.
            
            gdat.dictlist['dura'][:, j] = durafull
            
            print('j')
            print(j)
            print('fimp')
            summgene(fimp)
            #print('depth_obs')
            #summgene(depth_obs)
            print('duratotl')
            summgene(duratotl)
            print('durafull')
            summgene(durafull)
            print('')

        gdat.liststrgfeat = gdat.dictlist.keys()
        for strgfeat in gdat.liststrgfeat:
            gdat.dictpost[strgfeat] = np.empty((3, gdat.numbplan))
            gdat.dicterrr[strgfeat] = np.empty((3, gdat.numbplan))
            gdat.dictpost[strgfeat][0, :] = np.percentile(gdat.dictlist[strgfeat], 16., 0)
            gdat.dictpost[strgfeat][1, :] = np.percentile(gdat.dictlist[strgfeat], 50., 0)
            gdat.dictpost[strgfeat][2, :] = np.percentile(gdat.dictlist[strgfeat], 84., 0)
            gdat.dicterrr[strgfeat][0, :] = gdat.dictpost[strgfeat][0, :]
            gdat.dicterrr[strgfeat][1, :] = gdat.dictpost[strgfeat][1, :] - gdat.dictpost[strgfeat][0, :]
            gdat.dicterrr[strgfeat][2, :] = gdat.dictpost[strgfeat][2, :] - gdat.dictpost[strgfeat][1, :]
       
            print('strgfeat')
            print(strgfeat)
            print('gdat.dicterrr[strgfeat]')
            print(gdat.dicterrr[strgfeat])
            print('')

        # get GP-detrended model light curves
        gdat.lcurmodl = alles.get_posterior_median_model(strginst, 'flux', xx=gdat.time)
        gdat.lcurbasealle = alles.get_posterior_median_baseline(strginst, 'flux', xx=gdat.time)
        gdat.lcurdetrgaus = gdat.arrylcurdetr[:, 1] - gdat.lcurbasealle
        gdat.arrylcurdetrgaus = np.copy(gdat.arrylcurdetr)
        gdat.arrylcurdetrgaus[:, 1] = gdat.lcurdetrgaus
        
        gdat.arrypcurdetrgaus = [[] for j in gdat.indxplan]
        gdat.arrypcurdetrgausbind = [[] for j in gdat.indxplan]
        for j in gdat.indxplan:
            gdat.arrypcurdetrgaus[j] = tesstarg.util.fold_lcur(gdat.arrylcurdetrgaus, gdat.epocprio[j], gdat.periprio[j])
            gdat.arrypcurdetrgausbind[j] = tesstarg.util.rebn_lcur(gdat.arrypcurdetrgaus[j], numbbins)
        
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
            
        for j in gdat.indxplan:
            
            xposmaxm = gdat.dicterrr['smax'][0, j] / gdat.factaurj
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
            w1 = mpl.patches.Circle((gdat.dicterrr['smax'][0, j]/gdat.factaurj, 0), radius= gdat.dicterrr['radiplan'][0, j]/gdat.factaurj*fact, color=gdat.listcolrplan[j], zorder=3)
            axis.add_artist(w1)
            
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
   
        if gdat.makeprioplot:
            # plot all exoplanets and overplot this one
            dictexarcomp = retr_exarcomp(gdat)
            
            numbplanexar = dictexarcomp['radiplan'].size
            indxplanexar = np.arange(numbplanexar)

            # mass radius
            for strgzoom in liststrgzoom:
                
                figr, axis = plt.subplots(figsize=gdat.figrsize)
                indx = np.where(np.isfinite(dictexarcomp['stdvmassplan']) & np.isfinite(dictexarcomp['stdvradiplan']))[0]
                
                xerr = dictexarcomp['stdvmassplan'][indx] * gdat.factmjme
                yerr = dictexarcomp['stdvradiplan'][indx] * gdat.factrjre
                axis.errorbar(dictexarcomp['massplan'][indx] * gdat.factmjme, dictexarcomp['radiplan'][indx] * gdat.factrjre, \
                                                                                                    xerr=xerr, yerr=yerr, color='grey', alpha=0.1)
                gdat.listlabldenscomp = ['Earth-like', 'Pure Water', 'Pure Iron']
                listdenscomp = [1., 0.1813, 1.428]
                listposicomp = [[13., 2.7], [5., 3.5], [14., 1.7]]
                gdat.numbdenscomp = len(gdat.listlabldenscomp)
                gdat.indxdenscomp = np.arange(gdat.numbdenscomp)
                masscompdens = np.linspace(0.5, 16.) # M_E
                for i in gdat.indxdenscomp:
                    radicompdens = (masscompdens / listdenscomp[i])**(1. / 3.)
                    axis.plot(masscompdens, radicompdens, color='grey')
                for i in gdat.indxdenscomp:
                    axis.text(listposicomp[i][0], listposicomp[i][1], gdat.listlabldenscomp[i])
                for j in gdat.indxplan:
                    axis.errorbar(gdat.dicterrr['massplan'][0, j, None] * gdat.factmjme, gdat.dicterrr['radiplan'][0, j, None] * gdat.factrjre, marker='o', \
                                                           xerr=gdat.dicterrr['massplan'][1:3, j, None], yerr=gdat.dicterrr['radiplan'][1:3, j, None], \
                                                                                                                        color=gdat.listcolrplan[j])
                axis.set_ylabel(r'Radius [$R_E$]')
                axis.set_xlabel(r'Mass [$M_E$]')
                if strgzoom == 'rb14':
                    axis.set_xlim([1.5, 16])
                    axis.set_ylim([1.5, 4.])
                
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
            for strgzoom in liststrgzoom:
                figr, axis = plt.subplots(figsize=gdat.figrsize)
                
                temp = dictexarcomp['radiplan'] * gdat.factrjre
                indx = np.where(np.isfinite(temp))[0]
                if strgzoom == 'rb14':
                    binsradi = np.linspace(1.5, 4., 40)
                else:
                    radi = temp[indx]
                    binsradi = np.linspace(np.amin(radi), np.amax(radi), 40)
                meanradi = (binsradi[1:] + binsradi[:-1]) / 2.
                deltradi = binsradi[1] - binsradi[0]
                print('meanradi')
                summgene(meanradi)
                print('np.histogram(temp[indx], bins=binsradi)[0]')
                summgene(np.histogram(temp[indx], bins=binsradi)[0])
                print('deltradi')
                print(deltradi)
                axis.bar(meanradi, np.histogram(temp[indx], bins=binsradi)[0], width=deltradi, color='grey')
                for j in gdat.indxplan:
                    xposlowr = gdat.factrjre *  gdat.dictpost['radiplan'][0, j]
                    xposuppr = gdat.factrjre *  gdat.dictpost['radiplan'][2, j]
                    axis.axvspan(xposlowr, xposuppr, alpha=0.5, color=gdat.listcolrplan[j])
                    axis.axvline(gdat.factrjre *  gdat.dicterrr['radiplan'][0, j], color=gdat.listcolrplan[j], ls='--', label=gdat.liststrgplan[j])
                axis.set_xlabel('Radius [$R_E$]')
                axis.set_ylabel('N')
                plt.subplots_adjust(bottom=0.2)
                path = gdat.pathimag + 'histradi%s.%s' % (strgzoom, gdat.strgplotextn)
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()
   
                # radius vs period
                figr, axis = plt.subplots(figsize=gdat.figrsize)
                axis.errorbar(dictexarcomp['peri'][indx], temp[indx], ls='')
                for j in gdat.indxplan:
                    xerr = gdat.dicterrr['peri'][1:3, j, None]
                    yerr = gdat.factrjre * gdat.dicterrr['radiplan'][1:3, j, None]
                    axis.errorbar(gdat.dicterrr['peri'][0, j, None], gdat.factrjre * gdat.dicterrr['radiplan'][0, j, None], color=gdat.listcolrplan[j], \
                                                    xerr=xerr, yerr=yerr, ls='', marker='o')
                axis.set_xlabel('Period [days]')
                axis.set_ylabel('Radius [$R_E$]')
                if strgzoom == 'rb14':
                    axis.set_ylim([1.5, 4.])
                axis.set_xlim([None, 100.])
                plt.subplots_adjust(bottom=0.2)
                path = gdat.pathimag + 'radiperi%s.%s' % (strgzoom, gdat.strgplotextn)
                plt.savefig(path)
                plt.close()
            
            tmptplan = dictexarcomp['tmptplan']
            radiplan = dictexarcomp['radiplan'] * gdat.factrjre # R_E
            
            # known planets
            ## ESM
            esmm = tesstarg.util.retr_esmm(dictexarcomp['tmptplan'], dictexarcomp['tmptstar'], dictexarcomp['radiplan'], dictexarcomp['radistar'], \
                                                                                                                dictexarcomp['kmag'])
            
            ## TSM
            tsmm = tesstarg.util.retr_tsmm(dictexarcomp['radiplan'], dictexarcomp['tmptplan'], dictexarcomp['massplan'], dictexarcomp['radistar'], \
                                                                                                                dictexarcomp['jmag'])
            numbtext = 5
            liststrgmetr = ['tsmm']
            if np.isfinite(gdat.tmptstar):
                liststrgmetr.append('esmm')
            
            liststrglimt = ['allm', 'gmas']
            for strgmetr in liststrgmetr:
                
                if strgmetr == 'tsmm':
                    metr = np.log(tsmm)
                    lablmetr = 'TSM'
                    metrthis = gdat.dicterrr['ltsm'][0, :]
                    errrmetrthis = gdat.dicterrr['ltsm']
                else:
                    metr = np.log(esmm)
                    lablmetr = 'ESM'
                    metrthis = gdat.dicterrr['lesm'][0, :]
                    errrmetrthis = gdat.dicterrr['lesm']
                for strgzoom in liststrgzoom:
                    if strgzoom == 'allr':
                        indxzoom = np.where(np.isfinite(radiplan) & np.isfinite(tmptplan) \
                                                                                    & np.isfinite(metr) & (dictexarcomp['booltran'] == 1.))[0]
                    if strgzoom == 'rb14':
                        indxzoom = np.where((radiplan < 4) & (radiplan > 1.5) & np.isfinite(radiplan) & np.isfinite(tmptplan) \
                                                                                    & np.isfinite(metr) & (dictexarcomp['booltran'] == 1.))[0]
            
                    for strglimt in liststrglimt:
                        if strglimt == 'gmas':
                            indxlimt = np.where(dictexarcomp['stdvmassplan'] / dictexarcomp['massplan'] < 0.3)[0]
                        if strglimt == 'allm':
                            indxlimt = np.arange(esmm.size)
                    
                        indx = np.intersect1d(indxzoom, indxlimt)
                        
                        maxmmetr = max(np.amax(metrthis), np.amax(metr[indx]))
                        minmmetr = min(np.amin(metrthis), np.amin(metr[indx]))
                        
                        metr -= minmmetr
                        metrthis -= minmmetr
    
                        size = 0.05 + metr[indx] / (maxmmetr - minmmetr)
                        sizethis = 0.05 + metrthis / (maxmmetr - minmmetr)
                        
                        # sort
                        indxsort = np.argsort(metr[indx])[::-1]
                        
                        # radius vs equilibrium temperature
                        figr, axis = plt.subplots(figsize=gdat.figrsize)
                        
                        ## this system
                        for j in gdat.indxplan:
                            axis.errorbar(gdat.dicterrr['radiplan'][0, j, None] * gdat.factrjre, gdat.dicterrr['tmptplan'][0, j, None], marker='', ls='', \
                                         ms=6, lw=1, xerr=gdat.dicterrr['radiplan'][1:3, j, None], yerr=gdat.dicterrr['tmptplan'][1:3, j, None], \
                                                                                                    color=gdat.listcolrplan[j])
                            axis.text(gdat.dicterrr['radiplan'][0, j] * gdat.factrjre + gdat.offstextatmo[j][0], \
                                      gdat.dicterrr['tmptplan'][0, j] + gdat.offstextatmo[j][1], gdat.liststrgplanfull[j], \
                                                             color=gdat.listcolrplan[j], ha='center', va='center')
                        
                        ## known planets
                        axis.errorbar(radiplan[indx], tmptplan[indx], ms=3)
                        for k in indxsort[:numbtext]:
                            axis.text(radiplan[indx[k]], tmptplan[indx[k]], '%s' % dictexarcomp['nameplan'][indx[k]], ha='center', va='center')
                        
                        axis.set_ylabel(r'Planet Equilibrium Temperature [K]')
                        axis.set_xlabel('Radius [$R_E$]')
                        #axis.set_yscale('log')
                        plt.tight_layout()
                        path = gdat.pathimag + 'radiplan_tmptplan_%s_%s_%s_targ.%s' % (strgmetr, strgzoom, strglimt, gdat.strgplotextn)
                        print('Writing to %s...' % path)
                        plt.savefig(path)
                        plt.close()
                
                        # metric vs radius
                        figr, axis = plt.subplots(figsize=gdat.figrsize)
                        ## this system
                        for j in gdat.indxplan:
                            axis.errorbar(gdat.dicterrr['radiplan'][0, j, None] * gdat.factrjre, metrthis[j, None], lw=1, ms=6, \
                                    xerr=gdat.dicterrr['radiplan'][1:3, j, None], yerr=errrmetrthis[1:3, j, None], color=gdat.listcolrplan[j])
                            axis.text(gdat.dicterrr['radiplan'][0, j] * gdat.factrjre, metrthis[j], gdat.liststrgplanfull[j], \
                                                                                                    color=gdat.listcolrplan[j], ha='center', va='center')
                        
                        ## known planets
                        axis.errorbar(radiplan[indx], metr[indx], ms=3)
                        for k in indxsort[:numbtext]:
                            axis.text(radiplan[indx[k]], metr[indx[k]], '%s' % dictexarcomp['nameplan'][indx[k]], ha='center', va='center')
                        
                        axis.set_ylabel(lablmetr)
                        axis.set_xlabel('Radius [$R_E$]')
                        #axis.set_yscale('log')
                        plt.tight_layout()
                        path = gdat.pathimag + '%s_radiplan_%s_%s_targ.%s' % (strgmetr, strgzoom, strglimt, gdat.strgplotextn)
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

        # phase curve
        if gdat.boolphascurv:
            
            if gdat.pcurtype == 'alle':
                path = gdat.pathallepcur + 'TESS.csv'
                print('Writing to %s...' % path)
                np.savetxt(path, gdat.arrylcurdetr, delimiter=',', header='time,flux,flux_err')
        
                # update the transit duration for fastfit
                if not gdat.boolalleprev['pcur']['sett']:
                    lineadde = [['fast_fit,*', 'fast_fit,False']]
                    evol_file(gdat, 'settings.csv', gdat.pathallepcur, lineadde)
                
                # update the parameters
                lineadde = []
                for j in gdat.indxplan:
                    strgrrat = '%s_rr' % gdat.liststrgplan[j]
                    strgrsma = '%s_rsuma' % gdat.liststrgplan[j]
                    strgcosi = '%s_cosi' % gdat.liststrgplan[j]
                    strgepoc = '%s_epoch' % gdat.liststrgplan[j]
                    strgperi = '%s_period' % gdat.liststrgplan[j]
                    lineadde.extend([ \
                                [strgrrat + '*', '%s,%f,1,uniform %f %f,$R_{%s} / R_\star$,\n' % \
                                              (strgrrat, gdat.dicterrr['rrat'][0, j], 0, 2 * gdat.rratprio[j], gdat.liststrgplan[j])], \
                                [strgrsma + '*', '%s,%f,1,uniform %f %f,$(R_\star + R_{%s}) / a_{%s}$,\n' % \
                                              (strgrsma, gdat.dicterrr['rsma'][0, j], 0, 2 * gdat.rsmaprio[j], gdat.liststrgplan[j], gdat.liststrgplan[j])], \
                                [strgcosi + '*', '%s,%f,1,uniform %f %f,$\cos{i_{%s}}$,\n' % \
                                              (strgcosi, gdat.dicterrr['cosi'][0, j], 0, max(0.1, gdat.cosiprio[j] * 2), gdat.liststrgplan[j])], \
                                [strgepoc + '*', '%s,%f,1,uniform %f %f,$T_{0;%s}$,$\mathrm{BJD}$\n' % \
                                    (strgepoc, gdat.dictpost['epoc'][1, j], gdat.dictpost['epoc'][0, j], gdat.dictpost['epoc'][2, j], gdat.liststrgplan[j])], \
                                [strgperi + '*', '%s,%f,1,uniform %f %f,$P_{%s}$,$\mathrm{d}$\n' % \
                                              (strgperi, gdat.dicterrr['peri'][0, j], gdat.periprio[j] - 0.01, gdat.periprio[j] + 0.01, gdat.liststrgplan[j])], \
                               ])
                if not gdat.boolalleprev['pcur']['para']:
                    evol_file(gdat, 'params.csv', gdat.pathallepcur, lineadde)
                
                ## initial plot
                path = gdat.pathallepcur + 'results/initial_guess_b.pdf'
                if not os.path.exists(path):
                    allesfitter.show_initial_guess(gdat.pathallepcur)
                
                ## do the run
                path = gdat.pathallepcur + 'results/mcmc_save.h5'
                if not os.path.exists(path):
                    allesfitter.mcmc_fit(gdat.pathallepcur)
                else:
                    print('%s exists... Skipping the phase curve run.' % path)

                ## make the final plots
                path = gdat.pathallepcur + 'results/mcmc_corner.pdf'
                if not os.path.exists(path):
                    allesfitter.mcmc_output(gdat.pathallepcur)
        
                if strgalletype == 'ther':
                    listphasshft = 360. * alles.posterior_params['b_thermal_emission_timeshift_TESS'] / alles.posterior_params['b_period']
                else:
                    listalbgalle = alles.posterior_params['b_geom_albedo_TESS']
                    listdeptnigh = alles.posterior_params['b_sbratio_TESS'] * gdat.rratpost**2
            if gdat.pcurtype == 'sinu':
                numbsampwalk = 10000
                numbsampburnwalk = 0
                numbsampburnwalkseco = 8000
                
                
                numbdata = gdat.arrylcurdetr.shape[0]
                
                # mask out the primary transit
                gdat.arrypcurcentdetr = [[] for j in gdat.indxplan]
                gdat.arrypcurcentdetrbind = [[] for j in gdat.indxplan]
                gdat.indxphasotpr = [[] for j in gdat.indxplan]
                gdat.indxphasotprotse = [[] for j in gdat.indxplan]
                for j in gdat.indxplan:
                    gdat.arrypcurcentdetr[j] = tesstarg.util.fold_lcur(gdat.arrylcurdetr, gdat.epocprio[j], gdat.periprio[j], phasshft=0.25)
                    gdat.arrypcurcentdetrbind[j] = tesstarg.util.rebn_lcur(gdat.arrypcurcentdetr[j], numbbins)
                    phasmask = 2. * gdat.dicterrr['dura'][0, j] / gdat.dicterrr['peri'][0, j]
                    gdat.indxphasotpr[j] = np.where(abs(gdat.arrypcurcentdetr[j][:, 0]) > phasmask)[0]
                    gdat.indxphasotprotse[j] = np.where(abs(gdat.arrypcurcentdetr[j][gdat.indxphasotpr[j], 0] - 0.5) > phasmask)[0]
                
                # list of models
                gdat.listpcursinetype = ['simp', 'shft']
                
                for gdat.pcursinetype in gdat.listpcursinetype:
                    listlablpara = [['Offset', ''], ['Nightside', 'ppm'], ['Thermal', 'ppm'], ['Reflected', 'ppm'], ['Ellipsoidal', 'ppm']]
                    if gdat.pcursinetype == 'sinushft':
                        listlablpara += ['Phase shift', '']
                    numbpara = len(listlablpara)
                    indxpara = np.arange(numbpara)
                    listscalpara = ['self' for k in indxpara]
                    listminmpara = np.array([ -1.,  0.,  0.,  0.,  0.])
                    listmaxmpara = np.array([  1., 1e4, 1e4, 1e4, 1e4])
                    if gdat.pcursinetype == 'sinushft':
                        listminmpara = np.concatenate([listminmpara, np.array([0.])])
                        listmaxmpara = np.concatenate([listmaxmpara, np.array([360.])])
                    listmeangauspara = None
                    liststdvgauspara = None
                    
                    print('listlablpara')
                    print(listlablpara)
                    print('listminmpara')
                    print(listminmpara)
                    print('listmaxmpara')
                    print(listmaxmpara)
                    print('listscalpara')
                    print(listscalpara)
                    print('')

                    # sample
                    parapost = tdpy.mcmc.samp(gdat, gdat.pathimag, numbsampwalk, numbsampburnwalk, numbsampburnwalkseco, retr_llik_sinu, \
                                        listlablpara, listscalpara, listminmpara, listmaxmpara, listmeangauspara, liststdvgauspara, numbdata)
                    
                    numbsamp = parapost.shape[0]
                    indxsamp = np.arange(numbsamp)
                
                    listlablparafull = listlablpara[:]
                    listlablparafull.extend([['Secondary', 'ppm'], ['Modulation', 'ppm'], ['Geometric Albedo', '']])
                    numbparafull = len(listlablparafull)
                    parapostfull = np.empty((numbsamp, numbparafull))
                    parapostfull[:, :numbpara] = np.copy(parapost)
                    listdeptrefl = parapost[:, 3]
                    listdeptpmod = parapost[:, 2] + listdeptrefl
                    listdeptseco = parapost[:, 1] + listdeptpmod
                    listalbgalle = 1e-6 * listdeptpmod / (gdat.dicterrr['rrat'][0, 0] * gdat.dicterrr['rsma'][0, 0] / (1. + gdat.dicterrr['rrat'][0, 0]))**2
                    parapostfull[:, numbpara] = listdeptseco
                    parapostfull[:, numbpara+1] = listdeptpmod
                    parapostfull[:, numbpara+2] = listalbgalle
                    tdpy.mcmc.plot_grid(gdat.pathimag, 'post_alle_full', parapostfull, listlablparafull, plotsize=2.5)
                    
                    ### sample model phas
                    numbsampplot = 100
                    indxsampplot = np.random.choice(indxsamp, numbsampplot, replace=False)
                    
                    # plot the posterior
                    numbphasfine = 1000
                    gdat.meanphasfine = np.linspace(0.1, 0.9, numbphasfine)
                    phasmodlfine = np.empty((numbsampplot, numbphasfine))
                    
                    figr, axis = plt.subplots(figsize=gdat.figrsizeydob)
                    axis.errorbar(gdat.arrypcurcentdetrbind[0][:, 0], (gdat.arrypcurcentdetrbind[0][:, 1] - 1) * 1e6, \
                                                                yerr=1e6*gdat.arrypcurcentdetrbind[0][:, 2], color='k', marker='o', ls='', markersize=1)
                    for k, indxsampplottemp in enumerate(indxsampplot):
                        axis.plot(gdat.meanphasfine, (phasmodlfine[k, :] - 1) * 1e6, alpha=0.5, color='b')
                    axis.set_xlim([0.1, 0.9])
                    axis.set_ylim([-400, 1000])
                    axis.set_ylabel('Relative Flux - 1 [ppm]')
                    axis.set_xlabel('Phase')
                    plt.tight_layout()
                    path = gdat.pathimag + 'pcur_sine_%s.%s' % (strgextn, gdat.strgplotextn)
                    print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()

                    gdat.numbtime = gdat.arrylcurdetr.shape[0]
                    numbsamp = parapost.shape[0]
                    indxsamp = np.arange(numbsamp)
                    listarrysamp = np.empty((numbsampplot, gdat.numbtime, 3))
                    listarrysamp[:, :, 0] = gdat.arrylcur[None, :, 0]
                    for kk, k in enumerate(indxsampplot):
                        listarrysamp[k, :, 1], _ = retr_modl_sinu(gdat, parapost[k, :], phasmodlfine, indxphasfineotse)
                    listarrysamp[:, :, 2] = 0.
    
                    # plot light curve + samples from the trapezoid model
                    figr, axis = plt.subplots(figsize=gdat.figrsizeydob)
                    axis.plot(gdat.arrylcurdetr[:, 0], gdat.arrylcurdetr[:, 1], color='grey', alpha=0.3)
                    axis.plot(gdat.arrylcurdetrbind[:, 0], gdat.arrylcurdetrbind[:, 1], color='k', ls='')
                    for i in np.arange(numbsampplot):
                        axis.plot(gdat.arrylcurdetr[:, 0], listarrysamp[i, :, 1], alpha=0.2, color='b')
                    axis.set_xlabel('Time [BJD]')
                    plt.subplots_adjust()
                    path = gdat.pathimag + 'lcur_pcur_sinu.%s' % gdat.strgplotextn
                    print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()

                    # plot phase curve + samples from the trapezoid model
                    figr, axis = plt.subplots(figsize=gdat.figrsizeydob)
                    axis.plot(gdat.arrypcurdetr[j][:, 0], gdat.arrypcurdetr[j][:, 1], color='grey', alpha=0.3)
                    axis.plot(gdat.arrypcurdetrbind[j][:, 0], gdat.arrypcurdetrbind[j][:, 1], color='k', ls='')
                    for i in np.arange(numbsampplot):
                        pcur = tesstarg.util.fold_lcur(listarrysamp[i, :, :], gdat.epocpost[j], gdat.peripost[j])
                        axis.plot(pcur[:, 0], pcur[:, 1], alpha=0.2, color='b')
                    axis.set_xlim([-0.01, 0.01])
                    axis.set_xlabel('Phase')
                    plt.subplots_adjust()
                    path = gdat.pathimag + 'pcur_pcur_sinu.%s' % gdat.strgplotextn
                    print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()
                    
                    # write to text file
                    fileoutp = open(pathdata + 'post.csv', 'w')
                    fileoutp.write(' & ')
                    fileoutp.write(' & ')
                    for n in indxparafull:
                        fileoutp.write('%s & ' % listlablparafull[n])
                        ydat = np.median(postpara[:, n])
                        uerr = np.percentile(postpara[:, n], 84.) - ydat
                        lerr = ydat - np.percentile(postpara[:, n], 16.)
                        fileoutp.write('$%.3g \substack{+%.3g \\\\ -%.3g}$' % (ydat, uerr, lerr))
                        fileoutp.write(' & ')
                        fileoutp.write('\\\\\n')
                    fileoutp.write('\\hline\n')
                    fileoutp.close()
    
            path = pathdata + 'PC-Solar-NEW-OPA-TiO-LR.dat'
            arryvivi = np.loadtxt(path, delimiter=',')
            phasvivi = (arryvivi[:, 0] / 360. + 0.75) % 1. - 0.25
            deptvivi = arryvivi[:, 4]
            indxphasvivisort = np.argsort(phasvivi)
            phasvivi = phasvivi[indxphasvivisort]
            deptvivi = deptvivi[indxphasvivisort]
            path = pathdata + 'PC-Solar-NEW-OPA-TiO-LR-AllK.dat'
            arryvivi = np.loadtxt(path, delimiter=',')
            wlenvivi = arryvivi[:, 1]
            specvivi = arryvivi[:, 2]

            # determine data gaps for overplotting model without the data gaps
            gdat.indxtimegapp = np.argmax(gdat.time[1:] - gdat.time[:-1]) + 1
            figr = plt.figure(figsize=gdat.figrsizeydob)
            axis = [[] for k in range(3)]
            axis[0] = figr.add_subplot(3, 1, 1)
            axis[1] = figr.add_subplot(3, 1, 2)
            axis[2] = figr.add_subplot(3, 1, 3, sharex=axis[1])
            
            for k in range(len(axis)):
                
                if k > 0:
                    xdat = gdat.arrypcurdetrbind[j][:, 0]
                else:
                    xdat = gdat.time - gdat.timetess
                
                if k > 0:
                    ydat = gdat.arrypcurdetrbind[j][:, 1]
                else:
                    ydat = gdat.lcurdetr
                if k == 2:
                    ydat = (ydat - 1. + medideptnigh) * 1e6
                axis[k].plot(xdat, ydat, '.', color='grey', alpha=0.3, label='Raw data')
                
                if k > 0:
                    xdat = gdat.arrypcurdetrbind[j][:, 0]
                    ydat = gdat.arrypcurdetrbind[j][:, 1]
                    yerr = np.copy(gdat.arrypcurdetrbind[j][:, 2])
                else:
                    xdat = gdat.arrylcurdetrbind[:, 0] - gdat.timetess
                    ydat = gdat.arrylcurdetrbind[:, 1]
                    yerr = gdat.arrylcurdetrbind[:, 2]
                if k == 2:
                    ydat = (ydat - 1. + medideptnigh) * 1e6
                    yerr *= 1e6
                axis[k].errorbar(xdat, ydat, marker='o', yerr=yerr, capsize=0, ls='', color='k', label='Binned data')
                
                if k > 0:
                    xdat = gdat.arrypcurdetr[j][:, 0]
                    ydat = gdat.lcurmodl
                else:
                    xdat = gdat.time - gdat.timetess
                    ydat = gdat.lcurmodl
                if k == 2:
                    ydat = (ydat - 1. + medideptnigh) * 1e6
                
                if k == 0:
                    axis[k].plot(xdat[:gdat.indxtimegapp], ydat[:gdat.indxtimegapp], color='b', lw=2, label='Total Model')
                    axis[k].plot(xdat[gdat.indxtimegapp:], ydat[gdat.indxtimegapp:], color='b', lw=2)
                else:
                    axis[k].plot(xdat, ydat, color='b', lw=2, label='Model (This work)')
                
                if k == 2:
                    #axis[k].plot(phasvivi, deptvivi, color='orange', lw=2, label='GCM (Parmentier+2018)')

                    axis[k].axhline(0., ls='-.', alpha=0.3, color='grey')

                if k == 0:
                    axis[k].set(xlabel='Time (BJD)')
                if k > 0:
                    axis[k].set(xlabel='Phase')
            axis[0].set(ylabel='Relative Flux')
            axis[1].set(ylabel='Relative Flux')
            axis[2].set(ylabel='Relative Flux - 1 [ppm]')
            #axis[1].set(ylim=[-800,1000])
            axis[2].set(ylim=[-400, 1000])
        
            ## plot components in the zoomed panel
            ### EV
            alles = allesfitter.allesclass(gdat.pathallepcur)
            alles.posterior_params_median['b_sbratio_TESS'] = 0
            #alles.settings['host_shape_TESS'] = 'sphere'
            #alles.settings['b_shape_TESS'] = 'sphere'
            alles.posterior_params_median['b_geom_albedo_TESS'] = 0
            alles.posterior_params_median['host_gdc_TESS'] = 0
            alles.posterior_params_median['host_bfac_TESS'] = 0
            gdat.lcurmodlcomp = alles.get_posterior_median_model(strginst, 'flux', xx=gdat.time)
            gdat.lcurmodlevvv = np.copy(gdat.lcurmodlcomp)
            
            xdat = gdat.arrypcurdetr[j][:, 0]
            ydat = (gdat.lcurmodlcomp - 1.) * 1e6
            indxfrst = np.where(xdat < -0.07)[0]
            indxseco = np.where(xdat > 0.07)[0]
            axis[2].plot(xdat[indxfrst], ydat[indxfrst], lw=2, color='r', label='Ellipsoidal', ls='--')
            axis[2].plot(xdat[indxseco], ydat[indxseco], lw=2, color='r', ls='--')
            
            objtalle = allesfitter.allesclass(gdat.pathallepcur)
            alles.posterior_params_median['b_sbratio_TESS'] = 0
            alles.posterior_params_median['b_geom_albedo_TESS'] = 0
            alles.posterior_params_median['host_gdc_TESS'] = 0
            alles.posterior_params_median['host_bfac_TESS'] = 0

            ### planetary modulation
            alles = allesfitter.allesclass(gdat.pathallepcur)
            alles.posterior_params_median['b_sbratio_TESS'] = 0
            alles.settings['host_shape_TESS'] = 'sphere'
            alles.settings['b_shape_TESS'] = 'sphere'
            #alles.posterior_params_median['b_geom_albedo_TESS'] = 0
            alles.posterior_params_median['host_gdc_TESS'] = 0
            alles.posterior_params_median['host_bfac_TESS'] = 0
            gdat.lcurmodlcomp = alles.get_posterior_median_model(strginst, 'flux', xx=gdat.time)
            #axis[2].plot(gdat.arrypcurdetr[j][:, 0], medideptnigh + (gdat.lcurmodlcomp - 1.) * 1e6, \
            #                                                            lw=2, color='g', label='Planetary Modulation', ls='--', zorder=11)
            
            axis[2].legend(ncol=2)
            
            path = gdat.pathobjt + 'pcur_alle.%s' % gdat.strgplotextn
            plt.savefig(path)
            plt.close()
        
        
            # plot the spherical limits
            figr, axis = plt.subplots(figsize=gdat.figrsizeydob)
            
            alles = allesfitter.allesclass(gdat.pathallepcur)
            alles.posterior_params_median['b_sbratio_TESS'] = 0
            alles.settings['host_shape_TESS'] = 'sphere'
            alles.settings['b_shape_TESS'] = 'roche'
            alles.posterior_params_median['b_geom_albedo_TESS'] = 0
            alles.posterior_params_median['host_gdc_TESS'] = 0
            alles.posterior_params_median['host_bfac_TESS'] = 0
            lcurmodltemp = alles.get_posterior_median_model(strginst, 'flux', xx=gdat.time)
            axis.plot(gdat.arrypcurdetr[j][:, 0], (gdat.lcurmodlevvv - lcurmodltemp) * 1e6, lw=2, label='Spherical star')
            
            alles = allesfitter.allesclass(gdat.pathallepcur)
            alles.posterior_params_median['b_sbratio_TESS'] = 0
            alles.settings['host_shape_TESS'] = 'roche'
            alles.settings['b_shape_TESS'] = 'sphere'
            alles.posterior_params_median['b_geom_albedo_TESS'] = 0
            alles.posterior_params_median['host_gdc_TESS'] = 0
            alles.posterior_params_median['host_bfac_TESS'] = 0
            lcurmodltemp = alles.get_posterior_median_model(strginst, 'flux', xx=gdat.time)
            axis.plot(gdat.arrypcurdetr[j][:, 0], (gdat.lcurmodlevvv - lcurmodltemp) * 1e6, lw=2, label='Spherical planet')
            axis.legend()
            axis.set_ylim([-100, 100])
            axis.set(xlabel='Phase')
            axis.set(ylabel='Relative flux [ppm]')
            plt.subplots_adjust(hspace=0.)
            path = pathimag + 'pcurmodldiff.%s' % gdat.strgplotextn
            plt.savefig(path)
            plt.close()

            # calculate postr on the mass ratio (Stassun+2017)
            Mp = np.random.normal(loc=(375.99289 *c.M_earth/c.M_sun).value, scale=(20.34112*c.M_earth/c.M_sun).value, size=10000)
            Ms = np.random.normal(loc=1.52644, scale=0.361148, size=10000)
            ratimass = massplan / massstar
            fig = plt.figure()
            plt.hist(Mp)
            fig = plt.figure()
            plt.hist(Ms)
            fig = plt.figure()
            plt.hist(q)
  
            # use psi posterior to infer Bond albedo and heat circulation efficiency

            # get data
            ## from Tom
            path = pathdata + 'ascii_output/EmissionModelArray.txt'
            arrymodl = np.loadtxt(path)
            path = pathdata + 'ascii_output/EmissionDataArray.txt'
            arrydata = np.loadtxt(path)
            # update Tom's array with the new secondary depth
            arrydata[0, 2] = medideptseco
            arrydata[0, 3] = stdvdeptseco
            ### add the nightsiide emission
            arrydata = np.concatenate((arrydata, np.array([[arrydata[0, 0], arrydata[0, 1], medideptnigh, stdvdeptnigh, 0, 0, 0, 0]])), axis=0)
            ### spectrum of the host star
            gdat.meanwlenthomraww = arrymodl[:, 0]
            gdat.specstarthomraww = arrymodl[:, 9]
    
            # make a contour plot of geometric albedo without and with thermal component postr
            ## calculate the geometric albedo with the ATMO postr
            wlenmodl = arrymodl[:, 0]
            deptmodl = arrymodl[:, 1]
            indxwlenmodltess = np.where((wlenmodl > 0.6) & (wlenmodl < 0.95))[0]
            deptmodlther = np.mean(deptmodl[indxwlenmodltess])
                
            arrydata = np.empty((2, 4))
            arrydata[0, 0] = 0.8
            arrydata[0, 1] = 0.2
            arrydata[0, 2] = medideptseco
            arrydata[0, 3] = stdvdeptseco
            arrydata[1, 0] = 0.8
            arrydata[1, 1] = 0.2
            arrydata[1, 2] = medideptnigh
            arrydata[1, 3] = stdvdeptnigh
            listdeptrefl = listdeptpmod * np.random.rand(numbsamp)
            
            print('HACKING')
            listalbg = gdat.dictlist['deptrefl'] * (gdat.dictlist['smax'] / gdat.dictlist['radiplan'])**2
            listalbg = listalbg[listalbg > 0]
            
            prnt_list(listalbg, 'Albedo')
            
            # wavelength axis
            gdat.conswlentmpt = 0.0143877735e6 # [um K]
        
            figr, axis = plt.subplots(figsize=gdat.figrsizeydob)

            binsalbg = np.linspace(min(np.amin(listalbgtess), np.amin(listalbg)), max(np.amax(listalbgtess), np.amax(listalbg)), 100)
            meanalbg = (binsalbg[1:] + binsalbg[:-1]) / 2.
            pdfnalbgtess = scipy.stats.gaussian_kde(listalbgtess, bw_method=.2)(meanalbg)
            pdfnalbg = scipy.stats.gaussian_kde(listalbg, bw_method=.2)(meanalbg)
            #pdfnalbgtess = np.histogram(listalbgtess, bins=binsalbg)[0] / float(listalbgtess.size)
            #pdfnalbg = np.histogram(listalbg, bins=binsalbg)[0] / float(listalbg.size)
            axis.plot(meanalbg, pdfnalbgtess, label='TESS only', lw=2)
            axis.plot(meanalbg, pdfnalbg, label='TESS + ATMO postr', lw=2)
            axis.set_xlabel('$A_g$')
            axis.set_ylabel('$P(A_g)$')
            axis.legend()
            plt.subplots_adjust()
            path = pathimag + 'pdfn_albg.%s' % gdat.strgplotextn
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
            
            
            ## read eclipse data
            liststrgfile = ['ContribFuncArr.txt', \
                            'EmissionDataArray.txt', \
                            #'RetrievalParamSamples.txt', \
                            'ContribFuncWav.txt', \
                            'EmissionModelArray.txt', \
                            'RetrievalPTSamples.txt', \
                            'pdependent_abundances/', \
                            ]
            
            
            ## get posterior on irradiation efficiency
            path = pathdata + 'ascii_output/RetrievalParamSamples.txt'
            listsampatmo = np.loadtxt(path)
            gdat.listpsii = listsampatmo[:, 2]
            histpsii, binspsii = np.histogram(gdat.listpsii)
            meanpsii = (binspsii[1:] + binspsii[:-1]) / 2.
            gdat.likeintp = scipy.interpolate.interp1d(meanpsii, np.log(histpsii), fill_value=-np.inf, bounds_error=False)
            
            ## sample from Bond albedo and circulation efficiency
            numbwalk = 10
            dictalbb = [gdat]
            indxwalk = np.arange(numbwalk)
            parainit = []
            for i in indxwalk:
                parainit.append(np.random.randn(2) * 0.001 + 0.02)
            objtsamp = emcee.EnsembleSampler(numbwalk, 2, retr_lpos_albb, args=dictalbb)
            objtsamp.run_mcmc(parainit, 10000)
            listalbb = objtsamp.flatchain[:, 0]
            listepsi = objtsamp.flatchain[:, 1]

            listsamp = np.empty((listalbb.size, 2))
            listsamp[:, 0] = listalbb
            listsamp[:, 1] = listepsi
            tdpy.mcmc.plot_grid(pathimag, 'post_albbespi', listsamp, ['$A_b$', r'$\varepsilon$'], plotsize=2.5)

            # plot ATMO posterior
            tdpy.mcmc.plot_grid(pathimag, 'post_atmo', listsampatmo, ['$\kappa_{IR}$', '$\gamma$', '$\psi$', \
                                                                                '[M/M$_{\odot}$]', '[C/C$_{\odot}$]', '[O/O$_{\odot}$]'], plotsize=2.5)
   

            # plot spectrum, depth, brightness temp
            path = pathdata + 'ascii_output/ContribFuncWav.txt'
            wlen = np.loadtxt(path)
            listcolr = ['k', 'm', 'purple', 'olive', 'olive', 'r', 'g']
            for i in range(15):
                listcolr.append('r')
            for i in range(28):
                listcolr.append('g')
   
            # calculate brightness temperatures
            numbpara = 1
            numbsampwalk = 1000
            numbsampburnwalk = 100
            numbwalk = 5 * numbpara
            indxwalk = np.arange(numbwalk)
            numbsamp = numbsampwalk * numbwalk
            numbsampburn = numbsampburnwalk * numbwalk
            indxsampwalk = np.arange(numbsampwalk)
            gdat.numbdatatmpt = arrydata.shape[0]
            gdat.indxdatatmpt = np.arange(gdat.numbdatatmpt)
            tmpt = np.empty((numbsamp, gdat.numbdatatmpt))
            specarry = np.empty((2, 3, gdat.numbdatatmpt))
            for k in gdat.indxdatatmpt:
                
                gdat.minmwlen = arrydata[k, 0] - arrydata[k, 1]
                gdat.maxmwlen = arrydata[k, 0] + arrydata[k, 1]
                gdat.binswlen = np.linspace(gdat.minmwlen, gdat.maxmwlen, 100)
                gdat.meanwlen = (gdat.binswlen[1:] + gdat.binswlen[:-1]) / 2.
                gdat.diffwlen = (gdat.binswlen[1:] - gdat.binswlen[:-1]) / 2.
                gdat.cntrwlen = np.mean(gdat.meanwlen)
                if not (k == 0 or k == gdat.numbdatatmpt - 1):
                    continue
                gdat.indxenerdata = k

                gdat.specstarintg = retr_modl_spec(gdat, gdat.tmptstarpost, strgtype='intg')
                
                gdat.specstarthomlogt = scipy.interpolate.interp1d(gdat.meanwlenthomraww, gdat.specstarthomraww)(gdat.cntrwlen)
                gdat.specstarthomdiff = gdat.specstarthomlogt / gdat.cntrwlen
                gdat.specstarthomintg = np.sum(gdat.diffwlen * \
                                        scipy.interpolate.interp1d(gdat.meanwlenthomraww, gdat.specstarthomraww)(gdat.meanwlen) / gdat.meanwlen)

                gdat.deptobsd = arrydata[k, 2]
                gdat.stdvdeptobsd = arrydata[k, 3]
                gdat.varideptobsd = gdat.stdvdeptobsd**2
            
                listlablpara = ['Temperature']
                liststrgpara = ['tmpt']
                indxpara = np.arange(numbpara)
                gdat.limtpara = np.empty((2, numbpara))
                # cons
                gdat.limtpara[0, 0] = 1000.
                gdat.limtpara[1, 0] = 4000.
                dictspec = [gdat]
                indxwalk = np.arange(numbwalk)
                parainit = []
                for i in indxwalk:
                    parainit.append(np.empty(numbpara))
                    meannorm = (gdat.limtpara[0, :] + gdat.limtpara[1, :]) / 2.
                    stdvnorm = (gdat.limtpara[0, :] - gdat.limtpara[1, :]) / 10.
                    parainit[i]  = (scipy.stats.truncnorm.rvs((gdat.limtpara[0, :] - meannorm) / stdvnorm, \
                                                                (gdat.limtpara[1, :] - meannorm) / stdvnorm)) * stdvnorm + meannorm
                if numbsampburnwalk == 0:
                    raise Exception('')
                objtsamp = emcee.EnsembleSampler(numbwalk, numbpara, retr_lpos_spec, args=dictspec)
                parainitburn, prob, state = objtsamp.run_mcmc(parainit, numbsampburnwalk)
                objtsamp.reset()
                objtsamp.run_mcmc(parainitburn, numbsampwalk)
                tmpt[:, k] = objtsamp.flatchain[:, 0]

            posttmpttess = tmpt[:, np.array([0, gdat.numbdatatmpt - 1])]
            
            prnt_list(posttmpttess[:, 0], 'Dayside temperature')
            prnt_list(posttmpttess[:, 1], 'Nightside temperature')
            
            listtmptcont = (posttmpttess[:, 0] - posttmpttess[:, 1]) / posttmpttess[:, 0]
            prnt_list(listtmptcont, 'Temperature contrast')
            
            figr, axis = plt.subplots(4, 1, figsize=gdat.figrsizeydob, sharex=True)
            ## stellar spectrum and TESS throughput
            axis[0].plot(arrymodl[:, 0], 1e-9 * arrymodl[:, 9], label='Host star', color='grey')
            axis[0].plot(0., 0., ls='--', label='TESS Throughput', color='grey')
            axis[0].set_ylabel(r'$\nu F_{\nu}$ [10$^9$ erg/s/cm$^2$]')
            axis[0].legend(fancybox=True, bbox_to_anchor=[0.8, 0.17, 0.2, 0.2])
            axistwin = axis[0].twinx()
            axistwin.plot(gdat.meanwlenband, gdat.thptband, color='grey', ls='--', label='TESS')
            axistwin.set_ylabel(r'Throughput')
            
            ## eclipse depths
            ### model
            axis[1].plot(arrymodl[:, 0], arrymodl[:,1], label='1D Retrieval (This work)')
            axis[1].plot(arrymodl[:, 0], arrymodl[:,2], label='Blackbody (This work)', alpha=0.3, color='skyblue')
            axis[1].fill_between(arrymodl[:, 0], arrymodl[:, 3], arrymodl[:, 4], alpha=0.3, color='skyblue')
            #objtplotvivi, = axis[1].plot(wlenvivi, specvivi * 1e6, color='orange', alpha=0.6, lw=2)
            ### data
            for k in range(5):
                axis[1].errorbar(arrydata[k, 0], arrydata[k, 2], xerr=arrydata[k, 1], yerr=arrydata[k, 3], ls='', marker='o', color=listcolr[k])
            axis[1].errorbar(arrydata[5:22, 0], arrydata[5:22, 2], xerr=arrydata[5:22, 1], yerr=arrydata[5:22, 3], ls='', marker='o', color='r')
            axis[1].errorbar(arrydata[22:-1, 0], arrydata[22:-1, 2], xerr=arrydata[22:-1, 1], yerr=arrydata[22:-1, 3], ls='', marker='o', color='g')
            axis[1].set_ylabel(r'Depth [ppm]')
            axis[1].set_xticklabels([])
            
            ## spectra
            ### model
            objtplotretr, = axis[2].plot(arrymodl[:, 0], 1e-9 * arrymodl[:, 5], label='1D Retrieval (This work)', color='b')
            objtplotmblc, = axis[2].plot(arrymodl[:, 0], 1e-9 * arrymodl[:, 6], label='Blackbody (This work)', color='skyblue', alpha=0.3)
            objtploteblc = axis[2].fill_between(arrymodl[:, 0], 1e-9 * arrymodl[:, 7], 1e-9 * arrymodl[:, 8], color='skyblue', alpha=0.3)
            axis[2].legend([objtplotretr, (objtplotmblc, objtploteblc), objtplotvivi], \
                                                                        ['1D Retrieval (This work)', 'Blackbody (This work)', 'GCM (Parmentier+2018)'], \
                                                                                    bbox_to_anchor=[0.8, 1.3, 0.2, 0.2])
            ### data
            for k in range(5):
                axis[2].errorbar(arrydata[k, 0],  1e-9 * arrydata[k, 6], xerr=arrydata[k, 1], yerr=1e-9*arrydata[k, 7], ls='', marker='o', color=listcolr[k])
            axis[2].errorbar(arrydata[5:22, 0], 1e-9 * arrydata[5:22, 6], xerr=arrydata[5:22, 1], yerr=1e-9*arrydata[5:22, 7], ls='', marker='o', color='r')
            axis[2].errorbar(arrydata[22:-1, 0], 1e-9 * arrydata[22:-1, 6], xerr=arrydata[22:-1, 1], yerr=1e-9*arrydata[22:-1, 7], ls='', marker='o', color='g')
            

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
            axis[3].errorbar(arrydata[5:22, 0], arrydata[5:22, 4], xerr=arrydata[5:22, 1], yerr=arrydata[5:22, 5], label='HST G102 (Evans+2019)', ls='', marker='o', color='r')
            axis[3].errorbar(arrydata[22:-1, 0], arrydata[22:-1, 4], xerr=arrydata[22:-1, 1], yerr=arrydata[22:-1, 5], label='HST G141 (Evans+2017)', ls='', marker='o', color='g')
            #axis[3].errorbar(arrydata[:, 0], np.median(tmpt, 0), xerr=arrydata[:, 1], yerr=np.std(tmpt, 0), label='My calc', ls='', marker='o', color='c')
            axis[3].set_ylabel(r'$T_B$ [K]')
            axis[3].set_xlabel(r'$\lambda$ [$\mu$m]')
            axis[3].legend(fancybox=True, bbox_to_anchor=[0.8, 3.8, 0.2, 0.2], ncol=2)
            
            axis[1].set_ylim([20, None])
            axis[1].set_yscale('log')
            for i in range(4):
                axis[i].set_xscale('log')
            axis[3].set_xlim([0.5, 5])
            axis[3].xaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
            axis[3].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            plt.subplots_adjust(hspace=0., wspace=0.)
            path = pathimag + 'spec.%s' % gdat.strgplotextn
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
            
            
            # get contribution function
            # interpolate the throughput
            path = pathdata + 'ascii_output/ContribFuncArr.txt'
            ctrb = np.loadtxt(path)
            presctrb = ctrb[0, :]
            path = pathdata + 'ascii_output/ContribFuncWav.txt'
            wlenctrb = np.loadtxt(path, skiprows=1)
            gdat.thptbandctrb = scipy.interpolate.interp1d(gdat.meanwlenband, gdat.thptband, fill_value=0, bounds_error=False)(wlenctrb)
            numbwlenctrb = wlenctrb.size
            indxwlenctrb = np.arange(numbwlenctrb)
            numbpresctrb = presctrb.size
            indxpresctrb = np.arange(numbpresctrb)
            ctrbtess = np.empty(numbpresctrb)
            for k in indxpresctrb:
                ctrbtess[k] = np.sum(ctrb[1:, k] * gdat.thptbandctrb)
            ctrbtess *= 1e-12 / np.amax(ctrbtess)

            # plot pressure-temperature, contribution
            path = pathdata + 'ascii_output/RetrievalPTSamples.txt'
            dataptem = np.loadtxt(path)
            
            liststrgcomp = ['CH4.txt', 'CO.txt', 'FeH.txt', 'H+.txt', 'H.txt', 'H2.txt', 'H2O.txt', 'H_.txt', 'He.txt', 'K+.txt', \
                                                                'K.txt', 'NH3.txt', 'Na+.txt', 'Na.txt', 'TiO.txt', 'VO.txt', 'e_.txt']
            listlablcomp = ['CH$_4$', 'CO', 'FeH', 'H$^+$', 'H', 'H$_2$', 'H$_2$O', 'H$^-$', 'He', 'K$^+$', \
                                                                'K', 'NH$_3$', 'Na$^+$', 'Na', 'TiO', 'VO', 'e$^-$']
            listdatacomp = []
            for strg in liststrgcomp:
                path = pathdata + 'ascii_output/pdependent_abundances/' + strg
                listdatacomp.append(np.loadtxt(path))
            
            ## contibution/PT/abun
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
            axistwin.fill(ctrbtess, presctrb, alpha=0.5, color='grey')
            axistwin.set_xticklabels([])
            ## abundance
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
                
            #liststrgcomp = ['CH4.txt', 'CO.txt', 'FeH.txt', 'H+.txt', 'H.txt', 'H2.txt', 'H2O.txt', 'H_.txt', 'He.txt', 'K+.txt', \
            #                                                    'K.txt', 'NH3.txt', 'Na+.txt', 'Na.txt', 'TiO.txt', 'VO.txt', 'e_.txt']
            #axis[1].set_yticklabels([])
            axis[1].set_xscale('log')
            axis[1].set_xlabel('Volume Mixing Ratio')
            axis[1].set_yscale('log')
            axis[1].set_xlim([1e-16, 1])
            #axis[1].invert_yaxis()
            plt.subplots_adjust(hspace=0., wspace=0., bottom=0.15)
            path = pathimag + 'ptem.%s' % gdat.strgplotextn
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
  
    if gdat.infetype == 'trap':
        numbsampwalk = 10000
        numbsampburnwalk = 1000
        # dilution, offset, duration
        listlablpara = [['D', ''], ['O', ''], ['T', '']]
        listscalpara = ['self', 'self', 'self']
        listminmpara = [0., -0.01, 0.002]
        listmaxmpara = [0.5, 0.01, 0.01]
        listmeangauspara = None
        liststdvgauspara = None
        
        numbdata = gdat.arrylcur.shape[0]
        
        print('listlablpara')
        print(listlablpara)
        print('listminmpara')
        print(listminmpara)
        print('listmaxmpara')
        print(listmaxmpara)
        print('listscalpara')
        print(listscalpara)
        print('')

        parapost = tdpy.mcmc.samp(gdat, gdat.pathimag, numbsampwalk, numbsampburnwalk, retr_llik_trap, \
                            listlablpara, listscalpara, listminmpara, listmaxmpara, listmeangauspara, liststdvgauspara, numbdata)
            
        print('parapost')
        print(parapost)
        numbsampplot = 4
        gdat.numbtime = gdat.arrylcur.shape[0]
        numbsamp = parapost.shape[0]
        indxsamp = np.arange(numbsamp)
        indxsampplot = np.random.choice(indxsamp, size=numbsampplot)
        listarrysamp = np.empty((numbsampplot, gdat.numbtime, 3))
        listarrysamp[:, :, 0] = gdat.arrylcur[None, :, 0]
        for i in np.arange(numbsampplot):
            listarrysamp[i, :, 1], _ = retr_modl_trap(gdat, parapost[i, :])
        listarrysamp[:, :, 2] = 0.
    
        # plot light curve + samples from the trapezoid model
        figr, axis = plt.subplots(figsize=gdat.figrsizeydob)
        axis.plot(gdat.arrylcurdetr[:, 0], gdat.arrylcurdetr[:, 1], color='grey', alpha=0.3)
        axis.plot(gdat.arrylcurdetrbind[:, 0], gdat.arrylcurdetrbind[:, 1], color='k', ls='')
        for i in np.arange(numbsampplot):
            axis.plot(gdat.arrylcurdetr[:, 0], listarrysamp[i, :, 1], alpha=0.2, color='b')
        axis.set_xlabel('Time [BJD]')
        plt.subplots_adjust()
        path = gdat.pathimag + 'lcurtrap.%s' % gdat.strgplotextn
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()

        # plot phase curve + samples from the trapezoid model
        figr, axis = plt.subplots(figsize=gdat.figrsizeydob)
        axis.plot(gdat.arrypcurdetr[j][:, 0], gdat.arrypcurdetr[j][:, 1], color='grey', alpha=0.3)
        axis.plot(gdat.arrypcurdetrbind[j][:, 0], gdat.arrypcurdetrbind[j][:, 1], color='k', ls='')
        for i in np.arange(numbsampplot):
            pcur = tesstarg.util.fold_lcur(listarrysamp[i, :, :], gdat.epocpost[j], gdat.peripost[j])
            axis.plot(pcur[:, 0], pcur[:, 1], alpha=0.2, color='b')
        axis.set_xlim([-0.01, 0.01])
        axis.set_xlabel('Phase')
        plt.subplots_adjust()
        path = gdat.pathimag + 'pcurtrap.%s' % gdat.strgplotextn
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()

