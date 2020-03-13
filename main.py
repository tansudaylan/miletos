import allesfitter
import allesfitter.config

import tdpy.util
import tdpy.mcmc
from tdpy.util import prnt_list
from tdpy.util import summgene
import tcat.main

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


def retr_lpos_four(para, gdat, modltype):
    
    cons = para[0]
    deptnigh =para[1]
    amfo = para[2:8]
    if modltype == 'shft':
        shft = para[8]
    else:
        shft = 0.
    
    if ((para < gdat.limtpara[0, :]) | (para > gdat.limtpara[1, :])).any():
        lpos = -np.inf
    else:
        llik, deptpmod = retr_llik_four(gdat, cons, deptnigh, amfo, shft, modltype)
        #lpri = - 0.5 * ((amfo[3] / 0.1)**2)
        #lpos = llik + lpri
        lpos = llik

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
        deptplan = gdat.rratprio**2 * specboloplan / gdat.specstarintg
        llik = -0.5 * np.sum((deptplan - gdat.deptobsd)**2 / gdat.varideptobsd)
        lpos = llik
    
    return lpos


def retr_llik_four(gdat, cons, deptnigh, amfo, shft, modltype):
    
    modl, deptpmod = retr_modl_four(gdat, gdat.meanphas, gdat.indxphasseco, cons, deptnigh, amfo, shft, modltype)
    
    llik = -0.5 * np.sum((modl - gdat.data)**2 / gdat.datastdv**2)
    
    return llik, deptpmod


def retr_modl_four(gdat, phas, indxphasseco, cons, deptnigh, amfo, shft, modltype):
    
    phasshft = phas + shft * np.pi / 180.
    deptpmod = amfo[0]
    modl = cons + deptnigh + 0.5 * deptpmod * np.cos(phasshft * 2. * np.pi)
    for k in range(2):
        modl += 0.5e-6 * amfo[k+1] * np.cos((k + 2) * phas * 2. * np.pi)
    for k in range(3):
        modl += 0.5e-6 * amfo[k+3] * np.sin((k + 1) * phas * 2. * np.pi) 
    deptseco = -deptpmod + deptnigh
    modl[indxphasseco] -= deptseco
    
    return modl, deptpmod


def icdf(para):
    
    icdf = gdat.limtpara[0, :] + para * (gdat.limtpara[1, :] - gdat.limtpara[0, :])

    return icdf


def retr_reso(listperi, maxmordr=10):
    
    numbplan = len(listperi)
    indxplan = np.arange(numbplan)
    ratiperi = np.zeros((numbplan, numbplan))
    reso = np.zeros((numbplan, numbplan, 2))
    for j in indxplan:
        for jj in indxplan:
            if listperi[j] > listperi[jj]:
                ratiperi[j, jj] = listperi[j] / listperi[jj]

                #print('P(.%02d) / (.%02d): %g' % (j, jj, ratiperi[j, jj]))
                minmdiff = 1e12
                for a in range(1, maxmordr):
                    for aa in range(1, maxmordr):
                        diff = abs(float(a) / aa - ratiperi[j, jj])
                        if diff < minmdiff:
                            minmdiff = diff
                            minmreso = a, aa
                reso[j, jj, :] = minmreso
                #print('minmdiff') 
                #print(minmdiff)
                #print('minmreso')
                #print(minmreso)
                #print
    
    return reso


def evol_file(gdat, namefile, pathalle, strgtype, lineadde=None):
    
    print('HACKING evol_file')
    return False

    ## read the CSV file
    pathfile = pathalle + namefile
    objtfile = open(pathfile, 'r')
    listline = []
    for line in objtfile:
        listline.append(line)
    objtfile.close()
            
    if lineadde is not None:
        numbline = len(lineadde)
        indxline = np.arange(numbline)
    
    listlineneww = []
    for k, line in enumerate(listline):
        linesplt = line.split(',')
        # delete the line if necessary
        if lineadde is not None:
            booltemp = True
            for m in indxline:
                if len(lineadde[m][0]) > 0:
                    if lineadde[m][0][-1] == '*':
                        if line[:-1].startswith(lineadde[m][0][:-1]):
                            booltemp = False
                    else:
                        if line[:-1] == lineadde[m][0]:
                            booltemp = False
            if booltemp:
                listlineneww.append(line)
        else:
            if strgtype == 'orbt' and gdat.boolallebkgdgaus:
                # from background
                for strg in gdat.objtallebkgd.posterior_params_at_maximum_likelihood:
                    if linesplt[0] == strg:
                        linesplt[1] = '%s' % gdat.objtallebkgd.posterior_params_at_maximum_likelihood[strg][0]
                        linesplt[3] = 'normal %g %g' % (np.median(gdat.objtallebkgd.posterior_params[strg]), \
                                                                                    5. * np.std(gdat.objtallebkgd.posterior_params[strg]))
            
            if strgtype == 'orbt':
                # from provided priors
                for j in gdat.indxplan:
                    for valu, strg in zip([gdat.epocprio[j], gdat.periprio[j], gdat.rratprio[j], gdat.rsmaprio[j], gdat.cosiprio[j]], \
                                                ['b_epoch', 'b_period', 'b_rr', 'b_rsuma', 'b_cosi']):
                        if linesplt[0] == strg:
                            # initial value
                            if strg == 'b_epoch' or strg == 'b_period' or strg == 'b_rr' or strg == 'b_rsuma' or strg == 'b_cosi':
                                linesplt[1] = '%f' % valu
                            if strg == 'b_epoch':
                                linesplt[3] = 'uniform %f %f' % (valu - 0.5, valu + 0.5)
                            if strg == 'b_period':
                                linesplt[3] = 'uniform %f %f' % (valu - 0.01, valu + 0.01)
                            if strg == 'b_rr':
                                linesplt[3] = 'uniform 0 %f' % (2 * valu)
                            if strg == 'b_rsuma':
                                linesplt[3] = 'uniform 0 %f' % (2 * valu)
                            if strg == 'cosi':
                                linesplt[3] = 'uniform 0 %f' % (2 * valu)
            
            if strgtype == 'pcur':
                # from the orbital run
                for j in gdat.indxplan:
                    for valu, strg in zip([gdat.epocmedi[j], gdat.perimedi[j], gdat.rratmedi[j], gdat.rsmamedi[j], gdat.cosimedi[j]], \
                                                ['b_epoch', 'b_period', 'b_rr', 'b_rsuma', 'b_cosi']):
                        if linesplt[0] == strg:
                            if strg == 'b_epoch' or strg == 'b_period' or strg == 'b_rr' or strg == 'b_rsuma' or strg == 'b_cosi':
                                # initial value
                                linesplt[1] = '%f' % valu
                                # freeze
                                linesplt[2] = '0'
            listlineneww.append(','.join(linesplt))
    
    # add the line
    if lineadde is not None:
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
        print('indx')
        print(indx)
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
    
    #print('gdat.arrylcur')
    #summgene(gdat.arrylcur)
    #print('gdat.arrypcurdetr[0][:, 0]')
    #summgene(gdat.arrypcurdetr[0][:, 0])
    #print('dilu')
    #print(dilu)
    #print('offs')
    #print(offs)
    #print('dura')
    #print(dura)
    #print('dept')
    #print(dept)
    #print('phasbegn')
    #print(phasbegn)
    #print('indxphas')
    #summgene(indxphas)
    #print('indxphasinnn')
    #summgene(indxphasinnn)
    #print('pcurmodl')
    #summgene(pcurmodl)

    return lcurmodl, []
        
        
def retr_llik_trap(gdat, para):
    
    lcurmodl, _ = retr_modl_trap(gdat, para)
    llik = -0.5 * np.sum(((gdat.arrylcurdetr[:, 1] - lcurmodl) / gdat.arrylcurdetr[:, 2])**2)
    
    #print('gdat.arrylcur[:, 1]')
    #summgene(gdat.arrylcur[:, 1])
    #print('llik')
    #summgene(llik)
    #print('')

    return llik


def get_color(color):

    if isinstance(color, tuple) and len(color) == 3: # already a tuple of RGB values
        return color

    import matplotlib.colors as mplcolors

    print('mplcolors.cnames[green]')
    print(mplcolors.cnames['green'])
    hexcolor = mplcolors.cnames['green']

    hexcolor = hexcolor.lstrip('#')
    lv = len(hexcolor)
    return tuple(int(hexcolor[i:i + lv // 3], 16)/255. for i in range(0, lv, lv // 3)) # tuple of rgb values


def retr_objtlinefade(x, y, color='black', alpha_initial=1., alpha_final=0., glow=False, **kwargs):
    
    if glow:
        glow = False
        kwargs["lw"] = 1
        fl1 = fading_line(x, y, color, alpha_initial, alpha_final, glow=False, **kwargs)
        kwargs["lw"] = 2
        alpha_initial *= 0.5
        alpha_final *= 0.5
        fl2 = fading_line(x, y, color, alpha_initial, alpha_final, glow=False, **kwargs)
        kwargs["lw"] = 6
        alpha_initial *= 0.5
        alpha_final *= 0.5
        fl3 = fading_line(x, y, color, alpha_initial, alpha_final, glow=False, **kwargs)
        return [fl3,fl2,fl1]

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
    lc = mpl.collections.LineCollection(segments, cmap=individual_cm, **kwargs)
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
            print('gdat.arrylcurdetr')
            summgene(gdat.arrylcurdetr)
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
        results = objtmodltlss.power()
        
        print('Period', format(results.period, '.5f'), 'd at T0=', results.T0)
        gdat.periprio.append(results.period)
        gdat.epocprio.append(results.T0)
        gdat.duraprio.append(results.duration)
        gdat.deptprio.append(results.depth)
        print('results.transit_times')
        print(results.transit_times)
        print(len(results.transit_times), 'transit times in time series:', ['{0:0.5f}'.format(i) for i in results.transit_times])
        print('Number of data points during each unique transit', results.per_transit_count)
        print('The number of transits with intransit data points', results.distinct_transit_count)
        print('The number of transits with no intransit data points', results.empty_transit_count)
        print('Transit depth', format(results.depth, '.5f'), '(at the transit bottom)')
        print('Transit duration (days)', format(results.duration, '.5f'))
        print('Transit depths (mean)', results.transit_depths)
        print('Transit depth uncertainties', results.transit_depths_uncertainties)
        print('FAP: %g' % results.FAP) 
        # plot TLS power spectrum
        figr, axis = plt.subplots(figsize=(6, 4))
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
        figr, axis = plt.subplots(figsize=(6, 4))
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
        figr, axis = plt.subplots(figsize=(6, 4))
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
            if results.SDE < 7.1:
                break
        else:
            if j == gdat.numbplan:
                break
        j += 1

    gdat.epocprio = np.array(gdat.epocprio)
    gdat.periprio = np.array(gdat.periprio)
    gdat.deptprio = np.array(gdat.deptprio)
    gdat.duraprio = np.array(gdat.duraprio)
            

def main( \
         strgtarg=None, \
         ticitarg=None, \
         strgmast=None, \
         strgtoii=None, \
         labltarg=None, \
         
         boolphascurv=False, \
         
         # type of light curve to be used for analysis
         datatype=None, \
                    
         # Boolean flag to use SAP instead of PDC by default, when SPOC data is being used.
         boolsapp=False, \
         
         boolexar=None, \
        
         boolinfetlss=False, \
         
         maxmnumbstartcat=1, \
        
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

         ## baseline
         detrtype='spln', \
         weigsplndetr=1., \
         durakerndetrmedi=1., \
         
         # type of priors used for allesfitter
         priotype=None, \

         boolallebkgdgaus=False, \
        
         liststrgplan=['b'], \

         # Boolean flag to perform TLS
         booltlss=True, \

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
    gdat.pathobjt = gdat.pathbase + '%s/' % gdat.strgtarg
    gdat.pathdata = gdat.pathobjt + 'data/'
    gdat.pathimag = gdat.pathobjt + 'imag/'
    
    os.system('mkdir -p %s' % gdat.pathdata)
    os.system('mkdir -p %s' % gdat.pathimag)

    print('TESS TOI/allesfitter pipeline initialized at %s...' % gdat.strgtimestmp)
    
    # plotting
    gdat.strgplotextn = 'png'

    # conversion factors
    gdat.factmjme = 317.907
    gdat.factmsmj = 1048.
    gdat.factrjre = 11.2
    gdat.factrsrj = 9.95
    
    if gdat.strgmast is None:
        raise Exception('')
    print('gdat.strgmast')
    print(gdat.strgmast)
    dictexarcomp = retr_exarcomp(gdat, strgtarg=gdat.strgmast)
    gdat.boolexar = dictexarcomp is not None
    
    print('dictexarcomp')
    print(dictexarcomp)
    if dictexarcomp is not None:
        # stellar properties
        
        gdat.priotype = 'exar'
        gdat.periprio = dictexarcomp['peri']
        gdat.radiplanprio = dictexarcomp['radiplan']
        gdat.radistarprio = dictexarcomp['radistar']
        gdat.smaxprio = dictexarcomp['smax']
        gdat.massplanprio = dictexarcomp['massplan']
        gdat.massstarprio = dictexarcomp['massstar']
    
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
    
    if gdat.strgtoii is not None:
        
        gdat.priotype = 'exof'
        print('A TOI number is provided. Retreiving the TCE attributes from ExoFOP-TESS...')
        # read ExoFOP-TESS
        path = gdat.pathbase + 'data/exofop_toilists.csv'
        print('Reading from %s...' % path)
        NASA_Arc = pd.read_csv(path, skiprows=0)
        indx = []
        for k, strg in enumerate(NASA_Arc['TOI']):
            for strgtoii in gdat.strgtoii:
                if str(strg) == strgtoii:
                    indx.append(k)
        indx = np.array(indx)
        if indx.size == 0:
            print('Did not find the TOI iin the ExoFOP-TESS TOI list.')
            raise Exception('')
        
        gdat.epocprio = NASA_Arc['Epoch (BJD)'].values[indx]
        gdat.periprio = NASA_Arc['Period (days)'].values[indx]
        gdat.deptprio = NASA_Arc['Depth (ppm)'].values[indx] * 1e-6
        gdat.radiplanprio = NASA_Arc['Planet Radius (R_Earth)'].values[indx] / gdat.factrjre # [R_J]
        gdat.duraprio = NASA_Arc['Duration (hours)'].values[indx] / 24. # [days]
        gdat.radistar = NASA_Arc['Stellar Radius (R_Sun)'].values[indx][0] * gdat.factrsrj # [R_J]
        gdat.tmptplanprio = NASA_Arc['Planet Equil Temp (K)'].values[indx] # [K]
        gdat.tmptstar = NASA_Arc['Stellar Eff Temp (K)'].values[indx] # [K]
        gdat.cosiprio = np.zeros_like(gdat.epocprio)
    if gdat.priotype == None:
        if gdat.epocprio is None:
            gdat.priotype = 'tlss'
        else:
            gdat.priotype = 'inpt'
    
    gdat.numbplan = None


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
        gdat.cosiprio = 0.

    #if gdat.smaxprio is None:
    #    gdat.smaxprio = tesstarg.util.retr_smaxkepl(gdat.periprio, gdat.massstar) * 2093. # [R_J]
    
    gdat.rratprio = np.sqrt(gdat.deptprio)
    
    if gdat.priotype == 'exof' or gdat.priotype == 'tlss' or gdat.priotype == 'inpt':
        gdat.radiplanprio = gdat.rratprio * gdat.radistar
        gdat.inclprio = np.arccos(gdat.cosiprio) * 180. / np.pi
   
    if gdat.priotype == 'exar' or gdat.priotype == 'exof' or gdat.priotype == 'tlss':
        print('gdat.duraprio')
        print(gdat.duraprio)
        print('gdat.cosiprio')
        print(gdat.cosiprio)
        print('gdat.periprio')
        print(gdat.periprio)
        gdat.rsmaprio = np.sqrt(np.sin(np.pi * gdat.duraprio / gdat.periprio)**2 + gdat.cosiprio**2)
        #gdat.duraprio = gdat.periprio / np.pi * np.arcsin(np.sqrt(gdat.rsmaprio**2 - gdat.cosiprio**2))
    
    if gdat.priotype == 'exof' or gdat.priotype == 'tlss' or gdat.priotype == 'inpt':
        gdat.smaxprio = (gdat.radistar + gdat.radiplanprio) / gdat.rsmaprio
    
    gdat.massplanprio = tesstarg.util.retr_massfromradi(gdat.radiplanprio)
    
    # temp
    print('HACKING for TOI 1233')
    gdat.tmptplanprio = 11871. / np.sqrt(gdat.smaxprio)
    print('gdat.epocprio')
    print(gdat.epocprio)
    print('gdat.periprio')
    print(gdat.periprio)
    print('gdat.radiplanprio')
    print(gdat.radiplanprio)
    print('gdat.radistar')
    print(gdat.radistar)
    print('gdat.massstar')
    print(gdat.massstar)
    print('gdat.smaxprio')
    print(gdat.smaxprio)
    print('gdat.tmptplanprio')
    print(gdat.tmptplanprio)
    print('gdat.tmptplanprio * np.sqrt(gdat.smaxprio)')
    print(gdat.tmptplanprio * np.sqrt(gdat.smaxprio))
    print('gdat.rratprio')
    print(gdat.rratprio)
    print('gdat.rsmaprio')
    print(gdat.rsmaprio)
    print('gdat.cosiprio')
    print(gdat.cosiprio)
    print('gdat.duraprio')
    print(gdat.duraprio)
    #fracmass = gdat.massplanprio / gdat.massstar
    #print('fracmass')
    #print(fracmass)
    
    if gdat.booldiagmode:
        if not np.isfinite(gdat.duraprio).all():
            raise Exception('')
    
    gdat.ecceprio = 0.
    print('gdat.ecceprio')
    print(gdat.ecceprio)

    rvsa = tesstarg.util.retr_rvsa(gdat.periprio, 13.8, gdat.massstar, gdat.inclprio, gdat.ecceprio)
    print('RV semi-amplitude assuming a 13.8 Msun companion [m/s]:')
    print(rvsa)


    # settings
    gdat.numbplan = gdat.epocprio.size
    gdat.indxplan = np.arange(gdat.numbplan)

    gdat.listcolrplan = ['r', 'g', 'y', 'c', 'm']
    gdat.listlablplan = ['04', '03', '01', '02']
    ## plotting
    gdat.figrsize = np.empty((5, 2))
    gdat.figrsize[0, :] = np.array([12., 4.])
    gdat.figrsize[1, :] = np.array([12., 6.])
    gdat.figrsize[2, :] = np.array([12., 10.])
    gdat.figrsize[3, :] = np.array([12., 14.])
    gdat.figrsize[4, :] = np.array([6., 6.])
    boolpost = False
    if boolpost:
        gdat.figrsize /= 1.5
    
    if gdat.liststrgplan is None:
        gdat.liststrgplan = ['b', 'c', 'd', 'e']
    gdat.listcolrplan = ['g', 'r', 'c', 'm']
    
    gdat.timetess = 2457000.
    
    gdat.indxplan = np.arange(gdat.numbplan)
    
    # check the TIC ID
    print('gdat.strgmast')
    print(gdat.strgmast)
    print
    if gdat.strgmast is None:
        gdat.strgmast = 'TIC %d' % gdat.ticitarg

    catalogData = astroquery.mast.Catalogs.query_object(gdat.strgmast, catalog='TIC', radius='5m')
    rasc = catalogData[0]['ra']
    decl = catalogData[0]['dec']
    gdat.tmptstar = catalogData[0]['Teff']
    gdat.jmag = catalogData[0]['Jmag']
    print('rasc')
    print(rasc)
    print('decl')
    print(decl)
    strgtici = '%s' % catalogData[0]['ID']
    if strgtici != str(gdat.ticitarg):
        raise Exception('')
    
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
    
    print('gdat.epocprio')
    print(gdat.epocprio)
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
        figr, axis = plt.subplots(1, 1, figsize=gdat.figrsize[1, :])
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
        
    
    if gdat.makeprioplot:
        # plot all exoplanets and overplot this one
        dictexarcomp = retr_exarcomp(gdat)
        
        numbplanexar = dictexarcomp['radiplan'].size
        indxplanexar = np.arange(numbplanexar)

        # mass radius
        figr, axis = plt.subplots(figsize=(6, 4))
        print('dictexarcomp[massplan]')
        summgene(dictexarcomp['massplan'])
        print('dictexarcomp[massplan][where(np.isfinite(dictexarcomp[massplan]))]')
        summgene(dictexarcomp['massplan'][np.where(np.isfinite(dictexarcomp['massplan']))])
        print('dictexarcomp[stdvmassplan]')
        summgene(dictexarcomp['stdvmassplan'])
        print('dictexarcomp[stdvmassplan][where(np.isfinite(dictexarcomp[massplan]))]')
        summgene(dictexarcomp['stdvmassplan'][np.where(np.isfinite(dictexarcomp['stdvmassplan']))])
        indx = np.where(np.isfinite(dictexarcomp['stdvmassplan']) & np.isfinite(dictexarcomp['stdvradiplan']))[0]
        axis.scatter(dictexarcomp['massplan'][indx] * 317.8, dictexarcomp['radiplan'][indx] * gdat.factrjre, color='grey', alpha=0.4)
        gdat.listlabldenscomp = ['H$_2$O']
        listdenscomp = [1.]
        gdat.listcolrdenscomp = ['b']
        gdat.numbdenscomp = len(gdat.listlabldenscomp)
        gdat.indxdenscomp = np.arange(gdat.numbdenscomp)
        masscompdens = np.linspace(0.5, 16.) # M_E
        for i in gdat.indxdenscomp:
            radicompdens = (masscompdens / listdenscomp[i])**(1. / 3.)
            axis.plot(masscompdens, radicompdens, color=gdat.listcolrdenscomp[i])
        for i in gdat.indxdenscomp:
            axis.text(0.3, 1., gdat.listlabldenscomp[i])
        for j in gdat.indxplan:
            print('gdat.radiplanprio')
            print(gdat.radiplanprio)
            print('gdat.massplanprio')
            print(gdat.massplanprio)
            axis.scatter(gdat.massplanprio[j] * 317.8, gdat.radiplanprio[j] * gdat.factrjre, color=gdat.listcolrplan[j])
        axis.set_xlim([0., 16])
        axis.set_ylim([0., 4.])
        axis.set_ylabel(r'Radius [$R_E$]')
        axis.set_xlabel(r'Mass [$M_E$]')
        plt.subplots_adjust()
        path = gdat.pathimag + 'massradii.%s' % gdat.strgplotextn
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()

        # period ratios
        figr, axis = plt.subplots(figsize=(6, 4))
        ## all 
        listratiperi = []
        liststrgstarcomp = []
        for m in indxplanexar:
            strgstar = dictexarcomp['namestar'][m]
            if not strgstar in liststrgstarcomp:
                indxexarstar = np.where(dictexarcomp['namestar'] == strgstar)[0]
                if indxexarstar[0] != m:
                    raise Exception('')
                if dictexarcomp['nameplan'][m][-1] != 'b':
                    print('dictexarcomp[nameplan][m]')
                    print(dictexarcomp['nameplan'][m])
                #    raise Exception('')
                for k in indxexarstar:
                    for kk in indxexarstar:
                        if dictexarcomp['peri'][k] > dictexarcomp['peri'][kk]:
                            ratiperi = dictexarcomp['peri'][k] / dictexarcomp['peri'][kk]
                            listratiperi.append(ratiperi)
                liststrgstarcomp.append(strgstar)
        listratiperi = np.array(listratiperi)
        axis.hist(listratiperi, 100)
        
        ## this system
        for j in gdat.indxplan:
            for jj in gdat.indxplan:
                if gdat.periprio[j] > gdat.periprio[jj]:
                    ratiperi = gdat.periprio[j] / gdat.periprio[jj]
                    axis.axvline(ratiperi, color=gdat.listcolrplan[j])
        ## resonances
        peritemp = np.linspace(1., 10.)
        for perifrst in peritemp:
            for periseco in peritemp:
                if perifrst > periseco:
                    axis.axvline(ratiperi, color='k')
        axis.set_xlabel('Period ratio')
        plt.subplots_adjust()
        path = gdat.pathimag + 'ratiperi.%s' % gdat.strgplotextn
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()

        # radius histogram
        for a in range(2): 
            figr, axis = plt.subplots(figsize=(6, 4))
            
            if a == 0:
                bins = np.linspace(1., 4., 100)
                strgzoom = '_rb14'
            else:
                bins = 100
                strgzoom = ''
            temp = dictexarcomp['radiplan'] * gdat.factrjre
            indx = np.where(np.isfinite(temp))[0]
            axis.hist(temp[indx], bins=bins, color='k')
            for j in gdat.indxplan:
                axis.axvline(gdat.factrjre * gdat.radiplanprio[j], color=gdat.listcolrplan[j], ls='--', label=gdat.liststrgplan[j])
            axis.set_xlabel('Radius [$R_E$]')
            axis.set_ylabel('N')
            plt.subplots_adjust()
            path = gdat.pathimag + 'histradi%s.%s' % (strgzoom, gdat.strgplotextn)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
   
            # radius vs period
            figr, axis = plt.subplots(figsize=(12, 12))
            axis.scatter(dictexarcomp['peri'][indx], temp[indx])
            axis.scatter(gdat.periprio, gdat.factrjre * gdat.radiplanprio, color=gdat.listcolrplan[j])
            axis.set_xlabel('Period [days]')
            axis.set_ylabel('Radius [$R_E$]')
            if a == 0:
                axis.set_ylim([1., 4.])
            path = gdat.pathimag + 'radiperi%s.%s' % (strgzoom, gdat.strgplotextn)
            plt.savefig(path)
            plt.close()
        

        
        esmm = tesstarg.util.retr_esmm(dictexarcomp['tmptplan'], dictexarcomp['tmptstar'], dictexarcomp['radiplan'], dictexarcomp['radistar'], \
                                                                                                            dictexarcomp['kmag'])
        
        gdat.tsmmprio = np.empty(gdat.numbplan)
        gdat.esmmprio = np.empty(gdat.numbplan)
        for j in gdat.indxplan:
            print('gdat.radiplanprio[j]')
            print(gdat.radiplanprio[j])
            print('gdat.tmptplanprio[j]')
            print(gdat.tmptplanprio[j])
            print('gdat.massplanprio[j]')
            print( gdat.massplanprio[j])
            print('gdat.radistar')
            print(gdat.radistar)
            print('gdat.jmag')
            print(gdat.jmag)
            gdat.tsmmprio[j] = tesstarg.util.retr_tsmm(gdat.radiplanprio[j], gdat.tmptplanprio[j], gdat.massplanprio[j], gdat.radistar, gdat.jmag)
            if not np.isfinite(gdat.tsmmprio[j]):
                raise Exception('')
            print('gdat.tmptstar')
            print(gdat.tmptstar)
            print('gdat.tmptplanprio[j]')
            print(gdat.tmptplanprio[j])
            if np.isfinite(gdat.tmptstar):
                gdat.esmmprio[j] = tesstarg.util.retr_esmm(gdat.tmptplanprio[j], gdat.tmptstar, gdat.radiplanprio[j], gdat.radistar, gdat.jmag)
                if not np.isfinite(gdat.esmmprio[j]):
                    raise Exception('')
        print('gdat.tsmmprio')
        print(gdat.tsmmprio)
        tsmm = tesstarg.util.retr_tsmm(dictexarcomp['radiplan'], dictexarcomp['tmptplan'], dictexarcomp['massplan'], dictexarcomp['radistar'], \
                                                                                                            dictexarcomp['jmag'])
        
        tmptplan = dictexarcomp['tmptplan']
        radiplan = dictexarcomp['radiplan'] * gdat.factrjre # R_E
        
        numbtext = 10
        liststrgmetr = ['tsmm']
        if np.isfinite(gdat.tmptstar):
            liststrgmetr.append('esmm')
        liststrgzoom = ['allr', 'rb24']
        liststrglimt = ['allm', 'gmas']
        for strgmetr in liststrgmetr:
            
            if strgmetr == 'tsmm':
                metr = np.log(tsmm)
                lablmetr = 'TSM'
                metrthis = np.log(gdat.tsmmprio)
            else:
                metr = np.log(esmm)
                lablmetr = 'ESM'
                metrthis = np.log(gdat.esmmprio)
            
            for strgzoom in liststrgzoom:
                if strgzoom == 'allr':
                    indxzoom = np.where(np.isfinite(radiplan) & np.isfinite(tmptplan) \
                                                                                & np.isfinite(metr) & (dictexarcomp['booltran'] == 1.))[0]
                if strgzoom == 'rb24':
                    indxzoom = np.where((radiplan < 4) & (radiplan > 1.5) & np.isfinite(radiplan) & np.isfinite(tmptplan) \
                                                                                & np.isfinite(metr) & (dictexarcomp['booltran'] == 1.))[0]
        
                for strglimt in liststrglimt:
                    if strglimt == 'gmas':
                        indxlimt = np.where(dictexarcomp['stdvmassplan'] / dictexarcomp['massplan'] < 0.3)[0]
                    if strglimt == 'allm':
                        indxlimt = np.arange(esmm.size)
                
                    indx = np.intersect1d(indxzoom, indxlimt)
                    
                    # normalize
                    
                    print('metrthis')
                    print(metrthis)
                    print('indx')
                    summgene(indx)
                    print('metr[indx]')
                    summgene(metr[indx])
                    fact = 100. / max(np.amax(metrthis), np.amax(metr[indx]))
                    metr *= fact
                    metrthis *= fact
                    print('gdat.tsmmprio[j]*fact') 
                    print(gdat.tsmmprio[j]*fact)
                    print('')
                    # sort
                    indxsort = np.argsort(metr[indx])[::-1]
                    
                    figr, axis = plt.subplots(figsize=(12, 6))
                    for j in gdat.indxplan:
                        axis.scatter(gdat.radiplanprio[j] * gdat.factrjre, gdat.tmptplanprio[j], color=gdat.listcolrplan[j], s=gdat.tsmmprio[j]*fact)
                        axis.text(gdat.radiplanprio[j] * gdat.factrjre * 1.1, gdat.tmptplanprio[j] * 1.1, \
                                                         'HD 108236 %s' % gdat.liststrgplan[j], color=gdat.listcolrplan[j], ha='center', va='center')
                    axis.scatter(radiplan[indx], tmptplan[indx], s=metr[indx])
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
            
                    figr, axis = plt.subplots(figsize=(12, 6))
                    axis.scatter(radiplan[indx], metr[indx], s=metr[indx])
                    for k in indxsort[:numbtext]:
                        axis.text(radiplan[indx[k]], metr[indx[k]], '%s' % dictexarcomp['nameplan'][indx[k]], ha='center', va='center')
                    for j in gdat.indxplan:
                        axis.scatter(gdat.radiplanprio[j] * gdat.factrjre, gdat.tsmmprio[j]*fact, color=gdat.listcolrplan[j], s=gdat.tsmmprio[j]*fact)
                        axis.text(gdat.radiplanprio[j] * gdat.factrjre * 1.1, gdat.tsmmprio[j] * fact * 1.1, 'HD 108236 %s' % gdat.liststrgplan[j], \
                                                                                                color=gdat.listcolrplan[j], ha='center', va='center')
                    axis.set_ylabel(lablmetr)
                    axis.set_xlabel('Radius [$R_E$]')
                    #axis.set_yscale('log')
                    plt.tight_layout()
                    path = gdat.pathimag + '%s_radiplan_%s_%s_targ.%s' % (strgmetr, strgzoom, strglimt, gdat.strgplotextn)
                    print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()
        

                    print('')

        
        gdat.boolplotspec = False
        
        print('gdat.datatype')
        print(gdat.datatype)
        if gdat.boolplotspec:
            ## TESS throughput 
            gdat.data = np.loadtxt(pathdata + 'band.csv', delimiter=',', skiprows=9)
            gdat.meanwlenband = gdat.data[:, 0] * 1e-3
            gdat.thptband = gdat.data[:, 1]
        
        if gdat.datatype == 'spoc':
            # plot PDCSAP and SAP light curves
            figr, axis = plt.subplots(2, 1, figsize=gdat.figrsize[1, :])
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
        
        figr, axis = plt.subplots(1, 1, figsize=gdat.figrsize[1, :])
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
        plt.subplots_adjust(hspace=0.)
        path = gdat.pathimag + 'lcur.%s' % (gdat.strgplotextn)
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()
        
        fact = 1000000.
        # plot the orbit
        figr, axis = plt.subplots(figsize=(8, 4))
        # star 1
        w1 = mpl.patches.Wedge((0, 0), gdat.radistar/2093.*6, 270, 360, fc='k', zorder=1, edgecolor='k')
        axis.add_artist(w1)
            
        for j in gdat.indxplan:
            colrredd = gdat.tmptplanprio[j, None]
            colrblue = 1. - colrredd
            colr = np.zeros((4, 1))
            colr[1, 0] = colrredd
            colr[2, 0] = colrblue
            #colr = colr.T
            size = gdat.radiplanprio[j] * 5.
            
            # orbit ellipse
            #axis.add_patch(mpl.patches.Ellipse(xy=(0, 0), width=2*gdat.smaxprio[j]/2093., \
            #                               height=0.2*gdat.smaxprio[j]/2093., edgecolor=gdat.listcolrplan[j], fc='None', lw=2, alpha=0.3, zorder=2))
            

            yposelli = 0.2*gdat.smaxprio[j]/2093. * np.linspace(-1., 1., 100)
            xposelli = np.sqrt(1. - (yposelli / 0.2*gdat.smaxprio[j]/2093.)**2)
            objt = retr_objtlinefade(xposelli, yposelli, color=gdat.listcolrplan[j], alpha_initial=1., alpha_final=0., glow=False)
            #axis.add_patch(objt)
            axis.add_collection(objt)

            # temperature contours?
            #for tmpt in [500., 700,]:
            #    smaj = tmpt
            #    axis.axvline(smaj, ls='--')
            
            # planet
            axis.scatter(gdat.smaxprio[j]/2093., 0, marker='o', s=gdat.radiplanprio[j]/2093.*fact, color=gdat.listcolrplan[j], zorder=3)
            
        axis.scatter(0.387, 0, marker='o', s=0.0349/2093.*fact, color='grey', zorder=3)





        ##axistwin.set_xlim(axis.get_xlim())
        #xpostemp = axistwin.get_xticks()
        #print('xpostemp')
        #print(xpostemp)
        ##axistwin.set_xticks(xpostemp[1:])
        #axistwin.set_xticklabels(['%f' % tmpt for tmpt in listtmpt])
        
        axis.get_yaxis().set_visible(False)
        axis.axis('equal')

        # star 2
        w1 = mpl.patches.Wedge((0, 0), gdat.radistar/2093.*6, 0, 90, fc='k', zorder=4, edgecolor='k')
        axis.add_artist(w1)
        
        axis.set_xlim([0, None])
        
        #axistwin = axis.twiny()
            
        # centerline
        axis.set_xlabel('Distance from the star [AU]')
        plt.subplots_adjust()
        #axis.legend()
        path = gdat.pathimag + 'orbt.%s' % gdat.strgplotextn
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()
   
        if gdat.numbsect > 1:
            for o in gdat.indxsect:
                figr, axis = plt.subplots(1, 1, figsize=gdat.figrsize[1, :])
                
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
                
                plt.subplots_adjust(hspace=0.)
                path = gdat.pathimag + 'lcursc%02d.%s' % (gdat.listisec[o], gdat.strgplotextn)
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()
        
        if not np.isfinite(gdat.arrylcur).all():
            raise Exception('')
        
        if gdat.booldiagmode:
            for a in range(gdat.arrylcur[:, 0].size):
                if a != gdat.arrylcur[:, 0].size - 1 and gdat.arrylcur[a, 0] >= gdat.arrylcur[a+1, 0]:
                    raise Exception('')
        
        if gdat.booldetr:
            # plot detrending
            figr, axis = plt.subplots(2, 1, figsize=gdat.figrsize[1, :])
            for i in indxsplnregi:
                # plot the masked and detrended light curves
                indxtimetemp = indxtimeregi[i]
                axis[0].plot(gdat.arrylcur[indxtimetemp, 0] - gdat.timetess, gdat.arrylcur[indxtimetemp, 1], marker='o', ls='', ms=1, color='grey')
                indxtimetemp = indxtimeregi[i][indxtimeregioutt[i]]
                axis[0].plot(gdat.arrylcur[indxtimetemp, 0] - gdat.timetess, gdat.arrylcur[indxtimetemp, 1], marker='o', ls='', ms=1, color='k')
                
                if listobjtspln[i] != []:
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
            
            # produce a table for the spline coefficients
            fileoutp = open(gdat.pathdata + 'coef.csv', 'w')
            fileoutp.write(' & ')
            for j in indxsplnregi:
                print('$\beta$:', listobjtspln[i].get_coeffs())
                print('$t_k$:', listobjtspln[i].get_knots())
                print
            fileoutp.write('\\hline\n')
            fileoutp.close()
        
        if gdat.booltlss:
            exec_tlss(gdat)

        if gdat.booltlss and gdat.boolinfetlss:
            gdat.epocinfe = gdat.epocprio
            gdat.periinfe = gdat.periprio
            gdat.rratinfe = np.sqrt(gdat.deptprio)
            gdat.rsmainfe = np.sin(np.pi * gdat.duraprio / gdat.periprio)
            gdat.cosiinfe = 0.
        else:
            gdat.epocinfe = gdat.epocprio 
            gdat.periinfe = gdat.periprio
            gdat.rsmainfe = gdat.rsmaprio
            gdat.rratinfe = gdat.rratprio
            gdat.cosiinfe = gdat.cosiprio
            gdat.durainfe = gdat.duraprio
    
        
        # plot individual phase curves
        for j in gdat.indxplan:
            # phase on the horizontal axis
            figr, axis = plt.subplots(1, 1, figsize=gdat.figrsize[1, :])
            axis.plot(gdat.arrypcurdetr[j][:, 0], gdat.arrypcurdetr[j][:, 1], color='grey', alpha=0.3, marker='o', ls='', ms=1)
            axis.plot(gdat.arrypcurdetrbind[j][:, 0], gdat.arrypcurdetrbind[j][:, 1], color=gdat.listcolrplan[j], marker='o', ls='', ms=4)
            #axis.text(0.47, np.percentile(gdat.arrypcurdetr[j][:, 1], 99), r'\textbf{%s}' % gdat.liststrgplan[j], \
            #                                                                                    color=gdat.listcolrplan[j], va='center', ha='center')
            axis.text(0.9, 0.9, \
                                        r'\textbf{%s}' % gdat.liststrgplan[j], color=gdat.listcolrplan[j], va='center', ha='center', transform=axis.transAxes)
            axis.set_ylabel('Relative Flux')
            axis.set_xlabel('Phase')
            path = gdat.pathimag + 'pcurphasplan%04d.%s' % (j + 1, gdat.strgplotextn)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
        
            # time on the horizontal axis
            figr, axis = plt.subplots(1, 1, figsize=gdat.figrsize[1, :])
            axis.plot(gdat.periinfe[j] * gdat.arrypcurdetr[j][:, 0] * 24., gdat.arrypcurdetr[j][:, 1], color='grey', alpha=0.3, marker='o', ls='', ms=1)
            axis.plot(gdat.periinfe[j] * gdat.arrypcurdetrbind[j][:, 0] * 24., gdat.arrypcurdetrbind[j][:, 1], \
                                                                                color=gdat.listcolrplan[j], marker='o', ls='', ms=4)
            axis.text(0.9, 0.9, \
                                        r'\textbf{%s}' % gdat.liststrgplan[j], color=gdat.listcolrplan[j], va='center', ha='center', transform=axis.transAxes)
            axis.set_ylabel('Relative Flux')
            axis.set_xlabel('Time [hours]')
            axis.set_xlim([-2 * gdat.durainfe[j] * 24., 2 * gdat.durainfe[j] * 24.])
            path = gdat.pathimag + 'pcurtimeplan%04d.%s' % (j + 1, gdat.strgplotextn)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
        
        # plot all phase curves
        if gdat.numbplan > 1:
            figr, axis = plt.subplots(gdat.numbplan, 1, figsize=gdat.figrsize[1, :])
            if gdat.numbplan == 1:
                axis = [axis]
            for j in gdat.indxplan:
                axis[j].plot(gdat.arrypcurdetr[j][:, 0], gdat.arrypcurdetr[j][:, 1], \
                                                                                                color='grey', alpha=0.3, marker='o', ls='', ms=1)
                axis[j].plot(gdat.arrypcurdetrbind[j][:, 0], gdat.arrypcurdetrbind[j][:, 1], \
                                                                                                color=gdat.listcolrplan[j], marker='o', ls='', ms=4)
                axis[j].text(0.47, np.percentile(gdat.arrypcurdetr[j][:, 1], 99.9), r'\textbf{%s}' % gdat.liststrgplan[j], \
                                                                                        color=gdat.listcolrplan[j], va='center', ha='center')
                axis[j].minorticks_on()
                axis[j].set_ylabel('Relative Flux')
            axis[0].set_xlabel('Phase')
            plt.subplots_adjust(hspace=0.)
            path = gdat.pathimag + 'pcur.%s' % gdat.strgplotextn
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
    

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
        if gdat.boolphascurv:
            gdat.liststrgalle += ['pcur']

        for strg in ['params.csv', 'settings.csv', 'params_star.csv']:
            for strgalle in gdat.liststrgalle:
                pathinit = '%sdata/allesfit_templates/%s/%s' % (gdat.pathbase, strgalle, strg)
                pathfinl = '%sallesfits/allesfit_%s/%s' % (gdat.pathobjt, strgalle, strg)
                if not os.path.exists(pathfinl):
                    os.system('cp %s %s' % (pathinit, pathfinl))
   
        if gdat.boolallebkgdgaus:
            # background allesfitter run
            print('Setting up the background allesfitter run...')
            
            evol_file(gdat, 'params.csv', gdat.pathallebkgd, 'bkgd')
            
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
        
        evol_file(gdat, 'params.csv', gdat.pathalleorbt, 'orbt')
        
        # update the transit duration for fastfit
        lineadde = [ \
                    ['fast_fit_width*', 'fast_fit_width,%.3g\n' % np.amax(gdat.duramask)], \
                   ]
        linetemp = 'companions_phot,'
        for j in gdat.indxplan:
            if j != 0:
                linetemp += ' '
            linetemp += '%s' % gdat.liststrgplan[j]
        linetemp += '\n'
        lineadde.extend([ \
                        ['companions_phot,b', linetemp] \
                        ])
        evol_file(gdat, 'settings.csv', gdat.pathalleorbt, 'none', lineadde=lineadde)
        
        lineadde = []
        for j in gdat.indxplan:
            if gdat.liststrgplan[j] == 'b':
                continue
            lineadde.extend([ \
                        [[], '%s_rr,%f,1,uniform %f %f,$R_{%s} / R_\star$,\n' % \
                                            (gdat.liststrgplan[j], gdat.rratinfe[j], 0, 2 * gdat.rratinfe[j], gdat.liststrgplan[j])], \
                        [[], '%s_rsuma,%f,1,uniform %f %f,$(R_\star + R_{%s}) / a_{%s}$,\n' % \
                                            (gdat.liststrgplan[j], gdat.rsmainfe[j], 0, 2 * gdat.rsmainfe[j], gdat.liststrgplan[j], gdat.liststrgplan[j])], \
                        [[], '%s_epoch,%f,1,uniform %f %f,$\cos{i_{%s}}$,\n' % \
                                            (gdat.liststrgplan[j], gdat.cosiinfe[j], 0, gdat.cosiinfe[j] * 2, gdat.liststrgplan[j])], \
                        [[], '%s_period,%f,1,uniform %f %f,$T_{0;%s}$,$\mathrm{BJD}$\n' % \
                                        (gdat.liststrgplan[j], gdat.epocinfe[j], gdat.epocinfe[j] - 0.5, gdat.epocinfe[j] + 0.5, gdat.liststrgplan[j])], \
                        [[], '%s_cosi,%f,1,uniform %f %f,$P_{%s}$,$\mathrm{d}$\n' % \
                                        (gdat.liststrgplan[j], gdat.periinfe[j], gdat.periinfe[j] - 0.01, gdat.periinfe[j] + 0.01, gdat.liststrgplan[j])], \
                       ])
        evol_file(gdat, 'params.csv', gdat.pathalleorbt, 'none', lineadde=lineadde)
            
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
        companion = 'b'
        strginst = 'TESS'
        print('Reading from %s...' % gdat.pathalleorbt)
        alles = allesfitter.allesclass(gdat.pathalleorbt)
        allesfitter.config.init(gdat.pathalleorbt)
        
        numbsamp = alles.posterior_params[list(alles.posterior_params.keys())[0]].size

        gdat.epocmedi = np.empty(gdat.numbplan)
        gdat.perimedi = np.empty(gdat.numbplan)
        gdat.rratmedi = np.empty(gdat.numbplan)
        gdat.rsmamedi = np.empty(gdat.numbplan)
        gdat.cosimedi = np.empty(gdat.numbplan)
        for j in gdat.indxplan:
            gdat.epocmedi[j] = np.median(alles.posterior_params['%s_epoch' % gdat.liststrgplan[j]])
            gdat.perimedi[j] = np.median(alles.posterior_params['%s_period' % gdat.liststrgplan[j]])
            gdat.rratmedi[j] = np.median(alles.posterior_params['%s_rr' % gdat.liststrgplan[j]])
            gdat.rsmamedi[j] = np.median(alles.posterior_params['%s_rsuma' % gdat.liststrgplan[j]])
            gdat.cosimedi[j] = np.median(alles.posterior_params['%s_cosi' % gdat.liststrgplan[j]])
        if 'b_rr' in alles.posterior_params.keys():
            listfracradi = alles.posterior_params['b_rr']
        else:
            listfracradi = np.zeros(numbsamp) + allesfitter.config.BASEMENT.params['b_rr']
        
        if 'b_rsuma' in alles.posterior_params.keys():
            listrsma = alles.posterior_params['b_rsuma']
        else:
            listrsma = np.zeros(numbsamp) + allesfitter.config.BASEMENT.params['b_rsuma']
        
        listfracradi = listfracradi[int(numbsamp/4):]
        listrsma = listrsma[int(numbsamp/4):]
        
        if gdat.boolphascurv:
            
            gdat.pathallepcur = gdat.pathalle + 'allesfit_pcur/'
            cmnd = 'mkdir -p %s' % gdat.pathallepcur
            os.system(cmnd)
            
            evol_file(gdat, 'params.csv', gdat.pathallepcur, 'pcur')
        
            path = gdat.pathallepcur + 'TESS.csv'
            print('Writing to %s...' % path)
            np.savetxt(path, gdat.arrylcurdetr, delimiter=',', header='time,flux,flux_err')
        
            # update the transit duration for fastfit
            lineadde = [['fast_fit,*', 'fast_fit,False']]
            evol_file(gdat, 'settings.csv', gdat.pathallepcur, 'none', lineadde=lineadde)
        
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
                
                prnt_list(listphasshft, 'Phase shift')
                return

            else:
                listalbgalle = alles.posterior_params['b_geom_albedo_TESS']
                listdeptnigh = alles.posterior_params['b_sbratio_TESS'] * listfracradi**2
        
            listalbgalle = listalbgalle[int(numbsamp/4):]
            listdeptnigh = listdeptnigh[int(numbsamp/4):]
        
            # calculate nightside, secondary and planetary modulation from allesfitter output
            ## what allesfitter calls 'geometric albedo' is not the actual geometric albedo
            listdeptpmod = listalbgalle * (listfracradi * listrsma / (1. + listfracradi))**2
            print('listalbgalle')
            #summgene(listalbgalle)
            print('listfracradi')
            #summgene(listfracradi)
            print('listrsma')
            #summgene(listrsma)
            listsamp = np.empty((numbsamp, 6))
            listsamp[:, 0] = listdeptnigh
            listsamp[:, 1] = listdeptnigh + listdeptpmod
            listsamp[:, 2] = listdeptpmod
            listsamp[:, 3] = listdeptpmod * np.random.rand(listdeptpmod.size)
            listsamp[:, 4] = listdeptpmod - listsamp[:, 3]
            listsamp[:, 5] = listsamp[:, 4] * (1. + listfracradi)**2 / listrsma**2 / listfracradi**2
            listalbgtess = listsamp[:, 5]
            listlabl = ['Nightside [ppm]', 'Secondary [ppm]', 'Modulation [ppm]', 'Thermal [ppm]', 'Reflected [ppm]', 'Geometric Albedo']
            #tdpy.mcmc.plot_grid(pathimag, 'post_alle', listsamp, listlabl, plotsize=2.5)
            
            listdeptseco = listsamp[:, 1]
            medideptseco = np.median(listsamp[:, 1])
            stdvdeptseco = np.std(listsamp[:, 1])
            medideptnigh = np.median(listsamp[:, 0])
            stdvdeptnigh = np.std(listsamp[:, 0])
            
            prnt_list(listalbgtess, 'Albedo TESS only')
            prnt_list(listdeptseco * 1e6, 'Secondary depth [ppm]')
            prnt_list(listdeptnigh * 1e6, 'Nightside depth [ppm]')
        
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

        # plot a lightcurve from the posteriors
        gdat.lcurmodl = alles.get_posterior_median_model(strginst, 'flux', xx=gdat.time)
        gdat.lcurbasealle = alles.get_posterior_median_baseline(strginst, 'flux', xx=gdat.time)
        
        gdat.lcurdetr = gdat.arrylcurdetr[:, 1] - gdat.lcurbasealle

        # determine data gaps for overplotting model without the data gaps
        gdat.indxtimegapp = np.argmax(gdat.time[1:] - gdat.time[:-1]) + 1
        figr = plt.figure(figsize=gdat.figrsize[3, :])
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
        figr, axis = plt.subplots(figsize=gdat.figrsize[0, :])
        
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

        # calculate prior on the mass ratio (Stassun+2017)
        Mp = np.random.normal(loc=(375.99289 *c.M_earth/c.M_sun).value, scale=(20.34112*c.M_earth/c.M_sun).value, size=10000)
        Ms = np.random.normal(loc=1.52644, scale=0.361148, size=10000)
        q = Mp / Ms
        print( 'q', np.mean(q), np.std(q), np.percentile(q, [16,50,84] ) )
        print( 'q', np.percentile(q,50), np.percentile(q,50)-np.percentile(q,16), np.percentile(q,84)-np.percentile(q,50) )
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
        numbsampfeww = 4
        gdat.numbtime = gdat.arrylcur.shape[0]
        numbsamp = parapost.shape[0]
        indxsamp = np.arange(numbsamp)
        indxsampplot = np.random.choice(indxsamp, size=numbsampfeww)
        listarrysamp = np.empty((numbsampfeww, gdat.numbtime, 3))
        listarrysamp[:, :, 0] = gdat.arrylcur[None, :, 0]
        for i in np.arange(numbsampfeww):
            listarrysamp[i, :, 1], _ = retr_modl_trap(gdat, parapost[i, :])
        listarrysamp[:, :, 2] = 0.
    
        # plot light curve + samples from the trapezoid model
        figr, axis = plt.subplots(figsize=(6, 4))
        axis.plot(gdat.arrylcurdetr[:, 0], gdat.arrylcurdetr[:, 1], color='grey', alpha=0.3)
        axis.plot(gdat.arrylcurdetrbind[:, 0], gdat.arrylcurdetrbind[:, 1], color='k', ls='')
        for i in np.arange(numbsampfeww):
            axis.plot(gdat.arrylcurdetr[:, 0], listarrysamp[i, :, 1], alpha=0.2, color='b')
        axis.set_xlabel('Time [BJD]')
        plt.subplots_adjust()
        path = gdat.pathimag + 'lcurtrap.%s' % gdat.strgplotextn
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()

        # plot phase curve + samples from the trapezoid model
        figr, axis = plt.subplots(figsize=(6, 4))
        axis.plot(gdat.arrypcurdetr[j][:, 0], gdat.arrypcurdetr[j][:, 1], color='grey', alpha=0.3)
        axis.plot(gdat.arrypcurdetrbind[j][:, 0], gdat.arrypcurdetrbind[j][:, 1], color='k', ls='')
        for i in np.arange(numbsampfeww):
            pcur = tesstarg.util.fold_lcur(listarrysamp[i, :, :], gdat.epocinfe[j], gdat.periinfe[j])
            axis.plot(pcur[:, 0], pcur[:, 1], alpha=0.2, color='b')
        axis.set_xlim([-0.01, 0.01])
        axis.set_xlabel('Phase')
        plt.subplots_adjust()
        path = gdat.pathimag + 'pcurtrap.%s' % gdat.strgplotextn
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()

    # make a contour plot of geometric albedo without and with thermal component prior
    ## calculate the geometric albedo with the ATMO prior
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
    listalbg = listdeptrefl * (smaxprio / radiplanprio)**2
    listalbg = listalbg[listalbg > 0]
    
    prnt_list(listalbg, 'Albedo')
    
    # wavelength axis
    gdat.conswlentmpt = 0.0143877735e6 # [um K]
    
    print('Resonances...')
    gdat.prioreso = retr_reso(gdat.prioperi)

    ## eliminate those without radius measurements
    listradiknwn = listradiknwn[np.isfinite(listradiknwn)]

    for j in gdat.indxplan:
        time = np.empty(500)
        for n in range(500):
            time[n] = gdat.mediepoc[j] + gdat.mediperi[j] * n
        objttime = astropy.time.Time(time, format='jd', scale='utc', out_subfmt='date_hm')
        listtimelabl = objttime.iso
        for n in range(500):
            if time[n] > 2458788 and time[n] < 2458788 + 200:
                print('%f, %s' % (time[n], listtimelabl[n]))

    # plot TTVs
    

    figr, axis = plt.subplots(figsize=gdat.figrsize[4, :])

    binsalbg = np.linspace(min(np.amin(listalbgtess), np.amin(listalbg)), max(np.amax(listalbgtess), np.amax(listalbg)), 100)
    meanalbg = (binsalbg[1:] + binsalbg[:-1]) / 2.
    pdfnalbgtess = scipy.stats.gaussian_kde(listalbgtess, bw_method=.2)(meanalbg)
    pdfnalbg = scipy.stats.gaussian_kde(listalbg, bw_method=.2)(meanalbg)
    #pdfnalbgtess = np.histogram(listalbgtess, bins=binsalbg)[0] / float(listalbgtess.size)
    #pdfnalbg = np.histogram(listalbg, bins=binsalbg)[0] / float(listalbg.size)
    axis.plot(meanalbg, pdfnalbgtess, label='TESS only', lw=2)
    axis.plot(meanalbg, pdfnalbg, label='TESS + ATMO prior', lw=2)
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
   
    # infer brightness temperatures
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

        gdat.specstarintg = retr_modl_spec(gdat, gdat.tmptstarprio, strgtype='intg')
        
        gdat.specstarthomlogt = scipy.interpolate.interp1d(gdat.meanwlenthomraww, gdat.specstarthomraww)(gdat.cntrwlen)
        gdat.specstarthomdiff = gdat.specstarthomlogt / gdat.cntrwlen
        gdat.specstarthomintg = np.sum(gdat.diffwlen * \
                                scipy.interpolate.interp1d(gdat.meanwlenthomraww, gdat.specstarthomraww)(gdat.meanwlen) / gdat.meanwlen)

        gdat.deptobsd = arrydata[k, 2]
        gdat.stdvdeptobsd = arrydata[k, 3]
        gdat.varideptobsd = gdat.stdvdeptobsd**2
    
        numbdoff = gdat.numbdatatmpt - numbpara
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
    
    figr, axis = plt.subplots(4, 1, figsize=gdat.figrsize[3, :], sharex=True)
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
    figr, axis = plt.subplots(nrows=1, ncols=2, sharey=True, gridspec_kw={'width_ratios': [1, 2]}, figsize=gdat.figrsize[0, :])
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
  
    return


    # type of sampling
    samptype = 'emce'
    
    listaminchi2runs = []
    listsamppararuns = []
    listlistlablpara = []
    listlistlablparafull = []
        
    # get data
    path = pathdata + 'data_preparation/TESS.csv'
    arryfrst = np.loadtxt(path, delimiter=',', skiprows=1)
    
    # phase-fold the data
    arrythrd = np.copy(arryfrst)
    arrythrd[:, 0] = ((arryfrst[:, 0] - gdat.epocprio) % gdat.periprio) / gdat.peri
    
    # parse the data
    gdat.meanphas = arrythrd[:, 0]
    gdat.data = arrythrd[:, 1]
    gdat.datastdv = arrythrd[:, 2]
    
    # sort the data
    indxsort = np.argsort(gdat.meanphas)
    gdat.meanphas = gdat.meanphas[indxsort]
    gdat.data = gdat.data[indxsort]
    gdat.datastdv = gdat.datastdv[indxsort]

    # mask out the primary transit
    gdat.timetole = 6. / 24.
    indx = np.where(abs(gdat.meanphas - 0.5) < (1. - gdat.timetole / gdat.peri) / 2.)[0]
    if indx.size == gdat.meanphas.size:
        raise Exception('')
    gdat.data = gdat.data[indx]
    gdat.datastdv = gdat.datastdv[indx]
    gdat.meanphas = gdat.meanphas[indx]
    numbphas = gdat.data.size
    indxphas = np.arange(numbphas)
    
    fileoutp = open(pathdata + 'post.csv', 'w')
    fileoutp.write(' & ')
    fileoutp.write(' & ')
    
    # list of models
    listmodltype = ['simp', 'shft']
    
    fileoutp.write('\\\\\n')
    fileoutp.write('\\hline\n')
    fileoutp.write('$\chi^2_{\\nu}$ & ')
    fileoutp.write(' & ')
    fileoutp.write('\\hline\n')
    fileoutp.write('\\hline\n')
    fileoutp.write('\\\\\n')
    
    strgbins = 'unbd'
    strgmask = 'defa'
    for modltype in listmodltype:
        if modltype == 'simp':
            numbpara = 8
        if modltype == 'shft':
            numbpara = 9
        
        if samptype == 'emce':
            numbwalk = 5 * numbpara
            numbsampwalk = 1000
            numbsampburnwalk = numbsampwalk / 10
            if numbsampwalk == 0:
                raise Exception('')
            if numbsampburnwalk == 0:
                raise Exception('')
            numbsamp = numbsampwalk * numbwalk
            numbsampburn = numbsampburnwalk * numbwalk
        
        numbdoff = numbphas - numbpara
        
        listlablpara = ['Constant', 'Nightside emission [ppm]', \
                        'Planetary Modulation (B1 amplitude) [ppm]', 'B2 Amplitude [ppm]', 'B3 Amplitude [ppm]', \
                        'A1 Amplitude [ppm]', 'A2 Amplitude [ppm]', 'A3 Amplitude [ppm]']

        liststrgpara = ['cons', 'deptnigh', 'aco1', 'aco2', 'aco3', 'asi1', 'asi2', 'asi3']
        if modltype == 'shft':
            listlablpara += ['Phase offset [degree]']
            liststrgpara += ['shft']
        listlablparafull = listlablpara[:]
        liststrgparafull = liststrgpara[:]
        listlablparafull += ['Secondary transit depth [ppm]', 'Geometric albedo']
        liststrgparafull += ['deptseco', 'albg']
        
        numbparafull = len(liststrgparafull)
        indxpara = np.arange(numbpara)
        indxparafull = np.arange(numbparafull)
        gdat.limtpara = np.empty((2, numbpara))
        # cons
        gdat.limtpara[0, 0] = 0.9
        gdat.limtpara[1, 0] = 1.1
        # nightside emission [ppm]
        gdat.limtpara[0, 1] = 0.
        gdat.limtpara[1, 1] = 500.
        # amfo
        gdat.limtpara[0, 2] = -1e3
        gdat.limtpara[1, 2] = 0.
        # amfo
        gdat.limtpara[0, 3:8] = -1e3
        gdat.limtpara[1, 3:8] = 1e3
        if modltype == 'shft':
            # phas
            gdat.limtpara[0, 8] = -10.
            gdat.limtpara[1, 8] = 10.
        
        gdat.indxphasseco = np.where(abs(gdat.meanphas - 0.5) < gdat.durapost / gdat.peripost)[0]
        dictfour = [gdat, modltype]
        dicticdf = []
        
        if samptype == 'emce':
            indxwalk = np.arange(numbwalk)
            parainit = []
            for k in indxwalk:
                parainit.append(np.empty(numbpara))
                meannorm = (gdat.limtpara[0, :] + gdat.limtpara[1, :]) / 2.
                stdvnorm = (gdat.limtpara[0, :] - gdat.limtpara[1, :]) / 10.
                parainit[k]  = (scipy.stats.truncnorm.rvs((gdat.limtpara[0, :] - meannorm) / stdvnorm, \
                                                            (gdat.limtpara[1, :] - meannorm) / stdvnorm)) * stdvnorm + meannorm
            objtsamp = emcee.EnsembleSampler(numbwalk, numbpara, retr_lpos_four, args=dictfour)
            parainitburn, prob, state = objtsamp.run_mcmc(parainit, numbsampburnwalk)
            objtsamp.reset()
            objtsamp.run_mcmc(parainitburn, numbsampwalk)
            objtsave = objtsamp
        else:
        
            sampler = dynesty.NestedSampler(retr_llik_four, icdf, numbpara, logl_args=dictfour, ptform_args=dicticdf, bound='single', dlogz=1000.)
            sampler.run_nested()
            results = sampler.results
            results.summary()
            objtsave = results
            
        if samptype == 'emce':
            numbsamp = objtsave.flatchain.shape[0]
            indxsampwalk = np.arange(numbsampwalk)
        else:
            numbsamp = objtsave['samples'].shape[0]
        
        indxsamp = np.arange(numbsamp)
        
        # resample the nested posterior
        if samptype == 'nest':
            weights = np.exp(results['logwt'] - results['logz'][-1])
            samppara = dynesty.utils.resample_equal(results.samples, weights)
            assert samppara.size == results.samples.size
        
        if samptype == 'emce':
            postpara = objtsave.flatchain
        else:
            postpara = samppara
        
        sampllik = objtsave.lnprobability.flatten()
        aminchi2 = (-2. * np.amax(sampllik) / numbdoff)
        listaminchi2runs.append(aminchi2)
        
        postparatemp = np.copy(postpara)
        postpara = np.empty((numbsamp, numbparafull))
        postpara[:, :-(numbparafull - numbpara)] = postparatemp
        for k in indxsamp:
            # secondary depth
            postpara[k, -2] = postpara[k, 2] - postpara[k, 3]
            # geometric albedo
            postpara[:, -1] = -(smaxprio**2 / radiplan**2) * postpara[:, 2]
    
        listsamppararuns.append(postpara)
        listlistlablpara.append(listlablpara)
        listlistlablparafull.append(listlablparafull)
    
        # plot the posterior
        
        ### histogram
        for k in indxparafull:
            figr, axis = plt.subplots()
            if samptype == 'emce':
                axis.hist(postpara[:, k]) 
            else:
                axis.hist(samppara[:, k]) 
            axis.set_xlabel(listlablparafull[k])
            path = pathimag + 'diag/hist_%s_%s_%s_%s.%s' % (liststrgparafull[k], modltype, strgmask, strgbins, gdat.strgplotextn)
            plt.tight_layout()
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
        
        if samptype == 'nest':
            for keys in objtsave:
                if isinstance(objtsave[keys], np.ndarray) and objtsave[keys].size == numbsamp:
                    figr, axis = plt.subplots()
                    axis.plot(indxsamp, objtsave[keys])
                    path = pathimag + '%s_%s.%s' % (keys, modltype, gdat.strgplotextn)
                    print('Writing to %s...' % path)
                    plt.savefig(path)
        else:
            ## log-likelihood
            figr, axis = plt.subplots()
            if samptype == 'emce':
                for i in indxwalk:
                    axis.plot(indxsampwalk[::10], objtsave.lnprobability[::10, i])
            else:
                axis.plot(indxsamp, objtsave['logl'])
            path = pathimag + 'diag/llik_%s_%s_%s.%s' % (modltype, strgmask, strgbins, gdat.strgplotextn)
            plt.tight_layout()
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
        
            chi2 = -2. * objtsave.lnprobability
            print('Posterior-mean chi2: ')
            print(np.mean(chi2))
            print('Posterior-mean chi2 per dof: ')
            print(np.mean(chi2) / numbdoff)
        
        tdpy.mcmc.plot_grid(pathimag, 'post_sinu_%s_%s_%s' % (modltype, strgmask, strgbins), postpara, listlablparafull, plotsize=2.5)
    
        ### sample model phas
        numbsampplot = 100
        indxsampplot = np.random.choice(indxsamp, numbsampplot, replace=False)
        yerr = gdat.datastdv
        
        numbphasfine = 1000
        gdat.meanphasfine = np.linspace(0.1, 0.9, numbphasfine)
        phasmodlfine = np.empty((numbsampplot, numbphasfine))
        indxphasfineseco = np.where(abs(gdat.meanphasfine - 0.5) < gdat.durapost / gdat.peri)[0]
        for k, indxsampplottemp in enumerate(indxsampplot):
            if samptype == 'emce':
                objttemp = objtsave.flatchain
            else:
                objttemp = samppara
            cons = objttemp[indxsampplottemp, 0]
            deptnigh = objttemp[indxsampplottemp, 1]
            amfo = objttemp[indxsampplottemp, 2:8]
            if modltype == 'shft':
                shft = objttemp[indxsampplottemp, 8]
            else:
                shft = 0.
            phasmodlfine[k, :], deptpmodfine = retr_modl_four(gdat, gdat.meanphasfine, indxphasfineseco, cons, deptnigh, amfo, shft, modltype)
        
        strgextn = '%s_%s_%s' % (modltype, strgmask, strgbins)
        ## log-likelihood
        figr, axis = plt.subplots(figsize=(12, 6))
        axis.errorbar(gdat.arrypcurdetrbind[:, 0] % 1., (gdat.pcurdetrbind - 1) * 1e6, \
                                                            yerr=1e6*gdat.stdvpcurdetrbind, color='k', marker='o', ls='', markersize=1)
        for k, indxsampplottemp in enumerate(indxsampplot):
            axis.plot(gdat.meanphasfine, (phasmodlfine[k, :] - 1) * 1e6, alpha=0.5, color='b')
        axis.set_xlim([0.1, 0.9])
        axis.set_ylim([-400, 1000])
        axis.set_ylabel('Relative Flux - 1 [ppm]')
        axis.set_xlabel('Phase')
        plt.tight_layout()
        path = pathimag + 'pcur_sine_%s.%s' % (strgextn, gdat.strgplotextn)
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()

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

