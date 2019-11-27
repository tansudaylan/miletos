import os, fnmatch
import sys
import numpy as np
import scipy.interpolate

import pandas as pd

import allesfitter
import allesfitter.config
import matplotlib.pyplot as plt
from allesfitter.exoworlds_rdx.lightcurves.index_transits import index_eclipses
from exoworlds.tess import extract_SPOC_data, extract_QLP_data
from allesfitter.exoworlds_rdx.lightcurves.lightcurve_tools import rebin_err, phase_fold

import astroquery
import astropy
import tdpy.mcmc
import tdpy.util
from tdpy.util import prnt_list
from tdpy.util import summgene
import tesstarg.util
import tcat.main

import emcee



def plot_toii(gdat):

    #listtici = [16740101]
    
    liststrgfiel = ['TIC / Confirmed Name', \
                    'TOI', \
                    'Comm Disp', \
                    'Epoch (BJD)', \
                    'Epoch err', \
                    'Perod (days)', \
                    'Period (days) Error', \
                    'Depth mmag', \
                    'Depth mmag Error', \
                    'Depth ppm', \
                    'Depth ppm Error', \
                    'Duration (hrs)', \
                    'Duration (hrs) Error', \
                    'Inclination (deg)', \
                    'Inclination (deg) Error', \
                    'Impact Param', \
                    'Impact Param Error', \
                    'Rad_p/Rad_s', \
                    'Rad_p/Rad_s Error', \
                    'a/Rad_s', \
                    'a/Rad_s Error', \
                    'Radius (R_Earth)', \
                    'Radius (R_Earth) Error', \
                    'Mass (M_Earth)', \
                    'Mass (M_Earth) Error', \
                    'Insolation (Earth Flux)', \
                    'Insolation (Earth Flux) Error', \
                    'Equilibrium Temp (K)', \
                    'Equilibrium Temp (K) Error', \
                    'Fitted Stellar Density (g/cm^3)', \
                    'Fitted Stellar Density (g/cm^3) Error', \
                    'Semi-Major Axis (AU)', \
                    'Semi-Major Axis (AU) Error', \
                    'Eccentricity', \
                    'Eccentricity (Error)', \
                    'Arg of Periastron', \
                    'Arg of Periastron Error' , \
                    'Time of Periastron (BJD)', \
                    'Time of Periastron (BJD) Error', \
                    'Velocity Semi-amp (m/s)', \
                    'Velocity Semi-amp (m/s) Error', \
                    'SNR', \
                    'Date', \
                    'User', \
                    'Group',\
                    'Tag', \
                    'Notes']
    
    numbstrgfiel = len(liststrgfiel)
    indxstrgfiel = np.arange(numbstrgfiel)
    
    numbtoii = len(listtici)
    dictvarb = dict()
    for n in indxstrgfiel:
        dictvarb[liststrgfiel[n]] = np.empty(numbtoii)
    for i, tici in enumerate(listtici):
        strgtici = '%s' % tici
        
        with open(pathfileexop, 'r') as objtfile:
            listline = []
            for line in objtfile:
                listline.append(line)
            numbline = len(listline)
            indxline = np.arange(numbline)
            for k in indxline:
                if listline[k].startswith('PLANET'):
                    listindx = []
                    for n in indxstrgfiel:
                        indx = listline[k+1].find(liststrgfiel[n])
                        listindx.append(indx)
                    listindx.append(-1)
                    for l in range(100):
                        if listline[k+2+l].startswith('\n'):
                            break
                        for n in indxstrgfiel:
                            linesplt = listline[k+2+l][listindx[n]:listindx[n+1]]
                            #if liststrgfiel[n] != 'TIC / Confirmed Name':
                                
    
                            try:
                                dictvarb[liststrgfiel[n]][i] = float(linesplt)
                            except:
                                pass
    
    print('dictvarb')
    print(dictvarb)
    print

    # make a plot of radius vs insolation
    figr, axis = plt.subplots(figsize=(12, 12))
    axis.scatter(dictvarb['Perod (days)'], dictvarb['Radius (R_Earth)'])
    axis.set_xlabel(r'Period [days]')
    axis.set_ylabel(r'Radis [R_{\earth}]')
    path = pathimag + 'radiperi.pdf'
    plt.savefig(path)
    plt.close()
    



    
    # download all txt files
    #for tici in listtici:
    #    strgtici = '%s' % tici 
    #    cmnd = 'wget https://exofop.ipac.caltech.edu/tess/download_target.php?id=%s' % strgtici
    #    os.system(cmnd)
    

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


def retr_specbbod(tmpt, wlen):
    
    #0.0143877735e6 # [um K]
    spec = 3.742e11 / wlen**5 / (np.exp(0.0143877735e6 / (wlen * tmpt)) - 1.)
    
    return spec


def retr_modl_spec(gdat, tmpt, booltess=False, strgtype='intg'):
    
    if booltess:
        thpt = scipy.interpolate.interp1d(gdat.meanwlenband, gdat.thptband)(wlen)
    else:
        thpt = 1.
    
    if strgtype == 'intg':
        spec = retr_specbbod(gdat, tmpt, gdat.meanwlen)
        spec = np.sum(gdat.diffwlen * spec)
        #print('gdat.meanwlen')
        #summgene(gdat.meanwlen)
        #print('spec')
        #summgene(spec)
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
        deptplan = gdat.fracradiprio**2 * specboloplan / gdat.specstarintg
        llik = -0.5 * np.sum((deptplan - gdat.deptobsd)**2 / gdat.varideptobsd)
        #print('tmpt')
        #print(tmpt)
        #print('specboloplan')
        #print(specboloplan)
        #print('deptplan')
        #print(deptplan)
        #print('gdat.deptobsd')
        #print(gdat.deptobsd)
        #print('gdat.varideptobsd')
        #print(gdat.varideptobsd)
        #print
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


def retr_mask(time, flag, strgmask):
    
    frst = time[0]
    four = time[-1]
    mean = np.mean(time)
    diff = time - mean
    seco = diff[np.where(diff < 0)][-1]
    thrd = diff[np.where(diff > 0)][0]
    
    indx = np.where(~((time < frst + 0.5) | ((time < thrd + 0.5) & (time > mean)) | (time > four - 0.25) | \
                                                                    ((time < 2458511.8) & (time > 2458511.3))))[0]
    
    if strgmask == 'ful1':
        indx = np.setdiff1d(indx, np.where(time > 2458504)[0])
    if strgmask == 'ful2':
        indx = np.setdiff1d(indx, np.where(time < 2458504)[0])
    
    return indx


def retr_reso(listperi, maxmordr=10):
    
    numbplan = len(listperi)
    indxplan = np.arange(numbplan)
    ratiperi = np.zeros((numbplan, numbplan))
    reso = np.zeros((numbplan, numbplan, 2))
    for j in indxplan:
        for jj in indxplan:
            if listperi[j] > listperi[jj]:
                ratiperi[j, jj] = listperi[j] / listperi[jj]

                print('P(.%02d) / (.%02d): %g' % (j, jj, ratiperi[j, jj]))
                minmdiff = 1e12
                for a in range(1, maxmordr):
                    for aa in range(1, maxmordr):
                        diff = abs(float(a) / aa - ratiperi[j, jj])
                        if diff < minmdiff:
                            minmdiff = diff
                            minmreso = a, aa
                reso[j, jj, :] = minmreso
                print('minmdiff') 
                print(minmdiff)
                print('minmreso')
                print(minmreso)
                print
    
    return reso


def main( \
         strgtarg=None, \
         strgmast=None, \
         boolphascurv=False, \
         strgdata=None, \
         
         # Boolean flag to use SAP instead of PDC by default, when strgdata is None.
         boolsapp=False, \
         
         listlimttimemask=None, \
        ):
    
#    
#    if 'SPOC' in liststrgdata:
#        
#        # download data
#        obsTable = astroquery.mast.Observations.query_criteria(target_name=tici, \
#                                                               obs_collection='TESS', \
#                                                               dataproduct_type='timeseries', \
#                                               )
#        listpath = []
#        for k in range(len(obsTable)):
#            dataProducts = astroquery.mast.Observations.get_product_list(obsTable[k])
#            want = (dataProducts['obs_collection'] == 'TESS') * (dataProducts['dataproduct_type'] == 'timeseries')
#            for k in range(len(dataProducts['productFilename'])):
#                if not dataProducts['productFilename'][k].endswith('_lc.fits'):
#                    want[k] = 0
#            listpath.append(pathtarg + dataProducts[want]['productFilename'].data[0]) 
#            manifest = astroquery.mast.Observations.download_products(dataProducts[want], download_dir=pathtarg)
#        
#        if len(obsTable) == 0:
#            return
#        else:
#            print('Found TESS SPOC data.')
#    elif 'QLOP' in liststrgdata:
#        print('Reading the QLP data on the target...')
#        catalogData = astroquery.mast.Catalogs.query_object(tici, catalog="TIC")
#        rasc = catalogData[0]['ra']
#        decl = catalogData[0]['dec']
#        sector_table = astroquery.mast.Tesscut.get_sectors(SkyCoord(rasc, decl, unit="deg"))
#        listisec = sector_table['sector'].data
#        listicam = sector_table['camera'].data
#        listiccd = sector_table['ccd'].data
#        for m, sect in enumerate(listisec):
#            path = '/pdo/qlp-data/orbit-%d/ffi/cam%d/ccd%d/LC/' % (listisec[m], listicam[m], strgiccd[m])
#            pathqlop = path + str(tici) + '.h5'
#            time, flux, stdvflux = read_qlop(pathqlop)
#
#    # read the files to make the CSV file
#    if 'SPOC' in liststrgdata:
#        pathdown = pathtarg + 'mastDownload/TESS/'
#        arry = read_tesskplr_fold(pathdown, pathalle)
#        pathoutp = '%sTESS.csv' % pathalle
#        np.savetxt(pathoutp, arry, delimiter=',')
#    
#    # construct target folder structure
#    pathalle = pathtarg + 'allesfit'
#    if strgalleextn is not None:
#        pathalle += '_' + strgalleextn
#    pathalle += '/'
#    cmnd = 'mkdir -p %s %s' % (pathtarg, pathalle)
#    os.system(cmnd)
#    


    gdat = tdpy.util.gdatstrt()
    
    gdat.strgdata = strgdata
    gdat.strgtarg = strgtarg
    gdat.strgmast = strgmast
    gdat.boolphascurv = boolphascurv
    gdat.boolsapp = boolsapp

    gdat.pathbase = os.environ['TESSTOII_DATA_PATH'] + '/'
    gdat.pathexop = gdat.pathbase + 'exofop/'
    gdat.pathobjt = gdat.pathbase + '%s/' % gdat.strgtarg
    gdat.pathdata = gdat.pathobjt + 'data/'
    gdat.pathimag = gdat.pathobjt + 'imag/'
    
    os.system('mkdir -p %s' % gdat.pathdata)
    os.system('mkdir -p %s' % gdat.pathimag)

    print('TESS TOI/allesfitter pipeline started.')

    # plotting settings
    gdat.liststrgplan = ['b', 'c', 'd', 'e']
    gdat.listcolrplan = ['g', 'r', 'c', 'm']
    
    gdat.numbplan = 4
    gdat.indxplan = np.arange(gdat.numbplan)
    print('Resonances...')
    # temp
    gdat.mediepoc = np.array([2458572.111694, 2458572.398315, 2458571.335571, 2458586.566895])
    gdat.medismaj = np.array([3.795304, 6.203183, 14.175671, 19.593409])
    gdat.meditmpt = np.array([3.795304, 6.203183, 14.175671, 19.593409])
    gdat.mediradi = np.array([1.7, 2.3, 2.8, 3.3])
    gdat.mediperi = np.array([3.795304, 6.203183, 14.175671, 19.593409])
    gdat.reso = retr_reso(gdat.mediperi)

    # get all known planets
    gdat.patharch = gdat.pathbase + 'data/NASA.csv'
    dataarch = pd.read_csv(gdat.patharch, skiprows=76)
    
    ## convert Jupiter radius to Earth radius
    listradiknwn = dataarch['pl_radj'] * 11.21
    
    ## eliminate those without radius measurements
    listradiknwn = listradiknwn[np.isfinite(listradiknwn)]

    gdat.listlablplan = ['04', '03', '01', '02']
    for j in gdat.indxplan:
        print(gdat.listlablplan[j])
        time = np.empty(500)
        for n in range(500):
            time[n] = gdat.mediepoc[j] + gdat.mediperi[j] * n
        objttime = astropy.time.Time(time, format='jd', scale='utc', out_subfmt='date_hm')
        listtimelabl = objttime.iso
        for n in range(500):
            if time[n] > 2458788 and time[n] < 2458788 + 200:
                print('%f, %s' % (time[n], listtimelabl[n]))
        print


    # plot resonances
    figr, axis = plt.subplots(figsize=(6, 4))
    for j in gdat.indxplan:
        for jj in gdat.indxplan:
            if gdat.mediperi[j] > gdat.mediperi[jj]:
                ratiperi = gdat.mediperi[j] / gdat.mediperi[jj]
                axis.axvline(ratiperi, color='k')
                axis.axvline(float(gdat.reso[j, jj,0]) / gdat.reso[j, jj, 1], color='grey', ls='--')
    axis.set_xlabel('Period ratio')
    plt.subplots_adjust()
    path = gdat.pathimag + 'reso.pdf'
    print('Writing to %s...' % path)
    plt.savefig(path)
    plt.close()

    # plot radii
    figr, axis = plt.subplots(figsize=(6, 4))
    for j in gdat.indxplan:
        colrredd = gdat.meditmpt[j, None]
        colrblue = 1. - colrredd
        colr = np.zeros((4, 1))
        colr[1, 0] = colrredd
        colr[2, 0] = colrblue
        #colr = colr.T
        size = gdat.mediradi[j] * 5.
        for tmpt in [500., 700,]:
            smaj = tmpt
            axis.axvline(smaj, ls='--')
        axis.scatter(gdat.medismaj[j, None], np.array([1.]), s=size)
        #axis.scatter(gdat.medismaj[j, None], np.array([1.]), s=gdat.mediradi[j], c=colr)
    axis.set_xlabel('Distance from the star [AU]')
    plt.subplots_adjust()
    path = gdat.pathimag + 'orbt.pdf'
    print('Writing to %s...' % path)
    plt.savefig(path)
    plt.close()
    
    
    # plot radius histogram
    figr, axis = plt.subplots(figsize=(6, 4))
    axis.hist(listradiknwn, bins=1000, color='k')
    for j in gdat.indxplan:
        axis.axvline(gdat.mediradi[j], color=gdat.listcolrplan[j], ls='--', label=gdat.liststrgplan[j])
    axis.set_xlim([0., 4])
    axis.set_xlabel('Radius [R]')
    axis.set_ylabel('N')
    plt.subplots_adjust()
    #axis.legend()
    path = gdat.pathimag + 'histradi.pdf'
    print('Writing to %s...' % path)
    plt.savefig(path)
    plt.close()
   
    # plot TTVs
    

    if gdat.strgdata != 'tcat':
        pathlcurspoc = gdat.pathobjt + 'mastDownload/TESS/'
        if not os.path.exists(pathlcurspoc):
            print('Trying to download SPOC data with keyword: %s' % gdat.strgmast)
            listpathdown = tesstarg.util.down_spoclcur(gdat.pathobjt, strgmast=gdat.strgmast)
            print('listpathdown')
            print(listpathdown)
    
        if os.path.exists(pathlcurspoc):
            if gdat.boolsapp:
                gdat.strgdata = 'sapp'
            else:
                gdat.strgdata = 'pdcc' 
        else:
            gdat.strgdata = 'qlop'
    
        listpathlcur = []
        if gdat.strgdata == 'sapp' or gdat.strgdata == 'pdcc':
            listpathlcurinte = []
            for extn in os.listdir(pathlcurspoc):
                pathlcurinte = pathlcurspoc + extn + '/'
                listpathlcurinte.append(pathlcurinte)
                pathlcur = pathlcurinte + fnmatch.filter(os.listdir(pathlcurinte), '*_lc.fits')[0]
                listpathlcur.append(pathlcur)
        if gdat.strgdata == 'qlop':
            pathlcurqlop = gdat.pathobjt + 'qlop/'
            os.system('mkdir -p %s' % pathlcurqlop)
            pathlcur = pathlcurqlop + fnmatch.filter(os.listdir(pathlcurqlop), 'sector-*')[0]
            print('pathlcur')
            print(pathlcur)
    else:
        tcat.main.main( \
             ticitarg=int(gdat.strgmast), \
             labltarg=strgtarg, \
             strgtarg=strgtarg, \
            )
    
    # make sure the list of paths to sector files are time-sorted
    listpathlcur.sort()

    # plot all TOIs and overplot this one
    print('HACKING, assuming that strgmast is TIC ID...')
    #catalogData = astroquery.mast.Catalogs.query_object(gdat.strgmast, catalog="TIC")
    #rasc = catalogData[0]['ra']
    #decl = catalogData[0]['dec']
    #gdat.strgtici = '%s' % catalogData[0]['ID']
    gdat.strgtici = gdat.strgmast

    #plot_toii(gdat)
    
    listpathsapp = []
    listpathpdcc = []
    if gdat.strgdata == 'tcat' or gdat.strgdata == 'qlop':
        listarry = [[] for a in range(1)]
    else:
        listarry = [[] for a in range(2)]
        
    for k, pathlcur in enumerate(listpathlcur):
        if gdat.strgdata == 'tcat':
            arry = np.loadtxt(pathdata + 'band.csv', delimiter=',', skiprows=9)
            listarry[0].append(arry)
        elif gdat.strgdata == 'qlop':
            arry = tesstarg.util.read_qlop(pathlcur, typeinst='tess', boolmask=True)
            listarry[0].append(arry)
        else:
            # temp
            boolextrmine = False
            if boolextrmine:
                arrysapp = tesstarg.util.read_tesskplr_file(pathlcur, typeinst='tess', strgtype='SAP_FLUX', boolmask=True)
                arrypdcc = tesstarg.util.read_tesskplr_file(pathlcur, typeinst='tess', strgtype='PDCSAP_FLUX', boolmask=True)
            else:
                # extract SPOC data
                foldsapp = listpathlcurinte[k] + 'TESS_SAP/'
                foldpdcc = listpathlcurinte[k] + 'TESS_PDCSAP/'
                extract_SPOC_data([pathlcur], outdir=foldsapp, PDC=False, auto_correct_dil=True, extract_centd=True, extract_dil=True)
                extract_SPOC_data([pathlcur], outdir=foldpdcc, PDC=True, auto_correct_dil=True, extract_centd=True, extract_dil=True)
                # read the file
                pathsapp = foldsapp + 'TESS.csv'
                pathpdcc = foldpdcc + 'TESS.csv'
                listpathsapp.append(pathpdcc)
                listpathpdcc.append(pathsapp)
                gdat.timepdcc, lcurpdcc, stdvlcurpdcc = np.genfromtxt(pathpdcc, delimiter=',', unpack=True)
                gdat.timesapp, lcursapp, stdvlcursapp = np.genfromtxt(pathsapp, delimiter=',', unpack=True)
                arrypdcc = np.empty((gdat.timepdcc.size, 3))
                arrypdcc[:, 0] = gdat.timepdcc
                arrypdcc[:, 1] = lcurpdcc
                arrypdcc[:, 2] = stdvlcurpdcc
                arrysapp = np.empty((gdat.timesapp.size, 3))
                arrysapp[:, 0] = gdat.timesapp
                arrysapp[:, 1] = lcursapp
                arrysapp[:, 2] = stdvlcursapp
    
            listarry[0].append(arrysapp)
            listarry[1].append(arrypdcc)
    
    if gdat.strgdata == 'tcat' or gdat.strgdata == 'qlop':
        listarry = np.concatenate(listarry[0], 0)
    if gdat.strgdata == 'tcat' or gdat.strgdata == 'qlop':
        for a in range(2):
            listarry = np.concatenate(listarry, 0)
   
    time = listarry[:, 0]
    numbtime = time.size
    indxtime = np.arange(numbtime)
    
    gdat.numbsect = len(listpathlcur)
    gdat.indxsect = np.arange(gdat.numbsect)
        
    if listlimttimemask is not None:
        # mask the data
        print('Masking the data...')
        arryumsk = np.copy(listarry)
        numbmask = listlimttimemask.shape[0]
        listindxtimemask = []
        for k in range(numbmask):
            indxtimemask = np.where((listarry[:, 0] < listlimttimemask[k, 1]) & (listarry[:, 0] > listlimttimemask[k, 0]))[0]
            listindxtimemask.append(indxtimemask)
        listindxtimemask = np.concatenate(listindxtimemask)
        listindxtimegood = np.setdiff1d(indxtime, listindxtimemask)
        listarry = listarry[listindxtimegood, :]
    
    # mask out the transits
    np.savetxt(gdat.pathobjt + 'allesfits/allesfit_bkgd/TESS.csv', listarry, delimiter=',', header='time,flux,flux_err')

    gdat.time = listarry[:, 0]
    gdat.lcur = listarry[:, 1]
    gdat.stdvlcur = listarry[:, 2]
    
    gdat.pathallebkgd = gdat.pathobjt + 'allesfits/allesfit_nomi/'
    gdat.pathallenomi = gdat.pathobjt + 'allesfits/allesfit_bkgd/'

    # background run
    os.system('mkdir -p %sallesfits/allesfit_bkgd' % gdat.pathobjt)
    for strg in ['params.csv', 'settings.csv', 'params_star.csv']:
        pathinit = '%sdata/%s' % (gdat.pathbase, strg)
        pathfinl = '%sallesfits/allesfit_bkgd/%s' % (gdat.pathobjt, strg)
        if not os.path.exists(pathfinl):
            os.system('cp %s %s' % (pathinit, pathfinl))
    ## mask out the transits for the background run
    epoc = 2458724.930827547
    peri = 6.134271796958897
    duramask = 0.5
    indxtimetran = tesstarg.util.retr_indxtimetran(gdat.time, epoc, peri, duramask)
    indxtimebkgd = np.setdiff1d(np.arange(gdat.time.size), indxtimetran)
    arrybkgd = listarry[indxtimebkgd, :]
    np.savetxt('%sallesfits/allesfit_bkgd/TESS.csv' % gdat.pathobjt, arrybkgd, delimiter=',', header='time,flux,flux_err')
    
    ### initial plot
    #path = gdat.pathallenomi + 'results/initial_guess_b.pdf'
    #if not os.path.exists(path):
    #    allesfitter.show_initial_guess(gdat.pathallenomi)
    #
    ### do the run
    #path = gdat.pathallenomi + 'results/mcmc_save.h5'
    #if False and not os.path.exists(path):
    #    allesfitter.mcmc_fit(gdat.pathallenomi)

    ### make the final plots
    #path = gdat.pathallenomi + 'results/mcmc_corner.pdf'
    #if False and not os.path.exists(path):
    #    allesfitter.mcmc_output(gdat.pathallenomi)
    #
    ##raise Exception('')
    ## nominal run
    ### copy the data to the nominal allesfitter path
    #path = gdat.pathobjt + 'TESS_PDCSAP/TESS.csv'
    #cmnd = 'cp %s %s' % (path, gdat.pathallenomi)
    #os.system(cmnd)
    #
    ### initial plot
    #path = gdat.pathallebkgd + 'results/initial_guess_b.pdf'
    #if not os.path.exists(path):
    #    allesfitter.show_initial_guess(gdat.pathallebkgd)
    #
    ### do the run
    #path = gdat.pathallenomi + 'results/mcmc_save.h5'
    #if not os.path.exists(path):
    #    allesfitter.mcmc_fit(gdat.pathallenomi)

    ### make the final plots
    #path = gdat.pathallenomi + 'results/mcmc_corner.pdf'
    #if not os.path.exists(path):
    #    allesfitter.mcmc_output(gdat.pathallenomi)

    # plotting
    gdat.figrsize = np.empty((5, 2))
    gdat.figrsize[0, :] = np.array([12., 4.])
    gdat.figrsize[1, :] = np.array([12., 6.])
    gdat.figrsize[2, :] = np.array([12., 10.])
    gdat.figrsize[3, :] = np.array([12., 14.])
    gdat.figrsize[4, :] = np.array([6., 6.])
    boolpost = False
    if boolpost:
        gdat.figrsize /= 1.5
    
    strgalletype = 'pdcc'
    
    # nominal run
    os.system('mkdir -p %sallesfits/allesfit_nomi' % gdat.pathobjt)
    
    gdat.periprio = 1.4811235
    gdat.epocprio = 2457095.68572
    radiplanprio = 1.783 # [RJ]
    radistarprio = radiplanprio / 0.08228 # [RJ]
    smaxprio = 0.03462 * 2093 # [RJ]
    inclprio = 86.79 # [deg]
    gdat.tmptstarprio = 9435. # [K]
    massplanprio = 2.44 # [Mjup]
    rsmaprio = (radiplanprio + radistarprio) / smaxprio
    gdat.fracradiprio = radiplanprio / radistarprio
    cosiprio = np.cos(inclprio * np.pi / 180.)
    massstarprio = 2.52 * 1048. # [MJ]
    fracmass = massplanprio / massstarprio
    print('rsmaprio')
    print(rsmaprio)
    print('gdat.fracradiprio')
    print(gdat.fracradiprio)
    print('cosiprio')
    print(cosiprio)
    print('fracmass')
    print(fracmass)
    
    gdat.boolplotspec = False

    if gdat.boolplotspec:
        ## TESS throughput 
        gdat.data = np.loadtxt(pathdata + 'band.csv', delimiter=',', skiprows=9)
        gdat.meanwlenband = gdat.data[:, 0] * 1e-3
        gdat.thptband = gdat.data[:, 1]
    
    gdat.timetess = 2457000.
    ## parameters
    # planet
    ## Half of the transit duration
    dura = 0.1203 / 2. # [day] Delrez2016
    
    ## expected ellipsoidal variation (EV) and Doppler beaming (DB)
    print('Predicting the ellipsoidal variation and Doppler beaming amplitudes...')
    ### EV
    #### limb and gravity darkening coefficients from Claret2017
    u = 0.4
    g = 0.2
    alphelli = 0.15 * (15 + u) * (1 + g) / (3 - u)
    deptelli = alphelli * massplanprio * np.sin(inclprio / 180. * np.pi)**2 / massstarprio * (radistarprio / smaxprio)**3
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
    
    # plot PDCSAP and SAP light curves
    if gdat.strgdata == 'sapp':
        numbframlcur = 3
    else:
        numbframlcur = 2
    figr, axis = plt.subplots(numbframlcur, 1, figsize=gdat.figrsize[1, :])
    axis[0].plot(gdat.timesapp - gdat.timetess, lcursapp, color='grey', marker='.', ls='')
    if listlimttimemask is not None:
        axis[0].plot(gdat.timesapp[listindxtimegood] - gdat.timetess, lcursapp[listindxtimegood], color='k', marker='.', ls='')
    axis[1].plot(gdat.timepdcc - gdat.timetess, lcurpdcc, color='grey', marker='.', ls='')
    if listlimttimemask is not None:
        axis[1].plot(gdat.timepdcc[listindxtimegood] - gdat.timetess, lcurpdcc[listindxtimegood], color='k', marker='.', ls='')
    if gdat.strgdata == 'sapp':
        axis[2].plot(timeflat - gdat.timetess, lcurflat, color='grey', marker='.', ls='')
    
    
    #axis[0].set_xticklabels([])
    axis[0].text(.97, .97, 'SAP', transform=axis[0].transAxes, size=20, color='r', ha='right', va='top')
    axis[1].text(.97, .97, 'PDC', transform=axis[1].transAxes, size=20, color='r', ha='right', va='top')
    if gdat.strgdata == 'sapp':
        axis[2].text(.97, .97, 'Flattened', transform=axis[2].transAxes, size=20, color='r', ha='right', va='top')
    axis[numbframlcur-1].set_xlabel('Time [BJD - 2457000]')
    for a in range(numbframlcur):
        axis[a].minorticks_on()
        axis[a].set_ylabel('Relative Flux')
    plt.subplots_adjust(hspace=0.)
    path = gdat.pathobjt + 'lcur.pdf'
    print('Writing to %s...' % path)
    plt.savefig(path)
    plt.close()
   
    gdat.timedata = gdat.timesapp
    gdat.lcurdata = gdat.lcursapp
    gdat.stdvlcurdata = gdat.lcursapp
    
    # plot colored light curve and individual light curves
    figr, axis = plt.subplots(1 + gdat.numbplan, 1, figsize=gdat.figrsize[1, :])
    axis[0].plot(gdat.timesapp - gdat.timetess, lcursapp, color='grey', marker='.', ls='')
    for j in gdat.indxplan:
        axis[0].plot(gdat.timesapp[indxtimetran[j]] - gdat.timetess, lcursapp[indxtimetran[j]], color=gdat.listcolrplan[j], marker='.', ls='')
        axis[j].text(.97, .97, gdat.listlablplan[j], transform=axis[0].transAxes, size=20, color='r', ha='right', va='top')
    axis[0].set_xlabel('Time [BJD - 2457000]')
    for a in range(numbframlcur):
        axis[a].minorticks_on()
        axis[a].set_ylabel('Relative Flux')
    plt.subplots_adjust(hspace=0.)
    path = gdat.pathobjt + 'data.pdf'
    print('Writing to %s...' % path)
    plt.savefig(path)
    plt.close()
    

    if gdat.strgdata == 'sapp':
        # fit a spline to the SAP light curve
        print('Fitting a spline to the SAP light curve...')
        timesplnedge = [0., 2458504., np.inf]
        numbsplnedge = len(timesplnedge)
        numbsplnknot = numbsplnedge - 1
        indxsplnknot = np.arange(numbsplnknot)

        # produce a table for the spline coefficients
        fileoutp = open(pathdata + 'coef.csv', 'w')
        fileoutp.write(' & ')
    
        figr, axis = plt.subplots(2, 1, figsize=gdat.figrsize[1, :])
        gdat.lcurspln = []
        lcursplnbase = []
        listobjtspln = []
        wdth = 4. / 24. # [days]
        for i in indxsplnknot:
            indxtimesplnregi = np.where((gdat.timesapp >= timesplnedge[i]) & (gdat.timesapp <= timesplnedge[i+1]))[0]
            timesplnregi = gdat.timesapp[indxtimesplnregi]
            lcursplnbaseregi = lcursapp[indxtimesplnregi]
            flux_err1 = stdvlcursapp[indxtimesplnregi]
            temp, indxtimesplntran, indxtimesplnoutt = index_eclipses(timesplnregi, gdat.epocprio, gdat.periprio, width_1=wdth, width_2=wdth)
            objtspln = UnivariateSpline(timesplnregi[indxtimesplnoutt], lcursplnbaseregi[indxtimesplnoutt]-1.)
            timesplnregifine = np.linspace(timesplnregi[0], timesplnregi[-1], 1000)
            gdat.lcurspln += list(lcursplnbaseregi-objtspln(timesplnregi))
            lcursplnbase += list(objtspln(timesplnregi))
            listobjtspln.append(objtspln)
        
            print('i')
            print(i)
            print('$\beta$:', listobjtspln[i].get_coeffs())
            print('$t_k$:', listobjtspln[i].get_knots())
            
            # plot the masked and detrended light curves
            axis[0].plot(timesplnregi[indxtimesplnoutt] - gdat.timetess, lcursplnbaseregi[indxtimesplnoutt], 'k.', color='k')
            axis[0].plot(timesplnregifine - gdat.timetess, listobjtspln[i](timesplnregifine)+1., 'b-', lw=3)
            axis[1].plot(timesplnregi - gdat.timetess, lcursplnbaseregi-listobjtspln[i](timesplnregi), '.', color='k')
        for a in range(2):
            axis[a].set_ylabel('Relative Flux')
        axis[0].set_xticklabels([])
        axis[1].set_xlabel('Time [BJD - 2457000]')
        path = gdat.pathobjt + 'lcur_spln.pdf'
        plt.subplots_adjust(hspace=0.)
        plt.savefig(path)
        plt.close()
   


        figr, axis = plt.subplots(4, 1, figsize=gdat.figrsize[2, :])
        axis[0].plot(gdat.timesapp - gdat.timetess, lcursapp, '.', color='k')
        axis[1].plot(gdat.timepdcc - gdat.timetess, lcurpdcc, 'k.')
        for a in range(4):
            axis[a].set_ylabel('Relative Flux')
        #axis[0].set_xticklabels([])
        
        # fit a spline to the SAP light curve
        print('Fitting a spline to the SAP light curve...')
        timesplnedge = [0., 2458504., np.inf]
        numbsplnedge = len(timesplnedge)
        numbsplnknot = numbsplnedge - 1
        indxsplnknot = np.arange(numbsplnknot)

        # produce a table for the spline coefficients
        fileoutp = open(pathdata + 'coef.csv', 'w')
        fileoutp.write(' & ')
        
        gdat.lcurspln = []
        lcursplnbase = []
        listobjtspln = []
        wdth = 4. / 24. # [days]
        for i in indxsplnknot:
            indxtimesplnregi = np.where((gdat.timesapp >= timesplnedge[i]) & (gdat.timesapp <= timesplnedge[i+1]))[0]
            timesplnregi = gdat.timesapp[indxtimesplnregi]
            lcursplnbaseregi = lcursapp[indxtimesplnregi]
            flux_err1 = stdvlcursapp[indxtimesplnregi]
            temp, indxtimesplntran, indxtimesplnoutt = index_eclipses(timesplnregi, gdat.epocprio, gdat.periprio, width_1=wdth, width_2=wdth)
            objtspln = UnivariateSpline(timesplnregi[indxtimesplnoutt], lcursplnbaseregi[indxtimesplnoutt]-1.)
            timesplnregifine = np.linspace(timesplnregi[0], timesplnregi[-1], 1000)
            gdat.lcurspln += list(lcursplnbaseregi-objtspln(timesplnregi))
            lcursplnbase += list(objtspln(timesplnregi))
            listobjtspln.append(objtspln)
        
            print('i')
            print(i)
            print('$\beta$:', listobjtspln[i].get_coeffs())
            print('$t_k$:', listobjtspln[i].get_knots())
            
            # plot the masked and detrended light curves
            axis[2].plot(timesplnregi[indxtimesplnoutt] - gdat.timetess, lcursplnbaseregi[indxtimesplnoutt], 'k.', color='k')
            axis[2].plot(timesplnregifine - gdat.timetess, listobjtspln[i](timesplnregifine)+1., 'b-', lw=3)
            axis[3].plot(timesplnregi - gdat.timetess, lcursplnbaseregi-listobjtspln[i](timesplnregi), '.', color='k')
        for a in range(2):
            axis[a].set_ylabel('Relative Flux')
        axis[2].set_xticklabels([])
        axis[3].set_xlabel('Time [BJD - 2457000]')
        path = gdat.pathobjt + 'lcur_totl.pdf'
        plt.subplots_adjust(hspace=0.)
        plt.savefig(path)
        plt.close()
   
        gdat.lcurspln = np.array(gdat.lcurspln)
        #lcursplnbase = np.array(lcursplnbase)
        temp, indxtimesplntran, indxtimesplnoutt = index_eclipses(gdat.timesapp, gdat.epocprio, gdat.periprio, width_1=4./24., width_2=2./24.)
        offset = np.mean(gdat.lcurspln[indxtimesplntran]) - 1.
        gdat.lcurspln -= offset
        
        fileoutp.write('\\hline\n')
        fileoutp.close()
    else:
        gdat.lcurspln = lcurpdcc

        gdat.time = gdat.timesapp

        ## phase-fold and save the detrended light curve
        dt = 0.01
        ferr_type = 'medsig'
        ferr_style = 'sem'
        sigmaclip = False
            
        gdat.timebind, gdat.lcursplnbind, gdat.stdvlcursplnbind, N = rebin_err(gdat.time, gdat.lcurspln, \
                                                                                    ferr_type=ferr_type, ferr_style=ferr_style, dt=dt, sigmaclip=sigmaclip)
        gdat.phasbind, gdat.pcursplnbind, gdat.stdvpcursplnbind, N, gdat.phas = phase_fold(gdat.time, gdat.lcurspln, gdat.periprio, gdat.epocprio, \
                                                                                    ferr_type=ferr_type, ferr_style=ferr_style, dt=dt, sigmaclip=sigmaclip)
        
        if False:
            data = np.column_stack((gdat.time, gdat.lcurspln, stdvlcursapp))
            path = gdat.pathobjt + 'allesfits/allesfit_spoc/TESS.csv'
            print('Writing to %s' % path)
            np.savetxt(path, data, delimiter=',', header='time,flux,flux_err')
    
    # read the allesfitter posterior
    companion = 'b'
    strginst = 'TESS'
    alles = allesfitter.allesclass(pathalle)
    allesfitter.config.init(pathalle)
    
    numbsamp = alles.posterior_params[list(alles.posterior_params.keys())[0]].size

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
        summgene(listalbgalle)
        print('listfracradi')
        summgene(listfracradi)
        print('listrsma')
        summgene(listrsma)
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
    
        if False:
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

    gdat.time = gdat.timepdcc

    # plot a lightcurve from the posteriors
    gdat.lcurmodl = alles.get_posterior_median_model(strginst, 'flux', xx=gdat.time)
    gdat.lcurgpro = alles.get_posterior_median_baseline(strginst, 'flux', xx=gdat.time)
    
    gdat.lcurdetr = gdat.lcurspln - gdat.lcurgpro

    gdat.timebind, gdat.lcurmodlbind, gdat.stdvlcurmodlbind, N = rebin_err(gdat.time, gdat.lcurmodl, \
                                                                            ferr_type=ferr_type, ferr_style=ferr_style, dt=dt, sigmaclip=sigmaclip)
    gdat.phasbind, gdat.pcurmodlbind, gdat.stdvpcurmodlbind, N, gdat.phas = phase_fold(gdat.time, gdat.lcurmodl, gdat.periprio, gdat.epocprio, \
                                                                            ferr_type=ferr_type, ferr_style=ferr_style, dt=dt, sigmaclip=sigmaclip)
    
    gdat.indxphassort = np.argsort(gdat.phas)

    gdat.timebind, gdat.lcurdetrbind, gdat.stdvlcurdetrbind, N = rebin_err(gdat.time, gdat.lcurdetr, \
                                                                            ferr_type=ferr_type, ferr_style=ferr_style, dt=dt, sigmaclip=sigmaclip)
    gdat.phasbind, gdat.pcurdetrbind, gdat.stdvpcurdetrbind, N, gdat.phas = phase_fold(gdat.time, gdat.lcurdetr, gdat.periprio, gdat.epocprio, \
                                                                            ferr_type=ferr_type, ferr_style=ferr_style, dt=dt, sigmaclip=sigmaclip)
    
    gdat.indxtimegapp = np.argmax(gdat.time[1:] - gdat.time[:-1]) + 1
    figr = plt.figure(figsize=gdat.figrsize[3, :])
    axis = [[] for k in range(3)]
    axis[0] = figr.add_subplot(3, 1, 1)
    axis[1] = figr.add_subplot(3, 1, 2)
    axis[2] = figr.add_subplot(3, 1, 3, sharex=axis[1])
    
    for k in range(len(axis)):
        
        if k > 0:
            xdat = gdat.phasbind
        else:
            xdat = gdat.time - gdat.timetess
        
        if k > 0:
            ydat = gdat.pcurdetrbind
        else:
            ydat = gdat.lcurdetr
        if k == 2:
            ydat = (ydat - 1. + medideptnigh) * 1e6
        axis[k].plot(xdat, ydat, '.', color='grey', alpha=0.3, label='Raw data')
        
        if k > 0:
            xdat = gdat.phasbind
            ydat = gdat.pcurdetrbind
            yerr = np.copy(gdat.stdvpcurdetrbind)
        else:
            xdat = gdat.timebind - gdat.timetess
            ydat = gdat.lcurdetrbind
            yerr = gdat.stdvlcurdetrbind
        if k == 2:
            ydat = (ydat - 1. + medideptnigh) * 1e6
            yerr *= 1e6
        axis[k].errorbar(xdat, ydat, marker='o', yerr=yerr, capsize=0, ls='', color='k', label='Binned data')
        
        if k > 0:
            xdat = gdat.phas[gdat.indxphassort]
            ydat = gdat.lcurmodl[gdat.indxphassort]
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
    alles = allesfitter.allesclass(pathalle)
    alles.posterior_params_median['b_sbratio_TESS'] = 0
    #alles.settings['host_shape_TESS'] = 'sphere'
    #alles.settings['b_shape_TESS'] = 'sphere'
    alles.posterior_params_median['b_geom_albedo_TESS'] = 0
    alles.posterior_params_median['host_gdc_TESS'] = 0
    alles.posterior_params_median['host_bfac_TESS'] = 0
    gdat.lcurmodlcomp = alles.get_posterior_median_model(strginst, 'flux', xx=gdat.time)
    gdat.lcurmodlevvv = np.copy(gdat.lcurmodlcomp)
    #gdat.phasbind, gdat.pcurmodlcompbind, gdat.stdvpcurmodlcompbind, N, gdat.phas = phase_fold(gdat.time, gdat.lcurmodlcomp, gdat.periprio, gdat.epocprio, \
    #                                                                        ferr_type=ferr_type, ferr_style=ferr_style, dt=dt, sigmaclip=sigmaclip)
    

    xdat = gdat.phas[gdat.indxphassort]
    ydat = (gdat.lcurmodlcomp[gdat.indxphassort] - 1.) * 1e6
    indxfrst = np.where(xdat < -0.07)[0]
    indxseco = np.where(xdat > 0.07)[0]
    axis[2].plot(xdat[indxfrst], ydat[indxfrst], lw=2, color='r', label='Ellipsoidal', ls='--')
    axis[2].plot(xdat[indxseco], ydat[indxseco], lw=2, color='r', ls='--')
    
    objtalle = allesfitter.allesclass(pathalle)
    alles.posterior_params_median['b_sbratio_TESS'] = 0
    alles.posterior_params_median['b_geom_albedo_TESS'] = 0
    alles.posterior_params_median['host_gdc_TESS'] = 0
    alles.posterior_params_median['host_bfac_TESS'] = 0

    ### planetary modulation
    alles = allesfitter.allesclass(pathalle)
    alles.posterior_params_median['b_sbratio_TESS'] = 0
    alles.settings['host_shape_TESS'] = 'sphere'
    alles.settings['b_shape_TESS'] = 'sphere'
    #alles.posterior_params_median['b_geom_albedo_TESS'] = 0
    alles.posterior_params_median['host_gdc_TESS'] = 0
    alles.posterior_params_median['host_bfac_TESS'] = 0
    gdat.lcurmodlcomp = alles.get_posterior_median_model(strginst, 'flux', xx=gdat.time)
    #gdat.phasbind, gdat.pcurmodlcompbind, gdat.stdvpcurmodlcompbind, N, gdat.phas = phase_fold(gdat.time, gdat.lcurmodlcomp, gdat.peri, gdat.epoc, \
    #                                                                        ferr_type=ferr_type, ferr_style=ferr_style, dt=dt, sigmaclip=sigmaclip)
    #axis[2].plot(gdat.phas[gdat.indxphassort], medideptnigh + (gdat.lcurmodlcomp[gdat.indxphassort] - 1.) * 1e6, \
    #                                                            lw=2, color='g', label='Planetary Modulation', ls='--', zorder=11)
    
    axis[2].legend(ncol=2)
    
    path = gdat.pathobjt + 'pcur_alle.pdf'
    plt.savefig(path)
    plt.close()
    
    
    if False:
        # plot the spherical limits
        figr, axis = plt.subplots(figsize=gdat.figrsize[0, :])
        
        alles = allesfitter.allesclass(pathalle)
        alles.posterior_params_median['b_sbratio_TESS'] = 0
        alles.settings['host_shape_TESS'] = 'sphere'
        alles.settings['b_shape_TESS'] = 'roche'
        alles.posterior_params_median['b_geom_albedo_TESS'] = 0
        alles.posterior_params_median['host_gdc_TESS'] = 0
        alles.posterior_params_median['host_bfac_TESS'] = 0
        lcurmodltemp = alles.get_posterior_median_model(strginst, 'flux', xx=gdat.time)
        axis.plot(gdat.phas[gdat.indxphassort], (gdat.lcurmodlevvv - lcurmodltemp)[gdat.indxphassort] * 1e6, lw=2, label='Spherical star')
        
        alles = allesfitter.allesclass(pathalle)
        alles.posterior_params_median['b_sbratio_TESS'] = 0
        alles.settings['host_shape_TESS'] = 'roche'
        alles.settings['b_shape_TESS'] = 'sphere'
        alles.posterior_params_median['b_geom_albedo_TESS'] = 0
        alles.posterior_params_median['host_gdc_TESS'] = 0
        alles.posterior_params_median['host_bfac_TESS'] = 0
        lcurmodltemp = alles.get_posterior_median_model(strginst, 'flux', xx=gdat.time)
        print('(gdat.lcurmodlevvv - lcurmodltemp)[gdat.indxphassort] * 1e6')
        summgene((gdat.lcurmodlevvv - lcurmodltemp)[gdat.indxphassort] * 1e6)
        axis.plot(gdat.phas[gdat.indxphassort], (gdat.lcurmodlevvv - lcurmodltemp)[gdat.indxphassort] * 1e6, lw=2, label='Spherical planet')
        axis.legend()
        axis.set_ylim([-100, 100])
        axis.set(xlabel='Phase')
        axis.set(ylabel='Relative flux [ppm]')
        plt.subplots_adjust(hspace=0.)
        path = pathimag + 'pcurmodldiff.pdf'
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
    
        # make a contour plot of geometric albedo without and with thermal component prior
        ## calculate the geometric albedo with the ATMO prior
        wlenmodl = arrymodl[:, 0]
        deptmodl = arrymodl[:, 1]
        indxwlenmodltess = np.where((wlenmodl > 0.6) & (wlenmodl < 0.95))[0]
        deptmodlther = np.mean(deptmodl[indxwlenmodltess])
    else:
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
    
    if False:
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
        path = pathimag + 'pdfn_albg.pdf'
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
        
        if False:
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
    path = pathimag + 'spec.pdf'
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
    listobjtcolr = sns.color_palette("hls", numbcomp)
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
    listobjtcolr = sns.color_palette("hls", numbcomp)
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
    path = pathimag + 'ptem.pdf'
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
        
        gdat.indxphasseco = np.where(abs(gdat.meanphas - 0.5) < dura / gdat.peri)[0]
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
            path = pathimag + 'diag/hist_%s_%s_%s_%s.pdf' % (liststrgparafull[k], modltype, strgmask, strgbins)
            plt.tight_layout()
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
        
        if samptype == 'nest':
            for keys in objtsave:
                if isinstance(objtsave[keys], np.ndarray) and objtsave[keys].size == numbsamp:
                    figr, axis = plt.subplots()
                    axis.plot(indxsamp, objtsave[keys])
                    path = pathimag + '%s_%s.pdf' % (keys, modltype)
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
            path = pathimag + 'diag/llik_%s_%s_%s.pdf' % (modltype, strgmask, strgbins)
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
        indxphasfineseco = np.where(abs(gdat.meanphasfine - 0.5) < dura / gdat.peri)[0]
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
        axis.errorbar(gdat.phasbind % 1., (gdat.pcurdetrbind - 1) * 1e6, yerr=1e6*gdat.stdvpcurdetrbind, color='k', marker='o', ls='', markersize=1)
        for k, indxsampplottemp in enumerate(indxsampplot):
            axis.plot(gdat.meanphasfine, (phasmodlfine[k, :] - 1) * 1e6, alpha=0.5, color='b')
        axis.set_xlim([0.1, 0.9])
        axis.set_ylim([-400, 1000])
        axis.set_ylabel('Relative Flux - 1 [ppm]')
        axis.set_xlabel('Phase')
        plt.tight_layout()
        path = pathimag + 'pcur_sine_%s.pdf' % strgextn
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


def cnfg_HD118203():

    strgtarg = 'HD118203'
    
    listlimttimemask = np.array([ \
                                [0, 1712], \
                                [1724.5, 1725.5], \
                                ])
    listlimttimemask += 2457000
    main( \
         strgtarg=strgtarg, \
         listlimttimemask=listlimttimemask, \
        )


def cnfg_WASP0121():

    strgtarg = 'WASP-121'

    main( \
         strgtarg=strgtarg, \
         boolphascurv=True, \
        )


def cnfg_toii1233():
    
    strgtarg = 'TOI1233'
    strgmast = '260647166'
    
    main( \
         #strgdata='qlop', \
         strgtarg=strgtarg, \
         strgmast=strgmast, \
        )


def cnfg_toii0203():

    strgtarg = 'TOI203'
    strgmast = '259962054'
    main( \
         strgdata='tcat', \
         strgtarg=strgtarg, \
         strgmast=strgmast, \
        )


if __name__ == "__main__":
    
    globals().get(sys.argv[1])(*sys.argv[2:])

