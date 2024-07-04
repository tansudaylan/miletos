import time as modutime

import os, fnmatch
import sys, datetime
import numpy as np
import scipy.interpolate
import scipy.stats

from tqdm import tqdm

from numba import jit, prange

import pandas as pd

import h5py

import astroquery

import astropy
import astropy.coordinates
import astropy.units
from astropy.coordinates import SkyCoord

import pickle
    
import celerite

from functools import partial

import matplotlib
import matplotlib.pyplot as plt

import tdpy
from tdpy.util import summgene
import nicomedia
import lygos
import ephesos

"""
Given a target, miletos is an time-domain astronomy tool that allows 
1) automatic search for, download and process TESS and Kepler data via MAST or use user-provided data
2) impose priors based on custom inputs, ExoFOP or NASA Exoplanet Archive
3) model radial velocity and photometric time-series data on N-body systems
4) Make characterization plots of the target after the analysis
"""

def retr_timetran(gdat, nametser):
    '''
    Determine times during transits
    '''
    
    print('Determining times during transits...')
    
    gdat.listindxtimeoutt = [[[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser] for j in gdat.fitt.prio.indxcomp]
    gdat.listindxtimetranindi = [[[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser] for j in gdat.fitt.prio.indxcomp]
    gdat.listindxtimetran = [[[[[] for m in range(2)] for p in gdat.indxinst[b]] for b in gdat.indxdatatser] for j in gdat.fitt.prio.indxcomp]
    gdat.listindxtimetranchun = [[[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser] for j in gdat.fitt.prio.indxcomp]
    gdat.numbtimeclen = [[np.empty((gdat.fitt.prio.numbcomp), dtype=int) for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    
    print('Determining indices...')
    gdat.numbtran = np.empty(gdat.fitt.prio.numbcomp, dtype=int)
    for j in gdat.fitt.prio.indxcomp:
        gdat.listtimeconc = []
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                
                if not np.isfinite(gdat.fitt.prio.meanpara.duratrantotlcomp[j]):
                    continue
                
                # determine time mask
                for y in gdat.indxchun[b][p]:
                
                    if len(gdat.listarrytser[nametser][b][p][y]) == 0:
                        continue

                    gdat.listindxtimetranchun[j][b][p][y] = retr_indxtimetran(gdat.listarrytser[nametser][b][p][y][:, 0, 0], \
                                                        gdat.fitt.prio.meanpara.epocmtracomp[j], gdat.fitt.prio.meanpara.pericomp[j], gdat.fitt.prio.meanpara.duratrantotlcomp[j])
                
                # primary
                gdat.listindxtimetran[j][b][p][0] = retr_indxtimetran(gdat.arrytser[nametser][b][p][:, 0, 0], \
                                                        gdat.fitt.prio.meanpara.epocmtracomp[j], gdat.fitt.prio.meanpara.pericomp[j], gdat.fitt.prio.meanpara.duratrantotlcomp[j])
                
                # primary individuals
                gdat.listindxtimetranindi[j][b][p] = retr_indxtimetran(gdat.arrytser[nametser][b][p][:, 0, 0], \
                                              gdat.fitt.prio.meanpara.epocmtracomp[j], gdat.fitt.prio.meanpara.pericomp[j], gdat.fitt.prio.meanpara.duratrantotlcomp[j], boolindi=True)
                
                # secondary
                gdat.listindxtimetran[j][b][p][1] = retr_indxtimetran(gdat.arrytser[nametser][b][p][:, 0, 0], \
                                              gdat.fitt.prio.meanpara.epocmtracomp[j], gdat.fitt.prio.meanpara.pericomp[j], gdat.fitt.prio.meanpara.duratrantotlcomp[j], boolseco=True)
                
                gdat.listindxtimeoutt[j][b][p] = np.setdiff1d(np.arange(gdat.arrytser[nametser][b][p].shape[0]), gdat.listindxtimetran[j][b][p][0])
                gdat.numbtimeclen[b][p][j] = gdat.listindxtimeoutt[j][b][p].size
                
                gdat.listtimeconc.append(gdat.arrytser[nametser][b][p][:, 0, 0])
        
        if len(gdat.listtimeconc) > 0:
            gdat.listtimeconc = np.concatenate(gdat.listtimeconc)
            gdat.listindxtran = retr_indxtran(gdat.listtimeconc, gdat.fitt.prio.meanpara.epocmtracomp[j], \
                                                            gdat.fitt.prio.meanpara.pericomp[j], gdat.fitt.prio.meanpara.duratrantotlcomp[j])
            gdat.numbtran[j] = len(gdat.listindxtran)
    
    # indices of times outside the transit for each companion
    # probably to be deleted since gdat.listindxtimeoutt is the same as gdat.listindxtimeoutt?
    #for b in gdat.indxdatatser:
    #    for p in gdat.indxinst[b]:
    #        for j in gdat.fitt.prio.indxcomp:
    #            listindxtimetemp = []
    #            for jj in gdat.fitt.prio.indxcomp:
    #                if jj != j:
    #                    listindxtimetemp.append(gdat.listindxtimetran[jj][b][p][0])
    #            if len(listindxtimetemp) > 0:
    #                listindxtimetemp = np.concatenate(listindxtimetemp)
    #                listindxtimetemp = np.unique(listindxtimetemp)
    #            else:
    #                listindxtimetemp = np.array([])
    #            gdat.listindxtimeoutt[j][b][p] = np.setdiff1d(np.arange(gdat.arrytser[nametser][b][p].shape[0]), listindxtimetemp)
                
    # ingress and egress times
    if gdat.fitt.typemodl == 'psysdisktran':
        gdat.fracineg = np.zeros(2)
        
        gdat.listindxtimetranineg = [[[[[] for k in range(4)] for pk in gdat.indxrratband[b]] for b in gdat.indxdatatser] for j in gdat.fitt.prio.indxcomp]

        for pk in gdat.indxanlsband:
            for j in gdat.fitt.prio.indxcomp:
                gdat.durafullprio = (1. - gdat.fitt.prio.meanpara.rratcomp[pk][j]) / (1. + gdat.fitt.prio.meanpara.rratcomp[pk][j]) * gdat.fitt.prio.meanpara.duratrantotlcomp
                
                if not gdat.fitt.prio.booltrancomp[j]:
                    continue

                gdat.listindxtimetranineg[j][0][pp][0] = retr_indxtimetran(gdat.arrytser[nametser][0][pp][:, 0, 0], \
                                                                                  gdat.fitt.prio.meanpara.epocmtracomp[j], gdat.fitt.prio.meanpara.pericomp[j], \
                                                                                  gdat.fitt.prio.meanpara.duratrantotlcomp[j], durafull=gdat.durafullprio[j], typeineg='ingrinit')
                gdat.listindxtimetranineg[j][0][pp][1] = retr_indxtimetran(gdat.arrytser[nametser][0][pp][:, 0, 0], \
                                                                                  gdat.fitt.prio.meanpara.epocmtracomp[j], gdat.fitt.prio.meanpara.pericomp[j], \
                                                                                  gdat.fitt.prio.meanpara.duratrantotlcomp[j], durafull=gdat.durafullprio[j], typeineg='ingrfinl')
                gdat.listindxtimetranineg[j][0][pp][2] = retr_indxtimetran(gdat.arrytser[nametser][0][pp][:, 0, 0], \
                                                                                  gdat.fitt.prio.meanpara.epocmtracomp[j], gdat.fitt.prio.meanpara.pericomp[j], \
                                                                                  gdat.fitt.prio.meanpara.duratrantotlcomp[j], durafull=gdat.durafullprio[j], typeineg='eggrinit')
                gdat.listindxtimetranineg[j][0][pp][3] = retr_indxtimetran(gdat.arrytser[nametser][0][pp][:, 0, 0], \
                                                                                  gdat.fitt.prio.meanpara.epocmtracomp[j], gdat.fitt.prio.meanpara.pericomp[j], \
                                                                                  gdat.fitt.prio.meanpara.duratrantotlcomp[j], durafull=gdat.durafullprio[j], typeineg='eggrfinl')
                
                for k in range(2):
                    indxtimefrst = gdat.listindxtimetranineg[j][0][pp][2*k+0]
                    indxtimeseco = gdat.listindxtimetranineg[j][0][pp][2*k+1]
                    if indxtimefrst.size == 0 or indxtimeseco.size == 0:
                        continue
                    rflxinit = np.mean(gdat.arrytser[nametser][0][pp][indxtimefrst, 1])
                    rflxfinl = np.mean(gdat.arrytser[nametser][0][pp][indxtimeseco, 1])
                    gdat.fracineg[k] = rflxinit / rflxfinl
                
                if (gdat.fracineg == 0).any():
                    print('')
                    print('')
                    print('')
                    print('rflxinit')
                    print(rflxinit)
                    print('rflxfinl')
                    print(rflxfinl)
                    print('gdat.arrytser[bdtr][0][p]')
                    summgene(gdat.arrytser[nametser][0][pp])
                    print('gdat.arrytser[bdtrnotr][0][pp][:, 1]')
                    summgene(gdat.arrytser[nametser][0][pp][:, 1])
                    print('gdat.arrytser[bdtrnotr][0][pp][indxtimefrst, 1]')
                    summgene(gdat.arrytser[nametser][0][pp][indxtimefrst, 1])
                    print('gdat.arrytser[bdtrnotr][0][pp][indxtimeseco, 1]')
                    summgene(gdat.arrytser[nametser][0][pp][indxtimeseco, 1])
                    raise Exception('(gdat.fracineg == 0).any()')

                path = gdat.pathdatatarg + 'fracineg%04d.csv' % j
                np.savetxt(path, gdat.fracineg, delimiter=',')
                gdat.dictmileoutp['fracineg%04d' % j] = gdat.fracineg


def retr_listtypeanls(typesyst, typepriocomp):
    
    listtypeanls = []
    if (typesyst == 'PlanetarySystem' or typesyst == 'PlanetarySystemEmittingCompanion') and typepriocomp == 'outlperi':
        listtypeanls += ['outlperi']
    if (typesyst == 'PlanetarySystem' or typesyst == 'PlanetarySystemEmittingCompanion') and typepriocomp == 'boxsperinega':
        listtypeanls += ['boxsperinega']
    if typesyst == 'CompactObjectStellarCompanion':
        listtypeanls += ['boxsperiposi']
    if typesyst == 'CompactObjectStellarCompanion' or typesyst == 'SpottedStar':
        listtypeanls += ['lspe']
    
    return listtypeanls


def retr_lliknegagpro(listparagpro, lcur, objtgpro):
    '''
    Compute the negative loglikelihood of the GP model
    '''
    
    objtgpro.set_parameter_vector(listparagpro)
    
    return -objtgpro.log_likelihood(lcur)


def retr_gradlliknegagpro(listparagpro, lcur, objtgpro):
    '''
    Compute the gradient of the negative loglikelihood of the GP model
    '''
    
    objtgpro.set_parameter_vector(listparagpro)
    
    return -objtgpro.grad_log_likelihood(lcur)[1]


def retr_tsecpathlocl( \
                      tici, \
                      
                      # type of verbosity
                      ## -1: absolutely no text
                      ##  0: no text output except critical warnings
                      ##  1: minimal description of the execution
                      ##  2: detailed description of the execution
                      typeverb=1, \
                     ):
    '''
    Retrieve the list of TESS sectors for which SPOC light curves are available for target in the local database of predownloaded light curves
    '''
    
    pathbase = os.environ['TESS_DATA_PATH'] + '/data/lcur/'
    path = pathbase + 'tsec/tsec_spoc_%016d.csv' % tici
    if not os.path.exists(path):
        listtsecsele = np.arange(1, 60)
        listpath = []
        listtsec = []
        strgtagg = '*-%016d-*.fits' % tici
        for tsec in listtsecsele:
            pathtemp = pathbase + 'sector-%02d/' % tsec
            listpathtemp = fnmatch.filter(os.listdir(pathtemp), strgtagg)
            
            if len(listpathtemp) > 0:
                listpath.append(pathtemp + listpathtemp[0])
                listtsec.append(tsec)
        
        listtsec = np.array(listtsec).astype(int)
        print('Writing to %s...' % path)
        objtfile = open(path, 'w')
        for k in range(len(listpath)):
            objtfile.write('%d,%s\n' % (listtsec[k], listpath[k]))
        objtfile.close()
    else:
        if typeverb > 0:
            print('Reading from %s...' % path)
        objtfile = open(path, 'r')
        listtsec = []
        listpath = []
        for line in objtfile:
            linesplt = line.split(',')
            listtsec.append(linesplt[0])
            listpath.append(linesplt[1][:-1])
        listtsec = np.array(listtsec).astype(int)
        objtfile.close()
    
    return listtsec, listpath


def retr_listtsectcut(strgtcut):
    '''
    Retrieve the list of sectors, cameras, and CCDs for which TESS data are available for the target.
    '''
    
    print('Calling TESSCut with keyword %s to get the list of sectors for which TESS data are available...' % strgtcut)
    tabltesscutt = astroquery.mast.Tesscut.get_sectors(coordinates=strgtcut, radius=0)

    listtsec = np.array(tabltesscutt['sector'])
    listtcam = np.array(tabltesscutt['camera'])
    listtccd = np.array(tabltesscutt['ccd'])
    
    print('listtsec')
    print(listtsec)

    return listtsec, listtcam, listtccd


def pars_para_mile(para, gdat, strgmodl):
    
    dictparainpt = dict()
    
    gmod = getattr(gdat, strgmodl)
    
    #if gdat.fitt.typemodlenerfitt == 'full':
    #    dictparainpt['consblin'] = para[gmod.dictindxpara['consblin']]
    #else:
    #    for nameparabase in gmod.listnameparabase:
    #        strg = nameparabase + gdat.liststrgdatafittiter[gdat.indxfittiterthis]
    #        if hasattr(gmod, strg):
    #            dictparainpt[strg] = getattr(gmod, strg)
    #        else:
    #            dictparainpt[strg] = para[gmod.dictindxpara[strg]]
    
    for name in gmod.listnameparafullfixd:
        #print('Found fixed value for parameter %s...' % name)
        dictparainpt[name] = getattr(gmod, name)
    
    for name in gmod.listnameparafullvari:
        
        dictparainpt[name] = para[gmod.dictindxpara[name]]
        if gdat.booldiag:

            if isinstance(gmod.dictindxpara[name], int) and gmod.dictindxpara[name] > 1e6 or \
                        not isinstance(gmod.dictindxpara[name], int) and (gmod.dictindxpara[name] > 1e6).any():
                print('')
                print('')
                print('')
                print('name')
                print(name)
                print('gmod.dictindxpara[name]')
                print(gmod.dictindxpara[name])
                raise Exception('')
    
    if gmod.boolmodlpsys:
        for name in ['radistar', 'masscomp', 'massstar']:
            dictparainpt[name] = None

    if gmod.typemodlblinshap == 'GaussianProcess':
        for name in ['sigmgprobase', 'rhoogprobase']:
            dictparainpt[name] = para[gmod.dictindxpara[name]]
    
    return dictparainpt


def retr_dictmodl_mile(gdat, time, dictparainpt, strgmodl):
    
    gmod = getattr(gdat, strgmodl)
    
    if gdat.booldiag:
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                
                if strgmodl == 'true' and gdat.liststrgtypedata[b][p] == 'obsd':
                    continue

                if isinstance(time[b][p], list):
                    print('')
                    print('')
                    print('')
                    print('b, p')
                    print(b, p)
                    print('time[b][p]')
                    summgene(time[b][p])
                    raise Exception('isinstance(time[b][p], list)')

                if time[b][p].ndim != 1:
                    print('')
                    print('')
                    print('')
                    print('b, p')
                    print(b, p)
                    print('time[b][p]')
                    summgene(time[b][p])
                    raise Exception('time[b][p].ndim != 1')

    dictlistmodl = dict()
    for name in gmod.listnamecompmodl:
        dictlistmodl[name] = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    
    if gmod.typemodl == 'StarFlaring':
        for p in gdat.indxinst[0]:
            
            if strgmodl == 'true' and gdat.liststrgtypedata[b][p] == 'obsd':
                continue

            rflxmodl = np.zeros((time[0][p].size, gdat.numbener[p], gmod.numbflar))
            for kk in range(gmod.numbflar):
                strgflar = '%04d' % kk
                timeflar = dictparainpt['timeflar%s' % strgflar]
                amplflar = dictparainpt['amplflar%s' % strgflar]
                tsclflar = dictparainpt['tsclflar%s' % strgflar]
                
                timediff = time[0][p] - timeflar
                #indxtime = np.where((timediff < 10 * tsclflar / 24.) & (timediff > -3 * tsclflar / 24.))[0]
                indxtime = np.where((timediff < 10 * tsclflar / 24.) & (timediff > 0))[0]
                
                if indxtime.size > 0:
                    rflxmodl[indxtime, 0, kk] = amplflar * np.exp(-(time[0][p][indxtime] - timeflar) / (tsclflar / 24.))
            
            dictlistmodl['StarFlaring'][0][p] = 1. + np.sum(rflxmodl, -1)
    
    timeredu = None
                            
    if gmod.typemodl.startswith('PlanetarySystem') or gmod.typemodl == 'CompactObjectStellarCompanion':
        
        timeredu = np.empty(gdat.numbenermodl)
        
        for p in gdat.indxinst[0]:
            if strgmodl == 'fitt' and gdat.fitt.typemodlenerfitt == 'full' or strgmodl == 'true':
                numbener = gdat.numbener[p]
            else:
                numbener = 1
            dictlistmodl['Transit'][0][p] = np.empty((time[0][p].size, numbener))
            #dictlistmodl['Signal'][0][p] = np.empty((time[0][p].size, numbener))
        
        # temp
        pericomp = np.empty(gmod.numbcomp)
        rsmacomp = np.empty(gmod.numbcomp)
        epocmtracomp = np.empty(gmod.numbcomp)
        cosicomp = np.empty(gmod.numbcomp)
        if gmod.typemodl == 'CompactObjectStellarCompanion':
            masscomp = np.empty(gmod.numbcomp)
        
        for j in gmod.indxcomp:
            pericomp[j] = dictparainpt['pericom%d' % j]
            rsmacomp[j] = dictparainpt['rsmacom%d' % j]
            epocmtracomp[j] = dictparainpt['epocmtracom%d' % j]
            cosicomp[j] = dictparainpt['cosicom%d' % j]
        if gmod.typemodl == 'CompactObjectStellarCompanion':
            for j in gmod.indxcomp:
                masscomp[j] = dictparainpt['masscom%d' % j]
            massstar = dictparainpt['massstar']
            radistar = dictparainpt['massstar']
        else:
            masscomp = None
            massstar = None
            radistar = None
        
        if gdat.booldiag:
            if cosicomp.size != pericomp.size:
                print('')
                print('pericomp')
                summgene(pericomp)
                print('cosicomp')
                summgene(cosicomp)
                raise Exception('')
        
        cntr = 0
        for p in gdat.indxinst[0]:
            
            if strgmodl == 'true' and gdat.liststrgtypedata[0][p] == 'obsd':
                continue

            # limb darkening
            if gmod.typemodllmdkener == 'ener':
                coeflmdk = np.empty((2, gdat.numbener[p]))
                for e in gdat.indxener[p]:
                    coeflmdk[0, e] = dictparainpt['coeflmdklinr' + gdat.liststrgener[p][e]]
                    coeflmdk[1, e] = dictparainpt['coeflmdkquad' + gdat.liststrgener[p][e]]
            elif gmod.typemodllmdkener == 'linr':
                coeflmdk = np.empty((2, gdat.numbener[p]))
            elif gmod.typemodllmdkener == 'cons':
                coeflmdklinr = dictparainpt['coeflmdklinr']
                coeflmdkquad = dictparainpt['coeflmdkquad']
                coeflmdk = np.array([coeflmdklinr, coeflmdkquad])

            if gmod.typemodllmdkener == 'line':
                coeflmdk *= ratiline
            
            if gmod.boolmodlpsys:
                if gdat.numbener[p] > 1:
                    rratcomp = np.empty((gmod.numbcomp, gdat.numbeneriter))
                    for e in gdat.indxener[p]:
                        for j in gmod.indxcomp:
                            rratcomp[j, e] = np.array([dictparainpt['rratcom%d%s' % (j, gdat.liststrgener[p][e])]])
                else:
                    rratcomp = np.empty(gmod.numbcomp)
                    for j in gmod.indxcomp:
                        rratcomp[j] = dictparainpt['rratcom%d' % j]
                
                cntr += time[0][p].size

            else:
                rratcomp = None
        
        if gmod.boolmodlpsys:
            
            if strgmodl == 'true':
                boolmakeanim = gdat.boolmakeanimefestrue
            else:
                boolmakeanim = False
            
            boolmakeimaglfov = False
            pathvisu = None
            if strgmodl == 'true':
                if gdat.boolplotefestrue:
                    pathvisu = gdat.pathvisutarg + 'EphesosOutputForSimulatedData/'
                    boolmakeimaglfov = True
                    os.system('mkdir -p %s' % pathvisu)

            if gdat.booldiag:
                if len(rratcomp) == 0 and gmod.typemodl.startswith('PlanetarySystem') and gmod.numbcomp > 1:
                    print('')
                    print('')
                    print('')
                    print('gmod.typemodl')
                    print(gmod.typemodl)
                    print('rratcomp')
                    print(rratcomp)
                    raise Exception('len(rratcomp) == 0')

            for p in gdat.indxinst[0]:
                
                strgextn = '%s' % gdat.liststrginst[0][p]
                
                dictoutpmodl = ephesos.eval_modl(time[0][p], \
                                                 pericomp=pericomp, \
                                                 epocmtracomp=epocmtracomp, \
                                                 rsmacomp=rsmacomp, \
                                                 cosicomp=cosicomp, \
                                                 
                                                 massstar=massstar, \
                                                 radistar=radistar, \
                                                 masscomp=masscomp, \
                                                 
                                                 boolmakeanim=boolmakeanim, \
                                                 pathvisu=pathvisu, \
                                                 boolmakeimaglfov=boolmakeimaglfov, \

                                                 typelmdk='quadkipp', \
                                                 
                                                 booldiag=gdat.booldiag, \

                                                 coeflmdk=coeflmdk, \
                                                 
                                                 strgextn=strgextn, \

                                                 rratcomp=rratcomp, \
                                                 typesyst=gmod.typemodl, \
                                                 
                                                 typeverb=0, \
                                                
                                                )
                
                dictlistmodl['Transit'][0][p] = dictoutpmodl['rflx']
                if dictlistmodl['Transit'][0][p].ndim == 1:
                    dictlistmodl['Transit'][0][p] = dictlistmodl['Transit'][0][p][:, None]

                if gdat.booldiag:
                    if np.amin(dictlistmodl['Transit'][0][p]) < 0 or dictlistmodl['Transit'][0][p].ndim == 1:
                        print('')
                        print('')
                        print('')
                        print('dictlistmodl[tran][0][p]')
                        summgene(dictlistmodl['Transit'][0][p])
                        print('rratcomp')
                        summgene(rratcomp)
                        print('WARNING! dictlistmodl[tran][0][p] has gone negative.')
                        #raise Exception('dictlistmodl[tran][0][p] has gone negative.')
                
                timeredu = dictoutpmodl['timeredu']

    # baseline
    if gmod.typemodlblinshap == 'cons':
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                
                if strgmodl == 'true' and gdat.liststrgtypedata[b][p] == 'obsd':
                    continue

                dictlistmodl['Baseline'][b][p] = np.ones((time[b][p].size, gdat.numbener[p]))
                if gdat.numbener[p] > 1 and gmod.typemodlblinener[p] == 'ener':
                    for e in gdat.indxener[p]:
                        dictlistmodl['Baseline'][b][p][:, e] += dictparainpt['consblinener%04d' % e] * 1e-3 * np.ones_like(time[b][p])
                else:
                    dictlistmodl['Baseline'][b][p][:, 0] += dictparainpt['consblin%s' % gdat.liststrginst[b][p]] * 1e-3 * np.ones_like(time[b][p])
    
    elif gmod.typemodlblinshap == 'step':
        for p in gdat.indxinst[0]:
            rflxbase = np.ones_like(dictlistmodl['Signal'][0][p])
            if gdat.fitt.typemodlenerfitt == 'full':
                consfrst = dictparainpt['consblinfrst'][None, :] * 1e-3
                consseco = dictparainpt['consblinseco'][None, :] * 1e-3
                timestep = dictparainpt['timestep'][None, :]
            
            else:
                consfrst = np.full((time[0][p].size, 1), dictparainpt['consblinfrst' + gdat.liststrgdatafittiter[gdat.indxfittiterthis]]) * 1e-3
                consseco = np.full((time[0][p].size, 1), dictparainpt['consblinseco' + gdat.liststrgdatafittiter[gdat.indxfittiterthis]]) * 1e-3
                timestep = np.full((time[0][p].size, 1), dictparainpt['timestep' + gdat.liststrgdatafittiter[gdat.indxfittiterthis]])
                scalstep = np.full((time[0][p].size, 1), dictparainpt['scalstep' + gdat.liststrgdatafittiter[gdat.indxfittiterthis]])
                
            dictlistmodl['Baseline'][0][p] = (consseco - consfrst) / (1. + np.exp(-(time[0][p][:, None] - timestep) / scalstep)) + consfrst
    
    if gdat.booldiag:
        if len(dictlistmodl['Baseline'][0][p]) == 0:
            print('')
            print('')
            print('')
            print('gdat.fitt.typemodlenerfitt')
            print(gdat.fitt.typemodlenerfitt)
            print('gmod.typemodlblinshap')
            print(gmod.typemodlblinshap)
            raise Exception('dictlistmodl[blin][0][p] is empty.')
    
    if gmod.typemodlblinshap != 'GaussianProcess':
        for p in gdat.indxinst[0]:
            # total model
            if gmod.typemodl.startswith('PlanetarySystem') or gmod.typemodl == 'CompactObjectStellarCompanion':
                sgnl = dictlistmodl['Transit'][0][p]
            
                if gdat.booldiag:
                    if dictlistmodl['Transit'][0][p].ndim != dictlistmodl['Baseline'][0][p].ndim:
                        print('')
                        print('')
                        print('')
                        print('Signal')
                        summgene(sgnl)
                        print('dictlistmodl[blin][0][p]')
                        summgene(dictlistmodl['Baseline'][0][p])
                        raise Exception('dictlistmodl[tran][0][p].ndim != dictlistmodl[blin][0][p]')
            
            elif gmod.typemodl == 'StarFlaring':
                sgnl = dictlistmodl['StarFlaring'][0][p]
            else:
                print('')
                print('')
                print('')
                print('gmod.typemodl')
                print(gmod.typemodl)
                raise Exception('')

            dictlistmodl['Total'][0][p] = sgnl + dictlistmodl['Baseline'][0][p] - 1.
    if gdat.booldiag:
        for name in dictlistmodl.keys():
            for b in gdat.indxdatatser:
                for p in gdat.indxinst[b]:
                    if not np.isfinite(dictlistmodl[name][0][p]).all():
                        print('')
                        print('')
                        print('')
                        print('gmod.typemodlblinshap')
                        print(gmod.typemodlblinshap)
                        print('time[0][p]')
                        summgene(time[0][p])
                        print('name')
                        print(name)
                        print('dictlistmodl[name][0][p]')
                        summgene(dictlistmodl[name][0][p])
                        raise Exception('not np.isfinite(dictlistmodl[name][0][p]).all()')
                    
                    if dictlistmodl[name][0][p].shape[0] != time[0][p].size:
                        print('')
                        print('')
                        print('')
                        print('name')
                        print(name)
                        print('time[0][p]')
                        summgene(time[0][p])
                        print('dictlistmodl[name][0][p]')
                        summgene(dictlistmodl[name][0][p])
                        raise Exception('dictlistmodl[name][0][p].shape[0] != time[0][p].size')
        
    if gmod.typemodl.startswith('PlanetarySystem') or gmod.typemodl == 'CompactObjectStellarCompanion':
        for p in gdat.indxinst[1]:
            dictlistmodl[1][p] = retr_rvel(time[1][p], dictparainpt['epocmtracomp'], dictparainpt['pericomp'], dictparainpt['masscomp'], \
                                                dictparainpt['massstar'], dictparainpt['inclcomp'], dictparainpt['eccecomp'], dictparainpt['argupericomp'])
    
    if gdat.booldiag:
        for namecompmodl in gmod.listnamecompmodl:
            for b in gdat.indxdatatser:
                for p in gdat.indxinst[b]:
                    if time[b][p].size != dictlistmodl[namecompmodl][b][p].shape[0]:
                        print('')
                        print('')
                        print('')
                        print('')
                        print('namecompmodl')
                        print(namecompmodl)
                        print('dictlistmodl[namecompmodl][b][p]')
                        summgene(dictlistmodl[namecompmodl][b][p])
                        print('time[b][p]')
                        summgene(time[b][p])
                        print('np.unique(gdat.time[b][p])')
                        summgene(np.unique(time[b][p]))
                        raise Exception('')

    return dictlistmodl, timeredu


def retr_rflxmodl_mile_gpro(gdat, strgmodl, timemodl, dictparainpt, timemodleval=None, rflxmodl=None):
    
    dictobjtkern, dictobjtgpro = setp_gpro(gdat, dictparainpt, strgmodl)
    dictmodl = dict()
    for name in gdat.listnamecompgpro:
        if name == 'excs':
            indxtimedata = np.where((timeoffsdata > 0) & (timeoffsdata < 2.))[0]
            indxtimemodl = np.where((timeoffsmodl > 0) & (timeoffsmodl < 2.))[0]
        if name == 'Total':
            indxtimedata = np.arange(gdat.timethisfitt[b][p].size)
            indxtimemodl = np.arange(timemodl.size)
        
        dictmodl[name] = np.ones((timemodl.size, gdat.numbener[p]))
        if timemodleval is not None:
            dictmodleval[name] = np.ones((timemodl.size, gdat.numbener[p]))
        
        if strgmodl == 'true':
            for e in gdat.indxenermodl:
                # compute the covariance matrix
                dictobjtgpro[name].compute(gdat.timethisfitt[b][p][indxtimedata])
                # get the GP model mean baseline
                dictmodl[name][indxtimemodl, e] = 1. + dictobjtgpro[name].sample()
                
        else:
            for e in gdat.indxenermodl:
                # compute the covariance matrix
                dictobjtgpro[name].compute(gdat.timethisfitt[b][p][indxtimedata], yerr=gdat.stdvrflxthisfittsele[indxtimedata, e])
                # get the GP model mean baseline
                dictmodl[name][indxtimemodl, e] = 1. + dictobjtgpro[name].predict(gdat.rflxthisfittsele[indxtimedata, e] - rflxmodl[indxtimedata, e], \
                                                                                                        t=timemodl[indxtimemodl], return_cov=False, return_var=False)
                if timemodleval is not None:
                    pass

    return dictmodl, dictmodleval


def setp_gpro(gdat, dictparainpt, strgmodl):
    
    dictobjtkern = dict()
    dictobjtgpro = dict()

    gmod = getattr(gdat, strgmodl)
    
    ## construct the kernel object
    if gmod.typemodlblinshap == 'GaussianProcess':
        dictobjtkern['Baseline'] = celerite.terms.Matern32Term(log_sigma=np.log(dictparainpt['sigmgprobase']*1e-3), log_rho=np.log(dictparainpt['rhoogprobase']))
    
    k = 0
    for name, valu in dictobjtkern.items():
        if k == 0:
            objtkerntotl = valu
        else:
            objtkerntotl += valu
        k += 1
    if dictobjtkern is not None:
        dictobjtkern['Total'] = objtkerntotl
    
    ## construct the GP model object
    for name in dictobjtkern:
        dictobjtgpro[name] = celerite.GP(dictobjtkern[name])
    
    return dictobjtkern, dictobjtgpro


def retr_llik_mile(para, gdat):
    
    """
    Return the likelihood.
    """
    
    gmod = gdat.fitt
    
    dictparainpt = pars_para_mile(para, gdat, 'fitt')
    dictmodl = retr_dictmodl_mile(gdat, gdat.timethisfitt, dictparainpt, 'fitt')[0]
    
    llik = 0.
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            
            if gdat.fitt.typemodlenerfitt == 'full':
                gdat.rflxthisfittsele = gdat.rflxthisfitt[b][p]
                gdat.varirflxthisfittsele = gdat.varirflxthisfitt[b][p]
                gdat.stdvrflxthisfittsele = gdat.stdvrflxthisfitt[b][p]
            else:
                gdat.rflxthisfittsele = gdat.rflxthisfitt[b][p][:, gdat.fitt.listindxinstener[p]]
                gdat.varirflxthisfittsele = gdat.varirflxthisfitt[b][p][:, gdat.fitt.listindxinstener[p]]
                gdat.stdvrflxthisfittsele = gdat.stdvrflxthisfitt[b][p][:, gdat.fitt.listindxinstener[p]]
            
            if gdat.booldiag:
                if gdat.rflxthisfittsele.ndim != dictmodl['Total'][b][p].ndim:
                    print('')
                    print('gdat.rflxthisfittsele')
                    summgene(gdat.rflxthisfittsele)
                    raise Exception('')
                if gdat.rflxthisfittsele.shape[0] != dictmodl['Total'][b][p].shape[0]:
                    print('')
                    print('gdat.rflxthisfittsele')
                    summgene(gdat.rflxthisfittsele)
                    raise Exception('')
                
            if gdat.typellik == 'GaussianProcess':
                
                for e in gdat.indxenermodl:
                    
                    resitemp = gdat.rflxthisfittsele[:, e] - dictmodl['Total'][0][p][:, e]
                    
                    # construct a Gaussian Process (GP) model
                    dictobjtkern, dictobjtgpro = setp_gpro(gdat, dictparainpt, 'fitt')
                
                    # compute the covariance matrix
                    dictobjtgpro['Total'].compute(gdat.timethisfitt[b][p], yerr=gdat.stdvrflxthisfittsele[:, e])
                
                    # get the initial parameters of the GP model
                    #parainit = objtgpro.get_parameter_vector()
                    
                    # get the bounds on the GP model parameters
                    #limtparagpro = objtgpro.get_parameter_bounds()
                    
                    # minimize the negative loglikelihood
                    #objtmini = scipy.optimize.minimize(retr_lliknegagpro, parainit, jac=retr_gradlliknegagpro, method="L-BFGS-B", \
                    #                                                                 bounds=limtparagpro, args=(lcurregi[indxtimeregioutt[i]], objtgpro))
                    
                    #print('GP Matern 3/2 parameters with maximum likelihood:')
                    #print(objtmini.x)

                    # update the GP model with the parameters that minimize the negative loglikelihood
                    #objtgpro.set_parameter_vector(objtmini.x)
                    
                    # get the GP model mean baseline
                    #lcurbase = objtgpro.predict(lcurregi[indxtimeregioutt[i]], t=timeregi, return_cov=False, return_var=False)#[0]
                    
                    # subtract the baseline from the data
                    #lcurbdtrregi[i] = 1. + lcurregi - lcurbase

                    #listobjtspln[i] = objtgpro
                    #gp.compute(gdat.time[0], yerr=gdat.stdvrflxthisfittsele)
                
                    llik += dictobjtgpro['Total'].log_likelihood(resitemp)
                    
                    #print('resitemp')
                    #summgene(resitemp)
                    #print('dictobjtkern')
                    #print(dictobjtkern)

            if gdat.typellik == 'sing':
                
                gdat.lliktemp[b][p] = -0.5 * (gdat.rflxthisfittsele - dictmodl['Total'][b][p])**2 / gdat.varirflxthisfittsele
                
                if gdat.boolrejeoutlllik:
                    #gdat.lliktemp[b][p] = np.sort(gdat.lliktemp.flatten())[1:]
                    gdat.lliktemp[b][p][0, 0] -= np.amin(gdat.lliktemp)
                
            llik += np.sum(gdat.lliktemp[b][p])
    
    if gdat.booldiag:
        if gdat.typellik == 'sing' and llik.size != 1:
            print('gdat.fitt.typemodlenerfitt')
            print(gdat.fitt.typemodlenerfitt)
            print('gdat.rflxthisfittsele')
            summgene(gdat.rflxthisfittsele)
            print('gdat.varirflxthisfittsele')
            summgene(gdat.varirflxthisfittsele)
            print('llik')
            print(llik)
            raise Exception('')
        if not np.isfinite(llik):
            print('')
            print('gdat.typellik')
            print(gdat.typellik)
            print('dictparainpt')
            print(dictparainpt)
            print('gdat.varirflxthisfittsele')
            summgene(gdat.varirflxthisfittsele)
            print('gdat.rflxthisfittsele')
            summgene(gdat.rflxthisfittsele)
            print('gdat.fitt.typemodlenerfitt')
            print(gdat.fitt.typemodlenerfitt)
            raise Exception('')

    return llik


def retr_lliknega_mile(para, gdat):
    
    llik = retr_llik_mile(para, gdat)
    
    return -llik


def retr_dictderi_mile(para, gdat):
    
    gmod = getattr(gdat, gdat.thisstrgmodl)

    dictparainpt = pars_para_mile(para, gdat, 'fitt')

    dictvarbderi = dict()
    dictmodlfine, temp = retr_dictmodl_mile(gdat, gdat.timethisfittfine, dictparainpt, gdat.thisstrgmodl)
    
    dictmodl, dictvarbderi['timeredu'] = retr_dictmodl_mile(gdat, gdat.timethisfitt, dictparainpt, gdat.thisstrgmodl)
    
    #for name in dictrflxmodl:
    #    dictvarbderi['rflxmodl%sfine' % name] = dictrflxmodl[name]
    
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
                
            strg = '%s_%s' % (gdat.liststrgdatatser[b], gdat.listlablinst[b][p])
            
            if gdat.typellik == 'GaussianProcess':
                
                dictmodlgprofine = retr_rflxmodl_mile_gpro(gdat, gdat.thisstrgmodl, gdat.timethisfittfine[b][p], dictparainpt, rflxmodl=dictmodlfine['Total'])
                dictmodlgpro = retr_rflxmodl_mile_gpro(gdat, gdat.thisstrgmodl, gdat.timethisfitt[b][p], dictparainpt, rflxmodl=dictmodl['Total'])
                
                dictmodl['GaussianProcess'] = dictmodlgpro['Total']
                
                dictmodlfine['Total'] += dictrflx['Total'] - 1.
                dictmodl['Total'] += dictrflxfine['Total'] - 1.
            
            for namecompmodl in gmod.listnamecompmodl:
                
                dictvarbderi['Model_Fine_%s_%s' % (namecompmodl, strg)] = dictmodlfine[namecompmodl][b][p]
                
                if gdat.booldiag:
                    if gdat.timethisfittfine[b][p].size != dictmodlfine[namecompmodl][b][p].size:
                        print('')
                        print('')
                        print('')
                        print('')
                        print('namecompmodl')
                        print(namecompmodl)
                        print('dictmodlfine[namecompmodl][b][p]')
                        summgene(dictmodlfine[namecompmodl][b][p])
                        print('gdat.timethisfittfine[b][p]')
                        summgene(gdat.timethisfittfine[b][p])
                        print('np.unique(gdat.timethisfittfine[b][p])')
                        summgene(np.unique(gdat.timethisfittfine[b][p]))
                        raise Exception('')

            dictvarbderi['resi%s' % strg] = gdat.rflxthisfitt[b][p][:, gdat.fitt.listindxinstener[p]] - dictmodl['Total'][b][p]

            dictvarbderi['stdvresi%s' % strg] = np.empty((gdat.numbrebn, gdat.numbener[p]))
            for k in gdat.indxrebn:
                delt = gdat.listdeltrebn[b][p][k]
                arry = np.zeros((dictvarbderi['resi%s' % strg].shape[0], gdat.numbener[p], 3))
                arry[:, 0, 0] = gdat.timethisfitt[b][p]
                for e in gdat.indxenermodl:
                    arry[:, e, 1] = dictvarbderi['resi%s' % strg][:, e]
                arryrebn = rebn_tser(arry, delt=delt)
                dictvarbderi['stdvresi%s' % strg][k, :] = np.nanstd(arryrebn[:, :, 1], axis=0)
                if gdat.booldiag:
                    for e in gdat.indxenermodl:
                        if not np.isfinite(dictvarbderi['stdvresi%s' % strg][k, e]):
                            print('')
                            print('')
                            print('')
                            print('arry')
                            summgene(arry)
                            print('rebn_tser(arry, delt=gdat.listdeltrebn[b][p][k])[:, 1]')
                            summgene(rebn_tser(arry, delt=gdat.listdeltrebn[b][p][k])[:, 1])
                            raise Exception('')
    
            if gdat.booldiag:
                print('dictvarbderi.keys()')
                print(dictvarbderi.keys())
                if dictvarbderi['Model_Fine_Total_%s' % strg].size != gdat.timethisfittfine[b][p].size:
                    raise Exception('')
    
    return dictvarbderi


def retr_llik_albbepsi(para, gdat):
    
    # Bond albedo
    albb = para[0]
    
    # heat recirculation efficiency
    epsi = para[2]

    psiimodl = (1 - albb)**.25
    #tmptirre = gdat.dictlist['tmptequi'][:, 0] * psiimodl
    tmptirre = gdat.gmeatmptequi * psiimodl
    
    tmptplandayy, tmptplannigh = retr_tmptplandayynigh(tmptirre)
    
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


def retr_modl_spec(gdat, tmpt, boolthpt=False, strgtype='intg'):
    
    if boolthpt:
        thpt = scipy.interpolate.interp1d(gdat.cntrwlenband, gdat.thptband)(wlen)
    else:
        thpt = 1.
    
    if strgtype == 'intg':
        spec = tdpy.retr_specbbod(tmpt, gdat.cntrwlen)
        spec = np.sum(gdat.diffwlen * spec)
    if strgtype == 'diff' or strgtype == 'logt':
        spec = tdpy.retr_specbbod(tmpt, gdat.cntrwlen)
        if strgtype == 'logt':
            spec *= gdat.cntrwlen
    
    return spec


def retr_llik_spec(para, gdat):
    
    tmpt = para[0]
    
    specboloplan = retr_modl_spec(gdat, tmpt, boolthpt=False, strgtype='intg')
    deptplan = 1e3 * gdat.rratmedi[0]**2 * specboloplan / gdat.specstarintg # [ppt]
    
    llik = -0.5 * np.sum((deptplan - gdat.deptobsd)**2 / gdat.varideptobsd)
    
    return llik


def writ_filealle(gdat, namefile, pathalle, dictalle, dictalledefa, \
                  # type of verbosity
                  ## -1: absolutely no text
                  ##  0: no text output except critical warnings
                  ##  1: minimal description of the execution
                  ##  2: detailed description of the execution
                  typeverb=1):
    
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
    objtfile = open(pathfile, 'w')
    for line in listline:
        objtfile.write('%s' % line)
    if typeverb > 0:
        print('Writing to %s...' % pathfile)
    objtfile.close()


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


def plot_pser_mile( \
              gdat, \
              strgmodl, \
              strgarry, \
              boolpost=False, \
              # type of verbosity
              ## -1: absolutely no text
              ##  0: no text output except critical warnings
              ##  1: minimal description of the execution
              ##  2: detailed description of the execution
              typeverb=1):
    
    if strgmodl == 'true':
        gmod = gdat.true
    else:
        gmod = gdat.fitt.prio
            
    for b in gdat.indxdatatser:
        
        arrypcur = gmod.arrypcur[strgarry]
        arrypcurbind = gmod.arrypcur[strgarry+'Binned']
        
        for p in gdat.indxinst[b]:
            
            titl = retr_tsertitl(gdat, b, p)
            
            # plot phase curves of individual companions
            for j in gmod.indxcomp:
                
                path = gdat.pathvisutarg + 'PhaseCurve_%s_%s_%s_%s_%s.%s' % (gdat.liststrginst[b][p], gdat.liststrgcomp[j], \
                                                                                            strgarry, gdat.strgtarg, gdat.typepriocomp, gdat.typefileplot)
                gdat.listdictdvrp[j+1].append({'path': path, 'limt':[0., 0.3, 0.5, 0.1]})
                if not os.path.exists(path):
                    # phase on the horizontal axis
                    figr, axis = plt.subplots(1, 1, figsize=gdat.figrsizeydob)
                    if b == 0:
                        yerr = None
                    if b == 1:
                        yerr = arrypcur[b][p][j][:, gdat.indxenerclip, 2]
                    
                    axis.errorbar(arrypcur[b][p][j][:, gdat.indxenerclip, 0], arrypcur[b][p][j][:, gdat.indxenerclip, 1], \
                                                                yerr=yerr, elinewidth=1, capsize=2, zorder=1, \
                                                                color='grey', alpha=gdat.alphdata, marker='o', ls='', ms=1, rasterized=gdat.boolrastraww)
                    # binned
                    if len(arrypcur[b][p][j]) > 0:
                        if b == 0:
                            yerr = None
                        if b == 1:
                            yerr = arrypcurbind[b][p][j][:, gdat.indxenerclip, 2]
                        axis.errorbar(arrypcurbind[b][p][j][:, gdat.indxenerclip, 0], arrypcurbind[b][p][j][:, gdat.indxenerclip, 1], \
                                                                                            yerr=yerr, \
                                                                                            color=gdat.listcolrcomp[j], elinewidth=1, capsize=2)
                    else:
                        print('')
                        print('')
                        print('')
                        print('Warning! Phase curve (%s) is empty, possibly due to being a zoom-in!' % strgarry)

                    if gdat.boolwritplan:
                        axis.text(0.9, 0.9, r'\textbf{%s}' % gdat.liststrgcomp[j], \
                                            color=gdat.listcolrcomp[j], va='center', ha='center', transform=axis.transAxes)
                    axis.set_ylabel(gdat.listlabltser[b])
                    axis.set_xlabel('Phase')
                    axis.set_title(titl)

                    # overlay the posterior model
                    if boolpost:
                        axis.plot(gmod.arrypcur[strgarry[:4]+'modltotl'+strgarry[-4:]][b][p][j][:, gdat.indxenerclip, 0], \
                                  gmod.arrypcur[strgarry[:4]+'modltotl'+strgarry[-4:]][b][p][j][:, gdat.indxenerclip, 1], color='b', zorder=3)
                    if gdat.listdeptdraw is not None:
                        for k in range(len(gdat.listdeptdraw)):  
                            axis.axhline(1. - 1e-3 * gdat.listdeptdraw[k], ls='-', color='grey')
                    if typeverb > 0:
                        print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()
            
                # same plot, with time on the horizontal axis
                if strgarry.startswith('Primary'):
                    path = gdat.pathvisutarg + 'RelativeFlux_PhaseFolded_%s_%s_%s_%s_%s.%s' % (gdat.liststrginst[b][p], gdat.liststrgcomp[j], \
                                                                                    strgarry, gdat.strgtarg, gdat.typepriocomp, gdat.typefileplot)
                    gdat.listdictdvrp[j+1].append({'path': path, 'limt':[0.5, 0.2, 0.5, 0.1]})
                    if not os.path.exists(path):
                        figr, axis = plt.subplots(1, 1, figsize=gdat.figrsize)
                        if b == 0:
                            yerr = None
                        if b == 1:
                            yerr = arrypcur[b][p][j][:, gdat.indxenerclip, 2]
                        
                        # the factor to multiply the time axis and its label
                        facttime, lablunittime = tdpy.retr_timeunitdays(gdat.fitt.prio.meanpara.pericomp[j])
                         
                        axis.errorbar(gdat.fitt.prio.meanpara.pericomp[j] * arrypcur[b][p][j][:, gdat.indxenerclip, 0] * facttime, \
                                                             arrypcur[b][p][j][:, gdat.indxenerclip, 1], yerr=yerr, elinewidth=1, capsize=2, \
                                                            zorder=1, color='grey', alpha=gdat.alphdata, marker='o', ls='', ms=1, rasterized=gdat.boolrastraww)
                        if b == 0:
                            yerr = None
                        if b == 1:
                            yerr = arrypcurbind[b][p][j][:, gdat.indxenerclip, 2]
                        
                        if np.isfinite(gdat.fitt.prio.meanpara.duratrantotlcomp[j]):
                            axis.errorbar(gdat.fitt.prio.meanpara.pericomp[j] * arrypcurbind[b][p][j][:, gdat.indxenerclip, 0] * facttime, \
                                                                 arrypcurbind[b][p][j][:, gdat.indxenerclip, 1], zorder=2, \
                                                                                                        yerr=yerr, elinewidth=1, capsize=2, \
                                                                                                              color=gdat.listcolrcomp[j], marker='o', ls='', ms=3)
                        if boolpost:
                            axis.plot(gdat.fitt.prio.meanpara.pericomp[j] * facttime * gmod.arrypcur[strgarry[:4]+'modltotl'+strgarry[-4:]][b][p][j][:, gdat.indxenerclip, 0], \
                                                                   gmod.arrypcur[strgarry[:4]+'modltotl'+strgarry[-4:]][b][p][j][:, gdat.indxenerclip, 1], \
                                                                                                                            color='b', zorder=3)
                        if gdat.boolwritplan:
                            axis.text(0.9, 0.1, \
                                            r'\textbf{%s}' % gdat.liststrgcomp[j], color=gdat.listcolrcomp[j], va='center', ha='center', transform=axis.transAxes)
                        axis.set_ylabel(gdat.listlabltser[b])
                        axis.set_xlabel('Time [%s]' % lablunittime)
                        axis.set_title(titl)
                        if gdat.fitt.duramask is not None and np.isfinite(gdat.fitt.duramask[j]):
                            axis.set_xlim(facttime * np.array([-np.nanmax(gdat.fitt.duramask), np.nanmax(gdat.fitt.duramask)]) / 24.)
                        if gdat.listdeptdraw is not None:
                            for k in range(len(gdat.listdeptdraw)):  
                                axis.axhline(1. - 1e-3 * gdat.listdeptdraw[k], ls='--', color='grey')
                        plt.subplots_adjust(hspace=0., bottom=0.25, left=0.25)
                        if gdat.typeverb > 0:
                            print('Writing to %s...' % path)
                        plt.savefig(path)
                        plt.close()
            
            # plot phase curves of all companions together
            if gmod.numbcomp > 1:
                path = gdat.pathvisutarg + 'PhaseCurve_All_%s_%s_%s_%s.%s' % (gdat.liststrginst[b][p], strgarry, \
                                                                                gdat.strgtarg, gdat.typepriocomp, gdat.typefileplot)
                #gdat.listdictdvrp[j+1].append({'path': path, 'limt':[0.4, 0.05, 0.5, 0.1]})
                if not os.path.exists(path):
                    figr, axis = plt.subplots(gmod.numbcomp, 1, figsize=gdat.figrsizeydob, sharex=True)
                    if gmod.numbcomp == 1:
                        axis = [axis]
                    for jj, j in enumerate(gdat.fitt.prio.indxcomp):
                        axis[jj].plot(arrypcur[b][p][j][:, gdat.indxenerclip, 0], arrypcur[b][p][j][:, gdat.indxenerclip, 1], color='grey', alpha=gdat.alphdata, \
                                                                                            marker='o', ls='', ms=1, rasterized=gdat.boolrastraww)
                        axis[jj].plot(arrypcurbind[b][p][j][:, gdat.indxenerclip, 0], \
                                            arrypcurbind[b][p][j][:, gdat.indxenerclip, 1], color=gdat.listcolrcomp[j], marker='o', ls='', ms=1)
                        if gdat.boolwritplan:
                            axis[jj].text(0.97, 0.8, r'\textbf{%s}' % gdat.liststrgcomp[j], transform=axis[jj].transAxes, \
                                                                                                color=gdat.listcolrcomp[j], va='center', ha='center')
                    axis[0].set_ylabel(gdat.listlabltser[b])
                    axis[0].set_xlim(-0.5, 0.5)
                    axis[0].yaxis.set_label_coords(-0.08, 1. - 0.5 * gmod.numbcomp)
                    axis[gmod.numbcomp-1].set_xlabel('Phase')
                    axis[0].set_title(titl)
                    
                    plt.subplots_adjust(hspace=0., bottom=0.2)
                    if gdat.typeverb > 0:
                        print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()
    

def calc_feat_alle(gdat, strgpdfn):

    gdat.liststrgfeat = ['epocmtracomp', 'pericomp', 'rratcomp', 'rsmacomp', 'cosicomp', 'ecos', 'esin', 'rvelsema']
    if strgpdfn == '0003' or strgpdfn == '0004':
        gdat.liststrgfeat += ['sbrtrati', 'amplelli', 'amplbeam']
    if strgpdfn == '0003':
        gdat.liststrgfeat += ['amplplan', 'timeshftplan']
    if strgpdfn == '0004':
        gdat.liststrgfeat += ['amplplanther', 'amplplanrefl', 'timeshftplanther', 'timeshftplanrefl']
    
    gdat.dictlist = {}
    gdat.dictpost = {}
    gdat.dicterrr = {}
    for strgfeat in gdat.liststrgfeat:
        gdat.dictlist[strgfeat] = np.empty((gdat.numbsamp, gmod.numbcomp))

        for j in gmod.indxcomp:
            if strgpdfn == 'prio' or strgpdfn in gdat.typepriocomp:
                mean = getattr(gdat, strgfeat + 'prio')
                stdv = getattr(gdat, 'stdv' + strgfeat + 'prio')
                if not np.isfinite(mean[j]):
                    continue

                gdat.dictlist[strgfeat][:, j] = mean[j] + np.random.randn(gdat.numbsamp) * stdv[j]
                if strgfeat == 'rratcomp':
                    gdat.dictlist[strgfeat][:, j] = tdpy.samp_gaustrun(gdat.numbsamp, mean[j], stdv[j], 0., np.inf)

            else:
                if strgfeat == 'epocmtracomp':
                    strg = '%s_epoch' % gdat.liststrgcomp[j]
                if strgfeat == 'pericomp':
                    strg = '%s_period' % gdat.liststrgcomp[j]
                if strgfeat == 'rratcomp':
                    strg = '%s_rr' % gdat.liststrgcomp[j]
                if strgfeat == 'rsmacomp':
                    strg = '%s_rsuma' % gdat.liststrgcomp[j]
                if strgfeat == 'cosicomp':
                    strg = '%s_cosi' % gdat.liststrgcomp[j]
    
                if strgpdfn == '0003' or strgpdfn == '0004':
                    if strgfeat == 'sbrtrati':
                        strg = '%s_sbratio_TESS' % gdat.liststrgcomp[j]
                    if strgfeat == 'amplbeam':
                        strg = '%s_phase_curve_beaming_TESS' % gdat.liststrgcomp[j]
                    if strgfeat == 'amplelli':
                        strg = '%s_phase_curve_ellipsoidal_TESS' % gdat.liststrgcomp[j]
                if strgpdfn == '0003':
                    if strgfeat == 'amplplan':
                        strg = '%s_phase_curve_atmospheric_TESS' % gdat.liststrgcomp[j]
                    if strgfeat == 'timeshftplan':
                        strg = '%s_phase_curve_atmospheric_shift_TESS' % gdat.liststrgcomp[j]
                if strgpdfn == '0004':
                    if strgfeat == 'amplplanther':
                        strg = '%s_phase_curve_atmospheric_thermal_TESS' % gdat.liststrgcomp[j]
                    if strgfeat == 'amplplanrefl':
                        strg = '%s_phase_curve_atmospheric_reflected_TESS' % gdat.liststrgcomp[j]
                    if strgfeat == 'timeshftplanther':
                        strg = '%s_phase_curve_atmospheric_thermal_shift_TESS' % gdat.liststrgcomp[j]
                    if strgfeat == 'timeshftplanrefl':
                        strg = '%s_phase_curve_atmospheric_reflected_shift_TESS' % gdat.liststrgcomp[j]
            
                if strg in gdat.objtalle[strgpdfn].posterior_params.keys():
                    gdat.dictlist[strgfeat][:, j] = gdat.objtalle[strgpdfn].posterior_params[strg][gdat.indxsamp]
                else:
                    gdat.dictlist[strgfeat][:, j] = np.zeros(gdat.numbsamp) + allesfitter.config.BASEMENT.params[strg]

    if gdat.typeverb > 0:
        print('Calculating derived variables...')
    # derived variables
    ## get samples from the star's variables

    # stellar features
    for featstar in gdat.listfeatstar:
        meantemp = getattr(gdat, featstar)
        stdvtemp = getattr(gdat, 'stdv' + featstar)
        
        # not a specific host star
        if meantemp is None:
            continue

        if not np.isfinite(meantemp):
            if gdat.typeverb > 0:
                print('Stellar feature %s is not finite!' % featstar)
                print('featstar')
                print(featstar)
                print('meantemp')
                print(meantemp)
                print('stdvtemp')
                print(stdvtemp)
            gdat.dictlist[featstar] = np.empty(gdat.numbsamp) + np.nan
        elif stdvtemp == 0.:
            gdat.dictlist[featstar] = meantemp + np.zeros(gdat.numbsamp)
        else:
            gdat.dictlist[featstar] = tdpy.samp_gaustrun(gdat.numbsamp, meantemp, stdvtemp, 0., np.inf)
        
        gdat.dictlist[featstar] = np.vstack([gdat.dictlist[featstar]] * gmod.numbcomp).T
    
    # inclination [degree]
    gdat.dictlist['incl'] = np.arccos(gdat.dictlist['cosicomp']) * 180. / np.pi
    
    # log g of the host star
    gdat.dictlist['loggstar'] = gdat.dictlist['massstar'] / gdat.dictlist['radistar']**2

    gdat.dictlist['ecce'] = gdat.dictlist['esin']**2 + gdat.dictlist['ecos']**2
    
    if gmod.boolmodlpsys:
        # radius of the planets
        gdat.dictlist['radicomp'] = gdat.dictfact['rsre'] * gdat.dictlist['radistar'] * gdat.dictlist['rratcomp'] # [R_E]
    
        # semi-major axis
        gdat.dictlist['smax'] = (gdat.dictlist['radicomp'] + gdat.dictlist['radistar']) / gdat.dictlist['rsmacomp']
    
        if strgpdfn == '0003' or strgpdfn == '0004':
            gdat.dictlist['amplnigh'] = gdat.dictlist['sbrtrati'] * gdat.dictlist['rratcomp']**2
        if strgpdfn == '0003':
            gdat.dictlist['phasshftplan'] = gdat.dictlist['timeshftplan'] * 360. / gdat.dictlist['pericomp']
        if strgpdfn == '0004':
            gdat.dictlist['phasshftplanther'] = gdat.dictlist['timeshftplanther'] * 360. / gdat.dictlist['pericomp']
            gdat.dictlist['phasshftplanrefl'] = gdat.dictlist['timeshftplanrefl'] * 360. / gdat.dictlist['pericomp']

        # planet equilibrium temperature
        gdat.dictlist['tmptplan'] = gdat.dictlist['tmptstar'] * np.sqrt(gdat.dictlist['radistar'] / 2. / gdat.dictlist['smax'])
        
        # stellar luminosity
        gdat.dictlist['lumistar'] = gdat.dictlist['radistar']**2 * (gdat.dictlist['tmptstar'] / 5778.)**4
        
        # insolation
        gdat.dictlist['inso'] = gdat.dictlist['lumistar'] / gdat.dictlist['smax']**2
    
        # predicted planet mass
        if gdat.typeverb > 0:
            print('Calculating predicted masses...')
        
        gdat.dictlist['masscomppred'] = np.full_like(gdat.dictlist['radicomp'], np.nan)
        gdat.dictlist['masscomppred'] = nicomedia.retr_massfromradi(gdat.dictlist['radicomp'])
        gdat.dictlist['masscomppred'] = gdat.dictlist['masscomppred']
        
        # mass used for later calculations
        gdat.dictlist['masscompused'] = np.empty_like(gdat.dictlist['masscomppred'])
        
        # temp
        gdat.dictlist['masscomp'] = np.zeros_like(gdat.dictlist['esin'])
        gdat.dictlist['masscompused'] = gdat.dictlist['masscomppred']
        #for j in gmod.indxcomp:
        #    if 
        #        gdat.dictlist['masscompused'][:, j] = 
        #    else:
        #        gdat.dictlist['masscompused'][:, j] = 
    
        # density of the planet
        gdat.dictlist['densplan'] = gdat.dictlist['masscompused'] / gdat.dictlist['radicomp']**3

        # escape velocity
        gdat.dictlist['vesc'] = nicomedia.retr_vesc(gdat.dictlist['masscompused'], gdat.dictlist['radicomp'])
        
        for j in gmod.indxcomp:
            strgratiperi = 'ratiperi_%s' % gdat.liststrgcomp[j]
            strgratiradi = 'ratiradi_%s' % gdat.liststrgcomp[j]
            for jj in gmod.indxcomp:
                gdat.dictlist[strgratiperi] = gdat.dictlist['pericomp'][:, j] / gdat.dictlist['pericomp'][:, jj]
                gdat.dictlist[strgratiradi] = gdat.dictlist['radicomp'][:, j] / gdat.dictlist['radicomp'][:, jj]
    
        gdat.dictlist['depttrancomp'] = 1e3 * gdat.dictlist['rratcomp']**2 # [ppt]
        # TSM
        gdat.dictlist['tsmm'] = nicomedia.retr_tsmm(gdat.dictlist['radicomp'], gdat.dictlist['tmptplan'], \
                                                                                    gdat.dictlist['masscompused'], gdat.dictlist['radistar'], gdat.jmagsyst)
        
        # ESM
        gdat.dictlist['esmm'] = nicomedia.retr_esmm(gdat.dictlist['tmptplan'], gdat.dictlist['tmptstar'], \
                                                                                    gdat.dictlist['radicomp'], gdat.dictlist['radistar'], gdat.kmagsyst)
        
    else:
        # semi-major axis
        gdat.dictlist['smax'] = (gdat.dictlist['radistar']) / gdat.dictlist['rsmacomp']
    
    # temp
    gdat.dictlist['sini'] = np.sqrt(1. - gdat.dictlist['cosicomp']**2)
    gdat.dictlist['omeg'] = 180. / np.pi * np.mod(np.arctan2(gdat.dictlist['esin'], gdat.dictlist['ecos']), 2 * np.pi)
    gdat.dictlist['rs2a'] = gdat.dictlist['rsmacomp'] / (1. + gdat.dictlist['rratcomp'])
    gdat.dictlist['sinw'] = np.sin(np.pi / 180. * gdat.dictlist['omeg'])
    gdat.dictlist['imfa'] = nicomedia.retr_imfa(gdat.dictlist['cosicomp'], gdat.dictlist['rs2a'], gdat.dictlist['ecce'], gdat.dictlist['sinw'])
   
    # RV semi-amplitude
    gdat.dictlist['rvelsemapred'] = nicomedia.retr_rvelsema(gdat.dictlist['pericomp'], gdat.dictlist['masscomppred'], \
                                                                        gdat.dictlist['massstar'], gdat.dictlist['incl'], gdat.dictlist['ecce'])
    
    ## expected Doppler beaming (DB)
    deptbeam = 1e3 * 4. * gdat.dictlist['rvelsemapred'] / 3e8 * gdat.consbeam # [ppt]

    ## expected ellipsoidal variation (EV)
    ## limb and gravity darkening coefficients from Claret2017
    if gdat.typeverb > 0:
        print('temp: connect these to Claret2017')
    # linear limb-darkening coefficient
    coeflidaline = 0.4
    # gravitational darkening coefficient
    coefgrda = 0.2
    alphelli = nicomedia.retr_alphelli(coeflidaline, coefgrda)
    gdat.dictlist['deptelli'] = 1e3 * alphelli * gdat.dictlist['masscompused'] * np.sin(gdat.dictlist['incl'] / 180. * np.pi)**2 / \
                                                                  gdat.dictlist['massstar'] * (gdat.dictlist['radistar'] / gdat.dictlist['smax'])**3 # [ppt]
    if gdat.typeverb > 0:
        print('Calculating durations...')
                      
    gdat.dictlist['duratranfull'] = nicomedia.retr_duratranfull(gdat.dictlist['pericomp'], gdat.dictlist['rsmacomp'], gdat.dictlist['cosicomp'], gdat.dictlist['rratcomp'])
    gdat.dictlist['duratrantotl'] = nicomedia.retr_duratrantotl(gdat.dictlist['pericomp'], gdat.dictlist['rsmacomp'], gdat.dictlist['cosicomp'])
    
    gdat.dictlist['maxmdeptblen'] = 1e3 * (1. - gdat.dictlist['duratranfull'] / gdat.dictlist['duratrantotl'])**2 / \
                                                                    (1. + gdat.dictlist['duratranfull'] / gdat.dictlist['duratrantotl'])**2 # [ppt]
    gdat.dictlist['minmdilu'] = gdat.dictlist['depttrancomp'] / gdat.dictlist['maxmdeptblen']
    gdat.dictlist['minmratiflux'] = gdat.dictlist['minmdilu'] / (1. - gdat.dictlist['minmdilu'])
    gdat.dictlist['maxmdmag'] = -2.5 * np.log10(gdat.dictlist['minmratiflux'])
    
    # orbital
    ## RM effect
    gdat.dictlist['amplrmef'] = 2. / 3. * gdat.dictlist['vsiistar'] * 1e-3 * gdat.dictlist['depttrancomp'] * np.sqrt(1. - gdat.dictlist['imfa'])
    gdat.dictlist['stnormefpfss'] = (gdat.dictlist['amplrmef'] / 0.9) * np.sqrt(gdat.dictlist['duratranfull'] / (10. / 60. / 24.))
    
    # 0003 single component, offset
    # 0004 double component, offset
    if strgpdfn == '0003':
        frac = np.random.rand(gdat.dictlist['amplplan'].size).reshape(gdat.dictlist['amplplan'].shape)
        gdat.dictlist['amplplanther'] = gdat.dictlist['amplplan'] * frac
        gdat.dictlist['amplplanrefl'] = gdat.dictlist['amplplan'] * (1. - frac)
    
    if strgpdfn == '0004':
        # temp -- this does not work for two component (thermal + reflected)
        gdat.dictlist['amplseco'] = gdat.dictlist['amplnigh'] + gdat.dictlist['amplplanther'] + gdat.dictlist['amplplanrefl']
    if strgpdfn == '0003':
        # temp -- this does not work when phase shift is nonzero
        gdat.dictlist['amplseco'] = gdat.dictlist['amplnigh'] + gdat.dictlist['amplplan']
    
    if strgpdfn == '0003' or strgpdfn == '0004':
        gdat.dictlist['albg'] = retr_albg(gdat.dictlist['amplplanrefl'], gdat.dictlist['radicomp'], gdat.dictlist['smax'])

    if gdat.typeverb > 0:
        print('Calculating the equilibrium temperature of the planets...')
    
    gdat.dictlist['tmptequi'] = gdat.dictlist['tmptstar'] * np.sqrt(gdat.dictlist['radistar'] / gdat.dictlist['smax'] / 2.)
    
    if False and gdat.labltarg == 'WASP-121' and strgpdfn != 'prio':
        
        ### spectrum of the host star
        gdat.cntrwlenthomraww = arrymodl[:, 0]
        gdat.specstarthomraww = arrymodl[:, 9]
        
        ## calculate the geometric albedo "informed" by the ATMO posterior
        wlenmodl = arrymodl[:, 0]
        deptmodl = arrymodl[:, 1]
        indxwlenmodltess = np.where((wlenmodl > 0.6) & (wlenmodl < 0.95))[0]
        gdat.amplplantheratmo = np.mean(deptmodl[indxwlenmodltess])
        gdat.dictlist['amplplanreflatmo'] = 1e-6 * arrydata[0, 2] + np.random.randn(gdat.numbsamp).reshape((gdat.numbsamp, 1)) \
                                                                                                    * arrydata[0, 3] * 1e-6 - gdat.amplplantheratmo
        #gdat.dictlist['amplplanreflatmo'] = gdat.dictlist['amplplan'] - gdat.amplplantheratmo
        gdat.dictlist['albginfo'] = retr_albg(gdat.dictlist['amplplanreflatmo'], gdat.dictlist['radicomp'], gdat.dictlist['smax'])
        
        ## update Tom's secondary (dayside) with the posterior secondary depth, since Tom's secondary was preliminary (i.e., 490+-50 ppm)
        print('Updating the multiband depth array with dayside and adding the nightside...')
        medideptseco = np.median(gdat.dictlist['amplseco'][:, 0])
        stdvdeptseco = (np.percentile(gdat.dictlist['amplseco'][:, 0], 84.) - np.percentile(gdat.dictlist['amplseco'][:, 0], 16.)) / 2.
        arrydata[0, 2] = medideptseco * 1e3 # [ppm]
        arrydata[0, 3] = stdvdeptseco * 1e3 # [ppm]
        
        ## add the nightside depth
        medideptnigh = np.median(gdat.dictlist['amplnigh'][:, 0])
        stdvdeptnigh = (np.percentile(gdat.dictlist['amplnigh'][:, 0], 84.) - np.percentile(gdat.dictlist['amplnigh'][:, 0], 16.)) / 2.
        arrydata = np.concatenate((arrydata, np.array([[arrydata[0, 0], arrydata[0, 1], medideptnigh * 1e3, stdvdeptnigh * 1e6, 0, 0, 0, 0]])), axis=0) # [ppm]
        
        # calculate brightness temperatures
        gmod.listlablpara = [['Temperature', 'K']]
        gdat.rratmedi = np.median(gdat.dictlist['rratcomp'], axis=0)
        listscalpara = ['self']
        gmod.listminmpara = np.array([1000.])
        gmod.listmaxmpara = np.array([4000.])
        meangauspara = None
        stdvgauspara = None
        numbpara = len(gmod.listlablpara)
        numbsampwalk = 1000
        numbsampburnwalk = 5
        gdat.numbdatatmpt = arrydata.shape[0]
        gdat.indxdatatmpt = np.arange(gdat.numbdatatmpt)
        listtmpt = []
        specarry = np.empty((2, 3, gdat.numbdatatmpt))
        for k in gdat.indxdatatmpt:
            
            if not (k == 0 or k == gdat.numbdatatmpt - 1):
                continue
            gdat.meanwlen = np.mean(gdat.cntrwlen)
            strgextn = 'tmpt_%d' % k
            gdat.indxenerdata = k

            gdat.specstarintg = retr_modl_spec(gdat, gdat.tmptstar, strgtype='intg')
            
            gdat.specstarthomlogt = scipy.interpolate.interp1d(gdat.cntrwlenthomraww, gdat.specstarthomraww)(gdat.cntrwlen)
            gdat.specstarthomdiff = gdat.specstarthomlogt / gdat.cntrwlen
            gdat.specstarthomintg = np.sum(gdat.diffwlen * \
                                    scipy.interpolate.interp1d(gdat.cntrwlenthomraww, gdat.specstarthomraww)(gdat.cntrwlen) / gdat.cntrwlen)

            gdat.deptobsd = arrydata[k, 2]
            gdat.stdvdeptobsd = arrydata[k, 3]
            gdat.varideptobsd = gdat.stdvdeptobsd**2
            listtmpttemp = tdpy.samp(gdat, gdat.pathalle[strgpdfn], numbsampwalk, \
                                     retr_llik_spec, \
                                     gmod.listlablpara, listscalpara, gmod.listminmpara, gmod.listmaxmpara, \
                                     meangauspara, stdvgauspara, numbdata, strgextn=strgextn, \
                                     pathbase=gdat.pathtargcnfg, \
                                     typeverb=gdat.typeverb, \
                                     numbsampburnwalk=numbsampburnwalk, boolplot=gdat.boolplot, \
                                    )
            listtmpt.append(listtmpttemp)
        listtmpt = np.vstack(listtmpt).T
        indxsamp = np.random.choice(np.arange(listtmpt.shape[0]), size=gdat.numbsamp, replace=False)
        # dayside and nightside temperatures to be used for albedo and circulation efficiency calculation
        gdat.dictlist['tmptdayy'] = listtmpt[indxsamp, 0, None]
        gdat.dictlist['tmptnigh'] = listtmpt[indxsamp, -1, None]
        # dayside/nightside temperature contrast
        gdat.dictlist['tmptcont'] = (gdat.dictlist['tmptdayy'] - gdat.dictlist['tmptnigh']) / gdat.dictlist['tmptdayy']
        
    # copy the prior
    gdat.dictlist['projoblq'] = np.random.randn(gdat.numbsamp)[:, None] * gdat.stdvprojoblqprio[None, :] + gdat.projoblqprio[None, :]
    
    gdat.boolsampbadd = np.zeros(gdat.numbsamp, dtype=bool)
    for j in gmod.indxcomp:
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
        gdat.dictpost[strgfeat][0, ...] = np.nanpercentile(gdat.dictlist[strgfeat], 16., 0)
        gdat.dictpost[strgfeat][1, ...] = np.nanpercentile(gdat.dictlist[strgfeat], 50., 0)
        gdat.dictpost[strgfeat][2, ...] = np.nanpercentile(gdat.dictlist[strgfeat], 84., 0)
        gdat.dicterrr[strgfeat][0, ...] = gdat.dictpost[strgfeat][1, ...]
        gdat.dicterrr[strgfeat][1, ...] = gdat.dictpost[strgfeat][1, ...] - gdat.dictpost[strgfeat][0, ...]
        gdat.dicterrr[strgfeat][2, ...] = gdat.dictpost[strgfeat][2, ...] - gdat.dictpost[strgfeat][1, ...]
        
    # augment
    gdat.dictfeatobjt['radistar'] = gdat.dicterrr['radistar'][0, :]
    gdat.dictfeatobjt['radicomp'] = gdat.dicterrr['radicomp'][0, :]
    gdat.dictfeatobjt['masscomp'] = gdat.dicterrr['masscomp'][0, :]
    gdat.dictfeatobjt['stdvradistar'] = np.mean(gdat.dicterrr['radistar'][1:, :], 0)
    gdat.dictfeatobjt['stdvmassstar'] = np.mean(gdat.dicterrr['massstar'][1:, :], 0)
    gdat.dictfeatobjt['stdvtmptstar'] = np.mean(gdat.dicterrr['tmptstar'][1:, :], 0)
    gdat.dictfeatobjt['stdvloggstar'] = np.mean(gdat.dicterrr['loggstar'][1:, :], 0)
    gdat.dictfeatobjt['stdvradicomp'] = np.mean(gdat.dicterrr['radicomp'][1:, :], 0)
    gdat.dictfeatobjt['stdvmasscomp'] = np.mean(gdat.dicterrr['masscomp'][1:, :], 0)
    gdat.dictfeatobjt['stdvtmptplan'] = np.mean(gdat.dicterrr['tmptplan'][1:, :], 0)
    gdat.dictfeatobjt['stdvesmm'] = np.mean(gdat.dicterrr['esmm'][1:, :], 0)
    gdat.dictfeatobjt['stdvtsmm'] = np.mean(gdat.dicterrr['tsmm'][1:, :], 0)
    

def proc_alle(gdat, typemodl):
    
    if gdat.typeverb > 0:
        print('Processing allesfitter model %s...' % typemodl)
    
    # allesfit run folder
    gdat.pathalle[typemodl] = gdat.pathallebase + 'allesfit_%s/' % typemodl
    
    # make sure the folder exists
    cmnd = 'mkdir -p %s' % gdat.pathalle[typemodl]
    os.system(cmnd)
    
    # write the input data file
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            path = gdat.pathalle[typemodl] + gdat.liststrginst[b][p] + '.csv'
            if not os.path.exists(path):
            
                if gdat.boolinfefoldbind:
                    listarrytserbdtrtemp = np.copy(gmod.arrypcur['DetrendedPrimaryCenteredBinned'][b][p][0])
                    listarrytserbdtrtemp[:, 0] *= gdat.fitt.prio.meanpara.pericomp[0]
                    listarrytserbdtrtemp[:, 0] += gdat.fitt.prio.meanpara.epocmtracomp[0]
                else:
                    listarrytserbdtrtemp = gdat.arrytser['Detrended'][b][p]
                
                # make sure the data are time-sorted
                #indx = np.argsort(listarrytserbdtrtemp[:, 0])
                #listarrytserbdtrtemp = listarrytserbdtrtemp[indx, :]
                    
                if gdat.typeverb > 0:
                    print('Writing to %s...' % path)
                np.savetxt(path, listarrytserbdtrtemp, delimiter=',', header=gdat.strgheadtser[b])
    
    ## params_star
    pathparastar = gdat.pathalle[typemodl] + 'params_star.csv'
    if not os.path.exists(pathparastar):
        objtfile = open(pathparastar, 'w')
        objtfile.write('#R_star,R_star_lerr,R_star_uerr,M_star,M_star_lerr,M_star_uerr,Teff_star,Teff_star_lerr,Teff_star_uerr\n')
        objtfile.write('#R_sun,R_sun,R_sun,M_sun,M_sun,M_sun,K,K,K\n')
        objtfile.write('%g,%g,%g,%g,%g,%g,%g,%g,%g' % (gdat.radistar, gdat.stdvradistar, gdat.stdvradistar, \
                                                       gdat.massstar, gdat.stdvmassstar, gdat.stdvmassstar, \
                                                                                                      gdat.tmptstar, gdat.stdvtmptstar, gdat.stdvtmptstar))
        if gdat.typeverb > 0:
            print('Writing to %s...' % pathparastar)
        objtfile.close()

    ## params
    dictalleparadefa = dict()
    pathpara = gdat.pathalle[typemodl] + 'params.csv'
    if not os.path.exists(pathpara):
        cmnd = 'touch %s' % (pathpara)
        print(cmnd)
        os.system(cmnd)
    
        for j in gmod.indxcomp:
            strgrrat = '%s_rr' % gdat.liststrgcomp[j]
            strgrsma = '%s_rsuma' % gdat.liststrgcomp[j]
            strgcosi = '%s_cosi' % gdat.liststrgcomp[j]
            strgepoc = '%s_epoch' % gdat.liststrgcomp[j]
            strgperi = '%s_period' % gdat.liststrgcomp[j]
            strgecos = '%s_f_c' % gdat.liststrgcomp[j]
            strgesin = '%s_f_s' % gdat.liststrgcomp[j]
            strgrvelsema = '%s_K' % gdat.liststrgcomp[j]
            dictalleparadefa[strgrrat] = ['%f' % gdat.fitt.prio.meanpara.rratcomp[j], '1', 'uniform 0 %f' % (4 * gdat.fitt.prio.meanpara.rratcomp[j]), \
                                                                            '$R_{%s} / R_\star$' % gdat.liststrgcomp[j], '']
            
            dictalleparadefa[strgrsma] = ['%f' % gdat.fitt.prio.meanpara.rsmacomp[j], '1', 'uniform 0 %f' % (4 * gdat.fitt.prio.meanpara.rsmacomp[j]), \
                                                                      '$(R_\star + R_{%s}) / a_{%s}$' % (gdat.liststrgcomp[j], gdat.liststrgcomp[j]), '']
            dictalleparadefa[strgcosi] = ['%f' % gdat.fitt.prio.meanpara.cosicomp[j], '1', 'uniform 0 %f' % max(0.1, 4 * gdat.fitt.prio.meanpara.cosicomp[j]), \
                                                                                        '$\cos{i_{%s}}$' % gdat.liststrgcomp[j], '']
            dictalleparadefa[strgepoc] = ['%f' % gdat.fitt.prio.meanpara.epocmtracomp[j], '1', \
                             'uniform %f %f' % (gdat.fitt.prio.meanpara.epocmtracomp[j] - gdat.stdvepocmtracompprio[j], gdat.fitt.prio.meanpara.epocmtracomp[j] + gdat.stdvepocmtracompprio[j]), \
                                                                    '$T_{0;%s}$' % gdat.liststrgcomp[j], '$\mathrm{BJD}$']
            dictalleparadefa[strgperi] = ['%f' % gdat.fitt.prio.meanpara.pericomp[j], '1', \
                                     'uniform %f %f' % (gdat.fitt.prio.meanpara.pericomp[j] - 3. * gdat.stdvpericompprio[j], gdat.fitt.prio.meanpara.pericomp[j] + 3. * gdat.stdvpericompprio[j]), \
                                                                    '$P_{%s}$' % gdat.liststrgcomp[j], 'days']
            dictalleparadefa[strgecos] = ['%f' % gdat.ecoscompprio[j], '0', 'uniform -0.9 0.9', \
                                                                '$\sqrt{e_{%s}} \cos{\omega_{%s}}$' % (gdat.liststrgcomp[j], gdat.liststrgcomp[j]), '']
            dictalleparadefa[strgesin] = ['%f' % gdat.esincompprio[j], '0', 'uniform -0.9 0.9', \
                                                                '$\sqrt{e_{%s}} \sin{\omega_{%s}}$' % (gdat.liststrgcomp[j], gdat.liststrgcomp[j]), '']
            dictalleparadefa[strgrvelsema] = ['%f' % gdat.rvelsemaprio[j], '0', \
                               'uniform %f %f' % (max(0, gdat.rvelsemaprio[j] - 5 * gdat.stdvrvelsemaprio[j]), gdat.rvelsemaprio[j] + 5 * gdat.stdvrvelsemaprio[j]), \
                                                                '$K_{%s}$' % gdat.liststrgcomp[j], '']
            if typemodl == '0003' or typemodl == '0004':
                for b in gdat.indxdatatser:
                    if b != 0:
                        continue
                    for p in gdat.indxinst[b]:
                        strgsbrt = '%s_sbratio_' % gdat.liststrgcomp[j] + gdat.liststrginst[b][p]
                        dictalleparadefa[strgsbrt] = ['1e-3', '1', 'uniform 0 1', '$J_{%s; \mathrm{%s}}$' % \
                                                                            (gdat.liststrgcomp[j], gdat.listlablinst[b][p]), '']
                        
                        dictalleparadefa['%s_phase_curve_beaming_%s' % (gdat.liststrgcomp[j], gdat.liststrginst[b][p])] = \
                                             ['0', '1', 'uniform 0 10', '$A_\mathrm{beam; %s; %s}$' % (gdat.liststrgcomp[j], gdat.listlablinst[b][p]), '']
                        dictalleparadefa['%s_phase_curve_atmospheric_%s' % (gdat.liststrgcomp[j], gdat.liststrginst[b][p])] = \
                                             ['0', '1', 'uniform 0 10', '$A_\mathrm{atmo; %s; %s}$' % (gdat.liststrgcomp[j], gdat.listlablinst[b][p]), '']
                        dictalleparadefa['%s_phase_curve_ellipsoidal_%s' % (gdat.liststrgcomp[j], gdat.liststrginst[b][p])] = \
                                             ['0', '1', 'uniform 0 10', '$A_\mathrm{elli; %s; %s}$' % (gdat.liststrgcomp[j], gdat.listlablinst[b][p]), '']

            if typemodl == '0003':
                for b in gdat.indxdatatser:
                    if b != 0:
                        continue
                    for p in gdat.indxinst[b]:
                        maxmshft = 0.25 * gdat.fitt.prio.meanpara.pericomp[j]
                        minmshft = -maxmshft

                        dictalleparadefa['%s_phase_curve_atmospheric_shift_%s' % (gdat.liststrgcomp[j], gdat.liststrginst[b][p])] = \
                                         ['0', '1', 'uniform %.3g %.3g' % (minmshft, maxmshft), \
                                            '$\Delta_\mathrm{%s; %s}$' % (gdat.liststrgcomp[j], gdat.listlablinst[b][p]), '']
        if typemodl == 'pfss':
            for p in gdat.indxinst[1]:
                                ['', 'host_vsini,%g,1,uniform %g %g,$v \sin i$$,\n' % (gdat.vsiistar, 0, \
                                                                                                                            10 * gdat.vsiistar)], \
                                ['', 'host_lambda_%s,%g,1,uniform %g %g,$v \sin i$$,\n' % (gdat.liststrginst[1][p], gdat.lambstarprio, 0, \
                                                                                                                            10 * gdat.lambstarprio)], \
        
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
                
        writ_filealle(gdat, 'params.csv', gdat.pathalle[typemodl], gdat.dictdictallepara[typemodl], dictalleparadefa)
    
    ## settings
    dictallesettdefa = dict()
    if typemodl == 'pfss':
        for j in gmod.indxcomp:
            dictallesettdefa['%s_flux_weighted_PFS' % gdat.liststrgcomp[j]] = 'True'
    
    pathsett = gdat.pathalle[typemodl] + 'settings.csv'
    if not os.path.exists(pathsett):
        cmnd = 'touch %s' % (pathsett)
        print(cmnd)
        os.system(cmnd)
        
        dictallesettdefa['fast_fit_width'] = '%.3g' % np.amax(gdat.fitt.duramask) / 24.
        dictallesettdefa['multiprocess'] = 'True'
        dictallesettdefa['multiprocess_cores'] = 'all'

        dictallesettdefa['mcmc_nwalkers'] = '100'
        dictallesettdefa['mcmc_total_steps'] = '100'
        dictallesettdefa['mcmc_burn_steps'] = '10'
        dictallesettdefa['mcmc_thin_by'] = '5'
        
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
        
        if typemodl == '0003' or typemodl == '0004':
            dictallesettdefa['phase_curve'] = 'True'
            dictallesettdefa['phase_curve_style'] = 'sine_physical'
        
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                for j in gmod.indxcomp:
                    dictallesettdefa['%s_grid_%s' % (gdat.liststrgcomp[j], gdat.liststrginst[b][p])] = 'very_sparse'
            
            if gdat.numbinst[b] > 0:
                if b == 0:
                    strg = 'companions_phot'
                if b == 1:
                    strg = 'companions_rv'
                varb = ''
                cntr = 0
                for j in gmod.indxcomp:
                    if cntr != 0:
                        varb += ' '
                    varb += '%s' % gdat.liststrgcomp[j]
                    cntr += 1
                dictallesettdefa[strg] = varb
        
        dictallesettdefa['fast_fit'] = 'True'

        writ_filealle(gdat, 'settings.csv', gdat.pathalle[typemodl], gdat.dictdictallesett[typemodl], dictallesettdefa)
    
    ## initial plot
    path = gdat.pathalle[typemodl] + 'results/initial_guess_b.pdf'
    if not os.path.exists(path):
        allesfitter.show_initial_guess(gdat.pathalle[typemodl])
    
    ## do the run
    path = gdat.pathalle[typemodl] + 'results/mcmc_save.h5'
    if not os.path.exists(path):
        allesfitter.mcmc_fit(gdat.pathalle[typemodl])
    else:
        print('%s exists... Skipping the orbit run.' % path)

    ## make the final plots
    path = gdat.pathalle[typemodl] + 'results/mcmc_corner.pdf'
    if not os.path.exists(path):
        allesfitter.mcmc_output(gdat.pathalle[typemodl])
        
    # read the allesfitter posterior
    if gdat.typeverb > 0:
        print('Reading from %s...' % gdat.pathalle[typemodl])
    gdat.objtalle[typemodl] = allesfitter.allesclass(gdat.pathalle[typemodl])
    
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

    gdat.numbsamp = gdat.objtalle[typemodl].posterior_params[list(gdat.objtalle[typemodl].posterior_params.keys())[0]].size
    
    print('gdat.numbsamp')
    print(gdat.numbsamp)

    # temp 
    if gdat.numbsamp > 10000:
        gdat.indxsamp = np.random.choice(np.arange(gdat.numbsamp), size=10000, replace=False)
        gdat.numbsamp = 10000
    else:
        gdat.indxsamp = np.arange(gdat.numbsamp)
    
    print('gdat.numbsamp')
    print(gdat.numbsamp)
    
    calc_feat_alle(gdat, typemodl)

    if gdat.boolsrchflar:
        gdat.arrytser['bdtrlowr'+typemodl] = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.listarrytser['bdtrlowr'+typemodl] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.arrytser['bdtrmedi'+typemodl] = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.listarrytser['bdtrmedi'+typemodl] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.arrytser['bdtruppr'+typemodl] = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.listarrytser['bdtruppr'+typemodl] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gdat.arrytser['Detrended'+typemodl] = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gdat.arrytser['modl'+typemodl] = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gdat.arrytser['resi'+typemodl] = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gdat.listarrytser['Detrended'+typemodl] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gdat.listarrytser['modl'+typemodl] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gdat.listarrytser['resi'+typemodl] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            gdat.arrytser['modl'+typemodl][b][p] = np.empty((gdat.time[b][p].size, 3))
            gdat.arrytser['modl'+typemodl][b][p][:, 0] = gdat.time[b][p]
            gdat.arrytser['modl'+typemodl][b][p][:, 1] = gdat.objtalle[typemodl].get_posterior_median_model(gdat.liststrginst[b][p], \
                                                                                                             gdat.liststrgdatatsercsvv[b], xx=gdat.time[b][p])
            gdat.arrytser['modl'+typemodl][b][p][:, 2] = 0.

            gdat.arrytser['resi'+typemodl][b][p] = np.copy(gdat.arrytser['Detrended'][b][p])
            gdat.arrytser['resi'+typemodl][b][p][:, 1] -= gdat.arrytser['modl'+typemodl][b][p][:, 1]
            for y in gdat.indxchun[b][p]:
                gdat.listarrytser['modl'+typemodl][b][p][y] = np.copy(gdat.listarrytser['Detrended'][b][p][y])
                gdat.listarrytser['modl'+typemodl][b][p][y][:, 1] = gdat.objtalle[typemodl].get_posterior_median_model(gdat.liststrginst[b][p], \
                                                                                                       gdat.liststrgdatatsercsvv[b], xx=gdat.listtime[b][p][y])
                
                gdat.listarrytser['resi'+typemodl][b][p][y] = np.copy(gdat.listarrytser['Detrended'][b][p][y])
                gdat.listarrytser['resi'+typemodl][b][p][y][:, 1] -= gdat.listarrytser['modl'+typemodl][b][p][y][:, 1]
    
                # plot residuals
                if gdat.boolplottser:
                    plot_tser_mile(gdat, strgmodl, b, p, y, 'resi' + typemodl)

    # write the model to file
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            path = gdat.pathdatatarg + 'arry%s_modl_%s.csv' % (gdat.liststrgdatatser[b], gdat.liststrginst[b][p])
            if not os.path.exists(path):
                if gdat.typeverb > 0:
                    print('Writing to %s...' % path)
                np.savetxt(path, gdat.arrytser['modl'+typemodl][b][p], delimiter=',', header=gdat.strgheadtser[b])

    # number of samples to plot
    gmod.arrypcur['DetrendedPrimaryCentered'+typemodl] = [[[[] for j in gmod.indxcomp] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gmod.arrypcur['DetrendedPrimaryCentered'+typemodl+'bindtotl'] = [[[[] for j in gmod.indxcomp] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gmod.arrypcur['DetrendedPrimaryCentered'+typemodl+'bindzoom'] = [[[[] for j in gmod.indxcomp] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    
    gdat.listarrypcur = dict()
    gdat.listarrypcur['quadmodl'+typemodl] = [[[[] for j in gmod.indxcomp] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            for j in gmod.indxcomp:
                gdat.listarrypcur['quadmodl'+typemodl][b][p][j] = np.empty((gdat.numbsampplot, gdat.numbtimeclen[b][p][j], 3))
    
    gmod.arrypcur['DetrendedPrimaryCentered'+typemodl] = [[[[] for j in gmod.indxcomp] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gmod.arrypcur['Primarymodltotl'+typemodl] = [[[[] for j in gmod.indxcomp] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gdat.arrytser['modlbase'+typemodl] = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gdat.listarrytser['modlbase'+typemodl] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    
    gdat.listarrytsermodl = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            gdat.listarrytsermodl[b][p] = np.empty((gdat.numbsampplot, gdat.numbtime[b][p], 3))
       
    for strgpcur in gdat.liststrgpcur:
        gdat.arrytser[strgpcur+typemodl] = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                gdat.arrytser[strgpcur+typemodl][b][p] = np.copy(gdat.arrytser['Detrended'][b][p])
    for strgpcurcomp in gdat.liststrgpcurcomp:
        gdat.arrytser[strgpcurcomp+typemodl] = [[[[] for j in gmod.indxcomp] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                for j in gmod.indxcomp:
                    gdat.arrytser[strgpcurcomp+typemodl][b][p][j] = np.copy(gdat.arrytser['Detrended'][b][p])
    for strgpcurcomp in gdat.liststrgpcurcomp + gdat.liststrgpcur:
        for strgextnbins in ['', 'bindtotl']:
            gmod.arrypcur['quad' + strgpcurcomp + typemodl + strgextnbins] = [[[[] for j in gmod.indxcomp] \
                                                                                    for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
        
            gdat.listlcurmodl = np.empty((gdat.numbsampplot, gdat.time[b][p].size))
            print('Phase-folding the posterior samples from the model light curve...')
            for ii in tqdm(range(gdat.numbsampplot)):
                i = gdat.indxsampplot[ii]
                
                # this is only the physical model and excludes the baseline, which is available separately via get_one_posterior_baseline()
                gdat.listarrytsermodl[b][p][ii, :, 1] = gdat.objtalle[typemodl].get_one_posterior_model(gdat.liststrginst[b][p], \
                                                                        gdat.liststrgdatatsercsvv[b], xx=gdat.time[b][p], sample_id=i)
                
                for j in gmod.indxcomp:
                    gdat.listarrypcur['quadmodl'+typemodl][b][p][j][ii, :, :] = \
                                            fold_tser(gdat.listarrytsermodl[b][p][ii, gmod.listindxtimeoutt[j][b][p], :, :], \
                                                                                   gdat.dicterrr['epocmtracomp'][0, j], gdat.dicterrr['pericomp'][0, j], phascntr=0.25)
                    
            ## plot components in the zoomed panel
            for j in gmod.indxcomp:
                
                gdat.objtalle[typemodl] = allesfitter.allesclass(gdat.pathalle[typemodl])
                ### total model for this planet
                gdat.arrytser['modltotl'+typemodl][b][p][j][:, 1] = gdat.objtalle[typemodl].get_posterior_median_model(gdat.liststrginst[b][p], \
                                                                                                                                'flux', xx=gdat.time[b][p])
                
                ### stellar baseline
                gdat.objtalle[typemodl] = allesfitter.allesclass(gdat.pathalle[typemodl])
                gdat.objtalle[typemodl].posterior_params_median['b_phase_curve_beaming_TESS'] = 0
                gdat.objtalle[typemodl].posterior_params_median['b_phase_curve_ellipsoidal_TESS'] = 0
                if typemodl == '0003':
                    gdat.objtalle[typemodl].posterior_params_median['b_phase_curve_atmospheric_TESS'] = 0
                if typemodl == '0004':
                    gdat.objtalle[typemodl].posterior_params_median['b_phase_curve_atmospheric_thermal_TESS'] = 0
                    gdat.objtalle[typemodl].posterior_params_median['b_phase_curve_atmospheric_reflected_TESS'] = 0
                gdat.objtalle[typemodl].posterior_params_median['b_sbratio_TESS'] = 0
                gdat.arrytser['modlstel'+typemodl][b][p][j][:, 1] = gdat.objtalle[typemodl].get_posterior_median_model(gdat.liststrginst[b][p], \
                                                                                                                                'flux', xx=gdat.time[b][p])
                
                ### EV
                gdat.objtalle[typemodl] = allesfitter.allesclass(gdat.pathalle[typemodl])
                gdat.objtalle[typemodl].posterior_params_median['b_sbratio_TESS'] = 0
                gdat.objtalle[typemodl].posterior_params_median['b_phase_curve_beaming_TESS'] = 0
                if typemodl == '0003':
                    gdat.objtalle[typemodl].posterior_params_median['b_phase_curve_atmospheric_TESS'] = 0
                if typemodl == '0004':
                    gdat.objtalle[typemodl].posterior_params_median['b_phase_curve_atmospheric_thermal_TESS'] = 0
                    gdat.objtalle[typemodl].posterior_params_median['b_phase_curve_atmospheric_reflected_TESS'] = 0
                gdat.arrytser['modlelli'+typemodl][b][p][j][:, 1] = gdat.objtalle[typemodl].get_posterior_median_model(gdat.liststrginst[b][p], \
                                                                                                                            'flux', xx=gdat.time[b][p])
                gdat.arrytser['modlelli'+typemodl][b][p][j][:, 1] -= gdat.arrytser['modlstel'+typemodl][b][p][j][:, 1]
                
                ### beaming
                gdat.objtalle[typemodl] = allesfitter.allesclass(gdat.pathalle[typemodl])
                gdat.objtalle[typemodl].posterior_params_median['b_sbratio_TESS'] = 0
                gdat.objtalle[typemodl].posterior_params_median['b_phase_curve_ellipsoidal_TESS'] = 0
                if typemodl == '0003':
                    gdat.objtalle[typemodl].posterior_params_median['b_phase_curve_atmospheric_TESS'] = 0
                if typemodl == '0004':
                    gdat.objtalle[typemodl].posterior_params_median['b_phase_curve_atmospheric_thermal_TESS'] = 0
                    gdat.objtalle[typemodl].posterior_params_median['b_phase_curve_atmospheric_reflected_TESS'] = 0
                gdat.arrytser['modlbeam'+typemodl][b][p][j][:, 1] = gdat.objtalle[typemodl].get_posterior_median_model(gdat.liststrginst[b][p], \
                                                                                                                            'flux', xx=gdat.time[b][p])
                gdat.arrytser['modlbeam'+typemodl][b][p][j][:, 1] -= gdat.arrytser['modlstel'+typemodl][b][p][j][:, 1]
                
                # planetary
                gdat.arrytser['modlplan'+typemodl][b][p][j][:, 1] = gdat.arrytser['modltotl'+typemodl][b][p][j][:, 1] \
                                                                      - gdat.arrytser['modlstel'+typemodl][b][p][j][:, 1] \
                                                                      - gdat.arrytser['modlelli'+typemodl][b][p][j][:, 1] \
                                                                      - gdat.arrytser['modlbeam'+typemodl][b][p][j][:, 1]
                
                offsdays = np.mean(gdat.arrytser['modlplan'+typemodl][b][p][j][gmod.listindxtimetran[j][b][p][1], 1])
                gdat.arrytser['modlplan'+typemodl][b][p][j][:, 1] -= offsdays

                # planetary nightside
                gdat.objtalle[typemodl] = allesfitter.allesclass(gdat.pathalle[typemodl])
                gdat.objtalle[typemodl].posterior_params_median['b_phase_curve_beaming_TESS'] = 0
                gdat.objtalle[typemodl].posterior_params_median['b_phase_curve_ellipsoidal_TESS'] = 0
                if typemodl == '0003':
                    gdat.objtalle[typemodl].posterior_params_median['b_phase_curve_atmospheric_TESS'] = 0
                else:
                    gdat.objtalle[typemodl].posterior_params_median['b_phase_curve_atmospheric_thermal_TESS'] = 0
                    gdat.objtalle[typemodl].posterior_params_median['b_phase_curve_atmospheric_reflected_TESS'] = 0
                gdat.arrytser['modlnigh'+typemodl][b][p][j][:, 1] = gdat.objtalle[typemodl].get_posterior_median_model(gdat.liststrginst[b][p], \
                                                                                                                            'flux', xx=gdat.time[b][p])
                gdat.arrytser['modlnigh'+typemodl][b][p][j][:, 1] += gdat.dicterrr['amplnigh'][0, 0]
                gdat.arrytser['modlnigh'+typemodl][b][p][j][:, 1] -= gdat.arrytser['modlstel'+typemodl][b][p][j][:, 1]
                
                ### planetary modulation
                gdat.arrytser['modlpmod'+typemodl][b][p][j][:, 1] = gdat.arrytser['modlplan'+typemodl][b][p][j][:, 1] - \
                                                                                    gdat.arrytser['modlnigh'+typemodl][b][p][j][:, 1]
                    
                ### planetary residual
                gdat.arrytser['bdtrplan'+typemodl][b][p][j][:, 1] = gdat.arrytser['Detrended'][b][p][:, 1] \
                                                                                - gdat.arrytser['modlstel'+typemodl][b][p][j][:, 1] \
                                                                                - gdat.arrytser['modlelli'+typemodl][b][p][j][:, 1] \
                                                                                - gdat.arrytser['modlbeam'+typemodl][b][p][j][:, 1]
                gdat.arrytser['bdtrplan'+typemodl][b][p][j][:, 1] -= offsdays
                    
            # get allesfitter baseline model
            gdat.arrytser['modlbase'+typemodl][b][p] = np.copy(gdat.arrytser['Detrended'][b][p])
            gdat.arrytser['modlbase'+typemodl][b][p][:, 1] = gdat.objtalle[typemodl].get_posterior_median_baseline(gdat.liststrginst[b][p], 'flux', \
                                                                                                                                xx=gdat.time[b][p])
            # get allesfitter-detrended data
            gdat.arrytser['Detrended'+typemodl][b][p] = np.copy(gdat.arrytser['Detrended'][b][p])
            gdat.arrytser['Detrended'+typemodl][b][p][:, 1] = gdat.arrytser['Detrended'][b][p][:, 1] - gdat.arrytser['modlbase'+typemodl][b][p][:, 1]
            for y in gdat.indxchun[b][p]:
                # get allesfitter baseline model
                gdat.listarrytser['modlbase'+typemodl][b][p][y] = np.copy(gdat.listarrytser['Detrended'][b][p][y])
                gdat.listarrytser['modlbase'+typemodl][b][p][y][:, 1] = gdat.objtalle[typemodl].get_posterior_median_baseline(gdat.liststrginst[b][p], \
                                                                                           'flux', xx=gdat.listarrytser['modlbase'+typemodl][b][p][y][:, 0])
                # get allesfitter-detrended data
                gdat.listarrytser['Detrended'+typemodl][b][p][y] = np.copy(gdat.listarrytser['Detrended'][b][p][y])
                gdat.listarrytser['Detrended'+typemodl][b][p][y][:, 1] = gdat.listarrytser['Detrended'+typemodl][b][p][y][:, 1] - \
                                                                                gdat.listarrytser['modlbase'+typemodl][b][p][y][:, 1]
           
            print('Phase folding and binning the light curve for inference named %s...' % typemodl)
            for j in gmod.indxcomp:
                
                gmod.arrypcur['Primarymodltotl'+typemodl][b][p][j] = fold_tser(gdat.arrytser['modltotl'+typemodl][b][p][j][gmod.listindxtimeoutt[j][b][p], :, :], \
                                                                                    gdat.dicterrr['epocmtracomp'][0, j], gdat.dicterrr['pericomp'][0, j])
                
                gmod.arrypcur['DetrendedPrimaryCentered'+typemodl][b][p][j] = fold_tser(gdat.arrytser['Detrended'+typemodl][b][p][gmod.listindxtimeoutt[j][b][p], :, :], \
                                                                                    gdat.dicterrr['epocmtracomp'][0, j], gdat.dicterrr['pericomp'][0, j])
                
                gmod.arrypcur['DetrendedPrimaryCentered'+typemodl+'bindtotl'][b][p][j] = rebn_tser(gmod.arrypcur['DetrendedPrimaryCentered'+typemodl][b][p][j], \
                                                                                                                    blimxdat=gdat.binsphasprimtotl)
                
                gmod.arrypcur['DetrendedPrimaryCentered'+typemodl+'bindzoom'][b][p][j] = rebn_tser(gmod.arrypcur['DetrendedPrimaryCentered'+typemodl][b][p][j], \
                                                                                                                    blimxdat=gmod.binsphasprimzoom[j])

                for strgpcurcomp in gdat.liststrgpcurcomp + gdat.liststrgpcur:
                    
                    arrytsertemp = gdat.arrytser[strgpcurcomp+typemodl][b][p][gmod.listindxtimeoutt[j][b][p], :, :]
                    
                    if strgpcurcomp == 'Detrended':
                        boolpost = True
                    else:
                        boolpost = False
                    gmod.arrypcur['quad'+strgpcurcomp+typemodl][b][p][j] = \
                                        fold_tser(arrytsertemp, gdat.dicterrr['epocmtracomp'][0, j], gdat.dicterrr['pericomp'][0, j], phascntr=0.25) 
                
                    gmod.arrypcur['quad'+strgpcurcomp+typemodl+'bindtotl'][b][p][j] = rebn_tser(gmod.arrypcur['quad'+strgpcurcomp+typemodl][b][p][j], \
                                                                                                                blimxdat=gdat.binsphasquadtotl)
                    
                    # write
                    path = gdat.pathdatatarg + 'arrypcur_quad_%sbindtotl_%s_%s.csv' % (strgpcurcomp, gdat.liststrgcomp[j], gdat.liststrginst[b][p])
                    if not os.path.exists(path):
                        if gdat.typeverb > 0:
                            print('Writing to %s...' % path)
                        np.savetxt(path, gmod.arrypcur['quad%s%sbindtotl' % (strgpcurcomp, typemodl)][b][p][j], delimiter=',', header=gdat.strgheadpser[b])
                    
                    if gdat.boolplot:
                        plot_pser_mile(gdat, strgmodl, 'quad'+strgpcurcomp+typemodl, boolpost=boolpost)
                
                
    # plots
    ## plot GP-detrended phase curves
    if gdat.boolplottser:
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                for y in gdat.indxchun[b][p]:
                    plot_tser_mile(gdat, strgmodl, b, p, y, 'Detrended'+typemodl)
        plot_pser_mile(gdat, strgmodl, 'DetrendedPrimaryCentered'+typemodl, boolpost=True)
    if gdat.boolplotpopl:
        plot_popl(gdat, gdat.typepriocomp + typemodl)
    
    # print out transit times
    for j in gmod.indxcomp:
        print(gdat.liststrgcomp[j])
        time = np.empty(500)
        for n in range(500):
            time[n] = gdat.dicterrr['epocmtracomp'][0, j] + gdat.dicterrr['pericomp'][0, j] * n
        objttime = astropy.time.Time(time, format='jd', scale='utc')#, out_subfmt='date_hm')
        listtimelabl = objttime.iso
        for n in range(500):
            if time[n] > 2458788 and time[n] < 2458788 + 200:
                print('%f, %s' % (time[n], listtimelabl[n]))


    if typemodl == '0003' or typemodl == '0004':
            
        if typemodl == '0003':
            gmod.listlablpara = [['Nightside', 'ppm'], ['Secondary', 'ppm'], ['Planetary Modulation', 'ppm'], ['Thermal', 'ppm'], \
                                                        ['Reflected', 'ppm'], ['Phase shift', 'deg'], ['Geometric Albedo', '']]
        else:
            gmod.listlablpara = [['Nightside', 'ppm'], ['Secondary', 'ppm'], ['Thermal', 'ppm'], \
                                  ['Reflected', 'ppm'], ['Thermal Phase shift', 'deg'], ['Reflected Phase shift', 'deg'], ['Geometric Albedo', '']]
        numbpara = len(gmod.listlablpara)
        indxpara = np.arange(numbpara)
        listpost = np.empty((gdat.numbsamp, numbpara))
        
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                for j in gmod.indxcomp:
                    listpost[:, 0] = gdat.dictlist['amplnigh'][:, j] * 1e6 # [ppm]
                    listpost[:, 1] = gdat.dictlist['amplseco'][:, j] * 1e6 # [ppm]
                    if typemodl == '0003':
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
                    tdpy.plot_grid(gdat.pathalle[typemodl], 'pcur_%s' % typemodl, listpost, gmod.listlablpara, plotsize=2.5)

        # plot phase curve
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                
                ## determine data gaps for overplotting model without the data gaps
                gdat.indxtimegapp = np.argmax(gdat.time[b][p][1:] - gdat.time[b][p][:-1]) + 1
                
                for j in gmod.indxcomp:
                    path = gdat.pathalle[typemodl] + 'pcur_grid_%s_%s_%s.%s' % (typemodl, gdat.strgtarg, gdat.typepriocomp, gdat.typefileplot)
                    gdat.listdictdvrp[j+2].append({'path': path, 'limt':[0.4, 0.05, 0.5, 0.1]})
                    if not os.path.exists(path):
                        figr = plt.figure(figsize=(10, 12))
                        axis = [[] for k in range(3)]
                        axis[0] = figr.add_subplot(3, 1, 1)
                        axis[1] = figr.add_subplot(3, 1, 2)
                        axis[2] = figr.add_subplot(3, 1, 3, sharex=axis[1])
                        
                        for k in range(len(axis)):
                            
                            ## unbinned data
                            if k < 2:
                                if k == 0:
                                    xdat = gdat.time[b][p] - gdat.timeoffs
                                    ydat = gdat.arrytser['Detrended'+typemodl][b][p][:, 1] + gdat.dicterrr['amplnigh'][0, 0]
                                if k == 1:
                                    xdat = gmod.arrypcur['DetrendedQuadratureCentered'+typemodl][b][p][j][:, 0]
                                    ydat = gmod.arrypcur['DetrendedQuadratureCentered'+typemodl][b][p][j][:, 1] + gdat.dicterrr['amplnigh'][0, 0]
                                axis[k].plot(xdat, ydat, '.', color='grey', alpha=0.3, label='Raw data')
                            
                            ## binned data
                            if k > 0:
                                xdat = gmod.arrypcur['DetrendedQuadratureCentered'+typemodl+'bindtotl'][b][p][j][:, 0]
                                ydat = gmod.arrypcur['DetrendedQuadratureCentered'+typemodl+'bindtotl'][b][p][j][:, 1] + gdat.dicterrr['amplnigh'][0, 0]
                                yerr = np.copy(gmod.arrypcur['DetrendedQuadratureCentered'+typemodl+'bindtotl'][b][p][j][:, 2])
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
                                xdat = gmod.arrypcur['quadmodl'+typemodl][b][p][j][:, 0]
                                ydat = gmod.arrypcur['quadmodl'+typemodl][b][p][j][:, 1] + gdat.dicterrr['amplnigh'][0, 0]
                            else:
                                xdat = gdat.arrytser['modltotl'+typemodl][b][p][j][:, 0] - gdat.timeoffs
                                ydat = gdat.arrytser['modltotl'+typemodl][b][p][j][:, 1] + gdat.dicterrr['amplnigh'][0, 0]
                            if k == 2:
                                ydat = (ydat - 1) * 1e6
                            if k == 0:
                                axis[k].plot(xdat[:gdat.indxtimegapp], ydat[:gdat.indxtimegapp], color='b', lw=2, label='Total Model', zorder=10)
                                axis[k].plot(xdat[gdat.indxtimegapp:], ydat[gdat.indxtimegapp:], color='b', lw=2, zorder=10)
                            else:
                                axis[k].plot(xdat, ydat, color='b', lw=2, label='Model', zorder=10)
                            
                            if k == 0:
                                axis[k].set(xlabel='Time [BJD - %d]' % gdat.timeoffs)
                            if k > 0:
                                axis[k].set(xlabel='Phase')
                        axis[0].set(ylabel=gdat.labltserphot)
                        axis[1].set(ylabel=gdat.labltserphot)
                        axis[2].set(ylabel='Relative flux - 1 [ppm]')
                        
                        axis[2].set_ylim(ylimpcur)
                        
                        xdat = gmod.arrypcur['quadmodlstel'+typemodl][b][p][j][:, 0]
                        ydat = (gmod.arrypcur['quadmodlstel'+typemodl][b][p][j][:, 1] - 1.) * 1e6
                        axis[2].plot(xdat, ydat, lw=2, color='orange', label='Stellar baseline', ls='--', zorder=11)
                        
                        xdat = gmod.arrypcur['quadmodlelli'+typemodl][b][p][j][:, 0]
                        ydat = (gmod.arrypcur['quadmodlelli'+typemodl][b][p][j][:, 1] - 1.) * 1e6
                        axis[2].plot(xdat, ydat, lw=2, color='r', ls='--', label='Ellipsoidal variation')
                        
                        xdat = gmod.arrypcur['quadmodlelli'+typemodl][b][p][j][:, 0]
                        ydat = (gmod.arrypcur['quadmodlelli'+typemodl][b][p][j][:, 1] - 1.) * 1e6
                        axis[2].plot(xdat, ydat, lw=2, color='r', ls='--', label='Ellipsoidal variation')
                        
                        xdat = gmod.arrypcur['quadmodlplan'+typemodl][b][p][j][:, 0]
                        ydat = (gmod.arrypcur['quadmodlplan'+typemodl][b][p][j][:, 1] - 1.) * 1e6
                        axis[2].plot(xdat, ydat, lw=2, color='g', label='Planetary', ls='--')
    
                        xdat = gmod.arrypcur['quadmodlnigh'+typemodl][b][p][j][:, 0]
                        ydat = (gmod.arrypcur['quadmodlnigh'+typemodl][b][p][j][:, 1] - 1.) * 1e6
                        axis[2].plot(xdat, ydat, lw=2, color='olive', label='Planetary baseline', ls='--', zorder=11)
    
                        xdat = gmod.arrypcur['quadmodlpmod'+typemodl][b][p][j][:, 0]
                        ydat = (gmod.arrypcur['quadmodlpmod'+typemodl][b][p][j][:, 1] - 1.) * 1e6
                        axis[2].plot(xdat, ydat, lw=2, color='m', label='Planetary modulation', ls='--', zorder=11)
                         
                        ## legend
                        axis[2].legend(ncol=3)
                        
                        if gdat.typeverb > 0:
                            print('Writing to %s...' % path)
                        plt.savefig(path)
                        plt.close()
                   

        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                for j in gmod.indxcomp:
        
                    path = gdat.pathalle[typemodl] + 'pcur_samp_%s_%s_%s.%s' % (typemodl, gdat.strgtarg, gdat.typepriocomp, gdat.typefileplot)
                    if not os.path.exists(path):
                        # replot phase curve
                        ### sample model phas
                        #numbphasfine = 1000
                        #gdat.meanphasfine = np.linspace(np.amin(gmod.arrypcur['DetrendedQuadratureCentered'][0][gdat.indxphasotpr, 0]), \
                        #                                np.amax(gmod.arrypcur['DetrendedQuadratureCentered'][0][gdat.indxphasotpr, 0]), numbphasfine)
                        #indxphasfineinse = np.where(abs(gdat.meanphasfine - 0.5) < phasseco)[0]
                        #indxphasfineotprleft = np.where(-gdat.meanphasfine > phasmask)[0]
                        #indxphasfineotprrght = np.where(gdat.meanphasfine > phasmask)[0]
       
                        indxphasmodlouttprim = [[] for a in range(2)]
                        indxphasdatabindouttprim = [[] for a in range(2)]
                        indxphasmodlouttprim[0] = np.where(gmod.arrypcur['quadmodl'+typemodl][b][p][j][:, 0] < -0.05)[0]
                        indxphasdatabindouttprim[0] = np.where(gmod.arrypcur['DetrendedQuadratureCenteredBinned'][b][p][j][:, 0] < -0.05)[0]
                        indxphasmodlouttprim[1] = np.where(gmod.arrypcur['quadmodl'+typemodl][b][p][j][:, 0] > 0.05)[0]
                        indxphasdatabindouttprim[1] = np.where(gmod.arrypcur['DetrendedQuadratureCenteredBinned'][b][p][j][:, 0] > 0.05)[0]

                    path = gdat.pathalle[typemodl] + 'pcur_comp_%s_%s_%s.%s' % (typemodl, gdat.strgtarg, gdat.typepriocomp, gdat.typefileplot)
                    gdat.listdictdvrp[j+2].append({'path': path, 'limt':[0., 0.05, 0.5, 0.1]})
                    if not os.path.exists(path):
                        # plot the phase curve with components
                        figr, axis = plt.subplots(figsize=gdat.figrsizeydob)
                        ## data
                        axis.errorbar(gmod.arrypcur['DetrendedQuadratureCenteredBinned'][b][p][j][:, 0], \
                                       (gmod.arrypcur['DetrendedQuadratureCenteredBinned'][b][p][j][:, 1] + gdat.dicterrr['amplnigh'][0, 0] - 1) * 1e6, \
                                       yerr=1e6*gmod.arrypcur['DetrendedQuadratureCenteredBinned'][b][p][j][:, 2], color='k', marker='o', ls='', markersize=2, lw=1, label='Data')
                        ## total model
                        axis.plot(gmod.arrypcur['quadmodl'+typemodl][b][p][j][:, 0], \
                                                        1e6*(gmod.arrypcur['quadmodl'+typemodl][b][p][j][:, 1]+gdat.dicterrr['amplnigh'][0, 0]-1), \
                                                                                                                        color='b', lw=3, label='Model')
                        
                        axis.plot(gmod.arrypcur['quadmodlplan'+typemodl][b][p][j][:, 0], 1e6*(gmod.arrypcur['quadmodlplan'+typemodl][b][p][j][:, 1]), \
                                                                                                                      color='g', label='Planetary', lw=1, ls='--')
                        
                        axis.plot(gmod.arrypcur['quadmodlbeam'+typemodl][b][p][j][:, 0], 1e6*(gmod.arrypcur['quadmodlbeam'+typemodl][b][p][j][:, 1]), \
                                                                                                              color='m', label='Beaming', lw=2, ls='--')
                        
                        axis.plot(gmod.arrypcur['quadmodlelli'+typemodl][b][p][j][:, 0], 1e6*(gmod.arrypcur['quadmodlelli'+typemodl][b][p][j][:, 1]), \
                                                                                                              color='r', label='Ellipsoidal variation', lw=2, ls='--')
                        
                        axis.plot(gmod.arrypcur['quadmodlstel'+typemodl][b][p][j][:, 0], 1e6*(gmod.arrypcur['quadmodlstel'+typemodl][b][p][j][:, 1]-1.), \
                                                                                                              color='orange', label='Stellar baseline', lw=2, ls='--')
                        
                        axis.set_ylim(ylimpcur)
                        axis.set_ylabel('Relative flux [ppm]')
                        axis.set_xlabel('Phase')
                        axis.legend(ncol=3)
                        plt.tight_layout()
                        if gdat.typeverb > 0:
                            print('Writing to %s...' % path)
                        plt.savefig(path)
                        plt.close()

                    path = gdat.pathalle[typemodl] + 'pcur_samp_%s_%s_%s.%s' % (typemodl, gdat.strgtarg, gdat.typepriocomp, gdat.typefileplot)
                    gdat.listdictdvrp[j+2].append({'path': path, 'limt':[0., 0.05, 0.5, 0.1]})
                    if not os.path.exists(path):
                        # plot the phase curve with samples
                        figr, axis = plt.subplots(figsize=gdat.figrsizeydob)
                        axis.errorbar(gmod.arrypcur['DetrendedQuadratureCenteredBinned'][b][p][j][:, 0], \
                                    (gmod.arrypcur['DetrendedQuadratureCenteredBinned'][b][p][j][:, 1] + gdat.dicterrr['amplnigh'][0, 0] - 1) * 1e6, \
                                                     yerr=1e6*gmod.arrypcur['DetrendedQuadratureCenteredBinned'][b][p][j][:, 2], color='k', marker='o', ls='', markersize=2, lw=1)
                        for ii, i in enumerate(gdat.indxsampplot):
                            axis.plot(gmod.arrypcur['quadmodl'+typemodl][b][p][j][:, 0], \
                                                        1e6 * (gdat.listarrypcur['quadmodl'+typemodl][b][p][j][ii, :] + gdat.dicterrr['amplnigh'][0, 0] - 1.), \
                                                                                                                                          alpha=0.1, color='b')
                        axis.set_ylabel('Relative flux [ppm]')
                        axis.set_xlabel('Phase')
                        axis.set_ylim(ylimpcur)
                        plt.tight_layout()
                        if gdat.typeverb > 0:
                            print('Writing to %s...' % path)
                        plt.savefig(path)
                        plt.close()

                    # plot all along with residuals
                    #path = gdat.pathalle[typemodl] + 'pcur_resi_%s_%s_%s.%s' % (typemodl, gdat.strgtarg, gdat.typepriocomp, gdat.typefileplot)
                    #gdat.listdictdvrp[j+1].append({'path': path, 'limt':[0.4, 0.05, 0.5, 0.1]})
                    #if not os.path.exists(path):
                    #   figr, axis = plt.subplots(3, 1, figsize=gdat.figrsizeydob)
                    #   axis.errorbar(gmod.arrypcur['DetrendedQuadratureCenteredBinned'][b][p][j][:, 0], (gmod.arrypcur['DetrendedQuadratureCenteredBinned'][b][p][j][:, 1]) * 1e6, \
                    #                          yerr=1e6*gmod.arrypcur['DetrendedQuadratureCenteredBinned'][b][p][j][:, 2], color='k', marker='o', ls='', markersize=2, lw=1)
                    #   for kk, k in enumerate(gdat.indxsampplot):
                    #       axis.plot(gdat.meanphasfine[indxphasfineotprleft], (listmodltotl[k, indxphasfineotprleft] - listoffs[k]) * 1e6, \
                    #                                                                                                               alpha=0.1, color='b')
                    #       axis.plot(gdat.meanphasfine[indxphasfineotprrght], (listmodltotl[k, indxphasfineotprrght] - listoffs[k]) * 1e6, \
                    #                                                                                                               alpha=0.1, color='b')
                    #   axis.set_ylabel('Relative flux - 1 [ppm]')
                    #   axis.set_xlabel('Phase')
                    #   plt.tight_layout()
                    #   print('Writing to %s...' % path)
                    #   plt.savefig(path)
                    #   plt.close()

                    # write to text file
                    path = gdat.pathalle[typemodl] + 'post_pcur_%s_tabl.csv' % (typemodl)
                    if not os.path.exists(path):
                        fileoutp = open(gdat.pathalle[typemodl] + 'post_pcur_%s_tabl.csv' % (typemodl), 'w')
                        for strgfeat in gdat.dictlist:
                            if gdat.dictlist[strgfeat].ndim == 2:
                                for j in gmod.indxcomp:
                                    fileoutp.write('%s,%s,%g,%g,%g,%g,%g\\\\\n' % (strgfeat, gdat.liststrgcomp[j], gdat.dictlist[strgfeat][0, j], gdat.dictlist[strgfeat][1, j], \
                                                                                gdat.dictlist[strgfeat][2, j], gdat.dicterrr[strgfeat][1, j], gdat.dicterrr[strgfeat][2, j]))
                            else:
                                fileoutp.write('%s,,%g,%g,%g,%g,%g\\\\\n' % (strgfeat, gdat.dictlist[strgfeat][0], gdat.dictlist[strgfeat][1], \
                                                                                gdat.dictlist[strgfeat][2], gdat.dicterrr[strgfeat][1], gdat.dicterrr[strgfeat][2]))
                            #fileoutp.write('\\\\\n')
                        fileoutp.close()
                    
                    path = gdat.pathalle[typemodl] + 'post_pcur_%s_cmnd.csv' % (typemodl)
                    if not os.path.exists(path):
                        fileoutp = open(gdat.pathalle[typemodl] + 'post_pcur_%s_cmnd.csv' % (typemodl), 'w')
                        for strgfeat in gdat.dictlist:
                            if gdat.dictlist[strgfeat].ndim == 2:
                                for j in gmod.indxcomp:
                                    fileoutp.write('%s,%s,$%.3g \substack{+%.3g \\\\ -%.3g}$\\\\\n' % (strgfeat, gdat.liststrgcomp[j], gdat.dicterrr[strgfeat][0, j], \
                                                                                                gdat.dicterrr[strgfeat][1, j], gdat.dicterrr[strgfeat][2, j]))
                            else:
                                fileoutp.write('%s,,$%.3g \substack{+%.3g \\\\ -%.3g}$\\\\\n' % (strgfeat, gdat.dicterrr[strgfeat][0], \
                                                                                                            gdat.dicterrr[strgfeat][1], gdat.dicterrr[strgfeat][2]))
                            #fileoutp.write('\\\\\n')
                        fileoutp.close()

                if typemodl == '0003':
                    
                    # wavelength axis
                    gdat.conswlentmpt = 0.0143877735e6 # [um K]

                    minmalbg = min(np.amin(gdat.dictlist['albginfo']), np.amin(gdat.dictlist['albg']))
                    maxmalbg = max(np.amax(gdat.dictlist['albginfo']), np.amax(gdat.dictlist['albg']))
                    binsalbg = np.linspace(minmalbg, maxmalbg, 100)
                    meanalbg = (binsalbg[1:] + binsalbg[:-1]) / 2.
                    pdfnalbg = tdpy.retr_kdegpdfn(gdat.dictlist['albg'][:, 0], binsalbg, 0.02)
                    pdfnalbginfo = tdpy.retr_kdegpdfn(gdat.dictlist['albginfo'][:, 0], binsalbg, 0.02)
                    
                    path = gdat.pathalle[typemodl] + 'pdfn_albg_%s_%s.%s' % (gdat.strgtarg, gdat.typepriocomp, gdat.typefileplot)
                    gdat.listdictdvrp[j+1].append({'path': path, 'limt':[0.4, 0.05, 0.5, 0.1]})
                    if not os.path.exists(path):
                        figr, axis = plt.subplots(figsize=gdat.figrsizeydob)
                        axis.plot(meanalbg, pdfnalbg, label='TESS only', lw=2)
                        axis.plot(meanalbg, pdfnalbginfo, label='TESS + ATMO', lw=2)
                        axis.set_xlabel('$A_g$')
                        axis.set_ylabel('$P(A_g)$')
                        axis.legend()
                        axis.set_xlim([0, None])
                        plt.subplots_adjust()
                        if gdat.typeverb > 0:
                            print('Writing to %s...' % path)
                        plt.savefig(path)
                        plt.close()
                
                    path = gdat.pathalle[typemodl] + 'hist_albg_%s_%s.%s' % (gdat.strgtarg, gdat.typepriocomp, gdat.typefileplot)
                    gdat.listdictdvrp[j+1].append({'path': path, 'limt':[0.4, 0.05, 0.5, 0.1]})
                    if not os.path.exists(path):
                        figr, axis = plt.subplots(figsize=gdat.figrsizeydob)
                        axis.hist(gdat.dictlist['albg'][:, 0], label='TESS only', bins=binsalbg)
                        axis.hist(gdat.dictlist['albginfo'][:, 0], label='TESS + ATMO', bins=binsalbg)
                        axis.set_xlabel('$A_g$')
                        axis.set_ylabel('$N(A_g)$')
                        axis.legend()
                        plt.subplots_adjust()
                        if gdat.typeverb > 0:
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
                    path = gdat.pathdatatarg + 'ascii_output/RetrievalParamSamples.txt'
                    listsampatmo = np.loadtxt(path)
                    
                    # plot ATMO posterior
                    gmod.listlablpara = [['$\kappa_{IR}$', ''], ['$\gamma$', ''], ['$\psi$', ''], ['[M/H]', ''], \
                                                                                                    ['[C/H]', ''], ['[O/H]', '']]
                    tdpy.plot_grid(gdat.pathalle[typemodl], 'post_atmo', listsampatmo, gmod.listlablpara, plotsize=2.5)
   
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
                    path = gdat.pathalle[typemodl] + 'kdeg_psii_%s_%s.%s' % (gdat.strgtarg, gdat.typepriocomp, gdat.typefileplot)
                    gdat.listdictdvrp[j+1].append({'path': path, 'limt':[0.4, 0.05, 0.5, 0.1]})
                    if not os.path.exists(path):
                        figr, axis = plt.subplots(figsize=gdat.figrsizeydob)
                        gdat.kdegpsii = tdpy.retr_kdeg(gdat.listpsii, gdat.meanpsii, gdat.kdegstdvpsii)
                        axis.plot(gdat.meanpsii, gdat.kdegpsii)
                        axis.set_xlabel('$\psi$')
                        axis.set_ylabel('$K_\psi$')
                        plt.subplots_adjust()
                        if gdat.typeverb > 0:
                            print('Writing to %s...' % path)
                        plt.savefig(path)
                        plt.close()
                
                    # use psi posterior to infer Bond albedo and heat circulation efficiency
                    numbsampwalk = 10000
                    numbsampburnwalk = 1000
                    gmod.listlablpara = [['$A_b$', ''], ['$E$', ''], [r'$\varepsilon$', '']]
                    listscalpara = ['self', 'self', 'self']
                    gmod.listminmpara = np.array([0., 0., 0.])
                    gmod.listmaxmpara = np.array([1., 1., 1.])
                    strgextn = 'albbepsi'
                    listpostheat = tdpy.samp(gdat, numbsampwalk, retr_llik_albbepsi, \
                                             gmod.listlablpara, listscalpara, gmod.listminmpara, gmod.listmaxmpara, boolplot=gdat.boolplot, \
                                             pathbase=gdat.pathtargcnfg, \
                                             typeverb=gdat.typeverb, \
                                             numbsampburnwalk=numbsampburnwalk, strgextn=strgextn, \
                                            )

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
                    axistwin.plot(gdat.cntrwlenband, gdat.thptband, color='grey', ls='--', label='TESS')
                    axistwin.set_ylabel(r'Throughput')
                    
                    ## secondary eclipse depths
                    ### model
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
                    path = gdat.pathalle[typemodl] + 'spec_%s_%s.%s' % (gdat.strgtarg, gdat.typepriocomp, gdat.typefileplot)
                    gdat.listdictdvrp[j+1].append({'path': path, 'limt':[0.4, 0.05, 0.5, 0.1]})
                    if gdat.typeverb > 0:
                        print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()
                    
                    # get contribution function
                    path = gdat.pathdatatarg + 'ascii_output/ContribFuncArr.txt'
                    if gdat.typeverb > 0:
                        print('Reading from %s...' % path)
                    ctrb = np.loadtxt(path)
                    presctrb = ctrb[0, :]
                    # interpolate the throughput
                    gdat.thptbandctrb = scipy.interpolate.interp1d(gdat.cntrwlenband, gdat.thptband, fill_value=0, bounds_error=False)(wlenctrb)
                    numbwlenctrb = wlenctrb.size
                    indxwlenctrb = np.arange(numbwlenctrb)
                    numbpresctrb = presctrb.size
                    indxpresctrb = np.arange(numbpresctrb)

                    # plot pressure-temperature, contribution function, abundances
                    ## get ATMO posterior
                    path = gdat.pathdatatarg + 'ascii_output/RetrievalPTSamples.txt'
                    dataptem = np.loadtxt(path)
                    liststrgcomp = ['CH4.txt', 'CO.txt', 'FeH.txt', 'H+.txt', 'H.txt', 'H2.txt', 'H2O.txt', 'H_.txt', 'He.txt', 'K+.txt', \
                                                                        'K.txt', 'NH3.txt', 'Na+.txt', 'Na.txt', 'TiO.txt', 'VO.txt', 'e_.txt']
                    listlablcomp = ['CH$_4$', 'CO', 'FeH', 'H$^+$', 'H', 'H$_2$', 'H$_2$O', 'H$^-$', 'He', 'K$^+$', \
                                                                        'K', 'NH$_3$', 'Na$^+$', 'Na', 'TiO', 'VO', 'e$^-$']
                    listdatacomp = []
                    for strg in liststrgcomp:
                        path = gdat.pathdatatarg + 'ascii_output/pdependent_abundances/' + strg
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
                    path = gdat.pathalle[typemodl] + 'ptem_%s_%s.%s' % (gdat.strgtarg, gdat.typepriocomp, gdat.typefileplot)
                    gdat.listdictdvrp[j+1].append({'path': path, 'limt':[0.4, 0.05, 0.5, 0.1]})
                    if gdat.typeverb > 0:
                        print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()
  

def plot_popl(gdat, strgpdfn):
    
    print('Plotting target features along with population features for strgpdfn: %s' % strgpdfn)
        
    pathvisufeatplan = getattr(gdat, 'pathvisufeatplan' + strgpdfn)
    pathvisudataplan = getattr(gdat, 'pathvisudataplan' + strgpdfn)
    pathvisufeatsyst = getattr(gdat, 'pathvisufeatsyst' + strgpdfn)
    
    ## occurence rate as a function of planet radius with highlighted radii of the system's planets
    ### get the CKS occurence rate as a function of planet radius
    path = gdat.pathbasemile + 'data/Fulton+2017/Means.csv'
    data = np.loadtxt(path, delimiter=',')
    timeoccu = data[:, 0]
    occumean = data[:, 1]
    path = gdat.pathbasemile + 'data/Fulton+2017/Lower.csv'
    occulowr = np.loadtxt(path, delimiter=',')
    occulowr = occulowr[:, 1]
    path = gdat.pathbasemile + 'data/Fulton+2017/Upper.csv'
    occuuppr = np.loadtxt(path, delimiter=',')
    occuuppr = occuuppr[:, 1]
    occuyerr = np.empty((2, occumean.size))
    occuyerr[0, :] = occuuppr - occumean
    occuyerr[1, :] = occumean - occulowr
    
    figr, axis = plt.subplots(figsize=gdat.figrsize)
    
    # this system
    for jj, j in enumerate(gmod.indxcomp):
        if strgpdfn == 'post':
            xposlowr = gdat.dictpost['radicomp'][0, j]
            xposmedi = gdat.dictpost['radicomp'][1, j]
            xposuppr = gdat.dictpost['radicomp'][2, j]
        else:
            xposmedi = gdat.fitt.prio.meanpara.rratcomp[j] * gdat.radistar
            xposlowr = xposmedi - gdat.stdvrratcompprio[j] * gdat.radistar
            xposuppr = xposmedi + gdat.stdvrratcompprio[j] * gdat.radistar
        xposlowr *= gdat.dictfact['rjre']
        xposuppr *= gdat.dictfact['rjre']
        axis.axvspan(xposlowr, xposuppr, alpha=0.5, color=gdat.listcolrcomp[j])
        axis.axvline(xposmedi, color=gdat.listcolrcomp[j], ls='--', label=gdat.liststrgcomp[j])
        axis.text(0.7, 0.9 - jj * 0.07, r'\textbf{%s}' % gdat.liststrgcomp[j], color=gdat.listcolrcomp[j], \
                                                                                    va='center', ha='center', transform=axis.transAxes)
    
    if typeplotback == 'white':
        colrbkgd = 'white'
        colrdraw = 'black'
    elif typeplotback == 'black':
        colrbkgd = 'black'
        colrdraw = 'white'
    
    # plot the occurrence rate
    xerr = (timeoccu[1:] - timeoccu[:-1]) / 2.
    xerr = np.concatenate([xerr[0, None], xerr])
    axis.errorbar(timeoccu, occumean, yerr=occuyerr, xerr=xerr, color=colrdraw, ls='', marker='o', lw=1, zorder=10)
    axis.set_xlabel('Radius [$R_E$]')
    axis.set_ylabel('Occurrence rate of planets per star')
    
    plt.subplots_adjust(bottom=0.2)
    plt.subplots_adjust(left=0.2)
    path = pathvisufeatplan + 'occuradi_%s_%s.%s' % (gdat.strgtarg, strgpdfn, gdat.typefileplot)
    #gdat.listdictdvrp[j+1].append({'path': path, 'limt':[0.4, 0.05, 0.5, 0.1]})
    print('Writing to %s...' % path)
    plt.savefig(path)
    plt.close()
  
    for strgpopl in gdat.liststrgpopl:
        
        if strgpopl == 'exar':
            dictpopl = gdat.dictexar
        else:
            dictpopl = gdat.dicttoii
        
        numbcomppopl = dictpopl['radicomp'].size
        indxtargpopl = np.arange(numbcomppopl)

        ### TSM and ESM
        numbsamppopl = 100
        dictlistplan = dict()
        for strgfeat in gdat.listfeatstarpopl:
            dictlistplan[strgfeat] = np.zeros((numbsamppopl, dictpopl['masscomp'].size)) + np.nan
            for k in range(dictpopl[strgfeat].size):
                meanvarb = dictpopl[strgfeat][k]
                if not np.isfinite(meanvarb):
                    continue
                if np.isfinite(dictpopl['stdv' + strgfeat][k]):
                    stdvvarb = dictpopl['stdv' + strgfeat][k]
                else:
                    stdvvarb = 0.
                
                dictlistplan[strgfeat][:, k] = tdpy.samp_gaustrun(numbsamppopl, meanvarb, stdvvarb, 0., np.inf)
                dictlistplan[strgfeat][:, k] /= np.mean(dictlistplan[strgfeat][:, k])
                dictlistplan[strgfeat][:, k] *= meanvarb
                
        #### TSM
        listtsmm = nicomedia.retr_tsmm(dictlistplan['radicomp'], dictlistplan['tmptplan'], dictlistplan['masscomp'], \
                                                                                        dictlistplan['radistar'], dictlistplan['jmagsyst'])

        #### ESM
        listesmm = nicomedia.retr_esmm(dictlistplan['tmptplan'], dictlistplan['tmptstar'], dictlistplan['radicomp'], dictlistplan['radistar'], \
                                                                                                                    dictlistplan['kmagsyst'])
        ## augment the 
        dictpopl['stdvtsmm'] = np.std(listtsmm, 0)
        dictpopl['tsmm'] = np.nanmedian(listtsmm, 0)
        dictpopl['stdvesmm'] = np.std(listesmm, 0)
        dictpopl['esmm'] = np.nanmedian(listesmm, 0)
        
        dictpopl['vesc'] = nicomedia.retr_vesc(dictpopl['masscomp'], dictpopl['radicomp'])
        dictpopl['vesc0060'] = dictpopl['vesc'] / 6.
        
        objticrs = astropy.coordinates.SkyCoord(ra=dictpopl['rascstar']*astropy.units.degree, \
                                               dec=dictpopl['declstar']*astropy.units.degree, frame='icrs')
        
        # galactic longitude
        dictpopl['lgalstar'] = np.array([objticrs.galactic.l])[0, :]
        
        # galactic latitude
        dictpopl['bgalstar'] = np.array([objticrs.galactic.b])[0, :]
        
        # ecliptic longitude
        dictpopl['loecstar'] = np.array([objticrs.barycentricmeanecliptic.lon.degree])[0, :]
        
        # ecliptic latitude
        dictpopl['laecstar'] = np.array([objticrs.barycentricmeanecliptic.lat.degree])[0, :]

        dictpopl['stnomass'] = dictpopl['masscomp'] / dictpopl['stdvmasscomp']

        #dictpopl['boollive'] = ~dictpopl['boolfpos']
        dictpopl['boolterr'] = dictpopl['radicomp'] < 1.8
        dictpopl['boolhabicons'] = (dictpopl['inso'] < 1.01) & (dictpopl['inso'] > 0.35)
        dictpopl['boolhabiopti'] = (dictpopl['inso'] < 1.78) & (dictpopl['inso'] > 0.29)
        # unlocked
        dictpopl['boolunlo'] = np.log10(dictpopl['massstar']) < (-2 + 3 * (np.log10(dictpopl['smax']) + 1))
        # Earth as a transiting planet
        dictpopl['booleatp'] = abs(dictpopl['laecstar']) < 0.25

        ## Hill sphere
        ## angular momentum
    
        dictpopl['sage'] = 1. / 365.2422 / 24. / 3600. * (1. / 486.) * (0.008406 * dictpopl['smax'] / 0.027 / dictpopl['massstar']**(1. / 3.))**6
        dictpopl['timelock'] = (1. / 486.) * (dictpopl['smax'] / 0.027 / dictpopl['massstar']**(1. / 3.))**6

        #for strg in dictpopl.keys():
        #    print(strg)
        #    summgene(dictpopl[strg])
        #    print('')
        
        # from SETI
        dictpopl['metrplan'] = (0.99 * np.heaviside(dictpopl['numbplantranstar'] - 2, 1.) + 0.01)
        dictpopl['metrhzon'] = (0.99 * dictpopl['boolhabiopti'].astype(float) + 0.01)
        dictpopl['metrunlo'] = (0.99 * dictpopl['boolunlo'].astype(float) + 0.01)
        dictpopl['metrterr'] = (0.99 * dictpopl['boolterr'].astype(float) + 0.01)
        dictpopl['metrhabi'] = dictpopl['metrunlo'] * dictpopl['metrhzon'] * dictpopl['metrterr']
        dictpopl['metrseti'] = dictpopl['metrhabi'] * dictpopl['metrplan'] * dictpopl['distsyst']**(-2.)
        dictpopl['metrhzon'] /= np.nanmax(dictpopl['metrhzon'])
        dictpopl['metrhabi'] /= np.nanmax(dictpopl['metrhabi'])
        dictpopl['metrplan'] /= np.nanmax(dictpopl['metrplan'])
        dictpopl['metrseti'] /= np.nanmax(dictpopl['metrseti'])
        
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
                
                listperi = dictpopl['pericomp'][None, indxexarstar]
                if not np.isfinite(listperi).all() or np.where(listperi == 0)[0].size > 0:
                    liststrgstarcomp.append(strgstar)
                    continue
                intgreso, ratiperi = nicomedia.retr_reso(listperi)
                
                numbcomp = indxexarstar.size
                
                gdat.listratiperi.append(ratiperi[0, :, :][np.triu_indices(numbcomp, k=1)])
                gdat.intgreso.append(intgreso)
                
                liststrgstarcomp.append(strgstar)
        
        gdat.listratiperi = np.concatenate(gdat.listratiperi)
        figr, axis = plt.subplots(figsize=gdat.figrsize)
        bins = np.linspace(1., 10., 400)
        axis.hist(gdat.listratiperi, bins=bins, rwidth=1)
        if gmod.numbcomp > 1:
            ## this system
            for j in gmod.indxcomp:
                for jj in gmod.indxcomp:
                    if gdat.dicterrr['pericomp'][0, j] > gdat.dicterrr['pericomp'][0, jj]:
                        ratiperi = gdat.dicterrr['pericomp'][0, j] / gdat.dicterrr['pericomp'][0, jj]
                        axis.axvline(ratiperi, color=gdat.listcolrcomp[jj])
                        axis.axvline(ratiperi, color=gdat.listcolrcomp[j], ls='--')
        
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
        path = pathvisufeatplan + 'histratiperi_%s_%s_%s.%s' % (gdat.strgtarg, strgpdfn, strgpopl, gdat.typefileplot)
        #gdat.listdictdvrp[j+1].append({'path': path, 'limt':[0.4, 0.05, 0.5, 0.1]})
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()
        
        # metastable helium absorption
        path = gdat.pathbasemile + '/data/wasp107b_transmission_spectrum.dat'
        if gdat.typeverb > 0:
            print('Reading from %s...' % path)
        arry = np.loadtxt(path, delimiter=',', skiprows=1)
        wlenwasp0107 = arry[:, 0]
        deptwasp0107 = arry[:, 1]
        deptstdvwasp0107 = arry[:, 2]
        
        stdvnirs = 0.24e-2
        for a in range(2):
            duratranplanwasp0107 = 2.74
            jmagsystwasp0107 = 9.4
            if a == 1:
                radicomp = gdat.dicterrr['radicomp'][0, :]
                masscomp = gdat.dicterrr['masscompused'][0, :]
                tmptplan = gdat.dicterrr['tmptplan'][0, :]
                duratranplan = gdat.dicterrr['duratrantotl'][0, :]
                radistar = gdat.radistar
                jmagsyst = gdat.jmagsyst
            else:
                print('WASP-107')
                radicomp = 0.924 * gdat.dictfact['rjre']
                masscomp = 0.119
                tmptplan = 736
                radistar = 0.66 # [R_S]
                jmagsyst = jmagsystwasp0107
                duratranplan = duratranplanwasp0107
            scalheig = nicomedia.retr_scalheig(tmptplan, masscomp, radicomp)
            deptscal = 1e3 * 2. * radicomp * scalheig / radistar**2 # [ppt]
            dept = 80. * deptscal
            factstdv = np.sqrt(10**((-jmagsystwasp0107 + jmagsyst) / 2.5) * duratranplanwasp0107 / duratranplan)
            stdvnirsthis = factstdv * stdvnirs
            for b in np.arange(1, 6):
                stdvnirsscal = stdvnirsthis / np.sqrt(float(b))
                sigm = dept / stdvnirsscal
        
            print('radicomp')
            print(radicomp)
            print('masscomp')
            print(masscomp)
            print('duratranplan')
            print(duratranplan)
            print('tmptplan')
            print(tmptplan)
            print('jmagsyst')
            print(jmagsyst)
            print('jmagsystwasp0107')
            print(jmagsystwasp0107)
            print('scalheig [R_E]')
            print(scalheig)
            print('scalheig [km]')
            print(scalheig * 71398)
            print('deptscal')
            print(deptscal)
            print('depttrancomp')
            print(dept)
            print('duratranplanwasp0107')
            print(duratranplanwasp0107)
            print('duratranplan')
            print(duratranplan)
            print('factstdv')
            print(factstdv)
            print('stdvnirsthis')
            print(stdvnirsthis)
            for b in np.arange(1, 6):
                print('With %d transits:' % b)
                print('stdvnirsscal')
                print(stdvnirsscal)
                print('sigm')
                print(sigm)
        print('James WASP107b scale height: 855 km')
        print('James WASP107b scale height: %g [R_E]' % (855. / 71398))
        print('James WASP107b depth per scale height: 5e-4')
        print('ampltide ratio fact: deptthis / 500e-6')
        fact = deptscal / 500e-6
        print('fact')
        print(fact)
        # 2 A * Rp * H / Rs**2
            
        figr, axis = plt.subplots(figsize=gdat.figrsize)
        #axis.errorbar(wlenwasp0107, deptwasp0107, yerr=deptstdvwasp0107, ls='', ms=1, lw=1, marker='o', color='k', alpha=1)
        axis.errorbar(wlenwasp0107-10833, deptwasp0107*fact[0], yerr=deptstdvwasp0107*factstdv[0], ls='', ms=1, lw=1, marker='o', color='k', alpha=1)
        axis.set_xlabel(r'Wavelength - 10,833 [$\AA$]')
        axis.set_ylabel('Depth [\%]')
        plt.subplots_adjust(bottom=0.2, left=0.2)
        path = pathvisudataplan + 'dept_%s_%s.%s' % (gdat.strgtarg, strgpdfn, gdat.typefileplot)
        #gdat.listdictdvrp[j+1].append({'path': path, 'limt':[0.4, 0.05, 0.5, 0.1]})
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()

        # optical magnitude vs number of planets
        for b in range(4):
            if b == 0:
                strgvarbmagt = 'vmag'
                lablxaxi = 'V Magnitude'
                varbtarg = gdat.vmagsyst
                varb = dictpopl['vmagsyst']
            if b == 1:
                strgvarbmagt = 'jmag'
                lablxaxi = 'J Magnitude'
                varbtarg = gdat.jmagsyst
                varb = dictpopl['jmagsyst']
            if b == 2:
                strgvarbmagt = 'rvelsemascal_vmag'
                lablxaxi = '$K^{\prime}_{V}$'
                varbtarg = np.sqrt(10**(-gdat.vmagsyst / 2.5)) / gdat.massstar**(2. / 3.)
                varb = np.sqrt(10**(-dictpopl['vmagsyst'] / 2.5)) / dictpopl['massstar']**(2. / 3.)
            if b == 3:
                strgvarbmagt = 'rvelsemascal_jmag'
                lablxaxi = '$K^{\prime}_{J}$'
                varbtarg = np.sqrt(10**(-gdat.vmagsyst / 2.5)) / gdat.massstar**(2. / 3.)
                varb = np.sqrt(10**(-dictpopl['jmagsyst'] / 2.5)) / dictpopl['massstar']**(2. / 3.)
            for a in range(2):
                figr, axis = plt.subplots(figsize=gdat.figrsize)
                if a == 0:
                    indx = np.where((dictpopl['numbplanstar'] > 3))[0]
                if a == 1:
                    indx = np.where((dictpopl['numbplantranstar'] > 3))[0]
                
                if (b == 2 or b == 3):
                    normfact = max(varbtarg, np.nanmax(varb[indx]))
                else:
                    normfact = 1.
                varbtargnorm = varbtarg / normfact
                varbnorm = varb[indx] / normfact
                axis.scatter(varbnorm, dictpopl['numbplanstar'][indx], s=1, color=colrdraw)
                
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
                axis.scatter(varbtargnorm, gmod.numbcomp, s=5, color=colrdraw, marker='x')
                axis.text(varbtargnorm, gmod.numbcomp + 0.5, gdat.labltarg, size=8, color=colrdraw, \
                                                                                            va='center', ha='center', rotation=45)
                axis.set_ylabel(r'Number of transiting planets')
                axis.set_xlabel(lablxaxi)
                plt.subplots_adjust(bottom=0.2)
                plt.subplots_adjust(left=0.2)
                path = pathvisufeatsyst + '%snumb_%s_%s_%s_%d.%s' % (strgvarbmagt, gdat.strgtarg, strgpdfn, strgpopl, a, gdat.typefileplot)
                #gdat.listdictdvrp[j+1].append({'path': path, 'limt':[0.4, 0.05, 0.5, 0.1]})
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()

                figr, axis = plt.subplots(figsize=gdat.figrsize)
                axis.hist(varbnorm, 50)
                axis.axvline(varbtargnorm, color=colrdraw, ls='--')
                axis.text(0.3, 0.9, gdat.labltarg, size=8, color=colrdraw, transform=axis.transAxes, va='center', ha='center')
                axis.set_ylabel(r'Number of systems')
                axis.set_xlabel(lablxaxi)
                plt.subplots_adjust(bottom=0.2)
                plt.subplots_adjust(left=0.2)
                path = pathvisufeatsyst + 'hist_%s_%s_%s_%s_%d.%s' % (strgvarbmagt, gdat.strgtarg, strgpdfn, strgpopl, a, gdat.typefileplot)
                #gdat.listdictdvrp[j+1].append({'path': path, 'limt':[0.4, 0.05, 0.5, 0.1]})
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()
            
        # planet feature distribution plots
        print('Will make the relevant distribution plots...')
        numbcomptext = min(10, numbcomppopl)
        liststrgtext = ['notx', 'text']
        
        # first is x-axis, second is y-axis
        liststrgfeatpairplot = [ \
                            #['smax', 'massstar'], \
                            #['rascstar', 'declstar'], \
                            #['lgalstar', 'bgalstar'], \
                            #['loecstar', 'laecstar'], \
                            #['distsyst', 'vmagsyst'], \
                            #['inso', 'radicomp'], \
                            ['radicomp', 'tmptplan'], \
                            ['radicomp', 'tsmm'], \
                            #['radicomp', 'esmm'], \
                            ['tmptplan', 'tsmm'], \
                            #['tagestar', 'vesc'], \
                            ['tmptplan', 'vesc0060'], \
                            #['radicomp', 'tsmm'], \
                            #['tmptplan', 'vesc'], \
                            #['pericomp', 'inso'], \
                            #['radistar', 'radicomp'], \
                            #['tmptplan', 'radistar'], \
                            #['projoblq', 'vsiistar'], \
                           ]
        
        numbpairfeatplot = len(liststrgfeatpairplot)

        indxpairfeatplot = np.arange(numbpairfeatplot)
        liststrgsort = ['none', \
                        #'esmm', \
                        'tsmm', \
                        #'metrhzon', 'metrhabi', 'metrplan', 'metrseti', \
                       ]
        numbstrgsort = len(liststrgsort)
        indxstrgsort = np.arange(numbstrgsort)

        indxcompfilt = dict()
        indxcompfilt['Total'] = indxtargpopl
        
        #indxcompfilt['Transit'] = np.where(dictpopl['booltran'])[0]
        strgcuttmain = 'Total'
        
        #indxcompfilt['box1'] = np.where((dictpopl['radicomp'] < 3.5) & (dictpopl['radicomp'] > 3.) & (dictpopl['tmptplan'] > 300) & \
        #                                                                                            (dictpopl['tmptplan'] < 500) & dictpopl['booltran'])[0]
        #indxcompfilt['box2'] = np.where((dictpopl['radicomp'] < 2.5) & (dictpopl['radicomp'] > 2.) & (dictpopl['tmptplan'] > 800) & \
        #                                                                                            (dictpopl['tmptplan'] < 1000) & dictpopl['booltran'])[0]
        #indxcompfilt['box2'] = np.where((dictpopl['radicomp'] < 3.) & (dictpopl['radicomp'] > 2.5) & (dictpopl['tmptplan'] > 1000) & \
        #                                                                                            (dictpopl['tmptplan'] < 1400) & dictpopl['booltran'])[0]
        #indxcompfilt['box3'] = np.where((dictpopl['radicomp'] < 3.) & (dictpopl['radicomp'] > 2.5) & (dictpopl['tmptplan'] > 1000) & \
        #                                                                                            (dictpopl['tmptplan'] < 1400) & dictpopl['booltran'])[0]
        #indxcompfilt['r4tr'] = np.where((dictpopl['radicomp'] < 4) & dictpopl['booltran'])[0]
        #indxcompfilt['r4trtess'] = np.where((dictpopl['radicomp'] < 4) & dictpopl['booltran'] & \
        #                                                                (dictpopl['facidisc'] == 'Transiting Exoplanet Survey Satellite (TESS)'))[0]

        #indxcompfilt['r154'] = np.where((dictpopl['radicomp'] > 1.5) & (dictpopl['radicomp'] < 4))[0]
        #indxcompfilt['r204'] = np.where((dictpopl['radicomp'] > 2) & (dictpopl['radicomp'] < 4))[0]
        #indxcompfilt['rb24'] = np.where((dictpopl['radicomp'] < 4) & (dictpopl['radicomp'] > 2.))[0]
        #indxcompfilt['gmtr'] = np.where(np.isfinite(stnomass) & (stnomass > 5) & (dictpopl['booltran']))[0]
        #indxcompfilt['Transit'] = np.where(dictpopl['booltran'])[0]
        #indxcompfilt['mult'] = np.where(dictpopl['numbplantranstar'] > 3)[0]
        #indxcompfilt['live'] = np.where(dictpopl['boollive'])[0]
        #indxcompfilt['terr'] = np.where(dictpopl['boolterr'] & dictpopl['boollive'])[0]
        #indxcompfilt['hzoncons'] = np.where(dictpopl['boolhabicons'] & dictpopl['boollive'])[0]
        #indxcompfilt['hzonopti'] = np.where(dictpopl['boolhabiopti'] & dictpopl['boollive'])[0]
        #indxcompfilt['unlo'] = np.where(dictpopl['boolunlo'] & dictpopl['boollive'])[0]
        #indxcompfilt['habi'] = np.where(dictpopl['boolterr'] & dictpopl['boolhabiopti'] & dictpopl['boolunlo'] & dictpopl['boollive'])[0]
        #indxcompfilt['eatp'] = np.where(dictpopl['booleatp'] & dictpopl['boollive'])[0]
        #indxcompfilt['seti'] = np.where(dictpopl['boolterr'] & dictpopl['boolhabicons'] & dictpopl['boolunlo'] & \
                                                                                            #dictpopl['booleatp'] & dictpopl['boollive'])[0]
        dicttemp = dict()
        dicttempmerg = dict()
        
        liststrgcutt = indxcompfilt.keys()
        
        liststrgvarb = [ \
                        'pericomp', 'inso', 'vesc0060', 'masscomp', \
                        'metrhzon', 'metrterr', 'metrplan', 'metrunlo', 'metrseti', \
                        'smax', \
                        'tmptstar', \
                        'rascstar', 'declstar', \
                        'loecstar', 'laecstar', \
                        'radistar', \
                        'massstar', \
                        'metastar', \
                        'radicomp', 'tmptplan', \
                        'metrhabi', 'metrplan', \
                        'lgalstar', 'bgalstar', 'distsyst', 'vmagsyst', \
                        'tsmm', 'esmm', \
                        'vsiistar', 'projoblq', \
                        'jmagsyst', \
                        'tagestar', \
                       ]

        listlablvarb, listscalpara = tdpy.retr_listlablscalpara(liststrgvarb)
        listlablvarbtotl = tdpy.retr_labltotl(listlablvarb)
        #listlablvarb = [ \
        #                ['P', 'days'], ['F', '$F_E$'], ['$v_{esc}^\prime$', 'kms$^{-1}$'], ['$M_p$', '$M_E$'], \
        #                [r'$\rho$_{HZ}', ''], [r'$\rho$_{T}', ''], [r'$\rho$_{MP}', ''], [r'$\rho$_{TL}', ''], [r'$\rho$_{SETI}', ''], \
        #                ['$a$', 'AU'], \
        #                ['$T_{eff}$', 'K'], \
        #                ['RA', 'deg'], ['Dec', 'deg'], \
        #                ['Ec. lon.', 'deg'], ['Ec. lat.', 'deg'], \
        #                ['$R_s$', '$R_S$'], \
        #                ['$M_s$', '$M_S$'], \
        #                ['[Fe/H]', 'dex'], \
        #                ['$R_p$', '$R_E$'], ['$T_p$', 'K'], \
        #                [r'$\rho_{SH}$', ''], [r'$\rho_{SP}$', ''], \
        #                ['$l$', 'deg'], ['$b$', 'deg'], ['$d$', 'pc'], ['$V$', ''], \
        #                ['TSM', ''], ['ESM', ''], \
        #                ['$v$sin$i$', 'kms$^{-1}$'], ['$\lambda$', 'deg'], \
        #                ['J', 'mag'], \
        #                ['$t_\star$', 'Gyr'], \
        #               ] 
        
        numbvarb = len(liststrgvarb)
        indxvarb = np.arange(numbvarb)
            
        # merge the target with the population
        for k, strgxaxi in enumerate(liststrgvarb + ['nameplan']):
            if not strgxaxi in dictpopl or not strgxaxi in gdat.dicterrr:
                continue
            dicttempmerg[strgxaxi] = np.concatenate([dictpopl[strgxaxi][indxcompfilt[strgcuttmain]], gdat.dicterrr[strgxaxi][0, :]])
        
        if not 'nameplan' in dictpopl:
            raise Exception('')

        #if not 'nameplan' in gdat.dicterrr:
        #    raise Exception('')

        if not 'nameplan' in dicttempmerg:
            raise Exception('')

        for k, strgxaxi in enumerate(liststrgvarb):
            
            if strgxaxi == 'tmptplan':
                print('strgxaxi in dictpopl')
                print(strgxaxi in dictpopl)
                print('strgxaxi in gdat.dicterrr')
                print(strgxaxi in gdat.dicterrr)
                raise Exception('')

            if not strgxaxi in dictpopl:
                continue

            if not strgxaxi in gdat.dicterrr:
                continue

            if not 'tmptplan' in dicttempmerg:
                print('dicttempmerg')
                print(dicttempmerg.keys())
                raise Exception('')

            for m, strgyaxi in enumerate(liststrgvarb):
                
                booltemp = False
                for l in indxpairfeatplot:
                    if strgxaxi == liststrgfeatpairplot[l][0] and strgyaxi == liststrgfeatpairplot[l][1]:
                        booltemp = True
                if not booltemp:
                    continue
                 
                # to be deleted
                #for strgfeat, valu in dictpopl.items():
                    #dicttempmerg[strgfeat] = np.concatenate([dictpopl[strgfeat][indxcompfilt[strgcuttmain]], gdat.dicterrr[strgfeat][0, :]])
                
                for strgcutt in liststrgcutt:
                    
                    # merge population with the target
                    #for strgfeat, valu in dictpopl.items():
                    #    dicttemp[strgfeat] = np.concatenate([dictpopl[strgfeat][indxcompfilt[strgcutt]], gdat.dicterrr[strgfeat][0, :]])
                    
                    liststrgfeatcsvv = [ \
                                        #'inso', 'metrhzon', 'radicomp', 'metrterr', 'massstar', 'smax', 'metrunlo', 'distsyst', 'metrplan', 'metrseti', \
                                        'rascstar', 'declstar', 'radicomp', 'masscomp', 'tmptplan', 'jmagsyst', 'radistar', 'tsmm', \
                                       ]
                    for y in indxstrgsort:
                        
                        if liststrgsort[y] != 'none':
                        
                            indxgood = np.where(np.isfinite(dicttempmerg[liststrgsort[y]]))[0]
                            indxsort = np.argsort(dicttempmerg[liststrgsort[y]][indxgood])[::-1]
                            indxcompsort = indxgood[indxsort]
                            
                            path = gdat.pathdatatarg + '%s_%s_%s.csv' % (strgpopl, strgcutt, liststrgsort[y])
                            objtfile = open(path, 'w')
                            
                            strghead = '%4s, %20s' % ('Rank', 'Name')
                            for strgfeatcsvv in liststrgfeatcsvv:
                                strghead += ', %12s' % listlablvarbtotl[liststrgvarb.index(strgfeatcsvv)]
                            strghead += '\n'
                            
                            objtfile.write(strghead)
                            cntr = 1
                            for l in indxcompsort:
                                
                                strgline = '%4d, %20s' % (cntr, dicttempmerg['nameplan'][l])
                                for strgfeatcsvv in liststrgfeatcsvv:
                                    strgline += ', %12.4g' % dicttempmerg[strgfeatcsvv][l]
                                strgline += '\n'
                                
                                objtfile.write(strgline)
                                cntr += 1 
                            print('Writing to %s...' % path)
                            objtfile.close()
                    
                        if gdat.boolplotpopl:
                            # repeat, one without text, one with text
                            for b, strgtext in enumerate(liststrgtext):
                                figr, axis = plt.subplots(figsize=gdat.figrsize)
                                
                                if liststrgsort[y] != 'none' and strgtext != 'text' or liststrgsort[y] == 'none' and strgtext == 'text':
                                    continue
                        
                                ## population
                                if strgcutt == strgcuttmain:
                                    axis.errorbar(dicttempmerg[strgxaxi], dicttempmerg[strgyaxi], ls='', ms=1, marker='o', color='k')
                                else:
                                    axis.errorbar(dicttempmerg[strgxaxi], dicttempmerg[strgyaxi], ls='', ms=1, marker='o', color='k')
                                    #axis.errorbar(dicttemp[strgxaxi], dicttemp[strgyaxi], ls='', ms=2, marker='o', color='r')
                                
                                ## this system
                                for j in gmod.indxcomp:
                                    if strgxaxi in gdat.dicterrr:
                                        xdat = gdat.dicterrr[strgxaxi][0, j, None]
                                        xerr = gdat.dicterrr[strgxaxi][1:3, j, None]
                                    if strgyaxi in gdat.dicterrr:
                                        ydat = gdat.dicterrr[strgyaxi][0, j, None]
                                        yerr = gdat.dicterrr[strgyaxi][1:3, j, None]
                                    
                                    # temp apply cut on this system
                                    
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
                                                axis.axhline(ydat, color=gdat.listcolrcomp[j], lw=1, ls='--', zorder=2)
                                        if not strgyaxi in gdat.dicterrr and strgxaxi in gdat.dicterrr:
                                            if strgxaxi in gdat.listfeatstar:
                                                axis.axvline(xdat, color='k', lw=1, ls='--', zorder=2)
                                                axis.text(0.85, 0.9 - j * 0.08, gdat.labltarg, color='k', \
                                                                                              va='center', ha='center', transform=axis.transAxes)
                                                break
                                            else:
                                                axis.axvline(xdat, color=gdat.listcolrcomp[j], lw=1, ls='--')
                                        if strgxaxi in gdat.dicterrr and strgyaxi in gdat.dicterrr:
                                            axis.errorbar(xdat, ydat, color=gdat.listcolrcomp[j], lw=1, xerr=xerr, yerr=yerr, ls='', marker='o', \
                                                                                                                                zorder=2, ms=6)
                                        
                                        if strgxaxi in gdat.dicterrr or strgyaxi in gdat.dicterrr:
                                            axis.text(0.85, 0.9 - j * 0.08, r'\textbf{%s}' % gdat.liststrgcomp[j], color=gdat.listcolrcomp[j], \
                                                                                            va='center', ha='center', transform=axis.transAxes)
                                
                                # include text
                                if liststrgsort[y] != 'none' and strgtext == 'text':
                                    for ll, l in enumerate(indxcompsort):
                                        if ll < numbcomptext:
                                            text = '%s' % dicttemp['nameplan'][l]
                                            xdat = dicttemp[strgxaxi][l]
                                            ydat = dicttemp[strgyaxi][l]
                                            if np.isfinite(xdat) and np.isfinite(ydat):
                                                objttext = axis.text(xdat, ydat, text, size=1, ha='center', va='center')
                                
                                if strgxaxi == 'tmptplan' and strgyaxi == 'vesc0060':
                                    xlim = [0, 0]
                                    xlim[0] = 0.5 * np.nanmin(dictpopl['tmptplan'])
                                    xlim[1] = 2. * np.nanmax(dictpopl['tmptplan'])
                                    arrytmptplan = np.linspace(xlim[0], xlim[1], 1000)
                                    cons = [1., 4., 16., 18., 28., 44.] # H, He, CH4, H20, CO, CO2
                                    for i in range(len(cons)):
                                        arryyaxi = (arrytmptplan / 40. / cons[i])**0.5
                                        axis.plot(arrytmptplan, arryyaxi, color='grey', alpha=0.5)
                                    axis.set_xlim(xlim)

                                if strgxaxi == 'radicomp' and strgyaxi == 'masscomp':
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
                                
                                #if strgxaxi == 'tmptplan':
                                #    axis.axvline(273., ls='--', alpha=0.3, color='k')
                                #    axis.axvline(373., ls='--', alpha=0.3, color='k')
                                #if strgyaxi == 'tmptplan':
                                #    axis.axhline(273., ls='--', alpha=0.3, color='k')
                                #    axis.axhline(373., ls='--', alpha=0.3, color='k')
        
                                axis.set_xlabel(listlablvarbtotl[k])
                                axis.set_ylabel(listlablvarbtotl[m])
                                if listscalpara[k] == 'logt':
                                    axis.set_xscale('log')
                                if listscalpara[m] == 'logt':
                                    axis.set_yscale('log')
                                
                                plt.subplots_adjust(left=0.2)
                                plt.subplots_adjust(bottom=0.2)
                                pathvisufeatplan = getattr(gdat, 'pathvisufeatplan' + strgpdfn)
                                path = pathvisufeatplan + 'feat_%s_%s_%s_%s_%s_%s_%s_%s.%s' % \
                                             (strgxaxi, strgyaxi, gdat.strgtarg, strgpopl, strgcutt, \
                                                                                   strgtext, liststrgsort[y], strgpdfn, gdat.typefileplot)
                                #gdat.listdictdvrp[j+1].append({'path': path, 'limt':[0.4, 0.05, 0.5, 0.1]})
                                print('Writing to %s...' % path)
                                plt.savefig(path)
                                plt.close()

   
def bdtr_wrap(gdat, b, p, y, epocmask, perimask, duramask, strgintp, strgoutp, strgtren, timescalbdtr):
    '''
    Wrap baseline-detrending function
    '''
    
    # output
    gdat.listarrytser[strgoutp][b][p][y] = np.copy(gdat.listarrytser[strgintp][b][p][y])
    
    # trend
    gdat.listarrytser[strgtren][b][p][y] = np.copy(gdat.listarrytser[strgintp][b][p][y])
    
    for e in gdat.indxener[p]:
        gdat.rflxbdtr, gdat.rflxbdtrregi, gdat.listindxtimeregi[b][p][y], gdat.indxtimeregioutt[b][p][y], gdat.listobjtspln[b][p][y], gdat.listtimebrek = \
                     bdtr_tser(gdat.listarrytser[strgintp][b][p][y][:, e, 0], gdat.listarrytser[strgintp][b][p][y][:, e, 1], \
                                       stdvlcur=gdat.listarrytser[strgintp][b][p][y][:, e, 2], \
                                       epocmask=epocmask, perimask=perimask, duramask=duramask, \
                                       timescalbdtr=timescalbdtr, \
                                       typeverb=gdat.typeverb, \
                                       timeedge=gdat.listtimebrek, \
                                       timebrekregi=gdat.timebrekregi, \
                                       ordrspln=gdat.ordrspln, \
                                       timescalbdtrmedi=gdat.timescalbdtrmedi, \
                                       boolbrekregi=gdat.boolbrekregi, \
                                       typebdtr=gdat.typebdtr, \
                                      )
    
        gdat.listarrytser[strgoutp][b][p][y][:, e, 1] = np.concatenate(gdat.rflxbdtrregi)
    
    numbsplnregi = len(gdat.rflxbdtrregi)
    gdat.indxsplnregi[b][p][y] = np.arange(numbsplnregi)


def plot_tser_mile_core(gdat, strgarry, b, p, y=None, boolcolrtran=True, boolflar=False):
    
    boolchun = y is not None
    
    if not boolchun and gdat.numbchun[b][p] == 1:
        return
        
    if boolchun:
        arrytser = gdat.listarrytser[strgarry][b][p][y]
    else:
        arrytser = gdat.arrytser[strgarry][b][p]
    
    if gdat.booldiag:
        if len(arrytser) == 0:
            print('')
            print('')
            print('')
            print('strgarry')
            print(strgarry)
            raise Exception('len(arrytser) == 0')
    
    # determine name of the file
    ## string indicating the prior on the transit ephemerides
    strgprioplan = ''
    if strgarry != 'Raw' and gdat.typepriocomp is not None:
        strgprioplan = '_%s' % gdat.typepriocomp
    strgcolr = ''
    if boolcolrtran:
        strgcolr = '_colr'
    strgchun = ''
    if boolchun and gdat.numbchun[b][p] > 1:
        strgchun = '_' + gdat.liststrgchun[b][p][y]
    path = gdat.pathvisutarg + '%s%s_%s%s_%s%s_%s%s.%s' % \
                    (gdat.liststrgdatatser[b], gdat.strgcnfg, strgarry, strgcolr, gdat.liststrginst[b][p], strgchun, gdat.strgtarg, strgprioplan, gdat.typefileplot)
    
    if not strgarry.startswith('bdtroutpit') and not strgarry.startswith('clipoutpit'):
        if strgarry == 'Raw':
            limt = [0., 0.9, 0.5, 0.1]
        elif strgarry == 'Detrended':
            limt = [0., 0.7, 0.5, 0.1]
        else:
            limt = [0., 0.5, 0.5, 0.1]
        gdat.listdictdvrp[0].append({'path': path, 'limt':limt})
        
    if not os.path.exists(path):
            
        figr, axis = plt.subplots(figsize=gdat.figrsizeydobskin)
        
        if arrytser.shape[1] > 1:
            extent = [gdat.listener[p][0], gdat.listener[p][1], arrytser[0, 0, 0] - gdat.timeoffs, arrytser[-1, 0, 0] - gdat.timeoffs]
            imag = axis.imshow(arrytser[:, :, 1].T, extent=extent)
        else:
            axis.plot(arrytser[:, 0, 0] - gdat.timeoffs, arrytser[:, 0, 1], color='grey', marker='.', ls='', ms=1, rasterized=True)
        
        if boolcolrtran:
            
            # color and name transits
            ylim = axis.get_ylim()
            listtimetext = []
            for j in gdat.fitt.prio.indxcomp:
                if boolchun:
                    indxtime = gdat.listindxtimetranchun[j][b][p][y] 
                else:
                    if y > 0:
                        continue
                    indxtime = gdat.listindxtimetran[j][b][p][0]
                
                colr = gdat.listcolrcomp[j]
                # plot data
                axis.plot(arrytser[indxtime, 0, 0] - gdat.timeoffs, arrytser[indxtime, 0, 1], color=colr, marker='.', ls='', ms=1, rasterized=True)
                # draw planet names
                for n in np.linspace(-gdat.numbcyclcolrplot, gdat.numbcyclcolrplot, 2 * gdat.numbcyclcolrplot + 1):
                    time = gdat.fitt.prio.meanpara.epocmtracomp[j] + n * gdat.fitt.prio.meanpara.pericomp[j] - gdat.timeoffs
                    if np.where(abs(arrytser[:, 0] - gdat.timeoffs - time) < 0.1)[0].size > 0:
                        
                        # add a vertical offset if overlapping
                        if np.where(abs(np.array(listtimetext) - time) < 0.5)[0].size > 0:
                            ypostemp = ylim[0] + (ylim[1] - ylim[0]) * 0.95
                        else:
                            ypostemp = ylim[0] + (ylim[1] - ylim[0]) * 0.9

                        # draw the planet letter
                        axis.text(time, ypostemp, r'\textbf{%s}' % gdat.liststrgcomp[j], color=gdat.listcolrcomp[j], va='center', ha='center')
                        listtimetext.append(time)

        if boolchun:
            if boolflar:
                ydat = axis.get_ylim()[1]
                for kk in range(len(gdat.listindxtimeflar[p][y])):
                    ms = 0.5 * gdat.listmdetflar[p][y][kk]
                    axis.plot(arrytser[gdat.listindxtimeflar[p][y][kk], 0, 0] - gdat.timeoffs, ydat, marker='v', color='b', ms=ms, rasterized=True)
                axis.plot(gdat.listarrytser['Detrended'][b][p][y][:, 0, 0] - gdat.timeoffs, \
                          gdat.listarrytser['bdtrmedi'][b][p][y][:, 0, 1], \
                          color='g', marker='.', ls='', ms=1, rasterized=True)
                
                axis.plot(gdat.listarrytser['Detrended'][b][p][y][:, 0, 0] - gdat.timeoffs, \
                          gdat.thrsrflxflar[p][y], \
                          color='orange', marker='.', ls='', ms=1, rasterized=True)
                
                axis.fill_between(gdat.listarrytser['Detrended'][b][p][y][:, 0, 0] - gdat.timeoffs, \
                                  gdat.listarrytser['bdtrlowr'][b][p][y][:, 0, 1], \
                                  gdat.listarrytser['bdtruppr'][b][p][y][:, 0, 1], \
                                  color='c', alpha=0.2, rasterized=True)
            
        axis.set_xlabel('Time [BJD - %d]' % gdat.timeoffs)
        if arrytser.shape[1] > 1:
            axis.set_ylabel(gdat.lablener)
            cbar = plt.colorbar(imag)
        else:
            axis.set_ylabel(gdat.listlabltser[b])
        titl = retr_tsertitl(gdat, b, p, y=y)
        
        if gdat.typeplotback == 'black':
            axis.set_facecolor('black')

        axis.set_title(titl)
        plt.subplots_adjust(bottom=0.2)
        
        if gdat.typeverb > 0:
            print('Writing to %s...' % path)
        plt.savefig(path, dpi=200)
        plt.close()
    

    if gdat.numbener[p] > 1:
        # plot each energy
        
        path = gdat.pathvisutarg + '%s%s_%s%s_%s%s_%s%s_ener.%s' % \
                    (gdat.liststrgdatatser[b], gdat.strgcnfg, strgarry, strgcolr, gdat.liststrginst[b][p], strgchun, gdat.strgtarg, strgprioplan, gdat.typefileplot)
    
        if not os.path.exists(path):
                
            figr, axis = plt.subplots(figsize=gdat.figrsizeydobskin)
            
            sprdrflx = np.amax(np.std(arrytser[:, :, 1], 0)) * 5.
            listdiffrflxener = np.linspace(-1., 1., gdat.numbener[p]) * 0.5 * gdat.numbener[p] * sprdrflx

            for e in gdat.indxener[p]:
                color = plt.cm.rainbow(e / (gdat.numbener[p] - 1))
                axis.plot(arrytser[:, e, 0] - gdat.timeoffs, arrytser[:, e, 1] + listdiffrflxener[e], color=color, marker='.', ls='', ms=1, rasterized=True)
            
            if boolcolrtran:
                # color and name transits
                ylim = axis.get_ylim()
                listtimetext = []
                for j in gdat.fitt.prio.indxcomp:
                    if boolchun:
                        indxtime = gdat.listindxtimetranchun[j][b][p][y] 
                    else:
                        if y > 0:
                            continue
                        indxtime = gdat.listindxtimetran[j][b][p][0]
                    
                    colr = gdat.listcolrcomp[j]
                    # plot data
                    axis.plot(arrytser[indxtime, 0, 0] - gdat.timeoffs, arrytser[indxtime, 0, 1], color=colr, marker='.', ls='', ms=1, rasterized=True)
                    # draw planet names
                    for n in np.linspace(-gdat.numbcyclcolrplot, gdat.numbcyclcolrplot, 2 * gdat.numbcyclcolrplot + 1):
                        time = gdat.fitt.prio.meanpara.epocmtracomp[j] + n * gdat.fitt.prio.meanpara.pericomp[j] - gdat.timeoffs
                        if np.where(abs(arrytser[:, 0] - gdat.timeoffs - time) < 0.1)[0].size > 0:
                            
                            # add a vertical offset if overlapping
                            if np.where(abs(np.array(listtimetext) - time) < 0.5)[0].size > 0:
                                ypostemp = ylim[0] + (ylim[1] - ylim[0]) * 0.95
                            else:
                                ypostemp = ylim[0] + (ylim[1] - ylim[0]) * 0.9

                            # draw the planet letter
                            axis.text(time, ypostemp, r'\textbf{%s}' % gdat.liststrgcomp[j], color=gdat.listcolrcomp[j], va='center', ha='center')
                            listtimetext.append(time)
            
            if boolchun:
                if boolflar:
                    ydat = axis.get_ylim()[1]
                    for kk in range(len(gdat.listindxtimeflar[p][y])):
                        ms = 0.5 * gdat.listmdetflar[p][y][kk]
                        axis.plot(arrytser[gdat.listindxtimeflar[p][y][kk], 0] - gdat.timeoffs, ydat, marker='v', color='b', ms=ms, rasterized=True)
                    axis.plot(gdat.listarrytser['Detrended'][b][p][y][:, 0, 0] - gdat.timeoffs, \
                              gdat.listarrytser['bdtrmedi'][b][p][y][:, 0, 1], color='g', marker='.', ls='', ms=1, rasterized=True)
                
                    axis.plot(gdat.listarrytser['Detrended'][b][p][y][:, 0, 0] - gdat.timeoffs, \
                              gdat.thrsrflxflar[p][y], \
                              color='orange', marker='.', ls='', ms=1, rasterized=True)
                
                    axis.fill_between(gdat.listarrytser['Detrended'][b][p][y][:, 0, 0] - gdat.timeoffs, 
                                      gdat.listarrytser['bdtrlowr'][b][p][y][:, 0, 1], \
                                      gdat.listarrytser['bdtruppr'][b][p][y][:, 0, 1], \
                                      color='c', alpha=0.2, rasterized=True)
                
            axis.set_xlabel('Time [BJD - %d]' % gdat.timeoffs)
            axis.set_ylabel(gdat.listlabltser[b])
            axis.set_title(gdat.labltarg)
            plt.subplots_adjust(bottom=0.2)
            
            if gdat.typeverb > 0:
                print('Writing to %s...' % path)
            plt.savefig(path, dpi=200)
            plt.close()


def retr_tsertitl(gdat, b, p, y=None):
    
    titl = '%s, %s' % (gdat.labltarg, gdat.listlablinst[b][p])
        
    if y is not None and len(gdat.listlablchun[b][p][y]) > 0 and gdat.listlablchun[b][p][y] != '':
       titl += ', %s' % gdat.listlablchun[b][p][y]
    
    if gdat.lablcnfg is not None and gdat.lablcnfg != '':
       titl += ', %s' % gdat.lablcnfg 
    
    if gdat.liststrginst[b][p] in gdat.dictmagtsyst:
        titl += ', %.3g mag' % gdat.dictmagtsyst[gdat.liststrginst[b][p]]
    
    return titl


def plot_tser_mile(gdat, b, p, y, strgarry, boolcolrtran=False, booltoge=True, boolflar=False):
    
    # plot each chunk
    plot_tser_mile_core(gdat, strgarry, b, p, y, boolcolrtran=False, boolflar=boolflar)
    
    # plot all chunks together if there is more than one chunk
    if y == 0 and gdat.numbchun[b][p] > 1 and booltoge:
        plot_tser_mile_core(gdat, strgarry, b, p, boolcolrtran=False, boolflar=boolflar)
    
    # highlight times in-transit
    if strgarry != 'Raw' and gdat.fitt.prio.numbcomp is not None:
        
        ## plot each chunk
        plot_tser_mile_core(gdat, strgarry, b, p, y, boolcolrtran=boolcolrtran, boolflar=boolflar)
        
        ## plot all chunks together if there is more than one chunk
        if y == 0 and gdat.numbchun[b][p] > 1 and not strgarry.startswith('ts0'):
            plot_tser_mile_core(gdat, strgarry, b, p, boolcolrtran=boolcolrtran, boolflar=boolflar)

        # plot in-transit data
        if b == 0 and boolcolrtran:
            path = gdat.pathvisutarg + 'rflx%s_intr%s_%s_%s_%s.%s' % \
                                            (strgarry, gdat.strgcnfg, gdat.liststrginst[b][p], gdat.strgtarg, gdat.typepriocomp, gdat.typefileplot)
            if not os.path.exists(path):
            
                # plot only the in-transit data
                figr, axis = plt.subplots(gdat.fitt.prio.numbcomp, 1, figsize=gdat.figrsizeydobskin, sharex=True)
                if gdat.fitt.prio.numbcomp == 1:
                    axis = [axis]
                for jj, j in enumerate(gdat.fitt.prio.indxcomp):
                    axis[jj].plot(gdat.listarrytser[strgarry][b][p][y][gdat.listindxtimetran[j][b][p][0], 0] - gdat.timeoffs, \
                                  gdat.listarrytser[strgarry][b][p][y][gdat.listindxtimetran[j][b][p][0], 1], \
                                                                                           color=gdat.listcolrcomp[j], marker='o', ls='', ms=0.2)
                
                axis[-1].set_ylabel(gdat.labltserphot)
                #axis[-1].yaxis.set_label_coords(0, gdat.fitt.prio.numbcomp * 0.5)
                axis[-1].set_xlabel('Time [BJD - %d]' % gdat.timeoffs)
                
                #plt.subplots_adjust(bottom=0.2)
                #gdat.listdictdvrp[j+1].append({'path': path, 'limt':[0.4, 0.05, 0.8, 0.8]})
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()
        

def plot_tser_bdtr(gdat, b, p, y, z, r, strgarryinpt, strgarryoutp):
    '''
    Plot baseline detrending.
    '''
    
    ## string indicating the prior on the transit ephemerides
    strgprioplan = ''
    
    #if strgarry != 'Raw' and gdat.typepriocomp is not None:
    #    strgprioplan = '_%s' % gdat.typepriocomp
    
    path = gdat.pathvisutarg + 'rflx_DetrendProcess%s_%s_%s_%s%s%s%s_ts%02dit%02d.%s' % (gdat.strgcnfg, gdat.liststrginst[b][p], \
                                 gdat.liststrgchun[b][p][y], gdat.strgtarg, strgprioplan, gdat.strgextncade[b][p], gdat.liststrgener[p][gdat.indxenerclip], z, r, gdat.typefileplot)
    gdat.listdictdvrp[0].append({'path': path, 'limt':[0., 0.05, 1.0, 0.2]})
    
    if not os.path.exists(path):
            
        figr, axis = plt.subplots(2, 1, figsize=gdat.figrsizeydob)
        for i in gdat.indxsplnregi[b][p][y]:
            ## non-baseline-detrended light curve
            axis[0].plot(gdat.listarrytser[strgarryinpt][b][p][y][:, gdat.indxenerclip, 0] - gdat.timeoffs, \
                         gdat.listarrytser[strgarryinpt][b][p][y][:, gdat.indxenerclip, 1], rasterized=True, \
                                                                                            marker='o', ls='', ms=1, color='grey')
            ## spline
            if gdat.listobjtspln[b][p][y] is not None and gdat.listobjtspln[b][p][y][i] is not None:
                minmtimeregi = gdat.listarrytser[strgarryinpt][b][p][y][0, gdat.indxenerclip, 0]
                maxmtimeregi = gdat.listarrytser[strgarryinpt][b][p][y][-1, gdat.indxenerclip, 0]
                timesplnregifine = np.linspace(minmtimeregi, maxmtimeregi, 1000)
                if gdat.typebdtr == 'Spline':
                    lcurtren = gdat.listobjtspln[b][p][y][i](timesplnregifine)
                if gdat.typebdtr == 'GaussianProcess':
                    lcurtren = gdat.listobjtspln[b][p][y][i].predict(gdat.listarrytser[strgarryinpt][b][p][y][gdat.indxtimeregioutt[b][p][y][i], gdat.indxenerclip, 1], \
                                                                                                                        t=timesplnregifine, return_cov=False, return_var=False)
                axis[0].plot(timesplnregifine - gdat.timeoffs, lcurtren, 'b-', lw=3, rasterized=True)
            ## baseline-detrended light curve
            axis[1].plot(gdat.listarrytser[strgarryoutp][b][p][y][:, gdat.indxenerclip, 0] - gdat.timeoffs, \
                         gdat.listarrytser[strgarryoutp][b][p][y][:, gdat.indxenerclip, 1], rasterized=True, \
                                                                                              marker='o', ms=1, ls='', color='grey')
        for a in range(2):
            axis[a].set_ylabel(gdat.labltserphot)
        axis[0].set_xticklabels([])
        axis[1].set_xlabel('Time [BJD - %d]' % gdat.timeoffs)
        
        titl = retr_tsertitl(gdat, b, p, y=y)
        facttime, lablunittime = tdpy.retr_timeunitdays(gdat.listtimescalbdtr[z])
        titl += ', DTS = %.3g %s' % (facttime * gdat.listtimescalbdtr[z], lablunittime)
        titl += ', Iteration %d' % r
        axis[0].set_title(titl)
        
        plt.subplots_adjust(hspace=0.)
        print('Writing to %s...' % path)
        plt.savefig(path, dpi=200)
        plt.close()
                            

def retr_namebdtrclip(e, r):

    strgarrybdtrinpt = 'ts%02dit%02dbdtrinpt' % (e, r)
    strgarryclipinpt = 'ts%02dit%02dclipinpt' % (e, r)
    strgarryclipoutp = 'ts%02dit%02dclipoutp' % (e, r)
    strgarrybdtrblin = 'ts%02dit%02dbdtrblin' % (e, r)
    strgarrybdtroutp = 'ts%02dit%02dbdtroutp' % (e, r)

    return strgarrybdtrinpt, strgarryclipoutp, strgarrybdtroutp, strgarryclipinpt, strgarrybdtrblin


def rebn_tser(arry, numbbins=None, delt=None, blimxdat=None):
    
    if not (numbbins is None and delt is None and blimxdat is not None or \
            numbbins is not None and delt is None and blimxdat is None or \
            numbbins is None and delt is not None and blimxdat is None):
        raise Exception('')
    
    if arry.shape[0] == 0:
        print('')
        print('')
        print('')
        print('arry')
        summgene(arry)
        raise Exception('Trying to bin an empty time-series...')
    
    if arry.ndim == 3:
        numbener = arry.shape[1]
    
        xdat = arry[:, 0, 0]
    elif arry.ndim == 2:
        xdat = arry[:, 0]
    
    if delt is not None:
        blimxdat = np.arange(np.amin(xdat), np.amax(xdat) + delt, delt)
    
    if delt is not None or blimxdat is not None:
        numbbins = blimxdat.size - 1

    if arry.ndim == 3:
        shaparryrebn = (numbbins, numbener, 3)
    else:
        shaparryrebn = (numbbins, 3)
    
    if numbbins is not None:
        arryrebn = np.full(shaparryrebn, fill_value=np.nan)
        blimxdat = np.linspace(np.amin(xdat), np.amax(xdat), numbbins + 1)
    
    if delt is not None or blimxdat is not None:
        numbbins = blimxdat.size - 1
        arryrebn = np.full(shaparryrebn, fill_value=np.nan)
    
    # bin centers
    bctrxdat = (blimxdat[:-1] + blimxdat[1:]) / 2.
    if arry.ndim == 3:
        arryrebn[:, 0, 0] = bctrxdat
    else:
        arryrebn[:, 0] = bctrxdat

    indxbins = np.arange(numbbins)
    for k in indxbins:
        indxxdat = np.where((xdat < blimxdat[k+1]) & (xdat > blimxdat[k]))[0]
        if indxxdat.size > 0:
            #if arry.ndim == 3:
            arryrebn[k, ..., 1] = np.mean(arry[indxxdat, ..., 1], axis=0)
            stdvfrst  = np.sqrt(np.nansum(arry[indxxdat, ..., 2]**2, axis=0)) / indxxdat.size
            stdvseco = np.std(arry[indxxdat, ..., 1], axis=0)
            arryrebn[k, ..., 2] = np.sqrt(stdvfrst**2 + stdvseco**2)
            #else:
            #    arryrebn[k, 1] = np.mean(arry[indxxdat, 1], axis=0)
            #    stdvfrst  = np.sqrt(np.nansum(arry[indxxdat, 2]**2, axis=0)) / indxxdat.size
            #    stdvseco = np.std(arry[indxxdat, 1], axis=0)
            #    arryrebn[k, 2] = np.sqrt(stdvfrst**2 + stdvseco**2)
    
    return arryrebn

    
def setp_para(gdat, strgmodl, nameparabase, minmpara, maxmpara, lablpara, strgener=None, strgcomp=None, strglmdk=None, boolvari=True):
    
    gmod = getattr(gdat, strgmodl)
    
    nameparabasefinl = nameparabase
    
    if strgcomp is not None:
        nameparabasefinl += strgcomp

    if strglmdk is not None:
        nameparabasefinl += strglmdk

    if strgener is not None:
        nameparabasefinl += strgener

    if gdat.typeverb > 0 or gdat.booldiag:
        if hasattr(gmod, nameparabasefinl):
            para = getattr(gmod, nameparabasefinl)
            if gdat.booldiag:
                if not isinstance(para, float) and len(para) == 0:
                    print('')
                    print('')
                    print('')
                    print('strgmodl')
                    print(strgmodl)
                    print('getattr(gmod, nameparabasefinl)')
                    print(getattr(gmod, nameparabasefinl))
                    print('nameparabasefinl')
                    print(nameparabasefinl)
                    raise Exception('gmod.nameparabasefinl is empty.')

            if gdat.typeverb > 0:
                print('%s has been fixed for %s to %g...' % (nameparabasefinl, strgmodl, getattr(gmod, nameparabasefinl)))
    
    if gdat.booldiag:
        if minmpara is not None and (not isinstance(minmpara, float) or not isinstance(maxmpara, float)):
            print('')
            print('')
            print('')
            print('minmpara')
            print(minmpara)
            print('maxmpara')
            print(maxmpara)
            print('nameparabase')
            print(nameparabase)
            raise Exception('not isinstance(minmpara, float) or not isinstance(maxmpara, float)')

    gmod.listlablpara.append(lablpara)
    gmod.listminmpara.append(minmpara)
    gmod.listmaxmpara.append(maxmpara)
    gmod.dictfeatpara['scal'].append('self')
    
    # add the name of the parameter to the list of the parameters of the model
    gmod.listnameparafull += [nameparabasefinl]
    if gdat.booldiag:
        if strgmodl == 'true':
            if not hasattr(gdat.true, nameparabasefinl):
                print('')
                print('')
                print('')
                print('strgmodl')
                print(strgmodl)
                print('gdat.true.boolmodlpsys')
                print(gdat.true.boolmodlpsys)
                print('gmod.typemodlblinener')
                print(gmod.typemodlblinener)
                print('gmod.typemodlblinshap')
                print(gmod.typemodlblinshap)
                print('nameparabasefinl')
                print(nameparabasefinl)
                raise Exception('The true model parameter you are defining lacks the default value!')

    if boolvari and strgmodl == 'fitt':
        ## varied parameters
        gmod.listnameparafullvari += [nameparabasefinl]
    
        print('setp_para: Setting up gmod.dictindxpara[%s] of %s with gmod.cntr=%d...' % (nameparabasefinl, strgmodl, gmod.cntr))
    
        gmod.dictindxpara[nameparabasefinl] = gmod.cntr
    
        #if strgener is not None:
        #    if gdat.fitt.typemodlenerfitt == 'full':
        #        intg = int(strgener[-2:])
        #    else:
        #        intg = 0
        #if strgcomp is not None and strgener is not None:
        #    if gdat.fitt.typemodlenerfitt == 'full':
        #        gmod.dictindxpara[nameparabase + 'comp%s' % strgener][int(strgcomp[-1]), intg] = gmod.cntr
        #    else:
        #        gmod.dictindxpara[nameparabase + 'comp%s' % strgener][int(strgcomp[-1]), 0] = gmod.cntr
        #elif strglmdk is not None and strgener is not None:
        #    if strglmdk == 'linr':
        #        intglmdk = 0
        #    if strglmdk == 'quad':
        #        intglmdk = 1
        #    gmod.dictindxpara[nameparabase + 'ener'][intglmdk, intg] = gmod.cntr
        #elif strgener is not None:
        #    gmod.dictindxpara[nameparabase + 'ener'][intg] = gmod.cntr
        #elif strgcomp is not None:
        #    gmod.dictindxpara[nameparabase + 'comp'][int(strgcomp[-1])] = gmod.cntr
        
        gmod.cntr += 1
    else:
        ## fixed parameters
        gmod.listnameparafullfixd += [nameparabasefinl]
    

def proc_modl(gdat, strgmodl, strgextn, h):
    
    gmod = getattr(gdat, strgmodl)

    strgextn = 'PosteriorMedian%s%s' % (gdat.strgcnfg, gdat.liststrgdatafittiter[h])
    
    if gdat.booldiag:
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                if len(gdat.arrytser['Detrended'][b][p]) == 0:
                    print('')
                    print('')
                    print('')
                    print('bp')
                    print(bp)
                    raise Exception('')
        
    #gdat.timethis = gdat.arrytser['Detrended'][b][p][:, 0, 0]
    #gdat.rflxthis = gdat.arrytser['Detrended'][b][p][:, :, 1]
    #gdat.stdvrflxthis = gdat.arrytser['Detrended'][b][p][:, :, 2]
    #gdat.varirflxthis = gdat.stdvrflxthis**2

    print('gmod.listminmpara')
    print(gmod.listminmpara)
    gmod.listminmpara = np.array(gmod.listminmpara)
    gmod.listmaxmpara = np.array(gmod.listmaxmpara)
    
    # get default parameter labels if they have not been provided
    gmod.listlablpara, _, _, _, _ = tdpy.retr_listlablscalpara(gdat.fitt.listnameparafull, gmod.listlablpara, booldiag=gdat.booldiag)
    
    # merge variable labels with units
    gmod.listlablparatotl = tdpy.retr_listlabltotl(gmod.listlablpara)
    
    if (gmod.listminmpara == None).any():
        print('')
        print('')
        print('')
        print('gmod.listlablpara[k], gmod.listminmpara[k], gmod.listmaxmpara[k]')
        for k in range(len(gmod.listminmpara)):
            print(gmod.listlablpara[k], gmod.listminmpara[k], gmod.listmaxmpara[k])
        print('gmod.listminmpara')
        print(gmod.listminmpara)
        print('gmod.listmaxmpara')
        print(gmod.listmaxmpara)
        print('gmod.listlablpara')
        print(gmod.listlablpara)
        raise Exception('(gmod.listminmpara == None).any()')

    if gdat.booldiag:
        if None in gmod.listlablpara:
            print('')
            print('')
            print('')
            print('gmod.listlablpara')
            print(gmod.listlablpara)
            raise Exception('')
    
    gdat.numbpara = len(gdat.fitt.listnameparafullvari)
    gdat.meanpara = np.empty(gdat.numbpara)
    gdat.stdvpara = np.empty(gdat.numbpara)
    
    gdat.bfitperi = 4.25 # [days]
    gdat.stdvperi = 1e-2 * gdat.bfitperi # [days]
    gdat.bfitduratran = 0.45 * 24. # [hours]
    gdat.stdvduratran = 1e-1 * gdat.bfitduratran # [hours]
    gdat.bfitamplslen = 0.14 # [relative]
    gdat.stdvamplslen = 1e-1 * gdat.bfitamplslen # [relative]
    
    meangauspara = None
    stdvgauspara = None
    numbpara = len(gmod.listlablpara)
    indxpara = np.arange(numbpara)
    
    listscalpara = gmod.dictfeatpara['scal']
    
    gdat.thisstrgmodl = 'fitt'
    # run the sampler
    if gdat.typeinfe == 'samp':
        print('Will call tdpy.samp()...')
        gdat.dictsamp = tdpy.samp(gdat, \
                                  gdat.numbsampwalk, \
                                  retr_llik_mile, \
                                  gdat.fitt.listnameparafullvari, gmod.listlablpara, listscalpara, gmod.listminmpara, gmod.listmaxmpara, \
                                  pathbase=gdat.pathtargcnfg, \
                                  retr_dictderi=retr_dictderi_mile, \
                                  numbsampburnwalk=gdat.numbsampburnwalk, \
                                  strgextn=strgextn, \
                                  typeverb=gdat.typeverb, \
                                  boolplot=gdat.boolplot, \
                                 )
        
        gdat.numbsamp = gdat.dictsamp['lpos'].size
        gdat.indxsamp = np.arange(gdat.numbsamp)
        gdat.numbsampplot = min(10, gdat.numbsamp)
        
        if gdat.booldiag:
            for b in gdat.indxdatatser:
                for p in gdat.indxinst[b]:
                    strg = '%s_%s' % (gdat.liststrgdatatser[b], gdat.listlablinst[b][p])
                    
                    for namecompmodl in gmod.listnamecompmodl:
                        namecompmodlextn = 'Model_Fine_%s_%s' % (namecompmodl, strg)
                        if gdat.typeinfe == 'samp':
                            if gdat.fitt.typemodlenerfitt == 'full':
                                lcurtemp = np.median(gdat.dictsamp[namecompmodlextn][:, :, :], 0)
                            else:
                                lcurtemp = np.median(gdat.dictsamp[namecompmodlextn][:, :, 0], 0)
                            strgtitl = 'Posterior median model'
                        else:
                            if gdat.fitt.typemodlenerfitt == 'full':
                                lcurtemp = gdat.dictmlik[namecompmodlextn][:, h]
                            else:
                                lcurtemp = gmod.listdictmlik[h][namecompmodlextn][:, 0]
                        
                    if p is None:
                        time = gdat.timethisfittconc[b]
                        timefine = gdat.timethisfittfineconc[b]
                    else:
                        time = gdat.timethisfitt[b][p]
                        timefine = gdat.timethisfittfine[b][p]
                        if gdat.fitt.typemodlenerfitt == 'full':
                            tser = gdat.rflxthisfitt[b][p][:, :]
                        else:
                            tser = gdat.rflxthisfitt[b][p][:, 0]
    
                        if timefine.size != lcurtemp.size:
                            print('')
                            print('')
                            print('')
                            print('p')
                            print(p)
                            print('gdat.typeinfe')
                            print(gdat.typeinfe)
                            print('gdat.fitt.typemodlenerfitt')
                            print(gdat.fitt.typemodlenerfitt)
                            print('namecompmodlextn')
                            print(namecompmodlextn)
                            print('lcurtemp')
                            summgene(lcurtemp)
                            print('timefine')
                            summgene(timefine)
                            raise Exception('timefine.size != lcurtemp.size')

    if gdat.typeinfe == 'opti':
        
        bounds = [[] for kk in range(gmod.listminmpara.size)]
        print('bounds')
        for kk in range(gmod.listminmpara.size):
            bounds[kk] = [gmod.listminmpara[kk], gmod.listmaxmpara[kk]]
            print('%s %s: %g %g' % (gdat.fitt.listnameparafullvari[kk], gmod.listlablpara[kk], gmod.listminmpara[kk], gmod.listmaxmpara[kk]))
        print('')
            
        indx = np.where(gdat.liststrgdatafittitermlikdone == gdat.liststrgdatafittiter[gdat.indxfittiterthis])[0]
        if indx.size == 1:
            print('Reading from the stored solution...')
            paramlik = gdat.datamlik[indx[0]][np.arange(0, 2 * gmod.listminmpara.size - 1, 2)]
            stdvmlik = gdat.datamlik[indx[0]][np.arange(0, 2 * gmod.listminmpara.size - 1, 2) + 1]
        else:

            gdat.parainit = gmod.listminmpara + 0.5 * (gmod.listmaxmpara - gmod.listminmpara)
            
            print('gdat.parainit')
            for kk in range(gmod.listminmpara.size):
                print('%s %s: %g' % (gdat.fitt.listnameparafullvari[kk], gmod.listlablpara[kk], gdat.parainit[kk]))
            print('')
            
            print('Maximizing the likelihood...')
            # minimize the negative loglikelihood
            objtmini = scipy.optimize.minimize(retr_lliknega_mile, gdat.parainit, \
                                                                            method='Nelder-Mead', \
                                                                            #method='BFGS', \
                                                                            #method='L-BFGS-B', \
                                                                            #ftol=0.1, \
                                                                            options={ \
                                                                            #"initial_simplex": simplex,
                                                                                        "disp": True, \
                                                                                        "maxiter" : gdat.parainit.size*200,
                                                                                        "fatol": 0.2, \
                                                                                        "adaptive": True, \
                                                                                        }, \
                                                                                          bounds=bounds, args=(gdat))
            
            paramlik = objtmini.x
            print('objtmini.success')
            print(objtmini.success)
            print(objtmini.status)
            print(objtmini.message)
            #print(objtmini.hess)
            print()
            gdat.indxpara = np.arange(paramlik.size)
            #stdvmlik = objtmini.hess_inv[gdat.indxpara, gdat.indxpara]
            stdvmlik = np.empty_like(paramlik)
            deltpara = 1e-6
            for kk in gdat.indxpara:
                paranewwfrst = np.copy(paramlik)
                paranewwfrst[kk] = (1 - deltpara) * paranewwfrst[kk]
                paranewwseco = np.copy(paramlik)
                paranewwseco[kk] = (1 + deltpara) * paranewwseco[kk]
                stdvmlik[kk] = 1. / np.sqrt(abs(retr_lliknega_mile(paranewwfrst, gdat) + retr_lliknega_mile(paranewwseco, gdat) \
                                                                             - 2. * retr_lliknega_mile(paramlik, gdat)) / (deltpara * paramlik[kk])**2)

            path = gdat.pathdatatarg + 'paramlik.csv'
            if gdat.typeverb > 0:
                print('Writing to %s...' % path)
            objtfile = open(path, 'a+')
            objtfile.write('%s' % gdat.liststrgdatafittiter[gdat.indxfittiterthis])
            for kk, paramliktemp in enumerate(paramlik):
                objtfile.write(', %g, %g' % (paramliktemp, stdvmlik[kk]))
            objtfile.write('\n')
            objtfile.close()
        
        print('paramlik')
        for kk in range(gmod.listminmpara.size):
            print('%s %s: %g +- %g' % (gdat.fitt.listnameparafullvari[kk], gmod.listlablpara[kk], paramlik[kk], stdvmlik[kk]))
        
        gdat.dictmlik = dict()
        for kk in range(gmod.listminmpara.size):
            gdat.dictmlik[gdat.fitt.listnameparafullvari[kk]] = paramlik[kk]
            gdat.dictmlik['stdv' + gdat.fitt.listnameparafullvari[kk]] = stdvmlik[kk]
        print('Computing derived variables...')
        dictderimlik = retr_dictderi_mile(paramlik, gdat)
        for name in dictderimlik:
            gdat.dictmlik[name] = dictderimlik[name]
                
    if gdat.typeinfe == 'samp':
        gmod.listdictsamp.append(gdat.dictsamp)
    if gdat.typeinfe == 'opti':
        gmod.listdictmlik.append(gdat.dictmlik)

    if gdat.boolplottser:
        
        #timedata = gdat.timethisfitt
        #lcurdata = gdat.rflxthisfitt[:, e]
            
        for b in gdat.indxdatatser:
            #plot_tsermodlpost(gdat, strgmodl, b, None, None, e)
            for p in gdat.indxinst[b]:
                for e in gdat.indxener[p]:
                    if e < 5:
                        #plot_tsermodlpost(gdat, strgmodl, b, p, None, e)
                        for y in gdat.indxchun[b][p]:
                            plot_tsermodlpost(gdat, strgmodl, b, p, y, e, h)


def plot_tsermodlpost(gdat, strgmodl, b, p, y, e, h):
    
    gmod = getattr(gdat, strgmodl)
            
    if p is None:
        time = gdat.timethisfittconc[b]
        timefine = gdat.timethisfittfineconc[b]
    else:
        time = gdat.timethisfitt[b][p]
        timefine = gdat.timethisfittfine[b][p]
        tser = gdat.rflxthisfitt[b][p][:, e]
    
    strg = '%s_%s' % (gdat.liststrgdatatser[b], gdat.listlablinst[b][p])
    
    # plot the data with the posterior median model
    strgextn = 'PosteriorMedian%s%s' % (gdat.strgcnfg, gdat.liststrgdatafittiter[h])
    dictmodl = dict()
    for namecompmodl in gmod.listnamecompmodl:
        namecompmodlextn = 'Model_Fine_%s_%s' % (namecompmodl, strg)
        if gdat.typeinfe == 'samp':
            if gdat.fitt.typemodlenerfitt == 'full':
                lcurtemp = np.median(gdat.dictsamp[namecompmodlextn][:, :, e], 0)
            else:
                lcurtemp = np.median(gmod.listdictsamp[e][namecompmodlextn][:, :, 0], 0)
            strgtitl = 'Posterior median model'
        else:
            if gdat.fitt.typemodlenerfitt == 'full':
                lcurtemp = gdat.dictmlik[namecompmodlextn][:, e]
            else:
                lcurtemp = gmod.listdictmlik[e][namecompmodlextn][:, 0]
        
        if gdat.booldiag:
            if timefine.size != lcurtemp.size:
                print('')
                print('')
                print('')
                print('p')
                print(p)
                print('gdat.typeinfe')
                print(gdat.typeinfe)
                print('gdat.fitt.typemodlenerfitt')
                print(gdat.fitt.typemodlenerfitt)
                print('namecompmodlextn')
                print(namecompmodlextn)
                print('lcurtemp')
                summgene(lcurtemp)
                print('timefine')
                summgene(timefine)
                raise Exception('timefine.size != lcurtemp.size')

        if namecompmodl == 'Total':
            colr = 'b'
            labl = 'Total Model'
        elif namecompmodl == 'Baseline':
            colr = 'orange'
            labl = 'Baseline'
        elif namecompmodl == 'Transit':
            colr = 'r'
            labl = 'Transit'
        elif namecompmodl == 'StarFlaring':
            colr = 'g'
            labl = 'Flares'
        elif namecompmodl == 'excs':
            colr = 'olive'
            labl = 'Excess'
        else:
            print('')
            print('namecompmodl')
            print(namecompmodl)
            raise Exception('')
        dictmodl['pmed' + namecompmodlextn] = {'tser': lcurtemp, 'time': timefine, 'labl': labl, 'colr': colr}
    
    if p is not None and gdat.listlablinst[b][p] != '':
        strglablinst = ', %s' % gdat.listlablinst[b][p]
    else:
        strglablinst = ''
    
    if gdat.lablcnfg != '':
        lablcnfgtemp = ', %s' % gdat.lablcnfg
    else:
        lablcnfgtemp = ''

    if e == 0 and gdat.numbener[p] == 1:
        strgtitl = '%s%s%s' % (gdat.labltarg, strglablinst, lablcnfgtemp)
    elif e == 0 and gdat.numbener[p] > 1:
        strgtitl = '%s%s%s, white' % (gdat.labltarg, strglablinst, lablcnfgtemp)
    else:
        strgtitl = '%s%s%s, %g micron' % (gdat.labltarg, strglablinst, lablcnfgtemp, gdat.listener[p][e-1])
    
    if gdat.booldiag:
        if tser.ndim != 1:
            print('')
            print('')
            print('')
            print('p')
            print(p)
            print('tser')
            summgene(tser)
            raise Exception('tser.ndim != 1')
    
    pathplot = plot_tser( \
                         gdat.pathvisutarg, \
                         timedata=time, \
                         tserdata=tser, \
                         timeoffs=gdat.timeoffs, \
                         strgextn=strgextn, \
                         strgtitl=strgtitl, \
                         boolwritover=gdat.boolwritover, \
                         boolbrekmodl=gdat.boolbrekmodl, \
                         dictmodl=dictmodl, \
                         booldiag=gdat.booldiag, \
                        )
    
    # plot the posterior median residual
    strgextn = 'ResidualPosteriorMedian%s%s' % (gdat.strgcnfg, gdat.liststrgdatafittiter[h])
    if gdat.typeinfe == 'samp':
        if gdat.fitt.typemodlenerfitt == 'full':
            tserdatatemp = np.median(gdat.dictsamp['resi%s' % strg][:, :, e], 0)
        else:
            tserdatatemp = np.median(gmod.listdictsamp[e]['resi%s' % strg][:, :, 0], 0)
    else:
        if gdat.fitt.typemodlenerfitt == 'full':
            tserdatatemp = gdat.dictmlik['resi%s' % strg][:, e]
        else:
            tserdatatemp = gmod.listdictmlik[e]['resi%s' % strg][:, 0]
    
    if gdat.booldiag:
        if tserdatatemp.ndim != 1:
            print('')
            print('')
            print('')
            print('tserdatatemp')
            summgene(tserdatatemp)
            raise Exception('tserdatatemp.ndim != 1')
    
    pathplot = plot_tser(gdat.pathvisutarg, \
                                 timedata=time, \
                                 tserdata=tserdatatemp, \
                                 timeoffs=gdat.timeoffs, \
                                 strgextn=strgextn, \
                                 strgtitl=strgtitl, \
                                 lablyaxi='Residual relative flux', \
                                 boolwritover=gdat.boolwritover, \
                                 boolbrekmodl=gdat.boolbrekmodl, \
                                )
    
    # plot the data with a number of total model samples
    if gdat.typeinfe == 'samp':
        strgextn = 'PosteriorSamples%s' % gdat.strgcnfg
        if gdat.numbener[p] > 1:
            strgextn += gdat.liststrgdatafittiter[h]
        dictmodl = dict()
        for w in range(gdat.numbsampplot):
            namevarbsamp = 'PosteriorSamplesmodl%04d' % w
            if gdat.fitt.typemodlenerfitt == 'full':
                dictmodl[namevarbsamp] = {'tser': gdat.dictsamp['Model_Fine_Total_%s' % strg][w, :, e], 'time': timefine}
            else:
                dictmodl[namevarbsamp] = {'tser': gmod.listdictsamp[e]['Model_Fine_Total_%s' % strg][w, :, 0], 'time': timefine}
            
            if gdat.booldiag:
                if dictmodl[namevarbsamp]['tser'].size != dictmodl[namevarbsamp]['time'].size:
                    print('')
                    print('strg')
                    print(strg)
                    print('dictmodl[namevarbsamp][tser]')
                    summgene(dictmodl[namevarbsamp]['tser'])
                    print('dictmodl[namevarbsamp][time]')
                    summgene(dictmodl[namevarbsamp]['time'])
                    raise Exception('')

            if w == 0:
                dictmodl[namevarbsamp]['labl'] = 'Model'
            else:
                dictmodl[namevarbsamp]['labl'] = None
            dictmodl[namevarbsamp]['colr'] = 'b'
            dictmodl[namevarbsamp]['alph'] = 0.2
        pathplot = plot_tser(gdat.pathvisutarg, \
                                     timedata=gdat.timethisfitt[b][p], \
                                     tserdata=gdat.rflxthisfitt[b][p][:, e], \
                                     timeoffs=gdat.timeoffs, \
                                     strgextn=strgextn, \
                                     boolwritover=gdat.boolwritover, \
                                     strgtitl=strgtitl, \
                                     boolbrekmodl=gdat.boolbrekmodl, \
                                     dictmodl=dictmodl)

        # plot the data with a number of model component samples
        strgextn = 'PosteriorSamplesComponent%s' % gdat.strgcnfg
        if gdat.numbener[p] > 1:
            strgextn += gdat.liststrgdatafittiter[h]
        dictmodl = dict()
        for namecompmodl in gdat.fitt.listnamecompmodl:
            if namecompmodl == 'Total':
                continue

            if namecompmodl == 'Total':
                colr = 'b'
                labl = 'Total Model'
            elif namecompmodl == 'Baseline':
                colr = 'orange'
                labl = 'Baseline'
            elif namecompmodl == 'Transit':
                colr = 'r'
                labl = 'Transit'
            elif namecompmodl == 'StarFlaring':
                colr = 'g'
                labl = 'Flares'
            elif namecompmodl == 'excs':
                colr = 'olive'
                labl = 'Excess'
            else:
                print('')
                print('namecompmodl')
                print(namecompmodl)
                raise Exception('')

            for w in range(gdat.numbsampplot):
                namevarbsamp = 'PosteriorSamples%s%04d' % (namecompmodl, w)
                if gdat.fitt.typemodlenerfitt == 'full':
                    dictmodl[namevarbsamp] = {'tser': gdat.dictsamp['Model_Fine_%s_%s' % (namecompmodl, strg)][w, :, e], 'time': gdat.timethisfittfine[b][p]}
                else:
                    dictmodl[namevarbsamp] = \
                                {'tser': gmod.listdictsamp[e]['Model_Fine_%s_%s' % (namecompmodl, strg)][w, :, 0], 'time': gdat.timethisfittfine[b][p]}
                if w == 0:
                    dictmodl[namevarbsamp]['labl'] = labl
                else:
                    dictmodl[namevarbsamp]['labl'] = None
                dictmodl[namevarbsamp]['colr'] = colr
                dictmodl[namevarbsamp]['alph'] = 0.6
        pathplot = plot_tser(gdat.pathvisutarg, \
                                     timedata=gdat.timethisfitt[b][p], \
                                     timeoffs=gdat.timeoffs, \
                                     tserdata=gdat.rflxthisfitt[b][p][:, e], \
                                     strgextn=strgextn, \
                                     boolwritover=gdat.boolwritover, \
                                     strgtitl=strgtitl, \
                                     boolbrekmodl=gdat.boolbrekmodl, \
                                     dictmodl=dictmodl)

    # plot the binned RMS
    path = gdat.pathvisutarg + 'stdvrebn%s%s.%s' % (gdat.strgcnfg, gdat.liststrgdatafittiter[h], gdat.typefileplot)
    if not os.path.exists(path):
        if gdat.typeinfe == 'samp':
            if gdat.fitt.typemodlenerfitt == 'full':
                stdvresi = np.median(gdat.dictsamp['stdvresi%s' % strg][:, :, e], 0)
            else:
                stdvresi = np.median(gmod.listdictsamp[e]['stdvresi%s' % strg][:, :, 0], 0)
        else:
            if gdat.fitt.typemodlenerfitt == 'full':
                stdvresi = gdat.dictmlik['stdvresi%s' % strg][:, e]
            else:
                stdvresi = gmod.listdictmlik[e]['stdvresi' % strg][:, 0]
    
        figr, axis = plt.subplots(figsize=gdat.figrsizeydob)
        axis.loglog(gdat.listdeltrebn[b][p] * 24., stdvresi * 1e6, ls='', marker='o', ms=1, label='Binned Std. Dev')
        axis.axvline(gdat.cadetime[b][p] * 24., ls='--', label='Sampling rate')
        axis.set_ylabel('RMS [ppm]')
        axis.set_xlabel('Bin width [hour]')
        axis.legend()
        plt.tight_layout()
        if gdat.typeverb > 0:
            print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()
        

def setp_modlinit(gdat, strgmodl):
    '''
    Set up the modeling variables for the model...
    '''
    gmod = getattr(gdat, strgmodl)
    
    print('Performing initial setup for model %s...' % strgmodl)
    gmod.boolmodlcosc = gmod.typemodl == 'CompactObjectStellarCompanion'
    
    print('gmod.boolmodlcosc')
    print(gmod.boolmodlcosc)
    
    gmod.boolmodlpsys = gmod.typemodl.startswith('PlanetarySystem')
    
    if gdat.typeverb > 0:
        print('gmod.boolmodlpsys')
        print(gmod.boolmodlpsys)
    
    gmod.boolmodlcomp = gmod.boolmodlpsys or gmod.boolmodlcosc
    print('gmod.boolmodlcomp')
    print(gmod.boolmodlcomp)
        
    gmod.boolmodlpcur = gmod.typemodl == 'PlanetarySystemEmittingCompanion'
    
    print('gmod.boolmodlpcur')
    print(gmod.boolmodlpcur)
    
    gmod.listnameparafullfixd = []
    gmod.listnameparafullvari = []

    # type of baseline shape
    tdpy.setp_para_defa(gdat, strgmodl, 'typemodlblinshap', 'cons')
    
    # Boolean flag to indicate that radius ratio can be different across passbands
    tdpy.setp_para_defa(gdat, strgmodl, 'boolvarirratband', False)
    
    # Boolean flag to indicate that radius ratio can be different across passbands
    tdpy.setp_para_defa(gdat, strgmodl, 'boolvariduratranband', False)
    

def setp_modlmedi(gdat, strgmodl):
    
    # type of baseline energy dependence
    typemodlblinener = ['cons' for p in gdat.indxinst[0]]
    for p in gdat.indxinst[0]:
        tdpy.setp_para_defa(gdat, strgmodl, 'typemodlblinener', typemodlblinener)
    

# this will likely be merged with setp_modlbase()
def init_modl(gdat, strgmodl):
    
    gmod = getattr(gdat, strgmodl)
    
    gmod.dictindxpara = dict()
    gmod.dictfeatpara = dict()
    gmod.dictfeatpara['scal'] = []
    
    gmod.listlablpara = []
    gmod.listminmpara = []
    gmod.listmaxmpara = []
    gmod.listnameparafull = []
            
    # counter for the parameter index
    gmod.cntr = 0


def setp_modlbase(gdat, strgmodl, h=None):
    
    print('')
    print('Setting up the model (%s) by running setp_modlbase()...' % strgmodl)
    
    gmod = getattr(gdat, strgmodl)
    
    print('gmod.typemodl')
    print(gmod.typemodl)

    # systematic baseline
    gmod.listnamecompmodl = ['Baseline']
    
    if gmod.typemodl == 'StarFlaring':
        gmod.listnamecompmodl += ['StarFlaring']
    
    if gmod.typemodl == 'CompactObjectStellarCompanion' or gmod.boolmodlpsys:
        gmod.listnamecompmodl += ['Transit']
        
        if gmod.boolmodlcomp:
            if strgmodl == 'true':
                gmod.numbcomp = gdat.true.epocmtracomp.size
            else:
                gmod.numbcomp = gdat.fitt.prio.meanpara.epocmtracomp.size
        else:
            gmod.numbcomp = 0
        
        if gdat.booldiag:
            if gmod.numbcomp > 100:
                print('')
                print('')
                print('')
                raise Exception('Too many components.')

        if gdat.typeverb > 0:
            print('gmod.numbcomp')
            print(gmod.numbcomp)
        
        gmod.indxcomp = np.arange(gmod.numbcomp)
     
        tdpy.setp_para_defa(gdat, strgmodl, 'typemodllmdkener', 'cons')
        tdpy.setp_para_defa(gdat, strgmodl, 'typemodllmdkterm', 'quad')
    
        if gdat.typeverb > 0:
            print('gmod.typemodllmdkener')
            print(gmod.typemodllmdkener)
            print('gmod.typemodllmdkterm')
            print(gmod.typemodllmdkterm)

        # number of terms in the LD law
        if gmod.typemodllmdkterm == 'line':
            gmod.numbcoeflmdkterm = 1
        if gmod.typemodllmdkterm == 'quad':
            gmod.numbcoeflmdkterm = 2
        
        # number of distinct coefficients for each term in the LD law
        if gmod.typemodllmdkener == 'ener':
            gmod.numbcoeflmdkener = gdat.numbener[p]
        else:
            gmod.numbcoeflmdkener = 1
            
    gdat.listnamecompgpro = ['Total']
    if gmod.typemodlblinshap == 'GaussianProcess':
        gdat.listnamecompgpro.append('Baseline')
    
    if len(gmod.listnamecompmodl) > 1:
        gmod.listnamecompmodl += ['Total']
    
    # baseline
    gmod.listnameparablin = []
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            # string extension for the data type and instrument
            if gdat.numbener[p] > 1 and gmod.typemodlblinener[p] == 'ener':
                for e in gdat.indxener[p]:
                    strgextninst = '%s%s%s' % (gdat.liststrginst[b][p], gdat.strgextncade[b][p], gdat.liststrgener[p][e]) 
            else:
                strgextninst = '%s' % (gdat.liststrginst[b][p])
            
            # list of parameters for each of photometry and RV and for each instrument
            listnameparablinshap = []
            if gmod.typemodlblinshap == 'cons':
                listnameparablinshap += ['consblin']
            if gmod.typemodlblinshap == 'GaussianProcess':
                listnameparablinshap += ['sigmgprobase', 'rhoogprobase']
            if gmod.typemodlblinshap == 'step':
                listnameparablinshap += ['consblinfrst', 'consblinseco', 'timestep', 'scalstep']
            
            for nameparablinshap in listnameparablinshap:
                
                if nameparablinshap.startswith('sigmgprobase'):
                    minmpara = 0.01 # [ppt]
                    maxmpara = 4. # [ppt]
                    lablpara = ['$\sigma_{GP}$', '']
                if nameparablinshap.startswith('rhoogprobase'):
                    minmpara = 1e-3
                    maxmpara = 0.3
                    lablpara = [r'$\rho_{GP}$', '']
                if nameparablinshap.startswith('consblin'):
                    if nameparablinshap == 'consblinfrst':
                        lablpara = ['$C_1$', 'ppt']
                        minmpara = -20. # [ppt]
                        maxmpara = 20. # [ppt]
                    elif nameparablinshap == 'consblinseco':
                        lablpara = ['$C_2$', 'ppt']
                        minmpara = -20. # [ppt]
                        maxmpara = -4. # [ppt]
                    else:
                        lablpara = ['$C$, %s' % gdat.listlablinst[b][p], 'ppt']
                        minmpara = -20. # [ppt]
                        maxmpara = 20. # [ppt]
                if nameparablinshap.startswith('timestep'):
                    minmpara = 791.11
                    maxmpara = 791.13
                    lablpara = '$T_s$'
                if nameparablinshap.startswith('scalstep'):
                    minmpara = 0.0001
                    maxmpara = 0.002
                    lablpara = '$A_s$'
                
                nameparablin = nameparablinshap + strgextninst
                setp_para(gdat, strgmodl, nameparablin, minmpara, maxmpara, lablpara)
                gmod.listnameparablin += [nameparablin]
    
    print('gmod.listnameparablin')
    print(gmod.listnameparablin)
    
    gmod.listindxdatainsteneriter = []
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            if strgmodl == 'true':
                gmod.listindxdatainsteneriter.append([b, p, gdat.indxener])
            else:
                if gdat.fitt.typemodlenerfitt == 'full':
                    gmod.listindxdatainsteneriter.append([b, p, gdat.indxener])
                else:
                    for e in gdat.indxener[p]:
                        gmod.listindxdatainsteneriter.append([np.array([e])])
        
    tdpy.setp_para_defa(gdat, strgmodl, 'timestep', 791.12)
    tdpy.setp_para_defa(gdat, strgmodl, 'scalstep', 0.00125147)
                        
    if gmod.typemodl == 'StarFlaring':
        setp_para(gdat, strgmodl, 'numbflar', 0, 10, ['$N_f$', ''], boolvari=False)
        
        # fixed parameters of the fitting model
        if strgmodl == 'fitt':
            tdpy.setp_para_defa(gdat, strgmodl, 'numbflar', 1)
        
        gmod.indxflar = np.arange(gmod.numbflar)
        for k in gmod.indxflar:
            setp_para(gdat, strgmodl, 'amplflar%04d' % k, 0., 0.15, ['$A_{%d}$' % k, ''])
            setp_para(gdat, strgmodl, 'tsclflar%04d' % k, 0., 2., ['$t_{s,%d}$' % k, 'hour'])
            setp_para(gdat, strgmodl, 'timeflar%04d' % k, 0., 0.15, ['$t_{f,%d}$' % k, 'day'])

    if strgmodl == 'true':
        if gmod.boolmodlpsys or gmod.typemodl == 'CompactObjectStellarCompanion':
            if gdat.true.typemodllmdkener == 'linr':
                pass
            elif gdat.true.typemodllmdkener == 'cons':
                tdpy.setp_para_defa(gdat, 'true', 'coeflmdklinr', 0.4)
                tdpy.setp_para_defa(gdat, 'true', 'coeflmdkquad', 0.25)
            elif gdat.true.typemodllmdkener == 'ener':
                tdpy.setp_para_defa(gdat, 'true', 'coeflmdklinrwhit', 0.4)
                tdpy.setp_para_defa(gdat, 'true', 'coeflmdkquadwhit', 0.25)
                for p in gdat.indxinst[0]:
                    tdpy.setp_para_defa(gdat, 'true', 'coeflmdklinr' % strginst, 0.4)
                    tdpy.setp_para_defa(gdat, 'true', 'coeflmdkquad' % strginst, 0.25)
            
    if gmod.boolmodlpsys or gmod.typemodl == 'CompactObjectStellarCompanion':
        
        #gmod.listnameparasyst = []

        # list of companion parameter names
        gmod.listnameparacomp = [[] for j in gmod.indxcomp]
        for j in gmod.indxcomp:
            if gmod.typemodl == 'PlanetarySystemWithTTVs':
                if gmod.typemodlttvr == 'indilineflot' or gmod.typemodlttvr == 'globlineflot':
                    gmod.listnameparacomp[j] += ['peri', 'epocmtra']
                if gmod.typemodlttvr == 'globlineuser' or gmod.typemodlttvr == 'globlineflot':
                    for lll in range(gdat.numbtran[j]):
                        gmod.listnameparacomp[j] += ['ttvr%04d' % lll]
            if gmod.typemodl == 'CompactObjectStellarCompanion':
                gmod.listnameparacomp[j] += ['mass']
            if gmod.boolmodlpsys and not (gmod.typemodl == 'PlanetarySystemWithTTVs' and gmod.typemodlttvr == 'indilineuser'):
                gmod.listnameparacomp[j] += ['rrat']
                gmod.listnameparacomp[j] += ['rsma', 'peri', 'epocmtra', 'cosi']
            
            if gdat.booldiag:
                if len(gmod.listnameparacomp[j]) == 0:
                    print('')
                    print('')
                    print('')
                    print('j')
                    print(j)
                    print('gmod.typemodl')
                    print(gmod.typemodl)
                    raise Exception('gmod.listnameparacomp[j] could not be defined.')

            # transfer system parameters from the true model dictionary to the global object
            if strgmodl == 'true' and 'epocmtracomp' in gdat.dicttrue:
                print('temp: this is over-writing different companions')
                for namepara in gdat.true.listnameparacomp[j]:
                    setattr(gmod, namepara + 'comp', gdat.dicttrue[namepara + 'comp'])
        
        if strgmodl == 'true':
            for j in gdat.true.indxcomp:
                # copy array base component parameters of the true model to scalar base parameters 
                for namepara in gdat.true.listnameparacomp[j]:
                    paracomp = getattr(gdat.true, namepara + 'comp')
                    
                    if gdat.booldiag:
                        if j > 100:
                            print('')
                            print('')
                            print('')
                            raise Exception('Too many components.')

                        if paracomp is None or np.isscalar(paracomp):
                            print('')
                            print('')
                            print('')
                            print('namepara')
                            print(namepara)
                            print('paracomp')
                            print(paracomp)
                            raise Exception('')
                    
                    if len(paracomp) > 0:
                        tdpy.setp_para_defa(gdat, 'true', namepara + 'com%d' % j, paracomp[j])
            
        if gdat.booldiag:
            if strgmodl == 'fitt' and gmod.boolmodlpsys and gmod.numbcomp == 0:
                print('')
                print('')
                print('')
                print('gmod.typemodl')
                print(gmod.typemodl)
                raise Exception('Fitting PlanetarySystem model does not have any companions.')

        # define arrays of parameter indices for companions
        #if strgmodl == 'fitt' or strgmodl == 'true' and gmod.numbcomp > 0:
        for j in gmod.indxcomp:
            for namepara in gmod.listnameparacomp[j]:
                if gmod.listnameparacomp[j] != 'rrat' or gmod.listnameparacomp[j] == 'rrat' and gdat.numbener[p] == 1:
                    gmod.dictindxpara[namepara + 'comp'] = np.empty(gmod.numbcomp, dtype=int)
                else:
                    if gdat.numbener[p] > 1 and gdat.fitt.typemodlenerfitt == 'full':
                        gmod.dictindxpara['rratcompener'] = np.empty((gmod.numbcomp, gdat.numbener[p]), dtype=int)
                    else:
                        gmod.dictindxpara['rratcompener'] = np.empty((gmod.numbcomp, gdat.numbener[p]), dtype=int)
        
        print('gmod.typemodllmdkterm')
        print(gmod.typemodllmdkterm)

        # limb darkening
        if gmod.typemodllmdkterm != 'none':
            if gmod.typemodllmdkener == 'ener' and gdat.fitt.typemodlenerfitt == 'full':
                gmod.dictindxpara['coeflmdkener'] = np.empty((gmod.numbcoeflmdkterm, gmod.numbcoeflmdkener), dtype=int)
            else:
                gmod.dictindxpara['coeflmdkener'] = np.empty((gmod.numbcoeflmdkterm, 1), dtype=int)
        
            #gmod.dictindxpara['coeflmdklinrener'] = np.empty(1, dtype=int)
            #gmod.dictindxpara['coeflmdkquadener'] = np.empty(1, dtype=int)
            
            print('setp_para calls relevant to coeflmdk...')
            
            print('gmod.typemodllmdkener')
            print(gmod.typemodllmdkener)

            if gmod.typemodllmdkener == 'cons':
                setp_para(gdat, strgmodl, 'coeflmdklinr', 0., 1., None)
                setp_para(gdat, strgmodl, 'coeflmdkquad', 0., 1., None)
            elif gdat.numbener[p] > 1 and gdat.fitt.typemodlenerfitt == 'full':
                for e in gdat.indxener[p]:
                    #setattr(gmod, 'coeflmdklinr' + gdat.liststrgener[p][e], 0.2)
                    #setattr(gmod, 'coeflmdkquad' + gdat.liststrgener[p][e], 0.4)
                    setp_para(gdat, strgmodl, 'coeflmdklinr', 0., 1., None, strgener=gdat.liststrgener[p][e])
            else:
                
                if gmod.typemodllmdkener == 'cons':
                    raise Exception('')
                #or gmod.typemodllmdkener == 'ener':
                #    
                #    strgener = gdat.liststrgdatafittiter[h]
                #    
                #    if gmod.typemodllmdkterm != 'none':
                #        # add linear coefficient
                #        setp_para(gdat, strgmodl, 'coeflmdklinr%s' % strgener, 0., 0.15, '$u_{1,%d}$' % e)
                #        
                #    if gmod.typemodllmdkterm == 'quad':
                #        # add quadratic coefficient
                #        setp_para(gdat, strgmodl, 'coeflmdkquad%s' % strgener, 0., 0.3, '$u_{2,%d}$' % e)
                    pass
                elif gmod.typemodllmdkener == 'line':
                    if gmod.typemodllmdkterm != 'none':
                        setp_para(gdat, strgmodl, 'ratecoeflmdklinr', 0., 1., None)
                    
                    if gmod.typemodllmdkterm == 'quad':
                        setp_para(gdat, strgmodl, 'ratecoeflmdkquad', 0., 1., None)
                else:
                    raise Exception('')
            
        for j in gmod.indxcomp:
            
            if gdat.booldiag:
                if j > 100:
                    print('')
                    print('')
                    print('')
                    raise Exception('Too many components.')

            # define parameter limits
            if gmod.typemodl == 'CompactObjectStellarCompanion':
                setp_para(gdat, strgmodl, 'radistar', 0.1, 100., ['$R_*$', ''])
                setp_para(gdat, strgmodl, 'massstar', 0.1, 100., ['$M_*$', ''])
            
            strgcomp = 'com%d' % j
            
            if strgmodl == 'true' and (gdat.true.boolsampsystnico or 'pericomp' in gdat.dicttrue):
                # draw system parameters from a realistic population model using nicomedia
                minmrsma = None
                maxmrsma = None
                
                minmperi = None
                maxmperi = None
                
                minmepocmtra = None
                maxmepocmtra = None
                
                minmcosi = None
                maxmcosi = None
                
                minmrrat = None
                maxmrrat = None
            
            else:
                # draw system parameters from simple marginal probability distributions over the parameters
                minmrsma = 0.06
                maxmrsma = 0.14
            
                minmperi = 0.99 * gmod.prio.meanpara.pericomp[j]
                maxmperi = 1.01 * gmod.prio.meanpara.pericomp[j]
                
                print('gmod.prio.meanpara.pericomp[j]')
                print(gmod.prio.meanpara.pericomp[j])

                if gdat.fitt.prio.meanpara.epocmtracomp is not None:
                    ## informed prior
                    minmepocmtra = gdat.fitt.prio.meanpara.epocmtracomp[j] - 0.2
                    maxmepocmtra = gdat.fitt.prio.meanpara.epocmtracomp[j] + 0.2
                else:
                    ## uniform over time
                    minmepocmtra = np.amin(gdat.timeconc[0])
                    maxmepocmtra = np.amax(gdat.timeconc[0])
            
                minmcosi = 0.
                maxmcosi = 0.1
            
                minmrrat = 0.11
                maxmrrat = 0.19
            
            print('minmperi')
            print(minmperi)

            setp_para(gdat, strgmodl, 'rsma', minmrsma, maxmrsma, None, strgcomp=strgcomp)
            
            setp_para(gdat, strgmodl, 'peri', minmperi, maxmperi, None, strgcomp=strgcomp)
            
            setp_para(gdat, strgmodl, 'epocmtra', minmepocmtra, maxmepocmtra, None, strgcomp=strgcomp)
            
            setp_para(gdat, strgmodl, 'cosi', minmcosi, maxmcosi, None, strgcomp=strgcomp)
            
            if gdat.numbener[p] > 1 and (strgmodl == 'true' or gdat.fitt.typemodlenerfitt == 'full'):
                for e in gdat.indxener[p]:
                    setp_para(gdat, strgmodl, 'rrat', minmrrat, maxmrrat, None, strgener=gdat.liststrgener[p][e], strgcomp=strgcomp)
            else:
                setp_para(gdat, strgmodl, 'rrat', minmrrat, maxmrrat, None, strgcomp=strgcomp)
            
            if gmod.typemodl == 'CompactObjectStellarCompanion':
                
                setp_para(gdat, strgmodl, 'mass', 0.1, 100., ['$M_c$', ''], strgcomp=strgcomp)


def exec_lspe( \
              arrylcur, \
              
              pathvisu=None, \
              
              pathdata=None, \
              
              strgextn='', \
              
              factnyqt=None, \
              
              # minimum frequency (1/days)
              minmfreq=None, \
              # maximum frequency (1/days)
              maxmfreq=None, \
              
              factosam=3., \

              # factor to scale the size of text in the figures
              factsizetextfigr=1., \

              ## file type of the plot
              typefileplot='png', \
              
              # type of verbosity
              ## -1: absolutely no text
              ##  0: no text output except critical warnings
              ##  1: minimal description of the execution
              ##  2: detailed description of the execution
              typeverb=1, \
             
             ):
    '''
    Calculate the LS periodogram of a time-series.
    '''
    
    if maxmfreq is not None and factnyqt is not None:
        raise Exception('')
    
    dictlspeoutp = dict()
    
    if pathvisu is not None:
        pathplot = pathvisu + 'LSPeriodogram_%s.%s' % (strgextn, typefileplot)

    if pathdata is not None:
        pathcsvv = pathdata + 'LSPeriodogram_%s.csv' % strgextn
    
    if pathdata is None or not os.path.exists(pathcsvv) or pathvisu is not None and not os.path.exists(pathplot):
        print('Calculating LS periodogram...')
        
        # factor by which the maximum frequency is compared to the Nyquist frequency
        if factnyqt is None:
            factnyqt = 1.
        
        time = arrylcur[:, 0]
        lcur = arrylcur[:, 1]
        numbtime = time.size
        minmtime = np.amin(time)
        maxmtime = np.amax(time)
        delttime = maxmtime - minmtime
        freqnyqt = numbtime / delttime / 2.
        
        if minmfreq is None:
            minmfreq = 1. / delttime
        
        if maxmfreq is None:
            maxmfreq = factnyqt * freqnyqt
        
        # determine the frequency sampling resolution with N samples per line
        deltfreq = minmfreq / factosam / 2.
        freq = np.arange(minmfreq, maxmfreq, deltfreq)
        peri = 1. / freq
        
        objtlspe = astropy.timeseries.LombScargle(time, lcur, nterms=1)

        powr = objtlspe.power(freq)
        
        if pathdata is not None:
            arry = np.empty((peri.size, 2))
            arry[:, 0] = peri
            arry[:, 1] = powr
            print('Writing to %s...' % pathcsvv)
            np.savetxt(pathcsvv, arry, delimiter=',')
    
    else:
        if typeverb > 0:
            print('Reading from %s...' % pathcsvv)
        arry = np.loadtxt(pathcsvv, delimiter=',')
        peri = arry[:, 0]
        powr = arry[:, 1]
    
    #listindxperipeak, _ = scipy.signal.find_peaks(powr)
    #indxperimpow = listindxperipeak[0]
    indxperimpow = np.argmax(powr)
    
    perimpow = peri[indxperimpow]
    powrmpow = powr[indxperimpow]

    if pathvisu is not None:
        if not os.path.exists(pathplot):
            
            sizefigr = np.array([7., 3.5])
            sizefigr /= factsizetextfigr

            figr, axis = plt.subplots(figsize=sizefigr)
            axis.plot(peri, powr, color='k')
            
            axis.axvline(perimpow, alpha=0.4, lw=3)
            minmxaxi = np.amin(peri)
            maxmxaxi = np.amax(peri)
            for n in range(2, 10):
                xpos = n * perimpow
                if xpos > maxmxaxi:
                    break
                axis.axvline(xpos, alpha=0.4, lw=1, linestyle='dashed')
            for n in range(2, 10):
                xpos = perimpow / n
                if xpos < minmxaxi:
                    break
                axis.axvline(xpos, alpha=0.4, lw=1, linestyle='dashed')
            
            strgtitl = 'Maximum power of %.3g at %.3f days' % (powrmpow, perimpow)
            
            listprob = [0.05]
            powrfals = objtlspe.false_alarm_level(listprob)
            for p in range(len(listprob)):
                axis.axhline(powrfals[p], ls='--')

            axis.set_xscale('log')
            axis.set_xlabel('Period [days]')
            axis.set_ylabel('Normalized Power')
            axis.set_title(strgtitl)
            print('Writing to %s...' % pathplot)
            plt.savefig(pathplot)
            plt.close()
        dictlspeoutp['pathplot'] = pathplot

    dictlspeoutp['perimpow'] = perimpow
    dictlspeoutp['powrmpow'] = powrmpow
    
    return dictlspeoutp


#@jit(nopython=True)
def srch_boxsperi_work_loop(m, phas, phasdiff, dydchalf):
    
    phasoffs = phas - phasdiff[m]
    
    if phasdiff[m] < dydchalf:
        booltemp = (phasoffs < dydchalf) | (1. - phas < dydchalf - phasoffs)
    elif 1. - phasdiff[m] < dydchalf:
        booltemp = (1. - phas - phasdiff[m] < dydchalf) | (phas < dydchalf - phasoffs)
    else:
        booltemp = np.abs(phasoffs) < dydchalf
    
    indxitra = np.where(booltemp)[0]
    
    #print('srch_boxsperi_work_loop()')
    #print('phas')
    #summgene(phas)
    #print('phasdiff[m]')
    #print(phasdiff[m])
    #print('phasoffs')
    #print(phasoffs)
    #print('dydchalf')
    #print(dydchalf)
    #print('indxitra')
    #summgene(indxitra)
    
    return indxitra


def srch_boxsperi_work(listperi, listarrytser, listdcyc, listepoc, listduratrantotllevl, boolrebn, pathvisu, i):
    
    numbperi = len(listperi[i])
    
    numblevlrebn = len(listduratrantotllevl)
    indxlevlrebn = np.arange(numblevlrebn)
    
    #conschi2 = np.sum(weig * arrytser[:, 1]**2)
    #listtermchi2 = np.empty(numbperi)
    
    rflxitraminm = np.full(numbperi, np.nan)#np.zeros(numbperi) + 1e100
    dcycmaxm = np.zeros(numbperi)
    epocmaxm = np.zeros(numbperi)
    
    listphas = [[] for b in indxlevlrebn]
    for k in tqdm(range(len(listperi[i]))):
        
        peri = listperi[i][k]
        
        if boolrebn:
            for b in indxlevlrebn:
                listphas[b] = (listarrytser[b][:, 0] % peri) / peri
        else:
            listphas = (listarrytser[0][:, 0] % peri) / peri
        
        cntr = 0
        for l in range(len(listdcyc[k])):
            
            if boolrebn:
                b = np.digitize(listdcyc[k][l] * peri * 24., listduratrantotllevl) - 1
                listphastemp = listphas[b]
                listarrytsertemp = listarrytser[b]
            else:
                listphastemp = listphas
                listarrytsertemp = listarrytser[0]
            
            if listphastemp.size == 1:
                print('listarrytser[b]')
                summgene(listarrytser[b])
                raise Exception('')

            dydchalf = listdcyc[k][l] / 2.

            phasdiff = (listepoc[k][l] % peri) / peri
            
            #print('listphas[b]')
            #summgene(listphas[b])
            #print('')
            
            for m in range(len(listepoc[k][l])):

                indxitra = srch_boxsperi_work_loop(m, listphastemp, phasdiff, dydchalf)
                
                if indxitra.size == 0:
                    continue
    
                rflxitra = np.mean(listarrytsertemp[:, 1][indxitra])
                
                #print('rflxitra')
                #print(rflxitra)
                #print('rflxitraminm[k]')
                #print(rflxitraminm[k])
                #print('')
                #print('')
                #print('')

                if cntr == 0 or rflxitra < rflxitraminm[k]:
                    rflxitraminm[k] = rflxitra
                    dcycmaxm[k] = listdcyc[k][l]
                    epocmaxm[k] = listepoc[k][l][m]
                    cntr += 1
                
                #raise Exception('')

                if not np.isfinite(rflxitra):
                    print('b')
                    print(b)
                    #print('depttrancomp')
                    #print(dept)
                    #print('np.std(rflx[indxitra])')
                    #summgene(np.std(rflx[indxitra]))
                    #print('rflx[indxitra]')
                    #summgene(rflx[indxitra])
                    raise Exception('')
                    
                #timechecloop[0][k, l, m] = modutime.time()
                #print('pericomp')
                #print(peri)
                #print('dcyc')
                #print(dcyc)
                #print('epocmtracomp')
                #print(epoc)
                #print('phasdiff')
                #summgene(phasdiff)
                #print('phasoffs')
                #summgene(phasoffs)
                
                #print('booltemp')
                #summgene(booltemp)
                #print('indxitra')
                #summgene(indxitra)
                #print('depttrancomp')
                #print(dept)
                #print('stdv')
                #print(stdv)
                #terr = np.sum(weig[indxitra])
                #ters = np.sum(weig[indxitra] * rflx[indxitra])
                #termchi2 = ters**2 / terr / (1. - terr)
                #print('ters')
                #print(ters)
                #print('terr')
                #print(terr)
                #print('depttrancomp')
                #print(dept)
                #print('indxitra')
                #summgene(indxitra)
                #print('s2nr')
                #print(s2nr)
                #print('')
                
                if pathvisu is not None:
                    for b in indxlevlrebn:
                        figr, axis = plt.subplots(2, 1, figsize=(8, 8))
                        axis[0].plot(listarrytser[b][:, 0], listarrytser[b][:, 1], color='b', ls='', marker='o', rasterized=True, ms=0.3)
                        axis[0].plot(listarrytser[b][:, 0][indxitra], listarrytser[b][:, 1][indxitra], color='firebrick', ls='', marker='o', ms=2., rasterized=True)
                        axis[0].axhline(1., ls='-.', alpha=0.3, color='k')
                        axis[0].set_xlabel('Time [BJD]')
                        
                        axis[1].plot(listphas[b], listarrytser[b][:, 1], color='b', ls='', marker='o', rasterized=True, ms=0.3)
                        axis[1].plot(listphas[b][indxitra], listarrytser[b][:, 1][indxitra], color='firebrick', ls='', marker='o', ms=2., rasterized=True)
                        axis[1].plot(np.mean(listphas[b][indxitra]), rflxitra, color='g', ls='', marker='o', ms=4., rasterized=True)
                        axis[1].axhline(1., ls='-.', alpha=0.3, color='k')
                        axis[1].set_xlabel('Phase')
                        titl = '$P$=%.3f, $T_0$=%.3f, $q_{tr}$=%.3g, $f$=%.6g' % (peri, listepoc[k][l][m], listdcyc[k][l], rflxitra)
                        axis[0].set_title(titl, usetex=False)
                        path = pathvisu + 'rflx_boxsperi_b%03d_%04d%04d.pdf' % (b, l, m)
                        print('Writing to %s...' % path)
                        plt.savefig(path, usetex=False)
                        plt.close()
        
    return rflxitraminm, dcycmaxm, epocmaxm


def srch_outlperi( \
                  # time of samples
                  time, \
                  # relative flux of samples
                  flux, \
                  # relative flux error of samples
                  stdvflux, \
                  # number of outliers to include in the search
                  numboutl=5, \
                  # Boolean flag to diagnose
                  booldiag=True, \
                 ):
    '''
    Search for periodic outliers in a computationally efficient way
    '''
    
    # indices of the outliers
    indxtimesort = np.argsort(flux)[::-1][:numboutl]
    
    # the times of the outliers
    timeoutl = time[indxtimesort]
    
    # number of differences between times of outlier samples
    numbdiff = int(numboutl * (numboutl - 1) / 2)
    
    # differences between times of outlier samples
    difftimeoutl = np.empty(numbdiff)
    
    # compute the differences between times of outlier samples
    listtemp = []
    c = 0
    indxoutl = np.arange(numboutl)
    for a in indxoutl:
        for b in indxoutl:
            if a >= b:
                continue
            listtemp.append([a, b])
            difftimeoutl[c] = abs(timeoutl[a] - timeoutl[b])
            c += 1
    
    # incides that sort the differences between times of outlier samples
    indxsort = np.argsort(difftimeoutl)
    
    # sorted differences between times of outlier samples
    difftimeoutlsort = difftimeoutl[indxsort]

    # fractional differences between differences of times of outlier samples
    frddtimeoutlsort = (difftimeoutlsort[1:] - difftimeoutlsort[:-1]) / ((difftimeoutlsort[1:] + difftimeoutlsort[:-1]) / 2.)

    # index of the minimum fractional difference between differences of times of outlier samples
    indxfrddtimeoutlsort = np.argmin(frddtimeoutlsort)
    
    # minimum fractional difference between differences of times of outlier samples
    minmfrddtimeoutlsort = frddtimeoutlsort[indxfrddtimeoutlsort]
    
    # estimate of the epoch
    epoccomp = timeoutl[0]
    
    # estimate of the period
    pericomp = difftimeoutlsort[indxfrddtimeoutlsort]
    
    # output dictionary
    dictoutp = dict()
    
    # populate the output dictionary
    if minmfrddtimeoutlsort < 0.1:
        dictoutp['boolposi'] = True
        dictoutp['peri'] = [pericomp]
        dictoutp['epoc'] = [epoccomp]
    else:
        dictoutp['boolposi'] = False
    dictoutp['minmfrddtimeoutlsort'] = [minmfrddtimeoutlsort]
    dictoutp['timeoutl'] = timeoutl 

    return dictoutp


def srch_boxsperi(arry, \
              
              # Boolean flag to search for positive boxes
              boolsrchposi=False, \

              ### maximum number of transiting objects
              maxmnumbboxsperi=1, \
              
              ticitarg=None, \
              
              dicttlsqinpt=None, \
              
              # temp: move this out of srch_boxsperi
              typecalc='native', \
              
              # minimum period
              minmperi=None, \

              # maximum period
              maxmperi=None, \

              # oversampling factor (wrt to transit duration) when rebinning data to decrease the time resolution
              factduracade=2., \

              # factor by which to oversample the frequency grid
              factosam=10., \
              
              # differential logarithm of duty cycle
              deltlogtdcyc=0.1, \
              
              # density of the star
              densstar=None, \
              
              # size of the kernel that will be used to median-detrend the signal spectrum and estimate noise inside a window
              sizekern = 51, \

              # epoc steps divided by trial duration
              factdeltepocdura=0.5, \

              # detection threshold
              thrss2nr=7.1, \
              
              # number of processes
              numbproc=None, \
              
              # Boolean flag to enable multiprocessing
              boolprocmult=True, \
              
              # string extension to output files
              strgextn='', \
              
              # path where the output data will be stored
              pathdata=None, \
              
              # Boolean flag to rebin the time-series
              boolchecrebn=True, \

              # plotting
              ## path where the output visuals will be written
              pathvisu=None, \
              ## file type of the plot
              typefileplot='png', \
              ## figure size
              figrsizeydobskin=(8, 2.5), \
              ## time offset
              timeoffs=0, \
              ## data transparency
              alphdata=0.2, \
              
              # type of verbosity
              ## -1: absolutely no text
              ##  0: no text output except critical warnings
              ##  1: minimal description of the execution
              ##  2: detailed description of the execution
              typeverb=1, \
              
              # type of plot background
              typeplotback='black', \

              # Boolean flag to turn on diagnostic mode
              ## diagnostic mode is always on by default, which should be turned off during large-scale runs, where speed is a concern
              booldiag=True, \

              # Boolean flag to force rerun and overwrite previous data and plots 
              boolover=True, \

             ):
    '''
    Search for periodic boxes in time-series data.
    '''
    
    boolproc = False
    listnameplot = ['ampl', 'sgnl', 'stdvsgnl', 's2nr', 'rflx', 'pcur']
    if pathdata is None:
        boolproc = True
    else:
        if strgextn == '':
            pathsave = pathdata + 'boxsperi.csv'
        else:
            pathsave = pathdata + 'boxsperi_%s.csv' % strgextn
        if not os.path.exists(pathsave):
            boolproc = True
        
        dictpathplot = dict()
        for strg in listnameplot:
            dictpathplot[strg] = []
            
        if os.path.exists(pathsave):
            if typeverb > 0:
                print('Reading from %s...' % pathsave)
            
            dictboxsperioutp = pd.read_csv(pathsave).to_dict(orient='list')
            for name in dictboxsperioutp.keys():
                dictboxsperioutp[name] = np.array(dictboxsperioutp[name])
                if len(dictboxsperioutp[name]) == 0:
                    dictboxsperioutp[name] = np.array([])
            
            if not pathvisu is None:
                for strg in listnameplot:
                    for j in range(len(dictboxsperioutp['peri'])):
                        dictpathplot[strg].append(pathvisu + strg + '_boxsperi_tce%d_%s.%s' % (j, strgextn, typefileplot))
         
                        if not os.path.exists(dictpathplot[strg][j]):
                            boolproc = True
            
    if typeplotback == 'white':
        colrbkgd = 'white'
        colrdraw = 'black'
    elif typeplotback == 'black':
        colrbkgd = 'black'
        colrdraw = 'white'
    
    if boolproc:
        dictboxsperioutp = dict()
        if pathvisu is not None:
            for name in listnameplot:
                dictboxsperioutp['listpathplot%s' % name] = []
    
        print('Searching for periodic boxes in time-series data...')
        
        print('factosam')
        print(factosam)
        if typecalc == 'TLS':
            import transitleastsquares
            if dicttlsqinpt is None:
                dicttlsqinpt = dict()
        
        # setup TLS
        # temp
        #ab, mass, mass_min, mass_max, radius, radius_min, radius_max = transitleastsquares.catalog_info(TIC_ID=int(ticitarg))
        
        dictboxsperiinte = dict()
        liststrgvarbsave = ['peri', 'epoc', 'ampl', 'dura', 's2nr']
        for strg in liststrgvarbsave:
            dictboxsperioutp[strg] = []
        
        if booldiag:
            if (abs(arry[:, 1]) > 1e10).any():
                print('Warning! There is almost certainly something wrong with the time-series data.')
                raise Exception('')
                
        arrysrch = np.copy(arry)
        if boolsrchposi:
            arrysrch[:, 1] = 2. - arrysrch[:, 1]

        j = 0
        
        timeinit = modutime.time()

        dictfact = tdpy.retr_factconv()
        
        numbtime = arrysrch[:, 0].size
        print('Number of data points: %d...' % numbtime)
        minmdcyc = 2. / numbtime
        print('Minimum duty cycle achievable with this number of data point: %g' % minmdcyc)
        minmtime = np.amin(arrysrch[:, 0])
        maxmtime = np.amax(arrysrch[:, 0])
        
        difftime = arrysrch[1:, 0] - arrysrch[:-1, 0]
        #arrysrch[:, 0] -= minmtime

        minmdifftime = np.amin(difftime)
        
        print('Initial:')
        print('minmperi')
        print(minmperi)
        print('maxmperi')
        print(maxmperi)
        
        if maxmperi is None:
            # maximum period will stretch out to the baseline
            minmfreq = 1. / (maxmtime - minmtime)
        else:
            minmfreq = 1. / maxmperi

        if minmperi is None:
            maxmfreq = 0.5 / minmdifftime
        else:
            maxmfreq = 1. / minmperi

        deltfreq = minmfreq / factosam
        
        listfreq = np.arange(minmfreq, maxmfreq, deltfreq)
        listperi = 1. / listfreq
        
        # cadence
        cade = minmdifftime * 24. * 3600. # [seconds]
        print('Cadence: %g [seconds]' % cade)
        
        if pathvisu is not None:
            numbtimeplot = 100000
            timemodlplot = np.linspace(minmtime, maxmtime, numbtimeplot)
        
        numbperi = listperi.size
        if numbperi < 3:
            print('maxmperi')
            print(maxmperi)
            print('minmperi')
            print(minmperi)
            print('numbperi')
            print(numbperi)
            raise Exception('')

        indxperi = np.arange(numbperi)
        minmperi = np.amin(listperi)
        maxmperi = np.amax(listperi)
        print('minmperi')
        print(minmperi)
        print('maxmperi')
        print(maxmperi)
        
        listdcyc = [[] for k in indxperi]
        listperilogt = np.log10(listperi)
        
        if deltlogtdcyc is None:
            deltlogtdcyc = np.log10(2.)
        
        # assuming Solar density
        maxmdcyclogt = -2. / 3. * listperilogt - 1. + deltlogtdcyc
        if densstar is not None:
            maxmdcyclogt += -1. / 3. * np.log10(densstar)
        
        listduratrantotl = []
        for k in indxperi:
            minmdcyclogt = max(np.log10(minmdcyc), maxmdcyclogt[k] - 3. * deltlogtdcyc)
            if maxmdcyclogt[k] >= minmdcyclogt:
                listdcyc[k] = np.logspace(minmdcyclogt, maxmdcyclogt[k], 2 + int((maxmdcyclogt[k] - minmdcyclogt) / deltlogtdcyc))
                listduratrantotl.append(listdcyc[k] * listperi[k])
        listduratrantotl = np.concatenate(listduratrantotl)
        
        if booldiag:
            if listduratrantotl.size == 0:
                raise Exception('')

        print('Trial transit duty cycles at the smallest period')
        print(listdcyc[-1])
        if len(listdcyc[-1]) > 0:
            print('Trial transit durations at the smallest period [hr]')
            print(listdcyc[-1] * listperi[-1] * 24)
        print('Trial transit duty cycles at the largest period')
        print(listdcyc[0])
        if len(listdcyc[0]) > 0:
            print('Trial transit durations at the largest period [hr]')
            print(listdcyc[0] * listperi[0] * 24)

        meancade = np.mean(difftime) * 24. * 3600. # [seconds]
        print('Average cadence: %g [seconds]' % meancade)
        
        if cade < 0:
            print('')
            print('')
            print('')
            raise Exception('The time array is not sorted.')
        
        # minimum transit duration
        #minmduratrantotl = listdcyc[-1][0] * listperi[-1] * 24
        minmduratrantotl = np.amin(listduratrantotl)

        # maximum transit duration
        #maxmduratrantotl = listdcyc[0][-1] * listperi[0] * 24
        maxmduratrantotl = np.amax(listduratrantotl)
        
        #if minmduratrantotl < factduracade * cade:
        #    print('')
        #    print('')
        #    print('')
        #    print('minmduratrantotl [hr]')
        #    print(minmduratrantotl)
        #    print('factduracade')
        #    print(factduracade)
        #    print('cade [hr]')
        #    print(cade)
        #    print('Warnin: either the minimum transit duration is too small or the cadence is too large.')
        
        listarrysrch = [arrysrch]
        
        print('arrysrch[:, 0]')
        summgene(arrysrch[:, 0])
        print('arrysrch[:, 1]')
        summgene(arrysrch[:, 1])
        print('arrysrch[:, 2]')
        summgene(arrysrch[:, 2])

        # Boolean flag to rebin the time-series
        boolrebn = boolchecrebn and meancade < 0.5 * minmduratrantotl
        
        if boolrebn:
            numblevlrebn = 10
            indxlevlrebn = np.arange(numblevlrebn)
            # list of transit durations when rebinned data sets will be used
            listduratrantotllevl = np.linspace(minmduratrantotl, maxmduratrantotl, numblevlrebn)
            
            print('factduracade')
            print(factduracade)
            for b in indxlevlrebn:
                print('Transit duration for the level: %g [hour]' % listduratrantotllevl[b])
                delt = listduratrantotllevl[b] / 24. / factduracade
                arryrebn = rebn_tser(arrysrch, delt=delt)
                print('Number of data points in rebinned to a delta time of %g [min]: %d' % (delt * 24. * 60., arryrebn.shape[0]))
                indx = np.where(np.isfinite(arryrebn[:, 1]))[0]
                arryrebn = arryrebn[indx, :]
                print('Number of finite data points in rebinned to a delta time of %g [min]: %d' % (delt * 24. * 60., arryrebn.shape[0]))
                print('')
                listarrysrch.append(arryrebn)
        else:
            print('Not rebinning the time-series...')
            listduratrantotllevl = []
            #numblevlrebn = 1
            indxlevlrebn = np.arange(1)

        listepoc = [[[] for l in range(len(listdcyc[k]))] for k in indxperi]
        numbtria = np.zeros(numbperi, dtype=int)
        for k in indxperi:
            if len(listdcyc[k]) > 0:
                for l in range(len(listdcyc[k])):
                    diffepoc = max(cade / 24., factdeltepocdura * listperi[k] * listdcyc[k][l])
                    listepoc[k][l] = np.arange(minmtime, minmtime + listperi[k], diffepoc)
                    numbtria[k] += len(listepoc[k][l])
                
        dflx = arrysrch[:, 1] - 1.
        stdvdflx = arrysrch[:, 2]
        varidflx = stdvdflx**2
        
        print('Number of trial periods: %d...' % numbperi)
        print('Number of trial computations for the smallest period: %d...' % numbtria[-1])
        print('Number of trial computations for the largest period: %d...' % numbtria[0])
        print('Total number of trial computations: %d...' % np.sum(numbtria))

        while True:
            
            if maxmnumbboxsperi is not None and j >= maxmnumbboxsperi:
                break
            
            # mask out the detected transit
            if j > 0:
                ## remove previously detected periodic box from the rebinned data
                pericomp = [dictboxsperioutp['peri'][j]]
                epocmtracomp = [dictboxsperioutp['epoc'][j]]
                radicomp = [dictfact['rsre'] * np.sqrt(dictboxsperioutp['depttrancomp'][j] * 1e-3)]
                cosicomp = [0]
                rsmacomp = [nicomedia.retr_rsmacomp(dictboxsperioutp['pericomp'][j], dictboxsperioutp['duracomp'][j], cosicomp[0])]
                    
                for b in indxlevlrebn:
                    ## evaluate model at all resolutions
                    dictoutp = eval_modl(listarrysrch[b][:, 0], 'PlanetarySystem', pericomp=pericomp, epocmtracomp=epocmtracomp, \
                                                                                        rsmacomp=rsmacomp, cosicomp=cosicomp, rratcomp=rratcomp)
                    ## subtract it from data
                    listarrysrch[b][:, 1] -= (dictoutp['rflx'][b] - 1.)
                
                    if (dictboxsperiinte['rflx'][b] == 1.).all():
                        raise Exception('')

            if typecalc == 'TLS':
                objtmodltlsq = transitleastsquares.transitleastsquares(arrysrch[:, 0], lcurboxsperimeta)
                objtresu = objtmodltlsq.power(\
                                              # temp
                                              #u=ab, \
                                              **dicttlsqinpt, \
                                              #use_threads=1, \
                                             )

                dictboxsperi = dict()
                dictboxsperiinte['listperi'] = objtresu.periods
                dictboxsperiinte['lists2nr'] = objtresu.power
                
                dictboxsperioutp['peri'].append(objtresu.period)
                dictboxsperioutp['epoc'].append(objtresu.T0)
                dictboxsperioutp['dura'].append(objtresu.duration)
                dictboxsperioutp['ampl'].append(-objtresu.depth * 1e3)
                dictboxsperioutp['s2nr'].append(objtresu.SDE)
                dictboxsperioutp['prfp'].append(objtresu.FAP)
                
                if objtresu.SDE < thrss2nr:
                    break
                
                dictboxsperiinte['rflxtsermodl'] = objtresu.model_lightcurve_model
                
                if pathvisu is not None:
                    dictboxsperiinte['listtimetran'] = objtresu.transit_times
                    dictboxsperiinte['timemodl'] = objtresu.model_lightcurve_time
                    dictboxsperiinte['phasmodl'] = objtresu.model_folded_phase
                    dictboxsperiinte['rflxpsermodl'] = objtresu.model_folded_model
                    dictboxsperiinte['phasdata'] = objtresu.folded_phase
                    dictboxsperiinte['rflxpserdata'] = objtresu.folded_y

            elif typecalc == 'astropy':

                model = astropy.timeseries.BoxLeastSquares(arrysrch[:, 0] * astropy.units.day, lcurboxsperimeta)#, dy=)
                periodogram = model.autopower(0.2)
                plt.plot(periodogram.period, periodogram.power)
                model = BoxLeastSquares(t * u.day, y, dy=0.01)
                periodogram = model.autopower(0.2, objective="snr")


            elif typecalc == 'native':
                
                if boolprocmult:
                    
                    import multiprocessing

                    if numbproc is None:
                        #numbproc = multiprocessing.cpu_count() - 1
                        numbproc = int(0.8 * multiprocessing.cpu_count())
                    
                    print('Generating %d processes...' % numbproc)
                    
                    objtpool = multiprocessing.Pool(numbproc)
                    numbproc = objtpool._processes
                    indxproc = np.arange(numbproc)

                    listperiproc = [[] for i in indxproc]
                    
                    binsperiproc = tdpy.icdf_powr(np.linspace(0., 1., numbproc + 1)[1:-1], np.amin(listperi), np.amax(listperi), 1.97)
                    binsperiproc = np.concatenate((np.array([-np.inf]), binsperiproc, np.array([np.inf])))
                    indxprocperi = np.digitize(listperi, binsperiproc, right=False) - 1
                    for i in indxproc:
                        indx = np.where(indxprocperi == i)[0]
                        listperiproc[i] = listperi[indx]
                    data = objtpool.map(partial(srch_boxsperi_work, listperiproc, listarrysrch, listdcyc, listepoc, listduratrantotllevl, boolrebn), indxproc)
                    listrflxitra = np.concatenate([data[k][0] for k in indxproc])
                    listamplmaxm = np.concatenate([data[k][1] for k in indxproc])
                    listdcycmaxm = np.concatenate([data[k][2] for k in indxproc])
                    listepocmaxm = np.concatenate([data[k][3] for k in indxproc])
                else:
                    print('Using a single process for the periodic box search...')
                    listrflxitra, listdcycmaxm, listepocmaxm = srch_boxsperi_work([listperi], listarrysrch, listdcyc, listepoc, listduratrantotllevl, boolrebn, pathvisu, 0)
                
                if booldiag:
                    if (~np.isfinite(listrflxitra)).all():
                        print('')
                        print('')
                        print('')
                        print('listrflxitra')
                        summgene(listrflxitra)
                        print('listrflxitra')
                        summgene(listrflxitra)
                        print('listarrysrch[0][:, 1]')
                        summgene(listarrysrch[0][:, 1])
                        raise Exception('')

                listampl = (np.median(listarrysrch[0][:, 1]) - listrflxitra) * 1e3 # [ppt])
                
                listsgnl = listampl - scipy.ndimage.median_filter(listampl, size=sizekern)
                
                liststdvsgnl = retr_stdvwind(listsgnl, sizekern, boolcuttpeak=True)
                
                lists2nr = np.zeros_like(listsgnl)
                indxperigood = np.where(liststdvsgnl > 0)
                lists2nr[indxperigood] = listsgnl[indxperigood] / liststdvsgnl[indxperigood]
                
                indxperimpow = np.nanargmax(lists2nr)
                
                s2nr = lists2nr[indxperimpow]
                
                if not np.isfinite(s2nr) or listampl[indxperimpow] < 0:
                    print('')
                    print('')
                    print('')
                    for b in indxlevlrebn:
                        print('listarrysrch[b]')
                        summgene(listarrysrch[b])
                    print('listampl')
                    summgene(listampl)
                    print('listampl[indxperimpow]')
                    print(listampl[indxperimpow])
                    print('liststdvsgnl')
                    summgene(liststdvsgnl)
                    print('listsgnl')
                    summgene(listsgnl)
                    print('lists2nr')
                    summgene(lists2nr)
                    print('indxperimpow')
                    print(indxperimpow)
                    print('lists2nr[indxperimpow]')
                    print(lists2nr[indxperimpow])
                    print('listsgnl[indxperimpow]')
                    print(listsgnl[indxperimpow])
                    print('liststdvsgnl[indxperimpow]')
                    print(liststdvsgnl[indxperimpow])
                    print('s2nr')
                    print(s2nr)
                    raise Exception('SNR is infinite or listampl[indxperimpow] < 0')

                dictboxsperioutp['s2nr'].append(s2nr)
                dictboxsperioutp['peri'].append(listperi[indxperimpow])
                dictboxsperioutp['dura'].append(24. * listdcycmaxm[indxperimpow] * listperi[indxperimpow]) # [hours]
                dictboxsperioutp['epoc'].append(listepocmaxm[indxperimpow])
                dictboxsperioutp['ampl'].append(listampl[indxperimpow])
                
                # best-fit orbit
                dictboxsperiinte['listperi'] = listperi
                
                print('temp: assuming power is SNR')
                dictboxsperiinte['listampl'] = listampl
                dictboxsperiinte['listsgnl'] = listsgnl
                dictboxsperiinte['liststdvsgnl'] = liststdvsgnl
                dictboxsperiinte['lists2nr'] = lists2nr
                
                if pathvisu is not None:
                    for strg in listnameplot:
                        for j in range(len(dictboxsperioutp['peri'])):
                            pathplot = pathvisu + strg + '_boxsperi_tce%d_%s.%s' % (j, strgextn, typefileplot)
                            dictpathplot[strg].append(pathplot)
            
                    pericomp = [dictboxsperioutp['peri'][j]]
                    epocmtracomp = [dictboxsperioutp['epoc'][j]]
                    cosicomp = [0]
                    rsmacomp = [nicomedia.retr_rsmacomp(dictboxsperioutp['peri'][j], dictboxsperioutp['dura'][j], cosicomp[0])]
                    rratcomp = [np.sqrt(dictboxsperioutp['ampl'][j] * 1e-3)]
                    
                    if booldiag:
                        if not np.isfinite(rratcomp).all():
                            print('')
                            print('')
                            print('')
                            print('listampl')
                            summgene(listampl)
                            print('lists2nr')
                            summgene(lists2nr)
                            print('indxperimpow')
                            print(indxperimpow)
                            print('listampl[indxperimpow]')
                            print(listampl[indxperimpow])
                            print('s2nr')
                            print(s2nr)
                            print('rratcomp')
                            print(rratcomp)
                            raise Exception('rratcomp is not finite.')

                    dictoutp = ephesos.eval_modl(timemodlplot, typesyst='PlanetarySystem', pericomp=pericomp, epocmtracomp=epocmtracomp, \
                                                                                                        rsmacomp=rsmacomp, cosicomp=cosicomp, rratcomp=rratcomp)
                    dictboxsperiinte['rflxtsermodl'] = dictoutp['rflx'][:, 0]
                    
                    arrymetamodl = np.zeros((numbtimeplot, 3))
                    arrymetamodl[:, 0] = timemodlplot
                    arrymetamodl[:, 1] = dictboxsperiinte['rflxtsermodl']
                    arrypsermodl = fold_tser(arrymetamodl, dictboxsperioutp['epoc'][j], dictboxsperioutp['peri'][j], phascntr=0.5)
                    arrypserdata = fold_tser(listarrysrch[0], dictboxsperioutp['epoc'][j], dictboxsperioutp['peri'][j], phascntr=0.5)
                        
                    dictboxsperiinte['timedata'] = listarrysrch[0][:, 0]
                    dictboxsperiinte['rflxtserdata'] = listarrysrch[0][:, 1]
                    dictboxsperiinte['phasdata'] = arrypserdata[:, 0]
                    dictboxsperiinte['rflxpserdata'] = arrypserdata[:, 1]

                    dictboxsperiinte['timemodl'] = arrymetamodl[:, 0]
                    dictboxsperiinte['phasmodl'] = arrypsermodl[:, 0]
                    dictboxsperiinte['rflxpsermodl'] = arrypsermodl[:, 1]
                
                    print('boolsrchposi')
                    print(boolsrchposi)
                    if boolsrchposi:
                        dictboxsperiinte['rflxpsermodl'] = 2. - dictboxsperiinte['rflxpsermodl']
                        dictboxsperiinte['rflxtsermodl'] = 2. - dictboxsperiinte['rflxtsermodl']
                        dictboxsperiinte['rflxpserdata'] = 2. - dictboxsperiinte['rflxpserdata']
           
            if pathvisu is not None:
                strgtitl = 'P=%.3f d, $T_0$=%.3f, Dep=%.3g ppt, Dur=%.2g hr, SDE=%.3g' % \
                            (dictboxsperioutp['peri'][j], dictboxsperioutp['epoc'][j], dictboxsperioutp['ampl'][j], \
                            dictboxsperioutp['dura'][j], dictboxsperioutp['s2nr'][j])
                
                # plot spectra
                for a in range(4):
                    if a == 0:
                        strg = 'ampl'
                    if a == 1:
                        strg = 'sgnl'
                    if a == 2:
                        strg = 'stdvsgnl'
                    if a == 3:
                        strg = 's2nr'

                    figr, axis = plt.subplots(figsize=figrsizeydobskin)
                    
                    axis.axvline(dictboxsperioutp['peri'][j], alpha=0.4, lw=3)
                    minmxaxi = np.amin(dictboxsperiinte['listperi'])
                    maxmxaxi = np.amax(dictboxsperiinte['listperi'])
                    for n in range(2, 10):
                        xpos = n * dictboxsperioutp['peri'][j]
                        if xpos > maxmxaxi:
                            break
                        axis.axvline(xpos, alpha=0.4, lw=1, linestyle='dashed')
                    for n in range(2, 10):
                        xpos = dictboxsperioutp['peri'][j] / n
                        if xpos < minmxaxi:
                            break
                        axis.axvline(xpos, alpha=0.4, lw=1, linestyle='dashed')
                    
                    axis.set_ylabel('Power')
                    axis.set_xlabel('Period [days]')
                    axis.set_xscale('log')
                    axis.plot(dictboxsperiinte['listperi'], dictboxsperiinte['list' + strg], color=colrdraw, lw=0.5)
                    axis.set_title(strgtitl)
                    plt.subplots_adjust(bottom=0.2)
                    path = dictpathplot[strg][j]
                    dictboxsperioutp['listpathplot%s' % strg].append(path)
                    print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()
                
                # plot data and model time-series
                figr, axis = plt.subplots(figsize=figrsizeydobskin)
                lcurboxsperimeta = listarrysrch[0][:, 1]
                if boolsrchposi:
                    lcurboxsperimetatemp = 2. - lcurboxsperimeta
                else:
                    lcurboxsperimetatemp = lcurboxsperimeta
                axis.plot(listarrysrch[0][:, 0] - timeoffs, lcurboxsperimetatemp, alpha=alphdata, marker='o', ms=1, ls='', color='gray')
                axis.plot(dictboxsperiinte['timemodl'] - timeoffs, dictboxsperiinte['rflxtsermodl'], color='b')
                if timeoffs == 0:
                    axis.set_xlabel('Time [days]')
                else:
                    axis.set_xlabel('Time [BJD-%d]' % timeoffs)
                axis.set_ylabel('Relative flux');
                if j == 0:
                    ylimtserinit = axis.get_ylim()
                else:
                    axis.set_ylim(ylimtserinit)
                axis.set_title(strgtitl)
                plt.subplots_adjust(bottom=0.2)
                path = dictpathplot['rflx'][j]
                dictboxsperioutp['listpathplotrflx'].append(path)
                print('Writing to %s...' % path)
                plt.savefig(path, dpi=200)
                plt.close()

                # plot data and model phase-series
                figr, axis = plt.subplots(figsize=figrsizeydobskin)
                axis.plot(dictboxsperiinte['phasdata'], dictboxsperiinte['rflxpserdata'], marker='o', ms=1, ls='', alpha=alphdata, color='gray')
                axis.plot(dictboxsperiinte['phasmodl'], dictboxsperiinte['rflxpsermodl'], color='b')
                axis.set_xlabel('Phase')
                axis.set_ylabel('Relative flux');
                if j == 0:
                    ylimpserinit = axis.get_ylim()
                else:
                    axis.set_ylim(ylimpserinit)
                axis.set_title(strgtitl)
                plt.subplots_adjust(bottom=0.2)
                path = dictpathplot['pcur'][j]
                dictboxsperioutp['listpathplotpcur'].append(path)
                print('Writing to %s...' % path)
                plt.savefig(path, dpi=200)
                plt.close()
            
            j += 1
        
            if s2nr < thrss2nr or indxperimpow == lists2nr.size - 1:
                break
        
        # make the BLS features arrays
        for name in dictboxsperioutp.keys():
            dictboxsperioutp[name] = np.array(dictboxsperioutp[name])
        
        pd.DataFrame.from_dict(dictboxsperioutp).to_csv(pathsave, index=False)
                
        timefinl = modutime.time()
        timetotl = timefinl - timeinit
        timeredu = timetotl / numbtime / np.sum(numbtria)
        
        print('srch_boxsperi() took %.3g seconds in total and %g ns per observation and trial.' % (timetotl, timeredu * 1e9))

    return dictboxsperioutp


def anim_tmptdete(timefull, lcurfull, meantimetmpt, lcurtmpt, pathvisu, listindxtimeposimaxm, corrprod, corr, strgextn='', \
                  ## file type of the plot
                  typefileplot='png', \
                  colr=None):
    
    numbtimefull = timefull.size
    numbtimekern = lcurtmpt.size
    numbtimefullruns = numbtimefull - numbtimekern
    indxtimefullruns = np.arange(numbtimefullruns)
    
    listpath = []
    gdat.cmndmakeanim = 'convert -delay 20'
    
    numbtimeanim = min(200, numbtimefullruns)
    indxtimefullrunsanim = np.random.choice(indxtimefullruns, size=numbtimeanim, replace=False)
    indxtimefullrunsanim = np.sort(indxtimefullrunsanim)

    for tt in indxtimefullrunsanim:
        
        path = pathvisu + 'lcur%s_%08d.%s' % (strgextn, tt, typefileplot)
        listpath.append(path)
        if not os.path.exists(path):
            plot_tmptdete(timefull, lcurfull, tt, meantimetmpt, lcurtmpt, path, listindxtimeposimaxm, corrprod, corr)
        gdat.cmndmakeanim += ' %s' % path
    
    pathanim = pathvisu + 'lcur%s.gif' % strgextn
    gdat.cmndmakeanim += ' %s' % pathanim
    print('gdat.cmndmakeanim')
    print(gdat.cmndmakeanim)
    os.system(gdat.cmndmakeanim)
    gdat.cmnddeleimag = 'rm'
    for path in listpath:
        gdat.cmnddeleimag += ' ' + path
    os.system(gdat.cmnddeleimag)


def plot_tmptdete(timefull, lcurfull, tt, meantimetmpt, lcurtmpt, path, listindxtimeposimaxm, corrprod, corr):
    
    numbtimekern = lcurtmpt.size
    indxtimekern = np.arange(numbtimekern)
    numbtimefull = lcurfull.size
    numbtimefullruns = numbtimefull - numbtimekern
    indxtimefullruns = np.arange(numbtimefullruns)
    difftime = timefull[1] - timefull[0]
    
    figr, axis = plt.subplots(5, 1, figsize=(8, 11))
    
    # plot the whole light curve
    proc_axiscorr(timefull, lcurfull, axis[0], listindxtimeposimaxm)
    
    # plot zoomed-in light curve
    minmindx = max(0, tt - int(numbtimekern / 4))
    maxmindx = min(numbtimefullruns - 1, tt + int(5. * numbtimekern / 4))
    indxtime = np.arange(minmindx, maxmindx + 1)
    print('indxtime')
    summgene(indxtime)
    proc_axiscorr(timefull, lcurfull, axis[1], listindxtimeposimaxm, indxtime=indxtime)
    
    # plot template
    axis[2].plot(timefull[0] + meantimetmpt + tt * difftime, lcurtmpt, color='b', marker='v')
    axis[2].set_ylabel('Template')
    axis[2].set_xlim(axis[1].get_xlim())

    # plot correlation
    axis[3].plot(timefull[0] + meantimetmpt + tt * difftime, corrprod[tt, :], color='red', marker='o')
    axis[3].set_ylabel('Correlation')
    axis[3].set_xlim(axis[1].get_xlim())
    
    # plot the whole total correlation
    print('indxtimefullruns')
    summgene(indxtimefullruns)
    print('timefull')
    summgene(timefull)
    print('corr')
    summgene(corr)
    axis[4].plot(timefull[indxtimefullruns], corr, color='m', marker='o', ms=1, rasterized=True)
    axis[4].set_ylabel('Total correlation')
    
    titl = 'C = %.3g' % corr[tt]
    axis[0].set_title(titl)

    limtydat = axis[0].get_ylim()
    axis[0].fill_between(timefull[indxtimekern+tt], limtydat[0], limtydat[1], alpha=0.4)
    print('Writing to %s...' % path)
    plt.savefig(path)
    plt.close()
    

def proc_axiscorr(time, lcur, axis, listindxtimeposimaxm, indxtime=None, colr='k', timeoffs=2457000):
    
    if indxtime is None:
        indxtimetemp = np.arange(time.size)
    else:
        indxtimetemp = indxtime
    axis.plot(time[indxtimetemp], lcur[indxtimetemp], ls='', marker='o', color=colr, rasterized=True, ms=0.5)
    maxmydat = axis.get_ylim()[1]
    for kk in range(len(listindxtimeposimaxm)):
        if listindxtimeposimaxm[kk] in indxtimetemp:
            axis.plot(time[listindxtimeposimaxm[kk]], maxmydat, marker='v', color='b')
    #print('timeoffs')
    #print(timeoffs)
    #axis.set_xlabel('Time [BJD-%d]' % timeoffs)
    axis.set_ylabel('Relative flux')
    

def srch_flar(time, lcur, \
              # type of verbosity
              ## -1: absolutely no text
              ##  0: no text output except critical warnings
              ##  1: minimal description of the execution
              ##  2: detailed description of the execution
              typeverb=1, \

              strgextn='', numbkern=3, minmscalfalltmpt=None, maxmscalfalltmpt=None, \
                                                                    pathvisu=None, boolplot=True, boolanim=False, thrs=None):

    minmtime = np.amin(time)
    timeflartmpt = 0.
    amplflartmpt = 1.
    scalrisetmpt = 0. / 24.
    difftime = np.amin(time[1:] - time[:-1])
    
    if minmscalfalltmpt is None:
        minmscalfalltmpt = 3 * difftime
    
    if maxmscalfalltmpt is None:
        maxmscalfalltmpt = 3. / 24.
    
    if typeverb > 1:
        print('lcurtmpt')
        summgene(lcurtmpt)
    
    indxscalfall = np.arange(numbkern)
    listscalfalltmpt = np.linspace(minmscalfalltmpt, maxmscalfalltmpt, numbkern)
    print('listscalfalltmpt')
    print(listscalfalltmpt)
    listcorr = []
    listlcurtmpt = [[] for k in indxscalfall]
    meantimetmpt = [[] for k in indxscalfall]
    for k in indxscalfall:
        numbtimekern = 3 * int(listscalfalltmpt[k] / difftime)
        print('numbtimekern')
        print(numbtimekern)
        meantimetmpt[k] = np.arange(numbtimekern) * difftime
        print('meantimetmpt[k]')
        summgene(meantimetmpt[k])
        if numbtimekern == 0:
            raise Exception('')
        listlcurtmpt[k] = retr_lcurmodl_flarsing(meantimetmpt[k], timeflartmpt, amplflartmpt, scalrisetmpt, listscalfalltmpt[k])
        if not np.isfinite(listlcurtmpt[k]).all():
            raise Exception('')
        
    corr, listindxtimeposimaxm, timefull, lcurfull = corr_tmpt(time, lcur, meantimetmpt, listlcurtmpt, thrs=thrs, boolanim=boolanim, boolplot=boolplot, \
                                                                                            typeverb=typeverb, strgextn=strgextn, pathvisu=pathvisu)

    #corr, listindxtimeposimaxm, timefull, rflxfull = corr_tmpt(gdat.timethis, gdat.rflxthis, gdat.listtimetmpt, gdat.listdflxtmpt, \
    #                                                                    thrs=gdat.thrstmpt, boolanim=gdat.boolanimtmpt, boolplot=gdat.boolplottmpt, \
     #                                                               typeverb=gdat.typeverb, strgextn=gdat.strgextnthis, pathvisu=gdat.pathtargimag)
                
    return corr, listindxtimeposimaxm, meantimetmpt, timefull, lcurfull


# template matching

#@jit(nopython=True, parallel=True, fastmath=True, nogil=True)
def corr_arryprod(lcurtemp, lcurtmpt, numbkern):
    
    # for each size, correlate
    corrprod = [[] for k in range(numbkern)]
    for k in range(numbkern):
        corrprod[k] = lcurtmpt[k] * lcurtemp[k]
    
    return corrprod


#@jit(parallel=True)
def corr_copy(indxtimefullruns, lcurstan, indxtimekern, numbkern):
    '''
    Make a matrix with rows as the shifted and windowed copies of the time series.
    '''
    
    listlcurtemp = [[] for k in range(numbkern)]
    
    # loop over kernel sizes
    for k in range(numbkern):
        numbtimefullruns = indxtimefullruns[k].size
        numbtimekern = indxtimekern[k].size
        listlcurtemp[k] = np.empty((numbtimefullruns, numbtimekern))
        
        # loop over time
        for t in range(numbtimefullruns):
            listlcurtemp[k][t, :] = lcurstan[indxtimefullruns[k][t]+indxtimekern[k]]
    
    return listlcurtemp


def corr_tmpt(time, lcur, meantimetmpt, listlcurtmpt, \
              # type of verbosity
              ## -1: absolutely no text
              ##  0: no text output except critical warnings
              ##  1: minimal description of the execution
              ##  2: detailed description of the execution
              typeverb=1, \

              thrs=None, strgextn='', pathvisu=None, boolplot=True, \
              ## file type of the plot
              typefileplot='png', \
              boolanim=False, \
             ):
    
    timeoffs = np.amin(time) // 1000
    timeoffs *= 1000
    time -= timeoffs
    
    if typeverb > 1:
        timeinit = modutime.time()
    
    print('corr_tmpt()')
    
    if lcur.ndim > 1:
        raise Exception('')
    
    for lcurtmpt in listlcurtmpt:
        if not np.isfinite(lcurtmpt).all():
            raise Exception('')

    if not np.isfinite(lcur).all():
        raise Exception('')

    numbtime = lcur.size
    
    numbkern = len(listlcurtmpt)
    indxkern = np.arange(numbkern)
    
    # count gaps
    difftime = time[1:] - time[:-1]
    minmdifftime = np.amin(difftime)
    difftimesort = np.sort(difftime)[::-1]
    print('difftimesort')
    for k in range(difftimesort.size):
        print(difftimesort[k] / minmdifftime)
        if k == 20:
             break
    
    boolthrsauto = thrs is None
    
    print('temp: setting boolthrsauto')
    thrs = 1.
    boolthrsauto = False
    
    # number of time samples in the kernel
    numbtimekern = np.empty(numbkern, dtype=int)
    indxtimekern = [[] for k in indxkern]
    
    # take out the mean
    listlcurtmptstan = [[] for k in indxkern]
    for k in indxkern:
        listlcurtmptstan[k] = np.copy(listlcurtmpt[k])
        listlcurtmptstan[k] -= np.mean(listlcurtmptstan[k])
        numbtimekern[k] = listlcurtmptstan[k].size
        indxtimekern[k] = np.arange(numbtimekern[k])

    minmtimechun = 3 * 3. / 24. / 60. # [days]
    print('minmdifftime * 24 * 60')
    print(minmdifftime * 24 * 60)
    listlcurfull = []
    indxtimebndr = np.where(difftime > minmtimechun)[0]
    indxtimebndr = np.concatenate([np.array([0]), indxtimebndr, np.array([numbtime - 1])])
    numbchun = indxtimebndr.size - 1
    indxchun = np.arange(numbchun)
    corrchun = [[[] for k in indxkern] for l in indxchun]
    listindxtimeposimaxm = [[[] for k in indxkern] for l in indxchun]
    listlcurchun = [[] for l in indxchun]
    listtimechun = [[] for l in indxchun]
    print('indxtimebndr')
    print(indxtimebndr)
    print('numbchun')
    print(numbchun)
    for l in indxchun:
        
        print('Chunk %d...' % l)
        
        minmindxtimeminm = 0
        minmtime = time[indxtimebndr[l]+1]
        print('minmtime')
        print(minmtime)
        print('indxtimebndr[l]')
        print(indxtimebndr[l])
        maxmtime = time[indxtimebndr[l+1]]
        print('maxmtime')
        print(maxmtime)
        numb = int(round((maxmtime - minmtime) / minmdifftime))
        print('numb')
        print(numb)
        
        if numb == 0:
            print('Skipping due to chunk with single point...')
            continue

        timechun = np.linspace(minmtime, maxmtime, numb)
        listtimechun[l] = timechun
        print('timechun')
        summgene(timechun)
        
        if float(indxtimebndr[l+1] - indxtimebndr[l]) / numb < 0.8:
            print('Skipping due to undersampled chunk...')
            continue

        numbtimefull = timechun.size
        print('numbtimefull')
        print(numbtimefull)
        
        indxtimechun = np.arange(indxtimebndr[l], indxtimebndr[l+1] + 1)
        
        print('time[indxtimechun]')
        summgene(time[indxtimechun])
        
        # interpolate
        lcurchun = scipy.interpolate.interp1d(time[indxtimechun], lcur[indxtimechun])(timechun)
        
        if indxtimechun.size != timechun.size:
            print('time[indxtimechun]')
            if timechun.size < 50:
                for timetemp in time[indxtimechun]:
                    print(timetemp)
            summgene(time[indxtimechun])
            print('timechun')
            if timechun.size < 50:
                for timetemp in timechun:
                    print(timetemp)
            summgene(timechun)
            #raise Exception('')

        # take out the mean
        lcurchun -= np.mean(lcurchun)
        
        listlcurchun[l] = lcurchun
        
        # size of the full grid minus the kernel size
        numbtimefullruns = np.empty(numbkern, dtype=int)
        indxtimefullruns = [[] for k in indxkern]
        
        # find the correlation
        for k in indxkern:
            print('Kernel %d...' % k)
            
            if numb < numbtimekern[k]:
                print('Skipping due to chunk shorther than the kernel...')
                continue
            
            # find the total correlation (along the time delay axis)
            corrchun[l][k] = scipy.signal.correlate(lcurchun, listlcurtmptstan[k], mode='valid')
            print('corrchun[l][k]')
            summgene(corrchun[l][k])
        
            numbtimefullruns[k] = numbtimefull - numbtimekern[k] + 1
            indxtimefullruns[k] = np.arange(numbtimefullruns[k])
        
            print('numbtimekern[k]')
            print(numbtimekern[k])
            print('numbtimefullruns[k]')
            print(numbtimefullruns[k])

            if boolthrsauto:
                perclowrcorr = np.percentile(corr[k],  1.)
                percupprcorr = np.percentile(corr[k], 99.)
                indx = np.where((corr[k] < percupprcorr) & (corr[k] > perclowrcorr))[0]
                medicorr = np.median(corr[k])
                thrs = np.std(corr[k][indx]) * 7. + medicorr

            if not np.isfinite(corrchun[l][k]).all():
                raise Exception('')

            # determine the threshold on the maximum correlation
            if typeverb > 1:
                print('thrs')
                print(thrs)

            # find triggers
            listindxtimeposi = np.where(corrchun[l][k] > thrs)[0]
            if typeverb > 1:
                print('listindxtimeposi')
                summgene(listindxtimeposi)
            
            # cluster triggers
            listtemp = []
            listindxtimeposiptch = []
            for kk in range(len(listindxtimeposi)):
                listtemp.append(listindxtimeposi[kk])
                if kk == len(listindxtimeposi) - 1 or listindxtimeposi[kk] != listindxtimeposi[kk+1] - 1:
                    listindxtimeposiptch.append(np.array(listtemp))
                    listtemp = []
            
            if typeverb > 1:
                print('listindxtimeposiptch')
                summgene(listindxtimeposiptch)

            listindxtimeposimaxm[l][k] = np.empty(len(listindxtimeposiptch), dtype=int)
            for kk in range(len(listindxtimeposiptch)):
                indxtemp = np.argmax(corrchun[l][k][listindxtimeposiptch[kk]])
                listindxtimeposimaxm[l][k][kk] = listindxtimeposiptch[kk][indxtemp]
            
            if typeverb > 1:
                print('listindxtimeposimaxm[l][k]')
                summgene(listindxtimeposimaxm[l][k])
            
            if boolplot or boolanim:
                strgextntotl = strgextn + '_kn%02d' % k
        
            if boolplot:
                if numbtimefullruns[k] <= 0:
                    continue
                numbdeteplot = min(len(listindxtimeposimaxm[l][k]), 10)
                figr, axis = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
                
                #proc_axiscorr(time[indxtimechun], lcur[indxtimechun], axis[0], listindxtimeposimaxm[l][k])
                proc_axiscorr(timechun, lcurchun, axis[0], listindxtimeposimaxm[l][k])
                
                axis[1].plot(timechun[indxtimefullruns[k]], corrchun[l][k], color='m', ls='', marker='o', ms=1, rasterized=True)
                axis[1].set_ylabel('C')
                axis[1].set_xlabel('Time [BJD-%d]' % timeoffs)
                
                path = pathvisu + 'lcurflar_ch%02d%s.%s' % (l, strgextntotl, gdat.typefileplot)
                plt.subplots_adjust(left=0.2, bottom=0.2, hspace=0)
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()
                
                for n in range(numbdeteplot):
                    figr, axis = plt.subplots(figsize=(8, 4), sharex=True)
                    for i in range(numbdeteplot):
                        indxtimeplot = indxtimekern[k] + listindxtimeposimaxm[l][k][i]
                        proc_axiscorr(timechun, lcurchun, axis, listindxtimeposimaxm[l][k], indxtime=indxtimeplot, timeoffs=timeoffs)
                    path = pathvisu + 'lcurflar_ch%02d%s_det.%s' % (l, strgextntotl, gdat.typefileplot)
                    print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()
                
            print('Done with the plot...')
            if False and boolanim:
                path = pathvisu + 'lcur%s.gif' % strgextntotl
                if not os.path.exists(path):
                    anim_tmptdete(timefull, lcurfull, meantimetmpt[k], listlcurtmpt[k], pathvisu, \
                                                                listindxtimeposimaxm[l][k], corrprod[k], corrchun[l][k], strgextn=strgextntotl)
                else:
                    print('Skipping animation for kernel %d...' % k)
    if typeverb > 1:
        print('Delta T (corr_tmpt, rest): %g' % (modutime.time() - timeinit))

    return corrchun, listindxtimeposimaxm, listtimechun, listlcurchun


def read_qlop(path, pathcsvv=None, stdvcons=None):
    
    print('Reading QLP light curve from %s...' % path)
    objtfile = h5py.File(path, 'r')
    time = objtfile['LightCurve/BJD'][()] + 2457000.
    tmag = objtfile['LightCurve/AperturePhotometry/Aperture_002/RawMagnitude'][()]
    flux = 10**(-(tmag - np.nanmedian(tmag)) / 2.5)
    flux /= np.nanmedian(flux) 
    arry = np.empty((flux.size, 3))
    arry[:, 0] = time
    arry[:, 1] = flux
    if stdvcons is None:
        stdvcons = 1e-3
        print('Assuming a constant photometric precision of %g for the QLP light curve.' % stdvcons)
    stdv = np.zeros_like(flux) + stdvcons
    arry[:, 2] = stdv
    
    # filter out bad data
    indx = np.where((objtfile['LightCurve/QFLAG'][()] == 0) & np.isfinite(flux) & np.isfinite(time) & np.isfinite(stdv))[0]
    arry = arry[indx, :]
    if not np.isfinite(arry).all():
        print('arry')
        summgene(arry)
        raise Exception('Light curve is not finite')
    if arry.shape[0] == 0:
        print('arry')
        summgene(arry)
        raise Exception('Light curve has no data points')

    if pathcsvv is not None:
        print('Writing to %s...' % pathcsvv)
        np.savetxt(pathcsvv, arry, delimiter=',')

    return arry


# transits
def retr_indxtran(time, epoc, peri, duratrantotl=None, booldiag=True):
    '''
    Find the transit indices for a given time axis, epoch, period, and optionally transit duration.
    '''
    
    if booldiag:
        if not np.isfinite(epoc) or not np.isfinite(peri):
            print('')
            print('')
            print('')
            print('time')
            summgene(time)
            print('epoc')
            print(epoc)
            print('peri')
            print(peri)
            print('duratrantotl')
            print(duratrantotl)
            raise Exception('not np.isfinite(epoc) or not np.isfinite(peri)')
    
    if np.isfinite(peri):
        if duratrantotl is None:
            duratemp = 0.
        else:
            duratemp = duratrantotl
        intgminm = np.ceil((np.amin(time) - epoc - duratemp / 48.) / peri)
        intgmaxm = np.ceil((np.amax(time) - epoc - duratemp / 48.) / peri)
        indxtran = np.arange(intgminm, intgmaxm)
    else:
        indxtran = np.arange(1)
    
    return indxtran


def retr_listepoctran(time, epoc, peri, duratrantotl=None):
    '''
    Find the list of epochs inside the time-series for a given ephemerides
    '''

    indxtran = retr_indxtran(time, epoc, peri, duratrantotl=duratrantotl)
    listepoc = epoc + indxtran * peri

    return listepoc

    
def retr_indxtimetran(time, epoc, peri, \
                      
                      # total transit duration [hours]
                      duratrantotl, \
                      
                      # full transit duration [hours]
                      duratranfull=None, \
                      
                      # type of the in-transit phase interval
                      typeineg=None, \
                      
                      # Boolean flag to find time indices of individual transits
                      boolindi=False, \

                      # Boolean flag to return the out-of-transit time indices instead
                      booloutt=False, \
                      
                      # Boolean flag to return the secondary transit time indices instead
                      boolseco=False, \
                     ):
    '''
    Return the indices of times during transit.
    '''

    if not np.isfinite(time).all():
        raise Exception('')
    
    if not np.isfinite(duratrantotl).all():
        print('duratrantotl')
        print(duratrantotl)
        raise Exception('')
    
    if booloutt and boolindi:
        raise Exception('')

    indxtran = retr_indxtran(time, epoc, peri, duratrantotl)
    
    # phase offset
    if boolseco:
        offs = 0.5
    else:
        offs = 0.

    listindxtimetran = []
    for n in indxtran:
        timetotlinit = epoc + (n + offs) * peri - duratrantotl / 48.
        timetotlfinl = epoc + (n + offs) * peri + duratrantotl / 48.
        if duratranfull is not None:
            timefullinit = epoc + (n + offs) * peri - duratranfull / 48.
            timefullfinl = epoc + (n + offs) * peri + duratranfull / 48.
            timeingrhalf = (timetotlinit + timefullinit) / 2.
            timeeggrhalf = (timetotlfinl + timefullfinl) / 2.
            if typeineg == 'inge':
                indxtime = np.where((time > timetotlinit) & (time < timefullinit) | (time > timefullfinl) & (time < timetotlfinl))[0]
            if typeineg == 'ingr':
                indxtime = np.where((time > timetotlinit) & (time < timefullinit))[0]
            if typeineg == 'eggr':
                indxtime = np.where((time > timefullfinl) & (time < timetotlfinl))[0]
            if typeineg == 'ingrinit':
                indxtime = np.where((time > timetotlinit) & (time < timeingrhalf))[0]
            if typeineg == 'ingrfinl':
                indxtime = np.where((time > timeingrhalf) & (time < timefullinit))[0]
            if typeineg == 'eggrinit':
                indxtime = np.where((time > timefullfinl) & (time < timeeggrhalf))[0]
            if typeineg == 'eggrfinl':
                indxtime = np.where((time > timeeggrhalf) & (time < timetotlfinl))[0]
        else:
            indxtime = np.where((time > timetotlinit) & (time < timetotlfinl))[0]
        if indxtime.size > 0:
            listindxtimetran.append(indxtime)
    
    if boolindi:
        return listindxtimetran
    else:
        if len(listindxtimetran) > 0:
            indxtimetran = np.concatenate(listindxtimetran)
            indxtimetran = np.unique(indxtimetran)
        else:
            indxtimetran = np.array([])

    if booloutt:
        indxtimeretr = np.setdiff1d(np.arange(time.size), indxtimetran)
    else:
        indxtimeretr = indxtimetran
    
    return indxtimeretr
    

def retr_timeedge(time, lcur, timebrekregi, \
                  # Boolean flag to add breaks at discontinuties
                  booladdddiscbdtr, \
                  timescal, \
                 ):
    
    difftime = time[1:] - time[:-1]
    indxtimebrekregi = np.where(difftime > timebrekregi)[0]
    
    if booladdddiscbdtr:
        listindxtimebrekregiaddi = []
        dif1 = lcur[:-1] - lcur[1:]
        indxtimechec = np.where(dif1 > 20. * np.std(dif1))[0]
        for k in indxtimechec:
            if np.mean(lcur[-3+k:k]) - np.mean(lcur[k:k+3]) < np.std(np.concatenate((lcur[-3+k:k], lcur[k:k+3]))):
                listindxtimebrekregiaddi.append(k)

            #diff = lcur[k] - lcur[k-1]
            #if abs(diff) > 5 * np.std(lcur[k-3:k]) and abs(diff) > 5 * np.std(lcur[k:k+3]):
            #    listindxtimebrekregiaddi.append(k)
            #    #print('k')
            #    #print(k)
            #    #print('diff')
            #    #print(diff)
            #    #print('np.std(lcur[k:k+3])')
            #    #print(np.std(lcur[k:k+3]))
            #    #print('np.std(lcur[k-3:k])')
            #    #print(np.std(lcur[k-3:k]))
            #    #print('')
        listindxtimebrekregiaddi = np.array(listindxtimebrekregiaddi, dtype=int)
        indxtimebrekregi = np.concatenate([indxtimebrekregi, listindxtimebrekregiaddi])
        indxtimebrekregi = np.unique(indxtimebrekregi)

    timeedge = [0, np.inf]
    for k in indxtimebrekregi:
        timeedgeprim = (time[k] + time[k+1]) / 2.
        timeedge.append(timeedgeprim)
    timeedge = np.array(timeedge)
    timeedge = np.sort(timeedge)

    return timeedge


def retr_lliknegagpro(listparagpro, lcur, objtgpro):
    '''
    Compute the negative loglikelihood of the GP model
    '''
    
    objtgpro.set_parameter_vector(listparagpro)
    
    return -objtgpro.log_likelihood(lcur)


def retr_gradlliknegagpro(listparagpro, lcur, objtgpro):
    '''
    Compute the gradient of the negative loglikelihood of the GP model
    '''
    
    objtgpro.set_parameter_vector(listparagpro)
    
    return -objtgpro.grad_log_likelihood(lcur)[1]


def bdtr_tser( \
              # times in days at which the time-series data have been collected
              time, \
              
              # time-series data to be detrended
              lcur, \
              
              # standard-deviation of the time-series data to be detrended
              stdvlcur, \

              # masking before detrending
              ## list of midtransit epochs in BJD for which the time-series will be masked before detrending
              epocmask=None, \

              ## list of epochs in days for which the time-series will be masked before detrending
              perimask=None, \

              ## list of durations in hours for which the time-series will be masked before detrending
              duramask=None, \
              
              # Boolean flag to break the time-series into regions
              boolbrekregi=True, \
            
              # times to break the time-series into regions
              timeedge=None, \

              # minimum gap to break the time-series into regions
              timebrekregi=None, \
              
              # Boolean flag to add breaks at vertical discontinuties
              booladdddiscbdtr=True, \
              
              # type of baseline detrending
              ## 'GaussianProcess': Gaussian process
              ## 'medi': median
              ## 'Spline': spline
              typebdtr=None, \
              
              # order of the spline
              ordrspln=None, \
              
              # time scale of the spline detrending
              timescalbdtr=None, \
              
              # time scale of the median detrending
              timescalbdtrmedi=None, \
              
              # type of verbosity
              ## -1: absolutely no text
              ##  0: no text output except critical warnings
              ##  1: minimal description of the execution
              ##  2: detailed description of the execution
              typeverb=1, \
              
             ):
    '''
    Detrend input time-series data.
    '''
    
    if typebdtr is None:
        typebdtr = 'Spline'
    
    if boolbrekregi and timebrekregi is None:
        timebrekregi = 0.1 # [day]
    if ordrspln is None:
        ordrspln = 3
    if timescalbdtr is None:
        timescalbdtr = 0.5 # [days]
    if timescalbdtrmedi is None:
        timescalbdtrmedi = 0.5 # [days]
    
    if typebdtr == 'Spline' or typebdtr == 'GaussianProcess':
        timescal = timescalbdtr
    else:
        timescal = timescalbdtrmedi
    if typeverb > 0:
        print('Detrending the light curve with at a time scale of %.g days...' % timescal)
        if epocmask is not None:
            print('Using a specific ephemeris to mask out transits while detrending...')
    
    if timeedge is not None and len(timeedge) > 2 and not boolbrekregi:
        raise Exception('')

    if boolbrekregi:
        # determine the times at which the light curve will be broken into pieces
        if timeedge is None:
            timeedge = retr_timeedge(time, lcur, timebrekregi, booladdddiscbdtr, timescal)
        numbedge = len(timeedge)
        numbregi = numbedge - 1
    else:
        timeedge = [np.amin(time), np.amax(time)]
        numbregi = 1
    
    if typeverb > 1:
        print('timebrekregi')
        print(timebrekregi)
        print('Number of regions: %d' % numbregi)
        print('Times at the edges of the regions:')
        print(timeedge)

    indxregi = np.arange(numbregi)
    lcurbdtrregi = [[] for i in indxregi]
    indxtimeregi = [[] for i in indxregi]
    indxtimeregioutt = [[] for i in indxregi]
    listobjtspln = [[] for i in indxregi]
    for i in indxregi:
        if typeverb > 1:
            print('Region %d' % i)
        # find times inside the region
        indxtimeregi[i] = np.where((time >= timeedge[i]) & (time <= timeedge[i+1]))[0]
        timeregi = time[indxtimeregi[i]]
        lcurregi = lcur[indxtimeregi[i]]
        stdvlcurregi = stdvlcur[indxtimeregi[i]]
        
        # mask out the transits
        if epocmask is not None and len(epocmask) > 0 and duramask is not None and perimask is not None:
            # find the out-of-transit times
            indxtimetran = []
            for k in range(epocmask.size):
                if np.isfinite(duramask[k]):
                    indxtimetran.append(retr_indxtimetran(timeregi, epocmask[k], perimask[k], duramask[k]))
            
            indxtimetran = np.concatenate(indxtimetran)
            indxtimeregioutt[i] = np.setdiff1d(np.arange(timeregi.size), indxtimetran)
        else:
            indxtimeregioutt[i] = np.arange(timeregi.size)
            
        if typeverb > 1:
            print('lcurregi[indxtimeregioutt[i]]')
            summgene(lcurregi[indxtimeregioutt[i]])
        
        if typebdtr == 'medi':
            listobjtspln = None
            size = int(timescalbdtrmedi / np.amin(timeregi[1:] - timeregi[:-1]))
            if size == 0:
                print('timescalbdtrmedi')
                print(timescalbdtrmedi)
                print('np.amin(timeregi[1:] - timeregi[:-1])')
                print(np.amin(timeregi[1:] - timeregi[:-1]))
                print('lcurregi')
                summgene(lcurregi)
                raise Exception('')
            lcurbdtrregi[i] = 1. + lcurregi - scipy.ndimage.median_filter(lcurregi, size=size)
        
        if typebdtr == 'GaussianProcess':
            # fit a Gaussian Process (GP) model to the data as baseline
            ## construct the kernel object
            objtkern = celerite.terms.Matern32Term(log_sigma=np.log(np.std(4. * lcurregi[indxtimeregioutt[i]])), log_rho=np.log(timescalbdtr))
            print('sigma for GP')
            print(np.std(lcurregi[indxtimeregioutt[i]]))
            print('rho for GP [days]')
            print(timescalbdtr)

            ## construct the GP model object
            objtgpro = celerite.GP(objtkern, mean=np.mean(lcurregi[indxtimeregioutt[i]]))
            
            # compute the covariance matrix
            objtgpro.compute(timeregi[indxtimeregioutt[i]], yerr=stdvlcurregi[indxtimeregioutt[i]])
            
            # get the initial parameters of the GP model
            #parainit = objtgpro.get_parameter_vector()
            
            # get the bounds on the GP model parameters
            #limtparagpro = objtgpro.get_parameter_bounds()
            
            # minimize the negative loglikelihood
            #objtmini = scipy.optimize.minimize(retr_lliknegagpro, parainit, jac=retr_gradlliknegagpro, method="L-BFGS-B", bounds=limtparagpro, args=(lcurregi[indxtimeregioutt[i]], objtgpro))
            
            #print('GP Matern 3/2 parameters with maximum likelihood:')
            #print(objtmini.x)

            # update the GP model with the parameters that minimize the negative loglikelihood
            #objtgpro.set_parameter_vector(objtmini.x)
            
            # get the GP model mean baseline
            lcurbase = objtgpro.predict(lcurregi[indxtimeregioutt[i]], t=timeregi, return_cov=False, return_var=False)#[0]
            
            # subtract the baseline from the data
            lcurbdtrregi[i] = 1. + lcurregi - lcurbase

            listobjtspln[i] = objtgpro
        if typebdtr == 'Spline':
            # fit the spline
            if lcurregi[indxtimeregioutt[i]].size > 0:
                if timeregi[indxtimeregioutt[i]].size < 4:
                    print('Warning! Only %d points available for spline! This will result in a trivial baseline-detrended light curve (all 1s).' \
                                                                                                                % timeregi[indxtimeregioutt[i]].size)
                    print('numbregi')
                    print(numbregi)
                    print('indxtimeregioutt[i]')
                    summgene(indxtimeregioutt[i])
                    for ii in indxregi:
                        print('indxtimeregioutt[ii]')
                        summgene(indxtimeregioutt[ii])
                        
                    #raise Exception('')

                    listobjtspln[i] = None
                    lcurbdtrregi[i] = np.ones_like(lcurregi)
                else:
                    
                    minmtime = np.amin(timeregi[indxtimeregioutt[i]])
                    maxmtime = np.amax(timeregi[indxtimeregioutt[i]])
                    numbknot = int((maxmtime - minmtime) / timescalbdtr) + 1
                    
                    timeknot = np.linspace(minmtime, maxmtime, numbknot)
                    timeknot = timeknot[1:-1]
                    numbknot = timeknot.size

                    indxknotregi = np.digitize(timeregi[indxtimeregioutt[i]], timeknot) - 1

                    if typeverb > 1:
                        print('minmtime')
                        print(minmtime)
                        print('maxmtime')
                        print(maxmtime)
                        print('timescalbdtr')
                        print(timescalbdtr)
                        print('%d knots used (exclduing the end points).' % (numbknot))
                        if numbknot > 1:
                            print('Knot separation: %.3g hours' % (24 * (timeknot[1] - timeknot[0])))
                    
                    if numbknot > 0:
                        try:
                            objtspln = scipy.interpolate.LSQUnivariateSpline(timeregi[indxtimeregioutt[i]], lcurregi[indxtimeregioutt[i]], timeknot, k=ordrspln)
                        except:
                            print('')
                            print('')
                            print('')
                            print('timeknot')
                            print(timeknot)
                            print('ordrspln')
                            print(ordrspln)
                            raise Exception('scipy.interpolate.LSQUnivariateSpline() failed.')
                        lcurbdtrregi[i] = lcurregi - objtspln(timeregi) + 1.
                        listobjtspln[i] = objtspln
                    else:
                        lcurbdtrregi[i] = lcurregi - np.median(lcurregi) + 1.
                        listobjtspln[i] = None
                    
            else:
                lcurbdtrregi[i] = lcurregi
                listobjtspln[i] = None
            
            if typeverb > 1:
                print('lcurbdtrregi[i]')
                summgene(lcurbdtrregi[i])
                print('')
    
    lcurbdtr = np.concatenate(lcurbdtrregi)

    return lcurbdtr, lcurbdtrregi, indxtimeregi, indxtimeregioutt, listobjtspln, timeedge


def retr_stdvwind(ydat, sizewind, boolcuttpeak=True):
    '''
    Return the standard deviation of a series inside a running windown.
    '''
    
    numbdata = ydat.size
    
    if sizewind % 2 != 1 or sizewind > numbdata:
        raise Exception('')

    sizewindhalf = int((sizewind - 1) / 2)
    
    indxdata = np.arange(numbdata)
    
    stdv = np.empty_like(ydat)
    for k in indxdata:
        


        minmindx = max(0, k - sizewindhalf)
        maxmindx = min(numbdata - 1, k + sizewindhalf)
        
        if boolcuttpeak:
            indxdatawind = np.arange(minmindx, maxmindx+1)
            #indxdatawind = indxdatawind[np.where(ydat[indxdatawind] < np.percentile(ydat[indxdatawind], 99.999))]
            indxdatawind = indxdatawind[np.where(ydat[indxdatawind] != np.amax(ydat[indxdatawind]))]
        
        else:
            if k > minmindx and k+1 < maxmindx:
                indxdatawind = np.concatenate((np.arange(minmindx, k), np.arange(k+1, maxmindx+1)))
            elif k > minmindx:
                indxdatawind = np.arange(minmindx, k)
            elif k+1 < maxmindx:
                indxdatawind = np.arange(k+1, maxmindx+1)

        stdv[k] = np.std(ydat[indxdatawind])
    
    return stdv


def plot_tser( \
              
              # path in which the plot will be placed
              pathvisu=None, \
              
              # a string that will be tagged onto the filename
              strgextn=None, \
              
              # dictionary holding the model time-series
              dictmodl=None, \
              
              # the time stamps of time-series data
              timedata=None, \
              
              # values of the time-series data
              tserdata=None, \
              
              # time stamps of the binned time-series data
              timedatabind=None, \
              
              # values of the binned time-series data
              tserdatabind=None, \
              
              # uncertainties of the binned time-series data
              tserdatastdvbind=None, \
              
              # Boolean flag to break the line of the model when separation is very large
              boolbrekmodl=True, \
              
              # Boolean flag to ignore any existing files and overwrite
              boolwritover=False, \
              
              # label for the horizontal axis, including the unit
              lablxaxi=None, \
              
              # label for the vertical axis, including the unit
              lablyaxi=None, \
              
              # phase folding
              ## Boolean flag to indicate if the provided horizontal axis is phase, not time
              boolphas=False, \
              
              ## Boolean flag to fold input time-series
              boolfold=False, \
              
              # the phase to center when phase folding
              phascntr=0., \

              ## epoch for optional phase-folding
              epoc=None, \
              
              ## period for optional phase-folding
              peri=None, \
              
              # type of the horizontal axis (only taken into account if boolphas is True)
              typephasunit='phas', \

              # size of the figure
              sizefigr=None, \
              
              # list of times at which to draw vertical dashed lines
              listxdatvert=None, \

              # colors of the vertical dashed lines
              listcolrvert=None, \
              
              # time offset
              timeoffs=None, \
              
              # phase offset
              phasoffs=0., \
              
              # limits for the horizontal axis in the form of a two-tuple
              limtxaxi=None, \
              
              # reference vertical value
              ydatcntr=1., \

              # limits for the vertical axis in the form of a two-tuple
              limtyaxi=None, \
              
              # type of signature for the generating code
              typesigncode=None, \
              
              # Boolean flag to draw a center line to guide the eye
              booldrawcntr=None, \

              # title for the plot
              strgtitl='', \
              
              # Boolean flag to diagnose
              ## diagnostic mode is always on by default, which should be turned off during large-scale runs, where speed is a concern
              booldiag=True, \
                      
              ## file type of the plot
              typefileplot='png', \
            
              # type of plot background
              typeplotback='black', \

             ):
    '''
    Plot time-series data and model
    '''
    
    if pathvisu is not None and (strgextn is None or strgextn == ''):
        print('')
        print('')
        print('')
        raise Exception('If pathvisu is not None, strgextn should not be an empty string or None.')
    
    if timeoffs != 0. and phasoffs != 0.:
        raise Exception('')
    
    if boolfold and boolphas:
        raise Exception('If input horizontal axis is phase (boolphas is True), then it cannot be phase-folded (boolfold must be False).')
    
    if boolfold:
        if epoc is None or peri is None:
            print('')
            print('')
            print('')
            print('epoc')
            print(epoc)
            print('peri')
            print(peri)
            raise Exception('boolfold is True but epoc is None or peri is None')

    if boolphas or boolfold:
        typexdat = 'phas'
    else:
        typexdat = 'time'
    
    if typeplotback == 'white':
        colrbkgd = 'white'
        colrdraw = 'black'
    elif typeplotback == 'black':
        colrbkgd = 'black'
        colrdraw = 'white'
    
    if pathvisu is not None:
        dicttdpy = tdpy.retr_dictstrg()

        if strgextn[0] == '_':
            strgextn = strgextn[1:]

        path = pathvisu + '%s_%s.%s' % (dicttdpy['tser'], strgextn, typefileplot)
    
        # skip plotting
        if not boolwritover and os.path.exists(path):
            print('Plot already exists at %s. Skipping...' % path)
            return path
    
    boollegd = False
    
    if boolfold:
        if timedata is not None:
            arrylcurdata = np.empty((timedata.size, 3))
            arrylcurdata[:, 0] = timedata
            arrylcurdata[:, 1] = tserdata
            arrypcurdata = fold_tser(arrylcurdata, epoc, peri, phascntr=phascntr)
    
    if sizefigr is None:
        sizefigr = [8., 2.5]

    figr, axis = plt.subplots(figsize=sizefigr)
    
    # time offset
    if typexdat == 'time' and timeoffs is None:
        # obtained a concatenated array of all times
        listarrytimeconc = []
        if timedata is not None:
            listarrytimeconc.append(timedata)
        if timedatabind is not None:
            listarrytimeconc.append(timedatabind)
        if dictmodl is not None:
            for attr in dictmodl:
                listarrytimeconc.append(dictmodl[attr]['time'])
        # determine the time offset
        timeoffs = tdpy.retr_offstime(np.concatenate(listarrytimeconc))

    if typexdat == 'phas':
        xdatoffs = phasoffs
    else:
        xdatoffs = timeoffs

    # the factor to multiply the time axis and its label
    if typexdat == 'phas' and typephasunit == 'time':
        facttime, lablunittime = tdpy.retr_timeunitdays(peri)
    
    # raw data
    if timedata is not None:
        axis.plot(timedata - xdatoffs, tserdata, color='gray', ls='', marker='o', ms=1, rasterized=True)
    
        if tserdata.ndim != 1:
            print('')
            print('')
            print('')
            print('tserdata')
            summgene(tserdata)
            raise Exception('tserdata.ndim != 1')
    
    # binned data
    if timedatabind is not None:
        axis.errorbar(timedatabind, tserdatabind, yerr=tserdatastdvbind, color='k', ls='', marker='o', ms=2)
    
        if tserdatabind.ndim != 1:
            print('')
            print('')
            print('')
            print('tserdatabind')
            summgene(tserdatabind)
            raise Exception('tserdatabind.ndim != 1')
    
    if dictmodl is not None:
        booldrawcntr = True
    else:
        booldrawcntr = False

    # model
    if dictmodl is not None:
        
        k = 0
        for attr in dictmodl:
            if 'lsty' in dictmodl[attr]:
                ls = dictmodl[attr]['lsty']
            else:
                ls = None
            
            if 'colr' in dictmodl[attr]:
                color = dictmodl[attr]['colr']
            else:
                color = None
                
            if 'alph' in dictmodl[attr]:
                alpha = dictmodl[attr]['alph']
            else:
                alpha = None
            
            if dictmodl[attr]['tser'].ndim != 1:
                print('')
                print('')
                print('')
                print('dictmodl[attr][tser]')
                summgene(dictmodl[attr]['tser'])
                raise Exception('dictmodl[attr][tser].ndim != 1')
            
            
            diftimemodl = dictmodl[attr]['time'][1:] - dictmodl[attr]['time'][:-1]
            if boolbrekmodl and np.std(diftimemodl) < 0.1 * np.mean(diftimemodl):

                minmdiftimemodl = np.amin(diftimemodl)

                if minmdiftimemodl < 0:
                    print('')
                    print('')
                    print('')
                    print('minmdiftimemodl [minutes]')
                    print(minmdiftimemodl * 24. * 60.)
                    print('diftimemodl')
                    summgene(diftimemodl)
                    raise Exception('minmdiftimemodl is negative when boolbrekmodl is True. Time array should be sorted.')

                indxtimebrekregi = np.where(diftimemodl > 2 * np.amin(diftimemodl))[0] + 1
                indxtimebrekregi = np.concatenate([np.array([0]), indxtimebrekregi, np.array([dictmodl[attr]['time'].size - 1])])
                numbtimebrekregi = indxtimebrekregi.size
                numbtimechun = numbtimebrekregi - 1
                
                xdat = []
                ydat = []
                for n in range(numbtimechun):
                    xdat.append(dictmodl[attr]['time'][indxtimebrekregi[n]:indxtimebrekregi[n+1]])
                    ydat.append(dictmodl[attr]['tser'][indxtimebrekregi[n]:indxtimebrekregi[n+1]])
                    
            else:
                xdat = [dictmodl[attr]['time']]
                ydat = [dictmodl[attr]['tser']]
            numbchun = len(xdat)
            
            if boolbrekmodl and numbchun > 0.5 * dictmodl[attr]['time'].size:
                print('')
                print('')
                print('')
                print('diftimemodl')
                summgene(diftimemodl)
                print('minmdiftimemodl')
                print(minmdiftimemodl)
                print('Miletos.plot_tser(): Warning! Number of regions (%d) is more than half the number of data points (%d).' % (numbchun, dictmodl[attr]['time'].size))
                print('Perhaps the data is undersampled. Model light curve plot will not look reasonable.')
                raise Exception('')

            for n in range(numbchun):
                if n == 0 and 'labl' in dictmodl[attr]:
                    label = dictmodl[attr]['labl']
                    boollegd = True
                else:
                    label = None
    
        
                if boolfold:
                    arrylcurmodl = np.empty((xdat[n].size, 3))
                    arrylcurmodl[:, 0] = xdat[n]
                    arrylcurmodl[:, 1] = ydat[n]
                    arrypcurmodl = fold_tser(arrylcurmodl, epoc, peri, phascntr=phascntr)
                    xdat[n] = arrypcurmodl[:, 0]
                    ydat[n] = arrypcurmodl[:, 1]
                
                xdattemp = xdat[n] - xdatoffs
                
                if typexdat == 'phas' and typephasunit == 'time':
                    if numbchun != 1:
                        raise Exception('')
                    xdattemp *= peri * facttime
                
                if booldiag:
                    if xdattemp.size != ydat[n].size:
                        print('')
                        print('')
                        print('')
                        print('xdattemp')
                        summgene(xdattemp)
                        print('ydat[n]')
                        summgene(ydat[n])
                        print('boolfold')
                        print(boolfold)
                        print('attr')
                        print(attr)
                        print('numbchun')
                        print(numbchun)
                        print('boolbrekmodl')
                        print(boolbrekmodl)
                        raise Exception('xdattemp.size != ydat[n].size')
                
                axis.plot(xdattemp, ydat[n], color=color, lw=1, label=label, ls=ls, alpha=alpha)
            k += 1
    
    if booldrawcntr:
        xdatcntr = np.array(axis.get_xlim())
        ydatcntr = np.full_like(xdatcntr, fill_value=ydatcntr)
        axis.plot(xdatcntr, ydatcntr, color='gray', lw=1, ls='-.', alpha=alpha, zorder=-1)
    
    if lablxaxi is None:
        if typexdat == 'time':
            if xdatoffs == 0:
                lablxaxi = 'Time [days]'
            else:
                lablxaxi = 'Time [BJD-%d]' % xdatoffs
        else:
            #if typephasunit == 'time':
            #    lablxaxi = 'Time [%s]' % lablunittime
            #else:
            lablxaxi = 'Phase'
            
    axis.set_xlabel(lablxaxi)
    
    if limtxaxi is not None:
        if not np.isfinite(limtxaxi).all():
            print('limtxaxi')
            print(limtxaxi)
            raise Exception('')

        axis.set_xlim(limtxaxi)
    
    if listxdatvert is not None:
        for k, xdatvert in enumerate(listxdatvert):
            if listcolrvert is None:
                colr = 'gray'
            else:
                colr = listcolrvert[k]
            axis.axvline(xdatvert, ls='--', color=colr, alpha=0.4)
    
    if limtyaxi is not None:
        axis.set_ylim(limtyaxi)

    if lablyaxi is None:
        lablyaxi = 'Relative flux'
    
    axis.set_ylabel(lablyaxi)
    axis.set_title(strgtitl)
    
    if typesigncode is not None:
        tdpy.sign_code(axis, typesigncode)

    if boollegd:
        axis.legend()

    plt.subplots_adjust(bottom=0.2, top=0.8)
    
    if pathvisu is not None:
        print('Writing to %s...' % path)
        plt.savefig(path, dpi=300)
        plt.close()
        return path
    else:
        plt.show()
        return None



def fold_tser(arry, epoc, peri, boolxdattime=False, boolsort=True, phascntr=0.5, booldiag=True):
    
    if arry.ndim == 3:
        time = arry[:, 0, 0]
    elif arry.ndim == 2:
        time = arry[:, 0]
    else:
        print('')
        print('')
        print('')
        print('arry')
        summgene(arry)
        raise Exception('arry should be two or three dimensional.')

    arryfold = np.empty_like(arry)
    
    phasinit = ((time - epoc) % peri) / peri
    phasdiff = 0.5 - phascntr
    xdat = (phasinit + phasdiff) % 1. - phasdiff
    
    if boolxdattime:
        xdat *= peri
    
    if arry.ndim == 3:
        arryfold[:, 0, 0] = xdat
    else:
        arryfold[:, 0] = xdat

    arryfold[:, ..., 1:3] = arry[:, ..., 1:3]
    
    if boolsort:
        indx = np.argsort(xdat)
        arryfold = arryfold[indx, :]
    
    return arryfold


def read_tesskplr_fold(pathfold, pathfoldsave, boolmaskqual=True, typeinst='TESS', strgtypelcur='PDCSAP_FLUX', boolnorm=None, liststrgtypelcursave=None):
    '''
    Reads all TESS or Kepler light curves in a folder and returns a data cube with time, flux and flux error.
    '''
    
    listpath = fnmatch.filter(os.listdir(pathfold), '%s*' % typeinst)
    listarry = []
    for path in listpath:
        arry = read_tesskplr_file(pathfold + path + '/' + path + '_lc.fits', typeinst=typeinst, strgtypelcur=strgtypelcur, boolmaskqual=boolmaskqual, boolnorm=boolnorm, \
                                  pathfoldsave=pathfoldsave, \
                                  liststrgtypelcursave=liststrgtypelcursave, \
                                 )
        listarry.append(arry)
    
    # merge sectors
    arry = np.concatenate(listarry, axis=0)
    
    # sort in time
    indxsort = np.argsort(arry[:, 0])
    arry = arry[indxsort, :]
    
    return arry 


def read_tesskplr_file(path, typeinst='TESS', strgtypelcur='PDCSAP_FLUX', boolmaskqual=True, boolmasknann=True, boolnorm=None, \
                       
                       booldiag=True, \
                       pathfoldsave=None, \
                       liststrgtypelcursave=None, \
                       # type of verbosity
                       ## -1: absolutely no text
                       ##  0: no text output except critical warnings
                       ##  1: minimal description of the execution
                       ##  2: detailed description of the execution
                       typeverb=1, \
                      ):
    '''
    Read a TESS or Kepler light curve file and returns a data cube with time, flux and flux error.
    '''
    
    if boolnorm is None:
        boolnorm = True
    
    if typeverb > 0:
        print('Reading from %s...' % path)
    listhdun = astropy.io.fits.open(path)
    
    if liststrgtypelcursave is None:
        liststrgtypelcursave = ['SAP_FLUX', 'SAP_BKG']
    
    tsec = listhdun[0].header['SECTOR']
    tcam = listhdun[0].header['CAMERA']
    tccd = listhdun[0].header['CCD']
    
    # Boolean flag indicating whether the target file is a light curve or target pixel file
    boollcur = 'lc.fits' in path
    
    liststrgtypelcur = [strgtypelcur]
    if liststrgtypelcursave is not None:
        liststrgtypelcur += liststrgtypelcursave

    for strgtypelcurtemp in liststrgtypelcur:
        
        # indices of times where the quality flag is not raised (i.e., good quality)
        if boollcur:
            indxtimequalgood = np.where((listhdun[1].data['QUALITY'] == 0) & np.isfinite(listhdun[1].data[strgtypelcurtemp]))[0]
        else:
            indxtimequalgood = np.where((listhdun[1].data['QUALITY'] == 0) & np.isfinite(listhdun[1].data['TIME']))[0]
        time = listhdun[1].data['TIME']
        
        if boollcur:

            time = listhdun[1].data['TIME']
            if typeinst == 'TESS':
                time += 2457000
            if typeinst == 'Kepler':
                time += 2454833
        
            flux = listhdun[1].data[strgtypelcurtemp]
            stdv = listhdun[1].data[strgtypelcurtemp+'_ERR']
            #print(listhdun[1].data.names)
            
            if boolmaskqual:
                # filtering for good quality
                if typeverb > 0:
                    print('Masking out bad data... %d temporal samples (%.3g%%) will survive.' % (indxtimequalgood.size, 100. * indxtimequalgood.size / time.size))
                time = time[indxtimequalgood]
                flux = flux[indxtimequalgood, ...]
                if boollcur:
                    stdv = stdv[indxtimequalgood]
        
            numbtime = time.size
            arry = np.empty((numbtime, 3))
            arry[:, 0] = time
            arry[:, 1] = flux
            arry[:, 2] = stdv
            
            #indxtimenanngood = np.where(~np.any(np.isnan(arry), axis=1))[0]
            #if boolmasknann:
            #    arry = arry[indxtimenanngood, :]
            
            # normalize
            if boolnorm:
                factnorm = np.median(arry[:, 1])
                arry[:, 1] /= factnorm
                arry[:, 2] /= factnorm
        
        if strgtypelcurtemp == strgtypelcur:
            arryretr = arry

        # save
        if pathfoldsave is not None:
            path = '%s%s_%s_Sector%02d.csv' % (pathfoldsave, typeinst, strgtypelcurtemp, tsec)
            print('Writing to %s...' % path)
            np.savetxt(path, arry, delimiter=',', header='time, relative flux, relative flux error')
        
    if boollcur:
        return arryretr, tsec, tcam, tccd
    else:
        return listhdun, indxtimequalgood, tsec, tcam, tccd


def setp_time(gdat, namevarb=None):

    gdat.listtime = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gdat.listtimefine = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gdat.timeconc = [[] for b in gdat.indxdatatser]
    gdat.timefineconc = [[] for b in gdat.indxdatatser]
    gdat.time = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gdat.timefine = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            if namevarb is None:
                gdat.time[b][p] = gdat.true.time[b][p]
            else:
                gdat.time[b][p] = gdat.arrytser[namevarb][b][p][:, 0, 0]
            for y in gdat.indxchun[b][p]:
                
                if namevarb is None and gdat.liststrgtypedata[b][p] != 'obsd':
                    gdat.listtime[b][p][y] = gdat.true.listtime[b][p][y]
                else:
                    gdat.listtime[b][p][y] = gdat.listarrytser['Raw'][b][p][y][:, 0, 0]
                
                if gdat.booldiag:
                    if isinstance(gdat.listtime[b][p][y], list):
                        print('')
                        print('')
                        print('')
                        print('namevarb')
                        print(namevarb)
                        print('gdat.listtime[b][p][y]')
                        print(gdat.listtime[b][p][y])
                        raise Exception('gdat.listtime[b][p][y] should be a numpy array.')

                difftimefine = 0.5 * np.amin(gdat.listtime[b][p][y][1:] - gdat.listtime[b][p][y][:-1])
                gdat.listtimefine[b][p][y] = np.arange(np.amin(gdat.listtime[b][p][y]), np.amax(gdat.listtime[b][p][y]) + difftimefine, difftimefine)
            gdat.timefine[b][p] = np.concatenate(gdat.listtimefine[b][p])
        if len(gdat.time[b]) > 0:
            gdat.timeconc[b] = np.concatenate(gdat.time[b])
            gdat.timefineconc[b] = np.concatenate(gdat.timefine[b])
    
    # time axis concatenated across light curve and RV
    listtimetemp = []
    for b in gdat.indxdatatser:
        if len(gdat.timeconc[b]) > 0:
            listtimetemp.append(gdat.timeconc[b])
    gdat.timeconcconc = np.concatenate(listtimetemp)

    # time offset
    if gdat.timeoffs is None:
        gdat.timeoffs = tdpy.retr_offstime(gdat.timeconcconc)


def retr_strginst(listlablinst):

    ## list of strings indicating instruments
    liststrginst = [[[] for p in range(len(listlablinst[b]))] for b in range(len(listlablinst))]
    for b in range(len(liststrginst)):
        for p in range(len(liststrginst[b])):
            liststrginst[b][p] = ''.join(listlablinst[b][p].split(' '))
                
    return liststrginst


def setup_miletos(gdat):

    # types of data
    gdat.listlabldatatser = ['Relative Flux', 'Relative Velocity']
    gdat.liststrgdatatser = ['RelativeFlux', 'RelativeVelocity']
    gdat.numbdatatser = len(gdat.listlabldatatser)
    gdat.indxdatatser = np.arange(gdat.numbdatatser)
    
    # instruments
    gdat.numbinst = np.empty(gdat.numbdatatser, dtype=int)
    gdat.indxinst = [[] for b in gdat.indxdatatser]
    for b in gdat.indxdatatser:
        gdat.numbinst[b] = len(gdat.listlablinst[b])
        gdat.indxinst[b] = np.arange(gdat.numbinst[b])
    
    if gdat.boolsimutargpartfprt is None:
        gdat.boolsimutargpartfprt = False

    # list of data types
    if gdat.liststrgtypedata is None:
        gdat.liststrgtypedata = [[] for b in gdat.indxdatatser]
        for b in gdat.indxdatatser:
            if gdat.boolsimutargpartfprt:
                gdat.liststrgtypedata[b] = ['simutargpartfprt' for p in gdat.indxinst[b]]
            else:
                gdat.liststrgtypedata[b] = ['obsd' for p in gdat.indxinst[b]]
    
    if gdat.typeverb > 0:
        print('gdat.liststrgtypedata')
        print(gdat.liststrgtypedata)


def setup1_miletos(gdat):

    # Boolean flag indicating if the simulated target is a synthetic one
    gdat.booltargsynt = False
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            if gdat.liststrgtypedata[b][p] == 'simutargsynt':
                gdat.booltargsynt = True
    
    # Boolean flag indicating if there is any simulated data
    gdat.boolsimusome = False
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            if gdat.liststrgtypedata[b][p] != 'obsd':
                gdat.boolsimusome = True

    # Boolean flag indicating if all data are observed
    gdat.booldataobsv = True
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            if gdat.liststrgtypedata[b][p] != 'obsd':
                gdat.booldataobsv = False


def init( \
         
         # a label distinguishing the run to be used in the plots
         lablcnfg=None, \
         
         # a string distinguishing the run to be used in the file names
         strgcnfg=None, \
         
         # target identifiers
         ## string to search on MAST
         strgmast=None, \
         
         ## TIC ID
         ticitarg=None, \
         
         ## TOI ID
         toiitarg=None, \
         
         ## RA
         rasctarg=None, \
         
         ## Dec
         decltarg=None, \

         ## a string for the label of the target
         labltarg=None, \
         
         ## a string for the folder name and file name extensions
         strgtarg=None, \
         
         # string indicating the cluster of targets
         strgclus=None, \
        
         ## Boolean flag indicating whether the input photometric data will be median-normalized
         boolnormphot=True, \
         
         # options changing the overall execution
         ## Boolean flag to enforce offline operation
         boolforcoffl=False, \

         # target visibility from a given observatory over a given night and year
         ## Boolean flag to calculate visibility of the target
         boolcalcvisi=False, \
         
         ## Boolean flag to plot visibility of the target
         boolplotvisi=None, \
         
         # Boolean flag to detrend
         boolbdtr=None, \

         ## latitude of the observatory for the visibility calculation
         latiobvt=None, \
         
         ## longitude of the observatory for the visibility calculation
         longobvt=None, \
         
         ## height of the observatory for the visibility calculation
         heigobvt=None, \
         
         ## string of time for the night
         strgtimeobvtnigh=None, \
         
         ## string of time for the beginning of the year
         strgtimeobvtyear=None, \
         
         # Boolean flag to use the highest cadence whenever possible
         boolutilcadehigh=True, \
         
         ## list of time difference samples for the year
         listdelttimeobvtyear=None, \
         
         # Boolean flag to calculate the time indices inside and outside transits
         boolcalctimetran=False, \

         ## local time offset for the visibility calculation
         offstimeobvt=0., \
         
         # Boolean flag to separate TESS sectors into separate instruments TESS_S*
         boolbrektess=False, \

         # dictionary for parameters of the true generative model
         dicttrue=None, \

         # dictionary for parameters of the fitting generative model
         dictfitt=None, \

         ## general plotting
         ## Boolean flag to make plots
         boolplot=True, \
         
         ## Boolean flag to plot target features along with the features of the parent population
         boolplotpopl=False, \
         
         ## Boolean flag to plot DV reports
         boolplotdvrp=None, \
         
         ## Boolean flag to plot the time-series
         boolplottser=None, \
         
         ## time offset to subtract from the time axes, which is otherwise automatically estimated
         timeoffs=None, \
         
         ## file type of the plot
         typefileplot='png', \

         ## Boolean flag to write planet name on plots
         boolwritplan=True, \
         
         ## Boolean flag to rasterize the raw time-series on plots
         boolrastraww=True, \
         
         ## list of transit depths to indicate on the light curve and phase curve plots
         listdeptdraw=None, \
         
         # list of experiments whose data are to be downloaded
         liststrgexpr=None, \
         
         # Boolean flag to plot the true ephesos model
         boolplotefestrue=True, \

         # Boolean flag to animate the true ephesos model
         boolmakeanimefestrue=False, \
        
         # paths
         ## the path of the folder in which the target folder will be placed
         pathbase=None, \
         
         ## the path of the target folder
         pathtarg=None, \
         
         ## the path of the folder for data on the target
         pathdatatarg=None, \
         
         ## the path of the folder for visuals on the target
         pathvisutarg=None, \
         
         # data
         ## string indicating the type of data
        
         ## data retrieval
         ### subset of TESS sectors to retrieve
         listtsecsele=None, \
         
         ### Boolean flag to apply quality mask
         boolmaskqual=True, \
         
         ### Boolean flag to only utilize SPOC light curves on a local disk
         boolutiltesslocl=False, \

         ### name of the data product of lygos indicating which analysis has been used for photometry
         nameanlslygo='psfn', \

         ### Boolean flag to use 20-sec TPF when available
         boolfasttpxf=True, \
         
         # Boolean flag to indicate a prior belief that the target is a planetary system
         ## This turns on relevant prior plotting without triggering an attempt to model it
         boolpriotargpsys=False, \

         # input data
         ## path of the CSV file containing the input data
         listpathdatainpt=None, \
         
         ## input data as a dictionary of lists of numpy arrays
         listarrytser=None, \
         
         ## list of pointing IDs for the input data
         listipntinpt=None, \
         
         
         ## list of values for the energy axis
         listener=None, \
         
         ## label for the energy axis
         lablener=None, \
         
         # Source of TESS TPF light curves (FFIs are always extracted by lygos)
         ## 'lygos': always lygos
         ## 'SPOC_first': SPOC whenever available, otherwise lygos
         ## 'SPOC': SPOC only, which leaves out some sectors because SPOC has a magnitude limit
         typelcurtpxftess='SPOC', \
         
         ## type of SPOC light curve: 'PDC', 'SAP'
         typedataspoc='PDC', \
                  
         # type of data for each data kind, instrument, and chunk
         ## 'simutargsynt': simulated data on a synthetic target
         ## 'simutargpartsynt': simulated data on a particular target over a synthetic temporal footprint
         ## 'simutargpartfprt': simulated data on a particular target over the observed temporal footprint 
         ## 'simutargpartinje': simulated data obtained by injecting a synthetic signal on observed data on a particular target with a particular observational baseline 
         ## 'obsd': observed data on a particular target
         liststrgtypedata=None, \
            
         # Boolean flag to default gdat.liststrgtypedata into 'simutargpartfprt' for all data types and instruments
         boolsimutargpartfprt=None, \

         ## list of labels indicating instruments
         listlablinst=None, \
         
         ## list of strings indicating chunks in the filenames
         liststrgchun=None, \
         
         ## list of strings indicating chunks in the plots
         listlablchun=None, \
         
         ## list of chunk indices for each instrument
         listindxchuninst=None, \
         
         ## input dictionary for lygos                                
         dictlygoinpt=None, \
         
         ## time limits to mask
         listlimttimemask=None, \
        
         # analyses
         ## list of types of analyses for time series data
         ### 'boxsperinega': search for periodic dimmings
         ### 'outlperi': search for periodic outliers
         ### 'boxsperiposi': search for periodic increases
         ### 'lspe': search for sinusoidal variability
         ### 'mfil': matched filter
         listtypeanls=None, \
         
         ## transit search
         ### input dictionary to the search pipeline for periodic boxes
         dictboxsperiinpt=None, \
        
         ### input dictionary to the search pipeline for single transits
         dictsrchtransinginpt=None, \
         
         ## flare search
         ### threshold percentile for detecting stellar flares
         thrssigmflar=7., \

         ### input dictionary to the search pipeline for flares
         dictsrchflarinpt=dict(), \
   
         # Boolean flag to search for flares
         boolsrchflar=None, \
        
         # fitting
         # Boolean flag to reject the lowest log-likelihood during log-likelihood calculation
         boolrejeoutlllik=False, \
            
         # dictionary of system magnitudes
         dictmagtsyst=dict(), \

         # model
         # type of inference
         ## 'samp': sample from the posterior
         ## 'opti': optimize the likelihood
         typeinfe='samp', \

         # list of types of models for time series data
         ## typemodl
         ### 'PlanetarySystem': gravitationally bound system of a star and potentially transiting planets
         ### 'psysphas': gravitationally bound system of a star and potentially transiting planets with phase modulations
         ### 'ssys': gravitationally bound system of potentially transiting two stars
         ### 'CompactObjectStellarCompanion': gravitationally bound system of a star and potentially transiting compact companion
         ### 'StarFlaring': stellar flare
         ### 'AGN': AGN
         ### 'SpottedStar': stellar spot
         ### 'stargpro': star with variability described by a Gaussian Process

         # stellar limb darkening
         ## a string indicating how limb darkening coefficients change across energies
         ### 'cons': constant at all energies
         ### 'line': linear change across energies
         ### 'ener': free at all energies
         #typemodllmdkener='cons', \
         ## a string indicating how limb darkening coefficients change across energies
         ### 'cons': constant at all angles
         ### 'line': linear in the cosine of the angle between the observer and the surface normal (i.e., gamma)
         ### 'quad': quadratic in the cosine of the angle between the observer and the surface normal (i.e., gamma)
         #typemodllmdkterm='quad', \
         
         ## flare model
         ### type of model for finding flares
         #typemodlflar='outl', \

         # limits of time between which the fit is performed
         limttimefitt=None, \

         ## transit model
         ## dilution: None (no correction), 'lygos' for estimation via lygos, or float value
         dilu=None, \
         
         ## priors
         ### baseline detrending
         #### minimum time interval for breaking the time-series into regions, which will be detrended separately
         timebrekregi=0.1, \
         #### Boolean flag to break the time-series into regions
         boolbrekregi=False, \

         #### type of the baseline model
         typebdtr='GaussianProcess', \
         #### order of the spline
         ordrspln=3, \
         #### time scale for median-filtering detrending [days]
         timescalbdtrmedi=2., \
         #### time scale for spline baseline detrending [days]
         listtimescalbdtr=[2.], \

         ### maximum frequency (per day) for LS periodogram
         maxmfreqlspe=None, \
         
         # threshold SNR for triggering the periodic box search pipeline
         thrss2nrcosc=10., \
                
         # threshold LS periodogram power for disposing the target as positive
         thrslspecosc=0.2, \
                
         ### type of priors for stars: 'TICID', 'exar', 'inpt'
         typepriostar=None, \

         # type of prior model parameters for companions
         typepriocomp=None, \
         
         ### photometric and RV model
         #### means
         rratcompprio=None, \
         rsmacompprio=None, \
         epocmtracompprio=None, \
         pericompprio=None, \
         cosicompprio=None, \
         ecoscompprio=None, \
         esincompprio=None, \
         rvelsemaprio=None, \
         #### uncertainties
         stdvrratcompprio=None, \
         stdvrsmacompprio=None, \
         stdvepocmtracompprio=None, \
         stdvpericompprio=None, \
         stdvcosicompprio=None, \
         stdvecoscompprio=None, \
         stdvesincompprio=None, \
         stdvrvelsemaprio=None, \
        
         ### others 
         #### mean
         projoblqprio=None, \
         #### uncertainties
         stdvprojoblqprio=None, \

         #radistar=None, \
         #massstar=None, \
         #tmptstar=None, \
         #rascstar=None, \
         #declstar=None, \
         #vsiistar=None, \
         
         # distance to the system
         #distsyst=None, \

         # magnitudes
         ## TESS
         tmagsyst=None, \
         umagsyst=None, \
         gmagsyst=None, \
         rmagsyst=None, \
         imagsyst=None, \
         zmagsyst=None, \
         ymagsyst=None, \
         vmagsyst=None, \
         jmagsyst=None, \
         hmagsyst=None, \
         kmagsyst=None, \

         #stdvradistar=None, \
         #stdvmassstar=None, \
         #stdvtmptstar=None, \
         #stdvrascstar=None, \
         #stdvdeclstar=None, \
         #stdvvsiistar=None, \
        
         # Boolean flag to perform initial analyses
         boolanls=True, \

         # Boolean flag to perform phase-folding based on model priors
         boolfoldprio=True, \

         # Boolean flag to perform inference based on model priors
         boolfitt=True, \

         ## Boolean flag to model the out-of-transit data to learn a background model
         boolallebkgdgaus=False, \
         # output
         ### list of offsets for the planet annotations in the TSM/ESM plot
         offstextatmoraditmpt=None, \
         offstextatmoradimetr=None, \
         
         # exoplanet specifics
         # planet names
         # string to pull the priors from the NASA Exoplanet Archive
         strgexar=None, \

         ## list of letters to be assigned to planets
         liststrgcomp=None, \
         
         # energy scale over which to detrend the inferred spectrum
         enerscalbdtr=None, \

         # factor to scale the size of text in the figures
         factsizetextfigr=1., \

         ## list of colors to be assigned to planets
         listcolrcomp=None, \
         
         ## Boolean flag to assign them letters *after* ordering them in orbital period, unless liststrgcomp is specified by the user
         boolordrplanname=True, \
        
         # population contexualization
         ## Boolean flag to include the ExoFOP catalog in the comparisons to exoplanet population
         boolexofpopl=True, \
        
         # Boolean flag to ignore any existing file and overwrite
         boolwritover=False, \
         
         # Boolean flag to diagnose the code using potentially computationally-expensive sanity checks, which may slow down the execution
         ## diagnostic mode is always on by default, which should be turned off during large-scale runs, where speed is a concern
         booldiag=True, \
         
         # type of plot background
         typeplotback='black', \

         # type of verbosity
         ## -1: absolutely no text
         ##  0: no text output except critical warnings
         ##  1: minimal description of the execution
         ##  2: detailed description of the execution
         typeverb=1, \

        ):
    '''
    Main function of the miletos pipeline.
    '''
    
    # construct global object
    gdat = tdpy.gdatstrt()
    
    # copy locals (inputs) to the global object
    dictinpt = dict(locals())
    for attr, valu in dictinpt.items():
        if '__' not in attr and attr != 'gdat':
            setattr(gdat, attr, valu)

    # paths
    ## path of the miletos data folder
    gdat.pathbasemile = os.environ['MILETOS_DATA_PATH'] + '/'
    ## base path of the run
    if gdat.pathbase is None:
        gdat.pathbase = gdat.pathbasemile
    
    # measure initial time
    gdat.timeinit = modutime.time()

    # string for date and time
    gdat.strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if gdat.typeverb > 0:
        print('miletos initialized at %s...' % gdat.strgtimestmp)
    
    # check input arguments
    if not gdat.boolplot and gdat.boolplottser:
        raise Exception('')
    
    if gdat.boolplotvisi and not gdat.boolcalcvisi:
        raise Exception('')
    
    if gdat.boolplotvisi is None:
        gdat.boolplotvisi = gdat.boolcalcvisi
    
    if gdat.typeplotback == 'black':
        plt.style.use('dark_background')
        
    # paths
    gdat.pathbaselygo = os.environ['LYGOS_DATA_PATH'] + '/'
    
    # check input arguments
    if not (gdat.pathtarg is not None and gdat.pathbase is None and gdat.pathdatatarg is None and gdat.pathvisutarg is None or \
            gdat.pathtarg is None and gdat.pathbase is not None and gdat.pathdatatarg is None and gdat.pathvisutarg is None or \
            gdat.pathtarg is None and gdat.pathbase is None and gdat.pathdatatarg is not None and gdat.pathvisutarg is not None):
        print('gdat.pathtarg')
        print(gdat.pathtarg)
        print('gdat.pathbase')
        print(gdat.pathbase)
        print('gdat.pathdatatarg')
        print(gdat.pathdatatarg)
        print('gdat.pathvisutarg')
        print(gdat.pathvisutarg)
        raise Exception('')
    
    ## ensure that target and star coordinates are not provided separately
    if gdat.rasctarg is not None and gdat.rascstar is not None:
        raise Exception('')
    if gdat.decltarg is not None and gdat.declstar is not None:
        raise Exception('')

    # human-readable labels of the instruments
    if gdat.listlablinst is None:
        gdat.listlablinst = [['TESS'], []]
    
    if gdat.typeverb > 1:
        print('gdat.listlablinst')
        print(gdat.listlablinst)
    
    setup_miletos(gdat)
    
    if gdat.booldiag:
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                if len(gdat.listlablinst[b][p]) == 0:
                    print('')
                    print('')
                    print('')
                    raise Exception('gdat.listlablinst[b][p] is empty.')
        
        if len(gdat.listlablinst) != 2:
            print('')
            print('')
            print('')
            raise Exception('gdat.listlablinst should be a list with two elements.')

    # strings for the instruments to be used in file names
    gdat.liststrginst = retr_strginst(gdat.listlablinst)
    
    gdat.liststrgband = gdat.liststrginst[0]# + ['Bolometric']
    gdat.numbband = len(gdat.liststrgband)
    gdat.indxband = np.arange(gdat.numbband)

    setup1_miletos(gdat)

    # if either of dictfitt or dicttrue is defined, mirror it to the other
    if gdat.dicttrue is None and gdat.dictfitt is None:
        if gdat.boolsimusome:
            gdat.dicttrue = dict()
        gdat.dictfitt = dict()
    elif gdat.dicttrue is None and gdat.dictfitt is not None:
        if gdat.boolsimusome:
            gdat.dicttrue = gdat.dictfitt
    elif gdat.dicttrue is not None and gdat.dictfitt is None:
        gdat.dictfitt = gdat.dicttrue
    
    if gdat.typeverb > 1:
        print('gdat.dicttrue')
        print(gdat.dicttrue)
        print('gdat.dictfitt')
        print(gdat.dictfitt)

    if gdat.booldiag:
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                if not gdat.liststrgtypedata[b][p] in ['simutargsynt', 'simutargpartsynt', 'simutargpartfprt', 'simutargpartinje', 'obsd']:
                    print('')
                    print('')
                    print('')
                    print('gdat.liststrgtypedata[b][p]')
                    print(gdat.liststrgtypedata[b][p])
                    raise Exception('Undefined entry in liststrgtypedata')

        if gdat.boolsimusome and not 'typemodl' in gdat.dicttrue:
            print('')
            print('')
            print('')
            print('gdat.dicttrue')
            print(gdat.dicttrue)
            raise Exception('Some of the data are simulated, but dicttrue does not have typemodl.')
        
        # check that if the data type for one instrument is synthetic target, then the data type for all instruments should be a synthetic target
        if gdat.booltargsynt:
            for b in gdat.indxdatatser:
                for p in gdat.indxinst[b]:
                    if gdat.liststrgtypedata[b][p] != 'simutargsynt':
                        print('')
                        print('')
                        print('')
                        print('gdat.booltargsynt')
                        print(gdat.booltargsynt)
                        print('gdat.liststrgtypedata')
                        print(gdat.liststrgtypedata)
                        raise Exception('If liststrgtypedata contains one simutargsynt then all data types should be simutargsynt.')

    gdat.arrytser = dict()
    ## ensure target identifiers are not conflicting
    if gdat.listarrytser is None:
        gdat.listarrytser = dict()
        if not gdat.booltargsynt and gdat.ticitarg is None and gdat.strgmast is None and gdat.toiitarg is None and (gdat.rasctarg is None or gdat.decltarg is None):
            raise Exception('Target is not sythetic and no TIC ID (ticitarg), RA&DEC (rasctarg and decltarg), MAST key (strgmast) or a TOI ID (toiitarg) was provided.')
        if gdat.ticitarg is not None and (gdat.strgmast is not None or gdat.toiitarg is not None or gdat.rasctarg is not None or gdat.decltarg is not None):
            raise Exception('Either a TIC ID (ticitarg), RA&DEC (rasctarg and decltarg), MAST key (strgmast) or a TOI ID (toiitarg) should be provided.')
        if gdat.strgmast is not None and (gdat.ticitarg is not None or gdat.toiitarg is not None or gdat.rasctarg is not None or gdat.decltarg is not None):
            raise Exception('Either a TIC ID (ticitarg), RA&DEC (rasctarg and decltarg), MAST key (strgmast) or a TOI ID (toiitarg) should be provided.')
        if gdat.toiitarg is not None and (gdat.strgmast is not None or gdat.ticitarg is not None or gdat.rasctarg is not None or gdat.decltarg is not None):
            raise Exception('Either a TIC ID (ticitarg), RA&DEC (rasctarg and decltarg), MAST key (strgmast) or a TOI ID (toiitarg) should be provided.')
        if gdat.strgmast is not None and (gdat.ticitarg is not None or gdat.toiitarg is not None or gdat.rasctarg is not None or gdat.decltarg is not None):
            raise Exception('Either a TIC ID (ticitarg), RA&DEC (rasctarg and decltarg), MAST key (strgmast) or a TOI ID (toiitarg) should be provided.')
    
    # dictionary to be returned
    gdat.dictmileoutp = dict()
    
    # instrument index of TESS
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            if gdat.listlablinst[b][p] == 'TESS':
                gdat.indxinsttess = p

    # number of energy bins for each photometric data set
    gdat.numbener = [[] for p in gdat.indxinst[0]]
    
    if gdat.booldiag:
        for b in gdat.indxdatatser:
            if len(gdat.liststrgtypedata[b]) != len(gdat.liststrginst[b]):
                print('')
                print('')
                print('')
                print('b')
                print(b)
                print('gdat.liststrgtypedata')
                print(gdat.liststrgtypedata)
                print('gdat.liststrginst')
                print(gdat.liststrginst)
                raise Exception('len(gdat.liststrgtypedata[b]) != len(gdat.liststrginst[b])')
    
    if gdat.listtimescalbdtr is not None:
        gdat.liststrgtimescalbdtr = []
        for timescalbdtr in gdat.listtimescalbdtr:
            facttime, lablunittime = tdpy.retr_timeunitdays(timescalbdtr)
            gdat.liststrgtimescalbdtr.append('%.3g%s' % (facttime * timescalbdtr, lablunittime))
                        
    gdat.listindxinst = [[] for b in gdat.indxdatatser]
    for b in gdat.indxdatatser:
        gdat.listindxinst[b] = np.arange(len(gdat.liststrginst[b]))

    if gdat.booldiag:
        if not isinstance(gdat.liststrginst[0], list):
            raise Exception('')
        if not isinstance(gdat.liststrginst[1], list):
            raise Exception('')
        if not isinstance(gdat.listlablinst[0], list):
            raise Exception('')
        if not isinstance(gdat.listlablinst[1], list):
            raise Exception('')
        
        for b in gdat.indxdatatser:
            if len(gdat.liststrginst[b]) != len(gdat.listlablinst[b]):
                print('')
                print('')
                print('')
                print('gdat.liststrginst')
                print(gdat.liststrginst)
                print('gdat.listlablinst')
                print(gdat.listlablinst)
                raise Exception('len(gdat.liststrginst[b]) != len(gdat.listlablinst[b])')
    
    ## Boolean flag to query IRSA for ZTF
    gdat.boolretrlcurzwtf = False
    
    ## Boolean flag to query MAST
    gdat.boolretrlcurmast = [[False for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gdat.boolretrlcurmastanyy = False
    
    # Boolean flag to indicate all data is simulated
    gdat.boolsimutotl = True

    gdat.booltargpartanyy = False
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            
            if gdat.liststrgtypedata[b][p] == 'obsd':
                gdat.boolsimutotl = False

            if gdat.liststrgtypedata[b][p] == 'simutargpartsynt' or gdat.liststrgtypedata[b][p] == 'simutargpartfprt' or \
                                                                    gdat.liststrgtypedata[b][p] == 'simutargpartinje' or gdat.liststrgtypedata[b][p] == 'obsd':
                gdat.booltargpartanyy = True

                if gdat.liststrginst[b][p] in ['K2', 'Kepler', 'HST'] or gdat.liststrginst[b][p].startswith('JWST') or gdat.liststrginst[b][p] == 'TESS' \
                                                                                                        and gdat.boolutilcadehigh and gdat.typelcurtpxftess != 'lygos':
                    gdat.boolretrlcurmast[b][p] = True
                    gdat.boolretrlcurmastanyy = True
                if gdat.liststrginst[b][p] == 'ZTF':
                    gdat.boolretrlcurzwtf = True
    
    if typeplotback == 'white':
        colrbkgd = 'white'
        colrdraw = 'black'
    elif typeplotback == 'black':
        colrbkgd = 'black'
        colrdraw = 'white'
    
    # decide whether to run in offline mode
    if gdat.boolforcoffl:
        gdat.boolexecoffl = True
    else:
        #import urllib
        import urllib.request
        
        def connect(host='http://google.com'):
            try:
                urllib.request.urlopen(host)
                return False
            except:
                return True
        # test
        gdat.boolexecoffl = connect()

        if gdat.boolexecoffl:
            print('Not connected to the internet')
        else:
            print('Connected to the internet.')

    if gdat.booldiag:
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                if gdat.boolretrlcurmast[b][p] and gdat.boolexecoffl:
                    print('')
                    print('')
                    print('')
                    print('b, p')
                    print(b, p)
                    print('gdat.liststrgtypedata')
                    print(gdat.liststrgtypedata)
                    print('gdat.liststrginst')
                    print(gdat.liststrginst)
                    raise Exception('gdat.boolretrlcurmast[b][p] AND boolexecoffl is True.')

    gdat.fitt = tdpy.gdatstrt()
    if gdat.dictfitt is not None:
        for name, valu in gdat.dictfitt.items():
            setattr(gdat.fitt, name, valu)
    
    gdat.boolmodl = gdat.boolfoldprio or gdat.boolfitt

    # list of models to be fitted to the data
    gdat.liststrgmodl = []
    if gdat.boolmodl:
        gdat.liststrgmodl += ['fitt']
    if gdat.boolsimusome:
        gdat.liststrgmodl += ['true']
    
        gdat.true = tdpy.gdatstrt()
        gdat.true.prio = tdpy.gdatstrt()
        gdat.true.prio.meanpara = tdpy.gdatstrt()
    
        if not hasattr(gdat.true, 'boolsampsystnico'):
            if 'pericomp' in gdat.dicttrue:
                gdat.true.boolsampsystnico = False
            else:
                gdat.true.boolsampsystnico = True

        if gdat.dicttrue is not None:
            print('Transferring the contents of dicttrue to gdat...')
            for name, valu in gdat.dicttrue.items():
                setattr(gdat.true, name, valu)

    gdat.maxmradisrchmast = 2. # arcsec
    gdat.strgradi = '%gs' % gdat.maxmradisrchmast
    
    if gdat.typeverb > 0:
        if gdat.boolmodl:
            print('Type of fitting model: %s' % gdat.fitt.typemodl)
        if not gdat.boolfitt:
            print('No fitting will be performed.')

    if gdat.boolmodl:
        setp_modlmedi(gdat, 'fitt')
    
    if gdat.boolplottser is None:
        gdat.boolplottser = gdat.boolplot
    
    if gdat.boolplotdvrp is None:
        gdat.boolplotdvrp = gdat.boolplot
    
    # list of list of dictionaries holding the paths and DV report positions of plots
    if gdat.boolplot:
        gdat.listdictdvrp = [[]]
        
    # conversion factors
    gdat.dictfact = tdpy.retr_factconv()

    # settings
    ## plotting
    gdat.numbcyclcolrplot = 300
    gdat.alphdata = 0.2
    
    if gdat.lablener is None:
        gdat.lablener = 'Wavelength'
    
    if gdat.enerscalbdtr is None:
        gdat.enerscalbdtr = 0.1 # [um]

    gdat.figrsize = [6, 4]
    gdat.figrsizeydob = [8., 4.]
    gdat.figrsizeydobskin = [8., 2.5]
        
    gdat.listfeatstar = ['radistar', 'massstar', 'tmptstar', 'rascstar', 'declstar', 'vsiistar', 'jmagsyst']
    gdat.listfeatstarpopl = ['radicomp', 'masscomp', 'tmptplan', 'radistar', 'jmagsyst', 'kmagsyst', 'tmptstar']
    
    # auxiliary array of incides of types of different dependences of radius ratio on bands
    gdat.indxrratband = [gdat.indxinst[0], []]
        
    for strgmodl in gdat.liststrgmodl:
        setp_modlinit(gdat, strgmodl)
    
    print('gdat.liststrgmodl')
    print(gdat.liststrgmodl)

    if gdat.boolplotpopl:
        gdat.liststrgpopl = []
        if gdat.boolpriotargpsys or gdat.fitt.boolmodlpsys or gdat.boolsimusome and gdat.true.boolmodlpsys:
            gdat.liststrgpopl += ['exar']
            if gdat.toiitarg is not None:
                gdat.liststrgpopl += ['exof']
        gdat.numbpopl = len(gdat.liststrgpopl)
    
    if gdat.toiitarg is not None or gdat.ticitarg is not None \
                                                               and (gdat.boolmodl and gdat.fitt.boolmodlpsys or gdat.boolsimusome and gdat.true.boolmodlpsys):
        gdat.dicttoii = nicomedia.retr_dicttoii()

    if gdat.boolfitt:
        # model
        # type of likelihood
        ## 'sing': assume model is a single realization
        ## 'GaussianProcess': assume model is a Gaussian Process (GP)
        if gdat.fitt.typemodlblinshap == 'GaussianProcess':
            gdat.typellik = 'GaussianProcess'
        else:
            gdat.typellik = 'sing'

    # determine target identifiers
    if gdat.ticitarg is not None:
        gdat.typetarg = 'TICID'
        if gdat.typeverb > 0:
            print('A TIC ID was provided as target identifier.')
        
        # check if this TIC is a TOI
        if gdat.fitt.boolmodlpsys or gdat.boolsimusome and gdat.true.boolmodlpsys:
            gdat.toiitarg = nicomedia.retr_toiitici(gdat.ticitarg, typeverb=gdat.typeverb)

        gdat.strgmast = 'TIC %d' % gdat.ticitarg

    elif gdat.toiitarg is not None:
        gdat.typetarg = 'TOIID'
        
        if gdat.toiitarg - int(gdat.toiitarg) == 0:
            if gdat.typeverb > 0:
                print('A TOI host (%d) was provided as the target identifier.' % gdat.toiitarg)
            gdat.typetoiitarg = 'host'
        else:
            if gdat.typeverb > 0:
                print('A TOI (%d) was provided as target identifier.' % gdat.toiitarg)
            gdat.typetoiitarg = 'toii'
        
        # determine TIC ID
        gdat.strgtoiibase = str(gdat.toiitarg)
        indx = []
        print('gdat.strgtoiibase')
        print(gdat.strgtoiibase)
        for k, strg in enumerate(gdat.dicttoii['TOIID'][0]):
            
            if gdat.typetoiitarg == 'toii':
                strgcomp = str(strg)
            else:
                strgcomp = str(strg).split('.')[0]
            
            if strgcomp == gdat.strgtoiibase:
                indx.append(k)
        indx = np.array(indx)
        if indx.size == 0:
            print('')
            print('')
            print('')
            print('gdat.toiitarg')
            print(gdat.toiitarg)
            print('gdat.dicttoii[TOI][0]')
            summgene(gdat.dicttoii['TOIID'][0])
            raise Exception('Did not find the TOI in the ExoFOP-TESS TOI list.')

        print('indx')
        print(indx)
        gdat.ticitarg = gdat.dicttoii['TICID'][0][indx[0]]

        if gdat.strgexar is None:
            gdat.strgexar = 'TOI-%d' % gdat.toiitarg
        gdat.strgmast = 'TIC %d' % gdat.ticitarg

    elif gdat.strgmast is not None:
        gdat.typetarg = 'MASTKey'
        if gdat.typeverb > 0:
            print('A MAST key (%s) was provided as target identifier.' % gdat.strgmast)

    elif gdat.rasctarg is not None and gdat.decltarg is not None:
        gdat.typetarg = 'Position'
        if gdat.typeverb > 0:
            print('RA and DEC (%g %g) are provided as target identifier.' % (gdat.rasctarg, gdat.decltarg))
        gdat.strgmast = '%g %g' % (gdat.rasctarg, gdat.decltarg)
    elif gdat.listarrytser is not None:
        gdat.typetarg = 'inptdata'

        if gdat.labltarg is None:
            raise Exception('')
    else:
        # synthetic target
        gdat.typetarg = 'synt'
    
    if gdat.boolmodl:
        gdat.fitt.prio = tdpy.gdatstrt()
        gdat.fitt.prio.meanpara = tdpy.gdatstrt()
        gdat.fitt.prio.numbcomp = None
    
    if gdat.dictfitt is not None:
        if hasattr(gdat.dictfitt, 'prio'):
            for namepara in gdat.dictfitt['prio']:
                gdat.fitt.prio.meanpara = gdat.dictfitt['prio'][namepara]
                
    # Boolean flag indicating whether MAST has been searched already
    gdat.boolsrchmastdone = False
    
    print('gdat.typetarg')
    print(gdat.typetarg)
    print('gdat.boolsrchmastdone')
    print(gdat.boolsrchmastdone)
    print('gdat.boolexecoffl')
    print(gdat.boolexecoffl)
    
    if not gdat.booltargsynt and (gdat.typetarg == 'TICID' or gdat.typetarg == 'TOIID' or gdat.typetarg == 'MASTKey') and not gdat.boolsrchmastdone and not gdat.boolexecoffl:
        # temp -- check that the closest TIC to a given TIC is itself
        if gdat.typeverb > 0:
            print('Querying the TIC on MAST with keyword %s within %s as to get the RA, DEC, Tmag, and TIC ID of the closest source...' % (gdat.strgmast, gdat.strgradi))
        listdictticinear = astroquery.mast.Catalogs.query_region(gdat.strgmast, radius=gdat.strgradi, catalog="TIC")
        gdat.boolsrchmastdone = True
        if gdat.typeverb > 0:
            print('Found %d TIC sources.' % len(listdictticinear))
        if len(listdictticinear) > 0:
            maxmanglmtch = 0.2 # [arcsecond]
            if listdictticinear[0]['dstArcSec'] < maxmanglmtch:
                print('The closest match via the MAST query was within %.g arcseconds. Will associate the TIC ID of the closest match to the target.' % maxmanglmtch)
                gdat.ticitarg = int(listdictticinear[0]['ID'])
                gdat.rasctarg = listdictticinear[0]['ra']
                gdat.decltarg = listdictticinear[0]['dec']
                gdat.tmagtarg = listdictticinear[0]['Tmag']
            else:
                print('The closest match via the MAST query was not within %.g arcseconds. Will not associate the TIC ID of the closest match to the target.' % maxmanglmtch)
        else:
            print('There was not match via the MAST query. Will not associate the TIC ID of the closest match to the target.')

    if gdat.typeverb > 0:
        print('gdat.typetarg')
        print(gdat.typetarg)

    if gdat.listipntinpt is not None and gdat.typetarg != 'inpt':
        raise Exception('List of pointings can only be input when typetarg is "inpt".')
    
    # check if any GPU is available
    try:
        import GPUtil
        temp = GPUtil.getGPUs()
        if len(temp) == 0:
            print('No GPU is detected...')
    except:
        pass
        print('temp: check if this can be done without try ... except.')

    gdat.maxmnumbiterbdtr = 5
    
    if gdat.typeverb > 0:
        print('gdat.ticitarg')
        print(gdat.ticitarg)
        print('gdat.strgmast')
        print(gdat.strgmast)
        print('gdat.rasctarg')
        print(gdat.rasctarg)
        print('gdat.decltarg')
        print(gdat.decltarg)
        print('gdat.toiitarg')
        print(gdat.toiitarg)
    
    # priors
    if gdat.typepriostar is None:
        if hasattr(gdat.fitt, 'radistar'):
            gdat.typepriostar = 'inpt'
        else:
            gdat.typepriostar = 'TICID'
    
    if gdat.boolmodl:
        if gdat.typeverb > 0:
            if gdat.fitt.boolmodlpsys:
                print('Stellar parameter prior type: %s' % gdat.typepriostar)
    
    # number of Boolean signal outputs
    gdat.numbtypeposi = 4
    gdat.indxtypeposi = np.arange(gdat.numbtypeposi)
    
    if gdat.typeverb > 0:
        print('gdat.boolplottser')
        print(gdat.boolplottser)
    
    if gdat.typeverb > 0:
        print('boolplotpopl')
        print(boolplotpopl)
    
    ## NASA Exoplanet Archive
    if gdat.boolplotpopl:
        gdat.dictexar = nicomedia.retr_dictexar(strgelem='comp', typeverb=gdat.typeverb)
    
    if gdat.strgclus is None:
        gdat.pathclus = gdat.pathbase
    else:
        if gdat.typeverb > 0:
            print('gdat.strgclus')
            print(gdat.strgclus)
        # data path for the cluster of targets
        gdat.pathclus = gdat.pathbase + '%s/' % gdat.strgclus
        gdat.pathdataclus = gdat.pathclus + 'data/'
        gdat.pathvisuclus = gdat.pathclus + 'visuals/'
        
        if gdat.typeverb > 0:
            print('Path for the cluster:')
            print(gdat.pathclus)

    if gdat.labltarg is None:
        if gdat.typetarg == 'MASTKey':
            gdat.labltarg = gdat.strgmast
        if gdat.typetarg == 'TOIID':
            gdat.labltarg = 'TOI-%d' % gdat.toiitarg
        if gdat.typetarg == 'TICID':
            gdat.labltarg = 'TIC %d' % gdat.ticitarg
        if gdat.typetarg == 'Position':
            gdat.labltarg = 'RA=%.4g, DEC=%.4g' % (gdat.rasctarg, gdat.decltarg)
        if gdat.typetarg == 'synt':
            gdat.labltarg = 'Sim Target'
    
    if gdat.typeverb > 0:
        print('gdat.labltarg')
        print(gdat.labltarg)
    
    # the string that describes the target
    if gdat.strgtarg is None:
        gdat.strgtarg = ''.join(gdat.labltarg.split(' '))
    
    gdat.lliktemp = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    
    if gdat.strgcnfg is None:
        if gdat.lablcnfg is None:
            gdat.strgcnfg = ''
        else:
            gdat.strgcnfg = ''.join(gdat.lablcnfg.split(' '))
    
    # the path for the target
    if gdat.pathtarg is None:
        
        gdat.pathtarg = gdat.pathclus + '%s/' % (gdat.strgtarg)
        
        if gdat.strgcnfg is None or gdat.strgcnfg == '':
            strgcnfgtemp = ''
        else:
            strgcnfgtemp = gdat.strgcnfg + '/'
        
        gdat.pathtargcnfg = gdat.pathtarg + strgcnfgtemp
        
        if gdat.booldiag:
            if gdat.pathtargcnfg.endswith('//'):
                print('')
                print('')
                print('')
                print('gdat.pathtargcnfg')
                print(gdat.pathtargcnfg)
                print('strgcnfgtemp')
                print(strgcnfgtemp)
                raise Exception('')
        
        gdat.pathdatatarg = gdat.pathtargcnfg + 'data/'
        gdat.pathvisutarg = gdat.pathtargcnfg + 'visuals/'

        if gdat.typeverb > 0:
            print('Path for this run configuration on the target:')
            print(gdat.pathtargcnfg)

    if gdat.typeverb > 0:
        print('gdat.strgtarg')
        print(gdat.strgtarg)
    
    # check if the run has been completed before
    path = gdat.pathdatatarg + 'dict_miletos_output.pickle'
    if not gdat.boolwritover and os.path.exists(path):
        
        if gdat.typeverb > 0:
            print('Reading from %s...' % path)
        with open(path, 'rb') as objthand:
            gdat.dictmileoutp = pickle.load(objthand)
        
        return gdat.dictmileoutp

    ## make folders
    for attr, valu in gdat.__dict__.items():
        if attr.startswith('path') and valu is not None and not isinstance(valu, dict) and valu.endswith('/'):
            os.system('mkdir -p %s' % valu)
            
    if gdat.booltargsynt:
        
        # determine magnitudes
        if hasattr(gdat.true, 'distsyst') and hasattr(gdat.true, 'tmptstar') and hasattr(gdat.true, 'radistar'):
            
            dictfluxband, _ = nicomedia.retr_dictfluxband(gdat.dicttrue['tmptstar'], gdat.liststrgband)

    if gdat.strgtarg == '' or gdat.strgtarg is None or gdat.strgtarg == 'None' or len(gdat.strgtarg) == 0:
        raise Exception('')
    
    for name in ['strgtarg', 'pathtarg']:
        gdat.dictmileoutp[name] = getattr(gdat, name)

    if gdat.listener is None:
        gdat.listener = [[] for p in gdat.indxinst[0]]
    else:
        if gdat.booldiag:
            if np.isscalar(gdat.listener[0]):
                print('')
                print('gdat.listener should be a list of arrays.')
                print('gdat.listener')
                print(gdat.listener)
                raise Exception('')
    
    if gdat.typeverb > 0:
        print('gdat.numbinst')
        print(gdat.numbinst)
        print('gdat.strgcnfg')
        print(gdat.strgcnfg)
        print('gdat.liststrginst')
        print(gdat.liststrginst)
    
    if gdat.booldiag:
        if gdat.strgcnfg == '_':
            print('')
            print('')
            print('')
            raise Exception('')

    gdat.pathalle = dict()
    gdat.objtalle = dict()
        
    gdat.indxenerclip = 0

    # Boolean flag to execute a search for flares
    if gdat.boolsrchflar is None:
        if gdat.boolmodl and gdat.fitt.typemodl == 'StarFlaring':
            gdat.boolsrchflar = True
        else:
            gdat.boolsrchflar = False
    
    if gdat.typeverb > 0:
        print('gdat.boolsimusome')
        print(gdat.boolsimusome)
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                print('gdat.boolretrlcurmast[b][p]')
                print(gdat.boolretrlcurmast[b][p])
    
    if gdat.boolretrlcurzwtf:
        fram = lightcurve.LCQuery.download_data(circle=[gdat.rasctarg, gdat.decltarg, 0.01])

        print('listrascdata')
        summgene(listrascdata)
        
        magt = fram['mag'].to_numpy()
        time = fram['mjd'].to_numpy()
        
        print('time')
        summgene(time)
        print('magt')
        summgene(magt)
    
    gdat.booltess = 'TESS' in gdat.liststrginst[0]
    gdat.booltesskepl = 'Kepler' in gdat.liststrginst[0] or 'TESS' in gdat.liststrginst[0] or 'K2' in gdat.liststrginst[0]
    
    # determine the MAST keyword to be used for the target
    if not gdat.booltargsynt:
        if gdat.strgmast is not None:
            strgmasttemp = gdat.strgmast
        elif gdat.rasctarg is not None:
            strgmasttemp = '%g %g' % (gdat.rasctarg, gdat.decltarg)
        elif gdat.ticitarg is not None:
            strgmasttemp = 'TIC %d' % gdat.ticitarg
        else:
            print('gdat.strgmast')
            print(gdat.strgmast)
            print('gdat.rasctarg')
            print(gdat.rasctarg)
            print('gdat.ticitarg')
            print(gdat.ticitarg)
            raise Exception('')
        
        if gdat.booltess:
            strgtcut = strgmasttemp
            # get the list of sectors for which TESS FFI data are available via TESSCut
            gdat.listtsectcut, temp, temp = retr_listtsectcut(strgtcut)
        
            print('List of TESS sectors for which FFI data are available via TESSCut:')
            print(gdat.listtsectcut)
        
    if (gdat.strgmast is not None or gdat.ticitarg is not None or rasctarg is not None or gdat.toiitarg is not None) and gdat.boolexecoffl and not gdat.boolsimutotl:
        print('')
        print('')
        print('')
        raise Exception('(gdat.strgmast is not None or gdat.ticitarg is not None or rasctarg is not None) AND boolexecoffl is True AND not gdat.boolsimutotl.')
    
    print('gdat.booltesskepl')
    print(gdat.booltesskepl)

    if gdat.boolretrlcurmastanyy:
        
        #if gdat.liststrginst is None:
        #    gdat.liststrginst = ['TESS', 'Kepler', 'K2', 'JWST_NIRSpec']
        
        gdat.pathdatamast = os.environ['MAST_DATA_PATH'] + '/'
        os.system('mkdir -p %s' % gdat.pathdatamast)
        
        if gdat.boolutiltesslocl:
            # determine the TIC ID to be used for the MAST search
            ticitsec = None
            if gdat.ticitarg is None:
                print('Will determine the TIC ID of the target using MAST keyword %s.' % strgmasttemp)
                print('Querying the TIC on MAST with keyword %s for sources within %s of %s...' % (strgmasttemp, gdat.strgradi))
                listdictticinear = astroquery.mast.Catalogs.query_object(strgmasttemp, catalog='TIC', radius=gdat.strgradi)
                if len(listdictticinear) > 0 and listdictticinear[0]['dstArcSec'] < 1.:
                    print('TIC associated with the search is %d' % ticitsec)
                    ticitsec = int(listdictticinear[0]['ID'])
                else:
                    print('Warning! No TIC match to the MAST keyword: %s' % strgmasttemp)
            
            # get the list of sectors for which TESS SPOC data are available
            listtsec2min, listpathdisk2min = retr_tsecpathlocl(ticitsec)
            print('List of TESS sectors for which SPOC data are available for the TIC ID %d:' % ticitsec)
            print(listtsec2min)
        
        # get observation tables
        if typeverb > 0:
            print('Querying observations on MAST with keyword %s within %g arcseconds...' % (strgmasttemp, gdat.maxmradisrchmast))
        try:
            listtablobsv = astroquery.mast.Observations.query_object(strgmasttemp, radius=gdat.strgradi)
        except:
            print('MAST search failed. Will quit.')
            print('')
            return gdat.dictmileoutp
        
        print('Found %d tables...' % len(listtablobsv))
        listname = list(listtablobsv.keys())
        
        gdat.liststrginstfinl = []
        for strgexpr in gdat.liststrginst[0]:
            if not (strgexpr == 'TESS' and gdat.boolutiltesslocl):
                gdat.liststrginstfinl.append(strgexpr)
        
        gdat.listarrylcurmast = [[] for strgexpr in gdat.liststrginst[0]]
        
        #print('listtablobsv')
        #for name in listname:
        #    print(name)
        #    summgene(listtablobsv[name].value, boolshowuniq=True)
        #    print('')

        gdat.listpathspocmast = []
            
        print('listname')
        print(listname)
        for mm, strgexpr in enumerate(gdat.liststrginstfinl):
            if strgexpr.startswith('JWST'):
                strgexprtemp = 'JWST'
                strgexprsubb = strgexpr[5:]
                if 'g395h' in strgexprsubb:
                    strgexprdete = strgexprsubb[-4:]
                    strgexprsubb = strgexprsubb[:-5]
            else:
                strgexprtemp = strgexpr
            
            #indx = np.where((listtablobsv['obs_collection'] == strgexprtemp) & (listtablobsv['dataproduct_type'] == 'timeseries'))
            boolgood = (listtablobsv['obs_collection'] == strgexprtemp)
            # & (listtablobsv['dataproduct_type'] == 'timeseries'))

            if strgexprtemp == 'TESS':
                
                if gdat.typelcurtpxftess == 'lygos':
                    continue

                gdat.listtsecspoc = []
                #print('gdat.ticitarg')
                #print(gdat.ticitarg)
                #print('type(gdat.ticitarg)')
                #print(type(gdat.ticitarg))
                #print('%s % gdat.ticitarg')
                #print('%s' % gdat.ticitarg)
                #print('listtablobsv[target_name].value')
                #print(listtablobsv['target_name'].value)
                #print('np.unique(listtablobsv[target_name].value)')
                #print(np.unique(listtablobsv['target_name'].value))
                #summgene(listtablobsv['target_name'].value)
                #print('')
                
                if gdat.ticitarg is not None:
                    print('Filtering the observation tables based on whether the TIC ID matches the target...')
                
                    boolgood = boolgood & (listtablobsv['target_name'].value == '%s' % gdat.ticitarg)
                
            if strgexprtemp == 'K2':
                boolgood = boolgood & (listtablobsv['dataproduct_type'] == 'timeseries') & (listtablobsv['target_name'] == gdat.strgmast)
            
            if strgexprtemp == 'JWST':
                boolgood = boolgood & (listtablobsv['provenance_name'] == 'CALJWST') & \
                           (listtablobsv['target_name'] == gdat.strgmast) & \
                           (listtablobsv['calib_level'] == 3) & \
                           (listtablobsv['dataRights'] == 'PUBLIC')
                           #(listtablobsv['obs_id'] == strgexpr[5:]) & \
                
                boolgoodtemp = np.empty_like(boolgood)
                for ll in range(len(boolgoodtemp)):
                    boolgoodtemp[ll] = strgexpr[5:] in listtablobsv['obs_id'][ll]
                boolgood = boolgood & boolgoodtemp

            indx = np.where(boolgood)[0]

            #print('indx')
            #print(indx)
            #print('listtablobsv')
            #print(listtablobsv)

            print('Found %d tables...' % len(listtablobsv[indx]))
            
            if strgexprtemp == 'K2':
                #print('K2 chunks')
                for obid in listtablobsv['obs_id'][indx]:
                    strgchun = obid.split('-')
                    #print('obid')
                    #print(obid)
                    #print('strgchun')
                    #print(strgchun)
                    #if len(strgchun) > 1:# and strgchun[1] != '':
                    #    listtsec2min.append(int(strgchun[1][1:3]))
                    #    #listpath2min

            listname = list(listtablobsv[indx].keys())
            
            #print('')
            #print('')
            #print('')
            #print('')
            #print('')
            #print('')
            #print('')
            #print('')
            #print('')
            #print('')
            #print('')
            #print('')
            #print('listtablobsv')
            #for name in listname:
            #    print(name)
            #    summgene(listtablobsv[name].value, boolshowuniq=True)
            #    print('')
            #
            #print('listname')
            #print(listname)
            #print('len(listname)')
            #print(len(listname))
            
            cntrtess = 0

            if indx.size > 0:
                print('Will get the list of products for each table...')
            
            for k, tablobsv in enumerate(listtablobsv[indx]):
            
                if listtablobsv['distance'][k] > 0:
                    print('Distance of table number %d: %g' % (k, listtablobsv['distance'][k]))
                    continue
                
                print('Table %d...' % k)
                print('Getting the product list for table %d...' % k)
                listprod = astroquery.mast.Observations.get_product_list(tablobsv)
                numbprod = len(listprod)
                print('numbprod')
                print(numbprod)

                listname = list(listprod.keys())
                
                #print('listname')
                #print(listname)
                #print('listprod')
                #for name in listname:
                #    print(name)
                #    summgene(listprod[name].value, boolshowuniq=True)
                #    print('')
                
                boolgood = np.ones(numbprod, dtype=bool)
                if strgexprtemp == 'JWST' or strgexprtemp == 'TESS':
                    boolgoodtemp = np.empty(numbprod, dtype=bool)
                    for kk in range(numbprod):
                        if strgexprtemp == 'JWST':
                            boolgoodtemp[kk] = listprod[kk]['productFilename'].endswith('_x1dints.fits') and strgexprsubb in listprod[kk]['productFilename'] and \
                                                                                                    not '-seg' in listprod[kk]['productFilename']
                    
                        if strgexprtemp == 'TESS':
                            # choose lc.fits instead of tp.fits
                            boolgoodtemp[kk] = listprod[kk]['productFilename'].endswith('lc.fits')
                    
                    #print((listprod['productSubGroupDescription'].value == 'S2D').dtype)
                    #print('listprod[productSubGroupDescription].value == S2D')
                    #print(listprod['productSubGroupDescription'].value == 'S2D')
                    #summgene(listprod['productSubGroupDescription'].value == 'S2D')
                    #print('listprod[productSubGroupDescription].value == X1D')
                    #print(listprod['productSubGroupDescription'].value == 'X1D')
                    #summgene(listprod['productSubGroupDescription'].value == 'X1D')
                    #boolgoodtemp = (listprod['productSubGroupDescription'].value == 'S2D' | listprod['productSubGroupDescription'].value == 'X1D')
                    #boolgoodtemp = (listprod['productSubGroupDescription'] == 'S2D' | listprod['productSubGroupDescription'] == 'X1D')
                    #print('boolgood')
                    #summgene(boolgood)
                    boolgood = boolgood & boolgoodtemp
                
                indxprodgood = np.where(boolgood)[0]
                if indxprodgood.size > 1:
                    if strgexprtemp == 'JWST':
                        print('')
                        print('')
                        print('')
                        print('More than one good product.')
                        for kk in range(indxprodgood.size):
                            print('listprod[indxprodgood[kk]][productFilename]')
                            print(listprod[indxprodgood[kk]]['productFilename'])
                        print('indxprodgood')
                        print(indxprodgood)
                        raise Exception('')
                
                if indxprodgood.size == 0:
                    print('No good product. Skipping the table...')

                #print('listprod')
                #for name in listname:
                #    print(name)
                #    summgene(listprod[name], boolshowuniq=True)
                #    print('')
                #print('')
                #if strgexpr == 'TESS':
                #    print('listprod')
                #    print(listprod)
                #    for a in range(len(listprod)):
                #        print('listprod[a]')
                #        print(listprod[a])
                #        boolfasttemp = listprod[a]['obs_id'].endswith('fast')
                #        if not boolfasttemp:
                #            tsec = int(listprod[a]['obs_id'].split('-')[1][1:])

                print('Downloading products for table number %d...' % k)
                # download data from MAST
                manifest = astroquery.mast.Observations.download_products(listprod[indxprodgood], download_dir=gdat.pathdatamast)
                
                if manifest is not None:
                    for path in manifest['Local Path']:
                        print('Reading from %s...' % path)
                        listhdun = astropy.io.fits.open(path)
                        listhdun.info()
                            
                        #'jw01366-o004_t001_nirspec_clear-prism-s1600a1-sub512_x1dints.fits'
                        #if path.endswith('allslits_x1d.fits') or path.endswith('s1600a1_x1d.fits') or 
                        
                        if 'tess' in path:
                            
                            if 'a_fast' in path:
                                print('temp: removing all fast (20 sec) cadence data...')
                                continue
                            
                            print('Appending path to gdat.listpathspocmast...')
                            gdat.listpathspocmast.append(path)

                            #'aylan/mast/mastDownload/TESS/tess2021091135823-s0037-0000000260647166-0208-s/tess2021091135823-s0037-0000000260647166-0208-s_lc.fits...'
                            #'aylan/mast/mastDownload/TESS/tess2021091135823-s0037-0000000260647166-0208-a_fast/tess2021091135823-s0037-0000000260647166-0208-a_fast-lc.fits...'
                            #pathsplt = path.split('/')
                            #pathtemp = '/'.join((pathsplt[:-2])) + pathsplt[-2][:-1] + 'a_fast/' + pathsplt[-1][:-9] + 'a_fast-lc.fits'
                            #print('pathtemp')
                            #print(pathtemp)
                            #print('manifest[Local Path]')
                            #print(manifest['Local Path'])
                            #if pathtemp in manifest['Local Path']:
                            #    print('This sector is available at higher cadence. Skipping this file...')
                            #    continue

                            arrylcur, tsec, tcam, tccd = \
                                read_tesskplr_file(path, typeinst='TESS', strgtypelcur='PDCSAP_FLUX', pathfoldsave=gdat.pathdatatarg, \
                                                                         booldiag=gdat.booldiag, boolmaskqual=gdat.boolmaskqual, boolnorm=gdat.boolnormphot)
                            
                            gdat.listtsecspoc.append(tsec)
                            gdat.listarrylcurmast[mm].append(arrylcur[:, None, :])
                            
                            if gdat.boolbrektess:
                                gdat.liststrginst[0].append('TESS_S%d' % tsec)
                        
                        elif 'niriss' in path or path.endswith('nis_x1dints.fits'):
                            pass
                            #listtime = listhdun['EXTRACT1D'].data
                            #wlen = listhdun[1].data['WAVELENGTH']
                            #print(listhdun[1].data.names)
                            #print('wlen')
                            #summgene(wlen)
                            #numbtime = len(listhdun[1].data)
                            #arry = np.empty((numbwlen, numbtime))
                            #indxtime = np.arange(numbtime)
                            #arry = listhdun[1].data['FLUX']
                            #print('arry')
                            #summgene(arry)
                        else:
                            
                            listtime = listhdun['INT_TIMES'].data
                            listhdun.info()
                            #print('listtime')
                            #print(listtime)
                            #summgene(listtime)
                            numbtime = len(listtime)
                            if 'g395h' in strgexprsubb:
                                if strgexprdete == 'NRS1':
                                    wlen = listhdun[2].data['WAVELENGTH']
                                if strgexprdete == 'NRS2':
                                    wlen = listhdun[-1].data['WAVELENGTH']
                            else:
                                wlen = listhdun[2].data['WAVELENGTH']
                                
                            numbwlen = wlen.size
                            gdat.listener[mm] = wlen
                            arry = np.empty((numbtime, numbwlen, 3))
                            
                            print('listhdun[2].data.names')
                            print(listhdun[2].data.names)
                            
                            print('numbwlen')
                            print(numbwlen)
                            if not path.endswith('.fits'):
                                raise Exception('')

                            pathmile = path[:-5] + '_mile.npy'
                            if os.path.exists(pathmile):
                                print('Reading from %s...' % pathmile)
                                arry = np.load(pathmile)
                            else:
                                numbener = np.empty(numbtime)
                                for t in tqdm(range(numbtime)):
                                    arry[t, 0, 0] = np.mean(listtime[t][1:]) + 2400000
                                    numbener[t] = listhdun[t+2].data['WAVELENGTH'].size
                                    
                                    if gdat.booldiag:
                                        if listhdun[t+2].data['FLUX'].ndim != 1:
                                            print('')
                                            print('')
                                            print('')
                                            print('listhdun[t+2].data[FLUX] should be one dimensional')
                                            print('listhdun[t+2].data[FLUX]')
                                            summgene(listhdun[t+2].data['FLUX'])
                                            raise Exception('')
                                    
                                    if listhdun[t+2].data['FLUX'].size == numbwlen:
                                        arry[t, :, 1] = listhdun[t+2].data['FLUX']
                                        arry[t, :, 2] = listhdun[t+2].data['FLUX_ERROR']
                                print('Writing to %s...' % pathmile)
                                np.save(pathmile, arry)
                            gdat.listarrylcurmast[mm] = [arry]

                cntrtess += 1
            
            if strgexprtemp == 'TESS':
                gdat.listtsecspoc = np.array(gdat.listtsecspoc, dtype=int)
        
        print('gdat.boolutiltesslocl')
        print(gdat.boolutiltesslocl)

    if gdat.booldiag:
        for p in gdat.indxinst[0]:
            if gdat.boolretrlcurmast[0][p]:
                if len(gdat.listarrylcurmast[p]) != gdat.listtsecspoc.size:
                    print('')
                    print('')
                    print('')
                    print('p')
                    print(p)
                    print('gdat.listarrylcurmast')
                    print(gdat.listarrylcurmast)
                    print('gdat.listtsecspoc')
                    print(gdat.listtsecspoc)
                    print('gdat.boolretrlcurmast')
                    print(gdat.boolretrlcurmast)
                    print('gdat.boolretrlcurmastanyy')
                    print(gdat.boolretrlcurmastanyy)
                    print('gdat.listarrylcurmast[p]')
                    summgene(gdat.listarrylcurmast[p])
                    raise Exception('len(gdat.listarrylcurmast[p]) == 0')
    
                for y in range(len(gdat.listarrylcurmast[p])):
                    if gdat.listarrylcurmast[p][y].ndim != 3:
                        print('')
                        print('')
                        print('')
                        print('p')
                        print(p)
                        print('y')
                        print(y)
                        print('gdat.listarrylcurmast')
                        print(gdat.listarrylcurmast)
                        print('gdat.boolretrlcurmast')
                        print(gdat.boolretrlcurmast)
                        print('gdat.listarrylcurmast[p][y]')
                        summgene(gdat.listarrylcurmast[p][y])
                        raise Exception('gdat.listarrylcurmast[p][y].ndim != 3')
    
        if gdat.boolretrlcurmastanyy:
            if np.unique(gdat.listtsecspoc).size != gdat.listtsecspoc.size:
                print('')
                print('')
                print('')
                print('gdat.listtsecspoc')
                print(gdat.listtsecspoc)
                raise Exception('gdat.listtsecspoc has repeating sectors.')

        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                if b == 0 and gdat.liststrginst[b][p] == 'TESS' and gdat.boolretrlcurmastanyy and len(gdat.listtsecspoc) != len(gdat.listarrylcurmast[p]):
                    print('')
                    print('')
                    print('')
                    print('len(gdat.listarrylcurmast[p])')
                    print(len(gdat.listarrylcurmast[p]))
                    print('gdat.listarrylcurmast[p]')
                    print(gdat.listarrylcurmast[p])
                    print('gdat.listipnt[b][p]')
                    print(gdat.listipnt[b][p])
                    print('gdat.listtsecspoc')
                    print(gdat.listtsecspoc)
                    summgene(gdat.listtsecspoc)
                    raise Exception('len(gdat.listarrylcurmast[p]) should match the length of gdat.listtsecspoc.')

    ## type of inference over energy axis to perform inference using
    ### 'full': fit all energy bins simultaneously
    ### 'iter': iterate over energy bins
    tdpy.setp_para_defa(gdat, 'fitt', 'typemodlenerfitt', 'iter')
    
    print('gdat.typelcurtpxftess')
    print(gdat.typelcurtpxftess)
    
    print('gdat.boolutilcadehigh')
    print(gdat.boolutilcadehigh)

    if gdat.typeverb > 1:
        print('gdat.fitt.typemodlenerfitt')
        print(gdat.fitt.typemodlenerfitt)
                    
    if gdat.booltesskepl and gdat.booltargpartanyy:
        numbtsec = len(gdat.listtsectcut)

        if gdat.typelcurtpxftess == 'lygos':
            boollygo = np.ones(numbtsec, dtype=bool)
            gdat.listtsecspoc = np.array([], dtype=int)
            gdat.listtseclygo = gdat.listtsectcut
        elif gdat.boolutilcadehigh:
            if gdat.typelcurtpxftess == 'SPOC_first':
                boollygo = ~tdpy.retr_boolsubb(gdat.listtsectcut, gdat.listtsecspoc)
                gdat.listtseclygo = gdat.listtsectcut[np.where(boollygo)]
            elif gdat.typelcurtpxftess == 'SPOC':
                boollygo = np.zeros_like(gdat.listtsectcut, dtype=bool)
                gdat.listtseclygo = []
            else:
                raise Exception('')
        else:
            raise Exception('')
        
        print('boollygo')
        print(boollygo)
        print('gdat.listtseclygo')
        print(gdat.listtseclygo)
    
    gdat.dictlygooutp = None
    
    print('gdat.booltesskepl')
    print(gdat.booltesskepl)

    if gdat.booldiag:
        # check if gdat.liststrginst has any 'TESS_S*'
        booltemp = False
        for strginst in gdat.liststrginst[0]:
            if strginst.startswith('TESS_S'):
                booltemp = True
    
        if 'TESS' in gdat.liststrginst[0] and booltemp:
            print('')
            print('')
            print('')
            print('gdat.liststrginst')
            print(gdat.liststrginst)
            raise Exception('gdat.liststrginst has both TESS and TESS_S*')

    if gdat.booltesskepl and gdat.booltargpartanyy and len(gdat.listtseclygo) > 0:
        
        # configure lygos
        print('Configuring lygos...')
        if gdat.dictlygoinpt is None:
            gdat.dictlygoinpt = dict()
        
        gdat.dictlygoinpt['pathtarg'] = gdat.pathtargcnfg + 'lygos/'
            
        if not 'liststrginst' in gdat.dictlygoinpt:
            gdat.dictlygoinpt['liststrginst'] = gdat.liststrginst[0]
            
            print('gdat.dictlygoinpt[liststrginst]')
            print(gdat.dictlygoinpt['liststrginst'])
        
        if not 'typepsfninfe' in gdat.dictlygoinpt:
            gdat.dictlygoinpt['typepsfninfe'] = 'fixd'
        
        ## target identifier
        if gdat.typetarg == 'MASTKey':
            gdat.dictlygoinpt['strgmast'] = gdat.strgmast
        elif gdat.typetarg == 'TICID' or gdat.typetarg == 'TOIID':
            gdat.dictlygoinpt['ticitarg'] = gdat.ticitarg
        elif gdat.typetarg == 'Position':
            gdat.dictlygoinpt['rasctarg'] = rasctarg
            gdat.dictlygoinpt['decltarg'] = decltarg
        else:
            raise Exception('')
        
        ## target label
        gdat.dictlygoinpt['labltarg'] = labltarg
        
        ## list of TESS sectors
        gdat.dictlygoinpt['listtsecsele'] = gdat.listtseclygo
        print('Will ask lygos to consider the following TESS sectors:')
        print(gdat.listtseclygo)

        # Boolean flag to use the TPFs
        if not 'boolutiltpxf' in gdat.dictlygoinpt:
            gdat.dictlygoinpt['boolutiltpxf'] = gdat.boolutilcadehigh
        
        # name of the lygos analysis from which the light curve will be derived
        gdat.dictlygoinpt['listnameanls'] = ['psfn']
        
        # type of data
        gdat.dictlygoinpt['liststrgtypedata'] = gdat.liststrgtypedata[0]

        # Boolean flag to use quality mask
        if not 'boolmaskqual' in gdat.dictlygoinpt:
            gdat.dictlygoinpt['boolmaskqual'] = gdat.boolmaskqual
        
        # Boolean flag to make lygos normalize the light curve by the median
        if not 'boolnorm' in gdat.dictlygoinpt:
            gdat.dictlygoinpt['boolnorm'] = True
        
        # run lygos
        if typeverb > 0:
            print('Will run lygos on the target...')
        
        gdat.dictlygooutp = lygos.main.init( \
                                            **gdat.dictlygoinpt, \
                                           )
        
        if gdat.booldiag:
            
            if gdat.dictlygooutp['listipnt'][0].size != gdat.listtseclygo.size:
                print('')
                print('')
                print('')
                print('gdat.listtseclygo')
                print(gdat.listtseclygo)
                summgene(gdat.listtseclygo)
                print('gdat.dictlygooutp[listipnt][0]')
                print(gdat.dictlygooutp['listipnt'][0])
                summgene(gdat.dictlygooutp['listipnt'][0])
                print('Warning! Sizes of gdat.dictlygooutp[listipnt][0] and gdat.listtseclygo are different!')

            elif not (gdat.dictlygooutp['listipnt'][0] - gdat.listtseclygo == 0).all():
                print('')
                print('')
                print('')
                print('gdat.listtseclygo')
                print(gdat.listtseclygo)
                print('gdat.dictlygooutp[listipnt][0]')
                print(gdat.dictlygooutp['listipnt'][0])
                print('Warning! gdat.dictlygooutp[listipnt][0] and gdat.listtseclygo are different!')

        # check if lygos has a missing sector
        print('b')
        print(b)
        print('gdat.indxinst[b]')
        print(gdat.indxinst[b])
        for p in gdat.indxinst[0]:
            listindxtseclygodele = []
            print('meyyy')
            print('gdat.dictlygooutp[arryrflx][gdat.nameanlslygo]')
            print(gdat.dictlygooutp['arryrflx'][gdat.nameanlslygo])
            for y, arry in enumerate(gdat.dictlygooutp['arryrflx'][gdat.nameanlslygo][0]):
                print('arry')
                summgene(arry)
                if len(arry) == 0:
                    print('lygos light curve for instrument %d and sector %d, is empty. Removing the sector...' % (0, gdat.dictlygooutp['listipnt'][0][y]))
                    listindxtseclygodele.append(y)
            if len(listindxtseclygodele) > 0:
                for indxtseclygodele in listindxtseclygodele:
                    del gdat.dictlygooutp['arryrflx'][gdat.nameanlslygo][0][indxtseclygodele]
                    gdat.dictlygooutp['listipnt'][p] = np.delete(gdat.dictlygooutp['listipnt'][p], indxtseclygodele)
                    gdat.dictlygooutp['listtcam'] = np.delete(gdat.dictlygooutp['listtcam'], indxtseclygodele)
                    gdat.dictlygooutp['listtccd'] = np.delete(gdat.dictlygooutp['listtccd'], indxtseclygodele)
        
        if gdat.booldiag:
            if gdat.booltesskepl and (gdat.typelcurtpxftess == 'lygos' or gdat.typelcurtpxftess == 'SPOC_first') and len(gdat.listtseclygo) > 0:
                for o, tseclygo in enumerate(gdat.dictlygooutp['listipnt'][0]):
                    if len(gdat.dictlygooutp['arryrflx'][gdat.nameanlslygo][0][o]) == 0:
                        print('')
                        print('')
                        print('')
                        print('o')
                        print(o)
                        print('gdat.dictlygooutp[arryrflx][gdat.nameanlslygo][0][o]')
                        print(gdat.dictlygooutp['arryrflx'][gdat.nameanlslygo][0][o])
                        raise Exception('len(gdat.dictlygooutp[arryrflx][gdat.nameanlslygo][0][o]) == 0')
                
        # remove bad times
        for p in gdat.indxinst[b]:
            for o, tseclygo in enumerate(gdat.dictlygooutp['listipnt'][p]):
                
                print('o')
                print(o)
                print('tseclygo')
                print(tseclygo)
                print('gdat.dictlygooutp[listipnt][p]')
                summgene(gdat.dictlygooutp['listipnt'][p])
                print('len(gdat.dictlygooutp[arryrflx][gdat.nameanlslygo][p])')
                print(len(gdat.dictlygooutp['arryrflx'][gdat.nameanlslygo][p]))

                # choose the current sector
                arry = gdat.dictlygooutp['arryrflx'][gdat.nameanlslygo][p][o]
                
                print('arry')
                print(arry)
                
                if gdat.booldiag:
                    if len(arry) == 0:
                        print('')
                        print('')
                        print('')
                        print('gdat.dictlygooutp[arryrflx]')
                        print(gdat.dictlygooutp['arryrflx'])
                        raise Exception('')

                # find good times
                indxtimegood = np.where(np.isfinite(arry[:, 1]) & np.isfinite(arry[:, 2]))[0]
                
                # filter for good times
                gdat.dictlygooutp['arryrflx'][gdat.nameanlslygo][p][o] = arry[indxtimegood, :]
            
        if gdat.booldiag:
            if gdat.booltesskepl and (gdat.typelcurtpxftess == 'lygos' or gdat.typelcurtpxftess == 'SPOC_first') and len(gdat.listtseclygo) > 0:
                for o, tseclygo in enumerate(gdat.dictlygooutp['listipnt'][0]):
                    if len(gdat.dictlygooutp['arryrflx'][gdat.nameanlslygo][0][o]) == 0:
                        print('')
                        print('')
                        print('')
                        print('o')
                        print(o)
                        print('gdat.dictlygooutp[arryrflx][gdat.nameanlslygo][0][o]')
                        print(gdat.dictlygooutp['arryrflx'][gdat.nameanlslygo][0][o])
                        print('')
                        raise Exception('len(gdat.dictlygooutp[arryrflx][gdat.nameanlslygo][0][o]) == 0')
                
    if gdat.booltesskepl and gdat.booltargpartanyy:
        print('List of TESS sectors returned by TESSCut')
        print(gdat.listtsectcut)
        print('List of TESS sectors to be reduced via lygos')
        print(gdat.listtseclygo)
        print('List of TESS sectors to be taken from SPOC')
        print(gdat.listtsecspoc)
        
        numbtsecspoc = gdat.listtsecspoc.size
        indxtsecspoc = np.arange(numbtsecspoc)

        gdat.listtsecpdcc = np.empty_like(gdat.listtsecspoc)
        gdat.listtsecsapp = np.empty_like(gdat.listtsecspoc)

        # merge list of sectors whose light curves will come from SPOC and lygos, respectively
        if not gdat.dictlygooutp is None:
            gdat.listtsec = np.unique(np.concatenate((gdat.dictlygooutp['listipnt'][0], gdat.listtsecspoc), dtype=int))
        else:
            gdat.listtsec = gdat.listtsecspoc

        # filter the list of sectors using the desired list of sectors, if any
        if listtsecsele is not None:
            print('List of TESS sectors before filtering:')
            print(gdat.listtsec)

            print('Filtering the list of sectors based on the user selection (listtsecsele)...')
            listtsecsave = np.copy(gdat.listtsec)
            gdat.listtsec = []
            for tsec in listtsecsele:
                if tsec in listtsecsave:
                    gdat.listtsec.append(tsec)
            gdat.listtsec = np.array(gdat.listtsec, dtype=int)
        
        print('List of TESS sectors')
        print(gdat.listtsec)

    else:
        if listtsecsele is not None:
            raise Exception('')

    gdat.numbdatagood = 0
    gdat.listipnt = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            if gdat.listlablinst[b][p] == 'TESS' and \
                        (gdat.liststrgtypedata[b][p] == 'simutargpartfprt' or gdat.liststrgtypedata[b][p] == 'simutargpartinje' or gdat.liststrgtypedata[b][p] == 'obsd'):
                gdat.listipnt[b][p] = gdat.listtsec
                if gdat.listtsec.size > 0:
                    gdat.numbdatagood += 1
            elif gdat.liststrgtypedata[b][p] == 'simutargsynt' or gdat.liststrgtypedata[b][p] == 'simutargpartsynt':
                gdat.listipnt[b][p] = np.array([0])
                gdat.numbdatagood += 1
            elif gdat.liststrgtypedata[b][p] == 'inpt':
                gdat.listipnt[b][p] = np.arange(len(gdat.listarrytser['Raw'][b][p]))
                if len(gdat.listarrytser['Raw'][b][p]) > 0:
                    gdat.numbdatagood += 1
            else:
                print('')
                print('')
                print('')
                print('gdat.listlablinst[b][p]')
                print(gdat.listlablinst[b][p])
                print('gdat.liststrgtypedata[b][p]')
                print(gdat.liststrgtypedata[b][p])
                raise Exception('Undefined case for gdat.listipnt.')

    if gdat.numbdatagood == 0:
        print('No good data has been found. Will quit.')
        print('')
        return gdat.dictmileoutp
        
    if gdat.booldiag:
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                if b == 0 and gdat.liststrginst[b][p] == 'TESS' and gdat.boolretrlcurmastanyy and len(gdat.listtsecspoc) != len(gdat.listarrylcurmast[p]):
                    print('')
                    print('')
                    print('')
                    print('gdat.listarrylcurmast[p]')
                    print(gdat.listarrylcurmast[p])
                    print('gdat.listipnt')
                    print(gdat.listipnt)
                    raise Exception('len(gdat.listarrylcurmast[p]) should match the length of gdat.listipnt.')

    if gdat.dictlygooutp is not None:
        for name in gdat.dictlygooutp:
            gdat.dictmileoutp['lygo_' + name] = gdat.dictlygooutp[name]
        
    if gdat.boolsimusome:
        
        gdat.indxener = [[] for p in gdat.indxinst[0]]
        for p in gdat.indxinst[0]:
            if gdat.listener[p] is not None and len(gdat.listener[p]) > 0:
                gdat.numbener[p] = gdat.listener[p].size
            else:
                gdat.numbener[p] = 1
            gdat.indxener[p] = np.arange(gdat.numbener[p])
        
        tdpy.setp_para_defa(gdat, 'true', 'typemodl', 'PlanetarySystem')

        tdpy.setp_para_defa(gdat, 'true', 'typemodlsupn', 'quad')
        tdpy.setp_para_defa(gdat, 'true', 'typemodlexcs', 'bump')

        setp_modlmedi(gdat, 'true')

    else:
        gdat.numbener[p] = 1
    
    if gdat.booldiag:
        for p in gdat.indxinst[b]:
            if gdat.liststrginst[0][p] == 'JWST':
                print('')
                print('')
                print('')
                print('gdat.liststrginst')
                print(gdat.liststrginst)
                print('gdat.listarrylcurmast')
                print(gdat.listarrylcurmast)
                raise Exception('')

    # generate a vector of random system parameters
    print('gdat.booltargsynt')
    print(gdat.booltargsynt)
    
    if gdat.booltargsynt:
        if gdat.true.boolsampsystnico:
            
            if hasattr(gdat.true, 'dictnicoinpt'):
                dictnicoinpt = gdat.true.dictnicoinpt
            else:
                dictnicoinpt = dict()

            gdat.dictnico = nicomedia.retr_dictpoplstarcomp( \
                                                       numbsyst=1, \
                                                       typesyst=gdat.true.typemodl, \
                                                       typepoplsyst='SyntheticPopulation', \
                                                       **dictnicoinpt, \
                                                      )
            for namepara in gdat.true.listnameparacomp[j]:
                gdat.dicttrue[namepara+'comp'] = gdat.dictnico['dictpopl']['comp']['compstar_SyntheticPopulation_All'][namepara + 'comp'][0]
    
    # determine whether the NASA Exoplanet Archive Composite PS catalog will be read for the target
    gdat.boolexar = False
    if gdat.typepriocomp is None and gdat.typetarg != 'synt' and gdat.typetarg != 'inptdata':
        gdat.boolexar = True

    # read NASA Excoplanet Archive
    if gdat.boolexar:
        if gdat.strgexar is None:
            gdat.strgexar = gdat.strgmast
        
        if gdat.typeverb > 0:
            print('gdat.strgexar')
            print(gdat.strgexar)
            
        gdat.dictexartarg = nicomedia.retr_dictexar(strgexar=gdat.strgexar, strgelem='comp', typeverb=gdat.typeverb)
        
        if gdat.dictexartarg['pericomp'][0].size > 20:
            print('gdat.strgmast')
            print(gdat.strgmast)
            print('gdat.strgexar')
            print(gdat.strgexar)
            print('gdat.typetarg')
            print(gdat.typetarg)
            raise Exception('')
        
        if gdat.typepriocomp == 'exar' and gdat.dictexartarg is None:
            raise Exception('Prior was forced to be NAE but the target is not in the NEA.')

    # determine whether to read TOI catalog
    gdat.boolexof = gdat.boolplotpopl or (gdat.boolsimusome or gdat.boolmodl) and gdat.boolexar and gdat.dictexartarg is None
    
    # read TOI catalog from ExoFOP
    if gdat.boolexof:
        gdat.dicttoiitarg = nicomedia.retr_dicttoii(toiitarg=gdat.toiitarg)
        
    if gdat.boolsimusome:
        # type of simulation parameters for companions
        if 'epocmtracomp' in gdat.dicttrue:
            gdat.typesourparasimucomp = 'inpt'
        else:
            if gdat.dictexartarg is not None:
                gdat.typesourparasimucomp = 'exar'
            else:
                if gdat.dicttoiitarg is not None:
                    gdat.typesourparasimucomp = 'exof'
                elif gdat.booltargsynt:
                    gdat.typesourparasimucomp = 'rand'
                else:
                    print('')
                    print('')
                    print('')
                    print('gdat.dicttrue')
                    print(gdat.dicttrue)
                    print('gdat.toiitarg')
                    print(gdat.toiitarg)
                    print('gdat.dicttoiitarg')
                    print(gdat.dicttoiitarg)
                    print('gdat.boolexar')
                    print(gdat.boolexar)
                    print('gdat.boolexof')
                    print(gdat.boolexof)
                    raise Exception('Could not define the type of the source of the parameters for the simulation, typesourparasimucomp.')
        
        if gdat.typeverb > 0:
            print('Source of simulation parameters for the companions (typesourparasimucomp): %s' % gdat.typesourparasimucomp)
        
    if gdat.boolmodl and gdat.fitt.boolmodlpsys:
        if gdat.typepriocomp is None:
            if gdat.boolexar and gdat.dictexartarg is not None:
                gdat.typepriocomp = 'exar'
            elif gdat.boolexof:
                gdat.typepriocomp = 'exof'
            else:
                if gdat.listtypeanls is not None:
                    if 'boxsperinega' in gdat.listtypeanls:
                        gdat.typepriocomp = 'boxsperinega'
                    elif 'boxsperiposi' in gdat.listtypeanls:
                        gdat.typepriocomp = 'boxsperinega'
                    elif 'outlperi' in gdat.listtypeanls:
                        gdat.typepriocomp = 'outlperi'
                elif gdat.fitt.boolmodlpsys:
                    gdat.typepriocomp = 'boxsperinega'
                else:
                    gdat.typepriocomp = 'boxsperiposi'
        
        if gdat.typeverb > 0:
            print('Source of prior parameters for the companions (typepriocomp): %s' % gdat.typepriocomp)
        
        if not gdat.boolexar and gdat.typepriocomp == 'exar':
            raise Exception('')
    
    ## list of analysis types
    if gdat.listtypeanls is None:
        gdat.listtypeanls = []
        if gdat.boolanls:
            gdat.listtypeanls.extend(retr_listtypeanls(gdat.fitt.typemodl, gdat.typepriocomp))

    if gdat.typeverb > 0:
        print('List of analysis types: %s' % gdat.listtypeanls)
    
    # Boolean flag to detrend the photometric time-series before estimating the priors
    if gdat.boolbdtr is None:
        gdat.boolbdtr = [[False for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                if gdat.boolanls and len(gdat.listtimescalbdtr) > 0 and \
                        (gdat.fitt.typemodl == 'PlanetarySystem' or gdat.fitt.typemodl == 'PlanetarySystemEmittingCompanion') and not gdat.liststrginst[b][p].startswith('LSST'):
                    gdat.boolbdtr[b][p] = True
    
    gdat.boolbdtranyy = False
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            if gdat.boolbdtr[b][p]:
                gdat.boolbdtranyy = True
    
    ## Boolean flag to calculate the power spectral density
    gdat.boolcalclspe = 'lspe' in gdat.listtypeanls
    gdat.dictmileoutp['boolcalclspe'] = gdat.boolcalclspe

    # Boolean flag to execute a search for negative periodic boxes
    gdat.boolsrchboxsperinega = 'boxsperinega' in gdat.listtypeanls

    # Boolean flag to execute a search for positive periodic boxes
    gdat.boolsrchboxsperiposi = 'boxsperiposi' in gdat.listtypeanls
    
    # Boolean flag to execute a search for periodic boxes
    gdat.boolsrchboxsperi = gdat.boolsrchboxsperiposi or gdat.boolsrchboxsperinega
    gdat.dictmileoutp['boolsrchboxsperi'] = gdat.boolsrchboxsperi
    
    # Boolean flag to execute a search for periodic outliers
    gdat.boolsrchoutlperi = 'outlperi' in gdat.listtypeanls
    gdat.dictmileoutp['boolsrchoutlperi'] = gdat.boolsrchoutlperi

    if gdat.typeverb > 0:
        print('gdat.boolcalclspe') 
        print(gdat.boolcalclspe)
        print('gdat.boolsrchoutlperi') 
        print(gdat.boolsrchoutlperi)
        print('gdat.boolsrchboxsperinega') 
        print(gdat.boolsrchboxsperinega)
        print('gdat.boolsrchboxsperiposi') 
        print(gdat.boolsrchboxsperiposi)
        print('gdat.boolsrchboxsperi') 
        print(gdat.boolsrchboxsperi)
        print('gdat.boolsrchflar') 
        print(gdat.boolsrchflar)
    
    if gdat.typeverb > 0:
        print('gdat.boolplotpopl')
        print(gdat.boolplotpopl)
    
    gdat.liststrgpdfn = ['prio']
        
    if gdat.boolplotpopl:
        ## define folders
        gdat.pathvisufeat = gdat.pathvisutarg + 'feat/'
    
        for strgpdfn in ['prio']:
            pathvisupdfn = gdat.pathvisufeat + strgpdfn + '/'
            setattr(gdat, 'pathvisufeatplan' + strgpdfn, pathvisupdfn + 'featplan/')
            setattr(gdat, 'pathvisufeatsyst' + strgpdfn, pathvisupdfn + 'featsyst/')
            setattr(gdat, 'pathvisudataplan' + strgpdfn, pathvisupdfn + 'dataplan/')
    
    ## make folders again (needed because of path definitions since the last mkdir)
    for attr, valu in gdat.__dict__.items():
        if attr.startswith('path') and valu is not None and not isinstance(valu, dict) and valu.endswith('/'):
            os.system('mkdir -p %s' % valu)
            
    if gdat.boolmodl:
        gdat.fitt.prio.meanpara.duratrantotlcomp = None

    gdat.nomipara = tdpy.gdatstrt()
    gdat.nomipara.duratrantotlcomp = None
    
    # retrieve values from literature
    if gdat.boolmodl and gdat.fitt.boolmodlpsys or gdat.boolsimusome and gdat.true.boolmodlpsys:
        
        if gdat.boolmodl and gdat.typepriocomp == 'exar' or gdat.boolsimusome and gdat.typesourparasimucomp == 'exar':
            if gdat.typeverb > 0:
                print('Retreiving the companion priors from the NASA Exoplanet Archive...')
            gdat.nomipara.pericomp = gdat.dictexartarg['pericomp'][0]
            gdat.nomipara.rsmacomp = gdat.dictexartarg['rsmacomp'][0]
            gdat.nomipara.depttrancomp = gdat.dictexartarg['depttrancomp'][0]
            gdat.nomipara.cosicomp = gdat.dictexartarg['cosicomp'][0]
            gdat.nomipara.epocmtracomp = gdat.dictexartarg['epocmtracomp'][0]
            
            if gdat.booldiag:
                if not np.isfinite(gdat.nomipara.epocmtracomp).all():
                    print('gdat.dictexartarg[epocmtracomp]')
                    print(gdat.dictexartarg['epocmtracomp'])
                    print('gdat.dictexartarg[epocmtracomp][0]')
                    summgene(gdat.dictexartarg['epocmtracomp'][0])
                    raise Exception('')

            gdat.nomipara.duratrantotlcomp = gdat.dictexartarg['duratrantotl'][0]
            indxcompbadd = np.where(~np.isfinite(gdat.nomipara.duratrantotlcomp))[0]
            if indxcompbadd.size > 0:
                dcyc = 0.15
                if gdat.typeverb > 0:
                    print('Duration from the Exoplanet Archive Composite PS table is infinite for some companions. Assuming a duty cycle of %.3g.' % dcyc)
                gdat.nomipara.duratrantotlcomp[indxcompbadd] = gdat.nomipara.pericomp[indxcompbadd] * dcyc
            gdat.nomipara.tmagsyst = gdat.dictexartarg['magtsystTESS'][0]
        
        if gdat.typepriocomp == 'exof':
            
            raise Exception('')

            if gdat.typeverb > 0:
                print('Retreiving the companion priors from ExoFOP-TESS...')
            
            gdat.nomipara.epocmtracomp = gdat.dicttoiitarg['epocmtracomp']
            if gdat.nomipara.pericomp is None:
                gdat.nomipara.pericomp = gdat.dicttoiitarg['pericomp']
            gdat.nomipara.rsmacomp = gdat.dicttoiitarg['rsmacomp']
            gdat.nomipara.depttrancomp = gdat.dicttoiitarg['depttrancomp']
            gdat.nomipara.duratrantotlcomp = gdat.dicttoiitarg['duratrantotl']
            gdat.nomipara.tmagsyst = gdat.dicttoiitarg['magtsystTESS'][0]
            gdat.nomipara.cosicomp = np.zeros_like(gdat.nomipara.epocmtracomp)
        
        if gdat.typepriocomp == 'inpt':
            gdat.nomipara.duratrantotlcomp = nicomedia.retr_duratrantotl(gdat.nomipara.pericomp, gdat.nomipara.rsmacomp, gdat.nomipara.cosicomp)

        if gdat.typepriocomp == 'exar' or gdat.typepriocomp == 'exof' or gdat.typepriocomp == 'inpt':
            gdat.nomipara.numbcomp = gdat.nomipara.pericomp.size
        
            if gdat.booldiag:
                if gdat.nomipara.numbcomp is None:
                    print('')
                    print('')
                    print('')
                    print('gdat.fitt.boolmodlpsys')
                    print(gdat.fitt.boolmodlpsys)
                    raise Exception('gdat.nomipara.numbcomp is None.')

            gdat.nomipara.rratcomp = [[] for pk in gdat.indxband]
            for pk in gdat.indxband:
                if gdat.typepriocomp == 'exar':
                    gdat.nomipara.rratcomp[pk] = gdat.dictexartarg['rratcomp'][0]
                if gdat.typepriocomp == 'exof':
                    gdat.nomipara.rratcomp[pk] = np.sqrt(gdat.nomipara.depttrancomp)
        
        # transfer nominal component parameters to prior means for the fitting model
        if gdat.boolmodl:
            if gdat.typepriocomp == 'exar' or gdat.typepriocomp == 'exof' or gdat.typepriocomp == 'inpt':
                for namepara in ['peri', 'epocmtra', 'cosi', 'rsma', 'rrat', 'duratrantotl', 'depttran']:
                    setattr(gdat.fitt.prio.meanpara, namepara + 'comp', getattr(gdat.nomipara, namepara + 'comp'))
        
                if gdat.booldiag:
                    if gdat.numbband != len(gdat.fitt.prio.meanpara.rratcomp):
                        print('')
                        print('')
                        print('')
                        print('gdat.numbband')
                        print(gdat.numbband)
                        #print('gdat.nomipara.rratcomp')
                        #print(gdat.nomipara.rratcomp)
                        raise Exception('gdat.numbband != len(gdat.fitt.prio.meanpara.rratcomp)')

        # check MAST
        if gdat.strgmast is None:
            if gdat.typetarg != 'inpt' and not gdat.booltargsynt:
                gdat.strgmast = gdat.labltarg

        if gdat.typeverb > 0:
            print('gdat.strgmast')
            print(gdat.strgmast)
        
        if not gdat.booltargsynt and not gdat.boolexecoffl and gdat.strgmast is not None and not gdat.boolsrchmastdone:
            listdictticinear = astroquery.mast.Catalogs.query_object(gdat.strgmast, catalog='TIC', radius=gdat.strgradi)
            gdat.boolsrchmastdone = True
            if listdictticinear[0]['dstArcSec'] > 0.1:
                if gdat.typeverb > 0:
                    print('The nearest source is more than 0.1 arcsec away from the target!')
            
            if gdat.typeverb > 0:
                print('Found the target on MAST!')
            
            gdat.rascstar = listdictticinear[0]['ra']
            gdat.declstar = listdictticinear[0]['dec']
            gdat.stdvrascstar = 0.
            gdat.stdvdeclstar = 0.
            if gdat.radistar is None:
                
                if gdat.typeverb > 0:
                    print('Setting the stellar radius from the TIC.')
                
                gdat.radistar = listdictticinear[0]['rad']
                gdat.stdvradistar = listdictticinear[0]['e_rad']
                
                if gdat.typeverb > 0:
                    if not np.isfinite(gdat.radistar):
                        print('Warning! TIC stellar radius is not finite.')
                    if not np.isfinite(gdat.radistar):
                        print('Warning! TIC stellar radius uncertainty is not finite.')
            if gdat.massstar is None:
                
                if gdat.typeverb > 0:
                    print('Setting the stellar mass from the TIC.')
                
                gdat.massstar = listdictticinear[0]['mass']
                gdat.stdvmassstar = listdictticinear[0]['e_mass']
                
                if gdat.typeverb > 0:
                    if not np.isfinite(gdat.massstar):
                        print('Warning! TIC stellar mass is not finite.')
                    if not np.isfinite(gdat.stdvmassstar):
                        print('Warning! TIC stellar mass uncertainty is not finite.')
            if gdat.tmptstar is None:
                
                if gdat.typeverb > 0:
                    print('Setting the stellar temperature from the TIC.')
                
                gdat.tmptstar = listdictticinear[0]['Teff']
                gdat.stdvtmptstar = listdictticinear[0]['e_Teff']
                
                if gdat.typeverb > 0:
                    if not np.isfinite(gdat.tmptstar):
                        print('Warning! TIC stellar temperature is not finite.')
                    if not np.isfinite(gdat.tmptstar):
                        print('Warning! TIC stellar temperature uncertainty is not finite.')
            gdat.jmagsyst = listdictticinear[0]['Jmag']
            gdat.hmagsyst = listdictticinear[0]['Hmag']
            gdat.kmagsyst = listdictticinear[0]['Kmag']
            gdat.vmagsyst = listdictticinear[0]['Vmag']
    
    # transfer literature values to model priors
    #for namepara in ['peri', 'epocmtra', 'cosi', 'rsma', 'rrat']:
    #for namepara in gdat.true.listnameparacomp[j]:
    #    if gdat.booltargsynt:
    #        para = gdat.dicttrue[namepara+'comp']
    #    else:
    #        para = getattr(gdat.nomipara, namepara + 'comp')
    #    setattr(gdat.fitt.prio.meanpara, namepara + 'comp', para)

    # list of strings to be attached to file names for each energy bin
    gdat.liststrgener = [[] for p in gdat.indxinst[0]]
    
    # determine number of chunks
    print('Determining the number of chunks...')
    gdat.numbchun = [np.zeros(gdat.numbinst[b], dtype=int) for b in gdat.indxdatatser]

    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            gdat.numbchun[b][p] = len(gdat.listipnt[b][p])
            
            if gdat.booldiag:
                if gdat.boolretrlcurmast[b][p]:
                    for y in range(len(gdat.listarrylcurmast[p])):
                        if gdat.listarrylcurmast[p][y].ndim != 3:
                            print('')
                            print('')
                            print('')
                            print('gdat.listarrylcurmast[p][y]')
                            summgene(gdat.listarrylcurmast[p][y])
                            raise Exception('gdat.listarrylcurmast[p][y].ndim != 3')
    
    gdat.indxchun = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            if gdat.booldiag:
                if gdat.numbchun[b][p] < 1:
                    print('')
                    print('')
                    print('')
                    print('gdat.numbchun[b][p]')
                    print(gdat.numbchun[b][p])
                    print('b, p')
                    print(b, p)
                    print('gdat.listipnt')
                    print(gdat.listipnt)
                    raise Exception('gdat.numbchun[b][p] < 1')

            gdat.indxchun[b][p] = np.arange(gdat.numbchun[b][p], dtype=int)


    ## simulated data
    if gdat.boolsimusome:
        
        # type of baseline
        ## 'cons': constant
        ## 'step': step function
        gdat.true.typemodlblinshap = 'cons'

        for p in gdat.indxinst[0]:
            for e in range(gdat.numbener[p]):
                gdat.liststrgener[p].append('ener%04d' % e)
        
        if gdat.true.typemodlblinshap == 'cons':
            for b in gdat.indxdatatser:
                for p in gdat.indxinst[b]:
                    if b == 0 and gdat.numbener[p] > 1 and gdat.true.typemodlblinener[p] == 'ener':
                        for e in range(gdat.numbener[p]):
                            tdpy.setp_para_defa(gdat, 'true', 'consblin%s%s' % (gdat.liststrginst[b][p], gdat.liststrgener[p][e]), np.array([0.]))
                    else:
                        tdpy.setp_para_defa(gdat, 'true', 'consblin%s' % gdat.liststrginst[b][p], np.array([0.]))
                    
        elif gdat.true.typemodlblinshap == 'step':
            tdpy.setp_para_defa(gdat, 'true', 'consblinfrst', np.array([0.]))
            tdpy.setp_para_defa(gdat, 'true', 'consblinseco', np.array([0.]))
            tdpy.setp_para_defa(gdat, 'true', 'timestep', np.array([0.]))
            tdpy.setp_para_defa(gdat, 'true', 'scalstep', np.array([1.]))
        
    
    if gdat.booldiag:
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                if not gdat.booltargsynt:
                    if b == 0:
                        if gdat.liststrginst[b][p] == 'TESS' and len(gdat.listtseclygo) > 0:
                            if not gdat.nameanlslygo in gdat.dictlygooutp['arryrflx']:
                                print('Warning: lygos data were not incorporated into miletos!')
            
    if gdat.numbener[p] == 1:
        gdat.numbenermodl = 1
        gdat.numbeneriter = 1
        gdat.numbenerefes = 1
    elif gdat.fitt.typemodlenerfitt == 'full':
        gdat.numbenermodl = gdat.numbener[p]
        gdat.numbeneriter = 2
        gdat.numbenerefes = 2
    else:
        gdat.numbenermodl = 1
        gdat.numbeneriter = gdat.numbener[p] + 1
        gdat.numbenerefes = gdat.numbener[p] + 1
    gdat.indxfittiter = np.arange(gdat.numbeneriter)
    gdat.indxenermodl = np.arange(gdat.numbenermodl)

    if not 'Raw' in gdat.listarrytser:
        gdat.listarrytser['Raw'] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    
    if gdat.boolsrchflar:
        gdat.arrytser['bdtrlowr'] = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.listarrytser['bdtrlowr'] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.arrytser['bdtrmedi'] = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.listarrytser['bdtrmedi'] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.arrytser['bdtruppr'] = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.listarrytser['bdtruppr'] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    
    gdat.arrytser['Raw'] = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gdat.arrytser['maskcust'] = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gdat.arrytser['Detrended'] = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    #gdat.arrytser['bdtrbind'] = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    
    gdat.listarrytser['maskcust'] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gdat.listarrytser['temp'] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gdat.listarrytser['trnd'] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gdat.listarrytser['Detrended'] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    #gdat.listarrytser['bdtrbind'] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    
    if gdat.booltesskepl and gdat.booltargpartanyy:
        gdat.dictmileoutp['listipnt'] = gdat.listipnt
        print('List of pointing IDs:')
        print(gdat.listipnt)
    
    # load data
    print('Loading data into gdat.listarrytser[raww]...')
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            
            ## from MAST
            if gdat.boolretrlcurmast[b][p]:
                for o in range(len(gdat.listipnt[b][p])):
                    indxspoc = np.where(gdat.listipnt[b][p][o] == gdat.listtsecspoc)[0][0]
                    gdat.listarrytser['Raw'][b][p][o] = gdat.listarrylcurmast[p][indxspoc]
            
            ## TESS and Kepler data via lygos
            if gdat.booltesskepl and gdat.booltargpartanyy and (gdat.typelcurtpxftess == 'lygos' or gdat.typelcurtpxftess == 'SPOC_first') and len(gdat.listtseclygo) > 0:
                for o in range(len(gdat.listipnt[b][p])):
                    #for o, tseclygo in enumerate(gdat.dictlygooutp['listipnt'][0]):
                    print('gdat.dictlygooutp[listipnt][0]')
                    print(gdat.dictlygooutp['listipnt'][0])
                    indxlygo = np.where(gdat.listipnt[b][p][o] == gdat.dictlygooutp['listipnt'][0])[0][0]
                    gdat.listarrytser['Raw'][b][p][o] = gdat.dictlygooutp['arryrflx'][gdat.nameanlslygo][0][indxlygo][:, None, :]
                
    ## time stamps of simulated data
    if gdat.boolsimusome:
        
        gdat.true.listtime = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.true.time = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.true.cade = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.true.timeexpo = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                
                if gdat.liststrgtypedata[b][p] == 'obsd':
                    continue

                if gdat.liststrgtypedata[b][p] == 'simutargsynt' or gdat.liststrgtypedata[b][p] == 'simutargpartsynt':
                    for y in gdat.indxchun[b][p]:
                        if gdat.liststrginst[b][p] == 'TESS':
                            gdat.true.cade[b][p] = 120. # [seconds]
                            # effective exposure time of TESS (80% efficiency due to cosmic ray rejection)
                            gdat.true.timeexpo[b][p] = 96. # [seconds]
                            delttime = gdat.true.cade[b][p] / 3600. / 24. # [day]
                            #gdat.true.listtime[b][p][y] = 2462000. + np.concatenate([np.arange(0., 13.2, delttime), np.arange(14.2, 27.3, delttime)])
                            lengobsv = 1.
                            gdat.true.listtime[b][p][y] = 2462000. + np.arange(0., lengobsv, delttime)
                        elif gdat.liststrginst[b][p].startswith('TESS-GEO'):
                            gdat.true.cade[b][p] = 6. # [seconds]
                            gdat.true.timeexpo[b][p] = 0.8 * gdat.true.cade[b][p]
                            delttime = gdat.true.cade[b][p] / 3600. / 24. # [day]
                            lengobsv = 1.
                            gdat.true.listtime[b][p][y] = 2462000. + np.arange(0., lengobsv, delttime)
                        elif gdat.liststrginst[b][p] == 'ULTRASAT':
                            # ULTRASAT short cadence will be five minutes
                            gdat.true.cade[b][p] = 300. # [seconds]
                            gdat.true.timeexpo[b][p] = 0.8 * gdat.true.cade[b][p]
                            delttime = gdat.true.cade[b][p] / 3600. / 24. # [day]
                            lengobsv = 1.
                            gdat.true.listtime[b][p][y] = 2462000. + np.arange(0., lengobsv, delttime)
                        elif gdat.liststrginst[b][p] == 'JWST':
                            gdat.true.listtime[b][p][y] = 2462000. + np.arange(0., lengobsv, delttime)
                        elif gdat.liststrginst[b][p].startswith('LSST'):
                            # WFD
                            numbtimelsst = 100
                            # LSST will have two 15-second exposures per visit
                            gdat.true.timeexpo[b][p] = 30. # [sec]
                            # DDF
                            #numbtimelsst = 1000
                            gdat.true.listtime[b][p][y] = (2460645. + np.random.rand(numbtimelsst)[:, None] * 365. * 0.7 + \
                                                                                np.arange(gdat.dicttrue['numbyearlsst'])[None, :] * 365.).flatten()
                            gdat.true.listtime[b][p][y] = np.sort(gdat.true.listtime[b][p][y])
                        else:
                            print('')
                            print('')
                            print('')
                            print('b, p')
                            print(b, p)
                            print('gdat.liststrginst[b][p]')
                            print(gdat.liststrginst[b][p])
                            raise Exception('')
                    
                elif gdat.liststrgtypedata[b][p] == 'simutargpartinje' or gdat.liststrgtypedata[b][p] == 'simutargpartfprt':
                    for y in gdat.indxchun[b][p]:
                        gdat.true.listtime[b][p][y] = gdat.listarrytser['Raw'][b][p][y][:, 0, 0] 
                else:
                    print('')
                    print('')
                    print('')
                    print('gdat.liststrgtypedata[b][p]')
                    print(gdat.liststrgtypedata[b][p])
                    raise Exception('Unknown gdat.liststrgtypedata[b][p]')
                
                gdat.true.time[b][p] = np.concatenate(gdat.true.listtime[b][p])
                
                if gdat.booldiag:
                    for y in gdat.indxchun[b][p]:
                        if isinstance(gdat.true.listtime[b][p][y], list):
                            print('')
                            print('')
                            print('')
                            print('gdat.liststrgtypedata[b][p]')
                            print(gdat.liststrgtypedata[b][p])
                            print('gdat.liststrginst[b][p]')
                            print(gdat.liststrginst[b][p])
                            print('gdat.true.listtime[b][p][y]')
                            print(gdat.true.listtime[b][p][y])
                            raise Exception('gdat.true.listtime[b][p][y] should be a numpy array.')

                    if gdat.liststrgtypedata[b][p] != 'obsd':
                        if len(gdat.true.time[b][p]) == 0:
                            print('')
                            print('')
                            print('')
                            print('b, p')
                            print(b, p)
                            print('gdat.liststrginst[b][p]')
                            print(gdat.liststrginst[b][p])
                            print('gdat.liststrgtypedata[b][p]')
                            print(gdat.liststrgtypedata[b][p])
                            print('gdat.true.time[b][p]')
                            summgene(gdat.true.time[b][p])
                            raise Exception('len(gdat.true.time[b][p]) == 0')

                        if np.amin(gdat.true.time[b][p][1:] - gdat.true.time[b][p][:-1]) < 0:
                            print('')
                            print('')
                            print('')
                            print('gdat.liststrgtypedata[b][p]')
                            print(gdat.liststrgtypedata[b][p])
                            raise Exception('The simulated time values are not sorted.')

                        for y in gdat.indxchun[b][p]:
                            if len(gdat.true.listtime[b][p][y]) == 0:
                                print('')
                                print('')
                                print('')
                                print('gdat.liststrgtypedata[b][p]')
                                print(gdat.liststrgtypedata[b][p])
                                raise Exception('len(gdat.true.listtime[b][p][y]) == 0')
        
        # generate the time axis
        setp_time(gdat)

    ## user-input data
    if gdat.listpathdatainpt is not None:
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                for y in gdat.indxchun[b][p]:
                    arry = np.loadtxt(gdat.listpathdatainpt[b][p][y], delimiter=',', skiprows=1)
                    gdat.listarrytser['Raw'][b][p][y] = np.empty((arry.shape[0], arry.shape[1], 3))
                    gdat.listarrytser['Raw'][b][p][y][:, :, 0:2] = arry[:, :, 0:2]
                    gdat.listarrytser['Raw'][b][p][y][:, :, 2] = 1e-4 * arry[:, :, 1]
                    indx = np.argsort(gdat.listarrytser['Raw'][b][p][y][:, 0])
                    gdat.listarrytser['Raw'][b][p][y] = gdat.listarrytser['Raw'][b][p][y][indx, :, :]
                    indx = np.where(gdat.listarrytser['Raw'][b][p][y][:, 1] < 1e6)[0]
                    gdat.listarrytser['Raw'][b][p][y] = gdat.listarrytser['Raw'][b][p][y][indx, :, :]
                    gdat.listisec = None
    
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            if gdat.liststrgtypedata[b][p] == 'simutargsynt' or gdat.liststrgtypedata[b][p] == 'simutargpartsynt':
                for y in gdat.indxchun[b][p]:
                    gdat.listarrytser['Raw'][b][p][y] = np.empty((gdat.true.listtime[b][p][y].size, gdat.numbener[p], 3))
                    gdat.listarrytser['Raw'][b][p][y][:, :, 0] = gdat.true.listtime[b][p][y][:, None]
        
    if gdat.booldiag:
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                if (gdat.liststrginst[b][p].startswith('TESS_S') or gdat.liststrginst[b][p] == 'TESS') and gdat.booltargpartanyy:
                    if len(gdat.listipnt[b][p]) != len(gdat.listarrytser['Raw'][b][p]):
                        print('')
                        print('')
                        print('')
                        print('b, p')
                        print(b, p)
                        print('gdat.listipnt')
                        print(gdat.listipnt)
                        print('gdat.listipnt[b][p]')
                        print(gdat.listipnt[b][p])
                        print('gdat.indxchun')
                        print(gdat.indxchun)
                        print('gdat.indxchun[b][p]')
                        print(gdat.indxchun[b][p])
                        print('len(gdat.listarrytser[raww][b][p])')
                        print(len(gdat.listarrytser['Raw'][b][p]))
                        raise Exception('len(gdat.listipnt[b][p]) != len(gdat.listarrytser[raww][b][p])')

    # list of strings to be attached to file names for type of run over energy bins
    gdat.liststrgdatafittiter = [[] for r in gdat.indxfittiter]
    for h in gdat.indxfittiter:
        if h == 0:
            if gdat.numbener[p] > 1:
                gdat.liststrgdatafittiter[0] = 'whit'
            else:
                gdat.liststrgdatafittiter[0] = ''
        else:
            e = h - 1
            
            gdat.liststrgdatafittiter[h] = 'ener%04d' % e
    
    if gdat.typeverb > 0:
        print('gdat.liststrgdatafittiter')
        print(gdat.liststrgdatafittiter)

    if gdat.fitt.typemodlenerfitt == 'iter':
        if gdat.typeinfe == 'samp':
            gdat.fitt.listdictsamp = []
        else:
            gdat.fitt.listdictmlik = []
    
    print('gdat.typepriocomp')
    print(gdat.typepriocomp)

    print('gdat.boolsrchboxsperi')
    print(gdat.boolsrchboxsperi)

    # define ephemerides if they are set by nominal values, not by period-search analyses
    if gdat.boolmodl and gdat.fitt.boolmodlpsys and not (gdat.boolsrchboxsperi or gdat.boolsrchoutlperi):
        gdat.fitt.prio.numbcomp = gdat.fitt.prio.meanpara.epocmtracomp.size
        gdat.fitt.prio.indxcomp = np.arange(gdat.fitt.prio.numbcomp)

    if gdat.boolsimusome:
    
        init_modl(gdat, 'true')

        if gdat.true.typemodl == 'StarFlaring':
            tdpy.setp_para_defa(gdat, 'true', 'numbflar', 1)
            gdat.true.indxflar = np.arange(gdat.true.numbflar)
        
            for k in gdat.true.indxflar:
                tdpy.setp_para_defa(gdat, 'true', 'amplflar%04d' % k, 0.1)
                timeflar = tdpy.icdf_self(np.random.rand(), gdat.minmtimeconc[0], gdat.maxmtimeconc[0]) 
                tdpy.setp_para_defa(gdat, 'true', 'timeflar%04d' % k, timeflar)
                tdpy.setp_para_defa(gdat, 'true', 'tsclflar%04d' % k, 1.) # [1 hour]
        
        # probably to be deleted
        #if gdat.booldiag:
        #    if gdat.true.typemodl == 'CompactObjectStellarCompanion' or gdat.true.typemodl == 'PlanetarySystem' or gdat.true.typemodl == 'PlanetarySystemEmittingCompanion' or gdat.true.typemodl == 'PlanetarySystemWithTTVs':
        #        if not hasattr(gdat.true, 'epocmtracomp'):
        #            raise Exception('not hasattr(gdat.true, epocmtracomp')
        
        # copy nominal parameters to the true parameters
        if (gdat.true.boolmodlpsys or gdat.true.typemodl == 'CompactObjectStellarCompanion') and not (gdat.boolsrchboxsperi or gdat.boolsrchoutlperi):
            for namepara in ['rrat', 'peri', 'epocmtra', 'rsma', 'cosi']:
                tdpy.setp_para_defa(gdat, 'true', '%scomp' % namepara, getattr(gdat.nomipara, namepara + 'comp'))
        
        setp_modlbase(gdat, 'true')
        
        if (gdat.true.typemodl.startswith('PlanetarySystem') or gdat.true.typemodl == 'CompactObjectStellarCompanion') and gdat.true.numbcomp > 0:
            if not 'pericom0' in gdat.true.listnameparafull:
                print('')
                print('')
                print('')
                print('gdat.true.listnameparafull')
                print(gdat.true.listnameparafull)
                raise Exception('pericom0 not in dictparainpt')

        if False and gdat.booldiag:
            if gdat.true.boolmodlcomp:
                for j in gdat.true.indxcomp:
                    for namepara in gdat.true.listnameparacomp[j]:
                        strgparacomp = '%scomp' % namepara
                        if not hasattr(gdat.true, strgparacomp):
                            print('')
                            print('')
                            print('')
                            print('namepara')
                            print(namepara)
                            print('strgparacomp')
                            print(strgparacomp)
                            print('gdat.true.listnameparacomp')
                            print(gdat.true.listnameparacomp)
                            print('gdat.liststrgtypedata')
                            print(gdat.liststrgtypedata)
                            print('gdat.liststrginst')
                            print(gdat.liststrginst)
                            print('gdat.true.typemodl')
                            print(gdat.true.typemodl)
                            print('gdat.true.keys()')
                            print(gdat.true.__dict__.keys())
                            raise Exception('not hasattr(gdat.true, strgparacomp)')

                        para = getattr(gdat.true, strgparacomp)
                        if len(para) == 0:
                            print('')
                            print('')
                            print('')
                            print('j')
                            print(j)
                            print('namepara')
                            print(namepara)
                            raise Exception('A true component parameter is empty.')
            
        #if gdat.true.boolmodlpsys or gdat.true.typemodl == 'CompactObjectStellarCompanion':
        #    print('gdat.fitt.prio.numbcomp')
        #    print(gdat.fitt.prio.numbcomp)
        #    print('gdat.typepriocomp')
        #    print(gdat.typepriocomp)
        #    if gdat.typepriocomp == 'exar' or gdat.typepriocomp == 'exof' or gdat.typepriocomp == 'inpt':
        #        
        #        numbcomp = gdat.fitt.prio.numbcomp
        #        
        #        if gdat.booldiag:
        #            if gdat.fitt.prio.numbcomp is None:
        #                print('')
        #                print('')
        #                print('')
        #                print('gdat.true.boolmodlpsys')
        #                print(gdat.true.boolmodlpsys)
        #                print('gdat.typepriocomp')
        #                print(gdat.typepriocomp)
        #                print('gdat.true.typemodl')
        #                print(gdat.true.typemodl)
        #                raise Exception('gdat.fitt.prio.numbcomp is None while generating true data.')
        #    else:
        #        numbcomp = 1
        #    print('numbcomp')
        #    print(numbcomp)
        #    #tdpy.setp_para_defa(gdat, 'true', 'numbcomp', numbcomp)

        #    #gdat.true.indxcomp = np.arange(gdat.true.numbcomp)
        #    print('gdat.true.numbcomp')
        #    print(gdat.true.numbcomp)
            
        if gdat.true.boolmodlpsys or gdat.true.typemodl == 'CompactObjectStellarCompanion':
            if gdat.true.boolmodlpsys:
                for j in gdat.true.indxcomp:
                    rratcomp = gdat.true.rratcomp[j]
                    
                    if False and gdat.booldiag:
                        if np.isscalar(rratcomp):
                            print('')
                            print('')
                            print('')
                            print('gdat.numbener[p]')
                            print(gdat.numbener[p])
                            print('rratcomp')
                            print(rratcomp)
                            raise Exception('')
                        
                    if gdat.numbener[p] == 1:
                        tdpy.setp_para_defa(gdat, 'true', 'rratcom%d' % j, rratcomp)
                    else:
                        #tdpy.setp_para_defa(gdat, 'true', 'rratcom0whit', 0.1)
                        for p in gdat.indxinst[b]:
                            if gdat.numbener[p] == 1:
                                tdpy.setp_para_defa(gdat, 'true', 'rratcom%dener%04d' % (j, p), rratcomp[p])
                            else:
                                for e in range(gdat.numbener[p]):
                                    tdpy.setp_para_defa(gdat, 'true', 'rratcom%dins%dener%04d' % (j, p, e), rratcomp[p, e])
                    
            if gdat.true.typemodl == 'CompactObjectStellarCompanion':
                tdpy.setp_para_defa(gdat, 'true', 'radistar', 1.)
                tdpy.setp_para_defa(gdat, 'true', 'massstar', 1.)
                tdpy.setp_para_defa(gdat, 'true', 'masscom0', 1.)
        
        if gdat.numbener[p] == 1:
            gdat.numbenermodl = gdat.numbener[p]
            gdat.numbeneriter = 1
        else:
            gdat.numbenermodl = 1
            gdat.numbeneriter = gdat.numbener[p]

        # to be deleted
        #if gdat.true.boolmodlcomp and not gdat.booltargsynt:
        #    for j in range(gdat.true.epocmtracomp.size):
        #        for name in gdat.true.listnameparacomp[j]:
        #            if name == 'rrat':
        #                compprio = getattr(gdat, '%scompprio' % name)[p][j]
        #            else:
        #                compprio = getattr(gdat, '%scompprio' % name)[j]
        #            setattr(gdat.true, '%scom%d' % (name, j), compprio)
        
        dictparainpt = dict()
        for namepara in gdat.true.listnameparafull:
            dictparainpt[namepara] = getattr(gdat.true, namepara)

        if gdat.booldiag:
            if len(dictparainpt) == 0:
                raise Exception('')
        
            for b in gdat.indxdatatser:
                for p in gdat.indxinst[b]:
                    if gdat.liststrgtypedata[b][p] != 'obsd' and gdat.true.time[b][p].size == 0:
                        print('')
                        print('')
                        print('')
                        print('b, p')
                        print(b, p)
                        raise Exception('gdat.true.time[b][p].size == 0')
        
            if (gdat.true.typemodl.startswith('PlanetarySystem') or gdat.true.typemodl == 'CompactObjectStellarCompanion') and gdat.true.numbcomp > 0:
                if not 'pericom0' in dictparainpt:
                    print('')
                    print('')
                    print('')
                    print('dictparainpt')
                    print(dictparainpt.keys())
                    print(dictparainpt)
                    print('gdat.true.listnameparafull')
                    print(gdat.true.listnameparafull)
                    raise Exception('pericom0 not in dictparainpt')
        
        gdat.true.dictmodl = retr_dictmodl_mile(gdat, gdat.true.time, dictparainpt, 'true')[0]
        

        if gdat.true.typemodlblinshap == 'GaussianProcess':
            dictrflx = retr_rflxmodl_mile_gpro(gdat, 'true', gdat.true.time, dictparainpt)
            gdat.true.dictrflxmodl['Baseline'] = dictrflx['Baseline']

        # generate data and its uncertainty
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                for y in gdat.indxchun[b][p]:
                    
                    if gdat.booldiag:
                        if len(gdat.listarrytser['Raw'][b][p][y]) == 0:
                            print('')
                            print('')
                            print('')
                            print('gdat.liststrgtypedata[b][p]')
                            print(gdat.liststrgtypedata[b][p])
                            print('gdat.liststrginst[b][p]')
                            print(gdat.liststrginst[b][p])
                            raise Exception('len(gdat.listarrytser[Raw][b][p][y]) == 0')

                    if gdat.liststrgtypedata[b][p] == 'simutargpartinje':
                        if gdat.numbchun[b][p] > 1:
                            raise Exception('')
                        gdat.listarrytser['Raw'][b][p][y][:, :, 1] += gdat.true.dictmodl['Total'][0][p]
                    
                    if gdat.liststrgtypedata[b][p] == 'simutargsynt' or gdat.liststrgtypedata[b][p] == 'simutargpartsynt' or gdat.liststrgtypedata[b][p] == 'simutargpartfprt':

                        if gdat.booldiag:
                            if gdat.dictmagtsyst[gdat.liststrginst[b][p]] is None:
                                print('')
                                print('')
                                print('')
                                print('gdat.typesourparasimucomp')
                                print(gdat.typesourparasimucomp)
                                raise Exception('When synthetic data is being generated, gdat.dictmagtsyst[gdat.liststrginst[b][p]] should not be None.')
                            
                            if gdat.listarrytser['Raw'][b][p][y].shape[0] != gdat.true.listtime[b][p][y].size:
                                print('')
                                print('')
                                print('')
                                print('gdat.listarrytser[raww][b][p][y][:, :, 1]')
                                summgene(gdat.listarrytser['Raw'][b][p][y][:, :, 1])
                                print('(gdat.true.listtime[b][p][y].size, gdat.numbener[p])')
                                print((gdat.true.listtime[b][p][y].size, gdat.numbener[p]))
                                raise Exception('gdat.listarrytser[raww][b][p][y].shape[0] != gdat.true.listtime[b][p][y].size')
                        
                        # noise per cadence
                        nois = nicomedia.retr_noisphot(gdat.dictmagtsyst[gdat.liststrginst[b][p]], gdat.liststrginst[b][p]) * np.sqrt(3600. / gdat.true.timeexpo[b][p]) # [ppt]
                        
                        gdat.listarrytser['Raw'][b][p][y][:, :, 2] = 1e-3 * nois
                        
                        gdat.listarrytser['Raw'][b][p][y][:, :, 1] = gdat.true.dictmodl['Total'][0][p]
                        
                        # add noise to the synthetic data
                        gdat.listarrytser['Raw'][b][p][y][:, :, 1] += \
                             np.random.randn(gdat.true.listtime[b][p][y].size * gdat.numbener[p]).reshape((gdat.true.listtime[b][p][y].size, gdat.numbener[p])) * \
                                                                                                                        gdat.listarrytser['Raw'][b][p][y][:, :, 2]
        
                    if gdat.booldiag:
                        if (abs(gdat.listarrytser['Raw'][b][p][y]) > 1e10).any():
                            print('')
                            print('')
                            print('')
                            print('gdat.listarrytser[Raw][b][p][y][:, :, 0]')
                            summgene(gdat.listarrytser['Raw'][b][p][y][:, :, 0])
                            print('gdat.listarrytser[Raw][b][p][y][:, :, 1]')
                            summgene(gdat.listarrytser['Raw'][b][p][y][:, :, 1])
                            print('gdat.listarrytser[Raw][b][p][y][:, :, 2]')
                            summgene(gdat.listarrytser['Raw'][b][p][y][:, :, 2])
                            if gdat.liststrgtypedata[b][p] == 'simutargsynt' or gdat.liststrgtypedata[b][p] == 'simutargpartsynt' or gdat.liststrgtypedata[b][p] == 'simutargpartfprt':
                                print('gdat.dictmagtsyst[gdat.liststrginst[b][p]]')
                                print(gdat.dictmagtsyst[gdat.liststrginst[b][p]])
                                print('gdat.liststrginst[b][p]')
                                print(gdat.liststrginst[b][p])
                                print('gdat.true.timeexpo[b][p]')
                                print(gdat.true.timeexpo[b][p])
                                print('nois')
                                summgene(nois)
                            raise Exception('')

    else:
        for p in gdat.indxinst[0]:
            # define number of energy bins if any photometric data exist
            gdat.numbener[p] = gdat.listarrytser['Raw'][0][p][0].shape[1]
            for e in range(gdat.numbener[p]):
                if e == 0 and gdat.numbener[p] == 1:
                    gdat.liststrgener[p].append('')
                else:
                    gdat.liststrgener[p].append('ener%04d' % e)

    if gdat.booldiag:
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                for y in gdat.indxchun[b][p]:
                    if not np.isfinite(gdat.listarrytser['Raw'][b][p][y]).all():
                        print('')
                        print('')
                        print('')
                        print('b, p, y')
                        print(b, p, y)
                        indxbadd = np.where(~np.isfinite(gdat.listarrytser['Raw'][b][p][y]))[0]
                        print('gdat.listarrytser[raww][b][p][y]')
                        summgene(gdat.listarrytser['Raw'][b][p][y])
                        print('indxbadd')
                        summgene(indxbadd)
                        raise Exception('not np.isfinite(gdat.listarrytser[raww][b][p]).all()')
                
    # make white light curve
    if gdat.numbener[p] > 1:
        
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                for y in gdat.indxchun[b][p]:
                    arrywhit = np.empty((gdat.listarrytser['Raw'][b][p][y].shape[0], 3))
                    arrywhit[:, 0] = gdat.listarrytser['Raw'][b][p][y][:, 0, 0]
                    arrywhit[:, 1] = np.mean(gdat.listarrytser['Raw'][b][p][y][:, :, 1], 1)
                    arrywhit[:, 2] = np.sqrt(np.sum(gdat.listarrytser['Raw'][b][p][y][:, :, 2]**2, 1)) / gdat.numbener[p]
                    arrytemp = np.empty((gdat.listarrytser['Raw'][b][p][y].shape[0], gdat.numbener[p] + 1, 3))
                    arrytemp[:, 0, :] = arrywhit
                    arrytemp[:, 1:, :] = gdat.listarrytser['Raw'][b][p][y]
                    gdat.listarrytser['Raw'][b][p][y] = arrytemp
    
    if gdat.typeverb > 1:
        print('gdat.numbener[p]')
        print(gdat.numbener[p])
        
    gdat.indxener = [[] for p in gdat.indxinst[0]]
    for p in gdat.indxinst[0]:
        gdat.indxener[p] = np.arange(gdat.numbener[p])
        
    if gdat.numbener[p] > 1 and gdat.typeverb > 0:
        print('gdat.fitt.typemodlenerfitt')
        print(gdat.fitt.typemodlenerfitt)

    # concatenate data across sectors
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            gdat.arrytser['Raw'][b][p] = np.concatenate(gdat.listarrytser['Raw'][b][p])
    
    # generate the time axis
    setp_time(gdat, 'Raw')
    
    if gdat.boolmodl and gdat.fitt.boolmodlpsys and (gdat.typepriocomp == 'inpt' or gdat.typepriocomp == 'exar' or gdat.typepriocomp == 'exof'):
        retr_timetran(gdat, 'Raw')
    
    if gdat.liststrgcomp is None:
        gdat.liststrgcomp = nicomedia.retr_liststrgcomp(gdat.fitt.prio.numbcomp)
    if gdat.listcolrcomp is None:
        gdat.listcolrcomp = nicomedia.retr_listcolrcomp(gdat.fitt.prio.numbcomp)
    
    # sampling rate (cadence)
    ## temporal
    gdat.cadetime = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            timeconctemp = gdat.arrytser['Raw'][0][p][:, 0, 0]
            gdat.cadetime[b][p] = np.amin(timeconctemp[1:] - timeconctemp[:-1]) * 3600.
            
            if gdat.booldiag:
                if not (np.sort(timeconctemp) - timeconctemp == 0).all() or gdat.cadetime[b][p] <= 0:
                    print('')
                    print('')
                    print('')
                    print('gdat.listarrytser[Raw][0][p]')
                    print(gdat.listarrytser['Raw'][0][p])
                    print('gdat.arrytser[Raw][0][p][:, 0]')
                    print(gdat.arrytser['Raw'][0][p][:, 0])
                    print('timeconctemp')
                    summgene(timeconctemp)
                    print('gdat.liststrgtypedata[b][p]')
                    print(gdat.liststrgtypedata[b][p])
                    print('gdat.cadetime[b][p]')
                    print(gdat.cadetime[b][p])
                    raise Exception('timeconctemp is not sorted or gdat.cadetime[b][p] <= 0!')
            
    if gdat.booldiag:
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                for y in gdat.indxchun[b][p]:
                    if len(gdat.listarrytser['Raw'][b][p][y]) == 0:
                        print('')
                        print('')
                        print('')
                        print('bpy')
                        print('%d, %d, %d' % (b, p, y))
                        print('gdat.listarrytser[raww][b][p][y]')
                        summgene(gdat.listarrytser['Raw'][b][p][y])
                        raise Exception('len(gdat.listarrytser[raww][b][p][y]) == 0')

                if gdat.liststrginst[b][p] == 'TESS' and gdat.booltargpartanyy:
                    if not hasattr(gdat, 'listtsecspoc'):
                        print('')
                        print('')
                        print('')
                        print('b, p')
                        print(b, p)
                        print('gdat.liststrgtypedata')
                        print(gdat.liststrgtypedata)
                        print('gdat.boolretrlcurmast[b][p]')
                        print(gdat.boolretrlcurmast[b][p])
                        raise Exception('listtsecspoc is not defined while accessing TESS data of particular target.')
                    
                    if len(gdat.listipnt[b][p]) != len(gdat.listarrytser['Raw'][b][p]):
                        print('')
                        print('')
                        print('')
                        print('b, p')
                        print(b, p)
                        print('gdat.listipnt')
                        print(gdat.listipnt)
                        print('len(gdat.listarrytser[raww][b][p])')
                        print(len(gdat.listarrytser['Raw'][b][p]))
                        raise Exception('len(gdat.listipnt) != len(gdat.listarrytser[raww][b][p])')
    
    if gdat.liststrgchun is None:
        gdat.listlablchun = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.liststrgchun = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                for y in gdat.indxchun[b][p]:
                    if gdat.liststrginst[b][p] == 'TESS' and (gdat.liststrgtypedata[b][p] == 'simutargpartfprt' or gdat.liststrgtypedata[b][p] == 'simutargpartinje' or \
                                                                                                                                            gdat.liststrgtypedata[b][p] == 'obsd'):
                        if gdat.booldiag:
                            if gdat.booltargpartanyy and gdat.numbchun[b][p] != len(gdat.listipnt[b][p]):
                                print('')
                                print('')
                                print('')
                                print('')
                                print('bpy')
                                print(b, p, y)
                                print('gdat.listipnt')
                                print(gdat.listipnt)
                                print('gdat.indxchun')
                                print(gdat.indxchun)
                                print('gdat.numbchun')
                                print(gdat.numbchun)
                                raise Exception('gdat.numbchun[b][p] != len(gdat.listipnt)')

                        gdat.listlablchun[b][p][y] = 'Sector %d' % gdat.listipnt[b][p][y]
                        gdat.liststrgchun[b][p][y] = 'Sector%02d' % gdat.listipnt[b][p][y]
                    else:
                        gdat.liststrgchun[b][p][y] = 'ch%02d' % y

    # check the user-defined gdat.listpathdatainpt
    if gdat.listpathdatainpt is not None:
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                if not isinstance(gdat.listpathdatainpt[b][p], list):
                    raise Exception('')
    
    if gdat.boolnormphot:
        gdat.labltserphot = 'Relative flux'
    else:
        gdat.labltserphot = 'ADC Counts [e$^-$/s]'
    gdat.listlabltser = [gdat.labltserphot, 'Radial Velocity [km/s]']
    gdat.liststrgdatatsercsvv = ['flux', 'rv']
    
    gdat.strgheadtserphot = 'time,%s,%s_err' % (gdat.liststrgdatatsercsvv[0], gdat.liststrgdatatsercsvv[0])
    gdat.strgheadpserphot = 'phase,%s,%s_err' % (gdat.liststrgdatatsercsvv[0], gdat.liststrgdatatsercsvv[0])
    gdat.strgheadtserrvel = 'time,%s,%s_err' % (gdat.liststrgdatatsercsvv[1], gdat.liststrgdatatsercsvv[1])
    gdat.strgheadpserrvel = 'phase,%s,%s_err' % (gdat.liststrgdatatsercsvv[1], gdat.liststrgdatatsercsvv[1])
    gdat.strgheadtser = [gdat.strgheadtserphot, gdat.strgheadtserrvel]
    gdat.strgheadpser = [gdat.strgheadpserphot, gdat.strgheadpserrvel]
    
    if gdat.offstextatmoraditmpt is None:
        gdat.offstextatmoraditmpt = [[0.3, -0.5], [0.3, -0.5], [0.3, -0.5], [0.3, 0.5]]
    if gdat.offstextatmoradimetr is None:
        gdat.offstextatmoradimetr = [[0.3, -0.5], [0.3, -0.5], [0.3, -0.5], [0.3, 0.5]]
    
    if gdat.typetarg == 'inpt':
        if gdat.vmagsyst is None:
            gdat.vmagsyst = 0.
        if gdat.jmagsyst is None:
            gdat.jmagsyst = 0.
        if gdat.hmagsyst is None:
            gdat.hmagsyst = 0.
        if gdat.kmagsyst is None:
            gdat.kmagsyst = 0.

    if gdat.booldiag:
        
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                for y in gdat.indxchun[b][p]:
                    if len(gdat.listarrytser['Raw'][b][p][y]) == 0:
                        print('')
                        print('')
                        print('')
                        print('bpy')
                        print(b, p, y)
                        print('gdat.indxchun')
                        print(gdat.indxchun)
                        raise Exception('')
        
                if not np.isfinite(gdat.arrytser['Raw'][b][p]).all():
                    print('')
                    print('')
                    print('')
                    print('b, p')
                    print(b, p)
                    indxbadd = np.where(~np.isfinite(gdat.arrytser['Raw'][b][p]))[0]
                    print('gdat.arrytser[raww][b][p]')
                    summgene(gdat.arrytser['Raw'][b][p])
                    print('indxbadd')
                    summgene(indxbadd)
                    raise Exception('not np.isfinite(gdat.arrytser[raww][b][p]).all()')
    
    # check availability of data 
    booldataaval = False
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            for y in gdat.indxchun[b][p]:
                if len(gdat.listarrytser['Raw'][b][p][y]) > 0:
                    booldataaval = True
    
    if not booldataaval:
        if gdat.typeverb > 0:
            print('No data found. Returning...')
        return gdat.dictmileoutp
    
    if gdat.numbener[p] > 1:
        gdat.ratesampener = np.amin(gdat.listener[p][1:] - gdat.listener[p][:-1])
    
    # string to indicate cadence in the file names
    gdat.strgextncade = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            if gdat.liststrginst[b][p] == 'TESS':
                gdat.strgextncade[b][p] = '_%dsec' % round(gdat.cadetime[b][p] * 3600 * 24)
            else:
                gdat.strgextncade[b][p] = ''
    
    # plot raw data
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            for y in gdat.indxchun[b][p]:
                if gdat.boolplottser:
                    plot_tser_mile(gdat, b, p, y, 'Raw', boolcolrtran=True)
    
            for e in gdat.indxener[p]:

                if gdat.numbchun[0][p] > 1:
                    path = gdat.pathdatatarg + '%s_DataCube_%s%s%s.csv' % (gdat.liststrgdatatser[0], gdat.liststrginst[0][p], gdat.strgextncade[b][p], gdat.liststrgener[p][e])
                    if not os.path.exists(path):
                        if gdat.typeverb > 0:
                            print('Writing to %s...' % path)
                        np.savetxt(path, gdat.arrytser['Raw'][0][p][:, e, :], delimiter=',', header=gdat.strgheadtser[b])
                
                for y in gdat.indxchun[0][p]:
                    path = gdat.pathdatatarg + '%s_DataCube_%s_%s%s%s.csv' % (gdat.liststrgdatatser[0], gdat.liststrginst[0][p], gdat.liststrgchun[0][p][y], \
                                                                                                                        gdat.strgextncade[b][p], gdat.liststrgener[p][e])
                    if not os.path.exists(path):
                        if gdat.typeverb > 0:
                            print('Writing to %s...' % path)
                        np.savetxt(path, gdat.listarrytser['Raw'][0][p][y][:, e, :], delimiter=',', header=gdat.strgheadtser[b])
    
    # obtain 'maskcust' (obtained after custom mask, if any) time-series bundle after applying user-defined custom mask, if any
    if gdat.listlimttimemask is not None:
        
        if gdat.typeverb > 0:
            print('Masking the data...')
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                numbmask = len(gdat.listlimttimemask[b][p])
                for y in gdat.indxchun[b][p]:
                    listindxtimemask = []
                    for k in range(numbmask):
                        indxtimemask = np.where((gdat.listarrytser['Raw'][b][p][y][:, 0] < gdat.listlimttimemask[b][p][k][1]) & \
                                                (gdat.listarrytser['Raw'][b][p][y][:, 0] > gdat.listlimttimemask[b][p][k][0]))[0]
                        listindxtimemask.append(indxtimemask)
                    listindxtimemask = np.concatenate(listindxtimemask)
                    listindxtimegood = np.setdiff1d(np.arange(gdat.listarrytser['Raw'][b][p][y].shape[0]), listindxtimemask)
                    gdat.listarrytser['maskcust'][b][p][y] = gdat.listarrytser['Raw'][b][p][y][listindxtimegood, :]
                gdat.arrytser['maskcust'][b][p] = np.concatenate(gdat.listarrytser['maskcust'][b][p], 0)
                
                if gdat.boolplottser:
                    plot_tser_mile(gdat, b, p, y, 'maskcust')
                    for y in gdat.indxchun[b][p]:
                        plot_tser_mile(gdat, b, p, y, 'maskcust')
    else:
        gdat.arrytser['maskcust'] = gdat.arrytser['Raw']
        gdat.listarrytser['maskcust'] = gdat.listarrytser['Raw']
    
    # detrending
    ## determine whether to use any mask for detrending
    if gdat.boolmodl and gdat.fitt.boolmodlcomp and gdat.nomipara.duratrantotlcomp is not None:
        # assign the prior orbital parameters as the baseline-detrend mask
        gdat.epocmask = gdat.fitt.prio.meanpara.epocmtracomp
        gdat.perimask = gdat.nomipara.pericomp
        gdat.fitt.duramask = 2. * gdat.nomipara.duratrantotlcomp
    else:
        gdat.epocmask = None
        gdat.perimask = None
        gdat.fitt.duramask = None

    # obtain bdtrnotr time-series bundle, the baseline-detrended light curve with no masking due to identified transiting object
    if gdat.numbinst[0] > 0 and gdat.boolbdtranyy:
        gdat.listobjtspln = [[[[] for y in gdat.indxchun[0][p]] for p in gdat.indxinst[0]] for b in gdat.indxdatatser]
        gdat.indxsplnregi = [[[[] for y in gdat.indxchun[0][p]] for p in gdat.indxinst[0]] for b in gdat.indxdatatser]
        gdat.listindxtimeregi = [[[[] for y in gdat.indxchun[0][p]] for p in gdat.indxinst[0]] for b in gdat.indxdatatser]
        gdat.indxtimeregioutt = [[[[] for y in gdat.indxchun[0][p]] for p in gdat.indxinst[0]] for b in gdat.indxdatatser]
        
        gdat.numbiterbdtr = [[0 for y in gdat.indxchun[0][p]] for p in gdat.indxinst[0]]
        numbtimecutt = [[1 for y in gdat.indxchun[0][p]] for p in gdat.indxinst[0]]
        
        print('Defining all intermediate variables related to clipping and detrending...')
        for z, timescalbdtr in enumerate(gdat.listtimescalbdtr):
            for wr in range(gdat.maxmnumbiterbdtr):
                strgarrybdtrinpt, strgarryclipoutp, strgarrybdtroutp, strgarryclipinpt, strgarrybdtrblin = retr_namebdtrclip(z, wr)
                gdat.listarrytser[strgarrybdtrinpt] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
                gdat.listarrytser[strgarryclipoutp] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
                gdat.listarrytser[strgarrybdtroutp] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
                gdat.listarrytser[strgarryclipinpt] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
                gdat.listarrytser[strgarrybdtrblin] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        
        # iterate over all detrending time scales (including, but not limited to the (first) time scale used for later analysis and model)
        gdat.indxenerclip = 0
        for z, timescalbdtr in enumerate(gdat.listtimescalbdtr):
            
            if timescalbdtr == 0:
                continue
            
            strgarrybdtr = 'bdtrts%02d' % z
            gdat.listarrytser[strgarrybdtr] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
            
            # baseline-detrending
            b = 0
            for p in gdat.indxinst[0]:
                if gdat.typeverb > 0:
                    if gdat.boolbdtr[0][p]:
                        print('Will detrend the photometric time-series before estimating the priors...')
                    else:
                        print('Will NOT detrend the photometric time-series before estimating the priors...')
                if not gdat.boolbdtr[0][p]:
                    continue

                for y in gdat.indxchun[0][p]:
                    
                    gdat.listtimebrek = None

                    if gdat.typeverb > 0:
                        print('Detrending data from %s...' % gdat.liststrgchun[0][p][y])
                    
                    indxtimetotl = np.arange(gdat.listarrytser['maskcust'][0][p][y].shape[0])
                    indxtimekeep = np.copy(indxtimetotl)
                    
                    r = 0
                    while True:
                        
                        if gdat.typeverb > 0:
                            print('Iteration %d' % r)
                        
                        # construct the variable names for this time scale and trial
                        strgarrybdtrinpt, strgarryclipoutp, strgarrybdtroutp, strgarryclipinpt, strgarrybdtrblin = retr_namebdtrclip(z, r)
                        
                        # perform trial mask
                        if gdat.typeverb > 0:
                            print('Trial filtering with %.3g percent of the data points...' % \
                                                        (100. * indxtimekeep.size / gdat.listarrytser['maskcust'][0][p][y].shape[0]))
                        gdat.listarrytser[strgarrybdtrinpt][0][p][y] = gdat.listarrytser['maskcust'][0][p][y][indxtimekeep, :, :]
                        
                        if gdat.booldiag and indxtimekeep.size < 2:
                            raise Exception('')

                        if gdat.boolplottser:
                            plot_tser_mile(gdat, 0, p, y, strgarrybdtrinpt, booltoge=False)
                        
                        # perform trial detrending
                        if gdat.typeverb > 0:
                            print('Trial detrending into %s...' % strgarryclipinpt)
                        bdtr_wrap(gdat, 0, p, y, gdat.epocmask, gdat.perimask, gdat.fitt.duramask, strgarrybdtrinpt, strgarryclipinpt, 'temp', \
                                                                                                                    timescalbdtr=timescalbdtr)
                        
                        if r == 0:
                            gdat.listtimebrekfrst = np.copy(gdat.listtimebrek)
                            gdat.numbregibdtr = len(gdat.rflxbdtrregi)
                            gdat.indxregibdtr = np.arange(gdat.numbregibdtr)
                            gdat.indxtimeregiouttfrst = [[] for gg in gdat.indxregibdtr]
                            for kk in gdat.indxregibdtr:
                                gdat.indxtimeregiouttfrst[kk] = np.copy(gdat.indxtimeregioutt[b][p][y][kk])
                        else:
                            if len(gdat.listtimebrek) != len(gdat.listtimebrekfrst):
                                print('gdat.listtimebrek')
                                print(gdat.listtimebrek)
                                print('gdat.listtimebrekfrst')
                                print(gdat.listtimebrekfrst)
                                print('Number of edges changed.')
                                raise Exception('')
                            elif gdat.boolbrekregi and ((gdat.listtimebrek[:-1] - gdat.listtimebrekfrst[:-1]) != 0.).any():
                                print('Edges moved.')
                                print('gdat.listtimebrek')
                                print(gdat.listtimebrek)
                                print('gdat.listtimebrekfrst')
                                print(gdat.listtimebrekfrst)
                                raise Exception('')

                        if gdat.boolplottser:
                            plot_tser_bdtr(gdat, b, p, y, z, r, strgarrybdtrinpt, strgarryclipinpt)
        
                            plot_tser_mile(gdat, 0, p, y, strgarryclipinpt, booltoge=False)
                
                        if gdat.typeverb > 0:
                            print('Determining outlier limits...')
                        
                        # sigma-clipping
                        lcurclip, lcurcliplowr, lcurclipuppr = scipy.stats.sigmaclip(gdat.listarrytser[strgarryclipinpt][0][p][y][:, :, 1], low=3., high=3.)
                        
                        indxtimeclipkeep = np.where((gdat.listarrytser[strgarryclipinpt][0][p][y][:, gdat.indxenerclip, 1] < lcurclipuppr) & \
                                                    (gdat.listarrytser[strgarryclipinpt][0][p][y][:, gdat.indxenerclip, 1] > lcurcliplowr))[0]
                        
                        if indxtimeclipkeep.size < 2:
                            print('No time samples left after clipping...')
                            print('gdat.listarrytser[strgarryclipinpt][0][p][y][:, gdat.indxenerclip, 1]')
                            summgene(gdat.listarrytser[strgarryclipinpt][0][p][y][:, gdat.indxenerclip, 1])
                            print('lcurcliplowr')
                            print(lcurcliplowr)
                            print('lcurclipuppr')
                            print(lcurclipuppr)
                            raise Exception('')
                        
                        #indxtimeclipmask = np.setdiff1d(np.arange(gdat.listarrytser[strgarryclipinpt][0][p][y][:, gdat.indxenerclip, 1].size), indxtimeclipkeep)
                        
                        # cluster indices of masked times
                        #listindxtimemaskclus = []
                        #for k in range(len(indxtimemask)):
                        #    if k == 0 or indxtimemask[k] != indxtimemask[k-1] + 1:
                        #        listindxtimemaskclus.append([indxtimemask[k]])
                        #    else:
                        #        listindxtimemaskclus[-1].append(indxtimemask[k])
                        #print('listindxtimemaskclus')
                        #print(listindxtimemaskclus)
                        
                        #print('Filtering clip times with index indxtimekeep into %s...' % strgarryclipoutp)
                        #gdat.listarrytser[strgarryclipoutp][0][p][y] = gdat.listarrytser['maskcust'][0][p][y][indxtimekeep, :]
                        
                        #print('Thinning the mask...')
                        #indxtimeclipmask = np.random.choice(indxtimeclipmask, size=int(indxtimeclipmask.size*0.7), replace=False)
                        
                        #print('indxtimeclipmask')
                        #summgene(indxtimeclipmask)

                        #indxtimeclipkeep = np.setdiff1d(np.arange(gdat.listarrytser[strgarryclipinpt][0][p][y][:, 1].size), indxtimeclipmask)
                        
                        indxtimekeep = indxtimekeep[indxtimeclipkeep]
                        
                        if gdat.booldiag and indxtimekeep.size < 2:
                            print('indxtimekeep')
                            print(indxtimekeep)
                            raise Exception('')

                        #boolexit = True
                        #for k in range(len(listindxtimemaskclus)):
                        #    # decrease mask

                        #    # trial detrending
                        #    bdtr_wrap(gdat, 0, p, y, gdat.epocmask, gdat.perimask, gdat.fitt.duramask, strgarrybdtrinpt, strgarryclipinpt, 'temp', timescalbdtr=timescalbdtr)
                        #    
                        #    chi2 = np.sum((gdat.listarrytser[strgarryclipinpt][0][p][y][:, 1] - gdat.listarrytser[strgarryclipinpt][0][p][y][:, 1])**2 / 
                        #                                                   gdat.listarrytser[strgarryclipinpt][0][p][y][:, 2]**2) / gdat.listarrytser[strgarryclipinpt][0][p][y][:, 1].size
                        #    if chi2 > 1.1:
                        #        boolexit = False
                        #
                        #    if gdat.boolplottser:
                        #        plot_tser_mile(gdat, 0, p, y, strgarryclipoutp, booltoge=False)
                        
                        #if gdat.boolplottser:
                        #    plot_tser_mile(gdat, 0, p, y, strgarrybdtroutp, booltoge=False)
                        

                        if r == gdat.maxmnumbiterbdtr - 1 or gdat.listarrytser[strgarryclipinpt][0][p][y][:, gdat.indxenerclip, 1].size == indxtimeclipkeep.size:
                            rflxtren = []
                            for kk in gdat.indxregibdtr:
                                if gdat.typebdtr == 'GaussianProcess':
                                    rflxtren.append(gdat.listobjtspln[b][p][y][kk].predict( \
                                                                 gdat.listarrytser['maskcust'][b][p][y][gdat.indxtimeregioutt[b][p][y][kk], gdat.indxenerclip, 1], \
                                                                      t=gdat.listarrytser['maskcust'][b][p][y][:, gdat.indxenerclip, 0], \
                                                                                                                             return_cov=False, return_var=False))
                                    
                                if gdat.typebdtr == 'Spline':
                                    rflxtren.append(gdat.listobjtspln[b][p][y][kk](gdat.listarrytser['maskcust'][b][p][y][:, gdat.indxenerclip, 0]))
                            gdat.listarrytser[strgarrybdtr][0][p][y] = np.copy(gdat.listarrytser['maskcust'][0][p][y])
                            
                            gdat.listarrytser[strgarrybdtr][0][p][y][:, gdat.indxenerclip, 1] = \
                                            1. + gdat.listarrytser['maskcust'][0][p][y][:, gdat.indxenerclip, 1] - np.concatenate(rflxtren)
                        
                            if r == gdat.maxmnumbiterbdtr - 1:
                                print('Maximum number of trial detrending iterations attained. Breaking the loop...')
                            if gdat.listarrytser[strgarryclipinpt][0][p][y][:, gdat.indxenerclip, 1].size == indxtimeclipkeep.size:
                                print('No more clipping is needed. Breaking the loop...')
                            if gdat.typeverb > 0:
                                print('')
                                print('')
                            
                            break
                            
                        else:
                            print('Have not achieved the desired stability in iteration %d. Will reiterate...' % r)
                            print('')
                            
                            r += 1
                        
                            if gdat.typeverb > 0:
                                print('')
                                print('')

                    # write the baseline-detrended light curve for this time scale
                    path = gdat.pathdatatarg + '%s_DataCube_Detrended_%s_%s%s%s_ts%02d.csv' % (gdat.liststrgdatatser[0], gdat.liststrginst[0][p], gdat.liststrgchun[0][p][y], \
                                                                                                            gdat.strgextncade[b][p], gdat.liststrgener[p][e], z)
                    if not os.path.exists(path):
                        if gdat.typeverb > 0:
                            print('Writing to %s...' % path)
                        np.savetxt(path, gdat.listarrytser[strgarrybdtr][0][p][y][:, e, :], delimiter=',', header=gdat.strgheadtser[0])
        
        # place the output of detrending into the baseline-detrended 'Detrended' light curve
        if gdat.listtimescalbdtr[0] == 0.:
            gdat.listarrytser['Detrended'] = gdat.listarrytser['maskcust']
        else:
            gdat.listarrytser['Detrended'] = gdat.listarrytser['bdtrts00']

        # merge chunks
        for p in gdat.indxinst[0]:
            gdat.arrytser['Detrended'][0][p] = np.concatenate(gdat.listarrytser['Detrended'][0][p], 0)
        
        # write baseline-detrended light curve
        for p in gdat.indxinst[0]:
            
            if not gdat.boolbdtr[0][p]:
                continue

            for e in gdat.indxener[p]:

                if gdat.numbchun[0][p] > 1:
                    path = gdat.pathdatatarg + '%s_DataCube_Detrended_%s%s%s.csv' % (gdat.liststrgdatatser[0], gdat.liststrginst[0][p], gdat.strgextncade[b][p], gdat.liststrgener[p][e])
                    if not os.path.exists(path):
                        if gdat.typeverb > 0:
                            print('Writing to %s...' % path)
                        np.savetxt(path, gdat.arrytser['Detrended'][0][p][:, e, :], delimiter=',', header=gdat.strgheadtser[0])
                
                for y in gdat.indxchun[0][p]:
                    path = gdat.pathdatatarg + '%s_DataCube_Detrended_%s_%s%s%s.csv' % (gdat.liststrgdatatser[0], gdat.liststrginst[0][p], \
                                                                                        gdat.liststrgchun[0][p][y], gdat.strgextncade[b][p], gdat.liststrgener[p][e])
                    if not os.path.exists(path):
                        if gdat.typeverb > 0:
                            print('Writing to %s...' % path)
                        np.savetxt(path, gdat.listarrytser['Detrended'][0][p][y][:, e, :], delimiter=',', header=gdat.strgheadtser[0])
    
        if gdat.boolplottser:
            for p in gdat.indxinst[0]:
                if gdat.boolbdtr[0][p]:
                    for y in gdat.indxchun[0][p]:
                        plot_tser_mile(gdat, 0, p, y, 'Detrended')
                    plot_tser_mile(gdat, 0, p, None, 'Detrended')
    
        # update the time axis since some data may have been clipped
        setp_time(gdat, 'Detrended')

    else:
        gdat.arrytser['Detrended'] = gdat.arrytser['maskcust']
        gdat.listarrytser['Detrended'] = gdat.listarrytser['maskcust']
    
    if gdat.booldiag:
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                if len(gdat.arrytser['Detrended'][b][p]) == 0:
                    raise Exception('')

    if gdat.boolsrchflar:
        # size of the window for the flare search
        gdat.sizewndwflar = np.empty(gdat.numbinst[0], dtype=int)
        for p in gdat.indxinst[0]:
            gdat.sizewndwflar[p] = int(3600. / gdat.cadetime[0][p])
    
    # rebinning
    gdat.numbrebn = 50
    gdat.indxrebn = np.arange(gdat.numbrebn)
    gdat.listdeltrebn = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            gdat.minmdeltrebn = max(100. * gdat.cadetime[b][p], 0.1 * 0.3 * (gdat.timeconc[0][-1] - gdat.timeconc[0][0]))
            gdat.maxmdeltrebn =  0.3 * (gdat.timeconc[0][-1] - gdat.timeconc[0][0])
            gdat.listdeltrebn[b][p] = np.linspace(gdat.minmdeltrebn, gdat.maxmdeltrebn, gdat.numbrebn)
    
    if gdat.booldiag:
        if not (gdat.boolsrchoutlperi or gdat.boolsrchboxsperi):
            if gdat.numbband != len(gdat.fitt.prio.meanpara.rratcomp):
                print('')
                print('')
                print('')
                raise Exception('gdat.numbband != len(gdat.fitt.prio.meanpara.rratcomp)')

    print('gdat.boolsrchoutlperi')
    print(gdat.boolsrchoutlperi)
    # match the outliers to search for periodicity
    if gdat.boolsrchoutlperi:
        
        listarry = []
        # temp
        for p in gdat.indxinst[0]:
            # input data to the periodic box search pipeline
            arry = np.copy(gdat.arrytser['Detrended'][0][p][:, 0, :])
            listarry.append(arry)
        arry = np.concatenate(listarry, 0)
        time = arry[:, 0]
        flux = arry[:, 1]
        stdvflux = arry[:, 2]
        
        gdat.dictoutlperi = srch_outlperi(time, flux, stdvflux)
            
        if gdat.dictoutlperi['boolposi']:
            gdat.fitt.prio.meanpara.pericomp = np.array([gdat.dictoutlperi['peri']])
            gdat.fitt.prio.meanpara.epocmtracomp = np.array([gdat.dictoutlperi['epoc']])
            gdat.fitt.prio.meanpara.rsmacomp = np.array([0.1])
            
            print('temp')
            gdat.fitt.prio.meanpara.depttrancomp = np.array([0.01])
            
            gdat.fitt.prio.meanpara.booltrancomp = np.ones_like(gdat.fitt.prio.meanpara.depttrancomp, dtype=bool)
            
            gdat.fitt.prio.meanpara.cosicomp = np.array([0.])
            
        gdat.dictmileoutp['dictoutlperi'] = gdat.dictoutlperi
            
    if gdat.typepriocomp == 'outlperi':
        gdat.fitt.prio.numbcomp = len(gdat.dictoutlperi['epoc'])
        
    if gdat.booldiag:

        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                for name in gdat.arrytser:
                    if (abs(gdat.arrytser[name][b][p]) > 1e10).any():
                        print('')
                        print('')
                        print('')
                        print('name')
                        print(name)
                        print('gdat.arrytser[name][b][p][:, 0]')
                        summgene(gdat.arrytser[name][b][p][:, 0])
                        print('gdat.arrytser[name][b][p][:, 1]')
                        summgene(gdat.arrytser[name][b][p][:, 1])
                        print('gdat.arrytser[name][b][p][:, 2]')
                        summgene(gdat.arrytser[name][b][p][:, 2])
                        raise Exception('')
                    for y in gdat.indxchun[b][p]:
                        if (abs(gdat.listarrytser[name][b][p][y]) > 1e10).any():
                            raise Exception('')
    
    
    if gdat.boolsrchoutlperi or gdat.boolsrchboxsperi:
        # Boolean flag to merge different bands during analyses on time-series photometric data
        gdat.boolanlsbandmerg = True

    # search for periodic boxes
    if gdat.boolsrchboxsperi:
        
        if gdat.dictboxsperiinpt is None:
            gdat.dictboxsperiinpt = dict()
            
        if 'boolsrchposi' in gdat.dictboxsperiinpt:
            raise Exception('Conflicting entry in dictboxsperiinpt.')
        else:
            gdat.dictboxsperiinpt['boolsrchposi'] = gdat.boolsrchboxsperiposi
        
        print('temp: an optional, model-dependent, physics-based minmperi should be implemented')
        #gdat.dictboxsperiinpt['minmperi'] = 

        gdat.dictboxsperiinpt['typefileplot'] = gdat.typefileplot
        gdat.dictboxsperiinpt['figrsizeydobskin'] = gdat.figrsizeydobskin
        gdat.dictboxsperiinpt['alphdata'] = gdat.alphdata
        
        if not 'typeverb' in gdat.dictboxsperiinpt:
            gdat.dictboxsperiinpt['typeverb'] = gdat.typeverb
        
        if not 'pathvisu' in gdat.dictboxsperiinpt:
            if gdat.boolplot:
                gdat.dictboxsperiinpt['pathvisu'] = gdat.pathvisutarg
            
        gdat.dictboxsperiinpt['pathdata'] = gdat.pathdatatarg
        gdat.dictboxsperiinpt['timeoffs'] = gdat.timeoffs
        
        if gdat.boolanlsbandmerg:
            gdat.numbanlsband = 1
        else:
            gdat.numbanlsband = gdat.numbband
        gdat.indxanlsband = np.arange(gdat.numbanlsband)

        for tk in gdat.indxanlsband:
            
            # input data to the periodic box search pipeline
            if gdat.boolanlsbandmerg:
                listarry = []
                for tk_ in gdat.indxanlsband:
                    listarry.append(gdat.arrytser['Detrended'][0][tk_][:, 0, :])
                arry = np.concatenate(listarry)
                strgextn = 'Merged'
            else:
                arry = np.copy(gdat.arrytser['Detrended'][0][tk][:, 0, :])
                strgextn = '%s' % gdat.liststrginst[0][tk]
            
            strgextn += '_%s' % gdat.strgtarg

            gdat.dictboxsperiinpt['strgextn'] = strgextn
            
            gdat.dictboxsperioutp = srch_boxsperi(arry, **gdat.dictboxsperiinpt)
            
            gdat.dictmileoutp['dictboxsperioutp'] = gdat.dictboxsperioutp
            
            if gdat.boolanlsbandmerg:
                if not hasattr(gdat.fitt.prio.meanpara, 'epocmtracomp'):
                    gdat.fitt.prio.meanpara.epocmtracomp = gdat.dictboxsperioutp['epoc']
                if not hasattr(gdat.fitt.prio.meanpara, 'pericomp'):
                    gdat.fitt.prio.meanpara.pericomp = gdat.dictboxsperioutp['peri']
                gdat.fitt.prio.meanpara.depttrancomp = 1. - 1e-3 * gdat.dictboxsperioutp['ampl']
                gdat.fitt.prio.meanpara.booltrancomp = np.ones_like(gdat.fitt.prio.meanpara.depttrancomp, dtype=bool)
                gdat.fitt.prio.meanpara.duratrantotlcomp = gdat.dictboxsperioutp['dura']
                gdat.fitt.prio.meanpara.cosicomp = np.zeros_like(gdat.dictboxsperioutp['epoc']) 
                gdat.fitt.prio.meanpara.rsmacomp = np.sin(np.pi * gdat.fitt.prio.meanpara.duratrantotlcomp / gdat.fitt.prio.meanpara.pericomp / 24.)
                
                gdat.perimask = gdat.fitt.prio.meanpara.pericomp
                gdat.epocmask = gdat.fitt.prio.meanpara.epocmtracomp
                gdat.fitt.duramask = 2. * gdat.fitt.prio.meanpara.duratrantotlcomp
    
    if gdat.boolsrchoutlperi or gdat.boolsrchboxsperi:
       gdat.fitt.prio.meanpara.rratcomp = [[] for pk in gdat.indxband]
       if gdat.boolanlsbandmerg:
           for pk in gdat.indxband:
                gdat.fitt.prio.meanpara.rratcomp[pk] = np.sqrt(1e-3 * gdat.fitt.prio.meanpara.depttrancomp)

    print('gdat.fitt.prio.meanpara.rratcomp')
    print(gdat.fitt.prio.meanpara.rratcomp)

    if gdat.booldiag:
        if gdat.numbband != len(gdat.fitt.prio.meanpara.rratcomp):
            print('')
            print('')
            print('')
            raise Exception('gdat.numbband != len(gdat.fitt.prio.meanpara.rratcomp)')

    # transfer true parameters to nominal parameters
    if gdat.boolsimusome  and gdat.booltargsynt:
        for namepara in ['peri', 'epocmtra', 'cosi', 'rsma', 'rrat']:
            setattr(gdat.nomipara, namepara + 'comp', gdat.dicttrue[namepara+'comp'])

    if gdat.typepriocomp == 'boxsperinega' or gdat.typepriocomp == 'boxsperiposi':
        gdat.fitt.prio.numbcomp = len(gdat.dictboxsperioutp['epoc'])
    
    if gdat.typeverb > 0:
        print('gdat.epocmask')
        print(gdat.epocmask)
        print('gdat.perimask')
        print(gdat.perimask)
        print('gdat.fitt.duramask')
        print(gdat.fitt.duramask)
    
    # define number of components
    if gdat.boolmodl and gdat.fitt.boolmodlpsys and (gdat.boolsrchoutlperi or gdat.boolsrchboxsperi):
        gdat.fitt.prio.numbcomp = gdat.fitt.prio.meanpara.epocmtracomp.size
        gdat.fitt.prio.indxcomp = np.arange(gdat.fitt.prio.numbcomp)
    
    # find time samples inside estimated transits
    # periodic outlier search is not included because it does not estimate the transit duration.
    if gdat.boolmodl and gdat.fitt.boolmodlpsys and gdat.boolsrchboxsperi:
        retr_timetran(gdat, 'Raw')

    # search for flares
    if gdat.boolsrchflar:
        dictsrchflarinpt['pathvisu'] = gdat.pathvisutarg
        
        gmod.listindxtimeflar = [[[] for y in gdat.indxchun[0][p]] for p in gdat.indxinst[0]]
        gdat.listmdetflar = [[[] for y in gdat.indxchun[0][p]] for p in gdat.indxinst[0]]
        gdat.precphot = [np.empty(gdat.numbchun[0][p]) for p in gdat.indxinst[0]]
        gdat.thrsrflxflar = [np.empty(gdat.numbchun[0][p]) for p in gdat.indxinst[0]]
        
        for p in gdat.indxinst[0]:
            for y in gdat.indxchun[0][p]:
                gdat.listarrytser['bdtrmedi'][0][p][y] = np.empty_like(gdat.listarrytser['Detrended'][0][p][y])
                gdat.listarrytser['bdtrlowr'][0][p][y] = np.empty_like(gdat.listarrytser['Detrended'][0][p][y])
                gdat.listarrytser['bdtruppr'][0][p][y] = np.empty_like(gdat.listarrytser['Detrended'][0][p][y])

                if gdat.typemodlflar == 'outl':
                    listydat = gdat.listarrytser['Detrended'][0][p][y][:, 0, 1]
                    numbtime = listydat.size
                    tsermedi = np.empty(numbtime)
                    tseruppr = np.empty(numbtime)
                    for t in range(listydat.size):
                        # time-series of the median inside a window
                        minmindxtimewind = max(0, t - gdat.sizewndwflar)
                        maxmindxtimewind = min(numbtime - 1, t + gdat.sizewndwflar)
                        indxtimewind = np.arange(minmindxtimewind, maxmindxtimewind + 1)
                        lowr, medi, uppr = np.percentile(listydat[indxtimewind], [5., 50., 95.])
                        gdat.listarrytser['bdtrlowr'][0][p][y][t, 0, 1] = lowr
                        gdat.listarrytser['bdtrmedi'][0][p][y][t, 0, 1] = medi
                        gdat.listarrytser['bdtruppr'][0][p][y][t, 0, 1] = uppr
                        
                        # time-series of the decision boundary
                        #indxcent = np.where((listydat > np.percentile(listydat, 1.)) & (listydat < np.percentile(listydat, 99.)))[0]
                        
                        # standard deviation inside the window without the outliers
                        #stdv = np.std(listydat[indxcent])
                        
                        #gdat.precphot[p][y] = stdv
                        stdv = uppr - lowr
                        listmdetflar = (listydat[t] - medi) / stdv
                        indxtimeposi = np.where(listmdetflar > gdat.thrssigmflar)[0]
                    
                    for n in range(len(indxtimeposi)):
                        if (n == len(indxtimeposi) - 1) or (n < len(indxtimeposi) - 1) and not ((indxtimeposi[n] + 1) in indxtimeposi):
                            gmod.listindxtimeflar[p][y].append(indxtimeposi[n])
                            mdetflar = listmdetflar[indxtimeposi[n]]
                            gdat.listmdetflar[p][y].append(mdetflar)
                    gmod.listindxtimeflar[p][y] = np.array(gmod.listindxtimeflar[p][y])
                    gdat.listmdetflar[p][y] = np.array(gdat.listmdetflar[p][y])

                if gdat.typemodlflar == 'tmpl':
                    dictsrchflaroutp = srch_flar(gdat.arrytser['Detrended'][0][p][:, 0], gdat.arrytser['Detrended'][0][p][:, 1], **dictsrchflarinpt)
            
        gdat.dictmileoutp['listindxtimeflar'] = gmod.listindxtimeflar
        gdat.dictmileoutp['listmdetflar'] = gdat.listmdetflar
        gdat.dictmileoutp['precphot'] = gdat.precphot
        
        for p in gdat.indxinst[0]:
            for y in gdat.indxchun[0][p]:
                if gdat.boolplottser:
                    plot_tser_mile(gdat, 0, p, y, 'Detrended', boolflar=True)
            if gdat.boolplottser:
                plot_tser_mile(gdat, 0, p, None, 'Detrended', boolflar=True)
        
        if gdat.typeverb > 0:
            print('temp: skipping masking out of flaress...')
        # mask out flares
        #numbkern = len(maxmcorr)
        #indxkern = np.arange(numbkern)
        #listindxtimemask = []
        #for k in indxkern:
        #    for indxtime in gmod.listindxtimeposimaxm[k]:
        #        indxtimemask = np.arange(indxtime - 60, indxtime + 60)
        #        listindxtimemask.append(indxtimemask)
        #indxtimemask = np.concatenate(listindxtimemask)
        #indxtimemask = np.unique(indxtimemask)
        #indxtimegood = np.setdiff1d(np.arange(gdat.time.size), indxtimemask)
        #gdat.time = gdat.time[indxtimegood]
        #gdat.tserdata = gdat.tserdata[indxtimegood]
        #gdat.tserdatastdv = gdat.tserdatastdv[indxtimegood]
        #gdat.numbtime = gdat.time.size

    # data validation (DV) report
    ## number of pages in the DV report
    if gdat.boolplot:
        # first page
        gdat.numbpage = 1
        
        # separate page for each component
        if gdat.fitt.prio.numbcomp is not None:
            gdat.numbpage += gdat.fitt.prio.numbcomp
        
        gdat.indxpage = np.arange(gdat.numbpage)
        
        for w in gdat.indxpage:
            gdat.listdictdvrp.append([])
        
        # add boxsperi plots to the DV report
        if gdat.boolsrchboxsperi and gdat.boolplot:
            for p in gdat.indxinst[0]:
                for g, name in enumerate(['ampl', 'sgnl', 'stdvsgnl', 's2nr', 'pcur', 'rflx']):
                    for j in range(len(gdat.dictboxsperioutp['epoc'])):
                        gdat.listdictdvrp[j+1].append({'path': gdat.dictboxsperioutp['listpathplot%s' % name][j], 'limt':[0., 0.9 - g * 0.1, 0.5, 0.1]})
    
    if gdat.booldiag:
        if gdat.numbband != len(gdat.fitt.prio.meanpara.rratcomp):
            print('')
            print('')
            print('')
            raise Exception('gdat.numbband != len(gdat.fitt.prio.meanpara.rratcomp)')

    gdat.dictmileoutp['numbcompprio'] = gdat.fitt.prio.numbcomp
    
    # calculate LS periodogram
    if gdat.boolcalclspe:
        if gdat.boolplot:
            pathvisulspe = gdat.pathvisutarg
        else:
            pathvisulspe = None

        liststrgarrylspe = ['Raw']
        if gdat.boolbdtranyy:
            liststrgarrylspe += ['Detrended']
        for b in gdat.indxdatatser:
            
            # temp -- neglects LS periodograms of RV data
            if b == 1:
                continue
            
            if gdat.numbinst[b] > 0:
                
                for h in gdat.indxfittiter:
                    if gdat.numbinst[b] > 1:
                        strgextn = '%s%s_%s' % (gdat.liststrgdatatser[b], gdat.liststrgdatafittiter[h], gdat.strgtarg)
                        gdat.dictlspeoutp = exec_lspe(gdat.arrytsertotl[b][:, e, :], pathvisu=pathvisulspe, strgextn=strgextn, maxmfreq=maxmfreqlspe, \
                                                                                  typeverb=gdat.typeverb, typefileplot=gdat.typefileplot, pathdata=gdat.pathdatatarg)
                    
                    for p in gdat.indxinst[b]:
                        for strg in liststrgarrylspe:
                            strgextn = '%s_%s_%s%s_%s' % (strg, gdat.liststrgdatatser[b], gdat.liststrginst[b][p], gdat.liststrgdatafittiter[h], gdat.strgtarg) 
                            gdat.dictlspeoutp = exec_lspe(gdat.arrytser[strg][b][p][:, e, :], pathvisu=pathvisulspe, strgextn=strgextn, maxmfreq=maxmfreqlspe, \
                                                                                  typeverb=gdat.typeverb, typefileplot=gdat.typefileplot, pathdata=gdat.pathdatatarg)
        
                    gdat.dictmileoutp['perilspempow'] = gdat.dictlspeoutp['perimpow']
                    gdat.dictmileoutp['powrlspempow'] = gdat.dictlspeoutp['powrmpow']
                    
                    if gdat.boolplot:
                        gdat.listdictdvrp[0].append({'path': gdat.dictlspeoutp['pathplot'], 'limt':[0., 0.8, 0.5, 0.1]})
        
    if gdat.typeverb > 0:
        print('Planet letters: ')
        print(gdat.liststrgcomp)
    
    if gdat.fitt.prio.meanpara.duratrantotlcomp is None:
        
        if gdat.booldiag:
            if gdat.fitt.prio.meanpara.pericomp is None or gdat.fitt.prio.meanpara.rsmacomp is None or gdat.fitt.prio.meanpara.cosicomp is None:
                print('')
                print('')
                print('')
                print('gdat.fitt.prio.meanpara.pericomp')
                print(gdat.fitt.prio.meanpara.pericomp)
                print('gdat.fitt.prio.meanpara.rsmacomp')
                print(gdat.fitt.prio.meanpara.rsmacomp)
                print('gdat.fitt.prio.meanpara.cosicomp')
                print(gdat.fitt.prio.meanpara.cosicomp)
                raise Exception('')

        gdat.fitt.prio.meanpara.duratrantotlcomp = nicomedia.retr_duratrantotl(gdat.fitt.prio.meanpara.pericomp, gdat.fitt.prio.meanpara.rsmacomp, gdat.fitt.prio.meanpara.cosicomp)
        if gdat.booldiag:
            if (gdat.fitt.prio.meanpara.duratrantotlcomp == 0).any():
                print('')
                print('')
                print('')
                print('gdat.fitt.prio.meanpara.pericomp')
                print(gdat.fitt.prio.meanpara.pericomp)
                print('gdat.fitt.prio.meanpara.rsmacomp')
                print(gdat.fitt.prio.meanpara.rsmacomp)
                print('gdat.fitt.prio.meanpara.cosicomp')
                print(gdat.fitt.prio.meanpara.cosicomp)
                print('gdat.fitt.prio.meanpara.duratrantotlcomp')
                print(gdat.fitt.prio.meanpara.duratrantotlcomp)
                raise Exception('(gdat.fitt.prio.meanpara.duratrantotlcomp == 0).any()')
    
    if gdat.boolmodl:
        if not gdat.boolsrchboxsperi:
            for pk in gdat.indxband:
                gdat.fitt.prio.meanpara.rratcomp[pk] = np.sqrt(1e-3 * gdat.fitt.prio.meanpara.depttrancomp)
    
        if gdat.fitt.prio.meanpara.rsmacomp is None:
            gdat.fitt.prio.meanpara.rsmacomp = np.sqrt(np.sin(np.pi * gdat.fitt.prio.meanpara.duratrantotlcomp / \
                                                                        gdat.fitt.prio.meanpara.pericomp / 24.)**2 + gdat.fitt.prio.meanpara.cosicomp**2)
    if gdat.ecoscompprio is None:
        gdat.ecoscompprio = np.zeros(gdat.fitt.prio.numbcomp)
    
    if gdat.esincompprio is None:
        
        if gdat.booldiag:
            if gdat.fitt.prio.numbcomp is None:
                print('')
                print('')
                print('')
                raise Exception('gdat.fitt.prio.numbcomp is None')
        
        gdat.esincompprio = np.zeros(gdat.fitt.prio.numbcomp)
    if gdat.rvelsemaprio is None:
        gdat.rvelsemaprio = np.zeros(gdat.fitt.prio.numbcomp)
    
    if gdat.stdvrratcompprio is None:
        gdat.stdvrratcompprio = 0.01 + np.zeros(gdat.fitt.prio.numbcomp)
    if gdat.stdvrsmacompprio is None:
        gdat.stdvrsmacompprio = 0.01 + np.zeros(gdat.fitt.prio.numbcomp)
    if gdat.stdvepocmtracompprio is None:
        gdat.stdvepocmtracompprio = 0.1 + np.zeros(gdat.fitt.prio.numbcomp)
    if gdat.stdvpericompprio is None:
        gdat.stdvpericompprio = 0.01 + np.zeros(gdat.fitt.prio.numbcomp)
    if gdat.stdvcosicompprio is None:
        gdat.stdvcosicompprio = 0.05 + np.zeros(gdat.fitt.prio.numbcomp)
    if gdat.stdvecoscompprio is None:
        gdat.stdvecoscompprio = 0.1 + np.zeros(gdat.fitt.prio.numbcomp)
    if gdat.stdvesincompprio is None:
        gdat.stdvesincompprio = 0.1 + np.zeros(gdat.fitt.prio.numbcomp)
    if gdat.stdvrvelsemaprio is None:
        gdat.stdvrvelsemaprio = 0.001 + np.zeros(gdat.fitt.prio.numbcomp)
    
    # others
    if gdat.projoblqprio is None:
        gdat.projoblqprio = 0. + np.zeros(gdat.fitt.prio.numbcomp)
    if gdat.stdvprojoblqprio is None:
        gdat.stdvprojoblqprio = 10. + np.zeros(gdat.fitt.prio.numbcomp)
    
    # order planets with respect to period
    if gdat.boolmodl and (gdat.fitt.boolmodlpsys or gdat.fitt.typemodl == 'CompactObjectStellarCompanion'):
        if gdat.typepriocomp != 'inpt' and False:
            
            if gdat.typeverb > 0:
                print('Sorting the planets with respect to orbital period...')
            
            indxcompsort = np.argsort(gdat.fitt.prio.meanpara.pericomp)
            
            strgpara = 'rratco%042' % ()
            for pk in gdat.indxband:
                gdat.fitt.prio.meanpara.rratcomp[pk] = gdat.fitt.prio.meanpara.rratcomp[pk][indxcompsort]
            
            print('temp: do this for other parameters as well')
            for namepara in ['rsma', 'epocmtra', 'peri', 'cosi']:
                setattr(gdat.fitt.prio.meanpara, namepara + 'comp', getattr(gdat.fitt.prio.meanpara, namepara + 'comp')[indxcompsort])
            
            gdat.fitt.prio.meanpara.duratrantotlcomp = gdat.fitt.prio.meanpara.duratrantotlcomp[indxcompsort]
    
            gdat.liststrgcomp = gdat.liststrgcomp[indxcompsort]
            gdat.listcolrcomp = gdat.listcolrcomp[indxcompsort]
    
    if gdat.booldiag:
        if gdat.fitt.prio.meanpara.pericomp is None or gdat.fitt.prio.meanpara.duratrantotlcomp is None:
            print('')
            print('')
            print('')
            raise Exception('gdat.fitt.prio.meanpara.pericomp or gdat.fitt.prio.meanpara.duratrantotlcomp is None.')

    # if stellar features are NaN, use Solar defaults
    for featstar in gdat.listfeatstar:
        if not hasattr(gdat, featstar) or getattr(gdat, featstar) is None or not np.isfinite(getattr(gdat, featstar)):
            if featstar == 'radistar':
                setattr(gdat, featstar, 1.)
            if featstar == 'massstar':
                setattr(gdat, featstar, 1.)
            if featstar == 'tmptstar':
                setattr(gdat, featstar, 5778.)
            if featstar == 'vsiistar':
                setattr(gdat, featstar, 1e3)
            if gdat.typeverb > 0:
                print('Setting %s to the Solar value!' % featstar)

    # if stellar feature uncertainties are NaN, use 10%
    for featstar in gdat.listfeatstar:
        if (not hasattr(gdat, 'stdv' + featstar) or getattr(gdat, 'stdv' + featstar) is None or not np.isfinite(getattr(gdat, 'stdv' + featstar))) \
                                                                    and not (featstar == 'rascstar' or featstar == 'declstar'):
            varb = hasattr(gdat, featstar)
            if varb is not None:
                setattr(gdat, 'stdv' + featstar, 0.5 * varb)
            if gdat.typeverb > 0:
                print('Setting %s uncertainty to 50%%!' % featstar)
    
    if gdat.booldiag:
        if gdat.numbband != len(gdat.fitt.prio.meanpara.rratcomp):
            print('')
            print('')
            print('')
            raise Exception('gdat.numbband != len(gdat.fitt.prio.meanpara.rratcomp)')

    if gdat.fitt.boolvarirratband:
        gdat.fitt.prio.radicomp = [[] for pk in gdat.indxband]
        for pk in gdat.indxband:
            gdat.fitt.prio.radicomp[pk] = gdat.fitt.prio.meanpara.rratcomp[pk] * gdat.radistar
    else:
        gdat.fitt.prio.radicomp = [[] for pk in gdat.indxband]
        for pk in gdat.indxband:
            gdat.fitt.prio.radicomp[pk] = gdat.fitt.prio.meanpara.rratcomp[pk] * gdat.radistar
    
    if gdat.typeverb > 0:
        
        print('Fitting model:')
        if gdat.fitt.typemodl == 'PlanetarySystem' or gdat.fitt.typemodl == 'PlanetarySystemEmittingCompanion':
            print('Stellar priors:')
            for nameparastar in ['rasc', 'decl', 'radi', 'mass', 'vsii', 'tmpt']:
                for strgstdv in ['', 'stdv']:
                    nameparastartotl = strgstdv + nameparastar + 'star'
                    if hasattr(gdat.fitt, nameparastartotl):
                        print(nameparastartotl)
                        print(getattr(gdat.fitt, nameparastartotl))
            
            print('Planetary priors:')
            print('gdat.fitt.prio.meanpara.duratrantotlcomp')
            print(gdat.fitt.prio.meanpara.duratrantotlcomp)
            print('gdat.fitt.prio.meanpara.rratcomp')
            print(gdat.fitt.prio.meanpara.rratcomp)
            print('gdat.fitt.prio.meanpara.rsmacomp')
            print(gdat.fitt.prio.meanpara.rsmacomp)
            print('gdat.fitt.prio.meanpara.epocmtracomp')
            print(gdat.fitt.prio.meanpara.epocmtracomp)
            print('gdat.fitt.prio.meanpara.pericomp')
            print(gdat.fitt.prio.meanpara.pericomp)
            print('gdat.fitt.prio.meanpara.cosicomp')
            print(gdat.fitt.prio.meanpara.cosicomp)
            print('gdat.ecoscompprio')
            print(gdat.ecoscompprio)
            print('gdat.esincompprio')
            print(gdat.esincompprio)
            print('gdat.rvelsemaprio')
            print(gdat.rvelsemaprio)
            print('gdat.stdvrratcompprio')
            print(gdat.stdvrratcompprio)
            print('gdat.stdvrsmacompprio')
            print(gdat.stdvrsmacompprio)
            print('gdat.stdvepocmtracompprio')
            print(gdat.stdvepocmtracompprio)
            print('gdat.stdvpericompprio')
            print(gdat.stdvpericompprio)
            print('gdat.stdvcosicompprio')
            print(gdat.stdvcosicompprio)
            print('gdat.stdvecoscompprio')
            print(gdat.stdvecoscompprio)
            print('gdat.stdvesincompprio')
            print(gdat.stdvesincompprio)
            print('gdat.stdvrvelsemaprio')
            print(gdat.stdvrvelsemaprio)
    
            if not np.isfinite(gdat.fitt.prio.meanpara.rratcomp).all():
                print('rrat is infinite!')
            if not np.isfinite(gdat.fitt.prio.meanpara.rsmacomp).all():
                print('rsma is infinite!')
            if not np.isfinite(gdat.fitt.prio.meanpara.epocmtracomp).all():
                print('epoc is infinite!')
            if not np.isfinite(gdat.fitt.prio.meanpara.pericomp).all():
                print('peri is infinite!')
            if not np.isfinite(gdat.fitt.prio.meanpara.cosicomp).all():
                print('cosi is infinite!')
            if not np.isfinite(gdat.ecoscompprio).all():
                print('ecos is infinite!')
            if not np.isfinite(gdat.esincompprio).all():
                print('esin is infinite!')
            if not np.isfinite(gdat.rvelsemaprio).all():
                print('rvelsema is infinite!')

    # carry over RV data as is, without any detrending
    gdat.arrytser['Detrended'][1] = gdat.arrytser['Raw'][1]
    gdat.listarrytser['Detrended'][1] = gdat.listarrytser['Raw'][1]
    
    if gdat.booldiag:
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                for y in gdat.indxchun[b][p]:
                    if len(gdat.listarrytser['Detrended'][b][p][y]) == 0:
                        print('')
                        print('')
                        print('')
                        print('bpy')
                        print(b, p, y)
                        raise Exception('len(gdat.listarrytser[bdtr][b][p][y]) == 0')
        
                if not np.isfinite(gdat.arrytser['Detrended'][b][p]).all():
                    print('')
                    print('')
                    print('')
                    print('b, p')
                    print(b, p)
                    indxbadd = np.where(~np.isfinite(gdat.arrytser['Detrended'][b][p]))[0]
                    print('gdat.arrytser[bdtr][b][p]')
                    summgene(gdat.arrytser['Detrended'][b][p])
                    print('indxbadd')
                    summgene(indxbadd)
                    raise Exception('not np.isfinite(gdat.arrytser[bdtr][b][p]).all()')
    
    if gdat.listindxchuninst is None:
        gdat.listindxchuninst = [gdat.indxchun]
    
    #    # plot PDCSAP and SAP light curves
    #    figr, axis = plt.subplots(2, 1, figsize=gdat.figrsizeydob)
    #    axis[0].plot(gdat.arrytsersapp[:, 0] - gdat.timeoffs, gdat.arrytsersapp[:, 1], color='k', marker='.', ls='', ms=1, rasterized=True)
    #    axis[1].plot(gdat.arrytserpdcc[:, 0] - gdat.timeoffs, gdat.arrytserpdcc[:, 1], color='k', marker='.', ls='', ms=1, rasterized=True)
    #    #axis[0].text(.97, .97, 'SAP', transform=axis[0].transAxes, size=20, color='r', ha='right', va='top')
    #    #axis[1].text(.97, .97, 'PDC', transform=axis[1].transAxes, size=20, color='r', ha='right', va='top')
    #    axis[1].set_xlabel('Time [BJD - %d]' % gdat.timeoffs)
    #    for a in range(2):
    #        axis[a].set_ylabel(gdat.labltserphot)
    #    plt.subplots_adjust(hspace=0.)
    #    path = gdat.pathvisutarg + 'lcurspoc_%s.%s' % (gdat.strgtarg, gdat.typefileplot)
    #    gdat.listdictdvrp[j+1].append({'path': path, 'limt':[0.4, 0.05, 0.8, 0.8]})
    #    if gdat.typeverb > 0:
    #        print('Writing to %s...' % path)
    #    plt.savefig(path)
    #    plt.close()
        
    # calculate the visibility of the target
    if gdat.boolcalcvisi:
        
        if gdat.listdelttimeobvtyear is None:
            gdat.listdelttimeobvtyear = np.linspace(0., 365., 10000)

        massairr = tdpy.calc_visitarg(gdat.rasctarg, gdat.decltarg, gdat.latiobvt, gdat.longobvt, gdat.strgtimeobvtyear, gdat.listdelttimeobvtyear, gdat.heigobvt)

        gdat.dictmileoutp['massairr'] = massairr

        # alt-az coordinate object for the Sun
        #objtcoorsunnalazyear = astropy.coordinates.get_sun(objttimeyear)
        #objtcoorsunnalazyear = objtcoorsunnalazyear.transform_to(objtframobvtyear)
            
        # quantities during a given night
        if gdat.strgtimeobvtnigh is not None:
            objttimenigh = astropy.time.Time(astropy.time.Time(gdat.strgtimeobvtnigh).jd, format='jd', location=objtlocaobvt)
            objttimenighcent = astropy.time.Time(int(objttimenigh.jd), format='jd', location=objtlocaobvt)
            objttimenighcen1 = astropy.time.Time(int(objttimenigh.jd + 1), format='jd', location=objtlocaobvt)
            objttimenigh = objttimenighcent + (12. + timedelt - gdat.offstimeobvt) * astropy.units.hour
        
            # frame object for the observatory during the selected night
            objtframobvtnigh = astropy.coordinates.AltAz(obstime=objttimenigh, location=objtlocaobvt)
        
            # alt-az coordinate object for the Sun
            objtcoorsunnalaznigh = astropy.coordinates.get_sun(objttimenigh).transform_to(objtframobvtnigh)
            # alt-az coordinate object for the Moon
            objtcoormoonalaznigh = astropy.coordinates.get_moon(objttimenigh).transform_to(objtframobvtnigh)
            # alt-az coordinate object for the target
            objtcoorplanalaznigh = astropy.coordinates.SkyCoord(ra=gdat.rasctarg, dec=gdat.decltarg, frame='icrs', unit='deg').transform_to(objtframobvtnigh)
        
            # air mass of the target during the night
            massairr = objtcoorplanalaznigh.secz
        
            for j in gmod.indxcomp:
                indx = retr_indxtimetran(timeyear, gdat.fitt.prio.meanpara.epocmtracomp[j], gdat.fitt.prio.meanpara.pericomp[j], gdat.fitt.prio.meanpara.duratrantotlcomp[j])
                
                import operator
                import itertools
                for k, g in itertools.groupby(enumerate(list(indx)), lambda ix : ix[0] - ix[1]):
                    print(map(operator.itemgetter(1), g))
            
            labltime = 'Local time to Midnight [hour]'
            print('%s, Air mass' % labltime)
            for ll in range(len(massairr)):
                print('%6g %6.3g' % (timedelt[ll], massairr[ll]))

            
    # plot visibility of the target
    if gdat.boolplotvisi:
        strgtitl = '%s, %s/%s' % (gdat.labltarg, objttimenighcent.iso[:10], objttimenighcen1.iso[:10])

        # plot air mass
        figr, axis = plt.subplots(figsize=(8, 4))
        
        indx = np.where(np.isfinite(massairr) & (massairr > 0))[0]
        plt.plot(timedelt[indx], massairr[indx])
        axis.fill_between(timedelt, 0, 90, objtcoorsunnalaznigh.alt < -0*astropy.units.deg, color='0.5', zorder=0)
        axis.fill_between(timedelt, 0, 90, objtcoorsunnalaznigh.alt < -18*astropy.units.deg, color='k', zorder=0)
        axis.fill_between(timedelt, 0, 90, (massairr > 2.) | (massairr < 1.), color='r', alpha=0.3, zorder=0)
        axis.set_xlabel(labltime)
        axis.set_ylabel('Airmass')
        limtxdat = [np.amin(timedelt), np.amax(timedelt)]
        axis.set_title(strgtitl)
        axis.set_xlim(limtxdat)
        axis.set_ylim([1., 2.])
        path = gdat.pathvisutarg + 'airmass_%s.%s' % (gdat.strgtarg, gdat.typefileplot)
        print('Writing to %s...' % path)
        plt.savefig(path)
        
        # plot altitude
        figr, axis = plt.subplots(figsize=(8, 4))
        axis.plot(timedelt, objtcoorsunnalaznigh.alt, color='orange', label='Sun')
        axis.plot(timedelt, objtcoormoonalaznigh.alt, color='gray', label='Moon')
        axis.plot(timedelt, objtcoorplanalaznigh.alt, color='blue', label=gdat.labltarg)
        axis.fill_between(timedelt, 0, 90, objtcoorsunnalaznigh.alt < -0*astropy.units.deg, color='0.5', zorder=0)
        axis.fill_between(timedelt, 0, 90, objtcoorsunnalaznigh.alt < -18*astropy.units.deg, color='k', zorder=0)
        axis.fill_between(timedelt, 0, 90, (massairr > 2.) | (massairr < 1.), color='r', alpha=0.3, zorder=0)
        axis.legend(loc='upper left')
        plt.ylim([0, 90])
        axis.set_title(strgtitl)
        axis.set_xlim(limtxdat)
        axis.set_xlabel(labltime)
        axis.set_ylabel('Altitude [deg]')
        
        path = gdat.pathvisutarg + 'altitude_%s.%s' % (gdat.strgtarg, gdat.typefileplot)
        print('Writing to %s...' % path)
        plt.savefig(path)

    ### bin the light curve
    #gdat.delttimebind = 1. # [days]
    #for b in gdat.indxdatatser:
    #    for p in gdat.indxinst[b]:
    #        gdat.arrytser['bdtrbind'][b][p] = rebn_tser(gdat.arrytser['Detrended'][b][p], delt=gdat.delttimebind)
    #        for y in gdat.indxchun[b][p]:
    #            gdat.listarrytser['bdtrbind'][b][p][y] = rebn_tser(gdat.listarrytser['Detrended'][b][p][y], delt=gdat.delttimebind)
    #            
    #            path = gdat.pathdatatarg + '%s_DataCube_Detrended_Binned_%s%s.csv' % (gdat.liststrgdatatser[b][p], gdat.liststrginst[b][p], gdat.liststrgchun[b][p][y])
    #            if not os.path.exists(path):
    #                if gdat.typeverb > 0:
    #                    print('Writing to %s' % path)
    #                np.savetxt(path, gdat.listarrytser['bdtrbind'][b][p][y], delimiter=',', header=gdat.strgheadtser)
    #        
    #            if gdat.boolplottser:
    #                plot_tser_mile(gdat, b, p, y, 'bdtrbind')
            
    gdat.dictmileoutp['boolposianls'] = np.empty(gdat.numbtypeposi, dtype=bool)
    if gdat.boolsrchboxsperi:
        gdat.dictmileoutp['boolposianls'][0] = gdat.dictboxsperioutp['s2nr'][0] > gdat.thrss2nrcosc
    if gdat.boolcalclspe:
        gdat.dictmileoutp['boolposianls'][1] = gdat.dictmileoutp['powrlspempow'] > gdat.thrslspecosc
    gdat.dictmileoutp['boolposianls'][2] = gdat.dictmileoutp['boolposianls'][0] or gdat.dictmileoutp['boolposianls'][1]
    gdat.dictmileoutp['boolposianls'][3] = gdat.dictmileoutp['boolposianls'][0] and gdat.dictmileoutp['boolposianls'][1]
    
    for strgmodl in gdat.liststrgmodl:
        gmod = getattr(gdat, strgmodl)

        if gdat.boolmodl and gmod.boolmodlpcur:
            ### Doppler beaming
            if gdat.typeverb > 0:
                print('Assuming TESS passband for estimating Dopller beaming...')
            gdat.binswlenbeam = np.linspace(0.6, 1., 101)
            gdat.cntrwlenbeam = (gdat.binswlenbeam[1:] + gdat.binswlenbeam[:-1]) / 2.
            gdat.diffwlenbeam = (gdat.binswlenbeam[1:] - gdat.binswlenbeam[:-1]) / 2.
            x = 2.248 / gdat.cntrwlenbeam
            gdat.funcpcurmodu = .25 * x * np.exp(x) / (np.exp(x) - 1.)
            gdat.consbeam = np.sum(gdat.diffwlenbeam * gdat.funcpcurmodu)

            #if ''.join(gdat.liststrgcomp) != ''.join(sorted(gdat.liststrgcomp)):
            #if gdat.typeverb > 0:
            #       print('Provided planet letters are not in order. Changing the TCE order to respect the letter order in plots (b, c, d, e)...')
            #    gmod.indxcomp = np.argsort(np.array(gdat.liststrgcomp))

    gdat.liststrgcompfull = np.empty(gdat.fitt.prio.numbcomp, dtype='object')
    if gdat.fitt.prio.numbcomp is not None:
        for j in gdat.fitt.prio.indxcomp:
            gdat.liststrgcompfull[j] = gdat.labltarg + ' ' + gdat.liststrgcomp[j]
        gdat.fitt.numbcomp = gdat.fitt.prio.numbcomp

    ## augment object dictinary
    gdat.dictfeatobjt = dict()
    if gdat.fitt.prio.numbcomp is not None:
        gdat.dictfeatobjt['namestar'] = np.array([gdat.labltarg] * gdat.fitt.prio.numbcomp)
        gdat.dictfeatobjt['nameplan'] = gdat.liststrgcompfull
        # temp
        gdat.dictfeatobjt['booltran'] = np.array([True] * gdat.fitt.prio.numbcomp, dtype=bool)
    for namemagt in ['vmag', 'jmag', 'hmag', 'kmag']:
        magt = getattr(gdat, '%ssyst' % namemagt)
        if magt is not None:
            gdat.dictfeatobjt['%ssyst' % namemagt] = np.zeros(gdat.fitt.prio.numbcomp) + magt
    if gdat.fitt.prio.numbcomp is not None:
        gdat.dictfeatobjt['numbplanstar'] = np.zeros(gdat.fitt.prio.numbcomp) + gdat.fitt.prio.numbcomp
        gdat.dictfeatobjt['numbplantranstar'] = np.zeros(gdat.fitt.prio.numbcomp) + gdat.fitt.prio.numbcomp
    
    if gdat.dilu == 'lygos':
        if gdat.typeverb > 0:
            print('Calculating the contamination ratio...')
        gdat.contrati = lygos.retr_contrati()

    # correct for dilution
    #if gdat.typeverb > 0:
    #print('Correcting for dilution!')
    #if gdat.dilucorr is not None:
    #    gdat.arrytserdilu = np.copy(gdat.listarrytser['Detrended'][b][p][y])
    #if gdat.dilucorr is not None:
    #    gdat.arrytserdilu[:, 1] = 1. - gdat.dilucorr * (1. - gdat.listarrytser['Detrended'][b][p][y][:, 1])
    #gdat.arrytserdilu[:, 1] = 1. - gdat.contrati * gdat.contrati * (1. - gdat.listarrytser['Detrended'][b][p][y][:, 1])
    
    if not ((gdat.boolsrchboxsperi or gdat.boolsrchoutlperi) and not gdat.dictmileoutp['boolposianls'].any()):
        
        ## number of bins in the phase curve
        gdat.numbbinspcurfull = 100
        
        gdat.liststrgpcur = ['Detrended', 'resi', 'modl']
        gdat.liststrgpcurcomp = ['modltotl', 'modlstel', 'modlplan', 'modlelli', 'modlpmod', 'modlnigh', 'modlbeam', 'bdtrplan']
        gdat.binsphasprimtotl = np.linspace(-0.5, 0.5, gdat.numbbinspcurfull + 1)
        gdat.binsphasquadtotl = np.linspace(-0.25, 0.75, gdat.numbbinspcurfull + 1)
        
        gdat.liststrgarrypcur = [ \
                            'DetrendedPrimaryCentered', \
                            'DetrendedPrimaryCenteredZoom', \
                            'DetrendedSecondaryCenteredZoom', \
                            'DetrendedQuadratureCentered', \
                           ]
        if gdat.typepriocomp != 'outlperi':
            gdat.liststrgarrypcur += ['DetrendedQuadratureCenteredMasked']
        
        if gdat.typeverb > 0:
            print('Phase folding and binning the light curve...')
        
        #for strgmodl in gdat.liststrgmodl:
        for strgmodl in ['fitt']:
            
            if strgmodl == 'true':
                gmod = gdat.true
                objtpara = gdat.true
            else:
                gmod = gdat.fitt.prio
                objtpara = gdat.fitt.prio.meanpara
            
            gmod.arrypcur = dict()
            
            # number of samples inside total transit
            gmod.numbsamptimetran = 50
            
            # time differential in binned and zoomed phase curve
            gmod.delttimebindzoom = objtpara.duratrantotlcomp / 24. / gmod.numbsamptimetran
        
            gmod.numbbinspcurzoom = (objtpara.pericomp / gmod.delttimebindzoom).astype(int)
        
            if gdat.booldiag:
                if (gmod.numbbinspcurzoom <= 0).any() or (gmod.delttimebindzoom == 0).any():
                    print('')
                    print('')
                    print('')
                    print('gmod.numbbinspcurzoom')
                    print(gmod.numbbinspcurzoom)
                    raise Exception('Bad gmod.numbbinspcurzoom.')
            
            objtpara.dcyctrantotlcomp = objtpara.duratrantotlcomp / objtpara.pericomp / 24.

            gdat.dictbinsphas = dict()
            for strgarrypcur in gdat.liststrgarrypcur:
                
                numbbins = 100
                if strgarrypcur == 'DetrendedPrimaryCentered':
                    limt = [-0.5, 0.5]
                if strgarrypcur == 'DetrendedPrimaryCenteredZoom':
                    limt = [-0.5 * objtpara.dcyctrantotlcomp, 0.5 * objtpara.dcyctrantotlcomp]
                if strgarrypcur == 'DetrendedSecondaryCenteredZoom':
                    limt = [0.5 - 0.5 * objtpara.dcyctrantotlcomp, 0.5 + 0.5 * objtpara.dcyctrantotlcomp]
                if strgarrypcur == 'DetrendedQuadratureCentered':
                    limt = [-0.25, 0.75]
                if strgarrypcur == 'DetrendedQuadratureCenteredMasked':
                    limt = [-0.25, 0.75]
                gdat.dictbinsphas[strgarrypcur], _, _, _, _ = tdpy.retr_axis(limt=limt, numbpntsgrid=numbbins)
                gmod.arrypcur[strgarrypcur] = [[[[] for j in gdat.fitt.prio.indxcomp] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
                gmod.arrypcur[strgarrypcur+'Binned'] = [[[[] for j in gdat.fitt.prio.indxcomp] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        
            for b in gdat.indxdatatser:
                for p in gdat.indxinst[b]:
                    for j in gmod.indxcomp:
                        
                        gmod.arrypcur['DetrendedPrimaryCentered'][b][p][j] = fold_tser(gdat.arrytser['Detrended'][b][p], \
                                                                                                     objtpara.epocmtracomp[j], objtpara.pericomp[j], phascntr=0.)
                        
                        if gmod.arrypcur['DetrendedPrimaryCentered'][b][p][j].ndim > 3:
                            print('')
                            print('')
                            print('')
                            raise Exception('arrypcur[DetrendedPrimaryCentered][b][p][j].ndim > 3')
                        
                        indx = np.where(abs(gmod.arrypcur['DetrendedPrimaryCentered'][b][p][j][:, 0, 0] - 0.5) < objtpara.dcyctrantotlcomp[j] / 2.)[0]
                        gmod.arrypcur['DetrendedPrimaryCenteredZoom'][b][p][j] = gmod.arrypcur['DetrendedPrimaryCentered'][b][p][j][indx, :, :]

                        gmod.arrypcur['DetrendedSecondaryCenteredZoom'][b][p][j] = fold_tser(gdat.arrytser['Detrended'][b][p], \
                                                                                                                objtpara.epocmtracomp[j], objtpara.pericomp[j], phascntr=0.5)
                        
                        gmod.arrypcur['DetrendedQuadratureCentered'][b][p][j] = fold_tser(gdat.arrytser['Detrended'][b][p], \
                                                                                      objtpara.epocmtracomp[j], objtpara.pericomp[j], phascntr=0.25)
                        
                        if gdat.typepriocomp != 'outlperi':
                            gmod.arrypcur['DetrendedQuadratureCenteredMasked'][b][p][j] = fold_tser(gdat.arrytser['Detrended'][b][p][gdat.listindxtimeoutt[j][b][p], :, :], \
                                                                                      objtpara.epocmtracomp[j], objtpara.pericomp[j], phascntr=0.25)
                        
                        for strgarrypcur in gdat.liststrgarrypcur:
                            
                            if len(gmod.arrypcur[strgarrypcur][b][p][j]) == 0:
                                print('Skipping binning %s because no data point exists...' % strgarrypcur)
                                continue

                            gmod.arrypcur[strgarrypcur + 'Binned'][b][p][j] = rebn_tser(gmod.arrypcur[strgarrypcur][b][p][j], blimxdat=gdat.dictbinsphas[strgarrypcur])
                        
                        for e in gdat.indxener[p]:
                            path = gdat.pathdatatarg + 'arrypcur_Primary_Detrended_Binned_%s%s%s_%s.csv' % (gdat.liststrginst[b][p], \
                                                                                    gdat.strgextncade[b][p], gdat.liststrgener[p][e], gdat.liststrgcomp[j])
                            if not os.path.exists(path):
                                temp = np.copy(gmod.arrypcur['DetrendedPrimaryCenteredBinned'][b][p][j][:, e, :])
                                temp[:, 0] *= objtpara.pericomp[j]
                                if gdat.typeverb > 0:
                                    print('Writing to %s...' % path)
                                np.savetxt(path, temp, delimiter=',', header=gdat.strgheadpser[b])
                    
            if gdat.boolplot:
                for strg in gdat.liststrgarrypcur:
                    plot_pser_mile(gdat, strgmodl, strg)
    
    gdat.numbsamp = 10

    if gdat.boolplotpopl:
        if gdat.typeverb > 0:
            print('Making plots highlighting the %s features of the target within its population...' % (strgpdfn))
        plot_popl(gdat, 'nomi')
    
    if gdat.boolsrchboxsperi and not gdat.dictmileoutp['boolposianls'].any():
        print('BLS was performed, but no super-threshold BLS signal was found.')
    
    gdat.booldeteposi = gdat.boolsrchboxsperi and not gdat.dictmileoutp['boolposianls'].any()

    # do not continue if there is no trigger
    # Boolean flag to continue modeling the data based on the feature extraction
    gdat.booltrig = gdat.boolmodl and gdat.booldeteposi
    
    if gdat.boolmodl:
        gdat.liststrgpdfn += ['post']

    if gdat.typeverb > 0:
        print('gdat.liststrgpdfn')
        print(gdat.liststrgpdfn)
    
    if not gdat.boolmodl:
        print('Skipping the forward modeling of this prior transiting object...')

    if gdat.boolfitt:
        
        # typemodlttvr
        # type of pipeline to fit transit times
        ## 'indilineuser': one fit for each transit, floating individual transits while fixing the orbital parameters including the
        ##                                                                              linear ephemerides (period and epoch) to user-defined values
        ## 'globlineuser': single fit across all transits with free transit times, but linear ephemerides from the user
        ## 'globlineflot': single fit across all transits with free transit times and linear ephemerides
        for strgmodl in gdat.liststrgmodl:
            gmod = getattr(gdat, strgmodl)
            if gmod.typemodl == 'PlanetarySystemWithTTVs':
                tdpy.setp_para_defa(gdat, strgmodl, 'typemodlttvr', 'globlineflot')

        gdat.boolbrekmodl = False

        gdat.timethisfitt = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.rflxthisfitt = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.stdvrflxthisfitt = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]  
        gdat.varirflxthisfitt = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]  
        gdat.timethisfittfine = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]  
        
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                if gdat.limttimefitt is None:
                    indxtimefitt = np.arange(gdat.listarrytser['Detrended'][b][p][y].shape[0])
                else:
                    indxtimefitt = np.where((gdat.listarrytser['Detrended'][b][p][y][:, 0, 0] < gdat.limttimefitt[b][p][1]) & (gdat.timethis > gdat.limttimefitt[b][p][0]))[0]
                gdat.timethisfitt[b][p] = gdat.listarrytser['Detrended'][b][p][y][indxtimefitt, 0, 0]
                gdat.rflxthisfitt[b][p] = gdat.listarrytser['Detrended'][b][p][y][indxtimefitt, :, 1]
                gdat.stdvrflxthisfitt[b][p] = gdat.listarrytser['Detrended'][b][p][y][indxtimefitt, :, 2]
                
                # temp
                if np.amax(gdat.stdvrflxthisfitt[b][p]) > 10.:
                    print('gdat.timethisfitt[b][p]')
                    summgene(gdat.timethisfitt[b][p])
                    print('gdat.rflxthisfitt[b][p]')
                    summgene(gdat.rflxthisfitt[b][p])
                    print('gdat.stdvrflxthisfitt[b][p]')
                    summgene(gdat.stdvrflxthisfitt[b][p])
                    raise Exception('')

                gdat.varirflxthisfitt[b][p] = gdat.stdvrflxthisfitt[b][p]**2
                
                minmtimethisfitt = np.amin(gdat.timethisfitt[b][p])
                maxmtimethisfitt = np.amax(gdat.timethisfitt[b][p])
                difftimethisfittfine = 0.3 * np.amin(gdat.timethisfitt[b][p][1:] - gdat.timethisfitt[b][p][:-1])
                gdat.timethisfittfine[b][p] = np.arange(minmtimethisfitt, maxmtimethisfitt + difftimethisfittfine, difftimethisfittfine)
            
        gmod = gdat.fitt
        
        meangauspara = None
        stdvgauspara = None
        
        gdat.numbsampwalk = 100
        gdat.numbsampburnwalkinit = 0
        gdat.numbsampburnwalk = int(0.3 * gdat.numbsampwalk)
        
        #for b in gdat.indxdatatser:
        #    for p in gdat.indxinst[b]:
                
        if gdat.typeverb > 0:
            if gdat.fitt.typemodl == 'PlanetarySystem' or gdat.fitt.typemodl == 'CompactObjectStellarCompanion' or gdat.fitt.typemodl == 'PlanetarySystemEmittingCompanion':
                print('gdat.dictmileoutp[boolposianls]')
                print(gdat.dictmileoutp['boolposianls'])
            
        if gdat.typeverb > 0:
            print('gmod.typemodlblinshap')
            print(gmod.typemodlblinshap)
            print('gmod.typemodlblinener')
            print(gmod.typemodlblinener)
        
        # iterate over different subsets of fits
        ## these can be over different subsets of data
        for h in gdat.indxfittiter:
        
            if gdat.typeinfe == 'opti':
                path = gdat.pathdatatarg + 'paramlik.csv'
                
                # temp
                if os.path.exists(path) and False:
                    print('Reading from %s...' % path)
                    objtfile = open(path, 'r')
                    gdat.liststrgdatafittitermlikdone = []
                    gdat.datamlik = []
                    for line in objtfile:
                        linesplt = line.split(',')
                        gdat.liststrgdatafittitermlikdone.append(linesplt[0]) 
                        gdat.datamlik.append(np.array(linesplt[1:]).astype(float))
                    objtfile.close()
                    gdat.liststrgdatafittitermlikdone = np.array(gdat.liststrgdatafittitermlikdone)
                else:
                    gdat.liststrgdatafittitermlikdone = np.array([])

            # restrict to white light curve
            # temp
            if e == 0 and False:
                listindxinstthis = [0]
            else:
                listindxinstthis = gdat.listindxinst
            
            if gdat.fitt.typemodl == 'StarFlaring':

                init_modl(gdat, 'fitt')

                setp_modlbase(gdat, 'fitt', h)
        
                proc_modl(gdat, 'fitt', strgextn, h)


            elif gdat.fitt.typemodl == 'AGN':

                init_modl(gdat, 'fitt')

                setp_modlbase(gdat, 'fitt', h)
        
                proc_modl(gdat, 'fitt', strgextn, h)


            elif gdat.fitt.typemodl == 'SpottedStar':

                # for each spot multiplicity, fit the spot model
                for gdat.numbspot in listindxnumbspot:
                    
                    init_modl(gdat, 'fitt')

                    setp_modlbase(gdat, 'fitt', h)
        
                    if gdat.typeverb > 0:
                        print('gdat.numbspot')
                        print(gdat.numbspot)

                    # list of parameter labels and units
                    gmod.listlablpara = [['$u_1$', ''], ['$u_2$', ''], ['$P$', 'days'], ['$i$', 'deg'], ['$\\rho$', ''], ['$C$', '']]
                    # list of parameter scalings
                    listscalpara = ['self', 'self', 'self', 'self', 'self', 'self']
                    # list of parameter minima
                    gmod.listminmpara = [-1., -1., 0.2,   0.,  0.,-1e-1]
                    # list of parameter maxima
                    gmod.listmaxmpara = [ 3.,  3., 0.4, 89.9, 0.6, 1e-1]
                    
                    for numbspottemp in range(gdat.numbspot):
                        gmod.listlablpara += [['$\\theta_{%d}$' % numbspottemp, 'deg'], \
                                                    ['$\\phi_{%d}$' % numbspottemp, 'deg'], ['$R_{%d}$' % numbspottemp, '']]
                        listscalpara += ['self', 'self', 'self']
                        gmod.listminmpara += [-90.,   0.,  0.]
                        gmod.listmaxmpara += [ 90., 360., 0.4]
                        if gdat.boolevol:
                            gmod.listlablpara += [['$T_{s;%d}$' % numbspottemp, 'day'], ['$\\sigma_{s;%d}$' % numbspottemp, '']]
                            listscalpara += ['self', 'self']
                            gmod.listminmpara += [gdat.minmtime, 0.1]
                            gmod.listmaxmpara += [gdat.maxmtime, 20.]
                            
                    # plot light curve
                    figr, axis = plt.subplots(figsize=(8, 4))
                    # plot samples from the posterior
                    ## the sample indices which will be plotted
                    indxsampplot = np.random.choice(gdat.indxsamp, size=gdat.numbsampplot, replace=False)
                    indxsampplot = np.sort(indxsampplot)
                    listlcurmodl = np.empty((gdat.numbsampplot, gdat.numbtime))
                    listlcurmodlevol = np.empty((gdat.numbsampplot, gdat.numbspot, gdat.numbtime))
                    listlcurmodlspot = np.empty((gdat.numbsampplot, gdat.numbspot, gdat.numbtime))
                    for kk, k in enumerate(indxsampplot):
                        # calculate the model light curve for this parameter vector
                        listlcurmodl[kk, :], listlcurmodlevol[kk, :, :], listlcurmodlspot[kk, :, :] = ephesos.eval_modl(gdat, listpost[k, :])
                        axis.plot(gdat.time, listlcurmodl[kk, :], color='b', alpha=0.1)
                    
                    # plot components of each sample
                    for kk, k in enumerate(indxsampplot):
                        dictpara = pars_para(gdat, listpost[k, :])
                        plot_totl(gdat, k, listlcurmodl[kk, :], listlcurmodlevol[kk, :, :], listlcurmodlspot[kk, :, :], dictpara)

                    # plot map
                    figr, axis = plt.subplots(figsize=(8, 4))
                    gdat.numbside = 2**10
                    
                    lati = np.empty((gdat.numbsamp, gdat.numbspot))
                    lngi = np.empty((gdat.numbsamp, gdat.numbspot))
                    rrat = np.empty((gdat.numbsamp, gdat.numbspot))
                    for n in gdat.indxsamp:
                        dictpara = pars_para(gdat, listpost[n, :])
                        lati[n, :] = dictpara['lati']
                        lngi[n, :] = dictpara['lngi']
                        rrat[n, :] = dictpara['rratcomp']
                    lati = np.median(lati, 0)
                    lngi = np.median(lngi, 0)
                    rrat = np.median(rrat, 0)

                    plot_moll(gdat, lati, lngi, rrat)
                    
                    #for k in indxsampplot:
                    #    lati = listpost[k, 1+0*gdat.numbparaspot+0]
                    #    lngi = listpost[k, 1+0*gdat.numbparaspot+1]
                    #    rrat = listpost[k, 1+0*gdat.numbparaspot+2]
                    #    plot_moll(gdat, lati, lngi, rrat)

                    for sp in ['right', 'top']:
                        axis.spines[sp].set_visible(False)

                    path = gdat.pathvisutarg + 'smap%s_ns%02d.%s' % (strgtarg, gdat.numbspot, gdat.typefileplot)
                    gdat.listdictdvrp[j+1].append({'path': path, 'limt':[0., 0.05, 1., 0.1]})
                    if gdat.typeverb > 0:
                        print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()


            elif gdat.fitt.typemodl == 'PlanetarySystem' or gdat.fitt.typemodl == 'CompactObjectStellarCompanion' or gdat.fitt.typemodl == 'PlanetarySystemEmittingCompanion' or gdat.fitt.typemodl == 'PlanetarySystemWithTTVs':
                
                if gdat.fitt.typemodl == 'PlanetarySystemWithTTVs':
                    if gdat.fitt.typemodlttvr == 'indilineuser':
                        gdat.numbiterfitt = gdat.numbtran
                    elif gdat.fitt.typemodlttvr == 'globlineuser':
                        gdat.numbiterfitt = 1
                    elif gdat.fitt.typemodlttvr == 'globlineflot':
                        gdat.numbiterfitt = 1
                else:
                    gdat.numbiterfitt = 1
                
                gdat.indxiterfitt = np.arange(gdat.numbiterfitt)
                
                gdat.fitt.listindxinstener = [[] for p in gdat.indxinst[0]]

                gdat.indxfittiterthis = 0
                for ll in gdat.indxiterfitt:
                    
                    for p in gdat.indxinst[0]:
                        if gdat.fitt.typemodlenerfitt == 'full':
                            gdat.fitt.listindxinstener[p] = gdat.indxener[p]
                        else:
                            gdat.fitt.listindxinstener[p] = gdat.indxener[p][ll-1, None]
                    
                    init_modl(gdat, 'fitt')

                    setp_modlbase(gdat, 'fitt', h)
                    
                    strgextn = gdat.strgcnfg + gdat.fitt.typemodl
                    if gdat.fitt.typemodlenerfitt == 'iter':
                        strgextn += gdat.liststrgdatafittiter[gdat.indxfittiterthis]
                    proc_modl(gdat, 'fitt', strgextn, h)

            elif gdat.fitt.typemodl == 'stargpro':
                
                init_modl(gdat, 'fitt')

                setp_modlbase(gdat, 'fitt', h)
        
                pass
            else:
                print('')
                print('A model type was not defined.')
                print('gdat.fitt.typemodl')
                print(gdat.fitt.typemodl)
                raise Exception('')
        
        if gdat.typeinfe == 'samp':
            gdat.indxsampmpos = np.argmax(gdat.dictsamp['lpos'])
            
            gdat.indxsampplot = np.random.choice(gdat.indxsamp, gdat.numbsampplot, replace=False)

            if gdat.typeverb > 0:
                print('gdat.numbsamp')
                print(gdat.numbsamp)
                print('gdat.numbsampplot')
                print(gdat.numbsampplot)
        
        if gdat.numbener[p] > 1 and (gdat.fitt.typemodl == 'PlanetarySystem' or gdat.fitt.typemodl == 'CompactObjectStellarCompanion' \
                                                                    or gdat.fitt.typemodl == 'PlanetarySystemEmittingCompanion'):
            # plot the radius ratio spectrum
            path = gdat.pathvisutarg + 'spec%s.%s' % (gdat.strgcnfg, gdat.typefileplot)
            figr, axis = plt.subplots(figsize=gdat.figrsizeydob)
            pmedrratcompspec = np.empty(gdat.numbener[p])
            perrrratcompspec = np.empty(gdat.numbener[p])
            for e in gdat.indxener[p]:
                if gdat.typeinfe == 'samp':
                    if gdat.fitt.typemodlenerfitt == 'full':
                        listrratcomp = gdat.dictsamp['rratcomp' + gdat.liststrgener[p][e]]
                    else:
                        listrratcomp = gmod.listdictsamp[e+1]['rratcomp' + gdat.liststrgener[p][e]]
                    pmedrratcompspec[e] = np.median(listrratcomp)
                    perrrratcompspec[e] = (np.percentile(listrratcomp, 86.) - np.percentile(listrratcomp, 14.)) / 2.
                else:
                    if gdat.fitt.typemodlenerfitt == 'full':
                        pmedrratcompspec[e] = gdat.dictmlik['rratcomp' + gdat.liststrgener[p][e]]
                    else:
                        pmedrratcompspec[e] = gmod.listdictmlik[e+1]['rratcomp' + gdat.liststrgener[p][e]]
                        perrrratcompspec[e] = gmod.listdictmlik[e+1]['stdvrratcomp' + gdat.liststrgener[p][e]]
            axis.plot(gdat.listener[p], pmedrratcompspec, ls='', marker='o')
            # plot binned spectrum
            #    arry = np.zeros((dictvarbderi['rflxresi'][:, e].size, 3))
            #    arry[:, 0] = gdat.timethisfitt
            #    arry[:, 1] = dictvarbderi['rflxresi'][:, e]
            #    stdvrflxresi = np.nanstd(rebn_tser(arry, delt=gdat.listdeltrebn[b][p])[:, 1])
            axis.plot(gdat.listener[p], pmedrratcompspec, ls='', marker='o')
            axis.set_ylabel('$R_p/R_*$')
            axis.set_xlabel('Wavelength [$\mu$m]')
            plt.tight_layout()
            if gdat.typeverb > 0:
                print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
            
            # load the spectrum to the output dictionary
            gdat.dictmileoutp['pmedrratcompspec'] = pmedrratcompspec
            gdat.dictmileoutp['perrrratcompspec'] = perrrratcompspec

            #path = gdat.pathvisutarg + 'stdvrebnener%s.%s' % (gdat.strgcnfg, gdat.typefileplot)
            #if not os.path.exists(path):
            #    figr, axis = plt.subplots(figsize=gdat.figrsizeydob)
            #    arry = np.zeros((dictvarbderi['rflxresi'][:, e].size, 3))
            #    arry[:, 0] = gdat.timethisfitt
            #    arry[:, 1] = dictvarbderi['rflxresi'][:, e]
            #    for k in gdat.indxrebn:
            #    stdvrflxresi = np.nanstd(rebn_tser(arry, delt=gdat.listdeltrebn[b][p])[:, 1])
            #    axis.loglog(gdat.listdeltrebn[b][p], stdvrflxresi * 1e6, ls='', marker='o', ms=1, label='Binned Std. Dev')
            #    axis.axvline(gdat.ratesampener, ls='--', label='Sampling rate')
            #    axis.axvline(gdat.enerscalbdtr, ls='--', label='Detrending scale')
            #    axis.set_ylabel('RMS [ppm]')
            #    axis.set_xlabel('Bin width [$\mu$m]')
            #    axis.legend()
            #    plt.tight_layout()
            #    if gdat.typeverb > 0:
            #        print('Writing to %s...' % path)
            #    plt.savefig(path)
            #    plt.close()

    if gdat.boolplot and gdat.boolplotdvrp:
        listpathdvrp = []
        # make data-validation report
        print('gdat.numbpage')
        print(gdat.numbpage)
        print('gdat.indxpage')
        print(gdat.indxpage)
        print('gdat.listdictdvrp')
        for w in gdat.indxpage:
            print('Page %d' % w)
            for temp in gdat.listdictdvrp[w]:
                print(temp)
            
        for w in gdat.indxpage:
            # path of DV report
            pathplot = gdat.pathvisutarg + 'Summary_Page%d_%s.png' % (w + 1, gdat.strgtarg)
            listpathdvrp.append(pathplot)
            
            if not os.path.exists(pathplot):
                # create page with A4 size
                figr = plt.figure(figsize=(8.25, 11.75))
                
                numbplot = len(gdat.listdictdvrp[w])
                indxplot = np.arange(numbplot)
                for dictdvrp in gdat.listdictdvrp[w]:
                    axis = figr.add_axes(dictdvrp['limt'])
                    print('Reading from %s...' % dictdvrp['path'])
                    axis.imshow(plt.imread(dictdvrp['path']))
                    axis.axis('off')
                if gdat.typeverb > 0:
                    print('Writing to %s...' % pathplot)
                plt.savefig(pathplot, dpi=600)
                #plt.subplots_adjust(top=1., bottom=0, left=0, right=1)
                plt.close()
        
        gdat.dictmileoutp['listpathdvrp'] = listpathdvrp

    # write the output dictionary to target file
    path = gdat.pathdatatarg + 'miletos_output.csv'
    objtfile = open(path, 'w')
    k = 0
    for name, valu in gdat.dictmileoutp.items():
        if isinstance(valu, str) or isinstance(valu, float) or isinstance(valu, int) or isinstance(valu, bool):
            objtfile.write('%s, ' % name)
        if isinstance(valu, str):
            objtfile.write('%s' % valu)
        elif isinstance(valu, float) or isinstance(valu, int) or isinstance(valu, bool):
            objtfile.write('%g' % valu)
        if isinstance(valu, str) or isinstance(valu, float) or isinstance(valu, int) or isinstance(valu, bool):
            objtfile.write('\n')
    if typeverb > 0:
        print('Writing to %s...' % path)
    objtfile.close()
    
    # write the output dictionary to the cluster file
    if gdat.strgclus is not None:
        path = gdat.pathdataclus + 'miletos_cluster_output.csv'
        boolappe = True
        if os.path.exists(path):
            print('Reading from %s...' % path)
            dicttemp = pd.read_csv(path).to_dict(orient='list')
            if gdat.strgtarg in dicttemp['strgtarg']:
                boolappe = False
            boolmakehead = False
        else:
            print('Opening %s...' % path)
            objtfile = open(path, 'w')
            boolmakehead = True
        
        if boolmakehead:
            print('Will construct a header...')
        else:
            print('Will not construct a header...')
        
        if boolappe:
            
            print('gdat.dictmileoutp')
            for name in gdat.dictmileoutp:
                if 'path' in name:
                    print(name)

            if boolmakehead:
                print('Constructing the header...')
                # if the header doesn't exist, make it
                k = 0
                listnamecols = []
                for name, valu in gdat.dictmileoutp.items():
                    
                    if name.startswith('lygo_pathsaverflx'): 
                        continue
                    
                    if name.startswith('lygo_strgtitlcntpplot'):
                        continue
                    
                    listnamecols.append(name)
                    if isinstance(valu, str) or isinstance(valu, float) or isinstance(valu, int) or isinstance(valu, bool):
                        if k > 0:
                            objtfile.write(',')
                        objtfile.write('%s' % name)
                        k += 1
                
            else:
                print('Reading from %s...' % path)
                objtfile = open(path, 'r')
                for line in objtfile:
                    listnamecols = line.split(',')
                    break
                listnamecols[-1] = listnamecols[-1][:-1]

                if not gdat.strgtarg in dicttemp['strgtarg']:
                    print('Opening %s to append...' % path)
                    objtfile = open(path, 'a')
            
            objtfile.write('\n')
            k = 0
            
            print('listnamecols')
            for name in listnamecols:
                if 'path' in name:
                    print(name)
            
            print('gdat.dictmileoutp.keys()')
            print(sorted(list(gdat.dictmileoutp.keys())))
            for name in listnamecols:
                valu = gdat.dictmileoutp[name]
                if isinstance(valu, str) or isinstance(valu, float) or isinstance(valu, int) or isinstance(valu, bool):
                    if k > 0:
                        objtfile.write(',')
                    if isinstance(valu, str):
                        objtfile.write('%s' % valu)
                    elif isinstance(valu, float) or isinstance(valu, int) or isinstance(valu, bool):
                        objtfile.write('%g' % valu)
                    k += 1
            #objtfile.write('\n')
            if typeverb > 0:
                print('Writing to %s...' % path)
            objtfile.close()

    # measure final time
    gdat.timefinl = modutime.time()
    gdat.timeexec = gdat.timefinl - gdat.timeinit
    if gdat.typeverb > 0:
        print('miletos ran in %.3g seconds.' % gdat.timeexec)
        print('')
        print('')
        print('')
    
    #'lygo_meannois', 'lygo_medinois', 'lygo_stdvnois', \
    for name in ['strgtarg', 'pathtarg', 'timeexec']:
        gdat.dictmileoutp[name] = getattr(gdat, name)

    return gdat.dictmileoutp


