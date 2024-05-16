import sys
from tqdm import tqdm
import inspect
import os
import numpy as np
import wget
import pandas as pd

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

import astropy

import miletos
import tdpy
from tdpy.util import summgene
import ephesos
import nicomedia


def cnfg_TOI1233():
    
    path = os.environ['DATA'] + '/general/TOI1233/HD108236_PFS_20220627.vels'
    print('Reading from %s...' % path)
    listtime = []
    listrvel = []
    liststdvrvel = []
    for line in open(path, 'r'):
        listlinesplt = line.split(' ')
        k = 0
        for line in listlinesplt:
            if line == ' ':
                continue
            if line == '':
                continue
            
            valu = float(line)
            if k == 0:
                listtime.append(valu)
            if k == 1:
                listrvel.append(valu)
            if k == 2:
                liststdvrvel.append(valu)
            k += 1
    listtime = np.array(listtime)
    
    listarrytser = dict()
    listarrytser['raww'] = [None ,[[[]]]]
    listarrytser['raww'][1][0][0] = np.empty((listtime.size, 1, 3))
    listarrytser['raww'][1][0][0][:, 0, 0] = listtime
    listarrytser['raww'][1][0][0][:, 0, 1] = np.array(listrvel)
    listarrytser['raww'][1][0][0][:, 0, 2] = np.array(liststdvrvel)

    listtypemodl = ['PlanetarySystem', 'PlanetarySystemWithTTVs']
    
    for typemodl in listtypemodl:
        dictfitt = dict()
        dictfitt['typemodl'] = typemodl
        
        toiitarg = 1233
        miletos.main.init( \
                          toiitarg=toiitarg, \
                          dictfitt=dictfitt, \
                          strgexar='HD 108236', \
                          listarrytser=listarrytser, \
                          typepriocomp='exar', \
                          boolplotpopl=True, \
                         )


def cnfg_WASP121b():
    
    # add Vivien's result
    axis[k].plot(gdat.phasvivi, gdat.deptvivi*1e3, color='orange', lw=2, label='GCM (Parmentier+2018)')
    axis[k].axhline(0., ls='-.', alpha=0.3, color='grey')
    
    # read and parse ATMO posterior
    ## get secondary depth data from Tom
    path = gdat.pathdatatarg + 'ascii_output/EmissionDataArray.txt'
    print('Reading from %s...' % path)
    arrydata = np.loadtxt(path)
    print('arrydata')
    summgene(arrydata)
    print('arrydata[0, :]')
    print(arrydata[0, :])
    path = gdat.pathdatatarg + 'ascii_output/EmissionModelArray.txt'
    print('Reading from %s...' % path)
    arrymodl = np.loadtxt(path)
    print('arrymodl')
    summgene(arrymodl)
    print('Secondary eclipse depth mean and standard deviation:')
    # get wavelengths
    path = gdat.pathdatatarg + 'ascii_output/ContribFuncWav.txt'
    print('Reading from %s...' % path)
    wlen = np.loadtxt(path)
    path = gdat.pathdatatarg + 'ascii_output/ContribFuncWav.txt'
    print('Reading from %s...' % path)
    wlenctrb = np.loadtxt(path, skiprows=1)
   
    # get Vivien's GCM model
    path = gdat.pathdatatarg + 'PC-Solar-NEW-OPA-TiO-LR.dat'
    arryvivi = np.loadtxt(path, delimiter=',')
    gdat.phasvivi = (arryvivi[:, 0] / 360. + 0.75) % 1. - 0.25
    gdat.deptvivi = arryvivi[:, 4]
    indxphasvivisort = np.argsort(gdat.phasvivi)
    gdat.phasvivi = gdat.phasvivi[indxphasvivisort]
    gdat.deptvivi = gdat.deptvivi[indxphasvivisort]
    path = gdat.pathdatatarg + 'PC-Solar-NEW-OPA-TiO-LR-AllK.dat'
    arryvivi = np.loadtxt(path, delimiter=',')
    gdat.wlenvivi = arryvivi[:, 1]
    gdat.specvivi = arryvivi[:, 2]
    
    ## TESS throughput 
    gdat.data = np.loadtxt(gdat.pathdatatarg + 'band.csv', delimiter=',', skiprows=9)
    gdat.cntrwlenband = gdat.data[:, 0] * 1e-3
    gdat.thptband = gdat.data[:, 1]
    
    dictfitt = dict()
    dictfitt['typemodl'] = 'PlanetarySystemEmittingCompanion'
    
    # the following two are mutually exclusive
    liststrgtypedata = [['simutargpartsynt'], []]
    #typepriocomp = 'pdim'

    listtypelcurtpxftess = ['lygos']
    
    for typelcurtpxftess in listtypelcurtpxftess:
        miletos.main.init( \
                       strgmast='WASP-121', \
                       liststrgtypedata=liststrgtypedata, \
                       boolplotpopl=True, \
                       typelcurtpxftess=typelcurtpxftess, \
                       dictfitt=dictfitt, \
                       #typepriocomp=typepriocomp, \
                      )


def cnfg_TOI1410():
    '''
    SPACE Program
    '''

    dictfitt = dict()
    dictfitt['typemodl'] = 'PlanetarySystem'
    
    miletos.main.init( \
                       toiitarg=1410, \
                       dictfitt=dictfitt, \
                      )


def cnfg_TOIs():
    '''
    Analyze all TOIs
    '''

    for k in range(101, 7000):
        miletos.main.init( \
                   toiitarg=k, \
                   strgclus='TOIs', \
                   boolforcoffl=True, \
                  )


def cnfg_TOI_multis():

    dictfitt = dict()
    dictfitt['typemodl'] = 'PlanetarySystem'
    
    for strgmast in ['TOI-270', 'TOI-700', 'TOI-1233', 'TOI-1339']:
        
        miletos.main.init( \
                       strgmast=strgmast, \
                       dictfitt=dictfitt, \
                       strgclus='TOI_multis', \
                      )


def cnfg_Interesting_TICs_PlanetarySystem():
    
    # include
    # random transitter
    strgmast = 'HD 139139'

    listtici = [ \
                
                # the random transiter (HD 139139; Rappaport+2019)
                61253912, \
                
                # A Sextuply-Eclipsing Sextuple Star System (Powell+2021)
                168789840, \

                # eclipsing brown dwarf binary (Triaud+2020)
                61253912, \
                
                # ramped bottom
                #85791385, \
                
               ]

    dictfitt = dict()
    dictfitt['typemodl'] = 'PlanetarySystem'
    
    for tici in listtici:
        miletos.main.init( \
                          ticitarg=tici, \
                          strgclus='Interesting_TICs', \
                          dictfitt=dictfitt, \
                         )


def cnfg_TOI_Host_Variability():
    '''
    Analyze the variability of the host star
    ''' 
    
    listtoii = [ \
                270, \
               ]

    dictfitt = dict()
    dictfitt['typemodl'] = 'stargpro'
    
    dictpboxinpt = {'boolmult': True}
    for toii in listtoii:
        miletos.main.init( \
                          toiitarg=toii, \
                          strgclus='TOI_Host_Variability', \
                          dictfitt=dictfitt, \
                          dictpboxinpt=dictpboxinpt, \
                         )


def cnfg_TICsFromGV():
    
    listtici = [ \
                # potential target for Redyan
                318180448, \
               ]

    dictpboxinpt = {'boolmult': True}
    for tici in listtici:
        miletos.main.init( \
                          ticitarg=tici, \
                          strgclus='TICsFromGV', \
                          dictpboxinpt=dictpboxinpt, \
                         )


def cnfg_FermiLAT():
    '''
    Targets drawn from Fermi-LAT data with Manel and Banafsheh
    '''
    
    liststrgmast = []

    ## Blazars for ADAP 2022
    #path = os.environ['MILETOS_DATA_PATH'] + '/data/FermiLAT_TESS_AGN/interesting_blazars_CVZ.txt'
    #print('Reading from %s...' % path)
    #dictagns = pd.read_csv(path, skiprows=2, delimiter='|').to_dict(orient='list')
    #for strg in dictagns['_Search_Offset               ']:
    #    liststrgmast.append(strg.split('(')[1].split(')')[0])

    #
    #liststrgmast.extend([ \
    #           'BL Lacertae', \
    #           'PKS 2155-304', \
    #           
    #           # CCD edge
    #           '3C 279', \
    #           
    #           # no data
    #           '3C 454.3', \
    #           'PG 1553+113', \
    #           'PKS 1502+106', \
    #           '4FGL J1800.6+7828', '4FGL J1700.0+6830', '4FGL J1821.6+6819', '4FGL J0601.1-7035', '4FGL J1748.6+7005', '3C 371', 'Mkn 421', 'Mkn 501', \
    #           'CGCG 050-083', '1RXS J234354.4+054713', \
    #           ])
    #
    #path = os.environ['MILETOS_DATA_PATH'] + '/data/FermiLAT_TESS_AGN/.txt'
    #k = 0
    #print('Reading from %s...' % path)
    #for line in open(path, 'r'):
    #    if k == 0:
    #        k += 1
    #        continue
    #    liststrgmast.append(line[1:18])
    #    k += 1

    #path = os.environ['MILETOS_DATA_PATH'] + '/data/FermiLAT_TESS_AGN/BLLac.txt'
    #k = 0
    #print('Reading from %s...' % path)
    #for line in open(path, 'r'):
    #    if k == 0:
    #        k += 1
    #        continue
    #    liststrgmast.append(line[1:18])
    #    k += 1


    path = os.environ['MILETOS_DATA_PATH'] + '/data/FermiLAT_TESS_AGN/AllNameListInfo_last.txt'
    k = 0
    print('Reading from %s...' % path)
    objtfile = open(path, 'r')
    for line in objtfile:
        if k == 0:
            k += 1
            continue
        liststrgmast.append(line[1:18])
        k += 1


    numbtarg = len(liststrgmast)
    print('Extracting the TESS light curves of Fermi-LAT AGNs...')
    print('Number of targets: %d' % numbtarg)
    
    dictlygoinpt = dict()
    dictlygoinpt['typepsfninfe'] = 'fixd'
    
    dictfitt = dict()
    dictfitt['typemodl'] = 'AGN'
    
    dictlygoinpt = dict()
    dictlygoinpt['boolutiltpxf'] = False

    strgclus = 'FermiLAT_Targets'
    
    typelcurtpxftess = 'SPOC_only'

    listtimescalbdtr = [[6.]]
    #listtimescalbdtr = [1. / 24, 1., 5.]
    listtimescalbdtr = []
    
    numbtarg = len(liststrgmast)
    for k, strgmast in enumerate(liststrgmast):
        print('Target number: %d out of %d' % (k, numbtarg))
        
        miletos.main.init( \
                          strgmast=strgmast, \
                          strgclus=strgclus, \
                          typelcurtpxftess=typelcurtpxftess, \
                          listtimescalbdtr=listtimescalbdtr, \
                          dictlygoinpt=dictlygoinpt, \
                          dictfitt=dictfitt, \
                         )


def cnfg_TargetsOfInterest():
    
    dictlygoinpt = dict()
    dictlygoinpt['boolplotcntp'] = True
    liststrgmast = [ \
                    # from Ben Rackham
                    'Ross 619', \
                    # planetary system from PFS team
                    #'HD 140283', \
                    # from Prajwal
                    #'GJ 504', \
                    # NGTS target from Max
                    #'TIC 125731343', \
                    # second TRAPPIST-1 candidate from Max
                    #'TIC 233965332', \
                   ]
    
    strgclus = inspect.stack()[0][3][5:]
    
    dictlygoinpt['numbside'] = 17
    for strgmast in liststrgmast:
        
        if strgmast == 'TIC 125731343':
            dictlygoinpt['limttimeignoqual'] = [2458609., 2458611.]
        elif strgmast == 'Ross 619':
            dictlygoinpt['limttimeignoqual'] = [-np.inf, np.inf]
            dictlygoinpt['boolmaskqual'] = False
        else:
            dictlygoinpt = None
        
        miletos.main.init( \
                          strgmast=strgmast, \
                          dictlygoinpt=dictlygoinpt, \
                          strgclus=strgclus, \
                         )


def cnfg_TOI2406():
    
    for a in range(1):
        
        if a == 0:
            strgmast = None
            ticitarg = 233965332
            toiitarg = None
            labltarg = None
        if a == 1:
            strgmast = None
            ticitarg = 212957637
            toiitarg = None
            labltarg = 'TOI-2406 Neigbor TIC212957637'
        if a == 2:
            strgmast = None
            ticitarg = None
            toiitarg = 2406
            labltarg = None
        miletos.main.init( \
                       strgmast=strgmast, \
                       ticitarg=ticitarg, \
                       toiitarg=toiitarg, \
                       labltarg=labltarg, \
                       #listlimttimemask=[[[[2458370, 2458385]]], []], \
                      )


def cnfg_ULTRASAT():
    
    dicttrue = dict()
    dicttrue['typemodl'] = 'StarFlaring'
    
    dictlygoinpt = dict()
    dictlygoinpt['numbside'] = 51
    dictlygoinpt['maxmtmagcatl'] = 9.
    liststrgtypedata = [['simutargpartinje', 'simutargpartsynt'], []]
    miletos.init( \
         strgmast='Sirius', \
         dicttrue=dicttrue, \
         listlablinst=[['TESS', 'ULTRASAT'], []], \
         liststrgtypedata=liststrgtypedata, \
         dictlygoinpt=dictlygoinpt, \
        )
        

def cnfg_TESSGEO():
    
    dicttrue = dict()
    #dicttrue['typemodl'] = 'PlanetarySystemEmittingCompanion'
    dicttrue['typemodl'] = 'PlanetarySystem'
    
    # temperature of the star [K]
    dicttrue['tmptstar'] = 10000. # [K]

    # distance to the star [parsec]
    dicttrue['distsyst'] = 20.

    # radius of the star [R_S]
    dicttrue['radistar'] = 0.01

    dictnicoinpt = dict()
    dictlygoinpt = dict()
    dictlygoinpt['numbside'] = 51
    
    liststrgtypedata = [[ \
                         'simutargsynt', \
                         'simutargsynt', \
                         'simutargsynt', \
                         'simutargsynt', \
                         #'simutargpartsynt', \
                         #'simutargpartinje', \
                         #'simutargpartsynt', \
                         ], []]
    
    listlablinst = [[ \
                     'TESS-GEO-UV', \
                     'TESS-GEO-VIS', \
                     'TESS', \
                     'ULTRASAT', \
                     ], []]
    
    numbwdwf = 3
    dictnicoinpt['typestar'] = 'wdwf'
    dictnicoinpt['minmnumbcompstar'] = 1
    dictnicoinpt['maxmnumbcompstar'] = 1
    dicttrue['dictnicoinpt'] = dictnicoinpt
    for k in range(numbwdwf):
        
        np.random.seed(k)
        
        miletos.init( \
             strgmast='Simulation %02d' % k, \
             
             strgclus='TESS-GEO', \
             dicttrue=dicttrue, \
             listlablinst=listlablinst, \
             liststrgtypedata=liststrgtypedata, \
             dictlygoinpt=dictlygoinpt, \
            )
        

def cnfg_Sirius():
    
    dictlygoinpt = dict()
    dictlygoinpt['numbside'] = 51
    dictlygoinpt['maxmtmagcatl'] = 9.
    miletos.init( \
         strgmast='Sirius', \
         listlablinst=[['TESS'], []], \
         dictlygoinpt=dictlygoinpt, \
        )
        

def cnfg_TRAPPIST1():
    
    dictfitt = dict()
    dictfitt['typemodl'] = 'PlanetarySystemWithTTVs'
    
    liststrgtypedata = [['simutargpartinje'], []]
    liststrgtypedata = None
    
    #liststrgexpr = ['TESS', 'Kepler', 'K2', 'JWST_NIRSpec']
    liststrgexpr = ['TESS', 'Kepler', 'K2']
    #liststrgexpr = ['JWST']

    miletos.main.init( \
                   strgmast='TRAPPIST-1', \
                   #boolplotpopl=True, \
                   
                   boolforcoffl=True, \
                   
                   typelcurtpxftess='SPOC', \
                   liststrgtypedata=liststrgtypedata, \

                   liststrgexpr=liststrgexpr, \

                   dictfitt=dictfitt, \
                   #typepriocomp='pdim', \
                  )


def cnfg_WASP18():
    
    dictfitt = dict()
    dictfitt['typemodl'] = 'PlanetarySystemWithTTVs'
    
    liststrgtypedata = [['simutargpartinje'], []]
    miletos.main.init( \
                   strgmast='WASP-18', \
                   #boolplotpopl=True, \
                   boolforcoffl=True, \
                   liststrgtypedata=liststrgtypedata, \
                   
                   dictfitt=dictfitt, \
                   #typepriocomp='pdim', \
                  )


def cnfg_WASP43():
    
    liststrgtypedata = [['simutargpartinje'], []]
    
    dicttrue = dict()
    dicttrue['typemodl'] = 'PlanetarySystemEmittingCompanion'
    listlablinst = [['JWST'], []]

    listener = [np.linspace(0.5, 5., 10)]
    miletos.main.init( \
                   strgmast='WASP-43', \
                   dicttrue=dicttrue, \
                   
                   #boolforcoffl=True, \
                   listlablinst=listlablinst, \
                   listener=listener, \

                   liststrgtypedata=liststrgtypedata, \
                  )


def cnfg_flare_simulated():
    
    typepopl = 'targtess_prms_2min'
    dicttic8 = nicomedia.retr_dictpopltic8(typepopl=typepopl)
        
    for a in range(2):
        if a == 0:
            lablcnfg = 'Simulated'
            strgcnfg = 'simu'
            liststrgtypedata = [['simutargpartinje'], []]
            dicttrue = dict()
            dicttrue['typemodl'] = 'StarFlaring'
        else:
            lablcnfg = ''
            strgcnfg = 'real'
        dictfitt = dict()
        dictfitt['typemodl'] = 'StarFlaring'
        
        print('dicttic8.keys()')
        print(dicttic8.keys())
        numbtarg = dicttic8['tici'].size

        indxtarg = np.arange(numbtarg)
        for k in indxtarg:
            
            #lablltarg = 'TIC %d' % dicttic8['labl'][k]
            miletos.main.init( \
                          #labltarg=labltarg, \
                          #rasctarg=rasctarg, \
                          #decltarg=decltarg, \
                          ticitarg=dicttic8['tici'][k], \
                           
                          typelcurtpxftess='SPOC_only', \
                        
                          strgclus='SimulatedFlares', \

                          lablcnfg=lablcnfg, \
                          strgcnfg=strgcnfg, \
                          
                          tmagsyst=dicttic8['tmag'][k], \

                          dictfitt=dictfitt, \
                          dicttrue=dicttrue, \
                          liststrgtypedata=liststrgtypedata, \

                          #refrlistlabltser=refrlistlabltser, \
                          #refrarrytser=refrarrytser, \
                          #dictlygoinpt=dictlygoinpt, \
                         )



def cnfg_WASP12():

    dictfitt = dict()
    dictfitt['typemodl'] = 'PlanetarySystemWithTTVs'
    
    miletos.main.init( \
                   strgmast='WASP-12', \
                   boolplotpopl=True, \
                   dictfitt=dictfitt, \
                  )


def cnfg_PhotCalibPPAStars():
    
    listtici = [381979590, 264221449, 233160374, 289572157, 165553746]
    
    dictfitt = dict()
    dictfitt['typemodl'] = 'PlanetarySystemWithTTVs'
    
    for tici in listtici:
        dictlygoinpt = dict()
        dictlygoinpt['boolplotrflx'] = True
        dictlygoinpt['boolplotcntp'] = True
        dictlygoinpt['boolplotquat'] = False
        dictlygoinpt['boolplothhistcntp'] = False
        #dictlygoinpt['typepsfninfe'] = 'locl'
        dictlygoinpt['typepsfninfe'] = 'fixd'
        dictlygoinpt['boolphotaper'] = True
        dictlygoinpt['boolanim'] = False
        dictlygoinpt['booloutpaper'] = True
        dictlygoinpt['numbside'] = 17
        dictlygoinpt['maxmdmag'] = 3
        dictlygoinpt['dictfitt'] = {'typepsfnshap': 'data'}
        
        boolnormphot = False

        if tici == 381979590:
            listtsecsele = np.concatenate((np.arange(1, 28), np.arange(29, 33), np.arange(34, 50)))
    
        # temp
        print('temp')
        listtsecsele = [40]

        # TESS EM2 proposal continuous-viewing zone target
        miletos.main.init( \
                       ticitarg=tici, \
                       dictlygoinpt=dictlygoinpt, \
                       boolnormphot=boolnormphot, \
                       listtsecsele=listtsecsele, \
                       boolffimonly=True, \
                       boolnorm=boolnormphot, \
                       strgclus='PhotCalibPPAStars', \
                       dictfitt=dictfitt, \
                       #boolplotpopl=True, \
                      )


def cnfg_WhiteDwarfs_Candidates_TESS_GI():
    
    pathoutp = os.environ['MILETOS_DATA_PATH'] + '/data/WD/'
    os.system('mkdir -p %s' % pathoutp)
    listisec = np.arange(27, 28)
    
    dictfitt = dict()
    dictfitt['typemodl'] = 'PlanetarySystem'
    
    for isec in listisec:
        path = 'https://heasarc.gsfc.nasa.gov/docs/tess/data/target_lists/sector%03d_targets_lists/GI_20s_S%03d.csv' % (isec, isec)
        pathfile = wget.download(path, out=pathoutp)
        arry = pd.read_csv(pathfile, delimiter=',').to_numpy()
        listtici = arry[:, 0]
        numbtici = len(listtici)
        indxtici = np.arange(numbtici)
        for i in indxtici:
            
            miletos.main.init( \
                 strgmast='TIC %d' % listtici[i], \
                 dictfitt=dictfitt, \
                 typepriocomp='pdim', \
                 strgclus='WhiteDwarfs', \
                )


def cnfg_WhiteDwarf_Candidates_TOI_Process():
    
    listticitarg = [ \
                    
                    #686072378, \
                    
                    # from TOI vetting, small (subdwarf) star, 30 September 2021
                    1400704733, \
                   ]
    
    dictfitt = dict()
    dictfitt['typemodl'] = 'PlanetarySystem'
    
    for ticitarg in listticitarg:
        miletos.main.init( \
             ticitarg=ticitarg, \
             strgclus='cnfg_WhiteDwarf_Candidates_TOI_Process', \
             dictfitt=dictfitt, \
            )


def cnfg_V563Lyr():
    
    miletos.main.init( \
         strgmast='V563 Lyr', \
         typepriocomp='pdim', \
         massstar=1., \
        )


def cnfg_AlphaCen():
    
    radistar = 1.
    massstar = 1.
    tmptstar = 1.
    stdvradistar = 1.
    stdvmassstar = 1.
    stdvtmptstar = 1.
    
    strgmast = 'Alpha Cen'

    path = os.environ['TDGU_DATA_PATH'] + '/alph_cent/AlphaCen_calibrated_2min.csv'
    pathbase = os.environ['TDGU_DATA_PATH'] + '/alph_cent/'
    listpathdatainpt = [[pathbase + 'AlphaCen_calibrated_1min.csv', pathbase + 'AlphaCen_calibrated_2min.csv']]
    
    dictsettalle = dict()
    dictsettalle['fast_fit'] = 'False'

    # from Dumusque+2012
    periprio = np.array([3.2357])
    peristdvprio = np.array([0.0008])
    epocprio = np.array([2455280.97892])
    epocstdvprio = np.array([0.17])
    
    listdeptdraw = np.array([35e-6])
    epocstdvprio *= 10.
    peristdvprio *= 10.

    miletos.main.init( \
                   # data
                   listpathdatainpt=listpathdatainpt, \
                   listlablinst=['ASTERIA'], \
                   
                   # priors
                   ## stellar
                   radistar=radistar, \
                   stdvradistar=stdvradistar, \
                   massstar=massstar, \
                   stdvmassstar=stdvmassstar, \
                   tmptstar=tmptstar, \
                   stdvtmptstar=stdvtmptstar, \
                   
                   ## planetary
                   typepriocomp='inpt', \
                   numbplan=1, \
                   periprio=periprio, \
                   peristdvprio=peristdvprio, \
                   epocprio=epocprio, \
                   epocstdvprio=epocstdvprio, \
                   
                   # file name extension
                   labltarg='Alpha Cen', \
                   boolwritplan=False, \
                   
                   strgmast=strgmast, \
                   
                   listdeptdraw=listdeptdraw, \

                   durabrek=0.25, \
                   ordrspln=2, \

                   dictsettalle=dictsettalle, \


                  )


def cnfg_HR6819():
    
    strgmast = 'HR 6819'
    miletos.main.init( \
                   strgmast=strgmast, \
                  )


def cnfg_HD136352():
    
    rratprio = np.array([0.05, 0.05])
    rsmaprio = np.array([0.1, 0.1])
    cosiprio = np.array([0., 0.])
    epocprio = np.array([2458631.7672, 2458650.89472])
    periprio = np.array([11.57872, 27.5951])
    miletos.main.init( \
                   boolmaskqual=False, \
                   rratprio=rratprio, \
                   rsmaprio=rsmaprio, \
                   epocprio=epocprio, \
                   periprio=periprio, \
                   cosiprio=cosiprio, \
                   strgmast='HD 136352', \
                  )


def cnfg_TOI2155():
    
    miletos.main.init( \
                   toiitarg=2155, \
                  )


def cnfg_TOI1431():
    
    miletos.main.init( \
                   toiitarg=1431, \
                  )


def cnfg_WASP(strgiwas):
    
    iwas = int(strgiwas)
    #listiwas = [18, 46]
    #for iwas in listiwas:
    strgmast = 'WASP-%d' % iwas
    
    dictfitt = dict()
    dictfitt['typemodl'] = 'PlanetarySystemWithTTVs'
    
    miletos.main.init( \
                  strgmast=strgmast, \
                  dictfitt=dictfitt, \
                 )


def cnfg_TOI193():
    
    radistar = 0.94 * 9.95 # [R_J]
    stdvradistar = 0.01 * 9.95 # [R_J]
    massstar = 1.02 * 1048. # [M_J]
    stdvmassstar = 0.04 * 1048. # [M_J]
    tmptstar = 5445. # [K]
    stdvtmptstar = 84. # [K]
    miletos.main.init( \
                   pcurtype='sinu', \
                   toiitarg=193, \
                   radistar=radistar, \
                   stdvradistar=stdvradistar, \
                   massstar=massstar, \
                   stdvmassstar=stdvmassstar, \
                   tmptstar=tmptstar, \
                   stdvtmptstar=stdvtmptstar, \

                  )


def cnfg_GJ299():
    
    radistar = 0.175016 * 9.95 # [R_J]
    massstar = 0.14502 * 1048. # [M_J]
    miletos.main.init( \
         ticitarg=334415465, \
         labltarg='GJ 299', \
         strgmast='GJ 299', \
         radistar=radistar, \
         massstar=massstar, \
        )


def cnfg_TOI_lists(name):
    '''
    Gran-Unified Hot Jupiter Survey
    '''

    # MuSCAT2
    if name == 'MuSCAT2':
        listhosttoiithis = np.array([1752, 2079, 2278, 2431, 3884, 3984, 4107, 4114, 4325, 4363, 4616, 4643])

    # Gran-Unified Hot Jupiter Survey (GUHJS2)
    if name == 'GUHJS2':
        listhosttoiithis = np.array([1937, 2364, 2583, 2587, 2796, 2803, 2818, 2842, 2977, 3023, 3364, 3688, 3807, 3819, 3912, 3976, 3980, 4087, 4145, 4463, 4791])
    
    listtoiifstr = ephesos.retr_toiifstr()
    listtoiifstr = listtoiifstr.astype(float)
    listhosttoiifstr = listtoiifstr.astype(int)
    
    listhosttoiithisfstr = []
    for hosttoii in listhosttoiithis:
        if hosttoii in listhosttoiifstr:
            listhosttoiithisfstr.append(hosttoii)
    listhosttoiithisfstr = np.array(listhosttoiithisfstr)
    
    
    print('listhosttoiithis')
    summgene(listhosttoiithis)
    print('listhosttoiifstr')
    summgene(listhosttoiifstr)
    print('listhosttoiithisfstr')
    summgene(listhosttoiithisfstr)
    
    print('Intersection')
    for hosttoii in listhosttoiithisfstr:
        print(hosttoii)


def cnfg_Rafael():
    
    miletos.main.init( \
         ticitarg=356069146, \
         strgclus='Rafael', \
        )


def cnfg_allesfitter():
    '''
    Examples in the Allesfitter paper (Guenther & Daylan, 2021)
    '''
    
    #factmsmj = 1048.
    #factrsrj = 9.95
    
    liststrgmast = ['Pi Mensae', 'TOI-216', 'WASP-18', 'KOI- 1003', 'GJ 1243']
    for strgmast in liststrgmast:
        miletos.main.init( \
             strgmast=strgmast, \
             #ticitarg=122374527, \
             #labltarg='KOI 1003', \
             #radistar=2.445*factmsmj, \
             #massstar=1.343*factrsrj, \
            )


def cnfg_HD118203():
    
    strgmast = 'HD 118203'
    labltarg = 'HD 118203'
    
    listlimttimemask = np.array([ \
                                [0, 1712], \
                                [1724.5, 1725.5], \
                                ])
    listlimttimemask += 2457000
    epocprio = np.array([2458712.662354])
    periprio = np.array([6.134842])
    duraprio = np.array([5.6457]) / 24. # [day]
    rratprio = np.sqrt(np.array([3516.19165]) * 1e-6)
    miletos.main.init( \
         labltarg=labltarg, \
         strgmast=strgmast, \
         epocprio=epocprio, \
         periprio=periprio, \
         duraprio=duraprio, \
         rratprio=rratprio, \
         listlimttimemask=listlimttimemask, \
        )


def cnfg_transit_asymmetry():
    '''
    Search for asymmetric transits
    '''
    
    # candidate with asymmetric transit from the TOI process
    listtici = [86263325, 233462817]
    
    listtypeanls = ['asymtran']
    for tici in listtici:
        miletos.main.init( \
                          ticitarg=tici, \
                          listtypeanls=listtypeanls, \
                         )


def cnfg_LSST_PlanetarySystem():
    
    dicttrue = dict()
    dicttrue['typemodl'] = 'PlanetarySystem'
    
    dictfitt = dict()
    dictfitt['typemodl'] = 'PlanetarySystem'
    
    liststrgtypedata = [[], []]
    listlablinst = [[], []]
    liststrglsst = ['u', 'g', 'r', 'i', 'z', 'y']
    for strglsst in liststrglsst:
        listlablinst[0].append('LSST %s band' % strglsst)
        liststrgtypedata[0].append('simutargsynt')
    
    # temperature of the star [K]
    dicttrue['tmptstar'] = 10000. # [K]

    # distance to the star [parsec]
    dicttrue['distsyst'] = 20.

    # radius of the star [R_S]
    dicttrue['radistar'] = 0.01

    dicttrue['rratcomp'] = [np.array([0.1]), np.array([0.1]), np.array([0.1]), np.array([0.1]), np.array([0.1]), np.array([0.1])]
    dicttrue['epocmtracomp'] = np.array([0.])
    dicttrue['pericomp'] = np.array([3.])
    dicttrue['rsmacomp'] = np.array([0.1])
    dicttrue['cosicomp'] = np.array([0.])
    
    for typeanls in ['outlperi', 'pdim']:

        for numbyearlsst in [1., 5., 10.]:
            
            dicttrue['numbyearlsst'] = numbyearlsst

            miletos.main.init( \
                              labltarg='Simulated Jupiter, %d year' % numbyearlsst, \
                              
                              dicttrue=dicttrue, \
                              dictfitt=dictfitt, \
                              
                              listtypeanls=[typeanls], \
                              
                              listlablinst=listlablinst, \
                              liststrgtypedata=liststrgtypedata, \
                             )


def cnfg_Cygnus1():
    
    miletos.main.init( \
                      strgmast='Cygnus-1', \
                     )




def cnfg_NGTS11():
    
    # TOI-1847
    ticitarg = 54002556
    strgmast = 'NGTS-11'
    
    epocprio = np.array([2458390.7043, 2459118])
    periprio = np.array([35.45533, np.nan])
    miletos.main.init( \
         # light curve
         ticitarg=ticitarg, \
         # stellar prior
         strgmast=strgmast, \
         # planetary prior
         epocprio=epocprio, \
         periprio=periprio, \
        )


def cnfg_HATP19():
    
    ticitarg = 267650535
    strgmast = 'HAT-P-19'
    labltarg = 'HAT-P-19'
    
    miletos.main.init( \
         labltarg=labltarg, \
         strgmast=strgmast, \
         ticitarg=ticitarg, \
        )


def cnfg_WD1856():

    strgmast = 'WD 1856+534'
    
    dictfitt = dict()
    dictfitt['typemodl'] = 'PlanetarySystem'
    
    typepriocomp = 'exar'

    miletos.main.init( \
         strgmast=strgmast, \
         dictfitt=dictfitt, \
         typepriocomp=typepriocomp, \
         boolfitt=False, \

         #boolmakeanim=True, \
         #dilucorr=0.01, \
        )


def cnfg_Faint():

    liststrgexpr = ['TESS']
    #liststrgtypedata = [['simutargsynt'], []]
    
    # simulate two-dimensional images
    boolsimutdim = True

    dicttrue = dict()
    dicttrue['typemodl'] = 'PlanetarySystem'
    
    miletos.main.init( \
                      dicttrue=dicttrue, \
                      liststrgexpr=liststrgexpr, \
                      #liststrgtypedata=liststrgtypedata, \
                     )


def cnfg_ASASSN20qc():
    '''
    13 July 2021, 2020adgm, AGN from DJ
    '''
    
    rasctarg = 63.260208 
    decltarg = -53.0727

    dictfitt = dict()
    dictfitt['typemodl'] = 'supn'

    labltarg = 'ASASSN-20qc'
    
    refrlistlabltser = [['Michael']]

    dictlygoinpt = dict()
    dictlygoinpt['boolplotrflx'] = True
    dictlygoinpt['boolplotcntp'] = True
    #dictlygoinpt['boolfittoffs'] = True
    dictlygoinpt['boolplotquat'] = True
    #dictlygoinpt['numbside'] = 9
    
    listtsecsele = [32]

    #path = os.environ['LYGOS_DATA_PATH'] + '/data/lc_2020adgm_cleaned_ASASSN20qc'
    #print(path)
    #objtfile = open(path, 'r')
    #k = 0
    #linevalu = []
    #for line in objtfile:
    #    if k == 0:
    #        k += 1
    #        continue
    #    linesplt = line.split(' ')
    #    linevalu.append([])
    #    for linesplttemp in linesplt:
    #        if linesplttemp != '':
    #            linevalu[k-1].append(float(linesplttemp))
    #    linevalu[k-1] = np.array(linevalu[k-1])
    #    k += 1
    #linevalu = np.vstack(linevalu)
    #refrarrytser = np.empty((linevalu.shape[0], 3))
    #refrarrytser[:, 0] = linevalu[:, 0]
    #refrarrytser[:, 1] = linevalu[:, 2]
    #refrarrytser[:, 2] = linevalu[:, 3]
   
    listlimttimemask = [[[-np.inf, 2457000 + 2175], [2457000 + 2186.5, 2457000 + 2187.5]]]
    listlimttimemask = [[[[2457000 + 2186.5, 2457000 + 2187.5]]]]

    listnumbside = [7, 11, 15]
    for numbside in listnumbside:
        if numbside == 11:
            listtimescalbdtr = [0., 0.1, 0.5]
        else:
            listtimescalbdtr = [0.]
        miletos.main.init( \
                      labltarg=labltarg, \
                      rasctarg=rasctarg, \
                      decltarg=decltarg, \

                      dictfitt=dictfitt, \
                      listtsecsele=listtsecsele, \

                      #refrlistlabltser=refrlistlabltser, \
                      #refrarrytser=refrarrytser, \

                      dictlygoinpt=dictlygoinpt, \
                     )



def cnfg_TESS_EB_Catalog():

    # random TICs from the TESS EB Catalog ()
    listtici = [91961, 101462, 120016, 627436, 737546]
    
    dictfitt = dict()
    dictfitt['typemodl'] = 'PlanetarySystem'
    
    for tici in listtici:
        miletos.main.init( \
             ticitarg=tici, \
             strgclus='TESS_EB_Catalog', \
             dictfitt=dictfitt, \
            )


globals().get(sys.argv[1])(*sys.argv[2:])
