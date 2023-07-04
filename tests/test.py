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


def cnfg_TOIs():
    
    for k in range(101, 5000):
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


def cnfg_FermiLAT_AGN():
    
    liststrgmast = []
    path = os.environ['MILETOS_DATA_PATH'] + '/data/ListVariability_AssN_BanafshehManel.txt'
    k = 0
    for line in open(path, 'r'):
        if k == 0:
            k += 1
            continue
        liststrgmast.append(line[1:18])
        k += 1

    numbtarg = len(liststrgmast)
    print('Extracting the TESS light curves of Fermi-LAT AGNs...')
    print('Number of targets: %d' % numbtarg)
    
    listtimescalbdtrspln = [[6.]]
    for strgmast in liststrgmast:
        print(strgmast)
    
    dictlygoinpt = dict()
    dictlygoinpt['boolutiltpxf'] = False

    strgclus = 'FermiLAT_AGN'
    
    typelcurtpxftess = 'SPOC_only'
    #liststrgmast = ['4FGL J1800.6+7828', '4FGL J1700.0+6830', '4FGL J1821.6+6819', '4FGL J0601.1-7035', '4FGL J1748.6+7005', '3C 371']
    liststrgmast = ['Mkn 421', 'Mkn 501', 'CGCG 050-083', '1RXS J234354.4+054713', '3C 371']
    #listtsecsele = [26]

    #listtimescalbdtrspln = [1. / 24, 1., 5.]
    listtimescalbdtrspln = []
    
    for strgmast in liststrgmast:
        miletos.main.init( \
                          strgmast=strgmast, \
                          strgclus=strgclus, \
                          typelcurtpxftess=typelcurtpxftess, \
                          #listtsecsele=listtsecsele, \
                          listtimescalbdtrspln=listtimescalbdtrspln, \
                          dictlygoinpt=dictlygoinpt, \
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


def cnfg_Sirius():
    
    dictlygoinpt = dict()
    dictlygoinpt['numbside'] = 51
    dictlygoinpt['maxmtmagcatl'] = 9.
    miletos.init( \
         strgmast='Sirius', \
         booldiag=True, \
         listlablinst=[['TESS'], []], \
         dictlygoinpt=dictlygoinpt, \
        )
        

def cnfg_TOI1797():
    
    dictfitt = dict()
    dictfitt['typemodl'] = 'PlanetarySystem'
    
    for a in range(2):
        
        if a == 0:
            boolsimutargpartfprt = True
            strgcnfg = 'Simulated'
        if a == 1:
            boolsimutargpartfprt = False
            strgcnfg = 'Observed'

        #dictlygoinpt = dict()
        #dictlygoinpt['numbside'] = 51
        #dictlygoinpt['maxmtmagcatl'] = 9.
        miletos.init( \
             toiitarg=1797, \
             dictfitt=dictfitt, \
             strgcnfg=strgcnfg, \
             boolsimutargpartfprt=boolsimutargpartfprt, \
             #booldiag=False, \
             #listlablinst=[['TESS'], []], \
             #dictlygoinpt=dictlygoinpt, \
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

                   booldiag=True, \
                   
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
                   #booldiag=True, \
                   
                   dictfitt=dictfitt, \
                   #typepriocomp='pdim', \
                  )


def cnfg_WASP121():
    
    dictfitt = dict()
    dictfitt['typemodl'] = 'PlanetarySystemWithPhaseCurve'
    
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


def cnfg_WASP43():
    
    liststrgtypedata = [['simutargpartinje'], []]
    
    dicttrue = dict()
    dicttrue['typemodl'] = 'PlanetarySystemWithPhaseCurve'
    listlablinst = [['JWST'], []]

    listener = [np.linspace(0.5, 5., 10)]
    miletos.main.init( \
                   strgmast='WASP-43', \
                   dicttrue=dicttrue, \
                   
                   #boolforcoffl=True, \
                   listlablinst=listlablinst, \
                   listener=listener, \

                   booldiag=True, \
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
            dicttrue['typemodl'] = 'flar'
        else:
            lablcnfg = ''
            strgcnfg = 'real'
        dictfitt = dict()
        dictfitt['typemodl'] = 'flar'
        
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
                          booldiag=True, \
                          
                          liststrgtypedata=liststrgtypedata, \

                          #refrlistlabltser=refrlistlabltser, \
                          #refrarrytser=refrarrytser, \
                          #dictlygoinpt=dictlygoinpt, \
                         )


def cnfg_ATEL15755():
    
    # https://www.astronomerstelegram.org/?read=15755
    # X-ray transient LXT 221107A
    labltarg = 'HD 251108'
    rasctarg = 91.062625
    decltarg = 12.764722
    
    for a in range(2):
        if a == 0:
            lablcnfg = 'Simulated'
            strgcnfg = 'simu'
            liststrgtypedata = [['simutargpartinje'], []]
            dicttrue = dict()
            dicttrue['typemodl'] = 'flar'
        else:
            lablcnfg = ''
            strgcnfg = 'real'
        dictfitt = dict()
        dictfitt['typemodl'] = 'flar'
        miletos.main.init( \
                      labltarg=labltarg, \
                      rasctarg=rasctarg, \
                      decltarg=decltarg, \
                      
                      lablcnfg=lablcnfg, \
                      strgcnfg=strgcnfg, \

                      dictfitt=dictfitt, \
                      dicttrue=dicttrue, \
                      booldiag=True, \
                      
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


def cnfg_ADAP2022_AGNs():
    '''
    ADAP 2022 AGN targets with Manel et al.
    '''
    path = os.environ['LYGOS_DATA_PATH'] + '/data/interesting_blazars_CVZ.txt'
    print('Reading from %s...' % path)
    dictagns = pd.read_csv(path, skiprows=2, delimiter='|').to_dict(orient='list')
    
    listname = []
    for strg in dictagns['_Search_Offset               ']:
        listname.append(strg.split('(')[1].split(')')[0])
    print('listname')
    print(listname)

    #listname = [ \
    #           #'BL Lacertae', \
    #           #'Mkn 501', 'PKS 2155-304', \
    #           #'Mkn 421', \
    #           
    #           # CCD edge
    #           '3C 279', \
    #           
    #           # no data
    #           #'3C 454.3', \
    #           #'PG 1553+113', \
    #           #'PKS 1502+106', \
    #           ]
    
    dictlygoinpt = dict()
    dictlygoinpt['typepsfninfe'] = 'fixd'
    
    #dictlygoinpt['numbside'] = 25
    
    dictfitt = dict()
    dictfitt['typemodl'] = 'agns'
    
    for name in listname:
        miletos.main.init( \
                   strgclus='ADAP2022_AGNs', \
                   strgmast=name, \
                   dictfitt=dictfitt, \
                   dictlygoinpt=dictlygoinpt, \
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
                  booldiag=True, \
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
    
    dicttrue['rratcomp'] = [np.array([0.1]), np.array([0.1]), np.array([0.1]), np.array([0.1]), np.array([0.1]), np.array([0.1])]
    dicttrue['epocmtracomp'] = np.array([0.])
    dicttrue['pericomp'] = np.array([3.])
    dicttrue['rsmacomp'] = np.array([0.1])
    dicttrue['cosicomp'] = np.array([0.])
    
    #rratcompprio = [np.array([0.1]), np.array([0.1]), np.array([0.1]), np.array([0.1]), np.array([0.1]), np.array([0.1])]
    #epocmtracompprio = np.array([0.])
    #pericompprio = np.array([3.])
    #rsmacompprio = np.array([0.1])
    #cosicompprio = np.array([0.])
    
    for typeanls in ['outlperi', 'pdim']:

        for numbyearlsst in [1., 5., 10.]:
            
            dicttrue['numbyearlsst'] = numbyearlsst

            miletos.main.init( \
                              labltarg='Simulated Jupiter, %d year' % numbyearlsst, \
                              
                              dicttrue=dicttrue, \
                              dictfitt=dictfitt, \
                              
                              listtypeanls=[typeanls], \
                              
                              #rratcompprio=rratcompprio, \
                              #epocmtracompprio=epocmtracompprio, \
                              #pericompprio=pericompprio, \
                              #cosicompprio=cosicompprio, \
                              #rsmacompprio=rsmacompprio, \
                              
                              umagsyst=20., \
                              gmagsyst=20., \
                              rmagsyst=20., \
                              imagsyst=20., \
                              zmagsyst=20., \
                              ymagsyst=20., \
                              
                              listlablinst=listlablinst, \
                              liststrgtypedata=liststrgtypedata, \
                             )


def cnfg_Cygnus1():
    
    miletos.main.init( \
                      strgmast='Cygnus-1', \
                     )


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
                          booldiag=True, \
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

    strgmast = 'WD1856+534'
    ticitarg = 267574918
    
    dictfitt = dict()
    dictfitt['typemodl'] = 'psys'
    
    miletos.main.init( \
         strgmast=strgmast, \
         dictfitt=dictfitt, \
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


def cnfg_SNeIa_Comp(strgruns):
    

    pathbase = os.environ['DATA'] + '/other/Type_Ia_TNS/'
    
    numbiter = 12
    indxiter = np.arange(numbiter)
    listdicttrns = [[] for k in indxiter]
    for k in indxiter:
        path = pathbase + 'tns_search-%d.csv' % (k + 1)
        print('Reading from %s...' % path)
        listdicttrns[k] = pd.read_csv(path, skiprows=0, delimiter=',').to_dict(orient='list')
    
    dicttrns = dict()
    for name in ['Name', 'RA', 'DEC']:
        listtemp = [listdicttrns[k][name] for k in indxiter]
        dicttrns[name] = np.concatenate(listtemp)
    
    dictlygoinpt = dict()
    #dictlygoinpt['typepsfninfe'] = 'fixd'
    #dictlygoinpt['boolplotrflx'] = True
    #dictlygoinpt['boolplotcntp'] = True
    #dictlygoinpt['numbside'] = 5
    dictlygoinpt['typenorm'] = 'mediinit'
    #dictlygoinpt['booldetrcbvs'] = False
    #dictlygoinpt['boolfittoffs'] = True

    dictfitt = dict()
    dictfitt['typemodl'] = 'supn'
    dicttrue = dict()
    
    typeinfe = 'samp'

    if strgruns == 'totl':
        liststrgruns = [ \
                        # fitting simulated data no bump using a model with quadratic SN but without a bump
                        'simutargsynt_cons_quad_none_none_none', \
                        # fitting simulated data with a bump using a model with quadratic SN but without a bump
                        'simutargsynt_cons_quad_none_none_bump', \
                        # fitting simulated data with a bump using a model with quadratic SN and a bump
                        'simutargsynt_cons_quad_bump_none_bump', \
                        
                        # fitting simulated data with a bump *and red noise* using a model with a bump
                        'simutargsynt_cons_quad_bump_gpro_bump', \
                        # fitting simulated data with a bump *and red noise* using a model with a bump and GP baseline
                        'simutargsynt_gpro_quad_bump_gpro_bump', \

                        # fitting real data using a model with a SN
                        'obsd_cons_quad_none', \
                        # fitting real data using a model with a SN and bump
                        'obsd_cons_quad_bump', \
                        # fitting real data using a model with a SN, bump, and GP baseline
                        'obsd_gpro_quad_bump', \
                       ]
    else:
        liststrgruns = [strgruns]

    for strgruns in liststrgruns:
        strgrunssplt = strgruns.split('_')
        strgtypedata = strgrunssplt[0]
        dictfitt['typemodlbase'] = strgrunssplt[1]
        dictfitt['typemodlsupn'] = strgrunssplt[2]
        dictfitt['typemodlexcs'] = strgrunssplt[3]
        
        dictlygoinpt['boolanim'] = strgruns == 'real_samp_cons_quad_none'

        if strgtypedata == 'simutargsynt':
            dicttrue['typemodlbase'] = strgrunssplt[4]
            dicttrue['typemodlexcs'] = strgrunssplt[5]
            liststrgtypedata = [['simutargsynt'], []]
            
            numbtarg = 1
            for k in range(numbtarg):
                listlabltarg = ['Target %04d' % k]
        
        else:
            liststrgtypedata = None

            liststrgmast = [ \
                            # disfavor
                            'SN 2018fub', \
                            'SN 2020tld', \
                            'SN 2020swy', \
                            'SN 2021udg', \
                            'SN 2021zny', \
                            'SN 2022eyw', \
                            
                            #'SN 2020tld', \
                            #'SN 2021zny', \
                            'SN 2018hib', \
                            'SN 2020ftl', \

                            # positives
                            'SN 2018hkx', \
                            'SN 2020aoi', \
                            'SN 2020abqu', \
                            'SN 2021ahmz', \
                            'SN 2022ajw', \

                           ]
                
            numbtarg = len(liststrgmast)

        indxtarg = np.arange(numbtarg)
        for k in indxtarg:    
            
            if strgtypedata == 'real':
                if liststrgmast[k] == 'SN2018fub':
                    limttimefitt = [[[-np.inf, 2458368.]], []]
                elif liststrgmast[k] == 'SN2020swy':
                    limttimefitt = [[[-np.inf, 2459110.]], []]
                #elif liststrgmast[k] == 'SN2020swy':
                #    limttimefitt = [[[-np.inf, np.inf]], []]
                else:
                    limttimefitt = None
                    
                strgmast = liststrgmast[k]
                labltarg = None
                strgtarg = None
            else:
                limttimefitt = None
                strgmast = None
                labltarg = 'Simulated Target'
                strgtarg = 'targsimu%04d' % k
            
            strgcnfg = '%s_%s_%s' % (dictfitt['typemodlbase'], dictfitt['typemodlsupn'], dictfitt['typemodlexcs'])
            if strgtypedata == 'real':
                if liststrgmast[k] == 'SN2018fub':
                    listtsecsele = [2]
                if liststrgmast[k] == 'SN2020swy':
                    listtsecsele = [29]
                if liststrgmast[k] == 'SN2022ajw':
                    listtsecsele = [47]
                
                print('dicttrns[Name]')
                print(dicttrns['Name'])
                print('liststrgmast[k]')
                print(liststrgmast[k])
                indx = np.where(dicttrns['Name'] == liststrgmast[k])[0]
                if indx.size != 1:
                    raise Exception('')
                rasctarg = dicttrns['RA'][indx[0]]
                decltarg = dicttrns['DEC'][indx[0]]
                print('rasctarg')
                print(rasctarg)
                print('decltarg')
                print(decltarg)
                objtcoor = astropy.coordinates.SkyCoord('%s %s' % (rasctarg, decltarg), unit=(astropy.units.hourangle, astropy.units.deg))

                #objttimenigh = astropy.time.Time(astropy.time.Time(gdat.strgtimeobvtnigh).jd, format='jd', location=objtlocaobvt)
                #timeyear = astropy.time.Time(gdat.strgtimeobvtyear).jd + gdat.listdelttimeobvtyear
                #objttimeyear = astropy.time.Time(timeyear, format='jd', location=objtlocaobvt)
                rasctarg = objtcoor.ra.degree
                decltarg = objtcoor.dec.degree
                strgmast = None
            else:
                strgcnfg += '_%s_%s' % (dicttrue['typemodlbase'], dicttrue['typemodlexcs'])
                listtsecsele = None
                rasctarg = None
                decltarg = None
            
            dictmileoutp = miletos.main.init( \
                                             rasctarg=rasctarg, \
                                             decltarg=decltarg, \
                                             strgmast=strgmast, \
                                             strgclus='SNeIa_Comp', \
                                             liststrgtypedata=liststrgtypedata, \
                                             limttimefitt=limttimefitt, \
                                             listtsecsele=listtsecsele, \
                                             strgtarg=strgtarg, \
                                             labltarg=labltarg, \
                                             dicttrue=dicttrue, \
                                             typeinfe=typeinfe, \
                                             strgcnfg=strgcnfg, \
                                             listtypeanls=[], \
                                             dictlygoinpt=dictlygoinpt, \
                                             boolplotdvrp=False, \
                                             dictfitt=dictfitt, \
                                            )
            

def cnfg_SNeIa():
    
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    pathbase = os.environ['PERGAMON_DATA_PATH'] + '/featsupntess/'
    pathdata = pathbase + 'data/'
    pathimag = pathbase + 'imag/'
    os.system('mkdir -p %s' % pathdata)
    os.system('mkdir -p %s' % pathimag)

    dictfitt = dict()
    dictfitt['typemodl'] = 'supn'

    for strgfile in [ \
                     'Cycle1-matched', \
                     'Cycle2-matched', \
                     'Cycle3-matched', \
                    ]:

        pathcsvv = pathdata + '%s.csv' % strgfile
        
        #if strgfile == 'Cycle1-matched':
        #    strgclus = 'SNIa_Cycle1'
        
        strgclus = 'SNIa_' + strgfile.split('-')[0]
        
        print('Reading from %s...' % pathcsvv)
        objtfile = open(pathcsvv, 'r')
        k = 0
        for line in objtfile:
            if k == 0:
                k += 1
                continue
            linesplt = line.split(',')
            labltarg = linesplt[2]
            c = SkyCoord('%s %s' % (linesplt[3], linesplt[4]), unit=(u.hourangle, u.deg))
            rasctarg = c.ra.degree
            decltarg = c.dec.degree
            
            dictlygoinpt = dict()
            dictlygoinpt['boolanim'] = False
            dictlygoinpt['typepsfninfe'] = 'fixd'
            dictlygoinpt['boolplotrflx'] = True
            dictlygoinpt['boolplotcntp'] = True
            dictlygoinpt['numbside'] = 5
            dictlygoinpt['booldetrcbvs'] = False
            #dictlygoinpt['boolfittoffs'] = True
            
            dictmileoutp = miletos.main.init( \
                                             rasctarg=rasctarg, \
                                             decltarg=decltarg, \
                                             labltarg=labltarg, \
                                             strgclus=strgclus, \
                                             dictlygoinpt=dictlygoinpt, \
                                             dictfitt=dictfitt, \
                                            )
            
            #listtsec = dictmileoutp['listtsec']
            #numbtsec = len(listtsec)
            #indxtsec = np.arange(numbtsec)
            #for o in indxtsec:
            #    cmnd = 'cp %s%s/%s/imag/* %s%s/imag/' % (pathlygo, strgclus, dictoutp['strgtarg'], pathlygo, strgclus)
            #    print(cmnd)
            #    os.system(cmnd)
            #    pathsaverflxtarg = dictoutp['pathsaverflxtargsc%02d' % listtsec[o]]
            #    cmnd = 'cp %s %s%s/data/' % (pathsaverflxtarg, pathbase, strgclus)
            #    print(cmnd)
            #    os.system(cmnd)
            #k += 1


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
            listtimescalbdtrspln = [0., 0.1, 0.5]
        else:
            listtimescalbdtrspln = [0.]
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



def target(strgmast):
    '''
    Execute miletos on a target from command line.
    '''
    
    miletos.main.init( \
         strgmast=strgmast, \
        )


globals().get(sys.argv[1])(*sys.argv[2:])
