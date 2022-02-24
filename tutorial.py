import miletos
import sys
import os
import numpy as np
import wget
import pandas as pd
from tdpy.util import summgene

def cnfg_toii():
    
    for k in range(101, 5000):
        miletos.main.init( \
                   toiitarg=k, \
                   typepriocomp='pdim', \
                  )


def cnfg_multis():

    #for strgmast in ['TOI-270', 'TOI-700', 'TOI-1233', 'TOI-1339']:
    dictlcurtessinpt = dict()
    dictlcurtessinpt['booltpxfonly'] = True
    for strgmast in ['TOI-700']:
        miletos.main.init( \
                       strgmast=strgmast, \
                       dictlcurtessinpt=dictlcurtessinpt, \
                      )


def cnfg_TICsFromGV():
    
    listtici = [ \
                # 30 September 2021, small star
                #1400704733, \
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


def cnfg_requests():
    
    liststrgmast = [ \
                    # from Ben Rackham
                    'Ross 619', \
                    # from Prajwal
                    #'GJ 504', \
                    # NGTS target from Max
                    #'TIC 125731343', \
                    # second TRAPPIST-1 candidate from Max
                    #'TIC 233965332', \
                   ]
    for strgmast in liststrgmast:
        
        if strgmast == 'TIC 125731343':
            dictlygoinpt['limttimeignoqual'] = [2458609., 2458611.]
        else:
            dictlygoinpt = None
        
        miletos.main.init( \
                          strgmast=strgmast, \
                          dictlygoinpt=dictlygoinpt, \
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


def cnfg_WASP121():
    
    dictlcurtessinpt = dict()
    dictlcurtessinpt['listtsecsele'] = [7, 33, 34]
    miletos.main.init( \
                   strgmast='WASP-121', \
                   #typedataspoc='SAP', \
                   #boolplotpopl=True, \
                   #boolinfefoldbind=True, \
                   listtypemodl=['psyspcur'], \
                   typepriocomp='pdim', \
                   dictlcurtessinpt=dictlcurtessinpt, \
                  )


def cnfg_WASP12():

    miletos.main.init( \
                   strgmast='WASP-12', \
                   boolplotpopl=True, \
                   boolinfefoldbind=True, \
                   listtypemodl=['psysttvr'], \
                  )


def cnfg_WD_GI():
    
    pathoutp = os.environ['MILETOS_DATA_PATH'] + '/data/WD/'
    os.system('mkdir -p %s' % pathoutp)
    listisec = np.arange(27, 28)
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
                 typepriocomp='pdim', \
                 #datatype='pand', \
                )


def cnfg_WD_candidates():
    
    listticitarg = [686072378]
    for ticitarg in listticitarg:
        miletos.main.init( \
             ticitarg=ticitarg, \
            )


def cnfg_V563Lyr():
    
    miletos.main.init( \
         strgmast='V563 Lyr', \
         typepriocomp='pdim', \
         massstar=1., \
         datatype='pand', \
         strgtarg='V563Lyr', \
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
                   liststrginst=['ASTERIA'], \
                   
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


def cnfg_WASP18():
    
    miletos.main.init( \
                   pcurtype='sinu', \
                   toiitarg=185, \
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
         strgtarg='GJ299', \
         labltarg='GJ 299', \
         strgmast='GJ 299', \
         epocpmot=2019.3, \
         radistar=radistar, \
         massstar=massstar, \
        )


def cnfg_bhol():
    
    listticitarg = [281562429]
    typemodl = 'bhol'
    for ticitarg in listticitarg:
        miletos.main.init( \
             ticitarg=ticitarg, \
             typemodl=typemodl, \
            )


def cnfg_Michelle():
    
    miletos.main.init( \
         ticitarg=126803899, \
        )

    
def cnfg_Rafael():
    
    miletos.main.init( \
         ticitarg=356069146, \
         strgclus='Rafael', \
        )


def cnfg_KOI1003():
    
    factmsmj = 1048.
    factrsrj = 9.95
    miletos.main.init( \
         ticitarg=122374527, \
         strgtarg='KOI1003', \
         labltarg='KOI 1003', \
         radistar=2.445*factmsmj, \
         massstar=1.343*factrsrj, \
         strgmast='KOI-1003', \
         epocpmot=2019.3, \
        )


def cnfg_HD118203():
    
    strgtarg = 'HD118203'
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
         strgtarg=strgtarg, \
         labltarg=labltarg, \
         strgmast=strgmast, \
         epocprio=epocprio, \
         periprio=periprio, \
         duraprio=duraprio, \
         rratprio=rratprio, \
         listlimttimemask=listlimttimemask, \
        )


def cnfg_TOI1233():
    
    toiitarg = 1233
    miletos.main.init( \
                      toiitarg=toiitarg, \
                      boolplotpopl=True, \
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


def target(strgmast):
    '''
    Execute miletos on a target from command line.
    '''
    
    miletos.main.init( \
         strgmast=strgmast, \
        )



def cnfg_HATP19():
    
    ticitarg = 267650535
    strgmast = 'HAT-P-19'
    labltarg = 'HAT-P-19'
    strgtarg = 'hatp0019'
    
    miletos.main.init( \
         strgtarg=strgtarg, \
         labltarg=labltarg, \
         strgmast=strgmast, \
         ticitarg=ticitarg, \
        )


def cnfg_WD1856():

    #strgtarg = 'WD1856+534'
    ticitarg = 267574918
    #strgmast = 'TIC 267574918'
    #labltarg = 'WD-1856'
    
    print('HACKING! MAKING UP THE STAR RADIUS')
    miletos.main.init( \
         #strgtarg=strgtarg, \
         #labltarg=labltarg, \
         #strgmast=strgmast, \
         ticitarg=ticitarg, \
         #boolmakeanim=True, \
         #makeprioplot=False, \

         #infetype='trap', \
         #dilucorr=0.01, \
         #jmag=15.677, \
         #contrati=10, \
         # temp
         #datatype='sapp', \
         radistar=1.4/11.2, \
         stdvradistar=0, \
         stdvmassstar=0, \
         stdvtmptstar=0, \
         massstar=0.518*1047, \
         tmptstar=4710., \
        )


def cnfg_cont():

    strgtarg = 'cont'
    ticitarg = 1717706276
    strgmast = 'TIC 1717706276'
    labltarg = 'Cont'
    
    miletos.main.init( \
         strgtarg=strgtarg, \
         labltarg=labltarg, \
         strgmast=strgmast, \
         ticitarg=ticitarg, \
         boolmakeanim=True, \
         contrati=10, \
        )


def cnfg_lindsey():
    
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    pathbase = os.environ['PERGAMON_DATA_PATH'] + '/featsupntess/'
    pathdata = pathbase + 'data/'
    pathimag = pathbase + 'imag/'
    os.system('mkdir -p %s' % pathdata)
    os.system('mkdir -p %s' % pathimag)

    listtypemodl = ['supn']

    for strgclus in [ \
                     'Cycle1-matched', \
                     'Cycle2-matched', \
                     'Cycle3-matched', \
                    ]:

        pathcsvv = pathdata + '%s.csv' % strgclus
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
            dictlygoinpt['boolplotrflx'] = True
            dictlygoinpt['boolplotcntp'] = True
            dictlygoinpt['numbside'] = 5
            dictlygoinpt['booldetrcbvs'] = False
            #dictlygoinpt['boolplotoffs'] = True
            
            dictmileoutp = miletos.main.init( \
                                             rasctarg=rasctarg, \
                                             decltarg=decltarg, \
                                             labltarg=labltarg, \
                                             strgclus=strgclus, \
                                             dictlygoinpt=dictlygoinpt, \
                                             listtypemodl=listtypemodl, \
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
    13 July 2021, AGN from DJ
    '''
    rasctarg = 63.260208 
    decltarg = -53.0727

    listtypemodl = ['supn']
    
    labltarg = 'ASASSN-20qc'

    dictlygoinpt = dict()
    dictlygoinpt['boolplotrflx'] = True
    dictlygoinpt['boolplotcntp'] = True
    dictlygoinpt['boolplotoffs'] = True
    
    dictlcurtessinpt = dict()
    dictlcurtessinpt['boolffimonly'] = True

    miletos.main.init( \
                      labltarg=labltarg, \
                      rasctarg=rasctarg, \
                      decltarg=decltarg, \

                      listtypemodl=listtypemodl, \
                      
                      dictlcurtessinpt=dictlcurtessinpt, \

                      dictlygoinpt=dictlygoinpt, \
                     )


def cnfg_ASASSN20qc_depr():
    '''
    13 July 2021, AGN from DJ
    '''
    
    rasctarg = 63.260208 
    decltarg = -53.0727

    labltarg = 'ASASSN-20qc'
    
    refrlistlabltser = [['Michael']]
    path = os.environ['LYGOS_DATA_PATH'] + '/data/lc_2020adgm_cleaned_ASASSN20qc'
    print(path)
    objtfile = open(path, 'r')
    k = 0
    linevalu = []
    for line in objtfile:
        if k == 0:
            k += 1
            continue
        linesplt = line.split(' ')
        linevalu.append([])
        for linesplttemp in linesplt:
            if linesplttemp != '':
                linevalu[k-1].append(float(linesplttemp))
        linevalu[k-1] = np.array(linevalu[k-1])
        k += 1
    linevalu = np.vstack(linevalu)
    refrarrytser = np.empty((linevalu.shape[0], 3))
    refrarrytser[:, 0] = linevalu[:, 0]
    refrarrytser[:, 1] = linevalu[:, 2]
    refrarrytser[:, 2] = linevalu[:, 3]
   
    dictmileinpt = dict()
    dictmileinpt['listtypemodl'] = ['supn']
    
    listnumbside = [7, 11, 15]
    #dictmileinpt['listlimttimemask'] = [[[[-np.inf, 2457000 + 2175], [2457000 + 2186.5, 2457000 + 2187.5]]]]
    dictmileinpt['listlimttimemask'] = [[[[2457000 + 2186.5, 2457000 + 2187.5]]]]
    for numbside in listnumbside:
        if numbside == 11:
            dictmileinpt['listtimescalbdtrspln'] = [0., 0.1, 0.5]
            boolfittoffs = True
        else:
            dictmileinpt['listtimescalbdtrspln'] = [0.]
            boolfittoffs = False

        lygos.init( \
                   boolplotrflx=True, \
                   boolplotcntp=True, \
                   boolfittoffs=boolfittoffs, \
                
                   refrlistlabltser=refrlistlabltser, \
                   refrarrytser=refrarrytser, \

                   labltarg=labltarg, \
                   
                   listtsecsele=[32], \
                   dictmileinpt=dictmileinpt, \
                   
                   timeoffs=2459000, \

                   numbside=numbside, \

                   rasctarg=rasctarg, \
                   decltarg=decltarg, \
                   
                   boolregrforc=True, \
                   boolplotforc=True, \
                  )


globals().get(sys.argv[1])(*sys.argv[2:])
