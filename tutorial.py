import miletos
import sys
import os
import numpy as np
import wget
import pandas as pd
from tdpy.util import summgene


def cnfg_multis():

    for strgmast in ['TOI-270', 'TOI-700', 'TOI-1233', 'TOI-1339']:
        miletos.main.init( \
                       strgmast=strgmast, \
                      )


def cnfg_TICsFromGV():
    
    listtici = [ \
                # 30 September 2021, small star
                1400704733, \
               ]

    for tici in listtici:
        miletos.main.init( \
                          ticitarg=tici, \
                         )


def cnfg_requests():
    
    liststrgmast = [ \
                    # from Ben Rackham
                    'Ross 619', \
                    # from Prajwal
                    'GJ 504', \
                    # NGTS target from Max
                    'TIC 125731343', \
                    # second TRAPPIST-1 candidate from Max
                    'TIC 233965332', \
                   ]
    for strgmast in liststrgmast:
        
        if strgmast == 'TIC 125731343':
            limttimeignoquallygo = [2458609., 2458611.]
        else:
            limttimeignoquallygo = None
        
        miletos.main.init( \
                          strgmast=strgmast, \
                          limttimeignoquallygo=limttimeignoquallygo, \
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
            labltarg = 'TOI2406 Neigbor TIC212957637'
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
                       #listtsecsele=[30], \
                       #listlimttimemask=[[[[2458370, 2458385]]], []], \
                      )


def cnfg_WASP121():

    miletos.main.init( \
                   strgmast='WASP-121', \
                   #typedataspoc='SAP', \
                   boolplotprio=False, \
                   boolallepcur=True, \
                   boolmodl=True, \
                   #boolinfefoldbind=True, \
                   listtypemodlexop=['0003'], \
                   timescalbdtrspln=2.5, \
                   typeprioplan='blsq', \
                   typedatatess='SPOC', \
                   listtsecsele=[7, 33, 34], \
                  )


def cnfg_WASP12():

    miletos.main.init( \
                   strgmast='WASP-12', \
                   boolplotprio=False, \
                   boolallepcur=True, \
                   boolmodl=True, \
                   boolinfefoldbind=True, \
                   listtypemodlexop=['0003'], \
                   timescalbdtrspln=2.5, \
                   typedatatess='SPOC', \
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
                 typeprioplan='blsq', \
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
         typeprioplan='blsq', \
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
                   typeprioplan='inpt', \
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
         datatype='pandora', \
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
                   maxmnumbplanblsq=4, \
                   typeprioplan='blsq', \
                  )


def cnfg_NGTS11():
    
    # TOI 1847
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
    
    print(strgmast)

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
         #maxmnumbstarpandora=40, \
         #makeprioplot=False, \
         booldatatser=False, \

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
         #boolblsq=False, \
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
         #maxmnumbstarpandora=40, \
         contrati=10, \
         datatype='pandora', \
         boolblsq=False, \
        )


def cnfg_ASASSN20qc():
    '''
    13 July 2021, AGN from DJ
    '''
    rasctarg = 63.260208 
    decltarg = -53.0727

    listtypeobjt = []
    
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

               listtypeobjt=listtypeobjt, \
               
               dictlcurtessinpt=dictlcurtessinpt, \

               dictlygoinpt=dictlygoinpt, \
              )


globals().get(sys.argv[1])(*sys.argv[2:])
