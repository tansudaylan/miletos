import numpy as np

import miletos

'''
Estimate the visibility of a target on the sky from a given observatory, for a given night and across a given year.

The script uses miletos.
'''

# inputs
liststrgmast = ['TOI-1233']
typeobvt = 'TUG'
strgtimeobvtnigh = '2022-07-13 00:00:00'
strgtimeobvtyear = '2022-01-01 00:00:00'

numbtarg = len(liststrgmast)
indxtarg = np.arange(numbtarg)

if typeobvt == 'LCO':
    # Las Campanas Observatory (LCO), Chile
    latiobvt = -29.01418
    longobvt = -70.69239
    heigobvt = 2515.819
    offstimeobvt = 0. # give the times in UT

if typeobvt == 'TUG':
    # TUBITAK National Observatory (TUG), Turkey
    longobvt = 30.335555
    latiobvt = 36.824166
    heigobvt = 2500.
    offstimeobvt = 3.

for n in indxtarg:
   
    miletos.main.init( \
                      # provide the keyword for the target
                      strgmast=liststrgmast[n], \
                      
                      # local time offset with respect to UTC
                      offstimeobvt=offstimeobvt, \

                      # latitude of the observatory
                      latiobvt=latiobvt, \
                      
                      # longtiude of the observatory
                      longobvt=longobvt, \
                      
                      # altitude of the observatory
                      heigobvt=heigobvt, \
                      
                      # a string indicating the midnight during the observation night
                      strgtimeobvtnigh=strgtimeobvtnigh, \
                      
                      # a string indicating the midnight in the beginning of the observation year
                      strgtimeobvtyear=strgtimeobvtyear, \
                      
                      # turn off time-domain data processing
                      booltserdata=False, \
                      
                      # turn on visibity estimation
                      boolcalcvisi=True, \
                      
                      # turn on visibity plotting
                      boolplotvisi=True, \
                     )

