import pexo.main
import os
pathbase = os.environ['PEXO_DATA_PATH'] + '/imag/'
radistar = 0.9

booldark = True

boolsingside = False
boolanim = True
              ## 'realblac': dark background, black planet
              ## 'realblaclcur': dark backgound, black planet, light curve
              ## 'realcolrlcur': dark backgound, colored planet, light curve
              ## 'cartcolr': cartoon backgound
listtypevisu = ['realblac', 'realblaclcur', 'realcolrlcur', 'cartcolr']
listtypevisu = ['realblac', 'cartcolr']
path = pathbase + 'orbt'

for a in range(2):

    radiplan = [1.6, 2.1, 2.7, 3.1]
    rsma = [0.0895, 0.0647, 0.0375, 0.03043]
    epoc = [2458572.1128, 2458572.3949, 2458571.3368, 2458586.5677]
    peri = [3.8, 6.2, 14.2, 19.6]
    cosi = [0., 0., 0., 0.]
    
    if a == 1:
        radiplan += [2.0]
        rsma += [0.88 / (215. * 0.1758)]
        epoc += [2458793.2786]
        peri += [29.54115]
        cosi += [0.]
    
    for typevisu in listtypevisu:
        
        if a == 0:
            continue

        pexo.main.plot_orbt( \
                            path, \
                            radiplan, \
                            rsma, \
                            epoc, \
                            peri, \
                            cosi, \
                            typevisu, \
                            radistar=radistar, \
                            boolsingside=boolsingside, \
                            boolanim=boolanim, \
                            #typefileplot='png', \
                           )
    


