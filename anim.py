import pexo.main
import os
pathbase = os.environ['PEXO_DATA_PATH'] + '/imag/'
radiplan = [1.6, 2.1, 2.7, 3.1]
rsma = [0.0895, 0.0647, 0.0375, 0.03043]
epoc = [2458572.1128, 2458572.3949, 2458571.3368, 2458586.5677]
peri = [3.8, 6.2, 14.2, 19.6]
cosi = [0., 0., 0., 0.]
radistar = 0.9

booldark = True

boolsingside = False
boolanim = True
              ## 'realblac': dark background, black planet
              ## 'realblaclcur': dark backgound, luminous planet, 
              ## 'realillu': dark background, illuminated planet, 
              ## 'cart': cartoon, 'realdark' 
listtypevisu = ['realblac', 'realblaclcur', 'realillu', 'cart', 'cartmerc']
listtypevisu = ['realblaclcur']
path = pathbase + 'orbt'
for typevisu in listtypevisu:

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
                        typefileplot='png', \
                       )
    


