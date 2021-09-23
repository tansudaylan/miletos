import time as modutime

import os, fnmatch
import json
import sys, datetime
import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.stats

from tqdm import tqdm

import astroquery
import astropy
import astropy.coordinates
import astropy.units

import celerite
from celerite import terms

import matplotlib as mpl
import matplotlib.pyplot as plt

#import seaborn as sns

import tdpy
from tdpy.util import summgene

import lygos

import ephesus

"""
Given a target, miletos is an time-domain astronomy tool that allows 
1) automatic search for, download and process TESS and Kepler data via MAST or use user-provided data
2) impose priors based on custom inputs, ExoFOP or NASA Exoplanet Archive
3) model radial velocity and photometric time-series data on planetary systems
4) Make characterization plots of the target after the analysis
"""


def retr_noisredd(time, logtsigm, logtrhoo):
    
    # set up a simple celerite model
    objtkern = celerite.terms.Matern32Term(logtsigm, logtrhoo)
    objtgpro = celerite.GP(objtkern)
    
    # simulate K datasets with N points
    objtgpro.compute(time)
    
    #y = objtgpro.sample(size=1)
    y = objtgpro.sample()
    
    return y[0]


def quer_mast(request):

    from urllib.parse import quote as urlencode
    import http.client as httplib 

    server='mast.stsci.edu'

    # Grab Python Version
    version = '.'.join(map(str, sys.version_info[:3]))

    # Create Http Header Variables
    headers = {'Content-type': 'application/x-www-form-urlencoded',
               'Accept': 'text/plain',
               'User-agent':'python-requests/'+version}

    # Encoding the request as a json string
    requestString = json.dumps(request)
    requestString = urlencode(requestString)

    # opening the https connection
    conn = httplib.HTTPSConnection(server)

    # Making the query
    conn.request('POST', '/api/v0/invoke', 'request='+requestString, headers)

    # Getting the response
    resp = conn.getresponse()
    head = resp.getheaders()
    content = resp.read().decode('utf-8')

    # Close the https connection
    conn.close()

    return head,content


def xmat_tici(listtici):
    
    if len(listtici) == 0:
        raise Exception('')
    
    # make sure the input is a python list of strings
    if isinstance(listtici[0], str):
        if isinstance(listtici, np.ndarray):
            listtici = list(listtici)
    else:
        if isinstance(listtici, list):
            listtici = np.array(listtici)
        if isinstance(listtici, np.ndarray):
            listtici = listtici.astype(str)
        listtici = list(listtici)

    request = {'service':'Mast.Catalogs.Filtered.Tic', 'format':'json', 'params':{'columns':'rad, mass', \
                                                                        'filters':[{'paramName':'ID', 'values':listtici}]}}
    headers, outString = quer_mast(request)
    dictquer = json.loads(outString)['data']
    
    return dictquer


def retr_dictcatltic8(typepopl, typeverb=1):
    """
    Get a dictionary of the sources in the TIC8 with the fields in the TIC8
    
    Keyword arguments   
        typepopl: type of the population
            'ffimhcon': TESS targets with contamination larger than
            'ffimm135': TESS targets brighter than mag 13.5
            'tessnomi2min': 2-minute TESS targets obtained by merging the SPOC 2-min bulk downloads

    Returns a dictionary with keys:
        rasc: RA
        decl: declination
        tmag: TESS magnitude
        radistar: radius of the star
        massstar: mass of the star
    """
    
    if typeverb > 0:
        print('Retrieving a dictionary of TIC8 for population %s...' % typepopl)
    
    if typepopl.startswith('tess'):
        if typepopl[4:].startswith('nomi'):
            listtsec = np.arange(1, 27)
        elif typepopl[4:].endswith('extd'):
            listtsec = np.arange(27, 39)
        else:
            listtsec = [int(typepopl[-2:])]
        numbtsec = len(listtsec)
        indxtsec = np.arange(numbtsec)

    pathlistticidata = os.environ['MILETOS_DATA_PATH'] + '/data/listticidata/'
    os.system('mkdir -p %s' % pathlistticidata)

    path = pathlistticidata + 'listticidata_%s.csv' % typepopl
    if not os.path.exists(path):
        
        # dictionary of strings that will be keys of the output dictionary
        dictstrg = dict()
        dictstrg['ID'] = 'tici'
        dictstrg['ra'] = 'rasc'
        dictstrg['dec'] = 'decl'
        dictstrg['Tmag'] = 'tmag'
        dictstrg['rad'] = 'radistar'
        dictstrg['mass'] = 'massstar'
        dictstrg['Teff'] = 'tmptstar'
        dictstrg['logg'] = 'loggstar'
        dictstrg['MH'] = 'metastar'
        liststrg = list(dictstrg.keys())
        
        print('typepopl')
        print(typepopl)
        if typepopl.startswith('tessnomi'):
            
            if typepopl[8:12] == '20sc':
                strgurll = '_20s_'
                labltemp = '20-second'
            if typepopl[8:12] == '2min':
                strgurll = '_'
                labltemp = '2-minute'

            dictquer = dict()
            listtici = []
            for o in indxtsec:
                if typepopl.endswith('bulk'):
                    pathtess = os.environ['TESS_DATA_PATH'] + '/data/lcur/sector-%02d' % listtsec[o]
                    listnamefile = fnmatch.filter(os.listdir(pathtess), '*.fits')
                    listticitsec = []
                    for namefile in listnamefile:
                        listticitsec.append(str(int(namefile.split('-')[2])))
                    listticitsec = np.array(listticitsec)
                else:
                    url = 'https://tess.mit.edu/wp-content/uploads/all_targets%sS%03d_v1.csv' % (strgurll, listtsec[o])
                    c = pd.read_csv(url, header=5)
                    listticitsec = c['TICID'].values
                    listticitsec = listticitsec.astype(str)
                numbtargtsec = listticitsec.size
                
                if typeverb > 0:
                    print('%d observed %s targets in Sector %d...' % (numbtargtsec, labltemp, listtsec[o]))
                
                if numbtargtsec > 0:
                    dictquertemp = xmat_tici(listticitsec)
                
                if o == 0:
                    dictquerinte = dict()
                    for name in dictstrg.keys():
                        dictquerinte[dictstrg[name]] = [[] for o in indxtsec]
                
                for name in dictstrg.keys():
                    for k in range(len(dictquertemp)):
                        dictquerinte[dictstrg[name]][o].append(dictquertemp[k][name])

            print('Concatenating arrays from different sectors...')
            for name in dictstrg.keys():
                dictquer[dictstrg[name]] = np.concatenate(dictquerinte[dictstrg[name]])
            
            u, indxuniq = np.unique(dictquer['tici'], return_index=True)
            for name in dictstrg.keys():
                dictquer[dictstrg[name]] = dictquer[dictstrg[name]][indxuniq]

            numbtarg = dictquer['radistar'].size
            if typeverb > 0:
                print('%d observed 2-min targets...' % numbtarg)
            
        elif typepopl.startswith('ffim'):
            if typepopl == 'ffimhcon':
                request = {'service':'Mast.Catalogs.Filtered.Tic.Rows', 'format':'json', 'params':{ \
                #"columns":"c.*", \
                'columns':'rad, mass, ID, contratio', \
                                                             'filters':[{'paramName':'contratio', 'values':[{"min":10., "max":1e3}]}]}}
            if typepopl == 'ffimm135':
                request = {'service':'Mast.Catalogs.Filtered.Tic.Rows', 'format':'json', 'params':{ \
                #"columns":"c.*", \
                'columns':'rad, mass, ID', \
                                                             'filters':[{'paramName':'Tmag', 'values':[{"min":5., "max":8.5}]}]}}
            headers, outString = quer_mast(request)
            listdictquer = json.loads(outString)['data']
            if typeverb > 0:
                print('%d matches...' % len(listdictquer))
            dictquer = dict()
            for name in listdictquer[0].keys():
                dictquer[name] = np.empty(len(listdictquer))
                for k in range(len(listdictquer)):
                    dictquer[name][k] = listdictquer[k][name]
        else:
            print('Unrecognized population name: %s' % typepopl)
            raise Exception('')
        
        if typeverb > 0:
            #print('%d targets...' % numbtarg)
            print('Writing to %s...' % path)
        pd.DataFrame.from_dict(dictquer).to_csv(path)
    else:
        if typeverb > 0:
            print('Reading from %s...' % path)
        dictquer = pd.read_csv(path).to_dict(orient='list')
        
        for name in dictquer.keys():
            dictquer[name] = np.array(dictquer[name])
        del dictquer['Unnamed: 0']

    return dictquer


def retr_listcolrcomp(numbcomp):
    
    listcolrcomp = np.array(['magenta', 'orange', 'red', 'green', 'purple', 'cyan'])[:numbcomp]

    return listcolrcomp


def retr_liststrgcomp(numbcomp):
    
    liststrgcomp = np.array(['b', 'c', 'd', 'e', 'f', 'g'])[:numbcomp]

    return liststrgcomp


def retr_llik_mile(para, gdat):
    
    """
    Returns the likelihood
    """
    
    if gdat.typemodl == 'comptran' or gdat.typemodl == 'complens':
        radistar = para[0]
        peri = para[1]
        masscomp = para[2]
        massstar = para[3]

        rflxtotl, dflxelli, dflxbeam, dflxslen = ephesus.retr_rflxmodl(gdat.time, para)
    
    if gdat.typemodl == 'rise':
        timerise = para[0]
        coeflinerise = para[1]
        coefquadrise = para[2]
        coefline = para[3]
        rflxmodl, rflxline, rflxrise = ephesus.retr_rflxmodlrise(gdat.time, timerise, coeflinerise, coefquadrise, coefline)
    
    llik = np.sum(-0.5 * (gdat.rflx - rflxmodl)**2 / gdat.varirflx)

    return llik


def retr_llik_albbepsi(para, gdat):
    
    # Bond albedo
    albb = para[0]
    
    # heat recirculation efficiency
    epsi = para[2]

    psiimodl = (1 - albb)**.25
    #tmptirre = gdat.dictlist['tmptequi'][:, 0] * psiimodl
    tmptirre = gdat.gmeatmptequi * psiimodl
    tmptdayy = tmptirre * (2. / 3. - 5. / 12. * epsi)**.25
    tmptnigh = tmptirre * (epsi / 4.)**.25
    
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


def retr_dilu(tmpttarg, tmptcomp, strgwlentype='tess'):
    
    if strgwlentype != 'tess':
        raise Exception('')
    else:
        binswlen = np.linspace(0.6, 1.)
    meanwlen = (binswlen[1:] + binswlen[:-1]) / 2.
    diffwlen = (binswlen[1:] - binswlen[:-1]) / 2.
    
    fluxtarg = tdpy.retr_specbbod(tmpttarg, meanwlen)
    fluxtarg = np.sum(diffwlen * fluxtarg)
    
    fluxcomp = tdpy.retr_specbbod(tmptcomp, meanwlen)
    fluxcomp = np.sum(diffwlen * fluxcomp)
    
    dilu = 1. - fluxtarg / (fluxtarg + fluxcomp)
    
    return dilu


def retr_modl_spec(gdat, tmpt, booltess=False, strgtype='intg'):
    
    if booltess:
        thpt = scipy.interpolate.interp1d(gdat.meanwlenband, gdat.thptband)(wlen)
    else:
        thpt = 1.
    
    if strgtype == 'intg':
        spec = tdpy.retr_specbbod(tmpt, gdat.meanwlen)
        spec = np.sum(gdat.diffwlen * spec)
    if strgtype == 'diff' or strgtype == 'logt':
        spec = tdpy.retr_specbbod(tmpt, gdat.cntrwlen)
        if strgtype == 'logt':
            spec *= gdat.cntrwlen
    
    return spec


def retr_llik_spec(para, gdat):
    
    tmpt = para[0]
    
    #timeinit = time.time()
    
    specboloplan = retr_modl_spec(gdat, tmpt, booltess=False, strgtype='intg')
    deptplan = 1e3 * gdat.rratmedi[0]**2 * specboloplan / gdat.specstarintg # [ppt]
    
    llik = -0.5 * np.sum((deptplan - gdat.deptobsd)**2 / gdat.varideptobsd)
    
    #timedelt = timeinit - time.time()
    
    #print('timedelt')
    #print(timedelt)
    #print('llik')
    #print(llik)
    
    return llik


def retr_reso(listperi, maxmordr=10):
    
    if np.where(listperi == 0)[0].size > 0:
        raise Exception('')

    numbsamp = listperi.shape[0]
    numbcomp = listperi.shape[1]
    indxcomp = np.arange(numbcomp)
    listratiperi = np.zeros((numbsamp, numbcomp, numbcomp))
    intgreso = np.zeros((numbcomp, numbcomp, 2))
    for j in indxcomp:
        for jj in indxcomp:
            if j >= jj:
                continue
                
            rati = listperi[:, j] / listperi[:, jj]
            #print('listperi')
            #print(listperi)
            #print('rati')
            #print(rati)
            if rati < 1:
                listratiperi[:, j, jj] = 1. / rati
            else:
                listratiperi[:, j, jj] = rati

            minmdiff = 1e100
            for a in range(1, maxmordr):
                for aa in range(1, maxmordr):
                    diff = abs(float(a) / aa - listratiperi[:, j, jj])
                    if np.mean(diff) < minmdiff:
                        minmdiff = np.mean(diff)
                        minmreso = a, aa
            intgreso[j, jj, :] = minmreso
            #print('minmdiff') 
            #print(minmdiff)
            #print('minmreso')
            #print(minmreso)
            #print
    
    return intgreso, listratiperi


def writ_filealle(gdat, namefile, pathalle, dictalle, dictalledefa, typeverb=1):
    
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
    if typeverb > 0:
        print('Writing to %s...' % pathfile)
    objtfile = open(pathfile, 'w')
    for line in listline:
        objtfile.write('%s' % line)
    objtfile.close()


def retr_dictcatlrvel():
    
    if typeverb > 0:
        print('Reading Sauls Gaia high RV catalog...')
    path = os.environ['TROIA_DATA_PATH'] + '/data/Gaia_high_RV_errors.txt'
    for line in open(path):
        listnamesaul = line[:-1].split('\t')
        break
    if typeverb > 0:
        print('Reading from %s...' % path)
    data = np.loadtxt(path, skiprows=1)
    dictcatl = dict()
    dictcatl['rasc'] = data[:, 0]
    dictcatl['decl'] = data[:, 1]
    dictcatl['stdvrvel'] = data[:, -4]
    
    return dictcatl


def retr_dictexof(toiitarg=None, boolreplexar=False, typeverb=1):
    
    dictfact = ephesus.retr_factconv()
    
    pathlygo = os.environ['LYGOS_DATA_PATH'] + '/'
    pathexof = pathlygo + 'data/exofop_tess_tois.csv'
    if typeverb > 0:
        print('Reading from %s...' % pathexof)
    objtexof = pd.read_csv(pathexof, skiprows=0)
    
    dictexof = {}
    dictexof['toii'] = objtexof['TOI'].values
    numbcomp = dictexof['toii'].size
    indxcomp = np.arange(numbcomp)
    toiitargexof = np.empty(numbcomp, dtype=object)
    for k in indxcomp:
        toiitargexof[k] = int(dictexof['toii'][k])
        
    if toiitarg is None:
        indxcomp = np.arange(numbcomp)
    else:
        indxcomp = np.where(toiitargexof == toiitarg)[0]
    
    dictexof['toii'] = dictexof['toii'][indxcomp]
    
    numbcomp = indxcomp.size
    
    if indxcomp.size == 0:
        if typeverb > 0:
            print('The host name, %s, was not found in the ExoFOP TOI Catalog.' % toiitarg)
        return None
    else:
        dictexof['namesyst'] = np.empty(numbcomp, dtype=object)
        dictexof['nameplan'] = np.empty(numbcomp, dtype=object)
        for kk, k in enumerate(indxcomp):
            dictexof['nameplan'][kk] = 'TOI ' + str(dictexof['toii'][kk])
            dictexof['namesyst'][kk] = 'TOI ' + str(dictexof['toii'][kk])[:-3]
        
        dictexof['dept'] = objtexof['Depth (ppm)'].values[indxcomp] * 1e-3 # [ppt]
        dictexof['rrat'] = np.sqrt(dictexof['dept'] * 1e-3)
        dictexof['radiplan'] = objtexof['Planet Radius (R_Earth)'][indxcomp].values
        dictexof['stdvradiplan'] = objtexof['Planet Radius (R_Earth) err'][indxcomp].values
        
        dictexof['rascstar'] = objtexof['RA'][indxcomp].values
        dictexof['declstar'] = objtexof['Dec'][indxcomp].values
        
        dictexof['strgcomm'] = np.empty(numbcomp, dtype=object)
        dictexof['strgcomm'][:] = objtexof['Comments'][indxcomp].values
        
        dictexof['facidisc'] = np.empty(numbcomp, dtype=object)
        dictexof['facidisc'][:] = 'Transiting Exoplanet Survey Satellite (TESS)'
        
        dictexof['peri'] = objtexof['Period (days)'][indxcomp].values
        dictexof['epoc'] = objtexof['Epoch (BJD)'][indxcomp].values
        dictexof['duratran'] = objtexof['Duration (hours)'].values[indxcomp] # [hours]

        dictexof['boolfrst'] = np.zeros(numbcomp)
        dictexof['numbcompstar'] = np.zeros(numbcomp)
        
        liststrgfeatstartici = ['massstar', 'vmagsyst', 'jmagsyst', 'hmagsyst', 'kmagsyst', 'distsyst', 'metastar', 'radistar', 'tmptstar', 'loggstar']
        liststrgfeatstarticiinhe = ['mass', 'Vmag', 'Jmag', 'Hmag', 'Kmag', 'd', 'MH', 'rad', 'Teff', 'logg']
        
        numbstrgfeatstartici = len(liststrgfeatstartici)
        indxstrgfeatstartici = np.arange(numbstrgfeatstartici)

        for strgfeat in liststrgfeatstartici:
            dictexof[strgfeat] = np.zeros(numbcomp)
            dictexof['stdv' + strgfeat] = np.zeros(numbcomp)
        
        ## crossmatch with TIC
        dictexof['tici'] = objtexof['TIC ID'][indxcomp].values
        listticiuniq = np.unique(dictexof['tici'].astype(str))
        request = {'service':'Mast.Catalogs.Filtered.Tic', 'format':'json', 'params':{'columns':"*", \
                                                              'filters':[{'paramName':'ID', 'values':list(listticiuniq)}]}}
        headers, outString = quer_mast(request)
        listdictquer = json.loads(outString)['data']
        
        # get magnitudes from crossmatches
        for k in range(len(listdictquer)):
         
            indxtemp = np.where(dictexof['tici'] == listdictquer[k]['ID'])[0]
            if indxtemp.size == 0:
                raise Exception('')
        
            for n in indxstrgfeatstartici:
                dictexof[liststrgfeatstartici[n]][indxtemp] = listdictquer[k][liststrgfeatstarticiinhe[n]]
                dictexof['stdv' + liststrgfeatstartici[n]][indxtemp] = listdictquer[k]['e_' + liststrgfeatstarticiinhe[n]]
        
        dictexof['boolfpos'] = objtexof['TFOPWG Disposition'][indxcomp].values == 'FP'
        
        # augment
        dictexof['numbcompstar'] = np.empty(numbcomp)
        dictexof['boolfrst'] = np.zeros(numbcomp, dtype=bool)
        for kk, k in enumerate(indxcomp):
            indxcompthis = np.where(dictexof['namesyst'][kk] == dictexof['namesyst'])[0]
            if kk == indxcompthis[0]:
                dictexof['boolfrst'][kk] = True
            dictexof['numbcompstar'][kk] = indxcompthis.size
        
        dictexof['numbcomptranstar'] = dictexof['numbcompstar']
        dictexof['lumistar'] = dictexof['radistar']**2 * (dictexof['tmptstar'] / 5778.)**4
        dictexof['stdvlumistar'] = dictexof['lumistar'] * np.sqrt((2 * dictexof['stdvradistar'] / dictexof['radistar'])**2 + \
                                                                        (4 * dictexof['stdvtmptstar'] / dictexof['tmptstar'])**2)

        # mass from radii
        path = pathlygo + 'exofop_toi_mass_saved.csv'
        if not os.path.exists(path):
            dicttemp = dict()
            dicttemp['massplan'] = np.ones_like(dictexof['radiplan']) + np.nan
            dicttemp['stdvmassplan'] = np.ones_like(dictexof['radiplan']) + np.nan
            
            numbsamppopl = 10
            indx = np.where(np.isfinite(dictexof['radiplan']))[0]
            for n in tqdm(range(indx.size)):
                k = indx[n]
                meanvarb = dictexof['radiplan'][k]
                stdvvarb = dictexof['stdvradiplan'][k]
                
                # if radius uncertainty is not available, assume that it is small, so the mass uncertainty will be dominated by population uncertainty
                if not np.isfinite(stdvvarb):
                    stdvvarb = 1e-3 * meanvarb
                listradiplan = scipy.stats.truncnorm.rvs(-meanvarb/stdvvarb, np.inf, size=numbsamppopl) * stdvvarb + meanvarb
                listradiplan /= np.mean(listradiplan)
                listradiplan *= meanvarb
                listmassplan = ephesus.retr_massfromradi(listradiplan)
                
                print('listmassplan')
                summgene(listmassplan)
                
                dicttemp['massplan'][k] = np.mean(listmassplan)
                dicttemp['stdvmassplan'][k] = np.std(listmassplan)
            if typeverb > 0:
                print('Writing to %s...' % path)
            pd.DataFrame.from_dict(dicttemp).to_csv(path)
        else:
            if typeverb > 0:
                print('Reading from %s...' % path)
            dicttemp = pd.read_csv(path).to_dict(orient='list')
            
            for name in dicttemp:
                dicttemp[name] = np.array(dicttemp[name])
                if toiitarg is not None:
                    dicttemp[name] = dicttemp[name][indxcomp]

        print('dicttemp[massplan]')
        summgene(dicttemp['massplan'])
        dictexof['massplan'] = dicttemp['massplan']
        
        dictexof['stdvmassplan'] = dicttemp['stdvmassplan']
        
        dictexof['masstotl'] = dictexof['massstar'] + dictexof['massplan'] / dictfact['msme']
        dictexof['smax'] = ephesus.retr_smaxkepl(dictexof['peri'], dictexof['masstotl'])
        
        dictexof['inso'] = dictexof['lumistar'] / dictexof['smax']**2
        
        dictexof['tmptplan'] = dictexof['tmptstar'] * np.sqrt(dictexof['radistar'] / dictexof['smax'] / 2. / dictfact['aurs'])
        # temp check if factor of 2 is right
        dictexof['stdvtmptplan'] = np.sqrt((dictexof['stdvtmptstar'] / dictexof['tmptstar'])**2 + \
                                                        0.5 * (dictexof['stdvradistar'] / dictexof['radistar'])**2) / np.sqrt(2.)
        
        dictexof['densplan'] = dictexof['massplan'] / dictexof['radiplan']**3
        dictexof['booltran'] = np.ones_like(dictexof['toii'], dtype=bool)
    
        print('temp: vsiistar and projoblq are NaNs')
        dictexof['vsiistar'] = np.ones(numbcomp) + np.nan
        dictexof['projoblq'] = np.ones(numbcomp) + np.nan
        
        # replace confirmed planet properties
        if boolreplexar:
            dictexar = ephesus.retr_dictexar()
            listdisptess = objtexof['TESS Disposition'][indxcomp].values.astype(str)
            listdisptfop = objtexof['TFOPWG Disposition'][indxcomp].values.astype(str)
            indxexofcpla = np.where((listdisptfop == 'CP') & (listdisptess == 'PC'))[0]
            listticicpla = dictexof['tici'][indxexofcpla]
            numbticicpla = len(listticicpla)
            indxticicpla = np.arange(numbticicpla)
            for k in indxticicpla:
                indxexartici = np.where((dictexar['tici'] == int(listticicpla[k])) & \
                                                    (dictexar['facidisc'] == 'Transiting Exoplanet Survey Satellite (TESS)'))[0]
                indxexoftici = np.where(dictexof['tici'] == int(listticicpla[k]))[0]
                for strg in dictexar.keys():
                    if indxexartici.size > 0:
                        dictexof[strg] = np.delete(dictexof[strg], indxexoftici)
                    dictexof[strg] = np.concatenate((dictexof[strg], dictexar[strg][indxexartici]))
    
    return dictexof


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


def retr_objtlinefade(x, y, colr='black', initalph=1., alphfinl=0.):
    
    colr = get_color(colr)
    cdict = {'red':   ((0.,colr[0],colr[0]),(1.,colr[0],colr[0])),
             'green': ((0.,colr[1],colr[1]),(1.,colr[1],colr[1])),
             'blue':  ((0.,colr[2],colr[2]),(1.,colr[2],colr[2])),
             'alpha': ((0.,initalph, initalph), (1., alphfinl, alphfinl))}
    
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
    lc = mpl.collections.LineCollection(segments, cmap=individual_cm)
    lc.set_array(np.linspace(0.,1.,len(segments)))
    
    return lc


def plot_pser(gdat, strgarry, boolpost=False, typeverb=1):
    
    for b in gdat.indxdatatser:
        arrypcur = gdat.arrypcur[strgarry]
        arrypcurbindtotl = gdat.arrypcur[strgarry+'bindtotl']
        if strgarry.startswith('prim'):
            arrypcurbindzoom = gdat.arrypcur[strgarry+'bindzoom']
        # plot individual phase curves
        for p in gdat.indxinst[b]:
            for j in gdat.indxcomp:
                
                # phase on the horizontal axis
                figr, axis = plt.subplots(1, 1, figsize=gdat.figrsizeydob)
                if b == 0:
                    yerr = None
                if b == 1:
                    yerr = arrypcur[b][p][j][:, 2]
                axis.errorbar(arrypcur[b][p][j][:, 0], arrypcur[b][p][j][:, 1], yerr=yerr, elinewidth=1, capsize=2, zorder=1, \
                                                            color='grey', alpha=gdat.alphraww, marker='o', ls='', ms=1, rasterized=gdat.boolrastraww)
                if b == 0:
                    yerr = None
                if b == 1:
                    yerr = arrypcurbindzoom[b][p][j][:, 2]
                axis.errorbar(arrypcurbindtotl[b][p][j][:, 0], arrypcurbindtotl[b][p][j][:, 1], color=gdat.listcolrcomp[j], elinewidth=1, capsize=2, \
                                                                                                                 zorder=2, marker='o', ls='', ms=3)
                if gdat.boolwritplan:
                    axis.text(0.9, 0.9, r'\textbf{%s}' % gdat.liststrgcomp[j], \
                                        color=gdat.listcolrcomp[j], va='center', ha='center', transform=axis.transAxes)
                axis.set_ylabel(gdat.listlabltser[b])
                axis.set_xlabel('Phase')
                # overlay the posterior model
                if boolpost:
                    axis.plot(gdat.arrypcur[strgarry[:4]+'modltotl'+strgarry[-4:]][b][p][j][:, 0], gdat.arrypcur[strgarry[:4]+'modltotl'+strgarry[-4:]][b][p][j][:, 1], color='b', zorder=3)
                if gdat.listdeptdraw is not None:
                    for k in range(len(gdat.listdeptdraw)):  
                        axis.axhline(1. - 1e-3 * gdat.listdeptdraw[k], ls='-', color='grey')
                path = gdat.pathimagtarg + 'pcurphas_%s_%s_%s_%s_%s.%s' % (gdat.liststrginst[b][p], gdat.liststrgcomp[j], \
                                                                                            strgarry, gdat.strgtarg, gdat.typeprioplan, gdat.typefileplot)
                if typeverb > 0:
                    print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()
            
                if strgarry.startswith('prim'):
                    # time on the horizontal axis
                    figr, axis = plt.subplots(1, 1, figsize=gdat.figrsize)
                    if b == 0:
                        yerr = None
                    if b == 1:
                        yerr = arrypcur[b][p][j][:, 2]
                    axis.errorbar(gdat.periprio[j] * arrypcur[b][p][j][:, 0] * 24., arrypcur[b][p][j][:, 1], yerr=yerr, elinewidth=1, capsize=2, \
                                                        zorder=1, color='grey', alpha=gdat.alphraww, marker='o', ls='', ms=1, rasterized=gdat.boolrastraww)
                    if b == 0:
                        yerr = None
                    if b == 1:
                        yerr = arrypcurbindzoom[b][p][j][:, 2]
                    
                    if np.isfinite(gdat.duraprio[j]):
                        axis.errorbar(gdat.periprio[j] * arrypcurbindzoom[b][p][j][:, 0] * 24., arrypcurbindzoom[b][p][j][:, 1], zorder=2, \
                                                                                                    yerr=yerr, elinewidth=1, capsize=2, \
                                                                                                          color=gdat.listcolrcomp[j], marker='o', ls='', ms=3)
                    if boolpost:
                        axis.plot(gdat.periprio[j] * 24. * gdat.arrypcur[strgarry[:4]+'modltotl'+strgarry[-4:]][b][p][j][:, 0], \
                                                                                        gdat.arrypcur[strgarry[:4]+'modltotl'+strgarry[-4:]][b][p][j][:, 1], \
                                                                                                                        color='b', zorder=3)
                    if gdat.boolwritplan:
                        axis.text(0.9, 0.9, \
                                        r'\textbf{%s}' % gdat.liststrgcomp[j], color=gdat.listcolrcomp[j], va='center', ha='center', transform=axis.transAxes)
                    axis.set_ylabel(gdat.listlabltser[b])
                    axis.set_xlabel('Time [hours]')
                    if np.isfinite(gdat.duramask[j]):
                        axis.set_xlim([-np.nanmax(gdat.duramask), np.nanmax(gdat.duramask)])
                    if gdat.listdeptdraw is not None:
                        for k in range(len(gdat.listdeptdraw)):  
                            axis.axhline(1. - 1e-3 * gdat.listdeptdraw[k], ls='--', color='grey')
                    plt.subplots_adjust(hspace=0., bottom=0.25, left=0.25)
                    path = gdat.pathimagtarg + 'pcurtime_%s_%s_%s_%s_%s.%s' % (gdat.liststrginst[b][p], gdat.liststrgcomp[j], \
                                                                                    strgarry, gdat.strgtarg, gdat.typeprioplan, gdat.typefileplot)
                    if gdat.typeverb > 0:
                        print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()
            
            if gdat.numbcomp > 1:
                # plot all phase curves
                figr, axis = plt.subplots(gdat.numbcomp, 1, figsize=gdat.figrsizeydob, sharex=True)
                if gdat.numbcomp == 1:
                    axis = [axis]
                for jj, j in enumerate(gdat.indxcomp):
                    axis[jj].plot(arrypcur[b][p][j][:, 0], arrypcur[b][p][j][:, 1], color='grey', alpha=gdat.alphraww, \
                                                                                        marker='o', ls='', ms=1, rasterized=gdat.boolrastraww)
                    axis[jj].plot(arrypcurbindtotl[b][p][j][:, 0], arrypcurbindtotl[b][p][j][:, 1], color=gdat.listcolrcomp[j], marker='o', ls='', ms=1)
                    if gdat.boolwritplan:
                        axis[jj].text(0.97, 0.8, r'\textbf{%s}' % gdat.liststrgcomp[j], transform=axis[jj].transAxes, \
                                                                                            color=gdat.listcolrcomp[j], va='center', ha='center')
                axis[0].set_ylabel(gdat.listlabltser[b])
                axis[0].set_xlim(-0.5, 0.5)
                axis[0].yaxis.set_label_coords(-0.08, 1. - 0.5 * gdat.numbcomp)
                axis[gdat.numbcomp-1].set_xlabel('Phase')
                
                plt.subplots_adjust(hspace=0., bottom=0.2)
                path = gdat.pathimagtarg + 'pcurphastotl_%s_%s_%s_%s.%s' % (gdat.liststrginst[b][p], strgarry, \
                                                                                gdat.strgtarg, gdat.typeprioplan, gdat.typefileplot)
                if gdat.typeverb > 0:
                    print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()
    

def retr_albg(amplplanrefl, radiplan, smax):
    
    albg = amplplanrefl / (radiplan / smax)**2
    
    return albg


def calc_prop(gdat, strgpdfn):

    gdat.liststrgfeat = ['epoc', 'peri', 'rrat', 'rsma', 'cosi', 'ecos', 'esin', 'rvsa']
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
        gdat.dictlist[strgfeat] = np.empty((gdat.numbsamp, gdat.numbcomp))

        for j in gdat.indxcomp:
            if strgpdfn == 'prio':
                mean = getattr(gdat, strgfeat + 'prio')
                stdv = getattr(gdat, 'stdv' + strgfeat + 'prio')
                if not np.isfinite(mean[j]):
                    continue

                gdat.dictlist[strgfeat][:, j] = mean[j] + np.random.randn(gdat.numbsamp) * stdv[j]
                if strgfeat == 'rrat':
                    print('mean')
                    print(mean)
                    print('stdv')
                    print(stdv)
                    gdat.dictlist[strgfeat][:, j] = scipy.stats.truncnorm.rvs(-mean[j]/stdv[j], np.inf, size=gdat.numbsamp) * stdv[j] + mean[j]

            else:
                if strgfeat == 'epoc':
                    strg = '%s_epoch' % gdat.liststrgcomp[j]
                if strgfeat == 'peri':
                    strg = '%s_period' % gdat.liststrgcomp[j]
                if strgfeat == 'rrat':
                    strg = '%s_rr' % gdat.liststrgcomp[j]
                if strgfeat == 'rsma':
                    strg = '%s_rsuma' % gdat.liststrgcomp[j]
                if strgfeat == 'cosi':
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

    # allesfitter phase curve depths are in ppt
    for strgfeat in gdat.liststrgfeat:
        if strgfeat.startswith('ampl'):
            gdat.dictlist[strgfeat] *= 1e-3
    
    if gdat.typeverb > 0:
        print('Calculating derived variables...')
    # derived variables
    ## get samples from the star's variables

    # stellar properties
    for featstar in gdat.listfeatstar:
        meantemp = getattr(gdat, featstar)
        stdvtemp = getattr(gdat, 'stdv' + featstar)
        
        # not a specific host star
        if meantemp is None:
            continue

        if not np.isfinite(meantemp):
            print('featstar')
            print(featstar)
            print('meantemp')
            print(meantemp)
            print('stdvtemp')
            print(stdvtemp)
            raise Exception('')
        
        if stdvtemp == 0.:
            gdat.dictlist[featstar] = meantemp + np.zeros(gdat.numbsamp)
        else:
            gdat.dictlist[featstar] = scipy.stats.truncnorm.rvs(-meantemp/stdvtemp, np.inf, size=gdat.numbsamp) * stdvtemp + meantemp
        
        gdat.dictlist[featstar] = np.vstack([gdat.dictlist[featstar]] * gdat.numbcomp).T
    
    # inclination [degree]
    gdat.dictlist['incl'] = np.arccos(gdat.dictlist['cosi']) * 180. / np.pi
    
    # log g of the host star
    gdat.dictlist['loggstar'] = gdat.dictlist['massstar'] / gdat.dictlist['radistar']**2

    gdat.dictlist['ecce'] = gdat.dictlist['esin']**2 + gdat.dictlist['ecos']**2
    
    if 'exop' in gdat.listtypeobjt:
        # radius of the planets
        gdat.dictlist['radiplan'] = gdat.dictfact['rsre'] * gdat.dictlist['radistar'] * gdat.dictlist['rrat'] # [R_E]
    
        # semi-major axis
        gdat.dictlist['smax'] = (gdat.dictlist['radiplan'] + gdat.dictlist['radistar']) / gdat.dictlist['rsma']
    
        if strgpdfn == '0003' or strgpdfn == '0004':
            gdat.dictlist['amplnigh'] = gdat.dictlist['sbrtrati'] * gdat.dictlist['rrat']**2
        if strgpdfn == '0003':
            gdat.dictlist['phasshftplan'] = gdat.dictlist['timeshftplan'] * 360. / gdat.dictlist['peri']
        if strgpdfn == '0004':
            gdat.dictlist['phasshftplanther'] = gdat.dictlist['timeshftplanther'] * 360. / gdat.dictlist['peri']
            gdat.dictlist['phasshftplanrefl'] = gdat.dictlist['timeshftplanrefl'] * 360. / gdat.dictlist['peri']

        # planet equilibrium temperature
        gdat.dictlist['tmptplan'] = gdat.dictlist['tmptstar'] * np.sqrt(gdat.dictlist['radistar'] / 2. / gdat.dictlist['smax'])
        
        # stellar luminosity
        gdat.dictlist['lumistar'] = gdat.dictlist['radistar']**2 * (gdat.dictlist['tmptstar'] / 5778.)**4
        
        # insolation
        gdat.dictlist['inso'] = gdat.dictlist['lumistar'] / gdat.dictlist['smax']**2
    
        # predicted planet mass
        if gdat.typeverb > 0:
            print('Calculating predicted masses...')
        
        gdat.dictlist['massplanpredchen'] = np.empty_like(gdat.dictlist['radiplan'])
        gdat.dictlist['massplanpredwolf'] = np.empty_like(gdat.dictlist['radiplan'])
        for j in gdat.indxcomp:
            if not gdat.booltrancomp[j]:
                continue

            gdat.dictlist['massplanpredchen'][:, j] = ephesus.retr_massfromradi(gdat.dictlist['radiplan'][:, j])
            gdat.dictlist['massplanpredwolf'][:, j] = ephesus.retr_massfromradi(gdat.dictlist['radiplan'][:, j], strgtype='wolf2016')
        gdat.dictlist['massplanpred'] = gdat.dictlist['massplanpredchen']
        
        # mass used for later calculations
        gdat.dictlist['massplanused'] = np.empty_like(gdat.dictlist['massplanpredchen'])
        
        # temp
        gdat.dictlist['massplan'] = np.zeros_like(gdat.dictlist['esin'])
        gdat.dictlist['massplanused'] = gdat.dictlist['massplanpredchen']
        #for j in gdat.indxcomp:
        #    if 
        #        gdat.dictlist['massplanused'][:, j] = 
        #    else:
        #        gdat.dictlist['massplanused'][:, j] = 
    
        # density of the planet
        gdat.dictlist['densplan'] = gdat.dictlist['massplanused'] / gdat.dictlist['radiplan']**3

        # escape velocity
        gdat.dictlist['vesc'] = ephesus.retr_vesc(gdat.dictlist['massplanused'], gdat.dictlist['radiplan'])
        
        for j in gdat.indxcomp:
            strgratiperi = 'ratiperi_%s' % gdat.liststrgcomp[j]
            strgratiradi = 'ratiradi_%s' % gdat.liststrgcomp[j]
            for jj in gdat.indxcomp:
                gdat.dictlist[strgratiperi] = gdat.dictlist['peri'][:, j] / gdat.dictlist['peri'][:, jj]
                gdat.dictlist[strgratiradi] = gdat.dictlist['radiplan'][:, j] / gdat.dictlist['radiplan'][:, jj]
    
        gdat.dictlist['dept'] = 1e3 * gdat.dictlist['rrat']**2 # [ppt]
        # TSM
        gdat.dictlist['tsmm'] = ephesus.retr_tsmm(gdat.dictlist['radiplan'], gdat.dictlist['tmptplan'], \
                                                                                    gdat.dictlist['massplanused'], gdat.dictlist['radistar'], gdat.jmagsyst)
        
        # ESM
        gdat.dictlist['esmm'] = ephesus.retr_esmm(gdat.dictlist['tmptplan'], gdat.dictlist['tmptstar'], \
                                                                                    gdat.dictlist['radiplan'], gdat.dictlist['radistar'], gdat.kmagsyst)
        
    else:
        # semi-major axis
        gdat.dictlist['smax'] = (gdat.dictlist['radistar']) / gdat.dictlist['rsma']
    
    # temp
    gdat.dictlist['sini'] = np.sqrt(1. - gdat.dictlist['cosi']**2)
    gdat.dictlist['omeg'] = 180. / np.pi * np.mod(np.arctan2(gdat.dictlist['esin'], gdat.dictlist['ecos']), 2 * np.pi)
    gdat.dictlist['rs2a'] = gdat.dictlist['rsma'] / (1. + gdat.dictlist['rrat'])
    gdat.dictlist['sinw'] = np.sin(np.pi / 180. * gdat.dictlist['omeg'])
    gdat.dictlist['imfa'] = ephesus.retr_imfa(gdat.dictlist['cosi'], gdat.dictlist['rs2a'], gdat.dictlist['ecce'], gdat.dictlist['sinw'])
   
    # RV semi-amplitude
    gdat.dictlist['rvsapred'] = ephesus.retr_rvelsema(gdat.dictlist['peri'], gdat.dictlist['massplanpred'], gdat.dictlist['massstar'], gdat.dictlist['incl'], gdat.dictlist['ecce'])
    
    ## expected Doppler beaming (DB)
    deptbeam = 1e3 * 4. * gdat.dictlist['rvsapred'] / 3e8 * gdat.consbeam # [ppt]

    ## expected ellipsoidal variation (EV)
    ## limb and gravity darkening coefficients from Claret2017
    print('temp: connect these to Claret2017')
    # linear limb-darkening coefficient
    coeflidaline = 0.4
    # gravitational darkening coefficient
    coefgrda = 0.2
    alphelli = ephesus.retr_alphelli(coeflidaline, coefgrda)
    gdat.dictlist['deptelli'] = 1e3 * alphelli * gdat.dictlist['massplanused'] * np.sin(gdat.dictlist['incl'] / 180. * np.pi)**2 / \
                                                                  gdat.dictlist['massstar']* (gdat.dictlist['radistar'] / gdat.dictlist['smax'])**3 # [ppt]
    if gdat.typeverb > 0:
        print('Calculating durations...')

    gdat.dictlist['duratranfull'] = ephesus.retr_duratranfull(gdat.dictlist['peri'], gdat.dictlist['rs2a'], gdat.dictlist['sini'], \
                                                                                    gdat.dictlist['rrat'], gdat.dictlist['imfa'])
    gdat.dictlist['duratrantotl'] = ephesus.retr_duratrantotl(gdat.dictlist['peri'], gdat.dictlist['rs2a'], gdat.dictlist['sini'], \
                                                                                    gdat.dictlist['rrat'], gdat.dictlist['imfa'])
    
    gdat.dictlist['maxmdeptblen'] = 1e3 * (1. - gdat.dictlist['duratranfull'] / gdat.dictlist['duratrantotl'])**2 / \
                                                                    (1. + gdat.dictlist['duratranfull'] / gdat.dictlist['duratrantotl'])**2 # [ppt]
    gdat.dictlist['minmdilu'] = gdat.dictlist['dept'] / gdat.dictlist['maxmdeptblen']
    gdat.dictlist['minmratiflux'] = gdat.dictlist['minmdilu'] / (1. - gdat.dictlist['minmdilu'])
    gdat.dictlist['maxmdmag'] = -2.5 * np.log10(gdat.dictlist['minmratiflux'])
    
    # orbital
    ## RM effect
    gdat.dictlist['amplrmef'] = 2. / 3. * gdat.dictlist['vsiistar'] * 1e-3 * gdat.dictlist['dept'] * np.sqrt(1. - gdat.dictlist['imfa'])
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
        gdat.dictlist['albg'] = retr_albg(gdat.dictlist['amplplanrefl'], gdat.dictlist['radiplan'], gdat.dictlist['smax'])

    if gdat.typeverb > 0:
        print('Calculating the equilibrium temperature of the planets...')
    
    gdat.dictlist['tmptequi'] = gdat.dictlist['tmptstar'] * np.sqrt(gdat.dictlist['radistar'] / gdat.dictlist['smax'] / 2.)
    
    if False and gdat.labltarg == 'WASP-121' and strgpdfn != 'prio':
        
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
   
        ### spectrum of the host star
        gdat.meanwlenthomraww = arrymodl[:, 0]
        gdat.specstarthomraww = arrymodl[:, 9]
        
        ## calculate the geometric albedo "informed" by the ATMO posterior
        wlenmodl = arrymodl[:, 0]
        deptmodl = arrymodl[:, 1]
        indxwlenmodltess = np.where((wlenmodl > 0.6) & (wlenmodl < 0.95))[0]
        gdat.amplplantheratmo = np.mean(deptmodl[indxwlenmodltess])
        print('gdat.amplplantheratmo')
        print(gdat.amplplantheratmo)
        gdat.dictlist['amplplanreflatmo'] = 1e-6 * arrydata[0, 2] + np.random.randn(gdat.numbsamp).reshape((gdat.numbsamp, 1)) \
                                                                                                    * arrydata[0, 3] * 1e-6 - gdat.amplplantheratmo
        #gdat.dictlist['amplplanreflatmo'] = gdat.dictlist['amplplan'] - gdat.amplplantheratmo
        gdat.dictlist['albginfo'] = retr_albg(gdat.dictlist['amplplanreflatmo'], gdat.dictlist['radiplan'], gdat.dictlist['smax'])
        
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
        
        print('arrydata[0, :]')
        print(arrydata[0, :])
        print('arrydata[-1, :]')
        print(arrydata[-1, :])
        
        # calculate brightness temperatures
        listlablpara = [['Temperature', 'K']]
        gdat.rratmedi = np.median(gdat.dictlist['rrat'], axis=0)
        listscalpara = ['self']
        listminmpara = np.array([1000.])
        listmaxmpara = np.array([4000.])
        meangauspara = None
        stdvgauspara = None
        numbpara = len(listlablpara)
        numbsampwalk = 10
        numbsampburnwalk = 5
        gdat.numbdatatmpt = arrydata.shape[0]
        gdat.indxdatatmpt = np.arange(gdat.numbdatatmpt)
        listtmpt = []
        specarry = np.empty((2, 3, gdat.numbdatatmpt))
        for k in gdat.indxdatatmpt:
            
            if not (k == 0 or k == gdat.numbdatatmpt - 1):
                continue
            gdat.minmwlen = arrydata[k, 0] - arrydata[k, 1]
            gdat.maxmwlen = arrydata[k, 0] + arrydata[k, 1]
            gdat.binswlen = np.linspace(gdat.minmwlen, gdat.maxmwlen, 100)
            gdat.meanwlen = (gdat.binswlen[1:] + gdat.binswlen[:-1]) / 2.
            gdat.diffwlen = (gdat.binswlen[1:] - gdat.binswlen[:-1]) / 2.
            gdat.cntrwlen = np.mean(gdat.meanwlen)
            strgextn = 'tmpt_%d' % k
            gdat.indxenerdata = k

            gdat.specstarintg = retr_modl_spec(gdat, gdat.tmptstar, strgtype='intg')
            
            gdat.specstarthomlogt = scipy.interpolate.interp1d(gdat.meanwlenthomraww, gdat.specstarthomraww)(gdat.cntrwlen)
            gdat.specstarthomdiff = gdat.specstarthomlogt / gdat.cntrwlen
            gdat.specstarthomintg = np.sum(gdat.diffwlen * \
                                    scipy.interpolate.interp1d(gdat.meanwlenthomraww, gdat.specstarthomraww)(gdat.meanwlen) / gdat.meanwlen)

            gdat.deptobsd = arrydata[k, 2]
            gdat.stdvdeptobsd = arrydata[k, 3]
            gdat.varideptobsd = gdat.stdvdeptobsd**2
            listtmpttemp = tdpy.mcmc.samp(gdat, gdat.pathalle[strgpdfn], numbsampwalk, \
                                          retr_llik_spec, \
                             listlablpara, listscalpara, listminmpara, listmaxmpara, meangauspara, stdvgauspara, numbdata, strgextn=strgextn, \
                                          numbsampburnwalk=numbsampburnwalk, \
                             )
            listtmpt.append(listtmpttemp)
        listtmpt = np.vstack(listtmpt).T
        print('listtmpt')
        summgene(listtmpt)
        indxsamp = np.random.choice(np.arange(listtmpt.shape[0]), size=gdat.numbsamp, replace=False)
        # dayside and nightside temperatures to be used for albedo and circulation efficiency calculation
        gdat.dictlist['tmptdayy'] = listtmpt[indxsamp, 0, None]
        gdat.dictlist['tmptnigh'] = listtmpt[indxsamp, -1, None]
        # dayside/nightside temperature contrast
        gdat.dictlist['tmptcont'] = (gdat.dictlist['tmptdayy'] - gdat.dictlist['tmptnigh']) / gdat.dictlist['tmptdayy']
        
    # copy the prior
    gdat.dictlist['projoblq'] = np.random.randn(gdat.numbsamp)[:, None] * gdat.stdvprojoblqprio[None, :] + gdat.projoblqprio[None, :]
    
    gdat.boolsampbadd = np.zeros(gdat.numbsamp, dtype=bool)
    for j in gdat.indxcomp:
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
        indxfini = np.isfinite(gdat.dictlist[strgfeat])
        if not indxfini.all():
            print('Warning! %s are not all finite!' % strgfeat)
            print('gdat.dictlist[strgfeat]')
            summgene(gdat.dictlist[strgfeat])
            print('indxfini')
            summgene(indxfini)
        gdat.dictpost[strgfeat][0, ...] = np.nanpercentile(gdat.dictlist[strgfeat], 16., 0)
        gdat.dictpost[strgfeat][1, ...] = np.nanpercentile(gdat.dictlist[strgfeat], 50., 0)
        gdat.dictpost[strgfeat][2, ...] = np.nanpercentile(gdat.dictlist[strgfeat], 84., 0)
        gdat.dicterrr[strgfeat][0, ...] = gdat.dictpost[strgfeat][1, ...]
        gdat.dicterrr[strgfeat][1, ...] = gdat.dictpost[strgfeat][1, ...] - gdat.dictpost[strgfeat][0, ...]
        gdat.dicterrr[strgfeat][2, ...] = gdat.dictpost[strgfeat][2, ...] - gdat.dictpost[strgfeat][1, ...]
        
        print(strgfeat)
        print(gdat.dicterrr[strgfeat])
        print('')
    # augment
    gdat.dictfeatobjt['radistar'] = gdat.dicterrr['radistar'][0, :]
    gdat.dictfeatobjt['radiplan'] = gdat.dicterrr['radiplan'][0, :]
    gdat.dictfeatobjt['massplan'] = gdat.dicterrr['massplan'][0, :]
    gdat.dictfeatobjt['stdvradistar'] = np.mean(gdat.dicterrr['radistar'][1:, :], 0)
    gdat.dictfeatobjt['stdvmassstar'] = np.mean(gdat.dicterrr['massstar'][1:, :], 0)
    gdat.dictfeatobjt['stdvtmptstar'] = np.mean(gdat.dicterrr['tmptstar'][1:, :], 0)
    gdat.dictfeatobjt['stdvloggstar'] = np.mean(gdat.dicterrr['loggstar'][1:, :], 0)
    gdat.dictfeatobjt['stdvradiplan'] = np.mean(gdat.dicterrr['radiplan'][1:, :], 0)
    gdat.dictfeatobjt['stdvmassplan'] = np.mean(gdat.dicterrr['massplan'][1:, :], 0)
    gdat.dictfeatobjt['stdvtmptplan'] = np.mean(gdat.dicterrr['tmptplan'][1:, :], 0)
    gdat.dictfeatobjt['stdvesmm'] = np.mean(gdat.dicterrr['esmm'][1:, :], 0)
    gdat.dictfeatobjt['stdvtsmm'] = np.mean(gdat.dicterrr['tsmm'][1:, :], 0)
    

def proc_alle(gdat, typemodl):
    
    #_0003: single component offset baseline
    #_0004: multiple components, offset baseline
        
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
            
            if gdat.boolinfefoldbind:
                listarrytserbdtrtemp = np.copy(gdat.arrypcur['primbdtrbindtotl'][b][p][0])
                listarrytserbdtrtemp[:, 0] *= gdat.periprio[0]
                listarrytserbdtrtemp[:, 0] += gdat.epocprio[0]
            else:
                listarrytserbdtrtemp = gdat.arrytser['bdtr'][b][p]
            
            # make sure the data are time-sorted
            #indx = np.argsort(listarrytserbdtrtemp[:, 0])
            #listarrytserbdtrtemp = listarrytserbdtrtemp[indx, :]
                
            if gdat.typeverb > 0:
                print('Writing to %s...' % path)
            np.savetxt(path, listarrytserbdtrtemp, delimiter=',', header='time,%s,%s_err' % (gdat.liststrgtseralle[b], gdat.liststrgtseralle[b]))
    
    ## params_star
    pathparastar = gdat.pathalle[typemodl] + 'params_star.csv'
    if not os.path.exists(pathparastar):
        if gdat.typeverb > 0:
            print('Writing to %s...' % pathparastar)
        objtfile = open(pathparastar, 'w')
        objtfile.write('#R_star,R_star_lerr,R_star_uerr,M_star,M_star_lerr,M_star_uerr,Teff_star,Teff_star_lerr,Teff_star_uerr\n')
        objtfile.write('#R_sun,R_sun,R_sun,M_sun,M_sun,M_sun,K,K,K\n')
        objtfile.write('%g,%g,%g,%g,%g,%g,%g,%g,%g' % (gdat.radistar, gdat.stdvradistar, gdat.stdvradistar, \
                                                       gdat.massstar, gdat.stdvmassstar, gdat.stdvmassstar, \
                                                                                                      gdat.tmptstar, gdat.stdvtmptstar, gdat.stdvtmptstar))
        objtfile.close()

    ## params
    dictalleparadefa = dict()
    pathpara = gdat.pathalle[typemodl] + 'params.csv'
    if not os.path.exists(pathpara):
        cmnd = 'touch %s' % (pathpara)
        print(cmnd)
        os.system(cmnd)
    
        for j in gdat.indxcomp:
            strgrrat = '%s_rr' % gdat.liststrgcomp[j]
            strgrsma = '%s_rsuma' % gdat.liststrgcomp[j]
            strgcosi = '%s_cosi' % gdat.liststrgcomp[j]
            strgepoc = '%s_epoch' % gdat.liststrgcomp[j]
            strgperi = '%s_period' % gdat.liststrgcomp[j]
            strgecos = '%s_f_c' % gdat.liststrgcomp[j]
            strgesin = '%s_f_s' % gdat.liststrgcomp[j]
            strgrvsa = '%s_K' % gdat.liststrgcomp[j]
            dictalleparadefa[strgrrat] = ['%f' % gdat.rratprio[j], '1', 'uniform 0 %f' % (4 * gdat.rratprio[j]), \
                                                                            '$R_{%s} / R_\star$' % gdat.liststrgcomp[j], '']
            
            dictalleparadefa[strgrsma] = ['%f' % gdat.rsmaprio[j], '1', 'uniform 0 %f' % (4 * gdat.rsmaprio[j]), \
                                                                      '$(R_\star + R_{%s}) / a_{%s}$' % (gdat.liststrgcomp[j], gdat.liststrgcomp[j]), '']
            dictalleparadefa[strgcosi] = ['%f' % gdat.cosiprio[j], '1', 'uniform 0 %f' % max(0.1, 4 * gdat.cosiprio[j]), \
                                                                                        '$\cos{i_{%s}}$' % gdat.liststrgcomp[j], '']
            dictalleparadefa[strgepoc] = ['%f' % gdat.epocprio[j], '1', \
                                            'uniform %f %f' % (gdat.epocprio[j] - gdat.stdvepocprio[j], gdat.epocprio[j] + gdat.stdvepocprio[j]), \
                                                                    '$T_{0;%s}$' % gdat.liststrgcomp[j], '$\mathrm{BJD}$']
            dictalleparadefa[strgperi] = ['%f' % gdat.periprio[j], '1', \
                                     'uniform %f %f' % (gdat.periprio[j] - 3. * gdat.stdvperiprio[j], gdat.periprio[j] + 3. * gdat.stdvperiprio[j]), \
                                                                    '$P_{%s}$' % gdat.liststrgcomp[j], 'days']
            dictalleparadefa[strgecos] = ['%f' % gdat.ecosprio[j], '0', 'uniform -0.9 0.9', \
                                                                '$\sqrt{e_{%s}} \cos{\omega_{%s}}$' % (gdat.liststrgcomp[j], gdat.liststrgcomp[j]), '']
            dictalleparadefa[strgesin] = ['%f' % gdat.esinprio[j], '0', 'uniform -0.9 0.9', \
                                                                '$\sqrt{e_{%s}} \sin{\omega_{%s}}$' % (gdat.liststrgcomp[j], gdat.liststrgcomp[j]), '']
            dictalleparadefa[strgrvsa] = ['%f' % gdat.rvsaprio[j], '0', \
                                    'uniform %f %f' % (max(0, gdat.rvsaprio[j] - 5 * gdat.stdvrvsaprio[j]), gdat.rvsaprio[j] + 5 * gdat.stdvrvsaprio[j]), \
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
                        maxmshft = 0.25 * gdat.periprio[j]
                        minmshft = -maxmshft

                        dictalleparadefa['%s_phase_curve_atmospheric_shift_%s' % (gdat.liststrgcomp[j], gdat.liststrginst[b][p])] = \
                                         ['0', '1', 'uniform %.3g %.3g' % (minmshft, maxmshft), \
                                            '$\Delta_\mathrm{%s; %s}$' % (gdat.liststrgcomp[j], gdat.listlablinst[b][p]), '']
        if typemodl == 'pfss':
            for p in gdat.indxinst[1]:
                                ['', 'host_vsini,%g,1,uniform %g %g,$v \sin i$$,\n' % (gdat.vsiistarprio, 0, \
                                                                                                                            10 * gdat.vsiistarprio)], \
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
        for j in gdat.indxcomp:
            dictallesettdefa['%s_flux_weighted_PFS' % gdat.liststrgcomp[j]] = 'True'
    
    pathsett = gdat.pathalle[typemodl] + 'settings.csv'
    if not os.path.exists(pathsett):
        cmnd = 'touch %s' % (pathsett)
        print(cmnd)
        os.system(cmnd)
        
        dictallesettdefa['fast_fit_width'] = '%.3g' % np.amax(gdat.duramask) / 24.
        dictallesettdefa['multiprocess'] = 'True'
        dictallesettdefa['multiprocess_cores'] = 'all'

        dictallesettdefa['mcmc_nwalkers'] = '100'
        dictallesettdefa['mcmc_total_steps'] = '20000'
        dictallesettdefa['mcmc_burn_steps'] = '10000'
        dictallesettdefa['mcmc_thin_by'] = '500'
        
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
                for j in gdat.indxcomp:
                    dictallesettdefa['%s_grid_%s' % (gdat.liststrgcomp[j], gdat.liststrginst[b][p])] = 'very_sparse'
            
            if gdat.numbinst[b] > 0:
                if b == 0:
                    strg = 'companions_phot'
                if b == 1:
                    strg = 'companions_rv'
                varb = ''
                cntr = 0
                for j in gdat.indxcomp:
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
        print('Thinning down the allesfitter chain!')
        gdat.indxsamp = np.random.choice(np.arange(gdat.numbsamp), size=10000, replace=False)
        gdat.numbsamp = 10000
    else:
        gdat.indxsamp = np.arange(gdat.numbsamp)
    
    print('gdat.numbsamp')
    print(gdat.numbsamp)

    calc_prop(gdat, typemodl)

    gdat.arrytser['bdtr'+typemodl] = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gdat.arrytser['modl'+typemodl] = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gdat.arrytser['resi'+typemodl] = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gdat.listarrytser['bdtr'+typemodl] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gdat.listarrytser['modl'+typemodl] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gdat.listarrytser['resi'+typemodl] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            gdat.arrytser['modl'+typemodl][b][p] = np.empty((gdat.time[b][p].size, 3))
            gdat.arrytser['modl'+typemodl][b][p][:, 0] = gdat.time[b][p]
            gdat.arrytser['modl'+typemodl][b][p][:, 1] = gdat.objtalle[typemodl].get_posterior_median_model(gdat.liststrginst[b][p], \
                                                                                                             gdat.liststrgtseralle[b], xx=gdat.time[b][p])
            gdat.arrytser['modl'+typemodl][b][p][:, 2] = 0.

            gdat.arrytser['resi'+typemodl][b][p] = np.copy(gdat.arrytser['bdtr'][b][p])
            gdat.arrytser['resi'+typemodl][b][p][:, 1] -= gdat.arrytser['modl'+typemodl][b][p][:, 1]
            for y in gdat.indxchun[b][p]:
                gdat.listarrytser['modl'+typemodl][b][p][y] = np.copy(gdat.listarrytser['bdtr'][b][p][y])
                gdat.listarrytser['modl'+typemodl][b][p][y][:, 1] = gdat.objtalle[typemodl].get_posterior_median_model(gdat.liststrginst[b][p], \
                                                                                                       gdat.liststrgtseralle[b], xx=gdat.listtime[b][p][y])
                
                gdat.listarrytser['resi'+typemodl][b][p][y] = np.copy(gdat.listarrytser['bdtr'][b][p][y])
                gdat.listarrytser['resi'+typemodl][b][p][y][:, 1] -= gdat.listarrytser['modl'+typemodl][b][p][y][:, 1]
    
                # plot residuals
                if gdat.boolplottser:
                    plot_tser(gdat, b, p, y, 'resi' + typemodl)

    # write the model to file
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            path = gdat.pathdatatarg + 'arry%smodl_%s.csv' % (gdat.liststrgdatatser[b], gdat.liststrginst[b][p])
            if gdat.typeverb > 0:
                print('Writing to %s...' % path)
            np.savetxt(path, gdat.arrytser['modl'+typemodl][b][p], delimiter=',', \
                                                    header='time,%s,%s_err' % (gdat.liststrgtseralle[b], gdat.liststrgtseralle[b]))

    # number of samples to plot
    gdat.numbsampplot = min(10, gdat.numbsamp)
    gdat.indxsampplot = np.random.choice(gdat.indxsamp, gdat.numbsampplot, replace=False)
    
    gdat.arrypcur['primbdtr'+typemodl] = [[[[] for j in gdat.indxcomp] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gdat.arrypcur['primbdtr'+typemodl+'bindtotl'] = [[[[] for j in gdat.indxcomp] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gdat.arrypcur['primbdtr'+typemodl+'bindzoom'] = [[[[] for j in gdat.indxcomp] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    
    gdat.listarrypcur = dict()
    gdat.listarrypcur['quadmodl'+typemodl] = [[[[] for j in gdat.indxcomp] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            for j in gdat.indxcomp:
                gdat.listarrypcur['quadmodl'+typemodl][b][p][j] = np.empty((gdat.numbsampplot, gdat.numbtimeclen[b][p][j], 3))
    
    print('gdat.numbsampplot')
    print(gdat.numbsampplot)
    gdat.arrypcur['primbdtr'+typemodl] = [[[[] for j in gdat.indxcomp] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    gdat.arrypcur['primmodltotl'+typemodl] = [[[[] for j in gdat.indxcomp] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
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
                gdat.arrytser[strgpcur+typemodl][b][p] = np.copy(gdat.arrytser['bdtr'][b][p])
    for strgpcurcomp in gdat.liststrgpcurcomp:
        gdat.arrytser[strgpcurcomp+typemodl] = [[[[] for j in gdat.indxcomp] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                for j in gdat.indxcomp:
                    gdat.arrytser[strgpcurcomp+typemodl][b][p][j] = np.copy(gdat.arrytser['bdtr'][b][p])
    for strgpcurcomp in gdat.liststrgpcurcomp + gdat.liststrgpcur:
        for strgextnbins in ['', 'bindtotl']:
            gdat.arrypcur['quad' + strgpcurcomp + typemodl + strgextnbins] = [[[[] for j in gdat.indxcomp] \
                                                                                    for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
    
    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
        
            gdat.listlcurmodl = np.empty((gdat.numbsampplot, gdat.time[b][p].size))
            print('Phase-folding the posterior samples from the model light curve...')
            for ii in tqdm(range(gdat.numbsampplot)):
                i = gdat.indxsampplot[ii]
                
                # this is only the physical model and excludes the baseline, which is available separately via get_one_posterior_baseline()
                gdat.listarrytsermodl[b][p][ii, :, 1] = gdat.objtalle[typemodl].get_one_posterior_model(gdat.liststrginst[b][p], \
                                                                        gdat.liststrgtseralle[b], xx=gdat.time[b][p], sample_id=i)
                
                for j in gdat.indxcomp:
                    gdat.listarrypcur['quadmodl'+typemodl][b][p][j][ii, :, :] = \
                                            ephesus.fold_tser(gdat.listarrytsermodl[b][p][ii, gdat.listindxtimeclen[j][b][p], :], \
                                                                                   gdat.dicterrr['epoc'][0, j], gdat.dicterrr['peri'][0, j], phasshft=0.25)

            
            ## plot components in the zoomed panel
            for j in gdat.indxcomp:
                
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
                
                offsdays = np.mean(gdat.arrytser['modlplan'+typemodl][b][p][j][gdat.listindxtimetran[j][b][p][1], 1])
                print('offsdays')
                print(offsdays)
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
                gdat.arrytser['bdtrplan'+typemodl][b][p][j][:, 1] = gdat.arrytser['bdtr'][b][p][:, 1] \
                                                                                - gdat.arrytser['modlstel'+typemodl][b][p][j][:, 1] \
                                                                                - gdat.arrytser['modlelli'+typemodl][b][p][j][:, 1] \
                                                                                - gdat.arrytser['modlbeam'+typemodl][b][p][j][:, 1]
                gdat.arrytser['bdtrplan'+typemodl][b][p][j][:, 1] -= offsdays
                    
            # get allesfitter baseline model
            gdat.arrytser['modlbase'+typemodl][b][p] = np.copy(gdat.arrytser['bdtr'][b][p])
            gdat.arrytser['modlbase'+typemodl][b][p][:, 1] = gdat.objtalle[typemodl].get_posterior_median_baseline(gdat.liststrginst[b][p], 'flux', \
                                                                                                                                xx=gdat.time[b][p])
            # get allesfitter-detrended data
            gdat.arrytser['bdtr'+typemodl][b][p] = np.copy(gdat.arrytser['bdtr'][b][p])
            gdat.arrytser['bdtr'+typemodl][b][p][:, 1] = gdat.arrytser['bdtr'][b][p][:, 1] - gdat.arrytser['modlbase'+typemodl][b][p][:, 1]
            for y in gdat.indxchun[b][p]:
                # get allesfitter baseline model
                gdat.listarrytser['modlbase'+typemodl][b][p][y] = np.copy(gdat.listarrytser['bdtr'][b][p][y])
                gdat.listarrytser['modlbase'+typemodl][b][p][y][:, 1] = gdat.objtalle[typemodl].get_posterior_median_baseline(gdat.liststrginst[b][p], \
                                                                                           'flux', xx=gdat.listarrytser['modlbase'+typemodl][b][p][y][:, 0])
                # get allesfitter-detrended data
                gdat.listarrytser['bdtr'+typemodl][b][p][y] = np.copy(gdat.listarrytser['bdtr'][b][p][y])
                gdat.listarrytser['bdtr'+typemodl][b][p][y][:, 1] = gdat.listarrytser['bdtr'+typemodl][b][p][y][:, 1] - \
                                                                                gdat.listarrytser['modlbase'+typemodl][b][p][y][:, 1]
           
            print('Phase folding and binning the light curve for inference named %s...' % typemodl)
            for j in gdat.indxcomp:
                
                gdat.arrypcur['primmodltotl'+typemodl][b][p][j] = ephesus.fold_tser(gdat.arrytser['modltotl'+typemodl][b][p][j][gdat.listindxtimeclen[j][b][p], :], \
                                                                                    gdat.dicterrr['epoc'][0, j], gdat.dicterrr['peri'][0, j])
                
                gdat.arrypcur['primbdtr'+typemodl][b][p][j] = ephesus.fold_tser(gdat.arrytser['bdtr'+typemodl][b][p][gdat.listindxtimeclen[j][b][p], :], \
                                                                                    gdat.dicterrr['epoc'][0, j], gdat.dicterrr['peri'][0, j])
                
                gdat.arrypcur['primbdtr'+typemodl+'bindtotl'][b][p][j] = ephesus.rebn_tser(gdat.arrypcur['primbdtr'+typemodl][b][p][j], \
                                                                                                                    binsxdat=gdat.binsphasprimtotl)
                
                gdat.arrypcur['primbdtr'+typemodl+'bindzoom'][b][p][j] = ephesus.rebn_tser(gdat.arrypcur['primbdtr'+typemodl][b][p][j], \
                                                                                                                    binsxdat=gdat.binsphasprimzoom[j])

                for strgpcurcomp in gdat.liststrgpcurcomp + gdat.liststrgpcur:
                    
                    if strgpcurcomp in gdat.liststrgpcurcomp:
                        arrytsertemp = gdat.arrytser[strgpcurcomp+typemodl][b][p][j][gdat.listindxtimeclen[j][b][p], :]
                    else:
                        arrytsertemp = gdat.arrytser[strgpcurcomp+typemodl][b][p][gdat.listindxtimeclen[j][b][p], :]
                    
                    if strgpcurcomp == 'bdtr':
                        boolpost = True
                    else:
                        boolpost = False
                    gdat.arrypcur['quad'+strgpcurcomp+typemodl][b][p][j] = \
                                        ephesus.fold_tser(arrytsertemp, gdat.dicterrr['epoc'][0, j], gdat.dicterrr['peri'][0, j], phasshft=0.25) 
                
                    gdat.arrypcur['quad'+strgpcurcomp+typemodl+'bindtotl'][b][p][j] = ephesus.rebn_tser(gdat.arrypcur['quad'+strgpcurcomp+typemodl][b][p][j], \
                                                                                                                binsxdat=gdat.binsphasquadtotl)
                    
                    # write
                    path = gdat.pathdatatarg + 'arrypcurquad%sbindtotl_%s_%s.csv' % (strgpcurcomp, gdat.liststrgcomp[j], gdat.liststrginst[b][p])
                    if gdat.typeverb > 0:
                        print('Writing to %s...' % path)
                    np.savetxt(path, gdat.arrypcur['quad%s%sbindtotl' % (strgpcurcomp, typemodl)][b][p][j], delimiter=',', \
                                                    header='phase,%s,%s_err' % (gdat.liststrgtseralle[b], gdat.liststrgtseralle[b]))
                    
                    plot_pser(gdat, 'quad'+strgpcurcomp+typemodl, boolpost=boolpost)
                
                
    # plots
    ## plot GP-detrended phase curves
    if gdat.boolplottser:
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                for y in gdat.indxchun[b][p]:
                    plot_tser(gdat, b, p, y, 'bdtr'+typemodl)
                    plot_pser(gdat, b, p, y, 'primbdtr'+typemodl, boolpost=True)
    if gdat.boolplotprop:
        plot_prop(gdat, gdat.typeprioplan + typemodl)
    
    # print out transit times
    for j in gdat.indxcomp:
        print(gdat.liststrgcomp[j])
        time = np.empty(500)
        for n in range(500):
            time[n] = gdat.dicterrr['epoc'][0, j] + gdat.dicterrr['peri'][0, j] * n
        objttime = astropy.time.Time(time, format='jd', scale='utc')#, out_subfmt='date_hm')
        listtimelabl = objttime.iso
        for n in range(500):
            if time[n] > 2458788 and time[n] < 2458788 + 200:
                print('%f, %s' % (time[n], listtimelabl[n]))


    if typemodl == '0003' or typemodl == '0004':
            
        if typemodl == '0003':
            listlablpara = [['Nightside', 'ppm'], ['Secondary', 'ppm'], ['Planetary Modulation', 'ppm'], ['Thermal', 'ppm'], \
                                                        ['Reflected', 'ppm'], ['Phase shift', 'deg'], ['Geometric Albedo', '']]
        else:
            listlablpara = [['Nightside', 'ppm'], ['Secondary', 'ppm'], ['Thermal', 'ppm'], \
                                  ['Reflected', 'ppm'], ['Thermal Phase shift', 'deg'], ['Reflected Phase shift', 'deg'], ['Geometric Albedo', '']]
        numbpara = len(listlablpara)
        indxpara = np.arange(numbpara)
        listpost = np.empty((gdat.numbsamp, numbpara))
        
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                for j in gdat.indxcomp:
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
                    tdpy.mcmc.plot_grid(gdat.pathalle[typemodl], 'pcur_%s' % typemodl, listpost, listlablpara, plotsize=2.5)

        # plot phase curve
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                
                ## determine data gaps for overplotting model without the data gaps
                gdat.indxtimegapp = np.argmax(gdat.time[b][p][1:] - gdat.time[b][p][:-1]) + 1
                
                for j in gdat.indxcomp:
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
                                ydat = gdat.arrytser['bdtr'+typemodl][b][p][:, 1] + gdat.dicterrr['amplnigh'][0, 0]
                            if k == 1:
                                xdat = gdat.arrypcur['quadbdtr'+typemodl][b][p][j][:, 0]
                                ydat = gdat.arrypcur['quadbdtr'+typemodl][b][p][j][:, 1] + gdat.dicterrr['amplnigh'][0, 0]
                            axis[k].plot(xdat, ydat, '.', color='grey', alpha=0.3, label='Raw data')
                        
                        ## binned data
                        if k > 0:
                            xdat = gdat.arrypcur['quadbdtr'+typemodl+'bindtotl'][b][p][j][:, 0]
                            ydat = gdat.arrypcur['quadbdtr'+typemodl+'bindtotl'][b][p][j][:, 1] + gdat.dicterrr['amplnigh'][0, 0]
                            yerr = np.copy(gdat.arrypcur['quadbdtr'+typemodl+'bindtotl'][b][p][j][:, 2])
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
                            xdat = gdat.arrypcur['quadmodl'+typemodl][b][p][j][:, 0]
                            ydat = gdat.arrypcur['quadmodl'+typemodl][b][p][j][:, 1] + gdat.dicterrr['amplnigh'][0, 0]
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
                        
                        # add Vivien's result
                        if k == 2 and gdat.labltarg == 'WASP-121':
                            axis[k].plot(gdat.phasvivi, gdat.deptvivi*1e3, color='orange', lw=2, label='GCM (Parmentier+2018)')
                            axis[k].axhline(0., ls='-.', alpha=0.3, color='grey')

                        if k == 0:
                            axis[k].set(xlabel='Time [BJD - %d]' % gdat.timeoffs)
                        if k > 0:
                            axis[k].set(xlabel='Phase')
                    axis[0].set(ylabel='Relative flux')
                    axis[1].set(ylabel='Relative flux')
                    axis[2].set(ylabel='Relative flux - 1 [ppm]')
                    
                    if gdat.labltarg == 'WASP-121':
                        ylimpcur = [-400, 1000]
                    else:
                        ylimpcur = [-100, 300]
                    axis[2].set_ylim(ylimpcur)
                    
                    xdat = gdat.arrypcur['quadmodlstel'+typemodl][b][p][j][:, 0]
                    ydat = (gdat.arrypcur['quadmodlstel'+typemodl][b][p][j][:, 1] - 1.) * 1e6
                    axis[2].plot(xdat, ydat, lw=2, color='orange', label='Stellar baseline', ls='--', zorder=11)
                    
                    xdat = gdat.arrypcur['quadmodlelli'+typemodl][b][p][j][:, 0]
                    ydat = (gdat.arrypcur['quadmodlelli'+typemodl][b][p][j][:, 1] - 1.) * 1e6
                    axis[2].plot(xdat, ydat, lw=2, color='r', ls='--', label='Ellipsoidal variation')
                    
                    xdat = gdat.arrypcur['quadmodlelli'+typemodl][b][p][j][:, 0]
                    ydat = (gdat.arrypcur['quadmodlelli'+typemodl][b][p][j][:, 1] - 1.) * 1e6
                    axis[2].plot(xdat, ydat, lw=2, color='r', ls='--', label='Ellipsoidal variation')
                    
                    xdat = gdat.arrypcur['quadmodlplan'+typemodl][b][p][j][:, 0]
                    ydat = (gdat.arrypcur['quadmodlplan'+typemodl][b][p][j][:, 1] - 1.) * 1e6
                    axis[2].plot(xdat, ydat, lw=2, color='g', label='Planetary', ls='--')
    
                    xdat = gdat.arrypcur['quadmodlnigh'+typemodl][b][p][j][:, 0]
                    ydat = (gdat.arrypcur['quadmodlnigh'+typemodl][b][p][j][:, 1] - 1.) * 1e6
                    axis[2].plot(xdat, ydat, lw=2, color='olive', label='Planetary baseline', ls='--', zorder=11)
    
                    xdat = gdat.arrypcur['quadmodlpmod'+typemodl][b][p][j][:, 0]
                    ydat = (gdat.arrypcur['quadmodlpmod'+typemodl][b][p][j][:, 1] - 1.) * 1e6
                    axis[2].plot(xdat, ydat, lw=2, color='m', label='Planetary modulation', ls='--', zorder=11)
                     
                    ## legend
                    axis[2].legend(ncol=3)
                    
                    path = gdat.pathalle[typemodl] + 'pcur_grid_%s_%s_%s.%s' % (typemodl, gdat.strgtarg, gdat.typeprioplan, gdat.typefileplot)
                    if gdat.typeverb > 0:
                        print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()
                   

        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                for j in gdat.indxcomp:
        
                    # replot phase curve
                    ### sample model phas
                    #numbphasfine = 1000
                    #gdat.meanphasfine = np.linspace(np.amin(gdat.arrypcur['quadbdtr'][0][gdat.indxphasotpr, 0]), \
                    #                                np.amax(gdat.arrypcur['quadbdtr'][0][gdat.indxphasotpr, 0]), numbphasfine)
                    #indxphasfineinse = np.where(abs(gdat.meanphasfine - 0.5) < phasseco)[0]
                    #indxphasfineotprleft = np.where(-gdat.meanphasfine > phasmask)[0]
                    #indxphasfineotprrght = np.where(gdat.meanphasfine > phasmask)[0]
       
                    indxphasmodlouttprim = [[] for a in range(2)]
                    indxphasdatabindouttprim = [[] for a in range(2)]
                    indxphasmodlouttprim[0] = np.where(gdat.arrypcur['quadmodl'+typemodl][b][p][j][:, 0] < -0.05)[0]
                    indxphasdatabindouttprim[0] = np.where(gdat.arrypcur['quadbdtrbindtotl'][b][p][j][:, 0] < -0.05)[0]
                    indxphasmodlouttprim[1] = np.where(gdat.arrypcur['quadmodl'+typemodl][b][p][j][:, 0] > 0.05)[0]
                    indxphasdatabindouttprim[1] = np.where(gdat.arrypcur['quadbdtrbindtotl'][b][p][j][:, 0] > 0.05)[0]

                    # plot the phase curve with components
                    figr, axis = plt.subplots(figsize=gdat.figrsizeydob)
                    ## data
                    axis.errorbar(gdat.arrypcur['quadbdtrbindtotl'][b][p][j][:, 0], \
                                   (gdat.arrypcur['quadbdtrbindtotl'][b][p][j][:, 1] + gdat.dicterrr['amplnigh'][0, 0] - 1) * 1e6, \
                                   yerr=1e6*gdat.arrypcur['quadbdtrbindtotl'][b][p][j][:, 2], color='k', marker='o', ls='', markersize=2, lw=1, label='Data')
                    ## total model
                    axis.plot(gdat.arrypcur['quadmodl'+typemodl][b][p][j][:, 0], \
                                                    1e6*(gdat.arrypcur['quadmodl'+typemodl][b][p][j][:, 1]+gdat.dicterrr['amplnigh'][0, 0]-1), \
                                                                                                                    color='b', lw=3, label='Model')
                    
                    axis.plot(gdat.arrypcur['quadmodlplan'+typemodl][b][p][j][:, 0], 1e6*(gdat.arrypcur['quadmodlplan'+typemodl][b][p][j][:, 1]), \
                                                                                                                  color='g', label='Planetary', lw=1, ls='--')
                    
                    axis.plot(gdat.arrypcur['quadmodlbeam'+typemodl][b][p][j][:, 0], 1e6*(gdat.arrypcur['quadmodlbeam'+typemodl][b][p][j][:, 1]), \
                                                                                                          color='m', label='Beaming', lw=2, ls='--')
                    
                    axis.plot(gdat.arrypcur['quadmodlelli'+typemodl][b][p][j][:, 0], 1e6*(gdat.arrypcur['quadmodlelli'+typemodl][b][p][j][:, 1]), \
                                                                                                          color='r', label='Ellipsoidal variation', lw=2, ls='--')
                    
                    axis.plot(gdat.arrypcur['quadmodlstel'+typemodl][b][p][j][:, 0], 1e6*(gdat.arrypcur['quadmodlstel'+typemodl][b][p][j][:, 1]-1.), \
                                                                                                          color='orange', label='Stellar baseline', lw=2, ls='--')
                    
                    axis.set_ylim(ylimpcur)
                    axis.set_ylabel('Relative flux [ppm]')
                    axis.set_xlabel('Phase')
                    axis.legend(ncol=3)
                    plt.tight_layout()
                    path = gdat.pathalle[typemodl] + 'pcur_comp_%s_%s_%s.%s' % (typemodl, gdat.strgtarg, gdat.typeprioplan, gdat.typefileplot)
                    if gdat.typeverb > 0:
                        print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()

                    # plot the phase curve with samples
                    figr, axis = plt.subplots(figsize=gdat.figrsizeydob)
                    axis.errorbar(gdat.arrypcur['quadbdtrbindtotl'][b][p][j][:, 0], \
                                (gdat.arrypcur['quadbdtrbindtotl'][b][p][j][:, 1] + gdat.dicterrr['amplnigh'][0, 0] - 1) * 1e6, \
                                                 yerr=1e6*gdat.arrypcur['quadbdtrbindtotl'][b][p][j][:, 2], color='k', marker='o', ls='', markersize=2, lw=1)
                    for ii, i in enumerate(gdat.indxsampplot):
                        axis.plot(gdat.arrypcur['quadmodl'+typemodl][b][p][j][:, 0], \
                                                    1e6 * (gdat.listarrypcur['quadmodl'+typemodl][b][p][j][ii, :] + gdat.dicterrr['amplnigh'][0, 0] - 1.), \
                                                                                                                                      alpha=0.1, color='b')
                    axis.set_ylabel('Relative flux [ppm]')
                    axis.set_xlabel('Phase')
                    axis.set_ylim(ylimpcur)
                    plt.tight_layout()
                    path = gdat.pathalle[typemodl] + 'pcur_samp_%s_%s_%s.%s' % (typemodl, gdat.strgtarg, gdat.typeprioplan, gdat.typefileplot)
                    if gdat.typeverb > 0:
                        print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()

                    # plot all along with residuals
                    #figr, axis = plt.subplots(3, 1, figsize=gdat.figrsizeydob)
                    #axis.errorbar(gdat.arrypcur['quadbdtrbindtotl'][b][p][j][:, 0], (gdat.arrypcur['quadbdtrbindtotl'][b][p][j][:, 1]) * 1e6, \
                    #                       yerr=1e6*gdat.arrypcur['quadbdtrbindtotl'][b][p][j][:, 2], color='k', marker='o', ls='', markersize=2, lw=1)
                    #for kk, k in enumerate(gdat.indxsampplot):
                    #    axis.plot(gdat.meanphasfine[indxphasfineotprleft], (listmodltotl[k, indxphasfineotprleft] - listoffs[k]) * 1e6, \
                    #                                                                                                            alpha=0.1, color='b')
                    #    axis.plot(gdat.meanphasfine[indxphasfineotprrght], (listmodltotl[k, indxphasfineotprrght] - listoffs[k]) * 1e6, \
                    #                                                                                                            alpha=0.1, color='b')
                    #axis.set_ylabel('Relative flux - 1 [ppm]')
                    #axis.set_xlabel('Phase')
                    #plt.tight_layout()
                    #path = gdat.pathalle[typemodl] + 'pcur_resi_%s_%s_%s.%s' % (typemodl, gdat.strgtarg, gdat.typeprioplan, gdat.typefileplot)
                    #print('Writing to %s...' % path)
                    #plt.savefig(path)
                    #plt.close()

                    # write to text file
                    fileoutp = open(gdat.pathalle[typemodl] + 'post_pcur_%s_tabl.csv' % (typemodl), 'w')
                    for strgfeat in gdat.dictlist:
                        if gdat.dictlist[strgfeat].ndim == 2:
                            for j in gdat.indxcomp:
                                fileoutp.write('%s,%s,%g,%g,%g,%g,%g\\\\\n' % (strgfeat, gdat.liststrgcomp[j], gdat.dictlist[strgfeat][0, j], gdat.dictlist[strgfeat][1, j], \
                                                                            gdat.dictlist[strgfeat][2, j], gdat.dicterrr[strgfeat][1, j], gdat.dicterrr[strgfeat][2, j]))
                        else:
                            fileoutp.write('%s,,%g,%g,%g,%g,%g\\\\\n' % (strgfeat, gdat.dictlist[strgfeat][0], gdat.dictlist[strgfeat][1], \
                                                                            gdat.dictlist[strgfeat][2], gdat.dicterrr[strgfeat][1], gdat.dicterrr[strgfeat][2]))
                        #fileoutp.write('\\\\\n')
                    fileoutp.close()
                    
                    fileoutp = open(gdat.pathalle[typemodl] + 'post_pcur_%s_cmnd.csv' % (typemodl), 'w')
                    for strgfeat in gdat.dictlist:
                        if gdat.dictlist[strgfeat].ndim == 2:
                            for j in gdat.indxcomp:
                                fileoutp.write('%s,%s,$%.3g \substack{+%.3g \\\\ -%.3g}$\\\\\n' % (strgfeat, gdat.liststrgcomp[j], gdat.dicterrr[strgfeat][0, j], \
                                                                                            gdat.dicterrr[strgfeat][1, j], gdat.dicterrr[strgfeat][2, j]))
                        else:
                            fileoutp.write('%s,,$%.3g \substack{+%.3g \\\\ -%.3g}$\\\\\n' % (strgfeat, gdat.dicterrr[strgfeat][0], \
                                                                                                        gdat.dicterrr[strgfeat][1], gdat.dicterrr[strgfeat][2]))
                        #fileoutp.write('\\\\\n')
                    fileoutp.close()

                if gdat.labltarg == 'WASP-121' and typemodl == '0003':
                    
                    # wavelength axis
                    gdat.conswlentmpt = 0.0143877735e6 # [um K]

                    minmalbg = min(np.amin(gdat.dictlist['albginfo']), np.amin(gdat.dictlist['albg']))
                    maxmalbg = max(np.amax(gdat.dictlist['albginfo']), np.amax(gdat.dictlist['albg']))
                    binsalbg = np.linspace(minmalbg, maxmalbg, 100)
                    meanalbg = (binsalbg[1:] + binsalbg[:-1]) / 2.
                    pdfnalbg = tdpy.retr_kdegpdfn(gdat.dictlist['albg'][:, 0], binsalbg, 0.02)
                    pdfnalbginfo = tdpy.retr_kdegpdfn(gdat.dictlist['albginfo'][:, 0], binsalbg, 0.02)
                    
                    figr, axis = plt.subplots(figsize=gdat.figrsizeydob)
                    axis.plot(meanalbg, pdfnalbg, label='TESS only', lw=2)
                    axis.plot(meanalbg, pdfnalbginfo, label='TESS + ATMO', lw=2)
                    axis.set_xlabel('$A_g$')
                    axis.set_ylabel('$P(A_g)$')
                    axis.legend()
                    axis.set_xlim([0, None])
                    plt.subplots_adjust()
                    path = gdat.pathalle[typemodl] + 'pdfn_albg_%s_%s.%s' % (gdat.strgtarg, gdat.typeprioplan, gdat.typefileplot)
                    if gdat.typeverb > 0:
                        print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()
                
                    figr, axis = plt.subplots(figsize=gdat.figrsizeydob)
                    axis.hist(gdat.dictlist['albg'][:, 0], label='TESS only', bins=binsalbg)
                    axis.hist(gdat.dictlist['albginfo'][:, 0], label='TESS + ATMO', bins=binsalbg)
                    axis.set_xlabel('$A_g$')
                    axis.set_ylabel('$N(A_g)$')
                    axis.legend()
                    plt.subplots_adjust()
                    path = gdat.pathalle[typemodl] + 'hist_albg_%s_%s.%s' % (gdat.strgtarg, gdat.typeprioplan, gdat.typefileplot)
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
                    listlablpara = [['$\kappa_{IR}$', ''], ['$\gamma$', ''], ['$\psi$', ''], ['[M/H]', ''], \
                                                                                                    ['[C/H]', ''], ['[O/H]', '']]
                    tdpy.mcmc.plot_grid(gdat.pathalle[typemodl], 'post_atmo', listsampatmo, listlablpara, plotsize=2.5)
   
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
                    figr, axis = plt.subplots(figsize=gdat.figrsizeydob)
                    gdat.kdegpsii = tdpy.retr_kdeg(gdat.listpsii, gdat.meanpsii, gdat.kdegstdvpsii)
                    axis.plot(gdat.meanpsii, gdat.kdegpsii)
                    axis.set_xlabel('$\psi$')
                    axis.set_ylabel('$K_\psi$')
                    plt.subplots_adjust()
                    path = gdat.pathalle[typemodl] + 'kdeg_psii_%s_%s.%s' % (gdat.strgtarg, gdat.typeprioplan, gdat.typefileplot)
                    if gdat.typeverb > 0:
                        print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()
                
                    # use psi posterior to infer Bond albedo and heat circulation efficiency
                    numbsampwalk = 10000
                    numbsampburnwalk = 1000
                    listlablpara = [['$A_b$', ''], ['$E$', ''], [r'$\varepsilon$', '']]
                    listscalpara = ['self', 'self', 'self']
                    listminmpara = np.array([0., 0., 0.])
                    listmaxmpara = np.array([1., 1., 1.])
                    strgextn = 'albbepsi'
                    listpostheat = tdpy.mcmc.samp(gdat, gdat.pathalle[typemodl], numbsampwalk, retr_llik_albbepsi, \
                                                  listlablpara, listscalpara, listminmpara, listmaxmpara, \
                                                  numbsampburnwalk=numbsampburnwalk, strgextn=strgextn)

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
                    axistwin.plot(gdat.meanwlenband, gdat.thptband, color='grey', ls='--', label='TESS')
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
                    path = gdat.pathalle[typemodl] + 'spec_%s_%s.%s' % (gdat.strgtarg, gdat.typeprioplan, gdat.typefileplot)
                    if gdat.typeverb > 0:
                        print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()
                    
                    # get contribution function
                    path = gdat.pathdatatarg + 'ascii_output/ContribFuncArr.txt'
                    print('Reading from %s...' % path)
                    ctrb = np.loadtxt(path)
                    presctrb = ctrb[0, :]
                    # interpolate the throughput
                    gdat.thptbandctrb = scipy.interpolate.interp1d(gdat.meanwlenband, gdat.thptband, fill_value=0, bounds_error=False)(wlenctrb)
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
                    path = gdat.pathalle[typemodl] + 'ptem_%s_%s.%s' % (gdat.strgtarg, gdat.typeprioplan, gdat.typefileplot)
                    if gdat.typeverb > 0:
                        print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()
  

def plot_orbt( \
              # path to write the plot
              path, \
              # radius of the planets [R_E]
              radiplan, \
              # sum of radius of planet and star divided by the semi-major axis
              rsma, \
              # epoc of the planets [BJD]
              epoc, \
              # orbital periods of the planets [days]
              peri, \
              # cosine of the inclination
              cosi, \
              # type of visualization: 
              ## 'realblac': dark background, black planets
              ## 'realblaclcur': dark backgound, luminous planets, with light curves 
              ## 'realcolrlcur': dark background, colored planets, with light curves 
              ## 'cartcolr': bright background, colored planets
              typevisu, \
              
              # radius of the star [R_S]
              radistar=1., \
              # mass of the star [M_S]
              massstar=1., \
              # Boolean flag to produce an animation
              boolanim=False, \

              # angle of view with respect to the orbital plane [deg]
              anglpers=5., \

              # size of the figure
              sizefigr=(8, 8), \
              listcolrcomp=None, \
              liststrgcomp=None, \
              boolsingside=True, \
              typefileplot='pdf', \

              # verbosity level
              typeverb=1, \
             ):

    from allesfitter.v2.classes import allesclass2
    from allesfitter.v2.translator import translate
    
    dictfact = ephesus.retr_factconv()
    
    mpl.use('Agg')

    numbcomp = len(radiplan)
    
    if isinstance(radiplan, list):
        radiplan = np.array(radiplan)

    if isinstance(rsma, list):
        rsma = np.array(rsma)

    if isinstance(epoc, list):
        epoc = np.array(epoc)

    if isinstance(peri, list):
        peri = np.array(peri)

    if isinstance(cosi, list):
        cosi = np.array(cosi)

    if listcolrcomp is None:
        listcolrcomp = retr_listcolrcomp(numbcomp)

    if liststrgcomp is None:
        liststrgcomp = retr_liststrgcomp(numbcomp)
    
    # semi-major axes of the planets [AU]
    smax = (radiplan / dictfact['rsre'] + radistar) / dictfact['aurs'] / rsma
    indxcomp = np.arange(numbcomp)
    
    # perspective factor
    factpers = np.sin(anglpers * np.pi / 180.)

    ## scale factor for the star
    factstar = 5.
    
    ## scale factor for the planets
    factplan = 20.
    
    # maximum y-axis value
    maxmyaxi = 0.05

    if typevisu == 'cartmerc':
        # Mercury
        smaxmerc = 0.387 # [AU]
        radiplanmerc = 0.3829 # [R_E]
    
    # scaled radius of the star [AU]
    radistarscal = radistar / dictfact['aurs'] * factstar
    
    time = np.arange(0., 30., 2. / 60. / 24.)
    numbtime = time.size
    indxtime = np.arange(numbtime)
   
    if boolanim:
        numbiter = min(500, numbtime)
    else:
        numbiter = 1
    indxiter = np.arange(numbiter)
    
    xposmaxm = smax
    yposmaxm = factpers * xposmaxm
    numbtimequad = 10
    
    if typevisu == 'realblaclcur':
        numbtimespan = 100

    # get transit model based on TESS ephemerides
    rrat = radiplan / radistar
    alles = allesclass2()
    
    rflxtranmodl = ephesus.retr_rflxmodl(time, peri, epoc, radiplan, radistar, rsma, cosi) - 1.
    
    #alles.settings = {'inst_phot':['telescope'], 'host_ld_law_telescope':'quad'}
    #alles.settings['companions_phot'] = []
    #alles.params['host_ldc_q1_telescope'] = 0.5
    #alles.params['host_ldc_q2_telescope'] = 0.2
    #for j, strgplan in enumerate(liststrgcomp):
    #    alles.settings['companions_phot'].append(strgplan)
    #    alles.params['%s_rr' % strgplan] = rrat[j]
    #    alles.params['%s_rsuma' % strgplan] = rsma[j]
    #    alles.params['%s_epoch' % strgplan] = epoc[j]
    #    alles.params['%s_period' % strgplan] = peri[j]
    #    alles.params['%s_cosi' % strgplan] = cosi[j]
    #alles.params_host = {'R_host':radistar, 'M_host':massstar}
    #alles.fill()
    #model_flux = alles.generate_model(time, inst='telescope', key='flux')
    lcur = rflxtranmodl + np.random.randn(numbtime) * 1e-6
    ylimrflx = [np.amin(lcur), np.amax(lcur)]
    
    phas = np.random.rand(numbcomp)[None, :] * 2. * np.pi + 2. * np.pi * time[:, None] / peri[None, :]
    yposelli = yposmaxm[None, :] * np.sin(phas)
    xposelli = xposmaxm[None, :] * np.cos(phas)
    
    # time indices for iterations
    indxtimeiter = np.linspace(0., numbtime - numbtime / numbiter, numbiter).astype(int)
    
    if typevisu.startswith('cart'):
        colrstar = 'k'
        colrface = 'w'
        plt.style.use('default')
    else:
        colrface = 'k'
        colrstar = 'w'
        plt.style.use('dark_background')
    
    if boolanim:
        cmnd = 'convert -delay 5'
        listpathtemp = []
    for k in indxiter:
        
        if typevisu == 'realblaclcur':
            numbrows = 2
        else:
            numbrows = 1
        figr, axis = plt.subplots(figsize=sizefigr)

        ### lower half of the star
        w1 = mpl.patches.Wedge((0, 0), radistarscal, 180, 360, fc=colrstar, zorder=1, edgecolor=colrstar)
        axis.add_artist(w1)
        
        for jj, j in enumerate(indxcomp):
            xposellishft = np.roll(xposelli[:, j], -indxtimeiter[k])[-numbtimequad:][::-1]
            yposellishft = np.roll(yposelli[:, j], -indxtimeiter[k])[-numbtimequad:][::-1]
        
            # trailing lines
            if typevisu.startswith('cart'):
                objt = retr_objtlinefade(xposellishft, yposellishft, colr=listcolrcomp[j], initalph=1., alphfinl=0.)
                axis.add_collection(objt)
            
            # add planets
            if typevisu.startswith('cart'):
                colrplan = listcolrcomp[j]
                # add planet labels
                axis.text(.6 + 0.03 * jj, 0.1, liststrgcomp[j], color=listcolrcomp[j], transform=axis.transAxes)
        
            if typevisu.startswith('real'):
                if typevisu == 'realillu':
                    colrplan = 'k'
                else:
                    colrplan = 'black'
            radi = radiplan[j] / dictfact['rsre'] / dictfact['aurs'] * factplan
            w1 = mpl.patches.Circle((xposelli[indxtimeiter[k], j], yposelli[indxtimeiter[k], j], 0), radius=radi, color=colrplan, zorder=3)
            axis.add_artist(w1)
            
        ## upper half of the star
        w1 = mpl.patches.Wedge((0, 0), radistarscal, 0, 180, fc=colrstar, zorder=4, edgecolor=colrstar)
        axis.add_artist(w1)
        
        if typevisu == 'cartmerc':
            ## add Mercury
            axis.text(.387, 0.01, 'Mercury', color='grey', ha='right')
            radi = radiplanmerc / dictfact['rsre'] / dictfact['aurs'] * factplan
            w1 = mpl.patches.Circle((smaxmerc, 0), radius=radi, color='grey')
            axis.add_artist(w1)
        
        # temperature axis
        #axistwin = axis.twiny()
        ##axistwin.set_xlim(axis.get_xlim())
        #xpostemp = axistwin.get_xticks()
        ##axistwin.set_xticks(xpostemp[1:])
        #axistwin.set_xticklabels(['%f' % tmpt for tmpt in listtmpt])
        
        # temperature contours
        #for tmpt in [500., 700,]:
        #    smaj = tmpt
        #    axis.axvline(smaj, ls='--')
        
        axis.get_yaxis().set_visible(False)
        axis.set_aspect('equal')
        
        if typevisu == 'cartmerc':
            maxmxaxi = max(1.2 * np.amax(smax), 0.4)
        else:
            maxmxaxi = 1.2 * np.amax(smax)
        
        if boolsingside:
            minmxaxi = 0.
        else:
            minmxaxi = -maxmxaxi

        axis.set_xlim([minmxaxi, maxmxaxi])
        axis.set_ylim([-maxmyaxi, maxmyaxi])
        axis.set_xlabel('Distance from the star [AU]')
        
        if typevisu == 'realblaclcur':
        #if False and typevisu == 'realblaclcur':
            print('indxtimeiter[k]')
            print(indxtimeiter[k])
            minmindxtime = max(0, indxtimeiter[k]-numbtimespan)
            print('minmindxtime')
            print(minmindxtime)
            xtmp = time[minmindxtime:indxtimeiter[k]]
            if len(xtmp) == 0:
                continue
            print('xtmp')
            print(xtmp)
            timescal = 2 * maxmxaxi * (xtmp - np.amin(xtmp)) / (np.amax(xtmp) - np.amin(xtmp)) - maxmxaxi
            print('timescal')
            print(timescal)
            axis.scatter(timescal, 10000. * lcur[minmindxtime:indxtimeiter[k]] + maxmyaxi * 0.8, rasterized=True, color='cyan', s=0.5)
            print('time[minmindxtime:indxtimeiter[k]]')
            summgene(time[minmindxtime:indxtimeiter[k]])
            print('lcur[minmindxtime:indxtimeiter[k]]')
            summgene(lcur[minmindxtime:indxtimeiter[k]])
            print('')

        #plt.subplots_adjust()
        #axis.legend()
        
        if boolanim:
            pathtemp = '%s_%s_%04d.%s' % (path, typevisu, k, typefileplot)
        else:
            pathtemp = '%s_%s.%s' % (path, typevisu, typefileplot)
        print('Writing to %s...' % pathtemp)
        plt.savefig(pathtemp)
        plt.close()
        
        if boolanim:
            listpathtemp.append(pathtemp)
            cmnd += ' %s' % pathtemp 
    if boolanim:
        cmnd += ' %s_%s.gif' % (path, typevisu)
        os.system(cmnd)
        for pathtemp in listpathtemp:
            cmnd = 'rm %s' % pathtemp
            os.system(cmnd)


def plot_prop(gdat, strgpdfn):
    
    print('Plotting properties for strgpdfn: %s' % strgpdfn)
        
    pathimagfeatplan = getattr(gdat, 'pathimagfeatplan' + strgpdfn)
    pathimagdataplan = getattr(gdat, 'pathimagdataplan' + strgpdfn)
    pathimagfeatsyst = getattr(gdat, 'pathimagfeatsyst' + strgpdfn)
    
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
    for jj, j in enumerate(gdat.indxcomp):
        xposlowr = gdat.dictfact['rjre'] * gdat.dictpost['radiplan'][0, j]
        xposuppr = gdat.dictfact['rjre'] * gdat.dictpost['radiplan'][2, j]
        axis.axvspan(xposlowr, xposuppr, alpha=0.5, color=gdat.listcolrcomp[j])
        axis.axvline(gdat.dictfact['rjre'] * gdat.dicterrr['radiplan'][0, j], color=gdat.listcolrcomp[j], ls='--', label=gdat.liststrgcomp[j])
        axis.text(0.7, 0.9 - jj * 0.07, r'\textbf{%s}' % gdat.liststrgcomp[j], color=gdat.listcolrcomp[j], \
                                                                                    va='center', ha='center', transform=axis.transAxes)
    xerr = (timeoccu[1:] - timeoccu[:-1]) / 2.
    xerr = np.concatenate([xerr[0, None], xerr])
    axis.errorbar(timeoccu, occumean, yerr=occuyerr, xerr=xerr, color='black', ls='', marker='o', lw=1, zorder=10)
    axis.set_xlabel('Radius [$R_E$]')
    axis.set_ylabel('Occurrence rate of planets per star')
    plt.subplots_adjust(bottom=0.2)
    plt.subplots_adjust(left=0.2)
    path = pathimagfeatplan + 'occuradi_%s_%s.%s' % (gdat.strgtarg, strgpdfn, gdat.typefileplot)
    print('Writing to %s...' % path)
    plt.savefig(path)
    plt.close()
  
    # orbital depiction
    ## 'realblac': dark background, black planet
    ## 'realblaclcur': dark backgound, black planet, light curve
    ## 'realcolrlcur': dark backgound, colored planet, light curve
    ## 'cartcolrlcur': cartoon backgound, colored planet
    path = pathimagfeatplan + 'orbt_%s_%s' % (gdat.strgtarg, strgpdfn)
    path = gdat.pathimagtarg + 'orbt'
    listtypevisu = ['realblac', 'realblaclcur', 'realcolrlcur', 'cartcolrlcur']
    listtypevisu = ['realblaclcur']
    
    for typevisu in listtypevisu:
        
        plot_orbt( \
                            path, \
                            gdat.dicterrr['radiplan'][0, :], \
                            gdat.dicterrr['rsma'][0, :], \
                            gdat.dicterrr['epoc'][0, :], \
                            gdat.dicterrr['peri'][0, :], \
                            gdat.dicterrr['cosi'][0, :], \
                            typevisu, \
                            radistar=gdat.radistar, \
                            sizefigr=gdat.figrsizeydob, \
                            typefileplot=gdat.typefileplot, \
                            boolsingside=False, \
                            boolanim=gdat.boolanimorbt, \
                            #typefileplot='png', \
                           )
    
    for strgpopl in gdat.liststrgpopl:
        
        if strgpopl == 'exar':
            dictpopl = gdat.dictexar
        else:
            dictpopl = gdat.dictexof
        
        numbcomppopl = dictpopl['radiplan'].size
        indxtargpopl = np.arange(numbcomppopl)

        ### TSM and ESM
        numbsamppopl = 100
        dictlistplan = dict()
        for strgfeat in gdat.listfeatstarpopl:
            dictlistplan[strgfeat] = np.zeros((numbsamppopl, dictpopl['massplan'].size)) + np.nan
            for k in range(dictpopl[strgfeat].size):
                meanvarb = dictpopl[strgfeat][k]
                if not np.isfinite(meanvarb):
                    continue
                if np.isfinite(dictpopl['stdv' + strgfeat][k]):
                    stdvvarb = dictpopl['stdv' + strgfeat][k]
                else:
                    stdvvarb = 0.
                
                dictlistplan[strgfeat][:, k] = scipy.stats.truncnorm.rvs(-meanvarb/stdvvarb, np.inf, size=numbsamppopl) * stdvvarb + meanvarb
                dictlistplan[strgfeat][:, k] /= np.mean(dictlistplan[strgfeat][:, k])
                dictlistplan[strgfeat][:, k] *= meanvarb
                
                #print('stdvvarb')
                #print(stdvvarb)
                #print('std(dictlistplan[strgfeat][:, k])')
                #print(np.std(dictlistplan[strgfeat][:, k]))
                #print('')
                #if k == 3026:
                #    print('strgfeat')
                #    print(strgfeat)
                #    print('meanvarb')
                #    print(meanvarb)
                #    print('stdvvarb')
                #    print(stdvvarb)
                #    print('dictlistplan[strgfeat][:, k]')
                #    print(dictlistplan[strgfeat][:, k])
                #    print('dictlistplan[strgfeat][:, k]')
                #    summgene(dictlistplan[strgfeat][:, k])
                #
                #if (dictlistplan[strgfeat][:, k] < 0).any():
                #    
                #    print('strgfeat')
                #    print(strgfeat)
                #    print('k')
                #    print(k)
                #    print('dictpopl[nameplan][k]')
                #    print(dictpopl['nameplan'][k])
                #    print('meanvarb')
                #    print(meanvarb)
                #    print('stdvvarb')
                #    print(stdvvarb)
                #    print('dictlistplan[strgfeat][:, k]')
                #    summgene(dictlistplan[strgfeat][:, k])
                #    raise Exception('')

        #### TSM
        listtsmm = ephesus.retr_tsmm(dictlistplan['radiplan'], dictlistplan['tmptplan'], dictlistplan['massplan'], \
                                                                                        dictlistplan['radistar'], dictlistplan['jmagsyst'])

        #### ESM
        listesmm = ephesus.retr_esmm(dictlistplan['tmptplan'], dictlistplan['tmptstar'], dictlistplan['radiplan'], dictlistplan['radistar'], \
                                                                                                                    dictlistplan['kmagsyst'])
        ## augment the 
        dictpopl['stdvtsmm'] = np.std(listtsmm, 0)
        dictpopl['tsmm'] = np.nanmedian(listtsmm, 0)
        dictpopl['stdvesmm'] = np.std(listesmm, 0)
        dictpopl['esmm'] = np.nanmedian(listesmm, 0)
        
        dictpopl['vesc'] = ephesus.retr_vesc(dictpopl['massplan'], dictpopl['radiplan'])
        dictpopl['vesc0060'] = dictpopl['vesc'] / 6.
        
        objticrs = astropy.coordinates.SkyCoord(ra=dictpopl['rascstar']*astropy.units.degree, \
                                               dec=dictpopl['declstar']*astropy.units.degree, frame='icrs')
        dictpopl['lgalstar'] = np.array([objticrs.galactic.l])[0, :]
        dictpopl['bgalstar'] = np.array([objticrs.galactic.b])[0, :]
        dictpopl['laecstar'] = np.array([objticrs.barycentricmeanecliptic.lon.degree])[0, :]
        dictpopl['beecstar'] = np.array([objticrs.barycentricmeanecliptic.lat.degree])[0, :]

        dictpopl['stnomass'] = dictpopl['massplan'] / dictpopl['stdvmassplan']

        dictpopl['boollive'] = ~dictpopl['boolfpos']
        dictpopl['boolterr'] = dictpopl['radiplan'] < 1.8
        dictpopl['boolhabicons'] = (dictpopl['inso'] < 1.01) & (dictpopl['inso'] > 0.35)
        dictpopl['boolhabiopti'] = (dictpopl['inso'] < 1.78) & (dictpopl['inso'] > 0.29)
        # unlocked
        dictpopl['boolunlo'] = np.log10(dictpopl['massstar']) < (-2 + 3 * (np.log10(dictpopl['smax']) + 1))
        # Earth as a transiting planet
        dictpopl['booleatp'] = abs(dictpopl['beecstar']) < 0.25

        ## Hill sphere
        ## angular momentum
    
        dictpopl['sage'] = 1. / 365.2422 / 24. / 3600. * (1. / 486.) * (0.008406 * dictpopl['smax'] / 0.027 / dictpopl['massstar']**(1. / 3.))**6
        dictpopl['timelock'] = (1. / 486.) * (dictpopl['smax'] / 0.027 / dictpopl['massstar']**(1. / 3.))**6

        #for strg in dictpopl.keys():
        #    print(strg)
        #    summgene(dictpopl[strg])
        #    print('')
        
        # from SETI
        dictpopl['metrplan'] = (0.99 * np.heaviside(dictpopl['numbcomptranstar'] - 2, 1.) + 0.01)
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
            strgstar = dictpopl['namesyst'][m]
            if not strgstar in liststrgstarcomp:
                indxexarstar = np.where(dictpopl['namesyst'] == strgstar)[0]
                if indxexarstar[0] != m:
                    raise Exception('')
                
                listperi = dictpopl['peri'][None, indxexarstar]
                if not np.isfinite(listperi).all() or np.where(listperi == 0)[0].size > 0:
                    liststrgstarcomp.append(strgstar)
                    continue
                intgreso, ratiperi = retr_reso(listperi)
                
                numbcomp = indxexarstar.size
                
                gdat.listratiperi.append(ratiperi[0, :, :][np.triu_indices(numbcomp, k=1)])
                gdat.intgreso.append(intgreso)
                
                liststrgstarcomp.append(strgstar)
        
        gdat.listratiperi = np.concatenate(gdat.listratiperi)
        figr, axis = plt.subplots(figsize=gdat.figrsize)
        bins = np.linspace(1., 10., 400)
        axis.hist(gdat.listratiperi, bins=bins, rwidth=1)
        if gdat.numbcomp > 1:
            ## this system
            for j in gdat.indxcomp:
                for jj in gdat.indxcomp:
                    if gdat.dicterrr['peri'][0, j] > gdat.dicterrr['peri'][0, jj]:
                        ratiperi = gdat.dicterrr['peri'][0, j] / gdat.dicterrr['peri'][0, jj]
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
        path = pathimagfeatplan + 'histratiperi_%s_%s_%s.%s' % (gdat.strgtarg, strgpdfn, strgpopl, gdat.typefileplot)
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()
        
        # metastable helium absorption
        path = gdat.pathbasemile + '/data/wasp107b_transmission_spectrum.dat'
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
                radiplan = gdat.dicterrr['radiplan'][0, :]
                massplan = gdat.dicterrr['massplanused'][0, :]
                tmptplan = gdat.dicterrr['tmptplan'][0, :]
                duratranplan = gdat.dicterrr['duratrantotl'][0, :]
                radistar = gdat.radistar
                jmagsyst = gdat.jmagsyst
            else:
                print('WASP-107')
                radiplan = 0.924 * gdat.dictfact['rjre']
                massplan = 0.119
                tmptplan = 736
                radistar = 0.66 # [R_S]
                jmagsyst = jmagsystwasp0107
                duratranplan = duratranplanwasp0107
            scalheig = ephesus.retr_scalheig(tmptplan, massplan, radiplan)
            deptscal = 1e3 * 2. * radiplan * scalheig / radistar**2 # [ppt]
            dept = 80. * deptscal
            factstdv = np.sqrt(10**((-jmagsystwasp0107 + jmagsyst) / 2.5) * duratranplanwasp0107 / duratranplan)
            stdvnirsthis = factstdv * stdvnirs
            for b in np.arange(1, 6):
                stdvnirsscal = stdvnirsthis / np.sqrt(float(b))
                sigm = dept / stdvnirsscal
        
            print('radiplan')
            print(radiplan)
            print('massplan')
            print(massplan)
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
            print('dept')
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
        path = pathimagdataplan + 'dept_%s_%s.%s' % (gdat.strgtarg, strgpdfn, gdat.typefileplot)
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
                strgvarbmagt = 'rvsascal_vmag'
                lablxaxi = '$K^{\prime}_{V}$'
                varbtarg = np.sqrt(10**(-gdat.vmagsyst / 2.5)) / gdat.massstar**(2. / 3.)
                varb = np.sqrt(10**(-dictpopl['vmagsyst'] / 2.5)) / dictpopl['massstar']**(2. / 3.)
            if b == 3:
                strgvarbmagt = 'rvsascal_jmag'
                lablxaxi = '$K^{\prime}_{J}$'
                varbtarg = np.sqrt(10**(-gdat.vmagsyst / 2.5)) / gdat.massstar**(2. / 3.)
                varb = np.sqrt(10**(-dictpopl['jmagsyst'] / 2.5)) / dictpopl['massstar']**(2. / 3.)
            for a in range(3):
                figr, axis = plt.subplots(figsize=gdat.figrsize)
                if a == 0:
                    indx = np.where(dictpopl['boolfrst'])[0]
                if a == 1:
                    indx = np.where(dictpopl['boolfrst'] & (dictpopl['numbcompstar'] > 3))[0]
                if a == 2:
                    indx = np.where(dictpopl['boolfrst'] & (dictpopl['numbcomptranstar'] > 3))[0]
                
                if (b == 2 or b == 3):
                    normfact = max(varbtarg, np.nanmax(varb[indx]))
                else:
                    normfact = 1.
                varbtargnorm = varbtarg / normfact
                varbnorm = varb[indx] / normfact
                axis.scatter(varbnorm, dictpopl['numbcompstar'][indx], s=1, color='black')
                
                indxsort = np.argsort(varbnorm)
                if b == 2 or b == 3:
                    indxsort = indxsort[::-1]

                listnameaddd = []
                cntr = 0
                maxmnumbname = min(5, varbnorm.size)
                while True:
                    k = indxsort[cntr]
                    nameadd = dictpopl['namesyst'][indx][k]
                    if not nameadd in listnameaddd:
                        axis.text(varbnorm[k], dictpopl['numbcompstar'][indx][k] + 0.5, nameadd, size=6, \
                                                                                                va='center', ha='right', rotation=45)
                        listnameaddd.append(nameadd)
                    cntr += 1
                    if len(listnameaddd) == maxmnumbname: 
                        break
                axis.scatter(varbtargnorm, gdat.numbcomp, s=5, color='black', marker='x')
                axis.text(varbtargnorm, gdat.numbcomp + 0.5, gdat.labltarg, size=8, color='black', \
                                                                                            va='center', ha='center', rotation=45)
                axis.set_ylabel(r'Number of transiting planets')
                axis.set_xlabel(lablxaxi)
                plt.subplots_adjust(bottom=0.2)
                plt.subplots_adjust(left=0.2)
                path = pathimagfeatsyst + '%snumb_%s_%s_%s_%d.%s' % (strgvarbmagt, gdat.strgtarg, strgpdfn, strgpopl, a, gdat.typefileplot)
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()

                figr, axis = plt.subplots(figsize=gdat.figrsize)
                axis.hist(varbnorm, 50)
                axis.axvline(varbtargnorm, color='black', ls='--')
                axis.text(0.3, 0.9, gdat.labltarg, size=8, color='black', transform=axis.transAxes, va='center', ha='center')
                axis.set_ylabel(r'Number of systems')
                axis.set_xlabel(lablxaxi)
                plt.subplots_adjust(bottom=0.2)
                plt.subplots_adjust(left=0.2)
                path = pathimagfeatsyst + 'hist_%s_%s_%s_%s_%d.%s' % (strgvarbmagt, gdat.strgtarg, strgpdfn, strgpopl, a, gdat.typefileplot)
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()
            
        if not gdat.boolfeatplan:
            return
        
        # planet feature distribution plots
        print('Will make the relevant distribution plots...')
        numbcomptext = min(10, numbcomppopl)
        liststrgtext = ['notx', 'text']
        
        # first is x-axis, second is y-axis
        liststrgfeatpairplot = [ \
                            #['smax', 'massstar'], \
                            #['rascstar', 'declstar'], \
                            #['lgalstar', 'bgalstar'], \
                            #['laecstar', 'beecstar'], \
                            #['distsyst', 'vmagsyst'], \
                            #['inso', 'radiplan'], \
                            ['radiplan', 'tmptplan'], \
                            ['radiplan', 'tsmm'], \
                            #['radiplan', 'esmm'], \
                            ['tmptplan', 'tsmm'], \
                            #['tagestar', 'vesc'], \
                            ['tmptplan', 'vesc0060'], \
                            #['radiplan', 'tsmm'], \
                            #['tmptplan', 'vesc'], \
                            #['peri', 'inso'], \
                            #['radistar', 'radiplan'], \
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
        indxcompfilt['totl'] = indxtargpopl
        
        #indxcompfilt['tran'] = np.where(dictpopl['booltran'])[0]
        strgcuttmain = 'totl'
        
        #indxcompfilt['box1'] = np.where((dictpopl['radiplan'] < 3.5) & (dictpopl['radiplan'] > 3.) & (dictpopl['tmptplan'] > 300) & \
        #                                                                                            (dictpopl['tmptplan'] < 500) & dictpopl['booltran'])[0]
        #indxcompfilt['box2'] = np.where((dictpopl['radiplan'] < 2.5) & (dictpopl['radiplan'] > 2.) & (dictpopl['tmptplan'] > 800) & \
        #                                                                                            (dictpopl['tmptplan'] < 1000) & dictpopl['booltran'])[0]
        #indxcompfilt['box2'] = np.where((dictpopl['radiplan'] < 3.) & (dictpopl['radiplan'] > 2.5) & (dictpopl['tmptplan'] > 1000) & \
        #                                                                                            (dictpopl['tmptplan'] < 1400) & dictpopl['booltran'])[0]
        #indxcompfilt['box3'] = np.where((dictpopl['radiplan'] < 3.) & (dictpopl['radiplan'] > 2.5) & (dictpopl['tmptplan'] > 1000) & \
        #                                                                                            (dictpopl['tmptplan'] < 1400) & dictpopl['booltran'])[0]
        #indxcompfilt['r4tr'] = np.where((dictpopl['radiplan'] < 4) & dictpopl['booltran'])[0]
        #indxcompfilt['r4trtess'] = np.where((dictpopl['radiplan'] < 4) & dictpopl['booltran'] & \
        #                                                                (dictpopl['facidisc'] == 'Transiting Exoplanet Survey Satellite (TESS)'))[0]

        #indxcompfilt['r154'] = np.where((dictpopl['radiplan'] > 1.5) & (dictpopl['radiplan'] < 4))[0]
        #indxcompfilt['r204'] = np.where((dictpopl['radiplan'] > 2) & (dictpopl['radiplan'] < 4))[0]
        #indxcompfilt['rb24'] = np.where((dictpopl['radiplan'] < 4) & (dictpopl['radiplan'] > 2.))[0]
        #indxcompfilt['gmtr'] = np.where(np.isfinite(stnomass) & (stnomass > 5) & (dictpopl['booltran']))[0]
        #indxcompfilt['tran'] = np.where(dictpopl['booltran'])[0]
        #indxcompfilt['mult'] = np.where(dictpopl['numbcomptranstar'] > 3)[0]
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
        dicttemptotl = dict()
        
        liststrgcutt = indxcompfilt.keys()
        
        liststrgvarb = [ \
                        'peri', 'inso', 'vesc0060', 'massplan', \
                        'metrhzon', 'metrterr', 'metrplan', 'metrunlo', 'metrseti', \
                        'smax', \
                        'tmptstar', \
                        'rascstar', 'declstar', \
                        'laecstar', 'beecstar', \
                        'radistar', \
                        'massstar', \
                        'metastar', \
                        'radiplan', 'tmptplan', \
                        'metrhabi', 'metrplan', \
                        'lgalstar', 'bgalstar', 'distsyst', 'vmagsyst', \
                        'tsmm', 'esmm', \
                        'vsiistar', 'projoblq', \
                        'jmagsyst', \
                        'tagestar', \
                       ]

        listlablvarb = [ \
                        ['P', 'days'], ['F', '$F_E$'], ['$v_{esc}^\prime$', 'kms$^{-1}$'], ['$M_p$', '$M_E$'], \
                        [r'$\rho$_{HZ}', ''], [r'$\rho$_{T}', ''], [r'$\rho$_{MP}', ''], [r'$\rho$_{TL}', ''], [r'$\rho$_{SETI}', ''], \
                        ['$a$', 'AU'], \
                        ['$T_{eff}$', 'K'], \
                        ['RA', 'deg'], ['Dec', 'deg'], \
                        ['Ec. lon.', 'deg'], ['Ec. lat.', 'deg'], \
                        ['$R_s$', '$R_S$'], \
                        ['$M_s$', '$M_S$'], \
                        ['[Fe/H]', 'dex'], \
                        ['$R_p$', '$R_E$'], ['$T_p$', 'K'], \
                        [r'$\rho_{SH}$', ''], [r'$\rho_{SP}$', ''], \
                        ['$l$', 'deg'], ['$b$', 'deg'], ['$d$', 'pc'], ['$V$', ''], \
                        ['TSM', ''], ['ESM', ''], \
                        ['$v$sin$i$', 'kms$^{-1}$'], ['$\lambda$', 'deg'], \
                        ['J', 'mag'], \
                        ['$t_\star$', 'Gyr'], \
                       ] 
        
        numbvarb = len(liststrgvarb)
        indxvarb = np.arange(numbvarb)
        listlablvarbtotl = []
        listscalvarb = [[] for k in indxvarb]
        liststrgvarbself = ['tmptstar', 'rascstar', 'declstar', 'laecstar', 'beecstar', 'metastar', 'lgalstar', 'bgalstar', 'vmagsyst']
        for k in indxvarb:
            if listlablvarb[k][1] == '':
                listlablvarbtotl.append('%s' % (listlablvarb[k][0]))
            else:
                listlablvarbtotl.append('%s [%s]' % (listlablvarb[k][0], listlablvarb[k][1]))
            if liststrgvarb[k] in liststrgvarbself:
                listscalvarb[k] = 'self'
            else:
                listscalvarb[k] = 'logt'
        
        for k, strgxaxi in enumerate(liststrgvarb):
            for m, strgyaxi in enumerate(liststrgvarb):
                
                booltemp = False
                for l in indxpairfeatplot:
                    if strgxaxi == liststrgfeatpairplot[l][0] and strgyaxi == liststrgfeatpairplot[l][1]:
                        booltemp = True
                if not booltemp:
                    continue
                        
                for strgfeat, valu in dictpopl.items():
                    dicttemptotl[strgfeat] = np.concatenate([dictpopl[strgfeat][indxcompfilt[strgcuttmain]], gdat.dicterrr[strgfeat][0, :]])
                
                for strgcutt in liststrgcutt:
                    
                    # merge population with the target
                    for strgfeat, valu in dictpopl.items():
                        dicttemp[strgfeat] = np.concatenate([dictpopl[strgfeat][indxcompfilt[strgcutt]], gdat.dicterrr[strgfeat][0, :]])
                    
                    liststrgfeatcsvv = [ \
                                        #'inso', 'metrhzon', 'radiplan', 'metrterr', 'massstar', 'smax', 'metrunlo', 'distsyst', 'metrplan', 'metrseti', \
                                        'rascstar', 'declstar', 'radiplan', 'massplan', 'tmptplan', 'jmagsyst', 'radistar', 'tsmm', \
                                       ]
                    for y in indxstrgsort:
                        
                        if liststrgsort[y] != 'none':
                        
                            indxgood = np.where(np.isfinite(dicttemp[liststrgsort[y]]))[0]
                            indxsort = np.argsort(dicttemp[liststrgsort[y]][indxgood])[::-1]
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
                                
                                strgline = '%4d, %20s' % (cntr, dicttemp['nameplan'][l])
                                for strgfeatcsvv in liststrgfeatcsvv:
                                    strgline += ', %12.4g' % dicttemp[strgfeatcsvv][l]
                                strgline += '\n'
                                
                                objtfile.write(strgline)
                                cntr += 1 
                            print('Writing to %s...' % path)
                            objtfile.close()
                    
                        if gdat.boolplotprop:
                            # repeat, one without text, one with text
                            for b, strgtext in enumerate(liststrgtext):
                                figr, axis = plt.subplots(figsize=gdat.figrsize)
                                
                                if liststrgsort[y] != 'none' and strgtext != 'text' or liststrgsort[y] == 'none' and strgtext == 'text':
                                    continue
                        
                                ## population
                                if strgcutt == strgcuttmain:
                                    axis.errorbar(dicttemptotl[strgxaxi], dicttemptotl[strgyaxi], ls='', ms=1, marker='o', color='k')
                                else:
                                    axis.errorbar(dicttemptotl[strgxaxi], dicttemptotl[strgyaxi], ls='', ms=1, marker='o', color='k')
                                    axis.errorbar(dicttemp[strgxaxi], dicttemp[strgyaxi], ls='', ms=2, marker='o', color='r')
                                
                                ## this system
                                for j in gdat.indxcomp:
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

                                if strgxaxi == 'radiplan' and strgyaxi == 'massplan':
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
                                if listscalvarb[k] == 'logt':
                                    axis.set_xscale('log')
                                if listscalvarb[m] == 'logt':
                                    axis.set_yscale('log')
                                
                                plt.subplots_adjust(left=0.2)
                                plt.subplots_adjust(bottom=0.2)
                                pathimagfeatplan = getattr(gdat, 'pathimagfeatplan' + strgpdfn)
                                path = pathimagfeatplan + 'feat_%s_%s_%s_%s_%s_%s_%s_%s.%s' % \
                                             (strgxaxi, strgyaxi, gdat.strgtarg, strgpopl, strgcutt, \
                                                                                   strgtext, liststrgsort[y], strgpdfn, gdat.typefileplot)
                                print('Writing to %s...' % path)
                                plt.savefig(path)
                                plt.close()

   
def bdtr_wrap(gdat, b, p, y, epocmask, perimask, duramask, strgintp, strgoutp, strgtren, timescalbdtrspln):
    
    gdat.rflxbdtrregi, gdat.listindxtimeregi[b][p][y], gdat.indxtimeregioutt[b][p][y], gdat.listobjtspln[b][p][y], timeedge = \
                     ephesus.bdtr_tser(gdat.listarrytser[strgintp][b][p][y][:, 0], gdat.listarrytser[strgintp][b][p][y][:, 1], \
                                                epocmask=epocmask, perimask=perimask, duramask=duramask, \
                                                typeverb=gdat.typeverb, timebrek=gdat.timebrek, ordrspln=gdat.ordrspln, \
                                                timescalbdtrspln=timescalbdtrspln, timescalbdtrmedi=gdat.timescalbdtrmedi, \
                                                typebdtr=gdat.typebdtr)
    gdat.listarrytser[strgoutp][b][p][y] = np.copy(gdat.listarrytser[strgintp][b][p][y])
    gdat.listarrytser[strgoutp][b][p][y][:, 1] = np.concatenate(gdat.rflxbdtrregi)
    
    # trend
    gdat.listarrytser[strgtren][b][p][y] = np.copy(gdat.listarrytser[strgintp][b][p][y])
    rflxtren = []
    for k in range(len(gdat.rflxbdtrregi)):
        if gdat.listobjtspln[b][p][y][k] is None:
            rflxtren.append(np.zeros(gdat.listindxtimeregi[b][p][y][k].size) + \
                                            np.mean(gdat.listarrytser[strgintp][b][p][y][gdat.listindxtimeregi[b][p][y][k], 0]))
        else:
            rflxtren.append(gdat.listobjtspln[b][p][y][k](gdat.listarrytser[strgintp][b][p][y][gdat.listindxtimeregi[b][p][y][k], 0]))
    gdat.listarrytser[strgtren][b][p][y][:, 1] = np.concatenate(rflxtren)

    numbsplnregi = len(gdat.rflxbdtrregi)
    gdat.indxsplnregi[b][p][y] = np.arange(numbsplnregi)


def plot_tserwrap(gdat, strgarry, b, p, y=None, boolcolr=True):
    
    boolchun = y is not None
    
    if not boolchun and gdat.numbchun[b][p] == 1:
        return
        
    # determine name of the file
    ## string indicating the prior on the transit ephemerides
    strgprioplan = ''
    if strgarry != 'raww' and gdat.typeprioplan is not None:
        strgprioplan = '_%s' % gdat.typeprioplan
    
    strgcolr = ''
    if boolcolr:
        strgcolr = '_colr'
    strgchun = ''
    if boolchun:
        strgchun = '_' + gdat.liststrgchun[b][p][y]
    path = gdat.pathimagtarg + '%s_%s%s%s_%s%s_%s%s.%s' % \
                    (gdat.liststrgtser[b], gdat.strgcnfg, strgarry, strgcolr, gdat.liststrginst[b][p], strgchun, gdat.strgtarg, strgprioplan, gdat.typefileplot)
    
    if os.path.exists(path):
        print('Plot exists at %s. Skipping the plotting...' % path)
    else:
        if boolchun:
            arrytser = gdat.listarrytser[strgarry][b][p][y]
            
            if len(gdat.listarrytser[strgarry][b][p][y]) == 0:
                raise Exception('')
            if strgarry == 'flar':
                arrytsertren = gdat.listarrytser['bdtrnocl'][b][p][y]
        else:
            arrytser = gdat.arrytser[strgarry][b][p]
            
            if strgarry == 'flar':
                arrytsertren = gdat.arrytser['bdtrnocl'][b][p]
        
        figr, axis = plt.subplots(figsize=gdat.figrsizeydobskin)
        axis.plot(arrytser[:, 0] - gdat.timeoffs, arrytser[:, 1], color='grey', marker='.', ls='', ms=1, rasterized=True)
        if boolcolr:
            # color and name transits
            ylim = axis.get_ylim()
            listtimetext = []
            for j in gdat.indxcomp:
                if boolchun:
                    indxtime = gdat.listindxtimetranchun[j][b][p][y] 
                else:
                    if y > 0:
                        continue
                    indxtime = gdat.listindxtimetran[j][b][p][0]
                
                colr = gdat.listcolrcomp[j]
                # plot data
                axis.plot(arrytser[indxtime, 0] - gdat.timeoffs, arrytser[indxtime, 1], color=colr, marker='.', ls='', ms=1, rasterized=True)
                # draw planet names
                for n in np.linspace(-gdat.numbcyclcolrplot, gdat.numbcyclcolrplot, 2 * gdat.numbcyclcolrplot + 1):
                    time = gdat.epocprio[j] + n * gdat.periprio[j] - gdat.timeoffs
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
            if strgarry == 'flar':
                ydat = axis.get_ylim()[1]
                for kk in range(len(gdat.listindxtimeflar[p][y])):
                    ms = 0.5 * gdat.listmdetflar[p][y][kk]
                    axis.plot(arrytser[gdat.listindxtimeflar[p][y][kk], 0] - gdat.timeoffs, ydat, marker='v', color='b', ms=ms, rasterized=True)
                axis.plot(arrytsertren[:, 0] - gdat.timeoffs, arrytsertren[:, 1], color='g', marker='.', ls='', ms=1, rasterized=True)
                axis.fill_between(arrytsertren[:, 0] - gdat.timeoffs, arrytsertren[:, 1] - gdat.thrsrflxflar[p][y] + 1., \
                                                                      arrytsertren[:, 1] + gdat.thrsrflxflar[p][y] - 1., \
                                                                      color='c', alpha=0.2, rasterized=True)
                
            if strgarry == 'flarprep':
                axis.axhline(gdat.thrsrflxflar[p][y], ls='--', alpha=0.5, color='r')
            
        axis.set_xlabel('Time [BJD - %d]' % gdat.timeoffs)
        axis.set_ylabel(gdat.listlabltser[b])
        axis.set_title(gdat.labltarg)
        plt.subplots_adjust(bottom=0.2)
        
        if gdat.typeverb > 0:
            print('Writing to %s...' % path)
        plt.savefig(path, dpi=200)
        plt.close()


def plot_tser(gdat, b, p, y, strgarry):
    
    # plot each chunck
    plot_tserwrap(gdat, strgarry, b, p, y, boolcolr=False)
    
    # plot all chunks together if there is more than one chunk
    if y == 0 and gdat.numbchun[b][p] > 1:
        plot_tserwrap(gdat, strgarry, b, p, boolcolr=False)
    
    if strgarry == 'bdtr':
        
        if gdat.numbcomp is not None and gdat.numbcomp > 0:
            
            # highlight times in-transit
            ## plot each chunck
            plot_tserwrap(gdat, strgarry, b, p, y, boolcolr=True)
            
            ## plot all chunks together if there is more than one chunk
            if y == 0 and gdat.numbchun[b][p] > 1:
                plot_tserwrap(gdat, strgarry, b, p, boolcolr=True)

            if b == 0:
                path = gdat.pathimagtarg + 'rflx%s_intr%s_%s_%s_%s.%s' % \
                                                (strgarry, gdat.strgcnfg, gdat.liststrginst[b][p], gdat.strgtarg, gdat.typeprioplan, gdat.typefileplot)
                if os.path.exists(path):
                    print('Plot exists at %s. Skipping the plotting...' % path)
                else:
                    # plot only the in-transit data
                    figr, axis = plt.subplots(gdat.numbcomp, 1, figsize=gdat.figrsizeydobskin, sharex=True)
                    if gdat.numbcomp == 1:
                        axis = [axis]
                    for jj, j in enumerate(gdat.indxcomp):
                        axis[jj].plot(gdat.arrytser[strgarry][b][p][gdat.listindxtimetran[j][b][p][0], 0] - gdat.timeoffs, \
                                                                             gdat.arrytser[strgarry][b][p][gdat.listindxtimetran[j][b][p][0], 1], \
                                                                                               color=gdat.listcolrcomp[j], marker='o', ls='', ms=0.2)
                    
                    axis[-1].set_ylabel('Relative flux')
                    #axis[-1].yaxis.set_label_coords(0, gdat.numbcomp * 0.5)
                    axis[-1].set_xlabel('Time [BJD - %d]' % gdat.timeoffs)
                    
                    #plt.subplots_adjust(bottom=0.2)
                    print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()
        

def plot_tser_bdtr(gdat, b, p, y, strgarry):
        
    if b == 0 and y is not None:
        ## string indicating the prior on the transit ephemerides
        strgprioplan = ''
        if strgarry != 'raww' and gdat.typeprioplan is not None:
            strgprioplan = '_%s' % gdat.typeprioplan
        path = gdat.pathimagtarg + 'rflxbdtr_bdtr%s_%s_%s_%s%s.%s' % (gdat.strgcnfg, gdat.liststrginst[b][p], \
                                            gdat.liststrgchun[b][p][y], gdat.strgtarg, strgprioplan, gdat.typefileplot)
        if os.path.exists(path):
            print('Plot exists at %s. Skipping the plotting...' % path)
        else:
            # plot baseline-detrending
            figr, axis = plt.subplots(2, 1, figsize=gdat.figrsizeydob)
            for i in gdat.indxsplnregi[b][p][y]:
                ## non-baseline-detrended light curve
                indxtimetemp = gdat.listindxtimeregi[b][p][y][i][gdat.indxtimeregioutt[b][p][y][i]]
                axis[0].plot(gdat.listarrytser['mask'][b][p][y][indxtimetemp, 0] - gdat.timeoffs, \
                                                 gdat.listarrytser['mask'][b][p][y][indxtimetemp, 1], rasterized=True, alpha=gdat.alphraww, \
                                                                                                marker='o', ls='', ms=1, color='grey')
                ## spline
                if gdat.listobjtspln[b][p][y] is not None and gdat.listobjtspln[b][p][y][i] is not None:
                    timesplnregifine = np.linspace(gdat.listarrytser[strgarry][b][p][y][gdat.listindxtimeregi[b][p][y][i], 0][0], \
                                                         gdat.listarrytser[strgarry][b][p][y][gdat.listindxtimeregi[b][p][y][i], 0][-1], 1000)
                    axis[0].plot(timesplnregifine - gdat.timeoffs, gdat.listobjtspln[b][p][y][i](timesplnregifine), 'b-', lw=3, rasterized=True)
                ## baseline-detrended light curve
                indxtimetemp = gdat.listindxtimeregi[b][p][y][i]
                axis[1].plot(gdat.listarrytser[strgarry][b][p][y][indxtimetemp, 0] - gdat.timeoffs, \
                                                                   gdat.listarrytser[strgarry][b][p][y][indxtimetemp, 1], rasterized=True, alpha=gdat.alphraww, \
                                                                                                  marker='o', ms=1, ls='', color='grey')
            for a in range(2):
                axis[a].set_ylabel('Relative flux')
            axis[0].set_xticklabels([])
            axis[1].set_xlabel('Time [BJD - %d]' % gdat.timeoffs)
            plt.subplots_adjust(hspace=0.)
            print('Writing to %s...' % path)
            plt.savefig(path, dpi=200)
            plt.close()
                            

def init( \
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

         # string to pull the priors from the NASA Exoplanet Archive
         strgexar=None, \

         # a string for the label of the target
         labltarg=None, \
         
         # a string for the folder name and file name extensions
         strgtarg=None, \
         
         # string indicating the cluster of targets
         strgclus=None, \
        
         # a string distinguishing the run
         strgcnfg=None, \

         # mode of operation
         ## Boolean flag to use and plot time-series data
         booldatatser=True, \
         
         # Boolean flag to only consider SPOC data on disk
         boolspoconly=False, \

         # output
         ## plotting
         ## Boolean flag to plot the properties of exoplanets
         boolplotprop=False, \
         ## Boolean flag to plot the time-series
         boolplottser=True, \
         ## Boolean flag to plot the prior
         boolplotprio=True, \
         
         ## Boolean flag to enforce offline operation
         boolforcoffl=False, \

         # the path of the folder in which the target folder will be placed
         pathbasetarg=None, \
         # the path of the target folder
         pathtarg=None, \
         # the path of the target data folder
         pathdatatarg=None, \
         # the path of the target image folder
         pathimagtarg=None, \
        
         listdatatype=None, \

         # input data
         ## path of the CSV file containing the input data
         listpathdatainpt=None, \
         ## input data
         listarrytser=None, \
         ## list of TESS sectors for the input data
         listtsec=None, \

         # plotting
         timeoffs=2457000., \

         # list of labels indicating instruments
         listlablinst=[['TESS'], []], \
         # list of strings indicating instruments
         liststrginst=None, \
         # list of strings indicating chunks
         liststrgchun=None, \
         # list of chunk indices for each instrument
         listindxchuninst=None, \
         
         # planet names
         ## list of letters to be assigned to planets
         liststrgcomp=None, \
         ## list of colors to be assigned to planets
         listcolrcomp=None, \
         ## Boolean flag to assign them letters *after* ordering them in orbital period, unless liststrgcomp is specified by the user
         boolordrplanname=True, \
        
         # plot orbit
         boolanimorbt=False, \
        
         # Boolean flag to analyze planet features
         boolfeatplan=False, \

         # input dictionary for lygos                                
         dictlygoinpt=dict(), \
         
         # preprocessing
         boolbdtr=True, \
         
         # Boolean flag to apply sigma-clipping
         boolclip=True, \
         ## dilution: None (no correction), 'lygos' for estimation via lygos, or float value
         dilu=None, \
         ## baseline detrending
         timebrek=0.1, \
         ordrspln=3, \
         typebdtr='spln', \
         ## time scale for median-filtering detrending
         timescalbdtrmedi=1., \
         ## time scale for spline baseline detrending
         listtimescalbdtrspln=[1.], \

         ## Boolean flag to mask bad data
         dictlcurtessinpt=dict(), \
         
         ## time limits to mask
         listlimttimemask=None, \
        
         ## Boolean flag to calculate the power spectral density
         boolcalcpden=False, \

         # signal search
         ### Boolean flag to estimate the LS periodogram
         boollspe=False, \
         ### maximum frequency (per day) for LS periodogram
         maxmfreqlspe=None, \
         
         ### input dictionary to the search pipeline for single transits
         dictsrchtransinginpt=None, \
         ### input dictionary to the search pipeline for flares
         dictsrchflarinpt=dict(), \
         
         # type of model for finding flares
         typemodlflar='outl', \

         ## transit search
         ### Boolean flag to search for periodic boxes
         boolsrchpbox=False, \
         ### input dictionary to the search pipeline for periodic boxes
         dictpboxinpt=dict(), \
        
         # include the ExoFOP catalog in the comparisons to exoplanet population
         boolexofpopl=True, \

         # model
         ## list of analysis types
         ### 'pbox': search for periodic boxes
         ### 'lspe': search for sinusoid variability
         ### 'mfil': matched filter
         listtypeanls=None, \
         
         # list of types of models for the stellar light curve
         ## 'plantran': transiting planet
         ## 'star': stellar companion
         ## 'plan': planetary companion
         ## 'planphas': planetary companion with phase curve
         ## 'coob': compact object companion
         ## 'flar': stellar flare
         listtypeobjt=None, \

         ## priors
         ### type of priors for stars: 'tici', 'exar', 'inpt'
         typepriostar=None, \

         # type of priors for planets
         typeprioplan=None, \
         
         # threshold percentile for detecting stellar flares
         thrssigmflar=7., \

         ### 
         # Boolean flag to turn on transit for each planet
         booltrancomp=None, \

         ### photometric and RV model
         #### means
         rratprio=None, \
         rsmaprio=None, \
         epocprio=None, \
         periprio=None, \
         cosiprio=None, \
         ecosprio=None, \
         esinprio=None, \
         rvsaprio=None, \
         #### uncertainties
         stdvrratprio=None, \
         stdvrsmaprio=None, \
         stdvepocprio=None, \
         stdvperiprio=None, \
         stdvcosiprio=None, \
         stdvecosprio=None, \
         stdvesinprio=None, \
         stdvrvsaprio=None, \
        
         ### others 
         #### mean
         projoblqprio=None, \
         #### uncertainties
         stdvprojoblqprio=None, \

         radistar=None, \
         massstar=None, \
         tmptstar=None, \
         rascstar=None, \
         declstar=None, \
         vsiistarprio=None, \
         
         vmagsyst=None, \
         jmagsyst=None, \
         hmagsyst=None, \
         kmagsyst=None, \

         stdvradistar=None, \
         stdvmassstar=None, \
         stdvtmptstar=None, \
         stdvrascstar=None, \
         stdvdeclstar=None, \
         stdvvsiistarprio=None, \

         # type of inference, mile for miletos, alle for allesfitter
         typeinfe='mile', \

         ## type of exoplanet model
         listtypemodl=None, \
         ## Boolean flag to perform inference on the phase-folded (onto the period of the first planet) and binned data
         boolinfefoldbind=False, \
         ## Boolean flag to model the out-of-transit data to learn a background model
         boolallebkgdgaus=False, \
         # allesfitter settings
         dictdictallesett=None, \
         dictdictallepara=None, \
         # output

         ## Boolean flag to write planet name on plots
         boolwritplan=True, \
         ## plotting
         makeprioplot=True, \
         ## Boolean flag to rasterize the raw time-series on plots
         boolrastraww=True, \
         ## list of transit depths to indicate on the light curve and phase curve plots
         listdeptdraw=None, \
         ### list of offsets for the planet annotations in the TSM/ESM plot
         offstextatmoraditmpt=None, \
         offstextatmoradimetr=None, \

         booldiagmode=True, \
         
         typeverb=1, \

         **args \
        ):
    
    # construct global object
    gdat = tdpy.gdatstrt()
    
    # measure initial time
    gdat.timeinit = modutime.time()

    # copy unnamed inputs to the global object
    for attr, valu in locals().items():
        if '__' not in attr and attr != 'gdat':
            setattr(gdat, attr, valu)

    # copy named arguments to the global object
    for strg, valu in args.items():
        setattr(gdat, strg, valu)

    # string for date and time
    gdat.strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if gdat.typeverb > 0:
        print('miletos initialized at %s...' % gdat.strgtimestmp)
    
    # paths
    gdat.pathbaselygo = os.environ['LYGOS_DATA_PATH'] + '/'
    gdat.pathbasemile = os.environ['MILETOS_DATA_PATH'] + '/'
    
    if gdat.listtypeobjt is None:
        gdat.listtypeobjt = ['exop']
    
    if gdat.listtypeanls is None:
        gdat.listtypeanls = []
        if 'exop' in gdat.listtypeobjt or 'bhol' in gdat.listtypeobjt:
            gdat.listtypeanls += ['pbox']

    if gdat.listtypemodl is None:
        gdat.listtypemodl = []
        if 'exop' in gdat.listtypeobjt or 'bhol' in gdat.listtypeobjt:
            gdat.listtypemodl += ['orbt']
        if 'supn' in gdat.listtypeobjt:
            gdat.listtypemodl += ['rise']

    if gdat.typeverb > 0:
        print('List of object types: %s' % gdat.listtypeobjt)
        print('List of analysis types: %s' % gdat.listtypeanls)
        print('List of model types: %s' % gdat.listtypemodl)

    # Boolean flag to perform inference
    gdat.boolinfe = len(gdat.listtypemodl) > 0 and gdat.booldatatser
    
    ## ensure that target and star coordinates are not provided separately
    if gdat.rasctarg is not None and gdat.rascstar is not None:
        raise Exception('')
    if gdat.decltarg is not None and gdat.declstar is not None:
        raise Exception('')

    ## ensure target identifiers are not conflicting
    if gdat.listarrytser is None:
        if gdat.ticitarg is None and gdat.strgmast is None and gdat.toiitarg is None and (gdat.rasctarg is None or gdat.decltarg is None):
            raise Exception('Either a TIC ID (ticitarg), RA&DEC (rasctarg and decltarg), MAST key (strgmast) or a TOI number (toiitarg) should be provided.')
        if gdat.ticitarg is not None and (gdat.strgmast is not None or gdat.toiitarg is not None or gdat.rasctarg is not None or gdat.decltarg is not None):
            raise Exception('Either a TIC ID (ticitarg), RA&DEC (rasctarg and decltarg), MAST key (strgmast) or a TOI number (toiitarg) should be provided.')
        if gdat.strgmast is not None and (gdat.ticitarg is not None or gdat.toiitarg is not None or gdat.rasctarg is not None or gdat.decltarg is not None):
            raise Exception('Either a TIC ID (ticitarg), RA&DEC (rasctarg and decltarg), MAST key (strgmast) or a TOI number (toiitarg) should be provided.')
        if gdat.toiitarg is not None and (gdat.strgmast is not None or gdat.ticitarg is not None or gdat.rasctarg is not None or gdat.decltarg is not None):
            raise Exception('Either a TIC ID (ticitarg), RA&DEC (rasctarg and decltarg), MAST key (strgmast) or a TOI number (toiitarg) should be provided.')
        if gdat.strgmast is not None and (gdat.ticitarg is not None or gdat.toiitarg is not None or gdat.rasctarg is not None or gdat.decltarg is not None):
            raise Exception('Either a TIC ID (ticitarg), RA&DEC (rasctarg and decltarg), MAST key (strgmast) or a TOI number (toiitarg) should be provided.')
    else:
        if gdat.ticitarg is not None or gdat.strgmast is not None or gdat.toiitarg is not None or gdat.rasctarg is not None or gdat.decltarg is not None:
            raise Exception('No TIC ID (ticitarg), RA&DEC (rasctarg and decltarg), MAST key (strgmast) or a TOI number (toiitarg) \
                                                                                        can be provided when data (listarrytser) is provided.')
    
    # dictionary to be returned
    dictmileoutp = dict()
    
    if 'exop' in gdat.listtypeobjt:
        gdat.pathtoii = gdat.pathbaselygo + 'data/exofop_tess_tois.csv'
    if 'exop' in gdat.listtypeobjt and gdat.typeprioplan != 'exar':
        if gdat.typeverb > 0:
            print('Reading from %s...' % gdat.pathtoii)
        objtexof = pd.read_csv(gdat.pathtoii, skiprows=0)
    
    # conversion factors
    gdat.dictfact = ephesus.retr_factconv()

    # settings
    ## plotting
    gdat.numbcyclcolrplot = 300
    gdat.alphraww = 0.2
    ### percentile for zoom plots of relative flux
    gdat.pctlrflx = 95.
    gdat.typefileplot = 'pdf'
    gdat.figrsize = [6, 4]
    gdat.figrsizeydob = [8., 4.]
    gdat.figrsizeydobskin = [8., 2.5]
    boolpost = False
    if boolpost:
        gdat.figrsize /= 1.5
        
    gdat.listfeatstar = ['radistar', 'massstar', 'tmptstar', 'rascstar', 'declstar', 'vsiistar', 'jmagsyst']
    gdat.listfeatstarpopl = ['radiplan', 'massplan', 'tmptplan', 'radistar', 'jmagsyst', 'kmagsyst', 'tmptstar']
    
    gdat.liststrgpopl = ['exar']
    if gdat.boolexofpopl:
        gdat.liststrgpopl += ['exof']
    gdat.numbpopl = len(gdat.liststrgpopl)
    
    # determine target identifiers
    if gdat.ticitarg is not None:
        gdat.typetarg = 'tici'
        if gdat.typeverb > 0:
            print('A TIC ID was provided as target identifier.')
        
        if 'exop' in gdat.listtypeobjt and gdat.typeprioplan != 'exar':
            indx = np.where(objtexof['TIC ID'].values == gdat.ticitarg)[0]
            if indx.size > 0:
                gdat.toiitarg = int(str(objtexof['TOI'][indx[0]]).split('.')[0])
                if gdat.typeverb > 0:
                    print('Matched the input TIC ID with TOI %d.' % gdat.toiitarg)
        
        gdat.strgmast = 'TIC %d' % gdat.ticitarg

    elif gdat.toiitarg is not None:
        gdat.typetarg = 'toii'
        if gdat.typeverb > 0:
            print('A TOI number (%d) was provided as target identifier.' % gdat.toiitarg)
        # determine TIC ID
        gdat.strgtoiibase = str(gdat.toiitarg)
        indx = []
        for k, strg in enumerate(objtexof['TOI']):
            if str(strg).split('.')[0] == gdat.strgtoiibase:
                indx.append(k)
        indx = np.array(indx)
        if indx.size == 0:
            print('Did not find the TOI in the ExoFOP-TESS TOI list.')
            print('objtexof[TOI]')
            summgene(objtexof['TOI'])
            raise Exception('')
        gdat.ticitarg = objtexof['TIC ID'].values[indx[0]]

        gdat.strgmast = 'TIC %d' % gdat.ticitarg

    elif gdat.strgmast is not None:
        gdat.typetarg = 'mast'
        if gdat.typeverb > 0:
            print('A MAST key (%s) was provided as target identifier.' % gdat.strgmast)

    elif gdat.rasctarg is not None and gdat.decltarg is not None:
        gdat.typetarg = 'posi'
        if gdat.typeverb > 0:
            print('RA and DEC (%g %g) are provided as target identifier.' % (gdat.rasctarg, gdat.decltarg))
        gdat.strgmast = '%g %g' % (gdat.rasctarg, gdat.decltarg)
    elif gdat.listarrytser is not None:
        gdat.typetarg = 'inpt'

        if gdat.labltarg is None:
            raise Exception('')
    
    # check input arguments
    if gdat.listtsec is not None and gdat.typetarg != 'inpt':
        raise Exception('List of TESS sectors can only be input when typetarg is "inpt".')
        
    gdat.maxmnumbiterbdtr = 5
    
    if gdat.strgcnfg is None:
        gdat.strgcnfg = ''
    else:
        gdat.strgcnfg = '_' + gdat.strgcnfg
    
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
    
    print('boolplotprop')
    print(boolplotprop)
    
    print('gdat.liststrginst')
    print(gdat.liststrginst)

    # ExoFOP
    if gdat.boolplotprop:
        gdat.dictexof = retr_dictexof()

    ## NASA Exoplanet Archive
    if gdat.boolplotprop:
        gdat.dictexar = ephesus.retr_dictexar()
        numbcompexar = gdat.dictexar['radiplan'].size
        gdat.indxcompexar = np.arange(numbcompexar)
    
    if gdat.strgclus is None:
        gdat.strgclus = ''
    else:
        gdat.strgclus += '/'
    
    if gdat.pathdatatarg is None or gdat.pathimagtarg is None:
        
        if gdat.pathbasetarg is None:
            gdat.pathbasetarg = gdat.pathbasemile
        if gdat.pathtarg is None:
            gdat.pathtarg = gdat.pathbasetarg + '%s%s/' % (gdat.strgclus, gdat.strgtarg)
        
        gdat.pathdatatarg = gdat.pathtarg + 'data/'
        gdat.pathimagtarg = gdat.pathtarg + 'imag/'
    
    
    print('gdat.strgclus')
    print(gdat.strgclus)
    if 'exop' in gdat.listtypeobjt and gdat.boolplotprop:
        gdat.pathimagfeat = gdat.pathimagtarg + 'prop/'
        for strgpdfn in gdat.liststrgpdfn:
            pathimagpdfn = gdat.pathimagfeat + strgpdfn + '/'
            setattr(gdat, 'pathimagfeatplan' + strgpdfn, pathimagpdfn + 'featplan/')
            setattr(gdat, 'pathimagfeatsyst' + strgpdfn, pathimagpdfn + 'featsyst/')
            setattr(gdat, 'pathimagdataplan' + strgpdfn, pathimagpdfn + 'dataplan/')
    
    if gdat.labltarg is None:
        if gdat.typetarg == 'mast':
            gdat.labltarg = gdat.strgmast
        if gdat.typetarg == 'toii':
            gdat.labltarg = 'TOI %d' % gdat.toiitarg
        if gdat.typetarg == 'tici':
            gdat.labltarg = 'TIC %d' % gdat.ticitarg
        if gdat.typetarg == 'posi':
            gdat.labltarg = 'RA=%.4g, DEC=%.4g' % (gdat.rasctarg, gdat.decltarg)
            gdat.strgtarg = 'RA%.4gDEC%.4g' % (gdat.rasctarg, gdat.decltarg)
    
    if gdat.strgtarg is None:
        gdat.strgtarg = ''.join(gdat.labltarg.split(' '))
    
    if gdat.strgtarg == '' or gdat.strgtarg is None:
        raise Exception('')

    if gdat.strgtarg is None or gdat.strgtarg == 'None':
        raise Exception('')
    
    if gdat.typeverb > 0:
        print('gdat.labltarg')
        print(gdat.labltarg)
        print('gdat.strgtarg')
        print(gdat.strgtarg)
    
    dictmileoutp['pathtarg'] = gdat.pathtarg

    #if os.path.exists(gdat.pathtarg):
    #    print('Path for the object exists... Returning.')
    #    return

    # make folders
    for attr, valu in gdat.__dict__.items():
        if attr.startswith('path') and valu is not None and valu.endswith('/'):
            os.system('mkdir -p %s' % valu)
            
    if gdat.booldatatser:
        gdat.liststrgdatatser = ['lcur', 'rvel']
        gdat.numbdatatser = len(gdat.liststrgdatatser)
        gdat.indxdatatser = np.arange(gdat.numbdatatser)

        gdat.numbinst = np.empty(gdat.numbdatatser, dtype=int)
        gdat.indxinst = [[] for b in gdat.indxdatatser]
        for b in gdat.indxdatatser:
            gdat.numbinst[b] = len(gdat.listlablinst[b])
            gdat.indxinst[b] = np.arange(gdat.numbinst[b])
        
        if gdat.typeverb > 0:
            print('gdat.numbinst')
            print(gdat.numbinst)
        
        if gdat.liststrginst is None:
            gdat.liststrginst = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
            for b in gdat.indxdatatser:
                for p in gdat.indxinst[b]:
                    gdat.liststrginst[b][p] = ''.join(gdat.listlablinst[b][p].split(' '))

        if gdat.typetarg != 'inpt' and 'TESS' in gdat.liststrginst[0]:
            if gdat.boolspoconly:
                # get the list of TESS sectors and file paths
                gdat.listtsec, listpath = ephesus.retr_tsectici(gdat.ticitarg)
                # get data
                listarrylcurtess = [[] for k in range(len(listpath))]
                gdat.listtsec = np.empty(len(listpath))
                listtcam = np.empty(len(listpath))
                listtccd = np.empty(len(listpath))
                for k in range(len(listpath)):
                    listarrylcurtess[k], indxtimequalgood, indxtimenanngood, gdat.listtsec[k], listtcam[k], listtccd[k] = ephesus.read_tesskplr_file(listpath[k], typeinst='tess', \
                                                                                                                                                                strgtype='PDCSAP_FLUX')
                # load data
                if len(listarrylcurtess) > 0:
                    arrylcurtess = np.concatenate(listarrylcurtess)
                gdat.listarrytsersapp = None
                gdat.listarrytserpdcc = None

            else:
            
                if gdat.ticitarg is None and gdat.strgmast is None:
                    rasctarg = gdat.rascstar
                    decltarg = gdat.declstar
                else:
                    rasctarg = None
                    decltarg = None
                
                dictlygoinpt['labltarg'] = gdat.labltarg

                arrylcurtess, gdat.arrytsersapp, gdat.arrytserpdcc, listarrylcurtess, gdat.listarrytsersapp, gdat.listarrytserpdcc, \
                                      gdat.listtsec, gdat.listtcam, gdat.listtccd, listpathdownspoclcur = \
                                      ephesus.retr_lcurtess( \
                                                    gdat.pathdatatarg, \
                                                    strgmast=gdat.strgmast, \
                                                    rasctarg=rasctarg, \
                                                    decltarg=decltarg, \
                                                    labltarg=gdat.labltarg, \
                                                    strgtarg=gdat.strgtarg, \
                                                    dictlygoinpt=gdat.dictlygoinpt, \
                                                    **gdat.dictlcurtessinpt, \
                                                    )
            if len(listarrylcurtess) == 0:
                print('No data found. Returning...')
                return dictmileoutp
        
    if gdat.typepriostar is None:
        if gdat.radistar is not None:
            gdat.typepriostar = 'inpt'
        else:
            gdat.typepriostar = 'tici'
    
    if gdat.typeverb > 0:
        print('Stellar parameter prior type: %s' % gdat.typepriostar)
    
    if 'exop' in gdat.listtypeobjt:
        if gdat.strgexar is None:
            gdat.strgexar = gdat.strgmast
    
        if gdat.typeverb > 0:
            print('gdat.strgexar')
            print(gdat.strgexar)

        # grab object properties from NASA Excoplanet Archive
        gdat.dictexartarg = ephesus.retr_dictexar(strgexar=gdat.strgexar)
        
        if gdat.typeverb > 0:
            if gdat.dictexartarg is None:
                print('The target name was **not** found in the NASA Exoplanet Archive planetary systems composite table.')
            else:
                print('The target name was found in the NASA Exoplanet Archive planetary systems composite table.')
        
        # grab object properties from ExoFOP
        if gdat.toiitarg is not None:
            gdat.dictexoftarg = retr_dictexof(toiitarg=gdat.toiitarg)
        else:
            gdat.dictexoftarg = None
        gdat.boolexof = gdat.toiitarg is not None and gdat.dictexoftarg is not None
        gdat.boolexar = gdat.strgexar is not None and gdat.dictexartarg is not None
        
        if gdat.typeprioplan is None:
            if gdat.epocprio is not None:
                gdat.typeprioplan = 'inpt'
            elif gdat.boolexar:
                gdat.typeprioplan = 'exar'
            elif gdat.boolexof:
                gdat.typeprioplan = 'exof'
            else:
                gdat.typeprioplan = 'pbox'

        if gdat.typeverb > 0:
            print('Planetary parameter prior type: %s' % gdat.typeprioplan)
        
        if not gdat.boolexar and gdat.typeprioplan == 'exar':
            raise Exception('')
    
    else:
        gdat.typeprioplan = None
    
    if 'exop' in gdat.listtypeobjt:
        gdat.liststrgpdfn = [gdat.typeprioplan] + [gdat.typeprioplan + typemodl for typemodl in gdat.listtypemodl]

    gdat.liststrgpdfn = ['prio']
    
    if 'flar' in gdat.listtypeobjt:
        gdat.boolsrchflar = True
    else:
        gdat.boolsrchflar = False
    
    gdat.numbcomp = None

    if gdat.typeverb > 0:
        print('gdat.liststrgpdfn')
        print(gdat.liststrgpdfn)

    if gdat.booldatatser:
        # determine number of chunks
        gdat.numbchun = [np.empty(gdat.numbinst[b], dtype=int) for b in gdat.indxdatatser]
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                if gdat.typetarg == 'path':
                    gdat.numbchun[b][p] = len(gdat.listpathdatainpt[b][p])
                elif gdat.typetarg == 'inpt':
                    gdat.numbchun[b][p] = len(gdat.listarrytser['raww'][b][p])
                else:
                    if b == 0 and gdat.liststrginst[b][p] == 'TESS':
                        gdat.numbchun[b][p] = len(listarrylcurtess)
        
        gdat.indxchun = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                gdat.indxchun[b][p] = np.arange(gdat.numbchun[b][p], dtype=int)
        
        if gdat.liststrgchun is None:
            gdat.liststrgchun = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
            for b in gdat.indxdatatser:
                for p in gdat.indxinst[b]:
                    for y in gdat.indxchun[b][p]:
                        if gdat.liststrginst[b][p] == 'TESS' and gdat.listtsec is not None:
                            gdat.liststrgchun[b][p][y] = 'sc%02d' % gdat.listtsec[y]
                        else:
                            gdat.liststrgchun[b][p][y] = 'ch%02d' % y
        
        # check the user-defined gdat.listpathdatainpt
        if gdat.listpathdatainpt is not None:
            for b in gdat.indxdatatser:
                for p in gdat.indxinst[b]:
                    if not isinstance(gdat.listpathdatainpt[b][p], list):
                        raise Exception('')
        
        # list of data types (real or mock) for each instrument for both light curve and RV data
        if gdat.listdatatype is None:
            gdat.listdatatype = [[] for b in gdat.indxdatatser]
            for b in gdat.indxdatatser:
                gdat.listdatatype[b] = ['real' for p in gdat.indxinst[b]]
    
        # Boolean flag to execute a search for periodic boxes
        gdat.boolsrchpbox = 'pbox' in gdat.listtypeanls
        if gdat.typeverb > 0:
            print('gdat.boolsrchpbox') 
            print(gdat.boolsrchpbox)

        if gdat.dictdictallesett is None:
            gdat.dictdictallesett = dict()
            for typemodl in gdat.listtypemodl:
                gdat.dictdictallesett[typemodl] = None

        if gdat.dictdictallepara is None:
            gdat.dictdictallepara = dict()
            for typemodl in gdat.listtypemodl:
                gdat.dictdictallepara[typemodl] = None

        gdat.listlabltser = ['Relative flux', 'Radial Velocity [km/s]']
        gdat.liststrgtser = ['rflx', 'rvel']
        gdat.liststrgtseralle = ['flux', 'rv']
    
        if gdat.typeverb > 0:
            print('Light curve data: %s' % gdat.listlablinst[0])
            print('RV data: %s' % gdat.listlablinst[1])
    
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

    gdat.epocprio = None
    gdat.periprio = None
    gdat.duraprio = None
        
    if 'exop' in gdat.listtypeobjt:
    
        if gdat.typeprioplan == 'exar':
            if gdat.periprio is None:
                gdat.periprio = gdat.dictexartarg['peri']
            gdat.deptprio = gdat.dictexartarg['dept']
            if gdat.cosiprio is None:
                gdat.cosiprio = gdat.dictexartarg['cosi']
            if gdat.epocprio is None:
                gdat.epocprio = gdat.dictexartarg['epoc']

            gdat.duraprio = gdat.dictexartarg['duratran']
            print('gdat.duraprio')
            print(gdat.duraprio)
            indx = np.where(~np.isfinite(gdat.duraprio) & gdat.dictexartarg['booltran'])[0]
            if indx.size > 0:
                dcyc = 0.15
                print('Duration from the Exoplanet Archive Composite PS table is infite for companions. Assuming a duty cycle of %.3g.' % dcyc)
                gdat.duraprio[indx] = gdat.periprio[indx] * dcyc
        if gdat.typeprioplan == 'exof':
            if gdat.typeverb > 0:
                print('A TOI number is provided. Retreiving the TCE attributes from ExoFOP-TESS...')
            
            # find the indices of the target in the TOI catalog
            objtexof = pd.read_csv(gdat.pathtoii, skiprows=0)
            
            if gdat.epocprio is None:
                gdat.epocprio = gdat.dictexoftarg['epoc']
            if gdat.periprio is None:
                gdat.periprio = gdat.dictexoftarg['peri']
            gdat.deptprio = gdat.dictexoftarg['dept']
            gdat.duraprio = gdat.dictexoftarg['duratran']
            if gdat.cosiprio is None:
                gdat.cosiprio = np.zeros_like(gdat.epocprio)

        if gdat.typeprioplan == 'inpt':
            if gdat.rratprio is None:
                gdat.rratprio = 0.1 + np.zeros_like(gdat.epocprio)
            if gdat.rsmaprio is None:
                gdat.rsmaprio = 0.2 * gdat.periprio**(-2. / 3.)
            
            if gdat.typeverb > 0:
                print('gdat.cosiprio')
                print(gdat.cosiprio)

            if gdat.cosiprio is None:
                gdat.cosiprio = np.zeros_like(gdat.epocprio)
            gdat.duraprio = ephesus.retr_dura(gdat.periprio, gdat.rsmaprio, gdat.cosiprio)
            gdat.deptprio = 1e3 * gdat.rratprio**2
        
        # check MAST
        if gdat.typetarg != 'inpt' and gdat.strgmast is None:
            gdat.strgmast = gdat.labltarg

        if gdat.typeverb > 0:
            print('gdat.strgmast')
            print(gdat.strgmast)
        
        if not gdat.boolforcoffl and gdat.strgmast is not None:
            listdictcatl = astroquery.mast.Catalogs.query_object(gdat.strgmast, catalog='TIC', radius='40s')
            if listdictcatl[0]['dstArcSec'] > 0.1:
                if gdat.typeverb > 0:
                    print('The nearest source is more than 0.1 arcsec away from the target!')
            
            if gdat.typeverb > 0:
                print('Found the target on MAST!')
            
            gdat.rascstar = listdictcatl[0]['ra']
            gdat.declstar = listdictcatl[0]['dec']
            gdat.stdvrascstar = 0.
            gdat.stdvdeclstar = 0.
            if gdat.radistar is None:
                
                if gdat.typeverb > 0:
                    print('Setting the stellar radius from the TIC.')
                
                gdat.radistar = listdictcatl[0]['rad']
                gdat.stdvradistar = listdictcatl[0]['e_rad']
                
                if gdat.typeverb > 0:
                    if not np.isfinite(gdat.radistar):
                        print('Warning! TIC stellar radius is not finite.')
                    if not np.isfinite(gdat.radistar):
                        print('Warning! TIC stellar radius uncertainty is not finite.')
            if gdat.massstar is None:
                
                if gdat.typeverb > 0:
                    print('Setting the stellar mass from the TIC.')
                
                gdat.massstar = listdictcatl[0]['mass']
                gdat.stdvmassstar = listdictcatl[0]['e_mass']
                
                if gdat.typeverb > 0:
                    if not np.isfinite(gdat.massstar):
                        print('Warning! TIC stellar mass is not finite.')
                    if not np.isfinite(gdat.stdvmassstar):
                        print('Warning! TIC stellar mass uncertainty is not finite.')
            if gdat.tmptstar is None:
                
                if gdat.typeverb > 0:
                    print('Setting the stellar temperature from the TIC.')
                
                gdat.tmptstar = listdictcatl[0]['Teff']
                gdat.stdvtmptstar = listdictcatl[0]['e_Teff']
                
                if gdat.typeverb > 0:
                    if not np.isfinite(gdat.tmptstar):
                        print('Warning! TIC stellar temperature is not finite.')
                    if not np.isfinite(gdat.tmptstar):
                        print('Warning! TIC stellar temperature uncertainty is not finite.')
            gdat.jmagsyst = listdictcatl[0]['Jmag']
            gdat.hmagsyst = listdictcatl[0]['Hmag']
            gdat.kmagsyst = listdictcatl[0]['Kmag']
            gdat.vmagsyst = listdictcatl[0]['Vmag']
    
    
    if gdat.booldatatser and gdat.boolinfe and gdat.typeinfe == 'alle':
        gdat.pathallebase = gdat.pathdatatarg + 'allesfits/'
        
    if gdat.typeverb > 0:
        print('gdat.typeprioplan')
        print(gdat.typeprioplan)
        print('gdat.booldatatser')
        print(gdat.booldatatser)

    if gdat.booldatatser:
        gdat.arrytser = dict()
        if not gdat.typetarg == 'inpt':
            gdat.listarrytser = dict()
    
        gdat.pathalle = dict()
        gdat.objtalle = dict()
        
        gdat.arrytser['raww'] = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.arrytser['mask'] = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.arrytser['temp'] = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        for r in range(gdat.maxmnumbiterbdtr):
            gdat.arrytser['clipoutp%04d' % r] = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
            gdat.arrytser['bdtroutp%04d' % r] = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.arrytser['bdtr'] = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.arrytser['bdtrbind'] = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.arrytser['bdtrnocl'] = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        
        if not gdat.typetarg == 'inpt':
            gdat.listarrytser['raww'] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        else:
            for b in gdat.indxdatatser:
                for p in gdat.indxinst[b]:
                    gdat.arrytser['raww'][b][p] = np.concatenate(gdat.listarrytser['raww'][b][p])
        gdat.listarrytser['mask'] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.listarrytser['temp'] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        for r in range(gdat.maxmnumbiterbdtr):
            gdat.listarrytser['clipoutp%04d' % r] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
            gdat.listarrytser['bdtroutp%04d' % r] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.listarrytser['bdtr'] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.listarrytser['bdtrbind'] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.listarrytser['bdtrnocl'] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        
        # load TESS data
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                if gdat.typetarg != 'inpt' and b == 0 and gdat.liststrginst[b][p] == 'TESS':
                    gdat.arrytser['raww'][b][p] = arrylcurtess
                    for y in gdat.indxchun[b][p]:
                        gdat.listarrytser['raww'][b][p][y] = listarrylcurtess[y]
        
        # load input data
        if gdat.listpathdatainpt is not None:
            for b in gdat.indxdatatser:
                for p in gdat.indxinst[b]:
                    for y in gdat.indxchun[b][p]:
                        arry = np.loadtxt(gdat.listpathdatainpt[b][p][y], delimiter=',', skiprows=1)
                        gdat.listarrytser['raww'][b][p][y] = np.empty((arry.shape[0], 3))
                        gdat.listarrytser['raww'][b][p][y][:, 0:2] = arry[:, 0:2]
                        gdat.listarrytser['raww'][b][p][y][:, 2] = 1e-4 * arry[:, 1]
                        indx = np.argsort(gdat.listarrytser['raww'][b][p][y][:, 0])
                        gdat.listarrytser['raww'][b][p][y] = gdat.listarrytser['raww'][b][p][y][indx, :]
                        indx = np.where(gdat.listarrytser['raww'][b][p][y][:, 1] < 1e6)[0]
                        gdat.listarrytser['raww'][b][p][y] = gdat.listarrytser['raww'][b][p][y][indx, :]
                        gdat.listisec = None

        # check availability of data 
        booldataaval = False
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                for y in gdat.indxchun[b][p]:
                    if len(gdat.listarrytser['raww'][b][p]) > 0:
                        booldataaval = True
        if not booldataaval:
            print('No data found. Returning...')
            return dictmileoutp
        
        # plot raw data
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                for y in gdat.indxchun[b][p]:
                    if gdat.boolplottser:
                        plot_tser(gdat, b, p, y, 'raww')
                if gdat.boolplottser:
                    gdat.arrytser['raww'][b][p] = np.concatenate(gdat.listarrytser['raww'][b][p])

        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                for y in gdat.indxchun[b][p]:
                    if len(gdat.listarrytser['raww'][b][p][y]) == 0:
                        print('bpy')
                        print(b, p, y)
                        raise Exception('')
        
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                if not np.isfinite(gdat.arrytser['raww'][b][p]).all():
                    print('b, p')
                    print(b, p)
                    indxbadd = np.where(~np.isfinite(gdat.arrytser['raww'][b][p]))[0]
                    print('gdat.arrytser[raww][b][p]')
                    summgene(gdat.arrytser['raww'][b][p])
                    print('indxbadd')
                    summgene(indxbadd)
                    raise Exception('')
        
        # mask out data
        if gdat.listlimttimemask is not None:
            
            if gdat.typeverb > 0:
                print('Masking the data...')
            for b in gdat.indxdatatser:
                for p in gdat.indxinst[b]:
                    numbmask = len(gdat.listlimttimemask[b][p])
                    for y in gdat.indxchun[b][p]:
                        listindxtimemask = []
                        for k in range(numbmask):
                            indxtimemask = np.where((gdat.listarrytser['raww'][b][p][y][:, 0] < gdat.listlimttimemask[b][p][k][1]) & \
                                                    (gdat.listarrytser['raww'][b][p][y][:, 0] > gdat.listlimttimemask[b][p][k][0]))[0]
                            listindxtimemask.append(indxtimemask)
                        listindxtimemask = np.concatenate(listindxtimemask)
                        listindxtimegood = np.setdiff1d(np.arange(gdat.listarrytser['raww'][b][p][y].shape[0]), listindxtimemask)
                        gdat.listarrytser['mask'][b][p][y] = gdat.listarrytser['raww'][b][p][y][listindxtimegood, :]
                        if gdat.boolplottser:
                            plot_tser(gdat, b, p, y, 'mask')
                    gdat.arrytser['mask'][b][p] = np.concatenate(gdat.listarrytser['mask'][b][p], 0)
                    if gdat.boolplottser:
                        plot_tser(gdat, b, p, y, 'mask')
        else:
            gdat.arrytser['mask'] = gdat.arrytser['raww']
            gdat.listarrytser['mask'] = gdat.listarrytser['raww']
        
        # detrending
        ## determine whether to use any mask for detrending
        if ('exop' in gdat.listtypeobjt or 'bhol' in gdat.listtypeobjt) and gdat.duraprio is not None and len(gdat.duraprio) > 0:
            # assign the prior orbital solution to the baseline-detrend mask
            gdat.epocmask = gdat.epocprio
            gdat.perimask = gdat.periprio
            gdat.duramask = 2. * gdat.duraprio
        else:
            gdat.epocmask = None
            gdat.perimask = None
            gdat.duramask = None
        
        if gdat.numbinst[0] > 0 and gdat.boolbdtr:
            gdat.listobjtspln = [[[[] for y in gdat.indxchun[0][p]] for p in gdat.indxinst[0]] for b in gdat.indxdatatser]
            gdat.indxsplnregi = [[[[] for y in gdat.indxchun[0][p]] for p in gdat.indxinst[0]] for b in gdat.indxdatatser]
            gdat.listindxtimeregi = [[[[] for y in gdat.indxchun[0][p]] for p in gdat.indxinst[0]] for b in gdat.indxdatatser]
            gdat.indxtimeregioutt = [[[[] for y in gdat.indxchun[0][p]] for p in gdat.indxinst[0]] for b in gdat.indxdatatser]
            
            gdat.numbiterbdtr = [[0 for y in gdat.indxchun[0][p]] for p in gdat.indxinst[0]]
            numbtimecutt = [[1 for y in gdat.indxchun[0][p]] for p in gdat.indxinst[0]]
            
            for e, timescalbdtrspln in enumerate(gdat.listtimescalbdtrspln):
                
                if timescalbdtrspln == 0:
                    continue

                strgarrybdtr = 'bdtrts%02d' % e
                gdat.listarrytser[strgarrybdtr] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
                
                # baseline-detrending
                for p in gdat.indxinst[0]:
                    for y in gdat.indxchun[0][p]:
                        
                        print('Detrending data from chunck %s...' % gdat.liststrgchun[0][p][y])
                        
                        strgarryclipoutp = 'mask'
                        r = 0
                        while numbtimecutt[p][y] > 0:
                            
                            if r == gdat.maxmnumbiterbdtr:
                                break

                            print('Iteration %d' % r)
            
                            if r == 0:
                                strgarrybdtrinpt = 'mask'
                            else:
                                strgarrybdtrinpt = 'clipoutpit%02dts%02d' % (r - 1, e)
                            strgarryclipoutp = 'clipoutpit%02dts%02d' % (r, e)
                            strgarrybdtroutp = 'bdtroutpit%02dts%02d' % (r, e)
                            if p == 0 and y == 0:
                                gdat.listarrytser[strgarryclipoutp] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
                                gdat.listarrytser[strgarrybdtroutp] = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
                            
                            # trial detrending
                            bdtr_wrap(gdat, 0, p, y, gdat.epocmask, gdat.perimask, gdat.duramask, strgarrybdtrinpt, strgarrybdtroutp, 'temp', timescalbdtrspln=timescalbdtrspln)
                            
                            # sigma-clipping
                            lcurclip, lcurcliplowr, lcurclipuppr = scipy.stats.sigmaclip(gdat.listarrytser['temp'][0][p][y][:, 1], low=7., high=7.)
                            indxtimekeep = np.where((gdat.listarrytser['temp'][0][p][y][:, 1] < lcurclipuppr) & \
                                                    (gdat.listarrytser['temp'][0][p][y][:, 1] > lcurcliplowr))[0]
                            
                            if indxtimekeep.size == 0:
                                raise Exception('')
                            
                            gdat.listarrytser[strgarryclipoutp][0][p][y] = gdat.listarrytser[strgarrybdtrinpt][0][p][y][indxtimekeep, :]
                            numbtimecutt[p][y] = gdat.listarrytser['temp'][0][p][y][:, 1].size - indxtimekeep.size
                            
                            # plot the trial detrended and sigma-clipped time-series data
                            if gdat.boolplottser and numbtimecutt[p][y] > 0:
                                plot_tser(gdat, 0, p, y, strgarrybdtroutp)
                                plot_tser(gdat, 0, p, y, strgarryclipoutp)
                    
                            r += 1
                            
                        gdat.numbiterbdtr[p][y] = r
                        bdtr_wrap(gdat, 0, p, y, gdat.epocmask, gdat.perimask, gdat.duramask, strgarryclipoutp, 'bdtr', 'temp', timescalbdtrspln=timescalbdtrspln)
                        gdat.listarrytser[strgarrybdtr][0][p][y] = gdat.listarrytser[strgarrybdtroutp][0][p][y]
            
                        plot_tser_bdtr(gdat, b, p, y, strgarrybdtr)
            
            if gdat.listtimescalbdtrspln[0] > 0.:
                gdat.listarrytser['bdtr'] = gdat.listarrytser['bdtrts00']
            else:
                gdat.listarrytser['bdtr'] = gdat.listarrytser['mask']
                
            # merge chunks
            for p in gdat.indxinst[0]:
                gdat.arrytser['bdtr'][0][p] = np.concatenate(gdat.listarrytser['bdtr'][0][p], 0)
            
            # perform diagnostic check
            for p in gdat.indxinst[0]:
                if not np.isfinite(gdat.arrytser['bdtr'][0][p]).all():
                    print('p')
                    print(p)
                    indxbadd = np.where(~np.isfinite(gdat.arrytser['bdtr'][0][p]))[0]
                    print('gdat.arrytser[bdtr][0][p]')
                    summgene(gdat.arrytser['bdtr'][0][p])
                    print('indxbadd')
                    summgene(indxbadd)
                    raise Exception('')
            
            # write baseline-detrended light curve
            for p in gdat.indxinst[0]:
                
                if gdat.numbchun[0][p] > 1:
                    path = gdat.pathdatatarg + 'arrytserbdtr%s.csv' % (gdat.liststrginst[0][p])
                    if gdat.typeverb > 0:
                        print('Writing to %s...' % path)
                    np.savetxt(path, gdat.arrytser['bdtr'][0][p], delimiter=',', \
                                                    header='time,%s,%s_err' % (gdat.liststrgtseralle[0], gdat.liststrgtseralle[0]))
                
                for y in gdat.indxchun[0][p]:
                    path = gdat.pathdatatarg + 'arrytserbdtr%s%s.csv' % (gdat.liststrginst[0][p], gdat.liststrgchun[0][p][y])
                    if gdat.typeverb > 0:
                        print('Writing to %s...' % path)
                    np.savetxt(path, gdat.listarrytser['bdtr'][0][p][y], delimiter=',', \
                                                   header='time,%s,%s_err' % (gdat.liststrgtseralle[0], gdat.liststrgtseralle[0]))
        
            
            if gdat.boolplottser:
                for p in gdat.indxinst[0]:
                    for y in gdat.indxchun[0][p]:
                        plot_tser(gdat, 0, p, y, 'bdtr')
                    plot_tser(gdat, 0, p, None, 'bdtr')
            
        else:
            gdat.arrytser['bdtr'] = gdat.arrytser['mask']
            gdat.listarrytser['bdtr'] = gdat.listarrytser['mask']
            
        # search for periodic boxes
        if gdat.boolsrchpbox:
            
            # temp
            for p in gdat.indxinst[0]:
                
                # check if log file has been created properly before
                strgextn = '%s_%s' % (gdat.liststrginst[0][p], gdat.strgtarg)
                arry = np.copy(gdat.arrytser['bdtr'][0][p])
                if 'bhol' in gdat.listtypeobjt:
                    boolpuls = True
                else:
                    boolpuls = False
                dictpboxoutp = ephesus.srch_pbox(arry, \
                                           boolpuls=boolpuls, \
                                           pathdata=gdat.pathdatatarg, \
                                           pathimag=gdat.pathimagtarg, \
                                           timeoffs=gdat.timeoffs, \
                                           strgextn=strgextn, \
                                           strgplotextn=gdat.typefileplot, \
                                           figrsizeydobskin=gdat.figrsizeydobskin, \
                                           alphraww=gdat.alphraww, \
                                           **gdat.dictpboxinpt, \
                                           )
                
                dictmileoutp['dictpboxoutp'] = dictpboxoutp

                if gdat.epocprio is None:
                    gdat.epocprio = dictpboxoutp['epoc']
                if gdat.periprio is None:
                    gdat.periprio = dictpboxoutp['peri']
                gdat.deptprio = 1. - 1e-3 * dictpboxoutp['dept']
                gdat.duraprio = dictpboxoutp['dura']
                gdat.cosiprio = np.zeros_like(dictpboxoutp['epoc']) 
                gdat.rratprio = np.sqrt(1e-3 * gdat.deptprio)
                gdat.rsmaprio = np.sin(np.pi * gdat.duraprio / gdat.periprio / 24.)
                
                gdat.perimask = gdat.periprio
                gdat.epocmask = gdat.epocprio
                gdat.duramask = 2. * gdat.duraprio
    
        if gdat.typeverb > 0:
            print('gdat.epocmask')
            print(gdat.epocmask)
            print('gdat.perimask')
            print(gdat.perimask)
            print('gdat.duramask')
            print(gdat.duramask)
        
        # search for flares
        if gdat.boolsrchflar:
            dictsrchflarinpt['pathimag'] = gdat.pathimagtarg
            
            gdat.listindxtimeflar = [[[] for y in gdat.indxchun[0][p]] for p in gdat.indxinst[0]]
            gdat.listmdetflar = [[[] for y in gdat.indxchun[0][p]] for p in gdat.indxinst[0]]
            gdat.precphot = [np.empty(gdat.numbchun[0][p]) for p in gdat.indxinst[0]]
            gdat.thrsrflxflar = [np.empty(gdat.numbchun[0][p]) for p in gdat.indxinst[0]]
            for p in gdat.indxinst[0]:
                for y in gdat.indxchun[0][0]:
                    
                    if gdat.typemodlflar == 'outl':
                        listydat = gdat.listarrytser['bdtr'][0][p][y][:, 1]
                        medi = np.median(listydat)
                        indxcent = np.where((listydat > np.percentile(listydat, 1.)) & (listydat < np.percentile(listydat, 99.)))[0]
                        stdv = np.std(listydat[indxcent])
                        gdat.precphot[p][y] = stdv
                        listmdetflar = (listydat - medi) / stdv
                        gdat.thrsrflxflar[p][y] = medi + stdv * gdat.thrssigmflar
                        indxtimeposi = np.where(listmdetflar > gdat.thrssigmflar)[0]
                        
                        for n in range(len(indxtimeposi)):
                            if (n == len(indxtimeposi) - 1) or (n < len(indxtimeposi) - 1) and not ((indxtimeposi[n] + 1) in indxtimeposi):
                                gdat.listindxtimeflar[p][y].append(indxtimeposi[n])
                                mdetflar = listmdetflar[indxtimeposi[n]]
                                gdat.listmdetflar[p][y].append(mdetflar)
                        gdat.listindxtimeflar[p][y] = np.array(gdat.listindxtimeflar[p][y])
                        gdat.listmdetflar[p][y] = np.array(gdat.listmdetflar[p][y])

                    if gdat.typemodlflar == 'tmpl':
                        dictsrchflaroutp = ephesus.srch_flar(gdat.arrytser['bdtr'][0][p][:, 0], gdat.arrytser['bdtr'][0][p][:, 1], **dictsrchflarinpt)
                
            dictmileoutp['listindxtimeflar'] = gdat.listindxtimeflar
            dictmileoutp['listmdetflar'] = gdat.listmdetflar
            dictmileoutp['precphot'] = gdat.precphot
            
            gdat.arrytser['flar'] = [[[] for p in gdat.indxinst[0]] for b in gdat.indxdatatser]
            gdat.listarrytser['flar'] = [[[[] for y in gdat.indxchun[0][p]] for p in gdat.indxinst[0]] for b in gdat.indxdatatser]
            for p in gdat.indxinst[0]:
                for y in gdat.indxchun[0][p]:
                    if gdat.boolplottser:
                        plot_tser(gdat, 0, p, y, 'flar')
                if gdat.boolplottser:
                    plot_tser(gdat, 0, p, None, 'flar')
            
            print('temp: skipping masking out of flaress...')
            # mask out flares
            #numbkern = len(maxmcorr)
            #indxkern = np.arange(numbkern)
            #listindxtimemask = []
            #for k in indxkern:
            #    for indxtime in gdat.listindxtimeposimaxm[k]:
            #        indxtimemask = np.arange(indxtime - 60, indxtime + 60)
            #        listindxtimemask.append(indxtimemask)
            #indxtimemask = np.concatenate(listindxtimemask)
            #indxtimemask = np.unique(indxtimemask)
            #indxtimegood = np.setdiff1d(np.arange(gdat.time.size), indxtimemask)
            #gdat.time = gdat.time[indxtimegood]
            #gdat.lcurdata = gdat.lcurdata[indxtimegood]
            #gdat.lcurdatastdv = gdat.lcurdatastdv[indxtimegood]
            #gdat.numbtime = gdat.time.size

        # calculate PSD
        if gdat.boolcalcpden:
            for b in gdat.indxdatatser:
                for p in gdat.indxinst[b]:
                    for y in gdat.indxchun[b][p]:
                        freq, gdat.psdn = scipy.signal.periodogram(gdat.listarrytser['bdtr'][b][p][y][:, 1], fs=fs)
                        perisamp = 1. / freq
        
    # priors
    if 'exop' in gdat.listtypeobjt or 'bhol' in gdat.listtypeobjt:
        gdat.numbcomp = gdat.epocprio.size
    else:
        gdat.numbcomp = 0
    dictmileoutp['numbcomp'] = gdat.numbcomp
    
    if 'pbox' in gdat.listtypeanls and gdat.booltrancomp is None:
        gdat.booltrancomp = np.zeros(gdat.numbcomp, dtype=bool)
        print('gdat.deptprio')
        print(gdat.deptprio)
        print('gdat.numbcomp')
        print(gdat.numbcomp)
        gdat.booltrancomp[np.where(np.isfinite(gdat.deptprio))] = True

    gdat.indxcomp = np.arange(gdat.numbcomp)
     
    if 'exop' in gdat.listtypeobjt or 'bhol' in gdat.listtypeobjt:
        if gdat.liststrgcomp is None:
            gdat.liststrgcomp = retr_liststrgcomp(gdat.numbcomp)
        if gdat.listcolrcomp is None:
            gdat.listcolrcomp = retr_listcolrcomp(gdat.numbcomp)
        
        if gdat.typeverb > 0:
            print('Planet letters: ')
            print(gdat.liststrgcomp)

        if gdat.duraprio is None:
            gdat.duraprio = ephesus.retr_dura(gdat.periprio, gdat.rsmaprio, gdat.cosiprio)
        
        if gdat.rratprio is None:
            gdat.rratprio = np.sqrt(1e-3 * gdat.deptprio)
        if gdat.rsmaprio is None:
            gdat.rsmaprio = np.sqrt(np.sin(np.pi * gdat.duraprio / gdat.periprio / 24.)**2 + gdat.cosiprio**2)
        if gdat.ecosprio is None:
            gdat.ecosprio = np.zeros(gdat.numbcomp)
        if gdat.esinprio is None:
            gdat.esinprio = np.zeros(gdat.numbcomp)
        if gdat.rvsaprio is None:
            gdat.rvsaprio = np.zeros(gdat.numbcomp)
        
        if gdat.stdvrratprio is None:
            gdat.stdvrratprio = 0.01 + np.zeros(gdat.numbcomp)
        if gdat.stdvrsmaprio is None:
            gdat.stdvrsmaprio = 0.01 + np.zeros(gdat.numbcomp)
        if gdat.stdvepocprio is None:
            gdat.stdvepocprio = 0.1 + np.zeros(gdat.numbcomp)
        if gdat.stdvperiprio is None:
            gdat.stdvperiprio = 0.01 + np.zeros(gdat.numbcomp)
        if gdat.stdvcosiprio is None:
            gdat.stdvcosiprio = 0.05 + np.zeros(gdat.numbcomp)
        if gdat.stdvecosprio is None:
            gdat.stdvecosprio = 0.1 + np.zeros(gdat.numbcomp)
        if gdat.stdvesinprio is None:
            gdat.stdvesinprio = 0.1 + np.zeros(gdat.numbcomp)
        if gdat.stdvrvsaprio is None:
            gdat.stdvrvsaprio = 0.001 + np.zeros(gdat.numbcomp)
        
        # others
        if gdat.projoblqprio is None:
            gdat.projoblqprio = 0. + np.zeros(gdat.numbcomp)
        if gdat.stdvprojoblqprio is None:
            gdat.stdvprojoblqprio = 10. + np.zeros(gdat.numbcomp)
        
        # order planets with respect to period
        if gdat.typeprioplan != 'inpt':
            
            if gdat.typeverb > 0:
                print('Sorting the planets with respect to orbital period...')
            
            indxcompsort = np.argsort(gdat.periprio)
            
            gdat.booltrancomp = gdat.booltrancomp[indxcompsort]
            gdat.rratprio = gdat.rratprio[indxcompsort]
            gdat.rsmaprio = gdat.rsmaprio[indxcompsort]
            gdat.epocprio = gdat.epocprio[indxcompsort]
            gdat.periprio = gdat.periprio[indxcompsort]
            gdat.cosiprio = gdat.cosiprio[indxcompsort]
            gdat.ecosprio = gdat.ecosprio[indxcompsort]
            gdat.esinprio = gdat.esinprio[indxcompsort]
            gdat.rvsaprio = gdat.rvsaprio[indxcompsort]
        
            gdat.duraprio = gdat.duraprio[indxcompsort]
        
        # if stellar properties are NaN, use Solar defaults
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

        # if stellar property uncertainties are NaN, use 10%
        for featstar in gdat.listfeatstar:
            if (not hasattr(gdat, 'stdv' + featstar) or getattr(gdat, 'stdv' + featstar) is None or not np.isfinite(getattr(gdat, 'stdv' + featstar))) \
                                                                        and not (featstar == 'rascstar' or featstar == 'declstar'):
                print('featstar')
                print(featstar)
                setattr(gdat, 'stdv' + featstar, 0.5 * getattr(gdat, featstar))
                if gdat.typeverb > 0:
                    print('Setting %s uncertainty to 50%%!' % featstar)

        if gdat.typeverb > 0:
            print('Stellar priors:')
            print('gdat.rascstar')
            print(gdat.rascstar)
            print('gdat.declstar')
            print(gdat.declstar)
            print('gdat.radistar [R_S]')
            print(gdat.radistar)
            print('gdat.stdvradistar [R_S]')
            print(gdat.stdvradistar)
            print('gdat.massstar')
            print(gdat.massstar)
            print('gdat.stdvmassstar')
            print(gdat.stdvmassstar)
            print('gdat.vsiistar')
            print(gdat.vsiistar)
            print('gdat.stdvvsiistar')
            print(gdat.stdvvsiistar)
            print('gdat.massstar [M_S]')
            print(gdat.massstar)
            print('gdat.stdvmassstar [M_S]')
            print(gdat.stdvmassstar)
            print('gdat.tmptstar')
            print(gdat.tmptstar)
            print('gdat.stdvtmptstar')
            print(gdat.stdvtmptstar)
            
            print('Planetary priors:')
            print('gdat.rratprio')
            print(gdat.rratprio)
            print('gdat.rsmaprio')
            print(gdat.rsmaprio)
            print('gdat.epocprio')
            print(gdat.epocprio)
            print('gdat.periprio')
            print(gdat.periprio)
            print('gdat.cosiprio')
            print(gdat.cosiprio)
            print('gdat.ecosprio')
            print(gdat.ecosprio)
            print('gdat.esinprio')
            print(gdat.esinprio)
            print('gdat.rvsaprio')
            print(gdat.rvsaprio)
            print('gdat.stdvrratprio')
            print(gdat.stdvrratprio)
            print('gdat.stdvrsmaprio')
            print(gdat.stdvrsmaprio)
            print('gdat.stdvepocprio')
            print(gdat.stdvepocprio)
            print('gdat.stdvperiprio')
            print(gdat.stdvperiprio)
            print('gdat.stdvcosiprio')
            print(gdat.stdvcosiprio)
            print('gdat.stdvecosprio')
            print(gdat.stdvecosprio)
            print('gdat.stdvesinprio')
            print(gdat.stdvesinprio)
            print('gdat.stdvrvsaprio')
            print(gdat.stdvrvsaprio)
        
            if not np.isfinite(gdat.rratprio).all():
                print('rrat is infinite!')
            if not np.isfinite(gdat.rsmaprio).all():
                print('rsma is infinite!')
            if not np.isfinite(gdat.epocprio).all():
                print('epoc is infinite!')
            if not np.isfinite(gdat.periprio).all():
                print('peri is infinite!')
            if not np.isfinite(gdat.cosiprio).all():
                print('cosi is infinite!')
            if not np.isfinite(gdat.ecosprio).all():
                print('ecos is infinite!')
            if not np.isfinite(gdat.esinprio).all():
                print('esin is infinite!')
            if not np.isfinite(gdat.rvsaprio).all():
                print('rvsa is infinite!')

        # carry over RV data as is, without any detrending
        gdat.arrytser['bdtr'][1] = gdat.arrytser['raww'][1]
        gdat.listarrytser['bdtr'][1] = gdat.listarrytser['raww'][1]
        
        # concatenate time-series data from different instruments
        gdat.arrytsertotl = [[] for b in gdat.indxdatatser]
        for b in gdat.indxdatatser:
            if gdat.numbinst[b] > 0:
                gdat.arrytsertotl[b] = np.concatenate(gdat.arrytser['raww'][b], axis=0)
        dictmileoutp['arrytsertotl'] = gdat.arrytsertotl
        
        # plot LS periodogram
        if gdat.boollspe:
            if gdat.typeverb > 0:
                print('Plotting LS periodograms...')
            for b in gdat.indxdatatser:
                if gdat.numbinst[b] > 0:
                    
                    if gdat.numbinst[b] > 1:
                        strgextn = '%s' % (gdat.liststrgtser[b])
                        perilspe, powrlspe = ephesus.exec_lspe(gdat.arrytsertotl[b], pathimag=gdat.pathimagtarg, strgextn=strgextn, maxmfreq=maxmfreqlspe)
                    
                    for p in gdat.indxinst[b]:
                        strgextn = '%s_%s' % (gdat.liststrgtser[b], gdat.liststrginst[b][p]) 
                        perilspe, powrlspe = ephesus.exec_lspe(gdat.arrytser['raww'][b][p], pathimag=gdat.pathimagtarg, strgextn=strgextn, maxmfreq=maxmfreqlspe)
        
            dictmileoutp['perilspe'] = perilspe
            dictmileoutp['powrlspe'] = powrlspe
        
        if 'exop' in gdat.listtypeobjt or 'bhol' in gdat.listtypeobjt:
            # determine times during transits
            gdat.listindxtimeoutt = [[[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser] for j in gdat.indxcomp]
            gdat.listindxtimetran = [[[[[] for m in range(2)] for p in gdat.indxinst[b]] for b in gdat.indxdatatser] for j in gdat.indxcomp]
            gdat.listindxtimetranchun = [[[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser] for j in gdat.indxcomp]
            gdat.listindxtimeclen = [[[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser] for j in gdat.indxcomp]
            gdat.numbtimeclen = [[np.empty((gdat.numbcomp), dtype=int) for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
            for b in gdat.indxdatatser:
                for p in gdat.indxinst[b]:
                    for j in gdat.indxcomp:
                        if not np.isfinite(gdat.duraprio[j]):
                            continue
                        # determine time mask
                        for y in gdat.indxchun[b][p]:
                            gdat.listindxtimetranchun[j][b][p][y] = ephesus.retr_indxtimetran(gdat.listarrytser['bdtr'][b][p][y][:, 0], \
                                                                                                   gdat.epocprio[j], gdat.periprio[j], gdat.duraprio[j])
                        
                        gdat.listindxtimetran[j][b][p][0] = ephesus.retr_indxtimetran(gdat.arrytser['bdtr'][b][p][:, 0], \
                                                                                                 gdat.epocprio[j], gdat.periprio[j], gdat.duraprio[j])
                        
                        # floor of the secondary
                        gdat.listindxtimetran[j][b][p][1] = ephesus.retr_indxtimetran(gdat.arrytser['bdtr'][b][p][:, 0], \
                                                                             gdat.epocprio[j], gdat.periprio[j], gdat.duraprio[j], boolseco=True)
                        
                        gdat.listindxtimeoutt[j][b][p] = np.setdiff1d(np.arange(gdat.arrytser['bdtr'][b][p].shape[0]), gdat.listindxtimetran[j][b][p][0])
                
            for b in gdat.indxdatatser:
                for p in gdat.indxinst[b]:
                    for j in gdat.indxcomp:
                        # clean times for each planet
                        listindxtimetemp = []
                        for jj in gdat.indxcomp:
                            if jj != j:
                                listindxtimetemp.append(gdat.listindxtimetran[jj][b][p][0])
                        if len(listindxtimetemp) > 0:
                            listindxtimetemp = np.concatenate(listindxtimetemp)
                            listindxtimetemp = np.unique(listindxtimetemp)
                        else:
                            listindxtimetemp = np.array([])
                        gdat.listindxtimeclen[j][b][p] = np.setdiff1d(np.arange(gdat.arrytser['bdtr'][b][p].shape[0]), listindxtimetemp)
                        gdat.numbtimeclen[b][p][j] = gdat.listindxtimeclen[j][b][p].size
                    
                    #for y in gdat.indxchun[b][p]:
                    #    for i in gdat.indxsplnregi[b][p][y]:
                    #        if gdat.listobjtspln[b][p][y] is not None and gdat.listobjtspln[b][p][y][i] is not None:
                    #            # produce a table for the spline coefficients
                    #            fileoutp = open(gdat.pathdatatarg + 'coefbdtr.csv', 'w')
                    #            fileoutp.write(' & ')
                    #            for i in gdat.indxsplnregi[b][p][y]:
                    #                print('$\beta$:', gdat.listobjtspln[b][p][y][i].get_coeffs())
                    #                print('$t_k$:', gdat.listobjtspln[b][p][y][i].get_knots())
                    #                print
                    #            fileoutp.write('\\hline\n')
                    #            fileoutp.close()
        
        # seti
        gdat.fracineg = np.zeros(2)
        gdat.listindxtimetranineg = [[[[[] for k in range(4)] for p in gdat.indxinst[b]] for b in gdat.indxdatatser] for j in gdat.indxcomp]
        gdat.durafullprio = (1. - gdat.rratprio) / (1. + gdat.rratprio) * gdat.duraprio
        for p in gdat.indxinst[0]:
            for j in gdat.indxcomp:
                if not gdat.booltrancomp[j]:
                    continue

                gdat.listindxtimetranineg[j][0][p][0] = ephesus.retr_indxtimetran(gdat.arrytser['bdtr'][0][p][:, 0], gdat.epocprio[j], gdat.periprio[j], \
                                                                                                                    gdat.duraprio[j], durafull=gdat.durafullprio[j], typeineg='ingrinit')
                gdat.listindxtimetranineg[j][0][p][1] = ephesus.retr_indxtimetran(gdat.arrytser['bdtr'][0][p][:, 0], gdat.epocprio[j], gdat.periprio[j], \
                                                                                                                    gdat.duraprio[j], durafull=gdat.durafullprio[j], typeineg='ingrfinl')
                gdat.listindxtimetranineg[j][0][p][2] = ephesus.retr_indxtimetran(gdat.arrytser['bdtr'][0][p][:, 0], gdat.epocprio[j], gdat.periprio[j], \
                                                                                                                    gdat.duraprio[j], durafull=gdat.durafullprio[j], typeineg='eggrinit')
                gdat.listindxtimetranineg[j][0][p][3] = ephesus.retr_indxtimetran(gdat.arrytser['bdtr'][0][p][:, 0], gdat.epocprio[j], gdat.periprio[j], \
                                                                                                                    gdat.duraprio[j], durafull=gdat.durafullprio[j], typeineg='eggrfinl')
                
                for k in range(2):
                    indxtimefrst = gdat.listindxtimetranineg[j][0][p][2*k+0]
                    indxtimeseco = gdat.listindxtimetranineg[j][0][p][2*k+1]
                    if indxtimefrst.size == 0 or indxtimeseco.size == 0:
                        continue
                    rflxinit = np.mean(gdat.arrytser['bdtr'][0][p][indxtimefrst, 1])
                    rflxfinl = np.mean(gdat.arrytser['bdtr'][0][p][indxtimeseco, 1])
                    gdat.fracineg[k] = rflxinit / rflxfinl
                if False and (gdat.fracineg == 0).any():
                    print('rflxinit')
                    print(rflxinit)
                    print('rflxfinl')
                    print(rflxfinl)
                    print('gdat.arrytser[bdtr][0][p]')
                    summgene(gdat.arrytser['bdtr'][0][p])
                    print('gdat.arrytser[bdtr][0][p][:, 1]')
                    summgene(gdat.arrytser['bdtr'][0][p][:, 1])
                    print('gdat.arrytser[bdtr][0][p][indxtimefrst, 1]')
                    summgene(gdat.arrytser['bdtr'][0][p][indxtimefrst, 1])
                    print('gdat.arrytser[bdtr][0][p][indxtimeseco, 1]')
                    summgene(gdat.arrytser['bdtr'][0][p][indxtimeseco, 1])
                    raise Exception('')

                path = gdat.pathdatatarg + 'fracineg%04d.csv' % j
                np.savetxt(path, gdat.fracineg, delimiter=',')
                dictmileoutp['fracineg%04d' % j] = gdat.fracineg
        
        ## bin the light curve
        gdat.numbbinspcurtotl = 100
        gdat.delttimebind = 1. # [days]
        if 'exop' in gdat.listtypeobjt or 'bhol' in gdat.listtypeobjt:
            gdat.delttimebindzoom = gdat.duraprio / 24. / 50.
            
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                gdat.arrytser['bdtrbind'][b][p] = ephesus.rebn_tser(gdat.arrytser['bdtr'][b][p], delt=gdat.delttimebind)
                for y in gdat.indxchun[b][p]:
                    gdat.listarrytser['bdtrbind'][b][p][y] = ephesus.rebn_tser(gdat.listarrytser['bdtr'][b][p][y], delt=gdat.delttimebind)
                    
                    path = gdat.pathdatatarg + 'arrytserbdtrbind%s%s.csv' % (gdat.liststrginst[b][p], gdat.liststrgchun[b][p][y])
                    if gdat.typeverb > 0:
                        print('Writing to %s' % path)
                    np.savetxt(path, gdat.listarrytser['bdtrbind'][b][p][y], delimiter=',', \
                                                    header='time,%s,%s_err' % (gdat.liststrgtseralle[b], gdat.liststrgtseralle[b]))
                
                    if gdat.boolplottser:
                        plot_tser(gdat, b, p, y, 'bdtrbind')
            
        gdat.listtime = [[[[] for y in gdat.indxchun[b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.time = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.indxtime = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.numbtime = [np.empty(gdat.numbinst[b], dtype=int) for b in gdat.indxdatatser]
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                gdat.time[b][p] = gdat.arrytser['bdtr'][b][p][:, 0]
                gdat.numbtime[b][p] = gdat.time[b][p].size
                gdat.indxtime[b][p] = np.arange(gdat.numbtime[b][p])
                for y in gdat.indxchun[b][p]:
                    gdat.listtime[b][p][y] = gdat.listarrytser['bdtr'][b][p][y][:, 0]
    
        if gdat.listindxchuninst is None:
            gdat.listindxchuninst = [gdat.indxchun]
    
        # plot raw data
        if gdat.typetarg != 'inpt' and 'TESS' in gdat.liststrginst[0] and gdat.listarrytsersapp is not None:
            for b in gdat.indxdatatser:
                for p in gdat.indxinst[b]:
                    if gdat.liststrginst[b][p] != 'TESS':
                        continue
                    for y in gdat.indxchun[b][p]:
                        path = gdat.pathdatatarg + gdat.liststrgchun[b][p][y] + '_SAP.csv'
                        if gdat.typeverb > 0:
                            print('Writing to %s...' % path)
                        np.savetxt(path, gdat.listarrytsersapp[y], delimiter=',', header='time,flux,flux_err')
                        path = gdat.pathdatatarg + gdat.liststrgchun[b][p][y] + '_PDCSAP.csv'
                        if gdat.typeverb > 0:
                            print('Writing to %s...' % path)
                        np.savetxt(path, gdat.listarrytserpdcc[y], delimiter=',', header='time,flux,flux_err')
            
            # plot PDCSAP and SAP light curves
            figr, axis = plt.subplots(2, 1, figsize=gdat.figrsizeydob)
            axis[0].plot(gdat.arrytsersapp[:, 0] - gdat.timeoffs, gdat.arrytsersapp[:, 1], color='k', marker='.', ls='', ms=1, rasterized=True)
            axis[1].plot(gdat.arrytserpdcc[:, 0] - gdat.timeoffs, gdat.arrytserpdcc[:, 1], color='k', marker='.', ls='', ms=1, rasterized=True)
            #axis[0].text(.97, .97, 'SAP', transform=axis[0].transAxes, size=20, color='r', ha='right', va='top')
            #axis[1].text(.97, .97, 'PDC', transform=axis[1].transAxes, size=20, color='r', ha='right', va='top')
            axis[1].set_xlabel('Time [BJD - %d]' % gdat.timeoffs)
            for a in range(2):
                axis[a].set_ylabel('Relative flux')
            
            plt.subplots_adjust(hspace=0.)
            path = gdat.pathimagtarg + 'lcurspoc_%s.%s' % (gdat.strgtarg, gdat.typefileplot)
            if gdat.typeverb > 0:
                print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
        
        # injection recovery test
    
    if 'exop' in gdat.listtypeobjt or 'bhol' in gdat.listtypeobjt:
        ### Doppler beaming
        print('temp: check Doppler beaming predictions.')
        gdat.binswlenbeam = np.linspace(0.6, 1., 101)
        gdat.meanwlenbeam = (gdat.binswlenbeam[1:] + gdat.binswlenbeam[:-1]) / 2.
        gdat.diffwlenbeam = (gdat.binswlenbeam[1:] - gdat.binswlenbeam[:-1]) / 2.
        x = 2.248 / gdat.meanwlenbeam
        gdat.funcpcurmodu = .25 * x * np.exp(x) / (np.exp(x) - 1.)
        gdat.consbeam = np.sum(gdat.diffwlenbeam * gdat.funcpcurmodu)

        #if ''.join(gdat.liststrgcomp) != ''.join(sorted(gdat.liststrgcomp)):
        #    print('Provided planet letters are not in order. Changing the TCE order to respect the letter order in plots (b, c, d, e)...')
        #    gdat.indxcomp = np.argsort(np.array(gdat.liststrgcomp))

    gdat.liststrgcompfull = np.empty(gdat.numbcomp, dtype='object')
    for j in gdat.indxcomp:
        gdat.liststrgcompfull[j] = gdat.labltarg + ' ' + gdat.liststrgcomp[j]

    ## augment object dictinary
    gdat.dictfeatobjt = dict()
    gdat.dictfeatobjt['namesyst'] = np.array([gdat.labltarg] * gdat.numbcomp)
    gdat.dictfeatobjt['nameplan'] = gdat.liststrgcompfull
    # temp
    gdat.dictfeatobjt['booltran'] = np.array([True] * gdat.numbcomp, dtype=bool)
    gdat.dictfeatobjt['boolfrst'] = np.array([True] + [False] * (gdat.numbcomp - 1), dtype=bool)
    gdat.dictfeatobjt['vmagsyst'] = np.zeros(gdat.numbcomp) + gdat.vmagsyst
    gdat.dictfeatobjt['jmagsyst'] = np.zeros(gdat.numbcomp) + gdat.jmagsyst
    gdat.dictfeatobjt['hmagsyst'] = np.zeros(gdat.numbcomp) + gdat.hmagsyst
    gdat.dictfeatobjt['kmagsyst'] = np.zeros(gdat.numbcomp) + gdat.kmagsyst
    gdat.dictfeatobjt['numbcompstar'] = np.zeros(gdat.numbcomp) + gdat.numbcomp
    gdat.dictfeatobjt['numbcomptranstar'] = np.zeros(gdat.numbcomp) + gdat.numbcomp
    
    if gdat.booldatatser and ('pbox' in gdat.listtypeanls or 'lspe' in gdat.listtypeanls):
        if gdat.dilu == 'lygos':
            if gdat.typeverb > 0:
                print('Calculating the contamination ratio...')
            gdat.contrati = lygos.retr_contrati()

        # correct for dilution
        #if gdat.typeverb > 0:
        #print('Correcting for dilution!')
        #if gdat.dilucorr is not None:
        #    gdat.arrytserdilu = np.copy(gdat.listarrytser['bdtr'][b][p][y])
        #if gdat.dilucorr is not None:
        #    gdat.arrytserdilu[:, 1] = 1. - gdat.dilucorr * (1. - gdat.listarrytser['bdtr'][b][p][y][:, 1])
        #gdat.arrytserdilu[:, 1] = 1. - gdat.contrati * gdat.contrati * (1. - gdat.listarrytser['bdtr'][b][p][y][:, 1])
        
        ## phase-fold and save the baseline-detrended light curve
        gdat.arrypcur = dict()

        gdat.arrypcur['quadbdtr'] = [[[[] for j in gdat.indxcomp] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.arrypcur['quadbdtrbindtotl'] = [[[[] for j in gdat.indxcomp] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.arrypcur['primbdtr'] = [[[[] for j in gdat.indxcomp] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.arrypcur['primbdtrbindtotl'] = [[[[] for j in gdat.indxcomp] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.arrypcur['primbdtrbindzoom'] = [[[[] for j in gdat.indxcomp] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
        gdat.liststrgpcur = ['bdtr', 'resi', 'modl']
        gdat.liststrgpcurcomp = ['modltotl', 'modlstel', 'modlplan', 'modlelli', 'modlpmod', 'modlnigh', 'modlbeam', 'bdtrplan']
        gdat.binsphasprimtotl = np.linspace(-0.5, 0.5, gdat.numbbinspcurtotl + 1)
        gdat.binsphasquadtotl = np.linspace(-0.25, 0.75, gdat.numbbinspcurtotl + 1)
        if 'exop' in gdat.listtypeobjt or 'bhol' in gdat.listtypeobjt:
            print('gdat.delttimebindzoom')
            print(gdat.delttimebindzoom)
            print('gdat.periprio')
            print(gdat.periprio)
            gdat.numbbinspcurzoom = (gdat.periprio / gdat.delttimebindzoom).astype(int)
            gdat.binsphasprimzoom = [[] for j in gdat.indxcomp]
            for j in gdat.indxcomp:
                if np.isfinite(gdat.duraprio[j]):
                    gdat.binsphasprimzoom[j] = np.linspace(-0.5, 0.5, gdat.numbbinspcurzoom[j] + 1)

            if gdat.typeverb > 0:
                print('Phase folding and binning the light curve...')
            for b in gdat.indxdatatser:
                for p in gdat.indxinst[b]:
                    for j in gdat.indxcomp:

                        gdat.arrypcur['primbdtr'][b][p][j] = ephesus.fold_tser(gdat.arrytser['bdtr'][b][p][gdat.listindxtimeclen[j][b][p], :], \
                                                                                                                gdat.epocprio[j], gdat.periprio[j])
                        
                        gdat.arrypcur['primbdtrbindtotl'][b][p][j] = ephesus.rebn_tser(gdat.arrypcur['primbdtr'][b][p][j], \
                                                                                                            binsxdat=gdat.binsphasprimtotl)
                        
                        if np.isfinite(gdat.duraprio[j]):
                            gdat.arrypcur['primbdtrbindzoom'][b][p][j] = ephesus.rebn_tser(gdat.arrypcur['primbdtr'][b][p][j], \
                                                                                                            binsxdat=gdat.binsphasprimzoom[j])
                        
                        gdat.arrypcur['quadbdtr'][b][p][j] = ephesus.fold_tser(gdat.arrytser['bdtr'][b][p][gdat.listindxtimeclen[j][b][p], :], \
                                                                                                gdat.epocprio[j], gdat.periprio[j], phasshft=0.25)
                        
                        gdat.arrypcur['quadbdtrbindtotl'][b][p][j] = ephesus.rebn_tser(gdat.arrypcur['quadbdtr'][b][p][j], \
                                                                                                            binsxdat=gdat.binsphasquadtotl)
                        
                        # write (good for Vespa)
                        path = gdat.pathdatatarg + 'arrypcurprimbdtrbind_%s_%s.csv' % (gdat.liststrgcomp[j], gdat.liststrginst[b][p])
                        if gdat.typeverb > 0:
                            print('Writing to %s...' % path)
                        temp = np.copy(gdat.arrypcur['primbdtrbindtotl'][b][p][j])
                        temp[:, 0] *= gdat.periprio[j]
                        np.savetxt(path, temp, delimiter=',', header='phase,%s,%s_err' % (gdat.liststrgtseralle[b], gdat.liststrgtseralle[b]))
            
    # number of samples to draw from the prior
    gdat.numbsamp = 10000
    gdat.indxsamp = np.arange(gdat.numbsamp)
    
    if gdat.booldatatser and ('pbox' in gdat.listtypeanls or 'lspe' in gdat.listtypeanls):
        plot_pser(gdat, 'primbdtr')
    
    if gdat.boolplotprio and gdat.numbcomp > 0:
        for typemodl in gdat.listtypemodl:
            calc_prop(gdat, 'prio')
            if gdat.boolplotprop:
                plot_prop(gdat, 'prio')
    
    if gdat.typeinfe == 'alle' and gdat.booldatatser:
        if gdat.boolallebkgdgaus:
            # background allesfitter run
            if gdat.typeverb > 0:
                print('Setting up the background allesfitter run...')
            
            if not gdat.boolalleprev['bkgd']['para']:
                writ_filealle(gdat, 'params.csv', gdat.pathallebkgd, gdat.dictdictallepara[typemodl], dictalleparadefa)
            
            ## mask out the transits for the background run
            for b in gdat.indxdatatser:
                for p in gdat.indxinst[b]:
                    path = gdat.pathallebkgd + gdat.liststrgchun[b][p][y]  + '.csv'
                    if not os.path.exists(path):
                        indxtimebkgd = np.setdiff1d(gdat.indxtime, np.concatenate(gdat.listindxtimemask[jj][b][p][0]))
                        gdat.arrytserbkgd = gdat.listarrytser['bdtr'][b][p][y][indxtimebkgd, :]
                        if gdat.typeverb > 0:
                            print('Writing to %s...' % path)
                        np.savetxt(path, gdat.arrytserbkgd, delimiter=',', header='time,%s,%s_err' % (gdat.liststrgtseralle[b], gdat.liststrgtseralle[b]))
                    else:
                        if gdat.typeverb > 0:
                            print('OoT light curve available for the background allesfitter run at %s.' % path)
                    
                    #liststrg = list(gdat.objtallebkgd.posterior_params.keys())
                    #for k, strg in enumerate(liststrg):
                    #   post = gdat.objtallebkgd.posterior_params[strg]
                    #   linesplt = '%s' % gdat.objtallebkgd.posterior_params_at_maximum_likelihood[strg][0]
    
    
    if gdat.labltarg == 'WASP-121':
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
        gdat.meanwlenband = gdat.data[:, 0] * 1e-3
        gdat.thptband = gdat.data[:, 1]
    
    print('gdat.listtypemodl')
    print(gdat.listtypemodl)

    for typemodl in gdat.listtypemodl:
        if gdat.typeinfe == 'mile':
            
            meangauspara = None
            stdvgauspara = None

            gdat.numbsampwalk = 4000
            gdat.numbsampburnwalkinit = 0
            gdat.numbsampburnwalk = 3000
            
            gdat.time = gdat.arrytser['bdtr'][0][0][:, 0] - gdat.timeoffs
            gdat.rflx = gdat.arrytser['bdtr'][0][0][:, 1]
            gdat.stdvrflx = gdat.arrytser['bdtr'][0][0][:, 2]
            gdat.varirflx = gdat.stdvrflx**2
            
            gdat.strgextn = gdat.strgcnfg
            if typemodl == 'rise':
                    
                # list of parameter names
                listnamepara = ['timerise', 'coeflinerise', 'coefquadrise', 'coefline']
                # list of parameter labels and units
                listlablpara = [['$T_0$', 'BJD-%d' % gdat.timeoffs], ['$u_0$', ''], ['$u_1$', ''], ['$c$', '']]
                # list of parameter scalings
                listscalpara = ['self', 'self', 'self', 'self']
                # list of parameter minima
                listminmpara = [188, 0., -10., -10.]
                # list of parameter maxima
                listmaxmpara = [195, 10., 10., 10.]
                    
                gdat.typemodl = typemodl
                dictparafitt, dictparaderi = tdpy.mcmc.samp(gdat, gdat.pathimagtarg, gdat.numbsampwalk, retr_llik_mile, \
                                                            listnamepara, listlablpara, listscalpara, listminmpara, listmaxmpara, \
                                                            numbsampburnwalk=gdat.numbsampburnwalk, strgextn=gdat.strgextn)
                indxsampmpos = np.argmax(dictparafitt['lpos'])

                for b in gdat.indxdatatser:
                    for p in gdat.indxinst[b]:
                        for y in gdat.indxchun[b][p]:
                            
                            if gdat.boolplottser:
                                rflxmodl, dflxline, dflxrise = ephesus.retr_rflxmodlrise(gdat.time, dictparafitt['timerise'][indxsampmpos], \
                                                                                                                            dictparafitt['coeflinerise'][indxsampmpos], \
                                                                                                                            dictparafitt['coefquadrise'][indxsampmpos], \
                                                                                                                            dictparafitt['coefline'][indxsampmpos])
                                #rflxline = 1. + dflxline
                                rflxrise = 1. + dflxrise

                                dictmodl = dict()
                                #dictmodl['medimodl'] = {'lcur': rflxmodl, 'time': gdat.time + gdat.timeoffs, 'labl': 'Model'}
                                #dictmodl['mediline'] = {'lcur': rflxline, 'time': gdat.time + gdat.timeoffs, 'labl': 'Trend'}
                                dictmodl['medirise'] = {'lcur': rflxrise, 'time': gdat.time + gdat.timeoffs, 'labl': 'Model'}
                                
                                strgextn = gdat.strgextn
                                timedata = gdat.time + gdat.timeoffs
                                lcurdata = gdat.listarrytser['bdtr'][b][p][y][:, 1] - dflxline
                                
                                pathplot = ephesus.plot_lcur(gdat.pathimagtarg, timedata=timedata, timeoffs=gdat.timeoffs, lcurdata=lcurdata, strgextn=strgextn, dictmodl=dictmodl)






            if typemodl == 'spot':

                # for each spot multiplicity, fit the spot model
                for gdat.numbspot in listindxnumbspot:
                    
                    print('gdat.numbspot')
                    print(gdat.numbspot)

                    # list of parameter labels and units
                    listlablpara = [['$u_1$', ''], ['$u_2$', ''], ['$P$', 'days'], ['$i$', 'deg'], ['$\\rho$', ''], ['$C$', '']]
                    # list of parameter scalings
                    listscalpara = ['self', 'self', 'self', 'self', 'self', 'self']
                    # list of parameter minima
                    listminmpara = [-1., -1., 0.2,   0.,  0.,-1e-1]
                    # list of parameter maxima
                    listmaxmpara = [ 3.,  3., 0.4, 89.9, 0.6, 1e-1]
                    
                    for numbspottemp in range(gdat.numbspot):
                        listlablpara += [['$\\theta_{%d}$' % numbspottemp, 'deg'], ['$\\phi_{%d}$' % numbspottemp, 'deg'], ['$R_{%d}$' % numbspottemp, '']]
                        listscalpara += ['self', 'self', 'self']
                        listminmpara += [-90.,   0.,  0.]
                        listmaxmpara += [ 90., 360., 0.4]
                        if gdat.boolevol:
                            listlablpara += [['$T_{s;%d}$' % numbspottemp, 'day'], ['$\\sigma_{s;%d}$' % numbspottemp, '']]
                            listscalpara += ['self', 'self']
                            listminmpara += [gdat.minmtime, 0.1]
                            listmaxmpara += [gdat.maxmtime, 20.]
                            
                    dictparafitt, dictparaderi = tdpy.mcmc.samp(gdat, gdat.pathimagtarg, gdat.numbsampwalk, retr_llik_mile, \
                                                                listnamepara, listlablpara, listscalpara, listminmpara, listmaxmpara, \
                                                                numbsampburnwalk=gdat.numbsampburnwalk, strgextn=gdat.strgextn)

                    # plot light curve
                    figr, axis = plt.subplots(figsize=(8, 4))
                    # plot samples from the posterior
                    ## the sample indices which will be plotted
                    gdat.numbsampplot = 10
                    indxsampplot = np.random.choice(gdat.indxsamp, size=gdat.numbsampplot, replace=False)
                    indxsampplot = np.sort(indxsampplot)
                    listlcurmodl = np.empty((gdat.numbsampplot, gdat.numbtime))
                    listlcurmodlevol = np.empty((gdat.numbsampplot, gdat.numbspot, gdat.numbtime))
                    listlcurmodlspot = np.empty((gdat.numbsampplot, gdat.numbspot, gdat.numbtime))
                    for kk, k in enumerate(indxsampplot):
                        # calculate the model light curve for this parameter vector
                        listlcurmodl[kk, :], listlcurmodlevol[kk, :, :], listlcurmodlspot[kk, :, :] = ephesus.retr_rflxmodl(gdat, listpost[k, :])
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
                        rrat[n, :] = dictpara['rrat']
                    lati = np.median(lati, 0)
                    lngi = np.median(lngi, 0)
                    rrat = np.median(rrat, 0)

                    print('lati')
                    print(lati)
                    print('lngi')
                    print(lngi)
                    print('rrat')
                    print(rrat)
                    plot_moll(gdat, lati, lngi, rrat)
                    
                    #for k in indxsampplot:
                    #    lati = listpost[k, 1+0*gdat.numbparaspot+0]
                    #    lngi = listpost[k, 1+0*gdat.numbparaspot+1]
                    #    rrat = listpost[k, 1+0*gdat.numbparaspot+2]
                    #    plot_moll(gdat, lati, lngi, rrat)

                    for sp in ['right', 'top']:
                        axis.spines[sp].set_visible(False)

                    path = gdat.pathimagtarg + 'smap%s_ns%02d.pdf' % (strgtarg, gdat.numbspot)
                    if gdat.typeverb > 0:
                        print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()


            if typemodl == 'bhol':
                
                # number of total samples after burn-in
                gdat.numbsamp = (gdat.numbsampwalk - gdat.numbsampburnwalk) * numbwalk
                gdat.indxsamp = np.arange(gdat.numbsamp)
    
                ### MCMC
                if gdat.boolmcmc:
                    gdat.listlablpara = [['T$_0$', 'day'], ['P', 'day'], ['M', r'M$_s$']]
                    
                    gdat.numbtargwalk = 1000
                    gdat.numbtargburnwalk = 100

                    gdat.numbpara = len(gdat.listlablpara)
                    gdat.meanpara = np.empty(gdat.numbpara)
                    gdat.stdvpara = np.empty(gdat.numbpara)
                    gdat.minmpara = np.empty(gdat.numbpara)
                    gdat.maxmpara = np.empty(gdat.numbpara)
                    gdat.scalpara = np.empty(gdat.numbpara, dtype='object')
                    gdat.fittminmmasscomp = 1.
                    gdat.fittmaxmmasscomp = 10.
                    gdat.minmpara[0] = -10.
                    gdat.maxmpara[0] = 10.
                    #gdat.meanpara[1] = 8.964
                    #gdat.stdvpara[1] = 0.001
                    gdat.minmpara[1] = 1.
                    gdat.maxmpara[1] = 20.
                    gdat.minmpara[2] = gdat.fittminmmasscomp
                    gdat.maxmpara[2] = gdat.fittmaxmmasscomp
                    gdat.scalpara[0] = 'self'
                    gdat.scalpara[1] = 'self'
                    gdat.scalpara[2] = 'self'
    
                    gdat.bfitperi = 4.25 # [days]
                    gdat.stdvperi = 1e-2 * gdat.bfitperi # [days]
                    gdat.bfitduratran = 0.45 * 24. # [hours]
                    gdat.stdvduratran = 1e-1 * gdat.bfitduratran # [hours]
                    gdat.bfitamplslen = 0.14 # [relative]
                    gdat.stdvamplslen = 1e-1 * gdat.bfitamplslen # [relative]
                    
                    listlablpara = [['$R_s$', 'R$_{\odot}$'], ['$P$', 'days'], ['$M_c$', 'M$_{\odot}$'], ['$M_s$', 'M$_{\odot}$']]
                    listlablparaderi = [['$A$', ''], ['$D$', 'hours'], ['$a$', 'R$_{\odot}$'], ['$R_{Sch}$', 'R$_{\odot}$']]
                    listminmpara = np.array([ 0.01, 0.1, 1e3, 1e-5])
                    listmaxmpara = np.array([ 1e4, 100., 1e8, 1e3])
                    #listlablpara += [['$M$', '$M_E$'], ['$T_{0}$', 'BJD'], ['$P$', 'days']]
                    #listminmpara = np.concatenate([listminmpara, np.array([ 10., minmtime,  50.])])
                    #listmaxmpara = np.concatenate([listmaxmpara, np.array([1e4, maxmtime, 200.])])
                    meangauspara = None
                    stdvgauspara = None
                    numbpara = len(listlablpara)
                    indxpara = np.arange(numbpara)
                    listscalpara = ['self' for k in indxpara]
                
                gdat.limtpara = tdpy.retr_limtpara(gdat.scalpara, gdat.minmpara, gdat.maxmpara, gdat.meanpara, gdat.stdvpara)
                gdat.indxparahard = np.where(gdat.scalpara == 'self')[0]


        if gdat.typeinfe == 'alle':

            proc_alle(gdat, typemodl)
        
            gdat.arrytsermodlinit = [[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser]
            for b in gdat.indxdatatser:
                for p in gdat.indxinst[b]:
                    gdat.arrytsermodlinit[b][p] = np.empty((gdat.arrytser['bdtr'][b][p].shape[0], 3))
                    gdat.arrytsermodlinit[b][p][:, 0] = gdat.arrytser['bdtr'][b][p][:, 0]
                    gdat.arrytsermodlinit[b][p][:, 1] = gdat.objtalle[typemodl].get_initial_guess_model(gdat.liststrginst[b][p], 'flux', \
                                                                                                                    xx=gdat.arrytser['bdtr'][b][p][:, 0])
                    gdat.arrytsermodlinit[b][p][:, 2] = 0.


    # measure final time
    gdat.timefinl = modutime.time()
    gdat.timeexec = gdat.timefinl - gdat.timeinit
    if gdat.typeverb > 0:
        print('miletos ran in %.3g seconds.' % gdat.timeexec)

    dictmileoutp['timeexec'] = gdat.timeexec

    return dictmileoutp


