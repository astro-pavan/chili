# type: ignore

# %%
#!/usr/bin/env python
#
# Copyright (c) 2021, Kaustubh Hakim
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
#
# CHILI 1.0 
#
# equilibrium.py (module to calculate dissolved inorganic carbon and pH)
# 
# Minerals included: wollastonite: woll, enstatite: enst, ferrosilite: ferr,
#                    forsterite: fors, fayalite: faya, anorthite: anor, 
#                    albite: albi, K-feldspar: kfel, muscovite: musc,
#                    phlogopite: phlo, annite: anni, anthophyllite: anth, 
#                    grunerite: grun
# Rocks included:    basalt (anor, woll, enst, ferr, albi),
#                    peridotite (woll, enst, fors, faya),
#                    granite (kfel, albi, phlo, anni, quar)
#
# If you are using this code, please cite the following publication
# Hakim et al. (2021) Lithologic Controls on Silicate Weathering Regimes of
# Temperate Planets. The Planetary Science Journal 2. doi:10.3847/PSJ/abe1b8

# %%
# Import packages
import numpy as np
import csv
import pandas as pd
pd.options.mode.chained_assignment = None
from os import path
from scipy.optimize import curve_fit
from scipy.optimize import newton
from scipy.interpolate import CubicSpline
# from scipy.interpolate import interp2d
from scipy.interpolate import RegularGridInterpolator as interpnd 
from astropy.constants import R

from parameters import *
from legacy_interp2d import legacy_interp2d_wrapper
interp2d = legacy_interp2d_wrapper

# %%
def import_thermo_data(path_species):
    '''Import thermodynamic data for given species from the CHNOSZ database.
    
    Example column names for albite: species.formula, species.state, 
    species.ispecies, out.albite.T, out.albite.P, out.albite.logK, 
    out.albite.G, out.albite.H, out.albite.S, out.albite.V, out.albite.Cp
    
    Note
    ----
    speciesNames is an imported table with predefined names that should not 
    be changed. Column names with special characters such as '+', '-', ' ' 
    are automatically replaced by '.'.
    
    Parameters
    ----------
    path_species: string
        Path to file with list of species names 

    Returns
    -------
    KeqFuncs : dict of {str : float}
        2D interpolated function in temperature, pressure with 4-letter keys
    
    '''
    
    speciesNames = pd.read_csv(path_species)
    speciesDict  = {} 
    logKDict     = {}
    KeqFuncs     = {}
    T = np.array([])
    P = np.array([])

    for name,colName in zip(speciesNames['species name'], speciesNames['species col name']):

        data = pd.read_csv('./database/' + name + '_th', \
                                  usecols=['out.' + colName + '.T', \
                                           'out.' + colName + '.P', \
                                           'out.' + colName + '.logK'])
        
        if data['out.' + colName + '.logK'].isna().any():
            
            def func(X,a,b,c,d,e):
                x,y = X
                return a * x**3 + b * x**2 + c * x + d * y + e
            
            guess = (0.5, 0.5, 0.5, 0.5, 0.5)

            fitData = data.dropna()
            colParams = {}

            x = fitData['out.' + colName + '.T'].values
            y = fitData['out.' + colName + '.P'].values
            z = fitData['out.' + colName + '.logK'].values

            params = curve_fit(func, (x,y), z, guess)
            colParams['out.' + colName + '.logK'] = params[0]

            x = data[pd.isnull(data['out.' + colName + '.logK'])]['out.' + colName + '.T'].values
            y = data[pd.isnull(data['out.' + colName + '.logK'])]['out.' + colName + '.P'].values
            n = data[pd.isnull(data['out.' + colName + '.logK'])].index.astype(float).values

            data.loc[n, 'out.' + colName + '.logK'] = func((x,y), *colParams['out.' + colName + '.logK'])

        speciesDict[name] = data
        T = np.unique(data['out.' + colName + '.T'].values)
        P = np.unique(data['out.' + colName + '.P'].values)
        logKDict[name] = np.reshape(data['out.' + colName + '.logK'].values,(len(P),len(T)))

    KeqFuncs['woll'] = interp2d(T, P, 10**(      logKDict['Ca+2'] \
                                           +   2*logKDict['HCO3-'] \
                                           +     logKDict['SiO2'] \
                                           -     logKDict['wollastonite']  \
                                           -     logKDict['water'] \
                                           -   2*logKDict['carbon dioxide'] \
                                          ), kind='linear')
    KeqFuncs['enst'] = interp2d(T, P, 10**(      logKDict['Mg+2'] \
                                           +   2*logKDict['HCO3-'] \
                                           +     logKDict['SiO2'] \
                                           -     logKDict['enstatite'] \
                                           -     logKDict['water'] \
                                           -   2*logKDict['carbon dioxide'] \
                                          ), kind='linear')
    KeqFuncs['ferr'] = interp2d(T, P, 10**(      logKDict['Fe+2'] \
                                           +   2*logKDict['HCO3-'] \
                                           +     logKDict['SiO2'] \
                                           -     logKDict['ferrosilite'] \
                                           -     logKDict['water'] \
                                           -   2*logKDict['carbon dioxide'] \
                                          ), kind='linear')
    KeqFuncs['fors'] = interp2d(T, P, 10**(      logKDict['Mg+2'] \
                                           +   2*logKDict['HCO3-'] \
                                           + 1/2*logKDict['SiO2'] \
                                           - 1/2*logKDict['forsterite'] \
                                           -     logKDict['water'] \
                                           -   2*logKDict['carbon dioxide'] \
                                          ), kind='linear')
    KeqFuncs['faya'] = interp2d(T, P, 10**(      logKDict['Fe+2'] \
                                           +   2*logKDict['HCO3-'] \
                                           + 1/2*logKDict['SiO2'] \
                                           - 1/2*logKDict['fayalite'] \
                                           -     logKDict['water'] \
                                           -   2*logKDict['carbon dioxide'] \
                                          ), kind='linear')
    KeqFuncs['anor'] = interp2d(T, P, 10**(  1/2*logKDict['Ca+2'] \
                                           +     logKDict['HCO3-'] \
                                           + 1/2*logKDict['kaolinite'] \
                                           - 1/2*logKDict['anorthite'] \
                                           - 3/2*logKDict['water'] \
                                           -     logKDict['carbon dioxide'] \
                                          ), kind='linear')
    KeqFuncs['kfel'] = interp2d(T, P, 10**(      logKDict['K+'] \
                                           +     logKDict['HCO3-'] \
                                           + 1/2*logKDict['kaolinite'] \
                                           +   2*logKDict['SiO2'] \
                                           -     logKDict['K-feldspar'] \
                                           - 3/2*logKDict['water'] \
                                           -     logKDict['carbon dioxide'] \
                                          ), kind='linear')
    KeqFuncs['albi'] = interp2d(T, P, 10**(      logKDict['Na+'] \
                                           +     logKDict['HCO3-'] \
                                           + 1/2*logKDict['kaolinite'] \
                                           +   2*logKDict['SiO2'] \
                                           -     logKDict['albite'] \
                                           - 3/2*logKDict['water'] \
                                           -     logKDict['carbon dioxide'] \
                                          ), kind='linear')
    KeqFuncs['musc'] = interp2d(T, P, 10**(      logKDict['K+'] \
                                           +     logKDict['HCO3-'] \
                                           + 3/2*logKDict['kaolinite'] \
                                           -     logKDict['muscovite'] \
                                           - 5/2*logKDict['water'] \
                                           -     logKDict['carbon dioxide'] \
                                          ), kind='linear')
    KeqFuncs['phlo'] = interp2d(T, P, 10**(  1/3*logKDict['K+'] \
                                           +     logKDict['Mg+2'] \
                                           + 7/3*logKDict['HCO3-'] \
                                           + 1/6*logKDict['kaolinite'] \
                                           + 2/3*logKDict['SiO2'] \
                                           - 1/3*logKDict['phlogopite'] \
                                           - 7/6*logKDict['water'] \
                                           - 7/3*logKDict['carbon dioxide'] \
                                          ), kind='linear')
    KeqFuncs['anni'] = interp2d(T, P, 10**(  1/3*logKDict['K+'] \
                                           +     logKDict['Fe+2'] \
                                           + 7/3*logKDict['HCO3-'] \
                                           + 1/6*logKDict['kaolinite'] \
                                           + 2/3*logKDict['SiO2'] \
                                           - 1/3*logKDict['annite'] \
                                           - 7/6*logKDict['water'] \
                                           - 7/3*logKDict['carbon dioxide'] \
                                          ), kind='linear')
    KeqFuncs['anoh'] = interp2d(T, P, 10**(  1/2*logKDict['Ca+2'] \
                                           +     logKDict['HCO3-'] \
                                           + 1/2*logKDict['halloysite'] \
                                           - 1/2*logKDict['anorthite'] \
                                           - 3/2*logKDict['water'] \
                                           -     logKDict['carbon dioxide'] \
                                          ), kind='linear')
    KeqFuncs['kfeh'] = interp2d(T, P, 10**(      logKDict['K+'] \
                                           +     logKDict['HCO3-'] \
                                           + 1/2*logKDict['halloysite'] \
                                           +   2*logKDict['SiO2'] \
                                           -     logKDict['K-feldspar'] \
                                           - 3/2*logKDict['water'] \
                                           -     logKDict['carbon dioxide'] \
                                          ), kind='linear')
    KeqFuncs['albh'] = interp2d(T, P, 10**(      logKDict['Na+'] \
                                           +     logKDict['HCO3-'] \
                                           + 1/2*logKDict['halloysite'] \
                                           +   2*logKDict['SiO2'] \
                                           -     logKDict['albite'] \
                                           - 3/2*logKDict['water'] \
                                           -     logKDict['carbon dioxide'] \
                                          ), kind='linear')
    KeqFuncs['mush'] = interp2d(T, P, 10**(      logKDict['K+'] \
                                           +     logKDict['HCO3-'] \
                                           + 3/2*logKDict['halloysite'] \
                                           -     logKDict['muscovite'] \
                                           - 5/2*logKDict['water'] \
                                           -     logKDict['carbon dioxide'] \
                                          ), kind='linear')
    KeqFuncs['phlh'] = interp2d(T, P, 10**(  1/3*logKDict['K+'] \
                                           +     logKDict['Mg+2'] \
                                           + 7/3*logKDict['HCO3-'] \
                                           + 1/6*logKDict['halloysite'] \
                                           + 2/3*logKDict['SiO2'] \
                                           - 1/3*logKDict['phlogopite'] \
                                           - 7/6*logKDict['water'] \
                                           - 7/3*logKDict['carbon dioxide'] \
                                          ), kind='linear')
    KeqFuncs['annh'] = interp2d(T, P, 10**(  1/3*logKDict['K+'] \
                                           +     logKDict['Fe+2'] \
                                           + 7/3*logKDict['HCO3-'] \
                                           + 1/6*logKDict['halloysite'] \
                                           + 2/3*logKDict['SiO2'] \
                                           - 1/3*logKDict['annite'] \
                                           - 7/6*logKDict['water'] \
                                           - 7/3*logKDict['carbon dioxide'] \
                                          ), kind='linear')
    KeqFuncs['anth'] = interp2d(T, P, 10**(      logKDict['Mg+2'] \
                                           +   2*logKDict['HCO3-'] \
                                           + 8/7*logKDict['SiO2'] \
                                           - 1/7*logKDict['anthophyllite'] \
                                           - 6/7*logKDict['water'] \
                                           -   2*logKDict['carbon dioxide'] \
                                          ), kind='linear')
    KeqFuncs['grun'] = interp2d(T, P, 10**(      logKDict['Fe+2'] \
                                           +   2*logKDict['HCO3-'] \
                                           + 8/7*logKDict['SiO2'] \
                                           - 1/7*logKDict['grunerite'] \
                                           - 6/7*logKDict['water'] \
                                           -   2*logKDict['carbon dioxide'] \
                                          ), kind='linear')
    KeqFuncs['bica'] = interp2d(T, P, 10**(      logKDict['H+'] \
                                           +     logKDict['HCO3-'] \
                                           -     logKDict['water'] \
                                           -     logKDict['carbon dioxide'] \
                                          ), kind='linear')
    KeqFuncs['carb'] = interp2d(T, P, 10**(      logKDict['H+'] \
                                           +     logKDict['CO3-2'] \
                                           -     logKDict['HCO3-'] \
                                          ), kind='linear')
    KeqFuncs['wate'] = interp2d(T, P, 10**(      logKDict['H+'] \
                                           +     logKDict['OH-'] \
                                           -     logKDict['water'] \
                                          ), kind='linear')
    KeqFuncs['quar'] = interp2d(T, P, 10**(      logKDict['SiO2'] \
                                           -     logKDict['quartz'] \
                                          ), kind='linear')
    KeqFuncs['co2a'] = interp2d(T, P, 10**(      logKDict['CO2'] \
                                           -     logKDict['carbon dioxide'] \
                                          ), kind='linear')
    
    csvData  = pd.read_csv('./database/henry_diamond2003.csv')
    lnKHFunc = CubicSpline(csvData['T (K)'], csvData['ln (kH, MPa)'], extrapolate=True)
    K_H_Tb   = 10 / 55.5084 * np.exp(lnKHFunc(T)) * np.ones((len(P),len(T))) # convert MPa to bar
    Keq_CO2b = 1 / K_H_Tb
    KeqFuncs['co2b'] = interp2d(T, P, Keq_CO2b, kind='linear')
    
    K_H_Tc   = 1600 / 55.5084 * np.exp(-2400*(1/T-1/298)) * np.ones((len(P),len(T)))
    Keq_CO2c = 1 / K_H_Tc
    KeqFuncs['co2c'] = interp2d(T, P, Keq_CO2c, kind='linear')
    
    return KeqFuncs

# %%
def get_DICeq(x_CO2g, Temp, Pres, KeqFuncs):
    '''Create a callable function to determine eq. DIC, pH and DIC components.
    
    Calls functions specific to fluid-mineral or fluid-rock systems to
    calcualte DIC, pH and DIC components as a function of temperature, 
    partial pressure of CO2 and pressure. 
    
    Parameters
    ----------
    x_CO2g : float
        Partial pressure of CO2 (ppm)
    Temp : float
        Temperature of the reactions
    Pres : float
        Pressure of the reactions
    KeqFuncs : dict of {str : float}
        2D interpolated function in temperature, pressure with 4-letter keys
    
    Returns
    -------
    DICeqFuncs : dict of {str : float}
        2D interpolated functions of temperature, pressure with 4-letter keys
        
    '''

    names   = ['woll', 'enst', 'ferr', 'fors', 'faya', 'anor', 'kfel', 'albi',\
               'musc', 'phlo', 'anni', 'anth', 'grun', 'anoh', 'kfeh', 'albh',
               'mush', 'phlh', 'annh', 'basa', 'peri', 'gran', 'bash', 'grah']
    f_names = [woll_eq,enst_eq,ferr_eq,fors_eq,faya_eq,anor_eq,kfel_eq,albi_eq,\
               musc_eq,phlo_eq,anni_eq,anth_eq,grun_eq,anoh_eq,kfeh_eq,albh_eq,\
               mush_eq,phlh_eq,annh_eq,basa_eq,peri_eq,gran_eq,bash_eq,grah_eq]    

    DIC     = np.zeros((len(names), len(x_CO2g), len(Temp), len(Pres)))
    pH      = np.zeros((len(names), len(x_CO2g), len(Temp), len(Pres)))
    HCO3    = np.zeros((len(names), len(x_CO2g), len(Temp), len(Pres)))
    CO2     = np.zeros((len(names), len(x_CO2g), len(Temp), len(Pres)))
    CO3     = np.zeros((len(names), len(x_CO2g), len(Temp), len(Pres)))
    ALK     = np.zeros((len(names), len(x_CO2g), len(Temp), len(Pres)))
    DIV     = np.zeros((len(names), len(x_CO2g), len(Temp), len(Pres)))
    MON     = np.zeros((len(names), len(x_CO2g), len(Temp), len(Pres)))
    
    DICeqFuncs = {}

    # Points for interpolation
    points = (x_CO2g, Temp, Pres)
    
    # Make a 3D grid for each of the parameters
    x_CO2g, Temp, Pres = np.meshgrid(x_CO2g, Temp, Pres, indexing='ij')
    
    for i, func in enumerate(f_names):
        # Ellipsis notation is equivalent to doing :, :, : (and so on)
        #DIC[i, ...], pH[i, ...], HCO3[i, ...], CO2[i, ...], CO3[i, ...] = func(x_CO2g, Temp, Pres, KeqFuncs)
                                                    
        allSpecies   = func(x_CO2g, Temp, Pres, KeqFuncs)
        #print(allSpecies)
        DIC[i, ...]  = allSpecies['DIC']
        pH[i, ...]   = allSpecies['pH']
        HCO3[i, ...] = allSpecies['HCO3']
        CO2[i, ...]  = allSpecies['CO2']
        CO3[i, ...]  = allSpecies['CO3']
        ALK[i, ...]  = allSpecies['ALK']
        DIV[i, ...]  = allSpecies['DIV']
        MON[i, ...]  = allSpecies['MON']
        
        DICeqFuncs[names[i]] = {}

        DICeqFuncs[names[i]]['DIC']  = interpnd(points=points, values=DIC [i])
        DICeqFuncs[names[i]]['pH']   = interpnd(points=points, values=pH  [i])
        DICeqFuncs[names[i]]['HCO3'] = interpnd(points=points, values=HCO3[i])
        DICeqFuncs[names[i]]['CO2']  = interpnd(points=points, values=CO2 [i])
        DICeqFuncs[names[i]]['CO3']  = interpnd(points=points, values=CO3 [i])
        DICeqFuncs[names[i]]['ALK']  = interpnd(points=points, values=ALK [i])
        DICeqFuncs[names[i]]['DIV']  = interpnd(points=points, values=DIV [i])
        DICeqFuncs[names[i]]['MON']  = interpnd(points=points, values=MON [i])
    
    return DICeqFuncs


# %%
def woll_eq(x_CO2g, Temp, Pres, KeqFuncs):
    '''Calculate DIC, pH and DIC components for wollastonite dissolution.
    
    Solves for the bicarbonate ion activity as a function of temperature, 
    partial pressure of CO2 and pressure. Simultaneous solution of activities
    of all species in the system of reactions.
    Polynomial equation of the form: a * x**3 + b * x**2 + c 
    Reactions included: wollastonite dissolution, atmospheric CO2 dissolution,
    bicarbonate ion formation, carbonate ion formation, water autoionization.
    
    Parameters
    ----------
    x_CO2g : float
        Partial pressure of CO2 (ppm)
    Temp : float
        Temperature of the reactions        
    Pres : float
        Pressure of the reactions
    KeqFuncs : dict of {str : float}
        2D interpolated function in temperature, pressure with 4-letter keys
    
    Returns
    -------
    dic : float
        Total dissolved inorganic carbon
    pH : float
        Negative logarithm of proton activity    
    act_HCO3_m : float
        Activity of the bicarbonate ion
    act_CO2_aq : float
        Activity of aqueous CO2
    act_CO3_mm : float
        Activity of the carbonate ion
        
    '''

    K_woll          =   KeqFuncs['woll'](Temp,Pres)
    K_bica          =   KeqFuncs['bica'](Temp,Pres)
    K_carb          =   KeqFuncs['carb'](Temp,Pres)
    K_wate          =   KeqFuncs['wate'](Temp,Pres)
    K_co2a          =   KeqFuncs['co2a'](Temp,Pres)
    act_CO2_aq      =   x_CO2g * K_co2a

    a_vals               =   2 * K_carb
    b_vals               =   K_wate + K_bica * x_CO2g
    c_vals               =   - x_CO2g**2 * (K_bica**2 + 2*K_bica * K_woll**(1/2))
    d_vals               =   0

    def f(x, a, b, c, d):
        return a * x**3 + b * x**2  +  c
    def f1(x, a, b, c, d):
        return 3 * a * x**2 + 2 * b * x
    def f2(x, a, b, c, d):
        return 6 * a * x + 2 * b
    
    x_init  = x_CO2g**(1/2)

    x0 = newton(f, x_init, fprime=f1, fprime2=f2, args=(a_vals, b_vals, c_vals, d_vals))

    act_HCO3_m      = x0
    act_CO3_mm      = K_carb / K_bica * act_HCO3_m**2 / x_CO2g
    alk             = 2 * act_CO3_mm + act_HCO3_m
    dic             = act_HCO3_m + act_CO2_aq + act_CO3_mm
    act_H_p         = K_bica * x_CO2g  / act_HCO3_m
    pH              = - np.log10(act_H_p)
    act_SiO2        = 0.5 * act_HCO3_m
    act_DIV_pp      = 0.5 * act_HCO3_m
    act_MON_p       = 0
    allSpecies = {'HCO3':act_HCO3_m, 'CO3':act_CO3_mm, 'ALK':alk, 'DIC':dic, 'pH':pH, 'H':act_H_p,
                  'SiO2':act_SiO2, 'DIV':act_DIV_pp, 'MON':act_MON_p, 'CO2':act_CO2_aq}
    return allSpecies

# %%
def enst_eq(x_CO2g, Temp, Pres, KeqFuncs):
    '''Calculate DIC, pH and DIC components for enstatite dissolution.
    
    Solves for the bicarbonate ion activity as a function of temperature, 
    partial pressure of CO2 and pressure. Simultaneous solution of activities
    of all species in the system of reactions.
    Polynomial equation of the form: a * x**3 + b * x**2 + c 
    Reactions included: enstatite dissolution, atmospheric CO2 dissolution,
    bicarbonate ion formation, carbonate ion formation, water autoionization.
    
    Parameters
    ----------
    x_CO2g : float
        Partial pressure of CO2 (ppm)
    Temp : float
        Temperature of the reactions        
    Pres : float
        Pressure of the reactions
    KeqFuncs : dict of {str : float}
        2D interpolated function in temperature, pressure with 4-letter keys
    
    Returns
    -------
    dic : float
        Total dissolved inorganic carbon
    pH : float
        Negative logarithm of proton activity    
    act_HCO3_m : float
        Activity of the bicarbonate ion
    act_CO2_aq : float
        Activity of aqueous CO2
    act_CO3_mm : float
        Activity of the carbonate ion
        
    '''


    K_enst          =   KeqFuncs['enst'](Temp,Pres)
    K_bica          =   KeqFuncs['bica'](Temp,Pres)
    K_carb          =   KeqFuncs['carb'](Temp,Pres)
    K_wate          =   KeqFuncs['wate'](Temp,Pres)
    K_co2a          =   KeqFuncs['co2a'](Temp,Pres)
    act_CO2_aq      =   x_CO2g * K_co2a
    a_vals               =   2 * K_carb
    b_vals               =   K_wate + K_bica * x_CO2g
    c_vals               =   - x_CO2g**2 * (K_bica**2 + 2*K_bica * K_enst**(1/2))
    d_vals               =   0
    def f(x, a, b, c, d):
        return a * x**3 + b * x**2  +  c 
    def f1(x, a, b, c, d):
        return 3 * a * x**2 + 2 * b * x
    def f2(x, a, b, c, d):
        return 6 * a * x + 2 * b 
    x_init  = x_CO2g**(1/2)
    x0 = newton(f, x_init, fprime=f1, fprime2=f2, args=(a_vals, b_vals, c_vals, d_vals))
    act_HCO3_m      = x0
    act_CO3_mm      = K_carb / K_bica * act_HCO3_m**2 / x_CO2g
    alk             = 2 * act_CO3_mm + act_HCO3_m
    dic             = act_HCO3_m + act_CO2_aq + act_CO3_mm
    act_H_p         = K_bica * x_CO2g  / act_HCO3_m
    pH              = - np.log10(act_H_p)
    act_SiO2        = 0.5 * act_HCO3_m
    act_DIV_pp      = 0.5 * act_HCO3_m
    act_MON_p       = 0
    allSpecies = {'HCO3':act_HCO3_m, 'CO3':act_CO3_mm, 'ALK':alk, 'DIC':dic, 'pH':pH, 'H':act_H_p,
                  'SiO2':act_SiO2, 'DIV':act_DIV_pp, 'MON':act_MON_p, 'CO2':act_CO2_aq}
    return allSpecies

# %%
def ferr_eq(x_CO2g, Temp, Pres, KeqFuncs):
    '''Calculate DIC, pH and DIC components for ferrosilite dissolution.
    
    Solves for the bicarbonate ion activity as a function of temperature, 
    partial pressure of CO2 and pressure. Simultaneous solution of activities
    of all species in the system of reactions.
    Polynomial equation of the form: a * x**3 + b * x**2 + c 
    Reactions included: ferrosilite dissolution, atmospheric CO2 dissolution,
    bicarbonate ion formation, carbonate ion formation, water autoionization.
    
    Parameters
    ----------
    x_CO2g : float
        Partial pressure of CO2 (ppm)
    Temp : float
        Temperature of the reactions        
    Pres : float
        Pressure of the reactions
    KeqFuncs : dict of {str : float}
        2D interpolated function in temperature, pressure with 4-letter keys
    
    Returns
    -------
    dic : float
        Total dissolved inorganic carbon
    pH : float
        Negative logarithm of proton activity    
    act_HCO3_m : float
        Activity of the bicarbonate ion
    act_CO2_aq : float
        Activity of aqueous CO2
    act_CO3_mm : float
        Activity of the carbonate ion
        
    '''


    K_ferr          =   KeqFuncs['ferr'](Temp,Pres)
    K_bica          =   KeqFuncs['bica'](Temp,Pres)
    K_carb          =   KeqFuncs['carb'](Temp,Pres)
    K_wate          =   KeqFuncs['wate'](Temp,Pres)
    K_co2a          =   KeqFuncs['co2a'](Temp,Pres)
    act_CO2_aq      =   x_CO2g * K_co2a
    a_vals               =   2 * K_carb
    b_vals               =   K_wate + K_bica * x_CO2g
    c_vals               =   - x_CO2g**2 * (K_bica**2 + 2*K_bica * K_ferr**(1/2))
    d_vals               =   0
    def f(x, a, b, c, d):
        return a * x**3 + b * x**2  +  c
    def f1(x, a, b, c, d):
        return 3 * a * x**2 + 2 * b * x
    def f2(x, a, b, c, d):
        return 6 * a * x + 2 * b
    x_init  = x_CO2g**(1/2)
    x0 = newton(f, x_init, fprime=f1, fprime2=f2, args=(a_vals, b_vals, c_vals, d_vals))
    act_HCO3_m      = x0
    act_CO3_mm      = K_carb / K_bica * act_HCO3_m**2 / x_CO2g
    alk             = 2 * act_CO3_mm + act_HCO3_m
    dic             = act_HCO3_m + act_CO2_aq + act_CO3_mm
    act_H_p         = K_bica * x_CO2g  / act_HCO3_m
    pH              = - np.log10(act_H_p)
    act_SiO2        = 0.5 * act_HCO3_m
    act_DIV_pp      = 0.5 * act_HCO3_m
    act_MON_p       = 0
    allSpecies = {'HCO3':act_HCO3_m, 'CO3':act_CO3_mm, 'ALK':alk, 'DIC':dic, 'pH':pH, 'H':act_H_p,
                  'SiO2':act_SiO2, 'DIV':act_DIV_pp, 'MON':act_MON_p, 'CO2':act_CO2_aq}
    return allSpecies

# %%
def fors_eq(x_CO2g, Temp, Pres, KeqFuncs):
    '''Calculate DIC, pH and DIC components for forsterite dissolution.
    
    Solves for the bicarbonate ion activity as a function of temperature, 
    partial pressure of CO2 and pressure. Simultaneous solution of activities
    of all species in the system of reactions.
    Polynomial equation of the form: a*x**(10/3) + b*x**(7/3) + c*x**(1/3) + d
    Reactions included: forsterite dissolution, atmospheric CO2 dissolution, 
    bicarbonate ion formation, carbonate ion formation, water autoionization.
    
    Parameters
    ----------
    x_CO2g : float
        Partial pressure of CO2 (ppm)
    Temp : float
        Temperature of the reactions        
    Pres : float
        Pressure of the reactions
    KeqFuncs : dict of {str : float}
        2D interpolated function in temperature, pressure with 4-letter keys
    
    Returns
    -------
    dic : float
        Total dissolved inorganic carbon
    pH : float
        Negative logarithm of proton activity    
    act_HCO3_m : float
        Activity of the bicarbonate ion
    act_CO2_aq : float
        Activity of aqueous CO2
    act_CO3_mm : float
        Activity of the carbonate ion
        
    '''


    K_fors          =   KeqFuncs['fors'](Temp,Pres)
    K_bica          =   KeqFuncs['bica'](Temp,Pres)
    K_carb          =   KeqFuncs['carb'](Temp,Pres)
    K_wate          =   KeqFuncs['wate'](Temp,Pres)
    K_co2a          =   KeqFuncs['co2a'](Temp,Pres)
    act_CO2_aq      =   x_CO2g * K_co2a
    a_vals               =   2 * K_carb
    b_vals               =   K_wate + K_bica * x_CO2g
    c_vals               =   - x_CO2g**2 * K_bica**2
    d_vals               =   - 2**(4/3) * K_bica * K_fors**(1/3) * x_CO2g**(7/3)
    def f(x, a, b, c, d):
        return a*x**(10/3) + b*x**(7/3) + c*x**(1/3) + d
    def f1(x, a, b, c, d):
        return (10/3)*a*x**(7/3) + (7/3)*b*x**(4/3) + (1/3)*c*x**(-2/3)
    def f2(x, a, b, c, d):
        return (70/9)*a*x**(4/3) + (28/9)*b*x**(1/3) - (2/9)*c*x**(-5/3)
    x_init  = x_CO2g**(1/2)
    x0 = newton(f, x_init, fprime=f1, fprime2=f2, args=(a_vals, b_vals, c_vals, d_vals))
    act_HCO3_m      = x0
    act_CO3_mm      = K_carb / K_bica * act_HCO3_m**2 / x_CO2g
    alk             = 2 * act_CO3_mm + act_HCO3_m
    dic             = act_HCO3_m + act_CO2_aq + act_CO3_mm
    act_H_p         = K_bica * x_CO2g  / act_HCO3_m
    pH              = - np.log10(act_H_p)
    act_SiO2        = 0.25 * act_HCO3_m
    act_DIV_pp      = 0.5 * act_HCO3_m
    act_MON_p       = 0
    allSpecies = {'HCO3':act_HCO3_m, 'CO3':act_CO3_mm, 'ALK':alk, 'DIC':dic, 'pH':pH, 'H':act_H_p,
                  'SiO2':act_SiO2, 'DIV':act_DIV_pp, 'MON':act_MON_p, 'CO2':act_CO2_aq}
    return allSpecies

# %%
def faya_eq(x_CO2g, Temp, Pres, KeqFuncs):
    '''Calculate DIC, pH and DIC components for fayalite dissolution.
    
    Solves for the bicarbonate ion activity as a function of temperature, 
    partial pressure of CO2 and pressure. Simultaneous solution of activities
    of all species in the system of reactions.
    Polynomial equation of the form: a*x**(10/3) + b*x**(7/3) + c*x**(1/3) + d
    Reactions included: fayalite dissolution, atmospheric CO2 dissolution, 
    bicarbonate ion formation, carbonate ion formation, water autoionization.
    
    Parameters
    ----------
    x_CO2g : float
        Partial pressure of CO2 (ppm)
    Temp : float
        Temperature of the reactions        
    Pres : float
        Pressure of the reactions
    KeqFuncs : dict of {str : float}
        2D interpolated function in temperature, pressure with 4-letter keys
    
    Returns
    -------
    dic : float
        Total dissolved inorganic carbon
    pH : float
        Negative logarithm of proton activity    
    act_HCO3_m : float
        Activity of the bicarbonate ion
    act_CO2_aq : float
        Activity of aqueous CO2
    act_CO3_mm : float
        Activity of the carbonate ion
        
    '''


    K_faya          =   KeqFuncs['faya'](Temp,Pres)
    K_bica          =   KeqFuncs['bica'](Temp,Pres)
    K_carb          =   KeqFuncs['carb'](Temp,Pres)
    K_wate          =   KeqFuncs['wate'](Temp,Pres)
    K_co2a          =   KeqFuncs['co2a'](Temp,Pres)
    act_CO2_aq      =   x_CO2g * K_co2a
    a_vals               =   2 * K_carb
    b_vals               =   K_wate + K_bica * x_CO2g
    c_vals               =   - x_CO2g**2 * K_bica**2
    d_vals               =   - 2**(4/3) * K_bica * K_faya**(1/3) * x_CO2g**(7/3)
    def f(x, a, b, c, d):
        return a*x**(10/3) + b*x**(7/3) + c*x**(1/3) + d
    def f1(x, a, b, c, d):
        return (10/3)*a*x**(7/3) + (7/3)*b*x**(4/3) + (1/3)*c*x**(-2/3)
    def f2(x, a, b, c, d):
        return (70/9)*a*x**(4/3) + (28/9)*b*x**(1/3) - (2/9)*c*x**(-5/3)
    x_init  = x_CO2g**(1/2)
    x0 = newton(f, x_init, fprime=f1, fprime2=f2, args=(a_vals, b_vals, c_vals, d_vals))
    act_HCO3_m      = x0
    act_CO3_mm      = K_carb / K_bica * act_HCO3_m**2 / x_CO2g
    alk             = 2 * act_CO3_mm + act_HCO3_m
    dic             = act_HCO3_m + act_CO2_aq + act_CO3_mm
    act_H_p         = K_bica * x_CO2g  / act_HCO3_m
    pH              = - np.log10(act_H_p)
    act_SiO2        = 0.25 * act_HCO3_m
    act_DIV_pp      = 0.5 * act_HCO3_m
    act_MON_p       = 0
    allSpecies = {'HCO3':act_HCO3_m, 'CO3':act_CO3_mm, 'ALK':alk, 'DIC':dic, 'pH':pH, 'H':act_H_p,
                  'SiO2':act_SiO2, 'DIV':act_DIV_pp, 'MON':act_MON_p, 'CO2':act_CO2_aq}
    return allSpecies

# %%
def anor_eq(x_CO2g, Temp, Pres, KeqFuncs):
    '''Calculate DIC, pH and DIC components for anorthite dissolution.
    
    Solves for the bicarbonate ion activity as a function of temperature, 
    partial pressure of CO2 and pressure. Simultaneous solution of activities
    of all species in the system of reactions.
    Polynomial equation of the form: a * x**4 + b * x**3 + c * x + d
    Reactions included: anorthite dissolution, atmospheric CO2 dissolution,
    bicarbonate ion formation, carbonate ion formation, water autoionization.
    
    Parameters
    ----------
    x_CO2g : float
        Partial pressure of CO2 (ppm)
    Temp : float
        Temperature of the reactions        
    Pres : float
        Pressure of the reactions
    KeqFuncs : dict of {str : float}
        2D interpolated function in temperature, pressure with 4-letter keys
    
    Returns
    -------
    dic : float
        Total dissolved inorganic carbon
    pH : float
        Negative logarithm of proton activity    
    act_HCO3_m : float
        Activity of the bicarbonate ion
    act_CO2_aq : float
        Activity of aqueous CO2
    act_CO3_mm : float
        Activity of the carbonate ion
        
    '''


    K_anor          =   KeqFuncs['anor'](Temp,Pres)
    K_bica          =   KeqFuncs['bica'](Temp,Pres)
    K_carb          =   KeqFuncs['carb'](Temp,Pres)
    K_wate          =   KeqFuncs['wate'](Temp,Pres)
    K_co2a          =   KeqFuncs['co2a'](Temp,Pres)
    act_CO2_aq      =   x_CO2g * K_co2a
    a_vals               =   2 * K_carb
    b_vals               =   K_wate + K_bica * x_CO2g
    c_vals               =   - K_bica**2 * x_CO2g**2
    d_vals               =   - 2 * K_anor**2 * K_bica * x_CO2g**3
    def f(x, a, b, c, d):
        return a * x**4 + b * x**3 + c * x + d
    def f1(x, a, b, c, d):
        return 4 * a * x**3 + 3 * b * x**2 + c 
    def f2(x, a, b, c, d):
        return 12 * a * x**2 + 6 * b * x 
    x_init  = x_CO2g**(1/2)
    x0 = newton(f, x_init, fprime=f1, fprime2=f2, args=(a_vals, b_vals, c_vals, d_vals))
    act_HCO3_m      = x0
    act_CO3_mm      = K_carb / K_bica * act_HCO3_m**2 / x_CO2g
    alk             = 2 * act_CO3_mm + act_HCO3_m
    dic             = act_HCO3_m + act_CO2_aq + act_CO3_mm
    act_H_p         = K_bica * x_CO2g  / act_HCO3_m
    pH              = - np.log10(act_H_p)
    act_SiO2        = 0
    act_DIV_pp      = 0.5 * act_HCO3_m
    act_MON_p       = 0
    allSpecies = {'HCO3':act_HCO3_m, 'CO3':act_CO3_mm, 'ALK':alk, 'DIC':dic, 'pH':pH, 'H':act_H_p,
                  'SiO2':act_SiO2, 'DIV':act_DIV_pp, 'MON':act_MON_p, 'CO2':act_CO2_aq}
    return allSpecies

# %%
def anoh_eq(x_CO2g, Temp, Pres, KeqFuncs):
    '''Calculate DIC, pH and DIC components for anorthite dissolution.
    
    Solves for the bicarbonate ion activity as a function of temperature, 
    partial pressure of CO2 and pressure. Simultaneous solution of activities
    of all species in the system of reactions.
    Polynomial equation of the form: a * x**4 + b * x**3 + c * x + d
    Reactions included: anorthite dissolution, atmospheric CO2 dissolution,
    bicarbonate ion formation, carbonate ion formation, water autoionization.
    
    Parameters
    ----------
    x_CO2g : float
        Partial pressure of CO2 (ppm)
    Temp : float
        Temperature of the reactions        
    Pres : float
        Pressure of the reactions
    KeqFuncs : dict of {str : float}
        2D interpolated function in temperature, pressure with 4-letter keys
    
    Returns
    -------
    dic : float
        Total dissolved inorganic carbon
    pH : float
        Negative logarithm of proton activity    
    act_HCO3_m : float
        Activity of the bicarbonate ion
    act_CO2_aq : float
        Activity of aqueous CO2
    act_CO3_mm : float
        Activity of the carbonate ion
        
    '''


    K_anor          =   KeqFuncs['anoh'](Temp,Pres)
    K_bica          =   KeqFuncs['bica'](Temp,Pres)
    K_carb          =   KeqFuncs['carb'](Temp,Pres)
    K_wate          =   KeqFuncs['wate'](Temp,Pres)
    K_co2a          =   KeqFuncs['co2a'](Temp,Pres)
    act_CO2_aq      =   x_CO2g * K_co2a
    a_vals               =   2 * K_carb
    b_vals               =   K_wate + K_bica * x_CO2g
    c_vals               =   - K_bica**2 * x_CO2g**2
    d_vals               =   - 2 * K_anor**2 * K_bica * x_CO2g**3
    def f(x, a, b, c, d):
        return a * x**4 + b * x**3 + c * x + d
    def f1(x, a, b, c, d):
        return 4 * a * x**3 + 3 * b * x**2 + c 
    def f2(x, a, b, c, d):
        return 12 * a * x**2 + 6 * b * x 
    x_init  = x_CO2g**(1/2)
    x0 = newton(f, x_init, fprime=f1, fprime2=f2, args=(a_vals, b_vals, c_vals, d_vals))
    act_HCO3_m      = x0
    act_CO3_mm      = K_carb / K_bica * act_HCO3_m**2 / x_CO2g
    alk             = 2 * act_CO3_mm + act_HCO3_m
    dic             = act_HCO3_m + act_CO2_aq + act_CO3_mm
    act_H_p         = K_bica * x_CO2g  / act_HCO3_m
    pH              = - np.log10(act_H_p)
    act_SiO2        = 0
    act_DIV_pp      = 0.5 * act_HCO3_m
    act_MON_p       = 0
    allSpecies = {'HCO3':act_HCO3_m, 'CO3':act_CO3_mm, 'ALK':alk, 'DIC':dic, 'pH':pH, 'H':act_H_p,
                  'SiO2':act_SiO2, 'DIV':act_DIV_pp, 'MON':act_MON_p, 'CO2':act_CO2_aq}
    return allSpecies


# %%
def kfel_eq(x_CO2g, Temp, Pres, KeqFuncs):
    '''Calculate DIC, pH and DIC components for K-feldspar dissolution.
    
    Solves for the bicarbonate ion activity as a function of temperature, 
    partial pressure of CO2 and pressure. Simultaneous solution of activities
    of all species in the system of reactions.
    Polynomial equation of the form: a * x**3 + b * x**2 + c * x**(2/3) + d
    Reactions included: K-feldspar dissolution, atmospheric CO2 dissolution,
    bicarbonate ion formation, carbonate ion formation, water autoionization.
    
    Parameters
    ----------
    x_CO2g : float
        Partial pressure of CO2 (ppm)
    Temp : float
        Temperature of the reactions        
    Pres : float
        Pressure of the reactions
    KeqFuncs : dict of {str : float}
        2D interpolated function in temperature, pressure with 4-letter keys
    
    Returns
    -------
    dic : float
        Total dissolved inorganic carbon
    pH : float
        Negative logarithm of proton activity    
    act_HCO3_m : float
        Activity of the bicarbonate ion
    act_CO2_aq : float
        Activity of aqueous CO2
    act_CO3_mm : float
        Activity of the carbonate ion
        
    '''


    K_kfel          =   KeqFuncs['kfel'](Temp,Pres)
    K_bica          =   KeqFuncs['bica'](Temp,Pres)
    K_carb          =   KeqFuncs['carb'](Temp,Pres)
    K_wate          =   KeqFuncs['wate'](Temp,Pres)
    K_co2a          =   KeqFuncs['co2a'](Temp,Pres)
    act_CO2_aq      =   x_CO2g * K_co2a
    a_vals               =   2 * K_carb
    b_vals               =   K_wate + K_bica * x_CO2g
    c_vals               =   - K_kfel**(1/3) / 4**(1/3) * K_bica * x_CO2g**(4/3)
    d_vals               =   - K_bica**2 * x_CO2g**2
    def f(x, a, b, c, d):
        return a * x**3 + b * x**2  +  c * x**(2/3) + d
    def f1(x, a, b, c, d):
        return 3 * a * x**2 + 2 * b * x  +  (2/3) * c * x**(-1/3)
    def f2(x, a, b, c, d):
        return 6 * a * x + 2 * b - (2/9) * c * x**(-4/3)
    x_init  = x_CO2g**(1/2)
    x0 = newton(f, x_init, fprime=f1, fprime2=f2, args=(a_vals, b_vals, c_vals, d_vals))
    act_HCO3_m      = x0
    act_CO3_mm      = K_carb / K_bica * act_HCO3_m**2 / x_CO2g
    alk             = 2 * act_CO3_mm + act_HCO3_m
    dic             = act_HCO3_m + act_CO2_aq + act_CO3_mm
    act_H_p         = K_bica * x_CO2g  / act_HCO3_m
    pH              = - np.log10(act_H_p)
    act_SiO2        = 2 * act_HCO3_m
    act_DIV_pp      = 0
    act_MON_p       = act_HCO3_m
    allSpecies = {'HCO3':act_HCO3_m, 'CO3':act_CO3_mm, 'ALK':alk, 'DIC':dic, 'pH':pH, 'H':act_H_p,
                  'SiO2':act_SiO2, 'DIV':act_DIV_pp, 'MON':act_MON_p, 'CO2':act_CO2_aq}
    return allSpecies

# %%
def kfeh_eq(x_CO2g, Temp, Pres, KeqFuncs):
    '''Calculate DIC, pH and DIC components for K-feldspar dissolution.
    
    Solves for the bicarbonate ion activity as a function of temperature, 
    partial pressure of CO2 and pressure. Simultaneous solution of activities
    of all species in the system of reactions.
    Polynomial equation of the form: a * x**3 + b * x**2 + c * x**(2/3) + d
    Reactions included: K-feldspar dissolution, atmospheric CO2 dissolution,
    bicarbonate ion formation, carbonate ion formation, water autoionization.
    
    Parameters
    ----------
    x_CO2g : float
        Partial pressure of CO2 (ppm)
    Temp : float
        Temperature of the reactions        
    Pres : float
        Pressure of the reactions
    KeqFuncs : dict of {str : float}
        2D interpolated function in temperature, pressure with 4-letter keys
    
    Returns
    -------
    dic : float
        Total dissolved inorganic carbon
    pH : float
        Negative logarithm of proton activity    
    act_HCO3_m : float
        Activity of the bicarbonate ion
    act_CO2_aq : float
        Activity of aqueous CO2
    act_CO3_mm : float
        Activity of the carbonate ion
        
    '''


    K_kfel          =   KeqFuncs['kfeh'](Temp,Pres)
    K_bica          =   KeqFuncs['bica'](Temp,Pres)
    K_carb          =   KeqFuncs['carb'](Temp,Pres)
    K_wate          =   KeqFuncs['wate'](Temp,Pres)
    K_co2a          =   KeqFuncs['co2a'](Temp,Pres)
    act_CO2_aq      =   x_CO2g * K_co2a
    a_vals               =   2 * K_carb
    b_vals               =   K_wate + K_bica * x_CO2g
    c_vals               =   - K_kfel**(1/3) / 4**(1/3) * K_bica * x_CO2g**(4/3)
    d_vals               =   - K_bica**2 * x_CO2g**2
    def f(x, a, b, c, d):
        return a * x**3 + b * x**2  +  c * x**(2/3) + d
    def f1(x, a, b, c, d):
        return 3 * a * x**2 + 2 * b * x  +  (2/3) * c * x**(-1/3)
    def f2(x, a, b, c, d):
        return 6 * a * x + 2 * b - (2/9) * c * x**(-4/3)
    x_init  = x_CO2g**(1/2)
    x0 = newton(f, x_init, fprime=f1, fprime2=f2, args=(a_vals, b_vals, c_vals, d_vals))
    act_HCO3_m      = x0
    act_CO3_mm      = K_carb / K_bica * act_HCO3_m**2 / x_CO2g
    alk             = 2 * act_CO3_mm + act_HCO3_m
    dic             = act_HCO3_m + act_CO2_aq + act_CO3_mm
    act_H_p         = K_bica * x_CO2g  / act_HCO3_m
    pH              = - np.log10(act_H_p)
    act_SiO2        = 2 * act_HCO3_m
    act_DIV_pp      = 0
    act_MON_p       = act_HCO3_m
    allSpecies = {'HCO3':act_HCO3_m, 'CO3':act_CO3_mm, 'ALK':alk, 'DIC':dic, 'pH':pH, 'H':act_H_p,
                  'SiO2':act_SiO2, 'DIV':act_DIV_pp, 'MON':act_MON_p, 'CO2':act_CO2_aq}
    return allSpecies


# %%
def albi_eq(x_CO2g, Temp, Pres, KeqFuncs):
    '''Calculate DIC, pH and DIC components for albite dissolution.
    
    Solves for the bicarbonate ion activity as a function of temperature, 
    partial pressure of CO2 and pressure. Simultaneous solution of activities
    of all species in the system of reactions.
    Polynomial equation of the form: a * x**3 + b * x**2 + c * x**(2/3) + d
    Reactions included: albite dissolution, atmospheric CO2 dissolution,
    bicarbonate ion formation, carbonate ion formation, water autoionization.
    
    Parameters
    ----------
    x_CO2g : float
        Partial pressure of CO2 (ppm)
    Temp : float
        Temperature of the reactions        
    Pres : float
        Pressure of the reactions
    KeqFuncs : dict of {str : float}
        2D interpolated function in temperature, pressure with 4-letter keys
    
    Returns
    -------
    dic : float
        Total dissolved inorganic carbon
    pH : float
        Negative logarithm of proton activity    
    act_HCO3_m : float
        Activity of the bicarbonate ion
    act_CO2_aq : float
        Activity of aqueous CO2
    act_CO3_mm : float
        Activity of the carbonate ion
        
    '''


    K_albi          =   KeqFuncs['albi'](Temp,Pres)
    K_bica          =   KeqFuncs['bica'](Temp,Pres)
    K_carb          =   KeqFuncs['carb'](Temp,Pres)
    K_wate          =   KeqFuncs['wate'](Temp,Pres)
    K_co2a          =   KeqFuncs['co2a'](Temp,Pres)
    act_CO2_aq      =   x_CO2g * K_co2a
    a_vals               =   2 * K_carb
    b_vals               =   K_wate + K_bica * x_CO2g
    c_vals               =   - K_albi**(1/3) / 4**(1/3) * K_bica * x_CO2g**(4/3)
    d_vals               =   - K_bica**2 * x_CO2g**2
    def f(x, a, b, c, d):
        return a * x**3 + b * x**2  +  c * x**(2/3) + d
    def f1(x, a, b, c, d):
        return 3 * a * x**2 + 2 * b * x  +  (2/3) * c * x**(-1/3)
    def f2(x, a, b, c, d):
        return 6 * a * x + 2 * b - (2/9) * c * x**(-4/3)
    x_init  = x_CO2g**(1/2)
    x0 = newton(f, x_init, fprime=f1, fprime2=f2, args=(a_vals, b_vals, c_vals, d_vals))
    act_HCO3_m      = x0
    act_CO3_mm      = K_carb / K_bica * act_HCO3_m**2 / x_CO2g
    alk             = 2 * act_CO3_mm + act_HCO3_m
    dic             = act_HCO3_m + act_CO2_aq + act_CO3_mm
    act_H_p         = K_bica * x_CO2g  / act_HCO3_m
    pH              = - np.log10(act_H_p)
    act_SiO2        = 2 * act_HCO3_m
    act_DIV_pp      = 0
    act_MON_p       = act_HCO3_m
    allSpecies = {'HCO3':act_HCO3_m, 'CO3':act_CO3_mm, 'ALK':alk, 'DIC':dic, 'pH':pH, 'H':act_H_p,
                  'SiO2':act_SiO2, 'DIV':act_DIV_pp, 'MON':act_MON_p, 'CO2':act_CO2_aq}
    return allSpecies

# %%
def albh_eq(x_CO2g, Temp, Pres, KeqFuncs):
    '''Calculate DIC, pH and DIC components for albite dissolution.
    
    Solves for the bicarbonate ion activity as a function of temperature, 
    partial pressure of CO2 and pressure. Simultaneous solution of activities
    of all species in the system of reactions.
    Polynomial equation of the form: a * x**3 + b * x**2 + c * x**(2/3) + d
    Reactions included: albite dissolution, atmospheric CO2 dissolution,
    bicarbonate ion formation, carbonate ion formation, water autoionization.
    
    Parameters
    ----------
    x_CO2g : float
        Partial pressure of CO2 (ppm)
    Temp : float
        Temperature of the reactions        
    Pres : float
        Pressure of the reactions
    KeqFuncs : dict of {str : float}
        2D interpolated function in temperature, pressure with 4-letter keys
    
    Returns
    -------
    dic : float
        Total dissolved inorganic carbon
    pH : float
        Negative logarithm of proton activity    
    act_HCO3_m : float
        Activity of the bicarbonate ion
    act_CO2_aq : float
        Activity of aqueous CO2
    act_CO3_mm : float
        Activity of the carbonate ion
        
    '''


    K_albi          =   KeqFuncs['albh'](Temp,Pres)
    K_bica          =   KeqFuncs['bica'](Temp,Pres)
    K_carb          =   KeqFuncs['carb'](Temp,Pres)
    K_wate          =   KeqFuncs['wate'](Temp,Pres)
    K_co2a          =   KeqFuncs['co2a'](Temp,Pres)
    act_CO2_aq      =   x_CO2g * K_co2a
    a_vals               =   2 * K_carb
    b_vals               =   K_wate + K_bica * x_CO2g
    c_vals               =   - K_albi**(1/3) / 4**(1/3) * K_bica * x_CO2g**(4/3)
    d_vals               =   - K_bica**2 * x_CO2g**2
    def f(x, a, b, c, d):
        return a * x**3 + b * x**2  +  c * x**(2/3) + d
    def f1(x, a, b, c, d):
        return 3 * a * x**2 + 2 * b * x  +  (2/3) * c * x**(-1/3)
    def f2(x, a, b, c, d):
        return 6 * a * x + 2 * b - (2/9) * c * x**(-4/3)
    x_init  = x_CO2g**(1/2)
    x0 = newton(f, x_init, fprime=f1, fprime2=f2, args=(a_vals, b_vals, c_vals, d_vals))
    act_HCO3_m      = x0
    act_CO3_mm      = K_carb / K_bica * act_HCO3_m**2 / x_CO2g
    alk             = 2 * act_CO3_mm + act_HCO3_m
    dic             = act_HCO3_m + act_CO2_aq + act_CO3_mm
    act_H_p         = K_bica * x_CO2g  / act_HCO3_m
    pH              = - np.log10(act_H_p)
    act_SiO2        = 2 * act_HCO3_m
    act_DIV_pp      = 0
    act_MON_p       = act_HCO3_m
    allSpecies = {'HCO3':act_HCO3_m, 'CO3':act_CO3_mm, 'ALK':alk, 'DIC':dic, 'pH':pH, 'H':act_H_p,
                  'SiO2':act_SiO2, 'DIV':act_DIV_pp, 'MON':act_MON_p, 'CO2':act_CO2_aq}
    return allSpecies


# %%
def musc_eq(x_CO2g, Temp, Pres, KeqFuncs):
    '''Calculate DIC, pH and DIC components for muscovite dissolution.
    
    Solves for the bicarbonate ion activity as a function of temperature, 
    partial pressure of CO2 and pressure. Simultaneous solution of activities
    of all species in the system of reactions.
    Polynomial equation of the form: a * x**3 + b * x**2 + c
    Reactions included: muscovite dissolution, atmospheric CO2 dissolution,
    bicarbonate ion formation, carbonate ion formation, water autoionization.
    
    Parameters
    ----------
    x_CO2g : float
        Partial pressure of CO2 (ppm)
    Temp : float
        Temperature of the reactions        
    Pres : float
        Pressure of the reactions
    KeqFuncs : dict of {str : float}
        2D interpolated function in temperature, pressure with 4-letter keys
    
    Returns
    -------
    dic : float
        Total dissolved inorganic carbon
    pH : float
        Negative logarithm of proton activity    
    act_HCO3_m : float
        Activity of the bicarbonate ion
    act_CO2_aq : float
        Activity of aqueous CO2
    act_CO3_mm : float
        Activity of the carbonate ion
        
    '''


    K_musc          =   KeqFuncs['musc'](Temp,Pres)
    K_bica          =   KeqFuncs['bica'](Temp,Pres)
    K_carb          =   KeqFuncs['carb'](Temp,Pres)
    K_wate          =   KeqFuncs['wate'](Temp,Pres)
    K_co2a          =   KeqFuncs['co2a'](Temp,Pres)
    act_CO2_aq      =   x_CO2g * K_co2a
    a_vals               =   2 * K_carb
    b_vals               =   K_wate + K_bica * x_CO2g
    c_vals               =   - (K_bica**2 + K_musc * K_bica) * x_CO2g**2
    d_vals               =   0
    def f(x, a, b, c, d):
        return a * x**3 + b * x**2  +  c 
    def f1(x, a, b, c, d):
        return 3 * a * x**2 + 2 * b * x
    def f2(x, a, b, c, d):
        return 6 * a * x + 2 * b
    x_init  = x_CO2g**(1/2)
    x0 = newton(f, x_init, fprime=f1, fprime2=f2, args=(a_vals, b_vals, c_vals, d_vals))
    act_HCO3_m      = x0
    act_CO3_mm      = K_carb / K_bica * act_HCO3_m**2 / x_CO2g
    alk             = 2 * act_CO3_mm + act_HCO3_m
    dic             = act_HCO3_m + act_CO2_aq + act_CO3_mm
    act_H_p         = K_bica * x_CO2g  / act_HCO3_m
    pH              = - np.log10(act_H_p)
    act_SiO2        = 0
    act_DIV_pp      = 0
    act_MON_p       = act_HCO3_m
    allSpecies = {'HCO3':act_HCO3_m, 'CO3':act_CO3_mm, 'ALK':alk, 'DIC':dic, 'pH':pH, 'H':act_H_p,
                  'SiO2':act_SiO2, 'DIV':act_DIV_pp, 'MON':act_MON_p, 'CO2':act_CO2_aq}
    return allSpecies

# %%
def mush_eq(x_CO2g, Temp, Pres, KeqFuncs):
    '''Calculate DIC, pH and DIC components for muscovite dissolution.
    
    Solves for the bicarbonate ion activity as a function of temperature, 
    partial pressure of CO2 and pressure. Simultaneous solution of activities
    of all species in the system of reactions.
    Polynomial equation of the form: a * x**3 + b * x**2 + c
    Reactions included: muscovite dissolution, atmospheric CO2 dissolution,
    bicarbonate ion formation, carbonate ion formation, water autoionization.
    
    Parameters
    ----------
    x_CO2g : float
        Partial pressure of CO2 (ppm)
    Temp : float
        Temperature of the reactions        
    Pres : float
        Pressure of the reactions
    KeqFuncs : dict of {str : float}
        2D interpolated function in temperature, pressure with 4-letter keys
    
    Returns
    -------
    dic : float
        Total dissolved inorganic carbon
    pH : float
        Negative logarithm of proton activity    
    act_HCO3_m : float
        Activity of the bicarbonate ion
    act_CO2_aq : float
        Activity of aqueous CO2
    act_CO3_mm : float
        Activity of the carbonate ion
        
    '''


    K_musc          =   KeqFuncs['mush'](Temp,Pres)
    K_bica          =   KeqFuncs['bica'](Temp,Pres)
    K_carb          =   KeqFuncs['carb'](Temp,Pres)
    K_wate          =   KeqFuncs['wate'](Temp,Pres)
    K_co2a          =   KeqFuncs['co2a'](Temp,Pres)
    act_CO2_aq      =   x_CO2g * K_co2a
    a_vals               =   2 * K_carb
    b_vals               =   K_wate + K_bica * x_CO2g
    c_vals               =   - (K_bica**2 + K_musc * K_bica) * x_CO2g**2
    d_vals               =   0
    def f(x, a, b, c, d):
        return a * x**3 + b * x**2  +  c 
    def f1(x, a, b, c, d):
        return 3 * a * x**2 + 2 * b * x
    def f2(x, a, b, c, d):
        return 6 * a * x + 2 * b
    x_init  = x_CO2g**(1/2)
    x0 = newton(f, x_init, fprime=f1, fprime2=f2, args=(a_vals, b_vals, c_vals, d_vals))
    act_HCO3_m      = x0
    act_CO3_mm      = K_carb / K_bica * act_HCO3_m**2 / x_CO2g
    alk             = 2 * act_CO3_mm + act_HCO3_m
    dic             = act_HCO3_m + act_CO2_aq + act_CO3_mm
    act_H_p         = K_bica * x_CO2g  / act_HCO3_m
    pH              = - np.log10(act_H_p)
    act_SiO2        = 0
    act_DIV_pp      = 0
    act_MON_p       = act_HCO3_m
    allSpecies = {'HCO3':act_HCO3_m, 'CO3':act_CO3_mm, 'ALK':alk, 'DIC':dic, 'pH':pH, 'H':act_H_p,
                  'SiO2':act_SiO2, 'DIV':act_DIV_pp, 'MON':act_MON_p, 'CO2':act_CO2_aq}
    return allSpecies


# %%
def phlo_eq(x_CO2g, Temp, Pres, KeqFuncs):
    '''Calculate DIC, pH and DIC components for phlogopite dissolution.
    
    Solves for the bicarbonate ion activity as a function of temperature, 
    partial pressure of CO2 and pressure. Simultaneous solution of activities
    of all species in the system of reactions.
    Polynomial equation of the form: a*x**(19/6)+b*x**(13/6)+c*x**(1/6)+d
    Reactions included: phlogopite dissolution, atmospheric CO2 dissolution,
    bicarbonate ion formation, carbonate ion formation, water autoionization.
    
    Parameters
    ----------
    x_CO2g : float
        Partial pressure of CO2 (ppm)
    Temp : float
        Temperature of the reactions        
    Pres : float
        Pressure of the reactions
    KeqFuncs : dict of {str : float}
        2D interpolated function in temperature, pressure with 4-letter keys
    
    Returns
    -------
    dic : float
        Total dissolved inorganic carbon
    pH : float
        Negative logarithm of proton activity    
    act_HCO3_m : float
        Activity of the bicarbonate ion
    act_CO2_aq : float
        Activity of aqueous CO2
    act_CO3_mm : float
        Activity of the carbonate ion
        
    '''


    K_phlo          =   KeqFuncs['phlo'](Temp,Pres)
    K_bica          =   KeqFuncs['bica'](Temp,Pres)
    K_carb          =   KeqFuncs['carb'](Temp,Pres)
    K_wate          =   KeqFuncs['wate'](Temp,Pres)
    K_co2a          =   KeqFuncs['co2a'](Temp,Pres)
    act_CO2_aq      =   x_CO2g * K_co2a
    a_vals               =   2 * K_carb
    b_vals               =   K_wate + K_bica * x_CO2g
    c_vals               =   - K_bica**2 * x_CO2g**2
    d_vals               =   - 2**(2/3) * 3**(1/2) * K_phlo**(1/2) * K_bica\
                        * x_CO2g**(13/6)
    def f(x, a, b, c, d):
        return a*x**(19/6) + b*x**(13/6) + c*x**(1/6) + d
    def f1(x, a, b, c, d):
        return (19/6)*a*x**(13/6) + (13/6)*b*x**(7/6) + (1/6)*c*x**(-5/6)
    def f2(x, a, b, c, d):
        return (247/36)*a*x**(7/6) + (91/36)*b*x**(1/6) - (5/36)*c*x**(-11/6)
    x_init  = x_CO2g**(1/2)
    x0 = newton(f, x_init, fprime=f1, fprime2=f2, args=(a_vals, b_vals, c_vals, d_vals))
    act_HCO3_m      = x0
    act_CO3_mm      = K_carb / K_bica * act_HCO3_m**2 / x_CO2g
    alk             = 2 * act_CO3_mm + act_HCO3_m
    dic             = act_HCO3_m + act_CO2_aq + act_CO3_mm
    act_H_p         = K_bica * x_CO2g  / act_HCO3_m
    pH              = - np.log10(act_H_p)
    act_SiO2        = 2/7 * act_HCO3_m
    act_DIV_pp      = 3/7 * act_HCO3_m
    act_MON_p       = 1/7 * act_HCO3_m
    allSpecies = {'HCO3':act_HCO3_m, 'CO3':act_CO3_mm, 'ALK':alk, 'DIC':dic, 'pH':pH, 'H':act_H_p,
                  'SiO2':act_SiO2, 'DIV':act_DIV_pp, 'MON':act_MON_p, 'CO2':act_CO2_aq}
    return allSpecies

# %%
def phlh_eq(x_CO2g, Temp, Pres, KeqFuncs):
    '''Calculate DIC, pH and DIC components for phlogopite dissolution.
    
    Solves for the bicarbonate ion activity as a function of temperature, 
    partial pressure of CO2 and pressure. Simultaneous solution of activities
    of all species in the system of reactions.
    Polynomial equation of the form: a*x**(19/6)+b*x**(13/6)+c*x**(1/6)+d
    Reactions included: phlogopite dissolution, atmospheric CO2 dissolution,
    bicarbonate ion formation, carbonate ion formation, water autoionization.
    
    Parameters
    ----------
    x_CO2g : float
        Partial pressure of CO2 (ppm)
    Temp : float
        Temperature of the reactions        
    Pres : float
        Pressure of the reactions
    KeqFuncs : dict of {str : float}
        2D interpolated function in temperature, pressure with 4-letter keys
    
    Returns
    -------
    dic : float
        Total dissolved inorganic carbon
    pH : float
        Negative logarithm of proton activity    
    act_HCO3_m : float
        Activity of the bicarbonate ion
    act_CO2_aq : float
        Activity of aqueous CO2
    act_CO3_mm : float
        Activity of the carbonate ion
        
    '''


    K_phlo          =   KeqFuncs['phlh'](Temp,Pres)
    K_bica          =   KeqFuncs['bica'](Temp,Pres)
    K_carb          =   KeqFuncs['carb'](Temp,Pres)
    K_wate          =   KeqFuncs['wate'](Temp,Pres)
    K_co2a          =   KeqFuncs['co2a'](Temp,Pres)
    act_CO2_aq      =   x_CO2g * K_co2a
    a_vals               =   2 * K_carb
    b_vals               =   K_wate + K_bica * x_CO2g
    c_vals               =   - K_bica**2 * x_CO2g**2
    d_vals               =   - 2**(2/3) * 3**(1/2) * K_phlo**(1/2) * K_bica\
                        * x_CO2g**(13/6)
    def f(x, a, b, c, d):
        return a*x**(19/6) + b*x**(13/6) + c*x**(1/6) + d
    def f1(x, a, b, c, d):
        return (19/6)*a*x**(13/6) + (13/6)*b*x**(7/6) + (1/6)*c*x**(-5/6)
    def f2(x, a, b, c, d):
        return (247/36)*a*x**(7/6) + (91/36)*b*x**(1/6) - (5/36)*c*x**(-11/6)
    x_init  = x_CO2g**(1/2)
    x0 = newton(f, x_init, fprime=f1, fprime2=f2, args=(a_vals, b_vals, c_vals, d_vals))
    act_HCO3_m      = x0
    act_CO3_mm      = K_carb / K_bica * act_HCO3_m**2 / x_CO2g
    alk             = 2 * act_CO3_mm + act_HCO3_m
    dic             = act_HCO3_m + act_CO2_aq + act_CO3_mm
    act_H_p         = K_bica * x_CO2g  / act_HCO3_m
    pH              = - np.log10(act_H_p)
    act_SiO2        = 2/7 * act_HCO3_m
    act_DIV_pp      = 3/7 * act_HCO3_m
    act_MON_p       = 1/7 * act_HCO3_m
    allSpecies = {'HCO3':act_HCO3_m, 'CO3':act_CO3_mm, 'ALK':alk, 'DIC':dic, 'pH':pH, 'H':act_H_p,
                  'SiO2':act_SiO2, 'DIV':act_DIV_pp, 'MON':act_MON_p, 'CO2':act_CO2_aq}
    return allSpecies


# %%
def anni_eq(x_CO2g, Temp, Pres, KeqFuncs):
    '''Calculate DIC, pH and DIC components for annite dissolution.
    
    Solves for the bicarbonate ion activity as a function of temperature, 
    partial pressure of CO2 and pressure. Simultaneous solution of activities
    of all species in the system of reactions.
    Polynomial equation of the form: a*x**(19/6)+b*x**(13/6)+c*x**(1/6)+d
    Reactions included: annite dissolution, atmospheric CO2 dissolution,
    bicarbonate ion formation, carbonate ion formation, water autoionization.
    
    Parameters
    ----------
    x_CO2g : float
        Partial pressure of CO2 (ppm)
    Temp : float
        Temperature of the reactions        
    Pres : float
        Pressure of the reactions
    KeqFuncs : dict of {str : float}
        2D interpolated function in temperature, pressure with 4-letter keys
    
    Returns
    -------
    dic : float
        Total dissolved inorganic carbon
    pH : float
        Negative logarithm of proton activity    
    act_HCO3_m : float
        Activity of the bicarbonate ion
    act_CO2_aq : float
        Activity of aqueous CO2
    act_CO3_mm : float
        Activity of the carbonate ion
        
    '''


    K_anni          =   KeqFuncs['anni'](Temp,Pres)
    K_bica          =   KeqFuncs['bica'](Temp,Pres)
    K_carb          =   KeqFuncs['carb'](Temp,Pres)
    K_wate          =   KeqFuncs['wate'](Temp,Pres)
    K_co2a          =   KeqFuncs['co2a'](Temp,Pres)
    act_CO2_aq      =   x_CO2g * K_co2a
    a_vals               =   2 * K_carb
    b_vals               =   K_wate + K_bica * x_CO2g
    c_vals               =   - K_bica**2 * x_CO2g**2
    d_vals               =   - 2**(2/3) * 3**(1/2) * K_anni**(1/2) * K_bica\
                        * x_CO2g**(13/6)
    def f(x, a, b, c, d):
        return a*x**(19/6) + b*x**(13/6) + c*x**(1/6) + d
    def f1(x, a, b, c, d):
        return (19/6)*a*x**(13/6) + (13/6)*b*x**(7/6) + (1/6)*c*x**(-5/6)
    def f2(x, a, b, c, d):
        return (247/36)*a*x**(7/6) + (91/36)*b*x**(1/6) - (5/36)*c*x**(-11/6)
    x_init  = x_CO2g**(1/2)
    x0 = newton(f, x_init, fprime=f1, fprime2=f2, args=(a_vals, b_vals, c_vals, d_vals))
    act_HCO3_m      = x0
    act_CO3_mm      = K_carb / K_bica * act_HCO3_m**2 / x_CO2g
    alk             = 2 * act_CO3_mm + act_HCO3_m
    dic             = act_HCO3_m + act_CO2_aq + act_CO3_mm
    act_H_p         = K_bica * x_CO2g  / act_HCO3_m
    pH              = - np.log10(act_H_p)
    act_SiO2        = 2/7 * act_HCO3_m
    act_DIV_pp      = 3/7 * act_HCO3_m
    act_MON_p       = 1/7 * act_HCO3_m
    allSpecies = {'HCO3':act_HCO3_m, 'CO3':act_CO3_mm, 'ALK':alk, 'DIC':dic, 'pH':pH, 'H':act_H_p,
                  'SiO2':act_SiO2, 'DIV':act_DIV_pp, 'MON':act_MON_p, 'CO2':act_CO2_aq}
    return allSpecies

# %%
def annh_eq(x_CO2g, Temp, Pres, KeqFuncs):
    '''Calculate DIC, pH and DIC components for annite dissolution.
    
    Solves for the bicarbonate ion activity as a function of temperature, 
    partial pressure of CO2 and pressure. Simultaneous solution of activities
    of all species in the system of reactions.
    Polynomial equation of the form: a*x**(19/6)+b*x**(13/6)+c*x**(1/6)+d
    Reactions included: annite dissolution, atmospheric CO2 dissolution,
    bicarbonate ion formation, carbonate ion formation, water autoionization.
    
    Parameters
    ----------
    x_CO2g : float
        Partial pressure of CO2 (ppm)
    Temp : float
        Temperature of the reactions        
    Pres : float
        Pressure of the reactions
    KeqFuncs : dict of {str : float}
        2D interpolated function in temperature, pressure with 4-letter keys
    
    Returns
    -------
    dic : float
        Total dissolved inorganic carbon
    pH : float
        Negative logarithm of proton activity    
    act_HCO3_m : float
        Activity of the bicarbonate ion
    act_CO2_aq : float
        Activity of aqueous CO2
    act_CO3_mm : float
        Activity of the carbonate ion
        
    '''


    K_anni          =   KeqFuncs['annh'](Temp,Pres)
    K_bica          =   KeqFuncs['bica'](Temp,Pres)
    K_carb          =   KeqFuncs['carb'](Temp,Pres)
    K_wate          =   KeqFuncs['wate'](Temp,Pres)
    K_co2a          =   KeqFuncs['co2a'](Temp,Pres)
    act_CO2_aq      =   x_CO2g * K_co2a
    a_vals               =   2 * K_carb
    b_vals               =   K_wate + K_bica * x_CO2g
    c_vals               =   - K_bica**2 * x_CO2g**2
    d_vals               =   - 2**(2/3) * 3**(1/2) * K_anni**(1/2) * K_bica\
                        * x_CO2g**(13/6)
    def f(x, a, b, c, d):
        return a*x**(19/6) + b*x**(13/6) + c*x**(1/6) + d
    def f1(x, a, b, c, d):
        return (19/6)*a*x**(13/6) + (13/6)*b*x**(7/6) + (1/6)*c*x**(-5/6)
    def f2(x, a, b, c, d):
        return (247/36)*a*x**(7/6) + (91/36)*b*x**(1/6) - (5/36)*c*x**(-11/6)
    x_init  = x_CO2g**(1/2)
    x0 = newton(f, x_init, fprime=f1, fprime2=f2, args=(a_vals, b_vals, c_vals, d_vals))
    act_HCO3_m      = x0
    act_CO3_mm      = K_carb / K_bica * act_HCO3_m**2 / x_CO2g
    alk             = 2 * act_CO3_mm + act_HCO3_m
    dic             = act_HCO3_m + act_CO2_aq + act_CO3_mm
    act_H_p         = K_bica * x_CO2g  / act_HCO3_m
    pH              = - np.log10(act_H_p)
    act_SiO2        = 2/7 * act_HCO3_m
    act_DIV_pp      = 3/7 * act_HCO3_m
    act_MON_p       = 1/7 * act_HCO3_m
    allSpecies = {'HCO3':act_HCO3_m, 'CO3':act_CO3_mm, 'ALK':alk, 'DIC':dic, 'pH':pH, 'H':act_H_p,
                  'SiO2':act_SiO2, 'DIV':act_DIV_pp, 'MON':act_MON_p, 'CO2':act_CO2_aq}
    return allSpecies


# %%
def anth_eq(x_CO2g, Temp, Pres, KeqFuncs):
    '''Calculate DIC, pH and DIC components for anthophyllite dissolution.
    
    Solves for the bicarbonate ion activity as a function of temperature, 
    partial pressure of CO2 and pressure. Simultaneous solution of activities
    of all species in the system of reactions.
    Polynomial equation of the form: a * x**3 + b * x**2 + c x**(1/15) + d
    Reactions included: anthophyllite dissolution, atmospheric CO2 dissolution,
    bicarbonate ion formation, carbonate ion formation, water autoionization.
    
    Parameters
    ----------
    x_CO2g : float
        Partial pressure of CO2 (ppm)
    Temp : float
        Temperature of the reactions        
    Pres : float
        Pressure of the reactions
    KeqFuncs : dict of {str : float}
        2D interpolated function in temperature, pressure with 4-letter keys
    
    Returns
    -------
    dic : float
        Total dissolved inorganic carbon
    pH : float
        Negative logarithm of proton activity    
    act_HCO3_m : float
        Activity of the bicarbonate ion
    act_CO2_aq : float
        Activity of aqueous CO2
    act_CO3_mm : float
        Activity of the carbonate ion
        
    '''


    K_anth          =   KeqFuncs['anth'](Temp,Pres)
    K_bica          =   KeqFuncs['bica'](Temp,Pres)
    K_carb          =   KeqFuncs['carb'](Temp,Pres)
    K_wate          =   KeqFuncs['wate'](Temp,Pres)
    K_co2a          =   KeqFuncs['co2a'](Temp,Pres)
    act_CO2_aq      =   x_CO2g * K_co2a
    a_vals               =   2 * K_carb
    b_vals               =   K_wate + K_bica * x_CO2g
    c_vals               =   - 2*(7/8)**(8/15)*K_anth**7*K_bica*x_CO2g**(29/15)
    d_vals               =   - K_bica**2 * x_CO2g**2
    def f(x, a, b, c, d):
        return a * x**3 + b * x**2 + c * x**(1/15) + d
    def f1(x, a, b, c, d):
        return 3 * a * x**2 + 2 * b * x + (1/15) * c * x**(-14/15)
    def f2(x, a, b, c, d):
        return 6 * a * x + 2 * b - (14/225) * c * x**(-29/15) 
    x_init  = x_CO2g**(1/2)
    x0 = newton(f, x_init, fprime=f1, fprime2=f2, args=(a_vals, b_vals, c_vals, d_vals))
    act_HCO3_m      = x0
    act_CO3_mm      = K_carb / K_bica * act_HCO3_m**2 / x_CO2g
    alk             = 2 * act_CO3_mm + act_HCO3_m
    dic             = act_HCO3_m + act_CO2_aq + act_CO3_mm
    act_H_p         = K_bica * x_CO2g  / act_HCO3_m
    pH              = - np.log10(act_H_p)
    act_SiO2        = 4/7 * act_HCO3_m
    act_DIV_pp      = 0.5 * act_HCO3_m
    act_MON_p       = 0
    allSpecies = {'HCO3':act_HCO3_m, 'CO3':act_CO3_mm, 'ALK':alk, 'DIC':dic, 'pH':pH, 'H':act_H_p,
                  'SiO2':act_SiO2, 'DIV':act_DIV_pp, 'MON':act_MON_p, 'CO2':act_CO2_aq}
    return allSpecies

# %%
def grun_eq(x_CO2g, Temp, Pres, KeqFuncs):
    '''Calculate DIC, pH and DIC components for grunerite dissolution.
    
    Solves for the bicarbonate ion activity as a function of temperature, 
    partial pressure of CO2 and pressure. Simultaneous solution of activities
    of all species in the system of reactions.
    Polynomial equation of the form: a * x**3 + b * x**2 + c x**(1/15) + d
    Reactions included: grunerite dissolution, atmospheric CO2 dissolution,
    bicarbonate ion formation, carbonate ion formation, water autoionization.
    
    Parameters
    ----------
    x_CO2g : float
        Partial pressure of CO2 (ppm)
    Temp : float
        Temperature of the reactions        
    Pres : float
        Pressure of the reactions
    KeqFuncs : dict of {str : float}
        2D interpolated function in temperature, pressure with 4-letter keys
    
    Returns
    -------
    dic : float
        Total dissolved inorganic carbon
    pH : float
        Negative logarithm of proton activity    
    act_HCO3_m : float
        Activity of the bicarbonate ion
    act_CO2_aq : float
        Activity of aqueous CO2
    act_CO3_mm : float
        Activity of the carbonate ion
        
    '''


    K_grun          =   KeqFuncs['grun'](Temp,Pres)
    K_bica          =   KeqFuncs['bica'](Temp,Pres)
    K_carb          =   KeqFuncs['carb'](Temp,Pres)
    K_wate          =   KeqFuncs['wate'](Temp,Pres)
    K_co2a          =   KeqFuncs['co2a'](Temp,Pres)
    act_CO2_aq      =   x_CO2g * K_co2a
    a_vals               =   2 * K_carb
    b_vals               =   K_wate + K_bica * x_CO2g
    c_vals               =   - 2*(7/8)**(8/15)*K_grun**7*K_bica*x_CO2g**(29/15)
    d_vals               =   - K_bica**2 * x_CO2g**2
    def f(x, a, b, c, d):
        return a * x**3 + b * x**2 + c * x**(1/15) + d
    def f1(x, a, b, c, d):
        return 3 * a * x**2 + 2 * b * x + (1/15) * c * x**(-14/15)
    def f2(x, a, b, c, d):
        return 6 * a * x + 2 * b - (14/225) * c * x**(-29/15) 
    x_init  = x_CO2g**(1/2)
    x0 = newton(f, x_init, fprime=f1, fprime2=f2, args=(a_vals, b_vals, c_vals, d_vals))
    act_HCO3_m      = x0
    act_CO3_mm      = K_carb / K_bica * act_HCO3_m**2 / x_CO2g
    alk             = 2 * act_CO3_mm + act_HCO3_m
    dic             = act_HCO3_m + act_CO2_aq + act_CO3_mm
    act_H_p         = K_bica * x_CO2g  / act_HCO3_m
    pH              = - np.log10(act_H_p)
    act_SiO2        = 4/7 * act_HCO3_m
    act_DIV_pp      = 0.5 * act_HCO3_m
    act_MON_p       = 0
    allSpecies = {'HCO3':act_HCO3_m, 'CO3':act_CO3_mm, 'ALK':alk, 'DIC':dic, 'pH':pH, 'H':act_H_p,
                  'SiO2':act_SiO2, 'DIV':act_DIV_pp, 'MON':act_MON_p, 'CO2':act_CO2_aq}
    return allSpecies

# %%
def basa_eq(x_CO2g, Temp, Pres, KeqFuncs):
    '''Calculate DIC, pH and DIC components for basalt dissolution.
    
    Solves for the bicarbonate ion activity as a function of temperature, 
    partial pressure of CO2 and pressure. Simultaneous solution of activities
    of all species in the system of reactions.
    Polynomial equation of the form: a * x**4 + b * x**3 + c * x + d
    Reactions included: wollastonite dissolution, enstatite dissolution, 
    ferrosilite dissolution, anorthite dissolution, albite dissolution,
    atmospheric CO2 dissolution, bicarbonate ion formation, 
    carbonate ion formation, water autoionization.
    
    Parameters
    ----------
    x_CO2g : float
        Partial pressure of CO2 (ppm)
    Temp : float
        Temperature of the reactions        
    Pres : float
        Pressure of the reactions
    KeqFuncs : dict of {str : float}
        2D interpolated function in temperature, pressure with 4-letter keys
    
    Returns
    -------
    dic : float
        Total dissolved inorganic carbon
    pH : float
        Negative logarithm of proton activity    
    act_HCO3_m : float
        Activity of the bicarbonate ion
    act_CO2_aq : float
        Activity of aqueous CO2
    act_CO3_mm : float
        Activity of the carbonate ion
        
    '''


    K_woll          =   KeqFuncs['woll'](Temp,Pres)
    K_enst          =   KeqFuncs['enst'](Temp,Pres)
    K_ferr          =   KeqFuncs['ferr'](Temp,Pres)
    K_anor          =   KeqFuncs['anor'](Temp,Pres)
    K_albi          =   KeqFuncs['albi'](Temp,Pres)
    K_bica          =   KeqFuncs['bica'](Temp,Pres)
    K_carb          =   KeqFuncs['carb'](Temp,Pres)
    K_wate          =   KeqFuncs['wate'](Temp,Pres)
    K_co2a          =   KeqFuncs['co2a'](Temp,Pres)
    act_CO2_aq      =   x_CO2g * K_co2a
    a_vals               =   2 * K_woll**2 * K_carb
    b_vals               =   K_woll**2 * K_wate + K_woll**2 * K_bica * x_CO2g
    c_vals               =   - x_CO2g**2 * (K_woll**2 * K_bica**2 \
                                           + K_anor**4 * K_bica * K_albi)
    d_vals               =   - x_CO2g**3 * 2 * K_woll * K_anor**2 * \
                                            K_bica * (K_woll + K_enst + K_ferr)
    def f(x, a, b, c, d):
        return x**4 + b/a * x**3 + c/a * x + d/a
    def f1(x, a, b, c, d):
        return 4 * x**3 + 3 * b/a * x**2 + c/a
    def f2(x, a, b, c, d):
        return 12 * x**2 + 6 * b/a * x
    x_init  = 10000 * x_CO2g**(1/2)
    x0 = newton(f, x_init, fprime=f1, fprime2=f2, args=(a_vals, b_vals, c_vals, d_vals))
    act_HCO3_m      = x0
    act_CO3_mm      = K_carb / K_bica * act_HCO3_m**2 / x_CO2g
    alk             = 2 * act_CO3_mm + act_HCO3_m
    dic             = act_HCO3_m + act_CO2_aq + act_CO3_mm
    act_H_p         = K_bica * x_CO2g  / act_HCO3_m
    pH              = - np.log10(act_H_p)
    act_SiO2        = K_woll / K_anor**2
    act_DIV_pp      = (1 + K_enst/K_woll + K_ferr/K_woll) * K_anor**2 * (x_CO2g**2 / act_HCO3_m**2)
    act_MON_p       = K_anor**4 / K_woll**2 * (x_CO2g / act_HCO3_m)
    allSpecies = {'HCO3':act_HCO3_m, 'CO3':act_CO3_mm, 'ALK':alk, 'DIC':dic, 'pH':pH, 'H':act_H_p,
                  'SiO2':act_SiO2, 'DIV':act_DIV_pp, 'MON':act_MON_p, 'CO2':act_CO2_aq}
    return allSpecies

# %%
def bash_eq(x_CO2g, Temp, Pres, KeqFuncs):
    '''Calculate DIC, pH and DIC components for basalt dissolution.
    
    Solves for the bicarbonate ion activity as a function of temperature, 
    partial pressure of CO2 and pressure. Simultaneous solution of activities
    of all species in the system of reactions.
    Polynomial equation of the form: a * x**4 + b * x**3 + c * x + d
    Reactions included: wollastonite dissolution, enstatite dissolution, 
    ferrosilite dissolution, anorthite dissolution, albite dissolution,
    atmospheric CO2 dissolution, bicarbonate ion formation, 
    carbonate ion formation, water autoionization. (Halloysite precipitation)
    
    Parameters
    ----------
    x_CO2g : float
        Partial pressure of CO2 (ppm)
    Temp : float
        Temperature of the reactions        
    Pres : float
        Pressure of the reactions
    KeqFuncs : dict of {str : float}
        2D interpolated function in temperature, pressure with 4-letter keys
    
    Returns
    -------
    dic : float
        Total dissolved inorganic carbon
    pH : float
        Negative logarithm of proton activity    
    act_HCO3_m : float
        Activity of the bicarbonate ion
    act_CO2_aq : float
        Activity of aqueous CO2
    act_CO3_mm : float
        Activity of the carbonate ion
        
    '''


    K_woll          =   KeqFuncs['woll'](Temp,Pres)
    K_enst          =   KeqFuncs['enst'](Temp,Pres)
    K_ferr          =   KeqFuncs['ferr'](Temp,Pres)
    K_anor          =   KeqFuncs['anoh'](Temp,Pres)
    K_albi          =   KeqFuncs['albh'](Temp,Pres)
    K_bica          =   KeqFuncs['bica'](Temp,Pres)
    K_carb          =   KeqFuncs['carb'](Temp,Pres)
    K_wate          =   KeqFuncs['wate'](Temp,Pres)
    K_co2a          =   KeqFuncs['co2a'](Temp,Pres)
    act_CO2_aq      =   x_CO2g * K_co2a
    a_vals               =   2 * K_woll**2 * K_carb
    b_vals               =   K_woll**2 * K_wate + K_woll**2 * K_bica * x_CO2g
    c_vals               =   - x_CO2g**2 * (K_woll**2 * K_bica**2 \
                                           + K_anor**4 * K_bica * K_albi)
    d_vals               =   - x_CO2g**3 * 2 * K_woll * K_anor**2 * \
                                            K_bica * (K_woll + K_enst + K_ferr)
    def f(x, a, b, c, d):
        return x**4 + b/a * x**3 + c/a * x + d/a
    def f1(x, a, b, c, d):
        return 4 * x**3 + 3 * b/a * x**2 + c/a
    def f2(x, a, b, c, d):
        return 12 * x**2 + 6 * b/a * x
    x_init  = 10000 * x_CO2g**(1/2)
    x0 = newton(f, x_init, fprime=f1, fprime2=f2, args=(a_vals, b_vals, c_vals, d_vals))
    act_HCO3_m      = x0
    act_CO3_mm      = K_carb / K_bica * act_HCO3_m**2 / x_CO2g
    alk             = 2 * act_CO3_mm + act_HCO3_m
    dic             = act_HCO3_m + act_CO2_aq + act_CO3_mm
    act_H_p         = K_bica * x_CO2g  / act_HCO3_m
    pH              = - np.log10(act_H_p)
    act_SiO2        = K_woll / K_anor**2
    act_DIV_pp      = (1 + K_enst/K_woll + K_ferr/K_woll) * K_anor**2 * (x_CO2g**2 / act_HCO3_m**2)
    act_MON_p       = K_anor**4 / K_woll**2 * (x_CO2g / act_HCO3_m)
    allSpecies = {'HCO3':act_HCO3_m, 'CO3':act_CO3_mm, 'ALK':alk, 'DIC':dic, 'pH':pH, 'H':act_H_p,
                  'SiO2':act_SiO2, 'DIV':act_DIV_pp, 'MON':act_MON_p, 'CO2':act_CO2_aq}
    return allSpecies


# %%
def peri_eq(x_CO2g, Temp, Pres, KeqFuncs):
    '''Calculate DIC, pH and DIC components for peridotite dissolution.
    
    Solves for the bicarbonate ion activity as a function of temperature, 
    partial pressure of CO2 and pressure. Simultaneous solution of activities
    of all species in the system of reactions.
    Polynomial equation of the form: a * x**4 + b * x**3 + c * x + d
    Reactions included: wollastonite dissolution, enstatite dissolution, 
    forsterite dissolution, fayalite dissolution, atm. CO2 dissolution,
    bicarbonate ion formation, carbonate ion formation, water autoionization.
    
    Parameters
    ----------
    x_CO2g : float
        Partial pressure of CO2 (ppm)
    Temp : float
        Temperature of the reactions        
    Pres : float
        Pressure of the reactions
    KeqFuncs : dict of {str : float}
        2D interpolated function in temperature, pressure with 4-letter keys
    
    Returns
    -------
    dic : float
        Total dissolved inorganic carbon
    pH : float
        Negative logarithm of proton activity    
    act_HCO3_m : float
        Activity of the bicarbonate ion
    act_CO2_aq : float
        Activity of aqueous CO2
    act_CO3_mm : float
        Activity of the carbonate ion
        
    '''
    

    K_woll          =   KeqFuncs['woll'](Temp,Pres)
    K_enst          =   KeqFuncs['enst'](Temp,Pres)
    K_fors          =   KeqFuncs['fors'](Temp,Pres)
    K_faya          =   KeqFuncs['faya'](Temp,Pres)
    K_bica          =   KeqFuncs['bica'](Temp,Pres)
    K_carb          =   KeqFuncs['carb'](Temp,Pres)
    K_wate          =   KeqFuncs['wate'](Temp,Pres)
    K_co2a          =   KeqFuncs['co2a'](Temp,Pres)
    act_CO2_aq      =   x_CO2g * K_co2a
    a_vals               =   2 * K_enst**2 * K_carb
    b_vals               =   K_enst**2 * ( K_wate +  K_bica * x_CO2g )
    c_vals               =   - x_CO2g**2 * K_enst**2 * K_bica**2 
    d_vals               =   -2*K_fors*K_bica*(K_fors*K_enst + K_faya*K_enst +\
                                         K_woll*K_fors)*x_CO2g**3 
    def f(x, a, b, c, d):
        return x**4 + b/a * x**3 + c/a * x + d/a 
    def f1(x, a, b, c, d):
        return 4 * x**3 + 3 * b/a * x**2 + c/a
    def f2(x, a, b, c, d):
        return 12 * x**2 + 6 * b/a * x 
    x_init  = x_CO2g**(1/2)
    x0 = newton(f, x_init, fprime=f1, fprime2=f2, args=(a_vals, b_vals, c_vals, d_vals))
    act_HCO3_m      = x0
    act_CO3_mm      = K_carb / K_bica * act_HCO3_m**2 / x_CO2g
    alk             = 2 * act_CO3_mm + act_HCO3_m
    dic             = act_HCO3_m + act_CO2_aq + act_CO3_mm
    act_H_p         = K_bica * x_CO2g  / act_HCO3_m
    pH              = - np.log10(act_H_p)
    act_SiO2        = K_enst**2 / K_fors**2
    act_DIV_pp      = (K_woll*K_fors/K_enst + K_fors + K_faya) * K_fors / K_enst * (x_CO2g**2 / act_HCO3_m**2)
    act_MON_p       = 0
    allSpecies = {'HCO3':act_HCO3_m, 'CO3':act_CO3_mm, 'ALK':alk, 'DIC':dic, 'pH':pH, 'H':act_H_p,
                  'SiO2':act_SiO2, 'DIV':act_DIV_pp, 'MON':act_MON_p, 'CO2':act_CO2_aq}
    return allSpecies

# %%
def gran_eq(x_CO2g, Temp, Pres, KeqFuncs):
    '''Calculate DIC, pH and DIC components for granite dissolution.
    
    Solves for the bicarbonate ion activity as a function of temperature, 
    partial pressure of CO2 and pressure. Simultaneous solution of activities
    of all species in the system of reactions.
    Polynomial equation of the form: a * x**4 + b * x**3 + c * x + d
    Reactions included: albite dissolution, K-feldspar dissolution, 
    phlogopite dissolution, annite dissolution, quartz dissolution, 
    atmospheric CO2 dissolution, bicarbonate ion formation, 
    carbonate ion formation, water autoionization.
    
    Parameters
    ----------
    x_CO2g : float
        Partial pressure of CO2 (ppm)
    Temp : float
        Temperature of the reactions        
    Pres : float
        Pressure of the reactions
    KeqFuncs : dict of {str : float}
        2D interpolated function in temperature, pressure with 4-letter keys
    
    Returns
    -------
    dic : float
        Total dissolved inorganic carbon
    pH : float
        Negative logarithm of proton activity    
    act_HCO3_m : float
        Activity of the bicarbonate ion
    act_CO2_aq : float
        Activity of aqueous CO2
    act_CO3_mm : float
        Activity of the carbonate ion
        
    '''
    

    K_albi          =   KeqFuncs['albi'](Temp,Pres)
    K_kfel          =   KeqFuncs['kfel'](Temp,Pres)
    K_phlo          =   KeqFuncs['phlo'](Temp,Pres)
    K_anni          =   KeqFuncs['anni'](Temp,Pres)
    K_quar          =   KeqFuncs['quar'](Temp,Pres)
    K_bica          =   KeqFuncs['bica'](Temp,Pres)
    K_carb          =   KeqFuncs['carb'](Temp,Pres)
    K_wate          =   KeqFuncs['wate'](Temp,Pres)
    K_co2a          =   KeqFuncs['co2a'](Temp,Pres)
    act_CO2_aq      =   x_CO2g * K_co2a
    a_vals               =   2 * K_carb * K_quar**2 * K_kfel**(1/3)
    b_vals               =   K_quar**2 * K_kfel**(1/3) *(K_wate + K_bica*x_CO2g)
    c_vals               =   -  K_bica * K_kfel**(1/3) * (K_bica * K_quar**2 \
                                            + (K_kfel + K_albi))*x_CO2g**2 
    d_vals               =   - 2 * K_bica * K_quar**2 \
                        * (K_phlo + K_anni) * x_CO2g**3 
    def f(x, a, b, c, d):
        return x**4 + b/a * x**3 + c/a * x + d/a
    def f1(x, a, b, c, d):
        return 4 * x**3 + 3 * b/a * x**2 + c/a
    def f2(x, a, b, c, d):
        return 12 * x**2 + 6 * b/a * x
    x_init  = 10 * x_CO2g**(1/2)
    x0 = newton(f, x_init, fprime=f1, fprime2=f2, args=(a_vals, b_vals, c_vals, d_vals))
    act_HCO3_m      = x0
    act_CO3_mm      = K_carb / K_bica * act_HCO3_m**2 / x_CO2g
    alk             = 2 * act_CO3_mm + act_HCO3_m
    dic             = act_HCO3_m + act_CO2_aq + act_CO3_mm
    act_H_p         = K_bica * x_CO2g  / act_HCO3_m
    pH              = - np.log10(act_H_p)
    act_SiO2        = K_quar
    act_DIV_pp      = (K_phlo + K_anni) / K_kfel**(1/3) * (x_CO2g**2 / act_HCO3_m**2)
    act_MON_p       = (K_albi + K_kfel) / K_quar**2 * (x_CO2g / act_HCO3_m)
    allSpecies = {'HCO3':act_HCO3_m, 'CO3':act_CO3_mm, 'ALK':alk, 'DIC':dic, 'pH':pH, 'H':act_H_p,
                  'SiO2':act_SiO2, 'DIV':act_DIV_pp, 'MON':act_MON_p, 'CO2':act_CO2_aq}
    return allSpecies

# %%
def grah_eq(x_CO2g, Temp, Pres, KeqFuncs):
    '''Calculate DIC, pH and DIC components for granite dissolution.
    
    Solves for the bicarbonate ion activity as a function of temperature, 
    partial pressure of CO2 and pressure. Simultaneous solution of activities
    of all species in the system of reactions.
    Polynomial equation of the form: a * x**4 + b * x**3 + c * x + d
    Reactions included: albite dissolution, K-feldspar dissolution, 
    phlogopite dissolution, annite dissolution, quartz dissolution, 
    atmospheric CO2 dissolution, bicarbonate ion formation, 
    carbonate ion formation, water autoionization. (Halloysite precipitation)
    
    Parameters
    ----------
    x_CO2g : float
        Partial pressure of CO2 (ppm)
    Temp : float
        Temperature of the reactions        
    Pres : float
        Pressure of the reactions
    KeqFuncs : dict of {str : float}
        2D interpolated function in temperature, pressure with 4-letter keys
    
    Returns
    -------
    dic : float
        Total dissolved inorganic carbon
    pH : float
        Negative logarithm of proton activity    
    act_HCO3_m : float
        Activity of the bicarbonate ion
    act_CO2_aq : float
        Activity of aqueous CO2
    act_CO3_mm : float
        Activity of the carbonate ion
        
    '''
    

    K_albi          =   KeqFuncs['albh'](Temp,Pres)
    K_kfel          =   KeqFuncs['kfeh'](Temp,Pres)
    K_phlo          =   KeqFuncs['phlh'](Temp,Pres)
    K_anni          =   KeqFuncs['annh'](Temp,Pres)
    K_quar          =   KeqFuncs['quar'](Temp,Pres)
    K_bica          =   KeqFuncs['bica'](Temp,Pres)
    K_carb          =   KeqFuncs['carb'](Temp,Pres)
    K_wate          =   KeqFuncs['wate'](Temp,Pres)
    K_co2a          =   KeqFuncs['co2a'](Temp,Pres)
    act_CO2_aq      =   x_CO2g * K_co2a
    a_vals               =   2 * K_carb * K_quar**2 * K_kfel**(1/3)
    b_vals               =   K_quar**2 * K_kfel**(1/3) *(K_wate + K_bica*x_CO2g)
    c_vals               =   -  K_bica * K_kfel**(1/3) * (K_bica * K_quar**2 \
                                            + (K_kfel + K_albi))*x_CO2g**2 
    d_vals               =   - 2 * K_bica * K_quar**2 \
                        * (K_phlo + K_anni) * x_CO2g**3 
    def f(x, a, b, c, d):
        return x**4 + b/a * x**3 + c/a * x + d/a
    def f1(x, a, b, c, d):
        return 4 * x**3 + 3 * b/a * x**2 + c/a
    def f2(x, a, b, c, d):
        return 12 * x**2 + 6 * b/a * x
    x_init  = 10 * x_CO2g**(1/2)
    x0 = newton(f, x_init, fprime=f1, fprime2=f2, args=(a_vals, b_vals, c_vals, d_vals))
    act_HCO3_m      = x0
    act_CO3_mm      = K_carb / K_bica * act_HCO3_m**2 / x_CO2g
    alk             = 2 * act_CO3_mm + act_HCO3_m
    dic             = act_HCO3_m + act_CO2_aq + act_CO3_mm
    act_H_p         = K_bica * x_CO2g  / act_HCO3_m
    pH              = - np.log10(act_H_p)
    act_SiO2        = K_quar
    act_DIV_pp      = (K_phlo + K_anni) / K_kfel**(1/3) * (x_CO2g**2 / act_HCO3_m**2)
    act_MON_p       = (K_albi + K_kfel) / K_quar**2 * (x_CO2g / act_HCO3_m)
    allSpecies = {'HCO3':act_HCO3_m, 'CO3':act_CO3_mm, 'ALK':alk, 'DIC':dic, 'pH':pH, 'H':act_H_p,
                  'SiO2':act_SiO2, 'DIV':act_DIV_pp, 'MON':act_MON_p, 'CO2':act_CO2_aq}
    return allSpecies


# %%
def fit_powerlaw(x_CO2g,C):
    '''Fits a power-law, C = m * x_CO2g**n and returns a and n
    
    Calls curve_fit to fit a power law.
    
    Parameters
    ----------
    x_CO2g : float
        Partial pressure of CO2 (ppm)
    C : float
        Concentration (mol/dm3)
    
    Returns
    -------
    popt : array of float
        Fitted parameters a and n
        
    '''
    
    x_CO2g = np.log10(x_CO2g)
    C = np.log10(C)
    def func(x,m,n):
        return m + n * x
    guess = (1, 0.5)
    popt, pcov = curve_fit(func,x_CO2g,C,guess)
    return popt

# %%
def fit_powerlaw_T(T,C):
    '''Fits a power-law, DIC = N * exp(x_CO2g) and returns a and n
    
    Calls curve_fit to fit a power law. 
    
    Parameters
    ----------
    T : float
        Temperature (K)
    C : float
        Concentration (mol/dm3)
    
    Returns
    -------
    popt : array of float
        Fitted parameters a and n
        
    '''
    
    C = np.log(C)
    def func(x,logN,E_a):
        return logN - E_a / (x * R.value)
        # return logN  + (x - 288 ) / T_e
    guess = (1, 10)
    popt, pcov = curve_fit(func,T,C,guess)
    return popt

# %%
#KeqFuncs   = import_thermo_data('./database/species.csv')

# %%
#print(KeqFuncs['anor'])

# %%
#anor_eq(xCO2, T, Pfull, KeqFuncs)

# %%
