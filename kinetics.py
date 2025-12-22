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
# kinetics.py (generates all figures (+ additional) from the published paper)
#
# Minerals included: kaolinite: kaol, wollastonite: woll, enstatite: enst,
#                    forsterite: fors, fayalite: faya, anorthite: anor, 
#                    albite: albi, K-feldspar: kfel, muscovite: musc,
#                    phlogopite: phlo, quartz: quar
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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.optimize import newton
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.interpolate import RegularGridInterpolator as interpnd
import astropy.units as u
u.imperial.enable()
from astropy.constants import R
from parameters import *

# %%
def import_kinetics_data():
    '''Import mineral dissolution rates from Palandri & Kharaka (2004).
    
    Column names: Mineral, A_acid, log10(k_acid), E_acid, n_acid, A_neut, 
    log10(k_neut), A_base, log10(k_base), E_base, n_base
    
    Note
    ----
    Units of A, k, and E are in mol/m2/s, mol/m2/s and kJ/mol, respectively.
    Imports a csv file with a specific format and a predefined path.
    
    Returns
    -------
    logkDict : dict of {str : float}
        Kinetics data with 4-letter keys
    
    '''
    
    logkDict = {}
    df = pd.read_csv('./database/kinetics_data.csv', \
                              skiprows=lambda x: x in [1, 1], \
                              usecols=['Mineral','log10(k_acid)','E_acid', \
                                       'n_acid','log10(k_neut)','E_neut', \
                                      'log10(k_base)','E_base','n_base'])
    logkDict['quar'] = df.loc[df['Mineral'] == 'quartz']
    logkDict['albi'] = df.loc[df['Mineral'] == 'albite']
    logkDict['anor'] = df.loc[df['Mineral'] == 'anorthite']
    logkDict['kfel'] = df.loc[df['Mineral'] == 'K-feldspar']
    logkDict['fors'] = df.loc[df['Mineral'] == 'forsterite']
    logkDict['faya'] = df.loc[df['Mineral'] == 'fayalite']
    logkDict['enst'] = df.loc[df['Mineral'] == 'enstatite']
    logkDict['woll'] = df.loc[df['Mineral'] == 'wollastonite']
    logkDict['anth'] = df.loc[df['Mineral'] == 'anthophyllite']
    logkDict['musc'] = df.loc[df['Mineral'] == 'muscovite']
    logkDict['phlo'] = df.loc[df['Mineral'] == 'phlogopite']
            
    return logkDict

# %%
def get_keff(Temp, pHall, logkDict):
    '''Import mineral dissolution rates from Palandri & Kharaka (2004).
    
    Column names: Mineral, A_acid, log10(k_acid), E_acid, n_acid, A_neut, 
    log10(k_neut), A_base, log10(k_base), E_base, n_base
    
    Note
    ----
    Units of A, k, and E are in mol/m2/s, mol/m2/s and kJ/mol, respectively.
    Imports a csv file with a specific format and a predefined path.

    Parameters
    ----------
    Temp : float
        Temperature of the reactions (K)
    pH : float
        pH of the reactions
    logkDict : dict of {str : float}
        Kinetics data with 4-letter keys
        
    Returns
    -------
    keff : dict of {str : float}
        Kinetics data with 4-letter keys (mol/m2/yr)
    
    '''
    
    kFuncs   = {}
    
    names = np.array(['quar','albi','anor','kfel','fors','faya','enst','ferr',\
                     'woll','anth','grun','musc','phlo','anni','albh','anoh',\
                     'kfeh','mush','phlh','annh'])
    f_names = np.array([quar_ki,albi_ki,anor_ki,kfel_ki,fors_ki,faya_ki,enst_ki,ferr_ki,\
                       woll_ki,anth_ki,grun_ki,musc_ki,phlo_ki,anni_ki,albi_ki,anor_ki,\
                       kfel_ki,musc_ki,phlo_ki,anni_ki])

    k_eff = np.zeros((len(names), len(Temp), len(pHall)))

    # Points for interpolation
    points = (Temp, pHall)
    
    # Make a 2D grid for each of the parameters
    Temp, pHall = np.meshgrid(Temp, pHall, indexing='ij')
    
    for i, func in enumerate(f_names):
        # Ellipsis notation is equivalent to doing :, :, : (and so on)
        k_eff[i, ...] = func(Temp, pHall, logkDict)

        kFuncs[names[i]] = lambda T, pH, f=func: f(T, pH, logkDict).value
        
    return kFuncs

# %%
def quar_ki(Temp, pH, logkDict):
    '''Calculate k_eff for quartz dissolution.
    
    Parameters
    ----------
    Temp : float
        Temperature of the reaction (K)
    pH : float
        pH of the reaction
    logkDict : dict of {str : float}
        Kinetics data with 4-letter keys
    
    Returns
    -------
    keff : float
        Kinetic rate coefficient (mol/m2/yr)
        
    '''

    k_n = 10**(logkDict['quar']['log10(k_neut)'].values[0])*u.mol/u.m/u.m/u.s
    k_b = 10**(logkDict['quar']['log10(k_base)'].values[0])*u.mol/u.m/u.m/u.s
    E_n = logkDict['quar']['E_neut'].values[0] * u.kJ/u.mol
    E_b = logkDict['quar']['E_base'].values[0] * u.kJ/u.mol
    n_b = logkDict['quar']['n_base'].values[0]
    
    neut = k_n * np.exp(-E_n/R*(1/Temp-1/T0)/u.K)
    base = k_b * 10**(-pH*n_b) * np.exp(-E_b/R*(1/Temp-1/T0)/u.K)
    return (neut + base).to(u.mol/(u.m)**2/u.a)

# %%
def kfel_ki(Temp, pH, logkDict):
    '''Calculate k_eff for K-feldspar dissolution.
    
    Parameters
    ----------
    Temp : float
        Temperature of the reaction (K)
    pH : float
        pH of the reaction
    logkDict : dict of {str : float}
        Kinetics data with 4-letter keys
    
    Returns
    -------
    keff : float
        Kinetic rate coefficient (mol/m2/yr)
        
    '''

    k_a = 10**(logkDict['kfel']['log10(k_acid)'].values[0])*u.mol/u.m/u.m/u.s
    k_n = 10**(logkDict['kfel']['log10(k_neut)'].values[0])*u.mol/u.m/u.m/u.s
    k_b = 10**(logkDict['kfel']['log10(k_base)'].values[0])*u.mol/u.m/u.m/u.s
    E_a = logkDict['kfel']['E_acid'].values[0] * u.kJ/u.mol
    E_n = logkDict['kfel']['E_neut'].values[0] * u.kJ/u.mol
    E_b = logkDict['kfel']['E_base'].values[0] * u.kJ/u.mol
    n_a = logkDict['kfel']['n_acid'].values[0]
    n_b = logkDict['kfel']['n_base'].values[0]
    
    acid = k_a * 10**(-pH*n_a) * np.exp(-E_a/R*(1/Temp-1/T0)/u.K)
    neut = k_n * np.exp(-E_n/R*(1/Temp-1/T0)/u.K)
    base = k_b * 10**(-pH*n_b) * np.exp(-E_b/R*(1/Temp-1/T0)/u.K)
    return (acid + neut + base).to(u.mol/(u.m)**2/u.a)

# %%
def albi_ki(Temp, pH, logkDict):
    '''Calculate k_eff for albite dissolution.
    
    Parameters
    ----------
    Temp : float
        Temperature of the reaction (K)
    pH : float
        pH of the reaction
    logkDict : dict of {str : float}
        Kinetics data with 4-letter keys
    
    Returns
    -------
    keff : float
        Kinetic rate coefficient (mol/m2/yr)
        
    '''

    k_a = 10**(logkDict['albi']['log10(k_acid)'].values[0])*u.mol/u.m/u.m/u.s
    k_n = 10**(logkDict['albi']['log10(k_neut)'].values[0])*u.mol/u.m/u.m/u.s
    k_b = 10**(logkDict['albi']['log10(k_base)'].values[0])*u.mol/u.m/u.m/u.s
    E_a = logkDict['albi']['E_acid'].values[0] * u.kJ/u.mol
    E_n = logkDict['albi']['E_neut'].values[0] * u.kJ/u.mol
    E_b = logkDict['albi']['E_base'].values[0] * u.kJ/u.mol
    n_a = logkDict['albi']['n_acid'].values[0]
    n_b = logkDict['albi']['n_base'].values[0]
    
    acid = k_a * 10**(-pH*n_a) * np.exp(-E_a/R*(1/Temp-1/T0)/u.K)
    neut = k_n * np.exp(-E_n/R*(1/Temp-1/T0)/u.K)
    base = k_b * 10**(-pH*n_b) * np.exp(-E_b/R*(1/Temp-1/T0)/u.K)
    return (acid + neut + base).to(u.mol/(u.m)**2/u.a)

# %%
def musc_ki(Temp, pH, logkDict):
    '''Calculate k_eff for muscovite dissolution.
    
    Parameters
    ----------
    Temp : float
        Temperature of the reaction (K)
    pH : float
        pH of the reaction
    logkDict : dict of {str : float}
        Kinetics data with 4-letter keys
    
    Returns
    -------
    keff : float
        Kinetic rate coefficient (mol/m2/yr)
        
    '''

    k_a = 10**(logkDict['musc']['log10(k_acid)'].values[0])*u.mol/u.m/u.m/u.s
    k_n = 10**(logkDict['musc']['log10(k_neut)'].values[0])*u.mol/u.m/u.m/u.s
    k_b = 10**(logkDict['musc']['log10(k_base)'].values[0])*u.mol/u.m/u.m/u.s
    E_a = logkDict['musc']['E_acid'].values[0] * u.kJ/u.mol
    E_n = logkDict['musc']['E_neut'].values[0] * u.kJ/u.mol
    E_b = logkDict['musc']['E_base'].values[0] * u.kJ/u.mol
    n_a = logkDict['musc']['n_acid'].values[0]
    n_b = logkDict['musc']['n_base'].values[0]
    
    acid = k_a * 10**(-pH*n_a) * np.exp(-E_a/R*(1/Temp-1/T0)/u.K)
    neut = k_n * np.exp(-E_n/R*(1/Temp-1/T0)/u.K)
    base = k_b * 10**(-pH*n_b) * np.exp(-E_b/R*(1/Temp-1/T0)/u.K)
    return (acid + neut + base).to(u.mol/(u.m)**2/u.a)

# %%
def kaol_ki(Temp, pH, logkDict):
    '''Calculate k_eff for kaolinite dissolution.
    
    Parameters
    ----------
    Temp : float
        Temperature of the reaction (K)
    pH : float
        pH of the reaction
    logkDict : dict of {str : float}
        Kinetics data with 4-letter keys
    
    Returns
    -------
    keff : float
        Kinetic rate coefficient (mol/m2/yr)
        
    '''

    k_a = 10**(logkDict['kaol']['log10(k_acid)'].values[0])*u.mol/u.m/u.m/u.s
    k_n = 10**(logkDict['kaol']['log10(k_neut)'].values[0])*u.mol/u.m/u.m/u.s
    k_b = 10**(logkDict['kaol']['log10(k_base)'].values[0])*u.mol/u.m/u.m/u.s
    E_a = logkDict['kaol']['E_acid'].values[0] * u.kJ/u.mol
    E_n = logkDict['kaol']['E_neut'].values[0] * u.kJ/u.mol
    E_b = logkDict['kaol']['E_base'].values[0] * u.kJ/u.mol
    n_a = logkDict['kaol']['n_acid'].values[0]
    n_b = logkDict['kaol']['n_base'].values[0]
    
    acid = k_a * 10**(-pH*n_a) * np.exp(-E_a/R*(1/Temp-1/T0)/u.K)
    neut = k_n * np.exp(-E_n/R*(1/Temp-1/T0)/u.K)
    base = k_b * 10**(-pH*n_b) * np.exp(-E_b/R*(1/Temp-1/T0)/u.K)
    return (acid + neut + base).to(u.mol/(u.m)**2/u.a)

# %%
def woll_ki(Temp, pH, logkDict):
    '''Calculate k_eff for wollastonite dissolution.
    
    Parameters
    ----------
    Temp : float
        Temperature of the reaction (K)
    pH : float
        pH of the reaction
    logkDict : dict of {str : float}
        Kinetics data with 4-letter keys
    
    Returns
    -------
    keff : float
        Kinetic rate coefficient (mol/m2/yr)
        
    '''
    
    k_a = 10**(logkDict['woll']['log10(k_acid)'].values[0])*u.mol/u.m/u.m/u.s
    k_n = 10**(logkDict['woll']['log10(k_neut)'].values[0])*u.mol/u.m/u.m/u.s
    E_a = logkDict['woll']['E_acid'].values[0] * u.kJ/u.mol
    E_n = logkDict['woll']['E_neut'].values[0] * u.kJ/u.mol
    n_a = logkDict['woll']['n_acid'].values[0] 
    
    acid = k_a * 10**(-pH*n_a) * np.exp(-E_a/R*(1/Temp-1/T0)/u.K)
    neut = k_n * np.exp(-E_n/R*(1/Temp-1/T0)/u.K)
    return (acid + neut).to(u.mol/(u.m)**2/u.a)

# %%
def enst_ki(Temp, pH, logkDict):
    '''Calculate k_eff for enstatite dissolution.
    
    Parameters
    ----------
    Temp : float
        Temperature of the reaction (K)
    pH : float
        pH of the reaction
    logkDict : dict of {str : float}
        Kinetics data with 4-letter keys
    
    Returns
    -------
    keff : float
        Kinetic rate coefficient (mol/m2/yr)
        
    '''
    
    k_a = 10**(logkDict['enst']['log10(k_acid)'].values[0])*u.mol/u.m/u.m/u.s
    k_n = 10**(logkDict['enst']['log10(k_neut)'].values[0])*u.mol/u.m/u.m/u.s
    E_a = logkDict['enst']['E_acid'].values[0] * u.kJ/u.mol
    E_n = logkDict['enst']['E_neut'].values[0] * u.kJ/u.mol
    n_a = logkDict['enst']['n_acid'].values[0] 
    
    acid = k_a * 10**(-pH*n_a) * np.exp(-E_a/R*(1/Temp-1/T0)/u.K)
    neut = k_n * np.exp(-E_n/R*(1/Temp-1/T0)/u.K)
    return (acid + neut).to(u.mol/(u.m)**2/u.a)

# %%
def ferr_ki(Temp, pH, logkDict):
    '''Calculate k_eff for ferrosilite dissolution.
    
    Calculations based on enstatite due to lack of data.
    
    Parameters
    ----------
    Temp : float
        Temperature of the reaction (K)
    pH : float
        pH of the reaction
    logkDict : dict of {str : float}
        Kinetics data with 4-letter keys
    
    Returns
    -------
    keff : float
        Kinetic rate coefficient (mol/m2/yr)
        
    '''
    
    k_a = 10**(logkDict['enst']['log10(k_acid)'].values[0])*u.mol/u.m/u.m/u.s
    k_n = 10**(logkDict['enst']['log10(k_neut)'].values[0])*u.mol/u.m/u.m/u.s
    E_a = logkDict['enst']['E_acid'].values[0] * u.kJ/u.mol
    E_n = logkDict['enst']['E_neut'].values[0] * u.kJ/u.mol
    n_a = logkDict['enst']['n_acid'].values[0] 
    
    acid = k_a * 10**(-pH*n_a) * np.exp(-E_a/R*(1/Temp-1/T0)/u.K)
    neut = k_n * np.exp(-E_n/R*(1/Temp-1/T0)/u.K)
    return (acid + neut).to(u.mol/(u.m)**2/u.a)

# %%
def fors_ki(Temp, pH, logkDict):
    '''Calculate k_eff for forsterite dissolution.
    
    Parameters
    ----------
    Temp : float
        Temperature of the reaction (K)
    pH : float
        pH of the reaction
    logkDict : dict of {str : float}
        Kinetics data with 4-letter keys
    
    Returns
    -------
    keff : float
        Kinetic rate coefficient (mol/m2/yr)
        
    '''
    
    k_a = 10**(logkDict['fors']['log10(k_acid)'].values[0])*u.mol/u.m/u.m/u.s
    k_n = 10**(logkDict['fors']['log10(k_neut)'].values[0])*u.mol/u.m/u.m/u.s
    E_a = logkDict['fors']['E_acid'].values[0] * u.kJ/u.mol
    E_n = logkDict['fors']['E_neut'].values[0] * u.kJ/u.mol
    n_a = logkDict['fors']['n_acid'].values[0] 
    
    acid = k_a * 10**(-pH*n_a) * np.exp(-E_a/R*(1/Temp-1/T0)/u.K)
    neut = k_n * np.exp(-E_n/R*(1/Temp-1/T0)/u.K)
    return (acid + neut).to(u.mol/(u.m)**2/u.a)

# %%
def faya_ki(Temp, pH, logkDict):
    '''Calculate k_eff for fayalite dissolution.
    
    Parameters
    ----------
    Temp : float
        Temperature of the reaction (K)
    pH : float
        pH of the reaction
    logkDict : dict of {str : float}
        Kinetics data with 4-letter keys
    
    Returns
    -------
    keff : float
        Kinetic rate coefficient (mol/m2/yr)
        
    '''
    
    k_a = 10**(logkDict['faya']['log10(k_acid)'].values[0])*u.mol/u.m/u.m/u.s
    k_n = 10**(logkDict['faya']['log10(k_neut)'].values[0])*u.mol/u.m/u.m/u.s
    E_a = logkDict['faya']['E_acid'].values[0] * u.kJ/u.mol
    E_n = logkDict['faya']['E_neut'].values[0] * u.kJ/u.mol
    n_a = logkDict['faya']['n_acid'].values[0] 
    
    acid = k_a * 10**(-pH*n_a) * np.exp(-E_a/R*(1/Temp-1/T0)/u.K)
    neut = k_n * np.exp(-E_n/R*(1/Temp-1/T0)/u.K)
    return (acid + neut).to(u.mol/(u.m)**2/u.a)

# %%
def anor_ki(Temp, pH, logkDict):
    '''Calculate k_eff for anorthite dissolution.
    
    Parameters
    ----------
    Temp : float
        Temperature of the reaction (K)
    pH : float
        pH of the reaction
    logkDict : dict of {str : float}
        Kinetics data with 4-letter keys
    
    Returns
    -------
    keff : float
        Kinetic rate coefficient (mol/m2/yr)
        
    '''
    
    k_a = 10**(logkDict['anor']['log10(k_acid)'].values[0])*u.mol/u.m/u.m/u.s
    k_n = 10**(logkDict['anor']['log10(k_neut)'].values[0])*u.mol/u.m/u.m/u.s
    E_a = logkDict['anor']['E_acid'].values[0] * u.kJ/u.mol
    E_n = logkDict['anor']['E_neut'].values[0] * u.kJ/u.mol
    n_a = logkDict['anor']['n_acid'].values[0] 
    
    acid = k_a * 10**(-pH*n_a) * np.exp(-E_a/R*(1/Temp-1/T0)/u.K)
    neut = k_n * np.exp(-E_n/R*(1/Temp-1/T0)/u.K)
    return (acid + neut).to(u.mol/(u.m)**2/u.a)

# %%
def anth_ki(Temp, pH, logkDict):
    '''Calculate k_eff for anthophyllite dissolution.
    
    Parameters
    ----------
    Temp : float
        Temperature of the reaction (K)
    pH : float
        pH of the reaction
    logkDict : dict of {str : float}
        Kinetics data with 4-letter keys
    
    Returns
    -------
    keff : float
        Kinetic rate coefficient (mol/m2/yr)
        
    '''
    
    k_a = 10**(logkDict['anth']['log10(k_acid)'].values[0])*u.mol/u.m/u.m/u.s
    k_n = 10**(logkDict['anth']['log10(k_neut)'].values[0])*u.mol/u.m/u.m/u.s
    E_a = logkDict['anth']['E_acid'].values[0] * u.kJ/u.mol
    E_n = logkDict['anth']['E_neut'].values[0] * u.kJ/u.mol
    n_a = logkDict['anth']['n_acid'].values[0] 
    
    acid = k_a * 10**(-pH*n_a) * np.exp(-E_a/R*(1/Temp-1/T0)/u.K)
    neut = k_n * np.exp(-E_n/R*(1/Temp-1/T0)/u.K)
    return (acid + neut).to(u.mol/(u.m)**2/u.a)

# %%
def grun_ki(Temp, pH, logkDict):
    '''Calculate k_eff for grunerite dissolution.
    
    Calculations based on anthophyllite due to lack of data.
    
    Parameters
    ----------
    Temp : float
        Temperature of the reaction (K)
    pH : float
        pH of the reaction
    logkDict : dict of {str : float}
        Kinetics data with 4-letter keys
        
    Returns
    -------
    keff : float
        Kinetic rate coefficient (mol/m2/yr)
        
    '''
    
    k_a = 10**(logkDict['anth']['log10(k_acid)'].values[0])*u.mol/u.m/u.m/u.s
    k_n = 10**(logkDict['anth']['log10(k_neut)'].values[0])*u.mol/u.m/u.m/u.s
    E_a = logkDict['anth']['E_acid'].values[0] * u.kJ/u.mol
    E_n = logkDict['anth']['E_neut'].values[0] * u.kJ/u.mol
    n_a = logkDict['anth']['n_acid'].values[0] 
    
    acid = k_a * 10**(-pH*n_a) * np.exp(-E_a/R*(1/Temp-1/T0)/u.K)
    neut = k_n * np.exp(-E_n/R*(1/Temp-1/T0)/u.K)
    return (acid + neut).to(u.mol/(u.m)**2/u.a)

# %%
def phlo_ki(Temp, pH, logkDict):
    '''Calculate k_eff for phlogopite dissolution.
    
    Parameters
    ----------
    Temp : float
        Temperature of the reaction (K)
    pH : float
        pH of the reaction
    logkDict : dict of {str : float}
        Kinetics data with 4-letter keys
    
    Returns
    -------
    keff : float
        Kinetic rate coefficient (mol/m2/yr)
        
    '''
    
    k_n = 10**(logkDict['phlo']['log10(k_neut)'].values[0])*u.mol/u.m/u.m/u.s
    E_n = logkDict['phlo']['E_neut'].values[0] * u.kJ/u.mol
    
    neut = k_n * np.exp(-E_n/R*(1/Temp-1/T0)/u.K)
    return neut.to(u.mol/(u.m)**2/u.a)

# %%
def anni_ki(Temp, pH, logkDict):
    '''Calculate k_eff for annite dissolution.
    
    Calculations based on phlogopite due to lack of data.
    
    Parameters
    ----------
    Temp : float
        Temperature of the reaction (K)
    pH : float
        pH of the reaction
    logkDict : dict of {str : float}
        Kinetics data with 4-letter keys
    
    Returns
    -------
    keff : float
        Kinetic rate coefficient (mol/m2/yr)
        
    '''
    
    k_n = 10**(logkDict['phlo']['log10(k_neut)'].values[0])*u.mol/u.m/u.m/u.s
    E_n = logkDict['phlo']['E_neut'].values[0] * u.kJ/u.mol
    
    neut = k_n * np.exp(-E_n/R*(1/Temp-1/T0)/u.K)
    return neut.to(u.mol/(u.m)**2/u.a)
