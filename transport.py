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
# transport.py (module to calculate transport effects on weathering)
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
from scipy.optimize import curve_fit
from scipy.optimize import newton
from scipy.interpolate import RegularGridInterpolator as interpnd
import astropy.units as u
from astropy.constants import R

try:
    from .parameters import *
except ImportError:
    from parameters import *

# %%
def C_tr(C_eq, Dw, q):
    '''Return transport-controlled concentration calculated using 
    the solute transport equation from Maher & Chamberlain (2014)
    
    Parameters
    ----------
    C_eq : float
        Equilibrium solute concentration (mol/dm3)
    Dw : float
        Damkoehler coefficient (m/yr)
    q : float
        Runoff or fluid flow rate (m/yr)
    
    Returns
    -------
    C_tr : float
        Transport-controlled solute concentration (mol/dm3)

    '''
    
    return C_eq / (1 + q/Dw)

# %%
def get_C(Ceq, Dw, q):
    '''Return transport-controlled concentration functions calculated using 
    the solute transport equation from Maher & Chamberlain (2014)
    
    Parameters
    ----------
    C_eq : float
        Equilibrium solute concentration (mol/dm3)
    Dw : float
        Damkoehler coefficient (m/yr)
    q : float
        Runoff or fluid flow rate (m/yr)
    
    Returns
    -------
    CFuncs : float
        3D interpolated function for transport-controlled concentration (mol/dm3)

    '''
    
    points     = (Ceq, Dw, q)
    Ceq, Dw, q = np.meshgrid(Ceq, Dw, q, indexing='ij')
    data       = C_tr(Ceq, Dw, q)
    CFuncs     = interpnd(points=points, values=data)
    return CFuncs

# %%
def DIC_tr(act_HCO3_m, x_CO2g, Temp, Pres, KeqFuncs):
    '''Calculate generalized DIC from transport-controlled bicarbonate activity
    using chemical equilibrium
    
    Parameters
    ----------
    act_HCO3_m : float
        Activity of bicarbonate ion
    x_CO2g : float
        Activity of CO2(g)
    Temp : float
        Temperature (K)
    Pres : float
        Pressure (bar)
    KeqFuncs: dict
        Equilibrium constants dictionary
    
    Returns
    -------
    dic : float
        Total dissolved inorganic carbon
    pH : float
        Negative logarithm of H+ activity
    act_HCO3_m : float
        Activity of bicarbonate ion
    act_CO2_aq : float
        Activity of CO2(aq)
    act_CO3_mm : float
        Activity of carbonate ion

    '''
    
    # Temp = Temp[0, 0, :, 0]
    # Pres = Pres[0, 0, 0, :]
    K_bica          =   KeqFuncs['bica'](Temp,Pres)
    K_carb          =   KeqFuncs['carb'](Temp,Pres)
    K_co2a          =   KeqFuncs['co2a'](Temp,Pres)
    act_CO2_aq      =   x_CO2g * K_co2a
    act_H_p         =   K_bica * x_CO2g  / act_HCO3_m
    pH              =   - np.log10(act_H_p)
    act_CO3_mm      =   K_carb / K_bica * act_HCO3_m**2 / x_CO2g
    dic             =   act_HCO3_m + act_CO2_aq + act_CO3_mm
    alk             =   act_HCO3_m + 2 * act_CO3_mm
    allSpecies = {'HCO3':act_HCO3_m, 'CO3':act_CO3_mm, 'ALK':alk, 'DIC':dic, 'pH':pH, 'H':act_H_p,
                  'CO2':act_CO2_aq}
    return allSpecies
    #return dic, pH, act_HCO3_m, act_CO2_aq, act_CO3_mm

# %%
def get_DICtr(HCO3tr, x_CO2g, Temp, Pres, KeqFuncs):
    '''Return generalized DIC concentration functions 
    
    Parameters
    ----------
    HCO3tr : float
        Transport-controlled bicarbonate ion concentration
    x_CO2g : float
        Activity of CO2(g)
    Temp : float
        Temperature (K)
    Pres : float
        Pressure (bar)
    KeqFuncs: dict
        Equilibrium constants dictionary
    
    Returns
    -------
    DICtrFuncs : float
        Generalized DIC concentration functions 

    '''
    
    DICtrFuncs = {}
    
    # Points for interpolation
    points = (HCO3tr, x_CO2g, Temp, Pres)
    
    # Make a 3D grid for each of the parameters
    HCO3tr, x_CO2g, Temp, Pres = np.meshgrid(HCO3tr, x_CO2g, Temp, Pres, indexing='ij')
    
    # Ellipsis notation is equivalent to doing :, :, : (and so on)
    
    allSpecies   = DIC_tr(HCO3tr, x_CO2g, Temp, Pres, KeqFuncs)
    #print(allSpecies)
    DIC  = allSpecies['DIC']
    pH   = allSpecies['pH']
    HCO3 = allSpecies['HCO3']
    CO2  = allSpecies['CO2']
    CO3  = allSpecies['CO3']
    ALK  = allSpecies['ALK']
        

    DICtrFuncs['DIC']  = interpnd(points=points, values=DIC)
    DICtrFuncs['pH']   = interpnd(points=points, values=pH)
    DICtrFuncs['HCO3'] = interpnd(points=points, values=HCO3)
    DICtrFuncs['CO2']  = interpnd(points=points, values=CO2)
    DICtrFuncs['CO3']  = interpnd(points=points, values=CO3)
    DICtrFuncs['ALK']  = interpnd(points=points, values=ALK)
    
    return DICtrFuncs

# %%
def w_flux(C, q):
    '''Return weathering flux for a given C and q.
    
    Parameters
    ----------
    C : float
        Solute concentration (mol/dm3)
    q : float
        Runoff or fluid flow rate (m/yr)
    
    Returns
    -------
    w : float
        Weathering flux (mol/m2/yr)

    '''

    return 1000 * C * q

# %%
def w_flux_total(C_eq, Dw, q):
    '''Return weathering flux for a given C_eq, q and Dw.
    
    Parameters
    ----------
    C_eq : float
        Equilibrium solute concentration (mol/dm3)
    q : float
        Runoff or fluid flow rate (m/yr)
    
    Returns
    -------
    w : float
        Weathering flux (mol/m2/yr)

    '''

    return 1000 * C_eq / (1/q + 1/Dw)

# %%
def Dw(C_eq,keff,L=1,t_soil=1e5,phi=0.175,sp_area=100,mol_mass=0.27,rho=2700,X_r=1):
    '''Calculate Dw for a given set of parameters.
    
    Parameters
    ----------
    C_eq : float
        Equilibrium solute concentration (mol/dm3)
    keff : float
        Effective kinetic rate coefficient (mol/m2/yr)
    L : float
        Flowpath length (m)
    t_soil : float
        Age of soils (yr)
    phi : float
        Porosity
    sp_area : float
        Specific reactive area of rock (m2/kg)
    mol_mass : float
        Mean molar mass of rock (kg/mol)
    rho : float
        Rock density (kg/m3)
    X_r : float
        Fraction of fresh minerals in fresh rock
    
    Returns
    -------
    Dw : float
        Damkoehler coefficient (m/yr)

    '''

    return L * rho * (1-phi) * X_r / ( C_eq * 1000 ) / (mol_mass * t_soil + 1 / (keff * sp_area))

# %%
def Dw_MACH(C_eq=380e-6,keff=8.7e-6,L=0.4,t_soil=1e5,phi=0.175,sp_area=100,mol_mass=0.27,rho=2700,X_r=0.36):
    '''Calculate Dw of granite using parameters listed in Maher & Chamberlain (2014).
    
    Parameters
    ----------
    C_eq : float
        Equilibrium solute concentration (mol/dm3)
    keff : float
        Effective kinetic rate coefficient (mol/m2/yr)
    L : float
        Flowpath length (m)
    t_soil : float
        Age of soils (yr)
    phi : float
        Porosity
    sp_area : float
        Specific reactive area of rock (m2/kg)
    mol_mass : float
        Mean molar mass of rock (kg/mol)
    rho : float
        Rock density (kg/m3)
    X_r : float
        Fraction of fresh minerals in fresh rock
    
    Returns
    -------
    Dw : float
        Damkoehler coefficient (m/yr)

    '''

    return L * rho * (1-phi) * X_r / ( C_eq * 1000 ) / (mol_mass * t_soil + 1 / (keff * sp_area))


# %%
def Dw_GRAN(C_eq=1555e-6,keff=7.33e-6,L=1,t_soil=1e5,phi=0.175,sp_area=100,mol_mass=0.27,rho=2700,X_r=1):
    '''Calculate Dw of granite using parameters listed in Hakim et al. (2020).
    
    Parameters
    ----------
    C_eq : float
        Equilibrium solute concentration (mol/dm3)
    keff : float
        Effective kinetic rate coefficient (mol/m2/yr)
    L : float
        Flowpath length (m)
    t_soil : float
        Age of soils (yr)
    phi : float
        Porosity
    sp_area : float
        Specific reactive area of rock (m2/kg)
    mol_mass : float
        Mean molar mass of rock (kg/mol)
    rho : float
        Rock density (kg/m3)
    X_r : float
        Fraction of fresh minerals in fresh rock
    
    Returns
    -------
    Dw : float
        Damkoehler coefficient (m/yr)

    '''

    return L * rho * (1-phi) * X_r / ( C_eq * 1000 ) / (mol_mass * t_soil + 1 / (keff * sp_area))


# %%
def get_Dw(x_CO2g, Temp, Pres, Length, tsoil, DICeqFuncs, kFuncs):
    '''Create a callable function to determine Damkohler coefficient Dw.
    
    Parameters
    ----------
    x_CO2g : float
        Partial pressure of CO2 (ppm)
    Temp : float
        Temperature of the reactions
    Pres : float
        Pressure of the reactions
    Length : float
        Flowpath length
    tsoil : float
        Age of soil or weathering zone
    DICFuncs : dict of {str : float}
        3D interpolated equilibrium DIC functions of x_CO2g, T, P
    kFuncs : dict of {str : float}
        3D interpolated kinetic rate functions of T, pH
    
    Returns
    -------
    DwFuncs : dict of {str : float}
        5D interpolated functions of x_CO2g, T, P, L, t_soil

    '''

    mine_names = ['woll', 'enst', 'ferr', 'fors', 'faya', 'anor', 'kfel', 'albi',\
                  'musc', 'phlo', 'anni', 'anth', 'grun', 'anoh', 'kfeh', 'albh',\
                  'mush', 'phlh', 'annh']
    rock_names = ['basa','peri','gran','bash','grah']
    
    comp_names = {'basa':np.array(['woll','enst','ferr','anor','albi']),\
                  'peri':np.array(['woll','enst','fors','faya']),\
                  'gran':np.array(['albi','kfel','phlo','anni','quar']),\
                  'bash':np.array(['woll','enst','ferr','anoh','albh']),\
                  'grah':np.array(['albh','kfeh','phlh','annh','quar'])}
    DwFuncs = {}

    # Points for interpolation
    points = (x_CO2g, Temp, Pres, Length, tsoil)
    x_CO2g_3, Temp_3, Pres_3 = np.meshgrid(x_CO2g, Temp, Pres, indexing='ij')
    x_CO2g, Temp, Pres, Length, tsoil = np.meshgrid(x_CO2g, Temp, Pres, Length, tsoil, indexing='ij')

    for name in mine_names:
        # Ellipsis notation is equivalent to doing :, :, : (and so on)
        
        HCO3  = DICeqFuncs[name]['HCO3'](np.transpose(np.array([x_CO2g_3, Temp_3, Pres_3]),(1,2,3,0)))
        ph   = DICeqFuncs[name]['pH'] (np.transpose(np.array([x_CO2g_3, Temp_3, Pres_3]),(1,2,3,0)))
        # k_min = kFuncs[name](np.transpose(np.array([Temp_3, ph]),(1,2,3,0)))
        k_min = kFuncs[name](Temp_3, ph)
        data = Dw(C_eq=HCO3[..., None, None], keff=k_min[..., None, None], L=Length, t_soil=tsoil)

        DwFuncs[name] = interpnd(points=points, values=data)
        
    for name in rock_names:
        # Ellipsis notation is equivalent to doing :, :, : (and so on)
        
        HCO3  = DICeqFuncs[name]['HCO3'](np.transpose(np.array([x_CO2g_3, Temp_3, Pres_3]),(1,2,3,0)))
        ph   = DICeqFuncs[name]['pH'] (np.transpose(np.array([x_CO2g_3, Temp_3, Pres_3]),(1,2,3,0)))
        k_min = 1e10 + np.zeros_like(x_CO2g_3)
        for mine in comp_names[name]:
            # k_min = np.minimum(k_min,kFuncs[mine](np.transpose(np.array([Temp_3, ph]),(1,2,3,0))))
            k_min = np.minimum(k_min, kFuncs[mine](Temp_3, ph))
        data = Dw(C_eq=HCO3[..., None, None], keff=k_min[..., None, None], L=Length, t_soil=tsoil)

        DwFuncs[name] = interpnd(points=points, values=data)
    
    return DwFuncs


# %%
def q_cont(Temp=288,epsi=0.03,S = 1360,a = 0.3,f_cont = 0.3):
    '''Calculate continental runoff using the formulations given in 
    Graham & Pierrehumbert (2020, energetic-limit included, normalized 
    to runoff of 0.3 m/yr)
    
    Parameters
    ----------
    Temp : float
        Temperature of the reactions
    epsi : float
        Temperature sensitivity
    S : float
        Stellar flux
    a : float
        Planetary albedo
    f_cont : float
        Continental area fraction
    
    Returns
    -------
    q_cont : float
        Continental runoff

    '''

    T_ref = 288
    p_ref = 0.9 # this value chosen to make q0 = 0.3 m/yr
    L = 1.918e9 * (Temp/(Temp-33.91))**2 * u.J / (u.m)**3
    p_lim = np.array([(1 - f_cont) * (S* u.J / u.s / (u.m)**2 *(1-a)/4/L).to(u.m / u.a).value])
    p_eps = np.array([p_ref * (1 + epsi * (Temp - T_ref))])
    p_act = np.minimum(p_eps,p_lim)[0]
    return f_cont * p_act

# %%
def q_contT(Temp=288,epsi=0.03,S = 1360,a = 0.3,f_cont = 0.3):
    '''Calculate continental runoff using the formulations given in 
    Graham & Pierrehumbert (2020, energetic-limit excluded, normalized 
    to runoff of 0.3 m/yr)
    
    Parameters
    ----------
    Temp : float
        Temperature of the reactions
    epsi : float
        Temperature sensitivity
    S : float
        Stellar flux
    a : float
        Planetary albedo
    f_cont : float
        Continental area fraction
    
    Returns
    -------
    q_cont : float
        Continental runoff

    '''

    T_ref = 288
    p_ref = 0.9 # this value chosen to make q0 = 0.3 m/yr
    L = 1.918e9 * (Temp/(Temp-33.91))**2 * u.J / (u.m)**3
    p_eps = p_ref * (1 + epsi * (Temp - T_ref))
    return f_cont * p_eps

# %%
def q_seaf(delT=9, H = 0.055, f_cont = 0.3):
    '''Calculate continental runoff using the formulations given in 
    Graham & Pierrehumbert (2020, energetic-limit excluded, normalized 
    to runoff of 0.3 m/yr)
    
    Parameters
    ----------
    delT : float
        Difference between pore-space and bottom sea temperatures
    H : float
        Hydrothermal heat flux
    f_cont : float
        Continental area fraction
    
    Returns
    -------
    q_cont : float
        Seafloor hydrothermal fluid flow rate

    '''

    rhow = 1000 * u.kg / (u.m)**3
    Cpw  = 4200 * u.J / u.kg / u.K
    f_cont0 = 0.3
    return ((1-f_cont)/(1-f_cont0) * H * u.J / u.s / (u.m)**2 / (rhow * Cpw * delT * u.K)).to(u.m / u.a).value
