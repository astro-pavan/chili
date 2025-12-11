# type:ignore

import numpy as np
from scipy.interpolate import RectBivariateSpline, CubicSpline, RegularGridInterpolator
from scipy.optimize import root, curve_fit, root_scalar, newton
import pandas as pd

from parameters import *
from legacy_interp2d import legacy_interp2d_wrapper
interp2d = legacy_interp2d_wrapper
from kinetics import import_kinetics_data, quar_ki, albi_ki, anor_ki, kfel_ki, fors_ki, faya_ki, enst_ki, ferr_ki, woll_ki, anth_ki, grun_ki, musc_ki, phlo_ki, anni_ki, albi_ki, anor_ki, kfel_ki, musc_ki, phlo_ki, anni_ki

# EQUILBRIUM

# import thermodynamics data

path_species = './database/species.csv'
speciesNames = pd.read_csv(path_species)

speciesDict  = {} 
logKDict     = {}
KeqFuncs     = {}

T = np.array([])
P = np.array([])

for name, colName in zip(speciesNames['species name'], speciesNames['species col name']):
    
    data = pd.read_csv('./database/' + name + '_th', usecols=['out.' + colName + '.T', 'out.' + colName + '.P', 'out.' + colName + '.logK'])
    
    if data['out.' + colName + '.logK'].isna().any():

        def func(X, a, b, c, d, e):
            x, y = X
            return a * (x ** 3) + b * (x ** 2) + c * x + d * y + e
        
        guess = (0.5, 0.5, 0.5, 0.5, 0.5)

        fitData = data.dropna()
        colParams = {}

        x = fitData['out.' + colName + '.T'].values
        y = fitData['out.' + colName + '.P'].values
        z = fitData['out.' + colName + '.logK'].values

        params = curve_fit(func, (x, y), z, guess)

        colParams['out.' + colName + '.logK'] = params[0]

        x = data[pd.isnull(data['out.' + colName + '.logK'])]['out.' + colName + '.T'].values
        y = data[pd.isnull(data['out.' + colName + '.logK'])]['out.' + colName + '.P'].values
        n = data[pd.isnull(data['out.' + colName + '.logK'])].index.astype(float).values
        
        data.loc[n, 'out.' + colName + '.logK'] = func((x, y), *colParams['out.' + colName + '.logK'])
    
    speciesDict[name] = data

    T = np.unique(data['out.' + colName + '.T'].values)
    P = np.unique(data['out.' + colName + '.P'].values)

    logKDict[name] = np.reshape(data['out.' + colName + '.logK'].values, (len(P),len(T)))

# makes intepolators for K values

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

K_H_Tb   = 10 / 55.5084 * np.exp(lnKHFunc(T)) * np.ones((len(P), len(T))) # convert MPa to bar
Keq_CO2b = 1 / K_H_Tb
KeqFuncs['co2b'] = interp2d(T, P, Keq_CO2b, kind='linear')

K_H_Tc   = 1600 / 55.5084 * np.exp(-2400*(1/T - 1/298)) * np.ones((len(P), len(T)))
Keq_CO2c = 1 / K_H_Tc
KeqFuncs['co2c'] = interp2d(T, P, Keq_CO2c, kind='linear')

# Calculate DIC and pH components for basalt dissolution

x_CO2g_range = np.logspace(-8, 0)
Temp_range = np.linspace(273.15, 372.15)
Pres_range = np.logspace(-2, 3) 

Pres, Temp, x_CO2g = np.meshgrid(Pres_range, Temp_range, x_CO2g_range)

K_woll = KeqFuncs['woll'](Temp, Pres)
K_enst = KeqFuncs['enst'](Temp, Pres)
K_ferr = KeqFuncs['ferr'](Temp, Pres)
K_anor = KeqFuncs['anoh'](Temp, Pres)
K_albi = KeqFuncs['albh'](Temp, Pres)
K_bica = KeqFuncs['bica'](Temp, Pres)
K_carb = KeqFuncs['carb'](Temp, Pres)
K_wate = KeqFuncs['wate'](Temp, Pres)
K_co2a = KeqFuncs['co2a'](Temp, Pres)

act_CO2_aq      =   x_CO2g * K_co2a
a_vals               =   2 * (K_woll ** 2) * K_carb
b_vals               =   (K_woll ** 2) * K_wate + K_woll**2 * K_bica * x_CO2g
c_vals               =   - (x_CO2g ** 2) * ((K_woll ** 2) * (K_bica ** 2) + (K_anor ** 4) * K_bica * K_albi)
d_vals               =   - (x_CO2g ** 3) * 2 * K_woll * (K_anor ** 2) * K_bica * (K_woll + K_enst + K_ferr)

# system of equations to solve for the activity of HCO3- 

def f(x, a, b, c, d):
    return x**4 + b/a * x**3 + c/a * x + d/a

def f1(x, a, b, c, d):
    return 4 * x**3 + 3 * b/a * x**2 + c/a

def f2(x, a, b, c, d):
    return 12 * x**2 + 6 * b/a * x

x0  = 10000 * (x_CO2g ** (1/2))

# sol = root_scalar(f, fprime=f1, fprime2=f2, x0=x0)
root = newton(f, x0, fprime=f1, fprime2=f2, args=(a_vals, b_vals, c_vals, d_vals))

# x0 = newton(f, x, fprime=f1, args=(), tol=1e-6, maxiter=1000, fprime2=f2)

act_HCO3_m      = root

act_CO3_mm      = K_carb / K_bica * act_HCO3_m**2 / x_CO2g

alk             = 2 * act_CO3_mm + act_HCO3_m
dic             = act_HCO3_m + act_CO2_aq + act_CO3_mm

act_H_p         = K_bica * x_CO2g  / act_HCO3_m
pH              = - np.log10(act_H_p)

act_SiO2        = K_woll / K_anor**2

act_DIV_pp      = (1 + K_enst/K_woll + K_ferr/K_woll) * K_anor**2 * (x_CO2g**2 / act_HCO3_m**2)
act_MON_p       = K_anor**4 / K_woll**2 * (x_CO2g / act_HCO3_m)

basalt_equilbrium_alkalinity_interp = RegularGridInterpolator((Pres_range, Temp_range, x_CO2g_range), alk)
basalt_equilbrium_bicarbonate_interp = RegularGridInterpolator((Pres_range, Temp_range, x_CO2g_range), act_HCO3_m)
basalt_equilbrium_carbon_interp = RegularGridInterpolator((Pres_range, Temp_range, x_CO2g_range), dic)
basalt_equilbrium_pH_interp = RegularGridInterpolator((Pres_range, Temp_range, x_CO2g_range), pH)

# KINETICS

# import kinetics data
kinetics_logKdict = import_kinetics_data()
    
kinetics_Funcs   = {}

names = np.array(['quar','albi','anor','kfel','fors','faya','enst','ferr',\
                    'woll','anth','grun','musc','phlo','anni','albh','anoh',\
                    'kfeh','mush','phlh','annh'])
f_names = np.array([quar_ki, albi_ki, anor_ki, kfel_ki, fors_ki, faya_ki, enst_ki, ferr_ki, \
                    woll_ki, anth_ki, grun_ki, musc_ki, phlo_ki, anni_ki, albi_ki, anor_ki, \
                    kfel_ki, musc_ki, phlo_ki, anni_ki])

for i, func in enumerate(f_names):
    kinetics_Funcs[names[i]] = lambda T, pH, f=func: f(T, pH, kinetics_logKdict).value
    
# TRANSPORT

def transport_concentration(P, T, xCO2, q, phi, m, A_sp, t_s):
    
    C_eq = basalt_equilbrium_bicarbonate_interp((P, T, xCO2))
    pH = basalt_equilbrium_pH_interp((P, T, xCO2))

    k_eff = np.min([
        kinetics_Funcs['woll'](T, pH),
        kinetics_Funcs['enst'](T, pH),
        kinetics_Funcs['ferr'](T, pH),
        kinetics_Funcs['anor'](T, pH),
        kinetics_Funcs['albi'](T, pH),
    ])

    Dw =  phi / (C_eq * (k_eff ** -1 + m * A_sp * t_s))

    return C_eq / (1 + (q / Dw))

print(transport_concentration(1, 280, 280e-6, 1, 1, 1, 1, 1))



