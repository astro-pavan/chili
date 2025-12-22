# -*- coding: utf-8 -*-
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
# plot_example.py (generates an example figures)
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
from matplotlib import rc
from matplotlib import rcParams
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from scipy.optimize import newton
from scipy.optimize import root
from scipy.optimize import broyden1
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.interpolate import RegularGridInterpolator as interpnd
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

rc('font', size='16')
rc('text', usetex=False)

try:
    from .parameters import *
    from . import kinetics    as ki
    from . import equilibrium as eq
    from . import transport   as tr
    from . import climate     as cl
except ImportError:
    from parameters import *
    import kinetics    as ki
    import equilibrium as eq
    import transport   as tr
    import climate     as cl

from pathlib import Path

BASE_DIR = Path(__file__).parent
DATABASE_DIR = BASE_DIR / "database"

import time
start = time.time()

# %%
# Execute CHILI
print(np.round(time.time() - start),'s, Running CHILI ...')
KeqFuncs   = eq.import_thermo_data(DATABASE_DIR / 'species.csv')
DICeqFuncs = eq.get_DICeq(xCO2, T, Pfull, KeqFuncs)
print(np.round(time.time() - start),'s, Equilibrium calculations complete.')

logkDict   = ki.import_kinetics_data()
kFuncs     = ki.get_keff(T, pHfull, logkDict)
print(np.round(time.time() - start),'s, Kinetics calculations complete.')

DwFuncs    = tr.get_Dw(xCO2, T, P, L_, t_soil_, DICeqFuncs, kFuncs)
# DwFuncs2   = tr.get_Dw(xCO2, T, P, L, t_soill, DICeqFuncs, kFuncs)
HCO3Funcs  = tr.get_C(HCO3full, Dwfull, q)
DICtrFuncs = tr.get_DICtr(HCO3full, xCO2, T, P, KeqFuncs)
print(np.round(time.time() - start),'s, Transport calculations complete.')

# %%
# Example Fig.: Plot equi. w vs xCO2 for rocks

xCO2    = np.logspace(-6,0)         #           Volume mixing ratio or activity of CO2(g) 

xCO2g0 = 280e-6
T288 = 288
argxCO2 = np.array([xCO2,T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2))]).T

HCO3eq_basa = DICeqFuncs['bash']['HCO3'](argxCO2)
HCO3eq_peri = DICeqFuncs['peri']['HCO3'](argxCO2)
HCO3eq_gran = DICeqFuncs['grah']['HCO3'](argxCO2)

w_basa_HCO3eq  = tr.w_flux(HCO3eq_basa,qc*np.ones(len(xCO2)))
w_peri_HCO3eq  = tr.w_flux(HCO3eq_peri,qc*np.ones(len(xCO2)))
w_gran_HCO3eq  = tr.w_flux(HCO3eq_gran,qc*np.ones(len(xCO2)))

basa1, basa2 = eq.fit_powerlaw(xCO2,HCO3eq_basa)
peri1, peri2 = eq.fit_powerlaw(xCO2,HCO3eq_peri)
gran1, gran2 = eq.fit_powerlaw(xCO2,HCO3eq_gran)

fig1 = plt.figure(figsize=(5,5))
spec1 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig1)

p11  = fig1.add_subplot(spec1[0,0])

l111, = p11.plot(xCO2,w_basa_HCO3eq,lw=wid_rock,c=col_basa,ls=lwi_mine)
l112, = p11.plot(xCO2,w_peri_HCO3eq,lw=wid_rock,c=col_peri,ls=lwi_mine)
l113, = p11.plot(xCO2,w_gran_HCO3eq,lw=wid_rock,c=col_gran,ls=lwi_mine)

p11.set_xscale('log')
p11.set_yscale('log')
p11.set_ylim([1e-2,1e4])
p11.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=20)
p11.set_ylabel('$w$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=20)
p11.set_title('Thermodynamic $P_{\mathrm{CO}_2}$ sensitivity',fontsize=18)
p11.set_xticks(np.logspace(-6,0,num=7))
p11.legend([l112,l111,l113], ['Peridotite','Basalt','Granite'],\
           frameon=True,prop={'size':14},loc='lower right',handlelength=2)

p11.text(0.68, 0.92,r'$\beta_{\mathrm{th}}$ = %.2f'%peri2,transform=p11.transAxes,fontsize=12,c=col_peri)
p11.text(0.68, 0.61,r'$\beta_{\mathrm{th}}$ = %.2f'%basa2,transform=p11.transAxes,fontsize=12,c=col_basa)
p11.text(0.68, 0.42,r'$\beta_{\mathrm{th}}$ = %.2f'%gran2,transform=p11.transAxes,fontsize=12,c=col_gran)

p11.text(0.02,0.9,r'$w$ = [HCO$_3^-$]$_{\mathrm{eq}}$ $q$ $\propto$ $P_{\mathrm{CO}_2}^{\beta_{\rm th}}$', \
         transform=p11.transAxes, fontsize=14)

plt.savefig('w_rock_example.pdf',format='pdf',bbox_inches='tight')

