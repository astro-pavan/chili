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
# plot_all.py (generates all figures (+ additional) from the published paper)
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

from parameters import *
import kinetics    as ki
import equilibrium as eq
import transport   as tr
import climate     as cl

import time
start = time.time()

# %%
# Execute CHILI
print(np.round(time.time() - start),'s, Running CHILI ...')
KeqFuncs   = eq.import_thermo_data('./database/species.csv')
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
# Fig. 3: Plot equilibrium HCO3- vs xCO2 for peri
fig1 = plt.figure(figsize=(6,10))
spec1 = gridspec.GridSpec(ncols=1, nrows=2, figure=fig1)

p11  = fig1.add_subplot(spec1[0,0])
p12  = fig1.add_subplot(spec1[1,0])

p11.text(-0.1, 1.15, '(a)', transform=p11.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p12.text(-0.1, 1.15, '(b)', transform=p12.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')

xCO2 = np.logspace(-8,0)
q0 = 0.3

argxCO2 = np.array([xCO2,T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2))]).T

pHeq    = DICeqFuncs['peri']['pH']  (argxCO2)
HCO3eq  = DICeqFuncs['peri']['HCO3'](argxCO2)
DICeq   = DICeqFuncs['peri']['DIC'] (argxCO2)
CO3eq   = DICeqFuncs['peri']['CO3'] (argxCO2)
CO2eq   = DICeqFuncs['peri']['CO2'] (argxCO2)
ALKeq   = DICeqFuncs['peri']['ALK'] (argxCO2)
DIVeq   = DICeqFuncs['peri']['DIV'] (argxCO2)

p11a   = p11.twinx()
l111a, = p11a.plot(xCO2,pHeq,lw=wid_mine,c=col_peri,ls=lwo_mine)

p11a.set_ylim([4,12])
p11a.set_ylabel('pH$_{\mathrm{eq}}$',c=col_peri)
p11a.tick_params(axis='y',labelcolor=col_peri)

l112, = p11.plot(xCO2,HCO3eq,lw=wid_mine,c=col_bica,ls=lwi_mine)
l113, = p11.plot(xCO2,CO3eq, lw=wid_mine,c=col_carb,ls=lwi_mine)
l111, = p11.plot(xCO2,ALKeq, lw=wid_rock,c=col_peri,ls=lwi_rock)

# print('HCO3eq/ALKeq',HCO3eq/ALKeq)
# print('HCO3eq/DIVeq',DIVeq/ALKeq)

p11.set_xscale('log')
p11.set_yscale('log')
p11.set_ylim([1e-9,5e1])
p11.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=20)
p11.set_ylabel('$C$ [mol dm$^{-3}$]',fontsize=20)
p11.set_title('Maximum Concentrations',fontsize=20)
# p11.set_xticks(np.logspace(-2,6,num=5))

p11.fill_between([1e-8,1e0],   [2e-10,2e-10],   [5e1,5e1],     color='yellow', alpha=0.2)

# p11.text(0.05, 0.3, '$T$ = 288 K', transform=p11.transAxes,fontsize=12)

l122, = p12.plot(xCO2,tr.w_flux(HCO3eq,q0*np.ones(len(xCO2))),lw=wid_mine,c=col_bica,ls=lwi_mine)
l123, = p12.plot(xCO2,tr.w_flux(CO3eq, q0*np.ones(len(xCO2))),lw=wid_mine,c=col_carb,ls=lwi_mine)
l121, = p12.plot(xCO2,tr.w_flux(ALKeq, q0*np.ones(len(xCO2))),lw=wid_rock,c=col_peri,ls=lwi_rock)

p12.set_xscale('log')
p12.set_yscale('log')
p12.set_ylim([5e-7,1e4])
p12.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=20)
p12.set_ylabel('$w$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=20)
p12.set_title('Maximum Weathering',fontsize=20)
# p12.set_xticks(np.logspace(-2,6,num=5))
p12.set_yticks(np.logspace(-6,4,num=6))

p12.fill_between([1e-8,1e0],   [5e-7,5e-7],   [1e4,1e4],     color='yellow', alpha=0.2)

fig1.legend([l121,l122,l123],\
            ['$A_{\mathrm{eq}}$','[HCO$^-_3$]$_{\mathrm{eq}}$','[CO$_3^{2-}$]$_{\mathrm{eq}}$'],\
            frameon=True,prop={'size':12},\
            bbox_to_anchor=(0.84,0.09),loc='lower right',bbox_transform=fig1.transFigure,handlelength=3)

fig1.legend([l111,l112,l113,l111a], ['$A_{\mathrm{eq}}$','[HCO$_3^-$]$_{\mathrm{eq}}$',\
            '[CO$_3^{2-}$]$_{\mathrm{eq}}$','pH$_{\mathrm{eq}}$'],\
            frameon=True,prop={'size':12},
            bbox_to_anchor=(0.84,0.58),loc='lower right',bbox_transform=fig1.transFigure,handlelength=3)

plt.savefig('CAeq_peri.pdf',format='pdf',bbox_inches='tight')

# %%
# Fig. 4: Plot generalized HCO3- vs xCO2 for peri

fig1 = plt.figure(figsize=(6,10))
spec1 = gridspec.GridSpec(ncols=1, nrows=2, figure=fig1)

p11  = fig1.add_subplot(spec1[0,0])
p12  = fig1.add_subplot(spec1[1,0])

p11.text(-0.1, 1.15, '(a)', transform=p11.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p12.text(-0.1, 1.15, '(b)', transform=p12.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')

xCO2 = np.logspace(-8,0)
q0 = 0.3

argxCO2   = np.array([xCO2,T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2))]).T

HCO3eq = DICeqFuncs['peri']['HCO3'](argxCO2)
pHeq   = DICeqFuncs['peri']['pH']  (argxCO2)

argxCO2c0 = np.array([xCO2,T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2)),\
                     Lc*np.ones(len(xCO2)),np.zeros(len(xCO2))]).T
argHCO3 = np.array([DICeqFuncs['peri']['HCO3'](argxCO2),DwFuncs['peri'](argxCO2c0),\
                    q0*np.ones(len(xCO2))]).T
argDICtr = np.array([HCO3Funcs(argHCO3), xCO2, T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2))]).T

pH   = DICtrFuncs['pH']  (argDICtr)
HCO3 = DICtrFuncs['HCO3'](argDICtr)
DIC  = DICtrFuncs['DIC'] (argDICtr)
CO3  = DICtrFuncs['CO3'] (argDICtr)
CO2  = DICtrFuncs['CO2'] (argDICtr)
ALK  = DICtrFuncs['ALK'] (argDICtr)

i = np.where((HCO3eq > 2 * HCO3) == True)[0][0]
#j = np.where((CO2 > HCO3 + CO3) == True)[0][0]

p11a   = p11.twinx()
l11a1, = p11a.plot(xCO2,pH,lw=wid_mine,c=col_peri,ls=lwo_mine)

p11a.set_ylim([4,12])
p11a.set_ylabel('pH',c=col_peri,fontsize=16)
p11a.tick_params(axis='y',labelcolor=col_peri)

l112, = p11.plot(xCO2,HCO3,lw=wid_mine,c=col_bica,ls=lwi_mine)
l113, = p11.plot(xCO2,CO3 ,lw=wid_mine,c=col_carb,ls=lwi_mine)
l111, = p11.plot(xCO2,ALK ,lw=wid_rock,c=col_peri,ls=lwi_rock)

p11.scatter(xCO2[i],HCO3[i],marker='o',s=80,color=col_bica,zorder=4)
p11.scatter(xCO2[i],ALK[i] ,marker='o',s=80,color=col_peri,zorder=4)

p11.set_xscale('log')
p11.set_yscale('log')
p11.set_ylim([1e-9,5e1])
p11.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=20)
p11.set_ylabel('$C$ [mol dm$^{-3}$]',fontsize=20)
p11.set_title('Generalized Concentrations',fontsize=20)
# p11.set_xticks(np.logspace(-2,6,num=5))

p11.fill_between([xCO2[0],xCO2[i]] , [1e-9,1e-9], [5e1,5e1], color='yellow', alpha=0.2)
p11.fill_between([xCO2[i],xCO2[-1]], [1e-9,1e-9], [5e1,5e1], color='red',    alpha=0.1)

p11.text(0.05, 0.7, 'Thermo-\ndynamic', transform=p11.transAxes)
p11.text(0.55, 0.8, 'Kinetic', transform=p11.transAxes)

l122, = p12.plot(xCO2,tr.w_flux(HCO3,q0*np.ones(len(xCO2))),lw=wid_mine,c=col_bica,ls=lwi_mine)
l123, = p12.plot(xCO2,tr.w_flux(CO3, q0*np.ones(len(xCO2))),lw=wid_mine,c=col_carb,ls=lwi_mine)
l121, = p12.plot(xCO2,tr.w_flux(ALK, q0*np.ones(len(xCO2))),lw=wid_rock,c=col_peri,ls=lwi_rock)

p12.scatter(xCO2[i],tr.w_flux(HCO3[i],q0),marker='o',s=80,color=col_bica,zorder=4)
p12.scatter(xCO2[i],tr.w_flux(ALK [i],q0),marker='o',s=80,color=col_peri,zorder=4)

p12.set_xscale('log')
p12.set_yscale('log')
p12.set_ylim([5e-7,1e4])
p12.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=20)
p12.set_ylabel('$w$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=20)
p12.set_title('Generalized Weathering',fontsize=20)
# p12.set_xticks(np.logspace(-2,6,num=5))
p12.set_yticks(np.logspace(-6,4,num=6))

p12.fill_between([xCO2[0],xCO2[i]] , [5e-7,5e-7], [1e4,1e4], color='yellow', alpha=0.2)
p12.fill_between([xCO2[i],xCO2[-1]], [5e-7,5e-7], [1e4,1e4], color='red',    alpha=0.1)

p12.text(0.05, 0.7, 'Thermo-\ndynamic', transform=p12.transAxes)
p12.text(0.55, 0.8, 'Kinetic', transform=p12.transAxes)


fig1.legend([l121,l122,l123],\
            ['$A$','[HCO$^-_3$]','[CO$_3^{2-}$]'],\
            frameon=True,prop={'size':12},\
            bbox_to_anchor=(0.6,0.09),loc='lower right',\
            bbox_transform=fig1.transFigure,handlelength=3,fontsize=12)

fig1.legend([l111,l112,l113,l11a1], ['$A$','[HCO$_3^-$]','[CO$_3^{2-}$]','pH'],
            frameon=True,prop={'size':12},bbox_to_anchor=(0.6,0.58),loc='lower right',
            bbox_transform=fig1.transFigure,handlelength=3,fontsize=12)

plt.savefig('CA_peri.pdf',format='pdf',bbox_inches='tight')

# %%
# Fig. 5: Plot equi. w vs xCO2 and T for rocks

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

T = np.linspace(273.15,372.15)
argT = np.array([xCO2g0*np.ones(len(T)),T,P0*np.ones(len(T))]).T

HCO3eqT_basa = DICeqFuncs['bash']['HCO3'](argT)
HCO3eqT_peri = DICeqFuncs['peri']['HCO3'](argT)
HCO3eqT_gran = DICeqFuncs['grah']['HCO3'](argT)

wT_basa_HCO3eq = tr.w_flux(HCO3eqT_basa,qc*np.ones(len(T)))
wT_peri_HCO3eq = tr.w_flux(HCO3eqT_peri,qc*np.ones(len(T)))
wT_gran_HCO3eq = tr.w_flux(HCO3eqT_gran,qc*np.ones(len(T)))

basaT1, basaT2 = eq.fit_powerlaw_T(T,HCO3eqT_basa)/1000
periT1, periT2 = eq.fit_powerlaw_T(T,HCO3eqT_peri)/1000
granT1, granT2 = eq.fit_powerlaw_T(T,HCO3eqT_gran)/1000

w_basa_HCO3eq0 = tr.w_flux(DICeqFuncs['bash']['HCO3']([xCO2g0,T288,P0]),qc)
w_peri_HCO3eq0 = tr.w_flux(DICeqFuncs['peri']['HCO3']([xCO2g0,T288,P0]),qc)
w_gran_HCO3eq0 = tr.w_flux(DICeqFuncs['grah']['HCO3']([xCO2g0,T288,P0]),qc)

# print(w_peri_HCO3eq0,w_basa_HCO3eq0,w_gran_HCO3eq0)

fig1 = plt.figure(figsize=(5,10))
spec1 = gridspec.GridSpec(ncols=1, nrows=2, figure=fig1)

p11  = fig1.add_subplot(spec1[0,0])
p12  = fig1.add_subplot(spec1[1,0])

p11.text(-0.1, 1.15, '(a)', transform=p11.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p12.text(-0.1, 1.15, '(b)', transform=p12.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')

l111, = p11.plot(xCO2,w_basa_HCO3eq,lw=wid_rock,c=col_basa,ls=lwi_mine)
l112, = p11.plot(xCO2,w_peri_HCO3eq,lw=wid_rock,c=col_peri,ls=lwi_mine)
l113, = p11.plot(xCO2,w_gran_HCO3eq,lw=wid_rock,c=col_gran,ls=lwi_mine)

# w_basa_eq.fit_p = w_basa_HCO3eq0 * (xCO2/xCO2g0)**basa2
# p11.plot(xCO2,w_basa_eq.fit_p,c='k')

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
# p11.text(0.05,0.82,'$q$ = 0.3 m yr$^{-1}$', transform=p11.transAxes, fontsize=14)
# p11.text(0.05,0.74,'$T$ = 288 K', transform=p11.transAxes, fontsize=14)

l121, = p12.plot(T,wT_basa_HCO3eq,lw=wid_rock,c=col_basa,ls=lwi_mine)
l122, = p12.plot(T,wT_peri_HCO3eq,lw=wid_rock,c=col_peri,ls=lwi_mine)
l123, = p12.plot(T,wT_gran_HCO3eq,lw=wid_rock,c=col_gran,ls=lwi_mine)

# w_basa_fit_T = w_basa_HCO3eq0 * np.exp(-basaT2 * (1/T - 1/T288))
# p12.plot(T,w_basa_fit_T,c='k')

p12.set_yscale('log')
p12.set_ylim([1e-2,1e4])
p12.set_xlabel('$T$ [K]',fontsize=20)
p12.set_ylabel('$w$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=20)
p12.set_title('Thermodynamic $T$ sensitivity',fontsize=18)
# p12.legend([l121,l122,l123], ['Basalt','Peridotite','Granite'],\
#            frameon=True,prop={'size':14},loc='center right',handlelength=2)

p12.text(0.03, 0.65,'$E_{\mathrm{th}}$ = %d kJ mol$^{-1}$'%periT2,transform=p12.transAxes,fontsize=12,c=col_peri,zorder=9)
p12.text(0.03, 0.38,'$E_{\mathrm{th}}$ = %d'%basaT2,transform=p12.transAxes,fontsize=12,c=col_basa,zorder=9)
p12.text(0.10, 0.34,'kJ mol$^{-1}$',transform=p12.transAxes,fontsize=12,c=col_basa,zorder=9)
p12.text(0.03, 0.13,'$E_{\mathrm{th}}$ = %d kJ mol$^{-1}$'%granT2,transform=p12.transAxes,fontsize=12,c=col_gran,zorder=9)

p12.text(0.02,0.9,r'$w$ = [HCO$_3^-$]$_{\mathrm{eq}}$ $q$ $\propto$ exp ($-E_{\mathrm{th}}/ R T$)', \
         transform=p12.transAxes, fontsize=14)
# p12.text(0.05,0.82,'$q$ = 0.3 m yr$^{-1}$', transform=p12.transAxes, fontsize=14)
# p12.text(0.05,0.74,'$x_{\mathrm{CO}_2(g)}$ = 280 xCO2v', transform=p12.transAxes, fontsize=14)

plt.savefig('w_rock_therm.pdf',format='pdf',bbox_inches='tight')

# %%
# Additional Fig.: Plot equi. w vs xCO2 and T for rocks

xCO2    = np.logspace(-6,0)         #           Volume mixing ratio or activity of CO2(g) 

xCO2g0 = 280e-6
T288 = 288
argxCO2 = np.array([xCO2,T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2))]).T

HCO3eq_basa = DICeqFuncs['basa']['HCO3'](argxCO2)
HCO3eq_peri = DICeqFuncs['peri']['HCO3'](argxCO2)
HCO3eq_gran = DICeqFuncs['gran']['HCO3'](argxCO2)

w_basa_HCO3eq  = tr.w_flux(HCO3eq_basa,qc*np.ones(len(xCO2)))
w_peri_HCO3eq  = tr.w_flux(HCO3eq_peri,qc*np.ones(len(xCO2)))
w_gran_HCO3eq  = tr.w_flux(HCO3eq_gran,qc*np.ones(len(xCO2)))

basa1, basa2 = eq.fit_powerlaw(xCO2,HCO3eq_basa)
peri1, peri2 = eq.fit_powerlaw(xCO2,HCO3eq_peri)
gran1, gran2 = eq.fit_powerlaw(xCO2,HCO3eq_gran)

T = np.linspace(273.15,372.15)
argT = np.array([xCO2g0*np.ones(len(T)),T,P0*np.ones(len(T))]).T

HCO3eqT_basa = DICeqFuncs['basa']['HCO3'](argT)
HCO3eqT_peri = DICeqFuncs['peri']['HCO3'](argT)
HCO3eqT_gran = DICeqFuncs['gran']['HCO3'](argT)

wT_basa_HCO3eq = tr.w_flux(HCO3eqT_basa,qc*np.ones(len(T)))
wT_peri_HCO3eq = tr.w_flux(HCO3eqT_peri,qc*np.ones(len(T)))
wT_gran_HCO3eq = tr.w_flux(HCO3eqT_gran,qc*np.ones(len(T)))

basaT1, basaT2 = eq.fit_powerlaw_T(T,HCO3eqT_basa)/1000
periT1, periT2 = eq.fit_powerlaw_T(T,HCO3eqT_peri)/1000
granT1, granT2 = eq.fit_powerlaw_T(T,HCO3eqT_gran)/1000

w_basa_HCO3eq0 = tr.w_flux(DICeqFuncs['basa']['HCO3']([xCO2g0,T288,P0]),qc)
w_peri_HCO3eq0 = tr.w_flux(DICeqFuncs['peri']['HCO3']([xCO2g0,T288,P0]),qc)
w_gran_HCO3eq0 = tr.w_flux(DICeqFuncs['gran']['HCO3']([xCO2g0,T288,P0]),qc)

# print(w_peri_HCO3eq0,w_basa_HCO3eq0,w_gran_HCO3eq0)

fig1 = plt.figure(figsize=(5,10))
spec1 = gridspec.GridSpec(ncols=1, nrows=2, figure=fig1)

p11  = fig1.add_subplot(spec1[0,0])
p12  = fig1.add_subplot(spec1[1,0])

p11.text(-0.1, 1.15, '(a)', transform=p11.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p12.text(-0.1, 1.15, '(b)', transform=p12.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')

l111, = p11.plot(xCO2,w_basa_HCO3eq,lw=wid_rock,c=col_basa,ls=lwi_mine)
l112, = p11.plot(xCO2,w_peri_HCO3eq,lw=wid_rock,c=col_peri,ls=lwi_mine)
l113, = p11.plot(xCO2,w_gran_HCO3eq,lw=wid_rock,c=col_gran,ls=lwi_mine)

# w_basa_eq.fit_p = w_basa_HCO3eq0 * (xCO2/xCO2g0)**basa2
# p11.plot(xCO2,w_basa_eq.fit_p,c='k')

p11.set_xscale('log')
p11.set_yscale('log')
p11.set_ylim([1e-2,1e4])
p11.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=20)
p11.set_ylabel('$w$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=20)
p11.set_title('Thermodynamic $P_{\mathrm{CO}_2}$ sensitivity',fontsize=20)
p11.set_xticks(np.logspace(-6,0,num=7))
p11.legend([l112,l111,l113], ['Peridotite','Basalt','Granite'],\
           frameon=True,prop={'size':14},loc='lower right',handlelength=2)

p11.text(0.68, 0.92,r'$\beta_{\mathrm{th}}$ = %.2f'%peri2,transform=p11.transAxes,fontsize=12,c=col_peri)
p11.text(0.68, 0.61,r'$\beta_{\mathrm{th}}$ = %.2f'%basa2,transform=p11.transAxes,fontsize=12,c=col_basa)
p11.text(0.68, 0.42,r'$\beta_{\mathrm{th}}$ = %.2f'%gran2,transform=p11.transAxes,fontsize=12,c=col_gran)

p11.text(0.02,0.9,r'$w$ = [HCO$_3^-$]$_{\mathrm{eq}}$ $q$ $\propto$ $P_{\mathrm{CO}_2}^{\beta_{\rm th}}$', \
         transform=p11.transAxes, fontsize=14)
# p11.text(0.05,0.82,'$q$ = 0.3 m yr$^{-1}$', transform=p11.transAxes, fontsize=14)
# p11.text(0.05,0.74,'$T$ = 288 K', transform=p11.transAxes, fontsize=14)

l121, = p12.plot(T,wT_basa_HCO3eq,lw=wid_rock,c=col_basa,ls=lwi_mine)
l122, = p12.plot(T,wT_peri_HCO3eq,lw=wid_rock,c=col_peri,ls=lwi_mine)
l123, = p12.plot(T,wT_gran_HCO3eq,lw=wid_rock,c=col_gran,ls=lwi_mine)

# w_basa_fit_T = w_basa_HCO3eq0 * np.exp(-basaT2 * (1/T - 1/T288))
# p12.plot(T,w_basa_fit_T,c='k')

p12.set_yscale('log')
p12.set_ylim([1e-2,1e4])
p12.set_xlabel('$T$ [K]',fontsize=20)
p12.set_ylabel('$w$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=20)
p12.set_title('Thermodynamic $T$ sensitivity',fontsize=20)
# p12.legend([l121,l122,l123], ['Basalt','Peridotite','Granite'],\
#            frameon=True,prop={'size':14},loc='center right',handlelength=2)

p12.text(0.05, 0.65,r'$E_{\mathrm{th}}$ = %d kJ mol$^{-1}$'%periT2,transform=p12.transAxes,fontsize=12,c=col_peri,zorder=9)
p12.text(0.05, 0.34,r'$E_{\mathrm{th}}$ = %d kJ mol$^{-1}$'%basaT2,transform=p12.transAxes,fontsize=12,c=col_basa,zorder=9)
p12.text(0.05, 0.13,r'$E_{\mathrm{th}}$ = %d kJ mol$^{-1}$'%granT2,transform=p12.transAxes,fontsize=12,c=col_gran,zorder=9)

p12.text(0.02,0.9,r'$w$ = [HCO$_3^-$]$_{\mathrm{eq}}$ $q$ $\propto$ exp ($-E_{\mathrm{th}}/ R T$)', \
         transform=p12.transAxes, fontsize=14)
# p12.text(0.05,0.82,'$q$ = 0.3 m yr$^{-1}$', transform=p12.transAxes, fontsize=14)
# p12.text(0.05,0.74,'$x_{\mathrm{CO}_2(g)}$ = 280 xCO2v', transform=p12.transAxes, fontsize=14)

plt.savefig('w_rock_therm_k.pdf',format='pdf',bbox_inches='tight')

# %%
# Additional Fig.: Plot equi. w vs xCO2 and T for rocks

xCO2    = np.logspace(-6,0)         #           Volume mixing ratio or activity of CO2(g) 

xCO2g0 = 280e-6
T288 = 288
argxCO2 = np.array([xCO2,T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2))]).T

HCO3eq_anor = DICeqFuncs['anoh']['HCO3'](argxCO2)
HCO3eq_albi = DICeqFuncs['albh']['HCO3'](argxCO2)
HCO3eq_kfel = DICeqFuncs['kfeh']['HCO3'](argxCO2)

w_anor_HCO3eq  = tr.w_flux(HCO3eq_anor,qc*np.ones(len(xCO2)))
w_albi_HCO3eq  = tr.w_flux(HCO3eq_albi,qc*np.ones(len(xCO2)))
w_kfel_HCO3eq  = tr.w_flux(HCO3eq_kfel,qc*np.ones(len(xCO2)))

anor1, anor2 = eq.fit_powerlaw(xCO2,HCO3eq_anor)
albi1, albi2 = eq.fit_powerlaw(xCO2,HCO3eq_albi)
kfel1, kfel2 = eq.fit_powerlaw(xCO2,HCO3eq_kfel)

T = np.linspace(273.15,372.15)
argT = np.array([xCO2g0*np.ones(len(T)),T,P0*np.ones(len(T))]).T

HCO3eqT_anor = DICeqFuncs['anoh']['HCO3'](argT)
HCO3eqT_albi = DICeqFuncs['albh']['HCO3'](argT)
HCO3eqT_kfel = DICeqFuncs['kfeh']['HCO3'](argT)

wT_anor_HCO3eq = tr.w_flux(HCO3eqT_anor,qc*np.ones(len(T)))
wT_albi_HCO3eq = tr.w_flux(HCO3eqT_albi,qc*np.ones(len(T)))
wT_kfel_HCO3eq = tr.w_flux(HCO3eqT_kfel,qc*np.ones(len(T)))

anorT1, anorT2 = eq.fit_powerlaw_T(T,HCO3eqT_anor)/1000
albiT1, albiT2 = eq.fit_powerlaw_T(T,HCO3eqT_albi)/1000
kfelT1, kfelT2 = eq.fit_powerlaw_T(T,HCO3eqT_kfel)/1000

w_anor_HCO3eq0 = tr.w_flux(DICeqFuncs['anoh']['HCO3']([xCO2g0,T288,P0]),qc)
w_albi_HCO3eq0 = tr.w_flux(DICeqFuncs['albh']['HCO3']([xCO2g0,T288,P0]),qc)
w_kfel_HCO3eq0 = tr.w_flux(DICeqFuncs['kfeh']['HCO3']([xCO2g0,T288,P0]),qc)

# print(w_albi_HCO3eq0,w_anor_HCO3eq0,w_kfel_HCO3eq0)

fig1 = plt.figure(figsize=(5,10))
spec1 = gridspec.GridSpec(ncols=1, nrows=2, figure=fig1)

p11  = fig1.add_subplot(spec1[0,0])
p12  = fig1.add_subplot(spec1[1,0])

p11.text(-0.1, 1.15, '(a)', transform=p11.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p12.text(-0.1, 1.15, '(b)', transform=p12.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')

l111, = p11.plot(xCO2,w_anor_HCO3eq,lw=wid_rock,c=col_anor,ls=lwi_mine)
l112, = p11.plot(xCO2,w_albi_HCO3eq,lw=wid_rock,c=col_albi,ls=lwi_mine)
l113, = p11.plot(xCO2,w_kfel_HCO3eq,lw=wid_rock,c=col_kfel,ls=lwi_mine)

# w_anor_eq.fit_p = w_anor_HCO3eq0 * (xCO2/xCO2g0)**anor2
# p11.plot(xCO2,w_anor_eq.fit_p,c='k')

p11.set_xscale('log')
p11.set_yscale('log')
p11.set_ylim([1e-4,1e4])
p11.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=20)
p11.set_ylabel('$w$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=20)
p11.set_title('Thermodynamic $P_{\mathrm{CO}_2}$ sensitivity',fontsize=20)
p11.set_xticks(np.logspace(-6,0,num=7))
p11.legend([l111,l112,l113], ['Anorthite','Albite','K-feldspar'],\
           frameon=True,prop={'size':14},loc='lower right',handlelength=2)

p11.text(0.68, 0.92,r'$\beta_{\mathrm{th}}$ = %.2f'%anor2,transform=p11.transAxes,fontsize=12,c=col_anor)
p11.text(0.68, 0.61,r'$\beta_{\mathrm{th}}$ = %.2f'%albi2,transform=p11.transAxes,fontsize=12,c=col_albi)
p11.text(0.68, 0.42,r'$\beta_{\mathrm{th}}$ = %.2f'%kfel2,transform=p11.transAxes,fontsize=12,c=col_kfel)

p11.text(0.02,0.9,r'$w$ = [HCO$_3^-$]$_{\mathrm{eq}}$ $q$ $\propto$ $P_{\mathrm{CO}_2}^{\beta_{\rm th}}$', \
         transform=p11.transAxes, fontsize=14)
# p11.text(0.05,0.82,'$q$ = 0.3 m yr$^{-1}$', transform=p11.transAxes, fontsize=14)
# p11.text(0.05,0.74,'$T$ = 288 K', transform=p11.transAxes, fontsize=14)

l121, = p12.plot(T,wT_anor_HCO3eq,lw=wid_rock,c=col_anor,ls=lwi_mine)
l122, = p12.plot(T,wT_albi_HCO3eq,lw=wid_rock,c=col_albi,ls=lwi_mine)
l123, = p12.plot(T,wT_kfel_HCO3eq,lw=wid_rock,c=col_kfel,ls=lwi_mine)

# w_anor_fit_T = w_anor_HCO3eq0 * np.exp(-anorT2 * (1/T - 1/T288))
# p12.plot(T,w_anor_fit_T,c='k')

p12.set_yscale('log')
p12.set_ylim([1e-4,1e4])
p12.set_xlabel('$T$ [K]',fontsize=20)
p12.set_ylabel('$w$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=20)
p12.set_title('Thermodynamic $T$ sensitivity',fontsize=20)
# p12.legend([l121,l122,l123], ['anorlt','albidotite','kfelite'],\
#            frameon=True,prop={'size':14},loc='center right',handlelength=2)

p12.text(0.05, 0.65,r'$E_{\mathrm{th}}$ = %d kJ mol$^{-1}$'%anorT2,transform=p12.transAxes,fontsize=12,c=col_anor,zorder=9)
p12.text(0.05, 0.34,r'$E_{\mathrm{th}}$ = %d kJ mol$^{-1}$'%albiT2,transform=p12.transAxes,fontsize=12,c=col_albi,zorder=9)
p12.text(0.05, 0.13,r'$E_{\mathrm{th}}$ = %d kJ mol$^{-1}$'%kfelT2,transform=p12.transAxes,fontsize=12,c=col_kfel,zorder=9)

p12.text(0.02,0.9,r'$w$ = [HCO$_3^-$]$_{\mathrm{eq}}$ $q$ $\propto$ exp ($-E_{\mathrm{th}}/ R T$)', \
         transform=p12.transAxes, fontsize=14)
# p12.text(0.05,0.82,'$q$ = 0.3 m yr$^{-1}$', transform=p12.transAxes, fontsize=14)
# p12.text(0.05,0.74,'$x_{\mathrm{CO}_2(g)}$ = 280 xCO2v', transform=p12.transAxes, fontsize=14)

plt.savefig('w_feld_therm.pdf',format='pdf',bbox_inches='tight')

# %%
# Additional Fig.: Plot equi. w vs xCO2 and T for rocks

xCO2    = np.logspace(-6,0)         #           Volume mixing ratio or activity of CO2(g) 

xCO2g0 = 280e-6
T288 = 288
argxCO2 = np.array([xCO2,T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2))]).T

HCO3eq_anor = DICeqFuncs['anor']['HCO3'](argxCO2)
HCO3eq_albi = DICeqFuncs['albi']['HCO3'](argxCO2)
HCO3eq_kfel = DICeqFuncs['kfel']['HCO3'](argxCO2)

w_anor_HCO3eq  = tr.w_flux(HCO3eq_anor,qc*np.ones(len(xCO2)))
w_albi_HCO3eq  = tr.w_flux(HCO3eq_albi,qc*np.ones(len(xCO2)))
w_kfel_HCO3eq  = tr.w_flux(HCO3eq_kfel,qc*np.ones(len(xCO2)))

anor1, anor2 = eq.fit_powerlaw(xCO2,HCO3eq_anor)
albi1, albi2 = eq.fit_powerlaw(xCO2,HCO3eq_albi)
kfel1, kfel2 = eq.fit_powerlaw(xCO2,HCO3eq_kfel)

T = np.linspace(273.15,372.15)
argT = np.array([xCO2g0*np.ones(len(T)),T,P0*np.ones(len(T))]).T

HCO3eqT_anor = DICeqFuncs['anor']['HCO3'](argT)
HCO3eqT_albi = DICeqFuncs['albi']['HCO3'](argT)
HCO3eqT_kfel = DICeqFuncs['kfel']['HCO3'](argT)

wT_anor_HCO3eq = tr.w_flux(HCO3eqT_anor,qc*np.ones(len(T)))
wT_albi_HCO3eq = tr.w_flux(HCO3eqT_albi,qc*np.ones(len(T)))
wT_kfel_HCO3eq = tr.w_flux(HCO3eqT_kfel,qc*np.ones(len(T)))

anorT1, anorT2 = eq.fit_powerlaw_T(T,HCO3eqT_anor)/1000
albiT1, albiT2 = eq.fit_powerlaw_T(T,HCO3eqT_albi)/1000
kfelT1, kfelT2 = eq.fit_powerlaw_T(T,HCO3eqT_kfel)/1000

w_anor_HCO3eq0 = tr.w_flux(DICeqFuncs['anor']['HCO3']([xCO2g0,T288,P0]),qc)
w_albi_HCO3eq0 = tr.w_flux(DICeqFuncs['albi']['HCO3']([xCO2g0,T288,P0]),qc)
w_kfel_HCO3eq0 = tr.w_flux(DICeqFuncs['kfel']['HCO3']([xCO2g0,T288,P0]),qc)

# print(w_albi_HCO3eq0,w_anor_HCO3eq0,w_kfel_HCO3eq0)

fig1 = plt.figure(figsize=(5,10))
spec1 = gridspec.GridSpec(ncols=1, nrows=2, figure=fig1)

p11  = fig1.add_subplot(spec1[0,0])
p12  = fig1.add_subplot(spec1[1,0])

p11.text(-0.1, 1.15, '(a)', transform=p11.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p12.text(-0.1, 1.15, '(b)', transform=p12.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')

l111, = p11.plot(xCO2,w_anor_HCO3eq,lw=wid_rock,c=col_anor,ls=lwi_mine)
l112, = p11.plot(xCO2,w_albi_HCO3eq,lw=wid_rock,c=col_albi,ls=lwi_mine)
l113, = p11.plot(xCO2,w_kfel_HCO3eq,lw=wid_rock,c=col_kfel,ls=lwi_mine)

# w_anor_eq.fit_p = w_anor_HCO3eq0 * (xCO2/xCO2g0)**anor2
# p11.plot(xCO2,w_anor_eq.fit_p,c='k')

p11.set_xscale('log')
p11.set_yscale('log')
p11.set_ylim([1e-4,1e4])
p11.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=20)
p11.set_ylabel('$w$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=20)
p11.set_title('Thermodynamic $P_{\mathrm{CO}_2}$ sensitivity',fontsize=20)
p11.set_xticks(np.logspace(-6,0,num=7))
p11.legend([l112,l111,l113], ['Albite','Anorthite','K-feldspar'],\
           frameon=True,prop={'size':14},loc='lower right',handlelength=2)

p11.text(0.68, 0.92,r'$\beta_{\mathrm{th}}$ = %.2f'%anor2,transform=p11.transAxes,fontsize=12,c=col_anor)
p11.text(0.68, 0.61,r'$\beta_{\mathrm{th}}$ = %.2f'%albi2,transform=p11.transAxes,fontsize=12,c=col_albi)
p11.text(0.68, 0.42,r'$\beta_{\mathrm{th}}$ = %.2f'%kfel2,transform=p11.transAxes,fontsize=12,c=col_kfel)

p11.text(0.02,0.9,r'$w$ = [HCO$_3^-$]$_{\mathrm{eq}}$ $q$ $\propto$ $P_{\mathrm{CO}_2}^{\beta_{\rm th}}$', \
         transform=p11.transAxes, fontsize=14)
# p11.text(0.05,0.82,'$q$ = 0.3 m yr$^{-1}$', transform=p11.transAxes, fontsize=14)
# p11.text(0.05,0.74,'$T$ = 288 K', transform=p11.transAxes, fontsize=14)

l121, = p12.plot(T,wT_anor_HCO3eq,lw=wid_rock,c=col_anor,ls=lwi_mine)
l122, = p12.plot(T,wT_albi_HCO3eq,lw=wid_rock,c=col_albi,ls=lwi_mine)
l123, = p12.plot(T,wT_kfel_HCO3eq,lw=wid_rock,c=col_kfel,ls=lwi_mine)

# w_anor_fit_T = w_anor_HCO3eq0 * np.exp(-anorT2 * (1/T - 1/T288))
# p12.plot(T,w_anor_fit_T,c='k')

p12.set_yscale('log')
p12.set_ylim([1e-4,1e4])
p12.set_xlabel('$T$ [K]',fontsize=20)
p12.set_ylabel('$w$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=20)
p12.set_title('Thermodynamic $T$ sensitivity',fontsize=20)
# p12.legend([l121,l122,l123], ['anorlt','albidotite','kfelite'],\
#            frameon=True,prop={'size':14},loc='center right',handlelength=2)

p12.text(0.05, 0.65,r'$E_{\mathrm{th}}$ = %d kJ mol$^{-1}$'%anorT2,transform=p12.transAxes,fontsize=12,c=col_anor,zorder=9)
p12.text(0.05, 0.34,r'$E_{\mathrm{th}}$ = %.2f kJ mol$^{-1}$'%albiT2,transform=p12.transAxes,fontsize=12,c=col_albi,zorder=9)
p12.text(0.05, 0.13,r'$E_{\mathrm{th}}$ = %.1f kJ mol$^{-1}$'%kfelT2,transform=p12.transAxes,fontsize=12,c=col_kfel,zorder=9)

p12.text(0.02,0.9,r'$w$ = [HCO$_3^-$]$_{\mathrm{eq}}$ $q$ $\propto$ exp ($-E_{\mathrm{th}}/ R T$)', \
         transform=p12.transAxes, fontsize=14)
# p12.text(0.05,0.82,'$q$ = 0.3 m yr$^{-1}$', transform=p12.transAxes, fontsize=14)
# p12.text(0.05,0.74,'$x_{\mathrm{CO}_2(g)}$ = 280 xCO2v', transform=p12.transAxes, fontsize=14)

plt.savefig('w_feld_therm_k.pdf',format='pdf',bbox_inches='tight')

# %%
# Additional Fig.: Plot equi. w vs xCO2 and T for rocks

xCO2    = np.logspace(-6,0)         #           Volume mixing ratio or activity of CO2(g) 

xCO2g0 = 280e-6
T288 = 288
argxCO2 = np.array([xCO2,T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2))]).T

HCO3eq_musc = DICeqFuncs['mush']['HCO3'](argxCO2)
HCO3eq_phlo = DICeqFuncs['phlh']['HCO3'](argxCO2)
HCO3eq_anni = DICeqFuncs['annh']['HCO3'](argxCO2)

w_musc_HCO3eq  = tr.w_flux(HCO3eq_musc,qc*np.ones(len(xCO2)))
w_phlo_HCO3eq  = tr.w_flux(HCO3eq_phlo,qc*np.ones(len(xCO2)))
w_anni_HCO3eq  = tr.w_flux(HCO3eq_anni,qc*np.ones(len(xCO2)))

musc1, musc2 = eq.fit_powerlaw(xCO2,HCO3eq_musc)
phlo1, phlo2 = eq.fit_powerlaw(xCO2,HCO3eq_phlo)
anni1, anni2 = eq.fit_powerlaw(xCO2,HCO3eq_anni)

T = np.linspace(273.15,372.15)
argT = np.array([xCO2g0*np.ones(len(T)),T,P0*np.ones(len(T))]).T

HCO3eqT_musc = DICeqFuncs['mush']['HCO3'](argT)
HCO3eqT_phlo = DICeqFuncs['phlh']['HCO3'](argT)
HCO3eqT_anni = DICeqFuncs['annh']['HCO3'](argT)

wT_musc_HCO3eq = tr.w_flux(HCO3eqT_musc,qc*np.ones(len(T)))
wT_phlo_HCO3eq = tr.w_flux(HCO3eqT_phlo,qc*np.ones(len(T)))
wT_anni_HCO3eq = tr.w_flux(HCO3eqT_anni,qc*np.ones(len(T)))

muscT1, muscT2 = eq.fit_powerlaw_T(T,HCO3eqT_musc)/1000
phloT1, phloT2 = eq.fit_powerlaw_T(T,HCO3eqT_phlo)/1000
anniT1, anniT2 = eq.fit_powerlaw_T(T,HCO3eqT_anni)/1000

w_musc_HCO3eq0 = tr.w_flux(DICeqFuncs['mush']['HCO3']([xCO2g0,T288,P0]),qc)
w_phlo_HCO3eq0 = tr.w_flux(DICeqFuncs['phlh']['HCO3']([xCO2g0,T288,P0]),qc)
w_anni_HCO3eq0 = tr.w_flux(DICeqFuncs['annh']['HCO3']([xCO2g0,T288,P0]),qc)

# print(w_phlo_HCO3eq0,w_musc_HCO3eq0,w_anni_HCO3eq0)

fig1 = plt.figure(figsize=(5,10))
spec1 = gridspec.GridSpec(ncols=1, nrows=2, figure=fig1)

p11  = fig1.add_subplot(spec1[0,0])
p12  = fig1.add_subplot(spec1[1,0])

p11.text(-0.1, 1.15, '(a)', transform=p11.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p12.text(-0.1, 1.15, '(b)', transform=p12.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')

l111, = p11.plot(xCO2,w_musc_HCO3eq,lw=wid_rock,c=col_musc,ls=lwi_mine)
l112, = p11.plot(xCO2,w_phlo_HCO3eq,lw=wid_rock,c=col_phlo,ls=lwi_mine)
l113, = p11.plot(xCO2,w_anni_HCO3eq,lw=wid_rock,c=col_anni,ls=lwi_mine)

# w_musc_eq.fit_p = w_musc_HCO3eq0 * (xCO2/xCO2g0)**musc2
# p11.plot(xCO2,w_musc_eq.fit_p,c='k')

p11.set_xscale('log')
p11.set_yscale('log')
p11.set_ylim([1e-4,1e4])
p11.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=20)
p11.set_ylabel('$w$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=20)
p11.set_title('Thermodynamic $P_{\mathrm{CO}_2}$ sensitivity',fontsize=20)
p11.set_xticks(np.logspace(-6,0,num=7))
p11.legend([l111,l112,l113], ['Muscovite','Phlogopite','Annite'],\
           frameon=True,prop={'size':14},loc='lower right',handlelength=2)

p11.text(0.68, 0.92,r'$\beta_{\mathrm{th}}$ = %.2f'%musc2,transform=p11.transAxes,fontsize=12,c=col_musc)
p11.text(0.68, 0.61,r'$\beta_{\mathrm{th}}$ = %.2f'%phlo2,transform=p11.transAxes,fontsize=12,c=col_phlo)
p11.text(0.68, 0.42,r'$\beta_{\mathrm{th}}$ = %.2f'%anni2,transform=p11.transAxes,fontsize=12,c=col_anni)

p11.text(0.02,0.9,r'$w$ = [HCO$_3^-$]$_{\mathrm{eq}}$ $q$ $\propto$ $P_{\mathrm{CO}_2}^{\beta_{\rm th}}$', \
         transform=p11.transAxes, fontsize=14)
# p11.text(0.05,0.82,'$q$ = 0.3 m yr$^{-1}$', transform=p11.transAxes, fontsize=14)
# p11.text(0.05,0.74,'$T$ = 288 K', transform=p11.transAxes, fontsize=14)

l121, = p12.plot(T,wT_musc_HCO3eq,lw=wid_rock,c=col_musc,ls=lwi_mine)
l122, = p12.plot(T,wT_phlo_HCO3eq,lw=wid_rock,c=col_phlo,ls=lwi_mine)
l123, = p12.plot(T,wT_anni_HCO3eq,lw=wid_rock,c=col_anni,ls=lwi_mine)

# w_musc_fit_T = w_musc_HCO3eq0 * np.exp(-muscT2 * (1/T - 1/T288))
# p12.plot(T,w_musc_fit_T,c='k')

p12.set_yscale('log')
p12.set_ylim([1e-4,1e4])
p12.set_xlabel('$T$ [K]',fontsize=20)
p12.set_ylabel('$w$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=20)
p12.set_title('Thermodynamic $T$ sensitivity',fontsize=20)
# p12.legend([l121,l122,l123], ['musclt','phlodotite','anniite'],\
#            frameon=True,prop={'size':14},loc='center right',handlelength=2)

p12.text(0.05, 0.65,r'$E_{\mathrm{th}}$ = %d kJ mol$^{-1}$'%muscT2,transform=p12.transAxes,fontsize=12,c=col_musc,zorder=9)
p12.text(0.05, 0.34,r'$E_{\mathrm{th}}$ = %d kJ mol$^{-1}$'%phloT2,transform=p12.transAxes,fontsize=12,c=col_phlo,zorder=9)
p12.text(0.05, 0.13,r'$E_{\mathrm{th}}$ = %d kJ mol$^{-1}$'%anniT2,transform=p12.transAxes,fontsize=12,c=col_anni,zorder=9)

p12.text(0.02,0.9,r'$w$ = [HCO$_3^-$]$_{\mathrm{eq}}$ $q$ $\propto$ exp ($-E_{\mathrm{th}}/ R T$)', \
         transform=p12.transAxes, fontsize=14)
# p12.text(0.05,0.82,'$q$ = 0.3 m yr$^{-1}$', transform=p12.transAxes, fontsize=14)
# p12.text(0.05,0.74,'$x_{\mathrm{CO}_2(g)}$ = 280 xCO2v', transform=p12.transAxes, fontsize=14)

plt.savefig('w_mica_therm.pdf',format='pdf',bbox_inches='tight')

# %%
# Additional Fig.: Plot equi. w vs xCO2 and T for rocks

xCO2    = np.logspace(-6,0)         #           Volume mixing ratio or activity of CO2(g) 

xCO2g0 = 280e-6
T288 = 288
argxCO2 = np.array([xCO2,T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2))]).T

HCO3eq_musc = DICeqFuncs['musc']['HCO3'](argxCO2)
HCO3eq_phlo = DICeqFuncs['phlo']['HCO3'](argxCO2)
HCO3eq_anni = DICeqFuncs['anni']['HCO3'](argxCO2)

w_musc_HCO3eq  = tr.w_flux(HCO3eq_musc,qc*np.ones(len(xCO2)))
w_phlo_HCO3eq  = tr.w_flux(HCO3eq_phlo,qc*np.ones(len(xCO2)))
w_anni_HCO3eq  = tr.w_flux(HCO3eq_anni,qc*np.ones(len(xCO2)))

musc1, musc2 = eq.fit_powerlaw(xCO2,HCO3eq_musc)
phlo1, phlo2 = eq.fit_powerlaw(xCO2,HCO3eq_phlo)
anni1, anni2 = eq.fit_powerlaw(xCO2,HCO3eq_anni)

T = np.linspace(273.15,372.15)
argT = np.array([xCO2g0*np.ones(len(T)),T,P0*np.ones(len(T))]).T

HCO3eqT_musc = DICeqFuncs['musc']['HCO3'](argT)
HCO3eqT_phlo = DICeqFuncs['phlo']['HCO3'](argT)
HCO3eqT_anni = DICeqFuncs['anni']['HCO3'](argT)

wT_musc_HCO3eq = tr.w_flux(HCO3eqT_musc,qc*np.ones(len(T)))
wT_phlo_HCO3eq = tr.w_flux(HCO3eqT_phlo,qc*np.ones(len(T)))
wT_anni_HCO3eq = tr.w_flux(HCO3eqT_anni,qc*np.ones(len(T)))

muscT1, muscT2 = eq.fit_powerlaw_T(T,HCO3eqT_musc)/1000
phloT1, phloT2 = eq.fit_powerlaw_T(T,HCO3eqT_phlo)/1000
anniT1, anniT2 = eq.fit_powerlaw_T(T,HCO3eqT_anni)/1000

w_musc_HCO3eq0 = tr.w_flux(DICeqFuncs['musc']['HCO3']([xCO2g0,T288,P0]),qc)
w_phlo_HCO3eq0 = tr.w_flux(DICeqFuncs['phlo']['HCO3']([xCO2g0,T288,P0]),qc)
w_anni_HCO3eq0 = tr.w_flux(DICeqFuncs['anni']['HCO3']([xCO2g0,T288,P0]),qc)

# print(w_phlo_HCO3eq0,w_musc_HCO3eq0,w_anni_HCO3eq0)

fig1 = plt.figure(figsize=(5,10))
spec1 = gridspec.GridSpec(ncols=1, nrows=2, figure=fig1)

p11  = fig1.add_subplot(spec1[0,0])
p12  = fig1.add_subplot(spec1[1,0])

p11.text(-0.1, 1.15, '(a)', transform=p11.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p12.text(-0.1, 1.15, '(b)', transform=p12.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')

l111, = p11.plot(xCO2,w_musc_HCO3eq,lw=wid_rock,c=col_musc,ls=lwi_mine)
l112, = p11.plot(xCO2,w_phlo_HCO3eq,lw=wid_rock,c=col_phlo,ls=lwi_mine)
l113, = p11.plot(xCO2,w_anni_HCO3eq,lw=wid_rock,c=col_anni,ls=lwi_mine)

# w_musc_eq.fit_p = w_musc_HCO3eq0 * (xCO2/xCO2g0)**musc2
# p11.plot(xCO2,w_musc_eq.fit_p,c='k')

p11.set_xscale('log')
p11.set_yscale('log')
p11.set_ylim([1e-4,1e4])
p11.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=20)
p11.set_ylabel('$w$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=20)
p11.set_title('Thermodynamic $P_{\mathrm{CO}_2}$ sensitivity',fontsize=20)
p11.set_xticks(np.logspace(-6,0,num=7))
p11.legend([l111,l112,l113], ['Muscovite','Phlogopite','Annite'],\
           frameon=True,prop={'size':14},loc='lower right',handlelength=2)

p11.text(0.68, 0.92,r'$\beta_{\mathrm{th}}$ = %.2f'%musc2,transform=p11.transAxes,fontsize=12,c=col_musc)
p11.text(0.68, 0.61,r'$\beta_{\mathrm{th}}$ = %.2f'%phlo2,transform=p11.transAxes,fontsize=12,c=col_phlo)
p11.text(0.68, 0.42,r'$\beta_{\mathrm{th}}$ = %.2f'%anni2,transform=p11.transAxes,fontsize=12,c=col_anni)

p11.text(0.02,0.9,r'$w$ = [HCO$_3^-$]$_{\mathrm{eq}}$ $q$ $\propto$ $P_{\mathrm{CO}_2}^{\beta_{\rm th}}$', \
         transform=p11.transAxes, fontsize=14)
# p11.text(0.05,0.82,'$q$ = 0.3 m yr$^{-1}$', transform=p11.transAxes, fontsize=14)
# p11.text(0.05,0.74,'$T$ = 288 K', transform=p11.transAxes, fontsize=14)

l121, = p12.plot(T,wT_musc_HCO3eq,lw=wid_rock,c=col_musc,ls=lwi_mine)
l122, = p12.plot(T,wT_phlo_HCO3eq,lw=wid_rock,c=col_phlo,ls=lwi_mine)
l123, = p12.plot(T,wT_anni_HCO3eq,lw=wid_rock,c=col_anni,ls=lwi_mine)

# w_musc_fit_T = w_musc_HCO3eq0 * np.exp(-muscT2 * (1/T - 1/T288))
# p12.plot(T,w_musc_fit_T,c='k')

p12.set_yscale('log')
p12.set_ylim([1e-4,1e4])
p12.set_xlabel('$T$ [K]',fontsize=20)
p12.set_ylabel('$w$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=20)
p12.set_title('Thermodynamic $T$ sensitivity',fontsize=20)
# p12.legend([l121,l122,l123], ['musclt','phlodotite','anniite'],\
#            frameon=True,prop={'size':14},loc='center right',handlelength=2)

p12.text(0.05, 0.65,r'$E_{\mathrm{th}}$ = %d kJ mol$^{-1}$'%muscT2,transform=p12.transAxes,fontsize=12,c=col_musc,zorder=9)
p12.text(0.05, 0.34,r'$E_{\mathrm{th}}$ = %d kJ mol$^{-1}$'%phloT2,transform=p12.transAxes,fontsize=12,c=col_phlo,zorder=9)
p12.text(0.05, 0.13,r'$E_{\mathrm{th}}$ = %d kJ mol$^{-1}$'%anniT2,transform=p12.transAxes,fontsize=12,c=col_anni,zorder=9)

p12.text(0.02,0.9,r'$w$ = [HCO$_3^-$]$_{\mathrm{eq}}$ $q$ $\propto$ exp ($-E_{\mathrm{th}}/ R T$)', \
         transform=p12.transAxes, fontsize=14)
# p12.text(0.05,0.82,'$q$ = 0.3 m yr$^{-1}$', transform=p12.transAxes, fontsize=14)
# p12.text(0.05,0.74,'$x_{\mathrm{CO}_2(g)}$ = 280 xCO2v', transform=p12.transAxes, fontsize=14)

plt.savefig('w_mica_therm_k.pdf',format='pdf',bbox_inches='tight')

# %%
# Additional Fig.: Plot equi. w vs xCO2 and T for rocks

xCO2    = np.logspace(-6,0)         #           Volume mixing ratio or activity of CO2(g) 

xCO2g0 = 280e-6
T288 = 288
argxCO2 = np.array([xCO2,T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2))]).T

HCO3eq_woll = DICeqFuncs['woll']['HCO3'](argxCO2)
HCO3eq_enst = DICeqFuncs['enst']['HCO3'](argxCO2)
HCO3eq_ferr = DICeqFuncs['ferr']['HCO3'](argxCO2)

w_woll_HCO3eq  = tr.w_flux(HCO3eq_woll,qc*np.ones(len(xCO2)))
w_enst_HCO3eq  = tr.w_flux(HCO3eq_enst,qc*np.ones(len(xCO2)))
w_ferr_HCO3eq  = tr.w_flux(HCO3eq_ferr,qc*np.ones(len(xCO2)))

woll1, woll2 = eq.fit_powerlaw(xCO2,HCO3eq_woll)
enst1, enst2 = eq.fit_powerlaw(xCO2,HCO3eq_enst)
ferr1, ferr2 = eq.fit_powerlaw(xCO2,HCO3eq_ferr)

T = np.linspace(273.15,372.15)
argT = np.array([xCO2g0*np.ones(len(T)),T,P0*np.ones(len(T))]).T

HCO3eqT_woll = DICeqFuncs['woll']['HCO3'](argT)
HCO3eqT_enst = DICeqFuncs['enst']['HCO3'](argT)
HCO3eqT_ferr = DICeqFuncs['ferr']['HCO3'](argT)

wT_woll_HCO3eq = tr.w_flux(HCO3eqT_woll,qc*np.ones(len(T)))
wT_enst_HCO3eq = tr.w_flux(HCO3eqT_enst,qc*np.ones(len(T)))
wT_ferr_HCO3eq = tr.w_flux(HCO3eqT_ferr,qc*np.ones(len(T)))

wollT1, wollT2 = eq.fit_powerlaw_T(T,HCO3eqT_woll)/1000
enstT1, enstT2 = eq.fit_powerlaw_T(T,HCO3eqT_enst)/1000
ferrT1, ferrT2 = eq.fit_powerlaw_T(T,HCO3eqT_ferr)/1000

w_woll_HCO3eq0 = tr.w_flux(DICeqFuncs['woll']['HCO3']([xCO2g0,T288,P0]),qc)
w_enst_HCO3eq0 = tr.w_flux(DICeqFuncs['enst']['HCO3']([xCO2g0,T288,P0]),qc)
w_ferr_HCO3eq0 = tr.w_flux(DICeqFuncs['ferr']['HCO3']([xCO2g0,T288,P0]),qc)

# print(w_enst_HCO3eq0,w_woll_HCO3eq0,w_ferr_HCO3eq0)

fig1 = plt.figure(figsize=(5,10))
spec1 = gridspec.GridSpec(ncols=1, nrows=2, figure=fig1)

p11  = fig1.add_subplot(spec1[0,0])
p12  = fig1.add_subplot(spec1[1,0])

p11.text(-0.1, 1.15, '(a)', transform=p11.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p12.text(-0.1, 1.15, '(b)', transform=p12.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')

l111, = p11.plot(xCO2,w_woll_HCO3eq,lw=wid_rock,c=col_woll,ls=lwi_mine)
l112, = p11.plot(xCO2,w_enst_HCO3eq,lw=wid_rock,c=col_enst,ls=lwi_mine)
l113, = p11.plot(xCO2,w_ferr_HCO3eq,lw=wid_rock,c=col_ferr,ls=lwi_mine)

# w_woll_eq.fit_p = w_woll_HCO3eq0 * (xCO2/xCO2g0)**woll2
# p11.plot(xCO2,w_woll_eq.fit_p,c='k')

p11.set_xscale('log')
p11.set_yscale('log')
p11.set_ylim([1e-4,1e4])
p11.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=20)
p11.set_ylabel('$w$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=20)
p11.set_title('Thermodynamic $P_{\mathrm{CO}_2}$ sensitivity',fontsize=20)
p11.set_xticks(np.logspace(-6,0,num=7))
p11.legend([l111,l112,l113], ['Wollastonite','Enstatite','Ferrosilite'],\
           frameon=True,prop={'size':14},loc='lower right',handlelength=2)

p11.text(0.68, 0.92,r'$\beta_{\mathrm{th}}$ = %.2f'%woll2,transform=p11.transAxes,fontsize=12,c=col_woll)
p11.text(0.68, 0.61,r'$\beta_{\mathrm{th}}$ = %.2f'%enst2,transform=p11.transAxes,fontsize=12,c=col_enst)
p11.text(0.68, 0.42,r'$\beta_{\mathrm{th}}$ = %.2f'%ferr2,transform=p11.transAxes,fontsize=12,c=col_ferr)

p11.text(0.02,0.9,r'$w$ = [HCO$_3^-$]$_{\mathrm{eq}}$ $q$ $\propto$ $P_{\mathrm{CO}_2}^{\beta_{\rm th}}$', \
         transform=p11.transAxes, fontsize=14)
# p11.text(0.05,0.82,'$q$ = 0.3 m yr$^{-1}$', transform=p11.transAxes, fontsize=14)
# p11.text(0.05,0.74,'$T$ = 288 K', transform=p11.transAxes, fontsize=14)

l121, = p12.plot(T,wT_woll_HCO3eq,lw=wid_rock,c=col_woll,ls=lwi_mine)
l122, = p12.plot(T,wT_enst_HCO3eq,lw=wid_rock,c=col_enst,ls=lwi_mine)
l123, = p12.plot(T,wT_ferr_HCO3eq,lw=wid_rock,c=col_ferr,ls=lwi_mine)

# w_woll_fit_T = w_woll_HCO3eq0 * np.exp(-wollT2 * (1/T - 1/T288))
# p12.plot(T,w_woll_fit_T,c='k')

p12.set_yscale('log')
p12.set_ylim([1e-4,1e4])
p12.set_xlabel('$T$ [K]',fontsize=20)
p12.set_ylabel('$w$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=20)
p12.set_title('Thermodynamic $T$ sensitivity',fontsize=20)
# p12.legend([l121,l122,l123], ['wolllt','enstdotite','ferrite'],\
#            frameon=True,prop={'size':14},loc='center right',handlelength=2)

p12.text(0.05, 0.65,r'$E_{\mathrm{th}}$ = %d kJ mol$^{-1}$'%wollT2,transform=p12.transAxes,fontsize=12,c=col_woll,zorder=9)
p12.text(0.05, 0.3,r'$E_{\mathrm{th}}$ = %d kJ mol$^{-1}$'%enstT2,transform=p12.transAxes,fontsize=12,c=col_enst,zorder=9)
p12.text(0.05, 0.13,r'$E_{\mathrm{th}}$ = %d kJ mol$^{-1}$'%ferrT2,transform=p12.transAxes,fontsize=12,c=col_ferr,zorder=9)

p12.text(0.02,0.9,r'$w$ = [HCO$_3^-$]$_{\mathrm{eq}}$ $q$ $\propto$ exp ($-E_{\mathrm{th}}/ R T$)', \
         transform=p12.transAxes, fontsize=14)
# p12.text(0.05,0.82,'$q$ = 0.3 m yr$^{-1}$', transform=p12.transAxes, fontsize=14)
# p12.text(0.05,0.74,'$x_{\mathrm{CO}_2(g)}$ = 280 xCO2v', transform=p12.transAxes, fontsize=14)

plt.savefig('w_pyro_therm.pdf',format='pdf',bbox_inches='tight')

# %%
# Additional Fig.: Plot equi. w vs xCO2 and T for rocks

xCO2    = np.logspace(-6,0)         #           Volume mixing ratio or activity of CO2(g) 

xCO2g0 = 280e-6
T288 = 288
argxCO2 = np.array([xCO2,T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2))]).T

HCO3eq_fors = DICeqFuncs['fors']['HCO3'](argxCO2)
HCO3eq_faya = DICeqFuncs['faya']['HCO3'](argxCO2)

w_fors_HCO3eq  = tr.w_flux(HCO3eq_fors,qc*np.ones(len(xCO2)))
w_faya_HCO3eq  = tr.w_flux(HCO3eq_faya,qc*np.ones(len(xCO2)))

fors1, fors2 = eq.fit_powerlaw(xCO2,HCO3eq_fors)
faya1, faya2 = eq.fit_powerlaw(xCO2,HCO3eq_faya)

T = np.linspace(273.15,372.15)
argT = np.array([xCO2g0*np.ones(len(T)),T,P0*np.ones(len(T))]).T

HCO3eqT_fors = DICeqFuncs['fors']['HCO3'](argT)
HCO3eqT_faya = DICeqFuncs['faya']['HCO3'](argT)

wT_fors_HCO3eq = tr.w_flux(HCO3eqT_fors,qc*np.ones(len(T)))
wT_faya_HCO3eq = tr.w_flux(HCO3eqT_faya,qc*np.ones(len(T)))

forsT1, forsT2 = eq.fit_powerlaw_T(T,HCO3eqT_fors)/1000
fayaT1, fayaT2 = eq.fit_powerlaw_T(T,HCO3eqT_faya)/1000

w_fors_HCO3eq0 = tr.w_flux(DICeqFuncs['fors']['HCO3']([xCO2g0,T288,P0]),qc)
w_faya_HCO3eq0 = tr.w_flux(DICeqFuncs['faya']['HCO3']([xCO2g0,T288,P0]),qc)

# print(w_faya_HCO3eq0,w_fors_HCO3eq0)

fig1 = plt.figure(figsize=(5,10))
spec1 = gridspec.GridSpec(ncols=1, nrows=2, figure=fig1)

p11  = fig1.add_subplot(spec1[0,0])
p12  = fig1.add_subplot(spec1[1,0])

p11.text(-0.1, 1.15, '(a)', transform=p11.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p12.text(-0.1, 1.15, '(b)', transform=p12.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')

l111, = p11.plot(xCO2,w_fors_HCO3eq,lw=wid_rock,c=col_fors,ls=lwi_mine)
l112, = p11.plot(xCO2,w_faya_HCO3eq,lw=wid_rock,c=col_faya,ls=lwi_mine)

# w_fors_eq.fit_p = w_fors_HCO3eq0 * (xCO2/xCO2g0)**fors2
# p11.plot(xCO2,w_fors_eq.fit_p,c='k')

p11.set_xscale('log')
p11.set_yscale('log')
p11.set_ylim([1e-4,1e4])
p11.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=20)
p11.set_ylabel('$w$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=20)
p11.set_title('Thermodynamic $P_{\mathrm{CO}_2}$ sensitivity',fontsize=20)
p11.set_xticks(np.logspace(-6,0,num=7))
p11.legend([l111,l112], ['Forsterite','Fayalite'],\
           frameon=True,prop={'size':14},loc='lower right',handlelength=2)

p11.text(0.68, 0.92,r'$\beta_{\mathrm{th}}$ = %.2f'%fors2,transform=p11.transAxes,fontsize=12,c=col_fors)
p11.text(0.68, 0.61,r'$\beta_{\mathrm{th}}$ = %.2f'%faya2,transform=p11.transAxes,fontsize=12,c=col_faya)

p11.text(0.02,0.9,r'$w$ = [HCO$_3^-$]$_{\mathrm{eq}}$ $q$ $\propto$ $P_{\mathrm{CO}_2}^{\beta_{\rm th}}$', \
         transform=p11.transAxes, fontsize=14)
# p11.text(0.05,0.82,'$q$ = 0.3 m yr$^{-1}$', transform=p11.transAxes, fontsize=14)
# p11.text(0.05,0.74,'$T$ = 288 K', transform=p11.transAxes, fontsize=14)

l121, = p12.plot(T,wT_fors_HCO3eq,lw=wid_rock,c=col_fors,ls=lwi_mine)
l122, = p12.plot(T,wT_faya_HCO3eq,lw=wid_rock,c=col_faya,ls=lwi_mine)

# w_fors_fit_T = w_fors_HCO3eq0 * np.exp(-forsT2 * (1/T - 1/T288))
# p12.plot(T,w_fors_fit_T,c='k')

p12.set_yscale('log')
p12.set_ylim([1e-4,1e4])
p12.set_xlabel('$T$ [K]',fontsize=20)
p12.set_ylabel('$w$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=20)
p12.set_title('Thermodynamic $T$ sensitivity',fontsize=20)
# p12.legend([l121,l122,l123], ['forslt','fayadotite','ferrite'],\
#            frameon=True,prop={'size':14},loc='center right',handlelength=2)

p12.text(0.05, 0.65,r'$E_{\mathrm{th}}$ = %d kJ mol$^{-1}$'%forsT2,transform=p12.transAxes,fontsize=12,c=col_fors,zorder=9)
p12.text(0.05, 0.34,r'$E_{\mathrm{th}}$ = %d kJ mol$^{-1}$'%fayaT2,transform=p12.transAxes,fontsize=12,c=col_faya,zorder=9)

p12.text(0.02,0.9,r'$w$ = [HCO$_3^-$]$_{\mathrm{eq}}$ $q$ $\propto$ exp ($-E_{\mathrm{th}}/ R T$)', \
         transform=p12.transAxes, fontsize=14)
# p12.text(0.05,0.82,'$q$ = 0.3 m yr$^{-1}$', transform=p12.transAxes, fontsize=14)
# p12.text(0.05,0.74,'$x_{\mathrm{CO}_2(g)}$ = 280 xCO2v', transform=p12.transAxes, fontsize=14)

plt.savefig('w_oliv_therm.pdf',format='pdf',bbox_inches='tight')

# %%
# Additional Fig.: Plot equi. w vs xCO2 and T for rocks

xCO2    = np.logspace(-6,0)         #           Volume mixing ratio or activity of CO2(g) 

xCO2g0 = 280e-6
T288   = 288
argxCO2 = np.array([xCO2,T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2))]).T

HCO3eq_anth = DICeqFuncs['anth']['HCO3'](argxCO2)
HCO3eq_grun = DICeqFuncs['grun']['HCO3'](argxCO2)

w_anth_HCO3eq  = tr.w_flux(HCO3eq_anth,qc*np.ones(len(xCO2)))
w_grun_HCO3eq  = tr.w_flux(HCO3eq_grun,qc*np.ones(len(xCO2)))

anth1, anth2 = eq.fit_powerlaw(xCO2,HCO3eq_anth)
grun1, grun2 = eq.fit_powerlaw(xCO2,HCO3eq_grun)

T = np.linspace(273.15,372.15)
argT = np.array([xCO2g0*np.ones(len(T)),T,P0*np.ones(len(T))]).T

HCO3eqT_anth = DICeqFuncs['anth']['HCO3'](argT)
HCO3eqT_grun = DICeqFuncs['grun']['HCO3'](argT)

wT_anth_HCO3eq = tr.w_flux(HCO3eqT_anth,qc*np.ones(len(T)))
wT_grun_HCO3eq = tr.w_flux(HCO3eqT_grun,qc*np.ones(len(T)))

anthT1, anthT2 = eq.fit_powerlaw_T(T,HCO3eqT_anth)/1000
grunT1, grunT2 = eq.fit_powerlaw_T(T,HCO3eqT_grun)/1000

w_anth_HCO3eq0 = tr.w_flux(DICeqFuncs['anth']['HCO3']([xCO2g0,T288,P0]),qc)
w_grun_HCO3eq0 = tr.w_flux(DICeqFuncs['grun']['HCO3']([xCO2g0,T288,P0]),qc)

# print(w_grun_HCO3eq0,w_anth_HCO3eq0)

fig1 = plt.figure(figsize=(5,10))
spec1 = gridspec.GridSpec(ncols=1, nrows=2, figure=fig1)

p11  = fig1.add_subplot(spec1[0,0])
p12  = fig1.add_subplot(spec1[1,0])

p11.text(-0.1, 1.15, '(a)', transform=p11.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p12.text(-0.1, 1.15, '(b)', transform=p12.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')

l111, = p11.plot(xCO2,w_anth_HCO3eq,lw=wid_rock,c=col_anth,ls=lwi_mine)
l112, = p11.plot(xCO2,w_grun_HCO3eq,lw=wid_rock,c=col_grun,ls=lwi_mine)

# w_anth_eq.fit_p = w_anth_HCO3eq0 * (xCO2/xCO2g0)**anth2
# p11.plot(xCO2,w_anth_eq.fit_p,c='k')

p11.set_xscale('log')
p11.set_yscale('log')
p11.set_ylim([1e-4,1e4])
p11.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=20)
p11.set_ylabel('$w$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=20)
p11.set_title('Thermodynamic $P_{\mathrm{CO}_2}$ sensitivity',fontsize=20)
p11.set_xticks(np.logspace(-6,0,num=7))
p11.legend([l111,l112], ['Anthophyllite','Grunerite'],\
           frameon=True,prop={'size':14},loc='lower right',handlelength=2)

p11.text(0.68, 0.92,r'$\beta_{\mathrm{th}}$ = %.2f'%anth2,transform=p11.transAxes,fontsize=12,c=col_anth)
p11.text(0.68, 0.61,r'$\beta_{\mathrm{th}}$ = %.2f'%grun2,transform=p11.transAxes,fontsize=12,c=col_grun)

p11.text(0.02,0.9,r'$w$ = [HCO$_3^-$]$_{\mathrm{eq}}$ $q$ $\propto$ $P_{\mathrm{CO}_2}^{\beta_{\rm th}}$', \
         transform=p11.transAxes, fontsize=14)
# p11.text(0.05,0.82,'$q$ = 0.3 m yr$^{-1}$', transform=p11.transAxes, fontsize=14)
# p11.text(0.05,0.74,'$T$ = 288 K', transform=p11.transAxes, fontsize=14)

l121, = p12.plot(T,wT_anth_HCO3eq,lw=wid_rock,c=col_anth,ls=lwi_mine)
l122, = p12.plot(T,wT_grun_HCO3eq,lw=wid_rock,c=col_grun,ls=lwi_mine)

# w_anth_fit_T = w_anth_HCO3eq0 * np.exp(-anthT2 * (1/T - 1/T288))
# p12.plot(T,w_anth_fit_T,c='k')

p12.set_yscale('log')
p12.set_ylim([1e-4,1e4])
p12.set_xlabel('$T$ [K]',fontsize=20)
p12.set_ylabel('$w$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=20)
p12.set_title('Thermodynamic $T$ sensitivity',fontsize=20)
# p12.legend([l121,l122,l123], ['anthlt','grundotite','ferrite'],\
#            frameon=True,prop={'size':14},loc='center right',handlelength=2)

p12.text(0.05, 0.65,r'$E_{\mathrm{th}}$ = %d kJ mol$^{-1}$'%anthT2,transform=p12.transAxes,fontsize=12,c=col_anth,zorder=9)
p12.text(0.05, 0.34,r'$E_{\mathrm{th}}$ = %d kJ mol$^{-1}$'%grunT2,transform=p12.transAxes,fontsize=12,c=col_grun,zorder=9)

p12.text(0.02,0.9,r'$w$ = [HCO$_3^-$]$_{\mathrm{eq}}$ $q$ $\propto$ exp ($-E_{\mathrm{th}}/ R T$)', \
         transform=p12.transAxes, fontsize=14)
# p12.text(0.05,0.82,'$q$ = 0.3 m yr$^{-1}$', transform=p12.transAxes, fontsize=14)
# p12.text(0.05,0.74,'$x_{\mathrm{CO}_2(g)}$ = 280 xCO2v', transform=p12.transAxes, fontsize=14)

plt.savefig('w_amph_therm.pdf',format='pdf',bbox_inches='tight')

# %%
# Fig. 6: Plot HCO3- and w vs xCO2 and T for peri

xCO2g1 = 0.1

fig1 = plt.figure(figsize=(10,10))
spec1 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig1)

p11  = fig1.add_subplot(spec1[0,0])
p12  = fig1.add_subplot(spec1[0,1])
p13  = fig1.add_subplot(spec1[1,0])
p14  = fig1.add_subplot(spec1[1,1])

p11.text(-0.1, 1.15, '(a)', transform=p11.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p12.text(-0.1, 1.15, '(b)', transform=p12.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p13.text(-0.1, 1.15, '(c)', transform=p13.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p14.text(-0.1, 1.15, '(d)', transform=p14.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')

PCO2_clim = np.logspace(-5,-0.42)

T_clim   = np.zeros(len(PCO2_clim))
for i in range(len(PCO2_clim)):
    T_clim[i] = cl.T_KATA(PCO2_clim[i])

PH2O_clim = cl.PH2O(T_clim)
xclim = PCO2_clim / (PCO2_clim + PH2O_clim + 1)

argxCO2   = np.array([xclim,T_clim,P0*np.ones(len(xclim))]).T

DICeq  = DICeqFuncs['peri']['ALK'] (argxCO2)
HCO3eq = DICeqFuncs['peri']['HCO3'](argxCO2)

pH = DICeqFuncs['peri']['pH'](argxCO2)

knet_peri = np.array([
    kFuncs['woll'](T_clim, pH),\
    kFuncs['enst'](T_clim, pH),\
    kFuncs['faya'](T_clim, pH),\
    kFuncs['fors'](T_clim, pH)]).min(axis=0)

argxCO2ki = np.array([xclim,T_clim,P0*np.ones(len(xclim)),\
                     Lc*np.ones(len(xclim)),0*np.ones(len(xclim))]).T
argHCO3ki = np.array([DICeqFuncs['peri']['HCO3'](argxCO2),DwFuncs['peri'](argxCO2ki),\
                    qc*np.ones(len(xclim))]).T
argDICki = np.array([HCO3Funcs(argHCO3ki), xclim, T_clim, P0*np.ones(len(xclim))]).T

HCO3ki = DICtrFuncs['HCO3'](argDICki)
DICki  = DICtrFuncs['ALK'] (argDICki)

argxCO2su = np.array([xclim,T_clim,P0*np.ones(len(xclim)),\
                     Lc*np.ones(len(xclim)),t_soilc*np.ones(len(xclim))]).T
argHCO3su = np.array([DICeqFuncs['peri']['HCO3'](argxCO2),DwFuncs['peri'](argxCO2su),\
                    qc*np.ones(len(xclim))]).T
argDICsu = np.array([HCO3Funcs(argHCO3su), xclim, T_clim, P0*np.ones(len(xclim))]).T

HCO3su = DICtrFuncs['HCO3'](argDICsu)
DICsu  = DICtrFuncs['ALK'] (argDICsu)

argxCO2b   = np.array([xclim,T288*np.ones(len(xclim)),P0*np.ones(len(xclim))]).T

DICbeq  = DICeqFuncs['peri']['ALK'] (argxCO2b)
HCO3beq = DICeqFuncs['peri']['HCO3'](argxCO2b)

pH = DICeqFuncs['peri']['pH'](argxCO2b)

knet_peri = np.array([
    kFuncs['woll'](T288*np.ones(len(xclim)),pH),\
    kFuncs['enst'](T288*np.ones(len(xclim)),pH),\
    kFuncs['faya'](T288*np.ones(len(xclim)),pH),\
    kFuncs['fors'](T288*np.ones(len(xclim)),pH)]).min(axis=0)

argxCO2bki = np.array([xclim,T288*np.ones(len(xclim)),P0*np.ones(len(xclim)),\
                     Lc*np.ones(len(xclim)),0*np.ones(len(xclim))]).T
argHCO3bki = np.array([DICeqFuncs['peri']['HCO3'](argxCO2b),DwFuncs['peri'](argxCO2bki),\
                    qc*np.ones(len(xclim))]).T
argDICbki = np.array([HCO3Funcs(argHCO3bki), xclim, T288*np.ones(len(xclim)), P0*np.ones(len(xclim))]).T

HCO3bki = DICtrFuncs['HCO3'](argDICbki)
DICbki  = DICtrFuncs['ALK'] (argDICbki)

argxCO2bsu = np.array([xclim,T288*np.ones(len(xclim)),P0*np.ones(len(xclim)),\
                     Lc*np.ones(len(xclim)),t_soilc*np.ones(len(xclim))]).T
argHCO3bsu = np.array([DICeqFuncs['peri']['HCO3'](argxCO2b),DwFuncs['peri'](argxCO2bsu),\
                    qc*np.ones(len(xclim))]).T
argDICbsu = np.array([HCO3Funcs(argHCO3bsu), xclim, T288*np.ones(len(xclim)), P0*np.ones(len(xclim))]).T

HCO3bsu = DICtrFuncs['HCO3'](argDICbsu)
DICbsu  = DICtrFuncs['ALK'] (argDICbsu)

T = np.linspace(280,350)
argT = np.array([xCO2g0*np.ones(len(T)),T,P0*np.ones(len(T))]).T
argT0 = np.array([xCO2g0*np.ones(len(T)),T,P0*np.ones(len(T)),\
                        Lc*np.ones(len(T)),np.zeros(len(T))]).T
argTc = np.array([xCO2g0*np.ones(len(T)),T,P0*np.ones(len(T)),\
                        Lc*np.ones(len(T)),t_soilc*np.ones(len(T))]).T
argTs = np.array([xCO2g0*np.ones(len(T)),T,P0*np.ones(len(T)),\
                        Lc*np.ones(len(T)),t_soils*np.ones(len(T))]).T

argTHCO3_peri01  = np.array([DICeqFuncs['peri']['HCO3'](argT),DwFuncs['peri'](argT0),qc*np.ones(len(T))]).T
argTDICtr_peri01 = np.array([HCO3Funcs(argTHCO3_peri01),xCO2g0*np.ones(len(T)),T,P0*np.ones(len(T))]).T
argTHCO3_peri0c  = np.array([DICeqFuncs['peri']['HCO3'](argT),DwFuncs['peri'](argTc),qc*np.ones(len(T))]).T
argTDICtr_peri0c = np.array([HCO3Funcs(argTHCO3_peri0c),xCO2g0*np.ones(len(T)),T,P0*np.ones(len(T))]).T

wT_peri0_HCO3eq = tr.w_flux(DICeqFuncs['peri']['HCO3'](argT),   qc*np.ones(len(T)))
wT_peri0_HCO3ki = tr.w_flux(DICtrFuncs['HCO3'](argTDICtr_peri01),qc*np.ones(len(T)))
wT_peri0_HCO3su = tr.w_flux(DICtrFuncs['HCO3'](argTDICtr_peri0c),qc*np.ones(len(T)))
wT_peri0_DICeq  = tr.w_flux(DICeqFuncs['peri']['ALK'](argT),   qc*np.ones(len(T)))
wT_peri0_DICki  = tr.w_flux(DICtrFuncs['ALK'](argTDICtr_peri01),qc*np.ones(len(T)))
wT_peri0_DICsu  = tr.w_flux(DICtrFuncs['ALK'](argTDICtr_peri0c),qc*np.ones(len(T)))

argT = np.array([xCO2g1*np.ones(len(T)),T,P0*np.ones(len(T))]).T
argT1 = np.array([xCO2g1*np.ones(len(T)),T,P0*np.ones(len(T)),\
                        Lc*np.ones(len(T)),np.zeros(len(T))]).T
argTc = np.array([xCO2g1*np.ones(len(T)),T,P0*np.ones(len(T)),\
                        Lc*np.ones(len(T)),t_soilc*np.ones(len(T))]).T
argTs = np.array([xCO2g1*np.ones(len(T)),T,P0*np.ones(len(T)),\
                        Lc*np.ones(len(T)),t_soils*np.ones(len(T))]).T

argTHCO3_peri01  = np.array([DICeqFuncs['peri']['HCO3'](argT),DwFuncs['peri'](argT1),qc*np.ones(len(T))]).T
argTDICtr_peri01 = np.array([HCO3Funcs(argTHCO3_peri01),xCO2g1*np.ones(len(T)),T,P0*np.ones(len(T))]).T
argTHCO3_peri0c  = np.array([DICeqFuncs['peri']['HCO3'](argT),DwFuncs['peri'](argTc),qc*np.ones(len(T))]).T
argTDICtr_peri0c = np.array([HCO3Funcs(argTHCO3_peri0c),xCO2g1*np.ones(len(T)),T,P0*np.ones(len(T))]).T

wT2_peri0_HCO3eq = tr.w_flux(DICeqFuncs['peri']['HCO3'](argT),   qc*np.ones(len(T)))
wT2_peri0_HCO3ki = tr.w_flux(DICtrFuncs['HCO3'](argTDICtr_peri01),qc*np.ones(len(T)))
wT2_peri0_HCO3su = tr.w_flux(DICtrFuncs['HCO3'](argTDICtr_peri0c),qc*np.ones(len(T)))
wT2_peri0_DICeq  = tr.w_flux(DICeqFuncs['peri']['ALK'](argT),   qc*np.ones(len(T)))
wT2_peri0_DICki  = tr.w_flux(DICtrFuncs['ALK'](argTDICtr_peri01),qc*np.ones(len(T)))
wT2_peri0_DICsu  = tr.w_flux(DICtrFuncs['ALK'](argTDICtr_peri0c),qc*np.ones(len(T)))
wT2_peri0_CO2su  = tr.w_flux(DICtrFuncs['CO2'](argTDICtr_peri0c),qc*np.ones(len(T)))

wb_peri_HCO3eq = tr.w_flux(HCO3beq,qc*np.ones(len(xclim)))
wb_peri_HCO3ki = tr.w_flux(HCO3bki,qc*np.ones(len(xclim)))
wb_peri_HCO3su = tr.w_flux(HCO3bsu,qc*np.ones(len(xclim)))
wb_peri_DICeq  = tr.w_flux(DICbeq, qc*np.ones(len(xclim)))
wb_peri_DICki  = tr.w_flux(DICbki, qc*np.ones(len(xclim)))
wb_peri_DICsu  = tr.w_flux(DICbsu, qc*np.ones(len(xclim)))

w_peri_HCO3eq = tr.w_flux(HCO3eq,qc*np.ones(len(xclim)))
w_peri_HCO3ki = tr.w_flux(HCO3ki,qc*np.ones(len(xclim)))
w_peri_HCO3su = tr.w_flux(HCO3su,qc*np.ones(len(xclim)))
w_peri_DICeq  = tr.w_flux(DICeq, qc*np.ones(len(xclim)))
w_peri_DICki  = tr.w_flux(DICki, qc*np.ones(len(xclim)))
w_peri_DICsu  = tr.w_flux(DICsu, qc*np.ones(len(xclim)))

j1_ki = np.where((wT_peri0_HCO3eq < 2 * wT_peri0_HCO3ki) == True)[0][0]

j2_ki = np.where((wT2_peri0_HCO3eq < 2 * wT2_peri0_HCO3ki) == True)[0][0]

k1_ki2 = np.where((wT2_peri0_DICki < 2 * wT2_peri0_HCO3ki) == True)[0][0]

k3_ki = np.where((w_peri_HCO3eq < 2 * w_peri_HCO3ki) == True)[0][0]

l111, = p11.plot(xclim,wb_peri_HCO3eq,lw=wid_rock,c=col_bica,ls=lwo_mine)
l112, = p11.plot(xclim,wb_peri_HCO3ki,lw=wid_rock,c=col_bica,ls=lwi_rock)
l113, = p11.plot(xclim,wb_peri_HCO3su,lw=wid_rock,c=col_bica,ls=lwi_mine)
l114, = p11.plot(xclim,wb_peri_DICeq ,lw=wid_rock,c=col_peri,ls=lwo_mine)
l115, = p11.plot(xclim,wb_peri_DICki ,lw=wid_rock,c=col_peri,ls=lwi_rock)
l116, = p11.plot(xclim,wb_peri_DICsu ,lw=wid_rock,c=col_peri,ls=lwi_mine)


p11.set_xscale('log')
p11.set_yscale('log')
p11.set_ylim([1e-2,5e3])
p11.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=20)
p11.set_ylabel('$w$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=20)
p11.set_title('Without climate model ($T$ = 288 K)',fontsize=14)
p11.set_xticks(np.logspace(-5,-1,num=5))

p11.text(xclim[8], wb_peri_DICeq[8]*0.5, 'Thermodynamic', rotation=25, ha='center', va='bottom',fontsize=12)
p11.text(xclim[14], wb_peri_DICki[14]*1.2, 'Kinetic', rotation=0, ha='center', va='bottom',fontsize=12)
p11.text(xclim[14], wb_peri_DICsu[14]*0.4, 'Supply', rotation=0, ha='center', va='bottom',fontsize=12)

l121, = p12.plot(T,wT_peri0_HCO3eq,lw=wid_rock,c=col_bica,ls=lwo_mine)
l122, = p12.plot(T,wT_peri0_HCO3ki,lw=wid_rock,c=col_bica,ls=lwi_rock)
l123, = p12.plot(T,wT_peri0_HCO3su,lw=wid_rock,c=col_bica,ls=lwi_mine)
l121, = p12.plot(T,wT_peri0_DICeq, lw=wid_rock,c=col_peri,ls=lwo_mine)
l122, = p12.plot(T,wT_peri0_DICki, lw=wid_rock,c=col_peri,ls=lwi_rock)
l123, = p12.plot(T,wT_peri0_DICsu, lw=wid_rock,c=col_peri,ls=lwi_mine)

p12.scatter(T[j1_ki],wT_peri0_HCO3ki[j1_ki],marker='o',s=80,color=col_bica,zorder=4)
p12.scatter(T[j1_ki],wT_peri0_DICki [j1_ki],marker='o',s=80,color=col_peri,zorder=4)

p12.set_yscale('log')
p12.set_ylim([1e-2,5e3])
p12.set_xlabel('$T$ [K]',fontsize=20)
p12.set_ylabel('$w$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=20)
p12.set_title(r'Without climate model ($P_{\mathrm{CO}_2}$ = 280 $\mu$bar)',fontsize=14)

p12.text(T[25], wT_peri0_DICeq[25], 'Thermodynamic', rotation=345, ha='center', va='bottom',fontsize=12)
p12.text(T[10], wT_peri0_HCO3ki[10], 'Kinetic', rotation=35, ha='center', va='bottom',fontsize=12)
p12.text(T[20], wT_peri0_HCO3su[20]*0.4, 'Supply', rotation=0, ha='center', va='bottom',fontsize=12)

l131, = p13.plot(xclim,w_peri_HCO3eq,lw=wid_rock,c=col_bica,ls=lwo_mine)
l132, = p13.plot(xclim,w_peri_HCO3ki,lw=wid_rock,c=col_bica,ls=lwi_rock)
l133, = p13.plot(xclim,w_peri_HCO3su,lw=wid_rock,c=col_bica,ls=lwi_mine)
l134, = p13.plot(xclim,w_peri_DICeq ,lw=wid_rock,c=col_peri,ls=lwo_mine)
l135, = p13.plot(xclim,w_peri_DICki ,lw=wid_rock,c=col_peri,ls=lwi_rock)
l136, = p13.plot(xclim,w_peri_DICsu ,lw=wid_rock,c=col_peri,ls=lwi_mine)

p13.scatter(xclim[k3_ki],w_peri_HCO3ki[k3_ki],marker='o',s=80,color=col_bica,zorder=4)
p13.scatter(xclim[k3_ki],w_peri_DICki[k3_ki],marker='o',s=80,color=col_peri,zorder=4)

p13.set_xscale('log')
p13.set_yscale('log')
p13.set_ylim([1e-2,5e3])
p13.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=20)
p13.set_ylabel('$w$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=20)
p13.set_title('With climate model ($T$ = $f$ $(P_{\mathrm{CO}_2})$)',fontsize=14)
p13.set_xticks(np.logspace(-5,-1,num=5))

p13.text(xclim[30], w_peri_DICeq[30], 'Thermodynamic', rotation=10, ha='center', va='bottom',fontsize=12)
p13.text(xclim[30], w_peri_DICki[30], 'Kinetic', rotation=30, ha='center', va='bottom',fontsize=12)
p13.text(xclim[14], w_peri_DICsu[14]*0.4, 'Supply', rotation=0, ha='center', va='bottom',fontsize=12)

l141, = p14.plot(T,wT2_peri0_HCO3eq,lw=wid_rock,c=col_bica,ls=lwo_mine)
l142, = p14.plot(T,wT2_peri0_HCO3ki,lw=wid_rock,c=col_bica,ls=lwi_rock)
l143, = p14.plot(T,wT2_peri0_HCO3su,lw=wid_rock,c=col_bica,ls=lwi_mine)
l141, = p14.plot(T,wT2_peri0_DICeq, lw=wid_rock,c=col_peri,ls=lwo_mine)
l142, = p14.plot(T,wT2_peri0_DICki, lw=wid_rock,c=col_peri,ls=lwi_rock)
l143, = p14.plot(T,wT2_peri0_DICsu, lw=wid_rock,c=col_peri,ls=lwi_mine)

p14.scatter(T[j2_ki],wT2_peri0_HCO3ki[j2_ki],marker='o',s=80,color=col_bica,zorder=4)
p14.scatter(T[j2_ki],wT2_peri0_DICki [j2_ki],marker='o',s=80,color=col_peri,zorder=4)
# p14.scatter(T[k1_ki2],wT2_peri0_DICki[k1_ki2],marker='v',s=90,color=col_peri,zorder=4)

p14.set_yscale('log')
p14.set_ylim([1e-2,5e3])
p14.set_xlabel('$T$ [K]',fontsize=20)
p14.set_ylabel('$w$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=20)
p14.set_title('Without climate model ($P_{\mathrm{CO}_2}$ = 0.1 bar)',fontsize=14)

p14.text(T[25], wT2_peri0_DICeq[25], 'Thermodynamic', rotation=345, ha='center', va='bottom',fontsize=12)
p14.text(T[25], wT2_peri0_HCO3ki[25], 'Kinetic', rotation=25, ha='center', va='bottom',fontsize=12)
p14.text(T[25], wT2_peri0_HCO3su[25]*0.4, 'Supply', rotation=0, ha='center', va='bottom',fontsize=12)

fig1.legend([l116,l113],\
            ['$A$','[HCO$^-_3$]'],\
            frameon=True,prop={'size':12},title='Peridotite Weathering',title_fontsize=12,\
            bbox_to_anchor=(0.11,0.92),loc='upper left',\
            bbox_transform=fig1.transFigure,handlelength=3,fontsize=12)

custom_lines3 = [Line2D([0], [0], c=col_woll, lw=wid_mine, ls=lwo_mine),
                 Line2D([0], [0], c=col_woll, lw=wid_rock, ls=lwi_rock),
                 Line2D([0], [0], c=col_woll, lw=wid_rock, ls=lwi_mine)]

fig1.legend(custom_lines3,\
            ['Maximum',\
             'Generalized (Young soils)',\
             'Generalized (Old soils)'],\
            frameon=True,prop={'size':12},\
            bbox_to_anchor=(0.48,0.78),loc='upper right',\
            bbox_transform=fig1.transFigure,handlelength=3,fontsize=12)

plt.savefig('w_peri_clim.pdf',format='pdf',bbox_inches='tight')

# %%
# Fig. 7: Plot w vs xCO2 for basalt and granite

PCO2_clim = np.logspace(-5,-0.42)

T_clim   = np.zeros(len(PCO2_clim))
for i in range(len(PCO2_clim)):
    T_clim[i] = cl.T_KATA(PCO2_clim[i])

Qconu = 10*np.ones(len(PCO2_clim))
Qcont = 0.3*np.ones(len(PCO2_clim))
Qconl = 0.01*np.ones(len(PCO2_clim))

Qseau = 1*np.ones(len(PCO2_clim))
Qseaf = 0.05*np.ones(len(PCO2_clim))
Qseal = 0.001*np.ones(len(PCO2_clim))

argxCO2 = np.array([PCO2_clim,T_clim,P0*np.ones(len(PCO2_clim))]).T

argxCO2cu = np.array([PCO2_clim,T_clim,P0*np.ones(len(PCO2_clim)), \
                     Lc*np.ones(len(PCO2_clim)),np.zeros(len(PCO2_clim))]).T
argxCO2cc = np.array([PCO2_clim,T_clim,P0*np.ones(len(PCO2_clim)), \
                     Lc*np.ones(len(PCO2_clim)),t_soilc*np.ones(len(PCO2_clim))]).T
argxCO2cl = np.array([PCO2_clim,T_clim,P0*np.ones(len(PCO2_clim)), \
                     Lc*np.ones(len(PCO2_clim)),1e6*np.ones(len(PCO2_clim))]).T

argxCO2su = np.array([PCO2_clim,T_clim,P0*np.ones(len(PCO2_clim)), \
                     Ls*np.ones(len(PCO2_clim)),np.zeros(len(PCO2_clim))]).T
argxCO2ss = np.array([PCO2_clim,T_clim,P0*np.ones(len(PCO2_clim)), \
                     Ls*np.ones(len(PCO2_clim)),t_soils*np.ones(len(PCO2_clim))]).T
argxCO2ll = np.array([PCO2_clim,T_clim,P0*np.ones(len(PCO2_clim)), \
                     Lc*np.ones(len(PCO2_clim)),1e8*np.ones(len(PCO2_clim))]).T

argHCO3_basac = np.array([DICeqFuncs['bash']['HCO3'](argxCO2),DwFuncs['bash'](argxCO2cc),Qcont]).T
argHCO3_baquc = np.array([DICeqFuncs['bash']['HCO3'](argxCO2),DwFuncs['bash'](argxCO2cc),Qconu]).T
argHCO3_baqlc = np.array([DICeqFuncs['bash']['HCO3'](argxCO2),DwFuncs['bash'](argxCO2cc),Qconl]).T
argHCO3_batuc = np.array([DICeqFuncs['bash']['HCO3'](argxCO2),DwFuncs['bash'](argxCO2cu),Qcont]).T
argHCO3_batlc = np.array([DICeqFuncs['bash']['HCO3'](argxCO2),DwFuncs['bash'](argxCO2cl),Qcont]).T

argHCO3_granc = np.array([DICeqFuncs['grah']['HCO3'](argxCO2),DwFuncs['grah'](argxCO2cc),Qcont]).T
argHCO3_grquc = np.array([DICeqFuncs['grah']['HCO3'](argxCO2),DwFuncs['grah'](argxCO2cc),Qconu]).T
argHCO3_grqlc = np.array([DICeqFuncs['grah']['HCO3'](argxCO2),DwFuncs['grah'](argxCO2cc),Qconl]).T
argHCO3_grtuc = np.array([DICeqFuncs['grah']['HCO3'](argxCO2),DwFuncs['grah'](argxCO2cu),Qcont]).T
argHCO3_grtlc = np.array([DICeqFuncs['grah']['HCO3'](argxCO2),DwFuncs['grah'](argxCO2cl),Qcont]).T

argDICtr_basac = np.array([HCO3Funcs(argHCO3_basac),PCO2_clim, T_clim,P0*np.ones(len(PCO2_clim))]).T
argDICtr_baquc = np.array([HCO3Funcs(argHCO3_baquc),PCO2_clim, T_clim,P0*np.ones(len(PCO2_clim))]).T
argDICtr_baqlc = np.array([HCO3Funcs(argHCO3_baqlc),PCO2_clim, T_clim,P0*np.ones(len(PCO2_clim))]).T
argDICtr_batuc = np.array([HCO3Funcs(argHCO3_batuc),PCO2_clim, T_clim,P0*np.ones(len(PCO2_clim))]).T
argDICtr_batlc = np.array([HCO3Funcs(argHCO3_batlc),PCO2_clim, T_clim,P0*np.ones(len(PCO2_clim))]).T

argDICtr_granc = np.array([HCO3Funcs(argHCO3_granc),PCO2_clim, T_clim,P0*np.ones(len(PCO2_clim))]).T
argDICtr_grquc = np.array([HCO3Funcs(argHCO3_grquc),PCO2_clim, T_clim,P0*np.ones(len(PCO2_clim))]).T
argDICtr_grqlc = np.array([HCO3Funcs(argHCO3_grqlc),PCO2_clim, T_clim,P0*np.ones(len(PCO2_clim))]).T
argDICtr_grtuc = np.array([HCO3Funcs(argHCO3_grtuc),PCO2_clim, T_clim,P0*np.ones(len(PCO2_clim))]).T
argDICtr_grtlc = np.array([HCO3Funcs(argHCO3_grtlc),PCO2_clim, T_clim,P0*np.ones(len(PCO2_clim))]).T

Hbasac = tr.w_flux(DICtrFuncs['HCO3'](argDICtr_basac),Qcont)
Hbaquc = tr.w_flux(DICtrFuncs['HCO3'](argDICtr_baquc),Qconu)
Hbaqlc = tr.w_flux(DICtrFuncs['HCO3'](argDICtr_baqlc),Qconl)
Hbatuc = tr.w_flux(DICtrFuncs['HCO3'](argDICtr_batuc),Qcont)
Hbatlc = tr.w_flux(DICtrFuncs['HCO3'](argDICtr_batlc),Qcont)

basac = tr.w_flux(DICtrFuncs['ALK'](argDICtr_basac),Qcont)
baquc = tr.w_flux(DICtrFuncs['ALK'](argDICtr_baquc),Qconu)
baqlc = tr.w_flux(DICtrFuncs['ALK'](argDICtr_baqlc),Qconl)
batuc = tr.w_flux(DICtrFuncs['ALK'](argDICtr_batuc),Qcont)
batlc = tr.w_flux(DICtrFuncs['ALK'](argDICtr_batlc),Qcont)

Hgranc = tr.w_flux(DICtrFuncs['HCO3'](argDICtr_granc),Qcont)
Hgrquc = tr.w_flux(DICtrFuncs['HCO3'](argDICtr_grquc),Qconu)
Hgrqlc = tr.w_flux(DICtrFuncs['HCO3'](argDICtr_grqlc),Qconl)
Hgrtuc = tr.w_flux(DICtrFuncs['HCO3'](argDICtr_grtuc),Qcont)
Hgrtlc = tr.w_flux(DICtrFuncs['HCO3'](argDICtr_grtlc),Qcont)

granc = tr.w_flux(DICtrFuncs['ALK'](argDICtr_granc),Qcont)
grquc = tr.w_flux(DICtrFuncs['ALK'](argDICtr_grquc),Qconu)
grqlc = tr.w_flux(DICtrFuncs['ALK'](argDICtr_grqlc),Qconl)
grtuc = tr.w_flux(DICtrFuncs['ALK'](argDICtr_grtuc),Qcont)
grtlc = tr.w_flux(DICtrFuncs['ALK'](argDICtr_grtlc),Qcont)

argHCO3_basas = np.array([DICeqFuncs['bash']['HCO3'](argxCO2),DwFuncs['bash'](argxCO2ss),Qseaf]).T
argHCO3_baqus = np.array([DICeqFuncs['bash']['HCO3'](argxCO2),DwFuncs['bash'](argxCO2ss),Qseau]).T
argHCO3_baqls = np.array([DICeqFuncs['bash']['HCO3'](argxCO2),DwFuncs['bash'](argxCO2ss),Qseal]).T
argHCO3_batus = np.array([DICeqFuncs['bash']['HCO3'](argxCO2),DwFuncs['bash'](argxCO2su),Qseaf]).T
argHCO3_batll = np.array([DICeqFuncs['bash']['HCO3'](argxCO2),DwFuncs['bash'](argxCO2ll),Qseaf]).T

argHCO3_grans = np.array([DICeqFuncs['grah']['HCO3'](argxCO2),DwFuncs['grah'](argxCO2ss),Qseaf]).T
argHCO3_grqus = np.array([DICeqFuncs['grah']['HCO3'](argxCO2),DwFuncs['grah'](argxCO2ss),Qseau]).T
argHCO3_grqls = np.array([DICeqFuncs['grah']['HCO3'](argxCO2),DwFuncs['grah'](argxCO2ss),Qseal]).T
argHCO3_grtus = np.array([DICeqFuncs['grah']['HCO3'](argxCO2),DwFuncs['grah'](argxCO2su),Qseaf]).T
argHCO3_grtll = np.array([DICeqFuncs['grah']['HCO3'](argxCO2),DwFuncs['grah'](argxCO2ll),Qseaf]).T

argDICtr_basas = np.array([HCO3Funcs(argHCO3_basas),PCO2_clim, T_clim,P0*np.ones(len(PCO2_clim))]).T
argDICtr_baqus = np.array([HCO3Funcs(argHCO3_baqus),PCO2_clim, T_clim,P0*np.ones(len(PCO2_clim))]).T
argDICtr_baqls = np.array([HCO3Funcs(argHCO3_baqls),PCO2_clim, T_clim,P0*np.ones(len(PCO2_clim))]).T
argDICtr_batus = np.array([HCO3Funcs(argHCO3_batus),PCO2_clim, T_clim,P0*np.ones(len(PCO2_clim))]).T
argDICtr_batll = np.array([HCO3Funcs(argHCO3_batll),PCO2_clim, T_clim,P0*np.ones(len(PCO2_clim))]).T

argDICtr_grans = np.array([HCO3Funcs(argHCO3_grans),PCO2_clim, T_clim,P0*np.ones(len(PCO2_clim))]).T
argDICtr_grqus = np.array([HCO3Funcs(argHCO3_grqus),PCO2_clim, T_clim,P0*np.ones(len(PCO2_clim))]).T
argDICtr_grqls = np.array([HCO3Funcs(argHCO3_grqls),PCO2_clim, T_clim,P0*np.ones(len(PCO2_clim))]).T
argDICtr_grtus = np.array([HCO3Funcs(argHCO3_grtus),PCO2_clim, T_clim,P0*np.ones(len(PCO2_clim))]).T
argDICtr_grtll = np.array([HCO3Funcs(argHCO3_grtll),PCO2_clim, T_clim,P0*np.ones(len(PCO2_clim))]).T

Hbasas = tr.w_flux(DICtrFuncs['HCO3'](argDICtr_basas),Qseaf)
Hbaqus = tr.w_flux(DICtrFuncs['HCO3'](argDICtr_baqus),Qseau)
Hbaqls = tr.w_flux(DICtrFuncs['HCO3'](argDICtr_baqls),Qseal)
Hbatus = tr.w_flux(DICtrFuncs['HCO3'](argDICtr_batus),Qseaf)
Hbatll = tr.w_flux(DICtrFuncs['HCO3'](argDICtr_batll),Qseaf)

basas = tr.w_flux(DICtrFuncs['ALK'](argDICtr_basas),Qseaf)
baqus = tr.w_flux(DICtrFuncs['ALK'](argDICtr_baqus),Qseau)
baqls = tr.w_flux(DICtrFuncs['ALK'](argDICtr_baqls),Qseal)
batus = tr.w_flux(DICtrFuncs['ALK'](argDICtr_batus),Qseaf)
batll = tr.w_flux(DICtrFuncs['ALK'](argDICtr_batll),Qseaf)

Hgrans = tr.w_flux(DICtrFuncs['HCO3'](argDICtr_grans),Qseaf)
Hgrqus = tr.w_flux(DICtrFuncs['HCO3'](argDICtr_grqus),Qseau)
Hgrqls = tr.w_flux(DICtrFuncs['HCO3'](argDICtr_grqls),Qseal)
Hgrtus = tr.w_flux(DICtrFuncs['HCO3'](argDICtr_grtus),Qseaf)
Hgrtll = tr.w_flux(DICtrFuncs['HCO3'](argDICtr_grtll),Qseaf)

grans = tr.w_flux(DICtrFuncs['ALK'](argDICtr_grans),Qseaf)
grqus = tr.w_flux(DICtrFuncs['ALK'](argDICtr_grqus),Qseau)
grqls = tr.w_flux(DICtrFuncs['ALK'](argDICtr_grqls),Qseal)
grtus = tr.w_flux(DICtrFuncs['ALK'](argDICtr_grtus),Qseaf)
grtll = tr.w_flux(DICtrFuncs['ALK'](argDICtr_grtll),Qseaf)

# DwFuncs['bash'](argxCO2cc)/Qconl
# DwFuncs['bash'](argxCO2cu)/Qcont

# DwFuncs['grah'](argxCO2cc)/Qconl

# DwFuncs['bash'](argxCO2ss)/Qseal

# DwFuncs['grah'](argxCO2ss)/Qseaf


j_grqlc = np.where((Qconl > DwFuncs['grah'](argxCO2cc)) == True)[0][0]
j_baqlc = np.where((Qconl > DwFuncs['bash'](argxCO2cc)) == True)[0][0]
j_batuc = np.where((Qcont < DwFuncs['bash'](argxCO2cu)) == True)[0][0]

j_baqls = np.where((Qseal > DwFuncs['bash'](argxCO2ss)) == True)[0][0]
j_grans = np.where((Qseaf > DwFuncs['grah'](argxCO2ss)) == True)[0][0]

fig2 = plt.figure(figsize=(10,10))
spec2 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig2)
p21 = fig2.add_subplot(spec2[0,0])
p22 = fig2.add_subplot(spec2[0,1])
p23 = fig2.add_subplot(spec2[1,0])
p24 = fig2.add_subplot(spec2[1,1])
p21.text(-0.1, 1.15, '(a)', transform=p21.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p22.text(-0.1, 1.15, '(b)', transform=p22.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p23.text(-0.1, 1.15, '(c)', transform=p23.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p24.text(-0.1, 1.15, '(d)', transform=p24.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')

l211,   =   p21.plot(PCO2_clim,baquc,lw=wid_rock,c=col_basa,ls=lwo_mine)
l212,   =   p21.plot(PCO2_clim,basac,lw=wid_rock,c=col_basa,ls=lwi_mine)
l213,   =   p21.plot(PCO2_clim,baqlc,lw=wid_rock,c=col_basa,ls=lwi_rock)
l214,   =   p21.plot(PCO2_clim,grquc,lw=wid_rock,c=col_gran,ls=lwo_mine)
l215,   =   p21.plot(PCO2_clim,granc,lw=wid_rock,c=col_gran,ls=lwi_mine)
l216,   =   p21.plot(PCO2_clim,grqlc,lw=wid_rock,c=col_gran,ls=lwi_rock)

p21.scatter(PCO2_clim[j_grqlc],grqlc[j_grqlc],marker='v',s=90,zorder=4,color=col_gran)
p21.scatter(PCO2_clim[j_baqlc],baqlc[j_baqlc],marker='v',s=90,zorder=4,color=col_basa)

p21.set_ylim([5e-5,2e1])
p21.set_xscale('log')
p21.set_yscale('log')
p21.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=20)
p21.set_ylabel('$w_{\mathrm{cont}}$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=20)
p21.set_title('Humid vs. Arid Climate',fontsize=20)
p21.set_xticks(np.logspace(-5,-1,num=5))

p21.text(PCO2_clim[20], grqlc[20]*0.2, 'Thermodynamic', rotation=15, ha='center', va='bottom',fontsize=12)
p21.text(PCO2_clim[30], baquc[30]*1.4, 'Supply', rotation=0, ha='center', va='bottom',fontsize=12)
p21.text(PCO2_clim[47], grqlc[47]*0.4, 'Supply', rotation=0, ha='center', va='bottom',fontsize=12)
p21.text(PCO2_clim[3], basac[3]*0.2, 'Thermo.', rotation=10, ha='center', va='bottom',fontsize=12)

l221,   =   p22.plot(PCO2_clim,batuc,lw=wid_rock,c=col_basa,ls=lwo_mine)
l222,   =   p22.plot(PCO2_clim,basac,lw=wid_rock,c=col_basa,ls=lwi_mine)
l223,   =   p22.plot(PCO2_clim,batlc,lw=wid_rock,c=col_basa,ls=lwi_rock)
l224,   =   p22.plot(PCO2_clim,grtuc,lw=wid_rock,c=col_gran,ls=lwo_mine)
l225,   =   p22.plot(PCO2_clim,granc,lw=wid_rock,c=col_gran,ls=lwi_mine)
l226,   =   p22.plot(PCO2_clim,grtlc,lw=wid_rock,c=col_gran,ls=lwi_rock)

p22.scatter(PCO2_clim[j_batuc],batuc[j_batuc],marker='o',s=80,zorder=4,color=col_basa)

p22.set_ylim([5e-5,2e1])
p22.set_xscale('log')
p22.set_yscale('log')
p22.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=20)
p22.set_ylabel('$w_{\mathrm{cont}}$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=20)
p22.set_title('Young vs. Old Soils',fontsize=20)
p22.set_xticks(np.logspace(-5,-1,num=5))

p22.text(PCO2_clim[30], batuc[30], 'Kinetic', rotation=20, ha='center', va='bottom',fontsize=12)
p22.text(PCO2_clim[45], batuc[45]*0.4, 'Thermo.', rotation=0, ha='center', va='bottom',fontsize=12)
p22.text(PCO2_clim[30], grtuc[30]*0.3, 'Thermodynamic', rotation=10, ha='center', va='bottom',fontsize=12)
p22.text(PCO2_clim[20], basac[20]*0.2, 'Supply', rotation=0, ha='center', va='bottom',fontsize=12)

l231,   =   p23.plot(PCO2_clim,baqus,lw=wid_rock,c=col_basa,ls=lwo_mine)
l232,   =   p23.plot(PCO2_clim,basas,lw=wid_rock,c=col_basa,ls=lwi_mine)
l233,   =   p23.plot(PCO2_clim,baqls,lw=wid_rock,c=col_basa,ls=lwi_rock)
l234,   =   p23.plot(PCO2_clim,grqus,lw=wid_rock,c=col_gran,ls=lwo_mine)
l235,   =   p23.plot(PCO2_clim,grans,lw=wid_rock,c=col_gran,ls=lwi_mine)
l236,   =   p23.plot(PCO2_clim,grqls,lw=wid_rock,c=col_gran,ls=lwi_rock)

p23.scatter(PCO2_clim[j_baqls],baqls[j_baqls],marker='v',s=90,zorder=4,color=col_basa)
p23.scatter(PCO2_clim[j_grans],grans[j_grans],marker='v',s=90,zorder=4,color=col_gran)

p23.set_ylim([5e-5,2e1])
p23.set_xscale('log')
p23.set_yscale('log')
p23.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=20)
p23.set_ylabel('$w_{\mathrm{seaf}}$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=20)
p23.set_title('High vs. Low Hydrothermal Heat',fontsize=20)
p23.set_xticks(np.logspace(-5,-1,num=5))

p23.text(PCO2_clim[20], grqls[20]*0.2, 'Thermodynamic', rotation=15, ha='center', va='bottom',fontsize=12)
p23.text(PCO2_clim[30], baqus[30]*1.4, 'Supply', rotation=0, ha='center', va='bottom',fontsize=12)
p23.text(PCO2_clim[2], basas[2]*0.1, 'Thermo.', rotation=10, ha='center', va='bottom',fontsize=12)

l243,   =   p24.plot(PCO2_clim,batus,lw=wid_rock,c=col_basa,ls=lwo_mine)
l242,   =   p24.plot(PCO2_clim,basas,lw=wid_rock,c=col_basa,ls=lwi_mine)
l243,   =   p24.plot(PCO2_clim,batll,lw=wid_rock,c=col_basa,ls=lwi_rock)
l244,   =   p24.plot(PCO2_clim,grtus,lw=wid_rock,c=col_gran,ls=lwo_mine)
l245,   =   p24.plot(PCO2_clim,grans,lw=wid_rock,c=col_gran,ls=lwi_mine)
l246,   =   p24.plot(PCO2_clim,grtll,lw=wid_rock,c=col_gran,ls=lwi_rock)

p24.scatter(PCO2_clim[j_grans],grans[j_grans],marker='v',s=90,zorder=4,color=col_gran)

p24.set_ylim([5e-5,2e1])
p24.set_xscale('log')
p24.set_yscale('log')
p24.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=20)
p24.set_ylabel('$w_{\mathrm{seaf}}$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=20)
p24.set_title('Young vs. Old Pore-space',fontsize=20)
p24.set_xticks(np.logspace(-5,-1,num=5))

p24.text(PCO2_clim[25],  grtus[25]*2,  'Thermodynamic', rotation=10, ha='center', va='bottom',fontsize=12)
p24.text(PCO2_clim[25],  batll[25]*1.2,  'Supply', rotation=0, ha='center', va='bottom',fontsize=12)
p24.text(PCO2_clim[25], basas[25]*1.2, 'Supply', rotation=0, ha='center', va='bottom',fontsize=12)


p21.text(0.05, 0.65, '$t_{s}$ = 100 kyr \n$\psi$ = $\psi_0$',\
         transform=p21.transAxes,fontsize=12)
p22.text(0.05, 0.85, '$q$ = 0.3 m yr$^{-1}$ \n$\psi$ = $\psi_0$',\
         transform=p22.transAxes,fontsize=12)
p23.text(0.05, 0.65, '$t_{s}$ = 50 Myr \n$\psi$ = 100$\psi_0$',\
         transform=p23.transAxes,fontsize=12)
p24.text(0.05, 0.85, '$q$ = 0.05 m yr$^{-1}$',\
         transform=p24.transAxes,fontsize=12)

custom_lines1 = [Line2D([0], [0], c=col_woll, lw=wid_rock, ls=lwo_mine),
                 Line2D([0], [0], c=col_woll, lw=wid_rock, ls=lwi_mine),
                 Line2D([0], [0], c=col_woll, lw=wid_rock, ls=lwi_rock)]

custom_lines2 = [Line2D([0], [0], c=col_woll, lw=wid_rock, ls=lwo_mine),
                 Line2D([0], [0], c=col_woll, lw=wid_rock, ls=lwi_mine)]

fig2.legend(custom_lines1,\
            ['Humid ($q$ = 10 m yr$^{-1}$)',\
             'Modern-mean ($q$ = 0.3 m yr$^{-1}$)',\
             'Arid ($q$ = 0.01 m yr$^{-1}$)',],\
            frameon=True,prop={'size':12},bbox_to_anchor=(0.11,0.93),loc='upper left',\
            bbox_transform=fig2.transFigure,ncol=1,handlelength=3)

fig2.legend(custom_lines1,\
            ['Young soils ($t_{s}$ = 0)',\
             'Modern-mean ($t_{s}$ = 100 kyr)',\
             'Old soils ($t_{s}$ = 1 Myr)',],\
            frameon=True,prop={'size':12},bbox_to_anchor=(0.6,0.58),loc='lower left',\
            bbox_transform=fig2.transFigure,ncol=1,handlelength=3)

fig2.legend(custom_lines1,\
            ['High heat flux ($q$ = 1 m yr$^{-1}$)',\
             'Modern-mean ($q$ = 0.05 m yr$^{-1}$)',\
             'Low heat flux ($q$ = 0.001 m yr$^{-1}$)',],\
            frameon=True,prop={'size':12},bbox_to_anchor=(0.11,0.44),loc='upper left',\
            bbox_transform=fig2.transFigure,ncol=1,handlelength=3)

fig2.legend(custom_lines1,\
            ['Young pore-space ($t_{s}$ = 0, $\psi = 100 \psi_0$)',\
             'Modern-mean ($t_{s}$ = 50 Myr, $\psi = 100 \psi_0$)',
             'Small flowpath ($t_{s}$ = 50 Myr, $\psi = \psi_0$)'],\
            frameon=True,prop={'size':12},bbox_to_anchor=(0.6,0.12),loc='lower left',\
            bbox_transform=fig2.transFigure,ncol=1,handlelength=3)

fig2.legend([l212,l215],\
            ['Basalt', 'Granite'],\
            frameon=True,prop={'size':14},bbox_to_anchor=(0.49,0.58),loc='lower right',\
            bbox_transform=fig2.transFigure,ncol=1,handlelength=2)

plt.tight_layout()
plt.savefig('w_xCO2_endmembers.pdf',format='pdf',bbox_inches='tight')

# %%
# Fig. 8: Plot w vs xCO2 and q for rocks

fig1 = plt.figure(figsize=(10,5))
spec1 = gridspec.GridSpec(ncols=2, nrows=1, figure=fig1)

p12  = fig1.add_subplot(spec1[0,0])
p11  = fig1.add_subplot(spec1[0,1])

p11.text(-0.1, 1.15, '(b)', transform=p11.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p12.text(-0.1, 1.15, '(a)', transform=p12.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')

# PCO2_clim = np.logspace(-5.5,-0.42)
PCO2_clim = np.logspace(-5.3,-0.42)

T_clim   = np.zeros(len(PCO2_clim))
for i in range(len(PCO2_clim)):
    T_clim[i] = cl.T_KATA(PCO2_clim[i])

PH2O_clim = cl.PH2O(T_clim)

argxCO2   = np.array([PCO2_clim,T_clim,P0*np.ones(len(PCO2_clim))]).T

argxCO2ki = np.array([PCO2_clim,T_clim,P0*np.ones(len(PCO2_clim)),\
                     Lc*np.ones(len(PCO2_clim)),np.zeros(len(PCO2_clim))]).T


argxCO2su = np.array([PCO2_clim,T_clim,P0*np.ones(len(PCO2_clim)),\
                     Lc*np.ones(len(PCO2_clim)),t_soilc*np.ones(len(PCO2_clim))]).T


argHCO3ki = np.array([DICeqFuncs['peri']['HCO3'](argxCO2),DwFuncs['peri'](argxCO2ki),\
                    qc*np.ones(len(PCO2_clim))]).T
argDICki = np.array([HCO3Funcs(argHCO3ki), PCO2_clim, T_clim, P0*np.ones(len(PCO2_clim))]).T

argHCO3su = np.array([DICeqFuncs['peri']['HCO3'](argxCO2),DwFuncs['peri'](argxCO2su),\
                    qc*np.ones(len(PCO2_clim))]).T
argDICsu = np.array([HCO3Funcs(argHCO3su), PCO2_clim, T_clim, P0*np.ones(len(PCO2_clim))]).T

DICeq  = DICeqFuncs['peri']['ALK'] (argxCO2)
HCO3eq = DICeqFuncs['peri']['HCO3'](argxCO2)

HCO3ki = DICtrFuncs['HCO3'](argDICki)
DICki  = DICtrFuncs['ALK'] (argDICki)

HCO3su = DICtrFuncs['HCO3'](argDICsu)
DICsu  = DICtrFuncs['ALK'] (argDICsu)

w_peri_HCO3eq = tr.w_flux(HCO3eq,qc*np.ones(len(PCO2_clim)))
w_peri_HCO3ki = tr.w_flux(HCO3ki,qc*np.ones(len(PCO2_clim)))
w_peri_HCO3su = tr.w_flux(HCO3su,qc*np.ones(len(PCO2_clim)))
w_peri_DICeq  = tr.w_flux(DICeq, qc*np.ones(len(PCO2_clim)))
w_peri_DICki  = tr.w_flux(DICki, qc*np.ones(len(PCO2_clim)))
w_peri_DICsu  = tr.w_flux(DICsu, qc*np.ones(len(PCO2_clim)))

argHCO3ki = np.array([DICeqFuncs['bash']['HCO3'](argxCO2),DwFuncs['bash'](argxCO2ki),\
                    qc*np.ones(len(PCO2_clim))]).T
argDICki = np.array([HCO3Funcs(argHCO3ki), PCO2_clim, T_clim, P0*np.ones(len(PCO2_clim))]).T

argHCO3su = np.array([DICeqFuncs['bash']['HCO3'](argxCO2),DwFuncs['bash'](argxCO2su),\
                    qc*np.ones(len(PCO2_clim))]).T
argDICsu = np.array([HCO3Funcs(argHCO3su), PCO2_clim, T_clim, P0*np.ones(len(PCO2_clim))]).T

DICeq  = DICeqFuncs['bash']['ALK'] (argxCO2)
HCO3eq = DICeqFuncs['bash']['HCO3'](argxCO2)

HCO3ki = DICtrFuncs['HCO3'](argDICki)
DICki  = DICtrFuncs['ALK'] (argDICki)

HCO3su = DICtrFuncs['HCO3'](argDICsu)
DICsu  = DICtrFuncs['ALK'] (argDICsu)

w_basa_HCO3eq = tr.w_flux(HCO3eq,qc*np.ones(len(PCO2_clim)))
w_basa_HCO3ki = tr.w_flux(HCO3ki,qc*np.ones(len(PCO2_clim)))
w_basa_HCO3su = tr.w_flux(HCO3su,qc*np.ones(len(PCO2_clim)))
w_basa_DICeq  = tr.w_flux(DICeq, qc*np.ones(len(PCO2_clim)))
w_basa_DICki  = tr.w_flux(DICki, qc*np.ones(len(PCO2_clim)))
w_basa_DICsu  = tr.w_flux(DICsu, qc*np.ones(len(PCO2_clim)))


argHCO3ki = np.array([DICeqFuncs['grah']['HCO3'](argxCO2),DwFuncs['grah'](argxCO2ki),\
                    qc*np.ones(len(PCO2_clim))]).T
argDICki = np.array([HCO3Funcs(argHCO3ki), PCO2_clim, T_clim, P0*np.ones(len(PCO2_clim))]).T

argHCO3su = np.array([DICeqFuncs['grah']['HCO3'](argxCO2),DwFuncs['grah'](argxCO2su),\
                    qc*np.ones(len(PCO2_clim))]).T
argDICsu = np.array([HCO3Funcs(argHCO3su), PCO2_clim, T_clim, P0*np.ones(len(PCO2_clim))]).T

DICeq  = DICeqFuncs['grah']['ALK'] (argxCO2)
HCO3eq = DICeqFuncs['grah']['HCO3'](argxCO2)

HCO3ki = DICtrFuncs['HCO3'](argDICki)
DICki  = DICtrFuncs['ALK'] (argDICki)

HCO3su = DICtrFuncs['HCO3'](argDICsu)
DICsu  = DICtrFuncs['ALK'] (argDICsu)

w_gran_HCO3eq = tr.w_flux(HCO3eq,qc*np.ones(len(PCO2_clim)))
w_gran_HCO3ki = tr.w_flux(HCO3ki,qc*np.ones(len(PCO2_clim)))
w_gran_HCO3su = tr.w_flux(HCO3su,qc*np.ones(len(PCO2_clim)))
w_gran_DICeq  = tr.w_flux(DICeq, qc*np.ones(len(PCO2_clim)))
w_gran_DICki  = tr.w_flux(DICki, qc*np.ones(len(PCO2_clim)))
w_gran_DICsu  = tr.w_flux(DICsu, qc*np.ones(len(PCO2_clim)))

ii_basa1 = np.where((w_basa_HCO3eq < 2 * w_basa_HCO3ki) == True)[0][0]
ii_peri1 = np.where((w_peri_HCO3eq < 2 * w_peri_HCO3ki) == True)[0][0]
ii_gran1 = np.where((w_gran_HCO3eq < 2 * w_gran_HCO3ki) == True)[0][0]




# l111, = p11.plot(PCO2_clim,w_basa_DICeq,lw=wid_rock,c=col_basa,ls=lwo_mine)
l112, = p11.plot(PCO2_clim,w_basa_DICki,lw=wid_rock,c=col_basa,ls=lwi_rock)
l113, = p11.plot(PCO2_clim,w_basa_DICsu,lw=wid_rock,c=col_basa,ls=lwi_mine)

# l114, = p11.plot(PCO2_clim,w_peri_DICeq,lw=wid_rock,c=col_peri,ls=lwo_mine)
l115, = p11.plot(PCO2_clim,w_peri_DICki,lw=wid_rock,c=col_peri,ls=lwi_rock)
l116, = p11.plot(PCO2_clim,w_peri_DICsu,lw=wid_rock,c=col_peri,ls=lwi_mine)

# l117, = p11.plot(PCO2_clim,w_gran_DICeq,lw=wid_rock,c=col_gran,ls=lwo_mine)
l118, = p11.plot(PCO2_clim,w_gran_DICki,lw=wid_rock,c=col_gran,ls=lwi_rock)
l119, = p11.plot(PCO2_clim,w_gran_DICsu,lw=wid_rock,c=col_gran,ls=lwi_mine)

p11.axvline(x=280e-6, color=col_woll)

p11.scatter(PCO2_clim[ii_basa1],w_basa_DICki[ii_basa1],marker='o',s=80,zorder=4,color=col_basa)
p11.scatter(PCO2_clim[ii_peri1],w_peri_DICki[ii_peri1],marker='o',s=80,zorder=4,color=col_peri)


p11.set_xscale('log')
p11.set_yscale('log')
p11.set_ylim([1e-5,1e2])
p11.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=20)
p11.set_ylabel('$w$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=20)
p11.set_title('Climate vs. Regimes',fontsize=20)
p11.set_xticks(np.logspace(-5,-1,num=5))

p11.text(0.82,0.83,'Thermo-\ndynamic', transform=p11.transAxes, fontsize=12)
p11.text(0.55,0.79,'Kinetic', rotation=30, transform=p11.transAxes, fontsize=12)
p11.text(0.4,0.5,'Supply', transform=p11.transAxes, fontsize=12)
p11.text(0.5,0.63,'Thermodynamic', rotation=10, transform=p11.transAxes, fontsize=12)

p11.text(0.37, 0.25, '$P_{\mathrm{CO}_2}$ = 280 $\mu$bar', transform=p11.transAxes,fontsize=12)

argHCO3 = np.array([xCO2g0,T288,P0]).T
argDw1 = np.array([xCO2g0,T288,P0,Lc,t_soilc]).T
argDw2 = np.array([xCO2g0,T288,P0,Lc,0]).T

HCO3eq_basa = DICeqFuncs['bash']['HCO3'](argHCO3)
HCO3eq_peri = DICeqFuncs['peri']['HCO3'](argHCO3)
HCO3eq_gran = DICeqFuncs['grah']['HCO3'](argHCO3)

weq_basa = tr.w_flux(DICeqFuncs['bash']['ALK'](argHCO3)*np.ones(len(q)),q)
weq_peri = tr.w_flux(DICeqFuncs['peri']['ALK'](argHCO3)*np.ones(len(q)),q)
weq_gran = tr.w_flux(DICeqFuncs['grah']['ALK'](argHCO3)*np.ones(len(q)),q)

Dw1_basa = DwFuncs['bash'](argDw1)
Dw1_peri = DwFuncs['peri'](argDw1)
Dw1_gran = DwFuncs['grah'](argDw1)

Hbasa1 = HCO3Funcs(np.array([HCO3eq_basa*np.ones(len(q)),Dw1_basa*np.ones(len(q)),q]).T)
Hperi1 = HCO3Funcs(np.array([HCO3eq_peri*np.ones(len(q)),Dw1_peri*np.ones(len(q)),q]).T)
Hgran1 = HCO3Funcs(np.array([HCO3eq_gran*np.ones(len(q)),Dw1_gran*np.ones(len(q)),q]).T)

Dbasa1 = DICtrFuncs['ALK'](np.array([Hbasa1,xCO2g0*np.ones(len(q)),T288*np.ones(len(q)),P0*np.ones(len(q))]).T)
Dperi1 = DICtrFuncs['ALK'](np.array([Hperi1,xCO2g0*np.ones(len(q)),T288*np.ones(len(q)),P0*np.ones(len(q))]).T)
Dgran1 = DICtrFuncs['ALK'](np.array([Hgran1,xCO2g0*np.ones(len(q)),T288*np.ones(len(q)),P0*np.ones(len(q))]).T)

wDbasa1 = tr.w_flux(Dbasa1,q)
wDperi1 = tr.w_flux(Dperi1,q)
wDgran1 = tr.w_flux(Dgran1,q)
wHbasa1 = tr.w_flux(Hbasa1,q)
wHperi1 = tr.w_flux(Hperi1,q)
wHgran1 = tr.w_flux(Hgran1,q)

Dw2_basa = DwFuncs['bash'](argDw2)
Dw2_peri = DwFuncs['peri'](argDw2)
Dw2_gran = DwFuncs['grah'](argDw2)

Hbasa2 = HCO3Funcs(np.array([HCO3eq_basa*np.ones(len(q)),Dw2_basa*np.ones(len(q)),q]).T)
Hperi2 = HCO3Funcs(np.array([HCO3eq_peri*np.ones(len(q)),Dw2_peri*np.ones(len(q)),q]).T)
Hgran2 = HCO3Funcs(np.array([HCO3eq_gran*np.ones(len(q)),Dw2_gran*np.ones(len(q)),q]).T)

Dbasa2 = DICtrFuncs['ALK'](np.array([Hbasa2,xCO2g0*np.ones(len(q)),T288*np.ones(len(q)),P0*np.ones(len(q))]).T)
Dperi2 = DICtrFuncs['ALK'](np.array([Hperi2,xCO2g0*np.ones(len(q)),T288*np.ones(len(q)),P0*np.ones(len(q))]).T)
Dgran2 = DICtrFuncs['ALK'](np.array([Hgran2,xCO2g0*np.ones(len(q)),T288*np.ones(len(q)),P0*np.ones(len(q))]).T)

wDbasa2 = tr.w_flux(Dbasa2,q)
wDperi2 = tr.w_flux(Dperi2,q)
wDgran2 = tr.w_flux(Dgran2,q)
wHbasa2 = tr.w_flux(Hbasa2,q)
wHperi2 = tr.w_flux(Hperi2,q)
wHgran2 = tr.w_flux(Hgran2,q)

i_basa1 = np.where((HCO3eq_basa > 2 * Hbasa1) == True)[0][0]
i_peri1 = np.where((HCO3eq_peri > 2 * Hperi1) == True)[0][0]
i_gran1 = np.where((HCO3eq_gran > 2 * Hgran1) == True)[0][0]
i_basa2 = np.where((HCO3eq_basa > 2 * Hbasa2) == True)[0][0]
i_peri2 = np.where((HCO3eq_peri > 2 * Hperi2) == True)[0][0]
i_gran2 = np.where((HCO3eq_gran > 2 * Hgran2) == True)[0][0]

l121,  =   p12.plot(q,wDbasa1,lw=wid_rock,c=col_basa,ls=lwi_mine)
l122,  =   p12.plot(q,wDperi1,lw=wid_rock,c=col_peri,ls=lwi_mine)
l123,  =   p12.plot(q,wDgran1,lw=wid_rock,c=col_gran,ls=lwi_mine)
l124,  =   p12.plot(q,wDbasa2,lw=wid_rock,c=col_basa,ls=lwi_rock)
l125,  =   p12.plot(q,wDperi2,lw=wid_rock,c=col_peri,ls=lwi_rock)
l126,  =   p12.plot(q,wDgran2,lw=wid_rock,c=col_gran,ls=lwi_rock)

p12.scatter(q[i_basa1],wDbasa1[i_basa1],marker='o',s=80,zorder=4,color=col_basa)
p12.scatter(q[i_peri1],wDperi1[i_peri1],marker='o',s=80,zorder=4,color=col_peri)
p12.scatter(q[i_gran1],wDgran1[i_gran1],marker='o',s=80,zorder=4,color=col_gran)
p12.scatter(q[i_basa2],wDbasa2[i_basa2],marker='o',s=80,zorder=4,color=col_basa)
p12.scatter(q[i_peri2],wDperi2[i_peri2],marker='o',s=80,zorder=4,color=col_peri)
p12.scatter(q[i_gran2],wDgran2[i_gran2],marker='o',s=80,zorder=4,color=col_gran)

p12.axvline(x=0.3, color=col_woll)

p12.set_xscale('log')
p12.set_yscale('log')
p12.set_ylim([1e-5,1e2])
p12.set_xlabel('$q$ [m yr$^{-1}$]',fontsize=20)
p12.set_ylabel('$w$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=20)
p12.set_title('Transport vs. Regimes',c=col_woll,fontsize=20)
p12.set_xticks(np.logspace(-5,3,num=5))

fig1.legend([l122,l121,l123], ['Peridotite','Basalt','Granite'],\
           frameon=True,prop={'size':12},bbox_to_anchor=(0.74,0.165),loc='lower right',\
           bbox_transform=fig1.transFigure,handlelength=2)

p12.text(0.01,0.5,'Thermo-\ndynamic', transform=p12.transAxes, fontsize=12)
p12.text(0.7,0.68,'Kinetic', transform=p12.transAxes, fontsize=12)
p12.text(0.7,0.5,'Supply', transform=p12.transAxes, fontsize=12)

p12.text(0.57, 0.25, '$q$ = 0.3 m yr$^{-1}$', transform=p12.transAxes,fontsize=12)

custom_lines2 = [Line2D([0], [0], c=col_woll, lw=wid_rock, ls=lwi_rock),
                 Line2D([0], [0], c=col_woll, lw=wid_rock, ls=lwi_mine)]

fig1.legend(custom_lines2, ['Young soils ($t_\mathrm{s}$ = 0)','Old soils ($t_\mathrm{s}$ = 100 kyr)'],\
            frameon=True,prop={'size':11},\
            bbox_to_anchor=(0.98,0.17),loc='lower right',\
            bbox_transform=fig1.transFigure,handlelength=3,fontsize=12)

plt.savefig('w_rock_regimes.pdf',format='pdf',bbox_inches='tight')

# %%
# Fig. 9: Plot w vs T for rocks

T = np.linspace(280,350)

argT = np.array([xCO2g0*np.ones(len(T)),T,P0*np.ones(len(T))]).T
argT1 = np.array([xCO2g0*np.ones(len(T)),T,P0*np.ones(len(T)),\
                        Lc*np.ones(len(T)),np.zeros(len(T))]).T
argTc = np.array([xCO2g0*np.ones(len(T)),T,P0*np.ones(len(T)),\
                        Lc*np.ones(len(T)),t_soilc*np.ones(len(T))]).T
argTs = np.array([xCO2g0*np.ones(len(T)),T,P0*np.ones(len(T)),\
                        Lc*np.ones(len(T)),t_soils*np.ones(len(T))]).T

argTHCO3_peri01  = np.array([DICeqFuncs['peri']['HCO3'](argT),DwFuncs['peri'](argT1),qc*np.ones(len(T))]).T
argTDICtr_peri01 = np.array([HCO3Funcs(argTHCO3_peri01),xCO2g0*np.ones(len(T)),T,P0*np.ones(len(T))]).T
argTHCO3_periT1  = np.array([DICeqFuncs['peri']['HCO3'](argT),DwFuncs['peri'](argT1),tr.q_contT(Temp=T,epsi=0.03)]).T
argTDICtr_periT1 = np.array([HCO3Funcs(argTHCO3_periT1),xCO2g0*np.ones(len(T)),T,P0*np.ones(len(T))]).T

argTHCO3_basa01  = np.array([DICeqFuncs['bash']['HCO3'](argT),DwFuncs['bash'](argT1),qc*np.ones(len(T))]).T
argTDICtr_basa01 = np.array([HCO3Funcs(argTHCO3_basa01),xCO2g0*np.ones(len(T)),T,P0*np.ones(len(T))]).T
argTHCO3_basaT1  = np.array([DICeqFuncs['bash']['HCO3'](argT),DwFuncs['bash'](argT1),tr.q_contT(Temp=T,epsi=0.03)]).T
argTDICtr_basaT1 = np.array([HCO3Funcs(argTHCO3_basaT1),xCO2g0*np.ones(len(T)),T,P0*np.ones(len(T))]).T

argTHCO3_gran01  = np.array([DICeqFuncs['grah']['HCO3'](argT),DwFuncs['grah'](argT1),qc*np.ones(len(T))]).T
argTDICtr_gran01 = np.array([HCO3Funcs(argTHCO3_gran01),xCO2g0*np.ones(len(T)),T,P0*np.ones(len(T))]).T
argTHCO3_granT1  = np.array([DICeqFuncs['grah']['HCO3'](argT),DwFuncs['grah'](argT1),tr.q_contT(Temp=T,epsi=0.03)]).T
argTDICtr_granT1 = np.array([HCO3Funcs(argTHCO3_granT1),xCO2g0*np.ones(len(T)),T,P0*np.ones(len(T))]).T

wT_peri0_DICki  = tr.w_flux(DICtrFuncs['ALK'](argTDICtr_peri01),qc*np.ones(len(T)))
wT_periT_DICki  = tr.w_flux(DICtrFuncs['ALK'](argTDICtr_periT1),tr.q_contT(Temp=T,epsi=0.03))
wT_basa0_DICki  = tr.w_flux(DICtrFuncs['ALK'](argTDICtr_basa01),qc*np.ones(len(T)))
wT_basaT_DICki  = tr.w_flux(DICtrFuncs['ALK'](argTDICtr_basaT1),tr.q_contT(Temp=T,epsi=0.03))
wT_gran0_DICki  = tr.w_flux(DICtrFuncs['ALK'](argTDICtr_gran01),qc*np.ones(len(T)))
wT_granT_DICki  = tr.w_flux(DICtrFuncs['ALK'](argTDICtr_granT1),tr.q_contT(Temp=T,epsi=0.03))

wT_peri0_DICeq  = tr.w_flux(DICeqFuncs['peri']['ALK'](argT),qc*np.ones(len(T)))
wT_periT_DICeq  = tr.w_flux(DICeqFuncs['peri']['ALK'](argT),tr.q_contT(Temp=T,epsi=0.03))
wT_basa0_DICeq  = tr.w_flux(DICeqFuncs['bash']['ALK'](argT),qc*np.ones(len(T)))
wT_basaT_DICeq  = tr.w_flux(DICeqFuncs['bash']['ALK'](argT),tr.q_contT(Temp=T,epsi=0.03))
wT_gran0_DICeq  = tr.w_flux(DICeqFuncs['grah']['ALK'](argT),qc*np.ones(len(T)))
wT_granT_DICeq  = tr.w_flux(DICeqFuncs['grah']['ALK'](argT),tr.q_contT(Temp=T,epsi=0.03))

i_peri0 = np.where((DwFuncs['peri'](argT1) > qc) == True)[0][0]
i_periT = np.where((DwFuncs['peri'](argT1) > tr.q_contT(Temp=T,epsi=0.03)) == True)[0][0]
i_basa0 = np.where((DwFuncs['bash'](argT1) > qc) == True)[0][0]
i_basaT = np.where((DwFuncs['bash'](argT1) > tr.q_contT(Temp=T,epsi=0.03)) == True)[0][0]
i_gran0 = np.where((DwFuncs['grah'](argT1) > qc) == True)[0][0]
i_granT = np.where((DwFuncs['grah'](argT1) > tr.q_contT(Temp=T,epsi=0.03)) == True)[0][0]

# i_gran0 = np.where((wT_gran0_DICeq < 2 * wT_gran0_DICki) == True)[0][0]
# i_granT = np.where((wT_granT_DICeq < 2 * wT_granT_DICki) == True)[0][0]

fig1 = plt.figure(figsize=(5,10))
spec1 = gridspec.GridSpec(ncols=1, nrows=2, figure=fig1)

p11  = fig1.add_subplot(spec1[0,0])
p12  = fig1.add_subplot(spec1[1,0])

p11.text(-0.1, 1.15, '(a)', transform=p11.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p12.text(-0.1, 1.15, '(b)', transform=p12.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')

l111, = p11.plot(T,wT_basa0_DICki, lw=wid_rock,c=col_basa,ls=lwi_mine)
l112, = p11.plot(T,wT_basaT_DICki, lw=wid_rock,c=col_basa,ls=lwi_rock)

l113, = p11.plot(T,wT_peri0_DICki, lw=wid_rock,c=col_peri,ls=lwi_mine)
l114, = p11.plot(T,wT_periT_DICki, lw=wid_rock,c=col_peri,ls=lwi_rock)

l115, = p11.plot(T,wT_gran0_DICki, lw=wid_rock,c=col_gran,ls=lwi_mine)
l116, = p11.plot(T,wT_granT_DICki, lw=wid_rock,c=col_gran,ls=lwi_rock)

p11.scatter(T[i_peri0],wT_peri0_DICki[i_peri0],marker='o',s=80,zorder=4,color=col_peri)
p11.scatter(T[i_periT],wT_periT_DICki[i_periT],marker='o',s=80,zorder=4,color=col_peri)
p11.scatter(T[i_basa0],wT_basa0_DICki[i_basa0],marker='o',s=80,zorder=4,color=col_basa)
p11.scatter(T[i_basaT],wT_basaT_DICki[i_basaT],marker='o',s=80,zorder=4,color=col_basa)
p11.scatter(T[i_gran0],wT_gran0_DICki[i_gran0],marker='o',s=80,zorder=4,color=col_gran)
p11.scatter(T[i_granT],wT_granT_DICki[i_granT],marker='o',s=80,zorder=4,color=col_gran)

p11.set_yscale('log')
p11.set_ylim([4e-2,1e2])
p11.set_xlabel('$T$ [K]',fontsize=20)
p11.set_ylabel('$w$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=20)
p11.set_title('Without climate model ($P_{\mathrm{CO}_2}$ = 280 $\mu$bar)',fontsize=14)


p11.text(T[14], wT_basaT_DICki[14]*1.1, 'Kinetic', rotation=45, ha='center', va='bottom',fontsize=12)
p11.text(T[43], wT_basaT_DICki[43]*0.7, 'Thermodynamic', rotation=330, ha='center', va='bottom',fontsize=12)



PCO2_clim = np.logspace(-5,-0.42)

T_clim   = np.zeros(len(PCO2_clim))
for i in range(len(PCO2_clim)):
    T_clim[i] = cl.T_KATA(PCO2_clim[i])


argT = np.array([PCO2_clim,T_clim,P0*np.ones(len(T_clim))]).T
argT1 = np.array([PCO2_clim,T_clim,P0*np.ones(len(T_clim)),Lc*np.ones(len(T_clim)),np.zeros(len(T_clim))]).T
argTc = np.array([PCO2_clim,T_clim,P0*np.ones(len(T_clim)),Lc*np.ones(len(T_clim)),t_soilc*np.ones(len(T_clim))]).T
argTs = np.array([PCO2_clim,T_clim,P0*np.ones(len(T_clim)),Lc*np.ones(len(T_clim)),t_soils*np.ones(len(T_clim))]).T

argTHCO3_peri01  = np.array([DICeqFuncs['peri']['HCO3'](argT),DwFuncs['peri'](argT1),\
                             qc*np.ones(len(T_clim))]).T
argTDICtr_peri01 = np.array([HCO3Funcs(argTHCO3_peri01),PCO2_clim,T_clim,P0*np.ones(len(T_clim))]).T
argTHCO3_periT1  = np.array([DICeqFuncs['peri']['HCO3'](argT),DwFuncs['peri'](argT1),\
                             tr.q_contT(Temp=T_clim,epsi=0.03)]).T
argTDICtr_periT1 = np.array([HCO3Funcs(argTHCO3_periT1),PCO2_clim,T_clim,P0*np.ones(len(T_clim))]).T

argTHCO3_basa01  = np.array([DICeqFuncs['bash']['HCO3'](argT),DwFuncs['bash'](argT1),\
                             qc*np.ones(len(T_clim))]).T
argTDICtr_basa01 = np.array([HCO3Funcs(argTHCO3_basa01),PCO2_clim,T_clim,P0*np.ones(len(T_clim))]).T
argTHCO3_basaT1  = np.array([DICeqFuncs['bash']['HCO3'](argT),DwFuncs['bash'](argT1),\
                             tr.q_contT(Temp=T_clim,epsi=0.03)]).T
argTDICtr_basaT1 = np.array([HCO3Funcs(argTHCO3_basaT1),PCO2_clim,T_clim,P0*np.ones(len(T_clim))]).T

argTHCO3_gran01  = np.array([DICeqFuncs['grah']['HCO3'](argT),DwFuncs['grah'](argT1),\
                             qc*np.ones(len(T_clim))]).T
argTDICtr_gran01 = np.array([HCO3Funcs(argTHCO3_gran01),PCO2_clim,T_clim,P0*np.ones(len(T_clim))]).T
argTHCO3_granT1  = np.array([DICeqFuncs['grah']['HCO3'](argT),DwFuncs['grah'](argT1),\
                             tr.q_contT(Temp=T_clim,epsi=0.03)]).T
argTDICtr_granT1 = np.array([HCO3Funcs(argTHCO3_granT1),PCO2_clim,T_clim,P0*np.ones(len(T_clim))]).T

wT_peri0_DICki  = tr.w_flux(DICtrFuncs['ALK'](argTDICtr_peri01),qc*np.ones(len(T_clim)))
wT_periT_DICki  = tr.w_flux(DICtrFuncs['ALK'](argTDICtr_periT1),tr.q_contT(Temp=T_clim,epsi=0.03))
wT_basa0_DICki  = tr.w_flux(DICtrFuncs['ALK'](argTDICtr_basa01),qc*np.ones(len(T_clim)))
wT_basaT_DICki  = tr.w_flux(DICtrFuncs['ALK'](argTDICtr_basaT1),tr.q_contT(Temp=T_clim,epsi=0.03))
wT_gran0_DICki  = tr.w_flux(DICtrFuncs['ALK'](argTDICtr_gran01),qc*np.ones(len(T_clim)))
wT_granT_DICki  = tr.w_flux(DICtrFuncs['ALK'](argTDICtr_granT1),tr.q_contT(Temp=T_clim,epsi=0.03))

wT_peri0_DICeq  = tr.w_flux(DICeqFuncs['peri']['ALK'](argT),qc*np.ones(len(T_clim)))
wT_periT_DICeq  = tr.w_flux(DICeqFuncs['peri']['ALK'](argT),tr.q_contT(Temp=T_clim,epsi=0.03))
wT_basa0_DICeq  = tr.w_flux(DICeqFuncs['bash']['ALK'](argT),qc*np.ones(len(T_clim)))
wT_basaT_DICeq  = tr.w_flux(DICeqFuncs['bash']['ALK'](argT),tr.q_contT(Temp=T_clim,epsi=0.03))
wT_gran0_DICeq  = tr.w_flux(DICeqFuncs['grah']['ALK'](argT),qc*np.ones(len(T_clim)))
wT_granT_DICeq  = tr.w_flux(DICeqFuncs['grah']['ALK'](argT),tr.q_contT(Temp=T_clim,epsi=0.03))

i_peri0 = np.where((DwFuncs['peri'](argT1) > qc) == True)[0][0]
# i_periT = np.where((DwFuncs['peri'](argT1) > tr.q_contT(Temp=T,epsi=0.03)) == True)[0][0]
i_basa0 = np.where((DwFuncs['bash'](argT1) > qc) == True)[0][0]
i_basaT = np.where((DwFuncs['bash'](argT1) > tr.q_contT(Temp=T,epsi=0.03)) == True)[0][0]
# i_gran0 = np.where((DwFuncs['grah'](argT1) > qc) == True)[0][0]
# i_granT = np.where((DwFuncs['grah'](argT1) > tr.q_contT(Temp=T,epsi=0.03)) == True)[0][0]

l121, = p12.plot(T_clim,wT_basa0_DICki, lw=wid_rock,c=col_basa,ls=lwi_mine)
l122, = p12.plot(T_clim,wT_basaT_DICki, lw=wid_rock,c=col_basa,ls=lwi_rock)

l123, = p12.plot(T_clim,wT_peri0_DICki, lw=wid_rock,c=col_peri,ls=lwi_mine)
l124, = p12.plot(T_clim,wT_periT_DICki, lw=wid_rock,c=col_peri,ls=lwi_rock)

l125, = p12.plot(T_clim,wT_gran0_DICki, lw=wid_rock,c=col_gran,ls=lwi_mine)
l126, = p12.plot(T_clim,wT_granT_DICki, lw=wid_rock,c=col_gran,ls=lwi_rock)

p12.scatter(T_clim[i_basa0],wT_basa0_DICki[i_basa0],marker='o',s=80,zorder=4,color=col_basa)
p12.scatter(T_clim[i_basaT],wT_basaT_DICki[i_basaT],marker='o',s=80,zorder=4,color=col_basa)
p12.scatter(T_clim[i_peri0],wT_peri0_DICki[i_peri0],marker='o',s=80,zorder=4,color=col_peri)
# p12.scatter(T_clim[i_periT],wT_periT_DICki[i_periT],marker='o',s=80,zorder=4,color=col_peri)
# p12.scatter(T_clim[i_gran0],wT_gran0_DICki[i_gran0],marker='o',s=80,zorder=4,color=col_gran)
# p12.scatter(T_clim[i_granT],wT_granT_DICki[i_granT],marker='o',s=80,zorder=4,color=col_gran)

p12.set_yscale('log')
p12.set_ylim([4e-2,1e2])
p12.set_xlabel('$T$ [K]',fontsize=20)
p12.set_ylabel('$w$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=20)
p12.set_title('With climate model ($P_{\mathrm{CO}_2}$ = $f$ ($T$))',fontsize=14)

p12.text(T_clim[5], wT_basaT_DICki[5], 'Kinetic', rotation=45, ha='center', va='bottom',fontsize=12)
p12.text(T_clim[45], wT_basa0_DICki[45]*0.4, 'Thermodynamic', rotation=5, ha='center', va='bottom',fontsize=12)
# p12.text(T_clim[48], wT_peri0_DICki[48]*0.4, 'Thermo-\ndynamic', rotation=0, ha='center', va='bottom',fontsize=12)

custom_lines2 = [Line2D([0], [0], c=col_woll, lw=wid_rock, ls=lwi_rock),
                 Line2D([0], [0], c=col_woll, lw=wid_rock, ls=lwi_mine)]


fig1.legend(custom_lines2, ['$q = 0.3(1 + \epsilon (T - T_0))$ m yr$^{-1}$','$q$ = 0.3 m yr$^{-1}$'],\
            frameon=True,prop={'size':12},\
            bbox_to_anchor=(0.21,0.44),loc='upper left',\
            bbox_transform=fig1.transFigure,handlelength=3)

fig1.legend([l123,l121,l125], ['Peridotite', 'Basalt', 'Granite'],\
            frameon=True,prop={'size':14},\
            bbox_to_anchor=(0.96,0.08),loc='lower right',\
            bbox_transform=fig1.transFigure,handlelength=3)

plt.savefig('w_T_rock.pdf',format='pdf',bbox_inches='tight')

# %%
# Fig. 10: Calculate weathering rates for Earth

PCO2Func = cl.import_PCO2Func(pd.read_csv('./database/KrisTott2018Fig3B_PCO2.csv'))      
TempFunc = cl.import_TempFunc(pd.read_csv('./database/KrisTott2018Fig3D_Temp.csv'))      
contFunc = cl.import_contFunc(pd.read_csv('./database/KrisTott2018Fig3E_cont.csv'))      
coloFunc = cl.import_contFunc(pd.read_csv('./database/KrisTott2018Fig3E_cont_lower.csv'))
coupFunc = cl.import_contFunc(pd.read_csv('./database/KrisTott2018Fig3E_cont_upper.csv'))
seafFunc = cl.import_seafFunc(pd.read_csv('./database/KrisTott2018Fig3F_seaf_B.csv'))    
seupFunc = cl.import_seafFunc(pd.read_csv('./database/KrisTott2018Fig3F_seaf_upper.csv'))

n       = len(tclim)
Pclim   = PCO2Func(tclim)
Tclim   = TempFunc(tclim)

P_H2O = cl.PH2O(T)
xclim = Pclim / (Pclim + P_H2O + 1)

con_area = 93 # Fekete (2002) # 55 # Gaillardet+1999 
sea_area = 147 # Johnson+2003 # 268

co_walk = cl.cont_walk1981(Tclim,Pclim)
co_kris = contFunc(tclim)
cu_kris = coupFunc(tclim)
cl_kris = coloFunc(tclim)

se_brad = cl.seaf_brad1997(Tclim,Pclim)
se_kris = seafFunc(tclim)
su_kris = seupFunc(tclim)
sl_kris = np.zeros(len(tclim))

argxCO2 = np.array([xclim,Tclim,P0*np.ones(n)]).T
argxCO2c1 = np.array([xclim,Tclim,P0*np.ones(n),1.1*np.ones(n),1e5*np.ones(n)]).T
argxCO2c2 = np.array([xclim,Tclim,P0*np.ones(n),Lc*np.ones(n),0*np.ones(n)]).T
argxCO2s1 = np.array([xclim,Tclim,P0*np.ones(n),18*np.ones(n),5e7*np.ones(n)]).T
argxCO2s2 = np.array([xclim,Tclim,P0*np.ones(n),Ls*np.ones(n),0*np.ones(n)]).T

Qcont = 0.3*np.ones(n) # tr.q_cont(Tclim)
Qseaf = 0.05*np.ones(n)

argHCO3_gran1 = np.array([DICeqFuncs['grah']['HCO3'](argxCO2),\
                         DwFuncs['grah'](argxCO2c1),Qcont]).T
argDICtr_gran1 = np.array([HCO3Funcs(argHCO3_gran1), xclim, Tclim, P0*np.ones(n)]).T

argHCO3_gran2 = np.array([DICeqFuncs['grah']['HCO3'](argxCO2),\
                         DwFuncs['grah'](argxCO2c2),Qcont]).T
argDICtr_gran2 = np.array([HCO3Funcs(argHCO3_gran2), xclim, Tclim, P0*np.ones(n)]).T

argHCO3_basa1 = np.array([DICeqFuncs['bash']['HCO3'](argxCO2),\
                         DwFuncs['bash'](argxCO2s1),Qseaf]).T
argDICtr_basa1 = np.array([HCO3Funcs(argHCO3_basa1), xclim, Tclim, P0*np.ones(n)]).T

argHCO3_basa2 = np.array([DICeqFuncs['bash']['HCO3'](argxCO2),\
                         DwFuncs['bash'](argxCO2s2),Qseaf]).T
argDICtr_basa2 = np.array([HCO3Funcs(argHCO3_basa2), xclim, Tclim, P0*np.ones(n)]).T

co_dicm = con_area*tr.w_flux(DICeqFuncs['grah']['ALK'] (argxCO2),Qcont)
co_hcom = con_area*tr.w_flux(DICeqFuncs['grah']['HCO3'](argxCO2),Qcont)
co_dic1 = con_area*tr.w_flux(DICtrFuncs['ALK'] (argDICtr_gran1),Qcont)
co_hco1 = con_area*tr.w_flux(DICtrFuncs['HCO3'](argDICtr_gran1),Qcont)
co_dic2 = con_area*tr.w_flux(DICtrFuncs['ALK'] (argDICtr_gran2),Qcont)
co_hco2 = con_area*tr.w_flux(DICtrFuncs['HCO3'](argDICtr_gran2),Qcont)

se_dicm = sea_area*tr.w_flux(DICeqFuncs['bash']['ALK'] (argxCO2),Qseaf)
se_hcom = sea_area*tr.w_flux(DICeqFuncs['bash']['HCO3'](argxCO2),Qseaf)
se_dic1 = sea_area*tr.w_flux(DICtrFuncs['ALK'] (argDICtr_basa1),Qseaf)
se_hco1 = sea_area*tr.w_flux(DICtrFuncs['HCO3'](argDICtr_basa1),Qseaf)
se_dic2 = sea_area*tr.w_flux(DICtrFuncs['ALK'] (argDICtr_basa2),Qseaf)
se_hco2 = sea_area*tr.w_flux(DICtrFuncs['HCO3'](argDICtr_basa2),Qseaf)

# print(DwFuncs['grah'](argxCO2c2)/Qcont)
# print(DwFuncs['bash'](argxCO2s2)/Qseaf)

i_co = np.where((co_hcom > 2 * co_hco2) == True)[0][0]
# i_se = np.where((se_hcom > 2 * se_hco2) == True)[0][0]

fig8 = plt.figure(figsize=(10,10))
spec8 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig8)
p81 = fig8.add_subplot(spec8[0,0])
p82 = fig8.add_subplot(spec8[0,1])
p83 = fig8.add_subplot(spec8[1,0])
p84 = fig8.add_subplot(spec8[1,1])
p81.text(-0.1, 1.15, '(a)', transform=p81.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p82.text(-0.1, 1.15, '(b)', transform=p82.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p83.text(-0.1, 1.15, '(c)', transform=p83.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p84.text(-0.1, 1.15, '(d)', transform=p84.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')

l811,   =   p81.plot(tclim,Tclim,lw=wid_mine,c=col_quar,ls=lwi_mine,label='Krissansen-Totton et al. (2018)')

p81.set_ylabel('$T$ [K]',fontsize=20)
# p81.set_yscale('log')
p81.set_ylim([280,300])
p81.set_xlabel('$t$ [Ga]',fontsize=20)
p81.legend(prop={'size':12})

l821,   =   p82.plot(tclim,Pclim,lw=wid_mine,c=col_quar,ls=lwi_mine,label='Krissansen-Totton et al. (2018)')

p82.set_ylabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=20)
p82.set_yscale('log')
p82.set_ylim([1e-4,3e-1])
p82.set_xlabel('$t$ [Ga]',fontsize=20)
p82.legend(prop={'size':12})

p83.errorbar(0,Fcstar,yerr=[[Fclowl],[Fcuppl]],linewidth=wid_pres,color=col_cont)
# l831,   =   p83.plot(tclim,co_dicm,lw=wid_mine,c=col_faya,ls=lwi_mine)
# l832,   =   p83.plot(tclim,co_hcom,lw=wid_mine,c=col_faya,ls=lwi_rock)
l831,   =   p83.plot(tclim,co_dic1,lw=wid_mine,c=col_gran,ls=lwi_mine)
l832,   =   p83.plot(tclim,co_hco1,lw=wid_mine,c=col_gran,ls=lwi_rock)
l833,   =   p83.plot(tclim,co_dic2,lw=wid_mine,c=col_walk,ls=lwi_mine)
l834,   =   p83.plot(tclim,co_hco2,lw=wid_mine,c=col_walk,ls=lwi_rock)
l835,   =   p83.plot(tclim,co_walk,lw=wid_walk,c=col_woll,ls=lwo_mine)
l836,   =   p83.plot(tclim,co_kris,lw=wid_walk,c=col_quar,ls=lwo_mine)
l837    =   p83.fill_between(tclim,cl_kris,cu_kris,facecolor=col_quar,alpha=0.2,interpolate=True)

p83.scatter(tclim[i_co],co_dic2[i_co],marker='o',s=80,color=col_walk)
p83.scatter(tclim[i_co],co_hco2[i_co],marker='o',s=80,color=col_walk)

p83.set_ylabel('$W_{\mathrm{cont}}$ [Tmol yr$^{-1}$]',fontsize=20)
p83.set_yscale('log')
p83.set_ylim([2e-1,5e3])
p83.set_xlabel('$t$ [Ga]',fontsize=20)

p83.text(tclim[20], co_hco1[20]*0.5, 'Supply', rotation=0, color=col_gran, ha='center', va='bottom',fontsize=12)
p83.text(tclim[12], co_dic2[12]*1.2, 'Kinetic', rotation=0, color=col_walk, ha='center', va='bottom',fontsize=12)
p83.text(tclim[2], co_dic2[2]*1.5, 'Thermo.', rotation=0, color=col_walk, ha='center', va='bottom',fontsize=12)

p84.errorbar(0,Fsstar,yerr=[[Fslowl],[Fsuppl]],linewidth=wid_pres,color=col_seaf)
# l841,   =   p84.plot(tclim,se_dicm,lw=wid_mine,c=col_grun,ls=lwi_mine)
# l842,   =   p84.plot(tclim,se_hcom,lw=wid_mine,c=col_grun,ls=lwi_rock)
l841,   =   p84.plot(tclim,se_dic1,lw=wid_mine,c=col_basa,ls=lwi_mine)
l842,   =   p84.plot(tclim,se_hco1,lw=wid_mine,c=col_basa,ls=lwi_rock)
l843,   =   p84.plot(tclim,se_dic2,lw=wid_mine,c=col_brad,ls=lwi_mine)
l844,   =   p84.plot(tclim,se_hco2,lw=wid_mine,c=col_brad,ls=lwi_rock)
l845,   =   p84.plot(tclim,se_brad,lw=wid_walk,c=col_woll,ls=lwo_mine)
l846,   =   p84.plot(tclim,se_kris,lw=wid_walk,c=col_quar,ls=lwo_mine)
l847    =   p84.fill_between(tclim,sl_kris,su_kris,facecolor=col_quar,alpha=0.2,interpolate=True)

# p84.scatter(tclim[i_se],se_dic2[i_se],marker='o',s=80,color=col_brad)
# p84.scatter(tclim[i_se],se_hco2[i_se],marker='o',s=80,color=col_brad)

p84.set_ylabel('$W_{\mathrm{seaf}}$ [Tmol yr$^{-1}$]',fontsize=20)
p84.set_yscale('log')
p84.set_ylim([2e-1,5e3])
p84.set_xlabel('$t$ [Ga]',fontsize=20)

p84.text(tclim[20], se_hco1[20]*0.5, 'Supply', rotation=0, color=col_basa, ha='center', va='bottom',fontsize=12)
# p84.text(tclim[25], se_dic2[25]*1.3, 'Kinetic', rotation=0, color=col_brad, ha='center', va='bottom',fontsize=12)
p84.text(tclim[20], se_dic2[20]*1.5, 'Thermo.', rotation=0, color=col_brad, ha='center', va='bottom',fontsize=12)

fig8.legend([l833,l831,l835,(l837,l836)],\
            ['This work (Granite, Young soils)',\
             'This work (Granite, Old soils)',\
             'Walker et al. (1981)','Krissansen-Totton et al. (2018)'],\
            frameon=True,prop={'size':12},\
            bbox_to_anchor=(0.1,0.44),loc='upper left',bbox_transform=fig8.transFigure,\
            ncol=1,handlelength=2)
fig8.legend([l843,l841,l845,(l847,l846)],\
            ['This work (Basalt, Young pore-space)',\
             'This work (Basalt, Old pore-space)',\
             'Brady \& Gislason (1997)','Krissansen-Totton et al. (2018)'],\
            frameon=True,prop={'size':12},\
            bbox_to_anchor=(0.59,0.31),loc='upper left',bbox_transform=fig8.transFigure,\
            ncol=1,handlelength=2)


custom_lines2 = [Line2D([0], [0], c=col_woll, lw=wid_mine, ls=lwi_mine),
                 Line2D([0], [0], c=col_woll, lw=wid_mine, ls=lwi_rock)]

fig8.legend(custom_lines2, ['$A$','[HCO$_3^-$]'],\
            frameon=True,prop={'size':12},\
            bbox_to_anchor=(0.1,0.08),loc='lower left',\
            bbox_transform=fig8.transFigure,handlelength=2)

plt.savefig('earth_weath.pdf',format='pdf',bbox_inches='tight')

# %%
# Fig. A1: Plot Keq

figB = plt.figure(figsize=(11,10))
specB = gridspec.GridSpec(ncols=2, nrows=2, figure=figB)
pB2 = figB.add_subplot(specB[0,0])
pB1 = figB.add_subplot(specB[0,1])
pB2.text(-0.1, 1.15, '(a)', transform=pB2.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
pB1.text(-0.1, 1.15, '(b)', transform=pB1.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')

pB1.text(0.9, 0.94, '$T = 278$ K', transform=pB1.transAxes, va='top', ha='right')
pB2.text(0.9, 0.94, '$P = 1$ bar', transform=pB2.transAxes, va='top', ha='right')

lB11,   =   pB1.plot(P,KeqFuncs['anoh'](np.full_like(P, T278),P),lw=wid_mine,c=col_anor,ls=lwi_mine)
lB12,   =   pB1.plot(P,KeqFuncs['fors'](np.full_like(P, T278),P),lw=wid_mine,c=col_fors,ls=lwi_mine)
lB13,   =   pB1.plot(P,KeqFuncs['woll'](np.full_like(P, T278),P),lw=wid_mine,c=col_woll,ls=lwi_mine)
lB14,   =   pB1.plot(P,KeqFuncs['co2a'](np.full_like(P, T278),P),lw=wid_mine,c=col_co2a,ls=lwi_mine)
lB15,   =   pB1.plot(P,KeqFuncs['enst'](np.full_like(P, T278),P),lw=wid_mine,c=col_enst,ls=lwi_mine)
lB16,   =   pB1.plot(P,KeqFuncs['mush'](np.full_like(P, T278),P),lw=wid_mine,c=col_musc,ls=lwi_mine)
lB18,   =   pB1.plot(P,KeqFuncs['anth'](np.full_like(P, T278),P),lw=wid_mine,c=col_anth,ls=lwi_mine)
lB17,   =   pB1.plot(P,KeqFuncs['quar'](np.full_like(P, T278),P),lw=wid_mine,c=col_quar,ls=lwi_mine)
lB19,   =   pB1.plot(P,KeqFuncs['faya'](np.full_like(P, T278),P),lw=wid_mine,c=col_faya,ls=lwi_mine)
lB110,  =   pB1.plot(P,KeqFuncs['phlh'](np.full_like(P, T278),P),lw=wid_mine,c=col_phlo,ls=lwi_mine)
lB111,  =   pB1.plot(P,KeqFuncs['ferr'](np.full_like(P, T278),P),lw=wid_mine,c=col_ferr,ls=lwi_mine)
lB112,  =   pB1.plot(P,KeqFuncs['bica'](np.full_like(P, T278),P),lw=wid_mine,c=col_bica,ls=lwi_mine)
lB113,  =   pB1.plot(P,KeqFuncs['annh'](np.full_like(P, T278),P),lw=wid_mine,c=col_anni,ls=lwi_mine)
lB114,  =   pB1.plot(P,KeqFuncs['grun'](np.full_like(P, T278),P),lw=wid_mine,c=col_grun,ls=lwi_mine)
lB115,  =   pB1.plot(P,KeqFuncs['albh'](np.full_like(P, T278),P),lw=wid_mine,c=col_albi,ls=lwi_mine)
lB116,  =   pB1.plot(P,KeqFuncs['carb'](np.full_like(P, T278),P),lw=wid_mine,c=col_carb,ls=lwi_mine)
lB117,  =   pB1.plot(P,KeqFuncs['kfeh'](np.full_like(P, T278),P),lw=wid_mine,c=col_kfel,ls=lwi_mine)
lB118,  =   pB1.plot(P,KeqFuncs['wate'](np.full_like(P, T278),P),lw=wid_mine,c=col_wate,ls=lwi_mine)
pB1.set_xscale('log')
pB1.set_yscale('log')
pB1.set_ylim([1e-15,1e2])
pB1.set_xlabel('$P$ [bar]',fontsize=20)
pB1.set_ylabel('$K$',fontsize=20)
pB1.set_xticks(np.logspace(-2,3,num=6))

lB21,   =   pB2.plot(T,KeqFuncs['anoh'](T,np.full_like(T,P0)),lw=wid_mine,c=col_anor,ls=lwi_mine)
lB22,   =   pB2.plot(T,KeqFuncs['fors'](T,np.full_like(T,P0)),lw=wid_mine,c=col_fors,ls=lwi_mine)
lB23,   =   pB2.plot(T,KeqFuncs['woll'](T,np.full_like(T,P0)),lw=wid_mine,c=col_woll,ls=lwi_mine)
lB24,   =   pB2.plot(T,KeqFuncs['co2a'](T,np.full_like(T,P0)),lw=wid_mine,c=col_co2a,ls=lwi_mine)
lB25,   =   pB2.plot(T,KeqFuncs['enst'](T,np.full_like(T,P0)),lw=wid_mine,c=col_enst,ls=lwi_mine)
lB26,   =   pB2.plot(T,KeqFuncs['mush'](T,np.full_like(T,P0)),lw=wid_mine,c=col_musc,ls=lwi_mine)
lB28,   =   pB2.plot(T,KeqFuncs['anth'](T,np.full_like(T,P0)),lw=wid_mine,c=col_anth,ls=lwi_mine)
lB27,   =   pB2.plot(T,KeqFuncs['quar'](T,np.full_like(T,P0)),lw=wid_mine,c=col_quar,ls=lwi_mine)
lB29,   =   pB2.plot(T,KeqFuncs['faya'](T,np.full_like(T,P0)),lw=wid_mine,c=col_faya,ls=lwi_mine)
lB210,  =   pB2.plot(T,KeqFuncs['phlh'](T,np.full_like(T,P0)),lw=wid_mine,c=col_phlo,ls=lwi_mine)
lB211,  =   pB2.plot(T,KeqFuncs['ferr'](T,np.full_like(T,P0)),lw=wid_mine,c=col_ferr,ls=lwi_mine)
lB212,  =   pB2.plot(T,KeqFuncs['bica'](T,np.full_like(T,P0)),lw=wid_mine,c=col_bica,ls=lwi_mine)
lB213,  =   pB2.plot(T,KeqFuncs['annh'](T,np.full_like(T,P0)),lw=wid_mine,c=col_anni,ls=lwi_mine)
lB214,  =   pB2.plot(T,KeqFuncs['grun'](T,np.full_like(T,P0)),lw=wid_mine,c=col_grun,ls=lwi_mine)
lB215,  =   pB2.plot(T,KeqFuncs['albh'](T,np.full_like(T,P0)),lw=wid_mine,c=col_albi,ls=lwi_mine)
lB216,  =   pB2.plot(T,KeqFuncs['carb'](T,np.full_like(T,P0)),lw=wid_mine,c=col_carb,ls=lwi_mine)
lB217,  =   pB2.plot(T,KeqFuncs['kfeh'](T,np.full_like(T,P0)),lw=wid_mine,c=col_kfel,ls=lwi_mine)
lB218,  =   pB2.plot(T,KeqFuncs['wate'](T,np.full_like(T,P0)),lw=wid_mine,c=col_wate,ls=lwi_mine)
#pB2.set_xscale('log')
pB2.set_yscale('log')
pB2.set_ylim([1e-15,1e2])
pB2.set_xlabel('$T$ [K]',fontsize=20)
pB2.set_ylabel('$K$',fontsize=20)

figB.legend([lB11,lB12,lB13,lB14,lB15,lB16,lB17,lB18,lB19,lB110,lB111,lB112,lB113,\
              lB114,lB115,lB116,lB117,lB118], \
             ['Reaction (f) (Anorthite weathering)',    'Reaction (d) (Forsterite weathering)',\
              'Reaction (a) (Wollastonite weathering)', 'Reaction (o) (CO$_2$ dissolution)',\
              'Reaction (b) (Enstatite weathering)',    'Reaction (i) (Muscovite weathering)',\
              'Reaction (n) (Quartz dissolution)',      'Reaction (l) (Anthophyllite weathering)',\
              'Reaction (e) (Fayalite weathering)',     'Reaction (j) (Phlogopite weathering)',\
              'Reaction (c) (Ferrosilite weathering)',  'Reaction (p) (CO$_2$ dissociation)',\
              'Reaction (k) (Annite weathering)',       'Reaction (m) (Grunerite weathering)',\
              'Reaction (g) (Albite weathering)',       'Reaction (q) (Bicarbonate dissociation)',\
              'Reaction (h) (K-feldspar weathering)',   'Reaction (r) (Water dissociation)'],\
             frameon=True,prop={'size':16},bbox_to_anchor=(0.94,0.05),ncol=2,loc='lower right',\
             handlelength=2)

plt.savefig('K.pdf',format='pdf',bbox_inches='tight')

# %%
# Fig. A2: Plot k_eff

figA = plt.figure(figsize=(11,15))
specA = gridspec.GridSpec(ncols=2, nrows=3, figure=figA)
pA2 = figA.add_subplot(specA[0,0])
pA1 = figA.add_subplot(specA[0,1])
pA4 = figA.add_subplot(specA[1,0])
pA3 = figA.add_subplot(specA[1,1])
pA2.text(-0.1, 1.15, '(a)', transform=pA2.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
pA1.text(-0.1, 1.15, '(b)', transform=pA1.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
pA4.text(-0.1, 1.15, '(a)', transform=pA4.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
pA3.text(-0.1, 1.15, '(b)', transform=pA3.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')

pA2.text(0.55, 0.95, 'pH = 7', transform=pA2.transAxes,va='top', ha='right')
pA1.text(0.55, 0.95, '$T$ = 288 K', transform=pA1.transAxes, va='top', ha='right')
pA4.text(0.55, 0.95, 'pH = 10', transform=pA4.transAxes,va='top', ha='right')
pA3.text(0.55, 0.95, '$T$ = 348 K', transform=pA3.transAxes,va='top', ha='right')

argpH = T, np.full_like(T, 7)
argTk = np.array(np.meshgrid(T,7)).T[:,0]
argpH2 = np.array(np.meshgrid(348,pHfull)).T[0]
argTk2 = np.array(np.meshgrid(T,10)).T[:,0]

lA11,   =   pA1.plot(pHfull,kFuncs['woll'](np.full_like(pHfull, T288), pHfull),lw=wid_mine,c=col_woll,ls=lwi_mine)
lA12,   =   pA1.plot(pHfull,kFuncs['anor'](np.full_like(pHfull, T288), pHfull),lw=wid_mine,c=col_anor,ls=lwi_mine)
lA13,   =   pA1.plot(pHfull,kFuncs['fors'](np.full_like(pHfull, T288), pHfull),lw=wid_mine,c=col_fors,ls=lwi_mine)
lA14,   =   pA1.plot(pHfull,kFuncs['faya'](np.full_like(pHfull, T288), pHfull),lw=wid_mine,c=col_faya,ls=lwi_mine)
lA15,   =   pA1.plot(pHfull,kFuncs['albi'](np.full_like(pHfull, T288), pHfull),lw=wid_mine,c=col_albi,ls=lwi_mine)
lA16,   =   pA1.plot(pHfull,kFuncs['quar'](np.full_like(pHfull, T288), pHfull),lw=wid_mine,c=col_quar,ls=lwi_mine)
lA17,   =   pA1.plot(pHfull,kFuncs['enst'](np.full_like(pHfull, T288), pHfull),lw=wid_mine,c=col_enst,ls=lwi_mine)
lA18,   =   pA1.plot(pHfull,kFuncs['phlo'](np.full_like(pHfull, T288), pHfull),lw=wid_mine,c=col_phlo,ls=lwi_mine)
lA19,   =   pA1.plot(pHfull,kFuncs['kfel'](np.full_like(pHfull, T288), pHfull),lw=wid_mine,c=col_kfel,ls=lwi_mine)
lA110,  =   pA1.plot(pHfull,kFuncs['musc'](np.full_like(pHfull, T288), pHfull),lw=wid_mine,c=col_musc,ls=lwi_mine)
lA111,  =   pA1.plot(pHfull,kFuncs['anth'](np.full_like(pHfull, T288), pHfull),lw=wid_mine,c=col_anth,ls=lwi_mine)

pA1.set_yscale('log')
pA1.set_xlabel('pH',fontsize=20)
pA1.set_ylabel('$k_{\mathrm{eff}}$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=20)
pA1.set_ylim([1e-8,1e5])
pA1.set_xticks(np.linspace(0,14,num=8))

lA21,   =   pA2.plot(T,kFuncs['woll'](T, np.full_like(T, 7)),lw=wid_mine,c=col_woll,ls=lwi_mine)
lA22,   =   pA2.plot(T,kFuncs['anor'](T, np.full_like(T, 7)),lw=wid_mine,c=col_anor,ls=lwi_mine)
lA23,   =   pA2.plot(T,kFuncs['fors'](T, np.full_like(T, 7)),lw=wid_mine,c=col_fors,ls=lwi_mine)
lA24,   =   pA2.plot(T,kFuncs['faya'](T, np.full_like(T, 7)),lw=wid_mine,c=col_faya,ls=lwi_mine)
lA25,   =   pA2.plot(T,kFuncs['albi'](T, np.full_like(T, 7)),lw=wid_mine,c=col_albi,ls=lwi_mine)
lA26,   =   pA2.plot(T,kFuncs['quar'](T, np.full_like(T, 7)),lw=wid_mine,c=col_quar,ls=lwi_mine)
lA27,   =   pA2.plot(T,kFuncs['enst'](T, np.full_like(T, 7)),lw=wid_mine,c=col_enst,ls=lwi_mine)
lA28,   =   pA2.plot(T,kFuncs['phlo'](T, np.full_like(T, 7)),lw=wid_mine,c=col_phlo,ls=lwi_mine)
lA29,   =   pA2.plot(T,kFuncs['kfel'](T, np.full_like(T, 7)),lw=wid_mine,c=col_kfel,ls=lwi_mine)
lA210,  =   pA2.plot(T,kFuncs['musc'](T, np.full_like(T, 7)),lw=wid_mine,c=col_musc,ls=lwi_mine)
lA211,  =   pA2.plot(T,kFuncs['anth'](T, np.full_like(T, 7)),lw=wid_mine,c=col_anth,ls=lwi_mine)

pA2.set_yscale('log')
pA2.set_xlabel('$T$ [K]',fontsize=20)
pA2.set_ylabel('$k_{\mathrm{eff}}$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=20)
pA2.set_ylim([1e-8,1e5])

lA31,   =   pA3.plot(pHfull,kFuncs['woll'](np.full_like(pHfull, 348), pHfull),lw=wid_mine,c=col_woll,ls=lwi_mine)
lA32,   =   pA3.plot(pHfull,kFuncs['anor'](np.full_like(pHfull, 348), pHfull),lw=wid_mine,c=col_anor,ls=lwi_mine)
lA33,   =   pA3.plot(pHfull,kFuncs['fors'](np.full_like(pHfull, 348), pHfull),lw=wid_mine,c=col_fors,ls=lwi_mine)
lA34,   =   pA3.plot(pHfull,kFuncs['faya'](np.full_like(pHfull, 348), pHfull),lw=wid_mine,c=col_faya,ls=lwi_mine)
lA35,   =   pA3.plot(pHfull,kFuncs['albi'](np.full_like(pHfull, 348), pHfull),lw=wid_mine,c=col_albi,ls=lwi_mine)
lA36,   =   pA3.plot(pHfull,kFuncs['quar'](np.full_like(pHfull, 348), pHfull),lw=wid_mine,c=col_quar,ls=lwi_mine)
lA37,   =   pA3.plot(pHfull,kFuncs['enst'](np.full_like(pHfull, 348), pHfull),lw=wid_mine,c=col_enst,ls=lwi_mine)
lA38,   =   pA3.plot(pHfull,kFuncs['phlo'](np.full_like(pHfull, 348), pHfull),lw=wid_mine,c=col_phlo,ls=lwi_mine)
lA39,   =   pA3.plot(pHfull,kFuncs['kfel'](np.full_like(pHfull, 348), pHfull),lw=wid_mine,c=col_kfel,ls=lwi_mine)
lA310,  =   pA3.plot(pHfull,kFuncs['musc'](np.full_like(pHfull, 348), pHfull),lw=wid_mine,c=col_musc,ls=lwi_mine)
lA311,  =   pA3.plot(pHfull,kFuncs['anth'](np.full_like(pHfull, 348), pHfull),lw=wid_mine,c=col_anth,ls=lwi_mine)

pA3.set_yscale('log')
pA3.set_xlabel('pH',fontsize=20)
pA3.set_ylabel('$k_{\mathrm{eff}}$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=20)
pA3.set_ylim([1e-8,1e5])
pA3.set_xticks(np.linspace(0,14,num=8))

lA41,   =   pA4.plot(T,kFuncs['woll'](T, np.full_like(T, 10)),lw=wid_mine,c=col_woll,ls=lwi_mine)
lA42,   =   pA4.plot(T,kFuncs['anor'](T, np.full_like(T, 10)),lw=wid_mine,c=col_anor,ls=lwi_mine)
lA43,   =   pA4.plot(T,kFuncs['fors'](T, np.full_like(T, 10)),lw=wid_mine,c=col_fors,ls=lwi_mine)
lA44,   =   pA4.plot(T,kFuncs['faya'](T, np.full_like(T, 10)),lw=wid_mine,c=col_faya,ls=lwi_mine)
lA45,   =   pA4.plot(T,kFuncs['albi'](T, np.full_like(T, 10)),lw=wid_mine,c=col_albi,ls=lwi_mine)
lA46,   =   pA4.plot(T,kFuncs['quar'](T, np.full_like(T, 10)),lw=wid_mine,c=col_quar,ls=lwi_mine)
lA47,   =   pA4.plot(T,kFuncs['enst'](T, np.full_like(T, 10)),lw=wid_mine,c=col_enst,ls=lwi_mine)
lA48,   =   pA4.plot(T,kFuncs['phlo'](T, np.full_like(T, 10)),lw=wid_mine,c=col_phlo,ls=lwi_mine)
lA49,   =   pA4.plot(T,kFuncs['kfel'](T, np.full_like(T, 10)),lw=wid_mine,c=col_kfel,ls=lwi_mine)
lA410,  =   pA4.plot(T,kFuncs['musc'](T, np.full_like(T, 10)),lw=wid_mine,c=col_musc,ls=lwi_mine)
lA411,  =   pA4.plot(T,kFuncs['anth'](T, np.full_like(T, 10)),lw=wid_mine,c=col_anth,ls=lwi_mine)

pA4.set_yscale('log')
pA4.set_xlabel('$T$ [K]',fontsize=20)
pA4.set_ylabel('$k_{\mathrm{eff}}$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=20)
pA4.set_ylim([1e-8,1e5])

figA.legend([lA11,lA12,lA13,lA14,lA15,lA16,lA17,lA18,lA19,lA110,lA111], \
             ['Wollastonite (CaSiO$_3$)','Anorthite (CaAl$_2$Si$_2$O$_8$)','Forsterite (Mg$_2$SiO$_4$)',\
              'Fayalite (Fe$_2$SiO$_4$)','Albite (NaAlSi$_3$O$_8$)','Quartz (SiO$_2$)',\
              'Enstatite (MgSiO$_3$)','Phlogopite (KMg$_3$AlSi$_3$O$_{10}$(OH)$_2$)',\
              'K-Feldspar (KAlSi$_3$O$_8$)','Muscovite (KAl$_3$Si$_3$O$_{10}$(OH)$_2$)',\
              'Anthophyllite (Mg$_7$Si$_8$O$_{22}$(OH)$_2$)'\
              ],frameon=True,prop={'size':18},\
             bbox_to_anchor=(0.9,0.06),loc='lower right',ncol=2,handlelength=2)

plt.savefig('k_eff.pdf',format='pdf',bbox_inches='tight')

# %%
# Fig. B1: Plot HCO3- vs xCO2, T and P for peridotite

fig3 = plt.figure(figsize=(5,15))
spec3 = gridspec.GridSpec(ncols=1, nrows=3, figure=fig3)

p31 = fig3.add_subplot(spec3[0,0])
p32 = fig3.add_subplot(spec3[1,0])
p33 = fig3.add_subplot(spec3[2,0])

p31.text(-0.1, 1.15, '(a)', transform=p31.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p32.text(-0.1, 1.15, '(b)', transform=p32.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p33.text(-0.1, 1.15, '(c)', transform=p33.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')

xCO2    = np.logspace(-8,0)

argxCO2a = np.array([xCO2,T278*np.ones(len(xCO2)),P1*np.ones(len(xCO2))]).T
argxCO2b = np.array([xCO2,T278*np.ones(len(xCO2)),P0*np.ones(len(xCO2))]).T
argxCO2c = np.array([xCO2,T278*np.ones(len(xCO2)),P2*np.ones(len(xCO2))]).T
argxCO2d = np.array([xCO2,T348*np.ones(len(xCO2)),P0*np.ones(len(xCO2))]).T
argxCO2e = np.array([xCO2,T348*np.ones(len(xCO2)),P2*np.ones(len(xCO2))]).T

argTa = np.array([xCO2g0*np.ones(len(T)),T,P0*np.ones(len(T))]).T
argTb = np.array([xCO2g0*np.ones(len(T)),T,P2*np.ones(len(T))]).T
argTc = np.array([xCO2g1*np.ones(len(T)),T,P0*np.ones(len(T))]).T
argTd = np.array([xCO2g1*np.ones(len(T)),T,P2*np.ones(len(T))]).T

argPa = np.array([xCO2g0*np.ones(len(P)),T278*np.ones(len(P)),P]).T
argPb = np.array([xCO2g1*np.ones(len(P)),T278*np.ones(len(P)),P]).T
argPc = np.array([xCO2g0*np.ones(len(P1bar)),T348*np.ones(len(P1bar)),P1bar]).T
argPd = np.array([xCO2g1*np.ones(len(P1bar)),T348*np.ones(len(P1bar)),P1bar]).T

l312,   =   p31.plot(xCO2,DICeqFuncs['peri']['HCO3'](argxCO2b),lw=wid_mine,c=col_peri,ls=lwi_mine)
l313,   =   p31.plot(xCO2,DICeqFuncs['peri']['HCO3'](argxCO2c),lw=wid_mine,c=col_peri,ls=lwi_rock)
l314,   =   p31.plot(xCO2,DICeqFuncs['peri']['HCO3'](argxCO2d),lw=wid_mine,c=col_quar,ls=lwi_mine)
l315,   =   p31.plot(xCO2,DICeqFuncs['peri']['HCO3'](argxCO2e),lw=wid_mine,c=col_quar,ls=lwi_rock)
l311,   =   p31.plot(xCO2,DICeqFuncs['peri']['HCO3'](argxCO2a),lw=wid_mine,c=col_peri,ls=lwo_mine)
p31.set_xscale('log')
p31.set_yscale('log')
p31.set_ylim([1e-7,5e2])
p31.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=20)
p31.set_ylabel('[HCO$^-_3$]$_{\mathrm{eq}}$ [mol dm$^{-3}$]',fontsize=20)
p31.set_title('Peridotite Weathering', fontsize=18)
p31.set_xticks(np.logspace(-8,0,num=5))

l321,   =   p32.plot(T,DICeqFuncs['peri']['HCO3'](argTa),lw=wid_mine,c=col_peri,ls=lwi_mine)
l322,   =   p32.plot(T,DICeqFuncs['peri']['HCO3'](argTb),lw=wid_mine,c=col_peri,ls=lwi_rock)
l323,   =   p32.plot(T,DICeqFuncs['peri']['HCO3'](argTc),lw=wid_mine,c=col_quar,ls=lwi_mine)
l324,   =   p32.plot(T,DICeqFuncs['peri']['HCO3'](argTd),lw=wid_mine,c=col_quar,ls=lwi_rock)
p32.set_yscale('log')
p32.set_ylim([1e-7,5e2])
p32.set_xlabel('$T$ [K]',fontsize=20)
p32.set_ylabel('[HCO$^-_3$]$_{\mathrm{eq}}$ [mol dm$^{-3}$]',fontsize=20)

l331,   =   p33.plot(P,DICeqFuncs['peri']['HCO3'](argPa),lw=wid_mine,c=col_peri,ls=lwi_mine)
l332,   =   p33.plot(P,DICeqFuncs['peri']['HCO3'](argPb),lw=wid_mine,c=col_peri,ls=lwi_rock)
l333,   =   p33.plot(P1bar,DICeqFuncs['peri']['HCO3'](argPc),lw=wid_mine,c=col_quar,ls=lwi_mine)
l334,   =   p33.plot(P1bar,DICeqFuncs['peri']['HCO3'](argPd),lw=wid_mine,c=col_quar,ls=lwi_rock)
p33.set_yscale('log')
p33.set_xscale('log')
p33.set_ylim([1e-7,5e2])
p33.set_xlabel('$P$ [bar]',fontsize=20)
p33.set_ylabel('[HCO$^-_3$]$_{\mathrm{eq}}$ [mol dm$^{-3}$]',fontsize=20)
p33.set_xticks(np.logspace(-2,3,num=6))

fig3.legend([l311,l312,l313,l314,l315], ['$T = 278$ K, $P = 0.01$ bar',\
                                         '$T = 278$ K, $P = 1$ bar',\
                                         '$T = 278$ K, $P = 10^{3}$ bar',\
                                         '$T = 348$ K, $P = 1$ bar',\
                                         '$T = 348$ K, $P = 10^{3}$ bar'],\
        frameon=True,prop={'size':12},bbox_to_anchor=(0.185,0.94),loc='upper left',handlelength=2)

fig3.legend([l321,l322,l323,l324], ['$P = 1$ bar,    $P_{\mathrm{CO}_2} = 280$ $\mu$bar',\
                                    '$P = 10^3$ bar, $P_{\mathrm{CO}_2} = 280$ $\mu$bar',\
                                    '$P = 1$ bar,    $P_{\mathrm{CO}_2} = 1$ bar',\
                                    '$P = 10^3$ bar, $P_{\mathrm{CO}_2} = 1$ bar'],\
        frameon=True,prop={'size':12},bbox_to_anchor=(0.185,0.38),loc='lower left',handlelength=2)

fig3.legend([l331,l332,l333,l334], ['$T = 278$ K, $P_{\mathrm{CO}_2} = 280$ $\mu$bar',\
                                    '$T = 278$ K, $P_{\mathrm{CO}_2} = 1$ bar',\
                                    '$T = 348$ K, $P_{\mathrm{CO}_2} = 280$ $\mu$bar',\
                                    '$T = 348$ K, $P_{\mathrm{CO}_2} = 1$ bar'],\
        frameon=True,prop={'size':12},bbox_to_anchor=(0.185,0.05),loc='lower left',handlelength=2)

plt.savefig('HCO3eq_all_peri.pdf',format='pdf',bbox_inches='tight')

# %%
# Fig. B3: Plot generalized DIC vs xCO2 for all rocks and minerals

fig2 = plt.figure(figsize=(15,20))
spec2 = gridspec.GridSpec(ncols=3, nrows=6, figure=fig2)

p21  = fig2.add_subplot(spec2[0,0])
p22  = fig2.add_subplot(spec2[0,1])
p23  = fig2.add_subplot(spec2[0,2])
p24  = fig2.add_subplot(spec2[1,0])
p25  = fig2.add_subplot(spec2[1,1])
p26  = fig2.add_subplot(spec2[1,2])
p27  = fig2.add_subplot(spec2[2,0])
p28  = fig2.add_subplot(spec2[2,1])
p29  = fig2.add_subplot(spec2[2,2])
p210 = fig2.add_subplot(spec2[3,0])
p211 = fig2.add_subplot(spec2[3,1])
p212 = fig2.add_subplot(spec2[3,2])
p213 = fig2.add_subplot(spec2[4,0])
p214 = fig2.add_subplot(spec2[4,1])
p215 = fig2.add_subplot(spec2[4,2])
p216 = fig2.add_subplot(spec2[5,0])

p21.text(-0.1, 1.15, '(a)', transform=p21.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p22.text(-0.1, 1.15, '(b)', transform=p22.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p23.text(-0.1, 1.15, '(c)', transform=p23.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p24.text(-0.1, 1.15, '(d)', transform=p24.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p25.text(-0.1, 1.15, '(e)', transform=p25.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p26.text(-0.1, 1.15, '(f)', transform=p26.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p27.text(-0.1, 1.15, '(g)', transform=p27.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p28.text(-0.1, 1.15, '(h)', transform=p28.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p29.text(-0.1, 1.15, '(i)', transform=p29.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p210.text(-0.1, 1.15, '(j)', transform=p210.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p211.text(-0.1, 1.15, '(k)', transform=p211.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p212.text(-0.1, 1.15, '(l)', transform=p212.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p213.text(-0.1, 1.15, '(m)', transform=p213.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p214.text(-0.1, 1.15, '(n)', transform=p214.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p215.text(-0.1, 1.15, '(o)', transform=p215.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p216.text(-0.1, 1.15, '(p)', transform=p216.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')

xCO2     = np.logspace(-8,0)

arg      = np.array([xCO2,T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2))]).T

q0 = 0.3

arg_peri = np.array([xCO2,T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2)),\
                           Lc*np.ones(len(xCO2)),np.zeros(len(xCO2))]).T
arg_basa = np.array([xCO2,T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2)),\
                           Lc*np.ones(len(xCO2)),np.zeros(len(xCO2))]).T
arg_gran = np.array([xCO2,T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2)),\
                           Lc*np.ones(len(xCO2)),np.zeros(len(xCO2))]).T
arg_anor = np.array([xCO2,T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2)),\
                           Lc*np.ones(len(xCO2)),np.zeros(len(xCO2))]).T
arg_albi = np.array([xCO2,T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2)),\
                           Lc*np.ones(len(xCO2)),np.zeros(len(xCO2))]).T
arg_kfel = np.array([xCO2,T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2)),\
                           Lc*np.ones(len(xCO2)),np.zeros(len(xCO2))]).T
arg_woll = np.array([xCO2,T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2)),\
                           Lc*np.ones(len(xCO2)),np.zeros(len(xCO2))]).T
arg_enst = np.array([xCO2,T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2)),\
                           Lc*np.ones(len(xCO2)),np.zeros(len(xCO2))]).T
arg_ferr = np.array([xCO2,T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2)),\
                           Lc*np.ones(len(xCO2)),np.zeros(len(xCO2))]).T
arg_fors = np.array([xCO2,T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2)),\
                           Lc*np.ones(len(xCO2)),np.zeros(len(xCO2))]).T
arg_faya = np.array([xCO2,T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2)),\
                           Lc*np.ones(len(xCO2)),np.zeros(len(xCO2))]).T
arg_phlo = np.array([xCO2,T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2)),\
                           Lc*np.ones(len(xCO2)),np.zeros(len(xCO2))]).T
arg_musc = np.array([xCO2,T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2)),\
                           Lc*np.ones(len(xCO2)),np.zeros(len(xCO2))]).T
arg_anni = np.array([xCO2,T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2)),\
                           Lc*np.ones(len(xCO2)),np.zeros(len(xCO2))]).T
arg_anth = np.array([xCO2,T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2)),\
                           Lc*np.ones(len(xCO2)),np.zeros(len(xCO2))]).T
arg_grun = np.array([xCO2,T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2)),\
                           Lc*np.ones(len(xCO2)),np.zeros(len(xCO2))]).T

Dw_peri = DwFuncs['peri'](arg_peri)
Dw_basa = DwFuncs['bash'](arg_basa)
Dw_gran = DwFuncs['grah'](arg_gran)
Dw_anor = DwFuncs['anoh'](arg_anor)
Dw_albi = DwFuncs['albh'](arg_albi)
Dw_kfel = DwFuncs['kfeh'](arg_kfel)
Dw_woll = DwFuncs['woll'](arg_woll)
Dw_enst = DwFuncs['enst'](arg_enst)
Dw_ferr = DwFuncs['ferr'](arg_ferr)
Dw_fors = DwFuncs['fors'](arg_fors)
Dw_faya = DwFuncs['faya'](arg_faya)
Dw_phlo = DwFuncs['phlh'](arg_phlo)
Dw_musc = DwFuncs['mush'](arg_musc)
Dw_anni = DwFuncs['annh'](arg_anni)
Dw_anth = DwFuncs['anth'](arg_anth)
Dw_grun = DwFuncs['grun'](arg_grun)

HCO3eq_peri = DICeqFuncs['peri']['HCO3'](arg)
HCO3eq_basa = DICeqFuncs['bash']['HCO3'](arg)
HCO3eq_gran = DICeqFuncs['grah']['HCO3'](arg)
HCO3eq_anor = DICeqFuncs['anoh']['HCO3'](arg)
HCO3eq_albi = DICeqFuncs['albh']['HCO3'](arg)
HCO3eq_kfel = DICeqFuncs['kfeh']['HCO3'](arg)
HCO3eq_woll = DICeqFuncs['woll']['HCO3'](arg)
HCO3eq_enst = DICeqFuncs['enst']['HCO3'](arg)
HCO3eq_ferr = DICeqFuncs['ferr']['HCO3'](arg)
HCO3eq_fors = DICeqFuncs['fors']['HCO3'](arg)
HCO3eq_faya = DICeqFuncs['faya']['HCO3'](arg)
HCO3eq_phlo = DICeqFuncs['phlh']['HCO3'](arg)
HCO3eq_musc = DICeqFuncs['mush']['HCO3'](arg)
HCO3eq_anni = DICeqFuncs['annh']['HCO3'](arg)
HCO3eq_anth = DICeqFuncs['anth']['HCO3'](arg)
HCO3eq_grun = DICeqFuncs['grun']['HCO3'](arg)

argHCO3_peri = np.array([HCO3eq_peri,Dw_peri,q0*np.ones(len(xCO2))]).T
argHCO3_basa = np.array([HCO3eq_basa,Dw_basa,q0*np.ones(len(xCO2))]).T
argHCO3_gran = np.array([HCO3eq_gran,Dw_gran,q0*np.ones(len(xCO2))]).T
argHCO3_anor = np.array([HCO3eq_anor,Dw_anor,q0*np.ones(len(xCO2))]).T
argHCO3_albi = np.array([HCO3eq_albi,Dw_albi,q0*np.ones(len(xCO2))]).T
argHCO3_kfel = np.array([HCO3eq_kfel,Dw_kfel,q0*np.ones(len(xCO2))]).T
argHCO3_woll = np.array([HCO3eq_woll,Dw_woll,q0*np.ones(len(xCO2))]).T
argHCO3_enst = np.array([HCO3eq_enst,Dw_enst,q0*np.ones(len(xCO2))]).T
argHCO3_ferr = np.array([HCO3eq_ferr,Dw_ferr,q0*np.ones(len(xCO2))]).T
argHCO3_fors = np.array([HCO3eq_fors,Dw_fors,q0*np.ones(len(xCO2))]).T
argHCO3_faya = np.array([HCO3eq_faya,Dw_faya,q0*np.ones(len(xCO2))]).T
argHCO3_phlo = np.array([HCO3eq_phlo,Dw_phlo,q0*np.ones(len(xCO2))]).T
argHCO3_musc = np.array([HCO3eq_musc,Dw_musc,q0*np.ones(len(xCO2))]).T
argHCO3_anni = np.array([HCO3eq_anni,Dw_anni,q0*np.ones(len(xCO2))]).T
argHCO3_anth = np.array([HCO3eq_anth,Dw_anth,q0*np.ones(len(xCO2))]).T
argHCO3_grun = np.array([HCO3eq_grun,Dw_grun,q0*np.ones(len(xCO2))]).T

argDICtr_peri = np.array([HCO3Funcs(argHCO3_peri), xCO2, T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2))]).T
argDICtr_basa = np.array([HCO3Funcs(argHCO3_basa), xCO2, T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2))]).T
argDICtr_gran = np.array([HCO3Funcs(argHCO3_gran), xCO2, T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2))]).T
argDICtr_anor = np.array([HCO3Funcs(argHCO3_anor), xCO2, T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2))]).T
argDICtr_albi = np.array([HCO3Funcs(argHCO3_albi), xCO2, T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2))]).T
argDICtr_kfel = np.array([HCO3Funcs(argHCO3_kfel), xCO2, T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2))]).T
argDICtr_woll = np.array([HCO3Funcs(argHCO3_woll), xCO2, T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2))]).T
argDICtr_enst = np.array([HCO3Funcs(argHCO3_enst), xCO2, T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2))]).T
argDICtr_ferr = np.array([HCO3Funcs(argHCO3_ferr), xCO2, T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2))]).T
argDICtr_fors = np.array([HCO3Funcs(argHCO3_fors), xCO2, T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2))]).T
argDICtr_faya = np.array([HCO3Funcs(argHCO3_faya), xCO2, T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2))]).T
argDICtr_phlo = np.array([HCO3Funcs(argHCO3_phlo), xCO2, T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2))]).T
argDICtr_musc = np.array([HCO3Funcs(argHCO3_musc), xCO2, T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2))]).T
argDICtr_anni = np.array([HCO3Funcs(argHCO3_anni), xCO2, T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2))]).T
argDICtr_anth = np.array([HCO3Funcs(argHCO3_anth), xCO2, T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2))]).T
argDICtr_grun = np.array([HCO3Funcs(argHCO3_grun), xCO2, T288*np.ones(len(xCO2)),P0*np.ones(len(xCO2))]).T

p21a  = p21.twinx()
l215, = p21a.plot(xCO2,DICtrFuncs['pH'] (argDICtr_peri),lw=wid_mine,c=col_peri,ls=lwo_mine) 
l212, = p21.plot(xCO2,DICtrFuncs['HCO3'](argDICtr_peri),lw=wid_rock,c=col_bica,ls=lwi_mine)
l213, = p21.plot(xCO2,DICtrFuncs['CO3'] (argDICtr_peri),lw=wid_mine,c=col_carb,ls=lwi_mine)
l211, = p21.plot(xCO2,DICtrFuncs['ALK'] (argDICtr_peri),lw=wid_rock,c=col_peri,ls=lwi_rock)

p21.set_xscale('log')
p21.set_yscale('log')
p21.set_ylim([2e-13,2e1])
p21.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=16)
p21.set_ylabel('$C$ [mol dm$^{-3}$]',fontsize=16)
p21.set_title('Peridotite',c=col_peri,fontsize=20)
p21.set_xticks(np.logspace(-8,0,num=5))
p21.set_yticks(np.logspace(-12,0,num=5))
p21a.set_yticks(np.linspace(4,12,num=5))
p21a.set_ylim([3.5,12.5])
p21a.set_ylabel('pH',c=col_peri,fontsize=16)
p21a.tick_params(axis='y',labelcolor=col_peri)

p22a  = p22.twinx()
l225, = p22a.plot(xCO2,DICtrFuncs['pH'] (argDICtr_basa),lw=wid_mine,c=col_basa,ls=lwo_mine) 
l222, = p22.plot(xCO2,DICtrFuncs['HCO3'](argDICtr_basa),lw=wid_rock,c=col_bica,ls=lwi_mine)
l223, = p22.plot(xCO2,DICtrFuncs['CO3'] (argDICtr_basa),lw=wid_mine,c=col_carb,ls=lwi_mine)
l221, = p22.plot(xCO2,DICtrFuncs['ALK'] (argDICtr_basa),lw=wid_rock,c=col_basa,ls=lwi_rock)

p22.set_xscale('log')
p22.set_yscale('log')
p22.set_ylim([2e-13,2e1])
p22.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=16)
p22.set_ylabel('$C$ [mol dm$^{-3}$]',fontsize=16)
p22.set_title('Basalt',c=col_basa,fontsize=20)
p22.set_xticks(np.logspace(-8,0,num=5))
p22.set_yticks(np.logspace(-12,0,num=5))
p22a.set_yticks(np.linspace(4,12,num=5))
p22a.set_ylim([3.5,12.5])
p22a.set_ylabel('pH',c=col_basa,fontsize=16)
p22a.tick_params(axis='y',labelcolor=col_basa)

p23a  = p23.twinx()
l235, = p23a.plot(xCO2,DICtrFuncs['pH'] (argDICtr_gran),lw=wid_mine,c=col_gran,ls=lwo_mine) 
l232, = p23.plot(xCO2,DICtrFuncs['HCO3'](argDICtr_gran),lw=wid_rock,c=col_bica,ls=lwi_mine)
l233, = p23.plot(xCO2,DICtrFuncs['CO3'] (argDICtr_gran),lw=wid_mine,c=col_carb,ls=lwi_mine)
l231, = p23.plot(xCO2,DICtrFuncs['ALK'] (argDICtr_gran),lw=wid_rock,c=col_gran,ls=lwi_rock)

p23.set_xscale('log')
p23.set_yscale('log')
p23.set_ylim([2e-13,2e1])
p23.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=16)
p23.set_ylabel('$C$ [mol dm$^{-3}$]',fontsize=16)
p23.set_title('Granite',c=col_gran,fontsize=20)
p23.set_xticks(np.logspace(-8,0,num=5))
p23.set_yticks(np.logspace(-12,0,num=5))
p23a.set_yticks(np.linspace(4,12,num=5))
p23a.set_ylim([3.5,12.5])
p23a.set_ylabel('pH',c=col_gran,fontsize=16)
p23a.tick_params(axis='y',labelcolor=col_gran)

p24a  = p24.twinx()
l245, = p24a.plot(xCO2,DICtrFuncs['pH'] (argDICtr_anor),lw=wid_mine,c=col_anor,ls=lwo_mine) 
l242, = p24.plot(xCO2,DICtrFuncs['HCO3'](argDICtr_anor),lw=wid_rock,c=col_bica,ls=lwi_mine)
l243, = p24.plot(xCO2,DICtrFuncs['CO3'] (argDICtr_anor),lw=wid_mine,c=col_carb,ls=lwi_mine)
l241, = p24.plot(xCO2,DICtrFuncs['ALK'] (argDICtr_anor),lw=wid_rock,c=col_anor,ls=lwi_rock)

p24.set_xscale('log')
p24.set_yscale('log')
p24.set_ylim([2e-13,2e1])
p24.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=16)
p24.set_ylabel('$C$ [mol dm$^{-3}$]',fontsize=16)
p24.set_title('Anorthite',c=col_anor,fontsize=20)
p24.set_xticks(np.logspace(-8,0,num=5))
p24.set_yticks(np.logspace(-12,0,num=5))
p24a.set_yticks(np.linspace(4,12,num=5))
p24a.set_ylim([3.5,12.5])
p24a.set_ylabel('pH',c=col_anor,fontsize=16)
p24a.tick_params(axis='y',labelcolor=col_anor)

p25a  = p25.twinx()
l255, = p25a.plot(xCO2,DICtrFuncs['pH'] (argDICtr_albi),lw=wid_mine,c=col_albi,ls=lwo_mine) 
l252, = p25.plot(xCO2,DICtrFuncs['HCO3'](argDICtr_albi),lw=wid_rock,c=col_bica,ls=lwi_mine)
l253, = p25.plot(xCO2,DICtrFuncs['CO3'] (argDICtr_albi),lw=wid_mine,c=col_carb,ls=lwi_mine)
l251, = p25.plot(xCO2,DICtrFuncs['ALK'] (argDICtr_albi),lw=wid_rock,c=col_albi,ls=lwi_rock)

p25.set_xscale('log')
p25.set_yscale('log')
p25.set_ylim([2e-13,2e1])
p25.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=16)
p25.set_ylabel('$C$ [mol dm$^{-3}$]',fontsize=16)
p25.set_title('Albite',c=col_albi,fontsize=20)
p25.set_xticks(np.logspace(-8,0,num=5))
p25.set_yticks(np.logspace(-12,0,num=5))
p25a.set_yticks(np.linspace(4,12,num=5))
p25a.set_ylim([3.5,12.5])
p25a.set_ylabel('pH',c=col_albi,fontsize=16)
p25a.tick_params(axis='y',labelcolor=col_albi)

p26a  = p26.twinx()
l265, = p26a.plot(xCO2,DICtrFuncs['pH'] (argDICtr_kfel),lw=wid_mine,c=col_kfel,ls=lwo_mine) 
l262, = p26.plot(xCO2,DICtrFuncs['HCO3'](argDICtr_kfel),lw=wid_rock,c=col_bica,ls=lwi_mine)
l263, = p26.plot(xCO2,DICtrFuncs['CO3'] (argDICtr_kfel),lw=wid_mine,c=col_carb,ls=lwi_mine)
l261, = p26.plot(xCO2,DICtrFuncs['ALK'] (argDICtr_kfel),lw=wid_rock,c=col_kfel,ls=lwi_rock)

p26.set_xscale('log')
p26.set_yscale('log')
p26.set_ylim([2e-13,2e1])
p26.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=16)
p26.set_ylabel('$C$ [mol dm$^{-3}$]',fontsize=16)
p26.set_title('K-feldspar',c=col_kfel,fontsize=20)
p26.set_xticks(np.logspace(-8,0,num=5))
p26.set_yticks(np.logspace(-12,0,num=5))
p26a.set_yticks(np.linspace(4,12,num=5))
p26a.set_ylim([3.5,12.5])
p26a.set_ylabel('pH',c=col_kfel,fontsize=16)
p26a.tick_params(axis='y',labelcolor=col_kfel)

p27a  = p27.twinx()
l275, = p27a.plot(xCO2,DICtrFuncs['pH'] (argDICtr_woll),lw=wid_mine,c=col_woll,ls=lwo_mine) 
l272, = p27.plot(xCO2,DICtrFuncs['HCO3'](argDICtr_woll),lw=wid_rock,c=col_bica,ls=lwi_mine)
l273, = p27.plot(xCO2,DICtrFuncs['CO3'] (argDICtr_woll),lw=wid_mine,c=col_carb,ls=lwi_mine)
l271, = p27.plot(xCO2,DICtrFuncs['ALK'] (argDICtr_woll),lw=wid_rock,c=col_woll,ls=lwi_rock)

p27.set_xscale('log')
p27.set_yscale('log')
p27.set_ylim([2e-13,2e1])
p27.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=16)
p27.set_ylabel('$C$ [mol dm$^{-3}$]',fontsize=16)
p27.set_title('Wollastonite',c=col_woll,fontsize=20)
p27.set_xticks(np.logspace(-8,0,num=5))
p27.set_yticks(np.logspace(-12,0,num=5))
p27a.set_yticks(np.linspace(4,12,num=5))
p27a.set_ylim([3.5,12.5])
p27a.set_ylabel('pH',c=col_woll,fontsize=16)
p27a.tick_params(axis='y',labelcolor=col_woll)

p28a  = p28.twinx()
l285, = p28a.plot(xCO2,DICtrFuncs['pH'] (argDICtr_enst),lw=wid_mine,c=col_enst,ls=lwo_mine) 
l282, = p28.plot(xCO2,DICtrFuncs['HCO3'](argDICtr_enst),lw=wid_rock,c=col_bica,ls=lwi_mine)
l283, = p28.plot(xCO2,DICtrFuncs['CO3'] (argDICtr_enst),lw=wid_mine,c=col_carb,ls=lwi_mine)
l281, = p28.plot(xCO2,DICtrFuncs['ALK'] (argDICtr_enst),lw=wid_rock,c=col_enst,ls=lwi_rock)

p28.set_xscale('log')
p28.set_yscale('log')
p28.set_ylim([2e-13,2e1])
p28.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=16)
p28.set_ylabel('$C$ [mol dm$^{-3}$]',fontsize=16)
p28.set_title('Enstatite',c=col_enst,fontsize=20)
p28.set_xticks(np.logspace(-8,0,num=5))
p28.set_yticks(np.logspace(-12,0,num=5))
p28a.set_yticks(np.linspace(4,12,num=5))
p28a.set_ylim([3.5,12.5])
p28a.set_ylabel('pH',c=col_enst,fontsize=16)
p28a.tick_params(axis='y',labelcolor=col_enst)

p29a  = p29.twinx()
l295, = p29a.plot(xCO2,DICtrFuncs['pH'] (argDICtr_ferr),lw=wid_mine,c=col_ferr,ls=lwo_mine) 
l292, = p29.plot(xCO2,DICtrFuncs['HCO3'](argDICtr_ferr),lw=wid_rock,c=col_bica,ls=lwi_mine)
l293, = p29.plot(xCO2,DICtrFuncs['CO3'] (argDICtr_ferr),lw=wid_mine,c=col_carb,ls=lwi_mine)
l291, = p29.plot(xCO2,DICtrFuncs['ALK'] (argDICtr_ferr),lw=wid_rock,c=col_ferr,ls=lwi_rock)

p29.set_xscale('log')
p29.set_yscale('log')
p29.set_ylim([2e-13,2e1])
p29.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=16)
p29.set_ylabel('$C$ [mol dm$^{-3}$]',fontsize=16)
p29.set_title('Ferrosilite',c=col_ferr,fontsize=20)
p29.set_xticks(np.logspace(-8,0,num=5))
p29.set_yticks(np.logspace(-12,0,num=5))
p29a.set_yticks(np.linspace(4,12,num=5))
p29a.set_ylim([3.5,12.5])
p29a.set_ylabel('pH',c=col_ferr,fontsize=16)
p29a.tick_params(axis='y',labelcolor=col_ferr)

p210a  = p210.twinx()
l2105, = p210a.plot(xCO2,DICtrFuncs['pH'] (argDICtr_fors),lw=wid_mine,c=col_fors,ls=lwo_mine)
l2102, = p210.plot(xCO2,DICtrFuncs['HCO3'](argDICtr_fors),lw=wid_rock,c=col_bica,ls=lwi_mine)
l2103, = p210.plot(xCO2,DICtrFuncs['CO3'] (argDICtr_fors),lw=wid_mine,c=col_carb,ls=lwi_mine)
l2101, = p210.plot(xCO2,DICtrFuncs['ALK'] (argDICtr_fors),lw=wid_rock,c=col_fors,ls=lwi_rock)

p210.set_xscale('log')
p210.set_yscale('log')
p210.set_ylim([2e-13,2e1])
p210.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=16)
p210.set_ylabel('$C$ [mol dm$^{-3}$]',fontsize=16)
p210.set_title('Forsterite',c=col_fors,fontsize=20)
p210.set_xticks(np.logspace(-8,0,num=5))
p210.set_yticks(np.logspace(-12,0,num=5))
p210a.set_yticks(np.linspace(4,12,num=5))
p210a.set_ylim([3.5,12.5])
p210a.set_ylabel('pH',c=col_fors,fontsize=16)
p210a.tick_params(axis='y',labelcolor=col_fors)

p211a  = p211.twinx()
l2115, = p211a.plot(xCO2,DICtrFuncs['pH'] (argDICtr_faya),lw=wid_mine,c=col_faya,ls=lwo_mine)
l2112, = p211.plot(xCO2,DICtrFuncs['HCO3'](argDICtr_faya),lw=wid_rock,c=col_bica,ls=lwi_mine)
l2113, = p211.plot(xCO2,DICtrFuncs['CO3'] (argDICtr_faya),lw=wid_mine,c=col_carb,ls=lwi_mine)
l2111, = p211.plot(xCO2,DICtrFuncs['ALK'] (argDICtr_faya),lw=wid_rock,c=col_faya,ls=lwi_rock)

p211.set_xscale('log')
p211.set_yscale('log')
p211.set_ylim([2e-13,2e1])
p211.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=16)
p211.set_ylabel('$C$ [mol dm$^{-3}$]',fontsize=16)
p211.set_title('Fayalite',c=col_faya,fontsize=20)
p211.set_xticks(np.logspace(-8,0,num=5))
p211.set_yticks(np.logspace(-12,0,num=5))
p211a.set_yticks(np.linspace(4,12,num=5))
p211a.set_ylim([3.5,12.5])
p211a.set_ylabel('pH',c=col_faya,fontsize=16)
p211a.tick_params(axis='y',labelcolor=col_faya)

p212a  = p212.twinx()
l2125, = p212a.plot(xCO2,DICtrFuncs['pH'] (argDICtr_musc),lw=wid_mine,c=col_musc,ls=lwo_mine)
l2122, = p212.plot(xCO2,DICtrFuncs['HCO3'](argDICtr_musc),lw=wid_rock,c=col_bica,ls=lwi_mine)
l2123, = p212.plot(xCO2,DICtrFuncs['CO3'] (argDICtr_musc),lw=wid_mine,c=col_carb,ls=lwi_mine)
l2121, = p212.plot(xCO2,DICtrFuncs['ALK'] (argDICtr_musc),lw=wid_rock,c=col_musc,ls=lwi_rock)

p212.set_xscale('log')
p212.set_yscale('log')
p212.set_ylim([2e-13,2e1])
p212.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=16)
p212.set_ylabel('$C$ [mol dm$^{-3}$]',fontsize=16)
p212.set_title('Muscovite',c=col_musc,fontsize=20)
p212.set_xticks(np.logspace(-8,0,num=5))
p212.set_yticks(np.logspace(-12,0,num=5))
p212a.set_yticks(np.linspace(4,12,num=5))
p212a.set_ylim([3.5,12.5])
p212a.set_ylabel('pH',c=col_musc,fontsize=16)
p212a.tick_params(axis='y',labelcolor=col_musc)

p213a  = p213.twinx()
l2135, = p213a.plot(xCO2,DICtrFuncs['pH'] (argDICtr_phlo),lw=wid_mine,c=col_phlo,ls=lwo_mine)
l2132, = p213.plot(xCO2,DICtrFuncs['HCO3'](argDICtr_phlo),lw=wid_rock,c=col_bica,ls=lwi_mine)
l2133, = p213.plot(xCO2,DICtrFuncs['CO3'] (argDICtr_phlo),lw=wid_mine,c=col_carb,ls=lwi_mine)
l2131, = p213.plot(xCO2,DICtrFuncs['ALK'] (argDICtr_phlo),lw=wid_rock,c=col_phlo,ls=lwi_rock)

p213.set_xscale('log')
p213.set_yscale('log')
p213.set_ylim([2e-13,2e1])
p213.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=16)
p213.set_ylabel('$C$ [mol dm$^{-3}$]',fontsize=16)
p213.set_title('Phlogopite',c=col_phlo,fontsize=20)
p213.set_xticks(np.logspace(-8,0,num=5))
p213.set_yticks(np.logspace(-12,0,num=5))
p213a.set_yticks(np.linspace(4,12,num=5))
p213a.set_ylim([3.5,12.5])
p213a.set_ylabel('pH',c=col_phlo,fontsize=16)
p213a.tick_params(axis='y',labelcolor=col_phlo)

p214a  = p214.twinx()
l2145, = p214a.plot(xCO2,DICtrFuncs['pH'] (argDICtr_anni),lw=wid_mine,c=col_anni,ls=lwo_mine)
l2142, = p214.plot(xCO2,DICtrFuncs['HCO3'](argDICtr_anni),lw=wid_rock,c=col_bica,ls=lwi_mine)
l2143, = p214.plot(xCO2,DICtrFuncs['CO3'] (argDICtr_anni),lw=wid_mine,c=col_carb,ls=lwi_mine)
l2141, = p214.plot(xCO2,DICtrFuncs['ALK'] (argDICtr_anni),lw=wid_rock,c=col_anni,ls=lwi_rock)

p214.set_xscale('log')
p214.set_yscale('log')
p214.set_ylim([2e-13,2e1])
p214.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=16)
p214.set_ylabel('$C$ [mol dm$^{-3}$]',fontsize=16)
p214.set_title('Annite',c=col_anni,fontsize=20)
p214.set_xticks(np.logspace(-8,0,num=5))
p214.set_yticks(np.logspace(-12,0,num=5))
p214a.set_yticks(np.linspace(4,12,num=5))
p214a.set_ylim([3.5,12.5])
p214a.set_ylabel('pH',c=col_anni,fontsize=16)
p214a.tick_params(axis='y',labelcolor=col_anni)

p215a  = p215.twinx()
l2155, = p215a.plot(xCO2,DICtrFuncs['pH'] (argDICtr_anth),lw=wid_mine,c=col_anth,ls=lwo_mine)
l2152, = p215.plot(xCO2,DICtrFuncs['HCO3'](argDICtr_anth),lw=wid_rock,c=col_bica,ls=lwi_mine)
l2153, = p215.plot(xCO2,DICtrFuncs['CO3'] (argDICtr_anth),lw=wid_mine,c=col_carb,ls=lwi_mine)
l2151, = p215.plot(xCO2,DICtrFuncs['ALK'] (argDICtr_anth),lw=wid_rock,c=col_anth,ls=lwi_rock)

p215.set_xscale('log')
p215.set_yscale('log')
p215.set_ylim([2e-13,2e1])
p215.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=16)
p215.set_ylabel('$C$ [mol dm$^{-3}$]',fontsize=16)
p215.set_title('Anthophyllite',c=col_anth,fontsize=20)
p215.set_xticks(np.logspace(-8,0,num=5))
p215.set_yticks(np.logspace(-12,0,num=5))
p215a.set_yticks(np.linspace(4,12,num=5))
p215a.set_ylim([3.5,12.5])
p215a.set_ylabel('pH',c=col_anth,fontsize=16)
p215a.tick_params(axis='y',labelcolor=col_anth)

p216a  = p216.twinx()
l2165, = p216a.plot(xCO2,DICtrFuncs['pH'] (argDICtr_grun),lw=wid_mine,c=col_grun,ls=lwo_mine) 
l2162, = p216.plot(xCO2,DICtrFuncs['HCO3'](argDICtr_grun),lw=wid_rock,c=col_bica,ls=lwi_mine)
l2163, = p216.plot(xCO2,DICtrFuncs['CO3'] (argDICtr_grun),lw=wid_mine,c=col_carb,ls=lwi_mine)
l2161, = p216.plot(xCO2,DICtrFuncs['ALK'] (argDICtr_grun),lw=wid_rock,c=col_grun,ls=lwi_rock)

p216.set_xscale('log')
p216.set_yscale('log')
p216.set_ylim([2e-13,2e1])
p216.set_xlabel('$P_{\mathrm{CO}_2}$ [bar]',fontsize=16)
p216.set_ylabel('$C$ [mol dm$^{-3}$]',fontsize=16)
p216.set_title('Grunerite',c=col_grun,fontsize=20)
p216.set_xticks(np.logspace(-8,0,num=5))
p216.set_yticks(np.logspace(-12,0,num=5))
p216a.set_yticks(np.linspace(4,12,num=5))
p216a.set_ylim([3.5,12.5])
p216a.set_ylabel('pH',c=col_grun,fontsize=16)
p216a.tick_params(axis='y',labelcolor=col_grun)

i_peri = np.where((Dw_peri < q0 * np.ones(len(xCO2))) == True)[0][0]
p21.scatter(xCO2[i_peri],DICtrFuncs['HCO3'](argDICtr_peri)[i_peri],marker='o',s=80,color=col_bica,zorder=4)
p21.scatter(xCO2[i_peri],DICtrFuncs['ALK'] (argDICtr_peri)[i_peri],marker='o',s=80,color=col_peri,zorder=4)

i_basa = np.where((Dw_basa < q0 * np.ones(len(xCO2))) == True)[0][0]
p22.scatter(xCO2[i_basa],DICtrFuncs['HCO3'](argDICtr_basa)[i_basa],marker='o',s=80,color=col_bica,zorder=4)
p22.scatter(xCO2[i_basa],DICtrFuncs['ALK'] (argDICtr_basa)[i_basa],marker='o',s=80,color=col_basa,zorder=4)

i_gran = np.where((Dw_gran < q0 * np.ones(len(xCO2))) == True)[0][0]
p23.scatter(xCO2[i_gran],DICtrFuncs['HCO3'](argDICtr_gran)[i_gran],marker='o',s=80,color=col_bica,zorder=4)
p23.scatter(xCO2[i_gran],DICtrFuncs['ALK'] (argDICtr_gran)[i_gran],marker='o',s=80,color=col_gran,zorder=4)

# i_anor = np.where((Dw_anor < q0 * np.ones(len(xCO2))) == True)[0][0]
# p24.scatter(xCO2[i_anor],DICtrFuncs['HCO3'](argDICtr_anor)[i_anor],marker='o',s=80,color=col_bica,zorder=4)
# p24.scatter(xCO2[i_anor],DICtrFuncs['ALK'] (argDICtr_anor)[i_anor],marker='o',s=80,color=col_anor,zorder=4)

# i_albi = np.where((Dw_albi < q0 * np.ones(len(xCO2))) == True)[0][0]
# p25.scatter(xCO2[i_albi],DICtrFuncs['HCO3'](argDICtr_albi)[i_albi],marker='o',s=80,color=col_bica,zorder=4)
# p25.scatter(xCO2[i_albi],DICtrFuncs['ALK'] (argDICtr_albi)[i_albi],marker='o',s=80,color=col_albi,zorder=4)

# i_kfel = np.where((Dw_kfel < q0 * np.ones(len(xCO2))) == True)[0][0]
# p26.scatter(xCO2[i_kfel],DICtrFuncs['HCO3'](argDICtr_kfel)[i_kfel],marker='o',s=80,color=col_bica,zorder=4)
# p26.scatter(xCO2[i_kfel],DICtrFuncs['ALK'] (argDICtr_kfel)[i_kfel],marker='o',s=80,color=col_kfel,zorder=4)

# i_woll = np.where((Dw_woll < q0 * np.ones(len(xCO2))) == True)[0][0]
# j_woll = np.where((DICtrFuncs['CO2'](argDICtr_woll) > DICtrFuncs['HCO3'](argDICtr_woll)) == True)[0][0]
# p27.scatter(xCO2[i_woll],DICtrFuncs['HCO3'](argDICtr_woll)[i_woll],marker='o',s=80,color=col_bica,zorder=4)
# p27.scatter(xCO2[i_woll],DICtrFuncs['ALK'] (argDICtr_woll)[i_woll],marker='o',s=80,color=col_woll,zorder=4)

i_enst = np.where((Dw_enst < q0 * np.ones(len(xCO2))) == True)[0][0]
p28.scatter(xCO2[i_enst],DICtrFuncs['HCO3'](argDICtr_enst)[i_enst],marker='o',s=80,color=col_bica,zorder=4)
p28.scatter(xCO2[i_enst],DICtrFuncs['ALK'] (argDICtr_enst)[i_enst],marker='o',s=80,color=col_enst,zorder=4)

i_ferr = np.where((Dw_ferr < q0 * np.ones(len(xCO2))) == True)[0][0]
p29.scatter(xCO2[i_ferr],DICtrFuncs['HCO3'](argDICtr_ferr)[i_ferr],marker='o',s=80,color=col_bica,zorder=4)
p29.scatter(xCO2[i_ferr],DICtrFuncs['ALK'] (argDICtr_ferr)[i_ferr],marker='o',s=80,color=col_ferr,zorder=4)

i_fors = np.where((Dw_fors < q0 * np.ones(len(xCO2))) == True)[0][0]
p210.scatter(xCO2[i_fors],DICtrFuncs['HCO3'](argDICtr_fors)[i_fors],marker='o',s=80,color=col_bica,zorder=4)
p210.scatter(xCO2[i_fors],DICtrFuncs['ALK'] (argDICtr_fors)[i_fors],marker='o',s=80,color=col_fors,zorder=4)

i_faya = np.where((Dw_faya < q0 * np.ones(len(xCO2))) == True)[0][0]
p211.scatter(xCO2[i_faya],DICtrFuncs['HCO3'](argDICtr_faya)[i_faya],marker='o',s=80,color=col_bica,zorder=4)
p211.scatter(xCO2[i_faya],DICtrFuncs['ALK'] (argDICtr_faya)[i_faya],marker='o',s=80,color=col_faya,zorder=4)

# i_musc = np.where((Dw_musc < q0 * np.ones(len(xCO2))) == True)[0][0]
# p212.scatter(xCO2[i_musc],DICtrFuncs['HCO3'](argDICtr_musc)[i_musc],marker='o',s=80,color=col_bica,zorder=4)
# p212.scatter(xCO2[i_musc],DICtrFuncs['ALK'] (argDICtr_musc)[i_musc],marker='o',s=80,color=col_musc,zorder=4)

i_phlo = np.where((Dw_phlo < q0 * np.ones(len(xCO2))) == True)[0][0]
p213.scatter(xCO2[i_phlo],DICtrFuncs['HCO3'](argDICtr_phlo)[i_phlo],marker='o',s=80,color=col_bica,zorder=4)
p213.scatter(xCO2[i_phlo],DICtrFuncs['ALK'] (argDICtr_phlo)[i_phlo],marker='o',s=80,color=col_phlo,zorder=4)

i_anni = np.where((Dw_anni < q0 * np.ones(len(xCO2))) == True)[0][0]
p214.scatter(xCO2[i_anni],DICtrFuncs['HCO3'](argDICtr_anni)[i_anni],marker='o',s=80,color=col_bica,zorder=4)
p214.scatter(xCO2[i_anni],DICtrFuncs['ALK'] (argDICtr_anni)[i_anni],marker='o',s=80,color=col_anni,zorder=4)

# i_anth = np.where((Dw_anth < q0 * np.ones(len(xCO2))) == True)[0][0]
# p215.scatter(xCO2[i_anth],DICtrFuncs['HCO3'](argDICtr_anth)[i_anth],marker='o',s=80,color=col_bica,zorder=4)
# p215.scatter(xCO2[i_anth],DICtrFuncs['ALK'] (argDICtr_anth)[i_anth],marker='o',s=80,color=col_anth,zorder=4)

# i_grun = np.where((Dw_grun < q0 * np.ones(len(xCO2))) == True)[0][0]
# p216.scatter(xCO2[i_grun],DICtrFuncs['HCO3'](argDICtr_grun)[i_grun],marker='o',s=80,color=col_bica,zorder=4)
# p216.scatter(xCO2[i_grun],DICtrFuncs['ALK'] (argDICtr_grun)[i_grun],marker='o',s=80,color=col_grun,zorder=4)

lT = p21.fill_between([xCO2[0],     xCO2[i_peri]], [2e-13,2e-13], [2e1,2e1], color='yellow', alpha=0.2)
lK = p21.fill_between([xCO2[i_peri],xCO2[-1]], [2e-13,2e-13], [2e1,2e1], color='red',    alpha=0.1)

p22.fill_between([xCO2[0],     xCO2[i_basa]], [2e-13,2e-13], [2e1,2e1], color='yellow', alpha=0.2)
p22.fill_between([xCO2[i_basa],xCO2[-1]],     [2e-13,2e-13], [2e1,2e1], color='red',    alpha=0.1)

p23.fill_between([xCO2[0],     xCO2[i_gran]], [2e-13,2e-13], [2e1,2e1], color='yellow', alpha=0.2)
p23.fill_between([xCO2[i_gran],xCO2[-1]],     [2e-13,2e-13], [2e1,2e1], color='red',    alpha=0.1)

p24.fill_between([xCO2[0],     xCO2[-1]],     [2e-13,2e-13], [2e1,2e1], color='yellow', alpha=0.2)

p25.fill_between([xCO2[0],     xCO2[-1]],     [2e-13,2e-13], [2e1,2e1], color='yellow', alpha=0.2)

p26.fill_between([xCO2[0],     xCO2[-1]],     [2e-13,2e-13], [2e1,2e1], color='yellow', alpha=0.2)

p27.fill_between([xCO2[0],     xCO2[-1]],     [2e-13,2e-13], [2e1,2e1], color='yellow', alpha=0.2)

p28.fill_between([xCO2[0],     xCO2[i_enst]], [2e-13,2e-13], [2e1,2e1], color='yellow', alpha=0.2)
p28.fill_between([xCO2[i_enst],xCO2[-1]],     [2e-13,2e-13], [2e1,2e1], color='red',    alpha=0.1)

p29.fill_between([xCO2[0],     xCO2[i_ferr]], [2e-13,2e-13], [2e1,2e1], color='yellow', alpha=0.2)
p29.fill_between([xCO2[i_ferr],xCO2[-1]],     [2e-13,2e-13], [2e1,2e1], color='red',    alpha=0.1)

p210.fill_between([xCO2[0],     xCO2[i_fors]], [2e-13,2e-13], [2e1,2e1], color='yellow', alpha=0.2)
p210.fill_between([xCO2[i_fors],xCO2[-1]],     [2e-13,2e-13], [2e1,2e1], color='red',    alpha=0.1)

p211.fill_between([xCO2[0],     xCO2[i_faya]], [2e-13,2e-13], [2e1,2e1], color='yellow', alpha=0.2)
p211.fill_between([xCO2[i_faya],xCO2[-1]],     [2e-13,2e-13], [2e1,2e1], color='red',    alpha=0.1)

p212.fill_between([xCO2[0],     xCO2[-1]],     [2e-13,2e-13], [2e1,2e1], color='yellow', alpha=0.2)

p213.fill_between([xCO2[0],     xCO2[i_phlo]], [2e-13,2e-13], [2e1,2e1], color='yellow', alpha=0.2)
p213.fill_between([xCO2[i_phlo],xCO2[-1]],     [2e-13,2e-13], [2e1,2e1], color='red',    alpha=0.1)

p214.fill_between([xCO2[0],     xCO2[i_anni]], [2e-13,2e-13], [2e1,2e1], color='yellow', alpha=0.2)
p214.fill_between([xCO2[i_anni],xCO2[-1]],     [2e-13,2e-13], [2e1,2e1], color='red',    alpha=0.1)

p215.fill_between([xCO2[0],     xCO2[-1]],     [2e-13,2e-13], [2e1,2e1], color='yellow', alpha=0.2)

p216.fill_between([xCO2[0],     xCO2[-1]],     [2e-13,2e-13], [2e1,2e1], color='yellow', alpha=0.2)

fig2.legend([l271,l272,l273,l275],['$A$','[HCO$_3^-$]','[CO$_3^{2-}$]','pH'],\
            frameon=True,prop={'size':14},bbox_to_anchor=(0.55,0.05),loc='lower right',\
            bbox_transform=fig2.transFigure,handlelength=4,fontsize=12,title='Species',title_fontsize=14)

fig2.legend([lT,lK],['Thermodynamic','Kinetic'],\
            frameon=True,prop={'size':14},bbox_to_anchor=(0.75,0.08),loc='lower right',\
            bbox_transform=fig2.transFigure,handlelength=4,fontsize=12,title='Regimes',title_fontsize=14)

plt.savefig('CA_mine.pdf',format='pdf',bbox_inches='tight')

# %%
# Fig. C1: Comparison between CO2 solubility obtained from experiments and models

a_henry = KeqFuncs['co2c'](T,P0)
a_expmt = KeqFuncs['co2b'](T,P0)
a_therm = KeqFuncs['co2a'](T,P0)

del_h = (a_henry-a_expmt)/a_expmt*100
del_t = (a_therm-a_expmt)/a_expmt*100

plt.figure(figsize=(5.5,5))
plt.plot(T,del_h,c='black',lw=2,label='Arrhenius-type model (Pierrehumbert 2010)')
plt.plot(T,del_t,c='red',lw=2,label='Thermodynamic database (CHNOSZ, Dick 2019)')
plt.legend(prop={'size':12})
plt.xlabel('$T$ [K]',fontsize=20)
plt.ylabel('$\Delta a_{\mathrm{CO}_2(aq)}$ [\%]',fontsize=20)
plt.text(275, -28, '$P_{\mathrm{CO}_2}$ = 280 $\mu$bar')
plt.savefig('henry_CO2.pdf',format='pdf',bbox_inches='tight')

print(np.round(time.time() - start),'s, Figures saved in the local directory.')

# %%
# Parameters, MACH
C_eq0     = 380e-6
keff0     = 8.7e-6
sp_area0  = 100
mol_mass0 = 0.27
t_soil0   = 1e5
L0        = 0.4
phi0      = 0.175
rho0      = 2700
X_r0      = 0.36
Dw0       = 0.03

# Parameters, This work, Granite
C_eq1     = DICeqFuncs['grah']['HCO3']([xCO2g0,T288,P0])
C_eq11    = C_eq1*1e6
pH        = DICeqFuncs['grah']['pH']([xCO2g0,T288,P0])
keff1     = kFuncs['kfel'](np.full_like(pH,T288), pH)
mol_mass1 = 0.27
L1        = 1
phi1      = 0.175
rho1      = 2700
X_r1      = 1
Dw1       = tr.Dw_GRAN()

# Fig9: Dw sensitivity to parameters and comparison between this work and MACH

fig1 = plt.figure(figsize=(15,15))
spec1 = gridspec.GridSpec(ncols=3, nrows=3, figure=fig1)
p11 = fig1.add_subplot(spec1[0,0])
p12 = fig1.add_subplot(spec1[0,1])
p13 = fig1.add_subplot(spec1[0,2])
p14 = fig1.add_subplot(spec1[1,0])
p15 = fig1.add_subplot(spec1[1,1])
p16 = fig1.add_subplot(spec1[1,2])
p17 = fig1.add_subplot(spec1[2,0])
p18 = fig1.add_subplot(spec1[2,1])
p19 = fig1.add_subplot(spec1[2,2])
p11.text(-0.1, 1.15, '(a)', transform=p11.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p12.text(-0.1, 1.15, '(b)', transform=p12.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p13.text(-0.1, 1.15, '(c)', transform=p13.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p14.text(-0.1, 1.15, '(d)', transform=p14.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p15.text(-0.1, 1.15, '(e)', transform=p15.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p16.text(-0.1, 1.15, '(f)', transform=p16.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p17.text(-0.1, 1.15, '(g)', transform=p17.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p18.text(-0.1, 1.15, '(h)', transform=p18.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p19.text(-0.1, 1.15, '(i)', transform=p19.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')

l111,   =   p11.plot(C_eq,tr.Dw_MACH(C_eq=C_eq),lw=wid_rock,c=col_quar,ls=lwi_mine)
l112,   =   p11.plot(C_eq,tr.Dw_GRAN(C_eq=C_eq),lw=wid_rock,c=col_gran,ls=lwi_mine)
p11.plot(C_eq0,Dw0,marker='o',markersize=8,c=col_quar)
p11.plot(C_eq1,Dw1,marker='o',markersize=8,c=col_gran)
p11.set_ylim([1e-8,1e4])
p11.set_xscale('log')
p11.set_yscale('log')
p11.set_xlabel('$C_{\mathrm{eq}}$ [mol dm$^{-3}$]',fontsize=16)
p11.set_ylabel('$D_w$ [m yr$^{-1}$]',fontsize=16)
p11.set_title('Equilibrium Solute Concentration',fontsize=18)

l121,   =   p12.plot(keff,tr.Dw_MACH(keff=keff),lw=wid_rock,c=col_quar,ls=lwi_mine)
l122,   =   p12.plot(keff,tr.Dw_GRAN(keff=keff),lw=wid_rock,c=col_gran,ls=lwi_mine)
p12.plot(keff0,Dw0,marker='o',markersize=8,c=col_quar)
p12.plot(keff1,Dw1,marker='o',markersize=8,c=col_gran)
p12.set_ylim([1e-8,1e4])
p12.set_xscale('log')
p12.set_yscale('log')
p12.set_xlabel('$k_{\mathrm{eff}}$ [mol m$^{-2}$ yr$^{-1}$]',fontsize=16)
p12.set_ylabel('$D_w$ [m yr$^{-1}$]',fontsize=16)
p12.set_title('Kinetic Rate Coefficient',fontsize=18)

l131,   =   p13.plot(t_soil,tr.Dw_MACH(t_soil=t_soil),lw=wid_rock,c=col_quar,ls=lwi_mine)
l132,   =   p13.plot(t_soil,tr.Dw_GRAN(t_soil=t_soil),lw=wid_rock,c=col_gran,ls=lwi_mine)
p13.plot(t_soil0,Dw0,marker='o',markersize=8,c=col_quar)
p13.plot(t_soil0,Dw1,marker='o',markersize=8,c=col_gran)
p13.set_ylim([1e-8,1e4])
p13.set_xscale('log')
p13.set_yscale('log')
p13.set_xlabel('$t_{s}$ [yr]',fontsize=16)
p13.set_ylabel('$D_w$ [m yr$^{-1}$]',fontsize=16)
p13.set_title('Age of Soils or Pore-space',fontsize=18)

l141,   =   p14.plot(L,tr.Dw_MACH(L=L),lw=wid_rock,c=col_quar,ls=lwi_mine)
l142,   =   p14.plot(L,tr.Dw_GRAN(L=L),lw=wid_rock,c=col_gran,ls=lwi_mine)
p14.plot(L0,Dw0,marker='o',markersize=8,c=col_quar)
p14.plot(L1,Dw1,marker='o',markersize=8,c=col_gran)
p14.set_ylim([1e-8,1e4])
p14.set_xscale('log')
p14.set_yscale('log')
p14.set_xlabel('$L$ [m]',fontsize=16)
p14.set_ylabel('$D_w$ [m yr$^{-1}$]',fontsize=16)
p14.set_title('Flowpath Length',fontsize=18)

l151,   =   p15.plot(phi,tr.Dw_MACH(phi=phi),lw=wid_rock,c=col_quar,ls=lwi_mine)
l152,   =   p15.plot(phi,tr.Dw_GRAN(phi=phi),lw=wid_rock,c=col_gran,ls=lwi_mine)
p15.plot(phi0,Dw0,marker='o',markersize=8,c=col_quar)
p15.plot(phi1,Dw1,marker='o',markersize=8,c=col_gran)
p15.set_ylim([1e-8,1e4])
p15.set_yscale('log')
p15.set_xlabel(r'$\phi$',fontsize=16)
p15.set_ylabel('$D_w$ [m yr$^{-1}$]',fontsize=16)
p15.set_title('Porosity',fontsize=18)

l161,   =   p16.plot(X_r,tr.Dw_MACH(X_r=X_r),lw=wid_rock,c=col_quar,ls=lwi_mine)
l162,   =   p16.plot(X_r,tr.Dw_GRAN(X_r=X_r),lw=wid_rock,c=col_gran,ls=lwi_mine)
p16.plot(X_r0,Dw0,marker='o',markersize=8,c=col_quar)
p16.plot(X_r1,Dw1,marker='o',markersize=8,c=col_gran)
p16.set_ylim([1e-8,1e4])
p16.set_yscale('log')
p16.set_xlabel('$X_{r}$',fontsize=16)
p16.set_ylabel('$D_w$ [m yr$^{-1}$]',fontsize=16)
p16.set_title('Fraction of React. Min. in Fresh Rock',fontsize=18)

l171,   =   p17.plot(sp_area,tr.Dw_MACH(sp_area=sp_area),lw=wid_rock,c=col_quar,ls=lwi_mine)
l172,   =   p17.plot(sp_area,tr.Dw_GRAN(sp_area=sp_area),lw=wid_rock,c=col_gran,ls=lwi_mine)
p17.plot(sp_area0,Dw0,marker='o',markersize=8,c=col_quar)
p17.plot(sp_area0,Dw1,marker='o',markersize=8,c=col_gran)
p17.set_ylim([1e-8,1e4])
p17.set_xscale('log')
p17.set_yscale('log')
p17.set_xlabel('$A_{\mathrm{sp}}$ [m$^{2}$ kg$^{-1}$]',fontsize=16)
p17.set_ylabel('$D_w$ [m yr$^{-1}$]',fontsize=16)
p17.set_title('Specific Surface Area',fontsize=18)

l181,   =   p18.plot(rho,tr.Dw_MACH(rho=rho),lw=wid_rock,c=col_quar,ls=lwi_mine)
l182,   =   p18.plot(rho,tr.Dw_GRAN(rho=rho),lw=wid_rock,c=col_gran,ls=lwi_mine)
p18.plot(rho0,Dw0,marker='o',markersize=8,c=col_quar)
p18.plot(rho1,Dw1,marker='o',markersize=8,c=col_gran)
p18.set_ylim([1e-8,1e4])
p18.set_yscale('log')
p18.set_xlabel(r'$\rho$ [kg m$^{-3}$]',fontsize=16)
p18.set_ylabel('$D_w$ [m yr$^{-1}$]',fontsize=16)
p18.set_title('Density of Solid',fontsize=18)

l191,   =   p19.plot(mol_mass,tr.Dw_MACH(mol_mass=mol_mass),lw=wid_rock,c=col_quar,ls=lwi_mine)
l192,   =   p19.plot(mol_mass,tr.Dw_GRAN(mol_mass=mol_mass),lw=wid_rock,c=col_gran,ls=lwi_mine)
p19.plot(mol_mass0,Dw0,marker='o',markersize=8,c=col_quar)
p19.plot(mol_mass1,Dw1,marker='o',markersize=8,c=col_gran)
p19.set_ylim([1e-8,1e4])
p19.set_yscale('log')
p19.set_xlabel('$m$ [kg mol$^{-1}$]',fontsize=16)
p19.set_ylabel('$D_w$ [m yr$^{-1}$]',fontsize=16)
p19.set_title('Molar Mass',fontsize=18)

p12.text(0.05, 0.57, 'Kinetic', transform=p12.transAxes,fontsize=12)
p12.text(0.7, 0.57, 'Supply', transform=p12.transAxes,fontsize=12)
p13.text(0.05, 0.5, 'Kinetic', transform=p13.transAxes,fontsize=12)
p13.text(0.7, 0.5, 'Supply', transform=p13.transAxes,fontsize=12)

fig1.legend([l111,l112],\
    ['Maher \& Chamberlain (2014, $C_{\mathrm{eq}}$ = [SiO$_2]_{\mathrm{eq}}$)',\
    'This work ($C_{\mathrm{eq}}$ = [HCO$_3^-]_{\mathrm{eq}}$)'],\
    frameon=True,prop={'size':11},bbox_to_anchor=(0.4,0.95),loc='upper left',\
    bbox_transform=fig1.transFigure,ncol=1,handlelength=2,fontsize=12)

plt.tight_layout()
plt.savefig('Dw_MACH.pdf',format='pdf',bbox_inches='tight')

# %%
# Fig. E1: Comparison between WHAK and KATA climate models

#Planetary albedo (albedo) as a free parameter
alb = np.linspace(0.1,0.5)
WHAKalb, KATAalb = np.zeros((2,len(alb)))
for i in range(len(alb)):
    WHAKalb[i] = cl.T_WHAK(280e-6, albedo = alb[i])
    KATAalb[i] = cl.T_KATA(280e-6, albedo = alb[i])
    
#PCO2 as a free parameter
PCO2 = np.logspace(-5,0) # volume mixing ratio of CO2(g)
WHAKp, KATAp = np.zeros((2,len(PCO2)))
for i in range(len(PCO2)):
    WHAKp[i] = cl.T_WHAK(PCO2[i], albedo = 0.3)
    KATAp[i] = cl.T_KATA(PCO2[i], albedo = 0.3)
    
fig1 = plt.figure(figsize=(10,5))
spec1 = gridspec.GridSpec(ncols=2, nrows=1, figure=fig1)
p11 = fig1.add_subplot(spec1[0,0])
p12 = fig1.add_subplot(spec1[0,1])

p11.text(-0.1, 1.15, '(a)', transform=p11.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
p12.text(-0.1, 1.15, '(b)', transform=p12.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')

p11.plot(alb,KATAalb,c=col_bica,lw=3,ls='-',\
         label=r'Kadoya \& Tajika 2019 ($P_{\mathrm{CO}_{2}}$ = 280 $\mu$bar)')
p11.plot(alb,WHAKalb,c=col_carb,lw=3,ls='-',\
         label=r'Walker et al. 1981 ($P_{\mathrm{CO}_{2}}$ = 280 $\mu$bar)')
p11.scatter(0.3,288,color=col_gran,marker='o',s=50,label=r'$\alpha$ = 0.3, $T$ = 288 K',zorder=4)

p11.legend(frameon=True,prop={'size':11},loc='lower left',handlelength=2)
#p11.set_xscale('log')
p11.set_xlabel(r'$\alpha$',fontsize=20)
p11.set_ylabel('$T$ [K]',fontsize=20)
p11.set_ylim([250,350])
p11.set_title('Planetary albedo',fontsize=20)

p12.plot(PCO2,KATAp,c=col_bica,lw=3,ls='-',label=r'Kadoya \& Tajika 2019 ($\alpha$ = 0.3)')
p12.plot(PCO2,WHAKp,c=col_carb,lw=3,ls='-',label=r'Walker et al. 1981 ($\alpha$ = 0.3)')
p12.scatter(280e-6,288,color=col_gran,marker='o',s=50,\
            label=r'$P_{\mathrm{CO}_{2}}$ = 280 $\mu$bar, $T$ = 288 K',zorder=4)

p12.legend(frameon=True,prop={'size':11},loc='best',handlelength=2)
p12.set_xscale('log')
p12.set_xlabel('$P_{\mathrm{CO}_{2}}$ [bar]',fontsize=20)
p12.set_ylabel('$T$ [K]',fontsize=20)
p12.set_ylim([250,350])
p12.set_title('CO$_2$ Partial Pressure',fontsize=20)

plt.savefig('climate_WHAK_KATA.pdf',format='pdf',bbox_inches='tight')
