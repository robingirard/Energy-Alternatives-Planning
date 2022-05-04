#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 23:18:37 2022

@author: pierrickdartois
"""

import os
if os.path.basename(os.getcwd())=="Belfort":
    os.chdir('..') 
    os.chdir('..') ## to work at project root  like in any IDE

InputFolder='Data/input/Conso_model/'

#region importation of modules
import numpy as np
import pandas as pd
import seaborn as sns # you might have to isntall seaborn yourself : conda install seaborn
import csv
import datetime
import copy
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn import linear_model

from functions.f_consumptionModels import *
from functions.f_graphicalTools import *

Conso_df=pd.read_csv(InputFolder+'areaConsumption2019_FR.csv',sep=';',parse_dates=['Date']).set_index(["Date"])
Temp_df=pd.read_csv(InputFolder+'Temp_FR_2017_2022.csv',sep=';',decimal='.',parse_dates=['Date']).set_index(["Date"])

index2019=(Temp_df.index.to_series().dt.minute==0)&(Temp_df.index.to_series().dt.year==2019)
Temp_2019_df=Temp_df[index2019]

ConsoTemp_2019_df=Conso_df.join(Temp_2019_df,on='Date')


T0=15# Température de référence chauffage
T1=20# Température seuil clim
T2=20# Temperature de référence pertes et ECS

## Pertes
taux_pertes=0.07
rho_pertes=-1e-3
ConsoTemp_2019_df.loc[:,'Consommation']=ConsoTemp_2019_df.loc[:,'Consommation']*(1-taux_pertes)\
    +ConsoTemp_2019_df.loc[:,'Consommation']*rho_pertes*ConsoTemp_2019_df.loc[:,'Temperature']\
    -ConsoTemp_2019_df.loc[:,'Consommation']*rho_pertes*T2

## Eau chaude sanitaire
ECS_df=pd.read_csv(InputFolder+'Profil_ECS.csv',sep=';',decimal=',').set_index(["Heure"])
for hour in range(24):
    indexHour=(ConsoTemp_2019_df.index.to_series().dt.hour==hour)
    ConsoTemp_2019_df.loc[indexHour,'Consommation']=ConsoTemp_2019_df.loc[indexHour,'Consommation']\
        -ECS_df.loc[hour,'ECS (20 degres)']\
        -ECS_df.loc[hour,'Thermosensibilite (GW/degre)']*ConsoTemp_2019_df.loc[indexHour,'Temperature']\
        +ECS_df.loc[hour,'Thermosensibilite (GW/degre)']*T2
        

## Decomposition Thermosensible et non-thermosensible
(ConsoSeparee_df, Thermosensitivity_winter,Thermosensitivity_summer)=Decomposeconso2(ConsoTemp_2019_df,T0,T1,'Temperature','Consommation','Date')
print(Thermosensitivity_winter)
print(Thermosensitivity_summer)

print(ConsoSeparee_df.head())

##Creation du profil
Conso_non_thermosensible = ConsoSeparee_df[["NTS_C"]].rename(columns= {"NTS_C":"Consumption"})

NTS_profil=  pd.read_csv(InputFolder+"Profil_NTS.csv",sep=";", decimal=",").\
    melt(id_vars=['Heure','Jour', 'Mois'],
          value_vars=['Industrie','Autres residentiel','Autres tertiaire','Eclairage','Cuisson'],
         var_name='type', value_name='poids').\
    set_index(["Jour","Mois","Heure"])

NTS_profil_hourly=ComplexProfile2Consumption(NTS_profil,Conso_non_thermosensible).\
    reset_index()[["Consumption","Date","type"]].\
    groupby(["Date","type"]).sum().reset_index().\
    pivot(index="Date", columns="type", values="Consumption")
### etrange d'aavoir à faire le grouby ci-dessus
### si on veut visualiser les poids, il faut remplacer "Consumption" par "poids" ci-dessus
# print(NTS_profil_hourly)
fig = MyStackedPlotly(y_df=NTS_profil_hourly)
plotly.offline.plot(fig, filename='file.html')## offline