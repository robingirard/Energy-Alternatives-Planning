#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 23:18:37 2022

@author: pierrickdartois
"""

import os
if os.path.basename(os.getcwd())=="Simulations":
    os.chdir('..')
    os.chdir('..')
    os.chdir('..') ## to work at project root  like in any IDE

InputFolder='Models/Belfort/Conso/'

from functions.f_consumptionModels import *
from Models.Belfort.Simulations.f_consumptionBelfort import *
from functions.f_graphicalTools import *

Conso_df=pd.read_csv(InputFolder+'areaConsumption2019_FR.csv',sep=';',parse_dates=['Date']).set_index(["Date"])
Temp_df=pd.read_csv(InputFolder+'Temp_FR_2017_2022.csv',sep=';',decimal='.',parse_dates=['Date']).set_index(["Date"])

index2019=(Temp_df.index.to_series().dt.minute==0)&(Temp_df.index.to_series().dt.year==2019)
Temp_2019_df=Temp_df[index2019].reset_index().set_index("Date").sort_index()
Temp_2019_df= CleanCETIndex(Temp_2019_df)# Traitement heure d'été et heure d'hiver

ConsoTemp_2019_df=Conso_df.join(Temp_2019_df,on='Date')

T0=15# Température de référence chauffage (15)
T1=20# Température seuil clim (20)
T2=20# Temperature de référence pertes et ECS (20)

## Pertes
taux_pertes=0.06927
rho_pertes=-1.2e-3
ConsoTemp_2019_df.loc[:,'Consommation']=ConsoTemp_2019_df.loc[:,'Consommation']*(1-taux_pertes)\
    +ConsoTemp_2019_df.loc[:,'Consommation']*rho_pertes*ConsoTemp_2019_df.loc[:,'Temperature']\
    -ConsoTemp_2019_df.loc[:,'Consommation']*rho_pertes*T2

## Eau chaude sanitaire
ECS_df=pd.read_csv(InputFolder+'Profil_ECS_RTE.csv',sep=';',decimal=',',encoding='utf-8').set_index(["Jour","Heure"])
L_week=["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"]
for weekday in range(7):
    for hour in range(24):
        indexHour=(ConsoTemp_2019_df.index.to_series().dt.hour==hour)&(ConsoTemp_2019_df.index.to_series().dt.weekday==weekday)
        ConsoTemp_2019_df.loc[indexHour,'Consommation']=ConsoTemp_2019_df.loc[indexHour,'Consommation']\
        -ECS_df.loc[(L_week[weekday],hour),'ECS en juin']\
        -ECS_df.loc[(L_week[weekday],hour),'Thermosensibilite (GW/degre)']*ConsoTemp_2019_df.loc[indexHour,'Temperature']\
        +ECS_df.loc[(L_week[weekday],hour),'Thermosensibilite (GW/degre)']*T2



## Decomposition Thermosensible et non-thermosensible
(ConsoSeparee_df, Thermosensitivity_winter,Thermosensitivity_summer)=Decomposeconso2(ConsoTemp_2019_df,T0,T1,'Temperature','Consommation','Date')

##Creation du profil
Conso_non_thermosensible = ConsoSeparee_df[["NTS_C"]].rename(columns= {"NTS_C":"Consumption"})

NTS_profil=  pd.read_csv(InputFolder+"Profil_NTS_RTE.csv",sep=";", decimal=",").\
    melt(id_vars=['Saison','Jour','Heure'],
          value_vars=['Climatisation et ventilation','Industrie hors metallurgie','Metallurgie','Energie',
                      'Cuisson','Eclairage','Autres usages'],
         var_name='type', value_name='poids').\
    set_index(['Saison','Jour','Heure'])

NTS_profil_hourly=ComplexProfile2Consumption_2(NTS_profil.loc[("Hiver",slice(None),slice(None))],Conso_non_thermosensible)

#Pour visualiser
fig = MyStackedPlotly(y_df=NTS_profil_hourly)
plotly.offline.plot(fig, filename='file.html')## offline

## Enregistrement du profil
NTS_profil_hourly.to_csv(InputFolder+"Conso_NTS_2019.csv",sep=";",decimal=".")

## Enregistrement de la thermosensibilité
df_TS=pd.DataFrame(list(Thermosensitivity_winter.items()),columns=["Heure","Thermosensibilite hiver (GW/degre)"]).set_index("Heure")
df_TS_summer=pd.DataFrame(list(Thermosensitivity_summer.items()),columns=["Heure","Thermosensibilite ete (GW/degre)"]).set_index("Heure")
df_TS=df_TS.merge(df_TS_summer,on="Heure")
df_TS.to_csv(InputFolder+"Thermosensitivity_2019.csv",sep=";",decimal=".")