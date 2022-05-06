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

import os
if os.path.basename(os.getcwd())=="Belfort":
    os.chdir('..')
    os.chdir('..') ## to work at project root  like in any IDE

InputFolder='Data/input/Conso_model/'

# Main scenario hypothesis
reindus=True
bati_hyp='ref'# Alternative SNBC
T0=15# Temperature when heating starts

NTS_profil_df=pd.read_csv(InputFolder+'Conso_NTS_2019.csv',sep=';',decimal='.',parse_dates=['Date']).set_index(['Date'])
Thermosensitivity_df=pd.read_csv(InputFolder+'Thermosensitivity_2019.csv',sep=';',decimal='.').set_index(["Heure"])
Projections_df=pd.read_csv(InputFolder+'Projections_NTS.csv',sep=';',decimal=',').set_index(['Annee'])

Conso_projected_df=ProjectionConsoNTS(NTS_profil_df,Projections_df,2050,reindus)
print(Conso_projected_df)

# Heating
Energy_houses_df=pd.read_csv(InputFolder+'Bati/Energie_TWh_maisons_type_de_chauffage_'+bati_hyp+'.csv',sep=';',decimal='.').set_index("Année")
Energy_apartments_df=pd.read_csv(InputFolder+'Bati/Energie_TWh_appartements_type_de_chauffage_'+bati_hyp+'.csv',sep=';',decimal='.').set_index("Année")
Energy_offices_df=pd.read_csv(InputFolder+'Bati/Energie_tertiaire_type_de_chauffage.csv',sep=';',decimal='.').set_index("Année")
Part_PAC_RCU_df=pd.read_csv(InputFolder+'Bati/Part_PAC_reseaux_chaleur.csv',sep=';',decimal=',').set_index("Annee")
#print(Energy_houses_df.head())

Temp_df=pd.read_csv(InputFolder+'Temp_FR_2017_2022.csv',sep=';',decimal='.',parse_dates=['Date']).set_index(["Date"])

index2019=(Temp_df.index.to_series().dt.minute==0)&(Temp_df.index.to_series().dt.year==2019)
Temp_2019_df=Temp_df[index2019]

# Thermosensitive consumption
Conso_TS_heat_df=ConsoHeat(Temp_2019_df,Thermosensitivity_df,
              Energy_houses_df,Energy_apartments_df,Energy_offices_df,Part_PAC_RCU_df,2050,
              bati_hyp,T0)
print(Conso_TS_heat_df)
print(Conso_TS_heat_df["Conso_TS_heat"].sum())

Conso_TS_air_con_df=ConsoAirCon(Temp_2019_df,Thermosensitivity_df,Energy_houses_df,Energy_apartments_df,Energy_offices_df,2050)
print(Conso_TS_air_con_df)
print(Conso_TS_air_con_df["Conso_TS_air_con"].sum())



