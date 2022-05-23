import numpy as np
import pandas as pd
import csv
import datetime
import copy

from functions.f_consumptionModels import *

import os
if os.path.basename(os.getcwd())=="Belfort":
    os.chdir('..')
    os.chdir('..') ## to work at project root  like in any IDE

InputFolder='Data/input/Conso_model/'
InputFolder_Europe='Data/input/Conso_model/Conso_Europe/'

Conso_areas_df = pd.read_csv(InputFolder_Europe + 'Conso_Europe.csv', delimiter=';', decimal='.',
                             parse_dates=['Date']).set_index('Date')
Conso_FR_ref_df=pd.read_csv(InputFolder + 'areaConsumption2019_FR.csv', delimiter=';', decimal='.')
Conso_NTS_ref_df=pd.read_csv(InputFolder + 'Conso_NTS_2019.csv', delimiter=';', decimal='.')

E_ref_FR=Conso_FR_ref_df['Consommation'].sum()/1.08
E_ref_steel_FR=Conso_NTS_ref_df['Metallurgie'].sum()
E_ref_FR-=E_ref_steel_FR
P_ref_steel_FR=E_ref_steel_FR/8760

def conso_model_europe(year,L_areas=['BE','CH','DE','ES','GB','IT'],Conso_areas_df=Conso_areas_df,
                       E_ref_FR=E_ref_FR,P_ref_steel_FR=P_ref_steel_FR):
    Conso_H2_areas_df=pd.read_csv(InputFolder_Europe+'Conso_H2_Europe.csv',delimiter=';',decimal=',').set_index('AREAS')
    Factor_demo_areas_df = pd.read_csv(InputFolder_Europe+'Demo_factors_Europe.csv', delimiter=';', decimal=',').set_index('AREAS')
    Factor_steel_areas_df= pd.read_csv(InputFolder_Europe+'Factor_steel_Europe.csv', delimiter=';', decimal=',').set_index('AREAS')
    Conso_FR_df=pd.read_csv(InputFolder+'Loads/Conso_'+str(year)+'_no_reindus_ref.csv',delimiter=';',decimal='.',parse_dates=['Date']).set_index('Date')


    P_steel_FR=Conso_FR_df['Metallurgie'].mean()
    E_FR=Conso_FR_df['Consommation hors metallurgie'].sum()

    dict_conso_areas={}
    for area in L_areas:
        df_area=pd.DataFrame()
        df_area['Taux_pertes']=Conso_FR_df['Taux_pertes']
        df_area['Consommation hors metallurgie'] = E_FR/E_ref_FR*(Conso_areas_df['Conso_'+area]/(1+df_area['Taux_pertes'])\
                                                   -P_ref_steel_FR*Factor_steel_areas_df.loc[area,'Factor'])
        df_area['Metallurgie']=P_steel_FR*Factor_steel_areas_df.loc[area,'Factor']
        df_area['Conso_VE']=Conso_FR_df['Conso_VE']*Factor_demo_areas_df.loc[area,str(year)]
        if area in Conso_H2_areas_df.index:
            df_area['Conso_H2']=Conso_H2_areas_df.loc[area,str(year)]*1e6/8760
        else:
            df_area['Conso_H2']=Conso_FR_df['Conso_H2']*Factor_demo_areas_df.loc[area,str(year)]
        df_area['AREAS']=area
        dict_conso_areas[area]=df_area

    df_ret=pd.DataFrame()
    for area in L_areas:
        df_ret=pd.concat([df_ret,dict_conso_areas[area]])
        print("Conso "+area+"= {} TWh".format(((dict_conso_areas[area]['Consommation hors metallurgie']\
                                              +dict_conso_areas[area]['Metallurgie']\
                                              +dict_conso_areas[area]['Conso_VE']\
                                              +dict_conso_areas[area]['Conso_H2'])\
                                              *(1+dict_conso_areas[area]['Taux_pertes'])).sum()/1e6))
    df_ret=df_ret.reset_index().set_index(['AREAS','Date'])

    return df_ret

for year in [2030,2040,2050,2060]:
    print(year)
    df_conso=conso_model_europe(year)
    df_conso.to_csv(InputFolder_Europe+'Conso_Europe_'+str(year)+'.csv',sep=';',decimal='.')

