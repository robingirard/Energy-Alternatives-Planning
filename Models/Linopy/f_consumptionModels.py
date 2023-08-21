import numpy as np
import pandas as pd
import csv
import os
import copy
import datetime
import matplotlib.pyplot as plt
from sklearn import linear_model
from datetime import time
from datetime import datetime
from datetime import date
from datetime import timedelta

#data_df=energy_demand.join(ConsoTempe_df)[['energy_demand','temperature']]
def Decomposeconso(data_df, temperatureThreshold=14, temperatureName='temperature',ConsumptionName='Consumption',TimeName='date') :
    '''
    fonction décomposant la consommation électrique d'une année en une part thermosensible et une part non thermosensible
    :param data: panda data frame with "temperature" and "Consumption" as columns
    :param temperatureThreshold: the threshold heating temperature
    :param temperatureName default 'temperature' name of column with temperature
    :param ConsumptionName default 'Consumption' name of column with consumption
    :param TimeName default 'date' name of column with time
    :return: a dictionary with Thermosensibilite, and a panda data frame with two new columns NTS_C and TS_C
    '''
    dataNoNA_df=data_df.dropna()
    ## Remove NA
    ConsoSeparee_df=dataNoNA_df.assign(NTS_C=dataNoNA_df[ConsumptionName], TS_C=dataNoNA_df[ConsumptionName]*0)

    Thermosensibilite={}
    #pd.DataFrame(data=np.zeros((24,1)), columns=['Thermosensibilite'])## one value per hour of the day
    #Thermosensibilite.index.name='hour of the day'
    for hour in range(24):
        indexesWinterHour = (dataNoNA_df[temperatureName] <= temperatureThreshold) & (dataNoNA_df.index.get_level_values(TimeName).to_series().dt.hour == hour)
        dataWinterHour_df = dataNoNA_df.loc[indexesWinterHour,:]
        lr = linear_model.LinearRegression().fit(dataWinterHour_df[[temperatureName]],dataWinterHour_df[ConsumptionName])
        Thermosensibilite[hour]=lr.coef_[0]
        ConsoSeparee_df.loc[indexesWinterHour,'TS_C']=Thermosensibilite[hour]*dataWinterHour_df.loc[:, temperatureName]-Thermosensibilite[hour]*temperatureThreshold
        ConsoSeparee_df.loc[indexesWinterHour,'NTS_C']=dataWinterHour_df.loc[:, ConsumptionName]-ConsoSeparee_df.TS_C.loc[indexesWinterHour]

    return(ConsoSeparee_df, Thermosensibilite)

def Decomposeconso2(data_df,T0=15,T1=20,temperatureName='temperature',
                    ConsumptionName='Consumption',TimeName='date'):
    '''
    Function decomposing the consumption into thermosensitive and non-thermosensitive part
    taking into account air condition in summer.

    parameters
    ----------
    data_df : panda data frame with "temperature" and "Consumption" as columns.
    T0 : float, optional
        Threshold temperature for heating in winter. The default is 15.
    T1 : TYPE, optional
        Threshold temperature for air condition in summer. The default is 20.
    temperatureName : str, optional
        The default is 'temperature'.
    ConsumptionName : str, optional
        The default is 'Consumption'.
    TimeName : str, optional
        The default is 'date'.

    Returns
    -------
    a dictionary with Thermosensitivity_winter, Thermosensitivity_summer, 
    and a panda data frame with two new columns NTS_C (non thermosensitive), 
    TSW_C (thermosensitive winter), TSS_C (thermosensitive summer)
    '''
    
    dataNoNA_df=data_df.dropna()
    ## Remove NA
    ConsoSeparee_df=dataNoNA_df.assign(NTS_C=dataNoNA_df[ConsumptionName], TSW_C=dataNoNA_df[ConsumptionName]*0,
                                       TSS_C=dataNoNA_df[ConsumptionName]*0)

    Thermosensitivity_winter={}
    Thermosensitivity_summer={}
    #pd.DataFrame(data=np.zeros((24,1)), columns=['Thermosensibilite'])## one value per hour of the day
    #Thermosensibilite.index.name='hour of the day'
    for hour in range(24):
        indexesWinterHour = (dataNoNA_df[temperatureName] <= T0) & (dataNoNA_df.index.get_level_values(TimeName).to_series().dt.hour == hour)
        indexesSummerHour = (dataNoNA_df[temperatureName] >= T1) & (dataNoNA_df.index.get_level_values(TimeName).to_series().dt.hour == hour)
        dataWinterHour_df = dataNoNA_df.loc[indexesWinterHour,:]
        dataSummerHour_df = dataNoNA_df.loc[indexesSummerHour,:]
        lrw = linear_model.LinearRegression().fit(dataWinterHour_df[[temperatureName]],dataWinterHour_df[ConsumptionName])
        Thermosensitivity_winter[hour]=lrw.coef_[0]
        lrs = linear_model.LinearRegression().fit(dataSummerHour_df[[temperatureName]],dataSummerHour_df[ConsumptionName])
        Thermosensitivity_summer[hour]=lrs.coef_[0]
        ConsoSeparee_df.loc[indexesWinterHour,'TSW_C']=Thermosensitivity_winter[hour]*dataWinterHour_df.loc[:, temperatureName]-Thermosensitivity_winter[hour]*T0
        ConsoSeparee_df.loc[indexesWinterHour,'NTS_C']=dataWinterHour_df.loc[:, ConsumptionName]-ConsoSeparee_df.TSW_C.loc[indexesWinterHour]
        ConsoSeparee_df.loc[indexesSummerHour,'TSS_C']=Thermosensitivity_summer[hour]*dataSummerHour_df.loc[:, temperatureName]-Thermosensitivity_summer[hour]*T1
        ConsoSeparee_df.loc[indexesSummerHour,'NTS_C']=dataSummerHour_df.loc[:, ConsumptionName]-ConsoSeparee_df.TSS_C.loc[indexesSummerHour]

    return (ConsoSeparee_df[['NTS_C','TSW_C','TSS_C']], Thermosensitivity_winter,Thermosensitivity_summer)

#ConsoSeparee_df=ConsoTempeYear_decomposed_df
def Recompose(ConsoSeparee_df,Thermosensibilite,Newdata_df=-1, temperatureThreshold=14,temperatureName='temperature',ConsumptionName='Consumption',TimeName='date'):
    '''
    fonction permettant de redécomposer la conso électrique en part thermosensible et
    non thermosensible de l'année x à partir de la thermosensibilité
    (calculée pour chaque heure de la journée) de l'année x et les température d'une année y
    should handle the dimension match problem related to bisextile years
    :param ConsoSeparee_df:
    :param Newdata_df:
    :param Thermosensibilite:
    :param temperatureThreshold:
    :param temperatureName:
    :param ConsumptionName:
    :param TimeName:
    :return:
    '''
    if (Newdata_df.__class__==int): Newdata_df=ConsoSeparee_df
    indexes_Old = np.nonzero(np.in1d(np.arange(0,ConsoSeparee_df.__len__()), np.arange(0,Newdata_df[temperatureName].__len__())))[0]
    indexes_New = np.nonzero(np.in1d(np.arange(0,Newdata_df[temperatureName].__len__()), np.arange(0,ConsoSeparee_df.__len__())))[0]

    ConsoSepareeNew_df=ConsoSeparee_df.iloc[indexes_Old,:].copy(deep=True) ## to suppress warning see https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
    ConsoSepareeNew_df.loc[:, temperatureName]=Newdata_df.iloc[indexes_New, Newdata_df.columns.get_loc(temperatureName)].tolist()
    ConsoSepareeNew_df.loc[:,"TS_C"]=0
    ConsoSepareeNew_df.loc[:,"NTS_C"]=ConsoSepareeNew_df.loc[:,"NTS_C"]

    for hour in range(24):
        indexesWinterHour = (ConsoSepareeNew_df[temperatureName] <= temperatureThreshold) & (ConsoSepareeNew_df.index.get_level_values(TimeName).to_series().dt.hour== hour)
        ## remove thermal sensitive part according to old temperature
        ConsoSepareeNew_df.loc[indexesWinterHour, 'TS_C'] = Thermosensibilite[hour] * ConsoSepareeNew_df.loc[indexesWinterHour,temperatureName] - Thermosensibilite[hour] * temperatureThreshold

    ConsoSepareeNew_df.loc[:,ConsumptionName]= (ConsoSepareeNew_df.loc[:,'TS_C']+ConsoSepareeNew_df.loc[:,'NTS_C'])
    return(ConsoSepareeNew_df)


def add_day_month_hour(df,TimeName="date",French=True,semaine_simplifie=False,day_name=True,to_index=False):
    if French:
        df = df.assign(
            day=df.index.get_level_values(TimeName).to_series().dt.dayofweek,
            Mois=df.index.get_level_values(TimeName).to_series().dt.month,
            hour=df.index.get_level_values(TimeName).to_series().dt.hour);
        if semaine_simplifie:
            df['day'] = df['day'].apply(lambda x: "Semaine" if x < 5 else "Samedi" if x == 5 else "Dimanche")
        elif day_name:
            df.day=df.day.replace({0:"Lundi",1:"Mardi",2:"Mercredi",3:"Jeudi",4:"Vendredi",5:"Samedi",6:"Dimanche"})

        if to_index:
            df = df.reset_index().set_index(["day", "Mois", "hour"])
    else:
        df = df.assign(
            Day=df.index.get_level_values(TimeName).to_series().dt.weekday,
            Month=df.index.get_level_values(TimeName).to_series().dt.month,
            Hour=df.index.get_level_values(TimeName).to_series().dt.hour);
        if semaine_simplifie:
            df['Day'] = df['Day'].apply(lambda x: "Week" if x < 5 else "Sat" if x == 5 else "Sun")
        elif day_name:
            df.day=df.day.replace({0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",4:"Friday",5:"Saturday",6:"Sunday"})

        if to_index:
            df = df.reset_index().set_index(["Day", "Month", "Hour"])
    return df

def Profile2Consumption(Profile_df,temperature_df, temperatureThreshold=14,
                        temperatureMinimum=0,temperatureName='temperature',
                        ConsumptionName='Consumption',TimeName='date',
                        VarName='electrical_power_per_million_ev'):
    '''
    fonction permettant de reconstruire la consommation annuelle à partir d'un profil hourxdayxseason en une part thermosensible et non thermosensible
    (la conso non thermosensible étant la conso type d'une semaine d'été)

    :param Profile_df: profil avec les colonnes hourxdayxseason
    :param temperature_df:
    :param temperatureThreshold:
    :param temperatureMinimum:
    :param temperatureName:
    :param ConsumptionName:
    :param TimeName:
    :param VarName:
    :return:
    '''
    ## initialisation
    ConsoSepareeNew_df=temperature_df.loc[:,[temperatureName]]
    ConsoSepareeNew_df.loc[:,[ConsumptionName]]=np.NaN
    ConsoSepareeNew_df.loc[:,['NTS_C']]=0
    ConsoSepareeNew_df.loc[:,['TS_C']]=0

    PivotedProfile_df = Profile_df.pivot( index=['hour','day'], columns='season', values=VarName ).reset_index()
    cte=(temperatureThreshold-temperatureMinimum)

    for index, row in PivotedProfile_df.iterrows():
        indexesWD=ConsoSepareeNew_df.index.get_level_values(TimeName).to_series().dt.weekday   == (PivotedProfile_df.loc[index,'day']-1)
        indexesHours= ConsoSepareeNew_df.index.get_level_values(TimeName).to_series().dt.hour   == (PivotedProfile_df.loc[index,'hour'])
        ConsoSepareeNew_df.loc[indexesWD&indexesHours, 'NTS_C']=PivotedProfile_df.loc[index,'Ete']

    PivotedProfile_df['NDifference'] = (PivotedProfile_df['Ete'] - PivotedProfile_df['Hiver'])
    Thermosensibilite = (PivotedProfile_df['NDifference'].loc[0:23] / cte).tolist()
    ConsoSepareeNew_df=Recompose(ConsoSepareeNew_df,Thermosensibilite)
    return(ConsoSepareeNew_df)

#temperature_df=Conso_non_thermosensible
#Profile_df_Week,Profile_df_Sat,Profile_df_Sun,ConsoTempeYear_df
#temperature_df=ConsoTempeYear_decomposed_df.loc[:,"NTS_C"]
#Profile_df=NTS_profil
def ComplexProfile2Consumption(Profile_df,
                               temperature_df, temperatureThreshold=14,
                        temperatureMinimum=0,temperatureName='temperature',
                        poidsName='poids',
                        ConsumptionName='Consumption',TimeName='date',
                        VarName='electrical_power_per_million_ev',french=True):

    ## initialisation
    ConsoSepareeNew_df=temperature_df.loc[:,[ConsumptionName]]
    #ConsoSepareeNew_df.loc[:,[ConsumptionName]]=np.NaN
    #ConsoSepareeNew_df.loc[:,['NTS_C']]=0
    #ConsoSepareeNew_df.loc[:,['TS_C']]=0
    ConsoSepareeNew_df = ConsoSepareeNew_df.assign(
        day=ConsoSepareeNew_df.index.get_level_values(TimeName).to_series().dt.weekday,
        Mois=ConsoSepareeNew_df.index.get_level_values(TimeName).to_series().dt.month,
        hour=ConsoSepareeNew_df.index.get_level_values(TimeName).to_series().dt.hour);
    if french:
        ConsoSepareeNew_df['day']=ConsoSepareeNew_df['day'].\
            apply(lambda x: "Semaine" if x<5 else "Samedi" if x==5 else "Dimanche")
    else:
        ConsoSepareeNew_df['day'] = ConsoSepareeNew_df['day'].\
            apply(lambda x: "Week" if x < 5 else "Sat" if x == 5 else "Sun")
    ConsoSepareeNew_df=ConsoSepareeNew_df.reset_index().set_index(["day","Mois","hour"])

    Profile_df_merged=Profile_df.join(ConsoSepareeNew_df,how="inner")
    Profile_df_merged.loc[:,[ConsumptionName]]=Profile_df_merged[ConsumptionName]*Profile_df_merged[poidsName]
    return(Profile_df_merged)
    #cte=(temperatureThreshold-temperatureMinimum)

# Tient compte des vacances et jours feriés en France en 2019
def ComplexProfile2ConsumptionCJO2019(Profile_df,
                               temperature_df, temperatureThreshold=14,
                        temperatureMinimum=0,temperatureName='temperature',
                        poidsName='poids',
                        ConsumptionName='Consumption',TimeName='date',
                        VarName='electrical_power_per_million_ev',french=True):

    ## initialisation
    ConsoSepareeNew_df=temperature_df.loc[:,[ConsumptionName]]
    #ConsoSepareeNew_df.loc[:,[ConsumptionName]]=np.NaN
    #ConsoSepareeNew_df.loc[:,['NTS_C']]=0
    #ConsoSepareeNew_df.loc[:,['TS_C']]=0
    ConsoSepareeNew_df = ConsoSepareeNew_df.assign(
        day=ConsoSepareeNew_df.index.get_level_values(TimeName).to_series().dt.weekday,
        Mois=ConsoSepareeNew_df.index.get_level_values(TimeName).to_series().dt.month,
        hour=ConsoSepareeNew_df.index.get_level_values(TimeName).to_series().dt.hour);
    if french:
        index_dim_feries=(ConsoSepareeNew_df["day"]==6)\
        |((ConsoSepareeNew_df.index.month==1)&(ConsoSepareeNew_df.index.day==1))\
        |((ConsoSepareeNew_df.index.month==4)&(ConsoSepareeNew_df.index.day==22))\
        |((ConsoSepareeNew_df.index.month==5)&(ConsoSepareeNew_df.index.day==1))\
        |((ConsoSepareeNew_df.index.month==5)&(ConsoSepareeNew_df.index.day==8))\
        |((ConsoSepareeNew_df.index.month==5)&(ConsoSepareeNew_df.index.day==30))\
        |((ConsoSepareeNew_df.index.month==6)&(ConsoSepareeNew_df.index.day==10))\
        |((ConsoSepareeNew_df.index.month==7)&(ConsoSepareeNew_df.index.day==14))\
        |((ConsoSepareeNew_df.index.month==8)&(ConsoSepareeNew_df.index.day==15))\
        |((ConsoSepareeNew_df.index.month==11)&(ConsoSepareeNew_df.index.day==1))\
        |((ConsoSepareeNew_df.index.month==11)&(ConsoSepareeNew_df.index.day==11))\
        |((ConsoSepareeNew_df.index.month==12)&(ConsoSepareeNew_df.index.day==25))
        
        index_sam_vacances=((ConsoSepareeNew_df["day"]==5)\
        |((ConsoSepareeNew_df.index.month==1)&(ConsoSepareeNew_df.index.day<=6))\
        |((ConsoSepareeNew_df.index.month==8)&(ConsoSepareeNew_df.index.day>=3)&(ConsoSepareeNew_df.index.day<=18))\
        |((ConsoSepareeNew_df.index.month==12)&(ConsoSepareeNew_df.index.day>=21))\
        &(index_dim_feries==False))
        
        index_semaine=(index_dim_feries==False)&(index_sam_vacances==False)
        
        ConsoSepareeNew_df.loc[index_dim_feries,"day"]="Dimanche"
        ConsoSepareeNew_df.loc[index_sam_vacances,"day"]="Samedi"
        ConsoSepareeNew_df.loc[index_semaine,"day"]="Semaine"
    else:
        ConsoSepareeNew_df['day'] = ConsoSepareeNew_df['day'].\
            apply(lambda x: "Week" if x < 5 else "Sat" if x == 5 else "Sun")
    ConsoSepareeNew_df=ConsoSepareeNew_df.reset_index().set_index(["day","Mois","hour"])

    Profile_df_merged=Profile_df.join(ConsoSepareeNew_df,how="inner")
    Profile_df_merged.loc[:,[ConsumptionName]]=Profile_df_merged[ConsumptionName]*Profile_df_merged[poidsName]
    return(Profile_df_merged)
    #cte=(temperatureThreshold-temperatureMinimum)


# Adaptée aux nouvelles données de RTE
def ComplexProfile2Consumption_2(Profile_df,
                                      temperature_df,poidsName='poids',
                                      ConsumptionName='Consumption', TimeName='date',GroupName='type'):

    ## Processing dates indexing temperature
    ConsoSepareeNew_df = temperature_df.loc[:, [ConsumptionName]]
    ConsoSepareeNew_df = ConsoSepareeNew_df.assign(
        day=ConsoSepareeNew_df.index.get_level_values(TimeName).to_series().dt.weekday,
        Mois=ConsoSepareeNew_df.index.get_level_values(TimeName).to_series().dt.month,
        hour=ConsoSepareeNew_df.index.get_level_values(TimeName).to_series().dt.hour)

    L_week = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]

    ConsoSepareeNew_df['day'] = ConsoSepareeNew_df['day']. \
            apply(lambda x: L_week[x])
    ConsoSepareeNew_df = ConsoSepareeNew_df.reset_index().set_index(["Mois", "day", "hour"])

    ## Processing the profile to make every month appear
    Profile_df=Profile_df.reset_index()
    Profile_summer=Profile_df[Profile_df.season=="Ete"][["day","hour",GroupName,poidsName]].reset_index()# Janvier
    Profile_winter= Profile_df[Profile_df.season == "Hiver"][["day","hour",GroupName,poidsName]].reset_index()# Juin

    Profile_month=Profile_winter.copy().assign(Mois=1)
    for month in range(2,13):
        Profile_temp=Profile_winter.copy().assign(Mois=month)
        Profile_temp[poidsName]=Profile_winter[poidsName]*np.cos(np.pi*(month-1)/12)**2\
                                +(Profile_summer[poidsName]-Profile_winter[poidsName]*np.cos(np.pi*5/12)**2)\
                                *np.sin(np.pi*(month-1)/12)**2/np.sin(np.pi*5/12)**2
        Profile_month=pd.concat([Profile_month,Profile_temp],ignore_index=True)
    Profile_month=Profile_month.reset_index().set_index(["Mois","day","hour"])


    Profile_month_merged = ConsoSepareeNew_df.join(Profile_month, how="right")
    Profile_month_merged.loc[:, [ConsumptionName]] = Profile_month_merged[ConsumptionName] * Profile_month_merged[poidsName]
    return Profile_month_merged.reset_index()[[ConsumptionName,TimeName,GroupName]].\
        groupby([TimeName,GroupName]).sum().reset_index().\
        pivot(index=TimeName, columns=GroupName, values=ConsumptionName)
    # cte=(temperatureThreshold-temperatureMinimum)

d_reindus={'reindus':' reindustrialisation','no_reindus':'','UNIDEN':' UNIDEN'}
def colReindus(col,reindus='reindus',industryName='Industrie hors metallurgie',steelName='Metallurgie'):
    if col in [industryName,steelName]:
        return col+d_reindus[reindus]
    else:
        return col

def ProjectionConsoNTS(Conso_profile_df,Projections_df,year,reindus='reindus',
                       industryName='Industrie hors metallurgie',steelName='Metallurgie'):
    '''
    Projette la consommation sectorisée.avec des coefficients pour les années futures.

    :param Conso_profile_df: consommation sectorisée à l'année de référence (2019)
    :param Projections_df: coefficients de chaque secteur
    :param year:
    :param reindus: True si scénario de réindustrialisation, False si scénario de référence
    :param industryName:
    :param steelName:
    :param reindusName:
    :return: Retourne un dataframe donnant la somme des consommations sectorielles projetées.
    '''
    Conso_profile_new_df=Conso_profile_df.copy()
    L_cols=list(Conso_profile_df.columns)
    L_years=list(Projections_df.index)
    if year<=L_years[0]:
        for col in L_cols:
            col_proj=colReindus(col,reindus,industryName,steelName)
            Conso_profile_new_df[col]=Projections_df.loc[L_years[0],col_proj]*Conso_profile_new_df[col]
    elif year>=L_years[-1]:
        for col in L_cols:
            col_proj = colReindus(col, reindus, industryName, steelName)
            Conso_profile_new_df[col] = Projections_df.loc[L_years[-1], col_proj] * Conso_profile_new_df[col]
    else:
        i=0
        while i<len(L_years) and year>=L_years[i]:
            i+=1
        for col in L_cols:
            col_proj = colReindus(col, reindus, industryName, steelName)
            Conso_profile_new_df[col] = (Projections_df.loc[L_years[i-1], col_proj]+(year-L_years[i-1])/(L_years[i]-L_years[i-1])*(Projections_df.loc[L_years[i], col_proj]-Projections_df.loc[L_years[i-1], col_proj])) * Conso_profile_new_df[col]

    Conso_profile_new_df=Conso_profile_new_df.assign(Total=0)
    for col in L_cols:
        if col!=steelName:
            Conso_profile_new_df["Total"]+=Conso_profile_new_df[col]
    Conso_profile_new_df=Conso_profile_new_df.rename(columns={"Total":"Consommation hors metallurgie"})
    return Conso_profile_new_df[["Consommation hors metallurgie",steelName]],Conso_profile_new_df[L_cols]

def COP_air_eau(T,year,COP_coeffs_air_eau=[4.8e-4,7e-2,2.26],
                efficiency_gain=0.01):
    return (1+min(0.3,efficiency_gain*(year-2018)))*(T**2*COP_coeffs_air_eau[0]+T*COP_coeffs_air_eau[1]\
                                                     +COP_coeffs_air_eau[2])

def COP_air_air(T,year,COP_coeffs_air_air=[0.05,1.85],
                efficiency_gain=0.01):
    return (1+min(0.3,efficiency_gain*(year-2018)))*(T*COP_coeffs_air_air[0]+COP_coeffs_air_air[1])

def Factor_joule(T,T0=15):
    if T>T0:
        return 0
    else:
        return 1

def Factor_air_eau(T,year,COP_coeffs_air_eau=[4.8e-4,7e-2,2.26],
                efficiency_gain=0.01,T0=15):
    if T>T0:
        return 0
    else:
        return 1/COP_air_eau(T,year,COP_coeffs_air_eau,efficiency_gain)

def Factor_air_air(T,year,COP_coeffs_air_air=[0.05,1.85],
                efficiency_gain=0.01,T0=15):
    if T>T0:
        return 0
    else:
        return 1/COP_air_air(T,year,COP_coeffs_air_air,efficiency_gain)

def Factor_hybrid(T,year,COP_coeffs_air_eau=[4.8e-4,7e-2,2.26],
                efficiency_gain=0.01,T0=15,T_hybrid0=3,T_hybrid1=5):
    if T>T0 or T<T_hybrid0:
        return 0
    elif T>=T_hybrid1:
        return 1/COP_air_eau(T,year,COP_coeffs_air_eau,efficiency_gain)
    else:
        return (1-(T-T_hybrid1)/(T_hybrid0-T_hybrid1))/COP_air_eau(T,year,COP_coeffs_air_eau,efficiency_gain)

def ConsoHeat(temperature_df,Thermosensitivity_df,
              Energy_houses_df,Energy_apartments_df,Energy_offices_df,Part_PAC_df,year,
              bati_hyp="ref",T0=15,T_hybrid0=3,T_hybrid1=5,
              COP_coeffs_air_eau=[4.8e-4,7e-2,2.26],COP_coeffs_air_air=[0.05,1.85],
              efficiency_gain=0.01,efficiency_rc=0.826,part_heat_rc=0.885,
              temperatureName="temperature",CUName="Chauffage urbain",
              JouleName="Chauffage électrique",PACaeName="Pompes à chaleur air-eau",
              PACaaName="Pompes à chaleur air-air",PAChName="Pompes à chaleur hybride",
              ThermoName="Thermosensibilite hiver (GW/degre)",HourName="hour",TimeName="date",
              PartPACName="Part PAC",
              year_ref=2021,year_end=2060):
    '''
    Calcul de la consommation thermosensible de chauffage.

    :param temperature_df:
    :param Thermosensitivity_df:
    :param Energy_houses_df:
    :param Energy_apartments_df:
    :param Energy_offices_df:
    :param Part_PAC_df:
    :param year:
    :param bati_hyp:
    :param T0:
    :param T_hybrid0:
    :param T_hybrid1:
    :param COP_coeffs_air_eau:
    :param COP_coeffs_air_air:
    :param efficiency_gain:
    :param efficiency_rc:
    :param part_heat_rc:
    :param temperatureName:
    :param CUName:
    :param JouleName:
    :param PACaeName:
    :param PACaaName:
    :param PAChName:
    :param ThermoName:
    :param HourName:
    :param TimeName:
    :param PartPACName:
    :param year_ref:
    :param year_end:
    :return:
    '''
    #temperature_new_df=temperature_df.rename(columns={temperatureName:"temperature"})
    temperature_new_df = temperature_df.assign(Factor_j=0,
                                                   Factor_ae=0,
                                                   Factor_aa=0,
                                                   Factor_h=0)

    temperature_new_df["Factor_j"]=temperature_new_df[temperatureName].apply(lambda x: Factor_joule(x,T0))
    temperature_new_df["Factor_ae"] = temperature_new_df[temperatureName].apply(
        lambda x: Factor_air_eau(x,year,COP_coeffs_air_eau,efficiency_gain,T0))
    temperature_new_df["Factor_aa"] = temperature_new_df[temperatureName].apply(
        lambda x: Factor_air_air(x,year,COP_coeffs_air_air,efficiency_gain,T0))
    temperature_new_df["Factor_h"] = temperature_new_df[temperatureName].apply(
        lambda x: Factor_hybrid(x,year,COP_coeffs_air_eau,efficiency_gain,T0,T_hybrid0,T_hybrid1))

    F_ae=temperature_new_df.loc[(temperature_new_df.temperature<=T0),"Factor_ae"].mean()
    F_aa=temperature_new_df.loc[(temperature_new_df.temperature <= T0), "Factor_aa"].mean()

    L_E0=[Energy_houses_df.loc[year_ref,name]+Energy_apartments_df.loc[year_ref,name]\
    +Energy_offices_df.loc[year_ref,name] for name in [JouleName,PACaeName,PACaaName,PAChName]]
    L_E0[-1]+=Part_PAC_df.loc[year_ref,PartPACName+" "+bati_hyp]/(efficiency_rc*part_heat_rc)* \
              (Energy_houses_df.loc[year_ref,CUName]+Energy_apartments_df.loc[year_ref,CUName]\
               +Energy_offices_df.loc[year_ref,CUName])

    if year<year_ref:
        year_ret=year_ref
    elif year>year_end:
        year_ret=year_end
    else:
        year_ret=year

    L_E_year=[Energy_houses_df.loc[year_ret,name]+Energy_apartments_df.loc[year_ret,name]\
    +Energy_offices_df.loc[year_ret,name] for name in [JouleName,PACaeName,PACaaName,PAChName]]
    L_E_year[-1]+=Part_PAC_df.loc[year_ret,PartPACName+" "+bati_hyp]/(efficiency_rc*part_heat_rc)* \
              (Energy_houses_df.loc[year_ret,CUName]+Energy_apartments_df.loc[year_ret,CUName]\
               +Energy_offices_df.loc[year_ret,CUName])

    denom=L_E0[0]+L_E0[1]*F_ae+L_E0[2]*F_aa+L_E0[3]*F_ae
    Thermosensitivity_new_df=Thermosensitivity_df.assign(Thermo_joule=L_E_year[0]/denom,
                                                         Thermo_ae=L_E_year[1]/denom,
                                                         Thermo_aa=L_E_year[2]/denom,
                                                         Thermo_h=L_E_year[3]/denom)

    for name in ["Thermo_joule","Thermo_ae","Thermo_aa","Thermo_h"]:
        Thermosensitivity_new_df[name]=Thermosensitivity_new_df[ThermoName]*Thermosensitivity_new_df[name]


    Thermosensitivity_new_df=Thermosensitivity_new_df.reset_index().rename(columns={HourName:"hour"})
    temperature_new_df=temperature_new_df.assign(hour=temperature_new_df.index.get_level_values(TimeName).to_series().dt.hour,Conso_TS_heat=0)
    temperature_new_df=pd.merge(temperature_new_df.reset_index(),Thermosensitivity_new_df,on="hour").set_index(TimeName).sort_index()

    temperature_new_df["Conso_TS_heat"]=(temperature_new_df[temperatureName]-T0)*(temperature_new_df["Thermo_joule"]*temperature_new_df["Factor_j"]\
                                                                           +temperature_new_df["Thermo_ae"]*temperature_new_df["Factor_ae"]\
                                                                           +temperature_new_df["Thermo_aa"]*temperature_new_df["Factor_aa"]\
                                                                           +temperature_new_df["Thermo_h"]*temperature_new_df["Factor_h"])

    return temperature_new_df[["Conso_TS_heat"]]

def ConsoAirCon(temperature_df,Thermosensitivity_df,
              Energy_houses_df,Energy_apartments_df,Energy_offices_df,year,
              T1=20,taux_clim_res0=0.22,taux_clim_res1=0.55,
              taux_clim_ter0=0.3,taux_clim_ter1=0.55,
              temperatureName="temperature",ThermoName="Thermosensibilite ete (GW/degre)",
              HourName="hour",TimeName="date",year_ref=2021,year_clim1=2050,year_end=2060):
    '''
    Calcul de la consommation thermosensible de climatisation.

    :param temperature_df:
    :param Thermosensitivity_df:
    :param Energy_houses_df:
    :param Energy_apartments_df:
    :param Energy_offices_df:
    :param Part_PAC_df:
    :param year:
    :param bati_hyp:
    :param T1:
    :param taux_clim_res0: part des logements climatisés aujourd'hui (d'après RTE 2050, chapitre consommation)
    :param taux_clim_res1: part de logements climatisés en 2050 (d'après RTE 2050, chapitre consommation)
    :param taux_clim_ter0: part de surfaces tertiaires climatisées aujourd'hui (d'après RTE, GT - consommation tertiaire)
    :param taux_clim_ter1: part de surfaces tertiaires climatisées en 2050 (hypothèse=logement)
    :param temperatureName:
    :param CUName:
    :param JouleName:
    :param PACaeName:
    :param PACaaName:
    :param PAChName:
    :param ThermoName:
    :param HourName:
    :param TimeName:
    :param PartPACName:
    :param year_ref:
    :param year_clim1:
    :param year_end:
    :return:
    '''

    L_cols=list(Energy_houses_df.columns)
    E0=taux_clim_res0*(sum([Energy_houses_df.loc[year_ref,col] for col in L_cols])\
        +sum([Energy_apartments_df.loc[year_ref,col] for col in L_cols]))\
        +taux_clim_ter0*sum([Energy_offices_df.loc[year_ref,col] for col in L_cols])

    if year<year_ref:
        year_ret=year_ref
    elif year>year_end:
        year_ret=year_end
    else:
        year_ret=year
    E_year=(taux_clim_res0+(year_ret-year_ref)/(year_clim1-year_ref)*(taux_clim_res1-taux_clim_res0))\
           *(sum([Energy_houses_df.loc[year_ref,col] for col in L_cols])\
        +sum([Energy_apartments_df.loc[year_ref,col] for col in L_cols]))\
        +(taux_clim_ter0+(year_ret-year_ref)/(year_clim1-year_ref)*(taux_clim_ter1-taux_clim_ter0))\
           *sum([Energy_offices_df.loc[year_ref,col] for col in L_cols])

    Thermosensitivity_new_df=Thermosensitivity_df.copy()
    Thermosensitivity_new_df[ThermoName]=E_year/E0*Thermosensitivity_new_df[ThermoName]

    Thermosensitivity_new_df = Thermosensitivity_new_df.reset_index().rename(columns={HourName: "hour"})
    temperature_new_df = temperature_df.assign(
        hour=temperature_df.index.get_level_values(TimeName).to_series().dt.hour, Conso_TS_air_con=0)
    temperature_new_df = pd.merge(temperature_new_df.reset_index(), Thermosensitivity_new_df, on="hour").set_index(
        TimeName).sort_index()
    temperature_new_df["Conso_TS_air_con"]=temperature_new_df[temperatureName].apply(lambda T: T-T1 if T>=T1 else 0)*temperature_new_df[ThermoName]

    return temperature_new_df[["Conso_TS_air_con"]]

def Conso_ECS(temperature_df,Profil_ECS_df,Projections_ECS_df,year,T2=20,TimeName="date",
              ThermoName="Thermosensibilite (MW/degre)",ECSName="ECS a 20 degres",
              ECSCoeffName="Eau chaude sanitaire"):
    '''
    Projection consommation ECS

    :param Profil_ECS_df:
    :param Projections_ECS_df:
    :param year:
    :param T2: temperature de reference de la courbe de charge de Profil_ECS_df
    :return:
    '''

    temperature_new_df = temperature_df.assign(
        day=temperature_df.index.get_level_values(TimeName).to_series().dt.weekday,
        hour=temperature_df.index.get_level_values(TimeName).to_series().dt.hour,
        Conso_ECS=0)

    L_week = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]

    temperature_new_df['day'] = temperature_new_df['day']. \
        apply(lambda x: L_week[x])
    temperature_new_df = temperature_new_df.reset_index().set_index(["day", "hour"])

    Profil_ECS_new_df=Profil_ECS_df.reset_index().set_index(["day", "hour"])
    temperature_new_df=temperature_new_df.join(Profil_ECS_new_df, how="right")

    L_years = list(Projections_ECS_df.index)
    if year <= L_years[0]:
        C=Projections_ECS_df.loc[L_years[0],ECSCoeffName]
    elif year >= L_years[-1]:
        C = Projections_ECS_df.loc[L_years[-1], ECSCoeffName]
    else:
        i = 0
        while i < len(L_years) and year >= L_years[i]:
            i += 1
        C= Projections_ECS_df.loc[L_years[i - 1],ECSCoeffName] + (year - L_years[i - 1]) / (L_years[i] - L_years[i - 1]) \
           * (Projections_ECS_df.loc[L_years[i],ECSCoeffName]- Projections_ECS_df.loc[L_years[i - 1],ECSCoeffName])

    temperature_new_df["Conso_ECS"]=C*((temperature_new_df["temperature"]-T2)*temperature_new_df[ThermoName]\
                                    +temperature_new_df[ECSName])

    temperature_new_df=temperature_new_df.reset_index().set_index(TimeName).sort_index()
    return temperature_new_df[["Conso_ECS"]]

def ConsoVE(temperature_df,N_VP_df,N_VUL_df,N_PL_df,N_bus_df,N_car_df,Profil_VE_df,Params_VE_df,year,
            T0=15,T1=20,temperatureName="temperature",TimeName="date",VLloadName="Puissance VL",
            PLloadName="Puissance PL",BusloadName="Puissance bus et car",VLThermoName="Thermosensibilite VL",
            PLThermoName="Thermosensibilite PL",BusThermoName="Thermosensibilite bus et car",
            ElName="Electrique",HybridName="Hybride rechargeable",H2Name="Hydrogène",
            ConsoElName="Consommation electrique (kWh/km)",ConsoHybridName="Consommation hybride rechargeable (kWh/km)",
            ConsoH2Name="Consommation hydrogene (kWh/km)",ProgressElName="Progres annuel electrique",
            ProgressHybridName="Progres annuel hybride rechargeable",ProgressH2Name="Progres annuel hydrogene",
            DistName="Kilometrage annuel",VPName="VP",VULName="VUL",PLName="PL",BusName="Bus",CarName="Car",
            year_ref=2020,year_end_progress=2050):
    '''
    Computes the consumption of electric vehicles (light and heavy) including hydrogen.

    :param temperature_df:
    :param N_VP_df:
    :param N_VUL_df:
    :param N_PL_df:
    :param N_bus_df:
    :param N_car_df:
    :param Profil_VE_df:
    :param Params_VE_df:
    :param year:
    :param T0:
    :param T1:
    :param temperatureName:
    :param TimeName:
    :param VLloadName:
    :param PLloadName:
    :param BusloadName:
    :param VLThermoName:
    :param PLThermoName:
    :param BusThermoName:
    :param ElName:
    :param HybridName:
    :param H2Name:
    :param ConsoElName:
    :param ConsoHybridName:
    :param ConsoH2Name:
    :param ProgressElName:
    :param ProgressHybridName:
    :param ProgressH2Name:
    :param DistName:
    :param VPName:
    :param VULName:
    :param PLName:
    :param BusName:
    :param CarName:
    :param year_ref:
    :param year_end_progress:
    :return: E_H2 in MWh, electric vehicle load Conso_VE in MW
    '''
    temperature_new_df = temperature_df.assign(
        day=temperature_df.index.get_level_values(TimeName).to_series().dt.weekday,
        hour=temperature_df.index.get_level_values(TimeName).to_series().dt.hour,
        Delta_T_thermo=0)

    temperature_new_df["Delta_T_thermo"]=temperature_new_df[temperatureName].apply(lambda T: T - T0 if T <= T0 else 0) \
        - temperature_new_df[temperatureName].apply(lambda T: T - T1 if T >= T1 else 0)

    L_week = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]

    temperature_new_df['day'] = temperature_new_df['day']. \
        apply(lambda x: L_week[x])
    temperature_new_df = temperature_new_df.reset_index().set_index(["day", "hour"])

    Profil_VE_new_df = Profil_VE_df.reset_index().set_index(["day", "hour"])
    temperature_new_df = temperature_new_df.join(Profil_VE_new_df, how="right")

    L_years = list(N_VP_df.index)
    if year <= L_years[0]:
        year_ret=L_years[0]
    elif year >= L_years[-1]:
        year_ret=L_years[-1]
    else:
        year_ret=year

    L_elec=[(1-Params_VE_df.loc[VPName,ProgressElName]*(min(year_ret,year_end_progress)-year_ref))\
            *Params_VE_df.loc[VPName,ConsoElName]*Params_VE_df.loc[VPName,DistName]*N_VP_df.loc[year_ret,ElName]\
            +(1-Params_VE_df.loc[VULName,ProgressElName]*(min(year_ret,year_end_progress)-year_ref))\
            *Params_VE_df.loc[VULName,ConsoElName]*Params_VE_df.loc[VULName,DistName]*N_VUL_df.loc[year_ret,ElName],
            (1 - Params_VE_df.loc[PLName, ProgressElName] * (min(year_ret, year_end_progress) - year_ref))\
            * Params_VE_df.loc[PLName, ConsoElName] * Params_VE_df.loc[PLName, DistName] * N_PL_df.loc[year_ret,ElName],
            (1 - Params_VE_df.loc[BusName, ProgressElName] * (min(year_ret, year_end_progress) - year_ref))\
            * Params_VE_df.loc[BusName, ConsoElName] * Params_VE_df.loc[BusName, DistName] * N_bus_df.loc[year_ret, ElName]\
            +(1 - Params_VE_df.loc[CarName, ProgressElName] * (min(year_ret, year_end_progress) - year_ref))\
            * Params_VE_df.loc[CarName, ConsoElName] * Params_VE_df.loc[CarName, DistName] * N_car_df.loc[year_ret, ElName]]
    # in kWh

    L_hybrid=[(1-Params_VE_df.loc[VPName,ProgressHybridName]*(min(year_ret,year_end_progress)-year_ref))\
            *Params_VE_df.loc[VPName,ConsoHybridName]*Params_VE_df.loc[VPName,DistName]*N_VP_df.loc[year_ret,HybridName]\
            +(1-Params_VE_df.loc[VULName,ProgressHybridName]*(min(year_ret,year_end_progress)-year_ref))\
            *Params_VE_df.loc[VULName,ConsoHybridName]*Params_VE_df.loc[VULName,DistName]*N_VUL_df.loc[year_ret,HybridName],
            (1 - Params_VE_df.loc[PLName, ProgressHybridName] * (min(year_ret, year_end_progress) - year_ref))\
            * Params_VE_df.loc[PLName, ConsoHybridName] * Params_VE_df.loc[PLName, DistName] * N_PL_df.loc[year_ret,HybridName],
            (1 - Params_VE_df.loc[BusName, ProgressHybridName] * (min(year_ret, year_end_progress) - year_ref))\
            * Params_VE_df.loc[BusName, ConsoHybridName] * Params_VE_df.loc[BusName, DistName] * N_bus_df.loc[year_ret, HybridName]\
            +(1 - Params_VE_df.loc[CarName, ProgressHybridName] * (min(year_ret, year_end_progress) - year_ref))\
            * Params_VE_df.loc[CarName, ConsoHybridName] * Params_VE_df.loc[CarName, DistName] * N_car_df.loc[year_ret, HybridName]]
    # in kWh

    E_H2=((1-Params_VE_df.loc[VPName,ProgressH2Name]*(min(year_ret,year_end_progress)-year_ref))\
            *Params_VE_df.loc[VPName,ConsoH2Name]*Params_VE_df.loc[VPName,DistName]*N_VP_df.loc[year_ret,H2Name]\
            +(1-Params_VE_df.loc[VULName,ProgressH2Name]*(min(year_ret,year_end_progress)-year_ref))\
            *Params_VE_df.loc[VULName,ConsoH2Name]*Params_VE_df.loc[VULName,DistName]*N_VUL_df.loc[year_ret,H2Name]\
            +(1 - Params_VE_df.loc[PLName, ProgressH2Name] * (min(year_ret, year_end_progress) - year_ref))\
            * Params_VE_df.loc[PLName, ConsoH2Name] * Params_VE_df.loc[PLName, DistName] * N_PL_df.loc[year_ret,H2Name]\
            +(1 - Params_VE_df.loc[BusName, ProgressH2Name] * (min(year_ret, year_end_progress) - year_ref))\
            * Params_VE_df.loc[BusName, ConsoH2Name] * Params_VE_df.loc[BusName, DistName] * N_bus_df.loc[year_ret, H2Name]\
            +(1 - Params_VE_df.loc[CarName, ProgressH2Name] * (min(year_ret, year_end_progress) - year_ref))\
            * Params_VE_df.loc[CarName, ConsoH2Name] * Params_VE_df.loc[CarName, DistName] * N_car_df.loc[year_ret, H2Name])/1e3
    # in MWh

    temperature_new_df[VLloadName]+=temperature_new_df["Delta_T_thermo"]*temperature_new_df[VLThermoName]
    temperature_new_df[PLloadName] += temperature_new_df["Delta_T_thermo"] * temperature_new_df[PLThermoName]
    temperature_new_df[BusloadName] += temperature_new_df["Delta_T_thermo"] * temperature_new_df[BusThermoName]

    S_VL_load=temperature_new_df[VLloadName].sum()
    S_PL_load=temperature_new_df[PLloadName].sum()
    S_bus_car_load=temperature_new_df[BusloadName].sum()

    temperature_new_df[VLloadName]=temperature_new_df[VLloadName]*(L_elec[0]+L_hybrid[0])/(S_VL_load*1e3)
    temperature_new_df[PLloadName]=temperature_new_df[PLloadName]*(L_elec[1]+L_hybrid[1])/(S_PL_load*1e3)
    temperature_new_df[BusloadName] = temperature_new_df[BusloadName] * (L_elec[2] + L_hybrid[2]) / (S_bus_car_load * 1e3)

    temperature_new_df.assign(Conso_VE=0)
    temperature_new_df["Conso_VE"]=temperature_new_df[VLloadName]+temperature_new_df[PLloadName]+temperature_new_df[BusloadName]

    temperature_new_df = temperature_new_df.reset_index().set_index(TimeName).sort_index()
    return temperature_new_df[["Conso_VE"]],E_H2

def ConsoH2(Conso_H2_df,year,reindus='reindus',
            refName='Reference',reindusName='Reindustrialisation',UNIDENName='UNIDEN'):
    '''
    Returns hydrogen consumption, vehicles not included.

    :param Conso_H2_df:
    :param year:
    :param reindus:
    :param refName:
    :param reindusName:
    :return: Result in MWh
    '''
    if reindus=='reindus':
        name=reindusName
    elif reindus=='no_reindus':
        name=refName
    else:
        name=UNIDENName

    L_years = list(Conso_H2_df.index)
    if year <= L_years[0]:
        E_H2=Conso_H2_df.loc[L_years[0], name]*1e6
    elif year >= L_years[-1]:
        E_H2=Conso_H2_df.loc[L_years[-1], name]*1e6
    else:
        i = 0
        while i < len(L_years) and year >= L_years[i]:
            i += 1
        E_H2 = (Conso_H2_df.loc[L_years[i-1], name]+(year-L_years[i-1])/(L_years[i]-L_years[i-1])* \
                (Conso_H2_df.loc[L_years[i], name]-Conso_H2_df.loc[L_years[i-1], name]))*1e6
    return E_H2

def Losses(temperature_df,T_ref=20,taux_pertes=0.06927,rho_pertes=-1.2e-3,
           temperatureName="temperature"):
    '''
    Computes the losses (thermosensitive).

    :param temperature_df:
    :param T_ref:
    :param taux_pertes:
    :param rho_pertes:
    :param temperatureName:
    :return: Dataframe of losses in percent of the consumption.
    '''

    temperature_new_df=temperature_df.assign(Taux_pertes=taux_pertes)
    temperature_new_df["Taux_pertes"]+=rho_pertes*(temperature_new_df[temperatureName]-T_ref)

    return temperature_new_df[["Taux_pertes"]]

def CleanCETIndex(Temp_df,TimeName="date"):
    '''
    To clean winter and summer hour index in temperature (or other) dataframe.

    :param Temp_df:
    :param TimeName:
    :return:
    '''
    Temp_df_new=Temp_df.reset_index()
    Temp_df_new[TimeName]=Temp_df_new[TimeName].apply(lambda x: x.to_pydatetime())
    d1h=timedelta(hours=1)
    for i in Temp_df_new.index:
        if i>0 and Temp_df_new.loc[i,TimeName]-Temp_df_new.loc[i-1,TimeName]>d1h:
            i_start=i
        if i>0 and Temp_df_new.loc[i,TimeName]==Temp_df_new.loc[i-1,TimeName]:
            i_end=i
    for i in range(i_start,i_end):
        Temp_df_new.loc[i, TimeName]=Temp_df_new.loc[i, TimeName]-d1h
    Temp_df_new[TimeName]=Temp_df_new[TimeName].apply(lambda x: pd.Timestamp(x))
    Temp_df_new=Temp_df_new.set_index(TimeName)
    return Temp_df_new



def CleanProfile(df,Nature,type,Usages,UsagesGroupe):
    df=df.assign(Nature=df.loc[:,"Branche Nom"]).replace({"Nature": Nature})
    df=df.assign(type=df.loc[:,"Branche Nom"]).replace({"type": type}).drop(columns=['Branche Nom'])
    df_melted = pd.melt(frame=df,id_vars=['Mois',"heures","Nature","type"], var_name="Usage", value_name="Conso")
    df_melted=df_melted.assign(UsageDetail=df_melted.Usage).replace({"UsageDetail": Usages})
    df_melted=df_melted.assign(UsagesGroupe=df_melted.Usage).replace({"UsagesGroupe": UsagesGroupe}).drop(columns=["Usage"])
    df_melted=df_melted.set_index(['Mois',"heures","Nature","type","UsagesGroupe","UsageDetail"])
    return(df_melted)


#region initialisation variables
Nature_PROFILE={'AUT Autres (dont Branche Θnergie, Transport et Agriculture)':"Autre",
        'IND Agroalimentaires' :"Agroalimentaires",
        'IND Chimie & matΘriaux plastiques ' :"ChimieMateriaux",
        'IND Composants Θlectriques et Θlectroniques':"ComposantsElec",
        'IND Construction MΘcanique':"ConstructionMecanique",
        'IND Construction navale et aΘronautique, armement':"ConstructionNavArm",
        'IND ╔quipement de transport ':"EquipTransport",
        'IND Fonderie, travail des mΘtaux':"FonderieMetaux",
        'IND Minerai, mΘtal ferreux/non ferreux ':"MineraiMetal",
        'IND Papier':"papier",
        'IND Parachimie, industrie pharmaceutique':"ParachimiePharma",
        'IND Textile ':"Textile",
        'RΘsidence Principale':"Principal",
        'RΘsidence Secondaire':"Secondaire",
        'SPE MatΘriaux de construction':"MatConstruction",
        'SPE Verre':"MatVerre",
        'Tertiaire - Autres': "Autres",
        'Tertiaire - CHR': "CHR",
        'Tertiaire - COM': "COM",
        'Tertiaire - ENS': "ENS",
        'Tertiaire - EP': "EP",
        'Tertiaire - HCO': "HCO",
        'Tertiaire - SAN': "SAN",
        'Tertiaire - SPL': "SPL",
        'Tertiaire - TRA': "TRA",
        'Tertiaire - ADM-BUR': "ADMBUR"}
type_PROFILE={  'AUT Autres (dont Branche Θnergie, Transport et Agriculture)':"Autre",
        'IND Agroalimentaires' :"Ind",
        'IND Chimie & matΘriaux plastiques ' :"Ind",
        'IND Composants Θlectriques et Θlectroniques':"Ind",
        'IND Construction MΘcanique':"Ind",
        'IND Construction navale et aΘronautique, armement':"Ind",
        'IND ╔quipement de transport ':"Ind",
        'IND Fonderie, travail des mΘtaux':"Ind",
        'IND Minerai, mΘtal ferreux/non ferreux ':"Ind",
        'IND Papier':"Ind",
        'IND Parachimie, industrie pharmaceutique':"Ind",
        'IND Textile ':"Ind",
        'RΘsidence Principale':"Residentiel",
        'RΘsidence Secondaire':"Residentiel",
        'SPE MatΘriaux de construction':"Ind",
        'SPE Verre':"Ind",
        'Tertiaire - Autres': "Tertiaire",
        'Tertiaire - CHR': "Tertiaire",
        'Tertiaire - COM': "Tertiaire",
        'Tertiaire - ENS': "Tertiaire",
        'Tertiaire - EP': "Tertiaire",
        'Tertiaire - HCO': "Tertiaire",
        'Tertiaire - SAN': "Tertiaire",
        'Tertiaire - SPL': "Tertiaire",
        'Tertiaire - TRA': "Tertiaire",
        'Tertiaire - ADM-BUR': "Tertiaire"}
Usages_PROFILE={
        'Somme de Lave-Linge':"LaveLinge",
        'Somme de Lave-vaisselle':"Lavevaisselle",
        'Somme de SΦche-linge':"SecheLinge",
        'Somme de RΘfrigΘrateurs et combinΘs':"Refrigirateur",
        'Somme de CongΘlateurs': "Congelateur",
        'Somme de TΘlΘvisions':"TV",
        'Somme de Lecteurs DVD':"LecteurDVD",
        'Somme de MagnΘtoscopes':"Magnetoscope",
        'Somme de Ordinateurs fixes, secondaires et portables':"Ordis",
        'Somme de DΘcodeurs, TNT, Satellite et TV ADSL':"DecodeurTV",
        'Somme de Imprimantes':"Imprimante",
        'Somme de Boxs et modems':"Box",
        'Somme de TΘlΘphones fixes':"TelFixe",
        'Somme de Chaεne-Hifi':"HiFi",
        'Somme de Consoles de jeux':"JeuxVideos",
        'Somme de Grille Pain':"GrillePain",
        'Somme de Bouilloires Θlectriques':"Bouilloires",
        'Somme de CafetiΦres filtres / Machine α expresso':"Cafetiaire",
        'Somme de Micro-ondes':"MicroOnde",
        'Somme de Mini-four':"MiniFour",
        'Somme de Friteuses':"Fritteuses",
        'Somme de Hottes':"Hottes",
        'Somme de CuisiniΦres':"Cuisiniaire",
        'Somme de Aspirateur':"Aspirateur",
        'Somme de Fers α repasser / Centrales vapeures':"FerRepasser",
        'Somme de Ordinateurs':"Ordis",
        'Somme de Imprimantes / Photocopieurs / Scanners':"Imprimante",
        'Somme de Serveurs':"Serveurs",
        'Somme de Faxs':"Faxs",
        'Somme de Traceurs':"Traceurs",
        'Somme de Cuisson':"Cuisson",
        'Somme de ElectromΘnager':"ElectroMenager",
        'Somme de Froid Alimentaire':"FroidAlimentaire",
        'Somme de Process':"Process",
        'Somme de Energie':"Energie",
        'Somme de Agriculture':"Agriculture",
        'Somme de Auxiliaire de Chauffage':"AuxiliaireChauffage",
        'Somme de Chauffage':"Chauffage",
        'Somme de Eclairage Public':"EclairagePublic",
        'Somme de Eclairage':"Eclairage",
        'Somme de Climatisation':"Clim",
        'Somme de ECS':"ECS",
        'Somme de Autres (Inconnu)':"Autres",
        'Somme de Pertes':"Pertes"
}
UsagesGroupe_PROFILE={
        'Somme de Lave-Linge':"LaveLinge",
        'Somme de Lave-vaisselle':"Lavevaisselle",
        'Somme de SΦche-linge':"SecheLinge",
        'Somme de RΘfrigΘrateurs et combinΘs':"Refrigirateur",
        'Somme de CongΘlateurs': "Congelateur",
        'Somme de TΘlΘvisions':"TVAutreElectroMen",
        'Somme de Lecteurs DVD':"TVAutreElectroMen",
        'Somme de MagnΘtoscopes':"TVAutreElectroMen",
        'Somme de Ordinateurs fixes, secondaires et portables':"Ordis",
        'Somme de DΘcodeurs, TNT, Satellite et TV ADSL':"TVAutreElectroMen",
        'Somme de Imprimantes':"TVAutreElectroMen",
        'Somme de Boxs et modems':"Ordis",
        'Somme de TΘlΘphones fixes':"Ordis",
        'Somme de Chaεne-Hifi':"TVAutreElectroMen",
        'Somme de Consoles de jeux':"TVAutreElectroMen",
        'Somme de Grille Pain':"Cuisson",
        'Somme de Bouilloires Θlectriques':"Cuisson",
        'Somme de CafetiΦres filtres / Machine α expresso':"Cuisson",
        'Somme de Micro-ondes':"Cuisson",
        'Somme de Mini-four':"Cuisson",
        'Somme de Friteuses':"Cuisson",
        'Somme de Hottes':"Cuisson",
        'Somme de CuisiniΦres':"Cuisson",
        'Somme de Aspirateur':"TVAutreElectroMen",
        'Somme de Fers α repasser / Centrales vapeures':"TVAutreElectroMen",
        'Somme de Ordinateurs':"Ordis",
        'Somme de Imprimantes / Photocopieurs / Scanners':"Ordis",
        'Somme de Serveurs':"Ordis",
        'Somme de Faxs':"Ordis",
        'Somme de Traceurs':"Ordis",
        'Somme de Cuisson':"Cuisson",
        'Somme de ElectromΘnager':"TVAutreElectroMen",
        'Somme de Froid Alimentaire':"FroidAlimentaire",
        'Somme de Process':"Process",
        'Somme de Energie':"Energie",
        'Somme de Agriculture':"Agriculture",
        'Somme de Auxiliaire de Chauffage':"Chauffage",
        'Somme de Chauffage':"Chauffage",
        'Somme de Eclairage Public':"Eclairage",
        'Somme de Eclairage':"Eclairage",
        'Somme de Climatisation':"Clim",
        'Somme de ECS':"ECS",
        'Somme de Autres (Inconnu)':"Autres",
        'Somme de Pertes':"Pertes"
}
#endregion


