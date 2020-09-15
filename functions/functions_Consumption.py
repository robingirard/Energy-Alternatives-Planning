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



def Decomposeconso(data_df, TemperatureThreshold=14, TemperatureName='Temperature',ConsumptionName='Consumption',TimeName='Date') :
    '''
    fonction décomposant la consommation électrique d'une année en une part thermosensible et une part non thermosensible
    :param data: panda data frame with "Temperature" and "Consumption" as columns
    :param TemperatureThreshold: the threshold heating temperature
    :param TemperatureName default 'Temperature' name of column with Temperature
    :param ConsumptionName default 'Consumption' name of column with consumption
    :param TimeName default 'Date' name of column with time
    :return: a dictionary with Thermosensibilite, and a panda data frame with two new columns NTS_C and TS_C
    '''
    dataNoNA_df=data_df.dropna()
    ## Remove NA
    ConsoSeparee_df=dataNoNA_df.assign(NTS_C=dataNoNA_df[ConsumptionName], TS_C=dataNoNA_df[ConsumptionName]*0)

    Thermosensibilite={}
    #pd.DataFrame(data=np.zeros((24,1)), columns=['Thermosensibilite'])## one value per hour of the day
    #Thermosensibilite.index.name='hour of the day'
    for hour in range(24):
        indexesWinterHour = (dataNoNA_df[TemperatureName] <= TemperatureThreshold) & (pd.to_datetime(dataNoNA_df[TimeName]).dt.hour == hour)
        dataWinterHour_df = dataNoNA_df.loc[indexesWinterHour,:]
        lr = linear_model.LinearRegression().fit(dataWinterHour_df[[TemperatureName]],dataWinterHour_df[ConsumptionName])
        Thermosensibilite[hour]=lr.coef_[0]
        ConsoSeparee_df.loc[indexesWinterHour,'TS_C']=Thermosensibilite[hour]*dataWinterHour_df.loc[:, TemperatureName]-Thermosensibilite[hour]*TemperatureThreshold
        ConsoSeparee_df.loc[indexesWinterHour,'NTS_C']=dataWinterHour_df.loc[:, ConsumptionName]-ConsoSeparee_df.TS_C.loc[indexesWinterHour]

    return(ConsoSeparee_df, Thermosensibilite)



def Recompose(ConsoSeparee_df,Thermosensibilite,Newdata_df=-1, TemperatureThreshold=14,TemperatureName='Temperature',ConsumptionName='Consumption',TimeName='Date'):
    '''
    fonction permettant de redécomposer la conso électrique en part thermosensible et
    non thermosensible de l'année x à partir de la thermosensibilité
    (calculée pour chaque heure de la journée) de l'année x et les température d'une année y
    should handle the dimension match problem related to bisextile years
    :param ConsoSeparee_df:
    :param Newdata_df:
    :param Thermosensibilite:
    :param TemperatureThreshold:
    :param TemperatureName:
    :param ConsumptionName:
    :param TimeName:
    :return:
    '''
    if (Newdata_df==-1): Newdata_df=ConsoSeparee_df
    indexes_Old = np.nonzero(np.in1d(np.arange(0,ConsoSeparee_df.__len__()), np.arange(0,Newdata_df[TemperatureName].__len__())))[0]
    indexes_New = np.nonzero(np.in1d(np.arange(0,Newdata_df[TemperatureName].__len__()), np.arange(0,ConsoSeparee_df.__len__())))[0]

    ConsoSepareeNew_df=ConsoSeparee_df.iloc[indexes_Old,:]
    ConsoSepareeNew_df.iloc[:, :][TemperatureName]=Newdata_df.iloc[indexes_New, :][TemperatureName].tolist()
    ConsoSepareeNew_df.TS_C=0
    ConsoSepareeNew_df.NTS_C=ConsoSeparee_df.NTS_C

    for hour in range(24):
        indexesWinterHour = (ConsoSepareeNew_df[TemperatureName] <= TemperatureThreshold) & (pd.to_datetime(ConsoSepareeNew_df[TimeName]).dt.hour == hour)
        ## remove thermal sensitive part according to old temperature
        ConsoSepareeNew_df.loc[indexesWinterHour, 'TS_C'] = Thermosensibilite[hour] * ConsoSepareeNew_df.loc[:,TemperatureName] - Thermosensibilite[hour] * TemperatureThreshold

    ConsoSepareeNew_df[ConsumptionName]=ConsoSepareeNew_df.TS_C+ConsoSepareeNew_df.NTS_C
    return(ConsoSepareeNew_df)

# fonction permettant de redécomposer la conso électrique en une part thermosensible et non thermosensible avec un nouveau tableau de thermosensibilité
def RecomposeTemperature(decomposedconso, newthermosensibilite, templimite=14):
    for hour in range(24):
        match_timestamp=time(hour).isoformat()
        tabhour=decomposedconso.loc[decomposedconso.index.strftime("%H:%M:%S") == match_timestamp]
        (date, DJU)=GetDatesDJU(tabhour,templimite)
        for k in range(len(date)):
            decomposedconso.loc[date[k]][0]=DJU[k]*newthermosensibilite.iloc[hour,0]
    return(decomposedconso)

# fonction qui fournit le tableau de thermosensibilité (pour chaque heure de la journée) pour une année choisie
def EstimateThermosensibilite(year, templimite=14,data='CSV/input/ConsumptionTemperature_1996TO2019-sanschangementheure.csv'):
    ConsoYear=SelectYear(year,data)
    Thermosensibilite1=np.zeros((24,1))
    Thermosensibilite=pd.DataFrame(data=Thermosensibilite1, columns=['Thermosensibilite'])
    Thermosensibilite.index.name='Heure de la journee'
    for hour in range(24):
        match_timestamp=time(hour).isoformat()
        tabhour=ConsoYear.loc[ConsoYear.index.strftime("%H:%M:%S") == match_timestamp]
        (date, DJU)=GetDatesDJU(tabhour,templimite)
        n=len(date)
        Tabreglin1=np.zeros((n,2))
        Tabreglin=pd.DataFrame(data=Tabreglin1, columns=['Conso','Température'])
        for j in range(n):
            Tabreglin.iloc[j,0]=ConsoYear.loc[date[j]][0]
            Tabreglin.iloc[j,1]=ConsoYear.loc[date[j]][1]
        x=Tabreglin.Température.values.reshape(n,1)
        y=Tabreglin.Conso.values.reshape(n,1)
        lr=linear_model.LinearRegression().fit(x,y)
        thermosensibilitecoef=lr.coef_[0][0]
        Thermosensibilite.iloc[hour,0]=thermosensibilitecoef
    return(Thermosensibilite)

# fonction qui permet d'extraire la timiserie de l'année qui nous intéresse à partir d'une timeserie qui couvre les années de 1996 à 2018
def SelectYear(year,data='CSV/input/ConsumptionTemperature_1996TO2019-sanschangementheure.csv'):
    ConsoTemp=pd.read_csv(data, parse_dates = True, index_col = 0)
    if year>=1996 and year<=2018 :
        date_start = pd.Timestamp(year, 1, 1, 0)
        date_end =pd.Timestamp(year+1, 1, 1, 0)
        mask=(ConsoTemp.index >= date_start) & (ConsoTemp.index <  date_end )
        ConsoTempan= ConsoTemp.loc[mask]
    else :
        return('Pas de données')
    return(ConsoTempan)


# plt.plot(Consotmp['Consumption'])
# plt.show()

# fonction permettant de décomposer la consommation annuelle d'un véhicule électrique en une part thermosensible et non thermosensible (la conso non thermosensible étant la conso type d'une semaine d'été)
def GetVEconso(year, templimite=14, mintemp=0,data='CSV/input/ConsumptionTemperature_1996TO2019-sanschangementheure.csv'):
    TempAnnee=SelectYear(year,data)
    ConsoAnnee1=np.zeros((TempAnnee.shape[0],3))
    ConsoAnnee=pd.DataFrame(data=ConsoAnnee1, columns=['Conso TH','Conso NTH','Temperature']) #Conso en Puissance.MW.par.million
    ConsoAnnee.index=TempAnnee.index
    ConsoAnnee['Temperature']=TempAnnee['Temperature']
    tmpVE=pd.read_csv('EVModel.csv', sep=';')
    VESummer=tmpVE[tmpVE['Saison'] == 'Ete'].reset_index(drop=True)
    i=0
    while i<TempAnnee.shape[0]:
        ConsoAnnee['Conso NTH'][i]=VESummer['Puissance.MW.par.million'][i%VESummer.shape[0]]
        i=i+1

    difference=tmpVE[tmpVE['Saison'] == 'Hiver']['Puissance.MW.par.million'].reset_index(drop=True)-tmpVE[tmpVE['Saison'] == 'Ete']['Puissance.MW.par.million'].reset_index(drop=True)
    difference.index.name='Heure de la semaine'

    Thermosensibilite1=np.zeros((24,1))
    Thermosensibilite=pd.DataFrame(data=Thermosensibilite1, columns=['Thermosensibilite'])
    Thermosensibilite.index.name='Heure de la semaine'
    Thermosensibilite['Thermosensibilite']=-difference.loc[0:23]/(templimite-mintemp)


    for hour in range(24):
        match_timestamp=time(hour).isoformat()
        tabhour=TempAnnee.loc[TempAnnee.index.strftime("%H:%M:%S") == match_timestamp]
        (date, DJU)=GetDatesDJU(tabhour,templimite)
        for k in range(len(date)):
            ConsoAnnee.loc[date[k]][0]=DJU[k]*Thermosensibilite.iloc[hour,0]
    return(ConsoAnnee)