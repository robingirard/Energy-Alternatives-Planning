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
    if (Newdata_df.__class__==int): Newdata_df=ConsoSeparee_df
    indexes_Old = np.nonzero(np.in1d(np.arange(0,ConsoSeparee_df.__len__()), np.arange(0,Newdata_df[TemperatureName].__len__())))[0]
    indexes_New = np.nonzero(np.in1d(np.arange(0,Newdata_df[TemperatureName].__len__()), np.arange(0,ConsoSeparee_df.__len__())))[0]

    ConsoSepareeNew_df=ConsoSeparee_df.iloc[indexes_Old,:].copy(deep=True) ## to suppress warning see https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
    ConsoSepareeNew_df.loc[:, TemperatureName]=Newdata_df.iloc[indexes_New, Newdata_df.columns.get_loc(TemperatureName)].tolist()
    ConsoSepareeNew_df.loc[:,"TS_C"]=0
    ConsoSepareeNew_df.loc[:,"NTS_C"]=ConsoSeparee_df.loc[:,"NTS_C"]

    for hour in range(24):
        indexesWinterHour = (ConsoSepareeNew_df[TemperatureName] <= TemperatureThreshold) & (pd.to_datetime(ConsoSepareeNew_df[TimeName]).dt.hour == hour)
        ## remove thermal sensitive part according to old temperature
        ConsoSepareeNew_df.loc[indexesWinterHour, 'TS_C'] = Thermosensibilite[hour] * ConsoSepareeNew_df.loc[:,TemperatureName] - Thermosensibilite[hour] * TemperatureThreshold

    ConsoSepareeNew_df.loc[:,ConsumptionName]= (ConsoSepareeNew_df.loc[:,'TS_C']+ConsoSepareeNew_df.loc[:,'NTS_C'])
    return(ConsoSepareeNew_df)


# fonction permettant de décomposer la consommation annuelle d'un véhicule électrique en une part thermosensible et non thermosensible (la conso non thermosensible étant la conso type d'une semaine d'été)
def Profile2Consumption(Profile_df,Temperature_df, TemperatureThreshold=14,
                        TemperatureMinimum=0,TemperatureName='Temperature',
                        ConsumptionName='Consumption',TimeName='Date',
                        VarName='Puissance.MW.par.million'):

    ## initialisation
    ConsoSepareeNew_df=Temperature_df[[TimeName,TemperatureName]]
    ConsoSepareeNew_df[ConsumptionName]=np.NaN
    ConsoSepareeNew_df['NTS_C']=0
    ConsoSepareeNew_df['TS_C']=0

    PivotedProfile_df = Profile_df.pivot( index=['Heure','Jour'], columns='Saison', values=VarName ).reset_index()
    cte=(TemperatureThreshold-TemperatureMinimum)

    for index, row in PivotedProfile_df.iterrows():
        indexesWD=pd.to_datetime(ConsoSepareeNew_df['Date']).dt.weekday   == (PivotedProfile_df.loc[index,'Jour']-1)
        indexesHours= pd.to_datetime(ConsoSepareeNew_df['Date']).dt.hour   == (PivotedProfile_df.loc[index,'Heure']-1)
        ConsoSepareeNew_df.loc[indexesWD&indexesHours, 'NTS_C']=PivotedProfile_df.loc[index,'Ete']

    PivotedProfile_df['NDifference'] = (PivotedProfile_df['Ete'] - PivotedProfile_df['Hiver'])
    Thermosensibilite = (PivotedProfile_df['NDifference'].loc[0:23] / cte).tolist()
    ConsoSepareeNew_df=Recompose(ConsoSepareeNew_df,Thermosensibilite)
    return(ConsoSepareeNew_df)