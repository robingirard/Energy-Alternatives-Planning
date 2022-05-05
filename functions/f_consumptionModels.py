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

#data_df=areaConsumption.join(ConsoTempe_df)[['areaConsumption','Temperature']]
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
        indexesWinterHour = (dataNoNA_df[TemperatureName] <= TemperatureThreshold) & (dataNoNA_df.index.get_level_values(TimeName).to_series().dt.hour == hour)
        dataWinterHour_df = dataNoNA_df.loc[indexesWinterHour,:]
        lr = linear_model.LinearRegression().fit(dataWinterHour_df[[TemperatureName]],dataWinterHour_df[ConsumptionName])
        Thermosensibilite[hour]=lr.coef_[0]
        ConsoSeparee_df.loc[indexesWinterHour,'TS_C']=Thermosensibilite[hour]*dataWinterHour_df.loc[:, TemperatureName]-Thermosensibilite[hour]*TemperatureThreshold
        ConsoSeparee_df.loc[indexesWinterHour,'NTS_C']=dataWinterHour_df.loc[:, ConsumptionName]-ConsoSeparee_df.TS_C.loc[indexesWinterHour]

    return(ConsoSeparee_df, Thermosensibilite)

def Decomposeconso2(data_df,T0=15,T1=20,TemperatureName='Temperature',
                    ConsumptionName='Consumption',TimeName='Date'):
    '''
    Function decomposing the consumption into thermosensitive and non-thermosensitive part
    taking into account air condition in summer.

    Parameters
    ----------
    data_df : panda data frame with "Temperature" and "Consumption" as columns.
    T0 : float, optional
        Threshold temperature for heating in winter. The default is 15.
    T1 : TYPE, optional
        Threshold temperature for air condition in summer. The default is 20.
    TemperatureName : str, optional
        The default is 'Temperature'.
    ConsumptionName : str, optional
        The default is 'Consumption'.
    TimeName : str, optional
        The default is 'Date'.

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
        indexesWinterHour = (dataNoNA_df[TemperatureName] <= T0) & (dataNoNA_df.index.get_level_values(TimeName).to_series().dt.hour == hour)
        indexesSummerHour = (dataNoNA_df[TemperatureName] >= T1) & (dataNoNA_df.index.get_level_values(TimeName).to_series().dt.hour == hour)
        dataWinterHour_df = dataNoNA_df.loc[indexesWinterHour,:]
        dataSummerHour_df = dataNoNA_df.loc[indexesSummerHour,:]
        lrw = linear_model.LinearRegression().fit(dataWinterHour_df[[TemperatureName]],dataWinterHour_df[ConsumptionName])
        Thermosensitivity_winter[hour]=lrw.coef_[0]
        lrs = linear_model.LinearRegression().fit(dataSummerHour_df[[TemperatureName]],dataSummerHour_df[ConsumptionName])
        Thermosensitivity_summer[hour]=lrs.coef_[0]
        ConsoSeparee_df.loc[indexesWinterHour,'TSW_C']=Thermosensitivity_winter[hour]*dataWinterHour_df.loc[:, TemperatureName]-Thermosensitivity_winter[hour]*T0
        ConsoSeparee_df.loc[indexesWinterHour,'NTS_C']=dataWinterHour_df.loc[:, ConsumptionName]-ConsoSeparee_df.TSW_C.loc[indexesWinterHour]
        ConsoSeparee_df.loc[indexesSummerHour,'TSS_C']=Thermosensitivity_summer[hour]*dataSummerHour_df.loc[:, TemperatureName]-Thermosensitivity_summer[hour]*T1
        ConsoSeparee_df.loc[indexesSummerHour,'NTS_C']=dataSummerHour_df.loc[:, ConsumptionName]-ConsoSeparee_df.TSS_C.loc[indexesSummerHour]

    return (ConsoSeparee_df[['NTS_C','TSW_C','TSS_C']], Thermosensitivity_winter,Thermosensitivity_summer)

#ConsoSeparee_df=ConsoTempeYear_decomposed_df
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
    ConsoSepareeNew_df.loc[:,"NTS_C"]=ConsoSepareeNew_df.loc[:,"NTS_C"]

    for hour in range(24):
        indexesWinterHour = (ConsoSepareeNew_df[TemperatureName] <= TemperatureThreshold) & (ConsoSepareeNew_df.index.get_level_values(TimeName).to_series().dt.hour== hour)
        ## remove thermal sensitive part according to old temperature
        ConsoSepareeNew_df.loc[indexesWinterHour, 'TS_C'] = Thermosensibilite[hour] * ConsoSepareeNew_df.loc[indexesWinterHour,TemperatureName] - Thermosensibilite[hour] * TemperatureThreshold

    ConsoSepareeNew_df.loc[:,ConsumptionName]= (ConsoSepareeNew_df.loc[:,'TS_C']+ConsoSepareeNew_df.loc[:,'NTS_C'])
    return(ConsoSepareeNew_df)




def Profile2Consumption(Profile_df,Temperature_df, TemperatureThreshold=14,
                        TemperatureMinimum=0,TemperatureName='Temperature',
                        ConsumptionName='Consumption',TimeName='Date',
                        VarName='Puissance.MW.par.million'):
    '''
    fonction permettant de reconstruire la consommation annuelle à partir d'un profil HeurexJourxSaison en une part thermosensible et non thermosensible
    (la conso non thermosensible étant la conso type d'une semaine d'été)

    :param Profile_df: profil avec les colonnes HeurexJourxSaison
    :param Temperature_df:
    :param TemperatureThreshold:
    :param TemperatureMinimum:
    :param TemperatureName:
    :param ConsumptionName:
    :param TimeName:
    :param VarName:
    :return:
    '''
    ## initialisation
    ConsoSepareeNew_df=Temperature_df.loc[:,[TemperatureName]]
    ConsoSepareeNew_df.loc[:,[ConsumptionName]]=np.NaN
    ConsoSepareeNew_df.loc[:,['NTS_C']]=0
    ConsoSepareeNew_df.loc[:,['TS_C']]=0

    PivotedProfile_df = Profile_df.pivot( index=['Heure','Jour'], columns='Saison', values=VarName ).reset_index()
    cte=(TemperatureThreshold-TemperatureMinimum)

    for index, row in PivotedProfile_df.iterrows():
        indexesWD=ConsoSepareeNew_df.index.get_level_values(TimeName).to_series().dt.weekday   == (PivotedProfile_df.loc[index,'Jour']-1)
        indexesHours= ConsoSepareeNew_df.index.get_level_values(TimeName).to_series().dt.hour   == (PivotedProfile_df.loc[index,'Heure']-1)
        ConsoSepareeNew_df.loc[indexesWD&indexesHours, 'NTS_C']=PivotedProfile_df.loc[index,'Ete']

    PivotedProfile_df['NDifference'] = (PivotedProfile_df['Ete'] - PivotedProfile_df['Hiver'])
    Thermosensibilite = (PivotedProfile_df['NDifference'].loc[0:23] / cte).tolist()
    ConsoSepareeNew_df=Recompose(ConsoSepareeNew_df,Thermosensibilite)
    return(ConsoSepareeNew_df)


#Profile_df_Week,Profile_df_Sat,Profile_df_Sun,ConsoTempeYear_df
#Temperature_df=ConsoTempeYear_decomposed_df.loc[:,"NTS_C"]
#Profile_df=NTS_profil
def ComplexProfile2Consumption(Profile_df,
                               Temperature_df, TemperatureThreshold=14,
                        TemperatureMinimum=0,TemperatureName='Temperature',
                        poidsName='poids',
                        ConsumptionName='Consumption',TimeName='Date',
                        VarName='Puissance.MW.par.million',french=True):

    ## initialisation
    ConsoSepareeNew_df=Temperature_df.loc[:,[ConsumptionName]]
    #ConsoSepareeNew_df.loc[:,[ConsumptionName]]=np.NaN
    #ConsoSepareeNew_df.loc[:,['NTS_C']]=0
    #ConsoSepareeNew_df.loc[:,['TS_C']]=0
    ConsoSepareeNew_df = ConsoSepareeNew_df.assign(
        Jour=ConsoSepareeNew_df.index.get_level_values(TimeName).to_series().dt.weekday,
        Mois=ConsoSepareeNew_df.index.get_level_values(TimeName).to_series().dt.month,
        Heure=ConsoSepareeNew_df.index.get_level_values(TimeName).to_series().dt.hour);
    if french:
        ConsoSepareeNew_df['Jour']=ConsoSepareeNew_df['Jour'].\
            apply(lambda x: "Semaine" if x<5 else "Samedi" if x==5 else "Dimanche")
    else:
        ConsoSepareeNew_df['Jour'] = ConsoSepareeNew_df['Jour'].\
            apply(lambda x: "Week" if x < 5 else "Sat" if x == 5 else "Sun")
    ConsoSepareeNew_df=ConsoSepareeNew_df.reset_index().set_index(["Jour","Mois","Heure"])

    Profile_df_merged=Profile_df.join(ConsoSepareeNew_df,how="inner")
    Profile_df_merged.loc[:,[ConsumptionName]]=Profile_df_merged[ConsumptionName]*Profile_df_merged[poidsName]
    return(Profile_df_merged)
    #cte=(TemperatureThreshold-TemperatureMinimum)

# Tient compte des vacances et jours feriés en France en 2019
def ComplexProfile2ConsumptionCJO2019(Profile_df,
                               Temperature_df, TemperatureThreshold=14,
                        TemperatureMinimum=0,TemperatureName='Temperature',
                        poidsName='poids',
                        ConsumptionName='Consumption',TimeName='Date',
                        VarName='Puissance.MW.par.million',french=True):

    ## initialisation
    ConsoSepareeNew_df=Temperature_df.loc[:,[ConsumptionName]]
    #ConsoSepareeNew_df.loc[:,[ConsumptionName]]=np.NaN
    #ConsoSepareeNew_df.loc[:,['NTS_C']]=0
    #ConsoSepareeNew_df.loc[:,['TS_C']]=0
    ConsoSepareeNew_df = ConsoSepareeNew_df.assign(
        Jour=ConsoSepareeNew_df.index.get_level_values(TimeName).to_series().dt.weekday,
        Mois=ConsoSepareeNew_df.index.get_level_values(TimeName).to_series().dt.month,
        Heure=ConsoSepareeNew_df.index.get_level_values(TimeName).to_series().dt.hour);
    if french:
        index_dim_feries=(ConsoSepareeNew_df["Jour"]==6)\
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
        
        index_sam_vacances=((ConsoSepareeNew_df["Jour"]==5)\
        |((ConsoSepareeNew_df.index.month==1)&(ConsoSepareeNew_df.index.day<=6))\
        |((ConsoSepareeNew_df.index.month==8)&(ConsoSepareeNew_df.index.day>=3)&(ConsoSepareeNew_df.index.day<=18))\
        |((ConsoSepareeNew_df.index.month==12)&(ConsoSepareeNew_df.index.day>=21))\
        &(index_dim_feries==False))
        
        index_semaine=(index_dim_feries==False)&(index_sam_vacances==False)
        
        ConsoSepareeNew_df.loc[index_dim_feries,"Jour"]="Dimanche"
        ConsoSepareeNew_df.loc[index_sam_vacances,"Jour"]="Samedi"
        ConsoSepareeNew_df.loc[index_semaine,"Jour"]="Semaine"
    else:
        ConsoSepareeNew_df['Jour'] = ConsoSepareeNew_df['Jour'].\
            apply(lambda x: "Week" if x < 5 else "Sat" if x == 5 else "Sun")
    ConsoSepareeNew_df=ConsoSepareeNew_df.reset_index().set_index(["Jour","Mois","Heure"])

    Profile_df_merged=Profile_df.join(ConsoSepareeNew_df,how="inner")
    Profile_df_merged.loc[:,[ConsumptionName]]=Profile_df_merged[ConsumptionName]*Profile_df_merged[poidsName]
    return(Profile_df_merged)
    #cte=(TemperatureThreshold-TemperatureMinimum)


# Adaptée aux nouvelles données de RTE
def ComplexProfile2Consumption_2(Profile_df,
                                      Temperature_df,poidsName='poids',
                                      ConsumptionName='Consumption', TimeName='Date',GroupName='type'):

    ## Processing dates indexing Temperature
    ConsoSepareeNew_df = Temperature_df.loc[:, [ConsumptionName]]
    ConsoSepareeNew_df = ConsoSepareeNew_df.assign(
        Jour=ConsoSepareeNew_df.index.get_level_values(TimeName).to_series().dt.weekday,
        Mois=ConsoSepareeNew_df.index.get_level_values(TimeName).to_series().dt.month,
        Heure=ConsoSepareeNew_df.index.get_level_values(TimeName).to_series().dt.hour);

    L_week = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]

    ConsoSepareeNew_df['Jour'] = ConsoSepareeNew_df['Jour']. \
            apply(lambda x: L_week[x])
    ConsoSepareeNew_df = ConsoSepareeNew_df.reset_index().set_index(["Mois", "Jour", "Heure"])

    ## Processing the profile to make every month appear
    Profile_df=Profile_df.reset_index()
    Profile_summer=Profile_df[Profile_df.Saison=="Ete"][["Jour","Heure",GroupName,poidsName]].reset_index()# Janvier
    Profile_winter= Profile_df[Profile_df.Saison == "Hiver"][["Jour","Heure",GroupName,poidsName]].reset_index()# Juin

    Profile_month=Profile_winter.copy().assign(Mois=1)
    for month in range(2,13):
        Profile_temp=Profile_winter.copy().assign(Mois=month)
        Profile_temp[poidsName]=Profile_winter[poidsName]*np.cos(np.pi*(month-1)/12)**2\
                                +(Profile_summer[poidsName]-Profile_winter[poidsName]*np.cos(np.pi*5/12)**2)\
                                *np.sin(np.pi*(month-1)/12)**2/np.sin(np.pi*5/12)**2
        Profile_month=pd.concat([Profile_month,Profile_temp],ignore_index=True)
    Profile_month=Profile_month.reset_index().set_index(["Mois","Jour","Heure"])


    Profile_month_merged = ConsoSepareeNew_df.join(Profile_month, how="right")
    Profile_month_merged.loc[:, [ConsumptionName]] = Profile_month_merged[ConsumptionName] * Profile_month_merged[poidsName]
    return Profile_month_merged.reset_index()[[ConsumptionName,TimeName,GroupName]].\
        groupby([TimeName,GroupName]).sum().reset_index().\
        pivot(index=TimeName, columns=GroupName, values=ConsumptionName)
    # cte=(TemperatureThreshold-TemperatureMinimum)

def colReindus(col,reindus=False,industryName='Industrie hors metallurgie',steelName='Metallurgie',
                       reindusName='reindustrialisation'):
    if reindus and col in [industryName,steelName]:
        return col+' '+reindusName
    else:
        return col

def ProjectionConsoNTS(Conso_profile_df,Projections_df,year,reindus=False,
                       industryName='Industrie hors metallurgie',steelName='Metallurgie',
                       reindusName='reindustrialisation'):
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
            col_proj=colReindus(col,reindus,industryName,steelName,reindusName)
            Conso_profile_new_df[col]=Projections_df.loc[L_years[0],col_proj]*Conso_profile_new_df[col]
    elif year>=L_years[-1]:
        for col in L_cols:
            col_proj = colReindus(col, reindus, industryName, steelName, reindusName)
            Conso_profile_new_df[col] = Projections_df.loc[L_years[-1], col_proj] * Conso_profile_new_df[col]
    else:
        i=0
        while i<len(L_years) and year>=L_years[i]:
            i+=1
        for col in L_cols:
            col_proj = colReindus(col, reindus, industryName, steelName, reindusName)
            Conso_profile_new_df[col] = (Projections_df.loc[L_years[i-1], col_proj]+(year-L_years[i-1])/(L_years[i]-L_years[i-1])*(Projections_df.loc[L_years[i], col_proj]-Projections_df.loc[L_years[i-1], col_proj])) * Conso_profile_new_df[col]

    Conso_profile_new_df=Conso_profile_new_df.assign(Total=0)
    for col in L_cols:
        Conso_profile_new_df["Total"]+=Conso_profile_new_df[col]
    return Conso_profile_new_df[["Total"]]


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


