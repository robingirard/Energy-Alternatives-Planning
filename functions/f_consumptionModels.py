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


