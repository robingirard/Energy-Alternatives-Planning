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

#os.chdir("D:\GIT\Etude_TP_CapaExpPlaning-Python") #pour vous placer dans le dossier où se trouvent les donnnées csv
#ConsoTemp=pd.read_csv('CSV/input/ConsumptionTemperature_1996TO2019-sanschangementheure.csv', parse_dates = True, index_col = 0)

# fonction décomposant la consommation électrique d'une année en une part thermosensible et une part non thermosensible
def Decomposeconso(year, templimite=14) :
    ConsoTemp1=SelectYear(year)
    ConsoSeparee1=np.zeros(shape=(ConsoTemp1.shape[0],2))
    ConsoSeparee=pd.DataFrame(data=ConsoSeparee1, index=ConsoTemp1.index, columns=['Conso thermo','Conso Nonthermo'])
    ConsoSeparee['Conso Nonthermo']=ConsoTemp1['Consumption']
    ConsoSeparee['Temperature']=ConsoTemp1['Temperature'].values
    Thermosensibilite1=np.zeros((24,1))
    Thermosensibilite=pd.DataFrame(data=Thermosensibilite1, columns=['Thermosensibilite'])
    Thermosensibilite.index.name='Heure de la journee'
    for hour in range(24):
        match_timestamp=time(hour).isoformat()
        tabhour=ConsoTemp1.loc[ConsoTemp1.index.strftime("%H:%M:%S") == match_timestamp]
        (date, DJU)=GetDatesDJU(tabhour,templimite)
        n=len(date)
        Tabreglin1=np.zeros((n,2))
        Tabreglin=pd.DataFrame(data=Tabreglin1, columns=['Conso','Température'])
        for j in range(n):
            Tabreglin.iloc[j,0]=ConsoTemp1.loc[date[j]][0]
            Tabreglin.iloc[j,1]=ConsoTemp1.loc[date[j]][1]
        x=Tabreglin.Température.values.reshape(n,1)
        y=Tabreglin.Conso.values.reshape(n,1)
        lr=linear_model.LinearRegression().fit(x,y)
        thermosensibilitecoef=lr.coef_[0][0]
        Thermosensibilite.iloc[hour,0]=thermosensibilitecoef
        for k in range(n):
            ConsoSeparee.loc[date[k]][0]=DJU[k]*thermosensibilitecoef
    ConsoSeparee['Conso Nonthermo']=ConsoSeparee['Conso Nonthermo']-ConsoSeparee['Conso thermo']
#   pd.set_option('display.max_rows', None)
    return(ConsoSeparee, Thermosensibilite)

# fonction qui extrait les dates et heures et les DJU d'une timeserie pour les heures de l'année où la température est inférieure à templimite (ici 14°C)
def GetDatesDJU(Tablconsotemp, templimite=14) :
    dates=[]
    DJU=[]
    for i in range(Tablconsotemp.shape[0]):
        if Tablconsotemp['Temperature'].iloc[i]<templimite:
            dates.append(Tablconsotemp.index[i])
            DJU.append(Tablconsotemp['Temperature'].iloc[i]-templimite)
    return(dates,DJU)

# fonction permettant de redécomposer la conso électrique en part thermosensible et non thermosensible de l'année x à partir de la thermosensibilité (calculée pour chaque heure de la journée) de l'année x et les température d'une année y
## BE CAREFUL WITH THE BISEXTIL YEAR FOR THE DIMENSION MATCH
def ChangeTemperature(decomposedconso, thermosensibilite, yeartemperature, templimite=14):
    temphourly=SelectYear(yeartemperature)
    decomposedconso['Temperature']=temphourly['Temperature'].values
    for hour in range(24):
        match_timestamp=time(hour).isoformat()
        tabhour=decomposedconso.loc[decomposedconso.index.strftime("%H:%M:%S") == match_timestamp]
        (date, DJU)=GetDatesDJU(tabhour,templimite)
        for k in range(len(date)):
            decomposedconso.loc[date[k]][0]=DJU[k]*thermosensibilite.iloc[hour,0]
    return(decomposedconso)

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
def EstimateThermosensibilite(year, templimite=14):
    ConsoYear=SelectYear(year)
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

year0=2013
Consotmp=SelectYear(year0)
# plt.plot(Consotmp['Consumption'])
# plt.show()

# fonction permettant de décomposer la consommation annuelle d'un véhicule électrique en une part thermosensible et non thermosensible (la conso non thermosensible étant la conso type d'une semaine d'été)
def GetVEconso(year, templimite=14, mintemp=0):
    TempAnnee=SelectYear(year)
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