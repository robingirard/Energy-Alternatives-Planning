
#region imports
import os
import pandas as pd
import numpy as np
InputFolder='Data/input/'
#endregion

#region ajout des facteurs de charge dy NewNuke et de l'éolien off shore
Zones="FR" ;

for year in range(2013,2017):
    availabilityFactor = pd.read_csv(InputFolder + 'availabilityFactor' + str(year) + '_' + str(Zones) + '.csv',
                                     sep=',', decimal='.', skiprows=0).set_index(["TIMESTAMP", "TECHNOLOGIES"])
    #NewNuke first
    availabilityFactor_NewNuke = availabilityFactor.loc[(slice(None), "OldNuke"), :]
    availabilityFactor = availabilityFactor.append(availabilityFactor_NewNuke.rename(index={"OldNuke": "NewNuke"}))

    #Eolien off shore
    availabilityFactor_WindOffShore = availabilityFactor.loc[(slice(None), "WindOnShore"), :]
    availabilityFactor_WindOffShore = availabilityFactor_WindOffShore.rename(index={"WindOnShore": "WindOffShore"})
    alpha = 1.8
    print(availabilityFactor_WindOffShore.mean())
    availabilityFactor_WindOffShore = availabilityFactor_WindOffShore.assign(
        availabilityFactor=np.where(availabilityFactor_WindOffShore['availabilityFactor'] * alpha >= 1.0, 1.0,
                                    availabilityFactor_WindOffShore[
                                        'availabilityFactor'] * alpha))  # df: 1.0 if df['availabilityFactor']*1.7>=1.0 else df['availabilityFactor']*1.7)
    print(availabilityFactor_WindOffShore.mean())
    availabilityFactor = availabilityFactor.append(availabilityFactor_WindOffShore)
    availabilityFactor.to_csv(InputFolder + 'availabilityFactor_new' + str(year) + '_' + str(Zones) + '.csv',sep=',', decimal='.')

#endregion

#regions creation d'un fichier la consommation toutes zones avec des dates comme index
Zones="FR" ;
for year in range(2013,2017):
    areaConsumption = pd.read_csv(InputFolder + 'areaConsumption' + str(year) + '_' + str(Zones) + '.csv',
                                     sep=',', decimal='.', skiprows=0)
    areaConsumption['Date']=pd.date_range(start='1/1/'+str(year), periods=len(areaConsumption), freq='H')
    areaConsumption=areaConsumption[['Date','areaConsumption']].set_index(["Date"])
    areaConsumption.to_csv(InputFolder + 'areaConsumption_new' + str(year) + '_' + str(Zones) + '.csv',sep=',', decimal='.')

Zones="FR_DE_GB_ES"
year=2016
areaConsumption = pd.read_csv(InputFolder + 'areaConsumption' + str(year) + '_' + str(Zones) + '.csv',sep=',', decimal='.', skiprows=0)
Date=pd.date_range(start='1/1/'+str(year), periods=len(areaConsumption.TIMESTAMP.unique()), freq='H')
TIMESTAMP_asso=pd.DataFrame({'TIMESTAMP' : range(1,len(areaConsumption.TIMESTAMP.unique())+1),'Date' : Date})
areaConsumption['TIMESTAMP']=areaConsumption['TIMESTAMP'].replace(range(1,len(areaConsumption.TIMESTAMP.unique())+1),Date)
areaConsumption=areaConsumption.rename(columns = {'TIMESTAMP' : 'Date'})
areaConsumption=areaConsumption[['Date',"AREAS",'areaConsumption']].set_index(["Date","AREAS"])
areaConsumption.to_csv(InputFolder + 'areaConsumption_new' + str(year) + '_' + str(Zones) + '.csv',sep=',', decimal='.')
#endregions


#regions creation d'un fichier availability toutes zones avec des dates comme index
Zones="FR" ;
for year in range(2013,2017):
    availabilityFactor = pd.read_csv(InputFolder + 'availabilityFactor' + str(year) + '_' + str(Zones) + '.csv',
                                     sep=',', decimal='.', skiprows=0)
    Date=pd.date_range(start='1/1/'+str(year), periods=len(availabilityFactor.TIMESTAMP.unique()), freq='H')
    TIMESTAMP_asso = pd.DataFrame({'TIMESTAMP': range(1, len(availabilityFactor.TIMESTAMP.unique()) + 1), 'Date': Date})
    availabilityFactor['TIMESTAMP'] = availabilityFactor['TIMESTAMP'].replace(range(1, len(availabilityFactor.TIMESTAMP.unique()) + 1), Date)
    availabilityFactor = availabilityFactor.rename(columns={'TIMESTAMP': 'Date'})
    availabilityFactor = availabilityFactor[['Date', 'TECHNOLOGIES', 'availabilityFactor']].set_index(["Date", 'TECHNOLOGIES'])
    availabilityFactor.to_csv(InputFolder + 'availabilityFactor_new' + str(year) + '_' + str(Zones) + '.csv',sep=',', decimal='.')

Zones="FR_DE_GB_ES"
year=2016
availabilityFactor = pd.read_csv(InputFolder + 'availabilityFactor' + str(year) + '_' + str(Zones) + '.csv',
                                 sep=',', decimal='.', skiprows=0)
Date = pd.date_range(start='1/1/' + str(year), periods=len(availabilityFactor.TIMESTAMP.unique()), freq='H')
TIMESTAMP_asso = pd.DataFrame({'TIMESTAMP': range(1, len(availabilityFactor.TIMESTAMP.unique()) + 1), 'Date': Date})
availabilityFactor['TIMESTAMP'] = availabilityFactor['TIMESTAMP'].replace(
    range(1, len(availabilityFactor.TIMESTAMP.unique()) + 1), Date)
availabilityFactor = availabilityFactor.rename(columns={'TIMESTAMP': 'Date'})
availabilityFactor = availabilityFactor[['Date', "AREAS",'TECHNOLOGIES', 'availabilityFactor']].set_index(
    ["Date", "AREAS",'TECHNOLOGIES'])
availabilityFactor.to_csv(InputFolder + 'availabilityFactor_new' + str(year) + '_' + str(Zones) + '.csv', sep=',',
                          decimal='.')
#endregions