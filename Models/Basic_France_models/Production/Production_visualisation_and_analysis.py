InputFolder='Models/Basic_France_models/Production/Data/'

#region importation of modules
import numpy as np
import pandas as pd
import seaborn as sns
import csv
import datetime
import copy
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from functions.f_graphicalTools import * #Il faut préciser le chemin où vous avez sauvegardé les données csv
#endregion

#region Disponibilité sur la France période 2013-2016
Zones="FR"
year=2013

MyTech= 'OldNuke'  ### 'Thermal' 'OldNuke' 'HydroRiver' 'HydroReservoir' 'WindOnShore' 'Solar'
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_FR.csv',
                                sep=',',decimal='.',skiprows=0,parse_dates=['Date'])
availabilityFactor.loc[:,"TIMESTAMP"]=pd.to_datetime(availabilityFactor.loc[:,"Date"])
tabl=availabilityFactor[availabilityFactor['TECHNOLOGIES']==MyTech]
tabl_ = tabl.reset_index().assign(TIMESTAMP_=range(1, len(tabl) + 1)).drop(
    columns="TIMESTAMP")

fig=MyPlotly(x_df=tabl_.TIMESTAMP_,y_df=tabl[['availabilityFactor']],fill=False)

fig = go.Figure()
fig=fig.add_trace(go.Scatter(x=tabl_['TIMESTAMP_'],y=tabl['availabilityFactor'],line=dict(color="#000000"),name="original"))
for newyear in range(2013,2016):
    availabilityFactor = pd.read_csv(InputFolder + 'availabilityFactor' + str(newyear) + '_' + str(Zones) + '.csv',
                                     sep=',', decimal='.', skiprows=0,parse_dates=['Date'])
    availabilityFactor.loc[:, "TIMESTAMP"] = pd.to_datetime(availabilityFactor.loc[:, "Date"])
    tabl = availabilityFactor[availabilityFactor['TECHNOLOGIES'] == MyTech]
    tabl_ = tabl.reset_index().assign(TIMESTAMP_=range(1, len(tabl) + 1)).drop(
        columns="TIMESTAMP")
    fig.add_trace(go.Scatter(x=tabl_['TIMESTAMP_'],y=tabl['availabilityFactor'],
                             line=dict(color="#9CA2A8",width=1),
                             name=newyear))
#fig.show()
plotly.offline.plot(fig, filename='file.html')
#endregion

#region la même chose mais avec l'éolien
Zones="FR"
year=2013

MyTech= 'WindOnShore'  ### 'Thermal' 'OldNuke' 'HydroRiver' 'HydroReservoir' 'WindOnShore' 'Solar'
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_FR.csv',
                                sep=',',decimal='.',skiprows=0,parse_dates=['Date'])
availabilityFactor.loc[:,"TIMESTAMP"]=pd.to_datetime(availabilityFactor.loc[:,"Date"])
tabl=availabilityFactor[availabilityFactor['TECHNOLOGIES']==MyTech]
tabl_ = tabl.reset_index().assign(TIMESTAMP_=range(1, len(tabl) + 1)).drop(
    columns="TIMESTAMP")

fig=MyPlotly(x_df=tabl_.TIMESTAMP_,y_df=tabl[['availabilityFactor']],fill=False)

fig = go.Figure()
fig=fig.add_trace(go.Scatter(x=tabl_['TIMESTAMP_'],y=tabl['availabilityFactor'],line=dict(color="#000000"),name="original"))
for newyear in range(2013,2016):
    availabilityFactor = pd.read_csv(InputFolder + 'availabilityFactor' + str(newyear) + '_' + str(Zones) + '.csv',
                                     sep=',', decimal='.', skiprows=0,parse_dates=['Date'])
    availabilityFactor.loc[:, "TIMESTAMP"] = pd.to_datetime(availabilityFactor.loc[:, "Date"])
    tabl = availabilityFactor[availabilityFactor['TECHNOLOGIES'] == MyTech]
    tabl_ = tabl.reset_index().assign(TIMESTAMP_=range(1, len(tabl) + 1)).drop(
        columns="TIMESTAMP")
    fig.add_trace(go.Scatter(x=tabl_['TIMESTAMP_'],y=tabl['availabilityFactor'],
                             line=dict(color="#9CA2A8",width=1),
                             name=newyear))
#fig.show()
plotly.offline.plot(fig, filename='file.html')
#endregion

#region la même chose mais avec le PV
Zones="FR"
year=2013

MyTech= 'Solar'  ### 'Thermal' 'OldNuke' 'HydroRiver' 'HydroReservoir' 'WindOnShore' 'Solar'
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_FR.csv',
                                sep=',',decimal='.',skiprows=0,parse_dates=['Date'])
availabilityFactor.loc[:,"TIMESTAMP"]=pd.to_datetime(availabilityFactor.loc[:,"Date"])
tabl=availabilityFactor[availabilityFactor['TECHNOLOGIES']==MyTech]
tabl_ = tabl.reset_index().assign(TIMESTAMP_=range(1, len(tabl) + 1)).drop(
    columns="TIMESTAMP")

fig=MyPlotly(x_df=tabl_.TIMESTAMP_,y_df=tabl[['availabilityFactor']],fill=False)

fig = go.Figure()
fig=fig.add_trace(go.Scatter(x=tabl_['TIMESTAMP_'],y=tabl['availabilityFactor'],line=dict(color="#000000"),name="original"))
for newyear in range(2013,2016):
    availabilityFactor = pd.read_csv(InputFolder + 'availabilityFactor' + str(newyear) + '_' + str(Zones) + '.csv',
                                     sep=',', decimal='.', skiprows=0,parse_dates=['Date'])
    availabilityFactor.loc[:, "TIMESTAMP"] = pd.to_datetime(availabilityFactor.loc[:, "Date"])
    tabl = availabilityFactor[availabilityFactor['TECHNOLOGIES'] == MyTech]
    tabl_ = tabl.reset_index().assign(TIMESTAMP_=range(1, len(tabl) + 1)).drop(
        columns="TIMESTAMP")
    fig.add_trace(go.Scatter(x=tabl_['TIMESTAMP_'],y=tabl['availabilityFactor'],
                             line=dict(color="#9CA2A8",width=1),
                             name=newyear))
#fig.show()
plotly.offline.plot(fig, filename='file.html')
#endregion

#region more data from eco2Mix 2012-2019
Eco2mix = pd.read_csv(InputFolder+'Eco2Mix_Hourly_National_xts.csv',
                                sep=';',decimal=',',skiprows=0,
                      dtype={'Index':str, 'Consommation':np.float64, 'Prevision.J1':np.float64, 'Prevision.J':np.float64, 'Fioul':np.float64,
       'Charbon':np.float64, 'Gaz':np.float64, 'Nucleaire':np.float64, 'Eolien':np.float64, 'Solaire':np.float64, 'Hydraulique':np.float64,
       'Pompage':np.float64, 'Bioenergies':np.float64, 'Ech.physiques':np.float64, 'Taux.de.Co2':np.float64,
       'Ech.comm.Angleterre':np.float64, 'Ech.comm.Espagne':np.float64, 'Ech.comm.Italie':np.float64,
       'Ech.comm.Suisse':np.float64, 'Ech.comm.AllemagneBelgique':np.float64, 'Fioul..TAC':np.float64,
       'Fioul..Cogen':np.float64, 'Fioul..Autres':np.float64, 'Gaz..TAC':np.float64, 'Gaz..Cogen':np.float64, 'Gaz..CCG':np.float64,
       'Gaz..Autres':np.float64, 'Hydraulique..Fil.de.leau..eclusee':np.float64, 'Hydraulique..Lacs':np.float64,
       'Hydraulique..STEP.turbinage':np.float64, 'Bioenergies..Dechets':np.float64,
       'Bioenergies..Biomasse':np.float64, 'Bioenergies..Biogaz':np.float64})
Eco2mix['Date'] = pd.to_datetime(Eco2mix['Index'], errors='coerce')
Eco2mix.columns
Eco2mix['year'] = pd.DatetimeIndex(Eco2mix['Date']).year
Eco2mixYear=Eco2mix[Eco2mix['year']==2019]
Hydraulique=Eco2mixYear[['Date','Hydraulique..Lacs','Hydraulique..Fil.de.leau..eclusee']].set_index('Date').rename(
    columns={'Hydraulique..Fil.de.leau..eclusee':'HydroRiver','Hydraulique..Lacs':'HydroLake'})
Hydraulique.max()

Hydraulique.HydroLake.sum()/7000
#endregion

#region more nuke availability for France
DispoNukeTotal = pd.read_csv(InputFolder+'DispoNukeTotal2007_2017.csv',
                                sep=';',decimal=',',skiprows=0,
                      dtype={'Dates':str, 'Availability':np.float64})
DispoNukeTotal.loc[:,"TIMESTAMP"]=pd.to_datetime(DispoNukeTotal.loc[:,"Dates"])
DispoNukeTotal=DispoNukeTotal.assign(Year=DispoNukeTotal.loc[:,"TIMESTAMP"].dt.year).drop(columns="Dates")


DispoNukeTotal=DispoNukeTotal.assign(Year=DispoNukeTotal.TIMESTAMP.dt.year)
fig = go.Figure()
for newyear in range(2007,2016):
    DispoNukeTotalYear=DispoNukeTotal.loc[DispoNukeTotal.Year==newyear]
    DispoNukeTotal_=DispoNukeTotalYear.reset_index().assign(TIMESTAMP_=range(1,len(DispoNukeTotalYear)+1)).drop(columns="TIMESTAMP")
    fig.add_trace(go.Scatter(x=DispoNukeTotal_['TIMESTAMP_'],y=DispoNukeTotal_['Availability'],
                             line=dict(color="#9CA2A8",width=1),
                             name=newyear))
#fig.show()
plotly.offline.plot(fig, filename='file.html')

#endregion