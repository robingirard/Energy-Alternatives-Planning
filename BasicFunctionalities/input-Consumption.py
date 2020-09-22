InputFolder='Data/input/'

#region importation of modules
import numpy as np
import pandas as pd
import csv
import datetime
import copy
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn import linear_model

#os.chdir('D:\GIT\Etude_TP_CapaExpPlaning-Python')
from functions.f_consumptionModels import * #Il faut préciser le chemin où vous avez sauvegardé les données csv
from functions.f_graphicalTools import * #Il faut préciser le chemin où vous avez sauvegardé les données csv
#data=pd.read_csv('CSV/input/ConsumptionTemperature_1996TO2019_FR.csv')
#endregion

#region  Load and visualize consumption
ConsoTempe_df=pd.read_csv(InputFolder+'ConsumptionTemperature_1996TO2019_FR.csv')

year = 2012
hour = 19
TemperatureThreshold = 15

ConsoTempeYear_df=ConsoTempe_df[pd.to_datetime(ConsoTempe_df['Date']).dt.year==year]
plt.plot(ConsoTempeYear_df['Temperature'],ConsoTempeYear_df['Consumption']/1000, '.', color='black');
plt.show()
#endregion

#region  Thermal sensitivity estimation, consumption decomposition and visualisation
#select dates to do the linear regression
indexHeatingHour = (ConsoTempeYear_df['Temperature'] <= TemperatureThreshold) &\
                    (pd.to_datetime(ConsoTempeYear_df['Date']).dt.hour == hour)
ConsoHeatingHour= ConsoTempeYear_df[indexHeatingHour]
lr=linear_model.LinearRegression().fit(ConsoHeatingHour[['Temperature']],
                                       ConsoHeatingHour['Consumption'])
lr.coef_[0]

#Generic function Thermal sensitivity estimation
(ConsoTempeYear_decomposed_df,Thermosensibilite)=Decomposeconso(ConsoTempeYear_df,TemperatureThreshold=TemperatureThreshold)
fig=MyStackedPlotly(x_df=ConsoTempeYear_decomposed_df['Date'],
                    y_df=ConsoTempeYear_decomposed_df[["NTS_C","TS_C"]],
                    Names=['Conso non thermosensible','conso thermosensible'])
fig.update_layout(title_text="Consommation (MWh)", xaxis_title="Date")
 plotly.offline.plot(fig, filename='file.html') ## offline
#fig.show()
#endregion

#region Thermal sensitivity model to change meteo
## change meteo year
## example for year 2012
newyear=2012
NewConsoTempeYear_df = ConsoTempe_df[pd.to_datetime(ConsoTempe_df['Date']).dt.year==newyear]
NewConsoTempeYear_decomposed_df=Recompose(ConsoTempeYear_decomposed_df,Thermosensibilite,
                                          Newdata_df=NewConsoTempeYear_df,
                                          TemperatureThreshold=TemperatureThreshold)
### loop over years
fig = go.Figure()
fig.add_trace(go.Scatter(x=ConsoTempeYear_decomposed_df['Date'],y=ConsoTempeYear_decomposed_df['Consumption'],line=dict(color="#000000"),name="original"))
for newyear in range(2000,2012):
    NewConsoTempeYear_df = ConsoTempe_df[pd.to_datetime(ConsoTempe_df['Date']).dt.year==newyear]
    ConsoSepareeNew_df=Recompose(ConsoTempeYear_decomposed_df,Thermosensibilite,
                                 Newdata_df=NewConsoTempeYear_df,
                                 TemperatureThreshold=TemperatureThreshold)
    fig.add_trace(go.Scatter(x=ConsoSepareeNew_df['Date'],
                             y=ConsoSepareeNew_df['Consumption'],
                             line=dict(color="#9CA2A8",width=1),
                             name=newyear))
plotly.offline.plot(fig, filename='file.html') ## offline
#fig.show()
#endregion

#region Thermal sensitivity model to change thermal sensitivity
## change thermal sensitivity
NewThermosensibilite={}
for key in Thermosensibilite:    NewThermosensibilite[key]=1/3 * Thermosensibilite[key]
NewConsoTempeYear_decomposed_df=Recompose(ConsoTempeYear_decomposed_df,NewThermosensibilite,
                                          TemperatureThreshold=TemperatureThreshold)
fig = go.Figure()
fig.add_trace(go.Scatter(x=ConsoTempeYear_decomposed_df['Date'],
                         y=ConsoTempeYear_decomposed_df['Consumption'],
                         line=dict(color="#000000"),name="original"))
fig.add_trace(go.Scatter(x=NewConsoTempeYear_decomposed_df['Date'],
                             y=NewConsoTempeYear_decomposed_df['Consumption'],
                             line=dict(color="#9CA2A8",width=1),
                             name=newyear))
plotly.offline.plot(fig, filename='file.html') ## offline
#fig.show()
#endregion

#region Electric Vehicle

VEProfile_df=pd.read_csv(InputFolder+'EVModel.csv', sep=';')
year=2012
NewConsoTempeYear_df = ConsoTempe_df[pd.to_datetime(ConsoTempe_df['Date']).dt.year==year]
EV_Consumption_df=Profile2Consumption(Profile_df=VEProfile_df,Temperature_df = NewConsoTempeYear_df[['Date', 'Temperature']])
fig=MyStackedPlotly(x_df=EV_Consumption_df['Date'],
                    y_df=EV_Consumption_df[["NTS_C","TS_C"]],
                    Names=['Conso VE non thermosensible','conso VE thermosensible'])
fig.update_layout(title_text="Consommation (MWh)", xaxis_title="Date")
plotly.offline.plot(fig, filename='file.html') ## offline
#fig.show()
#endregion