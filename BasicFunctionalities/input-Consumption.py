InputFolder='Data/input/'

#region importation of modules
import numpy as np
import seaborn as sns
import pandas as pd
import csv
import datetime
import copy
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn import linear_model
from functions.f_consumptionModels import * #Il faut préciser le chemin où vous avez sauvegardé les données csv
from functions.f_graphicalTools import * #Il faut préciser le chemin où vous avez sauvegardé les données csv
#endregion

#region  Load and visualize consumption
ConsoTempe_df=pd.read_csv(InputFolder+'ConsumptionTemperature_1996TO2019_FR.csv')
ConsoTempe_df["TIMESTAMP"]=pd.to_datetime(ConsoTempe_df['Date'])
ConsoTempe_df=ConsoTempe_df.drop(columns=["Date"]).set_index(["TIMESTAMP"])


year = 2012
ConsoTempeYear_df=ConsoTempe_df[str(year)]
hour = 19
TemperatureThreshold = 15


ConsoTempeYear_df=ConsoTempe_df[str(year)]
plt.plot(ConsoTempeYear_df['Temperature'],ConsoTempeYear_df['Consumption']/1000, '.', color='black');
plt.show()
#endregion

#region  Thermal sensitivity estimation, consumption decomposition and visualisation
#select dates to do the linear regression
#ConsoTempeYear_df.index.get_level_values("TIMESTAMP").to_series().dt.hour
indexHeatingHour = (ConsoTempeYear_df['Temperature'] <= TemperatureThreshold) &\
                    (ConsoTempeYear_df.index.to_series().dt.hour == hour)
ConsoHeatingHour= ConsoTempeYear_df[indexHeatingHour]
lr=linear_model.LinearRegression().fit(ConsoHeatingHour[['Temperature']],
                                       ConsoHeatingHour['Consumption'])
lr.coef_[0]

#Generic function Thermal sensitivity estimation
(ConsoTempeYear_decomposed_df,Thermosensibilite)=Decomposeconso(ConsoTempeYear_df,TemperatureThreshold=TemperatureThreshold)
#ConsoTempeYear_decomposed_df=ConsoTempeYear_decomposed_df.rename(columns={'NTS_C':'Conso non thermosensible', "TS_C": 'conso thermosensible'})
fig=MyStackedPlotly(y_df=ConsoTempeYear_decomposed_df[["NTS_C","TS_C"]],
                    Names=['Conso non thermosensible','conso thermosensible'])
fig=fig.update_layout(title_text="Consommation (MWh)", xaxis_title="Date")
plotly.offline.plot(fig, filename='file.html') ## offline
#fig.show()
#endregion

#region Thermal sensitivity model to change meteo
## change meteo year
## example for year 2012
newyear=2012
NewConsoTempeYear_df = ConsoTempe_df[str(newyear)]
(ConsoTempeYear_decomposed_df,Thermosensibilite)=Decomposeconso(NewConsoTempeYear_df,TemperatureThreshold=TemperatureThreshold)

NewConsoTempeYear_decomposed_df=Recompose(ConsoTempeYear_decomposed_df,Thermosensibilite,
                                          Newdata_df=NewConsoTempeYear_df,
                                          TemperatureThreshold=TemperatureThreshold)
### loop over years
fig = go.Figure()
TMP=ConsoTempeYear_decomposed_df.copy()
TMP = TMP.reset_index().drop(columns="TIMESTAMP").assign(TIMESTAMP=range(1, len(TMP) + 1)).set_index(["TIMESTAMP"])
fig = fig.add_trace(
    go.Scatter(x=TMP.index,y=ConsoTempeYear_decomposed_df['Consumption'],line=dict(color="#000000"),name="original"))
for newyear in range(2000,2012):
    NewConsoTempeYear_df = ConsoTempe_df[str(newyear)]
    ConsoSepareeNew_df=Recompose(ConsoTempeYear_decomposed_df,Thermosensibilite,
                                 Newdata_df=NewConsoTempeYear_df,
                                 TemperatureThreshold=TemperatureThreshold)
    ConsoSepareeNew_df = ConsoSepareeNew_df.reset_index().drop(columns="TIMESTAMP").assign(
        TIMESTAMP=range(1, len(ConsoSepareeNew_df) + 1)).set_index(["TIMESTAMP"])

    fig.add_trace(go.Scatter(x=ConsoSepareeNew_df.index,
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
fig.add_trace(go.Scatter(x=ConsoTempeYear_decomposed_df.index,
                         y=ConsoTempeYear_decomposed_df['Consumption'],
                         line=dict(color="#000000"),name="original"))
fig.add_trace(go.Scatter(x=NewConsoTempeYear_decomposed_df.index,
                             y=NewConsoTempeYear_decomposed_df['Consumption'],
                             line=dict(color="#9CA2A8",width=1),
                             name=newyear))
plotly.offline.plot(fig, filename='file.html') ## offline
#fig.show()
#endregion

#region Electric Vehicle

VEProfile_df=pd.read_csv(InputFolder+'EVModel.csv', sep=';')
year=2012
NewConsoTempeYear_df = ConsoTempe_df[str(year)]
EV_Consumption_df=Profile2Consumption(Profile_df=VEProfile_df,Temperature_df = NewConsoTempeYear_df[['Temperature']])
fig=MyStackedPlotly(y_df=EV_Consumption_df[["NTS_C","TS_C"]],
                    Names=['Conso VE non thermosensible','conso VE thermosensible'])
fig=fig.update_layout(title_text="Consommation (MWh)", xaxis_title="Date")
plotly.offline.plot(fig, filename='file.html') ## offline
#fig.show()
#endregion

#region multi zone
Zones="FR_DE_GB_ES"
year=2016 #only possible year

#### reading CSV files
areaConsumption = pd.read_csv(InputFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0).set_index(["AREAS","TIMESTAMP"])
fig = go.Figure()
pal = sns.color_palette("bright", 4); i=0; #https://chrisalbon.com/python/data_visualization/seaborn_color_palettes/
for region in ["FR","DE","ES"]: ### problem with spain data
    #tabl=areaConsumption[(region,slice(None))]
    fig.add_trace(go.Scatter(x=areaConsumption.index.get_level_values("TIMESTAMP"),
                             y=areaConsumption.loc[(region,slice(None)),"areaConsumption"],
                             line=dict(color=pal.as_hex()[i],width=1),
                             name=region))
    i=i+1;
#fig.show()
plotly.offline.plot(fig, filename='file.html')
#endregion

#region consumption decomposition
ConsoTempe_df=pd.read_csv(InputFolder+'ConsumptionTemperature_1996TO2019_FR.csv')
ConsoTempe_df["TIMESTAMP"]=pd.to_datetime(ConsoTempe_df['Date'])
ConsoTempe_df=ConsoTempe_df.drop(columns=["Date"]).set_index(["TIMESTAMP"])
year = 2012
ConsoTempeYear_df=ConsoTempe_df[str(year)]
Profile_df=pd.read_csv(InputFolder+"ConsumptionDetailedProfiles.csv").set_index(["Mois", "heures",'Nature', 'type', 'UsagesGroupe', 'UsageDetail', "WeekDay"])
Profile_df_merged=ComplexProfile2Consumption(Profile_df,ConsoTempeYear_df)
Profile_df_merged_spread = Profile_df_merged.groupby(["TIMESTAMP", "type"]).sum().reset_index(). \
    drop(columns=["Temperature"]). \
    pivot(index="TIMESTAMP", columns='type', values='Conso');
Profile_df_merged_spread
fig = MyStackedPlotly(y_df=Profile_df_merged_spread)
plotly.offline.plot(fig, filename='file.html')  ## offline
#endregion


####
#Day="Samedi"
#df=pd.read_csv(InputFolder+Day+'.csv', sep=';', encoding='cp437',decimal=",")
#Profile_df_Sat=CleanProfile(df,Nature_PROFILE,type_PROFILE,Usages_PROFILE,UsagesGroupe_PROFILE)
#Day="Dimanche"
#df=pd.read_csv(InputFolder+Day+'.csv', sep=';', encoding='cp437',decimal=",")
#Profile_df_Sun=CleanProfile(df,Nature_PROFILE,type_PROFILE,Usages_PROFILE,UsagesGroupe_PROFILE)
#Day="Semaine"
#df=pd.read_csv(InputFolder+Day+'.csv', sep=';', encoding='cp437',decimal=",")
#Profile_df_Week=CleanProfile(df,Nature_PROFILE,type_PROFILE,Usages_PROFILE,UsagesGroupe_PROFILE)
#Profile_df_Week["WeekDay"] = "Week"; Profile_df_Sat["WeekDay"] = "Sat"; Profile_df_Sun["WeekDay"] = "Sun"
#Profile_df = Profile_df_Week.append(Profile_df_Sat).append(Profile_df_Sun).set_index(['WeekDay'], append=True)
#Profile_df.to_csv(InputFolder+"ConsumptionDetailedProfiles.csv")
