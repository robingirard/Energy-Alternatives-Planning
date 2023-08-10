InputFolder='Models/Basic_France_models/Consumption/Data/'
import sys
sys.path.extend(['.'])
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
from EnergyAlternativesPlanning.f_consumptionModels import * #Il faut préciser le chemin où vous avez sauvegardé les données csv
from EnergyAlternativesPlanning.f_graphicalTools import * #Il faut préciser le chemin où vous avez sauvegardé les données csv
from EnergyAlternativesPlanning.f_heat_pump import *
#endregion

#region  Load and visualize consumption
ConsoTempe_df=pd.read_csv(InputFolder+'ConsumptionTemperature_1996TO2019_FR.csv',parse_dates=['Date']).set_index(["Date"])
year = 2012
ConsoTempeYear_df=ConsoTempe_df.loc[str(year)]
hour = 19
TemperatureThreshold = 15
plt.plot(ConsoTempeYear_df['Temperature'],ConsoTempeYear_df['Consumption']/1000, '.', color='black');
plt.show()
#endregion

#region  Thermal sensitivity estimation, consumption decomposition and visualisation
#select dates to do the linear regression
#ConsoTempeYear_df.index.get_level_values("Date").to_series().dt.hour
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
#part thermosensible :
ConsoTempeYear_decomposed_df.TS_C.sum()/ConsoTempeYear_df.sum()
#endregion

#region Thermal sensitivity model to change meteo
## change meteo year
## example for year 2012
newyear=2012
NewConsoTempeYear_df = ConsoTempe_df.loc[str(newyear)]
(ConsoTempeYear_decomposed_df,Thermosensibilite)=Decomposeconso(NewConsoTempeYear_df,TemperatureThreshold=TemperatureThreshold)

NewConsoTempeYear_decomposed_df=Recompose(ConsoTempeYear_decomposed_df,Thermosensibilite,
                                          Newdata_df=NewConsoTempeYear_df,
                                          TemperatureThreshold=TemperatureThreshold)
### loop over years
fig = go.Figure()
TMP=ConsoTempeYear_decomposed_df.copy()
TMP = TMP.reset_index().drop(columns="Date").assign(Date=range(1, len(TMP) + 1)).set_index(["Date"])
fig = fig.add_trace(
    go.Scatter(x=TMP.index,y=ConsoTempeYear_decomposed_df['Consumption'],line=dict(color="#000000"),name="original"))
for newyear in range(2000,2012):
    NewConsoTempeYear_df = ConsoTempe_df.loc[str(newyear)]
    ConsoSepareeNew_df=Recompose(ConsoTempeYear_decomposed_df,Thermosensibilite,
                                 Newdata_df=NewConsoTempeYear_df,
                                 TemperatureThreshold=TemperatureThreshold)
    ConsoSepareeNew_df = ConsoSepareeNew_df.reset_index().drop(columns="Date").assign(
        Date=range(1, len(ConsoSepareeNew_df) + 1)).set_index(["Date"])

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

VEProfile_df=pd.read_csv(InputFolder+'EVModel.csv', sep=';')#.set_index(["Date"])
year=2012
EV_Consumption_df=Profile2Consumption(Profile_df=VEProfile_df,Temperature_df = ConsoTempe_df.loc[str(year)][['Temperature']])
fig=MyStackedPlotly(y_df=EV_Consumption_df[["NTS_C","TS_C"]],
                    Names=['Conso VE non thermosensible','conso VE thermosensible'])
fig=fig.update_layout(title_text="Consommation (MWh)", xaxis_title="Date")
plotly.offline.plot(fig, filename='file.html') ## offline

#même chose avec l'ECS
ECS_Profile_df=pd.read_csv(InputFolder+'Profil_ECS_RTE.csv',sep=';',decimal=',',encoding='utf-8').\
    melt(id_vars=["Jour","Heure"],value_vars=["ECS en juin","ECS en janvier"],value_name="Puissance.MW",var_name="Saison").\
    replace({"Saison":{"ECS en juin":"Ete","ECS en janvier":"Hiver"},
             "Jour":{"Lundi":1,"Mardi":2,"Mercredi":3,"Jeudi":4,"Vendredi":5,"Samedi":6,"Dimanche":7}})
ECS_Consumption_df=Profile2Consumption(Profile_df=ECS_Profile_df,Temperature_df = ConsoTempe_df.loc[str(year)][['Temperature']],VarName="Puissance.MW")
fig=MyStackedPlotly(y_df=ECS_Consumption_df[["NTS_C","TS_C"]],
                    Names=['Conso ECS non thermosensible','conso ECS thermosensible'])
fig=fig.update_layout(title_text="Consommation (MWh)", xaxis_title="Date")
plotly.offline.plot(fig, filename='file.html') ## offline


#fig.show()
#endregion

#region consumption decomposition [Tertiaire,résidentiel,Indus,Autre]x[ChauffagexAutre]
year = 2012
TemperatureThreshold = 15
ConsoTempe_df=pd.read_csv(InputFolder+'ConsumptionTemperature_1996TO2019_FR.csv',parse_dates=['Date']).\
    set_index(["Date"])[str(year)]
ConsoTempe_df=ConsoTempe_df[~ConsoTempe_df.index.duplicated(keep='first')]
(ConsoTempeYear_decomposed_df,Thermosensibilite)=Decomposeconso(ConsoTempe_df,TemperatureThreshold=TemperatureThreshold)

Profile_df_sans_chauffage=pd.read_csv(InputFolder+"ConsumptionDetailedProfiles.csv").\
    rename(columns={'heures':'Heure',"WeekDay":"Jour"}).\
    replace({"Jour" :{"Sat": "Samedi" , "Week":"Semaine"  , "Sun": "Dimanche"}}). \
    query('UsagesGroupe != "Chauffage"'). \
    set_index(["Mois", "Heure",'Nature', 'type', 'UsagesGroupe', 'UsageDetail', "Jour"]).\
    groupby(["Mois","Jour","Heure","type"]).sum().\
    merge(add_day_month_hour(df=ConsoTempe_df,semaine_simplifie=True,French=True,to_index=True),
          how="outer",left_index=True,right_index=True).reset_index().set_index("Date")[["type","Conso"]]. \
    pivot_table(index="Date", columns=["type"], values='Conso')

Profile_df_sans_chauffage=Profile_df_sans_chauffage.loc[:,Profile_df_sans_chauffage.sum(axis=0)>0]
Profile_df_n=Profile_df_sans_chauffage.div(Profile_df_sans_chauffage.sum(axis=1), axis=0) ### normalisation par 1 et multiplication
for col in Profile_df_sans_chauffage.columns:
    Profile_df_sans_chauffage[col]=Profile_df_n[col]*ConsoTempeYear_decomposed_df["NTS_C"]

Profile_df_seulement_chauffage=pd.read_csv(InputFolder+"ConsumptionDetailedProfiles.csv").\
    rename(columns={'heures':'Heure',"WeekDay":"Jour"}).\
    replace({"Jour" :{"Sat": "Samedi" , "Week":"Semaine"  , "Sun": "Dimanche"}}). \
    query('UsagesGroupe == "Chauffage"'). \
    set_index(["Mois", "Heure",'Nature', 'type', 'UsagesGroupe', 'UsageDetail', "Jour"]).\
    groupby(["Mois","Jour","Heure","type"]).sum().\
    merge(add_day_month_hour(df=ConsoTempe_df,semaine_simplifie=True,French=True,to_index=True),
          how="outer",left_index=True,right_index=True).reset_index().set_index("Date")[["type","Conso"]]. \
    pivot_table(index="Date", columns=["type"], values='Conso')

Profile_df_seulement_chauffage=Profile_df_seulement_chauffage.loc[:,Profile_df_seulement_chauffage.sum(axis=0)>0]
Profile_df_n=Profile_df_seulement_chauffage.div(Profile_df_seulement_chauffage.sum(axis=1), axis=0) ### normalisation par 1 et multiplication
for col in Profile_df_seulement_chauffage.columns:
    Profile_df_seulement_chauffage[col]=Profile_df_n[col]*ConsoTempeYear_decomposed_df["TS_C"]

Profile_df_seulement_chauffage.columns=[(col,"chauffage") for col in Profile_df_seulement_chauffage.columns]
Profile_df_sans_chauffage.columns=[(col,"autre") for col in Profile_df_sans_chauffage.columns]
Profile_df=pd.concat([Profile_df_seulement_chauffage,Profile_df_sans_chauffage],axis=1)
Profile_df.columns=pd.MultiIndex.from_tuples(Profile_df.columns, names=('type', 'usage'))
fig = MyStackedPlotly(y_df=Profile_df)
plotly.offline.plot(fig, filename='file.html')  ## offline

Profile_df.sum(axis=0)/10**6
#endregion

#region consumption decomposition [Tertiaire,résidentiel,Indus,Autre]x[ChauffagexECSxCuissonx...]
year = 2012
ConsoTempe_df=pd.read_csv(InputFolder+'ConsumptionTemperature_1996TO2019_FR.csv',parse_dates=['Date']).\
    set_index(["Date"])[str(year)]
ConsoTempe_df=ConsoTempe_df[~ConsoTempe_df.index.duplicated(keep='first')]
(ConsoTempeYear_decomposed_df,Thermosensibilite)=Decomposeconso(ConsoTempe_df,TemperatureThreshold=TemperatureThreshold)

UsagesGroupe_simplified_dict={'Autres': "Specifique et autre",  'Congelateur':"Specifique et autre",
       'Eclairage': "Specifique et autre" ,
       'LaveLinge':"Specifique et autre", 'Lavevaisselle':"Specifique et autre",
       'Ordis': "Specifique et autre",
       'Refrigirateur' :"Specifique et autre" , 'SecheLinge' :"Specifique et autre",
        'TVAutreElectroMen' : "Specifique et autre"}

Profile_df_sans_chauffage=pd.read_csv(InputFolder+"ConsumptionDetailedProfiles.csv").\
    rename(columns={'heures':'Heure',"WeekDay":"Jour"}).\
    replace({"Jour" :{"Sat": "Samedi" , "Week":"Semaine"  , "Sun": "Dimanche"},
             'UsagesGroupe':UsagesGroupe_simplified_dict}). \
    query('UsagesGroupe != "Chauffage"'). \
    set_index(["Mois", "Heure",'Nature', 'type','UsagesGroupe', 'UsageDetail', "Jour"]).\
    groupby(["Mois","Jour","Heure",'type','UsagesGroupe']).sum().\
    merge(add_day_month_hour(df=ConsoTempe_df,semaine_simplifie=True,French=True,to_index=True),
          how="outer",left_index=True,right_index=True).reset_index().set_index("Date")[['type','UsagesGroupe',"Conso"]]. \
    pivot_table(index="Date", columns=['type','UsagesGroupe'], values='Conso')

Profile_df_sans_chauffage=Profile_df_sans_chauffage.loc[:,Profile_df_sans_chauffage.sum(axis=0)>0]
Profile_df_n=Profile_df_sans_chauffage.div(Profile_df_sans_chauffage.sum(axis=1), axis=0) ### normalisation par 1 et multiplication
for col in Profile_df_sans_chauffage.columns:
    Profile_df_sans_chauffage[col]=Profile_df_n[col]*ConsoTempeYear_decomposed_df["NTS_C"]


Profile_df_seulement_chauffage=pd.read_csv(InputFolder+"ConsumptionDetailedProfiles.csv").\
    rename(columns={'heures':'Heure',"WeekDay":"Jour"}).\
    replace({"Jour" :{"Sat": "Samedi" , "Week":"Semaine"  , "Sun": "Dimanche"},
             'UsagesGroupe':UsagesGroupe_simplified_dict}). \
    query('UsagesGroupe == "Chauffage"'). \
    set_index(["Mois", "Heure",'Nature', 'type','UsagesGroupe', 'UsageDetail', "Jour"]).\
    groupby(["Mois","Jour","Heure",'type','UsagesGroupe']).sum().\
    merge(add_day_month_hour(df=ConsoTempe_df,semaine_simplifie=True,French=True,to_index=True),
          how="outer",left_index=True,right_index=True).reset_index().set_index("Date")[['type','UsagesGroupe',"Conso"]]. \
    pivot_table(index="Date", columns=['type','UsagesGroupe'], values='Conso')

Profile_df_seulement_chauffage=Profile_df_seulement_chauffage.loc[:,Profile_df_seulement_chauffage.sum(axis=0)>0]
Profile_df_n=Profile_df_seulement_chauffage.div(Profile_df_seulement_chauffage.sum(axis=1), axis=0) ### normalisation par 1 et multiplication
for col in Profile_df_seulement_chauffage.columns:
    Profile_df_seulement_chauffage[col]=Profile_df_n[col]*ConsoTempeYear_decomposed_df["TS_C"]

Profile_df=pd.concat([Profile_df_seulement_chauffage,Profile_df_sans_chauffage],axis=1)
fig = MyStackedPlotly(y_df=Profile_df)
plotly.offline.plot(fig, filename='file.html')  ## offline



#endregion

#region brouillon decomposition avec les profils de Pierrick

#ECS_profil= pd.read_csv(InputFolder+"Conso_model/Profil_ECS.csv")
ConsoTempe_df=pd.read_csv(InputFolder+'ConsumptionTemperature_1996TO2019_FR.csv',
                          parse_dates=['Date']).\
    set_index(["Date"])
TemperatureThreshold=15
year = 2012
(ConsoTempeYear_decomposed_df,Thermosensibilite)=Decomposeconso(ConsoTempe_df[str(year)],TemperatureThreshold=TemperatureThreshold)
ConsoTempeYear_decomposed_df.loc[:,"NTS_C"]# partie non thermosensible
ConsoTempeYear_decomposed_df.loc[:,"TS_C"] # partie thermosensible

Conso_non_thermosensible = ConsoTempeYear_decomposed_df[["NTS_C"]].rename(columns= {"NTS_C":"Consumption"})
NTS_profil=  pd.read_csv(InputFolder+"Profil_NTS_RTE.csv",sep=";", decimal=",").set_index(['Saison','Jour','Heure'])
Profil={}
for Saison in ["Hiver","Ete"]:
    Profil[Saison]=add_day_month_hour(Conso_non_thermosensible).reset_index().set_index(['Heure','Jour',"Mois"]).\
        merge(NTS_profil.loc[(Saison,slice(None),slice(None))].reset_index().set_index(['Heure','Jour']).drop(columns="Saison"),
              how='outer',right_index = True,left_index=True).reset_index().set_index(["Date"]).\
        drop(columns=["Heure","Jour" ,"Mois","Consumption"])

Periodi_signal = np.concatenate((np.linspace(0,1,int(len(Conso_non_thermosensible.index)/2)),np.linspace(1,0,int(len(Conso_non_thermosensible.index)/2))), axis=None)
Alpha=pd.DataFrame(Periodi_signal,index= Conso_non_thermosensible.index)
Conso_NTS_par_usage=pd.DataFrame(None)
for col in Profil["Hiver"].columns:
    Conso_NTS_par_usage[col] = (Profil["Hiver"][col]*(1-Alpha[0])+Profil["Ete"][col]*Alpha[0])*Conso_non_thermosensible["Consumption"]

fig = MyStackedPlotly(y_df=Conso_NTS_par_usage)
plotly.offline.plot(fig, filename='file.html')  ## offline
#endregion

#region heat pump
ConsoTempe_df=pd.read_csv(InputFolder+'ConsumptionTemperature_1996TO2019_FR.csv',parse_dates=['Date']).set_index(["Date"]).\
                    rename(columns={'Temperature' :'temp'})[["temp"]]
year=2018
Simulation_PAC_input_parameter = {
    "System": "A/A HP", "Technology": "Inverter", "Mode": "Bivalent", "Emitters": "Fan coil unit",
    "N_stages": np.nan, "Power_ratio": 3.0, "PLF_biv": 1.4, "Ce": 0.7, "Lifetime": 17, "Temperature_limit": -10,
    "Share_Power": 0.5, "regulation": "Y", "T_start" : 15, "T_target" : 18   }

SCOP=estim_SCOP(ConsoTempe_df, Simulation_PAC_input_parameter,year=year)
MyConsoData =SCOP["meteo_data_heating_period"][["P_calo",'P_app',"P_elec"]]
index_year =ConsoTempe_df.loc[str(year)].index
MyConsoData_filled=pd.DataFrame([0]*len(index_year),index=index_year).\
    merge(MyConsoData,how="outer",left_index=True,right_index=True).drop(columns=0).fillna(0)
fig=MyStackedPlotly(y_df=MyConsoData_filled[["P_calo",'P_app']])
fig.add_trace(go.Scatter(x=MyConsoData_filled.index,
                         y=MyConsoData_filled["P_elec"], name="Puissance PAC",
                         line=dict(color='red', width=0.4)))
fig=fig.update_layout(title_text="Conso (en Delta°C)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline

#endregion