#region imports

from EnergyAlternativesPlaning.f_heat_pump import *
from EnergyAlternativesPlaning.f_graphicalTools import *
Data_Folder = "Models/HeatPump_models/data/"
GraphicalResultsFolder = "Models/HeatPump_models/graphics/"
#endregion


Zone = "AL01"
temp_ts=pd.read_csv(Data_Folder+"era-nuts-t2m-nuts2-hourly-singleindex.csv",parse_dates=['time']).set_index(["time"])
temp_ts_Zone=temp_ts.loc[temp_ts.index.year>2000,[Zone]].rename(columns={Zone : "temp"})
temp_ts_Zone.head()

##ON DUPLIQUE POUR AVEC ET SANS REGULATION DE LA TEMPERATURE DE L'EAU
Systems = pd.read_csv(Data_Folder+"chauffage.csv")
Systems_reg=Systems.copy(); Systems_reg["regulation"]="Y"
Systems_noreg=Systems.copy(); Systems_noreg["regulation"]="N"
Systems= pd.concat([Systems_reg , Systems_noreg],axis=0)

population=pd.read_csv(Data_Folder+"Population_NUTS2.csv").set_index(["unit"]).loc[Zone,:]
# Base temperature is considered as the temperature which occured 5 days per year in the last years (in mean)

Systems.System.unique()
Systems.Technology.unique()
System=Systems.iloc[10,:]
Heating_params = {  "T_start" : 15, "T_target" : 18   }



Simulation_PAC_input_parameter = {**Heating_params, **System.to_dict()}
Simulation_PAC_input_parameter.keys()

Power_Ratio={}
Energy_Ratio={}
Consumption_Ratio={}
T_biv={}
for Value in np.linspace(0.1,0.6,100):
    Simulation_PAC_input_parameter['Share_Power']=Value
    SCOP=estim_SCOP(meteo_data=temp_ts_Zone,
                    Simulation_PAC_input_parameter=Simulation_PAC_input_parameter,year=2018)

    MyConsoData =SCOP["meteo_data_heating_period"][["temp","P_calo",'P_app']]
    Maximums = MyConsoData.max()
    Means =  MyConsoData.mean()
    mean_P_elec = SCOP["meteo_data_heating_period"]["P_elec"].mean()
    ## Sur que c'est ce qu'on veut faire ? Pas le Min de P_calo plutot ?
    Power_Ratio[Value] = Maximums["P_calo"] / (Maximums["P_calo"]+Maximums["P_app"])
    Energy_Ratio[Value] = Means["P_calo"] / (Means["P_calo"]+Means["P_app"])
    Consumption_Ratio[Value] = mean_P_elec / (mean_P_elec+Means["P_app"])
    T_biv[Value] =SCOP["T_biv"]


MyData = pd.DataFrame.from_dict({"T_biv" : T_biv.values(),
                                 "Power_Ratio" : Power_Ratio.values(),
                                 "Energy_Ratio" : Energy_Ratio.values(),
                                 "Consumption_Ratio" : Consumption_Ratio.values()}).set_index("T_biv")

MyData.plot(y=['Power_Ratio',"Energy_Ratio","Consumption_Ratio"], use_index=True)
import plotly.express as px

fig = px.line(MyData*100,
              labels={
                  "T_biv": "Température de bivalence [°C]",
                  "value": "Contribution de la PAC [%]",
                  "Power_Ratio" : "a la puissance",
              },
              x=MyData.index, y=['Power_Ratio',"Energy_Ratio","Consumption_Ratio"])
plotly.offline.plot(fig, filename=GraphicalResultsFolder+'file.html') ## offline

fig=MyStackedPlotly(y_df=MyConsoData)
fig=fig.update_layout(title_text="Conso (en Delta°C)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename=GraphicalResultsFolder+'file.html') ## offline

SCOP["COP_data"].mean()
