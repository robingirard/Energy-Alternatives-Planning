
#region initialisation
import os
import sys
sys.path.extend(['.'])
import highspy # if using highs solver
import linopy
import pandas as pd
import requests
pd.options.display.width = 0
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from Models.Linopy.f_graphicalTools import *
from Models.Linopy.f_consumptionModels import *
from Models.Linopy.f_tools import *
from Models.Linopy.f_planningModels_linopy import Build_EAP_Model,run_highs
#endregion

#region Download data
graphical_results_folder="Models/Basic_France_models/Planning_optimisation/GraphicalResults/"
InputExcelFolder="Models/Linopy/"
from urllib.request import urlretrieve

xls_7_nodes_file = InputExcelFolder+"EU_7_2050.xlsx"
if not os.path.isfile(xls_7_nodes_file):
    response = requests.get("https://cloud.minesparis.psl.eu/index.php/s/cyYnD3nV2BJgYeg")
    with open(xls_7_nodes_file, mode="wb") as file:
        file.write(response.content)
    print(f"Downloaded EU_7_2050.xlsx and saved to {xls_7_nodes_file}\n Do not sync excel file with git.")
#endregion

#region I - Simple single area (with ramp) : loading parameters
selected_area_to=["FR"]
selected_conversion_technology=['old_nuke', 'ccgt',"demand_not_served"] #you'll add 'solar' after
#selected_conversion_technology=['old_nuke','wind_power_on_shore', 'ccgt',"demand_not_served",'hydro_river', 'hydro_reservoir',"solar"] ## try adding 'hydro_river', 'hydro_reservoir'

xls_file=pd.ExcelFile(InputExcelFolder+"EU_7_2050.xlsx")
#TODO create an excel file with only two country to accelerate the code here
conversion_technology_parameters = pd.read_excel(xls_file, "conversion_technology").dropna().\
    set_index(["area_to", "conversion_technology","energy_vector_out"]).to_xarray()
storage_parameters = pd.read_excel(xls_file, "storage_technology").set_index(["energy_vector_out","area_to", "storage_technology"]).to_xarray()
exogeneous_electricity_demand = pd.read_excel(xls_file, "electricity_demand",parse_dates=['date']).dropna().\
    set_index(["area_to", "date"]).to_xarray().expand_dims(dim={"energy_vector_out": ["electricity"]}, axis=1).transpose("energy_vector_out","area_to", "date")
conversion_technology_parameters["operation_efficiency"].sum(["conversion_technology"])
exogeneous_energy_demand = exogeneous_electricity_demand
operation_conversion_availability_factor = pd.read_excel(xls_file, "operation_conversion_availabili",parse_dates=['date']).\
    dropna().set_index(["area_to", "date", "conversion_technology"]).to_xarray()
energy_vector_in = pd.read_excel(xls_file, "energy_vector_in").dropna().set_index(["area_to", "energy_vector_in"]).to_xarray()

selected_energy_vector_in_value = list(np.unique(conversion_technology_parameters.loc[{"conversion_technology" : selected_conversion_technology,"area_to": selected_area_to}]["energy_vector_in_value"].squeeze().to_numpy()))

parameters= xr.merge([  exogeneous_energy_demand.select({"area_to": selected_area_to}),
                        operation_conversion_availability_factor.select({"conversion_technology" : selected_conversion_technology,
                                                            "area_to": selected_area_to}),
                        conversion_technology_parameters.loc[{"conversion_technology" : selected_conversion_technology,"area_to": selected_area_to}],
                        energy_vector_in.loc[{"energy_vector_in" : selected_energy_vector_in_value,"area_to": selected_area_to}]])

parameters["operation_conversion_availability_factor"]=parameters["operation_conversion_availability_factor"].fillna(1) ## 1 is the default value for availability factor
parameters["operation_efficiency"]=parameters["operation_efficiency"].fillna(0)

parameters["operation_min_1h_ramp_rate"].loc[{"conversion_technology" :"old_nuke"}] = 0.01
parameters["operation_max_1h_ramp_rate"].loc[{"conversion_technology" :"old_nuke"}] = 0.02
parameters["planning_max_capacity"].loc[{"conversion_technology" :"old_nuke"}]=80000
parameters["planning_max_capacity"].loc[{"conversion_technology" :"ccgt"}]=50000
#parameters["exogeneous_energy_demand"]
#parameters=parameters.expand_dims(dim={"energy_vector_out": ["electricity"]}, axis=1)

#endregion

#region I - Simple single area (with ramp) : building and solving problem, results visualisation
#building model and solving the problem
model = Build_EAP_Model(parameters=parameters)
model.solve(solver_name='cbc')
## synthèse Energie/Puissance/Coûts
print(extractCosts_l(model))
print(extractEnergyCapacity_l(model))

### Check sum Prod = Consumption
abs(model.solution['operation_conversion_power_out'].sum(['conversion_technology'])-parameters['exogeneous_energy_demand']).max()

## visualisation de la série
production_df=model.solution['operation_conversion_power_out'].to_dataframe().\
    reset_index().pivot(index="date",columns='conversion_technology', values='operation_conversion_power_out')
fig=MyStackedPlotly(y_df=production_df,Conso = exogeneous_electricity_demand.to_dataframe())
fig=fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename=graphical_results_folder+'file.html') ## offline
#endregion

#region II - addition of Storage to single area with ramp : loading parameters
selected_area_to=["FR"]
selected_conversion_technology=['old_nuke','wind_power_on_shore', 'ccgt',"demand_not_served",'hydro_river', 'hydro_reservoir',"solar"] ## try adding 'hydro_river', 'hydro_reservoir'
selected_storage_technology = ['storage_hydro']
xls_file=pd.ExcelFile(InputExcelFolder+"EU_7_2050.xlsx")
#TODO create an excel file with only two country to accelerate the code here
conversion_technology_parameters = pd.read_excel(xls_file, "conversion_technology").dropna().\
    set_index(["area_to", "conversion_technology","energy_vector_out"]).to_xarray()
storage_parameters = pd.read_excel(xls_file, "storage_technology").set_index(["energy_vector_out","area_to", "storage_technology"]).to_xarray()
exogeneous_electricity_demand = pd.read_excel(xls_file, "electricity_demand",parse_dates=['date']).dropna().\
    set_index(["area_to", "date"]).to_xarray().expand_dims(dim={"energy_vector_out": ["electricity"]}, axis=1).transpose("energy_vector_out","area_to", "date")
conversion_technology_parameters["operation_efficiency"].sum(["conversion_technology"])
exogeneous_energy_demand = exogeneous_electricity_demand
operation_conversion_availability_factor = pd.read_excel(xls_file, "operation_conversion_availabili",parse_dates=['date']).\
    dropna().set_index(["area_to", "date", "conversion_technology"]).to_xarray()
energy_vector_in = pd.read_excel(xls_file, "energy_vector_in").dropna().set_index(["area_to", "energy_vector_in"]).to_xarray()

selected_energy_vector_in_value = list(np.unique(conversion_technology_parameters.loc[{"conversion_technology" : selected_conversion_technology,"area_to": selected_area_to}]["energy_vector_in_value"].squeeze().to_numpy()))

parameters= xr.merge([  storage_parameters.select({"area_to": selected_area_to,
                                                   "storage_technology" : selected_storage_technology}),
                        exogeneous_energy_demand.select({"area_to": selected_area_to}),
                        operation_conversion_availability_factor.select({"conversion_technology" : selected_conversion_technology,
                                                            "area_to": selected_area_to}),
                        conversion_technology_parameters.loc[{"conversion_technology" : selected_conversion_technology,"area_to": selected_area_to}],
                        energy_vector_in.loc[{"energy_vector_in" : selected_energy_vector_in_value,"area_to": selected_area_to}]])

parameters["operation_conversion_availability_factor"]=parameters["operation_conversion_availability_factor"].fillna(1) ## 1 is the default value for availability factor
parameters["operation_efficiency"]=parameters["operation_efficiency"].fillna(0)

parameters["operation_min_1h_ramp_rate"].loc[{"conversion_technology" :"old_nuke"}] = 0.01
parameters["operation_max_1h_ramp_rate"].loc[{"conversion_technology" :"old_nuke"}] = 0.02
parameters["planning_max_capacity"].loc[{"conversion_technology" :"old_nuke"}]=80000
parameters["planning_max_capacity"].loc[{"conversion_technology" :"ccgt"}]=50000

#endregion

#region II -addition of Storage to single area with ramp : building and solving problem, results visualisation
#building model and solving the problem
model = Build_EAP_Model(parameters=parameters)
model.solve(solver_name='cbc')
#res= run_highs(model) #res= linopy.solvers.run_highs(model)

## synthèse Energie/Puissance/Coûts
print(extractCosts_l(model))
print(extractEnergyCapacity_l(model))

### Check sum Prod = Consumption
abs(model.solution['operation_conversion_power_out'].sum(['conversion_technology'])-parameters['exogeneous_energy_demand']).max()

#TODO add storage in production table
## visualisation de la série
production_df=model.solution['operation_conversion_power_out'].to_dataframe().\
    reset_index().pivot(index="date",columns='conversion_technology', values='operation_conversion_power_out')
fig=MyStackedPlotly(y_df=production_df,Conso = exogeneous_electricity_demand.to_dataframe())
fig=fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename=graphical_results_folder+'file.html') ## offline
#endregion

#region III -- multi-zone without storage - loading parameters
selected_area_to=["FR","DE"]
selected_conversion_technology=['old_nuke', 'ccgt','wind_power_on_shore',"demand_not_served"] #you'll add 'solar' after #'new_nuke', 'hydro_river', 'hydro_reservoir','wind_power_on_shore', 'wind_power_off_shore', 'solar', 'Curtailement'}
#selected_conversion_technology=['old_nuke','wind_power_on_shore', 'ccgt',"demand_not_served",'hydro_river', 'hydro_reservoir',"solar"] ## try adding 'hydro_river', 'hydro_reservoir'

xls_file=pd.ExcelFile(InputExcelFolder+"EU_7_2050.xlsx")
#TODO create an excel file with only two country to accelerate the code here
conversion_technology_parameters = pd.read_excel(xls_file, "conversion_technology").dropna().\
    set_index(["area_to", "conversion_technology","energy_vector_out"]).to_xarray()
storage_parameters = pd.read_excel(xls_file, "storage_technology").set_index(["energy_vector_out","area_to", "storage_technology"]).to_xarray()
exogeneous_electricity_demand = pd.read_excel(xls_file, "electricity_demand",parse_dates=['date']).dropna().\
    set_index(["area_to", "date"]).to_xarray().\
    expand_dims(dim={"energy_vector_out": ["electricity"]}, axis=1).\
    transpose("energy_vector_out","area_to", "date")
conversion_technology_parameters["operation_efficiency"].sum(["conversion_technology"])
exogeneous_energy_demand = exogeneous_electricity_demand
operation_conversion_availability_factor = pd.read_excel(xls_file, "operation_conversion_availabili",parse_dates=['date']).\
    dropna().set_index(["area_to", "date", "conversion_technology"]).to_xarray()
energy_vector_in = pd.read_excel(xls_file, "energy_vector_in").dropna().set_index(["area_to", "energy_vector_in"]).to_xarray()
Exchangeparameters = pd.read_excel(xls_file, "interconnexions").dropna().\
    set_index(["area_to", "area_from"]).to_xarray().\
    expand_dims(dim={"energy_vector_out": ["electricity"]}, axis=1)
Exchangeparameters.fillna(0)

selected_energy_vector_in_value = list(np.unique(conversion_technology_parameters.loc[{"conversion_technology" : selected_conversion_technology,"area_to": selected_area_to}]["energy_vector_in_value"].squeeze().to_numpy()))

parameters= xr.merge([  Exchangeparameters.select({"area_to": selected_area_to,"area_from": selected_area_to}),
                        exogeneous_energy_demand.select({"area_to": selected_area_to}),
                        operation_conversion_availability_factor.select({"conversion_technology" : selected_conversion_technology,
                                                            "area_to": selected_area_to}),
                        conversion_technology_parameters.loc[{"conversion_technology" : selected_conversion_technology,"area_to": selected_area_to}],
                        energy_vector_in.loc[{"energy_vector_in" : selected_energy_vector_in_value,"area_to": selected_area_to}]])

parameters["operation_conversion_availability_factor"]=parameters["operation_conversion_availability_factor"].fillna(1) ## 1 is the default value for availability factor
parameters["operation_efficiency"]=parameters["operation_efficiency"].fillna(0)

parameters["operation_min_1h_ramp_rate"].loc[{"conversion_technology" :"old_nuke"}] = 0.01
parameters["operation_max_1h_ramp_rate"].loc[{"conversion_technology" :"old_nuke"}] = 0.02
parameters["planning_max_capacity"].loc[{"conversion_technology" :"old_nuke"}]=80000
parameters["planning_max_capacity"].loc[{"conversion_technology" :"ccgt"}]=50000

#endregion

#region III -- multi-zone without storage -: building and solving problem, results visualisation
#building model and solving the problem
model = Build_EAP_Model(parameters=parameters)
model.solve(solver_name='highs',parallel = "on")
model.solve(solver_name='cbc')
#res= run_highs(model) #res= linopy.solvers.run_highs(model)

## synthèse Energie/Puissance/Coûts
print(extractCosts_l(model))
print(extractEnergyCapacity_l(model))

### Check sum Prod = Consumption
production_df = EnergyAndExchange2Prod(model)
abs(production_df.sum(axis=1)-parameters['exogeneous_energy_demand'].to_dataframe()["exogeneous_energy_demand"]).max()

## visualisation de la série
#TODO nettoyer le code des fonctions graphiques
fig=MyAreaStackedPlot(df_=production_df,Conso=exogeneous_energy_demand.to_dataframe())
fig=fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename=graphical_results_folder+'file.html') ## offline
#endregion

#region IV - Simple single area +4 million EV +  demande side management +30TWh H2: loading parameters
#TODO adapter le code ici pour le multi-énergie
Zones="FR" ; year=2013
#### reading energy_demand operation_conversion_availability_factor and conversion_technology_parameters CSV files
#energy_demand = pd.read_csv(InputConsumptionFolder+'energy_demand'+str(year)+'_'+str(Zones)+'.csv',sep=',',decimal='.',skiprows=0,parse_dates=['date']).set_index(["date"])


InputConsumptionFolder='Models/Basic_France_models/Consumption/Data/'
InputProductionFolder='Models/Basic_France_models/Production/Data/'
InputPlanningFolder='Models/Basic_France_models/Planning_optimisation/Data/'
graphical_results_folder="Models/Basic_France_models/Planning_optimisation/GraphicalResults/"


temperatureThreshold = 15
ConsoTempe_df=pd.read_csv(InputConsumptionFolder+'Consumptiontemperature_1996TO2019_FR.csv',parse_dates=['date']).\
    set_index(["date"]).loc[str(year)]
ConsoTempe_df=ConsoTempe_df[~ConsoTempe_df.index.duplicated(keep='first')]
(ConsoTempeYear_decomposed_df,Thermosensibilite)=Decomposeconso(ConsoTempe_df,temperatureThreshold=temperatureThreshold)


#obtaining industry-metal consumption
#  & x["type"] == "Ind" & x["UsageDetail"] == "Process").\
Profile_df_sans_chauffage=pd.read_csv(InputConsumptionFolder+"ConsumptionDetailedProfiles.csv").\
    rename(columns={'heures':'hour',"WeekDay":"day"}).\
    replace({"day" :{"Sat": "Samedi" , "Week":"Semaine"  , "Sun": "Dimanche"}}). \
    query('UsagesGroupe != "Chauffage"'). \
    assign(is_steel=lambda x: x["Nature"].isin(["MineraiMetal"])).\
    set_index(["Mois", "hour",'Nature', 'type',"is_steel", 'UsagesGroupe', 'UsageDetail', "day"]).\
    groupby(["Mois","day","hour","type","is_steel"]).sum().\
    merge(add_day_month_hour(df=ConsoTempeYear_decomposed_df,semaine_simplifie=True,French=True,to_index=True),
          how="outer",left_index=True,right_index=True).reset_index().set_index("date")[["type","is_steel","Conso"]]. \
    pivot_table(index="date", columns=["type","is_steel"], values='Conso')
Profile_df_sans_chauffage.columns = ["Autre","Ind_sans_acier","Ind_acier","Residentiel","Tertiaire"]

Profile_df_sans_chauffage=Profile_df_sans_chauffage.loc[:,Profile_df_sans_chauffage.sum(axis=0)>0]
Profile_df_n=Profile_df_sans_chauffage.div(Profile_df_sans_chauffage.sum(axis=1), axis=0) ### normalisation par 1 et multiplication
for col in Profile_df_sans_chauffage.columns:
    Profile_df_sans_chauffage[col]=Profile_df_n[col]*ConsoTempeYear_decomposed_df["NTS_C"]

steel_consumption=Profile_df_sans_chauffage.loc[:,"Ind_acier"]
steel_consumption.max()
steel_consumption[steel_consumption.isna()]=110
steel_consumption.isna().sum()
# if you want to change thermal sensitivity + add electric vehicle

VEProfile_df=pd.read_csv(InputConsumptionFolder+'EVModel.csv', sep=';')
NbVE=10 # millions
ev_consumption = NbVE*Profile2Consumption(Profile_df=VEProfile_df,temperature_df = ConsoTempe_df.loc[str(year)][['temperature']])[['Consumption']]

h2_Energy = 30000## H2 volume in GWh/year
h2_Energy_flat_consumption = ev_consumption.Consumption*0+h2_Energy/8760
to_flexible_consumption=pd.DataFrame({'to_flex_consumption': steel_consumption,'flexible_demand' : 'Steel'}).reset_index().set_index(['date','flexible_demand']).\
    append(pd.DataFrame({'to_flex_consumption': ev_consumption.Consumption,'flexible_demand' : 'EV'}).reset_index().set_index(['date','flexible_demand'])).\
    append(pd.DataFrame({'to_flex_consumption': h2_Energy_flat_consumption,'flexible_demand' : 'H2'}).reset_index().set_index(['date','flexible_demand']))

operation_conversion_availability_factor = pd.read_csv(InputProductionFolder+'operation_conversion_availability_factor'+str(year)+'_'+str(Zones)+'.csv',sep=',',decimal='.',skiprows=0,parse_dates=['date']).set_index(["date","conversion_technology"])



conversion_technology_parameters = pd.read_csv(InputPlanningFolder+'Planning-RAMP1BIS_conversion_technology.csv',sep=',',decimal='.',skiprows=0,comment="#").set_index(["conversion_technology"])
storage_parameters = pd.read_csv(InputPlanningFolder + 'Planning-RAMP1_storage_technology.csv', sep=',', decimal='.',
                                skiprows=0).set_index(["storage_technology"])
Consoparameters = pd.read_csv(InputPlanningFolder + "Planning-Conso-flexible_demand.csv", sep=";").set_index(["flexible_demand"])
Consoparameters_ = Consoparameters.join(
    to_flexible_consumption.groupby("flexible_demand").max().rename(columns={"to_flexible_consumption": "max_power"}))

selected_conversion_technology=['old_nuke','ccgt','ocgt', 'wind_power_on_shore', 'wind_power_off_shore','hydro_reservoir','hydro_river','solar','demand_not_served']#you can add technologies here
operation_conversion_availability_factor=operation_conversion_availability_factor.loc[(slice(None),selected_conversion_technology),:]
conversion_technology_parameters=conversion_technology_parameters.loc[selected_conversion_technology,:]

conversion_technology_parameters.loc["ccgt",'operation_fuel_cost']=100
conversion_technology_parameters.loc["ccgt",'planning_max_capacity']=50000
conversion_technology_parameters.loc["wind_power_on_shore",'planning_conversion_cost']=120000 #€/MW/year - investment+O&M fixed cost
conversion_technology_parameters.loc["solar",'planning_conversion_cost']=65000 #€/MW/year
conversion_technology_parameters.loc["ccgt",'operation_min_1h_ramp_rate']=0.4 ## a bit strong to put in light the effect
conversion_technology_parameters.loc["ccgt",'operation_max_1h_ramp_rate']=0.4 ## a bit strong to put in light the effect
storage_parameters.loc["Battery1","planning_storage_max_power"]=10000 # this is not optimized - batteries
storage_parameters.loc["Battery2","planning_storage_max_power"]=7000 # this is not optimized - Pumped HS
storage_parameters.loc["Battery2","planning_storage_max_capacity"]=storage_parameters.loc["Battery2","planning_storage_max_power"]*20 # this is not optimized 20h of Pumped HS

energy_demand=pd.DataFrame(ConsoTempeYear_decomposed_df.loc[:,"Consumption"]-steel_consumption,columns=["energy_demand"])

def labour_ratio_cost(df):  # higher labour costs at night
    if df.hour in range(7, 17):
        return 1
    elif df.hour in range(17, 23):
        return 1.5
    else:
        return 2


labour_ratio = pd.DataFrame()
labour_ratio["date"] = energy_demand.index.get_level_values('date')
labour_ratio["flexible_demand"] = "Steel"
labour_ratio["labour_ratio"] = labour_ratio["date"].apply(labour_ratio_cost)
labour_ratio.set_index(["date","flexible_demand"], inplace=True)
#model.labour_ratio = Param(model.date, initialize=labour_ratio.squeeze().to_dict())

if "to_flex_consumption" not in Consoparameters:
    Consoparameters_ = Consoparameters.join(
        to_flexible_consumption.groupby("flexible_demand").max().rename(columns={"to_flex_consumption": "max_power"}))
else:
    Consoparameters_ = Consoparameters.rename(columns={"to_flex_consumption": "max_power"})

parameters= xr.merge([  energy_demand.to_xarray(),
                        operation_conversion_availability_factor.to_xarray().select({"conversion_technology" : selected_conversion_technology}),
                        conversion_technology_parameters.to_xarray().loc[{"conversion_technology" : selected_conversion_technology}],
                        storage_parameters.to_xarray(),
                        to_flexible_consumption.to_xarray(),
                        labour_ratio.to_xarray(),
                        Consoparameters_.to_xarray()])
parameters=parameters.expand_dims(dim={"area_from": [Zones]}, axis=0)

parameters["operation_conversion_availability_factor"]=parameters["operation_conversion_availability_factor"].fillna(1) ## 1 is the default value for availability factor
# endregion

#region IV -- Simple single area +4 million EV +  demande side management +30TWh H2 : building and solving problem, results visualisation
#building model and solving the problem
model = Build_EAP_Model(parameters=parameters)
model.solve(solver_name='cbc')# highs not faster than cbc
#res= run_highs(model) #res= linopy.solvers.run_highs(model)

## synthèse Energie/Puissance/Coûts
print(extractCosts_l(model))
print(extractEnergyCapacity_l(model))

### Check sum Prod == sum Consumption
Prod_minus_conso = model.solution['operation_conversion_power_out'].sum(['conversion_technology']) - model.solution['total_demand'] + model.solution['operation_storage_power_out'].sum(['storage_technology']) - model.solution['operation_storage_power_in'].sum(['storage_technology']) ## Storage
abs(Prod_minus_conso).max()

Storage_production = (model.solution['operation_storage_power_out'] - model.solution['operation_storage_power_in']).rename({"storage_technology":"conversion_technology"})
Storage_production.name = "operation_conversion_power_out"
production_xr = xr.combine_by_coords([model.solution['operation_conversion_power_out'],Storage_production])

## visualisation de la série
production_df=production_xr.to_dataframe().reset_index().pivot(index="date",columns='conversion_technology', values='operation_conversion_power_out')
fig=MyStackedPlotly(y_df=production_df,Conso = model.solution['total_demand'].to_dataframe())
fig=fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename=graphical_results_folder+'file.html') ## offline
#endregion

#region V - 7 node EU model - loading parameters
graphical_results_folder="Models/Basic_France_Germany_models/Planning_optimisation/GraphicalResults/"
InputExcelFolder = "Models/Seven_node_Europe/"
xls_file=pd.ExcelFile(InputExcelFolder+"EU_7_2050.xlsx")
year=2018

#### reading tables in xls
conversion_technology_parameters = pd.read_excel(xls_file, "TECHNO_area_from").dropna().set_index(["area_from", "conversion_technology"]).to_xarray()
storage_parameters = pd.read_excel(xls_file, "storage_technology_area_from").set_index(["area_from", "storage_technology"]).to_xarray()
energy_demand = pd.read_excel(xls_file, "energy_demand",parse_dates=['date']).dropna().set_index(["area_from", "date"]).to_xarray()
operation_conversion_availability_factor = pd.read_excel(xls_file, "availability_factor",parse_dates=['date']).\
    dropna().set_index(["area_from", "date", "conversion_technology"]).to_xarray()
Exchangeparameters = pd.read_excel(xls_file, "interconnexions").\
    rename(columns = {"area_from.1":"area_from_1"}).set_index(["area_from", "area_from_1"]).to_xarray()

Exchangeparameters.fillna(0)
parameters= xr.merge([  energy_demand,operation_conversion_availability_factor,storage_parameters,conversion_technology_parameters,Exchangeparameters])
#parameters= xr.merge([  energy_demand.select({"area_from" : Selected_area_from}),
#                        operation_conversion_availability_factor.select({"area_from" : Selected_area_from,"conversion_technology" : selected_conversion_technology}),
#                        storage_parameters,
#                        conversion_technology_parameters.loc[{"conversion_technology" : selected_conversion_technology}],
#                        Exchangeparameters.select({"area_from" : Selected_area_from,"area_from_1" : Selected_area_from})])

parameters["operation_conversion_availability_factor"]=parameters["operation_conversion_availability_factor"].fillna(1) ## 1 is the default value for availability factor

#endregion

#region V -- 7node EU model : building and solving problem, results visualisation
#building model and solving the problem
model = Build_EAP_Model(parameters=parameters)
model.solve(solver_name='highs',parallel = "on")
model.solve(solver_name='cplex')
#res= run_highs(model) #res= linopy.solvers.run_highs(model)

## synthèse Energie/Puissance/Coûts
print(extractCosts_l(model))
print(extractEnergyCapacity_l(model))

### Check sum Prod = Consumption
Variables = {name: model.solution[name].to_dataframe().reset_index() for name in list(model.solution.keys())}
production_df = EnergyAndExchange2Prod(Variables)
abs(production_df.sum(axis=1)-parameters['energy_demand'].to_dataframe()["energy_demand"]).max()

## visualisation de la série
production_df = EnergyAndExchange2Prod(Variables)
fig=MyAreaStackedPlot(df_=production_df,Conso=energy_demand.to_dataframe())
fig=fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename=graphical_results_folder+'file.html') ## offline
#endregion


## sur le nombre de contraintes
A_T = 4; A_ST = 3; D_A = 2; D_A_T = 4; D_A_ST = 5
A = 7 ; ST =2 ; D = 8760 ; T = 10

A*T*A_T + A*ST*A_ST+D*A*D_A+D*A*T*D_A_T+D*A*ST*D_A_ST


exogeneous_energy_demand = xr.DataArray(
    0,
    dims=["area_to", "date","energy_vector_out"],
    coords={"area_to": conversion_technology_parameters.get_index("area_to"),
            "date": exogeneous_electricity_demand.get_index("date"),
            "energy_vector_out" :  conversion_technology_parameters.get_index("energy_vector_out")}
)