
#region initialisation
#sys.path.append("/Users/robin.girard/opt/anaconda3/envs/energyalternatives/lib/python3.10/site-packages/highspy/.dylibs/")
#import highspy

import pandas as pd
import sys
sys.path.extend(['.'])

pd.options.display.width = 0
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


InputConsumptionFolder='Models/Basic_France_models/Consumption/Data/'
InputProductionFolder='Models/Basic_France_models/Production/Data/'
InputPlanningFolder='Models/Basic_France_models/Planning_optimisation/Data/'
GraphicalResultsFolder="Models/Basic_France_models/Planning_optimisation/GraphicalResults/"

from EnergyAlternativesPlanning.f_graphicalTools import *
from Models.Linopy.f_tools import *
from Models.Linopy.f_planningModels_linopy import Build_EAP_Model

#endregion

#region I - Simple single area (with ramp) : loading parameters
Zones="FR"
year=2013
Selected_TECHNOLOGIES=['OldNuke', 'CCG',"curtailment"] #you'll add 'Solar' after

#### reading areaConsumption availabilityFactor and TechParameters CSV files
areaConsumption = pd.read_csv(InputConsumptionFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["Date"]).to_xarray()
availabilityFactor = pd.read_csv(InputProductionFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["Date","TECHNOLOGIES"]).to_xarray()
TechParameters = pd.read_csv(InputPlanningFolder+'Planning-RAMP_TECHNOLOGIES.csv',sep=',',decimal='.',skiprows=0,comment="#").set_index(["TECHNOLOGIES"]).to_xarray()

TechParameters["RampConstraintMoins"].loc[{"TECHNOLOGIES" :"OldNuke"}] = 0.01
TechParameters["RampConstraintPlus"].loc[{"TECHNOLOGIES" :"OldNuke"}] = 0.02

Parameters= xr.merge([  areaConsumption,
                        availabilityFactor.select({"TECHNOLOGIES" : Selected_TECHNOLOGIES}),
                        TechParameters.loc[{"TECHNOLOGIES" : Selected_TECHNOLOGIES}]])
Parameters=Parameters.expand_dims(dim={"AREAS": [Zones]}, axis=0)

Parameters["availabilityFactor"]=Parameters["availabilityFactor"].fillna(1) ## 1 is the default value for availability factor
#endregion

#region I - Simple single area (with ramp) : building and solving problem, results visualisation
#building model and solving the problem
model = Build_EAP_Model(Parameters=Parameters)
model.solve(solver_name='cbc')
#res= run_highs(model) #res= linopy.solvers.run_highs(model)

## synthèse Energie/Puissance/Coûts
print(extractCosts_l(model))
print(extractEnergyCapacity_l(model))

### Check sum Prod = Consumption
abs(model.solution['production_op_power'].sum(['TECHNOLOGIES'])-Parameters['areaConsumption']).max()

## visualisation de la série
production_df=model.solution['production_op_power'].to_dataframe().\
    reset_index().pivot(index="Date",columns='TECHNOLOGIES', values='production_op_power')
fig=MyStackedPlotly(y_df=production_df,Conso = areaConsumption.to_dataframe())
fig=fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename=GraphicalResultsFolder+'file.html') ## offline
#endregion

#region II - addition of Storage to single area with ramp : loading parameters
Zones="FR"
year=2013

Selected_TECHNOLOGIES=['OldNuke','WindOnShore', 'CCG',"curtailment",'HydroRiver', 'HydroReservoir',"Solar"] ## try adding 'HydroRiver', 'HydroReservoir'

#### reading CSV files
areaConsumption = pd.read_csv(InputConsumptionFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["Date"]).to_xarray()
availabilityFactor = pd.read_csv(InputProductionFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["Date","TECHNOLOGIES"]).to_xarray()
TechParameters = pd.read_csv(InputPlanningFolder+'Planning-RAMP1BIS_TECHNOLOGIES.csv',
                             sep=',',decimal='.',skiprows=0,comment="#").set_index(["TECHNOLOGIES"]).to_xarray()
StorageParameters = pd.read_csv(InputPlanningFolder+'Planning-RAMP1_STOCK_TECHNO.csv',sep=',',decimal='.',skiprows=0).set_index(["STOCK_TECHNO"]).to_xarray()


Parameters= xr.merge([  areaConsumption,
                        availabilityFactor.select({"TECHNOLOGIES" : Selected_TECHNOLOGIES}),
                        TechParameters.loc[{"TECHNOLOGIES" : Selected_TECHNOLOGIES}],
                        StorageParameters])
Parameters=Parameters.expand_dims(dim={"AREAS": [Zones]}, axis=0)

Parameters["availabilityFactor"]=Parameters["availabilityFactor"].fillna(1) ## 1 is the default value for availability factor

#endregion

#region II -addition of Storage to single area with ramp : building and solving problem, results visualisation
#building model and solving the problem
model = Build_EAP_Model(Parameters=Parameters)
model.solve(solver_name='cbc')
#res= run_highs(model) #res= linopy.solvers.run_highs(model)

## synthèse Energie/Puissance/Coûts
print(extractCosts_l(model))
print(extractEnergyCapacity_l(model))

### Check sum Prod = Consumption
abs(model.solution['production_op_power'].sum(['TECHNOLOGIES'])-Parameters['areaConsumption']).max()

## visualisation de la série
production_df=model.solution['production_op_power'].to_dataframe().\
    reset_index().pivot(index="Date",columns='TECHNOLOGIES', values='production_op_power')
fig=MyStackedPlotly(y_df=production_df,Conso = areaConsumption.to_dataframe())
fig=fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename=GraphicalResultsFolder+'file.html') ## offline
#endregion

#region III -- multi-zone without storage - loading parameters

InputConsumptionFolder='Models/Basic_France_Germany_models/Consumption/Data/'
InputProductionFolder='Models/Basic_France_Germany_models/Production/Data/'
InputPlanningFolder='Models/Basic_France_Germany_models/Planning_optimisation/Data/'
GraphicalResultsFolder="Models/Basic_France_Germany_models/Planning_optimisation/GraphicalResults/"
InputEcoAndTech = 'Models/Basic_France_Germany_models/Economic_And_Tech_Assumptions/'

Zones="FR_DE_GB_ES"
year=2016
Selected_AREAS=["FR","DE"]
Selected_TECHNOLOGIES=['OldNuke', 'CCG','WindOnShore',"curtailment"] #you'll add 'Solar' after #'NewNuke', 'HydroRiver', 'HydroReservoir','WindOnShore', 'WindOffShore', 'Solar', 'Curtailement'}

#### reading CSV files
areaConsumption = pd.read_csv(InputConsumptionFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["AREAS","Date"]).to_xarray()
availabilityFactor = pd.read_csv(InputProductionFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["AREAS","Date","TECHNOLOGIES"]).to_xarray()
TechParameters = pd.read_csv(InputPlanningFolder+'Planning_MultiNode_DE-FR_TECHNOLOGIES_AREAS.csv',
                             sep=',',decimal='.',skiprows=0,comment="#").set_index(["AREAS","TECHNOLOGIES"]).to_xarray()
ExchangeParameters = pd.read_csv(InputEcoAndTech+'Hypothese_DE-FR_AREAS_AREAS.csv',sep=',',decimal='.',skiprows=0,comment="#").\
    rename(columns = {"AREAS.1":"AREAS_1"}).set_index(["AREAS","AREAS_1"]).to_xarray()
ExchangeParameters.fillna(0)
Parameters= xr.merge([  areaConsumption.select({"AREAS" : Selected_AREAS}),
                        availabilityFactor.select({"AREAS" : Selected_AREAS,"TECHNOLOGIES" : Selected_TECHNOLOGIES}),
                        TechParameters.loc[{"TECHNOLOGIES" : Selected_TECHNOLOGIES}],
                        ExchangeParameters.select({"AREAS" : Selected_AREAS,"AREAS_1" : Selected_AREAS})])

Parameters["availabilityFactor"]=Parameters["availabilityFactor"].fillna(1) ## 1 is the default value for availability factor

#endregion

#region III -- multi-zone without storage -: building and solving problem, results visualisation
#building model and solving the problem
model = Build_EAP_Model(Parameters=Parameters)
model.solve(solver_name='cbc')
#res= run_highs(model) #res= linopy.solvers.run_highs(model)

## synthèse Energie/Puissance/Coûts
print(extractCosts_l(model))
print(extractEnergyCapacity_l(model))

### Check sum Prod = Consumption
Variables = {name: model.solution[name].to_dataframe().reset_index() for name in list(model.solution.keys())}
production_df = EnergyAndExchange2Prod(Variables)
abs(production_df.sum(axis=1)-Parameters['areaConsumption'].to_dataframe()["areaConsumption"]).max()

## visualisation de la série
production_df = EnergyAndExchange2Prod(Variables)
fig=MyAreaStackedPlot(df_=production_df,Conso=areaConsumption.to_dataframe())
fig=fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename=GraphicalResultsFolder+'file.html') ## offline
#endregion
