
#region initialisation
#sys.path.append("/Users/robin.girard/opt/anaconda3/envs/energyalternatives/lib/python3.10/site-packages/highspy/.dylibs/")
import highspy

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
from EnergyAlternativesPlanning.f_consumptionModels import *
from Models.Linopy.f_tools import *
from Models.Linopy.f_planningModels_linopy import Build_EAP_Model,run_highs

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
model.solve(solver_name='highs')
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
model.solve(solver_name='highs',parallel="on",simplex_max_concurrency=8)
# https://ergo-code.github.io/HiGHS/dev/parallel/
#"Unless an LP has significantly more variables than constraints, the parallel dual simplex solver is unlikely to be worth using."
#model.solve(solver_name='cbc',threads = 16) ## multi-threads does not improve results ?
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

#region IV - Simple single area +4 million EV +  demande side management +30TWh H2: loading parameters
Zones="FR" ; year=2013
#### reading areaConsumption availabilityFactor and TechParameters CSV files
#areaConsumption = pd.read_csv(InputConsumptionFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["Date"])

TemperatureThreshold = 15
ConsoTempe_df=pd.read_csv(InputConsumptionFolder+'ConsumptionTemperature_1996TO2019_FR.csv',parse_dates=['Date']).\
    set_index(["Date"])[str(year)]
ConsoTempe_df=ConsoTempe_df[~ConsoTempe_df.index.duplicated(keep='first')]
(ConsoTempeYear_decomposed_df,Thermosensibilite)=Decomposeconso(ConsoTempe_df,TemperatureThreshold=TemperatureThreshold)


#obtaining industry-metal consumption
#  & x["type"] == "Ind" & x["UsageDetail"] == "Process").\
Profile_df_sans_chauffage=pd.read_csv(InputConsumptionFolder+"ConsumptionDetailedProfiles.csv").\
    rename(columns={'heures':'Heure',"WeekDay":"Jour"}).\
    replace({"Jour" :{"Sat": "Samedi" , "Week":"Semaine"  , "Sun": "Dimanche"}}). \
    query('UsagesGroupe != "Chauffage"'). \
    assign(is_steel=lambda x: x["Nature"].isin(["MineraiMetal"])).\
    set_index(["Mois", "Heure",'Nature', 'type',"is_steel", 'UsagesGroupe', 'UsageDetail', "Jour"]).\
    groupby(["Mois","Jour","Heure","type","is_steel"]).sum().\
    merge(add_day_month_hour(df=ConsoTempeYear_decomposed_df,semaine_simplifie=True,French=True,to_index=True),
          how="outer",left_index=True,right_index=True).reset_index().set_index("Date")[["type","is_steel","Conso"]]. \
    pivot_table(index="Date", columns=["type","is_steel"], values='Conso')
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
ev_consumption = NbVE*Profile2Consumption(Profile_df=VEProfile_df,Temperature_df = ConsoTempe_df.loc[str(year)][['Temperature']])[['Consumption']]

h2_Energy = 30000## H2 volume in GWh/year
h2_Energy_flat_consumption = ev_consumption.Consumption*0+h2_Energy/8760
to_flexible_consumption=pd.DataFrame({'to_flex_consumption': steel_consumption,'FLEX_CONSUM' : 'Steel'}).reset_index().set_index(['Date','FLEX_CONSUM']).\
    append(pd.DataFrame({'to_flex_consumption': ev_consumption.Consumption,'FLEX_CONSUM' : 'EV'}).reset_index().set_index(['Date','FLEX_CONSUM'])).\
    append(pd.DataFrame({'to_flex_consumption': h2_Energy_flat_consumption,'FLEX_CONSUM' : 'H2'}).reset_index().set_index(['Date','FLEX_CONSUM']))

availabilityFactor = pd.read_csv(InputProductionFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["Date","TECHNOLOGIES"])



TechParameters = pd.read_csv(InputPlanningFolder+'Planning-RAMP1BIS_TECHNOLOGIES.csv',sep=',',decimal='.',skiprows=0,comment="#").set_index(["TECHNOLOGIES"])
StorageParameters = pd.read_csv(InputPlanningFolder + 'Planning-RAMP1_STOCK_TECHNO.csv', sep=',', decimal='.',
                                skiprows=0).set_index(["STOCK_TECHNO"])
ConsoParameters = pd.read_csv(InputPlanningFolder + "Planning-Conso-FLEX_CONSUM.csv", sep=";").set_index(["FLEX_CONSUM"])
ConsoParameters_ = ConsoParameters.join(
    to_flexible_consumption.groupby("FLEX_CONSUM").max().rename(columns={"to_flexible_consumption": "max_power"}))

Selected_TECHNOLOGIES=['OldNuke','CCG','TAC', 'WindOnShore', 'WindOffShore','HydroReservoir','HydroRiver','Solar','curtailment']#you can add technologies here
availabilityFactor=availabilityFactor.loc[(slice(None),Selected_TECHNOLOGIES),:]
TechParameters=TechParameters.loc[Selected_TECHNOLOGIES,:]

TechParameters.loc["CCG",'energyCost']=100
TechParameters.loc["CCG",'maxCapacity']=50000
TechParameters.loc["WindOnShore",'capacityCost']=120000 #€/MW/year - investment+O&M fixed cost
TechParameters.loc["Solar",'capacityCost']=65000 #€/MW/year
TechParameters.loc["CCG",'RampConstraintMoins']=0.4 ## a bit strong to put in light the effect
TechParameters.loc["CCG",'RampConstraintPlus']=0.4 ## a bit strong to put in light the effect
StorageParameters.loc["Battery1","p_max"]=10000 # this is not optimized - batteries
StorageParameters.loc["Battery2","p_max"]=7000 # this is not optimized - Pumped HS
StorageParameters.loc["Battery2","c_max"]=StorageParameters.loc["Battery2","p_max"]*20 # this is not optimized 20h of Pumped HS

areaConsumption=pd.DataFrame(ConsoTempeYear_decomposed_df.loc[:,"Consumption"]-steel_consumption,columns=["areaConsumption"])

def labour_ratio_cost(df):  # higher labour costs at night
    if df.hour in range(7, 17):
        return 1
    elif df.hour in range(17, 23):
        return 1.5
    else:
        return 2


labour_ratio = pd.DataFrame()
labour_ratio["Date"] = areaConsumption.index.get_level_values('Date')
labour_ratio["FLEX_CONSUM"] = "Steel"
labour_ratio["labour_ratio"] = labour_ratio["Date"].apply(labour_ratio_cost)
labour_ratio.set_index(["Date","FLEX_CONSUM"], inplace=True)
#model.labour_ratio = Param(model.Date, initialize=labour_ratio.squeeze().to_dict())

if "to_flex_consumption" not in ConsoParameters:
    ConsoParameters_ = ConsoParameters.join(
        to_flexible_consumption.groupby("FLEX_CONSUM").max().rename(columns={"to_flex_consumption": "max_power"}))
else:
    ConsoParameters_ = ConsoParameters.rename(columns={"to_flex_consumption": "max_power"})

Parameters= xr.merge([  areaConsumption.to_xarray(),
                        availabilityFactor.to_xarray().select({"TECHNOLOGIES" : Selected_TECHNOLOGIES}),
                        TechParameters.to_xarray().loc[{"TECHNOLOGIES" : Selected_TECHNOLOGIES}],
                        StorageParameters.to_xarray(),
                        to_flexible_consumption.to_xarray(),
                        labour_ratio.to_xarray(),
                        ConsoParameters_.to_xarray()])
Parameters=Parameters.expand_dims(dim={"AREAS": [Zones]}, axis=0)

Parameters["availabilityFactor"]=Parameters["availabilityFactor"].fillna(1) ## 1 is the default value for availability factor
# endregion

#region IV -- Simple single area +4 million EV +  demande side management +30TWh H2 : building and solving problem, results visualisation
#building model and solving the problem
model = Build_EAP_Model(Parameters=Parameters)
model.solve(solver_name='cbc')# highs not faster than cbc
#res= run_highs(model) #res= linopy.solvers.run_highs(model)

## synthèse Energie/Puissance/Coûts
print(extractCosts_l(model))
print(extractEnergyCapacity_l(model))

### Check sum Prod == sum Consumption
Prod_minus_conso = model.solution['production_op_power'].sum(['TECHNOLOGIES']) - model.solution['conso_op_totale'] + model.solution['storage_op_power_out'].sum(['STOCK_TECHNO']) - model.solution['storage_op_power_in'].sum(['STOCK_TECHNO']) ## Storage
abs(Prod_minus_conso).max()

Storage_production = (model.solution['storage_op_power_out'] - model.solution['storage_op_power_in']).rename({"STOCK_TECHNO":"TECHNOLOGIES"})
Storage_production.name = "production_op_power"
production_xr = xr.combine_by_coords([model.solution['production_op_power'],Storage_production])

## visualisation de la série
production_df=production_xr.to_dataframe().reset_index().pivot(index="Date",columns='TECHNOLOGIES', values='production_op_power')
fig=MyStackedPlotly(y_df=production_df,Conso = model.solution['conso_op_totale'].to_dataframe())
fig=fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename=GraphicalResultsFolder+'file.html') ## offline
#endregion

#region V - 7 node EU model - loading parameters
GraphicalResultsFolder="Models/Basic_France_Germany_models/Planning_optimisation/GraphicalResults/"
InputExcelFolder = "Models/Seven_node_Europe/"
xls_file=pd.ExcelFile(InputExcelFolder+"Multinode_2050.xlsx")
year=2018

#### reading tables in xls
TechParameters = pd.read_excel(xls_file, "TECHNO_AREAS").dropna().set_index(["AREAS", "TECHNOLOGIES"]).to_xarray()
StorageParameters = pd.read_excel(xls_file, "STOCK_TECHNO_AREAS").set_index(["AREAS", "STOCK_TECHNO"]).to_xarray()
areaConsumption = pd.read_excel(xls_file, "areaConsumption",parse_dates=['Date']).dropna().set_index(["AREAS", "Date"]).to_xarray()
availabilityFactor = pd.read_excel(xls_file, "availability_factor",parse_dates=['Date']).\
    dropna().set_index(["AREAS", "Date", "TECHNOLOGIES"]).to_xarray()
ExchangeParameters = pd.read_excel(xls_file, "interconnexions").\
    rename(columns = {"AREAS.1":"AREAS_1"}).set_index(["AREAS", "AREAS_1"]).to_xarray()

ExchangeParameters.fillna(0)
Parameters= xr.merge([  areaConsumption,availabilityFactor,StorageParameters,TechParameters,ExchangeParameters])
#Parameters= xr.merge([  areaConsumption.select({"AREAS" : Selected_AREAS}),
#                        availabilityFactor.select({"AREAS" : Selected_AREAS,"TECHNOLOGIES" : Selected_TECHNOLOGIES}),
#                        StorageParameters,
#                        TechParameters.loc[{"TECHNOLOGIES" : Selected_TECHNOLOGIES}],
#                        ExchangeParameters.select({"AREAS" : Selected_AREAS,"AREAS_1" : Selected_AREAS})])

Parameters["availabilityFactor"]=Parameters["availabilityFactor"].fillna(1) ## 1 is the default value for availability factor

#endregion

#region V -- 7node EU model : building and solving problem, results visualisation
#building model and solving the problem
model = Build_EAP_Model(Parameters=Parameters)
model.solve(solver_name='cbc')# highs but cbc is faster
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

