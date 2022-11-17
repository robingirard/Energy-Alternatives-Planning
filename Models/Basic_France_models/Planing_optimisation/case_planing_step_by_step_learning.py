
#region importation of modules
import os
import sys
if sys.platform != 'win32':
    myhost = os.uname()[1]
else : myhost = ""
if (myhost=="jupyter-sop"):
    ## for https://jupyter-sop.mines-paristech.fr/ users, you need to
    #  (1) run the following in a terminal
    if (os.system(
            "/opt/mosek/9.2/tools/platform/linux64x86/bin/lmgrd -c /opt/mosek/9.2/tools/platform/linux64x86/bin/mosek.lic -l lmgrd.log") == 0):
        os.system("/opt/mosek/9.2/tools/platform/linux64x86/bin/lmutil lmstat -c 27007@127.0.0.1 -a")
    #  (2) definition of license
    os.environ["MOSEKLM_LICENSE_FILE"] = '@jupyter-sop'

#import docplex


from EnergyAlternativesPlaning.f_graphicalTools import *
from EnergyAlternativesPlaning.f_consumptionModels import *
from EnergyAlternativesPlaning.f_model_definition import *
from Models.Basic_France_models.Planing_optimisation.f_planingModels import *

#endregion

#region Solver and data location definition
InputFolder='Data/input/'
solver= 'mosek' ## no need for solverpath with mosek.
BaseSolverPath='/Users/robin.girard/Documents/Code/Packages/solvers/ampl_macosx64'
sys.path.append(BaseSolverPath)

solvers= ['gurobi','knitro','cbc'] # 'glpk' is too slow 'cplex' and 'xpress' do not work
solverpath= {}
for solver in solvers : solverpath[solver]=BaseSolverPath+'/'+solver
cplexPATH='/Applications/CPLEX_Studio1210/cplex/bin/x86-64_osx'
sys.path.append(cplexPATH)
solverpath['cplex']=cplexPATH+"/"+"cplex"
solver = 'mosek'
#endregion

#region location of data + output visualisation settings
pd.options.display.width = 0
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

InputConsumptionFolder='Models/Basic_France_models/Consumption/Data/'
InputProductionFolder='Models/Basic_France_models/Production/Data/'
InputPlaningFolder='Models/Basic_France_models/Planing_optimisation/Data/'
GraphicalResultsFolder="Models/Basic_France_models/Planing_optimisation/GraphicalResults/"
#endregion

#region I - Simple single area : loading parameters
year=2013
#### reading areaConsumption availabilityFactor and TechParameters CSV files
areaConsumption = pd.read_csv(InputConsumptionFolder+'areaConsumption'+str(year)+'_FR.csv',sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["Date"])
availabilityFactor = pd.read_csv(InputProductionFolder+'availabilityFactor'+str(year)+'_FR.csv',sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["Date","TECHNOLOGIES"])
TechParameters = pd.read_csv(InputPlaningFolder+'Planing-Simple_TECHNOLOGIES.csv',sep=',',decimal='.',skiprows=0,comment="#").set_index(["TECHNOLOGIES"])
TechParameters.head()
#### Selection of subset
Selected_TECHNOLOGIES=['OldNuke','CCG'] #you can add technologies here
availabilityFactor=availabilityFactor.loc[(slice(None),Selected_TECHNOLOGIES),:]
TechParameters=TechParameters.loc[Selected_TECHNOLOGIES,:]
#TechParameters.loc[TechParameters.TECHNOLOGIES=="OldNuke",'maxCapacity']=63000 ## Limit to actual installed capacity
#endregion

#region I - Simple single area  : Solving and loading results
model = GetElectricSystemModel_PlaningSingleNode(Parameters={"areaConsumption"      :   areaConsumption,
                                                   "availabilityFactor"   :   availabilityFactor,
                                                   "TechParameters"       :   TechParameters})

if solver in solverpath :  opt = SolverFactory(solver,executable=solverpath[solver])
else : opt = SolverFactory(solver)
results=opt.solve(model)
## result analysis
Variables=getVariables_panda_indexed(model)
print(extractCosts(Variables))
print(extractEnergyCapacity(Variables))

#pour avoir la production en KWh de chaque moyen de prod chaque heure
production_df=Variables['energy'].pivot(index="Date",columns='TECHNOLOGIES', values='energy')
### Check sum Prod = Consumption
Delta=(production_df.sum(axis=1) - areaConsumption.areaConsumption);
abs(Delta).max()
#endregion

#region I - Simple single area  : visualisation and lagrange multipliers
### representation des résultats
fig=MyStackedPlotly(y_df=production_df,Conso = areaConsumption)
fig=fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename=GraphicalResultsFolder+'file.html') ## offline
#fig2.show()

#### lagrange multipliers
Constraints= getConstraintsDual_panda(model)

# Analyse energyCtr
energyCtrDual=Constraints['energyCtr']; energyCtrDual['energyCtr']=energyCtrDual['energyCtr']
energyCtrDual
round(energyCtrDual.energyCtr,2).unique()

# Analyse CapacityCtr
#CapacityCtrDual=Constraints['maxCapacityCtr'].pivot(index="Date",columns='TECHNOLOGIES', values='CapacityCtr')*1000000;
#round(CapacityCtrDual,2)
#round(CapacityCtrDual.OldNuke,2).unique() ## if you increase by Delta the installed capacity of nuke you decrease by xxx the cost when nuke is not sufficient
#round(CapacityCtrDual.CCG,2).unique() ## increasing the capacity of CCG as no effect on prices
#endregion

#region II - Ramp Single area : loading parameters loading parameterscase with ramp constraints
year=2013
Selected_TECHNOLOGIES=['OldNuke', 'CCG',"curtailment"] #you'll add 'Solar' after
#### reading CSV files
areaConsumption = pd.read_csv(InputConsumptionFolder+'areaConsumption'+str(year)+'_FR.csv',
                                sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["Date"])
availabilityFactor = pd.read_csv(InputProductionFolder+'availabilityFactor'+str(year)+'_FR.csv',
                                sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["Date","TECHNOLOGIES"])
TechParameters = pd.read_csv(InputPlaningFolder+'Planing-RAMP1BIS_TECHNOLOGIES.csv',sep=',',decimal='.',skiprows=0,comment="#").set_index(["TECHNOLOGIES"])

#### Selection of subset
availabilityFactor=availabilityFactor.loc[(slice(None),Selected_TECHNOLOGIES),:]
TechParameters=TechParameters.loc[Selected_TECHNOLOGIES,:]
TechParameters.loc["OldNuke",'RampConstraintMoins']=0.01 ## a bit strong to put in light the effect
TechParameters.loc["OldNuke",'RampConstraintPlus']=0.02 ## a bit strong to put in light the effect
#endregion

#region II - Ramp Single area : solving and loading results
model = GetElectricSystemModel_PlaningSingleNode(Parameters={"areaConsumption"      :   areaConsumption,
                                                   "availabilityFactor"   :   availabilityFactor,
                                                   "TechParameters"       :   TechParameters})
opt = SolverFactory(solver)
results=opt.solve(model)
Variables=getVariables_panda_indexed(model)
print(extractCosts(Variables))
print(extractEnergyCapacity(Variables))

#pour avoir la production en KWh de chaque moyen de prod chaque heure
production_df=Variables['energy'].pivot(index="Date",columns='TECHNOLOGIES', values='energy')
### Check sum Prod = Consumption
Delta=(production_df.sum(axis=1) - areaConsumption.areaConsumption);
abs(Delta).max()
#endregion

#region III Ramp+Storage single area : loading parameters
Zones="FR"
year=2013

Selected_TECHNOLOGIES=['OldNuke','WindOnShore', 'CCG',"curtailment",'HydroRiver', 'HydroReservoir',"Solar"] ## try adding 'HydroRiver', 'HydroReservoir'

#### reading CSV files
areaConsumption = pd.read_csv(InputConsumptionFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["Date"])
availabilityFactor = pd.read_csv(InputProductionFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["Date","TECHNOLOGIES"])
TechParameters = pd.read_csv(InputPlaningFolder+'Planing-RAMP1BIS_TECHNOLOGIES.csv',
                             sep=',',decimal='.',skiprows=0,comment="#").set_index(["TECHNOLOGIES"])
StorageParameters = pd.read_csv(InputPlaningFolder+'Planing-RAMP1_STOCK_TECHNO.csv',sep=',',decimal='.',skiprows=0).set_index(["STOCK_TECHNO"])

#### Selection of subset
availabilityFactor=availabilityFactor.loc[(slice(None),Selected_TECHNOLOGIES),:]
TechParameters=TechParameters.loc[Selected_TECHNOLOGIES,:]
#TechParameters.loc["CCG",'capacity']=100000 ## margin to make everything work
TechParameters.loc["CCG",'maxCapacity']=70000
TechParameters.loc["OldNuke",'maxCapacity']=35000
TechParameters.loc["OldNuke",'RampConstraintMoins']=0.03 ## a bit strong to put in light the effect
TechParameters.loc["OldNuke",'RampConstraintPlus']=0.03 ## a bit strong to put in light the effect
#endregion

#region III Ramp+Storage single area : solving and loading results
model = GetElectricSystemModel_PlaningSingleNode_withStorage(Parameters={"areaConsumption"      :   areaConsumption,
                                                   "availabilityFactor"   :   availabilityFactor,
                                                   "TechParameters"       :   TechParameters,
                                                 "StorageParameters": StorageParameters})

if solver in solverpath :  opt = SolverFactory(solver,executable=solverpath[solver])
else : opt = SolverFactory(solver)
results=opt.solve(model)
Variables = getVariables_panda_indexed(model)
Constraints = getConstraintsDual_panda(model)

production_df=Variables['energy'].pivot(index="Date",columns='TECHNOLOGIES', values='energy')
production_df.loc[:,'Storage'] = Variables['storageOut'].pivot(index='Date',columns='STOCK_TECHNO',values='storageOut').sum(axis=1)-Variables['storageIn'].pivot(index='Date',columns='STOCK_TECHNO',values='storageIn').sum(axis=1) ### put storage in the production time series
production_df.sum(axis=0)/10**6 ### energies produites TWh
production_df[production_df>0].sum(axis=0)/10**6 ### energies produites TWh
production_df.max(axis=0)/1000 ### Pmax en GW

fig=MyStackedPlotly(y_df=production_df, Conso=areaConsumption)
fig=fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline
#fig.show()
#endregion

#region IV Case Storage + CCG + PV + Wind + hydro (Ramp+Storage single area) : loading parameters
Zones="FR"
year=2013

Selected_TECHNOLOGIES=['CCG', 'WindOnShore','WindOffShore','Solar',"curtailment",'HydroRiver', 'HydroReservoir']

#### reading CSV files
areaConsumption = pd.read_csv(InputConsumptionFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["Date"])
availabilityFactor = pd.read_csv(InputProductionFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["Date","TECHNOLOGIES"])
TechParameters = pd.read_csv(InputPlaningFolder+'Planing-RAMP1BIS_TECHNOLOGIES.csv',
                             sep=',',decimal='.',skiprows=0,comment="#").set_index(["TECHNOLOGIES"])
StorageParameters = pd.read_csv(InputPlaningFolder+'Planing-RAMP1_STOCK_TECHNO.csv',
                                sep=',',decimal='.',skiprows=0).set_index(["STOCK_TECHNO"])
#### Selection of subset
availabilityFactor=availabilityFactor.loc[(slice(None),Selected_TECHNOLOGIES),:]
TechParameters=TechParameters.loc[Selected_TECHNOLOGIES,:]
#TechParameters.loc["CCG",'capacity']=100000 ## margin to make everything work
TechParameters.loc["CCG",'energyCost']=100
TechParameters.loc["CCG",'maxCapacity']=50000
TechParameters.loc["WindOnShore",'capacityCost']=120000 #€/MW/year - investment+O&M fixed cost
TechParameters.loc["Solar",'capacityCost']=65000 #€/MW/year
TechParameters.loc["CCG",'RampConstraintMoins']=0.4 ## a bit strong to put in light the effect
TechParameters.loc["CCG",'RampConstraintPlus']=0.4 ## a bit strong to put in light the effect
StorageParameters.loc["Battery1","p_max"]=10000 # this is not optimized - batteries
StorageParameters.loc["Battery2","p_max"]=7000 # this is not optimized - Pumped HS
StorageParameters.loc["Battery2","c_max"]=StorageParameters.loc["Battery2","p_max"]*20 # this is not optimized 20h of Pumped HS
#endregion

#region IV Case Storage + CCG + PV + Wind + hydro  (Ramp+Storage single area) : solving and loading results
model = GetElectricSystemModel_PlaningSingleNode_withStorage(Parameters={"areaConsumption"      :   areaConsumption,
                                                   "availabilityFactor"   :   availabilityFactor,
                                                   "TechParameters"       :   TechParameters,
                                                   "StorageParameters"   : StorageParameters})

if solver in solverpath :  opt = SolverFactory(solver,executable=solverpath[solver])
else : opt = SolverFactory(solver)
results=opt.solve(model)
Variables = getVariables_panda_indexed(model)
Constraints = getConstraintsDual_panda(model)

production_df=Variables['energy'].pivot(index="Date",columns='TECHNOLOGIES', values='energy')
production_df.loc[:,'Storage'] = Variables['storageOut'].pivot(index='Date',columns='STOCK_TECHNO',values='storageOut').sum(axis=1)-Variables['storageIn'].pivot(index='Date',columns='STOCK_TECHNO',values='storageIn').sum(axis=1) ### put storage in the production time series
production_df.sum(axis=0)/10**6 ### energies produites TWh
production_df[production_df>0].sum(axis=0)/10**6 ### energies produites TWh
production_df.max(axis=0)/1000 ### Pmax en GW

fig=MyStackedPlotly(y_df=production_df, Conso=areaConsumption)
fig=fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline
#fig.show()
#endregion

#region V - Simple single area +4 million EV +  demande side management +30TWh H2: loading parameters
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



TechParameters = pd.read_csv(InputPlaningFolder+'Planing-RAMP1BIS_TECHNOLOGIES.csv',sep=',',decimal='.',skiprows=0,comment="#").set_index(["TECHNOLOGIES"])
StorageParameters = pd.read_csv(InputPlaningFolder + 'Planing-RAMP1_STOCK_TECHNO.csv', sep=',', decimal='.',
                                skiprows=0).set_index(["STOCK_TECHNO"])
ConsoParameters = pd.read_csv(InputPlaningFolder + "Planing-Conso-FLEX_CONSUM.csv", sep=";").set_index(["FLEX_CONSUM"])
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


# endregion

#region V - Simple single area +4 million EV +  demande side management +30TWh H2: solving and loading results

# €/kW/an coût fixe additionnel pour un GW d'électrolyseur en plus en supposant que l'on construit
model =  Model_SingleNode_online_flex(Parameters={"areaConsumption"      :   areaConsumption,
                                           "availabilityFactor"   :   availabilityFactor,
                                           "TechParameters"       :   TechParameters,
                                           "StorageParameters"   : StorageParameters,
                                           "to_flexible_consumption" : to_flexible_consumption,
                                           "ConsoParameters" : ConsoParameters_,
                                           "labour_ratio": labour_ratio})


if solver in solverpath :  opt = SolverFactory(solver,executable=solverpath[solver])
else : opt = SolverFactory(solver)
results=opt.solve(model)
Variables = getVariables_panda_indexed(model)
Variables.keys()
Variables['increased_max_power'] ## on a ajouté X GW à ce qui existait.
print(extractCosts(Variables))
print(extractEnergyCapacity(Variables))
Constraints = getConstraintsDual_panda(model)

production_df=Variables['energy'].pivot(index="Date",columns='TECHNOLOGIES', values='energy')
production_df.loc[:,'Storage'] = Variables['storageOut'].pivot(index='Date',columns='STOCK_TECHNO',values='storageOut').sum(axis=1)-Variables['storageIn'].pivot(index='Date',columns='STOCK_TECHNO',values='storageIn').sum(axis=1) ### put storage in the production time series
production_df.sum(axis=0)/10**6 ### energies produites TWh
production_df[production_df>0].sum(axis=0)/10**6 ### energies produites TWh
production_df.max(axis=0)/1000 ### Pmax en GW


fig=MyStackedPlotly(y_df=production_df, Conso=areaConsumption)
fig=fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline

#endregion

