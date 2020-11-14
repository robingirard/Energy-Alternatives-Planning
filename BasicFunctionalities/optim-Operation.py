
#region importation of modules
import os

import numpy as np
import pandas as pd
import csv

import datetime
import copy
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn import linear_model
import sys

from functions.f_operationModels import *
from functions.f_optimization import *
from functions.f_graphicalTools import *
#endregion

#region Solver and data location definition

InputFolder='Data/input/'

myhost = os.uname()[1]
if (myhost=="jupyter-sop"):
    ## for https://jupyter-sop.mines-paristech.fr/ users, you need to
    #  (1) run the following to loanch the license server
    os.system("/opt/mosek/9.2/tools/platform/linux64x86/bin/lmgrd -c /opt/mosek/9.2/tools/platform/linux64x86/bin/mosek.lic -l lmgrd.log")
    #  (2) definition of license
    os.environ["MOSEKLM_LICENSE_FILE"] = '@jupyter-sop'

BaseSolverPath='/Users/robin.girard/Documents/Code/Packages/solvers/ampl_macosx64' ### change this to the folder with knitro ampl ...
## in order to obtain more solver see see https://ampl.com/products/solvers/open-source/
## for eduction this site provides also several professional solvers, that are more efficient than e.g. cbc
sys.path.append(BaseSolverPath)
solvers= ['gurobi','knitro','cbc'] # try 'glpk', 'cplex'
solverpath= {}
for solver in solvers : solverpath[solver]=BaseSolverPath+'/'+solver
solver= 'mosek' ## no need for solverpath with mosek.
#endregion

#region I - Simple single area : loading parameters
Zones="FR" ; year=2013
#### reading areaConsumption availabilityFactor and TechParameters CSV files
areaConsumption = pd.read_csv(InputFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',sep=',',decimal='.',skiprows=0)
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',sep=',',decimal='.',skiprows=0)
TechParameters = pd.read_csv(InputFolder+'Gestion-Simple_TECHNOLOGIES.csv',sep=',',decimal='.',skiprows=0)

#### Selection of subset
Selected_TECHNOLOGIES=['OldNuke','CCG'] #you can add technologies here
availabilityFactor=availabilityFactor[ availabilityFactor.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]
TechParameters=TechParameters[TechParameters.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]
TechParameters.loc[TechParameters.TECHNOLOGIES=="CCG",'capacity']=100000 ## margin to make everything work
#TechParameters.loc[TechParameters.TECHNOLOGIES=="WindOnShore",'capacity']=117000
#TechParameters.loc[TechParameters.TECHNOLOGIES=="Solar",'capacity']=67000
#endregion

#region I - Simple single area  : Solving and loading results
model = GetElectricSystemModel_GestionSingleNode(areaConsumption,availabilityFactor,TechParameters)
if solver in solverpath :  opt = SolverFactory(solver,executable=solverpath[solver])
else : opt = SolverFactory(solver)
results=opt.solve(model)
## result analysis
Variables=getVariables_panda(model)



#pour avoir la production en KWh de chaque moyen de prod chaque heure
production_df=Variables['energy'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='energy')
### Check sum Prod = Consumption
areaConsumptionOut=areaConsumption.copy();     areaConsumptionOut.index += 1 ## index of input start at 0, index of output start at 1
Delta=(production_df.sum(axis=1) - areaConsumptionOut["areaConsumption"]);
abs(Delta).max()

print(production_df.sum(axis=0)/10**6) ### energies produites TWh
print(Variables['energyCosts']) #pour avoir le coût de chaque moyen de prod à l'année
#endregion

#region I - Simple single area  : visualisation and lagrange multipliers
### representation des résultats

fig=MyStackedPlotly(x_df=production_df.index,
                    y_df=production_df[list(Selected_TECHNOLOGIES)],
                    Conso = areaConsumptionOut,
                    Names=list(Selected_TECHNOLOGIES))
fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline
#fig2.show()

#### lagrange multipliers
Constraints= getConstraintsDual_panda(model)

# Analyse energyCtr
energyCtrDual=Constraints['energyCtr']; energyCtrDual['energyCtr']=energyCtrDual['energyCtr']
energyCtrDual
round(energyCtrDual.energyCtr,2).unique()

# Analyse CapacityCtr
CapacityCtrDual=Constraints['CapacityCtr'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='CapacityCtr');
round(CapacityCtrDual,2)
round(CapacityCtrDual.OldNuke,2).unique() ## if you increase by Delta the installed capacity of nuke you decrease by xxx the cost when nuke is not sufficient
round(CapacityCtrDual.CCG,2).unique() ## increasing the capacity of CCG as no effect on prices
#endregion

#region II - Ramp Ctrs Single area : loading parameters loading parameterscase with ramp constraints
Zones="FR"
year=2013
Selected_TECHNOLOGIES=['OldNuke','CCG'] #you'll add 'Solar' after
#### reading CSV files
areaConsumption = pd.read_csv(InputFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0)
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0)
TechParameters = pd.read_csv(InputFolder+'Gestion-RAMP1_TECHNOLOGIES.csv',sep=',',decimal='.',skiprows=0)

#### Selection of subset
availabilityFactor=availabilityFactor[availabilityFactor.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]
TechParameters=TechParameters[TechParameters.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]
TechParameters.loc[TechParameters.TECHNOLOGIES=="CCG",'capacity']=100000 ## margin to make everything work
TechParameters.loc[TechParameters.TECHNOLOGIES=="OldNuke",'RampConstraintMoins']=0.02 ## a bit strong to put in light the effect
TechParameters.loc[TechParameters.TECHNOLOGIES=="OldNuke",'RampConstraintPlus']=0.02 ## a bit strong to put in light the effect
#endregion

#region II - Ramp Ctrs Single area : solving and loading results
model = GetElectricSystemModel_GestionSingleNode(areaConsumption,availabilityFactor,TechParameters)
opt = SolverFactory(solver)
results=opt.solve(model)
Variables=getVariables_panda(model)

#pour avoir la production en KWh de chaque moyen de prod chaque heure
production_df=Variables['energy'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='energy')

### Check sum Prod = Consumption
areaConsumptionOut=areaConsumption.copy();     areaConsumptionOut.index += 1 ## index of input start at 0, index of output start at 1
Delta=(production_df.sum(axis=1) - areaConsumptionOut["areaConsumption"]);
abs(Delta).max()

production_df.sum(axis=0)/10**6 ### energies produites TWh
print(Variables['energyCosts']) #pour avoir le coût de chaque moyen de prod à l'année
#endregion

#region II - Ramp Ctrs Single area : visualisation and lagrange multipliers
fig=MyStackedPlotly(x_df=production_df.index,
                    y_df=production_df[list(Selected_TECHNOLOGIES)],
                    Conso=areaConsumption,
                    Names=list(Selected_TECHNOLOGIES))
fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline
#fig2.show()

#### lagrange multipliers
Constraints= getConstraintsDual_panda(model)

# Analyse energyCtr
energyCtrDual=Constraints['energyCtr']; energyCtrDual['energyCtr']=energyCtrDual['energyCtr']
energyCtrDual
round(energyCtrDual.energyCtr,2).unique()

# Analyse CapacityCtr
CapacityCtrDual=Constraints['CapacityCtr'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='CapacityCtr');
round(CapacityCtrDual,2)
round(CapacityCtrDual.OldNuke,2).unique() ## if you increase by Delta the installed capacity of nuke you decrease by xxx the cost when nuke is not sufficient
round(CapacityCtrDual.CCG,2).unique() ## increasing the capacity of CCG as no effect on prices
#endregion

#region III - Ramp Ctrs multiple area : loading parameters
Zones="FR_DE_GB_ES"
year=2016
Selected_AREAS=["FR","DE"]
Selected_TECHNOLOGIES=['OldNuke','CCG'] #'NewNuke', 'HydroRiver', 'HydroReservoir','WindOnShore', 'WindOffShore', 'Solar', 'Curtailement'}

#### reading CSV files
TechParameters = pd.read_csv(InputFolder+'Gestion_MultiNode_DE-FR_AREAS_TECHNOLOGIES.csv',sep=',',decimal='.',comment="#",skiprows=0)
areaConsumption = pd.read_csv(InputFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0)
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0)

ExchangeParameters = pd.read_csv(InputFolder+'Hypothese_DE-FR_AREAS_AREAS.csv',sep=',',decimal='.',skiprows=0,comment="#")
#ExchangeParameters.loc[ExchangeParameters.AREAS=="FR",'maxExchangeCapacity']=90000 ## margin to make everything work
#ExchangeParameters.loc[ExchangeParameters.AREAS=="DE",'maxExchangeCapacity']=90000 ## margin to make everything work
#### Selection of subset
TechParameters=TechParameters[TechParameters.AREAS.isin(Selected_AREAS)&TechParameters.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]
areaConsumption=areaConsumption[areaConsumption.AREAS.isin(Selected_AREAS)]
availabilityFactor=availabilityFactor[availabilityFactor.AREAS.isin(Selected_AREAS)& availabilityFactor.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]
TechParameters.loc[TechParameters.TECHNOLOGIES=="CCG",'capacity']=100000 ## margin to make everything work
#endregion

#region III - Ramp Ctrs multiple area : solving and loading results
### small data cleaning
availabilityFactor.availabilityFactor[availabilityFactor.availabilityFactor>1]=1
model = GetElectricSystemModel_GestionMultiNode(areaConsumption,availabilityFactor,TechParameters,ExchangeParameters)
opt = SolverFactory(solver)
results=opt.solve(model)
Variables=getVariables_panda(model)
production_df=EnergyAndExchange2Prod(Variables)

### Check sum Prod = Consumption
areaConsumptionOut=areaConsumption.copy();   #  areaConsumptionOut.index += 1 ## index of input start at 0, index of output start at 1
areaConsumptionOut=areaConsumptionOut.set_index(["TIMESTAMP","AREAS"])
Delta= production_df.sum(axis=1)-areaConsumptionOut.areaConsumption
abs(Delta).sum()
df_=production_df
fig=MyAreaStackedPlot(production_df,Conso=areaConsumptionOut)
fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline

production_df.sum(axis=0)/10**6 ### energies produites TWh
production_df.groupby(by="AREAS").sum()/10**6 ### energies produites TWh
production_df.groupby(by=["AREAS"]).max()

Constraints= getConstraintsDual_panda(model)
Constraints.keys()
Constraints['energyCtr']
#endregion

#region IV Ramp+Storage single area : loading parameters
Zones="FR"
year=2013

Selected_TECHNOLOGIES=['OldNuke','WindOnShore', 'CCG']

#### reading CSV files
areaConsumption = pd.read_csv(InputFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0)
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0)
TechParameters = pd.read_csv(InputFolder+'Gestion-Simple_TECHNOLOGIES.csv',sep=',',decimal='.',skiprows=0)

#### Selection of subset
availabilityFactor=availabilityFactor[ availabilityFactor.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]
TechParameters=TechParameters[TechParameters.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]
TechParameters.loc[TechParameters.TECHNOLOGIES=="CCG",'capacity']=100000 ## margin to make everything work
p_max=5000
StorageParameters={"p_max" : p_max , "c_max": p_max*10,"efficiency_in": 0.9,"efficiency_out" : 0.9}
#endregion

#region IV Ramp+Storage single area : solving and loading results
res= GetElectricSystemModel_GestionSingleNode_with1Storage(areaConsumption,availabilityFactor,
                                                      TechParameters,StorageParameters)

Variables = getVariables_panda(res['model'])
Constraints = getConstraintsDual_panda(res['model'])
areaConsumption = res["areaConsumption"]

production_df=Variables['energy'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='energy')
production_df.sum(axis=1)-areaConsumption["NewConsumption"]
production_df.loc[:,'Storage'] = -areaConsumption["Storage"] ### put storage in the production time series
production_df.sum(axis=0)/10**6 ### energies produites TWh
production_df[production_df>0].sum(axis=0)/10**6 ### energies produites TWh
production_df.max(axis=0)/1000 ### Pmax en GW

Selected_TECHNOLOGIES_Sto=list(Selected_TECHNOLOGIES)
Selected_TECHNOLOGIES_Sto.append("Storage")
fig=MyStackedPlotly(x_df=production_df.index,
                    y_df=production_df[Selected_TECHNOLOGIES_Sto],
                    Conso=areaConsumption,
                    Names=Selected_TECHNOLOGIES_Sto)
fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline
stats=res["stats"]

#endregion

#region V Ramp+Storage Multi area : loading parameters
Zones="FR_DE_GB_ES"
year=2016
Selected_AREAS=["FR","DE"]
Selected_TECHNOLOGIES=['OldNuke','CCG'] #'NewNuke', 'HydroRiver', 'HydroReservoir','WindOnShore', 'WindOffShore', 'Solar', 'Curtailement'}

#### reading CSV files
TechParameters = pd.read_csv(InputFolder+'Gestion_MultiNode_DE-FR_AREAS_TECHNOLOGIES.csv',sep=',',decimal='.',comment="#",skiprows=0)
areaConsumption = pd.read_csv(InputFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0)
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0)

ExchangeParameters = pd.read_csv(InputFolder+'Hypothese_DE-FR_AREAS_AREAS.csv',sep=',',decimal='.',skiprows=0,comment="#")
#### Selection of subset
TechParameters=TechParameters[TechParameters.AREAS.isin(Selected_AREAS)&TechParameters.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]
areaConsumption=areaConsumption[areaConsumption.AREAS.isin(Selected_AREAS)]
availabilityFactor=availabilityFactor[availabilityFactor.AREAS.isin(Selected_AREAS)& availabilityFactor.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]
TechParameters.loc[TechParameters.TECHNOLOGIES=="CCG",'capacity']=100000 ## margin to make everything work
p_max=10000

StorageParameters=pd.DataFrame([])
for AREA in Selected_AREAS :
    StorageParameters_ = {"AREA": AREA, "p_max": p_max, "c_max": p_max * 10, "efficiency_in": 0.9,
                          "efficiency_out": 0.9}
    StorageParameters=StorageParameters.append(pd.DataFrame([StorageParameters_]))

#endregion

#region V Ramp+Storage multi area : solving and loading results
res= GetElectricSystemModel_GestionMultiNode_with1Storage(areaConsumption,availabilityFactor,
                                                      TechParameters,ExchangeParameters,StorageParameters)

Variables = getVariables_panda(res['model'])
production_df=EnergyAndExchange2Prod(Variables)
areaConsumption = res["areaConsumption"]
areaConsumptionOut=areaConsumption.copy();   #  areaConsumptionOut.index += 1 ## index of input start at 0, index of output start at 1
if "TIMESTAMP" in areaConsumptionOut.columns :
    areaConsumptionOut=areaConsumptionOut.set_index(["TIMESTAMP","AREAS"])
else :
    areaConsumptionOut = areaConsumptionOut.reset_index().set_index([ "TIMESTAMP","AREAS"])

production_df.loc[:,'Storage'] = -areaConsumptionOut["Storage"]
abs(areaConsumption["Storage"]).groupby(by="AREAS").sum() ## stockage
production_df.groupby(by="AREAS").sum()/10**6 ### energies produites TWh
production_df[production_df>0].groupby(by="AREAS").sum()/10**6 ### energies produites TWh
production_df.groupby(by="AREAS").max()/1000 ### Pmax en GW ### le stockage ne fait rien en Allemagne ??? bizarre
production_df.groupby(by="AREAS").min()/1000 ### Pmax en GW

### Check sum Prod = Consumption
Delta= production_df.sum(axis=1)-areaConsumptionOut.areaConsumption
abs(Delta).sum()
df_=production_df
fig=MyAreaStackedPlot(production_df,Conso=areaConsumptionOut)
fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline
#endregion

#region VI Complete "simple" France loading parameters
Zones="FR"
year=2013

Selected_TECHNOLOGIES=['OldNuke','Coal','CCG','TAC', 'WindOnShore','HydroReservoir','HydroRiver','Solar','curtailment']

#### reading CSV files
areaConsumption = pd.read_csv(InputFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0)
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0)
TechParameters = pd.read_csv(InputFolder+'Gestion-Simple_TECHNOLOGIES.csv',sep=',',decimal='.',skiprows=0)
TechParameters.TECHNOLOGIES
#### Selection of subset
availabilityFactor=availabilityFactor[ availabilityFactor.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]
TechParameters=TechParameters[TechParameters.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]
#TechParameters.loc[TechParameters.TECHNOLOGIES=="CCG",'capacity']=15000 ## margin to make everything work
p_max=5000
StorageParameters={"p_max" : p_max , "c_max": p_max*30,"efficiency_in": 0.9,"efficiency_out" : 0.9}

#endregion

#region VI Complete "simple" France : solving and loading results
res= GetElectricSystemModel_GestionSingleNode_with1Storage(areaConsumption,availabilityFactor,
                                                      TechParameters,StorageParameters)

Variables = getVariables_panda(res['model'])
Constraints = getConstraintsDual_panda(res['model'])
areaConsumption = res["areaConsumption"]

production_df=Variables['energy'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='energy')
production_df.sum(axis=1)-areaConsumption["NewConsumption"]
production_df.loc[:,'Storage'] = -areaConsumption["Storage"] ### put storage in the production time series
production_df.sum(axis=0)/10**6 ### energies produites TWh
production_df[production_df>0].sum(axis=0)/10**6 ### energies produites TWh
production_df.max(axis=0)/1000 ### Pmax en GW

Selected_TECHNOLOGIES_Sto=list(Selected_TECHNOLOGIES)
Selected_TECHNOLOGIES_Sto.append("Storage")
fig=MyStackedPlotly(x_df=production_df.index,
                    y_df=production_df[Selected_TECHNOLOGIES_Sto],
                    Conso=areaConsumption,
                    Names=Selected_TECHNOLOGIES_Sto)
fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline
stats=res["stats"]
#endregion