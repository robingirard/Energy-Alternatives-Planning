
#region importation of modules
import os
myhost = os.uname()[1]
if (myhost=="jupyter-sop"):
    ## for https://jupyter-sop.mines-paristech.fr/ users, you need to
    #  (1) run the following in a terminal
    os.system("/opt/mosek/9.2/tools/platform/linux64x86/bin/lmgrd -c /opt/mosek/9.2/tools/platform/linux64x86/bin/mosek.lic -l lmgrd.log")
    #  (2) definition of license
    os.environ["MOSEKLM_LICENSE_FILE"] = '@jupyter-sop'

import numpy as np
import pandas as pd
import csv
#import docplex
import datetime
import copy
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn import linear_model
import sys

from functions.f_planingModels import *
from functions.f_optimization import *
from functions.f_graphicalTools import *
# Change this if you have other solvers obtained here
## https://ampl.com/products/solvers/open-source/
## for eduction this site provides also several professional solvers, that are more efficient than e.g. cbc
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

#region I - Simple single area : loading parameters
Zones="FR" ; year=2013
#### reading areaConsumption availabilityFactor and TechParameters CSV files
areaConsumption = pd.read_csv(InputFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',sep=',',decimal='.',skiprows=0)
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',sep=',',decimal='.',skiprows=0)
TechParameters = pd.read_csv(InputFolder+'Planing-Simple_TECHNOLOGIES.csv',sep=',',decimal='.',skiprows=0,comment="#")

#### Selection of subset
Selected_TECHNOLOGIES=['OldNuke','CCG'] #you can add technologies here
availabilityFactor=availabilityFactor[ availabilityFactor.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]
TechParameters=TechParameters[TechParameters.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]
#TechParameters.loc[TechParameters.TECHNOLOGIES=="OldNuke",'maxCapacity']=63000 ## Limit to actual installed capacity
#endregion

#region I - Simple single area  : Solving and loading results
model = GetElectricSystemModel_PlaningSingleNode(areaConsumption,availabilityFactor,TechParameters)

if solver in solverpath :  opt = SolverFactory(solver,executable=solverpath[solver])
else : opt = SolverFactory(solver)
results=opt.solve(model)
## result analysis
Variables=getVariables_panda(model)
Variables['capacity'] ## en ordre de grandeur, on installe juste un peu plus de nucléaire
#pour avoir la production en KWh de chaque moyen de prod chaque heure
production_df=Variables['energy'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='energy')
production_df.sum(axis=0)/10**6 ### energies produites TWh

energyCosts_df=Variables['energyCosts'].set_index("TECHNOLOGIES")
capacityCosts_df=Variables['capacityCosts'].set_index("TECHNOLOGIES")
costs_df=pd.concat([energyCosts_df,capacityCosts_df],axis=1)
costs_df["totalCosts"]=costs_df["energyCosts"]+costs_df["capacityCosts"]

#endregion

#region I - Simple single area  : visualisation and lagrange multipliers
### representation des résultats
fig=MyStackedPlotly(x_df=production_df.index,
                    y_df=production_df[list(Selected_TECHNOLOGIES)],
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
CapacityCtrDual=Constraints['CapacityCtr'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='CapacityCtr')*1000000;
round(CapacityCtrDual,2)
round(CapacityCtrDual.OldNuke,2).unique() ## if you increase by Delta the installed capacity of nuke you decrease by xxx the cost when nuke is not sufficient
round(CapacityCtrDual.CCG,2).unique() ## increasing the capacity of CCG as no effect on prices
#endregion

#region II - Ramp Single area : loading parameters loading parameterscase with ramp constraints
Zones="FR"
year=2013
Selected_TECHNOLOGIES=['OldNuke','CCG'] #you'll add 'Solar' ,'WindOnShore' after
#### reading CSV files
areaConsumption = pd.read_csv(InputFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0)
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0)
TechParameters = pd.read_csv(InputFolder+'Planing-RAMP1_TECHNOLOGIES.csv',sep=',',decimal='.',skiprows=0,comment="#")

#### Selection of subset
availabilityFactor=availabilityFactor[availabilityFactor.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]
TechParameters=TechParameters[TechParameters.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]
#endregion

#region II - Ramp Single area : solving and loading results
model = GetElectricSystemModel_PlaningSingleNode(areaConsumption,availabilityFactor,TechParameters)
opt = SolverFactory(solver)
results=opt.solve(model)
Variables=getVariables_panda(model)

#pour avoir la production en KWh de chaque moyen de prod chaque heure
production_df=Variables['energy'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='energy')
production_df.sum(axis=0)/10**6 ### energies produites TWh
print(Variables['energyCosts']) #pour avoir le coût de chaque moyen de prod à l'année
#endregion

#region II - Ramp Single area : visualisation and lagrange multipliers
fig=MyStackedPlotly(x_df=production_df.index,
                    y_df=production_df[list(Selected_TECHNOLOGIES)],
                    Names=list(Selected_TECHNOLOGIES))
fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline
#fig2.show()

#### lagrange multipliers
Constraints= getConstraintsDual_panda(model)

# Analyse energyCtr
energyCtrDual=Constraints['energyCtr']; energyCtrDual['energyCtr']=energyCtrDual['energyCtr']*1000000
energyCtrDual
round(energyCtrDual.energyCtr,2).unique()

# Analyse CapacityCtr
CapacityCtrDual=Constraints['CapacityCtr'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='CapacityCtr')*1000000;
round(CapacityCtrDual,2)
round(CapacityCtrDual.OldNuke,2).unique() ## if you increase by Delta the installed capacity of nuke you decrease by xxx the cost when nuke is not sufficient
round(CapacityCtrDual.CCG,2).unique() ## increasing the capacity of CCG as no effect on prices
#endregion

#region III - Ramp multiple area : loading parameters
Zones="FR_DE_GB_ES"
year=2016
Selected_AREAS=["FR","DE"]
Selected_TECHNOLOGIES=['OldNuke','CCG']#'NewNuke', 'HydroRiver', 'HydroReservoir','WindOnShore', 'WindOffShore', 'Solar', 'Curtailement'}
#Selected_TECHNOLOGIES={'CCG', 'OldNuke', 'NewNuke', 'HydroRiver', 'HydroReservoir',
#       'WindOnShore', 'WindOffShore', 'Solar', 'Curtailement'}
#### reading CSV files
areaConsumption = pd.read_csv(InputFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0)
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0)
TechParameters = pd.read_csv(InputFolder+'Planing_MultiNode_DE-FR_TECHNOLOGIES_AREAS.csv',sep=',',decimal='.',skiprows=0,comment="#")

ExchangeParameters = pd.read_csv(InputFolder+'Hypothese_DE-FR_AREAS_AREAS.csv',sep=',',decimal='.',skiprows=0,comment="#")
#### Selection of subset
TechParameters=TechParameters[TechParameters.AREAS.isin(Selected_AREAS)&TechParameters.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]
areaConsumption=areaConsumption[areaConsumption.AREAS.isin(Selected_AREAS)]
availabilityFactor=availabilityFactor[availabilityFactor.AREAS.isin(Selected_AREAS)& availabilityFactor.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]
#endregion

#region III - Ramp multiple area : solving and loading results
### small data cleaning
availabilityFactor.availabilityFactor[availabilityFactor.availabilityFactor>1]=1
model = GetElectricSystemModel_PlaningMultiNode(areaConsumption,availabilityFactor,TechParameters,ExchangeParameters)
opt = SolverFactory(solver)
results=opt.solve(model)
Variables=getVariables_panda(model)
Variables['capacity']
production_df=EnergyAndExchange2Prod(Variables)
fig=MyAreaStackedPlot(production_df,Selected_TECHNOLOGIES=Selected_TECHNOLOGIES+Selected_AREAS)
fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline

production_df=Variables['energy'].pivot(index=["TIMESTAMP","AREAS"], columns='TECHNOLOGIES', values='energy')
production_df.sum(axis=0)/10**6 ### energies produites TWh
production_df.groupby(by="AREAS").sum()/10**6 ### energies produites TWh

energyCosts_df=Variables['energyCosts'].set_index(["AREAS","TECHNOLOGIES"])
capacityCosts_df=Variables['capacityCosts'].set_index(["AREAS","TECHNOLOGIES"])
costs_df=pd.concat([energyCosts_df,capacityCosts_df],axis=1)
costs_df["totalCosts"]=costs_df["energyCosts"]+costs_df["capacityCosts"]
costs_df.groupby("AREAS")["totalCosts"].sum()

Variables["exchange"].head()
Variables["exchange"].exchange.sum()
Constraints= getConstraintsDual_panda(model)
Constraints.keys()
#endregion

#region IV Ramp+Storage single area : loading parameters
Zones="FR"
year=2013

Selected_TECHNOLOGIES={'CCG', 'OldNuke', 'WindOnShore','curtailment'}

#### reading CSV files
areaConsumption = pd.read_csv(InputFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0)
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0)
TechParameters = pd.read_csv(InputFolder+'Planing-RAMP1_TECHNOLOGIES.csv',sep=',',decimal='.',skiprows=0,comment="#")


#### Selection of subset
availabilityFactor=availabilityFactor[ availabilityFactor.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]
TechParameters=TechParameters[TechParameters.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]

p_max=10000
StorageParameters={"p_max" : p_max , "c_max": p_max*10,"efficiency_in": 0.9,"efficiency_out" : 0.9}
#endregion

#region IV Ramp+Storage single area : solving and loading results
res= GetElectricSystemModel_PlaningSingleNode_with1Storage(areaConsumption,availabilityFactor,
                                                      TechParameters,StorageParameters)

Variables = getVariables_panda(res['model'])
Constraints = getConstraintsDual_panda(res['model'])
areaConsumption = res["areaConsumption"]

production_df=Variables['energy'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='energy')
production_df.loc[:,'Storage'] = areaConsumption["Storage"]### put storage in the production time series
production_df.sum(axis=0)/10**6 ### energies produites TWh
production_df[production_df>0].sum(axis=0)/10**6 ### energies produites TWh
production_df.max(axis=0)/1000 ### Pmax en GW

Selected_TECHNOLOGIES_Sto=list(Selected_TECHNOLOGIES)
Selected_TECHNOLOGIES_Sto.append("Storage")
fig=MyStackedPlotly(x_df=production_df.index,
                    y_df=production_df[Selected_TECHNOLOGIES_Sto],
                    Names=Selected_TECHNOLOGIES_Sto)
fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline
stats=res["stats"]
#endregion

#region V Case Storage + CCG + PV + Wind (Ramp+Storage single area) : loading parameters
Zones="FR"
year=2013

Selected_TECHNOLOGIES={'CCG', 'WindOnShore','Solar',"curtailment"}

#### reading CSV files
areaConsumption = pd.read_csv(InputFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0)
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0)
TechParameters = pd.read_csv(InputFolder+'Planing-RAMP1_TECHNOLOGIES.csv',sep=',',decimal='.',skiprows=0,comment="#")


#### Selection of subset
availabilityFactor=availabilityFactor[ availabilityFactor.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]
TechParameters=TechParameters[TechParameters.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]
TechParameters.keys()
TechParameters.loc[TechParameters.TECHNOLOGIES=='Solar','minCapacity']=90000
TechParameters.loc[TechParameters.TECHNOLOGIES=='WindOnShore','minCapacity']=12000
TechParameters[['TECHNOLOGIES','minCapacity','maxCapacity']]
p_max=10000 ## storage capacity is not optimized
StorageParameters={"p_max" : p_max , "c_max": p_max*10,"efficiency_in": 0.9,"efficiency_out" : 0.9}
#endregion

#region V Case Storage + CCG + PV + Wind (Ramp+Storage single area) : solving and loading results
res= GetElectricSystemModel_PlaningSingleNode_with1Storage(areaConsumption,availabilityFactor,
                                                      TechParameters,StorageParameters)
Variables = getVariables_panda(res['model'])
Constraints = getConstraintsDual_panda(res['model'])
areaConsumption = res["areaConsumption"]

production_df=Variables['energy'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='energy')
production_df.loc[:,'Storage'] = areaConsumption["Storage"]### put storage in the production time series
production_df[production_df>0].sum(axis=0)/10**6 ### energies produites TWh
production_df.max(axis=0)/1000 ### Pmax en GW
Variables['capacity'] ### capacity en MW


Selected_TECHNOLOGIES_Sto=list(Selected_TECHNOLOGIES)
Selected_TECHNOLOGIES_Sto.append("Storage")
fig=MyStackedPlotly(x_df=production_df.index,
                    y_df=production_df[Selected_TECHNOLOGIES_Sto],
                    Names=Selected_TECHNOLOGIES_Sto)
fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline
stats=res["stats"]
#endregion

#region VI Case Storage + CCG + Nuke (Ramp+Storage single area) : loading parameters
Zones="FR"
year=2013

Selected_TECHNOLOGIES={'CCG', 'NewNuke',"curtailment"}

#### reading CSV files
areaConsumption = pd.read_csv(InputFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0)
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0)
TechParameters = pd.read_csv(InputFolder+'Planing-RAMP1_TECHNOLOGIES.csv',sep=',',decimal='.',skiprows=0,comment="#")


#### Selection of subset
availabilityFactor=availabilityFactor[ availabilityFactor.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]
TechParameters=TechParameters[TechParameters.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]
TechParameters.keys()
TechParameters.minCapacity[TechParameters.TECHNOLOGIES=='NewNuke']=63000
TechParameters.maxCapacity[TechParameters.TECHNOLOGIES=='NewNuke']=120000
TechParameters[['TECHNOLOGIES','minCapacity','maxCapacity']]
p_max=10000
StorageParameters={"p_max" : p_max , "c_max": p_max*10,"efficiency_in": 0.9,"efficiency_out" : 0.9}
#endregion

#region VI Case Storage + CCG + Nuke (Ramp+Storage single area) : solving and loading results
res= GetElectricSystemModel_PlaningSingleNode_with1Storage(areaConsumption,availabilityFactor,
                                                      TechParameters,StorageParameters)
Variables = getVariables_panda(res['model'])
Constraints = getConstraintsDual_panda(res['model'])
areaConsumption = res["areaConsumption"]

production_df=Variables['energy'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='energy')
production_df.loc[:,'Storage'] = areaConsumption["Storage"]### put storage in the production time series
production_df[production_df>0].sum(axis=0)/10**6 ### energies produites TWh
production_df.max(axis=0)/1000 ### Pmax en GW
Variables['capacity'] ### capacity en MW


Selected_TECHNOLOGIES_Sto=list(Selected_TECHNOLOGIES)
Selected_TECHNOLOGIES_Sto.append("Storage")
fig=MyStackedPlotly(x_df=production_df.index,
                    y_df=production_df[Selected_TECHNOLOGIES_Sto],
                    Names=Selected_TECHNOLOGIES_Sto)
fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline
stats=res["stats"]
#endregion

#region VI Ramp+Storage Multi area : loading parameters
Zones="FR_DE_GB_ES"
year=2016
Selected_AREAS={"FR","DE"}
Selected_TECHNOLOGIES={'CCG', 'OldNuke' } #'NewNuke', 'HydroRiver', 'HydroReservoir','WindOnShore', 'WindOffShore', 'Solar', 'Curtailement'}

#### reading CSV files
TechParameters = pd.read_csv(InputFolder+'Planing_MultiNode_DE-FR_TECHNOLOGIES_AREAS.csv',sep=',',decimal='.',comment="#")
areaConsumption = pd.read_csv(InputFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0)
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0)

ExchangeParameters = pd.read_csv(InputFolder+'Hypothese_DE-FR_AREAS_AREAS.csv',sep=',',decimal='.',skiprows=0,comment="#")
#### Selection of subset
TechParameters=TechParameters[TechParameters.AREAS.isin(Selected_AREAS)&TechParameters.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]
areaConsumption=areaConsumption[areaConsumption.AREAS.isin(Selected_AREAS)]
availabilityFactor=availabilityFactor[availabilityFactor.AREAS.isin(Selected_AREAS)& availabilityFactor.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]

p_max=10000

StorageParameters=pd.DataFrame([])
for AREA in Selected_AREAS :
    StorageParameters_ = {"AREA": AREA, "p_max": p_max, "c_max": p_max * 10, "efficiency_in": 0.9,
                          "efficiency_out": 0.9}
    StorageParameters=StorageParameters.append(pd.DataFrame([StorageParameters_]))

#endregion

#region VI Ramp+Storage multi area : solving and loading results
res= GetElectricSystemModel_PlaningMultiNode_with1Storage(areaConsumption,availabilityFactor,
                                                      TechParameters,ExchangeParameters,StorageParameters)

Variables = getVariables_panda(res['model'])
Constraints = getConstraintsDual_panda(res['model'])
areaConsumption = res["areaConsumption"]
#Variables['energy'].loc['Storage'] = areaConsumption["Storage"]

#areaConsumption['Storage'].range()
model= res["model"]
stats=res["stats"]
#endregion
