
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

if sys.platform != 'win32':
    myhost = os.uname()[1]
else : myhost = ""
if (myhost=="jupyter-sop"):
    ## for https://jupyter-sop.mines-paristech.fr/ users, you need to
    #  (1) run the following to loanch the license server
    if (os.system("/opt/mosek/9.2/tools/platform/linux64x86/bin/lmgrd -c /opt/mosek/9.2/tools/platform/linux64x86/bin/mosek.lic -l lmgrd.log")==0):
        os.system("/opt/mosek/9.2/tools/platform/linux64x86/bin/lmutil lmstat -c 27007@127.0.0.1 -a")
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
areaConsumption = pd.read_csv(InputFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',sep=',',decimal='.',skiprows=0).set_index(["TIMESTAMP"])
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',sep=',',decimal='.',skiprows=0).set_index(["TIMESTAMP","TECHNOLOGIES"])
TechParameters = pd.read_csv(InputFolder+'Gestion-Simple_TECHNOLOGIES.csv',sep=',',decimal='.',skiprows=0).set_index(["TECHNOLOGIES"])

#### Selection of subset
Selected_TECHNOLOGIES=['OldNuke','CCG'] #you can add technologies here
availabilityFactor=availabilityFactor.loc[(slice(None),Selected_TECHNOLOGIES),:]
TechParameters=TechParameters.loc[Selected_TECHNOLOGIES,:]
TechParameters.loc["CCG",'capacity']=100000 ## margin to make everything work
#TechParameters.loc["WindOnShore",'capacity']=117000
#TechParameters.loc["Solar",'capacity']=67000
#endregion

#region I - Simple single area  : Solving and loading results
model = GetElectricSystemModel_GestionSingleNode(areaConsumption,availabilityFactor,TechParameters)
if solver in solverpath :  opt = SolverFactory(solver,executable=solverpath[solver])
else : opt = SolverFactory(solver)
results=opt.solve(model)
## result analysis
Variables=getVariables_panda_indexed(model)
extractCosts(Variables)
extractEnergyCapacity(Variables)
#pour avoir la production en KWh de chaque moyen de prod chaque heure
### Check sum Prod = Consumption
production_df=Variables['energy'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='energy')
Delta=(production_df.sum(axis=1) - areaConsumption.areaConsumption);
#endregion

#region I - Simple single area  : visualisation and lagrange multipliers
### representation des résultats
TIMESTAMP_d=pd.date_range(start=str(year)+"-01-01 00:00:00",end=str(year)+"-12-31 23:00:00",   freq="1H")
production_df.index=TIMESTAMP_d; areaConsumption.index=TIMESTAMP_d;
fig=MyStackedPlotly(y_df=production_df,Conso = areaConsumption)
fig=fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline
#fig2.show()

#### lagrange multipliers
Constraints= getConstraintsDual_panda_indexed(model)

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
Selected_TECHNOLOGIES=['OldNuke', 'CCG',"curtailment"] #you'll add 'Solar' after
#### reading CSV files

areaConsumption = pd.read_csv(InputFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0).set_index(["TIMESTAMP"])
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0).set_index(["TIMESTAMP","TECHNOLOGIES"])
TechParameters = pd.read_csv(InputFolder+'Gestion-RAMP1_TECHNOLOGIES.csv',sep=',',decimal='.',skiprows=0).set_index(["TECHNOLOGIES"])

#### Selection of subset
availabilityFactor=availabilityFactor.loc[(slice(None),Selected_TECHNOLOGIES),:]
TechParameters=TechParameters.loc[Selected_TECHNOLOGIES,:]

TechParameters.loc["CCG",'capacity']=100000 ## margin to make everything work
TechParameters.loc["OldNuke",'RampConstraintMoins']=0.01 ## a bit strong to put in light the effect
TechParameters.loc["OldNuke",'RampConstraintPlus']=0.02 ## a bit strong to put in light the effect
#TechParameters.loc["WindOnShore","energyCost"]=0.001 ## a bit strong to put in light the effect

#endregion

#region II - Ramp Ctrs Single area : solving and loading results
model = GetElectricSystemModel_GestionSingleNode(areaConsumption,availabilityFactor,TechParameters)
opt = SolverFactory(solver)
results=opt.solve(model)
Variables=getVariables_panda_indexed(model)


#pour avoir la production en KWh de chaque moyen de prod chaque heure
production_df=Variables['energy'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='energy')
### Check sum Prod = Consumption
Delta=(production_df.sum(axis=1) - areaConsumption.areaConsumption);
abs(Delta).max()

print(production_df.sum(axis=0)/10**6) ### energies produites TWh
print(Variables['energyCosts']) #pour avoir le coût de chaque moyen de prod à l'année
#endregion

#region II - Ramp Ctrs Single area : visualisation and lagrange multipliers


### representation des résultats
TIMESTAMP_d=pd.date_range(start=str(year)+"-01-01 00:00:00",end=str(year)+"-12-31 23:00:00",   freq="1H")
production_df.index=TIMESTAMP_d; areaConsumption.index=TIMESTAMP_d;
fig=MyStackedPlotly(y_df=production_df,Conso = areaConsumption)
fig=fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline
#fig2.show()

#### lagrange multipliers
Constraints= getConstraintsDual_panda_indexed(model)

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
Selected_TECHNOLOGIES=['OldNuke', 'CCG','WindOnShore',"curtailment"] #you'll add 'Solar' after #'NewNuke', 'HydroRiver', 'HydroReservoir','WindOnShore', 'WindOffShore', 'Solar', 'Curtailement'}

#### reading CSV files
TechParameters = pd.read_csv(InputFolder+'Gestion_MultiNode_DE-FR_AREAS_TECHNOLOGIES.csv',sep=',',decimal='.',comment="#",skiprows=0).set_index(["AREAS","TECHNOLOGIES"])
areaConsumption = pd.read_csv(InputFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0).set_index(["AREAS","TIMESTAMP"])
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0).set_index(["AREAS","TIMESTAMP","TECHNOLOGIES"])

ExchangeParameters = pd.read_csv(InputFolder+'Hypothese_DE-FR_AREAS_AREAS.csv',sep=',',decimal='.',skiprows=0,comment="#").set_index(["AREAS","AREAS.1"])
#ExchangeParameters.loc[ExchangeParameters.AREAS=="FR",'maxExchangeCapacity']=90000 ## margin to make everything work
#ExchangeParameters.loc[ExchangeParameters.AREAS=="DE",'maxExchangeCapacity']=90000 ## margin to make everything work
#### Selection of subset
TechParameters=TechParameters.loc[(Selected_AREAS,Selected_TECHNOLOGIES),:]
areaConsumption=areaConsumption.loc[(Selected_AREAS,slice(None)),:]
availabilityFactor=availabilityFactor.loc[(Selected_AREAS,slice(None),Selected_TECHNOLOGIES),:]
TechParameters.loc[(slice(None),'CCG'),'capacity']=100000 ## margin to make everything work
TechParameters.loc[(slice(None),"OldNuke"),'RampConstraintMoins']=0.01 ## a bit strong to put in light the effect
TechParameters.loc[(slice(None),"OldNuke"),'RampConstraintPlus']=0.02 ## a bit strong to put in light the effect
#endregion

#region III - Ramp Ctrs multiple area : solving and loading results
### small data cleaning
availabilityFactor.availabilityFactor[availabilityFactor.availabilityFactor>1]=1
model = GetElectricSystemModel_GestionMultiNode(areaConsumption,availabilityFactor,TechParameters,ExchangeParameters)
opt = SolverFactory(solver)
results=opt.solve(model)
Variables=getVariables_panda_indexed(model)
production_df=EnergyAndExchange2Prod(Variables)

### Check sum Prod = Consumption
Delta= production_df.sum(axis=1)-areaConsumption.areaConsumption
abs(Delta).sum()

## adding dates
production_df_=ChangeTIMESTAMP2Dates(production_df,year)
areaConsumption_=ChangeTIMESTAMP2Dates(areaConsumption,year)
fig=MyAreaStackedPlot(production_df_,Conso=areaConsumption_)
fig=fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
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

Selected_TECHNOLOGIES=['OldNuke','WindOnShore', 'CCG',"curtailment"]

#### reading CSV files
areaConsumption = pd.read_csv(InputFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0).set_index(["TIMESTAMP"])
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0).set_index(["TIMESTAMP","TECHNOLOGIES"])
TechParameters = pd.read_csv(InputFolder+'Gestion-RAMP1_TECHNOLOGIES.csv',sep=',',decimal='.',skiprows=0).set_index(["TECHNOLOGIES"])

#### Selection of subset
availabilityFactor=availabilityFactor.loc[(slice(None),Selected_TECHNOLOGIES),:]
TechParameters=TechParameters.loc[Selected_TECHNOLOGIES,:]
TechParameters.loc["CCG",'capacity']=100000 ## margin to make everything work
TechParameters.loc["OldNuke",'RampConstraintMoins']=0.02 ## a bit strong to put in light the effect
TechParameters.loc["OldNuke",'RampConstraintPlus']=0.02 ## a bit strong to put in light the effect
p_max=5000
StorageParameters={"p_max" : p_max , "c_max": p_max*10,"efficiency_in": 0.9,"efficiency_out" : 0.9}
#endregion

#region IV Ramp+Storage single area : solving and loading results
res= GetElectricSystemModel_GestionSingleNode_with1Storage(areaConsumption,availabilityFactor,
                                                      TechParameters,StorageParameters)

Variables = getVariables_panda_indexed(res['model'])
Constraints = getConstraintsDual_panda(res['model'])
areaConsumption = res["areaConsumption"]

production_df=Variables['energy'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='energy')
Delta= production_df.sum(axis=1)-areaConsumption["NewConsumption"]
sum(abs(Delta))
production_df.loc[:,'Storage'] = -areaConsumption["Storage"] ### put storage in the production time series
production_df.sum(axis=0)/10**6 ### energies produites TWh
production_df[production_df>0].sum(axis=0)/10**6 ### energies produites TWh
production_df.max(axis=0)/1000 ### Pmax en GW

TIMESTAMP_d=pd.date_range(start=str(year)+"-01-01 00:00:00",end=str(year)+"-12-31 23:00:00",   freq="1H")
production_df.index=TIMESTAMP_d; areaConsumption.index=TIMESTAMP_d;
fig=MyStackedPlotly(y_df=production_df, Conso=areaConsumption)
fig=fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline
stats=res["stats"]

#endregion

#region V Ramp+Storage Multi area : loading parameters
Zones="FR_DE_GB_ES"
year=2016
Selected_AREAS=["FR","DE"]
Selected_TECHNOLOGIES=['OldNuke', 'CCG','WindOnShore',"curtailment"] #you'll add 'Solar' after #'NewNuke', 'HydroRiver', 'HydroReservoir','WindOnShore', 'WindOffShore', 'Solar', 'Curtailement'}

#### reading CSV files
TechParameters = pd.read_csv(InputFolder+'Gestion_MultiNode_DE-FR_AREAS_TECHNOLOGIES.csv',
                             sep=',',decimal='.',comment="#",skiprows=0).set_index(["AREAS","TECHNOLOGIES"])
areaConsumption = pd.read_csv(InputFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0).set_index(["AREAS","TIMESTAMP"])
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0).set_index(["AREAS","TIMESTAMP","TECHNOLOGIES"])

ExchangeParameters = pd.read_csv(InputFolder+'Hypothese_DE-FR_AREAS_AREAS.csv',sep=',',decimal='.',skiprows=0,comment="#").set_index(["AREAS","AREAS.1"])
#ExchangeParameters.loc[ExchangeParameters.AREAS=="FR",'maxExchangeCapacity']=90000 ## margin to make everything work
#ExchangeParameters.loc[ExchangeParameters.AREAS=="DE",'maxExchangeCapacity']=90000 ## margin to make everything work
#### Selection of subset
TechParameters=TechParameters.loc[(Selected_AREAS,Selected_TECHNOLOGIES),:]
areaConsumption=areaConsumption.loc[(Selected_AREAS,slice(None)),:]
availabilityFactor=availabilityFactor.loc[(Selected_AREAS,slice(None),Selected_TECHNOLOGIES),:]
TechParameters.loc[(slice(None),'CCG'),'capacity']=100000 ## margin to make everything work
TechParameters.loc[(slice(None),"OldNuke"),'RampConstraintMoins']=0.01 ## a bit strong to put in light the effect
TechParameters.loc[(slice(None),"OldNuke"),'RampConstraintPlus']=0.02 ## a bit strong to put in light the effect
p_max=10000

StorageParameters=pd.DataFrame([])
for AREA in Selected_AREAS :
    StorageParameters_ = {"AREA": AREA, "p_max": p_max, "c_max": p_max * 10, "efficiency_in": 0.9,
                          "efficiency_out": 0.9}
    StorageParameters=StorageParameters.append(pd.DataFrame([StorageParameters_]))
StorageParameters=StorageParameters.set_index("AREA")
#endregion

#region V Ramp+Storage multi area : solving and loading results
res= GetElectricSystemModel_GestionMultiNode_with1Storage(areaConsumption,availabilityFactor,
                                                      TechParameters,ExchangeParameters,StorageParameters)

Variables = getVariables_panda(res['model'])
production_df=EnergyAndExchange2Prod(Variables)
areaConsumption = res["areaConsumption"]

### Check sum Prod = Consumption
Delta=(production_df.sum(axis=1) - areaConsumption.NewConsumption); ## comparaison à la conso incluant le stockage
abs(Delta).max()
production_df.loc[:,'Storage'] = -areaConsumption["Storage"] #### ajout du stockage comme production
Delta=(production_df.sum(axis=1) - areaConsumption.areaConsumption);
abs(Delta).max()


production_df_=ChangeTIMESTAMP2Dates(production_df,year)
areaConsumption_=ChangeTIMESTAMP2Dates(areaConsumption,year)


fig=MyAreaStackedPlot(production_df_,Conso=areaConsumption_)
fig=fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline

abs(areaConsumption["Storage"]).groupby(by="AREAS").sum() ## stockage
production_df.groupby(by="AREAS").sum()/10**6 ### energies produites TWh
production_df[production_df>0].groupby(by="AREAS").sum()/10**6 ### energies produites TWh
production_df.groupby(by="AREAS").max()/1000 ### Pmax en GW ### le stockage ne fait rien en Allemagne ??? bizarre
production_df.groupby(by="AREAS").min()/1000 ### Pmax en GW
#endregion

#region VI Complete "simple" France loading parameters
Zones="FR"
year=2013

Selected_TECHNOLOGIES=['OldNuke','Coal','CCG','TAC', 'WindOnShore','HydroReservoir','HydroRiver','Solar','curtailment']

#### reading CSV files
areaConsumption = pd.read_csv(InputFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0).set_index(["TIMESTAMP"])
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0).set_index(["TIMESTAMP","TECHNOLOGIES"])
TechParameters = pd.read_csv(InputFolder+'Gestion-Simple_TECHNOLOGIES.csv',sep=',',decimal='.',skiprows=0).set_index(["TECHNOLOGIES"])

#### Selection of subset
availabilityFactor=availabilityFactor.loc[(slice(None),Selected_TECHNOLOGIES),:]
TechParameters=TechParameters.loc[Selected_TECHNOLOGIES,:]
#TechParameters.loc[TechParameters.TECHNOLOGIES=="CCG",'capacity']=15000 ## margin to make everything work
p_max=5000
StorageParameters={"p_max" : p_max , "c_max": p_max*10,"efficiency_in": 0.9,"efficiency_out" : 0.9}

#endregion

#region VI Complete "simple" France : solving and loading results
res= GetElectricSystemModel_GestionSingleNode_with1Storage(areaConsumption,availabilityFactor,
                                                      TechParameters,StorageParameters)

Variables = getVariables_panda_indexed(res['model'])
Constraints = getConstraintsDual_panda_indexed(res['model'])
areaConsumption = res["areaConsumption"]

production_df=Variables['energy'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='energy')
Delta= production_df.sum(axis=1)-areaConsumption["NewConsumption"]
sum(abs(Delta))
production_df.loc[:,'Storage'] = -areaConsumption["Storage"] ### put storage in the production time series
production_df.sum(axis=0)/10**6 ### energies produites TWh
production_df[production_df>0].sum(axis=0)/10**6 ### energies produites TWh
production_df.max(axis=0)/1000 ### Pmax en GW

TIMESTAMP_d=pd.date_range(start=str(year)+"-01-01 00:00:00",end=str(year)+"-12-31 23:00:00",   freq="1H")
production_df.index=TIMESTAMP_d; areaConsumption.index=TIMESTAMP_d;
fig=MyStackedPlotly(y_df=production_df, Conso=areaConsumption)
fig=fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline
stats=res["stats"]
#endregion