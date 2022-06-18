
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
import sys

from functions.f_planingModels import *
from functions.f_tools import *
from functions.f_graphicalTools import *
from functions.f_consumptionModels import *
from functions.f_model_definition import *
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
areaConsumption = pd.read_csv(InputFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["Date"])
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["Date","TECHNOLOGIES"])


# if you want to change thermal sensitivity + add electric vehicle
ConsoTempe_df=pd.read_csv(InputFolder+'ConsumptionTemperature_1996TO2019_FR.csv',parse_dates=['Date']).set_index(["Date"]) #
ConsoTempe_df_nodup=ConsoTempe_df.loc[~ConsoTempe_df.index.duplicated(),:]
(ConsoTempeYear_decomposed_df,Thermosensibilite)=Decomposeconso(areaConsumption.join(ConsoTempe_df_nodup)[['areaConsumption','Temperature']],
                                                                TemperatureThreshold=15,ConsumptionName="areaConsumption")
VEProfile_df=pd.read_csv(InputFolder+'EVModel.csv', sep=';')
NbVE=4 # millions
VE_consumption = NbVE*Profile2Consumption(Profile_df=VEProfile_df,Temperature_df = ConsoTempe_df_nodup.loc[str(year)][['Temperature']])[['Consumption']]
areaConsumption = areaConsumption.assign(areaConsumption = ConsoTempeYear_decomposed_df.loc[:,'TS_C']+1*ConsoTempeYear_decomposed_df.loc[:,'NTS_C']+VE_consumption.loc[:,'Consumption'])





TechParameters = pd.read_csv(InputFolder+'Planing-Simple_TECHNOLOGIES.csv',sep=',',decimal='.',skiprows=0,comment="#").set_index(["TECHNOLOGIES"])
TechParameters.head()
#### Selection of subset
Selected_TECHNOLOGIES=['OldNuke','CCG'] #you can add technologies here
availabilityFactor=availabilityFactor.loc[(slice(None),Selected_TECHNOLOGIES),:]
TechParameters=TechParameters.loc[Selected_TECHNOLOGIES,:]
#TechParameters.loc[TechParameters.TECHNOLOGIES=="OldNuke",'maxCapacity']=63000 ## Limit to actual installed capacity
#endregion

#region I - Simple single area  : Solving and loading results



model = GetElectricSystemModel_Planing(Parameters={"areaConsumption"      :   areaConsumption,
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
plotly.offline.plot(fig, filename='file.html') ## offline
#fig2.show()

#### lagrange multipliers
Constraints= getConstraintsDual_panda(model)

# Analyse energyCtr
energyCtrDual=Constraints['energyCtr']; energyCtrDual['energyCtr']=energyCtrDual['energyCtr']
energyCtrDual
round(energyCtrDual.energyCtr,2).unique()

# Analyse CapacityCtr
CapacityCtrDual=Constraints['CapacityCtr'].pivot(index="Date",columns='TECHNOLOGIES', values='CapacityCtr')*1000000;
round(CapacityCtrDual,2)
round(CapacityCtrDual.OldNuke,2).unique() ## if you increase by Delta the installed capacity of nuke you decrease by xxx the cost when nuke is not sufficient
round(CapacityCtrDual.CCG,2).unique() ## increasing the capacity of CCG as no effect on prices
#endregion

#region II - Ramp Single area : loading parameters loading parameterscase with ramp constraints
Zones="FR"
year=2013
Selected_TECHNOLOGIES=['OldNuke', 'CCG',"curtailment"] #you'll add 'Solar' after
#### reading CSV files
areaConsumption = pd.read_csv(InputFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["Date"])
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["Date","TECHNOLOGIES"])
TechParameters = pd.read_csv(InputFolder+'Planing-RAMP1BIS_TECHNOLOGIES.csv',sep=',',decimal='.',skiprows=0,comment="#").set_index(["TECHNOLOGIES"])

#### Selection of subset
availabilityFactor=availabilityFactor.loc[(slice(None),Selected_TECHNOLOGIES),:]
TechParameters=TechParameters.loc[Selected_TECHNOLOGIES,:]
TechParameters.loc["OldNuke",'RampConstraintMoins']=0.01 ## a bit strong to put in light the effect
TechParameters.loc["OldNuke",'RampConstraintPlus']=0.02 ## a bit strong to put in light the effect
#endregion

#region II - Ramp Single area : solving and loading results
model = GetElectricSystemModel_Planing(Parameters={"areaConsumption"      :   areaConsumption,
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

#region II - Ramp Single area : visualisation and lagrange multipliers
fig=MyStackedPlotly(y_df=production_df,Conso = areaConsumption)
fig=fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline
#fig2.show()

#### lagrange multipliers
Constraints= getConstraintsDual_panda(model)

# Analyse energyCtr
energyCtrDual=Constraints['energyCtr']; energyCtrDual['energyCtr']=energyCtrDual['energyCtr']*1000000
energyCtrDual
round(energyCtrDual.energyCtr,2).unique()

# Analyse CapacityCtr
CapacityCtrDual=Constraints['CapacityCtr'].pivot(index="Date",columns='TECHNOLOGIES', values='CapacityCtr')*1000000;
round(CapacityCtrDual,2)
round(CapacityCtrDual.OldNuke,2).unique() ## if you increase by Delta the installed capacity of nuke you decrease by xxx the cost when nuke is not sufficient
round(CapacityCtrDual.CCG,2).unique() ## increasing the capacity of CCG as no effect on prices
#endregion

#region III - Ramp multiple area : loading parameters
Zones="FR_DE_GB_ES"
year=2016
Selected_AREAS=["FR","DE"]
Selected_TECHNOLOGIES=['OldNuke', 'CCG','WindOnShore',"curtailment"] #you'll add 'Solar' after #'NewNuke', 'HydroRiver', 'HydroReservoir','WindOnShore', 'WindOffShore', 'Solar', 'Curtailement'}

#### reading CSV files
areaConsumption = pd.read_csv(InputFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["AREAS","Date"])
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["AREAS","Date","TECHNOLOGIES"])
TechParameters = pd.read_csv(InputFolder+'Planing_MultiNode_DE-FR_TECHNOLOGIES_AREAS.csv',
                             sep=',',decimal='.',skiprows=0,comment="#").set_index(["AREAS","TECHNOLOGIES"])

ExchangeParameters = pd.read_csv(InputFolder+'Hypothese_DE-FR_AREAS_AREAS.csv',sep=',',decimal='.',skiprows=0,comment="#").set_index(["AREAS","AREAS.1"])
#### Selection of subset
TechParameters=TechParameters.loc[(Selected_AREAS,Selected_TECHNOLOGIES),:]
areaConsumption=areaConsumption.loc[(Selected_AREAS,slice(None)),:]
availabilityFactor=availabilityFactor.loc[(Selected_AREAS,slice(None),Selected_TECHNOLOGIES),:]
TechParameters.loc[(slice(None),"OldNuke"),'RampConstraintMoins']=0.01 ## a bit strong to put in light the effect
TechParameters.loc[(slice(None),"OldNuke"),'RampConstraintPlus']=0.02 ## a bit strong to put in light the effect
#endregion

#region III - Ramp multiple area : solving and loading results
### small data cleaning
availabilityFactor.availabilityFactor[availabilityFactor.availabilityFactor>1]=1
model = GetElectricSystemModel_Planing(Parameters={"areaConsumption"      :   areaConsumption,
                                                   "availabilityFactor"   :   availabilityFactor,
                                                   "TechParameters"       :   TechParameters,
                                                   "ExchangeParameters"   : ExchangeParameters})

opt = SolverFactory(solver)
results=opt.solve(model)
Variables=getVariables_panda_indexed(model)
print(extractCosts(Variables))
print(extractEnergyCapacity(Variables))

production_df=EnergyAndExchange2Prod(Variables)

### Check sum Prod = Consumption
Delta= production_df.sum(axis=1)-areaConsumption.areaConsumption
abs(Delta).sum()

## adding dates

fig=MyAreaStackedPlot(production_df,Conso=areaConsumption)
fig=fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline

Constraints= getConstraintsDual_panda(model)
Constraints.keys()
#endregion

#region IV Ramp+Storage single area : loading parameters
Zones="FR"
year=2013

Selected_TECHNOLOGIES=['OldNuke','WindOnShore', 'CCG',"curtailment",'HydroRiver', 'HydroReservoir',"Solar"] ## try adding 'HydroRiver', 'HydroReservoir'

#### reading CSV files
areaConsumption = pd.read_csv(InputFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["Date"])
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["Date","TECHNOLOGIES"])
TechParameters = pd.read_csv(InputFolder+'Planing-RAMP1BIS_TECHNOLOGIES.csv',
                             sep=',',decimal='.',skiprows=0,comment="#").set_index(["TECHNOLOGIES"])
StorageParameters = pd.read_csv(InputFolder+'Planing-RAMP1_STOCK_TECHNO.csv',sep=',',decimal='.',skiprows=0).set_index(["STOCK_TECHNO"])

#### Selection of subset
availabilityFactor=availabilityFactor.loc[(slice(None),Selected_TECHNOLOGIES),:]
TechParameters=TechParameters.loc[Selected_TECHNOLOGIES,:]
#TechParameters.loc["CCG",'capacity']=100000 ## margin to make everything work
TechParameters.loc["CCG",'maxCapacity']=50000
TechParameters.loc["OldNuke",'maxCapacity']=30000
TechParameters.loc["OldNuke",'RampConstraintMoins']=0.02 ## a bit strong to put in light the effect
TechParameters.loc["OldNuke",'RampConstraintPlus']=0.02 ## a bit strong to put in light the effect
#endregion

#region IV Ramp+Storage single area : solving and loading results
model = GetElectricSystemModel_Planing(Parameters={"areaConsumption"      :   areaConsumption,
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

#region V Case Storage + CCG + PV + Wind + hydro (Ramp+Storage single area) : loading parameters
Zones="FR"
year=2013

Selected_TECHNOLOGIES=['CCG', 'WindOnShore','WindOffShore','Solar',"curtailment",'HydroRiver', 'HydroReservoir']

#### reading CSV files
areaConsumption = pd.read_csv(InputFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["Date"])
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["Date","TECHNOLOGIES"])
TechParameters = pd.read_csv(InputFolder+'Planing-RAMP1BIS_TECHNOLOGIES.csv',
                             sep=',',decimal='.',skiprows=0,comment="#").set_index(["TECHNOLOGIES"])
StorageParameters = pd.read_csv(InputFolder+'Planing-RAMP1_STOCK_TECHNO.csv',
                                sep=',',decimal='.',skiprows=0).set_index(["STOCK_TECHNO"])
#### Selection of subset
availabilityFactor=availabilityFactor.loc[(slice(None),Selected_TECHNOLOGIES),:]
TechParameters=TechParameters.loc[Selected_TECHNOLOGIES,:]
#TechParameters.loc["CCG",'capacity']=100000 ## margin to make everything work
TechParameters.loc["CCG",'energyCost']=100
TechParameters.loc["CCG",'maxCapacity']=50000
TechParameters.loc["CCG",'RampConstraintMoins']=0.5 ## a bit strong to put in light the effect
TechParameters.loc["CCG",'RampConstraintPlus']=0.5 ## a bit strong to put in light the effect
#endregion

#region V Case Storage + CCG + PV + Wind (Ramp+Storage single area) : solving and loading results
model = GetElectricSystemModel_Planing(Parameters={"areaConsumption"      :   areaConsumption,
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

#region VI Case Storage + CCG + Nuke (Ramp+Storage single area) : loading parameters
Zones="FR"
year=2013

Selected_TECHNOLOGIES=['CCG', 'NewNuke',"curtailment",'HydroRiver', 'HydroReservoir']

#### reading CSV files
areaConsumption = pd.read_csv(InputFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["Date"])
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["Date","TECHNOLOGIES"])
TechParameters = pd.read_csv(InputFolder+'Planing-RAMP1BIS_TECHNOLOGIES.csv',
                             sep=',',decimal='.',skiprows=0,comment="#").set_index(["TECHNOLOGIES"])
StorageParameters = pd.read_csv(InputFolder+'Planing-RAMP1_STOCK_TECHNO.csv',sep=',',decimal='.',skiprows=0).set_index(["STOCK_TECHNO"])


#### Selection of subset
availabilityFactor=availabilityFactor.loc[(slice(None),Selected_TECHNOLOGIES),:]
TechParameters=TechParameters.loc[Selected_TECHNOLOGIES,:]
TechParameters.loc["CCG",'energyCost']=300
TechParameters.loc["CCG",'maxCapacity']=50000
TechParameters.loc["CCG",'RampConstraintMoins']=0.05 ## a bit strong to put in light the effect
TechParameters.loc["CCG",'RampConstraintPlus']=0.05 ## a bit strong to put in light the effect
TechParameters.loc["NewNuke",'RampConstraintMoins']=0.01 ## a bit strong to put in light the effect
TechParameters.loc["NewNuke",'RampConstraintPlus']=0.02 ## a bit strong to put in light the effect
#endregion

#region VI Case Storage + CCG + Nuke (Ramp+Storage single area) : solving and loading results
model = GetElectricSystemModel_Planing(Parameters={"areaConsumption"      :   areaConsumption,
                                                   "availabilityFactor"   :   availabilityFactor,
                                                   "TechParameters"       :   TechParameters,
                                                   "StorageParameters"   : StorageParameters})
if solver in solverpath :  opt = SolverFactory(solver,executable=solverpath[solver])
else : opt = SolverFactory(solver)
results=opt.solve(model)
Variables = getVariables_panda_indexed(model)
Constraints = getConstraintsDual_panda(model)

print(extractCosts(Variables))
print(extractEnergyCapacity(Variables))


production_df=Variables['energy'].pivot(index="Date",columns='TECHNOLOGIES', values='energy')
stockage=Variables['storageOut'].pivot(index='Date',columns='STOCK_TECHNO',values='storageOut').sum(axis=1)-Variables['storageIn'].pivot(index='Date',columns='STOCK_TECHNO',values='storageIn').sum(axis=1)
areaConsumption['Storage']=stockage
areaConsumption['NewConsumption']=areaConsumption['areaConsumption']-stockage
Delta= production_df.sum(axis=1)-areaConsumption["NewConsumption"]
print(sum(abs(Delta)))
production_df.loc[:,'Storage'] = areaConsumption["Storage"] ### put storage in the production time series
production_df.sum(axis=0)/10**6 ### energies produites TWh
production_df[production_df>0].sum(axis=0)/10**6 ### energies produites TWh
production_df.max(axis=0)/1000 ### Pmax en GW

Date_d=pd.date_range(start=str(year)+"-01-01 00:00:00",end=str(year)+"-12-31 23:00:00",   freq="1H")
production_df.index=Date_d; areaConsumption.index=Date_d;
fig=MyStackedPlotly(y_df=production_df, Conso=areaConsumption)
fig=fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline
#fig.show()
#endregion

#region VI Ramp+Storage Multi area : loading parameters
Zones="FR_DE_GB_ES"
year=2016
Selected_AREAS=["FR","DE"]
Selected_TECHNOLOGIES=['OldNuke', 'CCG','WindOnShore',"curtailment"] #you'll add 'Solar' after #'NewNuke', 'HydroRiver', 'HydroReservoir','WindOnShore', 'WindOffShore', 'Solar', 'Curtailement'}

#### reading CSV files
TechParameters = pd.read_csv(InputFolder+'Planing_MultiNode_DE-FR_TECHNOLOGIES_AREAS.csv',
                             sep=',',decimal='.',comment="#").set_index(["AREAS","TECHNOLOGIES"])
areaConsumption = pd.read_csv(InputFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["AREAS","Date"])
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["AREAS","Date","TECHNOLOGIES"])

ExchangeParameters = pd.read_csv(InputFolder+'Hypothese_DE-FR_AREAS_AREAS.csv',
                                 sep=',',decimal='.',skiprows=0,comment="#").set_index(["AREAS","AREAS.1"])
StorageParameters = pd.read_csv(InputFolder+'Planing_MultiNode_AREAS_DE-FR_STOCK_TECHNO.csv',sep=',',decimal='.',comment="#",skiprows=0).set_index(["AREAS","STOCK_TECHNO"])

#### Selection of subset
TechParameters=TechParameters.loc[(Selected_AREAS,Selected_TECHNOLOGIES),:]
areaConsumption=areaConsumption.loc[(Selected_AREAS,slice(None)),:]
availabilityFactor=availabilityFactor.loc[(Selected_AREAS,slice(None),Selected_TECHNOLOGIES),:]
TechParameters.loc[(slice(None),'CCG'),'energyCost']=300 ## margin to make everything work
TechParameters.loc[(slice(None),"OldNuke"),'RampConstraintMoins']=0.01 ## a bit strong to put in light the effect
TechParameters.loc[(slice(None),"OldNuke"),'RampConstraintPlus']=0.02 ## a bit strong to put in light the effect
#endregion

#region VI Ramp+Storage multi area : solving and loading results
model = GetElectricSystemModel_Planing(Parameters={"areaConsumption"      :   areaConsumption,
                                                   "availabilityFactor"   :   availabilityFactor,
                                                   "TechParameters"       :   TechParameters,
                                                   "StorageParameters"   : StorageParameters,
                                                   "ExchangeParameters" : ExchangeParameters})
if solver in solverpath :  opt = SolverFactory(solver,executable=solverpath[solver])
else : opt = SolverFactory(solver)
results=opt.solve(model)
Variables = getVariables_panda_indexed(model)
Constraints = getConstraintsDual_panda(model)

print(extractCosts(Variables))
#print(extractEnergyCapacity(Variables))

production_df=EnergyAndExchange2Prod(Variables)
stockage=Variables['storageOut'].pivot(index=['AREAS','Date'],columns='STOCK_TECHNO',values='storageOut').sum(axis=1)-Variables['storageIn'].pivot(index=['AREAS','Date'],columns='STOCK_TECHNO',values='storageIn').sum(axis=1)
areaConsumption['Storage']=stockage
areaConsumption['NewConsumption']=areaConsumption['areaConsumption']-stockage
Delta=(production_df.sum(axis=1) - areaConsumption.NewConsumption); ## comparaison à la conso incluant le stockage
abs(Delta).max()
production_df.loc[:,'Storage'] = -areaConsumption["Storage"] #### ajout du stockage comme production
Delta=(production_df.sum(axis=1) - areaConsumption.areaConsumption);
abs(Delta).max()

fig=MyAreaStackedPlot(production_df,Conso=areaConsumption)
fig=fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline
#fig.show()

abs(areaConsumption["Storage"]).groupby(by="AREAS").sum() ## stockage
production_df.groupby(by="AREAS").sum()/10**6 ### energies produites TWh
production_df[production_df>0].groupby(by="AREAS").sum()/10**6 ### energies produites TWh
production_df.groupby(by="AREAS").max()/1000 ### Pmax en GW ### le stockage ne fait rien en Allemagne ??? bizarre
production_df.groupby(by="AREAS").min()/1000 ### Pmax en GW
#endregion


#region VII - Simple single area +4 million EV +  demande side management +30TWh H2: loading parameters
Zones="FR" ; year=2013
#### reading areaConsumption availabilityFactor and TechParameters CSV files
areaConsumption = pd.read_csv(InputFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["Date"])
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["Date","TECHNOLOGIES"])

#obtaining industry-metal consumption
Profile_df=pd.read_csv(InputFolder+"ConsumptionDetailedProfiles.csv").set_index(["Mois", "heures",'Nature', 'type', 'UsagesGroupe', 'UsageDetail', "WeekDay"])
Profile_df_merged=ComplexProfile2Consumption(Profile_df,areaConsumption,TemperatureName='areaConsumption') #TODO : change ComplexProfile2Consumption to adapt to the new date format
#Profile_df_merged_spread_0=Profile_df_merged.groupby(["Date","type"]).sum().reset_index().drop(columns=["areaConsumption"]).pivot(index="Date", columns=['type'], values='Conso');
Profile_df_merged_spread = Profile_df_merged.groupby(["Date","Nature","UsagesGroupe","type"]).sum().reset_index().drop(columns=["areaConsumption"]).pivot(index="Date", columns=["Nature",'type',"UsagesGroupe"], values='Conso');
steel_consumption=Profile_df_merged_spread.loc[:,("MineraiMetal","Ind",'Process')]*areaConsumption.loc[:,"areaConsumption"]
steel_consumption.iloc[0]=110
steel_consumption.max()
steel_consumption[steel_consumption.isna()]=110
steel_consumption.isna().sum()
# if you want to change thermal sensitivity + add electric vehicle
ConsoTempe_df=pd.read_csv(InputFolder+'ConsumptionTemperature_1996TO2019_FR.csv',parse_dates=['Date']).set_index(["Date"]) #
ConsoTempe_df_nodup=ConsoTempe_df.loc[~ConsoTempe_df.index.duplicated(),:]
(ConsoTempeYear_decomposed_df,Thermosensibilite)=Decomposeconso(areaConsumption.join(ConsoTempe_df_nodup)[['areaConsumption','Temperature']],
                                                                TemperatureThreshold=15,ConsumptionName="areaConsumption")
areaConsumption = areaConsumption.assign(areaConsumption = ConsoTempeYear_decomposed_df.loc[:,'TS_C']+ConsoTempeYear_decomposed_df.loc[:,'NTS_C'])#+VE_consumption.loc[:,'Consumption'])

VEProfile_df=pd.read_csv(InputFolder+'EVModel.csv', sep=';')
NbVE=10 # millions
ev_consumption = NbVE*Profile2Consumption(Profile_df=VEProfile_df,Temperature_df = ConsoTempe_df_nodup.loc[str(year)][['Temperature']])[['Consumption']]

TechParameters = pd.read_csv(InputFolder+'Planing-RAMP1BIS_TECHNOLOGIES.csv',sep=',',decimal='.',skiprows=0,comment="#").set_index(["TECHNOLOGIES"])
StorageParameters = pd.read_csv(InputFolder + 'Planing-RAMP1_STOCK_TECHNO.csv', sep=',', decimal='.',
                                skiprows=0).set_index(["STOCK_TECHNO"])
ConsoParameters = pd.read_csv(InputFolder + "Planing-Conso-FLEX_CONSUM.csv", sep=";").set_index(["FLEX_CONSUM"])

Selected_TECHNOLOGIES=['OldNuke','CCG','TAC', 'WindOnShore', 'WindOffShore','HydroReservoir','HydroRiver','Solar','curtailment']#you can add technologies here
availabilityFactor=availabilityFactor.loc[(slice(None),Selected_TECHNOLOGIES),:]
TechParameters=TechParameters.loc[Selected_TECHNOLOGIES,:]



h2_Energy = 30000## H2 volume in GWh/year
h2_Energy_flat_consumption = ev_consumption.Consumption*0+h2_Energy/8760
to_flex_consumption=pd.DataFrame({'to_flex_consumption': steel_consumption,'FLEX_CONSUM' : 'Steel'}).reset_index().set_index(['Date','FLEX_CONSUM']).\
    append(pd.DataFrame({'to_flex_consumption': ev_consumption.Consumption,'FLEX_CONSUM' : 'EV'}).reset_index().set_index(['Date','FLEX_CONSUM'])).\
    append(pd.DataFrame({'to_flex_consumption': h2_Energy_flat_consumption,'FLEX_CONSUM' : 'H2'}).reset_index().set_index(['Date','FLEX_CONSUM']))

ConsoParameters_ = ConsoParameters.join(
    to_flex_consumption.groupby("FLEX_CONSUM").max().rename(columns={"to_flex_consumption": "max_power"}))


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

labour_ratio.head()
to_flex_consumption.head()

ConsoParameters.loc['H2','LoadCost']


# endregion
# €/kW/an coût fixe additionnel pour un GW d'électrolyseur en plus en supposant que l'on construit

model =  GetElectricSystemModel_Planing(Parameters={"areaConsumption"      :   areaConsumption,
                                           "availabilityFactor"   :   availabilityFactor,
                                           "TechParameters"       :   TechParameters,
                                           "StorageParameters"   : StorageParameters,
                                           "to_flex_consumption" : to_flex_consumption,
                                           "ConsoParameters_" : ConsoParameters_,
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