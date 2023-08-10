
#region importation of modules

## global tools
from Models.Basic_France_models.Operation_optimisation.f_operationModels import *
from EnergyAlternativesPlanning.f_tools import *
from EnergyAlternativesPlanning.f_graphicalTools import *
from EnergyAlternativesPlanning.f_consumptionModels import *

## locally defined models
from Models.Basic_France_models.Operation_optimisation.f_operationModels import *
#endregion

#region Solver and data location definition


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
sys.path.append('/opt/mosek')
solvers= ['gurobi','knitro','cbc'] # try 'glpk', 'cplex'
solverpath= {}
for solver in solvers : solverpath[solver]=BaseSolverPath+'/'+solver
solver= 'mosek' ## no need for solverpath with mosek.
#endregion

InputConsumptionFolder='Models/Basic_France_models/Consumption/Data/'
InputProductionFolder='Models/Basic_France_models/Production/Data/'
InputOperationFolder='Models/Basic_France_models/Operation_optimisation/Data/'
GraphicalResultsFolder="Models/Basic_France_Germany_models/Operation_optimisation/GraphicalResults/"


#region I - Simple single area : loading parameters
year=2013;
#### reading areaConsumption availabilityFactor and TechParameters CSV files
areaConsumption = pd.read_csv(InputConsumptionFolder+'areaConsumption'+str(year)+'_FR.csv',sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["Date"])
availabilityFactor = pd.read_csv(InputProductionFolder+'availabilityFactor'+str(year)+'_FR.csv',sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["Date","TECHNOLOGIES"])
TechParameters = pd.read_csv(InputOperationFolder+'Gestion-Simple_TECHNOLOGIES.csv',sep=',',decimal='.',skiprows=0).set_index(["TECHNOLOGIES"])

#### Selection of subset
Selected_TECHNOLOGIES=['OldNuke','CCG'] #you can add technologies here
availabilityFactor=availabilityFactor.loc[(slice(None),Selected_TECHNOLOGIES),:]
TechParameters=TechParameters.loc[Selected_TECHNOLOGIES,:]
TechParameters.loc["CCG",'capacity']=100000 ## margin to make everything work
#TechParameters.loc["WindOnShore",'capacity']=117000
#TechParameters.loc["Solar",'capacity']=67000
#endregion

#region I - Simple single area  : Solving and loading results
model = GetElectricSystemModel_GestionSingleNode(Parameters = {"areaConsumption" : areaConsumption,
                                                               "availabilityFactor" : availabilityFactor,
                                                               "TechParameters" : TechParameters})
if solver in solverpath :  opt = SolverFactory(solver,executable=solverpath[solver])
else : opt = SolverFactory(solver)
results=opt.solve(model)
## result analysis
Variables=getVariables_panda_indexed(model)


extractCosts(Variables)
#extractEnergyCapacity(Variables)
#pour avoir la production en KWh de chaque moyen de prod chaque heure
### Check sum Prod = Consumption
production_df=Variables['energy'].pivot(index="Date",columns='TECHNOLOGIES', values='energy')
Delta=(production_df.sum(axis=1) - areaConsumption.areaConsumption);
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
CapacityCtrDual=Constraints['CapacityCtr'].pivot(index="Date",columns='TECHNOLOGIES', values='CapacityCtr');
round(CapacityCtrDual,2)
round(CapacityCtrDual.OldNuke,2).unique() ## if you increase by Delta the installed capacity of nuke you decrease by xxx the cost when nuke is not sufficient
round(CapacityCtrDual.CCG,2).unique() ## increasing the capacity of CCG as no effect on prices
#endregion

#region II - Ramp Ctrs Single area : loading parameters loading parameterscase with ramp constraints
year=2013
Selected_TECHNOLOGIES=['OldNuke', 'CCG',"curtailment"] #you'll add 'Solar' after

#### reading areaConsumption availabilityFactor and TechParameters CSV files
areaConsumption = pd.read_csv(InputConsumptionFolder+'areaConsumption'+str(year)+'_FR.csv',sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["Date"])
availabilityFactor = pd.read_csv(InputProductionFolder+'availabilityFactor'+str(year)+'_FR.csv',sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["Date","TECHNOLOGIES"])
TechParameters = pd.read_csv(InputOperationFolder+'Gestion-RAMP1_TECHNOLOGIES.csv',sep=',',decimal='.',skiprows=0).set_index(["TECHNOLOGIES"])

#### Selection of subset
availabilityFactor=availabilityFactor.loc[(slice(None),Selected_TECHNOLOGIES),:]
TechParameters=TechParameters.loc[Selected_TECHNOLOGIES,:]

TechParameters.loc["CCG",'capacity']=100000 ## margin to make everything work
TechParameters.loc["OldNuke",'RampConstraintMoins']=0.01 ## a bit strong to put in light the effect
TechParameters.loc["OldNuke",'RampConstraintPlus']=0.02 ## a bit strong to put in light the effect
#endregion

#region II - Ramp Ctrs Single area : solving and loading results
model = GetElectricSystemModel_GestionSingleNode(Parameters = {"areaConsumption" : areaConsumption,
                                                               "availabilityFactor" : availabilityFactor,
                                                               "TechParameters" : TechParameters})
opt = SolverFactory(solver)
results=opt.solve(model)
Variables=getVariables_panda_indexed(model)


#pour avoir la production en KWh de chaque moyen de prod chaque heure
production_df=Variables['energy'].pivot(index="Date",columns='TECHNOLOGIES', values='energy')
### Check sum Prod = Consumption
Delta=(production_df.sum(axis=1) - areaConsumption.areaConsumption);
abs(Delta).max()
print(production_df.loc[:,'OldNuke'].diff(1).max()/TechParameters.loc["OldNuke","capacity"])
print(production_df.loc[:,'OldNuke'].diff(1).min()/TechParameters.loc["OldNuke","capacity"])



print(production_df.sum(axis=0)/10**6) ### energies produites TWh
print(Variables['energyCosts']) #pour avoir le coût de chaque moyen de prod à l'année
#endregion

#region II - Ramp Ctrs Single area : visualisation and lagrange multipliers


### representation des résultats
Date_d=pd.date_range(start=str(year)+"-01-01 00:00:00",end=str(year)+"-12-31 23:00:00",   freq="1H")
production_df.index=Date_d; areaConsumption.index=Date_d;
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
CapacityCtrDual=Constraints['CapacityCtr'].pivot(index="Date",columns='TECHNOLOGIES', values='CapacityCtr');
round(CapacityCtrDual,2)
round(CapacityCtrDual.OldNuke,2).unique() ## if you increase by Delta the installed capacity of nuke you decrease by xxx the cost when nuke is not sufficient
round(CapacityCtrDual.CCG,2).unique() ## increasing the capacity of CCG as no effect on prices
#endregion

#region III Ramp+Storage single area : loading parameters
year=2013
Selected_TECHNOLOGIES=['OldNuke','WindOnShore', 'CCG',"curtailment"]

#### reading CSV files
areaConsumption = pd.read_csv(InputConsumptionFolder+'areaConsumption'+str(year)+'_FR.csv',sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["Date"])
availabilityFactor = pd.read_csv(InputProductionFolder+'availabilityFactor'+str(year)+'_FR.csv',sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["Date","TECHNOLOGIES"])
TechParameters = pd.read_csv(InputOperationFolder+'Gestion-RAMP1_TECHNOLOGIES.csv',sep=',',decimal='.',skiprows=0).set_index(["TECHNOLOGIES"])
StorageParameters = pd.read_csv(InputOperationFolder+'Gestion-RAMP1_STOCK_TECHNO.csv',sep=',',decimal='.',skiprows=0).set_index(["STOCK_TECHNO"])

#### Selection of subset
availabilityFactor=availabilityFactor.loc[(slice(None),Selected_TECHNOLOGIES),:]
TechParameters=TechParameters.loc[Selected_TECHNOLOGIES,:]
TechParameters.loc["CCG",'capacity']=100000 ## margin to make everything work
TechParameters.loc["OldNuke",'RampConstraintMoins']=0.02 ## a bit strong to put in light the effect
TechParameters.loc["OldNuke",'RampConstraintPlus']=0.02 ## a bit strong to put in light the effect
#endregion

#region III Ramp+Storage single area : solving and loading results
model= GetElectricSystemModel_GestionSingleNode_withStorage(Parameters = {"areaConsumption" : areaConsumption,
                                                               "availabilityFactor" : availabilityFactor,
                                                               "TechParameters" : TechParameters,
                                                               "StorageParameters" : StorageParameters})

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

Date_d=pd.date_range(start=str(year)+"-01-01 00:00:00",end=str(year)+"-12-31 23:00:00",   freq="1H")
production_df.index=Date_d; areaConsumption.index=Date_d;
fig=MyStackedPlotly(y_df=production_df, Conso=areaConsumption)
fig=fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename=GraphicalResultsFolder+'file.html') ## offline


#endregion
