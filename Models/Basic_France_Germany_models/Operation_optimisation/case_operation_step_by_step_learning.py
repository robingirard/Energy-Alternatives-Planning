
#region importation of modules

from Models.Basic_France_models.Operation_optimisation.f_operationModels import *
from EnergyAlternativesPlanning.f_tools import *
from EnergyAlternativesPlanning.f_graphicalTools import *
from EnergyAlternativesPlanning.f_consumptionModels import *

## locally defined models
from Models.Basic_France_Germany_models.Operation_optimisation.f_operationModels import *

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
solvers= ['gurobi','knitro','cbc'] # try 'glpk', 'cplex'
solverpath= {}
for solver in solvers : solverpath[solver]=BaseSolverPath+'/'+solver
solver= 'mosek' ## no need for solverpath with mosek.
#endregion

InputConsumptionFolder='Models/Basic_France_Germany_models/Consumption/Data/'
InputProductionFolder='Models/Basic_France_Germany_models/Production/Data/'
InputOperationFolder='Models/Basic_France_Germany_models/Operation_optimisation/Data/'
InputEcoAndTech = 'Models/Basic_France_Germany_models/Economic_And_Tech_Assumptions/'
GraphicalResultsFolder="Models/Basic_France_Germany_models/Operation_optimisation/GraphicalResults/"

#region I - Ramp Ctrs multiple area : loading parameters
Zones="FR_DE_GB_ES"
year=2016
Selected_AREAS=["FR","DE"]
Selected_TECHNOLOGIES=['OldNuke', 'CCG','WindOnShore',"curtailment"] #you'll add 'Solar' after #'NewNuke', 'HydroRiver', 'HydroReservoir','WindOnShore', 'WindOffShore', 'Solar', 'Curtailement'}

#### reading CSV files
TechParameters = pd.read_csv(InputOperationFolder+'Gestion_MultiNode_DE-FR_AREAS_TECHNOLOGIES.csv',sep=',',decimal='.',comment="#",skiprows=0).set_index(["AREAS","TECHNOLOGIES"])
areaConsumption = pd.read_csv(InputConsumptionFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["AREAS","Date"])
availabilityFactor = pd.read_csv(InputProductionFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["AREAS","Date","TECHNOLOGIES"])

ExchangeParameters = pd.read_csv(InputEcoAndTech+'Hypothese_DE-FR_AREAS_AREAS.csv',sep=',',decimal='.',skiprows=0,comment="#").set_index(["AREAS","AREAS.1"])
#ExchangeParameters.loc[ExchangeParameters.AREAS=="FR",'maxExchangeCapacity']=90000 ## margin to make everything work
#ExchangeParameters.loc[ExchangeParameters.AREAS=="DE",'maxExchangeCapacity']=90000 ## margin to make everything work
#### Selection of subset
TechParameters=TechParameters.loc[(Selected_AREAS,Selected_TECHNOLOGIES),:]
areaConsumption=areaConsumption.loc[(Selected_AREAS,slice(None)),:]
availabilityFactor=availabilityFactor.loc[(Selected_AREAS,slice(None),Selected_TECHNOLOGIES),:]
TechParameters.loc[(slice(None),'CCG'),'capacity']=100000 ## margin to make everything work
TechParameters.loc[(slice(None),"OldNuke"),'RampConstraintMoins']=0.001 ## a bit strong to put in light the effect
TechParameters.loc[(slice(None),"OldNuke"),'RampConstraintPlus']=0.001 ## a bit strong to put in light the effect
#endregion

#region I - Ramp Ctrs multiple area : solving and loading results
### small data cleaning
availabilityFactor.availabilityFactor[availabilityFactor.availabilityFactor>1]=1
model = GetElectricSystemModel_GestionMultiNode(Parameters = {"areaConsumption" : areaConsumption,
                                                               "availabilityFactor" : availabilityFactor,
                                                               "TechParameters" : TechParameters,
                                                              "ExchangeParameters":ExchangeParameters})
opt = SolverFactory(solver)
results=opt.solve(model)
Variables=getVariables_panda_indexed(model)
production_df=EnergyAndExchange2Prod(Variables)

### Check sum Prod = Consumption
Delta= production_df.sum(axis=1)-areaConsumption.areaConsumption
abs(Delta).sum()

print(production_df.loc[('DE',slice(None)),'OldNuke'].diff(1).max()/TechParameters.loc[("DE","OldNuke"),"capacity"])
print(production_df.loc[('DE',slice(None)),'OldNuke'].diff(1).min()/TechParameters.loc[("DE","OldNuke"),"capacity"])
print(production_df.loc[('FR',slice(None)),'OldNuke'].diff(1).max()/TechParameters.loc[("FR","OldNuke"),"capacity"])
print(production_df.loc[('FR',slice(None)),'OldNuke'].diff(1).min()/TechParameters.loc[("FR","OldNuke"),"capacity"])
## adding dates

fig=MyAreaStackedPlot(production_df,Conso=areaConsumption)
fig=fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename=GraphicalResultsFolder+'file.html') ## offline

production_df.sum(axis=0)/10**6 ### energies produites TWh
production_df.groupby(by="AREAS").sum()/10**6 ### energies produites TWh
production_df.groupby(by=["AREAS"]).max()

Constraints= getConstraintsDual_panda(model)
Constraints.keys()
Constraints['energyCtr']
#endregion

#region II Ramp+Storage Multi area : loading parameters
Zones="FR_DE_GB_ES"
year=2016
Selected_AREAS=["FR","DE"]
Selected_TECHNOLOGIES=['OldNuke', 'CCG','WindOnShore',"curtailment"] #you'll add 'Solar' after #'NewNuke', 'HydroRiver', 'HydroReservoir','WindOnShore', 'WindOffShore', 'Solar', 'Curtailement'}

#### reading CSV files
TechParameters = pd.read_csv(InputOperationFolder+'Gestion_MultiNode_DE-FR_AREAS_TECHNOLOGIES.csv',
                             sep=',',decimal='.',comment="#",skiprows=0).set_index(["AREAS","TECHNOLOGIES"])
StorageParameters = pd.read_csv(InputOperationFolder+'Gestion_MultiNode_AREAS_DE-FR_STOCK_TECHNO.csv',sep=',',decimal='.',comment="#",skiprows=0).set_index(["AREAS","STOCK_TECHNO"])
areaConsumption = pd.read_csv(InputConsumptionFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["AREAS","Date"])
availabilityFactor = pd.read_csv(InputProductionFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["AREAS","Date","TECHNOLOGIES"])

ExchangeParameters = pd.read_csv(InputEcoAndTech+'Hypothese_DE-FR_AREAS_AREAS.csv',sep=',',decimal='.',skiprows=0,comment="#").set_index(["AREAS","AREAS.1"])
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

#region II Ramp+Storage multi area : solving and loading results
model= GetElectricSystemModel_GestionMultiNode_withStorage(Parameters = {"areaConsumption" : areaConsumption,
                                                               "availabilityFactor" : availabilityFactor,
                                                               "TechParameters" : TechParameters,
                                                              "ExchangeParameters":ExchangeParameters,
                                                              "StorageParameters": StorageParameters})
if solver in solverpath :  opt = SolverFactory(solver,executable=solverpath[solver])
else : opt = SolverFactory(solver)
results=opt.solve(model)
Variables = getVariables_panda_indexed(model)
Constraints = getConstraintsDual_panda(model)

production_df=EnergyAndExchange2Prod(Variables)
stockage=Variables['storageOut'].pivot(index=['AREAS','Date'],columns='STOCK_TECHNO',values='storageOut').sum(axis=1)-Variables['storageIn'].pivot(index=['AREAS','Date'],columns='STOCK_TECHNO',values='storageIn').sum(axis=1)
areaConsumption['Storage']=stockage
areaConsumption['NewConsumption']=areaConsumption['areaConsumption']+stockage

### Check sum Prod = Consumption
Delta=(production_df.sum(axis=1) - areaConsumption.NewConsumption); ## comparaison à la conso incluant le stockage
abs(Delta).max()
production_df.loc[:,'Storage'] = -areaConsumption["Storage"] #### ajout du stockage comme production
Delta=(production_df.sum(axis=1) - areaConsumption.areaConsumption);
abs(Delta).max()

fig=MyAreaStackedPlot(production_df,Conso=areaConsumption)
fig=fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename=GraphicalResultsFolder+'file.html') ## offline

abs(areaConsumption["Storage"]).groupby(by="AREAS").sum() ## stockage
production_df.groupby(by="AREAS").sum()/10**6 ### energies produites TWh
production_df[production_df>0].groupby(by="AREAS").sum()/10**6 ### energies produites TWh
production_df.groupby(by="AREAS").max()/1000 ### Pmax en GW ### le stockage ne fait rien en Allemagne ??? bizarre
production_df.groupby(by="AREAS").min()/1000 ### Pmax en GW
#endregion
