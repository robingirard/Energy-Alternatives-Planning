
## intro
InputFolder='Data/input/'
## solver can be found here
# https://ampl.com/products/solvers/open-source/

solverpath={}
solverpath['cbc']='/Users/robin.girard/Documents/Code/Packages/solvers/cbc'
solverpath['cplex']='/Users/robin.girard/Documents/Code/Packages/solvers/cplex'

solver= 'mosek' ## glpk
import numpy as np
import pandas as pd
import csv

import datetime
import copy
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn import linear_model

from functions.functions_Operation import *


## first example
Zones="FR" ; year=2013
#### reading areaConsumption availabilityFactor and TechParameters CSV files
areaConsumption = pd.read_csv(InputFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',sep=',',decimal='.',skiprows=0)
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',sep=',',decimal='.',skiprows=0)
TechParameters = pd.read_csv(InputFolder+'Gestion-Simple_TECHNOLOGIES.csv',sep=';',decimal=',',skiprows=0)

#### Selection of subset
Selected_TECHNOLOGIES={'OldNuke','Thermal'} #you can add technologies here
availabilityFactor=availabilityFactor[ availabilityFactor.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]
TechParameters=TechParameters[TechParameters.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]

model = GetElectricSystemModel_GestionSingleNode(areaConsumption,availabilityFactor,TechParameters)
if solver in solverpath :  opt = SolverFactory(solver,executable=solverpath[solver])
else : opt = SolverFactory(solver)
results=opt.solve(model)

## result analysis
Variables=getVariables_panda(model)

#pour avoir la production en KWh de chaque moyen de prod chaque heure
production_df=Variables['energy'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='energy')
print(production_df.head())
print(Variables['energyCosts']) #pour avoir le coût de chaque moyen de prod à l'année

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
energyCtrDual=Constraints['energyCtr']; energyCtrDual['energyCtr']=energyCtrDual['energyCtr']*1000000
energyCtrDual
round(energyCtrDual.energyCtr,2).unique()

# Analyse CapacityCtr
CapacityCtrDual=Constraints['CapacityCtr'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='CapacityCtr')*1000000;
round(CapacityCtrDual,2)
round(CapacityCtrDual.OldNuke,2).unique() ## if you increase by Delta the installed capacity of nuke you decrease by xxx the cost when nuke is not sufficient
round(CapacityCtrDual.Thermal,2).unique() ## increasing the capacity of Thermal as no effect on prices


#### case with ramp constraints

Zones="FR"
year=2013
Selected_TECHNOLOGIES={'OldNuke','Thermal'} #you'll add 'Solar' after
#### reading CSV files
areaConsumption = pd.read_csv(InputFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0)
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0)
TechParameters = pd.read_csv(InputFolder+'Gestion-RAMP1_TECHNOLOGIES.csv',sep=';',decimal=',',skiprows=0)

#### Selection of subset
availabilityFactor=availabilityFactor[availabilityFactor.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]
TechParameters=TechParameters[TechParameters.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]

model = GetElectricSystemModel_GestionSingleNode(areaConsumption,availabilityFactor,TechParameters)
opt = SolverFactory(solver)
results=opt.solve(model)
Variables=getVariables_panda(model)

#pour avoir la production en KWh de chaque moyen de prod chaque heure
production_df=Variables['energy'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='energy')
print(production_df.head())
print(Variables['energyCosts']) #pour avoir le coût de chaque moyen de prod à l'année

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
energyCtrDual=Constraints['energyCtr']; energyCtrDual['energyCtr']=energyCtrDual['energyCtr']*1000000
energyCtrDual
round(energyCtrDual.energyCtr,2).unique()

# Analyse CapacityCtr
CapacityCtrDual=Constraints['CapacityCtr'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='CapacityCtr')*1000000;
round(CapacityCtrDual,2)
round(CapacityCtrDual.OldNuke,2).unique() ## if you increase by Delta the installed capacity of nuke you decrease by xxx the cost when nuke is not sufficient
round(CapacityCtrDual.Thermal,2).unique() ## increasing the capacity of Thermal as no effect on prices


Zones="FR_DE_GB_ES"
year=2016
Selected_AREAS={"FR","DE"}
Selected_TECHNOLOGIES={'Thermal', 'OldNuke', 'NewNuke', 'HydroRiver', 'HydroReservoir',
       'WindOnShore', 'WindOffShore', 'Solar', 'Curtailement'}

#### reading CSV files
TechParameters = pd.read_csv(InputFolder+'Gestion_MultiNode_DE-FR_AREAS_TECHNOLOGIES.csv',sep=';',decimal=',',comment="#",skiprows=0)
areaConsumption = pd.read_csv(InputFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0)
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0)
ExchangeParameters = pd.read_csv(InputFolder+'Hypothese_DE-FR_AREAS_AREAS.csv',sep=';',decimal=',',skiprows=0,comment="#")
#### Selection of subset
TechParameters=TechParameters[TechParameters.AREAS.isin(Selected_AREAS)&TechParameters.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]
areaConsumption=areaConsumption[areaConsumption.AREAS.isin(Selected_AREAS)]
availabilityFactor=availabilityFactor[availabilityFactor.AREAS.isin(Selected_AREAS)& availabilityFactor.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]

### small data cleaning
availabilityFactor.availabilityFactor[availabilityFactor.availabilityFactor>1]=1
model = GetElectricSystemModel_GestionMultiNode(areaConsumption,availabilityFactor,TechParameters,ExchangeParameters)
opt = SolverFactory(solver)
results=opt.solve(model)
Variables=getVariables_panda(model)
Variables.keys()
Variables["energy"].head()
Variables["exchange"].head()
Variables["exchange"].exchange.sum()
Constraints= getConstraintsDual_panda(model)
Constraints.keys()

