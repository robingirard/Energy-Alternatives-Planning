#region Importation of modules
import numpy as np
import pandas as pd
import csv
#import docplex
import datetime
import copy
import plotly
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import sys
import time
import datetime
#import seaborn as sb

from Models.MultiRessource.f_Models_TP_Robin import *
from EnergyAlternativesPlaning.f_tools import *
from Models.MultiRessource.scenarios_TP import *

#endregion

#region Solver and data location definition
outputPath='Data/output/'
solver= 'mosek' ## no need for solverpath with mosek.
inputPath = 'Models/MultiRessource/Data/'
#endregion
#Dates = pd.DataFrame.from_dict({"ind" : range(1,8761),"Dates": availabilityFactor.reset_index().Date.unique()}).set_index(["ind"])
#Calend=pd.read_csv(inputPath + 'H2Demand_TIME.csv', sep=',', decimal='.', skiprows=0, comment="#").rename(columns={'TIMESTAMP': "Dates"})
#Calend.loc[:,"Dates"] = list(Dates.loc[list(Calend.Dates),"Dates"])
#Calend.to_csv(inputPath+'H2Demand_TIME.csv',index=False)

scenario=Scenario_Heat(2030,inputPath=inputPath)

print('Building model...')
Parameters = loadScenario(scenario, False)
model = systemModel_MultiResource_WithStorage(Parameters)
#start_clock = time.time()
print('Calculating model...')
opt = SolverFactory(solver)
results = opt.solve(model)
#end_clock = time.time()
#print('Computational time: {:.0f} s'.format(end_clock - start_clock))

res_HEAT = {
    'variables': getVariables_panda(model),
    'constraints': getConstraintsDual_panda(model)
}


#region H2 model
scenario=Scenario_H2(2030,inputPath=inputPath)

print('Building model...')
Parameters = loadScenario(scenario, False)
model = systemModel_MultiResource_WithStorage(Parameters)
start_clock = time.time()
print('Calculating model...')
opt = SolverFactory(solver)
results = opt.solve(model)
end_clock = time.time()
print('Computational time: {:.0f} s'.format(end_clock - start_clock))

res = {
    'variables': getVariables_panda(model),
    'constraints': getConstraintsDual_panda(model)
}

#endregion

