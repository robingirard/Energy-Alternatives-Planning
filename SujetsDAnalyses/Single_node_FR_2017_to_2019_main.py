# region importation of modules
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import pandas as pd


import pickle
if os.path.basename(os.getcwd()) == "SujetsDAnalyses":
    os.chdir('..')  ## to work at project root  like in any IDE

import sys

from functions.f_graphicalTools import *
from functions.f_planingModels import *
#from functions.f_optimization import *
from functions.f_graphicalTools import *
from functions.f_consumptionModels import *
from pyomo.core import *
from pyomo.opt import SolverFactory

# endregion


def main(year,with_exchanges=True):
    start_time = datetime.now()
    print("Simulation for "+str(year))
    if year in range(2017,2020):
        # region Solver and data location definition
        InputFolder = 'Data/input/France 1 noeud/'+str(year)+"/"
    else:
        print("No data for that year --> Data is available for years between 2017 and 2019")
        return None


    if sys.platform != 'win32':
        myhost = os.uname()[1]
    else:
        myhost = ""

    solver = 'mosek'  ## no need for solverpath with mosek.


    # endregion





    #### reading areaConsumption availabilityFactor and TechParameters CSV files
    TechParameters = pd.read_csv(InputFolder+str(year)+'_SingleNode_TECHNOLOGIES.csv',
                                 decimal='.',comment="#")
    TechParameters.set_index(["TECHNOLOGIES"],inplace=True)


    if with_exchanges:
        areaConsumption = pd.read_csv(InputFolder + str(year) + '_SingleNode_areaConsumption_with_exchanges.csv',
                                      decimal='.', skiprows=0, parse_dates=['Date'])
        areaConsumption["Date"] = pd.to_datetime(areaConsumption["Date"])
        areaConsumption.set_index(["Date"], inplace=True)
        ##
    else:
        areaConsumption = pd.read_csv(InputFolder + str(year) + '_SingleNode_areaConsumption_no_exchanges.csv',
                                      decimal='.', skiprows=0, parse_dates=['Date'])
        areaConsumption["Date"] = pd.to_datetime(areaConsumption["Date"])
        areaConsumption.set_index(["Date"], inplace=True)
    availabilityFactor = pd.read_csv(InputFolder+str(year)+'_SingleNode_availability.csv',
                                    decimal='.',skiprows=0,parse_dates=['Date']).set_index(["Date","TECHNOLOGIES"])
    availabilityFactor.loc[availabilityFactor.availabilityFactor > 1,"availabilityFactor"]=1
    StorageParameters = pd.read_csv(InputFolder+str(year)+'_SingleNode_STOCK_TECHNO.csv',
                                    decimal='.',comment="#",skiprows=0).set_index(["STOCK_TECHNO"])


    # Marginal Cost at power=0 adjustment
    energyCost = {"Biomass": 37.36, "OldNuke": 17.87, "TAC": 45.75, "CCG": 37.24, "Coal": 38.76}
    TechParameters.reset_index(inplace=True)
    for tech in energyCost.keys():
        TechParameters[TechParameters.TECHNOLOGIES==tech]["energyCost"]=energyCost[tech]
    TechParameters.set_index(["TECHNOLOGIES"], inplace=True)

    TechParameters["energyCost"] = TechParameters["energyCost"] - TechParameters["margvarCost"] * TechParameters[
        "minCapacity"] / 2

    # Add RampCtr2 to Nuke
    TechParameters.reset_index(inplace=True)
    TechParameters.loc[TechParameters.TECHNOLOGIES == "OldNuke", "RampConstraintPlus2"] = 0.002
    TechParameters.loc[TechParameters.TECHNOLOGIES == "OldNuke", "RampConstraintMoins2"] = 0.002
    TechParameters.set_index(["TECHNOLOGIES"], inplace=True)


    end_time = datetime.now()
    print('\t Model creation at {}'.format(end_time - start_time))
    model =  GetElectricSystemModel_Planing(Parameters={"areaConsumption"      :   areaConsumption,
                                               "availabilityFactor"   :   availabilityFactor,
                                               "TechParameters"       :   TechParameters,
                                               "StorageParameters"   : StorageParameters,
                                            })
    end_time = datetime.now()
    print('\t Start solving at {}'.format(end_time - start_time))
    opt = SolverFactory(solver)
    results=opt.solve(model)

    end_time = datetime.now()
    print('\t Solved at {}'.format(end_time - start_time))
    Variables = getVariables_panda_indexed(model)
    print(Variables)
    if with_exchanges:
        with open('SujetsDAnalyses/'+str(year)+'_singlenode_results.pickle', 'wb') as f:
            pickle.dump(Variables, f,protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('SujetsDAnalyses/'+str(year)+'_singlenode_results_no_exchanges.pickle', 'wb') as f:
            pickle.dump(Variables, f,protocol=pickle.HIGHEST_PROTOCOL)
    end_time = datetime.now()
    print('\t Total duration: {}'.format(end_time - start_time))

# main(2015)
# main(2016)
# main(2017)
main(2018,True)
# main(2019)