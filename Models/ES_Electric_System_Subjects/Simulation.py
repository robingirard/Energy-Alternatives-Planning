import os
import warnings

import pandas as pd
warnings.filterwarnings("ignore")

#from EnergyAlternativesPlaning.f_optimization import *
from EnergyAlternativesPlaning.f_consumptionModels import *
from Models.ES_Electric_System_Subjects.Data_processing_functions import *
from Models.ES_Electric_System_Subjects.Electric_System_model import *

from pyomo.opt import SolverFactory
import pickle

if os.path.basename(os.getcwd()) != "ES_Electric_System_Subjects":
    os.chdir('Models/ES_Electric_System_Subjects/')


def Simulation_multinode(xls_file,serialize=False):
    start_time = datetime.now()
    year=2018


    ###############
    # Data import #
    ###############

    #Import generation and storage technologies data
    print("Import generation and storage technologies data")
    TechParameters = pd.read_excel(xls_file,"TECHNO_AREAS")
    TechParameters.dropna(inplace=True)
    TechParameters.set_index(["AREAS", "TECHNOLOGIES"], inplace=True)
    StorageParameters = pd.read_excel(xls_file,"STOCK_TECHNO_AREAS").set_index(["AREAS", "STOCK_TECHNO"])

    #Import consumption data
    print("Import consumption data")
    areaConsumption = pd.read_excel(xls_file,"areaConsumption")
    areaConsumption.dropna(inplace=True)
    areaConsumption["Date"] = pd.to_datetime(areaConsumption["Date"])
    areaConsumption.set_index(["AREAS", "Date"], inplace=True)


    #Import availibility factor data
    print("Import availibility factor data")
    availabilityFactor = pd.read_excel(xls_file,"availability_factor")
    availabilityFactor.loc[availabilityFactor.availabilityFactor > 1, "availabilityFactor"] = 1
    availabilityFactor.dropna(inplace=True)
    availabilityFactor["Date"]=pd.to_datetime(availabilityFactor["Date"])
    availabilityFactor.set_index(["AREAS", "Date", "TECHNOLOGIES"],inplace=True)

    #Import interconnections data
    print("Import interconnections data")
    ExchangeParameters = pd.read_excel(xls_file,"interconnexions").set_index(["AREAS", "AREAS.1"])

    ###################
    # Data adjustment #
    ###################
    print("Data adjustment")
    # Temperature sensitivity inclusion
    areaConsumption = Thermosensibility(areaConsumption, xls_file)
    # CHP inclusion for France
    areaConsumption = CHP_processing(areaConsumption, xls_file)
    #Flexibility data inclusion
    ConsoParameters_,labour_ratio,to_flex_consumption=Flexibility_data_processing(areaConsumption,year,xls_file)

    ##############################
    # Model creation and solving #
    ##############################
    end_time = datetime.now()
    print('Model creation at {}'.format(end_time - start_time))
    model = GetElectricitySystemModel(Parameters={"areaConsumption": areaConsumption,
                                                       "availabilityFactor": availabilityFactor,
                                                       "TechParameters": TechParameters,
                                                       "StorageParameters": StorageParameters,
                                                       "ExchangeParameters": ExchangeParameters,
                                                       "to_flex_consumption": to_flex_consumption,
                                                       "ConsoParameters_": ConsoParameters_,
                                                       "labour_ratio": labour_ratio
                                                       })
    solver = 'mosek'  # 'mosek'  ## no need for solverpath with mosek.
    tee_value = True
    solver_native_list = ["mosek", "glpk"]

    end_time = datetime.now()
    print('Start solving at {}'.format(end_time - start_time))

    if solver in solver_native_list:
        opt = SolverFactory(solver)
    else:
        opt = SolverFactory(solver,executable=solver_path,tee=tee_value)

    results=opt.solve(model)
    end_time = datetime.now()
    print('Solved at {}'.format(end_time - start_time))
    ##############################
    # Data extraction and saving #
    ##############################
    Variables = getVariables_panda_indexed(model)

    if serialize:
        with open('Result.pickle', 'wb') as f:
            pickle.dump(Variables, f, protocol=pickle.HIGHEST_PROTOCOL)

    end_time = datetime.now()
    print('Total duration: {}'.format(end_time - start_time))
    return Variables

def Simulation_singlenode(xls_file,serialize=False):
    start_time = datetime.now()
    year=2018


    ###############
    # Data import #
    ###############

    #Import generation and storage technologies data
    print("Import generation and storage technologies data")
    TechParameters = pd.read_excel(xls_file,"TECHNO_AREAS")
    TechParameters.dropna(inplace=True)
    TechParameters.set_index(["TECHNOLOGIES"], inplace=True)
    StorageParameters = pd.read_excel(xls_file,"STOCK_TECHNO_AREAS").set_index(["STOCK_TECHNO"])

    #Import consumption data
    print("Import consumption data")
    areaConsumption = pd.read_excel(xls_file,"areaConsumption")
    areaConsumption.dropna(inplace=True)
    areaConsumption["Date"] = pd.to_datetime(areaConsumption["Date"])
    areaConsumption.set_index(["Date"], inplace=True)


    #Import availibility factor data
    print("Import availibility factor data")
    availabilityFactor = pd.read_excel(xls_file,"availability_factor")
    availabilityFactor.loc[availabilityFactor.availabilityFactor > 1, "availabilityFactor"] = 1
    availabilityFactor.dropna(inplace=True)
    availabilityFactor["Date"]=pd.to_datetime(availabilityFactor["Date"])
    availabilityFactor.set_index(["Date", "TECHNOLOGIES"],inplace=True)


    ###################
    # Data adjustment #
    ###################
    print("Data adjustment")
    #Temperature sensitivity inclusion
    areaConsumption=Thermosensibility_single_node(areaConsumption,xls_file)
    # CHP inclusion for France
    areaConsumption = CHP_processing_single_node(areaConsumption, xls_file)
    #Flexibility data inclusion
    ConsoParameters_,labour_ratio,to_flex_consumption=Flexibility_data_processing_single_node(areaConsumption,year,xls_file)

    ##############################
    # Model creation and solving #
    ##############################
    end_time = datetime.now()
    print('Model creation at {}'.format(end_time - start_time))
    model = GetElectricitySystemModel(Parameters={"areaConsumption": areaConsumption,
                                                       "availabilityFactor": availabilityFactor,
                                                       "TechParameters": TechParameters,
                                                       "StorageParameters": StorageParameters,
                                                       "to_flex_consumption": to_flex_consumption,
                                                       "ConsoParameters_": ConsoParameters_,
                                                       "labour_ratio": labour_ratio
                                                       })
    solver = 'mosek'  # 'mosek'  ## no need for solverpath with mosek.
    tee_value = True
    solver_native_list = ["mosek", "glpk"]

    end_time = datetime.now()
    print('Start solving at {}'.format(end_time - start_time))

    if solver in solver_native_list:
        opt = SolverFactory(solver)
    else:
        opt = SolverFactory(solver,executable=solver_path,tee=tee_value)

    results=opt.solve(model)
    end_time = datetime.now()
    print('Solved at {}'.format(end_time - start_time))
    ##############################
    # Data extraction and saving #
    ##############################
    Variables = getVariables_panda_indexed(model)
    if serialize:
        with open('Result.pickle', 'wb') as f:
            pickle.dump(Variables, f, protocol=pickle.HIGHEST_PROTOCOL)

    end_time = datetime.now()
    print('Total duration: {}'.format(end_time - start_time))
    return Variables


# xls_file=pd.ExcelFile("Single_node_input.xlsx")
# Simulation_singlenode(xls_file)
#
# xls_file=pd.ExcelFile("FR_ES_input.xlsx")
# Simulation_multinode(xls_file)