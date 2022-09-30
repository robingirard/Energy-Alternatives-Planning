import os
import warnings
warnings.filterwarnings("ignore")

#from functions.f_optimization import *
from functions.f_consumptionModels import *
from pyomo.opt import SolverFactory

from Models.Seven_node_Europe.Data_processing_functions import *
from Models.Seven_node_Europe.Multinode_model import *

import pickle
if os.path.basename(os.getcwd()) != "Seven_node_Europe":
    os.chdir('Models/Seven_node_Europe/')





def main_historical(year=2018,number_of_sub_techs=1,error_deactivation=False):
    start_time = datetime.now()

    InputFolder = 'Input data/'+str(year)+'/'
    InputFolder_other = "Input data/Conso flex files/"
    if year not in range(2017, 2020):
        print('No data for ' + str(year))
        print("Data is available from 2017 to 2019")
        return None

    if error_deactivation:
        import logging  # to deactivate pyomo false error warnings
        logging.getLogger('pyomo.core').setLevel(logging.ERROR)

    ###############
    # Data import #
    ###############

    #Import generation and storage technologies data
    TechParameters = pd.read_csv(InputFolder + str(year) + '_Planing_MultiNode_TECHNOLOGIES_AREAS.csv')
    TechParameters.dropna(inplace=True)
    StorageParameters = pd.read_csv(InputFolder + str(year) + '_Planing_MultiNode_STOCK_TECHNO.csv', sep=",",
                                    decimal='.', comment="#", skiprows=0).set_index(["AREAS", "STOCK_TECHNO"])
    #Import consumption data
    areaConsumption = pd.read_csv(
        InputFolder + str(year) + '_MultiNodeAreaConsumption.csv',
        decimal='.', skiprows=0, parse_dates=['Date'])
    areaConsumption.dropna(inplace=True)
    areaConsumption["Date"] = pd.to_datetime(areaConsumption["Date"])
    areaConsumption.set_index(["AREAS", "Date"], inplace=True)


    #Import availibility factor data
    availabilityFactor = pd.read_csv(
        InputFolder + str(year) +'_Multinode_availability_factor.csv',
        decimal='.', skiprows=0, parse_dates=['Date']).set_index(["AREAS", "Date", "TECHNOLOGIES"])
    availabilityFactor.loc[availabilityFactor.availabilityFactor > 1, "availabilityFactor"] = 1
    availabilityFactor.dropna(inplace=True)

    #Import interconnections data
    ExchangeParameters = pd.read_csv(InputFolder + str(year) + '_interconnexions.csv', sep=",",
                                     decimal='.').set_index(["AREAS", "AREAS.1"])

    ###################
    # Data adjustment #
    ###################
    techs = TechParameters.TECHNOLOGIES.unique()
    areas = TechParameters.AREAS.unique()
    TechParameters.set_index(["AREAS", "TECHNOLOGIES"], inplace=True)

    #CHP inclusion
    areaConsumption=CHP_processing(areaConsumption,year)

    #Marginal cost adjustment and merit order simulation
    TechParameters=Marginal_cost_adjustment(TechParameters,number_of_sub_techs,techs,areas)

    #Curtailment adjustment
    TechParameters=Curtailment_adjustment(TechParameters,minCapacity=20000,maxCapacity=20000,EnergyNbHourCap=35)

    ##############################
    # Model creation and solving #
    ##############################
    end_time = datetime.now()
    print('\t Model creation at {}'.format(end_time - start_time))
    model = GetElectricitySystemModel(Parameters={"areaConsumption": areaConsumption,
                                                       "availabilityFactor": availabilityFactor,
                                                       "TechParameters": TechParameters,
                                                       "StorageParameters": StorageParameters,
                                                       "ExchangeParameters": ExchangeParameters
                                                       })
    solver = 'mosek'  # 'mosek'  ## no need for solverpath with mosek.
    # solver_path="ampl.mswin64/"+solver
    tee_value = True
    solver_native_list = ["mosek", "glpk"]

    end_time = datetime.now()
    print('\t Start solving at {}'.format(end_time - start_time))

    if solver in solver_native_list:
        opt = SolverFactory(solver)
    else:
        opt = SolverFactory(solver,executable=solver_path,tee=tee_value)#'C:/Program Files/Artelys/Knitro 13.0.1/knitroampl/knitroampl')

    results=opt.solve(model)
    end_time = datetime.now()
    print('\t Solved at {}'.format(end_time - start_time))
    ##############################
    # Data extraction and saving #
    ##############################
    Variables = getVariables_panda_indexed(model)


    with open('Result_'+str(year)+'.pickle', 'wb') as f:
        pickle.dump(Variables, f, protocol=pickle.HIGHEST_PROTOCOL)

    end_time = datetime.now()
    print('\t Total duration: {}'.format(end_time - start_time))
    return Variables
main_historical(year=2018,number_of_sub_techs=1)