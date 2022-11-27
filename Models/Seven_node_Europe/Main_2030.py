import os
import warnings

import pandas as pd
warnings.filterwarnings("ignore")

#from EnergyAlternativesPlaning.f_optimization import *
from EnergyAlternativesPlaning.f_consumptionModels import *
from Models.Seven_node_Europe.Data_processing_functions import *
from Models.Seven_node_Europe.Multinode_model import *

from pyomo.opt import SolverFactory
import pickle
if os.path.basename(os.getcwd()) != "Seven_node_Europe":
    os.chdir('Models/Seven_node_Europe/')






InputFolder = 'Input data/2030/'
InputFolder_other="Input data/Conso flex files/"


def main_2030(weather_year=2018,carbon_tax=60,gas_price_coef=1,coal_price_coef=1,number_of_sub_techs=1,error_deactivation=False,no_fossil_mini=False,no_flex=False,noREmax=False,noREmin=False):
    start_time = datetime.now()
    year=2030
    ctax_ini = 30

    instance="_ctax"+str(carbon_tax)+"gpc"+str(gas_price_coef)+"cpc"+str(coal_price_coef)
    if no_fossil_mini:
        instance+="_no_fossil_mini"
    if no_flex:
        instance+="_no"
    instance+="_flex"

    if noREmax:
        instance+="_noREmax"
    if noREmin:
        instance+="_noREmin"
    print("Ongoing instance: "+instance)
    if weather_year not in range(2017, 2020):
        print('No weather data for ' + str(weather_year))
        print("Weather data is available from 2017 to 2019")
        return None
    if weather_year==2019:
        print("ConsumptionTemperature_1996TO2019_FR.csv is not complete \nTherefore 2019 weather data cannot be processed")
        return None
    if error_deactivation:
        import logging  # to deactivate pyomo false error warnings
        logging.getLogger('pyomo.core').setLevel(logging.ERROR)

    ###############
    # Data import #
    ###############

    #Import generation and storage technologies data
    TechParameters = pd.read_csv(InputFolder + str(year) + '_Planing_MultiNode_scenarios_TECHNOLOGIES_AREAS.csv',sep=";", decimal='.', comment="#")
    TechParameters.dropna(inplace=True)
    StorageParameters = pd.read_csv(InputFolder + str(year) + '_Planing_MultiNode_STOCK_TECHNO.csv', sep=";",
                                    decimal='.', comment="#", skiprows=0).set_index(["AREAS", "STOCK_TECHNO"])
    #Import consumption data
    areaConsumption = pd.read_csv(
        InputFolder + str(year) + '_MultiNodeAreaConsumption_' + str(weather_year) + '_data_no_H2_EV_Steel.csv',
        decimal='.', skiprows=0, parse_dates=['Date'])
    areaConsumption.dropna(inplace=True)
    areaConsumption["Date"] = pd.to_datetime(areaConsumption["Date"])
    areaConsumption.set_index(["AREAS", "Date"], inplace=True)


    #Import availibility factor data
    availabilityFactor = pd.read_csv(
        InputFolder + str(year) + "_" + str(weather_year) + '_Multinode_availability_factor.csv',
        decimal='.', skiprows=0, parse_dates=['Date'])
    availabilityFactor.loc[availabilityFactor.availabilityFactor > 1, "availabilityFactor"] = 1
    availabilityFactor.dropna(inplace=True)
    availabilityFactor["Date"]=pd.to_datetime(availabilityFactor["Date"])
    availabilityFactor.set_index(["AREAS", "Date", "TECHNOLOGIES"],inplace=True)
    #Import interconnections data
    ExchangeParameters = pd.read_csv(InputFolder + str(year) + '_interconnexions.csv', sep=";",
                                     decimal='.').set_index(["AREAS", "AREAS.1"])

    ###################
    # Data adjustment #
    ###################
    techs = TechParameters.TECHNOLOGIES.unique()
    areas = TechParameters.AREAS.unique()
    TechParameters.set_index(["AREAS", "TECHNOLOGIES"], inplace=True)
    if no_fossil_mini:
        for tech in ["Coal","Lignite","CCG","TAC"]:
            for area in areas:
                TechParameters.loc[(area,tech),"minCapacity"]=0
    if noREmax:
        for tech in ["Solar","WindOnShore","WindOffShore"]:
            for area in areas:
                TechParameters.loc[(area,tech),"maxCapacity"]=20*TechParameters.loc[(area,tech),"maxCapacity"]
    if noREmin:
        for tech in ["Solar","WindOnShore","WindOffShore"]:
            for area in areas:
                TechParameters.loc[(area,tech),"minCapacity"]=0
    #CHP inclusion
    areaConsumption=CHP_processing_future(areaConsumption,year,weather_year)

    #Flexibility data inclusion
    ConsoParameters_,labour_ratio,to_flex_consumption=Flexibility_data_processing(areaConsumption,2030,weather_year,no_flex)

    #Marginal cost adjustment and merit order simulation
    TechParameters=Marginal_cost_adjustment(TechParameters,number_of_sub_techs,techs,areas,carbon_tax,ctax_ini,gas_price_coef,coal_price_coef)

    #Curtailment adjustment
    TechParameters=Curtailment_adjustment(TechParameters,minCapacity=5000,maxCapacity=5000,EnergyNbHourCap=20)
    TechParameters.to_csv("test.csv")
    ##############################
    # Model creation and solving #
    ##############################
    end_time = datetime.now()
    print('\t Model creation at {}'.format(end_time - start_time))
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

    with open('Result files/Result_2030'+instance+'.pickle', 'wb') as f:
        pickle.dump(Variables, f, protocol=pickle.HIGHEST_PROTOCOL)

    end_time = datetime.now()
    print('\t Total duration: {}'.format(end_time - start_time))
    return Variables

for (ctax,gpc,cpc) in [(90,2.5,2.5),(30,1,1)]:
    for (no_fossil_mini,noREmax,noREmin) in [(False,False,False),(True,False,False),(True,True,False),(True,True,True)]:
        for no_flex in [True,False]:
            ldir=os.listdir("Result files")

            instance = "_ctax" + str(ctax) + "gpc" + str(gpc) + "cpc" + str(cpc)
            if no_fossil_mini:
                instance += "_no_fossil_mini"
            if no_flex:
                instance += "_no"
            instance += "_flex"
            if noREmax:
                instance += "_noREmax"
            if noREmin:
                instance += "_noREmin"

            if "Result_2030"+instance+".pickle" not in ldir:
                main_2030(weather_year=2018,carbon_tax=ctax,gas_price_coef=gpc,coal_price_coef=cpc,number_of_sub_techs=7,no_fossil_mini=no_fossil_mini,no_flex=no_flex,noREmax=noREmax,noREmin=noREmin)
            else:
                print("Instance "+instance+" already calculated")