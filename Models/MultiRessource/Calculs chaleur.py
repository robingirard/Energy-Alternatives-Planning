#region importation of modules
import os
if os.path.basename(os.getcwd())=="BasicFunctionalities":
    os.chdir('../../../../../..') ## to work at project root  like in any IDE
import sys
if sys.platform != 'win32':
    myhost = os.uname()[1]
else : myhost = ""
if (myhost=="jupyter-sop"):
    ## for https://jupyter-sop.mines-paristech.fr/ users, you need to
    #  (1) run the following in a terminal
    if (os.system("/opt/mosek/9.2/tools/platform/linux64x86/bin/lmgrd -c /opt/mosek/9.2/tools/platform/linux64x86/bin/mosek.lic -l lmgrd.log")==0):
        os.system("/opt/mosek/9.2/tools/platform/linux64x86/bin/lmutil lmstat -c 27007@127.0.0.1 -a")
    #  (2) definition of license
    os.environ["MOSEKLM_LICENSE_FILE"] = '@jupyter-sop'

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
from sklearn import linear_model
import sys
import time
import datetime
import seaborn as sb

from functions.anaelle.f_multiResourceModels import *
from functions.anaelle.f_optimization import *
from functions.anaelle.f_graphicalTools import *
from functions.anaelle.f_optimModel_elec import *
from functions.anaelle.f_InputScenario import *

# Change this if you have other solvers obtained here
## https://ampl.com/products/solvers/open-source/
## for eduction this site provides also several professional solvers, that are more efficient than e.g. cbc
#endregion

#region Solver and data location definition

InputFolder='Models/MultiRessource/Data/Input_chaleur/'
OutputFolder='Models/MultiRessource/Data/Output_chaleur/'
d=datetime.date.today()

solver= 'mosek' ## no need for solverpath with mosek.
BaseSolverPath='/Users/robin.girard/Documents/Code/Packages/solvers/ampl_macosx64'
sys.path.append(BaseSolverPath)

#endregion

#region Creation Data

availabilityFactor=pd.read_csv(InputFolder + 'availabilityFactor2013.csv',sep=',', decimal='.', skiprows=0).set_index(["TIMESTAMP", "TECHNOLOGIES"])
ResParameters=pd.read_csv(InputFolder + 'elec_prices2013.csv',sep=',', decimal='.', skiprows=0).set_index(['TIMESTAMP','RESOURCES'])
ResParameters.loc[(slice(None),"electricity"),:]= ResParameters.loc[(slice(None),"electricity"),:]+30
areaConsumption=pd.read_csv(InputFolder + 'areaConsumption2013.csv',sep=',', decimal='.', skiprows=0).set_index(['TIMESTAMP','RESOURCES'])
TechParameters=pd.DataFrame([['Wind',0,1200000,45000,30,0,0,10000],
                             ['Solar',0,700000,15000,20,0,0,10000],
                             ['PAC',0,600000,36000,20,0,0,10000]],
            columns=['TECHNOLOGIES','powerCost','investCost','operationCost','lifeSpan','EmissionCO2','minCapacity','maxCapacity']).set_index('TECHNOLOGIES')
StorageParameters=pd.DataFrame([['Battery','electricity',223600,239940,4472,15,100,1000],
                                ['Heat_storage','heat',30000,20000,650,20,10000,100000]],
            columns=['STOCK_TECHNO','resource','storagePowerInvestCost','storageEnergyInvestCost','storageOperationCost','storagelifeSpan','p_max','c_max']).set_index('STOCK_TECHNO')
ConversionFactors=pd.DataFrame([['electricity','Wind',1],
                                ['electricity','Solar',1],
                                ['electricity','PAC',-0.33],
                                ['heat','Wind',0],
                                ['heat','Solar',0],['heat','PAC',1]],columns=['RESOURCES','TECHNOLOGIES','conversionFactor']).set_index(['RESOURCES','TECHNOLOGIES'])
StorageFactors=pd.DataFrame([['electricity','Battery',0.92,0.0085,1.09],
                             ['electricity','Heat_storage',0,0,0],
                             ['heat','Battery',0,0,0],
                             ['heat','Heat_storage',1,0.004,1]],columns=['RESOURCES','STOCK_TECHNO','storageFactorIn','dissipation','storageFactorOut']).set_index(['RESOURCES','STOCK_TECHNO'])

Param_list={'availabilityFactor': availabilityFactor,'ResParameters': ResParameters,'areaConsumption':areaConsumption,'TechParameters':TechParameters,'StorageParameters':StorageParameters,'ConversionFactors':ConversionFactors,'StorageFactors':StorageFactors}

#endregion

def GetElectricSystemModel_MultiResources_SingleNode_WithStorage(Param_list, isAbstract=False):
    """
    This function creates the pyomo model and initlize the Parameters and (pyomo) Set values
    :param areaConsumption: panda table with consumption
    :param availabilityFactor: panda table
    :param isAbstract: boolean true is the model should be abstract. ConcreteModel otherwise
    :return: pyomo model
    """
    isAbstract = False
    Param_list['availabilityFactor'].isna().sum()

    ### Cleaning
    Param_list['availabilityFactor'] = Param_list['availabilityFactor'] .fillna(method='pad');
    Param_list['ResParameters'] = Param_list['ResParameters'].fillna(method='pad');
    Param_list['areaConsumption'] = Param_list['areaConsumption'] .fillna(method='pad');

    ### obtaining dimensions values

    TECHNOLOGIES = set(Param_list['TechParameters'].index.get_level_values('TECHNOLOGIES').unique())
    STOCK_TECHNO = set(Param_list['StorageParameters'].index.get_level_values('STOCK_TECHNO').unique())
    RESOURCES = set(Param_list['ResParameters'].index.get_level_values('RESOURCES').unique())
    TIMESTAMP = set(Param_list['areaConsumption'].index.get_level_values('TIMESTAMP').unique())
    TIMESTAMP_list = Param_list['areaConsumption'].index.get_level_values('TIMESTAMP').unique()

    #####################
    #    Pyomo model    #
    #####################

    if (isAbstract):
        model = pyomo.environ.AbstractModel()
    else:
        model = pyomo.environ.ConcreteModel()

    ###############
    # Sets       ##
    ###############
    model.TECHNOLOGIES = Set(initialize=TECHNOLOGIES, ordered=False)
    model.STOCK_TECHNO = Set(initialize=STOCK_TECHNO, ordered=False)
    model.RESOURCES = Set(initialize=RESOURCES, ordered=False)
    model.TIMESTAMP = Set(initialize=TIMESTAMP, ordered=False)
    model.TIMESTAMP_TECHNOLOGIES = model.TIMESTAMP * model.TECHNOLOGIES
    model.RESOURCES_TECHNOLOGIES = model.RESOURCES * model.TECHNOLOGIES
    model.RESOURCES_STOCKTECHNO = model.RESOURCES * model.STOCK_TECHNO
    model.TIMESTAMP_RESOURCES = model.TIMESTAMP * model.RESOURCES

    # Subset of Simple only required if ramp constraint
    model.TIMESTAMP_MinusOne = Set(initialize=TIMESTAMP_list[: len(TIMESTAMP) - 1], ordered=False)
    model.TIMESTAMP_MinusThree = Set(initialize=TIMESTAMP_list[: len(TIMESTAMP) - 3], ordered=False)

    ###############
    # Parameters ##
    ###############

    model.areaConsumption = Param(model.TIMESTAMP_RESOURCES, default=0,initialize=Param_list['areaConsumption'].loc[:, "areaConsumption"].squeeze().to_dict(), domain=Any)
    model.availabilityFactor = Param(model.TIMESTAMP_TECHNOLOGIES, domain=PercentFraction, default=1,initialize=Param_list['availabilityFactor'].loc[:, "availabilityFactor"].squeeze().to_dict())
    model.conversionFactor = Param(model.RESOURCES_TECHNOLOGIES, default=0,initialize=Param_list['ConversionFactors'].loc[:, "conversionFactor"].squeeze().to_dict())
    model.importCost = Param(model.TIMESTAMP_RESOURCES, default=0,initialize=Param_list['ResParameters'].loc[:, "importCost"].squeeze().to_dict(), domain=Any)

    # with test of existing columns on TechParameters
    for COLNAME in Param_list['TechParameters']:
        if COLNAME not in ["TECHNOLOGIES", "AREAS"]:  ### each column in TechParameters will be a parameter
            exec("model." + COLNAME + " =          Param(model.TECHNOLOGIES, domain=NonNegativeReals,default=0," +
                 "initialize=TechParameters." + COLNAME + ".squeeze().to_dict())")

    for COLNAME in Param_list['StorageParameters']:
        if COLNAME not in ["STOCK_TECHNO", "AREAS"]:  ### each column in StorageParameters will be a parameter
            exec("model." + COLNAME + " =Param(model.STOCK_TECHNO,domain=Any,default=0," +
                 "initialize=StorageParameters." + COLNAME + ".squeeze().to_dict())")

    for COLNAME in Param_list['StorageFactors']:
        exec("model." + COLNAME + " =Param(model.RESOURCES_STOCKTECHNO,domain=NonNegativeReals,default=0," +
             "initialize=StorageFactors." + COLNAME + ".squeeze().to_dict())")

    ################
    # Variables    #
    ################

    # In this section, variables are separated in two categories : decision variables wich are the reals variables of the otimisation problem (these are noted Dvar), and problem variables which are resulting of calculation and are convenient for the readability and the analyse of results (these are noted Pvar)

    ### Operation variables
    model.power_Dvar = Var(model.TIMESTAMP, model.TECHNOLOGIES,domain=NonNegativeReals)  ### Power of a conversion mean at time t
    model.powerCosts_Pvar = Var(model.TECHNOLOGIES)  ### Marginal cost for a conversion mean, explicitely defined by definition powerCostsDef
    model.importation_Dvar = Var(model.TIMESTAMP, model.RESOURCES, domain=NonNegativeReals,initialize=0)  ### Improtation of a resource at time t
    model.importCosts_Pvar = Var(model.RESOURCES)  ### Cost of ressource imported, explicitely defined by definition importCostsDef
    model.energy_Pvar = Var(model.TIMESTAMP, model.RESOURCES,domain=NonNegativeReals)  ### Amount of a resource at time t

    ### Planing variables
    model.capacity_Dvar = Var(model.TECHNOLOGIES, domain=NonNegativeReals)  ### Capacity of a conversion mean
    model.capacityCosts_Pvar = Var(model.TECHNOLOGIES)  ### Fixed costs for a conversion mean, explicitely defined by definition capacityCostsDef

    ### Storage variables
    model.storageIn_Pvar = Var(model.TIMESTAMP, model.RESOURCES, model.STOCK_TECHNO,domain=NonNegativeReals)  ### Energy stored in a storage mean at time t
    model.storageOut_Pvar = Var(model.TIMESTAMP, model.RESOURCES, model.STOCK_TECHNO,domain=NonNegativeReals)  ### Energy taken out of the in a storage mean at time t
    model.storageConsumption_Pvar = Var(model.TIMESTAMP, model.RESOURCES, model.STOCK_TECHNO,domain=NonNegativeReals)  ### Energy consumed the in a storage mean at time t (other than the one stored)
    model.stockLevel_Pvar = Var(model.TIMESTAMP, model.STOCK_TECHNO,domain=NonNegativeReals)  ### level of the energy stock in a storage mean at time t
    model.storageCosts_Pvar = Var(model.STOCK_TECHNO)  ### Cost of storage for a storage mean, explicitely defined by definition storageCostsDef
    model.Cmax_Dvar = Var(model.STOCK_TECHNO)  # Maximum capacity of a storage mean
    model.Pmax_Dvar = Var(model.STOCK_TECHNO)  # Maximum flow of energy in/out of a storage mean

    ### Other variables
    model.dual = Suffix(direction=Suffix.IMPORT)
    model.rc = Suffix(direction=Suffix.IMPORT)
    model.slack = Suffix(direction=Suffix.IMPORT)

    ########################
    # Objective Function   #
    ########################

    def ObjectiveFunction_rule(model):  # OBJ
        return (sum(model.powerCosts_Pvar[tech] + model.capacityCosts_Pvar[tech] for tech in model.TECHNOLOGIES) + sum(
            model.importCosts_Pvar[res] for res in model.RESOURCES)) + sum(
            model.storageCosts_Pvar[s_tech] for s_tech in STOCK_TECHNO)
    model.OBJ = Objective(rule=ObjectiveFunction_rule, sense=minimize)

    #################
    # Constraints   #
    #################

    # powerCosts definition Constraint
    def powerCostsDef_rule(model,tech):  # EQ forall tech in TECHNOLOGIES powerCosts  = sum{t in TIMESTAMP} powerCost[tech]*power[t,tech] / 1E6;
        return sum(model.powerCost[tech] * model.power_Dvar[t, tech] for t in model.TIMESTAMP) == model.powerCosts_Pvar[
            tech]
    model.powerCostsCtr = Constraint(model.TECHNOLOGIES, rule=powerCostsDef_rule)

    # capacityCosts definition Constraint
    def capacityCostsDef_rule(model,tech):  # EQ forall tech in TECHNOLOGIES   capacityCosts  = sum{t in TIMESTAMP} capacityCost[tech]*capacity[t,tech] / 1E6;
        r = 0.04
        factor1 = r / ((1 + r) * (1 - (1 + r) ** - model.lifeSpan[ tech]))
        factor2 = (1 + r) ** (-10)
        return (model.investCost[ tech] * factor1 * factor2 + model.operationCost[tech] * factor2) * model.capacity_Dvar[tech] == model.capacityCosts_Pvar[tech]
    model.capacityCostsCtr = Constraint(model.TECHNOLOGIES, rule=capacityCostsDef_rule)

    # importCosts definition Constraint
    def importCostsDef_rule(model, res):  # ;
        return sum((model.importCost[t, res] * model.importation_Dvar[t, res]) for t in model.TIMESTAMP) == model.importCosts_Pvar[res]
    model.importCostsCtr = Constraint(model.RESOURCES, rule=importCostsDef_rule)

    # storageCosts definition Constraint
    def storageCostsDef_rule(model,s_tech):  # EQ forall s_tech in STOCK_TECHNO storageCosts=storageCost[s_tech]*Cmax[s_tech] / 1E6;
        r=0.04
        factor1=r/((1+r)*(1-(1+r)**-model.storagelifeSpan[s_tech]))
        factor2=(1+r)**(-10)
        return (model.storageEnergyInvestCost[s_tech] * model.Cmax_Dvar[s_tech] + model.storagePowerInvestCost[s_tech] * model.Pmax_Dvar[s_tech]) * factor1 * factor2 + model.storageOperationCost[s_tech]*factor2* model.Pmax_Dvar[s_tech]  == model.storageCosts_Pvar[s_tech]
    model.storageCostsCtr = Constraint(model.STOCK_TECHNO, rule=storageCostsDef_rule)

    # Storage max capacity constraint
    def storageCapacity_rule(model, s_tech):  # INEQ forall s_tech
        return model.Cmax_Dvar[s_tech] <= model.c_max[s_tech]
    model.storageCapacityCtr = Constraint(model.STOCK_TECHNO, rule=storageCapacity_rule)

    # Storage max power constraint
    def storagePower_rule(model, s_tech):  # INEQ forall s_tech
        return model.Pmax_Dvar[s_tech] <= model.p_max[s_tech]
    model.storagePowerCtr = Constraint(model.STOCK_TECHNO, rule=storagePower_rule)

    # contraintes de stock puissance
    def StoragePowerUB_rule(model, t, res, s_tech):  # INEQ forall t
        if res == model.resource[s_tech]:
            return model.storageIn_Pvar[t, res, s_tech] - model.Pmax_Dvar[s_tech] <= 0
        else:
            return model.storageIn_Pvar[t, res, s_tech] == 0
    model.StoragePowerUBCtr = Constraint(model.TIMESTAMP, model.RESOURCES, model.STOCK_TECHNO, rule=StoragePowerUB_rule)

    def StoragePowerLB_rule(model, t, res, s_tech, ):  # INEQ forall t
        if res == model.resource[s_tech]:
            return model.storageOut_Pvar[t, res, s_tech] - model.Pmax_Dvar[s_tech] <= 0
        else:
            return model.storageOut_Pvar[t, res, s_tech] == 0
    model.StoragePowerLBCtr = Constraint(model.TIMESTAMP, model.RESOURCES, model.STOCK_TECHNO, rule=StoragePowerLB_rule)

    # contrainte de consommation du stockage (autre que l'énergie stockée)
    def StorageConsumption_rule(model, t, res, s_tech):  # EQ forall t
        temp = model.resource[s_tech]
        if res == temp:
            return model.storageConsumption_Pvar[t, res, s_tech] == 0
        else:
            return model.storageConsumption_Pvar[t, res, s_tech] == model.storageFactorIn[res, s_tech] * \
                   model.storageIn_Pvar[t, temp, s_tech] + model.storageFactorOut[res, s_tech] * model.storageOut_Pvar[
                       t, temp, s_tech]
    model.StorageConsumptionCtr = Constraint(model.TIMESTAMP, model.RESOURCES, model.STOCK_TECHNO,
                                             rule=StorageConsumption_rule)

    # contraintes de stock capacité
    def StockLevel_rule(model, t, s_tech):  # EQ forall t
        res = model.resource[s_tech]
        if t > 1:
            return model.stockLevel_Pvar[t, s_tech] == model.stockLevel_Pvar[t - 1, s_tech] * (
                        1 - model.dissipation[res, s_tech]) + model.storageIn_Pvar[t, res, s_tech] * \
                   model.storageFactorIn[res, s_tech] - model.storageOut_Pvar[t, res, s_tech] * model.storageFactorOut[
                       res, s_tech]
        else:
            return model.stockLevel_Pvar[t, s_tech] == model.storageIn_Pvar[t, res, s_tech] * model.storageFactorIn[
                res, s_tech] - model.storageOut_Pvar[t, res, s_tech] * model.storageFactorOut[res, s_tech]
    model.StockLevelCtr = Constraint(model.TIMESTAMP, model.STOCK_TECHNO, rule=StockLevel_rule)

    def StockCapacity_rule(model, t, s_tech, ):  # INEQ forall t
        return model.stockLevel_Pvar[t, s_tech] <= model.Cmax_Dvar[s_tech]
    model.StockCapacityCtr = Constraint(model.TIMESTAMP, model.STOCK_TECHNO, rule=StockCapacity_rule)

    # Capacity constraint
    def Capacity_rule(model, t, tech):  # INEQ forall t, tech
        return model.capacity_Dvar[tech] * model.availabilityFactor[t, tech] >= model.power_Dvar[t, tech]
    model.CapacityCtr = Constraint(model.TIMESTAMP, model.TECHNOLOGIES, rule=Capacity_rule)

    # Ressource production constraint
    def Production_rule(model, t, res):  # EQ forall t, res
        return sum(model.power_Dvar[t, tech] * model.conversionFactor[res, tech] for tech in model.TECHNOLOGIES) + \
               model.importation_Dvar[t, res] + sum(
            model.storageOut_Pvar[t, res, s_tech] - model.storageIn_Pvar[t, res, s_tech] -
            model.storageConsumption_Pvar[t, res, s_tech] for s_tech in STOCK_TECHNO) == model.energy_Pvar[t, res]
    model.ProductionCtr = Constraint(model.TIMESTAMP, model.RESOURCES, rule=Production_rule)

    # contrainte d'equilibre offre demande
    def energyCtr_rule(model, t, res):  # INEQ forall t
        return model.energy_Pvar[t, res] == model.areaConsumption[t, res]
    model.energyCtr = Constraint(model.TIMESTAMP, model.RESOURCES, rule=energyCtr_rule)

    if "maxCapacity" in TechParameters:
        def maxCapacity_rule(model, tech):  # INEQ forall t, tech
            return model.maxCapacity[tech] >= model.capacity_Dvar[tech]
        model.maxCapacityCtr = Constraint(model.TECHNOLOGIES, rule=maxCapacity_rule)

    if "minCapacity" in TechParameters:
        def minCapacity_rule(model, tech):  # INEQ forall t, tech
            return model.minCapacity[tech] <= model.capacity_Dvar[tech]
        model.minCapacityCtr = Constraint(model.TECHNOLOGIES, rule=minCapacity_rule)

    if "EnergyNbhourCap" in TechParameters:
        def storage_rule(model, tech):  # INEQ forall t, tech
            if model.EnergyNbhourCap[tech] > 0:
                return model.EnergyNbhourCap[tech] * model.capacity_Dvar[tech] >= sum(
                    model.power_Dvar[t, tech] for t in model.TIMESTAMP)
            else:
                return Constraint.Skip
        model.storageCtr = Constraint(model.TECHNOLOGIES, rule=storage_rule)

    if "RampConstraintPlus" in TechParameters:
        def rampCtrPlus_rule(model, t, tech):  # INEQ forall t<
            if model.RampConstraintPlus[tech] > 0:
                return model.power_Dvar[t + 1, tech] - model.power_Dvar[t, tech] <= model.capacity_Dvar[tech] * \
                       model.RampConstraintPlus[tech];
            else:
                return Constraint.Skip
        model.rampCtrPlus = Constraint(model.TIMESTAMP_MinusOne, model.TECHNOLOGIES, rule=rampCtrPlus_rule)

    if "RampConstraintMoins" in TechParameters:
        def rampCtrMoins_rule(model, t, tech):  # INEQ forall t<
            if model.RampConstraintMoins[tech] > 0:
                return model.power_Dvar[t + 1, tech] - model.power_Dvar[t, tech] >= - model.capacity_Dvar[tech] * \
                       model.RampConstraintMoins[tech];
            else:
                return Constraint.Skip
        model.rampCtrMoins = Constraint(model.TIMESTAMP_MinusOne, model.TECHNOLOGIES, rule=rampCtrMoins_rule)

    if "RampConstraintPlus2" in TechParameters:
        def rampCtrPlus2_rule(model, t, tech):  # INEQ forall t<
            if model.RampConstraintPlus2[tech] > 0:
                var = (model.power_Dvar[t + 2, tech] + model.power_Dvar[t + 3, tech]) / 2 - (
                            model.power_Dvar[t + 1, tech] + model.power_Dvar[t, tech]) / 2;
                return var <= model.capacity_Dvar[tech] * model.RampConstraintPlus[tech];
            else:
                return Constraint.Skip
        model.rampCtrPlus2 = Constraint(model.TIMESTAMP_MinusThree, model.TECHNOLOGIES, rule=rampCtrPlus2_rule)

    if "RampConstraintMoins2" in TechParameters:
        def rampCtrMoins2_rule(model, t, tech):  # INEQ forall t<
            if model.RampConstraintMoins2[tech] > 0:
                var = (model.power_Dvar[t + 2, tech] + model.power_Dvar[t + 3, tech]) / 2 - (
                            model.power_Dvar[t + 1, tech] + model.power_Dvar[t, tech]) / 2;
                return var >= - model.capacity_Dvar[tech] * model.RampConstraintMoins2[tech];
            else:
                return Constraint.Skip
        model.rampCtrMoins2 = Constraint(model.TIMESTAMP_MinusThree, model.TECHNOLOGIES, rule=rampCtrMoins2_rule)

    return model;


#region Solve

model = GetElectricSystemModel_MultiResources_SingleNode_WithStorage(Param_list, isAbstract=False)

start_clock=time.time()
opt = SolverFactory(solver)
results=opt.solve(model)
end_clock=time.time()
Clock=end_clock-start_clock
print('temps de calcul = ',Clock, 's')

# result analysis
Variables=getVariables_panda_indexed(model)
Constraints= getConstraintsDual_panda(model)

Variables.keys()
Variables["importation_Dvar"]=Variables["importation_Dvar"].set_index(["TIMESTAMP","RESOURCES"])
Variables['power_Dvar']=Variables['power_Dvar'].set_index(["TIMESTAMP","TECHNOLOGIES"])
Variables["capacity_Dvar"]

Import_Elec = Variables["importation_Dvar"].loc[(slice(None),"electricity"),:].\
    reset_index()[["TIMESTAMP","importation_Dvar"]].set_index(["TIMESTAMP"]).rename(columns = {"importation_Dvar" :"power_Dvar" })
PAC = (Variables['power_Dvar'].loc[(slice(None),"PAC"),:]*0.33).reset_index()[["TIMESTAMP","power_Dvar"]].set_index(["TIMESTAMP"])
Wind = (Variables['power_Dvar'].loc[(slice(None),"Wind"),:]).reset_index()[["TIMESTAMP","power_Dvar"]].set_index(["TIMESTAMP"])
Solar = (Variables['power_Dvar'].loc[(slice(None),"Solar"),:]).reset_index()[["TIMESTAMP","power_Dvar"]].set_index(["TIMESTAMP"])
(PAC-(Wind+Solar+Import_Elec))
elec_frame = {"Wind" : Wind.power_Dvar, "Solar" : Solar.power_Dvar, "Import_Elec": Import_Elec.power_Dvar}
Elec = pd.DataFrame(elec_frame)
Elec.sum(axis=0)

Variables['Cmax_Dvar']
Variables['Pmax_Dvar']




#endregion