#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 07:51:58 2020

@author: robin.girard
"""
from __future__ import division

from pyomo.core import *
from pyomo.opt import SolverFactory


from EnergyAlternativesPlanning.f_model_definition import *
from EnergyAlternativesPlanning.f_model_cost_functions import *
from EnergyAlternativesPlanning.f_model_planning_constraints import *
from EnergyAlternativesPlanning.f_model_operation_constraints import *

def GetElectricSystemModel_PlaningMultiNode(Parameters):
    """
    This function creates the pyomo model and initlize the Parameters and (pyomo) Set values
    :param areaConsumption: panda table with consumption
    :param availabilityFactor: panda table
    :param isAbstract: boolean true is the model should be abstract. ConcreteModel otherwise
    :return: pyomo model
    """
    # isAbstract=False

    Parameters["availabilityFactor"].isna().sum()

    ### Cleaning
    availabilityFactor=Parameters["availabilityFactor"].fillna(method='pad');
    areaConsumption=Parameters["areaConsumption"].fillna(method='pad');
    TechParameters = Parameters["TechParameters"]
    ExchangeParameters= Parameters["ExchangeParameters"]

    ### obtaining dimensions values
    TECHNOLOGIES = set(TechParameters.index.get_level_values('TECHNOLOGIES').unique())
    Date = set(areaConsumption.index.get_level_values('Date').unique())
    Date_list = areaConsumption.index.get_level_values('Date').unique()
    AREAS = set(areaConsumption.index.get_level_values('AREAS').unique())

    model = ConcreteModel()

    ###############
    # Sets       ##
    ###############

    # Simple
    model.AREAS= Set(initialize=AREAS ,doc = "Area" ,ordered=False)
    model.TECHNOLOGIES = Set(initialize=TECHNOLOGIES ,ordered=False)
    model.Date = Set(initialize=Date ,ordered=False)

    # Subset of Simple only required if ramp constraint
    model.Date_MinusOne = Set(initialize=Date_list[: len(Date) - 1] ,ordered=False)
    model.Date_MinusThree = Set(initialize=Date_list[: len(Date) - 3] ,ordered=False)

    # Products
    model.Date_TECHNOLOGIES =  model.Date *model.TECHNOLOGIES
    model.AREAS_AREAS = model.AREAS * model.AREAS
    model.AREAS_TECHNOLOGIES = model.AREAS * model.TECHNOLOGIES
    model.AREAS_Date = model.AREAS * model.Date
    model.AREAS_Date_TECHNOLOGIES = model.AREAS * model.Date * model.TECHNOLOGIES

    ###############
    # Parameters ##
    ###############

    model.areaConsumption = Param(model.AREAS_Date,
                                  initialize=areaConsumption.loc[:, "areaConsumption"].squeeze().to_dict(), domain=Any,
                                  mutable=True)
    model.availabilityFactor = Param(model.AREAS_Date_TECHNOLOGIES, domain=PercentFraction, default=1,
                                     initialize=availabilityFactor.squeeze().to_dict())

    model.maxExchangeCapacity = Param(model.AREAS_AREAS, initialize=ExchangeParameters.squeeze().to_dict(),
                                      domain=NonNegativeReals, default=0)
    # with test of existing columns on TechParameters
    for COLNAME in TechParameters:
        exec("model." + COLNAME + " =          Param(model.AREAS_TECHNOLOGIES, domain=NonNegativeReals,default=0," +
             "initialize=TechParameters." + COLNAME + ".squeeze().to_dict())")

    ################
    # Variables    #
    ################

    model.energy = Var(model.AREAS, model.Date, model.TECHNOLOGIES,
                       domain=NonNegativeReals)  ### Energy produced by a production mean at time t
    model.exchange = Var(model.AREAS_AREAS, model.Date)
    model.energyCosts = Var(model.AREAS,
                            model.TECHNOLOGIES)  ### Cost of energy by a production mean for area at time t (explicitely defined by constraint energyCostsCtr)

    model.capacityCosts = Var(model.AREAS,
                              model.TECHNOLOGIES)  ### Cost of energy for a production mean, explicitely defined by definition energyCostsDef
    model.capacity = Var(model.AREAS, model.TECHNOLOGIES,
                         domain=NonNegativeReals)  ### Energy produced by a production mean at time t

    model.dual = Suffix(direction=Suffix.IMPORT)
    model.rc = Suffix(direction=Suffix.IMPORT)

    # model.slack = Suffix(direction=Suffix.IMPORT)

    ########################
    # Objective Function   #
    ########################

    def ObjectiveFunction_rule(model):  # OBJ
        return sum(
            model.energyCosts[area, tech] + model.capacityCosts[area, tech] for tech in model.TECHNOLOGIES for area in
            model.AREAS)

    model.OBJ = Objective(rule=ObjectiveFunction_rule, sense=minimize)

    #################
    # Constraints   #
    #################

    #### 1 - Basics
    ########

    # Variables muettes : energyCosts,capacityCosts
    def energyCostsDef_rule(model, area,tech):  # EQ forall tech in TECHNOLOGIES   energyCosts  = sum{t in Date} energyCost[tech]*energy[t,tech] / 1E6;
        temp = model.energyCost[area, tech]  # /10**6 ;
        return sum(temp * model.energy[area, t, tech] for t in model.Date) == model.energyCosts[area, tech];
    model.energyCostsDef = Constraint(model.AREAS, model.TECHNOLOGIES, rule=energyCostsDef_rule)

    def capacityCostsDef_rule(model, area,
                              tech):  # EQ forall tech in TECHNOLOGIES   energyCosts  = sum{t in Date} energyCost[tech]*energy[t,tech] / 1E6;
        temp = model.capacityCosts[area, tech]  # /10**6 ;
        return model.capacityCost[area, tech] * len(model.Date) / 8760 * model.capacity[area, tech] == \
               model.capacityCosts[area, tech]  # .. ....... / 10**6
    model.capacityCostsCtr = Constraint(model.AREAS, model.TECHNOLOGIES, rule=capacityCostsDef_rule)

    # Exchange capacity constraint (duplicate of variable definition)
    # AREAS x AREAS x Date
    def exchangeCtrPlus_rule(model, a, b, t):  # INEQ forall area.axarea.b in AREASxAREAS  t in Date
        if a != b:
            return model.exchange[a, b, t] <= model.maxExchangeCapacity[a, b];
        else:
            return model.exchange[a, a, t] == 0
    model.exchangeCtrPlus = Constraint(model.AREAS, model.AREAS, model.Date, rule=exchangeCtrPlus_rule)

    def exchangeCtrMoins_rule(model, a, b, t):  # INEQ forall area.axarea.b in AREASxAREAS  t in Date
        if a != b:
            return model.exchange[a, b, t] >= -model.maxExchangeCapacity[a, b];
        else:
            return model.exchange[a, a, t] == 0
    model.exchangeCtrMoins = Constraint(model.AREAS, model.AREAS, model.Date, rule=exchangeCtrMoins_rule)

    def exchangeCtr2_rule(model, a, b, t):  # INEQ forall area.axarea.b in AREASxAREAS  t in Date
        return model.exchange[a, b, t] == -model.exchange[b, a, t];
    model.exchangeCtr2 = Constraint(model.AREAS, model.AREAS, model.Date, rule=exchangeCtr2_rule)

    # contrainte d'equilibre offre demande
    # AREAS x Date x TECHNOLOGIES
    def energyCtr_rule(model, area, t):  # INEQ forall t
        return sum(model.energy[area, t, tech] for tech in model.TECHNOLOGIES) + sum(
            model.exchange[b, area, t] for b in model.AREAS) == model.areaConsumption[area, t]
    model.energyCtr = Constraint(model.AREAS, model.Date, rule=energyCtr_rule)

    #other classical operation constraints
    model   =   set_Operation_Constraints_CapacityCtr(model)#energy <= capacity * availabilityFactor
    model   =   set_Operation_Constraints_stockCtr(model)
    model   =   set_Operation_Constraints_Ramp(model)

    #other classical planing constraints
    model = set_Planing_Constraints_maxCapacityCtr(model) #  model.maxCapacity[tech] >= model.capacity[tech]
    model = set_Planing_Constraints_minCapacityCtr(model) # model.minCapacity[tech] <= model.capacity[tech]

    return model;


def GetElectricSystemModel_PlaningMultiNode_withStorage(Parameters):
    """
    This function takes storage caracteristics, system caracteristics and optimise operation Set values
    :param areaConsumption: panda table with consumption
    :param availabilityFactor: panda table
    :param isAbstract: boolean true is the model should be abstract. ConcreteModel otherwise
    :param StorageParameters is a panda with p_max (maximal power), c_max (energy capacity in the storage : maximal energy),
    :efficiency_in (input efficiency of storage),
    :efficiency_out (output efficiency of storage).
    :return: a dictionary with model : pyomo model without storage, storage : storage infos
    """
    Parameters["availabilityFactor"].isna().sum()

    ### Cleaning
    availabilityFactor=Parameters["availabilityFactor"].fillna(method='pad');
    areaConsumption=Parameters["areaConsumption"].fillna(method='pad');
    TechParameters=Parameters["TechParameters"]
    StorageParameters=Parameters["StorageParameters"]
    ExchangeParameters=Parameters["ExchangeParameters"]

    ### obtaining dimensions values
    TECHNOLOGIES = set(TechParameters.index.get_level_values('TECHNOLOGIES').unique())
    STOCK_TECHNO = set(StorageParameters.index.get_level_values('STOCK_TECHNO').unique())
    Date = set(areaConsumption.index.get_level_values('Date').unique())
    Date_list = areaConsumption.index.get_level_values('Date').unique()
    AREAS = set(areaConsumption.index.get_level_values('AREAS').unique())

    model = ConcreteModel()

    ###############
    # Sets       ##
    ###############

    # Simple
    model.AREAS = Set(initialize=AREAS, doc="Area", ordered=False)
    model.TECHNOLOGIES = Set(initialize=TECHNOLOGIES, ordered=False)
    model.STOCK_TECHNO = Set(initialize=STOCK_TECHNO, ordered=False)
    model.Date = Set(initialize=Date, ordered=False)

    # Subset of Simple only required if ramp constraint
    model.Date_MinusOne = Set(initialize=Date_list[: len(Date) - 1], ordered=False)
    model.Date_MinusThree = Set(initialize=Date_list[: len(Date) - 3], ordered=False)

    # Products
    model.Date_TECHNOLOGIES = model.Date * model.TECHNOLOGIES
    model.AREAS_AREAS = model.AREAS * model.AREAS
    model.AREAS_TECHNOLOGIES = model.AREAS * model.TECHNOLOGIES
    model.AREAS_STOCKTECHNO = model.AREAS * model.STOCK_TECHNO
    model.AREAS_Date = model.AREAS * model.Date
    model.AREAS_Date_TECHNOLOGIES = model.AREAS * model.Date * model.TECHNOLOGIES

    ###############
    # Parameters ##
    ###############

    model.areaConsumption = Param(model.AREAS_Date,
                                  initialize=areaConsumption.loc[:, "areaConsumption"].squeeze().to_dict(), domain=Any,
                                  mutable=True)
    model.availabilityFactor = Param(model.AREAS_Date_TECHNOLOGIES, domain=PercentFraction, default=1,
                                     initialize=availabilityFactor.squeeze().to_dict())
    model.maxExchangeCapacity = Param(model.AREAS_AREAS, initialize=ExchangeParameters.squeeze().to_dict(),
                                      domain=NonNegativeReals, default=0)

    # with test of existing columns on TechParameters
    for COLNAME in TechParameters:
        exec("model." + COLNAME + " =          Param(model.AREAS_TECHNOLOGIES, domain=NonNegativeReals,default=0," +
             "initialize=TechParameters." + COLNAME + ".squeeze().to_dict())")

    for COLNAME in StorageParameters:
        exec("model." + COLNAME + " =Param(model.AREAS_STOCKTECHNO,domain=NonNegativeReals,default=0," +
             "initialize=StorageParameters." + COLNAME + ".squeeze().to_dict())")

    ################
    # Variables    #
    ################

    ###Production means :
    model.energy = Var(model.AREAS, model.Date, model.TECHNOLOGIES,domain=NonNegativeReals)  ### Energy produced by a production mean at time t
    model.exchange = Var(model.AREAS_AREAS, model.Date)
    model.energyCosts = Var(model.AREAS,model.TECHNOLOGIES)  ### Cost of energy by a production mean for area at time t (explicitely defined by constraint energyCostsCtr)
    model.capacityCosts = Var(model.AREAS,model.TECHNOLOGIES)  ### Cost of energy for a production mean, explicitely defined by definition energyCostsDef
    model.capacity = Var(model.AREAS, model.TECHNOLOGIES,domain=NonNegativeReals)  ### Energy produced by a production mean at time t

    ###Storage means :
    model.storageIn = Var(model.AREAS, model.Date, model.STOCK_TECHNO,domain=NonNegativeReals)  ### Energy stored by a storage mean for areas at time t
    model.storageOut = Var(model.AREAS, model.Date, model.STOCK_TECHNO,domain=NonNegativeReals)  ### Energy taken out of a storage mean for areas at time t
    model.stockLevel = Var(model.AREAS, model.Date, model.STOCK_TECHNO,domain=NonNegativeReals)  ### level of the energy stock in a storage mean at time t
    model.storageCosts = Var(model.AREAS,model.STOCK_TECHNO)  ### Cost of storage for a storage mean, explicitely defined by definition storageCostsDef
    model.Cmax = Var(model.AREAS, model.STOCK_TECHNO)  # Maximum capacity of a storage mean
    model.Pmax = Var(model.AREAS, model.STOCK_TECHNO)  # Maximum flow of energy in/out of a storage mean
    model.stockLevel_ini=Var(model.AREAS,model.STOCK_TECHNO,domain=NonNegativeReals)

    model.dual = Suffix(direction=Suffix.IMPORT)
    model.rc = Suffix(direction=Suffix.IMPORT)

    # model.slack = Suffix(direction=Suffix.IMPORT)

    ########################
    # Objective Function   #
    ########################

    def ObjectiveFunction_rule(model):  # OBJ
        return sum(
            model.energyCosts[area, tech] + model.capacityCosts[area, tech] for tech in model.TECHNOLOGIES for area in
            model.AREAS) + sum(
            model.storageCosts[area, s_tech] for s_tech in model.STOCK_TECHNO for area in model.AREAS)
    model.OBJ = Objective(rule=ObjectiveFunction_rule, sense=minimize)

    #################
    # Constraints   #
    #################

    # Variable muette : energyCosts and capacityCosts, storageCosts
    def energyCostsDef_rule(model, area,
                            tech):  # EQ forall tech in TECHNOLOGIES   energyCosts  = sum{t in Date} energyCost[tech]*energy[t,tech] / 1E6;
        temp = model.energyCost[area, tech]  # /10**6 ;
        return sum(temp * model.energy[area, t, tech] for t in model.Date) == model.energyCosts[area, tech];
    model.energyCostsDef = Constraint(model.AREAS, model.TECHNOLOGIES, rule=energyCostsDef_rule)

    def capacityCostsDef_rule(model, area,
                              tech):  # EQ forall tech in TECHNOLOGIES   energyCosts  = sum{t in Date} energyCost[tech]*energy[t,tech] / 1E6;
        temp = model.capacityCosts[area, tech]  # /10**6 ;
        return model.capacityCost[area, tech] * len(model.Date) / 8760 * model.capacity[area, tech] == \
               model.capacityCosts[area, tech]  # .. ....... / 10**6
    model.capacityCostsCtr = Constraint(model.AREAS, model.TECHNOLOGIES, rule=capacityCostsDef_rule)

    def storageCostsDef_rule(model, area,s_tech):  # EQ forall s_tech in STOCK_TECHNO storageCosts = storageCost[area, s_tech]*c_max[area, s_tech] / 1E6;
        return model.storageCost[area, s_tech] * model.Cmax[area, s_tech] == model.storageCosts[area, s_tech]  # /10**6 ;;
    model.storageCostsDef = Constraint(model.AREAS, model.STOCK_TECHNO, rule=storageCostsDef_rule)

    # Exchange capacity constraint (duplicate of variable definition)
    def exchangeCtrPlus_rule(model, a, b, t):  # INEQ forall area.axarea.b in AREASxAREAS  t in Date
        if a != b:
            return model.exchange[a, b, t] <= model.maxExchangeCapacity[a, b];
        else:
            return model.exchange[a, a, t] == 0
    model.exchangeCtrPlus = Constraint(model.AREAS, model.AREAS, model.Date, rule=exchangeCtrPlus_rule)

    def exchangeCtrMoins_rule(model, a, b, t):  # INEQ forall area.axarea.b in AREASxAREAS  t in Date
        if a != b:
            return model.exchange[a, b, t] >= -model.maxExchangeCapacity[a, b];
        else:
            return model.exchange[a, a, t] == 0
    model.exchangeCtrMoins = Constraint(model.AREAS, model.AREAS, model.Date, rule=exchangeCtrMoins_rule)

    def exchangeCtr2_rule(model, a, b, t):  # INEQ forall area.axarea.b in AREASxAREAS  t in Date
        return model.exchange[a, b, t] == -model.exchange[b, a, t];
    model.exchangeCtr2 = Constraint(model.AREAS, model.AREAS, model.Date, rule=exchangeCtr2_rule)


    # contrainte d'equilibre offre demande [AREAS x Date x TECHNOLOGIES]
    def energyCtr_rule(model, area, t):  # INEQ forall t
        return sum(model.energy[area, t, tech] for tech in model.TECHNOLOGIES) + sum(
            model.exchange[b, area, t] for b in model.AREAS) + sum(
            model.storageOut[area, t, s_tech] - model.storageIn[area, t, s_tech] for s_tech in model.STOCK_TECHNO) == \
               model.areaConsumption[area, t]
    model.energyCtr = Constraint(model.AREAS,model.Date, rule=energyCtr_rule)

    #other classical operation constraints
    model   =   set_Operation_Constraints_CapacityCtr(model)#energy <= capacity * availabilityFactor
    model   =   set_Operation_Constraints_stockCtr(model)
    model   =   set_Operation_Constraints_Ramp(model)
    model   =   set_Operation_Constraints_Storage(model)

    #other classical planing constraints
    model   =   set_Planing_Constraints_maxCapacityCtr(model) #  model.maxCapacity[tech] >= model.capacity[tech]
    model   =   set_Planing_Constraints_minCapacityCtr(model) # model.minCapacity[tech] <= model.capacity[tech]
    model   =   set_Planing_Constraints_storageCapacityCtr(model)
    model   =   set_Planing_Constraints_storagePowerCtr(model)

    return model;