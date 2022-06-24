#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 19:07:50 2020

@author: robin.girard
"""

from __future__ import division
from datetime import timedelta
import pandas as pd
import pyomo.environ
from pyomo.environ import *
from pyomo.core import *
from functions.f_tools import *
from functions.f_model_definition import *
from functions.f_model_operation_constraints import *

def GetElectricSystemModel_GestionSingleNode(Parameters):
    """
    This function creates the pyomo model and initlize the Parameters and (pyomo) Set values
    :param Parameters is a dictionnary with different panda tables :
        - areaConsumption: panda table with consumption
        - availabilityFactor: panda table
        - TechParameters : panda tables indexed by TECHNOLOGIES with eco and tech parameters
    """
    #isAbstract=False
    Parameters["availabilityFactor"].isna().sum()

    ### Cleaning
    availabilityFactor=Parameters["availabilityFactor"].fillna(method='pad');
    areaConsumption=Parameters["areaConsumption"].fillna(method='pad');
    TechParameters=Parameters["TechParameters"]
    ### obtaining dimensions values

    TECHNOLOGIES=   set(TechParameters.index.get_level_values('TECHNOLOGIES').unique())
    Date = set(areaConsumption.index.get_level_values('Date').unique())
    Date_list=areaConsumption.index.get_level_values('Date').unique()

    model = pyomo.environ.ConcreteModel()

    ###############
    # Sets       ##
    ###############
    #model.Date_vals = Date_vals
    model.TECHNOLOGIES  = Set(initialize=TECHNOLOGIES,ordered=False)
    model.Date     = Set(initialize=Date,ordered=False)
    model.Date_TECHNOLOGIES =  model.Date * model.TECHNOLOGIES

    #Subset of Simple only required if ramp constraint
    model.Date_MinusOne = Set(initialize=Date_list[: len(Date) - 1],ordered=False)
    model.Date_MinusThree = Set(initialize=Date_list[: len(Date) - 3],ordered=False)

    ###############
    # Parameters ##
    ###############

    model.areaConsumption =     Param(model.Date, mutable=True,
                                      initialize=areaConsumption.loc[:,"areaConsumption"].squeeze().to_dict(), domain=Any)
    model.availabilityFactor =  Param( model.Date_TECHNOLOGIES, domain=PercentFraction,default=1,
                                      initialize=availabilityFactor.loc[:,"availabilityFactor"].squeeze().to_dict())

    #with test of existing columns on TechParameters
    for COLNAME in TechParameters:
        if COLNAME not in ["TECHNOLOGIES","AREAS"]: ### each column in TechParameters will be a parameter
            exec("model."+COLNAME+" =          Param(model.TECHNOLOGIES, domain=NonNegativeReals,default=0,"+
                                      "initialize=TechParameters."+COLNAME+".squeeze().to_dict())")

    ################
    # Variables    #
    ################

    model.energy=Var(model.Date, model.TECHNOLOGIES, domain=NonNegativeReals) ### Energy produced by a production mean at time t
    model.energyCosts=Var(model.TECHNOLOGIES)  ### Cost of energy for a production mean, explicitely defined by definition energyCostsDef
    model.dual = Suffix(direction=Suffix.IMPORT)
    model.rc = Suffix(direction=Suffix.IMPORT)
    #model.slack = Suffix(direction=Suffix.IMPORT)

    ########################
    # Objective Function   #
    ########################

    def ObjectiveFunction_rule(model): #OBJ
    	return sum(model.energyCosts[tech] for tech in model.TECHNOLOGIES)
    model.OBJ = Objective(rule=ObjectiveFunction_rule, sense=minimize)

    #################
    # Constraints   #
    #################

    # energyCosts definition Constraints
    def energyCostsDef_rule(model,tech): #EQ forall tech in TECHNOLOGIES   energyCosts  = sum{t in Date} energyCost[tech]*energy[t,tech] / 1E6;
        temp=model.energyCost[tech]# /10**6 ;
        return sum(temp*model.energy[t,tech] for t in model.Date) == model.energyCosts[tech]
    model.energyCostsCtr = Constraint(model.TECHNOLOGIES, rule=energyCostsDef_rule)

    #contrainte d'equilibre offre demande
    def energyCtr_rule(model,t): #INEQ forall t
    	return sum(model.energy[t,tech] for tech in model.TECHNOLOGIES ) == model.areaConsumption[t]
    model.energyCtr = Constraint(model.Date,rule=energyCtr_rule)

    model   =   set_Operation_Constraints_CapacityCtr(model) # energy <= capacity * availabilityFactor
    model   =   set_Operation_Constraints_stockCtr(model)
    model   =   set_Operation_Constraints_Ramp(model)

    return model ;

def GetElectricSystemModel_GestionSingleNode_withStorage(Parameters):
    """
    This function creates the pyomo model and initlize the Parameters and (pyomo) Set values
    :param Parameters is a dictionnary with different panda tables :
        - areaConsumption: panda table with consumption
        - availabilityFactor: panda table
        - TechParameters : panda tables indexed by TECHNOLOGIES with eco and tech parameters
        - StorageParameters : panda tables indexed by STOCK_TECHNO with eco and tech parameters
    :return: pyomo model
    """
    import pandas as pd
    import numpy as np
    
    #isAbstract=False
    Parameters["availabilityFactor"].isna().sum()

    ### Cleaning
    availabilityFactor= Parameters["availabilityFactor"].fillna(method='pad');
    areaConsumption= Parameters["areaConsumption"].fillna(method='pad');
    TechParameters=Parameters["TechParameters"]
    StorageParameters=Parameters["StorageParameters"]
    ### obtaining dimensions values

    TECHNOLOGIES=   set(TechParameters.index.get_level_values('TECHNOLOGIES').unique())
    STOCK_TECHNO= set(StorageParameters.index.get_level_values('STOCK_TECHNO').unique())
    Date = set(areaConsumption.index.get_level_values('Date').unique())
    Date_list= areaConsumption.index.get_level_values('Date').unique()

    #####################
    #    Pyomo model    #
    #####################
    model = pyomo.environ.ConcreteModel()

    ###############
    # Sets       ##
    ###############
    model.TECHNOLOGIES  = Set(initialize=TECHNOLOGIES,ordered=False)
    model.STOCK_TECHNO = Set(initialize=STOCK_TECHNO,ordered=False)
    model.Date     = Set(initialize=Date,ordered=False)
    model.Date_TECHNOLOGIES =  model.Date * model.TECHNOLOGIES

    #Subset of Simple only required if ramp constraint
    model.Date_MinusOne = Set(initialize=Date_list[: len(Date) - 1],ordered=False)
    model.Date_MinusThree = Set(initialize=Date_list[: len(Date) - 3],ordered=False)


    ###############
    # Parameters ##
    ###############

    model.areaConsumption =     Param(model.Date, mutable=True,
                                      initialize=areaConsumption.loc[:,"areaConsumption"].squeeze().to_dict(), domain=Any)
    model.availabilityFactor =  Param( model.Date_TECHNOLOGIES, domain=PercentFraction,default=1,
                                      initialize=availabilityFactor.loc[:,"availabilityFactor"].squeeze().to_dict())

    #with test of existing columns on TechParameters
    for COLNAME in TechParameters:
        if COLNAME not in ["TECHNOLOGIES","AREAS"]: ### each column in TechParameters will be a parameter
            exec("model."+COLNAME+" =          Param(model.TECHNOLOGIES, domain=NonNegativeReals,default=0,"+
                                      "initialize=TechParameters."+COLNAME+".squeeze().to_dict())")

    for COLNAME in StorageParameters:
        if COLNAME not in ["STOCK_TECHNO","AREAS"]: ### each column in StorageParameters will be a parameter
            exec("model."+COLNAME+" =Param(model.STOCK_TECHNO,domain=NonNegativeReals,default=0,"+
                                      "initialize=StorageParameters."+COLNAME+".squeeze().to_dict())")

    ################
    # Variables    #
    ################

    model.energy=Var(model.Date, model.TECHNOLOGIES, domain=NonNegativeReals) ### Energy produced by a production mean at time t
    model.energyCosts=Var(model.TECHNOLOGIES)  ### Cost of energy for a production mean, explicitely defined by definition energyCostsDef
    model.storageIn=Var(model.Date,model.STOCK_TECHNO,domain=NonNegativeReals) ### Energy stored in a storage mean at time t 
    model.storageOut=Var(model.Date,model.STOCK_TECHNO,domain=NonNegativeReals) ### Energy taken out of a storage mean at time t 
    model.stockLevel=Var(model.Date,model.STOCK_TECHNO,domain=NonNegativeReals) ### level of the energy stock in a storage mean at time t
    model.stockLevel_ini=Var(model.STOCK_TECHNO,domain=NonNegativeReals)
    model.dual = Suffix(direction=Suffix.IMPORT)
    model.rc = Suffix(direction=Suffix.IMPORT)
    #model.slack = Suffix(direction=Suffix.IMPORT)

    ########################
    # Objective Function   #
    ########################

    def ObjectiveFunction_rule(model): #OBJ
    	return sum(model.energyCosts[tech] for tech in model.TECHNOLOGIES) + sum(model.Cmax[s_tech]*model.storageCost[s_tech]
                                                                                 for s_tech in model.STOCK_TECHNO)
    model.OBJ = Objective(rule=ObjectiveFunction_rule, sense=minimize)

    #################
    # Constraints   #
    #################

    # energyCosts definition Constraints
    def energyCostsDef_rule(model,tech): #EQ forall tech in TECHNOLOGIES   energyCosts  = sum{t in Date} energyCost[tech]*energy[t,tech] / 1E6;
        temp=model.energyCost[tech]# /10**6 ;
        return sum(temp*model.energy[t,tech] for t in model.Date) == model.energyCosts[tech]
    model.energyCostsCtr = Constraint(model.TECHNOLOGIES, rule=energyCostsDef_rule)

    #contrainte d'equilibre offre demande
    def energyCtr_rule(model,t): #INEQ forall t
    	return sum(model.energy[t,tech] for tech in model.TECHNOLOGIES)+sum(model.storageOut[t,s_tech]-model.storageIn[t,s_tech] for s_tech in model.STOCK_TECHNO) == model.areaConsumption[t]
    model.energyCtr = Constraint(model.Date,rule=energyCtr_rule)

    model   =   set_Operation_Constraints_CapacityCtr(model)#energy <= capacity * availabilityFactor
    model   =   set_Operation_Constraints_Storage(model)
    model   =   set_Operation_Constraints_stockCtr(model)
    model   =   set_Operation_Constraints_Ramp(model)

    return model ;
