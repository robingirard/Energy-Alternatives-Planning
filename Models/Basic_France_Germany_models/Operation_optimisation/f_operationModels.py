#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 19:07:50 2020

@author: robin.girard
"""

from __future__ import division
from datetime import timedelta
import pandas as pd
from pyomo.environ import *
from pyomo.core import *
from functions.f_tools import *
from functions.f_model_definition import *
from functions.f_model_operation_constraints import *


def GetElectricSystemModel_GestionMultiNode(Parameters,LineEfficiency=1):
    """
    This function creates the pyomo model and initlize the Parameters and (pyomo) Set values
    :param Parameters is a dictionnary with different panda tables :
        - areaConsumption: panda table with consumption
        - availabilityFactor: panda table
        - TechParameters : panda tables indexed by TECHNOLOGIES with eco and tech parameters
        - ExchangeParameters

    :param isAbstract: boolean true is the model should be abstract. ConcreteModel otherwise
    :return: pyomo model
    """

    #isAbstract=False
    Parameters["availabilityFactor"].isna().sum()

    ### Cleaning
    availabilityFactor=Parameters["availabilityFactor"].fillna(method='pad');
    areaConsumption=Parameters["areaConsumption"].fillna(method='pad');
    ExchangeParameters=Parameters["ExchangeParameters"]
    TechParameters=Parameters["TechParameters"]

    ### obtaining dimensions values
    TECHNOLOGIES= set(TechParameters.index.get_level_values('TECHNOLOGIES').unique())
    Date = set(areaConsumption.index.get_level_values('Date').unique())
    Date_list = areaConsumption.index.get_level_values('Date').unique()
    AREAS= set(areaConsumption.index.get_level_values('AREAS').unique())

    model = ConcreteModel()
    ###############
    # Sets       ##
    ###############

    #Simple
    model.AREAS= Set(initialize=AREAS,doc = "Area",ordered=False)
    model.TECHNOLOGIES = Set(initialize=TECHNOLOGIES,ordered=False)
    model.Date = Set(initialize=Date,ordered=False)

    #Subset of Simple only required if ramp constraint
    model.Date_MinusOne = Set(initialize=Date_list[: len(Date) - 1], ordered=False)
    model.Date_MinusThree = Set(initialize=Date_list[: len(Date) - 3],ordered=False)

    #Products
    model.Date_TECHNOLOGIES =  model.Date *model.TECHNOLOGIES
    model.AREAS_AREAS= model.AREAS * model.AREAS
    model.AREAS_TECHNOLOGIES= model.AREAS * model.TECHNOLOGIES
    model.AREAS_Date=model.AREAS * model.Date
    model.AREAS_Date_TECHNOLOGIES= model.AREAS*model.Date * model.TECHNOLOGIES


    ###############
    # Parameters ##
    ###############

    model.areaConsumption =     Param(model.AREAS_Date,
                                      initialize=areaConsumption.loc[:, "areaConsumption"].squeeze().to_dict(), domain=Any)
    model.availabilityFactor =  Param( model.AREAS_Date_TECHNOLOGIES, domain=PercentFraction,default=1,
                                      initialize=availabilityFactor.squeeze().to_dict())
    model.maxExchangeCapacity=Param(model.AREAS_AREAS,  initialize=ExchangeParameters.squeeze().to_dict(),
                                    domain=NonNegativeReals,default=0)
    #with test of existing columns on TechParameters
    for COLNAME in TechParameters:
            exec("model."+COLNAME+" =Param(model.AREAS_TECHNOLOGIES, default=0,"+
                                      "initialize=TechParameters."+COLNAME+".squeeze().to_dict())")


    ################
    # Variables    #
    ################

    model.energy=Var(model.AREAS,model.Date, model.TECHNOLOGIES, domain=NonNegativeReals) ### Energy produced by a production mean at time t
    model.exchange=Var(model.AREAS_AREAS,model.Date)
    model.energyCosts=Var(model.AREAS,model.TECHNOLOGIES)   ### Cost of energy by a production mean for area at time t (explicitely defined by constraint energyCostsCtr)

    model.dual = Suffix(direction=Suffix.IMPORT)
    model.rc = Suffix(direction=Suffix.IMPORT)
    #model.slack = Suffix(direction=Suffix.IMPORT)

    ########################
    # Objective Function   #
    ########################

    def ObjectiveFunction_rule(model): #OBJ
    	return sum(model.energyCosts[area,tech] for tech in model.TECHNOLOGIES for area in model.AREAS);
    model.OBJ = Objective(rule=ObjectiveFunction_rule, sense=minimize)


    #################
    # Constraints   #
    #################


    # Variable muette : energyCosts
    def energyCostsDef_rule(model,area,tech): #EQ forall tech in TECHNOLOGIES   energyCosts  = sum{t in Date} energyCost[tech]*energy[t,tech] / 1E6;
        temp=model.energyCost[area,tech]#/10**6 ;
        return sum(temp*model.energy[area,t,tech] for t in model.Date) == model.energyCosts[area,tech];
    model.energyCostsDef = Constraint(model.AREAS,model.TECHNOLOGIES, rule=energyCostsDef_rule)

    #contrainte d'equilibre offre demande [AREAS x Date x TECHNOLOGIES]
    def energyCtr_rule(model,area,t): #INEQ forall t
    	return sum(model.energy[area,t,tech] for tech in model.TECHNOLOGIES ) + sum(model.exchange[b,area,t]*LineEfficiency for b in model.AREAS ) == model.areaConsumption[area,t]
    model.energyCtr = Constraint(model.AREAS,model.Date,rule=energyCtr_rule)

    #Exchange capacity constraint (duplicate of variable definition) [AREAS x AREAS x Date]
    def exchangeCtrPlus_rule(model,a, b, t): #INEQ forall area.axarea.b in AREASxAREAS  t in Date
        if a!=b:
            return model.exchange[a,b,t]  <= model.maxExchangeCapacity[a,b] ;
        else:
            return model.exchange[a, a, t] == 0
    model.exchangeCtrPlus = Constraint(model.AREAS,model.AREAS,model.Date, rule=exchangeCtrPlus_rule)

    def exchangeCtrMoins_rule(model,a, b, t): #INEQ forall area.axarea.b in AREASxAREAS  t in Date
        if a!=b:
            return model.exchange[a,b,t]  >= -model.maxExchangeCapacity[a,b] ;
        else:
            return model.exchange[a, a, t] == 0
    model.exchangeCtrMoins = Constraint(model.AREAS,model.AREAS,model.Date, rule=exchangeCtrMoins_rule)

    def exchangeCtr2_rule(model,a, b, t): #INEQ forall area.axarea.b in AREASxAREAS  t in Date
        return model.exchange[a,b,t]  == -model.exchange[b,a,t] ;
    model.exchangeCtr2 = Constraint(model.AREAS,model.AREAS,model.Date, rule=exchangeCtr2_rule)

    #other Classical operation constraints
    model   =   set_Operation_Constraints_CapacityCtr(model) #energy <= capacity * availabilityFactor
    model   =   set_Operation_Constraints_stockCtr(model) #limit on the stock over a year
    model   =   set_Operation_Constraints_Ramp(model)

    return model ;


def GetElectricSystemModel_GestionMultiNode_withStorage(Parameters,LineEfficiency=1):
    """
    This function takes storage caracteristics, system caracteristics and optimise operation Set values
        :param Parameters is a dictionnary with different panda tables :
        - areaConsumption: panda table with consumption
        - availabilityFactor: panda table
        - TechParameters : panda tables indexed by TECHNOLOGIES with eco and tech parameters
        - StorageParameters : panda tables indexed by STOCK_TECHNO with eco and tech parameters
        - ExchangeParameters
    :return: a dictionary with model : pyomo model without storage, storage : storage infos
    """
    
    import pandas as pd
    import numpy as np

    #isAbstract=False
    Parameters["availabilityFactor"].isna().sum()

    ### Cleaning
    availabilityFactor=Parameters["availabilityFactor"].fillna(method='pad');
    areaConsumption=Parameters["areaConsumption"].fillna(method='pad');
    TechParameters=Parameters["TechParameters"]
    StorageParameters=Parameters["StorageParameters"]
    ExchangeParameters = Parameters["ExchangeParameters"]

    ### obtaining dimensions values
    TECHNOLOGIES= set(TechParameters.index.get_level_values('TECHNOLOGIES').unique())
    STOCK_TECHNO= set(StorageParameters.index.get_level_values('STOCK_TECHNO').unique())
    Date = set(areaConsumption.index.get_level_values('Date').unique())
    Date_list = areaConsumption.index.get_level_values('Date').unique()
    AREAS= set(areaConsumption.index.get_level_values('AREAS').unique())

    model = ConcreteModel()

    ###############
    # Sets       ##
    ###############

    #Simple
    model.AREAS= Set(initialize=AREAS,doc = "Area",ordered=False)
    model.TECHNOLOGIES = Set(initialize=TECHNOLOGIES,ordered=False)
    model.STOCK_TECHNO= Set(initialize=STOCK_TECHNO,ordered=False)
    model.Date = Set(initialize=Date,ordered=False)

    #Subset of Simple only required if ramp constraint
    model.Date_MinusOne = Set(initialize=Date_list[: len(Date) - 1], ordered=False)
    model.Date_MinusThree = Set(initialize=Date_list[: len(Date) - 3],ordered=False)

    #Products
    model.Date_TECHNOLOGIES =  model.Date *model.TECHNOLOGIES
    model.AREAS_AREAS= model.AREAS * model.AREAS
    model.AREAS_TECHNOLOGIES= model.AREAS * model.TECHNOLOGIES
    model.AREAS_STOCKTECHNO=model.AREAS * model.STOCK_TECHNO
    model.AREAS_Date=model.AREAS * model.Date
    model.AREAS_Date_TECHNOLOGIES= model.AREAS*model.Date * model.TECHNOLOGIES


    ###############
    # Parameters ##
    ###############

    model.areaConsumption =     Param(model.AREAS_Date,
                                      initialize=areaConsumption.loc[:, "areaConsumption"].squeeze().to_dict(), domain=Any,mutable=True)
    model.availabilityFactor =  Param(  model.AREAS_Date_TECHNOLOGIES, domain=PercentFraction,default=1,
                                      initialize=availabilityFactor.squeeze().to_dict())

    model.maxExchangeCapacity = Param( model.AREAS_AREAS,  initialize=ExchangeParameters.squeeze().to_dict(), domain=NonNegativeReals,default=0)
    #with test of existing columns on TechParameters
    for COLNAME in TechParameters:
            exec("model."+COLNAME+" =Param(model.AREAS_TECHNOLOGIES, default=0,"+
                                      "initialize=TechParameters."+COLNAME+".squeeze().to_dict())")
            
    for COLNAME in StorageParameters:
            exec("model."+COLNAME+" =Param(model.AREAS_STOCKTECHNO,domain=NonNegativeReals,default=0,"+
                                      "initialize=StorageParameters."+COLNAME+".squeeze().to_dict())")

    ################
    # Variables    #
    ################

    model.energy=Var(model.AREAS,model.Date, model.TECHNOLOGIES, domain=NonNegativeReals) ### Energy produced by a production mean at time t
    model.exchange=Var(model.AREAS_AREAS,model.Date)
    model.energyCosts=Var(model.AREAS,model.TECHNOLOGIES)   ### Cost of energy by a production mean for area at time t (explicitely defined by constraint energyCostsCtr)
    model.storageCosts=Var(model.AREAS,model.STOCK_TECHNO) ### Cost of storage by a storage mean for area at time t (explicitely defined by constraint storageCostsCtr)
    model.storageIn=Var(model.AREAS,model.Date,model.STOCK_TECHNO,domain=NonNegativeReals) ### Energy stored by a storage mean for areas at time t 
    model.storageOut=Var(model.AREAS,model.Date,model.STOCK_TECHNO,domain=NonNegativeReals) ### Energy taken out of a storage mean for areas at time t 
    model.stockLevel=Var(model.AREAS,model.Date,model.STOCK_TECHNO,domain=NonNegativeReals) ### level of the energy stock in a storage mean for areas at time t
    model.stockLevel_ini=Var(model.AREAS,model.STOCK_TECHNO,domain=NonNegativeReals)
    model.dual = Suffix(direction=Suffix.IMPORT)
    model.rc = Suffix(direction=Suffix.IMPORT)
    #model.slack = Suffix(direction=Suffix.IMPORT)

    ########################
    # Objective Function   #
    ########################

    def ObjectiveFunction_rule(model): #OBJ
    	return sum(model.energyCosts[area,tech] for tech in model.TECHNOLOGIES for area in model.AREAS)+sum(model.storageCosts[area,s_tech] for s_tech in model.STOCK_TECHNO for area in model.AREAS);
    model.OBJ = Objective(rule=ObjectiveFunction_rule, sense=minimize)

    #################
    # Constraints   #
    #################

    # Variable muettes : energyCosts and storageCosts
    def energyCostsDef_rule(model,area,tech): #EQ forall tech in TECHNOLOGIES   energyCosts  = sum{t in Date} energyCost[tech]*energy[t,tech] / 1E6;
        temp=model.energyCost[area,tech]#/10**6 ;
        return sum(temp*model.energy[area,t,tech] for t in model.Date) == model.energyCosts[area,tech];
    model.energyCostsDef = Constraint(model.AREAS,model.TECHNOLOGIES, rule=energyCostsDef_rule)

    def storageCostsDef_rule(model,area,s_tech): #EQ forall s_tech in STOCK_TECHNO   storageCosts  = storageCost[s_tech]*Cmax[s_tech] / 1E6;
        return model.storageCost[area,s_tech]*model.Cmax[area,s_tech] == model.storageCosts[area,s_tech]#/10**6 ;;
    model.storageCostsDef = Constraint(model.AREAS,model.STOCK_TECHNO, rule=storageCostsDef_rule)

    #contrainte d'equilibre offre demande [AREAS x Date x TECHNOLOGIES]
    def energyCtr_rule(model,area,t): #INEQ forall t
    	return sum(model.energy[area,t,tech] for tech in model.TECHNOLOGIES ) +sum(model.storageOut[area,t,s_tech]-model.storageIn[area,t,s_tech]  for s_tech in model.STOCK_TECHNO)+ sum(model.exchange[b,area,t]*LineEfficiency for b in model.AREAS) == model.areaConsumption[area,t]
    model.energyCtr = Constraint(model.AREAS,model.Date,rule=energyCtr_rule)

    #Exchange capacity constraint (duplicate of variable definition)
    # AREAS x AREAS x Date
    def exchangeCtrPlus_rule(model,a, b, t): #INEQ forall area.axarea.b in AREASxAREAS  t in Date
        if a!=b:
            return model.exchange[a,b,t]  <= model.maxExchangeCapacity[a,b] ;
        else:
            return model.exchange[a, a, t] == 0
    model.exchangeCtrPlus = Constraint(model.AREAS,model.AREAS,model.Date, rule=exchangeCtrPlus_rule)

    def exchangeCtrMoins_rule(model,a, b, t): #INEQ forall area.axarea.b in AREASxAREAS  t in Date
        if a!=b:
            return model.exchange[a,b,t]  >= -model.maxExchangeCapacity[a,b] ;
        else:
            return model.exchange[a, a, t] == 0
    model.exchangeCtrMoins = Constraint(model.AREAS,model.AREAS,model.Date, rule=exchangeCtrMoins_rule)

    def exchangeCtr2_rule(model,a, b, t): #INEQ forall area.axarea.b in AREASxAREAS  t in Date
        return model.exchange[a,b,t]  == -model.exchange[b,a,t] ;
    model.exchangeCtr2 = Constraint(model.AREAS,model.AREAS,model.Date, rule=exchangeCtr2_rule)

    #Other classical operation constraints
    model = set_Operation_Constraints_CapacityCtr(model)#energy <= capacity * availabilityFactor
    model = set_Operation_Constraints_Storage(model)
    model = set_Operation_Constraints_stockCtr(model)
    model = set_Operation_Constraints_Ramp(model)

    return model ;

