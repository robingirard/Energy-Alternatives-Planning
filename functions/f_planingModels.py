#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 07:51:58 2020

@author: robin.girard
"""
from __future__ import division
from pyomo.core import *
from pyomo.opt import SolverFactory

from functions.f_optimization import *

def GetElectricSystemModel_PlaningSingleNode(areaConsumption,availabilityFactor,TechParameters,isAbstract=False):

    #isAbstract=False
    availabilityFactor.isna().sum()
    
    ### Cleaning
    availabilityFactor=availabilityFactor.fillna(method='pad');
    areaConsumption=areaConsumption.fillna(method='pad');
    
    ### obtaining dimensions values 
    TECHNOLOGIES=   set(TechParameters.index.get_level_values('TECHNOLOGIES').unique())
    TIMESTAMP=      set(areaConsumption.index.get_level_values('TIMESTAMP').unique())
    TIMESTAMP_list= areaConsumption.index.get_level_values('TIMESTAMP').unique()
    
    #####################
    #    Pyomo model    #
    #####################

    if (isAbstract) : 
        model = AbstractModel()
    else:
        model = ConcreteModel() 
        
    ###############
    # Sets       ##
    ###############
    model.TECHNOLOGIES = Set(initialize=TECHNOLOGIES,ordered=False)
    model.TIMESTAMP = Set(initialize=TIMESTAMP,ordered=False)
    model.TIMESTAMP_TECHNOLOGIES =  model.TIMESTAMP *model.TECHNOLOGIES

    #Subset of Simple only required if ramp constraint
    model.TIMESTAMP_MinusOne = Set(initialize=TIMESTAMP_list[: len(TIMESTAMP) - 1],ordered=False)
    model.TIMESTAMP_MinusThree = Set(initialize=TIMESTAMP_list[: len(TIMESTAMP) - 3],ordered=False)
    
    
    ###############
    # Parameters ##
    ###############
    
    model.areaConsumption =     Param(model.TIMESTAMP, mutable=True,
                                      initialize=areaConsumption.loc[:,"areaConsumption"].squeeze().to_dict(), domain=Any)
    model.availabilityFactor =  Param( model.TIMESTAMP_TECHNOLOGIES, domain=PercentFraction,default=1,
                                      initialize=availabilityFactor.loc[:,"availabilityFactor"].squeeze().to_dict())
   
    #with test of existing columns on TechParameters
    for COLNAME in TechParameters:
        if COLNAME not in ["TECHNOLOGIES","AREAS"]: ### each column in TechParameters will be a parameter
            exec("model."+COLNAME+" =          Param(model.TECHNOLOGIES, domain=NonNegativeReals,default=0,"+
                                      "initialize=TechParameters."+COLNAME+".squeeze().to_dict())")
    ## manière générique d'écrire pour toutes les colomnes COL de TechParameters quelque chose comme 
    # model.COLNAME =          Param(model.TECHNOLOGIES, domain=NonNegativeReals,default=0,
    #                                  initialize=TechParameters.set_index([ "TECHNOLOGIES"]).COLNAME.squeeze().to_dict())
    
    ################
    # Variables    #
    ################
    
   
    model.energy=Var(model.TIMESTAMP, model.TECHNOLOGIES, domain=NonNegativeReals) ### Energy produced by a production mean at time t
    model.energyCosts=Var(model.TECHNOLOGIES)  ### Cost of energy for a production mean, explicitely defined by definition energyCostsDef    
    model.capacityCosts=Var(model.TECHNOLOGIES)  ### Cost of energy for a production mean, explicitely defined by definition energyCostsDef    
    model.capacity=Var(model.TECHNOLOGIES, domain=NonNegativeReals) ### Energy produced by a production mean at time t
    
    model.dual = Suffix(direction=Suffix.IMPORT)
    model.rc = Suffix(direction=Suffix.IMPORT)
    model.slack = Suffix(direction=Suffix.IMPORT)
    
    ########################
    # Objective Function   #
    ########################
    
    def ObjectiveFunction_rule(model): #OBJ
    	return sum(model.energyCosts[tech]+model.capacityCosts[tech] for tech in model.TECHNOLOGIES)
    model.OBJ = Objective(rule=ObjectiveFunction_rule, sense=minimize)
    
    
    #################
    # Constraints   #
    #################
    
    
    # energyCosts definition Constraints       
    def energyCostsDef_rule(model,tech): #EQ forall tech in TECHNOLOGIES   energyCosts  = sum{t in TIMESTAMP} energyCost[tech]*energy[t,tech] / 1E6;
        temp=model.energyCost[tech]#/10**6 ;
        return sum(temp*model.energy[t,tech] for t in model.TIMESTAMP) == model.energyCosts[tech]
    model.energyCostsCtr = Constraint(model.TECHNOLOGIES, rule=energyCostsDef_rule)    
    
    # capacityCosts definition Constraints
    def capacityCostsDef_rule(model,tech): #EQ forall tech in TECHNOLOGIES   energyCosts  = sum{t in TIMESTAMP} energyCost[tech]*energy[t,tech] / 1E6;
        temp=model.capacityCosts[tech]#/10**6 ;
        return model.capacityCost[tech]*model.capacity[tech] == model.capacityCosts[tech]
        #return model.capacityCost[tech] * len(model.TIMESTAMP) / 8760 * model.capacity[tech] / 10 ** 6 == model.capacityCosts[tech]

    model.capacityCostsCtr = Constraint(model.TECHNOLOGIES, rule=capacityCostsDef_rule)    


    
    #Capacity constraint
    def Capacity_rule(model,t,tech): #INEQ forall t, tech 
    	return model.capacity[tech] * model.availabilityFactor[t,tech] >= model.energy[t,tech]
    model.CapacityCtr = Constraint(model.TIMESTAMP,model.TECHNOLOGIES, rule=Capacity_rule)    
    
    #contrainte de stock annuel 

    
    #contrainte d'equilibre offre demande 
    def energyCtr_rule(model,t): #INEQ forall t
    	return sum(model.energy[t,tech] for tech in model.TECHNOLOGIES ) >= model.areaConsumption[t]
    model.energyCtr = Constraint(model.TIMESTAMP,rule=energyCtr_rule)
    
    if "maxCapacity" in TechParameters:
        def maxCapacity_rule(model,tech) : #INEQ forall t, tech 
            if model.maxCapacity[tech]>0 :
                return model.maxCapacity[tech] >= model.capacity[tech] 
            else:
                return Constraint.Skip
        model.maxCapacityCtr = Constraint(model.TECHNOLOGIES, rule=maxCapacity_rule)
    
    if "minCapacity" in TechParameters:
        def minCapacity_rule(model,tech) : #INEQ forall t, tech 
            if model.minCapacity[tech]>0 :
                return model.minCapacity[tech] <= model.capacity[tech] 
            else:
                return Constraint.Skip
        model.minCapacityCtr = Constraint(model.TECHNOLOGIES, rule=minCapacity_rule)
    
    if "EnergyNbhourCap" in TechParameters:
        def storage_rule(model,tech) : #INEQ forall t, tech 
            if model.EnergyNbhourCap[tech]>0 :
                return model.EnergyNbhourCap[tech]*model.capacity[tech] >= sum(model.energy[t,tech] for t in model.TIMESTAMP)
            else:
                return Constraint.Skip
        model.storageCtr = Constraint(model.TECHNOLOGIES, rule=storage_rule)

    if "RampConstraintPlus" in TechParameters:
        def rampCtrPlus_rule(model,t,tech): #INEQ forall t<
            if model.RampConstraintPlus[tech]>0 :
                return model.energy[t+1,tech]  - model.energy[t,tech] <= model.capacity[tech]*model.RampConstraintPlus[tech] ;
            else:
                return Constraint.Skip
        model.rampCtrPlus = Constraint(model.TIMESTAMP_MinusOne,model.TECHNOLOGIES,rule=rampCtrPlus_rule)
        
    if "RampConstraintMoins" in TechParameters:
        def rampCtrMoins_rule(model,t,tech): #INEQ forall t<
            if model.RampConstraintMoins[tech]>0 :
                return model.energy[t+1,tech]  - model.energy[t,tech] >= - model.capacity[tech]*model.RampConstraintMoins[tech] ;
            else:
                return Constraint.Skip
        model.rampCtrMoins = Constraint(model.TIMESTAMP_MinusOne,model.TECHNOLOGIES,rule=rampCtrMoins_rule)
        
    if "RampConstraintPlus2" in TechParameters:
        def rampCtrPlus2_rule(model,t,tech): #INEQ forall t<
            if model.RampConstraintPlus2[tech]>0 :
                var=(model.energy[t+2,tech]+model.energy[t+3,tech])/2 -  (model.energy[t+1,tech]+model.energy[t,tech])/2;
                return var <= model.capacity[tech]*model.RampConstraintPlus[tech] ;
            else:
                return Constraint.Skip
        model.rampCtrPlus2 = Constraint(model.TIMESTAMP_MinusThree,model.TECHNOLOGIES,rule=rampCtrPlus2_rule)
        
    if "RampConstraintMoins2" in TechParameters:
        def rampCtrMoins2_rule(model,t,tech): #INEQ forall t<
            if model.RampConstraintMoins2[tech]>0 :
                var=(model.energy[t+2,tech]+model.energy[t+3,tech])/2 -  (model.energy[t+1,tech]+model.energy[t,tech])/2;
                return var >= - model.capacity[tech]*model.RampConstraintMoins2[tech] ;
            else:
                return Constraint.Skip
        model.rampCtrMoins2 = Constraint(model.TIMESTAMP_MinusThree,model.TECHNOLOGIES,rule=rampCtrMoins2_rule)
    
    return model ;

    ##################################
    # Loop Optimisation with Storage #
    ##################################


def GetElectricSystemModel_PlaningSingleNode_with1Storage(areaConsumption,availabilityFactor,
                                                          TechParameters,StorageParameters,solverpath=-1,isAbstract=False,
                                                          solver='mosek'):
    """
    This function takes storage caracteristics, system caracteristics and optimise operation Set values
    :param areaConsumption: panda table with consumption
    :param availabilityFactor: panda table
    :param isAbstract: boolean true is the model should be abstract. ConcreteModel otherwise
    :param StorageParameters is a dictionary with p_max (maximal power), c_max (energy capacity in the storage : maximal energy),
    :efficiency_in (input efficiency of storage),
    :efficiency_out (output efficiency of storage).
    :return: a dictionary with model : pyomo model without storage, storage : storage infos
    """
    #isAbstract=False

    availabilityFactor.isna().sum()
    
    ### Cleaning
    availabilityFactor=availabilityFactor.fillna(method='pad');
    areaConsumption=areaConsumption.fillna(method='pad');
    
    ### obtaining dimensions values 
    TECHNOLOGIES=   set(TechParameters.index.get_level_values('TECHNOLOGIES').unique())
    STOCK_TECHNO= set(StorageParameters.index.get_level_values('STOCK_TECHNO').unique())
    TIMESTAMP=      set(areaConsumption.index.get_level_values('TIMESTAMP').unique())
    TIMESTAMP_list= areaConsumption.index.get_level_values('TIMESTAMP').unique()
    
    #####################
    #    Pyomo model    #
    #####################

    if (isAbstract) : 
        model = AbstractModel()
    else:
        model = ConcreteModel() 
        
    ###############
    # Sets       ##
    ###############
    model.TECHNOLOGIES = Set(initialize=TECHNOLOGIES,ordered=False)
    model.STOCK_TECHNO = Set(initialize=STOCK_TECHNO,ordered=False)  
    model.TIMESTAMP = Set(initialize=TIMESTAMP,ordered=False)
    model.TIMESTAMP_TECHNOLOGIES =  model.TIMESTAMP *model.TECHNOLOGIES

    #Subset of Simple only required if ramp constraint
    model.TIMESTAMP_MinusOne = Set(initialize=TIMESTAMP_list[: len(TIMESTAMP) - 1],ordered=False)
    model.TIMESTAMP_MinusThree = Set(initialize=TIMESTAMP_list[: len(TIMESTAMP) - 3],ordered=False)
    
    
    ###############
    # Parameters ##
    ###############
    
    model.areaConsumption =     Param(model.TIMESTAMP, mutable=True,
                                      initialize=areaConsumption.loc[:,"areaConsumption"].squeeze().to_dict(), domain=Any)
    model.availabilityFactor =  Param( model.TIMESTAMP_TECHNOLOGIES, domain=PercentFraction,default=1,
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
    ## manière générique d'écrire pour toutes les colomnes COL de TechParameters quelque chose comme 
    # model.COLNAME =          Param(model.TECHNOLOGIES, domain=NonNegativeReals,default=0,
    #                                  initialize=TechParameters.set_index([ "TECHNOLOGIES"]).COLNAME.squeeze().to_dict())
    
    ################
    # Variables    #
    ################
    
   ###Production means :
    model.energy=Var(model.TIMESTAMP, model.TECHNOLOGIES, domain=NonNegativeReals) ### Energy produced by a production mean at time t
    model.energyCosts=Var(model.TECHNOLOGIES)  ### Cost of energy for a production mean, explicitely defined by definition energyCostsDef    
    model.capacityCosts=Var(model.TECHNOLOGIES)  ### Cost of energy for a production mean, explicitely defined by definition energyCostsDef 
    model.capacity=Var(model.TECHNOLOGIES, domain=NonNegativeReals) ### Energy produced by a production mean at time t
    
    ###Storage means :
    model.storage=Var(model.TIMESTAMP,model.STOCK_TECHNO) ### Energy stored in a storage mean at time t (if Storage < 0, then the energy is taken from the storage, if Storage > 0, then the is stored)
    model.stockLevel=Var(model.TIMESTAMP,model.STOCK_TECHNO,domain=NonNegativeReals) ### level of the energy stock in a storage mean at time t
    model.storageCosts=Var(model.STOCK_TECHNO)  ### Cost of storage for a storage mean, explicitely defined by definition storageCostsDef    
    model.Cmax=Var(model.STOCK_TECHNO) # Maximum capacity of a storage mean
    model.Pmax=Var(model.STOCK_TECHNO) # Maximum flow of energy in/out of a storage mean

    ### Other :
    model.dual = Suffix(direction=Suffix.IMPORT)
    model.rc = Suffix(direction=Suffix.IMPORT)
    model.slack = Suffix(direction=Suffix.IMPORT)
    
    ########################
    # Objective Function   #
    ########################
    
    def ObjectiveFunction_rule(model): #OBJ
    	return sum(model.energyCosts[tech]+model.capacityCosts[tech] for tech in model.TECHNOLOGIES)+sum(model.storageCosts[s_tech] for s_tech in STOCK_TECHNO)
    model.OBJ = Objective(rule=ObjectiveFunction_rule, sense=minimize)
    
    
    #################
    # Constraints   #
    #################
    
    
    # energyCosts definition Constraints       
    def energyCostsDef_rule(model,tech): #EQ forall tech in TECHNOLOGIES   energyCosts  = sum{t in TIMESTAMP} energyCost[tech]*energy[t,tech] / 1E6;
        temp=model.energyCost[tech]#/10**6 ;
        return sum(temp*model.energy[t,tech] for t in model.TIMESTAMP) == model.energyCosts[tech]
    model.energyCostsCtr = Constraint(model.TECHNOLOGIES, rule=energyCostsDef_rule)    
    
    # capacityCosts definition Constraints
    def capacityCostsDef_rule(model,tech): #EQ forall tech in TECHNOLOGIES   energyCosts  = sum{t in TIMESTAMP} energyCost[tech]*energy[t,tech] / 1E6;
        temp=model.capacityCosts[tech]#/10**6 ;
        return model.capacityCost[tech]*model.capacity[tech] == model.capacityCosts[tech]
        #return model.capacityCost[tech] * len(model.TIMESTAMP) / 8760 * model.capacity[tech] / 10 ** 6 == model.capacityCosts[tech]
    model.capacityCostsCtr = Constraint(model.TECHNOLOGIES, rule=capacityCostsDef_rule)    

    def storageCostsDef_rule(model,s_tech): #EQ forall s_tech in STOCK_TECHNO storageCosts=storageCost[s_tech]*Cmax[s_tech] / 1E6;
        return model.storageCost[s_tech]*model.Cmax[s_tech] == model.storageCosts[s_tech]
    model.storageCostsCtr = Constraint(model.STOCK_TECHNO, rule=storageCostsDef_rule) 
    
    #Capacity constraint
    def Capacity_rule(model,t,tech): #INEQ forall t, tech 
    	return model.capacity[tech] * model.availabilityFactor[t,tech] >= model.energy[t,tech]
    model.CapacityCtr = Constraint(model.TIMESTAMP,model.TECHNOLOGIES, rule=Capacity_rule)    
    
    #Storage max capacity constraint
    def storageCapacity_rule(model,s_tech): #INEQ forall s_tech 
    	return model.Cmax[s_tech]  <= model.c_max[s_tech]
    model.storageCapacityCtr = Constraint(model.STOCK_TECHNO, rule=storageCapacity_rule)   
    
    #Storage max power constraint
    def storagePower_rule(model,s_tech): #INEQ forall s_tech 
    	return model.Pmax[s_tech]  <= model.p_max[s_tech]
    model.storagePowerCtr = Constraint(model.STOCK_TECHNO, rule=storagePower_rule) 
    
    #contraintes de stock puissance
    def StoragePowerUB_rule(model, t,s_tech):  # INEQ forall t
        return model.storage[t,s_tech] - model.Pmax[s_tech] <= 0
    model.StoragePowerUBCtr = Constraint(model.TIMESTAMP,model.STOCK_TECHNO, rule=StoragePowerUB_rule)

    def StoragePowerLB_rule(model,t,s_tech,):  # INEQ forall t
        return  model.storage[t,s_tech] + model.Pmax[s_tech] >= 0
    model.StoragePowerLBCtr = Constraint(model.TIMESTAMP,model.STOCK_TECHNO, rule=StoragePowerLB_rule)

    #contraintes de stock capacité
    def StockLevel_rule(model, t,s_tech):  # EQ forall t
        if t>1 :
            return model.stockLevel[t,s_tech] == model.stockLevel[t-1,s_tech] + model.storage[t,s_tech]
        else :
            return model.stockLevel[t,s_tech] == 0
    model.StockLevelCtr = Constraint(model.TIMESTAMP,model.STOCK_TECHNO, rule=StockLevel_rule)
    
    def StockCapacity_rule(model,t,s_tech,):  # INEQ forall t
        return model.stockLevel[t,s_tech] <= model.Cmax[s_tech]
    model.StockCapacityCtr = Constraint(model.TIMESTAMP,model.STOCK_TECHNO, rule=StockCapacity_rule)
    
    #contrainte d'equilibre offre demande 
    def energyCtr_rule(model,t): #INEQ forall t
    	return sum(model.energy[t,tech] for tech in model.TECHNOLOGIES)+sum(model.storage[t,s_tech] for s_tech in model.STOCK_TECHNO) >= model.areaConsumption[t]
    model.energyCtr = Constraint(model.TIMESTAMP,rule=energyCtr_rule)
    
    if "maxCapacity" in TechParameters:
        def maxCapacity_rule(model,tech) : #INEQ forall t, tech 
            if model.maxCapacity[tech]>0 :
                return model.maxCapacity[tech] >= model.capacity[tech] 
            else:
                return Constraint.Skip
        model.maxCapacityCtr = Constraint(model.TECHNOLOGIES, rule=maxCapacity_rule)
    
    if "minCapacity" in TechParameters:
        def minCapacity_rule(model,tech) : #INEQ forall t, tech 
            if model.minCapacity[tech]>0 :
                return model.minCapacity[tech] <= model.capacity[tech] 
            else:
                return Constraint.Skip
        model.minCapacityCtr = Constraint(model.TECHNOLOGIES, rule=minCapacity_rule)
    
    if "EnergyNbhourCap" in TechParameters:
        def storage_rule(model,tech) : #INEQ forall t, tech 
            if model.EnergyNbhourCap[tech]>0 :
                return model.EnergyNbhourCap[tech]*model.capacity[tech] >= sum(model.energy[t,tech] for t in model.TIMESTAMP)
            else:
                return Constraint.Skip
        model.storageCtr = Constraint(model.TECHNOLOGIES, rule=storage_rule)

    if "RampConstraintPlus" in TechParameters:
        def rampCtrPlus_rule(model,t,tech): #INEQ forall t<
            if model.RampConstraintPlus[tech]>0 :
                return model.energy[t+1,tech]  - model.energy[t,tech] <= model.capacity[tech]*model.RampConstraintPlus[tech] ;
            else:
                return Constraint.Skip
        model.rampCtrPlus = Constraint(model.TIMESTAMP_MinusOne,model.TECHNOLOGIES,rule=rampCtrPlus_rule)
        
    if "RampConstraintMoins" in TechParameters:
        def rampCtrMoins_rule(model,t,tech): #INEQ forall t<
            if model.RampConstraintMoins[tech]>0 :
                return model.energy[t+1,tech]  - model.energy[t,tech] >= - model.capacity[tech]*model.RampConstraintMoins[tech] ;
            else:
                return Constraint.Skip
        model.rampCtrMoins = Constraint(model.TIMESTAMP_MinusOne,model.TECHNOLOGIES,rule=rampCtrMoins_rule)
        
    if "RampConstraintPlus2" in TechParameters:
        def rampCtrPlus2_rule(model,t,tech): #INEQ forall t<
            if model.RampConstraintPlus2[tech]>0 :
                var=(model.energy[t+2,tech]+model.energy[t+3,tech])/2 -  (model.energy[t+1,tech]+model.energy[t,tech])/2;
                return var <= model.capacity[tech]*model.RampConstraintPlus[tech] ;
            else:
                return Constraint.Skip
        model.rampCtrPlus2 = Constraint(model.TIMESTAMP_MinusThree,model.TECHNOLOGIES,rule=rampCtrPlus2_rule)
        
    if "RampConstraintMoins2" in TechParameters:
        def rampCtrMoins2_rule(model,t,tech): #INEQ forall t<
            if model.RampConstraintMoins2[tech]>0 :
                var=(model.energy[t+2,tech]+model.energy[t+3,tech])/2 -  (model.energy[t+1,tech]+model.energy[t,tech])/2;
                return var >= - model.capacity[tech]*model.RampConstraintMoins2[tech] ;
            else:
                return Constraint.Skip
        model.rampCtrMoins2 = Constraint(model.TIMESTAMP_MinusThree,model.TECHNOLOGIES,rule=rampCtrMoins2_rule)
    
    
    return model ;


def GetElectricSystemModel_PlaningMultiNode(areaConsumption,availabilityFactor,TechParameters,ExchangeParameters,isAbstract=False):
    """
    This function creates the pyomo model and initlize the Parameters and (pyomo) Set values
    :param areaConsumption: panda table with consumption
    :param availabilityFactor: panda table 
    :param isAbstract: boolean true is the model should be abstract. ConcreteModel otherwise 
    :return: pyomo model
    """
    #isAbstract=False
    availabilityFactor.isna().sum()
    
    ### Cleaning
    availabilityFactor=availabilityFactor.fillna(method='pad');
    areaConsumption=areaConsumption.fillna(method='pad');
    
    ### obtaining dimensions values 
    TECHNOLOGIES = set(TechParameters.index.get_level_values('TECHNOLOGIES').unique())
    TIMESTAMP = set(areaConsumption.index.get_level_values('TIMESTAMP').unique())
    TIMESTAMP_list = areaConsumption.index.get_level_values('TIMESTAMP').unique()
    AREAS = set(areaConsumption.index.get_level_values('AREAS').unique())
    
    #####################
    #    Pyomo model    #
    #####################
    
    if (isAbstract) : 
        model = AbstractModel()
    else:
        model = ConcreteModel() 
        
    ###############
    # Sets       ##
    ###############
    
    #Simple
    model.AREAS= Set(initialize=AREAS,doc = "Area",ordered=False)
    model.TECHNOLOGIES = Set(initialize=TECHNOLOGIES,ordered=False)
    model.TIMESTAMP = Set(initialize=TIMESTAMP,ordered=False)
    
    #Subset of Simple only required if ramp constraint
    model.TIMESTAMP_MinusOne = Set(initialize=TIMESTAMP_list[: len(TIMESTAMP) - 1],ordered=False)
    model.TIMESTAMP_MinusThree = Set(initialize=TIMESTAMP_list[: len(TIMESTAMP) - 3],ordered=False)
    
    #Products
    model.TIMESTAMP_TECHNOLOGIES =  model.TIMESTAMP *model.TECHNOLOGIES
    model.AREAS_AREAS= model.AREAS* model.AREAS
    model.AREAS_TECHNOLOGIES= model.AREAS *model.TECHNOLOGIES
    model.AREAS_TIMESTAMP=model.AREAS * model.TIMESTAMP 
    model.AREAS_TIMESTAMP_TECHNOLOGIES= model.AREAS*model.TIMESTAMP * model.TECHNOLOGIES


    ###############
    # Parameters ##
    ###############
    
    model.areaConsumption =     Param(model.AREAS_TIMESTAMP,
                                      initialize=areaConsumption.loc[:, "areaConsumption"].squeeze().to_dict(), domain=Any,mutable=True)
    model.availabilityFactor =  Param(  model.AREAS_TIMESTAMP_TECHNOLOGIES, domain=PercentFraction,default=1,
                                      initialize=availabilityFactor.squeeze().to_dict())

    model.maxExchangeCapacity = Param( model.AREAS_AREAS,  initialize=ExchangeParameters.squeeze().to_dict(), domain=NonNegativeReals,default=0)
    #with test of existing columns on TechParameters
    for COLNAME in TechParameters:
            exec("model."+COLNAME+" =          Param(model.AREAS_TECHNOLOGIES, domain=NonNegativeReals,default=0,"+
                                      "initialize=TechParameters."+COLNAME+".squeeze().to_dict())")
    
    
    ################
    # Variables    #
    ################
    
    model.energy=Var(model.AREAS,model.TIMESTAMP, model.TECHNOLOGIES, domain=NonNegativeReals) ### Energy produced by a production mean at time t
    model.exchange=Var(model.AREAS_AREAS,model.TIMESTAMP)  
    model.energyCosts=Var(model.AREAS,model.TECHNOLOGIES)   ### Cost of energy by a production mean for area at time t (explicitely defined by constraint energyCostsCtr)
    
    model.capacityCosts=Var(model.AREAS,model.TECHNOLOGIES)  ### Cost of energy for a production mean, explicitely defined by definition energyCostsDef    
    model.capacity=Var(model.AREAS,model.TECHNOLOGIES, domain=NonNegativeReals) ### Energy produced by a production mean at time t
    
    model.dual = Suffix(direction=Suffix.IMPORT)
    model.rc = Suffix(direction=Suffix.IMPORT)
    #model.slack = Suffix(direction=Suffix.IMPORT)
    
    ########################
    # Objective Function   #
    ########################
    
    def ObjectiveFunction_rule(model): #OBJ
    	return sum(model.energyCosts[area,tech]+model.capacityCosts[area,tech] for tech in model.TECHNOLOGIES for area in model.AREAS)
    model.OBJ = Objective(rule=ObjectiveFunction_rule, sense=minimize)
    
    
    
    #################
    # Constraints   #
    ################# 
    
    #### 1 - Basics 
    ########
    
    
    # energyCosts definition Constraints
    # AREAS x TECHNOLOGIES       
    def energyCostsDef_rule(model,area,tech): #EQ forall tech in TECHNOLOGIES   energyCosts  = sum{t in TIMESTAMP} energyCost[tech]*energy[t,tech] / 1E6;
        temp=model.energyCost[area,tech]#/10**6 ;
        return sum(temp*model.energy[area,t,tech] for t in model.TIMESTAMP) == model.energyCosts[area,tech];
    model.energyCostsDef = Constraint(model.AREAS,model.TECHNOLOGIES, rule=energyCostsDef_rule)    
    
    # capacityCosts definition Constraints
    # AREAS x TECHNOLOGIES    
    def capacityCostsDef_rule(model,area,tech): #EQ forall tech in TECHNOLOGIES   energyCosts  = sum{t in TIMESTAMP} energyCost[tech]*energy[t,tech] / 1E6;
        temp=model.capacityCosts[area,tech]#/10**6 ;
        return model.capacityCost[area,tech]*len(model.TIMESTAMP)/8760*model.capacity[area,tech]  == model.capacityCosts[area,tech] #  .. ....... / 10**6
    model.capacityCostsCtr = Constraint(model.AREAS,model.TECHNOLOGIES, rule=capacityCostsDef_rule)    

    
    #Capacity constraint
    #AREAS x TIMESTAMP x TECHNOLOGIES
    def CapacityCtr_rule(model,area,t,tech): #INEQ forall t, tech 
    	return model.capacity[area,tech] * model.availabilityFactor[area,t,tech] >= model.energy[area,t,tech]
    model.CapacityCtr = Constraint(model.AREAS,model.TIMESTAMP,model.TECHNOLOGIES, rule=CapacityCtr_rule)

    #Exchange capacity constraint (duplicate of variable definition)
    # AREAS x AREAS x TIMESTAMP
    def exchangeCtrPlus_rule(model,a, b, t): #INEQ forall area.axarea.b in AREASxAREAS  t in TIMESTAMP
        if a!=b:
            return model.exchange[a,b,t]  <= model.maxExchangeCapacity[a,b] ;
        else:
            return model.exchange[a, a, t] == 0
    model.exchangeCtrPlus = Constraint(model.AREAS,model.AREAS,model.TIMESTAMP, rule=exchangeCtrPlus_rule)

    def exchangeCtrMoins_rule(model,a, b, t): #INEQ forall area.axarea.b in AREASxAREAS  t in TIMESTAMP
        if a!=b:
            return model.exchange[a,b,t]  >= -model.maxExchangeCapacity[a,b] ;
        else:
            return model.exchange[a, a, t] == 0
    model.exchangeCtrMoins = Constraint(model.AREAS,model.AREAS,model.TIMESTAMP, rule=exchangeCtrMoins_rule)

    def exchangeCtr2_rule(model,a, b, t): #INEQ forall area.axarea.b in AREASxAREAS  t in TIMESTAMP
        return model.exchange[a,b,t]  == -model.exchange[b,a,t] ;
    model.exchangeCtr2 = Constraint(model.AREAS,model.AREAS,model.TIMESTAMP, rule=exchangeCtr2_rule)


    #contrainte d'equilibre offre demande
    #AREAS x TIMESTAMP x TECHNOLOGIES
    def energyCtr_rule(model,area,t): #INEQ forall t
    	return sum(model.energy[area,t,tech] for tech in model.TECHNOLOGIES ) + sum(model.exchange[b,area,t] for b in model.AREAS ) >= model.areaConsumption[area,t]
    model.energyCtr = Constraint(model.AREAS,model.TIMESTAMP,rule=energyCtr_rule)



    #### 2 - Optional 
    ########

    #contrainte de stock annuel 
    if "maxCapacity" in TechParameters:
        def maxCapacity_rule(model,area,tech) : #INEQ forall t, tech 
            if model.maxCapacity[area,tech]>0 :
                return model.maxCapacity[area,tech] >= model.capacity[area,tech] 
            else:
                return Constraint.Skip
        model.maxCapacityCtr = Constraint(model.AREAS,model.TECHNOLOGIES, rule=maxCapacity_rule)
    
    if "minCapacity" in TechParameters:
        def minCapacity_rule(model,area,tech) : #INEQ forall t, tech 
            if model.minCapacity[area,tech]>0 :
                return model.minCapacity[area,tech] <= model.capacity[area,tech] 
            else:
                return Constraint.Skip
        model.minCapacityCtr = Constraint(model.AREAS,model.TECHNOLOGIES, rule=minCapacity_rule)

    
    
    #contrainte de stock annuel 
    #AREAS x TECHNOLOGIES
    if "EnergyNbhourCap" in TechParameters:
        def storageCtr_rule(model,area,tech) : #INEQ forall t, tech 
            if model.EnergyNbhourCap[area,tech]>0 :
                return model.EnergyNbhourCap[area,tech]*model.capacity[area,tech] >= sum(model.energy[area,t,tech] for t in model.TIMESTAMP)
            else:
                return Constraint.Skip
        model.storageCtr = Constraint(model.AREAS,model.TECHNOLOGIES, rule=storageCtr_rule)

    if "RampConstraintPlus" in TechParameters:
        def rampCtrPlus_rule(model,area,t,tech): #INEQ forall t<
            if model.RampConstraintPlus[area,tech]>0 :
                return model.energy[area,t+1,tech]  - model.energy[area,t,tech] <= model.capacity[area,tech]*model.RampConstraintPlus[area,tech] ;
            else:
                return Constraint.Skip
        model.rampCtrPlus = Constraint(model.AREAS,model.TIMESTAMP_MinusOne,model.TECHNOLOGIES,rule=rampCtrPlus_rule)
        
    if "RampConstraintMoins" in TechParameters:
        def rampCtrMoins_rule(model,area,t,tech): #INEQ forall t<
            if model.RampConstraintMoins[area,tech]>0 :
                return model.energy[area,t+1,tech]  - model.energy[area,t,tech] >= - model.capacity[area,tech]*model.RampConstraintMoins[area,tech] ;
            else:
                return Constraint.Skip
        model.rampCtrMoins = Constraint(model.AREAS,model.TIMESTAMP_MinusOne,model.TECHNOLOGIES,rule=rampCtrMoins_rule)
        
    if "RampConstraintPlus2" in TechParameters:
        def rampCtrPlus2_rule(model,area,t,tech): #INEQ forall t<
            if model.RampConstraintPlus2[area,tech]>0 :
                var=(model.energy[area,t+2,tech]+model.energy[area,t+3,tech])/2 -  (model.energy[area,t+1,tech]+model.energy[area,t,tech])/2;
                return var <= model.capacity[area,tech]*model.RampConstraintPlus[area,tech] ;
            else:
                return Constraint.Skip
        model.rampCtrPlus2 = Constraint(model.AREAS,model.TIMESTAMP_MinusThree,model.TECHNOLOGIES,rule=rampCtrPlus2_rule)
        
    if "RampConstraintMoins2" in TechParameters:
        def rampCtrMoins2_rule(model,area,t,tech): #INEQ forall t<
            if model.RampConstraintMoins2[area,tech]>0 :
                var=(model.energy[area,t+2,tech]+model.energy[area,t+3,tech])/2 -  (model.energy[area,t+1,tech]+model.energy[area,t,tech])/2;
                return var >= - model.capacity[area,tech]*model.RampConstraintMoins2[area,tech] ;
            else:
                return Constraint.Skip
        model.rampCtrMoins2 = Constraint(model.AREAS,model.TIMESTAMP_MinusThree,model.TECHNOLOGIES,rule=rampCtrMoins2_rule)

    ### Contraintes de rampe
    # C1
    #     subject to rampCtrPlus{a in AREAS, h in TIMESTAMPMOINS1, t in TECHNOLOGIES : RampConstraintPlus[a,t]>0 } :
    #         energy[a,h+1,t] - energy[a,h,t] <= capacity[a,t]*RampConstraintPlus[a,t] ;
 
    # subject to rampCtrMoins{a in AREAS, h in TIMESTAMPMOINS1, t in TECHNOLOGIES : RampConstraintMoins[a,t]>0 } :
    #  energy[a,h+1,t] - energy[a,h,t] >= - capacity[a,t]*RampConstraintMoins[a,t] ;
 
    #  /*contrainte de rampe2 */
    # subject to rampCtrPlus2{a in AREAS, h in TIMESTAMPMOINS4, t in TECHNOLOGIES : RampConstraintPlus2[a,t]>0 } :
    #  (energy[a,h+2,t]+energy[a,h+3,t])/2 -  (energy[a,h+1,t]+energy[a,h,t])/2 <= capacity[a,t]*RampConstraintPlus2[a,t] ;
 
    # subject to rampCtrMoins2{a in AREAS, h in TIMESTAMPMOINS4, t in TECHNOLOGIES : RampConstraintMoins2[a,t]>0 } :
    #   (energy[a,h+2,t]+energy[a,h+3,t])/2 -  (energy[a,h+1,t]+energy[a,h,t])/2 >= - capacity[a,t]*RampConstraintMoins2[a,t] ;
    
    return model ;


def GetElectricSystemModel_PlaningMultiNode_with1Storage(areaConsumption,availabilityFactor,
                                                          TechParameters,ExchangeParameters,StorageParameters,isAbstract=False,
                                                          solver='mosek'):
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
    #isAbstract=False
    availabilityFactor.isna().sum()
    
    ### Cleaning
    availabilityFactor=availabilityFactor.fillna(method='pad');
    areaConsumption=areaConsumption.fillna(method='pad');
    
    ### obtaining dimensions values 
    TECHNOLOGIES = set(TechParameters.index.get_level_values('TECHNOLOGIES').unique())
    STOCK_TECHNO = set(StorageParameters.index.get_level_values('STOCK_TECHNO').unique())
    TIMESTAMP = set(areaConsumption.index.get_level_values('TIMESTAMP').unique())
    TIMESTAMP_list = areaConsumption.index.get_level_values('TIMESTAMP').unique()
    AREAS = set(areaConsumption.index.get_level_values('AREAS').unique())
    
    #####################
    #    Pyomo model    #
    #####################
    
    if (isAbstract) : 
        model = AbstractModel()
    else:
        model = ConcreteModel() 
        
    ###############
    # Sets       ##
    ###############
    
    #Simple
    model.AREAS= Set(initialize=AREAS,doc = "Area",ordered=False)
    model.TECHNOLOGIES = Set(initialize=TECHNOLOGIES,ordered=False)
    model.STOCK_TECHNO = Set(initialize=STOCK_TECHNO,ordered=False)
    model.TIMESTAMP = Set(initialize=TIMESTAMP,ordered=False)
    
    #Subset of Simple only required if ramp constraint
    model.TIMESTAMP_MinusOne = Set(initialize=TIMESTAMP_list[: len(TIMESTAMP) - 1],ordered=False)
    model.TIMESTAMP_MinusThree = Set(initialize=TIMESTAMP_list[: len(TIMESTAMP) - 3],ordered=False)
    
    #Products
    model.TIMESTAMP_TECHNOLOGIES =  model.TIMESTAMP *model.TECHNOLOGIES
    model.AREAS_AREAS= model.AREAS* model.AREAS
    model.AREAS_TECHNOLOGIES= model.AREAS *model.TECHNOLOGIES
    model.AREAS_STOCKTECHNO=model.AREAS*model.STOCK_TECHNO
    model.AREAS_TIMESTAMP=model.AREAS * model.TIMESTAMP 
    model.AREAS_TIMESTAMP_TECHNOLOGIES= model.AREAS*model.TIMESTAMP * model.TECHNOLOGIES


    ###############
    # Parameters ##
    ###############
    
    model.areaConsumption =     Param(model.AREAS_TIMESTAMP,
                                      initialize=areaConsumption.loc[:, "areaConsumption"].squeeze().to_dict(), domain=Any,mutable=True)
    model.availabilityFactor =  Param(  model.AREAS_TIMESTAMP_TECHNOLOGIES, domain=PercentFraction,default=1,
                                      initialize=availabilityFactor.squeeze().to_dict())

    model.maxExchangeCapacity = Param( model.AREAS_AREAS,  initialize=ExchangeParameters.squeeze().to_dict(), domain=NonNegativeReals,default=0)
    
    #with test of existing columns on TechParameters
    for COLNAME in TechParameters:
            exec("model."+COLNAME+" =          Param(model.AREAS_TECHNOLOGIES, domain=NonNegativeReals,default=0,"+
                                      "initialize=TechParameters."+COLNAME+".squeeze().to_dict())")
    
    for COLNAME in StorageParameters:
            exec("model."+COLNAME+" =Param(model.AREAS_STOCKTECHNO,domain=NonNegativeReals,default=0,"+
                                      "initialize=StorageParameters."+COLNAME+".squeeze().to_dict())")
    
    ################
    # Variables    #
    ################
    
    ###Production means :
    model.energy=Var(model.AREAS,model.TIMESTAMP, model.TECHNOLOGIES, domain=NonNegativeReals) ### Energy produced by a production mean at time t
    model.exchange=Var(model.AREAS_AREAS,model.TIMESTAMP)  
    model.energyCosts=Var(model.AREAS,model.TECHNOLOGIES)   ### Cost of energy by a production mean for area at time t (explicitely defined by constraint energyCostsCtr)
    
    model.capacityCosts=Var(model.AREAS,model.TECHNOLOGIES)  ### Cost of energy for a production mean, explicitely defined by definition energyCostsDef    
    model.capacity=Var(model.AREAS,model.TECHNOLOGIES, domain=NonNegativeReals) ### Energy produced by a production mean at time t
    
    ###Storage means :
    model.storage=Var(model.AREAS,model.TIMESTAMP,model.STOCK_TECHNO) ### Energy stored in a storage mean at time t (if Storage < 0, then the energy is taken from the storage, if Storage > 0, then the is stored)
    model.stockLevel=Var(model.AREAS,model.TIMESTAMP,model.STOCK_TECHNO,domain=NonNegativeReals) ### level of the energy stock in a storage mean at time t
    model.storageCosts=Var(model.AREAS,model.STOCK_TECHNO)  ### Cost of storage for a storage mean, explicitely defined by definition storageCostsDef    
    model.Cmax=Var(model.AREAS,model.STOCK_TECHNO) # Maximum capacity of a storage mean
    model.Pmax=Var(model.AREAS,model.STOCK_TECHNO) # Maximum flow of energy in/out of a storage mean
    
    model.dual = Suffix(direction=Suffix.IMPORT)
    model.rc = Suffix(direction=Suffix.IMPORT)
    #model.slack = Suffix(direction=Suffix.IMPORT)
    
    ########################
    # Objective Function   #
    ########################
    
    def ObjectiveFunction_rule(model): #OBJ
    	return sum(model.energyCosts[area,tech]+model.capacityCosts[area,tech] for tech in model.TECHNOLOGIES for area in model.AREAS)+sum(model.storageCosts[area,s_tech] for s_tech in model.STOCK_TECHNO for area in model.AREAS)
    model.OBJ = Objective(rule=ObjectiveFunction_rule, sense=minimize)
    
    
    
    #################
    # Constraints   #
    ################# 
    
    #### 1 - Basics 
    ########
    
    
    # energyCosts definition Constraints
    # AREAS x TECHNOLOGIES       
    def energyCostsDef_rule(model,area,tech): #EQ forall tech in TECHNOLOGIES   energyCosts  = sum{t in TIMESTAMP} energyCost[tech]*energy[t,tech] / 1E6;
        temp=model.energyCost[area,tech]#/10**6 ;
        return sum(temp*model.energy[area,t,tech] for t in model.TIMESTAMP) == model.energyCosts[area,tech];
    model.energyCostsDef = Constraint(model.AREAS,model.TECHNOLOGIES, rule=energyCostsDef_rule)    
    
    # capacityCosts definition Constraints
    # AREAS x TECHNOLOGIES    
    def capacityCostsDef_rule(model,area,tech): #EQ forall tech in TECHNOLOGIES   energyCosts  = sum{t in TIMESTAMP} energyCost[tech]*energy[t,tech] / 1E6;
        temp=model.capacityCosts[area,tech]#/10**6 ;
        return model.capacityCost[area,tech]*len(model.TIMESTAMP)/8760*model.capacity[area,tech]  == model.capacityCosts[area,tech] #  .. ....... / 10**6
    model.capacityCostsCtr = Constraint(model.AREAS,model.TECHNOLOGIES, rule=capacityCostsDef_rule)    
    
    # capacityCosts definition Constraints
    # AREAS x STOCK_TECHNO   
    def storageCostsDef_rule(model,area,s_tech): #EQ forall s_tech in STOCK_TECHNO storageCosts = storageCost[area, s_tech]*c_max[area, s_tech] / 1E6;
        return model.storageCost[area,s_tech]*model.Cmax[area,s_tech] == model.storageCosts[area,s_tech]#/10**6 ;;
    model.storageCostsDef = Constraint(model.AREAS,model.STOCK_TECHNO, rule=storageCostsDef_rule)
    
    #Capacity constraint
    #AREAS x TIMESTAMP x TECHNOLOGIES
    def CapacityCtr_rule(model,area,t,tech): #INEQ forall t, tech 
    	return model.capacity[area,tech] * model.availabilityFactor[area,t,tech] >= model.energy[area,t,tech]
    model.CapacityCtr = Constraint(model.AREAS,model.TIMESTAMP,model.TECHNOLOGIES, rule=CapacityCtr_rule)

    #Exchange capacity constraint (duplicate of variable definition)
    # AREAS x AREAS x TIMESTAMP
    def exchangeCtrPlus_rule(model,a, b, t): #INEQ forall area.axarea.b in AREASxAREAS  t in TIMESTAMP
        if a!=b:
            return model.exchange[a,b,t]  <= model.maxExchangeCapacity[a,b] ;
        else:
            return model.exchange[a, a, t] == 0
    model.exchangeCtrPlus = Constraint(model.AREAS,model.AREAS,model.TIMESTAMP, rule=exchangeCtrPlus_rule)

    def exchangeCtrMoins_rule(model,a, b, t): #INEQ forall area.axarea.b in AREASxAREAS  t in TIMESTAMP
        if a!=b:
            return model.exchange[a,b,t]  >= -model.maxExchangeCapacity[a,b] ;
        else:
            return model.exchange[a, a, t] == 0
    model.exchangeCtrMoins = Constraint(model.AREAS,model.AREAS,model.TIMESTAMP, rule=exchangeCtrMoins_rule)

    def exchangeCtr2_rule(model,a, b, t): #INEQ forall area.axarea.b in AREASxAREAS  t in TIMESTAMP
        return model.exchange[a,b,t]  == -model.exchange[b,a,t] ;
    model.exchangeCtr2 = Constraint(model.AREAS,model.AREAS,model.TIMESTAMP, rule=exchangeCtr2_rule)
  
    #Storage max capacity constraint
    #AREAS x STOCK_TECHNO
    def storageCapacity_rule(model,area,s_tech): #INEQ forall s_tech 
    	return model.Cmax[area,s_tech]  <= model.c_max[area,s_tech]
    model.storageCapacityCtr = Constraint(model.AREAS,model.STOCK_TECHNO, rule=storageCapacity_rule)   
    
    #Storage max power constraint
    #AREAS x STOCK_TECHNO
    def storagePower_rule(model,area,s_tech): #INEQ forall s_tech 
    	return model.Pmax[area,s_tech]  <= model.p_max[area,s_tech]
    model.storagePowerCtr = Constraint(model.AREAS,model.STOCK_TECHNO, rule=storagePower_rule) 
    
    #contraintes de stock puissance
    #AREAS x TIMESTAMP x STOCK_TECHNO
    def StoragePowerUB_rule(model,area, t,s_tech):  # INEQ forall t
        return model.storage[area,t,s_tech] - model.Pmax[area,s_tech] <= 0
    model.StoragePowerUBCtr = Constraint(model.AREAS,model.TIMESTAMP,model.STOCK_TECHNO, rule=StoragePowerUB_rule)

    def StoragePowerLB_rule(model,area,t,s_tech,):  # INEQ forall t
        return  model.storage[area,t,s_tech] + model.Pmax[area,s_tech] >= 0
    model.StoragePowerLBCtr = Constraint(model.AREAS,model.TIMESTAMP,model.STOCK_TECHNO, rule=StoragePowerLB_rule)

    #contraintes de stock capacité
    #AREAS x TIMESTAMP x STOCK_TECHNO
    def StockLevel_rule(model,area, t,s_tech):  # EQ forall t
        if t>1 :
            return model.stockLevel[area,t,s_tech] == model.stockLevel[area,t-1,s_tech] + model.storage[area,t,s_tech]
        else :
            return model.stockLevel[area,t,s_tech] == 0
    model.StockLevelCtr = Constraint(model.AREAS,model.TIMESTAMP,model.STOCK_TECHNO, rule=StockLevel_rule)
    
    def StockCapacity_rule(model,area,t,s_tech,):  # INEQ forall t
        return model.stockLevel[area,t,s_tech] <= model.Cmax[area,s_tech]
    model.StockCapacityCtr = Constraint(model.AREAS,model.TIMESTAMP,model.STOCK_TECHNO, rule=StockCapacity_rule)
    

    #contrainte d'equilibre offre demande
    #AREAS x TIMESTAMP x TECHNOLOGIES
    def energyCtr_rule(model,area,t): #INEQ forall t
    	return sum(model.energy[area,t,tech] for tech in model.TECHNOLOGIES ) + sum(model.exchange[b,area,t] for b in model.AREAS )+sum(model.storage[area,t,s_tech] for s_tech in model.STOCK_TECHNO) >= model.areaConsumption[area,t]
    model.energyCtr = Constraint(model.AREAS,model.TIMESTAMP,rule=energyCtr_rule)


    #### 2 - Optional 
    ########

    #contrainte de stock annuel 
    if "maxCapacity" in TechParameters:
        def maxCapacity_rule(model,area,tech) : #INEQ forall t, tech 
            if model.maxCapacity[area,tech]>0 :
                return model.maxCapacity[area,tech] >= model.capacity[area,tech] 
            else:
                return Constraint.Skip
        model.maxCapacityCtr = Constraint(model.AREAS,model.TECHNOLOGIES, rule=maxCapacity_rule)
    
    if "minCapacity" in TechParameters:
        def minCapacity_rule(model,area,tech) : #INEQ forall t, tech 
            if model.minCapacity[area,tech]>0 :
                return model.minCapacity[area,tech] <= model.capacity[area,tech] 
            else:
                return Constraint.Skip
        model.minCapacityCtr = Constraint(model.AREAS,model.TECHNOLOGIES, rule=minCapacity_rule)
    
    #contrainte de stock annuel 
    #AREAS x TECHNOLOGIES
    if "EnergyNbhourCap" in TechParameters:
        def storageCtr_rule(model,area,tech) : #INEQ forall t, tech 
            if model.EnergyNbhourCap[area,tech]>0 :
                return model.EnergyNbhourCap[area,tech]*model.capacity[area,tech] >= sum(model.energy[area,t,tech] for t in model.TIMESTAMP)
            else:
                return Constraint.Skip
        model.storageCtr = Constraint(model.AREAS,model.TECHNOLOGIES, rule=storageCtr_rule)

    if "RampConstraintPlus" in TechParameters:
        def rampCtrPlus_rule(model,area,t,tech): #INEQ forall t<
            if model.RampConstraintPlus[area,tech]>0 :
                return model.energy[area,t+1,tech]  - model.energy[area,t,tech] <= model.capacity[area,tech]*model.RampConstraintPlus[area,tech] ;
            else:
                return Constraint.Skip
        model.rampCtrPlus = Constraint(model.AREAS,model.TIMESTAMP_MinusOne,model.TECHNOLOGIES,rule=rampCtrPlus_rule)
        
    if "RampConstraintMoins" in TechParameters:
        def rampCtrMoins_rule(model,area,t,tech): #INEQ forall t<
            if model.RampConstraintMoins[area,tech]>0 :
                return model.energy[area,t+1,tech]  - model.energy[area,t,tech] >= - model.capacity[area,tech]*model.RampConstraintMoins[area,tech] ;
            else:
                return Constraint.Skip
        model.rampCtrMoins = Constraint(model.AREAS,model.TIMESTAMP_MinusOne,model.TECHNOLOGIES,rule=rampCtrMoins_rule)
        
    if "RampConstraintPlus2" in TechParameters:
        def rampCtrPlus2_rule(model,area,t,tech): #INEQ forall t<
            if model.RampConstraintPlus2[area,tech]>0 :
                var=(model.energy[area,t+2,tech]+model.energy[area,t+3,tech])/2 -  (model.energy[area,t+1,tech]+model.energy[area,t,tech])/2;
                return var <= model.capacity[area,tech]*model.RampConstraintPlus[area,tech] ;
            else:
                return Constraint.Skip
        model.rampCtrPlus2 = Constraint(model.AREAS,model.TIMESTAMP_MinusThree,model.TECHNOLOGIES,rule=rampCtrPlus2_rule)
        
    if "RampConstraintMoins2" in TechParameters:
        def rampCtrMoins2_rule(model,area,t,tech): #INEQ forall t<
            if model.RampConstraintMoins2[area,tech]>0 :
                var=(model.energy[area,t+2,tech]+model.energy[area,t+3,tech])/2 -  (model.energy[area,t+1,tech]+model.energy[area,t,tech])/2;
                return var >= - model.capacity[area,tech]*model.RampConstraintMoins2[area,tech] ;
            else:
                return Constraint.Skip
        model.rampCtrMoins2 = Constraint(model.AREAS,model.TIMESTAMP_MinusThree,model.TECHNOLOGIES,rule=rampCtrMoins2_rule)

    ### Contraintes de rampe
    # C1
    #     subject to rampCtrPlus{a in AREAS, h in TIMESTAMPMOINS1, t in TECHNOLOGIES : RampConstraintPlus[a,t]>0 } :
    #         energy[a,h+1,t] - energy[a,h,t] <= capacity[a,t]*RampConstraintPlus[a,t] ;
 
    # subject to rampCtrMoins{a in AREAS, h in TIMESTAMPMOINS1, t in TECHNOLOGIES : RampConstraintMoins[a,t]>0 } :
    #  energy[a,h+1,t] - energy[a,h,t] >= - capacity[a,t]*RampConstraintMoins[a,t] ;
 
    #  /*contrainte de rampe2 */
    # subject to rampCtrPlus2{a in AREAS, h in TIMESTAMPMOINS4, t in TECHNOLOGIES : RampConstraintPlus2[a,t]>0 } :
    #  (energy[a,h+2,t]+energy[a,h+3,t])/2 -  (energy[a,h+1,t]+energy[a,h,t])/2 <= capacity[a,t]*RampConstraintPlus2[a,t] ;
 
    # subject to rampCtrMoins2{a in AREAS, h in TIMESTAMPMOINS4, t in TECHNOLOGIES : RampConstraintMoins2[a,t]>0 } :
    #   (energy[a,h+2,t]+energy[a,h+3,t])/2 -  (energy[a,h+1,t]+energy[a,h,t])/2 >= - capacity[a,t]*RampConstraintMoins2[a,t] ;
    return model;

