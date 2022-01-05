#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 19:07:50 2020

@author: robin.girard
"""


from __future__ import division
from pyomo.environ import *
from pyomo.core import *
from functions.f_tools import *

def GetElectricSystemModel_MultiResources_SingleNode(areaConsumption, availabilityFactor, TechParameters, ResParameters,conversionFactor,isAbstract=False):
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
    conversionFactor = conversionFactor.fillna(method='pad');

    ### obtaining dimensions values

    TECHNOLOGIES= set(TechParameters.index.get_level_values('TECHNOLOGIES').unique())
    RESOURCES= set(ResParameters.index.get_level_values('RESOURCES').unique())
    TIMESTAMP= set(areaConsumption.index.get_level_values('TIMESTAMP').unique())
    TIMESTAMP_list= areaConsumption.index.get_level_values('TIMESTAMP').unique()

    #####################
    #    Pyomo model    #
    #####################

    if (isAbstract) :
        model = pyomo.environ.AbstractModel()
    else:
        model = pyomo.environ.ConcreteModel()

    ###############
    # Sets       ##
    ###############
    model.TECHNOLOGIES  = Set(initialize=TECHNOLOGIES,ordered=False)
    model.RESOURCES  = Set(initialize=RESOURCES,ordered=False)
    model.TIMESTAMP     = Set(initialize=TIMESTAMP,ordered=False)
    model.TIMESTAMP_TECHNOLOGIES =  model.TIMESTAMP * model.TECHNOLOGIES
    model.RESOURCES_TECHNOLOGIES  = model.RESOURCES * model.TECHNOLOGIES
    model.TIMESTAMP_RESOURCES = model.TIMESTAMP * model.RESOURCES

    #Subset of Simple only required if ramp constraint
    model.TIMESTAMP_MinusOne = Set(initialize=TIMESTAMP_list[: len(TIMESTAMP) - 1],ordered=False)
    model.TIMESTAMP_MinusThree = Set(initialize=TIMESTAMP_list[: len(TIMESTAMP) - 3],ordered=False)


    ###############
    # Parameters ##
    ###############

    model.areaConsumption =     Param(model.TIMESTAMP_RESOURCES,default=0,
                                      initialize=areaConsumption.loc[:,"areaConsumption"].squeeze().to_dict(),domain=Any)
    model.availabilityFactor =  Param( model.TIMESTAMP_TECHNOLOGIES, domain=PercentFraction,default=1,
                                      initialize=availabilityFactor.loc[:,"availabilityFactor"].squeeze().to_dict())
    model.conversionFactor = Param(model.RESOURCES_TECHNOLOGIES, default=0,
                                   initialize=conversionFactor.loc[:, "conversionFactor"].squeeze().to_dict())
    model.importCost = Param(model.TIMESTAMP_RESOURCES, mutable=True,default=0,
                                      initialize=ResParameters.loc[:,"importCost"].squeeze().to_dict(), domain=Any)
    #with test of existing columns on TechParameters
    for COLNAME in TechParameters:
        if COLNAME not in ["TECHNOLOGIES","AREAS"]: ### each column in TechParameters will be a parameter
            exec("model."+COLNAME+" =          Param(model.TECHNOLOGIES, domain=NonNegativeReals,default=0,"+
                                      "initialize=TechParameters."+COLNAME+".squeeze().to_dict())")

    ################
    # Variables    #
    ################
    
    #In this section, variables are separated in two categories : decision variables wich are the reals variables of the otimisation problem (these are noted Dvar), and problem variables which are resulting of calculation and are convenient for the readability and the analyse of results (these are noted Pvar)
    
    model.power_Dvar = Var(model.TIMESTAMP, model.TECHNOLOGIES, domain=NonNegativeReals) ### Power of a conversion mean at time t
    model.powerCosts_Pvar = Var(model.TECHNOLOGIES)  ### Marginal cost for a conversion mean, explicitely defined by definition powerCostsDef
    model.capacity_Dvar = Var(model.TECHNOLOGIES, domain=NonNegativeReals) ### Capacity of a conversion mean
    model.capacityCosts_Pvar = Var(model.TECHNOLOGIES) ### Fixed costs for a conversion mean, explicitely defined by definition capacityCostsDef
    model.importation_Dvar = Var(model.TIMESTAMP, model.RESOURCES, domain=NonNegativeReals,initialize=0) ### Improtation of a resource at time t
    model.importCosts_Pvar = Var(model.RESOURCES) ### Cost of ressource imported, explicitely defined by definition importCostsDef
    model.energy_Pvar = Var(model.TIMESTAMP, model.RESOURCES)  ### Amount of a resource at time t
    
    model.dual = Suffix(direction=Suffix.IMPORT)
    model.rc = Suffix(direction=Suffix.IMPORT)
    model.slack = Suffix(direction=Suffix.IMPORT)
    
    ########################
    # Objective Function   #
    ########################
    
    def ObjectiveFunction_rule(model): #OBJ
    	return (sum(model.powerCosts_Pvar[tech] + model.capacityCosts_Pvar[tech] for tech in model.TECHNOLOGIES) + sum(model.importCosts_Pvar[res] for res in model.RESOURCES))
    model.OBJ = Objective(rule=ObjectiveFunction_rule, sense=minimize)
    
    
    #################
    # Constraints   #
    #################
    
    
    # energyCosts definition Constraints       
    def powerCostsDef_rule(model,tech): #EQ forall tech in TECHNOLOGIES powerCosts  = sum{t in TIMESTAMP} powerCost[tech]*power[t,tech] / 1E6;
        return sum(model.powerCost[tech]*model.power_Dvar[t,tech] for t in model.TIMESTAMP) == model.powerCosts_Pvar[tech]
    model.powerCostsCtr = Constraint(model.TECHNOLOGIES, rule=powerCostsDef_rule)    
    
    # capacityCosts definition Constraints
    def capacityCostsDef_rule(model,tech): #EQ forall tech in TECHNOLOGIES   capacityCosts  = sum{t in TIMESTAMP} capacityCost[tech]*capacity[t,tech] / 1E6;
        return model.capacityCost[tech]*model.capacity_Dvar[tech] == model.capacityCosts_Pvar[tech]
    model.capacityCostsCtr = Constraint(model.TECHNOLOGIES, rule=capacityCostsDef_rule)    

    # importCosts definition Constraints
    def importCostsDef_rule(model,res):  # ;
        return sum((model.importCost[t,res] * model.importation_Dvar[t, res]) for t in model.TIMESTAMP) == model.importCosts_Pvar[res]
    model.importCostsCtr = Constraint(model.RESOURCES, rule=importCostsDef_rule)

    #Capacity constraint
    def Capacity_rule(model,t,tech): #INEQ forall t, tech 
    	return model.capacity_Dvar[tech] * model.availabilityFactor[t,tech] >= model.power_Dvar[t,tech]
    model.CapacityCtr = Constraint(model.TIMESTAMP,model.TECHNOLOGIES, rule=Capacity_rule)    
    
    # Ressource production constraint
    def Production_rule(model, t, res):  # EQ forall t, res
        return sum(model.power_Dvar[t, tech] * model.conversionFactor[res, tech] for tech in model.TECHNOLOGIES) + model.importation_Dvar[t, res] == model.energy_Pvar[t, res]
    model.ProductionCtr = Constraint(model.TIMESTAMP, model.RESOURCES, rule=Production_rule)
    
    #contrainte d'equilibre offre demande 
    def energyCtr_rule(model,t,res): #INEQ forall t
    	return model.energy_Pvar[t,res]  >= model.areaConsumption[t,res]
    model.energyCtr = Constraint(model.TIMESTAMP,model.RESOURCES,rule=energyCtr_rule)
    
    if "maxCapacity" in TechParameters:
        def maxCapacity_rule(model,tech) : #INEQ forall t, tech 
            return model.maxCapacity[tech] >= model.capacity_Dvar[tech] 
        model.maxCapacityCtr = Constraint(model.TECHNOLOGIES, rule=maxCapacity_rule)
    
    if "minCapacity" in TechParameters:
        def minCapacity_rule(model,tech) : #INEQ forall t, tech 
            return model.minCapacity[tech] <= model.capacity_Dvar[tech] 
        model.minCapacityCtr = Constraint(model.TECHNOLOGIES, rule=minCapacity_rule)
    
    if "EnergyNbhourCap" in TechParameters:
        def storage_rule(model,tech) : #INEQ forall t, tech 
            if model.EnergyNbhourCap[tech]>0 :
                return model.EnergyNbhourCap[tech]*model.capacity_Dvar[tech] >= sum(model.power_Dvar[t,tech] for t in model.TIMESTAMP)
            else:
                return Constraint.Skip
        model.storageCtr = Constraint(model.TECHNOLOGIES, rule=storage_rule)

    if "RampConstraintPlus" in TechParameters:
        def rampCtrPlus_rule(model,t,tech): #INEQ forall t<
            if model.RampConstraintPlus[tech]>0 :
                return model.power_Dvar[t+1,tech]  - model.power_Dvar[t,tech] <= model.capacity_Dvar[tech]*model.RampConstraintPlus[tech] ;
            else:
                return Constraint.Skip
        model.rampCtrPlus = Constraint(model.TIMESTAMP_MinusOne,model.TECHNOLOGIES,rule=rampCtrPlus_rule)
        
    if "RampConstraintMoins" in TechParameters:
        def rampCtrMoins_rule(model,t,tech): #INEQ forall t<
            if model.RampConstraintMoins[tech]>0 :
                return model.power_Dvar[t+1,tech]  - model.power_Dvar[t,tech] >= - model.capacity_Dvar[tech]*model.RampConstraintMoins[tech] ;
            else:
                return Constraint.Skip
        model.rampCtrMoins = Constraint(model.TIMESTAMP_MinusOne,model.TECHNOLOGIES,rule=rampCtrMoins_rule)
        
    if "RampConstraintPlus2" in TechParameters:
        def rampCtrPlus2_rule(model,t,tech): #INEQ forall t<
            if model.RampConstraintPlus2[tech]>0 :
                var=(model.power_Dvar[t+2,tech]+model.power_Dvar[t+3,tech])/2 -  (model.power_Dvar[t+1,tech]+model.power_Dvar[t,tech])/2;
                return var <= model.capacity_Dvar[tech]*model.RampConstraintPlus[tech] ;
            else:
                return Constraint.Skip
        model.rampCtrPlus2 = Constraint(model.TIMESTAMP_MinusThree,model.TECHNOLOGIES,rule=rampCtrPlus2_rule)
        
    if "RampConstraintMoins2" in TechParameters:
        def rampCtrMoins2_rule(model,t,tech): #INEQ forall t<
            if model.RampConstraintMoins2[tech]>0 :
                var=(model.power_Dvar[t+2,tech]+model.power_Dvar[t+3,tech])/2 -  (model.power_Dvar[t+1,tech]+model.power_Dvar[t,tech])/2;
                return var >= - model.capacity_Dvar[tech]*model.RampConstraintMoins2[tech] ;
            else:
                return Constraint.Skip
        model.rampCtrMoins2 = Constraint(model.TIMESTAMP_MinusThree,model.TECHNOLOGIES,rule=rampCtrMoins2_rule)
    
    return model ;

def GetElectricSystemModel_MultiResources_SingleNode_WithStorage(areaConsumption, availabilityFactor, TechParameters,ResParameters,conversionFactor,StorageParameters,storageFactor,isAbstract=False):
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
    conversionFactor = conversionFactor.fillna(method='pad');

    ### obtaining dimensions values

    TECHNOLOGIES= set(TechParameters.index.get_level_values('TECHNOLOGIES').unique())
    STOCK_TECHNO= set(StorageParameters.index.get_level_values('STOCK_TECHNO').unique())
    RESOURCES= set(ResParameters.index.get_level_values('RESOURCES').unique())
    TIMESTAMP= set(areaConsumption.index.get_level_values('TIMESTAMP').unique())
    TIMESTAMP_list= areaConsumption.index.get_level_values('TIMESTAMP').unique()

    #####################
    #    Pyomo model    #
    #####################

    if (isAbstract) :
        model = pyomo.environ.AbstractModel()
    else:
        model = pyomo.environ.ConcreteModel()

    ###############
    # Sets       ##
    ###############
    model.TECHNOLOGIES  = Set(initialize=TECHNOLOGIES,ordered=False)
    model.STOCK_TECHNO = Set(initialize=STOCK_TECHNO,ordered=False)  
    model.RESOURCES  = Set(initialize=RESOURCES,ordered=False)
    model.TIMESTAMP     = Set(initialize=TIMESTAMP,ordered=False)
    model.TIMESTAMP_TECHNOLOGIES =  model.TIMESTAMP * model.TECHNOLOGIES
    model.RESOURCES_TECHNOLOGIES  = model.RESOURCES * model.TECHNOLOGIES
    model.RESOURCES_STOCKTECHNO =model.RESOURCES * model.STOCK_TECHNO
    model.TIMESTAMP_RESOURCES = model.TIMESTAMP * model.RESOURCES

    #Subset of Simple only required if ramp constraint
    model.TIMESTAMP_MinusOne = Set(initialize=TIMESTAMP_list[: len(TIMESTAMP) - 1],ordered=False)
    model.TIMESTAMP_MinusThree = Set(initialize=TIMESTAMP_list[: len(TIMESTAMP) - 3],ordered=False)


    ###############
    # Parameters ##
    ###############

    model.areaConsumption =     Param(model.TIMESTAMP_RESOURCES,default=0,
                                      initialize=areaConsumption.loc[:,"areaConsumption"].squeeze().to_dict(),domain=Any)
    model.availabilityFactor =  Param( model.TIMESTAMP_TECHNOLOGIES, domain=PercentFraction,default=1,
                                      initialize=availabilityFactor.loc[:,"availabilityFactor"].squeeze().to_dict())
    model.conversionFactor = Param(model.RESOURCES_TECHNOLOGIES, default=0,
                                   initialize=conversionFactor.loc[:, "conversionFactor"].squeeze().to_dict())
    model.importCost = Param(model.TIMESTAMP_RESOURCES, mutable=True,default=0,
                                      initialize=ResParameters.loc[:,"importCost"].squeeze().to_dict(), domain=Any)
    #with test of existing columns on TechParameters
    for COLNAME in TechParameters:
        if COLNAME not in ["TECHNOLOGIES","AREAS"]: ### each column in TechParameters will be a parameter
            exec("model."+COLNAME+" =          Param(model.TECHNOLOGIES, domain=NonNegativeReals,default=0,"+
                                      "initialize=TechParameters."+COLNAME+".squeeze().to_dict())")
    for COLNAME in StorageParameters:
        if COLNAME not in ["STOCK_TECHNO","AREAS"]: ### each column in StorageParameters will be a parameter
            exec("model."+COLNAME+" =Param(model.STOCK_TECHNO,domain=Any,default=0,"+
                                      "initialize=StorageParameters."+COLNAME+".squeeze().to_dict())")
            
    for COLNAME in storageFactor:
        exec("model."+COLNAME+" =Param(model.RESOURCES_STOCKTECHNO,domain=NonNegativeReals,default=0,"+
                                      "initialize=storageFactor."+COLNAME+".squeeze().to_dict())")        
            
    ################
    # Variables    #
    ################

    #In this section, variables are separated in two categories : decision variables wich are the reals variables of the otimisation problem (these are noted Dvar), and problem variables which are resulting of calculation and are convenient for the readability and the analyse of results (these are noted Pvar)
    
    ### Operation variables
    model.power_Dvar = Var(model.TIMESTAMP, model.TECHNOLOGIES, domain=NonNegativeReals) ### Power of a conversion mean at time t
    model.powerCosts_Pvar = Var(model.TECHNOLOGIES)  ### Marginal cost for a conversion mean, explicitely defined by definition powerCostsDef
    model.importation_Dvar = Var(model.TIMESTAMP, model.RESOURCES, domain=NonNegativeReals,initialize=0) ### Improtation of a resource at time t
    model.importCosts_Pvar = Var(model.RESOURCES) ### Cost of ressource imported, explicitely defined by definition importCostsDef
    model.energy_Pvar = Var(model.TIMESTAMP, model.RESOURCES)  ### Amount of a resource at time t
    
    ### Planing variables
    model.capacity_Dvar = Var(model.TECHNOLOGIES, domain=NonNegativeReals) ### Capacity of a conversion mean
    model.capacityCosts_Pvar = Var(model.TECHNOLOGIES) ### Fixed costs for a conversion mean, explicitely defined by definition capacityCostsDef
    
    ### Storage variables
    model.storageIn_Pvar=Var(model.TIMESTAMP,model.RESOURCES,model.STOCK_TECHNO,domain=NonNegativeReals) ### Energy stored in a storage mean at time t 
    model.storageOut_Pvar=Var(model.TIMESTAMP,model.RESOURCES,model.STOCK_TECHNO,domain=NonNegativeReals) ### Energy taken out of the in a storage mean at time t
    model.storageConsumption_Pvar=Var(model.TIMESTAMP,model.RESOURCES,model.STOCK_TECHNO,domain=NonNegativeReals) ### Energy consumed the in a storage mean at time t (other than the one stored)
    model.stockLevel_Pvar=Var(model.TIMESTAMP,model.STOCK_TECHNO,domain=NonNegativeReals) ### level of the energy stock in a storage mean at time t
    model.storageCosts_Pvar=Var(model.STOCK_TECHNO)  ### Cost of storage for a storage mean, explicitely defined by definition storageCostsDef    
    model.Cmax_Dvar=Var(model.STOCK_TECHNO) # Maximum capacity of a storage mean
    model.Pmax_Dvar=Var(model.STOCK_TECHNO) # Maximum flow of energy in/out of a storage mean
    
    ### Other variables
    model.dual = Suffix(direction=Suffix.IMPORT)
    model.rc = Suffix(direction=Suffix.IMPORT)
    model.slack = Suffix(direction=Suffix.IMPORT)
    
    ########################
    # Objective Function   #
    ########################
    
    def ObjectiveFunction_rule(model): #OBJ
    	return (sum(model.powerCosts_Pvar[tech] + model.capacityCosts_Pvar[tech] for tech in model.TECHNOLOGIES) + sum(model.importCosts_Pvar[res] for res in model.RESOURCES))+sum(model.storageCosts_Pvar[s_tech] for s_tech in STOCK_TECHNO)
    model.OBJ = Objective(rule=ObjectiveFunction_rule, sense=minimize)
    
    
    #################
    # Constraints   #
    #################
    
    
    # energyCosts definition Constraint    
    def powerCostsDef_rule(model,tech): #EQ forall tech in TECHNOLOGIES powerCosts  = sum{t in TIMESTAMP} powerCost[tech]*power[t,tech] / 1E6;
        return sum(model.powerCost[tech]*model.power_Dvar[t,tech] for t in model.TIMESTAMP) == model.powerCosts_Pvar[tech]
    model.powerCostsCtr = Constraint(model.TECHNOLOGIES, rule=powerCostsDef_rule)    
    
    # capacityCosts definition Constraint
    def capacityCostsDef_rule(model,tech): #EQ forall tech in TECHNOLOGIES   capacityCosts  = sum{t in TIMESTAMP} capacityCost[tech]*capacity[t,tech] / 1E6;
        return model.capacityCost[tech]*model.capacity_Dvar[tech] == model.capacityCosts_Pvar[tech]
    model.capacityCostsCtr = Constraint(model.TECHNOLOGIES, rule=capacityCostsDef_rule)    

    # importCosts definition Constraint
    def importCostsDef_rule(model,res):  # ;
        return sum((model.importCost[t,res] * model.importation_Dvar[t, res]) for t in model.TIMESTAMP) == model.importCosts_Pvar[res]
    model.importCostsCtr = Constraint(model.RESOURCES, rule=importCostsDef_rule)

    # storageCosts definition Constraint
    def storageCostsDef_rule(model,s_tech): #EQ forall s_tech in STOCK_TECHNO storageCosts=storageCost[s_tech]*Cmax[s_tech] / 1E6;
        return model.storageCost[s_tech]*model.Cmax_Dvar[s_tech] == model.storageCosts_Pvar[s_tech]
    model.storageCostsCtr = Constraint(model.STOCK_TECHNO, rule=storageCostsDef_rule) 
    
    #Storage max capacity constraint
    def storageCapacity_rule(model,s_tech): #INEQ forall s_tech 
    	return model.Cmax_Dvar[s_tech]  <= model.c_max[s_tech]
    model.storageCapacityCtr = Constraint(model.STOCK_TECHNO, rule=storageCapacity_rule)   
    
    #Storage max power constraint
    def storagePower_rule(model,s_tech): #INEQ forall s_tech 
    	return model.Pmax_Dvar[s_tech]  <= model.p_max[s_tech]
    model.storagePowerCtr = Constraint(model.STOCK_TECHNO, rule=storagePower_rule) 
    
    #contraintes de stock puissance
    def StoragePowerUB_rule(model,t,res,s_tech):  # INEQ forall t
        if res == model.resource[s_tech]:
            return model.storageIn_Pvar[t,res,s_tech] - model.Pmax_Dvar[s_tech] <= 0
        else :
            return model.storageIn_Pvar[t,res,s_tech] == 0
    model.StoragePowerUBCtr = Constraint(model.TIMESTAMP,model.RESOURCES,model.STOCK_TECHNO, rule=StoragePowerUB_rule)

    def StoragePowerLB_rule(model,t,res,s_tech,):  # INEQ forall t
        if res == model.resource[s_tech]:
            return  model.storageOut_Pvar[t,res,s_tech] - model.Pmax_Dvar[s_tech] <= 0
        else :
            return model.storageOut_Pvar[t,res,s_tech] == 0
    model.StoragePowerLBCtr = Constraint(model.TIMESTAMP,model.RESOURCES,model.STOCK_TECHNO, rule=StoragePowerLB_rule)
    
    # contrainte de consommation du stockage (autre que l'énergie stockée)
    def StorageConsumption_rule(model,t,res,s_tech):  # EQ forall t
        temp=model.resource[s_tech]
        if res == temp :
            return model.storageConsumption_Pvar[t,res,s_tech] == 0
        else :
            return model.storageConsumption_Pvar[t,res,s_tech] == +model.storageFactorIn[res,s_tech]*model.storageIn_Pvar[t,temp,s_tech]+model.storageFactorOut[res,s_tech]*model.storageOut_Pvar[t,temp,s_tech]
    model.StorageConsumptionCtr = Constraint(model.TIMESTAMP,model.RESOURCES,model.STOCK_TECHNO, rule=StorageConsumption_rule)
    
    #contraintes de stock capacité
    def StockLevel_rule(model,t,s_tech):  # EQ forall t
        res=model.resource[s_tech]
        if t>1 :
            return model.stockLevel_Pvar[t,s_tech] == model.stockLevel_Pvar[t-1,s_tech] + model.storageIn_Pvar[t,res,s_tech]*model.storageFactorIn[res,s_tech] - model.storageOut_Pvar[t,res,s_tech]*model.storageFactorOut[res,s_tech]
        else :
            return model.stockLevel_Pvar[t,s_tech] == model.storageIn_Pvar[t,res,s_tech]*model.storageFactorIn[res,s_tech] - model.storageOut_Pvar[t,res,s_tech]*model.storageFactorOut[res,s_tech]
    model.StockLevelCtr = Constraint(model.TIMESTAMP,model.STOCK_TECHNO, rule=StockLevel_rule)
    
    def StockCapacity_rule(model,t,s_tech,):  # INEQ forall t
        return model.stockLevel_Pvar[t,s_tech] <= model.Cmax_Dvar[s_tech]
    model.StockCapacityCtr = Constraint(model.TIMESTAMP,model.STOCK_TECHNO, rule=StockCapacity_rule)
    
    #Capacity constraint
    def Capacity_rule(model,t,tech): #INEQ forall t, tech 
    	return model.capacity_Dvar[tech] * model.availabilityFactor[t,tech] >= model.power_Dvar[t,tech]
    model.CapacityCtr = Constraint(model.TIMESTAMP,model.TECHNOLOGIES, rule=Capacity_rule)    
    
    # Ressource production constraint
    def Production_rule(model, t, res):  # EQ forall t, res
        return sum(model.power_Dvar[t, tech] * model.conversionFactor[res, tech] for tech in model.TECHNOLOGIES) + model.importation_Dvar[t, res] + sum(model.storageOut_Pvar[t,res,s_tech]-model.storageIn_Pvar[t,res,s_tech]-model.storageConsumption_Pvar[t,res,s_tech] for s_tech in STOCK_TECHNO) == model.energy_Pvar[t, res]
    model.ProductionCtr = Constraint(model.TIMESTAMP, model.RESOURCES, rule=Production_rule)
    
    #contrainte d'equilibre offre demande 
    def energyCtr_rule(model,t,res): #INEQ forall t
    	return model.energy_Pvar[t,res]  >= model.areaConsumption[t,res]
    model.energyCtr = Constraint(model.TIMESTAMP,model.RESOURCES,rule=energyCtr_rule)
    
    if "maxCapacity" in TechParameters:
        def maxCapacity_rule(model,tech) : #INEQ forall t, tech 
            return model.maxCapacity[tech] >= model.capacity_Dvar[tech] 
        model.maxCapacityCtr = Constraint(model.TECHNOLOGIES, rule=maxCapacity_rule)
    
    if "minCapacity" in TechParameters:
        def minCapacity_rule(model,tech) : #INEQ forall t, tech 
            return model.minCapacity[tech] <= model.capacity_Dvar[tech] 
        model.minCapacityCtr = Constraint(model.TECHNOLOGIES, rule=minCapacity_rule)
    
    if "EnergyNbhourCap" in TechParameters:
        def storage_rule(model,tech) : #INEQ forall t, tech 
            if model.EnergyNbhourCap[tech]>0 :
                return model.EnergyNbhourCap[tech]*model.capacity_Dvar[tech] >= sum(model.power_Dvar[t,tech] for t in model.TIMESTAMP)
            else:
                return Constraint.Skip
        model.storageCtr = Constraint(model.TECHNOLOGIES, rule=storage_rule)

    if "RampConstraintPlus" in TechParameters:
        def rampCtrPlus_rule(model,t,tech): #INEQ forall t<
            if model.RampConstraintPlus[tech]>0 :
                return model.power_Dvar[t+1,tech]  - model.power_Dvar[t,tech] <= model.capacity_Dvar[tech]*model.RampConstraintPlus[tech] ;
            else:
                return Constraint.Skip
        model.rampCtrPlus = Constraint(model.TIMESTAMP_MinusOne,model.TECHNOLOGIES,rule=rampCtrPlus_rule)
        
    if "RampConstraintMoins" in TechParameters:
        def rampCtrMoins_rule(model,t,tech): #INEQ forall t<
            if model.RampConstraintMoins[tech]>0 :
                return model.power_Dvar[t+1,tech]  - model.power_Dvar[t,tech] >= - model.capacity_Dvar[tech]*model.RampConstraintMoins[tech] ;
            else:
                return Constraint.Skip
        model.rampCtrMoins = Constraint(model.TIMESTAMP_MinusOne,model.TECHNOLOGIES,rule=rampCtrMoins_rule)
        
    if "RampConstraintPlus2" in TechParameters:
        def rampCtrPlus2_rule(model,t,tech): #INEQ forall t<
            if model.RampConstraintPlus2[tech]>0 :
                var=(model.power_Dvar[t+2,tech]+model.power_Dvar[t+3,tech])/2 -  (model.power_Dvar[t+1,tech]+model.power_Dvar[t,tech])/2;
                return var <= model.capacity_Dvar[tech]*model.RampConstraintPlus[tech] ;
            else:
                return Constraint.Skip
        model.rampCtrPlus2 = Constraint(model.TIMESTAMP_MinusThree,model.TECHNOLOGIES,rule=rampCtrPlus2_rule)
        
    if "RampConstraintMoins2" in TechParameters:
        def rampCtrMoins2_rule(model,t,tech): #INEQ forall t<
            if model.RampConstraintMoins2[tech]>0 :
                var=(model.power_Dvar[t+2,tech]+model.power_Dvar[t+3,tech])/2 -  (model.power_Dvar[t+1,tech]+model.power_Dvar[t,tech])/2;
                return var >= - model.capacity_Dvar[tech]*model.RampConstraintMoins2[tech] ;
            else:
                return Constraint.Skip
        model.rampCtrMoins2 = Constraint(model.TIMESTAMP_MinusThree,model.TECHNOLOGIES,rule=rampCtrMoins2_rule)
    
    return model ;
