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
from functions.f_optimization import *
from functions.f_constraints import *


def GetElectricSystemModel_GestionSingleNode(areaConsumption,availabilityFactor,TechParameters,isAbstract=False):
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

    TECHNOLOGIES=   set(TechParameters.index.get_level_values('TECHNOLOGIES').unique())
    Date = set(areaConsumption.index.get_level_values('Date').unique())
    Date_list=areaConsumption.index.get_level_values('Date').unique()

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
    ## manière générique d'écrire pour toutes les colomnes COL de TechParameters quelque chose comme
    # model.COLNAME =          Param(model.TECHNOLOGIES, domain=NonNegativeReals,default=0,
    #                                  initialize=TechParameters.set_index([ "TECHNOLOGIES"]).COLNAME.squeeze().to_dict())

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

    #Capacity constraint
    def Capacity_rule(model,t,tech): #INEQ forall t, tech
    	return    model.capacity[tech] * model.availabilityFactor[t,tech] >= model.energy[t,tech]
    model.CapacityCtr = Constraint(model.Date,model.TECHNOLOGIES, rule=Capacity_rule)

    #contrainte de stock annuel


    #contrainte d'equilibre offre demande
    def energyCtr_rule(model,t): #INEQ forall t
    	return sum(model.energy[t,tech] for tech in model.TECHNOLOGIES ) == model.areaConsumption[t]
    model.energyCtr = Constraint(model.Date,rule=energyCtr_rule)

    model=set_EnergyNbHourCap_single_area(model, TechParameters)

    model=set_RampConstraints_single_area(model, TechParameters)
    return model ;

def GetElectricSystemModel_GestionSingleNode_withStorage(areaConsumption,availabilityFactor,
                                                          TechParameters,StorageParameters,solverpath=-1,isAbstract=False,
                                                          solver='mosek'):
    """
    This function creates the pyomo model and initlize the Parameters and (pyomo) Set values
    :param areaConsumption: panda table with consumption
    :param availabilityFactor: panda table
    :param isAbstract: boolean true is the model should be abstract. ConcreteModel otherwise
    :return: pyomo model
    """
    import pandas as pd
    import numpy as np
    
    #isAbstract=False
    availabilityFactor.isna().sum()

    ### Cleaning
    availabilityFactor=availabilityFactor.fillna(method='pad');
    areaConsumption=areaConsumption.fillna(method='pad');

    ### obtaining dimensions values

    TECHNOLOGIES=   set(TechParameters.index.get_level_values('TECHNOLOGIES').unique())
    STOCK_TECHNO= set(StorageParameters.index.get_level_values('STOCK_TECHNO').unique())
    Date = set(areaConsumption.index.get_level_values('Date').unique())
    Date_list= areaConsumption.index.get_level_values('Date').unique()

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
    ## manière générique d'écrire pour toutes les colomnes COL de TechParameters quelque chose comme
    # model.COLNAME =          Param(model.TECHNOLOGIES, domain=NonNegativeReals,default=0,
    #                                  initialize=TechParameters.set_index([ "TECHNOLOGIES"]).COLNAME.squeeze().to_dict())

    ################
    # Variables    #
    ################

    model.energy=Var(model.Date, model.TECHNOLOGIES, domain=NonNegativeReals) ### Energy produced by a production mean at time t
    model.energyCosts=Var(model.TECHNOLOGIES)  ### Cost of energy for a production mean, explicitely defined by definition energyCostsDef
    model.storageIn=Var(model.Date,model.STOCK_TECHNO,domain=NonNegativeReals) ### Energy stored in a storage mean at time t 
    model.storageOut=Var(model.Date,model.STOCK_TECHNO,domain=NonNegativeReals) ### Energy taken out of a storage mean at time t 
    model.stockLevel=Var(model.Date,model.STOCK_TECHNO,domain=NonNegativeReals) ### level of the energy stock in a storage mean at time t
    model.dual = Suffix(direction=Suffix.IMPORT)
    model.rc = Suffix(direction=Suffix.IMPORT)
    #model.slack = Suffix(direction=Suffix.IMPORT)

    ########################
    # Objective Function   #
    ########################

    def ObjectiveFunction_rule(model): #OBJ
    	return sum(model.energyCosts[tech] for tech in model.TECHNOLOGIES) + sum(model.c_max[s_tech]*model.storageCost[s_tech] 
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

    #Capacity constraint
    def Capacity_rule(model,t,tech): #INEQ forall t, tech
    	return    model.capacity[tech] * model.availabilityFactor[t,tech] >= model.energy[t,tech]
    model.CapacityCtr = Constraint(model.Date,model.TECHNOLOGIES, rule=Capacity_rule)

    model=set_storage_operation_constraints_single_area(model)

    def StockCapacity_rule(model,t,s_tech,):  # INEQ forall t
        return model.stockLevel[t,s_tech] <= model.c_max[s_tech]
    model.StockCapacityCtr = Constraint(model.Date,model.STOCK_TECHNO, rule=StockCapacity_rule)
    
    #contrainte d'equilibre offre demande
    def energyCtr_rule(model,t): #INEQ forall t
    	return sum(model.energy[t,tech] for tech in model.TECHNOLOGIES)+sum(model.storageOut[t,s_tech]-model.storageIn[t,s_tech] for s_tech in model.STOCK_TECHNO) == model.areaConsumption[t]
    model.energyCtr = Constraint(model.Date,rule=energyCtr_rule)


    model=set_EnergyNbHourCap_single_area(model, TechParameters)

    model=set_RampConstraints_single_area(model, TechParameters)
    return model ;

def GetElectricSystemModel_GestionMultiNode(areaConsumption,availabilityFactor,TechParameters,ExchangeParameters,isAbstract=False,LineEfficiency=1):
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
    TECHNOLOGIES= set(TechParameters.index.get_level_values('TECHNOLOGIES').unique())
    Date = set(areaConsumption.index.get_level_values('Date').unique())
    Date_list = areaConsumption.index.get_level_values('Date').unique()
    AREAS= set(areaConsumption.index.get_level_values('AREAS').unique())


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
    ## manière générique d'écrire pour toutes les colomnes COL de TechParameters quelque chose comme
    # model.COLNAME =          Param(model.TECHNOLOGIES, domain=NonNegativeReals,default=0,
    #                                  initialize=TechParameters.set_index([ "TECHNOLOGIES"]).COLNAME.squeeze().to_dict())



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

    #### 1 - Basics
    ########


    # energyCost/totalCosts definition Constraints
    # AREAS x TECHNOLOGIES
    def energyCostsDef_rule(model,area,tech): #EQ forall tech in TECHNOLOGIES   energyCosts  = sum{t in Date} energyCost[tech]*energy[t,tech] / 1E6;
        temp=model.energyCost[area,tech]#/10**6 ;
        return sum(temp*model.energy[area,t,tech] for t in model.Date) == model.energyCosts[area,tech];
    model.energyCostsDef = Constraint(model.AREAS,model.TECHNOLOGIES, rule=energyCostsDef_rule)

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


    #Capacity constraint
    #AREAS x Date x TECHNOLOGIES
    def CapacityCtr_rule(model,area,t,tech): #INEQ forall t, tech
    	return  model.capacity[area,tech] * model.availabilityFactor[area,t,tech] >=  model.energy[area,t,tech]
    model.CapacityCtr = Constraint(model.AREAS,model.Date,model.TECHNOLOGIES, rule=CapacityCtr_rule)


    #contrainte d'equilibre offre demande
    #AREAS x Date x TECHNOLOGIES
    def energyCtr_rule(model,area,t): #INEQ forall t
    	return sum(model.energy[area,t,tech] for tech in model.TECHNOLOGIES ) + sum(model.exchange[b,area,t]*LineEfficiency for b in model.AREAS ) == model.areaConsumption[area,t]
    model.energyCtr = Constraint(model.AREAS,model.Date,rule=energyCtr_rule)

   # def energyCtr_rule(model,t): #INEQ forall t
   # 	return sum(model.energy[t,tech] for tech in model.TECHNOLOGIES ) >= model.areaConsumption[t]
   # model.energyCtr = Constraint(model.Date,rule=energyCtr_rule)

    #### 2 - Optional
    ########

    #contrainte de stock annuel
    #AREAS x TECHNOLOGIES
    model=set_EnergyNbHourCap_multiple_areas(model, TechParameters)

    model=set_RampConstraints_multiple_areas(model, TechParameters)

    return model ;


def GetElectricSystemModel_GestionMultiNode_withStorage(areaConsumption,availabilityFactor,TechParameters,ExchangeParameters,
                                                         StorageParameters,LineEfficiency=1,isAbstract=False,solver='mosek'):
    """
    This function takes storage caracteristics, system caracteristics and optimise operation Set values
    :param areaConsumption: panda table with consumption
    :param availabilityFactor: panda table
    :param isAbstract: boolean true is the model should be abstract. ConcreteModel otherwise
    :param StorageParameters is a panda with p_max (maximal power), c_max (energy capacity in the storage : maximal energy)
    :return: a dictionary with model : pyomo model without storage, storage : storage infos
    """
    
    import pandas as pd
    import numpy as np

    #isAbstract=False
    availabilityFactor.isna().sum()

    ### Cleaning
    availabilityFactor=availabilityFactor.fillna(method='pad');
    areaConsumption=areaConsumption.fillna(method='pad');

    ### obtaining dimensions values
    TECHNOLOGIES= set(TechParameters.index.get_level_values('TECHNOLOGIES').unique())
    STOCK_TECHNO= set(StorageParameters.index.get_level_values('STOCK_TECHNO').unique())
    Date = set(areaConsumption.index.get_level_values('Date').unique())
    Date_list = areaConsumption.index.get_level_values('Date').unique()
    AREAS= set(areaConsumption.index.get_level_values('AREAS').unique())

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
    ## manière générique d'écrire pour toutes les colomnes COL de TechParameters quelque chose comme
    # model.COLNAME =          Param(model.TECHNOLOGIES, domain=NonNegativeReals,default=0,
    #                                  initialize=TechParameters.set_index([ "TECHNOLOGIES"]).COLNAME.squeeze().to_dict())



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

    #### 1 - Basics
    ########


    # energyCost/totalCosts definition Constraints
    # AREAS x TECHNOLOGIES
    def energyCostsDef_rule(model,area,tech): #EQ forall tech in TECHNOLOGIES   energyCosts  = sum{t in Date} energyCost[tech]*energy[t,tech] / 1E6;
        temp=model.energyCost[area,tech]#/10**6 ;
        return sum(temp*model.energy[area,t,tech] for t in model.Date) == model.energyCosts[area,tech];
    model.energyCostsDef = Constraint(model.AREAS,model.TECHNOLOGIES, rule=energyCostsDef_rule)
    
    def storageCostsDef_rule(model,area,s_tech): #EQ forall s_tech in STOCK_TECHNO   storageCosts  = storageCost[s_tech]*c_max[s_tech] / 1E6;
        return model.storageCost[area,s_tech]*model.c_max[area,s_tech] == model.storageCosts[area,s_tech]#/10**6 ;;
    model.storageCostsDef = Constraint(model.AREAS,model.STOCK_TECHNO, rule=storageCostsDef_rule)

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


    #Capacity constraint
    #AREAS x Date x TECHNOLOGIES
    def CapacityCtr_rule(model,area,t,tech): #INEQ forall t, tech
    	return  model.capacity[area,tech] * model.availabilityFactor[area,t,tech] >=  model.energy[area,t,tech]
    model.CapacityCtr = Constraint(model.AREAS,model.Date,model.TECHNOLOGIES, rule=CapacityCtr_rule)

    
    #contraintes de stock puissance
    #AREAS x Date x STOCK_TECHNO
    def StoragePowerUB_rule(model,area,t,s_tech):  # INEQ forall t
        return model.storageIn[area,t,s_tech] - model.p_max[area,s_tech] <= 0
    model.StoragePowerUBCtr = Constraint(model.AREAS,model.Date,model.STOCK_TECHNO, rule=StoragePowerUB_rule)

    def StoragePowerLB_rule(model,area,t,s_tech,):  # INEQ forall t
        return  model.storageOut[area,t,s_tech] - model.p_max[area,s_tech] <= 0
    model.StoragePowerLBCtr = Constraint(model.AREAS,model.Date,model.STOCK_TECHNO, rule=StoragePowerLB_rule)

    #contraintes de stock capacité
    #AREAS x Date x STOCK_TECHNO
    def StockLevel_rule(model,area,t,s_tech):  # EQ forall t
        if t!=Date_list[0] :
            t_moins_1 = str(pd.to_datetime(t) - timedelta(hours=1))
            return model.stockLevel[area,t,s_tech] == model.stockLevel[area,t_moins_1,s_tech]*(1-model.dissipation[area,s_tech]) + model.storageIn[area,t,s_tech]*model.efficiency_in[area,s_tech]-model.storageOut[area,t,s_tech]/model.efficiency_out[area,s_tech]
        else :
            return model.stockLevel[area,t,s_tech] == 0
    model.StockLevelCtr = Constraint(model.AREAS,model.Date,model.STOCK_TECHNO, rule=StockLevel_rule)
    
    def StockCapacity_rule(model,area,t,s_tech,):  # INEQ forall t
        return model.stockLevel[area,t,s_tech] <= model.c_max[area,s_tech]
    model.StockCapacityCtr = Constraint(model.AREAS,model.Date,model.STOCK_TECHNO, rule=StockCapacity_rule)
    

    #contrainte d'equilibre offre demande
    #AREAS x Date x TECHNOLOGIES
    def energyCtr_rule(model,area,t): #INEQ forall t
    	return sum(model.energy[area,t,tech] for tech in model.TECHNOLOGIES ) +sum(model.storageOut[area,t,s_tech]-model.storageIn[area,t,s_tech]  for s_tech in model.STOCK_TECHNO)+ sum(model.exchange[b,area,t]*LineEfficiency for b in model.AREAS) == model.areaConsumption[area,t]
    model.energyCtr = Constraint(model.AREAS,model.Date,rule=energyCtr_rule)

   # def energyCtr_rule(model,t): #INEQ forall t
   # 	return sum(model.energy[t,tech] for tech in model.TECHNOLOGIES ) >= model.areaConsumption[t]
   # model.energyCtr = Constraint(model.Date,rule=energyCtr_rule)

    #### 2 - Optional
    ########

    #contrainte de stock annuel
    #AREAS x TECHNOLOGIES
    model=set_EnergyNbHourCap_multiple_areas(model, TechParameters)

    model = set_RampConstraints_multiple_areas(model, TechParameters)

    return model ;

