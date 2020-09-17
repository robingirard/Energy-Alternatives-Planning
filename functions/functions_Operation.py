#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 19:07:50 2020

@author: robin.girard
"""


from __future__ import division
from pyomo.environ import *
from pyomo.core import *
from pyomo.opt import SolverFactory
from dynprogstorage.Wrapper_dynprogstorage import Pycplfunction, Pycplfunctionvec
from dynprogstorage.wrappers import *
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def GetElectricSystemModel_GestionSingleNode(areaConsumption,availabilityFactor,TechParameters,isAbstract=False):
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
    TECHNOLOGIES= set(TechParameters.TECHNOLOGIES.unique())
    TIMESTAMP=set(areaConsumption.TIMESTAMP.squeeze().unique())
    TIMESTAMP_list=areaConsumption.TIMESTAMP.squeeze().unique()

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
    model.TECHNOLOGIES = Set(initialize=TECHNOLOGIES)
    model.TIMESTAMP = Set(initialize=TIMESTAMP)
    model.TIMESTAMP_TECHNOLOGIES =  model.TIMESTAMP *model.TECHNOLOGIES

    #Subset of Simple only required if ramp constraint
    model.TIMESTAMP_MinusOne = Set(initialize=TIMESTAMP_list[: len(TIMESTAMP) - 1])
    model.TIMESTAMP_MinusThree = Set(initialize=TIMESTAMP_list[: len(TIMESTAMP) - 3])


    ###############
    # Parameters ##
    ###############

    model.areaConsumption =     Param(model.TIMESTAMP, mutable=True,
                                      initialize=areaConsumption.set_index([ "TIMESTAMP"]).areaConsumption.squeeze().to_dict(), domain=Any)
    model.availabilityFactor =  Param( model.TIMESTAMP_TECHNOLOGIES, domain=PercentFraction,default=1,
                                      initialize=availabilityFactor.set_index(["TIMESTAMP", "TECHNOLOGIES"]).squeeze().to_dict())

    #with test of existing columns on TechParameters
    for COLNAME in TechParameters:
        if COLNAME not in ["TECHNOLOGIES","AREAS"]: ### each column in TechParameters will be a parameter
            exec("model."+COLNAME+" =          Param(model.TECHNOLOGIES, domain=NonNegativeReals,default=0,"+
                                      "initialize=TechParameters.set_index([ \"TECHNOLOGIES\"])."+COLNAME+".squeeze().to_dict())")
    ## manière générique d'écrire pour toutes les colomnes COL de TechParameters quelque chose comme
    # model.COLNAME =          Param(model.TECHNOLOGIES, domain=NonNegativeReals,default=0,
    #                                  initialize=TechParameters.set_index([ "TECHNOLOGIES"]).COLNAME.squeeze().to_dict())

    ################
    # Variables    #
    ################

    model.energy=Var(model.TIMESTAMP, model.TECHNOLOGIES, domain=NonNegativeReals) ### Energy produced by a production mean at time t
    model.energyCosts=Var(model.TECHNOLOGIES)  ### Cost of energy for a production mean, explicitely defined by definition energyCostsDef
    model.dual = Suffix(direction=Suffix.IMPORT)
    model.rc = Suffix(direction=Suffix.IMPORT)
    model.slack = Suffix(direction=Suffix.IMPORT)

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
    def energyCostsDef_rule(model,tech): #EQ forall tech in TECHNOLOGIES   energyCosts  = sum{t in TIMESTAMP} energyCost[tech]*energy[t,tech] / 1E6;
        temp=model.energyCost[tech]/10**6 ;
        return sum(temp*model.energy[t,tech] for t in model.TIMESTAMP) == model.energyCosts[tech]
    model.energyCostsCtr = Constraint(model.TECHNOLOGIES, rule=energyCostsDef_rule)

    #Capacity constraint
    def Capacity_rule(model,t,tech): #INEQ forall t, tech
    	return model.energy[t,tech] <=  model.capacity[tech] * model.availabilityFactor[t,tech]
    model.CapacityCtr = Constraint(model.TIMESTAMP,model.TECHNOLOGIES, rule=Capacity_rule)

    #contrainte de stock annuel


    #contrainte d'equilibre offre demande
    def energyCtr_rule(model,t): #INEQ forall t
    	return sum(model.energy[t,tech] for tech in model.TECHNOLOGIES ) >= model.areaConsumption[t]
    model.energyCtr = Constraint(model.TIMESTAMP,rule=energyCtr_rule)


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

def GetElectricSystemModel_GestionSingleNode_with1Storage(areaConsumption,availabilityFactor,
                                                          TechParameters,p_max,c_max,isAbstract=False,
                                                          solver='mosek',n=10,efficiency_in=1,
                                                          efficiency_out=1,tol=exp(-4)):
    """
    This function takes storage caracteristics, system caracteristics and optimise operation Set values
    :param areaConsumption: panda table with consumption
    :param availabilityFactor: panda table
    :param isAbstract: boolean true is the model should be abstract. ConcreteModel otherwise
    :p_max is maximal power
    :c_max is energy capacity in the storage (maximal energy)
    :efficiency_in is input efficiency of storage
    :efficiency_out is output efficiency of storage
    :return: a dictionary with model : pyomo model without storage, storage : storage infos
    """
    import pandas as pd
    import numpy as np
    from dynprogstorage.wrappers import GenCostFunctionFromMarketPrices
    from dynprogstorage.wrappers import GenCostFunctionFromMarketPrices_dict
    #isAbstract=False

    model = GetElectricSystemModel_GestionSingleNode(areaConsumption, availabilityFactor, TechParameters,isAbstract=isAbstract)
    opt = SolverFactory(solver)
   # results = opt.solve(model)
   # Variables = getVariables_panda(model) #### obtain optimized variables in panda form
   # Constraints = getConstraintsDual_panda(model)  #### obtain lagrange multipliers in panda form

    ##### Loop
    PrixTotal = {}
    Consommation = {}
    LMultipliers = {}
    DeltaPrix = {}
    Deltazz = {}
    CostFunction = {}
    TotalCols = {}
    zz = {}
    # p_max_out=100.; p_max_in=100.; c_max=10.;

    areaConsumption["NewConsumption"] = areaConsumption["areaConsumption"]
    nbTime = len(areaConsumption["areaConsumption"])
    cpt = 0
    for i in model.areaConsumption:  model.areaConsumption[i] = areaConsumption.NewConsumption[i - 1]
    DeltaPrix_=tol+1
    while ((DeltaPrix_ >  tol)&(n<cpt)) :
        print(cpt)
        if (cpt == 0):
            zz[cpt] = [0] * nbTime
        else:
            zz[cpt] = areaConsumption["Storage"]
        results = opt.solve(model)
        Constraints = getConstraintsDual_panda(model)
        TotalCols[cpt] = getVariables_panda(model)['energyCosts'].sum()[1]
        Prix = Constraints["energyCtr"].assign(Prix=lambda x: x.energyCtr * 10 ** 6).Prix.to_numpy()
        valueAtZero = Prix * (TotalCols[cpt] / sum(Prix * Prix) - zz[cpt])
        tmpCost = GenCostFunctionFromMarketPrices_dict(Prix, r_in=efficiency_in, r_out=efficiency_out, valueAtZero=valueAtZero)
        if (cpt == 0):
            CostFunction[cpt] = GenCostFunctionFromMarketPrices(Prix, r_in=efficiency_in, r_out=efficiency_out, valueAtZero=valueAtZero)
        else:
            tmpCost = GenCostFunctionFromMarketPrices_dict(Prix, r_in=efficiency_in, r_out=efficiency_out, valueAtZero=valueAtZero)
            tmpCost2 = CostFunction[cpt - 1]
            if efficiency_in*efficiency_out==1:
                tmpCost2.Maxf_1Breaks_withO(tmpCost['S1'], tmpCost['B1'], tmpCost[
                    'f0'])
            else:
                tmpCost2.Maxf_2Breaks_withO(tmpCost['S1'], tmpCost['S2'], tmpCost['B1'], tmpCost['B2'], tmpCost[
                'f0'])  ### etape clé, il faut bien comprendre ici l'utilisation du max de deux fonction de coût
            CostFunction[cpt] = tmpCost2
        LMultipliers[cpt] = Prix
        if cpt > 0:
            DeltaPrix[cpt] = sum(abs(LMultipliers[cpt] - LMultipliers[cpt - 1]))/sum(abs(LMultipliers[cpt]))
            Deltazz[cpt] = sum(abs(zz[cpt] - zz[cpt - 1]))/sum(abs(zz[cpt]))
            DeltaPrix_= DeltaPrix[cpt]

        areaConsumption["Storage"] = CostFunction[cpt].OptimMargInt([-p_max] * nbTime, [p_max] * nbTime, [0] * nbTime,
                                                                    [c_max] * nbTime)
        areaConsumption["NewConsumption"] = areaConsumption["areaConsumption"] + areaConsumption["Storage"]
        for i in model.areaConsumption:  model.areaConsumption[i] = areaConsumption.NewConsumption[i - 1]
        cpt=cpt+1
    stats = {"DeltaPrix" : DeltaPrix,"Deltazz" : Deltazz}
    return {"areaConsumption" : areaConsumption, "model" : model, "stats" : stats};

def GetElectricSystemModel_GestionMultiNode(areaConsumption,availabilityFactor,TechParameters,ExchangeParameters,isAbstract=False):
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
    TECHNOLOGIES= set(TechParameters.TECHNOLOGIES.unique())
    TIMESTAMP=set(areaConsumption.TIMESTAMP.squeeze())
    TIMESTAMP_list=areaConsumption.TIMESTAMP.squeeze().unique()
    AREAS= set(areaConsumption.AREAS.unique())

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
    model.AREAS= Set(initialize=AREAS,doc = "Area")
    model.TECHNOLOGIES = Set(initialize=TECHNOLOGIES)
    model.TIMESTAMP = Set(initialize=TIMESTAMP)

    #Subset of Simple only required if ramp constraint
    model.TIMESTAMP_MinusOne = Set(initialize=TIMESTAMP.remove(max(TIMESTAMP)))
    model.TIMESTAMP_MinusThree = Set(initialize=TIMESTAMP_list[: len(TIMESTAMP) - 3])

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
                                      initialize=areaConsumption.set_index(["AREAS", "TIMESTAMP",]).areaConsumption.squeeze().to_dict(), domain=Any)
    model.availabilityFactor =  Param(  model.AREAS_TIMESTAMP_TECHNOLOGIES, domain=PercentFraction,default=1,
                                      initialize=availabilityFactor.set_index(["AREAS","TIMESTAMP", "TECHNOLOGIES"]).squeeze().to_dict())

    model.maxExchangeCapacity = Param( model.AREAS_AREAS,  initialize=ExchangeParameters.set_index(["AREAS","AREAS.1"]).maxExchangeCapacity.squeeze().to_dict(), domain=NonNegativeReals,default=0)
    #with test of existing columns on TechParameters
    for COLNAME in TechParameters:
        if COLNAME not in ["TECHNOLOGIES","AREAS"]: ### each column in TechParameters will be a parameter
            exec("model."+COLNAME+" =          Param(model.AREAS_TECHNOLOGIES, domain=NonNegativeReals,default=0,"+
                                      "initialize=TechParameters.set_index([ \"AREAS\",\"TECHNOLOGIES\"])."+COLNAME+".squeeze().to_dict())")
    ## manière générique d'écrire pour toutes les colomnes COL de TechParameters quelque chose comme
    # model.COLNAME =          Param(model.TECHNOLOGIES, domain=NonNegativeReals,default=0,
    #                                  initialize=TechParameters.set_index([ "TECHNOLOGIES"]).COLNAME.squeeze().to_dict())



    ################
    # Variables    #
    ################

    model.energy=Var(model.AREAS,model.TIMESTAMP, model.TECHNOLOGIES, domain=NonNegativeReals) ### Energy produced by a production mean at time t
    model.exchange=Var(model.AREAS_AREAS,model.TIMESTAMP)
    model.energyCosts=Var(model.AREAS,model.TECHNOLOGIES)   ### Cost of energy by a production mean for area at time t (explicitely defined by constraint energyCostsCtr)

    model.dual = Suffix(direction=Suffix.IMPORT)
    model.rc = Suffix(direction=Suffix.IMPORT)
    model.slack = Suffix(direction=Suffix.IMPORT)

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
    def energyCostsDef_rule(model,area,tech): #EQ forall tech in TECHNOLOGIES   energyCosts  = sum{t in TIMESTAMP} energyCost[tech]*energy[t,tech] / 1E6;
        temp=model.energyCost[area,tech]/10**6 ;
        return sum(temp*model.energy[area,t,tech] for t in model.TIMESTAMP) == model.energyCosts[area,tech];
    model.energyCostsDef = Constraint(model.AREAS,model.TECHNOLOGIES, rule=energyCostsDef_rule)

    #Exchange capacity constraint (duplicate of variable definition)
    # AREAS x AREAS x TIMESTAMP
    def exchangeCtr_rule(model,a, b, t): #INEQ forall area.axarea.b in AREASxAREAS  t in TIMESTAMP
        return model.exchange[a,b,t]  <= model.maxExchangeCapacity[a,b] ;
    model.exchangeCtr = Constraint(model.AREAS,model.AREAS,model.TIMESTAMP, rule=exchangeCtr_rule)

    #Capacity constraint
    #AREAS x TIMESTAMP x TECHNOLOGIES
    def CapacityCtr_rule(model,area,t,tech): #INEQ forall t, tech
    	return  model.energy[area,t,tech] <= model.capacity[area,tech] * model.availabilityFactor[area,t,tech]
    model.CapacityCtr = Constraint(model.AREAS,model.TIMESTAMP,model.TECHNOLOGIES, rule=CapacityCtr_rule)


    #contrainte d'equilibre offre demande
    #AREAS x TIMESTAMP x TECHNOLOGIES
    def energyCtr_rule(model,area,t): #INEQ forall t
    	return sum(model.energy[area,t,tech] for tech in model.TECHNOLOGIES ) >= model.areaConsumption[area,t]
    model.energyCtr = Constraint(model.AREAS,model.TIMESTAMP,rule=energyCtr_rule)

    #### 2 - Optional
    ########

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




def get_SimpleSets(model):
    """
    This function finds all SimpleSets and returns a set with pyomo Sets and associated values
    :param model: pyomo model
    :return: a Set (not a pyomo set) with names
    """
    res={}
    for v in model.component_objects(Set, active=True):
        setobject = getattr(model, str(v))
        if not (isinstance(setobject,pyomo.core.base.set.SetProduct)):
            res[str(v)]=setobject.data()
    return res;

def getSetNames(model,setobject):
    """
    This function finds the sets associated to a flat set product object
    :param model: pyomo model
    :param setobject: a pyomo set object
    :return: a Set (not a pyomo set) with names
    """
    SimpleSets=get_SimpleSets(model)
    if not isinstance(setobject,pyomo.core.base.set.SetProduct):
        print("warning setobject should be a SetProduct")
    cpt=0;
    res={}
    for subset in setobject.subsets():
        for i in SimpleSets:
            if SimpleSets[i]==subset.data():
                res[cpt]=i
        cpt+=1;
    return res;

def getVariables_panda(model):
    """
    This function takes variables and return values in panda form
    :param model: pyomo model
    :return: a Set (not a pyomo set) with names of variables and associated values in panda table
    """
    import pandas as pd
    Variables = {}
    for v in model.component_objects(Var, active=True):
        # print ("Variables",v)
        varobject = getattr(model, str(v))
        VAL = pd.DataFrame.from_dict(varobject.extract_values(), orient='index', columns=[str(varobject)])
        #print(get_set_name(model,varobject))
        if isinstance(varobject.index_set(),pyomo.core.base.set.SetProduct):
            #Variables[str(v)].rename_axis(index=[str(varobject.index_set())])
            DIM = pd.DataFrame(VAL.index.tolist()).rename(columns=getSetNames(model,varobject.index_set())).reset_index(drop=True)
            VAL.reset_index(drop=True, inplace=True)
            Variables[str(v)]= pd.concat([DIM,VAL],axis=1,sort=False)
        else:
            DIM= pd.DataFrame(VAL.index.tolist()).reset_index(drop=True)
            VAL.reset_index(drop=True, inplace=True)
            Variables[str(v)]= pd.concat([DIM,VAL],axis=1,sort=False)
            Variables[str(v)]=Variables[str(v)].rename(columns={0:str(varobject.index_set())})
    return Variables;

def getConstraintsDual_panda(model):
    """
    This function takes dual values associated to Constraints and return values in panda form
    :param model: pyomo model
    :return: a Set (not a pyomo set) with names of Constraints and associated dual values in panda table
    """
    import pandas as pd
    Constraints = {}
    for v in model.component_objects(Constraint, active=True):
        # print ("Constraints",v)
        cobject = getattr(model, str(v))
        VAL = pd.DataFrame.from_dict(get_dualValues(model,cobject), orient='index', columns=[str(cobject)])
        #print(get_set_name(model,varobject))
        if isinstance(cobject.index_set(),pyomo.core.base.set.SetProduct):
            #Constraints[str(v)].rename_axis(index=[str(varobject.index_set())])
            DIM = pd.DataFrame(VAL.index.tolist()).rename(columns=getSetNames(model,cobject.index_set())).reset_index(drop=True)
            VAL.reset_index(drop=True, inplace=True)
            Constraints[str(v)]= pd.concat([DIM,VAL],axis=1,sort=False)
        else:
            DIM= pd.DataFrame(VAL.index.tolist()).reset_index(drop=True)
            VAL.reset_index(drop=True, inplace=True)
            Constraints[str(v)]= pd.concat([DIM,VAL],axis=1,sort=False)
            Constraints[str(v)]=Constraints[str(v)].rename(columns={0:str(cobject.index_set())})
    return Constraints;

def get_dualValues(model,cobject):
    """
    This function takes variables and return values in panda form
    :param model: pyomo model
    :return: a Set (not a pyomo set) with names of variables and associated values in panda table
    """
    res={};
    for index in cobject:
         res[index] = model.dual[cobject[index]]
    return res;
