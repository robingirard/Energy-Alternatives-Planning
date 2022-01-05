from datetime import timedelta
import pandas as pd
from pyomo.core import *
from functions.f_tools import *
#default={"areaConsumption":"0","availabilityFactor":"1"}; mutable ={"areaConsumption" : "True"}; domain ={"availabilityFactor" : "PercentFraction","maxExchangeCapacity" : "NonNegativeReals"}
def Create_pyomo_model_sets_parameters(Parameters,
                                 default={"areaConsumption":"0","availabilityFactor":"1"},
                                 mutable ={"areaConsumption" : "True"},
                                 domain ={"availabilityFactor" : "PercentFraction","maxExchangeCapacity" : "NonNegativeReals"}):
    # isAbstract=False
    #Parameters.availabilityFactor.isna().sum()
    #model=self
    model = ConcreteModel()

    ### Cleaning
    #TODO : workout something more generic here
    for key_name in ["availabilityFactor","areaConsumption"]:
        if key_name in Parameters: Parameters[key_name] = Parameters[key_name].fillna(method='pad');

    #gathering Set names through indexes names
    Set_vals_names= set([])
    for key_name in Parameters:
        for set_name in Parameters[key_name].index.names:
            Set_vals_names= Set_vals_names.union([set_name])

    # empty dictionary for set values
    Set_vals={}
    for set_name in Set_vals_names:
        Set_vals[set_name]=set([])

    ### obtaining set values
    for key_name in Parameters:
        for set_name in Set_vals:
            if set_name in Parameters[key_name].index.names:
                Set_vals[set_name]=Set_vals[set_name].union(set(Parameters[key_name].index.get_level_values(set_name).unique()))
                if set_name == "Date": Date_list = Parameters[key_name].index.get_level_values(set_name).unique()

    ###    Pyomo model definition


    ###############
    # Sets       ##
    ###############
    for set_name in Set_vals:
        setattr(model,set_name,Set(initialize=Set_vals[set_name], ordered=False))
    #get_allSets(model).keys()

    # product set are defined depending on existing parameters multi_index
    #TODO better codding bellow using "exec" ?
    for key_name in Parameters:
        Index_names =Parameters[key_name].index.names
        if len(Index_names)==2:
            Dim_name="_".join(Index_names)
            if Dim_name=="AREAS_AREAS.1": Dim_name="AREAS_AREAS";
            setattr(model,Dim_name, getattr(model,Index_names[0]) * getattr(model,Index_names[1]))
        if len(Index_names)==3:
            setattr(model,Index_names[0] +"_" +Index_names[1]+"_" +Index_names[2],
                    getattr(model,Index_names[0]) * getattr(model,Index_names[1])* getattr(model,Index_names[2]))
        if len(Index_names) == 4:
            setattr(model,Index_names[0] +"_" +Index_names[1]+"_" +Index_names[2]+"_" +Index_names[3],
                    getattr(model,Index_names[0]) * getattr(model,Index_names[1])* getattr(model,Index_names[2])* getattr(model,Index_names[3]))



    #TODO Make this part more generic --> defined name for AREAS-DATE, ....
    # Subset of Simple only required if ramp constraint
    model.Date_MinusOne = Set(initialize=Date_list[: len(Set_vals["Date"]) - 1], ordered=False)
    model.Date_MinusThree = Set(initialize=Date_list[: len(Set_vals["Date"]) - 3], ordered=False)

    ###############
    # Parameters ##
    ###############
    for key_name in Parameters:
        Index_names = Parameters[key_name].index.names
        for COLNAME in Parameters[key_name]:
            if COLNAME not in default: default[COLNAME] = "0"
            if COLNAME not in mutable: mutable[COLNAME] = "False"
            if COLNAME not in domain: domain[COLNAME] =  "Any"
            Dim_name="_".join(Index_names)
            if Dim_name=="AREAS_AREAS.1": Dim_name="AREAS_AREAS"; #TODO try to treat that in a cleaner way ... I had to twist function get_SimpleSets also
            exec("model." + COLNAME + " =Param(model."+Dim_name+", default="+default[COLNAME]+"," +"initialize=Parameters[key_name]." + COLNAME + ".squeeze().to_dict(),mutable="+mutable[COLNAME]+",domain="+domain[COLNAME]+")")

    #self = model
    return model

#pyomo.core.base.PyomoModel.ConcreteModel.Create_pyomo_model_sets_parameters=Create_pyomo_model_sets_parameters


def set_Planing_base_variables(model):
    """
    Defined variables :

    [All] energy | energyCosts | capacityCosts | capacity

    [if AREAS] exchange

    [if STOCK_TECHNO]   storageIn | storageOut | stockLevel | storageCosts | Cmax | Pmax

    :param model:
    :return:
    """
    #model = self
    Set_names = get_allSetsnames(model)
    if "AREAS" in Set_names:
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
    else:
        model.energy = Var(model.Date, model.TECHNOLOGIES,
                           domain=NonNegativeReals)  ### Energy produced by a production mean at time t
        model.energyCosts = Var(
            model.TECHNOLOGIES)  ### Cost of energy for a production mean, explicitely defined by definition energyCostsDef
        model.capacityCosts = Var(
            model.TECHNOLOGIES)  ### Cost of energy for a production mean, explicitely defined by definition energyCostsDef
        model.capacity = Var(model.TECHNOLOGIES,
                             domain=NonNegativeReals)  ### Energy produced by a production mean at time t

        model.dual = Suffix(direction=Suffix.IMPORT)
        model.rc = Suffix(direction=Suffix.IMPORT)
        model.slack = Suffix(direction=Suffix.IMPORT)

    Set_names = get_allSetsnames(model)
    if 'STOCK_TECHNO' in Set_names:
        ## storage case
        model = set_Planing_storage_variables(model)
    #self = model
    return model


def set_Planing_storage_variables(model):
    """
    defined variables : energy energyCosts capacityCosts capacity
    if AREAS --> exchange
    :param model:
    :return:
    """
    #model = self
    Set_names = get_allSetsnames(model)
    if "AREAS" in Set_names:
        model.storageIn = Var(model.AREAS, model.Date, model.STOCK_TECHNO,
                              domain=NonNegativeReals)  ### Energy stored by a storage mean for areas at time t
        model.storageOut = Var(model.AREAS, model.Date, model.STOCK_TECHNO,
                               domain=NonNegativeReals)  ### Energy taken out of a storage mean for areas at time t
        model.stockLevel = Var(model.AREAS, model.Date, model.STOCK_TECHNO,
                               domain=NonNegativeReals)  ### level of the energy stock in a storage mean at time t
        model.storageCosts = Var(model.AREAS,
                                 model.STOCK_TECHNO)  ### Cost of storage for a storage mean, explicitely defined by definition storageCostsDef
        model.Cmax = Var(model.AREAS, model.STOCK_TECHNO)  # Maximum capacity of a storage mean
        model.Pmax = Var(model.AREAS, model.STOCK_TECHNO)  # Maximum flow of energy in/out of a storage mean

    else:
        model.storageIn = Var(model.Date, model.STOCK_TECHNO,
                              domain=NonNegativeReals)  ### Energy stored in a storage mean at time t
        model.storageOut = Var(model.Date, model.STOCK_TECHNO,
                               domain=NonNegativeReals)  ### Energy taken out of a storage mean at time t
        model.stockLevel = Var(model.Date, model.STOCK_TECHNO,
                               domain=NonNegativeReals)  ### level of the energy stock in a storage mean at time t
        model.storageCosts = Var(
            model.STOCK_TECHNO)  ### Cost of storage for a storage mean, explicitely defined by definition storageCostsDef
        model.Cmax = Var(model.STOCK_TECHNO)  # Maximum capacity of a storage mean
        model.Pmax = Var(model.STOCK_TECHNO)  # Maximum flow of energy in/out of a storage mean

    #self = model
    return model

