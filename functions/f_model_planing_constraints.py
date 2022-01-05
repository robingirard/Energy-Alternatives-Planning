from datetime import timedelta
import pandas as pd
from pyomo.core import *
from functions.f_tools import *

def set_Planing_Constraints_maxminCapacityCtr(model):
    """
    maxCapacity>=capacity

    minCapacity<=capacity

    :param model:
    :return:
    """
    model = set_Planing_Constraints_maxCapacityCtr(model) #  model.maxCapacity[tech] >= model.capacity[tech]
    model = set_Planing_Constraints_minCapacityCtr(model) # model.minCapacity[tech] <= model.capacity[tech]
    return model;

def set_Planing_Constraints_maxCapacityCtr(model):
    """
    maxCapacity>=capacity

    :param model:
    :return:
    """
    All_parameters = get_ParametersNames(model)
    Set_names = get_allSetsnames(model)
    if "maxCapacity" in All_parameters:
        if "AREAS" in Set_names:
            def maxCapacity_rule(model, area, tech):  # INEQ forall t, tech
                if model.maxCapacity[area, tech] > 0:
                    return model.maxCapacity[area, tech] >= model.capacity[area, tech]
                else:
                    return Constraint.Skip
            model.maxCapacityCtr = Constraint(model.AREAS, model.TECHNOLOGIES, rule=maxCapacity_rule)

        else:
            def maxCapacity_rule(model, tech):  # INEQ forall t, tech
                if model.maxCapacity[tech] > 0:
                    return model.maxCapacity[tech] >= model.capacity[tech]
                else:
                    return Constraint.Skip
            model.maxCapacityCtr = Constraint(model.TECHNOLOGIES, rule=maxCapacity_rule)

    return model;

def set_Planing_Constraints_minCapacityCtr(model):
    """
    minCapacity<=capacity

    :param model:
    :return:
    """
    All_parameters = get_ParametersNames(model)
    Set_names = get_allSetsnames(model)
    if "maxCapacity" in All_parameters:
        if "AREAS" in Set_names:
            def minCapacity_rule(model, area, tech):  # INEQ forall t, tech
                if model.minCapacity[area, tech] > 0:
                    return model.minCapacity[area, tech] <= model.capacity[area, tech]
                else:
                    return Constraint.Skip

            model.minCapacityCtr = Constraint(model.AREAS, model.TECHNOLOGIES, rule=minCapacity_rule)

        else:
            def minCapacity_rule(model, tech):  # INEQ forall t, tech
                if model.minCapacity[tech] > 0:
                    return model.minCapacity[tech] <= model.capacity[tech]
                else:
                    return Constraint.Skip

            model.minCapacityCtr = Constraint(model.TECHNOLOGIES, rule=minCapacity_rule)

    return model;

def set_Planing_Constraints_storageCapacityPowerCtr(model):
    """
    Cmax <= c_max

    Pmax <= p_max

    :param model:
    :return:
    """
    Set_names = get_allSetsnames(model)
    if 'STOCK_TECHNO' in Set_names:
        model = set_Planing_Constraints_storageCapacityCtr(model)
        model = set_Planing_Constraints_storagePowerCtr(model)
    return model

def set_Planing_Constraints_storageCapacityCtr(model):
    """
    Cmax <= c_max

    :param model:
    :return:
    """
    # Storage max capacity constraint
    Set_names = get_allSetsnames(model)
    match list(Set_names):
        case [*my_set_names] if "AREAS" in my_set_names:
            # multiple area without storage
            def storageCapacity_rule(model, area, s_tech):  # INEQ forall s_tech
                return model.Cmax[area, s_tech] <= model.c_max[area, s_tech]
            model.storageCapacityCtr = Constraint(model.AREAS, model.STOCK_TECHNO, rule=storageCapacity_rule)
        case _:
            # single area without storage
            def storageCapacity_rule(model, s_tech):  # INEQ forall s_tech
                return model.Cmax[s_tech] <= model.c_max[s_tech]
            model.storageCapacityCtr = Constraint(model.STOCK_TECHNO, rule=storageCapacity_rule)

    return model

def set_Planing_Constraints_storagePowerCtr(model):
    """
    Pmax <= p_max

    :param model:
    :return:
    """
    # Storage max power constraint
    Set_names = get_allSetsnames(model)
    match list(Set_names):
        case [*my_set_names] if "AREAS" in my_set_names:
            # multiple area without storage
            def storagePower_rule(model, area, s_tech):  # INEQ forall s_tech
                return model.Pmax[area, s_tech] <= model.p_max[area, s_tech]
            model.storagePowerCtr = Constraint(model.AREAS, model.STOCK_TECHNO, rule=storagePower_rule)
        case _:
            # single area without storage
            def storagePower_rule(model, s_tech):  # INEQ forall s_tech
                return model.Pmax[s_tech] <= model.p_max[s_tech]
            model.storagePowerCtr = Constraint(model.STOCK_TECHNO, rule=storagePower_rule)

    return model

