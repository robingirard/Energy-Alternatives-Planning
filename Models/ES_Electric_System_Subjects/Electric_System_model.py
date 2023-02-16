from datetime import timedelta
import pandas as pd
from pyomo.core import *

# from EnergyAlternativesPlanning import *
from EnergyAlternativesPlanning.f_tools import *
from EnergyAlternativesPlanning.f_model_definition import *
from EnergyAlternativesPlanning.f_model_cost_functions import *
from EnergyAlternativesPlanning.f_model_planning_constraints import *
from EnergyAlternativesPlanning.f_model_operation_constraints import *

def GetElectricitySystemModel(Parameters):
    #####################
    # Sets & Parameters #
    #####################
    model=Create_pyomo_model_sets_parameters(Parameters)

    #############
    # Variables #
    #############
    model=set_Operation_variables(model)
    model=set_Planning_variables(model)

    ################################
    # Cost EnergyAlternativesPlanning and objective #
    ################################
    model=set_Planning_base_cost_function(model)

    #########################
    # Operation constraints #
    #########################
    model=set_Operation_Constraints_energyCapacityexchange({}, model)
    model = set_Operation_Constraints_Storage(model)
    model = set_Operation_Constraints_stockCtr(model)
    model = set_Operation_Constraints_Ramp(model)
    model = set_Operation_Constraints_flex(model)
    ########################
    # Planning constraints #
    ########################
    model=set_Planning_Constraints_maxminCapacityCtr(model)
    model=set_Planning_Constraints_storageCapacityPowerCtr(model)
    #model.= Constraint()

    ##########################
    # Additional constraints #
    ##########################
    def storage_mw_mwh_binding(model):
        Set_names = get_allSetsnames(model)
        my_set_names = list(Set_names)
        if "AREAS" in my_set_names:
            # multiple area without storage
            def Storage_Power_Capacity_binding_rule(model, area, s_tech):  # INEQ forall s_tech
                if model.strhours[area,s_tech]>0:
                    return model.Cmax[area, s_tech] == model.strhours[area,s_tech] * model.Pmax[area, s_tech]
                else:
                    return Constraint.Skip

            model.Storage_Power_Capacity_bindingCtr = Constraint(model.AREAS, model.STOCK_TECHNO,
                                                                 rule=Storage_Power_Capacity_binding_rule)
        else:
            # single area without storage
            def Storage_Power_Capacity_binding_rule(model, s_tech):  # INEQ forall s_tech
                if model.strhours[s_tech] > 0:
                    return model.Cmax[s_tech] == model.strhours[s_tech] * model.Pmax[s_tech]
                else:
                    return Constraint.Skip

            model.Storage_Power_Capacity_bindingCtr = Constraint(model.STOCK_TECHNO,
                                                                 rule=Storage_Power_Capacity_binding_rule)
        return model

    model = storage_mw_mwh_binding(model)


    return model

