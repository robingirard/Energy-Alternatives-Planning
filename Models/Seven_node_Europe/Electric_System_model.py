from datetime import timedelta
import pandas as pd
from pyomo.core import *

# from EnergyAlternativesPlaning import *
from EnergyAlternativesPlaning.f_tools import *
from EnergyAlternativesPlaning.f_model_definition import *
from EnergyAlternativesPlaning.f_model_cost_functions import *
from EnergyAlternativesPlaning.f_model_planing_constraints import *
from EnergyAlternativesPlaning.f_model_operation_constraints import *

def GetElectricitySystemModel(Parameters):
    #####################
    # Sets & Parameters #
    #####################
    model=Create_pyomo_model_sets_parameters(Parameters)

    #############
    # Variables #
    #############
    model=set_Operation_variables(model)
    model=set_Planing_variables(model)

    ################################
    # Cost EnergyAlternativesPlaning and objective #
    ################################
    model=set_Planing_base_cost_function(model)

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
    model=set_Planing_Constraints_maxminCapacityCtr(model)
    model=set_Planing_Constraints_storageCapacityPowerCtr(model)

    ##########################
    # Additional constraints #
    ##########################

    #STORAGE POWER_CAPACITY_BINDING
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

    #H2 PRODUCTION FOR STORAGE
    def set_h2_storage_Ctr(model):
        Set_names = get_allSetsnames(model)
        my_set_names = list(Set_names)
        if "AREAS" in my_set_names:
            if "CCG - H2" in list(model.TECHNOLOGIES) and "TAC - H2" in list(model.TECHNOLOGIES):
                model.h2_storage = Var(model.AREAS, model.Date, domain=NonNegativeReals)
                def h2_storage_rule(model,area):
                    return sum(model.h2_storage[area,t] for t in model.Date) == sum(model.energy[area,t, "CCG - H2"]/0.6 + model.energy[area,t, "TAC - H2"]/0.4 for t in model.Date)
                model.h2_storageCtr=Constraint(model.AREAS,rule=h2_storage_rule)


        else:
            if "CCG - H2" in list(model.TECHNOLOGIES) and "TAC - H2" in list(model.TECHNOLOGIES):
                model.h2_storage = Var(model.Date, domain=NonNegativeReals)

                def h2_storage_rule(model):
                    return sum(model.h2_storage[t] for t in model.Date) == sum(
                        model.energy[t, "CCG - H2"]/(0.6*0.55) + model.energy[t, "TAC - H2"]/(0.4*0.55) for t in model.Date)

                model.h2_storageCtr = Constraint(rule=h2_storage_rule)

        return model

    def set_Operation_Constraints_consum_flex_Ctr(model): #existing constraints overwritten
        Set_names = get_allSetsnames(model)
        my_set_names = list(Set_names)
        model.del_component("consum_flex_Ctr")
        model.del_component("consum_flex_Ctr_index")
        if "CCG - H2" in list(model.TECHNOLOGIES) and "TAC - H2" in list(model.TECHNOLOGIES):
            if "AREAS" in my_set_names:
                def consum_flex_rule(model, area, t, conso_type):
                    if conso_type=="H2":
                        return model.flex_consumption[area, t, conso_type] == model.to_flex_consumption[area, t, conso_type]*(
                            1 - model.flex[area, t, conso_type])+ model.h2_storage[area,t]
                    else:
                        return model.flex_consumption[area, t, conso_type] == model.to_flex_consumption[area, t, conso_type] * (
                            1 - model.flex[area, t, conso_type])

                model.consum_flex_Ctr = Constraint(model.AREAS, model.Date, model.FLEX_CONSUM, rule=consum_flex_rule)
            else:
                def consum_flex_rule(model, t, conso_type):
                    if conso_type == "H2":
                        return model.flex_consumption[t, conso_type] == model.to_flex_consumption[
                            t, conso_type] * (
                                       1 - model.flex[t, conso_type]) + model.h2_storage[t]
                    else:
                        return model.flex_consumption[t, conso_type] == model.to_flex_consumption[t, conso_type] * (
                                1 - model.flex[t, conso_type])

                model.consum_flex_Ctr = Constraint(model.Date, model.FLEX_CONSUM, rule=consum_flex_rule)
        else:
            if 'AREAS' in Set_names:
                def consum_flex_rule(model, area, t, conso_type):
                    return model.flex_consumption[area, t, conso_type] == model.to_flex_consumption[
                        area, t, conso_type] * (
                                   1 - model.flex[area, t, conso_type])

                model.consum_flex_Ctr = Constraint(model.AREAS, model.Date, model.FLEX_CONSUM, rule=consum_flex_rule)
            else:
                def consum_flex_rule(model, t, conso_type):
                    return model.flex_consumption[t, conso_type] == model.to_flex_consumption[t, conso_type] * (
                            1 - model.flex[t, conso_type])

                model.consum_flex_Ctr = Constraint(model.Date, model.FLEX_CONSUM, rule=consum_flex_rule)
        return model

    def set_Operation_Constraints_consum_eq_year_Ctr(model):
        """
        sum(flex_consumption) = sum(to_flex_consumption)

        :param model:
        :return:
        """
        Set_names = get_allSetsnames(model)
        my_set_names = list(Set_names)
        model.del_component("consum_eq_year_Ctr")
        model.del_component("consum_eq_year_Ctr_index")
        if "CCG - H2" in list(model.TECHNOLOGIES) and "TAC - H2" in list(model.TECHNOLOGIES):
            if 'AREAS' in my_set_names:
                def consum_eq_year(model, area, conso_type):
                    if model.flex_type[area, conso_type] == 'year':
                        if conso_type == "H2":
                            return sum(model.flex_consumption[area, t, conso_type] for t in model.Date) == sum(
                                model.to_flex_consumption[area, t, conso_type] + model.h2_storage[area,t] for t in model.Date)
                        else:
                            return sum(model.flex_consumption[area, t, conso_type] for t in model.Date) == sum(
                                model.to_flex_consumption[area, t, conso_type] for t in model.Date)
                    else:
                        return Constraint.Skip

                model.consum_eq_year_Ctr = Constraint(model.AREAS, model.FLEX_CONSUM, rule=consum_eq_year)

            else:
                def consum_eq_year(model, conso_type):
                    if model.flex_type[conso_type] == 'year':
                        if conso_type == "H2":
                            return sum(model.flex_consumption[t, conso_type] for t in model.Date) == sum(
                                model.to_flex_consumption[t, conso_type] + model.h2_storage[t] for t in model.Date)
                        else:
                            return sum(model.flex_consumption[t, conso_type] for t in model.Date) == sum(
                                model.to_flex_consumption[t, conso_type] for t in model.Date)
                    else:
                        return Constraint.Skip

                model.consum_eq_year_Ctr = Constraint(model.FLEX_CONSUM, rule=consum_eq_year)
        else:
            if 'AREAS' in my_set_names:
                def consum_eq_year(model, area, conso_type):
                    if model.flex_type[area, conso_type] == 'year':
                        return sum(model.flex_consumption[area, t, conso_type] for t in model.Date) == sum(
                            model.to_flex_consumption[area, t, conso_type] for t in model.Date)
                    else:
                        return Constraint.Skip

                model.consum_eq_year_Ctr = Constraint(model.AREAS, model.FLEX_CONSUM, rule=consum_eq_year)

            else:
                def consum_eq_year(model, conso_type):
                    if model.flex_type[conso_type] == 'year':
                        return sum(model.flex_consumption[t, conso_type] for t in model.Date) == sum(
                            model.to_flex_consumption[t, conso_type] for t in model.Date)
                    else:
                        return Constraint.Skip

                model.consum_eq_year_Ctr = Constraint(model.FLEX_CONSUM, rule=consum_eq_year)
        return model


    model = storage_mw_mwh_binding(model)
    model = set_h2_storage_Ctr(model)
    model= set_Operation_Constraints_consum_flex_Ctr(model)
    model= set_Operation_Constraints_consum_eq_year_Ctr(model)


    return model

