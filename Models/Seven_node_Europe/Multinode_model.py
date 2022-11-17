from datetime import timedelta
import pandas as pd
from pyomo.core import *

from EnergyAlternativesPlaning import *
from EnergyAlternativesPlaning.f_tools import *
from EnergyAlternativesPlaning.f_model_definition import *
from EnergyAlternativesPlaning.f_model_cost_functions import *
from EnergyAlternativesPlaning.f_model_planing_constraints import *
from EnergyAlternativesPlaning.f_model_operation_constraints import *

def GetElectricitySystemModel(Parameters):
    ########
    # Sets #
    ########
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

    #Empty for now


    return model

