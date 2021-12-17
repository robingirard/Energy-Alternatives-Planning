from datetime import timedelta
import pandas as pd
from pyomo.core import *


def set_EnergyNbHourCap_single_area(model, TechParameters):
    if "EnergyNbhourCap" in TechParameters:
        def storage_rule(model,tech) : #INEQ forall t, tech
            if model.EnergyNbhourCap[tech]>0 :
                return model.EnergyNbhourCap[tech]*model.capacity[tech] >= sum(model.energy[t,tech] for t in model.Date)
            else:
                return Constraint.Skip
        model.storageCtr = Constraint(model.TECHNOLOGIES, rule=storage_rule)
        return model;

def set_EnergyNbHourCap_multiple_areas(model, TechParameters):
    if "EnergyNbhourCap" in TechParameters:
        def storageCtr_rule(model,area,tech) : #INEQ forall t, tech
            if model.EnergyNbhourCap[(area,tech)]>0 :
                return model.EnergyNbhourCap[area,tech]*model.capacity[area,tech] >= sum(model.energy[area,t,tech] for t in model.Date)
            else:
                return Constraint.Skip
        model.storageCtr = Constraint(model.AREAS,model.TECHNOLOGIES, rule=storageCtr_rule)
        return model;

def set_RampConstraints_single_area(model, TechParameters):
    '''
    |  Set ramp constraints for single area model.
    |  e_{t+1,tech}-e_{t,tech}<= cap_{tech} RampConstraintPlus_{tech}
    |  e_{t+1,tech}-e_{t,tech}>= cap_{tech} RampConstraintMoins_{tech}
    |  (e_{t+2,tech}+e_{t+3,tech})/2-(e_{t+1,tech}+e_{t,tech})/2<= cap_{tech} RampConstraintPlus2_{tech}
    |  (e_{t+2,tech}+e_{t+3,tech})/2-(e_{t+1,tech}+e_{t,tech})/2>= cap_{tech} RampConstraintMoins2_{tech}
     Constraints are set only if TechParameters
    contains the corresponding columns
    :param model:
    :param TechParameters:
    :return:
    '''
    if "RampConstraintPlus" in TechParameters:
        def rampCtrPlus_rule(model, t, tech):  # INEQ forall t<
            if model.RampConstraintPlus[tech] > 0:
                t_plus_1 = t + timedelta(hours=1)
                return model.energy[t_plus_1, tech] - model.energy[t, tech] <= model.capacity[tech] * \
                       model.RampConstraintPlus[tech];
            else:
                return Constraint.Skip

        model.rampCtrPlus = Constraint(model.Date_MinusOne, model.TECHNOLOGIES, rule=rampCtrPlus_rule)

    if "RampConstraintMoins" in TechParameters:
        def rampCtrMoins_rule(model, t, tech):  # INEQ forall t<
            if model.RampConstraintMoins[tech] > 0:
                t_plus_1 = t + timedelta(hours=1)
                return model.energy[t_plus_1, tech] - model.energy[t, tech] >= - model.capacity[tech] * \
                       model.RampConstraintMoins[tech];
            else:
                return Constraint.Skip

        model.rampCtrMoins = Constraint(model.Date_MinusOne, model.TECHNOLOGIES, rule=rampCtrMoins_rule)

    if "RampConstraintPlus2" in TechParameters:
        def rampCtrPlus2_rule(model, t, tech):  # INEQ forall t<
            if model.RampConstraintPlus2[tech] > 0:
                t_plus_1 = t + timedelta(hours=1)
                t_plus_2 = t + timedelta(hours=2)
                t_plus_3 = t + timedelta(hours=3)
                var = (model.energy[t_plus_2, tech] + model.energy[t_plus_3, tech]) / 2 - (
                            model.energy[t_plus_1, tech] + model.energy[t, tech]) / 2;
                return var <= model.capacity[tech] * model.RampConstraintPlus[tech];
            else:
                return Constraint.Skip

        model.rampCtrPlus2 = Constraint(model.Date_MinusThree, model.TECHNOLOGIES, rule=rampCtrPlus2_rule)

    if "RampConstraintMoins2" in TechParameters:
        def rampCtrMoins2_rule(model, t, tech):  # INEQ forall t<
            if model.RampConstraintMoins2[tech] > 0:
                t_plus_1 = t + timedelta(hours=1)
                t_plus_2 = t + timedelta(hours=2)
                t_plus_3 = t + timedelta(hours=3)
                var = (model.energy[t_plus_2, tech] + model.energy[t_plus_3, tech]) / 2 - (
                            model.energy[t_plus_1, tech] + model.energy[t, tech]) / 2;
                return var >= - model.capacity[tech] * model.RampConstraintMoins2[tech];
            else:
                return Constraint.Skip

        model.rampCtrMoins2 = Constraint(model.Date_MinusThree, model.TECHNOLOGIES, rule=rampCtrMoins2_rule)

    return model;


def set_RampConstraints_multiple_areas(model,TechParameters):
    '''
    |  Set ramp constraints for multiple area model.
    |  e_{area,t+1,tech}-e_{area,t,tech}<= cap_{area,tech} RampConstraintPlus_{area,tech}
    |  e_{area,t+1,tech}-e_{area,t,tech}>= cap_{area,tech} RampConstraintMoins_{area,tech}
    |  (e_{area,t+2,tech}+e_{area,t+3,tech})/2-(e_{area,t+1,tech}+e_{area,t,tech})/2<= cap_{area,tech} RampConstraintPlus2_{area,tech}
    |  (e_{area,t+2,tech}+e_{area,t+3,tech})/2-(e_{area,t+1,tech}+e_{area,t,tech})/2>= cap_{area,tech} RampConstraintMoins2_{area,tech}
    Constraints are set only if TechParameters
    contains the corresponding columns
    :param model:
    :param TechParameters:
    :return:
    '''
    if "RampConstraintPlus" in TechParameters:
        def rampCtrPlus_rule(model,area,t,tech): #INEQ forall t<
            if model.RampConstraintPlus[(area,tech)]>0 :
                t_plus_1 = t + timedelta(hours=1)
                return model.energy[area,t_plus_1,tech]  - model.energy[area,t,tech] <= model.capacity[area,tech]*model.RampConstraintPlus[area,tech] ;
            else:
                return Constraint.Skip
        model.rampCtrPlus = Constraint(model.AREAS,model.Date_MinusOne,model.TECHNOLOGIES,rule=rampCtrPlus_rule)

    if "RampConstraintMoins" in TechParameters:
        def rampCtrMoins_rule(model,area,t,tech): #INEQ forall t<
            if model.RampConstraintMoins[area,tech]>0. :
                t_plus_1 = t + timedelta(hours=1)
                return model.energy[area,t_plus_1,tech]  - model.energy[area,t,tech] >= - model.capacity[area,tech]*model.RampConstraintMoins[area,tech] ;
            else:
                return Constraint.Skip
        model.rampCtrMoins = Constraint(model.AREAS,model.Date_MinusOne,model.TECHNOLOGIES,rule=rampCtrMoins_rule)

    if "RampConstraintPlus2" in TechParameters:
        def rampCtrPlus2_rule(model,area,t,tech): #INEQ forall t<
            if model.RampConstraintPlus2[(area,tech)]>0. :
                t_plus_1 = t + timedelta(hours=1)
                t_plus_2 = t + timedelta(hours=2)
                t_plus_3 = t + timedelta(hours=3)
                var=(model.energy[area,t_plus_2,tech]+model.energy[area,t_plus_3,tech])/2 -  (model.energy[area,t_plus_1,tech]+model.energy[area,t,tech])/2;
                return var <= model.capacity[area,tech]*model.RampConstraintPlus[area,tech] ;
            else:
                return Constraint.Skip
        model.rampCtrPlus2 = Constraint(model.AREAS,model.Date_MinusThree,model.TECHNOLOGIES,rule=rampCtrPlus2_rule)

    if "RampConstraintMoins2" in TechParameters:
        def rampCtrMoins2_rule(model,area,t,tech): #INEQ forall t<
            if model.RampConstraintMoins2[(area,tech)]>0 :
                t_plus_1 = t + timedelta(hours=1)
                t_plus_2 = t + timedelta(hours=2)
                t_plus_3 = t + timedelta(hours=3)
                var=(model.energy[area,t_plus_2,tech]+model.energy[area,t_plus_3,tech])/2 -  (model.energy[area,t_plus_1,tech]+model.energy[area,t,tech])/2;
                return var >= - model.capacity[area,tech]*model.RampConstraintMoins2[area,tech] ;
            else:
                return Constraint.Skip
        model.rampCtrMoins2 = Constraint(model.AREAS,model.Date_MinusThree,model.TECHNOLOGIES,rule=rampCtrMoins2_rule)

    return model;


    ### Contraintes de rampe
    # C1
    #     subject to rampCtrPlus{a in AREAS, h in DateMOINS1, t in TECHNOLOGIES : RampConstraintPlus[a,t]>0 } :
    #         energy[a,h+1,t] - energy[a,h,t] <= capacity[a,t]*RampConstraintPlus[a,t] ;

    # subject to rampCtrMoins{a in AREAS, h in DateMOINS1, t in TECHNOLOGIES : RampConstraintMoins[a,t]>0 } :
    #  energy[a,h+1,t] - energy[a,h,t] >= - capacity[a,t]*RampConstraintMoins[a,t] ;

    #  /*contrainte de rampe2 */
    # subject to rampCtrPlus2{a in AREAS, h in DateMOINS4, t in TECHNOLOGIES : RampConstraintPlus2[a,t]>0 } :
    #  (energy[a,h+2,t]+energy[a,h+3,t])/2 -  (energy[a,h+1,t]+energy[a,h,t])/2 <= capacity[a,t]*RampConstraintPlus2[a,t] ;

    # subject to rampCtrMoins2{a in AREAS, h in DateMOINS4, t in TECHNOLOGIES : RampConstraintMoins2[a,t]>0 } :
    #   (energy[a,h+2,t]+energy[a,h+3,t])/2 -  (energy[a,h+1,t]+energy[a,h,t])/2 >= - capacity[a,t]*RampConstraintMoins2[a,t] ;

def set_storage_operation_constraints_single_area(model,Date_list):
    # contraintes de stock puissance
    def StoragePowerUB_rule(model, t, s_tech):  # INEQ forall t
        return model.storageIn[t, s_tech] - model.p_max[s_tech] <= 0

    model.StoragePowerUBCtr = Constraint(model.Date, model.STOCK_TECHNO, rule=StoragePowerUB_rule)

    def StoragePowerLB_rule(model, t, s_tech, ):  # INEQ forall t
        return model.storageOut[t, s_tech] - model.p_max[s_tech] <= 0

    model.StoragePowerLBCtr = Constraint(model.Date, model.STOCK_TECHNO, rule=StoragePowerLB_rule)

    # contraintes de stock capacité
    def StockLevel_rule(model, t, s_tech):  # EQ forall t
        if t != Date_list[0]:
            t_moins_1 = t - timedelta(hours=1)
            return model.stockLevel[t, s_tech] == model.stockLevel[t_moins_1, s_tech] * (
                        1 - model.dissipation[s_tech]) + model.storageIn[t, s_tech] * model.efficiency_in[s_tech] - \
                   model.storageOut[t, s_tech] / model.efficiency_out[s_tech]
        else:
            return model.stockLevel[t, s_tech] == 0

    model.StockLevelCtr = Constraint(model.Date, model.STOCK_TECHNO, rule=StockLevel_rule)
    return model;

def set_storage_operation_constraints_multiple_area(model,Date_list):
    # contraintes de stock puissance
    # AREAS x Date x STOCK_TECHNO
    def StoragePowerUB_rule(model, area, t, s_tech):  # INEQ forall t
        return model.storageIn[area, t, s_tech] - model.Pmax[area, s_tech] <= 0

    model.StoragePowerUBCtr = Constraint(model.AREAS, model.Date, model.STOCK_TECHNO, rule=StoragePowerUB_rule)

    def StoragePowerLB_rule(model, area, t, s_tech, ):  # INEQ forall t
        return model.storageOut[area, t, s_tech] - model.Pmax[area, s_tech] <= 0

    model.StoragePowerLBCtr = Constraint(model.AREAS, model.Date, model.STOCK_TECHNO, rule=StoragePowerLB_rule)

    # contraintes de stock capacité
    # AREAS x Date x STOCK_TECHNO
    def StockLevel_rule(model, area, t, s_tech):  # EQ forall t
        if t != Date_list[0]:
            t_moins_1 = t - timedelta(hours=1)
            return model.stockLevel[area, t, s_tech] == model.stockLevel[area, t_moins_1, s_tech] * (
                        1 - model.dissipation[area, s_tech]) + model.storageIn[area, t, s_tech] * \
                   model.efficiency_in[area, s_tech] - model.storageOut[area, t, s_tech] / model.efficiency_out[
                       area, s_tech]
        else:
            return model.stockLevel[area, t, s_tech] == 0

    model.StockLevelCtr = Constraint(model.AREAS, model.Date, model.STOCK_TECHNO, rule=StockLevel_rule)
    return model;