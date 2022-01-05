from functions.f_tools import *

def set_Operation_Constraints_CapacityCtr(model):
    """
    energy <= capacity * availabilityFactor

    :param model:
    :return:
    """
    Set_names = get_allSetsnames(model)

    ### CapacityCtr definition
    match list(Set_names):
        case [*my_set_names] if "AREAS" in my_set_names:
            #multiple area (with or without storage)
            def CapacityCtr_rule(model, area, t, tech):  # INEQ forall t, tech
                return model.capacity[area, tech] * model.availabilityFactor[area, t, tech] >= model.energy[
                    area, t, tech]
            model.CapacityCtr = Constraint(model.AREAS, model.Date, model.TECHNOLOGIES, rule=CapacityCtr_rule)

        case _:
            # single area (with or without storage)
            def Capacity_rule(model, t, tech):  # INEQ forall t, tech
                return model.capacity[tech] * model.availabilityFactor[t, tech] >= model.energy[t, tech]
            model.CapacityCtr = Constraint(model.Date, model.TECHNOLOGIES, rule=Capacity_rule)
    return model

def set_Operation_Constraints_energyCtr(model):
    """
    [Default] energy = areaConsumption

    [if STOCK_TECHNO] energy + storageOut-storageIn = areaConsumption

    [if AREAS] energy + exchange = areaConsumption

    [if AREAS & STOCK_TECHNO] energy + exchange + storageOut-storageIn = areaConsumption

    :param model:
    :return:
    """

    Set_names = get_allSetsnames(model)
    ### CapacityCtr definition
    match list(Set_names):
        case [*my_set_names] if allin(["AREAS", 'STOCK_TECHNO'], my_set_names):
            # multiple area and storage
            def energyCtr_rule(model, area, t):  # INEQ forall t
                return sum(model.energy[area, t, tech] for tech in model.TECHNOLOGIES) + sum(
                    model.exchange[b, area, t] for b in model.AREAS) + sum(
                    model.storageOut[area, t, s_tech] - model.storageIn[area, t, s_tech] for s_tech in model.STOCK_TECHNO) == \
                       model.areaConsumption[area, t]
            model.energyCtr = Constraint(model.AREAS, model.Date, rule=energyCtr_rule)

        case [*my_set_names] if "AREAS" in my_set_names:
            # multiple area without storage
            def energyCtr_rule(model, area, t):  # INEQ forall t
                return sum(model.energy[area, t, tech] for tech in model.TECHNOLOGIES) + sum(
                    model.exchange[b, area, t] for b in model.AREAS) == model.areaConsumption[area, t]
            model.energyCtr = Constraint(model.AREAS, model.Date, rule=energyCtr_rule)

        case [*my_set_names] if 'STOCK_TECHNO' in my_set_names:
            # single area with storage
            def energyCtr_rule(model, t):  # INEQ forall t
                return sum(model.energy[t, tech] for tech in model.TECHNOLOGIES) + sum(
                    model.storageOut[t, s_tech] - model.storageIn[t, s_tech] for s_tech in model.STOCK_TECHNO) == \
                       model.areaConsumption[t]
            model.energyCtr = Constraint(model.Date, rule=energyCtr_rule)

        case _:
            # single area without storage
            def energyCtr_rule(model, t):  # INEQ forall t
                return sum(model.energy[t, tech] for tech in model.TECHNOLOGIES) == model.areaConsumption[t]
            model.energyCtr = Constraint(model.Date, rule=energyCtr_rule)

    return model

def set_Operation_Constraints_exchangeCtr(model):
    """
    [Applies only if "AREAS" in Set_names]

    exchange[a,b] <=  maxExchangeCapacity[a,b]

    exchange[a,b] >= - maxExchangeCapacity[a,b]

    exchange[a,b] == -exchange[b,a]

    :param model:
    :return:
    """
    Set_names = get_allSetsnames(model)
    if "AREAS" in Set_names:
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
    return model

def set_Operation_Constraints_stockCtr(model):
    """
    [if "EnergyNbhourCap"] EnergyNbhourCap *capacity >= sum energy

    :param model:
    :return:
    """
    Set_names = get_allSetsnames(model)
    All_parameters = get_ParametersNames(model)
    if "EnergyNbhourCap" in All_parameters:
        if "AREAS" in Set_names:
            def stock_rule(model, area, tech):  # INEQ forall t, tech
                if model.EnergyNbhourCap[(area, tech)] > 0:
                    return model.EnergyNbhourCap[area, tech] * model.capacity[area, tech] >= sum(
                        model.energy[area, t, tech] for t in model.Date)
                else:
                    return Constraint.Skip
            model.stockCtr = Constraint(model.AREAS, model.TECHNOLOGIES, rule=stock_rule)
        else:
            def stock_rule(model, tech):  # INEQ forall t, tech
                if model.EnergyNbhourCap[tech] > 0:
                    return model.EnergyNbhourCap[tech] * model.capacity[tech] >= sum(
                        model.energy[t, tech] for t in model.Date)
                else:
                    return Constraint.Skip
            model.stockCtr = Constraint(model.TECHNOLOGIES, rule=stock_rule)
    return model

def set_Operation_Constraints_Ramp(model):
    """
    [if RampConstraintPlus] energy[t+1]-energy[t]<= capacity RampConstraintPlus

    [if RampConstraintMoins] energy[t+1]-energy[t]>= - capacity RampConstraintMoins

    [if RampConstraintPlus2] (energy[t+2]+energy[t+3])/2-(energy[t+1]+energy[t])/2<= capacity RampConstraintPlus2

    [if RampConstraintMoins2] (energy[t+2]+energy[t+3])/2-(energy[t+1]+energy[t])/2<= capacity RampConstraintPlus2

    :param model:
    :return:
    """

    All_parameters = get_ParametersNames(model)
    if "RampConstraintPlus" in All_parameters:
        model = set_Operation_Constraints_rampCtrPlus(model)
    if "RampConstraintMoins" in All_parameters:
        model = set_Operation_Constraints_rampCtrMoins(model)
    if "RampConstraintPlus2" in All_parameters:
        model = set_Operation_Constraints_rampCtrPlus2(model)
    if "RampConstraintMoins2" in All_parameters:
        model = set_Operation_Constraints_rampCtrMoins2(model)

    return model;

def set_Operation_Constraints_rampCtrPlus(model):
    """
    energy[t+1]-energy[t]<= capacity RampConstraintPlus

    :param model:
    :return:
    """
    Set_names = get_allSetsnames(model)
    match list(Set_names):
        case [*my_set_names] if "AREAS" in my_set_names:
            # multiple area
            def rampCtrPlus_rule(model, area, t, tech):  # INEQ forall t<
                if model.RampConstraintPlus[(area, tech)] > 0:
                    t_plus_1 = t + timedelta(hours=1)
                    return model.energy[area, t_plus_1, tech] - model.energy[area, t, tech] <= model.capacity[
                        area, tech] * \
                           model.RampConstraintPlus[area, tech];
                else:
                    return Constraint.Skip
            model.rampCtrPlus = Constraint(model.AREAS, model.Date_MinusOne, model.TECHNOLOGIES, rule=rampCtrPlus_rule)
        case _:
            # single area
            def rampCtrPlus_rule(model, t, tech):  # INEQ forall t<
                if model.RampConstraintPlus[tech] > 0:
                    t_plus_1 = t + timedelta(hours=1)
                    return model.energy[t_plus_1, tech] - model.energy[t, tech] <= model.capacity[tech] * \
                           model.RampConstraintPlus[tech];
                else:
                    return Constraint.Skip
            model.rampCtrPlus = Constraint(model.Date_MinusOne, model.TECHNOLOGIES, rule=rampCtrPlus_rule)

    return model;

def set_Operation_Constraints_rampCtrMoins(model):
    """
    energy[t+1]-energy[t]>= - capacity RampConstraintMoins

    :param model:
    :return:
    """
    Set_names = get_allSetsnames(model)
    match list(Set_names):
        case [*my_set_names] if "AREAS" in my_set_names:
            # multiple area
            def rampCtrMoins_rule(model, area, t, tech):  # INEQ forall t<
                if model.RampConstraintMoins[area, tech] > 0.:
                    t_plus_1 = t + timedelta(hours=1)
                    return model.energy[area, t_plus_1, tech] - model.energy[area, t, tech] >= - model.capacity[
                        area, tech] * \
                           model.RampConstraintMoins[area, tech];
                else:
                    return Constraint.Skip
            model.rampCtrMoins = Constraint(model.AREAS, model.Date_MinusOne, model.TECHNOLOGIES,
                                            rule=rampCtrMoins_rule)
        case _:
            # single area
            def rampCtrMoins_rule(model, t, tech):  # INEQ forall t<
                if model.RampConstraintMoins[tech] > 0:
                    t_plus_1 = t + timedelta(hours=1)
                    return model.energy[t_plus_1, tech] - model.energy[t, tech] >= - model.capacity[tech] * \
                           model.RampConstraintMoins[tech];
                else:
                    return Constraint.Skip
            model.rampCtrMoins = Constraint(model.Date_MinusOne, model.TECHNOLOGIES, rule=rampCtrMoins_rule)

    return model;

def set_Operation_Constraints_rampCtrPlus2(model):
    """
        (energy[t+2]+energy[t+3])/2-(energy[t+1]+energy[t])/2<= capacity RampConstraintPlus2

        :param model:
        :return:
    """
    Set_names = get_allSetsnames(model)
    match list(Set_names):
        case [*my_set_names] if "AREAS" in my_set_names:
            # multiple area
            def rampCtrPlus2_rule(model, area, t, tech):  # INEQ forall t<
                if model.RampConstraintPlus2[(area, tech)] > 0.:
                    t_plus_1 = t + timedelta(hours=1)
                    t_plus_2 = t + timedelta(hours=2)
                    t_plus_3 = t + timedelta(hours=3)
                    var = (model.energy[area, t_plus_2, tech] + model.energy[area, t_plus_3, tech]) / 2 - (
                            model.energy[area, t_plus_1, tech] + model.energy[area, t, tech]) / 2;
                    return var <= model.capacity[area, tech] * model.RampConstraintPlus[area, tech];
                else:
                    return Constraint.Skip

            model.rampCtrPlus2 = Constraint(model.AREAS, model.Date_MinusThree, model.TECHNOLOGIES,
                                            rule=rampCtrPlus2_rule)
        case _:
            # single area
            def rampCtrPlus2_rule(model, t, tech):  # INEQ forall t<
                if model.RampConstraintPlus2[tech] > 0:
                    t_plus_1 = t + timedelta(hours=1)
                    t_plus_2 = t + timedelta(hours=2)
                    t_plus_3 = t + timedelta(hours=3)
                    var = (model.energy[t_plus_2, tech] + model.energy[t_plus_3, tech]) / 2 - (
                            model.energy[t_plus_1, tech] + model.energy[t, tech]) / 2;
                    return var <= model.capacity[tech] * model.RampConstraintPlus2[tech];
                else:
                    return Constraint.Skip

            model.rampCtrPlus2 = Constraint(model.Date_MinusThree, model.TECHNOLOGIES, rule=rampCtrPlus2_rule)

    return model;

def set_Operation_Constraints_rampCtrMoins2(model):
    """
    (energy[t+2]+energy[t+3])/2-(energy[t+1]+energy[t])/2>= - capacity RampConstraintMoins2

    :param model:
    :return:
    """
    Set_names = get_allSetsnames(model)
    match list(Set_names):
        case [*my_set_names] if "AREAS" in my_set_names:
            # multiple area
            def rampCtrMoins2_rule(model, area, t, tech):  # INEQ forall t<
                if model.RampConstraintMoins2[(area, tech)] > 0:
                    t_plus_1 = t + timedelta(hours=1)
                    t_plus_2 = t + timedelta(hours=2)
                    t_plus_3 = t + timedelta(hours=3)
                    var = (model.energy[area, t_plus_2, tech] + model.energy[area, t_plus_3, tech]) / 2 - (
                            model.energy[area, t_plus_1, tech] + model.energy[area, t, tech]) / 2;
                    return var >= - model.capacity[area, tech] * model.RampConstraintMoins2[area, tech];
                else:
                    return Constraint.Skip
            model.rampCtrMoins2 = Constraint(model.AREAS, model.Date_MinusThree, model.TECHNOLOGIES,
                                             rule=rampCtrMoins2_rule)
        case _:
            # single area
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

def set_Operation_Constraints_Storage(model):
    """
    is applied only if 'STOCK_TECHNO' exists

    storageIn <= Pmax

    storageOut <= Pmax

    stockLevel[t] = stockLevel[t-1]*(1 - dissipation) -storageOut/efficiency_out

    stockLevel <= Cmax

    :param model:
    :return:
    """
    Set_names = get_allSetsnames(model)
    if 'STOCK_TECHNO' in Set_names:
        model = set_Operation_Constraints_StoragePowerUBCtr(model) # storageIn-Pmax <= 0
        model = set_Operation_Constraints_StoragePowerLBCtr(model) # storageOut-Pmax <= 0
        model = set_Operation_Constraints_StorageLevelCtr(model) # stockLevel[t] = stockLevel[t-1]*(1 - dissipation) -storageOut/efficiency_out
        model = set_Operation_Constraints_StorageCapacityCtr(model) # stockLevel <= Cmax
    return model

def set_Operation_Constraints_StoragePowerUBCtr(model):
    """
    storageIn-Pmax <= 0

    :param model:
    :return:
    """
    Set_names = get_allSetsnames(model)
    match list(Set_names):
        case [*my_set_names] if "AREAS" in my_set_names:
            # multiple area
            def StoragePowerUB_rule(model, area, t, s_tech):  # INEQ forall t
                return model.storageIn[area, t, s_tech] - model.Pmax[area, s_tech] <= 0
            model.StoragePowerUBCtr = Constraint(model.AREAS, model.Date, model.STOCK_TECHNO, rule=StoragePowerUB_rule)
        case _:
            # single area
            def StoragePowerUB_rule(model, t, s_tech):  # INEQ forall t
                return model.storageIn[t, s_tech] - model.Pmax[s_tech] <= 0
            model.StoragePowerUBCtr = Constraint(model.Date, model.STOCK_TECHNO, rule=StoragePowerUB_rule)

    return model

def set_Operation_Constraints_StoragePowerLBCtr(model):
    """
    storageOut-Pmax<=0

    :param model:
    :return:
    """
    Set_names = get_allSetsnames(model)
    match list(Set_names):
        case [*my_set_names] if "AREAS" in my_set_names:
            # multiple area
            def StoragePowerLB_rule(model, area, t, s_tech, ):  # INEQ forall t
                return model.storageOut[area, t, s_tech] - model.Pmax[area, s_tech] <= 0
            model.StoragePowerLBCtr = Constraint(model.AREAS, model.Date, model.STOCK_TECHNO, rule=StoragePowerLB_rule)
        case _:
            # single area
            def StoragePowerLB_rule(model, t, s_tech, ):  # INEQ forall t
                return model.storageOut[t, s_tech] - model.Pmax[s_tech] <= 0
            model.StoragePowerLBCtr = Constraint(model.Date, model.STOCK_TECHNO, rule=StoragePowerLB_rule)

    return model

def set_Operation_Constraints_StorageLevelCtr(model):
    """
    stockLevel[t] = stockLevel[t-1]*(1 - dissipation) -storageOut/efficiency_out

    :param model:
    :return:
    """
    Set_names = get_allSetsnames(model)
    match list(Set_names):
        case [*my_set_names] if "AREAS" in my_set_names:
            # multiple area
            def StorageLevel_rule(model, area, t, s_tech):  # EQ forall t
                if t != min(getattr(model, "Date").data()):
                    t_moins_1 = t - timedelta(hours=1)
                    return model.stockLevel[area, t, s_tech] == model.stockLevel[area, t_moins_1, s_tech] * (
                            1 - model.dissipation[area, s_tech]) + model.storageIn[area, t, s_tech] * \
                           model.efficiency_in[area, s_tech] - model.storageOut[area, t, s_tech] / model.efficiency_out[
                               area, s_tech]
                else:
                    return model.stockLevel[area, t, s_tech] == 0
            model.StorageLevelCtr = Constraint(model.AREAS, model.Date, model.STOCK_TECHNO, rule=StorageLevel_rule)
        case _:
            # single area
            def StorageLevel_rule(model, t, s_tech):  # EQ forall t
                if t != min(getattr(model, "Date").data()):
                    t_moins_1 = t - timedelta(hours=1)
                    return model.stockLevel[t, s_tech] == model.stockLevel[t_moins_1, s_tech] * (
                            1 - model.dissipation[s_tech]) + model.storageIn[t, s_tech] * model.efficiency_in[s_tech] - \
                           model.storageOut[t, s_tech] / model.efficiency_out[s_tech]
                else:
                    return model.stockLevel[t, s_tech] == 0
            model.StorageLevelCtr = Constraint(model.Date, model.STOCK_TECHNO, rule=StorageLevel_rule)

    return model

def set_Operation_Constraints_StorageCapacityCtr(model):
    """
    stockLevel <= Cmax

    :param model:
    :return:
    """
    Set_names = get_allSetsnames(model)
    match list(Set_names):
        case [*my_set_names] if "AREAS" in my_set_names:
            # multiple area
            def StorageCapacity_rule(model, area, t, s_tech, ):  # INEQ forall t
                return model.stockLevel[area, t, s_tech] <= model.Cmax[area, s_tech]

            model.StorageCapacityCtr = Constraint(model.AREAS, model.Date, model.STOCK_TECHNO, rule=StorageCapacity_rule)
        case _:
            # single area
            def StorageCapacity_rule(model, t, s_tech, ):  # INEQ forall t
                return model.stockLevel[t, s_tech] <= model.Cmax[s_tech]
            model.StorageCapacityCtr = Constraint(model.Date, model.STOCK_TECHNO, rule=StorageCapacity_rule)

    return model



