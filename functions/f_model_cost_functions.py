from functions.f_tools import *

def set_Planing_base_cost_function(model):
    """
    Definition of the cost function

    min energyCosts + capacityCosts + storageCosts + consumption_power_cost + lab_cost

    energyCosts = sum (energy * energyCost)

    capacityCosts = capacityCost * len(model.Date)/ 8760 *capacity

    storageCosts = storageCost * Cmax

    consumption_power_cost = LoadCost * increased_max_power

    lab_cost = labour_ratio * labourcost *(a_minus+a_plus)

    :param model:
    :return:
    """

    model = set_Planing_cost_OBJ(model) # min energyCosts + capacityCosts + storageCosts
    # model = set_Planing_cost_energyCostsCtr(model) # energyCosts = sum (energy * energyCost)
    model = set_Planing_cost_capacityCostsCtr(model)#  capacityCosts = capacityCost * len(model.Date)/ 8760 *capacity
    Set_names = get_allSetsnames(model)
    if 'STOCK_TECHNO' in Set_names:
        model = set_Planing_cost_storageCostsCtr(model) #  storageCosts = storageCost * Cmax
    if "FLEX_CONSUM" in Set_names:
        model = Planing_cost_consumption_power_costCtr(model) #  consumption_power_cost = LoadCost * increased_max_power
        model = set_Planing_cost_labour_costCtr(model) # lab_cost = labour_ratio * labourcost *(a_minus+a_plus)

    return model

def set_Planing_cost_OBJ(model):
    Set_names = get_allSetsnames(model)
    match list(Set_names):
        #case [*my_set_names] if allin(["AREAS", 'STOCK_TECHNO'], my_set_names):
        case [*my_set_names] if allin(["FLEX_CONSUM","AREAS", 'STOCK_TECHNO'], my_set_names):
            # multiple area and storage
            def ObjectiveFunction_rule(model):  # OBJ
                return sum( model.energy[area, t, tech] * model.energyCost[area, tech] +\
                            #+ model.margvarCost[area, tech] * model.energy[area, t, tech] ** 2 +\
                    model.capacityCosts[area, tech] for tech in model.TECHNOLOGIES for area in model.AREAS for t in model.Date) + \
                       sum( model.storageCosts[area, s_tech] for s_tech in model.STOCK_TECHNO for area in model.AREAS) + \
                       sum(model.consumption_power_cost[area,name] for name in model.FLEX_CONSUM for area in model.AREAS)+ \
                       sum(model.lab_cost[area,t, name] for t in model.Date for name in model.FLEX_CONSUM for area in model.AREAS)

            model.OBJ = Objective(rule=ObjectiveFunction_rule, sense=minimize)
        case[*my_set_names] if  allin(["FLEX_CONSUM", 'STOCK_TECHNO'], my_set_names):
            # single area with storage and Flex consumption
            def ObjectiveFunction_rule(model):  # OBJ
                return sum(model.energy[t, tech] * model.energyCost[tech] + model.margvarCost[tech] * model.energy[
                        t, tech] ** 2 + model.capacityCosts[tech] for tech in model.TECHNOLOGIES for t in model.Date) + sum(
                    model.storageCosts[s_tech] for s_tech in model.STOCK_TECHNO) + sum(model.consumption_power_cost[name] + \
                                                                                 sum(model.lab_cost[t, name] for t in
                                                                                     model.Date) \
                                                                                 for name in model.FLEX_CONSUM)
            model.OBJ = Objective(rule=ObjectiveFunction_rule, sense=minimize)

        case [*my_set_names] if allin(["AREAS", 'STOCK_TECHNO'], my_set_names):
            # multiple area with storage
            def ObjectiveFunction_rule(model):  # OBJ ##TODO à corriger pour rajouter le storage cost
                return sum(
                    model.energy[area, t, tech] * model.energyCost[area, tech] +\
                    #model.margvarCost[area, tech] * model.energy[area, t, tech] ** 2 + \
                     model.capacityCosts[area, tech] for t in model.Date for tech in model.TECHNOLOGIES for area
                    in model.AREAS) + sum(model.storageCosts[area,s_tech] for area in model.AREAS for s_tech in model.STOCK_TECHNO)

            model.OBJ = Objective(rule=ObjectiveFunction_rule, sense=minimize)

        case [*my_set_names] if "AREAS" in my_set_names:
            # multiple area without storage
            def ObjectiveFunction_rule(model):  # OBJ
                return sum(model.energy[area, t, tech]*model.energyCost[area, tech]#+ model.margvarCost[area,tech]*model.energy[area, t, tech]**2
                           + model.capacityCosts[area, tech] for t in model.Date for tech in model.TECHNOLOGIES for area
                    in model.AREAS)
            model.OBJ = Objective(rule=ObjectiveFunction_rule, sense=minimize)
        case [*my_set_names] if 'STOCK_TECHNO' in my_set_names:
            # single area with storage
            def ObjectiveFunction_rule(model):  # OBJ
                return sum(model.energy[t, tech] * model.energyCost[tech] +\
                           # model.margvarCost[tech] *
                           # model.energy[
                           #     t, tech] ** 2
                           + model.capacityCosts[tech] for t in model.Date for tech in model.TECHNOLOGIES) + sum(
                    model.storageCosts[s_tech] for s_tech in model.STOCK_TECHNO)
            model.OBJ = Objective(rule=ObjectiveFunction_rule, sense=minimize)

        case _:
            # single area without storage
            def ObjectiveFunction_rule(model):  # OBJ
                return sum(model.energy[t, tech] * model.energyCost[tech] +\
                           # model.margvarCost[tech] *
                           # model.energy[
                           #     t, tech] ** 2  +\
                           model.capacityCosts[tech] for tech in model.TECHNOLOGIES for t in model.Date)
            model.OBJ = Objective(rule=ObjectiveFunction_rule, sense=minimize)
    return model

# def set_Planing_cost_energyCostsCtr(model):
#     Set_names = get_allSetsnames(model)
#     match list(Set_names):
#         case [*my_set_names] if "AREAS" in my_set_names:
#             # multiple area without storage
#
#             def energyCostsDef_rule(model, area,tech):  # EQ forall tech in TECHNOLOGIES   energyCosts  = sum{t in Date} energyCost[tech]*energy[t,tech] / 1E6;
#                 temp = model.energyCost[area, tech]  # /10**6 ;
#                 return sum(temp * model.energy[area, t, tech] for t in model.Date) == model.energyCosts[area, tech];
#             # model.energyCostsDef = Constraint(model.AREAS, model.TECHNOLOGIES, rule=energyCostsDef_rule)
#
#         case _:
#             # single area without storage
#             def energyCostsDef_rule(model,tech):  # EQ forall tech in TECHNOLOGIES   energyCosts  = sum{t in Date} energyCost[tech]*energy[t,tech] / 1E6;
#                 temp = model.energyCost[tech]  # /10**6 ;
#                 return sum(temp * model.energy[t, tech] for t in model.Date) == model.energyCosts[tech];
#             # model.energyCostsCtr = Constraint(model.TECHNOLOGIES, rule=energyCostsDef_rule)
#     return model

def set_Planing_cost_capacityCostsCtr(model):
    Set_names = get_allSetsnames(model)
    match list(Set_names):
        case [*my_set_names] if "AREAS" in my_set_names:
            # multiple area
            def capacityCostsDef_rule(model, area,
                                      tech):  # EQ forall tech in TECHNOLOGIES   energyCosts  = sum{t in Date} energyCost[tech]*energy[t,tech] / 1E6;
                temp = model.capacityCosts[area, tech]  # /10**6 ;
                return model.capacityCost[area, tech] * len(model.Date) / 8760 * model.capacity[area, tech] == \
                       model.capacityCosts[area, tech]  # .. ....... / 10**6

            model.capacityCostsCtr = Constraint(model.AREAS, model.TECHNOLOGIES, rule=capacityCostsDef_rule)
        case _:
            # single area
            def capacityCostsDef_rule(model,tech):  # EQ forall tech in TECHNOLOGIES   energyCosts  = sum{t in Date} energyCost[tech]*energy[t,tech] / 1E6;
                temp = model.capacityCosts[tech]  # /10**6 ;
                return model.capacityCost[tech] * model.capacity[tech] == model.capacityCosts[tech]
                # return model.capacityCost[tech] * len(model.Date) / 8760 * model.capacity[tech] / 10 ** 6 == model.capacityCosts[tech]
            model.capacityCostsCtr = Constraint(model.TECHNOLOGIES, rule=capacityCostsDef_rule)
    return model

def set_Planing_cost_storageCostsCtr(model):
    Set_names = get_allSetsnames(model)
    match list(Set_names):
        case [*my_set_names] if "AREAS" in my_set_names:
            # multiple area without storage
            def storageCostsDef_rule(model, area,
                                     s_tech):  # EQ forall s_tech in STOCK_TECHNO storageCosts = storageCost[area, s_tech]*c_max[area, s_tech] / 1E6;
                return model.storageCost[area, s_tech] * model.Cmax[area, s_tech] == model.storageCosts[
                    area, s_tech]  # /10**6 ;;

            model.storageCostsDef = Constraint(model.AREAS, model.STOCK_TECHNO, rule=storageCostsDef_rule)

        case _:
            # single area without storage
            def storageCostsDef_rule(model,
                                     s_tech):  # EQ forall s_tech in STOCK_TECHNO storageCosts=storageCost[s_tech]*Cmax[s_tech] / 1E6;
                return model.storageCost[s_tech] * model.Cmax[s_tech] == model.storageCosts[s_tech]

            model.storageCostsCtr = Constraint(model.STOCK_TECHNO, rule=storageCostsDef_rule)
    return model


def Planing_cost_consumption_power_costCtr(model):
    # definition of demand investiement cost
    Set_names = get_allSetsnames(model)
    match list(Set_names):
        case [*my_set_names] if "AREAS" in my_set_names:
            def consumption_power_cost_rule(model, area, conso_type):
                return model.consumption_power_cost[area,conso_type] == model.LoadCost[area,conso_type] * model.increased_max_power[
                    area,conso_type]
            model.consumption_power_costCtr = Constraint(model.AREAS,model.FLEX_CONSUM, rule=consumption_power_cost_rule)
        case _:
            def consumption_power_cost_rule(model, conso_type):
                return model.consumption_power_cost[conso_type] == model.LoadCost[conso_type] * model.increased_max_power[
                    conso_type]

            model.consumption_power_costCtr = Constraint(model.FLEX_CONSUM, rule=consumption_power_cost_rule)

    return model

def set_Planing_cost_labour_costCtr(model):
    Set_names = get_allSetsnames(model)
    match list(Set_names):
        case [*my_set_names] if "AREAS" in my_set_names:
            def labour_cost_rule(model, area, conso_type, t):
                return model.lab_cost[area,t, conso_type] == model.labour_ratio[area,t,conso_type] * model.labourcost[area,conso_type] * (
                        model.a_plus[area,t, conso_type] + model.a_minus[area,t, conso_type])
            model.labour_costCtr = Constraint(model.AREAS,model.FLEX_CONSUM, model.Date, rule=labour_cost_rule)
        case _:
            def labour_cost_rule(model, conso_type, t):
                return model.lab_cost[t, conso_type] == model.labour_ratio[t,conso_type] * model.labourcost[conso_type] * (
                        model.a_plus[t, conso_type] + model.a_minus[t, conso_type])
            model.labour_costCtr = Constraint(model.FLEX_CONSUM, model.Date, rule=labour_cost_rule)
    return model