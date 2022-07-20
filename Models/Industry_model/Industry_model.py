from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.core import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


input_path = "Input/Steel/Data/"
Resources_characteristics=pd.read_excel(input_path+"Resources_characteristics.xlsx").set_index("Resource")
Technologies_Parameters = pd.read_excel(input_path + "Steel_Technologies.xlsx").fillna(0)
Available_Technologies = pd.read_excel(input_path + "Steel_available_techs_2015.xlsx").fillna(0)
Production = pd.read_excel(input_path + "Steel_production_2015.xlsx").fillna(0)

def main(Technologies_Parameters,Available_Technologies,Production,intermediary_resources=[],intermediary_techs=[],opti2mini="emissions",carbon_tax=0):
    model = ConcreteModel()

    #####################
    # Data preparation ##
    #####################
    Technologies_Parameters_other = Technologies_Parameters[
        Technologies_Parameters.Resource.isin(["Emissions", "flow_cost","capex","CRF"])].set_index('Resource')
    Technologies_Parameters = Technologies_Parameters[
        ~Technologies_Parameters.Resource.isin(["Emissions", "flow_cost","capex","lifetime","discount_rate","CRF"])].set_index('Resource')
    Available_Technologies = Available_Technologies.set_index('Technologies')
    Production = Production.set_index('Resource')

    resources_techs = ["Coal", "Gas", "Biogas","Biomass", "Oil", "Electricity", "E_boiler", "Gas_boiler", "SMR", "Electrolyser"]+intermediary_techs
    resources_techs=list(dict.fromkeys(resources_techs)) #to avoid double values

    energy_resources_list = ["hydrogen","gas","coal","biomass","oil","steam","electricity"]
    energy_resources_list=list(dict.fromkeys(energy_resources_list))

    production_tech_list = list(dict.fromkeys(Available_Technologies.index.get_level_values("Technologies").unique().to_list()))
    tech_list = production_tech_list + resources_techs
    Technologies_Parameters = Technologies_Parameters[Technologies_Parameters.columns[:3].to_list() + tech_list]
    Technologies_Parameters_other = Technologies_Parameters_other[
        Technologies_Parameters.columns[:3].to_list() + tech_list]

    Tech_param = Technologies_Parameters
    Tech_param = Tech_param.reset_index().melt(id_vars=["Resource"], value_vars=Tech_param.columns,
                                               var_name="Technologies", value_name="Flow")
    Tech_param = Tech_param[~Tech_param.Technologies.isin(["Zero", "unit"])].set_index(
        ["Technologies", "Resource"])
    ###############
    # Sets       ##
    ###############
    TECHNOLOGIES = set(tech_list)
    RESOURCES_TECHS = set(resources_techs)
    PRODUCTION_TECHS=set(production_tech_list)
    RESOURCES = set(Technologies_Parameters.index.get_level_values("Resource").unique().tolist())
    ENERGY_RESOURCES = set(energy_resources_list)
    PRODUCED_RESOURCES = set(Production.index.get_level_values("Resource").unique())
    INTERMEDIARY_RESOURCES=set(intermediary_resources)


    model.TECHNOLOGIES = Set(initialize=TECHNOLOGIES, ordered=False)
    model.RESOURCES_TECHS = Set(initialize=RESOURCES_TECHS, ordered=False)
    model.PRODUCTION_TECHS=Set(initialize= PRODUCTION_TECHS,ordered=False)
    model.RESOURCES = Set(initialize=RESOURCES, ordered=False)
    model.ENERGY_RESOURCES = Set(initialize=ENERGY_RESOURCES, ordered=False)
    model.PRODUCED_RESOURCES = Set(initialize=PRODUCED_RESOURCES, ordered=False)
    model.INTERMEDIARY_RESOURCES=Set(initialize=INTERMEDIARY_RESOURCES,ordered=False)
    ###############
    # Parameters ##
    ###############
    model.P_emissions = Param(model.TECHNOLOGIES, default=0, initialize=Technologies_Parameters_other.loc[
        "Emissions", Technologies_Parameters_other.columns[3:]].to_frame().squeeze().to_dict(), domain=Reals)
    model.P_flow_cost = Param(model.TECHNOLOGIES, default=0, initialize=Technologies_Parameters_other.loc[
        "flow_cost", Technologies_Parameters_other.columns[3:]].to_frame().squeeze().to_dict(), domain=Reals)
    model.P_capex = Param(model.TECHNOLOGIES, default=0, initialize=Technologies_Parameters_other.loc[
        "capex", Technologies_Parameters_other.columns[3:]].to_frame().squeeze().to_dict(), domain=NonNegativeReals)
    model.P_CRF = Param(model.TECHNOLOGIES, default=0, initialize=Technologies_Parameters_other.loc[
        "CRF", Technologies_Parameters_other.columns[3:]].to_frame().squeeze().to_dict(), domain=NonNegativeReals)

    model.P_tech_flows = Param(model.TECHNOLOGIES, model.RESOURCES, default=0,
                               initialize=Tech_param.squeeze().to_dict())
    model.P_equality_flow = Param(model.RESOURCES, default=0,
                                  initialize=Technologies_Parameters["Zero"].squeeze().to_dict(), domain=Binary)

    model.P_production = Param(model.PRODUCED_RESOURCES, default=0,
                               initialize=Production["Production"].squeeze().to_dict(), within=Any)
    model.P_productionCtr = Param(model.PRODUCED_RESOURCES, default=0,
                                  initialize=Production["Constraint"].squeeze().to_dict(), within=Any)
    model.P_productionmargin = Param(model.PRODUCED_RESOURCES, default=0,
                                     initialize=Production["Margin"].squeeze().to_dict(), within=Any)

    model.P_forced_prod_ratio = Param(model.TECHNOLOGIES, default=0,
                                      initialize=Available_Technologies["Forced_prod_ratio"].squeeze().to_dict(),
                                      within=Any)
    model.P_forced_resource = Param(model.TECHNOLOGIES, default=0,
                                    initialize=Available_Technologies["Forced_resource"].squeeze().to_dict(),
                                    within=Any)
    model.P_max_capacity = Param(model.TECHNOLOGIES, default=0,
                                 initialize=Available_Technologies["Max_capacity_t"].squeeze().to_dict(),
                                 domain=NonNegativeReals)
    model.P_max_capacity_resource = Param(model.TECHNOLOGIES, default=0, initialize=Available_Technologies[
        "Max_capacity_resource"].squeeze().to_dict(), within=Any)

    ################
    # Variables    #
    ################
    model.V_cost = Var(domain=NonNegativeReals)
    model.V_emissions = Var(domain=Reals)
    model.V_resources_flow = Var(model.RESOURCES, domain=Reals)
    model.V_technology_usage = Var(model.TECHNOLOGIES, domain=NonNegativeReals)

    model.V_plusflow=Var(model.ENERGY_RESOURCES, domain=NonNegativeReals)
    model.V_minusflow = Var(model.ENERGY_RESOURCES, domain=NegativeReals)
    ########################
    # Objective Function   #
    ########################

    def Objective_rule(model):
        if opti2mini == "cost":
            return model.V_cost
        elif opti2mini == "emissions":  # TODO: does not work for now as biogas emissions are negative ==> infinite minimization of emissions
            return model.V_emissions
        # elif opti2mini=="production": #TODO: does not work for now as the sum is negative thus minimization lasts forever
        #     return sum(model.V_technology_usage[tech]*model.P_tech_flows[tech,resource] for resource in model.RESOURCES for tech in model.TECHNOLOGIES)

        else:
            return model.V_cost

    model.OBJ = Objective(rule=Objective_rule, sense=minimize)

    #################
    # Constraints   #
    #################
    def Cost_rule(model):
        return model.V_cost == sum(model.P_flow_cost[tech] * model.V_technology_usage[tech] for tech in model.TECHNOLOGIES)+\
               sum(model.P_capex[tech]*model.P_CRF[tech] * model.V_technology_usage[tech] for tech in model.TECHNOLOGIES)+\
                carbon_tax*model.V_emissions

    model.CostCtr = Constraint(rule=Cost_rule)


    def Emissions_rule(model):
        return model.V_emissions == sum(
            model.P_emissions[tech] * model.V_technology_usage[tech] for tech in model.TECHNOLOGIES)

    model.EmissionsCtr = Constraint(rule=Emissions_rule)

    def Production_rule(model, produced_resource):
        # if model.P_productionCtr[produced_resource]=="equal":
        #     return sum(-model.P_tech_flows[tech,produced_resource]*model.V_technology_usage[tech] for tech in model.TECHNOLOGIES) ==model.P_production[produced_resource]
        if model.P_productionCtr[produced_resource] == "inf":
            return sum(-model.P_tech_flows[tech, produced_resource] * model.V_technology_usage[tech] for tech in
                       model.TECHNOLOGIES) <= model.P_production[produced_resource]
        elif model.P_productionCtr[produced_resource] == "sup":
            return sum(-model.P_tech_flows[tech, produced_resource] * model.V_technology_usage[tech] for tech in
                       model.TECHNOLOGIES) >= model.P_production[produced_resource]
        else:
            return Constraint.Skip

    model.ProductionsupinfCtr = Constraint(model.PRODUCED_RESOURCES, rule=Production_rule)

    def Production_equalsup_rule(model, produced_resource):
        if model.P_productionCtr[produced_resource] == "equal":
            return sum(-model.P_tech_flows[tech, produced_resource] * model.V_technology_usage[tech] for tech in
                       model.TECHNOLOGIES) \
                   <= model.P_production[produced_resource] * (1 + model.P_productionmargin[produced_resource])
        else:
            return Constraint.Skip

    model.ProductionequalsupCtr = Constraint(model.PRODUCED_RESOURCES, rule=Production_equalsup_rule)

    def Production_equalinf_rule(model, produced_resource):
        if model.P_productionCtr[produced_resource] == "equal":
            return sum(-model.P_tech_flows[tech, produced_resource] * model.V_technology_usage[tech] for tech in
                       model.TECHNOLOGIES) \
                   >= model.P_production[produced_resource] * (1 - model.P_productionmargin[produced_resource])
        else:
            return Constraint.Skip

    model.ProductionequalinfCtr = Constraint(model.PRODUCED_RESOURCES, rule=Production_equalinf_rule)

    def Flow_rule(model, resource):
        return sum(
            model.P_tech_flows[tech, resource] * model.V_technology_usage[tech] for tech in model.TECHNOLOGIES) == \
               model.V_resources_flow[resource]

    model.FlowCtr = Constraint(model.RESOURCES, rule=Flow_rule)

    def Forced_prod_rule(model, tech):
        if model.P_forced_prod_ratio[tech] > 0:
            return -model.P_tech_flows[tech, model.P_forced_resource[tech]] * model.V_technology_usage[tech] == \
                   model.P_forced_prod_ratio[tech] * model.P_production[model.P_forced_resource[tech]]
        else:
            return Constraint.Skip

    model.Forced_prodCtr = Constraint(model.TECHNOLOGIES, rule=Forced_prod_rule)

    def Max_capacity_rule(model, tech):
        if model.P_max_capacity[tech] > 0:
            return -model.P_tech_flows[tech, model.P_max_capacity_resource[tech]] * model.V_technology_usage[tech] <= \
                   model.P_max_capacity[tech]
        else:
            return Constraint.Skip

    model.Max_capacityCtr = Constraint(model.TECHNOLOGIES, rule=Max_capacity_rule)

    # def Resource_flow_rule(model, energy_resource):
    #     return model.V_plusflow[energy_resource]+model.V_minusflow[energy_resource] <= 0
    # model.Resource_flowCtr = Constraint(model.ENERGY_RESOURCES, rule=Resource_flow_rule)

    def Intermediary_resources_rule(model,inter_resource):
        return sum(model.P_tech_flows[tech, inter_resource] * model.V_technology_usage[tech] for tech in model.TECHNOLOGIES) <=0
    model.Intermediary_resourcesCtr=Constraint(model.INTERMEDIARY_RESOURCES,rule=Intermediary_resources_rule)

    def Resource_required_4_prod_rule(model,energy_resource):
        return model.V_plusflow[energy_resource]+model.V_minusflow[energy_resource]==sum(model.P_tech_flows[tech, energy_resource] * model.V_technology_usage[tech] for tech in
                       model.PRODUCTION_TECHS)
    model.Resource_for_prodCtr=Constraint(model.ENERGY_RESOURCES,rule=Resource_required_4_prod_rule)

    def Resource_production_rule(model,energy_resource):
        return model.V_plusflow[energy_resource]==-1*sum(model.P_tech_flows[tech, energy_resource] * model.V_technology_usage[tech] for tech in
                       model.RESOURCES_TECHS)
    model.Resource_productionCtr=Constraint(model.ENERGY_RESOURCES,rule=Resource_production_rule)


    def Equilibrium_flow_rule(model, resource):
        if model.P_equality_flow[resource] == True:
            return sum(-model.P_tech_flows[tech, resource] * model.V_technology_usage[tech] for tech in
                       model.TECHNOLOGIES) == 0
        else:
            return Constraint.Skip

    model.Equilibrium_flowCtr = Constraint(model.RESOURCES, rule=Equilibrium_flow_rule)




    opt = SolverFactory('mosek')

    results = opt.solve(model)

    ######################
    # Results treatment  #
    ######################
    print("Print values for all variables")
    for v in model.component_data_objects(Var):
        if v.name[:6]!="V_plus" and v.name[:7]!="V_minus" and v.name[:16]!="V_resources_flow":
            print(v, v.value)

main(Technologies_Parameters,Available_Technologies,Production,intermediary_resources=['ore_pellet','scrap_steel'],intermediary_techs=["Scrap","Pellet_making"],opti2mini="cost",carbon_tax=0)
