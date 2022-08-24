from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.core import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def GetIndustryModel(Parameters,opti2mini="cost"):
    model = ConcreteModel()

    #####################
    # Data preparation ##
    #####################

    TECHNOLOGIES = set(Parameters["TECHNOLOGIES_TECH_TYPE_parameters"].index.get_level_values('TECHNOLOGIES').unique())
    TECH_TYPE = set(Parameters["TECHNOLOGIES_TECH_TYPE_parameters"].index.get_level_values('TECH_TYPE').unique())
    RESOURCES = set(Parameters["RESOURCES_parameters"].index.get_level_values('RESOURCES').unique())
    YEAR=set(Parameters["RESOURCES_parameters"].index.get_level_values('YEAR').unique())
    # print(Parameters["RESOURCES_parameters"].index.get_level_values('RESOURCES').unique())
    # print(Parameters["TECHNOLOGIES_TECH_TYPE_parameters"].index.get_level_values('TECHNOLOGIES').unique())

    ########
    # SETS #
    ########
    model.TECHNOLOGIES = Set(initialize=TECHNOLOGIES, ordered=False)
    model.RESOURCES = Set(initialize=RESOURCES, ordered=False)
    model.TECHNOLOGIES_RESOURCES = model.TECHNOLOGIES * model.RESOURCES
    model.YEAR=Set(initialize=YEAR,ordered=False)
    model.TECH_TYPE=Set(initialize=TECH_TYPE,ordered=False)
    model.TECHNOLOGIES_TECH_TYPE_RESOURCES_YEAR = model.TECHNOLOGIES * model.TECH_TYPE * model.RESOURCES * model.YEAR
    ###############
    # Parameters ##
    ###############

    
    for COLNAME in Parameters["TECHNOLOGIES_TECH_TYPE_parameters"]:
        exec("model.P_" + COLNAME + " =  Param(model.TECHNOLOGIES,model.TECH_TYPE,model.YEAR, mutable=False, domain=Any,default=0," +
                 "initialize=Parameters[\"TECHNOLOGIES_TECH_TYPE_parameters\"]." + COLNAME + ".squeeze().to_dict())")

    for COLNAME in Parameters["RESOURCES_parameters"]:
        exec("model.P_" + COLNAME + " =  Param(model.RESOURCES,model.YEAR, mutable=False, domain=Any,default=0," +
                 "initialize=Parameters[\"RESOURCES_parameters\"]." + COLNAME + ".squeeze().to_dict())")

    for COLNAME in Parameters["TECHNOLOGIES_RESOURCES_parameters"]:
        exec("model.P_" + COLNAME + " =  Param(model.TECHNOLOGIES,model.TECH_TYPE,model.RESOURCES,model.YEAR, mutable=False, domain=Any,default=0," +
                 "initialize=Parameters[\"TECHNOLOGIES_RESOURCES_parameters\"]." + COLNAME + ".squeeze().to_dict())")

    for COLNAME in Parameters["TECHNOLOGIES_parameters"]:
        exec("model.P_" + COLNAME + " =  Param(model.TECHNOLOGIES,model.YEAR, mutable=False, domain=Any,default=0," +
                 "initialize=Parameters[\"TECHNOLOGIES_parameters\"]." + COLNAME + ".squeeze().to_dict())")

    ################
    # Variables    #
    ################
    model.V_cost_total=Var(domain=NonNegativeReals)
    model.V_emissions_total=Var(domain=Reals)
    model.V_cost = Var(model.YEAR,domain=NonNegativeReals)
    model.V_technology_cost=Var(model.TECHNOLOGIES,model.YEAR,domain=NonNegativeReals)
    model.V_emissions = Var(model.YEAR,domain=Reals)
    model.V_emissions_tech=Var(model.TECHNOLOGIES,model.TECH_TYPE,model.YEAR,domain=Reals)
    model.V_emissions_tech_plus=Var(model.TECHNOLOGIES,model.TECH_TYPE,model.YEAR,domain=PositiveReals)
    model.V_emissions_tech_minus=Var(model.TECHNOLOGIES,model.TECH_TYPE,model.YEAR,domain=NegativeReals)
    model.V_resource_flow = Var(model.RESOURCES,model.YEAR, domain=Reals)
    model.V_resource_inflow = Var(model.RESOURCES,model.YEAR, domain=PositiveReals)
    model.V_resource_outflow = Var(model.RESOURCES,model.YEAR, domain=PositiveReals)
    model.V_resource_tech_inflow = Var(model.TECHNOLOGIES,model.TECH_TYPE,model.RESOURCES,model.YEAR, domain=PositiveReals)
    model.V_resource_tech_outflow = Var(model.TECHNOLOGIES,model.TECH_TYPE,model.RESOURCES,model.YEAR, domain=PositiveReals)

    model.V_technology_use_coef=Var(model.TECHNOLOGIES,model.TECH_TYPE,model.YEAR,domain=PositiveReals)

    model.V_technology_use_coef_capacity = Var(model.TECHNOLOGIES, model.YEAR, domain=PositiveReals)
    model.V_technology_tech_type_use_coef_capacity=Var(model.TECHNOLOGIES,model.TECH_TYPE,model.YEAR,domain=PositiveReals)
    model.V_added_capacity=Var(model.TECHNOLOGIES,model.YEAR,domain=PositiveReals)

    ########################
    # Objective Function   #
    ########################

    def Objective_rule(model):
        if opti2mini == "cost":
            return model.V_cost_total ### see the constraint below
        elif opti2mini == "emissions":
            return model.V_emissions_total ### see the constraint below
        else:
            return model.V_cost ### see the constraint below
    model.OBJ = Objective(rule=Objective_rule, sense=minimize)

    ############################################
    # Cost and emission variables definition   #
    ############################################
    def Total_cost_definition_rule(model):
        return model.V_cost_total==sum(model.V_cost[year] for year in model.YEAR)
    model.Total_cost_definitionCtr=Constraint(rule=Total_cost_definition_rule)
    def Total_emissions_definition_rule(model):
        return model.V_emissions_total==sum(model.V_emissions[year] for year in model.YEAR)
    model.Total_emissions_definitionCtr=Constraint(rule=Total_emissions_definition_rule)

    def Cost_definition_rule(model,year):
        return model.V_cost[year] == sum(model.P_flow_cost_r[resource,year]*model.V_resource_flow[resource,year]*(1-model.P_is_product[resource,year]) for resource in model.RESOURCES) +\
               sum(model.P_flow_cost_t[tech,year] * model.V_technology_use_coef[tech,tech_type,year] for tech in model.TECHNOLOGIES for tech_type in model.TECH_TYPE) + \
               sum(model.V_technology_cost[tech,year] for tech in model.TECHNOLOGIES)+\
               sum(model.P_carbon_cost[tech,year] * model.V_emissions_tech_plus[tech,tech_type,year] for tech in model.TECHNOLOGIES for tech_type in model.TECH_TYPE)

    model.Cost_definitionCtr = Constraint(model.YEAR,rule=Cost_definition_rule)


    def Technology_cost_definition_rule(model,tech,year):
        Tot=0
        for tech_type in model.TECH_TYPE:
            if model.P_lifetime[tech,year]!=0 or model.P_construction_time[tech, year]!=0:
                n=model.P_construction_time[tech, year]+ model.P_lifetime[tech, year]
                i=model.P_discount_rate[tech,year]
                CRF=i*(i+1)**n/((1+i)**n-1)
                Tot+=model.P_capex[tech, year]*CRF*model.V_technology_tech_type_use_coef_capacity[tech, tech_type, year]
        return model.V_technology_cost[tech,year]==Tot

    model.Technology_cost_definitionCtr=Constraint(model.TECHNOLOGIES,model.YEAR,rule=Technology_cost_definition_rule)

    def Emissions_definition_rule(model,year):
        return model.V_emissions[year] == sum(model.V_emissions_tech[tech,tech_type,year] for tech in model.TECHNOLOGIES for tech_type in model.TECH_TYPE)+ \
               sum(model.P_emissions_r[resource,year] * model.V_resource_flow[resource,year] for resource in model.RESOURCES)
    model.Emissions_definitionCtr = Constraint(model.YEAR,rule=Emissions_definition_rule)


    def Technology_emissions_definition_1st_rule(model,tech,tech_type,year):
        return model.V_emissions_tech[tech,tech_type,year]==model.V_emissions_tech_plus[tech,tech_type,year]+model.V_emissions_tech_minus[tech,tech_type,year]
    model.Technology_emissions_definition_1stCtr=Constraint(model.TECHNOLOGIES,model.TECH_TYPE,model.YEAR,rule=Technology_emissions_definition_1st_rule)

    def Technology_emissions_definition_2nd_rule(model,tech,tech_type,year):
        return model.V_emissions_tech[tech,tech_type,year]==model.P_emissions_t[tech,tech_type,year] * model.V_technology_use_coef[tech,tech_type,year]

    model.Technology_emissions_definition_2ndCtr = Constraint(model.TECHNOLOGIES,model.TECH_TYPE, model.YEAR,
                                                              rule=Technology_emissions_definition_2nd_rule)


    #################
    # Constraints   #
    #################



    #decomposition (+/-) of resource flow
    def resource_flow_definition_1st_rule(model,resource,year):
        return model.V_resource_flow[resource,year]==model.V_resource_inflow[resource,year]-model.V_resource_outflow[resource,year]
    model.resource_flow_definition_1stCtr=Constraint(model.RESOURCES,model.YEAR,rule=resource_flow_definition_1st_rule)

    #decomposition (+/-) of tech flow
    def resource_flow_definition_2nd_rule(model,tech,tech_type,resource,year):
        return model.V_resource_tech_inflow[tech,tech_type,resource,year]-model.V_resource_tech_outflow[tech,tech_type,resource,year]==model.V_technology_use_coef[tech,tech_type,year]*model.P_conversion_factor[tech,tech_type,resource,year]
    model.resource_flow_definition_2ndCtr = Constraint(model.TECHNOLOGIES,model.TECH_TYPE,model.RESOURCES, model.YEAR,rule=resource_flow_definition_2nd_rule)

    def resource_flow_tech_rule(model,tech,tech_type,resource,year):
        if model.P_conversion_factor[tech,tech_type,resource,year]>0:
            return model.V_resource_tech_outflow[tech,tech_type,resource,year]==0
        else:
            return model.V_resource_tech_inflow[tech,tech_type, resource,year] == 0
    model.resource_flow_techCtr=Constraint(model.TECHNOLOGIES,model.TECH_TYPE,model.RESOURCES,model.YEAR,rule=resource_flow_tech_rule)

    def resource_flow_definition_3rd_rule(model,resource,year):
        return model.V_resource_inflow[resource,year]==sum(model.V_resource_tech_inflow[tech,tech_type,resource,year] for tech in model.TECHNOLOGIES for tech_type in model.TECH_TYPE)
    model.resource_flow_definition_3rdCtr = Constraint(model.RESOURCES,model.YEAR, rule=resource_flow_definition_3rd_rule)

    def resource_flow_definition_4th_rule(model,resource,year):
        return model.V_resource_outflow[resource,year]==sum(model.V_resource_tech_outflow[tech,tech_type,resource,year] for tech in model.TECHNOLOGIES for tech_type in model.TECH_TYPE)
    model.resource_flow_definition_4thCtr = Constraint(model.RESOURCES,model.YEAR, rule=resource_flow_definition_4th_rule)

    ###Production Constraints###
    def Production_moins_rule(model,resource,year):
        if model.P_output[resource,year]!=0:
            return model.P_output[resource,year]*(1+model.P_production_error_margin[resource,year])>=model.V_resource_outflow[resource,year]
        else:
            return Constraint.Skip

    def Production_plus_rule(model,resource,year):
        if model.P_output[resource,year]!=0:
            return model.P_output[resource,year]*(1-model.P_production_error_margin[resource,year])<=model.V_resource_outflow[resource,year]
        else:
            return Constraint.Skip

    model.Production_moinsCtr=Constraint(model.RESOURCES,model.YEAR,rule=Production_moins_rule)
    model.Production_plusCtr = Constraint(model.RESOURCES,model.YEAR, rule=Production_plus_rule)

    def Resource_flow_rule(model,resource,year):
        if model.P_is_product[resource,year]==0:
            return model.V_resource_flow[resource,year]>=0
        else:
            return Constraint.Skip
    model.Resource_flowCtr=Constraint(model.RESOURCES,model.YEAR,rule=Resource_flow_rule)

    def Technology_Production_Min_rule(model,tech,tech_type,year):
        if model.P_forced_prod_ratio_min[tech,tech_type,year]!=0:
            resource = model.P_forced_resource[tech,tech_type,year]
            return model.P_forced_prod_ratio_min[tech,tech_type,year]*model.V_resource_outflow[resource,year]<=-model.V_technology_use_coef[tech,tech_type,year]*model.P_conversion_factor[tech,tech_type,resource,year]
        else:
            return Constraint.Skip

    model.Technology_Production_MinCtr=Constraint(model.TECHNOLOGIES,model.TECH_TYPE,model.YEAR,rule=Technology_Production_Min_rule)

    def Technology_Production_Max_rule(model,tech,tech_type,year):
        if model.P_forced_prod_ratio_max[tech,tech_type,year]!=0:
            resource = model.P_forced_resource[tech,tech_type,year]
            return model.P_forced_prod_ratio_max[tech,tech_type,year]*model.V_resource_outflow[resource,year]>=-model.V_technology_use_coef[tech,tech_type,year]*model.P_conversion_factor[tech,tech_type,resource,year]
        else:
            return Constraint.Skip

    model.Technology_Production_MaxCtr=Constraint(model.TECHNOLOGIES,model.TECH_TYPE,model.YEAR,rule=Technology_Production_Max_rule)


    def Technology_Capacity_rule(model,tech,tech_type,year):
        if model.P_max_capacity_t[tech,year]>0:
            resource = model.P_capacity_associated_resource[tech,year]
            return sum(model.V_resource_tech_outflow[tech,tech_type,resource,year] for tech_type in model.TECH_TYPE)<=model.P_max_capacity_t[tech,year]
        else:
            return Constraint.Skip
    model.Technology_CapacityCtr=Constraint(model.TECHNOLOGIES,model.TECH_TYPE,model.YEAR,rule=Technology_Capacity_rule)

    def Added_capacity_rule(model,tech,year):
        Tot=0
        lifetime=model.P_lifetime[tech,year]

        for y in model.YEAR:
            if y<=year:
                past_year=y
                year_ini = model.P_year_ini[tech, y]
                if year_ini!=0:
                    past_year=year_ini
                if year-past_year<=max(lifetime-1,0):
                    Tot+=model.V_added_capacity[tech,y]

        return model.V_technology_use_coef_capacity[tech,year]==Tot
    model.Added_capacity_Ctr=Constraint(model.TECHNOLOGIES,model.YEAR,rule=Added_capacity_rule)
    def Tech_use_coef_1st_rule(model,tech,tech_type,year):
        return model.V_technology_tech_type_use_coef_capacity[tech, tech_type, year]>=model.V_technology_use_coef[tech,tech_type,year]
    model.Tech_use_coef_1stCtr=Constraint(model.TECHNOLOGIES,model.TECH_TYPE,model.YEAR,rule=Tech_use_coef_1st_rule)
    def Tech_use_coef_2nd_rule(model,tech,year):
        return sum(model.V_technology_tech_type_use_coef_capacity[tech, tech_type,year] for tech_type in model.TECH_TYPE)==model.V_technology_use_coef_capacity[tech,year]

    model.Tech_use_coef_2ndCtr = Constraint(model.TECHNOLOGIES, model.YEAR,
                                            rule=Tech_use_coef_2nd_rule)

    def Tech_use_coef_3rd_rule(model,tech,year):
        if model.P_max_capacity_t[tech,year]!=0:
            resource = model.P_capacity_associated_resource[tech, year]
            return sum(model.V_technology_tech_type_use_coef_capacity[tech, tech_type,year]*-model.P_conversion_factor[tech,tech_type,resource,year] for tech_type in model.TECH_TYPE)<=model.P_max_capacity_t[tech,year]
        else:
            return Constraint.Skip
    model.Tech_use_coef_3rdCtr = Constraint(model.TECHNOLOGIES, model.YEAR, rule=Tech_use_coef_3rd_rule)



    return model

