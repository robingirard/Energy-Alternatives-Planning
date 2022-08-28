from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.core import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def optim_industry(Resources_Technologies,Resource_Availability,Production_Technologies,Available_Technologies,Production,opti2mini="cost",carbon_tax=0):
    model = ConcreteModel()

    #####################
    # Data preparation ##
    #####################

    Resources_Technologies_other = Resources_Technologies[
        Resources_Technologies.Resource.isin(["Emissions", "flow_cost", "capex", "CRF"])].set_index('Resource').drop(
        "unit", axis=1)
    Resources_Technologies = Resources_Technologies[
        ~Resources_Technologies.Resource.isin(
            ["Emissions", "flow_cost", "capex", "lifetime", "discount_rate", "CRF"])].set_index('Resource').drop("unit",
                                                                                                                 axis=1)
    Production_Technologies_other = Production_Technologies[
        Production_Technologies.Resource.isin(["Emissions", "flow_cost", "capex", "CRF"])].set_index('Resource')
    Production_Technologies = Production_Technologies[
        ~Production_Technologies.Resource.isin(
            ["Emissions", "flow_cost", "capex", "lifetime", "discount_rate", "CRF"])].set_index('Resource')

    Available_Technologies = Available_Technologies.set_index('Technologies')
    Production = Production.set_index('Resource')
    resource_tech_list = list(dict.fromkeys(Resources_Technologies.columns.to_list()))
    resource_tech_list = list(dict.fromkeys(resource_tech_list))

    production_tech_list = list(
        dict.fromkeys(Available_Technologies.index.get_level_values("Technologies").unique().to_list()))
    production_tech_list = list(dict.fromkeys(production_tech_list))
    production_tech_list_copy = production_tech_list.copy()
    for tech in resource_tech_list:
        if tech in production_tech_list_copy:
            production_tech_list.remove(tech)

    tech_list = production_tech_list + resource_tech_list
    tech_list = list(dict.fromkeys(tech_list))

    resource_list = Production_Technologies.index.get_level_values("Resource").unique().tolist()
    primary_resource_list = Resources_Technologies.index.get_level_values("Resource").unique().tolist()

    Technologies_Parameters = pd.concat([Resources_Technologies, Production_Technologies[production_tech_list]],
                                        axis=1).fillna(0)
    Technologies_Parameters_other = pd.concat(
        [Resources_Technologies_other, Production_Technologies_other[production_tech_list]], axis=1).fillna(0)
    ###Preparation for P_resource_prod and P_production_error_margin parameters
    if len(Production)==1:
        production_dict = Production["Production"].to_dict()
    else:
        production_dict = Production["Production"].squeeze().to_dict()

    products_output_list = list(production_dict.keys())
    u = products_output_list.copy()
    ##add forgotten products to the production list (if chosen technologies have other outputs [e.g for ethylene production, naphtha cracking produces also propylene and other chemicals])
    for resource in resource_list:
        for tech in production_tech_list:
            for product in u:
                flow_given_product_val = Technologies_Parameters.loc[product, tech]
                flow_val = Technologies_Parameters.loc[resource, tech]
                if flow_val < 0 and flow_given_product_val < 0:
                    products_output_list.append(resource)
    products_output_list = list(dict.fromkeys(products_output_list))
    products_output_state_dict = {}
    for product in products_output_list:
        products_output_state_dict[product] = 1

    ###Error margin for fixed production
    if len(Production) == 1:
        error_margin_dict = Production["Margin"].to_dict()
    else:
        error_margin_dict = Production["Margin"].squeeze().to_dict()

    ###Preparation for P_tech_resource_flow_coef_min and max parameters
    tech_resource_flow_min_dict=Available_Technologies.reset_index().set_index(['Technologies','Forced_resource'])['Min_prod_ratio'].squeeze().to_dict()
    tech_resource_flow_max_dict=Available_Technologies.reset_index().set_index(['Technologies', 'Forced_resource'])['Max_prod_ratio'].squeeze().to_dict()

    ###Preparation for P_tech_resource_capacity parameter
    tech_resource_capacity_dict = Resource_Availability.reset_index().set_index(['Technologies', 'Forced_resource'])['Max_capacity'].squeeze().to_dict()

    ###Preparation for P_tech_flows parameter
    Tech_param = Technologies_Parameters
    Tech_param = Tech_param.reset_index().melt(id_vars=["Resource"], value_vars=Tech_param.columns,
                                               var_name="Technologies", value_name="Flow")
    Tech_param = Tech_param[~Tech_param.Technologies.isin(["unit"])].set_index(
        ["Technologies", "Resource"])

    ###############
    # Sets       ##
    ###############
    TECHNOLOGIES = set(tech_list)
    RESOURCE_TECHS = set(resource_tech_list)
    RESOURCES = set(resource_list)
    PRIMARY_RESOURCES = set(primary_resource_list)

    model.TECHNOLOGIES = Set(initialize=TECHNOLOGIES, ordered=False)
    model.PRIMARY_RESOURCE_TECHS = Set(initialize=RESOURCE_TECHS, ordered=False)
    model.RESOURCES = Set(initialize=RESOURCES, ordered=False)
    model.PRIMARY_RESOURCES = Set(initialize=PRIMARY_RESOURCES, ordered=False)

    ###############
    # Parameters ##
    ###############

    model.P_emissions = Param(model.TECHNOLOGIES, default=0, initialize=Technologies_Parameters_other.loc[
        "Emissions", Technologies_Parameters_other.columns].to_frame().squeeze().to_dict(), domain=Reals)
    model.P_flow_cost = Param(model.TECHNOLOGIES, default=0, initialize=Technologies_Parameters_other.loc[
        "flow_cost", Technologies_Parameters_other.columns].to_frame().squeeze().to_dict(), domain=Reals)
    model.P_capex = Param(model.TECHNOLOGIES, default=0, initialize=Technologies_Parameters_other.loc[
        "capex", Technologies_Parameters_other.columns].to_frame().squeeze().to_dict(), domain=NonNegativeReals)
    model.P_CRF = Param(model.TECHNOLOGIES, default=0, initialize=Technologies_Parameters_other.loc[
        "CRF", Technologies_Parameters_other.columns].to_frame().squeeze().to_dict(), domain=NonNegativeReals)
    model.P_carbon_tax = carbon_tax

    model.P_products_boolean = Param(model.RESOURCES, default=0, initialize=products_output_state_dict)
    d={}
    for res in model.RESOURCES:
        d[res]=model.P_products_boolean[res]
    print('P_products_boolean')
    print(d)
    model.P_resource_prod = Param(model.RESOURCES, default=0, initialize=production_dict, within=Any)
    d = {}
    for res in model.RESOURCES:
        d[res] = model.P_resource_prod[res]
    print('P_resource_prod')
    print(d)
    model.P_production_error_margin = Param(model.RESOURCES, default=0,
                                            initialize=error_margin_dict, within=Any)
    d = {}
    for res in model.RESOURCES:
        d[res] = model.P_production_error_margin[res]
    print('P_production_error_margin')
    print(d)
    model.P_tech_resource_flow_coef_min = Param(model.TECHNOLOGIES, model.RESOURCES, default=0,
                                            initialize=tech_resource_flow_min_dict,
                                            within=Any)  # min % of total resource flow for a given tech, e.g. we want minimum 60% of the steel coming from recycling technologies
    #mutable=True avoids a strange error due to the fact that model.P_tech_resource_flow_coef_min[tech,resource]=0 for all resource for a given tech
    d = {}
    d['Technologies']=model.TECHNOLOGIES
    for res in model.RESOURCES:
        e={}
        for tech in model.TECHNOLOGIES:
            e[tech] = model.P_tech_resource_flow_coef_min[tech,res]
        d[res]=e
    print('P_tech_resource_flow_coef_min')
    print(d)
    model.P_tech_resource_flow_coef_max = Param(model.TECHNOLOGIES, model.RESOURCES, default=1,
                                                initialize=tech_resource_flow_max_dict,
                                                within=Any) # max % of total resource flow for a given tech, e.g. we want maximum 60% of the steel coming from recycling technologies
    d = {}
    d['Technologies'] = model.TECHNOLOGIES
    for res in model.RESOURCES:
        e = {}
        for tech in model.TECHNOLOGIES:
            e[tech] = model.P_tech_resource_flow_coef_max[tech, res]
        d[res] = e
    print('P_tech_resource_flow_coef_max')
    print(d)
    model.P_tech_resource_capacity = Param(model.TECHNOLOGIES, model.RESOURCES, default=0,
                                           initialize=tech_resource_capacity_dict,
                                           within=Any)  # production capacity of a technology for an associated resource
    d = {}
    d['Technologies'] = model.TECHNOLOGIES
    for res in model.RESOURCES:
        e = {}
        for tech in model.TECHNOLOGIES:
            e[tech] = model.P_tech_resource_capacity[tech, res]
        d[res] = e
    print('P_tech_resource_capacity')
    print(d)
    model.P_tech_flows = Param(model.TECHNOLOGIES, model.RESOURCES, default=0,
                               initialize=Tech_param.squeeze().to_dict())
    d = {}
    d['Technologies'] = model.TECHNOLOGIES
    for res in model.RESOURCES:
        e = {}
        for tech in model.TECHNOLOGIES:
            e[tech] = model.P_tech_flows[tech, res]
        d[res] = e
    print('P_tech_flows')
    print(d)
    #Params = {}
    #for p in model.component_data_objects(Param):
        #print(p)

    ################
    # Variables    #
    ################
    model.V_cost = Var(domain=NonNegativeReals)
    model.V_emissions = Var(domain=Reals)
    model.V_emissions_plus = Var(domain=NonNegativeReals)
    model.V_emissions_minus = Var(domain=NonPositiveReals)
    model.V_resource_flow = Var(model.RESOURCES, domain=NonPositiveReals)
    model.V_resource_inflow = Var(model.RESOURCES, domain=NonNegativeReals)
    model.V_resource_outflow = Var(model.RESOURCES, domain=NonNegativeReals)
    #model.V_primary_resource_production = Var(model.PRIMARY_RESOURCE_TECHS, model.PRIMARY_RESOURCES,
                                              #domain=PositiveReals)
    model.V_technology_use_coef = Var(model.TECHNOLOGIES, domain=NonNegativeReals)
    model.V_resource_tech_inflow = Var(model.TECHNOLOGIES, model.RESOURCES, domain=NonNegativeReals)
    model.V_resource_tech_outflow = Var(model.TECHNOLOGIES, model.RESOURCES, domain=NonNegativeReals)

    ########################
    # Objective Function   #
    ########################

    def Objective_rule(model):
        if opti2mini == "cost":
            return model.V_cost
        elif opti2mini == "emissions":
            return model.V_emissions

        else:
            return model.V_cost

    model.OBJ = Objective(rule=Objective_rule, sense=minimize)

    #################
    # Constraints   #
    #################
    def Cost_definition_rule(model):
        return model.V_cost == sum(
            model.P_flow_cost[tech] * model.V_technology_use_coef[tech] for tech in model.TECHNOLOGIES) + \
               sum(model.P_capex[tech] * model.P_CRF[tech] * model.V_technology_use_coef[tech] for tech in
                   model.TECHNOLOGIES) + \
               model.P_carbon_tax * model.V_emissions_plus

    model.Cost_definitionCtr = Constraint(rule=Cost_definition_rule)

    def Emissions_definition_rule(model):
        return model.V_emissions == sum(
            model.P_emissions[tech] * model.V_technology_use_coef[tech] for tech in model.TECHNOLOGIES)

    model.Emissions_definitionCtr = Constraint(rule=Emissions_definition_rule)

    def Emissions_definition_2nd_rule(model):
        return model.V_emissions == model.V_emissions_plus + model.V_emissions_minus

    model.Emissions_definition_2ndCtr = Constraint(rule=Emissions_definition_2nd_rule)

    def Resource_flow_definition_1st_rule(model, resource):
        return model.V_resource_flow[resource] == model.V_resource_inflow[resource] - model.V_resource_outflow[resource]

    model.Resource_flow_definition_1stCtr = Constraint(model.RESOURCES, rule=Resource_flow_definition_1st_rule)

    def Resource_flow_definition_2nd_rule(model, tech, resource):
        return model.V_resource_tech_inflow[tech, resource] - model.V_resource_tech_outflow[tech, resource] == \
               model.V_technology_use_coef[tech] * model.P_tech_flows[tech, resource]

    model.Resource_flow_definition_2ndCtr = Constraint(model.TECHNOLOGIES, model.RESOURCES,
                                                       rule=Resource_flow_definition_2nd_rule)

    def Resource_flow_tech_rule(model, tech, resource):
        if model.P_tech_flows[tech, resource] > 0:
            return model.V_resource_tech_outflow[tech, resource] == 0
        else:
            return model.V_resource_tech_inflow[tech, resource] == 0

    model.Resource_flow_techCtr = Constraint(model.TECHNOLOGIES, model.RESOURCES, rule=Resource_flow_tech_rule)

    def Resource_flow_definition_3rd_rule(model, resource):
        return model.V_resource_inflow[resource] == sum(
            model.V_resource_tech_inflow[tech, resource] for tech in model.TECHNOLOGIES)

    model.Resource_flow_definition_3rdCtr = Constraint(model.RESOURCES, rule=Resource_flow_definition_3rd_rule)

    def Resource_flow_definition_4th_rule(model, resource):
        return model.V_resource_outflow[resource] == sum(
            model.V_resource_tech_outflow[tech, resource] for tech in model.TECHNOLOGIES)

    model.Resource_flow_definition_4thCtr = Constraint(model.RESOURCES, rule=Resource_flow_definition_4th_rule)

    def Resource_flow_equilibrium_rule(model, resource):
        if model.P_products_boolean[resource] == 0:
            return model.V_resource_flow[resource] == 0
        else:
            return Constraint.Skip

    model.Resource_flow_equilibriumCtr = Constraint(model.RESOURCES, rule=Resource_flow_equilibrium_rule)

    def Primary_resources_prod_limit_rule(model, resource):# do not produce more than demand
        return model.V_resource_inflow[resource] >= sum(
            model.V_resource_tech_outflow[tech, resource] for tech in model.PRIMARY_RESOURCE_TECHS)

    model.Primary_resource_prod_limit_rule = Constraint(model.PRIMARY_RESOURCES, rule=Primary_resources_prod_limit_rule)

    ###Production Constraints###
    def Production_moins_rule(model, resource):
        if model.P_resource_prod[resource] != 0:
            return model.P_resource_prod[resource] * (1 + model.P_production_error_margin[resource]) >= \
                   model.V_resource_outflow[resource]
        else:
            return Constraint.Skip

    def Production_plus_rule(model, resource):
        if model.P_resource_prod[resource] != 0:
            return model.P_resource_prod[resource] * (1 - model.P_production_error_margin[resource]) <= \
                   model.V_resource_outflow[resource]
        else:
            return Constraint.Skip

    model.Production_moinsCtr = Constraint(model.RESOURCES, rule=Production_moins_rule)
    model.Production_plusCtr = Constraint(model.RESOURCES, rule=Production_plus_rule)

    def Technology_Production_minus_rule(model, tech, resource):
        if model.P_tech_resource_flow_coef_min[tech, resource]!=0:
            return model.P_tech_resource_flow_coef_min[tech, resource] * model.V_resource_outflow[resource]<=-model.V_technology_use_coef[tech] * model.P_tech_flows[tech, resource]
        else:
            return Constraint.Skip

    def Technology_Production_plus_rule(model, tech, resource):
        if model.P_tech_resource_flow_coef_max[tech, resource]!=1:
            return model.P_tech_resource_flow_coef_max[tech, resource] * model.V_resource_outflow[resource] >= - \
            model.V_technology_use_coef[tech] * model.P_tech_flows[tech, resource]
        else:
            return Constraint.Skip

    model.Technology_Production_minus_Ctr = Constraint(model.TECHNOLOGIES, model.RESOURCES, rule=Technology_Production_minus_rule)
    model.Technology_Production_plus_Ctr = Constraint(model.TECHNOLOGIES, model.RESOURCES,rule=Technology_Production_plus_rule)

    def Technology_Capacity_rule(model, tech, resource):
        if model.P_tech_resource_capacity[tech, resource] > 0:
            return model.V_resource_tech_outflow[tech, resource] <= model.P_tech_resource_capacity[tech, resource]
        else:
            return Constraint.Skip

    model.Technology_CapacityCtr = Constraint(model.TECHNOLOGIES, model.RESOURCES, rule=Technology_Capacity_rule)

    # Setting V_technology_use_coef
    model.V_technology_use_coef['Primary_grinding'].set_value(18200000*0.76)
    model.V_technology_use_coef['Pre-heating'].set_value(18200000*0.76)
    model.V_technology_use_coef['Kiln'].set_value(18200000 * 0.76)
    model.V_technology_use_coef['Cement_grinding'].set_value(18200000)
    d_res_tech={'Coal':'coal','Biomass':'biomass','Oil':'oil','Electricity':'electricity','Gas':'gas','Water':'water'}
    L_used_res=['Coal','Biomass','Oil','Electricity','Gas','Water']
    L_used_tech=['Primary_grinding','Pre-heating','Kiln','Cement_grinding']
    for x in L_used_res:
        model.V_technology_use_coef[x].set_value(sum(model.V_technology_use_coef[y]*model.P_tech_flows[y,d_res_tech[x]] for y in L_used_tech))
    for tech in model.TECHNOLOGIES:
        if tech not in L_used_res+L_used_tech:
            model.V_technology_use_coef[tech].set_value(0)
    # Setting V_emissions
    model.V_emissions.set_value(sum(
            model.P_emissions[tech] * model.V_technology_use_coef[tech] for tech in model.TECHNOLOGIES))
    model.V_emissions_plus.set_value(sum(
            model.P_emissions[tech] * model.V_technology_use_coef[tech] for tech in model.TECHNOLOGIES))
    model.V_emissions_minus.set_value(0)

    # Setting V_cost
    model.V_cost.set_value(sum(
            model.P_flow_cost[tech] * model.V_technology_use_coef[tech] for tech in model.TECHNOLOGIES) + \
               sum(model.P_capex[tech] * model.P_CRF[tech] * model.V_technology_use_coef[tech] for tech in
                   model.TECHNOLOGIES) + \
               model.P_carbon_tax * model.V_emissions_plus)
    # Setting V_resource_tech_inflow/outflow
    for tech in model.TECHNOLOGIES:
        for res in model.RESOURCES:
            if model.P_tech_flows[tech,res]>0:
                model.V_resource_tech_outflow[tech,res].set_value(model.V_technology_use_coef[tech]*model.P_tech_flows[tech,res])
                model.V_resource_tech_inflow[tech, res].set_value(0)
            else:
                model.V_resource_tech_inflow[tech, res].set_value(
                    -model.V_technology_use_coef[tech] * model.P_tech_flows[tech, res])
                model.V_resource_tech_outflow[tech, res].set_value(0)
    # Setting V_resource_inflow/outflow and V_resource_flow
    for res in model.RESOURCES:
        model.V_resource_outflow[res].set_value(sum(model.V_resource_tech_outflow[tech,res] for tech in model.TECHNOLOGIES))
        model.V_resource_inflow[res].set_value(sum(model.V_resource_tech_inflow[tech,res] for tech in model.TECHNOLOGIES))
        model.V_resource_flow[res].set_value(model.V_resource_inflow[res]-model.V_resource_outflow[res])
    for c in model.component_objects(ctype=Constraint):
        if c.slack() < 0:  # constraint is not met
            print(f'Constraint {c.name} is not satisfied')
            c.display()  # show the evaluation of c
            c.pprint()  # show the construction of c
            print()

    #opt = SolverFactory('mosek')

    #results = opt.solve(model)

    ######################
    # Results treatment  #
    ######################
    #Results = {}
    #for v in model.component_data_objects(Var):
        #Results[v.name] = v.value

    #return Results


def main(Resources_Technologies,Production_Technologies,Available_Technologies,Production,opti2mini="cost",carbon_tax=0):
    model = ConcreteModel()

    #####################
    # Data preparation ##
    #####################
    Resources_Technologies_other = Resources_Technologies[
        Resources_Technologies.Resource.isin(["Emissions", "flow_cost","capex","CRF"])].set_index('Resource').drop("unit",axis=1)
    Resources_Technologies = Resources_Technologies[
        ~Resources_Technologies.Resource.isin(["Emissions", "flow_cost","capex","lifetime","discount_rate","CRF"])].set_index('Resource').drop("unit",axis=1)
    Production_Technologies_other = Production_Technologies[
        Production_Technologies.Resource.isin(["Emissions", "flow_cost", "capex", "CRF"])].set_index('Resource').drop("unit",axis=1)
    Production_Technologies = Production_Technologies[
        ~Production_Technologies.Resource.isin(
            ["Emissions", "flow_cost", "capex", "lifetime", "discount_rate", "CRF"])].set_index('Resource').drop("unit",axis=1)



    Available_Technologies = Available_Technologies.set_index('Technologies')
    Production = Production.set_index('Resource')

    resource_tech_list = list(dict.fromkeys(Resources_Technologies.columns.to_list()))
    resource_tech_list = list(dict.fromkeys(resource_tech_list))
    if 'blank' in resource_tech_list:
        resource_tech_list.remove('blank')

    production_tech_list = list(dict.fromkeys(Available_Technologies.index.get_level_values("Technologies").unique().to_list()))
    production_tech_list=list(dict.fromkeys(production_tech_list))
    production_tech_list_copy=production_tech_list.copy()
    if 'blank' in production_tech_list:
        production_tech_list.remove('blank')
    for tech in resource_tech_list:
        if tech in production_tech_list_copy:
            production_tech_list.remove(tech)


    tech_list=production_tech_list+resource_tech_list
    tech_list=list(dict.fromkeys(tech_list))

    resource_list = Production_Technologies.index.get_level_values("Resource").unique().tolist()
    if "blank" in resource_list:
        resource_list.remove("blank")
    primary_resource_list = Resources_Technologies.index.get_level_values("Resource").unique().tolist()
    if "blank" in primary_resource_list:
        primary_resource_list.remove("blank")

    Technologies_Parameters = pd.concat([Resources_Technologies, Production_Technologies[production_tech_list]], axis=1).fillna(0)
    Technologies_Parameters_other= pd.concat([Resources_Technologies_other, Production_Technologies_other[production_tech_list]], axis=1).fillna(0)

    ###Preparation for P_resource_prod and P_production_error_margin parameters
    production_dict = Production["Production"].squeeze().to_dict()
    production_dict.pop('blank', None)

    products_output_list=list(production_dict.keys())
    u=products_output_list.copy()
    ##add forgotten products to the production list (if chosen technologies have other outputs [e.g for ethylene production, naphtha cracking produces also propylene and other chemicals])
    for resource in resource_list:
        for tech in production_tech_list:
            for product in u:
                flow_given_product_val=Technologies_Parameters.loc[product,tech]
                flow_val=Technologies_Parameters.loc[resource,tech]
                if flow_val<0 and flow_given_product_val<0:
                    products_output_list.append(resource)
    products_output_list=list(dict.fromkeys(products_output_list))
    products_output_state_dict={}
    for product in products_output_list:
        products_output_state_dict[product]=1

    ###Error margin for fixed production
    error_margin_dict = Production["Margin"].squeeze().to_dict()
    error_margin_dict.pop('blank', None)
    ###Preparation for P_tech_resource_flow_coef parameter
    tech_resource_flow_dict=Available_Technologies.reset_index().set_index(['Technologies','Forced_resource'])['Forced_prod_ratio'].squeeze().to_dict()
    tech_resource_flow_dict.pop(('blank',0),None)
    keylist=list(tech_resource_flow_dict.keys())
    for key in keylist:
        if key[1] == 0:
            tech_resource_flow_dict.pop(key, None)
    ###Preparation for P_tech_resource_capacity parameter
    tech_resource_capacity_dict = Available_Technologies.reset_index().set_index(['Technologies', 'Forced_resource'])['Max_capacity_t'].squeeze().to_dict()
    tech_resource_capacity_dict.pop(('blank',0), None)
    keylist=list(tech_resource_capacity_dict.keys())
    for key in keylist:
        if key[1]==0:
            tech_resource_capacity_dict.pop(key,None)
    ###Preparation for P_tech_flows parameter
    Tech_param = Technologies_Parameters
    Tech_param = Tech_param.reset_index().melt(id_vars=["Resource"], value_vars=Tech_param.columns,
                                               var_name="Technologies", value_name="Flow")
    Tech_param = Tech_param[~Tech_param.Technologies.isin(["unit"])].set_index(
        ["Technologies", "Resource"])




    ###############
    # Sets       ##
    ###############
    TECHNOLOGIES = set(tech_list)
    RESOURCE_TECHS = set(resource_tech_list)
    RESOURCES = set(resource_list)
    PRIMARY_RESOURCES = set(primary_resource_list)

    model.TECHNOLOGIES = Set(initialize=TECHNOLOGIES, ordered=False)
    model.PRIMARY_RESOURCE_TECHS = Set(initialize=RESOURCE_TECHS, ordered=False)
    model.RESOURCES = Set(initialize=RESOURCES, ordered=False)
    model.PRIMARY_RESOURCES = Set(initialize=PRIMARY_RESOURCES, ordered=False)

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
    model.P_carbon_tax=carbon_tax

    model.P_products_boolean=Param(model.RESOURCES,default=0,initialize=products_output_state_dict)

    model.P_resource_prod=Param(model.RESOURCES,default=0,initialize=production_dict, within=Any)


    model.P_production_error_margin = Param(model.RESOURCES, default=0,
                                     initialize=error_margin_dict, within=Any)

    model.P_tech_resource_flow_coef=Param(model.TECHNOLOGIES,model.RESOURCES,default=0,initialize=tech_resource_flow_dict,within=Any) #% of total resource flow for a given tech, e.g. we want 60% of the steel coming from recycling technologies

    model.P_tech_resource_capacity = Param(model.TECHNOLOGIES, model.RESOURCES,default=0,initialize=tech_resource_capacity_dict,within=Any) #production capacity in t of a technology for an associated resource

    model.P_tech_flows = Param(model.TECHNOLOGIES, model.RESOURCES, default=0,
                               initialize=Tech_param.squeeze().to_dict())
    ################
    # Variables    #
    ################
    model.V_cost = Var(domain=NonNegativeReals)
    model.V_emissions = Var(domain=Reals)
    model.V_emissions_plus=Var(domain=PositiveReals)
    model.V_emissions_minus=Var(domain=NegativeReals)
    model.V_resource_flow = Var(model.RESOURCES, domain=NegativeReals)
    model.V_resource_inflow = Var(model.RESOURCES, domain=PositiveReals)
    model.V_resource_outflow = Var(model.RESOURCES, domain=PositiveReals)
    model.V_primary_resource_production=Var(model.PRIMARY_RESOURCE_TECHS,model.PRIMARY_RESOURCES,domain=PositiveReals)
    model.V_technology_use_coef=Var(model.TECHNOLOGIES,domain=PositiveReals)
    model.V_resource_tech_inflow = Var(model.TECHNOLOGIES,model.RESOURCES, domain=PositiveReals)
    model.V_resource_tech_outflow = Var(model.TECHNOLOGIES,model.RESOURCES, domain=PositiveReals)

    ########################
    # Objective Function   #
    ########################

    def Objective_rule(model):
        if opti2mini == "cost":
            return model.V_cost
        elif opti2mini == "emissions":
            return model.V_emissions

        else:
            return model.V_cost

    model.OBJ = Objective(rule=Objective_rule, sense=minimize)

    #################
    # Constraints   #
    #################
    def Cost_definition_rule(model):
        return model.V_cost == sum(
            model.P_flow_cost[tech] * model.V_technology_use_coef[tech] for tech in model.TECHNOLOGIES) + \
               sum(model.P_capex[tech] * model.P_CRF[tech] * model.V_technology_use_coef[tech] for tech in
                   model.TECHNOLOGIES) + \
               model.P_carbon_tax * model.V_emissions_plus

    model.Cost_definitionCtr = Constraint(rule=Cost_definition_rule)

    def Emissions_definition_rule(model):
        return model.V_emissions == sum(
            model.P_emissions[tech] * model.V_technology_use_coef[tech] for tech in model.TECHNOLOGIES)

    model.Emissions_definitionCtr = Constraint(rule=Emissions_definition_rule)

    def Emissions_definition_2nd_rule(model):
        return model.V_emissions==model.V_emissions_plus+model.V_emissions_minus
    model.Emissions_definition_2ndCtr=Constraint(rule=Emissions_definition_2nd_rule)

    def Resource_flow_definition_1st_rule(model,resource):
        return model.V_resource_flow[resource]==model.V_resource_inflow[resource]-model.V_resource_outflow[resource]
    model.Resource_flow_definition_1stCtr=Constraint(model.RESOURCES,rule=Resource_flow_definition_1st_rule)

    def Resource_flow_definition_2nd_rule(model,tech,resource):
        return model.V_resource_tech_inflow[tech,resource]-model.V_resource_tech_outflow[tech,resource]==model.V_technology_use_coef[tech]*model.P_tech_flows[tech,resource]
    model.Resource_flow_definition_2ndCtr = Constraint(model.TECHNOLOGIES,model.RESOURCES, rule=Resource_flow_definition_2nd_rule)

    def Resource_flow_tech_rule(model,tech,resource):
        if model.P_tech_flows[tech,resource]>0:
            return model.V_resource_tech_outflow[tech,resource]==0
        else:
            return model.V_resource_tech_inflow[tech, resource] == 0
    model.Resource_flow_techCtr=Constraint(model.TECHNOLOGIES,model.RESOURCES,rule=Resource_flow_tech_rule)

    def Resource_flow_definition_3rd_rule(model,resource):
        return model.V_resource_inflow[resource]==sum(model.V_resource_tech_inflow[tech,resource] for tech in model.TECHNOLOGIES)
    model.Resource_flow_definition_3rdCtr = Constraint(model.RESOURCES, rule=Resource_flow_definition_3rd_rule)

    def Resource_flow_definition_4th_rule(model,resource):
        return model.V_resource_outflow[resource]==sum(model.V_resource_tech_outflow[tech,resource] for tech in model.TECHNOLOGIES)
    model.Resource_flow_definition_4thCtr = Constraint(model.RESOURCES, rule=Resource_flow_definition_4th_rule)


    def Resource_flow_equilibrium_rule(model,resource):
        if model.P_products_boolean[resource] == 0:
            return model.V_resource_flow[resource]==0
        else:
            return Constraint.Skip
    model.Resource_flow_equilibriumCtr=Constraint(model.RESOURCES,rule=Resource_flow_equilibrium_rule)

    def Primary_resources_prod_limit_rule(model,resource):
        return model.V_resource_inflow[resource]>=sum(model.V_resource_tech_outflow[tech,resource] for tech in model.PRIMARY_RESOURCE_TECHS)
    model.Primary_resource_prod_limit_rule=Constraint(model.PRIMARY_RESOURCES,rule=Primary_resources_prod_limit_rule)



    ###Production Constraints###
    def Production_moins_rule(model,resource):
        if model.P_resource_prod[resource]!=0:
            return model.P_resource_prod[resource]*(1+model.P_production_error_margin[resource])>=model.V_resource_outflow[resource]
        else:
            return Constraint.Skip
    def Production_plus_rule(model,resource):
        if model.P_resource_prod[resource]!=0:
            return model.P_resource_prod[resource]*(1-model.P_production_error_margin[resource])<=model.V_resource_outflow[resource]
        else:
            return Constraint.Skip

    model.Production_moinsCtr=Constraint(model.RESOURCES,rule=Production_moins_rule)
    model.Production_plusCtr = Constraint(model.RESOURCES, rule=Production_plus_rule)

    def Technology_Production_rule(model,tech,resource):
        if model.P_tech_resource_flow_coef[tech,resource]!=0:
            return model.P_tech_resource_flow_coef[tech,resource]*model.V_resource_outflow[resource]==-model.V_technology_use_coef[tech]*model.P_tech_flows[tech,resource]
        else:
            return Constraint.Skip
    model.Technology_ProductionCtr=Constraint(model.TECHNOLOGIES,model.RESOURCES,rule=Technology_Production_rule)

    def Technology_Capacity_rule(model,tech,resource):
        if model.P_tech_resource_capacity[tech,resource]>0:
            return model.V_resource_tech_outflow[tech,resource]<=model.P_tech_resource_capacity[tech,resource]
        else:
            return Constraint.Skip
    model.Technology_CapacityCtr=Constraint(model.TECHNOLOGIES,model.RESOURCES,rule=Technology_Capacity_rule)

    opt = SolverFactory('mosek')

    results = opt.solve(model)

    ######################
    # Results treatment  #
    ######################
    # print("Print values for all variables")
    Results = {}
    for v in model.component_data_objects(Var):
        if  v.name[:29]!='V_primary_resource_production' and v.name[:23]!='V_resource_tech_outflow' and \
            v.name[:22]!='V_resource_tech_inflow' and v.name[:15]!='V_resource_flow':
            # print(v,v.value)
            Results[v.name]= v.value

    return Results

#
# input_path = "Input/Steel/Data/"
# Resources_Technologies=pd.read_excel(input_path+"Resources_Technologies.xlsx").fillna(0)
# Production_Technologies = pd.read_excel(input_path + "Steel_Technologies.xlsx").fillna(0)
# Available_Technologies = pd.read_excel(input_path + "Steel_available_techs_2015.xlsx").fillna(0)
# Production = pd.read_excel(input_path + "Steel_production_2015.xlsx").fillna(0)
# main(Resources_Technologies,Production_Technologies,Available_Technologies,Production,opti2mini="cost",carbon_tax=0)