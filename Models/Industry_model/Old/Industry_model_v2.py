from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.core import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


input_path = "Input/Steel/Data/v2/"
Resources_Technologies=pd.read_excel(input_path+"Resources_Technologies.xlsx").fillna(0)
Production_Technologies = pd.read_excel(input_path + "Steel_Technologies.xlsx").fillna(0)
Available_Technologies = pd.read_excel(input_path + "Steel_available_techs_2015.xlsx").fillna(0)
Production = pd.read_excel(input_path + "Steel_production_2015.xlsx").fillna(0)

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
    def Cost_rule(model):
        return model.V_cost == sum(
            model.P_flow_cost[tech] * model.V_technology_use_coef[tech] for tech in model.TECHNOLOGIES) + \
               sum(model.P_capex[tech] * model.P_CRF[tech] * model.V_technology_use_coef[tech] for tech in
                   model.TECHNOLOGIES) + \
               model.P_carbon_tax * model.V_emissions

    model.CostCtr = Constraint(rule=Cost_rule)

    def Emissions_rule(model):
        return model.V_emissions == sum(
            model.P_emissions[tech] * model.V_technology_use_coef[tech] for tech in model.TECHNOLOGIES)

    model.EmissionsCtr = Constraint(rule=Emissions_rule)

    def Resource_flow_definition_1st_rule(model,resource):
        return model.V_resource_flow[resource]==model.V_resource_inflow[resource]-model.V_resource_outflow[resource]
    model.Resource_flow_definition_1stCtr=Constraint(model.RESOURCES,rule=Resource_flow_definition_1st_rule)

    # def Resource_flow_definition_2nd_rule(model,resource):
    #     return model.V_resource_flow[resource]==sum(model.V_technology_use_coef[tech]*model.P_tech_flows[tech,resource] for tech in model.TECHNOLOGIES)
    # model.Resource_flow_definition_2ndCtr = Constraint(model.RESOURCES, rule=Resource_flow_definition_2nd_rule)

    def Resource_flow_definition_3rd_rule(model,tech,resource):
        return model.V_resource_tech_inflow[tech,resource]-model.V_resource_tech_outflow[tech,resource]==model.V_technology_use_coef[tech]*model.P_tech_flows[tech,resource]
    model.Resource_flow_definition_3ndCtr = Constraint(model.TECHNOLOGIES,model.RESOURCES, rule=Resource_flow_definition_3rd_rule)

    def Resource_flow_definition_4th_rule(model,resource):
        return model.V_resource_inflow[resource]==sum(model.V_resource_tech_inflow[tech,resource] for tech in model.TECHNOLOGIES)
    model.Resource_flow_definition_4thCtr = Constraint(model.RESOURCES, rule=Resource_flow_definition_4th_rule)

    def Resource_flow_definition_5th_rule(model,resource):
        return model.V_resource_outflow[resource]==sum(model.V_resource_tech_outflow[tech,resource] for tech in model.TECHNOLOGIES)
    model.Resource_flow_definition_5thCtr = Constraint(model.RESOURCES, rule=Resource_flow_definition_5th_rule)

    # def Primary_resource_production_definition_rule(model,tech,resource):
    #     return model.V_primary_resource_production[tech,resource]==model.V_technology_use_coef[tech]*-model.P_tech_flows[tech,resource]
    # model.Primary_resource_production_definitionCtr=Constraint(model.PRIMARY_RESOURCE_TECHS,model.PRIMARY_RESOURCES,rule=Primary_resource_production_definition_rule)

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

    ###Resources Constraints

    # def Primary_resource_rule(model,resource):
    #     return model.V_resource_inflow[resource]>=sum(model.V_primary_resource_production[tech,resource] for tech in model.PRIMARY_RESOURCE_TECHS)
    # model.Primary_resourceCtr=Constraint(model.PRIMARY_RESOURCES,rule=Primary_resource_rule)
    #
    # def Primary_resource_2nd_rule(model,resource):
    #     return model.V_resource_flow[resource]<=0
    # model.Primary_resource_2ndCtr=Constraint(model.PRIMARY_RESOURCES,rule=Primary_resource_2nd_rule)



    opt = SolverFactory('mosek')

    results = opt.solve(model)

    ######################
    # Results treatment  #
    ######################
    print("Print values for all variables")
    for v in model.component_data_objects(Var):
        if  v.name[:29]!='V_primary_resource_production' and v.name[:23]!='V_resource_tech_outflow' and \
            v.name[:22]!='V_resource_tech_inflow': #v.name[:15]!='V_resource_flow' and
            print(v, v.value)

main(Resources_Technologies,Production_Technologies,Available_Technologies,Production,opti2mini="cost",carbon_tax=0)