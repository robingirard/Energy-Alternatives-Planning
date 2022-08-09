from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.core import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def main(RESOURCES_Technologies,Production_Technologies,Available_Technologies,Production,opti2mini="cost",carbon_tax=0):
    model = ConcreteModel()

    #####################
    # Data preparation ##
    #####################
    RESOURCES_Technologies_other = RESOURCES_Technologies[
        RESOURCES_Technologies.RESOURCES.isin(["Emissions", "flow_cost","capex","CRF"])].set_index('RESOURCES').drop("unit",axis=1)
    RESOURCES_Technologies = RESOURCES_Technologies[
        ~RESOURCES_Technologies.RESOURCES.isin(["Emissions", "flow_cost","capex","lifetime","discount_rate","CRF"])].set_index('RESOURCES').drop("unit",axis=1)
    Production_Technologies_other = Production_Technologies[
        Production_Technologies.RESOURCES.isin(["Emissions", "flow_cost", "capex", "CRF"])].set_index('RESOURCES').drop("unit",axis=1)
    Production_Technologies = Production_Technologies[
        ~Production_Technologies.RESOURCES.isin(
            ["Emissions", "flow_cost", "capex", "lifetime", "discount_rate", "CRF"])].set_index('RESOURCES').drop("unit",axis=1)



    Available_Technologies = Available_Technologies.set_index('Technologies')
    Production = Production.set_index('RESOURCES')

    RESOURCES_tech_list = list(dict.fromkeys(RESOURCES_Technologies.columns.to_list()))
    RESOURCES_tech_list = list(dict.fromkeys(RESOURCES_tech_list))
    if 'blank' in RESOURCES_tech_list:
        RESOURCES_tech_list.remove('blank')

    production_tech_list = list(dict.fromkeys(Available_Technologies.index.get_level_values("Technologies").unique().to_list()))
    production_tech_list=list(dict.fromkeys(production_tech_list))
    production_tech_list_copy=production_tech_list.copy()
    if 'blank' in production_tech_list:
        production_tech_list.remove('blank')
    for tech in RESOURCES_tech_list:
        if tech in production_tech_list_copy:
            production_tech_list.remove(tech)


    tech_list=production_tech_list+RESOURCES_tech_list
    tech_list=list(dict.fromkeys(tech_list))

    RESOURCES_list = Production_Technologies.index.get_level_values("RESOURCES").unique().tolist()
    if "blank" in RESOURCES_list:
        RESOURCES_list.remove("blank")
    primary_RESOURCES_list = RESOURCES_Technologies.index.get_level_values("RESOURCES").unique().tolist()
    if "blank" in primary_RESOURCES_list:
        primary_RESOURCES_list.remove("blank")

    Technologies_Parameters = pd.concat([RESOURCES_Technologies, Production_Technologies[production_tech_list]], axis=1).fillna(0)
    Technologies_Parameters_other= pd.concat([RESOURCES_Technologies_other, Production_Technologies_other[production_tech_list]], axis=1).fillna(0)

    ###Preparation for P_RESOURCES_prod and P_production_error_margin parameters
    production_dict = Production["Production"].squeeze().to_dict()
    production_dict.pop('blank', None)

    products_output_list=list(production_dict.keys())
    u=products_output_list.copy()
    ##add forgotten products to the production list (if chosen technologies have other outputs [e.g for ethylene production, naphtha cracking produces also propylene and other chemicals])
    for RESOURCES in RESOURCES_list:
        for tech in production_tech_list:
            for product in u:
                flow_given_product_val=Technologies_Parameters.loc[product,tech]
                flow_val=Technologies_Parameters.loc[RESOURCES,tech]
                if flow_val<0 and flow_given_product_val<0:
                    products_output_list.append(RESOURCES)
    products_output_list=list(dict.fromkeys(products_output_list))
    products_output_state_dict={}
    for product in products_output_list:
        products_output_state_dict[product]=1

    ###Error margin for fixed production
    error_margin_dict = Production["Margin"].squeeze().to_dict()
    error_margin_dict.pop('blank', None)
    ###Preparation for P_tech_RESOURCES_flow_coef parameter
    tech_RESOURCES_flow_dict=Available_Technologies.reset_index().set_index(['Technologies','Forced_RESOURCES'])['Forced_prod_ratio'].squeeze().to_dict()
    tech_RESOURCES_flow_dict.pop(('blank',0),None)
    keylist=list(tech_RESOURCES_flow_dict.keys())
    for key in keylist:
        if key[1] == 0:
            tech_RESOURCES_flow_dict.pop(key, None)
    ###Preparation for P_tech_RESOURCES_capacity parameter
    tech_RESOURCES_capacity_dict = Available_Technologies.reset_index().set_index(['Technologies', 'Forced_RESOURCES'])['Max_capacity_t'].squeeze().to_dict()
    tech_RESOURCES_capacity_dict.pop(('blank',0), None)
    keylist=list(tech_RESOURCES_capacity_dict.keys())
    for key in keylist:
        if key[1]==0:
            tech_RESOURCES_capacity_dict.pop(key,None)
    ###Preparation for P_tech_flows parameter
    Tech_param = Technologies_Parameters
    Tech_param = Tech_param.reset_index().melt(id_vars=["RESOURCES"], value_vars=Tech_param.columns,
                                               var_name="Technologies", value_name="Flow")
    Tech_param = Tech_param[~Tech_param.Technologies.isin(["unit"])].set_index(
        ["Technologies", "RESOURCES"])




    ###############
    # Sets       ##
    ###############
    TECHNOLOGIES = set(tech_list)
    RESOURCES_TECHS = set(RESOURCES_tech_list)
    RESOURCES = set(RESOURCES_list)
    PRIMARY_RESOURCES = set(primary_RESOURCES_list)

    model.TECHNOLOGIES = Set(initialize=TECHNOLOGIES, ordered=False)
    model.PRIMARY_RESOURCES_TECHS = Set(initialize=RESOURCES_TECHS, ordered=False)
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

    model.P_RESOURCES_prod=Param(model.RESOURCES,default=0,initialize=production_dict, within=Any)


    model.P_production_error_margin = Param(model.RESOURCES, default=0,
                                     initialize=error_margin_dict, within=Any)

    model.P_tech_RESOURCES_flow_coef=Param(model.TECHNOLOGIES,model.RESOURCES,default=0,initialize=tech_RESOURCES_flow_dict,within=Any) #% of total RESOURCES flow for a given tech, e.g. we want 60% of the steel coming from recycling technologies

    model.P_tech_RESOURCES_capacity = Param(model.TECHNOLOGIES, model.RESOURCES,default=0,initialize=tech_RESOURCES_capacity_dict,within=Any) #production capacity in t of a technology for an associated RESOURCES

    model.P_tech_flows = Param(model.TECHNOLOGIES, model.RESOURCES, default=0,
                               initialize=Tech_param.squeeze().to_dict())
    ################
    # Variables    #
    ################
    model.V_cost = Var(domain=NonNegativeReals)
    model.V_emissions = Var(domain=Reals)
    model.V_emissions_plus=Var(domain=PositiveReals)
    model.V_emissions_minus=Var(domain=NegativeReals)
    model.V_RESOURCES_flow = Var(model.RESOURCES, domain=NegativeReals)
    model.V_RESOURCES_inflow = Var(model.RESOURCES, domain=PositiveReals)
    model.V_RESOURCES_outflow = Var(model.RESOURCES, domain=PositiveReals)
    model.V_primary_RESOURCES_production=Var(model.PRIMARY_RESOURCES_TECHS,model.PRIMARY_RESOURCES,domain=PositiveReals)
    model.V_technology_use_coef=Var(model.TECHNOLOGIES,domain=PositiveReals)
    model.V_RESOURCES_tech_inflow = Var(model.TECHNOLOGIES,model.RESOURCES, domain=PositiveReals)
    model.V_RESOURCES_tech_outflow = Var(model.TECHNOLOGIES,model.RESOURCES, domain=PositiveReals)

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

    def RESOURCES_flow_definition_1st_rule(model,RESOURCES):
        return model.V_RESOURCES_flow[RESOURCES]==model.V_RESOURCES_inflow[RESOURCES]-model.V_RESOURCES_outflow[RESOURCES]
    model.RESOURCES_flow_definition_1stCtr=Constraint(model.RESOURCES,rule=RESOURCES_flow_definition_1st_rule)

    def RESOURCES_flow_definition_2nd_rule(model,tech,RESOURCES):
        return model.V_RESOURCES_tech_inflow[tech,RESOURCES]-model.V_RESOURCES_tech_outflow[tech,RESOURCES]==model.V_technology_use_coef[tech]*model.P_tech_flows[tech,RESOURCES]
    model.RESOURCES_flow_definition_2ndCtr = Constraint(model.TECHNOLOGIES,model.RESOURCES, rule=RESOURCES_flow_definition_2nd_rule)

    def RESOURCES_flow_tech_rule(model,tech,RESOURCES):
        if model.P_tech_flows[tech,RESOURCES]>0:
            return model.V_RESOURCES_tech_outflow[tech,RESOURCES]==0
        else:
            return model.V_RESOURCES_tech_inflow[tech, RESOURCES] == 0
    model.RESOURCES_flow_techCtr=Constraint(model.TECHNOLOGIES,model.RESOURCES,rule=RESOURCES_flow_tech_rule)

    def RESOURCES_flow_definition_3rd_rule(model,RESOURCES):
        return model.V_RESOURCES_inflow[RESOURCES]==sum(model.V_RESOURCES_tech_inflow[tech,RESOURCES] for tech in model.TECHNOLOGIES)
    model.RESOURCES_flow_definition_3rdCtr = Constraint(model.RESOURCES, rule=RESOURCES_flow_definition_3rd_rule)

    def RESOURCES_flow_definition_4th_rule(model,RESOURCES):
        return model.V_RESOURCES_outflow[RESOURCES]==sum(model.V_RESOURCES_tech_outflow[tech,RESOURCES] for tech in model.TECHNOLOGIES)
    model.RESOURCES_flow_definition_4thCtr = Constraint(model.RESOURCES, rule=RESOURCES_flow_definition_4th_rule)


    def RESOURCES_flow_equilibrium_rule(model,RESOURCES):
        if model.P_products_boolean[RESOURCES] == 0:
            return model.V_RESOURCES_flow[RESOURCES]==0
        else:
            return Constraint.Skip
    model.RESOURCES_flow_equilibriumCtr=Constraint(model.RESOURCES,rule=RESOURCES_flow_equilibrium_rule)

    def Primary_RESOURCES_prod_limit_rule(model,RESOURCES):
        return model.V_RESOURCES_inflow[RESOURCES]>=sum(model.V_RESOURCES_tech_outflow[tech,RESOURCES] for tech in model.PRIMARY_RESOURCES_TECHS)
    model.Primary_RESOURCES_prod_limit_rule=Constraint(model.PRIMARY_RESOURCES,rule=Primary_RESOURCES_prod_limit_rule)



    ###Production Constraints###
    def Production_moins_rule(model,RESOURCES):
        if model.P_RESOURCES_prod[RESOURCES]!=0:
            return model.P_RESOURCES_prod[RESOURCES]*(1+model.P_production_error_margin[RESOURCES])>=model.V_RESOURCES_outflow[RESOURCES]
        else:
            return Constraint.Skip
    def Production_plus_rule(model,RESOURCES):
        if model.P_RESOURCES_prod[RESOURCES]!=0:
            return model.P_RESOURCES_prod[RESOURCES]*(1-model.P_production_error_margin[RESOURCES])<=model.V_RESOURCES_outflow[RESOURCES]
        else:
            return Constraint.Skip

    model.Production_moinsCtr=Constraint(model.RESOURCES,rule=Production_moins_rule)
    model.Production_plusCtr = Constraint(model.RESOURCES, rule=Production_plus_rule)

    def Technology_Production_rule(model,tech,RESOURCES):
        if model.P_tech_RESOURCES_flow_coef[tech,RESOURCES]!=0:
            return model.P_tech_RESOURCES_flow_coef[tech,RESOURCES]*model.V_RESOURCES_outflow[RESOURCES]==-model.V_technology_use_coef[tech]*model.P_tech_flows[tech,RESOURCES]
        else:
            return Constraint.Skip
    model.Technology_ProductionCtr=Constraint(model.TECHNOLOGIES,model.RESOURCES,rule=Technology_Production_rule)

    def Technology_Capacity_rule(model,tech,RESOURCES):
        if model.P_tech_RESOURCES_capacity[tech,RESOURCES]>0:
            return model.V_RESOURCES_tech_outflow[tech,RESOURCES]<=model.P_tech_RESOURCES_capacity[tech,RESOURCES]
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
        if  v.name[:29]!='V_primary_RESOURCES_production' and v.name[:23]!='V_RESOURCES_tech_outflow' and \
            v.name[:22]!='V_RESOURCES_tech_inflow' and v.name[:15]!='V_RESOURCES_flow':
            # print(v,v.value)
            Results[v.name]= v.value

    return Results

#
# input_path = "Input/Steel/Data/"
# RESOURCES_Technologies=pd.read_excel(input_path+"RESOURCES_Technologies.xlsx").fillna(0)
# Production_Technologies = pd.read_excel(input_path + "Steel_Technologies.xlsx").fillna(0)
# Available_Technologies = pd.read_excel(input_path + "Steel_available_techs_2015.xlsx").fillna(0)
# Production = pd.read_excel(input_path + "Steel_production_2015.xlsx").fillna(0)
# main(RESOURCES_Technologies,Production_Technologies,Available_Technologies,Production,opti2mini="cost",carbon_tax=0)


def GetIndustryModel(Parameters,opti2mini="cost",carbon_tax=0):
    model = ConcreteModel()

    #####################
    # Data preparation ##
    #####################

    TECHNOLOGIES = set(Parameters["TECHNOLOGIES_parameters"].index.get_level_values('TECHNOLOGIES').unique())
    RESOURCES = set(Parameters["RESOURCES_parameters"].index.get_level_values('RESOURCES').unique())

    ###
    # SETS
    ###
    model.TECHNOLOGIES = Set(initialize=TECHNOLOGIES, ordered=False)
    model.RESOURCES = Set(initialize=RESOURCES, ordered=False)
    model.TECHNOLOGIES_RESOURCES = model.TECHNOLOGIES * model.RESOURCES

    ###############
    # Parameters ##
    ###############

    for COLNAME in Parameters["TECHNOLOGIES_parameters"]:
        exec("model.P_" + COLNAME + " =  Param(model.TECHNOLOGIES, mutable=False, domain=Any,default=0," +
                 "initialize=Parameters[\"TECHNOLOGIES_parameters\"]." + COLNAME + ".squeeze().to_dict())")

    for COLNAME in Parameters["RESOURCES_parameters"]:
        exec("model.P_" + COLNAME + " =  Param(model.RESOURCES, mutable=False, domain=Any,default=0," +
                 "initialize=Parameters[\"RESOURCES_parameters\"]." + COLNAME + ".squeeze().to_dict())")

    for COLNAME in Parameters["TECHNOLOGIES_RESOURCES_parameters"]:
        exec("model.P_" + COLNAME + " =  Param(model.TECHNOLOGIES_RESOURCES, mutable=False, domain=Any,default=0," +
                 "initialize=Parameters[\"TECHNOLOGIES_RESOURCES_parameters\"]." + COLNAME + ".squeeze().to_dict())")

    ################
    # Variables    #
    ################

    model.V_cost = Var(domain=NonNegativeReals)
    model.V_emissions = Var(domain=Reals)
    model.V_emissions_plus=Var(domain=PositiveReals)
    model.V_emissions_minus=Var(domain=NegativeReals)
    model.V_RESOURCES_flow = Var(model.RESOURCES, domain=NegativeReals)
    model.V_RESOURCES_inflow = Var(model.RESOURCES, domain=PositiveReals)
    model.V_RESOURCES_outflow = Var(model.RESOURCES, domain=PositiveReals)
    model.V_technology_use_coef=Var(model.TECHNOLOGIES,domain=PositiveReals)
    model.V_RESOURCES_tech_inflow = Var(model.TECHNOLOGIES,model.RESOURCES, domain=PositiveReals)
    model.V_RESOURCES_tech_outflow = Var(model.TECHNOLOGIES,model.RESOURCES, domain=PositiveReals)

    ########################
    # Objective Function   #
    ########################

    def Objective_rule(model):
        if opti2mini == "cost":
            return model.V_cost ### see the constraint below
        elif opti2mini == "emissions":
            return model.V_emissions ### see the constraint below
        else:
            return model.V_cost ### see the constraint below
    model.OBJ = Objective(rule=Objective_rule, sense=minimize)

    def Cost_definition_rule(model):
        return model.V_cost == sum(
            model.P_flow_cost_t[tech] * model.V_technology_use_coef[tech] for tech in model.TECHNOLOGIES) + \
               sum(model.P_capex[tech] * model.P_CRF[tech] * model.V_technology_use_coef[tech] for tech in
                   model.TECHNOLOGIES) + \
               model.P_carbon_tax * model.V_emissions_plus

    model.Cost_definitionCtr = Constraint(rule=Cost_definition_rule)

    def Emissions_definition_rule(model):
        return model.V_emissions == sum(
            model.P_emissions_t[tech] * model.V_technology_use_coef[tech] for tech in model.TECHNOLOGIES)

    model.Emissions_definitionCtr = Constraint(rule=Emissions_definition_rule)


    #################
    # Constraints   #
    #################

    #decomposition (+/-) of emission to avoid negative emissions remuneration (through carbon tax)
    def Emissions_definition_2nd_rule(model):
        return model.V_emissions==model.V_emissions_plus+model.V_emissions_minus
    model.Emissions_definition_2ndCtr=Constraint(rule=Emissions_definition_2nd_rule)

    #decomposition (+/-) of RESOURCES flow
    def RESOURCES_flow_definition_1st_rule(model,RESOURCES):
        return model.V_RESOURCES_flow[RESOURCES]==model.V_RESOURCES_inflow[RESOURCES]-model.V_RESOURCES_outflow[RESOURCES]
    model.RESOURCES_flow_definition_1stCtr=Constraint(model.RESOURCES,rule=RESOURCES_flow_definition_1st_rule)

    #decomposition (+/-) of tech flow
    def RESOURCES_flow_definition_2nd_rule(model,tech,RESOURCES):
        return model.V_RESOURCES_tech_inflow[tech,RESOURCES]-model.V_RESOURCES_tech_outflow[tech,RESOURCES]==model.V_technology_use_coef[tech]*model.[tech,RESOURCES]
    model.RESOURCES_flow_definition_2ndCtr = Constraint(model.TECHNOLOGIES,model.RESOURCES, rule=RESOURCES_flow_definition_2nd_rule)

    def RESOURCES_flow_tech_rule(model,tech,RESOURCES):
        if model.P_tech_flows[tech,RESOURCES]>0:
            return model.V_RESOURCES_tech_outflow[tech,RESOURCES]==0
        else:
            return model.V_RESOURCES_tech_inflow[tech, RESOURCES] == 0
    model.RESOURCES_flow_techCtr=Constraint(model.TECHNOLOGIES,model.RESOURCES,rule=RESOURCES_flow_tech_rule)

    def RESOURCES_flow_definition_3rd_rule(model,RESOURCES):
        return model.V_RESOURCES_inflow[RESOURCES]==sum(model.V_RESOURCES_tech_inflow[tech,RESOURCES] for tech in model.TECHNOLOGIES)
    model.RESOURCES_flow_definition_3rdCtr = Constraint(model.RESOURCES, rule=RESOURCES_flow_definition_3rd_rule)

    def RESOURCES_flow_definition_4th_rule(model,RESOURCES):
        return model.V_RESOURCES_outflow[RESOURCES]==sum(model.V_RESOURCES_tech_outflow[tech,RESOURCES] for tech in model.TECHNOLOGIES)
    model.RESOURCES_flow_definition_4thCtr = Constraint(model.RESOURCES, rule=RESOURCES_flow_definition_4th_rule)


    def RESOURCES_flow_equilibrium_rule(model,RESOURCES):
        if model.P_products_boolean[RESOURCES] == 0:
            return model.V_RESOURCES_flow[RESOURCES]==0
        else:
            return Constraint.Skip
    model.RESOURCES_flow_equilibriumCtr=Constraint(model.RESOURCES,rule=RESOURCES_flow_equilibrium_rule)


    ###Production Constraints###
    def Production_moins_rule(model,RESOURCES):
        if model.P_RESOURCES_prod[RESOURCES]!=0:
            return model.P_RESOURCES_prod[RESOURCES]*(1+model.P_production_error_margin[RESOURCES])>=model.V_RESOURCES_outflow[RESOURCES]
        else:
            return Constraint.Skip
    def Production_plus_rule(model,RESOURCES):
        if model.P_RESOURCES_prod[RESOURCES]!=0:
            return model.P_RESOURCES_prod[RESOURCES]*(1-model.P_production_error_margin[RESOURCES])<=model.V_RESOURCES_outflow[RESOURCES]
        else:
            return Constraint.Skip

    model.Production_moinsCtr=Constraint(model.RESOURCES,rule=Production_moins_rule)
    model.Production_plusCtr = Constraint(model.RESOURCES, rule=Production_plus_rule)

    def Technology_Production_rule(model,tech,RESOURCES):
        if model.P_tech_RESOURCES_flow_coef[tech,RESOURCES]!=0:
            return model.P_tech_RESOURCES_flow_coef[tech,RESOURCES]*model.V_RESOURCES_outflow[RESOURCES]==-model.V_technology_use_coef[tech]*model.P_tech_flows[tech,RESOURCES]
        else:
            return Constraint.Skip
    model.Technology_ProductionCtr=Constraint(model.TECHNOLOGIES,model.RESOURCES,rule=Technology_Production_rule)

    def Technology_Capacity_rule(model,tech,RESOURCES):
        if model.P_tech_RESOURCES_capacity[tech,RESOURCES]>0:
            return model.V_RESOURCES_tech_outflow[tech,RESOURCES]<=model.P_tech_RESOURCES_capacity[tech,RESOURCES]
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
        if  v.name[:29]!='V_primary_RESOURCES_production' and v.name[:23]!='V_RESOURCES_tech_outflow' and \
            v.name[:22]!='V_RESOURCES_tech_inflow' and v.name[:15]!='V_RESOURCES_flow':
            # print(v,v.value)
            Results[v.name]= v.value

    return Results

