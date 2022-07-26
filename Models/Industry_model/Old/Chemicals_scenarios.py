from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.core import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###Preliminary Sources###
#France Transistion Plan - https://finance-climact.fr/wp-content/uploads/2021/06/memo-pts-chimie-2021.pdf
#Historical Production - https://www.statistiques.developpement-durable.gouv.fr/sites/default/files/2021-01/datalab_essentiel_235_activite_de_la_petrochimie_en_france_en_2019.pdf
#Technologies consumption - https://www.sciencedirect.com/science/article/pii/S2095495620302606#tbl0002 & https://www.sciencedirect.com/science/article/pii/S0959652615019241
#Technologies consumption - https://www.sciencedirect.com/science/article/pii/S0959652622014949 & https://www.sciencedirect.com/science/article/pii/S0360319918328854
#Technologies emissions - https://www.sciencedirect.com/science/article/pii/S1364032118305732#s0100
#Cost - https://www.iea.org/data-and-statistics/charts/simplified-levelised-cost-of-petrochemicals-for-selected-feedstocks-and-regions-2017
#############

#TODO: add the possibility of minimising costs, emissions and/or production at will DONE
#TODO: add, if steam value in t is >0, energy consumption for its generation (gas or elec) DONE (without biogas)
#TODO: add biogas (as a percentage of the gas resource? that would diminish the emissions of gas consuming processes ==> how do I address that?)

def Chemicals_Scenario(year=2018,version=1,opti2mini="production"):
    input_path = "../Input/Chemistry/Data/Old/"
    Resources_characteristics=pd.read_excel(input_path+"Resources_characteristics.xlsx").set_index("Resource")
    Technologies_Parameters = pd.read_excel(input_path + "Chemistry_Technologies.xlsx").fillna(0)
    Available_Technologies = pd.read_excel(input_path + "Chemistry_available_techs_"+str(year)+"_v"+str(version)+".xlsx").fillna(0)
    Production = pd.read_excel(input_path + "Chemistry_production_"+str(year)+"_v"+str(version)+".xlsx").fillna(0)


    model=ConcreteModel()

    #####################
    # Data preparation ##
    #####################
    Technologies_Parameters_other=Technologies_Parameters[Technologies_Parameters.Resource.isin(["Emissions","cost"])].set_index('Resource')
    Technologies_Parameters = Technologies_Parameters[~Technologies_Parameters.Resource.isin(["Emissions", "cost"])].set_index('Resource')
    Available_Technologies=Available_Technologies.set_index('Technologies')
    Production=Production.set_index('Resource')


    resources_techs=["Coal","Gas","Biogas","Oil","Biomass","Electricity","E_boiler","Gas_boiler","SMR","Electrolyser","Refining","BioNaphta"]
    energy_resources_list=['sc_input',"hydrogen","gas","coal","biomass","oil","steam","electricity"]
    tech_list=Available_Technologies.index.get_level_values("Technologies").unique().to_list()
    tech_list = list(dict.fromkeys(tech_list)) + resources_techs
    Technologies_Parameters=Technologies_Parameters[Technologies_Parameters.columns[:3].to_list()+tech_list]
    Technologies_Parameters_other = Technologies_Parameters_other[Technologies_Parameters.columns[:3].to_list() + tech_list]


    Tech_param = Technologies_Parameters
    Tech_param = Tech_param.reset_index().melt(id_vars=["Resource"], value_vars=Tech_param.columns,var_name="Technologies", value_name="Flow")
    Tech_param = Tech_param[~Tech_param.Technologies.isin(["Zero", "Positive", "unit"])].set_index(["Technologies", "Resource"])
    ###############
    # Sets       ##
    ###############
    TECHNOLOGIES=set(tech_list)
    RESOURCES_TECHS=set(resources_techs)
    RESOURCES=set(Technologies_Parameters.index.get_level_values("Resource").unique().tolist())
    ENERGY_RESOURCES=set(energy_resources_list)
    PRODUCED_RESOURCES=set(Production.index.get_level_values("Resource").unique())
    model.TECHNOLOGIES=Set(initialize=TECHNOLOGIES,ordered=False)
    model.RESOURCES_TECHS=Set(initialize=RESOURCES_TECHS,ordered=False)
    model.RESOURCES=Set(initialize=RESOURCES,ordered=False)
    model.ENERGY_RESOURCES=Set(initialize=ENERGY_RESOURCES,ordered=False)
    model.PRODUCED_RESOURCES=Set(initialize=PRODUCED_RESOURCES,ordered=False)
    ###############
    # Parameters ##
    ###############
    model.P_emissions=Param(model.TECHNOLOGIES,default=0,initialize=Technologies_Parameters_other.loc["Emissions",Technologies_Parameters_other.columns[3:]].to_frame().squeeze().to_dict(),domain=Any)
    model.P_cost=Param(model.TECHNOLOGIES,default=0,initialize=Technologies_Parameters_other.loc["cost",Technologies_Parameters_other.columns[3:]].to_frame().squeeze().to_dict(),domain=Any)
    model.P_tech_flows=Param(model.TECHNOLOGIES,model.RESOURCES,default=0,initialize=Tech_param.squeeze().to_dict())
    model.P_equality_flow = Param(model.RESOURCES, default=0,initialize=Technologies_Parameters["Zero"].squeeze().to_dict(),domain=Binary)

    model.P_production=Param(model.PRODUCED_RESOURCES,default=0,initialize=Production["Production"].squeeze().to_dict(),within=Any)
    model.P_productionCtr = Param(model.PRODUCED_RESOURCES, default=0, initialize=Production["Constraint"].squeeze().to_dict(),within=Any)
    model.P_productionmargin = Param(model.PRODUCED_RESOURCES, default=0,initialize=Production["Margin"].squeeze().to_dict(), within=Any)

    model.P_forced_prod_ratio=Param(model.TECHNOLOGIES,default=0,initialize=Available_Technologies["Forced_prod_ratio"].squeeze().to_dict(),within=Any)
    model.P_forced_resource = Param(model.TECHNOLOGIES, default=0,initialize=Available_Technologies["Forced_resource"].squeeze().to_dict(),within=Any)
    model.P_max_capacity = Param(model.TECHNOLOGIES, default=0,initialize=Available_Technologies["Max_capacity_t"].squeeze().to_dict(),domain=NonNegativeReals)
    model.P_max_capacity_resource = Param(model.TECHNOLOGIES, default=0,initialize=Available_Technologies["Max_capacity_resource"].squeeze().to_dict(),within=Any)





    ################
    # Variables    #
    ################
    model.V_cost=Var(domain=NonNegativeReals)
    model.V_emissions=Var(domain=Reals)
    model.V_resources_flow=Var(model.RESOURCES,domain=Reals)
    model.V_technology_usage=Var(model.TECHNOLOGIES,domain=NonNegativeReals)
    ########################
    # Objective Function   #
    ########################
    def Objective_rule(model):
        if opti2mini=="cost":
            return model.V_cost
        elif opti2mini=="emissions": #TODO: does not work for now as biogas emissions are negative ==> infinite minimization of emissions
            return model.V_emissions
        # elif opti2mini=="production": #TODO: does not work for now as the sum is negative thus minimization lasts forever
        #     return sum(model.V_technology_usage[tech]*model.P_tech_flows[tech,resource] for resource in model.RESOURCES for tech in model.TECHNOLOGIES)

        else:
            return model.V_cost
    model.OBJ=Objective(rule=Objective_rule,sense=minimize)

    #################
    # Constraints   #
    #################
    def Cost_rule(model):
        return model.V_cost==sum(model.P_cost[tech]*model.V_technology_usage[tech] for tech in model.TECHNOLOGIES)
    model.CostCtr=Constraint(rule=Cost_rule)

    def Emissions_rule(model):
        return model.V_emissions==sum(model.P_emissions[tech]*model.V_technology_usage[tech] for tech in model.TECHNOLOGIES)
    model.EmissionsCtr=Constraint(rule=Emissions_rule)

    def Production_rule(model,produced_resource):
        # if model.P_productionCtr[produced_resource]=="equal":
        #     return sum(-model.P_tech_flows[tech,produced_resource]*model.V_technology_usage[tech] for tech in model.TECHNOLOGIES) ==model.P_production[produced_resource]
        if model.P_productionCtr[produced_resource]=="inf":
            return sum(-model.P_tech_flows[tech,produced_resource]*model.V_technology_usage[tech] for tech in model.TECHNOLOGIES) <=model.P_production[produced_resource]
        elif model.P_productionCtr[produced_resource]=="sup":
            return sum(-model.P_tech_flows[tech,produced_resource]*model.V_technology_usage[tech] for tech in model.TECHNOLOGIES) >=model.P_production[produced_resource]
        else:
            return Constraint.Skip

    model.ProductionsupinfCtr=Constraint(model.PRODUCED_RESOURCES,rule=Production_rule)

    def Production_equalsup_rule(model,produced_resource):
        if model.P_productionCtr[produced_resource]=="equal":
            return sum(-model.P_tech_flows[tech,produced_resource]*model.V_technology_usage[tech] for tech in model.TECHNOLOGIES) \
                   <=model.P_production[produced_resource]*(1+model.P_productionmargin[produced_resource])
        else:
            return Constraint.Skip

    model.ProductionequalsupCtr=Constraint(model.PRODUCED_RESOURCES,rule=Production_equalsup_rule)

    def Production_equalinf_rule(model,produced_resource):
        if model.P_productionCtr[produced_resource]=="equal":
            return sum(-model.P_tech_flows[tech,produced_resource]*model.V_technology_usage[tech] for tech in model.TECHNOLOGIES) \
                   >=model.P_production[produced_resource]*(1-model.P_productionmargin[produced_resource])
        else:
            return Constraint.Skip

    model.ProductionequalinfCtr=Constraint(model.PRODUCED_RESOURCES,rule=Production_equalinf_rule)

    def Flow_rule(model,resource):
        return sum(model.P_tech_flows[tech,resource]*model.V_technology_usage[tech] for tech in model.TECHNOLOGIES) ==model.V_resources_flow[resource]
    model.FlowCtr=Constraint(model.RESOURCES,rule=Flow_rule)


    def Forced_prod_rule(model,tech):
        if model.P_forced_prod_ratio[tech]>0:
            return -model.P_tech_flows[tech,model.P_forced_resource[tech]]*model.V_technology_usage[tech] == model.P_forced_prod_ratio[tech]*model.P_production[model.P_forced_resource[tech]]
        else:
            return Constraint.Skip
    model.Forced_prodCtr=Constraint(model.TECHNOLOGIES,rule=Forced_prod_rule)



    def Max_capacity_rule(model,tech):
        if model.P_max_capacity[tech]>0:
            return -model.P_tech_flows[tech, model.P_max_capacity_resource[tech]] * model.V_technology_usage[tech]<=model.P_max_capacity[tech]
        else:
            return Constraint.Skip
    model.Max_capacityCtr=Constraint(model.TECHNOLOGIES,rule=Max_capacity_rule)

    def Resource_flow_rule(model,energy_resource):
        return sum(-model.P_tech_flows[tech, energy_resource] * model.V_technology_usage[tech] for tech in model.TECHNOLOGIES)>=0

    model.Resource_flowCtr=Constraint(model.ENERGY_RESOURCES,rule=Resource_flow_rule)

    def Equilibrium_flow_rule(model,resource):
        if model.P_equality_flow[resource]==True:
            return sum(-model.P_tech_flows[tech, resource] * model.V_technology_usage[tech] for tech in model.TECHNOLOGIES)==0
        else:
            return Constraint.Skip
    model.Equilibrium_flowCtr=Constraint(model.RESOURCES,rule=Equilibrium_flow_rule)




    opt = SolverFactory('mosek')

    results = opt.solve(model)

    ######################
    # Results treatment  #
    ######################
    print("Print values for all variables")
    for v in model.component_data_objects(Var):
        print(v, v.value)

    optimal_values = [[key,value(model.V_resources_flow[key])] for key in model.V_resources_flow]
    df = pd.DataFrame(optimal_values,columns=['Resources','Flow (t)'])

    df["Result"]=np.array(["Result"]*len(df['Resources']))
    df["Energy (MWh)"]=np.array([0]*len(df['Resources']))
    for name in Resources_characteristics.index.get_level_values("Resource").unique():
        df.loc[df.Resources==name,"Energy (MWh)"]=df.loc[df.Resources==name]["Flow (t)"]*Resources_characteristics.loc[name,"calorific_value_MWh_t"]


    #Mass flow figure
    import plotly.express as px
    fig = px.bar(df[~df.Resources.isin(["water","electricity","steam"])], x='Result', y='Flow (t)', color="Resources",title="Mass flow")
    # fig.show()

    #Energy consumption figure
    fig = px.bar(df[df.Resources.isin(Resources_characteristics.index.get_level_values("Resource").unique())], x='Result', y='Energy (MWh)', color="Resources",title="Energy consumption")
    # fig.show()



Chemicals_Scenario(year=2018,version=1,opti2mini="cost")