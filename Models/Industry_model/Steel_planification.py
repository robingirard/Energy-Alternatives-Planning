import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from Models.Industry_model.Industry_model_planification import *

# Create an array with the colors you want to use
colors = ["#004776","#b8e1ff","#72c5fe","#2baaff","#f8b740","#005f9e","#000000",
          "#e7e7e7","#fef3e8","#e8f1ff","#ebf5ff","#c69131","#2087cb"]# Set your custom color palette
customPalette = sns.set_palette(sns.color_palette(colors))

input_path = "Input/Steel/Data/"
Parameters_data= pd.read_excel("Input/Steel/Data/parameters_planification.xlsx",sheet_name=["TECHNOLOGIES","RESOURCES","TECHNOLOGIES_RESOURCES","TECHNOLOGIES_TECH_TYPE"])

Parameters={"TECHNOLOGIES_TECH_TYPE_parameters" : Parameters_data["TECHNOLOGIES_TECH_TYPE"].fillna(0).set_index(['TECHNOLOGIES','TECH_TYPE','YEAR']),
            "RESOURCES_parameters" : Parameters_data["RESOURCES"].fillna(0).set_index(['RESOURCES','YEAR']),
            "TECHNOLOGIES_RESOURCES_parameters" : Parameters_data["TECHNOLOGIES_RESOURCES"].fillna(0).\
                melt(id_vars=['TECHNOLOGIES','TECH_TYPE','YEAR'], var_name="RESOURCES",value_name='conversion_factor').\
                set_index(['TECHNOLOGIES','TECH_TYPE','RESOURCES','YEAR']),
            "TECHNOLOGIES_parameters":Parameters_data["TECHNOLOGIES"].fillna(0).set_index(["TECHNOLOGIES","YEAR"])
}


model=GetIndustryModel(Parameters,opti2mini="cost")#emission
opt = SolverFactory('mosek')

results = opt.solve(model)
######################
# Results treatment  #
######################
# print("Print values for all variables")
Results = {}
for v in model.component_data_objects(Var):
    #if v.name[:29] != 'V_primary_RESOURCES_production' and v.name[:23] != 'V_resource_tech_outflow' and \
    #        v.name[:22] != 'V_resource_tech_inflow' and v.name[:15] != 'V_resource_flow':
        # print(v,v.value)
    Results[v.name] = v.value
    # if v.value!=0 and v.value != None:
    #     print(v.name,v.value)

# print(Results)
Year_list=list(Parameters_data["TECHNOLOGIES"]["YEAR"].unique())
Steel_production={}
Cost_evolution={}
Cost_per_t={}
Emissions_evolution={}
Emissions_per_t={}

for year in Year_list:
    Steel_production[year]=Parameters["RESOURCES_parameters"].loc[('Steel',year),'output']
    Cost_evolution[year]=Results["V_cost["+str(year)+"]"]
    Emissions_evolution[year]=Results["V_emissions["+str(year)+"]"]
    Cost_per_t[year]=Cost_evolution[year]/Steel_production[year]
    Emissions_per_t[year] = Emissions_evolution[year] / Steel_production[year]

#
fig,ax=plt.subplots()
ax.set_title("Cost per t steel")
for y in Cost_per_t.keys():
    ax.bar([str(y)],[Cost_per_t[y]])
# ax.bar(["Emissions (Mt)"],Results["V_emissions"]/1e6)
for bars in ax.containers:
    ax.bar_label(bars)
plt.ylabel("Cost (€)")
plt.show()

fig,ax=plt.subplots()
ax.set_title("Emissions per t steel")
for y in Cost_per_t.keys():
    ax.bar([str(y)],[Emissions_per_t[y]])
for bars in ax.containers:
    ax.bar_label(bars)
plt.ylabel("tCO2")
plt.show()

Steel_techno_dict={"BF-BOF":["Coal","H2","Bio"],"EAF":[0],"DRI-EAF":["H2","CH4"],"Kiln-DRI-EAF":['Bio','Coal']}

Steel_prod_tech_prod_emissions=pd.DataFrame(columns=["Technology","Year","Production (Mt)","Emissions (MtCO2)"])
Steel_prod_tech_cost_capacity=pd.DataFrame(columns=["Technology","Year","Added capacity (Mt)","Capacity (Mt)","Annualized CAPEX (B€/yr)"])

for year in Year_list:
    for tech in Steel_techno_dict.keys():
        cost=Results["V_technology_cost["+tech+","+str(year)+"]"]*1e-9
        capacity=Results["V_technology_use_coef_capacity["+tech+","+str(year)+"]"]*1e-6
        added_capacity=Results["V_added_capacity["+tech+","+str(year)+"]"]*1e-6
        Steel_prod_tech_cost_capacity=pd.concat([Steel_prod_tech_cost_capacity,
                                                 pd.DataFrame(np.array([[tech,year,added_capacity,capacity,cost]]),columns=["Technology","Year","Added capacity (Mt)","Capacity (Mt)","Annualized CAPEX (B€/yr)"])
                                                 ],ignore_index=True)
        for tech_types in Steel_techno_dict.values():
            for tech_type in tech_types:
                tech_name=tech
                if tech_type!=0:
                    tech_name=tech_type+"-"+tech_name
                production=Results["V_resource_tech_outflow["+tech+","+str(tech_type)+",Steel,"+str(year)+"]"]*1e-6
                emissions=Results["V_emissions_tech["+tech+","+str(tech_type)+","+str(year)+"]"]*1e-6
                if production!=0:
                    Steel_prod_tech_prod_emissions=pd.concat([Steel_prod_tech_prod_emissions,
                                                          pd.DataFrame(np.array([[tech_name,year,production,emissions]]),
                                                          columns=["Technology", "Year", "Production (Mt)",
                                                                                "Emissions (MtCO2)"])
                                                          ],ignore_index=True)
    Steel_prod_tech_prod_emissions.drop_duplicates(inplace=True)

df_steel_prod_tech=Steel_prod_tech_prod_emissions.pivot(index="Year", columns="Technology", values="Production (Mt)").fillna(0).astype("float")
df_steel_emissions_tech=Steel_prod_tech_prod_emissions.pivot(index="Year", columns="Technology", values="Emissions (MtCO2)").fillna(0).astype("float")

df_steel_cost_tech=Steel_prod_tech_cost_capacity.pivot(index="Year",columns="Technology",values="Annualized CAPEX (B€/yr)").fillna(0).astype("float")
df_steel_capacity_tech=Steel_prod_tech_cost_capacity.pivot(index="Year",columns="Technology",values="Capacity (Mt)").fillna(0).astype("float")
df_steel_added_capacity_tech=Steel_prod_tech_cost_capacity.pivot(index="Year",columns="Technology",values="Added capacity (Mt)").fillna(0).astype("float")


sns.set(palette=customPalette)
df_steel_prod_tech.plot(kind="bar",stacked="True")
plt.xlabel("Year")
plt.ylabel("Production (Mt)")
plt.title("Total steel production")
plt.xticks(rotation=0)
plt.show()
#
sns.set(palette=customPalette)
df_steel_emissions_tech.plot(kind="bar",stacked="True")
plt.xlabel("Year")
plt.ylabel("Emissions (Mt) ")
plt.title("Direct steel production emissions")
plt.xticks(rotation=0)
plt.show()

sns.set(palette=customPalette)
df_steel_capacity_tech.plot(kind="bar",stacked="True")
plt.xlabel("Year")
plt.ylabel("Capacity (Mt)")
plt.title("Installed steel production capacity")
plt.xticks(rotation=0)
plt.show()

df_steel_added_capacity_tech.plot(kind="bar",stacked="True")
plt.xlabel("Year")
plt.ylabel("Added capacity (Mt)")
plt.title("Yearly installed new steel production capacity ")
plt.xticks(rotation=0)
plt.show()

sns.set(palette=customPalette)
df_steel_cost_tech.plot(kind="bar",stacked="True")
plt.xlabel("Year")
plt.ylabel("Cost (B€)")
plt.title("Steel production technology annualised CAPEX")
plt.xticks(rotation=0)
plt.show()


##################
# Resource flows #
##################

resource_characteristics=pd.read_excel(input_path+"Resources_characteristics.xlsx").set_index("Resource")
Resources=["Coal","Biomass","Oil","Electricity","Gas","Hydrogen"]
Inflows={}
Outflows={}
df_flows=pd.DataFrame(columns=["Resource","Year","Inflows (TWh)","Outflows (TWh)"])
for year in Year_list:
    for resource in Resources:
        inflows= Results["V_resource_inflow[" + resource + "," + str(year) + ']'] * resource_characteristics.loc[resource, "calorific_value_MWh_t"]*1e-6
        outflows= -Results["V_resource_outflow[" + resource + "," + str(year) + ']'] * resource_characteristics.loc[resource, "calorific_value_MWh_t"]*1e-6
        df_flows=pd.concat ([df_flows,
                             pd.DataFrame(np.array([[resource,year,inflows,outflows]]),
                             columns=["Resource","Year","Inflows (TWh)","Outflows (TWh)"])
                             ],ignore_index=True)
df_flows["Inflows (TWh)"]=df_flows["Inflows (TWh)"].astype("float")
df_flows["Outflows (TWh)"]=df_flows["Outflows (TWh)"].astype("float")

df_inflows=df_flows.pivot(index='Year',columns="Resource",values="Inflows (TWh)")
df_outflows=df_flows.pivot(index='Year',columns="Resource",values="Outflows (TWh)")

ax=df_inflows.plot(kind="bar",stacked="True",grid=True)
df_outflows.plot(kind="bar",stacked="True",ax=ax,legend=False)
plt.ylabel("Flow (TWh)")
plt.xlabel("Year")
plt.grid(axis="x",which="major")
plt.xticks(rotation=0)
plt.title("Resource inflows and outflows through time")
plt.show()


##########################################################
# Hydrogen (incl. steam for SMR) and Gas self-production #
##########################################################

df_h2_gas_flows=pd.DataFrame(columns=["Resource-Technology","Year","Production (kt)"])


for tech in ["Gas_boiler","E_boiler","SMR","Electrolyser","Methanisation"]:
    for resource in ["Gas","Hydrogen","Steam"]:
        for year in Year_list:
            sign=1
            if resource=="Steam":
                sign=-1
            resource_tech=resource+"-"+tech
            prod=sign*Results["V_resource_tech_outflow["+tech+",0,"+resource+","+str(year)+']']*1e-3
            if prod!=0:
                df_h2_gas_flows=pd.concat([df_h2_gas_flows,
                                           pd.DataFrame(np.array([[resource_tech,year,prod]]),
                                           columns=["Resource-Technology","Year","Production (kt)"])
                                           ])
df_h2_gas_flows=df_h2_gas_flows.pivot(index='Year',columns="Resource-Technology",values="Production (kt)").astype("float")
# print(df_h2_gas_flows)
#
sns.set(palette=customPalette)
df_h2_gas_flows.plot(kind="bar",stacked="True")
plt.xlabel("Year")
plt.ylabel("Production (kt)")
plt.title("Hydrogen, Gas & Steam production")
plt.grid(axis="x",which="major")
plt.show()