import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from Models.Industry_model.Industry_model_final import *

# Create an array with the colors you want to use
colors = ["#004776","#b8e1ff","#72c5fe","#2baaff","#f8b740","#005f9e","#000000",
          "#e7e7e7","#fef3e8","#e8f1ff","#ebf5ff","#c69131","#2087cb"]# Set your custom color palette
customPalette = sns.set_palette(sns.color_palette(colors))

input_path = "Models/Industry_model/Input/Steel/Data/"
Parameters_data= pd.read_excel("Models/Industry_model/Input/Steel/Data/parameters.xlsx",sheet_name=["TECHNOLOGIES","RESOURCES","TECHNOLOGIES_RESOURCES"])

Parameters={"TECHNOLOGIES_parameters" : Parameters_data["TECHNOLOGIES"].fillna(0).set_index('TECHNOLOGIES'),
            "RESOURCES_parameters" : Parameters_data["RESOURCES"].fillna(0).set_index('RESOURCES'),
            "TECHNOLOGIES_RESOURCES_parameters" : Parameters_data["TECHNOLOGIES_RESOURCES"].fillna(0).\
                melt(id_vars='TECHNOLOGIES', var_name="RESOURCES",value_name='conversion_factor').\
                set_index(['TECHNOLOGIES','RESOURCES'])
}
model=GetIndustryModel(Parameters,opti2mini="cost",carbon_tax=0)#emission
opt = SolverFactory('mosek')

results = opt.solve(model)

######################
# Results treatment  #
######################
# print("Print values for all variables")
Results = {}
for v in model.component_data_objects(Var):
    if v.name[:29] != 'V_primary_RESOURCES_production' and v.name[:23] != 'V_resource_tech_outflow' and \
            v.name[:22] != 'V_resource_tech_inflow' and v.name[:15] != 'V_resource_flow':
        # print(v,v.value)
        Results[v.name] = v.value

return Results

fig,ax=plt.subplots()
ax.set_title("Total cost and emissions")
ax.bar(["Cost (B€)"],[Results["V_cost"]/1e9])
ax.bar(["Emissions (Mt)"],Results["V_emissions"]/1e6)
for bars in ax.containers:
    ax.bar_label(bars)
plt.show()

fig,ax=plt.subplots()
ax.set_title("Cost and emissions per ton of steel")
ax.bar(["Cost (k€/tsteel)"],[Results["V_cost"]/Results["V_RESOURCES_outflow[steel]"]/1e3])
ax.bar(["Emissions (t/tsteel)"],Results["V_emissions"]/Results["V_RESOURCES_outflow[steel]"])
for bars in ax.containers:
    ax.bar_label(bars)
plt.show()


Consumption_MWh={}
Consumption_MWh["Oil"]=Results["V_technology_use_coef[Oil]"]*RESOURCESs_Characteristics.loc["oil","calorific_value_MWh_t"]
Consumption_MWh["Natural Gas"]=Results["V_technology_use_coef[Gas]"]*RESOURCESs_Characteristics.loc["gas","calorific_value_MWh_t"]
Consumption_MWh["Biogas"]=Results["V_technology_use_coef[Biogas]"]*RESOURCESs_Characteristics.loc["gas","calorific_value_MWh_t"]
Consumption_MWh["Coal"]=Results["V_technology_use_coef[Coal]"]*RESOURCESs_Characteristics.loc["coal","calorific_value_MWh_t"]
Consumption_MWh["Electricity"]=Results["V_technology_use_coef[Electricity]"]*RESOURCESs_Characteristics.loc["electricity","calorific_value_MWh_t"]

df=pd.DataFrame.from_dict(Consumption_MWh,orient="index",columns=['Consumption']).reset_index()
df["Consumption"]=df["Consumption"].astype("float")/1e6
df.rename(columns={"index":"RESOURCES"},inplace=True)
df["Year"]=2015
# fig = px.bar(df, x="Year", y="Consumption",color="RESOURCES")
# fig.show()



df=df.pivot(index="Year",columns="RESOURCES",values="Consumption")

sns.set(palette=customPalette)
df.plot(kind="bar",stacked="True")
plt.xlabel("Year")
plt.ylabel("Consumption (TWh)")
plt.show()



Available_Technologies = pd.read_excel(input_path + "Steel_available_techs_opti_test.xlsx").fillna(0)
Techno_list=Available_Technologies["Technologies"].to_numpy()[3:].tolist()

Result_dict={}
Consumption_df=pd.DataFrame(columns=["Carbon tax (€/tCO2)","Biogas","Coal","Electricity","Natural Gas","Oil"]).set_index("Carbon tax (€/tCO2)")
Hydrogen_df=pd.DataFrame(columns=["Carbon tax (€/tCO2)","SMR","Electrolyser"]).set_index("Carbon tax (€/tCO2)")
Steel_prod_techs_df=pd.DataFrame(columns=["Carbon tax (€/tCO2)"]+Techno_list).set_index("Carbon tax (€/tCO2)")
Costs={}
Emissions={}

for carbon_tax in [0,50,100,200,300,400]:
    Results=main(RESOURCESs_Technologies,Production_Technologies,Available_Technologies,Production,opti2mini="cost",carbon_tax=carbon_tax)
    Result_dict[carbon_tax]=Results

    #Consumption

    Consumption_MWh = {}
    Consumption_MWh["Oil"] = Results["V_technology_use_coef[Oil]"] * RESOURCESs_Characteristics.loc[
        "oil", "calorific_value_MWh_t"]
    Consumption_MWh["Natural Gas"] = Results["V_technology_use_coef[Gas]"] * RESOURCESs_Characteristics.loc[
        "gas", "calorific_value_MWh_t"]
    Consumption_MWh["Biogas"] = Results["V_technology_use_coef[Biogas]"] * RESOURCESs_Characteristics.loc[
        "gas", "calorific_value_MWh_t"]
    Consumption_MWh["Coal"] = Results["V_technology_use_coef[Coal]"] * RESOURCESs_Characteristics.loc[
        "coal", "calorific_value_MWh_t"]
    Consumption_MWh["Electricity"] = Results["V_technology_use_coef[Electricity]"] * RESOURCESs_Characteristics.loc[
        "electricity", "calorific_value_MWh_t"]

    df = pd.DataFrame.from_dict(Consumption_MWh, orient="index", columns=['Consumption']).reset_index()
    df["Consumption"] = df["Consumption"].astype("float") / 1e6
    df.rename(columns={"index": "RESOURCES"}, inplace=True)
    df["Carbon tax (€/tCO2)"] = carbon_tax
    df = df.pivot(index="Carbon tax (€/tCO2)", columns="RESOURCES", values="Consumption")
    Consumption_df=pd.concat([Consumption_df,df])

    #Hydrogen production
    Hydrogen={}
    for tech in Hydrogen_df.columns:
        Hydrogen[tech]=Results["V_technology_use_coef["+tech+"]"]*RESOURCESs_Characteristics.loc[
        "hydrogen", "calorific_value_MWh_t"]
    df = pd.DataFrame.from_dict(Hydrogen, orient="index", columns=['Production']).reset_index()
    df["Production"] = df["Production"].astype("float") / 1e6
    df.rename(columns={"index": "Technology"}, inplace=True)
    df["Carbon tax (€/tCO2)"] = carbon_tax
    df = df.pivot(index="Carbon tax (€/tCO2)", columns="Technology", values="Production")
    Hydrogen_df=pd.concat([Hydrogen_df,df])

    # Cost of steel
    Costs[str(carbon_tax)]=Results["V_cost"]/Results["V_RESOURCES_outflow[steel]"]
    # Emissions
    Emissions[str(carbon_tax)]=Results["V_emissions"]/Results["V_RESOURCES_outflow[steel]"]

    #Technologies for production
    Techs={}
    for tech in Techno_list:
        Techs[tech]=Results["V_technology_use_coef["+tech+"]"]*-1*Production_Technologies.loc[Production_Technologies.RESOURCES=="steel",tech].values[0]
    df = pd.DataFrame.from_dict(Techs, orient="index", columns=['Production']).reset_index()
    df["Production"] = df["Production"].astype("float") / 1e6
    df.rename(columns={"index": "Technology"}, inplace=True)
    df["Carbon tax (€/tCO2)"] = carbon_tax
    df = df.pivot(index="Carbon tax (€/tCO2)", columns="Technology", values="Production")
    Steel_prod_techs_df = pd.concat([Steel_prod_techs_df, df])

###Cost & emissions comparison
fig,ax=plt.subplots()
ax.set_title("Cost per ton of steel (40% recycling)")
ax.bar(Costs.keys(),Costs.values())
for bars in ax.containers:
    ax.bar_label(bars)
plt.xlabel("Carbon tax (€/tCO2)")
plt.ylabel("Cost (€/tsteel)")
plt.show()

fig,ax=plt.subplots()
ax.set_title("Emissions per ton of steel (40% recycling)")
ax.bar(Emissions.keys(),Emissions.values())
for bars in ax.containers:
    ax.bar_label(bars)
plt.xlabel("Carbon tax (€/tCO2)")
plt.ylabel("Emissions (tCO2/tsteel)")
plt.show()
###Energy consumption
sns.set(palette=customPalette)
plt.figure(figsize=(15,15))
Consumption_df.plot(kind="bar",stacked="True")
plt.xlabel("Carbon tax (€/tCO2)")
plt.ylabel("Consumption (TWh)")
plt.xticks(rotation=0)
plt.show()
###Technologies for primary steel
sns.set(palette=customPalette)
Steel_prod_techs_df.plot(kind="bar",stacked="True")
plt.xlabel("Carbon tax (€/tCO2)")
plt.ylabel("Steel production (t)")
plt.xticks(rotation=0)
plt.show()
###Hydrogen production technologies
sns.set(palette=customPalette)
Hydrogen_df.plot(kind="bar",stacked="True")
plt.xlabel("Carbon tax (€/tCO2)")
plt.ylabel("Hydrogen production (TWh)")
plt.xticks(rotation=0)
plt.show()