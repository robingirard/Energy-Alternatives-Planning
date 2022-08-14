import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from Models.Industry_model.Industry_model_planification import *

# Create an array with the colors you want to use
colors = ["#004776","#b8e1ff","#72c5fe","#2baaff","#f8b740","#005f9e","#000000",
          "#e7e7e7","#fef3e8","#e8f1ff","#ebf5ff","#c69131","#2087cb"]# Set your custom color palette
customPalette = sns.set_palette(sns.color_palette(colors))

input_path = "Models/Industry_model/Input/Steel/Data/"
Parameters_data= pd.read_excel("Models/Industry_model/Input/Steel/Data/parameters_planification.xlsx",sheet_name=["TECHNOLOGIES","RESOURCES","TECHNOLOGIES_RESOURCES","TECHNOLOGIES_capacity"])

Parameters={"TECHNOLOGIES_parameters" : Parameters_data["TECHNOLOGIES"].fillna(0).set_index(['TECHNOLOGIES','TECH_TYPE','YEAR']),
            "RESOURCES_parameters" : Parameters_data["RESOURCES"].fillna(0).set_index(['RESOURCES','YEAR']),
            "TECHNOLOGIES_RESOURCES_parameters" : Parameters_data["TECHNOLOGIES_RESOURCES"].fillna(0).\
                melt(id_vars=['TECHNOLOGIES','TECH_TYPE','YEAR'], var_name="RESOURCES",value_name='conversion_factor').\
                set_index(['TECHNOLOGIES','TECH_TYPE','RESOURCES','YEAR']),
            "TECHNOLOGIES_capacity_parameters":Parameters_data["TECHNOLOGIES_capacity"].fillna(0).set_index(["TECHNOLOGIES","YEAR"])
}
# print(Parameters["TECHNOLOGIES_RESOURCES_parameters"])

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
    if v.value!=0 and v.value != None:
        Results[v.name] = v.value
        print(v.name,v.value)

# print(Results)

# fig,ax=plt.subplots()
# ax.set_title("Total cost and emissions")
# ax.bar(["Cost (B€)"],[Results["V_cost"]/1e9])
# ax.bar(["Emissions (Mt)"],Results["V_emissions"]/1e6)
# for bars in ax.containers:
#     ax.bar_label(bars)
# plt.show()
#
# fig,ax=plt.subplots()
# ax.set_title("Cost and emissions per ton of steel")
# ax.bar(["Cost (k€/tsteel)"],[Results["V_cost"]/Results["V_resource_outflow[Steel]"]/1e3])
# ax.bar(["Emissions (t/tsteel)"],Results["V_emissions"]/Results["V_resource_outflow[Steel]"])
# for bars in ax.containers:
#     ax.bar_label(bars)
# plt.show()