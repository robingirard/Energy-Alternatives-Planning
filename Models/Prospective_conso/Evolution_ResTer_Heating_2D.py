# region Chargement des packages
# from IPython import get_ipython;
# get_ipython().magic('reset -sf')
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
from functions.f_tools import *
from functions.f_graphicalTools import *
from Models.Prospective_conso.f_evolution_tools import *
import plotly.express as px
from mycolorpy import colorlist as mcp
import qgrid  # great package https://github.com/quantopian/qgrid
import numpy as np
import time
from functools import reduce

# print(os.getcwd())
dpe_colors = ['#009900', '#33cc33', '#B3FF00', '#e6e600', '#FFB300', '#FF4D00', '#FF0000', "#000000"]
Graphic_folder = "Models/Prospective_conso/Graphics/"
Data_folder = "Models/Prospective_conso/data/"
pd.options.display.width = 0
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# endregion


# Four dimensions in the dataset
dim_names = ["Energy_source", "building_type", "Vecteur", "year"];
# Two main indexes
Index_names = ["Energy_source", "building_type"];
Energy_system_name = "Energy_source"
# Reading the data
data_set_from_excel = pd.read_excel(Data_folder + "Hypotheses_ResTer_Heating_2D.xlsx", None);
# Extracting info from sheets and creating indexes etc
sim_param = extract_sim_param(data_set_from_excel, Index_names=Index_names, dim_names=dim_names,
                              Energy_system_name=Energy_system_name)
# Creating the initial building description
sim_param["init_sim_stock"] = create_initial_parc(sim_param).sort_index()
sim_param["volume_variable_name"] = "surface"
sim_param["init_sim_stock"]["surface"] = sim_param["init_sim_stock"]["surface"] * sim_param["init_sim_stock"]["IPONDL"]

# We adjust energy need for electricity considering the primary energy factor
# sim_param["conso_unitaire_elec"]=sim_param["conso_unitaire_elec"]/2.3

# When data is not given for every numeric index (typically years), we interpolate
sim_param = interpolate_sim_param(sim_param)
sim_param["retrofit_change_surface"] = sim_param["retrofit_change_total_proportion_surface"].diff().fillna(0)

# We fill the missing parameters/indexes
Para_2_fill = {param: sim_param["base_index_year"] for param in
               ["retrofit_improvement", "retrofit_change_surface", "retrofit_Transition", "new_yearly_surface",
                "new_energy"]}
sim_param = complete_parameters(sim_param, Para_2_fill=Para_2_fill)

sim_param["retrofit_change_surface"] = sim_param["retrofit_change_surface"] *sim_param["init_sim_stock"]["surface"]*(1-sim_param["old_taux_disp"].sum())

# When data is not given for every string index (typically vecteurs), we complete
sim_param = complete_missing_indexes(data_set_from_excel, sim_param, Index_names, dim_names)

## We define some functions which will calculate at each time step the energy need, consumption, emissions...
sim_param = set_model_functions(sim_param)

# We lanch the simulation
sim_stock = launch_simulation(sim_param)

sim_stock_df = pd.concat(sim_stock, axis=0).reset_index(). \
    rename(columns={"level_0": "year"}).set_index(["year", "Energy_source", "building_type", "old_new"])

col_class_dict = {'elec': 1, 'bois': 2, 'ordures': 2, 'autres': 2, 'gaz': 3, 'fioul': 3,
                  'charbon': 3}

Var = "conso"
all_columns = [Var + "_" + Vec for Vec in sim_param['Vecteurs']]
y_df = sim_stock_df.loc[(2021, slice(None), slice(None))].reset_index().set_index(
    ['Energy_source', 'building_type', 'old_new']).filter(all_columns)
y_df = pd.melt(y_df, value_name=Var, ignore_index=False)
y_df[['v1', 'Vecteur']] = y_df['variable'].str.split('_', expand=True)
y_df = y_df.drop(['v1', 'variable'], axis=1)
y_df = y_df.reset_index().groupby(['building_type', 'Vecteur']).sum().reset_index()
# y_df = y_df.loc[y_df["year"]==2021]
y_df[Var] = y_df[Var] / 10 ** 9
color_dict = gen_grouped_color_map(col_class_dict)
y_df["class"] = [col_class_dict[cat] for cat in y_df["Vecteur"]]
y_df = y_df.sort_values(by=['class'])

locals()[Var] = y_df.copy()

fig = px.bar(y_df, x="building_type", y=Var, color="Vecteur", title="Wide-Form Input", color_discrete_map=color_dict)
fig = fig.update_layout(title_text="Conso énergie finale par mode de transport (en TWh)", xaxis_title="Categorie",
                        yaxis_title="Conso [TWh]")
plotly.offline.plot(fig, filename=Graphic_folder + 'file.html')  ## offline

Var = "Conso"
y_df = sim_stock_df.groupby(["year", Energy_system_name])[Var].sum().to_frame().reset_index(). \
    pivot(index=['year'], columns=Energy_system_name).loc[[year for year in sim_param["years"][1:]], Var]
tmp=sim_stock_df.copy().reset_index()
tmp=tmp[tmp["year"]==2049]
fig = MyStackedPlotly(y_df=y_df)
fig = fig.update_layout(title_text="Conso énergie finale par mode de transport (en TWh)", xaxis_title="Année",
                        yaxis_title="Conso [TWh]")
# plotly.offline.plot(fig, filename=Graphic_folder+'file.html') ## offline
plotly.offline.plot(fig, filename=Graphic_folder + 'file.html')  ## offline

Var = "emissions"
y_df = sim_stock_df.groupby(["year", Energy_system_name])[Var].sum().to_frame().reset_index(). \
    pivot(index=['year'], columns=Energy_system_name).loc[[year for year in sim_param["years"][1:]], Var]
fig = MyStackedPlotly(y_df=y_df)
fig = fig.update_layout(title_text="Conso énergie finale par mode de transport (en TWh)", xaxis_title="Année",
                        yaxis_title="Conso [TWh]")
# plotly.offline.plot(fig, filename=Graphic_folder+'file.html') ## offline
plotly.offline.plot(fig, filename=Graphic_folder + 'file.html')  ## offline
