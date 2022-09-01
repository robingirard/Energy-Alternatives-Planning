
#region Chargement des packages
#from IPython import get_ipython;
#get_ipython().magic('reset -sf')
import pandas as pd
from functions.f_tools import *
from functions.f_graphicalTools import *
from Models.Prospective_conso.f_evolution_tools import *
from mycolorpy import colorlist as mcp
import numpy as np
import time
from functools import partial
dpe_colors = ['#009900', '#33cc33', '#B3FF00', '#e6e600', '#FFB300', '#FF4D00', '#FF0000',"#000000"]
Graphic_folder = "Models/Prospective_conso/Graphics/"
Data_folder = "Models/Prospective_conso/data/"
pd.options.display.width = 0
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
#endregion

#region chargement des données
start = time.process_time()
dim_names=["Energy_source","building_type","year"];Index_names = ["Energy_source","building_type"];Energy_system_name="Energy_source"
data_set_from_excel =  pd.read_excel(Data_folder+"Hypotheses_residential_tertiary_BASIC.xlsx", None);
sim_param = extract_sim_param(data_set_from_excel,Index_names = Index_names,dim_names=dim_names,Energy_system_name=Energy_system_name)
sim_param["init_sim_stock"]=create_initial_parc(sim_param).sort_index()
sim_param["volume_variable_name"] = "surface"
sim_param["init_sim_stock"]["surface"]=sim_param["init_sim_stock"]["surface"]*sim_param["init_sim_stock"]["IPONDL"]

Para_2_fill = {param : sim_param["base_index_year"] for param in ["retrofit_improvement","retrofit_change_surface","retrofit_Transition"]}
sim_param   =   complete_parameters(sim_param,Para_2_fill=Para_2_fill)

## initialize all "new_yearly_surface"
# sim_param["new_yearly_surface"]=sim_param["new_yearly_surface"]*sim_param["new_yearly_repartition_per_Energy_source"]

def f_Compute_conso(x,sim_param,Vecteur):
    return x[sim_param["volume_variable_name"]]*x["conso_unitaire_" + Vecteur]*x[sim_param["energy_need_variable_name"]]
def f_Compute_conso_totale(x,sim_param):
    res=0.
    for Vecteur in sim_param["Vecteurs"]:
        res+=x["conso_"+Vecteur]
    return res
#
for Vecteur in sim_param["Vecteurs"]:
    sim_param["f_Compute_conso_"+Vecteur]={"conso_"+Vecteur : partial(f_Compute_conso,Vecteur =Vecteur)}
sim_param["f_Compute_conso_totale"]={"Conso" : lambda x,sim_param: f_Compute_conso_totale(x,sim_param)}


#def f_Compute_besoin(x,sim_param): return x["energy_need_per_surface"] * x["surface"]
#sim_param["f_Compute_besoin"]={"energy_need" : f_Compute_besoin}

#A rajouter ci-dessous :
#def f_compute_emissions(x,sim_param):
#    emissions = 0
#    for vecteur_ in sim_param["vecteurs"]: emissions += x["conso_unitaire_"+vecteur]*float(sim_param["emissions_scope_2_3"][vecteur])
#    emissions+=x["emissions_unitaire"]* x[sim_param["volume_variable_name"]]
#    return emissions

#sim_param["f_Compute_emissions"]={"Emissions" : lambda x,sim_param: f_Compute_emissions(x,sim_param)}


end = time.process_time()
print("Chargement des données, des modèles et interpolation terminés en : "+str(end-start)+" secondes")
#endregion

#region simulation
sim_stock = launch_simulation(sim_param)
#endregion

#region représentation des résultats

sim_stock_df = pd.concat(sim_stock, axis=0).reset_index().\
    rename(columns={"level_0":"year"}).set_index([ "year"  ,  "Energy_source"  , "old_new"])

Var = "Conso"
y_df = sim_stock_df.groupby(["year","Energy_source"])[Var].sum().to_frame().reset_index().\
    pivot(index=['year'], columns='Energy_source').loc[[year for year in range(2021,2050)],Var]/10**9
fig = MyStackedPlotly(y_df=y_df)
fig=fig.update_layout(title_text="Conso énergie finale par mode de chauffage (en TWh)", xaxis_title="Année",yaxis_title="Conso [TWh]")
plotly.offline.plot(fig, filename=Graphic_folder+'file.html') ## offline

Var = "surface"
y_df = sim_stock_df.groupby(["year","Energy_source"])[Var].sum().to_frame().reset_index().\
    pivot(index='year', columns='Energy_source').loc[[year for year in range(2021,2050)],Var]/10**9
fig = MyStackedPlotly(y_df=y_df)
fig=fig.update_layout(title_text="Surface ", xaxis_title="Année",yaxis_title="Surface [Mm2]")
plotly.offline.plot(fig, filename=Graphic_folder+'file.html') ## offline

y_df = sim_stock_df.groupby(["year"])[ 'conso_elec', 'conso_gaz', 'conso_fioul', 'conso_bois'].sum().loc[[year for year in range(2021,2050)],:]/10**9
fig = MyStackedPlotly(y_df=y_df)
fig=fig.update_layout(title_text="Conso énergie finale par vecteur (en TWh)", xaxis_title="Année",yaxis_title="Conso [TWh]")
plotly.offline.plot(fig, filename=Graphic_folder+'file.html') ## offline
#fig.show()

#endregion
Var = "surface"
y_df = sim_stock_df.groupby(["year","Energy_source","old_new"])[Var].sum().to_frame().reset_index().\
    pivot(index=['year',"old_new"], columns='Energy_source').loc[[year for year in range(2021,2050)],Var]/10**9
y_df=y_df.loc[(slice(None),"new"),:].reset_index().drop(columns ="old_new").set_index(["year"])
fig = MyStackedPlotly(y_df=y_df)
fig=fig.update_layout(title_text="Surface ", xaxis_title="Année",yaxis_title="Surface [Mm2]")
plotly.offline.plot(fig, filename=Graphic_folder+'file.html') ## offline
