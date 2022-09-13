
#region Chargement des packages
from IPython import get_ipython;
get_ipython().magic('reset -sf')
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
dim_names=["Categorie","year","Vecteur"];
Index_names = ["Categorie"];Energy_system_name="Categorie"
data_set_from_excel =  pd.read_excel(Data_folder+"Hypotheses_Transport_1D.xlsx", None);
sim_param = extract_sim_param(data_set_from_excel,Index_names = Index_names,dim_names=dim_names,Energy_system_name=Energy_system_name)

sim_param["init_sim_stock"]=create_initial_parc(sim_param).sort_index()
sim_param["volume_variable_name"] = "Mds_voy_km"
sim_param["retrofit_improvement"]=pd.DataFrame([sim_param["retrofit_improvement"]]*len(sim_param['base_index_year']),index =sim_param['base_index_year'])[0]
Para_2_fill = {param : sim_param["base_index_year"] for param in ["retrofit_change_Mds_voy_km","retrofit_Transition"]}
sim_param = interpolate_sim_param(sim_param)
sim_param["retrofit_change_Mds_voy_km"]=sim_param["retrofit_change_total_proportion_Mds_voy_km"].diff().fillna(0)
sim_param   =   complete_parameters(sim_param,Para_2_fill=Para_2_fill)
sim_param["retrofit_change_Mds_voy_km"]=sim_param["retrofit_change_Mds_voy_km"]*sim_param["init_sim_stock"]["Mds_voy_km"]
sim_param["init_sim_stock"]["energy_need_per_Mds_voy_km"] = sim_param["init_sim_stock"].apply(
                            lambda x: 1 /(100*x["remplissage"]), axis=1).fillna(0)


def f_Compute_conso(x,sim_param,Vecteur):
    return x[sim_param["volume_variable_name"]]*x["conso_unitaire_" + Vecteur]*x[sim_param["energy_need_variable_name"]]
def f_Compute_conso_totale(x,sim_param):
    res=0.
    for Vecteur in sim_param["Vecteurs"]:
        res+=x["conso_"+Vecteur]
    return res
def f_compute_emissions(x,sim_param,year,Vecteur):
    return sim_param["direct_emission"].loc[Vecteur,year]*x["conso_"+Vecteur]
def f_Compute_emissions_totale(x,sim_param):
    res=0.
    for Vecteur in sim_param["Vecteurs"]:
        res+=x["emissions_"+Vecteur]
    return res
#
for Vecteur in sim_param["Vecteurs"]:
    sim_param["f_Compute_conso_"+Vecteur]={"conso_"+Vecteur : partial(f_Compute_conso,Vecteur =Vecteur)}
    sim_param["f_Compute_emissions_" + Vecteur] = {"emissions_" + Vecteur: partial(f_compute_emissions, Vecteur=Vecteur)}
sim_param["f_Compute_conso_totale"]={"Conso" : lambda x,sim_param: f_Compute_conso_totale(x,sim_param)}
sim_param["f_Compute_emissions_totale"]={"emissions" : lambda x,sim_param: f_Compute_emissions_totale(x,sim_param)}


sim_param["retrofit_Transition"]=sim_param["retrofit_Transition"].fillna(0)
end = time.process_time()
print("Chargement des données, des modèles et interpolation terminés en : "+str(end-start)+" secondes")
#endregion

#region simulation
sim_stock = launch_simulation(sim_param)
#endregion


sim_stock_df = pd.concat(sim_stock, axis=0).reset_index().\
    rename(columns={"level_0":"year"}).set_index([ "year"  ,  Energy_system_name  , "old_new"])

Var = "Conso"
y_df = sim_stock_df.groupby(["year",Energy_system_name])[Var].sum().to_frame().reset_index().\
    pivot(index=['year'], columns=Energy_system_name).loc[[year for year in sim_param["years"][1:]],Var]
fig = MyStackedPlotly(y_df=y_df)
fig=fig.update_layout(title_text="Conso énergie finale par mode de transport (en TWh)", xaxis_title="Année",yaxis_title="Conso [TWh]")
plotly.offline.plot(fig, filename=Graphic_folder+'file.html') ## offline

Var = "Mds_voy_km"
y_df = sim_stock_df.groupby(["year",Energy_system_name])[Var].sum().to_frame().reset_index().\
    pivot(index=['year'], columns=Energy_system_name).loc[[year for year in sim_param["years"][1:]],Var]
fig = MyStackedPlotly(y_df=y_df)
fig=fig.update_layout(title_text="Mds_voy_km par mode de transport", xaxis_title="Année",yaxis_title="Mds_voy_km")
plotly.offline.plot(fig, filename=Graphic_folder+'file.html') ## offline

Var = "emissions"
y_df = sim_stock_df.groupby(["year",Energy_system_name])[Var].sum().to_frame().reset_index().\
    pivot(index=['year'], columns=Energy_system_name).loc[[year for year in sim_param["years"][1:]],Var]/10**3
fig = MyStackedPlotly(y_df=y_df)
fig=fig.update_layout(title_text="Emissions de GES par mode de chauffage (en MtCO2e)", xaxis_title="Année",yaxis_title="Conso [MtCO2e]")
plotly.offline.plot(fig, filename=Graphic_folder+'file.html') ## offline

### représentation par vecteur

y_df = sim_stock_df.groupby(["year"])[[ 'conso_'+Vecteur for Vecteur in sim_param["Vecteurs"]]].sum().loc[[year for year in sim_param["years"][1:]],:]
fig = MyStackedPlotly(y_df=y_df)
fig=fig.update_layout(title_text="Conso énergie finale par vecteur (en TWh)", xaxis_title="Année",yaxis_title="Conso [TWh]")
plotly.offline.plot(fig, filename=Graphic_folder+'file.html') ## offline

y_df = sim_stock_df.groupby(["year"])[[ 'emissions_'+Vecteur for Vecteur in sim_param["Vecteurs"]]].sum().loc[[year for year in sim_param["years"][1:]],:]
fig = MyStackedPlotly(y_df=y_df)
fig=fig.update_layout(title_text="Emissions par vecteur [MT CO2]", xaxis_title="Année",yaxis_title="CO2 [MT]")
plotly.offline.plot(fig, filename=Graphic_folder+'file.html') ## offline
