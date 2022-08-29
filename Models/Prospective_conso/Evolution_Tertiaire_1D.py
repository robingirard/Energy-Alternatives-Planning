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

data_set_from_excel =  pd.read_excel(Data_folder+"Hypotheses_tertiaire_1D.xlsx", None);
sim_param = extract_sim_param(data_set_from_excel,Index_names = ["Energy_source"],
                              dim_names=["Energy_source","year"])
sim_param["init_sim_stock"]=create_initial_parc(sim_param).sort_index()

### example (un peu trop simpliste car ici on pourrait le faire formellement) de callage de paramètre
### modèle avec alpha paramètre libre : rénovation tous les ans de alpha *  sim_param["init_sim_stock"].loc[:, "Surface"]
### cible : on veut que sur toute la période de simulation que cela fasse sim_param["retrofit_change"] * sim_param["init_sim_stock"]["Surface"]
def Error_function(alpha,sim_param):
    total_change_target = sim_param["retrofit_change_total_proportion_surface"] * sim_param["init_sim_stock"]["Surface"]
    Total_change = pd.Series(0.,index=sim_param["base_index"])
    for year in range(int(sim_param["date_debut"])+1,int(sim_param["date_fin"])):
        Total_change+=alpha *  sim_param["init_sim_stock"].loc[:, "Surface"]
    return  ((Total_change-total_change_target)**2).sum()

alpha = scop.minimize(Error_function, x0=1, method='BFGS',args=(sim_param))["x"][0]
sim_param["retrofit_change_surface"] = alpha * sim_param["init_sim_stock"]["Surface"]
Pameter_to_fill_along_index_year = {param : sim_param["base_index_year"] for param in ["retrofit_improvement","retrofit_change_surface","retrofit_Transition",
                                                                                                   "new_energy","new_yearly_repartition_per_Energy_source"]}
sim_param   =   complete_parameters(sim_param,Para_2_fill=Pameter_to_fill_along_index_year)

## initialize all "new_yearly_surface"
sim_param["new_yearly_surface"]=sim_param["new_yearly_surface"]*sim_param["new_yearly_repartition_per_Energy_source"]

def f_Compute_conso(x,Vecteur = "total"):
    if Vecteur=="total":
        conso_unitaire = x["conso_unitaire_elec"]+x["conso_unitaire_gaz"]+x["conso_unitaire_fioul"]+x["conso_unitaire_bois"]
    else: conso_unitaire = x["conso_unitaire_"+Vecteur]
    return x["Besoin_surfacique"] * x["Surface"]*x["proportion_besoin_chauffage"]*conso_unitaire
sim_param["f_Compute_conso"]=f_Compute_conso

def f_Compute_besoin(x): return x["Besoin_surfacique"] * x["Surface"]*x["proportion_besoin_chauffage"]
sim_param["f_Compute_besoin"]=f_Compute_besoin

end = time.process_time()
print("Chargement des données, des modèles et interpolation terminés en : "+str(end-start)+" secondes")
#endregion

#region simulation
sim_stock = loanch_simulation(sim_param)
#endregion

#region représentation des résultats

sim_stock_df = pd.concat(sim_stock, axis=0).reset_index().\
    rename(columns={"level_0":"year"}).set_index([ "year"  ,  "Energy_source"  , "old_new"])

Var = "Conso"
y_df = sim_stock_df.groupby(["year","Energy_source"])[Var].sum().to_frame().reset_index().\
    pivot(index='year', columns='Energy_source').loc[[year for year in range(2021,2050)],Var]/10**9
fig = MyStackedPlotly(y_df=y_df)
fig=fig.update_layout(title_text="Conso énergie finale par mode de chauffage (en TWh)", xaxis_title="Année",yaxis_title="Conso [TWh]")
plotly.offline.plot(fig, filename=Graphic_folder+'file.html') ## offline

Var = "Besoin"
y_df = sim_stock_df.groupby(["year","Energy_source"])[Var].sum().to_frame().reset_index().\
    pivot(index='year', columns='Energy_source').loc[[year for year in range(2021,2050)],Var]/10**9
fig = MyStackedPlotly(y_df=y_df)
fig=fig.update_layout(title_text="Besoin de chaleur par mode de chauffage (en TWh)", xaxis_title="Année",yaxis_title="Besoin [TWh]")
plotly.offline.plot(fig, filename=Graphic_folder+'file.html') ## offline

y_df = sim_stock_df.groupby(["year"])[ 'Conso_elec', 'Conso_gaz', 'Conso_fioul', 'Conso_bois'].sum().loc[[year for year in range(2021,2050)],:]/10**9
fig = MyStackedPlotly(y_df=y_df)
fig=fig.update_layout(title_text="Conso énergie finale par vecteur (en TWh)", xaxis_title="Année",yaxis_title="Conso [TWh]")
plotly.offline.plot(fig, filename=Graphic_folder+'file.html') ## offline


#endregion
