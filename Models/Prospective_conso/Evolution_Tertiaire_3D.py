
#region Chargement des packages
from IPython import get_ipython;
get_ipython().magic('reset -sf')
import pandas as pd
from functions.f_graphicalTools import *
from functions.f_tools import *
from mycolorpy import colorlist as mcp
from Models.Prospective_conso.f_evolution_tools import *
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
data_set_from_excel     =   pd.read_excel(Data_folder+"Hypotheses_tertiaire_3D.xlsx", None);
sim_param = extract_sim_param(data_set_from_excel,Index_names =  ["Categories", "Efficiency_class" ,"Energy_source"],
                              dim_names=["Categories","Energy_source","Efficiency_class","year","year"])
sim_param["init_sim_stock"]=create_initial_parc(sim_param).sort_index()
sim_param["volume_variable_name"] = "surface"
def f_Compute_surface(x):
    return x["Energy_source_per_Category"]*x["Efficiency_class_per_Category"]*x["total_surface"]
sim_param["init_sim_stock"]=sim_param["init_sim_stock"].assign(surface = lambda x : f_Compute_surface(x))

sim_param["energy_class_dictionnary" ]= {"A" : 50,"B" : 90, "C": 150,"D":230,"E":330,"F":450}
def energy_to_class(x,energy_class_dictionnary):
    for key in energy_class_dictionnary.keys():
        if x<=energy_class_dictionnary[key]:
            return key
    return "G"
sim_param["f_energy_to_class"]=energy_to_class
### example (un peu trop simpliste car ici on pourrait le faire formellement) de callage de paramètre
### modèle avec alpha paramètre libre : rénovation tous les ans de alpha *  sim_param["init_sim_stock"].loc[:, "Surface"]
### cible : on veut que sur toute la période de simulation que cela fasse sim_param["retrofit_change"] * sim_param["init_sim_stock"]["Surface"]
def Error_function(alpha,sim_param):
    total_change_target = sim_param["retrofit_change_total_proportion_surface"] * sim_param["init_sim_stock"]["surface"]
    Total_change = pd.Series(0.,index=sim_param["base_index"])
    for year in range(int(sim_param["date_debut"])+1,int(sim_param["date_fin"])):
        Total_change+=alpha *  sim_param["init_sim_stock"].loc[:, "surface"]
    return  ((Total_change-total_change_target)**2).sum()

alpha = scop.minimize(Error_function, x0=1, method='BFGS',args=(sim_param))["x"][0]
sim_param["retrofit_change_surface"] = alpha * sim_param["init_sim_stock"]["surface"]

Para_2_fill = {param : sim_param["base_index_year"] for param in ["retrofit_improvement","retrofit_change_surface","retrofit_Transition",
                                                                                                   "new_energy","new_yearly_repartition"]}
sim_param   =   complete_parameters(sim_param,Para_2_fill=Para_2_fill)


## initialize all "new_yearly_surface"
sim_param["new_yearly_surface"]=sim_param["new_yearly_surface"]*sim_param["new_yearly_repartition"]

def f_Compute_conso(x,Vecteur = "total"):
    if Vecteur=="total":
        conso_unitaire = x["conso_unitaire_elec"]+x["conso_unitaire_gaz"]+x["conso_unitaire_fioul"]+x["conso_unitaire_bois"]
    else: conso_unitaire = x["conso_unitaire_"+Vecteur]
    return x["energy_need_per_surface"] * x["surface"]*x["proportion_besoin_chauffage"]*conso_unitaire
sim_param["f_Compute_conso"]=f_Compute_conso

def f_Compute_besoin(x): return x["energy_need_per_surface"] * x["surface"]*x["proportion_besoin_chauffage"]
sim_param["f_Compute_besoin"]=f_Compute_besoin

end = time.process_time()
print("Chargement et interpolation des données terminé en : "+str(end-start)+" secondes")
#endregion

#region simulation
sim_stock = launch_simulation(sim_param)
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

#region graphique de Marrimenko
colors = mcp.gen_color(cmap='Accent',n=8)
# see https://matplotlib.org/stable/gallery/color/colormap_reference.html
fig=marimekko(sim_param["init_sim_stock"][["surface"]].reset_index().groupby(['Categories',"Energy_source"], as_index=False).sum(),
                x_var_name = "Energy_source",
                y_var_name= "Categories",
                effectif_var_name='surface',
              color_discrete_sequence=colors )
plotly.offline.plot(fig, filename=Graphic_folder+'surface_Energy_source_Categories.html')

fig=marimekko(sim_param["init_sim_stock"][["surface"]].reset_index().groupby(['Efficiency_class',"Categories"], as_index=False).sum(),
                x_var_name = "Categories",
                y_var_name= "Efficiency_class",
                effectif_var_name='surface',
              color_discrete_sequence= dpe_colors)
plotly.offline.plot(fig, filename=Graphic_folder+'surface_Categories_Efficiency_class.html')

fig=marimekko(sim_param["init_sim_stock"][["surface"]].reset_index().groupby(['Efficiency_class',"Energy_source"], as_index=False).sum(),
                x_var_name = "Energy_source",
                y_var_name= "Efficiency_class",
                effectif_var_name='surface',
              color_discrete_sequence= dpe_colors)
plotly.offline.plot(fig, filename=Graphic_folder+'surface_Energy_source_Efficiency_class.html')

#not working
# fig = marimekko_2(df =sim_stock_initiale[["surface"]].reset_index().groupby(['Categories',"Efficiency_class","Energy_source"], as_index=False).sum() ,
#             effectif_var_name = "surface",
#             ColorY_var_name="Categories",
#             horizontalX_var_name="Energy_source",
#             TextureX_var_name="Efficiency_class",
#             color_discrete_sequence=colors)
# plotly.offline.plot(fig, filename=Graphic_folder+'surface_Energy_source_Categories.html')

#endregion



