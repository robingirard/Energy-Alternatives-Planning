
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
dim_names=["Production_system","year","Vecteurs"];Index_names = ["Production_system"];Energy_system_name="Production_system"
data_set_from_excel =  pd.read_excel(Data_folder+"Hypotheses_acier_1D_QR.xlsx", None);
sim_param = extract_sim_param(data_set_from_excel,Index_names = Index_names,dim_names=dim_names,Energy_system_name=Energy_system_name)
sim_param["init_sim_stock"]=create_initial_parc(sim_param).sort_index()
sim_param["volume_variable_name"] = "unite_prod"

### example (un peu trop simpliste car ici on pourrait le faire formellement) de callage de paramètre
### modèle avec alpha paramètre libre : rénovation tous les ans de alpha *  sim_param["init_sim_stock"].loc[:, "surface"]
### cible : on veut que sur toute la période de simulation que cela fasse sim_param["retrofit_change"] * sim_param["init_sim_stock"]["Surface"]
def Error_function(alpha,sim_param):
    total_change_target = sim_param["retrofit_change_total_proportion_init_"+sim_param["volume_variable_name"]] * sim_param["init_sim_stock"][sim_param["volume_variable_name"]]
    Total_change = pd.Series(0.,index=sim_param["base_index"])
    for year in sim_param["years"]:
        Total_change+=alpha *  sim_param["init_sim_stock"].loc[:, sim_param["volume_variable_name"]]
    return  ((Total_change-total_change_target)**2).sum()

alpha = scop.minimize(Error_function, x0=1, method='BFGS',args=(sim_param))["x"][0]
sim_param["retrofit_change_"+sim_param["volume_variable_name"]] = alpha * sim_param["init_sim_stock"][sim_param["volume_variable_name"]]

Para_2_fill = {param : sim_param["base_index_year"] for param in ["retrofit_improvement","retrofit_change_unite_prod","retrofit_Transition","energy_need_per_unite_prod",
                                                                                                   "new_energy","new_yearly_repartition_per_Energy_source"]}
sim_param   =   complete_parameters(sim_param,Para_2_fill=Para_2_fill)


## initialize all "new_yearly_surface"
#sim_param["new_yearly_surface"]=sim_param["new_yearly_surface"]*sim_param["new_yearly_repartition_per_Energy_source"]
def f_Compute_conso(x,sim_param,Vecteur = "total"):
    if Vecteur=="total":
        conso_unitaire=0
        for Vecteur_ in sim_param["Vecteurs"]: conso_unitaire+=f_Compute_conso(x,sim_param,Vecteur =Vecteur_)
    else: conso_unitaire = x["conso_unitaire_"+Vecteur]
    return x[sim_param["volume_variable_name"]]*conso_unitaire
#
sim_param["f_Compute_conso"]={"Conso" : lambda x,sim_param: f_Compute_conso(x,sim_param,Vecteur ="total")}
for Vecteur in sim_param["Vecteurs"]:
    sim_param["f_Compute_conso_"+Vecteur]={"conso_"+Vecteur : lambda x,sim_param: f_Compute_conso(x,sim_param,Vecteur =Vecteur)}

def f_Compute_emissions(x,sim_param):
    emissions = 0
    for Vecteur_ in sim_param["Vecteurs"]: emissions += x["conso_unitaire_"+Vecteur]*float(sim_param["Emissions_scope_2_3"][Vecteur])
    emissions+=x["emissions_unitaire"]* x[sim_param["volume_variable_name"]]
    return emissions

sim_param["f_Compute_emissions"]={"Emissions" : lambda x,sim_param: f_Compute_emissions(x,sim_param)}

end = time.process_time()
print("Chargement des données, des modèles et interpolation terminés en : "+str(end-start)+" secondes")
#endregion

#region simulation
sim_stock = launch_simulation(sim_param)
#endregion

#region représentation des résultats

sim_stock_df = pd.concat(sim_stock, axis=0).reset_index().\
    rename(columns={"level_0":"year"}).set_index([ "year"  ,  Energy_system_name  , "old_new"])

Var = "Conso"
y_df = sim_stock_df.groupby(["year",Energy_system_name])[Var].sum().to_frame().reset_index().\
    pivot(index='year', columns=Energy_system_name).loc[[year for year in range(2021,2050)],Var]/10**9
fig = MyStackedPlotly(y_df=y_df)
fig=fig.update_layout(title_text="Conso énergie finale par mode de chauffage (en TWh)", xaxis_title="Année",yaxis_title="Conso [TWh]")
plotly.offline.plot(fig, filename=Graphic_folder+'file.html') ## offline


Var = "Emissions"
y_df = sim_stock_df.groupby(["year",Energy_system_name])[Var].sum().to_frame().reset_index().\
    pivot(index='year', columns=Energy_system_name).loc[[year for year in range(2021,2050)],Var]/10**6
fig = MyStackedPlotly(y_df=y_df)
fig=fig.update_layout(title_text="Emissions CO2 Mt", xaxis_title="Année",yaxis_title="Emissions [MtCO2]")
plotly.offline.plot(fig, filename=Graphic_folder+'file.html') ## offline

y_df = sim_stock_df.groupby(["year"])[["conso_"+Vec for Vec in sim_param["Vecteurs"]]].sum().loc[[year for year in range(2021,2050)],:]/10**9
fig = MyStackedPlotly(y_df=y_df)
fig=fig.update_layout(title_text="Conso énergie finale par vecteur (en TWh)", xaxis_title="Année",yaxis_title="Conso [TWh]")
plotly.offline.plot(fig, filename=Graphic_folder+'file.html') ## offline
#fig.show()

#endregion

