#region Chargement des packages
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

sheet_name_and_dim={"Energy_source" : ["Energy_source"],
                    "retrofit_Transition" : ["Energy_source","year"] ,"0D" : [],
                    "Energy_source_year": ["Energy_source","year"],"year":["year"]}
data_set_from_excel =  pd.read_excel(Data_folder+"Hypotheses_tertiaire_1D.xlsx",
                            sheet_name=list(sheet_name_and_dim.keys()))
simulation_parameters = extract_simulation_parameters(data_set_from_excel,sheet_name_and_dim,Index_names = ["Energy_source"])
simulation_parameters["type_index"] = data_set_from_excel["Energy_source"]["Energy_source"] ### l'index des archetypes
simulation_parameters   =   complete_parameters(simulation_parameters,Para_2_fill=["retrofit_improvement","retrofit_Transition"])
simulation_parameters["init_Description_Parc"]=create_initial_parc(simulation_parameters).sort_index()

end = time.process_time()
print("Chargement et interpolation des données terminé en : "+str(end-start)+" secondes")
#endregion

#region models
# ce dont on a besoin à la fin pour la simulation
# une formule pour calculer la conso et les émissions
start = time.process_time()

def f_Compute_conso(x,Vecteur = "total"):
    if Vecteur=="total":
        conso_unitaire = x["conso_unitaire_elec"]+x["conso_unitaire_gaz"]+x["conso_unitaire_fioul"]+x["conso_unitaire_bois"]
    else: conso_unitaire = x["conso_unitaire_"+Vecteur]
    return x["Besoin_surfacique"] * x["Surface"]*x["proportion_besoin_chauffage"]*conso_unitaire
simulation_parameters["f_Compute_conso"]=f_Compute_conso

def f_Compute_besoin(x): return x["Besoin_surfacique"] * x["Surface"]*x["proportion_besoin_chauffage"]
simulation_parameters["f_Compute_besoin"]=f_Compute_besoin


## initialize all "new_yearly_surface"
simulation_parameters["new_yearly_surface"]= pd.Series(0.,index=simulation_parameters["type_index_year_new"])
simulation_parameters["new_energy"]= pd.Series(simulation_parameters["new_energy"],index=simulation_parameters["type_index_year_new"] )
TMP = ((simulation_parameters["new_yearly_surface_total"]*simulation_parameters["new_yearly_repartition_per_Energy_source"])).to_frame()
TMP.columns =["new_yearly_surface"];TMP=TMP.assign(old_new="new").set_index(["old_new"],append=True)["new_yearly_surface"]
simulation_parameters["new_yearly_surface"][TMP.index]+=TMP

def f_retrofit_total_yearly_surface(year,alpha,simulation_parameters):
    ## Ici ne dépend pas de year
    return  alpha *  simulation_parameters["init_Description_Parc"].loc[:, "Surface"]
simulation_parameters["f_retrofit_total_yearly_surface"]=f_retrofit_total_yearly_surface
calibration_parameters={"f_retrofit_total_yearly_surface" :
                            {"total_change_target": simulation_parameters["retrofit_change"] *
                                                    simulation_parameters["init_Description_Parc"]["Surface"]}}
simulation_parameters=calibrate_simulation_parameters(simulation_parameters,calibration_parameters)
end = time.process_time()
print("Chargement et calibration des modèles terminé en : "+str(end-start)+" secondes")
#endregion

#region simulation et représentation des résultats
Description_Parc= loanch_simulation(simulation_parameters)

Description_Parc_df = pd.concat(Description_Parc, axis=0).reset_index().\
    rename(columns={"level_0":"year"}).set_index([ "year"  ,  "Energy_source"  , "old_new"])

Var = "Conso"
y_df = Description_Parc_df.groupby(["year","Energy_source"])[Var].sum().to_frame().reset_index().\
    pivot(index='year', columns='Energy_source').loc[[year for year in range(2021,2050)],Var]/10**9
fig = MyStackedPlotly(y_df=y_df)
fig=fig.update_layout(title_text="Conso énergie finale par mode de chauffage (en TWh)", xaxis_title="Année",yaxis_title="Conso [TWh]")
plotly.offline.plot(fig, filename=Graphic_folder+'file.html') ## offline

Var = "Besoin"
y_df = Description_Parc_df.groupby(["year","Energy_source"])[Var].sum().to_frame().reset_index().\
    pivot(index='year', columns='Energy_source').loc[[year for year in range(2021,2050)],Var]/10**9
fig = MyStackedPlotly(y_df=y_df)
fig=fig.update_layout(title_text="Besoin de chaleur par mode de chauffage (en TWh)", xaxis_title="Année",yaxis_title="Besoin [TWh]")
plotly.offline.plot(fig, filename=Graphic_folder+'file.html') ## offline

y_df = Description_Parc_df.groupby(["year"])[ 'Conso_elec', 'Conso_gaz', 'Conso_fioul', 'Conso_bois'].sum().loc[[year for year in range(2021,2050)],:]/10**9
fig = MyStackedPlotly(y_df=y_df)
fig=fig.update_layout(title_text="Conso énergie finale par vecteur (en TWh)", xaxis_title="Année",yaxis_title="Conso [TWh]")
plotly.offline.plot(fig, filename=Graphic_folder+'file.html') ## offline


#endregion
