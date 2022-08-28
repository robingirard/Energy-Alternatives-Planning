#region Chargement des packages
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
sheet_name_and_dim={"Categories":["Categories"],
                    "Categories_year":["Categories","year"],
                    "Efficiency_class":["Efficiency_class"],
                    "Energy_source" : ["Energy_source"],
                    "retrofit_Transition" : ["Categories","Energy_source","year"] ,
                    "Categories_Energy_source":["Categories","Energy_source"],
                    "Categories_Efficiency_class":["Categories","Efficiency_class"],
                    "Categories_Energy_source_year":["Categories","Energy_source","year"],
                    "year":["year"],
                    "0D":[]}#"Energy_source_year": ["Energy_source","year"],

data_set_from_excel     =   pd.read_excel(Data_folder+"Hypotheses_tertiaire_3D.xlsx", sheet_name=list(sheet_name_and_dim.keys()))
simulation_parameters   =   extract_simulation_parameters(data_set_from_excel,sheet_name_and_dim,Index_names = ["Categories", "Efficiency_class" ,"Energy_source"])
simulation_parameters   =   complete_parameters(simulation_parameters,Para_2_fill=["retrofit_improvement","retrofit_Transition"])
simulation_parameters["init_Description_Parc"]  =   create_initial_parc(simulation_parameters).sort_index()

def f_Compute_surface(x):
    return x["Energy_source_per_Category"]*x["Efficiency_class_per_Category"]*x["total_surface"]
simulation_parameters["init_Description_Parc"]=simulation_parameters["init_Description_Parc"].assign(Surface = lambda x : f_Compute_surface(x))
end = time.process_time()
print("Chargement et interpolation des données terminé en : "+str(end-start)+" secondes")
#endregion

#region graphique de Marrimenko
colors = mcp.gen_color(cmap='Accent',n=8)
# see https://matplotlib.org/stable/gallery/color/colormap_reference.html
fig=marimekko(simulation_parameters["init_Description_Parc"][["Surface"]].reset_index().groupby(['Categories',"Energy_source"], as_index=False).sum(),
                x_var_name = "Energy_source",
                y_var_name= "Categories",
                effectif_var_name='Surface',
              color_discrete_sequence=colors )
plotly.offline.plot(fig, filename=Graphic_folder+'surface_Energy_source_Categories.html')

fig=marimekko(simulation_parameters["init_Description_Parc"][["Surface"]].reset_index().groupby(['Efficiency_class',"Categories"], as_index=False).sum(),
                x_var_name = "Categories",
                y_var_name= "Efficiency_class",
                effectif_var_name='Surface',
              color_discrete_sequence= dpe_colors)
plotly.offline.plot(fig, filename=Graphic_folder+'surface_Categories_Efficiency_class.html')

fig=marimekko(simulation_parameters["init_Description_Parc"][["Surface"]].reset_index().groupby(['Efficiency_class',"Energy_source"], as_index=False).sum(),
                x_var_name = "Energy_source",
                y_var_name= "Efficiency_class",
                effectif_var_name='Surface',
              color_discrete_sequence= dpe_colors)
plotly.offline.plot(fig, filename=Graphic_folder+'surface_Energy_source_Efficiency_class.html')

#not working
# fig = marimekko_2(df =Description_Parc_initiale[["Surface"]].reset_index().groupby(['Categories',"Efficiency_class","Energy_source"], as_index=False).sum() ,
#             effectif_var_name = "Surface",
#             ColorY_var_name="Categories",
#             horizontalX_var_name="Energy_source",
#             TextureX_var_name="Efficiency_class",
#             color_discrete_sequence=colors)
# plotly.offline.plot(fig, filename=Graphic_folder+'surface_Energy_source_Categories.html')

#endregion

#region models
# ce dont on a besoin à la fin pour la simulation
# une formule pour calculer la conso et les émissions
start = time.process_time()
def f_Compute_conso(x): return x["Besoin_surfacique"] * x["Surface"]*x["proportion_besoin_chauffage"]/x["COP"]
simulation_parameters["f_Compute_conso"]=f_Compute_conso

## initialize all "new_yearly_surface"
simulation_parameters["type_index_year_new"]
simulation_parameters["new_yearly_surface"]= pd.Series(0.,index=simulation_parameters["type_index_year_new"])
simulation_parameters["new_energy"]= pd.Series(simulation_parameters["new_energy"],index=simulation_parameters["type_index_year_new"] )
TMP = (simulation_parameters["new_yearly_surface_per_cat"]*simulation_parameters["new_repartition"]).to_frame().assign(Efficiency_class = "A").\
    reset_index().set_index(simulation_parameters["Index_names"]+["year"])
TMP.columns =["new_yearly_surface"]; TMP=TMP.assign(old_new = "new").set_index(["old_new"],append=True)["new_yearly_surface"]
simulation_parameters["new_yearly_surface"][TMP.index]+=TMP

def f_retrofit_total_yearly_surface(year,alpha,simulation_parameters):
    ## Ici ne dépend pas de year
    return  alpha *  simulation_parameters["init_Description_Parc"].loc[:,"Surface"]
simulation_parameters["f_retrofit_total_yearly_surface"]=f_retrofit_total_yearly_surface
calibration_parameters={"f_retrofit_total_yearly_surface" :
                            {"total_change_target": simulation_parameters["retrofit_change"] *
                                                    simulation_parameters["init_Description_Parc"]["Surface"]}}


simulation_parameters=calibrate_simulation_parameters(simulation_parameters,calibration_parameters)
end = time.process_time()
print("Chargement et calibration des modèles terminé en : "+str(end-start)+" secondes")

#endregion

Description_Parc= loanch_simulation(simulation_parameters)

#region Chargement des données
test = data_tertiaire["Categories_Energy_source_year"].set_index(["Categories","Energy_source","year"])
new_index = pd.MultiIndex.from_frame(simulation_parameters["type_index"])
new_index = [(*type,year) for year in test.reset_index()["year"] for idx,type in simulation_parameters["type_index"].iterrows()]
new_index = [(*type,year) for year in range(2020,2050) for idx,type in simulation_parameters["type_index"].iterrows()]
test=test.reindex(new_index,method="pad")
for idx,type in simulation_parameters["type_index"].iterrows():
    test.loc[(*type)] = test.interpolate()
pd.multiIndex


taux_disp=0.005 # taux de disparition chaque année sur l'ancien
year_optim=2030
year_initial = 2020
year_final = 2060
amelioration_renovation_initiale=0.15
#endregion

#region trouver les surfaces à transitionner et les rythmes de changement
Surfaces_a_transitionner =Description_Parc_initiale[["Surface"]].groupby(["Categories","Energy_source" ]).sum()
S_trans_all = Transition
for col in S_trans_all.columns:
    S_trans_all.loc[:,col]=S_trans_all.loc[:,col]*Surfaces_a_transitionner.loc[:,"Surface"]
#endregion

#region définition de fonctions utiles
def energy_to_class(energy):
    if energy<=50:
        return "A"
    elif energy<=90:
        return "B"
    elif energy<=150:
        return "C"
    elif energy<=230:
        return "D"
    elif energy<=330:
        return "E"
    elif energy<=450:
        return "F"
    else:
        return "G"


def taux_reno(year):
    taux_2020 = 0.015
    taux_2030 = 0.02
    taux_2040 = 0.025
    taux_2050 = 0.025
    taux_2060 = 0.025
    if year<=2030:
        return taux_2020+(year-2020)/10*(taux_2030-taux_2020)
    elif year<=2040:
        return taux_2030+(year-2030)/10*(taux_2040-taux_2030)
    elif year<=2050:
        return taux_2040+(year-2040)/10*(taux_2050-taux_2040)
    else:
        return taux_2050+(year-2050)/10*(taux_2060-taux_2050)
#endregion

#region graphique de Marrimenko
colors = mcp.gen_color(cmap='Accent',n=8)
# see https://matplotlib.org/stable/gallery/color/colormap_reference.html
fig=marimekko(Description_Parc_initiale[["Surface"]].reset_index().groupby(['Categories',"Energy_source"], as_index=False).sum(),
                x_var_name = "Energy_source",
                y_var_name= "Categories",
                effectif_var_name='Surface',
              color_discrete_sequence=colors )
plotly.offline.plot(fig, filename=Graphic_folder+'surface_Energy_source_Categories.html')

fig=marimekko(Description_Parc_initiale[["Surface"]].reset_index().groupby(['Efficiency_class',"Categories"], as_index=False).sum(),
                x_var_name = "Categories",
                y_var_name= "Efficiency_class",
                effectif_var_name='Surface',
              color_discrete_sequence= dpe_colors)
plotly.offline.plot(fig, filename=Graphic_folder+'surface_Categories_Efficiency_class.html')

fig=marimekko(Description_Parc_initiale[["Surface"]].reset_index().groupby(['Efficiency_class',"Energy_source"], as_index=False).sum(),
                x_var_name = "Energy_source",
                y_var_name= "Efficiency_class",
                effectif_var_name='Surface',
              color_discrete_sequence= dpe_colors)
plotly.offline.plot(fig, filename=Graphic_folder+'surface_Energy_source_Efficiency_class.html')

#not working
# fig = marimekko_2(df =Description_Parc_initiale[["Surface"]].reset_index().groupby(['Categories',"Efficiency_class","Energy_source"], as_index=False).sum() ,
#             effectif_var_name = "Surface",
#             ColorY_var_name="Categories",
#             horizontalX_var_name="Energy_source",
#             TextureX_var_name="Efficiency_class",
#             color_discrete_sequence=colors)
# plotly.offline.plot(fig, filename=Graphic_folder+'surface_Energy_source_Categories.html')

#endregion

#region ordre des indexs : Categories Energy_source Efficiency_class old_new
Description_Parc = {}
year=year_initial
Description_Parc[year]=pd.concat( [ Description_Parc_initiale.assign(old_new="old"),
                                Description_Parc_initiale.assign(old_new="new")]).set_index(['old_new'], append=True)
Description_Parc[year].loc[(slice(None),slice(None),slice(None),"new"),"surface"]=0 ## on commence sans "neuf"

for year in range(year_initial,year_final):
    Taux_renovation = taux_reno(year)
    Description_Parc[year] = Description_Parc[year-1]
    #remove a proportion taux_disp of old buildings
    Description_Parc[year].loc[(slice(None),slice(None),slice(None),"old"),"surface"] =  (1-taux_disp) * Description_Parc[year].loc[(slice(None),slice(None),slice(None),"old"),"surface"]
    #Retrofit of old buildings
    for Efficiency_class in ["C","D","E","F","G"]:
        old_index = (slice(None), slice(None), Efficiency_class, "old")
        if year > year_optim:
            Ancien_besoin = Description_Parc[year-1].loc[old_index,"Besoin_surfacique"]
            Amelioration = Description_Parc[year-1].loc[old_index,"Amelioration_max"]
        else:
            Ancien_besoin = Description_Parc[year-1].loc[old_index,"Besoin_surfacique"]
            Amelioration_max = Description_Parc[year-1].loc[old_index,"Amelioration_max"]
            #Amelioration(year = year_initiale) =  amelioration_renovation_initiale et Amelioration(year = year_optim) = Amelioration_max. Evolution linéaire
            Amelioration = (amelioration_renovation_initiale+(year-year_initial)/(year_optim-year_initial))*(Amelioration_max-amelioration_renovation_initiale)
        for key in Amelioration.keys():
            Nouveau_besoin =  (1 - Amelioration[key]) * Ancien_besoin[key]
            New_Energy_Class = energy_to_class(Nouveau_besoin)### chauffage ?
            Surface_renovee = Taux_renovation * Description_Parc[year].loc[key, "Surface"]
            #mettre à jour la performance moyenne dans la nouvelle classe
            Ancien_new_surface = Description_Parc[year].loc[(key[0],key[1],New_Energy_Class,"new"), "Surface"]
            Ancien_new_Besoin_surfacique = Description_Parc[year].loc[(key[0], key[1], New_Energy_Class, "new"), "Besoin_surfacique"]
            Description_Parc[year].loc[(key[0], key[1], New_Energy_Class, "new"), "Besoin_surfacique"]= (Surface_renovee*Nouveau_besoin+Ancien_new_surface*Ancien_new_Besoin_surfacique)/(Surface_renovee+Ancien_new_surface)
            #mettre à jour les surfaces (ancienne classe et nouvelle classe)
            Description_Parc[year].loc[key, "Surface"]-= Surface_renovee
            Description_Parc[year].loc[(key[0],key[1],New_Energy_Class,"new"), "Surface"] += Surface_renovee


#endregion

from scipy.interpolate import interp1d
x = np.linspace(0, 10, num=11, endpoint=True)
y = np.cos(-x**2/9.0)
f = interp1d(x, y)
f2 = interp1d(x, y, kind='cubic')
interpolation_functions = {}
Energy_source = data_tertiaire["Energy_source"].Energy_source.unique()
for EE in Energy_source:
    # fioul : P_trans_all(year_ref_trans) = S_trans_all *
if year <= year_optim_fioul:
    for k in range(8):
        for i in [0, 2, 3, 4, 5, 6, 7]:
            for j in range(8):
                P_trans_all[k][i][j] = S_trans_all[k][i][j] / (
                            year_goal_trans - (year_ref_trans + year_optim_trans) / 2 + 1 / 2) \
                                       * (year - year_ref_trans) / (year_optim_trans - year_ref_trans)
        for j in range(8):
            P_trans_all[k][1][j] = S_trans_all[k][1][j] / (
                        year_goal_fioul - (year_ref_trans + year_optim_fioul) / 2 + 1 / 2) \
                                   * (year - year_ref_trans) / (year_optim_fioul - year_ref_trans)

            (
                    year_goal_fioul - (year_ref_trans + year_optim_fioul) / 2 + 1 / 2) \
            * (year - year_ref_trans) / (year_optim_fioul - year_ref_trans)
            elif year <= year_optim_trans:
                for k in range(8):
                    for i in [0, 2, 3, 4, 5, 6, 7]:
                        for j in range(8):
                            P_trans_all[k][i][j] = S_trans_all[k][i][j] / (
                                        year_goal_trans - (year_ref_trans + year_optim_trans) / 2 + 1 / 2) \
                                                   * (year - year_ref_trans) / (year_optim_trans - year_ref_trans)
                    for j in range(8):
                        P_trans_all[k][1][j] = S_trans_all[k][1][j] / (
                                    year_goal_fioul - (year_ref_trans + year_optim_fioul) / 2 + 1 / 2)
            else:
                for k in range(8):
                    for i in [0, 2, 3, 4, 5, 6, 7]:
                        for j in range(8):
                            P_trans_all[k][i][j] = S_trans_all[k][i][j] / (
                                        year_goal_trans - (year_ref_trans + year_optim_trans) / 2 + 1 / 2)
                    for j in range(8):
                        P_trans_all[k][1][j] = S_trans_all[k][1][j] / (
                                    year_goal_fioul - (year_ref_trans + year_optim_fioul) / 2 + 1 / 2)
            for k in range(8):
                for i in range(8):
                    denom_k = max(L_heat_all_t[k][i], sum(P_trans_all[k][i]), 1)
                    for j in range(8):
                        P_trans_all[k][i][j] = P_trans_all[k][i][j] / denom_k
            return P_trans_all