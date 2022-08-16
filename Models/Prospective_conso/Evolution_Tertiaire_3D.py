#region Chargement des packages
import pandas as pd
from functions.f_graphicalTools import *
from functions.f_tools import *
from mycolorpy import colorlist as mcp
import numpy as np

dpe_colors = ['#009900', '#33cc33', '#B3FF00', '#e6e600', '#FFB300', '#FF4D00', '#FF0000',"#000000"]
Graphic_folder = "Models/Prospective_conso/Graphics/"
Data_folder = "Models/Prospective_conso/data/"
#endregion

#region Chargement des données
data_tertiaire =  pd.read_excel(Data_folder+"Hypotheses_tertiaire_3D.xlsx",
                            sheet_name=["Categories","Efficiency_class","Energy_source","Transition",
                                        "Categories_Energy_source","Categories_Efficiency_class"])
#Efficacite_max = data_tertiaire["Categories"].set_index(["Categories"])[["Efficacite_max"]]
Description_Parc_initiale = data_tertiaire["Categories"].set_index(["Categories"]).\
    merge(data_tertiaire["Categories_Energy_source"].set_index(["Categories","Energy_source"]),how = 'outer',left_index=True,right_index=True). \
    merge(data_tertiaire["Categories_Efficiency_class"].set_index(["Categories","Efficiency_class"]), how='outer', left_index=True, right_index=True). \
    merge(data_tertiaire["Efficiency_class"].set_index(["Efficiency_class"]),how = 'outer',left_index=True,right_index=True). \
    merge(data_tertiaire["Energy_source"].set_index(["Energy_source"]), how='outer', left_index=True,right_index=True). \
    assign(Surface=lambda  x: x["Energy_source_per_Category"]*x["Efficiency_class_per_Category"]*x["total_surface"]).\
    assign(Conso=lambda x: x["Besoin_surfacique"] * x["Surface"]*x["proportion_besoin_chauffage"]/x["COP"])

Transition=data_tertiaire["Transition"].set_index(["Categories","Energy_source"])

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