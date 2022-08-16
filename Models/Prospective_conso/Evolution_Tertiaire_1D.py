#region Chargement des packages
import pandas as pd
from functions.f_tools import *
#from Models.Prospective_conso.f_evolution_simulation import Renovation_changement_chauff
from mycolorpy import colorlist as mcp
import numpy as np

dpe_colors = ['#009900', '#33cc33', '#B3FF00', '#e6e600', '#FFB300', '#FF4D00', '#FF0000',"#000000"]
Graphic_folder = "Models/Prospective_conso/Graphics/"
Data_folder = "Models/Prospective_conso/data/"
#endregion

#region chargement des données
data_tertiaire =  pd.read_excel(Data_folder+"Hypotheses_tertiaire_1D.xlsx",
                            sheet_name=["Energy_source","Transition","0D"])

Description_Parc_initiale = data_tertiaire["Energy_source"].set_index(["Energy_source"]).\
    assign(Besoin_surfacique = float(data_tertiaire["0D"]["Besoin_surfacique"])).\
    assign(proportion_besoin_chauffage =  float(data_tertiaire["0D"]["proportion_besoin_chauffage"])).\
    assign(Conso=lambda x: x["Besoin_surfacique"] * x["Surface"]*x["proportion_besoin_chauffage"]/x["COP"])

data_fin = 2050
data_debut = 2020

#hypothèses pour la destruction
taux_disp=0.005 # taux de disparition chaque année sur l'ancien

#hypothèses pour la réno
Transition=data_tertiaire["Transition"].assign(old_new="new").rename(columns ={"Energy_source":"Energy_source_old"}).set_index(["Energy_source_old","old_new"]).\
    melt(ignore_index=False,var_name="Energy_source",value_name="Transition").set_index(["Energy_source"],append=True)

Changement_total = 1 # fraction de la surface initiale rénovée/changée dans le parc à la date de fin
rythme_changement_m2_par_chauff_par_an = {chauff : Changement_total/(data_fin-data_debut) *Description_Parc_initiale.loc[chauff,"Surface"] for chauff in Description_Parc_initiale.index}
Renovation_changement_chauff_assumptions={"rythme_changement_m2_par_chauff_par_an":rythme_changement_m2_par_chauff_par_an,
                                          "perf_reno" : 0.30 ,#taux de diminution
                                          "Transition" : Transition}
#hypothèses pour le neuf
Repartition_chauffage_neuf=Description_Parc_initiale[["Repartition_chauffage_neuf_premieres_annees","Repartition_chauffage_neuf_regime_normal"]].\
            assign(old_new="new").reset_index().\
            rename(columns = {"index" : "Energy_source"}).\
            set_index(["Energy_source","old_new"])
New_buildings_assumption ={"Repartition_chauffage_neuf" : Repartition_chauffage_neuf,
                 "energy_new" : 50,
                 "date_changement_neuf": 2025,
                 "Nouvelle_surface_par_an" :float(data_tertiaire["0D"]["Nouvelle_surface_par_an"])}



#rythme_changement_m2_par_chauff_par_an = pd.DataFrame.from_dict({chauff : Changement_total/(data_fin-data_debut) *Description_Parc_initiale.loc[chauff,"Surface"] for chauff in Description_Parc_initiale.index},
#                                                                orient='index').reset_index().\
#    assign(old_new="new").rename(columns = {"index" : "Energy_source",0:"Surface"}).set_index(["Energy_source","old_new"])
#endregion

#region creation des modèles

#lorsque l'on met à jour l'ensemble des surfaces et le besoin associé
def update_surface_heat_need(Description_Parc_year,Surfaces,heat_need):
    Surfaces=Surfaces.reset_index().set_index(["Energy_source","old_new"])
    Surfaces=Surfaces["Surface"]
    Ancien_besoin = Description_Parc_year.loc[(slice(None), "new"), "Besoin_surfacique"]
    Anciennes_surfaces = Description_Parc_year.loc[(slice(None), "new"), "Surface"]
    Description_Parc_year.loc[(slice(None), "new"), "Surface"] += Surfaces
    Description_Parc_year.loc[(slice(None), "new"), "Besoin_surfacique"] = (Ancien_besoin * Anciennes_surfaces + heat_need *Surfaces) / \
                                                                            Description_Parc_year.loc[(slice(None), "new"), "Surface"]
    return Description_Parc_year


#renovation et changement de mode de chauffage
def Renovation_changement_chauff(Description_Parc,year,Renovation_changement_chauff_assumptions):
    Energy_sources = Description_Parc[year].reset_index()["Energy_source"].unique()
    for chauff in Energy_sources:
        Surface_renovee = Renovation_changement_chauff_assumptions["rythme_changement_m2_par_chauff_par_an"][chauff]
        Nouveau_besoin = (1 - Renovation_changement_chauff_assumptions["perf_reno"]) * Description_Parc[year - 1].loc[(chauff, "old"), "Besoin_surfacique"]
        Surfaces_renovee_par_chauff = Surface_renovee*Renovation_changement_chauff_assumptions["Transition"].loc[(chauff,slice(None),slice(None)),].rename(columns={"Transition" :"Surface"})
        Description_Parc[year]=update_surface_heat_need(Description_Parc_year=Description_Parc[year],
                                                        Surfaces=Surfaces_renovee_par_chauff,
                                                        heat_need=Nouveau_besoin)
        Description_Parc[year].loc[(chauff, "old"), "Surface"] -= Surface_renovee
    return Description_Parc

def Construction_neuf(Description_Parc,year,New_buildings_assumption):
    if year<New_buildings_assumption["date_changement_neuf"]:
        surface_neuf="Repartition_chauffage_neuf_premieres_annees"
    else:
        surface_neuf = "Repartition_chauffage_neuf_regime_normal"

    Nouvelles_Surfaces = New_buildings_assumption["Repartition_chauffage_neuf"][surface_neuf] * New_buildings_assumption["Nouvelle_surface_par_an"]
    Description_Parc[year] = update_surface_heat_need(Description_Parc_year=Description_Parc[year],
                                                      Surfaces=Nouvelles_Surfaces.to_frame().rename(columns={surface_neuf: "Surface"}),
                                                      heat_need=New_buildings_assumption["energy_new"])
    return Description_Parc

def Destruction_ancien(Description_Parc,year,taux_disp):
    Description_Parc[year].loc[(slice(None),"old"),"Surface"] -=  taux_disp * Description_Parc[year].loc[(slice(None),"old"),"Surface"]
    return Description_Parc
#endregion


#region ordre des indexs : Categories Energy_source Efficiency_class old_new
Description_Parc = {}
year=data_debut
Description_Parc[year]=pd.concat( [ Description_Parc_initiale.assign(old_new="old"),
                                Description_Parc_initiale.assign(old_new="new")]).set_index(['old_new'], append=True)
Description_Parc[year].loc[(slice(None),"new"),"Surface"]=0 ## on commence sans "neuf"

for year in progressbar(range(data_debut+1,data_fin), "Computing: ", 40):
    Description_Parc[year] = Description_Parc[year-1]
    Description_Parc    =   Destruction_ancien(Description_Parc, year, taux_disp)
    Description_Parc    =   Renovation_changement_chauff(Description_Parc, year, Renovation_changement_chauff_assumptions)
    Description_Parc    =   Construction_neuf(Description_Parc, year, New_buildings_assumption)
    Description_Parc[year]= Description_Parc[year].assign(Conso=lambda x: x["Besoin_surfacique"] * x["Surface"]*x["proportion_besoin_chauffage"]/x["COP"])
    #energy_new
#endregion

year=data_debut+1
Description_Parc_df = pd.concat(Description_Parc, axis=0).reset_index().\
    rename(columns={"level_0":"year"}).set_index([ "year"  ,  "Energy_source"  , "old_new"])
Description_Parc_df.groupby(["year","Energy_source"]).Conso.sum().to_frame().reset_index().\
    pivot(index='year', columns='Energy_source')['Conso']

Description_Parc_df.groupby(["year","Energy_source"]).Surface.sum().to_frame().reset_index().\
    pivot(index='year', columns='Energy_source')['Surface']
