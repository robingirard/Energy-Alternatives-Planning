
#region Chargement des packages
import numpy as np
import pandas as pd
from functions.f_graphicalTools import *
from functions.f_graphicalTools import *
from functions.f_tools import *
from Models.Prospective_conso.f_evolution_tools import *
Data_folder = "Models/Prospective_conso/data/"

pd.options.display.width = 0
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
#endregion

#region creation des données résidentielles 2D simples vers Hypotheses_residentiel_2D_bis
base_dpe_residentiel_df=pd.read_hdf(Data_folder+'district_level_census_latest.hdf')
NEW_DATA=True
base_dpe_residentiel_df=base_dpe_residentiel_df.rename(columns={"ipondl":"IPONDL",
                                                                "occupant_status":"occupancy_status"})
#base_dpe_residentiel_df=pd.read_csv(Data_folder+'base_logement_agregee.csv', sep=';', decimal='.')
index= base_dpe_residentiel_df["occupancy_status"].isin(['free accomodation', 'low rent housing','owner', 'renter'])
base_dpe_residentiel_df=base_dpe_residentiel_df[index]
base_dpe_residentiel_df.IPONDL.sum()
Fraction_PACS ={    "Pompes à chaleur air-air": 6.5 / 8,
                    "Pompes à chaleur air-eau": 1.5 / 8,
                    "Pompes à chaleur hybride": 0.}



#### attention !!! ce que je fais ci-dessous est faux, les données ne sont pas bonnes et on ne
#### peut pas concidérer que tout ça c'est du bois
#### https://www.connaissancedesenergies.org/transition-energetique-le-chauffage-domestique-au-bois-incontournable-220218
#### L’Ademe fait état de 6,8 millions d’utilisateurs d’appareils de chauffage au bois - les utilisant comme chauffage principal dans 48% des cas
#### là ça fait presque 10 millions de chauffage principal bois, avec 3 millions d'appartements ...
print(base_dpe_residentiel_df['heating_system'].unique().astype("str"))

if NEW_DATA:
    change_dict = {'oil boiler' : "Chaudière fioul",'electric heater' : 'Chauffage électrique',
     'biomass/coal stove' : 'Biomasse','liquified petroleum gas boiler': 'Chaudière gaz',
     'oil stove' : "Chaudière fioul",'electric heat pump' : 'Pompes à chaleur','fossil gas boiler' : 'Chaudière gaz',
     'urban heat network':'Chauffage urbain','fossil gas stove': "Chaudière fioul",'liquified petroleum gas stove': 'Chaudière gaz'}
    base_dpe_residentiel_df['heating_system']=base_dpe_residentiel_df['heating_system'].replace(change_dict)
else:
    base_dpe_residentiel_df['heating_system']=base_dpe_residentiel_df['heating_system'].replace({"Chaudière - autres":"Biomasse",
                                                                                             "Autres":"Biomasse"})
### les stats
if NEW_DATA:
    base_dpe_residentiel_df.groupbyAndAgg(group_along= ['heating_system',"heating_fuel"],
                aggregation_dic={"energy_consumption" : "wmean",
                                 "surface": "wmean",
                                 "IPONDL" : "sum"},
                weightedMean_weight="IPONDL")
else:
    base_dpe_residentiel_df.groupbyAndAgg(group_along= ['heating_system'],
                aggregation_dic={"energy_consumption" : "wmean",
                                 "surface": "wmean",
                                 "IPONDL" : "sum"},
                weightedMean_weight="IPONDL")

Index_to_add = expand_grid_from_dict({"Energy_source": list(Fraction_PACS.keys()),
                                      "residential_type":base_dpe_residentiel_df["residential_type"].unique()}).\
    assign(IPONDL = 0,energy_need_per_surface=0,surface=0).\
        set_index(["residential_type", "Energy_source"])
base_dpe_residentiel_df=base_dpe_residentiel_df.\
        groupbyAndAgg(group_along= ["residential_type",'heating_system'],
            aggregation_dic={"energy_consumption" : "wmean",
                             "surface": "wmean",
                             "IPONDL" : "sum"},
            weightedMean_weight="IPONDL").\
        rename(columns = {"heating_system":"Energy_source",
                          "energy_consumption" : "energy_need_per_surface"}).\
        set_index(["residential_type", "Energy_source"])



base_dpe_residentiel_df = pd.concat([base_dpe_residentiel_df,Index_to_add])


for key in Fraction_PACS.keys():
    for col in base_dpe_residentiel_df.columns:
        base_dpe_residentiel_df.loc[(slice(None),key),col]=(Fraction_PACS[key]*base_dpe_residentiel_df.loc[(slice(None),"Pompes à chaleur électricité"),col]).to_list()
base_dpe_residentiel_df.loc[(slice(None),"Pompes à chaleur électricité"),:]=np.nan
base_dpe_residentiel_df=base_dpe_residentiel_df.dropna().sort_index()

variable_dict={}
variable_dict["0D"] = pd.DataFrame.from_dict({"retrofit_improvement": 0.3,"date_debut": 2020,
 "date_fin": 2050, "retrofit_change_total_proportion_surface": 1},orient="index").reset_index()
variable_dict["0D"].columns = ["Nom","Valeur"]
variable_dict["0D"]=variable_dict["0D"].set_index(["Nom"])
variable_dict["retrofit_Transition"] =  pd.read_excel(Data_folder+"Hypotheses_tertiaire_1D.xlsx", ["retrofit_Transition"])['retrofit_Transition'].\
    set_index(["Energy_source" , "year"])
variable_dict["Energy_source"] =  pd.read_excel(Data_folder+"Hypotheses_tertiaire_1D.xlsx", ["Energy_source"])['Energy_source'].\
    set_index(["Energy_source"])[["init_conso_unitaire_elec"	,"init_conso_unitaire_gaz",	"init_conso_unitaire_fioul",	"init_conso_unitaire_bois"]]

base_dpe_residentiel_df.columns= ["init_"+col for col in base_dpe_residentiel_df.columns]
variable_dict["res_type_Energy_source"] = base_dpe_residentiel_df.reset_index().set_index(["Energy_source","residential_type"])

### il faudrait améliorer extract_sim_param pour ne pas avoir à créer ces deux tables ...
#variable_dict["residential_type"] = pd.DataFrame.from_dict({"residential_type": base_dpe_residentiel_df.reset_index()["residential_type"].unique(),
#                                                  "Null":"null"},orient="columns").set_index(["residential_type"])

with pd.ExcelWriter(Data_folder + "Hypotheses_residentiel_2D_bis.xlsx") as writer:
    for V in variable_dict:
        variable_dict[V].to_excel(writer, sheet_name=V)
#endregion