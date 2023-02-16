
#region Chargement des packages
import pandas as pd
from EnergyAlternativesPlanning.f_graphicalTools import *
from EnergyAlternativesPlanning.f_graphicalTools import *
from EnergyAlternativesPlanning.f_tools import *
from Models.Prospective_conso.f_evolution_tools import *
Data_folder = "Models/Prospective_conso/data/"

pd.options.display.width = 0
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#endregion

## this file has two regions :
## 1 - gathering 3D data from xlsx file
## 2 - creation of 1D data to xlsx file

##region récupération des données 3D
data_set_from_excel     =   pd.read_excel(Data_folder+"Hypotheses_tertiaire_3D.xlsx", None);
sim_param = extract_sim_param(data_set_from_excel,Index_names =  ["Categories", "Efficiency_class" ,"Energy_source"],
                              dim_names=["Categories","Energy_source","Efficiency_class","year","year"])
sim_param["init_sim_stock"]=create_initial_parc(sim_param).sort_index()
def f_Compute_surface(x):
    return x["Energy_source_per_Category"]*x["Efficiency_class_per_Category"]*x["total_surface"]
sim_param["init_sim_stock"]=sim_param["init_sim_stock"].assign(surface = lambda x : f_Compute_surface(x))



init_replace_dict = {key : "init_"+key for key in sim_param["init_sim_stock"].columns}

Para_2_fill = {param : sim_param["base_index_year"] for param in ["retrofit_improvement","retrofit_Transition",
                                                                   "new_energy","new_yearly_repartition"]}
sim_param   =   complete_parameters(sim_param,Para_2_fill=Para_2_fill)
sim_param["new_yearly_surface"]=sim_param["new_yearly_surface"]*sim_param["new_yearly_repartition"]
#sim_param["retrofit_Transition"]
### 0D dictionnary
#endregion

##region créaction des données 1D
variable_dict = {
    "0D" : {
        "dims" : [],
        "values": {
            'date_debut': "",
            'energy_need_per_surface': "mean",
            'proportion_besoin_chauffage': 'mean',
            'new_energy': 'mean',
            'new_yearly_surface': 'sum'
        }},
    "year" : {
        "dims": ["year"],
        "values": {
            "old_taux_disp": "mean",
            "retrofit_improvement": "mean"
    }},
    "Energy_source" : {
        "dims": ["Energy_source"],
        "values": {
            "surface": "sum",
            "conso_unitaire_elec": "sum",
            "conso_unitaire_gaz": "sum",
            "conso_unitaire_fioul": "sum",
            "conso_unitaire_bois": "sum"
    }},
    "Energy_source_year" : {
        "dims": ["Energy_source","year"],
        "values": {
            "new_yearly_repartition": "mean"
        },
        "post_op" : ".loc[(slice(None),[2020,2025]),:].reset_index().sort_values(by=['year', 'Energy_source'], ascending=True).set_index([\"Energy_source\",\"year\"])"
    }
}

Transition = sim_param["retrofit_Transition"].groupby([ "Categories","Energy_source"]).mean().\
    merge(sim_param["init_sim_stock"].groupby([ "Categories","Energy_source"])["surface"].sum(),
          how = 'outer',left_index=True,right_index=True).reset_index().\
    groupbyAndAgg(group_along= ["Energy_source"],
            aggregation_dic={chauff : "wmean" for chauff in sim_param["Energy_source_index"]},
            weightedMean_weight="surface").\
    assign(year=2020).set_index(["Energy_source","year"]).fillna(0)

with pd.ExcelWriter(Data_folder + "Hypotheses_tertiaire_1D_bis.xlsx") as writer:
    for V in variable_dict:
        if len(variable_dict[V]["dims"])==0:
           df = Create_0D_df(variable_dict[V]["values"], sim_param).reset_index()
           df.columns = ["Nom", "Valeur"]
           df.set_index(["Nom"]).to_excel(writer, sheet_name=V)
        else:
            cur_dict = variable_dict[V]["values"]
            cur_dims = variable_dict[V]["dims"]
            extension=""
            if "post_op" in variable_dict[V]: extension += variable_dict[V]["post_op"]
            extension+=".to_excel(writer, sheet_name=V)"
            exec("Create_XD_df(cur_dict, sim_param,group_along = cur_dims)"+extension)
    Transition.to_excel(writer, sheet_name='retrofit_Transition')
#endregion