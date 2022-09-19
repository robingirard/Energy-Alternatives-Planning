
#region Chargement des packages
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

#region chargement des données acier
start = time.process_time()
target_year = 2050 # 2030 ou 2050 --> two excel sheets
dim_names=["Production_system","year","Vecteurs"];Index_names = ["Production_system"];Energy_system_name="Production_system"
data_set_from_excel =  pd.read_excel(Data_folder+"Hypotheses_acier_1D_"+str(target_year)+".xlsx", None);
sim_param = extract_sim_param(data_set_from_excel,Index_names = Index_names,dim_names=dim_names,Energy_system_name=Energy_system_name)
sim_param["init_sim_stock"]=create_initial_parc(sim_param).sort_index()
sim_param["volume_variable_name"] = "unite_prod"
sim_param["retrofit_change_"+sim_param["volume_variable_name"]] = sim_param["retrofit_change_total_proportion_init_unite_prod"]/len(sim_param["years"]) *\
                                                                  sim_param["init_sim_stock"][sim_param["volume_variable_name"]]

Para_2_fill = {param : sim_param["base_index_year"] for param in ["retrofit_improvement","retrofit_change_unite_prod","retrofit_Transition","energy_need_per_unite_prod",
                                                                                                   "new_energy","new_yearly_repartition_per_Energy_source"]}
sim_param   =   complete_parameters(sim_param,Para_2_fill=Para_2_fill)

def f_Compute_conso(x,sim_param,Vecteur):
    conso_unitaire = x["conso_unitaire_"+Vecteur]
    return x["energy_need_per_"+sim_param["volume_variable_name"]] * x[sim_param["volume_variable_name"]]*conso_unitaire
def f_Compute_conso_totale(x,sim_param):
    res=0.
    for Vecteur in sim_param["Vecteurs"]:
        res+=x["conso_"+Vecteur]
    return res

for Vecteur in sim_param["Vecteurs"]:
    sim_param["f_Compute_conso_"+Vecteur]={"conso_"+Vecteur : partial(f_Compute_conso,Vecteur =Vecteur)}
sim_param["f_Compute_conso_totale"]={"Conso" : lambda x,sim_param: f_Compute_conso_totale(x,sim_param)}


def f_Compute_emissions(x,sim_param):
    emissions = 0
    for Vecteur_ in sim_param["Vecteurs"]: emissions += x["conso_unitaire_"+Vecteur_]*sim_param["Emissions_scope_2_3"][Vecteur_]
    emissions+= x["emissions_unitaire"]* x[sim_param["volume_variable_name"]]
    return emissions

def f_Compute_emissions_year(x,sim_param,year):
    emissions = 0
    for Vecteur_ in sim_param["Vecteurs"]: emissions += x["conso_unitaire_"+Vecteur_]*sim_param["Emissions_scope_2_3"][(Vecteur_,year)]
    emissions+= x["emissions_unitaire"]* x[sim_param["volume_variable_name"]]
    return emissions

if type(sim_param["Emissions_scope_2_3"].index) == pd.MultiIndex:
    sim_param["f_Compute_emissions"]= {"Emissions" : f_Compute_emissions_year }#{"Emissions" : partial(f_Compute_emissions,year =year)}
else:
    sim_param["f_Compute_emissions"]={"Emissions" : f_Compute_emissions}
sim_param_acier=sim_param
end = time.process_time()
print("Chargement des données, des modèles et interpolation terminés en : "+str(end-start)+" secondes")
#endregion

#region chargement des données ceramique
start = time.process_time()
dim_names=["Production_system","year","Vecteurs"];Index_names = ["Production_system"];Energy_system_name="Production_system"
data_set_from_excel =  pd.read_excel(Data_folder+"Hypotheses_ceramics_1D.xlsx", None);
sim_param = extract_sim_param(data_set_from_excel,Index_names = Index_names,dim_names=dim_names,Energy_system_name=Energy_system_name)
sim_param["init_sim_stock"]=create_initial_parc(sim_param).sort_index()
sim_param["volume_variable_name"] = "unite_prod"
sim_param["retrofit_change_"+sim_param["volume_variable_name"]] = sim_param["retrofit_change_total_proportion_init_unite_prod"]/len(sim_param["years"]) *\
                                                                  sim_param["init_sim_stock"][sim_param["volume_variable_name"]]

Para_2_fill = {param : sim_param["base_index_year"] for param in ["retrofit_improvement","retrofit_change_unite_prod","retrofit_Transition","energy_need_per_unite_prod",
                                                                                                   "new_energy","new_yearly_repartition_per_Energy_source"]}
sim_param   =   complete_parameters(sim_param,Para_2_fill=Para_2_fill)

def f_Compute_conso(x,sim_param,Vecteur):
    conso_unitaire = x["conso_unitaire_"+Vecteur]
    return x["energy_need_per_"+sim_param["volume_variable_name"]] * x[sim_param["volume_variable_name"]]*conso_unitaire
def f_Compute_conso_totale(x,sim_param):
    res=0.
    for Vecteur in sim_param["Vecteurs"]:
        res+=x["conso_"+Vecteur]
    return res

for Vecteur in sim_param["Vecteurs"]:
    sim_param["f_Compute_conso_"+Vecteur]={"conso_"+Vecteur : partial(f_Compute_conso,Vecteur =Vecteur)}
sim_param["f_Compute_conso_totale"]={"Conso" : lambda x,sim_param: f_Compute_conso_totale(x,sim_param)}


def f_Compute_emissions(x,sim_param):
    emissions = 0
    for Vecteur_ in sim_param["Vecteurs"]: emissions += x["conso_unitaire_"+Vecteur_]*sim_param["Emissions_scope_2_3"][Vecteur_]
    emissions+= x["emissions_unitaire"]* x[sim_param["volume_variable_name"]]
    return emissions

def f_Compute_emissions_year(x,sim_param,year):
    emissions = 0
    for Vecteur_ in sim_param["Vecteurs"]: emissions += x["conso_unitaire_"+Vecteur_]*sim_param["Emissions_scope_2_3"][(Vecteur_,year)]
    emissions+= x["emissions_unitaire"]* x[sim_param["volume_variable_name"]]
    return emissions

if type(sim_param["Emissions_scope_2_3"].index) == pd.MultiIndex:
    sim_param["f_Compute_emissions"]= {"Emissions" : f_Compute_emissions_year }#{"Emissions" : partial(f_Compute_emissions,year =year)}
else:
    sim_param["f_Compute_emissions"]={"Emissions" : f_Compute_emissions}
sim_param_acier=sim_param
end = time.process_time()
print("Chargement des données, des modèles et interpolation terminés en : "+str(end-start)+" secondes")
#endregion

#region chargement des données ceramique
start = time.process_time()
dim_names=["Production_system","year","Vecteurs"];Index_names = ["Production_system"];Energy_system_name="Production_system"
data_set_from_excel =  pd.read_excel(Data_folder+"Hypotheses_ceramics_1D.xlsx", None);
sim_param = extract_sim_param(data_set_from_excel,Index_names = Index_names,dim_names=dim_names,Energy_system_name=Energy_system_name)
sim_param["init_sim_stock"]=create_initial_parc(sim_param).sort_index()
sim_param["volume_variable_name"] = "unite_prod"
sim_param["retrofit_change_"+sim_param["volume_variable_name"]] = sim_param["retrofit_change_total_proportion_init_unite_prod"]/len(sim_param["years"]) *\
                                                                  sim_param["init_sim_stock"][sim_param["volume_variable_name"]]

Para_2_fill = {param : sim_param["base_index_year"] for param in ["retrofit_improvement","retrofit_change_unite_prod","retrofit_Transition","energy_need_per_unite_prod",
                                                                                                   "new_energy","new_yearly_repartition_per_Energy_source"]}
sim_param   =   complete_parameters(sim_param,Para_2_fill=Para_2_fill)

def f_Compute_conso(x,sim_param,Vecteur):
    conso_unitaire = x["conso_unitaire_"+Vecteur]
    return x["energy_need_per_"+sim_param["volume_variable_name"]] * x[sim_param["volume_variable_name"]]*conso_unitaire
def f_Compute_conso_totale(x,sim_param):
    res=0.
    for Vecteur in sim_param["Vecteurs"]:
        res+=x["conso_"+Vecteur]
    return res

for Vecteur in sim_param["Vecteurs"]:
    sim_param["f_Compute_conso_"+Vecteur]={"conso_"+Vecteur : partial(f_Compute_conso,Vecteur =Vecteur)}
sim_param["f_Compute_conso_totale"]={"Conso" : lambda x,sim_param: f_Compute_conso_totale(x,sim_param)}


def f_Compute_emissions(x,sim_param):
    emissions = 0
    for Vecteur_ in sim_param["Vecteurs"]: emissions += x["conso_unitaire_"+Vecteur_]*sim_param["Emissions_scope_2_3"][Vecteur_]
    emissions+= x["emissions_unitaire"]* x[sim_param["volume_variable_name"]]
    return emissions

def f_Compute_emissions_year(x,sim_param,year):
    emissions = 0
    for Vecteur_ in sim_param["Vecteurs"]: emissions += x["conso_unitaire_"+Vecteur_]*sim_param["Emissions_scope_2_3"][(Vecteur_,year)]
    emissions+= x["emissions_unitaire"]* x[sim_param["volume_variable_name"]]
    return emissions

if type(sim_param["Emissions_scope_2_3"].index) == pd.MultiIndex:
    sim_param["f_Compute_emissions"]= {"Emissions" : f_Compute_emissions_year }#{"Emissions" : partial(f_Compute_emissions,year =year)}
else:
    sim_param["f_Compute_emissions"]={"Emissions" : f_Compute_emissions}
sim_param_acier=sim_param
end = time.process_time()
print("Chargement des données, des modèles et interpolation terminés en : "+str(end-start)+" secondes")
#endregion



#region simulation
sim_stock_acier = launch_simulation(sim_param_acier)
#endregion
sim_stock_df = pd.concat(sim_stock, axis=0).reset_index().\
    rename(columns={"level_0":"year"}).set_index([ "year"  ,  Energy_system_name  , "old_new"])

sim_stock[2021].reset_index().Production_system.unique()

### categories pour avoir des groupes de couleurs dans les graphiques
col_class_dict={'BF-BOF' : 1, 'Bio-BF': 1, 'Bio-DRI-EAF': 1, 'CH4-DRI-EAF': 1, 'Coal-DRI-EAF': 1,
       'EAF': 1, 'EW-EAF': 1, 'H-DRI-EAF': 1, 'H2-BF': 1}

import plotly.express as px
Var = "Conso"
y_df = sim_stock_df.loc[(2021,slice(None),slice(None))].groupby([Energy_system_name])[Var].sum().to_frame().reset_index()
#y_df.loc[:,"Categorie"]=pd.MultiIndex.from_tuples([(str(col_class_dict[key]),key) for key in y_df.Categorie])
color_dict = gen_grouped_color_map(col_class_dict)
y_df["class"]=[col_class_dict[cat] for cat in y_df.Production_system]
y_df=y_df.sort_values(by=['class'])
fig = px.bar(y_df,x="class", y=Var, color="Production_system", title="Wide-Form Input",color_discrete_map=color_dict)
fig=fig.update_layout(title_text="Conso énergie finale (en TWh)", xaxis_title="Categorie",yaxis_title="Conso [TWh]")
#fig.show()
plotly.offline.plot(fig, filename=Graphic_folder+'file.html') ## offline


#region représentation des résultats
sim_stock_df = pd.concat(sim_stock, axis=0).reset_index().\
    rename(columns={"level_0":"year"}).set_index([ "year"  ,  Energy_system_name  , "old_new"])

Var = "Conso"
y_df = sim_stock_df.groupby(["year",Energy_system_name])[Var].sum().to_frame().reset_index().\
    pivot(index='year', columns=Energy_system_name).loc[[year for year in sim_param["years"][1:]],Var]/10**6
fig = MyStackedPlotly(y_df=y_df)
fig=fig.update_layout(title_text="Conso énergie finale par mode de chauffage (en TWh)", xaxis_title="Année",yaxis_title="Conso [TWh]")
plotly.offline.plot(fig, filename=Graphic_folder+'file.html') ## offline

Var = "Emissions"
y_df = sim_stock_df.groupby(["year",Energy_system_name])[Var].sum().to_frame().reset_index().\
    pivot(index='year', columns=Energy_system_name).loc[[year for year in sim_param["years"][1:]],Var]/10**6
fig = MyStackedPlotly(y_df=y_df)
fig=fig.update_layout(title_text="Emissions CO2 Mt", xaxis_title="Année",yaxis_title="Emissions [MtCO2]")
plotly.offline.plot(fig, filename=Graphic_folder+'file.html') ## offline

y_df = sim_stock_df.groupby(["year"])[["conso_"+Vec for Vec in sim_param["Vecteurs"]]].sum().loc[[year for year in sim_param["years"][1:]],:]/10**6
fig = MyStackedPlotly(y_df=y_df)
fig=fig.update_layout(title_text="Conso énergie finale par vecteur (en TWh)", xaxis_title="Année",yaxis_title="Conso [TWh]")
plotly.offline.plot(fig, filename=Graphic_folder+'file.html') ## offline
#fig.show()

#endregion

#region test
test_df = pd.DataFrame([[1,0,0,0,1,1],[0,1,0,0,1,1],[0,0,1,0,1,1],[0,0,0,1,1,1]],
                       columns = ["conso_unitaire_elec","conso_unitaire_gaz","conso_unitaire_fioul","conso_unitaire_bois",'energy_need_per_surface',"surface"])
print(test_df.apply(lambda x: sim_param['f_Compute_conso_bois']['conso_bois'](x,sim_param) ,axis =1))
print(test_df.apply(lambda x: sim_param['f_Compute_conso_elec']['conso_elec'](x,sim_param) ,axis =1))
#endregion