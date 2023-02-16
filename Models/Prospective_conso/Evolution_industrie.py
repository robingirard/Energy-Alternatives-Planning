
#region Chargement des packages
import pandas as pd
from EnergyAlternativesPlanning.f_tools import *
from EnergyAlternativesPlanning.f_graphicalTools import *
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
    for Vecteur_ in sim_param["Vecteurs"]: emissions += x["conso_unitaire_"+Vecteur_]* x[sim_param["volume_variable_name"]]*sim_param["Emissions_scope_2_3"][Vecteur_]
    emissions+= x["emissions_unitaire"]* x[sim_param["volume_variable_name"]]
    return emissions

def f_Compute_emissions_year(x,sim_param,year):
    emissions = 0
    for Vecteur_ in sim_param["Vecteurs"]: emissions += x["conso_unitaire_"+Vecteur_]* x[sim_param["volume_variable_name"]]*sim_param["Emissions_scope_2_3"][(Vecteur_,year)]
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
sim_param["retrofit_change_"+sim_param["volume_variable_name"]] = sim_param["retrofit_change_total_proportion_init_"+sim_param["volume_variable_name"]]/len(sim_param["years"]) *\
                                                                  sim_param["init_sim_stock"][sim_param["volume_variable_name"]]

Para_2_fill = {param : sim_param["base_index_year"] for param in ["retrofit_improvement","retrofit_change_"+sim_param["volume_variable_name"],
                                                                  "retrofit_Transition","energy_need_per_"+sim_param["volume_variable_name"]]}
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
    for Vecteur_ in sim_param["Vecteurs"]: emissions += x["conso_unitaire_"+Vecteur_]* x[sim_param["volume_variable_name"]]*sim_param["direct_emissions"][Vecteur_]
    for Vecteur_ in sim_param["Vecteurs"]: emissions += x["conso_unitaire_"+Vecteur_]* x[sim_param["volume_variable_name"]]*sim_param["indirect_emissions"][Vecteur_]
    emissions+= x["process_emissions_"+sim_param["volume_variable_name"]]* x[sim_param["volume_variable_name"]]
    return emissions

def f_Compute_emissions_year(x,sim_param,year):
    emissions = 0
    for Vecteur_ in sim_param["Vecteurs"]: emissions += x["conso_unitaire_"+Vecteur_]* x[sim_param["volume_variable_name"]]*sim_param["direct_emissions"][(Vecteur_,year)]
    for Vecteur_ in sim_param["Vecteurs"]: emissions += x["conso_unitaire_"+Vecteur_]* x[sim_param["volume_variable_name"]]*sim_param["indirect_emissions"][(Vecteur_,year)]
    emissions+= x["process_emissions_"+sim_param["volume_variable_name"]]* x[sim_param["volume_variable_name"]]
    return emissions

if type(sim_param["indirect_emissions"].index) == pd.MultiIndex:
    sim_param["f_Compute_emissions"]= {"Emissions" : f_Compute_emissions_year }#{"Emissions" : partial(f_Compute_emissions,year =year)}
else:
    sim_param["f_Compute_emissions"]={"Emissions" : f_Compute_emissions}
sim_param_ceramique=sim_param
end = time.process_time()
print("Chargement des données, des modèles et interpolation terminés en : "+str(end-start)+" secondes")
#endregion

#region chargement des données hydrogene
start = time.process_time()
target_year = 2050 # 2030 ou 2050 --> two excel sheets
dim_names=["Production_system","year","Vecteurs"];Index_names = ["Production_system"];Energy_system_name="Production_system"
data_set_from_excel =  pd.read_excel(Data_folder+"Hypotheses_hydrogen_1D_"+str(target_year)+".xlsx", None);
sim_param = extract_sim_param(data_set_from_excel,Index_names = Index_names,dim_names=dim_names,Energy_system_name=Energy_system_name)
sim_param["init_sim_stock"]=create_initial_parc(sim_param).sort_index()
sim_param["volume_variable_name"] = "unite_prod"
sim_param["retrofit_change_"+sim_param["volume_variable_name"]] = sim_param["retrofit_change_total_proportion_init_unite_prod"]/len(sim_param["years"]) *\
                                                                  sim_param["init_sim_stock"][sim_param["volume_variable_name"]]

Para_2_fill = {param : sim_param["base_index_year"] for param in ["new_energy","new_yearly_unite_prod","retrofit_improvement","retrofit_change_unite_prod","retrofit_Transition","energy_need_per_unite_prod"]}
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
    for Vecteur_ in sim_param["Vecteurs"]: emissions += x["conso_unitaire_"+Vecteur_]* x[sim_param["volume_variable_name"]]*sim_param["Emissions_scope_2_3"][Vecteur_]
    emissions+= x["emissions_unitaire"]* x[sim_param["volume_variable_name"]]
    return emissions

def f_Compute_emissions_year(x,sim_param,year):
    emissions = 0
    for Vecteur_ in sim_param["Vecteurs"]: emissions += x["conso_unitaire_"+Vecteur_]* x[sim_param["volume_variable_name"]]*sim_param["Emissions_scope_2_3"][(Vecteur_,year)]
    emissions+= x["emissions_unitaire"]* x[sim_param["volume_variable_name"]]
    return emissions

if type(sim_param["Emissions_scope_2_3"].index) == pd.MultiIndex:
    sim_param["f_Compute_emissions"]= {"Emissions" : f_Compute_emissions_year }#{"Emissions" : partial(f_Compute_emissions,year =year)}
else:
    sim_param["f_Compute_emissions"]={"Emissions" : f_Compute_emissions}
sim_param_H2=sim_param
end = time.process_time()
print("Chargement des données, des modèles et interpolation terminés en : "+str(end-start)+" secondes")
#endregion

#region chargement des données olefine
start = time.process_time()
target_year = 2050 # 2030 ou 2050 --> two excel sheets
dim_names=["Production_system","year","Vecteurs"];Index_names = ["Production_system"];Energy_system_name="Production_system"
data_set_from_excel =  pd.read_excel(Data_folder+"Hypotheses_olefines_1D_QR.xlsx", None);
sim_param = extract_sim_param(data_set_from_excel,Index_names = Index_names,dim_names=dim_names,Energy_system_name=Energy_system_name)
sim_param["init_sim_stock"]=create_initial_parc(sim_param).sort_index()
sim_param["volume_variable_name"] = "unite_prod"
sim_param["retrofit_change_"+sim_param["volume_variable_name"]] = sim_param["retrofit_change_total_proportion_init_unite_prod"]/len(sim_param["years"]) *\
                                                                  sim_param["init_sim_stock"][sim_param["volume_variable_name"]]

Para_2_fill = {param : sim_param["base_index_year"] for param in ["retrofit_improvement","retrofit_change_unite_prod","retrofit_Transition","energy_need_per_unite_prod",
                                                                                                   "new_energy","new_yearly_repartition_per_Energy_source"]}
sim_param   =   complete_parameters(sim_param,Para_2_fill=Para_2_fill)
sim_param = set_model_function_indus(sim_param)
sim_param_olefines=sim_param
end = time.process_time()
print("Chargement des données, des modèles et interpolation terminés en : "+str(end-start)+" secondes")
#endregion

#region chargement des données ammonia
start = time.process_time()
target_year = 2050 # 2030 ou 2050 --> two excel sheets
dim_names=["Production_system","year","Vecteurs"];Index_names = ["Production_system"];Energy_system_name="Production_system"
data_set_from_excel =  pd.read_excel(Data_folder+"Hypotheses_ammonia_1D_QR.xlsx", None);
sim_param = extract_sim_param(data_set_from_excel,Index_names = Index_names,dim_names=dim_names,Energy_system_name=Energy_system_name)
sim_param["init_sim_stock"]=create_initial_parc(sim_param).sort_index()
sim_param["volume_variable_name"] = "unite_prod"
sim_param["retrofit_change_"+sim_param["volume_variable_name"]] = sim_param["retrofit_change_total_proportion_init_unite_prod"]/len(sim_param["years"]) *\
                                                                  sim_param["init_sim_stock"][sim_param["volume_variable_name"]]

Para_2_fill = {param : sim_param["base_index_year"] for param in ["retrofit_improvement","retrofit_change_unite_prod","retrofit_Transition","energy_need_per_unite_prod",
                                                                                                   "new_energy","new_yearly_repartition_per_Energy_source"]}
sim_param   =   complete_parameters(sim_param,Para_2_fill=Para_2_fill)
sim_param = set_model_function_indus(sim_param)
sim_param_amonia=sim_param
end = time.process_time()
print("Chargement des données, des modèles et interpolation terminés en : "+str(end-start)+" secondes")
#endregion


#region simulation
sim_stock_acier = launch_simulation(sim_param_acier)
sim_stock_acier_df = pd.concat(sim_stock_acier, axis=0).reset_index().\
    rename(columns={"level_0":"year"}).set_index([ "year"  ,  Energy_system_name  , "old_new"])

sim_stock_ceramique = launch_simulation(sim_param_ceramique)
sim_stock_ceramique_df = pd.concat(sim_stock_ceramique, axis=0).reset_index().\
    rename(columns={"level_0":"year"}).set_index([ "year"  ,  Energy_system_name  , "old_new"])

sim_stock_H2 = launch_simulation(sim_param_H2)
sim_stock_stock_H2_df = pd.concat(sim_stock_H2, axis=0).reset_index().\
    rename(columns={"level_0":"year"}).set_index([ "year"  ,  Energy_system_name  , "old_new"])

sim_stock_olefines = launch_simulation(sim_param_olefines)
sim_stock_stock_olefines_df = pd.concat(sim_stock_olefines, axis=0).reset_index().\
    rename(columns={"level_0":"year"}).set_index([ "year"  ,  Energy_system_name  , "old_new"])

sim_stock_amonia = launch_simulation(sim_param_amonia)
sim_stock_stock_amonia_df = pd.concat(sim_stock_amonia, axis=0).reset_index().\
    rename(columns={"level_0":"year"}).set_index([ "year"  ,  Energy_system_name  , "old_new"])


sim_stock_df = pd.concat([sim_stock_acier_df,sim_stock_ceramique_df,sim_stock_stock_H2_df,sim_stock_stock_olefines_df,sim_stock_stock_amonia_df])

#endregion

#region etat initial
### categories pour avoir des groupes de couleurs dans les graphiques
col_class_dict={'BF-BOF' : 1, 'Bio-BF': 1, 'Bio-DRI-EAF': 1, 'CH4-DRI-EAF': 1, 'Coal-DRI-EAF': 1,
       'EAF': 1, 'EW-EAF': 1, 'H-DRI-EAF': 1, 'H2-BF': 1,
        'Séchoir microondes + Four biomasse':2,
       'Séchoir microondes + Four electrique':2,
       'Séchoir microondes + Four gaz':2,
       'Séchoir thermique + Four biomasse':2,
       'Séchoir thermique + Four electrique':2,
       'Séchoir thermique + Four gaz':2,
       'Séchoir thermique + RC + Four biomasse':2,
       'Séchoir thermique + RC + Four electrique':2,
       'Séchoir thermique + RC + Four gaz':2,
    'Biomass-Gasification':3,
       'Coal-Gasification':3, 'Electrolysis':3, 'Gas-Pyrolysis':3, 'SMR':3,
       'eSMR':3,
        'Biomass to Olefins':4, 'Biomass to Olefins - Eboiler':4,
       'Biomass-OCM':4, 'CO2-H2 to Olefins':4, 'CO2-H2 to Olefins - Eboiler':4,
       'CO2-H2-OCM':4, 'Coal to Olefins':4, 'Coal to Olefins - Eboiler':4,
       'Gas to Olefins':4, 'Gas to Olefins - Eboiler':4,
       'Naphtha steam cracking':4, 'Naphtha steam cracking - Eboiler':4,
       'Oxidative Coupling of Methane (OCM)':4,
        'Haber-Bosch' : 5, 'Molten Salt': 5, 'SSAS': 5, 'eNH3 (electrolysis)': 5
                }
#unité de production : kt.
# Energie : MWh/tk émissions en tCO2 par kt produit.

import plotly.express as px
Var = "Conso"
y_df = sim_stock_df.loc[(2020,slice(None),slice(None))].groupby([Energy_system_name])[Var].sum().to_frame().reset_index()
#y_df.loc[:,"Categorie"]=pd.MultiIndex.from_tuples([(str(col_class_dict[key]),key) for key in y_df.Categorie])
color_dict = gen_grouped_color_map(col_class_dict)
y_df["class"]=[col_class_dict[cat] for cat in y_df.Production_system]
y_df=y_df.sort_values(by=['class'])
y_df[Var]=y_df[Var]/10**6
fig = px.bar(y_df,x="class", y=Var, color="Production_system", title="Wide-Form Input",color_discrete_map=color_dict)
fig=fig.update_layout(title_text="Conso énergie finale (en TWh)", xaxis_title="Categorie",yaxis_title="Conso [TWh]")
#fig.show()
plotly.offline.plot(fig, filename=Graphic_folder+'file.html') ## offline

Var = "Emissions"
y_df = sim_stock_df.loc[(2020,slice(None),slice(None))].groupby([Energy_system_name])[Var].sum().to_frame().reset_index()
#y_df.loc[:,"Categorie"]=pd.MultiIndex.from_tuples([(str(col_class_dict[key]),key) for key in y_df.Categorie])
color_dict = gen_grouped_color_map(col_class_dict)
y_df["class"]=[col_class_dict[cat] for cat in y_df.Production_system]
y_df=y_df.sort_values(by=['class'])
y_df[Var]=y_df[Var]/10**6
fig = px.bar(y_df,x="class", y=Var, color="Production_system", title="Wide-Form Input",color_discrete_map=color_dict)
fig=fig.update_layout(title_text="Emissions CO2 [MTCO2]", xaxis_title="Categorie",yaxis_title="Emissions CO2 [MTCO2]")
#fig.show()
plotly.offline.plot(fig, filename=Graphic_folder+'file.html') ## offline
#endregion

#region représentation des résultats
Var = "Conso"
y_df = sim_stock_df.groupby(["year",Energy_system_name])[Var].sum().to_frame().reset_index().\
    pivot(index='year', columns=Energy_system_name).loc[[year for year in sim_param["years"][1:]],Var]/10**6
y_df.columns=pd.MultiIndex.from_tuples([(str(col_class_dict[key]),key) for key in y_df.columns])

fig = MyStackedPlotly(y_df=y_df)
fig=fig.update_layout(title_text="Conso énergie finale par mode de chauffage (en TWh)", xaxis_title="Année",yaxis_title="Conso [TWh]")
plotly.offline.plot(fig, filename=Graphic_folder+'file.html') ## offline

Var = "Emissions"
y_df = sim_stock_df.groupby(["year",Energy_system_name])[Var].sum().to_frame().reset_index().\
    pivot(index='year', columns=Energy_system_name).loc[[year for year in sim_param["years"][1:]],Var]/10**6
y_df.columns=pd.MultiIndex.from_tuples([(str(col_class_dict[key]),key) for key in y_df.columns])

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