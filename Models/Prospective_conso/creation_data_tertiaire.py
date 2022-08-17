import pandas as pd
from functions.f_graphicalTools import *
#region Chargement des packages

from functions.f_graphicalTools import *
from functions.f_tools import *
Data_folder = "Models/Prospective_conso/data/"
#endregion

pd.options.display.width = 0
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


##region creation données 1D
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

Description_Parc_initiale_2D = Description_Parc_initiale.reset_index().groupbyAndAgg(group_along= ["Categories","Energy_source"],
                                        aggregation_dic={"Surface" : "sum",
                                                         "Nouvelle_surface_par_an":"first"}).set_index(["Categories","Energy_source"])

Description_Parc_initiale_1D = Description_Parc_initiale.reset_index().groupbyAndAgg(group_along= ["Energy_source"],
                                        aggregation_dic={"Conso" : "sum",
                                                        "Surface" : "sum",
                                                        "COP" : "first",
                                                        "Besoin_surfacique" : "wmean",
                                                        "proportion_besoin_chauffage": "wmean"
                                                         },
                                        weightedMean_weight="Surface").set_index(["Energy_source"]).\
    merge(Description_Parc_initiale.reset_index().groupbyAndAgg(group_along= ["Energy_source"],
                                        aggregation_dic={ "Repartition_chauffage_neuf_premieres_annees": "wmean",
                                                         "Repartition_chauffage_neuf_regime_normal": "wmean"
                                                         },
                                        weightedMean_weight="Nouvelle_surface_par_an").set_index(["Energy_source"]),
          left_index=True, right_index=True)




Transition = data_tertiaire["Transition"].set_index([ "Categories","Energy_source"]).\
    merge(Description_Parc_initiale_2D,how = 'outer',left_index=True,right_index=True).reset_index().\
    groupbyAndAgg(group_along= ["Energy_source"],
                                        aggregation_dic={chauff : "wmean" for chauff in Description_Parc_initiale_1D.index},
                                        weightedMean_weight="Surface").set_index(["Energy_source"])

Description_Parc_initiale["dummy"]=1
Description_Parc_initiale_0D = Description_Parc_initiale.reset_index().groupbyAndAgg(group_along= ["dummy"],
                                        aggregation_dic={"Conso" : "sum",
                                                        "Surface" : "sum",
                                                        "COP" : "first",
                                                        "Besoin_surfacique" : "wmean",
                                                         "proportion_besoin_chauffage": "wmean",
                                                         },
                                        weightedMean_weight="Surface")
Description_Parc_initiale_0D["Nouvelle_surface_par_an"] = data_tertiaire["Categories"]["Nouvelle_surface_par_an"].sum()
with pd.ExcelWriter(Data_folder + "Hypotheses_tertiaire_1D.xlsx") as writer:
    Description_Parc_initiale_1D[["Surface", "COP",
                                  "Repartition_chauffage_neuf_premieres_annees",
                                  "Repartition_chauffage_neuf_regime_normal"
                                  ]].to_excel(writer,sheet_name='Energy_source')
    Description_Parc_initiale_0D[["Besoin_surfacique" , "proportion_besoin_chauffage","Nouvelle_surface_par_an"
                                  ]].to_excel(writer, sheet_name='0D')
    Transition.to_excel(writer, sheet_name='Transition')

#endregion

##region creation données 2D
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

Description_Parc_initiale_2D = Description_Parc_initiale.reset_index().groupbyAndAgg(group_along= ["Categories","Energy_source"],
                                        aggregation_dic={"Energy_source_per_Category" : "sum",
                                                        "Surface" : "sum",
                                                        "COP" : "first",
                                                        "Besoin_surfacique" : "wmean",
                                                         "proportion_besoin_chauffage": "wmean",
                                                         },
                                        weightedMean_weight="Surface").set_index(["Categories","Energy_source"])

Transition = data_tertiaire["Transition"].set_index([ "Categories","Energy_source"])

Description_Parc_initiale_1D_Categories = Description_Parc_initiale.reset_index().groupbyAndAgg(group_along= ["Categories"],
                                        aggregation_dic={"Surface" : "sum",
                                                        "Besoin_surfacique" : "wmean",
                                                         "proportion_besoin_chauffage": "wmean",
                                                         },
                                        weightedMean_weight="Surface").set_index(["Categories"])

Description_Parc_initiale_1D_Energy_source = Description_Parc_initiale.reset_index().groupbyAndAgg(group_along= ["Energy_source"],
                                        aggregation_dic={"COP" : "first"}).set_index(["Energy_source"])


with pd.ExcelWriter(Data_folder + "Hypotheses_tertiaire_2D.xlsx") as writer:
    Description_Parc_initiale_2D[["Surface", "COP"]].to_excel(writer,sheet_name='Energy_source')
    Description_Parc_initiale_1D[["Besoin_surfacique" , "proportion_besoin_chauffage"]].to_excel(writer, sheet_name='0D')
    Description_Parc_initiale_1D_Energy_source.to_excel(writer, sheet_name='Energy_source')
    Transition.to_excel(writer, sheet_name='Transition')

#endregion


#region données neuf
## Neuf
data_tertiaire =  pd.read_excel(Data_folder+"Hypotheses_tertiaire_3D.xlsx",
                            sheet_name=["Categories","Efficiency_class","Energy_source","Transition",
                                        "Categories_Energy_source","Categories_Efficiency_class"])
Energy_source = data_tertiaire["Energy_source"]
Categories = data_tertiaire["Categories"]
Repartition_chauffage0=pd.DataFrame([[0.02,0,0.19,0.04,0.08,0.61,0.06,0],
                 [0.02,0,0.22,0.02,0.2,0.43,0.11,0],
                 [0.04,0,0.54,0.09,0.07,0.21,0.05,0],
                 [0.01,0,0.15,0,0.19,0.48,0.17,0],
                 [0.02,0,0.22,0.02,0.2,0.43,0.11,0],
                 [0.02,0,0.22,0.02,0.2,0.43,0.11,0],
                 [0.02,0,0.22,0.02,0.2,0.38,0.16,0],
                 [0.02,0,0.22,0.02,0.2,0.36,0.18,0]])
Repartition_chauffage0=Repartition_chauffage0.rename(columns = Energy_source.Energy_source).\
    assign(Categories=Categories.Categories).set_index("Categories").\
    melt(ignore_index=False,var_name="Energy_source",value_name="Repartition_chauffage_neuf_premieres_annees").\
    set_index("Energy_source",append=True)
##le neuf comme aujourd'hui
Repartition_chauffage1=pd.DataFrame([[0.03,0,0,0.2,0.02,0.68,0.07,0],
                [0.02,0,0,0.05,0.1,0.67,0.16,0],
                [0.04,0,0,0.2,0.05,0.56,0.15,0],
                [0.02,0,0,0.1,0.1,0.58,0.2,0],
                [0.02,0,0,0.2,0,0.63,0.15,0],
                [0.02,0,0,0.2,0,0.63,0.15,0],
                [0.15,0,0,0.2,0,0.45,0.2,0],
                [0.04,0,0,0.2,0,0.52,0.24,0]])

Repartition_chauffage=Repartition_chauffage1.rename(columns = Energy_source.Energy_source).\
    assign(Categories=Categories.Categories).set_index("Categories").\
    melt(ignore_index=False,var_name="Energy_source",value_name="Repartition_chauffage_neuf_regime_normal"). \
    set_index("Energy_source", append=True).\
    merge(Repartition_chauffage0, left_index=True, right_index=True)
Repartition_chauffage.to_csv(Data_folder+"tmp.csv")

S_new_year=[1730417,
526649,
1655181,
1429475,
526649,
902826,
526649,
225707]

#endregion
