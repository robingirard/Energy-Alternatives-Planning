from math import gamma

import pandas as pd

from functions.f_tools import *
#from scipy.optimize import minimize, rosen, rosen_der
import scipy.optimize as scop
#renovation et changement de mode de chauffage

def extract_simulation_parameters(data_set_from_excel,sheet_name_and_dim,Index_names = ["Energy_source"],Energy_source_name="Energy_source"):
    simulation_parameters={}
    for key in sheet_name_and_dim.keys():
        if ~(key == "retrofit_Transition"):
            if len(sheet_name_and_dim[key])>0:
                for col in data_set_from_excel[key].set_index(sheet_name_and_dim[key]).columns:
                    simulation_parameters[col] = data_set_from_excel[key].set_index(sheet_name_and_dim[key])[col]
            else:
                simulation_parameters_tmp = data_set_from_excel[key].set_index('Nom').to_dict()['Valeur']
                simulation_parameters = {**simulation_parameters, **simulation_parameters_tmp}

    simulation_parameters["retrofit_Transition"] = data_set_from_excel["retrofit_Transition"].set_index(sheet_name_and_dim["retrofit_Transition"])
    simulation_parameters["Index_names"]=Index_names
    Complementary_Index_names=Index_names.copy(); Complementary_Index_names.remove(Energy_source_name)
    simulation_parameters["Energy_source_name"]=Energy_source_name
    simulation_parameters["Complementary_Index_names"]=Complementary_Index_names
    data_set_from_excel["years"]=list(range(int(simulation_parameters["date_debut"]),int(simulation_parameters["date_fin"])))
    simulation_parameters["years"]=data_set_from_excel["years"]
    My_dict = {key: data_set_from_excel[key][key] for key in Index_names}
    simulation_parameters["type_index_year"] =  expand_grid_from_dict({**My_dict,"year" : simulation_parameters["years"]},as_MultiIndex=True)
    simulation_parameters["type_index_year_new"] =  expand_grid_from_dict({**My_dict,"year" : simulation_parameters["years"],"old_new":"new"},as_MultiIndex=True)
    simulation_parameters["type_index_year_old"] =  expand_grid_from_dict({**My_dict,"year" : simulation_parameters["years"],"old_new":"old"},as_MultiIndex=True)
    simulation_parameters["type_index"] = expand_grid_from_dict(My_dict={ key :  data_set_from_excel[key][key] for key in Index_names },as_MultiIndex=True)
    simulation_parameters["Energy_source_index"] = data_set_from_excel[Energy_source_name][Energy_source_name]
    if len(simulation_parameters["Complementary_Index_names"])>0:
        simulation_parameters["Complementary_index"] = expand_grid_from_dict(My_dict={ key :  data_set_from_excel[key][key] for key in simulation_parameters["Complementary_Index_names"] },as_MultiIndex=True)
        simulation_parameters["Complementary_index_tuple"] = tuple([slice(None)] * len(list(simulation_parameters["Complementary_index"].to_frame().columns)))
    simulation_parameters["type_index_tuple"] = tuple([slice(None)] * len(list(simulation_parameters["type_index"].to_frame().columns)))

    return simulation_parameters

def fill_simulation_parameters(simulation_parameters,Param_names):
    for key in Param_names:
        simulation_parameters[key] = pd.DataFrame(None, index=simulation_parameters["type_index_year"]).\
            merge(simulation_parameters[key], how='outer', left_index=True, right_index=True)
    return simulation_parameters
#key="retrofit_improvement"
def complete_parameters(simulation_parameters,Para_2_fill=["retrofit_improvement","retrofit_Transition"]):

    simulation_parameters = interpolate_simulation_parameters(simulation_parameters)
    simulation_parameters = fill_simulation_parameters(simulation_parameters,Param_names = Para_2_fill)

    #particular processing for "retrofit_Transition" : kind of matrix transpose
    past_dim = simulation_parameters["retrofit_Transition"].index.names
    Other_dims = list(map(lambda x: x.replace(simulation_parameters["Energy_source_name"] , simulation_parameters["Energy_source_name"] + "_out"), past_dim))
    simulation_parameters["retrofit_Transition"] = simulation_parameters["retrofit_Transition"].reset_index().assign(old_new="new").set_index(past_dim+  ["old_new"]). \
        melt(ignore_index=False, var_name=simulation_parameters["Energy_source_name"]+"_out", value_name="retrofit_Transition").set_index(
        [simulation_parameters["Energy_source_name"]+"_out"], append=True).\
        pivot_table(values='retrofit_Transition',index=Other_dims+  ["old_new"],columns=simulation_parameters["Energy_source_name"]).\
        reset_index().rename(columns={simulation_parameters["Energy_source_name"]+"_out": simulation_parameters["Energy_source_name"] }).set_index(past_dim+  ["old_new"])
    return simulation_parameters


#data_series_to_transition= simulation_parameters["init_Description_Parc"]["total_surface"]
def apply_transition(Surface_a_renover,Transition,simulation_parameters):
    Name = Surface_a_renover.name
    if type(Name)==type(None): Name=0
    X_df=Surface_a_renover.copy().to_frame().pivot_table(values = Name,index=simulation_parameters["Complementary_Index_names"],columns=simulation_parameters["Energy_source_name"])
    res = Surface_a_renover.copy()*0
    #res = res.reset_index().assign(old_new="new").set_index(simulation_parameters["Complementary_Index_names"]+[simulation_parameters["Energy_source_name"]]+  ["old_new"])[Name]
    for  Energy_source in simulation_parameters["Energy_source_index"]:
        if len(X_df.loc[:,Energy_source])==1:
            res += Transition.loc[:, Energy_source] * float(X_df.loc[:,Energy_source])  # implicite merge because Energy_source_index is not in X_df rows but is in transition and res
        else:
            res += Transition.loc[:, Energy_source] * X_df.loc[:,Energy_source]  # implicite merge because Energy_source_index is not in X_df rows but is in transition and res

    return res

def interpolate_simulation_parameters(simulation_parameters):
    #interpolations
    for key in simulation_parameters.keys():
        if isinstance(simulation_parameters[key], pd.Series):
            if simulation_parameters[key].index.name == "year":
                simulation_parameters[key] = simulation_parameters[key].interpolate()
            elif len(simulation_parameters[key].index.names)>1:
                if ("year" in list(simulation_parameters[key].index.to_frame().columns)):
                    years = range(int(simulation_parameters["date_debut"]), int(simulation_parameters["date_fin"]))
                    simulation_parameters[key] = simulation_parameters[key].interpolate_along_one_index_column(new_x_values = years,x_columns_for_interpolation = "year")
        elif isinstance(simulation_parameters[key], pd.DataFrame):
            if ("year" in list(simulation_parameters[key].index.to_frame().columns)):
                years = range(int(simulation_parameters["date_debut"]), int(simulation_parameters["date_fin"]))
                simulation_parameters[key] = simulation_parameters[key].interpolate_along_one_index_column(new_x_values = years,x_columns_for_interpolation = "year")
    return simulation_parameters

def interpolate_along_one_index_column(df,new_x_values,x_columns_for_interpolation = "year"):
    df=df.copy()
    isSeries=False
    if type(df)==pd.Series:
        isSeries = True
        df = df.to_frame()
    res = pd.DataFrame(None)
    x_columns_for_interpolation_input_values = df.reset_index()[x_columns_for_interpolation].unique()
    Index_name = list(df.index.to_frame().drop(columns=[x_columns_for_interpolation]).columns)
    Index_set = df.index.to_frame()[Index_name].drop_duplicates().reset_index(drop=True)
    if len(x_columns_for_interpolation_input_values) == 1:
        # only one year in data, create a duplicate of first year to put in first and last year of simulation
        to_append_df = df.reset_index().copy()
        to_append_df[x_columns_for_interpolation]=new_x_values[-1]
        to_append_df=to_append_df.set_index(df.index.names)[df.columns]
        df=df.append(to_append_df)
        for col in df.columns:
            df.loc[cat_tuple(tuple(len(Index_set.columns) * [slice(None)]), new_x_values[-1]),col] = df.loc[cat_tuple(tuple(len(Index_set.columns) * [slice(None)]), float(x_columns_for_interpolation_input_values)),col]
    for key in df.columns:
        f_out = {}
        if len(Index_set.columns) == 1:
            for index, value in Index_set.iterrows():
                My_index = value.values[0]
                f_in = [df.loc[(My_index, year),key] for year in x_columns_for_interpolation_input_values]
                f_out[My_index] = pd.DataFrame.from_dict(
                    {key: np.interp(new_x_values, x_columns_for_interpolation_input_values, f_in), x_columns_for_interpolation: new_x_values, Index_name[0]: My_index})
        else:
            for index, value in Index_set.iterrows():
                My_index = value.values
                f_in = [df.loc[(*My_index, year),key] for year in x_columns_for_interpolation_input_values]
                TMP_d = {Index_name[i]: My_index[i] for i in range(len(My_index))}
                f_out[tuple(My_index)] = pd.DataFrame.from_dict(
                    {key: np.interp(new_x_values, x_columns_for_interpolation_input_values, f_in), x_columns_for_interpolation: new_x_values, **TMP_d})
        res[key] = pd.concat(f_out, axis=0).set_index(Index_name + [x_columns_for_interpolation])[[key]]
    if isSeries: return res[key]
    else: return res
pd.DataFrame.interpolate_along_one_index_column = interpolate_along_one_index_column
pd.Series.interpolate_along_one_index_column = interpolate_along_one_index_column

def create_initial_parc(simulation_parameters):
    """
    Create a data frame indexed by simulation_parameters["type_index"] with columns obtained from all
    variables indexed by "init_..."
    :param simulation_parameters:
    :return:
    """
    col_names = []
    res = pd.DataFrame(None, index=simulation_parameters["type_index"])
    index_names = list(simulation_parameters["type_index"].to_frame().columns)
    for key in simulation_parameters.keys():
        if len(key)>5:
            if key[0: 5]=="init_":
                #print(key)
                if type(simulation_parameters[key])==float:
                    res = res.merge(pd.DataFrame([simulation_parameters[key]]*len(simulation_parameters["type_index"]),
                                    index=simulation_parameters["type_index"]) , how='outer', left_index=True, right_index=True)
                else:
                    res =  res.merge(simulation_parameters[key],how = 'outer',left_index=True,right_index=True)
                col_names = col_names + [key[5:len(key)]]

    res.columns = col_names
    return res


def calibrate_simulation_parameters(simulation_parameters,calibration_parameters):

    ### calibration
    for key in calibration_parameters.keys():
        simulation_parameters[key+"_alpha"] =find_optimal_alpha(
                    total_change_target=calibration_parameters[key]["total_change_target"],
                    Change_func=simulation_parameters[key],
                    simulation_parameters=simulation_parameters)["x"]
    return simulation_parameters



def total_Change_distance_to_target(alpha,total_change_target,Change_func,simulation_parameters):
    """
    Compute the total change of function Change_func with parameter alpha over all types and all simulation period
    :param alpha:
    :param Change_func:
    :param simulation_parameters:
    :return:
    """
    #total_change_target = simulation_parameters["retrofit_total_change_target"]
    Total_change = pd.Series(0.,index=simulation_parameters["type_index"])
    for year in range(int(simulation_parameters["date_debut"])+1,int(simulation_parameters["date_fin"])):
    #    for archetype in Total_change.index:
    #        Total_change[archetype]+=Change_func(year,archetype,alpha,simulation_parameters)
        Total_change+=Change_func(year=year,alpha=alpha,simulation_parameters=simulation_parameters)
    error = ((Total_change-total_change_target)**2)
    if len(error)>1:
        res=((Total_change-total_change_target)**2).sum()
    else: res = error
    return res

def find_optimal_alpha(total_change_target,Change_func,simulation_parameters,alpha_0=1.,gtol= 1e-3, disp=False):
    res = scop.minimize(total_Change_distance_to_target, alpha_0, method='BFGS',
                   args = (total_change_target,Change_func,simulation_parameters),
                   options={'gtol': gtol, 'disp': disp})
    return res


def initialize_Simulation(simulation_parameters):
    """
    Creates a dictionnary with Description_parc and fill it the initial year data.
    :param simulation_parameters:
    :return:
    """
    Description_Parc = {}
    year=int(simulation_parameters["date_debut"])
    Description_Parc[year]=pd.concat( [ simulation_parameters["init_Description_Parc"].assign(old_new="old"),
                                    simulation_parameters["init_Description_Parc"].assign(old_new="new")]).set_index(['old_new'], append=True).sort_index()
    Description_Parc[year].loc[(*simulation_parameters["type_index_tuple"],"new"),"Surface"]=0 ## on commence sans "neuf"

    return Description_Parc

def loanch_simulation(simulation_parameters):
    Description_Parc = initialize_Simulation(simulation_parameters)
    for year in progressbar(range(int(simulation_parameters["date_debut"]) + 1, int(simulation_parameters["date_fin"])), "Computing: ", 40):
        Description_Parc[year] = Description_Parc[year-1].copy()

        #destruction
        Description_Parc[year].loc[:,"Surface"] -= simulation_parameters["old_taux_disp"][year] * Description_Parc[year].loc[:, "Surface"]

        #renovation
        Surface_a_renover = simulation_parameters["f_retrofit_total_yearly_surface"](year, simulation_parameters["f_retrofit_total_yearly_surface_alpha"], simulation_parameters)
        Surfaces_restantes = soustrait_mais_reste_positif(Description_Parc[year].loc[(*simulation_parameters["type_index_tuple"], "old"), "Surface"],Surface_a_renover)
        Surface_a_renover = (Description_Parc[year]["Surface"].loc[(*simulation_parameters["type_index_tuple"], "old")] - Surfaces_restantes.remove_index_from_name(index_name="old_new"))
        Transition = simulation_parameters["retrofit_Transition"].loc[(*simulation_parameters["type_index_tuple"], year, "new"), :].remove_index_from_name("year").remove_index_from_name("old_new")
        Description_Parc[year] = update_surface_heat_need(Description_Parc_year=Description_Parc[year],
                                          Nouvelles_surfaces=apply_transition(Surface_a_renover,Transition,simulation_parameters),
                                          Nouveau_besoin=(1 -simulation_parameters["retrofit_improvement"]["retrofit_improvement"].loc[(*simulation_parameters["type_index_tuple"],year)]) * \
                                                         Description_Parc[year - 1]["Besoin_surfacique"].loc[(*simulation_parameters["type_index_tuple"],"old")],
                                          simulation_parameters=simulation_parameters)
        Description_Parc[year].loc[(*simulation_parameters["type_index_tuple"], "old"), "Surface"] = Surfaces_restantes

        #neuf
        Description_Parc[year] = update_surface_heat_need(Description_Parc_year=Description_Parc[year],
            Nouvelles_surfaces=simulation_parameters["new_yearly_surface"].loc[(*simulation_parameters["type_index_tuple"], year, "new")],
            Nouveau_besoin=simulation_parameters["new_energy"].loc[(*simulation_parameters["type_index_tuple"], year, "new")],
            simulation_parameters=simulation_parameters)

        Description_Parc[year]  = Description_Parc[year].assign(Conso=lambda x: simulation_parameters["f_Compute_conso"](x)).fillna(0)
        Description_Parc[year]  = Description_Parc[year].assign(Besoin=lambda x: simulation_parameters["f_Compute_besoin"](x)).fillna(0)
        for vecteur in ["elec", "gaz", "fioul", "bois"]:
            Description_Parc[year]["Conso_"+vecteur]=Description_Parc[year].apply(lambda x: simulation_parameters["f_Compute_conso"](x,vecteur), axis = 1).fillna(0)
    return Description_Parc

#lorsque l'on met à jour l'ensemble des surfaces et le besoin associé
def update_surface_heat_need(Description_Parc_year,Nouvelles_surfaces,Nouveau_besoin,simulation_parameters):

    #mise à jour des surfaces
    old_col_names=Nouvelles_surfaces.name
    if type(old_col_names) == type(None): old_col_names = 0
    Nouvelles_surfaces_neuf = Nouvelles_surfaces.loc[simulation_parameters["type_index_tuple"]]
    Description_Parc_year.loc[(*simulation_parameters["type_index_tuple"], "new"), "Surface"] +=Nouvelles_surfaces_neuf.to_frame().assign(old_new="new").set_index(["old_new"],append=True)[old_col_names]

    #mise à jouro du besoin
    Ancien_besoin_neuf = Description_Parc_year["Besoin_surfacique"].loc[(*simulation_parameters["type_index_tuple"], "new")]
    Anciennes_surfaces = Description_Parc_year["Surface"].loc[(*simulation_parameters["type_index_tuple"], "new")]
    Nouveau_besoin_neuf = Description_Parc_year["Besoin_surfacique"].loc[(*simulation_parameters["type_index_tuple"], "new")].copy()
    sub_index = Nouvelles_surfaces_neuf > 0
    Nouveau_besoin_neuf.loc[sub_index] = (Ancien_besoin_neuf.loc[sub_index] * Anciennes_surfaces.loc[sub_index] + Nouveau_besoin.loc[sub_index] *Nouvelles_surfaces_neuf.loc[sub_index]) / \
                                                                (Anciennes_surfaces.loc[sub_index]+Nouvelles_surfaces_neuf.loc[sub_index])
    Nouveau_besoin_neuf=Nouveau_besoin_neuf.to_frame().assign(old_new="new").set_index(["old_new"],append=True)["Besoin_surfacique"]
    Description_Parc_year.loc[(*simulation_parameters["type_index_tuple"], "new"),"Besoin_surfacique"] =Nouveau_besoin_neuf

    return Description_Parc_year

#import time
#start = time.process_time()

#end = time.process_time()
#end-start

def retrofit_yearly_surface_repartition(year, archetype,simulation_parameters):
    Taille_index = len(list(simulation_parameters["type_index"].to_frame().columns))
    #Index_name = list(simulation_parameters["type_index"].to_frame().columns)

    Surfaces_renovee_par_type = pd.Series(0,index=simulation_parameters["type_index"]).to_frame().\
        rename(columns = {0:"Surface"}).assign(old_new="new").set_index("old_new",append=True)
    #for new_archetype in simulation_parameters["type_index"]:
    #    Surfaces_renovee_par_type.loc[cat_tuple(new_archetype, "new"),"Surface"] = Surface_renovee * simulation_parameters["get_retrofit_Transition"](year,archetype,new_archetype,simulation_parameters)
    Surfaces_renovee_par_type.loc[(*tuple([slice(None)]*Taille_index), "new"), "Surface"] =  simulation_parameters["get_retrofit_Transition"](year,archetype,simulation_parameters)
    return Surfaces_renovee_par_type



def retrofit(Description_Parc,year,simulation_parameters):

    ## estimation des surfaces à rénover, on ne peut pas rénover plus que la surface restante
    Surface_a_renover   = simulation_parameters["f_retrofit_total_yearly_surface"](year, simulation_parameters["f_retrofit_total_yearly_surface_alpha"],simulation_parameters)
    Surfaces_restantes = soustrait_mais_reste_positif(Description_Parc[year].loc[(*simulation_parameters["type_index_tuple"], "old"), "Surface"],Surface_a_renover)
    Surface_a_renover = (Description_Parc[year]["Surface"].loc[(*simulation_parameters["type_index_tuple"], "old")]-Surfaces_restantes.remove_index_from_name(index_name="old_new"))
    Transition    = simulation_parameters["retrofit_Transition"].loc[(*simulation_parameters["type_index_tuple"], year, "new"), :].remove_index_from_name("year").remove_index_from_name("old_new")
    Description_Parc[year]=update_surface_heat_need(Description_Parc_year=Description_Parc[year],
                Nouvelles_surfaces=apply_transition(Surface_a_renover,Transition,simulation_parameters),
                Nouveau_besoin=(1 - simulation_parameters["retrofit_improvement"]["retrofit_improvement"].loc[(*simulation_parameters["type_index_tuple"],year)]) * \
                            Description_Parc[year - 1]["Besoin_surfacique"].loc[(*simulation_parameters["type_index_tuple"], "old")],
                simulation_parameters=simulation_parameters)
    Description_Parc[year].loc[(*simulation_parameters["type_index_tuple"], "old"), "Surface"] =Surfaces_restantes

    return Description_Parc

def Construction_neuf(Description_Parc,year,simulation_parameters):
    Nouvelles_Surfaces =simulation_parameters["new_yearly_surface"].loc[(*simulation_parameters["type_index_tuple"], year,"new")]
    Nouveau_besoin = simulation_parameters["new_energy"].loc[(*simulation_parameters["type_index_tuple"], year,"new")]
    Description_Parc[year] = update_surface_heat_need(Description_Parc_year=Description_Parc[year],
                                                      Nouvelles_surfaces=Nouvelles_Surfaces,
                                                      Nouveau_besoin=Nouveau_besoin,
                                                      simulation_parameters=simulation_parameters)
    return Description_Parc



#def Destruction_ancien(Description_Parc,year,simulation_parameters):
#    TMP_df=pd.DataFrame(None)
#    TMP_df["New_surface"]=Description_Parc[year].loc[:,"Surface"]-simulation_parameters["old_taux_disp"][year] * Description_Parc[year].loc[:, "Surface"]
#    TMP_df["0"]=0
#    Description_Parc[year].loc[:,"Surface"] = TMP_df.max(axis=1)
#    return Description_Parc
#endregion

def cat_tuple(tuple1,tuple2):
    index=0
    if type(tuple1) == tuple:
        if type(tuple2)==tuple:
            index = tuple1 + tuple2
        elif type(tuple2) in [str,float,int]:
            index = tuple1 + (tuple2,)
        else: print("cat_tuple implemented only for tuple or str,float,int")
    elif type(tuple1) == str:
        if type(tuple2) == tuple:
            index = (tuple1,) + tuple2
        elif type(tuple2) in [str,float,int]:
            index = (tuple1,tuple2)
        else:
            print("cat_tuple implemented only for tuple or str,float,int")
    else:
        print("cat_tuple implemented only for tuple or str,float,int")
    return index



def age(alpha, d):
    """
    calcul l'espérance de vie d'un véhicule
    pour lequel la probabilité de mort à l'année i est 1 / gamma(1 + alpha * (i + 1)) pour i entre
    1 et d la durée maximale de vie.
    :param alpha:
    :param d:
    :return:
    """
    s = sum([1 / gamma(1 + alpha * (i + 1)) for i in range(d)])
    return sum([(i + 1 / 2) * 1 / gamma(1 + alpha * (i + 1)) for i in range(d)]) / s


def find_alpha(age_moy, duree_vie):
    """
    trouve par dychotomie alpha tq |E[X_alpha] - age_moy|< eps
    où P(X_alpha =i) ~ 1/gamma(1+alpha(1+i))
    alpha == 1 :  P(X_alpha =i) = (1+i)!
    :param age_moy:
    :param duree_vie:
    :return:
    """
    d = 2 * duree_vie
    alpha0 = 0
    alpha1 = 1
    eps = 1e-3
    while age(alpha1, d) > age_moy:
        alpha1 += 1
    while age(alpha0, d) > age_moy + eps:
        alpha = (alpha0 + alpha1) / 2
        if age(alpha, d) > age_moy:
            alpha0 = alpha
        else:
            alpha1 = alpha
    return alpha0

#find_alpha(1,40)