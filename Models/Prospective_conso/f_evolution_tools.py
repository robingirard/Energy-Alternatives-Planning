from math import gamma
import inspect
import pandas as pd

from functions.f_tools import *
#from scipy.optimize import minimize, rosen, rosen_der
import scipy.optimize as scop
#renovation et changement de mode de chauffage
def get_colnames_in_dim_names(df,dim_names):
    colnames = df.columns
    res = []
    for col in colnames:
        if col in dim_names : res=res+[col]
    return res

def get_index_vals(data_set_from_excel,key):
    for tmp_key in data_set_from_excel.keys():
        colnames = data_set_from_excel[tmp_key].columns
        res = []
        for col in colnames:
            if col ==key :
                res=data_set_from_excel[tmp_key][key].unique()
                return res

def extract_sim_param(data_set_from_excel,Index_names = ["Energy_source"],
                      dim_names=["Energy_source","year"],Energy_system_name="Energy_source"):

    sheet_name_and_dim = {key : get_colnames_in_dim_names(data_set_from_excel[key],dim_names) for key in data_set_from_excel.keys()}
    sheet_name = list(sheet_name_and_dim.keys())
    sim_param={}



    for key in sheet_name_and_dim.keys():
        if ~(key == "retrofit_Transition"):
            if len(sheet_name_and_dim[key])>0:
                for col in data_set_from_excel[key].set_index(sheet_name_and_dim[key]).columns:
                    sim_param[col] = data_set_from_excel[key].set_index(sheet_name_and_dim[key])[col]
            else:
                sim_param_tmp = data_set_from_excel[key].set_index('Nom').to_dict()['Valeur']
                sim_param = {**sim_param, **sim_param_tmp}

    sim_param["retrofit_Transition"] = data_set_from_excel["retrofit_Transition"].set_index(sheet_name_and_dim["retrofit_Transition"])
    sim_param["Index_names"]=Index_names
    Complementary_Index_names=Index_names.copy(); Complementary_Index_names.remove(Energy_system_name)
    sim_param["Energy_system_name"]=Energy_system_name
    sim_param["Complementary_Index_names"]=Complementary_Index_names

    Index_val_dict = {key: get_index_vals(data_set_from_excel,key) for key in Index_names}
    if "date_step" in sim_param:
        sim_param["years"]=list(range(int(sim_param["date_debut"]),int(sim_param["date_fin"]),int(sim_param["date_step"])))
    else:
        sim_param["years"]=list(range(int(sim_param["date_debut"]),int(sim_param["date_fin"])))
    # data_set_from_excel["years"]=sim_param["years"]

    sim_param["base_index_year"] =  expand_grid_from_dict({**Index_val_dict,"year" : sim_param["years"]},as_MultiIndex=True)
    sim_param["base_index_year_new"] =  expand_grid_from_dict({**Index_val_dict,"year" : sim_param["years"],"old_new":"new"},as_MultiIndex=True)
    sim_param["base_index_year_old"] =  expand_grid_from_dict({**Index_val_dict,"year" : sim_param["years"],"old_new":"old"},as_MultiIndex=True)
    if len(sim_param["Index_names"]) == 1:
        sim_param["base_index"]=data_set_from_excel[sim_param["Index_names"][0]][sim_param["Index_names"][0]]
    else:     sim_param["base_index"] = expand_grid_from_dict(Index_val_dict,as_MultiIndex=True)


    sim_param["Energy_source_index"] = data_set_from_excel[Energy_system_name][Energy_system_name]
    if len(sim_param["Complementary_Index_names"])>0:
        sim_param["Complementary_index"] = expand_grid_from_dict(My_dict={ key :  Index_val_dict[key] for key in sim_param["Complementary_Index_names"] },as_MultiIndex=True)
        sim_param["Complementary_index_tuple"] = tuple([slice(None)] * len(list(sim_param["Complementary_index"].to_frame().columns)))
    sim_param["base_index_tuple"] = tuple([slice(None)] * len(list(sim_param["base_index"].to_frame().columns)))

    sim_param["Vecteurs_"] = []
    for key in sim_param:
        if len(key) > len("init_conso_unitaire_"):
            if key[0:len("init_conso_unitaire_")] == "init_conso_unitaire_":
                sim_param["Vecteurs_"] += [key[len("init_conso_unitaire_"):len(key)]]
    if not ("Vecteurs" in sim_param):
        sim_param["Vecteurs"]=sim_param["Vecteurs_"]
    if not (sim_param["Vecteurs_"] == sim_param["Vecteurs"]):
        print("Attention, la liste des vecteurs de la table Vecteurs doit correspondre aux colonnes de la table contenant les consos unitaires")

    return sim_param

def get_function_list(sim_param):
    res=[]
    for key in sim_param:
        if len(key) > len("f_"):
            if key[0:len("f_")] == "f_":
                res += [key]
    return res


def fill_sim_param(sim_param,Para_2_fill):
    for key in Para_2_fill:
        if key in sim_param:
            if type(sim_param[key]) in [float,int,str]: sim_param[key] = pd.Series([sim_param[key]] * len(Para_2_fill[key]),index=Para_2_fill[key],name=key)
            else :
                initial_type = type(sim_param[key])
                sim_param[key] = pd.DataFrame(None, index=Para_2_fill[key]). \
                    merge(sim_param[key], how='outer', left_index=True, right_index=True)
                TMP = pd.DataFrame(None, index=Para_2_fill[key])
                new_cols = Para_2_fill[key].to_frame(index=False).columns.reindex(list(sim_param[key].index.names))
                re_indexed_multi_index = pd.MultiIndex.from_frame(Para_2_fill[key].to_frame(index=False).reindex(columns=new_cols[0]))
                for col in sim_param[key].columns:
                    TMP[col] = sim_param[key][col].loc[re_indexed_multi_index].to_numpy()
                sim_param[key] = TMP
                if initial_type == pd.Series:
                    ## complete NAs
                    sim_param[key]=sim_param[key].fillna(0)
                    sim_param[key]=sim_param[key].iloc[:,0]
    return sim_param
#key="retrofit_improvement"
def complete_parameters(sim_param,Para_2_fill={}):

    sim_param = interpolate_sim_param(sim_param)
    if len(Para_2_fill)>0:
        sim_param = fill_sim_param(sim_param,Para_2_fill = Para_2_fill)

    #particular processing for "retrofit_Transition" : kind of matrix transpose
    past_dim = sim_param["retrofit_Transition"].index.names
    Other_dims = list(map(lambda x: x.replace(sim_param["Energy_system_name"] , sim_param["Energy_system_name"] + "_out"), past_dim))
    sim_param["retrofit_Transition"] = sim_param["retrofit_Transition"].reset_index().assign(old_new="new").set_index(past_dim+  ["old_new"]). \
        melt(ignore_index=False, var_name=sim_param["Energy_system_name"]+"_out", value_name="retrofit_Transition").set_index(
        [sim_param["Energy_system_name"]+"_out"], append=True).\
        pivot_table(values='retrofit_Transition',index=Other_dims+  ["old_new"],columns=sim_param["Energy_system_name"]).\
        reset_index().rename(columns={sim_param["Energy_system_name"]+"_out": sim_param["Energy_system_name"] }).set_index(past_dim+  ["old_new"])
    return sim_param


#data_series_to_transition= sim_param["init_sim_stock"]["total_surface"]
def apply_transition(Surf_2_retrofit,Transition,sim_param):
    Name = Surf_2_retrofit.name
    if type(Name)==type(None): Name=0
    X_df=Surf_2_retrofit.copy().to_frame().pivot_table(values = Name,index=sim_param["Complementary_Index_names"],columns=sim_param["Energy_system_name"])
    res = Surf_2_retrofit.copy()*0
    #res = res.reset_index().assign(old_new="new").set_index(sim_param["Complementary_Index_names"]+[sim_param["Energy_system_name"]]+  ["old_new"])[Name]
    for  Energy_source in sim_param["Energy_source_index"]:
        if len(X_df.loc[:,Energy_source])==1:
            res += Transition.loc[:, Energy_source] * float(X_df.loc[:,Energy_source])  # implicite merge because Energy_source_index is not in X_df rows but is in transition and res
        else:
            res += Transition.loc[:, Energy_source] * X_df.loc[:,Energy_source]  # implicite merge because Energy_source_index is not in X_df rows but is in transition and res

    return res

def interpolate_sim_param(sim_param):
    #interpolations
    for key in sim_param.keys():
        if isinstance(sim_param[key], pd.Series):
            if sim_param[key].index.name == "year":
                sim_param[key] = sim_param[key].interpolate_along_one_index_column(new_x_values = sim_param["years"],x_columns_for_interpolation = "year")
            elif len(sim_param[key].index.names)>1:
                if ("year" in list(sim_param[key].index.to_frame().columns)):
                    sim_param[key] = sim_param[key].interpolate_along_one_index_column(new_x_values = sim_param["years"],x_columns_for_interpolation = "year")
        elif isinstance(sim_param[key], pd.DataFrame):
            if ("year" in list(sim_param[key].index.to_frame().columns)):
                sim_param[key] = sim_param[key].interpolate_along_one_index_column(new_x_values = sim_param["years"],x_columns_for_interpolation = "year")
    return sim_param

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
        df=pd.concat([df,to_append_df])
        # df=df.append(to_append_df)
        for col in df.columns:
            df.loc[cat_tuple(tuple(len(Index_set.columns) * [slice(None)]), new_x_values[-1]),col] = df.loc[cat_tuple(tuple(len(Index_set.columns) * [slice(None)]), float(x_columns_for_interpolation_input_values)),col]
    for key in df.columns:
        f_out = {}
        if len(Index_name) == 0:
            f_in = [df.loc[(year, key)] for year in x_columns_for_interpolation_input_values]
            res[key] = pd.DataFrame.from_dict(
                {key: np.interp(new_x_values, x_columns_for_interpolation_input_values, f_in),
                 x_columns_for_interpolation: new_x_values}).set_index(["year"])[key]
        elif len(Index_set.columns) == 1:
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
                        {key: np.interp(new_x_values, x_columns_for_interpolation_input_values, f_in),
                         x_columns_for_interpolation: new_x_values, **TMP_d})

        if not (len(Index_name)==0):
            res[key] = pd.concat(f_out, axis=0).set_index(Index_name + [x_columns_for_interpolation])[[key]]
    if isSeries: return res[key]
    else: return res
pd.DataFrame.interpolate_along_one_index_column = interpolate_along_one_index_column
pd.Series.interpolate_along_one_index_column = interpolate_along_one_index_column

def create_initial_parc(sim_param):
    """
    Create a data frame indexed by sim_param["base_index"] with columns obtained from all
    variables indexed by "init_..."
    :param sim_param:
    :return:
    """
    col_names = []
    res = pd.DataFrame(None, index=sim_param["base_index"])
    for key in sim_param.keys():
        if len(key)>5:
            if key[0: 5]=="init_":
                #print(key)
                if type(sim_param[key])==float:
                    res = res.merge(pd.DataFrame([sim_param[key]]*len(sim_param["base_index"]),
                                    index=sim_param["base_index"]) , how='outer', left_index=True, right_index=True)
                else:
                    res =  res.merge(sim_param[key],how = 'outer',left_index=True,right_index=True)
                col_names = col_names + [key[5:len(key)]]

    res.columns = col_names
    return res



def initialize_Simulation(sim_param):
    """
    Creates a dictionnary with sim_stock and fill it the initial year data.
    :param sim_param:
    :return:
    """
    sim_stock = {}
    year=int(sim_param["date_debut"])
    sim_stock[year]=pd.concat( [ sim_param["init_sim_stock"].assign(old_new="old"),
                                    sim_param["init_sim_stock"].assign(old_new="new")]).set_index(['old_new'], append=True).sort_index()
    sim_stock[year].loc[(*sim_param["base_index_tuple"],"new"),sim_param["volume_variable_name"] ]=0 ## on commence sans "neuf"

    return sim_stock

def non_valid_data(sim_param):
    is_there_a_problem = False
    Index_names_year =sim_param["Index_names"] + ["year"]

    keys =[sim_param["retrofit_change_variable_name"],sim_param["new_yearly_variable_name"],"retrofit_improvement"]
    for key in keys:
        if key in sim_param:
            My_index_name = list(sim_param[key].index.to_frame().columns)
            if not (My_index_name == Index_names_year):
                print("sim_param[\""+key+"\"] should have dimensions : "+ str(Index_names_year) +", \n but dimensions are : "+ str(My_index_name))
                is_there_a_problem = True

    return is_there_a_problem


def launch_simulation(sim_param):
    sim_param["energy_need_variable_name"]= "energy_need_per_"+sim_param["volume_variable_name"]
    sim_param["new_yearly_variable_name"]=  "new_yearly_"+sim_param["volume_variable_name"]
    sim_param["retrofit_change_variable_name"]="retrofit_change_"+sim_param["volume_variable_name"]
    if non_valid_data(sim_param):
        print("Error")
        sim_stock=0
    else:
        sim_stock = initialize_Simulation(sim_param)
        for year in progressbar(range(int(sim_param["date_debut"]) + 1, int(sim_param["date_fin"])), "Computing: ", 40):
            sim_stock[year] = sim_stock[year-1].copy()

            #destruction
            if "old_taux_disp" in sim_param:
                sim_stock[year].loc[:,sim_param["volume_variable_name"]] -= \
                    sim_param["old_taux_disp"][year] * sim_stock[year].loc[:, sim_param["volume_variable_name"]]

            #renovation
            base_index_old =(*sim_param["base_index_tuple"], "old")
            base_index_year_new =(*sim_param["base_index_tuple"], year, "new")
<<<<<<< HEAD
            Surf_2_retrofit = sim_param[sim_param["retrofit_change_variable_name"]].loc[(*sim_param["base_index_tuple"], year)]
            Surf_remain = sub_keep_positive(sim_stock[year].loc[base_index_old, sim_param["volume_variable_name"]],Surf_2_retrofit)
            Surf_2_retrofit = (sim_stock[year].loc[base_index_old,sim_param["volume_variable_name"]] - Surf_remain.rm_index("old_new"))
||||||| b982435
            Surf_2_retrofit = sim_param[sim_param["retrofit_change_variable_name"]].loc[(*sim_param["base_index_tuple"], year)]
            Surf_remain = sub_keep_positive(sim_stock[year].loc[base_index_old, sim_param["volume_variable_name"]],Surf_2_retrofit)
            Surf_2_retrofit = (sim_stock[year][sim_param["volume_variable_name"]].loc[base_index_old] - Surf_remain.rm_index("old_new"))
=======
            Unit_2_retrofit_TMP = sim_param[sim_param["retrofit_change_variable_name"]].loc[(*sim_param["base_index_tuple"], year)]
            Unit_remain = sub_keep_positive(sim_stock[year].loc[base_index_old, sim_param["volume_variable_name"]],Unit_2_retrofit_TMP)
            Unit_2_retrofit = (sim_stock[year][sim_param["volume_variable_name"]].loc[base_index_old] - Unit_remain.rm_index("old_new"))
            if ((Unit_2_retrofit_TMP-Unit_2_retrofit).sum()>0.005*Unit_2_retrofit_TMP.sum()):
                print("warning, too much retrofit, excess of "+str((Unit_2_retrofit_TMP-Unit_2_retrofit).sum()/Unit_2_retrofit_TMP.sum()*100)+" %")

>>>>>>> e71dcc74ce51e8dbe805bb5c6e4531a46223f93c
            Transition = sim_param["retrofit_Transition"].loc[base_index_year_new, :].rm_index("year").rm_index("old_new")
            New_units = apply_transition(Unit_2_retrofit,Transition,sim_param)
            if ((Unit_2_retrofit.sum()-New_units.sum())>0.005*Unit_2_retrofit_TMP.sum()):
                print("warning, Transition does not sum to one")
            New_energy_need = (1 - sim_param["retrofit_improvement"].loc[(*sim_param["base_index_tuple"], year)]) * \
                             sim_stock[year - 1][sim_param["energy_need_variable_name"]].loc[
                                 (*sim_param["base_index_tuple"], "old")]
            sim_stock = update_heat_need(sim_stock=sim_stock,year=year,
                      New_units=New_units,
                      New_energy_need=New_energy_need,
                      sim_param=sim_param)

            sim_stock[year].loc[(*sim_param["base_index_tuple"], "old"), sim_param["volume_variable_name"]] = Unit_remain

            #neuf
            if sim_param["new_yearly_variable_name"] in sim_param:
                sim_stock = update_heat_need(sim_stock=sim_stock,year=year,
                    New_units=sim_param[sim_param["new_yearly_variable_name"]].loc[(*sim_param["base_index_tuple"], year)],
                    New_energy_need=sim_param["new_energy"].loc[(*sim_param["base_index_tuple"], year)],
                    sim_param=sim_param)

            Functions_ = get_function_list(sim_param)
            for func in Functions_:
                for key in sim_param[func]:
                    args = inspect.getfullargspec(sim_param[func][key]).args
                    if args==['x']:
                        sim_stock[year].loc[:, key] = sim_stock[year].apply(
                            lambda x: sim_param[func][key](x), axis=1).fillna(0)
                    elif args==['x','sim_param']:
                        sim_stock[year].loc[:, key] = sim_stock[year].apply(
                            lambda x: sim_param[func][key](x, sim_param), axis=1).fillna(0)
                    elif args==['x','sim_param','year']:
                        sim_stock[year].loc[:, key] = sim_stock[year].apply(
                            lambda x: sim_param[func][key](x, sim_param, year), axis=1).fillna(0)
                    else: print("Warnings, function func defined with arguments "+str(args)+" only x | x,sim_param | x,sim_param,year implemented")


    return sim_stock
#lorsque l'on met à jour l'ensemble des surfaces et le besoin associé

def get_index_name(xx):
    if type(xx) == pd.Series:
        res = xx.name
        if type(res)==type(None): res = 0
    elif type(xx) == pd.MultiIndex:
        res = xx.names
    else : res = xx.names
    return res

def update_heat_need(sim_stock,year,New_units,New_energy_need,sim_param):

    #mise à jour des surfaces
    old_col_names=New_units.name
    if type(old_col_names) == type(None): old_col_names = 0
    Nouvelles_unite_neuf = New_units.loc[sim_param["base_index_tuple"]]
    My_index_names = get_index_name(sim_param["base_index"])
    if 'Efficiency_class' in My_index_names:
        Nouvelles_unite_neuf=Nouvelles_unite_neuf.update_Efficiency_class(New_energy_need,sim_param)
        New_energy_need=New_energy_need.update_Efficiency_class(New_energy_need,sim_param)
    #mise à jouro du besoin
    Old_energy_need_new = sim_stock[year][sim_param["energy_need_variable_name"]].loc[(*sim_param["base_index_tuple"], "new")]
    Anciennes_unite= sim_stock[year][sim_param["volume_variable_name"]].loc[(*sim_param["base_index_tuple"], "new")]
    New_energy_need_new = sim_stock[year][sim_param["energy_need_variable_name"]].loc[(*sim_param["base_index_tuple"], "new")].copy()
    sub_index = Nouvelles_unite_neuf > 0
    New_energy_need_new.loc[sub_index] = (Old_energy_need_new.loc[sub_index] * Anciennes_unite.loc[sub_index] + New_energy_need.loc[sub_index] *Nouvelles_unite_neuf.loc[sub_index]) / \
                                                                (Anciennes_unite.loc[sub_index]+Nouvelles_unite_neuf.loc[sub_index])
    New_energy_need_new=New_energy_need_new.to_frame().assign(old_new="new").set_index(["old_new"],append=True)[sim_param["energy_need_variable_name"]]

    sim_stock[year].loc[(*sim_param["base_index_tuple"], "new"), sim_param["volume_variable_name"]] +=Nouvelles_unite_neuf.to_frame().assign(old_new="new").set_index(["old_new"],append=True)[old_col_names]
    sim_stock[year].loc[(*sim_param["base_index_tuple"], "new"),sim_param["energy_need_variable_name"]] =New_energy_need_new

    return sim_stock

def update_Efficiency_class(df,New_energy_need,sim_param,Efficiency_class_name = "Efficiency_class"):
    Name = df.name
    if type(Name)==type(None): Name = 0
    Index_names = df.index.names
    TMP=pd.DataFrame(None,index = New_energy_need.index)
    TMP[Efficiency_class_name] = New_energy_need.apply(lambda x: sim_param["_f_energy_to_class"](x,sim_param["energy_class_dictionnary" ]))
    df_TMP = df.copy().reset_index()
    df_TMP[Efficiency_class_name] = TMP[Efficiency_class_name].to_numpy()
    df_TMP = df_TMP.groupby(Index_names).sum()
    df_2 = df.copy()
    df_2[df_TMP.index] = df_TMP[Name]
    return df_2
pd.Series.update_Efficiency_class = update_Efficiency_class


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

def Create_0D_df(variable_dict,sim_param):
    res = {}
    for key in variable_dict.keys():
        if key in sim_param.keys():
            ds = sim_param[key]
        else :
            if variable_dict[key]=="wmean":
                ds = sim_param["init_sim_stock"][[key,sim_param["volume_variable_name"] ]]
            else:
                ds = sim_param["init_sim_stock"][key]
        if len(variable_dict[key])==0:
            res[key]= ds
        else:
            exec("res[key] = ds."+variable_dict[key]+"()")
    return pd.DataFrame.from_dict(res,orient="index")[0]

def Create_XD_df(variable_dict,sim_param,group_along = ["Energy_source"]):
    res = {}
    weightedMean_weight = None
    for key in variable_dict.keys():
        if key in sim_param.keys():
            ds = sim_param[key]
        else :
            if variable_dict[key]=="wmean":
                weightedMean_weight=[sim_param["volume_variable_name"] ]
                ds = sim_param["init_sim_stock"][[key,sim_param["volume_variable_name"] ]]
            else:
                weightedMean_weight =None
                ds = sim_param["init_sim_stock"][key]
        res[key]=ds.to_frame().reset_index().groupbyAndAgg(group_along=group_along,
                                             aggregation_dic={key : variable_dict[key]},
                                             weightedMean_weight=weightedMean_weight).\
            set_index(group_along)[key]

    return pd.concat(res,axis=1)



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
# tmp=sim_stock[year].loc[base_index_old, sim_param["volume_variable_name"]]-Surf_2_retrofit
# tmp2=sim_stock[year].loc[base_index_old,sim_param["volume_variable_name"]]-tmp
# Surf_2_retrofit-tmp2==0