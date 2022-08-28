
from pyomo.environ import *
from pyomo.core import *
from pyomo.opt import SolverFactory
from datetime import timedelta
import pandas as pd
import numpy as np
import re
import sys


#TODO code a version of expand_grid that takes a dictionnary as input. Or even data frames
def expand_grid_3D(x, y , z,names):
    res=pd.DataFrame()
    xG, yG, zG = np.meshgrid(x, y, z) # create the actual grid
    res.loc[:,names[0]] = xG.flatten() # make the grid 1d
    res.loc[:,names[1]] = yG.flatten() # same
    res.loc[:,names[2]] = zG.flatten() # same
    return res # return a dataframe

def expand_DF_2D(df1,df2,names):
    res=pd.DataFrame()
    xG, yG = np.meshgrid(df1.index, df2.index) # create the actual grid
    res.loc[:,names[0]] = xG.flatten() # make the grid 1d
    res.loc[:,names[1]] = yG.flatten() # same
    res=res.set_index(names)

    My_indexes = []
    for key, value in My_dict.items():
        My_indexes.append(value.index)
    G = np.meshgrid(*My_indexes) # create the actual grid
    for key in range(0,len(My_indexes)):
        res.loc[:,names[key]] = G[key].flatten() # make the grid 1d

    res=res.set_index(names)

    i=0
    for key, value in My_dict.items():
        for j in range(0,len(value.index)):
            My_key = insert_in_slice_None(i,len(My_dict),value.index[j])
            res.loc[My_key, value.columns] = value.loc[value.index[j],value.columns]
        i=i+1

    return res # return a dataframe

### not finished

def expand_grid_3D(x, y , z,names):
    res=pd.DataFrame()
    xG, yG, zG = np.meshgrid(x, y, z) # create the actual grid
    res.loc[:,names[0]] = xG.flatten() # make the grid 1d
    res.loc[:,names[1]] = yG.flatten() # same
    res.loc[:,names[2]] = zG.flatten() # same
    return res # return a dataframe

def expand_grid_from_dict(My_dict,as_MultiIndex=False):
    names= list(My_dict.keys())
    res=pd.DataFrame()
    G = np.meshgrid(*list(My_dict.values())) # create the actual grid
    for key in range(0,len(names)):
        res.loc[:,names[key]] = G[key].flatten() # make the grid 1d
    if as_MultiIndex:res=pd.MultiIndex.from_frame(res)
    return res # return a dataframe or a multiindex


def expand_grid_fromc_dict_2(My_dict,names ):
    res=pd.DataFrame()
    My_indexes = []
    for key, value in My_dict.items():
        My_indexes.append(value.index)
    G = np.meshgrid(*My_indexes) # create the actual grid
    for key in range(0,len(My_indexes)):
        res.loc[:,names[key]] = G[key].flatten() # make the grid 1d

    res=res.set_index(names)

    i=0
    for key, value in My_dict.items():
        for j in range(0,len(value.index)):
            My_key = insert_in_slice_None(i,len(My_dict),value.index[j])
            res.loc[My_key, value.columns] = value.loc[value.index[j],value.columns]
        i=i+1

    return res # return a dataframe

def insert_slice_None(position,to_insert_tuple):
    tuple_l = list(to_insert_tuple)
    tuple_l.insert(position, 'slice(None)')
    return tuple(tuple_l)

def tuple_insert(tup,pos,ele):
    tup = tup[:pos]+(ele,)+tup[pos:]
    return tup



def insert_in_slice_None(position,length,to_insert_tuple):
    My_tuple_list = (length-1)*['slice(None)']
    return tuple_insert((*My_tuple_list,),position,to_insert_tuple)


def change_index_values(df,index_name,change_dict):
    old_index_names = df.index.names
    if isinstance(df, pd.Series):
        old_col_names = df.name
        if type(old_col_names)==type(None): old_col_names =0
        df = df.reset_index()
        df.loc[:,index_name] = df[index_name].replace(change_dict)
        return df.set_index(old_index_names)[old_col_names]
    else:
        old_col_names = df.columns
        df = df.reset_index()
        df.loc[:,index_name] = df[index_name].replace(change_dict)
        return df.set_index(old_index_names)[old_col_names]
pd.DataFrame.change_index_values = change_index_values
pd.Series.change_index_values = change_index_values


def remove_index_from_name(df,index_name):
    old_index_names = list(df.index.names)
    old_index_names.remove(index_name)
    if isinstance(df, pd.Series):
        old_col_names = df.name
        if type(old_col_names)==type(None): old_col_names =0
        df = df.reset_index().drop(columns=[index_name])
        return df.set_index(old_index_names)[old_col_names]
    else:
        old_col_names = list(df.columns)
        df = df.reset_index().drop(columns=[index_name])
        return df.set_index(old_index_names)[old_col_names]
pd.DataFrame.remove_index_from_name = remove_index_from_name
pd.Series.remove_index_from_name = remove_index_from_name


import sys
def progressbar(it, prefix="", size=60, out=sys.stdout): # Python3.3+
    count = len(it)
    def show(j):
        x = int(size*j/count)
        print("{}[{}{}] {}/{}".format(prefix, "#"*x, "."*(size-x), j, count),
                end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)

def groupbyAndAgg(self,group_along,aggregation_dic,weightedMean_weight=None):
    '''
    :self pandas dataframe
    :group_along list of column name to group by along
    :aggregation_dic aggregation dictionary with the possibility to use a weighted mean 'wmean'
    :weightedMean_weight a column name to do the weighted mean(s) along
    Split data set between duplicated and non duplicated to avoid memory issues
    '''
    df_loc=self.copy()
    Duplicated_rows = df_loc.duplicated(subset=group_along,keep=False)
    df_nodup = df_loc[~Duplicated_rows][group_along+list(aggregation_dic)]

    ### calcul des prod et modifiction du dictionnaire des aggregations pour faire sommeprod
    values = [elem for elem in aggregation_dic.keys() if aggregation_dic[elem] == 'wmean']
    new_aggregation_dic={key:aggregation_dic[key] for key in aggregation_dic.keys() if key not in values}

    for value_col in values:
        prod_name = 'prod_{v}_{w}'.format(v=value_col, w=weightedMean_weight)
        weights_name = 'weights_{v}_{w}'.format(v=value_col, w=weightedMean_weight)
        df_loc[prod_name] = df_loc[value_col] * df_loc[weightedMean_weight]
        df_loc[weights_name] = df_loc[weightedMean_weight].where(~df_loc[prod_name].isnull())
        new_aggregation_dic[prod_name] = 'sum'
        new_aggregation_dic[weights_name] = 'sum'

    output = df_loc[Duplicated_rows]. \
        groupby(group_along, observed=True).aggregate(new_aggregation_dic, engine='cython'). \
        reset_index()


    for value_col in values:
        prod_name = 'prod_{v}_{w}'.format(v=value_col, w=weightedMean_weight)
        weights_name = 'weights_{v}_{w}'.format(v=value_col,w=weightedMean_weight)
        output[value_col] = output[prod_name] / output[weights_name]
        output.drop(columns=[prod_name,weights_name],inplace=True)

    output = output.append(df_nodup)
    del df_nodup,df_loc
    return output

pd.DataFrame.groupbyAndAgg = groupbyAndAgg


def soustrait_mais_reste_positif(df_1, df_2):
    TMP_df = pd.DataFrame(None);
    TMP_df["df_1_moins_df_2"] = df_1 - df_2;
    TMP_df["0"] = 0
    return TMP_df.max(axis=1)

def allin(vec,dest):
    return(all([name in dest for name in vec]))

def MySolverFactory(solver,solverpath=1):
    if (solverpath==1):
        opt = SolverFactory(solver)
    else :
        if (solver in solverpath):
            opt = SolverFactory(solver, executable=solverpath[solver])
        else :
            opt = SolverFactory(solver)
    return(opt)


def get_ParametersNames(model):
    """
    This function returns a list of parameters names
    :param model: pyomo model
    :return: a set (not a pyomo set) with names
    """
    res=set([])
    for v in model.component_objects(Param, active=True):
        res=res.union([str(v)])
    return res;


def get_allSetsnames(model):
    """
    This function finds all SimpleSets and returns a set with pyomo Sets and associated values
    :param model: pyomo model
    :return: a Set (not a pyomo set) with names
    """

    res=set([])
    for v in model.component_objects(Set, active=True):
        res=res.union([str(v)])
    return res;

def get_allSets(model):
    """
    This function finds all SimpleSets and returns a set with pyomo Sets and associated values
    :param model: pyomo model
    :return: a Set (not a pyomo set) with names
    """
    res={}
    for v in model.component_objects(Set, active=True):
        setobject = getattr(model, str(v))
        res[str(v)]=setobject.data()
    return res;

def get_SimpleSets(model):
    """
    This function finds all SimpleSets and returns a set with pyomo Sets and associated values
    :param model: pyomo model
    :return: a Set (not a pyomo set) with names
    """
    res={}
    for v in model.component_objects(Set, active=True):
        if not str(v) == "'AREAS.1'":
            setobject = getattr(model, str(v))
            if not (isinstance(setobject,pyomo.core.base.set.SetProduct)):
                res[str(v)]=setobject.data()
    return res;
#setobject=varobject.index_set()
def getSetNames(model,setobject):
    """
    This function finds the sets associated to a flat set product object
    :param model: pyomo model
    :param setobject: a pyomo set object
    :return: a Set (not a pyomo set) with names
    """
    SimpleSets=get_SimpleSets(model)
    #if not isinstance(setobject,pyomo.core.base.set.SetProduct):
    #    print("warning setobject should be a SetProduct") ### not really actually
    cpt=0;
    res={}
    for subset in setobject.subsets():
        for i in SimpleSets:
            if SimpleSets[i]==subset.data():
                res[cpt]=i
        cpt+=1;
    return res;

def getSetNamesList(model,setobject):
    """
    This function finds the sets associated to a flat set product object
    :param model: pyomo model
    :param setobject: a pyomo set object
    :return: a Set (not a pyomo set) with names
    """
    SimpleSets=get_SimpleSets(model)
    #if not isinstance(setobject,pyomo.core.base.set.SetProduct):
    #    print("warning setobject should be a SetProduct") ### not really actually
    cpt=0;
    res=[]
    for subset in setobject.subsets():
        for i in SimpleSets:
            if SimpleSets[i]==subset.data():
                res.append(i)
        cpt+=1;
    return res;



def getParameters_panda(model):
    """
    This function takes Parameters and return values in panda form
    :param model: pyomo model
    :return: a Set (not a pyomo set) with names of variables and associated values in panda table
    """
    import pandas as pd
    Parameters = {}
    for v in model.component_objects(Param, active=True):
        # print ("Variables",v)
        varobject = getattr(model, str(v))
        VAL = pd.DataFrame.from_dict(varobject.extract_values(), orient='index', columns=[str(varobject)])
        #print(get_set_name(model,varobject))
        if isinstance(varobject.index_set(),pyomo.core.base.set.SetProduct):
            #Variables[str(v)].rename_axis(index=[str(varobject.index_set())])
            DIM = pd.DataFrame(VAL.index.tolist()).rename(columns=getSetNames(model,varobject.index_set())).reset_index(drop=True)
            VAL.reset_index(drop=True, inplace=True)
            Parameters[str(v)]= pd.concat([DIM,VAL],axis=1,sort=False)
        else:
            DIM= pd.DataFrame(VAL.index.tolist()).rename(columns=getSetNames(model,varobject.index_set())).reset_index(drop=True)
            VAL.reset_index(drop=True, inplace=True)
            Parameters[str(v)]= pd.concat([DIM,VAL],axis=1,sort=False)
            Parameters[str(v)]=Parameters[str(v)].rename(columns={0:str(varobject.index_set())})
    return Parameters;



def get_ParametersNameWithSet(model):
    """
    This function takes Parameters and return values in panda form
    :param model: pyomo model
    :return: a Set (not a pyomo set) with names of variables and associated values in panda table
    """
    import pandas as pd
    Parameters = {}
    for v in model.component_objects(Param, active=True):
        # print ("Variables",v)
        varobject = getattr(model, str(v))
        Parameters[str(v)] = getSetNamesList(model, varobject.index_set())
    return Parameters;

def get_VariableNameWithSet(model):
    """
    This function takes Parameters and return values in panda form
    :param model: pyomo model
    :return: a Set (not a pyomo set) with names of variables and associated values in panda table
    """
    import pandas as pd
    Parameters = {}
    for v in model.component_objects(Var, active=True):
        # print ("Variables",v)
        varobject = getattr(model, str(v))
        Parameters[str(v)] = getSetNamesList(model, varobject.index_set())
    return Parameters;



def getVariables_panda(model):
    """
    This function takes variables and return values in panda form
    :param model: pyomo model
    :return: a Set (not a pyomo set) with names of variables and associated values in panda table
    """
    import pandas as pd
    Variables = {}
    for v in model.component_objects(Var, active=True):
        # print ("Variables",v)
        varobject = getattr(model, str(v))
        VAL = pd.DataFrame.from_dict(varobject.extract_values(), orient='index', columns=[str(varobject)])
        #print(get_set_name(model,varobject))
        if isinstance(varobject.index_set(),pyomo.core.base.set.SetProduct):
            #Variables[str(v)].rename_axis(index=[str(varobject.index_set())])
            DIM = pd.DataFrame(VAL.index.tolist()).rename(columns=getSetNames(model,varobject.index_set())).reset_index(drop=True)
            VAL.reset_index(drop=True, inplace=True)
            Variables[str(v)]= pd.concat([DIM,VAL],axis=1,sort=False)
        else:
            DIM= pd.DataFrame(VAL.index.tolist()).rename(columns=getSetNames(model,varobject.index_set())).reset_index(drop=True)
            VAL.reset_index(drop=True, inplace=True)
            Variables[str(v)]= pd.concat([DIM,VAL],axis=1,sort=False)
            Variables[str(v)]=Variables[str(v)].rename(columns={0:str(varobject.index_set())})
    return Variables;


def getVariables_panda_indexed(model,Belfort=False):
    """
    This function takes variables and return values in panda form
    :param model: pyomo model
    :return: a Set (not a pyomo set) with names of variables and associated values in panda table
    """
    import pandas as pd
    Variables = {}
    for v in model.component_objects(Var, active=True):
        # print ("Variables",v)
        varobject = getattr(model, str(v))
        VAL = pd.DataFrame.from_dict(varobject.extract_values(), orient='index', columns=[str(varobject)])
        #print(get_set_name(model,varobject))
        if isinstance(varobject.index_set(),pyomo.core.base.set.SetProduct):
            #Variables[str(v)].rename_axis(index=[str(varobject.index_set())])
            DIM = pd.DataFrame(VAL.index.tolist()).rename(columns=getSetNames(model,varobject.index_set())).reset_index(drop=True)
            VAL.reset_index(drop=True, inplace=True)
            Variables[str(v)]= pd.concat([DIM,VAL],axis=1,sort=False)
            if not Belfort:
                if ('exchange' in Variables[str(v)].columns)  :
                    Variables[str(v)].columns=['AREAS', 'AREAS1', 'TIMESTAMP', 'exchange']
                    Variables[str(v)].set_index(['AREAS', 'AREAS1', 'TIMESTAMP'])
                else:
                    Variables[str(v)].set_index(DIM.columns.tolist())
        else:
            DIM= pd.DataFrame(VAL.index.tolist()).rename(columns=getSetNames(model,varobject.index_set())).reset_index(drop=True)
            VAL.reset_index(drop=True, inplace=True)
            Variables[str(v)]= pd.concat([DIM,VAL],axis=1,sort=False)
            Variables[str(v)]=Variables[str(v)].rename(columns={0:str(varobject.index_set())})
            if not Belfort:
                if ('exchange' in Variables[str(v)].columns):
                    Variables[str(v)].columns = ['AREAS', 'AREAS1', 'TIMESTAMP', 'exchange']
                    Variables[str(v)].set_index(['AREAS', 'AREAS1', 'TIMESTAMP'])
                else:
                    Variables[str(v)].set_index(DIM.columns.tolist())
    return Variables;


def getConstraintsDual_panda_indexed(model):
    """
    This function takes dual values associated to Constraints and return values in panda form
    :param model: pyomo model
    :return: a Set (not a pyomo set) with names of Constraints and associated dual values in panda table
    """
    import pandas as pd
    Constraints = {}
    for v in model.component_objects(Constraint, active=True):
        # print ("Constraints",v)
        cobject = getattr(model, str(v))
        VAL = pd.DataFrame.from_dict(get_dualValues(model,cobject), orient='index', columns=[str(cobject)])
        #print(get_set_name(model,varobject))
        if isinstance(cobject.index_set(),pyomo.core.base.set.SetProduct):
            #Constraints[str(v)].rename_axis(index=[str(varobject.index_set())])
            DIM = pd.DataFrame(VAL.index.tolist()).rename(columns=getSetNames(model,cobject.index_set())).reset_index(drop=True)
            VAL.reset_index(drop=True, inplace=True)
            Constraints[str(v)]= pd.concat([DIM,VAL],axis=1,sort=False)
            #if (set(["exchangeCtrMoins", "exchangeCtrPlus", "exchangeCtr2"]).intersection(Constraints[str(v)].columns)):
                #AAAAA = 1;
                # TODO set the right columns here
                # Variables[str(v)].columns=['AREAS', 'AREAS1', 'TIMESTAMP', 'exchange']
                # Variables[str(v)].set_index(['AREAS', 'AREAS1', 'TIMESTAMP'])
            #else:
            Constraints[str(v)].set_index(DIM.columns.tolist())
        else:
            DIM= pd.DataFrame(VAL.index.tolist()).rename(columns=getSetNames(model,cobject.index_set())).reset_index(drop=True)
            VAL.reset_index(drop=True, inplace=True)
            Constraints[str(v)]= pd.concat([DIM,VAL],axis=1,sort=False)
            Constraints[str(v)]=Constraints[str(v)].rename(columns={0:str(cobject.index_set())})
            #if (set(["exchangeCtrMoins", "exchangeCtrPlus", "exchangeCtr2"]).intersection(Constraints[str(v)].columns)):
                #AAAAA = 1;
                # TODO set the right columns here
                # Variables[str(v)].columns=['AREAS', 'AREAS1', 'TIMESTAMP', 'exchange']
                # Variables[str(v)].set_index(['AREAS', 'AREAS1', 'TIMESTAMP'])
            #else:
            Constraints[str(v)].set_index(DIM.columns.tolist())
    return Constraints;


def getConstraintsDual_panda(model):
    """
    This function takes dual values associated to Constraints and return values in panda form
    :param model: pyomo model
    :return: a Set (not a pyomo set) with names of Constraints and associated dual values in panda table
    """
    import pandas as pd
    Constraints = {}
    for v in model.component_objects(Constraint, active=True):
        # print ("Constraints",v)
        cobject = getattr(model, str(v))
        VAL = pd.DataFrame.from_dict(get_dualValues(model,cobject), orient='index', columns=[str(cobject)])
        #print(get_set_name(model,varobject))
        if isinstance(cobject.index_set(),pyomo.core.base.set.SetProduct):
            #Constraints[str(v)].rename_axis(index=[str(varobject.index_set())])
            DIM = pd.DataFrame(VAL.index.tolist()).rename(columns=getSetNames(model,cobject.index_set())).reset_index(drop=True)
            VAL.reset_index(drop=True, inplace=True)
            Constraints[str(v)]= pd.concat([DIM,VAL],axis=1,sort=False)
        else:
            DIM= pd.DataFrame(VAL.index.tolist()).reset_index(drop=True)
            VAL.reset_index(drop=True, inplace=True)
            Constraints[str(v)]= pd.concat([DIM,VAL],axis=1,sort=False)
            Constraints[str(v)]=Constraints[str(v)].rename(columns={0:str(cobject.index_set())})
    return Constraints;


def get_dualValues(model,cobject):
    """
    This function takes variables and return values in panda form
    :param model: pyomo model
    :return: a Set (not a pyomo set) with names of variables and associated values in panda table
    """
    res={};
    for index in cobject:
         res[index] = model.dual[cobject[index]]
    return res;


def math_to_pyomo_constraint(EQs,model,verbose =False):
    Set_names = get_allSetsnames(model)
    Parameters_names = get_ParametersNameWithSet(model)
    Variables_names = get_VariableNameWithSet(model)

    for curConstraintName in EQs.keys():
        EQ = EQs[curConstraintName];
        EQ["name"]= curConstraintName
        for param in Parameters_names:
            SET_val_names=[name+"_val" for name in Parameters_names[param]]
            EQ["equation"]=EQ["equation"].replace(param,"model."+param+"["+",".join(SET_val_names)+"]")

        for variable in Variables_names:
            SET_val_names=[name+"_val" for name in Variables_names[variable]]
            EQ["equation"]=EQ["equation"].replace(variable,"model."+variable+"["+",".join(SET_val_names)+"]")

        for SET in Set_names:
            SET_val_names=[name+"_val" for name in Variables_names[variable]]
            EQ["equation"]=EQ["equation"].replace("|"+SET," for "+SET+"_val"+" in model."+SET)

        regexp = re.compile("sum\([^\)]* for AREAS_val in model.AREAS\)")

        searched = regexp.search(EQ["equation"])
        if bool(searched):
            replacement = regexp.search(EQ["equation"])[0].replace("AREAS_val,AREAS_val","b,AREAS_val")
            replacement= replacement.replace("for AREAS_val in model.AREAS","for b in model.AREAS")
            EQ["equation"] = regexp.sub(replacement,EQ["equation"])

        #EQ["equation"]

        domain_name = [name+"_val" for name in EQ["domain"]]
        Constraint_function_definition = "def "+EQ["name"]+"_rule(model,"+",".join(domain_name)+"):\n"+"\t return "+EQ["equation"]
        if verbose: print(Constraint_function_definition)
        exec(Constraint_function_definition)
        domain_model = ["model."+name for name in EQ["domain"]]
        Constraint_assignation = "model."+EQ["name"]+"=Constraint("+",".join(domain_model)+",rule="+EQ["name"]+"_rule)"
        if verbose: print(Constraint_assignation)
        exec(Constraint_assignation)
    return model


def math_to_pyomo_Vardef(Vars, model, verbose=False):
    Set_names = get_allSetsnames(model)
    Parameters_names = get_ParametersNameWithSet(model)
    Variables_names = get_VariableNameWithSet(model)
    Date_list = pd.DataFrame({"Date" : getattr(model, "Date").data()})
    Date_list["week"] = Date_list.Date.apply(lambda x: x.isocalendar().week)

    for curVarName in Vars:

        Var_definition_script_splitted = re.compile("[\[\]]").split(curVarName)
        for SET in Set_names:
            Var_definition_script_splitted[1] = Var_definition_script_splitted[1].replace(SET, "model." + SET)
        Domain_text = Var_definition_script_splitted[2]
        match Domain_text:
            case "":
                Var_Domain = ""
            case ">=0":
                Var_Domain = ",domain=NonNegativeReals"
            case "<=0":
                Var_Domain = ",domain=NonPositiveReals"
            case [*Domain_text] if bool(re.compile(Domain_text).search("(.+)\-(.+)")):
                domain_split = re.compile(Domain_text).split("\-")
                Var_Domain = ",bounds=(" + domain_split[0] + "," + domain_split[1] + ")"
            
        Var_definition_script = "model." + Var_definition_script_splitted[0] + "=Var(" + Var_definition_script_splitted[
            1] + Var_Domain + ")"
        if verbose: print(Var_definition_script)
        exec(Var_definition_script)

    return model