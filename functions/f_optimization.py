
from pyomo.environ import *
from pyomo.core import *
from pyomo.opt import SolverFactory


def MySolverFactory(solver,solverpath=1):
    if (solverpath==1):
        opt = SolverFactory(solver)
    else :
        if (solver in solverpath):
            opt = SolverFactory(solver, executable=solverpath[solver])
        else :
            opt = SolverFactory(solver)
    return(opt)

def get_SimpleSets(model):
    """
    This function finds all SimpleSets and returns a set with pyomo Sets and associated values
    :param model: pyomo model
    :return: a Set (not a pyomo set) with names
    """
    res={}
    for v in model.component_objects(Set, active=True):
        setobject = getattr(model, str(v))
        if not (isinstance(setobject,pyomo.core.base.set.SetProduct)):
            res[str(v)]=setobject.data()
    return res;

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


def getVariables_panda_indexed(model):
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

