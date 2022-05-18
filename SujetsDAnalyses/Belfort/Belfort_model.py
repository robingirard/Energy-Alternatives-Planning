import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import pickle
if os.path.basename(os.getcwd()) == "Belfort":
    os.chdir('..')  ## to work at project root  like in any IDE
    os.chdir('..')

Input_folder="Data/input/"
import sys

#from functions.f_graphicalTools import *
#from functions.f_planingModels import *
#from functions.f_optimization import *
#from functions.f_graphicalTools import *
#from functions.f_consumptionModels import *
import numpy as np
from functions.f_model_Belfort import *
from pyomo.core import *
from pyomo.opt import SolverFactory
from time import time


def labour_ratio_cost(df):  # higher labour costs at night
    if df.hour in range(7, 17):
        return 1
    elif df.hour in range(17, 23):
        return 1.5
    else:
        return 2

def run_model(year,bati_hyp='ref',reindus=True,nuc='plus'):
    t1=time()
    print("Loading parameters")
    if year==2030:
        tech_suffix='2030'
    elif year in [2040,2050,2060] and nuc == 'plus':
        tech_suffix=str(year)+'_nuclear_plus'
    elif year in [2040, 2050, 2060] and nuc == 'minus':
        tech_suffix = str(year) + '_nuclear_minus'
    else:
        raise ValueError("Variable year should be 2030, 2040, 2050 or 2060 and variable nuc should be 'plus' or 'minus'.")

    TechParameters=pd.read_csv(Input_folder+"Prod_model/Technos_"+tech_suffix+".csv",decimal=',',sep=';').set_index(["TECHNOLOGIES"])
    StorageParameters=pd.read_csv(Input_folder+"Prod_model/Stock_technos_"+str(year)+".csv",decimal=',',sep=';').set_index(["STOCK_TECHNO"])

    ## Availability
    if year==2030:
        tech_suffix+='_nuclear_'+nuc
    NucAvailability=pd.read_csv(Input_folder+"Prod_model/availability_nuclear_"+tech_suffix+".csv",decimal='.',sep=';',parse_dates=['Date']).set_index('Date')

    WindSolarAvailability=pd.read_csv(Input_folder+"Prod_model/availabilityWindSolar.csv",decimal=',',sep=';',parse_dates=['Date']).set_index('Date')
    availabilityFactor=NucAvailability.join(WindSolarAvailability)
    HydroRiverAvailability=pd.read_csv(Input_folder+"Prod_model/availability_hydro_river.csv",decimal='.',sep=';',parse_dates=['Date']).set_index('Date')
    availabilityFactor = availabilityFactor.join(HydroRiverAvailability)

    availabilityFactor=availabilityFactor.rename(columns={'New nuke':'NewNuke','Old nuke':'OldNuke','availability':'HydroRiver'})
    availabilityFactor['NewHydroRiver']=availabilityFactor['HydroRiver']

    for col in TechParameters.index:
        if col not in availabilityFactor.columns:
            availabilityFactor[col]=1

    availabilityFactor=availabilityFactor.reset_index().melt(id_vars=['Date'],var_name="TECHNOLOGIES",value_name='availabilityFactor').set_index(['Date','TECHNOLOGIES'])

    ## Consumption
    d_reindus={True:'reindus',False:'no_reindus'}
    conso_suffix=str(year)+'_'+d_reindus[reindus]+'_'+bati_hyp
    try:
        areaConsumption=pd.read_csv(Input_folder+"Conso_model/Loads/Conso_"+conso_suffix+".csv",decimal='.',sep=';',parse_dates=['Date']).set_index('Date')
    except:
        raise ValueError("Variable bati_hyp should be 'ref' or 'SNBC'.")

    lossesRate=areaConsumption[['Taux_pertes']]

    ## Flexibility
    to_flex_consumption=areaConsumption[['Metallurgie','Conso_VE','Conso_H2']].rename(columns={'Metallurgie':'Steel','Conso_VE':'EV','Conso_H2':'H2'})
    to_flex_consumption=to_flex_consumption.reset_index().melt(id_vars=['Date'],var_name="FLEX_CONSUM",value_name='Consumption').set_index(['Date','FLEX_CONSUM'])

    areaConsumption=areaConsumption[['Consommation hors metallurgie']].rename(columns={'Consommation hors metallurgie':'areaConsumption'})

    FlexParameters=pd.read_csv(Input_folder+"Conso_model/Flex/Flex_"+str(year)+".csv",decimal=',',sep=';').set_index('FLEX_CONSUM')
    #print(FlexParameters)

    ## Labor ratio
    labour_ratio = pd.DataFrame()
    labour_ratio["Date"] = areaConsumption.index.get_level_values('Date')
    labour_ratio["FLEX_CONSUM"] = "Steel"
    labour_ratio["labour_ratio"] = labour_ratio["Date"].apply(labour_ratio_cost)

    for flex_consum in ["EV", "H2"]:
        u = pd.DataFrame()
        u["Date"] = areaConsumption.index.get_level_values('Date')
        u["FLEX_CONSUM"] = flex_consum
        u["labour_ratio"] = np.array(len(u["Date"]) * [1])
        labour_ratio = pd.concat([labour_ratio, u], ignore_index=True)

    labour_ratio=labour_ratio.set_index(["Date", "FLEX_CONSUM"])
    t2=time()
    print("Parameters loaded in: {} s".format(t2-t1))
    ## Model definition
    print("Defining model")
    model=GetElectricSystemModel_Belfort_SingleNode(areaConsumption, lossesRate, availabilityFactor,
                                                    TechParameters, StorageParameters,
                                                    to_flex_consumption, FlexParameters, labour_ratio)
    t3=time()
    print("Model defined in: {} s".format(t3-t2))

    ## Solving model
    print("Solving model")
    solver='mosek'
    opt = SolverFactory(solver)
    results = opt.solve(model)
    Variables = getVariables_panda_indexed(model)
    t4=time()
    print("Model solved in: {} s".format(t4 - t3))

    ##

    print("Total timing: {} s".format(t4 - t1))
    print(Variables)


run_model(2050,bati_hyp='ref',reindus=True,nuc='plus')

