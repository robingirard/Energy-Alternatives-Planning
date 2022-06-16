# region importation of modules
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import psutil


import pickle
if os.path.basename(os.getcwd()) == "SujetsDAnalyses":
    os.chdir('..')  ## to work at project root  like in any IDE

import sys

from functions.f_graphicalTools import *
from functions.f_planingModels import *
#from functions.f_optimization import *
from functions.f_graphicalTools import *
from functions.f_consumptionModels import *
from pyomo.core import *
from pyomo.opt import SolverFactory

# endregion
###region Profile extraction function
def LoadProfile2Consumption(Profile_df, year, annual_consum):
    ## initialisation
    ConsoSepareeNew_df = pd.DataFrame()
    ConsoSepareeNew_df["Time"] = pd.date_range(start="1/1/" + str(year), end="31/12/" + str(year) + " 23:00", freq="H")
    # ConsoSepareeNew_df.set_index("Time",inplace=True)
    ConsoSepareeNew_df = ConsoSepareeNew_df.assign(
        WeekDay=ConsoSepareeNew_df["Time"].dt.weekday,
        Mois=ConsoSepareeNew_df["Time"].dt.month,
        heures=ConsoSepareeNew_df["Time"].dt.hour);
    ConsoSepareeNew_df['WeekDay'] = ConsoSepareeNew_df['WeekDay'].apply(
        lambda x: "Week" if x < 5 else "Sat" if x == 5 else "Sun")
    Profile_df["heures"].replace(24, 0, inplace=True)
    Profile_df_merged = ConsoSepareeNew_df.merge(Profile_df, on=["WeekDay", "Mois", "heures"])
    Profile_df_merged = Profile_df_merged[["Time", "Conso"]]
    Profile_df_merged.sort_values("Time", inplace=True)
    Profile_df_merged["Conso"] = Profile_df_merged["Conso"] * annual_consum * 10 ** 6 / Profile_df_merged["Conso"].sum()
    Profile_df_merged.reset_index(drop=True,inplace=True)
    Profile_df_merged.rename(columns={"Time":"Date"},inplace=True)
    return Profile_df_merged


def EVProfile2Consumption(Profile_df,year,nb_EV):
    ConsoSepareeNew_df = pd.DataFrame()
    ConsoSepareeNew_df["Time"] = pd.date_range(start="1/1/" + str(year), end="31/12/" + str(year) + " 23:00", freq="H")
    # ConsoSepareeNew_df.set_index("Time",inplace=True)
    ConsoSepareeNew_df = ConsoSepareeNew_df.assign(
        Saison=ConsoSepareeNew_df["Time"].dt.month,
        Jour=ConsoSepareeNew_df["Time"].dt.weekday,
        Heure=ConsoSepareeNew_df["Time"].dt.hour);
    ConsoSepareeNew_df['Jour'] = ConsoSepareeNew_df['Jour']+1
    ConsoSepareeNew_df['Saison'] = ConsoSepareeNew_df['Saison'].apply(
        lambda x: "Hiver" if x in [10,11,12,1,2,3] else "Ete")

    Profile_df_merged = ConsoSepareeNew_df.merge(Profile_df, on=["Saison","Jour", "Heure"])

    Profile_df_merged = Profile_df_merged[["Time", "Puissance.MW.par.million"]]
    Profile_df_merged.sort_values("Time", inplace=True)
    Profile_df_merged["Puissance.MW.par.million"] = Profile_df_merged["Puissance.MW.par.million"] * nb_EV
    Profile_df_merged.rename(columns={"Puissance.MW.par.million":"Conso"},inplace=True)
    Profile_df_merged.reset_index(drop=True,inplace=True)
    return Profile_df_merged

###endregion

###region Other functions
def bisextile(year):
    if year%4==0:
        return 8784
    else:
        return 8760


def main(year,with_external_nodes=True,number_of_sub_techs=1,error_deactivation=False,fr_flexibility=False,fr_flex_consum={'conso':{"nbVE":0,"steel_twh":0,"H2_twh":0},'ratio':{"VE_ratio":0,"steel_ratio":0,"H2_ratio":0}}):
    start_time = datetime.now()
    print("Simulation for "+str(year))
    if year in range(2017,2020):
        # region Solver and data location definition
        InputFolder = 'Data/input/Europe 7 noeuds/'+str(year)+"/"
        InputFolder_other="Data/input/Europe 7 noeuds/Conso flex files/"
    else:
        print("No data for that year --> Data is available for years between 2017 and 2019")
        return None

    if error_deactivation:
        import logging #to deactivate pyomo false error warnings
        logging.getLogger('pyomo.core').setLevel(logging.ERROR)

    if sys.platform != 'win32':
        myhost = os.uname()[1]
    else:
        myhost = ""

    solver = 'mosek' #'mosek'  ## no need for solverpath with mosek.
    tee_value=True
    # solver_path="ampl.mswin64/"+solver #when other solvers were tested
    # print("Solver is "+solver)
    solver_native_list=["mosek","glpk"]
    # endregion



    #### reading areaConsumption availabilityFactor and TechParameters CSV files
    TechParameters = pd.read_csv(InputFolder+str(year)+'_Planing_MultiNode_TECHNOLOGIES_AREAS.csv',decimal='.',comment="#")

    #Readjustment of ramps
    # TechParameters.loc[TechParameters["TECHNOLOGIES"].isin(["OldNuke","NewNuke"]),"RampConstraintPlus"]=0.25
    # TechParameters.loc[TechParameters["TECHNOLOGIES"].isin(["Coal","Biomass"]),"RampConstraintPlus"]=0.5

    #Power capacity readjustment


    techs=TechParameters.TECHNOLOGIES.unique()
    areas=TechParameters.AREAS.unique()
    TechParameters.set_index(["AREAS","TECHNOLOGIES"],inplace=True)
    ##
    areaConsumption = pd.read_csv(InputFolder+str(year)+'_MultiNodeAreaConsumption.csv',
                                    decimal='.',skiprows=0,parse_dates=['Date'])
    areaConsumption["Date"]=pd.to_datetime(areaConsumption["Date"])
    areaConsumption.set_index(["AREAS","Date"],inplace=True)

    chp_production=pd.read_csv(InputFolder+str(year)+'_chpProduction.csv')
    chp_production["Date"]=pd.to_datetime(chp_production["Date"])
    chp_production.set_index(["AREAS","Date"],inplace=True)
    chp_production["chpProduction"]=chp_production.chpProduction.astype(float)
    for country in chp_production.reset_index().AREAS.unique():
        areaConsumption.loc[country,"areaConsumption"] = areaConsumption.loc[country,"areaConsumption"].to_numpy() -  chp_production.loc[country,"chpProduction"].to_numpy()


    if with_external_nodes:
        ##Exchanges add/removal from areaConsumption
        outsideExchanges=pd.read_csv(InputFolder+str(year)+'_Multinode_Outside_Fixed_Exchanges.csv')
        outsideExchanges["Date"]=pd.to_datetime(outsideExchanges["Date"])
        outsideExchanges.set_index(["AREAS","Date"],inplace=True)
        outsideExchanges.rename(columns={"Load":"areaConsumption"},inplace=True)
        areaConsumption= areaConsumption+outsideExchanges
        ##

    availabilityFactor = pd.read_csv(InputFolder+str(year)+'_Multinode_availability_factor.csv',
                                    decimal='.',skiprows=0,parse_dates=['Date']).set_index(["AREAS","Date","TECHNOLOGIES"])

    # u=availabilityFactor.reset_index()
    #print(u[(u["AREAS"]=="FR") & (u["TECHNOLOGIES"]=="HydroReservoir")])
    # print(availabilityFactor[availabilityFactor.availabilityFactor>1])
    availabilityFactor.loc[availabilityFactor.availabilityFactor > 1,"availabilityFactor"]=1
    ExchangeParameters = pd.read_csv(InputFolder+str(year)+'_interconnexions.csv',
                                     decimal='.').set_index(["AREAS","AREAS.1"])
    StorageParameters = pd.read_csv(InputFolder+str(year)+'_Planing_MultiNode_STOCK_TECHNO.csv',
                                    decimal='.',comment="#",skiprows=0).set_index(["AREAS","STOCK_TECHNO"])

    #Marginal Cost at power=0 adjustment
    # TechParameters["energyCost"]=TechParameters["energyCost"]-TechParameters["margvarCost"]*TechParameters["minCapacity"]/2
    number_of_sub_techs= number_of_sub_techs#must be odd (impair)
    n=(number_of_sub_techs-1)*2
    step=1/(number_of_sub_techs+1)
    n_list=np.arange(step,1,step).tolist()
    for tech in techs:
        for area in areas:
            # print(TechParameters[TechParameters.index==(area,tech)])
            if tech in ["OldNuke","NewNuke","Solar",'WindOnShore','WindOffShore','HydroRiver','HydroReservoir','curtailment']:
                break
            u=TechParameters[TechParameters.index==(area,tech)]
            TechParameters.loc[(area, tech),"minCapacity"]=TechParameters.loc[(area, tech),"minCapacity"]/2
            TechParameters.loc[(area, tech), "maxCapacity"] = TechParameters.loc[(area, tech), "maxCapacity"] / 2

            for sub in n_list:
                if sub!=0.5:
                    v=u.reset_index().copy()
                    # print(u25)
                    v["TECHNOLOGIES"]=tech+str(sub)[2:]
                    v["energyCost"]=v["energyCost"]+v["margvarCost"]*v["minCapacity"]*(sub-0.5)*2
                    v["minCapacity"]=v["minCapacity"]/n
                    v["maxCapacity"] = v["maxCapacity"] / n
                    v.set_index(["AREAS","TECHNOLOGIES"],inplace=True)
                    TechParameters=pd.concat([TechParameters,v])
    # TechParameters.to_csv('test.csv')
    # print(TechParameters)
    # return 0
    #Add RampCtr2 to Nuke
    # TechParameters.reset_index(inplace=True)
    # TechParameters.loc[TechParameters.TECHNOLOGIES=="OldNuke","RampConstraintPlus2"] = 0.003
    # TechParameters.loc[TechParameters.TECHNOLOGIES=="OldNuke","RampConstraintMoins2"] = 0.003
    # TechParameters.set_index(["AREAS", "TECHNOLOGIES"], inplace=True)


    if fr_flexibility:
        ConsoParameters = pd.read_csv(InputFolder_other + "Planing-Conso-FLEX_CONSUM.csv",
                                      sep=";").set_index(["AREAS", "FLEX_CONSUM"])

        # obtaining industry-metal consumption
        Profile_df = pd.read_csv(InputFolder_other + "ConsumptionDetailedProfiles.csv")
        Profile_df = Profile_df[Profile_df.type == "Ind"]
        Profile_df = Profile_df[Profile_df.Nature == "MineraiMetal"]
        Profile_df = Profile_df[Profile_df.UsagesGroupe == "Process"]
        steel_consumption = LoadProfile2Consumption(Profile_df, year, fr_flex_consum["conso"]["steel_twh"]).set_index(["Date"])

        ConsoTempe_df = pd.read_csv(InputFolder_other + 'ConsumptionTemperature_1996TO2019_FR.csv',
                                    parse_dates=['Date']).set_index(["Date"])  #
        ConsoTempe_df_nodup = ConsoTempe_df.loc[~ConsoTempe_df.index.duplicated(), :]

        VEProfile_df = pd.read_csv(InputFolder_other + 'EVModel.csv', sep=';')
        NbVE = fr_flex_consum["conso"]["nbVE"]  # millions
        ev_consumption = NbVE * Profile2Consumption(Profile_df=VEProfile_df,
                                                    Temperature_df=ConsoTempe_df_nodup.loc[str(2018)][['Temperature']])[
            ['Consumption']]
        ev_consumption.reset_index(inplace=True)
        ev_consumption.set_index("Date", inplace=True)

        h2_Energy = fr_flex_consum["conso"]["H2_twh"] * 1000000  ## H2 volume in MWh/year
        h2_Energy_flat_consumption = ev_consumption.Consumption * 0 + h2_Energy /  bisextile(year)
        to_flex_consumption = pd.concat([pd.DataFrame(
            {'to_flex_consumption': steel_consumption.Conso, 'FLEX_CONSUM': 'Steel',
             'AREAS': 'FR'}).reset_index().set_index(['AREAS','Date', 'FLEX_CONSUM']),
                                         pd.DataFrame(
                                             {'to_flex_consumption': ev_consumption.Consumption, 'FLEX_CONSUM': 'EV',
                                              'AREAS': 'FR'}).reset_index().set_index(['AREAS','Date', 'FLEX_CONSUM']),
                                         pd.DataFrame(
                                             {'to_flex_consumption': h2_Energy_flat_consumption, 'FLEX_CONSUM': 'H2',
                                              'AREAS': 'FR'}).reset_index().set_index(
                                             ['AREAS','Date', 'FLEX_CONSUM'])])
        # print(to_flex_consumption)
        ConsoParameters_ = ConsoParameters.join(
            to_flex_consumption.groupby("FLEX_CONSUM").max().rename(columns={"to_flex_consumption": "max_power"}))
        ConsoParameters_.loc[("FR","Steel"), "flex_ratio"] = fr_flex_consum["ratio"]["steel_ratio"]
        ConsoParameters_.loc[("FR","EV"), "flex_ratio"] = fr_flex_consum["ratio"]["VE_ratio"]
        ConsoParameters_.loc[("FR","H2"), "flex_ratio"] = fr_flex_consum["ratio"]["H2_ratio"]

        def labour_ratio_cost(df):  # higher labour costs at night
            if df.hour in range(7, 17):
                return 1
            elif df.hour in range(17, 23):
                return 1.5
            else:
                return 2

        labour_ratio = pd.DataFrame()
        labour_ratio["Date"] = areaConsumption.index.get_level_values('Date')
        labour_ratio["FLEX_CONSUM"] = "Steel"
        labour_ratio["AREAS"] = "FR"
        labour_ratio["labour_ratio"] = labour_ratio["Date"].apply(labour_ratio_cost)

        for flex_consum in ["EV","H2"]:
            u=pd.DataFrame()
            u["Date"] = areaConsumption.index.get_level_values('Date')
            u["FLEX_CONSUM"] = flex_consum
            u["AREAS"] = "FR"
            u["labour_ratio"] = np.array(len(u["Date"])*[1])
            labour_ratio=pd.concat([labour_ratio,u],ignore_index=True)

        labour_ratio.set_index(["AREAS", "Date", "FLEX_CONSUM"], inplace=True)
    end_time = datetime.now()
    print('\t Model creation at {}'.format(end_time - start_time))

    if fr_flexibility:
        # print(to_flex_consumption.reset_index().FLEX_CONSUM.unique())
        # print(ConsoParameters_.reset_index().FLEX_CONSUM.unique())
        # print(labour_ratio.reset_index().FLEX_CONSUM.unique())
        model = GetElectricSystemModel_Planing(Parameters={"areaConsumption": areaConsumption,
                                                           "availabilityFactor": availabilityFactor,
                                                           "TechParameters": TechParameters,
                                                           "StorageParameters": StorageParameters,
                                                           "ExchangeParameters": ExchangeParameters,
                                                           "to_flex_consumption": to_flex_consumption,
                                                           "ConsoParameters_": ConsoParameters_,
                                                           "labour_ratio": labour_ratio
                                                           })
    else:
        model =  GetElectricSystemModel_Planing(Parameters={"areaConsumption"      :   areaConsumption,
                                               "availabilityFactor"   :   availabilityFactor,
                                               "TechParameters"       :   TechParameters,
                                               "StorageParameters"   : StorageParameters,
                                               "ExchangeParameters" : ExchangeParameters
                                                            })

    end_time = datetime.now()
    print('\t Start solving at {}'.format(end_time - start_time))


    # opt.options["InfUnbdInfo "] = 1
    # opt.options["ScaleFlag"] = -0.5

    # model.task.optimize()
    if solver in solver_native_list:
        opt = SolverFactory(solver)
    else:
        opt = SolverFactory(solver,executable=solver_path,tee=tee_value)#'C:/Program Files/Artelys/Knitro 13.0.1/knitroampl/knitroampl')
    #To improve performances
    # opt.options['Threads'] = int((psutil.cpu_count(logical=True) +
    #                                  psutil.cpu_count(logical=False)) / 2)
    results=opt.solve(model)#,tee=True,options={"dparam.intpnt_tol_step_size":1.0e-10,"dparam.intpnt_tol_rel_gap": 1.0e-10,
                                               # "dparam.mio_tol_rel_gap": 1.0e-10})

    end_time = datetime.now()
    print('\t Solved at {}'.format(end_time - start_time))
    Variables = getVariables_panda_indexed(model)
    # print(Variables)
    if with_external_nodes:
        with open('SujetsDAnalyses/'+str(year)+'_multinode_results_external_'+solver+'_'+str(number_of_sub_techs)+'_sub.pickle', 'wb') as f:
            pickle.dump(Variables, f,protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('SujetsDAnalyses/'+str(year)+'_multinode_results_no_external_'+solver+'_'+str(number_of_sub_techs)+'_sub.pickle', 'wb') as f:
            pickle.dump(Variables, f,protocol=pickle.HIGHEST_PROTOCOL)

    end_time = datetime.now()
    print('\t Total duration: {}'.format(end_time - start_time))


# main(2017)
# main(2018,with_external_nodes=True,number_of_sub_techs=7,error_deactivation=False,fr_flexibility=False,fr_flex_consum={'conso':{"nbVE":0,"steel_twh":0,"H2_twh":0},'ratio':{"VE_ratio":0,"steel_ratio":0,"H2_ratio":0}})
main(2018,with_external_nodes=False,number_of_sub_techs=7,error_deactivation=False,fr_flexibility=False,fr_flex_consum={'conso':{"nbVE":0,"steel_twh":0,"H2_twh":0},'ratio':{"VE_ratio":0,"steel_ratio":0,"H2_ratio":0}})


# main(2019)