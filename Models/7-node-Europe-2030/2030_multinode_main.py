# region importation of modules
import os

if os.path.basename(os.getcwd()) == "Basicfunctions" or os.path.basename(os.getcwd()) == "Code":
    os.chdir('../../../../../../Downloads/Avancement')  ## to work at project root  like in any IDE

from Models.Basic_France_models.Planing_optimisation.f_planingModels import *
#from functions.f_optimization import *
from functions.f_consumptionModels import *
from pyomo.opt import SolverFactory

# endregion

# region Solver and data location definition
InputFolder = 'Data/input/Multi_node/'
if sys.platform != 'win32':
    myhost = os.uname()[1]
else:
    myhost = ""

solver = 'mosek'  ## no need for solverpath with mosek.


# endregion

###region Functions
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
###endregion

#region Parameters
steel_elec=5
h2_elec=25
nb_EVs=5

steel_flex_ratio=0.9
h2_flex_ratio=1
ev_flex_ratio=0.15

#endregion
#### reading areaConsumption availabilityFactor and TechParameters CSV files
TechParameters = pd.read_csv(InputFolder+'2030_Planing_MultiNode_TECHNOLOGIES_AREAS.csv',
                             sep=';',decimal='.',comment="#").set_index(["AREAS","TECHNOLOGIES"])

areaConsumption = pd.read_csv(InputFolder+'MultiNodeAreaConsumption_FR_IT_GB_CH_DE_ES_BE_2030.csv',
                                sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["AREAS","Date"])
availabilityFactor = pd.read_csv(InputFolder+'2030_Multinode_availability_factor.csv',
                                sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["AREAS","Date","TECHNOLOGIES"])

ExchangeParameters = pd.read_csv(InputFolder+'2030_interconnexions.csv',
                                 sep=';',decimal='.',skiprows=0,comment="#").set_index(["AREAS","AREAS.1"])
StorageParameters = pd.read_csv(InputFolder+'2030_Planing_MultiNode_STOCK_TECHNO.csv',
                                sep=';',decimal='.',comment="#",skiprows=0).set_index(["AREAS","STOCK_TECHNO"])
ConsoParameters = pd.read_csv(InputFolder + "Planing-Conso-FLEX_CONSUM.csv",
                              sep=";").set_index(["AREAS","FLEX_CONSUM"])

#obtaining industry-metal consumption
Profile_df = pd.read_csv(InputFolder + "ConsumptionDetailedProfiles.csv")
Profile_df = Profile_df[Profile_df.type == "Ind"]
Profile_df = Profile_df[Profile_df.Nature == "MineraiMetal"]
Profile_df = Profile_df[Profile_df.UsagesGroupe == "Process"]
steel_consumption = LoadProfile2Consumption(Profile_df, 2030, steel_elec).set_index(["Date"])
steel_consumption.Conso = steel_consumption.Conso * steel_elec * 10 ** 6 / steel_consumption.Conso.sum()


ConsoTempe_df=pd.read_csv(InputFolder+'ConsumptionTemperature_1996TO2019_FR.csv',parse_dates=['Date']).set_index(["Date"]) #
ConsoTempe_df_nodup=ConsoTempe_df.loc[~ConsoTempe_df.index.duplicated(),:]


VEProfile_df=pd.read_csv(InputFolder+'EVModel.csv', sep=';')
NbVE=nb_EVs # millions
ev_consumption = NbVE*Profile2Consumption(Profile_df=VEProfile_df,Temperature_df = ConsoTempe_df_nodup.loc[str(2018)][['Temperature']])[['Consumption']]
ev_consumption.reset_index(inplace=True)
ev_consumption["Date"]=ev_consumption["Date"]+pd.offsets.DateOffset(years=12) #to have 2030 instead of 2018
ev_consumption.set_index("Date",inplace=True)





h2_Energy = h2_elec*1000## H2 volume in GWh/year
h2_Energy_flat_consumption = ev_consumption.Consumption*0+h2_Energy/8760

to_flex_consumption=pd.concat([pd.DataFrame({'to_flex_consumption': steel_consumption.Conso,'FLEX_CONSUM' : 'Steel','AREAS':'FR'}).reset_index().set_index(['Date','FLEX_CONSUM','AREAS']),
    pd.DataFrame({'to_flex_consumption': ev_consumption.Consumption,'FLEX_CONSUM' : 'EV','AREAS':'FR'}).reset_index().set_index(['Date','FLEX_CONSUM','AREAS']),
    pd.DataFrame({'to_flex_consumption': h2_Energy_flat_consumption,'FLEX_CONSUM' : 'H2','AREAS':'FR'}).reset_index().set_index(['Date','FLEX_CONSUM','AREAS'])])


ConsoParameters_ = ConsoParameters.join(
    to_flex_consumption.groupby("FLEX_CONSUM").max().rename(columns={"to_flex_consumption": "max_power"}))
ConsoParameters.loc["Steel","flex_ratio"]=steel_flex_ratio
ConsoParameters.loc["EV", "flex_ratio"] = ev_flex_ratio
ConsoParameters.loc["H2", "flex_ratio"] = h2_flex_ratio

def labour_ratio_cost(df):  # higher labour costs at night
    if df.hour in range(7, 17):
        return 1
    elif df.hour in range(17, 23):
        return 1.5
    else:
        return 2


labour_ratio = pd.DataFrame()
labour_ratio["AREAS"]="FR"
labour_ratio["Date"] = areaConsumption.index.get_level_values('Date')
labour_ratio["FLEX_CONSUM"] = "Steel"
labour_ratio["labour_ratio"] = labour_ratio["Date"].apply(labour_ratio_cost)
labour_ratio.set_index(["AREAS","Date","FLEX_CONSUM"], inplace=True)

model =  GetElectricSystemModel_Planing(Parameters={"areaConsumption"      :   areaConsumption,
                                           "availabilityFactor"   :   availabilityFactor,
                                           "TechParameters"       :   TechParameters,
                                           "StorageParameters"   : StorageParameters,
                                           "to_flex_consumption" : to_flex_consumption,
                                           "ConsoParameters_" : ConsoParameters_,
                                           "labour_ratio": labour_ratio})

 if solver in solverpath :  opt = SolverFactory(solver,executable=solverpath[solver])
 else : opt = SolverFactory(solver)
 results=opt.solve(model)
# Variables = getVariables_panda_indexed(model)
# print(Variables)