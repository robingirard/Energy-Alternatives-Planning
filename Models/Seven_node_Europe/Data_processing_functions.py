import pandas as pd
from EnergyAlternativesPlaning.f_consumptionModels import *

InputFolder = 'Input data/'
InputFolder_other="Input data/Conso flex files/"

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


def bisextile(year):
    if year%4==0:
        return 8784
    else:
        return 8760

def Marginal_cost_adjustment(TechParameters,number_of_sub_techs,techs,areas,carbon_tax=30,carbon_tax_ini=30,gas_price_coef=1,coal_price_coef=1):
    epsilon=0.005
    tech_emissions = {"CCG": 0.364, "TAC": 0.502, "Coal": 0.740, "Lignite": 0.889}
    fossil_factor = {"CCG": gas_price_coef, "TAC": gas_price_coef, "Coal": coal_price_coef, "Lignite": 1}

    TechParameters.reset_index(inplace=True)
    for tech in tech_emissions.keys():
        TechParameters.loc[TechParameters.TECHNOLOGIES == tech, "energyCost"] = \
            (TechParameters.loc[TechParameters.TECHNOLOGIES == tech, "energyCost"].astype(float) - \
             carbon_tax_ini * tech_emissions[tech]) * fossil_factor[tech] + carbon_tax* tech_emissions[tech]

    TechParameters.set_index(["AREAS", "TECHNOLOGIES"], inplace=True)

    if number_of_sub_techs >2:
        if number_of_sub_techs%2 != 0:
            n = (number_of_sub_techs - 1) * 2
        else:
            n=number_of_sub_techs
        step = 1 / (number_of_sub_techs + 1)
        n_list = np.arange(step, 1-step+epsilon, step).tolist()
        # print(TechParameters)
        for tech in techs:
            for area in areas:
                # print(TechParameters[TechParameters.index==(area,tech)])
                if tech in ["OldNuke", "NewNuke", "Solar", 'WindOnShore', 'WindOffShore', 'HydroRiver', 'HydroReservoir',
                            'curtailment']:
                    break
                u = TechParameters[TechParameters.index == (area, tech)]
                if number_of_sub_techs%2 != 0:

                    TechParameters.loc[(area, tech), "minCapacity"] = TechParameters.loc[(area, tech), "minCapacity"] / 2
                    TechParameters.loc[(area, tech), "maxCapacity"] = TechParameters.loc[(area, tech), "maxCapacity"] / 2
                else:
                    TechParameters.drop((area, tech),inplace=True)

                for sub in n_list:
                    if number_of_sub_techs%2 != 0:
                        if sub != 0.5 :
                            v = u.reset_index().copy()
                            v["TECHNOLOGIES"] = tech + str(sub)[2:]
                            v["energyCost"] = v["energyCost"] + v["margvarCost"] * v["minCapacity"] * (sub - 0.5) * 2
                            v["minCapacity"] = v["minCapacity"] / n
                            v["maxCapacity"] = v["maxCapacity"] / n
                            v.set_index(["AREAS", "TECHNOLOGIES"], inplace=True)
                            TechParameters = pd.concat([TechParameters, v])
                    else:
                        if abs(sub-(0.5-step/2))<epsilon or abs(sub-(0.5+step/2))<epsilon:
                            v = u.reset_index().copy()
                            v["TECHNOLOGIES"] = tech + str(sub)[2:]
                            v["energyCost"] = v["energyCost"] + v["margvarCost"] * v["minCapacity"] * (sub - 0.5) * 2
                            v["minCapacity"] = v["minCapacity"] / 4
                            v["maxCapacity"] = v["maxCapacity"] / 4
                            v.set_index(["AREAS", "TECHNOLOGIES"], inplace=True)
                            TechParameters = pd.concat([TechParameters, v])
                        else:
                            v = u.reset_index().copy()
                            v["TECHNOLOGIES"] = tech + str(sub)[2:]
                            v["energyCost"] = v["energyCost"] + v["margvarCost"] * v["minCapacity"] * (sub - 0.5) * 2
                            v["minCapacity"] = v["minCapacity"] / n
                            v["maxCapacity"] = v["maxCapacity"] / n
                            v.set_index(["AREAS", "TECHNOLOGIES"], inplace=True)
                            TechParameters = pd.concat([TechParameters, v])
        return TechParameters
    else:
        return TechParameters

def CHP_processing_future(areaConsumption,year,weather_year):
    chp_production = pd.read_csv(InputFolder + str(year)+"/"+str(year) + '_' + str(weather_year) + '_chpProduction.csv')
    chp_production["Date"] = pd.to_datetime(chp_production["Date"])
    chp_production.set_index(["AREAS", "Date"], inplace=True)
    chp_production["chpProduction"] = chp_production.chpProduction.astype(float)
    for country in chp_production.reset_index().AREAS.unique():
        areaConsumption.loc[country, "areaConsumption"] = areaConsumption.loc[country, "areaConsumption"].to_numpy() - \
                                                          chp_production.loc[country, "chpProduction"].to_numpy()
    return areaConsumption

def CHP_processing(areaConsumption,year):
    chp_production = pd.read_csv(InputFolder + str(year)+"/"+str(year) + '_chpProduction.csv')
    chp_production["Date"] = pd.to_datetime(chp_production["Date"])
    chp_production.set_index(["AREAS", "Date"], inplace=True)
    chp_production["chpProduction"] = chp_production.chpProduction.astype(float)
    for country in chp_production.reset_index().AREAS.unique():
        areaConsumption.loc[country, "areaConsumption"] = areaConsumption.loc[country, "areaConsumption"].to_numpy() - \
                                                          chp_production.loc[country, "chpProduction"].to_numpy()
    return areaConsumption


def Flexibility_data_processing(areaConsumption,year,weather_year):
    ConsoParameters = pd.read_csv(InputFolder_other + "2030_Planing-Conso-FLEX_CONSUM.csv",
                                  sep=";")
    areas_list=ConsoParameters.AREAS.unique()
    ConsoParameters_=pd.DataFrame(columns=["AREAS","FLEX_CONSUM","unit","add_consum","LoadCost","flex_ratio","flex_type","labourcost"],data=np.array([[None]*8])).set_index(["AREAS", "FLEX_CONSUM"])
    ConsoParameters.set_index(["AREAS", "FLEX_CONSUM"], inplace=True)
    to_flex_consumption=pd.DataFrame(columns=["AREAS","Date","FLEX_CONSUM","to_flex_consumption"],data=np.array([[None]*4])).set_index(["AREAS","Date","FLEX_CONSUM"])
    labour_ratios=pd.DataFrame(columns=["AREAS","Date","FLEX_CONSUM","labour_ratio"],data=np.array([[None]*4])).set_index(["AREAS", "Date", "FLEX_CONSUM"])
    for area in areas_list:
        # obtaining industry-metal consumption
        Profile_df = pd.read_csv(InputFolder_other + "ConsumptionDetailedProfiles.csv")
        # print(Profile_df)
        Profile_df = Profile_df[Profile_df.type == "Ind"]
        # print(Profile_df)
        Profile_df = Profile_df[Profile_df.Nature == "MineraiMetal"]
        # print(Profile_df.UsagesGroupe.unique())
        Profile_df = Profile_df[Profile_df.UsagesGroupe == "Process"]
        # print(Profile_df)
        steel_consumption = LoadProfile2Consumption(Profile_df, year, ConsoParameters.loc[(area,"Steel"),"add_consum"]).set_index(
            ["Date"])
        ConsoTempe_df = pd.read_csv(InputFolder_other + 'ConsumptionTemperature_1996TO2019_FR.csv',
                                    parse_dates=['Date']).set_index(["Date"])  #
        ConsoTempe_df_nodup = ConsoTempe_df.loc[~ConsoTempe_df.index.duplicated(), :]
        VEProfile_df = pd.read_csv(InputFolder_other + 'EVModel.csv', sep=';')
        NbVE = ConsoParameters.loc[(area,"EV"),"add_consum"] # millions
        ev_consumption = NbVE * Profile2Consumption(Profile_df=VEProfile_df,
                                                    Temperature_df=ConsoTempe_df_nodup.loc[str(weather_year)][
                                                        ['Temperature']])[ ['Consumption']]
        ev_consumption.reset_index(inplace=True)
        ev_consumption["Date"] = pd.to_datetime(ev_consumption["Date"]) + pd.DateOffset(years=year - weather_year)
        ev_consumption.set_index("Date", inplace=True)
        h2_Energy = ConsoParameters.loc[(area,"H2"),"add_consum"] * 10 ** 6  ## H2 volume in MWh/year
        h2_Energy_flat_consumption = ev_consumption.Consumption * 0 + h2_Energy / bisextile(year)
        to_flex_consumption = pd.concat([to_flex_consumption, pd.concat([pd.DataFrame(
            {'to_flex_consumption': steel_consumption.Conso, 'FLEX_CONSUM': 'Steel',
             'AREAS': area}).reset_index().set_index(['AREAS', 'Date', 'FLEX_CONSUM']),
                                         pd.DataFrame(
                                             {'to_flex_consumption': ev_consumption.Consumption, 'FLEX_CONSUM': 'EV',
                                              'AREAS': area}).reset_index().set_index(['AREAS', 'Date', 'FLEX_CONSUM']),
                                         pd.DataFrame(
                                             {'to_flex_consumption': h2_Energy_flat_consumption, 'FLEX_CONSUM': 'H2',
                                              'AREAS': area}).reset_index().set_index(
                                             ['AREAS', 'Date', 'FLEX_CONSUM'])])])
        ConsoParameters_ =pd.concat([ConsoParameters_, ConsoParameters.join(
            to_flex_consumption.groupby(["AREAS","FLEX_CONSUM"]).max().rename(columns={"to_flex_consumption": "max_power"}))])


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
        labour_ratio["AREAS"] = area
        labour_ratio["labour_ratio"] = labour_ratio["Date"].apply(labour_ratio_cost)

        for flex_consum in ["EV", "H2"]:
            u = pd.DataFrame()
            u["Date"] = areaConsumption.index.get_level_values('Date')
            u["FLEX_CONSUM"] = flex_consum
            u["AREAS"] = area
            u["labour_ratio"] = np.array(len(u["Date"]) * [1])
            labour_ratio = pd.concat([labour_ratio, u], ignore_index=True)

        labour_ratio.set_index(["AREAS", "Date", "FLEX_CONSUM"], inplace=True)
        labour_ratios=pd.concat([labour_ratios,labour_ratio])

    ConsoParameters_.drop(["unit", "add_consum"], axis=1, inplace=True)
    ConsoParameters_.dropna(inplace=True)
    ConsoParameters_.reset_index(inplace=True)
    ConsoParameters_.drop_duplicates(inplace=True)
    ConsoParameters_.set_index(["AREAS","FLEX_CONSUM"],inplace=True)
    to_flex_consumption.dropna(inplace=True)
    labour_ratios.dropna(inplace=True)
    return ConsoParameters_,labour_ratios, to_flex_consumption

def Curtailment_adjustment(TechParameters,minCapacity=5000,maxCapacity=5000,EnergyNbHourCap=20):
    TechParameters.reset_index(inplace=True)
    TechParameters.loc[TechParameters.TECHNOLOGIES == "curtailment", "minCapacity"] = minCapacity
    TechParameters.loc[TechParameters.TECHNOLOGIES == "curtailment", "maxCapacity"] = maxCapacity
    TechParameters.loc[TechParameters.TECHNOLOGIES == "curtailment", "EnergyNbhourCap"] = EnergyNbHourCap
    TechParameters.set_index(["AREAS", "TECHNOLOGIES"], inplace=True)
    return TechParameters