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
                            v["energyCost"] = v["energyCost"] + v["margvarCost"] * v["maxCapacity"] * (sub - 0.5) * 2
                            v["minCapacity"] = v["minCapacity"] / n
                            v["maxCapacity"] = v["maxCapacity"] / n
                            v.set_index(["AREAS", "TECHNOLOGIES"], inplace=True)
                            TechParameters = pd.concat([TechParameters, v])
                    else:
                        if abs(sub-(0.5-step/2))<epsilon or abs(sub-(0.5+step/2))<epsilon:
                            v = u.reset_index().copy()
                            v["TECHNOLOGIES"] = tech + str(sub)[2:]
                            v["energyCost"] = v["energyCost"] + v["margvarCost"] * v["maxCapacity"] * (sub - 0.5) * 2
                            v["minCapacity"] = v["minCapacity"] / 4
                            v["maxCapacity"] = v["maxCapacity"] / 4
                            v.set_index(["AREAS", "TECHNOLOGIES"], inplace=True)
                            TechParameters = pd.concat([TechParameters, v])
                        else:
                            v = u.reset_index().copy()
                            v["TECHNOLOGIES"] = tech + str(sub)[2:]
                            v["energyCost"] = v["energyCost"] + v["margvarCost"] * v["maxCapacity"] * (sub - 0.5) * 2
                            v["minCapacity"] = v["minCapacity"] / n
                            v["maxCapacity"] = v["maxCapacity"] / n
                            v.set_index(["AREAS", "TECHNOLOGIES"], inplace=True)
                            TechParameters = pd.concat([TechParameters, v])
        return TechParameters
    else:
        return TechParameters


def CHP_processing(areaConsumption,xls_file):
    chp_production = pd.read_excel(xls_file,"chpProduction")
    chp_production["Date"] = pd.to_datetime(chp_production["Date"])
    chp_production.set_index(["AREAS", "Date"], inplace=True)
    chp_production["chpProduction"] = chp_production.chpProduction.astype(float)
    for country in chp_production.reset_index().AREAS.unique():
        areaConsumption.loc[country, "areaConsumption"] = areaConsumption.loc[country, "areaConsumption"].to_numpy() - \
                                                          chp_production.loc[country, "chpProduction"].to_numpy()
    return areaConsumption


def Thermosensibility(areaConsumption,xls_file):
    th_sensi= pd.read_excel(xls_file, "Thermosensi")
    temp = pd.read_excel(xls_file, "ConsoTemp")
    temp["Date"] = pd.to_datetime(temp["Date"])
    temp = temp[temp.Date.dt.year == 2018]
    areas=temp.AREAS.unique()
    temp.set_index(["AREAS", "Date"], inplace=True)
    temp["Consumption"] = areaConsumption["areaConsumption"]
    for area in areas:
        temp_country = temp.loc[(area, slice(None)), :]
        # print(th_sensi)
        th_sensi_country=th_sensi.loc[th_sensi.AREAS==area,"Th_sensi"].sum()
        (ConsoTempeYear_decomposed_df, Thermosensibilite) = Decomposeconso(temp_country, TemperatureThreshold=15)
        coef=th_sensi_country/abs(np.array(list(Thermosensibilite.values())).mean()/1000)
        NewThermosensibilite = {}
        for key in Thermosensibilite:    NewThermosensibilite[key] = coef * Thermosensibilite[key]
        NewConsoTempeYear_decomposed_df = Recompose(ConsoTempeYear_decomposed_df, NewThermosensibilite,
                                                    TemperatureThreshold=15)
    areaConsumption.loc[(area,slice(None)),"areaConsumption"]=NewConsoTempeYear_decomposed_df["Consumption"]
    return areaConsumption

def Flexibility_data_processing(areaConsumption,year,xls_file):
    ConsoParameters = pd.read_excel(xls_file,"FLEX_CONSUM")

    areas_list=ConsoParameters.AREAS.unique()
    ConsoParameters_=pd.DataFrame(columns=["AREAS","FLEX_CONSUM","unit","add_consum","LoadCost","flex_ratio","flex_type","labourcost"],data=np.array([[None]*8])).set_index(["AREAS", "FLEX_CONSUM"])
    ConsoParameters.set_index(["AREAS", "FLEX_CONSUM"], inplace=True)
    to_flex_consumption=pd.DataFrame(columns=["AREAS","Date","FLEX_CONSUM","to_flex_consumption"],data=np.array([[None]*4])).set_index(["AREAS","Date","FLEX_CONSUM"])
    labour_ratios=pd.DataFrame(columns=["AREAS","Date","FLEX_CONSUM","labour_ratio"],data=np.array([[None]*4])).set_index(["AREAS", "Date", "FLEX_CONSUM"])
    for area in areas_list:
        ConsoTempe_df = pd.read_excel(xls_file,"ConsoTemp",parse_dates=['Date']).set_index(["AREAS","Date"])
        ConsoTempe_df_nodup = ConsoTempe_df.loc[~ConsoTempe_df.index.duplicated(), :]
        ConsoTempe_df_nodup=ConsoTempe_df_nodup.loc[(area,slice(None)),:].reset_index().set_index("Date")

        VEProfile_df = pd.read_excel(xls_file,'EVModel')
        NbVE = ConsoParameters.loc[(area,"EV"),"add_consum"] # millions
        ev_consumption = NbVE * Profile2Consumption(Profile_df=VEProfile_df,
                                                    Temperature_df=ConsoTempe_df_nodup.loc[str(year)][
                                                        ['Temperature']])[ ['Consumption']]
        ev_consumption.reset_index(inplace=True)
        ev_consumption["Date"] = pd.to_datetime(ev_consumption["Date"]) #+ pd.DateOffset(years=year - weather_year)
        ev_consumption.set_index("Date", inplace=True)
        h2_Energy = ConsoParameters.loc[(area,"H2"),"add_consum"] * 10 ** 6  ## H2 volume in MWh/year
        h2_Energy_flat_consumption = ev_consumption.Consumption * 0 + h2_Energy / bisextile(year)
        to_flex_consumption = pd.concat([to_flex_consumption, pd.concat([pd.DataFrame(
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

        labour_ratio = pd.DataFrame(columns=["AREAS", "Date", "FLEX_CONSUM","labour_ratio"])

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


def CHP_processing_single_node(areaConsumption,xls_file):
    chp_production = pd.read_excel(xls_file,"chpProduction")
    chp_production["Date"] = pd.to_datetime(chp_production["Date"])
    chp_production.set_index(["Date"], inplace=True)
    chp_production["chpProduction"] = chp_production.chpProduction.astype(float)
    areaConsumption["areaConsumption"] = areaConsumption["areaConsumption"].to_numpy() - \
                                                          chp_production["chpProduction"].to_numpy()
    return areaConsumption

def Thermosensibility_single_node(areaConsumption,xls_file):
    th_sensi_country= pd.read_excel(xls_file, "Thermosensi").loc[:,"Th_sensi"].sum()
    temp = pd.read_excel(xls_file, "ConsoTemp")
    temp["Date"] = pd.to_datetime(temp["Date"])
    temp = temp[temp.Date.dt.year == 2018]
    temp.set_index(["Date"], inplace=True)
    temp["Consumption"] = areaConsumption["areaConsumption"]
    temp_country = temp

    (ConsoTempeYear_decomposed_df, Thermosensibilite) = Decomposeconso(temp_country, TemperatureThreshold=15)
    coef=th_sensi_country/abs(np.array(list(Thermosensibilite.values())).mean()/1000)
    NewThermosensibilite = {}
    for key in Thermosensibilite:    NewThermosensibilite[key] = coef * Thermosensibilite[key]
    NewConsoTempeYear_decomposed_df = Recompose(ConsoTempeYear_decomposed_df, NewThermosensibilite,
                                                    TemperatureThreshold=15)
    areaConsumption["areaConsumption"]=NewConsoTempeYear_decomposed_df["Consumption"]
    return areaConsumption

def Flexibility_data_processing_single_node(areaConsumption,year,xls_file):
    ConsoParameters = pd.read_excel(xls_file,"FLEX_CONSUM")


    ConsoParameters_=pd.DataFrame(columns=["FLEX_CONSUM","unit","add_consum","LoadCost","flex_ratio","flex_type","labourcost"],data=np.array([[None]*7])).set_index(["FLEX_CONSUM"])
    ConsoParameters.set_index(["FLEX_CONSUM"], inplace=True)
    to_flex_consumption=pd.DataFrame(columns=["Date","FLEX_CONSUM","to_flex_consumption"],data=np.array([[None]*3])).set_index(["Date","FLEX_CONSUM"])
    labour_ratios=pd.DataFrame(columns=["Date","FLEX_CONSUM","labour_ratio"],data=np.array([[None]*3])).set_index(["Date", "FLEX_CONSUM"])
    ConsoTempe_df = pd.read_excel(xls_file,"ConsoTemp",parse_dates=['Date']).set_index(["Date"])
    ConsoTempe_df_nodup = ConsoTempe_df.loc[~ConsoTempe_df.index.duplicated(), :]
    VEProfile_df = pd.read_excel(xls_file,'EVModel')
    NbVE = ConsoParameters.loc[("EV"),"add_consum"] # millions
    ev_consumption = NbVE * Profile2Consumption(Profile_df=VEProfile_df,
                                                Temperature_df=ConsoTempe_df_nodup.loc[str(year)][
                                                    ['Temperature']])[ ['Consumption']]
    ev_consumption.reset_index(inplace=True)
    ev_consumption["Date"] = pd.to_datetime(ev_consumption["Date"]) #+ pd.DateOffset(years=year - weather_year)
    ev_consumption.set_index("Date", inplace=True)
    h2_Energy = ConsoParameters.loc[("H2"),"add_consum"] * 10 ** 6  ## H2 volume in MWh/year
    h2_Energy_flat_consumption = ev_consumption.Consumption * 0 + h2_Energy / bisextile(year)
    to_flex_consumption = pd.concat([to_flex_consumption, pd.concat([pd.DataFrame(
                                         {'to_flex_consumption': ev_consumption.Consumption, 'FLEX_CONSUM': 'EV'}).reset_index().set_index(
                                         ['Date', 'FLEX_CONSUM']),
                                     pd.DataFrame(
                                         {'to_flex_consumption': h2_Energy_flat_consumption, 'FLEX_CONSUM': 'H2'}).reset_index().set_index(
                                         ['Date', 'FLEX_CONSUM'])])])
    ConsoParameters_ =pd.concat([ConsoParameters_, ConsoParameters.join(
        to_flex_consumption.groupby(["FLEX_CONSUM"]).max().rename(columns={"to_flex_consumption": "max_power"}))])


    def labour_ratio_cost(df):  # higher labour costs at night
        if df.hour in range(7, 17):
            return 1
        elif df.hour in range(17, 23):
            return 1.5
        else:
            return 2

    labour_ratio = pd.DataFrame(columns=["Date", "FLEX_CONSUM","labour_ratio"])

    for flex_consum in ["EV", "H2"]:
        u = pd.DataFrame()
        u["Date"] = areaConsumption.index.get_level_values('Date')
        u["FLEX_CONSUM"] = flex_consum
        u["labour_ratio"] = np.array(len(u["Date"]) * [1])
        labour_ratio = pd.concat([labour_ratio, u], ignore_index=True)

    labour_ratio.set_index(["Date", "FLEX_CONSUM"], inplace=True)
    labour_ratios=pd.concat([labour_ratios,labour_ratio])

    ConsoParameters_.drop(["unit", "add_consum"], axis=1, inplace=True)
    ConsoParameters_.dropna(inplace=True)
    ConsoParameters_.reset_index(inplace=True)
    ConsoParameters_.drop_duplicates(inplace=True)
    ConsoParameters_.set_index(["FLEX_CONSUM"],inplace=True)
    to_flex_consumption.dropna(inplace=True)
    labour_ratios.dropna(inplace=True)
    return ConsoParameters_,labour_ratios, to_flex_consumption

