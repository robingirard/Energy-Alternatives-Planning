import numpy as np
import pandas as pd
import xarray as xr
import warnings
import csv
import os
import copy
import datetime
import matplotlib.pyplot as plt
from sklearn import linear_model
from datetime import time
from datetime import datetime
from datetime import date
from datetime import timedelta


# data_df=areaConsumption.join(ConsoTempe_df)[['areaConsumption','Temperature']]
def decompose_demand(temperature, exogenous_energy_demand, temperature_threshold=15):
    demand_name = list(exogenous_energy_demand.keys())[0]

    hours = exogenous_energy_demand.date.dt.hour
    # this produces a warning exogenous_energy_demand.date.dt.hour
    temperature_and_demand = xr.merge([exogenous_energy_demand, hours, temperature])

    def compute_thermosens(x):
        return xr.cov(x.temperature, x.exogenous_energy_demand) / x.temperature.var()

    thermal_sensitivity_estimate = xr.DataArray(
        data=0,
        dims=[ "energy_vector_out","area_to","hour" ],
        coords=dict(
            hour=list(set(temperature_and_demand.hour.to_numpy())),
            area_to=temperature_and_demand.get_index("area_to"),
            energy_vector_out=temperature_and_demand.get_index("energy_vector_out")
        )
    )

    cold_dates = temperature_and_demand.temperature <= temperature_threshold
    ### multi-dim groupby not implemented in xarray !!!
    for area in temperature_and_demand.get_index("area_to"):
        for energy_vector in temperature_and_demand.get_index("energy_vector_out"):
            thermal_sensitivity_estimate.loc[energy_vector, area, :] = temperature_and_demand.where(cold_dates). \
                sel({"energy_vector_out": energy_vector, "area_to": area}).groupby("hour").map(compute_thermosens)

    thermal_sensitive_demand = temperature_and_demand.exogenous_energy_demand.copy()

    for hour in list(set(temperature_and_demand.hour.to_numpy())):
        hour_date = temperature_and_demand.get_index("date")[temperature_and_demand.hour == hour]
        tmp_consumption = cold_dates.sel(date=hour_date) * thermal_sensitivity_estimate[:, :, hour] * (
                    temperature_and_demand.temperature.sel(date=hour_date) - temperature_threshold)
        thermal_sensitive_demand.loc[:, :, hour_date] = tmp_consumption.transpose("energy_vector_out", "area_to",
                                                                                     "date")

    not_thermal_sensitive_demand = temperature_and_demand.exogenous_energy_demand.copy() - thermal_sensitive_demand
    thermal_sensitive_demand = thermal_sensitive_demand.rename(new_name_or_name_dict="thermal_sensitive")
    not_thermal_sensitive_demand = not_thermal_sensitive_demand.rename(new_name_or_name_dict="non_thermal_sensitive")
    thermal_sensitivity_estimate=thermal_sensitivity_estimate.rename(new_name_or_name_dict="thermal_sensitivity_estimate")

    return xr.merge([thermal_sensitive_demand, not_thermal_sensitive_demand,thermal_sensitivity_estimate])

def recompose_demand(decomposed_demand,temperature,thermal_sensitivity,temperature_threshold=15):
    #TODO try to avoid using transpose below
    demand_thermal_sensitive_factor = -(thermal_sensitivity.expand_dims(dim={"hour": range(24)}, axis=1).\
                                transpose("energy_vector_out", "area_to","hour")/ \
                                decomposed_demand.thermal_sensitivity_estimate).thermal_sensitivity
    cold_dates = (temperature.temperature <= temperature_threshold).transpose("energy_vector_out", "area_to","date")
    res = decomposed_demand.non_thermal_sensitive*0
    dates = decomposed_demand.get_index("date")

    for hour in list(set(decomposed_demand.hour.to_numpy())):
        #TODO understand how to remove the generated warning here
        hour_date = dates[decomposed_demand.date.dt.hour == hour]
        tmp_consumption = cold_dates.sel(date=hour_date) * demand_thermal_sensitive_factor[:, :, hour] * decomposed_demand.thermal_sensitive.sel(date=hour_date)
        res.loc[:, :, hour_date] =  tmp_consumption.transpose("energy_vector_out", "area_to","date")
    res=res+decomposed_demand.non_thermal_sensitive
    res = res.rename(new_name_or_name_dict="exogenous_energy_demand")
    #TODO there is certainly a better practice than having to rename things like this with datasets/dataarray
    return res.to_dataset()


def compute_flexible_demand_to_optimise(flexible_demand_table, demand_profile,exogenous_energy_demand,
                                        temperature):
    flexible_demand = list(flexible_demand_table.get_index("flexible_demand").unique())

    # TODO implement profile type depending on area_to and energy_vector_out
    unique_profile_types = np.unique(flexible_demand_table.flexible_demand_profile_type.to_numpy())
    profile_types = flexible_demand_table.flexible_demand_profile_type.to_dataframe()
    full_flex_demand_profile_per_type = {}
    for profile_type in unique_profile_types:
        current_profile = demand_profile[['hour', 'day_of_week', 'season', profile_type]]. \
            pivot(index=['hour', 'day_of_week'], columns='season', values=profile_type).to_xarray()
        decomposed_profile_demand = generate_demand_from_profile(current_profile, temperature,
                                                                 exogenous_energy_demand,
                                                                 minimum_temperature=0, temperature_threshold=15)
        flexible_demand_to_optimise = (
                decomposed_profile_demand.non_thermal_sensitive + decomposed_profile_demand.thermal_sensitive). \
            rename(new_name_or_name_dict="flexible_demand_to_optimise")
        full_flex_demand_profile_per_type[profile_type] = normalize(flexible_demand_to_optimise)

    profile_to_merge = []
    for flexible_demand_type in flexible_demand:
        profile_type = \
        flexible_demand_table.flexible_demand_profile_type.loc[:, :, flexible_demand_type].to_numpy().squeeze()[0]
        profile_to_merge.append(full_flex_demand_profile_per_type[profile_type]. \
                                expand_dims(dim={"flexible_demand": [flexible_demand_type]}, axis=1))
    profiles = xr.merge(profile_to_merge)

    flexible_demand_to_optimise = profiles * flexible_demand_table.flexible_demand_yearly_energy_twh
    return flexible_demand_to_optimise

def normalize(demand):
    for area in demand.get_index("area_to"):
        for energy_vector in demand.get_index("energy_vector_out"):
            total_energy = demand.loc[energy_vector,area,:].sum()
            demand.loc[energy_vector, area, :]=demand.loc[energy_vector,area,:]/total_energy
    return demand



def generate_demand_from_profile(profile,temperature,exogenous_energy_demand,
                                 temperature_threshold = 15,minimum_temperature = 0):
    hours = exogenous_energy_demand.date.dt.hour
    days_of_week = exogenous_energy_demand.date.dt.weekday
    # .drop_duplicates( ###  hours, daysofweek,
    temperature_and_demand = xr.merge([exogenous_energy_demand,  temperature])

    ### initialisation
    demand_thermal_sensitive =  exogenous_energy_demand.exogenous_energy_demand.copy()*0
    demand_thermal_sensitive= demand_thermal_sensitive.rename(
        new_name_or_name_dict="thermal_sensitive")
    demand_non_thermal_sensitive=  exogenous_energy_demand.exogenous_energy_demand.copy()*0
    demand_non_thermal_sensitive= demand_non_thermal_sensitive.rename(
        new_name_or_name_dict="non_thermal_sensitive")
    decomposed_demand = xr.merge([demand_non_thermal_sensitive,demand_thermal_sensitive])

    #estimation of thermal sensitivity as the difference between summer and winter, normalized
    thermal_sensitivity_estimate = xr.DataArray(
        data=0,
        dims=[ "energy_vector_out","area_to","hour"],
        coords=dict(
            hour=list(set(hours.to_numpy())),
            area_to=exogenous_energy_demand.get_index("area_to"),
            energy_vector_out=exogenous_energy_demand.get_index("energy_vector_out")
        )
    )
    thermal_sensitivity_estimate= thermal_sensitivity_estimate.rename(
        new_name_or_name_dict="thermal_sensitivity")
    profile_df = profile.to_dataframe()
    for area in temperature_and_demand.get_index("area_to"):
        for energy_vector in temperature_and_demand.get_index("energy_vector_out"):
            thermal_sensitivity_estimate.loc[energy_vector, area, :] = \
                ((profile_df.summer.groupby("hour").mean() - profile_df.winter.groupby("hour").mean()) / (
                        temperature_threshold - minimum_temperature)).tolist()

    cold_dates = temperature_and_demand.temperature <= temperature_threshold

    consumption_thermal_sensitive = temperature_and_demand.exogenous_energy_demand.copy()
    ## fill the thermal sensitive part
    for hour in list(set(hours.to_numpy())):
        hour_date = temperature_and_demand.get_index("date")[hours == hour]
        tmp_consumption = cold_dates.sel(date=hour_date) * thermal_sensitivity_estimate[:, :, hour] * (
                temperature_and_demand.temperature.sel(date=hour_date) - temperature_threshold)
        decomposed_demand.thermal_sensitive.loc[:, :, hour_date] = tmp_consumption.transpose("energy_vector_out",
                                                                                       "area_to",
                                                                                       "date")
    ## fill the non thermal sensitive part
    for hour in list(set(hours.to_numpy())):
        for day in list(set(days_of_week.to_numpy())):
            hour_day_date = temperature_and_demand.get_index("date")[(hours == hour)  & (days_of_week == day)]
            decomposed_demand.non_thermal_sensitive.loc[:, :, hour_day_date] = profile.summer[hour,day]

    return xr.merge([decomposed_demand,thermal_sensitivity_estimate])


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

def bisextile(year):
    if year%4==0:
        return 8784
    else:
        return 8760


def Profile2Consumption(Profile_df,temperature_df, temperatureThreshold=14,
                        temperatureMinimum=0,temperatureName='temperature',
                        ConsumptionName='Consumption',TimeName='date',
                        VarName='electrical_power_per_million_ev'):
    '''
    fonction permettant de reconstruire la consommation annuelle à partir d'un profil hourxdayxseason en une part thermosensible et non thermosensible
    (la conso non thermosensible étant la conso type d'une semaine d'été)

    :param Profile_df: profil avec les colonnes hourxdayxseason
    :param temperature_df:
    :param temperatureThreshold:
    :param temperatureMinimum:
    :param temperatureName:
    :param ConsumptionName:
    :param TimeName:
    :param VarName:
    :return:
    '''
    ## initialisation
    ConsoSepareeNew_df=temperature_df.loc[:,[temperatureName]]
    ConsoSepareeNew_df.loc[:,[ConsumptionName]]=np.NaN
    ConsoSepareeNew_df.loc[:,['NTS_C']]=0
    ConsoSepareeNew_df.loc[:,['TS_C']]=0

    PivotedProfile_df = Profile_df.pivot( index=['hour','day'], columns='season', values=VarName ).reset_index()
    cte=(temperatureThreshold-temperatureMinimum)

    for index, row in PivotedProfile_df.iterrows():
        indexesWD=ConsoSepareeNew_df.index.get_level_values(TimeName).to_series().dt.weekday   == (PivotedProfile_df.loc[index,'day']-1)
        indexesHours= ConsoSepareeNew_df.index.get_level_values(TimeName).to_series().dt.hour   == (PivotedProfile_df.loc[index,'hour'])
        ConsoSepareeNew_df.loc[indexesWD&indexesHours, 'NTS_C']=PivotedProfile_df.loc[index,'Ete']

    PivotedProfile_df['NDifference'] = (PivotedProfile_df['Ete'] - PivotedProfile_df['Hiver'])
    Thermosensibilite = (PivotedProfile_df['NDifference'].loc[0:23] / cte).tolist()
    ConsoSepareeNew_df=Recompose(ConsoSepareeNew_df,Thermosensibilite)
    return(ConsoSepareeNew_df)
