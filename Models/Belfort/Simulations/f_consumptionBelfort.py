import numpy as np
import pandas as pd
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

def Decomposeconso2(data_df, T0=15, T1=20, TemperatureName='Temperature',
                    ConsumptionName='Consumption', TimeName='Date'):
    '''
    Function decomposing the consumption into thermosensitive and non-thermosensitive part
    taking into account air condition in summer.

    Parameters
    ----------
    data_df : panda data frame with "Temperature" and "Consumption" as columns.
    T0 : float, optional
        Threshold temperature for heating in winter. The default is 15.
    T1 : TYPE, optional
        Threshold temperature for air condition in summer. The default is 20.
    TemperatureName : str, optional
        The default is 'Temperature'.
    ConsumptionName : str, optional
        The default is 'Consumption'.
    TimeName : str, optional
        The default is 'Date'.

    Returns
    -------
    a dictionary with Thermosensitivity_winter, Thermosensitivity_summer,
    and a panda data frame with two new columns NTS_C (non thermosensitive),
    TSW_C (thermosensitive winter), TSS_C (thermosensitive summer)
    '''

    dataNoNA_df = data_df.dropna()
    ## Remove NA
    ConsoSeparee_df = dataNoNA_df.assign(NTS_C=dataNoNA_df[ConsumptionName], TSW_C=dataNoNA_df[ConsumptionName] * 0,
                                         TSS_C=dataNoNA_df[ConsumptionName] * 0)

    Thermosensitivity_winter = {}
    Thermosensitivity_summer = {}
    # pd.DataFrame(data=np.zeros((24,1)), columns=['Thermosensibilite'])## one value per hour of the day
    # Thermosensibilite.index.name='hour of the day'
    for hour in range(24):
        indexesWinterHour = (dataNoNA_df[TemperatureName] <= T0) & (
                    dataNoNA_df.index.get_level_values(TimeName).to_series().dt.hour == hour)
        indexesSummerHour = (dataNoNA_df[TemperatureName] >= T1) & (
                    dataNoNA_df.index.get_level_values(TimeName).to_series().dt.hour == hour)
        dataWinterHour_df = dataNoNA_df.loc[indexesWinterHour, :]
        dataSummerHour_df = dataNoNA_df.loc[indexesSummerHour, :]
        lrw = linear_model.LinearRegression().fit(dataWinterHour_df[[TemperatureName]],
                                                  dataWinterHour_df[ConsumptionName])
        Thermosensitivity_winter[hour] = lrw.coef_[0]
        lrs = linear_model.LinearRegression().fit(dataSummerHour_df[[TemperatureName]],
                                                  dataSummerHour_df[ConsumptionName])
        Thermosensitivity_summer[hour] = lrs.coef_[0]
        ConsoSeparee_df.loc[indexesWinterHour, 'TSW_C'] = Thermosensitivity_winter[hour] * dataWinterHour_df.loc[:,
                                                                                           TemperatureName] - \
                                                          Thermosensitivity_winter[hour] * T0
        ConsoSeparee_df.loc[indexesWinterHour, 'NTS_C'] = dataWinterHour_df.loc[:, ConsumptionName] - \
                                                          ConsoSeparee_df.TSW_C.loc[indexesWinterHour]
        ConsoSeparee_df.loc[indexesSummerHour, 'TSS_C'] = Thermosensitivity_summer[hour] * dataSummerHour_df.loc[:,
                                                                                           TemperatureName] - \
                                                          Thermosensitivity_summer[hour] * T1
        ConsoSeparee_df.loc[indexesSummerHour, 'NTS_C'] = dataSummerHour_df.loc[:, ConsumptionName] - \
                                                          ConsoSeparee_df.TSS_C.loc[indexesSummerHour]

    return (ConsoSeparee_df[['NTS_C', 'TSW_C', 'TSS_C']], Thermosensitivity_winter, Thermosensitivity_summer)

def ComplexProfile2Consumption_2(Profile_df,
                                      Temperature_df,poidsName='poids',
                                      ConsumptionName='Consumption', TimeName='Date',GroupName='type'):
    '''
    Decomposing the consumption profile into several consumption uses.

    :param Profile_df: Weights to divide the consumption into several uses
    (according to hour, weekday and season winter/summer).
    :param Temperature_df: Temperature and consumption.
    :param poidsName: Name of the weights.
    :param ConsumptionName: Name of consumption.
    :param TimeName: Name of the time index.
    :param GroupName: Name indicating the type of electricity use.
    :return: A dataframe with consumption per use (in columns) throughout the year
    (same time index as Temperature_df).
    '''
    ## Processing dates indexing Temperature
    ConsoSepareeNew_df = Temperature_df.loc[:, [ConsumptionName]]
    ConsoSepareeNew_df = ConsoSepareeNew_df.assign(
        Jour=ConsoSepareeNew_df.index.get_level_values(TimeName).to_series().dt.weekday,
        Mois=ConsoSepareeNew_df.index.get_level_values(TimeName).to_series().dt.month,
        Heure=ConsoSepareeNew_df.index.get_level_values(TimeName).to_series().dt.hour)

    L_week = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]

    ConsoSepareeNew_df['Jour'] = ConsoSepareeNew_df['Jour']. \
            apply(lambda x: L_week[x])
    ConsoSepareeNew_df = ConsoSepareeNew_df.reset_index().set_index(["Mois", "Jour", "Heure"])

    ## Processing the profile to make every month appear
    Profile_df=Profile_df.reset_index()
    Profile_summer=Profile_df[Profile_df.Saison=="Ete"][["Jour","Heure",GroupName,poidsName]].reset_index()# Janvier
    Profile_winter= Profile_df[Profile_df.Saison == "Hiver"][["Jour","Heure",GroupName,poidsName]].reset_index()# Juin

    Profile_month=Profile_winter.copy().assign(Mois=1)
    for month in range(2,13):
        Profile_temp=Profile_winter.copy().assign(Mois=month)
        Profile_temp[poidsName]=Profile_winter[poidsName]*np.cos(np.pi*(month-1)/12)**2\
                                +(Profile_summer[poidsName]-Profile_winter[poidsName]*np.cos(np.pi*5/12)**2)\
                                *np.sin(np.pi*(month-1)/12)**2/np.sin(np.pi*5/12)**2
        Profile_month=pd.concat([Profile_month,Profile_temp],ignore_index=True)
    Profile_month=Profile_month.reset_index().set_index(["Mois","Jour","Heure"])


    Profile_month_merged = ConsoSepareeNew_df.join(Profile_month, how="right")
    Profile_month_merged.loc[:, [ConsumptionName]] = Profile_month_merged[ConsumptionName] * Profile_month_merged[poidsName]
    return Profile_month_merged.reset_index()[[ConsumptionName,TimeName,GroupName]].\
        groupby([TimeName,GroupName]).sum().reset_index().\
        pivot(index=TimeName, columns=GroupName, values=ConsumptionName)
    # cte=(TemperatureThreshold-TemperatureMinimum)

d_reindus={'reindus':' reindustrialisation','no_reindus':'','UNIDEN':' UNIDEN'}
def colReindus(col,reindus='reindus',industryName='Industrie hors metallurgie',steelName='Metallurgie'):
    '''
    Intermediary function to pick the right column name (for consumption data) given an industry hypothesis.

    :param col: name of column (idependently of industry hypothesis)
    :param reindus: industry hypothesis ('no_reindus','reindus' or 'UNIDEN')
    :param industryName:
    :param steelName:
    :return: right name for consumption data given industry hypothesis.
    '''
    if col in [industryName,steelName]:
        return col+d_reindus[reindus]
    else:
        return col

def ProjectionConsoNTS(Conso_profile_df,Projections_df,year,reindus='reindus',
                       industryName='Industrie hors metallurgie',steelName='Metallurgie'):
    '''
    Forecasts non-thermosensitive consumption by sectors given forecasting coefficient
    for future years.

    :param Conso_profile_df: Consumption by sectors (non-thermosensitive) at reference year (2019).
    :param Projections_df: Coefficients by sector.
    :param year: Year to forecast.
    :param reindus: industry hypothesis ('no_reindus','reindus' or 'UNIDEN').
    'no_reindus' if industry is simply maintained.
    'reindus' if France is reindustrialized.
    'UNIDEN' for UNIDEN hypothesis (reindustrialization + high electrification of industry).
    :param industryName:
    :param steelName:
    :param reindusName:
    :return: two dataframes giving the sum of forecasted consumptions and forecasted consumptions
    by sector (in that order).
    '''
    Conso_profile_new_df=Conso_profile_df.copy()
    L_cols=list(Conso_profile_df.columns)
    L_years=list(Projections_df.index)
    if year<=L_years[0]:
        for col in L_cols:
            col_proj=colReindus(col,reindus,industryName,steelName)
            Conso_profile_new_df[col]=Projections_df.loc[L_years[0],col_proj]*Conso_profile_new_df[col]
    elif year>=L_years[-1]:
        for col in L_cols:
            col_proj = colReindus(col, reindus, industryName, steelName)
            Conso_profile_new_df[col] = Projections_df.loc[L_years[-1], col_proj] * Conso_profile_new_df[col]
    else:
        i=0
        while i<len(L_years) and year>=L_years[i]:
            i+=1
        for col in L_cols:
            col_proj = colReindus(col, reindus, industryName, steelName)
            Conso_profile_new_df[col] = (Projections_df.loc[L_years[i-1], col_proj]+(year-L_years[i-1])/(L_years[i]-L_years[i-1])*(Projections_df.loc[L_years[i], col_proj]-Projections_df.loc[L_years[i-1], col_proj])) * Conso_profile_new_df[col]

    Conso_profile_new_df=Conso_profile_new_df.assign(Total=0)
    for col in L_cols:
        if col!=steelName:
            Conso_profile_new_df["Total"]+=Conso_profile_new_df[col]
    Conso_profile_new_df=Conso_profile_new_df.rename(columns={"Total":"Consommation hors metallurgie"})
    return Conso_profile_new_df[["Consommation hors metallurgie",steelName]],Conso_profile_new_df[L_cols]

def COP_air_eau(T,year,COP_coeffs_air_eau=[4.8e-4,7e-2,2.26],
                efficiency_gain=0.01):
    return (1+min(0.3,efficiency_gain*(year-2018)))*(T**2*COP_coeffs_air_eau[0]+T*COP_coeffs_air_eau[1]\
                                                     +COP_coeffs_air_eau[2])

def COP_air_air(T,year,COP_coeffs_air_air=[0.05,1.85],
                efficiency_gain=0.01):
    return (1+min(0.3,efficiency_gain*(year-2018)))*(T*COP_coeffs_air_air[0]+COP_coeffs_air_air[1])

def Factor_joule(T,T0=15):
    if T>T0:
        return 0
    else:
        return 1

def Factor_air_eau(T,year,COP_coeffs_air_eau=[4.8e-4,7e-2,2.26],
                efficiency_gain=0.01,T0=15):
    if T>T0:
        return 0
    else:
        return 1/COP_air_eau(T,year,COP_coeffs_air_eau,efficiency_gain)

def Factor_air_air(T,year,COP_coeffs_air_air=[0.05,1.85],
                efficiency_gain=0.01,T0=15):
    if T>T0:
        return 0
    else:
        return 1/COP_air_air(T,year,COP_coeffs_air_air,efficiency_gain)

def Factor_hybrid(T,year,COP_coeffs_air_eau=[4.8e-4,7e-2,2.26],
                efficiency_gain=0.01,T0=15,T_hybrid0=3,T_hybrid1=5):
    if T>T0 or T<T_hybrid0:
        return 0
    elif T>=T_hybrid1:
        return 1/COP_air_eau(T,year,COP_coeffs_air_eau,efficiency_gain)
    else:
        return (1-(T-T_hybrid1)/(T_hybrid0-T_hybrid1))/COP_air_eau(T,year,COP_coeffs_air_eau,efficiency_gain)

def ConsoHeat(Temperature_df,Thermosensitivity_df,
              Energy_houses_df,Energy_apartments_df,Energy_offices_df,Part_PAC_df,year,
              bati_hyp="ref",T0=15,T_hybrid0=3,T_hybrid1=5,
              COP_coeffs_air_eau=[4.8e-4,7e-2,2.26],COP_coeffs_air_air=[0.05,1.85],
              efficiency_gain=0.01,efficiency_rc=0.826,part_heat_rc=0.885,
              TemperatureName="Temperature",CUName="Chauffage urbain",
              JouleName="Chauffage électrique",PACaeName="Pompes à chaleur air-eau",
              PACaaName="Pompes à chaleur air-air",PAChName="Pompes à chaleur hybride",
              ThermoName="Thermosensibilite hiver (GW/degre)",HourName="Heure",TimeName="Date",
              PartPACName="Part PAC",
              year_ref=2021,year_end=2060):
    '''
    Computes heating consumption (thermosensitive).

    :param Temperature_df:
    :param Thermosensitivity_df:
    :param Energy_houses_df:
    :param Energy_apartments_df:
    :param Energy_offices_df:
    :param Part_PAC_df:
    :param year:
    :param bati_hyp:
    :param T0:
    :param T_hybrid0:
    :param T_hybrid1:
    :param COP_coeffs_air_eau:
    :param COP_coeffs_air_air:
    :param efficiency_gain:
    :param efficiency_rc:
    :param part_heat_rc:
    :param TemperatureName:
    :param CUName:
    :param JouleName:
    :param PACaeName:
    :param PACaaName:
    :param PAChName:
    :param ThermoName:
    :param HourName:
    :param TimeName:
    :param PartPACName:
    :param year_ref:
    :param year_end:
    :return:
    '''
    #Temperature_new_df=Temperature_df.rename(columns={TemperatureName:"Temperature"})
    Temperature_new_df = Temperature_df.assign(Factor_j=0,
                                                   Factor_ae=0,
                                                   Factor_aa=0,
                                                   Factor_h=0)

    Temperature_new_df["Factor_j"]=Temperature_new_df[TemperatureName].apply(lambda x: Factor_joule(x,T0))
    Temperature_new_df["Factor_ae"] = Temperature_new_df[TemperatureName].apply(
        lambda x: Factor_air_eau(x,year,COP_coeffs_air_eau,efficiency_gain,T0))
    Temperature_new_df["Factor_aa"] = Temperature_new_df[TemperatureName].apply(
        lambda x: Factor_air_air(x,year,COP_coeffs_air_air,efficiency_gain,T0))
    Temperature_new_df["Factor_h"] = Temperature_new_df[TemperatureName].apply(
        lambda x: Factor_hybrid(x,year,COP_coeffs_air_eau,efficiency_gain,T0,T_hybrid0,T_hybrid1))

    F_ae=Temperature_new_df.loc[(Temperature_new_df.Temperature<=T0),"Factor_ae"].mean()
    F_aa=Temperature_new_df.loc[(Temperature_new_df.Temperature <= T0), "Factor_aa"].mean()

    L_E0=[Energy_houses_df.loc[year_ref,name]+Energy_apartments_df.loc[year_ref,name]\
    +Energy_offices_df.loc[year_ref,name] for name in [JouleName,PACaeName,PACaaName,PAChName]]
    L_E0[-1]+=Part_PAC_df.loc[year_ref,PartPACName+" "+bati_hyp]/(efficiency_rc*part_heat_rc)* \
              (Energy_houses_df.loc[year_ref,CUName]+Energy_apartments_df.loc[year_ref,CUName]\
               +Energy_offices_df.loc[year_ref,CUName])

    if year<year_ref:
        year_ret=year_ref
    elif year>year_end:
        year_ret=year_end
    else:
        year_ret=year

    L_E_year=[Energy_houses_df.loc[year_ret,name]+Energy_apartments_df.loc[year_ret,name]\
    +Energy_offices_df.loc[year_ret,name] for name in [JouleName,PACaeName,PACaaName,PAChName]]
    L_E_year[-1]+=Part_PAC_df.loc[year_ret,PartPACName+" "+bati_hyp]/(efficiency_rc*part_heat_rc)* \
              (Energy_houses_df.loc[year_ret,CUName]+Energy_apartments_df.loc[year_ret,CUName]\
               +Energy_offices_df.loc[year_ret,CUName])

    denom=L_E0[0]+L_E0[1]*F_ae+L_E0[2]*F_aa+L_E0[3]*F_ae
    Thermosensitivity_new_df=Thermosensitivity_df.assign(Thermo_joule=L_E_year[0]/denom,
                                                         Thermo_ae=L_E_year[1]/denom,
                                                         Thermo_aa=L_E_year[2]/denom,
                                                         Thermo_h=L_E_year[3]/denom)

    for name in ["Thermo_joule","Thermo_ae","Thermo_aa","Thermo_h"]:
        Thermosensitivity_new_df[name]=Thermosensitivity_new_df[ThermoName]*Thermosensitivity_new_df[name]


    Thermosensitivity_new_df=Thermosensitivity_new_df.reset_index().rename(columns={HourName:"Heure"})
    Temperature_new_df=Temperature_new_df.assign(Heure=Temperature_new_df.index.get_level_values(TimeName).to_series().dt.hour,Conso_TS_heat=0)
    Temperature_new_df=pd.merge(Temperature_new_df.reset_index(),Thermosensitivity_new_df,on="Heure").set_index(TimeName).sort_index()

    Temperature_new_df["Conso_TS_heat"]=(Temperature_new_df[TemperatureName]-T0)*(Temperature_new_df["Thermo_joule"]*Temperature_new_df["Factor_j"]\
                                                                           +Temperature_new_df["Thermo_ae"]*Temperature_new_df["Factor_ae"]\
                                                                           +Temperature_new_df["Thermo_aa"]*Temperature_new_df["Factor_aa"]\
                                                                           +Temperature_new_df["Thermo_h"]*Temperature_new_df["Factor_h"])

    return Temperature_new_df[["Conso_TS_heat"]]

def ConsoAirCon(Temperature_df,Thermosensitivity_df,
              Energy_houses_df,Energy_apartments_df,Energy_offices_df,year,
              T1=20,taux_clim_res0=0.22,taux_clim_res1=0.55,
              taux_clim_ter0=0.3,taux_clim_ter1=0.55,
              TemperatureName="Temperature",ThermoName="Thermosensibilite ete (GW/degre)",
              HourName="Heure",TimeName="Date",year_ref=2021,year_clim1=2050,year_end=2060):
    '''
    Computes air condition consumption (thermosensitive).

    :param Temperature_df:
    :param Thermosensitivity_df:
    :param Energy_houses_df:
    :param Energy_apartments_df:
    :param Energy_offices_df:
    :param Part_PAC_df:
    :param year:
    :param bati_hyp:
    :param T1:
    :param taux_clim_res0: share of housing equiped with air-conditioning today (RTE "Futurs énergétiques 2050", chapter "Consommation")
    :param taux_clim_res1: share of housing equiped with air-conditioning in 2050 (RTE "Futurs énergétiques 2050", chapter "Consommation")
    :param taux_clim_ter0: share of offices equiped with air-conditioning today (RTE, "GT - consommation tertiaire")
    :param taux_clim_ter1: share of offices equiped with air-conditioning in 2050 (as housing)
    :param TemperatureName:
    :param CUName:
    :param JouleName:
    :param PACaeName:
    :param PACaaName:
    :param PAChName:
    :param ThermoName:
    :param HourName:
    :param TimeName:
    :param PartPACName:
    :param year_ref:
    :param year_clim1:
    :param year_end:
    :return: dataframe with air condition consumption
    '''

    L_cols=list(Energy_houses_df.columns)
    E0=taux_clim_res0*(sum([Energy_houses_df.loc[year_ref,col] for col in L_cols])\
        +sum([Energy_apartments_df.loc[year_ref,col] for col in L_cols]))\
        +taux_clim_ter0*sum([Energy_offices_df.loc[year_ref,col] for col in L_cols])

    if year<year_ref:
        year_ret=year_ref
    elif year>year_end:
        year_ret=year_end
    else:
        year_ret=year
    E_year=(taux_clim_res0+(year_ret-year_ref)/(year_clim1-year_ref)*(taux_clim_res1-taux_clim_res0))\
           *(sum([Energy_houses_df.loc[year_ref,col] for col in L_cols])\
        +sum([Energy_apartments_df.loc[year_ref,col] for col in L_cols]))\
        +(taux_clim_ter0+(year_ret-year_ref)/(year_clim1-year_ref)*(taux_clim_ter1-taux_clim_ter0))\
           *sum([Energy_offices_df.loc[year_ref,col] for col in L_cols])

    Thermosensitivity_new_df=Thermosensitivity_df.copy()
    Thermosensitivity_new_df[ThermoName]=E_year/E0*Thermosensitivity_new_df[ThermoName]

    Thermosensitivity_new_df = Thermosensitivity_new_df.reset_index().rename(columns={HourName: "Heure"})
    Temperature_new_df = Temperature_df.assign(
        Heure=Temperature_df.index.get_level_values(TimeName).to_series().dt.hour, Conso_TS_air_con=0)
    Temperature_new_df = pd.merge(Temperature_new_df.reset_index(), Thermosensitivity_new_df, on="Heure").set_index(
        TimeName).sort_index()
    Temperature_new_df["Conso_TS_air_con"]=Temperature_new_df[TemperatureName].apply(lambda T: T-T1 if T>=T1 else 0)*Temperature_new_df[ThermoName]

    return Temperature_new_df[["Conso_TS_air_con"]]

def Conso_ECS(Temperature_df,Profil_ECS_df,Projections_ECS_df,year,T2=20,TimeName="Date",
              ThermoName="Thermosensibilite (MW/degre)",ECSName="ECS a 20 degres",
              ECSCoeffName="Eau chaude sanitaire"):
    '''
    Forecast for hot water consumption.

    :param Profil_ECS_df: Hourly profile for one week at reference temperature with thermosensitivity.
    :param Projections_ECS_df:
    :param year:
    :param T2: Reference temperature of the curve Profil_ECS_df
    :return:
    '''

    Temperature_new_df = Temperature_df.assign(
        Jour=Temperature_df.index.get_level_values(TimeName).to_series().dt.weekday,
        Heure=Temperature_df.index.get_level_values(TimeName).to_series().dt.hour,
        Conso_ECS=0)

    L_week = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]

    Temperature_new_df['Jour'] = Temperature_new_df['Jour']. \
        apply(lambda x: L_week[x])
    Temperature_new_df = Temperature_new_df.reset_index().set_index(["Jour", "Heure"])

    Profil_ECS_new_df=Profil_ECS_df.reset_index().set_index(["Jour", "Heure"])
    Temperature_new_df=Temperature_new_df.join(Profil_ECS_new_df, how="right")

    L_years = list(Projections_ECS_df.index)
    if year <= L_years[0]:
        C=Projections_ECS_df.loc[L_years[0],ECSCoeffName]
    elif year >= L_years[-1]:
        C = Projections_ECS_df.loc[L_years[-1], ECSCoeffName]
    else:
        i = 0
        while i < len(L_years) and year >= L_years[i]:
            i += 1
        C= Projections_ECS_df.loc[L_years[i - 1],ECSCoeffName] + (year - L_years[i - 1]) / (L_years[i] - L_years[i - 1]) \
           * (Projections_ECS_df.loc[L_years[i],ECSCoeffName]- Projections_ECS_df.loc[L_years[i - 1],ECSCoeffName])

    Temperature_new_df["Conso_ECS"]=C*((Temperature_new_df["Temperature"]-T2)*Temperature_new_df[ThermoName]\
                                    +Temperature_new_df[ECSName])

    Temperature_new_df=Temperature_new_df.reset_index().set_index(TimeName).sort_index()
    return Temperature_new_df[["Conso_ECS"]]

def ConsoVE(Temperature_df,N_VP_df,N_VUL_df,N_PL_df,N_bus_df,N_car_df,Profil_VE_df,Params_VE_df,year,
            T0=15,T1=20,TemperatureName="Temperature",TimeName="Date",VLloadName="Puissance VL",
            PLloadName="Puissance PL",BusloadName="Puissance bus et car",VLThermoName="Thermosensibilite VL",
            PLThermoName="Thermosensibilite PL",BusThermoName="Thermosensibilite bus et car",
            ElName="Electrique",HybridName="Hybride rechargeable",H2Name="Hydrogène",
            ConsoElName="Consommation electrique (kWh/km)",ConsoHybridName="Consommation hybride rechargeable (kWh/km)",
            ConsoH2Name="Consommation hydrogene (kWh/km)",ProgressElName="Progres annuel electrique",
            ProgressHybridName="Progres annuel hybride rechargeable",ProgressH2Name="Progres annuel hydrogene",
            DistName="Kilometrage annuel",VPName="VP",VULName="VUL",PLName="PL",BusName="Bus",CarName="Car",
            year_ref=2020,year_end_progress=2050):
    '''
    Computes the consumption of electric vehicles (light and heavy) including hydrogen.

    :param Temperature_df:
    :param N_VP_df:
    :param N_VUL_df:
    :param N_PL_df:
    :param N_bus_df:
    :param N_car_df:
    :param Profil_VE_df:
    :param Params_VE_df:
    :param year:
    :param T0:
    :param T1:
    :param TemperatureName:
    :param TimeName:
    :param VLloadName:
    :param PLloadName:
    :param BusloadName:
    :param VLThermoName:
    :param PLThermoName:
    :param BusThermoName:
    :param ElName:
    :param HybridName:
    :param H2Name:
    :param ConsoElName:
    :param ConsoHybridName:
    :param ConsoH2Name:
    :param ProgressElName:
    :param ProgressHybridName:
    :param ProgressH2Name:
    :param DistName:
    :param VPName:
    :param VULName:
    :param PLName:
    :param BusName:
    :param CarName:
    :param year_ref:
    :param year_end_progress:
    :return: E_H2 in MWh, electric vehicle load Conso_VE in MW
    '''
    Temperature_new_df = Temperature_df.assign(
        Jour=Temperature_df.index.get_level_values(TimeName).to_series().dt.weekday,
        Heure=Temperature_df.index.get_level_values(TimeName).to_series().dt.hour,
        Delta_T_thermo=0)

    Temperature_new_df["Delta_T_thermo"]=Temperature_new_df[TemperatureName].apply(lambda T: T - T0 if T <= T0 else 0) \
        - Temperature_new_df[TemperatureName].apply(lambda T: T - T1 if T >= T1 else 0)

    L_week = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]

    Temperature_new_df['Jour'] = Temperature_new_df['Jour']. \
        apply(lambda x: L_week[x])
    Temperature_new_df = Temperature_new_df.reset_index().set_index(["Jour", "Heure"])

    Profil_VE_new_df = Profil_VE_df.reset_index().set_index(["Jour", "Heure"])
    Temperature_new_df = Temperature_new_df.join(Profil_VE_new_df, how="right")

    L_years = list(N_VP_df.index)
    if year <= L_years[0]:
        year_ret=L_years[0]
    elif year >= L_years[-1]:
        year_ret=L_years[-1]
    else:
        year_ret=year

    L_elec=[(1-Params_VE_df.loc[VPName,ProgressElName]*(min(year_ret,year_end_progress)-year_ref))\
            *Params_VE_df.loc[VPName,ConsoElName]*Params_VE_df.loc[VPName,DistName]*N_VP_df.loc[year_ret,ElName]\
            +(1-Params_VE_df.loc[VULName,ProgressElName]*(min(year_ret,year_end_progress)-year_ref))\
            *Params_VE_df.loc[VULName,ConsoElName]*Params_VE_df.loc[VULName,DistName]*N_VUL_df.loc[year_ret,ElName],
            (1 - Params_VE_df.loc[PLName, ProgressElName] * (min(year_ret, year_end_progress) - year_ref))\
            * Params_VE_df.loc[PLName, ConsoElName] * Params_VE_df.loc[PLName, DistName] * N_PL_df.loc[year_ret,ElName],
            (1 - Params_VE_df.loc[BusName, ProgressElName] * (min(year_ret, year_end_progress) - year_ref))\
            * Params_VE_df.loc[BusName, ConsoElName] * Params_VE_df.loc[BusName, DistName] * N_bus_df.loc[year_ret, ElName]\
            +(1 - Params_VE_df.loc[CarName, ProgressElName] * (min(year_ret, year_end_progress) - year_ref))\
            * Params_VE_df.loc[CarName, ConsoElName] * Params_VE_df.loc[CarName, DistName] * N_car_df.loc[year_ret, ElName]]
    # in kWh

    L_hybrid=[(1-Params_VE_df.loc[VPName,ProgressHybridName]*(min(year_ret,year_end_progress)-year_ref))\
            *Params_VE_df.loc[VPName,ConsoHybridName]*Params_VE_df.loc[VPName,DistName]*N_VP_df.loc[year_ret,HybridName]\
            +(1-Params_VE_df.loc[VULName,ProgressHybridName]*(min(year_ret,year_end_progress)-year_ref))\
            *Params_VE_df.loc[VULName,ConsoHybridName]*Params_VE_df.loc[VULName,DistName]*N_VUL_df.loc[year_ret,HybridName],
            (1 - Params_VE_df.loc[PLName, ProgressHybridName] * (min(year_ret, year_end_progress) - year_ref))\
            * Params_VE_df.loc[PLName, ConsoHybridName] * Params_VE_df.loc[PLName, DistName] * N_PL_df.loc[year_ret,HybridName],
            (1 - Params_VE_df.loc[BusName, ProgressHybridName] * (min(year_ret, year_end_progress) - year_ref))\
            * Params_VE_df.loc[BusName, ConsoHybridName] * Params_VE_df.loc[BusName, DistName] * N_bus_df.loc[year_ret, HybridName]\
            +(1 - Params_VE_df.loc[CarName, ProgressHybridName] * (min(year_ret, year_end_progress) - year_ref))\
            * Params_VE_df.loc[CarName, ConsoHybridName] * Params_VE_df.loc[CarName, DistName] * N_car_df.loc[year_ret, HybridName]]
    # in kWh

    E_H2=((1-Params_VE_df.loc[VPName,ProgressH2Name]*(min(year_ret,year_end_progress)-year_ref))\
            *Params_VE_df.loc[VPName,ConsoH2Name]*Params_VE_df.loc[VPName,DistName]*N_VP_df.loc[year_ret,H2Name]\
            +(1-Params_VE_df.loc[VULName,ProgressH2Name]*(min(year_ret,year_end_progress)-year_ref))\
            *Params_VE_df.loc[VULName,ConsoH2Name]*Params_VE_df.loc[VULName,DistName]*N_VUL_df.loc[year_ret,H2Name]\
            +(1 - Params_VE_df.loc[PLName, ProgressH2Name] * (min(year_ret, year_end_progress) - year_ref))\
            * Params_VE_df.loc[PLName, ConsoH2Name] * Params_VE_df.loc[PLName, DistName] * N_PL_df.loc[year_ret,H2Name]\
            +(1 - Params_VE_df.loc[BusName, ProgressH2Name] * (min(year_ret, year_end_progress) - year_ref))\
            * Params_VE_df.loc[BusName, ConsoH2Name] * Params_VE_df.loc[BusName, DistName] * N_bus_df.loc[year_ret, H2Name]\
            +(1 - Params_VE_df.loc[CarName, ProgressH2Name] * (min(year_ret, year_end_progress) - year_ref))\
            * Params_VE_df.loc[CarName, ConsoH2Name] * Params_VE_df.loc[CarName, DistName] * N_car_df.loc[year_ret, H2Name])/1e3
    # in MWh

    Temperature_new_df[VLloadName]+=Temperature_new_df["Delta_T_thermo"]*Temperature_new_df[VLThermoName]
    Temperature_new_df[PLloadName] += Temperature_new_df["Delta_T_thermo"] * Temperature_new_df[PLThermoName]
    Temperature_new_df[BusloadName] += Temperature_new_df["Delta_T_thermo"] * Temperature_new_df[BusThermoName]

    S_VL_load=Temperature_new_df[VLloadName].sum()
    S_PL_load=Temperature_new_df[PLloadName].sum()
    S_bus_car_load=Temperature_new_df[BusloadName].sum()

    Temperature_new_df[VLloadName]=Temperature_new_df[VLloadName]*(L_elec[0]+L_hybrid[0])/(S_VL_load*1e3)
    Temperature_new_df[PLloadName]=Temperature_new_df[PLloadName]*(L_elec[1]+L_hybrid[1])/(S_PL_load*1e3)
    Temperature_new_df[BusloadName] = Temperature_new_df[BusloadName] * (L_elec[2] + L_hybrid[2]) / (S_bus_car_load * 1e3)

    Temperature_new_df.assign(Conso_VE=0)
    Temperature_new_df["Conso_VE"]=Temperature_new_df[VLloadName]+Temperature_new_df[PLloadName]+Temperature_new_df[BusloadName]

    Temperature_new_df = Temperature_new_df.reset_index().set_index(TimeName).sort_index()
    return Temperature_new_df[["Conso_VE"]],E_H2

def ConsoH2(Conso_H2_df,year,reindus='reindus',
            refName='Reference',reindusName='Reindustrialisation',UNIDENName='UNIDEN'):
    '''
    Returns hydrogen consumption, vehicles not included.

    :param Conso_H2_df:
    :param year:
    :param reindus:
    :param refName:
    :param reindusName:
    :return: Result in MWh
    '''
    if reindus=='reindus':
        name=reindusName
    elif reindus=='no_reindus':
        name=refName
    else:
        name=UNIDENName

    L_years = list(Conso_H2_df.index)
    if year <= L_years[0]:
        E_H2=Conso_H2_df.loc[L_years[0], name]*1e6
    elif year >= L_years[-1]:
        E_H2=Conso_H2_df.loc[L_years[-1], name]*1e6
    else:
        i = 0
        while i < len(L_years) and year >= L_years[i]:
            i += 1
        E_H2 = (Conso_H2_df.loc[L_years[i-1], name]+(year-L_years[i-1])/(L_years[i]-L_years[i-1])* \
                (Conso_H2_df.loc[L_years[i], name]-Conso_H2_df.loc[L_years[i-1], name]))*1e6
    return E_H2

def Losses(Temperature_df,T_ref=20,taux_pertes=0.06927,rho_pertes=-1.2e-3,
           TemperatureName="Temperature"):
    '''
    Computes the losses (thermosensitive).

    :param Temperature_df:
    :param T_ref:
    :param taux_pertes:
    :param rho_pertes:
    :param TemperatureName:
    :return: Dataframe of losses in percent of the consumption.
    '''

    Temperature_new_df=Temperature_df.assign(Taux_pertes=taux_pertes)
    Temperature_new_df["Taux_pertes"]+=rho_pertes*(Temperature_new_df[TemperatureName]-T_ref)

    return Temperature_new_df[["Taux_pertes"]]

def CleanCETIndex(Temp_df,TimeName="Date"):
    '''
    To clean winter and summer hour index in Temperature (or other) dataframe.

    :param Temp_df:
    :param TimeName:
    :return:
    '''
    Temp_df_new=Temp_df.reset_index()
    Temp_df_new[TimeName]=Temp_df_new[TimeName].apply(lambda x: x.to_pydatetime())
    d1h=timedelta(hours=1)
    for i in Temp_df_new.index:
        if i>0 and Temp_df_new.loc[i,TimeName]-Temp_df_new.loc[i-1,TimeName]>d1h:
            i_start=i
        if i>0 and Temp_df_new.loc[i,TimeName]==Temp_df_new.loc[i-1,TimeName]:
            i_end=i
    for i in range(i_start,i_end):
        Temp_df_new.loc[i, TimeName]=Temp_df_new.loc[i, TimeName]-d1h
    Temp_df_new[TimeName]=Temp_df_new[TimeName].apply(lambda x: pd.Timestamp(x))
    Temp_df_new=Temp_df_new.set_index(TimeName)
    return Temp_df_new

def Project_consumption(NTS_profil_df,Projections_df,
                    Temp_df,Thermosensitivity_df,
                    Energy_houses_df, Energy_apartments_df,Energy_offices_df,Part_PAC_RCU_df,
                    Profil_ECS_df,Projections_ECS_df,
                    N_VP_df,N_VUL_df,N_PL_df,N_bus_df,N_car_df,
                    Profil_VE_df,Params_VE_df,
                    Conso_H2_df,
                    Losses_df,
                    year,
                    bati_hyp='ref',reindus='reindus',ev_hyp='',T0=15,T1=20):
    '''
    Function projecting consumption in the future following hypothesis in several sectors.
    Caution: some entries like Energy_houses_df, Energy_apartments_df, N_VP_df and N_VUL_df
    depend on the hypothesis (bati_hyp, ev_hyp).

    :param NTS_profil_df: Non-thermosensitive consumption by sector at reference year (2019).
    :param Projections_df: Projections coeffs by sector (columns) and year (rows).
    :param Temp_df: Temperature.
    :param Thermosensitivity_df:
    :param Energy_houses_df: Energy consumption of houses.
    :param Energy_apartments_df: Energy consumption of apartments.
    :param Energy_offices_df: Energy consumption of offices.
    :param Part_PAC_RCU_df: Part of heat pumps in heat networks.
    :param Profil_ECS_df: Profile of electric consumption for hot water boiling.
    :param Projections_ECS_df: Projection of concumption for hot water boiling.
    :param N_VP_df: Number of light passenger vehicles by energy source (electric, hydrogen, gas, fossil liquid...).
    :param N_VUL_df: Number of light duty vehicles by energy source (electric, hydrogen, gas, fossil liquid...).
    :param N_PL_df: Number of heavy trucks by energy source (electric, hydrogen, gas, fossil liquid...).
    :param N_bus_df: Number of city buses by energy source (electric, hydrogen, gas, fossil liquid...).
    :param N_car_df: Number of coaches by energy source (electric, hydrogen, gas, fossil liquid...).
    :param Profil_VE_df: Profile of electric consumption of vehicles.
    :param Params_VE_df: Parameters (km, energy consumption, technology).
    :param Conso_H2_df: Hydrogen consumption, road transport not included.
    :param Losses_df: Losses rate (thermosensitive).
    :param year:
    :param bati_hyp: Thermal renovation hypothesis ('ref' or 'SNBC').
    :param reindus: Industry hypothesis ('no_reindus', 'reindus' or 'UNIDEN').
    :param ev_hyp: Electric vehicules hypothesis ('' or 'fit_55').
    :param T0: Temperature when heating starts.
    :param T1: Temperature when air condition starts.
    :return:
    - A dataframe with non-flexible consumption agregated (Consommation hors metallurgie)
    and fexible consumptions (electric vehicles, H2, losses).
    - A dataframe with detailed consumptions.
    '''

    Conso_projected_df, Conso_detailed_df = ProjectionConsoNTS(NTS_profil_df, Projections_df, year, reindus)
    Conso_TS_heat_df = ConsoHeat(Temp_df, Thermosensitivity_df,
                                         Energy_houses_df, Energy_apartments_df, Energy_offices_df, Part_PAC_RCU_df,
                                         year, bati_hyp, T0)
    Conso_TS_air_con_df = ConsoAirCon(Temp_df, Thermosensitivity_df, Energy_houses_df,
                                              Energy_apartments_df, Energy_offices_df, year,T1)

    Conso_ECS_df = Conso_ECS(Temp_df, Profil_ECS_df, Projections_ECS_df, year)
    Conso_VE_df, E_H2 = ConsoVE(Temp_df, N_VP_df, N_VUL_df, N_PL_df, N_bus_df, N_car_df,
                                        Profil_VE_df, Params_VE_df, year)

    E_H2 += ConsoH2(Conso_H2_df, year, reindus)
    Conso_projected_df["Consommation hors metallurgie"] += Conso_TS_heat_df["Conso_TS_heat"] \
                                                               + Conso_TS_air_con_df["Conso_TS_air_con"] + Conso_ECS_df[
                                                                   "Conso_ECS"]

    Conso_projected_df["Conso_VE"] = Conso_VE_df["Conso_VE"]
    Conso_projected_df["Conso_H2"] = E_H2 / 8760
    Conso_projected_df["Taux_pertes"] = Losses_df["Taux_pertes"]

    Conso_detailed_df["Conso_TS_heat"] = Conso_TS_heat_df["Conso_TS_heat"]
    Conso_detailed_df["Conso_TS_air_con"] = Conso_TS_air_con_df["Conso_TS_air_con"]
    Conso_detailed_df["Conso_ECS"] = Conso_ECS_df["Conso_ECS"]
    Conso_detailed_df["Conso_VE"] = Conso_VE_df["Conso_VE"]
    Conso_detailed_df["Conso_H2"] = E_H2 / 8760
    Conso_detailed_df["Taux_pertes"] = Losses_df["Taux_pertes"]

    return Conso_projected_df,Conso_detailed_df