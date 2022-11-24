import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

def getTechnoPrice(tech,year):
    #capex en €/kW et opex en €/kW/an
    years=[2020,2030,2040,2050]
    capex={
        'WindOnShore': interp1d(years, [1300, 710, 620, 530],fill_value=(1300,530),bounds_error=False),
        'Solar': interp1d(years, [747, 557, 497, 427],fill_value=(747,427),bounds_error=False),
        'Boiler_elec': interp1d(years, [350]*4,fill_value=(350,350),bounds_error=False),
        'Boiler_gas': interp1d(years, [100]*4,fill_value=(100,100),bounds_error=False),
        'PAC': interp1d(years, [1000]*4,fill_value=(1000,1000),bounds_error=False),
        'Electrolysis_Alka':interp1d(years, [1313, 641, 574, 507],fill_value=(1313,440),bounds_error=False),
        'Reforming':interp1d(years, [800]*4,fill_value=(800,800),bounds_error=False),
        'Reforming+CCS':interp1d(years, [900,887,875,850],fill_value=(900,850),bounds_error=False)
    }
    opex={
        'WindOnShore': interp1d(years, [40, 22, 18, 16],fill_value=(40,16),bounds_error=False),
        'Solar': interp1d(years,  [11, 9, 8, 7],fill_value=(11,7),bounds_error=False),
        'Boiler_elec': interp1d(years,  [18]*4,fill_value=(18,18),bounds_error=False),
        'Boiler_gas': interp1d(years,  [5]*4,fill_value=(5,5),bounds_error=False),
        'PAC': interp1d(years,  [20]*4,fill_value=(20,20),bounds_error=False),
        'Electrolysis_Alka': interp1d(years, [15]*4,fill_value=(15,15),bounds_error=False),
        'Reforming': interp1d(years, [40]*4, fill_value=(40, 40), bounds_error=False),
        'Reforming+CCS':interp1d(years, [40]*4, fill_value=(40, 40), bounds_error=False)
    }
    return [capex[tech](year)*1000,opex[tech](year)*1000]

def Scenario_Heat(year,inputPath='Data/Raw_TP/'):


    #t = np.arange(1, nHours + 1)

    scenario = {}
    Prices = pd.read_csv(inputPath + 'resPrice_YEARxTIMExRES.csv', sep=',', decimal='.', skiprows=0,
                parse_dates=['Date']).set_index(["Date", "RESOURCES", "YEAR"])
    Date = Prices.reset_index().Date.unique()
    demand=pd.read_csv(inputPath+'heatDemand_TIME.csv',parse_dates=['Date']).set_index(['Date'])
    nHours = len(Date)

    scenario['resourceDemand'] =pd.DataFrame(data = {
                'Date': Date, # We add the TIMESTAMP so that it can be used as an index later.
                'heat': np.array(demand['areaConsumption']), # incrising demand of electricity (hypothesis : ADEME)
                'electricity': np.zeros(nHours),
                'gas': np.zeros(nHours),
            })

    scenario['conversionTechs'] = []

    tech = "WindOnShore"
    scenario['conversionTechs'].append(
        pd.DataFrame(data={tech:
                                { 'Category': 'Electricity production',
                                 'lifeSpan': 30, 'powerCost': 0, 'investCost': getTechnoPrice(tech,year)[0], 'operationCost': getTechnoPrice(tech,year)[1],
                                 'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                                 'EnergyNbhourCap': 0, # used for hydroelectricity
                                 'minCapacity':0 ,'maxCapacity':1000 }
                            }
                         )
        )


    tech = "Solar"
    scenario['conversionTechs'].append(
        pd.DataFrame(data={tech:
                               {'Category': 'Electricity production',
                                'lifeSpan': 20, 'powerCost': 0, 'investCost': getTechnoPrice(tech, year)[0],
                                'operationCost': getTechnoPrice(tech, year)[1],
                                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                                'EnergyNbhourCap': 0,  # used for hydroelectricity
                                'minCapacity': 0, 'maxCapacity': 1000}
                           }
                     )
    )

    tech = "Boiler_elec"
    scenario['conversionTechs'].append(
        pd.DataFrame(data={tech:
                               {'Category': 'Heat production',
                                'lifeSpan': 50, 'powerCost': 0, 'investCost': getTechnoPrice(tech, year)[0],
                                'operationCost': getTechnoPrice(tech, year)[1],
                                'EmissionCO2': 0, 'Conversion': {'heat': 1,'electricity':-1},
                                'EnergyNbhourCap': 0,  # used for hydroelectricity
                                'minCapacity': 0, 'maxCapacity': 1000}
                           }
                     )
    )

    tech = "Boiler_gas"
    scenario['conversionTechs'].append(
        pd.DataFrame(data={tech:
                               {'Category': 'Heat production',
                                'lifeSpan': 50, 'powerCost': 0, 'investCost': getTechnoPrice(tech, year)[0],
                                'operationCost': getTechnoPrice(tech, year)[1],
                                'EmissionCO2': 0, 'Conversion': {'heat': 1,'gas':-1},
                                'EnergyNbhourCap': 0,  # used for hydroelectricity
                                'minCapacity': 0, 'maxCapacity': 1000}
                           }
                     )
    )

    tech = "PAC"
    scenario['conversionTechs'].append(
        pd.DataFrame(data={tech:
                               {'Category': 'Heat production',
                                'lifeSpan': 20, 'powerCost': 0, 'investCost': getTechnoPrice(tech, year)[0],
                                'operationCost': getTechnoPrice(tech, year)[1],
                                'EmissionCO2': 0, 'Conversion': {'heat': 1,'electricity':-0.5},
                                'EnergyNbhourCap': 0,  # used for hydroelectricity
                                'minCapacity': 0, 'maxCapacity': 1000}
                           }
                     )
    )

    tech = "curtailment"
    scenario['conversionTechs'].append(
        pd.DataFrame(data={tech:
                               {'Category': 'Heat production',
                                'lifeSpan': 100, 'powerCost': 3000, 'investCost': 0,
                                'operationCost': 0,
                                'EmissionCO2': 0, 'Conversion': {'heat': 1},
                                'EnergyNbhourCap': 0,  # used for hydroelectricity
                                'minCapacity': 0, 'maxCapacity': 1000}
                           }
                     )
    )

    scenario['conversionTechs'] = pd.concat(scenario['conversionTechs'], axis=1)

    scenario['storageTechs'] = []

    tech = "Tank"
    scenario['storageTechs'].append(
        pd.DataFrame(data={tech:
                               { 'resource': 'heat',
                                 'storagelifeSpan': 30,
                                 'storagePowerCost': 1000,
                                 'storageEnergyCost':10000,
                                 'storageOperationCost': 10,
                                 'p_max': 1000,
                                 'c_max':10000,
                                 'chargeFactors': {'heat': 1},
                                 'dischargeFactors': {'heat': 1},
                                 'dissipation': 0.001,
                                 }
                           }
                     )
    )

    tech = "battery"
    scenario['storageTechs'].append(
        pd.DataFrame(data={tech:
                               { 'resource': 'electricity',
                                 'storagelifeSpan': 15,
                                 'storagePowerCost': 1000,
                                 'storageEnergyCost':10000,
                                 'storageOperationCost': 10,
                                 'p_max': 1000,
                                 'c_max':10000,
                                 'chargeFactors': {'electricity': 1},
                                 'dischargeFactors': {'electricity': 1},
                                 'dissipation': 0.0085,
                                 }
                           }
                     )
    )

    scenario['storageTechs'] = pd.concat(scenario['storageTechs'], axis=1)

    scenario['carbonTax'] = 0.13

    scenario['carbonGoals'] = 500

    scenario['maxBiogasCap'] = 1000

    scenario['gridConnection'] = pd.read_csv(inputPath+'CalendrierHPHC_TIME.csv', sep=',', decimal='.',
                                             skiprows=0,comment="#",parse_dates = ['Date']).set_index(["Date"])

    scenario['economicParameters'] = pd.DataFrame({
        'discountRate':[0.04],
        'financeRate': [0.04]
    }
    )
    df_res_ref = pd.read_csv(inputPath + 'resPrice_YEARxTIMExRES.csv', sep=',', decimal='.', skiprows=0,
                parse_dates=['Date']).set_index(["YEAR","Date", "RESOURCES"])
    #df_res_ref = pd.read_csv(inputPath+'resPrice_YEARxTIMExRES.csv', sep=',', decimal='.', skiprows=0,comment="#").set_index(["YEAR", "TIMESTAMP",'RESOURCES'])
    gasBioPrice=interp1d([2020,2030,2040,2050], [150, 120, 100, 80],fill_value=(150,80),bounds_error=False)

    scenario['resourceImportPrices'] =pd.DataFrame(data={
        'Date': Date,
        'electricity': df_res_ref.loc[(year, slice(None), 'electricity'),'importCost'].values,
        'gasNatural': df_res_ref.loc[(year, slice(None), 'gasNatural'),'importCost'].values,
        'gasBio': gasBioPrice(year)*1,
        'heat': 1000000,
        'gas': 1000000
    })

    scenario['resourceImportCO2eq'] =pd.DataFrame(data={
        'Date': Date,
        'electricity': df_res_ref.loc[(year, slice(None), 'electricity'),'emission'].values,
        'gasNatural': max(0, 0.03 * 29 / 13.1 + 203.5), # Taking 100 yr GWP of methane and 3% losses due to upstream leaks. Losses drop to zero in 2050.
        'gasBio': max(0, 0.03 * 29 / 13.1),
        'heat': 0 * np.ones(nHours),
        'gas': max(0, 0.03 * 29 / 13.1 + 203.5) # Taking 100 yr GWP of methane and 3% losses due to upstream leaks. Losses drop to zero in 2050.
    })

    scenario['convTechList'] = ["WindOnShore",  "Solar", "PAC", 'Boiler_elec','Boiler_gas','curtailment']
    ctechs = scenario['convTechList']
    availabilityFactor = pd.read_csv(inputPath+'availabilityFactorTIMExTECH.csv',sep=',', decimal='.',
                        skiprows=0, parse_dates = ['Date']).set_index([ "Date", "TECHNOLOGIES"])
    itechs = availabilityFactor.index.isin(ctechs,level='TECHNOLOGIES')
    scenario['availability'] = availabilityFactor.loc[(itechs,slice(None))]

    return scenario

def Scenario_H2(year,inputPath='Data/Raw_TP/'):

    scenario={}
    Prices = pd.read_csv(inputPath + 'resPrice_YEARxTIMExRES.csv', sep=',', decimal='.', skiprows=0,
                parse_dates=['Date']).set_index(["Date", "RESOURCES", "YEAR"])
    Date = Prices.reset_index().Date.unique()
    demand=pd.read_csv(inputPath+'H2Demand_TIME.csv',parse_dates=['Date']).set_index(['Date'])
    nHours = len(Date)


    scenario['resourceDemand'] =pd.DataFrame(data = {
                'Date': Date, # We add the TIMESTAMP so that it can be used as an index later.
                'hydrogen': np.array(demand['areaConsumption']), # incrising demand of electricity (hypothesis : ADEME)
                'electricity': np.zeros(nHours),
                'gas': np.zeros(nHours),
            })

    scenario['conversionTechs'] = []

    tech = "WindOnShore"
    scenario['conversionTechs'].append(
        pd.DataFrame(data={tech:
                                { 'Category': 'Electricity production',
                                 'lifeSpan': 30, 'powerCost': 0, 'investCost': getTechnoPrice(tech,year)[0], 'operationCost': getTechnoPrice(tech,year)[1],
                                 'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                                 'EnergyNbhourCap': 0, # used for hydroelectricity
                                 'minCapacity':0 ,'maxCapacity':1000 }
                            }
                         )
        )


    tech = "Solar"
    scenario['conversionTechs'].append(
        pd.DataFrame(data={tech:
                               {'Category': 'Electricity production',
                                'lifeSpan': 20, 'powerCost': 0, 'investCost': getTechnoPrice(tech, year)[0],
                                'operationCost': getTechnoPrice(tech, year)[1],
                                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                                'EnergyNbhourCap': 0,  # used for hydroelectricity
                                'minCapacity': 0, 'maxCapacity': 1000}
                           }
                     )
    )

    tech = "Reforming"
    scenario['conversionTechs'].append(
        pd.DataFrame(data={tech:
                               {'Category': 'H2 production',
                                'lifeSpan': 40, 'powerCost': 0, 'investCost': getTechnoPrice(tech, year)[0],
                                'operationCost': getTechnoPrice(tech, year)[1],
                                'EmissionCO2': -169, 'Conversion': {'hydrogen': 1,'gas':-1.43},
                                'EnergyNbhourCap': 0,  # used for hydroelectricity
                                'minCapacity': 0, 'maxCapacity': 1000,'RampConstraintPlus':0.3,'RampConstraintMoins':0.3}
                           }
                     )
    )

    tech = "Reforming+CCS"
    scenario['conversionTechs'].append(
        pd.DataFrame(data={tech:
                               {'Category': 'H2 production',
                                'lifeSpan': 40, 'powerCost': 0, 'investCost': getTechnoPrice(tech, year)[0],
                                'operationCost': getTechnoPrice(tech, year)[1],
                                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'gas':-1.43,'electricity':-0.17},
                                'EnergyNbhourCap': 0,  # used for hydroelectricity
                                'minCapacity': 0, 'maxCapacity': 1000,'RampConstraintPlus':0.3,'RampConstraintMoins':0.3}
                           }
                     )
    )

    tech = "Electrolysis_Alka"
    scenario['conversionTechs'].append(
        pd.DataFrame(data={tech:
                               {'Category': 'H2 production',
                                'lifeSpan': 20, 'powerCost': 0, 'investCost': getTechnoPrice(tech, year)[0],
                                'operationCost': getTechnoPrice(tech, year)[1],
                                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'electricity':-1.54},
                                'EnergyNbhourCap': 0,  # used for hydroelectricity
                                'minCapacity': 0, 'maxCapacity': 1000}
                           }
                     )
    )

    tech = "curtailment"
    scenario['conversionTechs'].append(
        pd.DataFrame(data={tech:
                               {'Category': 'H2 production',
                                'lifeSpan': 100, 'powerCost': 3000, 'investCost': 0,
                                'operationCost': 0,
                                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1},
                                'EnergyNbhourCap': 0,  # used for hydroelectricity
                                'minCapacity': 0, 'maxCapacity': 1000}
                           }
                     )
    )

    scenario['conversionTechs'] = pd.concat(scenario['conversionTechs'], axis=1)

    scenario['storageTechs'] = []

    tech = "TankH2"
    scenario['storageTechs'].append(
        pd.DataFrame(data={tech:
                               { 'resource': 'hydrogen',
                                 'storagelifeSpan':50,
                                 'storagePowerCost': 10000,
                                 'storageEnergyCost':8000,
                                 'storageOperationCost': 650,
                                 'p_max': 1000,
                                 'c_max':3000000,
                                 'chargeFactors': {'hydrogen': 1,'electricity':0.0168},
                                 'dischargeFactors': {'hydrogen': 1},
                                 'dissipation': 0,
                                 }
                           }
                     )
    )

    tech = "battery"
    scenario['storageTechs'].append(
        pd.DataFrame(data={tech:
                               { 'resource': 'electricity',
                                 'storagelifeSpan': 15,
                                 'storagePowerCost': interp1d([2020,2030,2040,2050], [220, 220, 175, 160],fill_value=(220,160),bounds_error=False)(year)*1000,
                                 'storageEnergyCost':interp1d([2020,2030,2040,2050], [300,300, 230, 175],fill_value=(300,175),bounds_error=False)(year)*1000,
                                 'storageOperationCost': 11000,
                                 'p_max': 1000,
                                 'c_max':10000,
                                 'chargeFactors': {'electricity': 0.92},
                                 'dischargeFactors': {'electricity': 1.09},
                                 'dissipation': 0.0085,
                                 }
                           }
                     )
    )

    scenario['storageTechs'] = pd.concat(scenario['storageTechs'], axis=1)

    scenario['carbonTax'] = 0.13

    scenario['carbonGoals'] = 500

    scenario['maxBiogasCap'] = 1000

    scenario['gridConnection'] = pd.read_csv(inputPath+'CalendrierHPHC_TIME.csv', sep=',', decimal='.',
                                             skiprows=0,comment="#",parse_dates = ['Date']).set_index(["Date"])
    scenario['economicParameters'] = pd.DataFrame({
        'discountRate':[0.04],
        'financeRate': [0.04]
    }
    )

    df_res_ref = pd.read_csv(inputPath + 'resPrice_YEARxTIMExRES.csv', sep=',', decimal='.', skiprows=0,
                parse_dates=['Date']).set_index(["YEAR","Date", "RESOURCES"])
    gasBioPrice=interp1d([2020,2030,2040,2050], [150, 120, 100, 80],fill_value=(150,80),bounds_error=False)

    scenario['resourceImportPrices'] =pd.DataFrame(data={
        'Date': Date,
        'electricity': df_res_ref.loc[(year, slice(None), 'electricity'),'importCost'].values,
        'gasNatural': df_res_ref.loc[(year, slice(None), 'gasNatural'),'importCost'].values,
        'gasBio': gasBioPrice(year)*1,
        'hydrogen': 1000000,
        'gas': 1000000
    })

    scenario['resourceImportCO2eq'] =pd.DataFrame(data={
        'Date': Date,
        'electricity': df_res_ref.loc[(year, slice(None), 'electricity'),'emission'].values,
        'gasNatural': max(0, 0.03 * 29 / 13.1 + 203.5), # Taking 100 yr GWP of methane and 3% losses due to upstream leaks. Losses drop to zero in 2050.
        'gasBio': max(0, 0.03 * 29 / 13.1),
        'hydrogen': 0 * np.ones(nHours),
        'gas': max(0, 0.03 * 29 / 13.1 + 203.5) # Taking 100 yr GWP of methane and 3% losses due to upstream leaks. Losses drop to zero in 2050.
    })

    scenario['convTechList'] = ["WindOnShore",  "Solar", "Electrolysis_Alka", 'Reforming','Reforming+CCS','curtailment']
    ctechs = scenario['convTechList']
    availabilityFactor = pd.read_csv(inputPath + 'availabilityFactorTIMExTECH.csv', sep=',', decimal='.',
                                     skiprows=0, parse_dates=['Date']).set_index(["Date", "TECHNOLOGIES"])
    itechs = availabilityFactor.index.isin(ctechs,level='TECHNOLOGIES')
    scenario['availability'] = availabilityFactor.loc[(itechs,slice(None))]

    return scenario