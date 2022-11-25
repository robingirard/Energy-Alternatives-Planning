from pyomo.environ import *
from pyomo.core import *
#import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import timedelta

def loadScenario(scenario, printTables=False):

    areaConsumption = scenario['resourceDemand'].melt(id_vars=['Date'], var_name=['RESOURCES'],value_name='areaConsumption').set_index(['Date', 'RESOURCES'])

    TechParameters = scenario['conversionTechs'].transpose().fillna(0)
    TechParameters.index.name = 'TECHNOLOGIES'
    TechParametersList = ['powerCost', 'operationCost', 'investCost', 'EnergyNbhourCap', 'RampConstraintPlus', 'RampConstraintMoins', 'EmissionCO2', 'minCapacity', 'maxCapacity']
    for k in TechParametersList:
        if k not in TechParameters:
            TechParameters[k] = 0
    TechParameters.drop(columns=['Conversion', 'Category'], inplace=True)

    StorageParameters = scenario['storageTechs'].transpose().fillna(0)
    StorageParameters.index.name = 'STOCK_TECHNO'
    StorageParametersList = ['resource', 'storagePowerCost', 'storageEnergyCost', 'p_max', 'c_max']
    for k in StorageParametersList:
        if k not in StorageParameters:
            StorageParameters[k] = 0
    StorageParameters.drop(columns=['chargeFactors', 'dischargeFactors', 'dissipation'], inplace=True)

    CarbonTax = scenario['carbonTax']

    df_conv = scenario['conversionTechs'].transpose()['Conversion']
    conversionFactor = pd.DataFrame( data={tech: df_conv.loc[tech] for tech in scenario['conversionTechs'].columns}).fillna(0)
    conversionFactor.index.name = 'RESOURCES'
    conversionFactor = conversionFactor.reset_index('RESOURCES').melt(id_vars=['RESOURCES'], var_name='TECHNOLOGIES',value_name='conversionFactor').set_index(['RESOURCES', 'TECHNOLOGIES'])

    df_sconv = scenario['storageTechs'].transpose()
    stechSet = set([k for k in df_sconv.index.values])
    df = {}
    for k1, k2 in (('charge', 'In'), ('discharge', 'Out')):
        df[k1] = pd.DataFrame(data={tech: df_sconv.loc[tech, k1 + 'Factors'] for tech in stechSet}).fillna(
            0)
        df[k1].index.name = 'RESOURCES'
        df[k1] = df[k1].reset_index(['RESOURCES']).melt(id_vars=['RESOURCES'], var_name='TECHNOLOGIES',value_name='storageFactor' + k2)

    df['dissipation'] = pd.DataFrame.from_dict(
        data={'dissipation': [df_sconv.loc[tech]['dissipation'] for tech in stechSet],
              'RESOURCES': [df_sconv.loc[tech]['resource'] for tech in stechSet],
              'TECHNOLOGIES': [tech for tech in stechSet]})
    storageFactors = pd.merge(df['charge'], df['discharge'], how='outer').fillna(0)
    storageFactors = pd.merge(storageFactors, df['dissipation'], how='outer').fillna(0).set_index(
        ['RESOURCES', 'TECHNOLOGIES'])

    Calendrier = scenario['gridConnection']
    Economics = scenario['economicParameters'].melt(var_name='Eco').set_index('Eco')

    ResParameters = pd.concat((
        k.melt(id_vars=['Date'], var_name=['RESOURCES'], value_name=name).set_index(
            ['Date', 'RESOURCES'])
        for k, name in [(scenario['resourceImportPrices'], 'importCost'), (scenario['resourceImportCO2eq'], 'emission')]
    ), axis=1)

    availabilityFactor = scenario['availability']

    # Return hydrogen annual consumption in kt
    if printTables:
        print(areaConsumption.loc[slice(None), 'electricity'].sum() / 33e3)
        print(TechParameters)
        print(CarbonTax)
        print(conversionFactor)
        print(StorageParameters)
        print(storageFactors)
        print(ResParameters)
        print(availabilityFactor)

    inputDict = scenario.copy()
    inputDict["areaConsumption"] = areaConsumption
    inputDict["availabilityFactor"] = availabilityFactor
    inputDict["techParameters"] = TechParameters
    inputDict["resParameters"] = ResParameters
    inputDict["conversionFactor"] = conversionFactor
    inputDict["economics"] = Economics
    inputDict["calendar"] = Calendrier
    inputDict["storageParameters"] = StorageParameters
    inputDict["storageFactors"] = storageFactors
    inputDict["carbonTax"] = CarbonTax

    return inputDict

def systemModel_MultiResource_WithStorage(Parameters):
    """
    This function creates the pyomo model and initlize the Parameters and (pyomo) Set values
    :param areaConsumption: panda table with consumption
    :param availabilityFactor: panda table
    :param isAbstract: boolean true is the model should be abstract. ConcreteModel otherwise
    :return: pyomo model
    """


    areaConsumption = Parameters["areaConsumption"]
    availabilityFactor = Parameters["availabilityFactor"]
    TechParameters = Parameters["techParameters"]
    ResParameters = Parameters["resParameters"]
    conversionFactor = Parameters["conversionFactor"]
    Economics = Parameters["economics"]
    Calendrier = Parameters["calendar"]
    StorageParameters = Parameters["storageParameters"]
    storageFactors = Parameters["storageFactors"]
    CarbonTax = Parameters["carbonTax"]
    carbonGoals = Parameters["carbonGoals"]
    gasBio_max = Parameters["maxBiogasCap"]

    isAbstract = False
    availabilityFactor.isna().sum()

    ### Cleaning
    availabilityFactor = availabilityFactor.fillna(method='pad');
    areaConsumption = areaConsumption.fillna(method='pad');
    ResParameters = ResParameters.fillna(0);

    ### obtaining dimensions values
    TECHNOLOGIES = set(TechParameters.index.get_level_values('TECHNOLOGIES').unique())
    STOCK_TECHNO = set(StorageParameters.index.get_level_values('STOCK_TECHNO').unique())
    RESOURCES = set(ResParameters.index.get_level_values('RESOURCES').unique())
    Date = set(areaConsumption.index.get_level_values('Date').unique())

    Date_list = areaConsumption.index.get_level_values('Date').unique()
    Last_date = Date_list[len(Date_list)-1]
    HORAIRE = {'P', 'HPH', 'HCH', 'HPE', 'HCE'}
    # Subsets
    Date_HCH = set(Calendrier[Calendrier['Calendrier'] == 'HCH'].index.get_level_values('Date').unique())
    Date_HPH = set(Calendrier[Calendrier['Calendrier'] == 'HPH'].index.get_level_values('Date').unique())
    Date_HCE = set(Calendrier[Calendrier['Calendrier'] == 'HCE'].index.get_level_values('Date').unique())
    Date_HPE = set(Calendrier[Calendrier['Calendrier'] == 'HPE'].index.get_level_values('Date').unique())
    Date_P = set(Calendrier[Calendrier['Calendrier'] == 'P'].index.get_level_values('Date').unique())

    #####################
    #    Pyomo model    #
    #####################

    if (isAbstract):
        model = pyomo.environ.AbstractModel()
    else:
        model = pyomo.environ.ConcreteModel()

    ###############
    # Sets       ##
    ###############
    model.TECHNOLOGIES = Set(initialize=TECHNOLOGIES, ordered=False)
    model.STOCK_TECHNO = Set(initialize=STOCK_TECHNO, ordered=False)
    model.RESOURCES = Set(initialize=RESOURCES, ordered=False)
    model.Date = Set(initialize=Date, ordered=False)
    model.HORAIRE = Set(initialize=HORAIRE, ordered=False)
    model.Date_TECHNOLOGIES = model.Date * model.TECHNOLOGIES
    model.Date_STOCKTECHNO = model.Date * model.STOCK_TECHNO
    model.RESOURCES_TECHNOLOGIES = model.RESOURCES * model.TECHNOLOGIES
    model.RESOURCES_STOCKTECHNO = model.RESOURCES * model.STOCK_TECHNO
    model.Date_RESOURCES =model.Date * model.RESOURCES

    # Subset of Simple only required if ramp constraint
    model.Date_MinusOne = Set(initialize=Date_list[: len(Date) - 1], ordered=False)
    model.Date_MinusThree = Set(initialize=Date_list[: len(Date) - 3], ordered=False)

    ###############
    # Parameters ##
    ###############
    model.areaConsumption = Param(model.Date_RESOURCES, default=0,
                                  initialize=areaConsumption.loc[:, "areaConsumption"].squeeze().to_dict(),
                                  domain=Reals)
    model.availabilityFactor = Param(model.Date_TECHNOLOGIES, domain=PercentFraction, default=1,
                                     initialize=availabilityFactor.loc[:, "availabilityFactor"].squeeze().to_dict())
    model.conversionFactor = Param(model.RESOURCES_TECHNOLOGIES, default=0,
                                   initialize=conversionFactor.loc[:, "conversionFactor"].squeeze().to_dict())

    gasTypes = ['gasNatural','gasBio']

    # with test of existing columns on TechParameters

    for COLNAME in TechParameters:
        if COLNAME not in ["TECHNOLOGIES", "AREAS", "YEAR"]:  ### each column in TechParameters will be a parameter
            exec("model." + COLNAME + "= Param(model.TECHNOLOGIES, default=0, domain=Reals," +
                 "initialize=TechParameters." + COLNAME + ".squeeze().to_dict())")

    for COLNAME in ResParameters:
        if COLNAME not in ["TECHNOLOGIES", "AREAS", "YEAR"]:  ### each column in TechParameters will be a parameter
            exec("model." + COLNAME + "= Param(model.Date_RESOURCES, domain=NonNegativeReals,default=0," +
                 "initialize=ResParameters." + COLNAME + ".squeeze().to_dict())")

    for COLNAME in Calendrier:
        if COLNAME not in ["Date", "AREAS"]:
            exec("model." + COLNAME + " = Param(model.Date, default=0," +
                 "initialize=Calendrier." + COLNAME + ".squeeze().to_dict(),domain=Any)")

    for COLNAME in StorageParameters:
        if COLNAME not in ["STOCK_TECHNO", "AREAS", "YEAR"]:  ### each column in StorageParameters will be a parameter
            exec("model." + COLNAME + " =Param(model.STOCK_TECHNO,domain=Any,default=0," +
                 "initialize=StorageParameters." + COLNAME + ".squeeze().to_dict())")

    for COLNAME in storageFactors:
        if COLNAME not in ["TECHNOLOGIES", "RESOURCES"]:
            exec("model." + COLNAME + " =Param(model.RESOURCES_STOCKTECHNO,domain=NonNegativeReals,default=0," +
                 "initialize=storageFactors." + COLNAME + ".squeeze().to_dict())")

    ################
    # Variables    #
    ################

    # In this section, variables are separated in two categories : decision variables wich are the reals variables of the otimisation problem (these are noted Dvar), and problem variables which are resulting of calculation and are convenient for the readability and the analyse of results (these are noted Pvar)

    # Operation
    model.power_Dvar = Var(model.Date, model.TECHNOLOGIES,domain=NonNegativeReals)  ### Power of a conversion mean at time t
    model.importation_Dvar = Var( model.Date, model.RESOURCES, domain=NonNegativeReals,initialize=0)  ### Improtation of a resource at time t
    model.energy_Pvar = Var(model.Date, model.RESOURCES)  ### Amount of a resource at time t
    model.max_PS_Dvar = Var(model.HORAIRE,domain=NonNegativeReals)  ### Puissance souscrite max par plage horaire pour l'année d'opération y
    model.carbon_Pvar = Var( model.Date)  ### CO2 emission at each time t

    ### Storage operation variables
    model.stockLevel_Pvar = Var( model.Date, model.STOCK_TECHNO,domain=NonNegativeReals)  ### level of the energy stock in a storage mean at time t
    model.storageIn_Dvar = Var( model.Date, model.RESOURCES, model.STOCK_TECHNO,domain=NonNegativeReals)  ### Energy stored in a storage mean at time t
    model.storageOut_Dvar = Var( model.Date, model.RESOURCES, model.STOCK_TECHNO,domain=NonNegativeReals)  ### Energy taken out of the in a storage mean at time t
    model.storageConsumption_Pvar = Var( model.Date, model.RESOURCES, model.STOCK_TECHNO,domain=NonNegativeReals)  ### Energy consumed the in a storage mean at time t (other than the one stored)

    # Investment
    model.capacity_Dvar = Var(model.TECHNOLOGIES, domain=NonNegativeReals, initialize=0)
    model.Cmax_Dvar = Var(model.STOCK_TECHNO,domain=NonNegativeReals)  # Maximum capacity of a storage mean
    model.Pmax_Dvar = Var(model.STOCK_TECHNO,domain=NonNegativeReals)  # Maximum flow of energy in/out of a storage mean

    #
    model.powerCosts_Pvar = Var(model.TECHNOLOGIES)  ### Marginal cost for a conversion mean, explicitely defined by definition powerCostsDef
    model.capacityCosts_Pvar = Var(model.TECHNOLOGIES)  ### Fixed costs for a conversion mean, explicitely defined by definition capacityCostsDef
    model.importCosts_Pvar = Var(model.RESOURCES)  ### Cost of ressource imported, explicitely defined by definition importCostsDef
    model.turpeCosts_Pvar = Var(model.RESOURCES,domain=NonNegativeReals)  ### Coûts TURPE pour électricité
    model.storageCosts_Pvar = Var(model.STOCK_TECHNO)  ### Cost of storage for a storage mean, explicitely defined by definition storageCostsDef
    model.carbonCosts_Pvar = Var(model.Date,domain=NonNegativeReals)

    model.dual = Suffix(direction=Suffix.IMPORT)
    model.rc = Suffix(direction=Suffix.IMPORT)
    model.slack = Suffix(direction=Suffix.IMPORT)

    ########################
    # Objective Function   #
    ########################

    def ObjectiveFunction_rule(model):  # OBJ
        return sum(model.powerCosts_Pvar[tech] + model.capacityCosts_Pvar[tech] for tech in model.TECHNOLOGIES)\
               + sum(model.importCosts_Pvar[res] for res in model.RESOURCES)\
               + sum(model.storageCosts_Pvar[s_tech] for s_tech in STOCK_TECHNO)\
               + model.turpeCosts_Pvar['electricity']\
               + sum(model.carbonCosts_Pvar[t] for t in Date)
    model.OBJ = Objective(rule=ObjectiveFunction_rule, sense=minimize)

    #################
    # Constraints   #
    #################
    r = Economics.loc['discountRate'].value
    i = Economics.loc['financeRate'].value

    def f1(r, n):  # This factor converts the investment costs into n annual repayments
        return r / ((1 + r) * (1 - (1 + r) ** -n))

    # powerCosts definition Constraints
    def powerCostsDef_rule(model,tech):  # EQ forall tech in TECHNOLOGIES powerCosts  = sum{t in Date} powerCost[tech]*power[t,tech] / 1E6;
        return sum(model.powerCost[tech] * model.power_Dvar[t, tech] for t in model.Date) == model.powerCosts_Pvar[tech]
    model.powerCostsCtr = Constraint( model.TECHNOLOGIES, rule=powerCostsDef_rule)

    # capacityCosts definition Constraints
    def capacityCostsDef_rule(model, tech):  # EQ forall tech in TECHNOLOGIES
        return (model.investCost[tech]* f1(r, model.lifeSpan[tech]) + model.operationCost[tech]) * model.capacity_Dvar[tech] == model.capacityCosts_Pvar[tech]
    model.capacityCostsCtr = Constraint( model.TECHNOLOGIES, rule=capacityCostsDef_rule)

    # importCosts definition Constraints
    def importCostsDef_rule(model, res):
        return sum((model.importCost[t, res] * model.importation_Dvar[t, res]) for t in model.Date) == model.importCosts_Pvar[ res]
    model.importCostsCtr = Constraint(model.RESOURCES, rule=importCostsDef_rule)

    # # gaz definition Constraints
    # def BiogasDef_rule(model,res):
    #     if res == 'gasBio':
    #         return sum(model.importation_Dvar[t, res] for t in model.Date) <= gasBio_max
    #     else:
    #         return Constraint.Skip
    # model.BiogasCtr = Constraint(model.RESOURCES, rule=BiogasDef_rule)

    # Carbon emission definition Constraints
    def CarbonDef_rule(model,  t):
        return sum((model.power_Dvar[t, tech] * model.EmissionCO2[ tech]) for tech in model.TECHNOLOGIES) + sum(model.importation_Dvar[t, res] * model.emission[t, res] for res in model.RESOURCES) == model.carbon_Pvar[ t]
    model.CarbonDefCtr = Constraint( model.Date, rule=CarbonDef_rule)

    # def CarbonCtr_rule(model):
    #     return sum(model.carbon_Pvar[t] for t in model.Date) <= carbonGoals
    # model.CarbonCtr = Constraint(rule=CarbonCtr_rule)

    # CarbonCosts definition Constraint
    def CarbonCosts_rule(model,t):
        return model.carbonCosts_Pvar[t] == model.carbon_Pvar[t] * CarbonTax
    model.CarbonCostsCtr = Constraint(model.Date, rule=CarbonCosts_rule)

    # TURPE
    def PuissanceSouscrite_rule(model,t, res):
        if res == 'electricity':
            if t in Date_P:
                return model.max_PS_Dvar['P'] >= model.importation_Dvar[t, res]  # en MW
            elif t in Date_HPH:
                return model.max_PS_Dvar['HPH'] >= model.importation_Dvar[t, res]
            elif t in Date_HCH:
                return model.max_PS_Dvar['HCH'] >= model.importation_Dvar[t, res]
            elif t in Date_HPE:
                return model.max_PS_Dvar['HPE'] >= model.importation_Dvar[t, res]
            elif t in Date_HCE:
                return model.max_PS_Dvar['HCE'] >= model.importation_Dvar[t, res]
        else:
            return Constraint.Skip
    model.PuissanceSouscriteCtr = Constraint(model.Date, model.RESOURCES,rule=PuissanceSouscrite_rule)

    def TurpeCtr_rule(model,res):
        if res == 'electricity':
            return model.turpeCosts_Pvar[res] == (
                        sum(model.HTA[t] * model.importation_Dvar[t, res] for t in Date) + model.max_PS_Dvar['P'] * 16310
                        + (model.max_PS_Dvar['HPH'] - model.max_PS_Dvar[ 'P']) * 15760
                        + (model.max_PS_Dvar['HCH'] - model.max_PS_Dvar[ 'HPH']) * 13290
                        + (model.max_PS_Dvar[ 'HPE'] - model.max_PS_Dvar[ 'HCH']) * 8750
                        + (model.max_PS_Dvar[ 'HCE'] - model.max_PS_Dvar[ 'HPE']) * 1670)
        else:
            return model.turpeCosts_Pvar[res] == 0
    model.TurpeCtr = Constraint(model.RESOURCES, rule=TurpeCtr_rule)

    # Capacity constraints

    def Capacity_rule(model, t, tech):  # INEQ forall t, tech
        return model.capacity_Dvar[tech] * model.availabilityFactor[ t, tech] >= model.power_Dvar[ t, tech]
    model.CapacityCtr = Constraint(model.Date, model.TECHNOLOGIES, rule=Capacity_rule)

    # Ressource production constraint
    def Production_rule(model, t,res):  # EQ forall t, res
        if res == 'gas':
            return sum(model.power_Dvar[t, tech] * model.conversionFactor[res, tech] for tech in model.TECHNOLOGIES) \
                   + sum(model.importation_Dvar[t, resource] for resource in gasTypes) \
                   + sum(model.storageOut_Dvar[t, res, s_tech] - model.storageIn_Dvar[ t, res, s_tech] - model.storageConsumption_Pvar[ t, res, s_tech] for s_tech in STOCK_TECHNO) \
                   == model.energy_Pvar[ t, res]
        elif res in gasTypes:
            return model.energy_Pvar[t, res] == 0
        else:
            return sum(model.power_Dvar[ t, tech] * model.conversionFactor[res, tech] for tech in model.TECHNOLOGIES) \
                   + model.importation_Dvar[ t, res] \
                   + sum(model.storageOut_Dvar[ t, res, s_tech] - model.storageIn_Dvar[ t, res, s_tech] - model.storageConsumption_Pvar[ t, res, s_tech] for s_tech in STOCK_TECHNO) \
                   == model.energy_Pvar[t, res]
    model.ProductionCtr = Constraint( model.Date, model.RESOURCES, rule=Production_rule)

    # contrainte d'equilibre offre demande
    def energyCtr_rule(model, t, res):  # INEQ forall t
        return model.energy_Pvar[ t, res] == model.areaConsumption[ t, res]
    model.energyCtr = Constraint( model.Date, model.RESOURCES, rule=energyCtr_rule)

    # storageCosts definition Constraint
    def storageCostsDef_rule(model, s_tech):  # EQ forall s_tech in STOCK_TECHNO
        return (model.storageEnergyCost[ s_tech] * model.Cmax_Dvar[ s_tech] + model.storagePowerCost[ s_tech] * model.Pmax_Dvar[s_tech]) * f1(r,model.storagelifeSpan[s_tech]) \
               + model.storageOperationCost[s_tech]  * model.Pmax_Dvar[ s_tech] == \
               model.storageCosts_Pvar[s_tech]
    model.storageCostsCtr = Constraint( model.STOCK_TECHNO, rule=storageCostsDef_rule)

    # Storage max capacity constraint
    def storageCapacity_rule(model, s_tech):  # INEQ forall s_tech
        return model.Cmax_Dvar[ s_tech] <= model.c_max[ s_tech]
    model.storageCapacityCtr = Constraint( model.STOCK_TECHNO, rule=storageCapacity_rule)

    # Storage max power constraint
    def storagePower_rule(model,  s_tech):  # INEQ forall s_tech
        return model.Pmax_Dvar[ s_tech] <= model.p_max[ s_tech]
    model.storagePowerCtr = Constraint( model.STOCK_TECHNO, rule=storagePower_rule)

    # contraintes de stock puissance
    def StoragePowerUB_rule(model, t, res, s_tech):  # INEQ forall t
        if res == model.resource[ s_tech]:
            return model.storageIn_Dvar[ t, res, s_tech] <= model.Pmax_Dvar[ s_tech]
        else:
            return model.storageIn_Dvar[ t, res, s_tech] == 0
    model.StoragePowerUBCtr = Constraint( model.Date, model.RESOURCES, model.STOCK_TECHNO,rule=StoragePowerUB_rule)

    def StoragePowerLB_rule(model,  t, res, s_tech, ):  # INEQ forall t
        if res == model.resource[s_tech]:
            return model.storageOut_Dvar[ t, res, s_tech] <= model.Pmax_Dvar[ s_tech]
        else:
            return model.storageOut_Dvar[ t, res, s_tech] == 0
    model.StoragePowerLBCtr = Constraint( model.Date, model.RESOURCES, model.STOCK_TECHNO,rule=StoragePowerLB_rule)

    # contrainte de consommation du stockage (autre que l'énergie stockée)
    def StorageConsumption_rule(model,  t, res, s_tech):  # EQ forall t
        temp = model.resource[s_tech]
        if res == temp:
            return model.storageConsumption_Pvar[t, res, s_tech] == 0
        else:
            return model.storageConsumption_Pvar[ t, res, s_tech] == model.storageFactorIn[res, s_tech] * model.storageIn_Dvar[ t, temp, s_tech] + model.storageFactorOut[res, s_tech] * model.storageOut_Dvar[t, temp, s_tech]
    model.StorageConsumptionCtr = Constraint( model.Date, model.RESOURCES, model.STOCK_TECHNO,rule=StorageConsumption_rule)

    # contraintes de stock capacité
    def StockLevel_rule(model, t, s_tech):  # EQ forall t
        res = model.resource[s_tech]
        if t != min(getattr(model, "Date").data()):
            t_moins_1 = t_moins_1 = t - timedelta(hours=1)
            return model.stockLevel_Pvar[ t, s_tech] == \
                   model.stockLevel_Pvar[t_moins_1, s_tech] * (1 - model.dissipation[res, s_tech]) \
                   + model.storageIn_Dvar[ t, res, s_tech] * model.storageFactorIn[res, s_tech]\
                   - model.storageOut_Dvar[t, res, s_tech] * model.storageFactorOut[res, s_tech]
        else:
            return model.stockLevel_Pvar[t, s_tech] == \
                   model.stockLevel_Pvar[ Last_date, s_tech] \
                   + model.storageIn_Dvar[t, res, s_tech] * model.storageFactorIn[res, s_tech] \
                   - model.storageOut_Dvar[t, res, s_tech] * model.storageFactorOut[res, s_tech]
    model.StockLevelCtr = Constraint( model.Date, model.STOCK_TECHNO, rule=StockLevel_rule)

    def StockCapacity_rule(model, t, s_tech, ):  # INEQ forall t
        return model.stockLevel_Pvar[t, s_tech] <= model.Cmax_Dvar[ s_tech]
    model.StockCapacityCtr = Constraint( model.Date, model.STOCK_TECHNO, rule=StockCapacity_rule)

    if "minCapacity" in TechParameters:
        def maxCapacity_rule(model, tech):  # INEQ forall t, tech
            return model.minCapacity[tech] <= model.capacity_Dvar[ tech]
        model.maxCapacityCtr = Constraint( model.TECHNOLOGIES, rule=maxCapacity_rule)

    if "maxCapacity" in TechParameters:
        def minCapacity_rule(model, tech):  # INEQ forall t, tech
            return model.maxCapacity[tech] >= model.capacity_Dvar[ tech]
        model.minCapacityCtr = Constraint( model.TECHNOLOGIES, rule=minCapacity_rule)

    if "EnergyNbhourCap" in TechParameters:
        def storage_rule(model,tech):  # INEQ forall t, tech
            if model.EnergyNbhourCap[tech] > 0:
                return model.EnergyNbhourCap[tech] * model.capacity_Dvar[ tech] >= sum(model.power_Dvar[ t, tech] for t in model.TIMESTAMP)
            else:
                return Constraint.Skip
        model.storageCtr = Constraint( model.TECHNOLOGIES, rule=storage_rule)

    if "RampConstraintPlus" in TechParameters:
        def rampCtrPlus_rule(model, t, tech):  # INEQ forall t<
            if model.RampConstraintPlus[ tech] > 0:
                t_plus_1 = t + timedelta(hours=1)
                return model.power_Dvar[t_plus_1, tech] - model.power_Dvar[t, tech] <= model.capacity_Dvar[ tech] * model.RampConstraintPlus[ tech]
            else:
                return Constraint.Skip
        model.rampCtrPlus = Constraint(model.Date_MinusOne, model.TECHNOLOGIES,rule=rampCtrPlus_rule)

    if "RampConstraintMoins" in TechParameters:
        def rampCtrMoins_rule(model,t, tech):  # INEQ forall t<
            if model.RampConstraintMoins[ tech] > 0:
                t_plus_1 = t + timedelta(hours=1)
                return model.power_Dvar[t_plus_1, tech] - model.power_Dvar[t, tech] >= - model.capacity_Dvar[ tech] * model.RampConstraintMoins[ tech]
            else:
                return Constraint.Skip
        model.rampCtrMoins = Constraint(model.Date_MinusOne, model.TECHNOLOGIES,rule=rampCtrMoins_rule)

    if "RampConstraintPlus2" in TechParameters:
        def rampCtrPlus2_rule(model, t, tech):  # INEQ forall t<
            if model.RampConstraintPlus2[tech] > 0:
                t_plus_2 = t + timedelta(hours=2)
                t_plus_3 = t + timedelta(hours=3)
                t_plus_1 = t + timedelta(hours=1)
                var = (model.power_Dvar[ t_plus_2, tech] + model.power_Dvar[t_plus_3, tech]) / 2 - (model.power_Dvar[ t_plus_1, tech] + model.power_Dvar[ t, tech]) / 2
                return var <= model.capacity_Dvar[ tech] * model.RampConstraintPlus[tech]
            else:
                return Constraint.Skip
        model.rampCtrPlus2 = Constraint(model.Date_MinusThree, model.TECHNOLOGIES,rule=rampCtrPlus2_rule)

    if "RampConstraintMoins2" in TechParameters:
        def rampCtrMoins2_rule(model, t, tech):  # INEQ forall t<
            if model.RampConstraintMoins2[ tech] > 0:
                t_plus_2 = t + timedelta(hours=2)
                t_plus_3 = t + timedelta(hours=3)
                t_plus_1 = t + timedelta(hours=1)
                var = (model.power_Dvar[t_plus_2, tech] + model.power_Dvar[t_plus_3, tech]) / 2 - (model.power_Dvar[t_plus_1, tech] + model.power_Dvar[ t, tech]) / 2
                return var >= - model.capacity_Dvar[ tech] * model.RampConstraintMoins2[tech]
            else:
                return Constraint.Skip
        model.rampCtrMoins2 = Constraint(model.Date_MinusThree, model.TECHNOLOGIES,rule=rampCtrMoins2_rule)

    return model