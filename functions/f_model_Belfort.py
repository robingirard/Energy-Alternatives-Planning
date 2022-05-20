#from pyomo.environ import *
import pandas as pd
from pyomo.core import *
from functions.f_tools import *
from datetime import *

def GetElectricSystemModel_Belfort_SingleNode(areaConsumption,lossesRate,availabilityFactor,
                                              TechParameters, StorageParameters,
                                              to_flex_consumption,FlexParameters,labour_ratio,
                                              TimeName='Date',TechName='TECHNOLOGIES',
                                              StockTechName='STOCK_TECHNO',FlexName='FLEX_CONSUM',
                                              FlexConsumptionName='Consumption',
                                              areaConsumptionName='areaConsumption',
                                              availabilityFactorName='availabilityFactor',
                                              LaborRatioName='labour_ratio',lossesRateName='Taux_pertes'):
    ## Generic variables
    d1h = timedelta(hours=1)

    ## Cleaning
    availabilityFactor = availabilityFactor.fillna(method='pad');
    areaConsumption = areaConsumption.fillna(method='pad');
    to_flex_consumption = to_flex_consumption.fillna(method='pad');

    ### obtaining dimensions values

    TECHNOLOGIES = set(TechParameters.index.get_level_values(TechName).unique())
    STOCK_TECHNO = set(StorageParameters.index.get_level_values(StockTechName).unique())
    FLEX_CONSUM = set(FlexParameters.index.get_level_values(FlexName).unique())
    TIME = set(areaConsumption.index.get_level_values(TimeName).unique())
    TIME_list = areaConsumption.index.get_level_values(TimeName).unique()

    TIME_df=pd.DataFrame({TimeName:TIME_list})
    TIME_df['day']=TIME_df[TimeName].apply(lambda x: x.dayofyear)
    TIME_df['day_ev']=TIME_df[TimeName].apply(lambda x: (x-12*d1h).dayofyear)
    TIME_df['morn_ev']=TIME_df[TimeName].apply(lambda x: 1 if 8<=x.hour<=11 else 0)# morning
    TIME_df['aft_ev']=TIME_df[TimeName].apply(lambda x: 1 if 12<=x.hour<=17 else 0)# afternoon
    TIME_df['night_ev']=TIME_df[TimeName].apply(lambda x: 1 if 18<=x.hour or x.hour<=7 else 0)# night
    TIME_df['ctr_ev'] = TIME_df[TimeName].apply(lambda x: 1 if 7<=x.hour<=8 or 12<=x.hour<=18 else 0)# flex constrained time for electric vehicles
    TIME_df['week']=TIME_df[TimeName].apply(lambda x: x.isocalendar().week)
    TIME_df=TIME_df.set_index(TimeName)
    #TIME_df['hour']=TIME_df[TimeName].apply(lambda x: x.hour)

    DAY = set(TIME_df['day'].unique())
    DAY_EV=set(TIME_df['day_ev'].unique())
    WEEK=set(TIME_df['week'].unique())
    ## obtaining max powers


    ### model definition

    model = ConcreteModel()

    ###############
    # Sets       ##
    ###############
    model.TECHNOLOGIES = Set(initialize=TECHNOLOGIES, ordered=False)
    model.TIME = Set(initialize=TIME, ordered=False)
    model.STOCK_TECHNO = Set(initialize=STOCK_TECHNO, ordered=False)
    model.FLEX_CONSUM = Set(initialize=FLEX_CONSUM, ordered=False)
    model.TIME_TECHNOLOGIES = model.TIME * model.TECHNOLOGIES
    model.TIME_STOCK_TECHNO = model.TIME * model.STOCK_TECHNO
    model.TIME_FLEX_CONSUM = model.TIME * model.FLEX_CONSUM

    model.DAY=Set(initialize=DAY, ordered=False)
    model.DAY_EV=Set(initialize=DAY_EV, ordered=False)
    model.WEEK=Set(initialize=WEEK, ordered=False)

    ## For ramp constraints
    model.TIME_MinusOne = Set(initialize=TIME_list[: len(TIME) - 1], ordered=False)
    model.TIME_MinusThree = Set(initialize=TIME_list[: len(TIME) - 3], ordered=False)

    ###############
    # Parameters ##
    ###############

    model.areaConsumption = Param(model.TIME, default=0,
                                  initialize=areaConsumption.loc[:, areaConsumptionName].squeeze().to_dict(), domain=Any)
    model.lossesRate=Param(model.TIME, default=0,
                                  initialize=lossesRate.loc[:, lossesRateName].squeeze().to_dict(), domain=PercentFraction)
    model.availabilityFactor = Param(model.TIME_TECHNOLOGIES, domain=PercentFraction, default=1,
                                     initialize=availabilityFactor.loc[:, availabilityFactorName].squeeze().to_dict())
    model.to_flex_consumption = Param(model.TIME,model.FLEX_CONSUM, default=0,
                                  initialize=to_flex_consumption.loc[:, FlexConsumptionName].squeeze().to_dict(),
                                  domain=Any)
    model.max_power=Param(model.FLEX_CONSUM,default=0,
                          initialize=to_flex_consumption[FlexConsumptionName].groupby(FlexName).max().to_dict(),
                          domain=Any)
    model.labour_ratio=Param(model.TIME_FLEX_CONSUM,default=0,
                          initialize=labour_ratio.loc[:,LaborRatioName].squeeze().to_dict(),
                          domain=Any)
    if "EV" in FLEX_CONSUM:
        model.max_power_night_EV=Param(initialize=model.max_power["EV"],domain=Any)
        model.max_power_aft_EV=Param(initialize=(to_flex_consumption.loc[(slice(None),"EV"),FlexConsumptionName]*TIME_df['aft_ev']).max(),
                                                 domain=Any)
        model.max_power_morn_EV = Param(initialize=(to_flex_consumption.loc[(slice(None), "EV"), FlexConsumptionName] * TIME_df['morn_ev']).max(),
                                       domain=Any)
    # with test of existing columns on TechParameters
    for COLNAME in TechParameters:
        if COLNAME not in [TechName, "AREAS"]:  ### each column in TechParameters will be a parameter
            exec("model." + COLNAME + " =          Param(model.TECHNOLOGIES, domain=Any,default=0," +
                 "initialize=TechParameters." + COLNAME + ".squeeze().to_dict())")

    for COLNAME in StorageParameters:
        if COLNAME not in [StockTechName, "AREAS"]:  ### each column in TechParameters will be a parameter
            exec("model." + COLNAME + " =          Param(model.STOCK_TECHNO, domain=Any,default=0," +
                 "initialize=StorageParameters." + COLNAME + ".squeeze().to_dict())")

    for COLNAME in FlexParameters:
        if COLNAME not in [FlexName, "AREAS"]:  ### each column in TechParameters will be a parameter
            exec("model." + COLNAME + " =          Param(model.FLEX_CONSUM, domain=Any,default=0," +
                 "initialize=FlexParameters." + COLNAME + ".squeeze().to_dict())")

    ################
    # Variables    #
    ################

    # In this section, variables are separated in two categories : decision variables wich are the reals variables of the otimisation problem (these are noted Dvar), and problem variables which are resulting of calculation and are convenient for the readability and the analyse of results (these are noted Pvar)

    ### Operation variables
    model.power_Dvar = Var(model.TIME, model.TECHNOLOGIES,
                               domain=NonNegativeReals)  ### Power of a conversion mean at time t
    model.energy_Pvar =Var(model.TIME, domain=NonNegativeReals) ### Total power produced at time t
    model.powerCosts_Pvar = Var(
            model.TECHNOLOGIES)  ### Marginal cost for a conversion mean, explicitely defined by definition powerCostsDef

    ### Planing variables
    model.capacity_Dvar = Var(model.TECHNOLOGIES, domain=NonNegativeReals)  ### Capacity of a conversion mean
    model.capacityCosts_Pvar = Var(
            model.TECHNOLOGIES)  ### Fixed costs for a conversion mean, explicitely defined by definition capacityCostsDef

    ### Storage variables
    model.storageIn_Pvar = Var(model.TIME, model.STOCK_TECHNO,
                                   domain=NonNegativeReals)  ### Energy stored in a storage mean at time t
    model.storageOut_Pvar = Var(model.TIME, model.STOCK_TECHNO,
                                    domain=NonNegativeReals)  ### Energy taken out of the in a storage mean at time t
    # model.storageConsumption_Pvar = Var(model.TIMES, model.STOCK_TECHNO,
                                            #domain=NonNegativeReals)  ### Energy consumed the in a storage mean at time t (other than the one stored)
    model.stockLevel_Pvar = Var(model.TIME, model.STOCK_TECHNO,
                                    domain=NonNegativeReals)  ### level of the energy stock in a storage mean at time t
    model.stockLevel_ini_Dvar=Var(model.STOCK_TECHNO,
                                    domain=NonNegativeReals) ### initial level of storage
    model.storageCosts_Pvar = Var(
            model.STOCK_TECHNO)  ### Cost of storage for a storage mean, explicitely defined by definition storageCostsDef
    model.Pmax_in_Dvar = Var(model.STOCK_TECHNO)  # Maximum flow of energy in a storage mean
    model.Pmax_out_Dvar = Var(model.STOCK_TECHNO)  # Maximum flow of energy out of a storage mean
    model.Cmax_Pvar = Var(model.STOCK_TECHNO)  # Maximum capacity of a storage mean

    ## Flexibility variables
    model.flexCosts_Pvar = Var(model.FLEX_CONSUM) # Costs of flexibility means
    model.total_consumption_Pvar = Var(model.TIME, domain=NonNegativeReals)  # variable de calcul intermediaire
    model.flex_consumption_Pvar = Var(model.TIME, model.FLEX_CONSUM,
                                 domain=NonNegativeReals)  # flexible consumption variable
    #model.lab_cost_Pvar = Var(model.TIME, model.FLEX_CONSUM, domain=NonNegativeReals)  # labour cost
    #Intermediary variables for labor cost calculation
    model.a_plus_Pvar = Var(model.TIME, model.FLEX_CONSUM, domain=NonNegativeReals)
    model.a_minus_Pvar = Var(model.TIME, model.FLEX_CONSUM, domain=NonNegativeReals)


    model.flex_Dvar = Var(model.TIME, model.FLEX_CONSUM, domain=Reals)#Flexibility variable

    model.increased_max_power_Dvar=Var(model.FLEX_CONSUM, domain=NonNegativeReals)

    ### Other variables
    model.dual = Suffix(direction=Suffix.IMPORT)
    model.rc = Suffix(direction=Suffix.IMPORT)
    model.slack = Suffix(direction=Suffix.IMPORT)

    ########################
    # Objective Function   #
    ########################

    def ObjectiveFunction_rule(model):  # OBJ
        return (sum(model.powerCosts_Pvar[tech] + model.capacityCosts_Pvar[tech] for tech in model.TECHNOLOGIES)\
                + sum(model.storageCosts_Pvar[s_tech] for s_tech in model.STOCK_TECHNO)\
                + sum(model.flexCosts_Pvar[flex] for flex in model.FLEX_CONSUM))

    model.OBJ = Objective(rule=ObjectiveFunction_rule, sense=minimize)

    #################
    # Constraints   #
    #################

    ### Costs definitions

    # energyCosts definition Constraint
    def powerCostsDef_rule(model,tech):
        return sum(model.energyCost[tech] * model.power_Dvar[t, tech]#+model.margvarCost[tech] * model.power_Dvar[t, tech]**2\
                   for t in model.TIME) == model.powerCosts_Pvar[tech]

    model.powerCostsCtr = Constraint(model.TECHNOLOGIES, rule=powerCostsDef_rule)

    # capacityCosts definition Constraint
    def capacityCostsDef_rule(model,tech):
        return model.capacityCost[tech] * model.capacity_Dvar[tech] == model.capacityCosts_Pvar[tech]

    model.capacityCostsCtr = Constraint(model.TECHNOLOGIES, rule=capacityCostsDef_rule)

    # storageCosts definition Constraint
    def storageCostsDef_rule(model,s_tech):
        return model.storageCost_power_in[s_tech] * model.Pmax_in_Dvar[s_tech]\
               + model.storageCost_power_out[s_tech] * model.Pmax_out_Dvar[s_tech]\
               + model.storageCost_energy[s_tech] * model.Cmax_Pvar[s_tech]== model.storageCosts_Pvar[s_tech]

    model.storageCostsCtr = Constraint(model.STOCK_TECHNO, rule=storageCostsDef_rule)

    # Felxibility cost definition
    def consumption_flex_costDef_rule(model, conso_type):
        return model.flexCosts_Pvar[conso_type] == model.LoadCost[conso_type] * model.increased_max_power_Dvar[
            conso_type] + sum(model.labour_ratio[t, conso_type] * model.labourcost[conso_type] * (
                model.a_plus_Pvar[t, conso_type] - model.a_minus_Pvar[t, conso_type]) for t in model.TIME)

    model.consumption_flex_costCtr = Constraint(model.FLEX_CONSUM, rule=consumption_flex_costDef_rule)

    ### Storage constraints

    # Storage max power constraint
    def storagePower_in_rule(model, s_tech):  # INEQ forall s_tech
        return model.Pmax_in_Dvar[s_tech] <= model.p_max_in[s_tech]

    model.storagePowerInCtr = Constraint(model.STOCK_TECHNO, rule=storagePower_in_rule)

    def storagePower_out_rule(model, s_tech):  # INEQ forall s_tech
        return model.Pmax_out_Dvar[s_tech] <= model.p_max_out[s_tech]

    model.storagePowerOutCtr = Constraint(model.STOCK_TECHNO, rule=storagePower_out_rule)

    def storagePower_in_out_rule(model, s_tech):
        if model.p_max_out_eq_p_max_in[s_tech]=='yes':
            return model.Pmax_in_Dvar[s_tech]==model.Pmax_out_Dvar[s_tech]
        else:
            return Constraint.Skip

    model.storagePowerInOutCtr = Constraint(model.STOCK_TECHNO, rule=storagePower_in_out_rule)

    # Storage max capacity constraint
    def storageCapacity_rule(model, s_tech):  # INEQ forall s_tech
        if model.h_max[s_tech]>0:
            return model.Cmax_Pvar[s_tech] <= model.h_max[s_tech]*model.Pmax_out_Dvar[s_tech]/model.efficiency_out[s_tech]
        else:
            return Constraint.Skip

    model.storageCapacityCtr = Constraint(model.STOCK_TECHNO, rule=storageCapacity_rule)

    # contraintes de stock puissance
    def StoragePowerUB_rule(model, t, s_tech):  # INEQ forall t
        return model.storageIn_Pvar[t, s_tech] - model.Pmax_in_Dvar[s_tech] <= 0

    model.StoragePowerUBCtr = Constraint(model.TIME, model.STOCK_TECHNO, rule=StoragePowerUB_rule)

    def StoragePowerLB_rule(model, t, s_tech, ):  # INEQ forall t
        return model.storageOut_Pvar[t, s_tech] - model.Pmax_out_Dvar[s_tech] <= 0

    model.StoragePowerLBCtr = Constraint(model.TIME, model.STOCK_TECHNO, rule=StoragePowerLB_rule)

    # contraintes de stock capacité
    def StockLevel_rule(model, t, s_tech):  # EQ forall t
        if t>TIME_list[0]:
            return model.stockLevel_Pvar[t, s_tech] == model.stockLevel_Pvar[t - d1h, s_tech]*(1-model.dissipation[s_tech])\
                   + model.storageIn_Pvar[t, s_tech] * model.efficiency_in[s_tech] - model.storageOut_Pvar[t, s_tech]/model.efficiency_out[s_tech]
        else:
            return model.stockLevel_Pvar[t, s_tech] == model.storageIn_Pvar[t, s_tech] * model.efficiency_in[
                s_tech] - model.storageOut_Pvar[t, s_tech]/model.efficiency_out[s_tech]\
                   + model.stockLevel_ini_Dvar[s_tech]*(1-model.dissipation[s_tech])

    model.StockLevelCtr = Constraint(model.TIME, model.STOCK_TECHNO, rule=StockLevel_rule)

    def StockLevel_ini_rule(model,s_tech):
        return model.stockLevel_Pvar[TIME_list[-1], s_tech]>=model.stockLevel_ini_Dvar[s_tech]

    model.StockLevelIniCtr = Constraint(model.STOCK_TECHNO, rule=StockLevel_ini_rule)

    def StockCapacity_rule(model, t, s_tech, ):  # INEQ forall t
        return model.stockLevel_Pvar[t, s_tech] <= model.Cmax_Pvar[s_tech]

    model.StockCapacityCtr = Constraint(model.TIME, model.STOCK_TECHNO, rule=StockCapacity_rule)

    ### Production constraints

    # Capacity constraint
    def Capacity_rule(model, t, tech):  # INEQ forall t, tech
        return model.capacity_Dvar[tech] * model.availabilityFactor[t, tech] >= model.power_Dvar[t, tech]

    model.CapacityCtr = Constraint(model.TIME, model.TECHNOLOGIES, rule=Capacity_rule)

    # Ressource production constraint
    def Production_rule(model, t):  # EQ forall t, res
        return sum(model.power_Dvar[t, tech] for tech in model.TECHNOLOGIES) + \
               sum(model.storageOut_Pvar[t, s_tech] - model.storageIn_Pvar[t, s_tech] for s_tech in STOCK_TECHNO)\
               == model.energy_Pvar[t]

    model.ProductionCtr = Constraint(model.TIME, rule=Production_rule)

    # contrainte d'equilibre offre demande
    def energyCtr_rule(model, t):  # INEQ forall t
        return model.energy_Pvar[t] == model.total_consumption_Pvar[t]
        #return sum(model.power_Dvar[t, tech] for tech in model.TECHNOLOGIES) + \
               #sum(model.storageOut_Pvar[t, s_tech] - model.storageIn_Pvar[t, s_tech] for s_tech in STOCK_TECHNO)\
               #>= model.total_consumption_Pvar[t]

    model.energyCtr = Constraint(model.TIME, rule=energyCtr_rule)

    if "maxCapacity" in TechParameters:
        def maxCapacity_rule(model, tech):  # INEQ forall t, tech
            return model.maxCapacity[tech] >= model.capacity_Dvar[tech]

        model.maxCapacityCtr = Constraint(model.TECHNOLOGIES, rule=maxCapacity_rule)

    if "minCapacity" in TechParameters:
        def minCapacity_rule(model, tech):  # INEQ forall t, tech
            return model.minCapacity[tech] <= model.capacity_Dvar[tech]

        model.minCapacityCtr = Constraint(model.TECHNOLOGIES, rule=minCapacity_rule)

    if "EnergyNbhourCap" in TechParameters:
        def storage_rule(model, tech):  # INEQ forall t, tech
            if model.EnergyNbhourCap[tech] > 0:
                return model.EnergyNbhourCap[tech] * model.capacity_Dvar[tech] >= sum(
                    model.power_Dvar[t, tech] for t in model.TIME)
            else:
                return Constraint.Skip

        model.storageCtr = Constraint(model.TECHNOLOGIES, rule=storage_rule)

    ### Ramp constraints

    if "RampConstraintPlus" in TechParameters:
        def rampCtrPlus_rule(model, t, tech):  # INEQ forall t<
            if model.RampConstraintPlus[tech] > 0:
                return model.power_Dvar[t + d1h, tech] - model.power_Dvar[t, tech] <= model.capacity_Dvar[tech] * \
                       model.RampConstraintPlus[tech];
            else:
                return Constraint.Skip

        model.rampCtrPlus = Constraint(model.TIME_MinusOne, model.TECHNOLOGIES, rule=rampCtrPlus_rule)

    if "RampConstraintMoins" in TechParameters:
        def rampCtrMoins_rule(model, t, tech):  # INEQ forall t<
            if model.RampConstraintMoins[tech] > 0:
                return model.power_Dvar[t + d1h, tech] - model.power_Dvar[t, tech] >= - model.capacity_Dvar[tech] * \
                       model.RampConstraintMoins[tech];
            else:
                return Constraint.Skip

        model.rampCtrMoins = Constraint(model.TIME_MinusOne, model.TECHNOLOGIES, rule=rampCtrMoins_rule)

    if "RampConstraintPlus2" in TechParameters:
        def rampCtrPlus2_rule(model, t, tech):  # INEQ forall t<
            if model.RampConstraintPlus2[tech] > 0:
                var = (model.power_Dvar[t + 2*d1h, tech] + model.power_Dvar[t + 3*d1h, tech]) / 2 - (
                            model.power_Dvar[t + d1h, tech] + model.power_Dvar[t, tech]) / 2;
                return var <= model.capacity_Dvar[tech] * model.RampConstraintPlus[tech];
            else:
                return Constraint.Skip

        model.rampCtrPlus2 = Constraint(model.TIME_MinusThree, model.TECHNOLOGIES, rule=rampCtrPlus2_rule)

    if "RampConstraintMoins2" in TechParameters:
        def rampCtrMoins2_rule(model, t, tech):  # INEQ forall t<
            if model.RampConstraintMoins2[tech] > 0:
                var = (model.power_Dvar[t + 2*d1h, tech] + model.power_Dvar[t + 3*d1h, tech]) / 2 - (
                            model.power_Dvar[t + d1h, tech] + model.power_Dvar[t, tech]) / 2;
                return var >= - model.capacity_Dvar[tech] * model.RampConstraintMoins2[tech];
            else:
                return Constraint.Skip

        model.rampCtrMoins2 = Constraint(model.TIME_MinusThree, model.TECHNOLOGIES, rule=rampCtrMoins2_rule)

    ### Flexibility constraints

    # Total consumtion rule
    def total_consumption_rule(model, t):
        return model.total_consumption_Pvar[t] == (model.areaConsumption[t] + sum(
            model.flex_consumption_Pvar[t, name] for name in model.FLEX_CONSUM))*(1+model.lossesRate[t])

    model.total_consumptionCtr = Constraint(model.TIME, rule=total_consumption_rule)

    # Max power rule
    def max_power_rule(model, conso_type, t):
        if model.flex_type[conso_type] != 'day_ev':
            return model.max_power[conso_type] + model.increased_max_power_Dvar[conso_type] >= model.flex_consumption_Pvar[t, conso_type]
        else:
            if TIME_df.loc[t,'morn_ev']==1:
                return model.max_power_morn_EV + model.increased_max_power_Dvar[conso_type]>=model.flex_consumption_Pvar[t, conso_type]
            if TIME_df.loc[t,'aft_ev'] == 1:
                return model.max_power_aft_EV >= model.flex_consumption_Pvar[t, conso_type]
            else:
                return model.max_power_night_EV >= model.flex_consumption_Pvar[t, conso_type]

    model.max_powerCtr = Constraint(model.FLEX_CONSUM, model.TIME, rule=max_power_rule)

    # Day constraint
    def consum_eq_day(model, t_day, conso_type):
        if model.flex_type[conso_type] == 'day':
            t_range = TIME_df[TIME_df.day == t_day].index.get_level_values(TimeName)
            return sum(model.flex_consumption[t, conso_type] for t in t_range) == sum(
                model.to_flex_consumption[t, conso_type] for t in t_range)
        else:
            return Constraint.Skip

    model.consum_eq_day_Ctr = Constraint(model.DAY, model.FLEX_CONSUM, rule=consum_eq_day)

    # Morning and night constraint electric vehicle
    def consum_eq_night_morn_ev(model, t_day, conso_type):
        if model.flex_type[conso_type] == 'day_ev':
            t_range = TIME_df[(TIME_df.day_ev == t_day)&((TIME_df.morn_ev==1)|(TIME_df.night_ev==1))].index.get_level_values(TimeName)
            return sum(model.flex_consumption_Pvar[t, conso_type] for t in t_range) == sum(
                model.to_flex_consumption[t, conso_type] for t in t_range)
        else:
            return Constraint.Skip

    model.consum_eq_night_morn_ev_Ctr = Constraint(model.DAY_EV, model.FLEX_CONSUM, rule=consum_eq_night_morn_ev)

    # Afternoon constraint electric vehicle
    def consum_eq_aft_ev(model, t_day, conso_type):
        if model.flex_type[conso_type] == 'day_ev':
            t_range = TIME_df[(TIME_df.day == t_day)&(TIME_df.aft_ev==1)].index.get_level_values(TimeName)
            return sum(model.flex_consumption_Pvar[t, conso_type] for t in t_range) == sum(
                model.to_flex_consumption[t, conso_type] for t in t_range)
        else:
            return Constraint.Skip

    model.consum_eq_aft_ev_Ctr = Constraint(model.DAY, model.FLEX_CONSUM, rule=consum_eq_aft_ev)

    # Week constraint
    def consum_eq_week(model, t_week, conso_type):
        if model.flex_type[conso_type] == 'week':
            t_range = TIME_df[TIME_df.week == t_week].index.get_level_values(TimeName)
            return sum(model.flex_consumption_Pvar[t, conso_type] for t in t_range) == sum(
                model.to_flex_consumption[t, conso_type] for t in t_range)
        else:
            return Constraint.Skip

    model.consum_eq_week_Ctr = Constraint(model.WEEK, model.FLEX_CONSUM, rule=consum_eq_week)

    # Year constraint
    def consum_eq_year(model, conso_type):
        if model.flex_type[conso_type] == 'year':
            return sum(model.flex_consumption_Pvar[t, conso_type] for t in model.TIME) == sum(
                model.to_flex_consumption[t, conso_type] for t in model.TIME)
        else:
            return Constraint.Skip
    model.consum_eq_year_Ctr = Constraint(model.FLEX_CONSUM, rule=consum_eq_year)

    # Flex constraint
    def consum_flex_rule(model, t, conso_type):
        return model.flex_consumption_Pvar[t, conso_type] == model.to_flex_consumption[t, conso_type] * (
                1 - model.flex_Dvar[t, conso_type])

    model.consum_flex_Ctr = Constraint(model.TIME, model.FLEX_CONSUM, rule=consum_flex_rule)

    # Flex ratio constraint
    def flex_variation_sup_rule(model, t, conso_type):
        if model.flex_type[conso_type]!= 'day_ev' or TIME_df.loc[t,'ctr_ev']==1:
            return model.flex_Dvar[t, conso_type] <= model.flex_ratio[conso_type]
        else:
            return Constraint.Skip

    model.flex_variation_sup_Ctr = Constraint(model.TIME, model.FLEX_CONSUM, rule=flex_variation_sup_rule)

    def flex_variation_inf_rule(model, t, conso_type):
        if model.flex_type[conso_type] != 'day_ev' or TIME_df.loc[t, 'ctr_ev'] == 1:
            return model.flex_Dvar[t, conso_type] >= -model.flex_ratio[conso_type]
        else:
            return Constraint.Skip

    model.flex_variation_inf_Ctr = Constraint(model.TIME, model.FLEX_CONSUM, rule=flex_variation_inf_rule)

    # Definition a_plus, a_minus
    def a_plus_minus_rule(model, conso_type, t):
        return model.flex_consumption_Pvar[t, conso_type] - model.to_flex_consumption[t, conso_type] == \
               model.a_plus_Pvar[t, conso_type] - model.a_minus_Pvar[t, conso_type]

    model.a_plus_minusCtr = Constraint(model.FLEX_CONSUM, model.TIME, rule=a_plus_minus_rule)

    return model

def GetElectricSystemModel_Belfort_SingleNode_H2(areaConsumption,lossesRate,availabilityFactor,
                                              TechParameters, StorageParameters,
                                              to_flex_consumption,FlexParameters,labour_ratio,
                                              H2_consumption,
                                              TimeName='Date',TechName='TECHNOLOGIES',
                                              StockTechName='STOCK_TECHNO',FlexName='FLEX_CONSUM',
                                              PowertoH2toPowerName='PowertoH2toPower',
                                              FlexConsumptionName='Consumption',
                                              areaConsumptionName='areaConsumption',
                                              availabilityFactorName='availabilityFactor',
                                              LaborRatioName='labour_ratio',lossesRateName='Taux_pertes',
                                              H2ConsumptionName='H2'):

    ## Generic variables
    d1h = timedelta(hours=1)

    ## Cleaning
    availabilityFactor = availabilityFactor.fillna(method='pad');
    areaConsumption = areaConsumption.fillna(method='pad');
    to_flex_consumption = to_flex_consumption.fillna(method='pad');

    ### obtaining dimensions values

    TECHNOLOGIES = set(TechParameters.index.get_level_values(TechName).unique())
    STOCK_TECHNO = set(StorageParameters.index.get_level_values(StockTechName).unique())
    FLEX_CONSUM = set(FlexParameters.index.get_level_values(FlexName).unique())
    TIME = set(areaConsumption.index.get_level_values(TimeName).unique())
    TIME_list = areaConsumption.index.get_level_values(TimeName).unique()

    TIME_df=pd.DataFrame({TimeName:TIME_list})
    TIME_df['day']=TIME_df[TimeName].apply(lambda x: x.dayofyear)
    TIME_df['day_ev']=TIME_df[TimeName].apply(lambda x: (x-12*d1h).dayofyear)
    TIME_df['daytime_ev'] = TIME_df[TimeName].apply(lambda x: 1 if 8 <= x.hour <= 17 else 0)  # 1 if day
    TIME_df['night_ev']=TIME_df[TimeName].apply(lambda x: 1 if 18<=x.hour or x.hour<=7 else 0)# 1 if night
    TIME_df['ctr_ev'] = TIME_df[TimeName].apply(lambda x: 1 if 7<=x.hour<=8 or 17<=x.hour<=18 else 0)# flex constrained time for electric vehicles
    TIME_df['week']=TIME_df[TimeName].apply(lambda x: x.isocalendar().week)
    TIME_df=TIME_df.set_index(TimeName)

    DAY = set(TIME_df['day'].unique())
    DAY_EV=set(TIME_df['day_ev'].unique())
    WEEK=set(TIME_df['week'].unique())

    ### model definition

    model = ConcreteModel()

    ###############
    # Sets       ##
    ###############
    model.TECHNOLOGIES = Set(initialize=TECHNOLOGIES, ordered=False)
    model.TIME = Set(initialize=TIME, ordered=False)
    model.STOCK_TECHNO = Set(initialize=STOCK_TECHNO, ordered=False)
    model.FLEX_CONSUM = Set(initialize=FLEX_CONSUM, ordered=False)
    model.TIME_TECHNOLOGIES = model.TIME * model.TECHNOLOGIES
    model.TIME_STOCK_TECHNO = model.TIME * model.STOCK_TECHNO
    model.TIME_FLEX_CONSUM = model.TIME * model.FLEX_CONSUM

    model.DAY=Set(initialize=DAY, ordered=False)
    model.DAY_EV=Set(initialize=DAY_EV, ordered=False)
    model.WEEK=Set(initialize=WEEK, ordered=False)

    ## For ramp constraints
    model.TIME_MinusOne = Set(initialize=TIME_list[: len(TIME) - 1], ordered=False)
    model.TIME_MinusThree = Set(initialize=TIME_list[: len(TIME) - 3], ordered=False)

    ###############
    # Parameters ##
    ###############

    model.areaConsumption = Param(model.TIME, default=0,
                                  initialize=areaConsumption.loc[:, areaConsumptionName].squeeze().to_dict(), domain=Any)
    model.lossesRate=Param(model.TIME, default=0,
                                  initialize=lossesRate.loc[:, lossesRateName].squeeze().to_dict(), domain=PercentFraction)
    model.availabilityFactor = Param(model.TIME_TECHNOLOGIES, domain=PercentFraction, default=1,
                                     initialize=availabilityFactor.loc[:, availabilityFactorName].squeeze().to_dict())
    model.to_flex_consumption = Param(model.TIME,model.FLEX_CONSUM, default=0,
                                  initialize=to_flex_consumption.loc[:, FlexConsumptionName].squeeze().to_dict(),
                                  domain=Any)
    model.max_power=Param(model.FLEX_CONSUM,default=0,
                          initialize=to_flex_consumption[FlexConsumptionName].groupby(FlexName).max().to_dict(),
                          domain=Any)
    model.labour_ratio=Param(model.TIME_FLEX_CONSUM,default=0,
                          initialize=labour_ratio.loc[:,LaborRatioName].squeeze().to_dict(),
                          domain=Any)
    model.H2_consumption=Param(model.TIME,default=0,initialize=H2_consumption.loc[:,H2ConsumptionName])

    # with test of existing columns on TechParameters
    for COLNAME in TechParameters:
        if COLNAME not in [TechName, "AREAS"]:  ### each column in TechParameters will be a parameter
            exec("model." + COLNAME + " =          Param(model.TECHNOLOGIES, domain=Any,default=0," +
                 "initialize=TechParameters." + COLNAME + ".squeeze().to_dict())")

    for COLNAME in StorageParameters:
        if COLNAME not in [StockTechName, "AREAS"]:  ### each column in TechParameters will be a parameter
            exec("model." + COLNAME + " =          Param(model.STOCK_TECHNO, domain=Any,default=0," +
                 "initialize=StorageParameters." + COLNAME + ".squeeze().to_dict())")

    for COLNAME in FlexParameters:
        if COLNAME not in [FlexName, "AREAS"]:  ### each column in TechParameters will be a parameter
            exec("model." + COLNAME + " =          Param(model.FLEX_CONSUM, domain=Any,default=0," +
                 "initialize=FlexParameters." + COLNAME + ".squeeze().to_dict())")


    ################
    # Variables    #
    ################

    # In this section, variables are separated in two categories : decision variables wich are the reals variables of the otimisation problem (these are noted Dvar), and problem variables which are resulting of calculation and are convenient for the readability and the analyse of results (these are noted Pvar)

    ### Operation variables
    model.power_Dvar = Var(model.TIME, model.TECHNOLOGIES,
                               domain=NonNegativeReals)  ### Power of a conversion mean at time t
    model.energy_Pvar =Var(model.TIME, domain=NonNegativeReals) ### Total power produced at time t
    model.powerCosts_Pvar = Var(
            model.TECHNOLOGIES)  ### Marginal cost for a conversion mean, explicitely defined by definition powerCostsDef

    ### Planing variables
    model.capacity_Dvar = Var(model.TECHNOLOGIES, domain=NonNegativeReals)  ### Capacity of a conversion mean
    model.capacityCosts_Pvar = Var(
            model.TECHNOLOGIES)  ### Fixed costs for a conversion mean, explicitely defined by definition capacityCostsDef

    ### Storage variables
    model.storageIn_Pvar = Var(model.TIME, model.STOCK_TECHNO,
                                   domain=NonNegativeReals)  ### Energy stored in a storage mean at time t
    model.storageOut_Pvar = Var(model.TIME, model.STOCK_TECHNO,
                                    domain=NonNegativeReals)  ### Energy taken out of the in a storage mean at time t
    model.stockLevel_Pvar = Var(model.TIME, model.STOCK_TECHNO,
                                    domain=NonNegativeReals)  ### level of the energy stock in a storage mean at time t
    model.stockLevel_ini_Dvar=Var(model.STOCK_TECHNO,
                                    domain=NonNegativeReals) ### initial level of storage
    model.storageCosts_Pvar = Var(
            model.STOCK_TECHNO)  ### Cost of storage for a storage mean, explicitely defined by definition storageCostsDef
    model.Pmax_in_Dvar = Var(model.STOCK_TECHNO)  # Maximum flow of energy in a storage mean
    model.Pmax_out_Dvar = Var(model.STOCK_TECHNO)  # Maximum flow of energy out of a storage mean
    model.Cmax_Pvar = Var(model.STOCK_TECHNO)  # Maximum capacity of a storage mean

    ## Flexibility variables
    model.flexCosts_Pvar = Var(model.FLEX_CONSUM) # Costs of flexibility means
    model.total_consumption_Pvar = Var(model.TIME, domain=NonNegativeReals)  # variable de calcul intermediaire
    model.flex_consumption_Pvar = Var(model.TIME, model.FLEX_CONSUM,
                                 domain=NonNegativeReals)  # flexible consumption variable
    #Intermediary variables for labor cost calculation
    model.a_plus_Pvar = Var(model.TIME, model.FLEX_CONSUM, domain=NonNegativeReals)
    model.a_minus_Pvar = Var(model.TIME, model.FLEX_CONSUM, domain=NonNegativeReals)


    model.flex_Dvar = Var(model.TIME, model.FLEX_CONSUM, domain=Reals)#Flexibility variable

    model.increased_max_power_Dvar=Var(model.FLEX_CONSUM, domain=NonNegativeReals)

    ### Other variables
    model.dual = Suffix(direction=Suffix.IMPORT)
    model.rc = Suffix(direction=Suffix.IMPORT)
    model.slack = Suffix(direction=Suffix.IMPORT)

    ########################
    # Objective Function   #
    ########################

    def ObjectiveFunction_rule(model):  # OBJ
        return (sum(model.powerCosts_Pvar[tech] + model.capacityCosts_Pvar[tech] for tech in model.TECHNOLOGIES)\
                + sum(model.storageCosts_Pvar[s_tech] for s_tech in model.STOCK_TECHNO)\
                + sum(model.flexCosts_Pvar[flex] for flex in model.FLEX_CONSUM))

    model.OBJ = Objective(rule=ObjectiveFunction_rule, sense=minimize)

    #################
    # Constraints   #
    #################

    ### Costs definitions

    # energyCosts definition Constraint
    def powerCostsDef_rule(model,tech):
        return sum(model.energyCost[tech] * model.power_Dvar[t, tech]#+model.margvarCost[tech] * model.power_Dvar[t, tech]**2\
                   for t in model.TIME) == model.powerCosts_Pvar[tech]

    model.powerCostsCtr = Constraint(model.TECHNOLOGIES, rule=powerCostsDef_rule)

    # capacityCosts definition Constraint
    def capacityCostsDef_rule(model,tech):
        return model.capacityCost[tech] * model.capacity_Dvar[tech] == model.capacityCosts_Pvar[tech]

    model.capacityCostsCtr = Constraint(model.TECHNOLOGIES, rule=capacityCostsDef_rule)

    # storageCosts definition Constraint
    def storageCostsDef_rule(model,s_tech):
        return model.storageCost_power_in[s_tech] * model.Pmax_in_Dvar[s_tech]\
               + model.storageCost_power_out[s_tech] * model.Pmax_out_Dvar[s_tech]\
               + model.storageCost_energy[s_tech] * model.Cmax_Pvar[s_tech]== model.storageCosts_Pvar[s_tech]

    model.storageCostsCtr = Constraint(model.STOCK_TECHNO, rule=storageCostsDef_rule)

    # Felxibility cost definition
    def consumption_flex_costDef_rule(model, conso_type):
        return model.flexCosts_Pvar[conso_type] == model.LoadCost[conso_type] * model.increased_max_power_Dvar[
            conso_type] + sum(model.labour_ratio[t, conso_type] * model.labourcost[conso_type] * (
                model.a_plus_Pvar[t, conso_type] - model.a_minus_Pvar[t, conso_type]) for t in model.TIME)

    model.consumption_flex_costCtr = Constraint(model.FLEX_CONSUM, rule=consumption_flex_costDef_rule)

    # Storage max power constraint
    def storagePower_in_rule(model, s_tech):  # INEQ forall s_tech
        return model.Pmax_in_Dvar[s_tech] <= model.p_max_in[s_tech]

    model.storagePowerInCtr = Constraint(model.STOCK_TECHNO, rule=storagePower_in_rule)

    def storagePower_out_rule(model, s_tech):  # INEQ forall s_tech
        return model.Pmax_out_Dvar[s_tech] <= model.p_max_out[s_tech]

    model.storagePowerOutCtr = Constraint(model.STOCK_TECHNO, rule=storagePower_out_rule)

    def storagePower_in_out_rule(model, s_tech):
        if model.p_max_out_eq_p_max_in[s_tech]=='yes':
            return model.Pmax_in_Dvar[s_tech]==model.Pmax_out_Dvar[s_tech]
        else:
            return Constraint.Skip

    model.storagePowerInOutCtr = Constraint(model.STOCK_TECHNO, rule=storagePower_in_out_rule)

    # Storage max capacity constraint
    def storageCapacity_rule(model, s_tech):  # INEQ forall s_tech
        if model.h_max[s_tech]>0:
            return model.Cmax_Pvar[s_tech] <= model.h_max[s_tech]*model.Pmax_in_Dvar[s_tech]*model.efficiency_in[s_tech]
        else:
            return Constraint.Skip

    model.storageCapacityCtr = Constraint(model.STOCK_TECHNO, rule=storageCapacity_rule)

    # contraintes de stock puissance
    def StoragePowerUB_rule(model, t, s_tech):  # INEQ forall t
        return model.storageIn_Pvar[t, s_tech] - model.Pmax_in_Dvar[s_tech] <= 0

    model.StoragePowerUBCtr = Constraint(model.TIME, model.STOCK_TECHNO, rule=StoragePowerUB_rule)

    def StoragePowerLB_rule(model, t, s_tech, ):  # INEQ forall t
        return model.storageOut_Pvar[t, s_tech] - model.Pmax_out_Dvar[s_tech] <= 0

    model.StoragePowerLBCtr = Constraint(model.TIME, model.STOCK_TECHNO, rule=StoragePowerLB_rule)

    # contraintes de stock capacité
    def StockLevel_rule(model, t, s_tech):  # EQ forall t
        if s_tech!=PowertoH2toPowerName:
            if t>TIME_list[0]:
                return model.stockLevel_Pvar[t, s_tech] == model.stockLevel_Pvar[t - d1h, s_tech]*(1-model.dissipation[s_tech])\
                   + model.storageIn_Pvar[t, s_tech] * model.efficiency_in[s_tech] - model.storageOut_Pvar[t, s_tech]/model.efficiency_out[s_tech]
            else:
                return model.stockLevel_Pvar[t, s_tech] == model.storageIn_Pvar[t, s_tech] * model.efficiency_in[
                s_tech] - model.storageOut_Pvar[t, s_tech]/model.efficiency_out[s_tech]\
                   + model.stockLevel_ini_Dvar[s_tech]*(1-model.dissipation[s_tech])
        else:
            if t>TIME_list[0]:
                return model.stockLevel_Pvar[t, s_tech] == model.stockLevel_Pvar[t - d1h, s_tech]*(1-model.dissipation[s_tech])\
                   + model.storageIn_Pvar[t, s_tech] * model.efficiency_in[s_tech] - model.storageOut_Pvar[t, s_tech]/model.efficiency_out[s_tech]\
                   - model.H2_consumption[t]
            else:
                return model.stockLevel_Pvar[t, s_tech] == model.storageIn_Pvar[t, s_tech] * model.efficiency_in[
                s_tech] - model.storageOut_Pvar[t, s_tech]/model.efficiency_out[s_tech]\
                   + model.stockLevel_ini_Dvar[s_tech]*(1-model.dissipation[s_tech])\
                       - model.H2_consumption[t]

    model.StockLevelCtr = Constraint(model.TIME, model.STOCK_TECHNO, rule=StockLevel_rule)

    def StockLevel_ini_rule(model,s_tech):
        return model.stockLevel_Pvar[TIME_list[-1], s_tech]>=model.stockLevel_ini_Dvar[s_tech]

    model.StockLevelIniCtr = Constraint(model.STOCK_TECHNO, rule=StockLevel_ini_rule)

    def StockCapacity_rule(model, t, s_tech, ):  # INEQ forall t
        return model.stockLevel_Pvar[t, s_tech] <= model.Cmax_Pvar[s_tech]

    model.StockCapacityCtr = Constraint(model.TIME, model.STOCK_TECHNO, rule=StockCapacity_rule)

    ### Production constraints

    # Capacity constraint
    def Capacity_rule(model, t, tech):  # INEQ forall t, tech
        return model.capacity_Dvar[tech] * model.availabilityFactor[t, tech] >= model.power_Dvar[t, tech]

    model.CapacityCtr = Constraint(model.TIME, model.TECHNOLOGIES, rule=Capacity_rule)

    # Ressource production constraint
    def Production_rule(model, t):  # EQ forall t, res
        return sum(model.power_Dvar[t, tech] for tech in model.TECHNOLOGIES)\
               + sum(model.storageOut_Pvar[t, s_tech] - model.storageIn_Pvar[t, s_tech] for s_tech in STOCK_TECHNO) \
               == model.energy_Pvar[t]
               #+ model.H2_to_power_Pvar[t] - model.power_to_H2_Pvar[t]\


    model.ProductionCtr = Constraint(model.TIME, rule=Production_rule)

    # contrainte d'equilibre offre demande
    def energyCtr_rule(model, t):  # INEQ forall t
        return model.energy_Pvar[t] == model.total_consumption_Pvar[t]

    model.energyCtr = Constraint(model.TIME, rule=energyCtr_rule)

    if "maxCapacity" in TechParameters:
        def maxCapacity_rule(model, tech):  # INEQ forall t, tech
            return model.maxCapacity[tech] >= model.capacity_Dvar[tech]

        model.maxCapacityCtr = Constraint(model.TECHNOLOGIES, rule=maxCapacity_rule)

    if "minCapacity" in TechParameters:
        def minCapacity_rule(model, tech):  # INEQ forall t, tech
            return model.minCapacity[tech] <= model.capacity_Dvar[tech]

        model.minCapacityCtr = Constraint(model.TECHNOLOGIES, rule=minCapacity_rule)

    if "EnergyNbhourCap" in TechParameters:
        def storage_rule(model, tech):  # INEQ forall t, tech
            if model.EnergyNbhourCap[tech] > 0:
                return model.EnergyNbhourCap[tech] * model.capacity_Dvar[tech] >= sum(
                    model.power_Dvar[t, tech] for t in model.TIME)
            else:
                return Constraint.Skip

        model.storageCtr = Constraint(model.TECHNOLOGIES, rule=storage_rule)

    ### Ramp constraints

    if "RampConstraintPlus" in TechParameters:
        def rampCtrPlus_rule(model, t, tech):  # INEQ forall t<
            if model.RampConstraintPlus[tech] > 0:
                return model.power_Dvar[t + d1h, tech] - model.power_Dvar[t, tech] <= model.capacity_Dvar[tech] * \
                       model.RampConstraintPlus[tech];
            else:
                return Constraint.Skip

        model.rampCtrPlus = Constraint(model.TIME_MinusOne, model.TECHNOLOGIES, rule=rampCtrPlus_rule)

    if "RampConstraintMoins" in TechParameters:
        def rampCtrMoins_rule(model, t, tech):  # INEQ forall t<
            if model.RampConstraintMoins[tech] > 0:
                return model.power_Dvar[t + d1h, tech] - model.power_Dvar[t, tech] >= - model.capacity_Dvar[tech] * \
                       model.RampConstraintMoins[tech];
            else:
                return Constraint.Skip

        model.rampCtrMoins = Constraint(model.TIME_MinusOne, model.TECHNOLOGIES, rule=rampCtrMoins_rule)

    if "RampConstraintPlus2" in TechParameters:
        def rampCtrPlus2_rule(model, t, tech):  # INEQ forall t<
            if model.RampConstraintPlus2[tech] > 0:
                var = (model.power_Dvar[t + 2*d1h, tech] + model.power_Dvar[t + 3*d1h, tech]) / 2 - (
                            model.power_Dvar[t + d1h, tech] + model.power_Dvar[t, tech]) / 2;
                return var <= model.capacity_Dvar[tech] * model.RampConstraintPlus[tech];
            else:
                return Constraint.Skip

        model.rampCtrPlus2 = Constraint(model.TIME_MinusThree, model.TECHNOLOGIES, rule=rampCtrPlus2_rule)

    if "RampConstraintMoins2" in TechParameters:
        def rampCtrMoins2_rule(model, t, tech):  # INEQ forall t<
            if model.RampConstraintMoins2[tech] > 0:
                var = (model.power_Dvar[t + 2*d1h, tech] + model.power_Dvar[t + 3*d1h, tech]) / 2 - (
                            model.power_Dvar[t + d1h, tech] + model.power_Dvar[t, tech]) / 2;
                return var >= - model.capacity_Dvar[tech] * model.RampConstraintMoins2[tech];
            else:
                return Constraint.Skip

        model.rampCtrMoins2 = Constraint(model.TIME_MinusThree, model.TECHNOLOGIES, rule=rampCtrMoins2_rule)

    ### Flexibility constraints

    # Total consumtion rule
    def total_consumption_rule(model, t):
        return model.total_consumption_Pvar[t] == (model.areaConsumption[t] + sum(
            model.flex_consumption_Pvar[t, name] for name in model.FLEX_CONSUM))*(1+model.lossesRate[t])

    model.total_consumptionCtr = Constraint(model.TIME, rule=total_consumption_rule)

    # Max power rule
    def max_power_rule(model, conso_type, t):
        if model.flex_type[conso_type] != 'day_ev':
            return model.max_power[conso_type] + model.increased_max_power_Dvar[conso_type] >= model.flex_consumption_Pvar[t, conso_type]
        else:
            if TIME_df.loc[t,'daytime_ev']==1:
                return model.increased_max_power_Dvar[conso_type]>=model.a_plus_Pvar[t, conso_type]
            else:
                return model.max_power[conso_type] >= model.flex_consumption_Pvar[t, conso_type]

    model.max_powerCtr = Constraint(model.FLEX_CONSUM, model.TIME, rule=max_power_rule)

    # Day constraint
    def consum_eq_day(model, t_day, conso_type):
        if model.flex_type[conso_type] == 'day':
            t_range = TIME_df[TIME_df.day == t_day].index.get_level_values(TimeName)
            return sum(model.flex_consumption[t, conso_type] for t in t_range) == sum(
                model.to_flex_consumption[t, conso_type] for t in t_range)
        else:
            return Constraint.Skip

    model.consum_eq_day_Ctr = Constraint(model.DAY, model.FLEX_CONSUM, rule=consum_eq_day)

    # Morning and night constraint electric vehicle
    def consum_eq_day_ev(model, t_day, conso_type):
        if model.flex_type[conso_type] == 'day_ev':
            t_range = TIME_df[(TIME_df.day_ev == t_day)].index.get_level_values(TimeName)
            return sum(model.flex_consumption_Pvar[t, conso_type] for t in t_range) == sum(
                model.to_flex_consumption[t, conso_type] for t in t_range)
        else:
            return Constraint.Skip

    model.consum_eq_day_ev_Ctr = Constraint(model.DAY_EV, model.FLEX_CONSUM, rule=consum_eq_day_ev)

    # Week constraint
    def consum_eq_week(model, t_week, conso_type):
        if model.flex_type[conso_type] == 'week':
            t_range = TIME_df[TIME_df.week == t_week].index.get_level_values(TimeName)
            return sum(model.flex_consumption_Pvar[t, conso_type] for t in t_range) == sum(
                model.to_flex_consumption[t, conso_type] for t in t_range)
        else:
            return Constraint.Skip

    model.consum_eq_week_Ctr = Constraint(model.WEEK, model.FLEX_CONSUM, rule=consum_eq_week)

    # Year constraint
    def consum_eq_year(model, conso_type):
        if model.flex_type[conso_type] == 'year':
            return sum(model.flex_consumption_Pvar[t, conso_type] for t in model.TIME) == sum(
                model.to_flex_consumption[t, conso_type] for t in model.TIME)
        else:
            return Constraint.Skip
    model.consum_eq_year_Ctr = Constraint(model.FLEX_CONSUM, rule=consum_eq_year)

    # Flex constraint
    def consum_flex_rule(model, t, conso_type):
        return model.flex_consumption_Pvar[t, conso_type] == model.to_flex_consumption[t, conso_type] * (
                1 - model.flex_Dvar[t, conso_type])

    model.consum_flex_Ctr = Constraint(model.TIME, model.FLEX_CONSUM, rule=consum_flex_rule)

    # Flex ratio constraint
    def flex_variation_sup_rule(model, t, conso_type):
        if model.flex_type[conso_type]!= 'day_ev' or TIME_df.loc[t,'ctr_ev']==1:
            return model.flex_Dvar[t, conso_type] <= model.flex_ratio[conso_type]
        else:
            return Constraint.Skip

    model.flex_variation_sup_Ctr = Constraint(model.TIME, model.FLEX_CONSUM, rule=flex_variation_sup_rule)

    def flex_variation_inf_rule(model, t, conso_type):
        if model.flex_type[conso_type] != 'day_ev' or TIME_df.loc[t, 'ctr_ev'] == 1:
            return model.flex_Dvar[t, conso_type] >= -model.flex_ratio[conso_type]
        else:
            return Constraint.Skip

    model.flex_variation_inf_Ctr = Constraint(model.TIME, model.FLEX_CONSUM, rule=flex_variation_inf_rule)

    # Definition a_plus, a_minus
    def a_plus_minus_rule(model, conso_type, t):
        return model.flex_consumption_Pvar[t, conso_type] - model.to_flex_consumption[t, conso_type] == \
               model.a_plus_Pvar[t, conso_type] - model.a_minus_Pvar[t, conso_type]

    model.a_plus_minusCtr = Constraint(model.FLEX_CONSUM, model.TIME, rule=a_plus_minus_rule)

    return model


def GetElectricSystemModel_Belfort_MultiNode(areaConsumption, lossesRate, availabilityFactor,
                                                 TechParameters, StorageParameters,
                                                 to_flex_consumption, FlexParameters, labour_ratio,
                                                 H2_consumption,ExchangeParameters,
                                                 TimeName='Date', TechName='TECHNOLOGIES',
                                                 StockTechName='STOCK_TECHNO', FlexName='FLEX_CONSUM',
                                                 PowertoH2toPowerName='PowertoH2toPower',
                                                 FlexConsumptionName='Consumption',
                                                 areaConsumptionName='areaConsumption',
                                                 availabilityFactorName='availabilityFactor',
                                                 LaborRatioName='labour_ratio', lossesRateName='Taux_pertes',
                                                 H2ConsumptionName='H2',
                                                 AreaName='Area'):
    ## Generic variables
    d1h = timedelta(hours=1)

    ## Cleaning
    availabilityFactor = availabilityFactor.fillna(method='pad');
    areaConsumption = areaConsumption.fillna(method='pad');
    to_flex_consumption = to_flex_consumption.fillna(method='pad');

    ### obtaining dimensions values

    TECHNOLOGIES = set(TechParameters.index.get_level_values(TechName).unique())
    STOCK_TECHNO = set(StorageParameters.index.get_level_values(StockTechName).unique())
    FLEX_CONSUM = set(FlexParameters.index.get_level_values(FlexName).unique())
    AREAS = set(areaConsumption.index.get_level_values(AreaName).unique())
    TIME = set(areaConsumption.index.get_level_values(TimeName).unique())
    TIME_list = areaConsumption.index.get_level_values(TimeName).unique()

    TIME_df = pd.DataFrame({TimeName: TIME_list})
    TIME_df['day'] = TIME_df[TimeName].apply(lambda x: x.dayofyear)
    TIME_df['day_ev'] = TIME_df[TimeName].apply(lambda x: (x - 12 * d1h).dayofyear)
    TIME_df['daytime_ev'] = TIME_df[TimeName].apply(lambda x: 1 if 8 <= x.hour <= 17 else 0)  # 1 if day
    TIME_df['night_ev'] = TIME_df[TimeName].apply(lambda x: 1 if 18 <= x.hour or x.hour <= 7 else 0)  # 1 if night
    TIME_df['ctr_ev'] = TIME_df[TimeName].apply(
        lambda x: 1 if 7 <= x.hour <= 8 or 17 <= x.hour <= 18 else 0)  # flex constrained time for electric vehicles
    TIME_df['week'] = TIME_df[TimeName].apply(lambda x: x.isocalendar().week)
    TIME_df = TIME_df.set_index(TimeName)

    DAY = set(TIME_df['day'].unique())
    DAY_EV = set(TIME_df['day_ev'].unique())
    WEEK = set(TIME_df['week'].unique())

    ### model definition

    model = ConcreteModel()

    ###############
    # Sets       ##
    ###############
    model.TECHNOLOGIES = Set(initialize=TECHNOLOGIES, ordered=False)
    model.TIME = Set(initialize=TIME, ordered=False)
    model.STOCK_TECHNO = Set(initialize=STOCK_TECHNO, ordered=False)
    model.FLEX_CONSUM = Set(initialize=FLEX_CONSUM, ordered=False)
    model.AREAS = Set(initialize=AREAS, ordered=False)
    #model.TIME_TECHNOLOGIES = model.TIME * model.TECHNOLOGIES
    #model.TIME_STOCK_TECHNO = model.TIME * model.STOCK_TECHNO
    #model.TIME_FLEX_CONSUM = model.TIME * model.FLEX_CONSUM

    model.DAY = Set(initialize=DAY, ordered=False)
    model.DAY_EV = Set(initialize=DAY_EV, ordered=False)
    model.WEEK = Set(initialize=WEEK, ordered=False)

    ## For ramp constraints
    model.TIME_MinusOne = Set(initialize=TIME_list[: len(TIME) - 1], ordered=False)
    model.TIME_MinusThree = Set(initialize=TIME_list[: len(TIME) - 3], ordered=False)

    ###############
    # Parameters ##
    ###############

    model.areaConsumption = Param(model.AREAS, model.TIME, default=0,
                                  initialize=areaConsumption.loc[:, areaConsumptionName].squeeze().to_dict(),
                                  domain=Any)
    model.lossesRate = Param(model.AREAS, model.TIME, default=0,
                             initialize=lossesRate.loc[:, lossesRateName].squeeze().to_dict(), domain=PercentFraction)
    model.availabilityFactor = Param(model.AREAS, model.TIME, model.TECHNOLOGIES, domain=PercentFraction, default=1,
                                     initialize=availabilityFactor.loc[:, availabilityFactorName].squeeze().to_dict())
    model.to_flex_consumption = Param(model.AREAS, model.TIME, model.FLEX_CONSUM, default=0,
                                      initialize=to_flex_consumption.loc[:, FlexConsumptionName].squeeze().to_dict(),
                                      domain=Any)
    model.max_power = Param(model.AREAS, model.FLEX_CONSUM, default=0,
                            initialize=to_flex_consumption[FlexConsumptionName].groupby([AreaName,FlexName]).max().to_dict(),
                            domain=Any)
    model.labour_ratio = Param(model.AREAS,model.TIME,model.FLEX_CONSUM, default=0,
                               initialize=labour_ratio.loc[:, LaborRatioName].squeeze().to_dict(),
                               domain=Any)
    model.H2_consumption = Param(model.AREAS,model.TIME, default=0, initialize=H2_consumption.loc[:, H2ConsumptionName])

    # with test of existing columns on TechParameters
    for COLNAME in TechParameters:
        if COLNAME not in [TechName, "AREAS"]:  ### each column in TechParameters will be a parameter
            exec("model." + COLNAME + " =          Param(model.AREAS, model.TECHNOLOGIES, domain=Any,default=0," +
                 "initialize=TechParameters." + COLNAME + ".squeeze().to_dict())")

    for COLNAME in StorageParameters:
        if COLNAME not in [StockTechName, "AREAS"]:  ### each column in TechParameters will be a parameter
            exec("model." + COLNAME + " =          Param(model.AREAS, model.STOCK_TECHNO, domain=Any,default=0," +
                 "initialize=StorageParameters." + COLNAME + ".squeeze().to_dict())")

    for COLNAME in FlexParameters:
        if COLNAME not in [FlexName, "AREAS"]:  ### each column in TechParameters will be a parameter
            exec("model." + COLNAME + " =          Param(model.AREAS, model.FLEX_CONSUM, domain=Any,default=0," +
                 "initialize=FlexParameters." + COLNAME + ".squeeze().to_dict())")

    ################
    # Variables    #
    ################

    # In this section, variables are separated in two categories : decision variables wich are the reals variables of the otimisation problem (these are noted Dvar), and problem variables which are resulting of calculation and are convenient for the readability and the analyse of results (these are noted Pvar)

    ### Operation variables
    model.power_Dvar = Var(model.AREAS, model.TIME, model.TECHNOLOGIES,
                           domain=NonNegativeReals)  ### Power of a conversion mean at time t
    model.exchange_Pvar = Var(model.AREAS, model.AREAS, model.TIME) ### exchange[a,b,t] is the energy flowing from area a to area b at time t (negative if energy flowing from b to a)
    model.powerCosts_Pvar = Var(
        model.AREAS, model.TECHNOLOGIES)  ### Marginal cost for a conversion mean, explicitely defined by definition powerCostsDef

    ### Planing variables
    model.capacity_Dvar = Var(model.AREAS, model.TECHNOLOGIES, domain=NonNegativeReals)  ### Capacity of a conversion mean
    model.capacityCosts_Pvar = Var(
        model.AREAS, model.TECHNOLOGIES)  ### Fixed costs for a conversion mean, explicitely defined by definition capacityCostsDef

    ### Storage variables
    model.storageIn_Pvar = Var(model.AREAS, model.TIME, model.STOCK_TECHNO,
                               domain=NonNegativeReals)  ### Energy stored in a storage mean at time t
    model.storageOut_Pvar = Var(model.AREAS, model.TIME, model.STOCK_TECHNO,
                                domain=NonNegativeReals)  ### Energy taken out of the in a storage mean at time t
    model.stockLevel_Pvar = Var(model.AREAS, model.TIME, model.STOCK_TECHNO,
                                domain=NonNegativeReals)  ### level of the energy stock in a storage mean at time t
    model.stockLevel_ini_Dvar = Var(model.AREAS, model.STOCK_TECHNO,
                                    domain=NonNegativeReals)  ### initial level of storage
    model.storageCosts_Pvar = Var(model.AREAS,
        model.STOCK_TECHNO)  ### Cost of storage for a storage mean, explicitely defined by definition storageCostsDef
    model.Pmax_in_Dvar = Var(model.AREAS, model.STOCK_TECHNO)  # Maximum flow of energy in a storage mean
    model.Pmax_out_Dvar = Var(model.AREAS, model.STOCK_TECHNO)  # Maximum flow of energy out of a storage mean
    model.Cmax_Pvar = Var(model.AREAS, model.STOCK_TECHNO)  # Maximum capacity of a storage mean

    ## Flexibility variables
    model.flexCosts_Pvar = Var(model.AREAS, model.FLEX_CONSUM)  # Costs of flexibility means
    model.total_consumption_Pvar = Var(model.AREAS, model.TIME, domain=NonNegativeReals)  # variable de calcul intermediaire
    model.flex_consumption_Pvar = Var(model.AREAS, model.TIME, model.FLEX_CONSUM,
                                      domain=NonNegativeReals)  # flexible consumption variable
    # Intermediary variables for labor cost calculation
    model.a_plus_Pvar = Var(model.AREAS, model.TIME, model.FLEX_CONSUM, domain=NonNegativeReals)
    model.a_minus_Pvar = Var(model.AREAS, model.TIME, model.FLEX_CONSUM, domain=NonNegativeReals)

    model.flex_Dvar = Var(model.AREAS, model.TIME, model.FLEX_CONSUM, domain=Reals)  # Flexibility variable

    model.increased_max_power_Dvar = Var(model.AREAS, model.FLEX_CONSUM, domain=NonNegativeReals)

    ### Other variables
    model.dual = Suffix(direction=Suffix.IMPORT)
    model.rc = Suffix(direction=Suffix.IMPORT)
    model.slack = Suffix(direction=Suffix.IMPORT)

    ########################
    # Objective Function   #
    ########################

    def ObjectiveFunction_rule(model):  # OBJ
        return (sum(model.powerCosts_Pvar[area,tech] + model.capacityCosts_Pvar[area, tech] for area in model.AREAS for tech in model.TECHNOLOGIES) \
                + sum(model.storageCosts_Pvar[area,s_tech] for area in model.AREAS for s_tech in model.STOCK_TECHNO) \
                + sum(model.flexCosts_Pvar[area,flex] for area in model.AREAS for flex in model.FLEX_CONSUM))

    model.OBJ = Objective(rule=ObjectiveFunction_rule, sense=minimize)

    #################
    # Constraints   #
    #################

    ### Costs definitions

    # energyCosts definition Constraint
    def powerCostsDef_rule(model,area, tech):
        return sum(model.energyCost[area,tech] * model.power_Dvar[area,t, tech]
                   # +model.margvarCost[tech] * model.power_Dvar[t, tech]**2\
                   for t in model.TIME) == model.powerCosts_Pvar[area,tech]

    model.powerCostsCtr = Constraint(model.AREAS,model.TECHNOLOGIES, rule=powerCostsDef_rule)

    # capacityCosts definition Constraint
    def capacityCostsDef_rule(model, area, tech):
        return model.capacityCost[area, tech] * model.capacity_Dvar[area, tech] == model.capacityCosts_Pvar[area, tech]

    model.capacityCostsCtr = Constraint(model.AREAS, model.TECHNOLOGIES, rule=capacityCostsDef_rule)

    # storageCosts definition Constraint
    def storageCostsDef_rule(model, area, s_tech):
        return model.storageCost_power_in[area, s_tech] * model.Pmax_in_Dvar[area, s_tech] \
               + model.storageCost_power_out[area, s_tech] * model.Pmax_out_Dvar[area, s_tech] \
               + model.storageCost_energy[area, s_tech] * model.Cmax_Pvar[area, s_tech] == model.storageCosts_Pvar[area, s_tech]

    model.storageCostsCtr = Constraint(model.AREAS, model.STOCK_TECHNO, rule=storageCostsDef_rule)

    # Felxibility cost definition
    def consumption_flex_costDef_rule(model, area, conso_type):
        return model.flexCosts_Pvar[area, conso_type] == model.LoadCost[area, conso_type] * model.increased_max_power_Dvar[
            area, conso_type] + sum(model.labour_ratio[area, t, conso_type] * model.labourcost[area, conso_type] * (
                model.a_plus_Pvar[area, t, conso_type] - model.a_minus_Pvar[area, t, conso_type]) for t in model.TIME)

    model.consumption_flex_costCtr = Constraint(model.AREAS, model.FLEX_CONSUM, rule=consumption_flex_costDef_rule)

    ### Storage constraints

    # Storage max power constraint
    def storagePower_in_rule(model, area, s_tech):  # INEQ forall s_tech
        return model.Pmax_in_Dvar[area, s_tech] <= model.p_max_in[area, s_tech]

    model.storagePowerInCtr = Constraint(model.AREAS, model.STOCK_TECHNO, rule=storagePower_in_rule)

    def storagePower_out_rule(model, area, s_tech):  # INEQ forall s_tech
        return model.Pmax_out_Dvar[area, s_tech] <= model.p_max_out[area, s_tech]

    model.storagePowerOutCtr = Constraint(model.AREAS, model.STOCK_TECHNO, rule=storagePower_out_rule)

    def storagePower_in_out_rule(model, area, s_tech):
        if model.p_max_out_eq_p_max_in[area, s_tech] == 'yes':
            return model.Pmax_in_Dvar[area, s_tech] == model.Pmax_out_Dvar[area, s_tech]
        else:
            return Constraint.Skip

    model.storagePowerInOutCtr = Constraint(model.AREAS, model.STOCK_TECHNO, rule=storagePower_in_out_rule)

    # Storage max capacity constraint
    def storageCapacity_rule(model, area, s_tech):  # INEQ forall s_tech
        if model.h_max[area, s_tech] > 0:
            return model.Cmax_Pvar[area, s_tech] <= model.h_max[area, s_tech] * model.Pmax_in_Dvar[area, s_tech] * model.efficiency_in[
                area, s_tech]
        else:
            return Constraint.Skip

    model.storageCapacityCtr = Constraint(model.AREAS, model.STOCK_TECHNO, rule=storageCapacity_rule)

    # contraintes de stock puissance
    def StoragePowerUB_rule(model, area, t, s_tech):  # INEQ forall t
        return model.storageIn_Pvar[area, t, s_tech] - model.Pmax_in_Dvar[area, s_tech] <= 0

    model.StoragePowerUBCtr = Constraint(model.AREAS, model.TIME, model.STOCK_TECHNO, rule=StoragePowerUB_rule)

    def StoragePowerLB_rule(model, area, t, s_tech, ):  # INEQ forall t
        return model.storageOut_Pvar[area, t, s_tech] - model.Pmax_out_Dvar[area, s_tech] <= 0

    model.StoragePowerLBCtr = Constraint(model.AREAS, model.TIME, model.STOCK_TECHNO, rule=StoragePowerLB_rule)

    # contraintes de stock capacité
    def StockLevel_rule(model, area, t, s_tech):  # EQ forall t
        if s_tech != PowertoH2toPowerName:
            if t > TIME_list[0]:
                return model.stockLevel_Pvar[area, t, s_tech] == model.stockLevel_Pvar[area, t - d1h, s_tech] * (
                            1 - model.dissipation[area, s_tech]) \
                       + model.storageIn_Pvar[area, t, s_tech] * model.efficiency_in[area, s_tech] - model.storageOut_Pvar[
                           area, t, s_tech] / model.efficiency_out[area, s_tech]
            else:
                return model.stockLevel_Pvar[area, t, s_tech] == model.storageIn_Pvar[area, t, s_tech] * model.efficiency_in[
                    area, s_tech] - model.storageOut_Pvar[area, t, s_tech] / model.efficiency_out[area, s_tech] \
                       + model.stockLevel_ini_Dvar[area, s_tech] * (1 - model.dissipation[area, s_tech])
        else:
            if t > TIME_list[0]:
                return model.stockLevel_Pvar[area, t, s_tech] == model.stockLevel_Pvar[area, t - d1h, s_tech] * (
                            1 - model.dissipation[area, s_tech]) \
                       + model.storageIn_Pvar[area, t, s_tech] * model.efficiency_in[area, s_tech] - model.storageOut_Pvar[
                           area, t, s_tech] / model.efficiency_out[area, s_tech] \
                       - model.H2_consumption[area, t]
            else:
                return model.stockLevel_Pvar[area, t, s_tech] == model.storageIn_Pvar[area, t, s_tech] * model.efficiency_in[
                    area, s_tech] - model.storageOut_Pvar[area, t, s_tech] / model.efficiency_out[area, s_tech] \
                       + model.stockLevel_ini_Dvar[area, s_tech] * (1 - model.dissipation[area, s_tech]) \
                       - model.H2_consumption[area, t]

    model.StockLevelCtr = Constraint(model.AREAS, model.TIME, model.STOCK_TECHNO, rule=StockLevel_rule)

    def StockLevel_ini_rule(model, area, s_tech):
        return model.stockLevel_Pvar[TIME_list[-1], area, s_tech] >= model.stockLevel_ini_Dvar[area, s_tech]

    model.StockLevelIniCtr = Constraint(model.AREAS,model.STOCK_TECHNO, rule=StockLevel_ini_rule)

    def StockCapacity_rule(model, area, t, s_tech, ):  # INEQ forall t
        return model.stockLevel_Pvar[area, t, s_tech] <= model.Cmax_Pvar[area, s_tech]

    model.StockCapacityCtr = Constraint(model.AREAS, model.TIME, model.STOCK_TECHNO, rule=StockCapacity_rule)

    ### Production constraints

    # Capacity constraint
    def Capacity_rule(model, area, t, tech):  # INEQ forall t, tech
        return model.capacity_Dvar[area, tech] * model.availabilityFactor[area, t, tech] >= model.power_Dvar[area, t, tech]

    model.CapacityCtr = Constraint(model.AREAS, model.TIME, model.TECHNOLOGIES, rule=Capacity_rule)

    # contrainte d'equilibre offre demande
    def Production_rule(model, area, t):  # EQ forall t, res
        return sum(model.power_Dvar[area, t, tech] for tech in model.TECHNOLOGIES) \
               + sum(model.storageOut_Pvar[area, t, s_tech] - model.storageIn_Pvar[area, t, s_tech] for s_tech in STOCK_TECHNO) \
               + sum(model.exchange_Pvar[b, area, t] for b in model.AREAS)\
               == model.total_consumption_Pvar[t]

    model.ProductionCtr = Constraint(model.AREAS, model.TIME, rule=Production_rule)

    if "maxCapacity" in TechParameters:
        def maxCapacity_rule(model, area, tech):  # INEQ forall t, tech
            return model.maxCapacity[area, tech] >= model.capacity_Dvar[area, tech]

        model.maxCapacityCtr = Constraint(model.AREAS, model.TECHNOLOGIES, rule=maxCapacity_rule)

    if "minCapacity" in TechParameters:
        def minCapacity_rule(model, area, tech):  # INEQ forall t, tech
            return model.minCapacity[area, tech] <= model.capacity_Dvar[area, tech]

        model.minCapacityCtr = Constraint(model.AREAS, model.TECHNOLOGIES, rule=minCapacity_rule)

    if "EnergyNbhourCap" in TechParameters:
        def storage_rule(model, area, tech):  # INEQ forall t, tech
            if model.EnergyNbhourCap[area, tech] > 0:
                return model.EnergyNbhourCap[area, tech] * model.capacity_Dvar[area, tech] >= sum(
                    model.power_Dvar[area, t, tech] for t in model.TIME)
            else:
                return Constraint.Skip

        model.storageCtr = Constraint(model.AREAS, model.TECHNOLOGIES, rule=storage_rule)

    ### Ramp constraints

    if "RampConstraintPlus" in TechParameters:
        def rampCtrPlus_rule(model, area, t, tech):  # INEQ forall t<
            if model.RampConstraintPlus[area, tech] > 0:
                return model.power_Dvar[area, t + d1h, tech] - model.power_Dvar[area, t, tech] <= model.capacity_Dvar[area, tech] * \
                       model.RampConstraintPlus[area, tech];
            else:
                return Constraint.Skip

        model.rampCtrPlus = Constraint(model.AREAS, model.TIME_MinusOne, model.TECHNOLOGIES, rule=rampCtrPlus_rule)

    if "RampConstraintMoins" in TechParameters:
        def rampCtrMoins_rule(model, area, t, tech):  # INEQ forall t<
            if model.RampConstraintMoins[area, tech] > 0:
                return model.power_Dvar[area, t + d1h, tech] - model.power_Dvar[area, t, tech] >= - model.capacity_Dvar[area, tech] * \
                       model.RampConstraintMoins[area, tech];
            else:
                return Constraint.Skip

        model.rampCtrMoins = Constraint(model.AREAS, model.TIME_MinusOne, model.TECHNOLOGIES, rule=rampCtrMoins_rule)

    if "RampConstraintPlus2" in TechParameters:
        def rampCtrPlus2_rule(model, area, t, tech):  # INEQ forall t<
            if model.RampConstraintPlus2[area, tech] > 0:
                var = (model.power_Dvar[area, t + 2 * d1h, tech] + model.power_Dvar[area, t + 3 * d1h, tech]) / 2 - (
                        model.power_Dvar[area, t + d1h, tech] + model.power_Dvar[area, t, tech]) / 2;
                return var <= model.capacity_Dvar[area, tech] * model.RampConstraintPlus[area, tech];
            else:
                return Constraint.Skip

        model.rampCtrPlus2 = Constraint(model.AREAS, model.TIME_MinusThree, model.TECHNOLOGIES, rule=rampCtrPlus2_rule)

    if "RampConstraintMoins2" in TechParameters:
        def rampCtrMoins2_rule(model, area, t, tech):  # INEQ forall t<
            if model.RampConstraintMoins2[area, tech] > 0:
                var = (model.power_Dvar[area, t + 2 * d1h, tech] + model.power_Dvar[area, t + 3 * d1h, tech]) / 2 - (
                        model.power_Dvar[area, t + d1h, tech] + model.power_Dvar[area, t, tech]) / 2;
                return var >= - model.capacity_Dvar[area, tech] * model.RampConstraintMoins2[area, tech];
            else:
                return Constraint.Skip

        model.rampCtrMoins2 = Constraint(model.AREAS, model.TIME_MinusThree, model.TECHNOLOGIES, rule=rampCtrMoins2_rule)

    ### Flexibility constraints

    # Total consumtion rule
    def total_consumption_rule(model, area, t):
        return model.total_consumption_Pvar[area, t] == (model.areaConsumption[area, t] + sum(
            model.flex_consumption_Pvar[area, t, name] for name in model.FLEX_CONSUM)) * (1 + model.lossesRate[area, t])

    model.total_consumptionCtr = Constraint(model.AREAS, model.TIME, rule=total_consumption_rule)

    # Max power rule
    def max_power_rule(model, area, conso_type, t):
        if model.flex_type[area, conso_type] != 'day_ev':
            return model.max_power[area, conso_type] + model.increased_max_power_Dvar[area, conso_type] >= \
                   model.flex_consumption_Pvar[area, t, conso_type]
        else:
            if TIME_df.loc[t, 'daytime_ev'] == 1:
                return model.increased_max_power_Dvar[area, conso_type] >= model.a_plus_Pvar[area, t, conso_type]
            else:
                return model.max_power[area, conso_type] >= model.flex_consumption_Pvar[area, t, conso_type]

    model.max_powerCtr = Constraint(model.AREAS, model.FLEX_CONSUM, model.TIME, rule=max_power_rule)

    # Day constraint
    def consum_eq_day(model, area, t_day, conso_type):
        if model.flex_type[area, conso_type] == 'day':
            t_range = TIME_df[TIME_df.day == t_day].index.get_level_values(TimeName)
            return sum(model.flex_consumption[area, t, conso_type] for t in t_range) == sum(
                model.to_flex_consumption[area, t, conso_type] for t in t_range)
        else:
            return Constraint.Skip

    model.consum_eq_day_Ctr = Constraint(model.AREAS, model.DAY, model.FLEX_CONSUM, rule=consum_eq_day)

    # Morning and night constraint electric vehicle
    def consum_eq_day_ev(model, area, t_day, conso_type):
        if model.flex_type[area, conso_type] == 'day_ev':
            t_range = TIME_df[(TIME_df.day_ev == t_day)].index.get_level_values(TimeName)
            return sum(model.flex_consumption_Pvar[area, t, conso_type] for t in t_range) == sum(
                model.to_flex_consumption[area, t, conso_type] for t in t_range)
        else:
            return Constraint.Skip

    model.consum_eq_day_ev_Ctr = Constraint(model.AREAS, model.DAY_EV, model.FLEX_CONSUM, rule=consum_eq_day_ev)

    # Week constraint
    def consum_eq_week(model, area, t_week, conso_type):
        if model.flex_type[area, conso_type] == 'week':
            t_range = TIME_df[TIME_df.week == t_week].index.get_level_values(TimeName)
            return sum(model.flex_consumption_Pvar[area, t, conso_type] for t in t_range) == sum(
                model.to_flex_consumption[area, t, conso_type] for t in t_range)
        else:
            return Constraint.Skip

    model.consum_eq_week_Ctr = Constraint(model.AREAS, model.WEEK, model.FLEX_CONSUM, rule=consum_eq_week)

    # Year constraint
    def consum_eq_year(model, area, conso_type):
        if model.flex_type[area, conso_type] == 'year':
            return sum(model.flex_consumption_Pvar[area, t, conso_type] for t in model.TIME) == sum(
                model.to_flex_consumption[area, t, conso_type] for t in model.TIME)
        else:
            return Constraint.Skip

    model.consum_eq_year_Ctr = Constraint(model.AREAS, model.FLEX_CONSUM, rule=consum_eq_year)

    # Flex constraint
    def consum_flex_rule(model, area, t, conso_type):
        return model.flex_consumption_Pvar[area, t, conso_type] == model.to_flex_consumption[area, t, conso_type] * (
                1 - model.flex_Dvar[area, t, conso_type])

    model.consum_flex_Ctr = Constraint(model.AREAS, model.TIME, model.FLEX_CONSUM, rule=consum_flex_rule)

    # Flex ratio constraint
    def flex_variation_sup_rule(model, area, t, conso_type):
        if model.flex_type[area, conso_type] != 'day_ev' or TIME_df.loc[t, 'ctr_ev'] == 1:
            return model.flex_Dvar[area, t, conso_type] <= model.flex_ratio[area, conso_type]
        else:
            return Constraint.Skip

    model.flex_variation_sup_Ctr = Constraint(model.AREAS, model.TIME, model.FLEX_CONSUM, rule=flex_variation_sup_rule)

    def flex_variation_inf_rule(model, area, t, conso_type):
        if model.flex_type[area, conso_type] != 'day_ev' or TIME_df.loc[area, t, 'ctr_ev'] == 1:
            return model.flex_Dvar[area, t, conso_type] >= -model.flex_ratio[area, conso_type]
        else:
            return Constraint.Skip

    model.flex_variation_inf_Ctr = Constraint(model.AREAS,model.TIME, model.FLEX_CONSUM, rule=flex_variation_inf_rule)

    # Definition a_plus, a_minus
    def a_plus_minus_rule(model, area, conso_type, t):
        return model.flex_consumption_Pvar[area, t, conso_type] - model.to_flex_consumption[area, t, conso_type] == \
               model.a_plus_Pvar[area, t, conso_type] - model.a_minus_Pvar[area, t, conso_type]

    model.a_plus_minusCtr = Constraint(model.AREAS, model.FLEX_CONSUM, model.TIME, rule=a_plus_minus_rule)

    return model
