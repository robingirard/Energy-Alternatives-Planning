#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 2023
@author: Sylvain Brisson sylvain.brisson@ens.fr
"""

import pandas as pd
pd.options.plotting.backend = "plotly"

import pickle

import sys
sys.path.append("..")
import f_graphicalTools as nrjPlot

# import EnergyAlternativesPlanning.f_graphicalTools as nrjPlot

if __name__ == "__main__":
    
    # ---------------------------------
    # LOADING TEST DATA
    # 1/ Loading input xlsx file
    xls_file = pd.ExcelFile("data/inputs.xlsx")
    
    # Extracting data
    TechParameters = pd.read_excel(xls_file,"TECHNO_AREAS").dropna().set_index(["AREAS", "TECHNOLOGIES"])
    
    StorageParameters = pd.read_excel(xls_file,"STOCK_TECHNO_AREAS").set_index(["AREAS", "STOCK_TECHNO"])
    
    flexConsumInputs = pd.read_excel(xls_file,"FLEX_CONSUM").set_index(["FLEX_CONSUM", "AREAS"])
    
    # 2/ Loading output (serialized dictionary of dataframes)    
    with open("data/results.pkl",'rb') as f: Variables=pickle.load(f)
    
    # 3/ Metadata
    
    DATE = Variables["energy"]["Date"].unique()
    
    TECHNO_PROD = list(Variables["energy"]["TECHNOLOGIES"].unique())
    TECHNO_PROD_NO_COAL = ['curtailment','TAC','Solar','Biomass','CCG - H2','WindOnShore','CCG','TAC - H2','WindOffShore','OldNuke','HydroRiver','NewNuke','HydroReservoir']
    TECHNO_STORAGE = list(Variables["storageIn"]["STOCK_TECHNO"].unique())
    TECHNO_ORDER = ['OldNuke', 'NewNuke', 'Biomass', 'WindOffShore', 'WindOnShore', 'Solar', 'HydroRiver', 'HydroReservoir','CCG - H2', 'TAC - H2','CCG', 'TAC', 'curtailment']
    
    AREAS = list(Variables["energy"]["AREAS"].unique())
    AREAS_ORDER = ['FR','DE','GB','ES','IT','BE','CH']
    AREA_FULLNAME = {'BE':"Belgium", 'CH':"Switzerland", 'DE':"Deutshland", 'ES':"Spain", 'FR':"France", 'GB':"Great Britain", 'IT':"Italy"}
    
    # Extracting data
    
    cost_production = nrjPlot.extractCosts(Variables).round(2)

    energyCapacity = nrjPlot.extractEnergyCapacity(Variables).round(2).loc[AREAS_ORDER]

    installedCapacity = energyCapacity["Capacity_GW"]

    cost_storage = pd.pivot_table(Variables["storageCosts"], values="storageCosts", index="AREAS", columns="STOCK_TECHNO")/1e9

    cost_flex = pd.pivot_table(Variables['consumption_power_cost'], values="consumption_power_cost", index="AREAS", columns="FLEX_CONSUM")/1e9

    consumption = pd.pivot_table(Variables["total_consumption"], values="total_consumption", index=["AREAS","Date"]).rename(columns={"total_consumption":"areaConsumption"})*1e-3

    flex_conso = pd.pivot_table(Variables["flex_consumption"], values="flex_consumption", index=["AREAS","Date"], columns="FLEX_CONSUM")*1e-3

    production = pd.pivot_table(Variables["energy"], values="energy", index=["AREAS","Date"], columns="TECHNOLOGIES")*1e-3

    exchanges = pd.pivot_table(Variables["exchange"].rename(columns={"TIMESTAMP":"Date"}), values="exchange", index=["AREAS1","Date"],columns="AREAS2")*1e-3

    # get storage data
    storageIn = pd.pivot_table(Variables["storageIn"], values="storageIn", index=["AREAS","Date"],columns="STOCK_TECHNO")*1e-3
    storageOut = pd.pivot_table(Variables["storageOut"], values="storageOut", index=["AREAS","Date"],columns="STOCK_TECHNO")*1e-3

    # sum into a single df with negative and positive values (negative : storageIn)
    storage = storageOut - storageIn

    # get installed capacities and "produced energy"
    capa_storage = storage.abs().groupby(level=[[0]]).max()
    energy_out_storage = storageOut.groupby(level=[[0]]).sum()/1000

    # get total exchanges for each area
    exchanges_sumed = exchanges.sum(axis=1).rename("exchanges")

    stock_level = pd.pivot_table(Variables["stockLevel"], values="stockLevel", index=["AREAS","Date"], columns="STOCK_TECHNO")

    # concat with production
    prod_exchanges = pd.concat([production, -exchanges_sumed], axis=1)

    # concat with storage
    prod_ex_storage = pd.concat([prod_exchanges, storage], axis=1)
    
    # ----------------------
    
    # installed capacities / energy production
    
    fig = nrjPlot.installedCapa_barChart(energyCapacity, TechParameters)
    fig.update_layout(autosize=False,width=1000,height=500)
    fig.write_image("gallery/installedCapa_barChart.svg", scale=2)
    print("Wrote installedCapa_barChart.svg")
    
    fig = nrjPlot.productionCapa_stackedBarChart(energyCapacity,capaDisp="TWh")
    fig.write_image("gallery/productionProduction_stackedBarChart_energy.svg", scale=2)
    print("Wrote productionProduction_stackedBarChart_energy.svg")
    
    fig = nrjPlot.productionCapa_stackedBarChart(energyCapacity,capaDisp="GW")
    fig.write_image("gallery/productionProduction_stackedBarChart_capacity.svg", scale=2)
    print("Wrote productionProduction_stackedBarChart_capacity.svg")
    
    fig = nrjPlot.production_pieChart(energyCapacity)
    fig.update_layout(autosize=False,width=1200,height=500)
    fig.write_image("gallery/production_pieChart.svg", scale=2)
    print("Wrote production_pieChart.svg")
    
    fig = nrjPlot.loadFactors(energyCapacity)
    fig.write_image("gallery/loadFactors.svg", scale=2)
    print("Wrote loadFactors.svg")
    
    
    # production/consumption time series
    
    area = "FR"
    start_date = "01/01/2018"
    end_date = "15/01/2018"
    fig = nrjPlot.plotProduction(
        prod_ex_storage.loc[area], 
        conso=consumption.loc[area], 
        flex_conso=flex_conso.loc[area], 
        title=f"{AREA_FULLNAME[area]} production and consumption", yaxis_title="Power (GW)",
        start_date=start_date, end_date=end_date)
    
    fig.write_image("gallery/plotProduction.svg", scale=2)
    print("Wrote plotProduction.svg")
    
    # monotones de puissance
    
    monotones = nrjPlot.getMonotonesPower(production.loc["FR"][["Solar","WindOnShore","Biomass"]])
    
    fig = nrjPlot.MyPlotly(monotones, fill=False, no_slide=True, yaxis_title="Power (GW)", xaxis_title="Hours")
    fig.write_image("gallery/getMonotonesPower+MyPlotly.svg", scale=2)
    print("Wrote getMonotonesPower+MyPlotly.svg")
    
    # storage capacities
    
    fig = nrjPlot.installedCapaStoragePower_barChart(storage, flex_conso)
    fig.write_image("gallery/installedCapaStoragePower_barChart.svg", scale=2)
    print("Wrote installedCapaStoragePower_barChart.svg")
    
    # cost data
    
    fig = nrjPlot.costPerCountry(energyCapacity, cost_production, cost_storage, cost_flex)
    fig.write_image("gallery/costPerCountry.svg", scale=2)
    print("Wrote costPerCountry.svg")
    
    fig = nrjPlot.costDecomposed_barChart(cost_production, cost_storage, cost_flex)
    fig.update_layout(autosize=False,width=1000,height=500)
    fig.write_image("gallery/costDecomposed_barChart.svg", scale=2)
    print("Wrote costDecomposed_barChart.svg")
    
    # # material consumption (annex)
    
    # conso_materials = nrjPlot.get_material_consumption(energyCapacity, stock_level).loc[AREAS_ORDER]
    
    # fig = nrjPlot.materialUsage_barChart(conso_materials, "Cu")
    # fig.write_image("gallery/materialUsage_barChart.svg", scale=2)
    # print("Wrote materialUsage_barChart.svg")
    
    
    