#region importation of modules
import os
if os.path.basename(os.getcwd())=="BasicFunctionalities":
    os.chdir('..') ## to work at project root  like in any IDE
InputFolder='Data/input/'
import pandas as pd
import qgrid # great package https://github.com/quantopian/qgrid
import numpy as np
#endregion

#region - general economic assumption
General = pd.read_csv(InputFolder+'GeneralEconomicAssumptions.csv',sep=',',decimal='.',skiprows=0,comment="#")
col_def = {}#  { 'A': { 'editable': True } }
qgrid_widget = qgrid.show_grid(General, show_toolbar=True, column_definitions=col_def)
qgrid_widget
#endregion

General=qgrid_widget.get_changed_df()
General

#region - general economic assumption
Production = pd.read_csv(InputFolder+'ProductionEconomicAssumptions.csv',sep=',',decimal='.',skiprows=0,comment="#")
Production=Production[Production["AREAS"]=="FR"]
qgrid_widget = qgrid.show_grid(Production, show_toolbar=True, column_definitions=col_def)
qgrid_widget

#endregion

Production=qgrid_widget.get_changed_df()
Production

r=(General.discountPercent[0]/100)
Production=Production.assign(
    LLr = round((1+r)/r*(1-(1+r)**(-Production['LL'])),2))
Production = Production.assign(
    capacityCost = round(Production.CAPEX/Production.LLr+ Production.dismantling/(1+r)**(Production.LL) +Production.FOM,2)*1000,
    energyCost = round(Production.Variable + Production.CO2Emission*General.carbonTaxEurosPerT[0]/10**3,2)
)


#region - general economic assumption
ProductionTech = pd.read_csv(InputFolder+'ProductionTechnicalAssumptions.csv',sep=',',decimal='.',skiprows=0,comment="#")
ProductionTech=ProductionTech[ProductionTech["AREAS"]=="FR"]
qgrid_widget = qgrid.show_grid(ProductionTech, show_toolbar=True, column_definitions=col_def)
qgrid_widget
ProductionTech=qgrid_widget.get_changed_df()
ProductionTechPlus=ProductionTech.merge(Production, how='inner', left_on=["AREAS","TECHNOLOGIES"], right_on=["AREAS","TECHNOLOGIES"])
ProductionTechPlus
#endregion

SelectedCols=["TECHNOLOGIES","energyCost","capacity","EnergyNbhourCap"]
ProductionTechPlus[SelectedCols].to_csv(InputFolder+'Gestion-Simple_TECHNOLOGIES.csv',sep=',',decimal='.', index=False)

SelectedCols=["TECHNOLOGIES","energyCost","capacity","EnergyNbhourCap","RampConstraintPlus","RampConstraintMoins","RampConstraintPlus2","RampConstraintMoins2"]
ProductionTechPlus[SelectedCols].to_csv(InputFolder+'Gestion-RAMP1_TECHNOLOGIES.csv',sep=',',decimal='.', index=False)


ProductionTechPlus=ProductionTechPlus.assign(
    minCapacity = ProductionTechPlus["capacity"]*0.9,
    maxCapacity = ProductionTechPlus["capacity"]*100
)

ProductionTechPlus.maxCapacity.loc[ProductionTechPlus.TECHNOLOGIES=="HydroRiver"]=ProductionTechPlus.capacity.loc[ProductionTechPlus.TECHNOLOGIES=="HydroRiver"]
ProductionTechPlus.maxCapacity.loc[ProductionTechPlus.TECHNOLOGIES=="HydroReservoir"]=ProductionTechPlus.capacity.loc[ProductionTechPlus.TECHNOLOGIES=="HydroReservoir"]

SelectedCols=["TECHNOLOGIES","energyCost","capacityCost","EnergyNbhourCap",	"minCapacity","maxCapacity"]
ProductionTechPlus[SelectedCols].to_csv(InputFolder+'Planing-Simple_TECHNOLOGIES.csv',sep=',',decimal='.', index=False)

SelectedCols=["TECHNOLOGIES","energyCost","capacityCost","EnergyNbhourCap","minCapacity","maxCapacity","RampConstraintPlus","RampConstraintMoins","RampConstraintPlus2","RampConstraintMoins2"]
ProductionTechPlus[SelectedCols].to_csv(InputFolder+'Planing-RAMP1_TECHNOLOGIES.csv',sep=',',decimal='.', index=False)

ProductionTechPlus.capacityCost[ProductionTechPlus.TECHNOLOGIES=="NewNuke"]/6000


#region - general economic assumption
Production = pd.read_csv(InputFolder+'ProductionEconomicAssumptions.csv',sep=',',decimal='.',skiprows=0,comment="#")
qgrid_widget = qgrid.show_grid(Production, show_toolbar=True, column_definitions=col_def)
qgrid_widget

#endregion


Production=qgrid_widget.get_changed_df()


r=(General.discountPercent[0]/100)
Production=Production.assign(
    LLr = round((1+r)/r*(1-(1+r)**(-Production['LL'])),2))
Production = Production.assign(
    capacityCost = round(Production.CAPEX/Production.LLr+ Production.dismantling/(1+r)**(Production.LL) +Production.FOM,2)*1000,
    energyCost = round(Production.Variable + Production.CO2Emission*General.carbonTaxEurosPerT[0]/10**3,2)
)




#region - general economic assumption
ProductionTech = pd.read_csv(InputFolder+'ProductionTechnicalAssumptions.csv',sep=',',decimal='.',skiprows=0,comment="#")
qgrid_widget = qgrid.show_grid(ProductionTech, show_toolbar=True, column_definitions=col_def)
qgrid_widget
ProductionTech=qgrid_widget.get_changed_df()
ProductionTechPlus=ProductionTech.merge(Production, how='inner', left_on=["AREAS","TECHNOLOGIES"], right_on=["AREAS","TECHNOLOGIES"])
ProductionTechPlus
#endregion


SelectedCols=["AREAS","TECHNOLOGIES","energyCost","capacity","EnergyNbhourCap","RampConstraintPlus","RampConstraintMoins","RampConstraintPlus2","RampConstraintMoins2"]
ProductionTechPlus[SelectedCols].to_csv(InputFolder+'Gestion_MultiNode_DE-FR_AREAS_TECHNOLOGIES.csv',sep=',',decimal='.', index=False)

ProductionTechPlus=ProductionTechPlus.assign(
    minCapacity = ProductionTechPlus["capacity"]*0.9,
    maxCapacity = ProductionTechPlus["capacity"]*100
)



index=ProductionTechPlus["TECHNOLOGIES"]=="HydroReservoir"
ProductionTechPlus.maxCapacity.loc[index]=ProductionTechPlus.capacity[index]
index=ProductionTechPlus["TECHNOLOGIES"]=="HydroRiver"
ProductionTechPlus.maxCapacity.loc[index]=ProductionTechPlus.capacity[index]
SelectedCols=["AREAS","TECHNOLOGIES","energyCost","capacityCost","EnergyNbhourCap","minCapacity","maxCapacity","RampConstraintPlus","RampConstraintMoins","RampConstraintPlus2","RampConstraintMoins2"]
ProductionTechPlus[SelectedCols].to_csv(InputFolder+'Planing_MultiNode_DE-FR_TECHNOLOGIES_AREAS.csv',sep=',',decimal='.', index=False)
