import pytest
import time

from functions.functions_Operation import *

def test_solvers_operation_multizone():
    InputFolder = '../Data/input/'

    solverpath = {}
    solverpath['cbc'] = '/Users/robin.girard/Documents/Code/Packages/cbc-osx'
    solverpath['mosek'] = '/Users/robin.girard/Documents/Code/Packages/cbc-osx'
    solvers = solverpath.keys()
    Zones = "FR_DE_GB_ES"
    year = 2016
    Selected_AREAS = {"FR", "DE"}
    Selected_TECHNOLOGIES = {'Thermal', 'OldNuke', 'NewNuke', 'HydroRiver', 'HydroReservoir',
                             'WindOnShore', 'WindOffShore', 'Solar', 'Curtailement'}

    #### reading CSV files
    TechParameters = pd.read_csv(InputFolder + 'Gestion_MultiNode_DE-FR_AREAS_TECHNOLOGIES.csv', sep=';', decimal=',',
                                 comment="#", skiprows=0)
    areaConsumption = pd.read_csv(InputFolder + 'areaConsumption' + str(year) + '_' + str(Zones) + '.csv',
                                  sep=',', decimal='.', skiprows=0)
    availabilityFactor = pd.read_csv(InputFolder + 'availabilityFactor' + str(year) + '_' + str(Zones) + '.csv',
                                     sep=',', decimal='.', skiprows=0)
    ExchangeParameters = pd.read_csv(InputFolder + 'Hypothese_DE-FR_AREAS_AREAS.csv', sep=';', decimal=',', skiprows=0,
                                     comment="#")
    #### Selection of subset
    TechParameters = TechParameters[
        TechParameters.AREAS.isin(Selected_AREAS) & TechParameters.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]
    areaConsumption = areaConsumption[areaConsumption.AREAS.isin(Selected_AREAS)]
    availabilityFactor = availabilityFactor[
        availabilityFactor.AREAS.isin(Selected_AREAS) & availabilityFactor.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]

    ### small data cleaning
    availabilityFactor.availabilityFactor[availabilityFactor.availabilityFactor > 1] = 1
    print('Building model')
    model = GetElectricSystemModel_GestionMultiNode(areaConsumption, availabilityFactor, TechParameters,
                                                    ExchangeParameters)

    for solver in solvers:
        start = time.time()
        opt = SolverFactory(solver)
        results = opt.solve(model)
        print('Elapsed time for ' + solver + ': ' + str(time.time() - start))





