
import time

from Models.Basic_France_models.Operation_optimisation.f_operationModels import *
from functions.f_tools import *
import sys

def test_solvers_operation_multizone():
    InputFolder = '../Data/input/'

    BaseSolverPath = '/Users/robin.girard/Documents/Code/Packages/solvers/ampl_macosx64'
    sys.path.append(BaseSolverPath)
    solvers = [ 'knitro', 'cbc']  # 'glpk' is too slow 'cplex' and 'xpress' do not work
    solverpath = {}
    for solver in solvers: solverpath[solver] = BaseSolverPath + '/' + solver
    solvers.append('mosek')
    cplexPATH = '/Applications/CPLEX_Studio1210/cplex/bin/x86-64_osx'
    sys.path.append(cplexPATH)
    solvers.append('cplex')
    solverpath['cplex'] = cplexPATH + "/" + "cplex"

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
        opt = opt = MySolverFactory(solver,solverpath)
        results = opt.solve(model)
        print('Elapsed time for ' + solver + ': ' + str(time.time() - start))



def test_solvers_operation_StorageSinglezone():
    InputFolder = '../Data/input/'

    BaseSolverPath = '/Users/robin.girard/Documents/Code/Packages/solvers/ampl_macosx64'
    sys.path.append(BaseSolverPath)
    solvers = [ 'knitro', 'cbc']  # 'glpk' is too slow 'cplex' and 'xpress' do not work
    solverpath = {}
    for solver in solvers: solverpath[solver] = BaseSolverPath + '/' + solver
    solvers.append('mosek')
    cplexPATH = '/Applications/CPLEX_Studio1210/cplex/bin/x86-64_osx'
    sys.path.append(cplexPATH)
    solvers.append('cplex')
    solverpath['cplex'] = cplexPATH + "/" + "cplex"
    Zones = "FR"
    year = 2013

    Selected_TECHNOLOGIES = {'Thermal', 'OldNuke', 'WindOnShore', "Curtailement"}

    #### reading CSV files
    areaConsumption = pd.read_csv(InputFolder + 'areaConsumption' + str(year) + '_' + str(Zones) + '.csv',
                                  sep=',', decimal='.', skiprows=0)
    availabilityFactor = pd.read_csv(InputFolder + 'availabilityFactor' + str(year) + '_' + str(Zones) + '.csv',
                                     sep=',', decimal='.', skiprows=0)
    TechParameters = pd.read_csv(InputFolder + 'Gestion-Simple_TECHNOLOGIES.csv', sep=';', decimal=',', skiprows=0)

    #### Selection of subset
    availabilityFactor = availabilityFactor[availabilityFactor.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]
    TechParameters = TechParameters[TechParameters.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]

    p_max = 10000
    StorageParameters = {"p_max": p_max, "c_max": p_max * 10, "efficiency_in": 0.9, "efficiency_out": 0.9}
    # endregion

    # region IV Ramp+Storage single area : solving and loading results


    for solver in solvers:
        start = time.time()
        res = GetElectricSystemModel_GestionSingleNode_with1Storage(areaConsumption, availabilityFactor,
                                                                    TechParameters, StorageParameters,solver=solver,
                                                                    solverpath=solverpath)
        #results = opt.solve(model)
        print('Elapsed time for ' + solver + ': ' + str(time.time() - start))


