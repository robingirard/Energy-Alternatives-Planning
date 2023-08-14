import linopy
import xarray as xr
from linopy import Model
import os
import pandas as pd
import subprocess as sub
import highspy
from Models.Linopy.f_tools import *

def run_highs(
    model,
    io_api=None,
    problem_fn=None,
    solution_fn=None,
    log_fn=None,
    warmstart_fn=None,
    basis_fn=None,
    keep_files=False,
    **solver_options,
):
    """
    Highs solver function. Reads a linear problem file and passes it to the
    highs solver. If the solution is feasible the function returns the
    objective, solution and dual constraint variables. Highs must be installed
    for usage. Find the documentation at https://www.maths.ed.ac.uk/hall/HiGHS/

    . The full list of solver options is documented at
    https://www.maths.ed.ac.uk/hall/HiGHS/HighsOptions.set .

    Some exemplary options are:

        * presolve : "choose" by default - "on"/"off" are alternatives.
        * solver :"choose" by default - "simplex"/"ipm" are alternatives.
        * parallel : "choose" by default - "on"/"off" are alternatives.
        * time_limit : inf by default.

    Returns
    -------
    status : string,
        SolverStatus.ok or SolverStatus.warning
    termination_condition : string,
        Contains "optimal", "infeasible",
    variables_sol : series
    constraints_dual : series
    objective : float
    """
    CONDITION_MAP = {}

    if warmstart_fn:
        logger.warning("Warmstart not available with HiGHS solver. Ignore argument.")

    if io_api is None or io_api in ["lp", "mps"]:
        model.to_file(problem_fn)
        h = highspy.Highs()
        h.readModel(maybe_convert_path(problem_fn))
    elif io_api == "direct":
        h = model.to_highspy()
    else:
        raise ValueError(
            "Keyword argument `io_api` has to be one of `lp`, `mps`, `direct` or None"
        )

    if log_fn is None:
        log_fn = model.solver_dir / "highs.log"
    solver_options["log_file"] = maybe_convert_path(log_fn)
    logger.info(f"Log file at {solver_options['log_file']}.")

    for k, v in solver_options.items():
        h.setOptionValue(k, v)

    h.run()

    condition = h.modelStatusToString(h.getModelStatus()).lower()
    termination_condition = CONDITION_MAP.get(condition, condition)
    status = Status.from_termination_condition(termination_condition)
    status.legacy_status = condition

    def get_solver_solution() -> Solution:
        objective = h.getObjectiveValue()
        solution = h.getSolution()

        if io_api == "direct":
            sol = pd.Series(solution.col_value, model.matrices.vlabels, dtype=float)
            dual = pd.Series(solution.row_dual, model.matrices.clabels, dtype=float)
        else:
            sol = pd.Series(solution.col_value, h.getLp().col_names_, dtype=float).pipe(
                set_int_index
            )
            dual = pd.Series(solution.row_dual, h.getLp().row_names_, dtype=float).pipe(
                set_int_index
            )

        return Solution(sol, dual, objective)

    solution = safe_get_solution(status, get_solver_solution)

    return Result(status, solution, h)

def run_highs_(
    Model,
    io_api=None,
    problem_fn=None,
    solution_fn=None,
    log_fn=None,
    warmstart_fn=None,
    basis_fn=None,
    keep_files=False,
    **solver_options,
):
    """
    Highs solver function. Reads a linear problem file and passes it to the
    highs solver. If the solution is feasible the function returns the
    objective, solution and dual constraint variables. Highs must be installed
    for usage. Find the documentation at https://www.maths.ed.ac.uk/hall/HiGHS/
    . The full list of solver options is documented at
    https://www.maths.ed.ac.uk/hall/HiGHS/HighsOptions.set .
    Some examplary options are:
        * presolve : "choose" by default - "on"/"off" are alternatives.
        * solver :"choose" by default - "simplex"/"ipm" are alternatives.
        * parallel : "choose" by default - "on"/"off" are alternatives.
        * time_limit : inf by default.
    Returns
    -------
    status : string,
        "ok" or "warning"
    termination_condition : string,
        Contains "optimal", "infeasible",
    variables_sol : series
    constraints_dual : series
    objective : float
    """
    Model.to_file(problem_fn)

    if log_fn is None:
        log_fn = Model.solver_dir / "highs.log"

    options_fn = Model.solver_dir / "highs_options.txt"
    hard_coded_options = {
        "solution_file": solution_fn,
        "write_solution_to_file": True,
        "write_solution_style": 1,
        "log_file": log_fn,
    }
    solver_options.update(hard_coded_options)

    method = solver_options.pop("method", "choose")

    with open(options_fn, "w") as fn:
        fn.write("\n".join([f"{k} = {v}" for k, v in solver_options.items()]))

    command = f"highs --model_file {problem_fn} "
    if warmstart_fn:
        logger.warning("Warmstart not available with HiGHS solver. Ignore argument.")
    command += f"--solver {method} --options_file {options_fn}"

    p = sub.Popen(command.split(" "), stdout=sub.PIPE, stderr=sub.PIPE)

    if method in ['simplex', 'ipm']:
        for line in iter(p.stdout.readline, b""):
            line = line.decode()

            if line.startswith("Model   status"):
                model_status = line[len("Model   status      : ") : -2].lower()
                if "optimal" in model_status:
                    status = "ok"
                    termination_condition = model_status
                elif "infeasible" in model_status:
                    status = "warning"
                    termination_condition = model_status
                else:
                    status = "warning"
                    termination_condition = model_status

            if line.startswith("Objective value"):
                objective = float(line[len("Objective value     :  ") :])

            print(line, end="")
    else:
        for line in iter(p.stdout.readline, b""):
            line = line.decode()
            line = line.replace('\r\n', '')

            if 'Status' in line:
                model_status = line.split(' ')[-1].lower()
                if "optimal" in model_status:
                    status = "ok"
                    termination_condition = model_status
                elif "infeasible" in model_status:
                    status = "warning"
                    termination_condition = model_status
                else:
                    status = "warning"
                    termination_condition = model_status

            if '(objective)' in line.lower():
                objective = float(line.split(' ')[-2])

            print(line + '\n', end="")

    p.stdout.close()
    p.wait()

    os.remove(options_fn)

    f = open(solution_fn, "rb")
    f.readline()
    trimmed = re.sub(rb"\*\*\s+", b"", f.read())
    sol, sentinel, dual = trimmed.partition(bytes("Rows\r\n", "utf-8"))
    f.close()

    sol = pd.read_fwf(io.BytesIO(sol), infer_nrows=1e9)
    sol = sol.set_index("Name")["Primal"].pipe(set_int_index)

    dual = pd.read_fwf(io.BytesIO(dual), infer_nrows=1e9).iloc[:-2]["Dual"]
    dual.index = Model.constraints.ravel("labels", filter_missings=True)
    dual.fillna(0, inplace=True)

    return dict(
        status=status,
        termination_condition=termination_condition,
        solution=sol,
        dual=dual,
        objective=objective,
    )


def Build_EAP_Model(Parameters):
    """
    This function creates the pyomo model and initlize the Parameters and (pyomo) Set values
    :param Parameters is a dictionnary with different panda tables :
        - areaConsumption: panda table with consumption
        - availabilityFactor: panda table
        - TechParameters : panda tables indexed by TECHNOLOGIES with eco and tech parameters
    """

    ## Starting with an empty model object
    m = linopy.Model()

    ### Obtaining dimensions values
    Date = Parameters.get_index('Date').unique()
    AREAS = Parameters.get_index('AREAS')
    TECHNOLOGIES = Parameters.get_index('TECHNOLOGIES').unique()

    # Variables - Base - Operation & Planning
    production_op_power = m.add_variables(name="production_op_power", lower=0, coords=[Date,AREAS, TECHNOLOGIES]) ### Energy produced by a production mean at time t
    production_op_variable_costs = m.add_variables(name="production_op_variable_costs", lower=0, coords=[AREAS,TECHNOLOGIES]) ### Energy total marginal cost for production mean p
    production_pl_capacity_costs = m.add_variables(name="production_pl_capacity_costs", lower=0, coords=[AREAS,TECHNOLOGIES]) ### Energy produced by a production mean at time t
    production_pl_capacity = m.add_variables(name="production_pl_capacity", lower=0, coords=[AREAS,TECHNOLOGIES]) ### Energy produced by a production mean at time t
    conso_op_totale = m.add_variables(name="conso_op_totale",lower=0, coords=[Date, AREAS])

    # Variable - Storage - Operation & Planning
    # Objective Function (terms to be added later in the code for storage and flexibility)
    Cost_Function = production_op_variable_costs.sum() + production_pl_capacity_costs.sum()
    m.add_objective( Cost_Function)

    #################
    # Constraints   #
    #################

    ## table of content for the constraints definition - Operation (Op) & Planning (Pl)
    # 1 - Main constraints
    # 2 - Optional constraints
    # 3 - Storage constraints
    # 4 - exchange constraints

    #####################
    # 1 - a - Main Constraints - Operation (Op) & Planning (Pl)

    Ctr_Op_energyCosts = m.add_constraints(name="Ctr_Op_energyCosts", # contrainte de définition de energyCosts
        lhs = production_op_variable_costs == Parameters["energyCost"] * production_op_power.sum(["Date"]) )

    Ctr_Op_conso = m.add_constraints(name="Ctr_Op_conso",
        lhs = conso_op_totale == Parameters["areaConsumption"])

    Ctr_Op_energy = m.add_constraints(name="Ctr_Op_energy",
        lhs = production_op_power.sum(["TECHNOLOGIES"]) == conso_op_totale)

    Ctr_Op_capacity = m.add_constraints(name="Ctr_Op_capacit", # contrainte de maximum de production
        lhs = production_op_power <= production_pl_capacity * Parameters["availabilityFactor"])

    Ctr_Pl_capacityCosts = m.add_constraints(name="Ctr_Pl_capacityCosts", # contrainte de définition de capacityCosts
        lhs = production_pl_capacity_costs == Parameters["capacityCost"] * production_pl_capacity )

    Ctr_Pl_maxCapacity = m.add_constraints(name="Ctr_Pl_maxCapacity",
        lhs = production_pl_capacity <= Parameters["maxCapacity"],
        mask=Parameters["maxCapacity"] > 0)

    Ctr_Pl_minCapacity = m.add_constraints(name="Ctr_Pl_minCapacity",
        lhs = production_pl_capacity >= Parameters["minCapacity"],
        mask=Parameters["minCapacity"] > 0)


    #####################
    # 2 - Optional Constraints - Operation (Op) & Planning (Pl)

    if "EnergyNbhourCap" in Parameters: Ctr_Op_stock = m.add_constraints(name="stockCtr",
            lhs = Parameters["EnergyNbhourCap"] * production_pl_capacity >= production_op_power.sum(["Date"]),
            mask=Parameters["EnergyNbhourCap"] > 0)

    if "RampConstraintPlus" in Parameters: Ctr_Op_rampPlus = m.add_constraints(name="Ctr_Op_rampPlus",
            lhs = production_op_power.diff("Date", n=1) <=production_pl_capacity
                  * (Parameters["RampConstraintPlus"] * Parameters["availabilityFactor"])  ,
            mask= (Parameters["RampConstraintPlus"] > 0) *
                  (xr.DataArray(Date, coords=[Date])!=Date[0]))

    if "RampConstraintMoins" in Parameters: Ctr_Op_rampMoins = m.add_constraints(name="Ctr_Op_rampMoins",
            lhs = production_op_power.diff("Date", n=1) + production_pl_capacity
                  * (Parameters["RampConstraintMoins"] * Parameters["availabilityFactor"]) >= 0,
            # remark : "-" sign not possible in lhs, hence the inequality alternative formulation
            mask=(Parameters["RampConstraintMoins"] > 0) *
                 (xr.DataArray(Date, coords=[Date])!=Date[0]))

    if "RampConstraintPlus2" in Parameters:
        Ctr_Op_rampPlus2 = m.add_constraints(name="Ctr_Op_rampPlus2",
            lhs = production_op_power.diff("Date", n=2) <= production_pl_capacity
                  * (Parameters["RampConstraintPlus2"] * Parameters["availabilityFactor"]),
            mask=(Parameters["RampConstraintPlus2"] > 0) *
                 (xr.DataArray(Date, coords=[Date])!=Date[0]))

    if "RampConstraintMoins2" in Parameters:
        Ctr_Op_rampMoins2 = m.add_constraints(name="Ctr_Op_rampMoins2",
            lhs = production_op_power.diff("Date", n=2)+production_pl_capacity
                  * (Parameters["RampConstraintMoins2"] * Parameters["availabilityFactor"]) >= 0,
            mask=(Parameters["RampConstraintMoins2"] > 0) *
                 (xr.DataArray(Date, coords=[Date])!=Date[0]))



    #####################
    # 3 -  Storage Constraints - Operation (Op) & Planning (Pl)

    if "STOCK_TECHNO" in Parameters:
        STOCK_TECHNO = Parameters.get_index('STOCK_TECHNO')

        storage_op_power_in = m.add_variables(name="storage_op_power_in", lower=0,coords = [Date,AREAS,STOCK_TECHNO])  ### Energy stored in a storage mean at time t
        storage_op_power_out = m.add_variables(name="storage_op_power_out", lower=0,coords = [Date,AREAS,STOCK_TECHNO])  ### Energy taken out of a storage mean at time t
        storage_op_stock_level = m.add_variables(name="storage_op_stock_level", lower=0,coords = [Date,AREAS,STOCK_TECHNO])  ### level of the energy stock in a storage mean at time t
        storage_pl_capacity_costs = m.add_variables(name="storage_pl_capacity_costs",coords = [AREAS,STOCK_TECHNO])  ### Cost of storage for a storage mean, explicitely defined by definition storageCostsDef
        storage_pl_max_capacity = m.add_variables(name="storage_pl_max_capacity",coords = [AREAS,STOCK_TECHNO])  # Maximum capacity of a storage mean
        storage_pl_max_power = m.add_variables(name="storage_pl_max_power",coords = [AREAS,STOCK_TECHNO])  # Maximum flow of energy in/out of a storage mean
        #storage_op_stockLevel_ini = m.add_variables(name="storage_op_stockLevel_ini",coords = [AREAS,STOCK_TECHNO], lower=0)

        #update of the cost function and of the prod = conso constraint
        m.objective += storage_pl_capacity_costs.sum()
        m.constraints['Ctr_Op_energy'].lhs += storage_op_power_out.sum(['STOCK_TECHNO'])-storage_op_power_in.sum(['STOCK_TECHNO'])

        Ctr_Op_storage_level_definition = m.add_constraints(name="Ctr_Op_storage_level_definition",
            lhs=storage_op_stock_level.shift(Date=1) == storage_op_stock_level * (1 - Parameters["dissipation"])+storage_op_power_in* Parameters["efficiency_in"]
                                                        - storage_op_power_out / Parameters["efficiency_out"],
            mask=xr.DataArray(Date, coords=[Date]) != Date[0])

        Ctr_Op_storage_initial_level = m.add_constraints(name="Ctr_Op_storage_initial_level",
            lhs= storage_op_stock_level.loc[{"Date" : Date[0] }] == storage_op_stock_level.loc[{"Date" : Date[-1] }])

        Ctr_Op_storage_power_in_max = m.add_constraints(name="Ctr_Op_storage_power_in_max",
            lhs=storage_op_power_in <= storage_pl_max_power)

        Ctr_Op_storage_power_out_max = m.add_constraints(name="Ctr_Op_storage_power_out_max",
            lhs=storage_op_power_out <= storage_pl_max_power)

        Ctr_Op_storage_capacity_max = m.add_constraints(name="Ctr_Op_storage_capacity_max",
            lhs=storage_op_stock_level <= storage_pl_max_capacity)

        Ctr_Pl_storageCosts = m.add_constraints(name="Ctr_Pl_storageCosts",
             lhs=storage_pl_capacity_costs == Parameters["storageCost"] * storage_pl_max_capacity)

        Ctr_Pl_storage_max_capacity = m.add_constraints(name="Ctr_Pl_storage_max_capacity",
             lhs=storage_pl_max_capacity <= Parameters["c_max"])

        Ctr_Pl_storage_max_power = m.add_constraints(name="Ctr_Pl_storage_max_power",
             lhs=storage_pl_max_power <= Parameters["p_max"])

    #####################
    # 4 -  Exchange Constraints - Operation (Op) & Planning (Pl)

    if len(AREAS)>1:
        AREAS_1=  Parameters.get_index('AREAS_1')
        exchange_op_power = m.add_variables(name="exchange_op_power", lower=0,coords = [Date, AREAS,AREAS_1])  ### Energy stored in a storage mean at time t
        m.constraints['Ctr_Op_energy'].lhs += exchange_op_power.sum(['AREAS_1']) - exchange_op_power.rename({'AREAS_1':'AREAS','AREAS':'AREAS_1'}).sum(['AREAS_1'])
        Ctr_Op_exchange_max = m.add_constraints(name="Ctr_Op_exchange_max",
            lhs=exchange_op_power<= Parameters["maxExchangeCapacity"])
        m.objective += 0.00001 * exchange_op_power.sum()


    # Flex consumption
    if "FLEX_CONSUM" in Parameters:
        FLEX_CONSUM = Parameters.get_index('FLEX_CONSUM')
        # inscrire les équations ici ?
        # flexconso_pl_increased_max_power_costs = "LoadCost" * flexconso_pl_increased_max_power
        # conso_op_totale <= flexconso_pl_increased_max_power + "max_power"
        # flexconso_op_conso +"to_flex_consumption"*flexconso_op_flex_ratio == "to_flex_consumption"
        flexconso_op_conso = m.add_variables(name="flexconso_op_conso",
                                             lower=0, coords=[Date, AREAS,FLEX_CONSUM])
        flexconso_pl_increased_max_power = m.add_variables(name="flexconso_pl_increased_max_power",
                                                 lower=0, coords=[AREAS,FLEX_CONSUM])
        flexconso_pl_increased_max_power_costs = m.add_variables(name="flexconso_pl_increased_max_power_costs",
                                                                coords=[AREAS,FLEX_CONSUM])
        flexconso_op_flex_ratio = m.add_variables(name="flexconso_op_flex_ratio",coords=[AREAS,FLEX_CONSUM])

        # update of the cost function and of the prod = conso constraint
        m.objective += flexconso_pl_increased_max_power_costs.sum()
        m.constraints['Ctr_Op_conso'].lhs += flexconso_op_conso.sum(['FLEX_CONSUM'])

        Ctr_Op_flexconso_pl_increased_max_power_costs_def = m.add_constraints(name="Ctr_Op_flexconso_pl_increased_max_power_costs_def",
            lhs=flexconso_pl_increased_max_power_costs == Parameters["LoadCost"] * flexconso_pl_increased_max_power)

        Ctr_Op_max_power = m.add_constraints(name="Ctr_Op_max_power",
            lhs=flexconso_op_conso <= flexconso_pl_increased_max_power+ Parameters["max_power"])

        Ctr_Op_conso_flex = m.add_constraints(name="Ctr_Op_conso_flex",
            lhs=flexconso_op_conso +Parameters["to_flex_consumption"]*flexconso_op_flex_ratio == Parameters["to_flex_consumption"])

        Ctr_Op_conso_flex_sup_rule = m.add_constraints(name="Ctr_Op_conso_flex_sup_rule",
            lhs=flexconso_op_flex_ratio <= Parameters["flex_ratio"]) ## Parameters["flex_ratio"] should bve renamed --> Parameters["flex_ratio_max"]

        Ctr_Op_conso_flex_inf_rule = m.add_constraints(name="Ctr_Op_conso_flex_inf_rule",
            lhs=flexconso_op_flex_ratio + Parameters["flex_ratio"] >= 0 )


        Week_of_year_table = period_boolean_table(Date, period="weekofyear")
        Ctr_Op_consum_eq_week = m.add_constraints(name="Ctr_Op_consum_eq_week",
            lhs=(flexconso_op_conso*Week_of_year_table).sum(["Date"])
                <= (Parameters["to_flex_consumption"]*Week_of_year_table).sum(["Date"]),
            mask = Parameters["flex_type"] == "week")

        Day_of_year_table = period_boolean_table(Date, period="day_of_year")
        Ctr_Op_consum_eq_day = m.add_constraints(name="Ctr_Op_consum_eq_day",
            lhs=(flexconso_op_conso*Day_of_year_table).sum(["Date"])
                <= (Parameters["to_flex_consumption"]*Day_of_year_table).sum(["Date"]),
            mask = Parameters["flex_type"] == "day")

        Ctr_Op_consum_eq_year = m.add_constraints(name="Ctr_Op_consum_eq_year",
            lhs=(flexconso_op_conso).sum(["Date"])
                <= (Parameters["to_flex_consumption"]).sum(["Date"]),
            mask = Parameters["flex_type"] == "year")


    return m;


