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


def Build_EAP_Model(parameters):
    """
    This function creates the pyomo model and initlize the parameters and (pyomo) Set values
    :param parameters is a dictionnary with different panda tables :
        - energy_demand: panda table with consumption
        - operation_conversion_availability_factor: panda table
        - conversion_technology_parameters : panda tables indexed by conversion_technology with eco and tech parameters
    """

    ## Starting with an empty model object
    m = linopy.Model()

    ### Obtaining dimensions values
    date = parameters.get_index('date').unique()
    energy_vector_out = parameters.get_index('energy_vector_out').unique()
    energy_vector_in = parameters.get_index('energy_vector_in').unique()
    area_to = parameters.get_index('area_to')
    conversion_technology = parameters.get_index('conversion_technology').unique()
    #TODO : remove french

    # Variables - Base - Operation & Planning
    operation_conversion_power = m.add_variables(name="operation_conversion_power", lower=0, coords=[energy_vector_out,area_to,date,conversion_technology]) ### Energy produced by a production mean at time t
    operation_energy_cost = m.add_variables(name="operation_energy_cost", lower=0, coords=[area_to,energy_vector_in]) ### Energy total marginal cost for production mean p
    operation_total_hourly_demand = m.add_variables(name="operation_total_hourly_demand",lower=0, coords=[energy_vector_out, area_to,date])
    operation_yearly_importation = m.add_variables(name="operation_yearly_importation",lower=0, coords=[area_to, energy_vector_in])

    planning_conversion_cost = m.add_variables(name="planning_conversion_cost", lower=0, coords=[energy_vector_out,area_to,conversion_technology]) ### Energy produced by a production mean at time t
    planning_conversion_power_capacity = m.add_variables(name="planning_conversion_power_capacity", lower=0, coords=[energy_vector_out,area_to,conversion_technology]) ### Energy produced by a production mean at time t

    #operation_total_yearly_demand = m.add_variables(name="operation_total_yearly_demand",lower=0, coords=[energy_vector_out, area_to])

    # Variable - Storage - Operation & Planning
    # Objective Function (terms to be added later in the code for storage and flexibility)
    cost_function = planning_conversion_cost.sum()+ operation_energy_cost.sum()
    m.add_objective( cost_function)
    #TODO : implement time slices
    #################
    # Constraints   #
    #################

    ## table of content for the constraints definition - Operation (Op) & Planning (Pl)
    # 1 - Main constraints
    # 2 - Optional constraints
    # 3 - Storage constraints
    # 4 - exchange constraints
    # 5 - DSM constraints


    #####################
    # 1 - a - Main Constraints - Operation (Op) & Planning (Pl)

    Ctr_Op_operation_costs = m.add_constraints(name="Ctr_Op_operation_costs",
        # [ area_to x energy_vector_in ]
        lhs = operation_energy_cost == parameters["operation_energy_unit_cost"] * operation_yearly_importation)

    Ctr_Op_conso_hourly = m.add_constraints(name="Ctr_Op_conso_hourly",
        # [area_to x energy_vector_out x date]
        lhs = operation_total_hourly_demand == parameters["exogenous_energy_demand"])

    conversion_mean_energy_vector_in = (parameters['energy_vector_in'] == parameters["energy_vector_in_value"])/parameters["operation_conversion_efficiency"]
    Ctr_Op_conso_yearly_1 = m.add_constraints(#name="Ctr_Op_conso_yearly_1",
        # [energy_vector_in x area_to ]
        ## case where energy_vector_in value is not in energy_vector_out (e.g. all but elec), meaning that there is no Ctr_Op_conso_hourly associated constraint
        operation_yearly_importation == (conversion_mean_energy_vector_in * operation_conversion_power).sum(["date","energy_vector_out","conversion_technology"]),
        mask= ~parameters["operation_energy_unit_cost"]['energy_vector_in'].isin(energy_vector_out))

    #parameters["operation_energy_unit_cost"]['energy_vector_in_value']
    #(operation_conversion_power * parameters["operation_conversion_efficiency"]).sum(["conversion_technology"]))
    energy_vector_in_in_energy_vector_out = parameters["operation_energy_unit_cost"]['energy_vector_in']==parameters["energy_vector_out"]
    Ctr_Op_operation_demand = m.add_constraints(name="Ctr_Op_operation_demand",
        # [energy_vector_out x area_to ]
        ## case where energy_vector_in value is in energy_vector_out, meaning that there is Ctr_Op_conso_hourly associated constraint
        lhs =  operation_total_hourly_demand  == operation_conversion_power.sum(["conversion_technology"])
    )
    #(operation_yearly_importation * energy_vector_in_in_energy_vector_out).sum(["energy_vector_in"]) +

    Ctr_Pl_capacity = m.add_constraints(name="Ctr_Pl_capacity", # contrainte de maximum de production
        lhs = operation_conversion_power <= planning_conversion_power_capacity * parameters["operation_conversion_availability_factor"])

    Ctr_Pl_planning_conversion_costs = m.add_constraints(name="Ctr_Pl_planning_conversion_costs", # contrainte de définition de planning_conversion_costs
        lhs = planning_conversion_cost == parameters["planning_conversion_unit_cost"] * planning_conversion_power_capacity )

    Ctr_Pl_planning_max_capacity = m.add_constraints(name="Ctr_Pl_planning_max_capacity",
        lhs = planning_conversion_power_capacity <= parameters["planning_conversion_max_capacity"],
        mask=parameters["planning_conversion_max_capacity"] > 0)

    Ctr_Pl_planning_min_capacity = m.add_constraints(name="Ctr_Pl_planning_min_capacity",
        lhs = planning_conversion_power_capacity >= parameters["planning_conversion_min_capacity"],
        mask=parameters["planning_conversion_min_capacity"] > 0)


    #####################
    # 2 - Optional Constraints - Operation (Op) & Planning (Pl)

    if "operation_conversion_maximum_working_hours" in parameters: Ctr_Op_stock = m.add_constraints(name="stockCtr",
            lhs = parameters["operation_conversion_maximum_working_hours"] * planning_conversion_power_capacity >= operation_conversion_power.sum(["date"]),
            mask = parameters["operation_conversion_maximum_working_hours"] > 0)

    if "operation_max_1h_ramp_rate" in parameters: Ctr_Op_rampPlus = m.add_constraints(name="Ctr_Op_rampPlus",
            lhs = operation_conversion_power.diff("date", n=1) <=planning_conversion_power_capacity
                  * (parameters["operation_max_1h_ramp_rate"] * parameters["operation_conversion_availability_factor"])  ,
            mask= (parameters["operation_max_1h_ramp_rate"] > 0) *
                  (xr.DataArray(date, coords=[date])!=date[0]))

    if "operation_min_1h_ramp_rate" in parameters: Ctr_Op_rampMoins = m.add_constraints(name="Ctr_Op_rampMoins",
            lhs = operation_conversion_power.diff("date", n=1) + planning_conversion_power_capacity
                  * (parameters["operation_min_1h_ramp_rate"] * parameters["operation_conversion_availability_factor"]) >= 0,
            # remark : "-" sign not possible in lhs, hence the inequality alternative formulation
            mask=(parameters["operation_min_1h_ramp_rate"] > 0) *
                 (xr.DataArray(date, coords=[date])!=date[0]))

    if "operation_max_1h_ramp_rate2" in parameters:
        Ctr_Op_rampPlus2 = m.add_constraints(name="Ctr_Op_rampPlus2",
            lhs = operation_conversion_power.diff("date", n=2) <= planning_conversion_power_capacity
                  * (parameters["operation_max_1h_ramp_rate2"] * parameters["operation_conversion_availability_factor"]),
            mask=(parameters["operation_max_1h_ramp_rate2"] > 0) *
                 (xr.DataArray(date, coords=[date])!=date[0]))

    if "operation_min_1h_ramp_rate2" in parameters:
        Ctr_Op_rampMoins2 = m.add_constraints(name="Ctr_Op_rampMoins2",
            lhs = operation_conversion_power.diff("date", n=2)+planning_conversion_power_capacity
                  * (parameters["operation_min_1h_ramp_rate2"] * parameters["operation_conversion_availability_factor"]) >= 0,
            mask=(parameters["operation_min_1h_ramp_rate2"] > 0) *
                 (xr.DataArray(date, coords=[date])!=date[0]))



    #####################
    # 3 -  Storage Constraints - Operation (Op) & Planning (Pl)

    if "storage_technology" in parameters:
        storage_technology = parameters.get_index('storage_technology')

        operation_storage_power_in = m.add_variables(name="operation_storage_power_in", lower=0,coords = [date,area_to,energy_vector_out,storage_technology])  ### Energy stored in a storage mean at time t
        operation_storage_power_out = m.add_variables(name="operation_storage_power_out", lower=0,coords = [date,area_to,energy_vector_out,storage_technology])  ### Energy taken out of a storage mean at time t
        operation_storage_internal_energy_level = m.add_variables(name="operation_storage_internal_energy_level", lower=0,coords = [date,area_to,energy_vector_out,storage_technology])  ### level of the energy stock in a storage mean at time t
        planning_storage_energy_cost = m.add_variables(name="planning_storage_energy_cost",coords = [area_to,energy_vector_out,storage_technology])  ### Cost of storage for a storage mean, explicitely defined by definition planning_storage_capacity_costsDef
        planning_storage_energy_capacity = m.add_variables(name="planning_storage_energy_capacity",coords = [area_to,energy_vector_out,storage_technology])  # Maximum capacity of a storage mean
        planning_storage_power_capacity = m.add_variables(name="planning_storage_power_capacity",coords = [area_to,energy_vector_out,storage_technology])  # Maximum flow of energy in/out of a storage mean
        #storage_op_stockLevel_ini = m.add_variables(name="storage_op_stockLevel_ini",coords = [area_from,storage_technology], lower=0)

        #update of the cost function and of the prod = conso constraint
        m.objective += planning_storage_energy_cost.sum()
        m.constraints['Ctr_Op_operation_demand'].lhs += -operation_storage_power_out.sum(['storage_technology'])+operation_storage_power_in.sum(['storage_technology'])

        Ctr_Pl_planning_storage_capacity_costs = m.add_constraints(name="Ctr_Pl_planning_storage_capacity_costs",
             lhs=planning_storage_energy_cost == parameters["planning_storage_energy_unit_cost"] * planning_storage_energy_capacity)

        Ctr_Op_storage_level_definition = m.add_constraints(name="Ctr_Op_storage_level_definition",
            lhs=operation_storage_internal_energy_level.shift(date=1) == operation_storage_internal_energy_level * (1 - parameters["operation_storage_dissipation"])+operation_storage_power_in* parameters["operation_storage_efficiency_in"]
                                                        - operation_storage_power_out / parameters["operation_storage_efficiency_out"],
            mask=xr.DataArray(date, coords=[date]) != date[0]) # voir si ce filtre est vraiment nécessaire

        Ctr_Op_storage_initial_level = m.add_constraints(name="Ctr_Op_storage_initial_level",
            lhs= operation_storage_internal_energy_level.loc[{"date" : date[0] }] == operation_storage_internal_energy_level.loc[{"date" : date[-1] }])

        Ctr_Op_storage_power_in_max = m.add_constraints(name="Ctr_Op_storage_power_in_max",
            lhs=operation_storage_power_in <= planning_storage_power_capacity)

        Ctr_Op_storage_power_out_max = m.add_constraints(name="Ctr_Op_storage_power_out_max",
            lhs=operation_storage_power_out <= planning_storage_power_capacity)

        Ctr_Op_storage_capacity_max = m.add_constraints(name="Ctr_Op_storage_capacity_max",
            lhs=operation_storage_internal_energy_level <= planning_storage_energy_capacity)

        # TODO problem when parameters["planning_storage_max_capacity"] is set to zero

        Ctr_Pl_storage_max_capacity = m.add_constraints(name="Ctr_Pl_storage_max_capacity",
             lhs=planning_storage_energy_capacity <= parameters["planning_storage_max_energy_capacity"])

        Ctr_Pl_storage_min_capacity = m.add_constraints(name="Ctr_Pl_storage_min_capacity",
             lhs=planning_storage_energy_capacity >= parameters["planning_storage_min_energy_capacity"])

        Ctr_Pl_storage_max_power = m.add_constraints(name="Ctr_Pl_storage_max_power",
             lhs=planning_storage_energy_capacity == planning_storage_power_capacity * parameters["operation_storage_hours_of_stock"])
    #####################
    # 4 -  Exchange Constraints - Operation (Op) & Planning (Pl)

    if len(area_to)>1:
        area_from=  parameters.get_index('area_from')
        exchange_op_power = m.add_variables(name="exchange_op_power", lower=0,coords = [date, area_to,area_from,energy_vector_out])  ### Energy stored in a storage mean at time t
        #TODO utiliser swap_dims https://docs.xarray.dev/en/stable/generated/xarray.Dataset.swap_dims.html#xarray.Dataset.swap_dims
        m.constraints['Ctr_Op_operation_demand'].lhs += - exchange_op_power.sum(['area_from']) + exchange_op_power.rename({'area_to':'area_from','area_from':'area_to'}).sum(['area_from'])
        Ctr_Op_exchange_max = m.add_constraints(name="Ctr_Op_exchange_max",
            lhs=exchange_op_power<= parameters["operation_exchange_max_capacity"])
        m.objective += 0.01 * exchange_op_power.sum()
        #TODO change area_from_1 area_from_from  area_from_to


    # Flex consumption
    if "flexible_demand" in parameters:
        flexible_demand = parameters.get_index('flexible_demand')
        # inscrire les équations ici ?
        # planning_flexible_demand_max_power_increase_cost_costs = "flexible_demand_planning_cost" * planning_flexible_demand_max_power_increase_cost
        # operation_total_hourly_demand <= planning_flexible_demand_max_power_increase_cost + "max_power"
        # operation_flexible_demand +"flexible_demand_to_optimise"*operation_flexible_demand_variation_ratio == "flexible_demand_to_optimise"
        operation_flexible_demand = m.add_variables(name="operation_flexible_demand",
                                             lower=0, coords=[date, area_to,energy_vector_out,flexible_demand])
        planning_flexible_demand_max_power_increase = m.add_variables(name="planning_flexible_demand_max_power_increase",
                                                 lower=0, coords=[area_to,energy_vector_out,flexible_demand])
        planning_flexible_demand_cost = m.add_variables(name="planning_flexible_demand_cost",
                                                        lower=0,   coords=[area_to,energy_vector_out,flexible_demand])
        operation_flexible_demand_variation_ratio = m.add_variables(name="operation_flexible_demand_variation_ratio",lower=0,coords=[date,area_to,energy_vector_out,flexible_demand])

        # update of the cost function and of the prod = conso constraint
        m.objective += planning_flexible_demand_cost.sum()
        m.constraints['Ctr_Op_operation_demand'].lhs += operation_flexible_demand.sum(['flexible_demand'])

        Ctr_Op_planning_flexible_demand_max_power_increase_def = m.add_constraints(name="Ctr_Op_planning_flexible_demand_max_power_increase_def",
            lhs=planning_flexible_demand_cost == parameters["flexible_demand_planning_unit_cost"] * planning_flexible_demand_max_power_increase)

        Ctr_Oplanning_storage_max_power_power = m.add_constraints(name="Ctr_Oplanning_storage_max_power_power",
            lhs=operation_flexible_demand <= planning_flexible_demand_max_power_increase + parameters["flexible_demand_max_power"])

        Ctr_Op_conso_flex = m.add_constraints(name="Ctr_Op_conso_flex",
            lhs=operation_flexible_demand +parameters["flexible_demand_to_optimise"]*operation_flexible_demand_variation_ratio == parameters["flexible_demand_to_optimise"])

        Ctr_Op_conso_flex_sup_rule = m.add_constraints(name="Ctr_Op_conso_flex_sup_rule",
            lhs=operation_flexible_demand_variation_ratio <= parameters["flexible_demand_ratio"]) ## parameters["flexible_demand_ratio"] should bve renamed --> parameters["flexible_demand_ratio_max"]

        Ctr_Op_conso_flex_inf_rule = m.add_constraints(name="Ctr_Op_conso_flex_inf_rule",
            lhs=operation_flexible_demand_variation_ratio + parameters["flexible_demand_ratio"] >= 0 )


        week_of_year_table = period_boolean_table(date, period="weekofyear")
        Ctr_Op_consum_eq_week = m.add_constraints(name="Ctr_Op_consum_eq_week",
            lhs=(operation_flexible_demand*week_of_year_table).sum(["date"])
                <= (parameters["flexible_demand_to_optimise"]*week_of_year_table).sum(["date"]),
            mask = parameters["flexible_demand_period"] == "week")

        day_of_year_table = period_boolean_table(date, period="day_of_year")
        Ctr_Op_consum_eq_day = m.add_constraints(name="Ctr_Op_consum_eq_day",
            lhs=(operation_flexible_demand*day_of_year_table).sum(["date"])
                <= (parameters["flexible_demand_to_optimise"]*day_of_year_table).sum(["date"]),
            mask = parameters["flexible_demand_period"] == "day")

        Ctr_Op_consum_eq_year = m.add_constraints(name="Ctr_Op_consum_eq_year",
            lhs=(operation_flexible_demand).sum(["date"])
                <= (parameters["flexible_demand_to_optimise"]).sum(["date"]),
            mask = parameters["flexible_demand_period"] == "year")


    return m;



