import pandas as pd
import linopy
import xarray as xr
import pandas as pd


def download_file(url,filename):
    response = requests.get(url)
    with open(filename, mode="wb") as file:
        file.write(response.content)
    print(f"Downloaded file {filename}")

def period_boolean_table(date, period):
    """
    returns a boolean xarray table with the dimension date x value(period) with date == value(period)
    :param date:
    :param period: "day_of_year", "weekofyear"
    see https://pandas.pydata.org/docs/reference/api/pandas.Period.html
    :return:
    """
    x_of_year = getattr(date.to_period(), period)
    x_of_year_xr = xr.DataArray(x_of_year, coords={"date": date})
    x_of_year_values = pd.DataFrame(x_of_year).date.unique()
    x_of_year_values_xr = xr.DataArray(x_of_year_values, coords={period: x_of_year_values})
    x_of_year_table = x_of_year_xr == x_of_year_values_xr
    return x_of_year_table

def get_index_in(xr,index,subset):
    return xr.get_index(index)[xr.get_index(index).isin(subset)]

def select(xr,dic):
    reduced_index = {}
    for key in dic:
        reduced_index[key] = get_index_in(xr, key, dic[key])
    return xr.sel(reduced_index)

xr.Dataset.get_index_in=get_index_in
xr.Dataset.select = select

def extractCosts_l(model):

    res = {}
    res["Variable_costs"] = (model.solution["operation_energy_cost"]/ 10 ** 9).to_dataframe()
    res["Variable_costs"].columns = ["Cost_10e9_euros"]
    res["Variable_costs"]["type"] = "annual_energy"
    res["capacity_costs"] = (model.solution["planning_conversion_cost"]/ 10 ** 9).to_dataframe()
    res["capacity_costs"].columns = ["Cost_10e9_euros"]
    res["capacity_costs"]["type"] = "installed_capacity"
    if "planning_flexible_demand_max_power_increase_cost_costs" in model.solution :
        res["flexible_demand_capacity_cost"]= (model.solution["planning_flexible_demand_max_power_increase_cost_cost"]/ 10 ** 9).to_dataframe()
        res["flexible_demand_capacity_cost"].columns = ["Cost_10e9_euros"]
        res["flexible_demand_capacity_cost"]["type"] = "flexible_demand_capacity"
    if "planning_storage_capacity_cost" in model.solution:
        res["storage_capacity_costs"] = (model.solution["planning_storage_capacity_cost"]/ 10 ** 9).to_dataframe()
        res["storage_capacity_costs"].columns = ["Cost_10e9_euros"]
        res["storage_capacity_costs"]["type"] = "storage_capacity"
    return res ## implicitly assuming second index is conversion_technology... strange
    # compute total

    return res

def extractEnergyCapacity_l(model):

    res = {}
    res["production"] = (model.solution["operation_conversion_power_out"]/ 10 ** 6).sum(["date"]).to_dataframe()
    res["production"].columns = ["Energy_TWh"]
    res["production"]["type"] = "annual_energy"
    res["capacity"] = (model.solution["conversion_planning_capacity_out"]/ 10 ** 3).to_dataframe()
    res["capacity"].columns = ["Capacity_GW"]
    res["capacity"]["type"] = "installed_capacity"
    if "planning_flexible_demand_max_power_increase_cost_costs" in model.solution :
        res["flexconso_capacity"]= (model.solution["planning_flexible_demand_max_power_increase_cost"]/ 10 ** 3).to_dataframe()
        res["flexconso_capacity"].columns = ["Capacity_GW"]
        res["flexconso_capacity"]["type"] = "flexible_demand_capacity"
    if "planning_storage_capacity_cost" in model.solution:
        res["storage_capacity"] = (model.solution["planning_storage_max_power"]/ 10 ** 3).to_dataframe()
        res["storage_capacity"].columns = ["Capacity_GW"]
        res["storage_capacity"]["type"] = "storage_capacity"
        res["Variable_storage_in"] = (model.solution["operation_storage_power_in"] / 10 ** 6).sum(["date"]).to_dataframe()
        res["Variable_storage_in"].columns = ["Energy_TWh"]
        res["Variable_storage_in"]["type"] = "storage_in"
        res["Variable_storage_out"] = (model.solution["operation_storage_power_out"] / 10 ** 6).sum(["date"]).to_dataframe()
        res["Variable_storage_out"].columns = ["Energy_TWh"]
        res["Variable_storage_out"]["type"] = "storage_out"

    Myres={}
    Myres["Capacity_GW"] = pd.concat([res[r] for r in res]).set_index(["type"], append=True)[["Capacity_GW"]].\
        dropna().rename({"Capacity_GW": "Capacity_GW"})
    Myres["Energy_TWh"] = pd.concat([res[r] for r in res]).set_index(["type"], append=True)[["Energy_TWh"]].\
        dropna().rename({"Capacity_GW": "Energy_TWh"})
    return Myres


def EnergyAndExchange2Prod(model, EnergyName='energy', exchangeName='Exchange'):
    Variables = {name: model.solution[name].to_dataframe().reset_index() for name in list(model.solution.keys())}
    #Variables["exchange_op_power"].columns = ['area_from', 'area_from_1', 'exchange_op_power']
    area_to = Variables['operation_conversion_power_out'].area_to.unique()
    production_df = Variables['operation_conversion_power_out'].pivot(index=["area_to", "date"], columns='conversion_technology', values='operation_conversion_power_out')
    Import_Export = Variables['exchange_op_power'].groupby(["area_to", "date"]).sum()- Variables['exchange_op_power'].\
        groupby(["area_from", "date"]).sum()
    #if ((Variables['exchange_op_power'].groupby(["area_to", "date"]).sum()*Variables['exchange_op_power'].\
    #        rename(columns={"area_from":"area_from_1","area_from_1":"area_from"}).groupby(["area_from", "date"]).sum()).sum() >0).bool():
    #    print("Problem with import - export")

    production_df = production_df.merge(Import_Export, how='inner', left_on=["area_to", "date"], right_on=["area_to", "date"])
    # exchange analysis
    return (production_df);