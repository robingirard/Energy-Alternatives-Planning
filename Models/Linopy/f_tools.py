import pandas as pd
import linopy
import xarray as xr
import pandas as pd

def period_boolean_table(Date, period):
    """
    returns a boolean xarray table with the dimension Date x value(period) with Date == value(period)
    :param Date:
    :param period: "day_of_year", "weekofyear"
    see https://pandas.pydata.org/docs/reference/api/pandas.Period.html
    :return:
    """
    x_of_year = getattr(Date.to_period(), period)
    x_of_year_xr = xr.DataArray(x_of_year, coords={"Date": Date})
    x_of_year_values = pd.DataFrame(x_of_year).Date.unique()
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
    res["Variable_costs"] = (model.solution["production_op_variable_costs"]/ 10 ** 9).to_dataframe()
    res["Variable_costs"].columns = ["Cost_10e9_euros"]
    res["Variable_costs"]["type"] = "prod_energy"
    res["capacity_costs"] = (model.solution["production_pl_capacity_costs"]/ 10 ** 9).to_dataframe()
    res["capacity_costs"].columns = ["Cost_10e9_euros"]
    res["capacity_costs"]["type"] = "prod_capacity"
    if "flexconso_pl_increased_max_power_costs" in model.solution :
        res["flexconso_capacity_cost"]= (model.solution["flexconso_pl_increased_max_power_costs"]/ 10 ** 9).to_dataframe()
        res["flexconso_capacity_cost"].columns = ["Cost_10e9_euros"]
        res["flexconso_capacity_cost"]["type"] = "flexconso_capacity"
    if "storage_pl_capacity_costs" in model.solution:
        res["storage_capacity_costs"] = (model.solution["storage_pl_capacity_costs"]/ 10 ** 9).to_dataframe()
        res["storage_capacity_costs"].columns = ["Cost_10e9_euros"]
        res["storage_capacity_costs"]["type"] = "storage_capacity"
    return pd.concat([res[r] for r in res]).set_index(["type"],append=True) ## implicitly assuming second index is TECHNOLOGIES... strange
    # compute total

    return res

def extractEnergyCapacity_l(model):

    res = {}
    res["production"] = (model.solution["production_op_power"]/ 10 ** 6).sum(["Date"]).to_dataframe()
    res["production"].columns = ["Energy_TWh"]
    res["production"]["type"] = "prod_energy"
    res["capacity"] = (model.solution["production_pl_capacity"]/ 10 ** 3).to_dataframe()
    res["capacity"].columns = ["Capacity_GW"]
    res["capacity"]["type"] = "prod_capacity"
    if "flexconso_pl_increased_max_power_costs" in model.solution :
        res["flexconso_capacity"]= (model.solution["flexconso_pl_increased_max_power"]/ 10 ** 3).to_dataframe()
        res["flexconso_capacity"].columns = ["Capacity_GW"]
        res["flexconso_capacity"]["type"] = "flexconso_capacity"
    if "storage_pl_capacity_costs" in model.solution:
        res["storage_capacity"] = (model.solution["storage_pl_max_power"]/ 10 ** 3).to_dataframe()
        res["storage_capacity"].columns = ["Capacity_GW"]
        res["storage_capacity"]["type"] = "storage_capacity"
        res["Variable_storage_in"] = (model.solution["storage_op_power_in"] / 10 ** 6).sum(["Date"]).to_dataframe()
        res["Variable_storage_in"].columns = ["Energy_TWh"]
        res["Variable_storage_in"]["type"] = "storage_in"
        res["Variable_storage_out"] = (model.solution["storage_op_power_out"] / 10 ** 6).sum(["Date"]).to_dataframe()
        res["Variable_storage_out"].columns = ["Energy_TWh"]
        res["Variable_storage_out"]["type"] = "storage_out"

    Myres={}
    Myres["Capacity_GW"] = pd.concat([res[r] for r in res]).set_index(["type"], append=True)[["Capacity_GW"]].\
        dropna().rename({"Capacity_GW": "Capacity_GW"})
    Myres["Energy_TWh"] = pd.concat([res[r] for r in res]).set_index(["type"], append=True)[["Energy_TWh"]].\
        dropna().rename({"Capacity_GW": "Energy_TWh"})
    return Myres


def EnergyAndExchange2Prod(Variables, EnergyName='energy', exchangeName='Exchange'):
    #Variables["exchange_op_power"].columns = ['AREAS', 'AREAS_1', 'exchange_op_power']
    AREAS = Variables['production_op_power'].AREAS.unique()
    production_df = Variables['production_op_power'].pivot(index=["AREAS", "Date"], columns='TECHNOLOGIES', values='production_op_power')
    Import_Export = Variables['exchange_op_power'].groupby(["AREAS", "Date"]).sum()- Variables['exchange_op_power'].rename(columns={"AREAS":"AREAS_1","AREAS_1":"AREAS"}).\
        groupby(["AREAS", "Date"]).sum()
    if ((Variables['exchange_op_power'].groupby(["AREAS", "Date"]).sum()*Variables['exchange_op_power'].\
            rename(columns={"AREAS":"AREAS_1","AREAS_1":"AREAS"}).groupby(["AREAS", "Date"]).sum()).sum() >0).bool():
        print("Problem with import - export")

    production_df = production_df.merge(Import_Export, how='inner', left_on=["AREAS", "Date"], right_on=["AREAS", "Date"])
    # exchange analysis
    return (production_df);