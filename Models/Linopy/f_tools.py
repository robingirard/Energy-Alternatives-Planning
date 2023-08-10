import pandas as pd
import linopy
import xarray as xr

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
    Variables = {name: model.solution[name].to_dataframe().reset_index() for name in list(model.solution.keys())}
    df = Variables['production_pl_capacity_costs'].set_index(["AREAS", "TECHNOLOGIES"]) / 10 ** 9;
    df = df.merge(pd.DataFrame(Variables['production_op_variable_costs'].set_index(["AREAS", "TECHNOLOGIES"]) / 10 ** 9),
                  left_on=["AREAS", "TECHNOLOGIES"], right_on=["AREAS", "TECHNOLOGIES"])
    df.columns = ["Capacity_Milliards_euros", "Energy_Milliards_euros"]

    # compute total
    df["Total_Milliards_euros"] = df.sum(axis=1)
    return df

def extractEnergyCapacity_l(model):
    Variables = {name: model.solution[name].to_dataframe().reset_index() for name in list(model.solution.keys())}
    if len(Variables['production_op_power']['AREAS'].unique()) == 1:
        production_df = Variables['production_op_power'].pivot(index=['AREAS', "Date"], columns='TECHNOLOGIES', values='production_op_power')
    else:
        production_df = EnergyAndExchange2Prod(Variables)
    EnergyCapacity_df = Variables['production_pl_capacity'].set_index(["AREAS", "TECHNOLOGIES"]) / 10 ** 3;
    EnergyCapacity_df = EnergyCapacity_df.merge(
        production_df.reset_index().melt(id_vars=["AREAS","Date"],var_name="TECHNOLOGIES",value_name="Production_TWh").\
        groupby(by=["AREAS", "TECHNOLOGIES"]).sum() / 10 ** 6,
        left_on=["AREAS", "TECHNOLOGIES"], right_on=["AREAS", "TECHNOLOGIES"],
        how="outer")
    EnergyCapacity_df.columns = ["Capacity_GW", "Production_TWh"]

    return EnergyCapacity_df


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