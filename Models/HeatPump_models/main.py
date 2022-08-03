#region imports
import pandas as pd
import numpy as np

from f_heat_pump import *
Data_Folder = "Models/HeatPump_models/data/"
#endregion


Zone = "AL01"
temp_ts=pd.read_csv(Data_Folder+"era-nuts-t2m-nuts2-hourly-singleindex.csv",parse_dates=['time']).set_index(["time"])
temp_ts_Zone=temp_ts.loc[temp_ts.index.year>2000,[Zone]].rename(columns={Zone : "temp"})
temp_ts_Zone.head()

##ON DUPLIQUE POUR AVEC ET SANS REGULATION DE LA TEMPERATURE DE L'EAU
Systems = pd.read_csv(Data_Folder+"chauffage.csv")
Systems_reg=Systems.copy(); Systems_reg["regulation"]="Y"
Systems_noreg=Systems.copy(); Systems_noreg["regulation"]="N"
Systems= pd.concat([Systems_reg , Systems_noreg],axis=0)

population=pd.read_csv(Data_Folder+"Population_NUTS2.csv").set_index(["unit"]).loc[Zone,:]
# Base temperature is considered as the temperature which occured 5 days per year in the last years (in mean)

Systems.System.unique()
Systems.Technology.unique()
System=Systems.iloc[10,:]
Heating_params = {  "T_start" : 15, "T_target" : 18,
                    "T_base" : np.quantile(temp_ts_Zone, q = 5 / 365)}
Simulation_PAC_input_parameter = {**Heating_params, **System.to_dict()}
Simulation_PAC_input_parameter.keys()
meteo_data = temp_ts_Zone.loc["2018",:]
meteo_data_heating_period= get_heating_period_metdata(meteo_data)

SCOP=estim_SCOP(meteo_data_heating_period=meteo_data_heating_period,
                Simulation_PAC_input_parameter=Simulation_PAC_input_parameter)

tmp_results = estim_COPs(, T_base = T_base, Heating_params = Heating_params, Systems = Systems)