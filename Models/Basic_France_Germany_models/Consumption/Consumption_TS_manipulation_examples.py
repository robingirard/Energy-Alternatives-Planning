InputFolder='Models/Basic_France_Germany_models/Consumption/Data/'

#region importation of modules
import numpy as np
import seaborn as sns
import pandas as pd
import csv
import datetime
import copy
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn import linear_model
from EnergyAlternativesPlaning.f_consumptionModels import * #Il faut préciser le chemin où vous avez sauvegardé les données csv
from EnergyAlternativesPlaning.f_graphicalTools import * #Il faut préciser le chemin où vous avez sauvegardé les données csv
#endregion


#region multi zone
Zones="FR_DE_GB_ES"
year=2016 #only possible year

#### reading CSV files
areaConsumption = pd.read_csv(InputFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0,parse_dates=['Date']).set_index(["AREAS","Date"])
fig = go.Figure()
pal = sns.color_palette("bright", 4); i=0; #https://chrisalbon.com/python/data_visualization/seaborn_color_palettes/
for region in ["FR","DE","ES"]: ### problem with spain data
    #tabl=areaConsumption[(region,slice(None))]
    fig.add_trace(go.Scatter(x=areaConsumption.index.get_level_values("Date"),
                             y=areaConsumption.loc[(region,slice(None)),"areaConsumption"],
                             line=dict(color=pal.as_hex()[i],width=1),
                             name=region))
    i=i+1;
#fig.show()
plotly.offline.plot(fig, filename='file.html')
#endregion

