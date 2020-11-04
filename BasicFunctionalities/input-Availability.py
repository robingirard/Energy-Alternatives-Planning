InputFolder='Data/input/'

#region importation of modules
import numpy as np
import pandas as pd
import seaborn as sns
import csv
import datetime
import copy
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from functions.f_graphicalTools import * #Il faut préciser le chemin où vous avez sauvegardé les données csv
#endregion


#region Disponibilité sur la France période 2013-2016
Zones="FR"
year=2013

MyTech= 'OldNuke'  ### 'Thermal' 'OldNuke' 'HydroRiver' 'HydroReservoir' 'WindOnShore' 'Solar'
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0)
print(availabilityFactor['TECHNOLOGIES'].unique() ) ### available technologies
tabl=availabilityFactor[availabilityFactor['TECHNOLOGIES']==MyTech]
fig=MyPlotly(x_df=tabl.TIMESTAMP,y_df=tabl[['availabilityFactor']],fill=False)

fig = go.Figure()
fig.add_trace(go.Scatter(x=tabl['TIMESTAMP'],y=tabl['availabilityFactor'],line=dict(color="#000000"),name="original"))
for newyear in range(2013,2016):
    availabilityFactor = pd.read_csv(InputFolder + 'availabilityFactor' + str(newyear) + '_' + str(Zones) + '.csv',
                                     sep=',', decimal='.', skiprows=0)
    tabl = availabilityFactor[availabilityFactor['TECHNOLOGIES'] == MyTech]
    fig.add_trace(go.Scatter(x=tabl['TIMESTAMP'],y=tabl['availabilityFactor'],
                             line=dict(color="#9CA2A8",width=1),
                             name=newyear))
#fig.show()
plotly.offline.plot(fig, filename='file.html')
#endregion



#region multi zone
Zones="FR_DE_GB_ES"
year=2016
Selected_AREAS={"FR","DE"}
MyTech= 'WindOnShore'  ### 'Thermal' 'OldNuke' 'HydroRiver' 'HydroReservoir' 'WindOnShore' 'Solar'
 #'NewNuke', 'HydroRiver', 'HydroReservoir','WindOnShore', 'WindOffShore', 'Solar', 'Curtailement'}

#### reading CSV files
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0)
availabilityFactor0 = availabilityFactor[availabilityFactor['TECHNOLOGIES'] == MyTech]
fig = go.Figure()
pal = sns.color_palette("bright", 4); i=0; #https://chrisalbon.com/python/data_visualization/seaborn_color_palettes/
for region in ["FR","DE","GB"]: ### problem with spain data
    tabl=availabilityFactor0[availabilityFactor0['AREAS']==region]
    fig.add_trace(go.Scatter(x=tabl['TIMESTAMP'],y=tabl['availabilityFactor'],
                             line=dict(color=pal.as_hex()[i],width=1),
                             name=region))
    i=i+1;
#fig.show()
plotly.offline.plot(fig, filename='file.html')
#endregion