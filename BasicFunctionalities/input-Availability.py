#%%
InputFolder='Data/input/'
import numpy as np
import pandas as pd
import csv

import datetime
import copy

import plotly.graph_objects as go
Zones="FR"
year=2013

availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0)


#graphe montrant le facteur de disponibilité du nucléaire en fonction de l'heure de l'année
tabl=availabilityFactor[availabilityFactor['TECHNOLOGIES']=='Solar']
fig=MyPlotly(x_df=tabl.TIMESTAMP,y_df=tabl[['availabilityFactor']],fill=False)
#fig.show()
plotly.offline.plot(fig, filename='file.html')

Zones="FR"
Tech= 'Solar'
year=2013
fig = go.Figure()
for year in range(2013,2016):
    print(year)
    availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0)
    tabl = availabilityFactor[availabilityFactor['TECHNOLOGIES'] == Tech]
    fig.add_trace(go.Scatter(x=tabl['TIMESTAMP'], y=tabl['availabilityFactor'],
                             line=dict(color="#000000"), name="original"))



plotly.offline.plot(fig, filename='file.html')
#graphe montrant le facteur de disponibilité du nucléaire en fonction de l'heure de l'année

fig=MyPlotly(x_df=tabl.TIMESTAMP,y_df=tabl[['availabilityFactor']],fill=False)
#fig.show()
plotly.offline.plot(fig, filename='file.html')
