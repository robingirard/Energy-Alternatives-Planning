import numpy as np
import pandas as pd
import csv
import os
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go


## Q4.1) Multinode
#la méthode de Boiteux
availabilityFactor['availabilityFactor']=1 #ligne de code à rajouter pour enlever la dépendance aux facteurs de disponibilité
(PiNuke,BetaNuke)=(TechParameters[TechParameters['TECHNOLOGIES']=='OldNuke']['energyCost'][1]/1000,TechParameters[TechParameters['TECHNOLOGIES']=='OldNuke']['capacityCost'][1]/1000)
(PiTher,BetaTher)=(TechParameters[TechParameters['TECHNOLOGIES']=='Thermal']['energyCost'][0]/1000,TechParameters[TechParameters['TECHNOLOGIES']=='Thermal']['capacityCost'][0]/1000)
x=np.array(range(0,8760))
y=eval('PiNuke*x+BetaNuke')
y2=eval('PiTher*x+BetaTher')
axes = plt.gca()
axes.set_ylim([0,2000])
plt.ylabel('Average Price')
plt.xlabel('heure (h)')
plt.plot(x,y)
plt.plot(x,y2)
plt.show()



#la réalité qui est limitée par la capacité installée de nucléaire
Variables['energy'].rename(columns={'energy_index': 'TIMESTAMP', 1: 'TECHNOLOGIES'}, inplace=True)
Produc=Variables['energy'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='energy')
Produc=Produc.sort_values(['Thermal'], ascending=[False]).reset_index(drop=True)
Produc2=Produc.sort_values(['OldNuke'], ascending=[False]).reset_index(drop=True)
Produc['Thermal']=Produc['Thermal']+Produc2['OldNuke']

fig=go.Figure()
fig.add_trace(
    go.Scatter(x=list(Produc2.index), y=list(Produc2.OldNuke),fill='tozeroy', mode='none', name="Production Nucléaire"))
fig.add_trace(
    go.Scatter(x=list(Produc2.index), y=list(Produc.Thermal),fill='tonexty', mode='none', name="Production Thermal"))
fig.update_layout(
    title_text="Puissance MW",xaxis_title="heures de l'année")
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="en heures",
                     step="hour",
                     stepmode="backward")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="-"
    )
)
fig.show()

fig2=go.Figure()
fig2.add_trace(
    go.Scatter(x=list(areaConsumption.TIMESTAMP), y=list(areaConsumption.areaConsumption),fill='tozeroy', mode='none', name="Production Nucléaire"))
fig2.update_layout(
    title_text="Puissance MW",xaxis_title="heures de l'année")
fig2.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="en heures",
                     step="hour",
                     stepmode="backward")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="-"
    )
)
fig2.show()

