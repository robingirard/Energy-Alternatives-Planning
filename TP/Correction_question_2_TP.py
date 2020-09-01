import numpy as np
import pandas as pd
import csv
import os
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go

## Q2.1) Analyse des différents multiplicateurs de Lagranges
#graphe montrant le facteur de disponibilité en fonction de l'heure de l'année
tabl=availabilityFactor[availabilityFactor['TECHNOLOGIES']=='OldNuke']
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=list(tabl.TIMESTAMP), y=list(tabl.availabilityFactor)))
fig.update_layout(
    title_text="Facteur de dispo dans l'année",xaxis_title="heures de l'année")
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

#graphe montrant la proportion de chaque moyen de prod dans la production totale d'électricité
prod=Variables['energy'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='energy')
prod['Thermal']=prod['Thermal']+prod['OldNuke']
fig2=go.Figure()
fig2.add_trace(
    go.Scatter(x=list(prod.index), y=list(prod.Thermal),name="production Thermal qui complète le Nuke"))
fig2.add_trace(
    go.Scatter(x=list(prod.index), y=list(prod.OldNuke), name="production Nuke"))
fig2.update_layout(
    title_text="Production électrique (en KWh)",xaxis_title="heures de l'année")
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

Variables['energy'].rename(columns={'energy_index': 'TIMESTAMP', 1: 'TECHNOLOGIES'}, inplace=True)
Variables['energy'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='energy') #pour avoir la production en KWh de chaque moyen de prod chaque heure
Variables['energyCosts'] #pour avoir le coût de chaque moyen de prod à l'année

# Analyse energyCtr
round((Constraints['energyCtr']*1000000).energyCtr,2).mean() #pour avoir le coût moyen de l'élec chaque heure
Constraints['energyCtr']['energyCtr']*1000000 #pour obtenir le coût marginal du dernier appelé heure par heure (en €/MWh)
round((Constraints['energyCtr']*1000000).energyCtr,2).unique() #obligation d'arrondir au centième les valeurs du tableau pour retomber sur les deux coût marginaux des moyens de production utilisés (le nuke et le Thermal)

# Analyse CapacityCtr
Constraints['CapacityCtr'].rename(columns={'CapacityCtr_index': 'TIMESTAMP', 1: 'TECHNOLOGIES'}, inplace=True)
round((Constraints['CapacityCtr'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='CapacityCtr')*1000000),2) #permet de voir le surcoût (par techno) engendré par la mise en route d'un autre moyen de production, par exemple dans la colonne Old Nuke, on a un surcoût de 92,92€ lorsque le thermal se met en marche
round((Constraints['CapacityCtr'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='CapacityCtr')*1000000).OldNuke,2).unique() #permet de voir les différents surcoûts possibles lorsqu'on utilise le Nuke.

#Ajout de HydroReservoir et appartition de la contrainte storageCtr
Constraints['storageCtr']*1000000 #nous donne l'économie réalisée par MWh si on utilise l'eau à la place de le thermal que l'énergie la plus coûteuse (en €/MWh, correspond aussi à "coût marginal du thermal" - "coût marginal de HydroReservoir")

## Q2.2) Nouvelle analyse des différents multiplicateurs de Lagranges avec le fichier RAMP
a=Variables['energy'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='energy')['OldNuke']

#permet de vérifier que la contrainte Ramp est bien respectée pour la production nucléaire
def VerifRamp(tab):
    i=1
    while i<len(tab)-1:
        check=abs((tab[i+1]-tab[i])/TechParameters[TechParameters['TECHNOLOGIES']=='OldNuke']['capacity'][1])<=TechParameters[TechParameters['TECHNOLOGIES']=='OldNuke']['RampConstraintPlus'][1]
        if check==False:
            print(abs((tab[i+1]-tab[i])/TechParameters[TechParameters['TECHNOLOGIES']=='OldNuke']['capacity'][1]))
            return('La contrainte Ramp nest pas respectée')
        i=i+1
    return('La contrainte Ramp est respectée')



## Q2.3) Multinode
areaConsumption.pivot(index="TIMESTAMP",columns='AREAS', values='areaConsumption').max()['FR']
areaConsumption.pivot(index="TIMESTAMP",columns='AREAS', values='areaConsumption').min()['FR']
areaConsumption.pivot(index="TIMESTAMP",columns='AREAS', values='areaConsumption').max()['DE']
areaConsumption.pivot(index="TIMESTAMP",columns='AREAS', values='areaConsumption').max()['DE'] #pour connaître conso max et min dans l'année de FR et DE

Variables['exchange'][(Variables['exchange'].iloc[:, 0]=='DE') & (Variables['exchange'].iloc[:, 1]=='FR')] #pour voir les echanges de DE vers FR (que 2000)
Variables['exchange'][(Variables['exchange'].iloc[:, 0]=='FR') & (Variables['exchange'].iloc[:, 1]=='DE')]

Constraints['exchangeCtr'][(Constraints['exchangeCtr'].iloc[:, 0]=='FR') & (Constraints['exchangeCtr'].iloc[:, 1]=='DE')]['exchangeCtr'].unique() #que 0, problèmes ?
Variables['energy'][Variables['energy']['AREAS']=='FR'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='energy') #prod en de chaque type d'énergie en France
#Variables['energy'][Variables['energy']['AREAS']=='FR'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='energy').plot(y=['Thermal','OldNuke', 'WindOnShore', 'Solar'], use_index=True)
#plt.show()

#Consgestion rent ?