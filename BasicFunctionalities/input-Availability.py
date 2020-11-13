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

#region more data from eco2Mix 2012-2019
Eco2mix = pd.read_csv(InputFolder+'Eco2Mix_Hourly_National_xts.csv',
                                sep=';',decimal=',',skiprows=0,
                      dtype={'Index':str, 'Consommation':np.float64, 'Prevision.J1':np.float64, 'Prevision.J':np.float64, 'Fioul':np.float64,
       'Charbon':np.float64, 'Gaz':np.float64, 'Nucleaire':np.float64, 'Eolien':np.float64, 'Solaire':np.float64, 'Hydraulique':np.float64,
       'Pompage':np.float64, 'Bioenergies':np.float64, 'Ech.physiques':np.float64, 'Taux.de.Co2':np.float64,
       'Ech.comm.Angleterre':np.float64, 'Ech.comm.Espagne':np.float64, 'Ech.comm.Italie':np.float64,
       'Ech.comm.Suisse':np.float64, 'Ech.comm.AllemagneBelgique':np.float64, 'Fioul..TAC':np.float64,
       'Fioul..Cogen':np.float64, 'Fioul..Autres':np.float64, 'Gaz..TAC':np.float64, 'Gaz..Cogen':np.float64, 'Gaz..CCG':np.float64,
       'Gaz..Autres':np.float64, 'Hydraulique..Fil.de.leau..eclusee':np.float64, 'Hydraulique..Lacs':np.float64,
       'Hydraulique..STEP.turbinage':np.float64, 'Bioenergies..Dechets':np.float64,
       'Bioenergies..Biomasse':np.float64, 'Bioenergies..Biogaz':np.float64})
Eco2mix['Date'] = pd.to_datetime(Eco2mix['Index'], errors='coerce')
Eco2mix.columns
Eco2mix['year'] = pd.DatetimeIndex(Eco2mix['Date']).year
Eco2mixYear=Eco2mix[Eco2mix['year']==2019]
Hydraulique=Eco2mixYear[['Date','Hydraulique..Lacs','Hydraulique..Fil.de.leau..eclusee']].set_index('Date').rename(
    columns={'Hydraulique..Fil.de.leau..eclusee':'HydroRiver','Hydraulique..Lacs':'HydroLake'})
Hydraulique.max()

Hydraulique.HydroLake.sum()/7000
#endregion