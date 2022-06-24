
#regions importation of modules
import os
if os.path.basename(os.getcwd())=="BasicFunctionalities":
    os.chdir('../..') ## to work at project root  like in any IDE

InputFolder='Data/input/'
import numpy as np
import pandas as pd
import csv
import datetime
import copy
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from functions.f_graphicalTools import * #Il faut préciser le chemin où vous avez sauvegardé les données csv
#endregions

Zones="FR"
year=2013

availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0)
print(availabilityFactor.head())


availabilityFactor.iloc[0,2]
availabilityFactor.iloc[[0,1],[0,1,2]]

print(availabilityFactor['TECHNOLOGIES'].unique() ) ### available technologies

MyTech= 'OldNuke'  ### 'Thermal' 'OldNuke' 'HydroRiver' 'HydroReservoir' 'WindOnShore' 'Solar'

availabilityFactor.loc[availabilityFactor['TECHNOLOGIES']==MyTech,:].head()
availabilityFactor[availabilityFactor['TECHNOLOGIES']==MyTech].head()

availabilityFactor[availabilityFactor['TECHNOLOGIES']==MyTech].head()

availabilityFactor_with_index = availabilityFactor.set_index(["TIMESTAMP","TECHNOLOGIES"])
availabilityFactor_with_index.head()

availabilityFactor_with_index.loc[(1,'OldNuke'),]
availabilityFactor_with_index.loc[([1,2],'OldNuke'),]
availabilityFactor_with_index.loc[(range(1,10),'OldNuke'),]
availabilityFactor_with_index.loc[(slice(None),'OldNuke'),]

availabilityFactor_with_index.reset_index()
availabilityFactor_with_index.loc[(slice(None),'OldNuke'),].reset_index()[["TIMESTAMP", "availabilityFactor"]].set_index(["TIMESTAMP"])

availabilityFactor_pivot=availabilityFactor.pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='availabilityFactor')
availabilityFactor_pivot.head()
availabilityFactor_pivot.mean(axis=0)
availabilityFactor_pivot.reset_index().melt("TIMESTAMP", var_name='TECHNOLOGIES', value_name='availabilityFactor')