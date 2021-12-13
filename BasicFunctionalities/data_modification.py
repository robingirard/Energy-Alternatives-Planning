
#region imports
import os
import pandas as pd
import numpy as np
InputFolder='Data/input/'
#endregion

#region ajout des facteurs de charge dy NewNuke et de l'éolien off shore
Zones="FR" ;

for year in range(2013,2017):
    availabilityFactor = pd.read_csv(InputFolder + 'availabilityFactor' + str(year) + '_' + str(Zones) + '.csv',
                                     sep=',', decimal='.', skiprows=0).set_index(["TIMESTAMP", "TECHNOLOGIES"])
    #NewNuke first
    availabilityFactor_NewNuke = availabilityFactor.loc[(slice(None), "OldNuke"), :]
    availabilityFactor = availabilityFactor.append(availabilityFactor_NewNuke.rename(index={"OldNuke": "NewNuke"}))

    #Eolien off shore
    availabilityFactor_WindOffShore = availabilityFactor.loc[(slice(None), "WindOnShore"), :]
    availabilityFactor_WindOffShore = availabilityFactor_WindOffShore.rename(index={"WindOnShore": "WindOffShore"})
    alpha = 1.8
    print(availabilityFactor_WindOffShore.mean())
    availabilityFactor_WindOffShore = availabilityFactor_WindOffShore.assign(
        availabilityFactor=np.where(availabilityFactor_WindOffShore['availabilityFactor'] * alpha >= 1.0, 1.0,
                                    availabilityFactor_WindOffShore[
                                        'availabilityFactor'] * alpha))  # df: 1.0 if df['availabilityFactor']*1.7>=1.0 else df['availabilityFactor']*1.7)
    print(availabilityFactor_WindOffShore.mean())
    availabilityFactor = availabilityFactor.append(availabilityFactor_WindOffShore)
    availabilityFactor.to_csv(InputFolder + 'availabilityFactor_new' + str(year) + '_' + str(Zones) + '.csv',sep=',', decimal='.')


#endregion
