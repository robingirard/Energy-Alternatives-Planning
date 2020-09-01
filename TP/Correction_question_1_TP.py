import plotly.graph_objects as go

from functions.functions_decompose_thermosensibilite import *

os.chdir("le dossier où se trouve les fichiers données") #pour vous placer dans le dossier où se trouvent les donnnées csv


# Question 1 : Handling production and consumption data
# Q1.1. Compute load factor of consumption, production for different years,different thermal sensitivity (france) and different countries


# facteur de charge pour conso sans thermo pour des pays d'Europe
ConsoEuro2016=pd.read_csv('../CSV/input/areaConsumption2016_FR_DE_GB_ES.csv')
pd.set_option('display.max_columns', None)
ConsoEuro2016=ConsoEuro2016.pivot(index = 'TIMESTAMP', columns = 'AREAS', values = 'areaConsumption')
CF1=np.zeros(shape=(ConsoEuro2016.shape[1],1))
CF=pd.DataFrame(data=CF1,index=list(ConsoEuro2016.columns),columns=['load factor for consumption'])
for i in range(ConsoEuro2016.shape[1]):
    Country=list(ConsoEuro2016.columns)[i]
    CF.iloc[i,0]=ConsoEuro2016.mean()[Country]/ConsoEuro2016.max()[Country]
print(CF)


# facteur de charge pour conso avec thermo pour la France
ConsoTemp=pd.read_csv('../CSV/input/ConsumptionTemperature_1996TO2019_FR.csv')
del ConsoTemp['Date']
ConsoTemp=ConsoTemp.sort_values(by = 'Temperature')
ConsoTemp.set_index('Temperature', inplace=True)
maxconso=ConsoTemp.max()[0]
ConsoTemp['Consumption']=ConsoTemp['Consumption']/maxconso
ConsoTemp.columns=['Load factor']
#graph=ConsoTemp.reset_index().plot.scatter(x='Temperature', y='Load factor')
#plt.show(graph) #on voit pas grand chose (trop de points)

#On refait juste sur l'année 2016
ConsoTemp1=pd.read_csv('../CSV/input/ConsumptionTemperature_1996TO2019_FR.csv')
ConsoTemp1['Date'] = pd.to_datetime(ConsoTemp1['Date'])
ConsoTemp1=ConsoTemp1[(ConsoTemp1['Date'] > '2017-12-31 23:00:00') & (ConsoTemp1['Date'] < '2019-01-01 00:00:00')]
del ConsoTemp1['Date']
ConsoTemp1=ConsoTemp1.sort_values(by = 'Temperature')
ConsoTemp1.set_index('Temperature', inplace=True)
maxconso1=ConsoTemp1.max()[0]
ConsoTemp1['Consumption']=ConsoTemp1['Consumption']/maxconso1
ConsoTemp1.columns=['Load factor']
#graph1=ConsoTemp1.reset_index().plot.scatter(x='Temperature', y='Load factor')
#plt.show(graph1) #toujours beaucoup de points

# Q1.2 Renewable dimensioning ----------
#    --- a --- annual energy consumption of 1 country, resp. 5 countries.
#    --- b --- all hourly energy consumption of 1 country, resp. 5 countries.

# Uniquement pour la France pour l'année 2013
AvFactor2013=pd.read_csv("../CSV/input/availabilityFactor2013_FR.csv")
pd.set_option('display.max_columns', None)
AvFactor2013=AvFactor2013.pivot(index = 'TIMESTAMP', columns = 'TECHNOLOGIES', values = 'availabilityFactor')
ConsoFR2013=pd.read_csv('../CSV/input/areaConsumption2013_FR.csv')
ConsoAnnuelleFR2013=ConsoFR2013.sum()['areaConsumption'] #en MWh

InstalledCapa1=np.zeros(shape=(1,3))
InstalledCapa=pd.DataFrame(data=InstalledCapa1,index=['FR'],columns=['PVa à puissance nominale','WPa à puissance nominale','WPb à puissance nominale'])
InstalledCapa.iloc[0,0]=ConsoAnnuelleFR2013/AvFactor2013.sum()['Solar']
InstalledCapa.iloc[0,1]=ConsoAnnuelleFR2013/AvFactor2013.sum()['WindOnShore']
# Impossible de couvrir la conso heure par heure qu'avec du solaire car il y a des heures (la nuit) où la production solaire est nulle
InstalledCapa.iloc[0,2]=(ConsoFR2013['areaConsumption']/AvFactor2013['WindOnShore']).max()
InstalledCapa=InstalledCapa/1000 # en GW
print(InstalledCapa)

# Q1.3 impact of thermal sensitivity ----------
# for 2016 France what is the impact of thermal sensitivity on Q1.2--a-- and --b--
# this together with Q4

# On fait cette question pour la France avec l'année 2013

def EffetThermosensibilite(alpha):
    # alpha est un coefficient (choisi arbitrairement) que l'on fait varier entre 0 et 3 (par exemple) pour amplifier ou diminuer l'effet de la thermosensibilité sur la consommation d'électricité
    AvFactor2013=pd.read_csv("../CSV/input/availabilityFactor2013_FR.csv")
    pd.set_option('display.max_columns', None)
    AvFactor2013=AvFactor2013.pivot(index = 'TIMESTAMP', columns = 'TECHNOLOGIES', values = 'availabilityFactor')
    (DecomposedConso, Thermosens)=Decomposeconso(2013)
    NewThermosens=Thermosens*alpha
    NewDecomposedConso=RecomposeTemperature(DecomposedConso, NewThermosens)
    NewConsoAnnuelleFR2013=(NewDecomposedConso['Conso thermo']+NewDecomposedConso['Conso Nonthermo']).sum()
    InstalledCapa1=np.zeros(shape=(1,3))
    InstalledCapa=pd.DataFrame(data=InstalledCapa1,index=['FR'],columns=['PVa à puissance nominale','WPa à puissance nominale','WPb à puissance nominale'])
    InstalledCapa.iloc[0,0]=NewConsoAnnuelleFR2013/AvFactor2013.sum()['Solar']
    InstalledCapa.iloc[0,1]=NewConsoAnnuelleFR2013/AvFactor2013.sum()['WindOnShore']
    # Impossible de couvrir la conso heure par heure qu'avec du solaire car il y a des heures (la nuit) où la production solaire est nulle
    InstalledCapa.iloc[0,2]=((NewDecomposedConso['Conso thermo']+NewDecomposedConso['Conso Nonthermo']).reset_index(drop=True)/AvFactor2013['WindOnShore']).max()
    InstalledCapa=InstalledCapa/1000 # en GW
    return(InstalledCapa)

# Q1.4) How much nuclear should you build to cover consumption at any hour. Take into account availability. How does thermal sensitivity impact the results ?

InstalledCapaNuke=(ConsoFR2013['areaConsumption']/AvFactor2013['OldNuke']).max()
InstalledCapaNuke=InstalledCapaNuke/1000 # en GW
print('La puissance nucléaire installée devrait être', InstalledCapaNuke, 'GW pour couvrir la conso horaire.')

# Et si on veut tenir compte de la thermosensibilité

def EffetThermosensibiliteNuke(alpha):
    # alpha est un coefficient (choisi arbitrairement) que l'on fait varier entre 0 et 3 (par exemple) pour amplifier ou diminuer l'effet de la thermosensibilité sur la consommation d'électricité
    AvFactor2013=pd.read_csv("../CSV/input/availabilityFactor2013_FR.csv")
    pd.set_option('display.max_columns', None)
    AvFactor2013=AvFactor2013.pivot(index = 'TIMESTAMP', columns = 'TECHNOLOGIES', values = 'availabilityFactor')
    (DecomposedConso, Thermosens)=Decomposeconso(2013)
    NewThermosens=Thermosens*alpha
    NewDecomposedConso=RecomposeTemperature(DecomposedConso, NewThermosens)
    NewConsoAnnuelleFR2013=(NewDecomposedConso['Conso thermo']+NewDecomposedConso['Conso Nonthermo']).sum()
    InstalledCapaNuke=((NewDecomposedConso['Conso thermo']+NewDecomposedConso['Conso Nonthermo']).reset_index(drop=True)/AvFactor2013['OldNuke']).max()
    InstalledCapaNuke=InstalledCapaNuke/1000
    return('Avec leffet de la thermosensibilité :', InstalledCapaNuke, 'GW')

# Q1.5) Suppose you have a storage with infinite energy capacity, a WP installed capacity of 300 GW, a storage capacity of 65GW, what are the percentages (for a year) of storage WP energy, WP energy losses (if the effiency is 60%) and curtailment ?

AvFactor2013=pd.read_csv("../CSV/input/availabilityFactor2013_FR.csv")
AvFactor2013WP=AvFactor2013[AvFactor2013['TECHNOLOGIES']=='WindOnShore']
del AvFactor2013WP['TECHNOLOGIES']
InstallCapaWP=300000 #puissance éolienne intallée en MW
table1=np.zeros(shape=(ConsoFR2013.shape[0],2))
table=pd.DataFrame(data=table1,index=AvFactor2013WP['TIMESTAMP'],columns=['Conso','prodWP']).reset_index(drop=True)
table['prodWP']=(AvFactor2013WP['availabilityFactor']*InstallCapaWP).reset_index(drop=True)
table['Conso']=ConsoFR2013['areaConsumption']
table['WPconsommé']=table[['prodWP', 'Conso']].min(axis=1)
table['stockage']=table[['prodWP', 'Conso']].max(axis=1)
for i in range(ConsoFR2013.shape[0]):
    if table['stockage'][i]>table['Conso'][i]+65000: #65000MW est notre capacité de stockage
        table['stockage'][i]=table['Conso'][i]+65000
table['curtailment']=table[['prodWP', 'Conso']].max(axis=1)

fig=go.Figure()
fig.add_trace(
    go.Scatter(x=list(table.index), y=list(table.WPconsommé),fill='tozeroy', mode='none', name="Consommation d'origine éolienne"))
fig.add_trace(
    go.Scatter(x=list(table.index), y=list(table.Conso),fill='tonexty', mode='none', name="Consommation d'origine stockage"))
fig.add_trace(
    go.Scatter(x=list(table.index), y=list(table.stockage), fill='tonexty', mode='none', name="production stockée"))
fig.add_trace(
    go.Scatter(x=list(table.index), y=list(table.curtailment), fill='tonexty', mode='none', name="production perdue"))
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

result1=np.zeros(shape=(2,3))
index=['Energy (TWh)', 'Energy (% tot)']
result=pd.DataFrame(data=result1, index=index, columns=['Storage (in) ','Storage losses', 'curtailment' ])
result.iloc[0,0]=(table.sum()['stockage']-table.sum()['Conso'])/1000000
result.iloc[0,1]=result.iloc[0,0]*0.4
result.iloc[0,2]=(table.sum()['curtailment']-table.sum()['stockage'])/1000000
result.iloc[1,0]=result.iloc[0,0]/(table.sum()['prodWP']/1000000)*10
result.iloc[1,1]=result.iloc[0,1]/(table.sum()['prodWP']/1000000)*10
result.iloc[1,2]=result.iloc[0,2]/(table.sum()['prodWP']/1000000)*10