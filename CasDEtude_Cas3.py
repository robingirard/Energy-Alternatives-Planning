from dynprogstorage.wrappers import *
from functions.functions_Operation import *

#####GESTION
############################
#### Single Node Operation #
############################

Zones="FR"
year=2013

Selected_TECHNOLOGIES={'Thermal', 'WindOnShore', 'Solar'}

#### reading CSV files
areaConsumption = pd.read_csv('CSV/input/areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0)
availabilityFactor = pd.read_csv('CSV/input/availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0)
TechParameters = pd.read_csv('Data/input/Gestion-SimpleCas3_TECHNOLOGIES.csv', sep=';', decimal=',', skiprows=0)

#### Selection of subset
availabilityFactor=availabilityFactor[ availabilityFactor.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]
TechParameters=TechParameters[TechParameters.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]

availabilityFactor.head()

model = GetElectricSystemModel_GestionSingleNode(areaConsumption,availabilityFactor,TechParameters)
opt = SolverFactory('mosek')
results=opt.solve(model)


#### obtain optimized variables in panda form
Variables=getVariables_panda(model)
Variables.keys()

#### obtain lagrange multipliers in panda form
Constraints= getConstraintsDual_panda(model)
Constraints.keys()
Constraints["energyCtr"].assign(Prix=lambda x: x.energyCtr *10**6).head()
Variables['energy'].rename(columns={'energy_index': 'TIMESTAMP', 1: 'TECHNOLOGIES'}, inplace=True)

### Tracés

Var=Variables['energy'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='energy')

fig=go.Figure()
fig.add_trace(
    go.Scatter(x=list(Var.index), y=list(Var.WindOnShore), name="Production WP on"))
fig.add_trace(
    go.Scatter(x=list(Var.index), y=list(Var.Solar), name="Production Solar"))
fig.add_trace(
    go.Scatter(x=list(Var.index), y=list(Var.Thermal), name="Production Thermal"))
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

####WITH STORAGE
##### Loop
PrixTotal={}
Consommation={}
LMultipliers={}
DeltaPrix={}
Deltazz={}
CostFunction={}
TotalCols={}
zz={}
#p_max_out=100.; p_max_in=100.; c_max=10.;
p_max=1000.;  c_max=10.*p_max;
areaConsumption["NewConsumption"]=areaConsumption["areaConsumption"]
nbTime=len(areaConsumption["areaConsumption"])
cpt=0
for i in model.areaConsumption:  model.areaConsumption[i]=areaConsumption.NewConsumption[i-1]
for cpt in range(10):
    print(cpt)
    if (cpt==0): zz[cpt]=[0]*nbTime
    else : zz[cpt]=areaConsumption["Storage"]
    results=opt.solve(model)
    Constraints= getConstraintsDual_panda(model)
    TotalCols[cpt]=getVariables_panda(model)['energyCosts'].sum()[1]
    Prix=Constraints["energyCtr"].assign(Prix=lambda x: x.energyCtr *10**6).Prix.to_numpy()
    valueAtZero=Prix*(TotalCols[cpt]/sum(Prix*Prix)-zz[cpt])
    tmpCost=GenCostFunctionFromMarketPrices_dict(Prix,r_in=0.95,r_out=0.95,valueAtZero=valueAtZero)
    if (cpt==0): CostFunction[cpt]=GenCostFunctionFromMarketPrices(Prix,r_in=0.95,r_out=0.95,valueAtZero=valueAtZero)
    else:
        tmpCost = GenCostFunctionFromMarketPrices_dict(Prix, r_in=0.95, r_out=0.95, valueAtZero=valueAtZero)
        tmpCost2 = CostFunction[cpt-1]
        tmpCost2.Maxf_2Breaks_withO(tmpCost['S1'],tmpCost['S2'],tmpCost['B1'],tmpCost['B2'],tmpCost['f0']) ### etape clé, il faut bien comprendre ici l'utilisation du max de deux fonction de coût
        CostFunction[cpt]=tmpCost2
    LMultipliers[cpt]=Prix
    if cpt>0:
        DeltaPrix[cpt]=sum(abs(LMultipliers[cpt]-LMultipliers[cpt-1]))
        Deltazz[cpt]=sum(abs(zz[cpt]-zz[cpt-1]))
    areaConsumption["Storage"]=CostFunction[cpt].OptimMargInt([-p_max]*nbTime,[p_max]*nbTime,[0]*nbTime,[c_max]*nbTime)
    areaConsumption["NewConsumption"]=areaConsumption["areaConsumption"]+areaConsumption["Storage"]
    PrixTotal[cpt]=Prix.sum()
    Consommation[cpt]=areaConsumption.NewConsumption
    for i in model.areaConsumption:  model.areaConsumption[i]=areaConsumption.NewConsumption[i-1]

####PLANING
############################
#### Single Node Planing #
############################

Zones="FR"
year=2013

Selected_TECHNOLOGIES={'Thermal', 'WindOnShore', 'Solar'}

#### reading CSV files
areaConsumption = pd.read_csv('CSV/input/areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0)
availabilityFactor = pd.read_csv('CSV/input/availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0)
TechParameters = pd.read_csv('Data/input/Planing-SimpleCas3_TECHNOLOGIES.csv', sep=';', decimal=',', skiprows=0, comment="#")

#### Selection of subset
availabilityFactor=availabilityFactor[ availabilityFactor.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]
TechParameters=TechParameters[TechParameters.TECHNOLOGIES.isin(Selected_TECHNOLOGIES)]

availabilityFactor.head()

model = GetElectricSystemModel_PlaningSingleNode(areaConsumption,availabilityFactor,TechParameters)
opt = SolverFactory('mosek')
results=opt.solve(model)
Variables=getVariables_panda(model)
Variables.keys()
Constraints= getConstraintsDual_panda(model)
Constraints.keys()
