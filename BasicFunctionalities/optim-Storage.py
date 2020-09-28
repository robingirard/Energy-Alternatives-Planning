
#region imports
import os
InputFolder='Data/input/'
from dynprogstorage.wrappers import GenCostFunctionFromMarketPrices, pmax, pmin
from numpy import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#endregion

#region storage optimisation - perfect efficiency
### exemples simples d'utilisation de l'outil de programmation dynamique
###### storage operation example
nbTime=250
Prices=np.random.uniform(1, 1000, nbTime)
p_max=1.;  c_max=10.*p_max;

CostFunction=GenCostFunctionFromMarketPrices(Prices)
## x_i>0 : on stocke (on consomme du réseau)
## x_i<0 : on produit
### --> phi_i(x_i) est donc un coût Achat - vente que l'on veut minimiser
res=CostFunction.OptimMargInt([-p_max]*nbTime,[p_max]*nbTime,[0]*nbTime,[c_max]*nbTime)## min sum_i phi_i(x_i)
## -p_max <= x_i <=  p_max forall i
## 0 <= sum_j=1^ix_j <= C_max  forall i
plt.plot(res)
plt.show()
#endregion

#region storage operation example with efficiency
nbTime=2500
Prices=np.random.uniform(1, 1000, nbTime)
p_max=1.;  c_max=10.*p_max;
r_in = 0.95; ## rendement d'entrée
r_out=0.95 ## rendement de sortie
E_0 = 0. ## Energie initialement dans le stockage
### ici x_i est l'électricité qui rentre (resp. qui sort) du stockage "après" (resp. "avant") rendement.
## x_i>0 : on stocke (on consomme du réseau) donc on doit acheter x_i /r_in
## x_i<0 : on produit donc on peut vendre x_i *r_out
### phi_i : x_i -->   Prices[i]*x_i *r_out* (x_i<0) +  Prices[i]*x_i/r_in * (x_i>0)
# phi_i est convexe

CostFunction=GenCostFunctionFromMarketPrices(Prices,r_in=r_in,r_out=r_out)
res=CostFunction.OptimMargInt([-p_max/r_out]*nbTime,[p_max*r_in]*nbTime,[E_0]*nbTime,[c_max-E_0]*nbTime)
## p_max est la puissance maximal consommée ou injectée sur le réseau par le stockage

plt.plot(res) ### positive : energy consumed from the network. Negative : energy delivered to the network
plt.show()
#endregion

#region With real market data
Prices_df=pd.read_csv(InputFolder+'EuropeanMarket_Prices_UTC_2007_2017.csv',sep=',',decimal='.',skiprows=0)
year = 2012

PricesYear_df=Prices_df[pd.to_datetime(Prices_df['Dates']).dt.year==year]
PricesYear_df=PricesYear_df.reset_index()
nbTime=PricesYear_df.__len__()
if sum(PricesYear_df['Prices']<=0)>0 : PricesYear_df[PricesYear_df['Prices']<=0]=0.000001
p_max=1.;  c_max=10.*p_max;
r_in = 0.95; ## rendement d'entrée
r_out=0.95 ## rendement de sortie
CostFunction=GenCostFunctionFromMarketPrices(PricesYear_df.Prices.to_list(),r_in=r_in,r_out=r_out)
res=CostFunction.OptimMargInt([-p_max/r_out]*nbTime,[p_max*r_in]*nbTime,[0]*nbTime,[c_max]*nbTime)
Operation=pd.DataFrame(res, columns=["Operation"])
PricesYear_df=pd.concat([PricesYear_df,Operation],axis=1)
PricesYear_df['Revenu']= -PricesYear_df['Prices']*PricesYear_df['Operation']
PricesYear_df['Revenu'].sum() ## Revenu €/an pour 10 x 1MWh de batterie -- Coût 100 €/kWh 100 000 €/MWh
#Evolution entre 2007 et 2016 ?
#endregion

#region Annexes -- reflexion sur le cas de 2 stockage -- a intégrer au reste progressivement

##### idée à développer (qui sera utile pour le cas de deux stockages) :
    ## on doit pouvoir optimiser comme si il n'y avait pas de rendement
    ## puis il faut faire une fonction qui rejete les échanges "non rentables" avec rendement
    ## c'est à dire les block achat/vente avec Prix[achat]/r_in>Prix[vente]*r_out
    ## cela demaderait de contruire (en c++) une map() f qui a chaque pas de temps t associe la map g
    ## qui contient tous les block achetés présents dans le stockage par ordre décroissant de prix d'achat
    ## g(prix)= pair(Energie,t) où t est le temps d'achat
# ici il faudrait écrire en latex comme exercice :
# Exercice - 1  dessiner l'ensemble V(E,C) = { x1,x2 :
#                     -E <= x1 <= C-E
#                     -E <= x1+x2 <= C-E }
# Exercice - 2
# Montrer que V(E1,C1)+V(E2,C2)=V(E1+E1,C1+C2)
# Exercice - 3 (à vérifier j'ai fait çà un peu vite )
# dessiner l'ensemble {x1,x2 :
#                     -E <= x1*(x1<0)*r_out + x1*(x1>0)/r_in <= C-E
#                     -E <= x1*(x1<0)*r_out + x1*(x1>0)/r_in +  x2*(x2<0)*r_out + x2*(x2>0)/r_in <= C-E
## montrer que cet ensemble est V(Ebis,Cbis) avec Ebis et Cbis bien choisis
#Cbis-Ebis=(C-E0)*r_in
#Cbis=(C-E0)*r_in+E0

# note supplémentaire
# Si W(-P1,P2)={x : -P1 <= x<=P2} alors {x : -P <= x*(x<0)*r_out + x*(x>0)/r_in<= P}= W(-P/r_out,P*r_in)
## Application_1
nbTime=10
Prices=np.random.uniform(1, 1000, nbTime)
p_max=1.;  c_max=10.*p_max;
E_0 = 0. ; r_in = 0.95; r_out=0.95
E_0Bis=E_0/r_out
c_maxBis = (c_max-E_0)*r_in + E_0Bis
CostFunction1=GenCostFunctionFromMarketPrices(Prices,r_in=1.,r_out=1.)
CostFunction2=GenCostFunctionFromMarketPrices(Prices,r_in=0.95,r_out=0.95 )
res1=CostFunction1.OptimMargInt([-p_max]*nbTime,[p_max]*nbTime,[E_0Bis]*nbTime,[c_maxBis-E_0Bis]*nbTime)
res2=CostFunction2.OptimMargInt([-p_max/r_out]*nbTime,[p_max*r_in]*nbTime,[E_0]*nbTime,[c_max-E_0]*nbTime)
np.array(res1)
np.array(res2)
for i in range(len(res1)):
    if (res1[i]>0) : res1[i]=res1[i]*r_in
    else : res1[i]=res1[i]/r_out
max(abs(np.array(res1)-np.array(res2)))
### pas pareil, il faudra supprimer les échanges "non rentables"

##### two storages with perfect efficiency
## note :  si l'étape d'avant a marché il faudrait faire le cas de deux stockages "avec rendement"
##### peut-etre plus compliqué
r_in1 = 1.; r_in2 = 1.;  ## rendement d'entrée
r_out1=1.; r_out2=1. ## rendement de sortie

r_in=1. ; r_out=1.
p_max1=10.;  c_max1=10.*p_max1;
p_max2=1.;  c_max2=1000.*p_max2;
E_01=0; E_02 =0 ;
p_max=p_max1+p_max2; c_max=c_max1+c_max2; E_0=E_01+E_02;

nbTime=500
Prices=np.random.uniform(1, 1000, nbTime)
#on resoud d'abord un stockage "somme"
CostFunction=GenCostFunctionFromMarketPrices(Prices,r_in=r_in,r_out=r_out)
res=CostFunction.OptimMargInt([-p_max/r_out]*nbTime,[p_max*r_in]*nbTime,[E_0]*nbTime,[c_max-E_0]*nbTime)

### a partir de la somme z=x+y on déduit x
### il suffit de trouve une solution x faisable telle que
### (1) x satisfait les contraintes du premier stockage
### (2) y satisfait les contraintes du second stockage
### (3) x+y = z
### on peut trouver çà en prenant une fonction coût constante égale à zéro et en choisissant bien les contraintes
CostFunction0=GenCostFunctionFromMarketPrices([0]*nbTime,r_in=1.,r_out=1.)

P1plus=np.array([p_max1*r_in1]*nbTime); P1moins=np.array([-p_max1/r_out1]*nbTime);
P2plus=np.array([p_max2*r_in2]*nbTime); P2moins=np.array([-p_max2/r_out2]*nbTime);
C1plus=np.array([c_max1-E_01]*nbTime); C1moins=np.array([-E_01]*nbTime);
C2plus=np.array([c_max2-E_02]*nbTime); C2moins=np.array([-E_02]*nbTime);
zz=np.array(res)
intzz=zz.cumsum()
lbP = pmax(P1moins,zz-P2plus); ubP = pmin(P1plus,zz-P2moins)
lbC = pmax(C1moins,intzz-C2plus); ubC = pmin(C1plus,intzz-C2moins)
min(np.array(ubP)-np.array(lbP))
min(np.array(ubC)-np.array(lbC))

res1=CostFunction0.OptimMargInt(lbP,ubP,lbC,ubC)
res2=zz-np.array(res1)

#endregion

