import pandas as pd

# Constitution des bases de données

Ages=pd.DataFrame.from_dict({  "vp":[2.01,2.97,0,10.42,10.4],
                                        "vul":[3.42,1.22,0,11,9.98],
                                        "bus":[5.06,14.04,0,7.31,8.39],
                                        "car":[2.74,0,0,1.39,7.63],
                                        "pl":[5.1,0,0,2.5,9.1]},
                                     orient="index",columns =["Electrique","Hybride rechargeable","Hydrogène","GNV","Autres fossiles"])
Ages.index=Ages.index.rename('type')
Ages_long=Ages.melt(ignore_index=False,
                var_name='Energie', value_name='Age'). \
    reset_index().set_index(["type", "Energie"])


Nombre=pd.DataFrame.from_dict({  "vp":[244863,157446,0,153769,37790188],
                                        "vul":[48659,747,0,15954,5839037],
                                        "bus":[869,72,0,3496,23022],
                                        "car":[69,0,0,584,65393],
                                        "pl":[139,0,0,6603,593541]},
                                     orient="index",columns =["Electrique","Hybride rechargeable","Hydrogène","GNV","Autres fossiles"])
Nombre.index=Nombre.index.rename('type')
Nombre_long=Nombre.melt(ignore_index=False,
                var_name='Energie', value_name='Nombre'). \
    reset_index().set_index(["type", "Energie"])


infos=pd.DataFrame.from_dict({  "vp":[20,2],
                                        "vul":[20,2],
                                        "bus":[15,2],
                                        "car":[15,2],
                                        "pl":[18,2]},
                                     orient="index",columns =["duree_vie","k"])
infos.index=infos.index.rename('type')

Transport = Nombre_long.merge(Ages_long,how = 'outer',left_index=True,right_index=True).\
    merge(infos,how = 'outer',left_index=True,right_index=True)
Transport["year"]=2021
Transport=Transport.reset_index().set_index(["type", "Energie" ,"year"])

def make_database(L_ages,L_parc,duree_vie):
    L_base=[]
    for i in range(n_motor):
        if L_parc[i]>0:
            d=2*duree_vie
            alpha=find_alpha(L_ages[i],duree_vie)
            L_distrib=[1/gamma(1+alpha*(i+1)) for i in range(d)]
            s=sum(L_distrib)
            for year in range(year_ref-d,year_ref):
                L_base.append([L_motorisation[i],year,round(L_distrib[year_ref-year-1]/s*L_parc[i])])
                    #b*q**(year_ref-year-1)*np.exp(-((year_ref-year-1)/lamb)**k))])
    return L_base

L_base_vp=make_database(L_vp_ages,L_vp_parc_2021,duree_vie_vp)


# Constitution des bases de données
L_motorisation=["Electrique","Hybride rechargeable","Hydrogène","GNV","Autres fossiles"]
d_motorisation={"Electrique":0,"Hybride rechargeable":1,"Hydrogène":2,"GNV":3,"Autres fossiles":4}
L_vp_ages=[2.01,2.97,0,10.42,10.4]
L_vul_ages=[3.42,1.22,0,11,9.98]
L_bus_ages=[5.06,14.04,0,7.31,8.39]
L_car_ages=[2.74,0,0,1.39,7.63]
L_pl_ages=[5.1,0,0,2.5,9.1]
L_vp_parc_2021=[244863,
157446,
0,
153769,
37790188]
L_vul_parc_2021=[48659,
747,
0,
15954,
5839037]
L_bus_parc_2021=[869,72,0,3496,23022]
L_car_parc_2021=[69,0,0,584,65393]
L_pl_parc_2021=[139,0,0,6603,593541]
duree_vie_vp=20
duree_vie_vul=20
duree_vie_bus=15
duree_vie_car=15
duree_vie_pl=18
k_vp=2
k_vul=2
k_bus=2
k_car=2
k_pl=2

year_ref=2021

n_motor=len(L_motorisation)
