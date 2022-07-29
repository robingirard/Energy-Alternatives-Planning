from functions.f_graphicalTools import *
from Models.Belfort.Simulations.f_consumptionBelfort import *

import os
if os.path.basename(os.getcwd())=="Simulations":
    os.chdir('..')
    os.chdir('..')
    os.chdir('..') ## to work at project root  like in any IDE

InputFolder='Models/Belfort/Conso/'

# Main scenario hypothesis
T0=15# Temperature when heating starts
DeltaT_warming_year=0.01# To simulate global warming
eta_electrolysis=0.7
T1=20# Temperature when air-condition starts

# Non thermosensitive profile
NTS_profil_df=pd.read_csv(InputFolder+'Conso_NTS_2019.csv',sep=';',decimal='.',parse_dates=['Date']).set_index(['Date'])
Thermosensitivity_df=pd.read_csv(InputFolder+'Thermosensitivity_2019.csv',sep=';',decimal='.').set_index(["Heure"])
Projections_df=pd.read_csv(InputFolder+'Projections_NTS.csv',sep=';',decimal=',').set_index(['Annee'])

# Heating
Energy_houses_df=pd.read_csv(InputFolder+'Bati/Energie_TWh_maisons_type_de_chauffage_ref.csv',sep=';',decimal='.').set_index("Année")
Energy_apartments_df=pd.read_csv(InputFolder+'Bati/Energie_TWh_appartements_type_de_chauffage_ref.csv',sep=';',decimal='.').set_index("Année")
Energy_offices_df=pd.read_csv(InputFolder+'Bati/Energie_tertiaire_type_de_chauffage.csv',sep=';',decimal='.').set_index("Année")
Part_PAC_RCU_df=pd.read_csv(InputFolder+'Bati/Part_PAC_reseaux_chaleur.csv',sep=';',decimal=',').set_index("Annee")

Energy_houses_SNBC_df=pd.read_csv(InputFolder+'Bati/Energie_TWh_maisons_type_de_chauffage_SNBC.csv',sep=';',decimal='.').set_index("Année")
Energy_apartments_SNBC_df=pd.read_csv(InputFolder+'Bati/Energie_TWh_appartements_type_de_chauffage_SNBC.csv',sep=';',decimal='.').set_index("Année")

d_Energy_houses_df={'ref':Energy_houses_df,'SNBC':Energy_houses_SNBC_df}
d_Energy_apartments_df={'ref':Energy_apartments_df,'SNBC':Energy_apartments_SNBC_df}

Temp_df=pd.read_csv(InputFolder+'Temp_FR_2017_2022.csv',sep=';',decimal='.',parse_dates=['Date']).set_index(["Date"])

index2019=(Temp_df.index.to_series().dt.minute==0)&(Temp_df.index.to_series().dt.year==2019)
Temp_2019_df=Temp_df[index2019].reset_index().set_index("Date").sort_index()
Temp_2019_df= CleanCETIndex(Temp_2019_df)# Traitement heure d'été et heure d'hiver

# ECS (hot water)
Profil_ECS_df=pd.read_csv(InputFolder+'Profil_ECS_futur.csv',sep=';',decimal=',').set_index(["Jour","Heure"])
Projections_ECS_df=pd.read_csv(InputFolder+'Projections_ECS.csv',sep=';',decimal=',').set_index(["Annee"])

# Electric vehicles
N_VP_df=pd.read_csv(InputFolder+'Vehicles/Motorisations_vp.csv',sep=';',decimal='.').set_index(["Année"])
N_VUL_df=pd.read_csv(InputFolder+'Vehicles/Motorisations_vul.csv',sep=';',decimal='.').set_index(["Année"])
N_PL_df=pd.read_csv(InputFolder+'Vehicles/Motorisations_pl.csv',sep=';',decimal='.').set_index(["Année"])
N_bus_df=pd.read_csv(InputFolder+'Vehicles/Motorisations_bus.csv',sep=';',decimal='.').set_index(["Année"])
N_car_df=pd.read_csv(InputFolder+'Vehicles/Motorisations_car.csv',sep=';',decimal='.').set_index(["Année"])
N_VP_fit55_df=pd.read_csv(InputFolder+'Vehicles/Motorisations_vp_fit55.csv',sep=';',decimal='.').set_index(["Année"])
N_VUL_fit55_df=pd.read_csv(InputFolder+'Vehicles/Motorisations_vul_fit55.csv',sep=';',decimal='.').set_index(["Année"])

d_N_VP_df={'':N_VP_df,'_fit55':N_VP_fit55_df}
d_N_VUL_df={'':N_VUL_df,'_fit55':N_VUL_fit55_df}

Profil_VE_df=pd.read_csv(InputFolder+'Vehicles/Profil_VE.csv',sep=';',decimal=',').set_index(["Jour","Heure"])
Params_VE_df=pd.read_csv(InputFolder+'Vehicles/Params_VE.csv',sep=';',decimal=',').set_index(["Vehicule"])

# H2 (not related to electric vehicles)
Conso_H2_df=pd.read_csv(InputFolder+'Conso_H2.csv',sep=';',decimal=',').set_index(["Annee"])

for year in [2030,2040,2050,2060]:
    Temp_df=Temp_2019_df.loc[:,["Temperature"]]+(year-2019)*DeltaT_warming_year
    Losses_df=Losses(Temp_df)
    print("\nModel consumption "+str(year))
    for bati_hyp in ['ref','SNBC']:
        for reindus in ['no_reindus','reindus','UNIDEN']:
            for ev_hyp in ['','_fit55']:
                if ev_hyp=='_fit55' and (bati_hyp!='ref' or reindus!='reindus'):
                    pass
                else:
                    Conso_projected_df,Conso_detailed_df=Project_consumption(NTS_profil_df, Projections_df,
                                        Temp_df, Thermosensitivity_df,
                                        d_Energy_houses_df[bati_hyp], d_Energy_apartments_df[bati_hyp], Energy_offices_df, Part_PAC_RCU_df,
                                        Profil_ECS_df, Projections_ECS_df,
                                        d_N_VP_df[ev_hyp], d_N_VUL_df[ev_hyp], N_PL_df, N_bus_df, N_car_df,
                                        Profil_VE_df, Params_VE_df,
                                        Conso_H2_df,
                                        Losses_df,
                                        year,
                                        bati_hyp, reindus, ev_hyp, T0, T1)

                    Conso_detailed_df.to_csv(InputFolder+"Loads/Conso_detailed_"+str(year)+"_"+reindus+"_"+bati_hyp+ev_hyp+".csv", sep=";", decimal=".")

                    Conso_projected_df.to_csv(InputFolder+"Loads/Conso_"+str(year)+"_"+reindus+"_"+bati_hyp+ev_hyp+".csv", sep=";", decimal=".")
                    Conso_projected_df["Conso_Total"] =(1+Conso_projected_df["Taux_pertes"])*(Conso_projected_df["Consommation hors metallurgie"]+Conso_projected_df["Metallurgie"]+Conso_projected_df["Conso_VE"]+Conso_projected_df["Conso_H2"]/eta_electrolysis)
                    print(bati_hyp+" "+reindus+" "+ev_hyp)
                    print("Energy consumption (TWh): {}".format(Conso_projected_df["Conso_Total"].sum()/1E6))
                    print("Peak demand (GW): {}".format(Conso_projected_df["Conso_Total"].max()/1E3))








