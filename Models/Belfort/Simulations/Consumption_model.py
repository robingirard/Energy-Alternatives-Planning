from functions.f_consumptionModels import *
from functions.f_graphicalTools import *

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
                    Conso_projected_df, Conso_detailed_df = ProjectionConsoNTS(NTS_profil_df, Projections_df,year, reindus)
                    if bati_hyp=='ref':
                        Conso_TS_heat_df = ConsoHeat(Temp_df, Thermosensitivity_df,
                                         Energy_houses_df, Energy_apartments_df, Energy_offices_df, Part_PAC_RCU_df,
                                         year,'ref', T0)
                        Conso_TS_air_con_df = ConsoAirCon(Temp_df, Thermosensitivity_df, Energy_houses_df,
                                                  Energy_apartments_df, Energy_offices_df, year)
                    else:
                        Conso_TS_heat_df = ConsoHeat(Temp_df, Thermosensitivity_df,
                                             Energy_houses_SNBC_df, Energy_apartments_SNBC_df, Energy_offices_df, Part_PAC_RCU_df,
                                             year,'SNBC', T0)
                        Conso_TS_air_con_df = ConsoAirCon(Temp_df, Thermosensitivity_df, Energy_houses_SNBC_df,
                                                  Energy_apartments_SNBC_df, Energy_offices_df, year)

                    Conso_ECS_df = Conso_ECS(Temp_df, Profil_ECS_df, Projections_ECS_df,year)
                    if ev_hyp=='':
                        Conso_VE_df,E_H2=ConsoVE(Temp_df,N_VP_df,N_VUL_df,N_PL_df,N_bus_df,N_car_df,
                                     Profil_VE_df,Params_VE_df,year)
                    else:
                        Conso_VE_df, E_H2 = ConsoVE(Temp_df, N_VP_fit55_df, N_VUL_fit55_df, N_PL_df, N_bus_df, N_car_df,
                                                    Profil_VE_df, Params_VE_df, year)
                    E_H2 += ConsoH2(Conso_H2_df,year, reindus)
                    Conso_projected_df["Consommation hors metallurgie"]+=Conso_TS_heat_df["Conso_TS_heat"]\
                        +Conso_TS_air_con_df["Conso_TS_air_con"]+Conso_ECS_df["Conso_ECS"]

                    Conso_projected_df["Conso_VE"]=Conso_VE_df["Conso_VE"]
                    Conso_projected_df["Conso_H2"]=E_H2/8760
                    Conso_projected_df["Taux_pertes"]=Losses_df["Taux_pertes"]

                    Conso_detailed_df["Conso_TS_heat"]=Conso_TS_heat_df["Conso_TS_heat"]
                    Conso_detailed_df["Conso_TS_air_con"] = Conso_TS_air_con_df["Conso_TS_air_con"]
                    Conso_detailed_df["Conso_ECS"]=Conso_ECS_df["Conso_ECS"]
                    Conso_detailed_df["Conso_VE"] = Conso_VE_df["Conso_VE"]
                    Conso_detailed_df["Conso_H2"] = E_H2 / 8760
                    Conso_detailed_df["Taux_pertes"] = Losses_df["Taux_pertes"]


                    #if year==2050:
                        #fig = MyStackedPlotly(y_df=Conso_projected_df)
                        #plotly.offline.plot(fig, filename=InputFolder+'Loads/Conso_plot_2050_'+reindus+'_'+bati_hyp+'.html')
                    Conso_detailed_df.to_csv(InputFolder+"Loads/Conso_detailed_"+str(year)+"_"+reindus+"_"+bati_hyp+ev_hyp+".csv", sep=";", decimal=".")

                    Conso_projected_df.to_csv(InputFolder+"Loads/Conso_"+str(year)+"_"+reindus+"_"+bati_hyp+ev_hyp+".csv", sep=";", decimal=".")
                    Conso_projected_df["Conso_Total"] =(1+Conso_projected_df["Taux_pertes"])*(Conso_projected_df["Consommation hors metallurgie"]+Conso_projected_df["Metallurgie"]+Conso_projected_df["Conso_VE"]+Conso_projected_df["Conso_H2"]/eta_electrolysis)
                    print(bati_hyp+" "+reindus+" "+ev_hyp)
                    print("Energy consumption (TWh): {}".format(Conso_projected_df["Conso_Total"].sum()/1E6))
                    print("Peak demand (GW): {}".format(Conso_projected_df["Conso_Total"].max()/1E3))









