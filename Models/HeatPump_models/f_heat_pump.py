import math
import numpy as np
import pandas as pd
from numpy import linalg as LA

TemperatureValues = {
    "Fan coil unit": 35,
    "FloorHeating": 35,
    "RadiatorLT": 45,
    "RadiatorMT": 55,
    "RadiatorHT": 65,
    "radiateurTHT": 75}

# coefficients issus de la thèse d'Antoine https://pastel.archives-ouvertes.fr/tel-02969503/document
# equation (3.7) (3.8) p78
nominal_COP_curve_Coefficients = {
    "A/A HP": {"a1": 7.177, "b1": -0.156, "c1": 0.0005,"d1":-0.066,
           "a2": 4.27, "b2": -0.09, "c2": 0.0005},
    "A/W HP": {"a1": 8.507, "b1": -0.156, "c1": 0.0005,"d1":-0.066,
           "a2": 5.6, "b2": -0.09, "c2": 0.0005},
    "W/W HP": {"a1": 10.29, "b1": -0.21, "c1": 0.0012,"d1":0,
               "a2": 10.29, "b2": -0.21, "c2": 0.0012}
}



### COEFFICIENTS DE LA REGULATION DE LA TEMPERATURE DE L'EAU (LOI D'EAU)
def coeffs_T_fluid(T_base,Simulation_PAC_input_parameter):
    """
    :param T_base: 
    :param T_target: 
    :param Techno: 
    :return: 
    """
    global TemperatureValues
    T_target=Simulation_PAC_input_parameter["T_target"]
    res={}
    if Simulation_PAC_input_parameter["System"]=="A/A HP": #PAC AIR/AIR
        res["a"]=(35-T_target)/(T_base-T_target)
        res["b"]=35-res["a"]*T_base
    if ((Simulation_PAC_input_parameter["System"]=="A/W HP")|((Simulation_PAC_input_parameter["System"]=="chaudiere")&(Simulation_PAC_input_parameter["Technology"]=="condensation"))): #PAC AIR/EAU
        res["a"]=(TemperatureValues[Simulation_PAC_input_parameter["Emitters"]]-T_target)/(T_base-T_target)
        res["b"]=TemperatureValues[Simulation_PAC_input_parameter["Emitters"]]-res["a"]*T_base
    if Simulation_PAC_input_parameter["regulation"] == "N":
        res["b"]=res["a"] * T_base+res["b"]
        res["a"]=0
    return res;



def estim_COP(T_ext,T_fluid,type = "A/W HP"):
    """
    compute
    :param T_ext:
    :param T_fluid:
    :param type: "A/W HP" (for Air Water) "A/A HP" for Air air.
    :return:
    """
    global nominal_COP_curve_Coefficients
    Delta_T = T_fluid - T_ext
    a2 = nominal_COP_curve_Coefficients[type]["a2"]
    b2 = nominal_COP_curve_Coefficients[type]["b2"]
    c2 = nominal_COP_curve_Coefficients[type]["c2"]
    a1 = nominal_COP_curve_Coefficients[type]["a1"]
    b1 = nominal_COP_curve_Coefficients[type]["b1"]
    c1 = nominal_COP_curve_Coefficients[type]["c1"]
    d1 = nominal_COP_curve_Coefficients[type]["d1"]

    if T_ext <= -3:
        res = a2 + b2 * Delta_T + c2 * Delta_T**2
    elif T_ext >= 6:
        res =  a1 + b1 * Delta_T + c1 * Delta_T ** 2 + d1 * T_fluid
    else:
        res = (T_ext - 6) / (-9) * (a2 + b2 * Delta_T + c2 * Delta_T**2) + (T_ext + 3) / 9 * (
                          a1 + b1 * Delta_T + c1 * Delta_T ** 2 + d1 * T_fluid)
    return res

def compute_T_biv2(COP_biv,T_biv,a,b,Simulation_PAC_input_parameter):

    type = Simulation_PAC_input_parameter["System"]
    T_target = Simulation_PAC_input_parameter["T_target"]
    # DEFINITION DES COEFFICIENTS DE LA COURBE DE COP
    a1 = nominal_COP_curve_Coefficients[type]["a1"];
    a2 = nominal_COP_curve_Coefficients[type]["a2"]
    b1 = nominal_COP_curve_Coefficients[type]["b1"];
    b2 = nominal_COP_curve_Coefficients[type]["b2"]
    c1 = nominal_COP_curve_Coefficients[type]["c1"];
    c2 = nominal_COP_curve_Coefficients[type]["c2"]
    d1 = nominal_COP_curve_Coefficients[type]["d1"];

    if Simulation_PAC_input_parameter["Technology"] == "Inverter":
        Omega = Simulation_PAC_input_parameter["Power_ratio"] / Simulation_PAC_input_parameter["PLF_biv"]
    if Simulation_PAC_input_parameter["Technology"] == "Bi-compressor":
        Omega = Simulation_PAC_input_parameter["N_stages"]

    A1_degiv= float(Omega * COP_biv/ (T_target - T_biv) * T_target-(a2+b2 * b+c2 * b ** 2))
    A2_degiv=float(-Omega * COP_biv / (T_target - T_biv)+(b2+2 * c2 * b) * (1-a))
    A3_degiv=float(-c2 * (1-a) ** 2)

    # CALCUL DU DELTA ET DES RACINES
    delta_degiv=A2_degiv ** 2-4 * A1_degiv * A3_degiv
    if delta_degiv>=0:
        T_biv2_degiv=(-A2_degiv-math.sqrt(delta_degiv)) / (2 * A3_degiv)
    else: T_biv2_degiv = np.nan
    ## IDEM POUR LA DEUXIEME COURBES DE COP (AVEC DEGIVRAGE)
    A1_nodegiv=float(Omega* COP_biv / (T_target - T_biv) * T_target-(a1+(b1+d1) * b+c1 * b ** 2))
    A2_nodegiv=float(-Omega * COP_biv / (T_target - T_biv)+(b1+2 * c1 * b) * (1-a)-d1*a)
    A3_nodegiv=float(-c1 * (1-a) ** 2)

    delta_nodegiv=A2_nodegiv ** 2-4 * A1_nodegiv * A3_nodegiv
    if delta_nodegiv>=0:
        T_biv2_nodegiv=(-A2_nodegiv-math.sqrt(delta_nodegiv)) / (2 * A3_nodegiv)
    else: T_biv2_nodegiv = np.nan

    ## LA SOLUTION PEUT SE TROUVER DANS L'INTERVALE ENTRE LE DEGIVRAGE 100% ET PAS DE DEGIVRAGE
    ## ON RESOUD UNE EQUATION DU 3EME DEGRE
    N1=3 / 9; M1=1 / 9; N2=6 / 9; M2=-1 / 9

    alpha=a
    beta=b

    coeff1_1=N1 * (a1+ (b1 + d1) * beta+ c1 * beta** 2)
    coeff1_X=M1 * (a1+b1 * beta+c1 * beta** 2)-N1 * (b1 * (1-alpha)+2 * c1 * beta * (1-alpha) - d1 * alpha)
    coeff1_X2=-M1 * (b1 * (1-alpha)+2 * c1 * beta * (1-alpha))+N1 * c1 * (1-alpha) ** 2
    coeff1_X3=M1 * c1 * (1-alpha) ** 2

    coeff2_1=N2 * (a2+b2 * beta+c2 * beta ** 2)
    coeff2_X=M2 * (a2+b2 * beta+c2 * beta ** 2)-N2 * (b2 * (1-alpha)+2 * c2 * beta * (1-alpha))
    coeff2_X2=-M2 * (b2 * (1-alpha)+2 * c2 * beta * (1-alpha))+N2 * c2 * (1-alpha) ** 2
    coeff2_X3=M2 * c2 * (1-alpha) ** 2

    coeff_1=(coeff1_1+coeff2_1)
    coeff_X=(coeff1_X+coeff2_X)
    coeff_X2=(coeff1_X2+coeff2_X2)
    coeff_X3=(coeff1_X3+coeff2_X3)

    A1_partdegiv=-Omega * COP_biv / (T_target - T_biv) * T_target+coeff_1
    A2_partdegiv=Omega * COP_biv / (T_target - T_biv)+coeff_X
    A3_partdegiv=coeff_X2
    A4_partdegiv=coeff_X3

    ## SI LE COEFF DU CUBE EST NUL, ON RESOUD L'EQUATION DU 2ND DEGRE
    if A4_partdegiv == 0:
        delta_partdegiv = A2_partdegiv ** 2 - 4 * A1_partdegiv * A3_partdegiv
        if delta_partdegiv >= 0:
            T_biv2_partdegiv = (-A2_partdegiv - math.sqrt(delta_partdegiv)) / (2 * A3_partdegiv)
        else:
            T_biv2_partdegiv = np.nan
    else :
        ## ON RESOUD LEQUATION DU 3EME DEGRE A LAIDE DE LA MATRICE ET DES EIGENVALUES DE LA MATRICE
        ## IL Y A SUREMENT DES METHODES PLUS SIMPLES AVEC LES BONS PACKAGES
        m = np.matrix([[0, 0, float(-A1_partdegiv / A4_partdegiv)],
                       [1, 0, float(-A2_partdegiv / A4_partdegiv)],
                       [0, 1, float(-A3_partdegiv / A4_partdegiv)]])
        roots = LA.eigvals(m)
        if all(np.imag(roots) == 0):
            T_biv2_partdegiv=float(roots[roots > -3 & roots < 6])
        else: T_biv2_partdegiv=float(np.real(roots[np.imag(roots) == 0]))

    ## ON GARDE LA RACINE QUI SE TROUVE SUR LA BONNE PORTION DE LA COURBE (QUI CORRESPOND A CELLE SUR LAQUELLE ELLE EST CALCULEE)
    if ((not np.isnan(T_biv2_degiv))&(T_biv2_degiv < (-3))):
        T_biv2=T_biv2_degiv
    else :
        if ((not np.isnan(T_biv2_nodegiv))&(T_biv2_nodegiv > 6)):
            T_biv2=T_biv2_nodegiv
        else:
            T_biv2=T_biv2_partdegiv
    return T_biv2


def estim_T_biv(T_base,Simulation_PAC_input_parameter):
    global TemperatureValues
    if Simulation_PAC_input_parameter['Mode']=="Monovalent":
        return max(T_base, Simulation_PAC_input_parameter["Temperature_limit"])
    else: #bivalent
        COP_base = estim_COP(T_ext=T_base,
                             T_fluid=TemperatureValues[Simulation_PAC_input_parameter["Emitters"]],type=Simulation_PAC_input_parameter["System"])
        ab = coeffs_T_fluid(T_base,Simulation_PAC_input_parameter)
        T_target = Simulation_PAC_input_parameter["T_target"]
        T_dim = T_target+(T_base-T_target)*Simulation_PAC_input_parameter['Share_Power']
        return compute_T_biv2(COP_base,T_dim , ab["a"], ab["b"], Simulation_PAC_input_parameter)
        ## calcul à faire

def estim_SCOP(meteo_data_heating_period,Simulation_PAC_input_parameter):
    """

    :param meteo_data_heating_period:
    :param Simulation_PAC_input_parameter:
    :return:
    """
    global     nominal_COP_curve_Coefficients

    T_target = Simulation_PAC_input_parameter["T_target"]
    T_start = Simulation_PAC_input_parameter["T_start"]


    comp_params={}
    comp_params["T_base"] =  np.quantile(meteo_data_heating_period["temp"], q=5 / 365)
    comp_params["T_biv"] = estim_T_biv(T_base = comp_params["T_base"],
                                       Simulation_PAC_input_parameter=Simulation_PAC_input_parameter)
    comp_params["Besoin_chauff_biv"] = T_target - comp_params["T_biv"] # à la valeur de U près

    # SI IL Y A UNE REGULATION DE LA TEMPERATURE DE L'EAU, ON AJUSTE
    comp_params = {**comp_params,**coeffs_T_fluid(T_base = comp_params["T_base"], Simulation_PAC_input_parameter=Simulation_PAC_input_parameter)} ## calcul de a et b
    comp_params["T_fluid_biv"]  = comp_params["a"] * comp_params["T_biv"] + comp_params["b"]
    #comp_params["Delta_T_biv"] = comp_params["T_fluid_biv"] - comp_params["T_biv"] # pour calculer le COP à T_biv

    # ON CALCULE LE COP A LA TEMPERATURE DE BIVALENCE
    comp_params["COP_biv"] = estim_COP(T_ext=comp_params["T_biv"], T_fluid=comp_params["T_fluid_biv"],
                                       type=Simulation_PAC_input_parameter["System"])

    ## CALCUL DU DEUXIEME POINT DE BIVALENCE (PASSAGE DU FONCTIONNEMENT INVERTER A ON/OFF)
    ## RESOLUTION DE L'EQUATION DU 2ND DEGRE POUR TROUVER LES RACINES (SANS DEGIVRAGE)
    if Simulation_PAC_input_parameter['Technology'] == 'Inverter':
        comp_params["T_biv2"] = compute_T_biv2(COP_biv=comp_params["COP_biv"],
                                               T_biv=comp_params["T_biv"],
                                               a=comp_params["a"],
                                               b=comp_params["b"],
                                               Simulation_PAC_input_parameter=Simulation_PAC_input_parameter)
        Besoin_chauff_biv2 = T_target - comp_params["T_biv2"]
        comp_params["T_fluid_biv2"] = comp_params["a"] * comp_params["T_biv2"] + comp_params["b"]
        COP_biv2 = estim_COP(T_ext=comp_params["T_biv2"], T_fluid=comp_params["T_fluid_biv2"], type=Simulation_PAC_input_parameter["System"])
        # CE COP EST MODIFIE PAR LE REGIME DE CHARGE PARTIELLE (RATIO PLF APPLIQUE)
        PLR_T_biv2 = float(comp_params["COP_biv"] * Besoin_chauff_biv2 / (comp_params["Besoin_chauff_biv"] * COP_biv2))
        a_PLF = (Simulation_PAC_input_parameter["PLF_biv"] - 1) / (PLR_T_biv2 - 1)
        b_PLF = 1 - a_PLF
    else :
        a_PLF=0; b_PLF = 1;
        comp_params["T_biv2"]=comp_params["T_biv"]
        if Simulation_PAC_input_parameter['Technology'] == 'Bi-compressor':
            print('not implemented')
            #TODO implémenter le bi-compresseur

    meteo_data_heating_period= meteo_data_heating_period.\
        assign(T_fluid=lambda x: comp_params["a"]*x['temp']+comp_params["b"])
    meteo_data_heating_period["COP"] =meteo_data_heating_period.\
        apply(lambda x: estim_COP(x['temp'], x['T_fluid'], type=Simulation_PAC_input_parameter["System"]), axis=1)
    meteo_data_heating_period = meteo_data_heating_period.\
        assign(Besoin_chauff = lambda x: T_target - x['temp']).\
        assign(PLR_CR = lambda x: comp_params["COP_biv"] * x['Besoin_chauff']  / comp_params["Besoin_chauff_biv"] / x['COP'] ).\
        assign(PLF = lambda x: a_PLF * x['PLR_CR']  + b_PLF).\
        assign(PLR_ma = lambda x: Simulation_PAC_input_parameter["Power_ratio"] * x['PLR_CR'] / x['PLF']).\
        assign(Dp = lambda x:  x['PLR_ma']  / (1 + Simulation_PAC_input_parameter["Ce"] * (x['PLR_ma']-1)))
    meteo_data_heating_period.loc[meteo_data_heating_period['Dp']==0,'Dp']=0.001
    #P_calo : besoin calorifique du bâtiment, dépend de la diff de temp entre T_target et T_ext
    if Simulation_PAC_input_parameter["Mode"] == "Monovalent":
        meteo_data_heating_period["P_calo"]=meteo_data_heating_period.apply(lambda x: x['Besoin_chauff'] if x['temp']<Simulation_PAC_input_parameter["Temperature_limit"] else comp_params["Besoin_chauff_biv"] * x['COP']  / comp_params["COP_biv"]  if x['temp']<comp_params["T_biv"]  else x['Besoin_chauff']  , axis=1)
        meteo_data_heating_period["P_elec"]=meteo_data_heating_period.apply(lambda x: x['Besoin_chauff'] if x['temp'] < Simulation_PAC_input_parameter["Temperature_limit"] else comp_params["Besoin_chauff_biv"] / comp_params["COP_biv"]  if  x['temp']<comp_params["T_biv"] else x['Besoin_chauff'] / (x['COP'] * x['PLF']) if ((x['temp'] < comp_params["T_biv2"])&(x['temp'] > comp_params["T_biv"])) else x['Besoin_chauff'] / (x['COP'] * x['PLF'] *x['Dp']), axis=1)
        RE=0
        RP=np.zeros(len(meteo_data_heating_period))

    if Simulation_PAC_input_parameter["Mode"] == "Bivalent":
        meteo_data_heating_period["P_calo"] = meteo_data_heating_period.apply(lambda x: 0 if x['temp']<Simulation_PAC_input_parameter["Temperature_limit"] else comp_params["Besoin_chauff_biv"] * x['COP']  / comp_params["COP_biv"]  if x['temp']<comp_params["T_biv"] else x['Besoin_chauff']  , axis=1)
        meteo_data_heating_period["P_elec"] = meteo_data_heating_period.apply(lambda x: 0 if x['temp'] < Simulation_PAC_input_parameter["Temperature_limit"] else comp_params["Besoin_chauff_biv"] / comp_params["COP_biv"]  if  x['temp']<comp_params["T_biv"] else x['Besoin_chauff'] / (x['COP'] * x['PLF']) if ((x['temp'] < comp_params["T_biv2"])&(x['temp'] > comp_params["T_biv"])) else x['Besoin_chauff'] / (x['COP'] * x['PLF'] *x['Dp']), axis=1)
        meteo_data_heating_period["P_app"] = meteo_data_heating_period.apply(lambda x: x['Besoin_chauff'] if x['temp'] < Simulation_PAC_input_parameter["Temperature_limit"] else x['Besoin_chauff'] - comp_params["Besoin_chauff_biv"] * x['COP'] / comp_params["COP_biv"] if x['temp']<comp_params["T_biv"] else 0, axis=1)


        RE=meteo_data_heating_period["P_app"].sum() / (meteo_data_heating_period["P_calo"]+meteo_data_heating_period["P_app"]).sum()
        RP=meteo_data_heating_period["P_app"] / (meteo_data_heating_period["P_calo"]+meteo_data_heating_period["P_app"])

    ## ON STOCKE LES RESULTATS
    SCOP = meteo_data_heating_period["P_calo"].sum()/meteo_data_heating_period["P_elec"].sum()
    COP_data = meteo_data_heating_period["P_calo"] / meteo_data_heating_period["P_elec"]

    return {"SCOP":SCOP,"COP_data":COP_data,"T_biv":comp_params["T_biv"],
            "RP":RP,"RE":RE,"meteo_data_heating_period":meteo_data_heating_period}


def get_heating_period_metdata(meteo_data):
    year = meteo_data.index.year[0]
    meteo_data["day"] = meteo_data.index.day
    meteo_data["month"] = meteo_data.index.month
    period_chauff = meteo_data.groupby(["month", "day"]).temp.mean().to_frame().rename(columns = {"temp" : "T_mean"})

    ##ON CALCULE LA MOYENNE GLISSANTE SUR 3 JOURS POUR IDENFITIER UN JOUR DE DEMARRAGE
    rm_start = period_chauff.rolling(3).mean()
    ##ON CALCULE LA MOYENNE GLISSANTE SUR 7 JOURS POUR IDENFITIER UN JOUR D'ARRET
    rm_stop  = period_chauff.rolling(7).mean()

    ##ON SEPARE LES DONNEES EN DEUX POUR AVOIR UNE PERIODE DE CHAUFFE "CONTINUE"
    second_half = rm_start.iloc[(round(len(rm_start) / 2)):len(rm_start)]
    first_half = rm_stop.iloc[1:round((len(rm_stop) / 2))]

    ##ON IDENTIFIE LE MOMENT OU LE CHAUFFAGE EST DEMARRE --> MOYENNE DE TEMPERATURE SUR 3 JOURS INFERIEURE A 13°C (ARBITRAIRE)
    start_heating = second_half.loc[second_half.T_mean <= 13,:].index[0]
    ##ON IDENTIFIE LE MOMENT OU LE CHAUFFAGE EST ARRETE --> MOYENNE DE TEMPERATURE SUR 7 JOURS SUPERIEURE A 15°C (ARBITRAIRE)
    stop_heating = first_half.loc[first_half.T_mean >= 15,:].index[0]

    Periodes_chauffe=pd.DataFrame.from_dict(
        {"Method": ["Allyear", "Standard", "Adjusted"],
         "Start_month": [8, 10, start_heating[0]],
         "Start_day": [1, 1, start_heating[1]],
         "Stop_month": [7, 5, stop_heating[0]],
         "Stop_day": [31, 20, stop_heating[1]]}).set_index("Method")

    Periode_chauffe=Periodes_chauffe.loc["Adjusted",]
    start_date =pd.to_datetime( str(Periode_chauffe["Start_day"])+"/"+str(Periode_chauffe["Start_month"])+"/"+str(year))
    end_date =pd.to_datetime( str(Periode_chauffe["Stop_day"]) + "/" + str(Periode_chauffe["Stop_month"]) + "/" + str(year))

    meteo_data_heating_period=\
        pd.concat([ meteo_data.loc[pd.to_datetime("1/1/"+str(year)) : end_date,:],\
                    meteo_data.loc[start_date:pd.to_datetime("1/1/"+str(year+1)),:]])
    return meteo_data_heating_period

def estim_COPs(meteo_data,T_base,Heating_params,Systems):

    global TemperatureValues

    meteo_data_heating_period= get_heating_period_metdata(meteo_data)
    SCOPs = {}; T_fluid_biv = {}; T_biv = {}
    BaseCOP = {}; COPs_data = {}; RPs_data = {}
    for Systems_index, System in Systems.iterrows():
        if System["System"]=="A/W HP":
            T_biv[Systems_index] = max(T_base, System["Temperature_limit"])
            T_fluid_biv[Systems_index] = TemperatureValues[System["Emitters"]]
    ## LA TEMPERATURE DU FLUIDE EST 35°C POUR LA PAC AIR/AIR
            SCOP=estim_SCOP(meteo_data_heating_period,
                            T_base=T_base,T_biv = T_biv[Systems_index],
                            Techno = System,Heating_params=Heating_params,type="AA")
