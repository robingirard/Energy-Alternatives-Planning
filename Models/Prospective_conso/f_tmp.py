import pandas as pd
def Create_simplified_variables(df):
    df["construction_year_class_simple"] = df["construction_year_class"]
    df.loc[df["construction_year_class_simple"].isin(['[1919, 1945]','[1000, 1918]']),"construction_year_class_simple"]='avant 1945'
    df.loc[df["construction_year_class_simple"].isin(['[1946, 1970]','[1991, 2005]','[1971, 1990]']), 'construction_year_class_simple']='[1946, 1990]'
    df.loc[df["construction_year_class_simple"].isin(['[2006, 2012]','[2013, 2100]']),"construction_year_class_simple"]='après 2006'

    df["living_area_class_simple"] = df["living_area_class"]
    df.loc[df["living_area_class_simple"].isin(['De 40 à 60 m²', 'De 30 à 40 m²', 'Moins de 30 m²']),"living_area_class_simple"]='moins de 60m2'
    df.loc[df["living_area_class_simple"].isin(['De 60 à 80 m²','De 80 à 100 m²']),"living_area_class_simple"]='De 60 à 100 m²'
    df.loc[df["living_area_class_simple"].isin(['De 100 à 120 m²', '120 m² ou plus']),"living_area_class_simple"]='100 m² ou plus'

    df["occupancy_status_simple"] = df["occupancy_status"]
    df.loc[df["occupancy_status_simple"].isin(['owner','free accomodation']),"occupancy_status_simple"]='Propriétaire'
    df.loc[df["occupancy_status_simple"].isin(['renter', 'low rent housing']),"occupancy_status_simple"]='Locataire'


    df["living_area_class_simple"] = df["living_area_class"]
    df.loc[df["living_area_class_simple"].isin(['De 40 à 60 m²', 'De 30 à 40 m²', 'Moins de 30 m²']),"living_area_class_simple"]='moins de 60m2'
    df.loc[df["living_area_class_simple"].isin(['De 60 à 80 m²','De 80 à 100 m²']),"living_area_class_simple"]='De 60 à 100 m²'
    df.loc[df["living_area_class_simple"].isin(['De 100 à 120 m²', '120 m² ou plus']),"living_area_class_simple"]='100 m² ou plus'

    df["heating_system_simple"] = df["heating_system"]
    df.loc[df["heating_system_simple"].isin(['Autres','Chauffage urbain']),"heating_system_simple"]='Autres et Chauffage urbain'
    df.loc[df["heating_system_simple"].isin(['Chaudière fioul','Chaudière - autres']),"heating_system_simple"]='Chaudière fioul-autre'

    return df
