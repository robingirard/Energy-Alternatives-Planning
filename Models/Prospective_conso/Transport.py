import pandas as pd

Niveau_description = "B"
tmp=pd.read_excel("Models/Prospective_conso/data/Transport.xlsx",
              sheet_name=["Transport_"+Niveau_description+"_base","Transport_"+Niveau_description+"_trainavion"])
Transport_base=tmp["Transport_"+Niveau_description+"_base"].set_index(["Véhicule"])
Transport_trainavion=tmp["Transport_"+Niveau_description+"_trainavion"].set_index(["Véhicule"])