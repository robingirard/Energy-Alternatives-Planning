import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma
from time import time
from copy import deepcopy
from EnergyAlternativesPlanning.f_graphicalTools import *

Data_folder = "Models/Prospective_conso/data/"
base_dpe_residentiel_df=pd.read_csv(Data_folder+'base_logement_agregee.csv', sep=';', decimal='.')
base_dpe_residentiel_df.IPONDL.sum()
base_dpe_tertiaire =pd.read_csv(Data_folder+'Parc_tertiaire.csv', sep=',', decimal='.').set_index()