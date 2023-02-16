import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import os
import sys
import pickle
#import seaborn as sns
#sns.set_theme(style="whitegrid")
import warnings
warnings.filterwarnings("ignore")
colors = ["#004776","#b8e1ff","#72c5fe","#2baaff","#f8b740","#005f9e","#000000",
          "#e7e7e7","#fef3e8","#e8f1ff","#ebf5ff","#c69131","#2087cb"]# Set your custom color palette
#customPalette = sns.set_palette(sns.color_palette(colors))

#Excel file import
xls_file=pd.ExcelFile("Models/ES_Electric_System_subjects/Single_node_input.xlsx")
#Launching simulation (can take up to 5mn depending on the machine)
if os.path.basename(os.getcwd()) == "ES_Electric_System_Subjects":
    os.chdir('../..')
from Models.ES_Electric_System_Subjects.Simulation import *
Variables=Simulation_singlenode(xls_file)
del xls_file
if "Models" in os.listdir():
    os.chdir('Models/ES_Electric_System_Subjects')