import numpy as np
import pandas as pd
import csv
import os
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go

#revenue without efficiency
C=0
for i in range(len(Prices)):
    C=C+res[i]*Prices[i]
print(C)

#revenue with efficiency (=0.8 here)
C=0
for i in range(len(Prices)):
    if res[i]>0:
        C=C+res[i]*Prices[i]/0.8
    elif res[i]<0:
        C=C+res[i]*Prices[i]*0.8
print(C)

#courbes
period=100
plt.plot(res[:100])
plt.plot(-(Prices[:100]-Prices.mean())/Prices.max())
plt.ylabel("Puissance (MW)")
plt.xlabel("Index")
plt.show()

energie=np.cumsum(res)
plt.plot(energie[:100], color='g')
plt.plot([0]*100, color='b')
plt.plot([c_max]*100, color='b')
plt.ylabel("Energie (MWh)")
plt.xlabel("Index")
plt.show()