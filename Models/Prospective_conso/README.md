# Prospective consumption model

This model folder gives tools to model the annual energy consumption per sector. 
The tool is generic and works with an excel file as input. Its root principle are as follows :
 - it takes as input a description of a set of devices (says heating systems in buildings, and efficiency of associated building) and their energy consumption per energy vector
 - it simulates the evolution along several years of this set according to 3 actions :
   - destruction
   - retofit
   - new devices

 
All the python code relies on the use of module panda in a vectorised way. 

Several cases studies are available : 
 - Evolution of **residential** building heating systems, with a set of french building described along **residential types** (house/appartment) and **heating system**. 
These 2 "keys" for describing the whole building stock makes this case "2D". The associated code is available at **[Evolution_Residentiel_2D](Evolution_Residentiel_2D.py)**
The code for generating the input data from a more general descripion of the french building stock is available in **[creation_data_residentiel](creation_data_residentiel.py)**
 - Evolution of **tertiary** building heating systems, with a set of french building described along **heating system**. 
This single "key" for describing the whole building stock makes this case "1D". The associated code is available at **[Evolution_Tertiaire_1D](Evolution_Tertiaire_1D.py)**, 
and a jupyter notebook is available in [Tertiary_building_stock_evolution](Tertiary_building_stock_evolution.ipynb). 
The code for generating the input data from the 3D descripion (see below) of the french tertiary building stock is available in **[creation_data_tertiaire](creation_data_tertiaire.py)**
 - Evolution of **tertiary** building heating systems, with a set of french building described along **categories**, **energy class** and **heating system**. 
These 3 "keys" for describing the whole building stock makes this case "3D". The associated code is available at **[Evolution_Tertiaire_3D](Evolution_Tertiaire_3D.py)**


If you want to dig further and contribute, you can : 
- try to use proposed cases to reproduce common studies (such as those performed in France by RTE, ADEME, DGEC, ...) 
- try to propose new cases for other usages, other country
- help improving the code (e.g. for now it is not possible to do the retrofit action on a "new" device, meaning a device that was created or retrofited during the simulation)