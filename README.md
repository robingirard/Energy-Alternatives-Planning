# Etude_TP_CapaExpPlaning-Python

This project contains code and data to model the electric system. 
It relies mainly on a combined use of 
 - two well known python packages (panda and pyomo) 
for linear programming adapted for the specific use of electric system modeling. 
 - a package of my own [dynprogstorage](https://github.com/robingirard/dynprogstorage) for efficient storage optimisation (with ad-hoc ) that can be installed with pip

The installation relies on the use of a conda environment. Instruction is below

### Table of content

* [1. Installation](#installations)
* [2. CSV folder](#CSV)
* [3. Repertoire CasDEtude](#CasDEtude)
* [4. Repertoire TP](#TP)
* [5. Repertoire functions](#functions)

## 1 - Installations  <a class="anchor" id="1.introduction"></a>

    conda env create robin.girard/energyalternatives
    conda activate Etude_TP
    
Or just to update the environment 

    conda env update --file conda.yml 
    
If you're using PyCharm you should fix the environement in settings by choosing the right "python interpreter" 

I strongly recommend to use the keyboard shortcut "crtl+enter" for action "Execute selection". This can be set in PyCharm Settings -> keymap 

## 2 -  Data folder <a class="anchor" id="CSV"></a>
Contains CSV files for the project projet (Wind/PV power instantaneous load factor, 
availability time series of french nuke, consumption, 
economic data for operation and planing), 
These files are loaded in the different exemples given in the next section. 

New data and associated documentation are welcome here to allow modeling of more different asses. 

## 3- Repertoire BasicFunctionalities <a class="anchor" id="CasDEtude"></a>
Contains python files exposing basic functionalities. 
For each .py file there is an associated Jupyter Notebook that explains the content with text and math formulae
Hence, if you want to learn how to use the exposed tools, you can star having a look at how to 

 - [Model consumption](https://github.com/robingirard/Etude_TP_CapaExpPlaning-Python/blob/master/BasicFunctionalities/input-Consumption.ipynb) 
 - [optimisation of operation ](https://github.com/robingirard/Etude_TP_CapaExpPlaning-Python/blob/master/BasicFunctionalities/optim-Operation.ipynb)
 - [optimisation of storage operation](https://github.com/robingirard/Etude_TP_CapaExpPlaning-Python/blob/master/BasicFunctionalities/optim-Storage.ipynb)
 - [optimisation of planing](https://github.com/robingirard/Etude_TP_CapaExpPlaning-Python/blob/master/BasicFunctionalities/optim-Planing.ipynb)


## 4- Repertoire TP <a class="anchor" id="TP"></a>
In progress 

## 5- Repertoire functions <a class="anchor" id="functions"></a>
Contains used function for planing and operation of electric system, for demand modeling, together with graphical tools. n de la thermosensibilité
