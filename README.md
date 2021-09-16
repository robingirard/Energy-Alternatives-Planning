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
* [4. Repertoire functions](#functions)
* [5. Sujet d'analyse](#Analyse)



## 1 - Installations  <a class="anchor" id="installations"></a>

    conda env create robin.girard/energyalternatives
    conda activate energyalternatives

Or just to update the environment

    conda env update --file conda.yml

If you're using PyCharm you should fix the environement in settings by choosing the right "python interpreter"

I strongly recommend to use the keyboard shortcut "crtl+enter" for action "Execute selection". This can be set in PyCharm Settings -> keymap

## 2 -  Data folder <a class="anchor" id="CSV"></a>
Contains CSV files for the project projet (Wind/PV power instantaneous load factor,
availability time series of french nuke, consumption,
economic data for operation and planing),
These files are loaded in the different examples given in the next section (see in particular [Economic Assumptions](https://github.com/robingirard/Etude_TP_CapaExpPlaning-Python/blob/master/BasicFunctionalities/input-Economic.ipynb) in file input-Economic.ipynb
, [Availability/load factor Assumptions](https://github.com/robingirard/Etude_TP_CapaExpPlaning-Python/blob/master/BasicFunctionalities/input-Availability.ipynb) in file input-Availability.ipynb
[Model consumption](https://github.com/robingirard/Etude_TP_CapaExpPlaning-Python/blob/master/BasicFunctionalities/input-Consumption.ipynb)  in file input-Consumption.ipynb
).

New data and associated documentation are welcome here to allow modeling of more different asses.

## 3- Repertoire BasicFunctionalities <a class="anchor" id="CasDEtude"></a>
Contains python files exposing basic functionalities.
For each .py file there is an associated Jupyter Notebook that explains the content with text and math formulae
Hence, if you want to learn how to use the exposed tools, you can star having a look at how to

 - [Economic Assumptions](https://github.com/robingirard/Etude_TP_CapaExpPlaning-Python/blob/master/exportToHTML/input-Economic.html) in file input-Economic.ipynb
 - [Availability/load factor Assumptions](https://github.com/robingirard/Etude_TP_CapaExpPlaning-Python/blob/master/exportToHTML/input-Availability.html) in file input-Availability.ipynb
 - [Model consumption](https://github.com/robingirard/Etude_TP_CapaExpPlaning-Python/blob/master/exportToHTML/input-Consumption.html)  in file input-Consumption.ipynb
 - [optimisation of operation ](https://github.com/robingirard/Etude_TP_CapaExpPlaning-Python/blob/master/exportToHTML/optim-Operation.html) in file optim-Operation.ipynb
 - [optimisation of storage operation](https://github.com/robingirard/Etude_TP_CapaExpPlaning-Python/blob/master/exportToHTML/optim-Storage.html) in file optim-Storage.ipynb
 - [optimisation of planing](https://github.com/robingirard/Etude_TP_CapaExpPlaning-Python/blob/master/exportToHTML/optim-Planing.html) in file optim-Planing.ipynb
 - [Advanced optimisation of operation ](https://github.com/robingirard/Etude_TP_CapaExpPlaning-Python/blob/master/exportToHTML/optim-Operation-Advanced.html) in file optim-Operation-Advanced.ipynb
 - [Advanced optimisation of planing](https://github.com/robingirard/Etude_TP_CapaExpPlaning-Python/blob/master/exportToHTML/optim-Planing-Advanced.html) in file optim-Planing-Advanced.ipynb



## 4- Repertoire functions <a class="anchor" id="functions"></a>
Contains used function for planing and operation of electric system, for demand modeling, together with graphical tools. n de la thermosensibilité

## 5 -  Sujet d'analyse  <a class="anchor" id="Analyse"></a>
This folder contains subjects for analysis and a practical work

 - [Pracical work for the course](https://github.com/robingirard/Etude_TP_CapaExpPlaning-Python/blob/master/SujetsDAnalyses/TP_questions.ipynb)
 - [Sujet 1 Thermosensibilité]()
