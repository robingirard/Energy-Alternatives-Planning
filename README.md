# Energy-Alternatives-Planing

This project contains code and data to model the electric system.
It relies mainly on a combined use of two well known python packages (panda and pyomo)
for linear programming adapted for the specific use of electric system modeling.

The installation relies on the use of a conda environment. Instruction is below
#TODO : add links to illustrative exemples here. 

### Table of content

* [1. Installation](#installations)
* [2. Models Folder](#CasDEtude)
* [3. Repertoire functions](#functions)
* [4. other folder](#CSV)



## 1 - Installations  <a class="anchor" id="installations"></a>

Once you have downloaded the Energy-Alternatives-Planing folder, in the terminal, you need to create the conda environment thanks to the conda.yml file:

    conda env create conda.yml
    conda activate energyalternatives

If you're using PyCharm you should fix the environement in settings by choosing the right "python interpreter"

I strongly recommend to use the keyboard shortcut "crtl+enter" for action "Execute selection". This can be set in PyCharm Settings -> keymap
This project also contains Jupyter Notebook. 

## 2- Models Folder <a class="anchor" id="CasDEtude"></a>
Contains folders with Models. Each folder in Models Folder contains a set of Models and associated data and case studies. 
See the corresponding [README](Models/README.md). You can add your own Models folder to contribute, or by creating a case study in an existing Models Folder.

If you want to learn how to use the tool, you can jump directly into the one node [Basic_France_models](Models/Basic_France_models/README.md) or the multinode [Basic_France_Germany_models](Models/Basic_France_Germany_models/README.md). 
There you will find tutorials on how to use the simulation/modelisation tool, how to use pyomo and you will find a mathematical description of the models. 

## 3- functions folder <a class="anchor" id="functions"></a>
Contains:  
 - [tools](functions/f_tools.py) that can be used to facilitate the interface between pyomo optimisation models results and parameters and panda. 
 - a set of predefined [planing](functions/f_model_planing_constraints.py) and [operation](functions/f_model_operation_constraints.py) constraints to model the electric system, 
 - demand modeling tools in ([f_consumptionModels.py](functions/f_consumptionModels.py)) 
 - [graphical tools](functions/f_graphicalTools.py). 
 - tools that are under developpement for a more "automatic" construction of models : [f_model_cost_functions.py](functions/f_model_cost_functions.py) and [f_model_definition.py](functions/f_model_definition.py)


## 4 other folder  <a class="anchor" id="CSV"></a>
Other Folder (Data, SujetsDAnalyse)  should disappear soon. 