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
* [4. Pycharm tips](#pycharm)
* [5.Getting help](#GettingH)
* [6.Getting involved](#GettingI)


## 1 - Installations  <a class="anchor" id="installations"></a>

Once you have downloaded the Energy-Alternatives-Planing folder, in the terminal, you need to create the conda environment thanks to the conda.yml file:

    conda env create conda.yml
    conda activate energyalternatives


## 2- Models Folder <a class="anchor" id="CasDEtude"></a>
Contains folders with Models. Each folder in Models Folder contains a set of Models and associated data and case studies. 
See the corresponding [README](Models/README.md). You can add your own Models folder to contribute, or by creating a case study in an existing Models Folder.

If you want to learn how to use the energy system planing tool, you can jump directly into the one node [Basic_France_models](Models/Basic_France_models/README.md) or the multinode [Basic_France_Germany_models](Models/Basic_France_Germany_models/README.md). 
There, you will find tutorials on how to use the simulation/modelisation tool, how to use pyomo and you will find a mathematical description of the models.

If you want to learn how to use our simple consumption prospective models for France, you can jump directly in [Prospective_conso](Models/Prospective_conso/README.md). 

## 3- functions folder <a class="anchor" id="functions"></a>
Contains:  
 - [tools](functions/f_tools.py) that can be used to facilitate the interface between pyomo optimisation models results and parameters and panda. 
 - a set of predefined [planing](functions/f_model_planing_constraints.py) and [operation](functions/f_model_operation_constraints.py) constraints to model the electric system, 
 - demand modeling tools in ([f_consumptionModels.py](functions/f_consumptionModels.py)) 
 - [graphical tools](functions/f_graphicalTools.py). 
 - tools that are under developpement for a more "automatic" construction of models : [f_model_cost_functions.py](functions/f_model_cost_functions.py) and [f_model_definition.py](functions/f_model_definition.py)

## 4 Pycharm tips  <a class="anchor" id="pycharm"></a>
If you're using PyCharm you should fix the environement in settings by choosing the right "python interpreter"

I strongly recommend to use the keyboard shortcut "crtl+enter" for action "Execute selection". This can be set in PyCharm Settings -> keymap
This project also contains Jupyter Notebook. 

## 5 Getting help <a class="anchor" id="GettingH"></a>

If you have questions, concerns, bug reports, etc, please file an issue in this repository's Issue Tracker.

## 6 Getting involved <a class="anchor" id="GettingI"></a>

BuildingModel is looking for users to provide feedback and bug reports on the initial set of functionalities as well as
developers to contribute to the next versions, with a focus on validation of models, cooling need simulation,
adaptation to other countries' datasets and building usages.

Instructions on how to contribute are available at [CONTRIBUTING](CONTRIBUTING.md).


## Open source licensing info
1. [LICENSE](LICENSE)

----

## Credits and references
Energy-Alternatives-Planing models are directly derived from work performes for several courses given at MINES ParisTech by Robin Girard, and by students. 

### Main contributors : 
- [Robin Girard](https://www.minesparis.psl.eu/Services/Annuaire/robin-girard) -- [blog](https://www.energy-alternatives.eu/) [gitHub](https://github.com/robingirard) [LinkedIn](https://www.linkedin.com/in/robin-girard-a88baa4/) [google Scholar](https://scholar.google.fr/citations?user=cEYGStIAAAAJ&hl=fr)
- Quentin Raillard Cazanove
- Pierrick Dartois
- Anaelle Jodry

