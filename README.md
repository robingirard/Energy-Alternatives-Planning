# Energy-Alternatives-Planing

This project contains code and data to model the electric system.
It relies mainly on a combined use of two well known python packages (panda and pyomo)
for linear programming adapted for the specific use of energy system modeling.

The installation relies on the use of a conda environment. Instruction is below
#TODO : add links to illustrative exemples here. 

### Table of content

* [1. Installation](#installations)
* [2. Models Folder](#CasDEtude)
* [3. Functions Folder](#functions)
* [4. Pycharm tips](#pycharm)
* [5. Getting help](#GettingH)
* [6. Getting involved](#GettingI)


## 1 - Installations  <a class="anchor" id="installations"></a>

Once you have downloaded the Energy-Alternatives-Planing folder, in the terminal, you need to create the conda environment thanks to the conda.yml file:

    conda env create --file conda.yml
    conda activate energyalternatives


## 2- Models Folder <a class="anchor" id="CasDEtude"></a>
Contains folders with Models. Each folder in Models Folder contains a set of Models and associated data and case studies. 
See the corresponding [README](Models/README.md). You can add your own Models folder to contribute, or by creating a case study in an existing Models Folder.

If you want to learn how to use the energy system planing tool, you can jump directly into the one node [operation example](https://robingirard.github.io/Energy-Alternatives-Planing/Models/Basic_France_models/Operation_optimisation/case_operation_step_by_step_learning.html) or mode generally into the one node model [Basic_France_models](Models/Basic_France_models/README.md) or the multinode [Basic_France_Germany_models](Models/Basic_France_Germany_models/README.md). 
There, you will find tutorials on how to use the simulation/modelisation tool, how to use pyomo and you will find a mathematical description of the models.

If you want to learn how to use our simple consumption prospective models for France, you can jump directly in [Prospective_conso](Models/Prospective_conso/README.md). 

If you want to use a more advanced case with 7 nodes in Europe to model the future european electric system, you can jump directly in [Seven_node_Europe](Models/Seven_node_Europe/README.md). 

list of available web page with examples/documentation :
[one node operation optim](https://robingirard.github.io/Energy-Alternatives-Planing/Models/Basic_France_models/Operation_optimisation/case_operation_step_by_step_learning.html) 
[one node planification optim](https://robingirard.github.io/Energy-Alternatives-Planing/Models/Basic_France_models/Planing_optimisation/case_planing_step_by_step_learning.html)
[two node operation optim]
[two node planification optim]
[7 node EU model Vs historical data](https://robingirard.github.io/Energy-Alternatives-Planing/Models/Seven_node_Europe/Simulation_analysis_Historical.html)
[7 node EU model for 2030](https://robingirard.github.io/Energy-Alternatives-Planing/Models/Seven_node_Europe/Simulation_analysis_2030.html)
[time series of consumption model](https://robingirard.github.io/Energy-Alternatives-Planing/Models/Basic_France_models/Consumption/Consumption_TS_manipulation_examples.html)
[time series of availability/production]((https://robingirard.github.io/Energy-Alternatives-Planing/Models/Basic_France_models/Production/Production_visualisation_and_analysis.html))
[residential heating consumption evolution](https://robingirard.github.io/Energy-Alternatives-Planing/Models/Prospective_conso/Evolution_ResTer_Heating_2D.html)
[transport consumption evolution](https://robingirard.github.io/Energy-Alternatives-Planing/Models/Prospective_conso/Evolution_Transport_2D.html)
[industry consumption evolution](https://robingirard.github.io/Energy-Alternatives-Planing/Models/Prospective_conso/Evolution_industrie.html)



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
- Antoine Rogeau
- Quentin Raillard Cazanove
- Pierrick Dartois
- Anaelle Jodry

