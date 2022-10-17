# Basic France models

This model folder gives simple one node models for operation and planing with application to the french case. 
A very similar set of models exists for the multinode : [Basic_France_Germany_models](../Basic_France_Germany_models/README.md)

It contains 5 folders 
 - **Consumption**, with consumption time series and tools to manipulate this consumption (sector decomposition models, Electric heating models, Electric vehicles models). To see what is possible, you can check the [webpage generated from notebook](https://robingirard.github.io/Energy-Alternatives-Planing/Models/Basic_France_models/Consumption/Consumption_TS_manipulation_examples.html)
 - **Economic_And_Tech_assumptions** that allows to compute input economic and technic files for planing and operation optimisation (e.g. with annualized cost). Also contains explanation of these computations. 
 - **Production** contains normalized production data (also called "availability") and data analysis. To see what is possible, you can check the [webpage generated from notebook](https://robingirard.github.io/Energy-Alternatives-Planing/Models/Basic_France_models/Production/Production_visualisation_and_analysis.html)
 - **[Operation_optimisation](Operation_optimisation/README.md)** contains models, case studies and texts (with maths and introduction to pyomo) to learn step by step the optimisation of operation (in a case with one node). You can directly reach the demo web page [here](https://robingirard.github.io/Energy-Alternatives-Planing/Models/Basic_France_models/Operation_optimisation/case_operation_step_by_step_learning.html)). 
 - **[Planing_optimisation](Planing_optimisation/README.md)** contains models, case studies and texts (with maths and introduction to pyomo) to learn step by step the optimisation of planing (in a case with one node). You can directly reach the demo web page [here](https://robingirard.github.io/Energy-Alternatives-Planing/Models/Basic_France_models/Planing_optimisation/case_planing_step_by_step_learning.html). 

If you want to dig further you can go in other model folders. Feel free to add case studies for operation and planing. 
