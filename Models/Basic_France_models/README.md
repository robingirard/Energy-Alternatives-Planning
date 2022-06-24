# Basic France models

This model folder gives simple one node models for operation and planing with application to the french case. 
A very similar set of models exists for the multinode : [Basic_France_Germany_models](../Basic_France_Germany_models/README.md)

It contains 5 folders 
 - **Consumption**, with consumption time series and tools to manipulate this consumption (sector decomposition models, Electric heating models, Electric vehicles models)
 - **Economic_And_Tech_assumptions** that allows to compute input economic and technic files for planing and operation optimisation (e.g. with annualized cost). Also contains explanation of these computations. 
 - **Production** contains normalized production data (also called "availability") and data analysis
 - **[Operation_optimisation](Operation_optimisation/README.md)** contains models, case studies and texts (with maths and introduction to pyomo) to learn step by step the optimisation of operation (in a case with one node)
 - **[Planing_optimisation](Planing_optimisation/README.md)** contains models, case studies and texts (with maths and introduction to pyomo) to learn step by step the optimisation of planing (in a case with one node)

If you want to dig further you can go in other model. Feel free to add case studies for operation and planing. 