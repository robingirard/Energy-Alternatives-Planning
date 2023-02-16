
# Optimisation of Planning - One node
This folder contains 

 - texts with maths and introduction to the pyomo package to do the math in python (If you are just starting, you should try understand the maths and learn about how to use the pyomo code in the one node operation optimisation tutorial [here](https://robingirard.github.io/Energy-Alternatives-Planning/Models/Basic_France_models/Operation_optimisation/case_operation_step_by_step_learning.html))
 - Planning optimisation models for one node defined in [f_PlanningModels.py](./f_planningModels.py)
   - without storage in function GetElectricSystemModel_PlanningSingleNode 
   - with storage in function GetElectricSystemModel_PlanningSingleNode_withStorage 
 - case studies that apply these models:
   - several cases to learn step by step in [case_Planning_step_by_step_learning.ipynb](./case_planning_step_by_step_learning.ipynb). The associated web page is see [here](https://robingirard.github.io/Energy-Alternatives-Planning/Models/Basic_France_models/Planning_optimisation/case_Planning_step_by_step_learning.html)
 
If you are interested in **operation** (simpler than optimisation of investment with operation constraints), you should go to Folder [Operation_optimisation](./../Operation_optimisation/README.md) in the same models folder. 
If you are interested in **multi node** optimisation (of Planning and operation), you should start with the 2 nodes models [Basic_France_Germany_models](./../Basic_France_Germany_models/README.md) or you could go further with the more realistic European 7 nodes model [Seven_node_Europe](./../Seven_node_Europe/README.md). 

