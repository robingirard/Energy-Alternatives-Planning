
# Optimisation of operation - multiple nodes
This folder contains 

 - texts with maths and introduction to the pyomo package to do the math in python (in [Math_and_pyomo_background_for_operation.ipynb](./Math_and_pyomo_background_for_operation.ipynb))
 - operation optimisation models for one node defined in [f_operationModels.py](./f_operationModels.py)
   - without storage in function GetElectricSystemModel_GestionMultipleNode 
   - with storage in function GetElectricSystemModel_GestionMultipleNode_withStorage 
 - case studies that apply these models:
   - several cases to learn step by step in [case_step_by_step_learning.ipynb](./case_operation_step_by_step_learning.ipynb) and [case_step_by_step_learning.py](./case_operation_step_by_step_learning.py)
   - a more detailed case of France and Germany should be created 
   

If you are interested in **planing** (optimisation of investment with operation constraints), you should go to Folder [Planing_optimisation](./../Planing_optimisation/README.md) in the same models folder. 
If you are interested in **simple node** optimisation (of planing and operation), you should go to models [Basic_France_models](./../Basic_Germany_models/README.md). 

