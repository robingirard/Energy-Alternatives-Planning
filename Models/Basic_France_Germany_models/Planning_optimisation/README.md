
# Optimisation of Planning - multiple nodes
This folder contains 

 - texts with maths and introduction to the pyomo package to do the math in python (in [Math_and_pyomo_background_for_operation.ipynb](./Math_and_pyomo_background_for_operation.ipynb))
 - operation optimisation models for one node defined in [f_operationModels.py](./f_operationModels.py)
   - without storage in function GetElectricSystemModel_GestionSingleNode 
   - with storage in function GetElectricSystemModel_GestionSingleNode_withStorage 
 - case studies that apply these models:
   - several cases to learn step by step in [case_step_by_step_learning.ipynb](./case_step_by_step_learning.ipynb) and [case_step_by_step_learning.py](./case_step_by_step_learning.py)
   - a more detailed case of France in [case_simple_France.py](./case_simple_France.py) still simple because only one node !
 
If you are interested in **Planning** (optimisation of investment with operation constraints), you should go to Folder [Planning_optimisation](/README.md) in the same models folder. 
If you are interested in **multi node** optimisation (of Planning and operation), you should go to models [Basic_France_Germany_models](./../Basic_France_Germany_models/README.md). 

