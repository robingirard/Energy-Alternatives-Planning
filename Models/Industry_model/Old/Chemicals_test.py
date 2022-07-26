from pyomo.environ import *
from pyomo.opt import SolverFactory



model             = ConcreteModel()
model.NaphtaSC=Var(domain=NonNegativeReals)
model.EthaneSC=Var(domain=NonNegativeReals)
model.PDH=Var(domain=NonNegativeReals)
model.CPP=Var(domain=NonNegativeReals)
model.to_zero =Var(domain=NonNegativeReals)
model.Objective   = Objective(expr = model.NaphtaSC+model.CPP+model.PDH +model.EthaneSC, sense=minimize)
model.Constraint1 = Constraint(expr = 0.35*model.NaphtaSC + 0.66*model.EthaneSC+0.17*model.CPP == 2173)
model.Constraint2 = Constraint(expr = 0.15*model.NaphtaSC + 0.17*model.EthaneSC+0.93*model.PDH+0.25*model.CPP == 1341)
model.Constraint3 = Constraint(expr = 0.1*model.NaphtaSC + 0.08*model.EthaneSC+0.04*model.PDH+0.13*model.CPP <= 892)
model.Constraint5 = Constraint(expr=model.NaphtaSC*0.35==0.55*2173)

# ... more constraints

opt = SolverFactory('mosek')

results = opt.solve(model)

print("Print values for all variables")
for v in model.component_data_objects(Var):
  print(v, v.value)


