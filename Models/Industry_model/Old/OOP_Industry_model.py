import pandas as pd
import numpy as np

class Technologies:
    def __init__(self):
        self.techologies=dict()
    def add_technology(self,name,input,output):
        self.techologies[name]=Technology(name,input,output)

class Technology:
    def __init__(self,name,input,output):
        self.name=name
        self.input=input #energy&feedstock required
        self.output=output #output given for the input
        self.CO2=None


class Resources:
    def __init__(self):
        self.resources=dict()
    def add_resource(self,name,calorific,emissions,costperunit):
        self.resources[name]=Resource(name,calorific,emissions,costperunit)
    def update_consumed(self,technology,production):
        input=technology.input
        output=technology.output
        coef=0
        for name in production.keys():
            if name in output.keys():
                coef=production[name]/output[name]
                break
        for name in production.keys():
            if name in self.resources.keys():
                self.resources[name].consumed-=output[name]*coef
        for name in input.keys():
            if name in self.resources.keys():
                self.resources[name].consumed += input[name] * coef
    def print(self,config):
        print("\n\n"+str(config)+"\n")
        for name in self.resources.keys():
            print(name+": "+str(self.resources[name].consumed*10**-6)+"TWh")
    def reset(self):
        for name in self.resources.keys():
            self.resources[name].consumed=0



class Resource:
    def __init__(self,name,calorific,emissions,costperunit):
        self.name=name
        self.calorific=calorific #in MWh/t
        self.emissions=emissions #emissions in kgCO2/MWh
        self.costperunit=costperunit
        self.consumed=0

class Products:
    def __init__(self):
        self.products=dict()
    def set_products(self,technologies,resources):
        tech_list=technologies.techologies
        for name in tech_list.keys():
           for product_name in tech_list[name].output.keys():
               if product_name not in resources.resources.keys():
                   self.products[product_name]=0


    def set_production(self,data):
        pass




R=Resources()

R.add_resource(name="electricity",calorific=None,emissions=100,costperunit=60)
R.add_resource(name="natural_gas",calorific=14.889,emissions=400,costperunit=50)
R.add_resource(name="biogas",calorific=14.889,emissions=150,costperunit=80)
R.add_resource(name="coal",calorific=7.5,emissions=650,costperunit=25)
R.add_resource(name="biomass",calorific=5.00,emissions=100,costperunit=60)
R.add_resource(name="oil",calorific=11.630,emissions=500,costperunit=37)
R.add_resource(name="hydrogen",calorific=33.313,emissions=125,costperunit=125)
#source for energy density: https://en.wikipedia.org/wiki/Energy_density





#NB: every unit is MWh except for some production for which the unit is specified in the name (e.g. "steel_t")
