from OOP_Industry_model import *


def SteelProductionManagement(Config,Production,TechDistribution,ng2else={"biogas":0},coal2else={"biomass":0},H2Resource=False):
    """
    Config is the name of the simulation (does not affect results)
    Production is production of steel set in t
    TechDistribution is a dictionary setting the technology distribution coefficient for steel production
    ng2else is to choose natural gas resource consumption transfer (by default to biogas, set to 0%) coefficient
    coal2else is to choose coal resource consumption transfer (by default to biomass, set to 0%) coefficient
    H2Resource is to choose if we consider H2 as a resource, if not, we must simulate production out of other resources
    """
    P = Products()
    T = Technologies()


    T.add_technology(name="electrolyser", input={"electricity": 1}, output={"hydrogen": 0.8})
    T.add_technology(name="eaf", input={"electricity": 0.89, "natural_gas": 0.41},
                    output={"steel_t": 1})

    T.add_technology(name="hdri_eaf", input={"biomass":0.01,"electricity": 1.84, "hydrogen": 2.25, "natural_gas": 0.75,"coal":0.03,"oil":0.01},
                    output={"steel_t": 1})

    T.add_technology(name="ch4dri_eaf", input={"electricity": 1.4, "natural_gas": 2.72,"coal":0.03,"biomass":0.01,"oil":0.01}, output={"steel_t": 1})
    T.add_technology(name="bioch4dri_eaf", input={"electricity": 1.4, "natural_gas": 0.75,"coal":0.03,"biomass":0.01,"biogas":1.97,"oil":0.01}, output={"steel_t": 1})

    T.add_technology(name="coaldri_eaf", input={"electricity": 1.47, "natural_gas": 0.75,"coal":0.85,"biomass":0.01,"oil":0.01}, output={"steel_t": 1})
    T.add_technology(name="biocoaldri_eaf", input={"electricity": 1.47, "natural_gas": 0.75, "biomass": 0.86,"oil":0.01},
                     output={"steel_t": 1})


    T.add_technology(name="bf_bof", input={"electricity": 0.25, "natural_gas": 0.73, "coal": 3.66,"biomass":0.02,"oil":0.02},
                    output={"steel_t": 1})
    T.add_technology(name="hbf_bof", input={"electricity": 0.33, "natural_gas": 0.75, "coal": 3.04,"hydrogen":0.53,"oil":0.01},
                    output={"steel_t": 1}) #15% of coal replaced by hydrogen
    T.add_technology(name="ch4bf_bof", input={"electricity": 0.33, "natural_gas": 1.28, "coal": 3.04,"oil":0.01},
                    output={"steel_t": 1})#15% of coal replaced by hydrogen
    T.add_technology(name="biobf_bof", input={"electricity": 0.33, "natural_gas": 0.75, "coal": 1.67,"biomass":1.91,"oil":0.01},
                    output={"steel_t": 1})

    P.set_products(T, R)

    P.products["steel_t"] = Production

    for tech in TechDistribution.keys():
        R.update_consumed(T.techologies[tech],{"steel_t":Production*TechDistribution[tech]})


    #ng and coal adjustment
    val=0
    for name in ng2else.keys():
        val+=R.resources["natural_gas"].consumed*ng2else[name]
        R.resources[name].consumed+=val
    R.resources["natural_gas"].consumed-=val

    val = 0
    for name in coal2else.keys():
        val += R.resources["natural_gas"].consumed*coal2else[name]
        R.resources[name].consumed += val
    R.resources["natural_gas"].consumed -= val

    #hydrogen production
    if not H2Resource :
        if R.resources["hydrogen"].consumed>=0:
            R.update_consumed(T.techologies["electrolyser"],{"hydrogen":R.resources["hydrogen"].consumed})

    R.print(str(Config))

    R.reset()



SteelProductionManagement(Config='2015',Production=15e6,TechDistribution={"bf_bof":0.66,"eaf":0.34})

SteelProductionManagement(Config='2027',Production=13.5e6,TechDistribution={"ch4bf_bof":0.36,"eaf":0.41,"hdri_eaf":0.14})

SteelProductionManagement(Config='2030',Production=13.5e6,TechDistribution={"hbf_bof":0.36,"eaf":0.41,"hdri_eaf":0.14})

SteelProductionManagement(Config='2050_Standard',Production=14e6,TechDistribution={"hbf_bof":0.1,"eaf":0.5,"hdri_eaf":0.4})

SteelProductionManagement(Config='2050_HDRI',Production=14e6,TechDistribution={"eaf":0.4,"hdri_eaf":0.6})

SteelProductionManagement(Config='2050_Recycling',Production=14e6,TechDistribution={"eaf":0.75,"hdri_eaf":0.25})

SteelProductionManagement(Config='2050_Biomass',Production=14e6,TechDistribution={"biobf_bof":0.1,"eaf":0.5,"bioch4dri_eaf":0.2,"biocoaldri_eaf":0.2})