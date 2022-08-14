import pandas as pd

from Industry_multisctor import *


def import_industry_data(L_sectors=["Cement"], L_countries=["FR","BE","CH","DE","ES","GB","IT"],L_scenarios=["trends","reindus"]):
    folder = "Data/"

    Resource_Technologies=pd.read_excel(folder+"Resources_Technologies_all.xlsx").fillna(0)
    Resource_Availability=pd.read_excel(folder+"Resources_availability.xlsx").fillna(0)
    L_years=list(Resource_Technologies['Year'].unique())
    d_Resource_Technologies = {}
    d_Resource_Availability={}
    for year in L_years:
        d_Resource_Technologies[year]=Resource_Technologies[Resource_Technologies.Year==year].drop("Year",axis=1)
        d_Resource_Availability_year={}
        for country in L_countries:
            d_Resource_Availability_year[country]=Resource_Availability[(Resource_Availability.Year==year)&(Resource_Availability.Country==country)].drop(["Year","Country","Unit"],axis=1)
        d_Resource_Availability[year]=d_Resource_Availability_year

    d0_Production_Technologies={}
    d0_Available_Technologies={}
    d0_Production={}
    for sector in L_sectors:
        d0_Production_Technologies[sector]=pd.read_excel(folder+sector+"/"+sector+"_technologies.xlsx").fillna(0).drop("unit",axis=1)
        d0_Production_Technologies[sector].set_index(["Year", "Resource"], inplace=True)

        d0_Available_Technologies[sector] = pd.read_excel(folder + sector + "/" + sector + "_available_technologies.xlsx")
        d0_Available_Technologies[sector].set_index(["Year", "Technologies"], inplace=True)
        d0_Available_Technologies[sector]["Min_prod_ratio"].fillna(0,inplace=True)
        d0_Available_Technologies[sector]["Max_prod_ratio"].fillna(1, inplace=True)

        d0_Production[sector] = pd.read_excel(folder + sector + "/" + sector + "_production.xlsx").fillna(0)
        d0_Production[sector].set_index(["Year", "Country","Resource"], inplace=True)

    d_Production_Technologies = {}
    d_Available_Technologies = {}
    d_Production = {}
    df_Production_Technologies =d0_Production_Technologies[L_sectors[0]]
    df_Available_Technologies = d0_Available_Technologies[L_sectors[0]]
    df_Production=d0_Production[L_sectors[0]]
    for sector in L_sectors[1::]:
        df_Production_Technologies=df_Production_Technologies.join(d0_Production_Technologies[sector],how="outer")
        df_Available_Technologies = pd.concat([df_Available_Technologies,d0_Available_Technologies[sector]])
        df_Production= pd.concat([df_Production,d0_Production[sector]])

    df_Production_Technologies=df_Production_Technologies.fillna(0).reset_index()
    df_Available_Technologies = df_Available_Technologies.reset_index()
    df_Production = df_Production.fillna(0).reset_index()

    for year in L_years:
        d_Production_Technologies[year]=df_Production_Technologies[df_Production_Technologies.Year==year].drop("Year",axis=1)
        d_Available_Technologies[year] = df_Available_Technologies[df_Available_Technologies.Year == year].drop(
            "Year", axis=1)
        d_prod_year={}
        for country in L_countries:
            d_prod_year_country={}
            for scenario in L_scenarios:
                d_prod_year_country[scenario]=df_Production[(df_Production.Year==year)&(df_Production.Country==country)][["Resource","Production "+scenario,"Margin"]]
                d_prod_year_country[scenario].rename(columns={"Production "+scenario:"Production"},inplace=True)
            d_prod_year[country]=d_prod_year_country
        d_Production[year]=d_prod_year

    return d_Resource_Technologies,d_Resource_Availability,d_Production_Technologies,d_Available_Technologies,d_Production

d_Resource_Technologies,d_Resource_Availability,d_Production_Technologies,d_Available_Technologies,d_Production=import_industry_data()
results=optim_industry(d_Resource_Technologies[2015],d_Resource_Availability[2015]['FR'],d_Production_Technologies[2015],
               d_Available_Technologies[2015],d_Production[2015]["FR"]["trends"],opti2mini="cost",carbon_tax=0)
print(results)
