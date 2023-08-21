# Importations

import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import pandas as pd
pd.options.plotting.backend = "plotly"

import numpy as np
from scipy.signal import butter,filtfilt

from mycolorpy import colorlist as mcp

from datetime import datetime
from copy import deepcopy
import json
import pkgutil
from io import StringIO

# Global variables

area_from_ORDER = ['FR','DE','GB','ES','IT','BE','CH']
TECHNO_ORDER = ['old_nuke', 'new_nuke', 'biomass', 'wind_power_off_shore', 'wind_power_on_shore', 'solar', 'hydro_river', 'hydro_reservoir','ccgt_h2', 'ocgt_h2','ccgt', 'ocgt', 'demand_not_served']

# Post Processing tools

def extractCosts(Variables):
    
    if "area_from" in Variables['energy'].columns:
        if 'planning_conversion_costs' in Variables.keys():
            df = Variables['planning_conversion_costs'].set_index(["area_from", "conversion_technology"]) / 10 ** 9;
            df = df.merge(pd.DataFrame(Variables['operation_fuel_costs'].set_index(["area_from", "conversion_technology"]) / 10 ** 9),left_on=["area_from", "conversion_technology"], right_on=["area_from", "conversion_technology"])
            df.columns = ["Capacity_Milliards_euros", "Energy_Milliards_euros"]
            
            # compute total 
            df["Total_Milliards_euros"] = df.sum(axis=1)
            # df = df.sort_values("Total_Milliards_euros", ascending=False)

        else:
            df = pd.DataFrame(Variables['operation_fuel_costs'].set_index(["area_from", "conversion_technology"]) / 10 ** 9)
            df.columns = ["Energy_Milliards_euros"]
            # df = df.sort_values("Energy_Milliards_euros", ascending=False)


    else:
        if 'planning_conversion_costs' in Variables.keys():
            df = Variables['planning_conversion_costs'].set_index("conversion_technology") / 10 ** 9;
            df = df.merge(pd.DataFrame(Variables['operation_fuel_costs'].set_index("conversion_technology") / 10 ** 9),
                          left_on="conversion_technology", right_on="conversion_technology")
            df.columns = ["Capacity_Milliards_euros", "Energy_Milliards_euros"]
            # compute total 
            df["Total_Milliards_euros"] = df.sum(axis=1)
            # df = df.sort_values("Total_Milliards_euros", ascending=False)
        else :
            df = pd.DataFrame(Variables['operation_fuel_costs'].set_index("conversion_technology") / 10 ** 9)
            df.columns = ["Energy_Milliards_euros"]
            # df = df.sort_values("Energy_Milliards_euros", ascending=False)
            
            
    return df






def extractEnergyCapacity(Variables) :
    
    if "area_from" in Variables['energy'].columns:

        if len(Variables['energy']['area_from'].unique())==1:
            production_df = Variables['energy'].pivot(index=['area_from',"date"], columns='conversion_technology', values='energy')
        else:
            production_df = EnergyAndExchange2Prod(Variables)
        EnergyCapacity_df = Variables['capacity'].set_index(["area_from","conversion_technology"]) / 10 ** 3;
        EnergyCapacity_df = EnergyCapacity_df.merge(pd.DataFrame(Variables['energy'].groupby(by=["area_from","conversion_technology"]).sum() / 10 ** 6), left_on=["area_from","conversion_technology"], right_on=["area_from","conversion_technology"])
        EnergyCapacity_df.columns = ["Capacity_GW", "Production_TWh"]
        
    else:
        
        production_df = Variables['energy'].pivot(index="date", columns='conversion_technology', values='energy')
        EnergyCapacity_df = Variables['capacity'].set_index("conversion_technology") / 10 ** 3;
        EnergyCapacity_df = EnergyCapacity_df.merge(pd.DataFrame(production_df.sum(axis=0) / 10 ** 6),left_on="conversion_technology", right_on="conversion_technology")
        
        EnergyCapacity_df.columns = ["Capacity_GW", "Production_TWh"]
        
    return EnergyCapacity_df 

def EnergyAndExchange2Prod(Variables,EnergyName='energy',exchangeName='Exchange'):
    Variables["exchange"].columns = ['area_from1', 'area_from2', 'date', 'exchange']
    area_from = Variables['energy'].area_from.unique()
    production_df = Variables['energy'].pivot(index=["area_from","date"], columns='conversion_technology', values='energy')
    
    ToAREA=[]
    
    for AREA in area_from:
        ToAREA.append(Variables["exchange"].loc[(Variables["exchange"].area_from2 == AREA), ["date", "exchange","area_from1","area_from2"]].rename(columns={"area_from2": "area_from"}).pivot(index=["area_from","date"], columns='area_from1', values='exchange'))
        
    ToAREA_pd=pd.concat(ToAREA)
    production_df = production_df.merge(ToAREA_pd, how='inner', left_on=["area_from","date"], right_on=["area_from","date"])
    #exchange analysis
    return(production_df);

def expand_grid(x, y,names):
    
    res=pd.DataFrame()
    xG, yG = np.meshgrid(x, y) # create the actual grid
    res.loc[:,names[0]] = xG.flatten() # make the grid 1d
    res.loc[:,names[1]] = yG.flatten() # same
    return res # return a dataframe

# compute monotones

def getMonotonesPower(production):
    """From a production df with a (date) index and technologies as columns return for each area and each technology the associate dpower monotone
    """

    DATE = production.index.get_level_values("date").unique()
    TECHNO_PROD = production.columns

    monotones_power = pd.DataFrame(
        index = pd.Index(range(len(DATE)), name="HOURS"),
        columns = TECHNO_PROD,
        dtype = "float64"
    )

    for techno in TECHNO_PROD:
        
        x = production[techno].values
        x.sort()
        monotones_power.loc[slice(None),techno] = x[::-1]
        
    return monotones_power

def getMonotonesPower_multiAreas(production):
    """From a production df with a (areas,date) multiindex and technologies as columns return for each area and each technology the associate dpower monotone
    """

    area_from = production.index.get_level_values("area_from").unique()
    DATE = production.index.get_level_values("date").unique()
    TECHNO_PROD = production.columns

    monotones_power = pd.DataFrame(
        index = pd.MultiIndex.from_product([area_from, range(len(DATE))], names=["area_from","HOURS"]),
        columns = TECHNO_PROD,
        dtype = "float64"
    )

    for area in area_from:
        for techno in TECHNO_PROD:
            
            x = production.loc[area][techno].values
            x.sort()
            monotones_power.loc[(area,slice(None)),techno] = x[::-1]
            
    return monotones_power

# load color list, for each technos

def get_color_dict(alpha=0.8):
    """Load the colors associated with each technos, from the associated json file, return a dict with "techno" : "color"
    """

    data = pkgutil.get_data(__name__, "metadata/color_dict.json")
    color_dict = json.load(StringIO(data.decode("utf-8")))

    #transparency
    a_hex = "%0.2X" % int(alpha * 255)
    color_dict = dict([(name, color+a_hex) for name,color in color_dict.items()])
    
    return color_dict 

def get_color_list(techno_list, color_dict=None):
    """From a list of technos, return the list of associated colors
    """
    if color_dict is None : 
        color_dict = get_color_dict()

    color_list = [color_dict[techno] for techno in techno_list]
    
    return color_list

def set_transparency(color, alpha=255):
    """set the transparency, alpha in [0,255], return the new color in the same format
    0 : transparent
    255 : opaque
    color in format "#RRGGBB""
    """
    color_rgb = color[:7]
    alpha_hex = hex(alpha).upper()[2:]
    return color_rgb + alpha_hex
    
def rgbaHex2rgba(s):
    """Convert a rgba hex string (ex : "#FACB4E7F" to the string asked by plotly)
    """
    r = int(s[1:3], 16)
    g = int(s[3:5], 16)
    b = int(s[5:7], 16)
    a = int(s[7:9], 16)/255.
    return f"rgba({r},{g},{b},{a:.2f})"

# Filtering a signal

def lowpass_filter_df_hourly(df, cutoff_period_hour=10):
    """Apply a lowpass filter to time series
    """
    
    fs = 1 # sampling freq
    nyq =0.5*fs # nyquist freq
    cutoff_freq = 1/cutoff_period_hour
    normal_cutoff = cutoff_freq / nyq
    
    # Get the filter coefficients 
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
        
    for col in df.columns:
        df[col] = filtfilt(b, a, df[col].values)
        
    return df

# Graphical functions to plot production installed capacities and total production

def productionCapa_stackedBarChart(energyCapacity,capaDisp="TWh",text=False):
    """Stocked bar chart showing either installed capa or energy dispatch
    - energyCapacity [AREA,conversion_technology] dataframe with [Capacity_GW,Production_TWh] columns
    - capaDisp : either "TWh" or "GW"
    - text : add text displaying produced energy by each techno"""
    
    if capaDisp == "TWh" :
        values = "Production_TWh"
        ylabel = "Energy (TWh)"
        unit = "TWh"
    elif capaDisp == "GW" :
        values = "Capacity_GW"
        ylabel = "Power (GW)"
        unit = "GW"
    else :
        raise Exception("capaDisp must be either 'TWh' or 'GW'")
    
    dispatchTWh = energyCapacity.pivot_table(values=values, index="area_from", columns="conversion_technology").loc[area_from_ORDER, TECHNO_ORDER[::-1]]

    color_dict = get_color_dict()
    colors = [color_dict[tech][:-2] for tech in dispatchTWh.columns]
 
    text_auto = ".3s" if text else False
    fig = dispatchTWh.plot.bar(
        labels=dict(value=ylabel, index="", variable=""), 
        title=f"Electricity production in France in 2050", 
        color_discrete_sequence=colors,text_auto=text_auto)

    # get total production
    total_prod = {s:int(dispatchTWh.loc[s].sum()) for s in area_from_ORDER}

    for s,t in total_prod.items():
        
        fig.add_annotation(
            x=s, y=t,text = f"{t}{unit}",
            showarrow = False,
            yshift = 12,
            font=dict(family="Courier New, monospace",size=15,color="firebrick")
            )

    fig.update_layout(
        title = "",
        legend_traceorder="reversed",
        xaxis_title = "",
        )

    return fig


def installedCapa_barChart(energyCapacity, minmaxCapacities=None, color_dict=None, **kwargs):
    """Bar chart showing installed capa for each techno and each area"""

    capacity = pd.pivot_table(energyCapacity, values="Capacity_GW", index="area_from", columns="conversion_technology").loc[area_from_ORDER]

    area_from = capacity.index
    
    if color_dict is None:
        color_dict = {key:rgbaHex2rgba(color) for key,color in get_color_dict().items()}
    
    TECHNOS = [techno for techno in color_dict.keys() if techno in capacity.columns] # to get the wanted ordering
    
    if minmaxCapacities is not None:
        
        # if capacities bounds specified : check if upper bounds has been reached
        max_capacity = pd.pivot_table(minmaxCapacities, values="planning_max_capacity", index="area_from", columns="conversion_technology").loc[area_from_ORDER]/1000
        max_cap_reached = capacity.eq(max_capacity)
        max_cap_reached_txt = pd.DataFrame([[""]*len(max_cap_reached.columns)], index=max_cap_reached.index, columns=max_cap_reached.columns)
        max_cap_reached_txt[(max_cap_reached & ~ (max_capacity == 0))] = "➕"
        
        min_capacity = pd.pivot_table(minmaxCapacities, values="planning_min_capacity", index="area_from", columns="conversion_technology").loc[area_from_ORDER]/1000
        min_cap_reached = capacity.eq(min_capacity)
        min_cap_reached_txt = pd.DataFrame([[""]*len(min_cap_reached.columns)], index=min_cap_reached.index, columns=max_cap_reached.columns)
        min_cap_reached_txt[(min_cap_reached & ~ (capacity == 0))] = "➖"
        
        capacity_bound_reached = max_cap_reached_txt + min_cap_reached_txt
        
        # case min=max 
        capacity_bound_reached[min_capacity.eq(max_capacity)] = "○"
                
        data = [
            go.Bar(name=techno, x=area_from, y=capacity[techno], text=capacity_bound_reached[techno], yaxis='y', offsetgroup=i+1, marker_color=color_dict[techno]) for i,techno in enumerate(TECHNOS)
        ]
        
    else :  

        data = [
            go.Bar(name=techno, x=area_from, y=capacity[techno], yaxis='y', offsetgroup=i+1, marker_color=color_dict[techno]) for i,techno in enumerate(TECHNOS)
        ]

    fig = go.Figure(
        data = data,
        layout={
            'yaxis': {'title': 'Capacity (GW)'},
            "title": "Installed capacities"
        }
    )
    
    fig.update_traces(textposition="outside")
    
    if minmaxCapacities is not None : 
        annotations=[
            go.layout.Annotation(
                text='➕ : upper bound reached<br>➖ : lower bound reached<br>○ : fully constrained capacity',
                align='left',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=1,
                y=1,
                bordercolor='black',
                borderwidth=1,
                bgcolor="white"
            )
        ]
    else :
        annotations = []

    # Change the bar mode
    fig.update_layout(
        barmode='group',
        title=kwargs.get("title","Installed capacities"),
        paper_bgcolor='rgba(255,255,255,255)',
        plot_bgcolor='rgba(255,255,255,255)',
        annotations = annotations
    )
    
    fig.update_yaxes(showline=True, linewidth=1, gridcolor='black')
    return fig

def installedCapaStoragePower_barChart(storage, flex_conso):
    """Bar chart showing installed storage capacities (Power)"""
    
    storage = pd.concat([flex_conso, storage.drop(labels="storage_power_to_gaz_to_power", axis=1)], axis=1).rename(columns={"EV":'Battery_EV'})
        
    capacity = storage.abs().groupby(level=[0]).max()
    
    area_from = capacity.index
    
    color_dict = {key:rgbaHex2rgba(color) for key,color in get_color_dict().items()}

    data = [
        go.Bar(name=techno, x=area_from, y=capacity[techno], yaxis='y', offsetgroup=i+1, marker_color=color_dict[techno]) for i,techno in enumerate(capacity.columns)
    ]

    fig = go.Figure(
        data = data,
        layout={
            'yaxis': {'title': 'Capacity (GW)'},
            "title": "Installed capacities storage (power)"
        }
    )

    # Change the bar mode
    fig.update_layout(
        barmode='group',
        paper_bgcolor='rgba(255,255,255,255)',
        plot_bgcolor='rgba(255,255,255,255)',
    )
    
    fig.update_yaxes(showline=True, linewidth=1, gridcolor='black')

    
    return fig

def installedCapaStorageEnergy_barChart(storage):
    """Bar chart showing installed storage capacities (Energy)"""

    capacity = storage.abs().groupby(level=[0]).max()/1e3
    
    area_from = capacity.index
    
    color_dict = {key:rgbaHex2rgba(color) for key,color in get_color_dict().items()}
    
    TECHNOS = [techno for techno in color_dict.keys() if techno in capacity.columns] # to get the wanted ordering

    data = [
        go.Bar(name=techno, x=area_from, y=capacity[techno], yaxis='y', offsetgroup=i+1, marker_color=color_dict[techno]) for i,techno in enumerate(TECHNOS)
    ]

    fig = go.Figure(
        data = data,
        layout={
            'yaxis': {'title': 'Stock capacity (GWh)'},
            "title": "Installed capacities storage (stock)"
        }
    )

    # Change the bar mode
    fig.update_layout(
        barmode='group',
        paper_bgcolor='rgba(255,255,255,255)',
        plot_bgcolor='rgba(255,255,255,255)',
    )
    
    fig.update_yaxes(type="log", showline=True, linewidth=1, gridcolor='black')

    
    return fig

# Bar chart plot for costs

def costPerCountry(energyCapacity, cost_production, cost_storage, cost_flex):
    """Bar chart of the energy cost for each area (EUR/MWh)"""
    
    # Agregation of the different cost in a signe dataframe
    cost_system = pd.concat([

        cost_production.groupby(level=[0]).sum().rename(columns={"Capacity_Milliards_euros":"CAPEX-production","Energy_Milliards_euros":"Energy"})[["CAPEX-production","Energy"]],
        
        cost_storage.sum(axis=1).reset_index(name="Storage").set_index("area_from"),
        
        cost_flex.sum(axis=1).reset_index(name="Flexibility").set_index("area_from")
        ], axis=1)

    # On se ramène à des EUR/MWh
    cost_system_normed = cost_system.divide(energyCapacity["Production_TWh"].groupby(level=[0]).sum(), axis=0).loc[area_from_ORDER]*1e3

    colors = [
        "#435F7B", 
        "#F0875D",
        "#FF78F0",
        "#A2C62C"
    ]

    fig = cost_system_normed.plot.bar(labels=dict(value="EUR/MWh", index="", variable=""), title="Total cost of the electricity production system", color_discrete_sequence=colors)
    
    return fig

def costDecomposed_barChart(cost_production, cost_storage, cost_flex):
    """Bar chart showing the cost of each techno in each area
    """
        
    cost_prod_total = pd.pivot_table(cost_production, values="Total_Milliards_euros", index="area_from", columns="conversion_technology")
    # costs = pd.concat([cost_prod_total, cost_storage, cost_flex], axis=1).loc[area_from_ORDER]
    costs = pd.concat([cost_prod_total, cost_storage, cost_flex], axis=1)
    
    total_cost = costs.sum(axis=1).round(1)
    costs.rename(index = {
        area : f"{area} : {total_cost[area]}Md€/an" for area in total_cost.index
    }, inplace=True)

    area_from = costs.index
    color_dict = {key:rgbaHex2rgba(color) for key,color in get_color_dict().items()}
    
    TECHNOS = [techno for techno in color_dict.keys() if techno in costs.columns] # to get the wanted ordering

    data = [
        go.Bar(name=techno, x=area_from, y=costs[techno], yaxis='y', offsetgroup=i+1, marker_color=color_dict[techno]) for i,techno in enumerate(TECHNOS)
    ]

    fig = go.Figure(
        data = data,
        layout={
            'yaxis': {'title': 'Cost over one year (Md€/an)'},
            "title": "Decomposed system costs"
        }
    )

    # Change the bar mode
    fig.update_layout(
        barmode='group',
        title="Decomposed system costs",
        paper_bgcolor='rgba(255,255,255,255)',
        plot_bgcolor='rgba(255,255,255,255)',
    )
    
    fig.update_yaxes(showline=True, linewidth=1, gridcolor='black')
    return fig

# pie charts for porduction

def production_pieChart(production_annual, color_dict=None):
    """Plot production dsdtribution for the diffferent nodes, production_annual must be a df with multiindex (area_from, conversion_technology) and a column called "Production_TWh"
    """
    
    # area_from = areas_order
    # area_from = area_from
    area_from = production_annual.index.get_level_values(level="area_from").unique()

    # compute total productions
    total_prod = {area : production_annual.loc[area]["Production_TWh"].sum() for area in area_from}

    # get color list
    if color_dict is None:
        color_dict = get_color_dict()

    color_list = get_color_list(production_annual.index.get_level_values(level="conversion_technology").unique(), color_dict=color_dict)

    # Graph generation
    specs=[[{'type':'domain'}]*len(area_from)]
    subplots_title = [f"{area} - {int(total_prod[area])}TWh" for area in area_from]

    fig = make_subplots(rows=1, cols=len(area_from), specs=specs, subplot_titles=subplots_title)

    for i,area in enumerate(area_from):
        
        prod_area = production_annual.loc[area]["Production_TWh"].reset_index()
        prod_area["Production (%)"] = prod_area["Production_TWh"]/total_prod[area]*100
        prod_area = prod_area.round(2)
                
        fig.add_trace(
            go.Pie(labels=prod_area["conversion_technology"], values=prod_area["Production_TWh"], name=area, marker_colors=color_list),1,i+1
            )
        
    fig.update_traces(textposition='inside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    
    return fig 


# graphical functions to plot table data

def plotTable(df, title=""):
    """Plot a dataframe in a plotly table
    """
    fig = go.Figure(data=[go.Table(
        header=dict(values=[df.index.name] + list(df.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[df.index.values] + [df[col] for col in df.columns],
                fill_color='lavender',
                align='left'))
    ])

    fig.update_layout(
        title=title,
        )

    return fig

# graphical functions to plot load factors (facteurs de charge)

def loadFactors(energyCapacity, title="", **kwargs):
    """Plot a bar graph showing for each techno the associated load factor over the year.
    The energyCapacity dataframe must have a "Production_TWh" and a "Capacity_GW" column for different techno.
    """
        
    techno_prod_fatal = pd.DataFrame({
    "conversion_technology" : ['old_nuke', 'wind_power_off_shore', 'hydro_reservoir', 'new_nuke', 'demand_not_served', 'ccgt', 'wind_power_on_shore', 'biomass', 'solar', 'ocgt', 'hydro_river', 'coal', 'lignite', 'ccgt/ocgt CH4', 'ccgt/ocgt H2'],
    "is_fatal" : [False, True, False, False, False, False, True, False, True, False, True, False, False, False, False]
    }).set_index("conversion_technology")
    
    energyCapacity = energyCapacity.join(techno_prod_fatal)

    # calcul du facteur de charge
    nbofhours = 8760    
    
    energyCapacity["load_factor_%"] = energyCapacity.apply(lambda row: np.nan if row["Capacity_GW"] <= 0 else row["Production_TWh"]*1000/(nbofhours*row["Capacity_GW"])*100, axis=1)
    
    # energyCapacity.loc["EU"].to_csv("load_factor.csv")
    if isinstance(energyCapacity.index, pd.MultiIndex):
        idx1name = energyCapacity.index.names[0]
        fig = px.scatter(energyCapacity.reset_index().sort_values(by=["is_fatal", "load_factor_%"], ascending=False), y="load_factor_%", x="conversion_technology", color=idx1name, symbol=idx1name, color_discrete_map=kwargs.get("color_map",None), symbol_map=kwargs.get("symbol_map",None), opacity=kwargs.get("opacity",None))
    else:
        # no area_from index
        fig = px.scatter(energyCapacity.reset_index().sort_values(by=["is_fatal", "load_factor_%"], ascending=False), y="load_factor_%", x="conversion_technology", color_discrete_map=kwargs.get("color_map",None), symbol_map=kwargs.get("symbol_map",None), opacity=kwargs.get("opacity",None))
        
    
    fig.update_traces(marker_size=10)
    fig.update_layout(
        title = title,
        xaxis_title = "",
        yaxis_title = "Load factor (%)"
    )
    
    return fig



# graphical functions to plot time series 

def MyPlotly(df,Names=None,fill=True, **kwargs):
    
    default_colors = deepcopy(plotly.colors.DEFAULT_PLOTLY_COLORS)
    
    if fill:
        color_dict = {key:rgbaHex2rgba(color) for key,color in get_color_dict(1).items()}
    else : 
        # reset transparency to zero
        color_dict = {key:rgbaHex2rgba(set_transparency(color, alpha=255)) for key,color in get_color_dict(1).items()}

    fig = go.Figure()
    i=0
    
    if type(df) == pd.core.series.Series:
        
        if df.name is not None:
            color = color_dict[df.name] if df.name in color_dict.keys() else default_colors.pop(0)
        else : 
            color = default_colors.pop(0)
            
            
        if fill :
            fig.add_trace(go.Scatter(x=df.index, y=df.values , fill='tozeroy',mode='none', line=dict(color=color))) # fill down to xaxis
        else :
            fig.add_trace(go.Scatter(x=df.index, y=df.values ,mode='lines', line=dict(color=color)))
            
    else : # dataframe      
        
        if Names is None : Names = df.columns 
        
        for col,name in zip(df.columns, Names):
            
            color = color_dict[col] if col in color_dict.keys() else default_colors.pop(0)
                    
            if i==0:
                if fill :
                    fig.add_trace(go.Scatter(x=df.index, y=df[col] , fill='tozeroy',mode='none', line=dict(color=color), name=name)) # fill down to xaxis
                else :
                    fig.add_trace(go.Scatter(x=df.index, y=df[col],mode='lines', line=dict(color=color), name=name))
                colNames=[col]
            else:
                colNames.append(col)
                if fill :
                    fig.add_trace(go.Scatter(x=df.index, y=df[col], fill='tozeroy',mode='none', line=dict(color=color), name=name)) # fill to trace0 y
                else :
                    fig.add_trace(go.Scatter(x=df.index, y=df[col],mode='lines', line=dict(color=color), name=name)) 
                    
            i=i+1
        
    # add axes labels and title
    fig.update_layout(
        title=kwargs.get("title",""),
        xaxis_title=kwargs.get("xaxis_title",""),
        yaxis_title=kwargs.get("yaxis_title",""),
        legend_title=kwargs.get("legend_title",""),

    )
    
    fig.layout.template = 'plotly_white'
    
    no_slider = kwargs.get("no_slider",False)
    if not(no_slider):
        fig.update_xaxes(rangeslider_visible=True)
    
    log_yaxis = kwargs.get("log_yaxis",False)
    if log_yaxis:
        fig.update_yaxes(type="log")
        
    return(fig)


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    c_rgb= colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

    return mc.to_hex(c_rgb)

def gen_grouped_color_map(col_class_dict,cmap="Set1"):

    if type(col_class_dict)==dict:
        col_class_df= pd.DataFrame().from_dict(col_class_dict, orient='index')
        col_class_df=col_class_df.reset_index()
        col_class_df.columns=["col","Category"]
    n = max(col_class_df.Category.unique())
    base_color_codes = mcp.gen_color(cmap=cmap,n=n)
    my_color_dict={}
    col_class_df_grouped=col_class_df.groupby("Category")
    for name, group in col_class_df_grouped:
        i=0
        gradient = np.linspace(0.3, 1, len(group.col))
        for colname in group.col:
            my_color_dict[colname]=lighten_color(base_color_codes[name-1],gradient[i])
            i+=1
    return my_color_dict



def plotProduction(df, conso=None, flex_conso=None, color_dict=None, **kwargs):
    """From a dataframe with a date index and several columns, plot the times series one on top of another.

    Args:
        df : The main data dataframe
        Conso : To be plot as single line. Defaults to -1.
        isModifyOrder (bool, optional): _description_. Defaults to True.
        Names (int, optional): _description_. Defaults to -1.
        color_dict : A dictionary containing for each column name of y_df the associated color, the columns will be plot in the order of appearance in color_dict. Defaults to None.
        fill : fille between the stacked traces. Default to True.
    """
    
    color_dict = get_color_dict()
    
    # convert color format
    for col,value in color_dict.items():
        color_dict[col] = rgbaHex2rgba(value)

    colnames = list(color_dict.keys())  
    
    # exclude zeros columns
    for col in df.columns:
        if col in colnames :
            if df[col].sum() == 0.0: colnames.remove(col)
        else : 
            print(f"WARNING : {col} does not have an associated color, it won't be plotted")
   
    
    # getting index bounds
    df_idx = df.index
    
    dt1 = kwargs.get("start_date", df_idx[0])
    dt2 = kwargs.get("end_date", df_idx[-1])
    dt_fmt = kwargs.get("date_fmt", "%d/%m/%Y")
    
    if type(dt1) == str: dt1 = datetime.strptime(dt1, dt_fmt)
    if type(dt2) == str: dt2 = datetime.strptime(dt2, dt_fmt)
        
    df = df.loc[((df.index.get_level_values("date") > dt1) & (df.index.get_level_values("date") <= dt2))]
    df_idx = df.index

    if conso is not None: conso = conso.loc[((conso.index > dt1) & (conso.index <= dt2))]
    if flex_conso is not None: flex_conso = flex_conso.loc[((flex_conso.index > dt1) & (flex_conso.index <= dt2))]
    

    # separate the positive timeseries from the negative
    pos_data, neg_data = [],[]
    eps = 0.1

    for col in df.columns:
        if (df[col] < -eps).any():
            pos_data.append(df[col].clip(lower=0))
            neg_data.append((-df[col]).clip(lower=0))
        else:
            pos_data.append(df[col])
                        
    # check if there is time series with negative values
    plot_neg = not(neg_data == [])
    
    # concat them
    pos_df = pd.concat(pos_data, axis=1)
    if plot_neg: 
        neg_df = pd.concat(neg_data, axis=1)

    
    fig = go.Figure()
    
    # plot negative time series
    if plot_neg :
                
        # get columns order
        neg_colnames = []
        for col in colnames :
            if col in neg_df.columns : neg_colnames.append(col) 
                                
        # sort columns 
        neg_df = neg_df[neg_colnames]
        
        # cumulative sum
        neg_df = neg_df.cumsum(axis=1)
        
        # reverse colnames list : plot last first
        neg_colnames = neg_colnames[::-1]
                
        if len(neg_colnames) >= 2:
            # plot first time serie : no previous to fill to
            col = neg_colnames[0]
            col_next = neg_colnames[1]
            # line only
            fig.add_trace(go.Scatter(x=df_idx, y=-neg_df[col], line=dict(color=color_dict[col], width=0.1)))
            # fill to next negative time serie
            fig.add_trace(go.Scatter(x=df_idx, y=-neg_df[col_next], fill="tonexty", mode='none', fillcolor=color_dict[col], name=col))
            
            # plot intermediate time series
            for i,col in enumerate(neg_colnames[1:-1]):
                col_next = neg_colnames[i+2]
                # fill to next negative time serie
                fig.add_trace(go.Scatter(x=df_idx, y=-neg_df[col_next], fill="tonexty", mode='none', fillcolor=color_dict[col], name=col))
                
        # plot last time serie with fill to zero
        col = neg_colnames[-1]
        fig.add_trace(go.Scatter(x=df_idx, y=-neg_df[col], fill="tozeroy", mode='none', fillcolor=color_dict[col], name=col))
        
    # plot positive time series

    # get columns order
    pos_colnames = []
    for col in colnames :
        if col in pos_df.columns : pos_colnames.append(col)   
                
    # sort columns 
    pos_df = pos_df[pos_colnames]
    
    # cumulative sum
    pos_df = pos_df.cumsum(axis=1)
        
    # first : tozeroy
    col = pos_colnames[0]
    fig.add_trace(go.Scatter(x=df_idx, y=pos_df[col], fill="tozeroy", mode='none', fillcolor=color_dict[col], name=col))
    
    # next : fill to previous trace
    for col in pos_colnames[1:]:
        fig.add_trace(go.Scatter(x=df_idx, y=pos_df[col], fill="tonexty",fillcolor=color_dict[col], mode='none', name=col))
        
        
    # plot flexible consumption (EVs and H2)
    if flex_conso is not None:
        
        # we substract the flexible consumption to the total consumption
        flex_conso = pd.concat([conso, -flex_conso[["H2","EV"]]], axis=1).cumsum(axis=1)

        # first plot option
        # flex_color = {"H2" : "#A78ABE", "EV" : "#00690E"}
        # for flex in ["H2","EV"]:
        #     fig.add_trace(go.Scatter(x=conso.index, y=flex_conso[flex], name=flex, line=dict(color=flex_color[flex], width=1)))
    
        # second plot option
        fig.add_trace(go.Scatter(x=conso.index, y=flex_conso["EV"], name="-H2-EVs", line=dict(color="red", width=1)))
        
        
    # plot consumption
    if conso is not None:
        fig.add_trace(go.Scatter(x=conso.index, y=conso["energy_demand"], name="Consumption", line=dict(color='black', width=2)))
            
    # add axes labels and title
    fig.update_layout(
        title=kwargs.get("title",""),
        xaxis_title=kwargs.get("xaxis_title",""),
        yaxis_title=kwargs.get("yaxis_title",""),
        legend_title=kwargs.get("legend_title",""),
        paper_bgcolor='rgba(255,255,255,255)',
        plot_bgcolor='rgba(255,255,255,255)',
        legend=dict(yanchor="top", y=1.2, xanchor="left", x=1)
        # yaxis = dict(
        #     tickmode = 'linear',
        #     tick0 = 0,
        #     dtick = 20
        # )
    )

    # Change grid color and axis colors
    fig.update_yaxes(showline=True, linewidth=1, gridcolor='black')
    fig.update_xaxes(rangeslider_visible=True)
    
    return(fig)

def MyStackedPlotly(y_df, Conso=-1,isModifyOrder=True,Names=-1,color_dict=None):
    '''
    :param x: 
    :param y: 
    :param Names:
    :return: 
    '''
    if Conso.__class__.__name__=="DataFrame":
        if not "energy_demand" in Conso.columns:
            Conso= Conso.rename(columns ={Conso.columns[0] : "energy_demand"})
    if type(y_df.columns) == pd.MultiIndex:
        if len(y_df.columns[0]) == 2:
            i = 1
            col_class_dict = {}
            for col1, new_df in y_df.groupby(level=0, axis=1):
                for col2 in new_df.columns:
                    col_class_dict["_".join(col2)] = i
                i += 1
            y_df.columns = ["_".join(col) for col in y_df.columns]
            color_dict = gen_grouped_color_map(col_class_dict)
        else: "column multi index only implemented for 2 dimensions"

    if isModifyOrder: y_df=ModifyOrder_df(y_df) ### set Nuke first column
    if (Names.__class__ == int): Names=y_df.columns.unique().tolist()
    x_df=y_df.index
    fig = go.Figure()
    i = 0
    if color_dict == None:
        colnames = y_df.columns
    else:
        colnames = list(color_dict.keys())

    for col in colnames:
        if i == 0:
            if color_dict==None:
                fig.add_trace(go.Scatter(x=x_df, y=y_df[col], fill='tozeroy',
                                         mode='none', name=Names[i]))  # fill down to xaxis
            else:
                fig.add_trace(go.Scatter(x=x_df, y=y_df[col], fill='tozeroy',fillcolor=color_dict[col],
                                         mode='none', name=col))  # fill down to xaxis
            colNames = [col]
        else:
            colNames.append(col)
            if color_dict==None:
                fig.add_trace(go.Scatter(x=x_df, y=y_df.loc[:, y_df.columns.isin(colNames)].sum(axis=1), fill='tonexty',
                                         mode='none', name=Names[i]))  # fill to trace0 y
            else:
                fig.add_trace(go.Scatter(x=x_df, y=y_df.loc[:, y_df.columns.isin(colNames)].sum(axis=1), fill='tonexty',
                                         fillcolor=color_dict[col],mode='none', name=col))  # fill to trace0 y
        i = i + 1

    if (Conso.__class__ != int):
        fig.add_trace(go.Scatter(x=Conso.index,
                                 y=Conso["energy_demand"], name="Conso",
                                 line=dict(color='red', width=0.4)))  # fill down to xaxis
        if "NewConsumption" in Conso.keys():
            fig.add_trace(go.Scatter(x=Conso.index,
                                     y=Conso["NewConsumption"], name="Conso+stockage",
                                     line=dict(color='black', width=0.4)))  # fill down to xaxis

    fig.update_xaxes(rangeslider_visible=True)
    return(fig)

def AppendMyStackedPlotly(fig,y_df,Conso,isModifyOrder=True):
    '''
    :param x:
    :param y:
    :param Names:
    :return:
    '''
    if isModifyOrder: y_df=ModifyOrder_df(y_df) ### set Nuke first column
    Names=y_df.columns.unique().tolist()
    x_df=y_df.index
    i=0
    for col in y_df.columns:
        if i==0:
            fig.add_trace(go.Scatter(x=x_df, y=y_df[col] , fill='tozeroy',
                             mode='none' ,name=Names[i])) # fill down to xaxis
            colNames=[col]
        else:
            colNames.append(col)
            fig.add_trace(go.Scatter(x=x_df, y=y_df.loc[:,y_df.columns.isin(colNames)].sum(axis=1), fill='tonexty',
                                     mode='none', name=Names[i]))  # fill to trace0 y
        i=i+1
    fig.add_trace(go.Scatter(x=Conso.index,
                             y=Conso["energy_demand"], name="Conso",
                             line=dict(color='red', width=0.4)))  # fill down to xaxis
    if "NewConsumption" in Conso.keys():
        fig.add_trace(go.Scatter(x=Conso.index,
                                 y=Conso["NewConsumption"], name="Conso+stockage",
                                 line=dict(color='black', width=0.4)))  # fill down to xaxis
    if "ConsoImportExport" in Conso.keys():
        fig.add_trace(go.Scatter(x=Conso.index,
                                 y=Conso["ConsoImportExport"], name="Conso+export-import",
                                 line=dict(color='blue', width=0.4)))  # fill down to xaxis
    fig.update_xaxes(rangeslider_visible=True)
    return(fig)

def MyAreaStackedPlot(df_,Conso=-1,selected_conversion_technology=-1,AREA_name="area_from"):
    df=df_.copy()
    #df.reset_index(inplace=True)
    if (selected_conversion_technology.__class__ == int):
        selected_conversion_technology=df.columns.unique().tolist()
    area_from=df.index.get_level_values('area_from').unique().tolist()
    selected_conversion_technology=ModifyOrder(selected_conversion_technology)
    df=ModifyOrder_df(df)

    visible={}
    for AREA in area_from: visible[AREA] = []
    for AREA in area_from:
        for AREA2 in area_from:
            if AREA2==AREA:
                for TECH in selected_conversion_technology:
                    visible[AREA2].append(True)
                visible[AREA2].append(True)
                visible[AREA2].append(True)
                if 'Storage' in Conso.columns : visible[AREA2].append(True)
            else :
                for TECH in selected_conversion_technology:
                    visible[AREA2].append(False)
                visible[AREA2].append(False)
                visible[AREA2].append(False)
                if 'Storage' in Conso.columns: visible[AREA2].append(False)
    fig = go.Figure()
    dicts=[]
    for AREA in area_from:
        production_df_ = df.loc[(AREA,slice(None)),:]#.reset_index()
        Conso_=Conso.loc[(AREA,slice(None)),:];
        Conso_ = Conso.loc[(AREA,slice(None)),:].reset_index().set_index("date").drop(["area_from"], axis=1);
        production_df_ = df.loc[(AREA,slice(None)),:].reset_index().set_index("date").drop(["area_from"], axis=1);
        #Conso_.reset_index(inplace=True)
        Conso_.loc[:,"ConsoImportExport"] = Conso_.loc[:,"energy_demand"] - production_df_.sum(axis=1)

        fig = AppendMyStackedPlotly(fig,
                            y_df=production_df_,
                            Conso=Conso_)
        dicts.append(dict(label=AREA,
             method="update",
             args=[{"visible": visible[AREA]},
                   {"title": AREA }]))

    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=list(dicts),
            )
        ])
    #plotly.offline.plot(fig, filename='file.html')  ## offline
    return(fig)

def ModifyOrder(Names):
    if "old_nuke" in Names:
        Names.remove("old_nuke")
        Names.insert(0, "old_nuke")
    if "new_nuke" in Names:
        Names.remove("new_nuke")
        Names.insert(0, "new_nuke")
    if "NukeCarrene" in Names:
        Names.remove("NukeCarrene")
        Names.insert(0, "NukeCarrene")

    return(Names)

def ModifyOrder_df(df):
    if "old_nuke" in df.columns:
        Nuke=df.pop("old_nuke")
        df.insert(0, "old_nuke", Nuke)
    if "new_nuke" in df.columns:
        Nuke=df.pop("new_nuke")
        df.insert(0, "new_nuke", Nuke)
    if "NukeCarrene" in df.columns:
        Nuke=df.pop("NukeCarrene")
        df.insert(0, "NukeCarrene", Nuke)
    return(df);

def plotDecomposedConso(x_df,y_df, Tofile=False, TimeName='date'):
    '''
    Function for graphical representation of a consumption decomposed with thermal
    :param x:
    :param y:
    :return:
    '''

    fig=MyStackedPlotly(x_df=x_df,y_df=y_df,Names=y_df.columns.to_list())
    fig.update_layout(title_text="Consommation (MWh)", xaxis_title="date")
    if Tofile: plotly.offline.plot(fig, filename='file.html')
    else: fig.show()

def plotDecomposedConso(dataYear_decomposed_df, Tofile=False, TimeName='date'):
    '''
    Function for graphical representation of a consumption decomposed with thermal
    :param dataYear_decomposed:
    :param Tofile:
    :param TimeName:
    :return:
    '''

    fig=MyStackedPlotly(x_df=dataYear_decomposed_df[TimeName],y_df=dataYear_decomposed_df[["NTS_C","TS_C"]],
                        Names=['Conso non thermosensible','conso thermosensible'])
    fig.update_layout(title_text="Consommation (MWh)", xaxis_title="date")
    if Tofile: plotly.offline.plot(fig, filename='file.html')
    else: fig.show()


def plotProd(dataYear_df,prodNames, Tofile=False, TimeName='date'):
    '''
    Function for graphical representation of a consumption decomposed with thermal
    :param dataYear:
    :param prodNames:
    :param Tofile:
    :param TimeName:
    :return:
    '''

    fig=MyStackedPlotly(x_df=dataYear_df[TimeName],y_df=dataYear_df[[prodNames]],
                        Names=prodNames)
    fig.update_layout(title_text="Consommation (MWh)", xaxis_title="date")
    if Tofile: plotly.offline.plot(fig, filename='file.html')
    else: fig.show()


# To plot marimekko diagrams


def marimekko(df,x_var_name,y_var_name,effectif_var_name,color_discrete_sequence):
    """Graphiques de Marimekko : 

    Args:
        df (_type_): _description_
        x_var_name (_type_): _description_
        y_var_name (_type_): _description_
        effectif_var_name (_type_): _description_
        color_discrete_sequence (_type_): _description_

    Returns:
        _type_: _description_
    """
    labels = df[x_var_name].unique().tolist() #["apples","oranges","pears","bananas"]
    widths = np.array(df.groupby(x_var_name)[effectif_var_name].sum())/df[effectif_var_name].sum()*100
    Y_given_X = (df.groupby([x_var_name,y_var_name])[effectif_var_name].sum()/df.groupby(x_var_name)[effectif_var_name].sum()*100).reset_index()
    # test : Y_given_X.groupby(x_var_name).sum() == 100
    heights = {k: list(v) for k, v in Y_given_X.groupby(y_var_name)[effectif_var_name]}
    Total = df[effectif_var_name].sum()/10**6

    fig = go.Figure()
    for i,key in enumerate(heights):
        fig.add_trace(go.Bar(
            marker_color=color_discrete_sequence[i],
            name=key,
            y=heights[key],
            x=np.cumsum(widths)-widths,
            width=widths,
            offset=0,
            customdata=np.transpose([labels, np.around(widths*heights[key]/100,1),
                                     np.around(widths*heights[key]*Total/(100*100),1)]),
            texttemplate="Nb : %{customdata[2]} Millions, <br>%{customdata[1]} [%total]",
            textposition="inside",
            textangle=0,
            textfont_color="white",
            hovertemplate="<br>".join([
                "Nb : %{customdata[2]} Millions",
                "Prop : %{customdata[1]} [%total]"
            ])
        ))

    fig.update_xaxes(
        tickvals=np.cumsum(widths)-widths/2,
        ticktext= ["%s" % l for l in labels]
    )

    fig.update_xaxes(range=[0,100])
    fig.update_yaxes(range=[0,100])

    fig.update_layout(
        title_text="Marimekko Chart",
        barmode="stack",
        uniformtext=dict(mode="hide", minsize=10),
    )
    return fig
#cond_var_name="residential_type"

def marimekko_2(df,ColorY_var_name,horizontalX_var_name,TextureX_var_name,color_discrete_sequence,effectif_var_name='IPONDL'):
    ## ColorY_var_name : variable codée par couleur répartie sur la hauteur  --  e.g. classe énergétique
    ## horizontalX_var_name : variable codée par X -- e.g. age du bâtiment
    ## TextureX_var_name : variable codée par la texture sur la largeur -- e.g. type de logement
    pattern_sequence = [ '/', 'x', '-', '|', '+', '.',"\\",'']
    pattern_dic = dict(zip(df[TextureX_var_name].unique(),pattern_sequence))
    color_dic = dict(zip(df[ColorY_var_name].unique(), color_discrete_sequence))

    #calcul des distribution verticales et horizontales
    Total = df[effectif_var_name].sum()/10**6
    ColorY_given_horizontalX = (df.groupby([ColorY_var_name,horizontalX_var_name])[effectif_var_name].sum()/df.groupby(horizontalX_var_name)[effectif_var_name].sum()*100).reset_index()
    ColorY_given_horizontalX=ColorY_given_horizontalX.set_index([ColorY_var_name,horizontalX_var_name]). \
        rename(columns={"IPONDL": "Dheight"})
    ColorY_given_horizontalX["y1"] = ColorY_given_horizontalX["Dheight"].groupby([horizontalX_var_name]).cumsum()
    ColorY_given_horizontalX["Dheight0"] = ColorY_given_horizontalX["Dheight"].groupby([horizontalX_var_name]).shift().fillna(0)
    ColorY_given_horizontalX["y0"] = ColorY_given_horizontalX["Dheight0"].groupby([horizontalX_var_name]).cumsum()
    # test : ColorY_given_horizontalX.groupby(horizontalX_var_name)["Dheight"].sum() == 100
    AllX_given_ColorY= (df.groupby([ColorY_var_name,horizontalX_var_name,TextureX_var_name])[effectif_var_name].sum()/df[effectif_var_name].sum()*100).reset_index()
    AllX_given_ColorY=AllX_given_ColorY.set_index([ColorY_var_name,horizontalX_var_name,TextureX_var_name]).\
        rename(columns={"IPONDL": "Proportion"})
    # test : AllX_given_ColorY["Proportion"].sum() == 100
    AllDistrib = AllX_given_ColorY.join(ColorY_given_horizontalX, how="inner")
    AllDistrib["Dwidth"]=AllDistrib["Proportion"]/(AllDistrib["Dheight"]/100)
    AllDistrib["x1"] = AllDistrib["Dwidth"].groupby(ColorY_var_name).cumsum()
    AllDistrib["Dwidth0"] = AllDistrib["Dwidth"].groupby(ColorY_var_name).shift().fillna(0)
    AllDistrib["x0"] = AllDistrib["Dwidth0"].groupby(ColorY_var_name).cumsum()

    #to put labels on X axis
    widths = np.array(df.groupby([horizontalX_var_name])[effectif_var_name].sum()) / df[effectif_var_name].sum() * 100
    labels_X_axis  = df[horizontalX_var_name].unique().tolist()


    LegendList=[]
    fig = go.Figure()
    fig.update_xaxes(range=[0, 100])
    fig.update_yaxes(range=[0, 100])
    for (ColorY_val,horizontalX_val,TextureX_val) in AllDistrib.index:
        cur_distrib = AllDistrib.loc[(ColorY_val,horizontalX_val,TextureX_val),]
        if (ColorY_val,TextureX_val) in LegendList:
            showlegend=False
        else:
            showlegend = True
            LegendList=LegendList+[(ColorY_val,TextureX_val)]
        x0 = cur_distrib.x0; x1=cur_distrib.x1;  y0 = cur_distrib.y0; y1=cur_distrib.y1;
        Effectif = df.set_index([ColorY_var_name,horizontalX_var_name,TextureX_var_name]).\
                       loc[(ColorY_val,horizontalX_val,TextureX_val),effectif_var_name]/10**6
        Proportion = Effectif /Total*100
        fig.add_trace(go.Scatter(
            showlegend=showlegend,
            fillpattern={
                "bgcolor": color_dic[ColorY_val],
                "fgcolor": "grey",
                "shape": pattern_dic[TextureX_val]
            },
            fill='tonexty',
            mode='none',
            marker={"line":{"autocolorscale":False,"color":"#FFFFFF","width":1}},
            marker_size=0,
            name="Class " + ColorY_val + ", "+TextureX_val,
            y=[y0, y0, y1, y1, y0],
            x=[x0, x1, x1, x0, x0],
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            showlegend=False,
            fill='tonexty',
            mode='none',
            marker={"line":{"autocolorscale":False,"color":"#FFFFFF","width":1}},
            marker_size=0,
            name="Class " + ColorY_val + ", "+TextureX_val,
            y=[(y0+y1)/2],
            x=[(x0+x1)/2],
            customdata=np.transpose([ColorY_val, horizontalX_val,TextureX_val,np.around(Effectif, 1),np.around(Proportion, 1)]),
            texttemplate="Nb : %{customdata[3]} Millions, <br> %{customdata[4]} [%total]",
            textposition='middle center',
            #textfont_color="white",
            hovertemplate="<br>".join([
                ColorY_val+","+horizontalX_val+","+TextureX_val,
                "Nb : "+str(np.around(Effectif, 2))+" Millions",
                "Prop : "+str(np.around(Proportion, 2))+" [%total]",
            ])
        ))


        fig.update_xaxes(range=[0, 100])
        fig.update_yaxes(range=[0, 100])
        fig.add_trace(go.Scatter(
            showlegend=False,
            mode='lines',
            hoverinfo="skip",
            line=dict(color='white', width=0.5),
            y=[y0, y0, y1, y1, y0],
            x=[x0, x1, x1, x0, x0],
        ))


    fig.update_xaxes(
        tickvals=np.cumsum(widths)-widths/2,
        ticktext= ["%s" % l for l in labels_X_axis]
    )
    for i in range(0,len(widths)-1):
        x0=np.cumsum(widths)[i]
        fig.add_trace(go.Scatter(
            showlegend=False,
            mode='lines',
            line=dict(color='#DEDEDE', width=1.1),
            y=[0,100],
            x=[x0,x0],
        ))

    fig.update_xaxes(range=[0, 100])
    fig.update_yaxes(range=[0, 100])
    fig.update_layout(
        hovermode="closest",
        legend_title=ColorY_var_name+", <br>"+TextureX_var_name,
        legend=dict(itemsizing="trace",itemwidth=40),
        title_text="Distribution de ("+ ColorY_var_name+","+horizontalX_var_name+","+TextureX_var_name+")",
        uniformtext=dict(mode="hide", minsize=10),
    )
    #plotly.offline.plot(fig, filename='tmp.html')

    return fig
