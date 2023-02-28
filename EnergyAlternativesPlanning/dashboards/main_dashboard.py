import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State

import pickle
import io
import base64
import sys

import pandas as pd

import sys
sys.path.append("..")
import f_graphicalTools as nrjPlot

# import EnergyAlternativesPlaning.f_graphicalTools as nrjPlot

# Global variables

Variables = None
xls_file = None
AREAS_ORDER = ['FR','DE','GB','ES','IT','BE','CH']
TABLE_COLUMNS = ["AREAS","OldNuke","NewNuke","Solar","WindOnShore","WindOffShore","Biomass","Coal","Lignite","CCG","TAC","CCG - H2","TAC - H2","HydroRiver","HydroReservoir","curtailment","HydroStorage","Battery"]


# styling

style_table = dict(
    style_data={
        'color': 'black',
        'backgroundColor': 'white'
    },
    style_data_conditional=[
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(220, 220, 220)',
        }
    ],
    style_header={
        'backgroundColor': 'rgb(210, 210, 210)',
        'color': 'black',
        'fontWeight': 'bold'
    }
)


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    
    # stored data
    dcc.Store(id='energyCapacity'),
    dcc.Store(id='techParameters'),
        
    dcc.Markdown(children=f"""
# Energy Alternatives Planing - Results presentation dashboard

## Loading data

To begin with visualization, upload the two needed files : the input file, under the xlsx format, and the output file, under the pickle format.

"""),
    
    html.H5('xlsx input file'),
    dcc.Markdown(id='div_xls'),
    dcc.Upload(
        id='upload_xls',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select a File')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center'
        }
    ),
    
    dcc.Markdown(id='md_xlsx_sucess'),
    

    html.H5('pickle output file'),
    dcc.Markdown(id='div_pkl'),

    dcc.Upload(
        id='upload_pkl',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select a File')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center'
        }
    ),
    
    dcc.Markdown(id='md_pkl_sucess'),
    
    dcc.Markdown("***"),
    html.H2("Installed capacities and produced energy"),
    
    dcc.Graph(id="installed_capa_graph"),
    dcc.Graph(id="produced_energy_graph"),
    dcc.Graph(id="storage_capa_graph"),
    
    dcc.Markdown("***"),
    html.H2("Input : min/max capacities"),

    html.Div([
        dcc.Markdown("**Lower bound** on capacities (GW)"),
        dash.dash_table.DataTable(
            id='min_capa_table',
            columns=[{"name": i, "id": i} for i in TABLE_COLUMNS],
            **style_table
        )   
    ]),

    html.Div([
        dcc.Markdown("**Upper bound** on capacities (GW)"),
        dash.dash_table.DataTable(
            id='max_capa_table',
            columns=[{"name": i, "id": i} for i in TABLE_COLUMNS],
            **style_table
        )   
    ]),

    dcc.Markdown("***"),
    html.H2("Load factors"),
    dcc.Graph(id="load_factor_graph"),
    
    dcc.Markdown("***"),
    html.H2("System cost"),
    dcc.Graph(id="decomp_costs_graph"),
    

])

# display filenames
@app.callback(
    Output('div_xls','children'),
    Input('upload_xls', 'filename')
)
def display_xls_filename(filename):
    return  f"Loaded file : **{filename}**"


# display filenames
@app.callback(
    Output('div_pkl','children'),
    Input('upload_pkl', 'filename')
)
def display_xls_filename(filename):
    return  f"Loaded file : **{filename}**"


@app.callback(
    [
        Output("min_capa_table", "data"),
        Output("max_capa_table", "data"),
        Output("techParameters", "data"),
        Output('md_xlsx_sucess','children')
    ],
    [Input('upload_xls', 'contents')])
def loading_xls_data(content):
    
    if not content:
        return dash.no_update
    
    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    
    xls_file = io.BytesIO(decoded)    
    
    # extract data
    TechParameters = pd.read_excel(xls_file,"TECHNO_AREAS")
    StorageParameters = pd.read_excel(xls_file,"STOCK_TECHNO_AREAS").set_index(["AREAS", "STOCK_TECHNO"])
    
    StorageParameters["minCapacity"] = StorageParameters["c_min"] / StorageParameters["strhours"]
    StorageParameters["maxCapacity"] = StorageParameters["c_max"] / StorageParameters["strhours"]

    AREAS_ORDER_INPUT = ["FR","DE","GB","CH","IT","BE","ES"]
    TECHNOS_ORDER_INPUT = ["OldNuke","NewNuke","Solar","WindOnShore","WindOffShore","Biomass","Coal","Lignite","CCG","TAC","CCG - H2","TAC - H2","HydroRiver","HydroReservoir","curtailment"]

    minCapa = TechParameters.pivot_table(values="minCapacity", index="AREAS", columns="TECHNOLOGIES").loc[AREAS_ORDER_INPUT,TECHNOS_ORDER_INPUT]/1000
    minCapa_storage = StorageParameters.pivot_table(values="minCapacity", index="AREAS", columns="STOCK_TECHNO").drop("PtGtP", axis=1)/1000
    
    minCapa = pd.concat([minCapa, minCapa_storage], axis=1).reset_index()

    min_capa_dict = minCapa.to_dict(orient='records')
    
    maxCapa = TechParameters.pivot_table(values="maxCapacity", index="AREAS", columns="TECHNOLOGIES").loc[AREAS_ORDER_INPUT,TECHNOS_ORDER_INPUT]/1000
    maxCapa_storage = StorageParameters.pivot_table(values="maxCapacity", index="AREAS", columns="STOCK_TECHNO").drop("PtGtP", axis=1)/1000
    
    maxCapa = pd.concat([maxCapa, maxCapa_storage], axis=1).reset_index()

    max_capa_dict = maxCapa.to_dict(orient='records')

    # TechParameters dataframe, jsonified
    dataTechParameters = TechParameters.to_dict('records')

    # return fig_table_min, fig_table_max
    return min_capa_dict, max_capa_dict, dataTechParameters,"Data loaded ✅"

@app.callback(
    [
        Output("installed_capa_graph", "figure"),
        Output("produced_energy_graph", "figure"),
        Output("load_factor_graph", "figure"),
        Output("decomp_costs_graph", "figure"),
        Output("storage_capa_graph", "figure"),
        Output('md_pkl_sucess','children')
        ],
    [
        Input('upload_pkl', 'contents')
        ],
    [
        State('techParameters','data'),
        ],)
def loading_pickle_data(content, techParametersData):
    
    if not content:
        # return "No data loaded yet."
        return dash.no_update
                
    techParameters = None 
    if techParametersData:
        # techParameters = pd.read_json(techParametersData, orient='index')        
        techParameters = pd.DataFrame.from_dict(techParametersData)      
        # print(techParameters)  
    
    # decode data
    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)        
    Variables = pickle.load(io.BytesIO(decoded))

    # extract data
    energyCapacity = nrjPlot.extractEnergyCapacity(Variables).round(2).loc[AREAS_ORDER]
    
    # extract cost data
    cost_production = nrjPlot.extractCosts(Variables).round(2)
    cost_storage = pd.pivot_table(Variables["storageCosts"], values="storageCosts", index="AREAS", columns="STOCK_TECHNO")/1e9
    cost_flex = pd.pivot_table(Variables['consumption_power_cost'], values="consumption_power_cost", index="AREAS", columns="FLEX_CONSUM")/1e9

    # plot data
    
    # I
    if techParameters is not None:
        fig_graph1 = nrjPlot.installedCapa_barChart(energyCapacity, techParameters)
    else :
        fig_graph1 = nrjPlot.productionCapa_stackedBarChart(energyCapacity,capaDisp="GW")
    fig_graph2 = nrjPlot.productionCapa_stackedBarChart(energyCapacity,capaDisp="TWh")
    
    # load factor
    fig_graph3 = nrjPlot.loadFactors(energyCapacity)
    
    # system cost
    # fig_graph4 = nrjPlot.costPerCountry(energyCapacity, cost_production, cost_storage, cost_flex)
    fig_graph4 = nrjPlot.costDecomposed_barChart(cost_production, cost_storage, cost_flex)

    # storage and flex capa
    flex_conso = pd.pivot_table(Variables["flex_consumption"], values="flex_consumption", index=["AREAS","Date"], columns="FLEX_CONSUM")*1e-3

    # get storage data
    storageIn = pd.pivot_table(Variables["storageIn"], values="storageIn", index=["AREAS","Date"],columns="STOCK_TECHNO")*1e-3
    storageOut = pd.pivot_table(Variables["storageOut"], values="storageOut", index=["AREAS","Date"],columns="STOCK_TECHNO")*1e-3

    # sum into a single df with negative and positive values (negative : storageIn)
    storage = storageOut - storageIn
    
    fig_graph5 = nrjPlot.installedCapaStoragePower_barChart(storage, flex_conso)
    
    return fig_graph1, fig_graph2, fig_graph3, fig_graph4, fig_graph5, "Data loaded ✅"


if __name__ == '__main__':
    app.run_server(debug=True)
