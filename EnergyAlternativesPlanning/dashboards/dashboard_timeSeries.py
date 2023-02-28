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

AREA_CODENAME = {"Belgium" : 'BE', "Switzerland" : 'CH', "Germany" : 'DE', "Spain" : 'ES', "France" : 'FR', "Great Britain"  : 'GB', "Italy" : 'IT'}

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
    dcc.Store(id='prodConso_data'),
    dcc.Store(id='xaxis_range'),
        
    dcc.Markdown(children=f"""
# Energy Alternatives Planing - Results presentation dashboard

## Loading data

To begin with visualization, upload the two needed files : the input file, under the xlsx format, and the output file, under the pickle format.

"""),
    
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
    
    # Plots
    
    html.H2("Production time series"),
    
    dcc.Dropdown(
        id="area_choice",
        options=[
            {"label": region, "value": region}
            for region in ["France", "Germany","Great Britain","Spain","Italy","Belgium","Switzerland"]
        ],

        value="France",
        clearable=False,
        className="dropdown",
    ),
    dcc.Markdown(id='div_timeseries_plot'),
    
    dcc.Graph(id="timeSerie_graph"),

])

# display filenames
@app.callback(
    Output('div_pkl','children'),
    Input('upload_pkl', 'filename')
)
def display_xls_filename(filename):
    return  f"Loaded file : **{filename}**"

# display chosen area
@app.callback(
    Output('div_timeseries_plot','children'),
    Input('area_choice', 'value')
)
def display_chosen_area(area):
    return  f"Chosen area : **{AREA_CODENAME[area]}**"


# Loading production and consumption time series from pickle file

@app.callback(
    [
        Output('prodConso_data', "data"),
        Output('md_pkl_sucess','children')
    ],
    Input('upload_pkl', 'contents'),
)
def loading_pickle_data(content):
    
    if not content:
        return dash.no_update
                
    # decode data
    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)        
    Variables = pickle.load(io.BytesIO(decoded))
    
    # consumption
    consumption = pd.pivot_table(Variables["total_consumption"], values="total_consumption", index=["AREAS","Date"]).rename(columns={"total_consumption":"areaConsumption"})*1e-3
    
    # flexible consumption
    flex_conso = pd.pivot_table(Variables["flex_consumption"], values="flex_consumption", index=["AREAS","Date"], columns="FLEX_CONSUM")*1e-3
    
    # production
    production = pd.pivot_table(Variables["energy"], values="energy", index=["AREAS","Date"], columns="TECHNOLOGIES")*1e-3
    
    # print(Variables["exchange"])
    
    exchanges = pd.pivot_table(Variables["exchange"].rename(columns={"TIMESTAMP":"Date"}), values="exchange", index=["AREAS","Date"],columns="AREAS1")*1e-3
    
    # get total exchanges for each area
    exchanges_sumed = exchanges.sum(axis=1).rename("exchanges")
    
    # get storage data
    storageIn = pd.pivot_table(Variables["storageIn"], values="storageIn", index=["AREAS","Date"],columns="STOCK_TECHNO")*1e-3
    storageOut = pd.pivot_table(Variables["storageOut"], values="storageOut", index=["AREAS","Date"],columns="STOCK_TECHNO")*1e-3
    
    # sum into a single df with negative and positive values (negative : storageIn)
    storage = storageOut - storageIn

    # concat with production
    prod_exchanges = pd.concat([production, -exchanges_sumed], axis=1)

    # concat with storage
    prod_ex_storage = pd.concat([prod_exchanges, storage], axis=1)

    prodConso_data = {
        "production" : prod_ex_storage.reset_index().to_dict('records'),
        "consumption" : consumption.reset_index().to_dict('records'),
        "flex_conso" : flex_conso.reset_index().to_dict('records')
    }
        
    return prodConso_data, "Data loaded ✅"

# get axis range
@app.callback(
    Output('xaxis_range', 'data'),
    Input('timeSerie_graph', 'relayoutData'))
def display_relayout_data(relayoutData):
    return  relayoutData

# Plot the time serie

@app.callback(
    Output("timeSerie_graph", "figure"),
    Input('area_choice', 'value'),
    [
        State('prodConso_data','data'),
        State('xaxis_range','data')
    ]
)
def loading_pickle_data(area, prodConso_data, xaxis_range):
    
    if not prodConso_data:
        return dash.no_update
    
    area = AREA_CODENAME[area]
    
    # loading data from state
    production = pd.DataFrame.from_dict(prodConso_data["production"])
    production['Date'] = pd.to_datetime(production['Date'])
    production.set_index(["AREAS", "Date"], inplace=True)
    
    consumption = pd.DataFrame.from_dict(prodConso_data["consumption"])
    consumption['Date'] = pd.to_datetime(consumption['Date'])
    consumption.set_index(["AREAS", "Date"], inplace=True)
    
    flex_conso = pd.DataFrame.from_dict(prodConso_data["flex_conso"])
    flex_conso['Date'] = pd.to_datetime(flex_conso['Date'])
    flex_conso.set_index(["AREAS", "Date"], inplace=True)
        
    fig = nrjPlot.plotProduction(
        production.loc[area], 
        conso=consumption.loc[area], 
        flex_conso=flex_conso.loc[area], title=f"{area} production and consumption - 2050", 
        yaxis_title="Power (GW)"
        )
    
    # if figure already exists : get range info
    if xaxis_range is not None:
        if 'xaxis.range' in xaxis_range.keys():
            fig['layout']['xaxis'].update(range=xaxis_range['xaxis.range'])
    
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
