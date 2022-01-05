import plotly.graph_objects as go
import plotly
import pandas as pd
import numpy as np
def extractCosts(Variables):
    if "AREAS" in Variables['energy'].columns:
        if 'capacityCosts' in Variables.keys():
            df = Variables['capacityCosts'].set_index(["AREAS", "TECHNOLOGIES"]) / 10 ** 9;
            df = df.merge(pd.DataFrame(Variables['energyCosts'].set_index(["AREAS", "TECHNOLOGIES"]) / 10 ** 9),
                          left_on=["AREAS", "TECHNOLOGIES"], right_on=["AREAS", "TECHNOLOGIES"])
            df.columns = ["Capacity_Milliards_euros", "Energy_Milliards_euros"]
        else:
            df = pd.DataFrame(Variables['energyCosts'].set_index(["AREAS", "TECHNOLOGIES"]) / 10 ** 9)
            df.columns = ["Energy_Milliards_euros"]

    else:
        if 'capacityCosts' in Variables.keys():
            df = Variables['capacityCosts'].set_index("TECHNOLOGIES") / 10 ** 9;
            df = df.merge(pd.DataFrame(Variables['energyCosts'].set_index("TECHNOLOGIES") / 10 ** 9),
                          left_on="TECHNOLOGIES", right_on="TECHNOLOGIES")
            df.columns = ["Capacity_Milliards_euros", "Energy_Milliards_euros"]
        else :
            df = pd.DataFrame(Variables['energyCosts'].set_index("TECHNOLOGIES") / 10 ** 9)
            df.columns = ["Energy_Milliards_euros"]
    return df;

def extractEnergyCapacity(Variables) :
    if "AREAS" in Variables['energy'].columns:
        production_df = EnergyAndExchange2Prod(Variables)
        EnergyCapacity_df = Variables['capacity'].set_index(["AREAS","TECHNOLOGIES"]) / 10 ** 3;
        EnergyCapacity_df = EnergyCapacity_df.merge(pd.DataFrame(Variables['energy'].groupby(by=["AREAS","TECHNOLOGIES"]).sum() / 10 ** 6),
                                                    left_on=["AREAS","TECHNOLOGIES"], right_on=["AREAS","TECHNOLOGIES"])
        EnergyCapacity_df.columns = ["Capacity_GW", "Production_TWh"]
    else:
        production_df = Variables['energy'].pivot(index="Date", columns='TECHNOLOGIES', values='energy')
        EnergyCapacity_df = Variables['capacity'].set_index("TECHNOLOGIES") / 10 ** 3;
        EnergyCapacity_df = EnergyCapacity_df.merge(pd.DataFrame(production_df.sum(axis=0) / 10 ** 6),
                                                    left_on="TECHNOLOGIES", right_on="TECHNOLOGIES")
        EnergyCapacity_df.columns = ["Capacity_GW", "Production_TWh"]
    return(EnergyCapacity_df)

def expand_grid(x, y,names):
    res=pd.DataFrame()
    xG, yG = np.meshgrid(x, y) # create the actual grid
    res.loc[:,names[0]] = xG.flatten() # make the grid 1d
    res.loc[:,names[1]] = yG.flatten() # same
    return res # return a dataframe



def MyPlotly(x_df,y_df,Names="",fill=True):
    '''
    :param x:
    :param y:
    :param Names:
    :return:
    '''
    if Names=="" : Names=y_df.columns.values.tolist()
    fig = go.Figure()
    i=0
    for col in y_df.columns:
        if i==0:
            if fill :
                fig.add_trace(go.Scatter(x=x_df, y=y_df[col] , fill='tozeroy',
                             mode='none' ,name=Names[i])) # fill down to xaxis
            else :
                fig.add_trace(go.Scatter(x=x_df, y=y_df[col],mode='lines', name=Names[i]))  # fill down to xaxis
            colNames=[col]
        else:
            colNames.append(col)
            if fill :
                fig.add_trace(go.Scatter(x=x_df, y=y_df[col], fill='tozeroy',
                                     mode='none', name=Names[i]))  # fill to trace0 y
            else :
                fig.add_trace(go.Scatter(x=x_df, y=y_df[col],
                                     mode='lines', name=Names[i]))  # fill to trace0 y
        i=i+1

    fig.update_xaxes(rangeslider_visible=True)
    return(fig)

def MyStackedPlotly(y_df, Conso=-1,isModifyOrder=True,Names=-1):
    '''
    :param x: 
    :param y: 
    :param Names:
    :return: 
    '''

    if isModifyOrder: y_df=ModifyOrder_df(y_df) ### set Nuke first column
    if (Names.__class__ == int): Names=y_df.columns.unique().tolist()
    x_df=y_df.index
    fig = go.Figure()
    i = 0
    for col in y_df.columns:
        if i == 0:
            fig.add_trace(go.Scatter(x=x_df, y=y_df[col], fill='tozeroy',
                                     mode='none', name=Names[i]))  # fill down to xaxis
            colNames = [col]
        else:
            colNames.append(col)
            fig.add_trace(go.Scatter(x=x_df, y=y_df.loc[:, y_df.columns.isin(colNames)].sum(axis=1), fill='tonexty',
                                     mode='none', name=Names[i]))  # fill to trace0 y
        i = i + 1

    if (Conso.__class__ != int):
        fig.add_trace(go.Scatter(x=Conso.index,
                                 y=Conso["areaConsumption"], name="Conso",
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
                             y=Conso["areaConsumption"], name="Conso",
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


def EnergyAndExchange2Prod(Variables,EnergyName='energy',exchangeName='Exchange'):
    Variables["exchange"].columns = ['AREAS1', 'AREAS2', 'Date', 'exchange']
    AREAS = Variables['energy'].AREAS.unique()
    production_df = Variables['energy'].pivot(index=["AREAS","Date"], columns='TECHNOLOGIES', values='energy')
    ToAREA=[]
    for AREA in AREAS:
        ToAREA.append(Variables["exchange"].\
            loc[(Variables["exchange"].AREAS2 == AREA), ["Date", "exchange",
                                                                                              "AREAS1",
                                                                                              "AREAS2"]].\
            rename(columns={"AREAS2": "AREAS"}).\
            pivot(index=["AREAS","Date"], columns='AREAS1', values='exchange'))
    ToAREA_pd=pd.concat(ToAREA)
    production_df = production_df.merge(ToAREA_pd, how='inner', left_on=["AREAS","Date"], right_on=["AREAS","Date"])
    #exchange analysis
    return(production_df);

def MyAreaStackedPlot_tidy(df,Selected_TECHNOLOGIES=-1,AREA_name="AREAS",TechName='TECHNOLOGIES'):
    if (Selected_TECHNOLOGIES==-1):
        Selected_TECHNOLOGIES=df[TechName].unique().tolist()
    AREAS=df[AREA_name].unique().tolist()

    visible={}
    for AREA in AREAS: visible[AREA] = []
    for AREA in AREAS:
        for AREA2 in AREAS:
            if AREA2==AREA:
                for TECH in Selected_TECHNOLOGIES:
                    visible[AREA2].append(True)
            else :
                for TECH in Selected_TECHNOLOGIES:
                    visible[AREA2].append(False)

    fig = go.Figure()
    dicts=[]
    for AREA in AREAS:
        production_df = df[df[AREA_name] == AREA].pivot(index="Date",columns='TECHNOLOGIES',values='energy')
        fig = AppendMyStackedPlotly(fig,x_df=production_df.index,
                            y_df=production_df[list(Selected_TECHNOLOGIES)],
                            Names=list(Selected_TECHNOLOGIES))
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

    return(fig)

def ModifyOrder(Names):
    if "OldNuke" in Names:
        Names.remove("OldNuke")
        Names.insert(0, "OldNuke")
    if "NewNuke" in Names:
        Names.remove("NewNuke")
        Names.insert(0, "NewNuke")
    if "NukeCarrene" in Names:
        Names.remove("NukeCarrene")
        Names.insert(0, "NukeCarrene")

    return(Names)

def ModifyOrder_df(df):
    if "OldNuke" in df.columns:
        Nuke=df.pop("OldNuke")
        df.insert(0, "OldNuke", Nuke)
    if "NewNuke" in df.columns:
        Nuke=df.pop("NewNuke")
        df.insert(0, "NewNuke", Nuke)
    if "NukeCarrene" in df.columns:
        Nuke=df.pop("NukeCarrene")
        df.insert(0, "NukeCarrene", Nuke)
    return(df);

def MyAreaStackedPlot(df_,Conso=-1,Selected_TECHNOLOGIES=-1,AREA_name="AREAS"):
    df=df_.copy()
    #df.reset_index(inplace=True)
    if (Selected_TECHNOLOGIES.__class__ == int):
        Selected_TECHNOLOGIES=df.columns.unique().tolist()
    AREAS=df.index.get_level_values('AREAS').unique().tolist()
    Selected_TECHNOLOGIES=ModifyOrder(Selected_TECHNOLOGIES)
    df=ModifyOrder_df(df)


    visible={}
    for AREA in AREAS: visible[AREA] = []
    for AREA in AREAS:
        for AREA2 in AREAS:
            if AREA2==AREA:
                for TECH in Selected_TECHNOLOGIES:
                    visible[AREA2].append(True)
                visible[AREA2].append(True)
                visible[AREA2].append(True)
                if 'Storage' in Conso.columns : visible[AREA2].append(True)
            else :
                for TECH in Selected_TECHNOLOGIES:
                    visible[AREA2].append(False)
                visible[AREA2].append(False)
                visible[AREA2].append(False)
                if 'Storage' in Conso.columns: visible[AREA2].append(False)
    fig = go.Figure()
    dicts=[]
    for AREA in AREAS:
        production_df_ = df.loc[(AREA,slice(None)),:]#.reset_index()
        Conso_=Conso.loc[(AREA,slice(None)),:];
        Conso_ = Conso.loc[(AREA,slice(None)),:].reset_index().set_index("Date").drop(["AREAS"], axis=1);
        production_df_ = df.loc[(AREA,slice(None)),:].reset_index().set_index("Date").drop(["AREAS"], axis=1);
        #Conso_.reset_index(inplace=True)
        Conso_.loc[:,"ConsoImportExport"] = Conso_.loc[:,"areaConsumption"] - production_df_.loc[:,AREAS].sum(axis=1)

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



def plotDecomposedConso(x_df,y_df, Tofile=False, TimeName='Date'):
    '''
    Function for graphical representation of a consumption decomposed with thermal
    :param x:
    :param y:
    :return:
    '''

    fig=MyStackedPlotly(x_df=x_df,y_df=y_df,Names=y.columns.to_list())
    fig.update_layout(title_text="Consommation (MWh)", xaxis_title="Date")
    if Tofile: plotly.offline.plot(fig, filename='file.html')
    else: fig.show()

def plotDecomposedConso(dataYear_decomposed_df, Tofile=False, TimeName='Date'):
    '''
    Function for graphical representation of a consumption decomposed with thermal
    :param dataYear_decomposed:
    :param Tofile:
    :param TimeName:
    :return:
    '''

    fig=MyStackedPlotly(x_df=dataYear_decomposed_df[TimeName],y_df=dataYear_decomposed_df[["NTS_C","TS_C"]],
                        Names=['Conso non thermosensible','conso thermosensible'])
    fig.update_layout(title_text="Consommation (MWh)", xaxis_title="Date")
    if Tofile: plotly.offline.plot(fig, filename='file.html')
    else: fig.show()


def plotProd(dataYear_df,prodNames, Tofile=False, TimeName='Date'):
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
    fig.update_layout(title_text="Consommation (MWh)", xaxis_title="Date")
    if Tofile: plotly.offline.plot(fig, filename='file.html')
    else: fig.show()