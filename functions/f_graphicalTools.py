import plotly.graph_objects as go
import plotly
import pandas as pd

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

def MyStackedPlotly(x_df,y_df, Conso,Names):
    '''
    :param x: 
    :param y: 
    :param Names:
    :return: 
    '''
    fig = go.Figure()
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

    fig.add_trace(go.Scatter(x=Conso['TIMESTAMP'],
                             y=Conso["areaConsumption"], name="Conso",
                             line=dict(color='red', width=0.4)))  # fill down to xaxis
    if "NewConsumption" in Conso.keys():
        fig.add_trace(go.Scatter(x=Conso['TIMESTAMP'],
                                 y=Conso["NewConsumption"], name="Conso+stockage",
                                 line=dict(color='black', width=0.4)))  # fill down to xaxis
    fig.update_xaxes(rangeslider_visible=True)
    return(fig)

def AppendMyStackedPlotly(fig,x_df,y_df,Conso,Names):
    '''
    :param x:
    :param y:
    :param Names:
    :return:
    '''

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
    fig.add_trace(go.Scatter(x=Conso['TIMESTAMP'],
                             y=Conso["areaConsumption"], name="Conso",
                             line=dict(color='red', width=0.4)))  # fill down to xaxis
    if "NewConsumption" in Conso.keys():
        fig.add_trace(go.Scatter(x=Conso['TIMESTAMP'],
                                 y=Conso["NewConsumption"], name="Conso+stockage",
                                 line=dict(color='black', width=0.4)))  # fill down to xaxis
    if "ConsoImportExport" in Conso.keys():
        fig.add_trace(go.Scatter(x=Conso['TIMESTAMP'],
                                 y=Conso["ConsoImportExport"], name="Conso+export-import",
                                 line=dict(color='blue', width=0.4)))  # fill down to xaxis
    fig.update_xaxes(rangeslider_visible=True)
    return(fig)


def EnergyAndExchange2Prod(Variables,EnergyName='energy',exchangeName='Exchange'):
    Variables["exchange"].columns = ['AREAS1', 'AREAS2', 'TIMESTAMP', 'exchange']
    AREAS = Variables['energy'].AREAS.unique()
    production_df = Variables['energy'].pivot(index=["TIMESTAMP", "AREAS"], columns='TECHNOLOGIES', values='energy')
    ToAREA=[]
    for AREA in AREAS:
        ToAREA.append(Variables["exchange"].\
            loc[(Variables["exchange"].AREAS2 == AREA), ["TIMESTAMP", "exchange",
                                                                                              "AREAS1",
                                                                                              "AREAS2"]].\
            rename(columns={"AREAS2": "AREAS"}).\
            pivot(index=["TIMESTAMP", "AREAS"], columns='AREAS1', values='exchange'))
    ToAREA_pd=pd.concat(ToAREA)
    production_df = production_df.merge(ToAREA_pd, how='inner', left_on=["TIMESTAMP","AREAS"], right_on=["TIMESTAMP","AREAS"])
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
        production_df = df[df[AREA_name] == AREA].pivot(index="TIMESTAMP",columns='TECHNOLOGIES',values='energy')
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


def MyAreaStackedPlot(df_,Conso=-1,Selected_TECHNOLOGIES=-1,AREA_name="AREAS"):
    df=df_.copy()
    #df.reset_index(inplace=True)
    if (Selected_TECHNOLOGIES==-1):
        Selected_TECHNOLOGIES=df.columns.unique().tolist()
    AREAS=df.index.get_level_values('AREAS').unique().tolist()
    if "OldNuke" in Selected_TECHNOLOGIES:
        Selected_TECHNOLOGIES.remove("OldNuke")
        Selected_TECHNOLOGIES.insert(0, "OldNuke")
        Nuke=df.pop("OldNuke")
        df.insert(0, "OldNuke", Nuke)

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
        production_df_ = df.loc[(slice(None),AREA),:]#.reset_index()
        Conso_=Conso.loc[(slice(None),AREA),:];
        Conso_ = Conso_.reset_index().set_index("TIMESTAMP").drop(["AREAS"], axis=1).reset_index();
        production_df_ = production_df_.reset_index().set_index("TIMESTAMP").drop(["AREAS"], axis=1).reset_index();
        #Conso_.reset_index(inplace=True)
        Conso_.loc[:,"ConsoImportExport"] = Conso_.loc[:,"areaConsumption"] - production_df_.loc[:,AREAS].sum(axis=1)

        fig = AppendMyStackedPlotly(fig,x_df=production_df_.TIMESTAMP,
                            y_df=production_df_.reset_index()[Selected_TECHNOLOGIES],
                            Conso=Conso_,
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