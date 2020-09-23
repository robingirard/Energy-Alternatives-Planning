import plotly.graph_objects as go
import plotly

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

def MyStackedPlotly(x_df,y_df,Names):
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

    fig.update_xaxes(rangeslider_visible=True)
    return(fig)

def AppendMyStackedPlotly(fig,x_df,y_df,Names):
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

    fig.update_xaxes(rangeslider_visible=True)
    return(fig)

def MyAreaStackedPlot(df,Selected_TECHNOLOGIES=-1,AREA_name="AREAS",TechName='TECHNOLOGIES'):
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