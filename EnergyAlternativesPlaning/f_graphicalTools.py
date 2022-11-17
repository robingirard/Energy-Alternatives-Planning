import plotly.graph_objects as go
import plotly
import pandas as pd
import numpy as np
from mycolorpy import colorlist as mcp

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




def MyStackedPlotly(y_df, Conso=-1,isModifyOrder=True,Names=-1,color_dict=None):
    '''
    :param x: 
    :param y: 
    :param Names:
    :return: 
    '''

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



def marimekko(df,x_var_name,y_var_name,effectif_var_name,color_discrete_sequence):
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

