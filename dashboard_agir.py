import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from abc_analysis import abc_analysis, abc_plot
from dash.dependencies import Input, Output

# import from folders/theme changer
from app import *
from dash_bootstrap_templates import ThemeSwitchAIO

#app.py

import dash

FONT_AWESOME = ["https://use.fontawesome.com/releases/v5.10.2/css/all.css"]

app = dash.Dash(__name__, external_stylesheets=FONT_AWESOME)
server = app.server
app.scripts.config.serve_locally = True



# ========== Styles ============ #
tab_card = {'height': '100%'}

main_config = {
    "hovermode": "x unified",
    "legend": {"yanchor":"top", 
                "y":0.9, 
                "xanchor":"left",
                "x":0.1,
                "title": {"text": None},
                "font" :{"color":"white"},
                "bgcolor": "rgba(0,0,0,0.5)"},
    "margin": {"l":10, "r":10, "t":10, "b":10}
}

config_graph={"displayModeBar": False, "showTips": False}

template_theme1 = "flatly"
template_theme2 = "darkly"
url_theme1 = dbc.themes.FLATLY
url_theme2 = dbc.themes.DARKLY


# ===== Reading n cleaning File ====== #
df = pd.read_csv('trimestre1.csv')
df_cru = df.copy()

# Transformando em inteiros e retirando o cifrão R$
df['mes'] = df['mes'].astype(int)
df['ano'] = df['ano'].astype(int)
df['trimestre'] = df['trimestre'].astype(int)

df['valor_total'] = df['valor_total'].str.replace('R\$\s*', '', regex=True)  # Remove currency symbol
df['valor_total'] = df['valor_total'].str.replace('\.', '', regex=True)     # Remove thousands separators
df['valor_total'] = df['valor_total'].str.replace(',', '.', regex=True)     # Replace commas with dots
df['valor_total'] = df['valor_total'].astype(float)

sum_df1 = df.groupby('especie')['valor_total'].sum().reset_index(name='sum_especie_total')
sum_df2 = df.groupby('classe')['valor_total'].sum().reset_index(name='sum_classe_total')
sum_df3 = df.groupby('subclasse')['valor_total'].sum().reset_index(name='sum_subclasse_total')
sum_df4 = df.groupby('item')['valor_total'].sum().reset_index(name='sum_item_total')



df1 = sum_df1.sort_values(by=['sum_especie_total'], ascending=False)
df2 = sum_df2.sort_values(by=['sum_classe_total'], ascending=False)
df3 = sum_df3.sort_values(by=['sum_subclasse_total'], ascending=False)
df4 = sum_df4.sort_values(by=['sum_item_total'], ascending=False)


# Criando opções pros filtros que virão
options_trimestre = [{'label': 'Todos trimestres', 'value': 0}]
for i, j in zip(df_cru['trimestre'].unique(), df['trimestre'].unique()):
    options_trimestre.append({'label': i, 'value': j})
options_trimestre = sorted(options_trimestre, key=lambda x: x['value']) 

options_entidade = [{'label': 'Todas Entidades', 'value': 0}]
for i in df['entidade'].unique():
    options_entidade.append({'label': i, 'value': i})
# ========= Função dos Filtros ========= #
def trimestre_filter(trimestre):
    if trimestre == 0:
        mask = df['trimestre'].isin(df['trimestre'].unique())
    else:
        mask = df['trimestre'].isin([trimestre])
    return mask

def entidade_filter(entidade):
    if entidade == 0:
        mask = df['entidade'].isin(df['entidade'].unique())
    else:
        mask = df['entidade'].isin([entidade])
    return mask

def convert_to_text(trimestre):
    match trimestre:
        case 0:
            x = 'Todos trimestres'
        case 1:
            x = 'Primeiro trimestre de 2023'
        case 2:
            x = 'Segundo trimestre de 2023'
        case 3:
            x = 'Terceiro trimestre de 2023'
        case 4:
            x = '3'
     

    return x


# =========  Layout  =========== #
app.layout = dbc.Container(children=[
    # Armazenamento de dataset
    # dcc.Store(id='dataset', data=df_store),

    # Layout
    # Row 1
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([  
                            html.Legend("Análise de Compras")
                        ], sm=8),
                        dbc.Col([        
                            html.I(className='fa fa-balance-scale', style={'font-size': '300%'})
                        ], sm=4, align="center")
                    ]),
                    dbc.Row([
                        dbc.Col([
                            ThemeSwitchAIO(aio_id="theme", themes=[url_theme1, url_theme2]),
                            html.Legend("AGIR")
                        ])
                    ], style={'margin-top': '10px'}),
                    dbc.Row([
                        dbc.Button("Fonte dos dados", href="https://ecompras.agirsaude.org.br/v/", target="_blank")
                    ], style={'margin-top': '10px'})
                ])
            ], style=tab_card)
        ], sm=4, lg=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row(
                        dbc.Col(
                            html.Legend('Compras por trimestre: (0 = último de 2022)')
                        )
                    ),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='graph1', className='dbc', config=config_graph)
                        ], sm=12, md=7),
                        dbc.Col([
                            dcc.Graph(id='graph2', className='dbc', config=config_graph)
                        ], sm=12, lg=5)
                    ])
                ])
            ], style=tab_card)
        ], sm=12, lg=7),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row(
                        dbc.Col([
                            html.H5('Escolha o trimestre'),
                            dbc.RadioItems(
                                id="radio-trimestre",
                                options=options_trimestre,
                                value=0,
                                inline=True,
                                labelCheckedClassName="text-success",
                                inputCheckedClassName="border border-success bg-success",
                            ),
                            html.Div(id='trimestre-select', style={'text-align': 'center', 'margin-top': '30px'}, className='dbc')
                        ])
                    )
                ])
            ], style=tab_card)
        ], sm=12, lg=3)
    ], className='g-2 my-auto', style={'margin-top': '7px'}),

    # Row 2
    dbc.Row([
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='graph3', className='dbc', config=config_graph)
                        ])
                    ], style=tab_card)
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='graph4', className='dbc', config=config_graph)
                        ])
                    ], style=tab_card)
                ])
            ], className='g-2 my-auto', style={'margin-top': '7px'})
        ], sm=12, lg=5),
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='graph5', className='dbc', config=config_graph)
                        ])
                    ], style=tab_card)
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='graph7', className='dbc', config=config_graph)
                        ])
                    ], style=tab_card)
                ])
            ], className='g-2 my-auto', style={'margin-top': '7px'})
        ], sm=12, lg=5),
        dbc.Col([
            dbc.Card([
                dcc.Graph(id='graph8', className='dbc', config=config_graph)
            ], style=tab_card)
        ], sm=20, lg=2)
    ], className='g-2 my-auto', style={'margin-top': '7px'}),
    
    # Row 3
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4('6 items de maior gasto'),
                    dcc.Graph(id='graph9', className='dbc', config=config_graph)
                ])
            ], style=tab_card)
        ], sm=10, lg=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("6 subclasses de maior gasto"),
                    dcc.Graph(id='graph10', className='dbc', config=config_graph)
                ])
            ], style=tab_card)
        ], sm=10, lg=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='graph11', className='dbc', config=config_graph)
                ])
            ], style=tab_card)
        ], sm=10, lg=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5('Escolha a Entidade'),
                    dbc.RadioItems(
                        id="radio-entidade",
                        options=options_entidade,
                        value=0,
                        inline=True,
                        labelCheckedClassName="text-warning",
                        inputCheckedClassName="border border-warning bg-warning",
                    ),
                    html.Div(id='entidade-select', style={'text-align': 'center', 'margin-top': '30px'}, className='dbc')
                ])
            ], style=tab_card)
        ], sm=10, lg=2),
    ], className='g-2 my-auto', style={'margin-top': '7px'})
], fluid=True, style={'height': '100vh'})


# ======== Callbacks ========== #
# Graph 1 and 2
@app.callback(
    Output('graph1', 'figure'),
    Output('graph2', 'figure'),
    Output('trimestre-select', 'children'),
    Input('radio-trimestre', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def graph1(trimestre, toggle):
    template = template_theme1 if toggle else template_theme2

    mask = trimestre_filter(trimestre)
    df_1 = df.loc[mask]

    df_1 = df_1.groupby(['trimestre'])['valor_total'].sum().reset_index()

    fig1 = go.Figure(go.Bar(x=df_1['trimestre'], y=df_1['valor_total'], textposition='auto', text=df_1['valor_total']))
    fig2 = go.Figure(go.Pie(labels=df_1['trimestre'], values=df_1['valor_total'], hole=.6))
    fig1.update_layout(main_config, height=200, template=template)
    fig2.update_layout(main_config, height=200, template=template, showlegend=False)

    select = html.H1(convert_to_text(trimestre))

    return fig1, fig2, select


# Graph 3
@app.callback(
    Output('graph3', 'figure'),
    Input('radio-entidade', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def graph3(entidade, toggle):
    template = template_theme1 if toggle else template_theme2

    mask = entidade_filter(entidade)
    df_3 = df.loc[mask]

    df_3 = df_3.groupby('especie')['valor_total'].sum().reset_index(name='sum_especie_total')

    # Sort the dataframe by 'sum_especie_total' in descending order
    df_3 = df_3.sort_values(by='sum_especie_total', ascending=False)

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=df_3['especie'], y=df_3['sum_especie_total'], name='Valor Total'))
    fig3.add_trace(go.Scatter(x=df_3['especie'], y=df_3['sum_especie_total'].cumsum(), mode='lines', name='Cumulative Total'))
    fig3.update_layout(
        main_config,
        height=270,
        template=template,
        xaxis_title='ESPECIE',
        yaxis_title='Valor Total',
        xaxis_tickangle=-45,  # Adjust the angle of X-axis labels
        margin=dict(l=50, r=10, t=10, b=50)  # Adjust the margins around the graph area
    )
    return fig3

# Graph 4
@app.callback(
    Output('graph4', 'figure'),
    Input('radio-entidade', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def graph4(entidade, toggle):
    template = template_theme1 if toggle else template_theme2
    
    mask = entidade_filter(entidade)
    df_4 = df.loc[mask]

    df_4 = df_4.groupby('classe')['valor_total'].sum().reset_index(name='sum_classe_total')

    # Sort the dataframe by 'sum_classe_total' in descending order
    df_4 = df_4.sort_values(by='sum_classe_total', ascending=False)

    fig4 = go.Figure()
    fig4.add_trace(go.Bar(x=df_4['classe'], y=df_4['sum_classe_total'], name='Valor Total'))
    fig4.add_trace(go.Scatter(x=df_4['classe'], y=df_4['sum_classe_total'].cumsum(), mode='lines', name='Cumulative Total'))
    fig4.update_layout(
        main_config,
        height=270,
        template=template,
        xaxis_title='CLASSE',
        yaxis_title='Valor Total',
        xaxis_tickangle=-45,  # Adjust the angle of X-axis labels
        margin=dict(l=50, r=10, t=10, b=50)  # Adjust the margins around the graph area
    )
    return fig4

#  Graph 5 
@app.callback(
    Output('graph5', 'figure'),
    Input('radio-entidade', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def graph5(entidade, toggle):
    template = template_theme1 if toggle else template_theme2
    
    mask = entidade_filter(entidade)
    df_5 = df.loc[mask]

    df_5 = df_5.groupby('subclasse')['valor_total'].sum().reset_index(name='sum_subclasse_total')

    # Sort the dataframe by 'sum_subclasse_total' in descending order
    df_5 = df_5.sort_values(by='sum_subclasse_total', ascending=False)

    fig5 = go.Figure()
    fig5.add_trace(go.Bar(x=df_5['subclasse'], y=df_5['sum_subclasse_total'], name='Valor Total'))
    fig5.add_trace(go.Scatter(x=df_5['subclasse'], y=df_5['sum_subclasse_total'].cumsum(), mode='lines', name='Cumulative Total'))
    fig5.update_layout(
        main_config,
        height=270,
        template=template,
        xaxis_title='SUBCLASSE',
        yaxis_title='Valor Total',
        xaxis_tickangle=-45,  # Adjust the angle of X-axis labels
        margin=dict(l=50, r=10, t=10, b=50)  # Adjust the margins around the graph area
    )
    return fig5

# Graph 7
@app.callback(
    Output('graph7', 'figure'),
    Input('radio-entidade', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def graph7(entidade, toggle):
    template = template_theme1 if toggle else template_theme2
    
    mask = entidade_filter(entidade)
    df_7 = df.loc[mask]

    df_7 = df_7.groupby('item')['valor_total'].sum().reset_index(name='sum_item_total')

    # Sort the dataframe by 'sum_item_total' in descending order
    df_7 = df_7.sort_values(by='sum_item_total', ascending=False)

    fig7 = go.Figure()
    fig7.add_trace(go.Bar(x=df_7['item'], y=df_7['sum_item_total'], name='Valor Total'))
    fig7.add_trace(go.Scatter(x=df_7['item'], y=df_7['sum_item_total'].cumsum(), mode='lines', name='Cumulative Total'))
    fig7.update_layout(
        main_config,
        height=270,
        template=template,
        xaxis_title='ITEM',
        yaxis_title='Valor Total',
        xaxis_tickangle=-45,  # Adjust the angle of X-axis labels
        margin=dict(l=50, r=10, t=10, b=50)  # Adjust the margins around the graph area
    )
    return fig7

# Graph 8
@app.callback(
    Output('graph8', 'figure'),
    Input('radio-trimestre', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def graph8(trimestre, toggle):
    template = template_theme1 if toggle else template_theme2

    mask = trimestre_filter(trimestre)
    df_8 = df.loc[mask]

    df_8 = df_8.groupby('entidade')['valor_total'].sum().reset_index()
    fig8 = go.Figure(go.Bar(
        x=df_8['entidade'],  # Swap x and y axes
        y=df_8['valor_total'],  # Swap x and y axes
        orientation='v',  # Set orientation to 'v' for vertical bars
        textposition='auto',
        text=df_8['valor_total'],
        insidetextfont=dict(family='Times', size=15)))

    fig8.update_layout(main_config, height=360, template=template)
    return fig8


# Graph 9
@app.callback(
    Output('graph9', 'figure'),
    Input('radio-trimestre', 'value'),
    Input('radio-entidade', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def graph9(trimestre, entidade, toggle):
    template = template_theme1 if toggle else template_theme2

    mask = trimestre_filter(trimestre)
    df_9 = df.loc[mask]

    mask = entidade_filter(entidade)
    df_9 = df_9.loc[mask]

    df_9 = df_9.groupby('item')['valor_total'].sum().reset_index()

    # Get the top 5 items
    df9_top5 = df_9.sort_values(by='valor_total', ascending=False).head(6)
    top5_items = df9_top5['item'].tolist()

    # Filter the original DataFrame for the top 5 items
    df_9 = df[df['item'].isin(top5_items)]

    fig9 = px.bar(df_9, y="valor_total", x="trimestre", color="item")  # Using Plotly Express

    fig9.update_layout(main_config, height=250, template=template)

    # Update the legend position and display mode
    fig9.update_layout(
        legend=dict(orientation='h', yanchor='bottom', y=-1.82, xanchor='left', x=0),
        showlegend=True,
    )

    return fig9




# Graph 10
@app.callback(
    Output('graph10', 'figure'),
    Input('radio-trimestre', 'value'),
    Input('radio-entidade', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def graph10(trimestre, entidade, toggle):
    template = template_theme1 if toggle else template_theme2

    mask = trimestre_filter(trimestre)
    df_10 = df.loc[mask]

    mask = entidade_filter(entidade)
    df_10 = df_10.loc[mask]

    df_10 = df_10.groupby('subclasse')['valor_total'].sum().reset_index()

    # Get the top 6 items
    df10_top5 = df_10.sort_values(by='valor_total', ascending=False).head(6)
    top5_items = df10_top5['subclasse'].tolist()

    # Filter the original DataFrame for the top 6 items
    df_10 = df[df['subclasse'].isin(top5_items)]

    fig10 = px.bar(df_10, y="valor_total", x="trimestre", color="subclasse")  # Using Plotly Express

    fig10.update_layout(main_config, height=250, template=template)

    # Update the legend position and display mode
    fig10.update_layout(
        legend=dict(orientation='h', yanchor='bottom', y=-1.82, xanchor='left', x=0),
        showlegend=True,
    )

    return fig10

# Graph 11
@app.callback(
    Output('graph11', 'figure'),
    Output('entidade-select', 'children'),
    Input('radio-trimestre', 'value'),
    Input('radio-entidade', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def graph11(trimestre, entidade, toggle):
    template = template_theme1 if toggle else template_theme2

    mask = trimestre_filter(trimestre)
    df_11 = df.loc[mask]

    mask = entidade_filter(entidade)
    df_11 = df_11.loc[mask]

    df11 = df_11.groupby(['valor_total']).sum().reset_index()
    fig11 = go.Figure()
    fig11.add_trace(go.Indicator(mode='number',
        title = {"text": f"<span style='font-size:150%'>Valor Total</span><br><span style='font-size:70%'>Em Reais</span><br>"},
        value = df_11['valor_total'].sum(),  # Corrected line: use df_11 instead of df
        number = {'prefix': "R$"}
    ))

    fig11.update_layout(main_config, height=300, template=template)
    select = html.H1("Todas Entidades") if entidade == 0 else html.H1(entidade)

    return fig11, select

# Run server
if __name__ == '__main__':
    app.run_server(debug=False)
