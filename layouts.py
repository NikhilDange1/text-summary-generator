import dash
import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output


#import callbacks

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__) #external_stylesheets=external_stylesheets)
server = app.server
#app.config.suppress_callback_exceptions = True



app.layout = html.Div(children =
                [html.Div(html.H2('Text Summary Generator'),className='title-div'),
                 html.Div(children=[
                    html.Div(children=[
                        html.Div(html.H4('Paste text below'),className='flex-item1'),
                        html.Div(dcc.Textarea(id='text',value='',placeholder='Type or paste text with atleast 5 sentences',className="child tbox"),className='flex-item2')],className="grid-item grid-item1 inner-grid"),
                    html.Div(children=[
                        html.Div(html.H4('Word Cloud',style={'align':'center'}),className='flex-item1'),
                        html.Div(dcc.Graph(id='cloud',className='child'),className='flex-item2 cloud-div')
                        ],className="grid-item grid-item2 inner-grid"),
                    html.Div(children=[html.Button('Generate Summary',id='summary',className="Button Button-generate_summary",n_clicks_timestamp=0),
                                       ' Number of Summarized sentences ',
                                       dcc.Input(id='summarize_length',type='number',min=3,max=10,step=1,value=5),
                                       '  ',
                                       html.Button('Clear',id='clear',className="Button Button-generate_summary",n_clicks_timestamp=0)],className="grid-item grid-item3"),
                    html.Div(children=[],className="grid-item grid-item4"),
                    html.Div(children=[
                        html.Div(html.H4('Extractive Summary'),className='flex-item1'),
                        html.Div(children=dcc.Textarea(id='preview',value='',className="child tbox"),className='flex-item2')

                    ],className='grid-item grid-item4 inner-grid')

                 ],className="my-container")

            ],className="parent")
