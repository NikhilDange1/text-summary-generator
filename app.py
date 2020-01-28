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
                [html.H2('Text Summarizer'),
                 html.Div(children=[
                    html.Div(children=[dcc.Textarea(id='text',value='',placeholder='Enter or paste text',className='child')],className="grid-item grid-item1"),
                    html.Div(children=[dcc.Graph(id='cloud',className="child")],className="grid-item grid-item2"),
                    html.Div(children=[html.Button('Generate Summary',id='summary',className="Button Button-generate_summary")],className="grid-item grid-item3"),
                    html.Div(children=[dcc.Slider(id='summarize_length',min=3,max=7,step=1,marks={i: '{}'.format(i) for i in range(3,8)},value=5,className='child')],className="grid-item grid-item4"),
                    html.Div(children=[
                        html.Div(children=[html.H4('Extractive Summary'),
                        dcc.Textarea(id='preview',value='',className='child')],className="flex")

                    ])

                 ],className="child my-container")

            ],className="parent")
