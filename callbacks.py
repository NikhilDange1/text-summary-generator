
from dash.dependencies import Input, Output, State
import dash
from layouts import app
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
from wordcloud import WordCloud
import networkx as nx
from plotly.tools import mpl_to_plotly
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


'''def read_article(file_name):
    file = open(file_name, "r")
    filedata = file.readlines()
    article = filedata[0].split(". ")
    sentences = []

    for sentence in article:
        print(sentence)
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop()

    return sentences'''

server = app.server

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def generate_summary(text, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1 - Read text anc split it
    #sentences =  read_article(file_name)
    sentences = text.split(". ")

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    #print("Indexes of top ranked_sentence order are ", ranked_sentence)

    for i in range(top_n):
      summarize_text.append("".join(ranked_sentence[i][1]))

    # Step 5 - Offcourse, output the summarize texr
    #print("Summarize Text: \n", ". ".join(summarize_text))
    return "\n\n".join(summarize_text)

def plot_cloud(text):
    wordcloud = WordCloud().generate(text)
    fig = px.imshow(wordcloud)
    #fig.update_layout(width=500, height=300)
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    fig.update_layout(autosize=False,margin=go.layout.Margin(l=5,r=5,b=5,t=5,pad=2))
    return fig

@app.callback([Output('preview','value'),
               Output('cloud','figure'),
               Output('summarize_length','max')],
            [Input('summary','n_clicks')],
            [State('text','value'),
             State('summarize_length','value')])
def gen(summary,text,summarize_length):
    if len(text)==0:
        sentences=0
    else:
        sentences=len(text.split('. '))
    if summary is not None:
        text = text.replace('\n',' ')
        if sentences>=3:
            fig = plot_cloud(text)

            return generate_summary(text,summarize_length),fig,sentences
        else:
            return 'Please enter more than 5 sentences',{} ,10
    return '',dash.no_update,10


@app.callback(Output('cloud', 'style'),
              [Input('summary','n_clicks')])
def hide_graph(input):
    if input:
        return {'display':'block'}
    else:
        return {'display':'none'}



if __name__ == '__main__':
    app.run_server(debug=False)
