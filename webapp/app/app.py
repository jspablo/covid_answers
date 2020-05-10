import json
import os

import dash
import dash_bootstrap_components as dbc
import requests
from dash.dependencies import Input, Output, State
from tqdm import tqdm

from app.layout import LAYOUT
from app.utils import (load_index, load_metadata, query_embedding,
                       similar_papers_indexes, get_papers_metadata,
                       format_papers_metadata)


num_similar_papers = int(os.environ.get('NUM_SIMILAR_PAPERS'))
embedding_model = os.environ.get('EMBEDDING_MODEL')
v_dim = int(os.environ.get('V_DIM'))
metric = os.environ.get('METRIC')
valid_sources = os.environ.get('VALID_SOURCES').split('-')
search_k = int(os.environ.get('SEARCH_K'))

annoy_index = load_index(int(v_dim), metric)
metadata = load_metadata()


app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.FLATLY,
        "https://use.fontawesome.com/releases/v5.7.2/css/all.css"
        ],
    meta_tags=[
        {'name': 'viewport',
         'content': 'width=device-width, initial-scale=1'}])

app.title = 'covid19 answers'
app.layout = LAYOUT

server = app.server


@app.callback(
    Output('results', 'children'),
    [Input('query-button', 'n_clicks')],
    [State('query-text', 'value')])
def display_answers(n_clicks, query_text):
    if query_text:
        embedding = query_embedding(query_text, embedding_model)

        similar_papers = similar_papers_indexes(
            annoy_index, embedding, num_similar_papers, search_k)

        papers_metadata = get_papers_metadata(similar_papers, metadata)

        result = format_papers_metadata(papers_metadata)

        return result
