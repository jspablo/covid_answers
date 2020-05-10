import json
import os

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import requests
from annoy import AnnoyIndex


def load_index(
        v_dim: int,
        metric: str,
        index_path='data/metadata_filter.ann') -> AnnoyIndex:
    index = AnnoyIndex(v_dim, metric)
    abs_path = os.path.join(os.path.dirname(__file__), index_path)
    index.load(abs_path)

    return index


def load_metadata(metadata_path='data/metadata_filter.json') -> dict:
    abs_path = os.path.join(os.path.dirname(__file__), metadata_path)

    with open(abs_path, 'r') as f:
        metadata = json.load(f)

    return metadata


def query_embedding(query: str, url: str) -> list:
    data = {"instances": [{"keras_layer_input": query}]}
    result = requests.post(url, json=data)
    embedding = result.json()['predictions'][0]

    return embedding


def similar_papers_indexes(
        index, vector: list, num_papers: int, search_k: int) -> list:
    similar_papers = index.get_nns_by_vector(
        vector, num_papers, search_k)

    return similar_papers


def get_papers_metadata(similar_papers: list, metadata: dict) -> dict:
    papers_metadata = [metadata[str(paper_id)] for paper_id in similar_papers]

    return papers_metadata


def format_papers_metadata(papers_metadata: dict) -> list:
    cards = []

    for paper_metadata in papers_metadata:
        abstract = paper_metadata.get('abstract')
        max_abstract = 500
        card = dbc.Card(
            dbc.CardBody(
                [
                    dcc.Link(
                        html.H5(
                            paper_metadata.get('title'),
                            className="card-title"),
                        href=paper_metadata.get('url')),
                    html.H6(
                        paper_metadata.get('authors'),
                        className="card-subtitle"),
                    html.Hr(),
                    html.P(
                        (abstract[:max_abstract] + '...')
                        if len(abstract) > max_abstract else abstract),
                ]
            ),
            className="mb-3"
        )

        cards.append(card)

    return cards
