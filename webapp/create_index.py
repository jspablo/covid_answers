import json
import os

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from annoy import AnnoyIndex
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize
from tqdm import tqdm


def embedding(abstract):
    sentences = sent_tokenize(abstract)
    embeddings = embed(sentences)
    e = tf.reduce_mean(embeddings, axis=0)

    return tf.squeeze(e)


if __name__ == '__main__':
    load_dotenv()

    metadata_idx = os.environ.get('METADATA_INDEX')
    metadata_filter = os.environ.get('METADATA_FILENAME')

    # Reading and filtering data
    metadata = pd.read_csv(os.environ.get('METADATA_PATH'))
    columns_filter = os.environ.get('COLUMNS_FILTER').split('-')
    metadata_filter = metadata[columns_filter].dropna()

    assert metadata_filter.shape[0] == metadata_filter[metadata_idx].nunique()

    # Saving dataframe as JSON in order to be easily read by the app
    metadata_filter[metadata_idx] = metadata_filter[metadata_idx].astype(int)
    metadata_filter_json = metadata_filter.to_dict(orient='index')

    with open(os.environ.get('METADATA_FILENAME') + '.json', 'w') as f:
        json.dump(metadata_filter_json, f, sort_keys=True)

    # Creating embedding and index
    embed = hub.load(os.environ.get('EMBEDDING_MODEL_URL'))
    index = AnnoyIndex(
        int(os.environ.get('V_DIM')), os.environ.get('METRIC'))

    exceptions = []
    for idx in tqdm(metadata_filter_json.keys()):
        try:
            v = embed([metadata_filter_json[idx]
                       [os.environ.get('EMBED_COLUMN')]]).numpy()[0]
            index.add_item(idx, v)

        except Exception as e:
            exceptions.append(idx)

    index.build(int(os.environ.get('N_TREES')))
    index.save(os.environ.get('METADATA_FILENAME') + '.ann')
