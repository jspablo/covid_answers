import argparse

import kfp
import kfp.components as comp
import kfp.dsl as dsl


def filter_dataset(
    bucket: str,
    dataset: str,
    text_col: str,
    index_col: str,
    filter_columns: str
) -> str:
    import io
    import logging
    import sys
    import subprocess
    from datetime import datetime

    logging.getLogger().setLevel(logging.INFO)

    subprocess.run(
        [sys.executable,
         '-m', 'pip', 'install', '--upgrade', 'pip']
    )
    subprocess.run(
        [sys.executable,
         '-m', 'pip', 'install', 'pandas', 'google-cloud-storage']
    )

    import pandas as pd
    from google.cloud import storage

    storage_client = storage.Client()

    raw_buffer = io.BytesIO()
    dataset = dataset.split("gs://{}/".format(bucket))[-1]
    bucket = storage_client.get_bucket(bucket)
    bucket.get_blob(dataset).download_to_file(raw_buffer)
    raw_buffer.seek(0)

    # TODO: improve list of strings as input argument
    filter_columns = filter_columns.split('-')
    df = pd.read_csv(raw_buffer, encoding="utf-8")
    logging.info(df.shape)
    df = df[filter_columns][df[text_col].notna()]
    df[index_col] = df.index

    version = datetime.now().strftime('%m%d%Y_%H%M%S')
    filter_dataset_name = "filter_data/filter_jsonl_{}.csv".format(version)
    bucket.blob(filter_dataset_name).upload_from_string(
        df.to_csv(index=True), content_type='text/csv'
    )

    logging.info(df.shape)

    return filter_dataset_name


def create_embeddings(
    bucket: str,
    dataset: str,
    model: str,
    text_col: str,
    index_col: str
) -> str:
    import io
    import logging
    import sys
    import subprocess
    from datetime import datetime

    logging.getLogger().setLevel(logging.INFO)

    subprocess.run(
        [sys.executable,
         '-m', 'pip', 'install', '--upgrade', 'pip']
    )
    subprocess.run(
        [sys.executable,
         '-m', 'pip', 'install',
         'pandas', 'tensorflow-hub==0.7.0', 'google-cloud-storage']
    )

    import pandas as pd
    import tensorflow_hub as hub
    from google.cloud import storage

    storage_client = storage.Client()

    raw_buffer = io.BytesIO()
    bucket = storage_client.get_bucket(bucket)
    bucket.get_blob(dataset).download_to_file(raw_buffer)
    raw_buffer.seek(0)

    df = pd.read_csv(raw_buffer, encoding="utf-8")

    logging.info(df.shape)

    embed = hub.load(model)

    embed_col = "embedding"
    df[embed_col] = df[text_col].apply(
        lambda title: embed([title]).numpy()[0]
    )

    version = datetime.now().strftime('%m%d%Y_%H%M%S')
    embeddings_path = "embeddings/embeddings_{}.json".format(version)
    bucket.blob(embeddings_path).upload_from_string(
        df[[index_col, embed_col]].to_json(orient='values')
    )

    return embeddings_path


def create_index(
    bucket: str,
    embeddings_path: str,
    annoy_dim: int,
    annoy_metric: str,
    annoy_trees: int
) -> str:
    import io
    import json
    import logging
    from datetime import datetime

    logging.getLogger().setLevel(logging.INFO)

    from annoy import AnnoyIndex
    from google.cloud import storage

    storage_client = storage.Client()

    logging.info(embeddings_path)

    raw_buffer = io.BytesIO()
    bucket = storage_client.get_bucket(bucket)
    bucket.get_blob(embeddings_path).download_to_file(raw_buffer)
    raw_buffer.seek(0)

    embeddings = json.load(raw_buffer)

    assert isinstance(embeddings, list)

    annoy_index = AnnoyIndex(annoy_dim, annoy_metric)

    for idx, embedding in embeddings:
        annoy_index.add_item(idx, embedding)

    annoy_index.build(annoy_trees)

    version = datetime.now().strftime('%m%d%Y_%H%M%S')
    annoy_filename = "annoy_{}.ann".format(version)
    annoy_path = "annoy/{}".format(annoy_filename)
    annoy_index.save(annoy_filename)
    bucket.blob(annoy_path).upload_from_filename(annoy_filename)

    return annoy_filename


filter_op = comp.func_to_container_op(filter_dataset)

embeddings_op = comp.func_to_container_op(
    create_embeddings,
    base_image="tensorflow/tensorflow:2.0.1-py3"
)

index_op = comp.func_to_container_op(
    create_index,
    packages_to_install=["annoy", "google-cloud-storage"]
)


@dsl.pipeline(
    name='cord19 pipeline',
    description='Filtering, embeddings and Annoy index from cord19 dataset'
)
def file_passing_pipelines(
    bucket,
    dataset,
    text_col,
    index_col,
    filter_columns,
    project_id,
    region,
    tf_hub_model,
    annoy_dim,
    annoy_metric,
    annoy_trees
):
    filter_op_out = filter_op(
        bucket, dataset, text_col, index_col, filter_columns
    )
    embeddings_op_out = embeddings_op(
        bucket, filter_op_out.output, tf_hub_model, text_col, index_col
    )
    index_op(
        bucket, embeddings_op_out.output, annoy_dim, annoy_metric, annoy_trees
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", type=str)
    parser.add_argument("--text_col", type=str, default="title")
    parser.add_argument("--index_col", type=str, default="index")
    parser.add_argument("--region", type=str, default="europe-west1")
    parser.add_argument("--kf_host", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--filter_columns", type=str)
    parser.add_argument("--project_id", type=str)
    parser.add_argument("--tf_hub_model", type=str)
    parser.add_argument("--annoy_dim", type=int, default=128)
    parser.add_argument("--annoy_metric", type=str, default="euclidean")
    parser.add_argument("--annoy_trees", type=int, default=100)
    args = parser.parse_args()

    if args.kf_host:
        client = kfp.Client(host=args.kf_host)

        arguments = {
            "bucket": args.bucket,
            "dataset": args.dataset,
            "text_col": args.text_col,
            "index_col": args.index_col,
            "filter_columns": args.filter_columns,
            "project_id": args.project_id,
            "region": args.region,
            "tf_hub_model": args.tf_hub_model,
            "annoy_dim": args.annoy_dim,
            "annoy_metric": args.annoy_metric,
            "annoy_trees": args.annoy_trees
        }

        client.create_run_from_pipeline_func(
            file_passing_pipelines, arguments=arguments
        )

    else:
        kfp.compiler.Compiler().compile(
            file_passing_pipelines, __file__ + '.yaml'
        )
