# covid_answers

![](https://github.com/jspablo/covid_answers/blob/master/images/search.gif)

Covid19 answers is a web application aiming to provide a practical solution to query research papers about corona virus from the [Kaggle CORD19 dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks).

## Architecture

The architecture focus on creating a pipeline that allow rapid iteration as new papers are constantly being added. It is being developed using Google Cloud Platform and hosting public docker images on Docker Hub, so the app can be deploy anywhere.

The result of this pipeline is an app built around the following three modules:
* UI built with [Plotly Dash](https://plotly.com/dash/) framework to query documents.
* API that provides relevant documents given an input question or topic.
* AI server with natural language models for creating embeddings and question answering capabilities.

![](https://github.com/jspablo/covid_answers/blob/master/images/cord_architecture.png)

## Ideas and work in progress

* Refactor Annoy index creation, test a Kubeflow pipeline.
* Consider Annoy alternatives like Faiss or using a search engine like Elastic.
* Remove documents index from webapp image. Create a new API in order to query similar documents.
* Add QA functionality to UI


*Covid19 Icon made by Freepik from www.flaticon.com*
