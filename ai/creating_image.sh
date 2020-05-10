SERVICE=ai
CONTAINER_NAME=answers_to_corona_nlp

docker run -d --name $SERVICE tensorflow/serving

docker cp swivel $SERVICE:/models/swivel
docker cp distilbert $SERVICE:/models/distilbert
docker cp serving/model_config.config $SERVICE:/models/model_config.config

docker commit $SERVICE $CONTAINER_NAME

# docker run -p 8501:8501 -t $CONTAINER_NAME --model_config_file=/models/model_config.config