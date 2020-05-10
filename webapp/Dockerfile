FROM python:3.6

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN pip install -r app/requirements.txt

CMD exec gunicorn --bind :8000 --workers 1 --threads 8 app.app:server
