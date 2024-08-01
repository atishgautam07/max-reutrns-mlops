FROM --platform=linux/amd64 python:3.9.7-slim as build

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy the necessary scripts and configuration files
COPY ingest.py /app/
COPY configuration.py /app/
COPY data_ingestion.py /app/
COPY config_entity.py /app/
COPY common.py /app/
COPY Pipfile.lock /app/
COPY Pipfile /app/

# Install & use pipenv
RUN pip install -U pip
RUN pip install pipenv 

# COPY . /app
RUN pipenv install --system --deploy 

ENTRYPOINT ["python", "ingest.py"]