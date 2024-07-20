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


# # Set the environment variable to point to the service account key file
# COPY your-service-account-key.json /app/your-service-account-key.json
# ENV GOOGLE_APPLICATION_CREDENTIALS="/app/your-service-account-key.json"


# EXPOSE 9696
# ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "app:app" ]
ENTRYPOINT ["python", "ingest.py"]