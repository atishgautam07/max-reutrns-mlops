FROM --platform=linux/amd64 python:3.9.7-slim as build

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app
COPY Pipfile.lock /app/
COPY Pipfile /app/

# Install & use pipenv
RUN pip install -U pip
RUN pip install pipenv 

# COPY . /app
RUN pipenv install --system --deploy 

# Expose port 5000 to the outside world
EXPOSE 5000

# Run the MLflow server
CMD mlflow server --backend-store-uri postgresql://postgres:mlflowsma@10.2.80.3:5432/mlflow_v1 --default-artifact-root gs://sma-proj-bucket/pipeline_root/mlflow/ --host 0.0.0.0 --port 5000
