FROM --platform=linux/amd64 python:3.10-slim as build

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy the necessary scripts and configuration files
COPY predict.py /app/
COPY configuration.py /app/
COPY model_predict.py /app/
COPY config_entity.py /app/
COPY common.py /app/
COPY Pipfile.lock /app/
COPY Pipfile /app/


RUN pip install pipenv && \
  apt-get update && \
  apt-get install -y --no-install-recommends gcc python3-dev libssl-dev git && \
  pipenv install --deploy --system && \
  apt-get remove -y gcc python3-dev libssl-dev && \
  apt-get autoremove -y && \
  pip uninstall pipenv -y

ENTRYPOINT ["python", "predict.py"]