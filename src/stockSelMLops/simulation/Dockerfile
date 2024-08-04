FROM --platform=linux/amd64 python:3.10-slim as build

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy the necessary scripts and configuration files
COPY simulate.py /app/
COPY configuration.py /app/
COPY simulate_strategy.py /app/
COPY config_entity.py /app/
COPY common.py /app/
COPY Pipfile.lock /app/
COPY Pipfile /app/

# Install & use pipenv
RUN pip install -U pip
RUN pip install pipenv 

RUN apt-get update && apt-get install -y build-essential wget
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
  tar -xvzf ta-lib-0.4.0-src.tar.gz && \
  cd ta-lib/ && \
  ./configure --prefix=/usr --build=aarch64-unknown-linux-gnu && \
  make && \
  make install

# COPY . /app
RUN pipenv install --system --deploy 
RUN rm -R ta-lib ta-lib-0.4.0-src.tar.gz


ENTRYPOINT ["python", "simulate.py"]