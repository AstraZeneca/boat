# Python 3.6 Image without Dependecies
FROM python:3.8-slim

LABEL maintainer="YOUREMAIL"

RUN pip install poetry

ENV VIRTUAL_ENV=/opt/venv
RUN python -m virtualenv --python=/usr/local/bin/python $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY . /app
WORKDIR /app

RUN poetry install

# Add your bits here.
