FROM harbor.csis.astrazeneca.net/azimuth-iron/cai-bp/cpu:latest

USER jovyan
WORKDIR /home/jovyan

# Create an environment
COPY pyproject.toml poetry.lock .
RUN poetry config virtualenvs.in-project true \
    && poetry install --no-root \
    && rm -r ~/.cache

# Setup the environment
ENV PATH /home/jovyan/.venv/bin:$PATH
