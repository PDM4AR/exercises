# [Choice] Python version (use -bullseye variants on local arm64/Apple Silicon): 3, 3.10, 3.9, 3.8, 3.7, 3.6, 3-bullseye, 3.10-bullseye, 3.9-bullseye, 3.8-bullseye, 3.7-bullseye, 3.6-bullseye, 3-buster, 3.10-buster, 3.9-buster, 3.8-buster, 3.7-buster, 3.6-buster
ARG VARIANT="3.11-bullseye"
FROM mcr.microsoft.com/devcontainers/python:${VARIANT}

# Most of the choices for poetry and its environment have been made according to
# https://github.com/gianfa/poetry/blob/docs/docker-best-practices/docs/docker-best-practices.md

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    # pip
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_NO_CACHE_DIR=off \
    #poetry
    POETRY_VERSION=1.8.3 \
    POETRY_HOME=/opt/poetry \
    # Tell Poetry where to place its cache and virtual environment
    #POETRY_CACHE_DIR=/opt/.cache \
    # virtual environment in the project's root (it gets named `.venv`)
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_VIRTUALENVS_CREATE=1 \
    # do not ask any interactive question
    POETRY_NO_INTERACTION=1 \
    # for pycontracts
    DISABLE_CONTRACTS=1


# Install apt dependencies
COPY .devcontainer/dependencies-apt.txt /tmp/
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    $(awk -F: '/^[^#]/ { print $1 }' /tmp/dependencies-apt.txt | uniq) \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# Set working directory
WORKDIR /workspaces
ENV VIRTUAL_ENV=/workspaces/.venv
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

# Install poetry
RUN pipx install poetry==${POETRY_VERSION}

# Install package pdm4ar + dependencies
COPY poetry.lock pyproject.toml ./
RUN poetry lock --no-update
RUN poetry install --no-root --no-cache -vv
RUN rm poetry.lock pyproject.toml

# ENV PATH="${HOME}/.local/bin:$PATH"
# https://github.com/RamiKrispin/vscode-python-medium/blob/main/03_final_example/.devcontainer/set_python_env.sh
LABEL maintainer="A. Zanardi (azanardi@ethz.ch)"