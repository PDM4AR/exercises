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
# The base image ships a yarn apt source whose signing key periodically expires
# upstream (yarnpkg/yarn#9222), which breaks `apt-get update`. We don't use yarn,
# so drop that source to keep builds reproducible for everyone.
RUN rm -f /etc/apt/sources.list.d/yarn.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    $(awk -F: '/^[^#]/ { print $1 }' /tmp/dependencies-apt.txt | uniq) \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# Set working directory
ENV WORKDIR_PATH="/home/vscode"
WORKDIR ${WORKDIR_PATH}
ENV VIRTUAL_ENV=${WORKDIR_PATH}/.venv
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"
# Setting non-root user
USER vscode


## Install Suite sparse for arm64 to be used by cvxopt
#RUN mkdir SuiteSparse \
#   && curl -fsSL https://api.github.com/repos/DrTimothyAldenDavis/SuiteSparse/tarball/v5.8.1 \
#    | tar xzvpf - --no-same-owner --strip-components=1 -C SuiteSparse
#ENV CVXOPT_SUITESPARSE_SRC_DIR=${WORKDIR_PATH}/SuiteSparse

# Install the Antigravity CLI (`agy`). 
RUN curl -fsSL https://antigravity.google/cli/install.sh | bash

# Install the Claude Code CLI (`claude`) and the Codex CLI (`codex`) so the
# terminal tooling matches the `anthropic.claude-code` / `openai.chatgpt`
# VS Code extensions installed by devcontainer.json.
# RUN curl -fsSL https://claude.ai/install.sh | bash
# RUN curl -fsSL https://github.com/openai/codex/releases/latest/download/install.sh \
#     | CODEX_NON_INTERACTIVE=true sh

# # Install the GitHub Copilot CLI (`copilot`) from its release tarball rather than
# # the `@github/copilot` npm package, which would need a Node runtime this image
# # does not have. Asset names carry no version, so `latest/download` stays current.
# RUN case "$(uname -m)" in \
#     aarch64) COPILOT_ARCH=arm64 ;; \
#     x86_64) COPILOT_ARCH=x64 ;; \
#     *) echo "Unsupported architecture for Copilot CLI: $(uname -m)" >&2; exit 1 ;; \
#     esac \
#     && mkdir -p ${WORKDIR_PATH}/.local/bin \
#     && curl -fsSL "https://github.com/github/copilot-cli/releases/latest/download/copilot-linux-${COPILOT_ARCH}.tar.gz" \
#     | tar xz -C ${WORKDIR_PATH}/.local/bin \
#     && chmod +x ${WORKDIR_PATH}/.local/bin/copilot

# Install poetry
RUN pipx install poetry==${POETRY_VERSION}

# Install package pdm4ar + dependencies
COPY poetry.lock pyproject.toml ./
USER root
RUN chown vscode:vscode poetry.lock pyproject.toml
USER vscode
RUN poetry lock --no-update
RUN poetry install --no-root --no-cache -vv
RUN rm poetry.lock pyproject.toml

# ENV PATH="${HOME}/.local/bin:$PATH"
# https://github.com/RamiKrispin/vscode-python-medium/blob/main/03_final_example/.devcontainer/set_python_env.sh
LABEL maintainer="Y. Li (yuesli@ethz.ch)"