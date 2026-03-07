# syntax=docker/dockerfile:1

# Use the official UV Python base image with Python 3.13 on Debian Bookworm
# UV is a fast Python package manager that provides better performance than pip
# We use the slim variant to keep the image size smaller while still having essential tools
ARG PYTHON_VERSION=3.13
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS base

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/app" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

# Install build dependencies required for Python packages with native extensions
# gcc: C compiler needed for building Python packages with C extensions
# python3-dev: Python development headers needed for compilation
# We clean up the apt cache after installation to keep the image size down
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
  && rm -rf /var/lib/apt/lists/*

# Create a new directory for our application code
# And set it as the working directory
WORKDIR /app

# Copy just the dependency files first, for more efficient layer caching
COPY pyproject.toml ./
RUN mkdir -p src

# Install Python dependencies using UV's lock file (or pyproject if no lock)
RUN uv sync

# Copy all remaining application files into the container
COPY . .

# Change ownership of all app files to the non-privileged user
RUN chown -R appuser:appuser /app

# Switch to the non-privileged user for all subsequent operations
USER appuser

# Disable HuggingFace progress bars for cleaner build logs
ENV HF_HUB_DISABLE_PROGRESS_BARS=1

# Pre-download any ML models or files the agent needs
# The LiveKit CLI handles invoking plugins (like silero) to download their models
RUN uv run src/agent.py download-files || true

# Run the application using UV
CMD ["uv", "run", "python", "src/agent.py", "dev"]
