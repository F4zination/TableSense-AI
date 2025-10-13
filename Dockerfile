FROM python:3.13

# Set the working directory
WORKDIR /app

RUN touch README.md
RUN pip install setuptools

RUN pip install poetry

# Copy the requirements file into the container
COPY pyproject.toml poetry.lock* /app/

# Install dependencies (no virtualenv, install to system)
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --no-interaction --no-ansi

# Copy the rest of the application code into the container
COPY . /app/

# Expose the port the app runs on
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]