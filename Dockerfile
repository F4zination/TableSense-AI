FROM python:3.11.9

# Set the working directory
WORKDIR /app

RUN pip install setuptools
RUN pip install poetry

# Copy the requirements file into the container
COPY pyproject.toml poetry.lock* /app/

# Install dependencies (no virtualenv, install to system)
RUN poetry config virtualenvs.create false 

RUN poetry install --no-root --no-interaction --no-ansi

# Copy the rest of the application code into the container
COPY ./tablesense_ai /app/tablesense_ai

# Expose the port the app runs on
EXPOSE 8501

#ENTRYPOINT ["python3", "-m", "test"]
ENTRYPOINT ["streamlit", "run", "./tablesense_ai/app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]