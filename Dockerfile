# Use a slim version of Python 3.10 as the base image
FROM python:3.10-slim

# Set environment variables to prevent Python from writing .pyc files 
# and to ensure output is logged directly
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user and switch to that user
RUN useradd -m appuser

# Copy requirements file and install Python dependencies
COPY requirements.txt . 
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Create necessary directories
RUN mkdir -p cache/data/graphgen cache/logs

# Environment variables for application config
ENV SYNTHESIZER_MODEL="" 
ENV SYNTHESIZER_BASE_URL="" 
ENV SYNTHESIZER_API_KEY="" 
ENV TRAINEE_MODEL="" 
ENV TRAINEE_BASE_URL="" 
ENV TRAINEE_API_KEY="" 

# Expose the port the app will run on
EXPOSE 7860

# Switch to the non-root user
USER appuser

# Command to run the application
CMD ["python", "webui/app.py"]
