# Use official Python 3.11 base image
FROM python:3.11-bookworm-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Create working directory
WORKDIR /app

# Install system dependencies (if needed for pandas/parquet/etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && apt-get clean

# Install inspect-ai with CLI support
RUN pip install "inspect-ai" fastapi uvicorn

# Copy app code
COPY . /app

# Expose port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
