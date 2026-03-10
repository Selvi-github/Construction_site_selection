# Use official slim python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create app directory
WORKDIR /app

# Install system dependencies (needed for compiling some python packages like reportlab or scikit-learn depending on architecture)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Ensure data and logs directories exist
RUN mkdir -p data logs instance

# Expose port (Render uses the PORT environment variable)
ENV PORT=5000
EXPOSE 5000

# Run gunicorn server with shell mode to expand $PORT
CMD gunicorn --bind 0.0.0.0:$PORT --workers 3 --timeout 120 app:app
