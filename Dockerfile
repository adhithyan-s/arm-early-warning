FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependency required by LightGBM on Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (separate layer = faster rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and package setup
COPY setup.py .
COPY src/ ./src/

# Copy the trained model artifacts
COPY models/lgbm_model.pkl ./models/
COPY models/model_metadata.json ./models/

# Install the package in editable mode
RUN pip install -e .

# Expose the API port
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]